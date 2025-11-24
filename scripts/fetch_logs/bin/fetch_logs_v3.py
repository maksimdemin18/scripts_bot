#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import typing as t
from pathlib import Path
import zipfile

# --- optional yaml ---
try:
    import yaml  # type: ignore
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# ---------- utils ----------

def eprint(*args: t.Any, **kwargs: t.Any) -> None:
    print(*args, file=sys.stderr, **kwargs)

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)

def ssh_build_opts(options: t.Union[str, t.List[str], None]) -> t.List[str]:
    if options is None:
        return []
    if isinstance(options, list):
        return options
    return shlex.split(options)

def load_config(path: Path) -> dict:
    if not path.is_file():
        eprint(f"[ERR] Не найден файл конфигурации: {path}")
        sys.exit(2)
    if HAVE_YAML and path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)
    except Exception:
        eprint("[ERR] Установите PyYAML (pip install pyyaml) или используйте JSON вместо YAML.")
        sys.exit(2)

def parse_datetime(s: str) -> dt.datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    eprint(f"[ERR] Не удалось разобрать дату/время: {s}. Ожидается 'YYYY-MM-DD HH:MM:SS' или 'YYYY-MM-DDTHH:MM:SS'.")
    sys.exit(2)

def parse_duration(spec: str) -> dt.timedelta:
    s = (spec or "").strip()
    if not s:
        raise ValueError("Пустая длительность")
    unit = s[-1].lower()
    num = s[:-1]
    if not num.isdigit() or unit not in "smhd":
        raise ValueError("Неверный формат длительности. Используйте 10s, 5m, 2h, 1d")
    value = int(num)
    if unit == "s":
        return dt.timedelta(seconds=value)
    if unit == "m":
        return dt.timedelta(minutes=value)
    if unit == "h":
        return dt.timedelta(hours=value)
    if unit == "d":
        return dt.timedelta(days=value)
    raise ValueError("Неизвестная единица длительности")

def resolve_interval(args: argparse.Namespace) -> t.Tuple[dt.datetime, dt.datetime]:
    # support synonyms: --end/--until, --out/--outdir
    end_arg = args.end or args.until
    has_start = bool(args.start)
    has_end = bool(end_arg)
    has_since = bool(args.since)
    has_dur = bool(args.duration)

    if has_since and (has_start or has_end or has_dur):
        raise ValueError("--since несовместим с --start/--end/--until/--duration")

    if has_since:
        delta = parse_duration(args.since)
        now = dt.datetime.now()
        return now - delta, now

    if has_start and has_dur and not has_end:
        start_dt = parse_datetime(args.start)
        delta = parse_duration(args.duration)
        return start_dt, start_dt + delta

    if has_start and has_end:
        start_dt = parse_datetime(args.start)
        end_dt = parse_datetime(end_arg)
        if end_dt < start_dt:
            raise ValueError("Конец периода раньше начала")
        return start_dt, end_dt

    raise ValueError("Укажите либо: --since, либо: --start и --end/--until, либо: --start и --duration")

# ---------- hostname & ssh ----------

def get_remote_hostname(ip: str, user: str, port: int, opts: t.List[str]) -> str:
    remote_cmd = "bash -lc 'hostname -f || hostname'"
    cmd = ["ssh", "-p", str(port), *opts, f"{user}@{ip}", remote_cmd]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=15)
        name = out.strip().splitlines()[0].strip()
        if name:
            return sanitize_filename(name)
    except Exception:
        pass
    try:
        name = socket.gethostbyaddr(ip)[0]
        return sanitize_filename(name)
    except Exception:
        return sanitize_filename(ip)

def build_remote_command(svc: dict, use_sudo: bool) -> str:
    # Always force bash -lc to keep loops/quotes intact; zsh can choke otherwise.
    sudo = "sudo -n " if use_sudo else ""
    if "glob" in svc and svc["glob"]:
        glob = svc["glob"]
        inner = f"for f in {glob}; do [ -f \"$f\" ] || continue; case \"$f\" in (*.gz) zcat -f \"$f\" ;; (*) cat \"$f\" ;; esac; done"
        return f"bash -lc '{sudo}{inner}'"
    if "path" in svc and svc["path"]:
        path = str(svc["path"]).replace("'", "'\"'\"'")
        return f"bash -lc '{sudo}cat \\'{path}\\''"
    raise ValueError("В конфигурации службы должен быть указан ключ path или glob")

def iter_remote_lines(ip: str, user: str, port: int, opts: t.List[str], remote_command: str, verbose: int = 0):
    ssh_cmd = ["ssh", "-p", str(port), *opts, f"{user}@{ip}", remote_command]
    if verbose:
        eprint("[DBG] SSH cmd:", " ".join(ssh_cmd))
    proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else "")
        raise RuntimeError(f"Удалённая команда завершилась с кодом {proc.returncode}. STDERR: {err.strip()}")

# ---------- time parsing per-line ----------

def compile_time_parser(time_regex: str, time_format: str, start_year: int) -> t.Tuple[re.Pattern[str], str, bool]:
    rx = re.compile(time_regex)
    year_missing = "%Y" not in time_format
    effective_format = ("%Y " + time_format) if year_missing else time_format
    return rx, effective_format, year_missing

def parse_line_dt(line: str, rx: re.Pattern[str], fmt: str, year_missing: bool, start_year: int) -> t.Optional[dt.datetime]:
    m = rx.search(line)
    if not m:
        return None
    ts = m.group("ts")

    if "T" in ts and "T" not in fmt:
        ts = ts.replace("T", " ")

    if year_missing:
        ts = f"{start_year} {ts}"

    has_tz = "%z" in fmt
    tz_match = re.search(r"(Z|[+-]\d{2}:?\d{2})$", ts)
    if tz_match:
        tz = tz_match.group(1)
        if has_tz:
            if tz == "Z":
                ts = ts[:-len(tz)] + "+0000"
            elif ":" in tz:
                ts = ts[:-len(tz)] + tz.replace(":", "")
        else:
            ts = ts[:-len(tz)]

    if "%f" not in fmt and "." in ts:
        ts = re.sub(r"(\d{2}:\d{2}:\d{2})\.\d{1,6}", r"\1", ts)

    try:
        dt_obj = dt.datetime.strptime(ts, fmt)
        if dt_obj.tzinfo is not None:
            dt_obj = dt_obj.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return dt_obj
    except Exception:
        return None

# ---------- retention cleanup ----------

def cleanup_old_files(out_dir: Path, retention_days: int, verbose: int = 0) -> None:
    if retention_days <= 0:
        return
    now = dt.datetime.now().timestamp()
    pattern = re.compile(r".+__.+__\d{8}T\d{6}-\d{8}T\d{6}\.(log|zip)$")
    if not out_dir.exists():
        return
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        if not pattern.match(p.name):
            continue
        try:
            age_days = (now - p.stat().st_mtime) / 86400.0
            if age_days > retention_days:
                p.unlink(missing_ok=True)
                if verbose:
                    eprint(f"[INFO] Удалён старый файл: {p.name} (age {age_days:.1f} d)")
        except Exception as ex:
            if verbose:
                eprint(f"[WARN] Не удалось удалить {p.name}: {ex}")

# ---------- args ----------

def parse_args() -> argparse.Namespace:
    # Resolve repo layout: scripts/fetch_logs/...
    here = Path(__file__).resolve()
    root = here.parents[2]  # .../scripts
    app = here.parents[1]   # .../scripts/fetch_logs
    default_conf = (app / "config" / "config.yaml")
    default_out  = (app / "out")

    p = argparse.ArgumentParser(description="Fetch logs over SSH for a time interval")
    p.add_argument("--ip", required=True, help="IP или DNS хоста")
    p.add_argument("--service", required=True, help="Имя службы из конфигурации")

    p.add_argument("--start", required=False, help="Начало периода: 'YYYY-MM-DD HH:MM:SS' или 'YYYY-MM-DDTHH:MM:SS'")
    p.add_argument("--end",   required=False, help="Конец периода (включительно): как --start")
    p.add_argument("--until", required=False, help="Синоним --end")
    p.add_argument("--since", required=False, help="Относительно: 'Ns','Nm','Nh','Nd' (например, 5m)")
    p.add_argument("--duration", required=False, help="Длительность от --start: 'Ns','Nm','Nh','Nd'")

    p.add_argument("--config", default=str(default_conf), help="Путь к конфигу (YAML/JSON)")
    p.add_argument("--out",    default=str(default_out),  help="Каталог для результатов (бот подставит {out_dir}/...)")
    p.add_argument("--outdir", required=False, help="Синоним --out")

    # SSH
    p.add_argument("--ssh-user", default=None, help="Переопределить ssh.user")
    p.add_argument("--ssh-port", type=int, default=None, help="Переопределить ssh.port")
    p.add_argument("--ssh-key", default=None, help="Путь к приватному ключу (IdentityFile)")

    # Поведение
    p.add_argument("--dry-run", action="store_true", help="Не писать файл, только печатать совпавшие строки")
    p.add_argument("--no-stdout", action="store_true", help="Не печатать совпавшие строки в терминал")
    p.add_argument("--verbose", "-v", action="count", default=0, help="Подробный вывод (можно несколько раз)")
    p.add_argument("--retention-days", type=int, default=7, help="Хранить файлы в out не дольше N дней (0=не чистить)")

    # Артефакт: zip или plain
    p.add_argument("--format", choices=["plain", "zip"], default="plain", help="Формат результата: log-файл или zip-архив")

    return p.parse_args()

# ---------- main ----------

def main() -> int:
    args = parse_args()
    out_dir = Path(args.outdir or args.out).expanduser()
    cfg_path = Path(args.config).expanduser()

    cfg = load_config(cfg_path)

    ssh_cfg = cfg.get("ssh", {}) or {}
    user = args.ssh_user or ssh_cfg.get("user") or os.getenv("USER") or "root"
    port = int(args.ssh_port or ssh_cfg.get("port") or 22)
    opts = ssh_build_opts(ssh_cfg.get("options"))
    identity_file = args.ssh_key or ssh_cfg.get("identity_file")
    if identity_file:
        identity_file = os.path.expanduser(str(identity_file))
        opts = ["-i", identity_file, *opts]
    use_sudo = bool(ssh_cfg.get("use_sudo", False))

    services = cfg.get("services", {}) or {}
    if args.service not in services:
        eprint(f"[ERR] В конфигурации нет службы '{args.service}'. Доступно: {', '.join(services.keys()) or '—'}")
        return 2
    svc = services[args.service] or {}

    try:
        start_dt, end_dt = resolve_interval(args)
    except ValueError as ex:
        eprint(f"[ERR] {ex}")
        return 2

    time_regex = svc.get("time_regex")
    time_format = svc.get("time_format")
    if not time_regex or not time_format:
        eprint("[ERR] В службе должны быть заданы 'time_regex' и 'time_format'.")
        return 2

    rx, effective_fmt, year_missing = compile_time_parser(time_regex, time_format, start_dt.year)

    hostname = get_remote_hostname(args.ip, user, port, opts)
    remote_command = build_remote_command(svc, use_sudo)

    out_dir.mkdir(parents=True, exist_ok=True)
    start_tag = start_dt.strftime("%Y%m%dT%H%M%S")
    end_tag = end_dt.strftime("%Y%m%dT%H%M%S")
    base_name = f"{hostname}__{sanitize_filename(args.service)}__{start_tag}-{end_tag}"
    log_path = out_dir / f"{base_name}.log"

    # retention cleanup
    cleanup_old_files(out_dir, args.retention_days, verbose=args.verbose)

    matched = 0
    total = 0
    parsed_ok = 0
    unparsable = 0
    out_of_range = 0

    out_fh = None
    try:
        if not args.dry_run:
            out_fh = log_path.open("w", encoding="utf-8")
        for line in iter_remote_lines(args.ip, user, port, opts, remote_command, verbose=args.verbose):
            total += 1
            ts = parse_line_dt(line, rx, effective_fmt, year_missing, start_dt.year)
            if ts is None:
                unparsable += 1
                continue
            parsed_ok += 1
            if start_dt <= ts <= end_dt:
                matched += 1
                if not args.no_stdout:
                    print(line)
                if out_fh:
                    out_fh.write(line + "\n")
            else:
                out_of_range += 1
        if out_fh:
            out_fh.flush()
    except Exception as ex:
        if out_fh:
            out_fh.close()
        try:
            if out_fh and log_path.is_file() and log_path.stat().st_size == 0:
                log_path.unlink(missing_ok=True)
        except Exception:
            pass
        eprint(f"[ERR] {ex}")
        return 1
    finally:
        if out_fh and not out_fh.closed:
            out_fh.close()

    if args.format == "zip" and not args.dry_run:
        zip_path = out_dir / f"{base_name}.zip"
        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                if log_path.exists():
                    zf.write(log_path, arcname=log_path.name)
            print(f"[OK] Запаковано: {zip_path}")
        except Exception as ex:
            eprint(f"[WARN] Не удалось собрать zip: {ex}")

    if args.verbose:
        eprint(f"[INFO] total={total}, parsed={parsed_ok}, unmatched_by_time={out_of_range}, unparsable={unparsable}")

    if args.dry_run:
        print(f"[OK] Найдено строк в интервале: {matched} (из {total}). Файл НЕ создан: {log_path.name}")
    else:
        final_target = (out_dir / f"{base_name}.zip") if args.format == "zip" else log_path
        print(f"[OK] Сохранено строк: {matched} (из {total} просмотренных) → {final_target}")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

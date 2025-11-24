#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Многоцелевой ping/port-scan для адреса/списка/сети.
- Без внешних зависимостей (только стандартная библиотека).
- Устойчивый ICMP ping (Linux/macOS/Windows) с авто TCP-фолбэком.
- Port-scan по указанным портам/диапазонам (или all), параллельно.

Примеры — см. в конце --help.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import ipaddress
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

# -------------------- утилиты платформы --------------------

def is_windows() -> bool:
    return sys.platform.startswith("win")

# -------------------- разбор вывода ICMP ping --------------------

_re_win_packets_en = re.compile(r"Packets:\s*Sent\s*=\s*(\d+),\s*Received\s*=\s*(\d+)", re.I)
_re_win_packets_ru = re.compile(r"Пакетов:\s*отправлено\s*=\s*(\d+),\s*получено\s*=\s*(\d+)", re.I)
_re_nix_packets    = re.compile(r"(\d+)\s+packets\s+transmitted.*?(\d+)\s+(?:packets\s+)?received", re.I | re.S)

_re_nix_rtt = re.compile(r"=\s*[^/]+/([0-9.]+)/", re.I)  # min/avg/max/...
_re_win_avg_en = re.compile(r"Average\s*=\s*(\d+)\s*ms", re.I)
_re_win_avg_ru = re.compile(r"Средн\w*\s*=\s*(\d+)\s*мс", re.I)

_success_line_markers = (
    "bytes from", "icmp_seq=", "time=",       # *nix
    "Reply from", "Ответ от",                 # Windows EN/RU
    "Respuesta desde", "Antwort von",
)

def parse_counts_and_avg(stdout: str, requested_count: int) -> Tuple[int, Optional[float]]:
    """Вернёт (recv, avg_ms). При нераспознанном формате считает recv по эвристике."""
    recv = None
    avg_ms: Optional[float] = None

    m = _re_win_packets_en.search(stdout) or _re_win_packets_ru.search(stdout) or _re_nix_packets.search(stdout)
    if m:
        try:
            sent = int(m.group(1))
            got  = int(m.group(2))
            recv = got
        except Exception:
            recv = None

    if recv is None:
        got = 0
        for line in stdout.splitlines():
            s = line.strip()
            if any(marker.lower() in s.lower() for marker in _success_line_markers):
                got += 1
        recv = got

    mavg = _re_nix_rtt.search(stdout) or _re_win_avg_en.search(stdout) or _re_win_avg_ru.search(stdout)
    if mavg:
        try:
            avg_ms = float(mavg.group(1))
        except Exception:
            avg_ms = None

    recv = max(0, min(requested_count, int(recv)))
    return recv, avg_ms

# -------------------- ICMP ping с фолбэками --------------------

def _looks_like_usage_or_perm(out: str) -> bool:
    s = (out or "").lower()
    return (
        "usage:" in s
        or ("busybox" in s and "ping" in s)
        or "invalid option" in s
        or "illegal option" in s
        or "operation not permitted" in s
        or "permission denied" in s
        or "icmp open socket" in s
        or ("socket:" in s and "not permitted" in s)
    )

def _build_ping_candidates(host: str, count: int, timeout_ms: int, force_bin: Optional[str]) -> List[List[str]]:
    """
    Генерируем варианты команд ping для максимальной совместимости.
    Linux/Unix/macOS:
      1) <ping> -n -c COUNT host
      2) <ping> -c COUNT host
      3) <ping> -n -c COUNT -w DEADLINE host  (общий дедлайн, сек)
    Windows:
      ping -n COUNT -w TIMEOUT_MS host
    """
    if is_windows():
        return [["ping", "-n", str(count), "-w", str(timeout_ms), host]]

    bins = []
    if force_bin:
        bins.append(force_bin)
    else:
        bins = [shutil.which("ping"), "/bin/ping", "/usr/bin/ping"]
    seen = set()
    bins = [b for b in bins if b and (b not in seen and not seen.add(b))]

    candidates: List[List[str]] = []
    for b in bins:
        candidates.append([b, "-n", "-c", str(count), host])
        candidates.append([b, "-c", str(count), host])
        deadline = max(1, int(round(timeout_ms / 1000)) * max(1, count))
        candidates.append([b, "-n", "-c", str(count), "-w", str(deadline), host])
    return candidates

def ping_icmp(host: str, count: int, timeout_ms: int, force_bin: Optional[str]) -> Tuple[Dict[str, object], bool]:
    """
    Пытаемся ICMP. Возвращаем (result_dict, usable_icmp).
    usable_icmp=False — окружение не позволило корректный ICMP (usage/perm).
    """
    last_out = ""
    for cmd in _build_ping_candidates(host, count, timeout_ms, force_bin):
        try:
            if os.environ.get("PING_DEBUG") == "1":
                print("CMD:", " ".join(cmd), flush=True)
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out = p.stdout or ""
            last_out = out
        except Exception as e:
            last_out = f"exec error: {e}"
            continue

        if _looks_like_usage_or_perm(last_out):
            continue

        recv, avg_ms = parse_counts_and_avg(last_out, count)
        loss = 100.0 * (count - recv) / max(1, count)
        status = "UP" if recv > 0 else "DOWN"
        return {"host": host, "sent": count, "recv": recv, "loss_pct": loss, "avg_ms": avg_ms, "status": status}, True

    # ICMP непригоден
    return {"host": host, "sent": count, "recv": 0, "loss_pct": 100.0, "avg_ms": None, "status": "ERROR: ping unusable"}, False

# -------------------- TCP-пинг и порт-скан --------------------

def parse_ports(spec: str) -> List[int]:
    spec = (spec or "").strip().lower()
    if not spec:
        return []
    if spec in ("all", "1-65535"):
        return list(range(1, 65536))
    ports: set[int] = set()
    for part in re.split(r"[,\s]+", spec):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start, end = int(a), int(b)
                if start > end:
                    start, end = end, start
                for p in range(max(1, start), min(65535, end) + 1):
                    ports.add(p)
            except ValueError:
                continue
        else:
            try:
                p = int(part)
                if 1 <= p <= 65535:
                    ports.add(p)
            except ValueError:
                continue
    return sorted(ports)

def tcp_ping_alive(host: str, ports: List[int], timeout_ms: int) -> Tuple[bool, Optional[float]]:
    """
    TCP-пинг для определения «жив/не жив» (используется как фолбэк).
    UP, если:
      - connect() успешен, ИЛИ
      - получен быстрый RST (ECONNREFUSED) — признак, что узел отвечает.
    Возвращает (alive, best_ms).
    """
    if not ports:
        ports = [53, 161, 23, 8291, 80, 443, 22]  # «сетевой» набор по умолчанию
    timeout = max(0.05, timeout_ms / 1000.0)
    best_ms: Optional[float] = None
    for port in ports:
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                dt_ms = (time.perf_counter() - t0) * 1000.0
                best_ms = dt_ms if best_ms is None or dt_ms < best_ms else best_ms
                return True, best_ms
        except OSError as e:
            if getattr(e, "errno", None) in (111,):  # ECONNREFUSED — быстрый RST
                dt_ms = (time.perf_counter() - t0) * 1000.0
                best_ms = dt_ms if best_ms is None or dt_ms < best_ms else best_ms
                return True, best_ms
        except Exception:
            pass
    return False, best_ms

def _scan_port(host: str, port: int, timeout: float) -> Tuple[int, bool]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return port, True
    except Exception:
        return port, False

def ports_scan_for_hosts(hosts: List[str], ports: List[int], timeout_ms: int, jobs: int) -> Dict[str, List[int]]:
    """
    Сканирует пары (host,port) параллельно с общим пулом потоков.
    Возврат: {host: [open_ports...]}
    """
    timeout = max(0.05, timeout_ms / 1000.0)
    open_map: Dict[str, List[int]] = {h: [] for h in hosts}
    tasks: List[Tuple[str, int]] = []
    for h in hosts:
        for p in ports:
            tasks.append((h, p))
    if not tasks:
        return open_map
    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(_scan_port, h, p, timeout): (h, p) for h, p in tasks}
        for fut in cf.as_completed(futs):
            h, p = futs[fut]
            try:
                port, is_open = fut.result()
                if is_open:
                    open_map[h].append(port)
            except Exception:
                pass
    # отсортируем открытые порты по возрастанию
    for h in open_map:
        open_map[h].sort()
    return open_map

# -------------------- ввод целей --------------------

def read_hosts_file(path: str) -> List[str]:
    hosts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            hosts.append(s)
    # уникализация с сохранением порядка
    seen = set()
    uniq: List[str] = []
    for h in hosts:
        if h not in seen:
            uniq.append(h); seen.add(h)
    return uniq

def resolve_targets(target: str) -> List[str]:
    """
    Принимаем:
      - путь к файлу,
      - '-' (stdin),
      - строку адресов через пробел/запятую,
      - сеть в нотации CIDR (x.x.x.x/24).
    """
    if target == "-":
        lines = [ln.strip() for ln in sys.stdin.read().splitlines()]
        return [s for s in lines if s and not s.startswith("#")]

    # сеть?
    try:
        net = ipaddress.ip_network(target, strict=False)
        # перечислим только хосты (без network/broadcast для v4)
        return [str(ip) for ip in net.hosts()]
    except Exception:
        pass

    # файл?
    if os.path.exists(target) and os.path.isfile(target):
        return read_hosts_file(target)

    # иначе — список адресов
    parts = re.split(r"[,\s]+", target.strip())
    return [p for p in parts if p]

# -------------------- форматирование вывода --------------------

def fmt_table(rows: List[Dict[str, object]], show_ports: bool, ports_show_limit: int) -> str:
    host_w = max(4, min(48, max((len(str(r["host"])) for r in rows), default=4)))
    base_hdr = f"{'HOST'.ljust(host_w)}  {'SENT':>4}  {'RECV':>4}  {'LOSS%':>6}  {'AVG_MS':>6}  STATUS"
    if show_ports:
        base_hdr += f"  {'OPEN':>4}  PORTS"
    sep = "-" * len(base_hdr)
    lines = [base_hdr, sep]
    for r in rows:
        host = str(r["host"]).ljust(host_w)
        sent = f"{int(r.get('sent', 0)):>4d}"
        recv = f"{int(r.get('recv', 0)):>4d}"
        loss = f"{float(r.get('loss_pct', 0.0)):>6.1f}"
        avg  = f"{float(r['avg_ms']):>6.1f}" if r.get("avg_ms") is not None else f"{'—':>6}"
        status = str(r.get("status", ""))
        line = f"{host}  {sent}  {recv}  {loss}  {avg}  {status}"
        if show_ports:
            ports_list = r.get("open_ports") or []
            open_cnt = len(ports_list)
            if open_cnt and ports_show_limit > 0 and open_cnt > ports_show_limit:
                show = ports_list[:ports_show_limit]
                suffix = ",..."
            else:
                show = ports_list
                suffix = ""
            ports_str = ",".join(str(p) for p in show) + suffix
            line += f"  {open_cnt:>4d}  {ports_str}"
        lines.append(line)
    return "\n".join(lines)

def write_csv(rows: List[Dict[str, object]], fp, include_ports: bool) -> None:
    base = ["host", "sent", "recv", "loss_pct", "avg_ms", "status"]
    fieldnames = base + (["open", "open_ports"] if include_ports else [])
    w = csv.DictWriter(fp, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        row = {k: r.get(k) for k in base}
        if include_ports:
            ops = r.get("open_ports") or []
            row["open"] = len(ops)
            row["open_ports"] = ",".join(map(str, ops))
        w.writerow(row)

def dumps_json(rows: List[Dict[str, object]], pretty: bool) -> str:
    if pretty:
        return json.dumps(rows, ensure_ascii=False, indent=2)
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))

# -------------------- сортировка/фильтрация --------------------

def sort_rows(rows: List[Dict[str, object]], sort_key: Optional[str], desc: bool) -> List[Dict[str, object]]:
    if not sort_key:
        return rows
    if sort_key == "loss":
        key = lambda r: (r.get("loss_pct", 0.0), r.get("host", ""))
    elif sort_key == "rtt":
        key = lambda r: ((float("inf") if r.get("avg_ms") is None else r.get("avg_ms")), r.get("host", ""))
    elif sort_key == "host":
        key = lambda r: r.get("host", "")
    elif sort_key == "open":
        key = lambda r: (len(r.get("open_ports") or []), r.get("host", ""))
    else:
        return rows
    return sorted(rows, key=key, reverse=desc)

# -------------------- диагностика окружения --------------------

def run_diag() -> None:
    print("=== DIAG ===", flush=True)
    try:
        print("python:", sys.version.split()[0], "platform:", platform.platform(), flush=True)
    except Exception:
        pass
    try:
        uid = getattr(os, "getuid", lambda: "n/a")()
        euid = getattr(os, "geteuid", lambda: "n/a")()
        print("uid:", uid, "euid:", euid, flush=True)
    except Exception:
        pass
    wp = shutil.which("ping")
    print("which_ping:", wp, "PATH:", os.environ.get("PATH", ""), flush=True)
    if wp:
        if shutil.which("getcap"):
            try:
                out = subprocess.run(["getcap", wp], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                print("getcap:", (out.stdout or "").strip(), flush=True)
            except Exception:
                pass
        try:
            out = subprocess.run(["ls", "-l", wp], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print("ls -l:", (out.stdout or "").strip(), flush=True)
        except Exception:
            pass
    if shutil.which("ip"):
        for cmd in (["ip", "-br", "addr"], ["ip", "route"], ["ip", "rule"]):
            try:
                out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                print("$", " ".join(cmd), "\n", (out.stdout or out.stderr or "").strip(), sep="", flush=True)
            except Exception:
                pass
    # быстрая TCP-проверка до 8.8.8.8:53
    def try_tcp(host, port, to=0.5):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=to):
                dt = (time.perf_counter() - t0) * 1000
                print(f"TCP OK {host}:{port} {dt:.1f}ms", flush=True)
        except Exception as e:
            print(f"TCP FAIL {host}:{port} err={e}", flush=True)
    try_tcp("8.8.8.8", 53, 0.5)
    print("=== /DIAG ===", flush=True)

# -------------------- основной поток --------------------

def probe_host(host: str, count: int, timeout_ms: int, icmp_only: bool, tcp_alive_ports: List[int], force_ping_bin: Optional[str]) -> Dict[str, object]:
    # Сначала ICMP
    res, usable_icmp = ping_icmp(host, count, timeout_ms, force_bin=force_ping_bin)
    if usable_icmp or icmp_only:
        return res
    # Фолбэк на TCP-пинг
    alive, best_ms = tcp_ping_alive(host, tcp_alive_ports, timeout_ms)
    if alive:
        return {"host": host, "sent": 1, "recv": 1, "loss_pct": 0.0, "avg_ms": best_ms, "status": "UP(TCP)"}
    else:
        return {"host": host, "sent": 1, "recv": 0, "loss_pct": 100.0, "avg_ms": None, "status": "DOWN(TCP)"}

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    examples = r"""
Примеры:
  # 1) Один IP
  ping_net.py 192.168.113.254 -c 2

  # 2) Несколько адресов (через запятую/пробел)
  ping_net.py "192.168.10.254,192.168.99.254 192.168.30.254" -c 2

  # 3) Сеть /24
  ping_net.py 192.168.1.0/24 --only-up --sort rtt

  # 4) Файл со списком
  ping_net.py hosts.txt --format json --pretty --out out/ping.json

  # 5) stdin
  printf "8.8.8.8\n1.1.1.1\n" | ping_net.py - -c 2

  # 6) Порт-скан для сети (только UP-хосты, порты 22,80,443)
  ping_net.py 192.168.1.0/24 --scan-ports --scan-only-up

  # 7) Порт-скан заданных портов/диапазонов
  ping_net.py hosts.txt --scan-ports --ports 22,80,443,1-1024 --ps-jobs 500

  # 8) Скан всех портов (осторожно!)
  ping_net.py 192.168.1.10 --scan-ports --ports all -w 1500

  # 9) Только ICMP (без TCP-фолбэка)
  ping_net.py 192.168.1.0/24 --icmp-only

  # 10) Диагностика окружения
  ping_net.py --diag
"""
    p = argparse.ArgumentParser(
        prog="ping_net.py",
        description="Параллельный пинг (ICMP/TCP) и проверка портов для адреса/списка/сети. Форматы: table/CSV/JSON.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=examples,
    )
    # target опционален для --diag
    p.add_argument("target", nargs="?", help="Файл со списком, '-' (stdin), строка адресов или сеть в CIDR (напр., 192.168.1.0/24)")
    p.add_argument("-c", "--count", type=int, default=10, help="Число ICMP-пакетов на хост (по умолчанию 10)")
    p.add_argument("-w", "--timeout", type=int, default=1000, help="Таймаут на ICMP/TCP-операцию, мс (по умолчанию 1000)")
    p.add_argument("-j", "--jobs", type=int, default=max(8, (os.cpu_count() or 2) * 8), help="Параллелизм по хостам (по умолчанию CPU*8, мин 8)")

    p.add_argument("--only-up", action="store_true", help="Показывать только хосты с ответами (recv>0)")
    p.add_argument("--sort", choices=["loss", "rtt", "host", "open"], help="Сортировка: loss / rtt / host / open (кол-во открытых портов)")
    p.add_argument("--desc", action="store_true", help="Обратный порядок при сортировке (по убыванию)")

    p.add_argument("--format", choices=["table", "csv", "json"], default="table", help="Формат вывода")
    p.add_argument("--out", help="Файл для вывода (если не задан, печать в stdout)")
    p.add_argument("--pretty", action="store_true", help="Красивый JSON (только для --format json)")

    # Настройки ICMP/TCP ping
    p.add_argument("--icmp-only", action="store_true", help="Только ICMP; не использовать TCP-фолбэк")
    p.add_argument("--bin", help="Явный путь к ping-бинарнику (например, /bin/ping)")
    p.add_argument("--tcp-alive-ports", default="53,161,23,8291,80,443,22", help="Порты для TCP-фолбэка живости (по умолчанию сетевые)")

    # Port-scan
    p.add_argument("--scan-ports", action="store_true", help="Включить проверку открытых портов")
    p.add_argument("--ports", default="", help="Список портов/диапазонов ('22,80,443,1-1024' или 'all'); по умолчанию 22,80,443")
    p.add_argument("--ps-jobs", type=int, default=400, help="Параллелизм порт-скана (кол-во одновременных подключений)")
    p.add_argument("--scan-only-up", action="store_true", help="Сканировать порты только у хостов со статусом UP")

    # Диагностика
    p.add_argument("--diag", action="store_true", help="Диагностика окружения (ничего не пингует)")
    p.add_argument("--version", action="version", version="ping_net.py 2.0.0")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    t0 = time.time()
    args = parse_args(argv)

    if args.diag:
        run_diag()
        return 0

    if not args.target:
        print("Не указана цель (файл/адреса/сеть). См. --help", file=sys.stderr)
        return 2

    try:
        hosts = resolve_targets(args.target)
    except Exception as e:
        print(f"Не удалось разобрать цель '{args.target}': {e}", file=sys.stderr)
        return 2

    if not hosts:
        print("Список хостов пуст.")
        return 0

    # шапка лога
    print(
        f"start ts={time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"target={args.target} hosts={len(hosts)} count={args.count} timeout={args.timeout}ms jobs={args.jobs}",
        flush=True
    )

    # --- пинг по хостам (ICMP→TCP фолбэк) ---
    tcp_alive_ports = parse_ports(args.tcp_alive_ports)
    results_map: Dict[str, Dict[str, object]] = {}

    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(probe_host, h, args.count, args.timeout, args.icmp_only, tcp_alive_ports, args.bin): h for h in hosts}
        for fut in cf.as_completed(futs):
            h = futs[fut]
            try:
                results_map[h] = fut.result()
            except Exception as e:
                results_map[h] = {"host": h, "sent": args.count, "recv": 0, "loss_pct": 100.0, "avg_ms": None, "status": f"ERROR: {e}"}

    rows: List[Dict[str, object]] = [results_map[h] for h in hosts]

    # --- при необходимости — порт-скан ---
    do_scan = args.scan_ports or bool(args.ports.strip())
    open_map: Dict[str, List[int]] = {}
    ports_for_scan = parse_ports(args.ports) if args.ports.strip() else [22, 80, 443]

    if do_scan:
        if args.scan_only_up:
            hosts_to_scan = [r["host"] for r in rows if r.get("recv", 0) > 0]
        else:
            hosts_to_scan = [r["host"] for r in rows]

        if hosts_to_scan and ports_for_scan:
            open_map = ports_scan_for_hosts(hosts_to_scan, ports_for_scan, args.timeout, args.ps_jobs)
        else:
            open_map = {h: [] for h in hosts_to_scan}

        # прикрепим результаты
        for r in rows:
            r["open_ports"] = open_map.get(r["host"], [])

    # --- фильтрация/сортировка/вывод ---
    if args.only_up:
        rows = [r for r in rows if r.get("recv", 0) > 0]

    rows = sort_rows(rows, args.sort, args.desc)

    out_text: Optional[str] = None
    include_ports = do_scan

    if args.format == "table":
        out_text = fmt_table(rows, show_ports=include_ports, ports_show_limit=20)

    elif args.format == "csv":
        if args.out:
            out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8", newline="") as fp:
                write_csv(rows, fp, include_ports=include_ports)
            print(f"saved_csv={out_path}")
        else:
            write_csv(rows, sys.stdout, include_ports=include_ports)

    elif args.format == "json":
        s = dumps_json(rows, pretty=args.pretty)
        if args.out:
            out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(s, encoding="utf-8")
            print(f"saved_json={out_path}")
        else:
            out_text = s

    if out_text is not None:
        print(out_text)

    dur = time.time() - t0
    print(f"done hosts={len(hosts)} shown={len(rows)} dur={dur:.2f}s", flush=True)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(130)

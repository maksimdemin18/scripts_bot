 
#!/home/maksimd/TelegramBot/.venv/bin/python3
# -*- coding: utf-8 -*-

import argparse, json, sys, time, os, csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Настройки по умолчанию --------------------------------------------------
DEFAULT_KEY_PATH = os.environ.get("SSH_EXEC_KEY", str(Path.home() / ".ssh/rutoll_id_rsa"))
DEFAULT_USER = os.environ.get("SSH_EXEC_USER", "root")
# ------------------------------------------------------------------------------

def _preprocess_keyval_argv(argv: list[str]) -> list[str]:
    """
    Поддержка аргументов формата key=value (как присылает бот):
      user=support host=1.2.3.4 cmd="hostname -f" preset=...
    Преобразуем в флаги argparse. preset игнорируем.
    """
    if len(argv) <= 1:
        return argv

    map_to_flag = {
        "user": "--user",
        "port": "--port",
        "jobs": "--jobs",
        "cmd": "--cmd",
        "cmd_file": "--cmd-file",
        "cmd-file": "--cmd-file",
        "host": "--host",
        "hosts": "--hosts",
        "known_hosts": "--known-hosts",
        "known-hosts": "--known-hosts",
        "connect_timeout": "--connect-timeout",
        "cmd_timeout": "--cmd-timeout",
        "format": "--format",
        "out": "--out",
        "file": None,  # позиционный
    }
    bool_keys = {"sudo": "--sudo", "verbose": "--verbose"}

    outv = [argv[0]]
    positionals: list[str] = []

    for tok in argv[1:]:
        if "=" not in tok:
            outv.append(tok)
            continue
        key, val = tok.split("=", 1)
        k = key.strip().lower()
        v = val.strip()

        if k == "preset":
            continue

        if k in bool_keys:
            if v.lower() in ("1", "true", "yes", "y", "on"):
                outv.append(bool_keys[k])
            continue

        if k in map_to_flag:
            flag = map_to_flag[k]
            if flag is None:
                if v:
                    positionals.append(v)
            else:
                outv.extend([flag, v])
            continue

        outv.append(tok)

    outv.extend(positionals)
    return outv


def parse_args():
    p = argparse.ArgumentParser(
        description="SSH executor: выполняет команды на списке хостов по ключу"
    )
    # опциональный позиционный файл — можно использовать --host/--hosts
    p.add_argument("file", nargs="?", help="Путь к файлу со списком хостов (по одному в строке)")

    p.add_argument("--user", default=DEFAULT_USER, help="Имя пользователя SSH")
    p.add_argument("--port", type=int, default=22, help="SSH порт")
    p.add_argument("--jobs", "-j", type=int, default=8, help="Параллельных подключений")

    p.add_argument("--cmd", action="append", help="Команда для выполнения (можно несколько раз)")
    p.add_argument("--cmd-file", help="Файл с командами (по одной в строке)")

    p.add_argument("--sudo", action="store_true", help="Выполнять команды через sudo -n (без пароля)")
    p.add_argument("--verbose", action="store_true", help="Подробный лог в stdout")

    p.add_argument("--host", action="append", help="Хост/IP, можно указывать несколько раз")
    p.add_argument("--hosts", help="Список хостов через запятую")

    p.add_argument("--known-hosts", choices=["ignore", "add", "strict"], default="add",
                   help="Политика known_hosts: ignore=не проверять, add=добавлять, strict=строго")
    p.add_argument("--connect-timeout", type=float, default=10.0, help="Таймаут подключения, сек")
    p.add_argument("--cmd-timeout", type=float, default=60.0, help="Таймаут одной команды, сек")

    # ⬇ добавили 'plain' и сделали его значением по умолчанию
    p.add_argument("--format", choices=["plain", "json", "csv"], default="plain", help="Формат вывода")
    p.add_argument("--out", help="Путь к файлу для вывода (plain/json/csv)")
    return p.parse_args()


def load_hosts(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hosts file not found: {path}")
    hosts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        hosts.append(s)
    return hosts


def resolve_hosts(args):
    hosts = []
    if args.host:
        hosts.extend([h.strip() for h in args.host if h.strip()])
    if args.hosts:
        hosts.extend([h.strip() for h in args.hosts.split(",") if h.strip()])
    if not hosts and args.file:
        hosts = load_hosts(args.file)
    if not hosts:
        raise ValueError("Не заданы хосты: укажите --host/--hosts или файл со списком")
    return hosts


def load_commands(args) -> list[str]:
    cmds = []
    if args.cmd:
        cmds.extend([c for c in args.cmd if c and c.strip()])
    if args.cmd_file:
        fp = Path(args.cmd_file)
        if not fp.exists():
            raise FileNotFoundError(f"Commands file not found: {fp}")
        for line in fp.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cmds.append(s)
    if not cmds:
        raise ValueError("Не заданы команды: укажите хотя бы --cmd или --cmd-file")
    return cmds


def resolve_key_path() -> str:
    kp = Path(DEFAULT_KEY_PATH)
    if not kp.exists():
        raise FileNotFoundError(
            f"SSH ключ не найден: {kp}\n"
            "Укажите корректный путь в скрипте (DEFAULT_KEY_PATH) "
            "или в переменной окружения SSH_EXEC_KEY."
        )
    return str(kp)


def run_for_host(host: str, *, user: str, port: int, key_path: str,
                 known_hosts_policy: str, cmds: list[str],
                 sudo: bool, connect_timeout: float, cmd_timeout: float,
                 verbose: bool):
    """
    Возвращает результат по хосту.
    """
    t0_total = time.time()
    res = {
        "host": host,
        "ok": False,
        "connect_error": None,
        "commands": [],
        "total_duration": 0.0,
    }

    if verbose:
        print(f"[.] {host}: подключение...", flush=True)

    try:
        import paramiko

        client = paramiko.SSHClient()
        if known_hosts_policy == "ignore":
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        elif known_hosts_policy == "add":
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.load_system_host_keys()
        else:  # strict
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.load_system_host_keys()

        client.connect(
            hostname=host,
            username=user,
            port=port,
            key_filename=key_path,
            timeout=connect_timeout,
            banner_timeout=connect_timeout,
            auth_timeout=connect_timeout,
            allow_agent=False,
            look_for_keys=False,
            compress=True,
        )

        for raw_cmd in cmds:
            cmd = f"sudo -n {raw_cmd}" if sudo else raw_cmd
            t0 = time.time()
            if verbose:
                print(f"    -> {host}: exec `{cmd}`", flush=True)
            try:
                chan = client.get_transport().open_session(timeout=connect_timeout)
                chan.settimeout(cmd_timeout)
                chan.exec_command(cmd)

                stdout_chunks = []
                stderr_chunks = []

                while True:
                    if chan.recv_ready():
                        stdout_chunks.append(chan.recv(65536))
                    if chan.recv_stderr_ready():
                        stderr_chunks.append(chan.recv_stderr(65536))
                    if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                        break
                    time.sleep(0.02)

                exit_status = chan.recv_exit_status()
                duration = time.time() - t0
                stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
                stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")

                res["commands"].append({
                    "cmd": raw_cmd if not sudo else f"(sudo) {raw_cmd}",
                    "ok": exit_status == 0,
                    "exit_status": exit_status,
                    "stdout": stdout,
                    "stderr": stderr,
                    "duration": round(duration, 3),
                })
            except Exception as e:
                duration = time.time() - t0
                res["commands"].append({
                    "cmd": raw_cmd if not sudo else f"(sudo) {raw_cmd}",
                    "ok": False,
                    "exit_status": None,
                    "stdout": "",
                    "stderr": f"{type(e).__name__}: {e}",
                    "duration": round(duration, 3),
                })
                if verbose:
                    print(f"    !! {host}: ошибка команды `{raw_cmd}`: {e}", flush=True)

        client.close()
        res["ok"] = all(c["ok"] for c in res["commands"])
    except Exception as e:
        res["connect_error"] = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"!! {host}: ошибка подключения: {e}", flush=True)
    finally:
        res["total_duration"] = round(time.time() - t0_total, 3)

    return res


def write_output_json(path: Path, payload: dict, pretty: bool = True):
    data = json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    print(f"Saved to {path}")


def write_output_csv(path: Path, results: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host", "ok", "failed_count", "last_exit_status", "first_error"])
        for h in results:
            fails = [c for c in h["commands"] if not c["ok"]]
            last_status = h["commands"][-1]["exit_status"] if h["commands"] else None
            first_err = (fails[0]["stderr"][:500] if fails else "") if not h["connect_error"] else h["connect_error"]
            w.writerow([h["host"], h["ok"], len(fails), last_status, first_err])
    print(f"Saved to {path}")


def build_plain_output(results: list[dict]) -> str:
    """
    Печатаем просто: (опционально host), затем команда и её stdout.
    Для sudo убираем скобки: '(sudo) cmd' -> 'sudo cmd'
    """
    many_hosts = len(results) > 1
    blocks = []

    for host_res in results:
        if host_res["connect_error"]:
            blocks.append(f"# {host_res['host']}\n# connect error: {host_res['connect_error']}")
            continue

        cmd_blocks = []
        for c in host_res["commands"]:
            cmd_line = c["cmd"].replace("(sudo) ", "sudo ")
            out = c["stdout"].rstrip("\n")
            if out:
                cmd_blocks.append(f"{cmd_line}\n{out}")
            else:
                # если пусто, покажем хотя бы команду (stderr не печатаем в plain)
                cmd_blocks.append(f"{cmd_line}")

        host_block = "\n\n".join(cmd_blocks)
        if many_hosts:
            blocks.append(f"{host_res['host']}\n{host_block}")
        else:
            blocks.append(host_block)

    text = "\n\n".join(blocks).rstrip() + "\n"
    return text


def main():
    # поддержка key=value до argparse
    sys.argv = _preprocess_keyval_argv(sys.argv)

    args = parse_args()

    hosts = resolve_hosts(args)
    cmds = load_commands(args)
    key_path = resolve_key_path()

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {
            ex.submit(
                run_for_host,
                host,
                user=args.user,
                port=args.port,
                key_path=key_path,
                known_hosts_policy=args.known_hosts,
                cmds=cmds,
                sudo=bool(args.sudo),
                connect_timeout=args.connect_timeout,
                cmd_timeout=args.cmd_timeout,
                verbose=args.verbose
            ): host for host in hosts
        }

        # печатаем статусные строки только не в plain (или когда явно verbose)
        print_status = (args.format in ("json", "csv")) or args.verbose

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if print_status:
                status = "OK" if res["ok"] and not res["connect_error"] else "FAIL"
                print(f"[{status}] {res['host']} ({res['total_duration']}s)")

    if args.format == "plain":
        txt = build_plain_output(results)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(txt, encoding="utf-8")
            print(f"Saved to {out}")
        else:
            # печатаем напрямую в stdout
            sys.stdout.write(txt)
        return 0

    # JSON/CSV режимы как раньше
    payload = {
        "ok": all(h["ok"] and not h["connect_error"] for h in results),
        "ts": time.time(),
        "params": {
            "file": args.file,
            "user": args.user,
            "port": args.port,
            "jobs": args.jobs,
            "sudo": bool(args.sudo),
            "known_hosts": args.known_hosts,
            "connect_timeout": args.connect_timeout,
            "cmd_timeout": args.cmd_timeout,
            "format": args.format,
            "out": args.out,
        },
        "results": results,
    }

    if args.format == "json":
        if args.out:
            write_output_json(Path(args.out), payload, pretty=True)
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.format == "csv":
        if not args.out:
            print("Для CSV требуется --out (путь к файлу)", file=sys.stderr)
            return 2
        write_output_csv(Path(args.out), results)
    else:
        print("Not implemented here")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, sys, time, re, subprocess, ipaddress, shutil, socket
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

# ===== Параметры по умолчанию =====
SSH_KEY_PATH = "/home/maksimd/.ssh/rutoll_id_rsa"  # «зашитый» ключ
DEFAULT_JOBS = 100
SSH_JOBS_CAP = 32

# Каталоги по каркасу бота
ROOT_DIR = Path(__file__).resolve().parents[1]            # scripts/scan_inventory
OUT_DIR_DEFAULT = ROOT_DIR / "out"                        # итоговые артефакты

def parse_args():
    p = argparse.ArgumentParser(
        description="Скан сетей и генерация Ansible inventory. Сети передаются ключом --network (можно повторять)."
    )
    p.add_argument("--ssh-user", default="support", help="SSH пользователь (default: support)")
    p.add_argument("-n", "--network", action="append", required=True,
                   help="Сеть CIDR или одиночный IP. Можно указывать несколько раз.")
    p.add_argument("-j", "--jobs", type=int, default=DEFAULT_JOBS, help=f"Параллелизм (default: {DEFAULT_JOBS})")
    p.add_argument("--discovery", choices=["auto", "icmp", "nmap", "tcp22"], default="auto",
                   help="Метод обнаружения хостов (default: auto)")
    p.add_argument("--out", help="Путь к manifest.json (если относительный — от {out_dir})")
    # служебные (скрытые) флаги
    p.add_argument("--pretty", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()

def _normalize_networks(items: List[str]) -> List[str]:
    nets: List[str] = []
    for s in items:
        s = s.strip()
        if not s:
            continue
        try:
            _ = ipaddress.ip_network(s, strict=False)
            nets.append(s)
        except ValueError:
            try:
                ipaddress.ip_address(s)
                nets.append(f"{s}/32")
            except ValueError:
                # «только итог»: мусор тихо игнорируем
                pass
    return nets

def _iter_hosts(net: ipaddress._BaseNetwork) -> List[str]:
    if net.prefixlen == net.max_prefixlen:
        return [str(net.network_address)]
    return [str(ip) for ip in net.hosts()]

def _ping_icmp(ip: str) -> Optional[str]:
    try:
        subprocess.check_output(["ping", "-n", "-4", "-W", "1", "-c", "1", ip],
                                stderr=subprocess.DEVNULL)
        return ip
    except subprocess.CalledProcessError:
        return None

def _tcp_ping_22(ip: str, timeout: float = 0.5) -> Optional[str]:
    try:
        with socket.create_connection((ip, 22), timeout=timeout):
            return ip
    except Exception:
        return None

def _ssh_hostname(ip: str, ssh_user: str) -> Optional[str]:
    try:
        cmd = [
            "ssh",
            "-i", SSH_KEY_PATH,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=5",
            "-o", "BatchMode=yes",
            f"{ssh_user}@{ip}",
            "hostname"
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except subprocess.CalledProcessError:
        return None

def _clean_hostname(hn: str) -> str:
    hn = re.sub(r'^.*?ln', 'ln', hn)
    hn = re.sub(r'-1$', '', hn)
    return hn.replace('atm-lower', 'atm-low').replace('atm-upper', 'atm-up')

def _group_hosts(hosts: List[Tuple[Optional[str], str]]) -> Dict[str, List[Tuple[str, str]]]:
    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for hostname, ip in hosts:
        if hostname is None:
            groups['unknown'].append((ip, ip)); continue
        if hostname == ip:
            groups['ungrouped'].append((hostname, ip)); continue
        cleaned = _clean_hostname(hostname)
        parts = cleaned.split('-')
        if len(parts) >= 2 and parts[-2] == 'atm':
            group_name = f"atm-{'low' if parts[-1] == 'low' else 'up'}"
        elif len(parts) > 1:
            group_name = parts[-1]
        else:
            group_name = 'ungrouped'
        groups[group_name].append((cleaned, ip))
    return groups

def _sanitize_net_name(network: str) -> str:
    return re.sub(r'[^0-9A-Za-z_.-]+', '_', network)

def _create_inventory(out_base: Path,
                      network: str,
                      grouped: Dict[str, List[Tuple[str, str]]],
                      ssh_user: str) -> List[Path]:
    """
    Создаёт файлы inventory в out_base/<network>_inventory/.
    Возвращает список путей ко всем созданным файлам.
    """
    net_dir = out_base / f"{_sanitize_net_name(network)}_inventory"
    net_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    def _write_line(fh, hn, ip):
        fh.write(f"{hn} ansible_host={ip} ansible_user={ssh_user} "
                 f"ansible_ssh_private_key_file={SSH_KEY_PATH}\n")

    # all.ini
    all_path = net_dir / "all.ini"
    with all_path.open("w", encoding="utf-8") as f:
        f.write("[all]\n\n")
        for grp, items in grouped.items():
            f.write(f"[{grp}]\n")
            for hn, ip in items:
                _write_line(f, hn, ip)
            f.write("\n")
    created.append(all_path)

    # group files
    for grp, items in grouped.items():
        gp = net_dir / f"{grp}.ini"
        with gp.open("w", encoding="utf-8") as f:
            f.write(f"[{grp}]\n")
            for hn, ip in items:
                _write_line(f, hn, ip)
        created.append(gp)

    return created

def _scan_icmp(network: str, jobs: int) -> List[str]:
    ip_net = ipaddress.ip_network(network, strict=False)
    hosts = _iter_hosts(ip_net)
    active: List[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        fut = {ex.submit(_ping_icmp, ip): ip for ip in hosts}
        for f in as_completed(fut):
            r = f.result()
            if r: active.append(r)
    return active

def _scan_tcp22(network: str, jobs: int) -> List[str]:
    ip_net = ipaddress.ip_network(network, strict=False)
    hosts = _iter_hosts(ip_net)
    active: List[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        fut = {ex.submit(_tcp_ping_22, ip): ip for ip in hosts}
        for f in as_completed(fut):
            r = f.result()
            if r: active.append(r)
    return active

def _scan_nmap(network: str) -> List[str]:
    if not shutil.which("nmap"):
        return []
    try:
        out = subprocess.check_output(["nmap", "-n", "-sn", network],
                                      stderr=subprocess.DEVNULL).decode("utf-8", "ignore")
    except subprocess.CalledProcessError:
        return []
    ips: List[str] = []
    for line in out.splitlines():
        m = re.search(r"Nmap scan report for (\d+\.\d+\.\d+\.\d+)", line)
        if m:
            ips.append(m.group(1)); continue
        m2 = re.search(r"\((\d+\.\d+\.\d+\.\d+)\)", line)
        if m2:
            ips.append(m2.group(1))
    return sorted(set(ips))

def _discover_hosts(network: str, jobs: int, method: str) -> List[str]:
    if method == "icmp":
        return _scan_icmp(network, jobs)
    if method == "nmap":
        return _scan_nmap(network)
    if method == "tcp22":
        return _scan_tcp22(network, jobs)
    # auto: ICMP -> nmap -> tcp22
    ips = _scan_icmp(network, jobs)
    if ips: return ips
    ips = _scan_nmap(network)
    if ips: return ips
    return _scan_tcp22(network, jobs)

def _resolve_hostnames(ips: List[str], ssh_user: str, jobs: int) -> List[Tuple[Optional[str], str]]:
    """
    Параллельно получает hostname по SSH. Возвращает список (hostname_or_None, ip).
    """
    results: List[Tuple[Optional[str], str]] = []
    ssh_jobs = max(1, min(jobs, SSH_JOBS_CAP))
    with ThreadPoolExecutor(max_workers=ssh_jobs) as ex:
        fut = {ex.submit(_ssh_hostname, ip, ssh_user): ip for ip in ips}
        for f in as_completed(fut):
            ip = fut[f]
            try:
                hn = f.result()
            except Exception:
                hn = None
            results.append((hn, ip))
    return results

def _print_summary(net_dir: Path, up_count: int):
    ini_list = sorted(p.name for p in net_dir.glob("*.ini"))
    print("[РЕЗУЛЬТАТ]")
    print(f"  Директория: {net_dir}")
    print("  INI-файлы:")
    for name in ini_list:
        print(f"   - {name}")
    print(f"  Найдено хостов: {up_count}")

def main():
    args = parse_args()

    # Куда писать манифест/инвентари
    if args.out:
        out_manifest = Path(args.out)
        if not out_manifest.is_absolute():
            out_manifest = OUT_DIR_DEFAULT / out_manifest
        out_dir = out_manifest.parent
    else:
        out_manifest = None
        out_dir = OUT_DIR_DEFAULT
    out_dir.mkdir(parents=True, exist_ok=True)

    networks = _normalize_networks(args.network)
    if not networks:
        return 1  # «только итог»: без шума

    manifest = {
        "ok": True,
        "ts": time.time(),
        "ssh_user": args.ssh_user,
        "ssh_key_path": SSH_KEY_PATH,
        "jobs": args.jobs,
        "discovery": args.discovery,
        "networks": networks,
        "per_network": {},
        "created_files": []
    }

    for net in networks:
        up = _discover_hosts(net, jobs=args.jobs, method=args.discovery)

        if not up:
            groups: Dict[str, List[Tuple[str, str]]] = {}
            created = _create_inventory(out_dir, net, groups, args.ssh_user)
        else:
            hosts = _resolve_hostnames(up, args.ssh_user, jobs=args.jobs)
            groups = _group_hosts(hosts)
            created = _create_inventory(out_dir, net, groups, args.ssh_user)

        net_dir = Path(created[0]).parent
        _print_summary(net_dir, up_count=len(up))

        manifest["per_network"][net] = {
            "up": len(up),
            "groups": {g: len(items) for g, items in groups.items()},
            "files": [str(p) for p in created]
        }
        manifest["created_files"].extend(str(p) for p in created)

    if out_manifest:
        data = json.dumps(manifest, ensure_ascii=False, indent=2 if args.pretty else None)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(data, encoding="utf-8")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

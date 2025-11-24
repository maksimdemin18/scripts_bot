#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import html
import json
import logging
import os
import shlex
import sys
import uuid
import ast
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# --- uvloop (опционально) ---------------------------------------------------
try:
    import uvloop  # type: ignore
    uvloop.install()
except Exception:
    pass

import yaml
from aiogram import Bot, Dispatcher, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
from pydantic import BaseModel, Field, field_validator, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

# -----------------------------------------------------------------------------
# Пути приложения
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_PATH = APP_DIR / "bot.yaml"
SCRIPTS_ROOT = APP_DIR / "scripts"

# -----------------------------------------------------------------------------
# Настройки из .env
# -----------------------------------------------------------------------------
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(APP_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str = Field(alias="BOT_TOKEN")
    bot_config: Path = Field(default=DEFAULT_CONFIG_PATH, alias="BOT_CONFIG")
    logs_dir: Optional[Path] = Field(default=None, alias="LOGS_DIR")
    scripts_dir: Path = Field(default=SCRIPTS_ROOT, alias="SCRIPTS_DIR")
    run_timeout: int = Field(default=600, alias="RUN_TIMEOUT")
    max_upload_mb: int = Field(default=45, alias="MAX_UPLOAD_MB")

    owner_ids_raw: str = Field(default="", alias="OWNER_IDS")
    favorites_raw: str = Field(default="", alias="FAVORITES")

    # Настройки логирования
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_to_file: bool = Field(default=True, alias="LOG_TO_FILE")

    @field_validator("bot_config", mode="before")
    @classmethod
    def _resolve_cfg(cls, v: Any) -> Path:
        p = Path(v) if v else DEFAULT_CONFIG_PATH
        return p if p.is_absolute() else (APP_DIR / p).resolve()

    @field_validator("logs_dir", mode="before")
    @classmethod
    def _resolve_logs_dir(cls, v: Any) -> Optional[Path]:
        if not v:
            return None
        p = Path(v)
        return p if p.is_absolute() else (APP_DIR / p).resolve()

    @field_validator("scripts_dir", mode="before")
    @classmethod
    def _resolve_scripts_dir(cls, v: Any) -> Path:
        p = Path(v) if v else SCRIPTS_ROOT
        return p if p.is_absolute() else (APP_DIR / p).resolve()

    def finalize(self) -> None:
        if self.logs_dir is None:
            object.__setattr__(self, "logs_dir", (APP_DIR / "logs").resolve())
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    # парсинг списков
    @property
    def owner_ids(self) -> Set[int]:
        parts = _parse_list_any(self.owner_ids_raw, sep=",")
        out: Set[int] = set()
        for x in parts:
            with contextlib.suppress(Exception):
                out.add(int(str(x)))
        return out

    @property
    def favorites(self) -> List[str]:
        return [s for s in _parse_list_any(self.favorites_raw, sep=";") if s]


def _parse_list_any(raw: str | None, sep: str = ",") -> List[str]:
    """Принимает CSV-строку либо JSON-массив, возвращает список строк."""
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            val = json.loads(s)
        except Exception:
            try:
                val = ast.literal_eval(s)
            except Exception:
                val = None
        if isinstance(val, (list, tuple, set)):
            return [str(x).strip() for x in val if str(x).strip()]
        return []
    parts = [x.strip() for x in s.split(sep)]
    parts = [p.strip("[] ") for p in parts if p and p.strip("[] ")]
    return parts


SET = AppSettings()
SET.finalize()

BOT_TOKEN = SET.bot_token
BOT_CONFIG_PATH = SET.bot_config
LOGS_DIR = SET.logs_dir
SCRIPTS_DIR = SET.scripts_dir
RUN_TIMEOUT = SET.run_timeout
MAX_UPLOAD_MB = SET.max_upload_mb
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
OWNER_IDS = set(SET.owner_ids)
FAVORITES_STRS = list(SET.favorites)

# -----------------------------------------------------------------------------
# Логирование
# -----------------------------------------------------------------------------
BOT_LOG_PATH = LOGS_DIR / "bot.log"
root = logging.getLogger()

# уровень из .env
_level_name = getattr(SET, "log_level", "INFO")
LOG_LEVEL = getattr(logging, str(_level_name).upper(), logging.INFO)
root.setLevel(LOG_LEVEL)

fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

# очищаем предыдущие хендлеры
for h in list(root.handlers):
    root.removeHandler(h)

# всегда лог в stdout (особенно важно в Docker)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
root.addHandler(sh)

# опциональный лог в файл
if getattr(SET, "log_to_file", True):
    BOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(BOT_LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

log = logging.getLogger("script-runner-bot")

# -----------------------------------------------------------------------------
# Модели конфигов
# -----------------------------------------------------------------------------
class ScriptPreset(BaseModel):
    description: Optional[str] = None
    entry: Optional[str] = None
    args: List[Any] = Field(default_factory=list)
    defaults: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)


class ScriptPresetsFile(BaseModel):
    version: int = 1
    env: Dict[str, str] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    presets: Dict[str, ScriptPreset] = Field(default_factory=dict)


class ScriptSpec(BaseModel):
    description: Optional[str] = None
    root_dir: Path
    entry: str
    work_dir: Optional[Path] = None
    in_dir: Optional[Path] = None
    config_dir: Optional[Path] = None
    out_dir: Optional[Path] = None
    env: Dict[str, str] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    presets_file: Optional[Path] = None
    presets: Dict[str, ScriptPreset] = Field(default_factory=dict)

    @field_validator("root_dir", mode="before")
    @classmethod
    def _resolve_root(cls, v: Any) -> Path:
        p = Path(v)
        return p if p.is_absolute() else (APP_DIR / p).resolve()

    @field_validator("work_dir", "in_dir", "config_dir", "out_dir", "presets_file", mode="before")
    @classmethod
    def _resolve_child(cls, v: Any) -> Optional[Path]:
        if v is None or v == "":
            return None
        p = Path(v)
        return p if p.is_absolute() else p

    def finalize_paths(self) -> None:
        if self.work_dir is None:
            object.__setattr__(self, "work_dir", self.root_dir)
        else:
            wd = (self.root_dir / self.work_dir) if not self.work_dir.is_absolute() else self.work_dir
            object.__setattr__(self, "work_dir", wd.resolve())

        def abs_or_default(p: Optional[Path], fallback: str) -> Path:
            if p is None:
                return (self.root_dir / fallback).resolve()
            return ((self.root_dir / p) if not p.is_absolute() else p).resolve()

        object.__setattr__(self, "in_dir", abs_or_default(self.in_dir, "in"))
        object.__setattr__(self, "config_dir", abs_or_default(self.config_dir, "config"))
        object.__setattr__(self, "out_dir", abs_or_default(self.out_dir, "out"))

        pf = self.presets_file or (Path("config") / "presets.yaml")
        pf = (self.root_dir / pf) if not pf.is_absolute() else pf
        object.__setattr__(self, "presets_file", pf.resolve())

        for d in [self.in_dir, self.config_dir, self.out_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def resolve_entry_path(self, entry_override: Optional[str] = None) -> Path:
        raw = entry_override or self.entry
        p = Path(raw)
        if not p.is_absolute():
            p = (self.root_dir / p)
        return p.resolve()


class BotConfig(BaseModel):
    version: int = 1
    scripts: Dict[str, ScriptSpec] = Field(default_factory=dict)

# -----------------------------------------------------------------------------
# Глобальные структуры
# -----------------------------------------------------------------------------
SCRIPTS: Dict[str, ScriptSpec] = {}

@dataclass
class UploadContext:
    script: str
    target: str  # 'in' | 'config' | 'out'

UPLOAD_CTX_BY_CHAT: Dict[int, UploadContext] = {}

@dataclass
class RunInfo:
    run_id: str
    user_id: int
    chat_id: int
    script_name: str
    script: ScriptSpec
    entry: Path
    args: List[str]
    started_at: dt.datetime
    log_path: Path
    proc: asyncio.subprocess.Process
    artifacts: List[Path]
    finished_at: Optional[dt.datetime] = None
    returncode: Optional[int] = None

    @property
    def is_running(self) -> bool:
        return self.returncode is None and self.proc.returncode is None

RUNS: Dict[str, RunInfo] = {}


def resolve_run_for_user(token: str | None, user_id: int) -> Optional[RunInfo]:
    if not token or token == "latest":
        candidates = [r for r in RUNS.values() if r.user_id == user_id]
        if not candidates:
            return None
        candidates.sort(key=lambda r: r.started_at, reverse=True)
        return candidates[0]
    ri = RUNS.get(token)
    if ri:
        return ri
    return None

# -----------------------------------------------------------------------------
# Загрузка конфигов и пресетов
# -----------------------------------------------------------------------------
def load_script_presets(spec: ScriptSpec) -> None:
    pf = spec.presets_file
    spec.presets = {}
    if not pf or not pf.exists():
        log.info("Presets file not found for '%s': %s", spec.root_dir.name, pf)
        return
    try:
        data = yaml.safe_load(pf.read_text(encoding="utf-8")) or {}
        bundle = ScriptPresetsFile.model_validate(data)
        spec.presets = bundle.presets or {}
        if bundle.artifacts:
            spec.artifacts = bundle.artifacts
        if bundle.env:
            spec.env.update(bundle.env)
        log.info("Loaded %d presets for '%s' from %s", len(spec.presets), spec.root_dir.name, pf)
    except Exception as e:
        log.exception("Failed to load presets for '%s': %s", spec.root_dir.name, e)

def load_presets_for_all_scripts() -> None:
    for _name, spec in SCRIPTS.items():
        load_script_presets(spec)

def discover_scripts_from_dir(base_dir: Path) -> Dict[str, ScriptSpec]:
    found: Dict[str, ScriptSpec] = {}
    if not base_dir.exists():
        log.info("Scripts dir not found, skipping auto-discovery: %s", base_dir)
        return found
    for cfg_path in sorted(base_dir.glob("*/script.yaml")):
        name = cfg_path.parent.name
        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            data.setdefault("root_dir", str(cfg_path.parent))
            spec = ScriptSpec.model_validate(data)
            spec.finalize_paths()
            found[name] = spec
            log.info("Discovered script '%s' from %s", name, cfg_path)
        except Exception:
            log.exception("Failed to load script config %s", cfg_path)
    return found

def load_bot_config() -> None:
    global SCRIPTS
    if not BOT_CONFIG_PATH.exists():
        log.warning("Config not found: %s", BOT_CONFIG_PATH)
        SCRIPTS = discover_scripts_from_dir(SCRIPTS_DIR)
        return
    try:
        raw = yaml.safe_load(BOT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        cfg = BotConfig.model_validate(raw)
        for spec in cfg.scripts.values():
            spec.finalize_paths()
        config_scripts = cfg.scripts
        discovered = discover_scripts_from_dir(SCRIPTS_DIR)
        SCRIPTS = {**discovered, **config_scripts}
        load_presets_for_all_scripts()
        log.info(
            "Loaded %d scripts (config=%d, discovered=%d) from %s",
            len(SCRIPTS),
            len(config_scripts),
            len(discovered),
            BOT_CONFIG_PATH,
        )
    except Exception as e:
        log.exception("Failed to load bot config: %s", e)
        SCRIPTS = {}

# -----------------------------------------------------------------------------
# Плейсхолдеры / рендеринг / артефакты
# -----------------------------------------------------------------------------
_VAR_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")

def expand_placeholders_text(text: str, values: Dict[str, Any]) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        return str(values.get(key, m.group(0)))
    # несколько прогонов для вложенных случаев
    old = None
    cur = text
    for _ in range(3):
        if cur == old:
            break
        old = cur
        cur = _VAR_RE.sub(repl, cur)
    return cur

def expand_placeholders_in_values(values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in values.items():
        if isinstance(v, str):
            out[k] = expand_placeholders_text(v, values)
        else:
            out[k] = v
    return out

def expand_placeholders_in_args(args: List[str], values: Dict[str, Any]) -> List[str]:
    return [expand_placeholders_text(a, values) for a in args]

def values_with_paths(spec: ScriptSpec, user_values: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "root_dir": str(spec.root_dir),
        "work_dir": str(spec.work_dir),
        "in_dir": str(spec.in_dir),
        "config_dir": str(spec.config_dir),
        "out_dir": str(spec.out_dir),
    }
    out = dict(base)
    out.update(user_values or {})
    return out

def render_args(args_tpl: List[Any], values: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in args_tpl:
        s = str(item)
        if "?{" in s and s.endswith("}"):
            before, _, tail = s.partition("?{")
            var = tail[:-1]
            if values.get(var):
                out.append(before)
            continue
        if s.startswith("{") and s.endswith("}"):
            var = s[1:-1]
            val = values.get(var)
            if val is None or val == "":
                continue
            out.append(str(val))
        else:
            out.append(s)
    return out

def parse_overrides(kv: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for token in kv:
        if "=" in token:
            k, v = token.split("=", 1)
            k = k.strip()
            v = v.strip()
            vl = v.lower()
            if vl in {"true", "yes", "1", "on"}:
                out[k] = True
            elif vl in {"false", "no", "0", "off"}:
                out[k] = False
            else:
                out[k] = v
        else:
            out[token] = True
    return out

def coerce_with_defaults(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Tuple[type, Any]] = {}
    for k, dv in defaults.items():
        t = type(dv)
        if t in (bool, int, float, str):
            fields[k] = (t, dv)
        else:
            fields[k] = (type(dv), dv)
    Model = create_model("PresetValues", **fields)  # type: ignore
    known = {k: overrides.get(k, dv) for k, dv in defaults.items()}
    val = Model.model_validate(known).model_dump()
    unknown = {k: v for k, v in overrides.items() if k not in defaults}
    return {**val, **unknown}

def ensure_required(required: List[str], values: Dict[str, Any]) -> Optional[str]:
    missing = [k for k in (required or []) if not values.get(k)]
    if missing:
        return "Отсутствуют обязательные параметры: " + ", ".join(missing)
    return None

def resolve_artifacts(patterns: List[str], values: Dict[str, Any], base_dir: Path) -> List[Path]:
    paths: list[Path] = []
    for ptn in patterns:
        s = expand_placeholders_text(str(ptn), values)
        p = Path(s)
        if not p.is_absolute():
            p = (base_dir / p)
        name = p.name
        if any(ch in name for ch in ["*", "?", "["]):
            parent = p.parent.resolve()
            for m in parent.glob(name):
                if m.is_file():
                    paths.append(m.resolve())
            continue
        paths.append(p.resolve())
    uniq: list[Path] = []
    seen = set()
    for x in paths:
        rx = x.resolve()
        if rx not in seen:
            uniq.append(rx)
            seen.add(rx)
    return uniq

def detect_out_artifact(args: List[str], base_dir: Path) -> Optional[Path]:
    try:
        for i, tok in enumerate(args):
            if tok == "--out" and i + 1 < len(args):
                path = Path(args[i + 1])
            elif tok.startswith("--out="):
                path = Path(tok.split("=", 1)[1])
            else:
                continue
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            return path
    except Exception:
        return None
    return None

# -----------------------------------------------------------------------------
# Поиск скрипта по токену и автоподстановка путей
# -----------------------------------------------------------------------------
def find_script_by_token(token: str) -> Optional[tuple[str, ScriptSpec]]:
    if token in SCRIPTS:
        return token, SCRIPTS[token]

    p = Path(token)
    cand_stem = p.stem if p.suffix else token
    cand_name = p.name

    if cand_stem in SCRIPTS:
        return cand_stem, SCRIPTS[cand_stem]

    for name, spec in SCRIPTS.items():
        entry_name = Path(spec.entry).name
        entry_stem = Path(spec.entry).stem
        if cand_name == entry_name or cand_stem == entry_stem:
            return name, spec

    return None

def auto_pathify_args(script: ScriptSpec, args: List[str]) -> List[str]:
    result: List[str] = []
    i = 0

    def is_bare_name(s: str) -> bool:
        return ("/" not in s) and ("\\" not in s)

    def looks_like_filename(s: str) -> bool:
        return "." in s and not s.startswith("-")

    def find_existing_in_search(name: str) -> Optional[Path]:
        for base in [script.work_dir, script.config_dir, script.in_dir, script.root_dir]:
            cand = (base / name).resolve()
            if cand.exists() and cand.is_file():
                return cand
        return None

    while i < len(args):
        tok = args[i]

        # --out <val>
        if tok == "--out" and i + 1 < len(args):
            val = args[i + 1]
            if not os.path.isabs(val) and is_bare_name(val):
                val_path = (script.out_dir / val).resolve()
                result.extend([tok, str(val_path)])
            else:
                result.extend([tok, val])
            i += 2
            continue

        # --out=<val>
        if tok.startswith("--out="):
            _, val = tok.split("=", 1)
            if not os.path.isabs(val) and is_bare_name(val):
                val_path = (script.out_dir / val).resolve()
                result.append(f"--out={val_path}")
            else:
                result.append(tok)
            i += 1
            continue

        # --flag=value
        if tok.startswith("--") and "=" in tok:
            flag, val = tok.split("=", 1)
            if not os.path.isabs(val) and is_bare_name(val) and looks_like_filename(val):
                found = find_existing_in_search(val)
                result.append(f"{flag}={found if found else val}")
            else:
                result.append(tok)
            i += 1
            continue

        # --flag value
        if tok.startswith("--") and i + 1 < len(args):
            nxt = args[i + 1]
            if not nxt.startswith("-") and is_bare_name(nxt) and looks_like_filename(nxt):
                found = find_existing_in_search(nxt)
                if found:
                    result.extend([tok, str(found)])
                    i += 2
                    continue
            result.append(tok)
            i += 1
            continue

        # короткие флаги (-c 10)
        if tok.startswith("-"):
            result.append(tok)
            i += 1
            continue

        # позиционный
        if os.path.isabs(tok) or not is_bare_name(tok):
            result.append(tok)
            i += 1
            continue

        found = find_existing_in_search(tok)
        result.append(str(found) if found else tok)
        i += 1

    return result

# -----------------------------------------------------------------------------
# Запуск, логирование, таймаут, артефакты
# -----------------------------------------------------------------------------
def new_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]

def run_log_path(script_name: str, run_id: str) -> Path:
    return LOGS_DIR / f"{script_name}_{run_id}.log"

async def _pump_output(proc: asyncio.subprocess.Process, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            f.write(text)
            f.flush()

def autodetect_artifacts(script: ScriptSpec, started_at: dt.datetime) -> List[Path]:
    arts: List[Path] = []
    for base in [script.out_dir, script.root_dir]:
        try:
            for child in base.iterdir():
                if not child.is_file():
                    continue
                st = child.stat()
                if st.st_mtime >= started_at.timestamp() - 1:
                    arts.append(child.resolve())
        except Exception:
            continue
    out: list[Path] = []
    seen = set()
    for p in arts:
        rp = p.resolve()
        if rp not in seen:
            out.append(rp)
            seen.add(rp)
    return out

async def _watch_process(ri: RunInfo):
    try:
        await ri.proc.wait()
        ri.returncode = ri.proc.returncode
        ri.finished_at = dt.datetime.now()
        with ri.log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== [PROCESS EXIT] code={ri.returncode} at {ri.finished_at.isoformat()} ===\n")
        extra = autodetect_artifacts(ri.script, ri.started_at)
        for p in extra:
            if p not in ri.artifacts:
                ri.artifacts.append(p)
    except Exception as e:
        log.exception("Watcher error: %s", e)
    finally:
        await notify_completion(ri)

async def _enforce_timeout(ri: RunInfo, timeout: int):
    try:
        await asyncio.sleep(timeout)
        if ri.proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                ri.proc.kill()
            with ri.log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n=== [TIMEOUT] exceeded {timeout}s, process killed ===\n")
            log.warning("Run %s timed out and was killed", ri.run_id)
    except asyncio.CancelledError:
        pass

async def start_run(
    script_name: str,
    script: ScriptSpec,
    entry_path: Path,
    args: List[str],
    user_id: int,
    chat_id: int,
    extra_artifacts: Optional[List[Path]] = None,
) -> RunInfo:
    if not script.work_dir.exists():
        raise FileNotFoundError(f"Рабочая директория не найдена: {script.work_dir}")
    if not entry_path.exists():
        raise FileNotFoundError(f"Entry-файл не найден: {entry_path}")
    if not entry_path.is_file():
        raise FileNotFoundError(f"Entry не является файлом: {entry_path}")

    if entry_path.suffix.lower() == ".py":
        cmd = [sys.executable, str(entry_path), *args]
    else:
        cmd = [str(entry_path), *args]

    env = os.environ.copy()
    env.update(script.env or {})

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(script.work_dir),
        env=env,
    )

    run_id = new_run_id()
    lp = run_log_path(script_name, run_id)
    started_at = dt.datetime.now()
    with lp.open("w", encoding="utf-8") as f:
        f.write(f"=== [PROCESS START] {started_at.isoformat()} ===\n")
        f.write(f"script: {script_name}\n")
        f.write(f"entry: {entry_path}\n")
        f.write(f"args: {shlex.join(args) if args else '(none)'}\n")
        f.write(f"user_id: {user_id}\n\n")

    arts = [p.resolve() for p in (extra_artifacts or [])]
    detected = detect_out_artifact(args, script.work_dir)
    if detected:
        arts.append(detected)

    ri = RunInfo(
        run_id=run_id,
        user_id=user_id,
        chat_id=chat_id,
        script_name=script_name,
        script=script,
        entry=entry_path,
        args=list(args),
        started_at=started_at,
        log_path=lp,
        proc=proc,
        artifacts=arts,
    )

    RUNS[run_id] = ri
    asyncio.create_task(_pump_output(proc, lp))
    asyncio.create_task(_watch_process(ri))
    asyncio.create_task(_enforce_timeout(ri, RUN_TIMEOUT))
    log.info("Started run %s: %s %s", run_id, entry_path.name, shlex.join(args))
    return ri

# -----------------------------------------------------------------------------
# Уведомления и отправка файлов
# -----------------------------------------------------------------------------
_BOT: Optional[Bot] = None

async def safe_send_file(chat_id: int, path: Path, caption: str = "") -> None:
    try:
        if not path.exists() or not path.is_file():
            await _BOT.send_message(
                chat_id,
                f"Файл не найден: <code>{html.escape(str(path))}</code>",
                parse_mode=ParseMode.HTML,
            )
            return
        size = path.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            mb = round(size / (1024 * 1024), 2)
            await _BOT.send_message(
                chat_id,
                (
                    "Файл слишком большой для отправки ботом.\n"
                    f"Имя: <code>{html.escape(path.name)}</code>\n"
                    f"Размер: {mb} MB (лимит {MAX_UPLOAD_MB} MB)\n"
                    f"Путь на сервере: <code>{html.escape(str(path))}</code>"
                ),
                parse_mode=ParseMode.HTML,
            )
            return
        await _BOT.send_document(chat_id, FSInputFile(str(path)), caption=caption or path.name)
    except Exception as e:
        log.exception("send_file error: %s", e)

async def notify_completion(ri: RunInfo):
    if _BOT is None:
        return
    status = "✅ Завершён" if ri.returncode == 0 else f"⚠️ Завершён с кодом {ri.returncode}"
    started = ri.started_at.strftime("%Y-%m-%d %H:%M:%S")
    finished = ri.finished_at.strftime("%Y-%m-%d %H:%M:%S") if ri.finished_at else "—"
    text = (
        f"<b>Готово</b> run_id=<code>{ri.run_id}</code>\n"
        f"Скрипт: <code>{html.escape(ri.script_name)}</code>\n"
        f"Entry: <code>{html.escape(ri.entry.name)}</code>\n"
        f"Аргументы: <code>{html.escape(shlex.join(ri.args) if ri.args else '(нет)')}</code>\n"
        f"Начало: {started}\nКонец: {finished}\n"
        f"Статус: {status}\n"
        f"Лог: <code>{html.escape(ri.log_path.name)}</code>\n"
    )
    try:
        await _BOT.send_message(
            ri.chat_id,
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=build_run_kb(ri.run_id),
        )
        sent_any = False
        for ap in ri.artifacts:
            if ap.exists() and ap.is_file():
                await safe_send_file(ri.chat_id, ap, caption=f"run_id={ri.run_id}")
                sent_any = True
        if ri.script.artifacts:
            vals = values_with_paths(ri.script, {})
            vals = expand_placeholders_in_values(vals)
            script_level = resolve_artifacts(ri.script.artifacts, vals, ri.script.out_dir)
            for p in script_level:
                if p.exists() and p.is_file() and p not in ri.artifacts:
                    await safe_send_file(ri.chat_id, p, caption=f"run_id={ri.run_id}")
                    sent_any = True
        if not sent_any:
            await _BOT.send_message(ri.chat_id, "Артефактов не обнаружено.")
    except Exception as e:
        log.exception("Notify failed: %s", e)

# -----------------------------------------------------------------------------
# Клавиатуры
# -----------------------------------------------------------------------------
def build_run_kb(run_id: str) -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📄 Лог", callback_data=f"log:{run_id}")
    kb.button(text="⏹️ Kill", callback_data=f"kill:{run_id}")
    kb.button(text="ℹ️ Статус", callback_data=f"status:{run_id}")
    kb.button(text="📎 Файлы", callback_data=f"files:{run_id}")
    kb.adjust(2, 2)
    return kb.as_markup()

def build_favs_kb() -> Optional[types.InlineKeyboardMarkup]:
    if not FAVORITES_STRS:
        return None
    kb = InlineKeyboardBuilder()
    for i, s in enumerate(FAVORITES_STRS):
        label = s if len(s) <= 40 else s[:37] + "..."
        kb.button(text=label, callback_data=f"fav:{i}")
    kb.adjust(1)
    return kb.as_markup()

# -----------------------------------------------------------------------------
# Утилиты
# -----------------------------------------------------------------------------
def is_authorized(user_id: int) -> bool:
    return not OWNER_IDS or user_id in OWNER_IDS

def human_list(items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return "(пусто)"
    return "\n".join(f"• {x}" for x in items)

def tail_file(path: Path, lines: int = 100) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            dq: deque[str] = deque(maxlen=max(1, lines))
            for line in f:
                dq.append(line.rstrip("\n"))
        return "\n".join(dq)
    except FileNotFoundError:
        return "(лог ещё не создан)"

def html_code(s: str) -> str:
    return f"<code>{html.escape(s)}</code>"

async def handle_file_upload(m: Message):
    if not is_authorized(m.from_user.id):
        return await m.answer("⛔ Нет доступа к загрузке файлов.")
    ctx = UPLOAD_CTX_BY_CHAT.get(m.chat.id)
    if not ctx or ctx.script not in SCRIPTS:
        return await m.answer("Не задан контекст загрузки. Установите: " + html_code("/ctx <script> [in|config|out]"))

    spec = SCRIPTS[ctx.script]
    target_dir = {"in": spec.in_dir, "config": spec.config_dir, "out": spec.out_dir}.get(ctx.target, spec.in_dir)

    try:
        file_id = None
        filename = None

        if m.document:
            file_id = m.document.file_id
            filename = m.document.file_name or f"file_{uuid.uuid4().hex}"
        elif m.photo:
            file_id = m.photo[-1].file_id
            filename = f"photo_{uuid.uuid4().hex}.jpg"
        else:
            return await m.answer("Неизвестный тип вложения. Пришлите как документ/фото.")

        tg_file = await _BOT.get_file(file_id)
        dl_path = target_dir / filename
        await _BOT.download_file(tg_file.file_path, destination=dl_path)
        await m.answer(f"✅ Сохранено: <code>{html.escape(str(dl_path))}</code>", parse_mode=ParseMode.HTML)
    except Exception as e:
        log.exception("upload save failed: %s", e)
        await m.answer("Ошибка сохранения файла.")

# -----------------------------------------------------------------------------
# Инициализация бота и хэндлеры
# -----------------------------------------------------------------------------
async def make_bot() -> Tuple[Bot, Dispatcher]:
    global _BOT
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN не задан")

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    _BOT = bot
    dp = Dispatcher()

    load_bot_config()

    @dp.message(Command("start"))
    async def cmd_start(m: types.Message):
        text = (
            "Привет! Я универсальный бот для запуска скриптов по конфигу.\n\n"
            "Основные команды:\n"
            "<b>/scripts</b> — список скриптов\n"
            "<b>/run</b> &lt;script&gt; [args...] — запуск entry скрипта\n"
            "<b>/presets</b> [script] — список пресетов\n"
            "<b>/preset</b> &lt;script&gt; &lt;name&gt; [k=v флаги] — запуск пресета\n"
            "<b>/kill</b> &lt;run_id&gt; — остановить процесс\n"
            "<b>/log</b> [run_id|latest] [N] [file] — хвост лога/лог файлом\n"
            "<b>/logfile</b> [run_id|latest] — лог целиком файлом\n"
            "<b>/artifacts</b> [run_id|latest] — отослать артефакты\n"
            "<b>/ctx</b> [script] [in|config|out] — контекст для загрузки файлов\n"
            "<b>/cfg</b> [path|get|reload|example] — работа с конфигом\n\n"
            "Скрипты можно описывать в bot.yaml или как scripts/<name>/script.yaml (автопоиск).\n"
            "Пришлите файл — он сохранится в каталог, заданный командой /ctx.\n"
            + ("Доступ ограничен владельцами." if OWNER_IDS else "")
        )
        await m.answer(text, reply_markup=build_favs_kb())

    # Алиас на /scripts
    @dp.message(Command("list"))
    async def cmd_list_alias(m: types.Message):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        if not SCRIPTS:
            return await m.answer("Скриптов нет. Добавьте их в bot.yaml.")
        lines = []
        for name, spec in SCRIPTS.items():
            lines.append(f"• <b>{html.escape(name)}</b> — {html.escape(spec.description or '')}")
        await m.answer("Скрипты:\n" + "\n".join(lines))

    @dp.message(Command("scripts"))
    async def cmd_scripts(m: types.Message):
        return await cmd_list_alias(m)

    @dp.message(Command("presets"))
    async def cmd_presets(m: types.Message, command: types.CommandObject):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        script = (command.args or "").strip()
        if script:
            found = find_script_by_token(script)
            if not found:
                return await m.answer("Скрипт не найден.")
            script_name, spec = found
            if not spec.presets:
                return await m.answer("Пресетов нет.")
            lines = [f"Пресеты для <b>{html.escape(script_name)}</b>:"]
            for n, pr in spec.presets.items():
                lines.append(f"• <b>{html.escape(n)}</b> — {html.escape(pr.description or '')}")
            return await m.answer("\n".join(lines))
        if not SCRIPTS:
            return await m.answer("Скриптов нет.")
        parts = []
        for name, spec in SCRIPTS.items():
            if not spec.presets:
                continue
            items = ", ".join(sorted(spec.presets.keys()))
            parts.append(f"• <b>{html.escape(name)}</b>: {html.escape(items)}")
        await m.answer("Пресеты:\n" + ("\n".join(parts) if parts else "(пусто)"))

    @dp.message(Command("preset"))
    async def cmd_preset(m: types.Message, command: types.CommandObject):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        args_line = (command.args or "").strip()
        if not args_line:
            return await m.answer("Нужно: " + html_code("/preset <script> <name> [k=v...]"))
        try:
            parts = shlex.split(args_line)
        except ValueError as e:
            return await m.answer(f"Ошибка парсинга: {html.escape(str(e))}")
        if len(parts) < 2:
            return await m.answer("Нужно: " + html_code("/preset <script> <name> [k=v...]"))
        script_token, preset_name, *kv = parts

        found = find_script_by_token(script_token)
        if not found:
            return await m.answer("Скрипт не найден.")
        script_name, spec = found

        pr = spec.presets.get(preset_name)
        if not pr:
            return await m.answer("Пресет не найден.")

        entry_path = spec.resolve_entry_path(pr.entry)
        overrides = parse_overrides(kv)
        base_values = coerce_with_defaults(pr.defaults, overrides)

        # 1) добавляем системные пути
        values = values_with_paths(spec, base_values)
        # 2) раскрываем плейсхолдеры внутри defaults (file="{config_dir}/hosts.txt" и т.п.)
        values = expand_placeholders_in_values(values)

        if (err := ensure_required(pr.required, values)):
            return await m.answer(err)

        # 3) рендер аргументов из шаблонов
        final_args = render_args(pr.args, values)
        # 4) раскрываем плейсхолдеры прямо внутри аргументов (на случай, если они не были как {var})
        final_args = expand_placeholders_in_args(final_args, values)
        # 5) автоподстановка путей (hosts.txt → поиск по каталогам; --out → out_dir)
        final_args = auto_pathify_args(spec, final_args)

        preset_arts = resolve_artifacts(pr.artifacts or [], values, spec.out_dir)
        try:
            ri = await start_run(
                script_name,
                spec,
                entry_path,
                final_args,
                m.from_user.id,
                m.chat.id,
                extra_artifacts=preset_arts,
            )
        except FileNotFoundError as e:
            log.error("Failed to start preset '%s/%s': %s", script_name, preset_name, e)
            return await m.answer(f"Не удалось запустить: {html.escape(str(e))}")
        await m.answer(
            (
                "▶️ <b>Запуск пресета</b> <code>{}</code> / <code>{}</code>\n"
                "Entry: <code>{}</code>\n"
                "Аргументы: <code>{}</code>\n"
                "run_id: <code>{}</code>\n"
                "Таймаут: {} c\n"
                "Лог: <code>{}</code>\n"
            ).format(
                html.escape(script_name),
                html.escape(preset_name),
                html.escape(entry_path.name),
                html.escape(shlex.join(final_args) if final_args else "(нет)"),
                ri.run_id,
                RUN_TIMEOUT,
                html.escape(ri.log_path.name),
            ),
            reply_markup=build_run_kb(ri.run_id),
        )

    @dp.message(Command("run"))
    async def cmd_run(m: types.Message, command: types.CommandObject):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        args_line = (command.args or "").strip()
        if not args_line:
            return await m.answer("Использование: " + html_code("/run <script> [args...]"))
        try:
            parts = shlex.split(args_line)
        except ValueError as e:
            return await m.answer(f"Ошибка парсинга: {html.escape(str(e))}")

        script_token, *script_args = parts
        found = find_script_by_token(script_token)
        if not found:
            have = ", ".join(SCRIPTS.keys()) or "(нет)"
            return await m.answer(f"Скрипт не найден. Доступные: {html.escape(have)}")
        script_name, spec = found

        # Базовые значения плейсхолдеров для /run (пути скрипта)
        run_values = values_with_paths(spec, {})
        run_values = expand_placeholders_in_values(run_values)

        # 1) раскрываем {config_dir} и прочие в аргументах, если пользователь ввёл руками
        script_args = expand_placeholders_in_args(script_args, run_values)
        # 2) автоподстановка путей: hosts.txt, apd_803.txt, --out result.json → out_dir/result.json
        script_args = auto_pathify_args(spec, script_args)

        entry_path = spec.resolve_entry_path(None)
        try:
            ri = await start_run(script_name, spec, entry_path, script_args, m.from_user.id, m.chat.id)
        except FileNotFoundError as e:
            log.error("Failed to start run '%s': %s", script_name, e)
            return await m.answer(f"Не удалось запустить: {html.escape(str(e))}")
        await m.answer(
            (
                "▶️ <b>Запуск</b>: <code>{}</code>\n"
                "Entry: <code>{}</code>\n"
                "Аргументы: <code>{}</code>\n"
                "run_id: <code>{}</code>\n"
                "Таймаут: {} c\n"
                "Лог: <code>{}</code>\n"
            ).format(
                html.escape(script_name),
                html.escape(entry_path.name),
                html.escape(shlex.join(script_args) if script_args else "(нет)"),
                ri.run_id,
                RUN_TIMEOUT,
                html.escape(ri.log_path.name),
            ),
            reply_markup=build_run_kb(ri.run_id),
        )

    @dp.message(Command("kill"))
    async def cmd_kill(m: types.Message, command: types.CommandObject):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        run_id = (command.args or "").strip()
        if not run_id:
            return await m.answer("Использование: " + html_code("/kill <run_id>"))
        ri = RUNS.get(run_id)
        if not ri:
            return await m.answer("run_id не найден.")
        if ri.proc.returncode is not None:
            return await m.answer(f"Процесс уже завершён (code={ri.proc.returncode}).")
        with contextlib.suppress(ProcessLookupError):
            ri.proc.kill()
        with ri.log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== [KILLED BY USER] user_id={m.from_user.id} at {dt.datetime.now().isoformat()} ===\n")
        await m.answer("⏹️ Процесс остановлен.")
        log.warning("Run %s killed by user %s", run_id, m.from_user.id)

    @dp.message(Command("log"))
    async def cmd_log(m: types.Message, command: types.CommandObject):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        parts = shlex.split((command.args or "").strip()) if (command.args or "").strip() else []
        run_id: Optional[str] = None
        n_lines = 100
        force_file = False
        for tok in parts:
            lt = tok.lower()
            if lt in {"file", "as_file"}:
                force_file = True
            elif tok.isdigit():
                n_lines = int(tok)
            else:
                run_id = tok
        ri = resolve_run_for_user(run_id, m.from_user.id)
        if not ri:
            return await m.answer("Запусков пока не было или run_id не найден.")
        if force_file:
            return await m.answer_document(FSInputFile(str(ri.log_path)), caption=f"run_id={ri.run_id}")
        text = tail_file(ri.log_path, lines=n_lines)
        head = (
            f"<b>Лог</b> <code>{html.escape(ri.script_name)}</code> (run_id=<code>{html.escape(ri.run_id)}</code>)\n"
            f"Последние {n_lines} строк:\n"
        )
        body = html.escape(text)
        if len(head) + len(body) < 3500:
            await m.answer(head + f"<pre>{body}</pre>", reply_markup=build_run_kb(ri.run_id))
        else:
            await m.answer_document(FSInputFile(str(ri.log_path)), caption=f"run_id={ri.run_id}")

    @dp.message(Command("logfile"))
    async def cmd_logfile(m: types.Message, command: types.CommandObject):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        run_id = (command.args or "").strip() or "latest"
        ri = resolve_run_for_user(run_id, m.from_user.id)
        if not ri:
            return await m.answer("Запусков пока не было или run_id не найден.")
        await safe_send_file(m.chat.id, ri.log_path, caption=f"log run_id={ri.run_id}")

    @dp.message(Command("artifacts"))
    async def cmd_artifacts(m: types.Message, command: types.CommandObject):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        arg = (command.args or "").strip() or "latest"
        ri = resolve_run_for_user(arg, m.from_user.id)
        if not ri:
            return await m.answer("Запусков пока не было или run_id не найден.")
        if not ri.artifacts:
            await m.answer("Артефактов в реестре нет — попробую подтянуть по шаблонам скрипта…")
            vals = values_with_paths(ri.script, {})
            vals = expand_placeholders_in_values(vals)
            extra = resolve_artifacts(ri.script.artifacts or [], vals, ri.script.out_dir)
            for p in extra:
                if p not in ri.artifacts:
                    ri.artifacts.append(p)
        if not ri.artifacts:
            return await m.answer("Артефактов нет.")
        await m.answer(f"Отправляю {len(ri.artifacts)} файл(ов)…")
        for p in ri.artifacts:
            await safe_send_file(m.chat.id, p, caption=f"run_id={ri.run_id}")

    @dp.message(Command("ctx"))
    async def cmd_ctx(m: types.Message, command: types.CommandObject):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        parts = shlex.split((command.args or "").strip()) if (command.args or "").strip() else []
        if not parts:
            ctx = UPLOAD_CTX_BY_CHAT.get(m.chat.id)
            if not ctx:
                return await m.answer("Контекст не задан. Установите: " + html_code("/ctx <script> [in|config|out]"))
            spec = SCRIPTS.get(ctx.script)
            if not spec:
                return await m.answer("Контекст ссылается на несуществующий скрипт.")
            return await m.answer(
                f"Текущий контекст: <b>{html.escape(ctx.script)}</b> → <code>{html.escape(ctx.target)}</code>\n"
                f"in: <code>{html.escape(str(spec.in_dir))}</code>\n"
                f"config: <code>{html.escape(str(spec.config_dir))}</code>\n"
                f"out: <code>{html.escape(str(spec.out_dir))}</code>",
                parse_mode=ParseMode.HTML,
            )
        script = parts[0]
        target = parts[1] if len(parts) > 1 else "in"
        found = find_script_by_token(script)
        if not found:
            return await m.answer("Скрипт не найден.")
        script_name, _ = found
        if target not in {"in", "config", "out"}:
            return await m.answer("Некорректная цель. Допустимо: in|config|out")
        UPLOAD_CTX_BY_CHAT[m.chat.id] = UploadContext(script=script_name, target=target)
        await m.answer(f"✅ Контекст установлен: {script_name} → {target}")

    @dp.message(Command("cfg"))
    async def cmd_cfg(m: types.Message, command: types.CommandObject):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        sub = (command.args or "").strip() or "path"
        if sub == "path":
            return await m.answer(f"Путь к конфигу:\n<code>{html.escape(str(BOT_CONFIG_PATH))}</code>")
        if sub == "reload":
            load_bot_config()
            return await m.answer(f"Перечитал конфиг. Скриптов: {len(SCRIPTS)} (включая пресеты из файлов скриптов)")
        if sub == "get":
            if not BOT_CONFIG_PATH.exists():
                example = _save_example_bot_config()
                await m.answer("Конфиг не найден. Пример отправлен.")
                return await m.answer_document(FSInputFile(str(example)), caption=example.name)
            return await m.answer_document(FSInputFile(str(BOT_CONFIG_PATH)), caption=BOT_CONFIG_PATH.name)
        if sub == "example":
            path = _save_example_bot_config()
            return await m.answer_document(FSInputFile(str(path)), caption=path.name)
        await m.answer("Использование: " + html_code("/cfg [path|get|reload|example]"))

    # --- Избранное из .env ---------------------------------------------------
    @dp.message(Command("fav"))
    async def cmd_fav(m: types.Message):
        if not is_authorized(m.from_user.id):
            return await m.answer("⛔ Нет доступа.")
        kb = build_favs_kb()
        if not kb:
            return await m.answer("Избранные не настроены. Задайте FAVORITES в .env")
        await m.answer("Избранные задачи:", reply_markup=kb)

    @dp.callback_query(F.data.startswith("fav:"))
    async def cb_fav(cq: types.CallbackQuery):
        if not is_authorized(cq.from_user.id):
            return await cq.answer("Нет доступа", show_alert=True)
        try:
            idx = int(cq.data.split(":", 1)[1])
            s = FAVORITES_STRS[idx]
            parts = shlex.split(s)
        except Exception:
            return await cq.answer("Некорректно", show_alert=True)
        if not parts:
            return await cq.answer("Пусто", show_alert=True)

        if parts[0] == "run" and len(parts) >= 2:
            script_token = parts[1]
            script_args = parts[2:]
            found = find_script_by_token(script_token)
            if not found:
                return await cq.message.answer("Скрипт не найден.")
            script_name, spec = found
            run_values = values_with_paths(spec, {})
            run_values = expand_placeholders_in_values(run_values)
            script_args = expand_placeholders_in_args(script_args, run_values)
            script_args = auto_pathify_args(spec, script_args)
            entry_path = spec.resolve_entry_path(None)
            try:
                ri = await start_run(
                    script_name,
                    spec,
                    entry_path,
                    script_args,
                    cq.from_user.id,
                    cq.message.chat.id,
                )
            except FileNotFoundError as e:
                log.error("Failed to start run '%s' from keyboard: %s", script_name, e)
                await cq.answer(f"Не удалось запустить: {e}", show_alert=True)
                return
            await cq.message.answer(
                f"▶️ <b>Запуск (избранное)</b>: <code>{html.escape(script_name)}</code>\n"
                f"Аргументы: <code>{html.escape(shlex.join(script_args))}</code>\n"
                f"run_id: <code>{ri.run_id}</code>\n"
                f"Лог: <code>{html.escape(ri.log_path.name)}</code>",
                reply_markup=build_run_kb(ri.run_id),
            )
            return await cq.answer("Запущено")

        if parts[0] == "preset" and len(parts) >= 3:
            script_token = parts[1]
            preset_name = parts[2]
            kv = parts[3:]
            found = find_script_by_token(script_token)
            if not found:
                return await cq.message.answer("Скрипт не найден.")
            script_name, spec = found
            pr = spec.presets.get(preset_name)
            if not pr:
                return await cq.message.answer("Пресет не найден.")
            entry_path = spec.resolve_entry_path(pr.entry)
            overrides = parse_overrides(kv)
            values = values_with_paths(spec, coerce_with_defaults(pr.defaults, overrides))
            values = expand_placeholders_in_values(values)
            if (err := ensure_required(pr.required, values)):
                return await cq.message.answer(err)
            final_args = render_args(pr.args, values)
            final_args = expand_placeholders_in_args(final_args, values)
            final_args = auto_pathify_args(spec, final_args)
            preset_arts = resolve_artifacts(pr.artifacts or [], values, spec.out_dir)
            try:
                ri = await start_run(
                    script_name,
                    spec,
                    entry_path,
                    final_args,
                    cq.from_user.id,
                    cq.message.chat.id,
                    extra_artifacts=preset_arts,
                )
            except FileNotFoundError as e:
                log.error("Failed to start preset '%s/%s' from keyboard: %s", script_name, preset_name, e)
                await cq.answer(f"Не удалось запустить: {e}", show_alert=True)
                return
            await cq.message.answer(
                f"▶️ <b>Запуск пресета (избранное)</b>: <code>{html.escape(script_name)}/{html.escape(preset_name)}</code>\n"
                f"Аргументы: <code>{html.escape(shlex.join(final_args))}</code>\n"
                f"run_id: <code>{ri.run_id}</code>\n"
                f"Лог: <code>{html.escape(ri.log_path.name)}</code>",
                reply_markup=build_run_kb(ri.run_id),
            )
            return await cq.answer("Запущено")

        await cq.answer("Формат избранного: 'run ...' или 'preset ...'", show_alert=True)

    # --- callbacks для активных запусков ------------------------------------
    @dp.callback_query(F.data.startswith("kill:"))
    async def cb_kill(cq: types.CallbackQuery):
        if not is_authorized(cq.from_user.id):
            return await cq.answer("Нет доступа", show_alert=True)
        run_id = cq.data.split(":", 1)[1]
        ri = RUNS.get(run_id)
        if not ri:
            return await cq.answer("run_id не найден", show_alert=True)
        if ri.proc.returncode is not None:
            return await cq.answer("Уже завершён", show_alert=True)
        with contextlib.suppress(ProcessLookupError):
            ri.proc.kill()
        with ri.log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== [KILLED VIA BUTTON] user_id={cq.from_user.id} at {dt.datetime.now().isoformat()} ===\n")
        await cq.message.answer("⏹️ Процесс остановлен.")
        await cq.answer("Остановлен")

    @dp.callback_query(F.data.startswith("log:"))
    async def cb_log(cq: types.CallbackQuery):
        run_id = cq.data.split(":", 1)[1]
        ri = RUNS.get(run_id)
        if not ri:
            return await cq.answer("run_id не найден", show_alert=True)
        text = tail_file(ri.log_path, 100)
        body = html.escape(text)
        await cq.message.answer(
            f"<b>Лог</b> (последние 100 строк)\n<pre>{body}</pre>",
            reply_markup=build_run_kb(run_id),
        )
        await cq.answer()

    @dp.callback_query(F.data.startswith("status:"))
    async def cb_status(cq: types.CallbackQuery):
        run_id = cq.data.split(":", 1)[1]
        ri = RUNS.get(run_id)
        if not ri:
            return await cq.answer("run_id не найден", show_alert=True)
        status = "⏳ Выполняется" if ri.proc.returncode is None else f"✅ Завершён (code={ri.proc.returncode})"
        started = ri.started_at.strftime("%Y-%m-%d %H:%M:%S")
        finished = ri.finished_at.strftime("%Y-%m-%d %H:%M:%S") if ri.finished_at else "—"
        await cq.message.answer(
            (
                f"<b>Статус</b> run_id=<code>{html.escape(ri.run_id)}</code>\n"
                f"Скрипт: <code>{html.escape(ri.script_name)}</code>\n"
                f"Entry: <code>{html.escape(ri.entry.name)}</code>\n"
                f"Аргументы: <code>{html.escape(shlex.join(ri.args) if ri.args else '(нет)')}</code>\n"
                f"Начало: {started}\nКонец: {finished}\n"
                f"{status}\n"
                f"Лог: <code>{html.escape(ri.log_path.name)}</code>\n"
            ),
            reply_markup=build_run_kb(run_id),
        )
        await cq.answer()

    @dp.callback_query(F.data.startswith("files:"))
    async def cb_files(cq: types.CallbackQuery):
        run_id = cq.data.split(":", 1)[1]
        ri = RUNS.get(run_id)
        if not ri:
            return await cq.answer("run_id не найден", show_alert=True)
        if not ri.artifacts:
            await cq.message.answer("Артефактов нет.")
            return await cq.answer()
        kb = InlineKeyboardBuilder()
        for i, p in enumerate(ri.artifacts):
            label = p.name if len(p.name) <= 30 else ("…" + p.name[-29:])
            kb.button(text=label, callback_data=f"afile:{run_id}:{i}")
        kb.adjust(1)
        await cq.message.answer("Артефакты:", reply_markup=kb.as_markup())
        await cq.answer()

    @dp.callback_query(F.data.startswith("afile:"))
    async def cb_afile(cq: types.CallbackQuery):
        try:
            _, run_id, idx = cq.data.split(":", 2)
            idx = int(idx)
        except Exception:
            return await cq.answer("Некорректный запрос", show_alert=True)
        ri = RUNS.get(run_id)
        if not ri:
            return await cq.answer("run_id не найден", show_alert=True)
        if not (0 <= idx < len(ri.artifacts)):
            return await cq.answer("Файл не найден", show_alert=True)
        await safe_send_file(cq.message.chat.id, ri.artifacts[idx], caption=f"run_id={run_id}")
        await cq.answer()

    # --- загрузка файлов от пользователя ------------------------------------
    @dp.message(F.document | F.photo)
    async def on_file(m: types.Message):
        await handle_file_upload(m)

    return bot, dp

# -----------------------------------------------------------------------------
# Пример bot.example.yaml
# -----------------------------------------------------------------------------
def _save_example_bot_config() -> Path:
    example = BotConfig(
        version=1,
        scripts={
            "ping": ScriptSpec(
                description="Пример: пинг по списку хостов",
                root_dir=(APP_DIR / "scripts" / "ping"),
                entry="bin/ping.py",
                work_dir=Path("bin"),
                in_dir=Path("config"),
                config_dir=Path("config"),
                out_dir=Path("out"),
                artifacts=[],
                env={},
            ),
            "fetch_logs": ScriptSpec(
                description="Пример: сбор логов",
                root_dir=(APP_DIR / "scripts" / "fetch_logs"),
                entry="bin/fetch_logs_v3.py",
                work_dir=Path("bin"),
                in_dir=Path("config"),
                config_dir=Path("config"),
                out_dir=Path("out"),
                artifacts=[],
                env={},
            ),
        },
    )
    for spec in example.scripts.values():
        spec.finalize_paths()
    path = BOT_CONFIG_PATH.with_name("bot.example.yaml")
    dumped = yaml.safe_dump(example.model_dump(mode="python"), sort_keys=False, allow_unicode=True)
    path.write_text(dumped, encoding="utf-8")
    return path

# -----------------------------------------------------------------------------
# Точка входа
# -----------------------------------------------------------------------------
async def main() -> None:
    bot, dp = await make_bot()
    log.info("Config path: %s", BOT_CONFIG_PATH)
    log.info("Logs dir: %s", LOGS_DIR)
    log.info("Timeout: %s s, Max upload: %s MB", RUN_TIMEOUT, MAX_UPLOAD_MB)
    if OWNER_IDS:
        log.info("Owners only: %s", ", ".join(map(str, sorted(OWNER_IDS))))
    if FAVORITES_STRS:
        log.info("Favorites: %s", FAVORITES_STRS)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Bot stopped.")

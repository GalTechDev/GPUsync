#!/usr/bin/env python3
"""
gs-remote — client : synchronise un dossier vers des machines de test et y exécute des commandes.

    python gs_remote.py hosts add mac http://192.168.1.20:8765 <jeton>
    python gs_remote.py hosts list
    python gs_remote.py info all
    python gs_remote.py run all -- @python -c "import platform; print(platform.platform())"
    python gs_remote.py run win --timeout 1800 -- cargo test
    python gs_remote.py get win bench/results.json

Cibles : un nom d'hôte, plusieurs séparés par des virgules (mac,win), ou `all`.
`@python` est remplacé par l'interpréteur Python du serveur distant.

Par défaut, `run` synchronise d'abord le dossier courant (racine git si possible) vers un
workspace du même nom sur chaque machine. Seuls les fichiers modifiés sont envoyés. Les
fichiers produits à distance (target/, .venv/, résultats) sont conservés entre deux runs.

Python >= 3.9, bibliothèque standard uniquement.
"""

from __future__ import annotations

import argparse
import base64
import codecs
import fnmatch
import hashlib
import hmac
import io
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("GS_REMOTE_CONFIG", Path.home() / ".gpusync-remote" / "hosts.json"))
IGNORE_FILE = ".gsremoteignore"
DEFAULT_EXCLUDES = [
    ".git", "target", "__pycache__", "*.pyc", ".venv", "venv", "node_modules", ".DS_Store",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "*.egg-info", "dist", "build", ".gs_sync_state.json",
]

_print_lock = threading.Lock()


def out(line: str, stream=None) -> None:
    with _print_lock:
        (stream or sys.stdout).write(line)
        (stream or sys.stdout).flush()


# ─────────────────────────────────────────────────────────────────────────────
# Configuration des hôtes
# ─────────────────────────────────────────────────────────────────────────────


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text("utf-8"))
    except FileNotFoundError:
        return {"hosts": {}}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), "utf-8")
    if os.name != "nt":
        os.chmod(CONFIG_PATH, 0o600)


def resolve_targets(spec: str) -> list[tuple[str, dict]]:
    hosts = load_config()["hosts"]
    if not hosts:
        sys.exit("Aucun hôte configuré. Utilise : gs_remote.py hosts add <nom> <url> <jeton>")
    names = list(hosts) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in hosts]
    if unknown:
        sys.exit(f"Hôte(s) inconnu(s) : {', '.join(unknown)}. Configurés : {', '.join(hosts)}")
    return [(n, hosts[n]) for n in names]


# ─────────────────────────────────────────────────────────────────────────────
# HTTP signé
# ─────────────────────────────────────────────────────────────────────────────


class RemoteError(Exception):
    pass


class Remote:
    def __init__(self, name: str, host: dict):
        self.name = name
        self.url = host["url"].rstrip("/")
        self.token = host["token"]

    def request(self, method: str, path: str, body: bytes = b"", timeout: float = 60, raw: bool = False):
        ts = f"{time.time():.3f}"
        nonce = secrets.token_hex(16)
        msg = "\n".join([method, path, ts, nonce, hashlib.sha256(body).hexdigest()]).encode()
        sig = hmac.new(self.token.encode(), msg, hashlib.sha256).hexdigest()
        req = urllib.request.Request(
            self.url + path,
            data=body if method == "POST" else None,
            method=method,
            headers={"X-GS-Time": ts, "X-GS-Nonce": nonce, "X-GS-Sig": sig, "Content-Type": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", str(e))
            except Exception:
                msg = str(e)
            raise RemoteError(f"{self.name}: HTTP {e.code} — {msg}") from None
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise RemoteError(f"{self.name}: injoignable ({self.url}) — {reason}") from None
        return data if raw else json.loads(data)

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, payload=None, body: bytes | None = None, **kw):
        if body is None:
            body = json.dumps(payload or {}).encode()
        return self.request("POST", path, body=body, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Synchronisation
# ─────────────────────────────────────────────────────────────────────────────


def default_src() -> Path:
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=5)
        if top.returncode == 0 and top.stdout.strip():
            return Path(top.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return Path.cwd()


def load_excludes(src: Path, extra: list[str]) -> list[str]:
    patterns = list(DEFAULT_EXCLUDES) + list(extra)
    ignore = src / IGNORE_FILE
    if ignore.exists():
        for line in ignore.read_text("utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line.rstrip("/"))
    return patterns


def is_excluded(rel: str, patterns: list[str]) -> bool:
    parts = rel.split("/")
    for pat in patterns:
        if "/" in pat:
            if fnmatch.fnmatch(rel, pat.lstrip("/")) or rel.startswith(pat.lstrip("/") + "/"):
                return True
        elif any(fnmatch.fnmatch(p, pat) for p in parts):
            return True
    return False


def build_manifest(src: Path, patterns: list[str]) -> dict[str, list]:
    files: dict[str, list] = {}
    for dirpath, dirnames, filenames in os.walk(src):
        rel_dir = Path(dirpath).relative_to(src).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = [d for d in dirnames if not is_excluded(rel_dir + d, patterns)]
        for fn in filenames:
            rel = rel_dir + fn
            full = Path(dirpath) / fn
            if is_excluded(rel, patterns) or not full.is_file():
                continue
            h = hashlib.sha256()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            files[rel] = [full.stat().st_size, h.hexdigest()]
    return files


def sync(remote: Remote, src: Path, ws: str, files: dict) -> str:
    plan = remote.post(f"/manifest/{ws}", {"files": files})
    need, delete = plan["need"], plan["delete"]
    if not need and not delete:
        return f"à jour ({len(files)} fichiers)"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("__gs_meta__.json", json.dumps({"files": files, "delete": delete}))
        for rel in need:
            zf.write(src / rel, rel)
    payload = buf.getvalue()
    res = remote.post(f"/sync/{ws}", body=payload, timeout=900)
    return f"{res['written']} envoyé(s), {res['deleted']} supprimé(s), {len(payload) / 1e6:.1f} Mo"


# ─────────────────────────────────────────────────────────────────────────────
# Exécution
# ─────────────────────────────────────────────────────────────────────────────


class HostRun:
    def __init__(self, name: str, remote: Remote, prefix: bool):
        self.name = name
        self.remote = remote
        self.prefix = prefix
        self.job_id: str | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""

    def emit(self, text: str, final: bool = False) -> None:
        if not self.prefix:
            if text:
                out(text)
            return
        self._pending += text
        lines = self._pending.split("\n")
        last = lines.pop()  # ligne incomplète (ou "" si le texte finit par \n)
        if final and last:
            lines.append(last)
            last = ""
        self._pending = last
        tag = f"[{self.name}] "
        for line in lines:
            out(tag + line.rstrip("\r") + "\n")

    def run(self, ws: str, req: dict, sync_args: tuple | None) -> None:
        try:
            if sync_args:
                src, files = sync_args
                msg = sync(self.remote, src, ws, files)
                out(f"[{self.name}] sync : {msg}\n", sys.stderr)
            job = self.remote.post(f"/run/{ws}", req)
            self.job_id = job["id"]
            offset, delay = 0, 0.2
            while True:
                st = self.remote.get(f"/jobs/{self.job_id}?offset={offset}")
                data = base64.b64decode(st["data_b64"])
                offset = st["next_offset"]
                if data:
                    self.emit(self._decoder.decode(data))
                    delay = 0.2
                if st["status"] != "running":
                    self.emit(self._decoder.decode(b"", final=True), final=True)
                    self.result = st
                    return
                if not data:
                    time.sleep(delay)
                    delay = min(delay * 1.5, 1.0)
        except RemoteError as e:
            self.error = str(e)
            out(f"[{self.name}] ERREUR : {e}\n", sys.stderr)

    def kill(self) -> None:
        if self.job_id and self.result is None:
            try:
                self.remote.post(f"/jobs/{self.job_id}/kill")
            except RemoteError:
                pass


def cmd_run(args) -> int:
    targets = resolve_targets(args.target)
    if not args.command:
        sys.exit("Commande manquante après `--`. Exemple : gs_remote.py run all -- @python script.py")
    command = args.command
    src = Path(args.src).resolve() if args.src else default_src()
    ws = args.ws or src.name
    req = {
        "cmd": " ".join(command) if args.shell else command,
        "shell": args.shell,
        "timeout": args.timeout,
        "env": dict(kv.split("=", 1) for kv in args.env),
        "cwd": args.cwd,
    }
    sync_args = None
    if not args.no_sync:
        files = build_manifest(src, load_excludes(src, args.exclude))
        sync_args = (src, files)

    multi = len(targets) > 1
    runs = [HostRun(name, Remote(name, host), prefix=multi) for name, host in targets]
    threads = [threading.Thread(target=r.run, args=(ws, req, sync_args), daemon=True) for r in runs]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        out("\nInterruption : arrêt des jobs distants…\n", sys.stderr)
        for r in runs:
            r.kill()
        return 130

    worst = 0
    if multi or any(r.error for r in runs):
        out("\n" + "─" * 60 + "\n", sys.stderr)
    for r in runs:
        if r.error:
            worst = worst or 1
            if multi:
                out(f"  {r.name:<12} ERREUR      {r.error}\n", sys.stderr)
            continue
        st = r.result or {}
        code = st.get("exit_code")
        if st.get("status") != "done" or code != 0:
            worst = worst or (code if isinstance(code, int) and code > 0 else 1)
        if multi:
            label = "OK" if st.get("status") == "done" and code == 0 else st.get("status", "?").upper()
            out(f"  {r.name:<12} {label:<11} code {code}  {st.get('duration_s', 0):.1f} s\n", sys.stderr)
        elif st.get("status") != "done":
            out(f"[{r.name}] job terminé : {st.get('status')} (code {code})\n", sys.stderr)
    return worst


def cmd_sync(args) -> int:
    src = Path(args.src).resolve() if args.src else default_src()
    ws = args.ws or src.name
    files = build_manifest(src, load_excludes(src, args.exclude))
    rc = 0
    for name, host in resolve_targets(args.target):
        try:
            print(f"[{name}] {sync(Remote(name, host), src, ws, files)}")
        except RemoteError as e:
            print(f"[{name}] ERREUR : {e}", file=sys.stderr)
            rc = 1
    return rc


def cmd_info(args) -> int:
    rc = 0
    for name, host in resolve_targets(args.target):
        try:
            info = Remote(name, host).get("/info", timeout=60)
        except RemoteError as e:
            print(f"[{name}] ERREUR : {e}", file=sys.stderr)
            rc = 1
            continue
        if args.json:
            print(json.dumps({name: info}, indent=2, ensure_ascii=False))
            continue
        print(f"[{name}] {info['hostname']} — {info['os']} ({info['machine']}), {info['cpu_count']} cœurs CPU")
        print(f"    python : {info['python']}  ({info['python_executable']})")
        print(f"    rust   : {info.get('rustc') or 'absent'}")
        for g in info.get("gpus") or ["(aucun GPU détecté)"]:
            print(f"    gpu    : {g}")
        for g in info.get("nvidia_smi") or []:
            print(f"    nvidia : {g}")
        if info.get("workspaces"):
            print(f"    workspaces : {', '.join(info['workspaces'])}")
        for j in info.get("running_jobs") or []:
            print(f"    job en cours : {j['id']} ({j['workspace']}, {j['duration_s']:.0f} s)")
    return rc


def cmd_get(args) -> int:
    targets = resolve_targets(args.target)
    if len(targets) != 1:
        sys.exit("get : une seule cible à la fois")
    name, host = targets[0]
    ws = args.ws or default_src().name
    rel = args.remote_path.replace("\\", "/").lstrip("/")
    data = Remote(name, host).get(f"/files/{ws}/{rel}", raw=True, timeout=600)
    dest = Path(args.local or Path(rel).name)
    dest.write_bytes(data)
    print(f"[{name}] {rel} → {dest} ({len(data)} octets)")
    return 0


def cmd_reset(args) -> int:
    ws = args.ws or default_src().name
    for name, host in resolve_targets(args.target):
        try:
            Remote(name, host).post(f"/reset/{ws}")
            print(f"[{name}] workspace {ws} supprimé")
        except RemoteError as e:
            print(f"[{name}] ERREUR : {e}", file=sys.stderr)
    return 0


def cmd_kill(args) -> int:
    for name, host in resolve_targets(args.target):
        print(f"[{name}]", Remote(name, host).post(f"/jobs/{args.job}/kill"))
    return 0


def cmd_hosts(args) -> int:
    cfg = load_config()
    if args.action == "add":
        if not (args.name and args.url and args.token):
            sys.exit("Usage : gs_remote.py hosts add <nom> <url> <jeton>")
        cfg["hosts"][args.name] = {"url": args.url, "token": args.token}
        save_config(cfg)
        print(f"Hôte {args.name} ajouté ({args.url}). Config : {CONFIG_PATH}")
    elif args.action == "rm":
        cfg["hosts"].pop(args.name, None)
        save_config(cfg)
        print(f"Hôte {args.name} supprimé.")
    else:
        if not cfg["hosts"]:
            print("Aucun hôte configuré.")
        for n, h in cfg["hosts"].items():
            print(f"  {n:<12} {h['url']}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Client gs-remote : sync + exécution sur les machines de test.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("hosts", help="gérer les hôtes (add / rm / list)")
    p.add_argument("action", choices=["add", "rm", "list"])
    p.add_argument("name", nargs="?")
    p.add_argument("url", nargs="?")
    p.add_argument("token", nargs="?")
    p.set_defaults(func=cmd_hosts)

    p = sub.add_parser("info", help="matériel et logiciels de chaque hôte")
    p.add_argument("target")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_info)

    def sync_opts(p):
        p.add_argument("--src", help="dossier à synchroniser (défaut : racine git ou dossier courant)")
        p.add_argument("--ws", help="nom du workspace distant (défaut : nom du dossier)")
        p.add_argument("--exclude", action="append", default=[], help="motif à exclure (répétable)")

    p = sub.add_parser("sync", help="synchroniser le dossier sans rien exécuter")
    p.add_argument("target")
    sync_opts(p)
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("run", help="synchroniser puis exécuter une commande")
    p.add_argument("target")
    sync_opts(p)
    p.add_argument("--no-sync", action="store_true", help="ne pas synchroniser avant")
    p.add_argument("--timeout", type=float, default=600, help="secondes avant de tuer la commande (défaut 600)")
    p.add_argument("--env", action="append", default=[], metavar="K=V")
    p.add_argument("--cwd", help="sous-dossier du workspace où exécuter")
    p.add_argument("--shell", action="store_true", help="exécuter via le shell distant (cmd.exe / sh)")
    p.set_defaults(func=cmd_run, command=[])

    p = sub.add_parser("get", help="récupérer un fichier du workspace distant")
    p.add_argument("target")
    p.add_argument("remote_path")
    p.add_argument("local", nargs="?")
    p.add_argument("--ws")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("reset", help="supprimer le workspace distant")
    p.add_argument("target")
    p.add_argument("--ws")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("kill", help="tuer un job distant")
    p.add_argument("target")
    p.add_argument("job")
    p.set_defaults(func=cmd_kill)

    argv = sys.argv[1:]
    command: list[str] = []
    if "--" in argv:  # tout ce qui suit `--` est la commande distante, passée telle quelle
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    if args.cmd == "run":
        args.command = command
    try:
        sys.exit(args.func(args))
    except RemoteError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gs-remote — serveur d'exécution distante pour tester GPUsync sur plusieurs machines.

À lancer sur chaque machine de test (Mac, PC Windows, Linux). Python >= 3.9, bibliothèque
standard uniquement.

    python gs_remote_server.py                # écoute sur 0.0.0.0:8765
    python gs_remote_server.py --show-token   # affiche le jeton à donner au client

Sécurité :
  - chaque requête est signée en HMAC-SHA256 avec un jeton secret (jamais envoyé en clair),
    horodatée et munie d'un nonce (anti-rejeu) ;
  - seules les IP privées / locales sont acceptées par défaut (--allow pour restreindre) ;
  - les fichiers reçus ne peuvent pas sortir du dossier de leur workspace ;
  - chaque commande a un timeout et est tuée avec tous ses sous-processus.
  Le trafic n'est PAS chiffré : à utiliser sur un réseau local de confiance uniquement.
  Toute personne qui possède le jeton peut exécuter du code sur la machine.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import io
import ipaddress
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROTOCOL_VERSION = 1
MAX_CLOCK_SKEW_S = 300
LOG_CHUNK_MAX = 1024 * 1024
JOB_RETENTION_S = 24 * 3600
WS_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SYNC_STATE_FILE = ".gs_sync_state.json"
PY_PLACEHOLDER = "@python"


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def safe_rel_path(p: str) -> str:
    """Normalise un chemin relatif reçu du client et refuse tout ce qui sort du workspace."""
    if not isinstance(p, str) or not p:
        raise HttpError(400, "chemin vide")
    p = p.replace("\\", "/")
    if p.startswith("/") or ":" in p or "\x00" in p:
        raise HttpError(400, f"chemin refusé : {p!r}")
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if not parts or any(x == ".." for x in parts):
        raise HttpError(400, f"chemin refusé : {p!r}")
    return "/".join(parts)


def resolve_inside(base: Path, rel: str) -> Path:
    """Chemin `base/rel`, en refusant qu'il sorte de `base` (y compris via un lien symbolique)."""
    target = base / safe_rel_path(rel)
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        raise HttpError(400, f"chemin refusé (sort du workspace) : {rel!r}") from None
    return target


def run_probe(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
        if out.returncode != 0:
            return None
        return out.stdout.decode("utf-8", errors="replace").strip() or None
    except Exception:
        return None


def sign(token: str, method: str, path: str, ts: str, nonce: str, body_sha: str) -> str:
    msg = "\n".join([method.upper(), path, ts, nonce, body_sha]).encode()
    return hmac.new(token.encode(), msg, hashlib.sha256).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────────────────────────────────────


class Job:
    def __init__(self, server: "RemoteServer", ws: str, cmd, shell: bool, env: dict, timeout: float, cwd: Path):
        self.id = uuid.uuid4().hex[:12]
        self.ws = ws
        self.cmd = cmd
        self.shell = shell
        self.timeout = timeout
        self.cwd = cwd
        self.log_path = server.jobs_dir / f"{self.id}.log"
        self.status = "running"
        self.exit_code: int | None = None
        self.started = time.time()
        self.ended: float | None = None
        self._lock = threading.Lock()
        self._kill_reason: str | None = None

        full_env = os.environ.copy()
        full_env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        full_env.update({str(k): str(v) for k, v in env.items()})
        self.env = full_env

    def start(self) -> None:
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        with open(self.log_path, "wb") as log:
            try:
                self.proc = subprocess.Popen(
                    self.cmd,
                    shell=self.shell,
                    cwd=str(self.cwd),
                    env=self.env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    **kwargs,
                )
            except OSError as e:
                log.write(f"[gs-remote] impossible de lancer la commande : {e}\n".encode())
                self.status, self.exit_code, self.ended = "error", -1, time.time()
                return
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self) -> None:
        try:
            rc = self.proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.kill("timeout")
            rc = self.proc.wait()
        with self._lock:
            self.exit_code = rc
            self.ended = time.time()
            self.status = self._kill_reason or "done"

    def kill(self, reason: str = "killed") -> None:
        with self._lock:
            if self.status != "running":
                return
            self._kill_reason = reason
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def describe(self) -> dict:
        end = self.ended or time.time()
        return {
            "id": self.id,
            "workspace": self.ws,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_s": round(end - self.started, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Serveur
# ─────────────────────────────────────────────────────────────────────────────


class RemoteServer:
    def __init__(self, root: Path, token: str, allow: list, max_upload: int):
        self.root = root
        self.ws_dir = root / "workspaces"
        self.jobs_dir = root / "jobs"
        self.ws_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.allow = allow
        self.max_upload = max_upload
        self.jobs: dict[str, Job] = {}
        self.nonces: dict[str, float] = {}
        self.lock = threading.Lock()

    # --- contrôle d'accès ---------------------------------------------------

    def ip_allowed(self, addr: str) -> bool:
        ip = ipaddress.ip_address(addr)
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if self.allow:
            return any(ip in net for net in self.allow)
        return ip.is_private or ip.is_loopback or ip.is_link_local

    def check_signature(self, method: str, path: str, headers, body: bytes) -> None:
        ts, nonce, sig = headers.get("X-GS-Time"), headers.get("X-GS-Nonce"), headers.get("X-GS-Sig")
        if not (ts and nonce and sig):
            raise HttpError(401, "requête non signée")
        try:
            skew = abs(time.time() - float(ts))
        except ValueError:
            raise HttpError(401, "horodatage invalide") from None
        if skew > MAX_CLOCK_SKEW_S:
            raise HttpError(401, f"horloges désynchronisées ({skew:.0f} s d'écart)")
        expected = sign(self.token, method, path, ts, nonce, hashlib.sha256(body).hexdigest())
        if not hmac.compare_digest(expected, sig):
            raise HttpError(401, "signature invalide (mauvais jeton ?)")
        now = time.time()
        with self.lock:
            for n, t in list(self.nonces.items()):
                if now - t > 2 * MAX_CLOCK_SKEW_S:
                    del self.nonces[n]
            if nonce in self.nonces:
                raise HttpError(401, "requête rejouée")
            self.nonces[nonce] = now

    # --- workspaces ---------------------------------------------------------

    def workspace(self, name: str, create: bool = True) -> Path:
        if not WS_NAME_RE.match(name or ""):
            raise HttpError(400, f"nom de workspace invalide : {name!r}")
        p = self.ws_dir / name
        if create:
            p.mkdir(parents=True, exist_ok=True)
        elif not p.is_dir():
            raise HttpError(404, f"workspace inconnu : {name}")
        return p

    def busy(self, ws: str) -> bool:
        return any(j.ws == ws and j.status == "running" for j in self.jobs.values())

    def load_state(self, ws_path: Path) -> dict:
        try:
            return json.loads((ws_path / SYNC_STATE_FILE).read_text("utf-8"))
        except (OSError, ValueError):
            return {}

    def manifest(self, ws: str, files: dict) -> dict:
        ws_path = self.workspace(ws)
        state = self.load_state(ws_path)
        need, delete = [], []
        for rel, (size, sha) in files.items():
            rel = safe_rel_path(rel)
            target = resolve_inside(ws_path, rel)
            if state.get(rel) != sha or not target.is_file() or target.stat().st_size != size:
                need.append(rel)
        for rel in state:
            if rel not in files:
                delete.append(rel)
        return {"need": need, "delete": delete}

    def sync(self, ws: str, body: bytes) -> dict:
        ws_path = self.workspace(ws)
        if self.busy(ws):
            raise HttpError(409, f"un job tourne déjà dans le workspace {ws}")
        try:
            zf = zipfile.ZipFile(io.BytesIO(body))
        except zipfile.BadZipFile:
            raise HttpError(400, "archive zip invalide") from None
        with zf:
            total = sum(i.file_size for i in zf.infolist())
            if total > self.max_upload * 4:
                raise HttpError(413, "archive trop grosse une fois décompressée")
            meta = json.loads(zf.read("__gs_meta__.json"))
            files = {safe_rel_path(k): v for k, v in meta["files"].items()}
            written = 0
            for info in zf.infolist():
                if info.filename == "__gs_meta__.json" or info.is_dir():
                    continue
                target = resolve_inside(ws_path, info.filename)
                if target.is_symlink():
                    target.unlink()
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                mode = (info.external_attr >> 16) & 0o777
                if mode and os.name != "nt":
                    os.chmod(target, mode)
                written += 1
        state = self.load_state(ws_path)
        deleted = 0
        for rel in meta.get("delete", []):
            rel = safe_rel_path(rel)
            if rel in state and rel not in files:
                try:
                    resolve_inside(ws_path, rel).unlink()
                    deleted += 1
                except FileNotFoundError:
                    pass
        new_state = {k: v[1] for k, v in files.items()}
        (ws_path / SYNC_STATE_FILE).write_text(json.dumps(new_state), "utf-8")
        return {"written": written, "deleted": deleted, "total": len(files)}

    def reset(self, ws: str) -> dict:
        ws_path = self.workspace(ws, create=False)
        if self.busy(ws):
            raise HttpError(409, f"un job tourne déjà dans le workspace {ws}")
        shutil.rmtree(ws_path)
        return {"reset": ws}

    # --- jobs ----------------------------------------------------------------

    def run(self, ws: str, req: dict) -> dict:
        ws_path = self.workspace(ws)
        cmd = req.get("cmd")
        shell = bool(req.get("shell", False))
        if shell:
            if not isinstance(cmd, str):
                raise HttpError(400, "avec shell=true, cmd doit être une chaîne")
            cmd = cmd.replace(PY_PLACEHOLDER, f'"{sys.executable}"')
        else:
            if not (isinstance(cmd, list) and cmd and all(isinstance(c, str) for c in cmd)):
                raise HttpError(400, "cmd doit être une liste de chaînes non vide")
            cmd = [sys.executable if c == PY_PLACEHOLDER else c for c in cmd]
        cwd = resolve_inside(ws_path, req["cwd"]) if req.get("cwd") else ws_path
        if not cwd.is_dir():
            raise HttpError(400, f"dossier introuvable dans le workspace : {req.get('cwd')}")
        timeout = float(req.get("timeout", 600))
        job = Job(self, ws, cmd, shell, req.get("env") or {}, timeout, cwd)
        with self.lock:
            self.cleanup_jobs()
            self.jobs[job.id] = job
        job.start()
        return job.describe()

    def job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HttpError(404, f"job inconnu : {job_id}")
        return job

    def job_output(self, job_id: str, offset: int) -> dict:
        job = self.job(job_id)
        status = job.describe()  # lu AVANT le log : si "done", le log est complet
        with open(job.log_path, "rb") as f:
            f.seek(offset)
            data = f.read(LOG_CHUNK_MAX)
        status.update({"offset": offset, "next_offset": offset + len(data), "data_b64": base64.b64encode(data).decode()})
        if status["status"] != "running" and len(data) == LOG_CHUNK_MAX:
            status["status"] = "running"  # il reste du log à lire : le client continue de demander
            status["more_output"] = True
        return status

    def cleanup_jobs(self) -> None:
        now = time.time()
        for jid, job in list(self.jobs.items()):
            if job.ended and now - job.ended > JOB_RETENTION_S:
                del self.jobs[jid]
                try:
                    job.log_path.unlink()
                except OSError:
                    pass

    # --- informations ---------------------------------------------------------

    def info(self) -> dict:
        system = platform.system()
        gpus: list[str] = []
        nvidia = run_probe(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        if system == "Windows":
            ps = run_probe(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' (driver ' + $_.DriverVersion + ')' }",
                ],
                timeout=20,
            )
            gpus = ps.splitlines() if ps else []
        elif system == "Darwin":
            sp = run_probe(["system_profiler", "SPDisplaysDataType", "-json"], timeout=20)
            if sp:
                try:
                    for d in json.loads(sp).get("SPDisplaysDataType", []):
                        cores = d.get("sppci_cores")
                        gpus.append(d.get("sppci_model", "?") + (f" ({cores} cœurs)" if cores else ""))
                except ValueError:
                    pass
        else:
            lspci = run_probe(["sh", "-c", "lspci | grep -Ei 'vga|3d|display'"])
            gpus = lspci.splitlines() if lspci else []
        return {
            "protocol": PROTOCOL_VERSION,
            "hostname": socket.gethostname(),
            "os": f"{system} {platform.release()}",
            "os_version": platform.version(),
            "machine": platform.machine(),
            "cpu": platform.processor() or None,
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "gpus": gpus,
            "nvidia_smi": nvidia.splitlines() if nvidia else None,
            "rustc": run_probe(["rustc", "--version"]),
            "cargo": run_probe(["cargo", "--version"]),
            "workspaces": sorted(p.name for p in self.ws_dir.iterdir() if p.is_dir()),
            "running_jobs": [j.describe() for j in self.jobs.values() if j.status == "running"],
        }


def make_handler(server: RemoteServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "gs-remote/1"

        def log_message(self, fmt, *args):  # une ligne concise par requête
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]} {fmt % args}\n")

        def _send(self, status: int, payload: dict | None = None, raw: bytes | None = None) -> None:
            body = raw if raw is not None else json.dumps(payload or {}, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream" if raw is not None else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self, method: str) -> None:
            try:
                if not server.ip_allowed(self.client_address[0]):
                    raise HttpError(403, "adresse IP non autorisée")
                length = int(self.headers.get("Content-Length") or 0)
                if length > server.max_upload:
                    raise HttpError(413, "requête trop grosse")
                body = self.rfile.read(length) if length else b""
                server.check_signature(method, self.path, self.headers, body)
                url = urlparse(self.path)
                parts = [p for p in url.path.split("/") if p]
                query = parse_qs(url.query)
                result = self._route(method, parts, query, body)
                if isinstance(result, bytes):
                    self._send(200, raw=result)
                else:
                    self._send(200, result)
            except HttpError as e:
                self._send(e.status, {"error": e.message})
            except Exception as e:  # erreur interne : on la renvoie pour faciliter le diagnostic
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def _route(self, method, parts, query, body):
            if method == "GET" and parts == ["info"]:
                return server.info()
            if method == "POST" and len(parts) == 2 and parts[0] == "manifest":
                return server.manifest(parts[1], json.loads(body)["files"])
            if method == "POST" and len(parts) == 2 and parts[0] == "sync":
                return server.sync(parts[1], body)
            if method == "POST" and len(parts) == 2 and parts[0] == "reset":
                return server.reset(parts[1])
            if method == "POST" and len(parts) == 2 and parts[0] == "run":
                return server.run(parts[1], json.loads(body))
            if method == "GET" and len(parts) == 2 and parts[0] == "jobs":
                return server.job_output(parts[1], int(query.get("offset", ["0"])[0]))
            if method == "POST" and len(parts) == 3 and parts[0] == "jobs" and parts[2] == "kill":
                job = server.job(parts[1])
                job.kill("killed")
                return job.describe()
            if method == "GET" and len(parts) >= 3 and parts[0] == "files":
                ws_path = server.workspace(parts[1], create=False)
                target = resolve_inside(ws_path, "/".join(parts[2:]))
                if not target.is_file():
                    raise HttpError(404, "fichier introuvable")
                if target.stat().st_size > server.max_upload:
                    raise HttpError(413, "fichier trop gros")
                return target.read_bytes()
            raise HttpError(404, f"route inconnue : {method} /{'/'.join(parts)}")

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


def local_ips() -> list[str]:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:  # adresse de l'interface par défaut (aucun paquet n'est envoyé)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def load_or_create_token(root: Path, regen: bool) -> str:
    cfg_path = root / "server.json"
    cfg = {}
    if cfg_path.exists() and not regen:
        cfg = json.loads(cfg_path.read_text("utf-8"))
    if "token" not in cfg:
        cfg["token"] = secrets.token_urlsafe(32)
        root.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2), "utf-8")
        if os.name != "nt":
            os.chmod(cfg_path, 0o600)
    return cfg["token"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Serveur gs-remote : exécution distante de scripts pour tester GPUsync.")
    ap.add_argument("--bind", default="0.0.0.0", help="adresse d'écoute (défaut : 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", type=Path, default=Path.home() / ".gpusync-remote-server", help="dossier des workspaces et logs")
    ap.add_argument("--allow", action="append", default=[], metavar="CIDR",
                    help="réseaux autorisés, ex. 192.168.1.0/24 (répétable). Défaut : IP privées et locales")
    ap.add_argument("--max-upload-mb", type=int, default=2048)
    ap.add_argument("--show-token", action="store_true", help="affiche le jeton puis quitte")
    ap.add_argument("--regen-token", action="store_true", help="génère un nouveau jeton (l'ancien ne marche plus)")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    token = load_or_create_token(root, args.regen_token)
    if args.show_token:
        print(token)
        return

    allow = [ipaddress.ip_network(c, strict=False) for c in args.allow]
    server = RemoteServer(root, token, allow, args.max_upload_mb * 1024 * 1024)
    httpd = ThreadingHTTPServer((args.bind, args.port), make_handler(server))
    httpd.daemon_threads = True

    print(f"gs-remote serveur — {socket.gethostname()} ({platform.system()} {platform.machine()}, Python {sys.version.split()[0]})")
    print(f"  dossier   : {root}")
    print(f"  écoute    : {args.bind}:{args.port}")
    print(f"  autorisés : {', '.join(args.allow) if args.allow else 'IP privées et locales'}")
    print("\nSur la machine cliente, ajoute cet hôte avec :")
    for ip in local_ips() or ["<IP-de-cette-machine>"]:
        print(f"  python gs_remote.py hosts add <nom> http://{ip}:{args.port} <jeton>")
    print(f"\nJeton : python {Path(sys.argv[0]).name} --show-token  (à garder secret)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt : les jobs en cours sont tués.")
        for job in list(server.jobs.values()):
            job.kill("killed")


if __name__ == "__main__":
    main()

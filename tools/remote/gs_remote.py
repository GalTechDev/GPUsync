#!/usr/bin/env python3
"""
gs-remote — run commands on the GPUsync test machines.

Three roles, a single file (Python >= 3.9, standard library only):

  hub    on the dev machine (the Mac). It listens; agents come to it to fetch their tasks.
  agent  on each test machine. It LISTENS TO NOTHING: it connects to the hub (outgoing connection),
         reconnects by itself, and only runs tasks signed by the hub.
  CLI    on the dev machine: sends tasks to the hub and shows the results.

    python3 gs_remote.py hub                                   # on the Mac, left running
    python3 gs_remote.py agents add win                        # prints the command to run on the PC
    py gs_remote.py agent --hub http://192.168.1.20:8765 --name win --token ...   # on the PC
    python3 gs_remote.py info all
    python3 gs_remote.py run all -- @python -c "import platform; print(platform.platform())"

Security: every agent request is signed (HMAC-SHA256) with its token, and every hub response
is signed with the same token and bound to the request's nonce. An agent therefore runs nothing that
does not come from the real hub, and an old task cannot be replayed. The control API (CLI) is
only reachable from the hub's machine. Traffic is not encrypted: trusted local network only.
"""

from __future__ import annotations

import argparse
import base64
import codecs
import collections
import fnmatch
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
import urllib.error
import urllib.request
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TOOL_VERSION = 7                # to increment at every change of the protocol or of the agent
EXIT_RESTART = 75                 # exit code of the agent worker: "restart me" (after an update)
HOME = Path(os.environ.get("GS_REMOTE_HOME", Path.home() / ".gpusync-remote"))
HUB_CONFIG = HOME / "hub.json"
AGENT_CONFIG = HOME / "agent.json"
DEFAULT_PORT = 8765
CTL = "@ctl"                      # identity of the CLI with the hub
POLL_WAIT_S = 25                  # long-polling: maximum wait for a task
ONLINE_TIMEOUT_S = 45             # an agent silent for longer is "offline"
HEARTBEAT_S = 5                   # an agent running a task shows signs of life at least every 5 s
LOST_GRACE_S = 600                # CLI: wait for an unreachable agent before giving up a task
ORPHAN_S = 60                     # task no longer followed by any CLI for 60 s → stopped (CLI killed, terminal closed)
UNKILLABLE_S = 15                 # process still alive 15 s after the forced stop → the agent gives up on it
MAX_CLOCK_SKEW_S = 300
MAX_BODY = 2 * 1024 ** 3
MAX_GET_FILE = 512 * 1024 ** 2
LOG_CHUNK_MAX = 1024 * 1024
SYNC_BATCH_BYTES = 200 * 1024 ** 2
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SYNC_STATE_FILE = ".gs_sync_state.json"
PY_PLACEHOLDER = "@python"
DEFAULT_EXCLUDES = [
    ".git", "target", "__pycache__", "*.pyc", ".venv", "venv", "node_modules", ".DS_Store",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "*.egg-info", "dist", "build", SYNC_STATE_FILE,
]
IGNORE_FILE = ".gsremoteignore"

_print_lock = threading.Lock()


def out(text: str, stream=None) -> None:
    s = stream or sys.stdout
    with _print_lock:
        s.write(text)
        s.flush()


def log(msg: str) -> None:
    out(f"[{time.strftime('%H:%M:%S')}] {msg}\n", sys.stderr)


# ═════════════════════════════════════════════════════════════════════════════
# Commun : signatures, chemins, config
# ═════════════════════════════════════════════════════════════════════════════


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class RemoteError(Exception):
    pass


class Unreachable(RemoteError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mac(token: str, *parts: str) -> str:
    return hmac.new(token.encode(), "\n".join(parts).encode(), hashlib.sha256).hexdigest()


def sign_request(token, who, method, path, ts, nonce, body: bytes) -> str:
    return _mac(token, "req", who, method.upper(), path, ts, nonce, sha256_hex(body))


def sign_response(token, nonce, status: int, body: bytes) -> str:
    return _mac(token, "resp", nonce, str(status), sha256_hex(body))


def check_name(name: str, what: str = "name") -> str:
    if not NAME_RE.match(name or ""):
        raise HttpError(400, f"invalid {what}: {name!r} (letters, digits, - and _ only)")
    return name


def safe_rel_path(p: str) -> str:
    """Normalized relative path; rejects absolute paths, `..`, drive letters."""
    if not isinstance(p, str) or not p:
        raise HttpError(400, "empty path")
    p = p.replace("\\", "/")
    if p.startswith("/") or ":" in p or "\x00" in p:
        raise HttpError(400, f"path rejected: {p!r}")
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if not parts or ".." in parts:
        raise HttpError(400, f"path rejected: {p!r}")
    return "/".join(parts)


def resolve_inside(base: Path, rel: str) -> Path:
    """`base/rel`, refusing that it leaves `base` (including through a symbolic link)."""
    target = base / safe_rel_path(rel)
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        raise HttpError(400, f"path rejected (leaves the workspace): {rel!r}") from None
    return target


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text("utf-8"))
    except FileNotFoundError:
        return default


def write_private_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), "utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def local_ips() -> list[str]:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:  # default interface (UDP connect: no packet sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


class SignedClient:
    """HTTP client that signs its requests and only accepts responses signed by the hub."""

    def __init__(self, base_url: str, who: str, token: str):
        self.base, self.who, self.token = base_url.rstrip("/"), who, token

    def call(self, method: str, path: str, payload=None, body: bytes | None = None, timeout: float = 60, raw=False):
        if body is None:
            body = json.dumps(payload).encode() if payload is not None else b""
        ts, nonce = f"{time.time():.3f}", secrets.token_hex(16)
        headers = {
            "X-GS-Who": self.who, "X-GS-Time": ts, "X-GS-Nonce": nonce,
            "X-GS-Sig": sign_request(self.token, self.who, method, path, ts, nonce, body),
            "Content-Type": "application/octet-stream",
        }
        req = urllib.request.Request(self.base + path, data=body if method == "POST" else None, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status, data, rsig = resp.status, resp.read(), resp.headers.get("X-GS-Resp-Sig")
        except urllib.error.HTTPError as e:
            status, data, rsig = e.code, e.read(), e.headers.get("X-GS-Resp-Sig")
        except (urllib.error.URLError, OSError) as e:
            raise Unreachable(f"{self.base} injoignable — {getattr(e, 'reason', e)}") from None
        authentic = bool(rsig) and hmac.compare_digest(rsig, sign_response(self.token, nonce, status, data))
        if status != 200:
            try:
                msg = json.loads(data).get("error", "")
            except ValueError:
                msg = data[:200].decode("utf-8", "replace")
            suffix = "" if authentic else " [unauthenticated response: wrong token, or this is not the right hub]"
            raise RemoteError(f"HTTP {status} — {msg}{suffix}")
        if not authentic:
            raise RemoteError("UNAUTHENTICATED response: this server is not the expected hub (or the token is wrong). Nothing was run.")
        return data if raw else json.loads(data)


# ═════════════════════════════════════════════════════════════════════════════
# HUB (machine de dev)
# ═════════════════════════════════════════════════════════════════════════════


class HubJob:
    def __init__(self, hub: "Hub", agent: str, spec: dict, src: Path | None):
        self.id = uuid.uuid4().hex[:12]
        self.agent = agent
        self.spec = dict(spec, id=self.id)
        self.src = src
        self.manifest: dict = spec.get("manifest") or {}
        self.status = "queued"
        self.exit_code: int | None = None
        self.created = time.time()
        self.started: float | None = None
        self.ended: float | None = None
        self.kill_requested = False
        self.exec_s: float | None = None  # duration measured on the test machine
        self.last_watch = time.time()  # last time a CLI read this task's output
        self.result: bytes | None = None
        self.log_path = hub.jobs_dir / f"{self.id}.log"
        self.log_path.write_bytes(b"")

    def describe(self, hub: "Hub") -> dict:
        status = self.status
        if status == "running" and time.time() - hub.agents[self.agent].last_seen > ONLINE_TIMEOUT_S:
            status = "lost"  # the agent no longer shows signs of life
        ref = self.started or self.created
        return {"id": self.id, "agent": self.agent, "type": self.spec["type"], "status": status,
                "exit_code": self.exit_code, "duration_s": round((self.ended or time.time()) - ref, 3),
                "exec_s": self.exec_s}


class AgentState:
    def __init__(self, name: str):
        self.name = name
        self.last_seen = 0.0
        self.info: dict = {}
        self.queue: collections.deque = collections.deque()
        self.current: str | None = None
        self.address = ""


class Hub:
    def __init__(self, allow: list, orphan_s: float = ORPHAN_S):
        self.allow = allow
        self.orphan_s = orphan_s
        self.jobs_dir = HOME / "hub" / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.cond = threading.Condition()
        self.agents: dict[str, AgentState] = {}
        self.jobs: dict[str, HubJob] = {}
        self.nonces: dict[str, float] = {}

    # --- configuration (read again at every request: `agents add` needs no restart) ---

    def config(self) -> dict:
        return read_json(HUB_CONFIG, {})

    def token_for(self, who: str) -> str | None:
        cfg = self.config()
        if who == CTL:
            return cfg.get("control_token")
        return (cfg.get("agents") or {}).get(who, {}).get("token")

    def agent(self, name: str) -> AgentState:
        st = self.agents.get(name)
        if st is None:
            st = self.agents[name] = AgentState(name)
        return st

    # --- access control ---

    def ip_allowed(self, addr: str) -> bool:
        ip = ipaddress.ip_address(addr)
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if ip.is_loopback:
            return True
        if self.allow:
            return any(ip in net for net in self.allow)
        return ip.is_private or ip.is_link_local

    @staticmethod
    def is_loopback(addr: str) -> bool:
        ip = ipaddress.ip_address(addr)
        return (ip.ipv4_mapped or ip).is_loopback if ip.version == 6 else ip.is_loopback

    def verify(self, token: str, who: str, method: str, path: str, headers, body: bytes) -> None:
        ts, nonce, sig = headers.get("X-GS-Time"), headers.get("X-GS-Nonce"), headers.get("X-GS-Sig")
        if not (ts and nonce and sig):
            raise HttpError(401, "unsigned request")
        try:
            skew = abs(time.time() - float(ts))
        except ValueError:
            raise HttpError(401, "invalid timestamp") from None
        if skew > MAX_CLOCK_SKEW_S:
            raise HttpError(401, f"clocks out of sync ({skew:.0f} s apart)")
        if not hmac.compare_digest(sign_request(token, who, method, path, ts, nonce, body), sig):
            raise HttpError(401, "invalid signature (wrong token?)")
        now = time.time()
        with self.cond:
            for n, t in list(self.nonces.items()):
                if now - t > 2 * MAX_CLOCK_SKEW_S:
                    del self.nonces[n]
            if nonce in self.nonces:
                raise HttpError(401, "replayed request")
            self.nonces[nonce] = now

    # --- API agents ---

    def job_of(self, agent: str, job_id: str) -> HubJob:
        job = self.jobs.get(job_id)
        if job is None or job.agent != agent:
            raise HttpError(404, "unknown task")
        return job

    def a_hello(self, agent: str, req: dict, address: str) -> dict:
        with self.cond:
            st = self.agent(agent)
            st.info, st.last_seen, st.address = req.get("info") or {}, time.time(), address
            for job in self.jobs.values():  # the agent restarted: its running tasks are lost
                if job.agent == agent and job.status == "running" and job.spec["type"] != "update":
                    job.status, job.ended = "lost", time.time()
            st.current = None
        version = st.info.get("tool_version")
        warn = "" if version == TOOL_VERSION else f" — version {version} ≠ hub {TOOL_VERSION}: gs_remote.py update {agent}"
        log(f"agent {agent} connected from {address}{warn}")
        return {"ok": True}

    def a_poll(self, agent: str) -> dict:
        with self.cond:
            st = self.agent(agent)
            deadline = time.time() + POLL_WAIT_S
            while not st.queue:
                st.last_seen = time.time()
                remaining = deadline - time.time()
                if remaining <= 0:
                    return {"job": None}
                self.cond.wait(timeout=min(remaining, 5))
            job = st.queue.popleft()
            job.status, job.started, st.current, st.last_seen = "running", time.time(), job.id, time.time()
            return {"job": job.spec}

    def a_files(self, agent: str, req: dict) -> bytes:
        job = self.job_of(agent, req.get("job", ""))
        if job.src is None:
            raise HttpError(400, "this task has no files")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for rel in req.get("paths") or []:
                if rel not in job.manifest:  # the agent may ONLY request the task's files
                    raise HttpError(403, f"file outside the task: {rel!r}")
                zf.write(resolve_inside(job.src, rel), rel)
        self.agent(agent).last_seen = time.time()
        return buf.getvalue()

    def a_output(self, agent: str, req: dict) -> dict:
        job = self.job_of(agent, req.get("job", ""))
        data = base64.b64decode(req.get("data_b64") or "")
        if data:
            with open(job.log_path, "ab") as f:
                f.write(data)
        self.agent(agent).last_seen = time.time()
        if not job.kill_requested and time.time() - job.last_watch > self.orphan_s:
            job.kill_requested = True
            log(f"task {job.id} ({agent}): no longer followed by any CLI for {self.orphan_s:.0f} s → stopping")
        return {"kill": job.kill_requested}

    def a_upload(self, agent: str, job_id: str, body: bytes) -> dict:
        job = self.job_of(agent, job_id)
        job.result = body
        return {"ok": True}

    def a_done(self, agent: str, req: dict) -> dict:
        job = self.job_of(agent, req.get("job", ""))
        tail = base64.b64decode(req.get("data_b64") or "")
        if tail:  # written BEFORE switching to the final status: a final status implies a complete log
            with open(job.log_path, "ab") as f:
                f.write(tail)
        with self.cond:
            job.status = req.get("status") or "done"
            job.exit_code = req.get("exit_code")
            job.exec_s = req.get("exec_s")
            job.ended = time.time()
            st = self.agent(agent)
            st.current, st.last_seen = None, time.time()
            if req.get("info"):
                st.info = req["info"]
        return {"ok": True}

    # --- control API (CLI, local machine only) ---

    def c_agents(self) -> dict:
        now = time.time()
        known = set((self.config().get("agents") or {}).keys())
        rows = []
        for name in sorted(known | set(self.agents)):
            st = self.agents.get(name)
            rows.append({
                "name": name,
                "registered": name in known,
                "online": bool(st and now - st.last_seen < ONLINE_TIMEOUT_S),
                "last_seen_s": round(now - st.last_seen, 1) if st and st.last_seen else None,
                "address": st.address if st else "",
                "info": st.info if st else {},
                "current": st.current if st else None,
                "queued": len(st.queue) if st else 0,
            })
        return {"agents": rows}

    def c_submit(self, req: dict) -> dict:
        agent = check_name(req.get("agent", ""), "agent")
        if agent not in (self.config().get("agents") or {}):
            raise HttpError(404, f"unknown agent: {agent}")
        st = self.agents.get(agent)
        if not (st and time.time() - st.last_seen < ONLINE_TIMEOUT_S):
            raise HttpError(409, f"agent {agent} is offline")
        spec = req.get("spec") or {}
        if spec.get("type") not in ("run", "sync", "get", "reset", "update"):
            raise HttpError(400, "invalid task type")
        check_name(spec.get("ws", ""), "workspace")
        src = Path(req["src"]) if req.get("src") else None
        job = HubJob(self, agent, spec, src)
        with self.cond:
            self.jobs[job.id] = job
            st.queue.append(job)
            self.cond.notify_all()
        return job.describe(self)

    def c_job(self, job_id: str, offset: int) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise HttpError(404, f"unknown task: {job_id}")
        job.last_watch = time.time()
        d = job.describe(self)  # read BEFORE the log: if finished, the log is complete
        with open(job.log_path, "rb") as f:
            f.seek(offset)
            data = f.read(LOG_CHUNK_MAX)
        d.update({"next_offset": offset + len(data), "data_b64": base64.b64encode(data).decode()})
        if d["status"] not in ("queued", "running") and len(data) == LOG_CHUNK_MAX:
            d["status"] = "running"  # there is output left to read
        return d

    def c_jobs(self) -> dict:
        active = [j.describe(self) for j in self.jobs.values() if j.status in ("queued", "running")]
        return {"jobs": sorted(active, key=lambda d: d["agent"])}

    def c_kill(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise HttpError(404, f"unknown task: {job_id}")
        with self.cond:
            st = self.agent(job.agent)
            if job.status == "queued":
                try:
                    st.queue.remove(job)
                except ValueError:
                    pass
                job.status, job.ended = "killed", time.time()
            job.kill_requested = True
        return job.describe(self)

    def c_result(self, job_id: str) -> bytes:
        job = self.jobs.get(job_id)
        if job is None or job.result is None:
            raise HttpError(404, "no file for this task")
        return job.result


def make_hub_handler(hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "gs-remote-hub/2"

        def log_message(self, fmt, *args):
            pass  # the hub logs the useful events itself

        def _send(self, status: int, payload=None, raw: bytes | None = None, token=None, nonce=""):
            body = raw if raw is not None else json.dumps(payload or {}, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream" if raw is not None else "application/json")
            self.send_header("Content-Length", str(len(body)))
            if token:
                self.send_header("X-GS-Resp-Sig", sign_response(token, nonce, status, body))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self, method: str):
            nonce = self.headers.get("X-GS-Nonce", "")
            token = None
            addr = self.client_address[0]
            try:
                if not hub.ip_allowed(addr):
                    hint = f" — to allow it: gs_remote.py hub --allow {addr}/32"
                    ip = ipaddress.ip_address(addr)
                    if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):
                        hint += " (adresse Tailscale / CGNAT)"
                    log(f"rejected {addr}: IP address not allowed{hint}")
                    self._send(403, {"error": "IP address not allowed"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    raise HttpError(413, "request too large")
                body = self.rfile.read(length) if length else b""
                who = self.headers.get("X-GS-Who", "")
                if who == CTL and not hub.is_loopback(addr):
                    raise HttpError(403, "the control API is only reachable from the hub's machine")
                token = hub.token_for(who) if who == CTL or NAME_RE.match(who) else None
                if not token:
                    raise HttpError(401, f"unknown identity: {who!r}")
                hub.verify(token, who, method, self.path, self.headers, body)
                result = self._route(method, who, body, addr)
                if isinstance(result, bytes):
                    self._send(200, raw=result, token=token, nonce=nonce)
                else:
                    self._send(200, result, token=token, nonce=nonce)
            except HttpError as e:
                if e.status in (401, 403):
                    log(f"rejected {addr}: {e.message}")
                self._send(e.status, {"error": e.message}, token=token, nonce=nonce)
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"}, token=token, nonce=nonce)

        def _route(self, method, who, body, addr):
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            q = parse_qs(url.query)
            js = (lambda: json.loads(body or b"{}"))
            if who == CTL:
                if method == "GET" and parts == ["c", "agents"]:
                    return hub.c_agents()
                if method == "POST" and parts == ["c", "jobs"]:
                    return hub.c_submit(js())
                if method == "GET" and parts == ["c", "jobs"]:
                    return hub.c_jobs()
                if method == "GET" and len(parts) == 3 and parts[:2] == ["c", "jobs"]:
                    return hub.c_job(parts[2], int(q.get("offset", ["0"])[0]))
                if method == "POST" and len(parts) == 4 and parts[:2] == ["c", "jobs"] and parts[3] == "kill":
                    return hub.c_kill(parts[2])
                if method == "GET" and len(parts) == 4 and parts[:2] == ["c", "jobs"] and parts[3] == "result":
                    return hub.c_result(parts[2])
            else:
                if method == "POST" and parts == ["a", "hello"]:
                    return hub.a_hello(who, js(), addr)
                if method == "POST" and parts == ["a", "poll"]:
                    return hub.a_poll(who)
                if method == "POST" and parts == ["a", "files"]:
                    return hub.a_files(who, js())
                if method == "POST" and parts == ["a", "output"]:
                    return hub.a_output(who, js())
                if method == "POST" and parts == ["a", "upload"]:
                    return hub.a_upload(who, q.get("job", [""])[0], body)
                if method == "POST" and parts == ["a", "done"]:
                    return hub.a_done(who, js())
            raise HttpError(404, f"unknown route: {method} {url.path}")

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


def hub_config_ensure(port: int | None = None) -> dict:
    cfg = read_json(HUB_CONFIG, {})
    changed = False
    if "control_token" not in cfg:
        cfg["control_token"], changed = secrets.token_urlsafe(32), True
    cfg.setdefault("agents", {})
    if port is not None and cfg.get("port") != port:
        cfg["port"], changed = port, True
    if changed or not HUB_CONFIG.exists():
        write_private_json(HUB_CONFIG, cfg)
    return cfg


def cmd_hub(args) -> int:
    cfg = hub_config_ensure(args.port)
    allow = [ipaddress.ip_network(c, strict=False) for c in args.allow]
    hub = Hub(allow, args.orphan_timeout)
    httpd = ThreadingHTTPServer((args.bind, args.port), make_hub_handler(hub))
    httpd.daemon_threads = True
    print(f"gs-remote hub v{TOOL_VERSION} — {socket.gethostname()}, listening on {args.bind}:{args.port}")
    print(f"  allowed agents: {', '.join(args.allow) if args.allow else 'private and local IPs'}")
    print(f"  registered agents: {', '.join(cfg['agents']) or 'none'} (add: gs_remote.py agents add <name>)")
    print(f"  addresses of this hub: {', '.join(local_ips()) or '?'}\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nHub stopped.")
    return 0


def cmd_agents(args) -> int:
    cfg = hub_config_ensure()
    agents = cfg["agents"]
    port = cfg.get("port", DEFAULT_PORT)
    if args.action == "add":
        name = args.name or sys.exit("Usage: gs_remote.py agents add <name>")
        try:
            check_name(name, "agent name")
        except HttpError as e:
            sys.exit(e.message)
        if name in agents and not args.force:
            sys.exit(f"Agent {name} already exists (--force to regenerate its token).")
        token = secrets.token_urlsafe(32)
        agents[name] = {"token": token, "created": time.strftime("%Y-%m-%d %H:%M")}
        write_private_json(HUB_CONFIG, cfg)
        ip = (local_ips() or ["<Mac-IP>"])[0]
        print(f"Agent {name} registered. On the test machine, run (once with these options):\n")
        print(f"  Windows: py gs_remote.py agent --hub http://{ip}:{port} --name {name} --token {token}")
        print(f"  macOS   : python3 gs_remote.py agent --hub http://{ip}:{port} --name {name} --token {token}")
        print(f"  (agent on the hub's machine: --hub http://127.0.0.1:{port})\n")
        print("Afterwards, `gs_remote.py agent` is enough: the configuration is remembered on the test machine.")
        print("The token is secret: it allows running code through this hub.")
    elif args.action == "rm":
        if agents.pop(args.name or "", None) is None:
            sys.exit(f"Unknown agent: {args.name}")
        write_private_json(HUB_CONFIG, cfg)
        print(f"Agent {args.name} removed: its token is no longer accepted.")
    else:
        if not agents:
            print("No agent registered.")
        for n, a in agents.items():
            print(f"  {n:<12} created on {a.get('created', '?')}")
    return 0


# ═════════════════════════════════════════════════════════════════════════════
# AGENT (machines de test)
# ═════════════════════════════════════════════════════════════════════════════


def run_probe(cmd: list[str], timeout: float = 15) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout.decode("utf-8", "replace").strip() or None if r.returncode == 0 else None
    except Exception:
        return None


_STATIC_INFO: dict | None = None


def machine_info(ws_root: Path) -> dict:
    """Hardware info (computed once: some probes take ~1 s) + current workspaces."""
    global _STATIC_INFO
    if _STATIC_INFO is None:
        _STATIC_INFO = _probe_machine()
    info = dict(_STATIC_INFO)
    info["workspaces"] = sorted(p.name for p in ws_root.iterdir() if p.is_dir()) if ws_root.exists() else []
    return info


def _probe_machine() -> dict:
    system = platform.system()
    gpus: list[str] = []
    if system == "Windows":
        ps = run_probe(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' (driver ' + $_.DriverVersion + ')' }"], 30)
        gpus = ps.splitlines() if ps else []
    elif system == "Darwin":
        sp = run_probe(["system_profiler", "SPDisplaysDataType", "-json"], 30)
        try:
            for d in json.loads(sp or "{}").get("SPDisplaysDataType", []):
                cores = d.get("sppci_cores")
                gpus.append(d.get("sppci_model", "?") + (f" ({cores} cores)" if cores else ""))
        except ValueError:
            pass
    else:
        lspci = run_probe(["sh", "-c", "lspci | grep -Ei 'vga|3d|display'"])
        gpus = lspci.splitlines() if lspci else []
    nvidia = run_probe(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return {
        "tool_version": TOOL_VERSION,
        "hostname": socket.gethostname(),
        "os": f"{system} {platform.release()}",
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "gpus": gpus,
        "nvidia_smi": nvidia.splitlines() if nvidia else None,
        "rustc": run_probe(["rustc", "--version"]),
    }


class AgentJob:
    """Runs a task received from the hub and sends its output back as it goes."""

    def __init__(self, api: SignedClient, spec: dict, ws_root: Path):
        self.api, self.spec, self.ws_root = api, spec, ws_root
        self.id = spec["id"]
        self.ws = ws_root / check_name(spec["ws"], "workspace")
        self.final_text = ""  # end of output, sent with /a/done (saves a round trip)
        self.exec_s: float | None = None  # duration of the command, timed ON the test machine

    def _call(self, *args, **kwargs):
        """Call to the hub during a task: short timeout per attempt, new attempts for
        ~5 min. A network outage must not make a long task give up (seen: outages
        of more than 45 s between two continents through Tailscale)."""
        kwargs.setdefault("timeout", 20)
        deadline = time.time() + 300
        while True:
            try:
                return self.api.call(*args, **kwargs)
            except Unreachable:
                if time.time() > deadline:
                    raise
                time.sleep(5)

    def say(self, text: str) -> bool:
        """Sends output to the hub. Returns True if the hub asks to stop."""
        r = self._call("POST", "/a/output", {"job": self.id, "data_b64": base64.b64encode(text.encode()).decode()})
        return bool(r.get("kill"))

    def done(self, status: str, code: int | None) -> None:
        self._call("POST", "/a/done", {"job": self.id, "status": status, "exit_code": code, "exec_s": self.exec_s,
                                       "data_b64": base64.b64encode(self.final_text.encode()).decode(),
                                       "info": machine_info(self.ws_root)}, timeout=120)

    def execute(self) -> bool:
        """Runs the task. Returns True if the agent must restart (update installed)."""
        kind = self.spec["type"]
        try:
            if kind == "update":
                self.install_update()
                self.done("done", 0)
                return True
            if kind in ("run", "sync"):
                self.ws.mkdir(parents=True, exist_ok=True)
                if self.spec.get("manifest") is not None:
                    self.sync()
                if kind == "run":
                    status, code = self.run_command()
                    self.done(status, code)
                    return False
            elif kind == "get":
                target = resolve_inside(self.ws, self.spec["path"])
                if not target.is_file():
                    raise HttpError(404, f"file not found: {self.spec['path']}")
                if target.stat().st_size > MAX_GET_FILE:
                    raise HttpError(413, "file too large")
                self._call("POST", f"/a/upload?job={self.id}", body=target.read_bytes(), timeout=600)
            elif kind == "reset":
                if self.ws.exists():
                    shutil.rmtree(self.ws)
                self.say(f"workspace {self.ws.name} removed\n")
            self.done("done", 0)
        except HttpError as e:
            self.say(f"[gs-remote] error: {e.message}\n")
            self.done("error", -1)
        except RemoteError:
            raise
        except Exception as e:
            self.say(f"[gs-remote] error: {type(e).__name__}: {e}\n")
            self.done("error", -1)
        return False

    def install_update(self) -> None:
        """Replaces this script with the version sent by the hub (a task signed like the others)."""
        content = base64.b64decode(self.spec["content_b64"])
        if sha256_hex(content) != self.spec.get("sha256"):
            raise HttpError(400, "corrupted update (different hash)")
        compile(content, "gs_remote.py", "exec")  # rejects a file that would not start
        script = Path(__file__).resolve()
        tmp = script.with_name(script.name + ".new")
        tmp.write_bytes(content)
        os.replace(tmp, script)
        self.final_text = f"update installed in {script}: restarting the agent\n"

    def sync(self) -> None:
        manifest = {safe_rel_path(k): v for k, v in self.spec["manifest"].items()}
        state_path = self.ws / SYNC_STATE_FILE
        state = read_json(state_path, {})
        need = []
        for rel, (size, sha) in manifest.items():
            # Quick comparison (one stat). Paths are already validated by safe_rel_path;
            # resolve_inside (slower, especially on Windows) is only applied to writes.
            if state.get(rel) != sha:
                need.append(rel)
                continue
            try:
                if (self.ws / rel).stat().st_size != size:
                    need.append(rel)
            except OSError:
                need.append(rel)
        deleted = 0
        for rel in [r for r in state if r not in manifest]:
            try:
                resolve_inside(self.ws, rel).unlink()
                deleted += 1
            except (FileNotFoundError, HttpError):
                pass
            state.pop(rel, None)
        batch, batch_bytes, received = [], 0, 0
        for i, rel in enumerate(need):
            batch.append(rel)
            batch_bytes += manifest[rel][0]
            if batch_bytes >= SYNC_BATCH_BYTES or i == len(need) - 1:
                data = self._call("POST", "/a/files", {"job": self.id, "paths": batch}, timeout=900, raw=True)
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for member in zf.infolist():
                        if member.filename not in batch:
                            raise HttpError(400, f"file not requested in the archive: {member.filename!r}")
                        target = resolve_inside(self.ws, member.filename)
                        if target.is_symlink():
                            target.unlink()
                        target.parent.mkdir(parents=True, exist_ok=True)
                        content = zf.read(member)
                        target.write_bytes(content)
                        state[member.filename] = sha256_hex(content)
                        received += 1
                write_private_json(state_path, state)
                batch, batch_bytes = [], 0
        for rel in manifest:  # files already up to date
            state.setdefault(rel, manifest[rel][1])
        write_private_json(state_path, state)
        if need or deleted:
            self.say(f"[sync] {received} file(s) received, {deleted} removed\n")

    def run_command(self) -> tuple[str, int | None]:
        cmd, shell = self.spec.get("cmd"), bool(self.spec.get("shell"))
        if shell:
            if not isinstance(cmd, str):
                raise HttpError(400, "with --shell, the command must be a string")
            cmd = cmd.replace(PY_PLACEHOLDER, f'"{sys.executable}"')
        else:
            if not (isinstance(cmd, list) and cmd and all(isinstance(c, str) for c in cmd)):
                raise HttpError(400, "empty command")
            cmd = [sys.executable if c == PY_PLACEHOLDER else c for c in cmd]
        cwd = resolve_inside(self.ws, self.spec["cwd"]) if self.spec.get("cwd") else self.ws
        if not cwd.is_dir():
            raise HttpError(400, f"folder not found in the workspace: {self.spec.get('cwd')}")
        env = os.environ.copy()
        fresh_path = windows_registry_path()
        if fresh_path:  # tools installed after the agent started (e.g. rustup): visible without restarting it
            env["PATH"] = fresh_path
        env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        env.update({str(k): str(v) for k, v in (self.spec.get("env") or {}).items()})
        if not shell and os.name == "nt":
            if not os.path.dirname(cmd[0]):
                # On Windows, Popen looks for the executable on the AGENT's PATH, not in `env`:
                # resolve it ourselves with the task's PATH (e.g. cargo installed after the launch).
                found = shutil.which(cmd[0], path=env.get("PATH"))
                if found:
                    cmd = [found] + cmd[1:]
            elif not os.path.isabs(cmd[0]):
                # Relative path (.venv/Scripts/python.exe): Windows resolves it from the agent's
                # folder, not from `cwd` as macOS and Linux do.
                cmd = [str(cwd / cmd[0])] + cmd[1:]
        timeout = float(self.spec.get("timeout") or 600)

        log_path = self.ws_root.parent / "logs" / f"{self.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        with open(log_path, "wb") as logf:
            try:
                t0 = time.perf_counter()
                proc = subprocess.Popen(cmd, shell=shell, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                        stdout=logf, stderr=subprocess.STDOUT, **kwargs)
            except OSError as e:
                raise HttpError(400, f"cannot start the command: {e}") from None

        def _wait_exit():  # precise timing: exact instant the process ended, measured locally
            proc.wait()
            self.exec_s = time.perf_counter() - t0

        waiter = threading.Thread(target=_wait_exit, daemon=True)
        waiter.start()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        start = last_send = time.time()
        reason, kill_time = None, None
        try:
            with open(log_path, "rb") as reader:
                while True:
                    finished = proc.poll() is not None
                    if finished:
                        waiter.join(timeout=5)
                    chunk = reader.read(LOG_CHUNK_MAX)
                    if finished:
                        # The process has finished: all the remaining output goes with /a/done.
                        self.final_text = decoder.decode(chunk + reader.read(), final=True)
                        break
                    if chunk or time.time() - last_send > HEARTBEAT_S:
                        if self.say(decoder.decode(chunk)) and reason is None:
                            reason, kill_time = "killed", time.time()
                            self.say("\n[gs-remote] forced stop requested\n")
                            kill_tree(proc)
                        last_send = time.time()
                    if reason is None and time.time() - start > timeout:
                        reason, kill_time = "timeout", time.time()
                        self.say(f"\n[gs-remote] timeout ({timeout:.0f} s): stopping the command\n")
                        kill_tree(proc)
                    if kill_time and not finished and time.time() - kill_time > UNKILLABLE_S:
                        # Typically: a process stuck in a GPU driver call. Give up on it
                        # so that the agent stays available; the machine may need a restart.
                        reason = "unkillable"
                        self.say(f"\n[gs-remote] process {proc.pid} has not responded to the forced stop "
                                 f"for {UNKILLABLE_S} s (stuck in the driver?). Abandoned: "
                                 "the agent stays available, but the process may still be running.\n")
                        break
                    if not chunk:
                        time.sleep(0.3)
        finally:
            if proc.poll() is None and reason != "unkillable":
                kill_tree(proc)
            try:
                log_path.unlink()
            except OSError:
                pass
        return (reason or "done"), proc.returncode


def windows_registry_path() -> str | None:
    """PATH as a new Windows terminal would see it (system + user, read again from the registry)."""
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None
    parts = []
    for root, key in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                      (winreg.HKEY_CURRENT_USER, r"Environment")):
        try:
            with winreg.OpenKey(root, key) as k:
                value, _ = winreg.QueryValueEx(k, "Path")
                parts.append(os.path.expandvars(value))
        except OSError:
            pass
    return ";".join(p for p in parts if p) or None


def kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass  # run_command's loop watches what follows (gives up after UNKILLABLE_S)


def cmd_agent(args) -> int:
    cfg = read_json(AGENT_CONFIG, {})
    for key in ("hub", "name", "token"):
        if getattr(args, key):
            cfg[key] = getattr(args, key)
    missing = [k for k in ("hub", "name", "token") if not cfg.get(k)]
    if missing:
        sys.exit("First use: gs_remote.py agent --hub http://<hub-IP>:8765 --name <name> --token <token>\n"
                 "(command printed by `gs_remote.py agents add <name>` on the hub's machine)")
    write_private_json(AGENT_CONFIG, cfg)
    ws_root = Path(args.root).expanduser() if args.root else HOME / "agent" / "workspaces"
    ws_root.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        return agent_supervisor(cfg, args)
    return agent_worker(cfg, ws_root)


def agent_supervisor(cfg: dict, args) -> int:
    """Restarts the worker: after an update (new code) or a crash."""
    print(f"gs-remote agent v{TOOL_VERSION} « {cfg['name']} » → hub {cfg['hub']}", flush=True)
    print("This agent listens to no port: it connects to the hub and only runs signed tasks.\n", flush=True)
    script = str(Path(__file__).resolve())
    cmd = [sys.executable, script, "agent", "--worker"] + (["--root", args.root] if args.root else [])
    while True:
        try:
            rc = subprocess.call(cmd)
        except KeyboardInterrupt:
            print("\nAgent stopped.")
            return 0
        if rc == EXIT_RESTART:
            log("update installed: restarting the agent with the new code")
            continue
        if rc in (0, 130):
            return rc
        log(f"the agent stopped abnormally (code {rc}): restarting in 5 s")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nAgent stopped.")
            return 0


def agent_worker(cfg: dict, ws_root: Path) -> int:
    api = SignedClient(cfg["hub"], cfg["name"], cfg["token"])
    backoff, connected = 2.0, False
    while True:
        try:
            api.call("POST", "/a/hello", {"info": machine_info(ws_root)})
            if not connected:
                log("connected to the hub")
            connected, backoff = True, 2.0
            while True:
                r = api.call("POST", "/a/poll", {}, timeout=POLL_WAIT_S + 20)
                spec = r.get("job")
                if spec:
                    detail = spec.get("cmd") or spec.get("path") or ""
                    log(f"task {spec['id']}: {spec['type']} {spec.get('ws', '')} {detail}")
                    restart = AgentJob(api, spec, ws_root).execute()
                    log(f"task {spec['id']} finished")
                    if restart:
                        return EXIT_RESTART
        except KeyboardInterrupt:
            print("\nAgent stopped.")
            return 0
        except Unreachable as e:
            if connected:
                log(f"connection to the hub lost ({e}). Trying again…")
            elif backoff == 2.0:
                log(f"hub unreachable ({e}). Trying again regularly…")
            connected = False
        except RemoteError as e:
            log(f"rejected by the hub: {e}")
            connected = False
            backoff = max(backoff, 30.0)
        try:
            time.sleep(backoff)
        except KeyboardInterrupt:
            print("\nAgent stopped.")
            return 0
        backoff = min(backoff * 2, 30.0)


# ══════════════════════════════════════════════════════════════════════════
# CLI (hub machine)
# ══════════════════════════════════════════════════════════════════════════


def ctl() -> SignedClient:
    cfg = read_json(HUB_CONFIG, {})
    if "control_token" not in cfg:
        sys.exit("No hub configured on this machine. Start one first: gs_remote.py hub")
    return SignedClient(f"http://127.0.0.1:{cfg.get('port', DEFAULT_PORT)}", CTL, cfg["control_token"])


def list_agents(api: SignedClient) -> list[dict]:
    try:
        return api.call("GET", "/c/agents")["agents"]
    except Unreachable:
        sys.exit("The hub does not answer. Start it in another terminal: gs_remote.py hub")


def resolve_targets(api: SignedClient, spec: str) -> tuple[list[str], list[str]]:
    """Returns (targeted online agents, targeted offline agents)."""
    rows = {a["name"]: a for a in list_agents(api) if a["registered"]}
    if not rows:
        sys.exit("No agent registered. Add one: gs_remote.py agents add <name>")
    names = list(rows) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in rows]
    if unknown:
        sys.exit(f"Unknown agent(s): {', '.join(unknown)}. Registered: {', '.join(rows)}")
    online = [n for n in names if rows[n]["online"]]
    offline = [n for n in names if not rows[n]["online"]]
    if not online:
        sys.exit(f"No online agent among: {', '.join(names)}. Run `gs_remote.py agent` on the test machine.")
    return online, offline


def default_src() -> Path:
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return Path.cwd()


def load_excludes(src: Path, extra: list[str]) -> list[str]:
    patterns = list(DEFAULT_EXCLUDES) + list(extra)
    ignore = src / IGNORE_FILE
    if ignore.exists():
        patterns += [ln.strip().rstrip("/") for ln in ignore.read_text("utf-8").splitlines()
                     if ln.strip() and not ln.strip().startswith("#")]
    return patterns


def is_excluded(rel: str, patterns: list[str]) -> bool:
    parts = rel.split("/")
    for pat in patterns:
        if "/" in pat:
            p = pat.lstrip("/")
            if fnmatch.fnmatch(rel, p) or rel.startswith(p + "/"):
                return True
        elif any(fnmatch.fnmatch(part, pat) for part in parts):
            return True
    return False


def build_manifest(src: Path, patterns: list[str]) -> dict[str, list]:
    files: dict[str, list] = {}
    for dirpath, dirnames, filenames in os.walk(src):
        rel_dir = Path(dirpath).relative_to(src).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = [d for d in dirnames if not is_excluded(rel_dir + d, patterns)]
        for fn in filenames:
            rel, full = rel_dir + fn, Path(dirpath) / fn
            if is_excluded(rel, patterns) or not full.is_file() or full.is_symlink():
                continue
            h = hashlib.sha256()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            files[rel] = [full.stat().st_size, h.hexdigest()]
    return files


class Follow:
    """Submits a task to an agent and follows its output."""

    def __init__(self, api: SignedClient, agent: str, prefix: bool):
        self.api, self.agent, self.prefix = api, agent, prefix
        self.job_id: str | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self.ends_with_newline = True

    def emit(self, text: str, final=False) -> None:
        if not self.prefix:
            if text:
                out(text)
                self.ends_with_newline = text.endswith("\n")
            return
        lines = (self._pending + text).split("\n")
        last = lines.pop()
        if final and last:
            lines.append(last)
            last = ""
        self._pending = last
        for line in lines:
            out(f"[{self.agent}] {line.rstrip(chr(13))}\n")

    def run(self, spec: dict, src: Path | None) -> None:
        try:
            job = self.api.call("POST", "/c/jobs", {"agent": self.agent, "spec": spec, "src": str(src) if src else None})
            self.job_id, offset, delay = job["id"], 0, 0.2
            lost_since, warned_lost = None, False
            while True:
                st = self.api.call("GET", f"/c/jobs/{self.job_id}?offset={offset}")
                data = base64.b64decode(st["data_b64"])
                offset = st["next_offset"]
                if data:
                    self.emit(self._dec.decode(data))
                    delay = 0.2
                if st["status"] == "lost":
                    # Agent temporarily unreachable: keep waiting (the task may still be
                    # running); give up only after LOST_GRACE_S without news.
                    lost_since = lost_since or time.time()
                    if not warned_lost:
                        out(f"[{self.agent}] agent unreachable, waiting for it to come back…\n", sys.stderr)
                        warned_lost = True
                    if time.time() - lost_since < LOST_GRACE_S:
                        time.sleep(2)
                        continue
                else:
                    lost_since = None
                if st["status"] not in ("queued", "running"):
                    self.emit(self._dec.decode(b"", final=True), final=True)
                    self.result = st
                    return
                if not data:
                    time.sleep(delay)
                    delay = min(delay * 1.5, 1.0)
        except RemoteError as e:
            self.error = str(e)
            out(f"[{self.agent}] ERROR: {e}\n", sys.stderr)

    def kill(self) -> None:
        if self.job_id and self.result is None:
            try:
                self.api.call("POST", f"/c/jobs/{self.job_id}/kill")
            except RemoteError:
                pass


def fmt_timing(st: dict) -> str:
    """Run time measured on the test machine, then total time seen from the hub (network included)."""
    total = f"total {st.get('duration_s', 0):.2f} s"
    if st.get("exec_s") is None:
        return total
    return f"run {st['exec_s']:.3f} s on site · {total}"


def run_on_targets(api: SignedClient, target: str, spec: dict, src: Path | None, summary_always=False) -> tuple[int, list]:
    online, offline = resolve_targets(api, target)
    for n in offline:
        out(f"[{n}] offline: skipped\n", sys.stderr)
    multi = len(online) > 1
    follows = [Follow(api, n, prefix=multi) for n in online]
    threads = [threading.Thread(target=f.run, args=(spec, src), daemon=True) for f in follows]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        out("\nInterrupted: stopping the remote tasks…\n", sys.stderr)
        for f in follows:
            f.kill()
        return 130, follows

    worst = 0
    if not multi and follows and not follows[0].ends_with_newline:
        out("\n")  # the remote output did not end with a newline
    if multi or offline or summary_always:
        out("─" * 60 + "\n", sys.stderr)
    for f in follows:
        st = f.result or {}
        code = st.get("exit_code")
        ok = not f.error and st.get("status") == "done" and code == 0
        if not ok:
            worst = worst or (code if isinstance(code, int) and code > 0 else 1)
        timing = fmt_timing(st)
        if multi or offline or summary_always:
            label = "OK" if ok else ("ERROR" if f.error else str(st.get("status", "?")).upper())
            out(f"  {f.agent:<12} {label:<9} code {str(code):<5} {timing}\n", sys.stderr)
        elif not f.error:
            state = "" if ok else f"task finished: {st.get('status')} (code {code}) · "
            if state or st.get("exec_s") is not None:
                out(f"[{f.agent}] {state}{timing}\n", sys.stderr)
    for n in offline:
        out(f"  {n:<12} OFFLINE\n", sys.stderr)
        worst = worst or 1
    return worst, follows


def sync_spec(args) -> tuple[Path, str, dict | None]:
    src = Path(args.src).resolve() if args.src else default_src()
    ws = args.ws or re.sub(r"[^A-Za-z0-9_-]", "_", src.name)
    manifest = None if getattr(args, "no_sync", False) else build_manifest(src, load_excludes(src, args.exclude))
    return src, ws, manifest


def cmd_run(args) -> int:
    if not args.command:
        sys.exit("Command missing after `--`. Example: gs_remote.py run all -- @python script.py")
    src, ws, manifest = sync_spec(args)
    spec = {
        "type": "run", "ws": ws, "manifest": manifest,
        "cmd": " ".join(args.command) if args.shell else args.command, "shell": args.shell,
        "timeout": args.timeout, "cwd": args.cwd,
        "env": dict(kv.split("=", 1) for kv in args.env),
    }
    code, _ = run_on_targets(ctl(), args.target, spec, src if manifest is not None else None)
    return code


def cmd_sync(args) -> int:
    src, ws, manifest = sync_spec(args)
    code, _ = run_on_targets(ctl(), args.target, {"type": "sync", "ws": ws, "manifest": manifest}, src, summary_always=True)
    return code


def cmd_get(args) -> int:
    api = ctl()
    if args.target == "all" or "," in args.target:
        sys.exit("get: one target at a time")
    ws = args.ws or re.sub(r"[^A-Za-z0-9_-]", "_", default_src().name)
    code, follows = run_on_targets(api, args.target, {"type": "get", "ws": ws, "path": args.remote_path}, None)
    if len(follows) != 1:
        sys.exit("get: one target at a time")
    f = follows[0]
    if code == 0 and f.job_id:
        data = api.call("GET", f"/c/jobs/{f.job_id}/result", raw=True)
        dest = Path(args.local or Path(args.remote_path.replace("\\", "/")).name)
        dest.write_bytes(data)
        print(f"[{f.agent}] {args.remote_path} → {dest} ({len(data)} octets)")
    return code


def cmd_reset(args) -> int:
    ws = args.ws or re.sub(r"[^A-Za-z0-9_-]", "_", default_src().name)
    code, _ = run_on_targets(ctl(), args.target, {"type": "reset", "ws": ws}, None)
    return code


def cmd_update(args) -> int:
    content = Path(__file__).resolve().read_bytes()
    spec = {"type": "update", "ws": "gs-update", "content_b64": base64.b64encode(content).decode(),
            "sha256": sha256_hex(content)}
    code, _ = run_on_targets(ctl(), args.target, spec, None, summary_always=True)
    if code == 0:
        print(f"Agents updated to v{TOOL_VERSION}; they restart and reconnect within a few seconds.")
    return code


def cmd_jobs(args) -> int:
    jobs = ctl().call("GET", "/c/jobs")["jobs"]
    if not jobs:
        print("No task running.")
    for j in jobs:
        print(f"  {j['id']}  {j['agent']:<10} {j['type']:<6} {j['status']:<8} {j['duration_s']:.0f} s")
    return 0


def cmd_kill(args) -> int:
    api = ctl()
    jobs = api.call("GET", "/c/jobs")["jobs"]
    if args.target in {j["id"] for j in jobs}:
        victims = [j for j in jobs if j["id"] == args.target]
    else:
        agents = None if args.target == "all" else {s.strip() for s in args.target.split(",")}
        victims = [j for j in jobs if agents is None or j["agent"] in agents]
    if not victims:
        print(f"No task running for \"{args.target}\".")
        return 0
    for j in victims:
        api.call("POST", f"/c/jobs/{j['id']}/kill")
        print(f"  stop requested: {j['id']} on {j['agent']} ({j['type']})")
    print("The agent stops it within 5 s at most (check with: gs_remote.py jobs).")
    return 0


def cmd_info(args) -> int:
    rows = [a for a in list_agents(ctl()) if a["registered"]]
    if args.target != "all":
        wanted = {s.strip() for s in args.target.split(",")}
        rows = [a for a in rows if a["name"] in wanted]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    if not rows:
        print("No matching agent.")
    for a in rows:
        i = a["info"] or {}
        state = "ONLINE" if a["online"] else "offline"
        busy = f", task {a['current']} running" if a.get("current") else ""
        print(f"[{a['name']}] {state}{busy}" + (f" — {a['address']}" if a["address"] else ""))
        if not i:
            continue
        v = i.get("tool_version")
        if v != TOOL_VERSION:
            print(f"    outil   : v{v or '1-2'} ≠ v{TOOL_VERSION} → gs_remote.py update {a['name']}")
        print(f"    machine: {i.get('hostname')} — {i.get('os')} ({i.get('machine')}), {i.get('cpu_count')} CPU cores")
        print(f"    python  : {i.get('python')}  ({i.get('python_executable')})")
        print(f"    rust    : {i.get('rustc') or 'missing'}")
        for g in i.get("gpus") or ["(no GPU detected)"]:
            print(f"    gpu     : {g}")
        for g in i.get("nvidia_smi") or []:
            print(f"    nvidia  : {g}")
        if i.get("workspaces"):
            print(f"    workspaces: {', '.join(i['workspaces'])}")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description="gs-remote: hub, agents and CLI to test GPUsync on several machines.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("hub", help="start the hub (dev machine)")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--allow", action="append", default=[], metavar="CIDR",
                   help="networks from which agents may connect (repeatable). Default: private IPs")
    p.add_argument("--orphan-timeout", type=float, default=ORPHAN_S, metavar="S",
                   help=f"stop a task no CLI has followed for S seconds (default {ORPHAN_S})")
    p.set_defaults(func=cmd_hub)

    p = sub.add_parser("agents", help="register / remove / list the agents (hub machine)")
    p.add_argument("action", choices=["add", "rm", "list"])
    p.add_argument("name", nargs="?")
    p.add_argument("--force", action="store_true", help="with add: regenerate the token of an existing agent")
    p.set_defaults(func=cmd_agents)

    p = sub.add_parser("agent", help="lancer l'agent (machine de test)")
    p.add_argument("--hub", help="hub URL, e.g. http://192.168.1.20:8765 (remembered)")
    p.add_argument("--name", help="name of this agent (remembered)")
    p.add_argument("--token", help="token of this agent (remembered)")
    p.add_argument("--root", help="workspaces folder")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)  # usage interne (superviseur)
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("update", help="install this version of gs_remote.py on agents (they restart by themselves)")
    p.add_argument("target")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("info", help="status and hardware of the agents")
    p.add_argument("target", nargs="?", default="all")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_info)

    def sync_opts(p):
        p.add_argument("--src", help="folder to synchronize (default: git root or current folder)")
        p.add_argument("--ws", help="name of the remote workspace (default: name of the folder)")
        p.add_argument("--exclude", action="append", default=[], help="pattern to exclude (repeatable)")

    p = sub.add_parser("sync", help="synchronize the folder without running anything")
    p.add_argument("target")
    sync_opts(p)
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("run", help="synchronize, then run the command placed after --")
    p.add_argument("target")
    sync_opts(p)
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--timeout", type=float, default=600, help="seconds before stopping the command (default 600)")
    p.add_argument("--env", action="append", default=[], metavar="K=V")
    p.add_argument("--cwd", help="subfolder of the workspace to run in")
    p.add_argument("--shell", action="store_true", help="go through the remote shell (cmd.exe / sh)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("get", help="fetch a file from the remote workspace")
    p.add_argument("target")
    p.add_argument("remote_path")
    p.add_argument("local", nargs="?")
    p.add_argument("--ws")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("reset", help="remove the remote workspace")
    p.add_argument("target")
    p.add_argument("--ws")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("jobs", help="queued or running tasks")
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("kill", help="stop tasks: a task id, an agent, a list or all")
    p.add_argument("target")
    p.set_defaults(func=cmd_kill)

    argv, command = sys.argv[1:], []
    if "--" in argv:  # everything after `--` is the remote command, passed as is
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    args.command = command
    if args.cmd not in ("hub", "agent") and hasattr(signal, "SIGTERM"):
        # CLI stopped by SIGTERM (kill, task manager): same effect as Ctrl+C,
        # the followed remote tasks are stopped.
        def _term(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)
    try:
        sys.exit(args.func(args))
    except RemoteError as e:
        sys.exit(f"Error: {e}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

# gs-remote — remote execution on the test machines

Internal tool used to develop GPUsync while checking every step on real hardware.

```
                 ┌───────────── Mac (dev) ─────────────┐
                 │  CLI ──(localhost)──► HUB :8765      │
                 └──────────────────────▲──────▲───────┘
                   outgoing connection │      │ outgoing connection
                ┌───────────────────────┴┐   ┌─┴──────────────────────┐
                │ agent "mac"            │   │ agent "win"             │
                │ Apple M5 Pro           │   │ NVIDIA + integrated GPU │
                │ (same Mac, localhost)  │   │ (Windows PC)            │
                └────────────────────────┘   └─────────────────────────┘
```

- The **hub** runs on the Mac. It is the only program that listens on a port.
- The **agents** run on the test machines. **They listen on no port**: they connect to the hub, wait for their
  tasks, and reconnect by themselves if the hub restarts or the network drops.
- The **CLI** (on the Mac) sends tasks to the hub, shows their output live and returns the exit code.

Everything is in a single file, `gs_remote.py`, which only uses the Python standard library (≥ 3.9).

---

## 1. Setup

### On the Mac: the hub

```bash
caffeinate -i python3 tools/remote/gs_remote.py hub
```

Leave it running in a terminal. `caffeinate -i` prevents sleep. On the first launch, macOS asks to allow incoming
connections for Python: accept (this is what lets the PC reach the hub).

### On the Mac: register the agents

In another terminal, once per test machine:

```bash
python3 tools/remote/gs_remote.py agents add mac
python3 tools/remote/gs_remote.py agents add win
```

Each command creates a secret token for the agent and prints the exact command to run on the test machine.

### On the Mac: the local agent

With the command printed by `agents add mac`, replacing the IP with `127.0.0.1`:

```bash
python3 tools/remote/gs_remote.py agent --hub http://127.0.0.1:8765 --name mac --token <token>
```

### On the Windows PC: the agent

1. Install Python from python.org, with the "Add python.exe to PATH" box ticked (not the Microsoft Store alias).
2. Copy `gs_remote.py` to the PC (USB key, network share…).
3. Run the command printed by `agents add win`:

   ```powershell
   py gs_remote.py agent --hub http://192.168.1.113:8765 --name win --token <token>
   ```

**No firewall rule to create on the PC**: the agent only makes outgoing connections.

#### Through Tailscale

If the machines are linked by Tailscale, use the Mac's Tailscale address (`100.x.y.z`) in `--hub`, and allow the PC's
Tailscale address on the hub (by default, only private local-network IPs are accepted):

```bash
caffeinate -i python3 tools/remote/gs_remote.py hub --allow 100.90.141.102/32
```

Benefit: Tailscale encrypts all traffic between the machines (WireGuard), which lifts the "unencrypted traffic"
limit of the Security section. If the hub rejects a connection, its log prints the `--allow` command to use.

Afterwards, `py gs_remote.py agent` is enough: the hub URL, the name and the token are remembered in
`%USERPROFILE%\.gpusync-remote\agent.json`.

To build GPUsync, the PC also needs `rustup` and Visual Studio Build Tools (C++ workload), the NVIDIA driver (which
provides CUDA and OpenCL) and the Intel driver of the integrated GPU (which provides OpenCL). `gs_remote.py info`
shows what is already installed.

---

## 2. Usage (on the Mac)

```bash
R="python3 tools/remote/gs_remote.py"

$R info                                       # online agents, OS, Python, Rust, GPUs
$R run all -- @python -c "import platform; print(platform.platform())"
$R run win --timeout 1800 -- cargo test       # 30 min max
$R run all --cwd GPUsync-python -- @python -m pytest tests -q
$R run win --env RUST_LOG=debug -- cargo run --example bench
$R run win --shell -- "nvidia-smi && where python"
$R get win bench/results.json                 # fetch a file produced remotely
$R sync all                                   # synchronize without running anything
$R reset win                                  # remove the remote workspace
$R update all                                 # install this version of the tool on the agents
$R jobs                                       # queued or running tasks
$R kill win                                   # stop win's tasks (or a task id, or all)
$R agents list                                # registered agents
$R agents rm win                              # revoke an agent (its token is no longer accepted)
```

- **Target**: a name, a list (`mac,win`) or `all`. With several targets, the commands run in parallel, each line is
  prefixed with `[name]`, and a summary shows status, code and duration. With `all`, offline agents are reported
  and skipped.
- **`--`** separates the options from the remote command, which is passed as is.
- **`@python`** is replaced with the agent's Python interpreter (avoids the `python` / `py` / `python3` differences).
- **Exit code**: the one of the remote command (1 if a machine failed with several targets). `Ctrl+C` stops the
  remote tasks.
- Remote Python output is forced to UTF-8 and unbuffered.
- On Windows, `PATH` is read again from the registry before each task, as a new terminal would: a tool installed
  after the agent started (Rust, CUDA…) can be used without restarting it.

### Timing

The duration of each command is **measured on the test machine** (from its start to its end), then sent back:

```
[win] run 1.521 s on site · total 2.23 s
```

- **run … on site**: the real time of the command, without the network. This is the number that matters.
- **total**: seen from the hub, network included. With the Windows PC on the other side of the world (~340 ms per
  round trip), the difference can reach a second.

For a benchmark, also time *inside* the script (`time.perf_counter()`, `std::time::Instant`): these measurements run
on the test machine and exclude Python start-up and compilation.

### Updating the agents

After changing `gs_remote.py` on the Mac:

1. if the hub part changed, restart the hub (`Ctrl+C`, then the same command);
2. `gs_remote.py update all`: each agent receives the new version (a task signed like the others), checks its hash and
   its syntax, replaces its own file and restarts by itself.

`gs_remote.py info` reports agents whose version differs from the hub's. This relies on a small supervisor:
`gs_remote.py agent` runs the agent in a subprocess and restarts it after an update or a crash.

### Stopping a stuck script

| Situation | What happens |
|---|---|
| `Ctrl+C` or `SIGTERM` on the CLI | the followed tasks are stopped immediately |
| `gs_remote.py kill <agent \| id \| all>` | forced stop, from any terminal on the Mac |
| the CLI is killed or its terminal closed | the hub stops any task no CLI has followed for 60 s (`hub --orphan-timeout`) |
| the command exceeds `--timeout` (600 s by default) | forced stop by the agent |

A stuck script that prints nothing is still stopped within 5 s: during a task, the agent contacts the hub every 5 s
and receives the stop order then.

The stop is forced and reaches **every subprocess**: `SIGKILL` on the process group (macOS, Linux),
`taskkill /T /F` (Windows). No script can ignore it.

The only exception: a process stuck in a driver call (crashed GPU) may not die, even with `SIGKILL`. If the process
is still alive 15 s after the forced stop, the agent gives up on it with the status `unkillable`, prints its PID and
stays available for the next tasks. The abandoned process may then require restarting the machine.

### Synchronization

Before each `run` (unless `--no-sync`), the repository is synchronized to a **workspace** of the same name on the
agent:

- source: git root of the current folder (or `--src`); workspace: name of the folder (or `--ws`);
- only changed files are transferred (SHA-256 comparison); files deleted locally are deleted remotely too;
- **files created remotely are kept** (`target/`, `.venv/`, results): Rust builds stay incremental from one run to the
  next;
- default exclusions: `.git`, `target`, `__pycache__`, `.venv`, `venv`, `node_modules`, `dist`, `build`…; add your
  own with `--exclude PATTERN` or in a `.gsremoteignore` file at the root (one pattern per line).

---

## 3. Security

| Risk | Protection |
|---|---|
| Someone contacts a test machine to make it run code | **impossible**: the agents listen on no port |
| A fake hub (spoofed IP on the network) sends a task to an agent | every hub response is signed with HMAC-SHA256 and the agent's token: without the token, the agent refuses everything |
| An old, captured task is sent again to an agent | every response is bound to the unique nonce of the agent's request: a replayed response is refused |
| Someone sends tasks to the hub | the control API only accepts requests **local to the Mac**, signed with a token stored in `~/.gpusync-remote/hub.json` (mode 600) |
| An agent pretends to be another one | one token per agent; an agent only reaches its own tasks |
| An agent reads files on the Mac | it can only request the files listed in its task (the synchronized repository) |
| A captured request is replayed to the hub | timestamp (± 5 min) + single-use nonce |
| A request is modified on the way | the signature covers the identity, the method, the path, the timestamp and the body hash |
| Access to the hub from the Internet | only private and local IPs are accepted (`--allow 192.168.1.0/24` to restrict further) |
| A received file overwrites something outside the workspace | absolute paths, `..`, drive letters and outgoing symbolic links are refused |
| A command never ends | timeout (600 s by default), then the process **and its subprocesses** are stopped |

Tested: fake hub without a signature, signed with another token, and signed with the right token but replayed → the
agent refuses in all three cases and runs nothing.

Limits to know:

- traffic **is not encrypted**: the code sent and the output can be read on the local network. Use it on a trusted
  network (at home), not on public Wi-Fi;
- **an agent token lets whoever holds it make that agent run code**, provided they can reach the agent while
  pretending to be the hub. Keep it secret; when in doubt, `agents rm` then `agents add`;
- the machines' clocks must be on time (± 5 min);
- commands run with the rights of the user who started the agent: do not run it as administrator. Stop it
  (`Ctrl+C`) when you are not using it.

---

## 4. Files

| Machine | File | Contents |
|---|---|---|
| Mac | `~/.gpusync-remote/hub.json` | control token, agent tokens, port |
| Mac | `~/.gpusync-remote/hub/jobs/` | task output |
| test machine | `~/.gpusync-remote/agent.json` | hub URL, name, token |
| test machine | `~/.gpusync-remote/agent/workspaces/` | synchronized repositories and produced files |

`GS_REMOTE_HOME` changes the `~/.gpusync-remote` folder.

---

## 5. Protocol (for reference)

Requests: headers `X-GS-Who` (agent name, or `@ctl` for the CLI), `X-GS-Time`, `X-GS-Nonce`,
`X-GS-Sig = HMAC(token, "req\nwho\nMETHOD\npath\ntimestamp\nnonce\nsha256(body)")`.
Responses: `X-GS-Resp-Sig = HMAC(token, "resp\nrequest-nonce\nstatus\nsha256(body)")`.

| Route | Caller | Role |
|---|---|---|
| `POST /a/hello` | agent | announcement and machine info |
| `POST /a/poll` | agent | waits for a task (long-polling, 25 s max) |
| `POST /a/files` | agent | files of a synchronization task (zip) |
| `POST /a/output` | agent | sends output; the response says whether the task must stop |
| `POST /a/upload?job=` | agent | file requested by `get` |
| `POST /a/done` | agent | end of task (status, code) |
| `GET /c/agents` | CLI (local) | state of the agents |
| `POST /c/jobs` | CLI (local) | submit a task |
| `GET /c/jobs/<id>?offset=` | CLI (local) | status + output from byte N |
| `POST /c/jobs/<id>/kill` | CLI (local) | stop a task |
| `GET /c/jobs/<id>/result` | CLI (local) | file fetched by `get` |

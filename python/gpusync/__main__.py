"""`python -m gpusync`: diagnostic of the backends and devices of this machine.

    python -m gpusync                        # what is detected, and why a backend is unavailable
    python -m gpusync --check                # also runs small kernels on every device, compared with NumPy
    python -m gpusync --check -d metal:0     # only some devices (repeatable)

With `--check`, the exit code is 1 when a test fails on a device.
"""

import argparse
import platform
import sys

import gpusync as gs


def _size(n):
    return f"{n / 2**30:.1f} GB" if n is not None else "?"


def _diagnostic() -> None:
    print(f"gpusync {gs.__version__} — {platform.system()} {platform.machine()}, Python {platform.python_version()}\n")
    for st in gs.backends():
        state = f"{len(st.devices)} device(s)" if st.available else f"unavailable: {st.error}"
        print(f"[{st.name}] {state}")
        for d in st.devices:
            flags = ", ".join(f for f, on in (("f64", d.f64), ("unified memory", d.unified_memory)) if on) or "—"
            print(f"  {d.id:<10} {d.name}")
            print(f"             {d.kind}, {d.vendor or '?'}, {d.version or '?'} | memory {_size(d.memory_bytes)}, "
                  f"buffer max {_size(d.max_buffer_bytes)} | {flags}")
    print(f"\nDefault device: {gs.devices()[0] if gs.devices() else 'none'}")


MARK = {"ok": "✓", "skipped": "–", "failed": "✗"}


def _check(devices) -> int:
    from gpusync._check import TESTS, check

    print("\nSelf-test (each result compared with NumPy):\n")
    try:
        results = check(devices)
    except gs.DeviceError as e:
        print(e)
        return 2
    width = max((len(r.device.name) for r in results), default=0)
    for r in results:
        marks = "  ".join(f"{MARK[r.status[n]]} {n}" for n, _ in TESTS)
        print(f"  {r.device.id:<10} {r.device.name:<{width}}  {marks}  ({r.seconds:.1f} s)")
        for name, err in r.errors.items():
            print(f"             {name}: {err}")
    failed = [r.device.id for r in results if not r.ok]
    if failed:
        print(f"\nFAILED on {', '.join(failed)}.")
        return 1
    print(f"\nAll {len(results)} device(s) passed.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m gpusync", description="Diagnostic of GPUsync backends and devices.")
    p.add_argument("--check", action="store_true", help="run small kernels on every device and compare with NumPy")
    p.add_argument("-d", "--device", action="append", metavar="ID", help="with --check: only this device (repeatable)")
    args = p.parse_args(argv)
    if args.device and not args.check:
        p.error("--device goes with --check")
    _diagnostic()
    return _check(args.device) if args.check else 0


if __name__ == "__main__":
    sys.exit(main())

"""`python -m gpusync`: diagnostic of the backends and devices of this machine.

    python -m gpusync                        # what is detected, and why a backend is unavailable
    python -m gpusync --check                # also runs small kernels on every device, compared with NumPy
    python -m gpusync --check -d metal:0     # only some devices (repeatable)

With `--check`, the exit code is 1 when a test fails on a device. On a console that cannot show Unicode
(Windows code page 1252, output redirected to a file), the marks become ASCII: + passed, - skipped, X failed.
"""

import argparse
import platform
import sys

import gpusync as gs


def _size(n):
    return f"{n / 2**30:.1f} GB" if n is not None else "?"


def _unicode_ok() -> bool:
    """True if stdout can encode the marks; otherwise unknown characters are replaced instead of raising."""
    try:
        "✓✗–—→".encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")
        return False


def _diagnostic(dash: str) -> None:
    print(f"gpusync {gs.__version__} {dash} {platform.system()} {platform.machine()}, Python {platform.python_version()}\n")
    for st in gs.backends():
        state = f"{len(st.devices)} device(s)" if st.available else f"unavailable: {st.error}"
        print(f"[{st.name}] {state}")
        for d in st.devices:
            flags = ", ".join(f for f, on in (("f64", d.f64), ("unified memory", d.unified_memory)) if on) or dash
            print(f"  {d.id:<10} {d.name}")
            print(f"             {d.kind}, {d.vendor or '?'}, {d.version or '?'} | memory {_size(d.memory_bytes)}, "
                  f"buffer max {_size(d.max_buffer_bytes)} | {flags}")
    print(f"\nDefault device: {gs.devices()[0] if gs.devices() else 'none'}")


MARKS = {"ok": "✓", "skipped": "–", "failed": "✗"}
ASCII_MARKS = {"ok": "+", "skipped": "-", "failed": "X"}


def _check(devices, marks_for) -> int:
    from gpusync._check import TESTS, check

    print("\nSelf-test (each result compared with NumPy):\n")
    try:
        results = check(devices)
    except gs.DeviceError as e:
        print(e)
        return 2
    width = max((len(r.device.name) for r in results), default=0)
    for r in results:
        marks = "  ".join(f"{marks_for[r.status[n]]} {n}" for n, _ in TESTS)
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
    unicode = _unicode_ok()
    _diagnostic("—" if unicode else "-")
    return _check(args.device, MARKS if unicode else ASCII_MARKS) if args.check else 0


if __name__ == "__main__":
    sys.exit(main())

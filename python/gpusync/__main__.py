"""`python -m gpusync`: diagnostic of the backends and devices of this machine."""

import platform

import gpusync as gs


def _size(n):
    return f"{n / 2**30:.1f} GB" if n is not None else "?"


def main() -> None:
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


if __name__ == "__main__":
    main()

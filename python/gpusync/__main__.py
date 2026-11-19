"""`python -m gpusync` : diagnostic des backends et des périphériques de cette machine."""

import platform

import gpusync as gs


def _size(n):
    return f"{n / 2**30:.1f} Go" if n is not None else "?"


def main() -> None:
    print(f"gpusync {gs.__version__} — {platform.system()} {platform.machine()}, Python {platform.python_version()}\n")
    for st in gs.backends():
        state = f"{len(st.devices)} périphérique(s)" if st.available else f"indisponible : {st.error}"
        print(f"[{st.name}] {state}")
        for d in st.devices:
            flags = ", ".join(f for f, on in (("f64", d.f64), ("mémoire unifiée", d.unified_memory)) if on) or "—"
            print(f"  {d.id:<10} {d.name}")
            print(f"             {d.kind}, {d.vendor or '?'}, {d.version or '?'} | mémoire {_size(d.memory_bytes)}, "
                  f"buffer max {_size(d.max_buffer_bytes)} | {flags}")
    print(f"\nPériphérique par défaut : {gs.devices()[0] if gs.devices() else 'aucun'}")


if __name__ == "__main__":
    main()

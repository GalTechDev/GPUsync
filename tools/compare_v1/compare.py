"""Prototype Python (v1) contre GPUsync Rust (v2), mêmes calculs sur le même GPU.

    python tools/compare_v1/compare.py PYTHON_V1 V1_BACKEND V2_DEVICE [V1_BACKEND V2_DEVICE ...]

ex. Mac : `.venv-v1/bin/python opencl opencl:0` ; PC : `... cuda cuda:0 opencl/NVIDIA opencl:0`.
PYTHON_V1 : interpréteur de l'environnement du prototype (numpy, pyopencl, llvmlite, cupy si CUDA).
Chaque mesure tourne dans un processus neuf ; la tâche 1 Gio a le sien (pic mémoire par processus).
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "worker.py")
V2_PY = sys.executable


def run(py, impl, backend, tasks):
    r = subprocess.run([py, WORKER, impl, backend, *tasks], capture_output=True, text=True, timeout=1800)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        return {"error": (r.stderr or r.stdout).strip()[-400:]}
    return json.loads(line[7:])


def measure(py, impl, backend):
    res = run(py, impl, backend, ["lat", "big", "dtw391", "dtw2000"])
    res.update(run(py, impl, backend, ["dtw1n"]))
    res.update(run(py, impl, backend, ["huge"]))
    if "import_s" in res:
        res["import_ms"] = res["import_s"] * 1e3
    return res


ROWS = [
    ("import de la bibliothèque", "import_ms", "ms"),
    ("1er appel (compilation comprise)", "first_call_ms", "ms"),
    ("surcoût par appel, 1000 éléments", "call_ms", "ms"),
    ("10 M éléments, aller-retour", "big_ms", "ms"),
    ("1 Gio, aller-retour", "huge_s", "s"),
    ("1 Gio : mémoire en plus", "huge_extra_gib", "Gio"),
    ("DTW 391 séries OXYGENE (w=5)", "dtw391_ms", "ms"),
    ("DTW 2000 × 64 (w=5)", "dtw2000_ms", "ms"),
]


def fmt(v, unit):
    if v is None:
        return "—"
    if unit == "Gio":
        return f"{v:.2f} Gio"
    if unit == "s":
        return f"{v:.2f} s"
    return f"{v:.3f} ms" if v < 10 else f"{v:,.1f} ms".replace(",", " ")


def merge(path):
    """Résultats mesurés tâche par tâche (lignes RESULT d'un fichier), fusionnés."""
    res = {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("RESULT "):
            res.update(json.loads(line[7:]))
    if "import_s" in res:
        res["import_ms"] = res["import_s"] * 1e3
    return res


def main():
    if sys.argv[1] == "--from":  # tableau à partir de mesures faites ailleurs (une tâche par job)
        report(merge(sys.argv[2]), merge(sys.argv[3]), sys.argv[4], sys.argv[5])
        return
    py_v1, pairs = sys.argv[1], sys.argv[2:]
    for b1, b2 in zip(pairs[::2], pairs[1::2]):
        report(measure(py_v1, "v1", b1), measure(V2_PY, "v2", b2), b1, b2)


def report(v1, v2, b1, b2):
    if True:
        print(f"\n## {v2.get('device', b2)} — prototype ({b1}) contre Rust ({b2})\n")
        for r in (v1, v2):
            errs = {k: v for k, v in r.items() if "error" in k}
            if errs:
                print(f"  erreurs {r.get('impl', '?')} : {errs}")
        print(f"| mesure | prototype v1 | Rust v2 | gain |\n|---|---|---|---|")
        for label, key, unit in ROWS:
            a, b = v1.get(key), v2.get(key)
            gain = f"× {a / b:,.1f}".replace(",", " ") if a and b and unit != "Gio" else ""
            print(f"| {label} | {fmt(a, unit)} | {fmt(b, unit)} | {gain} |")
        checks = []
        for k in ("lat_ok", "big_ok", "huge_ok"):
            checks.append(f"{k[:-3]} v1 {'✓' if v1.get(k) else '✗'} v2 {'✓' if v2.get(k) else '✗'}")
        checks.append(f"DTW 391 vs référence : v1 {v1.get('dtw391_err')}, v2 {v2.get('dtw391_err')}")
        if v1.get("dtw2000_sum") and v2.get("dtw2000_sum"):
            rel = abs(v1["dtw2000_sum"] - v2["dtw2000_sum"]) / v2["dtw2000_sum"]
            corner = np.abs(np.array(v1["dtw2000_corner"]) - np.array(v2["dtw2000_corner"])).max()
            checks.append(f"DTW 2000 v1 = v2 : somme écart relatif {rel:.1e}, coin 4×4 écart {corner:.1e}")
        print("\nvérifications : " + " ; ".join(checks))
        report_1n(v1, v2)


def report_1n(v1, v2):
    """DTW 1 contre N (benchmark DTW du prototype) : temps par N et par variante, et accord des résultats."""
    a, b = v1.get("dtw1n"), v2.get("dtw1n")
    if not (a or b):
        return
    variants = list((a or {}).keys()) + list((b or {}).keys())
    rows = {**(a or {}), **(b or {})}
    sizes = sorted({int(n) for v in rows.values() for n in v})
    print("\nDTW 1 contre N (séries de 30 points complétées à 32, sans fenêtre) :\n")
    print("| N | " + " | ".join(f"v1 {k}" if k != "Rust" else "v2 Rust" for k in variants) + " | gain (meilleur v1 / v2) |")
    print("|---" * (len(variants) + 2) + "|")
    ref_rows = rows.get("Rust", {})

    def correct(k, n, c):
        """Résultat d'accord avec v2 (et la variante manuelle du prototype) ? Sinon, temps non retenu."""
        r = ref_rows.get(str(n), ref_rows.get(n))
        return k == "Rust" or r is None or abs(c["sum"] - r["sum"]) <= 1e-6 * abs(r["sum"])

    for n in sizes:
        cells = [rows[k].get(str(n), rows[k].get(n)) for k in variants]
        ok = [c is not None and correct(k, n, c) for k, c in zip(variants, cells)]
        ms = [c["ms"] if c else None for c in cells]
        v1_best = min((m for k, m, g in zip(variants, ms, ok) if k != "Rust" and m and g), default=None)
        v2_ms = next((m for k, m in zip(variants, ms) if k == "Rust"), None)
        gain = f"× {v1_best / v2_ms:.1f}" if v1_best and v2_ms else ""
        txt = [fmt(m, "ms") + ("" if g or m is None else " ✗ faux") for m, g in zip(ms, ok)]
        print(f"| {n:,} | ".replace(",", " ") + " | ".join(txt) + f" | {gain} |")
    ref = rows.get("Rust", {})
    for k in variants:
        if k == "Rust":
            continue
        diffs = []
        for n, c in rows[k].items():
            r = ref.get(n)
            if c and r:
                diffs.append(abs(c["sum"] - r["sum"]) / max(abs(r["sum"]), 1e-30))
        if diffs:
            print(f"accord v1 {k} / v2 : écart relatif des sommes ≤ {max(diffs):.1e}")
    errs = {k: v for k, v in {**v1, **v2}.items() if k == "dtw1n_error"}
    if errs:
        print(f"erreur : {errs['dtw1n_error'][:160]}")


if __name__ == "__main__":
    main()

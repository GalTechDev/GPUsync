"""Graphique : DTW 1 contre N, prototype v1 contre GPUsync Rust v2, sur le Mac et sur la RTX.

    python tools/compare_v1/plot_dtw1n.py [résultats.json] [image.png]

Lit les mesures enregistrées par `compare.py` (tools/compare_v1/results/) ; deux panneaux (un par
machine), temps en fonction de N. Le prototype est sa meilleure variante dont le résultat est juste.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "dtw1n_2026-11-21.json")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results", "dtw1n_v1_v2.png")

# Palette de référence (thème clair) : 3 premiers emplacements, validés toutes paires.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
COLORS = {"OpenCL": "#2a78d6", "CUDA": "#eb6834", "wgpu": "#1baf7a"}

data = json.load(open(SRC, encoding="utf-8"))


def series(entry, variant):
    d = (entry.get("dtw1n") or {}).get(variant) or {}
    pts = sorted((int(n), v) for n, v in d.items())
    return [n for n, _ in pts], [v["ms"] for _, v in pts], [v["sum"] for _, v in pts]


def best_v1(entry, ref_entry):
    """Meilleure variante v1 par N, parmi celles dont la somme est d'accord avec v2 (1e-6)."""
    ref = {n: s for n, _, s in zip(*series(ref_entry, "Rust"))}
    out = {}
    for variant in ("OOC Manuel", "OOC Auto"):
        for n, ms, s in zip(*series(entry, variant)):
            if n in ref and abs(s - ref[n]) <= 1e-6 * abs(ref[n]):
                out[n] = min(out.get(n, ms), ms)
    ns = sorted(out)
    return ns, [out[n] for n in ns]


panels = [
    ("Mac M5 Pro", "mac_v1", [("OpenCL", "mac_v2_opencl"), ("wgpu", "mac_v2_metal")], "mac_v2_opencl",
     {"wgpu": "Rust Metal"}),
    ("RTX 4080 SUPER", "rtx_v1_opencl", [("CUDA", "rtx_v2_cuda"), ("OpenCL", "rtx_v2_opencl"),
                                         ("wgpu", "rtx_v2_vulkan")], "rtx_v2_cuda", {"wgpu": "Rust Vulkan"}),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE)
for ax, (title, v1_key, v2s, ref_key, names) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    lines = []
    ns, ms = best_v1(data[v1_key], data[ref_key])
    lines.append(("Prototype v1", ns, ms, INK2, "--"))
    for kind, key in v2s:
        n2, m2, _ = series(data[key], "Rust")
        lines.append((names.get(kind, f"Rust {kind}"), n2, m2, COLORS[kind], "-"))
    ymax = max(max(m) for _, _, m, _, _ in lines)
    for label, n, m, color, style in lines:
        x = [v / 1e6 for v in n]
        ax.plot(x, m, style, color=color, lw=2, marker="o", ms=8, mfc=color, mec=SURFACE, mew=2,
                label=label, solid_capstyle="round", zorder=3)
    # Étiquettes directes en bout de ligne (texte en encre, pas en couleur de série), écartées si proches.
    ends = sorted(((m[-1], label) for label, _, m, _, _ in lines), reverse=True)
    placed = []
    for y, label in ends:
        yy = y
        for p in placed:
            if abs(yy - p) < ymax * 0.06:
                yy = p - ymax * 0.06
        placed.append(yy)
        ax.annotate(f"{label}  {y:.0f} ms", xy=(2.0, y), xytext=(2.06, yy), textcoords="data",
                    va="center", ha="left", fontsize=10, color=INK)
    v1_end = lines[0][2][-1]
    best_v2 = min(m[-1] for _, _, m, _, _ in lines[1:])
    gain = f"{v1_end / best_v2:.1f}".replace(".", ",")
    ax.set_title(f"{title}\nà 2 M : Rust × {gain} plus rapide", loc="left",
                 fontsize=13, color=INK, pad=12)
    ax.set_xlim(0, 2.75)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xticks([0.1, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0,1 M", "0,5 M", "1 M", "1,5 M", "2 M"])
    ax.set_xlabel("N (séries comparées à la référence)", color=INK2)
    ax.set_ylabel("temps (ms, moins = mieux)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK)

fig.suptitle("DTW 1 contre N : prototype Python contre GPUsync Rust", x=0.01, ha="left", fontsize=15,
             color=INK, fontweight="bold")
fig.text(0.01, 0.005,
         "Séries de 30 points complétées à 32, sans fenêtre (benchmark DTW du prototype). Prototype : sa meilleure "
         "variante au résultat juste.\nSur la RTX, sa variante « OOC Auto » (résultats faux) est exclue, et son kernel "
         "ne compile pas en CUDA : la courbe est son OpenCL.\nMesures du 21/11/2026, à chaud, meilleur de 3 ; "
         "résultats identiques entre versions (écart relatif ≤ 3e-9).", fontsize=9, color=INK2, va="bottom")
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(DST, dpi=160, facecolor=SURFACE)
print(DST)

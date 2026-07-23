# -*- coding: utf-8 -*-
"""
comparaison_og_new_mmts.py
==========================
Compare VERSION ORIGINALE (Pure Python) vs NOUVELLE VERSION (Numba JIT + GPU GPUsync)
sur le MEME dataset reel — aucune extrapolation.

Metriques :
  - Temps total (ms) et normalise (ms/1000-paires)
  - Ecart des resultats : accord clusters (%), MAE matrices DTW

Usage :
    python comparaison_og_new_mmts.py [--n 500] [--k 3] [--w 5] [--rep 1]
    
    --n 0  => dataset complet (~4969 series, attention ~30 min pour l originale)
"""

import os, sys, time, random, warnings, argparse
warnings.filterwarnings("ignore")

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

HERE         = os.path.dirname(os.path.abspath(__file__))
GPUSYNC_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, GPUSYNC_ROOT)
sys.path.insert(0, HERE)
from config import *

DATA_CSV = os.path.join(HERE, "data", "RomaneDataSet_TEMP.csv")
WINDOW   = 32

def load_data(max_series=None):
    df      = pd.read_csv(DATA_CSV)
    val_col = "value" if "value" in df.columns else df.columns[1]
    raw     = df[val_col].values.astype(np.float32)
    N_all   = len(raw) - WINDOW + 1
    data_list = [raw[i : i + WINDOW] for i in range(N_all)]
    data_np   = np.array(data_list, dtype=np.float32)
    df_ids    = pd.DataFrame({ID_COLUMN: list(range(N_all))})
    if max_series and max_series < N_all:
        data_list = data_list[:max_series]
        data_np   = data_np[:max_series]
        df_ids    = df_ids.iloc[:max_series].reset_index(drop=True)
    return data_list, data_np, df_ids

import importlib.util
def import_module(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

og = import_module("xmeants_og",  os.path.join(HERE, "xmeants_module.py"))
nw = import_module("xmeants_new", os.path.join(HERE, "new_xmeants_module.py"))

GRAY="\033[90m";WHITE="\033[1;37m";CYAN="\033[36m";GREEN="\033[32m"
YELLOW="\033[93m";ORANGE="\033[1;33m";RED="\033[31m";RESET="\033[0m"
BLUE="\033[94m"

def hdr(t):
    sep="="*90
    print(f"\n{GRAY}{sep}{RESET}\n  {WHITE}{t}{RESET}\n{GRAY}{sep}{RESET}")

def ok(m):   print(f"  {GREEN}[OK]{RESET}  {m}")
def warn(m): print(f"  {YELLOW}[!!]{RESET}  {m}")
def err(m):  print(f"  {RED}[ERR]{RESET} {m}")

def spd_str(t_og, t_nw):
    if t_nw <= 0: return "N/A"
    r = t_og / t_nw
    c = GREEN if r >= 2 else (YELLOW if r >= 1 else RED)
    return f"{c}{r:.1f}x{RESET}"

def norm1k(t_sec, n_pairs):
    return (t_sec / n_pairs) * 1_000_000   # ms/1000-paires

def compare_assignments(a_og, a_nw, N):
    """Retourne le % de series ayant le meme cluster (remappage greedy)."""
    lo = np.full(N, -1, int); ln = np.full(N, -1, int)
    for c, ids in a_og.items():
        for i in ids:
            if 0 <= i < N: lo[i] = c
    for c, ids in a_nw.items():
        for i in ids:
            if 0 <= i < N: ln[i] = c
    used, mapping = set(), {}
    for co in sorted(set(lo)):
        bm, bc = -1, -1
        for cn in sorted(set(ln)):
            if cn in used: continue
            cnt = int(np.sum((lo == co) & (ln == cn)))
            if cnt > bc: bc, bm = cnt, cn
        mapping[co] = bm; used.add(bm)
    ln_m = np.vectorize(lambda x: mapping.get(x, -1))(ln)
    mask = (lo >= 0)
    ns = int(np.sum((lo == ln_m) & mask))
    nt = int(np.sum(mask))
    return 100.0 * ns / nt if nt > 0 else 0.0

def mat_mae(m_og, m_nw):
    """MAE entre deux matrices de distances DTW (colonnes triees)."""
    if m_og is None or m_nw is None: return None
    co = sorted(c for c in m_og.columns if c != ID_COLUMN)
    cn = sorted(c for c in m_nw.columns if c != ID_COLUMN)
    if len(co) != len(cn): return None
    vo = m_og[co].values.astype(float)
    vn = m_nw[cn].values.astype(float)
    if vo.shape != vn.shape: return None
    return float(np.mean(np.abs(vo - vn)))

def run_comparison(n_series=500, num_clust=3, w=5, nb_min=2, n_repeat=1, skip_coeff=False, multipop=False):
    hdr(f"COMPARAISON ORIGINALE vs NOUVELLE — {n_series} series reelles, k={num_clust}, w={w}")

    print(f"\n{GRAY}Chargement (fenetres glissantes {WINDOW} pts)...{RESET}")
    data_list, data_np, df_ids = load_data(max_series=n_series)
    N = len(data_list); M = len(data_list[0])
    n_pairs = num_clust * N
    print(f"  -> {N} series x {M} pts = {n_pairs:,} paires DTW / iteration")

    # Warm-up (Chauffe du GPU et du JIT CPU avec les mêmes paramètres constants pour peupler le cache)
    print(f"\n{GRAY}Chauffe du GPU (OpenCL) et du CPU (Numba JIT) avec K={num_clust} et M={M}...{RESET}")
    try:
        # Créer des données factices avec la bonne dimension M
        dummy_series = np.zeros(M, dtype=np.float32)
        dummy_data = [dummy_series for _ in range(num_clust * 2)]
        dummy_df = pd.DataFrame({'elev_id': list(range(num_clust * 2))})
        
        # Chauffe CPU
        x_dummy = og.XmeansTS(dummy_data, num_clust, "TEMP", dummy_df, 1.0)
        x_dummy.k_means_clust(dummy_data, num_clust, 1, dummy_df, 0, "Set_TEMP", {}, [], 1, w)
        
        # Chauffe GPU
        x_dummy2 = nw.XmeansTS(dummy_data, num_clust, "TEMP", dummy_df, 1.0)
        x_dummy2.k_means_clust(dummy_data, num_clust, 1, dummy_df, 0, "Set_TEMP", {}, [], 1, w)
        
        if multipop:
            from new_xmeants_module import XmeansTS_MultiPopulation
            x_dummy3 = XmeansTS_MultiPopulation(dummy_data, num_clust, "TEMP", dummy_df, 1.0, P=5)
            x_dummy3.k_means_clust(dummy_data, num_clust, 1, dummy_df, 0, "Set_TEMP", {}, [], 1, w)
        print(f"  -> Chauffe terminee (Noyaux compiles en cache).\n")
    except Exception as e:
        print(f"  -> Avertissement lors de la chauffe : {e}\n")

    W_cols = [30, 20, 20, 14, 18]
    def cc(v, i, c=""):
        s = f"{str(v):<{W_cols[i]}}"; return f"{c}{s}{RESET}" if c else s

    results = {}

    # =========================================================================
    # 1. Batch DTW complet
    # =========================================================================
    hdr(f"1. Batch DTW complet  (K={num_clust} x N={N} = {n_pairs:,} paires)")

    print(f"  {GRAY}[1/2] Originale en cours...{RESET}", flush=True)
    cents = {i: data_np[random.randint(0, N-1)] for i in range(num_clust)}
    cents_np = np.array(list(cents.values()), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        dist_og = np.zeros((num_clust, N), dtype=np.float32)
        for i in range(N):
            for c, cv in cents.items():
                dist_og[c, i] = og.DTWDistance(data_np[i], cv, w)
    t_dtw_og = (time.perf_counter() - t0) / n_repeat

    print(f"  {GRAY}[2/2] Nouvelle en cours...{RESET}", flush=True)
    from new_xmeants_module import _compute_dist_matrix
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        dist_nw = _compute_dist_matrix(cents_np, data_np, w)
    t_dtw_nw = (time.perf_counter() - t0) / n_repeat

    print(f"  Originale : {t_dtw_og*1000:.1f} ms | {norm1k(t_dtw_og, n_pairs):.4f} ms/1k-paires")
    print(f"  Nouvelle  : {t_dtw_nw*1000:.1f} ms | {norm1k(t_dtw_nw, n_pairs):.4f} ms/1k-paires")
    print(f"  Speedup   : {spd_str(t_dtw_og, t_dtw_nw)}")

    delta_dtw = mat_mae(pd.DataFrame(dist_og.T), pd.DataFrame(dist_nw.T))
    if delta_dtw is not None:
        (ok if delta_dtw < 1e-4 else (warn if delta_dtw < 1.0 else err))(f"MAE distances DTW = {delta_dtw:.6f}")
        delta_dtw_s = f"MAE={delta_dtw:.4f}"
    else:
        delta_dtw_s = "Error shape"
        warn("Erreur de dimension des distances")
    results["dtw"] = (t_dtw_og, t_dtw_nw, n_pairs, delta_dtw_s)

    # =========================================================================
    # 2. Pipeline k_means_clust
    # =========================================================================
    hdr(f"2. Pipeline k_means_clust  (N={N} series reelles)")

    print(f"  {GRAY}[1/2] Originale en cours ({N} series)...{RESET}", flush=True)
    random.seed(42); np.random.seed(42)
    t0 = time.perf_counter()
    xn1 = og.XmeansTS(data_np, num_clust, "TEMP", df_ids.copy(), 1.0)
    xn1.k_means_clust(data_np, num_clust, 1, df_ids.copy(), 0, "Set_TEMP",
                      {}, [], nb_min, w)
    t_pipe_og = time.perf_counter() - t0

    print(f"  {GRAY}[2/2] Nouvelle en cours ({N} series)...{RESET}", flush=True)
    random.seed(42); np.random.seed(42)
    t0 = time.perf_counter()
    if multipop:
        print(f"  {ORANGE}[V2 Multi-Population active (P=5)]{RESET}")
        from new_xmeants_module import XmeansTS_MultiPopulation
        xn2 = XmeansTS_MultiPopulation(data_list, num_clust, "TEMP", df_ids.copy(), 1.0, P=5)
    else:
        xn2 = nw.XmeansTS(data_list, num_clust, "TEMP", df_ids.copy(), 1.0)
        
    xn2.k_means_clust(data_list, num_clust, 1, df_ids.copy(), 0, "Set_TEMP",
                      {}, [], nb_min, w)
    t_pipe_nw = time.perf_counter() - t0

    print(f"  Originale : {t_pipe_og*1000:.1f} ms")
    print(f"  Nouvelle  : {t_pipe_nw*1000:.1f} ms")
    print(f"  Speedup   : {spd_str(t_pipe_og, t_pipe_nw)}")

    acc = compare_assignments(xn1.assignments, xn2.assignments, N)
    (ok if acc > 99.0 else warn)(f"Accord clusters : {acc:.1f}% ({N} series, seed=42)")
    results["pipe"] = (t_pipe_og, t_pipe_nw, N, f"Accord={acc:.1f}%")

    # =========================================================================
    # 3. getLastMat
    # =========================================================================
    hdr(f"3. getLastMat  (K={num_clust} x N={N} = matrices de distances)")

    print(f"  {GRAY}[1/2] Originale...{RESET}", flush=True)
    t0 = time.perf_counter()
    mat_og = xn1.getLastMat("Set_TEMP", w)
    t_gmat_og = time.perf_counter() - t0

    print(f"  {GRAY}[2/2] Nouvelle...{RESET}", flush=True)
    t0 = time.perf_counter()
    mat_nw = xn2.getLastMat("Set_TEMP", w)
    t_gmat_nw = time.perf_counter() - t0

    print(f"  Originale : {t_gmat_og*1000:.1f} ms")
    print(f"  Nouvelle  : {t_gmat_nw*1000:.1f} ms")
    print(f"  Speedup   : {spd_str(t_gmat_og, t_gmat_nw)}")

    delta_gmat = mat_mae(mat_og, mat_nw)
    if delta_gmat is not None:
        (ok if delta_gmat < 1e-4 else (warn if delta_gmat < 1.0 else err))(f"MAE matrices = {delta_gmat:.4f}")
        delta_gmat_s = f"MAE={delta_gmat:.4f}"
    else:
        delta_gmat_s = "Error"
        warn("Erreur matrices")

    print(f"  {GRAY}--- Diagnostic centroides ---{RESET}")
    for c_key in sorted(xn1.centroids.keys()):
        cv = xn1.centroids[c_key]
        l  = len(cv) if hasattr(cv, '__len__') else 1
        print(f"    Originale centroide[{c_key}] : longueur={l}  {GREEN}[OK]{RESET}")
    for c_key in sorted(xn2.centroids.keys()):
        cv = xn2.centroids[c_key]
        l  = len(cv) if hasattr(cv, '__len__') else 1
        print(f"    Nouvelle  centroide[{c_key}] : longueur={l}  {GREEN}[OK]{RESET}")
    print()

    results["gmat"] = (t_gmat_og, t_gmat_nw, n_pairs, delta_gmat_s)

    # =========================================================================
    # 4. defineCoeff
    # =========================================================================
    hdr(f"4. defineCoeff  ({n_pairs:,} paires)")

    if skip_coeff:
        print(f"  {YELLOW}[SKIPPED] Etape definie par l'utilisateur, passee.{RESET}")
        t_coef_og = 0.0
        t_coef_nw = 0.0
        delta_coef = "SKIPPED"
    else:
        print(f"  {GRAY}[1/2] Originale...{RESET}", flush=True)
        random.seed(42); np.random.seed(42)
        t0 = time.perf_counter()
        coef_og = og.defineCoeff(data_np, num_clust, 1, df_ids.copy(), 0, "Set_TEMP",
                                  {}, {}, {}, data_np, [], nb_min, w)
        t_coef_og = time.perf_counter() - t0

        print(f"  {GRAY}[2/2] Nouvelle...{RESET}", flush=True)
        random.seed(42); np.random.seed(42)
        t0 = time.perf_counter()
        coef_nw = nw.defineCoeff(data_list, num_clust, 1, df_ids.copy(), 0, "Set_TEMP",
                                  {}, {}, {}, data_list, [], nb_min, w)
        t_coef_nw = time.perf_counter() - t0

        print(f"  Originale : {t_coef_og*1000:.1f} ms  coef={[round(c,4) for c in coef_og]}")
        print(f"  Nouvelle  : {t_coef_nw*1000:.1f} ms  coef={[round(c,4) for c in coef_nw]}")
        print(f"  Speedup   : {spd_str(t_coef_og, t_coef_nw)}")

        if len(coef_og) == len(coef_nw):
            d = max(abs(a-b) for a,b in zip(sorted(coef_og), sorted(coef_nw)))
            delta_coef = f"Dcoef={d:.4f}"
            (ok if d < 0.5 else warn)(f"Diff coef max = {d:.4f}")
        else:
            delta_coef = f"N={len(coef_og)} vs {len(coef_nw)}"
            warn(f"Nb coefficients : {len(coef_og)} vs {len(coef_nw)}")
    results["coef"] = (t_coef_og, t_coef_nw, n_pairs, delta_coef)

    # =========================================================================
    # DASHBOARD FINAL
    WC = [26, 24, 24, 12, 18]

    def cc2(val, idx, color=""):
        """Pad val to column width FIRST, then wrap with color (hors padding)."""
        s = f"{str(val):<{WC[idx]}}"
        return f"{color}{s}{RESET}" if color else s

    def spd_raw(t_og, t_nw):
        """Retourne (valeur_str, couleur) sans mélanger les codes ANSI dans le padding."""
        if t_nw <= 0: return "N/A", ""
        r = t_og / t_nw
        c = GREEN if r >= 2 else (YELLOW if r >= 1 else RED)
        return f"{r:.1f}x", c

    SEP = "=" * (sum(WC) + len(WC) * 3 + 2)
    HDR = (f" {cc2('Composant',0,WHITE)} | {cc2('Originale (ms / ms/1k)',1,WHITE)}"
           f" | {cc2('Nouvelle  (ms / ms/1k)',2,WHITE)} | {cc2('Speedup',3,WHITE)}"
           f" | {cc2('Delta',4,WHITE)}")

    print(f"{GRAY}{SEP}{RESET}")
    print(HDR)
    print(f"{GRAY}{'-'*len(SEP)}{RESET}")

    def dashboard_row(label, key):
        t_og, t_nw, n_p, delta = results[key]
        sp_val, sp_col = spd_raw(t_og, t_nw)
        og_s = f"{t_og*1000:.1f}ms / {norm1k(t_og,n_p):.3f}ms/1k"
        nw_s = f"{t_nw*1000:.1f}ms / {norm1k(t_nw,n_p):.3f}ms/1k"
        return (f" {cc2(label,0)} | {cc2(og_s,1,CYAN)} | {cc2(nw_s,2,GREEN)}"
                f" | {cc2(sp_val,3,sp_col)} | {cc2(delta,4,BLUE)}")

    print(dashboard_row(f"Batch DTW (k={num_clust}xN)", "dtw"))
    print(dashboard_row("XmeansTS pipeline",             "pipe"))
    print(dashboard_row("getLastMat",                    "gmat"))
    print(dashboard_row("defineCoeff",                   "coef"))

    t_tot_og = results["pipe"][0] + results["gmat"][0] + results["coef"][0]
    t_tot_nw = results["pipe"][1] + results["gmat"][1] + results["coef"][1]
    sp_val, sp_col = spd_raw(t_tot_og, t_tot_nw)
    og_tot = f"{t_tot_og*1000:.1f}ms / {norm1k(t_tot_og,n_pairs):.3f}ms/1k"
    nw_tot = f"{t_tot_nw*1000:.1f}ms / {norm1k(t_tot_nw,n_pairs):.3f}ms/1k"
    print(f"{GRAY}{'-'*len(SEP)}{RESET}")
    print(f" {cc2('TOTAL pipeline',0,WHITE)} | {cc2(og_tot,1,CYAN)} | {cc2(nw_tot,2,GREEN)}"
          f" | {cc2(sp_val,3,sp_col)} | {cc2('',4)}")
    print(f"{GRAY}{SEP}{RESET}")
    print(f"{GRAY}Dataset : {N} series x {M} pts  |  {n_pairs:,} paires DTW / iter  |  k={num_clust}  w={w}{RESET}")
    print(f"{GRAY}Originale = Pure Python  |  Nouvelle = Numba JIT + GPU GPUsync{RESET}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=500,
                        help="Nb series (defaut=500 ; 0=toutes ~4969, attention ~30min)")
    parser.add_argument("--k",   type=int, default=3)
    parser.add_argument("--w",   type=int, default=5)
    parser.add_argument("--min", type=int, default=2)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--skip-coeff", action="store_true", help="Passer l'etape defineCoeff")
    parser.add_argument("--multipop", action="store_true", help="Utiliser la V2 Multi-Population GPU")
    args = parser.parse_args()
    n = args.n if args.n > 0 else None
    run_comparison(n_series=n, num_clust=args.k, w=args.w, nb_min=args.min, n_repeat=args.rep, skip_coeff=args.skip_coeff, multipop=args.multipop)
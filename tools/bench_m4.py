"""Milestone M4: cost of a kernel's first call (analysis + compilation) in a fresh process.

    python tools/dev.py run tools/bench_m4.py

Each measurement runs in a subprocess: "cold" = first call of the process, "warm" = the next
call. With the disk cache, the first call of a second process reuses the compiled binary.
"""

import subprocess
import sys

import gpusync as gs

CHILD = r'''
import sys, time
import numpy as np
t0 = time.perf_counter()
import gpusync as gs
sys.path.insert(0, "tests")
from test_index import dtw
t1 = time.perf_counter()
x = np.random.default_rng(0).random((64, 21), dtype=np.float32)
dev = sys.argv[1]
t2 = time.perf_counter(); np.asarray(dtw(x, 5, device=dev)); t3 = time.perf_counter()
np.asarray(dtw(x, 5, device=dev)); t4 = time.perf_counter()
print(f"{(t1-t0)*1e3:.0f} {(t3-t2)*1e3:.1f} {(t4-t3)*1e3:.2f}")
'''


def main():
    print("device         import+analysis   1st call (cold)   next call   [2nd process: 1st call]")
    for d in gs.devices():
        if d.id == "cpu:0":
            continue
        runs = [subprocess.run([sys.executable, "-c", CHILD, d.id], capture_output=True, text=True) for _ in range(2)]
        if any(r.returncode for r in runs):
            print(f"  {d.id:10} erreur : {runs[0].stderr.strip()[-300:]}")
            continue
        (imp, cold, warm), (_, cold2, _) = (r.stdout.split() for r in runs)
        print(f"  {d.id:10} {imp:>10} ms {cold:>14} ms {warm:>14} ms {cold2:>14} ms")


if __name__ == "__main__":
    main()

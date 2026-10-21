"""M0: how Sparse Blossom's decoding time scales with code size (arXiv:2303.15933, Section 6, Figs. 10-12).

Absolute speed depends on the CPU, so this checks the claims that do not:
  - at p = 0.1% and 0.5%, running time is "roughly linear in the number of nodes";
  - at p = 1%, "the observed complexity of O(n^1.32) is only slightly worse than linear";
  - on one machine, d=29 at p=0.1% takes 3.5 us per round (Introduction) where d=17 takes a mean of
    0.62 us (Section 6): a ratio of 5.6x that any machine should roughly reproduce.

Same protocol as m0_sparse_blossom_speed.py (rounds = d, each basis timed separately, decoding only).
The exponent is a least-squares fit of log(time per shot) against log(number of detectors).

    python bench/m0_scaling.py --out results/m0_scaling.json
"""
import argparse
import json
import platform

import numpy as np
import pymatching
import stim

from m0_sparse_blossom_speed import cpu_name, run

DISTANCES = [5, 7, 9, 11, 13, 15, 17, 21, 25, 29]
SHOTS = {0.001: 100_000, 0.005: 20_000, 0.01: 10_000}


def fit_exponent(points):
    x = np.log([pt["detectors"] for pt in points])
    y = np.log([pt["seconds_per_shot"] for pt in points])
    return float(np.polyfit(x, y, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {"source": "Higgott & Gidney, Sparse Blossom, arXiv:2303.15933, Section 6",
           "protocol": "rounds = d, X basis, decode time only, bit-packed batches, one thread",
           "by_p": {}}
    for p, shots in SHOTS.items():
        points = []
        for d in DISTANCES:
            r = run("x", d, d, p, shots, a.seed + d, chunk=10_000)
            points.append({"distance": d, "detectors": r["detectors"], "shots": shots,
                           "us_per_round": r["us_per_round"],
                           "seconds_per_shot": r["decode_seconds"] / shots,
                           "logical_error_rate": r["logical_error_rate"],
                           "mean_detection_events_per_shot": r["mean_detection_events_per_shot"]})
            print(f"p={p} d={d:2d} detectors={r['detectors']:6d} {r['us_per_round']:8.3f} us/round", flush=True)
        out["by_p"][str(p)] = {"points": points, "exponent_vs_detectors": fit_exponent(points)}
        print(f"p={p} exponent {out['by_p'][str(p)]['exponent_vs_detectors']:.3f}", flush=True)
    low = {pt["distance"]: pt["us_per_round"] for pt in out["by_p"]["0.001"]["points"]}
    out["ratio_d29_over_d17_at_p0.001"] = {"measured": low[29] / low[17], "paper": 3.5 / 0.62}
    out["versions"] = {"stim": stim.__version__, "pymatching": pymatching.__version__,
                       "numpy": np.__version__, "python": platform.python_version()}
    out["machine"] = {"cpu": cpu_name(), "system": platform.platform()}
    print(json.dumps({k: out[k] for k in ("ratio_d29_over_d17_at_p0.001",)}, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

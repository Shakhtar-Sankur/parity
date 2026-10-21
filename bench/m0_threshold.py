"""M0: logical error rates and threshold of the rotated surface code under MWPM (PyMatching 2).

Rotated surface-code memory circuits (Stim), standard circuit-level depolarising noise with one
parameter p, rounds = d, both X and Z bases. Sampling and decoding run through sinter.
Each point stops at --max-errors logical errors or --max-shots shots, whichever comes first.

The threshold estimate is where the logical error rate stops falling with distance: for each pair
of consecutive distances, the p at which their (log-interpolated) curves cross.

    python bench/m0_threshold.py --workers 4 --out results/m0_threshold.json
"""
import argparse
import json
import math
import platform

import numpy as np
import pymatching
import sinter
import stim

from m0_sparse_blossom_speed import circuit, cpu_name

DISTANCES = [3, 5, 7, 9, 11]
PS = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012]


def crossing(ps, a, b):
    """p where log(LER_a) - log(LER_b) changes sign (larger distance b stops being better)."""
    diff = [math.log(x) - math.log(y) if x > 0 and y > 0 else None for x, y in zip(a, b)]
    for i in range(len(ps) - 1):
        if diff[i] is None or diff[i + 1] is None:
            continue
        if diff[i] > 0 >= diff[i + 1] or diff[i] >= 0 > diff[i + 1]:
            t = diff[i] / (diff[i] - diff[i + 1])
            return math.exp(math.log(ps[i]) + t * (math.log(ps[i + 1]) - math.log(ps[i])))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-shots", type=int, default=1_000_000)
    ap.add_argument("--max-errors", type=int, default=1_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tasks = [sinter.Task(circuit=circuit(b, d, d, p), json_metadata={"basis": b, "d": d, "p": p})
             for b in "xz" for d in DISTANCES for p in PS]
    stats = sinter.collect(num_workers=a.workers, tasks=tasks, decoders=["pymatching"],
                           max_shots=a.max_shots, max_errors=a.max_errors)
    rows = []
    for s in stats:
        fit = sinter.fit_binomial(num_shots=s.shots, num_hits=s.errors, max_likelihood_factor=1e3)
        rows.append({**s.json_metadata, "shots": s.shots, "errors": s.errors,
                     "logical_error_rate": s.errors / s.shots if s.shots else None,
                     "ci_low": fit.low, "ci_high": fit.high})
    rows.sort(key=lambda r: (r["basis"], r["d"], r["p"]))
    crossings = {}
    for b in "xz":
        ler = {d: [next(r["logical_error_rate"] for r in rows if r["basis"] == b and r["d"] == d and r["p"] == p)
                   for p in PS] for d in DISTANCES}
        crossings[b] = {f"d{d1}-d{d2}": crossing(PS, ler[d1], ler[d2]) for d1, d2 in zip(DISTANCES, DISTANCES[1:])}
        for d in DISTANCES:
            print(b, f"d={d:2d}", " ".join(f"{x:.2e}" for x in ler[d]), flush=True)
        print(b, "crossings", {k: (round(v, 5) if v else None) for k, v in crossings[b].items()}, flush=True)
    out = {"protocol": "rotated surface-code memory, circuit-level depolarising p, rounds = d, MWPM via PyMatching 2",
           "distances": DISTANCES, "ps": PS, "max_shots": a.max_shots, "max_errors": a.max_errors,
           "points": rows, "crossings": crossings,
           "versions": {"stim": stim.__version__, "pymatching": pymatching.__version__,
                        "sinter": sinter.__version__, "numpy": np.__version__,
                        "python": platform.python_version()},
           "machine": {"cpu": cpu_name(), "system": platform.platform()}}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

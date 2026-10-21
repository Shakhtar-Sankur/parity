"""M0: reproduce the Sparse Blossom speed claim (Higgott & Gidney, arXiv:2303.15933, Section 6).

Claim: at 0.1% circuit-level depolarising noise, sparse blossom (PyMatching 2) decodes distance-17
surface-code circuits in both the X and Z bases in under 1 microsecond per round of syndrome
extraction, on one core. The paper reports a mean of 0.62 us per round, one million shots per point,
on one core of an Apple M1 Max. Protocol (Section 6): "For distance d, we find the time to decode
d rounds of a distance d surface code circuit and divide this time by d."

So here: rounds = d, each basis is timed separately, time per round = decode time / (shots x d).
Only decoding is timed (sampling is excluded). Shots are sampled and decoded bit-packed in chunks
so a million shots fit in memory. PyMatching's decode_batch runs on one thread.

    python bench/m0_sparse_blossom_speed.py --distance 17 --p 0.001 --shots 1000000 --out results/m0_d17.json
"""
import argparse
import json
import platform
import time

import numpy as np
import pymatching
import stim


def circuit(basis, d, rounds, p):
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run(basis, d, rounds, p, shots, seed, chunk):
    c = circuit(basis, d, rounds, p)
    m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    sampler = c.compile_detector_sampler(seed=seed)
    warm, _ = sampler.sample(min(1000, shots), separate_observables=True, bit_packed=True)
    m.decode_batch(warm, bit_packed_shots=True, bit_packed_predictions=True)   # warm-up, not timed
    dt, errors, events, done = 0.0, 0, 0, 0
    while done < shots:
        n = min(chunk, shots - done)
        dets, obs = sampler.sample(n, separate_observables=True, bit_packed=True)
        t0 = time.perf_counter()
        pred = m.decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True)
        dt += time.perf_counter() - t0
        errors += int(np.any(pred != obs, axis=1).sum())
        events += int(np.unpackbits(dets, axis=1).sum())
        done += n
    return {
        "basis": basis,
        "detectors": c.num_detectors,
        "shots": shots,
        "decode_seconds": dt,
        "us_per_round": dt / (shots * rounds) * 1e6,
        "logical_errors": errors,
        "logical_error_rate": errors / shots,
        "mean_detection_events_per_shot": events / shots,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=17)
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--rounds", type=int, default=None, help="default: equal to the distance, as in the paper")
    ap.add_argument("--shots", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rounds = a.rounds or a.distance
    res = [run(b, a.distance, rounds, a.p, a.shots, a.seed + i, a.chunk) for i, b in enumerate("xz")]
    out = {
        "claim": "< 1 us per round in each of the X and Z bases (mean 0.62 us), d=17, p=0.1%, "
                 "rounds=d, 1M shots, one core of an Apple M1 Max",
        "source": "Higgott & Gidney, Sparse Blossom, arXiv:2303.15933, Section 6",
        "distance": a.distance, "p": a.p, "rounds": rounds, "shots_per_basis": a.shots,
        "us_per_round": {r["basis"]: r["us_per_round"] for r in res},
        "per_basis": res,
        "versions": {"stim": stim.__version__, "pymatching": pymatching.__version__,
                     "numpy": np.__version__, "python": platform.python_version()},
        "machine": {"cpu": cpu_name(), "system": platform.platform()},
    }
    print(json.dumps(out, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

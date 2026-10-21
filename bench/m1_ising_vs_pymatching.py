"""M1: NVIDIA Ising pre-decoders against PyMatching, uncorrelated and correlated, on identical samples.

Claim checked (Chamberland et al., "Fast and accurate AI-based pre-decoders for surface codes",
arXiv:2604.12841): an Ising pre-decoder followed by uncorrelated PyMatching lowers the logical
error rate, and the larger model outperforms *correlated* PyMatching up to distance 13.

Inputs come from NVIDIA's own pipeline (github.com/NVIDIA/Ising-Decoding), unmodified config:
  - samples_{X,Z}.dets + metadata_{X,Z}.json from WORKFLOW=generate_stim_data
  - per-shot arrays from WORKFLOW=inference with PREDECODER_DECODE_OUTPUT_DIR set:
      {basis}_observables.npy, {basis}_pymatching_predictions.npy           (pymatching_only)
      {basis}_predecoder_residual_detectors.npy,
      {basis}_ising_decoding_pymatching_predictions.npy                      (ising_decoding_pymatching)

This script rebuilds the exact Stim circuit with NVIDIA's own circuit and noise-model code (loaded
straight from their files, so no GPU or torch is needed), then on the same shots decodes with:
  1. uncorrelated PyMatching on the raw detectors   (must match NVIDIA's saved baseline bit for bit)
  2. correlated PyMatching on the raw detectors
  3. uncorrelated PyMatching on the pre-decoder residual (must match NVIDIA's saved Ising predictions)
  4. correlated PyMatching on the pre-decoder residual
and reports the logical error rate of each with 95% Wilson intervals, plus CPU decode time per shot.

    python bench/m1_ising_vs_pymatching.py --ising-repo Ising-Decoding \
        --samples outputs/d7/stim_samples --decoded fast=outputs/d7/fast_npy accurate=outputs/d7/acc_npy \
        --out results/m1_d7.json
"""
import argparse
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pymatching
import stim


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wilson(errors, shots, z=1.96):
    if shots == 0:
        return None, None
    p = errors / shots
    den = 1 + z * z / shots
    centre = (p + z * z / (2 * shots)) / den
    half = z * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(pred, obs):
    errors = int(np.any(pred != obs, axis=1).sum())
    lo, hi = wilson(errors, len(obs))
    return {"errors": errors, "shots": int(len(obs)), "ler": errors / len(obs), "ci95": [lo, hi]}


def build_circuit(repo, meta):
    code = Path(repo) / "code" / "qec"
    mc = load_module(code / "surface_code" / "memory_circuit.py", "ising_memory_circuit")
    nm_mod = load_module(code / "noise_model.py", "ising_noise_model")
    params = meta.get("noise_model_params")
    noise = nm_mod.NoiseModel.from_config_dict(dict(params)) if params else None
    p = float(noise.get_max_probability()) if noise is not None else float(meta["p_error"])
    rotation = str(meta["code_rotation"]).upper()
    rotation = {"O1": "XV", "O2": "XH", "O3": "ZV", "O4": "ZH"}.get(rotation, rotation)
    circ = mc.MemoryCircuit(distance=int(meta["distance"]), idle_error=p, sqgate_error=p, tqgate_error=p,
                            spam_error=(2.0 / 3.0) * p, n_rounds=int(meta["n_rounds"]),
                            basis=str(meta["basis"]).upper(), code_rotation=rotation,
                            noise_model=noise, add_boundary_detectors=True)
    circ.set_error_rates()
    c = circ.stim_circuit
    assert c.num_detectors == int(meta["num_detectors"]), (c.num_detectors, meta["num_detectors"])
    return c


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def evaluate_basis(repo, samples_dir, decoded, basis):
    samples_dir = Path(samples_dir)
    meta = json.loads((samples_dir / f"metadata_{basis}.json").read_text())
    c = build_circuit(repo, meta)
    dem = c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    n_det, n_obs = dem.num_detectors, dem.num_observables
    dets, obs = stim.read_shot_data_file(path=str(samples_dir / f"samples_{basis}.dets"), format="dets",
                                         num_detectors=n_det, num_observables=n_obs, separate_observables=True)
    dets = dets.astype(np.uint8)
    obs = obs.astype(np.uint8)
    m_unc = pymatching.Matching.from_detector_error_model(dem)
    m_cor = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)

    res = {"basis": basis, "metadata": meta, "detectors": n_det, "decoders": {}, "checks": {}}
    pred, dt = timed(lambda: m_unc.decode_batch(dets))
    res["decoders"]["pymatching"] = {**rate(pred, obs), "cpu_seconds_per_shot": dt / len(obs)}
    base_pred = pred
    pred, dt = timed(lambda: m_cor.decode_batch(dets, enable_correlations=True))
    res["decoders"]["pymatching_correlated"] = {**rate(pred, obs), "cpu_seconds_per_shot": dt / len(obs)}

    for label, d in decoded.items():
        d = Path(d)
        their_obs = np.load(d / f"{basis}_observables.npy").astype(np.uint8).reshape(obs.shape)
        res["checks"][f"{label}_observables_identical"] = bool(np.array_equal(their_obs, obs))
        their_base = d / f"{basis}_pymatching_predictions.npy"
        if their_base.exists():
            tb = np.load(their_base).astype(np.uint8).reshape(obs.shape)
            res["checks"][f"{label}_pymatching_baseline_identical"] = bool(np.array_equal(tb, base_pred))
        resid_path = d / f"{basis}_predecoder_residual_detectors.npy"
        if not resid_path.exists():
            continue
        resid = np.load(resid_path).astype(np.uint8)
        assert resid.shape == dets.shape, (resid.shape, dets.shape)
        res["checks"][f"{label}_mean_detection_events"] = {"raw": float(dets.sum(1).mean()),
                                                           "residual": float(resid.sum(1).mean())}
        pred, dt = timed(lambda: m_unc.decode_batch(resid))
        res["decoders"][f"ising_{label}+pymatching"] = {**rate(pred, obs), "cpu_seconds_per_shot": dt / len(obs)}
        their = d / f"{basis}_ising_decoding_pymatching_predictions.npy"
        if their.exists():
            tp = np.load(their).astype(np.uint8).reshape(obs.shape)
            res["checks"][f"{label}_ising_predictions_identical"] = bool(np.array_equal(tp, pred))
        pred, dt = timed(lambda: m_cor.decode_batch(resid, enable_correlations=True))
        res["decoders"][f"ising_{label}+pymatching_correlated"] = {**rate(pred, obs),
                                                                   "cpu_seconds_per_shot": dt / len(obs)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ising-repo", required=True)
    ap.add_argument("--samples", required=True, help="directory with samples_{X,Z}.dets and metadata_{X,Z}.json")
    ap.add_argument("--decoded", nargs="*", default=[], help="label=dir of NVIDIA's saved per-shot arrays")
    ap.add_argument("--bases", default="XZ")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    decoded = dict(x.split("=", 1) for x in a.decoded)
    out = {"source": "arXiv:2604.12841; github.com/NVIDIA/Ising-Decoding",
           "bases": [evaluate_basis(a.ising_repo, a.samples, decoded, b) for b in a.bases],
           "versions": {"stim": stim.__version__, "pymatching": pymatching.__version__,
                        "numpy": np.__version__, "python": platform.python_version()},
           "machine": {"cpu": platform.processor() or platform.machine(), "system": platform.platform()}}
    for b in out["bases"]:
        print(f"basis {b['basis']}  detectors {b['detectors']}  checks {b['checks']}")
        for name, r in b["decoders"].items():
            lo, hi = r["ci95"]
            print(f"  {name:38s} LER {r['ler']:.3e} [{lo:.2e}, {hi:.2e}]  errors {r['errors']:6d}/{r['shots']}"
                  f"  {r['cpu_seconds_per_shot'] * 1e6:8.1f} us/shot")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

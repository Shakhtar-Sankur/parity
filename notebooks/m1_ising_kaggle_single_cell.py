# parity M1: NVIDIA Ising pre-decoders vs PyMatching (uncorrelated and correlated).
# Kaggle settings: Accelerator = GPU T4, Internet = On, secret HF_TOKEN attached.
# Paste this whole cell into a new Kaggle notebook and run it.


DISTANCES = [5, 7, 9, 11, 13]      # NVIDIA's claim: beats correlated PyMatching "up to distance-13"
SHOTS = 262144                      # per basis; NVIDIA's own default test size
MODELS = {
    "fast": ("nvidia/Ising-Decoder-SurfaceCode-1-Fast",
             "ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors"),
    "accurate": ("nvidia/Ising-Decoder-SurfaceCode-1-Accurate",
                 "ising_decoder_surface_code_1_accurate_r13_v1.0.86_fp16.safetensors"),
}


import os, subprocess, sys, json, glob, shutil, time
W = "/kaggle/working"
os.chdir(W)
def sh(cmd, env=None, log=None):
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run(cmd, shell=True, env=e, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        open(log, "w").write(out)
    if p.returncode != 0:
        print(out[-4000:])
        raise RuntimeError(f"failed ({p.returncode}): {cmd}")
    return out

print(sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv"))
if not os.path.isdir("Ising-Decoding"):
    sh("git clone -q https://github.com/NVIDIA/Ising-Decoding")
if not os.path.isdir("parity"):
    sh("git clone -q https://github.com/Shakhtar-Sankur/parity")
ISING_COMMIT = sh("git -C Ising-Decoding rev-parse HEAD").strip()
PARITY_COMMIT = sh("git -C parity rev-parse HEAD").strip()
print("Ising-Decoding", ISING_COMMIT, "| parity", PARITY_COMMIT)


# NVIDIA's inference requirements, keeping Kaggle's preinstalled CUDA build of torch.
req = [l for l in open("Ising-Decoding/code/requirements_public_inference.txt").read().splitlines()
       if l.strip() and not l.startswith("#") and l.strip() != "torch"]
sh(f"{sys.executable} -m pip install -q " + " ".join(f"'{r}'" for r in req) + " huggingface_hub pyyaml")
print(sh(f"{sys.executable} -c 'import torch, stim, pymatching; print(torch.__version__, torch.cuda.is_available(), stim.__version__, pymatching.__version__)'"))


from kaggle_secrets import UserSecretsClient
from huggingface_hub import hf_hub_download
token = UserSecretsClient().get_secret("HF_TOKEN")
CKPT = {}
for label, (repo, fname) in MODELS.items():
    CKPT[label] = hf_hub_download(repo_id=repo, filename=fname, local_dir=f"{W}/Ising-Decoding/models", token=token)
    print(label, CKPT[label], os.path.getsize(CKPT[label]), "bytes")
del token


BASE = {"GPUS": "1", "CUDA_VISIBLE_DEVICES": "0", "PREDECODER_INFERENCE_NUM_WORKERS": "0",
        "PREDECODER_EMIT_INFERENCE_SUMMARY": "1", "PYTHONUNBUFFERED": "1"}
RUN = "bash code/scripts/local_run.sh"

def run_distance(d, shots):
    exp = f"parity_d{d}"
    env = {**BASE, "DISTANCE": str(d), "N_ROUNDS": str(d), "EXPERIMENT_NAME": exp,
           "PREDECODER_INFERENCE_NUM_SAMPLES": str(shots)}
    t0 = time.time()
    sh(f"cd Ising-Decoding && WORKFLOW=generate_stim_data {RUN}", env, log=f"{W}/logs/d{d}_generate.log")
    found = [f for f in glob.glob(f"{W}/Ising-Decoding/outputs/**/stim_samples/samples_X.dets", recursive=True)
             if exp in f and os.path.getmtime(f) >= t0]
    assert found, sh(f"find {W}/Ising-Decoding/outputs -maxdepth 3 | head -50")
    samples = os.path.dirname(max(found, key=os.path.getmtime))
    print("samples:", samples)
    decoded = []
    for label, ckpt in CKPT.items():
        outdir = f"{W}/decoded/d{d}_{label}"
        shutil.rmtree(outdir, ignore_errors=True)
        sh(f"cd Ising-Decoding && WORKFLOW=inference {RUN}",
           {**env, "PREDECODER_STIM_SAMPLES_DIR": samples, "PREDECODER_DECODE_MODE": "ising_decoding_pymatching",
            "PREDECODER_SAFETENSORS_CHECKPOINT": ckpt, "PREDECODER_DECODE_OUTPUT_DIR": outdir},
           log=f"{W}/logs/d{d}_{label}_inference.log")
        decoded.append(f"{label}={outdir}")
    os.makedirs(f"{W}/results", exist_ok=True)
    print(sh(f"{sys.executable} parity/bench/m1_ising_vs_pymatching.py --ising-repo Ising-Decoding "
             f"--samples {samples} --decoded {' '.join(decoded)} --out {W}/results/m1_d{d}.json"))
    print(f"d={d} done in {time.time() - t0:.0f} s")

# Smoke test first: d=5 with few shots, to catch setup problems in a minute rather than an hour.
run_distance(5, 8192)


for d in DISTANCES:
    run_distance(d, SHOTS)


# Pack results, NVIDIA's own logs (they contain its [Inference Summary] lines) and run provenance.
info = {"ising_commit": ISING_COMMIT, "parity_commit": PARITY_COMMIT, "distances": DISTANCES, "shots": SHOTS,
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").strip(),
        "cpu": sh("lscpu | grep 'Model name'").strip(),
        "pip": sh(f"{sys.executable} -m pip freeze | grep -iE 'torch|stim|pymatching|numpy|ldpc|beliefmatching'")}
json.dump(info, open(f"{W}/results/m1_provenance.json", "w"), indent=2)
import zipfile
with zipfile.ZipFile(f"{W}/parity_m1_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{W}/results/*") + glob.glob(f"{W}/logs/*"):
        z.write(f, os.path.relpath(f, W))
print(open(f"{W}/results/m1_provenance.json").read())
print("Download parity_m1_results.zip from the output panel.")

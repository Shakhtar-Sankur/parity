# parity M1 re-score: run in the SAME Kaggle session after the main cell has finished.
# Uses the decoded files already in /kaggle/working/decoded; no GPU work is repeated.
import glob, json, os, re, subprocess, sys, zipfile
W = "/kaggle/working"
os.chdir(W)
missing = [p for p in ("parity", "Ising-Decoding", "decoded") if not os.path.isdir(f"{W}/{p}")]
if missing:
    raise SystemExit(f"This session no longer has {missing} (Kaggle clears /kaggle/working when a "
                     "session ends). Run the main cell, m1_ising_kaggle_single_cell.py, instead: "
                     "it already includes the corrected scoring.")
subprocess.run("git -C parity pull -q", shell=True, check=True)
print(subprocess.run("git -C parity log -1 --format=%h", shell=True, capture_output=True, text=True).stdout)

for d in [5, 7, 9, 11, 13]:
    samples = f"{W}/Ising-Decoding/outputs/parity_d{d}/stim_samples"
    decoded = " ".join(f"{label}={W}/decoded/d{d}_{label}" for label in ("fast", "accurate"))
    out = subprocess.run(
        f"{sys.executable} parity/bench/m1_ising_vs_pymatching.py --ising-repo Ising-Decoding "
        f"--samples {samples} --decoded {decoded} --out {W}/results/m1_d{d}.json",
        shell=True, capture_output=True, text=True)
    print(f"===== d={d} =====")
    print(out.stdout[-6000:] or out.stderr[-3000:])

# NVIDIA's own logged LER for each run: the independent check on the Ising + PyMatching numbers.
pattern = re.compile(r"(logical error ra\w+[^\n]*|\[Inference Summary\][^\n]*)", re.IGNORECASE)
nvidia = {}
for log in sorted(glob.glob(f"{W}/logs/d*_inference.log")):
    lines = pattern.findall(open(log, errors="replace").read())
    nvidia[os.path.basename(log)] = lines[-12:]
    print(os.path.basename(log)); print("\n".join(lines[-12:]))
json.dump(nvidia, open(f"{W}/results/m1_nvidia_logged_ler.json", "w"), indent=2)

with zipfile.ZipFile(f"{W}/parity_m1_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{W}/results/*") + glob.glob(f"{W}/logs/*"):
        z.write(f, os.path.relpath(f, W))
print("Download parity_m1_results.zip from the output panel.")

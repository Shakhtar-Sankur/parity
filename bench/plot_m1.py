"""Plot results/m1/*.json: logical error rate against code distance for each decoder.

Errors and shots are pooled over the X and Z bases; whiskers are 95% Wilson intervals.
Writes a light and a dark SVG to results/m1/ for the README.

    python bench/plot_m1.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "m1"
DISTANCES = [5, 7, 9, 11, 13]
SERIES = [  # decoder key, label, marker
    ("pymatching", "PyMatching", "o"),
    ("pymatching_correlated", "Correlated PyMatching", "s"),
    ("ising_fast+pymatching", "Ising Fast + PyMatching", "^"),
    ("ising_accurate+pymatching", "Ising Accurate + PyMatching", "D"),
]
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#1f1f1e", "muted": "#6b6b67", "grid": "#e6e5e1",
              "colors": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "ink": "#ededea", "muted": "#a3a39e", "grid": "#33332f",
             "colors": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}


def wilson(errors, shots, z=1.96):
    p = errors / shots
    den = 1 + z * z / shots
    centre = (p + z * z / (2 * shots)) / den
    half = z * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / den
    return p, centre - half, centre + half


def pooled():
    table = {}
    for d in DISTANCES:
        bases = json.loads((OUT / f"m1_d{d}.json").read_text())["bases"]
        for key, _, _ in SERIES:
            errors = sum(b["decoders"][key]["errors"] for b in bases)
            shots = sum(b["decoders"][key]["shots"] for b in bases)
            table[(key, d)] = wilson(errors, shots)
    return table


def main():
    table = pooled()
    for name, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=t["surface"])
        ax.set_facecolor(t["surface"])
        for (key, label, marker), color in zip(SERIES, t["colors"]):
            ler = [table[(key, d)][0] for d in DISTANCES]
            err = [[table[(key, d)][0] - table[(key, d)][1] for d in DISTANCES],
                   [table[(key, d)][2] - table[(key, d)][0] for d in DISTANCES]]
            ax.errorbar(DISTANCES, ler, yerr=err, color=color, lw=2, marker=marker, ms=6,
                        mec=t["surface"], mew=1, elinewidth=1, capsize=0, label=label)
            ax.annotate(label, (DISTANCES[-1], ler[-1]), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=9, color=t["ink"])
        ax.set_yscale("log")
        ax.set_xticks(DISTANCES)
        ax.set_xlim(4.5, 17.5)
        ax.minorticks_off()
        ax.set_xlabel("code distance d (rounds = d)", color=t["muted"])
        ax.set_ylabel("logical error rate per shot", color=t["muted"])
        ax.grid(True, color=t["grid"], lw=0.8)
        ax.tick_params(colors=t["muted"], length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        leg = ax.legend(frameon=False, loc="lower left", fontsize=9)
        for txt in leg.get_texts():
            txt.set_color(t["ink"])
        fig.suptitle("Correlated PyMatching alone beats Ising + PyMatching from d = 7",
                     color=t["ink"], fontsize=12, x=0.01, ha="left")
        ax.set_title("Surface code, circuit-level depolarising noise p = 0.003, X and Z pooled, 95% intervals",
                     color=t["muted"], fontsize=9, loc="left")
        fig.tight_layout()
        fig.savefig(OUT / f"m1_ler_vs_distance_{name}.svg", facecolor=t["surface"])
        plt.close(fig)
    for d in DISTANCES:
        print(d, "  ".join(f"{label}: {table[(key, d)][0]:.2e}" for key, label, _ in SERIES))


if __name__ == "__main__":
    main()

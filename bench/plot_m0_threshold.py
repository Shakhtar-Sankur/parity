"""Plot results/m0_threshold.json: logical error rate vs physical error rate, one panel per basis.

Writes a light and a dark SVG to results/ for the README.

    python bench/plot_m0_threshold.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#1f1f1e", "muted": "#6b6b67", "grid": "#e6e5e1",
              "ramp": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]},
    "dark": {"surface": "#1a1a19", "ink": "#ededea", "muted": "#a3a39e", "grid": "#33332f",
             "ramp": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]},
}


def main():
    data = json.loads((ROOT / "results" / "m0_threshold.json").read_text())
    distances = data["distances"]
    for name, t in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=t["surface"])
        for ax, basis in zip(axes, "xz"):
            ax.set_facecolor(t["surface"])
            for color, d in zip(t["ramp"], distances):
                pts = sorted((r for r in data["points"] if r["basis"] == basis and r["d"] == d),
                             key=lambda r: r["p"])
                ps = [r["p"] * 100 for r in pts]
                ler = [r["logical_error_rate"] for r in pts]
                err = [[r["logical_error_rate"] - r["ci_low"] for r in pts],
                       [r["ci_high"] - r["logical_error_rate"] for r in pts]]
                ax.errorbar(ps, ler, yerr=err, color=color, lw=2, marker="o", ms=5,
                            mec=t["surface"], mew=1, elinewidth=1, capsize=0, label=f"d = {d}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([0.3, 0.5, 0.7, 1.0, 1.2])
            ax.set_xticklabels(["0.3", "0.5", "0.7", "1.0", "1.2"])
            ax.minorticks_off()
            ax.set_title(f"{basis.upper()} basis", color=t["ink"], fontsize=11, loc="left")
            ax.set_xlabel("physical error rate p (%)", color=t["muted"])
            ax.grid(True, color=t["grid"], lw=0.8)
            ax.tick_params(colors=t["muted"], length=0)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0].set_ylabel("logical error rate (per shot, d rounds)", color=t["muted"])
        leg = axes[1].legend(frameon=False, loc="lower right", fontsize=9)
        for txt in leg.get_texts():
            txt.set_color(t["ink"])
        fig.suptitle("Rotated surface code under MWPM (PyMatching 2): curves cross near p ≈ 0.7%",
                     color=t["ink"], fontsize=12, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(ROOT / "results" / f"m0_threshold_{name}.svg", facecolor=t["surface"])
        plt.close(fig)


if __name__ == "__main__":
    main()

"""Plot the results of benchmarks/throughput.py as assets/throughput-{light,dark}.png.

    python benchmarks/plot_throughput.py benchmarks/results_h100.jsonl
"""
import json
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

THEMES = {   # "old" is a recessive gray, "new" the accent; their lightness differs in both modes
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  base="#c3c2b7", old="#c3c2b7", new="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 base="#383835", old="#5a5955", new="#3987e5"),
}
MODELS = [("deberta-v3-base", "(a) DeBERTaV3-base, r = 8"), ("llama3-8b", "(b) LLaMA3-8B, r = 32")]
METHODS = [("RefLoRA", "reflora-per-layer", "reflora"), ("RefLoRA-S", "reflora-s-per-layer", "reflora-s")]
DPI = 200
BAR_PX, GAP_PX, RADIUS_PX = 24, 2, 4    # bar width, surface gap between paired bars, rounded data-end (at 1x)
YMAX = 112


def rounded_bar(ax, x, width, height, color, px_x, px_y):
    '''Bar with a rounded data-end and a square baseline; px_x, px_y are rendered px per data unit.'''
    rx, ry = RADIUS_PX * DPI / 100 / px_x, RADIUS_PX * DPI / 100 / px_y
    ax.add_patch(FancyBboxPatch((x - width / 2, 0), width, height, boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x - width / 2, 0), width, max(height - ry, 0), facecolor=color, edgecolor="none",
                           zorder=3))


def plot(results, theme, path):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=DPI, facecolor=t["surface"])
    fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.13, wspace=0.16)

    for ax, (model, title) in zip(axes, MODELS):
        lora = results[(model, "lora")]["steps_per_s"]
        ax.set_facecolor(t["surface"])
        ax.set_xlim(-0.5, len(METHODS) - 0.5)
        ax.set_ylim(0, YMAX)
        fig.canvas.draw()
        bbox = ax.get_window_extent()
        px_x, px_y = bbox.width / len(METHODS), bbox.height / YMAX
        width, gap = BAR_PX * DPI / 100 / px_x, GAP_PX * DPI / 100 / px_x

        for i, (label, old, new) in enumerate(METHODS):
            for dx, key, color in [(-(width + gap) / 2, old, t["old"]), ((width + gap) / 2, new, t["new"])]:
                rel = 100 * results[(model, key)]["steps_per_s"] / lora
                rounded_bar(ax, i + dx, width, rel, color, px_x, px_y)
                ax.text(i + dx, rel + 2.5, f"{rel:.0f}%", ha="center", va="bottom", color=t["ink2"], fontsize=8,
                        zorder=4, bbox=dict(facecolor=t["surface"], edgecolor="none", pad=0.8))    # halo over the LoRA line

        ax.axhline(100, color=t["muted"], linewidth=0.8, zorder=2)
        ax.text(len(METHODS) - 0.52, 101, "LoRA", ha="right", va="bottom", color=t["muted"], fontsize=8)
        ax.set_xticks(range(len(METHODS)), [m[0] for m in METHODS], color=t["ink2"])
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], color=t["muted"], fontsize=8)
        ax.tick_params(length=0, pad=4)
        ax.grid(axis="y", color=t["grid"], linewidth=0.6, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["base"])
        ax.spines["bottom"].set_linewidth(0.8)
        ax.set_title(title, color=t["ink"], fontsize=9.5, pad=6)
    axes[0].set_ylabel("Throughput (% of LoRA)", color=t["ink2"], fontsize=9)

    handles = [Patch(facecolor=t["old"], label="Layer-by-layer refactoring"),
               Patch(facecolor=t["new"], label="Batched refactoring (new)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=2, frameon=False,
               labelcolor=t["ink2"], fontsize=8.5, handlelength=1.0, handleheight=1.0, columnspacing=1.5)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def main():
    with open(sys.argv[1]) as f:
        rows = [json.loads(line) for line in f if line.startswith("{")]
    runs = defaultdict(list)    # median over repeated runs
    for r in rows:
        runs[(r["model"], r["method"])].append(r)
    results = {key: dict(rs[0], steps_per_s=statistics.median(r["steps_per_s"] for r in rs))
               for key, rs in runs.items()}
    for theme in THEMES:
        plot(results, theme, f"assets/throughput-{theme}.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot SGLang serving generation throughput at 8K context on A40."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


OUT_DIR = Path(__file__).resolve().parent / "throughput_results" / "figures"

QWEN = {
    "title": "Qwen, A40, 8K",
    "concurrency": [1, 2, 4, 8, 16, 24, 32],
    "flash": [21.18, 39.90, 73.00, 125.85, 125.85, 125.85, 125.85],
    "ditto": [18.81, 36.68, 71.10, 137.80, 242.80, 346.13, 448.00],
}

LLAMA = {
    "title": "Llama, A40, 8K",
    "concurrency": [1, 2, 4, 8, 16, 24, 32, 48, 64],
    "flash": [38.68, 70.86, 127.80, 213.64, 311.28, 311.28, 311.28, 311.28, 311.28],
    "ditto": [34.52, 66.59, 129.53, 249.45, 438.40, 635.14, 793.98, 1070.95, 1226.63],
}


def nice_number(value, _pos):
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def plot_panel(ax, data):
    x = data["concurrency"]
    flash = data["flash"]
    ditto = data["ditto"]

    ax.plot(
        x,
        flash,
        label="FullAttn-CuGraph",
        color="#2B6CB0",
        marker="o",
        markersize=6.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        linewidth=2.7,
    )
    ax.plot(
        x,
        ditto,
        label="Litecache",
        color="#E85D04",
        marker="s",
        markersize=6.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        linewidth=2.7,
    )

    ax.set_title(data["title"], fontsize=16, fontweight="semibold", pad=12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.yaxis.set_major_formatter(FuncFormatter(nice_number))
    ax.grid(True, which="major", axis="both", color="#9AA4B2", alpha=0.22, linewidth=0.9)
    ax.grid(True, which="minor", axis="x", color="#9AA4B2", alpha=0.10, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="both", labelsize=12, colors="#1F2937")

    ymax = max(max(flash), max(ditto))
    ax.set_ylim(0, ymax * 1.20)
    ax.margins(x=0.08)

    speedup = ditto[-1] / flash[-1]
    ax.annotate(
        f"{speedup:.2f}x",
        xy=(x[-1], ditto[-1]),
        xytext=(-34, 24),
        textcoords="offset points",
        color="#9A3412",
        fontsize=13,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#E85D04",
            "lw": 1.4,
            "shrinkA": 2,
            "shrinkB": 5,
        },
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.12",
            "facecolor": "#FFF7ED",
            "edgecolor": "#FDBA74",
            "linewidth": 0.8,
        },
    )

    ax.annotate(
        f"{ditto[-1]:.0f}",
        xy=(x[-1], ditto[-1]),
        xytext=(7, -2),
        textcoords="offset points",
        va="center",
        color="#9A3412",
        fontsize=11,
    )
    ax.annotate(
        f"{flash[-1]:.0f}",
        xy=(x[-1], flash[-1]),
        xytext=(7, -2),
        textcoords="offset points",
        va="center",
        color="#1E3A8A",
        fontsize=11,
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#111827",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
    plot_panel(axes[0], QWEN)
    plot_panel(axes[1], LLAMA)

    fig.suptitle(
        "SGLang Serving Generation Throughput",
        fontsize=21,
        fontweight="bold",
        y=1.04,
    )
    fig.supxlabel("Concurrency (requests, log2 scale)", fontsize=14, y=-0.02)
    fig.supylabel("Generation throughput (tokens/s)", fontsize=14, x=-0.015)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=2.4,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "sglang_serving_gen_throughput_fullattn_cugraph_a40_8k.png"
    pdf_path = OUT_DIR / "sglang_serving_gen_throughput_fullattn_cugraph_a40_8k.pdf"
    svg_path = OUT_DIR / "sglang_serving_gen_throughput_fullattn_cugraph_a40_8k.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)
    print(svg_path)


if __name__ == "__main__":
    main()

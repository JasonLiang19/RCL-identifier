import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path


# -------------------------------------------------
# Resolve project root (directory containing results/)
# -------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # adjust if your structure differs
RESULTS_DIR = PROJECT_ROOT / "analysis/results"    # where your .txt files live
FIGURES_DIR = RESULTS_DIR / "figures"    # output folder
FIGURES_DIR.mkdir(exist_ok=True)

CSV_FILES = {
    "Test – Residue Level (F1)": "test_residue_level_evaluation.csv",
    "Test – Sequence Level (Exact Match %)": "test_sequence_level_evaluation.csv",
    "Val – Residue Level (F1)": "val_residue_level_evaluation.csv",
    "Val – Sequence Level (Exact Match %)": "val_sequence_level_evaluation.csv",
}

# Metric to plot for each file
METRIC = {
    "Test – Residue Level (F1)": "f1",
    "Test – Sequence Level (Exact Match %)": "exact_match_pct",
    "Val – Residue Level (F1)": "f1",
    "Val – Sequence Level (Exact Match %)": "exact_match_pct",
}

# Human-readable y-labels
YLABEL = {
    "f1": "F1 Score",
    "exact_match_pct": "Exact Match (%)",
}

# Colors by architecture
COLORS = {
    "CNN":  "#b8b8b8",  # gray
    "UNET": "#1a80bb",  # blue
    "LSTM": "#ea801c",  # orange
}

# Force ordering
ENCODING_ORDER = ["onehot", "blosum", "esm2_650m"]
MODEL_ORDER = ["CNN", "UNET", "LSTM"]

# -----------------------------
#  PLOTTING HELPERS
# -----------------------------
def make_pivot(df, metric):
    df["encoding"] = pd.Categorical(df["encoding"], ENCODING_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df.pivot(index="encoding", columns="model", values=metric)

def add_subplot(ax, pivot, title, ylabel):
    bar_colors = [COLORS[m] for m in pivot.columns]

    pivot.plot(
        kind="bar",
        ax=ax,
        width=0.75,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Encoding", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# -----------------------------
#  CREATE 2×2 FIGURE
# -----------------------------
def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    handles, labels = None, None  # capture legend once

    for i, (title, filename) in enumerate(CSV_FILES.items()):
        df = pd.read_csv(RESULTS_DIR / filename)
        metric = METRIC[title]
        ylabel = YLABEL[metric]

        pivot = make_pivot(df, metric)
        add_subplot(axes[i], pivot, title, ylabel)

        # get legend items once
        if handles is None:
            handles, labels = axes[i].get_legend_handles_labels()

        axes[i].legend().remove()   # remove subplot legends

    # -------------------------
    # SINGLE GLOBAL LEGEND
    # -------------------------
    fig.legend(
        handles,
        labels,
        title="Architecture",
        loc="upper center",
        ncol=3,
        fontsize=12,
        title_fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03)
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outfile = FIGURES_DIR / "all_results_combined.png"
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"Saved combined figure to: {outfile}")

if __name__ == "__main__":
    main()
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


# -------------------------------------------------
# Plotting helper
# -------------------------------------------------

def plot_grouped_bars(df, metric, title, outfile, ylabel):
    """
    Create grouped bar plots with bars grouped by encoding and CNN/UNET/LSTM side-by-side.
    Legend is moved outside the plot and encodings are ordered.
    """

    COLORS = {
    "CNN":   "#b8b8b8",  # same as onehot
    "UNET":  "#1a80bb",  # blue
    "LSTM":  "#ea801c",  # orange
    }

    # Force encoding ordering
    encoding_order = ["onehot", "blosum", "esm2_650m"]
    df["encoding"] = pd.Categorical(df["encoding"], encoding_order, ordered=True)

    # Force architecture ordering
    architectures = ["CNN", "UNET", "LSTM"]
    df["model"] = pd.Categorical(df["model"], architectures, ordered=True)

    # Pivot: rows = encoding, columns = model
    pivot = df.pivot(index="encoding", columns="model", values=metric)

    plt.figure(figsize=(10, 6))

    bar_colors = [COLORS[m] for m in pivot.columns]
    # Main bar plot
    ax = pivot.plot(kind="bar", width=0.75, color=bar_colors, linewidth=.8)

    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel("Encoding", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)

    # Move legend outside the plot
    plt.legend(
        title="Architecture",
        fontsize=12,
        title_fontsize=12,
        bbox_to_anchor=(1.02, 1),  # push outside the right side
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")  # ensure legend isn't cut off
    plt.close()
    print(f"Saved: {outfile}")


# -------------------------------------------------
# Main script
# -------------------------------------------------

def main():

    files = {
        RESULTS_DIR / "test_residue_level_evaluation.csv": {
            "metric": "f1",
            "title": "Test Set — Residue-Level F1",
            "outfile": FIGURES_DIR / "test_residue_level.png",
            "ylabel": "F1"
        },
        RESULTS_DIR / "test_sequence_level_evaluation.csv": {
            "metric": "exact_match",
            "title": "Test Set — Sequence-Level Exact Match %",
            "outfile": FIGURES_DIR / "test_sequence_level.png",
            "ylabel": "Percentage of Perfectly Annotated RCLs"
        },
        RESULTS_DIR / "val_residue_level_evaluation.csv": {
            "metric": "f1",
            "title": "Validation Set — Residue-Level F1",
            "outfile": FIGURES_DIR / "val_residue_level.png",
            "ylabel": "F1"
        },
        RESULTS_DIR / "val_sequence_level_evaluation.csv": {
            "metric": "exact_match",
            "title": "Validation Set — Sequence-Level Exact Match %",
            "outfile": FIGURES_DIR / "val_sequence_level.png",
            "ylabel": "Percentage of Perfectly Annotated RCLs"
        },
    }

    for fname, info in files.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)

        # Handle both residue-level and sequence-level
        metric = info["metric"]
        if metric not in df.columns:
            # sequence-level files use "exact_match_pct" or similar
            for col in df.columns:
                if "match" in col.lower():
                    metric = col
                    break

        plot_grouped_bars(
            df,
            metric,
            info["title"],
            info["outfile"],
            info["ylabel"]
        )


if __name__ == "__main__":
    main()
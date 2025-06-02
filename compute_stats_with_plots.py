#!/usr/bin/env python3
"""
compute_stats.py (with automatic sign fix for physics energies for proper AUC ROC calculations)

The script computes, for every (target, pipeline) pair in a virtual-screening CSV file:

• ROC-AUC for a user-selectable score column
• Pearson / Spearman / Kendall correlations
    – ml_affinity_scorer   vs experimental_pIC50
    – physic_affinity_scorer (raw sign) vs experimental_pIC50
• Optional multi-page PDF with ROC curves

Sign convention:
    • For ml-type scores “higher = better”.
    • For physics energies “lower = better”.
The script therefore flips the sign of physic_affinity_scorer *only for*
ROC/AUC calculations and ROC plots; correlations keep the original sign.

Usage
-----
python compute_stats_with_plots.py <dataset>.csv
python compute_stats_with_plots.py <dataset>.csv --score_column ml_affinity_scorer --roc_pdf <output>.pdf
python compute_stats_with_plots.py <dataset>.csv --score_column physic_affinity_scorer --roc_pdf <output>.pdf

Developed by Dr. Serhii Vakal, Turku, Finland (June 2025).
"""

from __future__ import annotations

# head-less backend -----------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import argparse, sys, itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ─────────────────────────── CLI ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="compute_stats.py",
        description="Per-target / per-pipeline statistics for VS dataset")
    p.add_argument("csv_file", type=Path, help="Input CSV file")
    p.add_argument(
        "--score_column",
        default="ml_affinity_scorer",
        choices=["ml_affinity_scorer", "physic_affinity_scorer", "ml_pose_scorer"],
        help="Column used for ROC-AUC & ROC plots (default: ml_affinity_scorer)",
    )
    p.add_argument("--roc_pdf", type=Path,
                   help="Write ROC curves for every target to this PDF")
    return p.parse_args()

# ─────────────────────────── helpers ────────────────────────────────────────
def _safe_corr(func, x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return func(x, y)[0]

def correlations(x: pd.Series, y: pd.Series) -> dict[str, float]:
    xv, yv = x.to_numpy(), y.to_numpy()
    return {
        "pearson": _safe_corr(pearsonr, xv, yv),
        "spearman": _safe_corr(spearmanr, xv, yv),
        "kendall":  _safe_corr(kendalltau, xv, yv),
    }

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {
        "target", "pipeline",
        "is_experimentally_active", "experimental_pIC50",
        "ml_affinity_scorer", "physic_affinity_scorer"
    }
    miss = need - set(df.columns)
    if miss:
        sys.exit(f"❌  Missing column(s): {', '.join(sorted(miss))}")
    return df

# ROC helpers ----------------------------------------------------------------
PIPE_COLORS = {"P1": "#1f77b4", "P2": "#ff7f0e", "P3": "#2ca02c"}

def _prep_for_auc(series: pd.Series, score_col: str) -> pd.Series:
    """
    Return a score series with the correct orientation for ROC:
    higher value => more likely active.
    For physics energies we flip the sign.
    """
    return -series if score_col == "physic_affinity_scorer" else series

def plot_roc_for_target(df: pd.DataFrame, target: str, score_col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls="--", color="0.7", zorder=1)

    for pipe in sorted(df.pipeline.unique()):
        sel = df[(df.target == target) & (df.pipeline == pipe)]
        y_true = sel["is_experimentally_active"].astype(int)
        if y_true.nunique() < 2:
            continue
        y_score = _prep_for_auc(sel[score_col], score_col)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc  = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr,
                label=f"{pipe}  (AUC={auc:.3f})",
                lw=1.8,
                color=PIPE_COLORS.get(pipe, None))
    ax.set(title=f"Target {target}", xlabel="False-Positive Rate",
           ylabel="True-Positive Rate", xlim=(0,1), ylim=(0,1))
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig

# ─────────────────────────── main ───────────────────────────────────────────
def main() -> None:
    args = parse_args()
    df   = load_csv(args.csv_file)

    rows = []
    for (tgt, pipe), grp in df.groupby(["target", "pipeline"]):
        row = dict(target=tgt, pipeline=pipe, n_compounds=len(grp))

        # ---------- ROC-AUC (sign-aware) ---------
        y_true  = grp["is_experimentally_active"].astype(int)
        if y_true.nunique() < 2:
            row["roc_auc"] = np.nan
            row["auc_note"] = "one class only"
        else:
            y_score = _prep_for_auc(grp[args.score_column], args.score_column)
            row["roc_auc"] = roc_auc_score(y_true, y_score)
            row["auc_note"] = ""

        # ---------- Correlations (no sign change) ------------
        for tag, series in [("ml",   grp["ml_affinity_scorer"]),
                            ("phys", grp["physic_affinity_scorer"])]:
            for kind, val in correlations(series, grp["experimental_pIC50"]).items():
                row[f"{tag}_{kind}"] = val

        rows.append(row)

    res = pd.DataFrame(rows).sort_values(["target", "pipeline"])
    num_cols = res.select_dtypes("number").columns
    ffmt = {c: (lambda x: " NA" if pd.isna(x) else f"{x:6.3f}") for c in num_cols}
    print(res.to_string(index=False, formatters=ffmt))

    # -------- ROC PDF --------------------------
    if args.roc_pdf:
        with PdfPages(args.roc_pdf) as pdf:
            for tgt in sorted(df.target.unique()):
                fig = plot_roc_for_target(df, tgt, args.score_column)
                pdf.savefig(fig)
                plt.close(fig)
        print(f"✅  ROC curves written to {args.roc_pdf}")

if __name__ == "__main__":
    main()
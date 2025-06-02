#!/usr/bin/env python3
"""
pose_stereo_ml.py
The script to estimate the joint impact of pose quality & stereoisomer choice on the *ml_affinity_scorer* screening & ranking performance.

Usage:
------
python pose_stereo_ml.py <dataset>.csv           # default threshold = 0.6
python pose_stereo_ml.py <dataset>.csv --thr 0.7 # stricter

Developed by Dr. Serhii Vakal, Turku, Finland (June 2025).
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr, kendalltau


# ───────────────────────────── helpers ──────────────────────────────────────
def _safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    try:
        return (
            pearsonr(x, y)[0],
            spearmanr(x, y)[0],
            kendalltau(x, y)[0],
        )
    except Exception:
        return (np.nan, np.nan, np.nan)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df["is_experimentally_active"].astype(int)
    s = df["ml_affinity_scorer"]

    # ROC-AUC
    try:
        auc = roc_auc_score(y, s)
    except ValueError:
        auc = np.nan

    p, sp, k = _safe_corr(s, df["experimental_pIC50"])
    return dict(n=len(df), auc=auc, pear=p, spear=sp, kend=k)


def _stereo_reduce(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode:
        RAW   – keep every stereoisomer row
        BEST  – highest ml_affinity_scorer per compound_ID
        WORST – lowest  "
        MEAN  – average ml_affinity_scorer over isomers
    """
    if mode == "RAW":
        return df

    g = df.groupby(["target", "pipeline", "compound_ID"])

    if mode == "MEAN":
        return (
            g.agg(
                {
                    "ml_affinity_scorer": "mean",
                    "ml_pose_scorer": "max",
                    "experimental_pIC50": "first",
                    "is_experimentally_active": "first",
                }
            )
            .reset_index()
        )

    idx = (
        g["ml_affinity_scorer"]
        .idxmax() if mode == "BEST" else g["ml_affinity_scorer"].idxmin()
    )
    return df.loc[idx].reset_index(drop=True)


def _summarise(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    rows = []
    for (tgt, pipe), grp in df.groupby(["target", "pipeline"]):
        for pose_tag, pose_df in [
            ("HQ", grp[grp.ml_pose_scorer >= thr]),
            ("LQ", grp[grp.ml_pose_scorer < thr]),
        ]:
            for stereo_tag in ["BEST", "WORST", "MEAN", "RAW"]:
                sub = _stereo_reduce(pose_df, stereo_tag)
                row = dict(target=tgt, pipeline=pipe, pose=pose_tag, stereo=stereo_tag)
                row.update(_metrics(sub))
                rows.append(row)
    return pd.DataFrame(rows)


# ───────────────────────────── CLI / Main ───────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--thr", type=float, default=0.6, help="pose-quality threshold")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    res = _summarise(df, args.thr).sort_values(
        ["target", "pipeline", "pose", "stereo"]
    )

    ff = lambda x: " NA" if pd.isna(x) else f"{x:6.3f}"
    print(
        res.to_string(
            index=False,
            formatters={c: ff for c in ["auc", "pear", "spear", "kend"]},
        )
    )


if __name__ == "__main__":
    sys.exit(main())
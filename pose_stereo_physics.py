#!/usr/bin/env python3
"""
pose_stereo_physics.py
A script for the two-factor analysis for physics scorer

Usage
-----
python pose_stereo_physics.py <dataset>.csv --thr 0.6

Developed by Dr. Serhii Vakal, Turku, Finland (June 2025).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr, kendalltau

# Functions
def corr(x, y):
    try:
        return pearsonr(x, y)[0], spearmanr(x, y)[0], kendalltau(x, y)[0]
    except Exception:
        return (np.nan, np.nan, np.nan)

def metrics(df):
    y = df["is_experimentally_active"].astype(int)
    s = df["physic_affinity_scorer"]
    try:
        auc = roc_auc_score(y, s)
    except ValueError:
        auc = np.nan
    p, sp, k = corr(s, df["experimental_pIC50"])
    return dict(n=len(df), auc=auc, pear=p, spear=sp, kend=k)

def best_iso(df, mode):
    g = df.groupby(["target","pipeline","compound_ID"])
    if mode == "RAW":           return df
    if mode == "BEST":          idx = g["physic_affinity_scorer"].idxmin()  # lower ΔG = better
    elif mode == "WORST":       idx = g["physic_affinity_scorer"].idxmax()
    elif mode == "MEAN":
        return g.agg({
            "physic_affinity_scorer":"mean",
            "ml_pose_scorer":"max",
            "experimental_pIC50":"first",
            "is_experimentally_active":"first"
        }).reset_index()
    return df.loc[idx].reset_index(drop=True)

def summarise(df, thr):
    rows=[]
    for (tgt,pipe), grp in df.groupby(["target","pipeline"]):
        for pose_tag, pose_df in [("HQ", grp[grp.ml_pose_scorer>=thr]),
                                  ("LQ", grp[grp.ml_pose_scorer<thr])]:
            for iso_tag in ["BEST","WORST","MEAN","RAW"]:
                sub = best_iso(pose_df, iso_tag)
                row = dict(target=tgt, pipeline=pipe,
                           pose=pose_tag, stereo=iso_tag)
                row.update(metrics(sub))
                rows.append(row)
    return pd.DataFrame(rows)

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--thr", type=float, default=0.6)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    res = summarise(df, args.thr).sort_values(["target","pipeline","pose","stereo"])
    ff = lambda x: " NA" if pd.isna(x) else f"{x:6.3f}"
    print(res.to_string(index=False,
          formatters={c:ff for c in ["auc","pear","spear","kend"]}))
if __name__ == "__main__":
    sys.exit(main())
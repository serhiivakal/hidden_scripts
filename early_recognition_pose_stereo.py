#!/usr/bin/env python3
"""
early_recognition_pose_stereo.py
Calculate early recognition metrics (BEDROC and RIE) for virtual screening pipelines,
filtering for high-quality poses and best stereoisomers.

Usage:
------
python early_recognition_pose_stereo.py <dataset>.csv --alpha 160.9 --pose_thr 0.6

Developed based on the screening analysis framework by Dr. Serhii Vakal, Turku, 2025.
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
from math import exp

def calculate_bedroc_rie(df, alpha=160.9):
    """
    Calculate BEDROC (Boltzmann-Enhanced Discrimination of ROC) and RIE 
    (Robust Initial Enhancement) at given alpha value.
    
    References:
    Truchon & Bayly. (2007). Journal of Chemical Information and Modeling, 47(2), 488-508.
    """
    # Sort by ML affinity score (higher is better)
    df_sorted = df.sort_values('ml_affinity_scorer', ascending=False).reset_index(drop=True)
    
    # Total number of compounds and actives
    n = len(df_sorted)
    actives_mask = df_sorted['is_experimentally_active'] == True
    n_actives = actives_mask.sum()
    
    if n_actives == 0 or n == 0 or n_actives == n:
        return np.nan, np.nan
    
    # Calculate RIE
    ranks = np.arange(n)[actives_mask]  # Get ranks of actives (0-based)
    sum_exp = np.sum(np.exp(-alpha * ranks / n))
    
    ra = n_actives / n  # Ratio of actives
    
    # Random distribution probability
    random_sum = n_actives * (1 - np.exp(-alpha)) / (n * (1 - np.exp(-alpha/n)))
    rie = sum_exp / random_sum
    
    # Calculate BEDROC
    numerator = sum_exp - (n_actives * (1 - np.exp(-alpha/n)) / (n * (1 - np.exp(-alpha))))
    denominator = ra * (1 - np.exp(-alpha)) - (1 - np.exp(-alpha*ra)) * (1 - np.exp(-alpha/n)) / (1 - np.exp(-alpha))
    
    # Calculate BEDROC with correct scaling
    bedroc = numerator / denominator
    
    # Normalize BEDROC to [0,1]
    bedroc = (bedroc - 1/ra) / (1 / ra * (1 - 1/ra))
    
    return bedroc, rie

def best_pose_per_compound(df):
    """Select the pose with highest ml_affinity_scorer for each compound."""
    return df.loc[df.groupby(['target', 'pipeline', 'compound_ID'])['ml_affinity_scorer'].idxmax()]

def main():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Calculate early recognition metrics")
    ap.add_argument("csv", type=Path, help="Input CSV file")
    ap.add_argument("--alpha", type=float, default=160.9, 
                    help="Alpha parameter for BEDROC/RIE calculation (default: 160.9)")
    ap.add_argument("--pose_thr", type=float, default=0.6,
                    help="Threshold for ml_pose_scorer to filter good poses (default: 0.6)")
    args = ap.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv)
    
    # Initialize results for both filtered and unfiltered data
    results = []
    
    # Process all data first (for comparison)
    df_best_all = best_pose_per_compound(df)
    
    # Then filter for good poses
    df_good_poses = df[df['ml_pose_scorer'] >= args.pose_thr]
    df_best_good = best_pose_per_compound(df_good_poses)
    
    # Calculate metrics for each target and pipeline
    for (target, pipeline), group in df_best_all.groupby(['target', 'pipeline']):
        # Get corresponding filtered group
        filtered_group = df_best_good[(df_best_good['target'] == target) & 
                                     (df_best_good['pipeline'] == pipeline)]
        
        # All poses (for comparison)
        row_all = {
            'target': target, 
            'pipeline': pipeline,
            'filter': 'none',
            'n_compounds': len(group),
            'n_actives': group['is_experimentally_active'].sum(),
            'active_ratio': group['is_experimentally_active'].sum() / len(group)
        }
        
        # Good poses only
        row_filtered = {
            'target': target, 
            'pipeline': pipeline,
            'filter': f'pose>{args.pose_thr}',
            'n_compounds': len(filtered_group),
            'n_actives': filtered_group['is_experimentally_active'].sum(),
            'active_ratio': filtered_group['is_experimentally_active'].sum() / max(1, len(filtered_group))
        }
        
        # Calculate metrics for all poses
        bedroc_all, rie_all = calculate_bedroc_rie(group, args.alpha)
        row_all['BEDROC'] = bedroc_all
        row_all['RIE'] = rie_all
        
        # Calculate metrics for filtered poses
        if len(filtered_group) > 0:
            bedroc_filtered, rie_filtered = calculate_bedroc_rie(filtered_group, args.alpha)
            row_filtered['BEDROC'] = bedroc_filtered
            row_filtered['RIE'] = rie_filtered
        else:
            row_filtered['BEDROC'] = np.nan
            row_filtered['RIE'] = np.nan
        
        results.append(row_all)
        results.append(row_filtered)
    
    # Create results dataframe and sort
    results_df = pd.DataFrame(results).sort_values(['target', 'pipeline', 'filter'])
    
    # Format and print results
    ff = lambda x: " NA" if pd.isna(x) else f"{x:6.3f}"
    ff_counts = lambda x: f"{x:5d}" if pd.notna(x) else "    0"
    formatters = {
        'active_ratio': ff, 
        'BEDROC': ff, 
        'RIE': ff,
        'n_compounds': ff_counts,
        'n_actives': ff_counts
    }
    
    print(f"Early recognition metrics (alpha={args.alpha}, pose threshold={args.pose_thr}):")
    print(results_df.to_string(index=False, formatters=formatters))
    
if __name__ == "__main__":
    sys.exit(main())
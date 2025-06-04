#!/usr/bin/env python3
"""
early_enrichment.py
Calculate early enrichment factors for virtual screening pipelines.

Usage:
------
python early_enrichment.py <dataset>.csv --ef_percent 1,5,10

Developed based on the screening analysis framework by Dr. Serhii Vakal, Orion Pharma, Turku, Finland, 2025.
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

def calculate_ef(df, percent):
    """
    Calculate Enrichment Factor at given percentage of ranked database.
    
    EF = (Hits found in top X% / Total hits) / (X% / 100%)
    """
    # Sort by ML affinity score (higher is better)
    df_sorted = df.sort_values('ml_affinity_scorer', ascending=False)
    
    # Total number of compounds and actives
    n_total = len(df)
    n_actives = df['is_experimentally_active'].sum()
    
    if n_actives == 0:
        return np.nan
    
    # Number of compounds to consider for the top X%
    n_top = int(n_total * percent / 100)
    if n_top == 0:
        n_top = 1  # Ensure at least one compound is selected
    
    # Number of actives in top X%
    n_actives_top = df_sorted.iloc[:n_top]['is_experimentally_active'].sum()
    
    # Calculate enrichment factor
    ef = (n_actives_top / n_actives) / (percent / 100)
    
    return ef

def best_pose_per_compound(df):
    """Select the pose with highest ml_affinity_scorer for each compound."""
    return df.loc[df.groupby(['target', 'pipeline', 'compound_ID'])['ml_affinity_scorer'].idxmax()]

def main():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Calculate early enrichment factors")
    ap.add_argument("csv", type=Path, help="Input CSV file")
    ap.add_argument("--ef_percent", type=str, default="1,5,10", 
                    help="Comma-separated percentages for EF calculation (default: 1,5,10)")
    args = ap.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv)
    
    # Get percentages for EF calculation
    percentages = [float(x) for x in args.ef_percent.split(',')]
    
    # Select best pose per compound
    df_best = best_pose_per_compound(df)
    
    # Initialize results
    results = []
    
    # Calculate EF for each target and pipeline
    for (target, pipeline), group in df_best.groupby(['target', 'pipeline']):
        row = {'target': target, 'pipeline': pipeline}
        
        # Add number of compounds and actives
        row['n_compounds'] = len(group)
        row['n_actives'] = group['is_experimentally_active'].sum()
        
        # Calculate EF at different percentages
        for percent in percentages:
            ef = calculate_ef(group, percent)
            row[f'EF{percent:.1f}%'] = ef
        
        results.append(row)
    
    # Create results dataframe and sort
    results_df = pd.DataFrame(results).sort_values(['target', 'pipeline'])
    
    # Format and print results
    ff = lambda x: " NA" if pd.isna(x) else f"{x:6.2f}"
    ef_cols = [f'EF{p:.1f}%' for p in percentages]
    formatters = {col: ff for col in ef_cols}
    
    print(results_df.to_string(index=False, formatters=formatters))
    
if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phân tích submission.csv - kết quả dự đoán
Author: Võ Thị Diễm Thanh & Lê Đăng Hoàng Tuấn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Config
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def load_submission(path):
    """Load submission CSV"""
    print(f"\n{'='*60}")
    print("📊 LOADING SUBMISSION FILE")
    print(f"{'='*60}")
    
    df = pd.read_csv(path)
    print(f"✓ Loaded: {len(df):,} records")
    print(f"✓ Columns: {list(df.columns)}")
    
    return df

def analyze_statistics(df):
    """Thống kê cơ bản"""
    print(f"\n{'='*60}")
    print("📈 BASIC STATISTICS")
    print(f"{'='*60}")
    
    if 'probability_helpful' in df.columns:
        col = 'probability_helpful'
        print("\n[Probability Mode]")
    elif 'predicted_helpful' in df.columns:
        col = 'predicted_helpful'
        print("\n[Binary Mode]")
    else:
        raise ValueError("Unknown column format!")
    
    stats = {
        "Total Records": len(df),
        "Unique review_ids": df['review_id'].nunique(),
        "Duplicates": len(df) - df['review_id'].nunique(),
        "Mean": df[col].mean(),
        "Median": df[col].median(),
        "Std": df[col].std(),
        "Min": df[col].min(),
        "Max": df[col].max(),
        "Q1 (25%)": df[col].quantile(0.25),
        "Q3 (75%)": df[col].quantile(0.75)
    }
    
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.6f}")
        else:
            print(f"  {key:20s}: {val:,}")
    
    # Binary mode: count predictions
    if col == 'predicted_helpful':
        print(f"\n[Prediction Distribution]")
        counts = df[col].value_counts().sort_index()
        for val, count in counts.items():
            pct = count / len(df) * 100
            print(f"  Class {val}: {count:,} ({pct:.2f}%)")
    
    # Check for issues
    print(f"\n[Data Quality Checks]")
    if df['review_id'].duplicated().any():
        dups = df[df['review_id'].duplicated(keep=False)]
        print(f"  ⚠️  WARNING: {len(dups):,} duplicate review_ids!")
        print(f"  Example duplicates:\n{dups.head(10)}")
    else:
        print(f"  ✓ No duplicate review_ids")
    
    if col == 'probability_helpful':
        out_of_range = df[(df[col] < 0) | (df[col] > 1)]
        if len(out_of_range) > 0:
            print(f"  ⚠️  WARNING: {len(out_of_range)} values out of [0,1] range!")
        else:
            print(f"  ✓ All probabilities in [0, 1]")
    
    return stats, col

def plot_distribution(df, col, out_path):
    """Vẽ phân phối predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histogram
    ax1 = axes[0, 0]
    if col == 'probability_helpful':
        ax1.hist(df[col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Probability', fontsize=12, fontweight='bold')
    else:
        counts = df[col].value_counts().sort_index()
        ax1.bar(counts.index, counts.values, color=['#e74c3c', '#2ecc71'], 
                edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Not Helpful (0)', 'Helpful (1)'])
    
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Predictions', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add count labels
    if col == 'predicted_helpful':
        for i, (val, count) in enumerate(counts.items()):
            pct = count / len(df) * 100
            ax1.text(i, count, f'{count:,}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie Chart (Binary only or binned probability)
    ax2 = axes[0, 1]
    if col == 'probability_helpful':
        # Bin probabilities
        bins = [0, 0.3, 0.5, 0.7, 1.0]
        labels = ['Low (0-0.3)', 'Medium-Low (0.3-0.5)', 
                  'Medium-High (0.5-0.7)', 'High (0.7-1.0)']
        df['prob_bin'] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        counts = df['prob_bin'].value_counts()
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
    else:
        counts = df[col].value_counts().sort_index()
        labels = ['Not Helpful', 'Helpful']
        colors = ['#e74c3c', '#2ecc71']
    
    wedges, texts, autotexts = ax2.pie(counts.values, labels=counts.index, 
                                         autopct='%1.1f%%', startangle=90,
                                         colors=colors, explode=[0.05]*len(counts),
                                         shadow=True, textprops={'fontsize': 11})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax2.set_title('Proportion of Predictions', fontsize=14, fontweight='bold')
    
    # 3. Boxplot
    ax3 = axes[1, 0]
    if col == 'probability_helpful':
        bp = ax3.boxplot([df[col]], vert=True, patch_artist=True,
                         labels=['Probability'], widths=0.5)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_edgecolor('black')
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)
        
        # Add statistics annotations
        stats_text = f"Mean: {df[col].mean():.4f}\n"
        stats_text += f"Median: {df[col].median():.4f}\n"
        stats_text += f"Std: {df[col].std():.4f}"
        ax3.text(0.5, 0.02, stats_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # For binary, show value counts as bar
        counts = df[col].value_counts().sort_index()
        bars = ax3.bar(range(len(counts)), counts.values, 
                       color=['#e74c3c', '#2ecc71'], alpha=0.7)
        ax3.set_xticks(range(len(counts)))
        ax3.set_xticklabels(['Not Helpful', 'Helpful'])
        
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax3.set_title('Box Plot / Value Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative Distribution (for probability only)
    ax4 = axes[1, 1]
    if col == 'probability_helpful':
        sorted_vals = np.sort(df[col])
        cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax4.plot(sorted_vals, cumulative, linewidth=2, color='steelblue')
        ax4.axhline(0.5, color='red', linestyle='--', label='50th percentile')
        ax4.axvline(0.5, color='orange', linestyle='--', label='Threshold 0.5')
        ax4.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Cumulative Proportion', fontsize=12, fontweight='bold')
        ax4.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add percentile markers
        percentiles = [0.25, 0.5, 0.75, 0.9]
        for p in percentiles:
            val = df[col].quantile(p)
            ax4.plot(val, p, 'ro', markersize=8)
            ax4.annotate(f'{p*100:.0f}%: {val:.3f}', 
                        xy=(val, p), xytext=(10, 0),
                        textcoords='offset points', fontsize=9)
    else:
        # For binary: show predicted vs threshold comparison
        counts = df[col].value_counts().sort_index()
        ax4.bar(['Not Helpful\n(0)', 'Helpful\n(1)'], counts.values,
                color=['#e74c3c', '#2ecc71'], edgecolor='black', alpha=0.7)
        ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax4.set_title('Prediction Counts', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for i, (label, count) in enumerate(zip(['Not Helpful', 'Helpful'], counts.values)):
            pct = count / len(df) * 100
            ax4.text(i, count, f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Distribution plots saved to: {out_path}")
    plt.close()

def plot_simple_pie(df, col, out_path):
    """Vẽ biểu đồ tròn đơn giản (theo yêu cầu)"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if col == 'probability_helpful':
        # Bin probabilities
        bins = [0, 0.5, 1.0]
        labels = ['Not Helpful\n(prob < 0.5)', 'Helpful\n(prob ≥ 0.5)']
        df['binary_pred'] = (df[col] >= 0.5).astype(int)
        counts = df['binary_pred'].value_counts().sort_index()
    else:
        counts = df[col].value_counts().sort_index()
        labels = ['Not Helpful\n(Class 0)', 'Helpful\n(Class 1)']
    
    colors = ['#e74c3c', '#2ecc71']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(counts.values, labels=labels,
                                        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*counts.sum()):,})',
                                        startangle=90, colors=colors,
                                        explode=explode, shadow=True,
                                        textprops={'fontsize': 14, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')
    
    ax.set_title('Submission Predictions Distribution\n(Simple Pie Chart)', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Add legend with total count
    legend_labels = [f'{label.split(chr(10))[0]}: {count:,}' 
                     for label, count in zip(labels, counts.values)]
    ax.legend(legend_labels, loc='upper left', fontsize=12, 
              title=f'Total: {counts.sum():,} predictions', title_fontsize=13)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Simple pie chart saved to: {out_path}")
    plt.close()

def main():
    # Paths - Updated for V7 submissions
    import sys
    if len(sys.argv) > 1:
        submission_path = sys.argv[1]
    else:
        submission_path = "output_final/submission_v7.csv"  # Default to V7 baseline
    
    out_dir = Path("output_final/analysis")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Load
    df = load_submission(submission_path)
    
    # Analyze
    stats, col = analyze_statistics(df)
    
    # Plot distribution
    plot_distribution(df, col, out_dir / "submission_distribution.png")
    
    # Plot simple pie (theo yêu cầu)
    plot_simple_pie(df, col, out_dir / "submission_pie_chart.png")
    
    # Save statistics to JSON
    stats_json = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                  for k, v in stats.items()}
    
    import json
    with open(out_dir / "submission_statistics.json", "w") as f:
        json.dump(stats_json, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ ANALYSIS COMPLETED!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  1. {out_dir / 'submission_distribution.png'} (4 charts)")
    print(f"  2. {out_dir / 'submission_pie_chart.png'} (simple pie)")
    print(f"  3. {out_dir / 'submission_statistics.json'} (stats)")

if __name__ == "__main__":
    main()

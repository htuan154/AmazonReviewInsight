# code_v2/utils/evaluation_v2.py
# Author: Võ Thị Diễm Thanh (Features & Models)
# Day 3-7: Evaluation utilities với visualization
#
# Cải tiến so với V1:
# - More detailed metrics
# - Confusion matrix
# - Threshold optimization
# - Model comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, roc_auc_score, accuracy_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, f1_score, precision_score, recall_score
)

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate comprehensive metrics"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        "auc_pr": average_precision_score(y_true, y_pred_proba),
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold
    }
    
    return metrics

def find_optimal_threshold(y_true, y_pred_proba, metric="f1"):
    """Find optimal threshold based on metric"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = []
    
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append((t, score))
    
    best_threshold, best_score = max(scores, key=lambda x: x[1])
    
    print(f"[INFO] Optimal threshold for {metric}: {best_threshold:.2f} (score: {best_score:.4f})")
    
    return best_threshold, scores

def plot_confusion_matrix(y_true, y_pred, out_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Add percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1%})',
                    ha='center', va='center', color='gray', fontsize=10)
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Confusion matrix saved to {out_path}")
    else:
        plt.show()

def plot_pr_roc_curves(y_true, y_pred_proba, out_path=None):
    """Plot PR and ROC curves side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    
    ax1.plot(recall, precision, label=f'AUC-PR = {auc_pr:.4f}', linewidth=2)
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    ax2.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.4f}', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] PR/ROC curves saved to {out_path}")
    else:
        plt.show()

def plot_threshold_analysis(y_true, y_pred_proba, out_path=None):
    """Plot metrics vs threshold"""
    thresholds = np.arange(0.1, 0.9, 0.02)
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
    
    # Mark optimal F1
    best_idx = np.argmax(f1_scores)
    best_t = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.axvline(best_t, color='red', linestyle='--', alpha=0.5,
                label=f'Optimal (t={best_t:.2f}, F1={best_f1:.3f})')
    plt.scatter([best_t], [best_f1], color='red', s=100, zorder=5)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Threshold analysis saved to {out_path}")
    else:
        plt.show()
    
    return best_t, best_f1

def compare_models(results_dict, out_path=None):
    """Compare multiple models"""
    models = list(results_dict.keys())
    metrics = ["auc_pr", "auc_roc", "accuracy", "f1_score"]
    
    df = pd.DataFrame({
        model: [results_dict[model].get(m, 0) for m in metrics]
        for model in models
    }, index=metrics)
    
    print("\n[INFO] Model Comparison:")
    print(df.to_string())
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = df.loc[metric]
        bars = ax.bar(range(len(values)), values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(values)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom')
        
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').upper())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Model comparison saved to {out_path}")
    else:
        plt.show()
    
    return df

def print_classification_report(y_true, y_pred, threshold=0.5):
    """Print detailed classification report"""
    if isinstance(y_pred, np.ndarray) and y_pred.dtype == float:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred
    
    print("\n[INFO] Classification Report:")
    print(classification_report(y_true, y_pred_binary, 
                                target_names=['Not Helpful', 'Helpful']))

if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.rand(1000)
    
    print("=== Testing Evaluation Utils V2 ===\n")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred_proba)
    print("Metrics:", metrics)
    
    # Find optimal threshold
    opt_t, scores = find_optimal_threshold(y_true, y_pred_proba, metric="f1")
    
    # Print classification report
    print_classification_report(y_true, y_pred_proba, threshold=opt_t)

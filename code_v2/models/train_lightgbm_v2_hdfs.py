#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightGBM trainer that can read Parquet from:
- LOCAL paths (file or partitioned directory)
- HDFS via JNI (hdfs://host:port/...)  -> requires libhdfs (hdfs.dll on Windows)
- HDFS via WebHDFS (webhdfs://host:9870/...) -> NO libhdfs needed

Examples:
  python train_lightgbm_v2_hdfs.py --train webhdfs://localhost:9870/output_v2/features_train_v3 --out ./output_v2/lightgbm_v2_full_hdfs
  python train_lightgbm_v2_hdfs.py --train hdfs://localhost:9000/output_v2/features_train_v3 --out ./output_v2/lightgbm_v2_full_hdfs
"""

import argparse, os, sys, json, time
from typing import List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# JNI (hdfs://) still supported if libhdfs is available:
try:
    from pyarrow import fs as pafs
except Exception:
    pafs = None  # will disable JNI branch gracefully

# NEW: WebHDFS (no libhdfs needed)
from hdfs import InsecureClient

TARGET_COL = "is_helpful"

def log(m): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser("train_lightgbm_v2_hdfs")
    ap.add_argument("--train", required=True, help="Local dir/file, hdfs://..., or webhdfs://host:port/abs/path")
    ap.add_argument("--test", default=None, help="Optional test path (same formats as --train)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--boost_rounds", type=int, default=3000)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)
    ap.add_argument("--learning_rate", type=float, default=0.035)
    ap.add_argument("--num_leaves", type=int, default=128)
    ap.add_argument("--feature_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_freq", type=int, default=2)
    ap.add_argument("--min_child_samples", type=int, default=60)
    ap.add_argument("--max_bin", type=int, default=255)
    ap.add_argument("--bin_construct_sample_cnt", type=int, default=200000)
    ap.add_argument("--reg_alpha", type=float, default=0.1)
    ap.add_argument("--reg_lambda", type=float, default=0.2)
    ap.add_argument("--scale_pos_weight", default="auto")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--feature_cols", default=None, help="CSV list; if omitted, auto-detect numeric/bool cols minus target")
    return ap.parse_args()

# -------------------- WebHDFS helpers --------------------

def _is_webhdfs(p: str) -> bool:
    return p.startswith(("webhdfs://","webhdfs+http://","webhdfs+https://"))

def _parse_webhdfs(url: str):
    if url.startswith("webhdfs://"):
        scheme, rest = "http", url[len("webhdfs://"):]
    elif url.startswith("webhdfs+http://"):
        scheme, rest = "http", url[len("webhdfs+http://"):]
    elif url.startswith("webhdfs+https://"):
        scheme, rest = "https", url[len("webhdfs+https://"):]
    else:
        raise ValueError("Invalid WebHDFS scheme")
    host_port, _, path = rest.partition("/")
    base = f"{scheme}://{host_port}"
    if not path.startswith("/"): path = "/" + path
    return base, path

def _webhdfs_list_parquet_files(client: InsecureClient, path: str):
    files = []
    if path.lower().endswith(".parquet"):
        return [path]
    def walk(p):
        for name, st in client.list(p, status=True):
            full = p.rstrip("/") + "/" + name
            if st["type"] == "DIRECTORY":
                walk(full)
            elif name.lower().endswith(".parquet"):
                files.append(full)
    walk(path)
    if not files:
        raise FileNotFoundError(f"No .parquet under {path}")
    return sorted(files)

def _read_parquet_webhdfs(url: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    import pyarrow as pa
    base, path = _parse_webhdfs(url)
    client = InsecureClient(base)  # adjust for auth if needed
    parts = _webhdfs_list_parquet_files(client, path)
    tables = []
    for fp in parts:
        with client.read(fp) as rdr:
            data = rdr.read()
            tables.append(pq.read_table(pa.py_buffer(data), columns=columns))
    return pa.concat_tables(tables, promote=True).to_pandas()

# -------------------- JNI / Local helpers --------------------

def _get_fs_and_path(path: str):
    if path.startswith("hdfs://"):
        if pafs is None:
            raise RuntimeError("JNI HDFS requested but pyarrow.fs is unavailable. Use WebHDFS or install libhdfs.")
        parts = path.split("://",1)[1]
        host_port, _, subpath = parts.partition("/")
        if ":" in host_port:
            host, port = host_port.split(":",1); port = int(port)
        else:
            host, port = host_port, 8020
        fs = pafs.HadoopFileSystem(host=host, port=port)
        norm = "/" + subpath
        return fs, norm
    else:
        return ds.FileSystem.from_uri(os.path.abspath(path))[0], os.path.abspath(path)

def _downcast_mapper(arrow_type):
    if pa.types.is_int8(arrow_type) or pa.types.is_uint8(arrow_type):  return np.int16
    if pa.types.is_int16(arrow_type) or pa.types.is_uint16(arrow_type): return np.int16
    if pa.types.is_int32(arrow_type) or pa.types.is_uint32(arrow_type): return np.int32
    if pa.types.is_int64(arrow_type) or pa.types.is_uint64(arrow_type): return np.int64
    if pa.types.is_floating(arrow_type):                                return np.float32
    return None

def load_parquet_any(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if _is_webhdfs(path):
        log(f"Reading via WebHDFS: {path}")
        return _read_parquet_webhdfs(path, columns=columns)
    fs, norm_path = _get_fs_and_path(path)
    log(f"Reading via {type(fs).__name__}: {path}")
    dataset = ds.dataset(norm_path, filesystem=fs, format="parquet")
    table = dataset.to_table(columns=columns)
    return table.to_pandas(types_mapper=_downcast_mapper)

# -------------------- training utils --------------------

def detect_feature_columns(df: pd.DataFrame, target: str, specified: Optional[List[str]]):
    if specified: return [c for c in specified if c in df.columns and c != target]
    num = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return [c for c in num if c != target]

def to_efficient_dtypes(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns: continue
        dt = str(df[c].dtype)
        if dt.startswith("float"):      df[c] = df[c].astype("float32", copy=False)
        elif dt.startswith(("int","uint")): df[c] = df[c].astype("int32",  copy=False)
        elif dt == "bool":              df[c] = df[c].astype("uint8",  copy=False)
    return df

def compute_spw(y: np.ndarray) -> float:
    pos = float((y == 1).sum()); neg = float(len(y) - pos)
    return neg / max(pos, 1.0)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    yb = (y_true == 1).astype(int)
    ap  = average_precision_score(yb, y_prob)
    try: roc = roc_auc_score(yb, y_prob)
    except Exception: roc = None
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(yb, y_pred, labels=[0,1]).tolist()
    rep = classification_report(yb, y_pred, labels=[0,1], output_dict=True, zero_division=0)
    return {"average_precision": ap, "roc_auc": roc, "confusion_matrix": cm, "classification_report": rep}

def save_metrics(d: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    log(f"Saved metrics → {path}")

def save_feature_importance(model: lgb.Booster, feat: List[str], out_dir: str):
    gain  = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    df = pd.DataFrame({"feature": feat, "gain": gain, "split": split}).sort_values("gain", ascending=False)
    csv_path = os.path.join(out_dir, "feature_importance.csv"); df.to_csv(csv_path, index=False); log(f"Saved → {csv_path}")
    topn = min(40, len(df)); plt.figure(figsize=(10, max(4, topn*0.25)))
    plt.barh(df["feature"].head(topn)[::-1], df["gain"].head(topn)[::-1]); plt.xlabel("Gain"); plt.title("Top Feature Importance (gain)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150); plt.close()

def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str, prefix="val"):
    yb = (y_true == 1).astype(int)
    p,r,_ = precision_recall_curve(yb, y_prob)
    plt.figure(figsize=(6,5)); plt.plot(r,p); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR ({prefix})"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_pr_curve.png"), dpi=150); plt.close()
    try:
        fpr,tpr,_ = roc_curve(yb, y_prob)
        plt.figure(figsize=(6,5)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({prefix})"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_roc_curve.png"), dpi=150); plt.close()
    except Exception as e:
        log(f"ROC skipped: {e}")

def main():
    args = parse_args()
    ensure_dir(args.out)

    # Load train
    train_df = load_parquet_any(args.train)
    if TARGET_COL not in train_df.columns:
        log(f"FATAL: target '{TARGET_COL}' not found. First columns: {list(train_df.columns)[:30]}")
        sys.exit(2)

    # Pick features
    specified = [c.strip() for c in args.feature_cols.split(",")] if args.feature_cols else None
    feat = detect_feature_columns(train_df, TARGET_COL, specified)
    train_df[feat] = train_df[feat].fillna(0)
    train_df = to_efficient_dtypes(train_df, feat)

    y = train_df[TARGET_COL].values
    X = train_df[feat]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=args.val_ratio, random_state=42, stratify=y)
    log(f"Train shape: {X_tr.shape}; Val shape: {X_val.shape}")

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "min_child_samples": args.min_child_samples,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "bin_construct_sample_cnt": args.bin_construct_sample_cnt,
        "verbose": -1,
        "num_threads": args.threads,
    }
    if args.gpu:
        params.update({"device": "gpu"})

    spw = ( (len(y_tr) - float((y_tr==1).sum())) / max(float((y_tr==1).sum()), 1.0) ) \
          if (isinstance(args.scale_pos_weight, str) and args.scale_pos_weight=="auto") \
          else float(args.scale_pos_weight)
    params["scale_pos_weight"] = spw
    log(f"scale_pos_weight = {spw:.4f}")

    dtrain = lgb.Dataset(X_tr.values, label=y_tr, feature_name=feat, free_raw_data=False)
    dval   = lgb.Dataset(X_val.values, label=y_val, feature_name=feat, free_raw_data=False)

    model = lgb.train(params, dtrain, num_boost_round=args.boost_rounds,
                      valid_sets=[dtrain, dval], valid_names=["train","val"],
                      callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
                                 lgb.log_evaluation(period=200)])

    # Validation
    val_prob = model.predict(X_val.values, num_iteration=model.best_iteration)
    metrics = compute_metrics(y_val, val_prob)
    save_metrics(metrics, os.path.join(args.out, "metrics_val.json"))
    save_feature_importance(model, feat, args.out)
    plot_curves(y_val, val_prob, args.out, prefix="val")

    # Optional test
    if args.test:
        test_df = load_parquet_any(args.test)
        if TARGET_COL in test_df.columns:
            test_df[feat] = test_df[feat].fillna(0)
            test_df = to_efficient_dtypes(test_df, feat)
            y_test = test_df[TARGET_COL].values
            X_test = test_df[feat].values
            test_prob = model.predict(X_test, num_iteration=model.best_iteration)
            save_metrics(compute_metrics(y_test, test_prob), os.path.join(args.out, "metrics_test.json"))
            plot_curves(y_test, test_prob, args.out, prefix="test")
        else:
            log(f"WARNING: target '{TARGET_COL}' not found in test; skipping test metrics.")

    # Save model + manifest
    model.save_model(os.path.join(args.out, "model.txt"), num_iteration=model.best_iteration)
    with open(os.path.join(args.out, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"train": args.train, "test": args.test, "out": args.out,
                   "features": feat, "best_iteration": int(model.best_iteration)}, f, indent=2, ensure_ascii=False)
    log("Done.")

if __name__ == "__main__":
    main()

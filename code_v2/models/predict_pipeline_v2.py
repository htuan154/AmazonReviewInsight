# code_v2/models/predict_pipeline_v2.py
# V2.1 — add --mode {binary,proba}, --threshold
# Author: Lê Đăng Hoàng Tuấn

import argparse, json
from pathlib import Path
import pandas as pd
import lightgbm as lgb

def parse_args():
    ap = argparse.ArgumentParser("predict_pipeline_v2")
    ap.add_argument("--test", required=True, help="Parquet path (local)")
    ap.add_argument("--model", required=True, help="LightGBM model.txt")
    ap.add_argument("--features", required=True, help="run_manifest.json (chứa 'features')")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--batch_size", type=int, default=200_000)
    ap.add_argument("--mode", choices=["binary","proba"], default="binary",
                    help="binary → xuất predicted_helpful 0/1; proba → probability_helpful 0..1")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="ngưỡng cho nhị phân (chỉ dùng khi --mode=binary)")
    return ap.parse_args()

def _predict_in_batches(model, X, bs):
    n = len(X)
    out = []
    for i in range(0, n, bs):
        out.extend(model.predict(X.iloc[i:i+bs], num_iteration=model.best_iteration))
    return out

def main():
    args = parse_args()

    # 1) load model
    model = lgb.Booster(model_file=args.model)

    # 2) load feature list
    with open(args.features, "r", encoding="utf-8") as f:
        feats = json.load(f)["features"]

    # 3) load test
    df = pd.read_parquet(args.test)

    # 4) validate + fillna
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột features trong test: {missing[:10]}...")
    if "review_id" not in df.columns:
        raise ValueError("Thiếu cột 'review_id' trong test")
    X = df[feats].copy().fillna(0)
    rid = df["review_id"]

    # 5) predict
    proba = _predict_in_batches(model, X, args.batch_size)

    # 6) build submission theo mode
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "binary":
        pred = (pd.Series(proba) >= args.threshold).astype("int8")
        sub = pd.DataFrame({"review_id": rid, "predicted_helpful": pred})
    else:
        sub = pd.DataFrame({"review_id": rid, "probability_helpful": proba})

    sub.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()

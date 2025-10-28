# list_fields.py
# Liệt kê toàn bộ tên trường (keys) trong JSONL/JSONL.GZ (top-level hoặc nested)
# Hỗ trợ Windows path, auto-detect .jsonl/.jsonl.gz, giới hạn số dòng để test nhanh.

import json, gzip, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import argparse

def open_maybe_gzip(path: Path):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="ignore")
    return open(p, "r", encoding="utf-8", errors="ignore")

def collect_keys(path: Path, top_level_only: bool = True, max_lines: int | None = None):
    keys = Counter()
    types = defaultdict(set)

    def add_obj(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                name = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                keys[name] += 1
                types[name].add(type(v).__name__)
                if not top_level_only and isinstance(v, (dict, list)):
                    add_obj(v, name)  # đệ quy cho nested
        elif isinstance(obj, list):
            # list rất dài thì chỉ duyệt một ít phần tử đầu
            for v in obj[:50]:
                add_obj(v, prefix)

    total = 0
    with open_maybe_gzip(path) as f:
        for i, line in enumerate(f, 1):
            if max_lines and i > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                add_obj(obj)
                total += 1
            except json.JSONDecodeError:
                # bỏ qua dòng lỗi để tiếp tục
                continue

    print(f"\n=== {path.name} ===")
    print(f"Records scanned: {total:,}")
    all_keys = sorted(keys.keys())
    for k in all_keys:
        t = ",".join(sorted(types[k]))
        print(f"{k}  |  seen={keys[k]:,}  |  types={t}")
    
    # Print first 3 records
    print(f"\n--- First 3 records from {path.name} ---")
    with open_maybe_gzip(path) as f:
        for i, line in enumerate(f, 1):
            if i > 3:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                print(f"\nRecord {i}:")
                print(json.dumps(obj, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"\nRecord {i}: [JSON decode error]")

def resolve_inputs(root: Path) -> list[Path]:
    """
    Tự động tìm 2 file trong data/raw:
      - Movies_and_TV.jsonl(.gz)
      - meta_Movies_and_TV.jsonl(.gz)
    Nếu không thấy, báo lỗi rõ ràng.
    """
    data_raw = root / "data" / "raw"
    candidates = [
        "Movies_and_TV.jsonl", "Movies_and_TV.jsonl.gz",
        "meta_Movies_and_TV.jsonl", "meta_Movies_and_TV.jsonl.gz",
    ]
    files = []
    for name in candidates:
        p = data_raw / name
        if p.exists():
            files.append(p)

    # Ưu tiên đúng 2 file (1 review + 1 meta)
    wanted = {
        "Movies_and_TV.jsonl": None,
        "Movies_and_TV.jsonl.gz": None,
        "meta_Movies_and_TV.jsonl": None,
        "meta_Movies_and_TV.jsonl.gz": None,
    }
    for p in files:
        if p.name in wanted:
            wanted[p.name] = p

    resolved = []
    # pick review (jsonl.gz nếu có, else jsonl)
    review = wanted["Movies_and_TV.jsonl.gz"] or wanted["Movies_and_TV.jsonl"]
    meta   = wanted["meta_Movies_and_TV.jsonl.gz"] or wanted["meta_Movies_and_TV.jsonl"]

    if review is None or meta is None:
        raise FileNotFoundError(
            f"Không tìm thấy đủ file trong {data_raw}.\n"
            f"Yêu cầu: Movies_and_TV.jsonl(.gz) và meta_Movies_and_TV.jsonl(.gz)\n"
            f"Đã thấy: {[str(p) for p in files]}"
        )
    resolved.extend([review, meta])
    return resolved

def main():
    parser = argparse.ArgumentParser(
        description="Liệt kê tên trường (keys) từ JSONL/JSONL.GZ (top-level hoặc nested)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Thư mục gốc repo (mặc định: cha của thư mục chứa file này)."
    )
    parser.add_argument(
        "--nested",
        action="store_true",
        help="Bật quét nested keys (đệ quy). Mặc định chỉ top-level."
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Giới hạn số dòng đọc (ví dụ 100000 để test nhanh). Mặc định đọc hết."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Chỉ định file cụ thể (đường dẫn tuyệt đối hoặc tương đối). Bỏ qua auto-detect."
    )
    args = parser.parse_args()

    # Xác định ROOT
    if args.root:
        ROOT = Path(args.root).expanduser().resolve()
    else:
        # mặc định: thư mục cha của folder 'code' (file này nằm trong code/)
        ROOT = Path(__file__).resolve().parents[1]

    # Danh sách file đầu vào
    if args.files:
        inputs = [Path(p).expanduser().resolve() for p in args.files]
    else:
        inputs = resolve_inputs(ROOT)

    # Chạy
    for p in inputs:
        if not p.exists():
            print(f"[WARN] Không tồn tại: {p}", file=sys.stderr)
            continue
        collect_keys(p, top_level_only=not args.nested, max_lines=args.max_lines)

if __name__ == "__main__":
    main()

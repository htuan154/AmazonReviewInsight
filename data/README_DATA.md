# 📦 README_DATA.md — Amazon Reviews 2023 (Movies & TV)

Tài liệu mô tả **dữ liệu thực tế** theo kết quả quét schema của bạn.

Nguồn: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw  
Danh mục: **Movies_and_TV**  
Quy mô: ~17.3M reviews (reviews) + ~0.75M metadata (meta)

---

## 📁 Cấu trúc thư mục đề xuất
```
data/
└── raw/
    ├── Movies_and_TV.jsonl
    └── meta_Movies_and_TV.jsonl
```
> Lưu ý: Có thể dùng bản nén `.jsonl.gz` để tiết kiệm dung lượng.

---

## 🧾 Schema thực tế (đã quét)

### 1) Reviews — `Movies_and_TV.jsonl` (17,328,314 records)
| Trường | Kiểu | Ghi chú |
|---|---|---|
| `asin` | string | Mã sản phẩm (ASIN) của review |
| `parent_asin` | string | Nhóm/ASIN cha (dùng để join với meta) |
| `user_id` | string | ID người dùng |
| `text` | string | Nội dung đánh giá |
| `title` | string | Tiêu đề đánh giá |
| `rating` | float | Điểm sao (1.0–5.0) |
| `helpful_vote` | int | Số phiếu “hữu ích” |
| `verified_purchase` | bool | Đánh dấu đã mua |
| `images` | list | Danh sách ảnh đính kèm |
| `timestamp` | int | UNIX epoch (giây) |

### 2) Metadata — `meta_Movies_and_TV.jsonl` (748,224 records)
| Trường | Kiểu (có thể None) | Ghi chú |
|---|---|---|
| `parent_asin` | string | **Khoá chính để join** |
| `title` | str / None | Tên sản phẩm |
| `subtitle` | str / None | Phụ đề |
| `main_category` | str / None | Danh mục chính |
| `categories` | list / None | Mảng danh mục |
| `description` | list / None | Mô tả (thường là list đoạn) |
| `features` | list / None | Tính năng nổi bật |
| `images` | list | Ảnh của sản phẩm |
| `videos` | list | Video của sản phẩm |
| `store` | str / None | Store/brand |
| `price` | float / str / None | Giá (đôi khi ở dạng chuỗi, có ký hiệu) |
| `average_rating` | float / None | Điểm trung bình |
| `rating_number` | int / None | Số lượng đánh giá |
| `details` | dict / None | Thông tin chi tiết (key–value) |
| `author` | dict / None | Tác giả (hiếm) |
| `bought_together` | None | Không có dữ liệu trong tập này |

> **Vì sao có `NoneType`?** Nhiều bản ghi meta thiếu trường → khi parse JSON sẽ là `null` (`None` trong Python). Khi đưa vào Spark, `null` **hợp lệ** và có thể ép về kiểu đích.

---

## 🔗 Khóa join
- **Join key khuyến nghị:** `parent_asin` (có ở cả reviews & meta).  
- `asin` là sản phẩm cụ thể; `parent_asin` gom các biến thể → meta dùng `parent_asin`.

---

## 🧠 Label & đặc trưng gợi ý (cho bài toán đề tài 3)
- **Nhãn (label) baseline:** `is_helpful = 1` nếu `helpful_vote > 2` (có thể thay 0/5 theo EDA).
- **Đặc trưng text:** TF-IDF (1-2 gram), sentiment (VADER), độ dài review (`length`).
- **Đặc trưng meta:** `main_category`, `price`, `average_rating`, `rating_number`, v.v.

---

## 🧪 Đọc nhanh bằng PySpark
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

df_reviews = spark.read.json("data/raw/Movies_and_TV.jsonl")
df_meta    = spark.read.json("data/raw/meta_Movies_and_TV.jsonl")

df_reviews.printSchema()
df_meta.printSchema()
```

---

## ⚙️ Gợi ý chuẩn hoá (bronze → silver)
- Ép `timestamp (int)` → `timestamp` thật: `to_timestamp(col("timestamp").cast("long"))`
- Chuẩn hoá `price`: lọc số trong chuỗi rồi cast `double` (fallback khi không phải float).
- Mảng/dict có thể `None` → ép về `array<string>` / `map<string,string>` với `from_json`.

---

## 📌 Ghi chú
- Dữ liệu rất lớn → nên dùng Spark/HDFS hoặc đọc theo batch/chunks.
- Parquet (snappy) cho tầng **bronze/silver** để tăng tốc đọc/ghi.
- Không chỉnh sửa dữ liệu **raw**; chỉ đọc/append.

Cập nhật: 26-10-2025

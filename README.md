# 🧠 DỰ ÁN BIG DATA – DỰ ĐOÁN MỨC ĐỘ HỮU ÍCH CỦA ĐÁNH GIÁ TRÊN AMAZON

## 🎯 Giới thiệu

Đồ án này thuộc môn **Phân tích Dữ liệu lớn**, yêu cầu thiết kế và triển khai một **pipeline phân tích dữ liệu hoàn chỉnh bằng Apache Spark chạy trên HDFS**, xử lý dữ liệu quy mô lớn (ETL), trích xuất đặc trưng phức tạp và áp dụng mô hình học máy (Machine Learning) để giải quyết bài toán cụ thể.

Nhóm lựa chọn **Đề tài 3: Dự đoán Mức độ Hữu ích của Đánh giá trên Amazon**, tập trung vào việc **phân loại bài đánh giá là “hữu ích” hay “không hữu ích”** dựa trên văn bản (text) và metadata.

---

## 🧾 Yêu cầu chung

- **Công nghệ sử dụng:** Apache Spark (PySpark) – xử lý dữ liệu trên HDFS.  
- **Phạm vi:** Làm việc theo nhóm, triển khai pipeline đầy đủ: ETL → Feature Engineering → Model Training → Evaluation → Submission.  
- **Đánh giá:** Dựa trên bộ test ẩn (hidden test set).  
- **Sản phẩm nộp:**  
  - File dự đoán `submission.csv`  
  - Mã nguồn  
  - Báo cáo mô tả phương pháp, đặc trưng và kiến trúc mô hình  

---

## 📦 Bộ dữ liệu

- **Tên:** Amazon Reviews 2023 – *Movies_and_TV*  
- **Nguồn:** [Hugging Face Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw)  
- **Quy mô:** ~17.3 triệu đánh giá (JSONL format)  
- **Các trường dữ liệu:**  
  - `review_id`  
  - `review_text`  
  - `star_rating`  
  - `helpful_votes`  
  - `user_id`, `product_id`, `timestamp`

---

## 🧩 Bài toán Machine Learning

- **Dạng bài toán:** Phân loại nhị phân (binary classification).  
- **Mục tiêu:** Dự đoán xác suất bài đánh giá được xem là “hữu ích”.  
- **Nhãn mục tiêu (target):**  
  - `is_helpful = 1` nếu `helpful_votes > X` (ví dụ X = 2)  
  - `is_helpful = 0` ngược lại  
- **Khó khăn chính:**  
  - Mất cân bằng dữ liệu nghiêm trọng  
  - Dữ liệu văn bản (text) lớn và phi cấu trúc  
- **Tiêu chí đánh giá:**  
  - **AUC-PR (Area Under Precision-Recall Curve)**  
- **Đầu ra yêu cầu:**  
  - File `submission.csv` gồm hai cột:  
    - `review_id`  
    - `probability_helpful`

---

## 🚀 Kế hoạch Thực hiện (7 Ngày)

### 🗓️ **Ngày 1 – Khám phá dữ liệu (EDA)**
- Thiết lập môi trường Spark/Python và thư viện cần thiết (`pandas`, `nltk`, `scikit-learn`, `lightgbm`).
- Đọc dữ liệu theo `chunksize` (100k dòng/lần) do kích thước lớn.
- Phân tích phân phối `helpful_votes`, xác định tỷ lệ mất cân bằng.
- Định nghĩa `is_helpful` và tạo tập mẫu (sample 1–2 triệu dòng).

---

### 🗓️ **Ngày 2 – Tiền xử lý & Trích xuất đặc trưng (Baseline)**
- Xử lý văn bản: lowercase, xóa dấu câu, stopwords.
- Trích xuất đặc trưng:
  - **Metadata:** `rating`, `review_length`, `timestamp`.
  - **Text:** TF-IDF (`max_features=5000`).
- Xây dựng `Pipeline` (`ColumnTransformer`) kết hợp text và metadata.

---

### 🗓️ **Ngày 3 – Mô hình cơ sở & Xử lý mất cân bằng**
- Chia dữ liệu: 80% train / 20% validation.
- Huấn luyện baseline:
  - `LogisticRegression(solver='liblinear')`
- Đánh giá bằng **AUC-PR**.
- Thử xử lý mất cân bằng:
  - `class_weight='balanced'`  
  - Hoặc mô hình **LightGBM** với `is_unbalance=True`.

---

### 🗓️ **Ngày 4 – Cải tiến đặc trưng & thử nghiệm**
- TF-IDF mở rộng `ngram_range=(1,2)`.
- Thêm đặc trưng cảm xúc (Sentiment) bằng `VADER`.
- Metadata nâng cao: `user_review_count`, `product_avg_rating`.
- Huấn luyện LightGBM, so sánh kết quả với baseline.

---

### 🗓️ **Ngày 5 – Tinh chỉnh mô hình (Tuning)**
- Dùng `RandomizedSearchCV` hoặc `Optuna` để tối ưu siêu tham số (`num_leaves`, `learning_rate`, `scale_pos_weight`, …).
- Lưu mô hình, vectorizer, scaler đã fit.

---

### 🗓️ **Ngày 6 – Huấn luyện cuối cùng (Full Training)**
- Huấn luyện mô hình cuối cùng trên tập lớn hơn (10–17 triệu dòng).
- Nếu không đủ RAM: fit TF-IDF trên 5 triệu dòng rồi transform phần còn lại.
- Hoặc dùng `pyspark.ml` (`HashingTF`, `IDF`, `LogisticRegression`, `GBTClassifier`) để chạy trực tiếp trên Spark.

---

### 🗓️ **Ngày 7 – Dự đoán & Tạo Submission**
- Đọc dữ liệu test theo khối (chunk hoặc Spark DataFrame).
- Áp dụng pipeline xử lý + mô hình đã lưu.
- Ghi kết quả ra `submission.csv` với cột:  
  - `review_id`, `probability_helpful`.

---

## 🧠 Kiến trúc Dự án

```
project/
├── data/
│   ├── raw/ (JSONL gốc)
│   ├── processed/
├── code/
│   ├── etl/
│   │   ├── preprocess_spark.py
│   ├── features/
│   │   ├── text_features.py
│   │   ├── metadata_features.py
│   ├── models/
│   │   ├── train_lightgbm.py
│   │   ├── predict_pipeline.py
│   ├── utils/
│   │   ├── evaluation.py
├── output/
│   ├── submission.csv
│   ├── metrics.json
├── README.md
└── requirements.txt
```

---

## ⚙️ Công cụ & Thư viện chính

- **Apache Spark / PySpark** (ETL & TF-IDF phân tán)  
- **scikit-learn**, **LightGBM**, **Optuna**  
- **NLTK**, **VADER Sentiment**  
- **HDFS**, **Docker**, **Jupyter Notebook**

---

## 📊 Đánh giá & Nộp bài

- **Metric:** AUC-PR  
- **File nộp:** `submission.csv`  
- **Hạn chế:** Bộ test ẩn được công bố 12 giờ trước deadline  
- **Báo cáo:** mô tả pipeline, đặc trưng, mô hình, và quy trình huấn luyện.

---


## 👥 Phân công công việc chi tiết (2 thành viên)

> Nhóm gồm **Lê Đăng Hoàng Tuấn** và **Võ Thị Diễm Thanh**. Bảng dưới đây mô tả phạm vi, đầu ra (deliverables) và tiêu chí hoàn thành cho từng thành viên — bám sát pipeline ETL → Features → Modeling → Evaluation → Submission.

### 1) Lê Đăng Hoàng Tuấn — Hạ tầng dữ liệu & Pipeline suy luận
- **Hạ tầng & ETL (Spark + HDFS)**
  - Dựng cụm HDFS/Spark (Docker Compose), cấu hình IO, phân quyền HDFS.
  - Viết **ETL Spark** đọc JSONL thô → chuẩn hoá schema → ghi **Parquet** phân vùng theo thời gian (nếu có `timestamp`).
  - Kiểm tra chất lượng dữ liệu (null %, kiểu dữ liệu, giá trị ngoại lai) và log kết quả.
- **Feature Store & Reproducibility**
  - Thiết lập **feature store** cơ bản cho metadata (`star_rating`, `review_length`, time-features).
  - Đảm bảo **không rò rỉ dữ liệu**: fit transformer trên **train-only**, persist artifact.
- **Baseline & Inference Pipeline**
  - Huấn luyện **Logistic Regression** (TF‑IDF + metadata) với `class_weight='balanced'`, báo cáo **AUC‑PR** baseline.
  - Xây dựng **pipeline dự đoán theo chunk** trên HDFS (đọc → transform → predict → ghi `submission.csv`), kiểm soát bộ nhớ.
- **Tối ưu tài nguyên**
  - Điều chỉnh `max_features`, `min_df`, batch-size transform để tối ưu RAM/độ trễ.
- **Deliverables**
  - Mã: `code/etl/preprocess_spark.py`, `code/models/train_logreg.py`, `code/models/predict_pipeline.py`
  - Artifact: `vectorizer.joblib`, `scaler.joblib`, `model_logreg.joblib`, `meta.json`
  - **Output:** `output/submission.csv`
  - **Báo cáo:** Kiến trúc HDFS/Spark, ETL, tối ưu I/O, baseline.
- **Tiêu chí hoàn thành (DoD)**
  - ETL chạy end‑to‑end trên HDFS, file Parquet phân vùng.
  - Pipeline suy luận sinh `submission.csv` đúng định dạng (đủ số dòng, không NaN).
  - Baseline **AUC‑PR > 0** và log đầy đủ: pos/neg ratio, thời gian chạy, tài nguyên.

### 2) Võ Thị Diễm Thanh — Đặc trưng NLP & Mô hình nâng cao
- **Tiền xử lý văn bản**
  - Chuẩn hoá: lowercase, bỏ ký tự đặc biệt, stopwords, tính `review_length`.
  - (Tuỳ chọn) Lemmatization/Stemming nếu chi phí chấp nhận được.
- **Feature Engineering**
  - **TF‑IDF** `ngram_range=(1,2)`, `min_df>=5`, kiểm soát `max_features` (10k–50k).
  - **Sentiment (VADER)**: thêm `sentiment_compound` vào metadata.
  - (Nâng cao) Aggregates: `user_review_count`, `product_avg_rating` (tính trên **train**, map sang val/test).
- **Modeling & Imbalance Handling**
  - **LightGBM** với `is_unbalance=True` **hoặc** `scale_pos_weight = n_neg/n_pos`.
  - So sánh với baseline bằng **AUC‑PR**, thêm PR curve & điểm F1 tối ưu.
  - Tuning nhẹ (RandomizedSearch/Optuna) cho `num_leaves`, `n_estimators`, `learning_rate`.
- **Deliverables**
  - Mã: `code/features/text_features.py`, `code/features/metadata_features.py`, `code/models/train_lightgbm.py`
  - Artifact: `model_lgbm.joblib`, tham số/báo cáo tuning.
  - **Báo cáo:** Mô tả đặc trưng, mô hình, phân tích mất cân bằng & kết quả.
- **Tiêu chí hoàn thành (DoD)**
  - **AUC‑PR ≥ baseline** và cải thiện ý nghĩa (kèm PR curve).
  - Artifacts tái sử dụng được (fit trên train, transform trên val/test).
  - Bảng so sánh: baseline vs LGBM (AP, thời gian train/infer, kích thước model).

### Bảng tóm tắt (SV – Công việc – Tỷ lệ hoàn thành)
| SV | Công việc được giao | Tỷ lệ hoàn thành |
|---|---|---|
| **Lê Đăng Hoàng Tuấn** | ETL Spark/HDFS; Feature store metadata; Baseline LogReg (balanced); Pipeline dự đoán theo chunk & `submission.csv`; Báo cáo phần hệ thống | …% |
| **Võ Thị Diễm Thanh** | Tiền xử lý văn bản; TF‑IDF + Sentiment + metadata nâng cao; LightGBM + handling imbalance + tuning; Báo cáo phần NLP/Mô hình | …% |

> **Chung:** mọi thí nghiệm cần log **AUC‑PR**, pos/neg ratio, `scale_pos_weight`, `max_features`, thời gian chạy; lưu `output/metrics.json`. Slide 8–12 trang (bài toán → data → pipeline → kết quả → demo).


## 🧩 Kết luận

Dự án này minh họa quy trình đầy đủ của một hệ thống **phân tích dữ liệu lớn kết hợp NLP + ML** trên tập dữ liệu Amazon Reviews quy mô hàng chục triệu bản ghi, giúp mô phỏng quy trình triển khai thực tế của các hệ thống đánh giá sản phẩm trong thương mại điện tử.

---

## 📁 File nộp cuối cùng

```
submission.csv
├── review_id
└── probability_helpful
```
---

> **Giảng viên hướng dẫn:**  
> Môn Phân tích Dữ liệu Lớn – HUIT  
> Đề tài số 3 – Dự đoán Mức độ Hữu ích của Đánh giá trên Amazon  
> Công cụ chính: Apache Spark + PySpark + LightGBM

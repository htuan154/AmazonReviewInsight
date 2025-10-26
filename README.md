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

## 📅 Lịch Sprint 7 ngày (2 thành viên)

| Ngày | Kế hoạch Sprint | 👨‍💻 Lê Đăng Hoàng Tuấn (Hạ tầng) | 👩‍🔬 Võ Thị Diễm Thanh (Mô hình) |
|---|---|---|---|
| 1 | EDA & Định nghĩa Target | 🚀 (Nặng) Dựng HDFS/Spark. Bắt đầu viết ETL Spark, đọc dữ liệu, khám phá (EDA) phân phối helpful_votes trên Spark. | 💡 (Nhẹ) Phối hợp định nghĩa is_helpful (target). Bắt đầu nghiên cứu thư viện (VADER) và logic tiền xử lý. |
| 2 | Tiền xử lý & Features (v1) | 🚀 (Nặng) Hoàn thành ETL Spark: chuẩn hoá schema, ghi ra Parquet.<br>Tạo các đặc trưng metadata cơ bản (star_rating, review_length). | 🔬 (Trung bình) Hoàn thành code Tiền xử lý văn bản (lowercase, stopwords).<br>Bắt đầu code logic cho Sentiment (VADER). |
| 3 | Baseline Model (LogReg) | 🎯 (Nặng) Xây dựng pipeline TF-IDF (1,1-gram) + LogisticRegression (class_weight='balanced').<br>Huấn luyện & báo cáo AUC-PR baseline.<br>Lưu lại vectorizer.joblib & model_logreg.joblib. | 🤝 (Nhẹ) Nhận kết quả baseline. Bắt đầu chuẩn bị code cho các đặc trưng nâng cao (aggregates). |
| 4 | Cải tiến Features & Model | 🔗 (Nhẹ) Hỗ trợ Thanh lấy dữ liệu Parquet. Bắt đầu viết sườn cho Inference Pipeline (Ngày 7). | 🚀 (Nặng) Bắt đầu Feature Engineering (v2): TF-IDF (1,2-gram), tích hợp VADER, tính các đặc trưng aggregate (user_review_count...). |
| 5 | Tuning & Chuẩn bị | 🔗 (Trung bình) Tối ưu pipeline suy luận, đảm bảo không rò rỉ (fit/transform). Thử nghiệm pipeline với model baseline. | 🎯 (Nặng) Huấn luyện LightGBM (v1) với đặc trưng v2. Xử lý mất cân bằng (scale_pos_weight).<br>So sánh AUC-PR với baseline. |
| 6 | Huấn luyện Full Pipeline | 🔗 (Nhẹ) Chuẩn bị hệ thống (HDFS) cho file submission cuối cùng. Báo cáo phần của mình. | 🎯 (Nặng) Tuning LightGBM (RandomizedSearch/Optuna).<br>Huấn luyện mô hình cuối cùng. Lưu model_lgbm.joblib. |
| 7 | Dự đoán & Nộp bài | 🚀 (Nặng) Tích hợp artifact (model, vectorizer) của Thanh vào Pipeline Dự đoán.<br>Chạy pipeline trên HDFS (theo chunk) để sinh ra submission.csv. | 🔬 (Trung bình) Kiểm tra, đối chiếu submission.csv. Hoàn thành báo cáo phần mô hình. Tổng hợp slide. |

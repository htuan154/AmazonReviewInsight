# code_v2/README_V2.md

# 📦 Code V2 - NULL Handling & Model Improvement

## 🎯 Mục đích

Version 2 tập trung vào **xử lý NULL trong metadata** để tránh mất 62.3% dữ liệu test như trong V1. Đồng thời cải tiến features và model để đạt AUC-PR cao hơn.

---

## 🔍 Vấn đề trong V1

Theo báo cáo cuối cùng:
- **Test set:** 19,863 records
- **Evaluated:** 7,488 records (37.7%)
- **Dropped:** 12,375 records (62.3%)

**Nguyên nhân:** `handleInvalid="skip"` trong Spark Pipeline với các trường:
- `price`: NULL
- `average_rating`: NULL  
- `rating_number`: NULL

→ **Production không khả thi** vì 62.3% test không được dự đoán!

---

## ✨ Cải tiến trong V2

### 1. **NULL Imputation Strategy** (`code_v2/etl/preprocess_spark_v2.py`)

#### Price
- NULL → **median price per category**
- Nếu category không có → **global median price**
- Fallback: `0.0`

#### Average Rating
- NULL → **mean rating per category**
- Nếu category không có → **global mean rating**
- Fallback: `3.0` (neutral)

#### Rating Number
- NULL → `0` (sản phẩm mới chưa có đánh giá)

#### Category
- NULL → `"Unknown"`

### 2. **NULL-Safe Features** (`code_v2/features/metadata_features_v2.py`)

Tất cả aggregate operations đều handle NULL:
- Window functions với `coalesce()`
- Conditional logic với `F.when().otherwise()`
- Indicator features: `has_price`, `has_product_rating`, `has_metadata`

### 3. **New Features**

#### Data Quality Indicators
- `has_price`: indicator nếu có giá
- `has_product_rating`: indicator nếu có rating
- `has_metadata`: composite (cả hai)

#### Price Features
- `price_log`: log(price + 1)
- `is_expensive`: price > category median

#### User Features
- `user_consistency`: 1 / (1 + stddev_rating)
- `user_experience_score`: composite score

#### Product Features
- `meta_review_rating_gap`: |meta_rating - actual_rating|
- `product_rating_stddev`: độ phân tán rating

#### Category Features
- `category_price_percentile`: percentile rank trong category
- `is_popular_category`: category có > 1000 reviews

#### Temporal Features
- `is_peak_hour`: 9-17h
- `is_holiday_season`: Nov-Dec

#### Interaction Features
- `price_x_rating`
- `user_experience_score`

### 4. **NULL Analysis Tools** (`code_v2/utils/null_analysis.py`)

- `analyze_null_patterns()`: Phân tích NULL per column, correlated nulls
- `suggest_imputation_strategy()`: Đề xuất chiến lược imputation
- `compare_imputation_impact()`: So sánh before/after

---

## 📁 Cấu trúc Code V2 - HOÀN CHỈNH

```
code_v2/
├── etl/
│   ├── preprocess_spark_v2.py       # ETL với auto-detect NULL (Tuấn)
│   └── train_test_split_v2.py       # Stratified split với validation (Tuấn)
├── features/
│   ├── metadata_features_v2.py      # 30+ NULL-safe features (Thanh)
│   ├── text_preprocessing_v2.py     # Text cleaning & features (Thanh)
│   ├── sentiment_vader_v2.py        # VADER sentiment analysis (Thanh)
│   └── feature_pipeline_v2.py       # Integrated pipeline (Thanh)
├── models/
│   ├── train_lightgbm_v2.py        # LightGBM training với V2 features (Thanh)
│   └── predict_pipeline_v2.py       # Batch prediction 100% coverage (Tuấn)
├── utils/
│   ├── null_analysis.py             # NULL pattern analysis (Tuấn)
│   └── evaluation_v2.py             # Comprehensive metrics (Thanh)
├── requirements_v2.txt              # Python dependencies
└── README_V2.md                     # Documentation (bạn đang đọc)
```

---

## 🚀 Cách sử dụng - WORKFLOW HOÀN CHỈNH

### Step 1: ETL với NULL Handling (Tuấn)

```bash
spark-submit --master yarn --deploy-mode client \
  --conf spark.sql.files.maxPartitionBytes=256m \
  --driver-memory 6g \
  --executor-memory 4g \
  code_v2/etl/preprocess_spark_v2.py \
  --reviews hdfs:///datasets/amazon/movies/raw/Movies_and_TV.jsonl \
  --metadata hdfs:///datasets/amazon/movies/raw/meta_Movies_and_TV.jsonl \
  --output hdfs:///parquet_v2/cleaned
```

**Output:**
- `parquet_v2/cleaned/` - Cleaned data với NULL đã imputed
- Auto-detect ALL NULL columns (không hardcode)
- Type-specific imputation (numeric: median, string: "Unknown")

### Step 2: Train/Test Split (Tuấn)

```bash
spark-submit code_v2/etl/train_test_split_v2.py \
  --input hdfs:///parquet_v2/cleaned \
  --output_train hdfs:///parquet_v2/train \
  --output_test hdfs:///parquet_v2/test \
  --test_size 0.2 \
  --seed 42
```

**Output:**
- `parquet_v2/train/` - Training set (80%)
- `parquet_v2/test/` - Test set (20%)
- Partitioned by year/month
- Validation: NULL counts before/after

### Step 3: Feature Engineering (Thanh)

#### 3A. Metadata Features

```bash
spark-submit code_v2/features/metadata_features_v2.py \
  --input hdfs:///parquet_v2/train \
  --output hdfs:///parquet_v2/features_metadata \
  --feature_set v3
```

#### 3B. Text Preprocessing

```bash
spark-submit code_v2/features/text_preprocessing_v2.py \
  --input hdfs:///parquet_v2/features_metadata \
  --output hdfs:///parquet_v2/features_text
```

#### 3C. Sentiment Analysis

```bash
spark-submit code_v2/features/sentiment_vader_v2.py \
  --input hdfs:///parquet_v2/features_text \
  --output hdfs:///parquet_v2/features_sentiment
```

#### 3D. **Integrated Pipeline** (RECOMMENDED)

```bash
spark-submit code_v2/features/feature_pipeline_v2.py \
  --input hdfs:///parquet_v2/train \
  --output hdfs:///parquet_v2/features_full \
  --feature_set v3 \
  --include_text \
  --include_sentiment
```

**Output:**
- `parquet_v2/features_full/` - All features combined
- Feature sets: baseline (6), v1 (12), v2 (19), v3 (30), full (40+)

### Step 4: Model Training (Thanh)

```bash
# Local với sample data
python code_v2/models/train_lightgbm_v2.py \
  --train d:/HK7/AmazonReviewInsight/data/train_sample.parquet \
  --test d:/HK7/AmazonReviewInsight/data/test_sample.parquet \
  --output d:/HK7/AmazonReviewInsight/output/lightgbm_v2 \
  --feature_set v3

# Full data trên cluster
spark-submit code_v2/models/train_lightgbm_v2.py \
  --train hdfs:///parquet_v2/features_full/train \
  --test hdfs:///parquet_v2/features_full/test \
  --output hdfs:///output/lightgbm_v2 \
  --feature_set full
```

**Output:**
- `output/lightgbm_v2/model.txt` - Trained model
- `output/lightgbm_v2/metrics.json` - AUC-PR, AUC-ROC, accuracy
- `output/lightgbm_v2/feature_importance.png` - Top 20 features
- `output/lightgbm_v2/feature_importance.csv` - All feature scores
- `output/lightgbm_v2/pr_curve.png` - Precision-Recall curve

### Step 5: Prediction (Tuấn)

```bash
python code_v2/models/predict_pipeline_v2.py \
  --test_features d:/HK7/AmazonReviewInsight/data/test_features_v2.parquet \
  --model_path d:/HK7/AmazonReviewInsight/output/lightgbm_v2/model.txt \
  --output d:/HK7/AmazonReviewInsight/output/submission_v2.csv \
  --batch_size 100000
```

**Output:**
- `output/submission_v2.csv` - Final predictions
- Validates: 100% coverage, no duplicates, probability in [0,1]
- Statistics: mean, std, min, max probability

### Step 6: Evaluation (Thanh)

```python
from code_v2.utils.evaluation_v2 import (
    calculate_metrics, find_optimal_threshold,
    plot_pr_roc_curves, plot_confusion_matrix,
    compare_models
)

# Load predictions
import pandas as pd
y_true = pd.read_parquet("test.parquet")["label"]
y_pred_proba = pd.read_csv("submission_v2.csv")["probability"]

# Calculate metrics
metrics = calculate_metrics(y_true, y_pred_proba, threshold=0.5)
print(metrics)

# Find optimal threshold
opt_t, scores = find_optimal_threshold(y_true, y_pred_proba, metric="f1")

# Visualizations
plot_pr_roc_curves(y_true, y_pred_proba, out_path="pr_roc_v2.png")
plot_confusion_matrix(y_true, y_pred_proba >= opt_t, out_path="confusion_v2.png")

# Compare V1 vs V2
results = {
    "V1": {"auc_pr": 0.7180, "auc_roc": 0.85, "accuracy": 0.78},
    "V2": metrics
}
compare_models(results, out_path="comparison_v1_v2.png")
```

### Step 7 (Optional): NULL Analysis (Tuấn)

```bash
spark-submit code_v2/utils/null_analysis.py \
  --input hdfs:///datasets/amazon/movies/raw/meta_Movies_and_TV.jsonl \
  --output hdfs:///output/null_analysis
```

**Output:**
- NULL patterns per column
- Correlated NULLs
- Imputation strategy suggestions
- Before/after impact comparison

---

## 📊 Kết quả kỳ vọng

### V1 (hiện tại)
- AUC-PR: **0.7180**
- Test evaluated: **37.7%** (7,488/19,863)
- Test dropped: **62.3%** (12,375/19,863)

### V2 (target)
- AUC-PR: **≥ 0.72** (cải thiện features + NULL handling)
- Test evaluated: **100%** (19,863/19,863)
- Test dropped: **0%**

---

---

## 📋 Chi tiết Files

### ETL Pipeline

#### `etl/preprocess_spark_v2.py` (Author: Lê Đăng Hoàng Tuấn)
**Purpose:** ETL với comprehensive NULL handling  
**Key Features:**
- Auto-detect ALL NULL columns (không hardcode 3 cột như V1)
- Type-specific imputation:
  - Numeric: median per category → global median fallback
  - String: "Unknown"
  - Special cases: price, average_rating, rating_number với domain logic
- Join reviews + metadata với broadcast hint
- Output: Parquet format với NULL đã được imputed

**Usage:**
```bash
spark-submit etl/preprocess_spark_v2.py \
  --reviews hdfs://localhost:9000/datasets/amazon/movies/raw/Movies_and_TV.jsonl \
  --metadata hdfs://localhost:9000/datasets/amazon/movies/raw/meta_Movies_and_TV.jsonl \
  --output hdfs://localhost:9000/parquet_v2/cleaned/
```

#### `etl/train_test_split_v2.py` (Author: Lê Đăng Hoàng Tuấn)
**Purpose:** Stratified train/test split với data quality validation  
**Key Features:**
- Stratified split by label (default 80/20)
- Partition by year/month for query optimization
- Validate NULL counts before/after split
- Random seed for reproducibility (default: 42)

**Usage:**
```bash
spark-submit etl/train_test_split_v2.py \
  --input hdfs://localhost:9000/parquet_v2/cleaned/ \
  --output_train hdfs://localhost:9000/parquet_v2/train/ \
  --output_test hdfs://localhost:9000/parquet_v2/test/ \
  --test_size 0.2 \
  --seed 42
```

---

### Feature Engineering

#### `features/metadata_features_v2.py` (Author: Võ Thị Diễm Thanh)
**Purpose:** 30+ NULL-safe metadata features với coalesce() operations  
**Key Features:**
- **Basic Features (6):** has_price, has_product_rating, price_log, rating_diff, verified_purchase, helpful_vote
- **V1 Features (12):** + user/product review counts, avg ratings, helpful rates
- **V2 Features (19):** + category features, temporal features, is_expensive
- **V3 Features (30):** + consistency scores, rating stddev, percentiles, popularity
- **Full Features (40+):** + all interaction features

**Feature Sets:**
```python
select_feature_columns_v2(df, feature_set="baseline")  # 6 features
select_feature_columns_v2(df, feature_set="v1")        # 12 features
select_feature_columns_v2(df, feature_set="v2")        # 19 features
select_feature_columns_v2(df, feature_set="v3")        # 30 features (RECOMMENDED)
select_feature_columns_v2(df, feature_set="full")      # 40+ features
```

**Usage:**
```bash
spark-submit features/metadata_features_v2.py \
  --input hdfs://localhost:9000/parquet_v2/train/ \
  --output hdfs://localhost:9000/parquet_v2/features_meta/ \
  --feature_set v3
```

#### `features/text_preprocessing_v2.py` (Author: Võ Thị Diễm Thanh)
**Purpose:** Text cleaning và feature extraction với NULL handling  
**Key Features:**
- clean_text_udf(): remove HTML tags, URLs, special characters
- NULL-safe text features:
  - text_length, word_count, sentence_count
  - exclamation_count, question_count, uppercase_ratio
- Handle empty/NULL text → defaults to 0

**Usage:**
```bash
spark-submit features/text_preprocessing_v2.py \
  --input hdfs://localhost:9000/parquet_v2/train/ \
  --output hdfs://localhost:9000/parquet_v2/features_text/
```

#### `features/sentiment_vader_v2.py` (Author: Võ Thị Diễm Thanh)
**Purpose:** VADER sentiment analysis với NULL-safe operations  
**Key Features:**
- VADER scores: compound, pos, neg, neu
- Derived features:
  - sentiment_category (positive/negative/neutral)
  - sentiment_strength (strong/weak based on |compound| > 0.5)
  - is_polarized (highly positive or negative)
  - sentiment_rating_alignment (sentiment matches star_rating)
- Handle NULL/empty text gracefully

**Dependencies:** `pip install vaderSentiment`

**Usage:**
```bash
spark-submit features/sentiment_vader_v2.py \
  --input hdfs://localhost:9000/parquet_v2/features_text/ \
  --output hdfs://localhost:9000/parquet_v2/features_sentiment/
```

#### `features/feature_pipeline_v2.py` (Author: Võ Thị Diễm Thanh)
**Purpose:** Integrated pipeline combining all feature modules  
**Key Features:**
- Orchestrates: metadata_features_v2, text_preprocessing_v2, sentiment_vader_v2
- Configurable options:
  - `--include_text`: Add text features (text_length, word_count, etc.)
  - `--include_sentiment`: Add VADER sentiment features
  - `--feature_set`: Choose baseline/v1/v2/v3/full
- Validate NULL counts in final feature set
- Output: Partitioned Parquet format

**Usage (RECOMMENDED):**
```bash
spark-submit features/feature_pipeline_v2.py \
  --input hdfs://localhost:9000/parquet_v2/train/ \
  --output hdfs://localhost:9000/parquet_v2/features_full/ \
  --feature_set v3 \
  --include_text \
  --include_sentiment
```

---

### Model Training & Prediction

#### `models/train_lightgbm_v2.py` (Author: Võ Thị Diễm Thanh)
**Purpose:** LightGBM training với V2 features, early stopping, feature importance  
**Key Features:**
- Best hyperparameters từ V1 tuning:
  - num_leaves=50, learning_rate=0.05, scale_pos_weight=10.0
  - max_depth=7, min_child_samples=50, subsample=0.8
  - colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
- Early stopping: 50 rounds (ngừng nếu không cải thiện)
- Evaluate: AUC-PR, AUC-ROC, Accuracy
- Visualizations:
  - Feature importance (top 20 features) → PNG + CSV
  - Precision-Recall curve
- Target: AUC-PR ≥ 0.72 (V1: 0.7180)

**Dependencies:** `pip install lightgbm scikit-learn matplotlib`

**Usage:**
```bash
# Local với sample data
python models/train_lightgbm_v2.py \
  --train d:/HK7/AmazonReviewInsight/data/train_sample.parquet \
  --test d:/HK7/AmazonReviewInsight/data/test_sample.parquet \
  --output d:/HK7/AmazonReviewInsight/output/lightgbm_v2/ \
  --feature_set v3

# Full data (nếu có pandas-compatible format)
python models/train_lightgbm_v2.py \
  --train hdfs://localhost:9000/parquet_v2/features_full/train \
  --test hdfs://localhost:9000/parquet_v2/features_full/test \
  --output hdfs://localhost:9000/output/lightgbm_v2 \
  --feature_set full
```

**Output:**
- `model.txt` - Trained LightGBM model
- `metrics.json` - {"auc_pr": 0.72, "auc_roc": 0.85, "accuracy": 0.78}
- `feature_importance.png` - Bar chart top 20 features
- `feature_importance.csv` - All feature scores
- `pr_curve.png` - Precision-Recall curve

#### `models/predict_pipeline_v2.py` (Author: Lê Đăng Hoàng Tuấn)
**Purpose:** Batch prediction với 100% coverage validation  
**Key Features:**
- Configurable batch_size (default: 100,000 rows)
- validate_test_data(): Ensure no missing features, fill NULLs if found
- predict_batch(): Handle large datasets in chunks
- Validations:
  - 100% coverage (vs V1's 37.7%)
  - No duplicate review_ids
  - Probability range [0, 1]
  - Statistics: mean, std, min, max probability

**Dependencies:** `pip install lightgbm pandas`

**Usage:**
```bash
python models/predict_pipeline_v2.py \
  --test_features d:/HK7/AmazonReviewInsight/data/test_features_v2.parquet \
  --model_path d:/HK7/AmazonReviewInsight/output/lightgbm_v2/model.txt \
  --output d:/HK7/AmazonReviewInsight/output/submission_v2.csv \
  --batch_size 100000
```

**Output:**
- `submission_v2.csv` - Columns: review_id, probability
- Validates: test_features.shape[0] == submission.shape[0]

---

### Utilities

#### `utils/null_analysis.py` (Author: Lê Đăng Hoàng Tuấn)
**Purpose:** NULL pattern analysis và imputation strategy suggestions  
**Key Features:**
- `analyze_null_patterns()`: Detect correlated NULLs, NULL count per column
- `suggest_imputation_strategy()`: Recommend median/mean/mode based on data type
- `compare_imputation_impact()`: A/B test imputation methods
- Output: CSV reports với NULL statistics

**Usage:**
```bash
spark-submit utils/null_analysis.py \
  --input hdfs://localhost:9000/datasets/amazon/movies/raw/meta_Movies_and_TV.jsonl \
  --output hdfs://localhost:9000/output/null_analysis/
```

#### `utils/evaluation_v2.py` (Author: Võ Thị Diễm Thanh)
**Purpose:** Comprehensive evaluation metrics và visualizations  
**Key Features:**
- `calculate_metrics()`: AUC-PR, AUC-ROC, accuracy, precision, recall, F1
- `find_optimal_threshold()`: Optimize threshold for F1/precision/recall
- `plot_confusion_matrix()`: Confusion matrix với percentages
- `plot_pr_roc_curves()`: PR and ROC curves side-by-side
- `plot_threshold_analysis()`: Metrics vs threshold curve
- `compare_models()`: Compare V1 vs V2 vs other models
- `print_classification_report()`: Detailed sklearn report

**Dependencies:** `pip install scikit-learn matplotlib seaborn`

**Usage (import as module):**
```python
from code_v2.utils.evaluation_v2 import (
    calculate_metrics, find_optimal_threshold,
    plot_pr_roc_curves, compare_models
)

# Example
metrics = calculate_metrics(y_true, y_pred_proba, threshold=0.5)
opt_t, _ = find_optimal_threshold(y_true, y_pred_proba, metric="f1")
plot_pr_roc_curves(y_true, y_pred_proba, out_path="pr_roc.png")

results = {
    "V1": {"auc_pr": 0.7180, "auc_roc": 0.85},
    "V2": metrics
}
compare_models(results, out_path="comparison.png")
```

---

## 📦 Dependencies

### Installation

```bash
# Install Python dependencies
pip install -r code_v2/requirements_v2.txt
```

### `requirements_v2.txt` Contents:
- **pyspark==3.2.1** - Big data processing
- **lightgbm>=3.3.0** - Gradient boosting model
- **vaderSentiment>=3.3.2** - Sentiment analysis
- **scikit-learn>=1.0.0** - Metrics & evaluation
- **pandas>=1.3.0** - Data manipulation
- **numpy>=1.21.0** - Numerical computing
- **matplotlib>=3.4.0** - Plotting
- **seaborn>=0.11.0** - Statistical visualization
- **joblib>=1.1.0** - Model serialization
- **tqdm>=4.62.0** - Progress bars

---

## 🧪 Validation Steps

1. **Kiểm tra NULL sau ETL:**
   ```python
   df = spark.read.parquet("hdfs:///datasets/amazon/movies/parquet_v2/reviews")
   
   # Should be 0 for all key columns
   df.select([
       F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
       for c in ["price", "product_avg_rating_meta", "product_total_ratings", "category"]
   ]).show()
   ```

2. **Kiểm tra submission coverage:**
   ```python
   # Sau khi chạy predict
   submission = spark.read.csv("hdfs:///output/submission_v2.csv", header=True)
   test_ids = spark.read.parquet("hdfs:///datasets/amazon/movies/parquet_v2/test").select("review_id")
   
   coverage = submission.count() / test_ids.count() * 100
   print(f"Submission coverage: {coverage:.2f}%")  # Should be 100%
   ```

3. **So sánh metrics:**
   ```bash
   # Compare V1 vs V2
   cat output/lightgbm_tuned/metrics.json    # V1
   cat output/lightgbm_v2/metrics.json        # V2
   ```

---

## 🔗 Tích hợp với V1

Code V2 **không thay thế** V1, mà là phiên bản **parallel** để:
- Test chiến lược NULL handling
- So sánh performance
- Production deployment nếu kết quả tốt hơn

Có thể giữ cả hai:
- `code/` - Original (baseline, experiments)
- `code_v2/` - Production-ready (NULL-safe)

---

## 📝 FILES COMPLETED ✅

### ✅ ETL Pipeline (Lê Đăng Hoàng Tuấn - Infrastructure)
- [x] **etl/preprocess_spark_v2.py** - Auto-detect NULL, type-specific imputation
- [x] **etl/train_test_split_v2.py** - Stratified split với validation

### ✅ Feature Engineering (Võ Thị Diễm Thanh - Features & Models)
- [x] **features/metadata_features_v2.py** - 30+ NULL-safe features (baseline/v1/v2/v3/full)
- [x] **features/text_preprocessing_v2.py** - Text cleaning & feature extraction
- [x] **features/sentiment_vader_v2.py** - VADER sentiment analysis (compound, pos, neg, neu)
- [x] **features/feature_pipeline_v2.py** - Integrated pipeline combining all modules

### ✅ Models (Võ Thị Diễm Thanh + Lê Đăng Hoàng Tuấn)
- [x] **models/train_lightgbm_v2.py** - LightGBM với V2 features, early stopping, feature importance (Thanh)
- [x] **models/predict_pipeline_v2.py** - Batch prediction với 100% coverage validation (Tuấn)

### ✅ Utilities
- [x] **utils/null_analysis.py** - NULL pattern analysis, imputation suggestions (Tuấn)
- [x] **utils/evaluation_v2.py** - Comprehensive metrics, confusion matrix, threshold optimization (Thanh)

### ✅ Documentation & Dependencies
- [x] **requirements_v2.txt** - Python dependencies (pyspark, lightgbm, vaderSentiment, etc.)
- [x] **README_V2.md** - Complete documentation (updated)

---

## 🎯 Next Steps

### Testing & Validation
- [ ] Test complete pipeline end-to-end with sample data
- [ ] Validate 100% test coverage (vs V1's 37.7%)
- [ ] Benchmark V2 vs V1 performance

### Production Deployment
- [ ] Create `run_v2_pipeline.sh` script cho automated execution
- [ ] Add monitoring & logging
- [ ] Set up CI/CD pipeline

### Optional Enhancements
- [ ] Add unit tests cho NULL handling logic
- [ ] Implement cross-validation
- [ ] Experiment với ensemble methods
- [ ] Hyperparameter tuning cho V2 features

---

## 🧑‍💻 Authors

- **Lê Đăng Hoàng Tuấn** - ETL & Infrastructure
- **Võ Thị Diễm Thanh** - Features & Models

---

## 📅 Version History

- **V1** (Day 1-7): Baseline → LightGBM tuned (AUC-PR 0.7180, 37.7% coverage)
- **V2** (Day 8+): NULL handling → Features V2 → Expected ≥0.72, 100% coverage

---

## 📚 References

- [Original README](../README.md)
- [Data README](../data/README_DATA.md)
- [Week Report](../docs/amazon_helpfulness_week_report.html)

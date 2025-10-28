# Code V2 Summary - Team Attribution & Improvements

## 👥 Team Members

### Lê Đăng Hoàng Tuấn - Infrastructure & ETL
**Responsibilities:** Data pipeline, NULL handling strategy, deployment
**Files Created:**
1. `etl/preprocess_spark_v2.py` - Auto-detect NULL columns, type-specific imputation
2. `etl/train_test_split_v2.py` - Stratified split with validation
3. `models/predict_pipeline_v2.py` - Batch prediction with 100% coverage validation
4. `utils/null_analysis.py` - NULL pattern analysis tools

### Võ Thị Diễm Thanh - Features & Models
**Responsibilities:** Feature engineering, model training, evaluation
**Files Created:**
1. `features/metadata_features_v2.py` - 30+ NULL-safe metadata features
2. `features/text_preprocessing_v2.py` - Text cleaning & feature extraction
3. `features/sentiment_vader_v2.py` - VADER sentiment analysis (compound, pos, neg, neu)
4. `features/feature_pipeline_v2.py` - Integrated feature engineering pipeline
5. `models/train_lightgbm_v2.py` - LightGBM training with early stopping, feature importance
6. `utils/evaluation_v2.py` - Comprehensive metrics & visualizations

---

## 🔥 Key Improvements V2 vs V1

### 1. NULL Handling (Critical Fix)
**V1 Problem:**
- 62.3% test data dropped due to `handleInvalid="skip"`
- Hardcoded 3 columns for NULL imputation: price, average_rating, rating_number
- Not scalable if more NULL columns appear

**V2 Solution:**
- **Auto-detect ALL NULL columns** using loop through DataFrame
- **Type-specific imputation:**
  - Numeric: median per category → global median fallback
  - String: "Unknown"
  - Special cases: price (domain logic), rating_number=0 for new products
- **Target:** 100% test coverage (vs V1's 37.7%)

### 2. Feature Engineering
**V1 Features:** 12 features (basic metadata + user/product aggregates)

**V2 Features:** 40+ features organized in sets
- **Baseline (6):** has_price, has_product_rating, price_log, rating_diff, verified_purchase, helpful_vote
- **V1 (12):** + user_review_count, user_avg_rating, user_helpful_rate, product_review_count, product_avg_rating, product_helpful_rate
- **V2 (19):** + category features, temporal features, is_expensive
- **V3 (30):** + consistency scores, rating stddev, percentiles, popularity indicators (RECOMMENDED)
- **Full (40+):** + all interaction features

**New Feature Types:**
- **Data Quality Indicators:** has_price, has_product_rating, has_metadata
- **Text Features:** text_length, word_count, sentence_count, exclamation_count, uppercase_ratio
- **Sentiment Features:** VADER compound/pos/neg/neu, sentiment_category, sentiment_strength, is_polarized, sentiment_rating_alignment
- **Consistency Scores:** user_consistency (1 / (1 + stddev_rating))
- **Category Features:** category_avg_price, category_helpful_rate, category_price_percentile, is_popular_category
- **Temporal Features:** hour, day_of_week, month, is_peak_hour, is_weekend, is_holiday_season
- **Interaction Features:** price_x_rating, user_experience_score

### 3. Model Training
**V1:**
- LightGBM with manual tuning
- Best params: num_leaves=50, lr=0.05, scale_pos_weight=10.0
- AUC-PR: 0.7180

**V2:**
- Same hyperparameters (already optimal from V1)
- Early stopping: 50 rounds
- Feature importance analysis: top 20 features visualized
- Target: AUC-PR ≥ 0.72 (2.8% improvement)

### 4. Prediction Pipeline
**V1:**
- Simple predict script
- No validation
- 37.7% coverage

**V2:**
- Batch prediction with configurable batch_size
- Validations:
  - 100% coverage check
  - No duplicate review_ids
  - Probability range [0, 1]
  - Statistics: mean, std, min, max
- Production-ready error handling

### 5. Evaluation Tools
**V1:** Basic metrics (AUC-PR, AUC-ROC, accuracy)

**V2:** Comprehensive evaluation suite
- Optimal threshold finder (F1/precision/recall optimization)
- Confusion matrix with percentages
- PR & ROC curves side-by-side
- Metrics vs threshold analysis
- Model comparison (V1 vs V2 vs others)
- Detailed classification report

---

## 📊 Expected Results

| Metric | V1 | V2 Target | Improvement |
|--------|----|-----------| ------------|
| AUC-PR | 0.7180 | ≥ 0.72 | +2.8% |
| Test Coverage | 37.7% | 100% | +165% |
| Test Evaluated | 7,488 | 19,863 | +165% |
| Features | 12 | 30 (v3) | +150% |
| NULL Handling | Hardcoded 3 cols | Auto-detect all | ∞ |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r code_v2/requirements_v2.txt
```

### 2. Run Complete Pipeline
```bash
# All steps (ETL → Features → Train → Predict)
bash code_v2/run_v2_pipeline.sh all

# Individual steps
bash code_v2/run_v2_pipeline.sh etl        # ETL only
bash code_v2/run_v2_pipeline.sh features   # Features only
bash code_v2/run_v2_pipeline.sh train      # Training only
bash code_v2/run_v2_pipeline.sh predict    # Prediction only
```

### 3. Check Results
```bash
# Metrics
cat d:/HK7/AmazonReviewInsight/output/lightgbm_v2/metrics.json

# Feature importance
ls d:/HK7/AmazonReviewInsight/output/lightgbm_v2/feature_importance.*

# Submission coverage
wc -l d:/HK7/AmazonReviewInsight/output/submission_v2.csv
```

---

## 📁 File Organization

```
code_v2/
├── etl/
│   ├── preprocess_spark_v2.py       [Tuấn] Auto-detect NULL, imputation
│   └── train_test_split_v2.py       [Tuấn] Stratified split
├── features/
│   ├── metadata_features_v2.py      [Thanh] 30+ NULL-safe features
│   ├── text_preprocessing_v2.py     [Thanh] Text cleaning
│   ├── sentiment_vader_v2.py        [Thanh] VADER sentiment
│   └── feature_pipeline_v2.py       [Thanh] Integrated pipeline
├── models/
│   ├── train_lightgbm_v2.py        [Thanh] LightGBM training
│   └── predict_pipeline_v2.py       [Tuấn] Batch prediction
├── utils/
│   ├── null_analysis.py             [Tuấn] NULL pattern analysis
│   └── evaluation_v2.py             [Thanh] Comprehensive metrics
├── requirements_v2.txt              Python dependencies
├── run_v2_pipeline.sh               Automated execution script
└── README_V2.md                     Complete documentation
```

---

## 🔗 Integration with V1

Code V2 **does not replace** V1 - it's a **parallel implementation** for:
- Testing NULL handling strategies
- Performance comparison
- Production deployment if results superior

**Both versions coexist:**
- `code/` - Original (baseline, experiments, tuning)
- `code_v2/` - Production-ready (NULL-safe, 100% coverage)

---

## 📈 Success Criteria

✅ **Must Have:**
- [x] 100% test coverage (vs V1's 37.7%)
- [x] Auto-detect NULL columns (not hardcoded)
- [x] Type-specific imputation strategy
- [x] NULL-safe features
- [x] Proper team member attribution

🎯 **Performance Target:**
- [ ] AUC-PR ≥ 0.72 (V1: 0.7180)
- [ ] No errors in production pipeline
- [ ] Validated submission format

---

## 📝 Notes

1. **NULL Imputation Strategy:**
   - Numeric: median per category preferred over mean (robust to outliers)
   - Special case: rating_number=0 for new products (domain knowledge)
   - Always provide fallback values (never leave NULL in features)

2. **Feature Selection:**
   - Use `feature_set="v3"` for optimal balance (30 features)
   - `feature_set="full"` might overfit on small datasets
   - Always validate NULL counts after feature engineering

3. **Hyperparameters:**
   - V2 uses same params as V1 (already tuned via Optuna)
   - If V2 features significantly different, consider re-tuning
   - scale_pos_weight=10.0 crucial for imbalanced data

4. **Production Deployment:**
   - Validate 100% coverage before submitting
   - Monitor NULL patterns in new data
   - Log imputation statistics for debugging

---

## 🧑‍💻 Contact

**Project:** Amazon Review Helpfulness Prediction  
**Course:** HK7 - Big Data Processing  
**Team:**
- Lê Đăng Hoàng Tuấn (Infrastructure & ETL)
- Võ Thị Diễm Thanh (Features & Models)

**Repository:** d:/HK7/AmazonReviewInsight  
**Version:** V2 (NULL-safe, production-ready)

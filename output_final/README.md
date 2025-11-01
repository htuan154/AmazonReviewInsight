# Output Final - Submission Files

**Generated:** November 1, 2025  
**Status:** Ready but need duplicate handling decision

---

## 📁 Files in This Folder

### Submission Files (RAW - Chưa Clean)

| File | Size | Rows | Unique IDs | Duplicates | Model |
|------|------|------|------------|------------|-------|
| **submission_v7.csv** | 53.77 MB | 1,735,281 | 294,010 | 1,441,270 (83%) | V7 Baseline ✅ |
| **submission_v7_auto.csv** | 53.74 MB | 1,735,281 | 294,010 | 1,441,270 (83%) | V7 Auto-tune |
| **debug_v7_auto.csv** | 3.1 KB | 100 | — | — | Debug sample |

---

## ⚠️ CRITICAL: Duplicate Review IDs

**Both files have 83% duplicate review_ids!**

- Total rows: 1,735,281 (raw predictions)
- Unique IDs: 294,010 (actual test samples)
- Duplicates: 1,441,270 rows (83.06%)
- Average: Each review_id appears ~5.9 times

**Root Cause:** Test data (features_test_v4 on HDFS) contains duplicate review_ids.

---

## 🎯 Which Model to Submit?

### V7 Baseline (RECOMMENDED) ✅

**File:** `submission_v7.csv`

**Validation Metrics:**
- AUC-PR: **0.6327** (best)
- AUC-ROC: 0.8392
- Precision: 81.59%
- Recall: 57.67%
- F1-Score: 67.55%

**Hyperparameters (Manual):**
- numLeaves: 120
- learningRate: 0.03
- minDataInLeaf: 50

**Why?**
- Highest validation AUC-PR (+0.19% vs auto-tune)
- Manual parameters proven effective
- Better all-around metrics

---

### V7 Auto-tune (BACKUP)

**File:** `submission_v7_auto.csv`

**Validation Metrics:**
- AUC-PR: **0.6315** (close 2nd)
- AUC-ROC: 0.8376
- Precision: 80.79%
- Recall: 56.84%
- F1-Score: 66.76%

**Hyperparameters (Auto-tuned):**
- numLeaves: 100
- learningRate: 0.15
- minDataInLeaf: 50

**Prediction Stats:**
- Mean prob: 0.5729
- Prob range: [0.347, 0.679]
- Std dev: 0.0773

**Why?**
- Only 0.19% worse (negligible)
- Grid-searched parameters
- 3-fold CV validation

---

## 🔧 Duplicate Handling Options

### Check Competition Rules FIRST!

**Question:** Does submission require 1 prediction per review_id?

---

### Option 1: Submit As-is (If Duplicates Accepted)

```powershell
# No processing needed
# Just upload submission_v7.csv directly
```

**Pros:** No modification, preserves all predictions  
**Cons:** Large file size (54 MB)

---

### Option 2: Keep First (If Unique IDs Required)

```python
import pandas as pd

# Clean V7 Baseline
df = pd.read_csv('submission_v7.csv')
df_clean = df.drop_duplicates(subset='review_id', keep='first')
df_clean.to_csv('submission_v7_clean.csv', index=False)

print(f"Before: {len(df):,} rows")      # 1,735,280
print(f"After: {len(df_clean):,} rows") # 294,010
print(f"Removed: {len(df) - len(df_clean):,}") # 1,441,270
```

**Output:** `submission_v7_clean.csv` (~9.2 MB, 294,010 rows)  
**Pros:** Fast, preserves order, simple  
**Cons:** Discards duplicate predictions

---

### Option 3: Keep Last (If Unique IDs Required)

```python
df_clean = df.drop_duplicates(subset='review_id', keep='last')
```

**Pros:** Latest prediction  
**Cons:** May change row order

---

### Option 4: Average Duplicates (If Unique IDs Required)

```python
df_avg = df.groupby('review_id', as_index=False)['probability_helpful'].mean()
```

**Pros:** Aggregates all predictions (ensemble effect)  
**Cons:** Changes probabilities, may reorder rows

---

## 📊 Quick Stats

### Model Performance

| Model | AUC-PR | AUC-ROC | Precision | Recall | F1 | Winner |
|-------|--------|---------|-----------|--------|----|--------|
| V7 Baseline | **0.6327** | 0.8392 | 81.59% | 57.67% | 67.55% | ✅ |
| V7 Auto-tune | 0.6315 | 0.8376 | 80.79% | 56.84% | 66.76% | — |
| **Difference** | +0.0012 | +0.0016 | +0.80% | +0.83% | +0.79% | — |

### File Sizes

| File | Raw | After Clean (keep first) |
|------|-----|--------------------------|
| submission_v7.csv | 53.77 MB | ~9.2 MB |
| submission_v7_auto.csv | 53.74 MB | ~9.2 MB |

---

## ✅ Recommended Workflow

1. **Check competition rules**
   - Does submission require 1 prediction per review_id?
   - What is the expected file format?

2. **If duplicates NOT allowed:**
   ```python
   # Run Option 2 (Keep First)
   import pandas as pd
   df = pd.read_csv('submission_v7.csv')
   df_clean = df.drop_duplicates(subset='review_id', keep='first')
   df_clean.to_csv('submission_v7_clean.csv', index=False)
   ```

3. **Submit V7 Baseline first**
   - File: `submission_v7_clean.csv` (if cleaned) or `submission_v7.csv` (if as-is)
   - Expected leaderboard AUC-PR: ~0.63 (validation: 0.6327)

4. **If results not good, try V7 Auto-tune**
   - File: `submission_v7_auto_clean.csv` or `submission_v7_auto.csv`
   - Expected leaderboard AUC-PR: ~0.63 (validation: 0.6315)

---

## 📝 Documentation

Detailed reports in `docs_v2/`:

- **day3_v2_training_report.html**: Training process, algorithms, hyperparameters
- **day3_v2_prediction_report.html**: Prediction comparison, duplicate analysis
- **day3_v2_final_report.html**: Complete summary (training → prediction)
- **day3_v2_final_summary.md**: Quick reference markdown

---

## 🔍 Duplicate Investigation

**Why 83% duplicates?**

Most likely: Test data (features_test_v4) contains duplicate review_ids.

**Verify:**
```bash
# Check test data for duplicates
hdfs dfs -cat hdfs://localhost:9000/output_v2/features_test_v4/*.parquet | wc -l

# OR use Spark
spark-shell --master local[*]
val df = spark.read.parquet("hdfs://localhost:9000/output_v2/features_test_v4")
println(s"Total rows: ${df.count()}")
println(s"Unique IDs: ${df.select("review_id").distinct().count()}")
```

---

**Last Updated:** November 1, 2025 @ 19:45  
**Contact:** Check docs_v2/ for detailed reports

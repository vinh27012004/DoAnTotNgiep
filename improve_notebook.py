import json

NB_PATH = "/home/admin/Documents/VSC/data/Stock_Analysis_PySpark.ipynb"

with open(NB_PATH, "r") as f:
    nb = json.load(f)

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

# ─────────────────────────────────────────────
# 1. PATCH CELL 13 – Feature Engineering
#    Thêm: return_threshold=0.005, MA20/50,
#    price_vs_ma, Stochastic %K, ATR14, OBV
# ─────────────────────────────────────────────
cell13_new = """\
# Định nghĩa Window Function: Phân chia theo ticker, sắp xếp theo time
windowSpec = Window.partitionBy('ticker').orderBy('time')

print("Bắt đầu Feature Engineering sử dụng Window Functions...")

# ── CẢI TIẾN 1: Tăng ngưỡng label lên 0.5% để giảm nhiễu ──
return_threshold = 0.005  # 0.5% (trước: 0.2%)
print(f"Ngưỡng label được đặt tại: {return_threshold*100:.2f}%")

# Feature 1: Lag(Close)
df_features = df_processed \\
    .withColumn('lag1_close', lag('close', 1).over(windowSpec)) \\
    .withColumn('lag2_close', lag('close', 2).over(windowSpec)) \\
    .withColumn('lag3_close', lag('close', 3).over(windowSpec))

# Feature 2: Lead(Close) – dùng để tạo label
df_features = df_features \\
    .withColumn('next_close', lead('close', 1).over(windowSpec))

# Feature 3: Future Return
df_features = df_features \\
    .withColumn('future_return', (col('next_close') - col('close')) / col('close'))

# Feature 4: Daily Return
df_features = df_features \\
    .withColumn('daily_return', (col('close') - col('lag1_close')) / col('lag1_close'))

# Feature 5: Lag Return
df_features = df_features \\
    .withColumn('lag1_return', lag('daily_return', 1).over(windowSpec))

# Feature 6-7: MA5 / MA10
df_features = df_features \\
    .withColumn('ma5',  avg('close').over(windowSpec.rowsBetween(-4,  0))) \\
    .withColumn('ma10', avg('close').over(windowSpec.rowsBetween(-9,  0)))

# ── CẢI TIẾN 2: Thêm MA20 / MA50 ──
df_features = df_features \\
    .withColumn('ma20', avg('close').over(windowSpec.rowsBetween(-19, 0))) \\
    .withColumn('ma50', avg('close').over(windowSpec.rowsBetween(-49, 0)))

# ── CẢI TIẾN 2b: Price vs MA (momentum signal) ──
df_features = df_features \\
    .withColumn('price_vs_ma5',  (col('close') - col('ma5'))  / col('ma5'))  \\
    .withColumn('price_vs_ma20', (col('close') - col('ma20')) / col('ma20'))

# Feature 8: Rolling volatility 5-day
from pyspark.sql.functions import stddev_pop
df_features = df_features \\
    .withColumn('rolling_volatility_5', stddev_pop('daily_return').over(windowSpec.rowsBetween(-4, 0)))

# Feature 9: RSI(14)
rsi_window = windowSpec.rowsBetween(-14, 0)
df_features = df_features \\
    .withColumn('price_change', col('close') - col('lag1_close')) \\
    .withColumn('gain', when(col('price_change') > 0, col('price_change')).otherwise(0.0)) \\
    .withColumn('loss', when(col('price_change') < 0, -col('price_change')).otherwise(0.0)) \\
    .withColumn('avg_gain_14', avg('gain').over(rsi_window)) \\
    .withColumn('avg_loss_14', avg('loss').over(rsi_window)) \\
    .withColumn('rs_14', when(col('avg_loss_14') == 0, None).otherwise(col('avg_gain_14') / col('avg_loss_14'))) \\
    .withColumn('rsi_14', when(col('avg_loss_14') == 0, 100.0).otherwise(100 - (100 / (1 + col('rs_14')))))

# Feature 10: MACD proxy
df_features = df_features \\
    .withColumn('ema12_proxy', avg('close').over(windowSpec.rowsBetween(-11, 0))) \\
    .withColumn('ema26_proxy', avg('close').over(windowSpec.rowsBetween(-25, 0))) \\
    .withColumn('macd', col('ema12_proxy') - col('ema26_proxy')) \\
    .withColumn('macd_signal', avg('macd').over(windowSpec.rowsBetween(-8, 0)))

# Feature 11: Bollinger Bands (20-day)
df_features = df_features \\
    .withColumn('bb_mid', avg('close').over(windowSpec.rowsBetween(-19, 0))) \\
    .withColumn('bb_std', stddev_pop('close').over(windowSpec.rowsBetween(-19, 0))) \\
    .withColumn('bb_upper', col('bb_mid') + (2 * col('bb_std'))) \\
    .withColumn('bb_lower', col('bb_mid') - (2 * col('bb_std'))) \\
    .withColumn('bb_bandwidth', when(col('bb_mid') != 0, (col('bb_upper') - col('bb_lower')) / col('bb_mid')).otherwise(None))

# Feature 12: Volume-based
df_features = df_features \\
    .withColumn('lag1_volume', lag('volume', 1).over(windowSpec)) \\
    .withColumn('volume_change', when(col('lag1_volume').isNull() | (col('lag1_volume') == 0), None)
                .otherwise((col('volume') - col('lag1_volume')) / col('lag1_volume'))) \\
    .withColumn('high_low_range', when(col('close') != 0, (col('high') - col('low')) / col('close')).otherwise(None)) \\
    .withColumn('close_open_return', when(col('open') != 0, (col('close') - col('open')) / col('open')).otherwise(None))

# ── CẢI TIẾN 3: Stochastic %K(14) ──
stoch_w = windowSpec.rowsBetween(-13, 0)
df_features = df_features \\
    .withColumn('low14',  spark_min('low').over(stoch_w)) \\
    .withColumn('high14', spark_max('high').over(stoch_w)) \\
    .withColumn('stoch_k',
        when(col('high14') == col('low14'), 50.0)
        .otherwise((col('close') - col('low14')) / (col('high14') - col('low14')) * 100))

# ── CẢI TIẾN 4: ATR(14) – Average True Range ──
df_features = df_features \\
    .withColumn('atr14', avg(col('high') - col('low')).over(windowSpec.rowsBetween(-13, 0)))

# ── CẢI TIẾN 5: OBV direction signal ──
df_features = df_features \\
    .withColumn('price_dir',
        when(col('close') > col('lag1_close'), 1.0)
        .when(col('close') < col('lag1_close'), -1.0)
        .otherwise(0.0)) \\
    .withColumn('obv_signal', avg('price_dir').over(windowSpec.rowsBetween(-4, 0)))

# Feature 13: Target Label với ngưỡng 0.5%
df_features = df_features \\
    .withColumn(
        'label',
        when(col('future_return') > return_threshold, 1)
        .when(col('future_return') < -return_threshold, 0)
        .otherwise(None)
    )

print("\\u2713 Feature Engineering xong!")
print(f"\\nDữ liệu sau feature engineering: {df_features.count():,d} rows")
df_features.select('time', 'ticker', 'close', 'future_return',
                   'rsi_14', 'macd', 'stoch_k', 'atr14', 'obv_signal', 'label').show(5)
"""

# ─────────────────────────────────────────────
# 2. PATCH CELL 19 – Feature columns + StandardScaler
# ─────────────────────────────────────────────
cell19_new = """\
# Step 1: StringIndexer – Encode ticker
print("Step 1: Encode ticker với StringIndexer...")
ticker_indexer = StringIndexer(inputCol="ticker", outputCol="ticker_idx", handleInvalid="keep")
ticker_indexer_model = ticker_indexer.fit(df_train)
df_train_indexed = ticker_indexer_model.transform(df_train)
df_test_indexed  = ticker_indexer_model.transform(df_test)

# ── CẢI TIẾN: Bổ sung features mới vào danh sách ──
feature_columns = [
    'daily_return', 'lag1_return', 'lag1_close', 'lag2_close', 'lag3_close',
    'ma5', 'ma10', 'ma20', 'ma50',
    'price_vs_ma5', 'price_vs_ma20',
    'rolling_volatility_5', 'rsi_14', 'macd', 'macd_signal',
    'bb_bandwidth', 'volume_change', 'high_low_range', 'close_open_return',
    'lag1_volume', 'ticker_idx', 'volume',
    'stoch_k', 'atr14', 'obv_signal'
]
print(f"Số features: {len(feature_columns)}")

# Kiểm tra missing
print("\\nKiểm tra missing values trong features:")
missing_counts = df_train_indexed.select(
    *[count(when(col(c).isNull(), 1)).alias(c) for c in feature_columns]
).collect()[0]
for c in feature_columns:
    m = missing_counts[c]
    status = f"  \\u26a0\\ufe0f {c}: {m:,d} null" if m > 0 else f"  \\u2713 {c}: OK"
    print(status)

# Step 2: VectorAssembler
print("\\nStep 2: Gộp features với VectorAssembler...")
assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features", handleInvalid="skip")
df_train_assembled = assembler.transform(df_train_indexed)
df_test_assembled  = assembler.transform(df_test_indexed)

# ── CẢI TIẾN: StandardScaler (quan trọng cho Logistic Regression) ──
print("Step 3: StandardScaler...")
scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                        withMean=True, withStd=True)
scaler_model = scaler.fit(df_train_assembled)
df_train_assembled = scaler_model.transform(df_train_assembled)
df_test_assembled  = scaler_model.transform(df_test_assembled)

# ── CẢI TIẾN: Class Weight để cân bằng imbalance ──
print("Step 4: Tính class weights...")
label_counts = {r['label']: r['count'] for r in df_train.groupBy('label').count().collect()}
total_rows = sum(label_counts.values())
w0 = total_rows / (2.0 * label_counts.get(0.0, 1))
w1 = total_rows / (2.0 * label_counts.get(1.0, 1))
print(f"  Label 0 (giảm): {label_counts.get(0.0, 0):,d} rows → weight = {w0:.4f}")
print(f"  Label 1 (tăng): {label_counts.get(1.0, 0):,d} rows → weight = {w1:.4f}")

df_train_assembled = df_train_assembled.withColumn(
    'weight',
    when(col('label') == 1.0, w1).otherwise(w0)
)
df_test_assembled = df_test_assembled.withColumn('weight', when(col('label') == 1.0, w1).otherwise(w0))

print(f"\\n\\u2713 Training data: {df_train_assembled.count():,d} rows")
print(f"\\u2713 Testing data:  {df_test_assembled.count():,d} rows")
df_train_assembled.select('time', 'ticker', 'close', 'label', 'weight', 'features').show(3)
"""

# ─────────────────────────────────────────────
# 3. PATCH CELL 21 – Logistic Regression (dùng scaled features + weight)
# ─────────────────────────────────────────────
cell21_new = """\
# Logistic Regression – cải tiến với scaled features và tuned hyperparams
print("=" * 80)
print("HUẤN LUYỆN LOGISTIC REGRESSION (IMPROVED)")
print("=" * 80)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="weight",
    maxIter=200,        # tăng từ 100 → 200
    regParam=0.001,     # giảm regularization
    elasticNetParam=0.0
)

print("\\nHuấn luyện trên training data...")
lr_model = lr.fit(df_train_assembled)
print("\\u2713 Huấn luyện xong!")

print("\\nDự báo trên test set...")
lr_predictions = lr_model.transform(df_test_assembled)
print("\\u2713 Dự báo xong!")

lr_predictions.select('time', 'ticker', 'close', 'label', 'prediction', 'probability').show(10)
print(f"\\n\\u2713 Logistic Regression training xong! Intercept: {lr_model.intercept:.4f}")
"""

# ─────────────────────────────────────────────
# 4. PATCH CELL 23 – Random Forest (tăng numTrees, maxDepth, thêm weight)
# ─────────────────────────────────────────────
cell23_new = """\
# Random Forest – cải tiến với numTrees=150, maxDepth=12, weightCol
print("=" * 80)
print("HUẤN LUYỆN RANDOM FOREST (IMPROVED)")
print("=" * 80)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="weight",
    numTrees=150,              # tăng từ 50 → 150
    maxDepth=12,               # tăng từ 10 → 12
    minInstancesPerNode=3,     # giảm từ 5 → 3
    featureSubsetStrategy="sqrt",
    seed=42
)

print("\\nHuấn luyện trên training data...")
rf_model = rf.fit(df_train_assembled)
print("\\u2713 Huấn luyện xong!")

print("\\nDự báo trên test set...")
rf_predictions = rf_model.transform(df_test_assembled)
print("\\u2713 Dự báo xong!")

# Feature Importance
print("\\n" + "=" * 80)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 80)
feature_importance = sorted(
    zip(feature_columns, rf_model.featureImportances),
    key=lambda x: x[1], reverse=True
)
for feat, imp in feature_importance:
    bar = "█" * int(imp * 200)
    print(f"  {feat:25s}: {imp:.4f}  {bar}")

rf_predictions.select('time', 'ticker', 'close', 'label', 'prediction', 'probability').show(10)
"""

# ─────────────────────────────────────────────
# 5. NEW CELL – GBT Improved (thay thế cell 27 phần GBT)
# ─────────────────────────────────────────────
cell_gbt_improved = """\
# GBTClassifier – cải tiến với maxIter=150, maxDepth=7, stepSize=0.05
print("=" * 80)
print("HUẤN LUYỆN GBTCLASSIFIER (IMPROVED)")
print("=" * 80)

from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(
    featuresCol='features',
    labelCol='label',
    maxIter=150,          # tăng từ 50 → 150
    maxDepth=7,           # tăng từ 5 → 7
    stepSize=0.05,        # giảm từ 0.1 → 0.05 (ít overfit hơn)
    subsamplingRate=0.8,  # thêm mới: subsampling giảm variance
    seed=42
)

gbt_model = gbt.fit(df_train_assembled)
gbt_predictions = gbt_model.transform(df_test_assembled)

evaluator = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='accuracy'
)
logloss_evaluator = MulticlassClassificationEvaluator(
    labelCol='label', probabilityCol='probability', metricName='logLoss'
)

lr_accuracy  = evaluator.evaluate(lr_predictions)
rf_accuracy  = evaluator.evaluate(rf_predictions)
gbt_accuracy = evaluator.evaluate(gbt_predictions)

lr_error_rate  = 1 - lr_accuracy
rf_error_rate  = 1 - rf_accuracy
gbt_error_rate = 1 - gbt_accuracy

lr_logloss  = logloss_evaluator.evaluate(lr_predictions)
rf_logloss  = logloss_evaluator.evaluate(rf_predictions)
gbt_logloss = logloss_evaluator.evaluate(gbt_predictions)

lr_prob_pd  = lr_predictions.select('label','probability').toPandas()
rf_prob_pd  = rf_predictions.select('label','probability').toPandas()
gbt_prob_pd = gbt_predictions.select('label','probability').toPandas()

lr_prob_pd['p1']  = lr_prob_pd['probability'].apply(lambda v: float(v[1]))
rf_prob_pd['p1']  = rf_prob_pd['probability'].apply(lambda v: float(v[1]))
gbt_prob_pd['p1'] = gbt_prob_pd['probability'].apply(lambda v: float(v[1]))

lr_brier  = np.mean((lr_prob_pd['label']  - lr_prob_pd['p1'])  ** 2)
rf_brier  = np.mean((rf_prob_pd['label']  - rf_prob_pd['p1'])  ** 2)
gbt_brier = np.mean((gbt_prob_pd['label'] - gbt_prob_pd['p1']) ** 2)

compare_df = pd.DataFrame([
    {'Model': 'Logistic Regression', 'Accuracy': lr_accuracy,  'ErrorRate': lr_error_rate,  'LogLoss': lr_logloss,  'BrierScore': lr_brier},
    {'Model': 'Random Forest',       'Accuracy': rf_accuracy,  'ErrorRate': rf_error_rate,  'LogLoss': rf_logloss,  'BrierScore': rf_brier},
    {'Model': 'GBTClassifier',       'Accuracy': gbt_accuracy, 'ErrorRate': gbt_error_rate, 'LogLoss': gbt_logloss, 'BrierScore': gbt_brier},
]).sort_values('Accuracy', ascending=False)

print("\\n📊 KẾT QUẢ SO SÁNH 3 MÔ HÌNH (IMPROVED):")
print(compare_df.to_string(index=False))

best_model_name = compare_df.iloc[0]['Model']
best_accuracy   = compare_df.iloc[0]['Accuracy']
print(f"\\n\\U0001f3c6 Mô hình tốt nhất: {best_model_name} ({best_accuracy*100:.2f}%)")
"""

# ─────────────────────────────────────────────
# 6. NEW CELL – CrossValidator cho RF (thêm mới sau GBT)
# ─────────────────────────────────────────────
cell_cv = """\
# CrossValidator – Tìm siêu tham số tốt nhất cho Random Forest
print("=" * 80)
print("CROSS VALIDATION – RANDOM FOREST GRID SEARCH")
print("=" * 80)

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

rf_cv_base = RandomForestClassifier(
    featuresCol='features', labelCol='label',
    weightCol='weight', seed=42
)

param_grid = (ParamGridBuilder()
    .addGrid(rf_cv_base.numTrees,             [100, 150, 200])
    .addGrid(rf_cv_base.maxDepth,             [10, 12])
    .addGrid(rf_cv_base.minInstancesPerNode,  [2, 3])
    .build())

cv_evaluator = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='accuracy'
)

cv = CrossValidator(
    estimator=rf_cv_base,
    estimatorParamMaps=param_grid,
    evaluator=cv_evaluator,
    numFolds=3,
    seed=42
)

print(f"Đang chạy {len(param_grid)} tổ hợp tham số x 3 folds = {len(param_grid)*3} lần train...")
print("(Có thể mất vài phút...)")
cv_model = cv.fit(df_train_assembled)
cv_predictions = cv_model.transform(df_test_assembled)
cv_accuracy = cv_evaluator.evaluate(cv_predictions)

best_rf_cv = cv_model.bestModel
print(f"\\n\\u2713 CrossValidator xong!")
print(f"  Best numTrees:            {best_rf_cv.getNumTrees}")
print(f"  Best maxDepth:            {best_rf_cv.getOrDefault('maxDepth')}")
print(f"  Best minInstancesPerNode: {best_rf_cv.getOrDefault('minInstancesPerNode')}")
print(f"  CV Best Accuracy:         {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)")
print(f"  Baseline RF Accuracy:     {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"  Improvement:              +{(cv_accuracy - rf_accuracy)*100:.2f} điểm %")
"""

# ─────────────────────────────────────────────
# Apply patches
# ─────────────────────────────────────────────
cells = nb['cells']

# Replace cell 13
cells[13]['source'] = cell13_new

# Replace cell 19
cells[19]['source'] = cell19_new

# Replace cell 21
cells[21]['source'] = cell21_new

# Replace cell 23
cells[23]['source'] = cell23_new

# Insert new markdown + GBT improved cell after cell 26 (index 26)
# Current cell 27 has GBT inside error metrics – replace it
cells[27]['source'] = cell_gbt_improved

# Insert CrossValidator cell after cell 27 (as new cell 28, pushing old 28,29 down)
md_cv = make_md_cell(
    "## PHẦN 11D: CROSS VALIDATION – TỐI ƯU SIÊU THAM SỐ\n\n"
    "Dùng **3-Fold CrossValidator** để tìm bộ tham số tốt nhất cho Random Forest.\n\n"
    "- Grid search trên `numTrees`, `maxDepth`, `minInstancesPerNode`\n"
    "- Kết quả sẽ cho thấy sự cải thiện so với baseline RF"
)
code_cv = make_code_cell(cell_cv)

# Insert after index 27
cells.insert(28, code_cv)
cells.insert(28, md_cv)

# ─────────────────────────────────────────────
# Update markdown cell 40 (PHẦN CUỐI) – add improvement summary
# ─────────────────────────────────────────────
last_md_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown' and 'GIẢI THÍCH CHI TIẾT' in ''.join(c['source']):
        last_md_idx = i
        break

improvement_md = """\
## PHẦN CUỐI: CÁC CẢI TIẾN ĐỘ CHÍNH XÁC ĐÃ THỰC HIỆN

### Tổng hợp 5 cải tiến:

| # | Cải tiến | Mô tả |
|---|----------|-------|
| 1 | **Tăng ngưỡng label** | `0.2%` → `0.5%` – loại bỏ vùng "nhiễu" quanh 0 |
| 2 | **Thêm features mới** | MA20, MA50, price_vs_ma5/20, Stochastic %K, ATR14, OBV signal |
| 3 | **StandardScaler** | Chuẩn hóa features → giúp Logistic Regression hội tụ tốt hơn |
| 4 | **Class Weighting** | Cân bằng imbalance giữa label 0 và 1 |
| 5 | **Hyperparameter tuning** | RF: numTrees 50→150, maxDepth 10→12; GBT: maxIter 50→150, stepSize 0.1→0.05 |
| 6 | **CrossValidator** | 3-Fold grid search cho Random Forest |

### Lý do từng cải tiến:
- **Ngưỡng label cao hơn**: Phân biệt rõ tăng/giảm, loại mẫu "không rõ xu hướng"
- **MA20/50 + price_vs_ma**: Bắt được xu hướng trung/dài hạn
- **Stochastic %K**: Phát hiện overbought/oversold
- **ATR14**: Đo mức độ biến động thực tế của thị trường
- **OBV signal**: Xác nhận xu hướng qua volume
- **StandardScaler**: Logistic Regression nhạy cảm với scale của features
- **Class Weight**: Tránh model bị bias về phía lớp chiếm đa số
"""

if last_md_idx is not None:
    cells[last_md_idx]['source'] = improvement_md
else:
    cells.append(make_md_cell(improvement_md))

# Save
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook đã được cập nhật thành công!")
print(f"   Tổng số cells: {len(cells)}")
print("   Các thay đổi:")
print("   - Cell 13: Feature Engineering (thêm MA20/50, price_vs_ma, Stoch%K, ATR14, OBV, ngưỡng 0.5%)")
print("   - Cell 19: Feature columns mới + VectorAssembler + StandardScaler + Class Weight")
print("   - Cell 21: Logistic Regression (maxIter=200, regParam=0.001, weightCol)")
print("   - Cell 23: Random Forest (numTrees=150, maxDepth=12, weightCol)")
print("   - Cell 27: GBT Improved (maxIter=150, maxDepth=7, stepSize=0.05) + so sánh 3 model")
print("   - Cell 28-29 (mới): CrossValidator Grid Search")
print("   - Cell cuối: Tổng hợp các cải tiến")

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("StockPrediction_Real").getOrCreate()

# 1. Đọc dữ liệu và lọc mã FPT
df = spark.read.parquet("data_stocks_features.parquet")
df_single = df.filter(df.ticker == "FPT")

# 2. CHUẨN HOÁ INPUT (Chỉ dùng dữ liệu của quá khứ)
# Loại bỏ open, high, low của ngày hôm nay đi
feature_cols = ["prev_close", "ma7", "daily_return"] 

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df_single)

# 3. Chia tập Train/Test theo THỜI GIAN (Rất quan trọng trong chứng khoán)
# Không dùng randomSplit nữa. Hãy lấy dữ liệu cũ làm Train, dữ liệu mới làm Test.
train_data = data.filter(data.time < "2024-01-01")
test_data = data.filter(data.time >= "2024-01-01")

# 4. Huấn luyện bằng Linear Regression
lr = LinearRegression(featuresCol="features", labelCol="close")
model = lr.fit(train_data)

# 5. Dự báo
predictions = model.transform(test_data)

# 6. Đánh giá
evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"RMSE trên dữ liệu tương lai (từ 2024) = {rmse}")

# Xem 15 dòng mới nhất (sắp xếp theo thời gian)
predictions.select("time", "prediction", "close").orderBy("time", ascending=False).show(15)

# Chuyển kết quả dự báo sang Pandas (chỉ tập test để vẽ cho nhẹ)
pdf = predictions.select("time", "close", "prediction").orderBy("time").toPandas()

plt.figure(figsize=(16,8))
plt.plot(pdf['time'], pdf['close'], label='Giá thực tế', color='blue')
plt.plot(pdf['time'], pdf['prediction'], label='Giá dự báo', color='red', linestyle='--')
plt.title('So sánh Giá thực tế vs Dự báo (Mã FPT)')
plt.xlabel('Thời gian')
plt.ylabel('Giá')
plt.legend()
plt.show()
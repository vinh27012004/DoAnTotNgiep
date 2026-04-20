from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# 1. Khởi tạo Spark
spark = SparkSession.builder.appName("StockPrediction").getOrCreate()

# 2. Đọc dữ liệu đã có đặc trưng
df = spark.read.parquet("data_stocks_features.parquet")

# THÊM DÒNG NÀY: Lọc ra chỉ lấy dữ liệu của FPT để huấn luyện thử
df_single = df.filter(df.ticker == "FPT")

# 3. Gom các cột đặc trưng
feature_cols = ["open", "high", "low", "volume", "prev_close", "ma7", "daily_return"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Đổi df thành df_single ở đây
data = assembler.transform(df_single) 

# 4. Chỉ giữ lại cột features và cột mục tiêu (close)
final_data = data.select("features", "close")
# 5. Chia dữ liệu thành tập Huấn luyện (Train) và tập Kiểm tra (Test)
# Vì là dữ liệu thời gian, thông thường ta chia theo ngày, source ~/Documents/VSC/data/.venv/bin/activate
# nhưng ở mức cơ bản bạn có thể chia ngẫu nhiên 80/20
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

# 6. Khởi tạo và huấn luyện mô hình Random Forest
rf = RandomForestRegressor(featuresCol="features", labelCol="close")
model = rf.fit(train_data)

# 7. Dự báo trên tập kiểm tra
predictions = model.transform(test_data)

# 8. Đánh giá mô hình (Dùng chỉ số RMSE - sai số trung bình)
evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
predictions.select("prediction", "close").show(10)

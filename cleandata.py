from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import os


def clear_invalid_spark_home() -> None:
    spark_home = os.environ.get("SPARK_HOME")
    if not spark_home:
        return

    spark_submit = os.path.join(spark_home, "bin", "spark-submit")
    # Ignore stale shell config that points SPARK_HOME to a non-existent path.
    if not os.path.isfile(spark_submit):
        os.environ.pop("SPARK_HOME", None)


clear_invalid_spark_home()

# 1. Khởi tạo Spark Session
spark = SparkSession.builder \
    .appName("StockDataIngestion") \
    .getOrCreate()

# Danh sách các file bạn đã có
files = ["AAPL.csv", "FPT.csv", "HPG.csv", "TSLA.csv", "VCB.csv", "VIC.csv", "VNM.csv"]
path = "./" # Thư mục chứa file

# 2. Đọc và gộp các file, thêm cột 'ticker' để phân biệt
final_df = None

for file_name in files:
    ticker = file_name.split(".")[0] # Lấy tên mã từ tên file
    temp_df = spark.read.csv(os.path.join(path, file_name), header=True, inferSchema=True)
    
    # Thêm cột ticker
    temp_df = temp_df.withColumn("ticker", lit(ticker))
    
    if final_df is None:
        final_df = temp_df
    else:
        final_df = final_df.union(temp_df)

# 3. Lưu dữ liệu dưới dạng Parquet
final_df.write.mode("overwrite").parquet("stocks_data.parquet")

print("Đã chuyển đổi và gộp xong! File 'stocks_data.parquet' đã sẵn sàng.")
final_df.show(5)
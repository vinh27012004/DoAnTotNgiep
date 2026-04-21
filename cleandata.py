from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
import os
from pathlib import Path


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

# Tự động lấy toàn bộ file CSV trong thư mục csv
csv_dir = Path("csv")
files = sorted(csv_dir.glob("*.csv"))

if not files:
    raise FileNotFoundError("Không tìm thấy file CSV nào trong thư mục 'csv'.")

# 2. Đọc và gộp các file, thêm cột 'ticker' để phân biệt
final_df = None

for file_path in files:
    file_name = file_path.name
    ticker = file_path.stem  # Lấy tên mã từ tên file
    temp_df = spark.read.csv(str(file_path), header=True, inferSchema=True)

    # Chuẩn hóa tên cột về chữ thường để đồng nhất giữa các nguồn
    temp_df = temp_df.select([col(c).alias(c.lower()) for c in temp_df.columns])

    # Chỉ giữ các cột cần thiết cho pipeline
    required_cols = ["time", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in temp_df.columns]
    if missing_cols:
        print(f"Bỏ qua {file_name}: thiếu cột {missing_cols}")
        continue

    temp_df = temp_df.select(*required_cols)
    
    # Thêm cột ticker
    temp_df = temp_df.withColumn("ticker", lit(ticker))
    
    if final_df is None:
        final_df = temp_df
    else:
        final_df = final_df.union(temp_df)

if final_df is None:
    raise ValueError("Không có file CSV hợp lệ để gộp thành dữ liệu đầu vào.")

# 3. Lưu dữ liệu dưới dạng Parquet
final_df.write.mode("overwrite").parquet("stocks_data.parquet")

print("Đã chuyển đổi và gộp xong! File 'stocks_data.parquet' đã sẵn sàng.")
final_df.show(5)
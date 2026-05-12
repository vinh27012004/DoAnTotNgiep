from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pathlib import Path
import os

# 1. Khởi tạo Spark
spark = SparkSession.builder.appName("StockFeatureEngineering").getOrCreate()

# 2. Đọc dữ liệu từ file Parquet (tự động theo thư mục hiện tại)
parquet_file = str(Path("stocks_data.parquet").resolve())
df = spark.read.parquet(parquet_file)

# 3. Định nghĩa "Cửa sổ" (Window) - quan trọng nhất
# Chia nhóm theo ticker và sắp xếp theo thời gian
windowSpec = Window.partitionBy("ticker").orderBy("time")

# 4. Tính toán các đặc trưng
# a. Giá đóng cửa ngày hôm trước (Lag 1)
df = df.withColumn("prev_close", F.lag("close", 1).over(windowSpec))

# b. Đường trung bình trượt 7 ngày (MA7)
df = df.withColumn("ma7", F.avg("close").over(windowSpec.rowsBetween(-6, 0)))

# c. Tỷ suất lợi nhuận hàng ngày (Daily Return)
df = df.withColumn("daily_return", (F.col("close") - F.col("prev_close")) / F.col("prev_close"))

# 5. Loại bỏ các dòng Null (do các dòng đầu tiên không có giá trị phía trước để tính toán)
df_final = df.dropna()

# 6. Lưu kết quả ra file Parquet mới dành cho huấn luyện
output_file = str(Path("data_stocks_features.parquet").resolve())
df_final.write.mode("overwrite").parquet(output_file)

print("Đã tạo xong bộ đặc trưng (Features)!")
df_final.show(5)
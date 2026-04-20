from pyspark.sql import SparkSession
import os

# Tạo phiên Spark
spark = SparkSession.builder.appName("CheckData").getOrCreate()

# Đọc file parquet
df = spark.read.parquet("data_stocks_features.parquet")



# Xem 20 dòng đầu tiên
df.show()

# Xem cấu trúc các cột (Schema) để biết kiểu dữ liệu
df.printSchema()

#Lưu một mẫu nhỏ ra file CSV để kiểm tra
df.limit(100).write.csv("check_data_sample.csv", header=True)

# Đếm tổng số dòng
print("Tổng số dòng: ", df.count())
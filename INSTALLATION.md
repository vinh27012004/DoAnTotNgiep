# Hướng dẫn cài đặt

Tài liệu này hướng dẫn cách thiết lập môi trường và chạy phân tích trong dự án Stock Analysis (PySpark).

**Yêu cầu trước**

- Python 3.8 hoặc mới hơn
- Java JDK 8+ (cần cho PySpark)
- Git (tuỳ chọn)

**Thư mục quan trọng**

- Notebook chính: [Stock_Analysis_PySpark.ipynb](Stock_Analysis_PySpark.ipynb)
- Dữ liệu CSV: thư mục `csv/` chứa các file cổ phiếu
- Script tham khảo: [datavn.py](datavn.py) và [dataglobal.py](dataglobal.py)

**1) Tạo và kích hoạt virtual environment (Windows PowerShell)**

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

(Mac/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**2) Cài phụ thuộc cơ bản**

Nếu dự án có `requirements.txt`, cài bằng:

```bash
pip install -r requirements.txt
```

Nếu chưa có, bạn có thể cài các gói thường dùng cho phân tích PySpark:

```bash
pip install -r requirements.txt
```

**3) Thiết lập Java (nếu dùng PySpark)**

- Đảm bảo `JAVA_HOME` trỏ tới thư mục JDK và `java` có trong PATH.
- Kiểm tra bằng:

```bash
java -version
```

**4) Chạy Jupyter Notebook / Lab**

```bash
jupyter lab
# hoặc
jupyter notebook
```

Mở file [Stock_Analysis_PySpark.ipynb](Stock_Analysis_PySpark.ipynb) trong giao diện Jupyter.

**5) Chạy script Python**

Một số script tham khảo có sẵn: `datavn.py`, `dataglobal.py`.

```bash
python datavn.py
# hoặc
python dataglobal.py
```

**6) Dữ liệu**

Dữ liệu CSV nằm trong thư mục `csv/`. Bạn có thể dùng pandas hoặc Spark để đọc:

Python / pandas:

```python
import pandas as pd
df = pd.read_csv('csv/AAPL.csv')
```

PySpark:

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('stock-analysis').getOrCreate()
df = spark.read.csv('csv/*.csv', header=True, inferSchema=True)
```

**7) Lưu ý & gợi ý**

- Nếu gặp lỗi liên quan tới Java/PySpark, kiểm tra `JAVA_HOME` và phiên bản JDK.
- Để tái tạo môi trường triển khai, lưu `pip freeze > requirements.txt` sau khi cài xong.
- Nếu bạn muốn, tôi có thể tạo `requirements.txt` tự động từ môi trường hiện tại.

---

Nếu cần tôi sẽ mở rộng nội dung (cài đặt cụ thể cho Windows, Linux, cấu hình Spark local, hoặc tạo `requirements.txt`).

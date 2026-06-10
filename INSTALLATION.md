# Hướng dẫn cài đặt

Tài liệu này hướng dẫn cách thiết lập môi trường và chạy phân tích trong dự án Stock Analysis (PySpark).

**Yêu cầu trước**

- Python 3.10 hoặc mới hơn (đã kiểm thử trên 3.10)
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

**2) Cài thư viện**

```bash
pip install -r requirements.txt
```

Lệnh này cài toàn bộ thư viện cần thiết: PySpark, TensorFlow, XGBoost, scikit-learn,
pandas, Jupyter (để mở notebook) và Streamlit/Plotly (cho web app demo).

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

- Dữ liệu đã có sẵn trong `csv/` (bản train, đến 11/03/2026) và `csv_demo/` (bản đầy
  đủ cho web app demo). **Không cần chạy `datavn.py`/`dataglobal.py`** trừ khi muốn
  cập nhật dữ liệu mới.
- Nếu gặp lỗi liên quan tới Java/PySpark, kiểm tra `JAVA_HOME` và phiên bản JDK (`java -version`).
- Web app demo: xem hướng dẫn riêng trong [demo/README.md](demo/README.md).

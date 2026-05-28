# Phân tích Thị trường Chứng khoán và Dự báo Xu hướng Giá sử dụng PySpark

Đồ án tốt nghiệp — Phạm Nguyễn Trí Vinh

---

## Tổng quan

Dự án dự báo **xu hướng giá chứng khoán** (tăng/giảm) cho ngày tiếp theo dựa trên dữ liệu lịch sử từ **2013 đến 2026**, áp dụng Machine Learning và Deep Learning với nền tảng xử lý dữ liệu lớn PySpark.

**Phạm vi dữ liệu:**
- **65 cổ phiếu** gồm 27 mã US (NYSE/NASDAQ) và 33 mã Việt Nam (HOSE)
- Dữ liệu OHLCV theo ngày, ~196,000 bản ghi
- Macro context: VN-Index, S&P 500, VIX, TNX (10Y Treasury), IRX (3M T-Bill)

**Kết quả nổi bật:**
| Thị trường | Accuracy | AUC | Edge vs Naive |
|------------|----------|-----|---------------|
| Việt Nam (VN) | ~57–60% | ~0.63 | +7.3% |
| Hoa Kỳ (US) | ~51% | ~0.53 | +0.8% |

> Thị trường VN dự báo tốt hơn US ~9 lần — phù hợp với lý thuyết Efficient Market Hypothesis (Fama, 1970).

---

## Cấu trúc dự án

```
DoAnTotNgiep/
├── Stock_Analysis_PySpark.ipynb   # Notebook chính — toàn bộ pipeline phân tích
├── dataglobal.py                  # Thu thập dữ liệu cổ phiếu US (yfinance)
├── datavn.py                      # Thu thập dữ liệu cổ phiếu VN (vnstock)
├── csv/                           # Dữ liệu CSV (65 file cổ phiếu + macro)
├── figures/                       # Biểu đồ xuất ra
├── requirements.txt               # Danh sách thư viện
└── INSTALLATION.md                # Hướng dẫn cài đặt chi tiết
```

---

## Quy trình phân tích (Pipeline)

```
Thu thập dữ liệu
    ↓
Nạp & khám phá dữ liệu (EDA)
    ↓
Tiền xử lý (PySpark)
    ↓
Feature Engineering (34 features kỹ thuật)
    ↓
Tách thị trường US / VN + Tạo label
    ↓
Huấn luyện mô hình (Spark ML + XGBoost + LSTM/GRU)
    ↓
Đánh giá & Backtest chiến lược giao dịch
```

### Feature Engineering (34 features)

| Nhóm | Features |
|------|----------|
| Return & Lag | `daily_return`, `lag1–3_return`, `lag5/10_return` |
| Moving Average | MA5, MA10, MA20, MA50, `price_vs_ma` |
| Momentum | RSI(7/14/21), Stochastic %K, Williams %R, CCI(14), Momentum 5/10d |
| Trend | MACD, MACD Histogram, MACD Signal, ADX(14) |
| Volatility | ATR(14), ATR Ratio, Bollinger Bands (upper/lower/bandwidth/%B), Volatility 5/10/20d |
| Volume | Volume change, Volume MA ratio, OBV signal, Volume spike 3d |
| Macro | VN-Index return/momentum, S&P 500 return/momentum, VIX level, TNX/IRX rates |
| Đặc thù VN | Limit hit rate, Zero change rate, Vol consistency, Intraday position, ATO gap |

### Mô hình sử dụng

- **PySpark ML**: Logistic Regression, Random Forest, Gradient Boosted Trees, Linear SVC, Ensemble
- **XGBoost**: với EMA MACD thực (pandas/sklearn)
- **Deep Learning**: LSTM, GRU (TensorFlow/Keras)

---

## Cài đặt

**Yêu cầu:**
- Python 3.8+
- Java JDK 8+ (cho PySpark)

**Cài đặt môi trường:**

```powershell
# Windows PowerShell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Xem [INSTALLATION.md](INSTALLATION.md) để biết thêm chi tiết cài đặt Java và cấu hình Spark.

---

## Thu thập dữ liệu

```bash
# Tải dữ liệu cổ phiếu US (AAPL, MSFT, NVDA, ... + VIX, S&P500, TNX, IRX)
python dataglobal.py

# Tải dữ liệu cổ phiếu VN (VCB, HPG, FPT, ... + VN-Index)
python datavn.py
```

Dữ liệu sẽ được lưu vào thư mục `csv/` theo định dạng:

| Cột | Mô tả |
|-----|-------|
| `time` | Ngày giao dịch |
| `open` | Giá mở cửa |
| `high` | Giá cao nhất |
| `low` | Giá thấp nhất |
| `close` | Giá đóng cửa |
| `volume` | Khối lượng giao dịch |

---

## Chạy phân tích

```bash
jupyter lab
```

Mở [Stock_Analysis_PySpark.ipynb](Stock_Analysis_PySpark.ipynb) và chạy tuần tự các phần.

**Cấu hình SparkSession đề xuất (local):**

```python
spark = SparkSession.builder \
    .appName("StockAnalysisPySpark") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
```

---

## Thư viện chính

| Thư viện | Mục đích |
|----------|----------|
| `pyspark` | Xử lý dữ liệu phân tán, Spark ML |
| `tensorflow` / `keras` | Mô hình LSTM, GRU |
| `xgboost` | Gradient Boosting classifier |
| `scikit-learn` | Preprocessing, metrics |
| `pandas` / `numpy` | Xử lý dữ liệu dạng bảng |
| `matplotlib` / `seaborn` | Trực quan hóa |
| `yfinance` | Thu thập dữ liệu US (Yahoo Finance) |
| `vnstock` | Thu thập dữ liệu cổ phiếu Việt Nam |

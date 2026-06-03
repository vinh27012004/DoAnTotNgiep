# Demo trình bày hội đồng — Web app dự báo xu hướng giá

Web app **Streamlit** tương tác, dùng để demo đồ án trước hội đồng. App tái sử dụng
toàn bộ feature engineering & mô hình XGBoost của notebook chính, nhưng chạy bằng
**pandas (không cần Spark/Java)** nên khởi động nhanh và ổn định khi trình chiếu.

## Chạy nhanh (1 lệnh)

```powershell
.\demo\run_demo.ps1
```

Script sẽ tự cài thư viện thiếu, train model nếu chưa có, rồi mở app tại
<http://localhost:8501>.

## Chạy thủ công

```powershell
# 1. Cài thư viện (lần đầu)
.\.venv\Scripts\python.exe -m pip install streamlit plotly

# 2. Train model (lần đầu, ~1-2 phút) — tạo demo/artifacts/
.\.venv\Scripts\python.exe demo\train_models.py

# 3. Mở app
.\.venv\Scripts\python.exe -m streamlit run demo\app.py
```

## Ba màn hình demo

| Tab | Nội dung | Dùng để nói gì với hội đồng |
|-----|----------|------------------------------|
| **📊 Dữ liệu & Biểu đồ** | Nến giá + Bollinger + MA20, khối lượng, RSI; bảng 34+ chỉ báo phiên gần nhất | "Đây là dữ liệu thật và bộ chỉ báo kỹ thuật mô hình học từ đó" |
| **🔮 Dự báo** | Chọn mã → đồng hồ xác suất TĂNG phiên kế tiếp + backtest 30 phiên gần nhất (dự báo vs thực tế) | "Mô hình dự báo trực tiếp, và đây là độ chính xác kiểm chứng" |
| **🎯 Hiệu năng** | Accuracy/AUC/Edge của VN vs US, confusion matrix, ROC, độ chính xác từng mã, feature importance | "VN dự báo tốt hơn US — minh chứng cho EMH" |

## Kịch bản trình bày gợi ý (5–7 phút)

1. **Mở tab Dữ liệu**, chọn một mã quen thuộc (VCB / FPT / AAPL) → giới thiệu dữ liệu OHLCV
   2013–2026 và các chỉ báo kỹ thuật (RSI, MACD, Bollinger).
2. **Sang tab Dự báo**, chọn cùng mã → cho hội đồng thấy mô hình ra xác suất tăng/giảm
   cho phiên kế tiếp; cuộn xuống bảng backtest để chứng minh dự báo có kiểm chứng.
3. **Sang tab Hiệu năng** → nhấn mạnh phát hiện chính: **VN có edge dương, US edge ~0**,
   liên hệ Giả thuyết Thị trường Hiệu quả (Fama, 1970).

## Lưu ý kỹ thuật

- Mô hình demo: **XGBoost**, tách dữ liệu out-of-time (Train ≤ 2021, Test ≥ 2022),
  đúng như pipeline `run_xgb_pipeline()` trong notebook.
- Notebook gốc còn có Spark ML (LR/RF/GBT/SVC/Ensemble) và Deep Learning (LSTM/GRU);
  app chỉ dùng XGBoost để demo nhẹ và nhanh.
- File sinh ra trong `demo/artifacts/` (model `.pkl` + metrics `.json`) — train lại
  bất cứ lúc nào bằng `python demo/train_models.py`.

## Cấu trúc

```
demo/
├── features.py        # Feature engineering pandas (tái tạo PHẦN 5 notebook)
├── train_models.py    # Train XGBoost US & VN, lưu artifacts
├── app.py             # Web app Streamlit 3 tab
├── run_demo.ps1       # Script chạy 1 lệnh
├── artifacts/         # Model + metrics (tự sinh)
└── README.md
```

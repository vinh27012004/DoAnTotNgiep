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
| **📅 Walk-Forward** | Accuracy/AUC qua từng năm 2019–2024 (train lại mỗi năm, expanding window) cho VN & US | "Mô hình ổn định qua nhiều năm, không ăn may một giai đoạn — kiểm định không lookahead" |
| **🎯 Hiệu năng** | Accuracy/AUC/Edge của VN vs US, confusion matrix, ROC, độ chính xác từng mã, feature importance | "VN dự báo tốt hơn US — minh chứng cho EMH" |

## Kịch bản trình bày gợi ý (5–7 phút)

1. **Mở tab Dữ liệu**, chọn một mã quen thuộc (VCB / FPT / AAPL) → giới thiệu dữ liệu OHLCV
   2013–2026 và các chỉ báo kỹ thuật (RSI, MACD, Bollinger).
2. **Sang tab Dự báo**, chọn cùng mã → cho hội đồng thấy mô hình ra xác suất tăng/giảm
   cho phiên kế tiếp; cuộn xuống bảng backtest để chứng minh dự báo có kiểm chứng.
3. **Sang tab Hiệu năng** → nhấn mạnh phát hiện chính: **VN có edge dương, US edge ~0**,
   liên hệ Giả thuyết Thị trường Hiệu quả (Fama, 1970).

## Tự động cập nhật dữ liệu (csv_demo)

Khi mở app, app **tự kiểm tra** dữ liệu trong `csv_demo/` có tới phiên giao dịch gần
nhất chưa. Nếu cũ, app **tự tải bổ sung** (incremental) ngay lúc khởi động, có thanh
tiến trình:

- **US + macro** (VIX/SP500/TNX/IRX): qua `yfinance` — nhanh (vài giây).
- **VN**: qua `vnstock` — chậm hơn do rate-limit (~1.5–3 phút lần đầu trong ngày).
- **VNINDEX**: tải từ `vnstock` rồi **chia 1000** để khớp scale file cũ (file ghi
  `1.67` = ~1674 điểm), giữ feature `vni_*` chính xác.

Cơ chế an toàn:

- Chỉ chạy **1 lần / ngày** (marker `csv_demo/.last_sync.json`) và 1 lần / phiên app.
- Mỗi mã được ghi ngay sau khi tải → lần mở sau tự backfill phần còn thiếu nếu bị ngắt.
- Mọi lỗi mạng/thư viện đều được bắt, **không làm sập app**; chỉ ghi marker khi không lỗi.
- **Tắt** auto-update: đặt biến môi trường `DEMO_AUTO_UPDATE=0`.
  Chỉnh độ trễ vnstock: `VNSTOCK_REQUEST_DELAY` (mặc định `2.0` giây).
- Cập nhật thủ công bất kỳ lúc nào: nút **🔄 Dữ liệu csv_demo → Cập nhật ngay** ở sidebar,
  hoặc chạy `python demo/update_data.py` (thêm `--force` để ép cập nhật).

## Multi-model router — chọn model tốt nhất cho từng mã

`python demo/train_multi_models.py` train **model zoo** (LR, RF, GBT, SVC, XGBoost,
Ensemble — khớp Spark ML của notebook; thêm **GRU** cho VN từ artifact) trên toàn thị
trường, rồi **với mỗi mã chọn model có Edge cao nhất trên tập validation (2022–2023)**
và báo cáo trên **test (≥2024)** — tách out-of-time, không data snooping. Sinh ra:

- `artifacts/multi_models_{mkt}.pkl` — bundle {scaler, features, models}.
- `artifacts/router_{mkt}.json` — lựa chọn model/mã + metric test + đối chứng.

Trong app:
- **Tab Dự báo** có selector **Model**: *Auto* (model tốt nhất cho mã đó) hoặc chỉ định
  1 model để so sánh.
- **Tab Hiệu năng → 🧭 Model theo mã**: bảng model được chọn/mã, phân bố model, và
  **đối chứng trung thực** router-theo-mã vs 1 model global.

⚠️ **Phát hiện (trung thực):** định tuyến model theo từng mã **không** vượt trội hơn dùng
1 model tốt cho cả thị trường — chênh lệch nằm trong khoảng nhiễu. Mỗi mã chỉ có ~130
phiên validation nên "model thắng" phần lớn là may rủi (overfitting khi lựa chọn). Router
vì vậy đã **regularize**: chỉ đổi model khi vượt ≥5 điểm % edge, nên thực tế phần lớn mã
rút về model global-best (VN: GBT, US: XGBoost).

## Lưu ý kỹ thuật

- Mô hình demo: **XGBoost** (mặc định, nhẹ, không cần TF/Spark), tách dữ liệu out-of-time
  (Train ≤ 2021, Test ≥ 2022) như pipeline `run_xgb_pipeline()` trong notebook.
- **GRU cho VN:** train & xuất bằng script demo riêng (không còn train trong notebook):
  `python demo/train_gru.py --market VN` → sinh `gru_VN.keras` (+scaler, meta) vào
  `demo/artifacts/`. Khi có artifact:
  - Tab **Dự báo** dùng được **GRU** cho mã VN (lazy-load TensorFlow, chuỗi 20 ngày).
  - `train_multi_models.py` thêm **GRU** làm candidate khi chọn model/mã.
  - Nếu chưa có, app/router tự bỏ qua GRU và fallback — vẫn chạy bình thường.
- **Walk-Forward (notebook):** tab Walk-Forward đọc `walkforward.json` (GRU VN / GBT US)
  do **notebook** export (walk-forward retrain mỗi fold, ~30 phút/fold — quá nặng để đưa
  vào demo). Nếu thiếu file này, tab fallback kết quả XGBoost của `train_models.py`.
- Notebook gốc còn có Spark ML (LR/RF/GBT/SVC/Ensemble) và Deep Learning (LSTM/GRU).
- File sinh trong `demo/artifacts/`: `model_*.pkl` + `metrics_*.json` (XGBoost, từ
  `train_models.py`); `multi_models_*.pkl` + `router_*.json` (từ `train_multi_models.py`);
  `gru_VN.*` (từ `train_gru.py`); `walkforward.json` (từ notebook).

## Cấu trúc

```
demo/
├── features.py            # Feature engineering pandas (tái tạo PHẦN 5 notebook)
├── update_data.py         # Tự kiểm tra & cập nhật csv_demo khi mở app (US/VN/VNINDEX)
├── train_models.py        # Train XGBoost US & VN (+ ticker_acc, ticker_edge), lưu artifacts
├── train_multi_models.py  # Train model zoo (LR/RF/GBT/SVC/XGB/ENS+GRU) + router chọn model/mã
├── train_gru.py           # Train & xuất model GRU (VN) — tách từ notebook sang demo
├── app.py                 # Web app Streamlit 4 tab
├── run_demo.ps1       # Script chạy 1 lệnh
├── artifacts/         # Model + metrics (tự sinh)
└── README.md
```

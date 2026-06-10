import os
import time

from vnstock import Quote, register_user
import pandas as pd
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value

# Danh sách 33 mã VN — VN30 đầy đủ + midcap quan trọng
symbols = [
    # Banking (8)
    'VCB', 'BID', 'CTG', 'TCB', 'MBB', 'ACB', 'STB', 'HDB',
    # Real estate (6)
    'VIC', 'VHM', 'KDH', 'NLG', 'VRE', 'BCM',
    # Consumer & Retail (5)
    'VNM', 'MSN', 'SAB', 'MWG', 'PNJ',
    # Steel & Materials (3)
    'HPG', 'HSG', 'NKG',
    # Technology (1)
    'FPT',
    # Energy & Oil (3)
    'GAS', 'PLX', 'POW',
    # Aviation (1)
    'VJC',
    # Utilities & Infra (2)
    'REE', 'SSI',
    # Securities (1)
    'VPB',
    # Pharma (1)
    'DHG',
    # Logistics (1)
    'GMD',
    # Insurance (1)
    'BVH',
]

from datetime import datetime

# Thiết lập khoảng thời gian (Lấy ít nhất 3-5 năm để dự báo chính xác)
start_date = '2014-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
load_env_file(Path(".env"))
api_key = os.getenv("VNSTOCK_API_KEY", "").strip()
# Tăng delay lên 3.5s để tránh rate limit ở chế độ Guest
request_delay_seconds = float(os.getenv("VNSTOCK_REQUEST_DELAY", "3.5"))

if api_key:
    if register_user(api_key=api_key):
        print("Đã kích hoạt VNSTOCK API key.")
    else:
        print("Không đăng ký được VNSTOCK API key, tiếp tục ở chế độ Guest.")
else:
    print("Không tìm thấy VNSTOCK_API_KEY trong môi trường, đang chạy Guest.")

try:
    from vnai import check_api_key_status
    status = check_api_key_status()
    print(f"Trạng thái API key: {status.get('tier', 'unknown')} | {status.get('limits', {})}")
except Exception:
    pass

all_data = []

# Chia 2 bộ dữ liệu:
#   csv/      -> bản train (cắt đến CUTOFF) cho notebook nộp thầy
#   csv_demo/ -> bản đầy đủ cho app demo
CUTOFF = '2026-03-11'
train_dir = Path("csv")
demo_dir = Path("csv_demo")
train_dir.mkdir(parents=True, exist_ok=True)
demo_dir.mkdir(parents=True, exist_ok=True)


def save_split(data, name):
    """Lưu bản đầy đủ vào csv_demo/ và bản cắt (time <= CUTOFF) vào csv/."""
    data.to_csv(demo_dir / f"{name}.csv", index=False)
    cut = data[pd.to_datetime(data['time']) <= pd.Timestamp(CUTOFF)]
    cut.to_csv(train_dir / f"{name}.csv", index=False)
    return len(cut), len(data)

for symbol in symbols:
    print(f"Đang tải dữ liệu cho mã: {symbol}...")
    # Lấy dữ liệu lịch sử giá, với logic retry mạnh mẽ hơn
    quote = Quote(symbol=symbol, source='VCI', show_log=False)
    df = None
    for attempt in range(3):  # Thử lại tối đa 3 lần
        try:
            df = quote.history(start=start_date, end=end_date, interval='1D')
            break  # Thành công thì thoát loop
        except Exception as exc:
            message = str(exc).lower()
            if "rate limit" in message or "giới hạn" in message:
                wait_time = 60  # Chờ 60 giây
                print(f"Chạm giới hạn khi tải {symbol} (lần {attempt + 1}), "
                      f"chờ {wait_time} giây rồi thử lại...")
                time.sleep(wait_time)
            else:
                raise  # Lỗi khác thì báo ngay
    if df is None:
        print(f"  Bỏ qua {symbol} sau 3 lần thử không thành công.")
        continue
    df['symbol'] = symbol  # Thêm cột để biết mã nào

    # Lưu bản đầy đủ (csv_demo/) + bản cắt train (csv/)
    per_symbol = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    n_train, n_full = save_split(per_symbol, symbol)
    print(f"Hoàn thành lưu {symbol}: csv_demo {n_full} dòng, "
          f"csv {n_train} dòng (<= {CUTOFF})")

    all_data.append(df)

    if request_delay_seconds > 0:
        time.sleep(request_delay_seconds)

print("Hoàn thành lưu dữ liệu!")
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

# Thiết lập khoảng thời gian (Lấy ít nhất 3-5 năm để dự báo chính xác)
start_date = '2014-01-01'
end_date = '2026-03-10'
load_env_file(Path(".env"))
api_key = os.getenv("VNSTOCK_API_KEY", "").strip()
request_delay_seconds = float(os.getenv("VNSTOCK_REQUEST_DELAY", "1.0"))

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
output_dir = Path("csv")
output_dir.mkdir(parents=True, exist_ok=True)

for symbol in symbols:
    print(f"Đang tải dữ liệu cho mã: {symbol}...")
    # Lấy dữ liệu lịch sử giá
    quote = Quote(symbol=symbol, source='VCI', show_log=False)
    try:
        df = quote.history(start=start_date, end=end_date, interval='1D')
    except Exception as exc:
        message = str(exc).lower()
        if "rate limit" in message or "giới hạn" in message:
            print(f"Chạm giới hạn khi tải {symbol}, chờ 25 giây rồi thử lại...")
            time.sleep(25)
            df = quote.history(start=start_date, end=end_date, interval='1D')
        else:
            raise
    df['symbol'] = symbol  # Thêm cột để biết mã nào

    # Lưu từng mã để pipeline ingest xử lý tự động
    per_symbol = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    output_file = output_dir / f"{symbol}.csv"
    per_symbol.to_csv(output_file, index=False)
    print(f"Hoàn thành lưu {output_file} ({len(per_symbol)} dòng)")

    all_data.append(df)

    if request_delay_seconds > 0:
        time.sleep(request_delay_seconds)

# Gộp tất cả và lưu thành CSV để nạp vào PySpark
final_df = pd.concat(all_data, ignore_index=True)
# Đặt cột symbol lên đầu để dễ nhìn
final_df = final_df[['symbol', 'time', 'open', 'high', 'low', 'close', 'volume']]
final_df.to_csv('vietnam_stocks.csv', index=False)
print("Hoàn thành lưu dữ liệu!")
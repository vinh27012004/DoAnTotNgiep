import yfinance as yf
import pandas as pd
from pathlib import Path

# Danh sách các mã cần lấy
symbols = [
    # US mega-cap tech
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META',
    # EV / Auto
    'TSLA', 'RIVN', 'LCID', 'F', 'GM'
]

# Thiết lập khoảng thời gian
start_date = '2013-01-01'
end_date = '2026-03-10'

output_dir = Path("csv")
output_dir.mkdir(parents=True, exist_ok=True)

print("Đang tải dữ liệu US stocks...\n")

for symbol in symbols:
    print(f"Đang tải dữ liệu cho mã: {symbol}...")
    
    # Lấy dữ liệu lịch sử giá
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Reset index để cột Date trở thành một cột dữ liệu bình thường
    data.reset_index(inplace=True)
    
    # Rename cột để match với format
    data.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    
    # Lưu thành CSV riêng
    output_file = output_dir / f"{symbol}.csv"
    data.to_csv(output_file, index=False)
    print(f"Hoàn thành lưu {output_file} ({len(data)} dòng)\n")

print("Hoàn thành lưu tất cả dữ liệu!")
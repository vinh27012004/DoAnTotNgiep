import yfinance as yf
import pandas as pd

# Danh sách các mã cần lấy
symbols = ['AAPL', 'TSLA']  # Apple và Tesla

# Thiết lập khoảng thời gian
start_date = '2019-01-01'
end_date = '2026-03-10'

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
    data.to_csv(f'{symbol}.csv', index=False)
    print(f"Hoàn thành lưu {symbol}.csv ({len(data)} dòng)\n")

print("Hoàn thành lưu tất cả dữ liệu!")
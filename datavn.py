from vnstock import Quote
import pandas as pd

# Danh sách các mã cần lấy hoặc lấy cả rổ VN30
symbols = ['VCB', 'FPT', 'VIC', 'VNM', 'HPG'] 

# Thiết lập khoảng thời gian (Lấy ít nhất 3-5 năm để dự báo chính xác)
start_date = '2019-01-01'
end_date = '2026-03-10'

all_data = []

for symbol in symbols:
    print(f"Đang tải dữ liệu cho mã: {symbol}...")
    # Lấy dữ liệu lịch sử giá
    quote = Quote(symbol=symbol, source='VCI')
    df = quote.history(start=start_date, end=end_date, interval='1D')
    df['symbol'] = symbol  # Thêm cột để biết mã nào
    all_data.append(df)

# Gộp tất cả và lưu thành CSV để nạp vào PySpark
final_df = pd.concat(all_data, ignore_index=True)
# Đặt cột symbol lên đầu để dễ nhìn
final_df = final_df[['symbol', 'time', 'open', 'high', 'low', 'close', 'volume']]
final_df.to_csv('vietnam_stocks.csv', index=False)
print("Hoàn thành lưu dữ liệu!")
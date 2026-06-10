import yfinance as yf
import pandas as pd
from pathlib import Path

# Danh sách 28 mã US — mở rộng cho nghiên cứu đầy đủ
symbols = [
    # US mega-cap tech (6)
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META',
    # Semiconductor / Tech extra (2)
    'NFLX', 'AMD',
    # EV / Auto (3)
    'TSLA', 'F', 'GM',
    # Banking & Finance (4)
    'JPM', 'BAC', 'WFC', 'GS',
    # Energy / Oil & Gas (2)
    'XOM', 'CVX',
    # Consumer & Retail (3)
    'DIS', 'COST', 'WMT',
    # Healthcare / Pharma (2)
    'UNH', 'LLY',
    # Payments / Fintech (2)
    'V', 'MA',
    # Semiconductor extra (1)
    'INTC',
    # Aerospace & Defense (1)
    'BA',
    # Telecom (1)
    'VZ',
]

from datetime import datetime

# Thiết lập khoảng thời gian
start_date = '2013-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

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


print("Đang tải dữ liệu US stocks...\n")

ok, fail = [], []
for symbol in symbols:
    print(f"Đang tải dữ liệu cho mã: {symbol}...")
    try:
        data = yf.download(symbol, start=start_date, end=end_date,
                           progress=False, auto_adjust=True)
        if data is None or len(data) == 0:
            print(f"  Bỏ qua {symbol}: yfinance trả về 0 dòng")
            fail.append(symbol)
            continue

        data = data.reset_index()
        # Flatten MultiIndex columns nếu có (yfinance phiên bản mới)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

        col_map = {'Date': 'time', 'Open': 'open', 'High': 'high',
                   'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        data = data.rename(columns=col_map)
        keep = ['time', 'open', 'high', 'low', 'close', 'volume']
        data = data[[c for c in keep if c in data.columns]]

        n_train, n_full = save_split(data, symbol)
        print(f"  Hoàn thành {symbol}: csv_demo {n_full} dòng, "
              f"csv {n_train} dòng (<= {CUTOFF})")
        ok.append(symbol)
    except Exception as e:
        print(f"  LỖI {symbol}: {e}")
        fail.append(symbol)

print(f"\nHoàn thành: {len(ok)}/{len(symbols)} mã. Lỗi: {fail}")

# Tải VIX index (chỉ số sợ hãi thị trường) — macro feature cho US
print("\nĐang tải VIX index (^VIX)...")
try:
    vix = yf.download('^VIX', start=start_date, end=end_date,
                      progress=False, auto_adjust=True)
    vix = vix.reset_index()
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix = vix.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high',
                               'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    vix = vix[[c for c in ['time','open','high','low','close','volume'] if c in vix.columns]]
    n_train, n_full = save_split(vix, "VIX")
    print(f"  Hoàn thành VIX: csv_demo {n_full} dòng, csv {n_train} dòng (<= {CUTOFF})")
except Exception as e:
    print(f"  LỖI VIX: {e}")

# Tải S&P 500 index làm macro context cho US (đối xứng VN-Index cho VN)
print("\nĐang tải S&P 500 index (^GSPC)...")
try:
    sp = yf.download('^GSPC', start=start_date, end=end_date,
                     progress=False, auto_adjust=True)
    sp = sp.reset_index()
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = [c[0] if isinstance(c, tuple) else c for c in sp.columns]
    sp = sp.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high',
                            'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    sp = sp[[c for c in ['time','open','high','low','close','volume'] if c in sp.columns]]
    n_train, n_full = save_split(sp, "SP500")
    print(f"  Hoàn thành SP500: csv_demo {n_full} dòng, csv {n_train} dòng (<= {CUTOFF})")
except Exception as e:
    print(f"  LỖI S&P500: {e}")

# Tải lãi suất Fed — dùng ^TNX (10-year Treasury yield) và ^IRX (3-month T-bill)
for ticker, fname, label in [
    ('^TNX', 'TNX.csv',  '10-Year Treasury Yield'),
    ('^IRX', 'IRX.csv',  '3-Month T-Bill Rate'),
]:
    print(f"\nĐang tải {label} ({ticker})...")
    try:
        df_r = yf.download(ticker, start=start_date, end=end_date,
                           progress=False, auto_adjust=True)
        df_r = df_r.reset_index()
        if isinstance(df_r.columns, pd.MultiIndex):
            df_r.columns = [c[0] if isinstance(c, tuple) else c for c in df_r.columns]
        df_r = df_r.rename(columns={'Date': 'time', 'Close': 'close'})
        df_r = df_r[['time', 'close']].rename(columns={'close': 'rate'})
        n_train, n_full = save_split(df_r, Path(fname).stem)
        print(f"  Hoàn thành {label}: csv_demo {n_full} dòng, "
              f"csv {n_train} dòng (<= {CUTOFF})")
    except Exception as e:
        print(f"  LỖI {ticker}: {e}")
"""
Feature engineering bằng pandas — tái tạo PHẦN 5 của notebook (Spark Window Functions)
theo đúng cách dùng trong run_xgb_pipeline() (EMA MACD thật, không cần Spark).

Dùng cho app demo: nhanh, không phụ thuộc PySpark/Java.
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

# ── Thư mục dữ liệu CSV (mặc định: <repo>/csv) ──────────────────────────────
CSV_DIR = Path(__file__).resolve().parent.parent / "csv"

# ── Danh sách mã (khớp notebook PHẦN 7) ─────────────────────────────────────
US_TICKERS = [
    'AAPL', 'AMZN', 'F', 'GM', 'GOOGL', 'META', 'MSFT', 'NFLX', 'AMD', 'NVDA', 'TSLA',
    'JPM', 'BAC', 'WFC', 'GS', 'XOM', 'CVX', 'DIS', 'COST', 'WMT', 'UNH', 'LLY', 'V', 'MA',
    'INTC', 'BA', 'VZ',
]
VN_TICKERS = [
    'ACB', 'BID', 'CTG', 'FPT', 'GAS', 'HPG', 'HSG', 'KDH', 'MBB', 'MSN', 'NKG', 'NLG',
    'PNJ', 'SAB', 'SSI', 'TCB', 'VCB', 'VHM', 'VIC', 'VNM', 'VPB',
    'STB', 'HDB', 'VRE', 'BCM', 'MWG', 'PLX', 'POW', 'VJC', 'REE', 'DHG', 'GMD', 'BVH',
]

SECTOR_MAP = {
    'ACB': 0, 'BID': 0, 'CTG': 0, 'MBB': 0, 'TCB': 0, 'VCB': 0, 'VPB': 0, 'STB': 0, 'HDB': 0,
    'SSI': 1,
    'HPG': 2, 'HSG': 2, 'NKG': 2,
    'KDH': 3, 'NLG': 3, 'VHM': 3, 'VIC': 3, 'VRE': 3, 'BCM': 3,
    'MSN': 4, 'SAB': 4, 'VNM': 4, 'PNJ': 4, 'MWG': 4,
    'FPT': 5,
    'AAPL': 6, 'MSFT': 6, 'GOOGL': 6, 'META': 6, 'NVDA': 6, 'NFLX': 6, 'AMD': 6, 'INTC': 6,
    'TSLA': 7, 'F': 7, 'GM': 7,
    'JPM': 8, 'BAC': 8, 'WFC': 8, 'GS': 8,
    'AMZN': 9, 'DIS': 9, 'COST': 9, 'WMT': 9,
    'XOM': 10, 'CVX': 10,
    'UNH': 11, 'LLY': 11,
    'V': 12, 'MA': 12,
    'BA': 13, 'VZ': 13,
    'GAS': 14, 'PLX': 14, 'POW': 14, 'REE': 14,
    'VJC': 15, 'GMD': 15,
    'DHG': 16, 'BVH': 16,
}

# ── Feature lists (khớp notebook PHẦN 8) ────────────────────────────────────
COMMON_FEATURES = [
    'daily_return', 'lag1_return', 'lag2_return', 'lag3_return', 'lag5_return', 'lag10_return',
    'price_vs_ma5', 'price_vs_ma20', 'price_vs_ma50', 'momentum_5', 'momentum_10',
    'rsi_7', 'rsi_14', 'macd_hist', 'stoch_k', 'williams_r', 'cci14',
    'rolling_volatility_5', 'rolling_volatility_20', 'bb_bandwidth', 'bb_pct_b', 'atr_ratio', 'adx14',
    'log_volume', 'log_lag1_volume', 'volume_change', 'volume_ma_ratio',
    'high_low_range', 'close_open_return', 'obv_signal',
    'intraday_pos', 'vol_consistency',
    'sector_idx', 'ticker_idx',
    'bb_position', 'overnight_gap', 'up_days_5',
    'proximity_52w_high', 'volume_spike_3d',
    'ma5_vs_ma20', 'ma20_vs_ma50', 'hl_ratio',
]
VN_ONLY_FEATURES = [
    'limit_hit_rate', 'zero_change_rate', 'ato_gap',
    'vni_ret1d', 'vni_mom5', 'vni_ma_ratio', 'rs_vs_vnindex', 'price_percentile_20',
]
US_ONLY_FEATURES = [
    'sp500_ret1d', 'sp500_mom5', 'sp500_ma_ratio',
    'vix_level', 'vix_ret1d', 'vix_ma_ratio',
    'tnx_rate', 'tnx_chg1d', 'tnx_spread',
    'irx_rate', 'irx_chg1d', 'irx_spread',
    'rs_vs_sp500',
]
FEATURE_COLS_US = COMMON_FEATURES + US_ONLY_FEATURES
FEATURE_COLS_VN = COMMON_FEATURES + VN_ONLY_FEATURES


def xgb_feature_cols(base_feature_cols):
    """Khớp run_xgb_pipeline: bỏ ticker_idx/macd_hist/log_volume/log_lag1_volume,
    thêm log_volume, log_lag1_volume, macd_real, macd_hist_real."""
    base = [c for c in base_feature_cols
            if c not in ('ticker_idx', 'macd_hist', 'log_volume', 'log_lag1_volume')]
    return base + ['log_volume', 'log_lag1_volume', 'macd_real', 'macd_hist_real']


# ── Helpers đọc dữ liệu ─────────────────────────────────────────────────────
def _read_ohlcv(ticker, csv_dir=CSV_DIR):
    df = pd.read_csv(Path(csv_dir) / f"{ticker}.csv")
    df['time'] = pd.to_datetime(df['time'])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df[(df['close'] > 0) & (df['volume'] > 0)]
    df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    df['ticker'] = ticker
    return df


def load_market_raw(tickers, csv_dir=CSV_DIR):
    frames = []
    for t in tickers:
        p = Path(csv_dir) / f"{t}.csv"
        if p.exists():
            frames.append(_read_ohlcv(t, csv_dir))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(['ticker', 'time']).reset_index(drop=True)


def _macro_context(filename, prefix, csv_dir=CSV_DIR):
    """Tính context features cho 1 chỉ số macro (VNINDEX/SP500/VIX)."""
    p = Path(csv_dir) / filename
    if not p.exists():
        return None
    m = pd.read_csv(p)
    m['time'] = pd.to_datetime(m['time'])
    m['c'] = pd.to_numeric(m['close'], errors='coerce')
    m = m.dropna(subset=['c']).sort_values('time').reset_index(drop=True)
    out = pd.DataFrame({'time': m['time']})
    if prefix in ('vni', 'sp500'):
        out[f'{prefix}_ret1d'] = m['c'].pct_change()
        ma5 = m['c'].rolling(5, min_periods=1).mean()
        ma20 = m['c'].rolling(20, min_periods=1).mean()
        out[f'{prefix}_mom5'] = m['c'] / m['c'].shift(5) - 1
        out[f'{prefix}_ma_ratio'] = ma5 / ma20 - 1
    elif prefix == 'vix':
        out['vix_level'] = m['c']
        out['vix_ret1d'] = m['c'].pct_change()
        vix_ma5 = m['c'].rolling(5, min_periods=1).mean()
        out['vix_ma_ratio'] = m['c'] / vix_ma5 - 1
    return out


# ── Feature engineering chính (pandas, khớp PHẦN 5 notebook) ────────────────
def compute_features(df, market):
    """df: OHLCV nhiều mã (cột time, open, high, low, close, volume, ticker).
    market: 'US' hoặc 'VN'. Trả về DataFrame có đủ features + label + next_close."""
    df = df.sort_values(['ticker', 'time']).reset_index(drop=True).copy()
    g = df.groupby('ticker', group_keys=False)

    def roll_mean(col, n):
        return g[col].transform(lambda s: s.rolling(n, min_periods=1).mean())

    def roll_std(col, n):
        return g[col].transform(lambda s: s.rolling(n, min_periods=1).std(ddof=0))

    def roll_min(col, n):
        return g[col].transform(lambda s: s.rolling(n, min_periods=1).min())

    def roll_max(col, n):
        return g[col].transform(lambda s: s.rolling(n, min_periods=1).max())

    def shift(col, n):
        return g[col].shift(n)

    # Lags & lead
    df['lag1_close'] = shift('close', 1)
    df['lag1_high'] = shift('high', 1)
    df['lag1_low'] = shift('low', 1)
    df['next_close'] = g['close'].shift(-1)
    df['future_return'] = (df['next_close'] - df['close']) / df['close']
    df['daily_return'] = (df['close'] - df['lag1_close']) / df['lag1_close']

    g = df.groupby('ticker', group_keys=False)  # refresh sau khi thêm cột
    df['lag1_return'] = g['daily_return'].shift(1)
    df['lag2_return'] = g['daily_return'].shift(2)
    df['lag3_return'] = g['daily_return'].shift(3)
    df['lag5_return'] = g['daily_return'].shift(5)
    df['lag10_return'] = g['daily_return'].shift(10)

    # MAs
    df['ma5'] = roll_mean('close', 5)
    df['ma10'] = roll_mean('close', 10)
    df['ma20'] = roll_mean('close', 20)
    df['ma50'] = roll_mean('close', 50)
    df['price_vs_ma5'] = (df['close'] - df['ma5']) / df['ma5']
    df['price_vs_ma20'] = (df['close'] - df['ma20']) / df['ma20']
    df['price_vs_ma50'] = np.where(df['ma50'] != 0, (df['close'] - df['ma50']) / df['ma50'], np.nan)

    # Volatility
    df['rolling_volatility_5'] = roll_std('daily_return', 5)
    df['rolling_volatility_10'] = roll_std('daily_return', 10)
    df['rolling_volatility_20'] = roll_std('daily_return', 20)

    # RSI 7/14/21 — Spark rowsBetween(-n,0) = n+1 dòng: rsi14->15, rsi7->8, rsi21->22
    df['price_change'] = df['close'] - df['lag1_close']
    df['gain'] = df['price_change'].clip(lower=0).fillna(0.0)
    df['loss'] = (-df['price_change']).clip(lower=0).fillna(0.0)
    df['avg_gain_14'] = roll_mean('gain', 15)
    df['avg_loss_14'] = roll_mean('loss', 15)
    df['rsi_14'] = np.where(df['avg_loss_14'] == 0, 100.0,
                            100 - 100 / (1 + df['avg_gain_14'] / df['avg_loss_14'].replace(0, np.nan)))
    df['avg_gain_7'] = roll_mean('gain', 8)
    df['avg_loss_7'] = roll_mean('loss', 8)
    df['rsi_7'] = np.where(df['avg_loss_7'] == 0, 100.0,
                           100 - 100 / (1 + df['avg_gain_7'] / df['avg_loss_7'].replace(0, np.nan)))
    df['avg_gain_21'] = roll_mean('gain', 22)
    df['avg_loss_21'] = roll_mean('loss', 22)
    df['rsi_21'] = np.where(df['avg_loss_21'] == 0, 100.0,
                            100 - 100 / (1 + df['avg_gain_21'] / df['avg_loss_21'].replace(0, np.nan)))

    # MACD proxy (SMA) — giữ macd_hist để tương thích; XGB sẽ dùng macd_real
    df['ema12_proxy'] = roll_mean('close', 12)
    df['ema26_proxy'] = roll_mean('close', 26)
    df['macd'] = df['ema12_proxy'] - df['ema26_proxy']
    df['macd_signal'] = roll_mean('macd', 9)
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # MACD thật (EMA) — khớp run_xgb_pipeline
    ema12 = g['close'].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g['close'].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    macd_r = ema12 - ema26
    df['_macd_r'] = macd_r
    macd_sig = df.groupby('ticker')['_macd_r'].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    cs = df['close'].where(df['close'] != 0, np.nan)
    df['macd_real'] = (macd_r / cs).fillna(0)
    df['macd_hist_real'] = ((macd_r - macd_sig) / cs).fillna(0)
    df.drop(columns=['_macd_r'], inplace=True)

    # Bollinger Bands (20)
    df['bb_mid'] = roll_mean('close', 20)
    df['bb_std'] = roll_std('close', 20)
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_bandwidth'] = np.where(df['bb_mid'] != 0, (df['bb_upper'] - df['bb_lower']) / df['bb_mid'], np.nan)
    width = (df['bb_upper'] - df['bb_lower'])
    df['bb_pct_b'] = np.where(width != 0, (df['close'] - df['bb_lower']) / width, np.nan)
    df['bb_position'] = np.where(df['bb_std'] != 0, (df['close'] - df['bb_mid']) / (2 * df['bb_std']), 0.0)

    # Volume
    df['lag1_volume'] = shift('volume', 1)
    df['volume_change'] = np.where((df['lag1_volume'].isna()) | (df['lag1_volume'] == 0),
                                   np.nan, (df['volume'] - df['lag1_volume']) / df['lag1_volume'])
    df['high_low_range'] = np.where(df['close'] != 0, (df['high'] - df['low']) / df['close'], np.nan)
    df['close_open_return'] = np.where(df['open'] != 0, (df['close'] - df['open']) / df['open'], np.nan)
    df['avg_volume_20'] = roll_mean('volume', 20)
    df['volume_ma_ratio'] = np.where(df['avg_volume_20'] == 0, np.nan, df['volume'] / df['avg_volume_20'])

    # Stochastic %K / Williams %R (14)
    df['low14'] = roll_min('low', 14)
    df['high14'] = roll_max('high', 14)
    hl = df['high14'] - df['low14']
    df['stoch_k'] = np.where(df['high14'] == df['low14'], 50.0, (df['close'] - df['low14']) / hl * 100)
    df['williams_r'] = np.where(df['high14'] == df['low14'], -50.0, (df['high14'] - df['close']) / hl * -100)

    # CCI(14)
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3.0
    df['tp_ma14'] = roll_mean('tp', 14)
    df['tp_std14'] = roll_std('tp', 14)
    df['cci14'] = np.where(df['tp_std14'] == 0, 0.0,
                           (df['tp'] - df['tp_ma14']) / (0.015 * df['tp_std14'].replace(0, np.nan)))

    # ADX(14)
    up_move = df['high'] - df['lag1_high']
    down_move = df['lag1_low'] - df['low']
    df['plus_dm'] = np.where(up_move > down_move, np.clip(up_move, 0, None), 0.0)
    df['minus_dm'] = np.where(down_move > up_move, np.clip(down_move, 0, None), 0.0)
    df['true_range'] = np.maximum.reduce([
        (df['high'] - df['low']).values,
        (df['high'] - df['lag1_close']).abs().values,
        (df['low'] - df['lag1_close']).abs().values,
    ])
    df['smooth_tr'] = roll_mean('true_range', 14)
    df['plus_di14'] = np.where(df['smooth_tr'] > 0, 100 * roll_mean('plus_dm', 14) / df['smooth_tr'], 0.0)
    df['minus_di14'] = np.where(df['smooth_tr'] > 0, 100 * roll_mean('minus_dm', 14) / df['smooth_tr'], 0.0)
    di_sum = df['plus_di14'] + df['minus_di14']
    df['adx14'] = np.where(di_sum == 0, 0.0, (df['plus_di14'] - df['minus_di14']).abs() / di_sum * 100)

    # ATR(14) — trung bình (high-low) 14 phiên
    df['hl_raw'] = df['high'] - df['low']
    df['atr14'] = roll_mean('hl_raw', 14)
    df['atr_ratio'] = np.where(df['close'] != 0, df['atr14'] / df['close'], np.nan)

    # OBV signal
    df['price_dir'] = np.where(df['close'] > df['lag1_close'], 1.0,
                               np.where(df['close'] < df['lag1_close'], -1.0, 0.0))
    df['obv_signal'] = roll_mean('price_dir', 5)

    # Momentum
    df['lag5_close'] = shift('close', 5)
    df['lag10_close'] = shift('close', 10)
    df['momentum_5'] = np.where(df['lag5_close'] != 0, (df['close'] - df['lag5_close']) / df['lag5_close'], np.nan)
    df['momentum_10'] = np.where(df['lag10_close'] != 0, (df['close'] - df['lag10_close']) / df['lag10_close'], np.nan)

    # MA cross & range expansion
    df['ma5_vs_ma20'] = np.where(df['ma20'] != 0, df['ma5'] / df['ma20'] - 1, np.nan)
    df['ma20_vs_ma50'] = np.where(df['ma50'] != 0, df['ma20'] / df['ma50'] - 1, np.nan)
    df['hl5'] = roll_mean('high_low_range', 5)
    df['hl20'] = roll_mean('high_low_range', 20)
    df['hl_ratio'] = np.where(df['hl20'] != 0, df['hl5'] / df['hl20'], 1.0)

    # Microstructure / VN-specific
    df['near_limit'] = np.where(df['daily_return'].abs() >= 0.068, 1.0, 0.0)
    df['limit_hit_rate'] = roll_mean('near_limit', 20)
    df['no_change'] = np.where(df['daily_return'] == 0.0, 1.0, 0.0)
    df['zero_change_rate'] = roll_mean('no_change', 10)
    df['vol_std10'] = roll_std('volume', 10)
    df['vol_mean10'] = roll_mean('volume', 10)
    df['vol_consistency'] = np.where(df['vol_mean10'] > 0, df['vol_std10'] / df['vol_mean10'], np.nan)
    hlrange = df['high'] - df['low']
    df['intraday_pos'] = np.where(hlrange > 0, (df['close'] - df['low']) / hlrange, 0.5)
    df['ato_gap'] = np.where(df['lag1_close'] != 0, (df['open'] - df['lag1_close']) / df['lag1_close'], np.nan)
    df['overnight_gap'] = np.where(df['lag1_close'] != 0, (df['open'] - df['lag1_close']) / df['lag1_close'], 0.0)
    df['up_ret'] = np.where(df['daily_return'] > 0, 1.0, 0.0)
    df['up_days_5'] = roll_mean('up_ret', 5)

    # 52w high proximity, 3d volume spike
    df['max_close_252'] = roll_max('close', 252)
    df['proximity_52w_high'] = np.where(df['max_close_252'] > 0, df['close'] / df['max_close_252'], 1.0)
    df['vol3'] = roll_mean('volume', 3)
    df['volume_spike_3d'] = np.where(df['avg_volume_20'] > 0, df['vol3'] / df['avg_volume_20'], 1.0)

    # Price percentile 20
    df['price_min_20'] = roll_min('close', 20)
    df['price_max_20'] = roll_max('close', 20)
    rng20 = df['price_max_20'] - df['price_min_20']
    df['price_percentile_20'] = np.where(rng20 > 0, (df['close'] - df['price_min_20']) / rng20, 0.5)

    # log volume
    df['log_volume'] = np.log1p(df['volume'].clip(lower=0))
    df['log_lag1_volume'] = np.log1p(df['lag1_volume'].clip(lower=0))

    # sector encoding
    df['sector_idx'] = df['ticker'].map(SECTOR_MAP).astype(float)

    # ── Macro context joins ──────────────────────────────────────────────
    df['date'] = df['time'].dt.normalize()
    vni = _macro_context('VNINDEX.csv', 'vni')
    if vni is not None:
        vni['date'] = vni['time'].dt.normalize()
        df = df.merge(vni.drop(columns=['time']), on='date', how='left')
    for c in ['vni_ret1d', 'vni_mom5', 'vni_ma_ratio']:
        if c not in df:
            df[c] = 0.0
    df[['vni_ret1d', 'vni_mom5', 'vni_ma_ratio']] = df[['vni_ret1d', 'vni_mom5', 'vni_ma_ratio']].fillna(0.0)

    sp = _macro_context('SP500.csv', 'sp500')
    if sp is not None:
        sp['date'] = sp['time'].dt.normalize()
        df = df.merge(sp.drop(columns=['time']), on='date', how='left')
    for c in ['sp500_ret1d', 'sp500_mom5', 'sp500_ma_ratio']:
        if c not in df:
            df[c] = 0.0
    df[['sp500_ret1d', 'sp500_mom5', 'sp500_ma_ratio']] = df[['sp500_ret1d', 'sp500_mom5', 'sp500_ma_ratio']].fillna(0.0)

    vix = _macro_context('VIX.csv', 'vix')
    if vix is not None:
        vix['date'] = vix['time'].dt.normalize()
        df = df.merge(vix.drop(columns=['time']), on='date', how='left')
    for c in ['vix_level', 'vix_ret1d', 'vix_ma_ratio']:
        if c not in df:
            df[c] = 0.0
    df[['vix_level', 'vix_ret1d', 'vix_ma_ratio']] = df[['vix_level', 'vix_ret1d', 'vix_ma_ratio']].fillna(0.0)

    # TNX/IRX không có file -> 0 (khớp notebook khi thiếu)
    for c in ['tnx_rate', 'tnx_chg1d', 'tnx_spread', 'irx_rate', 'irx_chg1d', 'irx_spread']:
        df[c] = 0.0

    df['rs_vs_vnindex'] = df['daily_return'] - df['vni_ret1d']
    df['rs_vs_sp500'] = df['daily_return'] - df['sp500_ret1d']

    df['year'] = df['time'].dt.year
    return df


def make_labels(df, market):
    """Tạo label next-day. US threshold 1.0%, VN threshold 2.0% (khớp notebook PHẦN 7)."""
    thr = 0.010 if market == 'US' else 0.020
    out = df.copy()
    out['label'] = np.where(out['future_return'] > thr, 1.0,
                            np.where(out['future_return'] < -thr, 0.0, np.nan))
    return out

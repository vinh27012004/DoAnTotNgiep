"""
Demo Streamlit — Phân tích & Dự báo Xu hướng Giá Chứng khoán (PySpark + ML)
Đồ án tốt nghiệp — Phạm Nguyễn Trí Vinh

Chạy:  streamlit run demo/app.py
"""
import json
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

ART_DIR = Path(__file__).resolve().parent / "artifacts"

st.set_page_config(page_title="Dự báo Xu hướng Giá Chứng khoán",
                   page_icon="📈", layout="wide")

NAME_VN = {
    'VCB': 'Vietcombank', 'BID': 'BIDV', 'CTG': 'VietinBank', 'TCB': 'Techcombank',
    'MBB': 'MB Bank', 'ACB': 'ACB', 'STB': 'Sacombank', 'HDB': 'HDBank', 'VPB': 'VPBank',
    'VIC': 'Vingroup', 'VHM': 'Vinhomes', 'VRE': 'Vincom Retail', 'KDH': 'Khang Điền',
    'NLG': 'Nam Long', 'BCM': 'Becamex', 'VNM': 'Vinamilk', 'MSN': 'Masan', 'SAB': 'Sabeco',
    'MWG': 'Thế Giới Di Động', 'PNJ': 'PNJ', 'HPG': 'Hòa Phát', 'HSG': 'Hoa Sen',
    'NKG': 'Nam Kim', 'FPT': 'FPT', 'GAS': 'PV Gas', 'PLX': 'Petrolimex', 'POW': 'PV Power',
    'VJC': 'Vietjet', 'REE': 'REE', 'SSI': 'SSI', 'DHG': 'Dược Hậu Giang', 'GMD': 'Gemadept',
    'BVH': 'Bảo Việt',
}
NAME_US = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet', 'META': 'Meta', 'NVDA': 'NVIDIA',
    'AMZN': 'Amazon', 'TSLA': 'Tesla', 'NFLX': 'Netflix', 'AMD': 'AMD', 'INTC': 'Intel',
    'JPM': 'JPMorgan', 'BAC': 'Bank of America', 'WFC': 'Wells Fargo', 'GS': 'Goldman Sachs',
    'XOM': 'Exxon', 'CVX': 'Chevron', 'DIS': 'Disney', 'COST': 'Costco', 'WMT': 'Walmart',
    'UNH': 'UnitedHealth', 'LLY': 'Eli Lilly', 'V': 'Visa', 'MA': 'Mastercard',
    'BA': 'Boeing', 'VZ': 'Verizon', 'F': 'Ford', 'GM': 'General Motors',
}
NAMES = {**NAME_VN, **NAME_US}


# ── Cache loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    out = {}
    for mkt in ['VN', 'US']:
        mp = ART_DIR / f"model_{mkt}.pkl"
        jp = ART_DIR / f"metrics_{mkt}.json"
        if mp.exists() and jp.exists():
            out[mkt] = {
                'bundle': joblib.load(mp),
                'metrics': json.load(open(jp, encoding='utf-8')),
            }
    return out


@st.cache_data(show_spinner=False)
def load_walkforward_export():
    """Đọc kết quả walk-forward do notebook export (GRU cho VN, GBT cho US).
    Trả về dict {'VN': {'model':.., 'folds':[..]}, 'US': {..}} hoặc None nếu chưa có."""
    fp = ART_DIR / "walkforward.json"
    if fp.exists():
        try:
            return json.load(open(fp, encoding='utf-8'))
        except Exception:
            return None
    return None


def fetch_latest_data(ticker, market):
    """Fetch latest OHLCV data from yfinance (US) or vnstock (VN).
    Returns DataFrame with columns: time, open, high, low, close, volume, ticker
    or None if fetch fails."""
    try:
        if market == 'US':
            import yfinance as yf
            # Fetch last 30 days to ensure we get new data
            data = yf.download(ticker, period='1mo', progress=False, auto_adjust=True)
            if data is None or len(data) == 0:
                return None
            data = data.reset_index()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
            col_map = {'Date': 'time', 'Open': 'open', 'High': 'high',
                       'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
            data = data.rename(columns=col_map)
            data = data[['time', 'open', 'high', 'low', 'close', 'volume']]
            data['ticker'] = ticker
            return data
        else:  # VN
            # Dữ liệu VN dùng trực tiếp từ các file csv/<ticker>.csv.
            # Không gọi vnstock trong app để tránh rate limit và không cần tạo vietnam_stocks.csv.
            return None
    except Exception as e:
        st.warning(f"Không thể tải dữ liệu mới cho {ticker}: {str(e)}")
        return None


def merge_with_csv(ticker, market, fresh_data):
    """Merge fresh data with existing CSV data, keeping only unique dates."""
    try:
        csv_data = F.load_market_raw([ticker])
        if csv_data is None or len(csv_data) == 0:
            return fresh_data
        
        # Combine and remove duplicates (keep fresh data if same date)
        combined = pd.concat([csv_data, fresh_data], ignore_index=True)
        combined = combined.sort_values('time').reset_index(drop=True)
        combined = combined.drop_duplicates(subset=['time', 'ticker'], keep='last')
        return combined
    except Exception:
        return fresh_data


@st.cache_data(show_spinner=False)
def features_for(ticker, market):
    raw = F.load_market_raw([ticker])
    df = F.compute_features(raw, market)
    return df


def features_for_fresh(ticker, market):
    """Load features with fresh data (no cache). Used after data refresh."""
    raw = F.load_market_raw([ticker])
    
    # Try to fetch fresh data
    fresh = fetch_latest_data(ticker, market)
    if fresh is not None and len(fresh) > 0:
        raw = merge_with_csv(ticker, market, fresh)
    
    df = F.compute_features(raw, market)
    return df


def market_of(ticker):
    return 'VN' if ticker in F.VN_TICKERS else 'US'


def next_business_day(base_date=None):
    base = pd.Timestamp(base_date or pd.Timestamp.now()).normalize()
    return (base + pd.offsets.BDay(1)).date()


# ── Charts ──────────────────────────────────────────────────────────────────
def price_chart(df, ticker):
    d = df.tail(250).copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.03,
                        subplot_titles=(f"{ticker} — Giá & Bollinger Bands", "Khối lượng", "RSI(14) & MACD"))
    fig.add_trace(go.Candlestick(x=d['time'], open=d['open'], high=d['high'],
                                 low=d['low'], close=d['close'], name='Giá',
                                 increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['time'], y=d['ma20'], name='MA20', line=dict(color='#ffa726', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['time'], y=d['bb_upper'], name='BB Upper',
                             line=dict(color='rgba(120,120,200,0.4)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['time'], y=d['bb_lower'], name='BB Lower', fill='tonexty',
                             fillcolor='rgba(120,120,200,0.08)',
                             line=dict(color='rgba(120,120,200,0.4)', width=1)), row=1, col=1)
    colors = np.where(d['close'] >= d['open'], '#26a69a', '#ef5350')
    fig.add_trace(go.Bar(x=d['time'], y=d['volume'], name='Volume', marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=d['time'], y=d['rsi_14'], name='RSI(14)', line=dict(color='#ab47bc', width=1.2)), row=3, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dot', width=0.8), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dot', width=0.8), row=3, col=1)
    fig.update_layout(height=720, xaxis_rangeslider_visible=False, hovermode='x unified',
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation='h', y=1.02))
    return fig


def forecast_prices(df, horizon=5, lags=10, train_window=1000):
    """Simple autoregressive forecast using linear regression on past `lags` closes.
    Returns list of (date, predicted_close).
    """
    ser = df.sort_values('time')['close'].reset_index(drop=True)
    if len(ser) < lags + 1:
        return []
    X, y = [], []
    for i in range(lags, len(ser)):
        X.append(ser.iloc[i - lags:i].values.astype(np.float64))
        y.append(ser.iloc[i])
    X = np.vstack(X)
    y = np.asarray(y, dtype=np.float64)
    # restrict training window to recent data for stability
    if train_window is not None and len(y) > train_window:
        X = X[-train_window:]
        y = y[-train_window:]
    model = LinearRegression()
    try:
        model.fit(X, y)
    except Exception:
        return []

    last_window = ser.iloc[-lags:].values.astype(np.float64).tolist()
    preds = []
    cur = list(last_window)
    last_date = pd.Timestamp(df['time'].iloc[-1]).normalize()
    for i in range(1, horizon + 1):
        xi = np.array(cur[-lags:]).reshape(1, -1)
        p = float(model.predict(xi)[0])
        preds.append((pd.Timestamp(last_date + pd.offsets.BDay(i)).date(), p))
        cur.append(p)
    return preds


def knn_trend_probability(df, feat_cols, scaler, row, horizon=5, k=200, market='US'):
    """Estimate probability of upward trend over `horizon` business days by K-NN on past feature vectors.
    Uses `scaler` to normalize features. Returns probability (0..1) and neighbor sample counts.
    """
    dff = df.sort_values('time').copy()
    g = dff.groupby('ticker', group_keys=False)
    # compute horizon future close and return per ticker
    dff[f'future_close_{horizon}'] = g['close'].shift(-horizon)
    dff[f'future_return_{horizon}'] = (dff[f'future_close_{horizon}'] - dff['close']) / dff['close']

    hist = dff.dropna(subset=feat_cols + [f'future_return_{horizon}']).copy()
    if hist.empty:
        return None, 0
    X = scaler.transform(hist[feat_cols].values.astype(np.float64))
    xr = scaler.transform(row[feat_cols].values.reshape(1, -1))
    # distances and neighbors
    dist = np.linalg.norm(X - xr, axis=1)
    idx = np.argsort(dist)[:min(k, len(dist))]
    thr = 0.01 if market == 'US' else 0.02
    neigh_returns = hist.iloc[idx][f'future_return_{horizon}'].values
    prob = float(np.mean(neigh_returns > thr))
    return prob, len(idx)


# ── Tabs ────────────────────────────────────────────────────────────────────
def tab_charts(art):
    st.subheader("📊 Dữ liệu thị trường & Chỉ báo kỹ thuật")
    c1, c2 = st.columns([1, 3])
    with c1:
        mkt = st.radio("Thị trường", ['VN', 'US'], horizontal=True, key="chart_mkt")
        tickers = F.VN_TICKERS if mkt == 'VN' else F.US_TICKERS
        ticker = st.selectbox("Chọn mã cổ phiếu",
                              sorted(tickers),
                              format_func=lambda t: f"{t} — {NAMES.get(t, t)}", key="chart_tk")
    df = features_for(ticker, mkt)
    last = df.iloc[-1]
    with c2:
        m1, m2, m3, m4 = st.columns(4)
        chg = last['daily_return'] * 100 if pd.notna(last['daily_return']) else 0
        m1.metric("Giá đóng cửa", f"{last['close']:,.2f}", f"{chg:+.2f}%")
        m2.metric("RSI(14)", f"{last['rsi_14']:.1f}",
                  "Quá mua" if last['rsi_14'] > 70 else ("Quá bán" if last['rsi_14'] < 30 else "Trung tính"))
        m3.metric("Khối lượng", f"{last['volume']:,.0f}")
        m4.metric("Số phiên", f"{len(df):,d}")
    st.plotly_chart(price_chart(df, ticker), use_container_width=True)
    with st.expander("Xem 34+ chỉ báo kỹ thuật (features) của phiên gần nhất"):
        feat_show = ['daily_return', 'rsi_7', 'rsi_14', 'rsi_21', 'macd_real', 'macd_hist_real',
                     'stoch_k', 'williams_r', 'cci14', 'adx14', 'atr_ratio', 'bb_pct_b',
                     'bb_bandwidth', 'momentum_5', 'momentum_10', 'volume_ma_ratio', 'obv_signal',
                     'proximity_52w_high', 'price_vs_ma20', 'price_vs_ma50']
        show = {k: round(float(last[k]), 4) for k in feat_show if k in df.columns and pd.notna(last[k])}
        st.dataframe(pd.DataFrame([show]).T.rename(columns={0: 'Giá trị'}), use_container_width=True)


@st.cache_resource(show_spinner="Đang nạp model GRU (TensorFlow)...")
def load_gru_bundle(market):
    """Nạp model GRU do notebook export (chỉ VN). Lazy-import TensorFlow để app
    không phụ thuộc TF khi không dùng GRU. Trả về dict hoặc None."""
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    meta_p = ART_DIR / f"gru_{market}_meta.json"
    model_p = ART_DIR / f"gru_{market}.keras"
    scaler_p = ART_DIR / f"gru_{market}_scaler.pkl"
    if not (meta_p.exists() and model_p.exists() and scaler_p.exists()):
        return None
    try:
        import joblib
        from tensorflow.keras.models import load_model
        meta = json.load(open(meta_p, encoding='utf-8'))
        return {'model': load_model(model_p), 'scaler': joblib.load(scaler_p),
                'features': meta['features'], 'seq_len': int(meta['seq_len']),
                'accuracy': float(meta.get('accuracy', 0.0)),
                'auc': float(meta.get('auc', 0.0)),
                'ticker_acc': meta.get('ticker_acc', {})}
    except Exception as e:
        print(f"[GRU] load failed: {e}")
        return None


@st.cache_data(show_spinner=False)
def gru_prob_cached(ticker, market):
    """Xác suất TĂNG theo GRU cho từng phiên (chuỗi 20 ngày kết thúc tại phiên đó).
    Trả về DataFrame[time, prob_up] hoặc None."""
    gru = load_gru_bundle(market)
    if gru is None:
        return None
    df = features_for(ticker, market)
    feat_list, scaler = gru['features'], gru['scaler']
    seq_len, model = gru['seq_len'], gru['model']
    if any(c not in df.columns for c in feat_list):
        return None
    dfc = df.dropna(subset=feat_list).reset_index(drop=True)
    if len(dfc) <= seq_len:
        return None
    X_all = scaler.transform(dfc[feat_list].values.astype(np.float64))
    X_seq = np.array([X_all[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(dfc))],
                     dtype=np.float32)
    prob = model.predict(X_seq, verbose=0).flatten()
    out = dfc.iloc[seq_len - 1:][['time']].copy()
    out['prob_up'] = prob
    return out


def _predict_for_tab(df, ticker, mkt, art):
    """Trả về (df_có_prob_up, model_label, ticker_acc). Ưu tiên GRU (nếu có artifact),
    fallback XGBoost của app."""
    if load_gru_bundle(mkt) is not None:
        out = gru_prob_cached(ticker, mkt)
        if out is not None and len(out):
            gru = load_gru_bundle(mkt)
            d = df.merge(out, on='time', how='left')
            ta = gru['ticker_acc'].get(ticker, gru['accuracy'])
            return d, f"GRU (acc={gru['accuracy']:.3f})", float(ta)
    bundle = art[mkt]['bundle']
    feat_x = bundle['features']
    d = df.copy()
    d['prob_up'] = np.nan
    mask = d[feat_x].notna().all(axis=1)
    if mask.any():
        Xv = bundle['scaler'].transform(d.loc[mask, feat_x].values.astype(np.float64))
        d.loc[mask, 'prob_up'] = bundle['model'].predict_proba(Xv)[:, 1]
    ta = art[mkt]['metrics']['ticker_acc'].get(ticker, 0.0)
    return d, f"XGBoost (acc={art[mkt]['metrics'].get('accuracy', 0):.3f})", float(ta)


def tab_predict(art):
    st.subheader("🔮 Dự báo xu hướng giá phiên kế tiếp")
    st.caption("Dự báo xác suất giá TĂNG vượt ngưỡng vào phiên giao dịch tiếp theo "
               "(ngưỡng: VN ±2%, US ±1%). VN dùng **GRU**, US dùng **XGBoost** "
               "(tùy artifact đã export từ notebook).")
    c1, c2 = st.columns([1, 3])
    with c1:
        mkt = st.radio("Thị trường", ['VN', 'US'], horizontal=True, key="pred_mkt")
        tickers = F.VN_TICKERS if mkt == 'VN' else F.US_TICKERS
        ticker = st.selectbox("Chọn mã cổ phiếu", sorted(tickers),
                              format_func=lambda t: f"{t} — {NAMES.get(t, t)}", key="pred_tk")
        
        # Refresh button for real-time data
        refresh_button = st.button("🔄 Cập nhật dữ liệu mới nhất", key="refresh_data")
    
    if mkt not in art:
        st.error(f"Chưa có model cho thị trường {mkt}. Chạy: python demo/train_models.py")
        return
    bundle = art[mkt]['bundle']
    feat = bundle['features']
    
    # US có thể tải dữ liệu mới; VN luôn đọc trực tiếp từ csv/<ticker>.csv
    if refresh_button and mkt == 'US':
        df = features_for_fresh(ticker, mkt)
    else:
        df = features_for(ticker, mkt)
    df_p, model_label, ta_val = _predict_for_tab(df, ticker, mkt, art)
    valid = df_p.dropna(subset=['prob_up'])
    if valid.empty:
        st.warning("Không đủ dữ liệu để tính features cho mã này.")
        return
    row = valid.iloc[-1]
    forecast_date = next_business_day()
    prob_up = float(row['prob_up'])
    pred_up = prob_up >= 0.5

    with c2:
        g1, g2 = st.columns([1.2, 1])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob_up * 100,
                title={'text': f"Xác suất TĂNG — phiên kế tiếp {forecast_date}"},
                number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': '#26a69a' if pred_up else '#ef5350'},
                       'steps': [{'range': [0, 50], 'color': '#ffebee'},
                                 {'range': [50, 100], 'color': '#e8f5e9'}],
                       'threshold': {'line': {'color': 'black', 'width': 3}, 'value': 50}}))
            gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)
        with g2:
            st.markdown("###")
            if pred_up:
                st.success(f"### 📈 DỰ BÁO: TĂNG\nXác suất {prob_up*100:.1f}%")
            else:
                st.error(f"### 📉 DỰ BÁO: GIẢM/ĐI NGANG\nXác suất tăng {prob_up*100:.1f}%")
            st.caption(
                f"Mô hình: **{model_label}** · Phiên gần nhất: {row['time'].date()} · "
                f"Dự báo cho phiên kế tiếp: {forecast_date}"
            )
            st.metric(f"Độ chính xác {model_label.split(' ')[0]} trên mã này",
                      f"{ta_val*100:.1f}%")

    st.divider()
    st.markdown(f"**Backtest gần đây** — dự báo (**{model_label.split(' ')[0]}**) "
                "từ ngày hiện tại về trước (40 phiên gần nhất):")
    st.caption("⏳ = Chưa có kết quả thực tế (dự báo cho tương lai) · ✅/❌ = So khớp với kết quả đã biết")
    
    # Lấy tất cả phiên có prob_up (bao gồm cả phiên chưa có future_return)
    hist = df_p.dropna(subset=['prob_up']).tail(40).copy()
    if not hist.empty:
        hist['pred'] = np.where(hist['prob_up'] >= 0.5, 'TĂNG', 'GIẢM')
        thr = 0.02 if mkt == 'VN' else 0.01
        
        # Xử lý phiên có future_return (đã biết kết quả)
        has_result = hist['future_return'].notna()
        hist['thực tế'] = '⏳ Chờ KQ'
        hist.loc[has_result, 'thực tế'] = np.where(
            hist.loc[has_result, 'future_return'] > thr, 'TĂNG',
            np.where(hist.loc[has_result, 'future_return'] < -thr, 'GIẢM', 'đi ngang'))
        
        hist['đúng?'] = '⏳'
        hist.loc[has_result, 'đúng?'] = np.where(
            ((hist.loc[has_result, 'prob_up'] >= 0.5) & (hist.loc[has_result, 'future_return'] > thr)) |
            ((hist.loc[has_result, 'prob_up'] < 0.5) & (hist.loc[has_result, 'future_return'] <= thr)), '✅', '❌')
        
        tbl = hist[['time', 'close', 'prob_up', 'pred', 'future_return', 'thực tế', 'đúng?']].copy()
        tbl['time'] = tbl['time'].dt.date
        tbl['prob_up'] = (tbl['prob_up'] * 100).round(1).astype(str) + '%'
        tbl['future_return'] = tbl['future_return'].apply(
            lambda x: '—' if pd.isna(x) else f"{x*100:.2f}%")
        tbl.columns = ['Ngày', 'Giá đóng', 'P(tăng)', 'Dự báo', 'Lợi suất sau', 'Thực tế', 'Đúng?']
        
        # Sắp xếp từ mới → cũ (ngày hiện tại trên cùng)
        st.dataframe(tbl.iloc[::-1], use_container_width=True, hide_index=True, height=320)

    st.divider()
    st.subheader("🛠️ Dự báo giá đóng cửa tương lai (giá trị)")
    c3, c4 = st.columns([1, 3])
    with c3:
        horizon = st.slider("Số phiên dự báo", 1, 30, 5, key='forecast_horizon')
        method = st.selectbox("Phương pháp", ['AR Linear', 'Naive (last close)'], key='forecast_method')
        do_forecast = st.button("Tạo dự báo giá")
    with c4:
        st.caption("Mô hình AR tuyến tính dùng các giá đóng cửa trước đó làm đặc trưng; đơn giản và chạy nhanh.")

    if do_forecast:
        ser_df = df[['time', 'close']].dropna().sort_values('time')
        if method == 'Naive (last close)':
            last = float(ser_df['close'].iloc[-1])
            preds = []
            last_date = pd.Timestamp(ser_df['time'].iloc[-1]).normalize()
            for i in range(1, horizon + 1):
                preds.append((pd.Timestamp(last_date + pd.offsets.BDay(i)).date(), last))
        else:
            preds = forecast_prices(ser_df, horizon=horizon, lags=10)

        if not preds:
            st.warning("Không thể tạo dự báo — dữ liệu không đủ hoặc mô hình lỗi.")
        else:
            pred_df = pd.DataFrame(preds, columns=['date', 'pred_close'])
            pred_df['pred_close'] = pred_df['pred_close'].round(2)
            last_hist = ser_df.tail(120).copy()
            last_hist['date'] = last_hist['time'].dt.date

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=last_hist['date'], y=last_hist['close'], mode='lines', name='Giá lịch sử'))
            fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['pred_close'], mode='lines+markers', name='Dự báo', marker=dict(symbol='circle', size=8)))
            fig.update_layout(title=f'Dự báo {horizon} phiên — phương pháp: {method}', height=420)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Bảng dự báo**")
            st.dataframe(pred_df.rename(columns={'date': 'Ngày', 'pred_close': 'Giá dự báo'}), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔮 Dự báo XU HƯỚNG trong N phiên (xác suất)")
    c5, c6 = st.columns([1, 3])
    with c5:
        horizon_trend = st.slider("Số phiên để đánh giá xu hướng (horizon)", 1, 30, 5, key='trend_horizon')
        knn_k = st.number_input("Số neighbors (K)", min_value=10, max_value=2000, value=200, step=10, key='knn_k')
        do_trend = st.button("Dự báo xu hướng")
    with c6:
        st.caption("Ước lượng xác suất 'TĂNG' trong H phiên tiếp theo dựa trên các phiên lịch sử có features tương tự.")

    if do_trend:
        prob_trend, n_used = knn_trend_probability(df, feat, bundle['scaler'], row, horizon=horizon_trend, k=int(knn_k), market=mkt)
        if prob_trend is None:
            st.warning("Không đủ dữ liệu lịch sử để ước lượng xu hướng cho horizon này.")
        else:
            pred_tr = prob_trend >= 0.5
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob_trend * 100,
                title={'text': f"Xác suất TĂNG — trong {horizon_trend} phiên"},
                number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': '#26a69a' if pred_tr else '#ef5350'},
                       'steps': [{'range': [0, 50], 'color': '#ffebee'},
                                 {'range': [50, 100], 'color': '#e8f5e9'}],
                       'threshold': {'line': {'color': 'black', 'width': 3}, 'value': 50}}))
            g.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(g, use_container_width=True)
            if pred_tr:
                st.success(f"📈 DỰ BÁO XU HƯỚNG: TĂNG · Xác suất {prob_trend*100:.1f}%")
            else:
                st.error(f"📉 DỰ BÁO XU HƯỚNG: KHÔNG TĂNG/ GIẢM · Xác suất tăng {prob_trend*100:.1f}%")
            st.metric("Số phiên lịch sử dùng (neighbors)", f"{n_used}")


def _walkforward_source(art, mkt):
    """Ưu tiên kết quả walk-forward do notebook export (GRU/GBT); fallback XGBoost của app.
    Trả về (folds, model_name, from_notebook)."""
    export = load_walkforward_export()
    if export and mkt in export and export[mkt].get('folds'):
        return export[mkt]['folds'], export[mkt].get('model', '?'), True
    wf = (art.get(mkt, {}).get('metrics', {}) or {}).get('walk_forward') or []
    return wf, 'XGBoost', False


def tab_walkforward(art):
    st.subheader("📅 Walk-Forward Validation — Độ ổn định qua từng năm")
    st.caption(
        "Mỗi năm test, mô hình được **train lại trên toàn bộ quá khứ** (expanding window) rồi dự báo "
        "năm đó — đúng như giao dịch thực tế: chỉ dùng dữ liệu đã biết để dự đoán tương lai. "
        "Đây là cách kiểm định trung thực nhất, không rò rỉ dữ liệu (no lookahead).")

    from_nb = load_walkforward_export() is not None
    if from_nb:
        st.success(
            "✅ Kết quả lấy **trực tiếp từ notebook** — đúng mô hình tốt nhất mỗi thị trường "
            "(**GRU** cho VN, **GBT** cho US), đồng bộ với báo cáo.")
    else:
        st.info(
            "ℹ️ Đang dùng **XGBoost** (mô hình nhẹ của app) vì chưa có kết quả từ notebook. "
            "Chạy cell walk-forward trong notebook để sinh `demo/artifacts/walkforward.json` "
            "→ app sẽ tự hiển thị kết quả GRU/GBT.")

    have_any = any(_walkforward_source(art, m)[0] for m in ['VN', 'US'])
    if not have_any:
        st.warning("Chưa có dữ liệu walk-forward. Hãy chạy `python demo/train_models.py` "
                   "(XGBoost) hoặc cell walk-forward trong notebook (GRU/GBT).")
        return

    cols = st.columns(2)
    for i, mkt in enumerate(['VN', 'US']):
        if mkt not in art:
            continue
        wf, model_name, _src_nb = _walkforward_source(art, mkt)
        with cols[i]:
            flag = '🇻🇳 Việt Nam' if mkt == 'VN' else '🇺🇸 Hoa Kỳ'
            st.markdown(f"### {flag} · `{model_name}`")
            if not wf:
                st.caption("Không có dữ liệu.")
                continue
            years = [r['test_year'] for r in wf]
            accs  = [r['accuracy'] for r in wf]
            aucs  = [r['auc'] for r in wf]
            mean_acc, mean_auc = float(np.mean(accs)), float(np.mean(aucs))
            std_acc = float(np.std(accs))
            clr = '#16A34A' if mkt == 'VN' else '#2563EB'

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=accs, mode='lines+markers', name='Accuracy',
                                     line=dict(color=clr, width=3), marker=dict(size=10)))
            fig.add_trace(go.Scatter(x=years, y=aucs, mode='lines+markers', name='AUC-ROC',
                                     line=dict(color=clr, width=1.5, dash='dash'), marker=dict(size=7),
                                     opacity=0.7))
            fig.add_hline(y=0.5, line=dict(color='gray', dash='dot'),
                          annotation_text='Ngẫu nhiên (0.50)', annotation_position='bottom right')
            fig.add_hline(y=mean_acc, line=dict(color=clr, dash='dash', width=1),
                          annotation_text=f'TB acc={mean_acc:.3f}', annotation_position='top left')
            fig.update_layout(title=f"{mkt} — Accuracy & AUC theo năm test",
                              xaxis_title="Năm kiểm tra", yaxis_title="Score",
                              yaxis_range=[0.42, 0.74], height=360,
                              margin=dict(l=10, r=10, t=40, b=10),
                              legend=dict(orientation='h', y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            a, b, c = st.columns(3)
            a.metric("Acc trung bình", f"{mean_acc*100:.1f}%")
            b.metric("AUC trung bình", f"{mean_auc:.3f}")
            c.metric("Độ lệch chuẩn Acc", f"±{std_acc*100:.1f}%")

            tbl = pd.DataFrame(wf)[['test_year', 'train_end', 'accuracy', 'auc', 'n_test']].copy()
            tbl['accuracy'] = (tbl['accuracy'] * 100).round(1).astype(str) + '%'
            tbl['auc'] = tbl['auc'].round(3)
            tbl['train_end'] = '2013–' + tbl['train_end'].astype(str)
            tbl.columns = ['Năm test', 'Cửa sổ train', 'Accuracy', 'AUC', 'N_test']
            st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        "**Đọc kết quả:** đường Accuracy/AUC nằm **trên mốc 0.50** qua các năm cho thấy mô hình nắm "
        "được tín hiệu xu hướng *ổn định*, không phải ăn may một giai đoạn. Độ lệch chuẩn nhỏ = ổn định cao. "
        "Với dữ liệu tài chính (rất nhiễu), AUC ~0.55–0.60 đã là tín hiệu có giá trị.")


def tab_performance(art):
    st.subheader("🎯 Hiệu năng mô hình & So sánh thị trường")
    cols = st.columns(2)
    for i, mkt in enumerate(['VN', 'US']):
        if mkt not in art:
            continue
        m = art[mkt]['metrics']
        with cols[i]:
            st.markdown(f"### Thị trường {'🇻🇳 Việt Nam' if mkt=='VN' else '🇺🇸 Hoa Kỳ'}")
            a, b, c = st.columns(3)
            a.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
            b.metric("AUC-ROC", f"{m['auc']:.3f}")
            c.metric("Edge vs Naive", f"{m['edge']*100:+.1f}%")
            st.caption(f"Train {m['n_train']:,d} mẫu · Test {m['n_test']:,d} mẫu · {m['n_features']} features")

    st.divider()
    st.markdown("#### 💡 Phát hiện chính: Thị trường VN dự báo tốt hơn US")
    if 'VN' in art and 'US' in art:
        ev = art['VN']['metrics']['edge'] - art['US']['metrics']['edge']
        st.info(
            f"**Edge dự báo của VN cao hơn US ({art['VN']['metrics']['edge']*100:+.1f}% so với "
            f"{art['US']['metrics']['edge']*100:+.1f}%).** Kết quả này phù hợp với **Giả thuyết Thị trường "
            f"Hiệu quả (EMH — Fama, 1970)**: thị trường Mỹ hiệu quả cao nên giá gần như ngẫu nhiên, khó dự báo; "
            f"thị trường VN kém hiệu quả hơn nên còn tồn tại tín hiệu khai thác được.")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix & ROC", "Độ chính xác theo mã", "Tầm quan trọng features"])
    with tab1:
        cc = st.columns(2)
        for i, mkt in enumerate(['VN', 'US']):
            if mkt not in art:
                continue
            m = art[mkt]['metrics']
            with cc[i]:
                cm = np.array(m['confusion_matrix'])
                heat = go.Figure(go.Heatmap(
                    z=cm, x=['Dự báo GIẢM', 'Dự báo TĂNG'], y=['Thực GIẢM', 'Thực TĂNG'],
                    text=cm, texttemplate="%{text}", colorscale='Blues', showscale=False))
                heat.update_layout(title=f"Confusion Matrix — {mkt}", height=300,
                                   margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(heat, use_container_width=True)
        roc = go.Figure()
        for mkt in ['VN', 'US']:
            if mkt not in art:
                continue
            r = art[mkt]['metrics']['roc']
            roc.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], mode='lines',
                                     name=f"{mkt} (AUC={art[mkt]['metrics']['auc']:.3f})"))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Ngẫu nhiên',
                                 line=dict(dash='dash', color='gray')))
        roc.update_layout(title="Đường cong ROC", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate", height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(roc, use_container_width=True)
    with tab2:
        for mkt in ['VN', 'US']:
            if mkt not in art:
                continue
            ta = art[mkt]['metrics']['ticker_acc']
            s = pd.Series(ta).sort_values(ascending=True)
            bar = go.Figure(go.Bar(x=s.values * 100, y=[f"{t} ({NAMES.get(t,t)})" for t in s.index],
                                   orientation='h', marker_color='#42a5f5'))
            bar.add_vline(x=50, line=dict(color='red', dash='dot'))
            bar.update_layout(title=f"Độ chính xác theo mã — {mkt}", xaxis_title="Accuracy (%)",
                              height=max(300, len(s) * 20), margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(bar, use_container_width=True)
    with tab3:
        cc = st.columns(2)
        for i, mkt in enumerate(['VN', 'US']):
            if mkt not in art:
                continue
            fi = art[mkt]['metrics']['feature_importance']
            s = pd.Series(fi).sort_values(ascending=True)
            with cc[i]:
                bar = go.Figure(go.Bar(x=list(s.values), y=list(s.index), orientation='h',
                                       marker_color='#66bb6a'))
                bar.update_layout(title=f"Top features — {mkt}", height=420,
                                  margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(bar, use_container_width=True)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    st.title("📈 Phân tích & Dự báo Xu hướng Giá Chứng khoán")
    st.caption("Đồ án tốt nghiệp · PySpark + Machine Learning · 60 mã cổ phiếu US & VN · Dữ liệu 2013–2026")
    art = load_artifacts()
    if not art:
        st.error("Chưa có model. Hãy chạy trước: `python demo/train_models.py`")
        st.stop()
    with st.sidebar:
        st.header("Giới thiệu")
        st.markdown(
            "- **Bài toán:** dự báo xu hướng giá (tăng/giảm) phiên kế tiếp.\n"
            "- **Dữ liệu:** OHLCV 2013–2026 + macro (VN-Index, S&P500, VIX).\n"
            "- **Features:** 34+ chỉ báo kỹ thuật (RSI, MACD, Bollinger, ADX...).\n"
            "- **Mô hình:** VN dùng GRU, US dùng XGBoost/GBT (notebook gồm cả Spark ML & LSTM/GRU).\n"
            "- **Tách dữ liệu:** Train ≤2021 · Test ≥2022 (out-of-time)." )
        st.divider()
        for mkt in ['VN', 'US']:
            if mkt in art:
                m = art[mkt]['metrics']
                st.metric(f"{mkt} · Accuracy", f"{m['accuracy']*100:.1f}%", f"AUC {m['auc']:.3f}")
    t1, t2, t3, t4 = st.tabs(["📊 Dữ liệu & Biểu đồ", "🔮 Dự báo",
                              "📅 Walk-Forward", "🎯 Hiệu năng mô hình"])
    with t1:
        tab_charts(art)
    with t2:
        tab_predict(art)
    with t3:
        tab_walkforward(art)
    with t4:
        tab_performance(art)


if __name__ == '__main__':
    main()

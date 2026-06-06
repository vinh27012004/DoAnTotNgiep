"""
Demo Streamlit — Phân tích & Dự báo Xu hướng Giá Chứng khoán (PySpark + ML)
Đồ án tốt nghiệp — Phạm Nguyễn Trí Vinh

Chạy:  streamlit run demo/app.py
"""
import json
import sys
from pathlib import Path

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
def features_for(ticker, market):
    raw = F.load_market_raw([ticker])
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


def tab_predict(art):
    st.subheader("🔮 Dự báo xu hướng giá phiên kế tiếp")
    st.caption("Mô hình XGBoost dự báo xác suất giá TĂNG vượt ngưỡng vào phiên giao dịch tiếp theo "
               "(ngưỡng: VN ±2%, US ±1%).")
    c1, c2 = st.columns([1, 3])
    with c1:
        mkt = st.radio("Thị trường", ['VN', 'US'], horizontal=True, key="pred_mkt")
        tickers = F.VN_TICKERS if mkt == 'VN' else F.US_TICKERS
        ticker = st.selectbox("Chọn mã cổ phiếu", sorted(tickers),
                              format_func=lambda t: f"{t} — {NAMES.get(t, t)}", key="pred_tk")
    if mkt not in art:
        st.error(f"Chưa có model cho thị trường {mkt}. Chạy: python demo/train_models.py")
        return
    bundle = art[mkt]['bundle']
    feat = bundle['features']
    df = features_for(ticker, mkt)
    valid = df.dropna(subset=feat)
    if valid.empty:
        st.warning("Không đủ dữ liệu để tính features cho mã này.")
        return
    row = valid.iloc[-1]
    forecast_date = next_business_day()
    X = bundle['scaler'].transform(row[feat].values.astype(np.float64).reshape(1, -1))
    prob_up = float(bundle['model'].predict_proba(X)[0, 1])
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
                f"Phiên gần nhất trong dữ liệu: {row['time'].date()} · Dự báo hiển thị cho phiên kế tiếp của hôm nay: {forecast_date}"
            )
            st.metric("Độ chính xác mô hình trên mã này (test)",
                      f"{art[mkt]['metrics']['ticker_acc'].get(ticker, 0)*100:.1f}%")

    st.divider()
    st.markdown("**Backtest gần đây** — so khớp dự báo với kết quả thực tế (30 phiên có nhãn gần nhất):")
    hist = df.dropna(subset=feat + ['future_return']).tail(30).copy()
    if not hist.empty:
        Xh = bundle['scaler'].transform(hist[feat].values.astype(np.float64))
        hist['prob_up'] = bundle['model'].predict_proba(Xh)[:, 1]
        hist['pred'] = np.where(hist['prob_up'] >= 0.5, 'TĂNG', 'GIẢM')
        thr = 0.02 if mkt == 'VN' else 0.01
        hist['thực tế'] = np.where(hist['future_return'] > thr, 'TĂNG',
                                   np.where(hist['future_return'] < -thr, 'GIẢM', 'đi ngang'))
        hist['đúng?'] = np.where(
            ((hist['prob_up'] >= 0.5) & (hist['future_return'] > thr)) |
            ((hist['prob_up'] < 0.5) & (hist['future_return'] <= thr)), '✅', '❌')
        tbl = hist[['time', 'close', 'prob_up', 'pred', 'future_return', 'thực tế', 'đúng?']].copy()
        tbl['time'] = tbl['time'].dt.date
        tbl['prob_up'] = (tbl['prob_up'] * 100).round(1).astype(str) + '%'
        tbl['future_return'] = (tbl['future_return'] * 100).round(2).astype(str) + '%'
        tbl.columns = ['Ngày', 'Giá đóng', 'P(tăng)', 'Dự báo', 'Lợi suất sau', 'Thực tế', 'Đúng?']
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
            "- **Mô hình:** XGBoost (pipeline trong notebook gồm cả Spark ML & LSTM/GRU).\n"
            "- **Tách dữ liệu:** Train ≤2021 · Test ≥2022 (out-of-time)." )
        st.divider()
        for mkt in ['VN', 'US']:
            if mkt in art:
                m = art[mkt]['metrics']
                st.metric(f"{mkt} · Accuracy", f"{m['accuracy']*100:.1f}%", f"AUC {m['auc']:.3f}")
    t1, t2, t3 = st.tabs(["📊 Dữ liệu & Biểu đồ", "🔮 Dự báo", "🎯 Hiệu năng mô hình"])
    with t1:
        tab_charts(art)
    with t2:
        tab_predict(art)
    with t3:
        tab_performance(art)


if __name__ == '__main__':
    main()

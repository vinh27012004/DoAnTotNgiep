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
import plotly.io as pio
from plotly.subplots import make_subplots

# ── Bảng màu "Finance terminal" ─────────────────────────────────────────────
CLR = {
    "bg": "#0E1117", "panel": "#161B26", "border": "#232B38",
    "text": "#E6EDF3", "muted": "#8B97A7",
    "teal": "#26C6A6", "blue": "#5B9CF6", "red": "#EF5350",
    "amber": "#FFA726", "purple": "#AB7DF6", "green": "#26A69A",
}

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


# ── Theme: CSS + Plotly dark template ───────────────────────────────────────
_CSS = """
<style>
/* Ẩn chrome mặc định của Streamlit */
#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stHeader"], footer {visibility:hidden; height:0;}
.block-container {padding-top:2.2rem; padding-bottom:2rem; max-width:1400px;}

/* Nền */
.stApp {background:#0E1117;}

/* Hero header */
.hero {
  background:linear-gradient(135deg,#10202A 0%,#0E1117 60%);
  border:1px solid #232B38; border-left:4px solid #26C6A6;
  border-radius:14px; padding:18px 22px; margin-bottom:14px;
}
.hero-row {display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;}
.hero-title {font-size:1.55rem; font-weight:800; letter-spacing:.3px; color:#E6EDF3; margin:0;}
.hero-title .tk {color:#26C6A6;}
.hero-sub {color:#8B97A7; font-size:.9rem; margin-top:4px;}
.badge {display:inline-flex; flex-direction:column; align-items:center;
  background:#161B26; border:1px solid #232B38; border-radius:10px;
  padding:8px 16px; min-width:96px;}
.badge .b-lab {font-size:.7rem; color:#8B97A7; text-transform:uppercase; letter-spacing:.5px;}
.badge .b-val {font-size:1.2rem; font-weight:800; color:#26C6A6;}
.badge.us .b-val {color:#5B9CF6;}
.live {display:inline-flex; align-items:center; gap:6px; color:#26C6A6; font-size:.8rem; font-weight:700;}
.live .dot {width:8px; height:8px; border-radius:50%; background:#26C6A6; box-shadow:0 0 8px #26C6A6;}

/* st.metric -> card */
[data-testid="stMetric"] {
  background:#161B26; border:1px solid #232B38; border-left:3px solid #26C6A6;
  border-radius:10px; padding:12px 16px;
}
[data-testid="stMetricLabel"] p {color:#8B97A7 !important; font-size:.78rem;}
[data-testid="stMetricValue"] {color:#E6EDF3 !important; font-weight:800;}

/* Tabs bo tròn */
.stTabs [data-baseweb="tab-list"] {gap:6px; border-bottom:1px solid #232B38;}
.stTabs [data-baseweb="tab"] {
  background:#161B26; border:1px solid #232B38; border-bottom:none;
  border-radius:9px 9px 0 0; padding:9px 18px; color:#8B97A7;
}
.stTabs [aria-selected="true"] {background:#26C6A6 !important; color:#0E1117 !important; font-weight:700;}

/* Nút */
.stButton>button {
  background:#161B26; border:1px solid #2A3445; color:#E6EDF3; border-radius:9px;
}
.stButton>button:hover {border-color:#26C6A6; color:#26C6A6;}

/* Divider mảnh hơn, expander/sidebar */
hr {border-color:#232B38 !important;}
[data-testid="stSidebar"] {border-right:1px solid #232B38;}
[data-testid="stExpander"] {border:1px solid #232B38; border-radius:10px;}
</style>
"""


def inject_theme():
    """Inject CSS + đặt template Plotly tối để mọi biểu đồ đồng bộ với nền."""
    st.markdown(_CSS, unsafe_allow_html=True)
    if "finance_dark" not in pio.templates:
        pio.templates["finance_dark"] = go.layout.Template(layout=dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=CLR["text"], size=12),
            xaxis=dict(gridcolor=CLR["border"], zerolinecolor=CLR["border"], linecolor=CLR["border"]),
            yaxis=dict(gridcolor=CLR["border"], zerolinecolor=CLR["border"], linecolor=CLR["border"]),
            colorway=[CLR["teal"], CLR["blue"], CLR["amber"], CLR["purple"], CLR["red"], CLR["green"]],
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        ))
    pio.templates.default = "finance_dark"


def hero(art):
    """Header dạng terminal: tiêu đề + mô tả + badge accuracy VN/US + chỉ báo live."""
    badges = []
    for mkt, cls in [("VN", ""), ("US", "us")]:
        if mkt in art:
            acc = art[mkt]["metrics"]["accuracy"] * 100
            badges.append(
                f'<div class="badge {cls}"><span class="b-lab">{mkt} ACC</span>'
                f'<span class="b-val">{acc:.1f}%</span></div>')
    badges.append('<span class="live"><span class="dot"></span>live</span>')
    st.markdown(
        '<div class="hero"><div class="hero-row"><div>'
        '<div class="hero-title">▲ DỰ BÁO XU HƯỚNG GIÁ <span class="tk">CHỨNG KHOÁN</span></div>'
        '<div class="hero-sub">PySpark + Machine Learning · 60 mã US &amp; VN · Dữ liệu 2013–2026</div>'
        '</div><div class="hero-row" style="gap:10px;">' + "".join(badges) + '</div></div></div>',
        unsafe_allow_html=True)


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


def forecast_chart(df_p, ticker, mkt, prob_up, pred_up, forecast_date, n=45):
    """Biểu đồ giá n phiên gần nhất + điểm dự báo phiên kế tiếp (mũi tên TĂNG/GIẢM),
    đường chiếu nét đứt và dải ngưỡng ±2%/±1% (vùng 'đi ngang'). Hướng dựa theo dự báo;
    biên độ chiếu = đúng ngưỡng quyết định (không bịa độ lớn)."""
    d = df_p.dropna(subset=['close']).tail(n).copy()
    thr = 0.02 if mkt == 'VN' else 0.01
    last_t = d['time'].iloc[-1]
    last_c = float(d['close'].iloc[-1])
    fdate = pd.Timestamp(forecast_date)
    x_end = fdate + pd.Timedelta(days=max(4, n // 8))     # chừa khoảng trống bên phải
    col = '#26a69a' if pred_up else '#ef5350'
    proj = last_c * (1 + thr) if pred_up else last_c * (1 - thr)

    fig = go.Figure()
    # Dải ngưỡng quanh giá hiện tại (vùng coi như đi ngang) cho phiên kế tiếp
    fig.add_trace(go.Scatter(
        x=[last_t, x_end, x_end, last_t],
        y=[last_c * (1 - thr), last_c * (1 - thr), last_c * (1 + thr), last_c * (1 + thr)],
        fill='toself', fillcolor='rgba(120,120,200,0.10)', line=dict(width=0),
        hoverinfo='skip', showlegend=False))
    # Đường giá lịch sử
    fig.add_trace(go.Scatter(x=d['time'], y=d['close'], mode='lines', name='Giá đóng',
                             line=dict(color='#5B9CF6', width=2),
                             hovertemplate='%{x|%d/%m/%Y}: %{y:,.2f}<extra></extra>'))
    # Điểm hiện tại
    fig.add_trace(go.Scatter(x=[last_t], y=[last_c], mode='markers', showlegend=False,
                             marker=dict(color='#E6EDF3', size=9, line=dict(color='#0E1117', width=1)),
                             hovertemplate='Hiện tại: %{y:,.2f}<extra></extra>'))
    # Đường chiếu dự báo (nét đứt)
    fig.add_trace(go.Scatter(x=[last_t, fdate], y=[last_c, proj], mode='lines', showlegend=False,
                             line=dict(color=col, width=2, dash='dot'), hoverinfo='skip'))
    # Mũi tên dự báo
    fig.add_trace(go.Scatter(
        x=[fdate], y=[proj], mode='markers+text', showlegend=False,
        marker=dict(color=col, size=18, symbol='triangle-up' if pred_up else 'triangle-down',
                    line=dict(color='#0E1117', width=1)),
        text=[f"  {'TĂNG' if pred_up else 'GIẢM'} · {prob_up*100:.0f}%"],
        textposition='middle right', textfont=dict(color=col, size=13),
        hovertemplate=f"Dự báo {fdate.date()}<br>P(tăng) = {prob_up*100:.1f}%<extra></extra>"))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=46, b=10), autosize=True,
        title=dict(text=f"{ticker} — {n} phiên gần nhất & dự báo phiên {fdate.date()}", font=dict(size=14)),
        hovermode='x unified', showlegend=False,
        xaxis=dict(showgrid=False, range=[d['time'].iloc[0], x_end]),
        yaxis=dict(title='Giá đóng cửa'))
    return fig


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


# ── Multi-model router (chọn model tốt nhất theo từng mã) ────────────────────
MODEL_LABELS = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'GBT': 'Gradient Boosting',
                'SVC': 'Linear SVC', 'XGB': 'XGBoost', 'ENS': 'Ensemble', 'GRU': 'GRU'}


@st.cache_resource(show_spinner=False)
def load_multi(market):
    """Bundle đa model do train_multi_models.py xuất, hoặc None."""
    p = ART_DIR / f"multi_models_{market}.pkl"
    return joblib.load(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_router(market):
    """router_{market}.json: lựa chọn model/mã + metric test, hoặc None."""
    p = ART_DIR / f"router_{market}.json"
    return json.load(open(p, encoding='utf-8')) if p.exists() else None


def _proba_with(model_name, bundle, X):
    models = bundle['models']
    if model_name == 'ENS':
        return np.mean([models[m].predict_proba(X)[:, 1] for m in bundle['ens_members']], axis=0)
    return models[model_name].predict_proba(X)[:, 1]


@st.cache_data(show_spinner=False)
def predict_series(ticker, market, model_name):
    """Trả DataFrame features + cột prob_up theo model_name chỉ định
    (LR/RF/GBT/SVC/XGB/ENS hoặc GRU)."""
    df = features_for(ticker, market)
    if model_name == 'GRU':
        out = gru_prob_cached(ticker, market)
        return df.merge(out, on='time', how='left') if out is not None else df.assign(prob_up=np.nan)
    bundle = load_multi(market)
    if bundle is None:
        return df.assign(prob_up=np.nan)
    feat = bundle['features']
    d = df.copy()
    d['prob_up'] = np.nan
    mask = d[feat].notna().all(axis=1)
    if mask.any():
        X = bundle['scaler'].transform(d.loc[mask, feat].values.astype(np.float64))
        d.loc[mask, 'prob_up'] = _proba_with(model_name, bundle, X)
    return d


def _router_pick(router, ticker):
    """Model 'Auto' cho 1 mã = lựa chọn của router (hoặc global-best nếu mã không có)."""
    if not router:
        return None
    return router['per_ticker'].get(ticker, {}).get('best_model') or router.get('global_best')


def tab_predict(art):
    st.subheader("🔮 Dự báo xu hướng giá phiên kế tiếp")
    st.caption("Dự báo xác suất giá TĂNG vượt ngưỡng vào phiên giao dịch tiếp theo "
               "(ngưỡng: VN ±2%, US ±1%). Chọn **Auto** để dùng model tốt nhất cho mã đó "
               "(LR/RF/GBT/SVC/XGB/Ensemble + GRU cho VN), hoặc chỉ định 1 model để so sánh.")
    c1, c2 = st.columns([1, 3])
    with c1:
        mkt = st.radio("Thị trường", ['VN', 'US'], horizontal=True, key="pred_mkt")
        tickers = F.VN_TICKERS if mkt == 'VN' else F.US_TICKERS
        ticker = st.selectbox("Chọn mã cổ phiếu", sorted(tickers),
                              format_func=lambda t: f"{t} — {NAMES.get(t, t)}", key="pred_tk")

        # Selector model: Auto (chọn theo mã) hoặc chỉ định 1 model cụ thể
        router = load_router(mkt)
        multi = load_multi(mkt)
        chosen = None
        if multi is not None and router is not None:
            auto_pick = _router_pick(router, ticker)
            opts = ['Auto'] + router['candidates']

            def _mlabel(o, _ap=auto_pick):
                if o == 'Auto':
                    return f"Auto · tốt nhất cho mã ({MODEL_LABELS.get(_ap, _ap)})"
                return MODEL_LABELS.get(o, o)
            sel = st.selectbox("Model dự báo", opts, format_func=_mlabel, key="pred_model",
                               help="Auto = model có Edge validation tốt nhất cho mã này "
                                    "(đã regularize). Hoặc chọn tay để so sánh.")
            chosen = auto_pick if sel == 'Auto' else sel

        # Nút cập nhật dữ liệu THẬT (US + VN, ghi vào csv_demo) + hiển thị ngày dữ liệu hiện có
        import update_data as U
        ds = U.data_status()
        _fmt = lambda d: d.date().isoformat() if d is not None else "—"
        st.caption(f"Dữ liệu hiện có — US: **{_fmt(ds['us_latest'])}** · VN: **{_fmt(ds['vn_latest'])}**")
        if st.button("🔄 Cập nhật dữ liệu mới nhất", key="refresh_data", use_container_width=True):
            with st.spinner("Đang tải dữ liệu mới nhất (US + VN)..."):
                bar = st.progress(0.0)
                res = U.ensure_data_fresh(
                    progress_cb=lambda l, f: bar.progress(min(max(f, 0.0), 1.0)),
                    force=True)
            st.cache_data.clear()  # xoá cache để nạp lại dữ liệu mới
            if res.get("errors"):
                st.warning(f"{len(res['errors'])} mã chưa tải được — thử lại sau.")
            st.success(f"Đã cập nhật: +{res['us_added']} dòng US · +{res['vn_added']} dòng VN")
            st.rerun()

    if mkt not in art:
        st.error(f"Chưa có model cho thị trường {mkt}. Chạy: python demo/train_models.py")
        return

    df = features_for(ticker, mkt)
    if chosen is not None:
        df_p = predict_series(ticker, mkt, chosen)
        pinfo = router['per_ticker'].get(ticker, {})
        # Chỉ số test khớp ĐÚNG model đang chọn (Auto hoặc chọn tay), không chỉ model thắng
        cm = (pinfo.get('cand_metrics') or {}).get(chosen)
        if cm is None:                       # fallback JSON cũ: dùng metric top-level
            cm = {'acc': pinfo.get('test_acc'), 'edge': pinfo.get('test_edge'),
                  'auc': pinfo.get('test_auc'), 'f1': pinfo.get('test_f1'),
                  'precision': pinfo.get('test_precision'), 'recall': pinfo.get('test_recall')}
        edge_t = cm.get('edge')
        suffix = f" · Edge test {edge_t*100:+.1f}%" if edge_t is not None else ""
        model_short = MODEL_LABELS.get(chosen, chosen)
        model_label = f"{model_short}{suffix}"
        ta_val = float(cm.get('acc') or 0.0)
    else:
        df_p, model_label, ta_val = _predict_for_tab(df, ticker, mkt, art)
        model_short = model_label.split(' ')[0]
        cm = None
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
            st.plotly_chart(
                forecast_chart(df_p, ticker, mkt, prob_up, pred_up, forecast_date),
                use_container_width=True, key="pred_forecast",
                config={"responsive": True, "displayModeBar": False})
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

    # ── Bảng chỉ số đầy đủ của model đang dùng (minh bạch: đừng chỉ nhìn Accuracy) ──
    if chosen is not None and cm:
        st.caption(
            f"📊 **Chỉ số của {MODEL_LABELS.get(chosen, chosen)} trên mã {ticker}** "
            "(tập test out-of-time ≥2024). Nhãn lệch nên **Accuracy có thể \"cao giả\"** — "
            "hãy xem **Edge** (giỏi hơn đoán mò bao nhiêu) và **F1/Precision/Recall** của lớp "
            "*Tăng* để đánh giá thật chất lượng dự báo.")
        mc = st.columns(6)
        _p = lambda x: "—" if x is None else f"{x*100:.1f}%"
        mc[0].metric("Accuracy", _p(cm.get('acc')))
        ev = cm.get('edge')
        mc[1].metric("Edge vs Naive", "—" if ev is None else f"{ev*100:+.1f}%",
                     help="Accuracy − tỉ lệ đoán theo lớp đa số. > 0 ⇒ thực sự giỏi hơn đoán mò.")
        mc[2].metric("F1 (lớp Tăng)", _p(cm.get('f1')),
                     help="Trung bình điều hòa của Precision & Recall cho lớp 'Tăng'. "
                          "F1 thấp dù Accuracy cao ⇒ model bỏ lỡ tín hiệu tăng.")
        mc[3].metric("Precision", _p(cm.get('precision')),
                     help="Khi model báo 'Tăng' thì đúng bao nhiêu %.")
        mc[4].metric("Recall", _p(cm.get('recall')),
                     help="Trong số phiên thực sự Tăng, model bắt được bao nhiêu %.")
        av = cm.get('auc')
        mc[5].metric("AUC", "—" if av is None else f"{av:.3f}",
                     help="Khả năng phân biệt 2 lớp (0.5 = ngẫu nhiên, 1.0 = hoàn hảo).")
    else:
        st.metric(f"Độ chính xác ({model_short}) trên mã này", f"{ta_val*100:.1f}%")

    st.divider()
    st.markdown(f"**Backtest gần đây** — dự báo (**{model_short}**) "
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
    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix & ROC", "Độ chính xác theo mã",
                                      "Tầm quan trọng features", "🧭 Model theo mã"])
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
        st.caption(
            "**Accuracy** = tỉ lệ đoán đúng. **Edge** = Accuracy − Naive (đoán theo lớp đa số "
            "của riêng mã đó). Edge **> 0** ⇒ model thực sự đoán tốt hơn đoán mò; Edge **≈ 0** "
            "dù Accuracy cao ⇒ \"cao giả\" do nhãn lệch (mã ít biến động vượt ngưỡng).")
        for mkt in ['VN', 'US']:
            if mkt not in art:
                continue
            m = art[mkt]['metrics']
            ta = m['ticker_acc']
            te = m.get('ticker_edge')  # có thể thiếu nếu metrics cũ (chưa train lại)
            # Sắp xếp theo Edge nếu có, không thì theo Accuracy
            order = (pd.Series(te) if te else pd.Series(ta)).sort_values(ascending=True)
            labels = [f"{t} ({NAMES.get(t, t)})" for t in order.index]
            st.markdown(f"#### {'🇻🇳 Việt Nam' if mkt == 'VN' else '🇺🇸 Hoa Kỳ'}")
            cc = st.columns(2)
            with cc[0]:
                accv = pd.Series(ta).reindex(order.index) * 100
                bar = go.Figure(go.Bar(x=accv.values, y=labels, orientation='h',
                                       marker_color=CLR['blue']))
                bar.add_vline(x=50, line=dict(color=CLR['red'], dash='dot'),
                              annotation_text='50%', annotation_position='top')
                bar.update_layout(title="Độ chính xác theo mã", xaxis_title="Accuracy (%)",
                                  height=max(320, len(order) * 22), margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(bar, use_container_width=True)
            with cc[1]:
                if te:
                    edgev = (order * 100)
                    colors = [CLR['teal'] if v > 0 else CLR['red'] for v in edgev.values]
                    bar = go.Figure(go.Bar(x=edgev.values, y=labels, orientation='h',
                                           marker_color=colors))
                    bar.add_vline(x=0, line=dict(color=CLR['muted'], dash='dot'))
                    bar.update_layout(title="Edge vs Naive theo mã",
                                      xaxis_title="Edge (điểm %)",
                                      height=max(320, len(order) * 22), margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(bar, use_container_width=True)
                else:
                    st.info("Chưa có dữ liệu Edge theo mã. Chạy lại: "
                            "`python demo/train_models.py` để sinh `ticker_edge`.")
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
    with tab4:
        tab_model_router()


def tab_model_router():
    """So sánh model zoo & lựa chọn model tốt nhất theo từng mã (multi-model router)."""
    st.markdown("#### 🧭 Chọn model tốt nhất cho từng mã")
    st.caption(
        "Train **6 model** (LR, RF, GBT, SVC, XGBoost, Ensemble — khớp notebook; thêm **GRU** cho VN) "
        "trên toàn thị trường, rồi với mỗi mã chọn model có **Edge cao nhất trên tập validation "
        "(2022–2023)**, báo cáo trên **test (≥2024)** — out-of-time, không data snooping.")
    have = any(load_router(m) for m in ['VN', 'US'])
    if not have:
        st.warning("Chưa có dữ liệu router. Chạy: `python demo/train_multi_models.py`")
        return

    # ── Đối chứng trung thực: routed (per-ticker) vs 1 model global ──
    st.markdown("##### 📐 Định tuyến theo mã có thực sự tốt hơn dùng 1 model?")
    rows = []
    for mkt in ['VN', 'US']:
        r = load_router(mkt)
        if not r:
            continue
        me = r['mean_test_edge']
        rows.append({'Thị trường': mkt, 'Global-best (val)': MODEL_LABELS.get(r['global_best'], r['global_best']),
                     'Edge — 1 model global': f"{me['global']*100:+.2f}%",
                     'Edge — router (theo mã)': f"{me['reg']*100:+.2f}%",
                     'Chênh lệch': f"{(me['reg']-me['global'])*100:+.2f} đpt"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info(
        "**Kết luận khoa học:** chọn model riêng theo mã **không** vượt trội hơn dùng 1 model tốt "
        "cho cả thị trường — chênh lệch nằm trong khoảng nhiễu. Lý do: mỗi mã chỉ có ~130 phiên "
        "validation nên 'model thắng' phần lớn là *may rủi*, không lặp lại ở test (overfitting khi "
        "lựa chọn). Vì vậy router đã **regularize**: chỉ đổi model khi nó vượt ≥5 điểm % edge.")

    st.divider()
    for mkt in ['VN', 'US']:
        r = load_router(mkt)
        if not r:
            continue
        st.markdown(f"##### {'🇻🇳 Việt Nam' if mkt == 'VN' else '🇺🇸 Hoa Kỳ'} · "
                    f"global-best **{MODEL_LABELS.get(r['global_best'], r['global_best'])}**")
        cc = st.columns([1, 1.4])
        with cc[0]:
            wins = r.get('model_wins', {})
            ws = pd.Series(wins).sort_values(ascending=True)
            bar = go.Figure(go.Bar(x=ws.values, y=[MODEL_LABELS.get(k, k) for k in ws.index],
                                   orientation='h', marker_color=CLR['teal']))
            bar.update_layout(title="Số mã mỗi model được chọn", xaxis_title="Số mã",
                              height=max(260, len(ws) * 42), margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(bar, use_container_width=True)
        with cc[1]:
            pt = r['per_ticker']
            tbl = pd.DataFrame([
                {'Mã': t, 'Model chọn': MODEL_LABELS.get(v['best_model'], v['best_model']),
                 'Khác global?': '✓' if v.get('overridden') else '',
                 'Edge test': f"{v['test_edge']*100:+.1f}%",
                 'Acc test': f"{v['test_acc']*100:.1f}%",
                 'F1 (Tăng)': f"{v.get('test_f1', 0)*100:.1f}%",
                 'Precision': f"{v.get('test_precision', 0)*100:.1f}%",
                 'Recall': f"{v.get('test_recall', 0)*100:.1f}%"}
                for t, v in pt.items()])
            if not tbl.empty:
                tbl = tbl.sort_values('Edge test', ascending=False)
            st.dataframe(tbl, use_container_width=True, hide_index=True, height=360)


# ── Main ────────────────────────────────────────────────────────────────────
def sync_data_on_startup():
    """Khi mở app: kiểm tra csv_demo có mới nhất không, nếu cũ thì tự tải bổ sung
    (US + VN + VNINDEX). Chỉ chạy 1 lần / phiên app và tối đa 1 lần / ngày (marker).
    Tắt bằng biến môi trường DEMO_AUTO_UPDATE=0."""
    if st.session_state.get("_data_sync_done"):
        return
    st.session_state["_data_sync_done"] = True

    import update_data as U
    status = U.data_status()
    # Đã đồng bộ hôm nay hoặc dữ liệu đã mới -> không làm phiền người dùng
    if status["synced_today"] or (not status["us_stale"] and not status["vn_stale"]):
        return

    with st.status("🔄 Đang kiểm tra & cập nhật dữ liệu mới nhất...", expanded=True) as box:
        bar = st.progress(0.0)
        last = {"t": time.time()}

        def cb(label, frac):
            bar.progress(min(max(frac, 0.0), 1.0))
            # Cập nhật nhãn tối đa ~2 lần/giây để đỡ giật
            if time.time() - last["t"] > 0.4:
                box.update(label=f"🔄 {label}")
                last["t"] = time.time()

        try:
            res = U.ensure_data_fresh(progress_cb=cb)
        except Exception as e:
            box.update(label=f"⚠️ Bỏ qua cập nhật (lỗi: {e})", state="error")
            return

        bar.progress(1.0)
        if res.get("updated"):
            st.cache_data.clear()  # xoá cache features để nạp dữ liệu mới
            box.update(
                label=f"✅ Đã cập nhật: +{res['us_added']} dòng US, +{res['vn_added']} dòng VN",
                state="complete", expanded=False)
        else:
            box.update(label=f"✅ Dữ liệu đã mới nhất ({res.get('reason','')})",
                       state="complete", expanded=False)
        if res.get("errors"):
            st.warning("Một số mã không tải được (sẽ thử lại lần mở sau): "
                       + ", ".join(str(x) for x in res["errors"][:5])
                       + (" ..." if len(res["errors"]) > 5 else ""))


def main():
    inject_theme()
    sync_data_on_startup()
    art = load_artifacts()
    if not art:
        st.error("Chưa có model. Hãy chạy trước: `python demo/train_models.py`")
        st.stop()
    hero(art)
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

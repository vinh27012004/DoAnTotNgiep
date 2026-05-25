"""Thuc nghiem: so sanh cac dinh nghia label cho US de tim cach tang accuracy.
Test tren XGBoost voi features technical co ban (khong leak)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

US = ['AAPL','AMZN','F','GM','GOOGL','META','MSFT','NFLX','AMD','NVDA','TSLA',
      'JPM','BAC','WFC','GS','XOM','CVX','DIS']
VN = ['ACB','BID','CTG','FPT','GAS','HPG','HSG','KDH','MBB','MSN','NKG','NLG',
      'PNJ','SAB','SSI','TCB','VCB','VHM','VIC','VNM','VPB']
import sys
MARKET = sys.argv[1] if len(sys.argv) > 1 else 'US'

# Load S&P500 for context
sp = pd.read_csv('csv/SP500.csv')
sp['time'] = pd.to_datetime(sp['time'])
sp = sp.sort_values('time')
sp['sp500_ret1d'] = sp['close'].pct_change()
sp['sp500_mom5'] = sp['close'] / sp['close'].shift(5) - 1
sp['sp500_ma_ratio'] = sp['close'].rolling(5).mean() / sp['close'].rolling(20).mean() - 1
sp_feat = sp[['time','sp500_ret1d','sp500_mom5','sp500_ma_ratio']].fillna(0)

def build(ticker):
    df = pd.read_csv(f'csv/{ticker}.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    c = df['close']
    df['daily_return'] = c.pct_change()
    for n in [1,2,3,5,10]:
        df[f'lag{n}_return'] = df['daily_return'].shift(n)
    df['price_vs_ma5']  = c/c.rolling(5).mean()-1
    df['price_vs_ma20'] = c/c.rolling(20).mean()-1
    df['price_vs_ma50'] = c/c.rolling(50).mean()-1
    df['momentum_5']  = c/c.shift(5)-1
    df['momentum_10'] = c/c.shift(10)-1
    d = c.diff()
    for n in [7,14]:
        g = d.clip(lower=0).rolling(n).mean(); l=(-d.clip(upper=0)).rolling(n).mean()
        df[f'rsi_{n}'] = 100-100/(1+g/(l+1e-9))
    df['vol5']  = df['daily_return'].rolling(5).std()
    df['vol20'] = df['daily_return'].rolling(20).std()
    df['ticker'] = ticker
    # forward returns
    df['fwd1'] = c.shift(-1)/c - 1
    df['fwd3'] = c.shift(-3)/c - 1
    df['fwd5'] = c.shift(-5)/c - 1
    df = df.merge(sp_feat, on='time', how='left').fillna({'sp500_ret1d':0,'sp500_mom5':0,'sp500_ma_ratio':0})
    df['year'] = df['time'].dt.year
    return df

TICKERS = US if MARKET == 'US' else VN
all_df = pd.concat([build(t) for t in TICKERS], ignore_index=True)

FEATS = ['daily_return','lag1_return','lag2_return','lag3_return','lag5_return','lag10_return',
         'price_vs_ma5','price_vs_ma20','price_vs_ma50','momentum_5','momentum_10',
         'rsi_7','rsi_14','vol5','vol20','sp500_ret1d','sp500_mom5','sp500_ma_ratio']

def test_label(fwd_col, thr, name):
    df = all_df.copy()
    df['label'] = np.where(df[fwd_col] > thr, 1, np.where(df[fwd_col] < -thr, 0, np.nan))
    df = df.dropna(subset=FEATS + ['label'])
    tr = df[df['year'] <= 2021]; ts = df[df['year'] >= 2022]
    if len(ts) < 100 or tr['label'].nunique() < 2:
        print(f'{name}: insufficient'); return
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[FEATS]); ytr = tr['label'].astype(int)
    Xts = sc.transform(ts[FEATS]); yts = ts['label'].astype(int)
    n0,n1 = np.bincount(ytr)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                      scale_pos_weight=n0/max(n1,1), random_state=42, n_jobs=-1,
                      verbosity=0, eval_metric='logloss')
    m.fit(Xtr, ytr)
    yp = m.predict(Xts); pr = m.predict_proba(Xts)[:,1]
    acc = accuracy_score(yts, yp); auc = roc_auc_score(yts, pr)
    naive = max(yts.mean(), 1-yts.mean())
    pass_rate = len(df)/len(all_df.dropna(subset=FEATS))*100
    print(f'{name:32s} | rows={len(df):6d} ({pass_rate:4.0f}% pass) | acc={acc:.4f} | auc={auc:.4f} | naive={naive:.4f} | edge={acc-naive:+.4f}')

print('='*110)
print(f'THUC NGHIEM: cac dinh nghia label cho {MARKET} ({len(TICKERS)} ma, features technical + S&P500)')
print('='*110)
test_label('fwd1', 0.015, 'next-day, threshold 1.5% (HIEN TAI)')
test_label('fwd1', 0.010, 'next-day, threshold 1.0%')
test_label('fwd1', 0.005, 'next-day, threshold 0.5%')
test_label('fwd3', 0.008, '3-day fwd, threshold 0.8% (VN HIEN TAI)')
test_label('fwd3', 0.010, '3-day fwd, threshold 1.0%')
test_label('fwd3', 0.015, '3-day fwd, threshold 1.5%')
test_label('fwd3', 0.020, '3-day fwd, threshold 2.0%')
test_label('fwd5', 0.020, '5-day fwd, threshold 2.0%')
test_label('fwd5', 0.030, '5-day fwd, threshold 3.0%')
print('='*110)
print('edge = acc - naive baseline. edge > 0 nghia la model thuc su hoc duoc gi do.')

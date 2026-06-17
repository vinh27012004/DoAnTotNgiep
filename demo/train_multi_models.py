"""
Train NHIỀU loại model cho mỗi thị trường (tái hiện model zoo của notebook bằng
sklearn/pandas, không cần Spark) rồi CHỌN MODEL TỐT NHẤT CHO TỪNG MÃ.

Model zoo (khớp notebook):
  - LR   : Logistic Regression          (Spark ML LogisticRegression)
  - RF   : Random Forest                (Spark ML RandomForestClassifier)
  - GBT  : Gradient Boosted Trees       (Spark ML GBTClassifier ~ HistGradientBoosting)
  - SVC  : Linear SVC (calibrated)      (Spark ML LinearSVC)
  - XGB  : XGBoost                       (run_xgb_pipeline)
  - ENS  : Ensemble = trung bình xác suất {LR,RF,GBT,SVC}
  - GRU  : (chỉ VN) deep learning từ artifact gru_VN.* nếu có

Cách chọn (KHÔNG data snooping):
  - Tách out-of-time 3 phần: Train ≤2021 · Validation 2022–2023 · Test ≥2024.
  - Train trên Train. Với MỖI mã, chọn model có EDGE cao nhất trên VALIDATION.
  - Báo cáo hiệu năng của model đã chọn trên TEST (acc/auc/edge/naive).

Xuất:
  - artifacts/multi_models_{mkt}.pkl : {scaler, features, models{...}, market}
  - artifacts/router_{mkt}.json      : lựa chọn model/mã + metric test + phân bố model thắng

Chạy:  python demo/train_multi_models.py    (dùng .venv có sklearn/xgboost)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

ART_DIR = Path(__file__).resolve().parent / "artifacts"
ART_DIR.mkdir(exist_ok=True)

TRAIN_END = 2021          # Train: year <= 2021
VAL_YEARS = (2022, 2023)  # Validation: 2022–2023 (chọn model)
TEST_START = 2024         # Test: year >= 2024 (báo cáo)

# Các model tham gia ensemble (khớp notebook: LR/RF/GBT/SVC)
ENS_MEMBERS = ['LR', 'RF', 'GBT', 'SVC']


# ── Build & train từng model ────────────────────────────────────────────────
def _build_models(spw):
    """Khởi tạo model zoo. spw = scale_pos_weight (xử lý lệch nhãn)."""
    cw = 'balanced'
    return {
        'LR':  LogisticRegression(max_iter=1000, C=1.0, class_weight=cw),
        'RF':  RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20,
                                      class_weight=cw, random_state=42, n_jobs=-1),
        'GBT': HistGradientBoostingClassifier(max_iter=250, max_depth=4, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=42),
        'SVC': CalibratedClassifierCV(
                   LinearSVC(C=0.5, class_weight=cw, max_iter=5000, dual='auto'),
                   method='sigmoid', cv=3),
        'XGB': XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                             scale_pos_weight=spw, random_state=42, n_jobs=-1,
                             verbosity=0, eval_metric='logloss'),
    }


def _fit_all(models, X_tr, y_tr, sw):
    for name, m in models.items():
        if name == 'GBT':
            m.fit(X_tr, y_tr, sample_weight=sw)   # HGB nhận sample_weight để cân bằng nhãn
        else:
            m.fit(X_tr, y_tr)
    return models


def _proba_all(models, X):
    """Trả dict {model_name: prob_up[]} gồm cả ENS (trung bình LR/RF/GBT/SVC)."""
    p = {name: m.predict_proba(X)[:, 1] for name, m in models.items()}
    p['ENS'] = np.mean([p[k] for k in ENS_MEMBERS], axis=0)
    return p


# ── GRU (VN) từ artifact: xác suất theo từng (ticker,time) ───────────────────
def _gru_probs(df_market, market):
    """Tính prob_up của GRU cho toàn thị trường (nếu có artifact). Trả DataFrame
    [ticker, time, prob_up] hoặc None."""
    meta_p = ART_DIR / f"gru_{market}_meta.json"
    model_p = ART_DIR / f"gru_{market}.keras"
    scaler_p = ART_DIR / f"gru_{market}_scaler.pkl"
    if not (meta_p.exists() and model_p.exists() and scaler_p.exists()):
        return None
    try:
        import os
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        from tensorflow.keras.models import load_model
        meta = json.load(open(meta_p, encoding='utf-8'))
        feat_list, seq_len = meta['features'], int(meta['seq_len'])
        gscaler = joblib.load(scaler_p)
        gmodel = load_model(model_p)
    except Exception as e:
        print(f"  [GRU] bỏ qua ({e})")
        return None
    if any(c not in df_market.columns for c in feat_list):
        return None
    out = []
    for tk, g in df_market.groupby('ticker'):
        gc = g.dropna(subset=feat_list).sort_values('time').reset_index(drop=True)
        if len(gc) <= seq_len:
            continue
        Xa = gscaler.transform(gc[feat_list].values.astype(np.float64))
        Xs = np.array([Xa[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(gc))],
                      dtype=np.float32)
        prob = gmodel.predict(Xs, verbose=0).flatten()
        sub = gc.iloc[seq_len - 1:][['ticker', 'time']].copy()
        sub['GRU'] = prob
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else None


# ── Metric theo mã ──────────────────────────────────────────────────────────
def _edge(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan
    acc = accuracy_score(y_true, y_pred)
    naive = max(float(np.mean(y_true == 0)), float(np.mean(y_true == 1)))
    return acc, naive, acc - naive


def _auc(y_true, prob):
    return float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float('nan')


def train_market(market):
    tickers = F.US_TICKERS if market == 'US' else F.VN_TICKERS
    base_cols = F.FEATURE_COLS_US if market == 'US' else F.FEATURE_COLS_VN
    feat = F.xgb_feature_cols(base_cols)

    print(f"\n{'='*64}\n{market} — multi-model router\n{'='*64}")
    raw = F.load_market_raw(tickers)
    df_full = F.compute_features(raw, market)          # giữ bản đầy đủ cho GRU sequences
    df = F.make_labels(df_full, market)
    df = df.dropna(subset=feat + ['label']).copy()
    df['label'] = df['label'].astype(int)

    tr = df[df['year'] <= TRAIN_END]
    va = df[(df['year'] >= VAL_YEARS[0]) & (df['year'] <= VAL_YEARS[1])]
    te = df[df['year'] >= TEST_START]
    print(f"  Train ≤{TRAIN_END}: {len(tr):,d} | Val {VAL_YEARS[0]}–{VAL_YEARS[1]}: {len(va):,d} | "
          f"Test ≥{TEST_START}: {len(te):,d}")
    if len(va) < 200 or len(te) < 200:
        print("  ⚠️ Không đủ dữ liệu val/test — bỏ qua thị trường này.")
        return None

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr[feat].values.astype(np.float64))
    y_tr = tr['label'].values
    X_va = scaler.transform(va[feat].values.astype(np.float64))
    X_te = scaler.transform(te[feat].values.astype(np.float64))

    c = np.bincount(y_tr)
    spw = float(c[0]) / float(max(int(c[1]) if len(c) > 1 else 1, 1))
    sw = np.where(y_tr == 1, spw, 1.0)

    print("  Train model zoo: LR, RF, GBT, SVC, XGB ...")
    models = _fit_all(_build_models(spw), X_tr, y_tr, sw)

    # Xác suất trên val/test cho mọi model (gồm ENS)
    pv = _proba_all(models, X_va)
    pt = _proba_all(models, X_te)
    cand = list(pv.keys())  # LR,RF,GBT,SVC,XGB,ENS

    # GRU (VN): ghép theo (ticker,time)
    gru_df = _gru_probs(df_full, market)
    if gru_df is not None:
        va = va.merge(gru_df, on=['ticker', 'time'], how='left')
        te = te.merge(gru_df, on=['ticker', 'time'], how='left')
        pv['GRU'] = va['GRU'].values
        pt['GRU'] = te['GRU'].values
        cand.append('GRU')
        print(f"  + GRU candidate (VN): {gru_df['GRU'].notna().sum():,d} dự đoán")

    va = va.reset_index(drop=True)
    te = te.reset_index(drop=True)
    yv, yt = va['label'].values, te['label'].values

    MARGIN = 0.05  # mã chỉ được đổi khỏi model global khi edge val vượt ≥5 điểm %

    # ── Global-best: model có edge VALIDATION (gộp toàn thị trường) cao nhất ──
    val_edge_global = {}
    for m in cand:
        pvm = pv[m]
        if np.any(pd.isna(pvm)):           # GRU có thể thiếu vài dòng -> bỏ NaN
            ok = ~pd.isna(pvm)
            if ok.sum() == 0:
                continue
            _, _, e = _edge(yv[ok], (pvm[ok] >= 0.5).astype(int))
        else:
            _, _, e = _edge(yv, (pvm >= 0.5).astype(int))
        val_edge_global[m] = e
    global_best = max(val_edge_global, key=val_edge_global.get)
    print(f"  Global-best (val) = {global_best} (val_edge gộp={val_edge_global[global_best]:+.4f})")

    # ── Chọn model/mã: edge val cao nhất, NHƯNG chỉ override global khi vượt MARGIN ──
    per_ticker = {}
    wins_naive, wins_reg = {}, {}
    edges = {'global': [], 'naive': [], 'reg': []}
    for tk in tickers:
        mv = (va['ticker'] == tk).values
        mt = (te['ticker'] == tk).values
        if mv.sum() < 20 or mt.sum() < 20:
            continue
        val_scores = {}
        for m in cand:
            pvm = pv[m][mv]
            if np.any(pd.isna(pvm)):
                continue
            _, _, e = _edge(yv[mv], (pvm >= 0.5).astype(int))
            val_scores[m] = (e, _auc(yv[mv], pvm))
        if global_best not in val_scores:
            continue
        naive_pick = max(val_scores, key=lambda m: (val_scores[m][0], val_scores[m][1]))
        # regularized: chỉ đổi nếu vượt global ≥ MARGIN trên validation
        reg_pick = (naive_pick
                    if val_scores[naive_pick][0] - val_scores[global_best][0] >= MARGIN
                    else global_best)

        def _test_edge(m):
            ptm = pt[m][mt]
            if np.any(pd.isna(ptm)):
                return None
            acc, naive, edge = _edge(yt[mt], (ptm >= 0.5).astype(int))
            return acc, naive, edge, _auc(yt[mt], ptm)

        rg = _test_edge(global_best)
        rn = _test_edge(naive_pick)
        rr = _test_edge(reg_pick)
        if rg: edges['global'].append(rg[2])
        if rn: edges['naive'].append(rn[2])
        if rr: edges['reg'].append(rr[2])
        wins_naive[naive_pick] = wins_naive.get(naive_pick, 0) + 1
        wins_reg[reg_pick] = wins_reg.get(reg_pick, 0) + 1

        acc, naive, edge, auc = rr
        per_ticker[tk] = {
            'best_model': reg_pick,                 # model app sẽ dùng cho mã này
            'naive_pick': naive_pick,               # lựa chọn nếu KHÔNG regularize
            'overridden': bool(reg_pick != global_best),
            'val_edge': round(float(val_scores[reg_pick][0]), 4),
            'test_acc': round(float(acc), 4),
            'test_naive': round(float(naive), 4),
            'test_edge': round(float(edge), 4),
            'test_auc': round(float(auc), 4),
            'n_test': int(mt.sum()),
        }

    mean_edge = {k: (float(np.nanmean(v)) if v else float('nan')) for k, v in edges.items()}
    print(f"  Phân bố (regularized): {wins_reg}  | naive: {wins_naive}")
    print(f"  Edge TB test — global({global_best})={mean_edge['global']:+.4f} | "
          f"router-naive={mean_edge['naive']:+.4f} | router-reg={mean_edge['reg']:+.4f}")

    # ── Xuất artifacts ──
    joblib.dump({'scaler': scaler, 'features': feat, 'models': models,
                 'ens_members': ENS_MEMBERS, 'market': market},
                ART_DIR / f"multi_models_{market}.pkl")
    router = {
        'market': market,
        'split': {'train_end': TRAIN_END, 'val_years': list(VAL_YEARS), 'test_start': TEST_START},
        'candidates': cand,
        'margin': MARGIN,
        'global_best': global_best,
        'val_edge_global': {m: round(float(v), 4) for m, v in val_edge_global.items()},
        'per_ticker': per_ticker,
        'model_wins': wins_reg,
        'model_wins_naive': wins_naive,
        'mean_test_edge': {k: round(v, 4) for k, v in mean_edge.items()},
    }
    with open(ART_DIR / f"router_{market}.json", 'w', encoding='utf-8') as fp:
        json.dump(router, fp, ensure_ascii=False, indent=2)
    return router


if __name__ == '__main__':
    rs = {}
    for mkt in ['VN', 'US']:
        r = train_market(mkt)
        if r:
            rs[mkt] = r
    print(f"\n{'='*64}\nTONG KET — Edge TB trên TEST (out-of-time ≥{TEST_START})\n{'='*64}")
    for mkt, r in rs.items():
        me = r['mean_test_edge']
        print(f"  {mkt}: global({r['global_best']})={me['global']:+.4f} | "
              f"router-naive={me['naive']:+.4f} | router-reg(dùng)={me['reg']:+.4f}  "
              f"→ reg vs global: {me['reg'] - me['global']:+.4f}")
    print(f"\nArtifacts: {ART_DIR}")

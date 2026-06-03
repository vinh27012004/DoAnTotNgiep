"""
Train XGBoost cho thị trường US và VN — tái tạo run_xgb_pipeline() của notebook.
Lưu model + scaler + metrics ra demo/artifacts/ để app load nhanh (không cần Spark).

Chạy:  python demo/train_models.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

ART_DIR = Path(__file__).resolve().parent / "artifacts"
ART_DIR.mkdir(exist_ok=True)


def train_market(market):
    tickers = F.US_TICKERS if market == 'US' else F.VN_TICKERS
    base_cols = F.FEATURE_COLS_US if market == 'US' else F.FEATURE_COLS_VN
    feat = F.xgb_feature_cols(base_cols)

    print(f"\n{'='*60}\n{market} — load & feature engineering\n{'='*60}")
    raw = F.load_market_raw(tickers)
    df = F.compute_features(raw, market)
    df = F.make_labels(df, market)

    df = df.dropna(subset=feat + ['label']).copy()
    df['label'] = df['label'].astype(int)
    print(f"  {len(df):,d} rows | {df['ticker'].nunique()} tickers | {len(feat)} features")

    df_tr = df[df['year'] <= 2021]
    df_ts = df[df['year'] >= 2022]
    print(f"  Train (<=2021): {len(df_tr):,d} | Test (>=2022): {len(df_ts):,d}")

    # ── Grid search trên validation 2020-2021 (tránh lookahead) ──
    df_sub = df_tr[df_tr['year'] <= 2019]
    df_val = df_tr[(df_tr['year'] >= 2020) & (df_tr['year'] <= 2021)]
    sc_gs = StandardScaler()
    X_sub = sc_gs.fit_transform(df_sub[feat].values.astype(np.float64))
    y_sub = df_sub['label'].values
    X_val = sc_gs.transform(df_val[feat].values.astype(np.float64))
    y_val = df_val['label'].values
    cnt = np.bincount(y_sub)
    spw = float(cnt[0]) / float(max(int(cnt[1]) if len(cnt) > 1 else 1, 1))

    best_auc, best = -1.0, {'max_depth': 5, 'learning_rate': 0.03}
    for d in [3, 5]:
        for lr in [0.03, 0.1]:
            m = XGBClassifier(n_estimators=200, max_depth=d, learning_rate=lr,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                              scale_pos_weight=spw, random_state=42, n_jobs=-1, verbosity=0)
            m.fit(X_sub, y_sub, verbose=False)
            p = m.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, p) if len(np.unique(y_val)) > 1 else 0.0
            print(f"    depth={d} lr={lr}: val_AUC={auc:.4f}")
            if auc > best_auc:
                best_auc, best = auc, {'max_depth': d, 'learning_rate': lr}
    print(f"  Best: {best} (val_AUC={best_auc:.4f})")

    # ── Train final ──
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat].values.astype(np.float64))
    y_tr = df_tr['label'].values
    X_ts = scaler.transform(df_ts[feat].values.astype(np.float64))
    y_ts = df_ts['label'].values
    c = np.bincount(y_tr)
    n0 = int(c[0]) if len(c) > 0 else 1
    n1 = int(c[1]) if len(c) > 1 else 1

    model = XGBClassifier(n_estimators=400, max_depth=best['max_depth'],
                          learning_rate=best['learning_rate'],
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                          scale_pos_weight=float(n0) / float(max(n1, 1)),
                          random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss')
    model.fit(X_tr, y_tr, eval_set=[(X_ts, y_ts)], verbose=False)

    y_pred = model.predict(X_ts)
    y_prob = model.predict_proba(X_ts)[:, 1]
    acc = accuracy_score(y_ts, y_pred)
    auc = roc_auc_score(y_ts, y_prob)
    cm = confusion_matrix(y_ts, y_pred).tolist()

    df_ts_c = df_ts.copy()
    df_ts_c['pred'] = y_pred
    ticker_acc = (df_ts_c.groupby('ticker')
                  .apply(lambda gp: accuracy_score(gp['label'], gp['pred']))
                  .sort_values(ascending=False))
    fi = pd.Series(model.feature_importances_, index=feat).sort_values(ascending=False)

    # Naive baseline (luôn đoán theo lớp đa số) để tính "edge"
    naive_acc = max(np.mean(y_ts == 0), np.mean(y_ts == 1))

    print(f"  XGBoost {market}: Accuracy={acc:.4f}  AUC={auc:.4f}  (naive={naive_acc:.4f})")
    print(f"  Top 5: {', '.join(fi.head(5).index.tolist())}")

    # ── Lưu artifacts ──
    joblib.dump({'model': model, 'scaler': scaler, 'features': feat,
                 'market': market, 'best_params': best},
                ART_DIR / f"model_{market}.pkl")

    metrics = {
        'market': market,
        'accuracy': float(acc),
        'auc': float(auc),
        'naive_acc': float(naive_acc),
        'edge': float(acc - naive_acc),
        'n_train': int(len(df_tr)),
        'n_test': int(len(df_ts)),
        'n_features': len(feat),
        'best_params': best,
        'confusion_matrix': cm,
        'ticker_acc': {k: float(v) for k, v in ticker_acc.items()},
        'feature_importance': {k: float(v) for k, v in fi.head(15).items()},
        'roc': _roc_points(y_ts, y_prob),
        'label_dist_test': {'down': int(np.sum(y_ts == 0)), 'up': int(np.sum(y_ts == 1))},
    }
    with open(ART_DIR / f"metrics_{market}.json", 'w', encoding='utf-8') as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    return metrics


def _roc_points(y_true, y_prob, n=50):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int)
    return {'fpr': fpr[idx].tolist(), 'tpr': tpr[idx].tolist()}


if __name__ == '__main__':
    summary = {}
    for mkt in ['VN', 'US']:
        m = train_market(mkt)
        summary[mkt] = {'accuracy': m['accuracy'], 'auc': m['auc'], 'edge': m['edge']}
    print(f"\n{'='*60}\nTONG KET\n{'='*60}")
    for mkt, s in summary.items():
        print(f"  {mkt}: acc={s['accuracy']:.4f}  auc={s['auc']:.4f}  edge={s['edge']:+.4f}")
    print(f"\nArtifacts luu tai: {ART_DIR}")

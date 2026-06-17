"""
Train model GRU (deep learning) cho thị trường VN bằng pandas/TensorFlow — tái hiện
`run_dl_pipeline()` của notebook, KHÔNG cần Spark. Xuất artifact để app & router dùng:

  artifacts/gru_{mkt}.keras        — model Keras
  artifacts/gru_{mkt}_scaler.pkl   — StandardScaler (fit trên train ≤2021)
  artifacts/gru_{mkt}_meta.json    — {market, features, seq_len, accuracy, auc, ticker_acc}

Trước đây artifact này do notebook sinh; nay tách hẳn sang demo để demo tự dựng được GRU.
Định dạng giữ NGUYÊN để app (load_gru_bundle / gru_prob_cached) và train_multi_models.py
(candidate GRU) đọc không đổi.

Chạy:  .venv\\Scripts\\python.exe demo\\train_gru.py            (mặc định VN)
       .venv\\Scripts\\python.exe demo\\train_gru.py --market US
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

ART_DIR = Path(__file__).resolve().parent / "artifacts"
ART_DIR.mkdir(exist_ok=True)

SEQ_LEN = 20
EPOCHS = 40
BATCH = 128


def _build_sequences(df_labeled, df_context, feat, scaler, seq_len):
    """Dựng chuỗi từ CONTEXT (đủ phiên, kể cả phiên trung tính) cho liền mạch,
    nhưng chỉ phát ra các phiên cuối có nhãn 0/1 (khớp notebook)."""
    Xs, ys, tks, tms = [], [], [], []
    for tkr, g in df_context.sort_values('time').groupby('ticker'):
        if len(g) <= seq_len:
            continue
        Xg = scaler.transform(g[feat].fillna(0).values.astype(np.float32))
        tg = pd.to_datetime(g['time'].values)
        gl = df_labeled[df_labeled['ticker'] == tkr]
        lbl_map = dict(zip(pd.to_datetime(gl['time'].values), gl['label'].astype(int).values))
        for i in range(seq_len, len(Xg)):
            lbl = lbl_map.get(tg[i], -1)
            if lbl in (0, 1):
                Xs.append(Xg[i - seq_len + 1:i + 1])
                ys.append(lbl)
                tks.append(tkr)
                tms.append(tg[i])
    if not Xs:
        return (np.empty((0, seq_len, len(feat)), np.float32), np.array([], int), [], [])
    return np.array(Xs, np.float32), np.array(ys, int), tks, tms


def train_gru(market='VN', seq_len=SEQ_LEN, epochs=EPOCHS, batch=BATCH):
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    tickers = F.US_TICKERS if market == 'US' else F.VN_TICKERS
    base_cols = F.FEATURE_COLS_US if market == 'US' else F.FEATURE_COLS_VN
    feat = F.xgb_feature_cols(base_cols)   # == feat_dl của notebook

    print(f"\n{'='*64}\nGRU (deep learning) — {market} ({len(feat)} features)\n{'='*64}")
    raw = F.load_market_raw(tickers)
    df_all = F.compute_features(raw, market)
    df_all = F.make_labels(df_all, market)

    # context = mọi phiên (điền 0 chỗ thiếu) · labeled = phiên có nhãn 0/1
    df_ctx = df_all.copy()
    df_lbl = df_all.dropna(subset=feat + ['label']).copy()
    df_lbl = df_lbl[df_lbl['label'].isin([0, 1])].copy()
    df_lbl['label'] = df_lbl['label'].astype(int)

    lbl_tr = df_lbl[df_lbl['year'] <= 2021]
    ctx_tr = df_ctx[df_ctx['year'] <= 2021]
    lbl_ts = df_lbl[df_lbl['year'] >= 2022]
    ctx_ts = df_ctx[df_ctx['year'] >= 2022]

    scaler = StandardScaler()
    scaler.fit(lbl_tr[feat].values.astype(np.float32))

    X_tr, y_tr, _, _ = _build_sequences(lbl_tr, ctx_tr, feat, scaler, seq_len)
    X_ts, y_ts, tk_ts, _ = _build_sequences(lbl_ts, ctx_ts, feat, scaler, seq_len)
    print(f"  Train: {len(X_tr):,d} seqs | Test: {len(X_ts):,d} seqs | shape={X_tr.shape}")
    if len(X_tr) == 0 or len(X_ts) == 0:
        print("  ⚠️ Không đủ chuỗi — bỏ qua.")
        return None

    cnt = np.bincount(y_tr)
    n0, n1 = int(cnt[0]), int(cnt[1] if len(cnt) > 1 else 0)
    tot = n0 + n1
    cw = {0: tot / (2.0 * max(n0, 1)), 1: tot / (2.0 * max(n1, 1))}
    print(f"  Class 0={n0:,d} | 1={n1:,d} | weight={cw}")

    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        Input(shape=(seq_len, len(feat))),
        GRU(64, return_sequences=True, kernel_regularizer=l2(5e-4)),
        BatchNormalization(), Dropout(0.5),
        GRU(32, kernel_regularizer=l2(5e-4)),
        BatchNormalization(), Dropout(0.5),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=Adam(learning_rate=5e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    cbs = [EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
           ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6, monitor='val_loss')]
    print(f"  Training GRU (epochs≤{epochs}, batch={batch}) ...")
    hist = model.fit(X_tr, y_tr, validation_split=0.15, epochs=epochs, batch_size=batch,
                     class_weight=cw, callbacks=cbs, verbose=0)

    y_prob = model.predict(X_ts, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    acc = float(accuracy_score(y_ts, y_pred))
    auc = float(roc_auc_score(y_ts, y_prob))
    ta = (pd.DataFrame({'ticker': tk_ts, 'pred': y_pred, 'label': y_ts})
          .groupby('ticker').apply(lambda g: accuracy_score(g['label'], g['pred'])))
    print(f"  GRU {market}: Acc={acc:.4f}  AUC={auc:.4f}  (epochs={len(hist.history['loss'])})")

    # ── Xuất artifact (định dạng khớp app) ──
    model.save(str(ART_DIR / f"gru_{market}.keras"))
    joblib.dump(scaler, ART_DIR / f"gru_{market}_scaler.pkl")
    meta = {'market': market, 'features': list(feat), 'seq_len': int(seq_len),
            'accuracy': acc, 'auc': auc,
            'ticker_acc': {str(k): float(v) for k, v in ta.items()}}
    with open(ART_DIR / f"gru_{market}_meta.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Đã xuất: gru_{market}.keras (+scaler, meta) -> {ART_DIR}")
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--market', default='VN', choices=['VN', 'US'])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()
    train_gru(market=args.market, epochs=args.epochs)
    print("\nXong! App (tab Dự báo) & train_multi_models.py (candidate GRU) sẽ tự dùng artifact này.")

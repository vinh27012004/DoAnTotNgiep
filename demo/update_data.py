"""
Tự động kiểm tra & cập nhật dữ liệu csv_demo cho app demo.

Khi app khởi động sẽ gọi ``ensure_data_fresh()``:
  1. Đọc ngày mới nhất đang có trong csv_demo.
  2. Nếu đã đồng bộ trong hôm nay (marker ``.last_sync.json``) -> bỏ qua.
  3. Nếu dữ liệu cũ hơn phiên giao dịch gần nhất -> tải bổ sung (incremental):
       - Cổ phiếu US + macro VIX/SP500/TNX/IRX : yfinance (giữ nguyên scale gốc).
       - Cổ phiếu VN                          : vnstock (source VCI).
       - VNINDEX                              : vnstock, CHIA 1000 để khớp scale
                                                 file cũ (file ghi 1.67 = ~1674 điểm).
  4. Ghi nối (append) các dòng mới, khử trùng theo 'time', lưu lại csv_demo.
  5. Ghi marker = hôm nay để không lặp lại trong ngày.

Thiết kế phòng thủ: mọi lỗi mạng / thiếu thư viện đều được bắt và bỏ qua để
KHÔNG làm sập app. Hàm trả về dict tóm tắt để app hiển thị.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# Dùng chung danh sách mã & thư mục dữ liệu với app (features.py)
try:
    import features as F  # khi chạy trong context của app (sys.path đã thêm demo/)
except ImportError:  # pragma: no cover - fallback khi import dạng package
    from . import features as F  # type: ignore

CSV_DIR: Path = F.CSV_DIR
MARKER = CSV_DIR / ".last_sync.json"
# .env ở thư mục gốc dự án (demo/ -> ..)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# Macro US tải qua yfinance: (ticker yfinance, tên file, schema)
#   'ohlcv' -> time,open,high,low,close,volume ; 'rate' -> time,rate
US_MACROS = [
    ("^VIX", "VIX", "ohlcv"),
    ("^GSPC", "SP500", "ohlcv"),
    ("^TNX", "TNX", "rate"),
    ("^IRX", "IRX", "rate"),
]
VNINDEX_SCALE = 1000.0  # file VNINDEX lưu giá thực / 1000


# ── Tiện ích ────────────────────────────────────────────────────────────────
def _load_env() -> None:
    """Nạp biến từ .env (gốc dự án) vào os.environ nếu chưa có. KHÔNG ghi đè
    biến môi trường đã tồn tại."""
    if not ENV_PATH.exists():
        return
    try:
        for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_VN_REGISTERED = False


def _register_vnstock() -> bool:
    """Đăng ký VNSTOCK_API_KEY (nếu có) để nâng giới hạn truy cập. Chỉ làm 1 lần.
    Trả về True nếu đã đăng ký bằng API key, False nếu chạy chế độ Guest."""
    global _VN_REGISTERED
    if _VN_REGISTERED:
        return True
    _load_env()
    api_key = os.getenv("VNSTOCK_API_KEY", "").strip()
    if not api_key:
        return False
    try:
        from vnstock import register_user
        ok = register_user(api_key=api_key)
        _VN_REGISTERED = bool(ok)
        return _VN_REGISTERED
    except Exception:
        return False


def _today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now()).normalize()


def _expected_session() -> pd.Timestamp:
    """Phiên giao dịch gần nhất *đã chắc chắn đóng cửa* = ngày làm việc trước hôm nay.
    Dùng làm mốc: dữ liệu coi là 'mới' nếu ngày cuối >= mốc này (bỏ qua ngày lễ)."""
    return (_today() - pd.offsets.BDay(1)).normalize()


def _latest_date(path: Path) -> pd.Timestamp | None:
    """Ngày lớn nhất trong 1 file csv (cột 'time'), hoặc None nếu không đọc được."""
    try:
        s = pd.read_csv(path, usecols=["time"])["time"]
        return pd.to_datetime(s, errors="coerce").max()
    except Exception:
        return None


def _read_marker() -> str | None:
    try:
        return json.loads(MARKER.read_text(encoding="utf-8")).get("date")
    except Exception:
        return None


def _write_marker(date_str: str) -> None:
    try:
        MARKER.write_text(json.dumps({"date": date_str}), encoding="utf-8")
    except Exception:
        pass


def _append_csv(path: Path, new_df: pd.DataFrame) -> int:
    """Ghi nối new_df vào file (khử trùng theo 'time', ưu tiên dòng mới).
    Trả về SỐ DÒNG NGÀY MỚI thực sự được thêm."""
    if new_df is None or len(new_df) == 0:
        return 0
    new_df = new_df.copy()
    new_df["time"] = pd.to_datetime(new_df["time"], errors="coerce")
    new_df = new_df.dropna(subset=["time"])
    if path.exists():
        old = pd.read_csv(path)
        old["time"] = pd.to_datetime(old["time"], errors="coerce")
        old_dates = set(old["time"].dropna())
        n_new = int((~new_df["time"].isin(old_dates)).sum())
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        n_new = len(new_df)
        combined = new_df
    combined = (combined
                .drop_duplicates(subset=["time"], keep="last")
                .sort_values("time")
                .reset_index(drop=True))
    combined["time"] = pd.to_datetime(combined["time"]).dt.strftime("%Y-%m-%d")
    combined.to_csv(path, index=False)
    return n_new


# ── Chuẩn hoá dữ liệu yfinance ──────────────────────────────────────────────
def _yf_ohlcv(raw: pd.DataFrame) -> pd.DataFrame | None:
    if raw is None or len(raw) == 0:
        return None
    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.rename(columns={"Date": "time", "Open": "open", "High": "high",
                              "Low": "low", "Close": "close", "Volume": "volume"})
    keep = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in raw.columns]
    return raw[keep]


def _yf_rate(raw: pd.DataFrame) -> pd.DataFrame | None:
    if raw is None or len(raw) == 0:
        return None
    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.rename(columns={"Date": "time", "Close": "close"})
    if "close" not in raw.columns:
        return None
    return raw[["time", "close"]].rename(columns={"close": "rate"})


# ── Cập nhật từng thị trường ────────────────────────────────────────────────
def _update_yf_one(ticker: str, fname: str, schema: str, errors: list) -> int:
    """Tải bổ sung 1 mã/macro US qua yfinance, ghi nối vào csv_demo."""
    import yfinance as yf

    path = CSV_DIR / f"{fname}.csv"
    latest = _latest_date(path)
    start = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if latest is not None else "2013-01-01"
    end = (_today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if latest is not None and pd.Timestamp(start) > _today():
        return 0  # đã có tới hôm nay
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        new = _yf_rate(raw) if schema == "rate" else _yf_ohlcv(raw)
        return _append_csv(path, new)
    except Exception as e:  # pragma: no cover - lỗi mạng
        errors.append(f"{fname}: {e}")
        return 0


def update_us(progress_cb=None) -> tuple[int, list]:
    """Cập nhật toàn bộ cổ phiếu US + macro. Trả về (số dòng mới, danh sách lỗi)."""
    errors: list = []
    added = 0
    jobs = [(t, t, "ohlcv") for t in F.US_TICKERS] + US_MACROS
    for i, (ticker, fname, schema) in enumerate(jobs, 1):
        if progress_cb:
            progress_cb(f"US {i}/{len(jobs)}: {fname}", i / len(jobs))
        added += _update_yf_one(ticker, fname, schema, errors)
    return added, errors


def update_vn(progress_cb=None, delay=None) -> tuple[int, list]:
    """Cập nhật cổ phiếu VN + VNINDEX qua vnstock. Trả về (số dòng mới, lỗi)."""
    import time as _time
    from vnstock import Quote

    _register_vnstock()  # nâng giới hạn nếu có VNSTOCK_API_KEY trong .env
    if delay is None:
        # Có API key -> được phép gọi dày hơn; Guest -> để delay cao cho an toàn
        default_delay = "1.0" if _VN_REGISTERED else "2.0"
        delay = float(os.getenv("VNSTOCK_REQUEST_DELAY", default_delay))
    errors: list = []
    added = 0
    end = _today().strftime("%Y-%m-%d")
    jobs = [(t, t, 1.0) for t in F.VN_TICKERS] + [("VNINDEX", "VNINDEX", VNINDEX_SCALE)]

    for i, (symbol, fname, scale) in enumerate(jobs, 1):
        if progress_cb:
            progress_cb(f"VN {i}/{len(jobs)}: {fname}", i / len(jobs))
        path = CSV_DIR / f"{fname}.csv"
        latest = _latest_date(path)
        start = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if latest is not None else "2014-01-01"
        if latest is not None and pd.Timestamp(start) > _today():
            continue  # đã có tới hôm nay
        df = None
        for attempt in range(3):
            try:
                q = Quote(symbol=symbol, source="VCI", show_log=False)
                df = q.history(start=start, end=end, interval="1D")
                break
            except Exception as exc:
                msg = str(exc).lower()
                if "rate limit" in msg or "giới hạn" in msg:
                    _time.sleep(30)
                else:
                    errors.append(f"{fname}: {exc}")
                    df = None
                    break
        if df is not None and len(df) > 0:
            cols = ["time", "open", "high", "low", "close", "volume"]
            new = df[[c for c in cols if c in df.columns]].copy()
            if scale != 1.0:
                for c in ["open", "high", "low", "close"]:
                    if c in new.columns:
                        new[c] = new[c] / scale
            added += _append_csv(path, new)
        if delay > 0:
            _time.sleep(delay)
    return added, errors


# ── Kiểm tra mức độ "mới" của dữ liệu ───────────────────────────────────────
def _market_latest(tickers) -> pd.Timestamp | None:
    """Ngày cuối NHỎ NHẤT trong nhóm mã (mã cũ nhất quyết định độ mới của thị trường)."""
    dates = [d for d in (_latest_date(CSV_DIR / f"{t}.csv") for t in tickers) if d is not None]
    return min(dates) if dates else None


def data_status() -> dict:
    """Tóm tắt độ mới của dữ liệu (không tải gì). Dùng để hiển thị / quyết định."""
    exp = _expected_session()
    us_latest = _market_latest(F.US_TICKERS)
    vn_latest = _market_latest(F.VN_TICKERS)
    vni_latest = _latest_date(CSV_DIR / "VNINDEX.csv")
    return {
        "expected_session": exp,
        "us_latest": us_latest,
        "vn_latest": vn_latest,
        "vnindex_latest": vni_latest,
        "us_stale": us_latest is None or us_latest < exp,
        "vn_stale": (vn_latest is None or vn_latest < exp)
                    or (vni_latest is None or vni_latest < exp),
        "synced_today": _read_marker() == _today().strftime("%Y-%m-%d"),
    }


# ── Điều phối chính ─────────────────────────────────────────────────────────
def ensure_data_fresh(progress_cb=None, force=False) -> dict:
    """Đảm bảo csv_demo cập nhật tới phiên gần nhất.

    progress_cb(label, fraction) : callback hiển thị tiến trình (tuỳ chọn).
    force                        : bỏ qua marker, ép cập nhật.

    Trả về dict: {updated, us_added, vn_added, skipped, reason, errors, status}.
    """
    if os.getenv("DEMO_AUTO_UPDATE", "1") == "0" and not force:
        return {"updated": False, "skipped": True, "reason": "DEMO_AUTO_UPDATE=0",
                "us_added": 0, "vn_added": 0, "errors": [], "status": data_status()}

    today_str = _today().strftime("%Y-%m-%d")
    if not force and _read_marker() == today_str:
        return {"updated": False, "skipped": True, "reason": "đã đồng bộ hôm nay",
                "us_added": 0, "vn_added": 0, "errors": [], "status": data_status()}

    status = data_status()
    if not force and not status["us_stale"] and not status["vn_stale"]:
        _write_marker(today_str)  # đã mới -> đánh dấu để khỏi kiểm tra lại trong ngày
        return {"updated": False, "skipped": True, "reason": "dữ liệu đã mới nhất",
                "us_added": 0, "vn_added": 0, "errors": [], "status": status}

    us_added = vn_added = 0
    errors: list = []

    def _split_cb(market, lo, hi):
        if progress_cb is None:
            return None
        return lambda label, frac: progress_cb(label, lo + (hi - lo) * frac)

    if force or status["us_stale"]:
        n, e = update_us(_split_cb("US", 0.0, 0.5))
        us_added += n
        errors += e
    if force or status["vn_stale"]:
        n, e = update_vn(_split_cb("VN", 0.5, 1.0))
        vn_added += n
        errors += e

    # Chỉ ghi marker khi không có lỗi mạng (để lần mở sau còn thử lại)
    if not errors:
        _write_marker(today_str)

    return {"updated": (us_added + vn_added) > 0,
            "skipped": False,
            "reason": "đã cập nhật",
            "us_added": us_added,
            "vn_added": vn_added,
            "errors": errors,
            "status": data_status()}


if __name__ == "__main__":  # chạy tay: python demo/update_data.py
    def _cli(label, frac):
        print(f"[{frac*100:5.1f}%] {label}")
    print("Bắt đầu kiểm tra & cập nhật csv_demo...")
    res = ensure_data_fresh(progress_cb=_cli, force="--force" in os.sys.argv)
    print(json.dumps({k: str(v) for k, v in res.items()}, ensure_ascii=False, indent=2))

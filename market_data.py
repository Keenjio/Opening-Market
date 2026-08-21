"""
market_data.py
===============
Shared data-fetching and scoring logic for the Pre-Market Dashboard.

This module is imported by BOTH:
  - streamlit_app.py         (live dashboard, uses st.cache_data)
  - capture_score.py         (headless script run by GitHub Actions
                               at fixed times to log a score snapshot)

Keeping the logic here (instead of duplicating it) guarantees the score
shown live in the app and the score written to history are computed
identically.
"""

from datetime import datetime

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

US_INDICES = {
    "Nasdaq Composite": "^IXIC",
    "Dow Jones": "^DJI",
}

# Singapore market - Straits Times Index. Closes before Indian open,
# making it the best yfinance-available proxy for SGX Nifty direction.
SGX_TICKER = "^STI"
SGX_LABEL = "SGX / Straits Times Index"

# Nifty 50 index - tracks the actual market movement during the day
NIFTY_INDEX_TICKER = "^NSEI"

INDIA_VIX_TICKER = "^INDIAVIX"
VIX_THRESHOLD = 15.0

NIFTY_50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "WIPRO.NS",
    "HCLTECH.NS", "M&M.NS", "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "ADANIENT.NS", "JSWSTEEL.NS", "ONGC.NS", "COALINDIA.NS",
]

NIFTY_FUTURES_CANDIDATES = [SGX_TICKER, "NIFTY_FIN_SERVICE.NS", "NIFTY.NS", "^NSEI"]

# The intraday capture slots (IST). "09:00" and "16:00" are the two
# once-a-day "official" snapshots used for the next-day correlation;
# 09:30 to 15:30 is the intraday hourly tracking grid during market hours;
# "21:00" captures post-US-close data for next-morning scoring.
CAPTURE_SLOTS = [
    "09:00", "09:30", "10:30", "11:30", "12:30",
    "13:30", "14:30", "15:30", "16:00", "21:00",
]
PRIMARY_SLOTS = {"09:00", "16:00"}  # used for the day-over-day verdict

# ---------------------------------------------------------------------------
# Raw fetchers
# ---------------------------------------------------------------------------


def pct_change(hist):
    if hist is None or hist.empty or len(hist) < 2:
        return None, None, None
    last_close = hist["Close"].iloc[-1]
    prev_close = hist["Close"].iloc[-2]
    change = (last_close - prev_close) / prev_close * 100
    date_str = hist.index[-1].strftime("%d %b %Y") if hasattr(hist.index[-1], "strftime") else str(hist.index[-1])[:10]
    return float(last_close), float(change), date_str


def fetch_us_markets():
    results = {}
    for name, ticker in US_INDICES.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            results[name] = pct_change(hist)
        except Exception:
            results[name] = (None, None, None)
    return results


def fetch_india_vix():
    try:
        hist = yf.Ticker(INDIA_VIX_TICKER).history(period="5d")
        if hist.empty:
            return None, None
        date_str = hist.index[-1].strftime("%d %b %Y") if hasattr(hist.index[-1], "strftime") else str(hist.index[-1])[:10]
        return float(hist["Close"].iloc[-1]), date_str
    except Exception:
        return None, None


def fetch_market_breadth(symbols=None):
    """
    Scrapes full real-time Advances and Declines directly from official NSE India:
    - Complete 500-stock constituent table (LTP, % Change, Volume, Value, Status)
    - Full market breadth across Nifty 50, Nifty 500, Midcaps, Smallcaps, and Total NSE Universe.
    """
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/market-data/advance",
    }

    try:
        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com", timeout=6)

        # 1. Scrape full 500 stocks from NIFTY 500
        r_500 = session.get("https://www.nseindia.com/api/equity-stock-indices?index=NIFTY%20500", timeout=8)
        df_500 = pd.DataFrame()
        n500_adv, n500_dec, n500_unc = 0, 0, 0
        if r_500.status_code == 200:
            data = r_500.json().get("data", [])
            # data[0] is often the index summary record
            stock_data = data[1:] if len(data) > 1 and "symbol" in data[0] and data[0]["symbol"].startswith("NIFTY") else data
            rows = []
            for item in stock_data:
                sym = item.get("symbol", "")
                if not sym or sym.startswith("NIFTY"):
                    continue
                series = item.get("series", "EQ")
                try:
                    ltp = float(item.get("lastPrice", 0))
                    chg_pct = float(item.get("pChange", 0))
                    vol = int(item.get("totalTradedVolume", 0))
                    val = float(item.get("totalTradedValue", 0))
                except Exception:
                    continue
                status = "Advance" if chg_pct > 0 else ("Decline" if chg_pct < 0 else "Unchanged")
                rows.append({
                    "Symbol": sym,
                    "Series": series,
                    "Last": ltp,
                    "Change %": chg_pct,
                    "Volume": vol,
                    "Value (₹ Cr)": round(val / 1e7, 2),
                    "Status": status,
                })
            if rows:
                df_500 = pd.DataFrame(rows)
                n500_adv = int((df_500["Status"] == "Advance").sum())
                n500_dec = int((df_500["Status"] == "Decline").sum())
                n500_unc = int((df_500["Status"] == "Unchanged").sum())

        # 2. Fetch Nifty 50 exact breadth & macro matrix from allIndices
        n50_adv, n50_dec, n50_unc = 0, 0, 0
        all_adv, all_dec, all_unc = 0, 0, 0
        mid_adv, mid_dec, mid_unc = 0, 0, 0
        small_adv, small_dec, small_unc = 0, 0, 0
        total_mkt_adv, total_mkt_dec, total_mkt_unc = 0, 0, 0

        r_all = session.get("https://www.nseindia.com/api/allIndices", timeout=6)
        if r_all.status_code == 200:
            j_all = r_all.json()
            all_adv = int(j_all.get("advances", 0) or 0)
            all_dec = int(j_all.get("declines", 0) or 0)
            all_unc = int(j_all.get("unchanged", 0) or 0)

            for item in j_all.get("data", []):
                idx_name = item.get("index", "")
                if idx_name == "NIFTY 50":
                    n50_adv = int(item.get("advances", 0) or 0)
                    n50_dec = int(item.get("declines", 0) or 0)
                    n50_unc = int(item.get("unchanged", 0) or 0)
                elif idx_name == "NIFTY MIDCAP 150":
                    mid_adv = int(item.get("advances", 0) or 0)
                    mid_dec = int(item.get("declines", 0) or 0)
                    mid_unc = int(item.get("unchanged", 0) or 0)
                elif idx_name == "NIFTY SMALLCAP 250":
                    small_adv = int(item.get("advances", 0) or 0)
                    small_dec = int(item.get("declines", 0) or 0)
                    small_unc = int(item.get("unchanged", 0) or 0)
                elif idx_name == "NIFTY TOTAL MARKET":
                    total_mkt_adv = int(item.get("advances", 0) or 0)
                    total_mkt_dec = int(item.get("declines", 0) or 0)
                    total_mkt_unc = int(item.get("unchanged", 0) or 0)

        matrix = {
            "n500": (n500_adv, n500_dec, n500_unc),
            "midcap": (mid_adv, mid_dec, mid_unc),
            "smallcap": (small_adv, small_dec, small_unc),
            "total_mkt": (total_mkt_adv, total_mkt_dec, total_mkt_unc),
            "all_nse": (all_adv, all_dec, all_unc),
        }

        # If Nifty 50 was fetched or computed from 500, return
        if n50_adv > 0 or n50_dec > 0 or not df_500.empty:
            if n50_adv == 0 and not df_500.empty:
                # filter first 50
                n50_df = df_500.head(50)
                n50_adv = int((n50_df["Status"] == "Advance").sum())
                n50_dec = int((n50_df["Status"] == "Decline").sum())
                n50_unc = int((n50_df["Status"] == "Unchanged").sum())

            return n50_adv, n50_dec, n50_unc, df_500, matrix
    except Exception:
        pass

    # Fallback to yfinance if NSE direct endpoint is throttled
    symbols = symbols or NIFTY_50_SYMBOLS
    advances, declines, unchanged = 0, 0, 0
    rows = []
    try:
        data = yf.download(
            symbols, period="5d", group_by="ticker",
            progress=False, threads=True, auto_adjust=False,
        )
    except Exception:
        data = None

    for sym in symbols:
        try:
            if data is not None and sym in data.columns.get_level_values(0):
                hist = data[sym]
            else:
                hist = yf.Ticker(sym).history(period="5d")
            res = pct_change(hist)
            last, change = res[0], res[1]
            if change is None:
                continue
            if change > 0:
                advances += 1
            elif change < 0:
                declines += 1
            else:
                unchanged += 1
            status = "Advance" if change > 0 else ("Decline" if change < 0 else "Unchanged")
            rows.append({
                "Symbol": sym.replace(".NS", ""),
                "Series": "EQ",
                "Last": last,
                "Change %": change,
                "Volume": 0,
                "Value (₹ Cr)": 0.0,
                "Status": status,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    fallback_matrix = {
        "n500": (advances, declines, unchanged),
        "midcap": (0, 0, 0),
        "smallcap": (0, 0, 0),
        "total_mkt": (0, 0, 0),
        "all_nse": (advances, declines, unchanged),
    }
    return int(advances), int(declines), int(unchanged), df, fallback_matrix


def fetch_sgx_market():
    """Fetch Singapore market (Straits Times Index) % change and date."""
    try:
        hist = yf.Ticker(SGX_TICKER).history(period="5d")
        last, change, date_str = pct_change(hist)
        return last, change, date_str
    except Exception:
        return None, None, None


def fetch_nifty_index():
    """Fetch Nifty 50 index: open, last close, % change, and date."""
    try:
        hist = yf.Ticker(NIFTY_INDEX_TICKER).history(period="5d")
        if hist is None or hist.empty or len(hist) < 2:
            return None, None, None, None, None, None
        last_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])
        today_open = float(hist["Open"].iloc[-1])
        today_high = float(hist["High"].iloc[-1])
        today_low = float(hist["Low"].iloc[-1])
        change = (last_close - prev_close) / prev_close * 100
        date_str = hist.index[-1].strftime("%d %b %Y") if hasattr(hist.index[-1], "strftime") else str(hist.index[-1])[:10]
        return today_open, last_close, today_high, today_low, round(change, 4), date_str
    except Exception:
        return None, None, None, None, None, None


def fetch_nifty_intraday():
    """
    Fetch Nifty 50 intraday hourly data for the most recent trading day.
    Returns a DataFrame with hourly OHLC during market hours (9:15-15:30 IST).
    yfinance provides up to 7 days of 1h data.
    """
    try:
        hist = yf.Ticker(NIFTY_INDEX_TICKER).history(period="5d", interval="1h")
        if hist is None or hist.empty:
            return pd.DataFrame()
        # Convert to IST for display
        if hist.index.tz is not None:
            from datetime import timedelta, timezone
            IST = timezone(timedelta(hours=5, minutes=30))
            hist.index = hist.index.tz_convert(IST)
        # Add a date column and time column for easy filtering
        hist["date"] = hist.index.date
        hist["time"] = hist.index.strftime("%H:%M")
        return hist[["Open", "High", "Low", "Close", "Volume", "date", "time"]]
    except Exception:
        return pd.DataFrame()


def fetch_global_cue(us_results):
    # Try Nifty futures first (GIFT Nifty)
    for ticker in NIFTY_FUTURES_CANDIDATES:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            last, change, date_str = pct_change(hist)
            if change is not None:
                return ticker, last, change, False, date_str
        except Exception:
            continue

    valid_changes = [res[1] for res in us_results.values() if res[1] is not None]
    if valid_changes:
        avg_change = sum(valid_changes) / len(valid_changes)
        return "US-market proxy (avg Nasdaq+Dow)", None, avg_change, True, None

    return None, None, None, True, None


# ---------------------------------------------------------------------------
# Combined snapshot + score
# ---------------------------------------------------------------------------


def _score_from_signals(signals):
    """Shared scoring: sum signals to produce (score, n_signals, verdict)."""
    if signals:
        score = sum(signals)
        n_signals = len(signals)
        if score > 0:
            verdict = "GREEN"
        elif score < 0:
            verdict = "RED"
        else:
            verdict = "FLAT"
    else:
        score, n_signals, verdict = None, 0, "NO_DATA"
    return score, n_signals, verdict


def compute_snapshot():
    """
    Fetches everything needed right now and returns one dict containing
    every raw input plus the derived verdict/score, underlying dates, and fetch timestamp.
    """
    us_results = fetch_us_markets()
    vix, vix_date = fetch_india_vix()
    advances, declines, unchanged, breadth_df, matrix = fetch_market_breadth(NIFTY_50_SYMBOLS)
    total = advances + declines + unchanged
    sgx_last, sgx_change, sgx_date = fetch_sgx_market()
    nifty_open, nifty_last, nifty_high, nifty_low, nifty_change, nifty_date = fetch_nifty_index()
    nifty_intraday = fetch_nifty_intraday()
    src, gc_last, gc_change, is_fallback, gc_date = fetch_global_cue(us_results)

    # US session date (from Nasdaq or Dow)
    us_date = next((res[2] for res in us_results.values() if res[2] is not None), None)

    signals = []
    for _, res in us_results.items():
        change = res[1]
        if change is not None:
            signals.append(1 if change >= 0 else -1)
    if total > 0:
        signals.append(1 if advances > declines else (-1 if declines > advances else 0))
    if gc_change is not None:
        signals.append(1 if gc_change >= 0 else -1)

    score, n_signals, verdict = _score_from_signals(signals)

    breadth_sentiment = None
    if total > 0:
        if advances > declines:
            breadth_sentiment = "BULLISH"
        elif declines > advances:
            breadth_sentiment = "BEARISH"
        else:
            breadth_sentiment = "NEUTRAL"

    n500_adv, n500_dec, n500_unc = matrix.get("n500", (0, 0, 0))

    return {
        "fetched_at_ist": datetime.now().strftime("%A, %d %B %Y — %H:%M:%S IST"),
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "us_session_date": us_date,
        "sgx_session_date": sgx_date,
        "nifty_session_date": nifty_date,
        "vix_session_date": vix_date,
        "nasdaq_last": us_results.get("Nasdaq Composite", (None, None, None))[0],
        "nasdaq_change_pct": us_results.get("Nasdaq Composite", (None, None, None))[1],
        "dow_last": us_results.get("Dow Jones", (None, None, None))[0],
        "dow_change_pct": us_results.get("Dow Jones", (None, None, None))[1],
        "sgx_last": sgx_last,
        "sgx_change_pct": sgx_change,
        "nifty_open": nifty_open,
        "nifty_last": nifty_last,
        "nifty_high": nifty_high,
        "nifty_low": nifty_low,
        "nifty_change_pct": nifty_change,
        "india_vix": vix,
        "vix_condition": (
            "VOLATILE" if (vix is not None and vix > VIX_THRESHOLD)
            else "STABLE" if vix is not None else None
        ),
        "advances": advances,
        "declines": declines,
        "unchanged": unchanged,
        "n500_advances": n500_adv,
        "n500_declines": n500_dec,
        "n500_unchanged": n500_unc,
        "breadth_matrix": matrix,
        "breadth_sentiment": breadth_sentiment,
        "global_cue_source": src,
        "global_cue_last": gc_last,
        "global_cue_change_pct": gc_change,
        "global_cue_is_fallback": is_fallback,
        "score": score,
        "n_signals": n_signals,
        "verdict": verdict,
        # not persisted to CSV (too large) but returned for live UI use
        "_breadth_df": breadth_df,
        "_nifty_intraday": nifty_intraday,
    }


def fetch_nifty_intraday_history(period="1mo", interval="1h"):
    """
    Fetch historical Nifty 50 intraday bars (e.g. 1h or 15m).
    Returns DataFrame indexed by IST datetime with date, time, OHLCV.
    """
    try:
        hist = yf.Ticker(NIFTY_INDEX_TICKER).history(period=period, interval=interval)
        if hist is None or hist.empty:
            return pd.DataFrame()
        if hist.index.tz is not None:
            from datetime import timedelta, timezone
            IST = timezone(timedelta(hours=5, minutes=30))
            hist.index = hist.index.tz_convert(IST)
        hist["date_ist"] = hist.index.date
        hist["time_ist"] = hist.index.strftime("%H:%M")
        return hist
    except Exception:
        return pd.DataFrame()


def compute_historical_score(
    nasdaq_chg,
    dow_chg,
    sgx_chg,
    vix=None,
    nifty_open=None,
    nifty_last=None,
    nifty_high=None,
    nifty_low=None,
    nifty_change_pct=None,
):
    """
    Compute a score from pre-fetched historical daily % changes.

    Used by backfill_history.py. Breadth is unavailable historically,
    so only 3 signals are used: Nasdaq, Dow, SGX.

    Returns a dict with the same keys as compute_snapshot() (minus
    breadth and global cue), suitable for writing to score_log.csv.
    """
    signals = []
    if nasdaq_chg is not None:
        signals.append(1 if nasdaq_chg >= 0 else -1)
    if dow_chg is not None:
        signals.append(1 if dow_chg >= 0 else -1)
    if sgx_chg is not None:
        signals.append(1 if sgx_chg >= 0 else -1)

    score, n_signals, verdict = _score_from_signals(signals)

    return {
        "nasdaq_change_pct": nasdaq_chg,
        "dow_change_pct": dow_chg,
        "sgx_change_pct": sgx_chg,
        "nifty_open": nifty_open,
        "nifty_last": nifty_last,
        "nifty_high": nifty_high,
        "nifty_low": nifty_low,
        "nifty_change_pct": nifty_change_pct,
        "india_vix": vix,
        "vix_condition": (
            "VOLATILE" if (vix is not None and vix > VIX_THRESHOLD)
            else "STABLE" if vix is not None else None
        ),
        "advances": None,
        "declines": None,
        "unchanged": None,
        "breadth_sentiment": None,
        "global_cue_source": SGX_LABEL,
        "global_cue_change_pct": sgx_chg,
        "global_cue_is_fallback": False,
        "score": score,
        "n_signals": n_signals,
        "verdict": verdict,
    }

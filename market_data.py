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

NIFTY_FUTURES_CANDIDATES = ["NIFTY_FIN_SERVICE.NS", "NIFTY.NS", "^NSEI"]

# The intraday capture slots (IST). "09:00" and "16:00" are the two
# once-a-day "official" snapshots used for the next-day correlation;
# everything from 10:00-15:00 plus 15:30 is the intraday tracking grid.
CAPTURE_SLOTS = [
    "09:00", "10:00", "11:00", "12:00", "13:00",
    "14:00", "15:00", "15:30", "16:00",
]
PRIMARY_SLOTS = {"09:00", "16:00"}  # used for the day-over-day verdict

# ---------------------------------------------------------------------------
# Raw fetchers
# ---------------------------------------------------------------------------


def pct_change(hist):
    if hist is None or hist.empty or len(hist) < 2:
        return None, None
    last_close = hist["Close"].iloc[-1]
    prev_close = hist["Close"].iloc[-2]
    change = (last_close - prev_close) / prev_close * 100
    return float(last_close), float(change)


def fetch_us_markets():
    results = {}
    for name, ticker in US_INDICES.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            results[name] = pct_change(hist)
        except Exception:
            results[name] = (None, None)
    return results


def fetch_india_vix():
    try:
        hist = yf.Ticker(INDIA_VIX_TICKER).history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


def fetch_market_breadth(symbols):
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
            last, change = pct_change(hist)
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
                "Last": last,
                "Change %": change,
                "Status": status,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    return advances, declines, unchanged, df


def fetch_global_cue(us_results):
    for ticker in NIFTY_FUTURES_CANDIDATES:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            last, change = pct_change(hist)
            if change is not None:
                return ticker, last, change, False
        except Exception:
            continue

    valid_changes = [c for _, c in us_results.values() if c is not None]
    if valid_changes:
        avg_change = sum(valid_changes) / len(valid_changes)
        return "US-market proxy (avg Nasdaq+Dow)", None, avg_change, True

    return None, None, None, True


# ---------------------------------------------------------------------------
# Combined snapshot + score
# ---------------------------------------------------------------------------


def compute_snapshot():
    """
    Fetches everything needed right now and returns one dict containing
    every raw input plus the derived verdict/score. This dict is what
    gets shown live in the app AND what gets flattened into a CSV row
    by capture_score.py, so the two are always in sync.
    """
    us_results = fetch_us_markets()
    vix = fetch_india_vix()
    advances, declines, unchanged, breadth_df = fetch_market_breadth(NIFTY_50_SYMBOLS)
    total = advances + declines + unchanged
    src, gc_last, gc_change, is_fallback = fetch_global_cue(us_results)

    signals = []
    for _, (last, change) in us_results.items():
        if change is not None:
            signals.append(1 if change >= 0 else -1)
    if total > 0:
        signals.append(1 if advances > declines else (-1 if declines > advances else 0))
    if gc_change is not None:
        signals.append(1 if gc_change >= 0 else -1)

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

    breadth_sentiment = None
    if total > 0:
        if advances > declines:
            breadth_sentiment = "BULLISH"
        elif declines > advances:
            breadth_sentiment = "BEARISH"
        else:
            breadth_sentiment = "NEUTRAL"

    return {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "nasdaq_change_pct": us_results.get("Nasdaq Composite", (None, None))[1],
        "dow_change_pct": us_results.get("Dow Jones", (None, None))[1],
        "india_vix": vix,
        "vix_condition": (
            "VOLATILE" if (vix is not None and vix > VIX_THRESHOLD)
            else "STABLE" if vix is not None else None
        ),
        "advances": advances,
        "declines": declines,
        "unchanged": unchanged,
        "breadth_sentiment": breadth_sentiment,
        "global_cue_source": src,
        "global_cue_change_pct": gc_change,
        "global_cue_is_fallback": is_fallback,
        "score": score,
        "n_signals": n_signals,
        "verdict": verdict,
        # not persisted to CSV (too large) but returned for live UI use
        "_breadth_df": breadth_df,
    }

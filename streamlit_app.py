"""
Indian Stock Market Pre-Market Analysis Dashboard — Streamlit App
====================================================================
Run with:
    pip install streamlit yfinance
    streamlit run streamlit_app.py

Sections:
  1. US Markets (Nasdaq & Dow Jones)
  2. India VIX (volatility regime)
  3. Nifty 50 market breadth (advances vs declines)
  4. Global cue / GIFT Nifty proxy
  5. Combined pre-market verdict
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pre-Market Dashboard",
    page_icon="📈",
    layout="wide",
)

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

# ---------------------------------------------------------------------------
# Data fetching (cached so re-runs / widget interactions don't re-hit Yahoo)
# ---------------------------------------------------------------------------


def pct_change(hist):
    if hist is None or hist.empty or len(hist) < 2:
        return None, None
    last_close = hist["Close"].iloc[-1]
    prev_close = hist["Close"].iloc[-2]
    change = (last_close - prev_close) / prev_close * 100
    return float(last_close), float(change)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_us_markets():
    results = {}
    for name, ticker in US_INDICES.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            results[name] = pct_change(hist)
        except Exception:
            results[name] = (None, None)
    return results


@st.cache_data(ttl=300, show_spinner=False)
def fetch_india_vix():
    try:
        hist = yf.Ticker(INDIA_VIX_TICKER).history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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
# UI helpers
# ---------------------------------------------------------------------------


def status_badge(change, up_word="GREEN", down_word="RED"):
    if change is None:
        return "⚪ N/A"
    if change >= 0:
        return f"🟢 {up_word} ▲ {change:+.2f}%"
    return f"🔴 {down_word} ▼ {change:+.2f}%"


def color_change(val):
    if pd.isna(val):
        return ""
    color = "#16a34a" if val >= 0 else "#dc2626"
    return f"color: {color}; font-weight: 600;"


def status_color(val):
    if val == "Advance":
        return "color: #16a34a; font-weight: 600;"
    if val == "Decline":
        return "color: #dc2626; font-weight: 600;"
    return "color: #6b7280;"


def styled_table(df):
    """Format + color-code the Change % (and Status, if present) columns,
    compatible with old and new pandas (Styler.applymap was renamed to
    Styler.map in pandas 2.1+)."""
    styler = df.style.format({"Last": "{:.2f}", "Change %": "{:+.2f}%"})
    apply_fn = "map" if hasattr(styler, "map") else "applymap"
    styler = getattr(styler, apply_fn)(color_change, subset=["Change %"])
    if "Status" in df.columns:
        styler = getattr(styler, apply_fn)(status_color, subset=["Status"])
    return styler


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("📈 Indian Market Pre-Open Dashboard")
st.caption(datetime.now().strftime("%A, %d %B %Y — %H:%M:%S"))

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

st.divider()

# --- 1. US Markets ---------------------------------------------------------
st.subheader("1. US Markets (Previous Close)")
with st.spinner("Fetching US market data..."):
    us_results = fetch_us_markets()

cols = st.columns(len(us_results))
for col, (name, (last, change)) in zip(cols, us_results.items()):
    with col:
        st.metric(
            label=name,
            value=f"{last:,.2f}" if last is not None else "N/A",
            delta=f"{change:+.2f}%" if change is not None else None,
        )

st.divider()

# --- 2. India VIX ------------------------------------------------------------
st.subheader("2. India VIX (Volatility Gauge)")
with st.spinner("Fetching India VIX..."):
    vix = fetch_india_vix()

vix_col1, vix_col2 = st.columns([1, 2])
with vix_col1:
    st.metric("India VIX", f"{vix:.2f}" if vix is not None else "N/A")
with vix_col2:
    if vix is not None:
        if vix > VIX_THRESHOLD:
            st.error(f"⚠️ Market Condition: **VOLATILE** (VIX {vix:.2f} > {VIX_THRESHOLD})")
        else:
            st.success(f"✅ Market Condition: **STABLE** (VIX {vix:.2f} ≤ {VIX_THRESHOLD})")
    else:
        st.warning("India VIX data unavailable")

st.divider()

# --- 3. Market Breadth -------------------------------------------------------
st.subheader(f"3. Market Breadth (Nifty sample, {len(NIFTY_50_SYMBOLS)} stocks)")
with st.spinner("Fetching Nifty constituent data..."):
    advances, declines, unchanged, breadth_df = fetch_market_breadth(NIFTY_50_SYMBOLS)

total = advances + declines + unchanged
if total > 0:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Advances", advances, delta_color="normal")
    b2.metric("Declines", declines)
    b3.metric("Unchanged", unchanged)

    if advances > declines:
        sentiment, color = "BULLISH", "green"
    elif declines > advances:
        sentiment, color = "BEARISH", "red"
    else:
        sentiment, color = "NEUTRAL", "orange"
    b4.markdown(f"**Breadth Sentiment**  \n:{color}[**{sentiment}**]")

    # simple advance/decline bar chart
    st.bar_chart(
        pd.DataFrame({"Advances": [advances], "Declines": [declines], "Unchanged": [unchanged]}).T,
        use_container_width=True,
    )

    st.markdown("#### Advances vs Declines — Full List")
    adv_col, dec_col = st.columns(2)
    with adv_col:
        st.markdown(f"**🟢 Advances ({advances})**")
        adv_table = breadth_df[breadth_df["Status"] == "Advance"].sort_values(
            "Change %", ascending=False
        )
        if not adv_table.empty:
            st.dataframe(
                styled_table(adv_table[["Symbol", "Last", "Change %"]]),
                hide_index=True,
                use_container_width=True,
                height=min(38 * (len(adv_table) + 1), 400),
            )
        else:
            st.caption("No advancing stocks in this sample.")
    with dec_col:
        st.markdown(f"**🔴 Declines ({declines})**")
        dec_table = breadth_df[breadth_df["Status"] == "Decline"].sort_values(
            "Change %", ascending=True
        )
        if not dec_table.empty:
            st.dataframe(
                styled_table(dec_table[["Symbol", "Last", "Change %"]]),
                hide_index=True,
                use_container_width=True,
                height=min(38 * (len(dec_table) + 1), 400),
            )
        else:
            st.caption("No declining stocks in this sample.")

    if unchanged > 0:
        unch_table = breadth_df[breadth_df["Status"] == "Unchanged"][["Symbol", "Last", "Change %"]]
        with st.expander(f"⚪ Unchanged ({unchanged})"):
            st.dataframe(styled_table(unch_table), hide_index=True, use_container_width=True)

    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown("**Top Gainers**")
        top_gainers = breadth_df.sort_values("Change %", ascending=False).head(3)
        st.dataframe(
            styled_table(top_gainers),
            hide_index=True,
            use_container_width=True,
        )
    with gc2:
        st.markdown("**Top Losers**")
        top_losers = breadth_df.sort_values("Change %", ascending=True).head(3)
        st.dataframe(
            styled_table(top_losers),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Full breadth table (all stocks, with status)"):
        st.dataframe(
            styled_table(
                breadth_df[["Symbol", "Last", "Change %", "Status"]].sort_values(
                    "Change %", ascending=False
                )
            ),
            hide_index=True,
            use_container_width=True,
        )
else:
    st.warning("Breadth data unavailable")

st.divider()

# --- 4. Global Cue / GIFT Nifty proxy ---------------------------------------
st.subheader("4. Global Cue / GIFT Nifty Proxy")
with st.spinner("Fetching global cue data..."):
    src, last, change, is_fallback = fetch_global_cue(us_results)

if change is not None:
    gcol1, gcol2 = st.columns([1, 2])
    with gcol1:
        st.metric(
            label=src if not is_fallback else "Nifty Futures",
            value=f"{last:,.2f}" if last is not None else "—",
            delta=f"{change:+.2f}%",
        )
    with gcol2:
        if is_fallback:
            st.info(
                "⚠️ Live GIFT Nifty futures not available via this data source. "
                f"Showing a fallback estimate: **{src}**. "
                "Cross-check the official NSE IX GIFT Nifty page before trading."
            )
        else:
            st.write(f"Source ticker: `{src}`")
else:
    st.warning("No global cue data available")

st.divider()

# --- 5. Combined Verdict -----------------------------------------------------
st.subheader("5. Combined Pre-Market Verdict")

signals = []
for _, (last, change) in us_results.items():
    if change is not None:
        signals.append(1 if change >= 0 else -1)
if total > 0:
    signals.append(1 if advances > declines else (-1 if declines > advances else 0))
if change is not None:
    signals.append(1 if change >= 0 else -1)

if signals:
    score = sum(signals)
    if score > 0:
        st.success(f"### 🟢 LIKELY GREEN OPEN — Signal Score: {score}/{len(signals)}")
    elif score < 0:
        st.error(f"### 🔴 LIKELY RED OPEN — Signal Score: {score}/{len(signals)}")
    else:
        st.warning(f"### 🟡 MIXED / FLAT OPEN — Signal Score: {score}/{len(signals)}")
else:
    st.warning("Insufficient data for a verdict")

st.caption("Disclaimer: For informational purposes only. Not investment advice.")

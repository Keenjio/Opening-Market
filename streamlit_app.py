"""
Indian Stock Market Pre-Market & Intraday Analysis Dashboard
============================================================
Streamlit App with real-time scoring, live Nifty 50 intraday tracking,
and historical score vs market movement analytics.
"""

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data import (
    NIFTY_50_SYMBOLS,
    SGX_LABEL,
    VIX_THRESHOLD,
    compute_snapshot,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Indian Market Pre-Open & Intraday Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "score_log.csv")
PRIMARY_SLOTS = ("09:00", "16:00", "21:00")
INTRADAY_SLOTS = ("09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30")

# ---------------------------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def get_live_snapshot():
    return compute_snapshot()


@st.cache_data(ttl=120, show_spinner=False)
def load_history():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return df
    df["date_ist"] = pd.to_datetime(df["date_ist"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Styling & Formatting Helpers
# ---------------------------------------------------------------------------


def color_change(val):
    if pd.isna(val) or val is None:
        return ""
    try:
        val_float = float(val)
        color = "#16a34a" if val_float >= 0 else "#dc2626"
        return f"color: {color}; font-weight: 600;"
    except Exception:
        return ""


def status_color(val):
    if val == "Advance":
        return "color: #16a34a; font-weight: 600;"
    if val == "Decline":
        return "color: #dc2626; font-weight: 600;"
    return "color: #9ca3af;"


def verdict_color(val):
    if val == "GREEN":
        return "background-color: rgba(22, 163, 74, 0.15); color: #22c55e; font-weight: 700;"
    if val == "RED":
        return "background-color: rgba(220, 38, 38, 0.15); color: #ef4444; font-weight: 700;"
    if val == "FLAT":
        return "background-color: rgba(217, 119, 6, 0.15); color: #f59e0b; font-weight: 700;"
    return "color: #9ca3af;"


def styled_table(df, extra_cols=None):
    fmt = {}
    for col in df.columns:
        if "Last" in col or "Open" in col or "High" in col or "Low" in col or "Price" in col or "Nifty 50" in col:
            fmt[col] = "{:,.2f}"
        elif "Change" in col or "%" in col or "Pct" in col:
            fmt[col] = "{:+.2f}%"

    styler = df.style.format(fmt, na_rep="—")
    apply_fn = "map" if hasattr(styler, "map") else "applymap"

    for col in df.columns:
        if "Change" in col or "%" in col or "Pct" in col:
            styler = getattr(styler, apply_fn)(color_change, subset=[col])
    if "Status" in df.columns:
        styler = getattr(styler, apply_fn)(status_color, subset=["Status"])
    if "verdict" in df.columns:
        styler = getattr(styler, apply_fn)(verdict_color, subset=["verdict"])
    if "Verdict" in df.columns:
        styler = getattr(styler, apply_fn)(verdict_color, subset=["Verdict"])

    return styler


def slot_sort_key(slot):
    try:
        h, m = map(int, str(slot).split(":"))
        return h * 60 + m
    except Exception:
        return 9999


# ---------------------------------------------------------------------------
# Custom High-Quality Plotly Financial Charts
# ---------------------------------------------------------------------------


def create_nifty_price_chart(time_series, price_series, open_val=None, title="Nifty 50 Intraday Movement"):
    """
    Renders a tightly scaled, glowing financial chart for Nifty 50 price action.
    """
    valid_data = [(t, p) for t, p in zip(time_series, price_series) if pd.notna(p) and p > 0]
    if not valid_data:
        return None

    times = [x[0] for x in valid_data]
    prices = [float(x[1]) for x in valid_data]

    p_min, p_max = min(prices), max(prices)
    span = max(p_max - p_min, 15.0)
    y_min = p_min - (span * 0.20)
    y_max = p_max + (span * 0.20)

    # Determine direction
    is_up = prices[-1] >= prices[0]
    line_color = "#10b981" if is_up else "#ef4444"
    fill_color = "rgba(16, 185, 129, 0.12)" if is_up else "rgba(239, 68, 68, 0.12)"

    fig = go.Figure()

    # Intraday Price Line + Area
    fig.add_trace(
        go.Scatter(
            x=times,
            y=prices,
            mode="lines+markers",
            name="Nifty 50",
            line=dict(color=line_color, width=3, shape="spline"),
            marker=dict(size=6, color=line_color, line=dict(color="#ffffff", width=1.5)),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate="<b>Time:</b> %{x}<br><b>Price:</b> ₹%{y:,.2f}<extra></extra>",
        )
    )

    # Add Day Open Baseline if available
    if open_val is not None and pd.notna(open_val) and open_val > 0:
        fig.add_hline(
            y=open_val,
            line_dash="dash",
            line_color="rgba(156, 163, 175, 0.6)",
            line_width=1.5,
            annotation_text=f"Open: {open_val:,.2f}",
            annotation_position="bottom right",
            annotation_font_color="#9ca3af",
            annotation_font_size=11,
        )

    # Day High & Day Low Annotation Markers
    max_idx = prices.index(p_max)
    min_idx = prices.index(p_min)

    fig.add_annotation(
        x=times[max_idx],
        y=p_max,
        text=f"High {p_max:,.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor="#10b981",
        arrowwidth=1.5,
        ax=0,
        ay=-25,
        font=dict(size=10, color="#10b981", family="sans-serif"),
        bgcolor="rgba(16, 185, 129, 0.15)",
        bordercolor="#10b981",
        borderwidth=1,
    )

    if min_idx != max_idx:
        fig.add_annotation(
            x=times[min_idx],
            y=p_min,
            text=f"Low {p_min:,.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="#ef4444",
            arrowwidth=1.5,
            ax=0,
            ay=25,
            font=dict(size=10, color="#ef4444", family="sans-serif"),
            bgcolor="rgba(239, 68, 68, 0.15)",
            bordercolor="#ef4444",
            borderwidth=1,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#f3f4f6")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        height=340,
        margin=dict(l=40, r=40, t=50, b=30),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(51, 65, 85, 0.4)",
            tickangle=0,
            tickfont=dict(size=11, color="#cbd5e1"),
            type="category",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(51, 65, 85, 0.4)",
            range=[y_min, y_max],
            tickformat=",.2f",
            tickfont=dict(size=11, color="#cbd5e1"),
            side="right",
        ),
        showlegend=False,
    )

    return fig


def create_score_bar_chart(times, scores, verdicts=None, title="⚡ Signal Score Evolution"):
    """
    Renders clean color-coded bars for pre-market & intraday signal scores.
    """
    valid_data = [(t, s, v) for t, s, v in zip(times, scores, verdicts if verdicts else [None] * len(times)) if pd.notna(s)]
    if not valid_data:
        return None

    t_list = [x[0] for x in valid_data]
    s_list = [float(x[1]) for x in valid_data]
    v_list = [x[2] for x in valid_data]

    bar_colors = []
    for s in s_list:
        if s > 0:
            bar_colors.append("#22c55e")
        elif s < 0:
            bar_colors.append("#ef4444")
        else:
            bar_colors.append("#f59e0b")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=t_list,
            y=s_list,
            marker_color=bar_colors,
            marker_line=dict(width=1, color="rgba(255,255,255,0.2)"),
            text=[f"{s:+.0f}" if s != 0 else "0" for s in s_list],
            textposition="outside",
            textfont=dict(size=11, color="#f8fafc"),
            hovertemplate="<b>Time:</b> %{x}<br><b>Score:</b> %{y:+d}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_width=1.5, line_color="#64748b")

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#f3f4f6")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        height=340,
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(
            showgrid=False,
            tickangle=0,
            tickfont=dict(size=11, color="#cbd5e1"),
            type="category",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(51, 65, 85, 0.4)",
            range=[-4.5, 4.5],
            dtick=1,
            tickfont=dict(size=11, color="#cbd5e1"),
            zeroline=True,
            zerolinecolor="#94a3b8",
        ),
        showlegend=False,
    )

    return fig


def create_breadth_bar_chart(advances, declines, unchanged):
    """
    Renders horizontal stacked breakdown for Market Breadth.
    """
    total = advances + declines + unchanged
    if total == 0:
        return None

    adv_pct = (advances / total) * 100
    dec_pct = (declines / total) * 100
    unc_pct = (unchanged / total) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=["Breadth"],
            x=[advances],
            name=f"Advances ({advances})",
            orientation="h",
            marker=dict(color="#22c55e"),
            text=f"🟢 {advances} ({adv_pct:.0f}%)",
            textposition="inside",
            insidetextanchor="middle",
        )
    )
    if unchanged > 0:
        fig.add_trace(
            go.Bar(
                y=["Breadth"],
                x=[unchanged],
                name=f"Unchanged ({unchanged})",
                orientation="h",
                marker=dict(color="#94a3b8"),
                text=f"⚪ {unchanged}",
                textposition="inside",
                insidetextanchor="middle",
            )
        )
    fig.add_trace(
        go.Bar(
            y=["Breadth"],
            x=[declines],
            name=f"Declines ({declines})",
            orientation="h",
            marker=dict(color="#ef4444"),
            text=f"🔴 {declines} ({dec_pct:.0f}%)",
            textposition="inside",
            insidetextanchor="middle",
        )
    )

    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=70,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, total]),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Header & Refresh
# ---------------------------------------------------------------------------

st.title("📈 Indian Market Pre-Open & Intraday Dashboard")
st.caption(f"📅 Current IST Time: {datetime.now().strftime('%A, %d %B %Y — %H:%M:%S')}")

col_refresh, col_badge = st.columns([1, 4])
with col_refresh:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

tab_live, tab_history = st.tabs(["🔴 Live Dashboard", "📊 History & Score Analysis"])

# ===========================================================================
# TAB A: LIVE DASHBOARD
# ===========================================================================
with tab_live:
    with st.spinner("Fetching live market data & index movement..."):
        snap = get_live_snapshot()

    fetched_time = snap.get("fetched_at_ist", datetime.now().strftime("%d %b %Y — %H:%M:%S IST"))
    us_date = snap.get("us_session_date", "Latest Close")
    sgx_date = snap.get("sgx_session_date", "Latest Session")
    nifty_date = snap.get("nifty_session_date", "Today")
    vix_date = snap.get("vix_session_date", "Latest")

    # --- Data Freshness & Source Dates Info Box ---
    st.info(
        f"🕒 **Last Data Load:** {fetched_time} &nbsp;|&nbsp; "
        f"**US Session:** `{us_date}` &nbsp;|&nbsp; "
        f"**SGX Session:** `{sgx_date}` &nbsp;|&nbsp; "
        f"**Nifty 50 Session:** `{nifty_date}`"
    )

    # --- Pre-Market Verdict Banner ---
    score = snap.get("score")
    n_signals = snap.get("n_signals", 0)
    verdict = snap.get("verdict", "NO_DATA")

    if score is not None:
        if verdict == "GREEN":
            st.success(f"### 🟢 LIKELY GREEN OPEN — Signal Score: {score}/{n_signals}")
        elif verdict == "RED":
            st.error(f"### 🔴 LIKELY RED OPEN — Signal Score: {score}/{n_signals}")
        else:
            st.warning(f"### 🟡 MIXED / FLAT OPEN — Signal Score: {score}/{n_signals}")
    else:
        st.warning("Insufficient data for a verdict")

    st.markdown("---")

    # --- Row 1: Global Cues & Volatility ---
    st.subheader(f"1. Global Cues & Volatility Regime (US Close: {us_date} • SGX: {sgx_date})")
    g1, g2, g3, g4 = st.columns(4)

    nasdaq_last = snap.get("nasdaq_last")
    nasdaq_chg = snap.get("nasdaq_change_pct")
    with g1:
        v_str = f"{nasdaq_last:,.2f}" if nasdaq_last is not None else "N/A"
        st.metric(label=f"🇺🇸 Nasdaq ({us_date})", value=v_str, delta=f"{nasdaq_chg:+.2f}%" if nasdaq_chg is not None else None)

    dow_last = snap.get("dow_last")
    dow_chg = snap.get("dow_change_pct")
    with g2:
        v_str = f"{dow_last:,.2f}" if dow_last is not None else "N/A"
        st.metric(label=f"🇺🇸 Dow Jones ({us_date})", value=v_str, delta=f"{dow_chg:+.2f}%" if dow_chg is not None else None)

    sgx_last = snap.get("sgx_last")
    sgx_chg = snap.get("sgx_change_pct")
    with g3:
        v_str = f"{sgx_last:,.2f}" if sgx_last is not None else "N/A"
        st.metric(label=f"🇸🇬 SGX STI ({sgx_date})", value=v_str, delta=f"{sgx_chg:+.2f}%" if sgx_chg is not None else None)

    vix = snap.get("india_vix")
    with g4:
        if vix is not None:
            vix_cond = "VOLATILE ⚠️" if vix > VIX_THRESHOLD else "STABLE ✅"
            st.metric(label=f"🇮🇳 India VIX ({vix_date})", value=f"{vix:.2f}", delta=vix_cond, delta_color="inverse" if vix > VIX_THRESHOLD else "normal")
        else:
            st.metric(label="🇮🇳 India VIX", value="N/A")

    st.markdown("---")

    # --- Row 2: Nifty 50 Real-Time & Intraday Movement (09:15 - 15:30) ---
    st.subheader(f"2. Nifty 50 Movement (Session: {nifty_date} • 09:15 AM — 03:30 PM IST)")

    nifty_last = snap.get("nifty_last")
    nifty_open = snap.get("nifty_open")
    nifty_high = snap.get("nifty_high")
    nifty_low = snap.get("nifty_low")
    nifty_chg = snap.get("nifty_change_pct")

    n_col1, n_col2, n_col3, n_col4, n_col5 = st.columns(5)
    with n_col1:
        st.metric("Nifty 50 Current / Close", f"{nifty_last:,.2f}" if nifty_last else "N/A", f"{nifty_chg:+.2f}%" if nifty_chg else None)
    with n_col2:
        st.metric("Day Open", f"{nifty_open:,.2f}" if nifty_open else "N/A")
    with n_col3:
        st.metric("Day High", f"{nifty_high:,.2f}" if nifty_high else "N/A")
    with n_col4:
        st.metric("Day Low", f"{nifty_low:,.2f}" if nifty_low else "N/A")
    with n_col5:
        if nifty_high and nifty_low:
            st.metric("Intraday Range", f"{nifty_high - nifty_low:,.2f} pts")
        else:
            st.metric("Intraday Range", "—")

    # Intraday Plotly Chart
    nifty_intra = snap.get("_nifty_intraday")
    if nifty_intra is not None and not nifty_intra.empty:
        latest_date = nifty_intra["date"].max()
        today_bars = nifty_intra[nifty_intra["date"] == latest_date].copy()
        if not today_bars.empty:
            fig_live = create_nifty_price_chart(
                today_bars["time"].tolist(),
                today_bars["Close"].tolist(),
                open_val=nifty_open,
                title=f"📈 Nifty 50 Live Intraday Path ({latest_date.strftime('%d %b %Y')})",
            )
            if fig_live:
                st.plotly_chart(fig_live, use_container_width=True)

    st.markdown("---")

    # --- Row 3: Market Breadth ---
    st.subheader("3. Market Breadth — Official NSE India Live Scraper")
    st.caption("Live Advances & Declines extracted directly from [NSE India Official Advance/Decline Portal](https://www.nseindia.com/market-data/advance)")

    advances = snap.get("advances", 0)
    declines = snap.get("declines", 0)
    unchanged = snap.get("unchanged", 0)
    total_breadth = advances + declines + unchanged

    matrix = snap.get("breadth_matrix", {})
    n500_adv, n500_dec, n500_unc = matrix.get("n500", (0, 0, 0))
    mid_adv, mid_dec, mid_unc = matrix.get("midcap", (0, 0, 0))
    small_adv, small_dec, small_unc = matrix.get("smallcap", (0, 0, 0))
    all_adv, all_dec, all_unc = matrix.get("all_nse", (0, 0, 0))

    breadth_df = snap.get("_breadth_df")

    if total_breadth > 0 or (breadth_df is not None and not breadth_df.empty):
        # --- Macro Breadth Matrix ---
        if all_adv > 0 or all_dec > 0:
            st.markdown(
                f"#### 🌐 **All NSE Traded Equities:** &nbsp; "
                f":green[**Advance — {all_adv:,}**] &nbsp;&nbsp;|&nbsp;&nbsp; "
                f":red[**Decline — {all_dec:,}**] &nbsp;&nbsp;|&nbsp;&nbsp; "
                f":orange[**Unchanged — {all_unc:,}**]"
            )

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("🏛️ Nifty 50 (Bluechips)", f"{advances}A / {declines}D", delta=f"{advances - declines:+d} Net")
        with col_m2:
            st.metric("🏢 Nifty 500 (Broad)", f"{n500_adv}A / {n500_dec}D", delta=f"{n500_adv - n500_dec:+d} Net")
        with col_m3:
            st.metric("🏭 Midcap 150", f"{mid_adv}A / {mid_dec}D" if mid_adv else "N/A", delta=f"{mid_adv - mid_dec:+d} Net" if mid_adv else None)
        with col_m4:
            st.metric("🏬 Smallcap 250", f"{small_adv}A / {small_dec}D" if small_adv else "N/A", delta=f"{small_adv - small_dec:+d} Net" if small_adv else None)

        b_fig = create_breadth_bar_chart(advances, declines, unchanged)
        if b_fig:
            st.plotly_chart(b_fig, use_container_width=True)

        # --- Top Gainers & Losers ---
        if breadth_df is not None and not breadth_df.empty:
            gc1, gc2 = st.columns(2)
            with gc1:
                st.markdown("##### 🟢 **Top Gainers**")
                st.dataframe(
                    styled_table(breadth_df.sort_values("Change %", ascending=False).head(5)),
                    hide_index=True,
                    use_container_width=True,
                )
            with gc2:
                st.markdown("##### 🔴 **Top Losers**")
                st.dataframe(
                    styled_table(breadth_df.sort_values("Change %", ascending=True).head(5)),
                    hide_index=True,
                    use_container_width=True,
                )

            # --- Full Interactive 500-Stocks Table Explorer ---
            st.markdown("---")
            st.markdown(f"### 🔍 Full Advances & Declines Scraped Dataset ({len(breadth_df)} Stocks)")

            col_search, col_filter, col_dl = st.columns([2, 2, 1])
            with col_search:
                search_query = st.text_input("🔎 Search by Stock Symbol", "", placeholder="e.g. WELCORP, RELIANCE, HDFCBANK...")
            with col_filter:
                filter_status = st.selectbox("Filter Status", ["All", "🟢 Advances Only", "🔴 Declines Only", "⚪ Unchanged Only"])
            with col_dl:
                csv_bytes = breadth_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download (.csv)",
                    data=csv_bytes,
                    file_name=f"nse_advances_declines_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Apply filters
            filtered_df = breadth_df.copy()
            if search_query.strip():
                filtered_df = filtered_df[filtered_df["Symbol"].str.contains(search_query.strip().upper(), na=False)]
            if filter_status == "🟢 Advances Only":
                filtered_df = filtered_df[filtered_df["Status"] == "Advance"]
            elif filter_status == "🔴 Declines Only":
                filtered_df = filtered_df[filtered_df["Status"] == "Decline"]
            elif filter_status == "⚪ Unchanged Only":
                filtered_df = filtered_df[filtered_df["Status"] == "Unchanged"]

            st.dataframe(
                styled_table(filtered_df.sort_values("Change %", ascending=False)),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f"Showing {len(filtered_df)} of {len(breadth_df)} total scraped stocks.")
    else:
        st.warning("Market breadth data is currently fetching or unavailable.")

    st.caption("Disclaimer: For informational and market study purposes only. Not financial advice.")

# ===========================================================================
# TAB B: HISTORY & SCORE ANALYSIS
# ===========================================================================
with tab_history:
    st.subheader("📊 Historical Score & Nifty Movement Analysis")
    st.caption(
        "Analyze pre-market signal accuracy, 09:30–15:30 intraday hourly price trajectory, "
        "and market correlation over time."
    )

    hist = load_history()

    if hist.empty:
        st.info("No historical data found. Run `python backfill_history.py` to generate 1 year of historical scores and Nifty intraday data.")
    else:
        # Pre-process history
        hist["_slot_order"] = hist["slot_ist"].map(slot_sort_key)
        hist_sorted = hist.sort_values(["date_ist", "_slot_order"]).drop(columns="_slot_order")

        # -------------------------------------------------------------
        # 1. Pre-Market Signal Accuracy Scorecard
        # -------------------------------------------------------------
        st.markdown("### 🎯 Pre-Market Signal Accuracy Scorecard (09:00 Score vs Day's Nifty Return)")

        daily_0900 = hist_sorted[hist_sorted["slot_ist"] == "09:00"].copy()
        daily_1600 = hist_sorted[hist_sorted["slot_ist"] == "16:00"].copy()

        # Merge 0900 and 1600 on date_ist
        if not daily_0900.empty:
            merged_day = daily_0900.merge(
                daily_1600[["date_ist", "nifty_last", "nifty_change_pct"]],
                on="date_ist",
                suffixes=("_0900", "_1600"),
                how="left",
            )

            # Evaluate predictions
            def evaluate_prediction(row):
                v = row.get("verdict_0900")
                chg = row.get("nifty_change_pct_1600")
                if chg is None or pd.isna(chg):
                    chg = row.get("nifty_change_pct_0900")
                if chg is None or pd.isna(chg) or v == "NO_DATA" or pd.isna(v):
                    return "No Nifty Data"
                if v == "GREEN" and chg > 0:
                    return "✅ Bullish Hit"
                elif v == "RED" and chg < 0:
                    return "✅ Bearish Hit"
                elif v == "FLAT" and abs(chg) < 0.25:
                    return "✅ Neutral Hit"
                elif v == "GREEN" and chg <= 0:
                    return "❌ Bullish Miss"
                elif v == "RED" and chg >= 0:
                    return "❌ Bearish Miss"
                else:
                    return "⚪ Flat/Mixed"

            merged_day["Outcome"] = merged_day.apply(evaluate_prediction, axis=1)
            valid_outcomes = merged_day[~merged_day["Outcome"].isin(["No Nifty Data", "⚪ Flat/Mixed"])]

            if not valid_outcomes.empty:
                hits = valid_outcomes["Outcome"].str.startswith("✅").sum()
                total_evaluated = len(valid_outcomes)
                win_rate = (hits / total_evaluated * 100) if total_evaluated > 0 else 0

                green_days = merged_day[merged_day["verdict_0900"] == "GREEN"]
                avg_green_ret = green_days["nifty_change_pct_1600"].mean()

                red_days = merged_day[merged_day["verdict_0900"] == "RED"]
                avg_red_ret = red_days["nifty_change_pct_1600"].mean()

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Evaluated Trading Days", f"{total_evaluated} Days")
                kpi2.metric("Pre-Market Direction Win Rate", f"{win_rate:.1f}%", f"{hits} of {total_evaluated} days")
                kpi3.metric(
                    "Avg Nifty Return on GREEN Calls",
                    f"{avg_green_ret:+.2f}%" if pd.notna(avg_green_ret) else "—",
                    delta="Bullish Days",
                )
                kpi4.metric(
                    "Avg Nifty Return on RED Calls",
                    f"{avg_red_ret:+.2f}%" if pd.notna(avg_red_ret) else "—",
                    delta="Bearish Days",
                    delta_color="inverse",
                )

        st.markdown("---")

        # -------------------------------------------------------------
        # 2. Hourly Intraday Score & Nifty Movement Tracker (09:30 - 15:30)
        # -------------------------------------------------------------
        st.markdown("### ⏱️ Hourly Intraday Tracker (09:30 AM — 03:30 PM Grid)")
        st.caption("Select a trading day to view the hour-by-hour price evolution of Nifty 50 alongside the signal score.")

        available_dates = sorted(hist_sorted["date_ist"].unique(), reverse=True)
        pick_date = st.selectbox(
            "Select Trading Day to Inspect",
            options=available_dates,
            format_func=lambda d: d.strftime("%A, %d %b %Y"),
        )

        day_df = hist_sorted[hist_sorted["date_ist"] == pick_date].copy()
        day_df["_slot_order"] = day_df["slot_ist"].map(slot_sort_key)
        day_df = day_df.sort_values("_slot_order").drop(columns="_slot_order")

        if not day_df.empty:
            # Day high level summary metrics
            d_open_val = day_df["nifty_open"].dropna().iloc[0] if not day_df["nifty_open"].dropna().empty else None
            d_close_val = day_df["nifty_last"].dropna().iloc[-1] if not day_df["nifty_last"].dropna().empty else None
            d_high_val = day_df["nifty_high"].max() if not day_df["nifty_high"].dropna().empty else None
            d_low_val = day_df["nifty_low"].min() if not day_df["nifty_low"].dropna().empty else None
            d_chg_val = (
                day_df["nifty_change_pct"].dropna().iloc[-1] if not day_df["nifty_change_pct"].dropna().empty else None
            )

            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            with sm1:
                st.metric("Day Open", f"{d_open_val:,.2f}" if d_open_val else "—")
            with sm2:
                st.metric("Day Close", f"{d_close_val:,.2f}" if d_close_val else "—", f"{d_chg_val:+.2f}%" if d_chg_val else None)
            with sm3:
                st.metric("Day High", f"{d_high_val:,.2f}" if d_high_val else "—")
            with sm4:
                st.metric("Day Low", f"{d_low_val:,.2f}" if d_low_val else "—")
            with sm5:
                # Pre-market score for that day
                row_0900 = day_df[day_df["slot_ist"] == "09:00"]
                if not row_0900.empty:
                    sc = row_0900.iloc[0].get("score")
                    vd = row_0900.iloc[0].get("verdict")
                    st.metric("09:00 Verdict", f"{vd} ({sc:+.0f})" if pd.notna(sc) else f"{vd}")
                else:
                    st.metric("09:00 Verdict", "—")

            # High Quality Charts Row: Nifty Price Movement vs Score Evolution
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                nifty_valid = day_df.dropna(subset=["nifty_last"])
                if not nifty_valid.empty:
                    p_fig = create_nifty_price_chart(
                        nifty_valid["slot_ist"].tolist(),
                        nifty_valid["nifty_last"].tolist(),
                        open_val=d_open_val,
                        title=f"📈 Nifty 50 Intraday Movement ({pick_date.strftime('%d %b %Y')})",
                    )
                    if p_fig:
                        st.plotly_chart(p_fig, use_container_width=True)
                else:
                    st.caption("No intraday Nifty price points logged for this historical date.")

            with c_col2:
                score_valid = day_df.dropna(subset=["score"])
                if not score_valid.empty:
                    s_fig = create_score_bar_chart(
                        score_valid["slot_ist"].tolist(),
                        score_valid["score"].tolist(),
                        verdicts=score_valid["verdict"].tolist(),
                        title="⚡ Signal Score Evolution",
                    )
                    if s_fig:
                        st.plotly_chart(s_fig, use_container_width=True)
                else:
                    st.caption("No score points logged.")

            # Hourly Data Table
            st.markdown("📋 **Hourly Intraday Breakdown Table (09:00 to 16:00)**")
            display_cols = [
                "slot_ist", "nifty_last", "nifty_change_pct", "score", "verdict",
                "advances", "declines", "india_vix", "sgx_change_pct", "nasdaq_change_pct", "dow_change_pct",
            ]
            valid_cols = [c for c in display_cols if c in day_df.columns]
            table_show = day_df[valid_cols].rename(
                columns={
                    "slot_ist": "Time (IST)",
                    "nifty_last": "Nifty 50",
                    "nifty_change_pct": "Nifty Change %",
                    "score": "Score",
                    "verdict": "Verdict",
                    "advances": "Advances",
                    "declines": "Declines",
                    "india_vix": "India VIX",
                    "sgx_change_pct": "SGX %",
                    "nasdaq_change_pct": "Nasdaq %",
                    "dow_change_pct": "Dow %",
                }
            )
            st.dataframe(styled_table(table_show), hide_index=True, use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # 3. Date-wise Official 09:00 vs 16:00 Scores Table
        # -------------------------------------------------------------
        st.markdown("### 📅 Date-wise Official Scores (09:00 Open vs 16:00 Close)")
        official = hist_sorted[hist_sorted["slot_ist"].isin(PRIMARY_SLOTS)].copy()

        if not official.empty:
            pivot = official.pivot_table(
                index="date_ist",
                columns="slot_ist",
                values=["score", "verdict", "nifty_last", "nifty_change_pct"],
                aggfunc="first",
            )
            pivot.columns = [f"{slot} {metric}" for metric, slot in pivot.columns]
            pivot = pivot.reset_index().sort_values("date_ist", ascending=False)

            if "backfilled" in official.columns:
                bf_count = official["backfilled"].astype(str).str.lower().isin(["true", "1"]).sum()
                if bf_count > 0:
                    st.caption(f"ℹ️ {bf_count} snapshots backfilled from yfinance data. Live recordings automatically update with market breadth.")

            st.dataframe(styled_table(pivot), hide_index=True, use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # 4. Correlation: 09:00-today vs 16:00-previous-day
        # -------------------------------------------------------------
        st.markdown("### 🔁 Correlation: 09:00 (Today) vs 16:00 (Previous Day)")
        st.caption("Did the overnight global cues agree with where the Indian market had settled the evening before?")

        am = official[official["slot_ist"] == "09:00"][["date_ist", "score", "verdict"]].rename(
            columns={"score": "score_0900", "verdict": "verdict_0900"}
        )
        pm = official[official["slot_ist"] == "16:00"][["date_ist", "score", "verdict"]].rename(
            columns={"score": "score_1600", "verdict": "verdict_1600"}
        )

        if not am.empty and not pm.empty:
            pm_shifted = pm.sort_values("date_ist").copy()
            all_dates_set = sorted(set(am["date_ist"]) | set(pm_shifted["date_ist"]))
            pm_shifted["next_date"] = pm_shifted["date_ist"].apply(
                lambda d: next((x for x in all_dates_set if x > d), None)
            )
            merged_corr = pm_shifted.merge(
                am, left_on="next_date", right_on="date_ist", suffixes=("_prevclose", "_today")
            )
            merged_corr = merged_corr.rename(columns={"date_ist_prevclose": "prev_close_date", "date_ist_today": "next_open_date"})
            merged_corr["agree"] = merged_corr["verdict_1600"] == merged_corr["verdict_0900"]

            if not merged_corr.empty:
                agree_pct = 100 * merged_corr["agree"].mean()
                st.metric("Agreement Rate (Prior 16:00 vs Next 09:00)", f"{agree_pct:.1f}%")

                out = merged_corr[[
                    "prev_close_date", "verdict_1600", "score_1600",
                    "next_open_date", "verdict_0900", "score_0900", "agree",
                ]].sort_values("next_open_date", ascending=False)
                out = out.rename(columns={
                    "prev_close_date": "Prior Day Date",
                    "verdict_1600": "16:00 Verdict",
                    "score_1600": "16:00 Score",
                    "next_open_date": "Next Day Date",
                    "verdict_0900": "09:00 Verdict",
                    "score_0900": "09:00 Score",
                    "agree": "Agreed?",
                })
                st.dataframe(out, hide_index=True, use_container_width=True)

        with st.expander("🔍 Full Raw CSV Log"):
            st.dataframe(hist_sorted, hide_index=True, use_container_width=True)

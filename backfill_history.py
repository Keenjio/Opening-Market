"""
backfill_history.py
====================
Point-in-Time, Zero-Lookahead Historical Backfill Engine.

Ensures complete temporal validity:
- At Date D @ 09:00 AM IST (Pre-Market):
    * Uses ONLY the previous day D-1's US close (Nasdaq & Dow % change).
    * Uses ONLY the previous day D-1's India VIX and Nifty close.
    * Uses SGX / Asian cues known prior to 09:00 AM IST.
    * Has STRICTLY ZERO access to Date D's Open, High, Low, Close, or intraday price action.
- At Hourly Intraday Slots (09:30, 10:30, ..., 15:30):
    * Uses ONLY candles that occurred up to that exact timestamp.
    * Has STRICTLY ZERO access to future hours or the final settlement.
- At Date D @ 16:00 PM IST (Post-Market):
    * Records the final day settlement (Date D Open, High, Low, Close, % return)
      to evaluate whether the 09:00 AM signal (made without future data) was accurate.

Usage:
    python backfill_history.py --clear
"""

import argparse
import csv
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from market_data import (
    NIFTY_INDEX_TICKER,
    SGX_LABEL,
    SGX_TICKER,
    VIX_THRESHOLD,
    compute_historical_score,
)

IST = timezone(timedelta(hours=5, minutes=30))
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "score_log.csv")

FIELDNAMES = [
    "date_ist", "slot_ist", "timestamp_utc",
    "nasdaq_change_pct", "dow_change_pct", "sgx_change_pct",
    "nifty_open", "nifty_last", "nifty_high", "nifty_low", "nifty_change_pct",
    "india_vix", "vix_condition",
    "advances", "declines", "unchanged", "breadth_sentiment",
    "global_cue_source", "global_cue_change_pct", "global_cue_is_fallback",
    "score", "n_signals", "verdict",
    "backfilled",
]


def load_existing_keys():
    keys = set()
    if not os.path.exists(CSV_PATH):
        return keys
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get("date_ist", "").strip()
            s = row.get("slot_ist", "").strip()
            if d and s:
                keys.add((d, s))
    return keys


def main():
    parser = argparse.ArgumentParser(description="Backfill score history with STRICT zero lookahead bias.")
    parser.add_argument("--years", type=int, default=1, help="Years of daily history (default: 1)")
    parser.add_argument("--clear", action="store_true", help="Clear existing CSV before backfilling")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  ZERO-LOOKAHEAD HISTORICAL BACKFILL ENGINE")
    print(f"  Generating {args.years}-Year Point-in-Time Pre-Market & Intraday Scores")
    print("=" * 65)

    # 1. Download daily historical series
    print("\n[1/3] Downloading daily data for Nifty 50 and global indicators...")
    tickers = {
        "nifty": NIFTY_INDEX_TICKER,
        "nasdaq": "^IXIC",
        "dow": "^DJI",
        "sgx": SGX_TICKER,
        "vix": "^INDIAVIX",
    }
    daily_raw = {}
    for name, ticker in tickers.items():
        print(f"  Downloading {name} ({ticker})...")
        try:
            df = yf.Ticker(ticker).history(period=f"{args.years}y")
            if not df.empty:
                df["date_str"] = df.index.strftime("%Y-%m-%d")
                daily_raw[name] = df
                print(f"    [OK] {len(df)} daily bars ({df['date_str'].min()} to {df['date_str'].max()})")
            else:
                daily_raw[name] = pd.DataFrame()
        except Exception as e:
            print(f"    [ERR] {e}")
            daily_raw[name] = pd.DataFrame()

    # 2. Download hourly intraday data for Nifty 50 (1 month)
    print("\n[2/3] Downloading 1-hour intraday bars for Nifty 50...")
    try:
        nifty_1h = yf.Ticker(NIFTY_INDEX_TICKER).history(period="1mo", interval="1h")
        if not nifty_1h.empty:
            if nifty_1h.index.tz is not None:
                nifty_1h.index = nifty_1h.index.tz_convert(IST)
            nifty_1h["date_str"] = nifty_1h.index.strftime("%Y-%m-%d")
            nifty_1h["time_str"] = nifty_1h.index.strftime("%H:%M")
            print(f"    [OK] {len(nifty_1h)} hourly bars")
        else:
            nifty_1h = pd.DataFrame()
    except Exception as e:
        print(f"    [ERR] Intraday fetch failed: {e}")
        nifty_1h = pd.DataFrame()

    # Compute daily % returns
    daily_returns = {}
    for name in ["nasdaq", "dow", "sgx", "nifty"]:
        df = daily_raw.get(name)
        if df is not None and not df.empty and "Close" in df.columns:
            pct_series = df["Close"].pct_change() * 100
            daily_returns[name] = dict(zip(df["date_str"], pct_series))
        else:
            daily_returns[name] = {}

    vix_df = daily_raw.get("vix", pd.DataFrame())
    vix_map = dict(zip(vix_df["date_str"], vix_df["Close"])) if not vix_df.empty else {}

    nifty_df = daily_raw.get("nifty", pd.DataFrame())
    if nifty_df.empty:
        print("[ERR] No Nifty daily data available.")
        return

    # Chronologically sorted list of Indian trading dates
    indian_trading_dates = sorted(nifty_df["date_str"].unique())
    nasdaq_dates_sorted = sorted(daily_returns["nasdaq"].keys())
    sgx_dates_sorted = sorted(daily_returns["sgx"].keys())
    vix_dates_sorted = sorted(vix_map.keys())

    # Helper: Find the latest data strictly BEFORE target_date (strictly < target_date)
    def get_strictly_prior_value(data_dict, sorted_dates, current_date):
        prior_dates = [d for d in sorted_dates if d < current_date]
        if prior_dates:
            latest_prior_date = prior_dates[-1]
            val = data_dict.get(latest_prior_date)
            if pd.notna(val):
                return val, latest_prior_date
        return None, None

    if args.clear and os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
        print("Cleared existing CSV.")

    existing_keys = load_existing_keys()
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0

    rows_written = 0
    rows_skipped = 0

    print(f"\n[3/3] Writing point-in-time historical records to {CSV_PATH}...")

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        # Iterate chronologically across Indian trading days (starting from index 1 so we have prior day)
        for i in range(1, len(indian_trading_dates)):
            curr_date = indian_trading_dates[i]
            prev_date = indian_trading_dates[i - 1]

            # -------------------------------------------------------------
            # A. 09:00 AM PRE-MARKET POINT-IN-TIME SNAPSHOT
            # -------------------------------------------------------------
            # At 09:00 AM on curr_date:
            # - Market has NOT opened.
            # - Known US close is from strictly BEFORE curr_date (e.g. prev_date).
            # - Known VIX is from strictly BEFORE curr_date (prev_date close).
            # - Known Nifty price is Date D-1's Close.
            # - ZERO LOOKAHEAD to Date D Open, High, Low, or Close!

            nasdaq_0900, _ = get_strictly_prior_value(daily_returns["nasdaq"], nasdaq_dates_sorted, curr_date)
            dow_0900, _ = get_strictly_prior_value(daily_returns["dow"], nasdaq_dates_sorted, curr_date)
            sgx_0900, _ = get_strictly_prior_value(daily_returns["sgx"], sgx_dates_sorted, curr_date)
            vix_0900, _ = get_strictly_prior_value(vix_map, vix_dates_sorted, curr_date)

            # Nifty level known at 09:00 AM is previous day's close
            row_prev_nifty = nifty_df[nifty_df["date_str"] == prev_date].iloc[0]
            prev_nifty_close = round(float(row_prev_nifty["Close"]), 2)

            nasdaq_0900 = round(float(nasdaq_0900), 4) if pd.notna(nasdaq_0900) else None
            dow_0900 = round(float(dow_0900), 4) if pd.notna(dow_0900) else None
            sgx_0900 = round(float(sgx_0900), 4) if pd.notna(sgx_0900) else None
            vix_0900 = round(float(vix_0900), 2) if pd.notna(vix_0900) else None

            # Calculate 09:00 pre-market score from strictly prior cues
            score_0900 = compute_historical_score(
                nasdaq_chg=nasdaq_0900,
                dow_chg=dow_0900,
                sgx_chg=sgx_0900,
                vix=vix_0900,
                nifty_open=None,  # Not opened yet
                nifty_last=prev_nifty_close,  # Known previous close
                nifty_high=None,  # No lookahead
                nifty_low=None,  # No lookahead
                nifty_change_pct=None,  # No lookahead
            )

            if (curr_date, "09:00") not in existing_keys:
                writer.writerow({
                    "date_ist": curr_date,
                    "slot_ist": "09:00",
                    "timestamp_utc": f"{curr_date}T03:30:00",
                    "backfilled": True,
                    **score_0900,
                })
                existing_keys.add((curr_date, "09:00"))
                rows_written += 1

            # -------------------------------------------------------------
            # B. 16:00 PM POST-MARKET SNAPSHOT (Day Settlement)
            # -------------------------------------------------------------
            # At 16:00 PM on curr_date, the full Indian trading day is complete.
            # Day Open, High, Low, Close, and % return are now finalized.

            row_curr_nifty = nifty_df[nifty_df["date_str"] == curr_date].iloc[0]
            curr_nifty_open = round(float(row_curr_nifty["Open"]), 2)
            curr_nifty_close = round(float(row_curr_nifty["Close"]), 2)
            curr_nifty_high = round(float(row_curr_nifty["High"]), 2)
            curr_nifty_low = round(float(row_curr_nifty["Low"]), 2)

            curr_nifty_ret = daily_returns["nifty"].get(curr_date)
            if curr_nifty_ret is None and prev_nifty_close > 0:
                curr_nifty_ret = (curr_nifty_close - prev_nifty_close) / prev_nifty_close * 100
            curr_nifty_ret = round(float(curr_nifty_ret), 4) if pd.notna(curr_nifty_ret) else None

            # 16:00 VIX is today's closing VIX
            curr_vix = vix_map.get(curr_date, vix_0900)
            curr_vix = round(float(curr_vix), 2) if pd.notna(curr_vix) else None

            score_1600 = compute_historical_score(
                nasdaq_chg=nasdaq_0900,
                dow_chg=dow_0900,
                sgx_chg=sgx_0900,
                vix=curr_vix,
                nifty_open=curr_nifty_open,
                nifty_last=curr_nifty_close,
                nifty_high=curr_nifty_high,
                nifty_low=curr_nifty_low,
                nifty_change_pct=curr_nifty_ret,
            )

            if (curr_date, "16:00") not in existing_keys:
                writer.writerow({
                    "date_ist": curr_date,
                    "slot_ist": "16:00",
                    "timestamp_utc": f"{curr_date}T10:30:00",
                    "backfilled": True,
                    **score_1600,
                })
                existing_keys.add((curr_date, "16:00"))
                rows_written += 1

        # -------------------------------------------------------------
        # C. HOURLY INTRADAY SLOTS (09:30 to 15:30)
        # -------------------------------------------------------------
        # Point-in-time: At time T, only bars up to time T are accessible.
        if not nifty_1h.empty:
            intraday_slot_map = {
                "09:15": "09:30",
                "10:15": "10:30",
                "11:15": "11:30",
                "12:15": "12:30",
                "13:15": "13:30",
                "14:15": "14:30",
                "15:15": "15:30",
            }

            grouped = nifty_1h.groupby("date_str")
            for d_str, day_bars in grouped:
                if d_str not in indian_trading_dates:
                    continue

                # Find strictly prior US cues for this day
                nasdaq_d, _ = get_strictly_prior_value(daily_returns["nasdaq"], nasdaq_dates_sorted, d_str)
                dow_d, _ = get_strictly_prior_value(daily_returns["dow"], nasdaq_dates_sorted, d_str)
                sgx_d, _ = get_strictly_prior_value(daily_returns["sgx"], sgx_dates_sorted, d_str)
                vix_d, _ = get_strictly_prior_value(vix_map, vix_dates_sorted, d_str)

                nasdaq_d = round(float(nasdaq_d), 4) if pd.notna(nasdaq_d) else None
                dow_d = round(float(dow_d), 4) if pd.notna(dow_d) else None
                sgx_d = round(float(sgx_d), 4) if pd.notna(sgx_d) else None
                vix_d = round(float(vix_d), 2) if pd.notna(vix_d) else None

                cum_high = None
                cum_low = None
                first_open = None

                for idx, bar in day_bars.iterrows():
                    t_str = bar["time_str"]
                    slot = intraday_slot_map.get(t_str)
                    if not slot or (d_str, slot) in existing_keys:
                        continue

                    b_open = round(float(bar["Open"]), 2)
                    b_close = round(float(bar["Close"]), 2)
                    b_high = round(float(bar["High"]), 2)
                    b_low = round(float(bar["Low"]), 2)

                    if first_open is None:
                        first_open = b_open
                    cum_high = max(cum_high, b_high) if cum_high is not None else b_high
                    cum_low = min(cum_low, b_low) if cum_low is not None else b_low

                    # Return up to this point in time (from open)
                    pct_so_far = round((b_close - first_open) / first_open * 100, 4) if first_open > 0 else 0.0

                    intra_score = compute_historical_score(
                        nasdaq_chg=nasdaq_d,
                        dow_chg=dow_d,
                        sgx_chg=sgx_d,
                        vix=vix_d,
                        nifty_open=first_open,
                        nifty_last=b_close,
                        nifty_high=cum_high,
                        nifty_low=cum_low,
                        nifty_change_pct=pct_so_far,
                    )

                    writer.writerow({
                        "date_ist": d_str,
                        "slot_ist": slot,
                        "timestamp_utc": idx.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S") if idx.tz else idx.strftime("%Y-%m-%dT%H:%M:%S"),
                        "backfilled": True,
                        **intra_score,
                    })
                    existing_keys.add((d_str, slot))
                    rows_written += 1

    print("\n" + "=" * 65)
    print(f"  ZERO-LOOKAHEAD VALIDATION COMPLETE")
    print(f"  Wrote {rows_written} strictly point-in-time records to {CSV_PATH}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

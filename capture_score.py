"""
capture_score.py
=================
Run headlessly (no Streamlit) by a GitHub Actions cron job at fixed IST
times: 09:00, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 15:30, 16:00.

Each run:
  1. Computes a live snapshot using market_data.compute_snapshot()
     (the SAME logic the dashboard uses).
  2. Appends one row to data/score_log.csv, tagged with the IST date
     and the nearest scheduled slot.
  3. Skips writing a duplicate if a row for the same (date, slot)
     already exists (idempotent — safe to re-run / retry).

The workflow that calls this script is responsible for git-committing
the updated CSV back to the repo (see .github/workflows/capture.yml).

Usage:
    python capture_score.py                # auto-detect nearest slot from current IST time
    python capture_score.py --slot 09:00    # force a specific slot (useful for backfill/testing)
"""

import argparse
import csv
import os
from datetime import datetime, timedelta, timezone

from market_data import CAPTURE_SLOTS, compute_snapshot

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


def nearest_slot(now_ist):
    """Given a datetime in IST, find the closest scheduled slot (HH:MM)."""
    target_minutes = now_ist.hour * 60 + now_ist.minute
    best_slot, best_diff = None, None
    for slot in CAPTURE_SLOTS:
        h, m = map(int, slot.split(":"))
        diff = abs((h * 60 + m) - target_minutes)
        if best_diff is None or diff < best_diff:
            best_slot, best_diff = slot, diff
    return best_slot


def row_exists(date_ist, slot_ist):
    if not os.path.exists(CSV_PATH):
        return False
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("date_ist") == date_ist and row.get("slot_ist") == slot_ist:
                return True
    return False


def append_row(row):
    file_exists = os.path.exists(CSV_PATH)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slot", default=None,
        help="Force a specific HH:MM IST slot instead of auto-detecting "
             "from current time (useful for manual backfill/testing).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Write even if a row for this (date, slot) already exists "
             "(overwrites via full-file rewrite).",
    )
    args = parser.parse_args()

    now_ist = datetime.now(IST)
    date_ist = now_ist.strftime("%Y-%m-%d")
    slot_ist = args.slot if args.slot else nearest_slot(now_ist)

    if slot_ist not in CAPTURE_SLOTS:
        raise SystemExit(f"Invalid slot '{slot_ist}'. Must be one of {CAPTURE_SLOTS}")

    if row_exists(date_ist, slot_ist) and not args.force:
        print(f"Row for {date_ist} {slot_ist} already exists — skipping (use --force to overwrite).")
        return

    if row_exists(date_ist, slot_ist) and args.force:
        # rewrite file without the old row, then append fresh below
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        remaining = [r for r in reader if not (r["date_ist"] == date_ist and r["slot_ist"] == slot_ist)]
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(remaining)

    snap = compute_snapshot()
    row = {
        "date_ist": date_ist,
        "slot_ist": slot_ist,
        "timestamp_utc": snap["timestamp_utc"],
        "nasdaq_change_pct": snap["nasdaq_change_pct"],
        "dow_change_pct": snap["dow_change_pct"],
        "sgx_change_pct": snap["sgx_change_pct"],
        "nifty_open": snap.get("nifty_open"),
        "nifty_last": snap.get("nifty_last"),
        "nifty_high": snap.get("nifty_high"),
        "nifty_low": snap.get("nifty_low"),
        "nifty_change_pct": snap.get("nifty_change_pct"),
        "india_vix": snap["india_vix"],
        "vix_condition": snap["vix_condition"],
        "advances": snap["advances"],
        "declines": snap["declines"],
        "unchanged": snap["unchanged"],
        "breadth_sentiment": snap["breadth_sentiment"],
        "global_cue_source": snap["global_cue_source"],
        "global_cue_change_pct": snap["global_cue_change_pct"],
        "global_cue_is_fallback": snap["global_cue_is_fallback"],
        "score": snap["score"],
        "n_signals": snap["n_signals"],
        "verdict": snap["verdict"],
        "backfilled": False,
    }
    append_row(row)
    print(f"Logged {date_ist} {slot_ist} -> verdict={row['verdict']} score={row['score']}")


if __name__ == "__main__":
    main()

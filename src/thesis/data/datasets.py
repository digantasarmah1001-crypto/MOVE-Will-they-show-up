from pathlib import Path

import numpy as np
import pandas as pd

from thesis.data.features import add_history

root = Path(__file__).resolve().parents[3]


def noshow_data():
    df = add_history(pd.read_parquet(root / "data/processed/noshow.parquet"))
    X = df[["age", "scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received",
            "prior_visits", "prior_no_shows", "w_visits", "w_shows"]].assign(
        log_lead=np.log1p(df.lead_days), female=(df.gender == "F").astype(int),
        log_days_since_last=np.log1p(df.days_since_last.fillna(0)),
        weekday=df.appointment_day.dt.day_name().astype("category"), neighbourhood=df.neighbourhood.astype("category"))
    strata = pd.cut(df.prior_visits, [-1, 0, 2, np.inf], labels=["0", "1-2", "3+"])
    # MOVE: weighted shows and visits, no late cancels in this data
    return df, X, 1 - df.no_show, df.w_shows, df.w_visits, df.patient_id, strata, "appointment_day"


def hotel_data():
    df = pd.read_parquet(root / "data/processed/hotel.parquet")
    cats = ["hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
            "reserved_room_type", "deposit_type", "agent", "company", "customer_type"]
    X = df[["stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "group_size",
            "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled", "adr"]].assign(
        log_lead=np.log1p(df.lead_time), **{c: df[c].astype("category") for c in cats})
    prev = df.previous_cancellations + df.previous_bookings_not_canceled
    strata = pd.cut(prev, [-1, 0, 2, np.inf], labels=["0", "1-2", "3+"])
    df["date"] = df.arrival_date
    return df, X, 1 - df.is_canceled, df.previous_bookings_not_canceled, prev, df.index, strata, "date"

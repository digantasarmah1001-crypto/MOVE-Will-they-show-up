import numpy as np
import pandas as pd

SPLIT_DATE = "2016-05-25"


def add_history(df, half_life=30):
    # history only uses the same patient's visits on earlier days
    pairs = df[["appointment_id", "patient_id", "appointment_day"]].merge(
        df[["patient_id", "appointment_day", "no_show"]], on="patient_id", suffixes=("", "_past"))
    pairs = pairs[pairs.appointment_day_past < pairs.appointment_day]
    pairs["days_ago"] = (pairs.appointment_day - pairs.appointment_day_past).dt.days
    pairs["w"] = 0.5 ** (pairs.days_ago / half_life)
    pairs["w_shows"] = pairs.w * (1 - pairs.no_show)
    hist = pairs.groupby("appointment_id").agg(
        prior_visits=("no_show", "size"), prior_no_shows=("no_show", "sum"),
        w_visits=("w", "sum"), w_shows=("w_shows", "sum"), days_since_last=("days_ago", "min"))
    out = df.merge(hist, on="appointment_id", how="left")
    counts = ["prior_visits", "prior_no_shows", "w_visits", "w_shows"]
    out[counts] = out[counts].fillna(0)
    return out


def split(df, date_col="appointment_day", date=SPLIT_DATE):
    return df[df[date_col] <= date], df[df[date_col] > date]

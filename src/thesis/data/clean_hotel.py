import pandas as pd

# filled in or changed after booking, so they can give away the outcome
LEAKY = ["reservation_status", "reservation_status_date", "assigned_room_type", "days_in_waiting_list",
         "booking_changes", "required_car_parking_spaces", "total_of_special_requests"]


def clean_hotel(df):
    log = [("raw rows", 0, len(df))]
    df = df.fillna({"children": 0, "agent": 0, "company": 0, "country": "unknown"})

    df["arrival_date"] = pd.to_datetime(df.arrival_date_year.astype(str) + df.arrival_date_month
                                        + df.arrival_date_day_of_month.astype(str), format="%Y%B%d")
    # only for describing when people cancel, never a feature
    df["cancel_days_before_arrival"] = (df.arrival_date - pd.to_datetime(df.reservation_status_date)).dt.days.where(df.is_canceled == 1)
    df["group_size"] = df.adults + df.children + df.babies
    df[["children", "agent", "company"]] = df[["children", "agent", "company"]].astype(int)
    df[["agent", "company"]] = df[["agent", "company"]].astype(str).replace("0", "none")

    for reason, rule in [("no guests", lambda d: d.group_size == 0), ("negative price", lambda d: d.adr < 0)]:
        bad = rule(df)
        log.append((reason, bad.sum(), (~bad).sum()))
        df = df[~bad]

    # no booking ID, so identical rows are kept once (results are for unique bookings)
    n = len(df)
    df = df.drop_duplicates()
    log.append(("identical rows kept once", n - len(df), len(df)))

    log = pd.DataFrame(log, columns=["step", "rows_removed", "rows_left"]).assign(dataset="hotel")
    return df.drop(columns=LEAKY).sort_values("arrival_date").reset_index(drop=True), log

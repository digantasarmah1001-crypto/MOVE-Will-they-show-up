"""Clean the hotel booking demand data.

Every row we drop is written to a log so the thesis can report it.
"""

import pandas as pd

# Columns that are filled in or changed after the booking is made, so they can reveal
# the outcome. They are never used as features.
LEAKY_COLUMNS = [
    "reservation_status",       # is the outcome itself
    "reservation_status_date",  # date of check-out or cancellation
    "assigned_room_type",       # only known at check-in
    "days_in_waiting_list",     # keeps changing after booking
    "booking_changes",          # keeps changing after booking
    "required_car_parking_spaces",  # in this data, cancelled bookings almost never have one
    "total_of_special_requests",    # can be added after booking
]


def clean_hotel(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return the cleaned data and a log of what was dropped."""
    df = raw.copy()
    log = [{"dataset": "hotel", "step": "raw rows", "rows_removed": 0, "rows_left": len(df)}]

    def drop(mask, reason):
        nonlocal df
        log.append({"dataset": "hotel", "step": reason, "rows_removed": int(mask.sum()), "rows_left": int((~mask).sum())})
        df = df[~mask].copy()

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" + df["arrival_date_month"] + "-" + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )

    # Only used to describe outcomes (how close to arrival people cancel), never as a feature.
    status_date = pd.to_datetime(df["reservation_status_date"])
    df["cancel_days_before_arrival"] = (df["arrival_date"] - status_date).dt.days.where(df["is_canceled"] == 1)

    # The data has no booking ID, so identical rows may be real separate bookings (for example
    # several rooms in one group). We keep them and only flag them.
    df["is_exact_duplicate"] = raw.duplicated().astype(int)

    df["children"] = df["children"].fillna(0).astype(int)
    df["group_size"] = df["adults"] + df["children"] + df["babies"]

    drop(df["group_size"] == 0, "no guests")
    drop(df["adr"] < 0, "negative price")

    # Missing agent or company means the booking was made without one.
    df["agent"] = df["agent"].fillna(0).astype(int).astype(str).replace("0", "none")
    df["company"] = df["company"].fillna(0).astype(int).astype(str).replace("0", "none")
    df["country"] = df["country"].fillna("unknown")

    df = df.drop(columns=LEAKY_COLUMNS)
    df = df.sort_values("arrival_date", kind="stable").reset_index(drop=True)
    return df, log

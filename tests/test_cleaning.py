"""Check the cleaning rules on tiny made-up tables."""

import pandas as pd

from thesis.data.clean_hotel import LEAKY_COLUMNS, clean_hotel
from thesis.data.clean_noshow import clean_noshow


def noshow_rows():
    base = {
        "PatientId": 1.0, "AppointmentID": 1, "Gender": "F",
        "ScheduledDay": "2016-05-02T10:00:00Z", "AppointmentDay": "2016-05-02T00:00:00Z",
        "Age": 30, "Neighbourhood": "X", "Scholarship": 0, "Hipertension": 0, "Diabetes": 0,
        "Alcoholism": 0, "Handcap": 0, "SMS_received": 0, "No-show": "No",
    }
    rows = [
        base,                                                               # same day, showed
        {**base, "AppointmentID": 2, "ScheduledDay": "2016-05-01T09:00:00Z", "No-show": "Yes", "Handcap": 3},
        {**base, "AppointmentID": 3, "Age": -1},                            # bad age
        {**base, "AppointmentID": 4, "ScheduledDay": "2016-05-05T09:00:00Z"},  # booked after the appointment
    ]
    return pd.DataFrame(rows)


def test_noshow_outcome_and_lead_time():
    df, _ = clean_noshow(noshow_rows())
    df = df.set_index("appointment_id")
    assert df.loc[1, "no_show"] == 0 and df.loc[2, "no_show"] == 1  # "Yes" means did not show
    assert df.loc[1, "lead_days"] == 0  # same day booking, even though it has a time of day
    assert df.loc[2, "lead_days"] == 1
    assert df.loc[2, "handicap"] == 1


def test_noshow_drops_are_logged():
    df, log = clean_noshow(noshow_rows())
    assert set(df["appointment_id"]) == {1, 2}
    assert sum(step["rows_removed"] for step in log) == 2


def hotel_rows():
    base = {
        "hotel": "City Hotel", "is_canceled": 0, "lead_time": 10, "arrival_date_year": 2016,
        "arrival_date_month": "July", "arrival_date_week_number": 27, "arrival_date_day_of_month": 10,
        "stays_in_weekend_nights": 1, "stays_in_week_nights": 2, "adults": 2, "children": 0.0, "babies": 0,
        "meal": "BB", "country": "PRT", "market_segment": "Online TA", "distribution_channel": "TA/TO",
        "is_repeated_guest": 0, "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A", "assigned_room_type": "A", "booking_changes": 0, "deposit_type": "No Deposit",
        "agent": 9.0, "company": None, "days_in_waiting_list": 0, "customer_type": "Transient", "adr": 100.0,
        "required_car_parking_spaces": 0, "total_of_special_requests": 0,
        "reservation_status": "Check-Out", "reservation_status_date": "2016-07-13",
    }
    rows = [
        base,
        {**base, "is_canceled": 1, "reservation_status": "Canceled", "reservation_status_date": "2016-07-03"},
        {**base, "adults": 0},       # no guests
        {**base, "adr": -5.0},       # negative price
    ]
    return pd.DataFrame(rows)


def test_hotel_has_no_leaky_columns():
    df, _ = clean_hotel(hotel_rows())
    assert not set(LEAKY_COLUMNS) & set(df.columns)


def test_hotel_cleaning_rules():
    df, log = clean_hotel(hotel_rows())
    assert len(df) == 2
    assert sum(step["rows_removed"] for step in log) == 2
    cancelled = df[df["is_canceled"] == 1].iloc[0]
    assert cancelled["cancel_days_before_arrival"] == 7
    assert df[df["is_canceled"] == 0]["cancel_days_before_arrival"].isna().all()
    assert df["company"].iloc[0] == "none" and df["agent"].iloc[0] == "9"

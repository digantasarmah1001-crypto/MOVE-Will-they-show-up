import pandas as pd

from thesis.data.clean_hotel import LEAKY, clean_hotel
from thesis.data.clean_noshow import clean_noshow

appt = {"PatientId": 1.0, "AppointmentID": 1, "Gender": "F", "ScheduledDay": "2016-05-02T10:00:00Z",
        "AppointmentDay": "2016-05-02T00:00:00Z", "Age": 30, "Neighbourhood": "X", "Scholarship": 0,
        "Hipertension": 0, "Diabetes": 0, "Alcoholism": 0, "Handcap": 0, "SMS_received": 0, "No-show": "No"}

booking = {"hotel": "City Hotel", "is_canceled": 0, "lead_time": 10, "arrival_date_year": 2016,
           "arrival_date_month": "July", "arrival_date_week_number": 27, "arrival_date_day_of_month": 10,
           "stays_in_weekend_nights": 1, "stays_in_week_nights": 2, "adults": 2, "children": 0.0, "babies": 0,
           "meal": "BB", "country": "PRT", "market_segment": "Online TA", "distribution_channel": "TA/TO",
           "is_repeated_guest": 0, "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
           "reserved_room_type": "A", "assigned_room_type": "A", "booking_changes": 0, "deposit_type": "No Deposit",
           "agent": 9.0, "company": None, "days_in_waiting_list": 0, "customer_type": "Transient", "adr": 100.0,
           "required_car_parking_spaces": 0, "total_of_special_requests": 0,
           "reservation_status": "Check-Out", "reservation_status_date": "2016-07-13"}


def test_noshow():
    raw = pd.DataFrame([appt,
                        {**appt, "AppointmentID": 2, "ScheduledDay": "2016-05-01T09:00:00Z", "No-show": "Yes", "Handcap": 3},
                        {**appt, "AppointmentID": 3, "Age": -1},
                        {**appt, "AppointmentID": 4, "ScheduledDay": "2016-05-05T09:00:00Z"}])
    df, log = clean_noshow(raw)
    df = df.set_index("appointment_id")
    assert list(df.index) == [2, 1] and log.rows_removed.sum() == 2
    assert df.no_show.tolist() == [1, 0] and df.lead_days.tolist() == [1, 0] and df.loc[2, "handicap"] == 1


def test_hotel():
    cancelled = {**booking, "is_canceled": 1, "reservation_status": "Canceled", "reservation_status_date": "2016-07-03"}
    raw = pd.DataFrame([booking, booking, cancelled, {**booking, "adults": 0}, {**booking, "adr": -5.0}])
    df, log = clean_hotel(raw)
    assert not set(LEAKY) & set(df.columns)
    assert len(df) == 2
    assert df.set_index("is_canceled").cancel_days_before_arrival.fillna(-1).to_dict() == {0: -1, 1: 7}
    assert df.company.iloc[0] == "none" and df.agent.iloc[0] == "9"

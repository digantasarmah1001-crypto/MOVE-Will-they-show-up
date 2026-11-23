import pandas as pd

# filled in or changed after booking, so they can give away the outcome
LEAKY = ["reservation_status", "reservation_status_date", "assigned_room_type", "days_in_waiting_list",
         "booking_changes", "required_car_parking_spaces", "total_of_special_requests"]


def drop(df, bad, reason, log):
    log.append((reason, bad.sum(), (~bad).sum()))
    return df[~bad]


def clean_noshow(df):
    df = df.rename(columns={"PatientId": "patient_id", "AppointmentID": "appointment_id", "ScheduledDay": "scheduled_at",
                            "AppointmentDay": "appointment_day", "Hipertension": "hypertension", "Handcap": "handicap",
                            "SMS_received": "sms_received", "No-show": "no_show"})
    df.columns = df.columns.str.lower()
    log = [("raw rows", 0, len(df))]
    df["no_show"] = (df.no_show == "Yes").astype(int)
    df["patient_id"] = df.patient_id.astype("int64")
    df["scheduled_at"] = pd.to_datetime(df.scheduled_at).dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df.appointment_day).dt.tz_localize(None)
    df["lead_days"] = (df.appointment_day - df.scheduled_at.dt.normalize()).dt.days
    df["handicap"] = (df.handicap > 0).astype(int)
    df = drop(df, df.age < 0, "age below 0", log)
    df = drop(df, df.lead_days < 0, "appointment before it was scheduled", log)
    # one visit per patient per day, missed if any appointment that day was missed
    df = df.sort_values(["appointment_day", "scheduled_at", "appointment_id"])
    df["sms_mixed"] = df.groupby(["patient_id", "appointment_day"]).sms_received.transform("nunique") > 1
    for col in ["no_show", "sms_received"]:
        df[col] = df.groupby(["patient_id", "appointment_day"])[col].transform("max")
    df = drop(df, df.duplicated(["patient_id", "appointment_day"]), "same-day appointments merged into one visit", log)
    log = pd.DataFrame(log, columns=["step", "rows_removed", "rows_left"]).assign(dataset="noshow")
    return df.reset_index(drop=True), log


def clean_hotel(df):
    log = [("raw rows", 0, len(df))]
    df = df.fillna({"children": 0, "agent": 0, "company": 0, "country": "unknown"})
    df["arrival_date"] = pd.to_datetime(df.arrival_date_year.astype(str) + df.arrival_date_month
                                        + df.arrival_date_day_of_month.astype(str), format="%Y%B%d")
    status_date = pd.to_datetime(df.reservation_status_date)
    df["cancel_days_before_arrival"] = (df.arrival_date - status_date).dt.days.where(df.is_canceled == 1)
    df[["children", "agent", "company"]] = df[["children", "agent", "company"]].astype(int)
    df["group_size"] = df.adults + df.children + df.babies
    df[["agent", "company"]] = df[["agent", "company"]].astype(str).replace("0", "none")
    df = drop(df, df.group_size == 0, "no guests", log)
    df = drop(df, df.adr < 0, "negative price", log)
    df = drop(df, df.duplicated(), "identical rows kept once", log)
    log = pd.DataFrame(log, columns=["step", "rows_removed", "rows_left"]).assign(dataset="hotel")
    return df.drop(columns=LEAKY).sort_values("arrival_date").reset_index(drop=True), log

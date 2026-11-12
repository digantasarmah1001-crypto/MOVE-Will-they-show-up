import pandas as pd


def clean_noshow(df):
    df = df.rename(columns={"PatientId": "patient_id", "AppointmentID": "appointment_id", "ScheduledDay": "scheduled_at",
                            "AppointmentDay": "appointment_day", "Hipertension": "hypertension", "Handcap": "handicap",
                            "SMS_received": "sms_received", "No-show": "no_show"})
    df.columns = df.columns.str.lower()
    log = [("raw rows", 0, len(df))]

    df["no_show"] = (df.no_show == "Yes").astype(int)  # "Yes" in the raw file means they did not come
    df["patient_id"] = df.patient_id.astype("int64")
    df["scheduled_at"] = pd.to_datetime(df.scheduled_at).dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df.appointment_day).dt.tz_localize(None)
    df["lead_days"] = (df.appointment_day - df.scheduled_at.dt.normalize()).dt.days
    df["handicap"] = (df.handicap > 0).astype(int)

    for reason, rule in [("age below 0", lambda d: d.age < 0), ("appointment before it was scheduled", lambda d: d.lead_days < 0)]:
        bad = rule(df)
        log.append((reason, bad.sum(), (~bad).sum()))
        df = df[~bad]

    log = pd.DataFrame(log, columns=["step", "rows_removed", "rows_left"]).assign(dataset="noshow")
    return df.sort_values(["appointment_day", "scheduled_at"]).reset_index(drop=True), log

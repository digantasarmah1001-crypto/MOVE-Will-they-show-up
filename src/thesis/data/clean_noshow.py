"""Clean the medical appointment no-show data.

Every row we drop is written to a log so the thesis can report it.
"""

import pandas as pd

RENAME = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_at",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",
    "SMS_received": "sms_received",
    "No-show": "no_show",
}


def clean_noshow(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return the cleaned data and a log of what was dropped."""
    df = raw.rename(columns=RENAME)
    log = [{"dataset": "noshow", "step": "raw rows", "rows_removed": 0, "rows_left": len(df)}]

    def drop(mask, reason):
        nonlocal df
        log.append({"dataset": "noshow", "step": reason, "rows_removed": int(mask.sum()), "rows_left": int((~mask).sum())})
        df = df[~mask].copy()

    # Outcome: in the raw file "Yes" means the patient did NOT show up.
    df["no_show"] = (df["no_show"] == "Yes").astype(int)

    # Patient IDs are stored as floats in the CSV. They are whole numbers, so make them integers.
    df["patient_id"] = df["patient_id"].astype("int64")

    # Scheduling has a time of day, the appointment only has a date.
    # Lead time is counted in whole calendar days, so same-day booking = 0.
    df["scheduled_at"] = pd.to_datetime(df["scheduled_at"]).dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"]).dt.tz_localize(None).dt.normalize()
    df["lead_days"] = (df["appointment_day"] - df["scheduled_at"].dt.normalize()).dt.days

    drop(df["age"] < 0, "age below 0")
    drop(df["lead_days"] < 0, "appointment before it was scheduled")

    # Handicap is coded 0 to 4 with very few rows above 1, so we keep only yes or no.
    df["handicap"] = (df["handicap"] > 0).astype(int)

    df = df.sort_values(["appointment_day", "scheduled_at", "appointment_id"]).reset_index(drop=True)
    return df, log

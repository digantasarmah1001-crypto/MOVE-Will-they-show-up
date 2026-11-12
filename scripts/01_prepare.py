"""Clean both raw datasets and save them to data/processed/.

Run scripts/download_data.py first.
"""

from pathlib import Path

import pandas as pd

from thesis.data.clean_hotel import clean_hotel
from thesis.data.clean_noshow import clean_noshow

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    noshow, noshow_log = clean_noshow(pd.read_csv(RAW / "KaggleV2-May-2016.csv"))
    hotel, hotel_log = clean_hotel(pd.read_csv(RAW / "hotel_bookings.csv"))

    noshow.to_parquet(PROCESSED / "noshow.parquet", index=False)
    hotel.to_parquet(PROCESSED / "hotel.parquet", index=False)

    log = pd.DataFrame(noshow_log + hotel_log)
    log.to_csv(TABLES / "cleaning_log.csv", index=False)
    print(log.to_string(index=False))

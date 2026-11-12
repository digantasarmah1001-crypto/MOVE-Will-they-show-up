from pathlib import Path

import pandas as pd

from thesis.data.clean_hotel import clean_hotel
from thesis.data.clean_noshow import clean_noshow

root = Path(__file__).resolve().parent.parent
(root / "data/processed").mkdir(exist_ok=True)
(root / "results/tables").mkdir(parents=True, exist_ok=True)

noshow, log1 = clean_noshow(pd.read_csv(root / "data/raw/KaggleV2-May-2016.csv"))
hotel, log2 = clean_hotel(pd.read_csv(root / "data/raw/hotel_bookings.csv"))

noshow.to_parquet(root / "data/processed/noshow.parquet", index=False)
hotel.to_parquet(root / "data/processed/hotel.parquet", index=False)
log = pd.concat([log1, log2])[["dataset", "step", "rows_removed", "rows_left"]]
log.to_csv(root / "results/tables/cleaning_log.csv", index=False)
print(log.to_string(index=False))

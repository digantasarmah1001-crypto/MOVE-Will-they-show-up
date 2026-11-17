from pathlib import Path

import numpy as np
import pandas as pd

from thesis.data.features import add_history

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"
noshow = add_history(pd.read_parquet(root / "data/processed/noshow.parquet"))
hotel = pd.read_parquet(root / "data/processed/hotel.parquet")
noshow["broken"], hotel["broken"] = noshow.no_show, hotel.is_canceled
bins, labels = [-1, 0, 7, 30, 90, np.inf], ["same day", "1-7", "8-30", "31-90", "90+"]
noshow["lead"] = pd.cut(noshow.lead_days, bins, labels=labels)
hotel["lead"] = pd.cut(hotel.lead_time, bins, labels=labels)
noshow["past_no_shows"] = noshow.prior_no_shows.clip(upper=2).astype(int).astype(str).replace("2", "2+")
noshow.loc[noshow.prior_visits == 0, "past_no_shows"] = "no history"
hotel["past_cancellations"] = hotel.previous_cancellations.clip(upper=2).astype(str).replace("2", "2+")
hotel["group"] = hotel.group_size.clip(upper=4).astype(str).replace("4", "4+")
parts = []
for name, df, drivers in [("noshow", noshow, ["lead", "past_no_shows", "sms_received"]),
                          ("hotel", hotel, ["lead", "past_cancellations", "deposit_type", "group"])]:
    for driver in drivers:
        rates = df.groupby(driver, observed=True).broken.agg(rate="mean", n="size")
        parts.append(rates.rename_axis("level").reset_index().assign(dataset=name, driver=driver))
table = pd.concat(parts)
table.to_csv(tables / "drivers.csv", index=False)
print(table.round(3).to_string(index=False))
# SMS was only sent for longer lead times, so compare within lead time groups
sms = noshow.pivot_table(index="lead", columns="sms_received", values="broken", observed=True).round(3)
sms.to_csv(tables / "sms_by_lead.csv")
print(sms)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.data.features import add_history

root = Path(__file__).resolve().parent.parent
figs = root / "results/figures"
figs.mkdir(parents=True, exist_ok=True)

noshow = add_history(pd.read_parquet(root / "data/processed/noshow.parquet"))
hotel = pd.read_parquet(root / "data/processed/hotel.parquet")
noshow["broken"], hotel["broken"] = noshow.no_show, hotel.is_canceled

lead_bins = [-1, 0, 7, 30, 90, np.inf]
lead_labels = ["same day", "1-7", "8-30", "31-90", "90+"]
noshow["lead"] = pd.cut(noshow.lead_days, lead_bins, labels=lead_labels)
hotel["lead"] = pd.cut(hotel.lead_time, lead_bins, labels=lead_labels)
noshow["past_no_shows"] = noshow.prior_no_shows.clip(upper=2).astype(int).astype(str).replace("2", "2+")
noshow.loc[noshow.prior_visits == 0, "past_no_shows"] = "no history"
hotel["past_cancellations"] = hotel.previous_cancellations.clip(upper=2).astype(str).replace("2", "2+")
hotel["group"] = hotel.group_size.clip(upper=4).astype(str).replace("4", "4+")

drivers = [("noshow", "lead"), ("noshow", "past_no_shows"), ("noshow", "sms_received"),
           ("hotel", "lead"), ("hotel", "past_cancellations"), ("hotel", "deposit_type"), ("hotel", "group")]
data = {"noshow": noshow, "hotel": hotel}
table = pd.concat([data[d].groupby(c, observed=True).broken.agg(rate="mean", n="size")
                   .rename_axis("level").reset_index().assign(dataset=d, driver=c) for d, c in drivers])
table.to_csv(root / "results/tables/drivers.csv", index=False)
print(table.round(3).to_string(index=False))

# SMS was only sent for longer lead times, so compare within lead time groups
sms = noshow.pivot_table(index="lead", columns="sms_received", values="broken", observed=True).round(3)
sms.to_csv(root / "results/tables/sms_by_lead.csv")
print("\nno-show rate by lead time and SMS\n", sms)

fig, ax = plt.subplots(figsize=(6, 4))
for d in data:
    data[d].groupby("lead", observed=True).broken.mean().plot(ax=ax, marker="o", label=d)
ax.set(xlabel="lead time (days)", ylabel="broken commitment rate", ylim=(0, None))
ax.legend()
fig.tight_layout()
fig.savefig(figs / "lead_time.png", dpi=150)

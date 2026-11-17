import hashlib
import io
import os
import zipfile
from pathlib import Path

import requests

# needs a Kaggle API key in KAGGLE_USERNAME and KAGGLE_KEY
RAW = Path(__file__).resolve().parent.parent / "data/raw"
DATASETS = {"joniarroba/noshowappointments": ("KaggleV2-May-2016.csv", "9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d"),
            "jessemostipak/hotel-booking-demand": ("hotel_bookings.csv", "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06")}

RAW.mkdir(parents=True, exist_ok=True)
user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
for ref, (filename, sha) in DATASETS.items():
    if (RAW / filename).exists():
        print(f"already have {filename}")
        continue
    response = requests.get(f"https://www.kaggle.com/api/v1/datasets/download/{ref}",
                            auth=(user, key) if user and key else None, timeout=120)
    response.raise_for_status()
    data = zipfile.ZipFile(io.BytesIO(response.content)).read(filename)
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"{filename} has changed on Kaggle, check it before using it")
    (RAW / filename).write_bytes(data)
    print(f"saved {filename}")

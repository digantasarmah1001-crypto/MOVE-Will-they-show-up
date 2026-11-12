"""Download both datasets from Kaggle into data/raw/.

Needs a Kaggle API key. Set KAGGLE_USERNAME and KAGGLE_KEY, or run where
the key is already provided (like our cloud session).
"""

import hashlib
import io
import os
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Kaggle dataset, file inside the zip, and its SHA-256 so we know we have the exact same data.
DATASETS = {
    "joniarroba/noshowappointments": (
        "KaggleV2-May-2016.csv",
        "9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d",
    ),
    "jessemostipak/hotel-booking-demand": (
        "hotel_bookings.csv",
        "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06",
    ),
}


def kaggle_auth():
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    return (user, key) if user and key else None


def download(ref, filename, expected_sha):
    target = RAW_DIR / filename
    if target.exists():
        print(f"already have {filename}")
        return

    url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"
    response = requests.get(url, auth=kaggle_auth(), timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        data = archive.read(filename)

    sha = hashlib.sha256(data).hexdigest()
    if sha != expected_sha:
        raise ValueError(f"{filename} has changed on Kaggle (sha {sha}). Check before using it.")

    target.write_bytes(data)
    print(f"saved {filename}")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ref, (filename, sha) in DATASETS.items():
        download(ref, filename, sha)

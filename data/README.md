# Data

Raw and processed data are **not** committed to git. Get the raw files with:

```bash
python scripts/download_data.py
```

The script checks each file's SHA-256, so everyone gets the exact same data.

| Dataset | Source | Licence | Citation |
|---|---|---|---|
| Medical Appointment No Shows (110,527 rows) | Kaggle `joniarroba/noshowappointments` (Joni Hoppen / Aquarela Analytics), Vitória, Brazil, 2016 | **CC BY-NC-SA 4.0**: attribution, non-commercial, share-alike (licence checked via Kaggle API on 12 Nov 2026) | Hoppen, J. (2017). *Medical Appointment No Shows*. Kaggle. |
| Hotel booking demand (119,390 rows) | Kaggle `jessemostipak/hotel-booking-demand` (copy of the data from the article below) | **CC BY 4.0** | Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126 |

## What the licences mean for this project
- **Academic thesis use** is allowed for both datasets.
- **No-show data (NC-SA):** models trained on it are not shipped in the commercial MOVE app. MOVE retrains on its own data after launch. Any derived dataset that is shared must keep CC BY-NC-SA 4.0, so processed files stay out of git too.
- **Hotel data (CC BY):** cite the DOI wherever it is used.

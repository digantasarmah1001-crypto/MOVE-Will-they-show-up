# Will They Show Up?

Calibrated prediction and management of missed medical appointments with hierarchical Bayesian models.

This repository holds the code for my MSc Data Science thesis at Harbour.Space University. It studies why patients miss appointments, how well their risk can be predicted, and how those predictions can guide reminders and overbooking.

## Questions

1. What drives missed appointments, and do the same drivers show up in a second setting (hotel bookings)?
2. Can a hierarchical Bayesian model give well calibrated no-show risks, also for new patients, compared with a hand-made score and machine learning models?
3. What is the effect of SMS reminders? A week when the SMS system sent nothing gives a natural experiment.
4. Do the predictions lead to better clinic decisions (reminders and overbooking), and are they fair across patient groups?

## Data

- Medical appointment no-shows, Vitória, Brazil, 2016 (Kaggle, CC BY-NC-SA 4.0)
- Hotel booking demand (Antonio, de Almeida and Nunes, 2019, CC BY 4.0)

Data is not stored here. See `data/README.md` for sources and licences.

## How to run

```bash
make setup
```

```bash
make all
```

`make all` downloads the data (a Kaggle API key is needed) and rebuilds every table and figure in `results/`. The Bayesian models take about an hour the first time. Saved fits are reused after that.

```bash
make test
```

## Structure

| Folder | What is in it |
|---|---|
| `src/thesis/data` | cleaning, patient history features, the time split |
| `src/thesis/models` | the hand-made score, the hierarchical Bayesian model, causal estimators |
| `src/thesis/eval` | log loss, Brier score, calibration, bootstrap |
| `src/thesis/opt` | clinic overbooking |
| `scripts` | one script per step, numbered in the order they run |
| `results` | tables and figures used in the thesis |
| `tests` | checks for cleaning, leakage, the models and the estimators |

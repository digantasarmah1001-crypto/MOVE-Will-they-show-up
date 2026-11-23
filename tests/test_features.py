import pandas as pd

from thesis.data.features import add_history

df = pd.DataFrame({"appointment_id": [1, 2, 3, 4], "patient_id": [7, 7, 7, 8], "no_show": [1, 0, 1, 0],
                   "appointment_day": pd.to_datetime(["2016-05-01", "2016-05-31", "2016-05-31", "2016-05-31"])})


def test_only_earlier_days_count():
    h = add_history(df).set_index("appointment_id")
    assert h.loc[1, "prior_visits"] == 0
    assert h.loc[2, "prior_visits"] == 1 and h.loc[3, "prior_visits"] == 1
    assert h.loc[2, "prior_no_shows"] == 1 and h.loc[2, "days_since_last"] == 30
    assert h.loc[2, "w_visits"] == 0.5 and h.loc[2, "w_shows"] == 0
    assert h.loc[4, "prior_visits"] == 0


def test_row_order_does_not_matter():
    a = add_history(df).set_index("appointment_id").sort_index()
    b = add_history(df.sample(frac=1, random_state=0)).set_index("appointment_id").sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_strata_have_readable_labels():
    from thesis.data.datasets import STRATA
    s = pd.cut(pd.Series([0, 1, 2, 5]), STRATA[0], labels=STRATA[1])
    assert list(s.astype(str)) == ["0", "1-2", "1-2", "3+"]

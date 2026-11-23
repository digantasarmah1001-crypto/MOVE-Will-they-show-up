import numpy as np
from scipy.stats import poisson_binom


def clinic_cost(shows, slots, idle_cost=1.0, over_cost=1.0):
    # an empty slot wastes the doctor's time, a patient beyond the slots means overtime or waiting
    return idle_cost * np.maximum(slots - shows, 0) + over_cost * np.maximum(shows - slots, 0)


def best_extra(p_booked, p_extra, slots, **costs):
    # how many extra patients (each showing with chance p_extra) to book on top of the booked ones.
    # Each extra patient updates the distribution of shows; the expected cost falls and then rises
    pmf = poisson_binom.pmf(np.arange(len(p_booked) + 1), p_booked)
    best, k = np.inf, 0
    while True:
        cost = clinic_cost(np.arange(len(pmf)), slots, **costs) @ pmf
        if cost > best:
            return k - 1
        best, k = cost, k + 1
        pmf = np.append(pmf, 0) * (1 - p_extra) + np.append(0, pmf) * p_extra

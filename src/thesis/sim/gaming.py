import numpy as np
import pandas as pd
from scipy.stats import beta

from thesis.models.move import ATTENDANCE, bayesian_average, reliability, show_up_score, simple_average


def learned_score(days_ago, attendance, a, b, half_life=30):
    # posterior Beta for one user: prior from the data plus their recency-weighted history
    w = 0.5 ** (np.asarray(days_ago, dtype=float) / half_life)
    shows = w @ np.asarray(attendance, dtype=float)
    return beta(a + shows, b + w.sum() - shows)


def scorers(a, b):
    # each takes a history [(day, attendance), ...] and the current day, and returns P(show)
    return {"MOVE C": lambda h, day: reliability([day - d for d, _ in h], [s for _, s in h]),
            "learned": lambda h, day: learned_score([day - d for d, _ in h], [s for _, s in h], a, b).mean()}


def whitewash(histories, true_p, day, a, b):
    # how much would each user's score change if they deleted their account and started again?
    rows = []
    for name, score in scorers(a, b).items():
        fresh = score([], day)
        gain = np.array([fresh - score(h, day) for h in histories])
        rows += [{"scorer": name, "true_p": p, "gain": g} for p, g in zip(true_p, gain)]
    return pd.DataFrame(rows)


def recovery(a, b, target, gap=3, max_plans=60):
    # a flaky user (2 shows out of 10 plans in the last month) now shows up every `gap` days.
    # how many plans until their score reaches the target?
    start = [(d, int(i < 2)) for i, d in enumerate(range(-30, 0, 3))]
    out = {}
    for name, score in scorers(a, b).items():
        h, day = list(start), 0
        for n in range(1, max_plans + 1):
            h.append((day, 1))
            if score(h, day) >= target:
                break
            day += gap
        out[name] = n
    return out


# the five personas from sim.js in the MOVE prototype: chance to show (by days ago), late cancel share, rating
PERSONAS = {"Marta (reliable)": (20, 90, lambda d: 0.95, 0.5, 4.5, 1), "Leo (flaky)": (20, 90, lambda d: 0.3, 0.3, 4.0, 1),
            "Sam (brand new)": (1, 7, lambda d: 1.0, 0.0, 5.0, 0), "Nico (rating farmer)": (20, 90, lambda d: 0.25, 0.0, 5.0, 0),
            "Ana (reformed)": (20, 90, lambda d: 0.2 if d > 60 else 0.95, 0.3, 4.5, 1)}


def personas(a, b, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, (n, days, chance, late_share, rating, spread) in PERSONAS.items():
        ago = rng.integers(0, days, n)
        showed = rng.random(n) < [chance(d) for d in ago]
        late = ~showed & (rng.random(n) < late_share)
        att = np.where(showed, ATTENDANCE["showed"], np.where(late, ATTENDANCE["late"], ATTENDANCE["no_show"]))
        ratings = np.clip(np.round(rating + (rng.random(showed.sum()) - 0.5) * 2 * spread), 1, 5)
        post = learned_score(ago, att, a, b)
        rows.append({"persona": name, "events": n, "showed": showed.sum(), "A rating mean": simple_average(ratings),
                     "B Bayesian rating": bayesian_average(ratings), "C MOVE score": show_up_score(ago, att, ratings),
                     "learned P(show)": post.mean(), "learned 90% interval": f"{post.ppf(0.05):.2f} to {post.ppf(0.95):.2f}"})
    return pd.DataFrame(rows)

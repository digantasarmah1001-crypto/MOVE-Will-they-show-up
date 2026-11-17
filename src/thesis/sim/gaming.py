import numpy as np
import pandas as pd
from scipy.stats import beta

from thesis.models.move import ATTENDANCE, bayesian_average, reliability, show_up_score, simple_average

# MOVE's personas from sim.js: events, days of history, show chance before / after day 60, late cancel share,
# usual rating and its spread
PERSONAS = {"Marta (reliable)": (20, 90, 0.95, 0.95, 0.5, 4.5, 1), "Leo (flaky)": (20, 90, 0.3, 0.3, 0.3, 4.0, 1),
            "Sam (brand new)": (1, 7, 1.0, 1.0, 0.0, 5.0, 0), "Nico (rating farmer)": (20, 90, 0.25, 0.25, 0.0, 5.0, 0),
            "Ana (reformed)": (20, 90, 0.2, 0.95, 0.3, 4.5, 1)}


def learned_score(days_ago, attendance, a, b, half_life=30):
    w = 0.5 ** (np.asarray(days_ago, dtype=float) / half_life)
    shows = w @ np.asarray(attendance, dtype=float)
    return beta(a + shows, b + w.sum() - shows)


def scorers(a, b, half_life=30):
    def move(h, day):
        return reliability([day - d for d, _ in h], [s for _, s in h], half_life)
    def learned(h, day):
        return learned_score([day - d for d, _ in h], [s for _, s in h], a, b, half_life).mean()
    return {"MOVE C": move, "learned": learned}


def whitewash(histories, true_p, day, a, b):
    # score change if each user deleted their account and started again
    rows = []
    for name, score in scorers(a, b).items():
        fresh = score([], day)
        rows += [{"scorer": name, "true_p": p, "gain": fresh - score(h, day)} for p, h in zip(true_p, histories)]
    return pd.DataFrame(rows)


def recovery(a, b, target, half_life=30, gap=3, max_plans=60):
    # a flaky user (2 of 10 plans kept last month) now shows up every `gap` days: plans until target
    start = [(d, int(i < 2)) for i, d in enumerate(range(-30, 0, 3))]
    out = {}
    for name, score in scorers(a, b, half_life).items():
        h, day = list(start), 0
        for n in range(1, max_plans + 1):
            h.append((day, 1))
            if score(h, day) >= target:
                break
            day += gap
        out[name] = n
    return out


def personas(a, b, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, (n, days, before, after, late_share, rating, spread) in PERSONAS.items():
        ago = rng.integers(0, days, n)
        showed = rng.random(n) < np.where(ago > 60, before, after)
        late = ~showed & (rng.random(n) < late_share)
        att = np.where(showed, ATTENDANCE["showed"], np.where(late, ATTENDANCE["late"], ATTENDANCE["no_show"]))
        ratings = np.clip(np.round(rating + (rng.random(showed.sum()) - 0.5) * 2 * spread), 1, 5)
        post = learned_score(ago, att, a, b)
        rows.append({"persona": name, "events": n, "showed": showed.sum(), "A rating mean": simple_average(ratings),
                     "B Bayesian rating": bayesian_average(ratings), "C MOVE score": show_up_score(ago, att, ratings),
                     "learned P(show)": post.mean(), "learned 90% interval": f"{post.ppf(0.05):.2f} to {post.ppf(0.95):.2f}"})
    return pd.DataFrame(rows)

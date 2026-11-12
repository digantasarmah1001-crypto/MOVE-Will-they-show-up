# MOVE's current score, ported from score.js in the move-showup-score prototype
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

PRIOR_MEAN, PRIOR_CONFIDENCE = 3.5, 5          # B
HALF_LIFE, PRIOR_SHOWS, PRIOR_NO_SHOWS = 30, 2, 1  # C
RELIABILITY_WEIGHT = 0.7
ATTENDANCE = {"showed": 1, "late": 0.5, "no_show": 0}


def simple_average(ratings):
    return np.mean(ratings) if len(ratings) else None


def bayesian_average(ratings, prior_confidence=PRIOR_CONFIDENCE):
    return (prior_confidence * PRIOR_MEAN + np.sum(ratings)) / (prior_confidence + len(ratings))


def reliability(days_ago, attendance, half_life=HALF_LIFE, a=PRIOR_SHOWS, b=PRIOR_NO_SHOWS):
    w = 0.5 ** (np.asarray(days_ago, dtype=float) / half_life)
    return (w @ np.asarray(attendance, dtype=float) + a) / (w.sum() + a + b)


def show_up_score(days_ago, attendance, ratings, weight=RELIABILITY_WEIGHT, half_life=HALF_LIFE, prior_confidence=PRIOR_CONFIDENCE):
    rating_part = (bayesian_average(ratings, prior_confidence) - 1) / 4
    x = 100 * (weight * reliability(days_ago, attendance, half_life) + (1 - weight) * rating_part)
    return int(np.floor(x + 0.5))  # same as JS Math.round, python round() would round 64.5 down


def label(score):
    return "Trusted" if score >= 75 else "Building trust" if score >= 50 else "Low show-up"


def fit_prior(shows, visits, y):
    # "tuned C": learn the prior (a, b) from data instead of fixing it at 2 and 1
    nll = lambda t: log_loss(y, (shows + np.exp(t[0])) / (visits + np.exp(t[0]) + np.exp(t[1])))
    return np.exp(minimize(nll, np.log([PRIOR_SHOWS, PRIOR_NO_SHOWS]), method="Nelder-Mead").x)

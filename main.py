"""Trophy rarity forecaster for GTA VI (make.fun bounty gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy, five quantiles of the percent of
Exophase-tracked players who will have earned it at the case's target time.

Why the forecast looks the way it does
----------------------------------------
Exophase's page is frozen between refreshes and only moves when it
refreshes, hours apart. At a refresh the tracked player count jumps and
every trophy's rate is diluted toward the newcomers' earn profile (the
kit documents one: players 638 -> 3,868 while the non-trivial rates
summed 47.8 -> 10.6 points, a x0.221 factor). So a flat history means
"no refresh yet", not "nothing will change": the forecast must expect
the next refresh to dilute rates hard.

Two regimes, picked from the history:

* Flat history (no refresh observed, or under 50 players of growth):
  expect one refresh before the target. The median rate is the last
  rate times 0.221; refresh-count uncertainty (0, 1 or 2 refreshes with
  probabilities 0.25 / 0.50 / 0.25) gives the five quantiles as the
  scenario atoms, with the interior quantiles at the geometric midpoint
  of their neighbours.

* Rich history (refreshes observed): a cohort model and a log-time
  trend, blended 50/50 in logit space. The cohort model splits holder
  growth at each refresh into newcomers arriving (fraction n of the
  player jump) and the existing base earning more (rate lam applied to
  the players not yet holding the trophy, over the inter-refresh gap),
  then integrates that forward to the target under a saturating player
  growth curve. The trend is a weighted fit of logit(rate) against
  log time. The spread comes from a rolling-origin backtest over the
  last eight refresh events.

Standard library only, deterministic, no network, no wall-clock reads.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader runs with -P, which leaves this file's directory off
# sys.path; put it back first so a helper module shipped beside this
# file would import (none is needed -- everything lives here).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Time origin for the log-time trend and the player growth curve.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- flat-history branch -------------------------------------------------
# Median dilution of one refresh, and the five quantile factors that
# follow from 0/1/2 refreshes with probabilities 0.25/0.50/0.25: the
# atoms are 0.221**2, 0.221 and 1, and the interior quantiles sit at the
# geometric midpoint of their neighbours.
FLAT_FACTORS = (0.0488, 0.104, 0.221, 0.470, 1.0)
FLAT_GROWTH_TRIGGER = 50

# --- cohort branch -------------------------------------------------------
EVENT_HALF_LIFE_H = 24.0      # recency half-life for refresh-event weights
FIT_HALF_LIFE_H = 48.0        # recency half-life for player/trend fits
SHRINK_PRIOR_PLAYERS = 2000.0
PLAYER_TAUS_H = (12, 18, 24, 36, 48, 72, 96, 144, 216, 336, 504, 1008,
                 100000)
ODE_STEP_H = 6.0
SIGMA_FIT_FALLBACK = 0.45
SIGMA_HORIZON_COEF = 0.30
SIGMA_MIN, SIGMA_MAX = 0.25, 1.20
BACKTEST_EVENTS = 8
BACKTEST_MIN_RESIDUALS = 10

RATE_FLOOR = 0.01
RATE_CEIL = 100.0
LOGIT_CLAMP_LO, LOGIT_CLAMP_HI = 0.01, 99.99


# --- small numeric helpers ------------------------------------------------

def parse_time(text):
    """Parse an ISO-8601 timestamp ending in Z into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(moment):
    return (moment - EPOCH).total_seconds() / 3600.0


def clamp(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def logit(rate_pct):
    r = clamp(rate_pct, LOGIT_CLAMP_LO, LOGIT_CLAMP_HI) / 100.0
    return math.log(r / (1.0 - r))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x)) if x < 700 else 100.0
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def round4(value):
    """Round to 4 decimals, halves up (as JavaScript's Math.round does)."""
    return math.floor(value * 10000.0 + 0.5) / 10000.0


def quantiles_from(value):
    """Turn a raw number into a JSON-friendly value: whole numbers as int."""
    v = round4(value)
    return int(v) if v == int(v) else v


def tidy_quantiles(raw):
    """Floor at 0.01, cap at 100, keep non-decreasing, round to 4 dp."""
    out = []
    prev = RATE_FLOOR
    for v in raw:
        q = clamp(round4(v), RATE_FLOOR, RATE_CEIL)
        if q < prev:
            q = prev
        out.append(quantiles_from(q))
        prev = q
    return out


def solve_2x2(s11, s12, s22, t1, t2):
    """Solve [[s11, s12], [s12, s22]] x = [t1, t2]; None if singular."""
    det = s11 * s22 - s12 * s12
    scale = max(s11 * s22, 1e-300)
    if abs(det) <= 1e-12 * scale:
        return None
    return ((t1 * s22 - t2 * s12) / det, (t2 * s11 - t1 * s12) / det)


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x; None when degenerate."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (my - b * mx, b)


# --- history: refresh events ----------------------------------------------

def find_events(history):
    """Detect refreshes: consecutive reads where the player count grew
    or any trophy rate moved. Returns a list of dicts, each with the
    pre/post read indices, the player jump, and the post-read time."""
    events = []
    for i in range(1, len(history)):
        prev, cur = history[i - 1], history[i]
        players_grew = cur["players_tracked"] > prev["players_tracked"]
        rates_moved = cur["rates"] != prev["rates"]
        if players_grew or rates_moved:
            events.append({
                "pre": i - 1,
                "post": i,
                "d_players": cur["players_tracked"] - prev["players_tracked"],
                "post_at": parse_time(cur["at"]),
            })
    return events


def total_growth(events):
    return sum(e["d_players"] for e in events if e["d_players"] > 0)


# --- flat-history branch ----------------------------------------------------

def flat_quantiles(last_rate):
    if last_rate is None:
        last_rate = 0.0
    if last_rate >= RATE_CEIL:
        return [100, 100, 100, 100, 100]
    return tidy_quantiles([last_rate * f for f in FLAT_FACTORS])


# --- cohort branch: per-trophy newcomer/progression fit --------------------

def fit_newcomer_rate(events, history, trophy_idx, last_rate, case_at):
    """Fit dE = n*dP + lam*(P-E)*dt per trophy by weighted least squares.

    Returns (n, lam). Falls back to a shrunk marginal newcomer rate
    with lam = 0 when there are too few usable events or the design
    is singular.
    """
    rows = []  # (weight, dP, susceptible*dt, dE)
    prev_refresh_at = parse_time(history[0]["at"])
    for ev in events:
        pre = history[ev["pre"]]
        post = history[ev["post"]]
        dp = ev["d_players"]
        if dp <= 0:
            prev_refresh_at = ev["post_at"]
            continue
        r_pre = pre["rates"][trophy_idx]
        r_post = post["rates"][trophy_idx]
        if r_pre is None or r_post is None:
            prev_refresh_at = ev["post_at"]
            continue
        p_pre = pre["players_tracked"]
        e_pre = r_pre / 100.0 * p_pre
        e_post = r_post / 100.0 * post["players_tracked"]
        dt_h = (ev["post_at"] - prev_refresh_at).total_seconds() / 3600.0
        prev_refresh_at = ev["post_at"]
        if dt_h <= 0:
            continue
        age_h = (case_at - ev["post_at"]).total_seconds() / 3600.0
        w = dp * (0.5 ** (age_h / EVENT_HALF_LIFE_H))
        rows.append((w, dp, (p_pre - e_pre) * dt_h, e_post - e_pre))

    if len(rows) >= 3:
        s11 = sum(w * x1 * x1 for w, x1, _, _ in rows)
        s12 = sum(w * x1 * x2 for w, x1, x2, _ in rows)
        s22 = sum(w * x2 * x2 for w, _, x2, _ in rows)
        t1 = sum(w * x1 * y for w, x1, _, y in rows)
        t2 = sum(w * x2 * y for w, _, x2, y in rows)
        solved = solve_2x2(s11, s12, s22, t1, t2)
        if solved is not None:
            n, lam = solved
            return clamp(n, 0.0, 1.0), max(lam, 0.0)

    # Fallback: weighted mean of the clipped per-event marginal rates,
    # shrunk toward the trophy's current rate with a 2,000-player prior.
    num = 0.0
    den = 0.0
    for w, dp, _, de in rows:
        num += w * clamp(de / dp, 0.0, 1.0)
        den += w
    base = (last_rate or 0.0) / 100.0
    if den > 0:
        n = (num + SHRINK_PRIOR_PLAYERS * base) / (den + SHRINK_PRIOR_PLAYERS)
    else:
        n = base
    return clamp(n, 0.0, 1.0), 0.0


# --- cohort branch: player growth curve -------------------------------------

def fit_player_curve(history, case_at, target_at):
    """Fit P(t) = A*(1-exp(-t/tau)) + B*t over a grid of tau values.

    Returns (curve, p_target): curve maps hours-since-epoch to the
    fitted player count, and p_target is the fitted count at the
    target time, clamped to at least the last observed count."""
    points = []
    prev_players = None
    for read in history:
        p = read["players_tracked"]
        if p == prev_players:
            continue  # frozen page: consecutive identical counts add nothing
        prev_players = p
        moment = parse_time(read["at"])
        age_h = (case_at - moment).total_seconds() / 3600.0
        points.append((hours_since_epoch(moment),
                       p, 0.5 ** (age_h / FIT_HALF_LIFE_H)))
    last_players = history[-1]["players_tracked"]
    if not points:
        return (lambda t: float(last_players)), float(last_players)

    t_target = hours_since_epoch(target_at)
    best = None
    for tau in PLAYER_TAUS_H:
        f1 = [1.0 - math.exp(-t / tau) for t, _, _ in points]
        f2 = [t for t, _, _ in points]
        ys = [p for _, p, _ in points]
        ws = [w for _, _, w in points]
        s11 = sum(w * a * a for a, w in zip(f1, ws))
        s12 = sum(w * a * b for a, b, w in zip(f1, f2, ws))
        s22 = sum(w * b * b for b, w in zip(f2, ws))
        t1 = sum(w * a * y for a, y, w in zip(f1, ys, ws))
        t2 = sum(w * b * y for b, y, w in zip(f2, ys, ws))
        solved = solve_2x2(s11, s12, s22, t1, t2)
        if solved is None:
            a_coef, b_coef = 0.0, 0.0
        else:
            a_coef, b_coef = solved
        # Constrained fit: a negative coefficient is refit on its own.
        if a_coef < 0.0:
            a_coef = 0.0
            den = sum(w * b * b for b, w in zip(f2, ws))
            b_coef = (sum(w * b * y for b, y, w in zip(f2, ys, ws)) / den
                      if den > 0 else 0.0)
        if b_coef < 0.0:
            b_coef = 0.0
            den = sum(w * a * a for a, w in zip(f1, ws))
            a_coef = (sum(w * a * y for a, y, w in zip(f1, ys, ws)) / den
                      if den > 0 else 0.0)
        a_coef = max(a_coef, 0.0)
        b_coef = max(b_coef, 0.0)
        err = sum(w * (y - (a_coef * a + b_coef * b)) ** 2
                  for a, b, y, w in zip(f1, f2, ys, ws))
        if best is None or err < best[0]:
            best = (err, a_coef, b_coef, tau)
    _, a_coef, b_coef, tau = best

    def curve(t_hours):
        return a_coef * (1.0 - math.exp(-t_hours / tau)) + b_coef * t_hours

    p_target = max(curve(t_target), float(last_players))
    return curve, p_target


# --- cohort branch: forward integration --------------------------------------

def integrate_holders(history, trophy_idx, n, lam, curve, target_at):
    """Step the holder count forward from the last read to the target in
    6-hour steps: each step adds newcomers (fraction n of the player
    jump) plus progression (rate lam on the players not yet holding
    the trophy, over the step), with the player count taken from the
    fitted saturating curve. Returns the forecast rate in percent."""
    last = history[-1]
    p_now = float(last["players_tracked"])
    r_last = last["rates"][trophy_idx]
    if r_last is None:
        r_last = 0.0
    e_now = r_last / 100.0 * p_now
    start = parse_time(last["at"])
    total_h = (target_at - start).total_seconds() / 3600.0
    if total_h <= 0:
        return clamp(r_last, RATE_FLOOR, RATE_CEIL)
    steps = max(1, int(math.ceil(total_h / ODE_STEP_H)))
    step_h = total_h / steps
    t_now = hours_since_epoch(start)
    for _ in range(steps):
        t_next = t_now + step_h
        p_next = max(curve(t_next), p_now)
        d_players = max(p_next - p_now, 0.0)
        e_now += n * d_players + lam * max(p_now - e_now, 0.0) * step_h
        e_now = min(e_now, p_next)
        p_now = p_next
        t_now = t_next
    return clamp(100.0 * e_now / p_now, RATE_FLOOR, RATE_CEIL)


# --- cohort branch: log-time trend -------------------------------------------

def trend_rate(history, trophy_idx, target_at):
    """Weighted fit of logit(rate) on ln(1 + hours since epoch); read at
    the target. Falls back to the last rate on a degenerate fit."""
    xs, ys, ws = [], [], []
    last_rate = history[-1]["rates"][trophy_idx]
    case_at = parse_time(history[-1]["at"])
    seen_x = set()
    for read in history:
        r = read["rates"][trophy_idx]
        if r is None or r <= 0:
            continue
        moment = parse_time(read["at"])
        x = math.log1p(hours_since_epoch(moment))
        xs.append(x)
        ys.append(logit(r))
        age_h = (case_at - moment).total_seconds() / 3600.0
        ws.append(0.5 ** (age_h / FIT_HALF_LIFE_H))
        seen_x.add(round(x, 9))
    if len(seen_x) < 3:
        return last_rate if last_rate is not None else 0.0
    fitted = weighted_line(xs, ys, ws)
    if fitted is None:
        return last_rate if last_rate is not None else 0.0
    a, b = fitted
    x_target = math.log1p(hours_since_epoch(target_at))
    return clamp(inv_logit(a + b * x_target), RATE_FLOOR, RATE_CEIL)


# --- blended forecaster -------------------------------------------------------

def blended_medians(case, run_backtest=True):
    """Return the per-trophy median forecast (branch A or B)."""
    history = case["history"]
    target_at = parse_time(case["target"])
    case_at = parse_time(case["at"])
    trophies = case["trophies"]
    last = history[-1]
    events = find_events(history)

    if not events or total_growth(events) < FLAT_GROWTH_TRIGGER:
        medians = []
        for i, trophy in enumerate(trophies):
            r = last["rates"][i]
            medians.append(100.0 if (r or 0.0) >= RATE_CEIL
                           else (r or 0.0) * FLAT_FACTORS[2])
        return medians, events

    players_curve, _ = fit_player_curve(history, case_at, target_at)
    medians = []
    for i, trophy in enumerate(trophies):
        r_last = last["rates"][i]
        if r_last is not None and r_last >= RATE_CEIL:
            medians.append(100.0)
            continue
        try:
            n, lam = fit_newcomer_rate(events, history, i, r_last, case_at)
            cohort = integrate_holders(history, i, n, lam, players_curve,
                                       target_at)
            trend = trend_rate(history, i, target_at)
            medians.append(inv_logit((logit(cohort) + logit(trend)) / 2.0))
        except Exception:
            medians.append(r_last if r_last is not None else 0.0)
    return medians, events


# --- backtest spread ----------------------------------------------------------

def backtest_sigma(case):
    """Rolling-origin backtest over the last eight refresh events:
    cut the case just before each event, forecast at the post-refresh
    read, and take the logit residual of the median vs the actual rate.
    Returns the fitted sigma (0.45 when too few residuals)."""
    history = case["history"]
    events = find_events(history)
    residuals = []
    for ev in events[-BACKTEST_EVENTS:]:
        cut = dict(case)
        cut["history"] = history[:ev["pre"] + 1]
        cut["at"] = history[ev["pre"]]["at"]
        cut["target"] = history[ev["post"]]["at"]
        post = history[ev["post"]]
        try:
            sub_medians, _ = blended_medians(cut, run_backtest=False)
        except Exception:
            continue
        for i, actual in enumerate(post["rates"]):
            if actual is None:
                continue
            residuals.append(logit(actual) - logit(sub_medians[i]))
    if len(residuals) < BACKTEST_MIN_RESIDUALS:
        return SIGMA_FIT_FALLBACK
    mean = sum(residuals) / len(residuals)
    var = sum((r - mean) ** 2 for r in residuals) / len(residuals)
    return math.sqrt(var)


# --- per-case forecast ---------------------------------------------------------

def forecast_case(case):
    history = case["history"]
    trophies = case["trophies"]
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels: %r" % (case["levels"],))
    if not history:
        return [{"id": t["id"], "q": [0.01, 0.25, 1.0, 4.0, 100]}
                for t in trophies]

    medians, events = blended_medians(case)
    target_at = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    if not events or total_growth(events) < FLAT_GROWTH_TRIGGER:
        rows = []
        for i, trophy in enumerate(trophies):
            rows.append({"id": trophy["id"],
                         "q": flat_quantiles(history[-1]["rates"][i])})
        return rows

    sigma_fit = backtest_sigma(case)
    sigma = clamp(math.sqrt(sigma_fit ** 2 +
                            (SIGMA_HORIZON_COEF * math.sqrt(horizon_days)) ** 2),
                  SIGMA_MIN, SIGMA_MAX)
    rows = []
    for i, trophy in enumerate(trophies):
        med = medians[i]
        if med >= RATE_CEIL:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        rows.append({"id": trophy["id"],
                     "q": tidy_quantiles(
                         [inv_logit(logit(med) + z * sigma) for z in Z])})
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(INPUT_DIR.glob("*.in")):
        with open(in_path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        out_path = OUTPUT_DIR / (in_path.stem + ".out")
        payload = json.dumps({"trophies": rows}, ensure_ascii=False)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(payload)


if __name__ == "__main__":
    main()

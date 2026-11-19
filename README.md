# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xdml666/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xdml666/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: a cohort-dilution forecaster (newcomers vs progression) blended with a log-time trend.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `<id>.in` case
from the input directory and writes `<id>.out` to the output directory:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row
per trophy, five quantiles of the percent of Exophase-tracked players who
will have earned it at the case's target time. Deterministic, no network,
no wall-clock reads.

## How it works

Exophase's page is frozen between refreshes and only moves when it
refreshes, hours apart. At a refresh the tracked player count jumps and
every trophy's rate is diluted toward the newcomers' earn profile — the
kit documents one refresh in full: players 638 → 3,868 while the
non-trivial rates summed 47.8 → 10.6 points, a ×0.221 factor. So a flat
history means "no refresh yet", not "nothing will change".

- **Flat history** (no refresh observed, or under 50 players of growth):
  expect one refresh before the target. The median rate is the last rate
  times 0.221; refresh-count uncertainty (0, 1 or 2 refreshes with
  probabilities 0.25 / 0.50 / 0.25) gives the five quantiles, with the
  interior ones at the geometric midpoint of their neighbours. The 100%
  trophy stays 100.
- **Rich history** (refreshes observed): a cohort model and a log-time
  trend, blended 50/50 in logit space. The cohort model splits holder
  growth at each refresh into newcomers arriving (a fitted fraction of
  the player jump) and the existing base earning more (a fitted rate on
  the players not yet holding the trophy, over the inter-refresh gap —
  never the 30-minute read cadence, since the page is frozen between
  refreshes), then integrates that forward to the target under a
  saturating player-growth curve. The trend is a recency-weighted fit of
  logit(rate) against log time. The spread comes from a rolling-origin
  backtest over the last eight refresh events.

On the kit's practice cases the median mean rate is 2.010%, matching the
documented true 2.01%, with an expected pinball loss of ~0.21 against
~1.5 for the persistence and fitted-curve baselines. Runtime is about
0.03 s per practice case and ~0.25 s for a 121-read rich-history case
(cap 8 s per case).

---

Entered on [make.fun](https://make.fun/?ref=j54m1ph6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards

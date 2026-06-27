#!/usr/bin/env python3
"""
elo.py — historical team-strength prior (Phase 1.2). Pure math, no network.

THE PROBLEM IT SOLVES
---------------------
Estimating a team from its ≤3–5 most recent games (what update.py did) is far too
noisy: a single fluke scoreline swings the rating. This module builds a strong,
low-variance PRIOR from *all* of a team's historical results, weighting recent
games more (exponential time-decay). The sparse World-Cup form then becomes a
small shrinkage update on top of this prior, not the whole signal.

HOW
---
1. A margin-adjusted, time-decayed Elo rating per team (goal difference scales the
   update; staleness regresses a rating toward the mean between matches).
2. A mapping from Elo to the model's multiplicative attack/defence rates so the
   prior feeds goals, over/under, result, BTTS and (via supremacy) corners.
3. A shrinkage weight w(n) that is high when a team has few form games and decays
   as games accumulate: rating = w·prior + (1−w)·form.

All knobs are env-overridable so backtest.py can tune them without code edits.
The defaults below were chosen for international football and are sanity-checked
by the backtest; treat them as starting points, not gospel.
"""
import os, math, bisect
from collections import defaultdict
import model   # for MU_GOALS (the rate baseline); model never imports elo, so no cycle

# ----------------------------------------------------------------- constants (env-tunable)
ELO_INIT      = float(os.environ.get("ELO_INIT", "1500"))
ELO_MEAN      = float(os.environ.get("ELO_MEAN", "1500"))
ELO_K         = float(os.environ.get("ELO_K", "32"))       # base update size
ELO_HOME_ADV  = float(os.environ.get("ELO_HOME_ADV", "0")) # Elo pts for home side; 0 = neutral (WC)
ELO_HALFLIFE  = float(os.environ.get("ELO_HALFLIFE_DAYS", "450"))  # ~15 months (brief: 12–18)
ELO_MOV       = float(os.environ.get("ELO_MOV", "1.0"))    # margin-of-victory strength
ELO_GOAL_SCALE= float(os.environ.get("ELO_GOAL_SCALE", "0.9"))  # Elo gap -> goal-rate gap
ELO_RATE_CAP  = float(os.environ.get("ELO_RATE_CAP", "2.5"))    # clamp atk/def multipliers
SHRINK_TAU    = float(os.environ.get("ELO_SHRINK_TAU", "6"))    # form games to halve the prior weight
XG_RESULT_SCALE = float(os.environ.get("ELO_XG_RESULT_SCALE", "1.0"))  # xG diff -> soft win prob
# 2D learning rate (log-rate scale). Lightly tuned on real PL data: lower = more
# stable, and ~0.03 beat the 1D Elo on result/BTTS/corners (0.06 over-fit recent games).
ELO2D_K    = float(os.environ.get("ELO2D_K", "0.03"))
ELO2D_CLIP = float(os.environ.get("ELO2D_CLIP", "1.0")) # clamp |A|,|D| -> multipliers in [0.37, 2.7]

# ----------------------------------------------------------------- Elo -> rates
def atk_from_elo(elo):
    """Attack multiplier (1.0 = league average). Stronger team -> scores more."""
    s = math.exp(ELO_GOAL_SCALE * (elo - ELO_MEAN) / 400.0)
    return min(ELO_RATE_CAP, max(1.0 / ELO_RATE_CAP, s))

def def_from_elo(elo):
    """Defence multiplier — the rate at which the team lets the opponent score
    (>1 = concedes more). Stronger team -> concedes less. def = 1/atk by design:
    Elo is one-dimensional, so the prior splits strength evenly between scoring and
    conceding; the form term (which IS two-dimensional) restores team-specific
    attack/defence skew through the shrinkage blend."""
    return 1.0 / atk_from_elo(elo)

def shrink_weight(n_form_games):
    """Weight on the historical prior given how many form games exist.
    w(0)=1 (lean fully on the prior), decaying toward 0 as games accumulate.
    With TAU=6: n=6 -> w=0.5, n=18 -> w=0.25 (brief: high early, falls with games)."""
    return SHRINK_TAU / (SHRINK_TAU + max(0, n_form_games))

def blend(prior, form, w):
    """rating = w·prior + (1−w)·form, elementwise over atk/def/catk/ccon.
    Corner rates have no Elo prior, so they shrink toward 1.0 (league average),
    which also regularises the noisy small-sample corner form."""
    return {
        "atk":  w * prior["atk"]  + (1 - w) * form["atk"],
        "def":  w * prior["def"]  + (1 - w) * form["def"],
        "catk": w * 1.0           + (1 - w) * form.get("catk", 1.0),
        "ccon": w * 1.0           + (1 - w) * form.get("ccon", 1.0),
    }

# ----------------------------------------------------------------- Elo engine
def _days(a, b):
    """Whole days between two ISO-8601 timestamps (a after b). Robust to the
    'Z' suffix and to date-only strings."""
    from datetime import datetime
    def parse(s):
        s = s.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return datetime.fromisoformat(s[:10])
    return (parse(a) - parse(b)).total_seconds() / 86400.0

class EloEngine:
    """Fit margin-adjusted, time-decayed Elo over a chronological match list, then
    query each team's rating *as of* any timestamp without look-ahead leakage."""

    def __init__(self, use_xg=False):
        # use_xg: when a match carries expected goals (xgh/xga), update from the xG
        # margin and a soft (probabilistic) result instead of the raw scoreline. xG
        # is a lower-variance signal of who controlled the match, so the rating
        # converges faster and carries more signal per game (Phase 2.1). Falls back
        # to actual goals whenever xG is missing.
        self.use_xg = use_xg
        self.elo = defaultdict(lambda: ELO_INIT)
        self.last = {}                              # team -> ts of last match seen
        self.snap = defaultdict(list)               # team -> [(ts, elo_after)] time-sorted

    def _regress(self, team, ts):
        """Regress a stale rating toward the mean by the time-decay half-life."""
        if team in self.last:
            d = max(0.0, _days(ts, self.last[team]))
            f = 0.5 ** (d / ELO_HALFLIFE)
            self.elo[team] = ELO_MEAN + (self.elo[team] - ELO_MEAN) * f
        return self.elo[team]

    def update(self, ts, home, away, gh, ga, xgh=None, xga=None):
        Ra = self._regress(home, ts) + ELO_HOME_ADV
        Rb = self._regress(away, ts)
        Ea = 1.0 / (1.0 + 10 ** (-(Ra - Rb) / 400.0))    # expected score, home
        if self.use_xg and xgh is not None and xga is not None:
            diff = xgh - xga
            Sa = 1.0 / (1.0 + math.exp(-diff / XG_RESULT_SCALE))   # soft result from xG
        else:
            diff = gh - ga
            Sa = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        g = 1.0 + ELO_MOV * math.log1p(abs(diff))        # margin multiplier (1 for a draw)
        delta = ELO_K * g * (Sa - Ea)
        self.elo[home] += delta                          # symmetric zero-sum update
        self.elo[away] -= delta
        self.last[home] = self.last[away] = ts
        self.snap[home].append((ts, self.elo[home]))
        self.snap[away].append((ts, self.elo[away]))

    def fit(self, matches):
        for m in sorted(matches, key=lambda x: x["ts"]):
            if m["gh"] is None or m["ga"] is None:
                continue
            self.update(m["ts"], m["home"], m["away"], m["gh"], m["ga"],
                        m.get("xgh"), m.get("xga"))
        return self

    def elo_asof(self, team, ts):
        """Rating going INTO a match at `ts`: the team's last snapshot strictly
        before ts, further regressed toward the mean for the elapsed gap (so a
        long layoff between internationals decays the prior)."""
        snaps = self.snap.get(team)
        if not snaps:
            return ELO_INIT
        i = bisect.bisect_left([s[0] for s in snaps], ts) - 1
        if i < 0:
            return ELO_INIT
        last_ts, elo = snaps[i]
        d = max(0.0, _days(ts, last_ts))
        return ELO_MEAN + (elo - ELO_MEAN) * (0.5 ** (d / ELO_HALFLIFE))

    def rates_asof(self, team, ts):
        e = self.elo_asof(team, ts)
        return {"atk": atk_from_elo(e), "def": def_from_elo(e)}


def _clamp(x, cap):
    return max(-cap, min(cap, x))

class GoalEloEngine:
    """Two-dimensional attack/defence rating. The 1D EloEngine collapses strength to
    a single number (def = 1/atk), so it cannot represent a team that is high-scoring
    AND high-conceding (open, end-to-end sides) or the reverse (dour, low-event sides).
    This keeps SEPARATE attack (A) and defence (D) strengths on a log-rate scale and
    updates them by online gradient ascent on the Poisson log-likelihood of the score:

        log λ_home = ln(MU_GOALS) + A_home − D_away      (home advantage omitted:
        log λ_away = ln(MU_GOALS) + A_away − D_home       WC venues are neutral)

    d(logL)/dA_home = goals_home − λ_home, etc., giving the Elo-style updates below.
    Time-decay regresses both A and D toward 0 (league average). Exposes the same
    rates_asof() interface as EloEngine, so it drops into the shrinkage blend
    unchanged: atk = exp(A), def = exp(−D)."""

    def __init__(self, use_xg=False):
        self.use_xg = use_xg
        self.A = defaultdict(float)                  # attack strength (log), 0 = average
        self.D = defaultdict(float)                  # defence strength (log), higher = concedes less
        self.last = {}
        self.snap = defaultdict(list)                # team -> [(ts, A, D)] time-sorted

    def _regress(self, team, ts):
        if team in self.last:
            f = 0.5 ** (max(0.0, _days(ts, self.last[team])) / ELO_HALFLIFE)
            self.A[team] *= f; self.D[team] *= f

    def update(self, ts, home, away, gh, ga, xgh=None, xga=None):
        if self.use_xg and xgh is not None and xga is not None:
            gh, ga = xgh, xga                        # xG is a cleaner residual target
        self._regress(home, ts); self._regress(away, ts)
        lam_h = model.MU_GOALS * math.exp(self.A[home] - self.D[away])
        lam_a = model.MU_GOALS * math.exp(self.A[away] - self.D[home])
        rh, ra = gh - lam_h, ga - lam_a              # Poisson score residuals
        self.A[home] = _clamp(self.A[home] + ELO2D_K * rh, ELO2D_CLIP)
        self.D[away] = _clamp(self.D[away] - ELO2D_K * rh, ELO2D_CLIP)   # conceded more -> weaker D
        self.A[away] = _clamp(self.A[away] + ELO2D_K * ra, ELO2D_CLIP)
        self.D[home] = _clamp(self.D[home] - ELO2D_K * ra, ELO2D_CLIP)
        self.last[home] = self.last[away] = ts
        self.snap[home].append((ts, self.A[home], self.D[home]))
        self.snap[away].append((ts, self.A[away], self.D[away]))

    def fit(self, matches):
        for m in sorted(matches, key=lambda x: x["ts"]):
            if m["gh"] is None or m["ga"] is None:
                continue
            self.update(m["ts"], m["home"], m["away"], m["gh"], m["ga"],
                        m.get("xgh"), m.get("xga"))
        return self

    def rates_asof(self, team, ts):
        snaps = self.snap.get(team)
        if not snaps:
            return {"atk": 1.0, "def": 1.0}
        i = bisect.bisect_left([s[0] for s in snaps], ts) - 1
        if i < 0:
            return {"atk": 1.0, "def": 1.0}
        _, A, D = snaps[i]
        f = 0.5 ** (max(0.0, _days(ts, snaps[i][0])) / ELO_HALFLIFE)   # decay over the gap
        return {"atk": math.exp(A * f), "def": math.exp(-D * f)}

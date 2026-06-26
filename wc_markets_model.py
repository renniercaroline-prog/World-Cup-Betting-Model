"""
World Cup market model: corners, total goals, and half-time leader.

Design philosophy
-----------------
The PROBABILITIES come from statistical models, never from an LLM:
  * Goals  -> independent/bivariate Poisson (gives over/under, HT leader, result)
  * Corners-> Negative Binomial (counts are OVER-dispersed; Poisson under-prices
              the tails, i.e. the 5+ markets you care about)

"Team interaction" is encoded explicitly:
  expected corners for A  =  league_avg
                           * A.corner_attack        # how many A wins
                           * B.corner_concede        # how many B gives up
                           * game_state_multiplier   # chasing -> more corners

Game state is where Spain-Uruguay gets interesting: the side expected to TRAIL
attacks more (more crosses -> blocks -> corners), while the DOMINANT side wins
territorial corners. Both effects are modeled, so a naive single-team average
(the "Uruguay average 10 corners" trap) gets corrected by the matchup.

Swap the illustrative INPUTS below for real per-team rates from FBref / Understat
/ Opta. The structure is the deliverable; the example numbers are placeholders.
"""

import numpy as np
from scipy.stats import nbinom, poisson

# ----------------------------------------------------------------------
# INPUTS — replace with fitted values from real match data
# Rates are relative to a league average of 1.0 (so 1.15 = 15% above avg)
# ----------------------------------------------------------------------
LEAGUE_AVG_CORNERS = 5.1          # avg corners won per team per match
CORNER_DISPERSION  = 7.0          # NB dispersion r; lower = fatter tails

teams = {
    "Spain":   dict(xg_attack=1.85, xg_defense=0.70,   # goals model ratings
                    corner_attack=1.20, corner_concede=0.85),
    "Uruguay": dict(xg_attack=0.95, xg_defense=1.05,
                    corner_attack=1.00, corner_concede=1.10),
}

# Bookmaker odds you were quoted (decimal). Edit per market.
MARKET_ODDS = {
    "Uruguay 3+ corners": 1.667,   # 4/6
    "Uruguay 4+ corners": 2.45,    # 29/20
    "Uruguay 5+ corners": 4.00,    # 3/1
    "Over 1.5 goals":     1.40,    # 2/5
    "Spain leads at HT":  2.25,    # 5/4
}

# ----------------------------------------------------------------------
# GOALS MODEL  (independent Poisson)
# ----------------------------------------------------------------------
def expected_goals(att, dfn):
    mu = 1.35  # baseline goals per team
    lam_a = mu * att["xg_attack"] * dfn["xg_defense"]
    lam_b = mu * dfn["xg_attack"] * att["xg_defense"]
    return lam_a, lam_b

def over_under(lam_a, lam_b, line=1.5):
    total = lam_a + lam_b                      # sum of Poissons is Poisson
    k = int(np.floor(line))
    p_under = poisson.cdf(k, total)
    return 1 - p_under

def leads_at_half(lam_a, lam_b, first_half_share=0.45):
    # ~45% of goals arrive in the 1st half historically
    la, lb = lam_a * first_half_share, lam_b * first_half_share
    p = 0.0
    for i in range(8):
        for j in range(8):
            if i > j:
                p += poisson.pmf(i, la) * poisson.pmf(j, lb)
    return p

# ----------------------------------------------------------------------
# CORNERS MODEL  (Negative Binomial with interaction + game state)
# ----------------------------------------------------------------------
def game_state_multiplier(team_supremacy):
    """team_supremacy = expected goal difference for THIS team (+ = favored).
    Dominant teams win territorial corners; trailing teams chase and earn them.
    Net effect is a mild U-shape, strongest for the side expected to trail."""
    dominance = 1 + 0.10 * np.tanh(team_supremacy)       # territory
    chasing   = 1 + 0.18 * max(0.0, -team_supremacy)      # chasing late
    return dominance * chasing

def expected_corners(att, dfn, supremacy):
    return (LEAGUE_AVG_CORNERS
            * att["corner_attack"]
            * dfn["corner_concede"]
            * game_state_multiplier(supremacy))

def corner_tail_prob(lam, k, r=CORNER_DISPERSION):
    """P(corners >= k) under Negative Binomial with mean lam, dispersion r."""
    p = r / (r + lam)
    return 1 - nbinom.cdf(k - 1, r, p)

# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------
A, B = "Spain", "Uruguay"
lam_spain, lam_uru = expected_goals(teams[A], teams[B])
supremacy_uru = lam_uru - lam_spain          # negative => Uruguay expected to trail

uru_corners = expected_corners(teams[B], teams[A], supremacy_uru)
spa_corners = expected_corners(teams[A], teams[B], -supremacy_uru)

model_prob = {
    "Uruguay 3+ corners": corner_tail_prob(uru_corners, 3),
    "Uruguay 4+ corners": corner_tail_prob(uru_corners, 4),
    "Uruguay 5+ corners": corner_tail_prob(uru_corners, 5),
    "Over 1.5 goals":     over_under(lam_spain, lam_uru, 1.5),
    "Spain leads at HT":  leads_at_half(lam_spain, lam_uru),
}

print(f"Expected goals     Spain {lam_spain:.2f} - {lam_uru:.2f} Uruguay")
print(f"Expected corners   Spain {spa_corners:.1f} - {uru_corners:.1f} Uruguay")
print("-" * 64)
print(f"{'market':22}{'model':>8}{'implied':>9}{'edge':>8}  verdict")
print("-" * 64)
for m, p in model_prob.items():
    o = MARKET_ODDS[m]
    implied = 1 / o
    ev = p * o - 1
    verdict = "VALUE" if ev > 0 else "pass"
    print(f"{m:22}{p*100:7.1f}%{implied*100:8.1f}%{ev*100:+7.1f}%  {verdict}")

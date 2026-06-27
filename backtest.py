#!/usr/bin/env python3
"""
backtest.py — the scoreboard for every modelling change (Phase 1.1).

WHY THIS EXISTS
---------------
A flagged "VALUE" bet is noise until the model is shown to be *calibrated*: when
it says 30%, the thing should happen ~30% of the time. This harness measures
that. Build it BEFORE changing the model so every later change (Elo prior, xG,
re-baselined constants) can be judged on Brier / log-loss instead of vibes.

WHAT IT DOES
------------
1. Loads a set of completed historical fixtures (goals + corners).
2. Walks forward through them in time order. For each test match it derives each
   team's rating using ONLY matches that finished strictly earlier (no
   look-ahead leakage), feeds those ratings through the real model.py functions,
   and records (predicted probability, actual outcome) for every market.
3. Reports Brier score and log-loss per market, plus reliability/calibration
   tables (and plots if matplotlib is installed). It also reports a naive
   base-rate baseline so each number has context — a model is only useful if it
   beats "always predict the average".

DATA MODES (mirrors the pipeline's sample-mode philosophy)
----------------------------------------------------------
- live      : pull real fixtures from API-Football (PRO has historical seasons).
              Cached aggressively to backtest_cache.json — history never changes.
- synthetic : no API key needed. Generates matches from a *known* true model
              (latent team strengths -> Poisson goals / NB corners) that is
              deliberately NOT identical to model.py's hand-set constants, so the
              calibration error it reveals is genuine. This is how you exercise
              the metrics machinery and demo the harness without burning quota.
- auto      : live if API_FOOTBALL_KEY is set, else synthetic. (default)

The RATING MODEL is pluggable (see RATING_MODELS). Today only the "baseline"
rolling-form estimator exists — it replicates the current update.py logic, so its
Brier is the honest "before" number. Step 3 (Phase 1.2) registers an "elo" model;
re-run with RATING_MODEL=elo to get the "after" number on the same scoreboard.

    python backtest.py                 # auto mode, baseline ratings
    RATING_MODEL=elo python backtest.py
    BACKTEST_MODE=live BACKTEST_LEAGUE=39 BACKTEST_SEASONS=2022,2023 python backtest.py
"""
import os, json, math, bisect, datetime, urllib.request
from collections import defaultdict

import numpy as np
import model, elo

# ----------------------------------------------------------------- config (env)
API_KEY    = os.environ.get("API_FOOTBALL_KEY", "").strip()
BASE       = "https://v3.football.api-sports.io"
MODE       = os.environ.get("BACKTEST_MODE", "auto").strip().lower()
LEAGUE     = int(os.environ.get("BACKTEST_LEAGUE", "39"))          # default: a club league for volume
SEASONS    = [int(s) for s in os.environ.get("BACKTEST_SEASONS", "2021,2022,2023").split(",") if s.strip()]
WITH_CORNERS = os.environ.get("BACKTEST_CORNERS", "0") == "1"      # corners need 1 extra call/fixture
WINDOW     = int(os.environ.get("BACKTEST_WINDOW", "5"))           # rolling form window (matches current code)
MIN_GAMES  = int(os.environ.get("BACKTEST_MIN_GAMES", "4"))        # need this many priors before we predict
RATING_KEY = os.environ.get("RATING_MODEL", "baseline").strip().lower()
SEED       = int(os.environ.get("BACKTEST_SEED", "12345"))
OUTDIR     = os.path.join(os.path.dirname(__file__), "backtest_output")
CACHE      = os.path.join(os.path.dirname(__file__), "backtest_cache.json")

OU_LINES      = (1.5, 2.5, 3.5)
CORNER_LINES  = (8.5, 9.5, 10.5)
EPS = 1e-12

# ----------------------------------------------------------------- data: a Match
# Minimal record the harness needs. ts is a sortable timestamp used purely to
# enforce "strictly earlier" when deriving ratings (no leakage).
def Match(ts, home, away, gh, ga, ch=None, ca=None, xgh=None, xga=None):
    return {"ts": ts, "home": home, "away": away,
            "gh": gh, "ga": ga, "ch": ch, "ca": ca, "xgh": xgh, "xga": xga}

# ----------------------------------------------------------------- live loader
def _api(path):
    req = urllib.request.Request(BASE + path, headers={"x-apisports-key": API_KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.load(r)
    if payload.get("errors"):
        print(f"!! API errors on {path.split('?')[0]}: {payload['errors']}")
    return payload.get("response", [])

def load_live(cache):
    """Fetch finished fixtures (+ optional corners) for the configured league/seasons.
    Field paths reuse the ones already proven in update.py: fixture.date,
    teams.home.id, goals.home, and the /fixtures/statistics 'Corner Kicks' type."""
    matches = []
    for season in SEASONS:
        ckey = f"fx_{LEAGUE}_{season}"
        if ckey in cache:
            rows = cache[ckey]
        else:
            rows = _api(f"/fixtures?league={LEAGUE}&season={season}&status=FT")
            cache[ckey] = rows
            print(f"   fetched {len(rows)} fixtures for league {LEAGUE} season {season}")
        for f in rows:
            try:
                fid = f["fixture"]["id"]
                ts  = f["fixture"]["date"]                      # ISO string sorts correctly
                home = f["teams"]["home"]["name"]; away = f["teams"]["away"]["name"]
                gh = f["goals"]["home"]; ga = f["goals"]["away"]
                if gh is None or ga is None:
                    continue
                ch = ca = None
                if WITH_CORNERS:
                    ck = f"co_{fid}"
                    stats = cache.get(ck) or _api(f"/fixtures/statistics?fixture={fid}")
                    cache.setdefault(ck, stats)
                    for side in stats:
                        me = side["team"]["id"] == f["teams"]["home"]["id"]
                        for s in side["statistics"]:
                            if s["type"] == "Corner Kicks" and s["value"] is not None:
                                if me: ch = s["value"]
                                else:  ca = s["value"]
                matches.append(Match(ts, home, away, gh, ga, ch, ca))
            except (KeyError, TypeError):
                continue                                        # skip malformed rows
    return matches

# ----------------------------------------------------------------- synthetic loader
def load_synthetic(n_teams=int(os.environ.get("BACKTEST_TEAMS", "24")),
                   rounds=int(os.environ.get("BACKTEST_ROUNDS", "200"))):
    """Generate matches from a KNOWN true model so calibration error is genuine.

    True process (intentionally different from model.py's hand-set constants so
    the harness has room to show a re-baselining win in Phase 1.3):
      - each team has latent atk, def ~ lognormal around 1.0
      - true goal mean TRUE_MU = 1.45 (vs model's MU_GOALS = 1.35)
      - goals ~ Poisson(TRUE_MU * atk_i * def_j); corners ~ NB around a team rate
    The model never sees the latent strengths — it must estimate them from past
    results, so there is real estimation error to measure.
    """
    rng = np.random.default_rng(SEED)
    TRUE_MU = 1.45
    TRUE_CORNERS = 5.3
    TRUE_RHO = -0.10                                    # real low-score dependence to recover
    teams = [f"T{i:02d}" for i in range(n_teams)]
    atk = {t: float(np.exp(rng.normal(0, 0.30))) for t in teams}
    dfn = {t: float(np.exp(rng.normal(0, 0.25))) for t in teams}   # >1 == concedes more
    catk = {t: float(np.exp(rng.normal(0, 0.20))) for t in teams}
    ccon = {t: float(np.exp(rng.normal(0, 0.20))) for t in teams}

    matches, t0 = [], datetime.datetime(2021, 8, 1, tzinfo=datetime.timezone.utc)
    day = 0
    for _ in range(rounds):                                    # several double round-robins
        order = list(teams); rng.shuffle(order)
        for i in range(0, n_teams, 2):
            h, a = order[i], order[i + 1]
            lh = TRUE_MU * atk[h] * dfn[a]
            la = TRUE_MU * atk[a] * dfn[h]
            # sample the scoreline from a Dixon-Coles-correlated grid (not independent
            # Poisson) so the low-score dependence is really in the data and the DC
            # correction has something to recover.
            grid = model.score_matrix(lh, la, rho=TRUE_RHO)
            flat = np.array([grid[i][j] for i in range(model.GMAX+1) for j in range(model.GMAX+1)])
            idx = int(rng.choice(flat.size, p=flat / flat.sum()))
            gh, ga = divmod(idx, model.GMAX + 1)
            # xG: a lower-variance reading of the same underlying intensity (Gamma
            # with small CV around the true lambda). This is WHY xG carries more
            # signal per game than the integer scoreline — Phase 2.1 exploits it.
            xg_k = 12.0
            xgh = float(rng.gamma(xg_k, lh / xg_k)); xga = float(rng.gamma(xg_k, la / xg_k))
            # NB corners: mean = league * catk_for * ccon_against, dispersion r=CORNER_R
            mch = TRUE_CORNERS * catk[h] * ccon[a]
            mca = TRUE_CORNERS * catk[a] * ccon[h]
            ch = int(rng.negative_binomial(model.CORNER_R, model.CORNER_R / (model.CORNER_R + mch)))
            ca = int(rng.negative_binomial(model.CORNER_R, model.CORNER_R / (model.CORNER_R + mca)))
            ts = (t0 + datetime.timedelta(days=day)).isoformat()
            matches.append(Match(ts, h, a, gh, ga, ch, ca, xgh, xga))
            day += 1
    return matches

def load_matches():
    mode = MODE
    if mode == "auto":
        mode = "live" if API_KEY else "synthetic"
    if mode == "live":
        cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
        m = load_live(cache)
        json.dump(cache, open(CACHE, "w"))
        return m, "live"
    return load_synthetic(), "synthetic"

# ----------------------------------------------------------------- rating models
# A rating model answers: "given everything that finished before this match,
# what are each team's atk/def/catk/ccon rates?" The harness is agnostic to how.
class BaselineRatings:
    """Rolling-form estimator. Replicates the CURRENT update.py team_rates logic:
    last WINDOW finished matches, goals/corners averaged and divided by the model's
    hand-set reference constants. This is the honest 'before' baseline."""
    key = "baseline"

    def __init__(self, matches):
        # per-team time-sorted history of (ts, gf, ga, cf, ca) for fast as-of lookup
        self.hist = defaultdict(list)
        for m in sorted(matches, key=lambda x: x["ts"]):
            self.hist[m["home"]].append((m["ts"], m["gh"], m["ga"], m["ch"], m["ca"]))
            self.hist[m["away"]].append((m["ts"], m["ga"], m["gh"], m["ca"], m["ch"]))
        self._ts = {t: [r[0] for r in rows] for t, rows in self.hist.items()}

    def _prior(self, team, asof):
        rows = self.hist.get(team, [])
        cut = bisect.bisect_left(self._ts.get(team, []), asof)   # strictly-earlier slice
        return rows[:cut][-WINDOW:]

    def n_before(self, team, asof):
        return len(self._prior(team, asof))

    def rate(self, team, asof):
        rows = self._prior(team, asof)
        if not rows:
            return {"atk": 1.0, "def": 1.0, "catk": 1.0, "ccon": 1.0}
        n = len(rows)
        gf = sum(r[1] for r in rows) / n
        ga = sum(r[2] for r in rows) / n
        cf = [r[3] for r in rows if r[3] is not None]
        ca = [r[4] for r in rows if r[4] is not None]
        catk = (sum(cf) / len(cf) / model.LEAGUE_AVG_CORNERS) if cf else 1.0
        ccon = (sum(ca) / len(ca) / model.LEAGUE_AVG_CORNERS) if ca else 1.0
        return {"atk": gf / model.MU_GOALS or 1.0, "def": ga / model.MU_GOALS or 1.0,
                "catk": catk or 1.0, "ccon": ccon or 1.0}

class EloRatings(BaselineRatings):
    """Phase 1.2: a historical Elo prior shrunk toward rolling WC form.
        rating = w·(Elo-derived prior) + (1−w)·(rolling form),  w = shrink_weight(n).
    Reuses BaselineRatings for the form term and corner history, so the ONLY
    difference from the 'before' baseline is the added prior + shrinkage — making
    the backtest delta a clean measure of what the prior buys."""
    key = "elo"

    def __init__(self, matches):
        super().__init__(matches)                  # rolling-form machinery
        self.engine = elo.EloEngine().fit(matches) # time-decayed margin-adjusted Elo

    def rate(self, team, asof):
        form  = super().rate(team, asof)
        n     = self.n_before(team, asof)
        prior = self.engine.rates_asof(team, asof) # {atk, def} from Elo
        return elo.blend(prior, form, elo.shrink_weight(n))

class XgEloRatings(EloRatings):
    """Phase 2.1: identical to EloRatings but the prior's Elo is updated from xG
    instead of raw goals. Isolates the single question 'does an xG-driven prior
    beat a goals-driven one?' — the rest of the pipeline is unchanged."""
    key = "xgelo"

    def __init__(self, matches):
        BaselineRatings.__init__(self, matches)    # goal-based rolling form (unchanged)
        self.engine = elo.EloEngine(use_xg=True).fit(matches)

class Elo2DRatings(EloRatings):
    """2D attack/defence prior (GoalEloEngine) shrunk toward rolling form. Tests
    whether separating attack and defence beats the 1D Elo on the scoreboard."""
    key = "elo2d"

    def __init__(self, matches):
        BaselineRatings.__init__(self, matches)
        self.engine = elo.GoalEloEngine().fit(matches)

class XgElo2DRatings(EloRatings):
    """2D attack/defence prior driven by xG — the most expressive combination."""
    key = "xgelo2d"

    def __init__(self, matches):
        BaselineRatings.__init__(self, matches)
        self.engine = elo.GoalEloEngine(use_xg=True).fit(matches)

RATING_MODELS = {BaselineRatings.key: BaselineRatings,
                 EloRatings.key: EloRatings,
                 XgEloRatings.key: XgEloRatings,
                 Elo2DRatings.key: Elo2DRatings,
                 XgElo2DRatings.key: XgElo2DRatings}

def get_rating_model(matches):
    cls = RATING_MODELS.get(RATING_KEY)
    if cls is None:
        raise SystemExit(f"unknown RATING_MODEL={RATING_KEY!r}; have {list(RATING_MODELS)}")
    return cls(matches)

# ----------------------------------------------------------------- predictions
def predict(home_r, away_r):
    """Run the REAL model.py math and return {market_label: probability}."""
    lh, la = model.goals(home_r, away_r)
    M = model.score_matrix(lh, la)                      # DC-corrected; read all FT markets off it
    ph, pd, pa = model.match_result(M)
    out = {"result": (ph, pd, pa)}                      # multiclass entry
    for line in OU_LINES:
        out[f"over_{line}"] = model.over_under_m(M, line)
    out["btts"] = model.btts_m(M)
    if home_r.get("catk") is not None:
        ch = model.team_corners(home_r, away_r, la - lh)
        ca = model.team_corners(away_r, home_r, lh - la)
        for line in CORNER_LINES:
            out[f"corners_over_{line}"] = model.total_corners_over(ch, ca, line)
    return out

def outcomes(m):
    """Actual realised outcome for each market from a finished match."""
    gh, ga = m["gh"], m["ga"]
    res = 0 if gh > ga else (1 if gh == ga else 2)      # H / D / A index
    o = {"result": res}
    for line in OU_LINES:
        o[f"over_{line}"] = int(gh + ga > line)
    o["btts"] = int(gh > 0 and ga > 0)
    if m["ch"] is not None and m["ca"] is not None:
        for line in CORNER_LINES:
            o[f"corners_over_{line}"] = int(m["ch"] + m["ca"] > line)
    return o

# ----------------------------------------------------------------- metrics
def _clip(p):
    return min(1 - EPS, max(EPS, p))

def binary_scores(pairs):
    p = np.array([x[0] for x in pairs]); y = np.array([x[1] for x in pairs])
    brier = float(np.mean((p - y) ** 2))
    pc = np.clip(p, EPS, 1 - EPS)
    logloss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    base = float(np.mean(y))                            # naive: always predict base rate
    base_brier = float(np.mean((base - y) ** 2))
    return brier, logloss, base_brier, len(pairs), base

def multiclass_scores(pairs):
    # pairs: (prob_vec(len3), actual_index)
    P = np.array([x[0] for x in pairs]); Y = np.array([x[1] for x in pairs])
    onehot = np.zeros_like(P); onehot[np.arange(len(Y)), Y] = 1
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))    # 0..2
    pc = np.clip(P[np.arange(len(Y)), Y], EPS, 1)
    logloss = float(-np.mean(np.log(pc)))
    base = onehot.mean(axis=0)                          # class base rates
    base_brier = float(np.mean(np.sum((base - onehot) ** 2, axis=1)))
    return brier, logloss, base_brier, len(pairs)

def calibration_table(pairs, bins=10):
    p = np.array([x[0] for x in pairs]); y = np.array([x[1] for x in pairs])
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append((edges[b], edges[b + 1], int(mask.sum()),
                     float(p[mask].mean()), float(y[mask].mean())))
    return rows

# ----------------------------------------------------------------- plots (optional)
def save_plots(binary_preds, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("   (matplotlib not installed — skipping calibration plots)")
        return
    os.makedirs(OUTDIR, exist_ok=True)
    markets = [k for k in binary_preds if binary_preds[k]]
    cols = 3; rows = math.ceil(len(markets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for ax, mk in zip([a for row in axes for a in row], markets):
        tbl = calibration_table(binary_preds[mk])
        xs = [r[3] for r in tbl]; ys = [r[4] for r in tbl]
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.plot(xs, ys, "o-", color="#F5C542")
        ax.set_title(mk, fontsize=9); ax.set_xlabel("predicted"); ax.set_ylabel("observed")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for ax in [a for row in axes for a in row][len(markets):]:
        ax.axis("off")
    fig.suptitle(f"Reliability — {tag}")
    fig.tight_layout()
    path = os.path.join(OUTDIR, f"calibration_{tag}.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"   calibration plots -> {path}")

# ----------------------------------------------------------------- harness
def recalibrate(train_frac=0.6):
    """Phase 1.3: suggest model constants fit from data instead of hand-set defaults.
    Uses only the earliest `train_frac` of matches so the suggestion never sees the
    test period. Prints values to drop into model.py / pass as env overrides, then
    re-run the backtest to confirm the Brier change."""
    matches, mode = load_matches()
    matches = [m for m in matches if m["gh"] is not None and m["ga"] is not None]
    matches.sort(key=lambda m: m["ts"])
    train = matches[:int(len(matches) * train_frac)]

    goals = [g for m in train for g in (m["gh"], m["ga"])]
    mu = float(np.mean(goals))
    corners = [c for m in train for c in (m["ch"], m["ca"]) if c is not None]
    out = {"MU_GOALS": round(mu, 3)}
    if corners:
        cmean = float(np.mean(corners)); cvar = float(np.var(corners))
        out["LEAGUE_AVG_CORNERS"] = round(cmean, 3)
        # NB method-of-moments: var = mean + mean^2/r  ->  r = mean^2 / (var - mean)
        out["CORNER_R"] = round(cmean ** 2 / (cvar - cmean), 2) if cvar > cmean else 99.0
    # H1_SHARE needs half-time goals; only live data carries them (synthetic does not)
    h1 = [m.get("h1") for m in train if m.get("h1") is not None]
    if h1:
        out["H1_SHARE"] = round(float(np.mean(h1)), 3)

    # Dixon-Coles rho: grid-search the value that minimises the EXACT-SCORELINE
    # negative log-likelihood (the objective DC actually targets — 1X2 log-loss is
    # nearly blind to the low-score dependence). Scored with walk-forward xG-Elo
    # ratings, so no leakage; rho is a single global constant.
    rat = XgEloRatings(train)
    def score_nll(rho):
        tot = n = 0
        for m in train:
            if rat.n_before(m["home"], m["ts"]) < MIN_GAMES or \
               rat.n_before(m["away"], m["ts"]) < MIN_GAMES:
                continue
            hr = rat.rate(m["home"], m["ts"]); ar = rat.rate(m["away"], m["ts"])
            lh, la = model.goals(hr, ar)
            M = model.score_matrix(lh, la, rho=rho)
            gh, ga = min(m["gh"], model.GMAX), min(m["ga"], model.GMAX)
            tot += -math.log(max(M[gh][ga], EPS)); n += 1
        return tot / max(n, 1)
    grid = {r: score_nll(r) for r in (0.0, -0.03, -0.06, -0.09, -0.12, -0.15)}
    best = min(grid, key=grid.get)
    out["RHO_DC"] = best
    print("  rho grid (exact-score NLL): " + "  ".join(f"{r:+.2f}:{v:.4f}" for r, v in grid.items()))

    print("=" * 72)
    print(f"RECALIBRATE  mode={mode}  train_matches={len(train)} (first {int(train_frac*100)}%)")
    print("=" * 72)
    print("Current model.py constants vs data-fitted suggestion:")
    cur = {"MU_GOALS": model.MU_GOALS, "LEAGUE_AVG_CORNERS": model.LEAGUE_AVG_CORNERS,
           "CORNER_R": model.CORNER_R, "H1_SHARE": model.H1_SHARE}
    for k, v in cur.items():
        s = out.get(k)
        print(f"  {k:<20} current={v:<8} fitted={s if s is not None else 'n/a (no half-time data)'}")
    env = " ".join(f"{k}={v}" for k, v in out.items())
    print("\nVerify the gain by re-running with these as env overrides:")
    print(f"  {env} RATING_MODEL=elo python3 backtest.py")
    return out

def run():
    matches, mode = load_matches()
    matches = [m for m in matches if m["gh"] is not None and m["ga"] is not None]
    matches.sort(key=lambda m: m["ts"])
    if len(matches) < 50:
        raise SystemExit(f"only {len(matches)} matches loaded — too few to backtest "
                         f"(mode={mode}). Check API key / league / seasons.")
    ratings = get_rating_model(matches)

    binary_preds = defaultdict(list)       # market -> [(p, y), ...]
    result_preds = []                      # [(p_vec, actual_idx), ...]
    tested = skipped = 0
    for m in matches:
        # walk-forward gate: both teams need enough finished matches BEFORE kickoff
        if ratings.n_before(m["home"], m["ts"]) < MIN_GAMES or \
           ratings.n_before(m["away"], m["ts"]) < MIN_GAMES:
            skipped += 1
            continue
        hr = {**ratings.rate(m["home"], m["ts"]), "name": m["home"]}
        ar = {**ratings.rate(m["away"], m["ts"]), "name": m["away"]}
        pred = predict(hr, ar); act = outcomes(m)
        result_preds.append((pred["result"], act["result"]))
        for mk, p in pred.items():
            if mk == "result" or mk not in act:
                continue
            binary_preds[mk].append((_clip(p), act[mk]))
        tested += 1

    # ----- report
    print("=" * 72)
    print(f"BACKTEST  mode={mode}  rating_model={RATING_KEY}  "
          f"matches={len(matches)}  tested={tested}  skipped(warmup)={skipped}")
    print(f"window={WINDOW}  min_games={MIN_GAMES}  "
          f"constants: MU_GOALS={model.MU_GOALS} CORNERS={model.LEAGUE_AVG_CORNERS} "
          f"CORNER_R={model.CORNER_R} H1_SHARE={model.H1_SHARE}")
    print("=" * 72)
    hdr = f"{'market':<20}{'n':>7}{'brier':>10}{'logloss':>10}{'base_brier':>12}{'skill%':>9}"
    print(hdr); print("-" * len(hdr))

    metrics = {}
    rb, rl, rbb, rn = multiclass_scores(result_preds)
    skill = 100 * (1 - rb / rbb) if rbb else 0
    print(f"{'result (1X2)':<20}{rn:>7}{rb:>10.4f}{rl:>10.4f}{rbb:>12.4f}{skill:>8.1f}%")
    metrics["result"] = {"n": rn, "brier": rb, "logloss": rl, "base_brier": rbb}

    for mk in sorted(binary_preds):
        b, l, bb, n, base = binary_scores(binary_preds[mk])
        skill = 100 * (1 - b / bb) if bb else 0
        print(f"{mk:<20}{n:>7}{b:>10.4f}{l:>10.4f}{bb:>12.4f}{skill:>8.1f}%")
        metrics[mk] = {"n": n, "brier": b, "logloss": l, "base_brier": bb, "base_rate": base}

    # pooled calibration over all binary markets
    pooled = [pair for mk in binary_preds for pair in binary_preds[mk]]
    print("-" * len(hdr))
    print("Pooled calibration (all binary markets):")
    print(f"  {'pred range':<14}{'n':>8}{'mean_pred':>12}{'observed':>12}")
    for lo, hi, n, mp, ob in calibration_table(pooled):
        print(f"  {f'{lo:.1f}-{hi:.1f}':<14}{n:>8}{mp:>12.3f}{ob:>12.3f}")
    print("=" * 72)
    print("skill% = 1 - brier/base_brier (>0 means better than predicting the base rate).")
    print("Record these numbers; re-run with a new RATING_MODEL to compare before/after.")

    # persist for programmatic before/after comparison in later steps
    os.makedirs(OUTDIR, exist_ok=True)
    summary = {"mode": mode, "rating_model": RATING_KEY, "tested": tested,
               "constants": {"MU_GOALS": model.MU_GOALS, "LEAGUE_AVG_CORNERS": model.LEAGUE_AVG_CORNERS,
                             "CORNER_R": model.CORNER_R, "H1_SHARE": model.H1_SHARE},
               "metrics": metrics}
    mpath = os.path.join(OUTDIR, f"metrics_{RATING_KEY}.json")
    json.dump(summary, open(mpath, "w"), indent=2)
    print(f"metrics JSON -> {mpath}")
    save_plots(binary_preds, RATING_KEY)
    return summary

if __name__ == "__main__":
    import sys
    if "--recalibrate" in sys.argv or os.environ.get("BACKTEST_RECALIBRATE") == "1":
        recalibrate()
    else:
        run()

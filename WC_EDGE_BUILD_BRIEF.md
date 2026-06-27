# WC Edge — Implementation Brief for Claude Code

You are picking up an existing, working project and extending it. Read this whole
file before editing anything. The repo is small and the architecture is
deliberate — preserve the separation of concerns described below.

---

## 0. TL;DR of what you're being asked to do

The tool predicts football betting markets for the 2026 FIFA World Cup. The
problem: it currently builds every prediction from **this World Cup only** (≈3
games per team), which is far too little data, so its outputs are noisy and not
trustworthy. Your job is to **make the predictions accurate by leaning on
historical data**, treating the sparse World Cup games as a small adjustment on
top of a strong historical prior — and to add a **backtest/calibration harness**
so improvements can be *measured*, not guessed at.

Also fold in three small fixes (details in §4).

The user is a data-science MSc student but **not a football expert** — make
sensible football-domain decisions yourself, and explain non-obvious choices in
code comments.

---

## 1. What the product is

A static, self-updating web app for a single user (the owner's friend) to check
World Cup betting markets:

- A daily GitHub Action runs a Python pipeline that fetches data, runs a
  statistical model, and writes `data.json`.
- A single static `index.html` (vanilla JS) reads `data.json` and renders one
  card per upcoming fixture, grouped by market type.
- The user **types in the decimal odds his sportsbook offers** for any market;
  the page computes implied probability, expected value, and a fractional-Kelly
  stake, and highlights positive-EV ("VALUE") selections.
- Hosted free on GitHub Pages. Updated daily at 12:00 UTC, plus manual
  `workflow_dispatch` runs (the user triggers one ~45–60 min before kickoff to
  capture confirmed lineups).

### Markets currently produced (per fixture)
- Match result (1X2)
- Total goals over/under: 0.5, 1.5, 2.5, 3.5, 4.5
- Both teams to score (yes/no)
- First-half result (1X2)
- Half-time/full-time (top combinations)
- Corners: total over/under lines + per-team thresholds
- Players: to score, to score-or-assist, 1+ shot on target, to commit a foul,
  to be fouled

---

## 2. CRITICAL design principles — do not violate

1. **The LLM never computes a probability.** All probabilities come from the
   statistical model in `model.py`. The OpenAI agent (`agent.py`) is used ONLY
   to read team news and project lineups/minutes/availability. If you ever find
   yourself asking the LLM for a number that becomes a betting probability, stop
   — that's the wrong design (it hallucinates and is uncalibrated).

2. **Match the statistical model to each market's shape.** Goals → Poisson /
   bivariate Poisson with Dixon-Coles low-score correction. Corners & cards →
   negative binomial (overdispersed). Player props → per-90 rate × projected
   minutes × opponent/context adjustment.

3. **Edge comes from soft markets, not 1X2.** The match-result line is
   efficiently priced and very hard to beat. The realistic edge is in corners,
   cards, and player props. Optimise accuracy there.

4. **History is the backbone; World Cup games are fine-tuning.** Never estimate a
   team or player purely from their ≤3 World Cup matches. Always start from a
   historical prior and shrink the WC sample toward it (more on this in §5).

5. **Nothing is trustworthy until backtested.** Until a market passes
   calibration + closing-line checks, its flagged "VALUE" is noise. The UI
   already warns the user; keep that honesty.

6. **Preserve the `data.json` contract** (see §6) so `index.html` keeps working,
   or update both sides together in the same change.

7. **Keep "sample mode" working.** With no API keys set, the pipeline must still
   run end-to-end and write clearly-labelled sample `data.json` so the page
   renders. This is how the user tests without burning quota.

---

## 3. Current architecture & files

```
index.html                     # vanilla-JS UI; reads data.json; odds input + EV/Kelly
model.py                       # ALL probability math; pure functions, no network
agent.py                       # OpenAI Responses API (gpt-5.5) + web_search → lineups only
update.py                      # orchestrator: fetch → agent → model → data.json
data.json                      # generated output the UI reads
cache.json                     # per-team/per-player stat cache (quota saver)
.github/workflows/daily.yml    # cron 12:00 UTC + manual dispatch
README.md
```

### `model.py` (current)
Constants: `MU_GOALS=1.35`, `LEAGUE_AVG_CORNERS=5.1`, `CORNER_R=7.0`,
`H1_SHARE=0.45`. Key functions: `goals()`, `score_matrix()`, `match_result()`,
`over_under()`, `btts()`, `half_result()`, `htft()`, `team_corners()` (with a
hand-built `_gstate()` game-state multiplier), `total_corners_over()`,
`team_corner_tail()`, `player_markets()`, and `build_fixture(home, away, hp, ap,
agent_note)` which assembles the grouped market dict.

Team ratings are dicts: `{atk, def, catk, ccon, name}` where each rate is
relative to league average (1.0 = average). Player dicts:
`{name, g90, a90, sot90, fc90, fd90, min}`.

### `agent.py` (current)
`project(home, away, kickoff)` → calls OpenAI `/v1/responses` with `gpt-5.5` and
the hosted `web_search` tool, returns strict JSON:
```json
{"home":{"starters":[{"name":"...","min":90}],"out":["..."]},
 "away":{"starters":[...],"out":[...]}, "note":"..."}
```
`apply_minutes(player_pool, projection_side)` merges projected minutes into the
per-90 pool (out → dropped, unlisted → reduced minutes). Falls back to a mock
projection when `OPENAI_API_KEY` is unset. Model slug is overridable via
`OPENAI_MODEL`.

### `update.py` (current)
- `api(path)` → GETs `https://v3.football.api-sports.io{path}` with header
  `x-apisports-key`; prints `payload["errors"]` if present; returns
  `payload["response"]`.
- `team_rates(tid, cache)` → last 5 WC fixtures: goals for/against + corners
  for/against → rates relative to league avg.
- `player_pool(tid, cache)` → `/players` per team, per-90 rates, top 14 by goal
  involvement.
- `build_live()` → fetches fixtures by explicit UTC date for `range(4)` days
  (today + next 3), skips fixtures with undetermined teams (null ids), runs agent
  + model per fixture, caches to `cache.json`.
- `build_sample()` → illustrative demo fixtures (Spain/Uruguay, Brazil/NL).
- Env: `API_FOOTBALL_KEY`, `OPENAI_API_KEY`, `WC_LEAGUE_ID=1`, `WC_SEASON=2026`.

### Data source facts (verified)
- **API-Football** (api-sports.io, v3). The user is on the **PRO plan**:
  7,500 requests/day, 300 requests/minute, **all endpoints + all seasons**
  (the free tier was limited to seasons 2022–2024 and blocked 2026 — that was a
  past bug, now resolved by upgrading).
- World Cup 2026 = **`league=1`, `season=2026`**. Coverage confirmed true for:
  fixtures, lineups, statistics_fixtures (incl. corners), statistics_players,
  standings, players, injuries, predictions, odds.
- Knockout fixtures appear with **placeholder teams** ("Winner Group A", null
  ids) until the bracket resolves; the code skips these until both sides exist.
- **Always verify exact JSON field paths against a real API response** before
  relying on them — don't assume nested keys; print and inspect first.

---

## 4. Small fixes to include in this work (already agreed)

1. **Hide kicked-off/finished games.** In `build_live()`, filter fixtures to
   those not yet started. API-Football tags status at
   `fixture.status.short` — keep only `"NS"` (and arguably `"TBD"`/`"PST"`? no —
   keep `NS` only, since you can't place a pre-match bet otherwise). This stops
   last night's games lingering on the page.
2. **Sort fixtures by kickoff time** (soonest first) before writing `data.json`.
3. **Lineups must adjust TEAM strength, not just player props** (see §5.4).

Also currently the look-ahead window is `range(4)` (today + 3 days). Leave as is
unless the user asks; just make it a named constant `LOOKAHEAD_DAYS = 4`.

---

## 5. The accuracy build (this is the main task)

Overall strategy: **strong historical prior + small World-Cup update + measure
everything.** Build in the phases below, in order. Each phase should leave the
pipeline runnable and the site working.

### Phase 1 — Foundation (do first)

**1.1 Backtest & calibration harness** — build this BEFORE the modelling
changes so every later change can be evaluated.
- New file `backtest.py`. Pull a set of completed historical fixtures (use
  API-Football historical seasons — PRO has access; e.g. recent internationals
  and/or a past league season for volume). For each, compute the model's
  pre-match probabilities for the markets we support, then compare to actual
  outcomes.
- Metrics: **Brier score** and **log-loss** per market, plus **reliability /
  calibration curves** (bucket predictions by probability, check observed
  frequency matches). Print a summary table; optionally save calibration plots
  (matplotlib) to a `backtest_output/` folder.
- Use **walk-forward / time-based splits** (train on data before date T, test on
  matches after T) — never let the model see the match it's predicting. No
  look-ahead leakage.
- Provide a single command (`python backtest.py`) that runs it and prints
  results. This is the scoreboard for everything else.

**1.2 Team strength from historical Elo prior + time-decay.**
- Build an Elo-style international rating: ingest each team's historical results
  (goals-aware update, e.g. margin-adjusted Elo) over the last several years,
  weighting recent matches more (exponential time-decay, half-life ≈ 12–18
  months — make it a constant and let the backtest tune it). This does NOT need
  to be World-Cup-only; all international results count, recent weighted higher.
- Convert Elo (or a goal-supremacy estimate derived from it) into the model's
  `atk`/`def` rates so it feeds goals, over/under, result, BTTS, and corners.
- The current WC-only `team_rates` becomes a *small update* on top of the prior,
  not the whole signal. Implement as a shrinkage blend:
  `rating = w * historical_prior + (1 - w) * world_cup_form`, where `w` is high
  early (few WC games) and decreases as WC games accumulate. Let the backtest
  pick `w`'s schedule.
- Cache historical pulls aggressively (they don't change) to respect the
  7,500/day quota.

**1.3 Wire it together & re-baseline constants.** Re-fit `MU_GOALS`,
`LEAGUE_AVG_CORNERS`, `CORNER_R`, `H1_SHARE` from real data via the harness
rather than the current hand-set defaults. Record the before/after Brier scores
in a comment or the README so the improvement is documented.

### Phase 2 — Better inputs

**2.1 xG instead of raw goals as the rating signal.** Goals are noisy; xG
stabilises far faster, so it carries more signal per game. API-Football does not
provide clean historical xG, so pull from a free source — **Understat** or
**FBref** (via the `soccerdata` Python package, or direct scraping). Use xG
for/against to drive team attack/defence ratings where available, falling back
to goals when not. Add the dependency to the workflow. Respect each source's
rate limits and cache.

**2.2 Player-prop priors from club-season per-90s.** A striker's ~35-game club
season is a far better guide than 3 WC games. For each WC player, fetch their
**current club-season** per-90 rates (goals, assists, shots on target, fouls
committed, fouls drawn) — via API-Football club leagues (PRO covers them) or
FBref. Then shrink toward WC form the same way as teams:
`rate = w * club_season + (1 - w) * world_cup`. This makes "Vinícius to score"
reflect his club season, not three matches. Add opponent adjustment (e.g.
shots-on-target props scale up vs teams that concede many shots).

### Phase 3 — Proper modelling (likely spills past the tournament; fine)

**3.1 Bayesian hierarchical Poisson** for team ratings — partial pooling that
shrinks each team toward the population mean in proportion to data scarcity (the
statistically-correct version of the Phase-1 blend). PyMC or Stan. Gate behind
the backtest: only ship if it beats the Phase-1 model on Brier/log-loss.

**3.2 Monte Carlo simulation engine.** Simulate each match many thousands of
times from the fitted scoring intensities and read **every** market off the same
simulations. Benefits: internally-consistent prices across all markets, and —
the real prize — **same-game correlations** (a scorer prop is correlated with
over 2.5 and with the team winning; books often misprice these assuming
independence). This becomes the new `build_fixture` engine; markets are computed
as frequencies over simulations rather than closed-form.

---

## 6. `data.json` contract (keep the UI working)

The UI expects:
```json
{
  "updated": "ISO-8601 timestamp",
  "sample": false,
  "fixtures": [
    {
      "home": "Spain", "away": "Uruguay",
      "kickoff": "2026-06-27T00:00:00Z",
      "xg": [2.6, 0.9],
      "corners": [7.9, 4.8],
      "agent_note": "short string shown under the matchup",
      "groups": [
        { "name": "Match result",
          "markets": [ {"label": "Spain win", "p": 0.74}, ... ] },
        ...
        { "name": "Players",
          "markets": [ {"label": "...", "p": 0.31, "player": "..."}, ... ] }
      ]
    }
  ]
}
```
If you add fields (e.g. a confidence flag, or per-market historical-sample-size),
update `index.html` to render them. The UI does the implied-prob/EV/Kelly math
from the user-entered odds and each market's `p`; keep `p` a clean probability in
[0,1].

---

## 7. Constraints & gotchas

- **Quota:** PRO = 7,500 req/day, 300 req/min. Historical pulls are large —
  cache them to disk (they don't change) and only fetch new/changed data on each
  run. A daily run must stay well under the cap.
- **Secrets (GitHub → Settings → Secrets and variables → Actions):**
  `API_FOOTBALL_KEY`, `OPENAI_API_KEY`. Never hardcode keys; read from env. Repo
  is public (required for free Pages) — keys stay in encrypted Actions secrets.
- **Timezone:** the runner is UTC. Kickoffs near the UTC midnight boundary
  caused earlier bugs — fetch fixtures by explicit UTC date and rely on
  `fixture.status.short` for "started", not local date math.
- **Lineups land ~1h before kickoff.** The scheduled noon run gets *projected*
  XIs; a manual run near kickoff gets *confirmed* ones. Player props (and, after
  §5.4, team strength) should reflect whatever the agent returns.
- **Verify API field paths** against real responses before trusting them.
- **Keep the responsible-gambling footer** and the "estimates not certainties"
  honesty in the UI.
- **Don't add browser storage** to `index.html` (no localStorage); keep state in
  memory. (This is a constraint from the original build environment; on real
  Pages it's fine, but there's no need for persistence.)

## 5.4 (referenced above) Lineups → team strength
Today the agent's lineup only zeroes/reduces individual player props. Extend it
so missing/rested key players also dock the **team** attack/defence rating before
the goals and corners markets are computed. Approach: give each player an
attack/defence contribution weight (derivable from their club-season per-90s
from §2.2 — reuse that data), sum the projected XI's weights, and scale the team
rating by how much of its "normal" strength is on the pitch. Then a rested Yamal
ripples through result/over-under/corners, not just his own props. This pairs
naturally with §2.2 — build them together.

---

## 8. How to test your work

1. `pip install scipy` (plus any new deps: `pymc`, `soccerdata`, `matplotlib`).
2. **Sample mode:** with no env keys, `python update.py` must write a valid
   `data.json` with `"sample": true` and the page must render. Don't break this.
3. **Backtest:** `python backtest.py` prints Brier/log-loss per market and
   calibration; record numbers before vs after each modelling change so
   improvements are evidenced.
4. **Live mode** (needs keys): confirm fixture counts > 0, finished games are
   gone, fixtures sorted by kickoff, agent note present, player props respond to
   the projected XI, and team markets shift when key players are marked out.
5. Open `index.html` locally (`python -m http.server`) and verify every market
   group renders and odds entry produces VALUE/PASS + stake.

## 9. Definition of done for this round
- Phases 1 and 2 implemented; Phase 3 scaffolded or implemented if time allows.
- Backtest shows measurably better calibration than the pre-existing hand-set
  constants (report the delta).
- The three small fixes (§4) are in.
- Sample mode and the `data.json` contract still work; the site renders.
- New data sources/deps are added to `daily.yml` and documented in `README.md`.

---

## Appendix A — betting-domain rationale (so choices make sense)
- You beat a bookmaker by being **better calibrated than the de-vigged line on a
  specific selection**, not by predicting winners. De-vig = divide each
  outcome's implied prob (1/odds) by the overround so they sum to 1.
- **Kelly** stake fraction = edge ÷ (odds − 1); the UI applies a user-set
  fractional multiplier (default 0.25) for safety.
- **Closing-line value** is the best forward test of edge: if the market price
  moves toward your side by kickoff, you're probably right even before results.
  Consider logging model picks vs closing odds over time to measure this.
- Three games per team is tiny; that's why priors + shrinkage + time-decay are
  the core of this work.

## Appendix B — history of decisions made with the user
- Chose a hybrid: statistical model for all numbers, LLM agent only for lineups.
- Chose API-Football PRO ($19/mo, no auto-renew) over Sportmonks/TheStatsAPI
  after comparison — cheapest that covers corners + player stats + WC 2026; the
  pricier options' extra depth (xG, multi-book odds) isn't needed yet and the
  free options (football-data.org) lack corners/player stats.
- Switched the agent from Anthropic to OpenAI (gpt-5.5, Responses API + web
  search) at the user's request.
- Fixed earlier: workflow file path, public-repo requirement for Pages, the
  RapidAPI-vs-direct key confusion, the UTC date-window bug, and the free-tier
  season block.
- User is not a football expert and has delegated football-domain judgment to
  the implementer. Keep the honesty about un-backtested edges.

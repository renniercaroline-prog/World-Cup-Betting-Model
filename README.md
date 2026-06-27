# WC Edge

A phone-friendly web page showing model probabilities for World Cup markets, refreshed daily. No server, free to run. Your friend types in the odds his book offers and the page flags positive-value bets and suggests a stake.

## Markets covered
- **Match result**, **total goals** — both **over and under** each of 0.5–4.5, **both teams to score**
- **First-half result**, **half-time / full-time**
- **Corners** — total **over and under** each line, plus per-team thresholds
- **Players** — to score, score-or-assist, shot on target, foul committed, fouled

## Files
- **model.py** — all the probability math. Poisson goals, Negative-Binomial corners, per-90 player props. No AI, no network. Calibration constants (`MU_GOALS`, `LEAGUE_AVG_CORNERS`, `CORNER_R`, `H1_SHARE`) are env-overridable so the backtest can A/B re-fitted values.
- **elo.py** — historical team-strength prior. A time-decayed, margin-adjusted Elo over each team's past international results, mapped to attack/defence rates. The sparse World-Cup form is shrunk toward this prior (`rating = w·prior + (1−w)·form`), so a team is anchored by years of history, not 3 games. Also ships a 2D attack/defence variant (`GoalEloEngine`) that separates scoring and conceding — better on data-rich **club** football, but the international backtest showed plain 1D wins when teams play few games, so the live (international) pipeline uses 1D. Can update from xG instead of goals (Phase 2.1). No network.
- **xg.py** — optional expected-goals team ratings from FBref (via `soccerdata`), used in place of raw goals where available and falling back to goals otherwise. Best-effort and cached; the pipeline never breaks if xG is missing.
- **agent.py** — the AI layer. Uses an LLM + web search to project each team's lineup and minutes from the latest news. It returns lineups and minutes only — it never produces a probability.
- **update.py** — runs daily: fetch data → build the Elo prior → ask the agent for lineups → run the model → write `data.json`. Missing/rested key players now also dock **team** attack/defence, not just their own props.
- **backtest.py** — the scoreboard. Walk-forward Brier / log-loss + calibration per market. Runs on real history (with a key) or on synthetic data (no key). This is how every modelling change is judged.
- **index.html** — the page your friend opens. Reads `data.json`; shows sample data until the first live run.
- **.github/workflows/daily.yml** — runs it every day at 12:00 UTC.

## Backtest & calibration (earn the edge before trusting it)
`python backtest.py` prints Brier score, log-loss and a reliability table per market.
With no API key it runs in **synthetic mode** against a known true model — useful for
checking the harness and the *relative* effect of a change; real numbers need a key
(`BACKTEST_MODE=live BACKTEST_LEAGUE=39 BACKTEST_SEASONS=2022,2023 python backtest.py`).

Swap the rating method on the same scoreboard with `RATING_MODEL=baseline|elo`. The
Phase-1 change (Elo prior + shrinkage) was verified on **real Premier League data**
(708 walk-forward matches, 2022–23):

| market        | baseline Brier (skill%) | Elo Brier (skill%) |
|---------------|:-----------------------:|:------------------:|
| result (1X2)  |      0.666 (−5.2%)      |  **0.614 (+3.1%)** |
| over 2.5      |      0.278 (−14.6%)     |   0.247 (−1.7%)    |
| both-teams    |      0.281 (−14.2%)     |   0.253 (−3.1%)    |
| corners 8.5   |      0.231 (−9.6%)      |   0.214 (−1.4%)    |

The old baseline is negative-skill on every market (over-confident, worse than the base
rate). The Elo prior turns the result market positive and roughly halves the deficit
elsewhere, with calibration going from wild to near-diagonal. Synthetic mode (no key)
reproduces the same pattern offline. Re-baseline constants from real data with
`python backtest.py --recalibrate` (prints data-fitted `MU_GOALS` / corners / dispersion / rho).

### Validated on international football (the actual domain)
The numbers above are club football. The same harness, pointed at a multi-confederation
basket of **3,061 senior international matches** (World Cup, Euro, Copa, AFCON, Asian Cup,
Gold Cup, Nations Leagues, WC qualifiers, friendlies; pass the league IDs as a comma list
in `BACKTEST_LEAGUE`), confirms the prior transfers — and transfers *better*:

| model     | result Brier | skill% |
|-----------|:------------:|:------:|
| baseline  |    0.648     | −2.0%  |
| **elo (1D)** | **0.591** | **+6.9%** |
| elo2d (2D)|    0.595     | +6.3%  |

Two findings drove live defaults: (1) the 1D Elo prior gives **+6.9% skill on the result
market** for internationals (stronger than the +3.1% on the PL); (2) the **2D** model that
won on club football **loses** here — national teams play too few games to estimate
separate attack and defence reliably — so the live pipeline uses **1D**. A reminder that a
gain measured in the wrong domain doesn't always hold in the right one.

### Data coverage of the WC squads
Audited 276 key players across 35 WC squads: they play in **55 club leagues across 42
countries**. Player-prop priors (club-season per-90s via API-Football) cover all of them;
~**53%** are in a big-5 European league, which is also where FBref **xG** is available — so
xG enrichment is big-5-skewed while the core player data is genuinely worldwide.

## Better inputs (Phase 2)
- **xG drives team ratings.** Goals are noisy; xG stabilises faster, so attack/defence
  ratings use season xG-for/against where a free source has it (FBref via `soccerdata`),
  falling back to goals otherwise. In synthetic backtests an xG-driven prior beat a
  goals-driven one on the result market (Brier 0.635 → 0.612, skill 2.4% → 5.8%).
- **Player props anchored to the club season.** A striker's ~35-game club season is a far
  better guide than ≤3 World Cup games, so each WC per-90 (goals, assists, shots on target,
  fouls) is shrunk toward the player's club-season per-90: `rate = w·club + (1−w)·WC`, with
  `w` high when WC minutes are few. Shot-on-target props also scale with the **opponent's**
  shots-conceded rate (a leaky defence → more SoT chances).
- This also sharpens the lineup→team-strength dock (the per-90s the dock sums are now the
  more-stable club-season numbers). Local install for the backtest plots: `pip install matplotlib`.

## Dixon-Coles (implemented, but backtested OFF by default)
The goals model has a **Dixon-Coles** low-score correction (`RHO_DC`) and reads
match-result, totals, BTTS, half and HT/FT markets off one corrected score matrix, so
prices stay mutually consistent. In theory a small negative rho lifts draws/0-0/1-1.
**But** fitting rho by exact-scoreline likelihood on real Premier League data did not
support a nonzero value (NLL was flat-to-worse as rho went negative), so `RHO_DC`
defaults to **0** (plain Poisson) rather than shipping an un-evidenced edge. The
machinery and the fit (`python backtest.py --recalibrate`) stay, so you can switch it
on per competition if the data ever justifies it.

## A note on home advantage (deliberately omitted)
The goals model has **no home-advantage term**, on purpose: World Cup matches are played
at neutral venues, so a home edge would bias the predictions the product actually makes.
The side-effect is that the **club-league backtest understates the model** — home
advantage is large in domestic leagues, so real-PL Brier looks worse than the model
would do on neutral-venue internationals. If you ever retarget this at club football,
add a home multiplier first; it matters more than Dixon-Coles.

## How the agent fits (and its one real limit)
Player props depend on who actually starts. Confirmed lineups only appear ~1 hour before kickoff, so the noon run uses the agent's *projection* from team news and recent starts, not a confirmed sheet. Re-run the workflow manually closer to kickoff (Actions → Run workflow) for sharper player numbers. The agent's projection feeds minutes into the model; a player it marks "out" drops off entirely.

## Deploy (~10 min)
1. **API-Football key** — sign up at api-sports.io (free, 100 req/day), copy the key.
2. **OpenAI key** — from platform.openai.com, for the lineup agent (uses gpt-5.5 via the Responses API). (Optional: without it, player props use fallback minutes and everything else still works.)
3. **Create a repo** and upload these files (keep `.github/workflows/daily.yml` in that path).
4. **Add secrets** — Settings → Secrets and variables → Actions → New repository secret. Add `API_FOOTBALL_KEY` and `OPENAI_API_KEY`.
5. **Pages needs a public repo on the free plan** — either make the repo public (your keys live in encrypted secrets, not in any file, so they stay private), or host the page on Cloudflare Pages from a private repo. Then Settings → Pages → branch `main` / root.
6. **Run once** — Actions → Daily update → Run workflow. Refresh the page for live numbers.

## Run locally first
```bash
pip install scipy
python update.py        # writes sample data.json, no keys needed
python -m http.server   # open http://localhost:8000
```

## Tune before trusting an edge
`MU_GOALS`, `LEAGUE_AVG_CORNERS`, `CORNER_R` and the per-90 player rates are starting points. Fit them on real data and **backtest against closing odds** before believing any flagged value. The structure is sound; the edge has to be earned with calibration.

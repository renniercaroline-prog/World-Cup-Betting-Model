# World Cup Betting Model

A self-updating web app that predicts how likely each outcome of a 2026 World Cup match is
(who wins, how many goals, how many corners, which players score, and more), compares those
predictions to the prices bookmakers are offering, and flags when a price looks too generous,
all while honestly measuring whether the predictions are actually any good.

It runs with no server and no database. A scheduled GitHub Action runs a Python pipeline that
gathers data, runs a statistical model, fetches live odds, and saves the results; a single web
page displays them. Hosted free on GitHub Pages.

---

## First, a 90-second primer (no betting knowledge needed)

If you're not into betting, here are the only ideas you need to follow the rest of this README:

- **Odds** are just a price. Decimal odds of 2.50 mean that if you bet £1 and win, you get
  £2.50 back. A bigger number means the bookmaker thinks it's less likely.
- **Odds imply a probability.** Odds of 2.50 imply a `1 ÷ 2.50 = 40%` chance, so every price is
  really the bookmaker saying "I think this is 40% likely."
- **A "value" bet** is when your estimate of the chance is *higher* than the price implies. If
  the model thinks something is 50% likely but the bookmaker is pricing it as 40%, you're being
  offered a better deal than the true odds. That gap is an edge.
- **Why this is hard.** Bookmakers are very good at pricing and they bake in a profit margin, so
  most people lose. To win you don't need to predict winners; you need to be *better than the
  bookmaker's price* on specific bets. That's a high bar.
- **The honest test (used throughout).** A prediction looking good isn't proof. The real test is
  whether the model beats the *closing price*, the odds right before kickoff, which is the
  sharpest the market ever is. If the price drifts toward your side after you bet, you were
  probably right. This is called **closing-line value (CLV)**, and the project tracks it.

In short: estimate the true chances better than the price, and prove it with CLV before risking
anything.

---

## The core problem it solves

A team plays only 3 games in the World Cup group stage, and anywhere from 3 to 8 in total
depending on how far they go. Judging a team on a handful of games is noisy: especially early
in the tournament, one fluke result can throw your estimate well off. So the whole approach is:

> start from a strong estimate built on years of history, then nudge it with the few World Cup
> games as they accumulate, and measure everything.

The sparse tournament data is a small adjustment on top of a stable, long-history baseline, and
that adjustment grows as a team plays more games. Every change to the model is checked against
real results rather than assumed to help.

---

## How it's built (the moving parts)

```
 Data sources ─┐
 (results,     ├─►  one daily pipeline  ─►  predictions saved to a file  ─►  the web page
  stats, odds, │     (Python)                (data.json)                     (index.html)
  team news)  ─┘
                      │
                      └─ a separate "scoreboard" tests how good the predictions are
```

| File | What it does |
|---|---|
| `model.py` | The maths that turns team and player strength into probabilities for every market (match result, goals, corners, players, and so on). Pure functions, no network. |
| `elo.py` | Rates how strong each team is, from years of past results, weighting recent games more. |
| `agent.py` | The AI layer. Reads the latest team news and predicts the line-up. It never outputs a probability (details below). |
| `xg.py` | Optional: uses expected goals (a more stable measure of quality than raw goals) where a free source has it. |
| `update.py` | The orchestrator. Runs everything in order, fetches odds, and saves the day's predictions. |
| `backtest.py` | The scoreboard. Replays thousands of past matches to score how accurate the model is. |
| `clv.py` | Tracks whether the model beats the closing price (the honest test of a real edge). |
| `index.html` | The web page people use. Vanilla JavaScript, no framework, no build step. |
| `daily.yml` | The GitHub Action that runs the whole thing on a schedule, for free. |

---

## How the model makes its predictions

- **Team strength from history.** Each team gets an Elo rating (like chess) built from years of
  international results: beating strong teams by big margins raises it most, and recent games
  count more than old ones (a 15-month half-life time decay). The handful of World Cup games
  then make a small adjustment on top via a shrinkage blend, `rating = w·prior + (1−w)·form`,
  where the weight on the history `w` is high early and falls as more games are played, rather
  than the few WC games being trusted on their own.
- **Goals.** From the two teams' strengths the model computes each side's expected goals and
  builds a full grid of every possible scoreline (a bivariate Poisson with an optional
  Dixon-Coles low-score correction). Every goals-based market (match result, totals, both teams
  to score, correct score, halves, half-time/full-time, and more) is read off that one grid, so
  all the predictions stay consistent with each other.
- **Corners** use a negative-binomial model, which fits the way corner counts actually spread
  out in real games better than a simple Poisson.
- **Player bets** (to score, shots on target, fouls) start from each player's whole club season
  (~35 games, far more reliable than 3 World Cup games), shrink toward sparse WC form, and
  adjust for the opponent.

It prices **20 groups of markets per game**, from match result and goals down to individual
player bets.

---

## The AI agent, and why it's kept on a tight leash

The project uses a large language model, but under one firm architectural rule:

> The LLM never decides a probability. Every number a user could bet on comes from the
> statistical model. The agent's only job is to read the latest team news and return, as strict
> JSON, each side's projected starting XI, minutes, and who's ruled out.

Why the restriction: LLMs are strong at reading and summarising but unreliable at numbers, and
they sound confident even when guessing, so letting one set a betting probability would be the
wrong tool for the job. It's pointed at the task it's genuinely good at, reading team news and
projecting a line-up, and the calibrated maths is left to the model.

**Technical details:**
- Built on the **OpenAI Responses API** (GPT-5.5) with the hosted **web-search** tool, so it
  pulls current injury and team news at request time. It's prompted to return strict JSON only,
  which the pipeline parses directly into projected minutes.
- **How the line-up feeds the model:** projected minutes scale each player's per-90 rate into a
  prop probability (a player marked "out" drops off entirely), and a rested or missing key
  player also docks the team's attack/defence rating, so a benched star ripples through the
  result, totals, and corners markets, not just his own props.
- **Cost control (the agent is the only pay-per-call component):** it's only invoked when a
  fixture is within a configurable window of kickoff (`AGENT_WINDOW_HOURS`, default 3h, when
  line-ups actually confirm), and a cached projection younger than `AGENT_REFRESH_MIN`
  (default 60m) is reused instead of re-querying on every run. This cut LLM calls by roughly
  **90%** versus a naive per-fixture-per-run approach, while making projections sharper because
  they're made closer to the confirmed XI.
- **Graceful degradation:** with no API key the agent returns mock minutes and the rest of the
  pipeline runs unchanged, so the AI layer is an enhancement, never a dependency.

---

## Proving the predictions are any good (not just assuming)

The core engineering principle here: a flagged bet is worthless until it's measured. Two tools
do the measuring.

**1. The backtest (`backtest.py`), a scoreboard for accuracy.** It replays thousands of finished
matches, makes a prediction for each using only information available *before* that match (no
look-ahead leakage), and scores how close the predictions were (Brier score, log-loss, and
calibration curves). On **708 real Premier League matches**, switching from a naive
"last few games" approach to the history-based model flipped it from worse than guessing to
genuinely useful, and made its confidence honest (when it says 70%, it happens about 70% of the
time). On **3,000+ real international matches**, the improvement was even larger (+6.9% skill on
the match-result market).

Some honest decisions that came out of the testing, each showing the value of measuring rather
than guessing:
- A two-dimensional "attack vs defence" rating beat the simple one-number version on *club*
  football, but *lost* on international football, because national teams play too few games to
  estimate two parameters reliably. So the live app uses the simpler, more robust version.
- A well-known low-scoring-games correction (Dixon-Coles) is implemented but turned **off**,
  because fitting it on real data did not support it. Better to ship nothing than a made-up edge.
- Home advantage is deliberately left out, since World Cup games are at neutral venues.

**2. Closing-line value (`clv.py`), the real test of edge.** For every bet the model flags, it
records the price when flagged, rolls the price right up to kickoff, and checks whether the
market moved toward the model's side. Consistently beating that closing price is the strongest
sign a model genuinely has an edge, and it shows up in far fewer bets than waiting to see
profit. The web page shows this live and is blunt about it: a good-looking profit on a small
number of bets is luck, and the closing-line number is the signal to trust.

> **Current honest status:** the model is well-calibrated and the history-based approach
> measurably helps, but it is not yet proven to beat the market. By design, it's a
> "watch the data, don't bet real money yet" situation.

---

## What the user sees and does

The web page is for someone placing bets. They:
1. Enter their bankroll (how much they're playing with) and a safety setting (a fractional
   Kelly multiplier that controls stake size).
2. See a ranked list of the model's best value bets, plus a "best value" summary per game.
3. Type in the price their own bookmaker is offering on any market, and the page instantly says
   whether it's a good deal (**VALUE**) or not (**PASS**) and suggests a stake.

Behind that, the app automatically fetches live bookmaker prices, line-shops the best available
price across betting sites, removes the bookmaker's margin to estimate the fair price, and
computes the expected value and a recommended stake.

---

## Tech stack

- **Modelling:** Python (`scipy`/`numpy`), covering the probability distributions, the Elo
  rating system, the history-vs-recent-form shrinkage, and the accuracy metrics.
- **AI:** OpenAI Responses API (GPT-5.5 with hosted web search), strictly bounded to line-up
  projection.
- **Data:** API-Football for results, stats, and live odds; FBref for expected goals.
- **Frontend:** a single hand-written HTML/JavaScript file, no framework and no build step.
- **Infra:** GitHub Actions (cron plus manual trigger) is the entire backend, and GitHub Pages
  hosts the page. No server and no database; state lives in files committed to the repo. Secrets
  are stored as encrypted Actions secrets, never hardcoded.

---

## Running it

**On your own computer (no accounts needed, produces sample data):**
```bash
pip install scipy
python update.py        # creates a sample predictions file
python -m http.server   # then open http://localhost:8000 in a browser
```

**Backtest:**
```bash
python backtest.py                                  # synthetic data, baseline ratings
RATING_MODEL=elo python backtest.py                 # swap in the history-based model
python backtest.py --recalibrate                    # re-fit the constants from data
BACKTEST_MODE=live BACKTEST_LEAGUE=39 BACKTEST_SEASONS=2022,2023 python backtest.py
```

**Live:** add an API-Football key (and optionally an OpenAI key) to the repo's GitHub Actions
secrets, switch on GitHub Pages, and the scheduler keeps the page updated on its own. The person
using it just opens the link, with no setup on their end.

---

## What this project demonstrates

- **Making good predictions when data is scarce:** leaning on history and blending it with recent
  information, the way a careful forecaster (or a Bayesian statistician) would.
- **Using AI responsibly:** pointing a language model at the one task it's reliable for, keeping
  it cheap with kickoff-gating and caching, and never letting it touch the numbers that matter.
- **Trusting evidence over intuition:** every modelling choice is checked against real results,
  and the honest answer (including "this idea didn't work, so it's off") is documented.
- **Building end-to-end:** data, model, live odds, web page, and automation, plus a built-in way
  to keep checking over time whether it actually works.

# World Cup Betting Model

A self-updating web app that predicts how likely each outcome of a 2026 World Cup match is
(who wins, how many goals, how many corners, which players score…), compares those
predictions to the prices bookmakers are offering, and flags when a price looks **too
generous** — all while honestly **measuring whether the predictions are actually any good.**

It runs with no server and no database: a scheduled GitHub Action runs a Python pipeline that
gathers data, runs a statistical model, fetches live odds, and saves the results; a single web
page displays them. Hosted free on GitHub Pages.

---

## First, a 90-second primer (no betting knowledge needed)

If you're not into betting, here are the only ideas you need to follow the rest of this README:

- **Odds** are just a price. "Decimal odds of 2.50" means *if you bet £1 and win, you get
  £2.50 back*. Bigger number = the bookmaker thinks it's less likely.
- **Odds imply a probability.** Odds of 2.50 imply a `1 ÷ 2.50 = 40%` chance. So every price
  is really the bookmaker saying "I think this is 40% likely."
- **A "value" bet** is when *your* estimate of the chance is **higher** than the price implies.
  If the model thinks something is 50% likely but the bookmaker is pricing it as 40%, you're
  being offered a better deal than the true odds — that's an edge.
- **Why this is hard:** bookmakers are very good at pricing, and they bake in a profit margin,
  so *most* people lose. To win you don't need to predict winners — you need to be **better
  than the bookmaker's price** on specific bets. That's a high bar.
- **The honest test (used throughout):** a prediction looking good isn't proof. The real test
  is whether the model **beats the closing price** — the odds right before kickoff, which is
  the sharpest the market ever is. If the price drifts toward your side after you bet, you were
  probably right. This is called **closing-line value (CLV)**, and the project tracks it.

That's the whole game: **estimate the true chances better than the price, and prove it with
CLV before risking anything.**

---

## The core problem it solves

A football team only plays ~3 games at a World Cup. Trying to judge a team from 3 games is
hopelessly noisy — one fluke result and your estimate is way off. So the whole approach is:

> **start from a strong estimate built on years of history, then nudge it slightly with the few
> World Cup games — and measure everything.**

The sparse tournament data is a small adjustment on top of a stable, long-history baseline, and
every change to the model is checked against real results rather than assumed to help.

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

| File | What it does, in plain terms |
|---|---|
| `model.py` | The maths that turns team/player strength into probabilities for every market (match result, goals, corners, players, etc.). |
| `elo.py` | Rates how strong each team is, based on years of past results, weighting recent games more. |
| `agent.py` | The **AI assistant** — reads the latest team news and predicts the line-up (who's playing). *It never sets a probability* (more below). |
| `xg.py` | Optional: uses "expected goals" (a better measure of team quality than raw goals) where available. |
| `update.py` | The conductor — runs everything in order and saves the day's predictions + odds. |
| `backtest.py` | The **scoreboard** — replays thousands of past matches to score how accurate the model is. |
| `clv.py` | Tracks whether the model **beats the closing price** (the honest test of a real edge). |
| `index.html` | The web page people actually use. |
| `daily.yml` | The automation that runs the whole thing on a schedule, for free. |

---

## How the model makes its predictions

- **Team strength from history.** Each team gets a rating (an "Elo" rating, like chess) built
  from years of international results — winning big against strong teams raises it most, and
  recent games count more than old ones. The handful of World Cup games then make a *small*
  adjustment on top, rather than being trusted on their own.
- **Goals.** From the two teams' strengths, the model works out how many goals each is likely
  to score and builds a grid of every possible scoreline. **Every goals-based market** (who
  wins, total goals, both teams to score, correct score, half-by-half, etc.) is read off that
  one grid, so all the predictions are consistent with each other.
- **Corners** use a separate model suited to how corners actually bunch up in real games.
- **Player bets** (to score, shots, fouls…) start from each player's **whole club season**
  (~35 games — far more reliable than 3 World Cup games) and adjust for who they're playing.

It prices **20 groups of markets per game**, from match result and goals down to individual
player bets.

---

## The AI assistant — and why it's kept on a tight leash

The project uses an AI language model (the kind behind ChatGPT) — but under one firm rule:

> **The AI never decides a probability.** Every number you could bet on comes from the
> statistical model. The AI's *only* job is to read the latest team news and report back
> **who's likely to start, for how long, and who's injured.**

Why the restriction? Language models are great at reading and summarising, but they're
unreliable at numbers and tend to sound confident even when they're guessing. Letting one
invent a betting probability would be the wrong tool for the job. So it's pointed at the thing
it's genuinely good at — *reading the news and predicting the line-up* — and the trustworthy
maths is left to the statistical model.

**What the line-up is used for:** a star player being rested or injured lowers his own bet
chances **and** weakens his whole team's rating, which ripples through the match-result and
goals predictions too.

**Keeping it cheap:** the AI is the one part that costs money per use, so it only runs in the
**last few hours before kickoff** (when line-ups actually get confirmed) and remembers its
answer instead of re-asking every time — cutting its usage by roughly **90%** while making the
predictions *sharper* (closer to the real, confirmed line-up). The app works fine without it.

---

## Proving the predictions are any good (not just assuming)

The most important engineering principle here: **a flagged bet is worthless until it's
measured.** Two tools do the measuring.

**1. The backtest (`backtest.py`) — a scoreboard for accuracy.** It replays thousands of
finished matches, makes a prediction for each using *only information available before that
match* (no cheating with hindsight), and scores how close the predictions were to what actually
happened. Tested on **708 real Premier League matches**, switching from a naive "last few games"
approach to the history-based model flipped it from **worse than guessing** to genuinely
useful, and made its confidence levels honest (when it says 70%, it happens ~70% of the time).
On **3,000+ real international matches**, the improvement was even bigger.

**Some honest decisions that came out of the testing** (each one shows the value of measuring
instead of guessing):
- A fancier two-number "attack vs defence" rating beat the simple one on *club* football — but
  *lost* on international football, because national teams play too few games to support it. So
  the live app uses the simpler, more robust version. *A trick that works in one setting can
  fail in the one that matters.*
- A well-known statistical tweak for low-scoring games ("Dixon-Coles") is built in but **turned
  off**, because the real data didn't support it — better to ship nothing than a made-up edge.
- "Home advantage" is deliberately left out, because World Cup games are at neutral stadiums.

**2. Closing-line value (`clv.py`) — the real test of edge.** For every bet the model flags, it
records the price at the time, then the price right before kickoff, and checks whether the
market moved toward the model's side. Consistently beating that closing price is the strongest
sign a model genuinely has an edge — and it shows up in far fewer bets than waiting to see
profit. The web page shows this live, and is blunt about it: **a good-looking profit on a small
number of bets is luck; the closing-line number is the signal to trust.**

> **Current honest status:** the model is well-calibrated and the history-based approach
> measurably helps — but it is **not yet proven to beat the market.** It's a "watch the data,
> don't bet real money yet" situation by design.

---

## What the user sees and does

The web page is for someone placing bets. They:
1. Enter their **bankroll** (how much they're playing with) and a safety setting.
2. See a ranked list of the model's **best value bets**, and a "best value" summary per game.
3. Type in **the price their own bookmaker is offering** on any market — and the page instantly
   tells them whether it's a good deal (**VALUE**) or not (**PASS**), and suggests a sensible
   stake size.

Behind that, the app automatically fetches live bookmaker prices, picks the **best available
price** across betting sites, strips out the bookmaker's margin to estimate the "fair" price,
and works out the value and a recommended stake.

---

## Tech stack

- **Modelling:** Python (`scipy`/`numpy`) — probability distributions, the Elo rating system,
  the history-vs-recent-form blending, and the accuracy scoring.
- **AI:** OpenAI's API (with live web search), strictly limited to predicting line-ups.
- **Data:** API-Football for results, stats and live odds; FBref for expected-goals.
- **Web page:** a single hand-written HTML/JavaScript file — no framework, no build step.
- **Hosting & automation:** GitHub Actions (a free scheduler) is the *entire* backend, and
  GitHub Pages hosts the page. No server, no database — everything lives in files in the repo.

---

## Running it

**On your own computer (no accounts needed — produces sample data):**
```bash
pip install scipy
python update.py        # creates a sample predictions file
python -m http.server   # then open http://localhost:8000 in a browser
```

**Live:** add an API-Football key (and optionally an OpenAI key) to the repo's GitHub
*Actions secrets*, switch on GitHub Pages, and the scheduler keeps the page updated on its own.
The person using it just opens the link — no setup on their end.

---

## What this project demonstrates

- **Making good predictions when data is scarce** — leaning on history and blending it with a
  little recent information, the way a careful forecaster (or a Bayesian statistician) would.
- **Using AI responsibly** — pointing a language model at the one task it's reliable for,
  keeping it cheap, and never letting it touch the numbers that matter.
- **Trusting evidence over intuition** — every modelling choice is checked against real results,
  and the honest answer (including "this idea didn't work, so it's off") is documented.
- **Building the whole thing end-to-end** — data → model → live odds → web page → automation,
  plus a built-in way to keep checking, over time, whether it actually works.

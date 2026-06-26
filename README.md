# WC Edge

A phone-friendly web page that shows model probabilities for World Cup markets
(goals, half-time leader, team corners). Your friend types in the odds his
sportsbook offers; the page flags positive-value bets and suggests a stake.
It refreshes itself once a day — no server, free to run.

## How it fits together
- **index.html** — the page he opens. Reads `data.json`. Falls back to sample data if that file isn't there yet.
- **update.py** — the model. Pulls fixtures + corner/goal stats from API-Football, computes probabilities, writes `data.json`. With no API key it writes clearly-labelled sample data so you can see everything working first.
- **.github/workflows/daily.yml** — runs `update.py` every day at 12:00 UTC and commits the fresh `data.json`.

## Deploy in ~10 minutes
1. **Get a key**: sign up at api-sports.io (free plan, 100 requests/day) and copy your API key.
2. **Create a GitHub repo** and upload these files.
3. **Add the key as a secret**: repo → Settings → Secrets and variables → Actions → New repository secret → name `API_FOOTBALL_KEY`, paste the key.
4. **Confirm the league id**: call `https://v3.football.api-sports.io/leagues?search=world cup` with your key, find the World Cup id, and set `WC_LEAGUE_ID` in `daily.yml` if it isn't `1`.
5. **Turn on Pages**: repo → Settings → Pages → deploy from branch `main`, root. Your friend bookmarks the URL it gives you.
6. **Run it once**: Actions tab → Daily update → Run workflow. After it finishes, the page shows live numbers.

## Try it locally first
```bash
pip install scipy
python update.py          # writes sample data.json (no key needed)
python -m http.server     # open http://localhost:8000
```

## Things to tune as you go
- `MU_GOALS`, `LEAGUE_AVG_CORNERS`, `CORNER_R` in `update.py` are starting constants. Fit them on real data and **backtest against closing odds** before trusting any flagged edge.
- The game-state corner adjustment (`gstate`) is a simple hand-set curve. It's the right place to encode "the team that's losing chases and wins more corners."
- Injuries/lineups aren't in the model yet. That's the natural next layer — and the one place an LLM agent earns its keep: read team news, nudge a team's corner rate down when its set-piece taker is out, then hand the numbers back to the model.

## Honest limits
Three group games per team is very little data, so early ratings are noisy. The structure is sound; the edge is only real once it's calibrated and backtested. This is a tool for disciplined decisions, not a guaranteed profit.

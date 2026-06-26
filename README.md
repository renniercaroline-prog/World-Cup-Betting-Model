# WC Edge

A phone-friendly web page showing model probabilities for World Cup markets, refreshed daily. No server, free to run. Your friend types in the odds his book offers and the page flags positive-value bets and suggests a stake.

## Markets covered
- **Match result**, **total goals** (over 0.5–4.5), **both teams to score**
- **First-half result**, **half-time / full-time**
- **Corners** — total over/under and per-team lines
- **Players** — to score, score-or-assist, shot on target, foul committed, fouled

## Files
- **model.py** — all the probability math. Poisson goals, Negative-Binomial corners, per-90 player props. No AI, no network.
- **agent.py** — the AI layer. Uses an LLM + web search to project each team's lineup and minutes from the latest news. It returns lineups and minutes only — it never produces a probability.
- **update.py** — runs daily: fetch data → ask the agent for lineups → run the model → write `data.json`.
- **index.html** — the page your friend opens. Reads `data.json`; shows sample data until the first live run.
- **.github/workflows/daily.yml** — runs it every day at 12:00 UTC.

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

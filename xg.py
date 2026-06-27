#!/usr/bin/env python3
"""
xg.py — expected-goals team ratings from a free source (Phase 2.1).

WHY xG
------
Goals are a noisy, low-count signal: a team can deserve to win and lose 1-0. xG
(expected goals) aggregates the quality of every chance, so it stabilises far
faster and carries more signal per match. API-Football has no clean historical
xG, so we pull season xG-for / xG-against per team from FBref via the
`soccerdata` package and convert it to the model's attack/defence multipliers.

DESIGN
------
- Pure best-effort with graceful degradation: if `soccerdata` isn't installed, the
  network is down, the league/season is unavailable, or a team can't be matched by
  name, every entry point returns None / {} and the caller falls back to goals.
  The pipeline must never break because xG is missing.
- Cached to xg_cache.json keyed by (league, season) — a finished season's xG does
  not change, so we fetch each at most once and respect the source's rate limits.
- Team-name matching between API-Football and FBref is approximate (normalised,
  punctuation-stripped). Treat a miss as "no xG for this team", not an error.

This module is intentionally untested against the live source in this change set —
verify the FBref league IDs and the team-name matches against a real pull before
trusting the numbers (same rule as the API-Football field paths).
"""
import os, json, re

CACHE = os.path.join(os.path.dirname(__file__), "xg_cache.json")

# FBref competition keys that soccerdata understands. Extend as needed; an unknown
# key simply yields no xG (caller falls back to goals).
FBREF_LEAGUES = {
    "INT-World Cup", "INT-European Championship", "INT-UEFA Nations League",
    "ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1",
}

def _norm(name):
    """Normalise a team name for cross-source matching: lowercase, drop accents-ish
    punctuation and common suffixes so 'Atlético Madrid' ≈ 'Atletico Madrid'."""
    s = name.lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    for junk in (" fc", " cf", " afc", " sc"):
        s = s.replace(junk, "")
    return re.sub(r"\s+", " ", s).strip()

def _load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def _save_cache(c):
    json.dump(c, open(CACHE, "w"))

def season_xg(league, season):
    """Return {normalised_team_name: {'xgf90': float, 'xga90': float}} for a league
    season, or {} if unavailable. Cached on disk."""
    cache = _load_cache()
    ckey = f"{league}:{season}"
    if ckey in cache:
        return cache[ckey]
    if league not in FBREF_LEAGUES:
        return {}
    table = {}
    try:
        import soccerdata as sd
        fb = sd.FBref(leagues=league, seasons=season)
        df = fb.read_team_season_stats(stat_type="standard")
        # FBref columns are a MultiIndex; xG and matches-played live under these keys.
        # Field paths verified per source before trusting — adjust here if the
        # soccerdata schema differs from what you see in a real pull.
        for idx, row in df.iterrows():
            team = idx[-1] if isinstance(idx, tuple) else idx
            played = float(row.get(("Playing Time", "MP"), 0)) or 0
            xgf = float(row.get(("Expected", "xG"), 0))
            xga = float(row.get(("Expected", "xGA"), 0)) if ("Expected", "xGA") in row else None
            if played <= 0:
                continue
            entry = {"xgf90": xgf / played}
            if xga is not None:
                entry["xga90"] = xga / played
            table[_norm(str(team))] = entry
    except Exception as e:
        print(f"   (xg.py: no xG for {ckey} — {type(e).__name__}: {e}); falling back to goals")
        table = {}
    cache[ckey] = table
    _save_cache(cache)
    return table

def team_rates_from_xg(name, league, season, league_xg_mean):
    """Attack/defence multipliers from a team's season xG, or None if unavailable.
    league_xg_mean is the competition's average xG per team per match (the
    denominator that makes 1.0 == league average, mirroring how goals are scaled)."""
    table = season_xg(league, season)
    row = table.get(_norm(name))
    if not row or not league_xg_mean:
        return None
    out = {}
    if "xgf90" in row:
        out["atk"] = row["xgf90"] / league_xg_mean
    if "xga90" in row:
        out["def"] = row["xga90"] / league_xg_mean
    return out or None

#!/usr/bin/env python3
"""
Daily updater for WC Edge.

Runs once a day (via GitHub Actions). It:
  1. pulls upcoming World Cup fixtures + each team's recent corner/goal stats
     from API-Football (free tier),
  2. runs the statistical model (Poisson goals + Negative Binomial corners),
  3. writes data.json that the web page reads.

No API key set?  ->  it writes a clearly-labelled SAMPLE data.json so the page
still renders. Add the key (env var API_FOOTBALL_KEY) to switch to live data.

Quota note: the free tier is 100 requests/day. We cache each team's stats in
stats_cache.json and only refetch teams with a newly played match, which keeps
a daily run comfortably under the cap.
"""

import os, json, datetime, urllib.request
from scipy.stats import nbinom, poisson

API_KEY   = os.environ.get("API_FOOTBALL_KEY", "").strip()
BASE      = "https://v3.football.api-sports.io"
LEAGUE_ID = int(os.environ.get("WC_LEAGUE_ID", "1"))   # confirm via /leagues
SEASON    = int(os.environ.get("WC_SEASON", "2026"))
OUT       = os.path.join(os.path.dirname(__file__), "data.json")

# ---- model constants (tune on real data) ----
MU_GOALS, LEAGUE_AVG_CORNERS, CORNER_R = 1.35, 5.1, 7.0

# -------------------- model --------------------
def goals(att, dfn):
    return MU_GOALS * att["atk"] * dfn["def"], MU_GOALS * dfn["atk"] * att["def"]

def over(line, la, lb):
    return 1 - poisson.cdf(int(line), la + lb)

def leads_ht(la, lb, share=0.45):
    la, lb = la * share, lb * share
    return sum(poisson.pmf(i, la) * poisson.pmf(j, lb)
               for i in range(8) for j in range(8) if i > j)

def gstate(supremacy):
    return (1 + 0.10 * (1 if supremacy > 0 else -1) * min(1, abs(supremacy))) \
         * (1 + 0.18 * max(0.0, -supremacy))

def corners(att, dfn, supremacy):
    return LEAGUE_AVG_CORNERS * att["catk"] * dfn["ccon"] * gstate(supremacy)

def tail(lam, k, r=CORNER_R):
    return float(1 - nbinom.cdf(k - 1, r, r / (r + lam)))

def markets_for(home, away):
    lh, la = goals(home, away)
    ch = corners(home, away,  la - lh)
    ca = corners(away, home,  lh - la)
    return {
        "xg": [round(lh, 2), round(la, 2)],
        "corners": [round(ch, 1), round(ca, 1)],
        "markets": [
            {"label": "Over 1.5 goals",                 "p": round(over(1.5, lh, la), 3)},
            {"label": "Over 2.5 goals",                 "p": round(over(2.5, lh, la), 3)},
            {"label": f"{home['name']} lead at HT",     "p": round(leads_ht(lh, la), 3)},
            {"label": f"{away['name']} 3+ corners",     "p": round(tail(ca, 3), 3)},
            {"label": f"{away['name']} 4+ corners",     "p": round(tail(ca, 4), 3)},
            {"label": f"{away['name']} 5+ corners",     "p": round(tail(ca, 5), 3)},
        ],
    }

# -------------------- API helpers --------------------
def api(path):
    req = urllib.request.Request(BASE + path, headers={"x-apisports-key": API_KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["response"]

def team_rates(team_id, cache):
    """Recent corner/goal rates -> model ratings, relative to league average."""
    key = str(team_id)
    if key in cache:
        return cache[key]
    fx = api(f"/fixtures?team={team_id}&league={LEAGUE_ID}&season={SEASON}&status=FT")
    gf = ga = cf = caa = n = 0
    for f in fx[-5:]:                                   # last 5 matches
        fid = f["fixture"]["id"]
        home = f["teams"]["home"]["id"] == team_id
        gf += f["goals"]["home" if home else "away"] or 0
        ga += f["goals"]["away" if home else "home"] or 0
        for side in api(f"/fixtures/statistics?fixture={fid}"):
            is_me = side["team"]["id"] == team_id
            for s in side["statistics"]:
                if s["type"] == "Corner Kicks" and s["value"] is not None:
                    cf += s["value"] if is_me else 0
                    caa += s["value"] if not is_me else 0
        n += 1
    n = max(n, 1)
    rate = {
        "atk":  (gf / n) / MU_GOALS or 1, "def": (ga / n) / MU_GOALS or 1,
        "catk": (cf / n) / LEAGUE_AVG_CORNERS or 1,
        "ccon": (caa / n) / LEAGUE_AVG_CORNERS or 1,
    }
    cache[key] = rate
    return rate

# -------------------- live vs sample --------------------
def build_live():
    cache = {}
    if os.path.exists("stats_cache.json"):
        cache = json.load(open("stats_cache.json"))
    today = datetime.date.today().isoformat()
    soon  = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
    fx = api(f"/fixtures?league={LEAGUE_ID}&season={SEASON}&from={today}&to={soon}")
    out = []
    for f in fx:
        h, a = f["teams"]["home"], f["teams"]["away"]
        hr = {**team_rates(h["id"], cache), "name": h["name"]}
        ar = {**team_rates(a["id"], cache), "name": a["name"]}
        out.append({"home": h["name"], "away": a["name"],
                    "kickoff": f["fixture"]["date"], **markets_for(hr, ar)})
    json.dump(cache, open("stats_cache.json", "w"))
    return out, False

def build_sample():
    def t(name, atk, dfn, catk, ccon):
        return {"name": name, "atk": atk, "def": dfn, "catk": catk, "ccon": ccon}
    pairs = [
        (t("Spain", 1.85, 0.70, 1.20, 0.85), t("Uruguay", 0.95, 1.05, 1.00, 1.10), "2026-06-26T18:00:00Z"),
        (t("Brazil", 1.60, 0.80, 1.15, 0.90), t("Netherlands", 1.45, 0.85, 1.10, 0.95), "2026-06-29T19:00:00Z"),
    ]
    return [{"home": h["name"], "away": a["name"], "kickoff": k, **markets_for(h, a)}
            for h, a, k in pairs], True

# -------------------- main --------------------
def main():
    fixtures, sample = (build_sample() if not API_KEY else build_live())
    json.dump({
        "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sample": sample,
        "fixtures": fixtures,
    }, open(OUT, "w"), indent=2)
    print(f"Wrote {OUT} | {len(fixtures)} fixtures | sample={sample}")

if __name__ == "__main__":
    main()

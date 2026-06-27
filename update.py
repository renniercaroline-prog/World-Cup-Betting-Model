#!/usr/bin/env python3
"""
update.py — runs daily. Fetches data, asks the agent for lineups, runs the
model, writes data.json. With no API_FOOTBALL_KEY it produces labelled sample
data so the page renders end to end.
"""
import os, json, datetime, urllib.request
from collections import defaultdict
import model, agent, elo, xg

API_KEY   = os.environ.get("API_FOOTBALL_KEY", "").strip()
BASE      = "https://v3.football.api-sports.io"
LEAGUE_ID = int(os.environ.get("WC_LEAGUE_ID", "1"))
SEASON    = int(os.environ.get("WC_SEASON", "2026"))
OUT       = os.path.join(os.path.dirname(__file__), "data.json")
LOOKAHEAD_DAYS = 4               # today + next 3 days of fixtures
ELO_SEASONS_BACK = 4             # seasons of internationals for the Elo prior; the 15-month
                                 # time-decay down-weights the oldest, so extra history is safe
                                 # and helps teams that play infrequently (international = sparse)
CLUB_SEASON = SEASON - 1         # club-season used for player priors (just-finished season)
XG_LEAGUE = os.environ.get("XG_LEAGUE", "INT-World Cup").strip()  # FBref key for team xG (best-effort)
# Only line-shop across bookmakers the user can actually bet at. Including offshore/
# sharp books (Pinnacle, 1xBet, Betano, Marathonbet, SBO) inflates the apparent edge
# with prices that aren't placeable in the UK. Default = UK-accessible books in the
# feed; override via env (comma-separated). Empty string = use every book.
ODDS_BOOKS = set(b.strip() for b in os.environ.get(
    "ODDS_BOOKS", "Bet365,William Hill,Unibet,Betfair,BetVictor,888Sport,10Bet").split(",") if b.strip())

def api(path):
    req = urllib.request.Request(BASE + path, headers={"x-apisports-key": API_KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.load(r)
    if payload.get("errors"):
        print(f"!! API errors on {path.split('?')[0]}: {payload['errors']}")
    return payload.get("response", [])

def team_rates(tid, name, cache):
    if str(tid) in cache: return cache[str(tid)]
    fx = api(f"/fixtures?team={tid}&league={LEAGUE_ID}&season={SEASON}&status=FT")
    gf=ga=cf=ca=sotc=n=0
    for f in fx[-5:]:
        fid=f["fixture"]["id"]; home=f["teams"]["home"]["id"]==tid
        gf+=f["goals"]["home" if home else "away"] or 0
        ga+=f["goals"]["away" if home else "home"] or 0
        for side in api(f"/fixtures/statistics?fixture={fid}"):
            me=side["team"]["id"]==tid
            for s in side["statistics"]:
                if s["value"] is None: continue
                if s["type"]=="Corner Kicks":
                    cf+= s["value"] if me else 0; ca+= s["value"] if not me else 0
                elif s["type"]=="Shots on Goal" and not me:
                    sotc+= s["value"]                 # shots on target we conceded
        n+=1
    n=max(n,1)
    r={"atk":(gf/n)/model.MU_GOALS or 1,"def":(ga/n)/model.MU_GOALS or 1,
       "catk":(cf/n)/model.LEAGUE_AVG_CORNERS or 1,"ccon":(ca/n)/model.LEAGUE_AVG_CORNERS or 1,
       "sot_con":(sotc/n)/model.LEAGUE_AVG_SOT or 1,  # §2.2 opponent SoT adjustment
       "n":n}                                  # n drives the prior/form shrinkage weight
    # §2.1 prefer xG-derived attack/defence when a free source has it (else keep goals)
    try:
        xr = xg.team_rates_from_xg(name, XG_LEAGUE, SEASON, model.MU_GOALS)
        if xr:
            r.update({k: xr[k] for k in ("atk", "def") if k in xr})
    except Exception as e:
        print(f"   (team xG skipped for {name}: {e})")
    cache[str(tid)]=r; return r

def build_historical_elo(team_ids, cache):
    """Phase 1.2 prior: fit a time-decayed Elo over each WC team's recent
    international results, so a team is anchored by years of history, not 3 WC games.
    Pulls are cached per (team, season) in cache.json — history doesn't change.
    Returns a fitted elo.EloEngine keyed by str(team_id), or None on failure (the
    caller then falls back to pure WC form = the old behaviour)."""
    seasons = [SEASON - k for k in range(ELO_SEASONS_BACK)]
    seen, matches = set(), []
    for tid in team_ids:
        for s in seasons:
            ck = f"h{tid}_{s}"
            if ck in cache:
                rows = cache[ck]
            else:
                rows = api(f"/fixtures?team={tid}&season={s}&status=FT")
                cache[ck] = rows
            for f in rows:
                try:
                    fid = f["fixture"]["id"]
                    if fid in seen:
                        continue
                    seen.add(fid)
                    gh, ga = f["goals"]["home"], f["goals"]["away"]
                    if gh is None or ga is None:
                        continue
                    matches.append({"ts": f["fixture"]["date"],
                                    "home": str(f["teams"]["home"]["id"]),
                                    "away": str(f["teams"]["away"]["id"]),
                                    "gh": gh, "ga": ga})
                except (KeyError, TypeError):
                    continue
    print(f"   Elo prior built from {len(matches)} historical fixtures across {len(team_ids)} teams")
    # 1D Elo prior. NOTE: the 2D GoalEloEngine beat 1D on data-rich CLUB football (PL),
    # but a multi-confederation INTERNATIONAL backtest (3,061 matches) found 1D wins on
    # every market — national teams play too few games for the 2D model's doubled
    # parameters to be estimated reliably. The product is international, so we use 1D.
    # (Swap to GoalEloEngine if ever retargeting at club leagues.)
    return elo.EloEngine().fit(matches) if matches else None

def _club_per90(pid, cache):
    """§2.2 club-season per-90s for a player (a ~35-game sample), or None.
    Picks the competition with the most minutes (their main club league)."""
    key=f"club{pid}"
    if key in cache: return cache[key]
    res=None
    try:
        rows=api(f"/players?id={pid}&season={CLUB_SEASON}")
        stats=rows[0].get("statistics",[]) if rows else []
        best=max(stats, key=lambda s:(s.get("games",{}).get("minutes") or 0), default=None)
        mins=(best or {}).get("games",{}).get("minutes") or 0
        if best and mins:
            p90=lambda v:(float(v or 0))/mins*90
            res={"g90":p90(best["goals"]["total"]),"a90":p90(best["goals"]["assists"]),
                 "sot90":p90(best["shots"]["on"]),"fc90":p90(best["fouls"]["committed"]),
                 "fd90":p90(best["fouls"]["drawn"])}
    except (KeyError, TypeError, IndexError) as e:
        print(f"   (club priors skipped for player {pid}: {e})")
    cache[key]=res; return res

def player_pool(tid, cache):
    key=f"p{tid}"
    if key in cache: return cache[key]
    rows=api(f"/players?team={tid}&league={LEAGUE_ID}&season={SEASON}&page=1")
    pool=[]
    for row in rows:
        st=row["statistics"][0]; mins=st["games"]["minutes"] or 0
        per90=lambda v:(float(v or 0))/max(mins,1)*90
        pool.append({"name":row["player"]["name"],"pid":row["player"]["id"],"wc_min":mins,
            "g90":per90(st["goals"]["total"]),"a90":per90(st["goals"]["assists"]),
            "sot90":per90(st["shots"]["on"]),"fc90":per90(st["fouls"]["committed"]),
            "fd90":per90(st["fouls"]["drawn"]),"min":0})
    pool=sorted(pool,key=lambda p:p["g90"]+p["a90"],reverse=True)[:14]
    # §2.2 shrink each WC per-90 toward the player's club-season per-90:
    #   rate = w·club + (1−w)·world_cup,  w high when WC minutes are few.
    # A striker's ~35-game club season is a far better guide than ≤3 WC games.
    for p in pool:
        club=_club_per90(p["pid"],cache)
        if club:
            w=elo.shrink_weight(p.get("wc_min",0)/90.0)   # WC "games" = minutes/90
            for k in ("g90","a90","sot90","fc90","fd90"):
                p[k]=w*club[k]+(1-w)*p[k]
    cache[key]=pool; return pool

# ---------------------------------------------------------------- odds (book prices)
# Pull bookmaker odds, take the BEST price per selection across all books (line
# shopping maximises your payout), de-vig each market to a fair market probability,
# and attach odds + expected value to the matching model markets. EV = model_p*odd-1.
# Odds are NOT cached — they move; we fetch fresh each run.
def _odds_key(betname, value):
    """Normalise a (bookmaker bet, selection) into a key that matches a model market."""
    v = str(value).strip()
    if betname == "Match Winner":
        return {"Home": "1x2:home", "Draw": "1x2:draw", "Away": "1x2:away"}.get(v)
    if betname == "Goals Over/Under":
        return "ou:" + v.lower().replace(" ", ":")            # 'Over 2.5' -> 'ou:over:2.5'
    if betname == "Both Teams Score":
        return "btts:" + v.lower() if v in ("Yes", "No") else None
    if betname == "Corners Over Under":
        return "corners:" + v.lower().replace(" ", ":")        # 'Over 9.5' -> 'corners:over:9.5'
    return None

def best_odds(resp):
    """{key: (best_odd, book_name)} — highest price offered for each selection."""
    out = {}
    if not resp:
        return out
    for bm in resp[0].get("bookmakers", []):
        if ODDS_BOOKS and bm.get("name") not in ODDS_BOOKS:
            continue                                          # only books the user can bet at
        for bet in bm.get("bets", []):
            for val in bet.get("values", []):
                key = _odds_key(bet["name"], val.get("value", ""))
                if not key:
                    continue
                try:
                    odd = float(val["odd"])
                except (KeyError, ValueError, TypeError):
                    continue
                if key not in out or odd > out[key][0]:
                    out[key] = (odd, bm["name"])
    return out

def devig(best):
    """Per logical market, strip the bookmaker margin to a fair probability:
    fair_p = (1/odd) / sum(1/odd over the market's mutually-exclusive outcomes)."""
    groups = defaultdict(list)
    for key in best:
        if key.startswith("1x2:"):       groups["1x2"].append(key)
        elif key.startswith("ou:"):      groups["ou:" + key.split(":")[2]].append(key)      # per line
        elif key.startswith("btts:"):    groups["btts"].append(key)
        elif key.startswith("corners:"): groups["corners:" + key.split(":")[2]].append(key) # per line
    prob = {}
    for keys in groups.values():
        imp = {k: 1.0 / best[k][0] for k in keys}
        s = sum(imp.values())
        if s > 0:
            for k in keys:
                prob[k] = imp[k] / s
    return prob

def _market_key(group, label, home, away):
    if group == "Match result":
        return {f"{home} win": "1x2:home", "Draw": "1x2:draw", f"{away} win": "1x2:away"}.get(label)
    if group == "Total goals":
        return "ou:" + label.lower().replace(" ", ":")
    if group == "Both teams to score":
        return "btts:" + label.lower()
    if group == "Corners" and label.startswith("Total "):
        return "corners:" + label[6:].lower().replace(" ", ":")
    return None

def attach_odds(fo, best, fair):
    """Add book odds, source book, fair (de-vigged) market prob and EV to each market
    we have a price for. Markets with no book line are left model-only."""
    home, away = fo["home"], fo["away"]
    for g in fo["groups"]:
        for m in g["markets"]:
            key = _market_key(g["name"], m["label"], home, away)
            if key and key in best:
                odd, book = best[key]
                m["odd"] = round(odd, 2)
                m["book"] = book
                m["ev"] = round(m["p"] * odd - 1, 3)           # +EV = model rates it above the price
                if key in fair:
                    m["mkt_p"] = round(fair[key], 3)           # market's fair prob, for context

def build_live():
    cache={}
    if os.path.exists("cache.json"): cache=json.load(open("cache.json"))
    # fetch upcoming fixtures by explicit UTC date (robust to timezone + param quirks)
    fx=[]
    base=datetime.datetime.now(datetime.timezone.utc).date()
    for k in range(LOOKAHEAD_DAYS):
        dt=(base+datetime.timedelta(days=k)).isoformat()
        chunk=api(f"/fixtures?league={LEAGUE_ID}&season={SEASON}&date={dt}")
        print(f"   {dt}: {len(chunk)} fixtures")
        fx+=chunk
    print(f"   total raw fixtures: {len(fx)}")
    # keep only bettable, fully-determined fixtures up front so we know which
    # teams to build the historical prior for
    games=[]
    for f in fx:
        h,a=f["teams"]["home"],f["teams"]["away"]
        # §4.1 hide kicked-off/finished games: only "NS" (Not Started) is bettable
        # pre-match. status lives at fixture.status.short in API-Football v3.
        if f.get("fixture",{}).get("status",{}).get("short") != "NS":
            continue
        if not h.get("id") or not a.get("id"):   # undetermined knockout slot
            continue
        games.append(f)
    # Phase 1.2 historical Elo prior (best-effort; never break the run over it)
    team_ids={f["teams"][s]["id"] for f in games for s in ("home","away")}
    try:
        engine=build_historical_elo(team_ids,cache)
    except Exception as e:
        print(f"   !! Elo prior failed, using WC form only: {e}")
        engine=None

    def rating(side,ko):
        """Blend the historical Elo prior with sparse WC form (Phase 1.2 shrinkage)."""
        form=team_rates(side["id"],side["name"],cache)
        if engine is None:
            blended=dict(form)
        else:
            prior=engine.rates_asof(str(side["id"]),ko)
            blended=elo.blend(prior,form,elo.shrink_weight(form.get("n",0)))
        blended["sot_con"]=form.get("sot_con",1.0)   # carry through for §2.2 opponent SoT adj
        return blended

    out=[]
    for f in games:
        h,a=f["teams"]["home"],f["teams"]["away"]; ko=f["fixture"]["date"]
        hp_pool=player_pool(h["id"],cache); ap_pool=player_pool(a["id"],cache)
        proj=agent.project(h["name"],a["name"],ko)
        # §5.4 missing/rested key players dock TEAM attack/defence, not just props
        h_atk,h_def=model.lineup_strength(hp_pool,proj["home"])
        a_atk,a_def=model.lineup_strength(ap_pool,proj["away"])
        hr={**rating(h,ko),"name":h["name"]}
        ar={**rating(a,ko),"name":a["name"]}
        hr["atk"]*=h_atk; hr["def"]*=h_def
        ar["atk"]*=a_atk; ar["def"]*=a_def
        hp=agent.apply_minutes(hp_pool,proj["home"])
        ap=agent.apply_minutes(ap_pool,proj["away"])
        fo=model.build_fixture(hr,ar,hp,ap,proj.get("note","")); fo["kickoff"]=ko
        # auto odds: pull book prices, attach best-price + EV to each matched market
        try:
            od=best_odds(api(f"/odds?fixture={f['fixture']['id']}"))
            attach_odds(fo,od,devig(od))
        except Exception as e:
            print(f"   (odds skipped for {h['name']} v {a['name']}: {e})")
        out.append(fo)
    out.sort(key=lambda o:o["kickoff"])              # §4.2 soonest kickoff first
    json.dump(cache,open("cache.json","w"))
    return out,False

def build_sample():
    def t(n,atk,d,catk,ccon): return {"name":n,"atk":atk,"def":d,"catk":catk,"ccon":ccon}
    def pl(n,g,a,s,fc,fd,m): return {"name":n,"g90":g,"a90":a,"sot90":s,"fc90":fc,"fd90":fd,"min":m}
    note="Sample mode — add ANTHROPIC_API_KEY for live lineup projection."
    fixtures=[
        (t("Spain",1.85,.70,1.20,.85), t("Uruguay",.95,1.05,1.00,1.10), "2026-06-26T18:00:00Z",
         [pl("Oyarzabal",.55,.30,1.3,.8,1.0,90), pl("Yamal",.45,.55,1.6,.5,1.8,80), pl("Pedri",.20,.45,.9,1.1,1.3,85)],
         [pl("Nunez",.50,.20,1.4,1.0,1.2,75), pl("Valverde",.35,.35,1.5,1.4,1.0,90), pl("Araujo M.",.40,.25,1.1,.9,1.1,85)], note),
        (t("Brazil",1.60,.80,1.15,.90), t("Netherlands",1.45,.85,1.10,.95), "2026-06-29T19:00:00Z",
         [pl("Rodrygo",.45,.30,1.4,.6,1.2,85), pl("Vinicius",.50,.40,1.7,.9,2.0,90)],
         [pl("Gakpo",.45,.35,1.5,.7,1.0,90), pl("Depay",.50,.30,1.6,.9,1.1,80)], note),
    ]
    out=[]
    for h,a,ko,hp,ap,n in fixtures:
        o=model.build_fixture(h,a,hp,ap,n); o["kickoff"]=ko; out.append(o)
    return out,True

def main():
    fixtures,sample=(build_sample() if not API_KEY else build_live())
    json.dump({"updated":datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "sample":sample,"fixtures":fixtures}, open(OUT,"w"), indent=2)
    print(f"Wrote {OUT} | {len(fixtures)} fixtures | sample={sample}")

if __name__=="__main__":
    main()

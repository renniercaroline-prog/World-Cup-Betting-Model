#!/usr/bin/env python3
"""
update.py — runs daily. Fetches data, asks the agent for lineups, runs the
model, writes data.json. With no API_FOOTBALL_KEY it produces labelled sample
data so the page renders end to end.
"""
import os, json, datetime, urllib.request
import model, agent

API_KEY   = os.environ.get("API_FOOTBALL_KEY", "").strip()
BASE      = "https://v3.football.api-sports.io"
LEAGUE_ID = int(os.environ.get("WC_LEAGUE_ID", "1"))
SEASON    = int(os.environ.get("WC_SEASON", "2026"))
OUT       = os.path.join(os.path.dirname(__file__), "data.json")

def api(path):
    req = urllib.request.Request(BASE + path, headers={"x-apisports-key": API_KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["response"]

def team_rates(tid, cache):
    if str(tid) in cache: return cache[str(tid)]
    fx = api(f"/fixtures?team={tid}&league={LEAGUE_ID}&season={SEASON}&status=FT")
    gf=ga=cf=ca=n=0
    for f in fx[-5:]:
        fid=f["fixture"]["id"]; home=f["teams"]["home"]["id"]==tid
        gf+=f["goals"]["home" if home else "away"] or 0
        ga+=f["goals"]["away" if home else "home"] or 0
        for side in api(f"/fixtures/statistics?fixture={fid}"):
            me=side["team"]["id"]==tid
            for s in side["statistics"]:
                if s["type"]=="Corner Kicks" and s["value"] is not None:
                    cf+= s["value"] if me else 0; ca+= s["value"] if not me else 0
        n+=1
    n=max(n,1)
    r={"atk":(gf/n)/model.MU_GOALS or 1,"def":(ga/n)/model.MU_GOALS or 1,
       "catk":(cf/n)/model.LEAGUE_AVG_CORNERS or 1,"ccon":(ca/n)/model.LEAGUE_AVG_CORNERS or 1}
    cache[str(tid)]=r; return r

def player_pool(tid, cache):
    key=f"p{tid}"
    if key in cache: return cache[key]
    rows=api(f"/players?team={tid}&league={LEAGUE_ID}&season={SEASON}&page=1")
    pool=[]
    for row in rows:
        st=row["statistics"][0]; mins=st["games"]["minutes"] or 0
        per90=lambda v:(float(v or 0))/max(mins,1)*90
        pool.append({"name":row["player"]["name"],
            "g90":per90(st["goals"]["total"]),"a90":per90(st["goals"]["assists"]),
            "sot90":per90(st["shots"]["on"]),"fc90":per90(st["fouls"]["committed"]),
            "fd90":per90(st["fouls"]["drawn"]),"min":0})
    pool=sorted(pool,key=lambda p:p["g90"]+p["a90"],reverse=True)[:14]
    cache[key]=pool; return pool

def build_live():
    cache={}
    if os.path.exists("cache.json"): cache=json.load(open("cache.json"))
    fx=api(f"/fixtures?league={LEAGUE_ID}&season={SEASON}&next=20")
    out=[]
    for f in fx:
        h,a=f["teams"]["home"],f["teams"]["away"]; ko=f["fixture"]["date"]
        if not h.get("id") or not a.get("id"):   # undetermined knockout slot
            continue
        hr={**team_rates(h["id"],cache),"name":h["name"]}
        ar={**team_rates(a["id"],cache),"name":a["name"]}
        proj=agent.project(h["name"],a["name"],ko)
        hp=agent.apply_minutes(player_pool(h["id"],cache),proj["home"])
        ap=agent.apply_minutes(player_pool(a["id"],cache),proj["away"])
        fo=model.build_fixture(hr,ar,hp,ap,proj.get("note","")); fo["kickoff"]=ko; out.append(fo)
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

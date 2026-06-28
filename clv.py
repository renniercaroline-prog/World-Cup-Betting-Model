#!/usr/bin/env python3
"""
clv.py — closing-line-value logger. THE forward test of whether the model has edge.

WHY THIS IS THE IMPORTANT ONE
-----------------------------
Calibration and backtests tell you the model is internally sensible; they do NOT
tell you it beats the bookmaker. The single best signal for that — better than
waiting for results, because it needs far fewer bets to be meaningful — is
**closing-line value**: did you take a price *better* than where the market closed?
If the line consistently moves toward your side by kickoff, you are ahead of the
market. If it doesn't, your "edges" are noise, no matter how good the backtest looked.

HOW IT WORKS
------------
Every pipeline run, for each market the board would flag (same filter as the UI):
  - the FIRST time we see it, record the price we could take now (the "open").
  - on later runs, roll the latest pre-kickoff price (the "close").
  - once the fixture kicks off, freeze it and compute CLV = open_odds/close_odds - 1
    (positive = we beat the close).
  - once it finishes, settle win/lose for realised P/L at the price we took.

State lives in clv_log.json (committed by the workflow so it accumulates over time).
`python clv.py` prints the running scoreboard: average CLV, % of bets that beat the
close, and realised profit. Give it a few dozen settled bets before trusting it.
"""
import os, json, datetime

LOG = os.path.join(os.path.dirname(__file__), "clv_log.json")

# Candidate filter — MUST stay in sync with bestBets() in index.html, so we log
# exactly the bets the board would surface (small, believable edges; drops the
# model-vs-market blow-ups that are usually model error).
def is_candidate(m):
    if m.get("ev") is None or m.get("mkt_p") is None:
        return False
    ratio = m["p"] / m["mkt_p"] if m["mkt_p"] else 99
    return 0.03 <= m["ev"] <= 0.25 and ratio <= 1.6 and m["mkt_p"] >= 0.12

def load():
    return json.load(open(LOG)) if os.path.exists(LOG) else {}

def save(log):
    json.dump(log, open(LOG, "w"), indent=2)

def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

# ----------------------------------------------------------------- settling
def grade(group, label, home, away, gh, ga, corners):
    """Did the selection win? True/False, or None if ungradeable (no data, or a
    market we can't settle from the full-time score, e.g. first-half goals).
    home/away are team names so we can read '{team} ...' labels."""
    if gh is None or ga is None:
        return None
    total = gh + ga
    def ou(line_str, value):              # 'Over'/'Under' helper on a total
        side, line = line_str.split(); line = float(line)
        return value > line if side == "Over" else value < line
    if group == "Match result":
        win = home if gh > ga else (away if ga > gh else "Draw")
        return label == f"{win} win" or (label == "Draw" and win == "Draw")
    if group == "Double chance":
        if label == f"{home} or draw": return gh >= ga
        if label == "Either team (no draw)": return gh != ga
        if label == f"{away} or draw": return ga >= gh
        return None
    if group == "Total goals":
        return ou(label, total)
    if group == f"{home} total goals":
        return ou(label, gh)
    if group == f"{away} total goals":
        return ou(label, ga)
    if group == "Both teams to score":
        yes = gh > 0 and ga > 0
        return yes if label == "Yes" else (not yes)
    if group == "Clean sheet":
        if label == f"{home} yes": return ga == 0
        if label == f"{away} yes": return gh == 0
        return None
    if group == "Win to nil":
        if label == f"{home} yes": return gh > ga and ga == 0
        if label == f"{away} yes": return ga > gh and gh == 0
        return None
    if group == "Correct score":
        try:
            i, j = map(int, label.split(":")); return gh == i and ga == j
        except ValueError:
            return None
    if group == "Corners" and label.startswith("Total "):
        if corners is None:
            return None
        return ou(label[6:], corners)
    return None                           # first-half / player props / team-corner thresholds




# ----------------------------------------------------------------- main update
def update_log(snapshots, results_fn, now=None):
    """snapshots: list of dicts for CURRENT upcoming candidates, each with
    fid, home, away, kickoff, group, label, p, odd, book, mkt_p, ev.
    results_fn(fid) -> (status_short, gh, ga, corners_total) or None."""
    now = now or _now()
    log = load()
    current = set(s["fid"] for s in snapshots)

    # 1) open new candidates, roll the closing price on ones still open
    for s in snapshots:
        if not is_candidate(s):
            continue
        key = f"{s['fid']}|{s['group']}|{s['label']}"
        r = log.get(key)
        if r is None:
            log[key] = {
                "fixture": f"{s['home']} v {s['away']}", "home": s["home"], "away": s["away"],
                "kickoff": s["kickoff"], "group": s["group"], "label": s["label"],
                "model_p": s["p"], "open_mkt_p": s["mkt_p"],
                "open_odds": s["odd"], "open_book": s["book"], "open_ts": now,
                "close_odds": s["odd"], "close_book": s["book"], "close_ts": now,
                "status": "open", "clv": None, "result": None, "pnl": None,
            }
        elif r["status"] == "open":
            r.update(close_odds=s["odd"], close_book=s["book"], close_ts=now, model_p=s["p"])

    # 2) close + settle anything that has kicked off since we last looked
    for key, r in log.items():
        if r["status"] != "open":
            continue
        fid = int(key.split("|")[0])
        if fid in current:                # still upcoming — keep rolling
            continue
        r["status"] = "closed"
        if r["open_odds"] and r["close_odds"]:
            r["clv"] = round(r["open_odds"] / r["close_odds"] - 1, 4)   # +ve = beat the close
        res = results_fn(fid) if results_fn else None
        if res and res[0] in ("FT", "AET", "PEN"):
            won = grade(r["group"], r["label"], r["home"], r["away"], res[1], res[2], res[3])
            if won is not None:
                r["result"] = "win" if won else "lose"
                r["pnl"] = round(r["open_odds"] - 1 if won else -1, 3)  # 1 unit flat at our price
    save(log)
    return log

# ----------------------------------------------------------------- scoreboard
def summary_dict(log=None):
    """Machine-readable track record (embedded in data.json for the website panel)."""
    log = log if log is not None else load()
    recs = list(log.values())
    # CLV is only meaningful when we captured an EARLY 'open' and a LATER 'close'.
    # Records seen only once (open_ts == close_ts — first logged right before kickoff)
    # carry no closing-line information, so they're excluded from the CLV stats rather
    # than counted as "didn't beat the close" (which would falsely drag the number down).
    moved = [r for r in recs if r.get("clv") is not None and r.get("open_ts") != r.get("close_ts")]
    single = sum(1 for r in recs if r.get("clv") is not None and r.get("open_ts") == r.get("close_ts"))
    clv = [r["clv"] for r in moved]
    settled = [r for r in recs if r.get("result")]
    wins = sum(1 for r in settled if r["result"] == "win")
    pnl = sum(r["pnl"] for r in settled if r.get("pnl") is not None)
    return {
        "logged": len(recs),
        "open": sum(1 for r in recs if r["status"] == "open"),
        "closed": len(clv),                 # only bets with a real open-vs-close comparison
        "clv_single": single,               # logged once, too late to measure CLV
        "avg_clv": round(sum(clv) / len(clv), 4) if clv else None,
        "beat_close": round(sum(1 for c in clv if c > 0) / len(clv), 3) if clv else None,
        "settled": len(settled), "wins": wins, "losses": len(settled) - wins,
        "pnl": round(pnl, 2), "roi": round(pnl / len(settled), 3) if settled else None,
    }

def summary(log=None):
    log = log if log is not None else load()
    recs = list(log.values())
    openn = [r for r in recs if r["status"] == "open"]
    moved = [r for r in recs if r.get("clv") is not None and r.get("open_ts") != r.get("close_ts")]
    single = sum(1 for r in recs if r.get("clv") is not None and r.get("open_ts") == r.get("close_ts"))
    clv = [r["clv"] for r in moved]
    settled = [r for r in recs if r.get("result")]
    print("=" * 64)
    print(f"CLV SCOREBOARD — {len(recs)} logged ({len(openn)} open, {len(clv)} with a real "
          f"open→close, {single} logged too late to measure)")
    print("=" * 64)
    if clv:
        beat = sum(1 for c in clv if c > 0)
        print(f"Closing-line value (the edge signal):")
        print(f"  average CLV   : {sum(clv)/len(clv)*100:+.2f}%   (want this clearly > 0)")
        print(f"  beat the close: {beat}/{len(clv)} = {100*beat/len(clv):.0f}% of bets")
    else:
        print(f"No measurable CLV yet ({single} bets were logged right before kickoff, so there")
        print(" was no earlier 'opening' price to compare). Needs games captured days ahead.")
    if settled:
        wins = sum(1 for r in settled if r["result"] == "win")
        pnl = sum(r["pnl"] for r in settled if r.get("pnl") is not None)
        staked = len(settled)
        print(f"Realised results ({staked} settled, 1 unit flat at our taken price):")
        print(f"  record : {wins}W-{staked-wins}L")
        print(f"  profit : {pnl:+.2f} units   ROI {100*pnl/staked:+.1f}%")
    print("=" * 64)
    print("CLV is the leading indicator; realised P/L is noisier and needs more bets.")
    return log

if __name__ == "__main__":
    summary()

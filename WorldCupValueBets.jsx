import { useState, useMemo } from "react";

// ---------- palette (pitch + betting-slip) ----------
const C = {
  pitch: "#0A2A22",
  card: "#103328",
  card2: "#0E2C23",
  line: "#1E4A3B",
  chalk: "#F2F5EE",
  sage: "#8FA89B",
  gold: "#F5C542",
  red: "#E5564E",
  blue: "#7FB2E6",
};
const mono = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
const sans = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";

// ---------- preloaded real WC2026 group-stage results ----------
const SEED = `
Mexico 2-0 South Africa
South Korea 2-1 Czechia
Czechia 1-1 South Africa
Mexico 1-0 South Korea
Czechia 0-3 Mexico
South Africa 1-0 South Korea
Canada 1-1 Bosnia and Herzegovina
Switzerland 1-1 Qatar
Switzerland 4-1 Bosnia and Herzegovina
Canada 6-0 Qatar
Switzerland 3-1 Canada
Bosnia and Herzegovina 3-1 Qatar
Brazil 1-1 Morocco
Scotland 1-0 Haiti
Scotland 0-1 Morocco
Brazil 3-0 Haiti
Scotland 0-3 Brazil
Morocco 4-2 Haiti
USA 4-1 Paraguay
Australia 2-0 Turkiye
USA 2-0 Australia
Turkiye 0-1 Paraguay
Turkiye 3-2 USA
Paraguay 0-0 Australia
Germany 7-1 Curacao
Ivory Coast 1-0 Ecuador
Germany 2-1 Ivory Coast
Ecuador 0-0 Curacao
Curacao 0-2 Ivory Coast
Ecuador 2-1 Germany
Netherlands 2-2 Japan
Sweden 5-1 Tunisia
Netherlands 5-1 Sweden
Japan 4-0 Tunisia
Japan 1-1 Sweden
Netherlands 3-1 Tunisia
Belgium 1-1 Egypt
Iran 2-2 New Zealand
Belgium 0-0 Iran
Egypt 3-1 New Zealand
France 3-1 Senegal
Norway 4-1 Iraq
France 3-0 Iraq
Norway 3-2 Senegal
Norway 1-4 France
Spain 0-0 Cabo Verde
Saudi Arabia 1-1 Uruguay
Spain 4-0 Saudi Arabia
Uruguay 2-2 Cabo Verde
Portugal 1-1 DR Congo
Colombia 3-1 Uzbekistan
Portugal 5-0 Uzbekistan
Colombia 1-0 DR Congo
England 4-2 Croatia
Ghana 1-0 Panama
England 0-0 Ghana
Croatia 1-0 Panama
`.trim();

// ---------- parsing ----------
function parseResults(text) {
  const rows = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const m = line.match(/^(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)$/);
    if (!m) continue;
    rows.push({ home: m[1].trim(), hg: +m[2], ag: +m[3], away: m[4].trim() });
  }
  return rows;
}

// ---------- iterative Poisson attack/defense fit ----------
function fitRatings(rows) {
  const teams = {};
  const add = (t) => { if (!teams[t]) teams[t] = { gf: 0, ga: 0, gp: 0, atk: 1, def: 1 }; };
  let totalGoals = 0, totalTeamGames = 0;
  for (const r of rows) {
    add(r.home); add(r.away);
    teams[r.home].gf += r.hg; teams[r.home].ga += r.ag; teams[r.home].gp += 1;
    teams[r.away].gf += r.ag; teams[r.away].ga += r.hg; teams[r.away].gp += 1;
    totalGoals += r.hg + r.ag; totalTeamGames += 2;
  }
  const names = Object.keys(teams);
  if (!names.length) return { teams, names, mu: 1.3 };
  const mu = totalGoals / totalTeamGames; // avg goals scored per team per game

  for (let iter = 0; iter < 60; iter++) {
    // update attack: goals scored / expected if league-avg, given opp defense
    for (const t of names) {
      let scoredExp = 0, concededExp = 0;
      for (const r of rows) {
        if (r.home === t) { scoredExp += mu * teams[r.away].def; concededExp += mu * teams[r.away].atk; }
        else if (r.away === t) { scoredExp += mu * teams[r.home].def; concededExp += mu * teams[r.home].atk; }
      }
      if (scoredExp > 0) teams[t].atk = teams[t].gf / scoredExp;
      if (concededExp > 0) teams[t].def = teams[t].ga / concededExp;
    }
    // normalise to mean 1 (identifiability)
    const ma = names.reduce((s, t) => s + teams[t].atk, 0) / names.length;
    const md = names.reduce((s, t) => s + teams[t].def, 0) / names.length;
    for (const t of names) { teams[t].atk /= ma; teams[t].def /= md; }
  }
  // shrink toward 1 a touch — only 3 games each, ratings are noisy
  const k = 0.85;
  for (const t of names) {
    teams[t].atk = 1 + k * (teams[t].atk - 1);
    teams[t].def = 1 + k * (teams[t].def - 1);
  }
  return { teams, names: names.sort(), mu };
}

// ---------- Poisson helpers ----------
function poissonPmf(lambda, k) {
  let logp = -lambda + k * Math.log(lambda);
  for (let i = 2; i <= k; i++) logp -= Math.log(i);
  return Math.exp(logp);
}

function matchProbs(lamA, lamB, maxG = 10) {
  let pA = 0, pD = 0, pB = 0, over25 = 0, btts = 0;
  const grid = [];
  for (let i = 0; i <= maxG; i++) {
    for (let j = 0; j <= maxG; j++) {
      const p = poissonPmf(lamA, i) * poissonPmf(lamB, j);
      if (i > j) pA += p; else if (i === j) pD += p; else pB += p;
      if (i + j > 2) over25 += p;
      if (i > 0 && j > 0) btts += p;
      grid.push({ i, j, p });
    }
  }
  grid.sort((x, y) => y.p - x.p);
  return { pA, pD, pB, over25, btts, top: grid.slice(0, 4) };
}

const pct = (x) => (100 * x).toFixed(1) + "%";
const dec = (x, n = 2) => (isFinite(x) ? x.toFixed(n) : "–");

export default function WorldCupValueBets() {
  const [resultsText, setResultsText] = useState(SEED);
  const [showData, setShowData] = useState(false);
  const [showMethod, setShowMethod] = useState(false);

  const { teams, names, mu } = useMemo(() => fitRatings(parseResults(resultsText)), [resultsText]);

  const [teamA, setTeamA] = useState("Brazil");
  const [teamB, setTeamB] = useState("Netherlands");

  // bookmaker decimal odds
  const [oddsA, setOddsA] = useState("2.40");
  const [oddsD, setOddsD] = useState("3.30");
  const [oddsB, setOddsB] = useState("2.90");

  const [bankroll, setBankroll] = useState("100");
  const [kellyFrac, setKellyFrac] = useState(0.25);

  const A = teams[teamA], B = teams[teamB];
  const valid = A && B && teamA !== teamB;

  const model = useMemo(() => {
    if (!valid) return null;
    const lamA = mu * A.atk * B.def;
    const lamB = mu * B.atk * A.def;
    return { lamA, lamB, ...matchProbs(lamA, lamB) };
  }, [valid, A, B, mu]);

  // de-vig + value
  const market = useMemo(() => {
    const oA = parseFloat(oddsA), oD = parseFloat(oddsD), oB = parseFloat(oddsB);
    if (!(oA > 1 && oD > 1 && oB > 1)) return null;
    const impA = 1 / oA, impD = 1 / oD, impB = 1 / oB;
    const over = impA + impD + impB;
    return {
      odds: { A: oA, D: oD, B: oB },
      imp: { A: impA, D: impD, B: impB },
      fair: { A: impA / over, D: impD / over, B: impB / over },
      margin: over - 1,
    };
  }, [oddsA, oddsD, oddsB]);

  const bets = useMemo(() => {
    if (!model || !market) return [];
    const bk = parseFloat(bankroll) || 0;
    const defs = [
      { key: "A", label: teamA + " win", p: model.pA, o: market.odds.A, fair: market.fair.A },
      { key: "D", label: "Draw", p: model.pD, o: market.odds.D, fair: market.fair.D },
      { key: "B", label: teamB + " win", p: model.pB, o: market.odds.B, fair: market.fair.B },
    ];
    return defs.map((d) => {
      const ev = d.p * d.o - 1;              // EV per 1 unit staked
      const kelly = (d.p * d.o - 1) / (d.o - 1); // full-Kelly fraction
      const stake = ev > 0 ? Math.max(0, kelly) * kellyFrac * bk : 0;
      return { ...d, ev, kelly: Math.max(0, kelly), stake };
    });
  }, [model, market, bankroll, kellyFrac, teamA, teamB]);

  const ranked = useMemo(
    () => [...names].map((n) => ({ n, ...teams[n] })).sort((a, b) => (b.atk - b.def) - (a.atk - a.def)),
    [names, teams]
  );

  const inputStyle = {
    background: C.card2, color: C.chalk, border: `1px solid ${C.line}`,
    borderRadius: 8, padding: "10px 12px", fontFamily: mono, fontSize: 15, width: "100%",
    boxSizing: "border-box", outline: "none",
  };
  const label = { color: C.sage, fontSize: 11, letterSpacing: 0.8, textTransform: "uppercase", marginBottom: 6, display: "block" };

  return (
    <div style={{ background: C.pitch, minHeight: "100%", fontFamily: sans, color: C.chalk, padding: "20px 16px 48px" }}>
      <div style={{ maxWidth: 560, margin: "0 auto" }}>

        {/* header */}
        <div style={{ borderBottom: `1px solid ${C.line}`, paddingBottom: 14, marginBottom: 20 }}>
          <div style={{ color: C.gold, fontFamily: mono, fontSize: 11, letterSpacing: 2 }}>WORLD CUP 2026 · VALUE FINDER</div>
          <h1 style={{ margin: "6px 0 4px", fontSize: 26, fontWeight: 700, lineHeight: 1.1 }}>
            Beat the book, not the game.
          </h1>
          <p style={{ color: C.sage, fontSize: 13, margin: 0 }}>
            Model the goals → strip the bookmaker’s margin → only bet when your edge is positive.
          </p>
        </div>

        {/* matchup */}
        <Section title="1 · The matchup">
          <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 8, alignItems: "end" }}>
            <div>
              <span style={label}>Team A</span>
              <select value={teamA} onChange={(e) => setTeamA(e.target.value)} style={inputStyle}>
                {names.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <div style={{ color: C.sage, fontFamily: mono, paddingBottom: 10 }}>v</div>
            <div>
              <span style={label}>Team B</span>
              <select value={teamB} onChange={(e) => setTeamB(e.target.value)} style={inputStyle}>
                {names.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
          </div>

          {!valid && <p style={{ color: C.red, fontSize: 13 }}>Pick two different teams.</p>}

          {valid && model && (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 13, color: C.sage }}>
                <span>expected goals</span>
                <span style={{ color: C.chalk }}>{dec(model.lamA)} – {dec(model.lamB)}</span>
              </div>
              <ProbBar a={model.pA} d={model.pD} b={model.pB} la={teamA} lb={teamB} />
              <div style={{ display: "flex", gap: 8, marginTop: 12, fontFamily: mono, fontSize: 12, color: C.sage, flexWrap: "wrap" }}>
                <Chip>Over 2.5: {pct(model.over25)}</Chip>
                <Chip>BTTS: {pct(model.btts)}</Chip>
                {model.top.map((s, i) => <Chip key={i}>{s.i}-{s.j} ({pct(s.p)})</Chip>)}
              </div>
            </div>
          )}
        </Section>

        {/* odds */}
        <Section title="2 · The bookmaker’s odds">
          <p style={{ color: C.sage, fontSize: 12, marginTop: -4 }}>Enter decimal odds from your friend’s sportsbook.</p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
            <OddsInput label={teamA} value={oddsA} onChange={setOddsA} />
            <OddsInput label="Draw" value={oddsD} onChange={setOddsD} />
            <OddsInput label={teamB} value={oddsB} onChange={setOddsB} />
          </div>
          {market && (
            <div style={{ marginTop: 12, fontFamily: mono, fontSize: 12, color: C.sage }}>
              Bookmaker margin (overround): <span style={{ color: market.margin > 0.07 ? C.red : C.gold }}>{pct(market.margin)}</span>
              <span style={{ color: C.line }}> · </span>
              fair market: {pct(market.fair.A)} / {pct(market.fair.D)} / {pct(market.fair.B)}
            </div>
          )}
        </Section>

        {/* bankroll controls */}
        <Section title="3 · Stake settings">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <span style={label}>Bankroll</span>
              <input value={bankroll} onChange={(e) => setBankroll(e.target.value)} style={inputStyle} inputMode="decimal" />
            </div>
            <div>
              <span style={label}>Kelly fraction · {kellyFrac.toFixed(2)}×</span>
              <input type="range" min="0.1" max="1" step="0.05" value={kellyFrac}
                onChange={(e) => setKellyFrac(+e.target.value)}
                style={{ width: "100%", accentColor: C.gold, marginTop: 14 }} />
            </div>
          </div>
          <p style={{ color: C.sage, fontSize: 11, marginBottom: 0 }}>
            Full Kelly maximises growth but swings hard. Quarter-Kelly (0.25×) is the common real-world safety setting.
          </p>
        </Section>

        {/* VERDICT — signature element */}
        <div style={{ marginTop: 8 }}>
          <div style={{ ...label, color: C.gold }}>The verdict</div>
          {bets.map((b) => {
            const value = b.ev > 0;
            return (
              <div key={b.key} style={{
                background: value ? "rgba(245,197,66,0.08)" : C.card,
                border: `1px solid ${value ? C.gold : C.line}`,
                borderRadius: 12, padding: 14, marginBottom: 10,
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ fontWeight: 600, fontSize: 15 }}>{b.label}</div>
                  <div style={{
                    fontFamily: mono, fontSize: 11, letterSpacing: 1, padding: "3px 8px", borderRadius: 6,
                    background: value ? C.gold : "transparent", color: value ? C.pitch : C.sage,
                    border: value ? "none" : `1px solid ${C.line}`,
                  }}>
                    {value ? "VALUE" : "PASS"}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 16, marginTop: 10, fontFamily: mono, fontSize: 12, color: C.sage, flexWrap: "wrap" }}>
                  <span>model <b style={{ color: C.chalk }}>{pct(b.p)}</b></span>
                  <span>implied <b style={{ color: C.chalk }}>{pct(1 / b.o)}</b></span>
                  <span>EV <b style={{ color: value ? C.gold : C.red }}>{(b.ev * 100).toFixed(1)}%</b></span>
                  {value && <span>stake <b style={{ color: C.gold }}>{dec(b.stake)}</b></span>}
                </div>
              </div>
            );
          })}
          {bets.every((b) => b.ev <= 0) && market && model && (
            <p style={{ color: C.sage, fontSize: 13, fontStyle: "italic" }}>
              No edge here — the book has it priced fairly or tighter than the model. The discipline is to pass, not to force a bet.
            </p>
          )}
        </div>

        {/* ratings table */}
        <Collapse open={showData} setOpen={setShowData} title="Team ratings & results data">
          <p style={{ color: C.sage, fontSize: 12 }}>
            Edit the results below (one per line: <span style={{ fontFamily: mono }}>Team 2-1 Team</span>) and ratings refit live.
            Add knockout games as they finish. atk &gt; 1 = scores more than average; def &lt; 1 = concedes less.
          </p>
          <div style={{ maxHeight: 200, overflow: "auto", border: `1px solid ${C.line}`, borderRadius: 8, marginBottom: 12 }}>
            {ranked.map((t) => (
              <div key={t.n} style={{ display: "flex", justifyContent: "space-between", padding: "6px 10px", borderBottom: `1px solid ${C.card2}`, fontFamily: mono, fontSize: 12 }}>
                <span>{t.n}</span>
                <span style={{ color: C.sage }}>atk <b style={{ color: C.gold }}>{dec(t.atk)}</b> · def <b style={{ color: C.blue }}>{dec(t.def)}</b></span>
              </div>
            ))}
          </div>
          <textarea value={resultsText} onChange={(e) => setResultsText(e.target.value)}
            style={{ ...inputStyle, minHeight: 120, fontSize: 12, resize: "vertical" }} spellCheck={false} />
        </Collapse>

        {/* method */}
        <Collapse open={showMethod} setOpen={setShowMethod} title="How it works & honest caveats">
          <div style={{ color: C.sage, fontSize: 13, lineHeight: 1.6 }}>
            <p><b style={{ color: C.chalk }}>Goals model.</b> Each team gets an attack and defense rating fit by iterating a Poisson model over results. Expected goals = league average × A’s attack × B’s defense. A Poisson score matrix turns those into win/draw/loss, over/under and scoreline probabilities.</p>
            <p><b style={{ color: C.chalk }}>De-vig.</b> 1/odds is the implied probability; they sum to more than 100% — that excess is the bookmaker’s margin. Normalising gives the “fair” market view.</p>
            <p><b style={{ color: C.chalk }}>Value & Kelly.</b> Back an outcome only when model probability × odds &gt; 1 (positive EV). Kelly stake = edge ÷ (odds − 1), scaled by your fraction.</p>
            <p style={{ color: C.red }}><b>Caveats that matter:</b> 3 group games per team is tiny data — ratings are noisy (that’s why they’re shrunk toward average). It ignores injuries, red cards, fatigue, venue and knockout dynamics, and assumes goals are independent. A real edge needs years of weighted international results and probably the market odds themselves as a prior. Treat this as a disciplined framework, not a money printer.</p>
            <p style={{ color: C.sage, borderTop: `1px solid ${C.line}`, paddingTop: 10 }}>
              And the real talk for your friend: the house edge is built in, the model can be wrong, and bets should only ever be money he’s fully fine losing. If betting stops being fun, <span style={{ color: C.chalk }}>BeGambleAware.org</span> (UK) is there.
            </p>
          </div>
        </Collapse>

      </div>
    </div>
  );
}

// ---------- small components ----------
function Section({ title, children }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 14, padding: 16, marginBottom: 14 }}>
      <div style={{ color: C.sage, fontSize: 11, letterSpacing: 0.8, textTransform: "uppercase", marginBottom: 12 }}>{title}</div>
      {children}
    </div>
  );
}
function Chip({ children }) {
  return <span style={{ border: `1px solid ${C.line}`, borderRadius: 6, padding: "3px 7px" }}>{children}</span>;
}
function OddsInput({ label, value, onChange }) {
  return (
    <div>
      <span style={{ color: C.sage, fontSize: 11, marginBottom: 6, display: "block", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{label}</span>
      <input value={value} onChange={(e) => onChange(e.target.value)} inputMode="decimal"
        style={{ background: C.card2, color: C.chalk, border: `1px solid ${C.line}`, borderRadius: 8, padding: "10px 8px", fontFamily: mono, fontSize: 16, width: "100%", boxSizing: "border-box", textAlign: "center", outline: "none" }} />
    </div>
  );
}
function ProbBar({ a, d, b, la, lb }) {
  const seg = (w, color, txt) => (
    <div style={{ width: `${w * 100}%`, background: color, color: C.pitch, fontFamily: mono, fontSize: 11, textAlign: "center", padding: "6px 0", overflow: "hidden", whiteSpace: "nowrap" }}>{txt}</div>
  );
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: "flex", borderRadius: 8, overflow: "hidden" }}>
        {seg(a, C.gold, pct(a))}
        {seg(d, C.sage, pct(d))}
        {seg(b, C.blue, pct(b))}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 11, color: C.sage, marginTop: 5 }}>
        <span style={{ color: C.gold }}>{la}</span>
        <span>draw</span>
        <span style={{ color: C.blue }}>{lb}</span>
      </div>
    </div>
  );
}
function Collapse({ open, setOpen, title, children }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 14, marginTop: 14, overflow: "hidden" }}>
      <button onClick={() => setOpen(!open)} style={{ width: "100%", background: "transparent", border: "none", color: C.chalk, padding: 16, textAlign: "left", fontFamily: sans, fontSize: 14, fontWeight: 600, cursor: "pointer", display: "flex", justifyContent: "space-between" }}>
        <span>{title}</span><span style={{ color: C.sage }}>{open ? "–" : "+"}</span>
      </button>
      {open && <div style={{ padding: "0 16px 16px" }}>{children}</div>}
    </div>
  );
}

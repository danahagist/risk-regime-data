// Shared helpers for the dashboard, lessons and calculator pages.
const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function loadJSON(path) {
  const r = await fetch(path, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

const voteClass = (v) => (v === 1 ? 'pos' : v === -1 ? 'neg' : v === 0 ? 'neu' : 'na');
const signed = (x, d = 2) => (x == null ? 'n/a' : (x > 0 ? '+' : '') + x.toFixed(d));
const ASSETS = ['stocks', 'bonds', 'cash', 'alternatives'];
const ASSET_NAME = { stocks: 'Stocks', bonds: 'Bonds', cash: 'Cash', alternatives: 'Alternatives' };

function renderNav(active) {
  const pages = [['index.html', 'Dashboard'], ['learn.html', 'Learn'], ['calculator.html', 'Calculator'],
    ['briefs.html', 'Briefs'], ['backtest.html', 'Backtest'], ['methodology.html', 'Methodology'], ['classic.html', 'Classic view']];
  $('#topbar').innerHTML = `<h1>Market Regime Consensus</h1><nav>${pages.map(([href, name]) =>
    `<a href="${href}" class="${href === active ? 'active' : ''}">${name}</a>`).join('')}</nav>`;
}

function renderFooter(d) {
  $('footer').innerHTML = `
    <span>${d ? `Signals as of the end of ${esc(d.as_of_label)}.${d.generated_at ? ` Generated ${esc(d.generated_at)}.` : ''} Updated monthly.` : ''}</span>
    <span>Educational model output, the same for every visitor. It is not personal investment advice and does not know your goals, taxes or time horizon.</span>
    <span>This product uses the FRED&reg; API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
      <a href="https://fred.stlouisfed.org/docs/api/terms_of_use.html">FRED API terms of use</a>.</span>`;
}

function sampleBanner(d) {
  return d && d.sample ? '<div class="banner">Sample data. These numbers are synthetic and for layout preview only. Run the monthly workflow to load real data.</div>' : '';
}

function showError(e) {
  $('#app').innerHTML = `<div class="loading">No signal data yet. Run the "Macro Monthly" workflow to generate it.<br><span class="hint">${esc(e.message)}</span></div>`;
}

// Mirrors model_allocation() in src/frameworks.py.
function modelAllocation(neutral, equityOffset, altTilt, defensive, cashShareRule) {
  const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
  const stocks = clamp(neutral.stocks + (equityOffset || 0), 0, 100);
  const alts = clamp(neutral.alternatives + altTilt, 0, 100 - stocks);
  const rest = 100 - stocks - alts;
  let share = cashShareRule[defensive];
  if (share == null) {
    const base = neutral.bonds + neutral.cash;
    share = base ? neutral.cash / base : 0;
  }
  const cash = Math.round(rest * share);
  return { stocks: Math.round(stocks), bonds: Math.round(rest - cash), cash, alternatives: Math.round(alts) };
}

function mixBar(mix) {
  return `<div class="mix">${ASSETS.filter((a) => mix[a] > 0).map((a) =>
    `<div class="a-${a}" style="width:${mix[a]}%" title="${ASSET_NAME[a]} ${mix[a]}%">${mix[a] >= 8 ? mix[a] + '%' : ''}</div>`).join('')}</div>
    <div class="legend">${ASSETS.map((a) => `<span><i class="a-${a}"></i>${ASSET_NAME[a]} ${mix[a]}%</span>`).join('')}</div>`;
}

function miniTable(tb) {
  if (!tb || !tb.rows || !tb.rows.length) return '';
  return `<div class="scroll"><table class="mini"><tr>${tb.columns.map((c) => `<th>${esc(c)}</th>`).join('')}</tr>${
    tb.rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</table></div>`;
}

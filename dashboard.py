"""NexusQuant Alpha Arena — Multi-Agent Real-Time Trading Dashboard.

Launch with:
    conda run -n trading_bot streamlit run dashboard.py
"""

from __future__ import annotations

import logging
import time
from typing import Any

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NexusQuant · Alpha Arena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS — dark premium theme ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Global background ── */
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 50%, #0a1628 100%); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
    border-right: 1px solid #1e2d40;
}
[data-testid="stSidebar"] * { color: #c9d6e3 !important; }

/* ── Agent cards ── */
.agent-card {
    background: linear-gradient(145deg, #111827, #0f1e32);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: border-color 0.3s ease;
}
.agent-card:hover { border-color: #3b82f6; }

/* ── Signal badges ── */
.badge-buy  { background:#064e3b; color:#34d399; border:1px solid #059669; padding:4px 14px; border-radius:20px; font-weight:600; font-size:14px; }
.badge-sell { background:#450a0a; color:#f87171; border:1px solid #dc2626; padding:4px 14px; border-radius:20px; font-weight:600; font-size:14px; }
.badge-hold { background:#1c1917; color:#fbbf24; border:1px solid #d97706; padding:4px 14px; border-radius:20px; font-weight:600; font-size:14px; }

/* ── Metric value ── */
.big-metric { font-size:32px; font-weight:700; font-family:'JetBrains Mono',monospace; color:#60a5fa; letter-spacing:-1px; }
.metric-label { font-size:11px; font-weight:500; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }

/* ── Confidence bar ── */
.conf-bar-wrap { background:#1e293b; border-radius:6px; height:8px; margin:8px 0 4px 0; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:6px; background:linear-gradient(90deg,#3b82f6,#818cf8); transition:width 0.5s ease; }

/* ── Section header ── */
.section-header {
    font-size:13px; font-weight:600; color:#3b82f6;
    text-transform:uppercase; letter-spacing:2px;
    border-left:3px solid #3b82f6; padding-left:10px;
    margin:16px 0 10px 0;
}

/* ── Reason text ── */
.reason-box {
    background:#0f172a; border:1px solid #1e293b; border-radius:8px;
    padding:10px 14px; font-size:13px; color:#94a3b8;
    font-style:italic; line-height:1.6; margin-top:10px;
    min-height:52px;
}

/* ── Global metrics bar ── */
.global-bar {
    background:linear-gradient(90deg,#0f1e32,#111827);
    border:1px solid #1e3a5f; border-radius:12px;
    padding:16px 24px; margin-bottom:24px;
    display:flex; gap:40px; align-items:center;
}

/* ── Agent name header ── */
.agent-name { font-size:18px; font-weight:700; color:#e2e8f0; margin:0 0 4px 0; }
.agent-type { font-size:11px; color:#475569; font-weight:500; text-transform:uppercase; letter-spacing:1px; }

/* ── Error box ── */
.err-box { background:#450a0a; border:1px solid #991b1b; border-radius:8px; padding:12px 16px; color:#fca5a5; font-size:13px; }

/* ── Button override ── */
div[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#1d4ed8,#4f46e5) !important;
    color:white !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important;
    width:100% !important; padding:12px !important;
    font-size:14px !important; letter-spacing:0.5px !important;
    transition:all 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background:linear-gradient(135deg,#2563eb,#6366f1) !important;
    box-shadow:0 0 20px rgba(99,102,241,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Logging (suppressed in dashboard context) ─────────────────────────────────
logging.basicConfig(level=logging.WARNING)

# ── Agent metadata ────────────────────────────────────────────────────────────
AGENT_META: dict[str, dict] = {
    "DimmerForce": {
        "type": "LLM · Gemma 4",
        "icon": "🤖",
        "color": "#818cf8",
        "description": "Gemma 4 via Ollama — prompt-driven indicator reasoning",
    },
    "Savinov": {
        "type": "Statistical · RSI/EMA",
        "icon": "📊",
        "color": "#34d399",
        "description": "Momentum / mean-reversion with RSI & EMA alignment",
    },
    "DeepAlpha": {
        "type": "Neural Net · LSTM (stub)",
        "icon": "🧠",
        "color": "#f472b6",
        "description": "MACD-momentum proxy — full LSTM model in Phase 6",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _badge_html(action: str) -> str:
    cls = {"BUY": "badge-buy", "SELL": "badge-sell"}.get(action, "badge-hold")
    return f'<span class="{cls}">{action}</span>'


def _conf_bar(confidence: float) -> str:
    pct = int(confidence * 100)
    color = "#34d399" if confidence >= 0.7 else "#fbbf24" if confidence >= 0.5 else "#f87171"
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
        f'<span style="font-size:12px;color:{color};font-weight:600;">{pct}% confidence</span>'
    )


def _render_signal(signal: dict | None, error: str | None) -> None:
    """Renders a signal result card inside an agent column."""
    if error:
        st.markdown(f'<div class="err-box">⚠️ {error}</div>', unsafe_allow_html=True)
        return
    if signal is None:
        st.markdown('<div class="reason-box">No signal yet — click Run Simulation Tick.</div>',
                    unsafe_allow_html=True)
        return

    action = signal.get("action", "HOLD")
    confidence = float(signal.get("confidence", 0.0))
    reason = signal.get("reason", "—")

    st.markdown(_badge_html(action), unsafe_allow_html=True)
    st.markdown(_conf_bar(confidence), unsafe_allow_html=True)
    st.markdown(f'<div class="reason-box">💬 {reason}</div>', unsafe_allow_html=True)

    extras = {k: v for k, v in signal.items()
              if k not in {"action", "confidence", "reason", "agent", "model", "strategy",
                           "model_version", "volatility_regime"}}
    if extras:
        with st.expander("📎 Extra metadata", expanded=False):
            for k, v in extras.items():
                st.markdown(
                    f'<span style="color:#475569;font-size:12px;">{k}:</span> '
                    f'<span style="color:#94a3b8;font-family:monospace;">{v}</span>',
                    unsafe_allow_html=True,
                )


@st.cache_resource(show_spinner=False)
def _load_pipeline_modules():
    """Imports heavy project modules once and caches them for the session."""
    from src.agents import DeepAlphaAgent, DimmerForceAgent, SavinovAgent
    from src.data.features import FeatureEngineer
    from src.data.fetcher import MarketDataFetcher
    return MarketDataFetcher, FeatureEngineer, DimmerForceAgent, SavinovAgent, DeepAlphaAgent


# ── Session state defaults ────────────────────────────────────────────────────
def _init_state(allocations: dict[str, float]) -> None:
    if "signals" not in st.session_state:
        st.session_state.signals = {name: None for name in AGENT_META}
    if "errors" not in st.session_state:
        st.session_state.errors = {name: None for name in AGENT_META}
    if "balances" not in st.session_state:
        st.session_state.balances = dict(allocations)
    if "tick_count" not in st.session_state:
        st.session_state.tick_count = 0
    if "last_price" not in st.session_state:
        st.session_state.last_price = None
    if "last_tick_ms" not in st.session_state:
        st.session_state.last_tick_ms = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NexusQuant")
    st.markdown('<div style="color:#475569;font-size:12px;margin-bottom:20px;">Alpha Arena · v0.4</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">Global Settings</div>', unsafe_allow_html=True)

    # NOTE: This slider value is read-only in the UI for now.
    # Phase 6 will add a "Save to config" button that writes this value back
    # to config/settings.yaml via yaml.safe_dump() so all agents pick it up
    # on the next RiskManager instantiation.
    risk_pct = st.slider(
        "Risk Per Trade (%)",
        min_value=1.0, max_value=10.0, value=2.0, step=0.5,
        help="Percentage of each agent's capital risked per trade. "
             "Will be persisted to config/settings.yaml in Phase 6.",
    )

    symbol = st.selectbox("Trading Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT"], index=0)
    timeframe = st.selectbox("Timeframe", ["1h", "15m", "4h"], index=0)
    candle_limit = st.slider("Candles to Fetch", min_value=60, max_value=300, value=100, step=10)

    st.markdown('<div class="section-header">Agent Allocation (USDT)</div>',
                unsafe_allow_html=True)

    alloc_dimmer  = st.number_input("DimmerForce",  min_value=0.0, max_value=10000.0, value=50.0, step=5.0)
    alloc_savinov = st.number_input("Savinov",      min_value=0.0, max_value=10000.0, value=50.0, step=5.0)
    alloc_deep    = st.number_input("DeepAlpha",    min_value=0.0, max_value=10000.0, value=50.0, step=5.0)

    allocations = {
        "DimmerForce": alloc_dimmer,
        "Savinov":     alloc_savinov,
        "DeepAlpha":   alloc_deep,
    }

    total_capital = alloc_dimmer + alloc_savinov + alloc_deep
    st.markdown(
        f'<div style="margin-top:12px;padding:12px;background:#0f172a;border-radius:8px;border:1px solid #1e293b;">'
        f'<div class="metric-label">Total Capital</div>'
        f'<div style="font-size:22px;font-weight:700;color:#60a5fa;font-family:monospace;">'
        f'${total_capital:,.2f} USDT</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#334155;line-height:1.6;">'
                '⚠️ Paper trading only. No real funds at risk.</div>', unsafe_allow_html=True)


# ── Init state ────────────────────────────────────────────────────────────────
_init_state(allocations)

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-size:32px;font-weight:700;color:#e2e8f0;margin-bottom:4px;">'
    '⚡ NexusQuant <span style="color:#3b82f6;">Alpha Arena</span></h1>'
    '<p style="color:#475569;font-size:14px;margin-bottom:24px;">'
    'Multi-agent paper trading dashboard — Gemma 4 × Statistical × Neural Net</p>',
    unsafe_allow_html=True,
)

# ── Global status bar ─────────────────────────────────────────────────────────
tick   = st.session_state.tick_count
price  = st.session_state.last_price
tick_t = st.session_state.last_tick_ms

g1, g2, g3, g4 = st.columns(4)
with g1:
    st.metric("Total Capital", f"${total_capital:,.2f}")
with g2:
    st.metric("Simulation Ticks", str(tick))
with g3:
    st.metric("Last BTC Price", f"${price:,.2f}" if price else "—")
with g4:
    st.metric("Last Tick Duration", f"{tick_t:.1f}s" if tick_t else "—")

st.markdown("---")

# ── Run button ────────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    run_clicked = st.button("▶  Run 1 Simulation Tick", key="run_tick")

# ── Simulation tick logic ─────────────────────────────────────────────────────
if run_clicked:
    tick_start = time.time()
    MarketDataFetcher, FeatureEngineer, DimmerForceAgent, SavinovAgent, DeepAlphaAgent = (
        _load_pipeline_modules()
    )

    # ── Fetch & enrich ────────────────────────────────────────────────────────
    with st.spinner(f"Fetching {candle_limit} candles of {symbol} [{timeframe}]…"):
        try:
            fetcher = MarketDataFetcher()
            raw_df  = fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=candle_limit)
            enriched_df = FeatureEngineer().add_technical_indicators(raw_df)
            st.session_state.last_price = float(enriched_df["close"].iloc[-1])
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            st.stop()

    # ── Run each agent ────────────────────────────────────────────────────────
    agents_map = {
        "DimmerForce": DimmerForceAgent(),
        "Savinov":     SavinovAgent(),
        "DeepAlpha":   DeepAlphaAgent(),
    }

    for agent_name, agent_obj in agents_map.items():
        with st.spinner(f"[{agent_name}] Generating signal…"):
            try:
                sig = agent_obj.generate_signal(enriched_df)
                st.session_state.signals[agent_name] = sig
                st.session_state.errors[agent_name]  = None
            except Exception as exc:
                st.session_state.signals[agent_name] = None
                st.session_state.errors[agent_name]  = str(exc)

    st.session_state.tick_count   += 1
    st.session_state.last_tick_ms  = time.time() - tick_start
    st.rerun()

# ── Agent columns ─────────────────────────────────────────────────────────────
col_d, col_s, col_a = st.columns(3)
columns_map = {"DimmerForce": col_d, "Savinov": col_s, "DeepAlpha": col_a}

for agent_name, col in columns_map.items():
    meta    = AGENT_META[agent_name]
    signal  = st.session_state.signals.get(agent_name)
    error   = st.session_state.errors.get(agent_name)
    balance = st.session_state.balances.get(agent_name, allocations[agent_name])

    with col:
        st.markdown(
            f'<div class="agent-card">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">'
            f'<span style="font-size:28px;">{meta["icon"]}</span>'
            f'<div><div class="agent-name">{agent_name}</div>'
            f'<div class="agent-type">{meta["type"]}</div></div></div>'
            f'<div class="metric-label">Paper Balance</div>'
            f'<div class="big-metric">${balance:,.2f}</div>'
            f'<div style="font-size:11px;color:#334155;margin-top:2px;">'
            f'Allocated: ${allocations[agent_name]:,.2f} USDT</div>'
            f'<hr style="border:none;border-top:1px solid #1e293b;margin:14px 0;">'
            f'<div class="section-header">Latest Signal</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _render_signal(signal, error)

        if signal:
            with st.expander("📈 Market snapshot", expanded=False):
                from src.data.features import FeatureEngineer as _FE  # already imported
                cols_show = ["close", "RSI_14", "EMA_20", "EMA_50", "ATRr_14"]
                try:
                    MarketDataFetcher2, FeatureEngineer2, *_ = _load_pipeline_modules()
                    # Reuse already-enriched data cached in session_state
                    if "enriched_df" not in st.session_state:
                        st.info("Run a tick first to populate market data.")
                    else:
                        tail = st.session_state["enriched_df"]
                        avail = [c for c in cols_show if c in tail.columns]
                        st.dataframe(tail[avail].tail(5).round(4), use_container_width=True)
                except Exception:
                    st.info("Market snapshot available after first tick.")

# ── Cache enriched_df in session state for snapshot expanders ─────────────────
if run_clicked and "enriched_df" not in st.session_state:
    st.session_state["enriched_df"] = enriched_df  # type: ignore[name-defined]

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:11px;color:#1e293b;padding:8px 0;">'
    'NexusQuant Alpha Arena · Paper Trading Only · Not Financial Advice'
    '</div>',
    unsafe_allow_html=True,
)

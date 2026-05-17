"""NexusQuant Alpha Arena — Multi-Agent Real-Time Trading Dashboard.

Three Gemma 4 personas reason over the same market data simultaneously,
each guided by a distinct quantitative trading philosophy.

Launch with:
    conda run -n trading_bot streamlit run dashboard.py
"""

from __future__ import annotations

import logging
import time

import streamlit as st

st.set_page_config(
    page_title="NexusQuant · Alpha Arena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #060b18 0%, #0a0f1e 50%, #080d1a 100%); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d1a 0%, #0f1424 100%) !important;
    border-right: 1px solid #1a2740;
}
[data-testid="stSidebar"] * { color: #c9d6e3 !important; }
[data-testid="stSidebar"] .stSlider * { color: #c9d6e3 !important; }

.agent-card {
    background: linear-gradient(145deg, #0e1628, #0b1220);
    border-radius: 18px;
    padding: 22px 20px 16px 20px;
    margin-bottom: 14px;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.agent-card-dimmer  { border: 1px solid #2d3faa; box-shadow: 0 4px 28px rgba(99,102,241,0.12); }
.agent-card-zenith  { border: 1px solid #065f46; box-shadow: 0 4px 28px rgba(16,185,129,0.12); }
.agent-card-aegis   { border: 1px solid #7c1c6e; box-shadow: 0 4px 28px rgba(236,72,153,0.12); }
.agent-card-dimmer:hover  { border-color:#6366f1; box-shadow:0 4px 32px rgba(99,102,241,0.25); }
.agent-card-zenith:hover  { border-color:#10b981; box-shadow:0 4px 32px rgba(16,185,129,0.25); }
.agent-card-aegis:hover   { border-color:#ec4899; box-shadow:0 4px 32px rgba(236,72,153,0.25); }

.badge-buy  { background:#064e3b; color:#34d399; border:1px solid #059669; padding:5px 16px; border-radius:20px; font-weight:700; font-size:15px; letter-spacing:1px; }
.badge-sell { background:#450a0a; color:#f87171; border:1px solid #dc2626; padding:5px 16px; border-radius:20px; font-weight:700; font-size:15px; letter-spacing:1px; }
.badge-hold { background:#1c1208; color:#fbbf24; border:1px solid #d97706; padding:5px 16px; border-radius:20px; font-weight:700; font-size:15px; letter-spacing:1px; }

.big-metric { font-size:30px; font-weight:700; font-family:'JetBrains Mono',monospace; letter-spacing:-1px; }
.metric-label { font-size:10px; font-weight:600; color:#4b5563; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:3px; }

.conf-bar-wrap { background:#1a2130; border-radius:8px; height:10px; margin:10px 0 5px 0; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:8px; transition:width 0.6s ease; }

.reason-box {
    background:#070d1a; border:1px solid #1a2740; border-radius:10px;
    padding:11px 14px; font-size:13px; color:#8899aa;
    font-style:italic; line-height:1.65; margin-top:12px; min-height:56px;
}

.persona-tag {
    display:inline-block; font-size:10px; font-weight:600;
    text-transform:uppercase; letter-spacing:1.5px;
    padding:3px 10px; border-radius:12px; margin-bottom:12px;
}
.persona-dimmer { background:#1e1f5e; color:#818cf8; }
.persona-zenith  { background:#064030; color:#34d399; }
.persona-aegis   { background:#4a0d3f; color:#f472b6; }

.agent-name { font-size:20px; font-weight:700; color:#e2e8f0; margin:0 0 2px 0; }

.err-box { background:#3b0a0a; border:1px solid #7f1d1d; border-radius:8px; padding:11px 15px; color:#fca5a5; font-size:13px; margin-top:8px; }
.pending-box { background:#111827; border:1px solid #1f2937; border-radius:8px; padding:14px; color:#374151; font-size:13px; text-align:center; margin-top:8px; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg,#1d4ed8,#4f46e5) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; width: 100% !important; padding: 14px !important;
    font-size: 15px !important; letter-spacing: 0.5px !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg,#2563eb,#7c3aed) !important;
    box-shadow: 0 0 24px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.WARNING)

# ── Agent roster ──────────────────────────────────────────────────────────────
AGENT_META: dict[str, dict] = {
    "DimmerForce": {
        "icon": "📈",
        "color": "#818cf8",
        "persona": "Trend Follower",
        "persona_class": "persona-dimmer",
        "card_class": "agent-card-dimmer",
        "balance_color": "#818cf8",
        "description": "Rides momentum via MACD & EMA alignment",
    },
    "Zenith": {
        "icon": "🔄",
        "color": "#34d399",
        "persona": "Mean Reversion",
        "persona_class": "persona-zenith",
        "card_class": "agent-card-zenith",
        "balance_color": "#34d399",
        "description": "Fades extremes using RSI as primary signal",
    },
    "Aegis": {
        "icon": "🛡️",
        "color": "#f472b6",
        "persona": "Conservative",
        "persona_class": "persona-aegis",
        "card_class": "agent-card-aegis",
        "balance_color": "#f472b6",
        "description": "HOLDs unless ALL indicators align perfectly",
    },
}


def _badge(action: str) -> str:
    cls = {"BUY": "badge-buy", "SELL": "badge-sell"}.get(action, "badge-hold")
    return f'<span class="{cls}">{action}</span>'


def _conf_bar(confidence: float, color: str) -> str:
    pct = int(confidence * 100)
    bar_color = "#34d399" if confidence >= 0.7 else "#fbbf24" if confidence >= 0.5 else "#f87171"
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill" style="width:{pct}%;background:{bar_color};"></div>'
        f'</div>'
        f'<span style="font-size:12px;color:{bar_color};font-weight:600;">{pct}% confidence</span>'
    )


def _render_signal_card(signal: dict | None, error: str | None, meta: dict) -> None:
    """Renders the signal result block within an agent column."""
    if error:
        st.markdown(f'<div class="err-box">⚠️ {error}</div>', unsafe_allow_html=True)
        return
    if signal is None:
        st.markdown(
            '<div class="pending-box">⏳ No signal yet — click <b>Run Simulation Tick</b></div>',
            unsafe_allow_html=True,
        )
        return

    action = signal.get("action", "HOLD")
    confidence = float(signal.get("confidence", 0.0))
    reason = signal.get("reason", "—")

    st.markdown(_badge(action), unsafe_allow_html=True)
    st.markdown(_conf_bar(confidence, meta["color"]), unsafe_allow_html=True)
    st.markdown(f'<div class="reason-box">💬 {reason}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    """Loads all heavy modules once per session."""
    from src.agents import AegisAgent, DimmerForceAgent, ZenithAgent
    from src.data.features import FeatureEngineer
    from src.data.fetcher import MarketDataFetcher
    return MarketDataFetcher, FeatureEngineer, DimmerForceAgent, ZenithAgent, AegisAgent


def _init_state(allocations: dict[str, float]) -> None:
    if "signals" not in st.session_state:
        st.session_state.signals = {k: None for k in AGENT_META}
    if "errors" not in st.session_state:
        st.session_state.errors = {k: None for k in AGENT_META}
    if "balances" not in st.session_state:
        st.session_state.balances = dict(allocations)
    if "tick_count" not in st.session_state:
        st.session_state.tick_count = 0
    if "last_price" not in st.session_state:
        st.session_state.last_price = None
    if "last_tick_s" not in st.session_state:
        st.session_state.last_tick_s = None
    if "enriched_df" not in st.session_state:
        st.session_state.enriched_df = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NexusQuant")
    st.markdown(
        '<div style="color:#374151;font-size:12px;margin-bottom:20px;">'
        'Alpha Arena · v0.5 · 3× Gemma 4</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Global Settings")

    # NOTE: This slider is display-only in Phase 5.
    # Phase 6 will add a "Save" button that writes the value back to
    # config/settings.yaml via yaml.safe_dump() so RiskManager picks it up.
    risk_pct = st.slider(
        "Risk Per Trade (%)", min_value=1.0, max_value=10.0, value=2.0, step=0.5,
        help="Will persist to config/settings.yaml in Phase 6.",
    )

    symbol = st.selectbox("Trading Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT"], index=0)
    timeframe = st.selectbox("Timeframe", ["1h", "15m", "4h"], index=0)
    candle_limit = st.slider("Candles to Fetch", 60, 300, 100, step=10)

    st.markdown("### Agent Allocation (USDT)")
    alloc = {
        "DimmerForce": st.number_input("DimmerForce", 0.0, 10000.0, 50.0, step=5.0),
        "Zenith":      st.number_input("Zenith",      0.0, 10000.0, 50.0, step=5.0),
        "Aegis":       st.number_input("Aegis",       0.0, 10000.0, 50.0, step=5.0),
    }
    total_capital = sum(alloc.values())
    st.markdown(
        f'<div style="margin-top:12px;padding:12px;background:#070d1a;border-radius:10px;'
        f'border:1px solid #1a2740;">'
        f'<div class="metric-label">Total Capital</div>'
        f'<div style="font-size:22px;font-weight:700;color:#60a5fa;font-family:monospace;">'
        f'${total_capital:,.2f} USDT</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#1f2937;line-height:1.7;">'
        '⚠️ Paper trading only — no real funds at risk.<br>'
        '🤖 All three agents use Gemma 4 (local Ollama).<br>'
        '💡 Each persona has a unique system prompt.</div>',
        unsafe_allow_html=True,
    )


# ── Init ─────────────────────────────────────────────────────────────────────
_init_state(alloc)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-size:34px;font-weight:800;color:#e2e8f0;margin-bottom:4px;">'
    '⚡ NexusQuant <span style="color:#6366f1;">Alpha Arena</span></h1>'
    '<p style="color:#374151;font-size:14px;margin-bottom:20px;">'
    'Three <b style="color:#818cf8;">Gemma 4</b> personas · '
    '<b style="color:#34d399;">Trend</b> vs '
    '<b style="color:#34d399;">Mean-Reversion</b> vs '
    '<b style="color:#f472b6;">Conservative</b> · '
    'Same model, different minds</p>',
    unsafe_allow_html=True,
)

# ── Global metrics ────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
price = st.session_state.last_price
tick_s = st.session_state.last_tick_s
m1.metric("Total Capital", f"${total_capital:,.2f}")
m2.metric("Simulation Ticks", st.session_state.tick_count)
m3.metric("Last BTC Price", f"${price:,.2f}" if price else "—")
m4.metric("Last Tick", f"{tick_s:.1f}s" if tick_s else "—")

st.markdown("---")

# ── Run button ────────────────────────────────────────────────────────────────
btn_col, info_col = st.columns([1, 3])
with btn_col:
    run_clicked = st.button("▶  Run Simulation Tick", key="run_tick")
with info_col:
    st.markdown(
        '<div style="padding:12px 0;color:#374151;font-size:13px;">'
        '🔗 Fetches live OHLCV → FeatureEngineer → '
        'DimmerForce (trend) + Zenith (reversion) + Aegis (conservative) in sequence.</div>',
        unsafe_allow_html=True,
    )

# ── Tick execution ────────────────────────────────────────────────────────────
if run_clicked:
    t0 = time.time()
    MarketDataFetcher, FeatureEngineer, DimmerForceAgent, ZenithAgent, AegisAgent = (
        _load_pipeline()
    )

    with st.spinner(f"Fetching {candle_limit} candles of {symbol} [{timeframe}]…"):
        try:
            raw_df = MarketDataFetcher().fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, limit=candle_limit
            )
            enriched_df = FeatureEngineer().add_technical_indicators(raw_df)
            st.session_state.last_price = float(enriched_df["close"].iloc[-1])
            st.session_state.enriched_df = enriched_df
        except Exception as exc:
            st.error(f"❌ Data fetch failed: {exc}")
            st.stop()

    agents_map = {
        "DimmerForce": DimmerForceAgent(),
        "Zenith":      ZenithAgent(),
        "Aegis":       AegisAgent(),
    }

    for agent_name, agent_obj in agents_map.items():
        with st.spinner(f"[{agent_name}] Querying Gemma 4…"):
            try:
                sig = agent_obj.generate_signal(enriched_df)
                st.session_state.signals[agent_name] = sig
                st.session_state.errors[agent_name]  = None
            except Exception as exc:
                st.session_state.signals[agent_name] = None
                st.session_state.errors[agent_name]  = str(exc)

    st.session_state.tick_count += 1
    st.session_state.last_tick_s = time.time() - t0
    st.rerun()

# ── Agent columns ─────────────────────────────────────────────────────────────
col_d, col_z, col_a = st.columns(3)
col_map = {"DimmerForce": col_d, "Zenith": col_z, "Aegis": col_a}

for agent_name, col in col_map.items():
    meta    = AGENT_META[agent_name]
    signal  = st.session_state.signals.get(agent_name)
    error   = st.session_state.errors.get(agent_name)
    balance = st.session_state.balances.get(agent_name, alloc[agent_name])

    with col:
        st.markdown(
            f'<div class="agent-card {meta["card_class"]}">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
            f'<span style="font-size:30px;">{meta["icon"]}</span>'
            f'<div><div class="agent-name">{agent_name}</div></div></div>'
            f'<span class="persona-tag {meta["persona_class"]}">{meta["persona"]}</span><br>'
            f'<div style="font-size:11px;color:#374151;margin-bottom:14px;">{meta["description"]}</div>'
            f'<div class="metric-label">Paper Balance</div>'
            f'<div class="big-metric" style="color:{meta["balance_color"]};">'
            f'${balance:,.2f}</div>'
            f'<div style="font-size:11px;color:#1f2937;margin-top:2px;margin-bottom:14px;">'
            f'Allocated: ${alloc[agent_name]:,.2f} USDT</div>'
            f'<hr style="border:none;border-top:1px solid #111827;margin:10px 0 12px 0;">'
            f'<div style="font-size:10px;font-weight:600;color:{meta["color"]};'
            f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">'
            f'Latest Signal</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _render_signal_card(signal, error, meta)

        if signal and st.session_state.enriched_df is not None:
            with st.expander("📊 Market snapshot", expanded=False):
                df = st.session_state.enriched_df
                cols_show = [c for c in
                             ["close", "RSI_14", "MACD_12_26_9", "MACDh_12_26_9",
                              "EMA_20", "EMA_50", "ATRr_14"]
                             if c in df.columns]
                st.dataframe(df[cols_show].tail(5).round(4), use_container_width=True)

# ── Comparison table ──────────────────────────────────────────────────────────
any_signal = any(s is not None for s in st.session_state.signals.values())
if any_signal:
    st.markdown("---")
    st.markdown(
        '<div style="font-size:10px;font-weight:600;color:#374151;'
        'text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">'
        'Signal Comparison Matrix</div>',
        unsafe_allow_html=True,
    )
    comparison_cols = st.columns(3)
    for i, (agent_name, meta) in enumerate(AGENT_META.items()):
        sig = st.session_state.signals.get(agent_name)
        with comparison_cols[i]:
            if sig:
                action = sig.get("action", "—")
                conf   = sig.get("confidence", 0.0)
                color  = {"BUY": "#34d399", "SELL": "#f87171"}.get(action, "#fbbf24")
                st.markdown(
                    f'<div style="text-align:center;padding:14px;background:#0a101e;'
                    f'border-radius:12px;border:1px solid #1a2740;">'
                    f'<div style="font-size:12px;color:#374151;font-weight:600;">{agent_name}</div>'
                    f'<div style="font-size:26px;font-weight:800;color:{color};'
                    f'font-family:monospace;margin:6px 0;">{action}</div>'
                    f'<div style="font-size:13px;color:#4b5563;">{int(conf*100)}% confidence</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:11px;color:#111827;padding:6px 0;">'
    'NexusQuant Alpha Arena · Paper Trading Only · Not Financial Advice · '
    'Powered by Gemma 4 (local)</div>',
    unsafe_allow_html=True,
)

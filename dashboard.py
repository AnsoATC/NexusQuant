"""NexusQuant Alpha Arena — Session-Based Live Trading Dashboard.

Architecture note: uses the st.rerun() state-machine pattern instead of
a blocking while-loop so the STOP button always works.

Launch:
    conda run -n trading_bot streamlit run dashboard.py
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timezone

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NexusQuant · Alpha Arena", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#060b18 0%,#0a0f1e 50%,#080d1a 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#080d1a 0%,#0f1424 100%) !important;border-right:1px solid #1a2740;}
[data-testid="stSidebar"] *{color:#c9d6e3 !important;}
.agent-card{background:linear-gradient(145deg,#0e1628,#0b1220);border-radius:16px;padding:18px 16px;margin-bottom:12px;}
.card-dimmer{border:1px solid #2d3faa;} .card-zenith{border:1px solid #065f46;} .card-aegis{border:1px solid #7c1c6e;}
.badge-buy{background:#064e3b;color:#34d399;border:1px solid #059669;padding:4px 14px;border-radius:20px;font-weight:700;font-size:14px;}
.badge-sell{background:#450a0a;color:#f87171;border:1px solid #dc2626;padding:4px 14px;border-radius:20px;font-weight:700;font-size:14px;}
.badge-hold{background:#1c1208;color:#fbbf24;border:1px solid #d97706;padding:4px 14px;border-radius:20px;font-weight:700;font-size:14px;}
.conf-wrap{background:#1a2130;border-radius:6px;height:8px;margin:8px 0 4px;overflow:hidden;}
.conf-fill{height:100%;border-radius:6px;}
.reason-box{background:#070d1a;border:1px solid #1a2740;border-radius:8px;padding:10px 13px;font-size:12px;color:#8899aa;font-style:italic;margin-top:10px;min-height:48px;}
.err-box{background:#3b0a0a;border:1px solid #7f1d1d;border-radius:8px;padding:10px;color:#fca5a5;font-size:12px;margin-top:8px;}
.log-entry{font-family:'JetBrains Mono',monospace;font-size:11px;color:#64748b;padding:2px 0;border-bottom:1px solid #0f1628;}
.log-buy{color:#34d399;} .log-sell{color:#f87171;} .log-err{color:#f87171;} .log-info{color:#60a5fa;}
.session-active{background:#064e3b;border:1px solid #059669;border-radius:10px;padding:10px 16px;color:#34d399;font-weight:600;font-size:14px;margin-bottom:12px;}
.session-stopped{background:#1c1208;border:1px solid #d97706;border-radius:10px;padding:10px 16px;color:#fbbf24;font-weight:600;font-size:14px;margin-bottom:12px;}
.mode-badge{background:#064e3b;color:#34d399;border:1px solid #059669;padding:4px 14px;border-radius:16px;font-size:12px;font-weight:700;}
div[data-testid="stButton"]>button{background:linear-gradient(135deg,#1d4ed8,#4f46e5) !important;color:white !important;border:none !important;border-radius:10px !important;font-weight:700 !important;width:100% !important;padding:12px !important;font-size:14px !important;}
.stop-btn div[data-testid="stButton"]>button{background:linear-gradient(135deg,#991b1b,#7f1d1d) !important;}
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.WARNING)

# ── Agent metadata ─────────────────────────────────────────────────────────────
AGENT_META = {
    "DimmerForce": {"icon":"📈","color":"#818cf8","persona":"Trend Follower","card":"card-dimmer","executes":True},
    "Zenith":      {"icon":"🔄","color":"#34d399","persona":"Mean Reversion","card":"card-zenith","executes":False},
    "Aegis":       {"icon":"🛡️","color":"#f472b6","persona":"Conservative",  "card":"card-aegis", "executes":False},
}
EXEC_AGENT = "DimmerForce"   # the sole order-executing agent


# ── Session state initialisation ───────────────────────────────────────────────
def _init():
    defaults = {
        "session_active": False,
        "session_start_ts": None,
        "session_duration_h": 4.0,
        "tick_interval_s": 300,
        "tick_count": 0,
        "signals": {k: None for k in AGENT_META},
        "errors":  {k: None for k in AGENT_META},
        "last_price": None,
        "last_tick_s": None,
        "enriched_df": None,
        "order_log": [],
        "broker": None,
        "testnet_balance": None,
        # ── Performance tracking (reset on each new session start) ──────────
        # signal_counts: per-agent lifetime action counters for the session.
        "signal_counts": {k: {"BUY": 0, "SELL": 0, "HOLD": 0} for k in AGENT_META},
        # performance_history: list of dicts appended once per tick.
        # Keys: timestamp, btc_price, buy_hold_equity,
        #       DimmerForce_equity, Zenith_equity, Aegis_equity.
        "performance_history": [],
        # initial_btc_price: price captured on Tick 1 for Buy&Hold baseline.
        "initial_btc_price": None,
        # per-agent simulated equity (starts at each agent's allocation).
        "agent_equity": {k: 50.0 for k in AGENT_META},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ── Cached pipeline loader ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from src.agents import AegisAgent, DimmerForceAgent, ZenithAgent
    from src.data.features import FeatureEngineer
    from src.data.fetcher import MarketDataFetcher
    return MarketDataFetcher, FeatureEngineer, DimmerForceAgent, ZenithAgent, AegisAgent


# ── Helpers ────────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def _log(msg: str, kind: str = "info") -> None:
    entry = f'<div class="log-entry log-{kind}">[{_ts()}] {msg}</div>'
    st.session_state.order_log.insert(0, entry)
    if len(st.session_state.order_log) > 100:
        st.session_state.order_log.pop()

def _badge(action: str) -> str:
    cls = {"BUY":"badge-buy","SELL":"badge-sell"}.get(action,"badge-hold")
    return f'<span class="{cls}">{action}</span>'

def _conf_bar(conf: float) -> str:
    pct = int(conf * 100)
    c = "#34d399" if conf >= 0.7 else "#fbbf24" if conf >= 0.5 else "#f87171"
    return (f'<div class="conf-wrap"><div class="conf-fill" style="width:{pct}%;background:{c};"></div></div>'
            f'<span style="font-size:11px;color:{c};font-weight:600;">{pct}% confidence</span>')

def _get_broker():
    if st.session_state.broker is None:
        from src.execution.broker import BinanceBroker
        st.session_state.broker = BinanceBroker()
    return st.session_state.broker

def _elapsed_s() -> float:
    if st.session_state.session_start_ts is None:
        return 0.0
    return time.time() - st.session_state.session_start_ts

def _session_expired() -> bool:
    return _elapsed_s() >= st.session_state.session_duration_h * 3600


def _build_candlestick_chart(symbol: str) -> go.Figure | None:
    """Builds a dark-themed Plotly candlestick figure with EMA_20 and EMA_50 overlays.

    Returns None when no enriched DataFrame is available yet (pre-first-tick).

    Args:
        symbol: Trading pair label used as the chart title (e.g. ``"BTC/USDT"``).

    Returns:
        A configured :class:`plotly.graph_objects.Figure`, or ``None``.
    """
    df = st.session_state.enriched_df
    if df is None or df.empty:
        return None

    # Use the last 60 candles for readability — keep the chart uncluttered
    plot_df = df.tail(60).copy()
    timestamps = plot_df.index.astype(str).str[:19]  # trim timezone noise

    fig = go.Figure()

    # ── Candlestick body ──────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=timestamps,
        open=plot_df["open"],
        high=plot_df["high"],
        low=plot_df["low"],
        close=plot_df["close"],
        name=symbol,
        increasing_line_color="#34d399",
        decreasing_line_color="#f87171",
        increasing_fillcolor="rgba(52,211,153,0.25)",
        decreasing_fillcolor="rgba(248,113,113,0.25)",
        whiskerwidth=0.4,
    ))

    # ── EMA_20 overlay ────────────────────────────────────────────────────────
    if "EMA_20" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=timestamps, y=plot_df["EMA_20"],
            mode="lines", name="EMA 20",
            line=dict(color="#818cf8", width=1.5, dash="solid"),
        ))

    # ── EMA_50 overlay ────────────────────────────────────────────────────────
    if "EMA_50" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=timestamps, y=plot_df["EMA_50"],
            mode="lines", name="EMA 50",
            line=dict(color="#f472b6", width=1.5, dash="dot"),
        ))

    # ── RSI subplot via secondary y-axis ─────────────────────────────────────
    if "RSI_14" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=timestamps, y=plot_df["RSI_14"],
            mode="lines", name="RSI 14",
            line=dict(color="#fbbf24", width=1.2),
            yaxis="y2",
        ))
        # Overbought / oversold reference lines
        for level, color in [(70, "rgba(248,113,113,0.3)"), (30, "rgba(52,211,153,0.3)")]:
            fig.add_hline(y=level, line_color=color, line_dash="dash",
                          line_width=1, yref="y2")

    # ── Dark layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> · Last 60 Candles · EMA 20 / 50 overlaid",
            font=dict(color="#94a3b8", size=13),
            x=0.01,
        ),
        paper_bgcolor="#070d1a",
        plot_bgcolor="#0a1020",
        font=dict(color="#64748b", family="Inter"),
        xaxis=dict(
            gridcolor="#1a2740", showgrid=True,
            rangeslider=dict(visible=False),
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="#1a2740", showgrid=True,
            title=dict(text="Price (USDT)", font=dict(size=11)),
            tickfont=dict(size=10),
            side="left",
        ),
        yaxis2=dict(
            title=dict(text="RSI", font=dict(size=11)),
            overlaying="y", side="right",
            range=[0, 100],
            showgrid=False,
            tickfont=dict(size=10),
        ),
        legend=dict(
            bgcolor="rgba(10,16,32,0.8)",
            bordercolor="#1a2740",
            borderwidth=1,
            font=dict(size=11),
            x=0.01, y=0.99,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
    )

    return fig


# ── Core tick logic ────────────────────────────────────────────────────────────
def _run_tick(symbol: str, timeframe: str, candle_limit: int) -> None:
    """Runs one full pipeline tick: fetch → enrich → signal → execute."""
    MarketDataFetcher, FeatureEngineer, DimmerForceAgent, ZenithAgent, AegisAgent = _load_pipeline()
    t0 = time.time()

    # 1. Fetch & enrich
    try:
        raw_df = MarketDataFetcher().fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=candle_limit)
        enriched_df = FeatureEngineer().add_technical_indicators(raw_df)
        st.session_state.enriched_df = enriched_df
        current_price = float(enriched_df["close"].iloc[-1])
        st.session_state.last_price = current_price
        _log(f"Fetched {len(enriched_df)} candles. Close=${current_price:,.2f}", "info")
    except Exception as exc:
        _log(f"Data fetch failed: {exc}", "err")
        st.session_state.last_tick_s = time.time() - t0
        return

    # Capture initial price on the very first tick for the Buy&Hold baseline
    if st.session_state.initial_btc_price is None:
        st.session_state.initial_btc_price = current_price
        _log(f"Buy&Hold baseline set: ${current_price:,.2f}", "info")

    # 2. Agent signals + signal counter update
    agents_map = {"DimmerForce": DimmerForceAgent(), "Zenith": ZenithAgent(), "Aegis": AegisAgent()}
    enriched_df = st.session_state.enriched_df
    for name, agent in agents_map.items():
        try:
            sig = agent.generate_signal(enriched_df)
            st.session_state.signals[name] = sig
            st.session_state.errors[name] = None
            action = sig["action"]
            # Increment the session-lifetime signal counter for this agent
            if action in st.session_state.signal_counts[name]:
                st.session_state.signal_counts[name][action] += 1
            _log(f"[{name}] → {action} ({int(sig['confidence']*100)}%): {sig['reason'][:60]}",
                 "buy" if action == "BUY" else "sell" if action == "SELL" else "info")
        except Exception as exc:
            st.session_state.signals[name] = None
            st.session_state.errors[name] = str(exc)
            _log(f"[{name}] agent error: {exc}", "err")

    # 3. Execute order via DimmerForce signal
    exec_signal = st.session_state.signals.get(EXEC_AGENT)
    if exec_signal:
        _execute_testnet_order(exec_signal, symbol)

    # 4. Append performance snapshot for the equity chart
    init_price = st.session_state.initial_btc_price or current_price
    # Buy&Hold equity: normalise current price to starting allocation of $50
    buy_hold_equity = (current_price / init_price) * 50.0

    # Simulated advisory equity: +0.1% on correct-direction signal, -0.05% on wrong
    # This is a lightweight paper-equity proxy for Zenith and Aegis (advisory only).
    # DimmerForce equity is derived from the live testnet balance when available.
    for name in ["Zenith", "Aegis"]:
        sig = st.session_state.signals.get(name)
        if sig:
            act = sig.get("action", "HOLD")
            price_change_pct = (current_price - init_price) / init_price
            if act == "BUY" and price_change_pct > 0:
                st.session_state.agent_equity[name] *= 1.001
            elif act == "SELL" and price_change_pct < 0:
                st.session_state.agent_equity[name] *= 1.001
            elif act in ("BUY", "SELL"):
                st.session_state.agent_equity[name] *= 0.9995

    st.session_state.performance_history.append({
        "timestamp": datetime.now(timezone.utc).strftime("%H:%M"),
        "btc_price": current_price,
        "buy_hold": round(buy_hold_equity, 4),
        "DimmerForce": round(st.session_state.agent_equity["DimmerForce"], 4),
        "Zenith":      round(st.session_state.agent_equity["Zenith"], 4),
        "Aegis":       round(st.session_state.agent_equity["Aegis"], 4),
    })

    st.session_state.tick_count += 1
    st.session_state.last_tick_s = time.time() - t0
    _log(f"Tick #{st.session_state.tick_count} complete in {st.session_state.last_tick_s:.1f}s", "info")


def _execute_testnet_order(signal: dict, symbol: str) -> None:
    """Translates a DimmerForce signal into a testnet order."""
    from src.execution.broker import BrokerError
    from src.execution.risk_manager import RiskManager

    action = signal.get("action", "HOLD")
    if action == "HOLD":
        _log(f"[{EXEC_AGENT}] HOLD — no order placed.", "info")
        return

    try:
        broker = _get_broker()
        current_price = st.session_state.last_price or 0.0

        if action == "BUY":
            usdt_balance = broker.get_free_balance("USDT")
            enriched_df = st.session_state.enriched_df
            atr = float(enriched_df["ATRr_14"].iloc[-1]) if enriched_df is not None else 0.0

            if usdt_balance < 10.0:
                _log(f"[{EXEC_AGENT}] BUY skipped — USDT balance too low ({usdt_balance:.2f})", "err")
                return

            rm = RiskManager()
            metrics = rm.calculate_position_size(
                signal_action="BUY",
                current_price=current_price,
                atr=atr if atr > 0 else current_price * 0.003,
                available_capital=usdt_balance,
            )
            units = metrics.get("units", 0.0)
            if units <= 0:
                _log(f"[{EXEC_AGENT}] BUY skipped — RiskManager returned 0 units.", "err")
                return

            _log(f"[{EXEC_AGENT}] Placing MARKET BUY {units:.8f} {symbol} @ ~${current_price:,.2f}", "buy")
            order = broker.execute_order(symbol=symbol, side="buy", amount=units)
            _log(f"[{EXEC_AGENT}] Order filled — id={order.get('id')} status={order.get('status')}", "buy")

        elif action == "SELL":
            base_ticker = symbol.split("/")[0]  # e.g. "BTC"
            btc_balance = broker.get_free_balance(base_ticker)
            if btc_balance <= 0.0:
                _log(f"[{EXEC_AGENT}] SELL skipped — no {base_ticker} balance to sell.", "err")
                return

            _log(f"[{EXEC_AGENT}] Placing MARKET SELL {btc_balance:.8f} {symbol}", "sell")
            order = broker.execute_order(symbol=symbol, side="sell", amount=btc_balance)
            _log(f"[{EXEC_AGENT}] Order filled — id={order.get('id')} status={order.get('status')}", "sell")

    except BrokerError as exc:
        _log(f"[{EXEC_AGENT}] Broker error: {exc}", "err")
    except Exception as exc:
        _log(f"[{EXEC_AGENT}] Unexpected execution error: {exc}", "err")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NexusQuant")
    st.markdown('<div style="color:#374151;font-size:12px;margin-bottom:16px;">Alpha Arena · v0.7 · Session Trading</div>',
                unsafe_allow_html=True)

    # Strict mode indicator — no toggle
    st.markdown('<div class="mode-badge">🌐 BINANCE TESTNET</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#374151;margin:8px 0 16px;">Orders route to testnet.binance.vision<br>Keys loaded from local .env</div>',
                unsafe_allow_html=True)

    # Balance check
    if st.button("🔍 Check Testnet Balance", key="check_bal"):
        with st.spinner("Connecting…"):
            try:
                b = _get_broker()
                u = b.get_free_balance("USDT")
                c = b.get_free_balance("BTC")
                st.session_state.testnet_balance = {"USDT": u, "BTC": c}
                st.success(f"USDT: {u:,.4f}\nBTC: {c:.8f}")
                _log(f"Balance check — USDT={u:,.4f}  BTC={c:.8f}", "info")
            except Exception as exc:
                st.error(f"Connection failed: {exc}")

    st.markdown("---")
    st.markdown("### ⚙️ Market Settings")
    symbol       = st.selectbox(
        "Trading Pair",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "XRP/USDT"],
    )
    timeframe    = st.selectbox("Timeframe", ["1h","15m","4h"])
    candle_limit = st.slider("Candles", 60, 300, 100, step=10)

    st.markdown("### ⏱️ Session Settings")
    session_hours   = st.number_input("Duration (Hours)",  min_value=0.1, max_value=24.0, value=4.0, step=0.5)
    tick_interval_s = st.number_input("Tick Interval (s)", min_value=30,  max_value=3600,  value=300, step=30)

    st.markdown("---")
    if st.session_state.testnet_balance:
        b = st.session_state.testnet_balance
        st.markdown(f'<div style="font-size:12px;color:#374151;">💰 USDT: <b style="color:#60a5fa;">{b["USDT"]:,.2f}</b><br>'
                    f'₿ BTC: <b style="color:#fbbf24;">{b["BTC"]:.8f}</b></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;color:#1f2937;margin-top:12px;line-height:1.7;">'
                f'⚡ Exec Agent: <b>{EXEC_AGENT}</b><br>'
                '📊 Advisory: Zenith, Aegis<br>'
                '🔒 Keys from .env — never logged</div>', unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="font-size:32px;font-weight:800;color:#e2e8f0;margin-bottom:4px;">'
            '⚡ NexusQuant <span style="color:#6366f1;">Alpha Arena</span> '
            '<span class="mode-badge" style="font-size:14px;">🌐 TESTNET</span></h1>'
            '<p style="color:#374151;font-size:13px;margin-bottom:20px;">'
            'Session-based live trading · Gemma 4 × 3 personas · '
            f'Execution: <b style="color:#818cf8;">{EXEC_AGENT}</b> · '
            'Advisory: <b style="color:#34d399;">Zenith</b> + <b style="color:#f472b6;">Aegis</b></p>',
            unsafe_allow_html=True)

# ── Global metrics row ─────────────────────────────────────────────────────────
elapsed = _elapsed_s()
remaining = max(0, session_hours * 3600 - elapsed)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Session Status", "🟢 ACTIVE" if st.session_state.session_active else "⭕ IDLE")
m2.metric("Ticks Completed", st.session_state.tick_count)
m3.metric("Last BTC Price", f"${st.session_state.last_price:,.2f}" if st.session_state.last_price else "—")
m4.metric("Time Elapsed", f"{int(elapsed//60)}m {int(elapsed%60)}s" if st.session_state.session_active else "—")
m5.metric("Time Remaining", f"{int(remaining//60)}m {int(remaining%60)}s" if st.session_state.session_active else "—")

st.markdown("---")

# ── Live candlestick chart ─────────────────────────────────────────────────────
chart_placeholder = st.empty()
_fig = _build_candlestick_chart(symbol)
if _fig is not None:
    chart_placeholder.plotly_chart(_fig, use_container_width=True, config={"displayModeBar": False})
else:
    chart_placeholder.markdown(
        '<div style="background:#070d1a;border:1px solid #1a2740;border-radius:12px;'
        'height:120px;display:flex;align-items:center;justify-content:center;'
        'color:#1e3a5f;font-size:13px;">'
        '📊 Candlestick chart loads after the first session tick…</div>',
        unsafe_allow_html=True,
    )

# ── Performance equity chart ──────────────────────────────────────────────────
perf_placeholder = st.empty()
_perf_fig = _build_performance_chart()
if _perf_fig is not None:
    perf_placeholder.plotly_chart(_perf_fig, use_container_width=True,
                                  config={"displayModeBar": False})
else:
    perf_placeholder.markdown(
        '<div style="background:#070d1a;border:1px solid #1a2740;border-radius:12px;'
        'height:80px;display:flex;align-items:center;justify-content:center;'
        'color:#1e3a5f;font-size:13px;">'
        '📈 Equity chart appears after 2 ticks…</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Session control ────────────────────────────────────────────────────────────
ctrl_left, ctrl_right = st.columns([2, 3])

with ctrl_left:
    if not st.session_state.session_active:
        # ── START form ───────────────────────────────────────────────────────
        st.markdown('<div style="background:#0e1628;border:1px solid #1e3a5f;border-radius:14px;padding:20px;">'
                    '<div style="font-size:13px;font-weight:600;color:#60a5fa;margin-bottom:14px;'
                    'text-transform:uppercase;letter-spacing:1px;">🚀 Start Trading Session</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:#374151;margin-bottom:4px;">'
                    f'Duration: <b style="color:#e2e8f0;">{session_hours}h</b> · '
                    f'Tick every: <b style="color:#e2e8f0;">{tick_interval_s}s</b><br>'
                    f'Symbol: <b style="color:#e2e8f0;">{symbol}</b> · '
                    f'TF: <b style="color:#e2e8f0;">{timeframe}</b></div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶  Start Trading Session", key="start_session"):
            try:
                _get_broker()   # validate keys before starting
                st.session_state.session_active      = True
                st.session_state.session_start_ts    = time.time()
                st.session_state.session_duration_h  = session_hours
                st.session_state.tick_interval_s     = tick_interval_s
                st.session_state.tick_count          = 0
                st.session_state.order_log           = []
                # Reset tracking state for a fresh session
                st.session_state.signal_counts       = {k: {"BUY": 0, "SELL": 0, "HOLD": 0} for k in AGENT_META}
                st.session_state.performance_history = []
                st.session_state.initial_btc_price   = None
                st.session_state.agent_equity        = {k: 50.0 for k in AGENT_META}
                _log(f"Session started — duration={session_hours}h  interval={tick_interval_s}s  "
                     f"symbol={symbol}  agent={EXEC_AGENT}", "info")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Cannot start session: {exc}\n\nCheck .env has valid testnet keys.")
    else:
        # ── ACTIVE session status ────────────────────────────────────────────
        st.markdown(f'<div class="session-active">'
                    f'🟢 SESSION ACTIVE<br>'
                    f'<span style="font-size:12px;font-weight:400;">'
                    f'Tick #{st.session_state.tick_count} · '
                    f'{int(remaining//60)}m {int(remaining%60)}s remaining</span></div>',
                    unsafe_allow_html=True)

        # Progress bar
        progress = min(1.0, elapsed / (session_hours * 3600)) if session_hours > 0 else 0
        st.progress(progress)

        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        if st.button("⏹  STOP SESSION", key="stop_session"):
            st.session_state.session_active = False
            st.session_state.broker = None   # reset broker so next session gets fresh instance
            _log("Session stopped by user.", "err")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

with ctrl_right:
    # ── Live order log ───────────────────────────────────────────────────────
    st.markdown('<div style="font-size:10px;font-weight:600;color:#374151;'
                'text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">'
                '📋 Live Execution Log</div>', unsafe_allow_html=True)
    log_html = "".join(st.session_state.order_log[:40]) if st.session_state.order_log else (
        '<div class="log-entry log-info">No activity yet. Start a session to begin.</div>'
    )
    st.markdown(f'<div style="background:#070d1a;border:1px solid #1a2740;border-radius:10px;'
                f'padding:14px;height:220px;overflow-y:auto;">{log_html}</div>',
                unsafe_allow_html=True)

st.markdown("---")

# ── Agent signal columns ───────────────────────────────────────────────────────
agent_cols = st.columns(3)
agent_names = list(AGENT_META.keys())

for i, agent_name in enumerate(agent_names):
    meta   = AGENT_META[agent_name]
    signal = st.session_state.signals.get(agent_name)
    error  = st.session_state.errors.get(agent_name)
    is_exec = meta["executes"]

    with agent_cols[i]:
        exec_label = ' <span style="font-size:10px;background:#1e3a5f;color:#60a5fa;padding:2px 8px;border-radius:10px;">EXECUTES</span>' if is_exec else ' <span style="font-size:10px;background:#1c1208;color:#fbbf24;padding:2px 8px;border-radius:10px;">ADVISORY</span>'
        st.markdown(
            f'<div class="agent-card {meta["card"]}">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            f'<span style="font-size:26px;">{meta["icon"]}</span>'
            f'<div><div style="font-size:17px;font-weight:700;color:#e2e8f0;">{agent_name}{exec_label}</div>'
            f'<div style="font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:1px;">{meta["persona"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if error:
            st.markdown(f'<div class="err-box">⚠️ {error}</div>', unsafe_allow_html=True)
        elif signal is None:
            st.markdown('<div style="color:#374151;font-size:12px;padding:8px 0;">Awaiting first tick…</div>',
                        unsafe_allow_html=True)
        else:
            action = signal.get("action","HOLD")
            conf   = float(signal.get("confidence", 0.0))
            reason = signal.get("reason","—")
            st.markdown(_badge(action), unsafe_allow_html=True)
            st.markdown(_conf_bar(conf), unsafe_allow_html=True)
            st.markdown(f'<div class="reason-box">💬 {reason}</div>', unsafe_allow_html=True)

        # ── Signal counters ───────────────────────────────────────────────────
        counts = st.session_state.signal_counts.get(agent_name, {})
        b_cnt = counts.get("BUY", 0)
        s_cnt = counts.get("SELL", 0)
        h_cnt = counts.get("HOLD", 0)
        equity = st.session_state.agent_equity.get(agent_name, 50.0)
        pnl    = equity - 50.0
        pnl_color = "#34d399" if pnl >= 0 else "#f87171"
        pnl_sign  = "+" if pnl >= 0 else ""
        st.markdown(
            f'<div style="margin-top:12px;padding:8px 10px;background:#070d1a;'
            f'border:1px solid #1a2740;border-radius:8px;">'
            f'<div style="font-size:10px;color:#374151;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:1px;margin-bottom:5px;">Session Counts</div>'
            f'<span style="color:#34d399;font-size:12px;font-weight:600;">▲ BUY {b_cnt}</span>'
            f'<span style="color:#64748b;margin:0 6px;">│</span>'
            f'<span style="color:#f87171;font-size:12px;font-weight:600;">▼ SELL {s_cnt}</span>'
            f'<span style="color:#64748b;margin:0 6px;">│</span>'
            f'<span style="color:#fbbf24;font-size:12px;font-weight:600;">— HOLD {h_cnt}</span>'
            f'<div style="margin-top:5px;font-size:12px;">'  
            f'Equity: <b style="color:#60a5fa;font-family:monospace;">${equity:.2f}</b>'
            f' <span style="color:{pnl_color};font-size:11px;">({pnl_sign}{pnl:.2f})</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

# ── Comparison matrix ──────────────────────────────────────────────────────────
if any(s is not None for s in st.session_state.signals.values()):
    st.markdown("---")
    st.markdown('<div style="font-size:10px;font-weight:600;color:#374151;text-transform:uppercase;'
                'letter-spacing:2px;margin-bottom:10px;">Consensus Matrix</div>', unsafe_allow_html=True)
    cm_cols = st.columns(3)
    for i, (name, meta) in enumerate(AGENT_META.items()):
        sig = st.session_state.signals.get(name)
        with cm_cols[i]:
            if sig:
                action = sig.get("action","—")
                conf   = sig.get("confidence", 0.0)
                color  = {"BUY":"#34d399","SELL":"#f87171"}.get(action,"#fbbf24")
                st.markdown(
                    f'<div style="text-align:center;padding:12px;background:#0a101e;'
                    f'border-radius:10px;border:1px solid #1a2740;">'
                    f'<div style="font-size:11px;color:#374151;">{name}</div>'
                    f'<div style="font-size:24px;font-weight:800;color:{color};font-family:monospace;">{action}</div>'
                    f'<div style="font-size:12px;color:#4b5563;">{int(conf*100)}%</div></div>',
                    unsafe_allow_html=True)

# ── Session loop: sleep then rerun ────────────────────────────────────────────
if st.session_state.session_active:
    if _session_expired():
        st.session_state.session_active = False
        st.session_state.broker = None
        _log(f"Session ended — {st.session_state.tick_count} ticks completed.", "info")
        st.warning(f"✅ Session complete — {st.session_state.tick_count} ticks executed.")
        st.rerun()
    else:
        _run_tick(symbol, timeframe, candle_limit)
        with st.spinner(f"Next tick in {st.session_state.tick_interval_s}s…"):
            time.sleep(st.session_state.tick_interval_s)
        st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div style="text-align:center;font-size:11px;color:#111827;padding:8px 0;">'
            'NexusQuant Alpha Arena · Binance Testnet · Not Financial Advice</div>',
            unsafe_allow_html=True)

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
    "DimmerForce": {"icon":"📈","color":"#818cf8","persona":"Trend Follower","card":"card-dimmer", "active": True, "allocation_pct": 1.0},
    "Zenith":      {"icon":"🔄","color":"#34d399","persona":"Mean Reversion","card":"card-zenith", "active": False, "allocation_pct": 0.0},
    "Aegis":       {"icon":"🛡️","color":"#f472b6","persona":"Conservative",  "card":"card-aegis", "active": False, "allocation_pct": 0.0},
}


# ── Session state initialisation ───────────────────────────────────────────────
def _init():
    defaults = {
        "session_active": False,
        "session_start_ts": None,
        "session_duration_h": 6.0,
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
        "agent_equity": {k: 200.0 for k in AGENT_META},
        # per-agent simulated position sizes (starts at 0.0 units).
        "agent_positions": {k: 0.0 for k in AGENT_META},
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


def _build_performance_chart() -> go.Figure | None:
    """Builds the agent PnL equity curve chart with a Buy&Hold market baseline.

    Returns:
        A :class:`plotly.graph_objects.Figure` with 4 lines (3 agents + baseline),
        or ``None`` when fewer than 2 ticks have been recorded.
    """
    history = st.session_state.performance_history
    if len(history) < 2:
        return None

    timestamps   = [h["timestamp"]  for h in history]
    buy_hold     = [h["buy_hold"]   for h in history]
    df_equity    = [h["DimmerForce"] for h in history]
    zen_equity   = [h["Zenith"]     for h in history]
    aegis_equity = [h["Aegis"]      for h in history]

    fig = go.Figure()

    # ── Agent equity lines + Buy&Hold baseline ────────────────────────────────
    traces = [
        ("DimmerForce", df_equity,    "#818cf8", "solid", 2.0),
        ("Zenith",      zen_equity,   "#34d399", "solid", 1.5),
        ("Aegis",       aegis_equity, "#f472b6", "solid", 1.5),
        ("Buy & Hold",  buy_hold,     "#94a3b8", "dash",  1.5),
    ]
    for label, values, color, dash, width in traces:
        fig.add_trace(go.Scatter(
            x=timestamps, y=values,
            mode="lines",
            name=label,
            line=dict(color=color, width=width, dash=dash),
        ))

    starting_allocation = df_equity[0] if df_equity else 200.0
    # Dotted horizontal baseline at starting allocation
    fig.add_hline(
        y=starting_allocation,
        line_color="rgba(148,163,184,0.2)",
        line_dash="dot",
        line_width=1,
    )

    # ── Dark layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>Agent Equity Curves</b> · Simulated PnL vs Buy & Hold Baseline",
            font=dict(color="#94a3b8", size=13),
            x=0.01,
        ),
        paper_bgcolor="#070d1a",
        plot_bgcolor="#0a1020",
        font=dict(color="#64748b", family="Inter"),
        xaxis=dict(
            gridcolor="#1a2740", showgrid=True,
            tickfont=dict(size=10),
            title=dict(text="Tick Time (UTC)", font=dict(size=11)),
        ),
        yaxis=dict(
            gridcolor="#1a2740", showgrid=True,
            title=dict(text="Equity (USDT)", font=dict(size=11)),
            tickfont=dict(size=10),
            tickprefix="$",
        ),
        legend=dict(
            bgcolor="rgba(10,16,32,0.8)",
            bordercolor="#1a2740",
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            x=0.0, y=-0.25,
        ),
        margin=dict(l=10, r=10, t=40, b=50),
        height=320,
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
        if not AGENT_META[name].get("active", True):
            continue
        try:
            # Pass the agent's independent virtual position size
            pos_size = st.session_state.agent_positions.get(name, 0.0)
            sig = agent.generate_signal(enriched_df, current_position_size=pos_size)
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

    # 3. Execute orders autonomously for each active agent
    for name in AGENT_META.keys():
        if not AGENT_META[name].get("active", True):
            continue
        sig = st.session_state.signals.get(name)
        if sig:
            _execute_order(name, sig, symbol)

    # 4. Append performance snapshot for the equity chart
    init_price = st.session_state.initial_btc_price or current_price
    # Fetch actual starting allocation for Buy&Hold baseline
    starting_alloc = 200.0
    if st.session_state.performance_history:
        starting_alloc = st.session_state.performance_history[0].get("DimmerForce", 200.0)
    else:
        starting_alloc = st.session_state.agent_equity.get("DimmerForce", 200.0)

    buy_hold_equity = (current_price / init_price) * starting_alloc

    # Calculate total equity for each agent: cash + position_value
    agent_total_equity = {}
    for name in AGENT_META.keys():
        cash = st.session_state.agent_equity.get(name, 200.0)
        pos = st.session_state.agent_positions.get(name, 0.0)
        total_eq = cash + (pos * current_price)
        agent_total_equity[name] = total_eq

    # ── Emergency Drawdown stop-loss check ──────────────────────────────
    # If the total allocated capital ($600 USDT) experiences a maximum drawdown of 5%
    # (i.e. falls to or below $570 USDT), execute emergency stop-loss.
    df_equity = agent_total_equity.get("DimmerForce", 0.0)
    if df_equity <= 570.0:
        _log(f"[EMERGENCY WARNING] DimmerForce total equity fell to ${df_equity:.2f} (<= $570.00). Initiating immediate liquidation!", "err")
        # 1. Liquidate open positions on SOL
        pos_size = st.session_state.agent_positions.get("DimmerForce", 0.0)
        if pos_size > 0.0:
            try:
                _log(f"[EMERGENCY LIQUIDATION] Placing MARKET SELL for {pos_size:.4f} SOL...", "sell")
                broker = _get_broker()
                order = broker.execute_order(symbol="SOL/USDT", side="sell", amount=pos_size)
                _log(f"[EMERGENCY LIQUIDATION] Closed SOL position. Order ID: {order.get('id')}", "sell")
            except Exception as exc:
                _log(f"[EMERGENCY LIQUIDATION ERROR] Failed to liquidate SOL position: {exc}", "err")
            
            # Reset states
            st.session_state.agent_positions["DimmerForce"] = 0.0
            st.session_state.agent_equity["DimmerForce"] += pos_size * current_price
        
        # 2. Halt all trading loops and safely log out
        st.session_state.session_active = False
        st.session_state.broker = None
        _log("[EMERGENCY STOP-LOSS TRIGGERED] SOL position liquidated. Trading halted.", "err")
        st.warning("⚠️ EMERGENCY GLOBAL STOP-LOSS TRIGGERED: DimmerForce equity fell to or below $570.00 (5% drawdown). SOL position liquidated, trading session halted.")
        st.rerun()

    st.session_state.performance_history.append({
        "timestamp": datetime.now(timezone.utc).strftime("%H:%M"),
        "btc_price": current_price,
        "buy_hold": round(buy_hold_equity, 4),
        "DimmerForce": round(agent_total_equity["DimmerForce"], 4),
        "Zenith":      round(agent_total_equity["Zenith"], 4),
        "Aegis":       round(agent_total_equity["Aegis"], 4),
    })

    st.session_state.tick_count += 1
    st.session_state.last_tick_s = time.time() - t0
    _log(f"Tick #{st.session_state.tick_count} complete in {st.session_state.last_tick_s:.1f}s", "info")


def _execute_order(agent_name: str, signal: dict, symbol: str) -> None:
    """Translates an agent's signal into a broker order.

    Trade size is calculated against the agent's simulated equity —
    NOT the raw broker USDT balance. This enforces strict capital isolation
    so trades are sized precisely against the agent's virtual equity pool.
    """
    from src.execution.broker import BrokerError
    from src.execution.risk_manager import RiskManager

    action = signal.get("action", "HOLD")
    if action == "HOLD":
        _log(f"[{agent_name}] HOLD — no order placed.", "info")
        return

    agent_capital = st.session_state.agent_equity.get(agent_name, 200.0)
    current_price = st.session_state.last_price or 0.0

    try:
        broker = _get_broker()

        if action == "BUY":
            # Guard: simulated capital must be meaningful before placing an order.
            if agent_capital < 5.0:
                _log(
                    f"[{agent_name}] BUY skipped — simulated capital too low "
                    f"(${agent_capital:.2f})",
                    "err",
                )
                return

            enriched_df = st.session_state.enriched_df
            atr = float(enriched_df["ATRr_14"].iloc[-1]) if enriched_df is not None else 0.0

            rm = RiskManager()
            metrics = rm.calculate_position_size(
                signal_action="BUY",
                current_price=current_price,
                atr=atr if atr > 0 else current_price * 0.003,
                # STRICT CAPITAL ISOLATION
                # Use the agent's individual simulated equity (cash balance).
                available_capital=agent_capital,
            )
            units = metrics.get("units", 0.0)
            cost  = metrics.get("cost_usdt", 0.0)

            if units <= 0:
                _log(f"[{agent_name}] BUY skipped — RiskManager returned 0 units.", "err")
                return

            _log(
                f"[{agent_name}] Placing MARKET BUY {units:.8f} {symbol} "
                f"@ ~${current_price:,.2f}  (capital=${agent_capital:.2f}, cost=${cost:.2f})",
                "buy",
            )
            order = broker.execute_order(symbol=symbol, side="buy", amount=units)
            _log(f"[{agent_name}] Order filled — id={order.get('id')} status={order.get('status')}", "buy")

            # Deduct actual cost from simulated cash equity.
            st.session_state.agent_equity[agent_name] = max(0.0, agent_capital - cost)
            st.session_state.agent_positions[agent_name] = st.session_state.agent_positions.get(agent_name, 0.0) + units
            _log(
                f"[{agent_name}] Simulated cash equity updated: "
                f"${agent_capital:.2f} → ${st.session_state.agent_equity[agent_name]:.2f}",
                "info",
            )

        elif action == "SELL":
            position_size = st.session_state.agent_positions.get(agent_name, 0.0)
            if position_size <= 0.0:
                _log(f"[{agent_name}] SELL skipped — no virtual position to sell.", "err")
                return

            base_ticker = symbol.split("/")[0]  # e.g. "BTC"
            broker_balance = broker.get_free_balance(base_ticker)
            
            # Cap the physical order amount to what's actually available on the testnet account
            # to prevent insufficient balance errors due to slippage/fees.
            amount_to_sell = min(position_size, broker_balance)
            if amount_to_sell <= 0.0:
                _log(f"[{agent_name}] SELL skipped — physical {base_ticker} balance is 0.", "err")
                return

            sell_value = position_size * current_price
            _log(
                f"[{agent_name}] Placing MARKET SELL {amount_to_sell:.8f} {symbol} "
                f"@ ~${current_price:,.2f} (est. return=${sell_value:.2f})",
                "sell",
            )
            order = broker.execute_order(symbol=symbol, side="sell", amount=amount_to_sell)
            _log(f"[{agent_name}] Order filled — id={order.get('id')} status={order.get('status')}", "sell")

            # Credit estimated USDT proceeds back to simulated cash equity.
            st.session_state.agent_equity[agent_name] = agent_capital + sell_value
            st.session_state.agent_positions[agent_name] = 0.0
            _log(
                f"[{agent_name}] Simulated cash equity updated: "
                f"${agent_capital:.2f} → ${st.session_state.agent_equity[agent_name]:.2f}",
                "info",
            )

    except BrokerError as exc:
        _log(f"[{agent_name}] Broker error: {exc}", "err")
    except Exception as exc:
        _log(f"[{agent_name}] Unexpected execution error: {exc}", "err")


# Try to fetch connection mode dynamically from the broker
try:
    broker_instance = _get_broker()
    is_sandbox = broker_instance.sandbox_mode
except Exception:
    is_sandbox = True

# Try to fetch live balance at startup if not already loaded and session not active
if not st.session_state.session_active and not st.session_state.get("live_balance_loaded", False):
    try:
        real_usdt = broker_instance.get_free_balance("USDT")
        for k, meta in AGENT_META.items():
            st.session_state.agent_equity[k] = real_usdt * meta.get("allocation_pct", 1.0/3.0)
        st.session_state.live_balance_loaded = True
        _log(f"Startup: Loaded live wallet balance {real_usdt:,.2f} USDT.", "info")
    except Exception as exc:
        pass

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NexusQuant")
    st.markdown('<div style="color:#374151;font-size:12px;margin-bottom:16px;">Alpha Arena · v0.7 · Session Trading</div>',
                unsafe_allow_html=True)

    # Dynamic network mode indicator
    if is_sandbox:
        st.markdown('<div class="mode-badge">🌐 BINANCE TESTNET</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#374151;margin:8px 0 16px;">Orders route to testnet.binance.vision<br>Keys loaded from local .env</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-badge" style="background:#450a0a;color:#f87171;border:1px solid #dc2626;">🔴 LIVE MAINNET</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#f87171;margin:8px 0 16px;"><b>REAL MONEY LIVE TRADING ACTIVE</b><br>Orders route to api.binance.com</div>',
                    unsafe_allow_html=True)

    # Balance check
    btn_label = "🔍 Check Testnet Balance" if is_sandbox else "🔍 Check Live Balance"
    if st.button(btn_label, key="check_bal"):
        with st.spinner("Connecting…"):
            try:
                b = _get_broker()
                u = b.get_free_balance("USDT")
                c = b.get_free_balance("BTC")
                st.session_state.testnet_balance = {"USDT": u, "BTC": c}
                st.success(f"USDT: {u:,.4f}\nBTC: {c:.8f}")
                mode_label = "Testnet" if is_sandbox else "Mainnet"
                _log(f"Balance check ({mode_label}) — USDT={u:,.4f}  BTC={c:.8f}", "info")
            except Exception as exc:
                st.error(f"Connection failed: {exc}")

    st.markdown("---")
    st.markdown("### ⚙️ Market Settings")
    symbol       = st.selectbox(
        "Trading Pair",
        ["SOL/USDT"],
    )
    timeframe    = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h"])
    candle_limit = st.slider("Candles", 60, 300, 100, step=10)

    st.markdown("### ⏱️ Session Settings")
    session_hours   = st.number_input("Duration (Hours)",  min_value=0.1, max_value=24.0, value=6.0, step=0.5)
    tick_interval_s = st.number_input("Tick Interval (s)", min_value=30,  max_value=3600,  value=300, step=30)

    st.markdown("---")
    if st.session_state.testnet_balance:
        b = st.session_state.testnet_balance
        st.markdown(f'<div style="font-size:12px;color:#374151;">💰 USDT: <b style="color:#60a5fa;">{b["USDT"]:,.2f}</b><br>'
                    f'₿ BTC: <b style="color:#fbbf24;">{b["BTC"]:.8f}</b></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;color:#1f2937;margin-top:12px;line-height:1.7;">'
                '⚡ Execution: <b>Autonomous (All Agents)</b><br>'
                '🔒 Keys from .env — never logged</div>', unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
badge_html = '<span class="mode-badge" style="font-size:14px;">🌐 TESTNET</span>' if is_sandbox else '<span class="mode-badge" style="font-size:14px;background:#450a0a;color:#f87171;border:1px solid #dc2626;">🔴 LIVE MAINNET</span>'
st.markdown(f'<h1 style="font-size:32px;font-weight:800;color:#e2e8f0;margin-bottom:4px;">'
            f'⚡ NexusQuant <span style="color:#6366f1;">Alpha Arena</span> '
            f'{badge_html}</h1>'
            f'<p style="color:#374151;font-size:13px;margin-bottom:20px;">'
            f'Session-based live trading · Gemma 4 × 3 personas · '
            f'Execution: <b style="color:#6366f1;">Autonomous</b> (DimmerForce, Zenith, Aegis)</p>',
            unsafe_allow_html=True)

# ── Global metrics row ─────────────────────────────────────────────────────────
elapsed = _elapsed_s()
remaining = max(0, session_hours * 3600 - elapsed)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Session Status", "🟢 ACTIVE" if st.session_state.session_active else "⭕ IDLE")
m2.metric("Ticks Completed", st.session_state.tick_count)
m3.metric(f"Last {symbol.split('/')[0]} Price", f"${st.session_state.last_price:,.2f}" if st.session_state.last_price else "—")
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
                broker = _get_broker()   # validate credentials and fetch broker
                real_usdt = broker.get_free_balance("USDT")

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
                st.session_state.agent_equity        = {k: real_usdt * meta.get("allocation_pct", 1.0/3.0) for k, meta in AGENT_META.items()}
                st.session_state.agent_positions     = {k: 0.0 for k in AGENT_META}
                _log(f"Session started — duration={session_hours}h  interval={tick_interval_s}s  "
                     f"symbol={symbol}  agents=Autonomous Loop", "info")
                
                # Log detailed allocation information per agent
                for k, meta in AGENT_META.items():
                    alloc = st.session_state.agent_equity[k]
                    _log(f"[{k}] Allocation set to {meta.get('allocation_pct', 0.0)*100:.0f}% (${alloc:,.2f} USDT).", "info")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Cannot start session: {exc}\n\nCheck .env has valid keys.")
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
    with agent_cols[i]:
        is_active = meta.get("active", True)
        if is_active:
            exec_label = ' <span style="font-size:10px;background:#064e3b;color:#34d399;padding:2px 8px;border-radius:10px;">ACTIVE TRADER</span>'
        else:
            exec_label = ' <span style="font-size:10px;background:#374151;color:#94a3b8;padding:2px 8px;border-radius:10px;">DEACTIVATED</span>'
        
        st.markdown(
            f'<div class="agent-card {meta["card"]}">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            f'<span style="font-size:26px;">{meta["icon"]}</span>'
            f'<div><div style="font-size:17px;font-weight:700;color:#e2e8f0;">{agent_name}{exec_label}</div>'
            f'<div style="font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:1px;">{meta["persona"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if not is_active:
            st.markdown('<div style="color:#64748b;font-size:12px;padding:8px 0;font-style:italic;">Agent is deactivated.</div>',
                        unsafe_allow_html=True)
        else:
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
        
        # Total equity = cash + (position_size * current_price)
        cash = st.session_state.agent_equity.get(agent_name, 200.0)
        pos = st.session_state.agent_positions.get(agent_name, 0.0)
        current_price = st.session_state.last_price or 0.0
        equity = cash + (pos * current_price)
        
        # Calculate PnL against actual starting allocation of this agent
        starting_alloc = 200.0
        if st.session_state.performance_history:
            starting_alloc = st.session_state.performance_history[0].get(agent_name, 200.0)
        else:
            starting_alloc = st.session_state.agent_equity.get(agent_name, 200.0)
            
        pnl    = equity - starting_alloc
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
            'NexusQuant Alpha Arena · Binance Mainnet · Not Financial Advice</div>',
            unsafe_allow_html=True)

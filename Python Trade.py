"""
=================================================================================
NIFTY 50 OPTIONS — AUTOMATED SUPPORT/RESISTANCE RETEST ALGO (DhanHQ V2)
=================================================================================
Strategy   : Fractal-based S&R zone detection -> Breakout -> Retest -> Bounce
Instrument : NIFTY 50 Index (Spot, for signal generation) -> ATM Weekly Option (execution)
Timeframe  : 5-minute candles
Framework  : dhanhq (official SDK) + asyncio + threading

READ THIS BEFORE YOU DEPLOY WITH REAL MONEY
---------------------------------------------------------------------------------
This script is written to be structurally complete and runnable, but several
spots are marked "# VERIFY:". These are places where broker-side field names,
CSV schemas, or SDK response shapes are known to change over time or vary by
account/segment, and I cannot guarantee they match what your account will
return today. Run this for at least several sessions in PAPER_TRADING = True
before ever flipping it to live, and print/inspect the raw SDK responses at
each VERIFY point against what this code assumes.

In particular, verify before going live:
  1. dhanhq security master CSV column names (they have changed historically).
  2. dhan.get_positions() field names for realized/unrealized P&L (differs by
     SDK version — some versions expose 'realizedProfit'/'unrealizedProfit',
     others nest P&L under different keys).
  3. Current NIFTY lot size (NSE revises this periodically — do not assume 25).
  4. WebSocket feed response packet structure for LTP (byte layout can change
     between SDK versions — v2 uses a defined binary/JSON feed response, print
     and inspect one raw message before trusting field names).
=================================================================================
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from enum import Enum
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# Official DhanHQ v2 SDK
from dhanhq import dhanhq, marketfeed


# =================================================================================
# SECTION 1: CONFIGURATION
# =================================================================================

# --- ENVIRONMENT TOGGLE ------------------------------------------------------
# TRUE  = No real orders are sent. All order calls are intercepted and logged/
#         simulated with a fake order-book. Use this to validate signal logic.
# FALSE = Real orders sent to Dhan. Only flip this after multiple clean
#         paper-trading sessions and after verifying every "# VERIFY" block.
PAPER_TRADING = True

# --- CREDENTIALS --------------------------------------------------------------
# Never hardcode secrets in the file. Set these as environment variables on
# your AWS box, e.g. in /etc/environment or a systemd EnvironmentFile.
DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")

if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
    raise EnvironmentError(
        "Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables before running."
    )

# --- INSTRUMENT CONFIG ---------------------------------------------------------
NIFTY_SPOT_SECURITY_ID = "13"          # Nifty 50 Index security id on Dhan
NIFTY_SPOT_EXCHANGE_SEGMENT = 1        # 1 = NSE Index/Equity segment in Dhan feed enums
                                        # VERIFY: marketfeed.NSE / marketfeed.IDX constants
                                        # in your installed SDK version vs raw int 1.

STRIKE_STEP = 50                       # Nifty strikes are in steps of 50
CANDLE_TIMEFRAME_MINUTES = 5           # Strategy runs on 5-min candles

# VERIFY: Confirm current NSE-mandated Nifty lot size before going live.
# As of recent cycles it has been revised more than once — do not assume.
NIFTY_LOT_SIZE = 75                    # <-- CONFIRM THIS AGAINST NSE CIRCULAR BEFORE LIVE USE
LOTS_PER_TRADE = 4                     # Hardcoded per your requirement: 4 lots
ORDER_QUANTITY = NIFTY_LOT_SIZE * LOTS_PER_TRADE

# --- S&R / FRACTAL CONFIG -------------------------------------------------------
HISTORICAL_CANDLE_LOOKBACK = 100       # scan last 100 candles for fractals
FRACTAL_WINDOW = 5                     # 5 candles left & right for swing confirmation
ZONE_CLUSTER_TOLERANCE_POINTS = 25     # swing points within this range get merged into 1 zone
ZONE_BUFFER_POINTS = 15                # +/- padding around a zone (point-based, per your spec)
ATR_PERIOD = 14                        # used as an alternative/blended buffer basis
USE_ATR_BUFFER = False                 # False = pure point-based buffer (15 pts) as specified
ATR_BUFFER_MULTIPLIER = 0.3            # only used if USE_ATR_BUFFER = True

# --- RISK MANAGEMENT (HARDCODED GUARDRAILS) -------------------------------------
MAX_DAILY_LOSS = -10000.00             # Kill-switch threshold (₹)
MAX_TRADES_PER_DAY = 3                 # Overtrading filter
RISK_CHECK_BEFORE_EVERY_ENTRY = True   # Always re-check P&L right before firing an order

# --- SESSION TIMES ---------------------------------------------------------------
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
SQUARE_OFF_TIME = dtime(15, 15)        # stop taking new entries after this time

# --- LOGGING -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(threadName)s | %(message)s",
    handlers=[
        logging.FileHandler("nifty_sr_bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("SR_RETEST_BOT")


# =================================================================================
# SECTION 2: DHAN CLIENT INITIALIZATION
# =================================================================================

# The official REST client — used for orders, positions, security master, etc.
dhan = dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)


# =================================================================================
# SECTION 3: STATE MACHINE DEFINITION
# =================================================================================

class SRState(Enum):
    IDLE = 0            # Waiting for a breakout
    BREAKOUT = 1         # Candle closed outside a zone
    RETEST_DIP = 2        # Price has come back inside the zone buffer
    TRIGGER_BOUNCE = 3     # Confirmed bounce/rejection -> fire entry


@dataclass
class SRZone:
    """A consolidated support/resistance zone (clustered from multiple fractals)."""
    center: float
    zone_type: str          # "support" or "resistance"
    lower: float = 0.0
    upper: float = 0.0
    touch_count: int = 1

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper


@dataclass
class StrategyContext:
    """Mutable runtime state for the state machine, tracked per active zone."""
    state: SRState = SRState.IDLE
    active_zone: Optional[SRZone] = None
    breakout_direction: Optional[str] = None   # "up" (resistance broken) / "down" (support broken)
    breakout_candle_close: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)


# =================================================================================
# SECTION 4: CANDLE AGGREGATOR
# =================================================================================

class CandleAggregator:
    """
    Builds 5-minute OHLC candles from raw tick LTP data streamed off the
    WebSocket. Thread-safe via a lock since ticks arrive on the feed thread
    while the main asyncio loop consumes completed candles.
    """

    def __init__(self, timeframe_minutes: int = 5, max_candles: int = 300):
        self.timeframe = timedelta(minutes=timeframe_minutes)
        self.max_candles = max_candles
        self.candles: deque = deque(maxlen=max_candles)
        self._current_candle: Optional[Dict] = None
        self._current_bucket_start: Optional[datetime] = None
        self._lock = threading.Lock()

    def _bucket_start(self, ts: datetime) -> datetime:
        """Floor a timestamp down to its containing 5-min bucket start."""
        minute = (ts.minute // CANDLE_TIMEFRAME_MINUTES) * CANDLE_TIMEFRAME_MINUTES
        return ts.replace(minute=minute, second=0, microsecond=0)

    def on_tick(self, ltp: float, ts: Optional[datetime] = None):
        """Feed a single tick (LTP) into the aggregator. Called from the feed thread."""
        ts = ts or datetime.now()
        bucket = self._bucket_start(ts)

        with self._lock:
            if self._current_bucket_start is None:
                self._start_new_candle(bucket, ltp)
                return

            if bucket == self._current_bucket_start:
                # Update the candle in progress
                c = self._current_candle
                c["high"] = max(c["high"], ltp)
                c["low"] = min(c["low"], ltp)
                c["close"] = ltp
            elif bucket > self._current_bucket_start:
                # Bucket rolled over -> finalize previous candle, start new one
                self.candles.append(self._current_candle)
                self._start_new_candle(bucket, ltp)
            # bucket < current_bucket_start: stale/out-of-order tick, ignore

    def _start_new_candle(self, bucket_start: datetime, ltp: float):
        self._current_bucket_start = bucket_start
        self._current_candle = {
            "timestamp": bucket_start,
            "open": ltp,
            "high": ltp,
            "low": ltp,
            "close": ltp,
        }

    def get_dataframe(self) -> pd.DataFrame:
        """Return completed candles (NOT including the in-progress one) as a DataFrame."""
        with self._lock:
            data = list(self.candles)
        if not data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        return pd.DataFrame(data)

    def get_last_completed_candle(self) -> Optional[Dict]:
        with self._lock:
            if self.candles:
                return dict(self.candles[-1])
        return None


# =================================================================================
# SECTION 5: FRACTAL / SWING DETECTION + ZONE CLUSTERING
# =================================================================================

def detect_fractals(df: pd.DataFrame, window: int = FRACTAL_WINDOW) -> pd.DataFrame:
    """
    Classic Williams Fractal swing detection.
    A swing high at index i requires df.high[i] to be the max within
    [i-window, i+window]. Same logic (min) for swing lows.

    Returns the input df with two boolean columns added: 'swing_high', 'swing_low'.
    """
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"] = False

    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    for i in range(window, n - window):
        local_high_window = highs[i - window: i + window + 1]
        local_low_window = lows[i - window: i + window + 1]

        if highs[i] == local_high_window.max() and np.argmax(local_high_window) == window:
            df.at[df.index[i], "swing_high"] = True

        if lows[i] == local_low_window.min() and np.argmin(local_low_window) == window:
            df.at[df.index[i], "swing_low"] = True

    return df


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Simple ATR computed from OHLC (uses close-to-close as proxy for true range
    components since spot index candles built from LTP ticks won't have a
    separate traded 'high/low' beyond intra-bucket LTP excursions)."""
    if len(df) < period + 1:
        return 0.0
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return float(tr.rolling(period).mean().iloc[-1])


def cluster_into_zones(df: pd.DataFrame, atr_value: float = 0.0) -> List[SRZone]:
    """
    Takes swing highs/lows from `detect_fractals` and clusters nearby levels
    (within ZONE_CLUSTER_TOLERANCE_POINTS) into consolidated zones. Each zone
    gets a symmetric buffer (point-based by default, ATR-based if configured).
    """
    swing_highs = sorted(df.loc[df["swing_high"], "high"].tolist())
    swing_lows = sorted(df.loc[df["swing_low"], "low"].tolist())

    buffer_pts = (atr_value * ATR_BUFFER_MULTIPLIER) if (USE_ATR_BUFFER and atr_value > 0) \
        else ZONE_BUFFER_POINTS

    def cluster(levels: List[float], zone_type: str) -> List[SRZone]:
        if not levels:
            return []
        zones = []
        current_group = [levels[0]]

        for lvl in levels[1:]:
            if lvl - current_group[-1] <= ZONE_CLUSTER_TOLERANCE_POINTS:
                current_group.append(lvl)
            else:
                center = float(np.mean(current_group))
                zones.append(SRZone(
                    center=center,
                    zone_type=zone_type,
                    lower=center - buffer_pts,
                    upper=center + buffer_pts,
                    touch_count=len(current_group),
                ))
                current_group = [lvl]

        # flush last group
        center = float(np.mean(current_group))
        zones.append(SRZone(
            center=center,
            zone_type=zone_type,
            lower=center - buffer_pts,
            upper=center + buffer_pts,
            touch_count=len(current_group),
        ))
        return zones

    resistance_zones = cluster(swing_highs, "resistance")
    support_zones = cluster(swing_lows, "support")
    return resistance_zones + support_zones


def build_sr_zones(candle_df: pd.DataFrame) -> List[SRZone]:
    """Full pipeline: last N candles -> fractals -> ATR -> clustered zones."""
    if len(candle_df) < (2 * FRACTAL_WINDOW + 1):
        return []  # not enough data yet

    window_df = candle_df.tail(HISTORICAL_CANDLE_LOOKBACK).reset_index(drop=True)
    fractal_df = detect_fractals(window_df, FRACTAL_WINDOW)
    atr_val = compute_atr(window_df, ATR_PERIOD)
    return cluster_into_zones(fractal_df, atr_val)


# =================================================================================
# SECTION 6: STATE MACHINE ENGINE
# =================================================================================

class RetestStateMachine:
    """
    Implements the strict 4-state machine described in the spec:
      0 IDLE            -> watching for a breakout candle close outside any zone
      1 BREAKOUT        -> confirmed close outside zone; now watch for retrace
      2 RETEST_DIP      -> price re-enters the zone buffer (retest in progress)
      3 TRIGGER_BOUNCE  -> confirmed bounce (support) / rejection (resistance)
                            out of the zone -> fire an entry signal
    Operates strictly on COMPLETED 5-min candles (never on intra-candle ticks),
    since "candle closes outside the zone" is explicitly a closed-candle event.
    """

    def __init__(self):
        self.ctx = StrategyContext()

    def reset(self):
        self.ctx = StrategyContext()

    def on_new_candle(self, candle: Dict, zones: List[SRZone]) -> Optional[str]:
        """
        Feed one newly completed candle + the current zone list.
        Returns "CALL" / "PUT" if state 3 triggers this candle, else None.
        """
        close = candle["close"]
        signal = None

        if self.ctx.state == SRState.IDLE:
            self._check_breakout(close, zones)

        elif self.ctx.state == SRState.BREAKOUT:
            self._check_retest(close)

        elif self.ctx.state == SRState.RETEST_DIP:
            signal = self._check_bounce(close)

        elif self.ctx.state == SRState.TRIGGER_BOUNCE:
            # Terminal state for this cycle — caller should reset() after
            # consuming the signal and placing (or skipping) the trade.
            pass

        self.ctx.last_updated = datetime.now()
        return signal

    def _check_breakout(self, close: float, zones: List[SRZone]):
        for zone in zones:
            if zone.zone_type == "resistance" and close > zone.upper:
                self.ctx.state = SRState.BREAKOUT
                self.ctx.active_zone = zone
                self.ctx.breakout_direction = "up"
                self.ctx.breakout_candle_close = close
                log.info(f"[STATE 0->1] Breakout ABOVE resistance zone "
                         f"{zone.lower:.2f}-{zone.upper:.2f} | close={close:.2f}")
                return
            if zone.zone_type == "support" and close < zone.lower:
                self.ctx.state = SRState.BREAKOUT
                self.ctx.active_zone = zone
                self.ctx.breakout_direction = "down"
                self.ctx.breakout_candle_close = close
                log.info(f"[STATE 0->1] Breakdown BELOW support zone "
                         f"{zone.lower:.2f}-{zone.upper:.2f} | close={close:.2f}")
                return

    def _check_retest(self, close: float):
        zone = self.ctx.active_zone
        if zone is None:
            self.reset()
            return

        if zone.contains(close):
            self.ctx.state = SRState.RETEST_DIP
            log.info(f"[STATE 1->2] Retest: price back inside zone "
                     f"{zone.lower:.2f}-{zone.upper:.2f} | close={close:.2f}")
        else:
            # Price never came back to retest — invalidate if it moves further
            # away without retesting within a reasonable number of candles.
            # (Simple invalidation: if direction reverses hard, reset to IDLE.)
            if self.ctx.breakout_direction == "up" and close < zone.upper:
                log.info("[STATE 1 INVALIDATED] Reversed back through zone without clean retest. Resetting.")
                self.reset()
            elif self.ctx.breakout_direction == "down" and close > zone.lower:
                log.info("[STATE 1 INVALIDATED] Reversed back through zone without clean retest. Resetting.")
                self.reset()

    def _check_bounce(self, close: float) -> Optional[str]:
        zone = self.ctx.active_zone
        if zone is None:
            self.reset()
            return None

        # Bullish bounce out of a SUPPORT zone that was broken-then-retested from above
        # -> only valid if original breakout direction context implies support hold (i.e.
        #    this is a support zone acting as support, price dipped into buffer and bounced up)
        if zone.zone_type == "support" and close > zone.upper:
            self.ctx.state = SRState.TRIGGER_BOUNCE
            log.info(f"[STATE 2->3] BULLISH BOUNCE confirmed off support zone "
                     f"{zone.lower:.2f}-{zone.upper:.2f} | close={close:.2f} -> CALL signal")
            return "CALL"

        # Bearish rejection out of a RESISTANCE zone
        if zone.zone_type == "resistance" and close < zone.lower:
            self.ctx.state = SRState.TRIGGER_BOUNCE
            log.info(f"[STATE 2->3] BEARISH REJECTION confirmed off resistance zone "
                     f"{zone.lower:.2f}-{zone.upper:.2f} | close={close:.2f} -> PUT signal")
            return "PUT"

        return None


# =================================================================================
# SECTION 7: SECURITY MASTER LOOKUP (ATM WEEKLY OPTION RESOLUTION)
# =================================================================================

class SecurityMasterResolver:
    """
    Downloads/caches Dhan's Security Master (scrip master) CSV and resolves
    the exact tradable security_id for a given ATM strike + option type +
    the *current active weekly expiry* for NIFTY.

    # VERIFY: Dhan periodically updates the security master file URL and its
    # column schema. Print df.columns and a sample row before relying on the
    # exact column names used below (SEM_TRADING_SYMBOL / SEM_EXPIRY_DATE / etc.
    # are the commonly documented names as of recent SDK versions, but confirm
    # against the live file before trading).
    """

    SECURITY_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._last_fetch: Optional[datetime] = None

    def _load(self, force: bool = False):
        if self._df is not None and not force and self._last_fetch and \
                (datetime.now() - self._last_fetch) < timedelta(hours=6):
            return  # use cache, refresh at most every 6h

        log.info("Downloading Dhan security master CSV ...")
        self._df = pd.read_csv(self.SECURITY_MASTER_URL, low_memory=False)
        self._last_fetch = datetime.now()
        log.info(f"Security master loaded: {len(self._df)} rows.")

    def get_current_weekly_expiry(self) -> str:
        """Return the nearest (current active) weekly expiry date for NIFTY options."""
        self._load()
        df = self._df

        # VERIFY these column/value filters against the actual CSV schema.
        nifty_opts = df[
            (df["SEM_TRADING_SYMBOL"].astype(str).str.startswith("NIFTY")) &
            (df["SEM_EXCH_INSTRUMENT_TYPE"].astype(str).isin(["OPTIDX", "OP"]))
        ].copy()

        nifty_opts["SEM_EXPIRY_DATE"] = pd.to_datetime(
            nifty_opts["SEM_EXPIRY_DATE"], errors="coerce"
        )
        today = pd.Timestamp(datetime.now().date())
        future_expiries = nifty_opts[nifty_opts["SEM_EXPIRY_DATE"] >= today]

        if future_expiries.empty:
            raise RuntimeError("No future NIFTY option expiries found in security master.")

        nearest_expiry = future_expiries["SEM_EXPIRY_DATE"].min()
        return nearest_expiry.strftime("%Y-%m-%d")

    def resolve_atm_option(self, spot_price: float, option_type: str) -> Dict:
        """
        option_type: "CE" or "PE"
        Returns dict with keys: security_id, trading_symbol, strike, expiry
        """
        self._load()
        df = self._df

        atm_strike = round(spot_price / STRIKE_STEP) * STRIKE_STEP
        expiry = self.get_current_weekly_expiry()

        # VERIFY: exact column names for strike price and option type flag.
        candidates = df[
            (df["SEM_TRADING_SYMBOL"].astype(str).str.startswith("NIFTY")) &
            (df["SEM_STRIKE_PRICE"].astype(float) == float(atm_strike)) &
            (df["SEM_OPTION_TYPE"].astype(str).str.upper() == option_type.upper()) &
            (pd.to_datetime(df["SEM_EXPIRY_DATE"], errors="coerce").dt.strftime("%Y-%m-%d") == expiry)
        ]

        if candidates.empty:
            raise RuntimeError(
                f"Could not resolve ATM {option_type} contract for strike {atm_strike}, "
                f"expiry {expiry}. Inspect security master schema (# VERIFY blocks)."
            )

        row = candidates.iloc[0]
        return {
            "security_id": str(row["SEM_SMST_SECURITY_ID"]),   # VERIFY exact ID column name
            "trading_symbol": str(row["SEM_TRADING_SYMBOL"]),
            "strike": atm_strike,
            "expiry": expiry,
        }


# =================================================================================
# SECTION 8: ORDER EXECUTION LAYER (with PAPER_TRADING simulation)
# =================================================================================

class OrderExecutor:
    """
    Wraps dhan order placement. In PAPER_TRADING mode, no network order calls
    are made — everything is logged and a synthetic order id is returned so
    downstream logic (trade counters, position tracking) behaves identically
    to live mode.
    """

    def __init__(self, paper: bool = True):
        self.paper = paper
        self._paper_order_seq = 1000
        self._paper_positions: List[Dict] = []

    async def place_market_buy(self, security_id: str, exchange_segment: str,
                                quantity: int, trading_symbol: str = "") -> Dict:
        if self.paper:
            self._paper_order_seq += 1
            order_id = f"PAPER-{self._paper_order_seq}"
            log.info(f"[PAPER ORDER] BUY MARKET {trading_symbol} qty={quantity} "
                     f"security_id={security_id} -> simulated order_id={order_id}")
            self._paper_positions.append({
                "order_id": order_id,
                "security_id": security_id,
                "trading_symbol": trading_symbol,
                "quantity": quantity,
                "side": "BUY",
                "timestamp": datetime.now(),
            })
            return {"status": "success", "orderId": order_id, "paper": True}

        # LIVE ORDER PATH
        # VERIFY: exact kwarg names/enums for your installed dhanhq version.
        # This matches the commonly documented dhanhq v2 place_order signature.
        try:
            response = await asyncio.to_thread(
                dhan.place_order,
                security_id=security_id,
                exchange_segment=exchange_segment,          # e.g. dhan.NSE_FNO
                transaction_type=dhan.BUY,
                quantity=quantity,
                order_type=dhan.MARKET,
                product_type=dhan.INTRA,
                price=0,
            )
            log.info(f"[LIVE ORDER] Response: {response}")
            return response
        except Exception as e:
            log.error(f"Live order placement failed: {e}")
            traceback.print_exc()
            return {"status": "failure", "error": str(e)}

    async def exit_all_positions(self):
        """Emergency liquidation: closes every open options position at market."""
        if self.paper:
            log.warning(f"[PAPER] Emergency exit triggered — clearing "
                        f"{len(self._paper_positions)} simulated position(s).")
            self._paper_positions.clear()
            return

        try:
            positions = await asyncio.to_thread(dhan.get_positions)
            open_positions = [
                p for p in positions.get("data", [])
                if float(p.get("netQty", 0)) != 0
            ]
            for pos in open_positions:
                qty = abs(int(pos["netQty"]))
                side = dhan.SELL if int(pos["netQty"]) > 0 else dhan.BUY
                log.warning(f"[KILL-SWITCH] Squaring off {pos.get('tradingSymbol')} qty={qty}")
                await asyncio.to_thread(
                    dhan.place_order,
                    security_id=pos["securityId"],
                    exchange_segment=pos["exchangeSegment"],
                    transaction_type=side,
                    quantity=qty,
                    order_type=dhan.MARKET,
                    product_type=dhan.INTRA,
                    price=0,
                )
        except Exception as e:
            log.error(f"Emergency exit failed: {e}")
            traceback.print_exc()

    async def cancel_all_pending_orders(self):
        if self.paper:
            log.warning("[PAPER] Cancel-all-orders called (no-op in paper mode).")
            return
        try:
            orders = await asyncio.to_thread(dhan.get_order_list)
            pending = [o for o in orders.get("data", [])
                       if o.get("orderStatus") in ("PENDING", "TRANSIT", "OPEN")]
            for o in pending:
                await asyncio.to_thread(dhan.cancel_order, o["orderId"])
                log.warning(f"[KILL-SWITCH] Cancelled pending order {o['orderId']}")
        except Exception as e:
            log.error(f"Cancel-all-orders failed: {e}")
            traceback.print_exc()


# =================================================================================
# SECTION 9: RISK MANAGER (DAILY LOSS KILL-SWITCH + TRADE CAP)
# =================================================================================

class RiskManager:
    def __init__(self, executor: OrderExecutor):
        self.executor = executor
        self.trades_today = 0
        self.trading_day = datetime.now().date()
        self.kill_switch_triggered = False

    def _roll_day_if_needed(self):
        if datetime.now().date() != self.trading_day:
            self.trading_day = datetime.now().date()
            self.trades_today = 0
            self.kill_switch_triggered = False
            log.info("New trading day detected — risk counters reset.")

    async def get_daily_pnl(self) -> float:
        """
        Pulls current intraday P&L via dhan.get_positions().
        # VERIFY: field names for realized/unrealized P&L differ across SDK
        # versions. Print a raw get_positions() response once and confirm
        # 'realizedProfit' / 'unrealizedProfit' (or whatever your version
        # returns) before trusting this number for the kill-switch.
        """
        if self.executor.paper:
            # In paper mode there is no real P&L feed; return 0 so the
            # kill-switch never fires spuriously during simulation.
            return 0.0

        try:
            positions = await asyncio.to_thread(dhan.get_positions)
            total_pnl = 0.0
            for p in positions.get("data", []):
                realized = float(p.get("realizedProfit", 0) or 0)
                unrealized = float(p.get("unrealizedProfit", 0) or 0)
                total_pnl += realized + unrealized
            return total_pnl
        except Exception as e:
            log.error(f"Failed to fetch positions for P&L check: {e}")
            # Fail-safe: if we can't verify P&L, treat as unknown and block
            # new entries rather than assuming we're safe.
            return float("-inf")

    async def pre_trade_check(self) -> bool:
        """Returns True if it's safe to place a new trade right now."""
        self._roll_day_if_needed()

        if self.kill_switch_triggered:
            log.warning("Kill-switch already triggered today — no further trades.")
            return False

        if self.trades_today >= MAX_TRADES_PER_DAY:
            log.warning(f"Max trades per day ({MAX_TRADES_PER_DAY}) reached — skipping entry.")
            return False

        now_t = datetime.now().time()
        if now_t >= SQUARE_OFF_TIME:
            log.warning("Past square-off cutoff time — no new entries.")
            return False

        pnl = await self.get_daily_pnl()
        if pnl <= MAX_DAILY_LOSS:
            log.critical(f"DAILY LOSS LIMIT BREACHED: P&L={pnl:.2f} <= {MAX_DAILY_LOSS}. "
                         f"Engaging kill-switch.")
            await self.engage_kill_switch()
            return False

        return True

    async def engage_kill_switch(self):
        """Liquidate everything, cancel pending orders, and hard-exit the process."""
        self.kill_switch_triggered = True
        log.critical("=== KILL-SWITCH ENGAGED: liquidating all positions ===")
        await self.executor.exit_all_positions()
        log.critical("=== Cancelling all pending orders ===")
        await self.executor.cancel_all_pending_orders()
        log.critical("=== Kill-switch actions complete. Terminating script. ===")
        sys.exit(1)

    def register_trade(self):
        self.trades_today += 1
        log.info(f"Trade count today: {self.trades_today}/{MAX_TRADES_PER_DAY}")


# =================================================================================
# SECTION 10: WEBSOCKET FEED HANDLER (RUNS ON ITS OWN THREAD)
# =================================================================================

class LiveFeedHandler:
    """
    Wraps dhanhq.marketfeed to stream Nifty Spot Index ticks.
    Runs on a dedicated background thread so the asyncio event loop driving
    order/risk logic is never blocked by the feed's own event loop.

    # VERIFY: marketfeed.DhanFeed constructor signature and the shape of
    # incoming tick messages (field names for LTP, e.g. 'LTP' vs 'ltp' vs
    # 'last_traded_price') can differ between SDK releases — print a raw
    # message once on connect and confirm before trusting `on_tick` parsing.
    """

    def __init__(self, aggregator: CandleAggregator):
        self.aggregator = aggregator
        self._feed = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _on_message(self, message):
        try:
            # VERIFY: adapt this parsing to the actual message schema you see
            # printed from the feed in your SDK version.
            data = message if isinstance(message, dict) else json.loads(message)
            ltp = data.get("LTP") or data.get("ltp") or data.get("last_traded_price")
            if ltp is not None:
                self.aggregator.on_tick(float(ltp), datetime.now())
        except Exception as e:
            log.error(f"Feed message parse error: {e} | raw={message}")

    def _run_feed(self):
        instruments = [(NIFTY_SPOT_EXCHANGE_SEGMENT, NIFTY_SPOT_SECURITY_ID, marketfeed.Ticker)]
        try:
            self._feed = marketfeed.DhanFeed(
                DHAN_CLIENT_ID,
                DHAN_ACCESS_TOKEN,
                instruments,
            )
            log.info("Market feed connecting ...")
            while not self._stop_event.is_set():
                self._feed.run_forever()
                response = self._feed.get_data()
                if response:
                    self._on_message(response)
        except Exception as e:
            log.error(f"Market feed error: {e}")
            traceback.print_exc()
            # Basic reconnect backoff — production use should add exponential
            # backoff + max retry ceiling + alerting.
            time.sleep(5)
            if not self._stop_event.is_set():
                self._run_feed()

    def start(self):
        self._thread = threading.Thread(target=self._run_feed, name="DhanFeedThread", daemon=True)
        self._thread.start()
        log.info("Live feed thread started.")

    def stop(self):
        self._stop_event.set()
        if self._feed:
            try:
                self._feed.disconnect()
            except Exception:
                pass


# =================================================================================
# SECTION 11: MAIN STRATEGY ORCHESTRATOR (ASYNC)
# =================================================================================

class SRRetestBot:
    def __init__(self):
        self.aggregator = CandleAggregator(CANDLE_TIMEFRAME_MINUTES)
        self.state_machine = RetestStateMachine()
        self.resolver = SecurityMasterResolver()
        self.executor = OrderExecutor(paper=PAPER_TRADING)
        self.risk = RiskManager(self.executor)
        self.feed = LiveFeedHandler(self.aggregator)
        self._last_seen_candle_ts: Optional[datetime] = None
        self._zones: List[SRZone] = []

    def _within_market_hours(self) -> bool:
        now_t = datetime.now().time()
        return MARKET_OPEN <= now_t <= MARKET_CLOSE

    async def _refresh_zones(self):
        df = self.aggregator.get_dataframe()
        if len(df) >= (2 * FRACTAL_WINDOW + 1):
            self._zones = build_sr_zones(df)
            if self._zones:
                zone_summary = ", ".join(
                    f"{z.zone_type}:{z.lower:.0f}-{z.upper:.0f}" for z in self._zones
                )
                log.debug(f"Active zones: {zone_summary}")

    async def _handle_signal(self, signal: str, spot_price: float):
        """signal is 'CALL' or 'PUT' from state 3 trigger."""
        option_type = "CE" if signal == "CALL" else "PE"

        ok = await self.risk.pre_trade_check()
        if not ok:
            log.warning(f"Signal {signal} generated but blocked by risk manager.")
            self.state_machine.reset()
            return

        try:
            contract = self.resolver.resolve_atm_option(spot_price, option_type)
        except Exception as e:
            log.error(f"Could not resolve option contract for signal {signal}: {e}")
            self.state_machine.reset()
            return

        log.info(f"FIRING ENTRY: {signal} -> {contract['trading_symbol']} "
                 f"(strike={contract['strike']}, expiry={contract['expiry']}) "
                 f"qty={ORDER_QUANTITY}")

        result = await self.executor.place_market_buy(
            security_id=contract["security_id"],
            exchange_segment="NSE_FNO",   # VERIFY exact enum/string for your SDK version
            quantity=ORDER_QUANTITY,
            trading_symbol=contract["trading_symbol"],
        )

        if result.get("status") == "success":
            self.risk.register_trade()
        else:
            log.error(f"Order placement did not confirm success: {result}")

        # Cycle complete — return state machine to idle to look for the next setup.
        self.state_machine.reset()

    async def _strategy_loop(self):
        """
        Runs every CANDLE_TIMEFRAME_MINUTES, checks for a newly completed
        candle, updates zones, and steps the state machine.
        """
        while True:
            if not self._within_market_hours():
                log.info("Outside market hours — sleeping 60s.")
                await asyncio.sleep(60)
                continue

            last_candle = self.aggregator.get_last_completed_candle()
            if last_candle and last_candle["timestamp"] != self._last_seen_candle_ts:
                self._last_seen_candle_ts = last_candle["timestamp"]
                log.info(f"New 5-min candle closed: {last_candle}")

                await self._refresh_zones()
                signal = self.state_machine.on_new_candle(last_candle, self._zones)

                if signal:
                    await self._handle_signal(signal, last_candle["close"])

            # Poll frequently but cheaply; actual "new candle" detection is
            # timestamp-based above so this doesn't double-process.
            await asyncio.sleep(5)

    async def run(self):
        log.info("=" * 70)
        log.info(f"Starting SR Retest Bot | PAPER_TRADING={PAPER_TRADING}")
        log.info(f"Order size: {ORDER_QUANTITY} qty ({LOTS_PER_TRADE} lots x {NIFTY_LOT_SIZE})")
        log.info(f"Max daily loss: {MAX_DAILY_LOSS} | Max trades/day: {MAX_TRADES_PER_DAY}")
        log.info("=" * 70)

        self.feed.start()

        try:
            await self._strategy_loop()
        except (KeyboardInterrupt, SystemExit):
            log.info("Shutdown signal received.")
        finally:
            self.feed.stop()
            log.info("Bot stopped cleanly.")


# =================================================================================
# SECTION 12: ENTRYPOINT
# =================================================================================

if __name__ == "__main__":
    bot = SRRetestBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("Interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as fatal:
        log.critical(f"Fatal unhandled error: {fatal}")
        traceback.print_exc()
        sys.exit(1)

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://data-api.binance.vision"
MAX_SYMBOLS = int(os.getenv("EMA_MAX_SYMBOLS", "200"))
MIN_QUOTE_VOLUME = float(os.getenv("EMA_MIN_QUOTE_VOLUME", "5000000"))
CANDLE_LIMIT = 340
MIN_BARS = 230
TIMEFRAMES = {"4H": "4h", "1D": "1d"}
RECENT_BARS = {"4H": 6, "1D": 1}
LIFECYCLE_BARS = {"4H": 24, "1D": 12}
TRIGGER_EXPIRY = {"4H": 6, "1D": 3}
ALWAYS_INCLUDE = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT"}
STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "AEUR", "TRY",
    "BRL", "GBP", "AUD", "UAH", "RUB", "BIDR", "IDRT", "NGN", "ZAR"
}

RULES: dict[str, float | int] = {
    "ema_length": 200,
    "atr_length": 14,
    "ema_slope_lookback": 10,
    "touch_zone_atr": 0.18,
    "rejection_close_buffer_atr": 0.03,
    "rejection_max_penetration_atr": 0.75,
    "rejection_min_range_atr": 0.45,
    "rejection_min_clv": 0.58,
    "rejection_min_wick_fraction": 0.22,
    "rejection_context_bars": 4,
    "rejection_context_required": 2,
    "acceptance_close_atr": 0.12,
    "acceptance_min_body_atr": 0.38,
    "acceptance_min_body_fraction": 0.48,
    "acceptance_min_clv": 0.64,
    "acceptance_confirmation_bars": 4,
    "acceptance_retest_zone_atr": 0.22,
    "acceptance_hold_buffer_atr": 0.02,
    "acceptance_fail_buffer_atr": 0.10,
    "acceptance_required_hold_closes": 2,
    "stop_buffer_atr": 0.10,
    "touch_count_lookback": 30,
    "cross_count_lookback": 14,
    "pivot_left": 2,
    "pivot_right": 2,
    "structure_lookback": 80,
    "minimum_structure_room_r": 1.25,
    "no_chase_after_r": 0.75,
}

LOG = logging.getLogger("ema200")


@dataclass
class Instrument:
    symbol: str
    base: str
    quote: str
    tick_size: float
    step_size: float
    min_qty: float
    quote_volume_24h: float
    last_price: float


@dataclass
class Context:
    symbol: str
    timeframe: str
    completed_ts: int
    close: float
    ema200: float
    atr14: float
    ema_slope_atr: float
    distance_from_ema_atr: float
    touch_count: int
    cross_count: int
    regime: str


@dataclass
class Signal:
    symbol: str
    timeframe: str
    side: str
    setup: str
    status: str
    signal_ts: int
    signal_time_utc: str
    age_bars: int
    score: float
    grade: str
    last_price: float
    ema200: float
    atr14: float
    trigger: float
    stop: float
    target_1r: float
    target_2r: float
    target_3r: float
    structure_target: Optional[float]
    structure_target_r: Optional[float]
    risk_pct: float
    quote_volume_24h: float
    touch_count: int
    cross_count: int
    ema_slope_atr: float
    distance_from_ema_atr: float
    volume_ratio: float
    trigger_ts: Optional[int] = None
    trigger_time_utc: Optional[str] = None
    invalidation_ts: Optional[int] = None
    invalidation_time_utc: Optional[str] = None
    current_r: Optional[float] = None
    no_chase: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    is_recent: bool = True
    state_change: Optional[str] = None

    def key(self) -> str:
        return f"{self.symbol}|{self.timeframe}|{self.side}|{self.setup}|{self.signal_ts}"


class PublicClient:
    def __init__(self) -> None:
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=0.7,
            status_forcelist=(408, 425, 429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=24, pool_maxsize=24)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "EMA200-Crypto-Scanner/1.0"})
        self.session.mount("https://", adapter)

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        response = self.session.get(BASE_URL + path, params=params, timeout=25)
        response.raise_for_status()
        return response.json()

    def candles(self, symbol: str, interval: str) -> pd.DataFrame:
        rows = self.get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": CANDLE_LIMIT},
        )
        now_ms = int(time.time() * 1000)
        parsed = []
        for row in rows:
            if int(row[6]) > now_ms:
                continue
            parsed.append(
                {
                    "ts": int(row[0]),
                    "close_ts": int(row[6]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume_base": float(row[5]),
                    "volume_quote": float(row[7]),
                    "trades": int(row[8]),
                }
            )
        return pd.DataFrame(parsed).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def utc_iso(ts: Optional[int]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(timespec="minutes")


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 68:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def round_tick(value: float, tick: float, direction: str) -> float:
    if tick <= 0:
        return value
    units = value / tick
    rounded = math.ceil(units - 1e-10) if direction == "up" else math.floor(units + 1e-10)
    return rounded * tick


def wilder_atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_values("ts").reset_index(drop=True)
    data["ema200"] = data["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    data["atr14"] = wilder_atr(data, 14)
    data["range"] = (data["high"] - data["low"]).clip(lower=0)
    data["body"] = (data["close"] - data["open"]).abs()
    data["upper_wick"] = data["high"] - data[["open", "close"]].max(axis=1)
    data["lower_wick"] = data[["open", "close"]].min(axis=1) - data["low"]
    safe_range = data["range"].replace(0, np.nan)
    safe_atr = data["atr14"].replace(0, np.nan)
    data["body_fraction"] = (data["body"] / safe_range).fillna(0.0)
    data["upper_wick_fraction"] = (data["upper_wick"] / safe_range).fillna(0.0)
    data["lower_wick_fraction"] = (data["lower_wick"] / safe_range).fillna(0.0)
    data["clv"] = ((data["close"] - data["low"]) / safe_range).fillna(0.5)
    data["body_atr"] = (data["body"] / safe_atr).fillna(0.0)
    data["range_atr"] = (data["range"] / safe_atr).fillna(0.0)
    data["distance_ema_atr"] = ((data["close"] - data["ema200"]) / safe_atr).fillna(0.0)
    data["ema_slope_atr"] = ((data["ema200"] - data["ema200"].shift(10)) / safe_atr).fillna(0.0)
    zone = float(RULES["touch_zone_atr"]) * data["atr14"]
    data["ema_touch"] = (data["low"] <= data["ema200"] + zone) & (data["high"] >= data["ema200"] - zone)
    touch_start = data["ema_touch"] & ~data["ema_touch"].shift(1, fill_value=False)
    data["touch_count"] = touch_start.astype(int).rolling(30, min_periods=1).sum().astype(int)
    side = np.sign(data["close"] - data["ema200"])
    side = pd.Series(side, index=data.index).replace(0, np.nan).ffill().fillna(0)
    crosses = side.ne(side.shift(1)) & side.ne(0) & side.shift(1).fillna(0).ne(0)
    data["cross_count"] = crosses.astype(int).rolling(14, min_periods=1).sum().astype(int)
    median_volume = data["volume_quote"].rolling(20, min_periods=5).median()
    data["volume_ratio"] = (data["volume_quote"] / median_volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return data


def context_above(data: pd.DataFrame, index: int) -> bool:
    sample = data.iloc[max(0, index - 4):index]
    if sample.empty:
        return False
    return int((sample["close"] >= sample["ema200"] - 0.04 * sample["atr14"]).sum()) >= min(2, len(sample))


def context_below(data: pd.DataFrame, index: int) -> bool:
    sample = data.iloc[max(0, index - 4):index]
    if sample.empty:
        return False
    return int((sample["close"] <= sample["ema200"] + 0.04 * sample["atr14"]).sum()) >= min(2, len(sample))


def rejection_candidate(data: pd.DataFrame, index: int, side: str) -> bool:
    row = data.iloc[index]
    if not np.isfinite(row["ema200"]) or not np.isfinite(row["atr14"]) or row["atr14"] <= 0:
        return False
    atr = float(row["atr14"])
    ema = float(row["ema200"])
    if float(row["range_atr"]) < float(RULES["rejection_min_range_atr"]):
        return False
    zone = float(RULES["touch_zone_atr"]) * atr
    penetration = float(RULES["rejection_max_penetration_atr"]) * atr
    close_buffer = float(RULES["rejection_close_buffer_atr"]) * atr
    if side == "LONG":
        return (
            context_above(data, index)
            and float(row["low"]) <= ema + zone
            and float(row["low"]) >= ema - penetration
            and float(row["close"]) >= ema + close_buffer
            and float(row["clv"]) >= float(RULES["rejection_min_clv"])
            and (
                float(row["lower_wick_fraction"]) >= float(RULES["rejection_min_wick_fraction"])
                or float(row["close"]) > float(row["open"])
            )
        )
    return (
        context_below(data, index)
        and float(row["high"]) >= ema - zone
        and float(row["high"]) <= ema + penetration
        and float(row["close"]) <= ema - close_buffer
        and float(row["clv"]) <= 1.0 - float(RULES["rejection_min_clv"])
        and (
            float(row["upper_wick_fraction"]) >= float(RULES["rejection_min_wick_fraction"])
            or float(row["close"]) < float(row["open"])
        )
    )


def acceptance_candidate(data: pd.DataFrame, index: int, side: str) -> bool:
    if index <= 0:
        return False
    row = data.iloc[index]
    previous = data.iloc[index - 1]
    if not np.isfinite(row["ema200"]) or not np.isfinite(row["atr14"]) or row["atr14"] <= 0:
        return False
    atr = float(row["atr14"])
    ema = float(row["ema200"])
    close_distance = float(RULES["acceptance_close_atr"]) * atr
    common = (
        float(row["body_atr"]) >= float(RULES["acceptance_min_body_atr"])
        and float(row["body_fraction"]) >= float(RULES["acceptance_min_body_fraction"])
    )
    if side == "LONG":
        return (
            common
            and float(previous["close"]) <= float(previous["ema200"]) + 0.04 * float(previous["atr14"])
            and float(row["close"]) >= ema + close_distance
            and float(row["close"]) > float(row["open"])
            and float(row["clv"]) >= float(RULES["acceptance_min_clv"])
        )
    return (
        common
        and float(previous["close"]) >= float(previous["ema200"]) - 0.04 * float(previous["atr14"])
        and float(row["close"]) <= ema - close_distance
        and float(row["close"]) < float(row["open"])
        and float(row["clv"]) <= 1.0 - float(RULES["acceptance_min_clv"])
    )


def pivot_targets(data: pd.DataFrame, index: int, side: str, entry: float, risk: float) -> tuple[Optional[float], Optional[float]]:
    if risk <= 0:
        return None, None
    highs: list[float] = []
    lows: list[float] = []
    left = right = 2
    start = max(left, index - 80)
    stop = max(start, index - right)
    for i in range(start, stop + 1):
        high_window = data["high"].iloc[i - left:i + right + 1]
        low_window = data["low"].iloc[i - left:i + right + 1]
        if len(high_window) == 5 and float(data["high"].iat[i]) >= float(high_window.max()):
            highs.append(float(data["high"].iat[i]))
        if len(low_window) == 5 and float(data["low"].iat[i]) <= float(low_window.min()):
            lows.append(float(data["low"].iat[i]))
    if side == "LONG":
        candidates = sorted(level for level in highs if level > entry + 0.25 * risk)
        target = candidates[0] if candidates else None
        return target, ((target - entry) / risk if target is not None else None)
    candidates = sorted((level for level in lows if level < entry - 0.25 * risk), reverse=True)
    target = candidates[0] if candidates else None
    return target, ((entry - target) / risk if target is not None else None)


def lifecycle(data: pd.DataFrame, start_index: int, side: str, trigger: float, stop: float, base_status: str, expiry: int) -> tuple[str, Optional[int], Optional[int]]:
    triggered = False
    trigger_ts: Optional[int] = None
    invalidation_ts: Optional[int] = None
    for i in range(start_index + 1, len(data)):
        row = data.iloc[i]
        hit_trigger = float(row["high"]) >= trigger if side == "LONG" else float(row["low"]) <= trigger
        hit_stop = float(row["low"]) <= stop if side == "LONG" else float(row["high"]) >= stop
        if not triggered and hit_trigger and hit_stop:
            return "AMBIGUOUS", int(row["ts"]), int(row["ts"])
        if not triggered:
            if hit_stop:
                return "INVALIDATED", None, int(row["ts"])
            if hit_trigger:
                triggered = True
                trigger_ts = int(row["ts"])
        elif hit_stop:
            invalidation_ts = int(row["ts"])
            return "INVALIDATED", trigger_ts, invalidation_ts
    age = len(data) - 1 - start_index
    if triggered:
        return "TRIGGERED", trigger_ts, None
    if age > expiry:
        return "EXPIRED", None, None
    return base_status, None, None


def score_signal(row: pd.Series, signal: Signal) -> tuple[float, dict[str, float], list[str], list[str]]:
    components: dict[str, float] = {"valid_setup": 45.0}
    reasons: list[str] = []
    warnings: list[str] = []

    if signal.status == "CONFIRMED":
        components["confirmation"] = 10.0
        reasons.append("closed-candle acceptance confirmation")
    elif signal.status == "WATCH":
        components["confirmation"] = 8.0
        reasons.append("valid closed-candle rejection awaiting trigger")
    elif signal.status == "FORMING":
        components["confirmation"] = 3.0
        warnings.append("acceptance still needs a hold or retest")
    elif signal.status == "TRIGGERED":
        components["confirmation"] = 4.0

    if signal.setup == "EMA_REJECTION":
        close_quality = float(row["clv"]) if signal.side == "LONG" else 1.0 - float(row["clv"])
        wick = float(row["lower_wick_fraction"]) if signal.side == "LONG" else float(row["upper_wick_fraction"])
        components["close_quality"] = min(10.0, max(0.0, (close_quality - 0.50) / 0.35 * 10.0))
        components["rejection_wick"] = min(7.0, wick / 0.45 * 7.0)
        components["range"] = min(5.0, max(0.0, float(row["range_atr"]) - 0.45) / 1.3 * 5.0)
        reasons.append("EMA sweep/touch closed back on the defended side")
    else:
        components["body_displacement"] = min(8.0, float(row["body_atr"]) / 1.2 * 8.0)
        components["body_quality"] = min(6.0, float(row["body_fraction"]) / 0.85 * 6.0)
        components["ema_displacement"] = min(5.0, abs(float(row["distance_ema_atr"])) / 0.8 * 5.0)
        reasons.append("decisive body displacement through the EMA")

    slope = signal.ema_slope_atr if signal.side == "LONG" else -signal.ema_slope_atr
    if slope >= 0:
        components["ema_slope"] = min(8.0, slope / 0.8 * 8.0)
        if slope > 0.12:
            reasons.append("EMA slope supports direction")
    else:
        components["ema_slope"] = max(-6.0, slope / 0.5 * 6.0)
        warnings.append("EMA slope opposes the setup")

    if signal.touch_count <= 1:
        components["touch_freshness"] = 8.0
        reasons.append("first EMA test episode")
    elif signal.touch_count == 2:
        components["touch_freshness"] = 5.0
        reasons.append("second EMA test episode")
    elif signal.touch_count == 3:
        components["touch_freshness"] = 2.0
    else:
        components["touch_freshness"] = -5.0
        warnings.append("EMA is heavily tested")

    if signal.cross_count <= 1:
        components["cleanliness"] = 6.0
    elif signal.cross_count == 2:
        components["cleanliness"] = 3.0
    elif signal.cross_count >= 4:
        components["cleanliness"] = -6.0
        warnings.append("repeated EMA chop")

    if signal.volume_ratio >= 1.5:
        components["volume_impulse"] = 5.0
        reasons.append("volume expansion")
    elif signal.volume_ratio >= 1.1:
        components["volume_impulse"] = 3.0
    elif signal.volume_ratio < 0.7:
        components["volume_impulse"] = -3.0
        warnings.append("weak signal-candle volume")

    if signal.quote_volume_24h >= 1_000_000_000:
        components["liquidity"] = 4.0
    elif signal.quote_volume_24h >= 100_000_000:
        components["liquidity"] = 3.0
    elif signal.quote_volume_24h >= 20_000_000:
        components["liquidity"] = 2.0

    if signal.structure_target_r is not None:
        if signal.structure_target_r >= 3:
            components["structure_room"] = 5.0
        elif signal.structure_target_r >= 2:
            components["structure_room"] = 3.0
        elif signal.structure_target_r < 1.25:
            components["structure_room"] = -8.0
            warnings.append("less than 1.25R to nearby structure")

    return clamp(sum(components.values())), components, reasons, warnings


def make_signal(data: pd.DataFrame, instrument: Instrument, timeframe: str, side: str, setup: str, signal_index: int, lifecycle_index: int, trigger: float, stop: float, base_status: str) -> Signal:
    row = data.iloc[signal_index]
    trigger = round_tick(trigger, instrument.tick_size, "up" if side == "LONG" else "down")
    stop = round_tick(stop, instrument.tick_size, "down" if side == "LONG" else "up")
    risk = abs(trigger - stop)
    if risk <= 0:
        raise ValueError("non-positive setup risk")
    direction = 1.0 if side == "LONG" else -1.0
    status, trigger_ts, invalidation_ts = lifecycle(
        data, lifecycle_index, side, trigger, stop, base_status, TRIGGER_EXPIRY[timeframe]
    )
    structure_target, structure_r = pivot_targets(data, signal_index, side, trigger, risk)
    current_r = direction * (instrument.last_price - trigger) / risk
    signal = Signal(
        symbol=instrument.symbol,
        timeframe=timeframe,
        side=side,
        setup=setup,
        status=status,
        signal_ts=int(row["ts"]),
        signal_time_utc=str(utc_iso(int(row["ts"]))),
        age_bars=len(data) - 1 - signal_index,
        score=0.0,
        grade="D",
        last_price=instrument.last_price,
        ema200=float(row["ema200"]),
        atr14=float(row["atr14"]),
        trigger=trigger,
        stop=stop,
        target_1r=trigger + direction * risk,
        target_2r=trigger + direction * 2 * risk,
        target_3r=trigger + direction * 3 * risk,
        structure_target=structure_target,
        structure_target_r=structure_r,
        risk_pct=(risk / trigger * 100.0) if trigger else 0.0,
        quote_volume_24h=instrument.quote_volume_24h,
        touch_count=int(row["touch_count"]),
        cross_count=int(row["cross_count"]),
        ema_slope_atr=float(row["ema_slope_atr"]),
        distance_from_ema_atr=float(row["distance_ema_atr"]),
        volume_ratio=float(row["volume_ratio"]),
        trigger_ts=trigger_ts,
        trigger_time_utc=utc_iso(trigger_ts),
        invalidation_ts=invalidation_ts,
        invalidation_time_utc=utc_iso(invalidation_ts),
        current_r=current_r,
        no_chase=(status == "TRIGGERED" and current_r >= float(RULES["no_chase_after_r"])),
        is_recent=(len(data) - 1 - signal_index) <= RECENT_BARS[timeframe],
    )
    score, components, reasons, warnings = score_signal(row, signal)
    signal.score = score
    signal.grade = grade(score)
    signal.score_components = components
    signal.reasons = reasons
    signal.warnings = warnings
    if signal.no_chase:
        signal.warnings.append("DO NOT CHASE: price is already at least 0.75R beyond trigger")
    if signal.status in {"INVALIDATED", "AMBIGUOUS", "EXPIRED"}:
        signal.score = max(0.0, signal.score - 12.0)
        signal.grade = grade(signal.score)
    return signal


def detect(data: pd.DataFrame, instrument: Instrument, timeframe: str) -> tuple[list[Signal], Context]:
    if len(data) < MIN_BARS:
        raise ValueError(f"only {len(data)} completed bars")
    data = prepare(data)
    last = data.iloc[-1]
    if not np.isfinite(last["ema200"]) or not np.isfinite(last["atr14"]):
        raise ValueError("EMA/ATR unavailable")
    if float(last["close"]) > float(last["ema200"]) and float(last["ema_slope_atr"]) >= 0:
        regime = "bull"
    elif float(last["close"]) < float(last["ema200"]) and float(last["ema_slope_atr"]) <= 0:
        regime = "bear"
    else:
        regime = "mixed"
    context = Context(
        symbol=instrument.symbol,
        timeframe=timeframe,
        completed_ts=int(last["ts"]),
        close=float(last["close"]),
        ema200=float(last["ema200"]),
        atr14=float(last["atr14"]),
        ema_slope_atr=float(last["ema_slope_atr"]),
        distance_from_ema_atr=float(last["distance_ema_atr"]),
        touch_count=int(last["touch_count"]),
        cross_count=int(last["cross_count"]),
        regime=regime,
    )
    signals: list[Signal] = []
    start = max(200, len(data) - LIFECYCLE_BARS[timeframe] - 5)

    for i in range(start, len(data)):
        row = data.iloc[i]
        atr = float(row["atr14"])
        ema = float(row["ema200"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        if rejection_candidate(data, i, "LONG"):
            signals.append(make_signal(
                data, instrument, timeframe, "LONG", "EMA_REJECTION", i, i,
                float(row["high"]) + instrument.tick_size,
                min(float(row["low"]), ema - 0.10 * atr) - instrument.tick_size,
                "WATCH",
            ))
        if rejection_candidate(data, i, "SHORT"):
            signals.append(make_signal(
                data, instrument, timeframe, "SHORT", "EMA_REJECTION", i, i,
                float(row["low"]) - instrument.tick_size,
                max(float(row["high"]), ema + 0.10 * atr) + instrument.tick_size,
                "WATCH",
            ))

    failed_suppress: set[tuple[int, str]] = set()
    for i in range(start, len(data)):
        for side in ("LONG", "SHORT"):
            if not acceptance_candidate(data, i, side):
                continue
            confirm_index: Optional[int] = None
            fail_index: Optional[int] = None
            consecutive_holds = 0
            max_j = min(len(data) - 1, i + int(RULES["acceptance_confirmation_bars"]))
            for j in range(i + 1, max_j + 1):
                row = data.iloc[j]
                ema = float(row["ema200"])
                atr = float(row["atr14"])
                close = float(row["close"])
                if side == "LONG":
                    if close <= ema - float(RULES["acceptance_fail_buffer_atr"]) * atr:
                        fail_index = j
                        break
                    held = close >= ema + float(RULES["acceptance_hold_buffer_atr"]) * atr
                    retested = float(row["low"]) <= ema + float(RULES["acceptance_retest_zone_atr"]) * atr and held
                else:
                    if close >= ema + float(RULES["acceptance_fail_buffer_atr"]) * atr:
                        fail_index = j
                        break
                    held = close <= ema - float(RULES["acceptance_hold_buffer_atr"]) * atr
                    retested = float(row["high"]) >= ema - float(RULES["acceptance_retest_zone_atr"]) * atr and held
                consecutive_holds = consecutive_holds + 1 if held else 0
                if retested or consecutive_holds >= int(RULES["acceptance_required_hold_closes"]):
                    confirm_index = j
                    break

            initial = data.iloc[i]
            initial_atr = float(initial["atr14"])
            if fail_index is not None:
                failure = data.iloc[fail_index]
                structure = data.iloc[i:fail_index + 1]
                opposite = "SHORT" if side == "LONG" else "LONG"
                failed_suppress.add((int(failure["ts"]), opposite))
                if opposite == "SHORT":
                    trigger = float(failure["low"]) - instrument.tick_size
                    stop = max(float(structure["high"].max()), float(failure["ema200"]) + 0.10 * float(failure["atr14"])) + instrument.tick_size
                else:
                    trigger = float(failure["high"]) + instrument.tick_size
                    stop = min(float(structure["low"].min()), float(failure["ema200"]) - 0.10 * float(failure["atr14"])) - instrument.tick_size
                signals.append(make_signal(
                    data, instrument, timeframe, opposite, "FAILED_ACCEPTANCE", fail_index, fail_index,
                    trigger, stop, "WATCH"
                ))
                continue

            if confirm_index is not None:
                structure = data.iloc[i:confirm_index + 1]
                if side == "LONG":
                    trigger = float(structure["high"].max()) + instrument.tick_size
                    stop = min(float(structure["low"].min()), float(initial["ema200"]) - 0.10 * initial_atr) - instrument.tick_size
                else:
                    trigger = float(structure["low"].min()) - instrument.tick_size
                    stop = max(float(structure["high"].max()), float(initial["ema200"]) + 0.10 * initial_atr) + instrument.tick_size
                signals.append(make_signal(
                    data, instrument, timeframe, side, "EMA_ACCEPTANCE", i, confirm_index,
                    trigger, stop, "CONFIRMED"
                ))
            else:
                age = len(data) - 1 - i
                if side == "LONG":
                    trigger = float(initial["high"]) + instrument.tick_size
                    stop = min(float(initial["low"]), float(initial["ema200"]) - 0.10 * initial_atr) - instrument.tick_size
                else:
                    trigger = float(initial["low"]) - instrument.tick_size
                    stop = max(float(initial["high"]), float(initial["ema200"]) + 0.10 * initial_atr) + instrument.tick_size
                base = "FORMING" if age <= int(RULES["acceptance_confirmation_bars"]) else "EXPIRED"
                signals.append(make_signal(
                    data, instrument, timeframe, side, "EMA_ACCEPTANCE", i, i,
                    trigger, stop, base
                ))

    if failed_suppress:
        signals = [
            signal for signal in signals
            if not (
                signal.setup == "EMA_ACCEPTANCE"
                and (signal.signal_ts, signal.side) in failed_suppress
            )
        ]

    unique: dict[str, Signal] = {}
    for signal in signals:
        prior = unique.get(signal.key())
        if prior is None or signal.score > prior.score:
            unique[signal.key()] = signal
    return list(unique.values()), context


def select_universe(client: PublicClient) -> list[Instrument]:
    exchange = client.get("/api/v3/exchangeInfo")
    ticker_rows = client.get("/api/v3/ticker/24hr")
    tickers = {str(row.get("symbol")): row for row in ticker_rows if isinstance(row, dict)}
    candidates: list[Instrument] = []
    for row in exchange.get("symbols", []):
        symbol = str(row.get("symbol", ""))
        base = str(row.get("baseAsset", ""))
        quote = str(row.get("quoteAsset", ""))
        if row.get("status") != "TRADING" or quote != "USDT" or not row.get("isSpotTradingAllowed", False):
            continue
        if base in STABLE_BASES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        ticker = tickers.get(symbol, {})
        try:
            quote_volume = float(ticker.get("quoteVolume") or 0.0)
            last_price = float(ticker.get("lastPrice") or 0.0)
        except (TypeError, ValueError):
            continue
        if quote_volume < MIN_QUOTE_VOLUME and symbol not in ALWAYS_INCLUDE:
            continue
        tick_size = 0.0
        step_size = 0.0
        min_qty = 0.0
        for item in row.get("filters", []):
            if item.get("filterType") == "PRICE_FILTER":
                tick_size = float(item.get("tickSize") or 0.0)
            elif item.get("filterType") == "LOT_SIZE":
                step_size = float(item.get("stepSize") or 0.0)
                min_qty = float(item.get("minQty") or 0.0)
        if last_price <= 0 or tick_size <= 0:
            continue
        candidates.append(Instrument(symbol, base, quote, tick_size, step_size, min_qty, quote_volume, last_price))
    candidates.sort(key=lambda item: item.quote_volume_24h, reverse=True)
    selected = candidates[:MAX_SYMBOLS]
    selected_symbols = {item.symbol for item in selected}
    for item in candidates:
        if item.symbol in ALWAYS_INCLUDE and item.symbol not in selected_symbols:
            selected.append(item)
            selected_symbols.add(item.symbol)
    return selected


def apply_mtf(signals: list[Signal], contexts: dict[tuple[str, str], Context]) -> None:
    for signal in signals:
        other_tf = "1D" if signal.timeframe == "4H" else "4H"
        other = contexts.get((signal.symbol, other_tf))
        if other is None:
            continue
        desired = "bull" if signal.side == "LONG" else "bear"
        if other.regime == desired:
            signal.score = clamp(signal.score + 5.0)
            signal.score_components["mtf_alignment"] = 5.0
            signal.reasons.append(f"{other_tf} regime aligns")
        elif other.regime not in {"mixed", desired}:
            signal.score = clamp(signal.score - 5.0)
            signal.score_components["mtf_alignment"] = -5.0
            signal.warnings.append(f"{other_tf} regime conflicts")
        signal.grade = grade(signal.score)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"signals": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"signals": {}}
    except Exception:
        return {"signals": {}}


def mark_changes(signals: list[Signal], previous: dict[str, Any]) -> list[Signal]:
    prior_map = previous.get("signals", {}) if isinstance(previous.get("signals", {}), dict) else {}
    changed: list[Signal] = []
    for signal in signals:
        prior = prior_map.get(signal.key())
        if prior is None:
            signal.state_change = "NEW"
        elif prior.get("status") != signal.status:
            signal.state_change = f"{prior.get('status', 'UNKNOWN')}→{signal.status}"
        elif prior.get("grade") != signal.grade:
            signal.state_change = f"GRADE {prior.get('grade', '?')}→{signal.grade}"
        elif abs(float(prior.get("score", 0.0)) - signal.score) >= 8:
            signal.state_change = f"SCORE {float(prior.get('score', 0.0)):.0f}→{signal.score:.0f}"
        if signal.state_change:
            changed.append(signal)
    return changed


def fmt_price(value: Optional[float]) -> str:
    if value is None:
        return "—"
    value = float(value)
    absolute = abs(value)
    if absolute >= 1000:
        return f"{value:,.2f}"
    if absolute >= 10:
        return f"{value:.3f}"
    if absolute >= 1:
        return f"{value:.4f}"
    if absolute >= 0.01:
        return f"{value:.5f}"
    return f"{value:.8f}"


def fmt_volume(value: float) -> str:
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def status_rank(status: str) -> int:
    return {"CONFIRMED": 0, "WATCH": 1, "FORMING": 2, "TRIGGERED": 3, "INVALIDATED": 4, "AMBIGUOUS": 5, "EXPIRED": 6}.get(status, 99)


def sorted_signals(signals: list[Signal]) -> list[Signal]:
    return sorted(signals, key=lambda s: (status_rank(s.status), {"A": 0, "B": 1, "C": 2, "D": 3}.get(s.grade, 9), -s.score, 0 if s.timeframe == "1D" else 1, s.symbol))


def table(signals: list[Signal], limit: int = 40) -> str:
    if not signals:
        return "_None._\n"
    lines = [
        "| Pair | Side | TF | Setup | Status | Grade | Score | Last | EMA 200 | Trigger | Stop | 1R | 2R | 3R | Structure |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in signals[:limit]:
        status = s.status + (" / NO CHASE" if s.no_chase else "")
        structure = fmt_price(s.structure_target)
        if s.structure_target_r is not None:
            structure += f" ({s.structure_target_r:.1f}R)"
        lines.append(
            f"| {s.symbol} | {s.side} | {s.timeframe} | {s.setup.replace('_', ' ').title()} | {status} | {s.grade} | {s.score:.0f} | {fmt_price(s.last_price)} | {fmt_price(s.ema200)} | {fmt_price(s.trigger)} | {fmt_price(s.stop)} | {fmt_price(s.target_1r)} | {fmt_price(s.target_2r)} | {fmt_price(s.target_3r)} | {structure} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(signals: list[Signal], changed: list[Signal], contexts: dict[tuple[str, str], Context], errors: list[dict[str, str]], selected_count: int) -> None:
    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    all_sorted = sorted_signals(signals)
    qualified = [s for s in all_sorted if s.score >= 55 and s.status not in {"INVALIDATED", "AMBIGUOUS", "EXPIRED"}]
    actionable = [s for s in qualified if s.status in {"WATCH", "CONFIRMED"} and s.grade in {"A", "B"} and not s.no_chase]
    forming = [s for s in qualified if s.status == "FORMING" and s.grade in {"A", "B", "C"}]
    triggered = [s for s in qualified if s.status == "TRIGGERED" and s.is_recent]
    changed_sorted = sorted_signals(changed)
    top_three = actionable[:3]
    now = datetime.now(timezone.utc)
    scanned_symbols = len({symbol for symbol, _ in contexts})
    latest_4h = max((ctx.completed_ts for (_, tf), ctx in contexts.items() if tf == "4H"), default=None)
    latest_1d = max((ctx.completed_ts for (_, tf), ctx in contexts.items() if tf == "1D"), default=None)

    markdown = [
        "# EMA 200 Crypto Daily Scan",
        "",
        f"Generated: **{now.isoformat(timespec='seconds')}**",
        f"Source: **Binance official spot market-data API** · universe selected **{selected_count}** · symbols successfully scanned **{scanned_symbols}** · completed 4H/1D datasets **{len(contexts)}** · data errors **{len(errors)}**",
        f"Latest completed 4H candle opened: **{utc_iso(latest_4h) if latest_4h else '—'}** · latest completed daily candle opened: **{utc_iso(latest_1d) if latest_1d else '—'}**",
        "",
        "> Binance spot is being used as the cross-exchange reference feed. EMA structure should closely match BloFin, but verify the exact trigger/stop on BloFin before placing an order.",
        "",
        "## Bottom Line",
        "",
    ]
    if actionable:
        markdown.append(f"**{len(actionable)} actionable A/B setups are currently waiting on their trigger.**")
    else:
        markdown.append("**No A- or B-grade fresh entry qualified. Standards were not lowered.**")
    markdown.extend(["", "## Top Three Opportunities", "", table(top_three, 3), "## Actionable Now", "", table(actionable), "## Forming / Needs Confirmation", "", table(forming), "## Already Triggered / Do Not Chase Review", "", table(triggered), "## Changed Since Prior Run", "", table(changed_sorted)])

    if top_three:
        markdown.extend(["## Professional Read", ""])
        for rank, s in enumerate(top_three, 1):
            reason_text = "; ".join(s.reasons[:5]) or "valid EMA interaction"
            warning_text = "; ".join(s.warnings[:4])
            markdown.append(f"### {rank}. {s.symbol} · {s.timeframe} · {s.side} · {s.setup.replace('_', ' ').title()} · {s.grade}{s.score:.0f}")
            markdown.append("")
            markdown.append(f"Plan: trigger `{fmt_price(s.trigger)}`, stop `{fmt_price(s.stop)}`, 1R/2R/3R `{fmt_price(s.target_1r)}` / `{fmt_price(s.target_2r)}` / `{fmt_price(s.target_3r)}`. Current price `{fmt_price(s.last_price)}`; EMA 200 `{fmt_price(s.ema200)}`.")
            markdown.append("")
            markdown.append(f"Why it qualifies: {reason_text}.")
            if warning_text:
                markdown.append("")
                markdown.append(f"Risk flags: {warning_text}.")
            markdown.append("")

    if errors:
        markdown.extend(["## Data Errors", ""])
        for error in errors[:30]:
            markdown.append(f"- `{error.get('symbol', '?')} {error.get('timeframe', '')}` — {error.get('error', 'unknown error')}")
        markdown.append("")

    markdown.extend([
        "## Status Key",
        "",
        "- **WATCH:** valid rejection plan; entry trigger has not traded.",
        "- **FORMING:** initial acceptance exists but still needs a completed hold/retest.",
        "- **CONFIRMED:** acceptance held on completed candles and is waiting on its continuation trigger.",
        "- **TRIGGERED:** planned trigger traded. A NO CHASE warning appears once price is at least 0.75R extended.",
        "- **FAILED ACCEPTANCE:** the attempted acceptance closed decisively back through the EMA and created an opposite reversal watch.",
        "",
        "This is a rule-based market scan, not a guarantee of outcome. No order is submitted by this workflow.",
        "",
    ])

    report_text = "\n".join(markdown)
    (out / "latest_report.md").write_text(report_text, encoding="utf-8")
    stamp = now.strftime("%Y-%m-%d")
    (out / f"ema200_scan_{stamp}.md").write_text(report_text, encoding="utf-8")

    rows = []
    for s in all_sorted:
        payload = asdict(s)
        payload["reasons"] = "; ".join(s.reasons)
        payload["warnings"] = "; ".join(s.warnings)
        rows.append(payload)
    pd.DataFrame(rows).to_csv(out / "latest_signals.csv", index=False)
    json_payload = {
        "metadata": {
            "generated_utc": now.isoformat(timespec="seconds"),
            "source": "Binance official spot market-data API",
            "selected_symbols": selected_count,
            "scanned_symbols": scanned_symbols,
            "completed_datasets": len(contexts),
            "errors": len(errors),
        },
        "contexts": {f"{symbol}|{tf}": asdict(ctx) for (symbol, tf), ctx in contexts.items()},
        "signals": [asdict(s) for s in all_sorted],
        "errors": errors,
    }
    (out / "latest_scan.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    state = {
        "last_run_utc": now.isoformat(timespec="seconds"),
        "signals": {s.key(): asdict(s) for s in all_sorted},
    }
    (out / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def scan_one(client: PublicClient, instrument: Instrument) -> tuple[list[Signal], dict[tuple[str, str], Context], list[dict[str, str]]]:
    signals: list[Signal] = []
    contexts: dict[tuple[str, str], Context] = {}
    errors: list[dict[str, str]] = []
    for timeframe, interval in TIMEFRAMES.items():
        try:
            frame = client.candles(instrument.symbol, interval)
            found, context = detect(frame, instrument, timeframe)
            signals.extend(found)
            contexts[(instrument.symbol, timeframe)] = context
        except Exception as exc:
            errors.append({"symbol": instrument.symbol, "timeframe": timeframe, "error": str(exc)})
    return signals, contexts, errors


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = PublicClient()
    state_path = Path("reports/state.json")
    previous_state = load_state(state_path)
    selected = select_universe(client)
    LOG.info("Selected %d liquid USDT pairs", len(selected))
    all_signals: list[Signal] = []
    contexts: dict[tuple[str, str], Context] = {}
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(scan_one, client, instrument): instrument.symbol for instrument in selected}
        for number, future in enumerate(as_completed(future_map), 1):
            symbol = future_map[future]
            try:
                signals, local_contexts, local_errors = future.result()
                all_signals.extend(signals)
                contexts.update(local_contexts)
                errors.extend(local_errors)
            except Exception as exc:
                errors.append({"symbol": symbol, "timeframe": "ALL", "error": str(exc)})
            if number % 25 == 0 or number == len(selected):
                LOG.info("Scanned %d/%d", number, len(selected))
    if not contexts:
        raise RuntimeError("No usable market data was returned; report/state not updated")
    apply_mtf(all_signals, contexts)
    changed = mark_changes(all_signals, previous_state)
    write_outputs(all_signals, changed, contexts, errors, len(selected))
    LOG.info("Finished with %d signals and %d data errors", len(all_signals), len(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

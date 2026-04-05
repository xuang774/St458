from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt

@dataclass
class State:
    symbols: list[str]            # fixed symbol order
    lagged_price: np.ndarray      # shape (LOOKBACK, NUM_SYMBOLS), row 0 = most recent
    wealth: float                 # internal wealth tracker (matches R example logic)
    positions: np.ndarray         # current allocations, shape (NUM_SYMBOLS,)





# Pair trading strategy state initialization
def initialise_state(data):
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Selected usable pairs detected from the TRAIN split only
    # (2010-2011 for the 2012-2013 test)
    selected_pairs = [
    # strict 5% pairs
    ("SYDR", "WQX", 0.893523),
    ("JBDX", "JZKP", 0.861499),
    ("CDM", "HJC", 1.388441),
    ("KRV", "WYLR", 1.481454),
    ]

    symbols = sorted(data["symbol"].unique().tolist())

    # Keep full close-price history by symbol
    close_hist = {
        sym: data.loc[data["symbol"] == sym, ["date", "close"]]
            .set_index("date")["close"]
            .sort_index()
            .copy()
        for sym in symbols
    }

    state = {
        "symbols": symbols,
        "selected_pairs": selected_pairs,
        "close_hist": close_hist,
        "current_weights": {sym: 0.0 for sym in symbols},

        # Trading rule parameters
        "lookback": 100,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "pair_gross": 0.20,

        # +1 = long spread, -1 = short spread, 0 = flat
        "pair_position": {f"{y}_{x}": 0 for y, x, _ in selected_pairs}
    }

    return state



# Pair trading algorithm
def trading_algorithm(new_data, state):
    new_data = new_data.copy()
    new_data["date"] = pd.to_datetime(new_data["date"])
    new_data = new_data.sort_values("symbol").reset_index(drop=True)

    current_date = new_data["date"].iloc[0]

    # Update close history with today's close
    price_today = {}
    for _, row in new_data.iterrows():
        sym = row["symbol"]
        px = float(row["close"])
        price_today[sym] = px
        state["close_hist"][sym].loc[current_date] = px

    desired_weights = {sym: 0.0 for sym in state["symbols"]}

    lookback = state["lookback"]
    entry_z = state["entry_z"]
    exit_z = state["exit_z"]
    pair_gross = state["pair_gross"]

    for y_sym, x_sym, beta in state["selected_pairs"]:
        pair_key = f"{y_sym}_{x_sym}"

        y_hist = state["close_hist"][y_sym].sort_index().dropna()
        x_hist = state["close_hist"][x_sym].sort_index().dropna()

        pair_df = pd.concat([y_hist, x_hist], axis=1, join="inner")
        pair_df.columns = ["y", "x"]

        # Need enough history
        if len(pair_df) < lookback:
            continue

        # Log-price spread: U_t = log(Y_t) - beta * log(X_t)
        spread = np.log(pair_df["y"]) - beta * np.log(pair_df["x"])

        roll_mean = spread.rolling(lookback).mean()
        roll_std = spread.rolling(lookback).std()

        spread_t = spread.iloc[-1]
        mu_t = roll_mean.iloc[-1]
        sd_t = roll_std.iloc[-1]

        if pd.isna(mu_t) or pd.isna(sd_t) or sd_t <= 1e-12:
            continue

        z_t = (spread_t - mu_t) / sd_t

        # Price-level hedge ratio from the note:
        # dX2 = beta * X2/X1 * dX1
        # so hedge ratio in units is beta * Y/X if spread = log(Y) - beta log(X)
        y_px = price_today[y_sym]
        x_px = price_today[x_sym]
        hedge_units = beta * (y_px / x_px)

        # Convert to dollar weights.
        # We normalize so total gross pair exposure is approximately pair_gross.
        # One "bundle" is:
        #   +1 unit Y and -hedge_units unit X   (long spread)
        # or
        #   -1 unit Y and +hedge_units unit X   (short spread)
        gross_bundle = 1.0 + abs(hedge_units)
        if gross_bundle <= 1e-12:
            continue

        scale = pair_gross / gross_bundle

        current_pos = state["pair_position"][pair_key]
        target_pos = current_pos

        # Signal logic from the note:
        # spread high => short spread
        # spread low  => long spread
        # close when spread reverts near mean
        if current_pos == 0:
            if z_t > entry_z:
                target_pos = -1   # short spread
            elif z_t < -entry_z:
                target_pos = 1    # long spread
        else:
            if abs(z_t) < exit_z:
                target_pos = 0

        state["pair_position"][pair_key] = target_pos

        if target_pos == 1:
            # Long spread: +Y, -hedged X
            desired_weights[y_sym] += scale
            desired_weights[x_sym] += -scale * hedge_units

        elif target_pos == -1:
            # Short spread: -Y, +hedged X
            desired_weights[y_sym] += -scale
            desired_weights[x_sym] += scale * hedge_units

    # Optional safety cap on total gross exposure
    gross = sum(abs(w) for w in desired_weights.values())
    max_gross = 0.25
    if gross > max_gross and gross > 1e-12:
        shrink = max_gross / gross
        desired_weights = {k: v * shrink for k, v in desired_weights.items()}

    # Trades = desired weights - current weights
    trades = {
        sym: desired_weights[sym] - state["current_weights"].get(sym, 0.0)
        for sym in state["symbols"]
    }

    state["current_weights"] = desired_weights

    # Return as named vector-like object
    trades = pd.Series(trades)
    return trades, state
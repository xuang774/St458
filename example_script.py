#!/usr/bin/env python3
"""
example_strategy.py

Direct translation of example_script.R:
- Maintain a lookback-window of closes (lagged_price)
- Compute 10-day momentum sign(close_t - close_{t-10})
- Target position per symbol = (wealth / num_symbols) * momentum_sign
- Trades = target_position - current_position

This matches walk_forward.py’s interface:
  initialise_state(df_train) -> state
  trading_algorithm(new_data, state) -> (trades, new_state)

Trades are in "wealth units" (dollar allocation), same as the R version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


NUM_SYMBOLS = 100
LOOKBACK = 10  # 10-day momentum


@dataclass
class State:
    symbols: list[str]            # fixed symbol order
    lagged_price: np.ndarray      # shape (LOOKBACK, NUM_SYMBOLS), row 0 = most recent
    wealth: float                 # internal wealth tracker (matches R example logic)
    positions: np.ndarray         # current allocations, shape (NUM_SYMBOLS,)


def initialise_state(data: pd.DataFrame) -> State:
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    dates = np.sort(df["date"].unique())
    symbols = sorted(df["symbol"].unique().tolist())

    if len(symbols) != NUM_SYMBOLS:
        # Not strictly required, but keeps behavior aligned with the R template
        raise ValueError(f"Expected {NUM_SYMBOLS} symbols, got {len(symbols)}")

    # lagged_price[0] will become the next "new_data" close on first call.
    lagged_price = np.full((LOOKBACK, NUM_SYMBOLS), np.nan, dtype=float)

    # Fill with the last LOOKBACK closes from training, newest at row 0
    # R code uses: date <- dates[length(dates) - i + 1], i=1..LOOKBACK
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    for i in range(LOOKBACK):
        dt = dates[len(dates) - 1 - i]  # most recent, then backwards
        sub = df.loc[df["date"] == dt, ["symbol", "close"]]
        for _, row in sub.iterrows():
            lagged_price[i, sym_to_idx[row["symbol"]]] = float(row["close"])

    positions = np.zeros(NUM_SYMBOLS, dtype=float)
    return State(symbols=symbols, lagged_price=lagged_price, wealth=1.0, positions=positions)


def trading_algorithm(new_data: pd.DataFrame, state: State) -> Tuple[np.ndarray, State]:
    """
    new_data: dataframe for a single date, already symbol-sorted by walk_forward.py wrapper.
    Returns:
      trades: np.ndarray shape (NUM_SYMBOLS,)
      new_state: updated State
    """
    # Update lagged price matrix: shift down, insert today's close at row 0
    state.lagged_price[1:LOOKBACK, :] = state.lagged_price[0:LOOKBACK - 1, :]

    # Map today's closes into state.symbols order
    # walk_forward.py ensures consistent ordering, but we enforce it anyway
    new_data = new_data.copy()
    # If new_data is already in correct order, this is a no-op
    new_data["symbol"] = new_data["symbol"].astype(str)
    sym_rank = {s: i for i, s in enumerate(state.symbols)}
    new_data["_r"] = new_data["symbol"].map(sym_rank)
    new_data = new_data.sort_values("_r").drop(columns=["_r"]).reset_index(drop=True)

    closes_today = new_data["close"].to_numpy(dtype=float)
    if closes_today.shape[0] != NUM_SYMBOLS:
        raise ValueError(f"Expected {NUM_SYMBOLS} rows in new_data, got {closes_today.shape[0]}")

    state.lagged_price[0, :] = closes_today

    # Update wealth and positions to reflect 1-day PnL (same as R example)
    r1d = state.lagged_price[0, :] / state.lagged_price[1, :] - 1.0
    r1d = np.where(np.isfinite(r1d), r1d, 0.0)  # safety if any NA slipped through

    state.wealth = float(state.wealth + np.sum(state.positions * r1d))
    
    
    
    ########## 想想真的要调整这一步吗
    state.positions = state.positions * (1.0 + r1d)
    ############
    
    
    '''# Compute 10-day momentum sign(close_t - close_{t-10})
    momentum = np.sign(state.lagged_price[0, :] - state.lagged_price[LOOKBACK - 1, :])
    momentum = np.where(np.isfinite(momentum), momentum, 0.0)

    # Target new positions (wealth scaled, equal across symbols)
    new_positions = (state.wealth / NUM_SYMBOLS) * momentum'''

    # Trades are adjustments from current positions
    trades = new_positions - state.positions

    # Update state: store target positions (as in R code)
    state.positions = new_positions

    return trades.astype(float), state
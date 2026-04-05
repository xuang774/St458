import numpy as np
import pandas as pd


def initialise_state(data: pd.DataFrame):
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = df["symbol"].astype(str)

    symbols = sorted(df["symbol"].unique().tolist())
    n = len(symbols)

    # Need 120 closes to compute factor9:
    # factor9 = min over past 60 observations of 60-day returns
    lookback = 120
    dates = np.sort(df["date"].unique())
    if len(dates) < lookback:
        raise ValueError(f"Need at least {lookback} training dates, got {len(dates)}")

    close_hist = np.full((lookback, n), np.nan, dtype=float)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    # row 0 = most recent training close
    for i in range(lookback):
        dt = dates[len(dates) - 1 - i]
        sub = df.loc[df["date"] == dt, ["symbol", "close"]]
        for _, row in sub.iterrows():
            close_hist[i, sym_to_idx[row["symbol"]]] = float(row["close"])

    state = {
        "symbols": symbols,
        "positions": np.zeros(n, dtype=float),   # dollar positions
        "wealth": 1.0,
        "prev_close": close_hist[0].copy(),
        "close_hist": close_hist,
        "day_idx": 0,
        "hold_days": 5,
        "long_frac": 0.2,
        "short_frac": 0.2,
        "factor_signs": {
            "vol_skew_60d": -1,
            "factor9": -1,
            "factor10": -1,
            "mom_60d_mean": -1,
            "mom_5d_mean": 1,
        },
    }
    return state


def trading_algorithm(new_data: pd.DataFrame, state):
    symbols = state["symbols"]
    positions = state["positions"]
    wealth = state["wealth"]
    prev_close = state["prev_close"]
    close_hist = state["close_hist"]
    day_idx = state["day_idx"]
    hold_days = state["hold_days"]
    long_frac = state["long_frac"]
    short_frac = state["short_frac"]
    factor_signs = state["factor_signs"]

    # Align today's rows to fixed symbol order
    x = new_data.copy()
    x["symbol"] = x["symbol"].astype(str)
    x = x.set_index("symbol").reindex(symbols).reset_index()

    closes_today = x["close"].to_numpy(dtype=float)
    if np.isnan(closes_today).any():
        missing = x.loc[x["close"].isna(), "symbol"].tolist()
        raise ValueError(f"Missing close values for symbols: {missing[:10]}")

    # 1) Let yesterday's holdings earn today's close-to-close return
    if prev_close is not None:
        r1d = closes_today / prev_close - 1.0
        r1d = np.where(np.isfinite(r1d), r1d, 0.0)
        wealth = float(wealth + np.sum(positions * r1d))
        positions = positions * (1.0 + r1d)

    # 2) After today's close, add today's close into history
    #    so today's factor values can set tomorrow's portfolio
    close_hist[1:, :] = close_hist[:-1, :]
    close_hist[0, :] = closes_today

    # 3) Rebalance every hold_days, starting on the first test day
    if day_idx % hold_days == 0:
        # Last 60 one-day returns, now INCLUDING today's close
        ret_1d_60 = close_hist[:60, :] / close_hist[1:61, :] - 1.0

        mom_5d_mean = np.nanmean(ret_1d_60[:5, :], axis=0)
        mom_60d_mean = np.nanmean(ret_1d_60, axis=0)

        mu = np.nanmean(ret_1d_60, axis=0)
        sigma = np.nanstd(ret_1d_60, axis=0)
        sigma = np.where(sigma > 0, sigma, np.nan)
        vol_skew_60d = np.nanmean(((ret_1d_60 - mu) / sigma) ** 3, axis=0)

        # 60 observations of 30-day returns, ending no later than today
        ret_30_hist = close_hist[:60, :] / close_hist[30:90, :] - 1.0
        factor10 = np.nanmin(ret_30_hist, axis=0)

        # 60 observations of 60-day returns, ending no later than today
        ret_60_hist = close_hist[:60, :] / close_hist[60:120, :] - 1.0
        factor9 = np.nanmin(ret_60_hist, axis=0)

        factor_df = pd.DataFrame({
            "symbol": symbols,
            "vol_skew_60d": vol_skew_60d,
            "factor9": factor9,
            "factor10": factor10,
            "mom_60d_mean": mom_60d_mean,
            "mom_5d_mean": mom_5d_mean,
        })

        # Composite rank logic
        score_cols = []
        for col, sign in factor_signs.items():
            raw_rank = factor_df[col].rank(pct=True)
            score_col = f"{col}_score"

            if sign > 0:
                factor_df[score_col] = raw_rank
            else:
                factor_df[score_col] = 1.0 - raw_rank

            score_cols.append(score_col)

        factor_df["composite_score"] = factor_df[score_cols].mean(axis=1)

        # Long-short bucket logic
        y = factor_df[["symbol", "composite_score"]].dropna().copy()
        target_weights = pd.Series(0.0, index=symbols, dtype=float)

        n_valid = len(y)
        if n_valid >= 2 and wealth > 0:
            n_long = max(1, int(np.floor(n_valid * long_frac)))
            n_short = max(1, int(np.floor(n_valid * short_frac)))

            y = y.sort_values("composite_score", ascending=False).reset_index(drop=True)
            long_symbols = y.iloc[:n_long]["symbol"]
            short_symbols = y.iloc[-n_short:]["symbol"]

            target_weights.loc[long_symbols] = 0.5 / n_long
            target_weights.loc[short_symbols] = -0.5 / n_short

        target_positions = target_weights.to_numpy(dtype=float) * wealth
        trades = target_positions - positions
        positions = target_positions
    else:
        trades = np.zeros_like(positions)

    state["positions"] = positions
    state["wealth"] = wealth
    state["prev_close"] = closes_today
    state["close_hist"] = close_hist
    state["day_idx"] = day_idx + 1

    return trades.astype(float), state
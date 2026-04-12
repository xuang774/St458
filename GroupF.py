#!/usr/bin/env python3
"""
V3 Trading Strategy — Ridge Regression + Constrained Mean-Variance Optimisation

Architecture (two layers):
  Layer 1 (Signal):     Ridge regression combines alpha factors into a composite
                        expected-return estimate mu_hat for each ETF.
  Layer 2 (Portfolio):  Gross-exposure constrained mean-variance optimisation
                        translates mu_hat into dollar-neutral, l1-bounded weights.

Interface matches walk_forward.py:
  initialise_state(df_train) -> state
  trading_algorithm(new_data, state) -> (trades, new_state)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# congigureation parameters (tuned via time-series CV on training data)
HOLD_DAYS     = 21      # rebalance frequency (trading days)
SCALE         = 3.0     # leverage multiplier applied to optimiser output
RISK_AVERSION = 0.5     # risk aversion parameter 'a' in the mean-variance objective
GROSS_EXP     = 2.0     # l1 gross exposure constraint 'c'
MAX_POS       = 0.10    # max |weight| per asset before scaling
COV_WINDOW    = 126     # rolling window for covariance estimation (trading days)
RIDGE_ALPHA   = 10.0    # Ridge regularisation strength (tuned via time-series CV)
DD_PROTECT    = True    # enable drawdown protection
DD_THRESH_1   = 0.05    # first drawdown threshold (halve exposure)
DD_THRESH_2   = 0.10    # second drawdown threshold (quarter exposure)
NUM_SYMBOLS   = 100
LOOKBACK      = 131     # days of history to retain (>= max(COV_WINDOW, 126+5))

# Factor definitions and windows (tuned via IC-IR screening on training data)
# All 36 factors that pass |IC-IR| >= 0.05 screening on training data.
# Listed here for documentation; computed dynamically from rolling price buffer.

MOMENTUM_WINDOWS   = [5, 10, 20, 30, 42, 60, 126]
VOL_WINDOWS        = [5, 10, 20, 60]
ZSCORE_WINDOWS     = [21, 42, 63]
IDIO_MOM_WINDOWS   = [30, 42, 60]


# state dataclass to hold all necessary information across trading days
@dataclass
class State:
    symbols: list               # sorted symbol list (length NUM_SYMBOLS)
    # Rolling price buffer: DataFrames indexed by date
    close_buf: pd.DataFrame     # shape (<=LOOKBACK, NUM_SYMBOLS)
    high_buf: pd.DataFrame
    low_buf: pd.DataFrame
    vol_buf: pd.DataFrame
    # Ridge model (stored as raw arrays — no sklearn needed at runtime)
    ridge_coefs: np.ndarray     # shape (n_features,)
    ridge_intercept: float
    scaler_mean: np.ndarray     # shape (n_features,)
    scaler_scale: np.ndarray    # shape (n_features,)
    feature_names: list         # ordered feature names matching coefs
    # Position tracking
    positions: np.ndarray       # current positions in dollar terms
    wealth: float               # internal wealth estimate
    peak_wealth: float          # for drawdown calculation
    day_count: int              # days since last rebalance
    total_days: int             # total trading days elapsed


# feature engineering function: computes all features for the most recent date in the buffer
def compute_features_from_buffer(close_buf, high_buf, low_buf, vol_buf, symbols):
    """
    Compute all alpha factors for the most recent date in the buffer.
    Returns a dict: {factor_name: np.array of length NUM_SYMBOLS}.
    """
    n = len(symbols)
    features = {}

    # Returns matrix (needed for vol factors)
    ret_matrix = close_buf.pct_change()

    # Current (most recent) close
    current_close = close_buf.iloc[-1].values

    # ── Momentum factors ──
    for w in MOMENTUM_WINDOWS:
        if len(close_buf) > w:
            past_close = close_buf.iloc[-(w+1)].values
            features[f'mom_{w}d'] = current_close / past_close - 1.0
        else:
            features[f'mom_{w}d'] = np.full(n, np.nan)

    # Arithmetic mean return over window
    for w in [5, 10, 20, 60]:
        if len(ret_matrix) >= w:
            features[f'mom_{w}d_mean'] = ret_matrix.iloc[-w:].mean().values
        else:
            features[f'mom_{w}d_mean'] = np.full(n, np.nan)

    # Volatility factors (std, skew, kurtosis) over multiple windows
    for w in VOL_WINDOWS:
        if len(ret_matrix) >= w:
            chunk = ret_matrix.iloc[-w:]
            features[f'vol_std_{w}d']  = chunk.std().values
            features[f'vol_skew_{w}d'] = chunk.skew().values
            features[f'vol_kurt_{w}d'] = chunk.kurt().values
        else:
            features[f'vol_std_{w}d']  = np.full(n, np.nan)
            features[f'vol_skew_{w}d'] = np.full(n, np.nan)
            features[f'vol_kurt_{w}d'] = np.full(n, np.nan)

    # custom factor1: rolling 60d high/low range 
    if len(high_buf) >= 60 and len(low_buf) >= 60:
        max_high = high_buf.iloc[-60:].max().values
        min_low  = low_buf.iloc[-60:].min().values
        with np.errstate(divide='ignore', invalid='ignore'):
            features['factor1'] = max_high / min_low - 1.0
    else:
        features['factor1'] = np.full(n, np.nan)

    # custom factor4: skewness of 5d returns over 60d window 
    if len(close_buf) > 65:
        r5 = close_buf / close_buf.shift(5) - 1.0
        if len(r5.dropna()) >= 20:
            features['factor4'] = r5.iloc[-60:].skew().values
        else:
            features['factor4'] = np.full(n, np.nan)
    else:
        features['factor4'] = np.full(n, np.nan)

    # factor4_10d: skewness of positive 10d returns over 60d window 
    if len(close_buf) > 70:
        r10 = close_buf / close_buf.shift(10) - 1.0
        r10_pos = r10.where(r10 > 0)
        chunk = r10_pos.iloc[-60:]
        # skew only where enough data
        sk = chunk.apply(lambda s: s.dropna().skew() if s.dropna().shape[0] >= 10 else np.nan)
        features['factor4_10d'] = sk.values
    else:
        features['factor4_10d'] = np.full(n, np.nan)

    # custom factor5: price-volume correlation
    for wname, w in [('factor5_30', 30), ('factor5_10', 10)]:
        if len(close_buf) >= w + 1 and len(vol_buf) >= w:
            corrs = np.full(n, np.nan)
            lagged_close = close_buf.iloc[-(w+1):-1].values  # shape (w, n)
            volume_chunk = vol_buf.iloc[-w:].values           # shape (w, n)
            for j in range(n):
                lc = lagged_close[:, j]
                vc = volume_chunk[:, j]
                valid = np.isfinite(lc) & np.isfinite(vc)
                if valid.sum() >= 5:
                    corrs[j] = np.corrcoef(lc[valid], vc[valid])[0, 1]
            features[wname] = corrs
        else:
            features[wname] = np.full(n, np.nan)

    # factor7/8/9/10: rolling min of multi-horizon returns
    for fname, p in [('factor7', 7), ('factor8', 20), ('factor10', 30), ('factor9', 60)]:
        if len(close_buf) > p + 60:
            rk = close_buf / close_buf.shift(p) - 1.0
            features[fname] = rk.iloc[-60:].min().values
        else:
            features[fname] = np.full(n, np.nan)

    # Z-score factors 
    if len(close_buf) >= 2:
        log_price = np.log(close_buf)
        for w in ZSCORE_WINDOWS:
            if len(log_price) >= w:
                ma = log_price.iloc[-w:].mean().values
                sd = log_price.iloc[-w:].std().values
                with np.errstate(divide='ignore', invalid='ignore'):
                    features[f'zscore_{w}d'] = (np.log(current_close) - ma) / sd
            else:
                features[f'zscore_{w}d'] = np.full(n, np.nan)

    # Idiosyncratic momentum 
    for w in IDIO_MOM_WINDOWS:
        key = f'mom_{w}d'
        if key in features and np.any(np.isfinite(features[key])):
            xs_mean = np.nanmean(features[key])
            features[f'idio_mom_{w}d'] = features[key] - xs_mean
        else:
            features[f'idio_mom_{w}d'] = np.full(n, np.nan)

    return features



# PORTFOLIO OPTIMISATION (with scipy fallback if there is no CVXPY dependency installed)
def solve_portfolio_scipy(mu_hat, Sigma_hat, a=RISK_AVERSION, c=GROSS_EXP,
                          max_pos=MAX_POS, scale=SCALE):
    """
    Solve:  min  -a * w'mu + (a^2/2) * w'Sigma*w
            s.t. sum(w) = 0,  sum(|w|) <= c,  |w_i| <= max_pos

    Uses variable splitting: w = u - v, u,v >= 0, |w_i| = u_i + v_i.
    This converts the L1 constraint into a linear constraint on (u, v).
    """
    p = len(mu_hat)
    Sigma_reg = Sigma_hat + 1e-6 * np.eye(p)

    def objective(x):
        u, v = x[:p], x[p:]
        w = u - v
        return -a * mu_hat @ w + (a**2 / 2) * w @ Sigma_reg @ w

    def grad(x):
        u, v = x[:p], x[p:]
        w = u - v
        dw = -a * mu_hat + a**2 * Sigma_reg @ w
        return np.concatenate([dw, -dw])

    # Initial point: small equal long-short
    x0 = np.zeros(2 * p)

    # Bounds: u >= 0, v >= 0, each <= max_pos
    bounds = [(0, max_pos)] * p + [(0, max_pos)] * p

    # Constraints
    constraints = [
        # Dollar neutral: sum(u) - sum(v) = 0
        {'type': 'eq', 'fun': lambda x: np.sum(x[:p]) - np.sum(x[p:])},
        # Gross exposure: sum(u) + sum(v) <= c
        {'type': 'ineq', 'fun': lambda x: c - np.sum(x[:p]) - np.sum(x[p:])},
    ]

    result = minimize(objective, x0, jac=grad, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 500, 'ftol': 1e-10})

    if result.success or result.fun < objective(x0):
        u, v = result.x[:p], result.x[p:]
        w = u - v
        return w * scale
    else:
        return np.zeros(p)


def solve_portfolio(mu_hat, Sigma_hat, a=RISK_AVERSION, c=GROSS_EXP,
                    max_pos=MAX_POS, scale=SCALE):
    """Try CVXPY first (faster, more reliable), fall back to scipy."""
    try:
        import cvxpy as cp
        p = len(mu_hat)
        w = cp.Variable(p)
        Sigma_reg = Sigma_hat + 1e-6 * np.eye(p)
        objective = cp.Minimize(
            -a * mu_hat @ w + (a**2 / 2) * cp.quad_form(w, cp.psd_wrap(Sigma_reg))
        )
        constraints = [
            cp.sum(w) == 0,
            cp.norm(w, 1) <= c,
            w <=  max_pos,
            w >= -max_pos,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, max_iters=5000, verbose=False)
        if w.value is not None and prob.status in ['optimal', 'optimal_inaccurate']:
            return w.value * scale
    except (ImportError, Exception):
        pass

    # Fallback to scipy
    return solve_portfolio_scipy(mu_hat, Sigma_hat, a, c, max_pos, scale)


# initialisation function: fits Ridge regression on training data, prepares rolling buffers
def initialise_state(data: pd.DataFrame) -> State:
    """
    Fit Ridge model on training data, store coefficients and rolling buffers.
    """
    df = data.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date

    symbols = sorted(df['symbol'].unique().tolist())
    assert len(symbols) == NUM_SYMBOLS, f"Expected {NUM_SYMBOLS} symbols, got {len(symbols)}"

    dates = np.sort(df['date'].unique())

    # ── Step 1: Build full feature matrix on training data ──
    df_feat = _build_features_batch(df, symbols)

    # ── Step 2: Prepare Ridge training data ──
    df_feat['fwd_ret_21d'] = (
        df_feat.groupby('symbol')['close'].shift(-21) / df_feat['close'] - 1.0
    )

    # Identify factor columns
    base_cols = {'date', 'symbol', 'open', 'close', 'low', 'high', 'volume',
                 'ret', 'fwd_ret_21d'}
    all_factors = [c for c in df_feat.columns if c not in base_cols]

    # IC-IR screening (quick version: compute mean |IC| per factor)
    ridge_candidates = _screen_factors(df_feat, all_factors, threshold=0.05)

    # Drop rows with NaN in features or target
    ridge_data = df_feat.dropna(subset=ridge_candidates + ['fwd_ret_21d'])

    # ── Step 3: Fit Ridge regression ──
    X_train = ridge_data[ridge_candidates].values.astype(np.float64)
    y_train = ridge_data['fwd_ret_21d'].values.astype(np.float64)

    # Standardise features
    scaler_mean = np.nanmean(X_train, axis=0)
    scaler_scale = np.nanstd(X_train, axis=0)
    scaler_scale[scaler_scale < 1e-10] = 1.0  # avoid division by zero
    X_scaled = (X_train - scaler_mean) / scaler_scale

    # Ridge closed-form: coefs = (X'X + alpha*I)^{-1} X'y
    n_feat = X_scaled.shape[1]
    XtX = X_scaled.T @ X_scaled
    Xty = X_scaled.T @ y_train
    coefs = np.linalg.solve(XtX + RIDGE_ALPHA * np.eye(n_feat), Xty)
    intercept = np.mean(y_train) - np.mean(X_scaled @ coefs)
    # (This matches sklearn Ridge with fit_intercept=True to high precision)

    # ── Step 4: Build rolling price buffers from last LOOKBACK days ──
    last_dates = dates[-LOOKBACK:]

    close_wide = df.pivot_table(index='date', columns='symbol',
                                values='close', aggfunc='first')[symbols]
    high_wide  = df.pivot_table(index='date', columns='symbol',
                                values='high', aggfunc='first')[symbols]
    low_wide   = df.pivot_table(index='date', columns='symbol',
                                values='low', aggfunc='first')[symbols]
    vol_wide   = df.pivot_table(index='date', columns='symbol',
                                values='volume', aggfunc='first')[symbols]

    close_buf = close_wide.loc[close_wide.index.isin(last_dates)].copy()
    high_buf  = high_wide.loc[high_wide.index.isin(last_dates)].copy()
    low_buf   = low_wide.loc[low_wide.index.isin(last_dates)].copy()
    vol_buf   = vol_wide.loc[vol_wide.index.isin(last_dates)].copy()

    return State(
        symbols=symbols,
        close_buf=close_buf,
        high_buf=high_buf,
        low_buf=low_buf,
        vol_buf=vol_buf,
        ridge_coefs=coefs,
        ridge_intercept=intercept,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_names=ridge_candidates,
        positions=np.zeros(NUM_SYMBOLS),
        wealth=1.0,
        peak_wealth=1.0,
        day_count=0,
        total_days=0,
    )


# trading algorithm function: called daily with new data, updates state, returns trades
def trading_algorithm(new_data: pd.DataFrame, state: State) -> Tuple[np.ndarray, State]:
    """
    Called once per trading day with that day's market data.
    Returns (trades, updated_state).
    """
    # Sort new_data to match state.symbols ordering
    new_data = new_data.copy()
    new_data['symbol'] = new_data['symbol'].astype(str)
    sym_rank = {s: i for i, s in enumerate(state.symbols)}
    new_data['_r'] = new_data['symbol'].map(sym_rank)
    new_data = new_data.sort_values('_r').drop(columns=['_r']).reset_index(drop=True)

    closes_today = new_data['close'].to_numpy(dtype=float)
    dt = new_data['date'].iloc[0]
    if hasattr(dt, 'date'):
        dt = dt.date() if callable(dt.date) else dt

    # Update rolling buffers 
    new_close = pd.DataFrame([closes_today], columns=state.symbols, index=[dt])
    new_high  = pd.DataFrame([new_data['high'].to_numpy(dtype=float)],
                             columns=state.symbols, index=[dt])
    new_low   = pd.DataFrame([new_data['low'].to_numpy(dtype=float)],
                             columns=state.symbols, index=[dt])
    new_vol   = pd.DataFrame([new_data['volume'].to_numpy(dtype=float)],
                             columns=state.symbols, index=[dt])

    state.close_buf = pd.concat([state.close_buf, new_close]).iloc[-LOOKBACK:]
    state.high_buf  = pd.concat([state.high_buf,  new_high]).iloc[-LOOKBACK:]
    state.low_buf   = pd.concat([state.low_buf,   new_low]).iloc[-LOOKBACK:]
    state.vol_buf   = pd.concat([state.vol_buf,   new_vol]).iloc[-LOOKBACK:]

    # Track wealth and positions (mirror walk_forward.py accounting) 
    if state.total_days > 0:
        prev_close = state.close_buf.iloc[-2].values
        r1d = closes_today / prev_close - 1.0
        r1d = np.where(np.isfinite(r1d), r1d, 0.0)
        state.wealth += np.sum(state.positions * r1d)
        state.positions = state.positions * (1.0 + r1d)
    state.peak_wealth = max(state.peak_wealth, state.wealth)

    state.total_days += 1
    state.day_count += 1

    # Check if we should rebalance 
    if state.day_count < HOLD_DAYS and state.total_days > 1:
        # No rebalancing today — zero trades
        return np.zeros(NUM_SYMBOLS), state

    state.day_count = 0  # reset counter

    # Layer 1: Compute features and predict mu_hat 
    feat_dict = compute_features_from_buffer(
        state.close_buf, state.high_buf, state.low_buf, state.vol_buf, state.symbols
    )

    # Assemble feature matrix (1 row, n_features columns)
    X_today = np.array([feat_dict.get(f, np.full(NUM_SYMBOLS, np.nan))
                        for f in state.feature_names]).T  # shape (NUM_SYMBOLS, n_features)

    # Replace NaN with 0 (will be near-zero after standardisation)
    X_today = np.nan_to_num(X_today, nan=0.0)

    # Standardise
    X_scaled = (X_today - state.scaler_mean) / state.scaler_scale

    # Ridge prediction
    mu_hat = X_scaled @ state.ridge_coefs + state.ridge_intercept

    # ── Layer 2: Portfolio optimisation ──
    # Estimate covariance from rolling returns
    if len(state.close_buf) > COV_WINDOW + 1:
        ret_matrix = state.close_buf.pct_change().iloc[-(COV_WINDOW+1):].dropna()
        Sigma_hat = np.cov(ret_matrix.values, rowvar=False)
    else:
        # Not enough data — use identity (= equal-risk weighting)
        Sigma_hat = np.eye(NUM_SYMBOLS)

    # Solve constrained optimisation
    target = solve_portfolio(mu_hat, Sigma_hat,
                             a=RISK_AVERSION, c=GROSS_EXP,
                             max_pos=MAX_POS, scale=SCALE)

    # Drawdown protection for risk management: reduce target exposure if we are in a large drawdown
    if DD_PROTECT and state.peak_wealth > 0:
        dd = 1.0 - state.wealth / state.peak_wealth
        if dd > DD_THRESH_2:
            target *= 0.25
        elif dd > DD_THRESH_1:
            target *= 0.50

    # Compute trades as difference between target and current positions
    trades = target - state.positions
    state.positions = target  # update internal positions to target

    return trades.astype(float), state


#  Batch feature construction (used only in initialise_state)
def _build_features_batch(df, symbols):
    """Build all features on the full training DataFrame (batch mode)."""
    d = df.copy().sort_values(['symbol', 'date'])
    gp = d.groupby('symbol')['close']
    gh = d.groupby('symbol')['high']
    gl = d.groupby('symbol')['low']

    d['ret'] = gp.pct_change()
    gr = d.groupby('symbol')['ret']

    # Momentum
    for w in MOMENTUM_WINDOWS:
        d[f'mom_{w}d'] = gp.pct_change(w)
    for w in [5, 10, 20, 60]:
        d[f'mom_{w}d_mean'] = gr.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())

    # Volatility
    for w in VOL_WINDOWS:
        d[f'vol_std_{w}d']  = gr.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
        d[f'vol_skew_{w}d'] = gr.transform(lambda s, w=w: s.rolling(w, min_periods=w).skew())
        d[f'vol_kurt_{w}d'] = gr.transform(lambda s, w=w: s.rolling(w, min_periods=w).kurt())

    # Custom factors
    d['factor1'] = gh.transform(lambda s: s.rolling(60, 60).max()) / \
                   gl.transform(lambda s: s.rolling(60, 60).min()) - 1

    r5 = gp.pct_change(5)
    d['factor4'] = r5.groupby(d['symbol']).transform(lambda s: s.rolling(60, 20).skew())
    r10 = gp.pct_change(10)
    d['factor4_10d'] = r10.where(r10 > 0).groupby(d['symbol']).transform(
        lambda s: s.rolling(60, 20).skew())

    d['factor5_30'] = d.groupby('symbol', group_keys=False).apply(
        lambda g: g['close'].shift(1).rolling(30, 30).corr(g['volume'])).values
    d['factor5_10'] = d.groupby('symbol', group_keys=False).apply(
        lambda g: g['close'].shift(1).rolling(10, 10).corr(g['volume'])).values

    for name, p in [('factor7', 7), ('factor8', 20), ('factor10', 30), ('factor9', 60)]:
        rk = gp.pct_change(p)
        d[name] = rk.groupby(d['symbol']).transform(lambda s: s.rolling(60, 60).min())

    # Z-scores
    d['log_price'] = np.log(d['close'])
    glp = d.groupby('symbol')['log_price']
    for w in ZSCORE_WINDOWS:
        ma = glp.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        sd = glp.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
        d[f'zscore_{w}d'] = (d['log_price'] - ma) / sd
    d.drop(columns=['log_price'], inplace=True)

    # Idiosyncratic momentum
    for w in IDIO_MOM_WINDOWS:
        col = f'mom_{w}d'
        xs_mean = d.groupby('date')[col].transform('mean')
        d[f'idio_mom_{w}d'] = d[col] - xs_mean

    return d.sort_values(['date', 'symbol']).reset_index(drop=True)


def _screen_factors(df_feat, all_factors, threshold=0.05):
    """Quick IC-IR screening: keep factors with |IC-IR| >= threshold."""
    from scipy.stats import spearmanr

    df_eval = df_feat.dropna(subset=['fwd_ret_21d'])
    results = {}
    for f in all_factors:
        ics = []
        for dt, grp in df_eval.groupby('date'):
            fvals = grp[f].values
            yvals = grp['fwd_ret_21d'].values
            valid = np.isfinite(fvals) & np.isfinite(yvals)
            if valid.sum() >= 20:
                ic, _ = spearmanr(fvals[valid], yvals[valid])
                ics.append(ic)
        if len(ics) >= 30:
            mean_ic = np.mean(ics)
            std_ic  = np.std(ics) if np.std(ics) > 0 else 1e-6
            results[f] = abs(mean_ic / std_ic)

    selected = [f for f, ic_ir in results.items() if ic_ir >= threshold]
    # Sort by IC-IR for reproducibility
    selected.sort(key=lambda f: -results[f])
    return selected

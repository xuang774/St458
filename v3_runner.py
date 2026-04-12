import numpy as np
import pandas as pd

from scipy.stats import spearmanr, skew, kurtosis
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def initialise_state(data: pd.DataFrame):
    FORWARD_DAYS = 21
    HOLD_DAYS = 21
    IC_THRESHOLD = 0.05
    RIDGE_ALPHAS = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    RISK_AVERSION_A = 0.5
    GROSS_EXPOSURE_C = 3.0
    MAX_POS = 0.10
    SCALE = 1.0
    DD_PROTECT = False
    COST_RATE = 0.0005

    def add_momentum_factors(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy().sort_values(["symbol", "date"])
        gp = d.groupby("symbol")["close"]
        d["ret"] = gp.pct_change()

        for w in [5, 10, 20, 60]:
            d[f"mom_{w}d"] = gp.pct_change(w)
            d[f"mom_{w}d_mean"] = d.groupby("symbol")["ret"].transform(
                lambda s, w=w: s.rolling(w, min_periods=w).mean()
            )
        return d.sort_values(["date", "symbol"]).reset_index(drop=True)

    def add_vol_factors(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy().sort_values(["symbol", "date"])
        gp = d.groupby("symbol")["close"]
        gh = d.groupby("symbol")["high"]
        gl = d.groupby("symbol")["low"]

        if "ret" not in d.columns:
            d["ret"] = gp.pct_change()
        gr = d.groupby("symbol")["ret"]

        for w in [5, 10, 20, 60]:
            d[f"vol_std_{w}d"] = gr.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
            d[f"vol_skew_{w}d"] = gr.transform(lambda s, w=w: s.rolling(w, min_periods=w).skew())
            d[f"vol_kurt_{w}d"] = gr.transform(lambda s, w=w: s.rolling(w, min_periods=w).kurt())

        d["factor1"] = (
            gh.transform(lambda s: s.rolling(60, min_periods=60).max())
            / gl.transform(lambda s: s.rolling(60, min_periods=60).min())
            - 1.0
        )

        r5 = gp.pct_change(5)
        d["factor4"] = r5.groupby(d["symbol"]).transform(
            lambda s: s.rolling(60, min_periods=20).skew()
        )

        r10 = gp.pct_change(10)
        d["factor4_10d"] = (
            r10.where(r10 > 0)
            .groupby(d["symbol"])
            .transform(lambda s: s.rolling(60, min_periods=20).skew())
        )

        d["factor5_30"] = d.groupby("symbol", group_keys=False).apply(
            lambda g: g["close"].shift(1).rolling(30, min_periods=30).corr(g["volume"])
        ).values
        d["factor5_10"] = d.groupby("symbol", group_keys=False).apply(
            lambda g: g["close"].shift(1).rolling(10, min_periods=10).corr(g["volume"])
        ).values

        for name, p in [("factor7", 7), ("factor8", 20), ("factor10", 30), ("factor9", 60)]:
            rk = gp.pct_change(p)
            d[name] = rk.groupby(d["symbol"]).transform(
                lambda s: s.rolling(60, min_periods=60).min()
            )

        return d.sort_values(["date", "symbol"]).reset_index(drop=True)

    def add_new_factors(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy().sort_values(["symbol", "date"])
        gp = d.groupby("symbol")["close"]

        for w in [30, 42, 126]:
            d[f"mom_{w}d"] = gp.pct_change(w)

        d["log_price"] = np.log(d["close"])
        glp = d.groupby("symbol")["log_price"]
        for w in [21, 42, 63]:
            ma = glp.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
            sd = glp.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
            d[f"zscore_{w}d"] = (d["log_price"] - ma) / sd
        d.drop(columns=["log_price"], inplace=True)

        for w in [30, 42, 60]:
            col = f"mom_{w}d"
            xs_mean = d.groupby("date")[col].transform("mean")
            d[f"idio_mom_{w}d"] = d[col] - xs_mean

        return d.sort_values(["date", "symbol"]).reset_index(drop=True)

    def evaluate_factor_rank_ic(d: pd.DataFrame, factor_cols, forward_days=21) -> pd.DataFrame:
        d = d.copy().sort_values(["symbol", "date"])
        d["fwd_ret"] = d.groupby("symbol")["close"].shift(-forward_days) / d["close"] - 1.0

        rows = []
        for fac in factor_cols:
            ics = []
            for _, g in d.groupby("date"):
                x = g[fac].values
                y = g["fwd_ret"].values
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() >= 10:
                    ic = spearmanr(x[valid], y[valid]).correlation
                    if np.isfinite(ic):
                        ics.append(ic)

            mean_ic = float(np.mean(ics)) if ics else 0.0
            std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else np.nan
            ic_ir = mean_ic / std_ic if np.isfinite(std_ic) and std_ic > 0 else 0.0

            rows.append(
                {
                    "factor": fac,
                    "mean_ic": mean_ic,
                    "ic_ir": ic_ir,
                    "abs_ic_ir": abs(ic_ir),
                }
            )

        return pd.DataFrame(rows).sort_values("abs_ic_ir", ascending=False).reset_index(drop=True)

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    symbols = sorted(df["symbol"].unique().tolist())
    n = len(symbols)

    df_train_feat = add_new_factors(add_vol_factors(add_momentum_factors(df)))

    base_cols = ["date", "symbol", "open", "close", "low", "high", "volume", "ret"]
    factor_cols = [c for c in df_train_feat.columns if c not in base_cols]

    ic_summary = evaluate_factor_rank_ic(df_train_feat, factor_cols, forward_days=FORWARD_DAYS)
    ridge_candidates = ic_summary.loc[ic_summary["abs_ic_ir"] >= IC_THRESHOLD, "factor"].tolist()
    if not ridge_candidates:
        raise ValueError("No ridge candidates selected. Lower IC_THRESHOLD or check features.")

    df_ridge_train = df_train_feat.copy()
    df_ridge_train["fwd_ret_21d"] = (
        df_ridge_train.groupby("symbol")["close"].shift(-FORWARD_DAYS) / df_ridge_train["close"] - 1.0
    )
    ridge_data = df_ridge_train.dropna(subset=ridge_candidates + ["fwd_ret_21d"]).copy()
    if ridge_data.empty:
        raise ValueError("No valid ridge training rows after dropping NaNs.")

    train_dates = sorted(ridge_data["date"].unique())
    mid = len(train_dates) // 2
    val_start = train_dates[mid]

    cv_train = ridge_data[ridge_data["date"] < val_start].copy()
    cv_val = ridge_data[ridge_data["date"] >= val_start].copy()

    best_alpha = RIDGE_ALPHAS[0]
    best_ic = -np.inf

    for alpha in RIDGE_ALPHAS:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(cv_train[ridge_candidates].fillna(0.0))
        y_tr = cv_train["fwd_ret_21d"].values

        X_va = scaler.transform(cv_val[ridge_candidates].fillna(0.0))
        y_va = cv_val["fwd_ret_21d"].values

        model = Ridge(alpha=alpha).fit(X_tr, y_tr)
        y_pred = model.predict(X_va)

        ics = []
        for dt in cv_val["date"].unique():
            mask = cv_val["date"].values == dt
            yt = y_va[mask]
            yp = y_pred[mask]
            valid = np.isfinite(yt) & np.isfinite(yp)
            if valid.sum() >= 10:
                ic = spearmanr(yt[valid], yp[valid]).correlation
                if np.isfinite(ic):
                    ics.append(ic)

        mean_ic = np.mean(ics) if ics else -np.inf
        if mean_ic > best_ic:
            best_ic = mean_ic
            best_alpha = alpha

    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(ridge_data[ridge_candidates].fillna(0.0))
    y_all = ridge_data["fwd_ret_21d"].values
    ridge_model = Ridge(alpha=best_alpha).fit(X_all, y_all)

    lookback = 127
    dates = np.sort(df["date"].unique())
    if len(dates) < lookback:
        raise ValueError(f"Need at least {lookback} training dates, got {len(dates)}")

    close_hist = np.full((lookback, n), np.nan, dtype=float)
    high_hist = np.full((lookback, n), np.nan, dtype=float)
    low_hist = np.full((lookback, n), np.nan, dtype=float)
    vol_hist = np.full((lookback, n), np.nan, dtype=float)

    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    for i in range(lookback):
        dt = dates[len(dates) - 1 - i]
        sub = df.loc[df["date"] == dt, ["symbol", "close", "high", "low", "volume"]]
        for _, row in sub.iterrows():
            j = sym_to_idx[row["symbol"]]
            close_hist[i, j] = float(row["close"])
            high_hist[i, j] = float(row["high"])
            low_hist[i, j] = float(row["low"])
            vol_hist[i, j] = float(row["volume"])

    state = {
        "symbols": symbols,
        "positions": np.zeros(n, dtype=float),
        "wealth": 1.0,
        "peak_wealth": 1.0,
        "prev_close": close_hist[0].copy(),
        "close_hist": close_hist,
        "high_hist": high_hist,
        "low_hist": low_hist,
        "vol_hist": vol_hist,
        "day_idx": 0,
        "hold_days": HOLD_DAYS,
        "cost_rate": COST_RATE,
        "ridge_candidates": ridge_candidates,
        "scaler": scaler_final,
        "ridge_model": ridge_model,
        "a": RISK_AVERSION_A,
        "c": GROSS_EXPOSURE_C,
        "scale": SCALE,
        "max_pos": MAX_POS,
        "dd_protect": DD_PROTECT,
    }
    return state


def trading_algorithm(new_data: pd.DataFrame, state):
    def corr_cols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mx = np.nanmean(x, axis=0)
        my = np.nanmean(y, axis=0)
        xc = x - mx
        yc = y - my
        cov = np.nanmean(xc * yc, axis=0)
        sx = np.nanstd(x, axis=0)
        sy = np.nanstd(y, axis=0)
        denom = sx * sy
        return np.where(denom > 0, cov / denom, np.nan)

    def compute_today_factor_df(symbols, close_hist, high_hist, low_hist, vol_hist) -> pd.DataFrame:
        ret_1d_5 = close_hist[:5, :] / close_hist[1:6, :] - 1.0
        ret_1d_10 = close_hist[:10, :] / close_hist[1:11, :] - 1.0
        ret_1d_20 = close_hist[:20, :] / close_hist[1:21, :] - 1.0
        ret_1d_60 = close_hist[:60, :] / close_hist[1:61, :] - 1.0

        feat = {}

        for w in [5, 10, 20, 30, 42, 60, 126]:
            feat[f"mom_{w}d"] = close_hist[0, :] / close_hist[w, :] - 1.0

        feat["mom_5d_mean"] = np.nanmean(ret_1d_5, axis=0)
        feat["mom_10d_mean"] = np.nanmean(ret_1d_10, axis=0)
        feat["mom_20d_mean"] = np.nanmean(ret_1d_20, axis=0)
        feat["mom_60d_mean"] = np.nanmean(ret_1d_60, axis=0)

        for w, arr in [(5, ret_1d_5), (10, ret_1d_10), (20, ret_1d_20), (60, ret_1d_60)]:
            feat[f"vol_std_{w}d"] = np.nanstd(arr, axis=0, ddof=1)
            feat[f"vol_skew_{w}d"] = skew(arr, axis=0, bias=False, nan_policy="omit")
            feat[f"vol_kurt_{w}d"] = kurtosis(arr, axis=0, fisher=True, bias=False, nan_policy="omit")

        feat["factor1"] = np.nanmax(high_hist[:60, :], axis=0) / np.nanmin(low_hist[:60, :], axis=0) - 1.0

        r5_hist = close_hist[:60, :] / close_hist[5:65, :] - 1.0
        feat["factor4"] = skew(r5_hist, axis=0, bias=False, nan_policy="omit")

        r10_hist = close_hist[:60, :] / close_hist[10:70, :] - 1.0
        r10_pos = np.where(r10_hist > 0, r10_hist, np.nan)
        feat["factor4_10d"] = skew(r10_pos, axis=0, bias=False, nan_policy="omit")

        feat["factor5_30"] = corr_cols(close_hist[1:31, :], vol_hist[:30, :])
        feat["factor5_10"] = corr_cols(close_hist[1:11, :], vol_hist[:10, :])

        for name, p in [("factor7", 7), ("factor8", 20), ("factor10", 30), ("factor9", 60)]:
            ret_p_hist = close_hist[:60, :] / close_hist[p : p + 60, :] - 1.0
            feat[name] = np.nanmin(ret_p_hist, axis=0)

        log_close = np.log(close_hist)
        for w in [21, 42, 63]:
            lp = log_close[:w, :]
            mu = np.nanmean(lp, axis=0)
            sd = np.nanstd(lp, axis=0, ddof=1)
            feat[f"zscore_{w}d"] = np.where(sd > 0, (lp[0, :] - mu) / sd, np.nan)

        for w in [30, 42, 60]:
            base = feat[f"mom_{w}d"]
            feat[f"idio_mom_{w}d"] = base - np.nanmean(base)

        return pd.DataFrame({"symbol": symbols, **feat})

    def solve_constrained_portfolio(mu_hat, sigma_hat, a=0.5, c=3.0, scale=1.0, max_pos=0.10):
        p = len(mu_hat)
        sigma_reg = sigma_hat + 1e-6 * np.eye(p)

        def obj(x):
            u = x[:p]
            v = x[p:]
            w = u - v
            return -a * np.dot(mu_hat, w) + 0.5 * (a ** 2) * np.dot(w, sigma_reg @ w)

        def grad(x):
            u = x[:p]
            v = x[p:]
            w = u - v
            gw = -a * mu_hat + (a ** 2) * (sigma_reg @ w)
            return np.concatenate([gw, -gw])

        cons = [
            {
                "type": "eq",
                "fun": lambda x: np.sum(x[:p] - x[p:]),
                "jac": lambda x: np.concatenate([np.ones(p), -np.ones(p)]),
            },
            {
                "type": "ineq",
                "fun": lambda x: c - np.sum(x[:p] + x[p:]),
                "jac": lambda x: -np.ones(2 * p),
            },
        ]
        bounds = [(0.0, max_pos)] * (2 * p)
        x0 = np.zeros(2 * p)

        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 500, "ftol": 1e-9, "disp": False},
            )
            if res.success and np.all(np.isfinite(res.x)):
                u = res.x[:p]
                v = res.x[p:]
                return (u - v) * scale
        except Exception:
            pass

        return np.zeros(p, dtype=float)

    symbols = state["symbols"]
    positions = state["positions"]
    wealth = state["wealth"]
    peak_wealth = state["peak_wealth"]
    prev_close = state["prev_close"]
    close_hist = state["close_hist"]
    high_hist = state["high_hist"]
    low_hist = state["low_hist"]
    vol_hist = state["vol_hist"]
    day_idx = state["day_idx"]
    hold_days = state["hold_days"]
    cost_rate = state["cost_rate"]

    ridge_candidates = state["ridge_candidates"]
    scaler = state["scaler"]
    ridge_model = state["ridge_model"]

    a = state["a"]
    c = state["c"]
    scale = state["scale"]
    max_pos = state["max_pos"]
    dd_protect = state["dd_protect"]

    x = new_data.copy()
    x["symbol"] = x["symbol"].astype(str)
    x = x.set_index("symbol").reindex(symbols).reset_index()

    required_cols = ["close", "high", "low", "volume"]
    missing_cols = [col for col in required_cols if col not in x.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in new_data: {missing_cols}")

    closes_today = x["close"].to_numpy(dtype=float)
    highs_today = x["high"].to_numpy(dtype=float)
    lows_today = x["low"].to_numpy(dtype=float)
    vols_today = x["volume"].to_numpy(dtype=float)

    if prev_close is not None:
        r1d = closes_today / prev_close - 1.0
        r1d = np.where(np.isfinite(r1d), r1d, 0.0)
        wealth = float(wealth + np.sum(positions * r1d))
        positions = positions * (1.0 + r1d)

    close_hist[1:, :] = close_hist[:-1, :]
    close_hist[0, :] = closes_today

    high_hist[1:, :] = high_hist[:-1, :]
    high_hist[0, :] = highs_today

    low_hist[1:, :] = low_hist[:-1, :]
    low_hist[0, :] = lows_today

    vol_hist[1:, :] = vol_hist[:-1, :]
    vol_hist[0, :] = vols_today

    if day_idx % hold_days == 0:
        factor_df = compute_today_factor_df(symbols, close_hist, high_hist, low_hist, vol_hist)

        X_today = factor_df[ridge_candidates].fillna(0.0).values
        X_today = scaler.transform(X_today)
        mu_hat = ridge_model.predict(X_today)

        ret_cov = close_hist[:126, :] / close_hist[1:127, :] - 1.0
        sigma_hat = np.cov(ret_cov, rowvar=False)

        s = scale
        if dd_protect and peak_wealth > 0:
            dd = 1.0 - wealth / peak_wealth
            if dd > 0.10:
                s = scale * 0.25
            elif dd > 0.05:
                s = scale * 0.50

        target_weights = solve_constrained_portfolio(
            mu_hat=mu_hat,
            sigma_hat=sigma_hat,
            a=a,
            c=c,
            scale=s,
            max_pos=max_pos,
        )

        target_positions = target_weights * wealth
        trades = target_positions - positions
        positions = target_positions

        wealth = float(wealth - cost_rate * np.sum(np.abs(trades)))
    else:
        trades = np.zeros_like(positions)

    peak_wealth = max(peak_wealth, wealth)

    state["positions"] = positions
    state["wealth"] = wealth
    state["peak_wealth"] = peak_wealth
    state["prev_close"] = closes_today
    state["close_hist"] = close_hist
    state["high_hist"] = high_hist
    state["low_hist"] = low_hist
    state["vol_hist"] = vol_hist
    state["day_idx"] = day_idx + 1

    return trades.astype(float), state
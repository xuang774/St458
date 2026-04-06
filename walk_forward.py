# walk_forward.py
# Python version of the R script

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def walk_forward(strategy, initialiser, df_train, df_test, cost_rate=0.0005):
    # Initialise state
    state = initialiser(df_train)

    test_dates = sorted(df_test["date"].unique())
    n_test_dates = len(test_dates)

    num_symbols = len(df_test[df_test["date"] == test_dates[0]])
    positions = np.zeros(num_symbols)
    daily_pnl = np.zeros(n_test_dates)

    for i, dt in enumerate(test_dates):
        new_data = df_test[df_test["date"] == dt]

        trades, state = strategy(new_data, state)

        if i == 0:
            price = new_data["close"].to_numpy()
            positions = trades
            daily_pnl[i] = -cost_rate * np.sum(np.abs(trades))
        else:
            price_lag1 = price
            price = new_data["close"].to_numpy()
            r1d = price / price_lag1 - 1.0

            daily_pnl[i] = np.sum(positions * r1d) - cost_rate * np.sum(np.abs(trades))
            positions = positions * (1.0 + r1d) + trades

    wealth_seq = 1.0 + np.cumsum(daily_pnl)

    # Ensure wealth does not go negative
    if np.any(wealth_seq <= 0):
        first_idx = np.where(wealth_seq <= 0)[0][0]
        wealth_seq[first_idx:] = 0.0

    plt.plot(wealth_seq)
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.show()
    return wealth_seq


if __name__ == "__main__":
    df = pd.read_csv("df_train.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    train_idx = df["date"] < pd.to_datetime("2012-10-18").date()
    df_train = df[train_idx].copy()
    df_test = df[~train_idx].copy()

    import composite_long_short  # your strategy file

    wealth_seq = walk_forward(
        composite_long_short.trading_algorithm,
        composite_long_short.initialise_state,
        df_train,
        df_test,
        cost_rate=0.0005,
    )
    print("wealth sequence:", wealth_seq)
    print("log wealth =", np.log(wealth_seq[-1]) if wealth_seq[-1] > 0 else -np.inf)
    
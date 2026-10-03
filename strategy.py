"""Event filter on Bitget rNVDAUSDT hourly candles."""

import argparse
from pathlib import Path

import pandas as pd

SPIKE = 0.012
VOLUME_MULT = 1.8
HOLD = 6
STOP = 0.01
TAKE = 0.02
MAX_RANGE = 0.04
FEE = 0.0006  # 6 bps per side

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["signal"] = (
        (df["ret"] >= SPIKE)
        & (df["volume"] >= VOLUME_MULT * df["vol_ma"])
        & (df["range"] <= MAX_RANGE)
    )
    trades = []
    i = 20
    while i < len(df) - 2:
        if not bool(df.loc[i, "signal"]):
            i += 1
            continue
        entry_i = i + 1
        entry = float(df.loc[entry_i, "open"])
        exit_i = min(entry_i + HOLD, len(df) - 1)
        exit_px = float(df.loc[exit_i, "close"])
        reason = "time"
        for j in range(entry_i, exit_i + 1):
            if float(df.loc[j, "low"]) <= entry * (1 - STOP):
                exit_px, exit_i, reason = entry * (1 - STOP), j, "stop"
                break
            if float(df.loc[j, "high"]) >= entry * (1 + TAKE):
                exit_px, exit_i, reason = entry * (1 + TAKE), j, "take"
                break
        trades.append({
            "signal_time": df.loc[i, "timestamp"],
            "entry_time": df.loc[entry_i, "timestamp"],
            "exit_time": df.loc[exit_i, "timestamp"],
            "entry": round(entry, 4),
            "exit": round(exit_px, 4),
            "net_return": exit_px / entry - 1 - 2 * FEE,
            "reason": reason,
        })
        i = exit_i + 1
    return pd.DataFrame(trades)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/rNVDAUSDT_1h.csv")
    args = p.parse_args()
    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    trades = backtest(df)
    trades.to_csv("trades.csv", index=False)
    if trades.empty:
        print("no trades")
        return
    r = trades["net_return"]
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    print("symbol: rNVDAUSDT (Bitget spot rToken)")
    print(f"bars: {len(df)}  {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}")
    print(f"trades: {len(trades)}")
    print(f"win_rate: {(r > 0).mean():.2%}")
    print(f"avg_net: {r.mean():.4%}")
    print(f"total_net: {equity.iloc[-1] - 1:.2%}")
    print(f"max_drawdown: {dd.min():.2%}")
    print(trades.tail(8).to_string(index=False))

if __name__ == "__main__":
    main()


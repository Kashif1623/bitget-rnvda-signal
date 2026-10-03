"""Download real Bitget rNVDAUSDT (rToken) hourly candles. No API key."""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SYMBOL = "rNVDAUSDT"
OUT = Path("data/rNVDAUSDT_1h.csv")
URL = "https://api.bitget.com/api/v2/spot/market/candles"

def get(params: dict) -> list:
    q = urllib.parse.urlencode(params)
    with urllib.request.urlopen(URL + "?" + q, timeout=30) as r:
        body = json.loads(r.read().decode())
    if body.get("code") != "00000":
        raise SystemExit(body)
    return body["data"]

def main() -> None:
    rows = []
    end = None
    for _ in range(8):
        params = {"symbol": SYMBOL, "granularity": "1h", "limit": "200"}
        if end:
            params["endTime"] = str(end)
        batch = get(params)
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(x[0]) for x in batch)
        end = oldest - 1
        time.sleep(0.2)
        if len(batch) < 200:
            break
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "base_volume", "quote_volume", "usdt_volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "base_volume"]:
        df[c] = df[c].astype(float)
    df = df.rename(columns={"base_volume": "volume"})
    df = (
        df[["timestamp", "open", "high", "low", "close", "volume"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{SYMBOL} bars={len(df)} {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()

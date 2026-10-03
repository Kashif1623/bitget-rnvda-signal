"""Bitget rNVDAUSDT agent.

Senses a spike, asks the model allow/skip, then applies hard risk limits.
Without an API key it still runs, but the decision is marked fallback, not LLM.
"""

import json
import os
import urllib.request
import pandas as pd

SPIKE = 0.012
VOLUME_MULT = 1.8
MAX_RANGE = 0.04
STOP = 0.01
TAKE = 0.02
HOLD = 6
FEE = 0.0006
MAX_TRADES_PER_DAY = 3

def sense(row, vol_ma):
    jump = row["close"] / row["open"] - 1
    hot = row["volume"] >= VOLUME_MULT * vol_ma
    calm = (row["high"] - row["low"]) / row["close"] <= MAX_RANGE
    return jump >= SPIKE and hot and calm, jump

def ask_model(state):
    key = os.environ.get("QWEN_API_KEY", "")
    if not key:
        allow = state["jump"] >= SPIKE and state["range"] <= 0.03
        return {"allow": allow, "reason": "fallback_rule_no_api_key", "source": "fallback"}
    prompt = (
        "You are a risk-limited trading agent for Bitget rNVDAUSDT. "
        "Reply only JSON: {\"allow\": true/false, \"reason\": \"...\"}. "
        "Allow only if the spike is tradable and range is not chaotic. State: "
        + json.dumps(state)
    )
    body = json.dumps({
        "model": "qwen-plus",
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        text = json.loads(r.read().decode())["choices"][0]["message"]["content"]
    decision = json.loads(text)
    decision["source"] = "qwen"
    return decision

def main():
    df = pd.read_csv("data/rNVDAUSDT_1h.csv", parse_dates=["timestamp"]).reset_index(drop=True)
    df["vol_ma"] = df["volume"].rolling(20).mean()
    trades = []
    day_count = {}
    i = 20
    while i < len(df) - 2:
        ok, jump = sense(df.loc[i], df.loc[i, "vol_ma"])
        if not ok:
            i += 1
            continue
        day = str(df.loc[i, "timestamp"])[:10]
        if day_count.get(day, 0) >= MAX_TRADES_PER_DAY:
            i += 1
            continue
        state = {
            "time": str(df.loc[i, "timestamp"]),
            "jump": round(jump, 4),
            "range": round((df.loc[i, "high"] - df.loc[i, "low"]) / df.loc[i, "close"], 4),
            "volume_x": round(df.loc[i, "volume"] / df.loc[i, "vol_ma"], 2),
        }
        decision = ask_model(state)
        if not decision.get("allow"):
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
            **state,
            "agent_reason": decision.get("reason"),
            "agent_source": decision.get("source"),
            "entry": round(entry, 4),
            "exit": round(exit_px, 4),
            "net_return": exit_px / entry - 1 - 2 * FEE,
            "exit_reason": reason,
        })
        day_count[day] = day_count.get(day, 0) + 1
        i = exit_i + 1
    out = pd.DataFrame(trades)
    out.to_csv("agent_trades.csv", index=False)
    print(out.tail(5).to_string(index=False) if len(out) else "no trades")
    print("trades", len(out))

if __name__ == "__main__":
    main()

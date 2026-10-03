"""Bitget rNVDAUSDT agent with candle context and a hard risk veto.

Qwen sees the latest 6 candles plus the spike. It must return allow, confidence, and reason.
Risk limits still override the model. Without a key the run is fallback, not LLM.
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
MIN_CONFIDENCE = 0.6

def sense(row, vol_ma):
    jump = row["close"] / row["open"] - 1
    hot = row["volume"] >= VOLUME_MULT * vol_ma
    calm = (row["high"] - row["low"]) / row["close"] <= MAX_RANGE
    return jump >= SPIKE and hot and calm, jump

def context(df, i):
    window = df.loc[i - 5:i, ["timestamp", "open", "high", "low", "close", "volume"]]
    rows = []
    for _, r in window.iterrows():
        rows.append({
            "time": str(r["timestamp"]),
            "open": round(float(r["open"]), 4),
            "high": round(float(r["high"]), 4),
            "low": round(float(r["low"]), 4),
            "close": round(float(r["close"]), 4),
            "volume": round(float(r["volume"]), 2),
        })
    return rows

def ask_model(state):
    key = os.environ.get("QWEN_API_KEY", "")
    if not key:
        allow = state["jump"] >= SPIKE and state["range"] <= 0.03
        return {
            "allow": allow,
            "confidence": 0.5,
            "reason": "fallback_rule_no_api_key",
            "source": "fallback",
        }
    prompt = (
        "You are a risk-limited trading agent for Bitget rNVDAUSDT. "
        "Use the last 6 hourly candles and the spike state. "
        "Reply only JSON with keys allow (bool), confidence (0 to 1), reason (one sentence). "
        "Allow only if the move is tradable and the range is not chaotic. State: "
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
    decision = json.loads(text[text.find("{"):text.rfind("}") + 1])
    decision["source"] = "qwen"
    return decision

def veto(decision, state):
    if decision.get("source") != "qwen":
        return decision.get("allow", False), "fallback"
    if not decision.get("allow"):
        return False, "model_skip"
    if float(decision.get("confidence", 0)) < MIN_CONFIDENCE:
        return False, "low_confidence"
    if state["range"] > 0.03:
        return False, "range_veto"
    return True, "model_allow"

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
            "last_6_candles": context(df, i),
        }
        decision = ask_model(state)
        allowed, veto_reason = veto(decision, state)
        if not allowed:
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
            "time": state["time"],
            "jump": state["jump"],
            "range": state["range"],
            "volume_x": state["volume_x"],
            "agent_reason": decision.get("reason"),
            "confidence": decision.get("confidence"),
            "agent_source": decision.get("source"),
            "veto_reason": veto_reason,
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

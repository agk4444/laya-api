#!/usr/bin/env python3
"""Smoke-test the running Laya API: one election-preset call + one options-filter call."""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read())


print("health:", get("/health"))

print("\n--- election ---")
print(json.dumps(post("/decide/election", {
    "context": "Maine Senate: poll aggregate D+2, Polymarket 54% Dem. Incumbent retiring."
}), indent=1))

print("\n--- options ---")
print(json.dumps(post("/decide/options", {
    "state": "NVDA 2026-10-16 $200 call. Underlying $192.40. Moneyness 4.0% OTM. "
             "Volume 12,400 vs open interest 1,800 (6.9x). Implied vol 48%. "
             "28 days to expiry. Most prints: likely BUY."
}), indent=1))

print("\n--- batch (3 states, one call) ---")
b = post("/decide/batch", {
    "states": [
        "NVDA up 3% on strong data-center demand.",
        "NVDA down 8% on 2x average volume.",
        "NVDA flat, no news.",
    ],
    "questions": {"bullish": {"type": "noul", "instructions": "Is the tone bullish?"}},
})
print("count:", b["count"], "| latency_ms:", b["latency_ms"])

print("\n--- systemone (Jev wire format) ---")
s = post("/v1/systemone", {
    "state": "Maine Senate: poll aggregate D+2, Polymarket 54% Dem. Incumbent retiring.",
    "model": "jev-latest",
    "questions": {
        "outcome": {"type": "choice", "instructions": "Which party wins this race?",
                    "criteria": {"democrat_win": "Dem wins", "republican_win": "Rep wins",
                                 "toss_up": "Too close to call"}},
        "confidence": {"type": "noul", "instructions": "How confident is this call"},
    },
})
print("model:", s["model"], "| answers:", sorted(s["answers"]), "| usage:", s["usage"])

print("\n--- triage / moderation presets ---")
for path, state in (("/decide/triage", "Charged twice, refund or I cancel."),
                    ("/decide/moderation", "Great product, thanks for the help!")):
    r = post(path, {"state": state})
    print(path, "->", sorted(r["answers"]))

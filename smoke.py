#!/usr/bin/env python3
"""Smoke-test the running Laya API: one Chunav call + one options-filter call."""
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

print("\n--- chunav ---")
print(json.dumps(post("/decide/chunav", {
    "context": "Maine Senate: poll aggregate D+2, Polymarket 54% Dem. Incumbent retiring."
}), indent=1))

print("\n--- options ---")
print(json.dumps(post("/decide/options", {
    "state": "NVDA 2026-10-16 $200 call. Underlying $192.40. Moneyness 4.0% OTM. "
             "Volume 12,400 vs open interest 1,800 (6.9x). Implied vol 48%. "
             "28 days to expiry. Most prints: likely BUY."
}), indent=1))

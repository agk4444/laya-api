#!/usr/bin/env python3
"""Laya decision model served as an HTTP API (CPU).

Loads the Laya System-1 checkpoint once at startup, then serves typed
decisions over HTTP. No text generation - choice / score / noul heads only.

Endpoints:
    GET  /health            liveness + which model is loaded
    POST /decide            generic: {"state": ..., "questions": {...}}
    POST /decide/election     preset:  {"context": "..."}  -> outcome / dem_win_probability / confidence
    POST /decide/options    preset:  {"state": "..."}     -> action / direction / conviction / confidence

Run:
    python server.py --model agk4444/laya-typed-decisions     # HF id (downloaded on first run)
    python server.py --model /path/to/checkpoint       # local weights dir
    LAYA_MODEL=agk4444/laya-typed-decisions python server.py

Then, e.g.:
    curl -X POST localhost:8000/decide/election \
         -H 'Content-Type: application/json' \
         -d '{"context": "Maine Senate: polls D+2, markets 54% Dem..."}'
"""
import argparse
import json
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"

# ---------------------------------------------------------------- presets

ELECTION_QUESTIONS = {
    "outcome": {
        "type": "choice",
        "instructions": "Which party wins this race?",
        "criteria": {
            "democrat_win": "The Democratic candidate / party wins",
            "republican_win": "The Republican candidate / party wins",
            "toss_up": "Too close to call - neither side has a clear edge",
        },
    },
    "dem_win_probability": {
        "type": "score",
        "instructions": "Probability the Democrat wins, 0 to 100",
        "criteria": [
            "0: certain Republican win",
            "50: pure toss-up",
            "100: certain Democratic win",
        ],
    },
    "confidence": {
        "type": "noul",
        "instructions": "How confident is this call, given poll/market agreement and data freshness",
    },
}

OPTIONS_QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Is this unusual options flow worth flagging to a trader?",
        "criteria": {
            "actionable": "Genuine unusual positioning - likely informed, directional, or institutional flow. Flag it.",
            "noise": "Likely noise - spread legs, hedging, expiry rolling, stale prints, or low-liquidity artifacts. Do not flag.",
        },
    },
    "direction": {
        "type": "choice",
        "instructions": "If actionable, which directional lean does the flow imply?",
        "criteria": {
            "bullish": "Flow implies bullish positioning (e.g. bought calls, sold puts)",
            "bearish": "Flow implies bearish positioning (e.g. bought puts, sold calls)",
            "neutral": "No clear directional lean",
        },
    },
    "conviction": {
        "type": "score",
        "instructions": "Conviction that this flow is actionable unusual positioning, 0 to 100",
        "criteria": ["0: certain noise", "50: could be either", "100: certain actionable flow"],
    },
    "confidence": {
        "type": "noul",
        "instructions": "How confident is this verdict, given the quality and completeness of the flow data",
    },
}

# ---------------------------------------------------------------- app state

app = FastAPI(title="Laya CPU API", version="1.0.0")
agent = None
agent_lock = threading.Lock()
model_ref = None


@app.get("/", response_class=HTMLResponse)
def ui():
    """Single-page GUI for the API."""
    return (STATIC_DIR / "index.html").read_text()


def get_answer(res, key):
    """Defensively pull one question's answer out of a laya predict() result."""
    if isinstance(res, dict):
        for container in ("answers", "results", "output"):
            if isinstance(res.get(container), dict) and key in res[container]:
                return res[container][key]
        if key in res:
            return res[key]
    return {}


def normalize_answer(ans, question):
    """Normalize one raw laya answer to a stable JSON shape."""
    qtype = question.get("type")
    if qtype == "choice":
        if not isinstance(ans, dict):
            return {"choice": None, "probabilities": {}}
        dist = ans.get("probabilities") or ans.get("distribution") or ans.get("probs") or {}
        return {
            "choice": ans.get("choice"),
            "probabilities": {k: round(float(v), 4) for k, v in dist.items()}
            if isinstance(dist, dict) else {},
        }
    if qtype == "score":
        criteria = question.get("criteria") or []
        n = len(criteria)
        # criteria labels span 0..100 across (n-1) steps, e.g. 3 labels ->
        # raw level 0/1/2 maps to 0/50/100. Same convention as the bake-off.
        divisor = (n - 1) if n > 1 else 1
        raw = ans.get("score") if isinstance(ans, dict) else None
        pct = None
        if isinstance(raw, (int, float)):
            pct = max(0, min(100, round(raw / divisor * 100)))
        return {"score_0_100": pct, "raw_score": raw, "levels": n}
    if qtype == "noul":
        v = None
        if isinstance(ans, dict):
            v = ans.get("noul")
            if v is None:
                v = ans.get("probability")
        return {"noul": round(float(v), 4) if isinstance(v, (int, float)) else None}
    return {"raw": ans}


def decide(state, questions):
    """Run one typed decision through Laya, serialized behind a lock."""
    t0 = time.time()
    with agent_lock:
        try:
            res = agent.predict(state, questions)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"laya predict failed: {exc}") from exc
    ms = round((time.time() - t0) * 1000, 1)
    answers = {qid: normalize_answer(get_answer(res, qid), q) for qid, q in questions.items()}
    return {"answers": answers, "latency_ms": ms}


# ---------------------------------------------------------------- routes

@app.get("/health")
def health():
    return {"ok": agent is not None, "model": model_ref, "device": "cpu"}


@app.post("/decide")
def decide_generic(payload: dict):
    state = payload.get("state")
    questions = payload.get("questions")
    if state is None or not isinstance(questions, dict) or not questions:
        raise HTTPException(status_code=400, detail="need {'state': ..., 'questions': {...}}")
    return decide(state, questions)


@app.post("/decide/election")
def decide_election(payload: dict):
    context = payload.get("context")
    if not context:
        raise HTTPException(status_code=400, detail="need {'context': '...'}")
    return decide({"context": context}, ELECTION_QUESTIONS)


@app.post("/decide/options")
def decide_options(payload: dict):
    state = payload.get("state")
    if not state:
        raise HTTPException(status_code=400, detail="need {'state': '...'}")
    return decide({"context": state} if isinstance(state, str) else state, OPTIONS_QUESTIONS)


# ---------------------------------------------------------------- main

def main():
    global agent, model_ref
    ap = argparse.ArgumentParser(description="Serve Laya as an HTTP API on CPU")
    ap.add_argument("--model", default=os.environ.get("LAYA_MODEL", "agk4444/laya-typed-decisions"),
                    help="HF id or local checkpoint dir")
    ap.add_argument("--subfolder", default=os.environ.get("LAYA_SUBFOLDER"),
                    help="checkpoint subfolder inside a bundled repo (e.g. multilingual)")
    ap.add_argument("--device", default=os.environ.get("LAYA_DEVICE", "cpu"),
                    help="torch device (default cpu)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = ap.parse_args()

    import laya

    model_ref = args.model
    t0 = time.time()
    agent = laya.load(args.model, device=args.device, subfolder=args.subfolder)
    print(f"laya ready: {args.model} on {args.device} in {time.time()-t0:.1f}s", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

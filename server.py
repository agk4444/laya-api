#!/usr/bin/env python3
"""Laya decision model served as an HTTP API (CPU).

Loads the Laya System-1 checkpoint once at startup, then serves typed
decisions over HTTP. No text generation - choice / score / noul heads only.

Endpoints:
    GET  /health            liveness + which model is loaded
    POST /decide            generic: {"state": ..., "questions": {...}}
    POST /decide/batch      many states, one call: {"states": [...], "questions": {...}}
    POST /decide/election   preset:  {"context": "..."}  -> outcome / dem_win_probability / confidence
    POST /decide/options    preset:  {"state": "..."}     -> action / direction / conviction / confidence
    POST /decide/triage     preset:  {"state": "..."}     -> intent / urgency / frustration / churn_risk
    POST /decide/moderation preset:  {"state": "..."}     -> verdict / severity / needs_review
    POST /v1/systemone      Jev-compatible: {"state": ..., "model": ..., "questions": {...}}
                            -> {model, answers, usage} in TypeSafe System One wire format

Auth: if LAYA_API_KEY is set, every POST route requires
    Authorization: Bearer <key>  (GET / and /health stay open).

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

from fastapi import Depends, FastAPI, HTTPException, Request
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

TRIAGE_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What is the customer asking about?",
        "criteria": {
            "billing": "invoices, charges, payments, refunds",
            "technical": "bugs, outages, errors, something not working",
            "sales": "pricing, plans, new contracts, upgrades",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound?",
        "criteria": ["calm", "annoyed", "furious"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the customer threaten to cancel or leave?",
    },
}

MODERATION_QUESTIONS = {
    "verdict": {
        "type": "choice",
        "instructions": "How should this content be classified?",
        "criteria": {
            "safe": "benign content, no policy issue",
            "toxic": "insults, hate, or demeaning language",
            "harassment": "targeted abuse toward a person or group",
            "threat": "threats of violence or harm",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the policy violation?",
        "criteria": ["benign", "mild", "severe"],
    },
    "needs_review": {
        "type": "noul",
        "instructions": "Should a human moderator review this content?",
    },
}

# ---------------------------------------------------------------- app state

app = FastAPI(title="Laya CPU API", version="1.1.0")

API_KEY = os.environ.get("LAYA_API_KEY")
MAX_BATCH_STATES = 100


async def require_auth(request: Request):
    """Bearer <redacted> gate for every POST route. No-op unless LAYA_API_KEY is set."""
    if API_KEY and request.headers.get("authorization") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized: bad or missing bearer token")


AUTH = [Depends(require_auth)]

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


def _run_predict(state, questions):
    """Run agent.predict serialized behind a lock. Returns (raw_result, latency_ms)."""
    t0 = time.time()
    with agent_lock:
        try:
            res = agent.predict(state, questions)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"laya predict failed: {exc}") from exc
    return res, round((time.time() - t0) * 1000, 1)


def decide(state, questions):
    """Run one typed decision through Laya, normalized to the /decide shape."""
    res, ms = _run_predict(state, questions)
    answers = {qid: normalize_answer(get_answer(res, qid), q) for qid, q in questions.items()}
    return {"answers": answers, "latency_ms": ms}


def jev_answers(res, questions):
    """Raw laya answers in TypeSafe System One wire shape.

    The SDK already emits {type, choice|score|noul, probabilities, legend,
    confidence} per question - identical to what Jev returns. The internal
    `action` (RL act_probability) is stripped; Jev has no such field.
    """
    out = {}
    for qid in questions:
        ans = get_answer(res, qid)
        if isinstance(ans, dict):
            ans = {k: v for k, v in ans.items() if k != "action"}
        out[qid] = ans
    return out


def check_questions(questions):
    if not isinstance(questions, dict) or not questions:
        raise HTTPException(status_code=400, detail="need a non-empty 'questions' object")


# ---------------------------------------------------------------- routes

@app.get("/health")
def health():
    return {"ok": agent is not None, "model": model_ref, "device": "cpu",
            "auth": bool(API_KEY)}


@app.post("/decide", dependencies=AUTH)
def decide_generic(payload: dict):
    state = payload.get("state")
    questions = payload.get("questions")
    if state is None:
        raise HTTPException(status_code=400, detail="need {'state': ..., 'questions': {...}}")
    check_questions(questions)
    return decide(state, questions)


@app.post("/decide/batch", dependencies=AUTH)
def decide_batch(payload: dict):
    states = payload.get("states")
    questions = payload.get("questions")
    if not isinstance(states, list) or not states:
        raise HTTPException(status_code=400, detail="need {'states': [...], 'questions': {...}}")
    check_questions(questions)
    if len(states) > MAX_BATCH_STATES:
        raise HTTPException(status_code=400,
                            detail=f"at most {MAX_BATCH_STATES} states per batch call")
    batch_size = payload.get("batch_size")
    t0 = time.time()
    with agent_lock:
        try:
            if hasattr(agent, "predict_batch"):
                results = agent.predict_batch(states, questions,
                                              batch_size=batch_size if isinstance(batch_size, int) else None)
            else:  # older laya wheels: serial fallback, same shapes
                results = [agent.predict(s, questions) for s in states]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"laya batch predict failed: {exc}") from exc
    ms = round((time.time() - t0) * 1000, 1)
    out = [{"answers": {qid: normalize_answer(get_answer(r, qid), q)
                        for qid, q in questions.items()}} for r in results]
    return {"results": out, "count": len(out), "latency_ms": ms}


@app.post("/v1/systemone", dependencies=AUTH)
def systemone(payload: dict):
    """TypeSafe Jev wire protocol: {state, model?, questions} -> {model, answers, usage}.

    A client written against https://api.typesafe.ai/v1/systemone only needs its
    base URL repointed here. The `model` field is accepted and echoed back as the
    loaded checkpoint (this server hosts one checkpoint; no switching).
    """
    state = payload.get("state")
    questions = payload.get("questions")
    if state is None:
        raise HTTPException(status_code=400, detail="need {'state': ..., 'questions': {...}}")
    check_questions(questions)
    res, ms = _run_predict(state, questions)
    usage = res.get("usage") if isinstance(res, dict) else None
    return {
        "model": model_ref,
        "answers": jev_answers(res, questions),
        "usage": usage if isinstance(usage, dict) else {"input_tokens": 0, "output_tokens": 0},
        "latency_ms": ms,
    }


@app.post("/decide/election", dependencies=AUTH)
def decide_election(payload: dict):
    context = payload.get("context")
    if not context:
        raise HTTPException(status_code=400, detail="need {'context': '...'}")
    return decide({"context": context}, ELECTION_QUESTIONS)


@app.post("/decide/options", dependencies=AUTH)
def decide_options(payload: dict):
    state = payload.get("state")
    if not state:
        raise HTTPException(status_code=400, detail="need {'state': '...'}")
    return decide({"context": state} if isinstance(state, str) else state, OPTIONS_QUESTIONS)


@app.post("/decide/triage", dependencies=AUTH)
def decide_triage(payload: dict):
    state = payload.get("state")
    if not state:
        raise HTTPException(status_code=400, detail="need {'state': '...'}")
    return decide({"context": state} if isinstance(state, str) else state, TRIAGE_QUESTIONS)


@app.post("/decide/moderation", dependencies=AUTH)
def decide_moderation(payload: dict):
    state = payload.get("state")
    if not state:
        raise HTTPException(status_code=400, detail="need {'state': '...'}")
    return decide({"context": state} if isinstance(state, str) else state, MODERATION_QUESTIONS)


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

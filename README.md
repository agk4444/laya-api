# Laya CPU API

Serve the [Laya](https://huggingface.co/agk4444/laya-typed-decisions)
typed-decisions model as a small HTTP API on CPU. The model loads once at
startup (~808MB checkpoint), then each decision takes seconds on CPU —
fine for batch jobs and scheduled runs, not for realtime.

Developed by **AGK FIRE INC**.

---

## Install

**Requirements:** Python 3.10+, pip. No GPU needed.

```bash
git clone https://github.com/agk4444/laya-api
cd laya-api
pip install -r requirements.txt
```

That's it. The model weights download automatically from Hugging Face on
first run (~808MB, cached afterwards).

### Run on Google Colab (from your phone)

Use the bundled `laya-api-colab.ipynb`: open it in Colab with a CPU
runtime and run top to bottom. It installs dependencies, starts the server
in the background, and runs the smoke test.

Note: Colab's port isn't public — the API is for code running in the same
notebook/runtime (`http://localhost:8000`).

---

## Start the server

```bash
python server.py --port 8000
```

Defaults to the fine-tuned `agk4444/laya-typed-decisions` checkpoint.

Options:

| Flag / env | Default | What |
|---|---|---|
| `--model` / `LAYA_MODEL` | `agk4444/laya-typed-decisions` | HF repo or local checkpoint path |
| `--subfolder` / `LAYA_SUBFOLDER` | — | subfolder inside the repo (e.g. `multilingual`) |
| `--device` / `LAYA_DEVICE` | `cpu` | `cpu` or `cuda` |
| `--host` | `0.0.0.0` | bind address |
| `--port` / `PORT` | `8000` | bind port |

Other checkpoints: `--model convaiinnovations/laya` (original base),
`--model convaiinnovations/laya --subfolder multilingual`.

---

## Usage

### 1. Health check

```bash
curl localhost:8000/health
```

```json
{"ok": true, "model": "agk4444/laya-typed-decisions", "device": "cpu"}
```

### 2. Generic decision — `POST /decide`

Send any state plus a set of typed questions. Question types:

- `choice` — pick one of `criteria` (a dict of option → description)
- `score` — rate 0–100 against `criteria` (a list of levels)
- `noul` — confidence 0–1

```bash
curl -X POST localhost:8000/decide \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {"context": "Should we deploy Friday afternoon? Error budget is 90% consumed and on-call is thin over the weekend."},
    "questions": {
      "deploy": {
        "type": "choice",
        "instructions": "Should we deploy?",
        "criteria": {
          "ship_it": "Risk is acceptable, deploy now",
          "wait": "Too risky, wait until Monday"
        }
      },
      "risk_score": {
        "type": "score",
        "instructions": "Deployment risk, 0 to 100",
        "criteria": ["low", "medium", "high", "critical"]
      },
      "confidence": {"type": "noul", "instructions": "Confidence in this call"}
    }
  }'
```

Response:

```json
{
  "answers": {
    "deploy": {"choice": "wait"},
    "risk_score": {"score_0_100": 78.4},
    "confidence": {"noul": 0.82}
  },
  "latency_ms": 4123.5
}
```

### 3. Example preset — `POST /decide/election`

A worked example of a fixed question set (election outcome). Send:

```bash
curl -X POST localhost:8000/decide/election \
  -H 'Content-Type: application/json' \
  -d '{"context": "Maine Senate: poll aggregate D+2, betting markets 54% Dem. Incumbent retiring."}'
```

```json
{
  "answers": {
    "outcome": {"choice": "democrat_win"},
    "dem_win_probability": {"score_0_100": 61.2},
    "confidence": {"noul": 0.58}
  },
  "latency_ms": 3890.1
}
```

### 4. Example preset — `POST /decide/options`

A worked example for filtering unusual options flow. Send:

```bash
curl -X POST localhost:8000/decide/options \
  -H 'Content-Type: application/json' \
  -d '{"state": "NVDA 2026-10-16 $200 call. Underlying $192.40, 4% OTM. Volume 12,400 vs open interest 1,800 (6.9x). IV 48%. 28 days to expiry."}'
```

```json
{
  "answers": {
    "action": {"choice": "actionable"},
    "direction": {"choice": "bullish"},
    "conviction": {"score_0_100": 86.0},
    "confidence": {"noul": 0.71}
  },
  "latency_ms": 4012.7
}
```

To add your own preset, copy the `ELECTION_QUESTIONS` / `OPTIONS_QUESTIONS`
pattern in `server.py` — it's a plain dict of typed questions plus a route.

### Python client

```python
import json, urllib.request

BASE = "http://localhost:8000"

def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

result = post("/decide/election", {
    "context": "Maine Senate: poll aggregate D+2, betting markets 54% Dem."
})
print(result["answers"]["outcome"])   # {"choice": "democrat_win"}
```

---

## Smoke test

With the server running:

```bash
python smoke.py                 # localhost:8000
python smoke.py http://host:8000
```

Exercises each example preset and prints the JSON.

---

## Honest caveats

- This serves a checkpoint that ships **uncalibrated**: the choice head and
  score head can disagree with each other, and the model can show a
  systematic lean. In one 23-race election benchmark: 16/23 choice agreement
  with an independent judge model, all 7 misses leaning the same direction;
  0/2 on a trade-filter task. Calibrate (temperature fitting) per domain
  before trusting probabilities.
- CPU latency is seconds per call: fine for batch jobs and scheduled runs,
  not for realtime.

---

## Acknowledgments

Built on the original Laya model by Convai Innovations. The
`agk4444/laya-typed-decisions` fine-tune was developed by AGK FIRE INC.

© 2026 AGK FIRE INC. Released under Apache 2.0.

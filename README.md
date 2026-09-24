# Laya CPU API v1.0.0

Serves the Laya System-1 decision model as a small HTTP API on CPU.
Model loads once at startup (~800MB English checkpoint), then each
`POST /decide` takes ~4–8s on CPU.

## Run on Colab (from your phone)

New notebook, CPU runtime, then:

```
!pip -q install fastapi uvicorn laya
!git clone https://github.com/agk4444/laya-api
!python laya-api/server.py --model agk4444/laya-typed-decisions --port 8000
```

First run downloads the ~808MB checkpoint from Hugging Face
(`agk4444/laya-typed-decisions`, fine-tuned on the typed-decisions benchmark).
For the original base checkpoint: `--model convaiinnovations/laya`.
For the multilingual checkpoint: `--model convaiinnovations/laya --subfolder multilingual`.

Note: Colab's port isn't public — the API is for code running in the
same notebook / runtime. To call it from the notebook while the server
runs, start the server in the background (`!nohup python server.py &`)
then use the client examples below with `http://localhost:8000`.

## Run anywhere else

```
pip install -r requirements.txt
python server.py --model /path/to/local/checkpoint
```

Options: `--device cuda` if a GPU is available, `--host`, `--port`,
`LAYA_MODEL` / `LAYA_SUBFOLDER` env vars instead of flags.

## Endpoints

- `GET /health` → `{"ok": true, "model": ..., "device": "cpu"}`
- `POST /decide` — generic typed decision:
  ```json
  {"state": {"context": "..."},
   "questions": {"q1": {"type": "choice", "instructions": "...",
                        "criteria": {"a": "...", "b": "..."}}}}
  ```
  Question types: `choice` (criteria = option dict), `score`
  (criteria = level list, normalized to `score_0_100`), `noul`
  (returns `noul` 0–1).
- `POST /decide/chunav` — preset Chunav schema: `{"context": "..."}`
  → `outcome` (choice), `dem_win_probability` (score 0–100), `confidence` (noul)
- `POST /decide/options` — preset options-filter schema: `{"state": "..."}`
  → `action`, `direction`, `conviction` (0–100), `confidence`

All POST responses include `latency_ms`.

## Smoke test

With the server running:

```
python smoke.py                 # localhost:8000
python smoke.py http://host:8000
```

Runs one Chunav call and one options-filter call and prints the JSON.

## Honest caveats

- This serves the **base checkpoint**, which ships uncalibrated: the choice
  head and score head can disagree with each other, and the base model has a
  systematic lean (see the 2026-09-21 bake-off: 16/23 Chunav choice agreement
  with Jev, all 7 misses leaning Dem; 0/2 on the options filter). Calibrate
  (temperature fitting) per domain before trusting probabilities.
- CPU latency is seconds per call: fine for batch jobs (e.g. the 2-hourly
  Chunav run), not for realtime.

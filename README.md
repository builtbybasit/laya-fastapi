# laya-fastapi

A local, open-source drop-in for TypeSafe's [Jev](https://docs.typesafe.ai) `POST /v1/systemone` API, powered by [Laya](https://github.com/NandhaKishorM/laya). Send content plus typed questions, get structured answers with probabilities and confidence. Request logs are rendered with [Rich](https://github.com/Textualize/rich).

## Backends

The backend is picked by platform at install time (`uv sync`) and at startup:

| Platform                   | Package                                               | Device                      |
| -------------------------- | ----------------------------------------------------- | --------------------------- |
| macOS, Apple Silicon       | [`laya-mlx`](https://github.com/mizorewww/laya-mlx)   | Metal GPU (MLX)             |
| Linux / Windows / Intel Mac | [`laya`](https://github.com/NandhaKishorM/laya) (PyTorch) | CUDA → XPU → CPU (auto) |

On Linux, `uv sync` installs CUDA-enabled PyTorch from PyPI, which needs a recent NVIDIA driver (CUDA 13). For an older driver, or for CUDA on Windows (PyPI's Windows torch is CPU-only), install a matching build first: `uv pip install torch --torch-backend=auto`.

## Run

```bash
uv sync
uv run main.py

# or with settings from a file
cp .env.example .env
uv run --env-file .env main.py
```

Open http://127.0.0.1:8000/docs for the interactive API docs.

| Env var          | Default                                              | Meaning                              |
| ---------------- | ---------------------------------------------------- | ------------------------------------ |
| `LAYA_BACKEND`   | `mlx` on Apple Silicon, else `torch`                 | Force a backend                      |
| `LAYA_MODEL`     | `aac6fef/laya-mlx` (mlx), `convaiinnovations/laya` (torch) | Hugging Face repo or local path |
| `LAYA_DEVICE`    | auto                                                 | e.g. `cuda`, `cuda:1`, `cpu`, `mps`  |
| `LAYA_HOST`      | `127.0.0.1`                                          | Bind address                         |
| `LAYA_PORT`      | `8000`                                               | Bind port                            |
| `LAYA_LOG_LEVEL` | `info`                                               | `debug` also logs `/health`          |

## API

### `POST /v1/systemone`

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "jev-latest",
  "state": "I was charged twice this month. Please refund the extra charge.",
  "questions": {
    "billing":    {"type": "noul",   "instructions": "Is this message about billing?"},
    "tone":       {"type": "choice", "instructions": "What is the tone?",
                   "criteria": {"angry": "Upset or hostile", "calm": "Neutral or polite"}},
    "department": {"type": "choice", "instructions": "Who should handle this?",
                   "criteria": ["billing", "technical", "sales"]},
    "urgency":    {"type": "score",  "instructions": "How urgent?",
                   "criteria": ["Can wait", "This week", "Today"]}
  }
}'
```

| Question | `criteria`                                        | Answer fields                                 |
| -------- | ------------------------------------------------- | --------------------------------------------- |
| `noul`   | optional `{"true": "...", "false": "..."}`        | `noul` = P(yes)                               |
| `choice` | `{name: description}` or `[name, ...]`, 1–255     | `choice`, `probabilities`, `confidence`       |
| `score`  | ordered list, 1–10 levels (index = score)         | `score` (expected level), `legend`, `probabilities`, `confidence` |

`state` can be a string, an object, or an array. Answers are keyed by your question names:

```json
{
  "model": "laya",
  "answers": {
    "billing": {"type": "noul", "noul": 0.94, "confidence": 0.94},
    "department": {"type": "choice", "choice": "billing", "confidence": 0.78,
                   "probabilities": {"billing": 0.95, "technical": 0.03, "sales": 0.02}}
  },
  "usage": {"input_tokens": 181, "output_tokens": 0}
}
```

Errors: `422` for invalid requests (with the reason in `detail`).

### `GET /health`

`{"status": "ok", "model": "laya", "backend": "mlx"}`

## Logs

```
[10:23:25] INFO  inference 240ms · state 83 chars · 4 q · tok 181/0
           INFO     billing noul → 0.94 conf 0.94
           INFO     department choice → billing conf 0.78
           INFO     urgency score → 1.81 conf 0.52
           INFO  POST /v1/systemone 200 244.6ms
```

Confidence is colored green (≥ 0.8), yellow (≥ 0.5) or red. The `state` text is never logged, only its length.

## Differences from Jev

- No `box` questions or image `assets` (text-only model).
- `instructions` is required on every question (optional in Jev).
- `confidence` is Laya's normalized-entropy score, not Jev's formula — re-tune any thresholds ported from Jev.
- Answers carry extra fields (`answer_confidence`, `action`); Jev clients ignore them.
- No auth, `GET /v1/models`, or batch endpoint. The server binds to localhost by default.
- One inference runs at a time (single model, guarded by a lock).

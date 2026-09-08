# Running TalkTuner locally

All paths below are relative to the repository root unless stated otherwise.

## 1. What TalkTuner App includes

The app is five separate processes. The Python services communicate only through MongoDB and HTTP, so each one needs its own terminal.

| # | Service | Working directory | Port | Environment |
|---|---------|-------------------|------|-------------|
| 1 | MongoDB | — | 27017 | system service |
| 2 | Model backend | `dashboard_v1/` | 8505 | `.venv-backend` |
| 3 | Queue server | `dashboard_v1/dashboard_task_queue/queuer/` | 8510 | `.venv-queue` |
| 4 | Worker | `dashboard_v1/dashboard_task_queue/` | — | `.venv-queue` |
| 5 | React dashboard | `dashboard_v1/dashboard/` | 3000 | Node |

- MongoDB is used for storing user credentials (login token) and chat history persistently
- Model backend or /backend is used for handling all generation, probing, and intervention requests
- Queue server is used for queuing the incoming requests made by the frontend and insert the ready-to-be-complete request to MongoDB database. Without a queue, multiple users can easily overload the backend if they interact with the App at the same time.
- Worker pulling jobs from MongoDB and forwarding them to the backend for processing

Optional extras:
One optional extra, only needed if you want to train your own probes from the dashboard's "Add Probe" modal:

| # | Service | Working directory | Port | Environment |
|---|---------|-------------------|------|-------------|
| 6 | Probing server | `dashboard_v1/probing/` | 5001 | `.venv-backend` |

Request flow for a chat message:

```
React (3000) --POST /chat--> queue server (8510) --insert--> MongoDB
                                                                |
                                                worker polls ---+
                                                                |
                                           worker --POST /chat--> model backend (8505)
                                                                |
React polls GET /status/<task_id> <---- worker writes result to MongoDB
```

The queue server accepts a chat request and returns a task ID immediately, so no HTTP request stays open while the model generates. The worker picks the job up, calls the model backend, and writes the result back to MongoDB, where the frontend's polling finds it.

---

## 2. Prerequisites

- **Python 3.9–3.11**
- **Node.js 18+** and npm
- **MongoDB Community Server** — install from your distribution's package repository or https://www.mongodb.com/try/download/community
- **NVIDIA GPU with CUDA 12.1** — the backend loads Llama-3.1-8B and Gemma-2-9B in bf16, roughly 18 GB of VRAM each. See section 4.1 if you have a single GPU.
- **Hugging Face account** with granted access to `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-2-9b-it`. Both are gated repositories — request access on huggingface.co before you start, since approval is not instant.
- `unzip`, for the bundled probe checkpoints

---

## 3. Quick start

Below is quick start for launching the TalkTuner Web App. Run this from the repository root. It assumes MongoDB is installed and the default ports (3000 -> frontend, 8505 -> backend, 8510 -> task queue, 5001 -> probing server, and 27017 -> MongoDB) are free; if you need to change anything, read section 4 first.

```bash
# =============================================================
# ONE-TIME SETUP
# =============================================================

cd dashboard_v1

# If you ever encounter an error when installing the dependencies, try remove the torch, torchvision, and torchaudio dependencies from the requirements file and install them using `pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121` instead.

# --- Python environment for the model backend (and probing server) ---
python3 -m venv .venv-backend
source .venv-backend/bin/activate
python -m pip install --upgrade pip
pip install -r backend_requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Authenticate with Hugging Face so the gated models can download.
# Paste an access token from https://huggingface.co/settings/tokens
huggingface-cli login
deactivate

# If you ever encounter an error when installing the dependencies, try remove the torch, torchvision, and torchaudio dependencies from the requirements file and install them using `pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121` instead.

# --- Python environment for the queue server and worker ---
python3 -m venv .venv-queue
source .venv-queue/bin/activate
python -m pip install --upgrade pip
pip install -r queue_requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
deactivate

# --- Probe checkpoints ---
# Each archive expands into a directory of the same name, which is
# where the backend expects to find it.
unzip -q llama3_read_probes.zip
unzip -q llama3_control_probes.zip
unzip -q gemma2_read_probes.zip
unzip -q gemma2_control_probes.zip

# --- Directory the backend uses to store conversations ---
mkdir -p chat_history

# --- Frontend dependencies ---
cd dashboard
npm install
cd ..

# =============================================================
# LAUNCHING  (one terminal per service, started in this order)
# =============================================================

# ---------- Terminal 1: MongoDB ----------
sudo systemctl start mongod
# Not installed as a service? Run it in the foreground instead:
#   mongod --dbpath ~/mongodb-data

# ---------- Terminal 2: model backend, port 8505 ----------
# Must be launched from dashboard_v1/, not from backend/.
cd dashboard_v1
source .venv-backend/bin/activate
python backend/run.py --port 8505
# Loading two 8-9B models takes several minutes. Wait for the line
# "finish loading model" before continuing to Terminal 3.

# ---------- Terminal 3: queue server, port 8510 ----------
cd dashboard_v1/dashboard_task_queue/queuer
source ../../.venv-queue/bin/activate
python run.py --port 8510

# ---------- Terminal 4: worker ----------
cd dashboard_v1/dashboard_task_queue
source ../.venv-queue/bin/activate
bash restart_worker.sh

# ---------- Terminal 5: dashboard, port 3000 ----------
cd dashboard_v1/dashboard
npm start

# ---------- Terminal 6 (optional): probing server, port 5001 ----------
# Loads a second copy of both models. Skip unless you want to train
# custom probes from the "Add Probe" modal.
cd dashboard_v1/probing
source ../.venv-backend/bin/activate
bash start_server.sh
```

Then open http://localhost:3000 and enter the access token from section 7.

---

## 4. Configurable settings

The defaults work for a single-machine setup. Change these only if your environment differs.

### 4.1 GPU assignment

`dashboard_v1/backend/app/routes/index.py` places the two models on separate GPUs — line 100 uses `device_map={"": 1}` for Llama-3 and line 106 uses `device_map={"": 0}` for Gemma-2.

On a single-GPU machine, change line 100 to `{"": 0}`. Both models then share one device and need roughly 36 GB of VRAM. If that doesn't fit, comment out the Gemma-2 loading block (lines 105–107) along with the `gemma2_*` classifier dictionaries below it, and run with Llama-3 only.

The probing server (service 6) reads `torch.cuda.is_available()` and places its own copy of both models on the default device, so running it alongside the backend roughly doubles VRAM usage.

### 4.2 Service URLs

`dashboard_v1/dashboard/src/helpers/constants.js`, lines 19–21, tell the browser where the backends live:

```js
const BACKEND_ADDR = "http://localhost:8505";
const QUEUE_BACKEND_ADDR = "http://localhost:8510";
const PROBING_API_URL = "http://localhost:5001";
```


`dashboard_v1/dashboard_task_queue/worker/worker.py`, line 16, tells the worker where the model backend is:

```python
backend_api = 'http://localhost:8505'
```

Change this if the worker and the model backend run on different machines.

### 4.3 Ports

| Service | How to change |
|---------|---------------|
| Model backend | `python backend/run.py --port 8505` |
| Queue server | `python run.py --port 8510` |
| Probing server | `PORT=5001 bash start_server.sh` |
| Dashboard | `PORT=3000 npm start` |
| MongoDB | `mongod --port 27017`, plus the `MongoClient` calls in `queuer/app/routes/index.py` line 15 and `worker/worker.py` line 9 |

If you change 8505, 8510, or 5001, update the matching entries in `constants.js` (section 4.2) and, for 8505, in `worker.py`.

### 4.4 Probing server concurrency

`MAX_CONCURRENT_TASKS` caps how many probe-training jobs run at once, defaulting to 2:

```bash
MAX_CONCURRENT_TASKS=2 bash start_server.sh
```

---

## 5. Dependencies in detail

### 5.1 Backend environment (services 2 and 6)

The probing server shares this environment; `backend_requirements.txt` already covers its `transformers`, `torch`, and `openai` needs.

```bash
cd dashboard_v1
python3 -m venv .venv-backend
source .venv-backend/bin/activate
python -m pip install --upgrade pip
pip install -r backend_requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

The `--extra-index-url` is required: `torch==2.2.2+cu121` and its matching `torchvision` and `torchaudio` builds are not on plain PyPI.

### 5.2 Queue environment (services 3 and 4)

```bash
cd dashboard_v1
python3 -m venv .venv-queue
source .venv-queue/bin/activate
python -m pip install --upgrade pip
pip install -r queue_requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

Two notes on this file. It pins `asyncio==3.4.3`, an obsolete PyPI backport that installs a top-level `asyncio` module shadowing the standard library one; if you see unexpected `asyncio` import errors, run `pip uninstall asyncio` and the stdlib version takes over. It also pulls in `torch`, which neither the queue server nor the worker imports — you can drop the `torch`, `torchvision`, `torchaudio`, `nvidia-*`, and `triton` lines to save several GB.

### 5.3 Frontend

```bash
cd dashboard_v1/dashboard
npm install
```

---

## 6. Probes and data directories

### 6.1 Probe checkpoints

Four archives ship with the repository in `dashboard_v1/`:

```
llama3_read_probes.zip      llama3_control_probes.zip
gemma2_read_probes.zip      gemma2_control_probes.zip
```

Each expands into a directory of the same name, so extract them in place:

```bash
cd dashboard_v1
unzip -q llama3_read_probes.zip
unzip -q llama3_control_probes.zip
unzip -q gemma2_read_probes.zip
unzip -q gemma2_control_probes.zip
```

The result must be four sibling directories inside `dashboard_v1/`:

```
dashboard_v1/llama3_read_probes/
dashboard_v1/llama3_control_probes/
dashboard_v1/gemma2_read_probes/
dashboard_v1/gemma2_control_probes/
```


The backend prints `finish loading model` once weights and probes are both in place. Check for that line before using the dashboard.

Probes you train yourself through the probing server land in `dashboard_v1/{llama3,gemma2}_{control,read}_probes_extra/` and are picked up automatically.

### 6.2 Chat history directory

The backend reads and writes `chat_history/` relative to its working directory and does not create it:

```bash
mkdir -p dashboard_v1/chat_history
```

Without it, `/id` returns 500 and the frontend never receives a session ID.

---

## 7. Using the dashboard

Open http://localhost:3000. A welcome modal asks for an access token; the default is the literal string

```
Global Token
```

typed exactly as shown, including capitals. It is validated by the queue server's `/auth` endpoint and stored in a cookie for 7 days.

Conversations are logged to MongoDB as they happen — one document per session in `dashboardAI_db.chats` and one per exchange in `dashboardAI_db.messages`, including the probe readings shown in the dashboard. Inspect them with `mongosh` or any MongoDB client.

### Training custom probes (optional)

The "Add Probe" modal needs the probing server (service 6) running, and asks for your own OpenAI API key, which it uses to generate the synthetic conversations the new probe is trained on. The key is kept in browser local storage and sent to the probing server with each request; it is not stored server-side.

---

## 8. Troubleshooting

**Frontend loads but the dashboard stays empty, or `/id` fails.** `dashboard_v1/chat_history/` is missing (section 6.2), or the model backend isn't running on the port `constants.js` points at.

**Chat messages queue forever and never come back.** The worker isn't reaching the model backend. Confirm the worker terminal is running, that `backend_api` in `worker.py` matches the backend's actual address, and that the backend is serving on 8505.

**Backend starts but every chat returns 500.** Model loading failed. Scroll to the top of the backend terminal: if you see an exception instead of `finish loading model`, either Hugging Face authentication failed or the probe directories weren't extracted (section 6.1).

**`KeyError` while loading probes.** A `.pth` filename's prefix isn't a recognized attribute. The text before the first underscore must be one of the keys in the `num_classes` dictionary in `backend/app/chat/classifiers.py` (lines 62–86).

**`CUDA out of memory`.** See section 4.1 — put both models on one GPU only if you have the VRAM, otherwise run Llama-3 alone. Also check whether the probing server is running and holding a second copy of the weights.

**403 or gated-repo errors on startup.** Your Hugging Face account hasn't been granted access to `meta-llama/Llama-3.1-8B-Instruct` or `google/gemma-2-9b-it`, or `huggingface-cli login` was run in a different environment than the backend.

**CORS errors in the browser console.** Both Flask apps enable CORS for all origins, so this usually means the service isn't running at all and you're seeing a failed preflight rather than a real CORS rejection.

**Port already in use.** All three Python services bind `0.0.0.0`, so a stale process keeps the port held. Find it with `lsof -i :8505` (or `:8510`, `:5001`) and stop it.

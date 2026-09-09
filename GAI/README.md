# VNX Functional Music GAI — local service

The notebook, minus Colab, as a resident FastAPI service. Models load once at
startup and stay on the GPU; requests queue and render one at a time.

```
config.py     paths, model ids, generation + EQ constants
pipeline.py   notebook sections 5-9; entry point is generate_session()
server.py     FastAPI + single-worker job queue
static/       one HTML page, no build step
```

## Setup

```bash
# 1. torch trio first, matched set, cu126
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 2. flash-attn — prebuilt wheel for cu126 / torch2.7 / your Python version
#    (building from source takes 1-2 hours)

# 3. stable-audio-3 from Stability's repo
pip install -e path/to/stable-audio-3-main

# 4. the rest
pip install -r requirements.txt
```

Accept the gated terms in a browser for `stabilityai/stable-audio-3-medium`
and `stabilityai/SAME-L`, then set your token:

```powershell
setx HF_TOKEN hf_xxx          # Windows, reopen the terminal after
setx VNX_ROOT D:\VNX_GAI      # optional — weights are ~9 GB
```

## Run

```bash
python server.py
```

First start downloads ~9 GB and takes a few minutes; the page shows
`loading models` until both are resident. After that, open
`http://localhost:8000`.


## Notes

Segment `.wav` files are reused from disk if they already exist, same as the
notebook — delete one to force a re-render of just that segment. Sessions land
in `$VNX_ROOT/sessions/<slug>/`, with `completed.wav` raw and
`completed_EQed.wav` after post EQ.

`body_db` defaults to 0.0. The 90-130 Hz Body Melody boost is a per-product
decision, not a pipeline default — set it in `config.POST_EQ` when you want it.

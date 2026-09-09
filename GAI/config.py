"""Paths, environment and tunables.

Imported first by everything else — HF_HOME must be set before transformers or
huggingface_hub is imported anywhere, or the cache lands in the default
location instead of ROOT/hf_cache.
"""
import os
from pathlib import Path

# ---------- paths ----------
# Override with VNX_ROOT if you want weights/sessions somewhere other than
# alongside the code (e.g. a second drive with more space — weights are ~9 GB).
ROOT = Path(os.environ.get('VNX_ROOT', Path(__file__).parent / 'data'))
HF_HOME = ROOT / 'hf_cache'
WORK = ROOT / 'sessions'

for d in (HF_HOME, HF_HOME / 'hub', WORK):
    d.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(HF_HOME)
os.environ['HF_HUB_CACHE'] = str(HF_HOME / 'hub')

# Both repos are gated. Set HF_TOKEN in the environment (PowerShell:
# setx HF_TOKEN hf_xxx   /   bash: export HF_TOKEN=hf_xxx) — never in a file
# that goes into git.
if not os.environ.get('HF_TOKEN'):
    raise RuntimeError(
        'HF_TOKEN not set. Accept the terms at '
        'huggingface.co/stabilityai/stable-audio-3-medium and .../SAME-L, '
        'then set HF_TOKEN in your environment.'
    )

# ---------- models ----------
# AWQ int4 instead of the notebook's bf16: ~5.5 GB instead of ~15 GB, which is
# what lets Qwen and Stable Audio 3 both stay resident on a 16 GB card. Without
# this you'd have to unload one to load the other on every request.
QWEN_MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'
SA3_MODEL = 'medium'

# ---------- generation ----------
STEPS = 8                     # medium is post-trained; 8 is the tuned default
MAX_MODEL_DURATION_S = 360.0  # per-call cap we target (model's hard limit is 380)
RANDOM_SEED = -1              # -1 = random. Set an integer to reproduce a take.
CTX_S = 30.0                  # seconds of the previous phase used as conditioning

DEFAULT_TOTAL_S = 15 * 60     # fallback if no duration is mentioned in the request
SPLIT = (0.25, 0.35, 0.40)    # match, transition, settle

# ---------- post EQ ----------
POST_EQ = dict(
    hpf_hz=25, body_lo=90, body_hi=130, body_db=0.0,
    mud_lo=250, mud_hi=400, mud_db=-1.0,
    air_hz=12000, air_db=-0.5,
    target_lufs=-16.0, peak_db=-1.0,
)

# ---------- server ----------
HOST = '0.0.0.0'   # bind wide; reachable only over Tailscale unless you forward a port
PORT = 8000

"""Paths, environment and tunables.

Imported first by everything else — HF_HOME must be set before transformers or
huggingface_hub is imported anywhere, or the cache lands in the default
location instead of ROOT/hf_cache.
"""
import os
from pathlib import Path

# ---------- paths ----------
# Override with RESONALIFE_ROOT if you want weights/sessions somewhere other than
# alongside the code (e.g. a second drive with more space — weights are ~9 GB).
ROOT = Path(os.environ.get('RESONALIFE_ROOT', Path(__file__).parent / 'data'))
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

# ---------- pain relief mode ----------
# Functional Music Guideline — Hospice & Chronic Illness, Module 1 (Pain Relief).
# Unlike match/transition/settle this arc never opens with energy to "meet" the
# listener: it starts calm and only ever gets calmer. Shares are the guideline's
# session arc table; every phase is chunked across as many model calls as it
# needs, so the shares hold at any length. The names are also the prompt keys,
# the segment file names and the phase ids the page colours by.
PAIN_ARC = (
    ('immediate_calm',     0.10),   # enters softly at the slow tempo, no build-up
    ('release_tension',    0.15),   # harmony settles further, phrases lengthen
    ('comfort_and_safety', 0.20),   # stable, repeating, the most "held" part
    ('deepen_relaxation',  0.20),   # fewer notes, more space, darker and softer
    ('weightless_plateau', 0.25),   # pads and sustained tones float
    ('fade_or_loop',       0.10),   # fade to silence, or continue and loop
)
assert abs(sum(share for _, share in PAIN_ARC) - 1.0) < 1e-9, 'PAIN_ARC shares must add up to 1'

# guideline "Tailoring a session" safe ranges — requests outside them are pulled back in
PAIN_MINUTES = (10, 60)
PAIN_DEFAULT_MINUTES = 30
PAIN_BPM = (50, 65)
PAIN_DEFAULT_BPM = 58

PAIN_LOOP_CROSSFADE_S = 8.0   # loop ending: tail folded into the start so the file repeats with no click or gap
PAIN_LOOP_CHECK_S = 10.0      # loop ending: seconds either side of the join in loop_check.mp3

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

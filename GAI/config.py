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

# ---------- guideline modules ----------
# Functional Music Guideline — Hospice & Chronic Illness. Each module's session
# arc is a list of (phase, share of the session). The shares are the guideline's
# session arc tables; every phase is chunked across as many model calls as it
# needs, so the shares hold at any length. Phase names are also the prompt keys,
# the segment file names and the phase ids the page colours by, so they are
# unique across every arc.

# ---------- Module 1: pain relief ----------
# Unlike match/transition/settle this arc never opens with energy to "meet" the
# listener: it starts calm and only ever gets calmer.
PAIN_ARC = (
    ('immediate_calm',     0.10),   # enters softly at the slow tempo, no build-up
    ('release_tension',    0.15),   # harmony settles further, phrases lengthen
    ('comfort_and_safety', 0.20),   # stable, repeating, the most "held" part
    ('deepen_relaxation',  0.20),   # fewer notes, more space, darker and softer
    ('weightless_plateau', 0.25),   # pads and sustained tones float
    ('fade_or_loop',       0.10),   # fade to silence, or continue and loop
)

# guideline "Tailoring a session" safe ranges — requests outside them are pulled back in
PAIN_MINUTES = (10, 60)
PAIN_DEFAULT_MINUTES = 30
PAIN_BPM = (50, 65)
PAIN_DEFAULT_BPM = 58

# ---------- Module 2: dementia ----------
# Default arc: familiar -> comfortable -> gently engaging, then settles. Calming
# mode (anxiety, agitation or restlessness) reduces stimulation step by step.
# Reminiscence and Social mode change the style, not the arc, so they use the default.
DEMENTIA_ARCS = {
    'default': (
        ('familiar',               0.15),   # a simple, familiar-sounding phrase; nothing new or surprising
        ('safe',                   0.15),   # warm, consonant harmony, even volume; the carrying pattern is set up
        ('comfortable',            0.25),   # stable and predictable; phrases return with subtle variation
        ('engaging',               0.30),   # clear, memorable phrases invite humming or tapping along
        ('emotionally_supportive', 0.15),   # instruments thin out, the pace eases; a soft close or slow fade
    ),
    'calming': (
        ('steady_and_familiar',  0.15),     # steady tempo, simple familiar melody, few instruments
        ('simplify_rhythm',      0.15),     # rhythmic complexity drops; piano or guitar carries the pulse
        ('fewer_instruments',    0.20),     # down to one or two instruments; phrases repeat more often
        ('calming_settle',       0.20),     # tempo stabilizes, harmony moves less, dynamics even out
        ('spacious_calm',        0.20),     # more space, long sustained tones, one short repeated phrase
        ('calming_fade_or_loop', 0.10),     # slow fade to silence, or a seamless loop
    ),
}
DEMENTIA_MINUTES = (10, 60)
DEMENTIA_DEFAULT_MINUTES = 30
DEMENTIA_BPM = (45, 80)                 # the safe range, across every mode and time of day
DEMENTIA_DEFAULT_BPM_RANGE = (50, 75)   # "50–75 BPM by default", when no line below sets its own
DEMENTIA_DEFAULT_BPM = 65
# the ranges a Session mode or Time of day line sets; both apply when both are set
DEMENTIA_BPM_BY = {
    'calming':  (45, 65),
    'evening':  (45, 65),
    'presleep': (45, 65),
    'social':   (60, 80),
    'morning':  (60, 80),
}

# ---------- Module 3: depression ----------
# Starts where the listener is and only then moves; never forces a happy ending.
# Comfort is the default mode; each Session mode line switches the arc.
DEPRESSION_ARCS = {
    'comfort': (
        ('acknowledge_heaviness', 0.20),    # quiet, slow, a little subdued: sparse piano or pads
        ('safety',                0.25),    # warm, consonant chords; a soft, steady pulse
        ('warmth',                0.30),    # warm strings or guitar join; harmony opens gently
        ('calm',                  0.25),    # density eases back into long sustained tones
    ),
    'activation': (
        ('stillness',              0.20),   # minimal rhythm at the starting tempo
        ('comfort',                0.25),   # a gentle pulse begins
        ('engagement',             0.30),   # clearer pulse, more melodic movement
        ('subtle_positive_energy', 0.25),   # ends at a steady, gentle energy
    ),
    'relaxation': (
        ('tension',         0.15),          # meets restlessness with some rhythmic activity
        ('relaxation_calm', 0.25),          # rhythm and density begin to reduce
        ('spaciousness',    0.30),          # more space, fewer notes
        ('deep_relaxation', 0.30),          # minimal rhythm, long tones, slow fade
    ),
    'sleep': (
        ('emotional_fatigue', 0.10),        # slow and gentle, meeting tiredness
        ('sleep_safety',      0.20),        # warm, stable harmony
        ('sleep_calm',        0.25),        # calm, sustained tones
        ('low_stimulation',   0.25),        # very little movement
        ('sleep_readiness',   0.20),        # holds steady and loops back to the start
    ),
}
DEPRESSION_MINUTES = (10, 60)
DEPRESSION_DEFAULT_MINUTES = 30
DEPRESSION_BPM = (50, 65)               # comfort, relaxation and sleep; where activation starts
DEPRESSION_DEFAULT_BPM = 58
DEPRESSION_ACTIVATION_BPM = (65, 80)    # where Gentle activation ends, reached gradually
DEPRESSION_DEFAULT_ACTIVATION_BPM = 70

for _arc in (PAIN_ARC, *DEMENTIA_ARCS.values(), *DEPRESSION_ARCS.values()):
    assert abs(sum(share for _, share in _arc) - 1.0) < 1e-9, f'arc shares must add up to 1: {_arc}'
_names = [name for _arc in (PAIN_ARC, *DEMENTIA_ARCS.values(), *DEPRESSION_ARCS.values()) for name, _ in _arc]
assert len(_names) == len(set(_names)) and not {'match', 'transition', 'settle'} & set(_names), \
    'phase names must be unique across all arcs'

# ---------- endings (all modules) ----------
FADE_SHARE = 0.10         # fade: the last 10% of the session fades to exact silence (pain relief's Fade or loop phase)
CLOSE_FADE_S = 8.0        # dementia soft close: a short tail after the resolved phrase, so it never cuts off
LOOP_CROSSFADE_S = 8.0    # loop: tail folded into the start so the file repeats with no click or gap
LOOP_CHECK_S = 10.0       # loop: seconds either side of the join in loop_check.mp3

# ---------- fixing one part (inpainting) ----------
# A finished session can have parts regenerated: Stable Audio 3 gets the music
# around each part as context and rewrites only the part (its inpainting mask).
# One call holds a part plus INPAINT_CTX_S of music on each side, and a call is
# at most MAX_MODEL_DURATION_S, so the longest part is 360 - 2 x 30 = 300 s.
INPAINT_CTX_S = CTX_S                                       # music kept on each side of a part, as context
INPAINT_MAX_S = MAX_MODEL_DURATION_S - 2 * INPAINT_CTX_S    # the longest part: 5 min
INPAINT_MIN_S = 5.0          # shorter parts come back almost unchanged: the context outweighs the prompt
INPAINT_MAX_PARTS = 4        # parts per request; parts close together share one call
INPAINT_XFADE_S = 1.5        # crossfade into and out of each part, inside the music kept around it

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

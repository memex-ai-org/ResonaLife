"""Notebook sections 5-9 as a resident service.

Both models load once at process start and stay on the GPU. Nothing here
touches Colab, Drive, or `!pip`. The only entry point the server uses is
generate_session(), which runs one of two pipelines:

  iso          match -> transition -> settle, three prompts (the original)
  pain_relief  the Hospice & Chronic Illness guideline's Pain Relief arc:
               six phases that start calm and only ever get calmer
"""
import config  # noqa: F401  — must come first, sets HF_HOME before transformers

import gc
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt
import pyloudnorm as pyln

# ============================================================
# Model loading — called once, at server startup
# ============================================================

qwen_model = None
qwen_tokenizer = None
model = None      # Stable Audio 3
SR = None         # native sample rate, set once SA3 is loaded


def load_qwen():
    """Load the orchestrator on demand. Called once per session, then unloaded.

    Qwen writes the phase prompts and is then dead weight for the rest of the run,
    so on a 16 GB card it's worth the ~10s reload to get its ~2.5 GB back
    before diffusion starts.
    """
    global qwen_model, qwen_tokenizer

    if qwen_model is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    qwen_tokenizer = AutoTokenizer.from_pretrained(config.QWEN_MODEL_ID)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        config.QWEN_MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map='cuda:0',
    )
    qwen_model.eval()
    print(f'qwen loaded 4-bit ({time.time()-t0:.0f}s)', flush=True)


def unload_qwen():
    """Drop the orchestrator and return its VRAM before generation starts."""
    global qwen_model, qwen_tokenizer

    qwen_model = None
    qwen_tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f'qwen unloaded — VRAM {(total-free)/1e9:.1f} / {total/1e9:.1f} GB', flush=True)


def load_models():
    """Load Stable Audio 3 and leave it resident.

    SA3 plus its T5-Gemma text encoder is ~10.8 GB and stays on the GPU for the
    life of the process. Qwen is loaded separately, per session — see load_qwen.
    """
    global model, SR

    if model is not None:
        return

    # flash-attn is required by medium: its SAME-L autoencoder uses
    # sliding-window attention, and the SDPA fallback is not numerically
    # equivalent. Without it the output is noise, not slightly worse audio.
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        raise RuntimeError(
            'flash_attn is not installed. Stable Audio 3 medium produces noise '
            'without it. See requirements.txt for the wheel to install.'
        )

    from stable_audio_3 import StableAudioModel

    t0 = time.time()
    print('loading stable-audio-3 (first run downloads ~10 GB)...', flush=True)
    model = StableAudioModel.from_pretrained(config.SA3_MODEL)
    SR = model.model.sample_rate
    print(f'loaded stable-audio-3 {config.SA3_MODEL}  SR={SR}  ({time.time()-t0:.0f}s)', flush=True)

    assert SR, 'SR not set — StableAudioModel did not load'

    free, total = torch.cuda.mem_get_info()
    print(f'VRAM {(total-free)/1e9:.1f} / {total/1e9:.1f} GB used', flush=True)


# ============================================================
# Section 5 — orchestration
# ============================================================

# Everything the orchestrator is allowed to control: prompt text and nothing
# else (durations/seeds are Python's job). The tag format below must match what
# the generation code expects the prompts to look like.
ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestration layer of a functional-music \
generation pipeline. Given a user's session request, you decompose it into three \
text-to-audio prompts that together form one ISO-principle session:

  match       — matches the listener's presumed current state
  transition  — bridges from the current state toward the target state
  settle      — reflects the target state the session is meant to leave the listener in

Each prompt must:
- follow this exact tag-plus-description format:
  "Purpose: <purpose (NOT Transition, Match, Settle)>, TrackType: Music, VocalType: Instrumental, Genre: <genre1>, \
Genre: <genre2>, Instruments: <comma-separated instruments>. <one-paragraph \
descriptive continuation covering tempo/BPM, instrumentation behavior, and mood>, \
Addition Request: <add any more request that the prompt has that doesn't fit in the \
other sections>"
-Purpose is what the music is used for such as meditation, stress relief, etc and NOT
  what segment it is (match, settle, etc)
-Accommodate their needs in the prompt such as asking for specific sound
  or instrumentation, noise, etc (such as adding rain drop noise when they ask for rain sounds).
  if it doesn't fit in the other sections, add it to Addition Request.
- avoid hooks, choruses, dynamic contrast, or any sense of arrival — functional music \
is defined by what it withholds.
- carry a BPM (or explicit "no discernible tempo" for the settle phase) that \
descends from match to settle, appropriate to the requested session type.
- read as a continuation of the same piece, not three different songs — keep \
instrumentation and sonic palette consistent across all three while it evolves.

Do not decide durations — that is handled elsewhere. Respond with ONLY a JSON object:
{"match": "...", "transition": "...", "settle": "..."}"""


def ask_qwen(user_message, system_prompt, max_new_tokens=900):
    """One sampled reply from Qwen to user_message under system_prompt, as raw text."""
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_message},
    ]
    text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tokenizer(text, return_tensors='pt').to(qwen_model.device)
    with torch.inference_mode():
        out = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=qwen_tokenizer.eos_token_id,   # Qwen has no pad token by default
        )
    return qwen_tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def generate_phase_prompts(session_request, attempts=3):
    """Call Qwen until it returns valid {match, transition, settle} JSON, or give up.

    Instruct models occasionally wrap JSON in prose or markdown fences, or drop
    a key — retrying a couple of times is cheap and far more robust than
    failing the whole session on one bad sample.
    """
    last_error = None
    for attempt in range(1, attempts + 1):
        reply = ask_qwen(f'Session request: "{session_request}"', ORCHESTRATOR_SYSTEM_PROMPT)
        found = re.search(r'\{.*\}', reply, re.DOTALL)
        if not found:
            last_error = f'no JSON object in reply:\n{reply}'
            continue
        try:
            data = json.loads(found.group(0))
        except json.JSONDecodeError as e:
            last_error = f'malformed JSON ({e}):\n{found.group(0)}'
            continue
        missing = [k for k in ('match', 'transition', 'settle') if not data.get(k, '').strip()]
        if missing:
            last_error = f'missing key(s) {missing}:\n{reply}'
            continue
        return data
    raise ValueError(f'orchestrator failed after {attempts} attempt(s): {last_error}')


# ============================================================
# Section 5b — pain relief orchestration
# ============================================================
#
# Functional Music Guideline — Hospice & Chronic Illness, Module 1: Pain Relief.
# The master prompt's parts (goal, characteristics, sound design, trajectory,
# avoid, feel) live in the system prompt below; the carer's settings arrive in
# the user message. Qwen only writes what the music does in each phase. The
# code owns everything the guideline fixes — tags, instruments, tempo, the
# ending, the add-on lines — and assembles the six Stable Audio 3 prompts, so
# those hold in every phase however the sample comes out.
#
# Post-trained Stable Audio 3 ignores negative_prompt (base checkpoints only),
# so the guideline's "Avoid" list can only reach it as positive description.
# Wherever the guideline says "no X", the prompt says what to play instead.

PAIN_RELIEF_SYSTEM_PROMPT = """You are the orchestration layer of a functional-music \
generation pipeline, composing for Module 1 (Pain Relief) of the ResonaLife guideline \
for hospice and chronic illness. The listener may be seriously or chronically ill and \
in physical pain; often a carer sets the session up for them. You plan ONE continuous \
piece that moves through six phases. Code builds the final text-to-audio prompts \
around your words and fixes the tags, instruments, tempo, durations and ending — your \
job is what the music does inside them.

Goal: reduce emotional tension and stress associated with pain, encourage deep \
relaxation, slow breathing, and a sense of physical comfort and safety.

Musical characteristics:
- Very slow tempo with a stable pulse, constant for the whole piece. The code states \
the BPM; never write a number yourself.
- Smooth, predictable, repetitive musical patterns.
- Minimal harmonic tension and very gentle chord progressions; a warm, soft, consonant \
tonal palette in major or gently open harmony.
- Long sustained tones and gradual transitions; soft dynamics with extremely gradual \
volume changes.
- Low cognitive demand: nothing that asks for active attention.
- Spacious arrangement with a strong sense of silence and breathing room.

Sound design: warm, organic, acoustic timbres. Smooth, rounded high end; soft attacks; \
lows present but never heavy. Gradual frequency movement and gentle spatial depth, so \
the sound is warm, immersive and physically non-intrusive.

Session arc. It starts calm and only ever gets calmer. It never opens with energy to \
"meet" the listener, because any lift in intensity can feel like one more thing to \
cope with when in pain:
  immediate_calm      Enters softly at the slow tempo; warm sustained tones, no build-up.
  release_tension     Harmony settles further; phrases lengthen; volume eases down slightly.
  comfort_and_safety  A stable, repeating pattern; the most "held" and predictable part.
  deepen_relaxation   Fewer notes, more space and silence; timbre grows darker and softer.
  weightless_plateau  Pads and sustained tones float; melody nearly disappears.
  fade_or_loop        Fade ending: the texture thins into a slow, seamless fade to silence.
                      Loop ending: an unresolved continuation of the plateau that holds \
its level and texture to the last second, so the end flows straight back into the start.

Leave out entirely: sudden percussion, drums, dramatic crescendos, strong melodic \
hooks, fast rhythm, bright or harsh highs, sharp attacks, dissonance, dark or \
threatening atmosphere, vocals, lyrics, abrupt endings, and nature sounds that draw \
attention (birdsong, thunder, splashing water).

The piece should feel warm, safe, weightless, slow, comforting, spacious, peaceful, \
non-distracting and deeply relaxing.

How to write:
- Use only the instruments listed in the settings, in all six phases. Phases differ \
only in density, register, dynamics and space — never in instruments, key or tempo — \
so every join between phases is inaudible.
- Describe only what IS heard, in plain positive words: "beatless", "felt piano notes \
several seconds apart", "long, soft string chords". Text-to-audio models handle \
negation badly — a sound named after "no" tends to appear — so never name anything \
from the leave-out list, not even to exclude it.
- Each phase is one paragraph of 30 to 60 words on how the instruments behave in that \
phase: density, register, dynamics, space and mood. No BPM or other numbers, no tags, \
no phase names, no medical or therapeutic claims.
- Follow every setting and guideline line in the request. Add a sound the carer asks \
for only if it is soft, warm and steady; leave anything else out without mentioning \
it, and honour an exclusion simply by never using that sound.
- purpose: one of Relaxation, Rest, Sleep, Meditation.
- genres: two of Ambient, Neoclassical, New Age, Contemporary Classical, Minimalism.
- extra_instruments: instruments the carer asked for that fit the guideline; usually empty.
- addition: the carer's other wishes as a short positive phrase for the audio model; \
usually empty.

Respond with ONLY a JSON object:
{"purpose": "...", "genres": ["...", "..."], "extra_instruments": [], "addition": "", \
"immediate_calm": "...", "release_tension": "...", "comfort_and_safety": "...", \
"deepen_relaxation": "...", "weightless_plateau": "...", "fade_or_loop": "..."}"""

# Lead choices from the guideline's tailoring table. The code, not the orchestrator,
# writes the Instruments tag, so every phase carries the identical palette.
PAIN_LEADS = {
    'piano':   ('soft piano with gentle pads',   ['soft piano', 'gentle pads', 'warm strings', 'subtle sustained tones']),
    'strings': ('warm strings with gentle pads', ['warm strings', 'gentle pads', 'subtle sustained tones']),
    'pads':    ('pads only',                     ['gentle pads', 'subtle sustained tones']),
}

# The guideline's "Prompt add-ons" (behind the page's settings) and "Troubleshooting"
# lines (the page's fixes). Each is (the guideline's own line, its positive form).
# The guideline line goes to the orchestrator, exactly as the guideline says to add
# it to the prompt. The positive form is appended to every phase prompt, so the fix
# holds even where Qwen's wording drifts.
PAIN_ADDONS = {
    'no_melody':  ('Use pads and sustained tones only; no identifiable melody.',
                   'pads and sustained tones only, slow drifting chords'),
    'warm_piano': ('Piano should be felted and distant, with soft hammers and gentle sustain.',
                   'piano felted and distant, with soft hammers and gentle sustain'),
    'wind':       ('Add a very quiet, steady soft wind texture beneath the music, never louder than the pads.',
                   'a very quiet, steady soft wind texture beneath the music, never louder than the pads'),
    'waves':      ('Add a very quiet, steady distant ocean texture beneath the music, never louder than the pads.',
                   'a very quiet, steady distant ocean texture beneath the music, never louder than the pads'),
}
PAIN_FIXES = {
    'percussion': ('Strictly no percussion of any kind; the pulse is carried only by slow harmonic changes.',
                   'the pulse carried only by slow harmonic changes'),
    'catchy':     ('Use pads and sustained tones only; no identifiable melody.',
                   'pads and sustained tones only, slow drifting chords'),
    'bright':     ('Roll off high frequencies; everything should sound soft, dark-warm and distant.',
                   'high frequencies rolled off, everything soft, dark-warm and distant'),
    'bass':       ('Keep low frequencies light and gentle; no sub-bass or booming low notes.',
                   'low frequencies light and gentle'),
    'climax':     ('Keep intensity flat or decreasing for the entire piece; no build-ups.',
                   'intensity flat or decreasing for the entire piece'),
    'voices':     ('Instrumental only; no voices, choir, humming or vocal-like synths.',
                   'purely instrumental'),
    'dark':       ('Use major or gently open harmony; the mood is safe and reassuring, never melancholic.',
                   'major or gently open harmony, safe and reassuring'),
    # the guideline's line says "about 55 BPM"; the session's own tempo is used so
    # the fix never contradicts the tempo chosen in the settings
    'faster':     ('Hold a constant tempo of about {bpm} BPM from start to finish.',
                   'one constant {bpm} BPM from start to finish'),
}

# The last phase's ending, and a positive form of the guideline's Avoid list that
# goes into every phase prompt.
PAIN_ENDINGS = {
    'fade': 'a slow, seamless fade to silence',
    'loop': 'an unresolved continuation that loops seamlessly, steady to the very end',
}
PAIN_GUARDRAIL = ('beatless, soft attacks, smooth warm highs, gentle lows, '
                  'warm consonant harmony, intensity steady or falling')

PAIN_PURPOSES = ('Relaxation', 'Rest', 'Sleep', 'Meditation')
PAIN_GENRES = ('Ambient', 'Neoclassical', 'New Age', 'Contemporary Classical', 'Minimalism')

# A clause of Qwen's text that names one of these is dropped before it reaches the
# audio model: "no drums" in a text-to-audio prompt tends to bring drums in.
AVOIDED_SOUNDS = re.compile(
    r'\b(?:percussi\w*|drum\w*|crescendo\w*|climax\w*|build\w*|hooks?|catchy|'
    r'vocal\w*|voices?|choir\w*|lyric\w*|humming|spoken|'
    r'dissonan\w*|harsh\w*|bright\w*|piercing|shrill|glassy|'
    r'threatening|eerie|melanchol\w*|abrupt\w*|sudden\w*|'
    r'sharp\s+(?:attacks?|transients?)|fast\s+rhythm\w*|fast\s+tempo|'
    r'sub-?bass|boom\w*|rumbl\w*|bells?|chimes?|bird\w*|thunder\w*|splash\w*)\b',
    re.IGNORECASE,
)
# the code states the tempo once, identically in every phase
BPM_MENTION = re.compile(
    r'\s*(?:\b(?:at|around|about|approximately|roughly|near)\s+)?~?\s*\d{2,3}'
    r'(?:\s*(?:-|–|to)\s*\d{2,3})?\s*BPM\b',
    re.IGNORECASE,
)
PROMPT_TAG = re.compile(r'\b(?:Purpose|TrackType|VocalType|Genre|Instruments|Addition Request)\s*:')


def pain_relief_settings(options, session_request, log):
    """The carer's choices from the page, kept inside the guideline's safe ranges.

    Anything missing takes the guideline default; with no minutes in options the
    length falls back to the first "<n> min" in the request, then to 30 min.
    """
    o = options or {}

    lo, hi = config.PAIN_MINUTES
    minutes = o.get('minutes')
    if minutes is None:
        minutes = parse_duration(session_request, config.PAIN_DEFAULT_MINUTES * 60) // 60
    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        raise ValueError(f'minutes must be a whole number, got {minutes!r}')
    if not lo <= minutes <= hi:
        log(f'warning: {minutes} min is outside the pain relief range {lo}-{hi}, '
            f'using {min(max(minutes, lo), hi)} min')
        minutes = min(max(minutes, lo), hi)

    lo, hi = config.PAIN_BPM
    bpm = o.get('bpm', config.PAIN_DEFAULT_BPM)
    try:
        bpm = int(bpm)
    except (TypeError, ValueError):
        raise ValueError(f'bpm must be a whole number, got {bpm!r}')
    if not lo <= bpm <= hi:
        log(f'warning: {bpm} BPM is outside the pain relief range {lo}-{hi}, '
            f'using {min(max(bpm, lo), hi)} BPM')
        bpm = min(max(bpm, lo), hi)

    def choice(key, allowed, default):
        value = o.get(key, default)
        if value not in allowed:
            log(f'warning: unknown {key} {value!r}, using {default!r}')
            return default
        return value

    ending = choice('ending', PAIN_ENDINGS, 'fade')
    lead = choice('lead', PAIN_LEADS, 'piano')
    melody = choice('melody', ('sparse', 'none'), 'sparse')
    nature = choice('nature', ('off', 'wind', 'waves'), 'off')

    fixes = []
    for fix in o.get('fixes') or []:
        if fix not in PAIN_FIXES:
            log(f'warning: unknown fix {fix!r} ignored')
        elif fix not in fixes:
            fixes.append(fix)

    # warmer piano only means something while the piano plays (see pain_relief_instruments)
    warm_piano = bool(o.get('warm_piano')) and lead == 'piano' and not pads_only(melody, fixes)

    return {'minutes': minutes, 'bpm': bpm, 'ending': ending, 'lead': lead,
            'melody': melody, 'warm_piano': warm_piano, 'nature': nature, 'fixes': fixes}


def pads_only(melody, fixes):
    """No melody, or the "melody too catchy" fix: both are the guideline line
    "Use pads and sustained tones only; no identifiable melody"."""
    return melody == 'none' or 'catchy' in fixes


def pain_relief_instruments(settings):
    """The Instruments tag for every phase. With pads_only the piano leaves the
    palette: its struck notes are what carry a melody, and Stable Audio follows the
    tag more closely than any line of description."""
    instruments = list(PAIN_LEADS[settings['lead']][1])
    if pads_only(settings['melody'], settings['fixes']):
        return [i for i in instruments if 'piano' not in i]
    if settings['warm_piano']:
        instruments[0] = 'soft felt piano'
    return instruments


def pain_relief_lines(settings):
    """(guideline lines for the orchestrator, positive phrases for every phase prompt)
    for the add-ons and fixes these settings switch on, each once."""
    keys = []
    if settings['melody'] == 'none':
        keys.append('no_melody')
    if settings['warm_piano']:
        keys.append('warm_piano')
    if settings['nature'] != 'off':
        keys.append(settings['nature'])
    keys += settings['fixes']

    guideline_lines, audio_phrases = [], []
    for key in keys:
        line, phrase = {**PAIN_ADDONS, **PAIN_FIXES}[key]
        line, phrase = line.format(bpm=settings['bpm']), phrase.format(bpm=settings['bpm'])
        if line not in guideline_lines:
            guideline_lines.append(line)
        if phrase not in audio_phrases:
            audio_phrases.append(phrase)
    return guideline_lines, audio_phrases


def pain_relief_user_message(session_request, settings, instruments, guideline_lines):
    """What Qwen is told about this particular session."""
    melody = ('none (pads and sustained tones only)' if settings['melody'] == 'none'
              else 'very sparse, never a hook')
    nature = {'off': 'off', 'wind': 'a very quiet, steady soft wind',
              'waves': 'a very quiet, steady distant ocean'}[settings['nature']]
    ending = ('fade (a slow, seamless fade to silence)' if settings['ending'] == 'fade'
              else 'loop (an unresolved continuation that loops seamlessly, holding its level to the very end)')
    lines = [
        f'Session request: "{session_request}"',
        '',
        'Settings (the code enforces tempo, length, instruments and ending):',
        f'- Tempo: {settings["bpm"]} BPM, constant from start to finish',
        f'- Instruments, the same in all six phases: {", ".join(instruments)}',
        f'- Melody: {melody}',
        f'- Nature texture: {nature}',
        f'- Ending: {ending}',
    ]
    if guideline_lines:
        lines += ['', 'Guideline lines added for this session (follow every one):']
        lines += [f'- {line}' for line in guideline_lines]
    return '\n'.join(lines)


def audio_safe(text):
    """Qwen's text as the audio model should see it: tempo numbers removed (the code
    states the BPM) and any clause naming an avoided sound dropped.
    Returns (text, dropped clauses)."""
    text = BPM_MENTION.sub('', str(text)).strip()
    kept, dropped = [], []
    for clause in re.split(r'(?<=[,;.!?])\s+', text):
        (dropped if AVOIDED_SOUNDS.search(clause) else kept).append(clause)
    text = ' '.join(kept)
    text = re.sub(r'\s+([,;.!?])', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text).strip().rstrip(',;: ')
    text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    return text, dropped


def clean_pain_relief_plan(data, instruments):
    """Validate one parsed orchestrator reply and make it safe for the audio model.

    Raises ValueError for anything worth another sample (a missing or empty phase,
    prompt tags in a phase). Optional fields fall back to safe defaults instead.
    Returns (plan, notes) — notes say what was dropped, for the log.
    """
    if not isinstance(data, dict):
        raise ValueError('reply is not a JSON object')
    notes = []

    phases = {}
    for name, _ in config.PAIN_ARC:
        raw = data.get(name)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f'missing phase {name!r}')
        if PROMPT_TAG.search(raw):
            raise ValueError(f'{name} contains prompt tags, which the code adds itself')
        text, dropped = audio_safe(raw)
        if len(text.split()) < 8:
            raise ValueError(f'{name} is nearly empty once avoided sounds are removed: {raw!r}')
        if dropped:
            notes.append(f'{name}: dropped {dropped}')
        phases[name] = text

    purpose = str(data.get('purpose') or '').strip().capitalize()
    if purpose not in PAIN_PURPOSES:
        purpose = 'Relaxation'

    raw_genres = data.get('genres')
    if isinstance(raw_genres, str):
        raw_genres = [raw_genres]
    canonical = {g.lower(): g for g in PAIN_GENRES}
    genres = []
    for g in raw_genres if isinstance(raw_genres, list) else []:
        g = canonical.get(str(g).strip().lower())
        if g and g not in genres:
            genres.append(g)
    genres = genres[:2] or ['Ambient', 'Neoclassical']

    raw_extras = data.get('extra_instruments')
    if isinstance(raw_extras, str):
        raw_extras = [raw_extras]
    extras = []
    for item in raw_extras if isinstance(raw_extras, list) else []:
        item = re.sub(r'\s+', ' ', str(item)).strip().strip('.,;').lower()
        if not item or len(item) > 40 or item in instruments or item in extras:
            continue
        if AVOIDED_SOUNDS.search(item):
            notes.append(f'extra instrument dropped: {item!r}')
            continue
        extras.append(item)
    extras = extras[:2]

    addition, dropped = audio_safe(data.get('addition') or '')
    if dropped:
        notes.append(f'addition: dropped {dropped}')
    addition = ' '.join(addition.split()[:30]).rstrip(',;:. ')
    addition = addition[:1].lower() + addition[1:]   # it joins a lower-case list

    return {'purpose': purpose, 'genres': genres, 'extra_instruments': extras,
            'addition': addition, 'phases': phases}, notes


def generate_pain_relief_plan(session_request, settings, instruments, guideline_lines, log, attempts=3):
    """Call Qwen until one reply passes clean_pain_relief_plan, or give up."""
    user_message = pain_relief_user_message(session_request, settings, instruments, guideline_lines)
    last_error = None
    for attempt in range(1, attempts + 1):
        reply = ask_qwen(user_message, PAIN_RELIEF_SYSTEM_PROMPT, max_new_tokens=1400)
        found = re.search(r'\{.*\}', reply, re.DOTALL)
        if not found:
            last_error = f'no JSON object in reply:\n{reply}'
        else:
            try:
                plan, notes = clean_pain_relief_plan(json.loads(found.group(0)), instruments)
                for note in notes:
                    log(f'orchestrator note — {note}')
                return plan
            except json.JSONDecodeError as e:
                last_error = f'malformed JSON ({e}):\n{found.group(0)}'
            except ValueError as e:
                last_error = str(e)
        log(f'orchestrator reply {attempt}/{attempts} rejected: {last_error.splitlines()[0]}')
    raise ValueError(f'pain relief orchestrator failed after {attempts} attempt(s): {last_error}')


def assemble_pain_relief_prompts(plan, settings, instruments, audio_phrases):
    """The six Stable Audio 3 prompts, in the same tag format as the ISO pipeline.

    Tags, instruments, tempo and add-on phrases are identical in every phase —
    only Qwen's description changes — so nothing but density and space moves at
    the joins. Phrases of bpm/6 beats put one phrase near one slow breath: the
    guideline's 4-in/6-out at 60 BPM, or an 8-beat cycle at 50, about 6 a minute.
    """
    bpm = settings['bpm']
    tags = ('Purpose: ' + plan['purpose'] + ', TrackType: Music, VocalType: Instrumental, '
            + ''.join(f'Genre: {g}, ' for g in plan['genres'])
            + 'Instruments: ' + ', '.join(instruments + plan['extra_instruments']) + '.')
    tempo = f'Slow, steady {bpm} BPM, phrases of about {round(bpm / 6)} beats.'

    shared = list(audio_phrases)
    if plan['addition']:
        shared.append(plan['addition'])
    shared.append(PAIN_GUARDRAIL)

    prompts = {}
    for name, _ in config.PAIN_ARC:
        additions = [PAIN_ENDINGS[settings['ending']]] if name == 'fade_or_loop' else []
        description = plan['phases'][name].rstrip(' .,;:!?')
        prompts[name] = f'{tags} {tempo} {description}, Addition Request: {"; ".join(additions + shared)}'
    return prompts


# ============================================================
# Section 7 — session settings
# ============================================================

CAP_FIRST = config.MAX_MODEL_DURATION_S - config.CTX_S
CAP_MIDDLE = config.MAX_MODEL_DURATION_S - 2 * config.CTX_S
CAP_LAST = config.MAX_MODEL_DURATION_S - config.CTX_S


def slugify(text, max_len=40):
    """Turn a session request into a filesystem-safe folder name fragment."""
    s = re.sub(r'[^a-zA-Z0-9]+', '-', text.strip().lower()).strip('-')
    return s[:max_len] or 'session'


def unique_dir(base):
    """base, base_1, base_2, ... — first path that doesn't exist yet."""
    path, i = str(base), 1
    while os.path.isdir(path):
        path = f'{base}_{i}'
        i += 1
    return path


def parse_duration(session_request, default_s=config.DEFAULT_TOTAL_S):
    m = re.search(r'(\d+)\s*(?:-|\s)*min', session_request, re.IGNORECASE)
    return int(m.group(1)) * 60 if m else default_s


def clamp(name, requested, cap, log):
    d = min(requested, cap)
    if d < requested:
        log(f'warning: {name} clamped {requested}s -> {d:.0f}s (single-call limit)')
    return d


def build_segments(phase_prompts, phase_durations, total_duration_s, seed, log):
    """Turn one LLM orchestration result into the ordered list of model calls.

    match and transition are always exactly one call each — if their share of
    the request alone exceeds a single call, they're clamped down rather than
    chunked. Whatever total_duration_s has left over goes to settle, and settle
    is never clamped: it's chunked across as many calls as needed, each one
    continuing from the tail of the last, so any requested length is honored.
    """
    match_duration = clamp('match', phase_durations[0], CAP_FIRST, log)
    transition_duration = clamp('transition', phase_durations[1], CAP_MIDDLE, log)
    settle_total = max(total_duration_s - match_duration - transition_duration, 10.0)

    settle_chunk_durations = []
    remaining = settle_total
    while remaining > CAP_LAST:
        settle_chunk_durations.append(CAP_MIDDLE)   # a full-size middle chunk
        remaining -= CAP_MIDDLE
    settle_chunk_durations.append(remaining)        # final chunk, always <= CAP_LAST

    segments = [
        {'name': 'match', 'position': 'first', 'duration_s': match_duration,
         'seed': seed, 'prompt': phase_prompts['match']},
        {'name': 'transition', 'position': 'middle', 'duration_s': transition_duration,
         'seed': seed, 'prompt': phase_prompts['transition']},
    ]
    last_i = len(settle_chunk_durations) - 1
    for i, d in enumerate(settle_chunk_durations):
        segments.append({
            'name': 'settle' if last_i == 0 else f'settle_{i + 1}',
            'position': 'last' if i == last_i else 'middle',
            'duration_s': d,
            'seed': seed,
            'prompt': phase_prompts['settle'],
        })
    return segments


def build_arc_segments(arc, phase_prompts, total_duration_s, seed):
    """Model calls for a multi-phase arc (pain relief) where every phase keeps its share.

    Unlike build_segments nothing is clamped: each phase is split into as few
    equal chunks as fit under CAP_MIDDLE (safe at any position), and each chunk
    continues from the tail of the one before, exactly like settle's chunks.
    Phase boundaries are rounded cumulatively, so the phases add up to exactly
    total_duration_s — a 30-min session lands on the guideline's 3, 7.5, 13.5,
    19.5 and 27 min marks.
    """
    segments = []
    start, cumulative = 0, 0.0
    for i, (name, share) in enumerate(arc):
        cumulative += share
        end = total_duration_s if i == len(arc) - 1 else round(total_duration_s * cumulative)
        phase_s, start = end - start, end
        n = max(1, math.ceil(phase_s / CAP_MIDDLE))
        for k in range(n):
            segments.append({
                'name': name if n == 1 else f'{name}_{k + 1}',
                'phase': name,
                'position': 'middle',
                'duration_s': phase_s // n + (1 if k < phase_s % n else 0),
                'seed': seed,
                'prompt': phase_prompts[name],
            })
    segments[0]['position'] = 'first'
    segments[-1]['position'] = 'last'
    return segments


# ============================================================
# Section 8 — generation
# ============================================================

def size_for(seconds, mult=2048):
    """Round up to the sample-count grid the model's diffusion U-Net expects."""
    return int(np.ceil((seconds + 10) * SR / mult)) * mult


def audible_duration_s(audio, sr, window_s=0.05, floor_db=-60.0):
    """Seconds up to the last window whose RMS clears floor_db — a trailing-silence check."""
    mono = audio.mean(axis=1)
    w = int(window_s * sr)
    n = len(mono) // w
    rms = np.sqrt((mono[:n * w].reshape(n, w) ** 2).mean(axis=1) + 1e-12)
    above = np.nonzero(rms > 10 ** (floor_db / 20))[0]
    return ((above[-1] + 1) * w / sr) if len(above) else 0.0


def render_segment(path, prompt, duration_s, seed, steps, log,
                   inpaint_context=None, trim_front=0, trim_back=0):
    """Generate one segment, or reuse it from disk if a previous run already saved it.

    `duration_s` is the *requested generation length*, including any context/
    safety buffer — `trim_front`/`trim_back` (in samples) cut that buffer back
    out afterward, leaving just the real content. Delete the .wav to force a
    fresh generation of that segment.
    """
    if os.path.exists(path):
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        assert sr == SR, f'{path} was rendered at {sr}Hz, this session expects {SR}Hz'
        log(f'{os.path.basename(path)} reused from disk ({len(audio)/SR:.0f}s)')
        return audio

    gen_kwargs = dict(
        prompt=prompt,
        duration=duration_s,
        seed=seed,
        steps=steps,
        sample_size=size_for(duration_s),
    )
    if inpaint_context is not None:
        # continue from the tail of the previous segment: (sample_rate, tensor) —
        # note the order, Stability's own doc snippet passes them reversed
        gen_kwargs.update(
            inpaint_audio=(SR, torch.from_numpy(inpaint_context.T.copy())),
            inpaint_mask_start_seconds=config.CTX_S,   # end of the preserved (context) audio
            inpaint_mask_end_seconds=duration_s,       # generate through to the end
        )

    t0 = time.time()
    out = model.generate(**gen_kwargs)
    audio = out[0].detach().cpu().numpy().T.astype(np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]

    if trim_back:
        audio = audio[:-trim_back]     # drop the tail safety buffer (boundary artifacts live here)
    if trim_front:
        audio = audio[trim_front:]     # drop the leading context, already present in the prior segment

    aud = audible_duration_s(audio, SR)
    log(f'{os.path.basename(path)} generated {len(audio)/SR:.0f}s '
        f'audible {aud:.0f}s ({time.time()-t0:.0f}s)')
    sf.write(path, audio, SR, subtype='FLOAT')
    return audio


def build_session(segments, out_dir, log, on_segment=None):
    """Render every segment in order into out_dir, each conditioned on the tail
    of the one before it, and write the concatenated result to completed.wav.
    """
    ctx_samples = int(config.CTX_S * SR)
    rendered = []
    prev_tail = None
    for i, seg in enumerate(segments):
        is_first = seg['position'] == 'first'
        is_last = seg['position'] == 'last'

        if on_segment:
            on_segment(i, len(segments), seg['name'])

        # requested generation length = real content + whichever context buffer(s) this segment needs
        request_s = seg['duration_s'] + (0 if is_first else config.CTX_S) + (0 if is_last else config.CTX_S)

        audio = render_segment(
            os.path.join(out_dir, f"{seg['name']}.wav"),
            seg['prompt'], request_s, seg['seed'], config.STEPS, log,
            inpaint_context=prev_tail,
            trim_front=0 if is_first else ctx_samples,
            trim_back=0 if is_last else ctx_samples,
        )
        rendered.append(audio)
        prev_tail = audio[-ctx_samples:]   # becomes the next segment's leading context
        torch.cuda.empty_cache()           # release this segment's activation blocks

    track = np.concatenate(rendered)
    raw_path = os.path.join(out_dir, 'completed.wav')
    sf.write(raw_path, track, SR, subtype='FLOAT')
    log(f'total {len(track)/SR/60:.2f} min -> {raw_path}')
    return track, raw_path


# ============================================================
# Section 9 — post-processing
# ============================================================

def apply_post_eq(in_path, out_path, log,
                  hpf_hz=25, body_lo=90, body_hi=130, body_db=0.0,
                  mud_lo=250, mud_hi=400, mud_db=-1.0,
                  air_hz=12000, air_db=-0.5,
                  target_lufs=-16.0, peak_db=-1.0,
                  fade_out_s=0.0, loop_crossfade_s=0.0):
    """Gentle cleanup EQ + loudness normalization for a finished session render.

    hpf_hz      — high-pass cutoff; removes only inaudible/subsonic rumble
    body_*      — 90-130 Hz body band; body_db=0.0 means no artificial boost by default
    mud_*       — 250-400 Hz low-mid band; mud_db is a very gentle cleanup cut
    air_*       — high-shelf-ish trim above air_hz, keeps long sessions smooth
    target_lufs — consistent, comfortable functional-music loudness target
    peak_db     — safety ceiling so normalization never clips

    Pain relief endings (both 0 = off, the ISO default). They run after
    normalization, so loudness is measured on the music at full level and the
    ending never shifts the level the rest of the session plays at:
    fade_out_s       — the last N seconds fade smoothly to exact silence
    loop_crossfade_s — the last N seconds are crossfaded into the first N and
                       removed, so the file repeats with no click, gap or jump
    """
    x, sr = sf.read(in_path, always_2d=True)
    x = x.astype(np.float64)

    sos = butter(2, hpf_hz, btype='highpass', fs=sr, output='sos')
    x = sosfiltfilt(sos, x, axis=0)

    # each band below: filter out just that band, then add it back in scaled
    # by the target gain — a bandpass-and-recombine shelf/bell, not a full EQ
    sos = butter(4, [body_lo, body_hi], btype='bandpass', fs=sr, output='sos')
    band = sosfiltfilt(sos, x, axis=0)
    x = x + band * (10 ** (body_db / 20.0) - 1.0)

    sos = butter(4, [mud_lo, mud_hi], btype='bandpass', fs=sr, output='sos')
    band = sosfiltfilt(sos, x, axis=0)
    x = x + band * (10 ** (mud_db / 20.0) - 1.0)

    sos = butter(2, air_hz, btype='highpass', fs=sr, output='sos')
    band = sosfiltfilt(sos, x, axis=0)
    x = x + band * (10 ** (air_db / 20.0) - 1.0)

    # loudness-normalize to target_lufs, then enforce peak_db as a hard ceiling
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x)
    x = x * (10 ** ((target_lufs - loudness) / 20.0))

    if fade_out_s > 0:
        # raised cosine: no corner where the fade begins, exactly silent at the end
        n = min(int(fade_out_s * sr), len(x))
        x[-n:] *= (np.cos(np.linspace(0.0, np.pi / 2, n)) ** 2)[:, None]
        log(f'fade to silence over the last {n / sr:.0f}s')

    if loop_crossfade_s > 0:
        # equal-power crossfade of the tail into the head, then drop the tail. The
        # last sample of the result runs straight into the first, exactly as those
        # two samples sat next to each other in the render.
        n = min(int(loop_crossfade_s * sr), len(x) // 4)
        ramp = np.linspace(0.0, np.pi / 2, n, endpoint=False)[:, None]
        x[:n] = x[:n] * np.sin(ramp) + x[-n:] * np.cos(ramp)
        x = x[:-n]
        log(f'loop point: last {n / sr:.1f}s crossfaded into the start')

    ceiling = 10 ** (peak_db / 20.0)
    peak = np.max(np.abs(x))
    if peak > ceiling:
        x = x * (ceiling / peak)

    sf.write(out_path, x, sr, subtype='PCM_24')
    final_loudness = meter.integrated_loudness(x)
    log(f'{loudness:.2f} LUFS -> {final_loudness:.2f} LUFS, '
        f'peak {20*np.log10(np.max(np.abs(x))):.2f} dBFS')
    return out_path


# ============================================================
# Preview encoding and waveform peaks
# ============================================================

def make_preview(in_path, out_dir, log):
    """Write a small, browser-safe version of the finished session.

    The deliverable is 24-bit PCM, which Chrome plays inconsistently and which
    runs ~380 MB for a 20-minute session — unusable over a tunnel. ffmpeg gives
    us a ~20 MB MP3; without it we fall back to 16-bit WAV, which every browser
    handles but is still large.
    """
    if shutil.which('ffmpeg'):
        out = os.path.join(out_dir, 'preview.mp3')
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-i', in_path,
             '-codec:a', 'libmp3lame', '-b:a', '128k', out],
            check=True,
        )
        log(f'preview.mp3 {os.path.getsize(out)/1e6:.1f} MB')
        return out

    out = os.path.join(out_dir, 'preview.wav')
    x, sr = sf.read(in_path, always_2d=True)
    sf.write(out, x, sr, subtype='PCM_16')
    log(f'preview.wav {os.path.getsize(out)/1e6:.1f} MB (ffmpeg not found — install it for smaller previews)')
    return out


def make_loop_check(in_path, out_dir, log, seconds=None):
    """The guideline's "check the last and first seconds together" as one clip.

    The end of a loop-ending session followed straight by its start. A player's
    own looping (and MP3 encoder padding) can add a gap of its own; this clip
    can't, so what you hear at its midpoint is the file's actual loop point.
    """
    seconds = config.PAIN_LOOP_CHECK_S if seconds is None else seconds
    info = sf.info(in_path)
    n = min(int(seconds * info.samplerate), info.frames // 2)
    tail, sr = sf.read(in_path, start=info.frames - n, always_2d=True)
    head, _ = sf.read(in_path, frames=n, always_2d=True)

    wav = os.path.join(out_dir, 'loop_check.wav')
    sf.write(wav, np.concatenate([tail, head]), sr, subtype='PCM_16')
    if not shutil.which('ffmpeg'):
        log(f'loop_check.wav: last {n / sr:.0f}s, then first {n / sr:.0f}s')
        return wav

    mp3 = os.path.join(out_dir, 'loop_check.mp3')
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', wav,
         '-codec:a', 'libmp3lame', '-b:a', '192k', mp3],
        check=True,
    )
    os.remove(wav)
    log(f'loop_check.mp3: last {n / sr:.0f}s, then first {n / sr:.0f}s')
    return mp3


def compute_peaks(path, buckets=900):
    """Downsample to a peak envelope the browser can draw immediately.

    Computed here rather than client-side so the waveform doesn't require
    pulling the whole file down first.
    """
    x, _ = sf.read(path, always_2d=True)
    mono = np.abs(x.mean(axis=1))
    n = len(mono) // buckets
    if n < 1:
        return [float(v) for v in mono]
    trimmed = mono[:n * buckets].reshape(buckets, n)
    peaks = trimmed.max(axis=1)
    top = peaks.max()
    if top > 0:
        peaks = peaks / top
    return [round(float(v), 4) for v in peaks]


# ============================================================
# Entry point
# ============================================================

def generate_session(session_request, seed=None, log=None, on_segment=None,
                     mode='iso', options=None, on_plan=None):
    """Full pipeline for one request. Returns a dict of results.

    mode                           — 'iso' (match, transition, settle) or 'pain_relief'
    options                        — pain relief settings from the page, see
                                     pain_relief_settings; ignored by 'iso'
    log(str)                       — called with each progress line
    on_segment(i, total, name)     — called before each model call
    on_plan(names)                 — called once, with every segment name in order
    """
    if mode == 'pain_relief':
        return generate_pain_relief_session(session_request, seed=seed, log=log,
                                            on_segment=on_segment, options=options,
                                            on_plan=on_plan)
    if mode != 'iso':
        raise ValueError(f'unknown mode {mode!r}')

    log = log or print
    seed = config.RANDOM_SEED if seed is None else seed

    total_duration_s = parse_duration(session_request)
    phase_durations = [round(total_duration_s * f) for f in config.SPLIT]
    log(f'request "{session_request}" — {total_duration_s}s ({total_duration_s/60:.1f} min)')

    log('orchestrating...')
    load_qwen()
    try:
        phase_prompts = generate_phase_prompts(session_request)
    finally:
        unload_qwen()   # always give the VRAM back, even if orchestration failed
    for name in ('match', 'transition', 'settle'):
        log(f'[{name}] {phase_prompts[name]}')

    out_dir = unique_dir(config.WORK / slugify(session_request))
    os.makedirs(out_dir, exist_ok=True)
    log(f'output folder {out_dir}')

    segments = build_segments(phase_prompts, phase_durations, total_duration_s, seed, log)
    if on_plan:
        on_plan([s['name'] for s in segments])
    planned = sum(s['duration_s'] for s in segments)
    log(f'{len(segments)} segments, {planned:.0f}s planned vs {total_duration_s}s requested')

    track, raw_path = build_session(segments, out_dir, log, on_segment)
    eq_path = apply_post_eq(raw_path, os.path.join(out_dir, 'completed_EQed.wav'),
                            log, **config.POST_EQ)

    preview_path = make_preview(eq_path, out_dir, log)
    peaks = compute_peaks(eq_path)

    return {
        'mode': 'iso',
        'ending': None,
        'out_dir': out_dir,
        'raw_path': raw_path,
        'eq_path': eq_path,
        'preview_path': preview_path,
        'peaks': peaks,
        'duration_s': len(track) / SR,
        'prompts': phase_prompts,
        'segments': [{'name': s['name'], 'duration_s': s['duration_s']} for s in segments],
    }

def generate_pain_relief_session(session_request, seed=None, log=None, on_segment=None,
                                 options=None, on_plan=None):
    """The guideline's Pain Relief arc: six phases that start calm and only get calmer.

    Same rendering, stitching, EQ and preview as the ISO pipeline; what differs is
    the orchestrator, the six-phase segment plan, and the ending (a fade to
    silence, or a loop point folded into the start) applied to the finished file.
    """
    log = log or print
    seed = config.RANDOM_SEED if seed is None else seed

    settings = pain_relief_settings(options, session_request, log)
    total_duration_s = settings['minutes'] * 60
    log(f'pain relief request "{session_request}" — {total_duration_s}s '
        f'({settings["minutes"]} min), {settings["bpm"]} BPM, {settings["ending"]} ending')

    instruments = pain_relief_instruments(settings)
    guideline_lines, audio_phrases = pain_relief_lines(settings)

    log('orchestrating...')
    load_qwen()
    try:
        plan = generate_pain_relief_plan(session_request, settings, instruments, guideline_lines, log)
    finally:
        unload_qwen()   # always give the VRAM back, even if orchestration failed
    phase_prompts = assemble_pain_relief_prompts(plan, settings, instruments, audio_phrases)
    for name, _ in config.PAIN_ARC:
        log(f'[{name}] {phase_prompts[name]}')

    out_dir = unique_dir(config.WORK / slugify(session_request))
    os.makedirs(out_dir, exist_ok=True)
    log(f'output folder {out_dir}')

    segments = build_arc_segments(config.PAIN_ARC, phase_prompts, total_duration_s, seed)
    if on_plan:
        on_plan([s['name'] for s in segments])
    planned = sum(s['duration_s'] for s in segments)
    log(f'{len(segments)} segments across {len(config.PAIN_ARC)} phases, '
        f'{planned:.0f}s planned vs {total_duration_s}s requested')

    # [1]: keep the path, let the stitched array go before EQ loads its own copy
    raw_path = build_session(segments, out_dir, log, on_segment)[1]

    post = dict(config.POST_EQ)
    if settings['ending'] == 'fade':
        post['fade_out_s'] = sum(s['duration_s'] for s in segments if s['phase'] == 'fade_or_loop')
    else:
        post['loop_crossfade_s'] = config.PAIN_LOOP_CROSSFADE_S
    eq_path = apply_post_eq(raw_path, os.path.join(out_dir, 'completed_EQed.wav'), log, **post)

    preview_path = make_preview(eq_path, out_dir, log)
    loop_check_path = make_loop_check(eq_path, out_dir, log) if settings['ending'] == 'loop' else None
    peaks = compute_peaks(eq_path)

    return {
        'mode': 'pain_relief',
        'ending': settings['ending'],
        'settings': settings,
        'out_dir': out_dir,
        'raw_path': raw_path,
        'eq_path': eq_path,
        'preview_path': preview_path,
        'loop_check_path': loop_check_path,
        'peaks': peaks,
        'duration_s': sf.info(eq_path).duration,   # the loop ending is a few seconds shorter than the render
        'prompts': phase_prompts,
        'segments': [{'name': s['name'], 'duration_s': s['duration_s']} for s in segments],
    }

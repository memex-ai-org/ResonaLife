"""Notebook sections 5-9 as a resident service.

Both models load once at process start and stay on the GPU. Nothing here
touches Colab, Drive, or `!pip`. The only entry point the server uses is
generate_session(), which runs one of four pipelines:

  iso          match -> transition -> settle, three prompts (the original)
  pain_relief  the Hospice & Chronic Illness guideline's Module 1, Pain Relief:
               six phases that start calm and only ever get calmer
  dementia     Module 2, Dementia: familiar, simple and emotionally safe; five
               phases, or six in Calming mode
  depression   Module 3, Depression: meets the mood first, then comforts or
               gently lifts; four or five phases depending on the session mode
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
# Section 5b — guideline modules: shared pieces
# ============================================================
#
# Functional Music Guideline — Hospice & Chronic Illness. Every module works the
# same way. Its master prompt's parts (goal, characteristics, sound design,
# trajectory, avoid, feel) live in a system prompt; the carer's settings, the
# guideline lines they switch on and the listener profile arrive in the user
# message, as the guideline says to add them to the bottom of the prompt. Qwen
# only writes what the music does in each phase (and, where the guideline leaves
# it to the listener's taste, the style). The code owns everything the guideline
# fixes — tags, instruments, tempo, the ending, the add-on lines — and assembles
# the Stable Audio 3 prompts, so those hold in every phase however the sample
# comes out.
#
# Post-trained Stable Audio 3 ignores negative_prompt (base checkpoints only),
# so a module's "Avoid" list can only reach it as positive description.
# Wherever the guideline says "no X", the prompt says what to play instead.

# the code states the tempo once, identically in every phase
BPM_MENTION = re.compile(
    r'\s*(?:\b(?:at|around|about|approximately|roughly|near)\s+)?~?\s*\d{2,3}'
    r'(?:\s*(?:-|–|to)\s*\d{2,3})?\s*BPM\b',
    re.IGNORECASE,
)
PROMPT_TAG = re.compile(r'\b(?:Purpose|TrackType|VocalType|Genre|Instruments|Addition Request)\s*:')

# Word groups for the dementia and depression filters (see audio_safe). Percussion
# and voice words are let through only where the settings put those sounds.
PERCUSSION_WORDS = r'percussi\w*|drum\w*|snare\w*|cymbal\w*|tambourine\w*|shakers?|hi-?hats?'
VOICE_WORDS = r'vocal\w*|voices?|humm\w*|hums?|sing(?:s|ing|er|ers|able)?|sing-along'
# the guideline's Avoid lists and "Leave out" moods, shared by both modules
CARE_AVOIDED = (
    r'crescendo\w*|climax\w*|build-?ups?|dissonan\w*|harsh\w*|piercing|shrill|glassy|abrupt\w*|'
    r'sudden\w*|sharp\s+(?:attacks?|transients?)|sub-?bass|boom\w*|rumbl\w*|distort\w*|aggressi\w*|'
    r'chao\w*|syncopat\w*|polyrhythm\w*|lyric\w*|spoken|choir\w*|chant\w*|bells?|chimes?|'
    r'thunder\w*|siren\w*|alarm\w*|dramatic\w*|intens\w*|eerie|frighten\w*|scary|threatening|'
    r'melanchol\w*|sad\w*|sorrow\w*|mourn\w*|gloom\w*|somber|sombre|bleak|dark\w*|lonel\w*'
)
PROMPT_WORDS = 190   # about what the T5Gemma text encoder reads; longer prompts lose their end

# A genre Qwen suggests from the listener profile is used as given unless it is
# one of these: experimental, harsh, vocal-led, dark or euphoric styles.
GENRE_BLOCKED = re.compile(
    r'\b(?:metal|hard\s*rock|heavy|punk|grunge|hardcore|industrial|noise|experimental|avant\w*|'
    r'glitch|dubstep|drum\s*(?:and|&|n)\s*bass|techno|trance|house|edm|dance|rave|trap|rap|'
    r'hip\s*-?\s*hop|drill|horror|dark\w*|doom|goth\w*|drone|opera|choral|choir|a\s*cappella|emo)\b',
    re.IGNORECASE,
)
PROFILE_MAX = 400    # characters kept per listener profile field
TIME_OF_DAY_NAMES = {'none': 'not set', 'morning': 'morning', 'afternoon': 'afternoon',
                     'daytime': 'daytime', 'evening': 'evening', 'presleep': 'pre-sleep'}


def whole_number(key, value):
    """int(value), or a ValueError the page can show."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f'{key} must be a whole number, got {value!r}')


def in_range(value, lo, hi, unit, what, log):
    """value pulled back inside the safe range lo-hi, with a warning if it moved."""
    if lo <= value <= hi:
        return value
    clamped = min(max(value, lo), hi)
    log(f'warning: {value} {unit} is outside the {what} range {lo}-{hi}, using {clamped} {unit}')
    return clamped


def session_minutes(options, session_request, minutes_range, default_minutes, what, log):
    """options['minutes'], else the first "<n> min" in the request, else the default."""
    minutes = options.get('minutes')
    if minutes is None:
        minutes = parse_duration(session_request, default_minutes * 60) // 60
    return in_range(whole_number('minutes', minutes), *minutes_range, 'min', what, log)


def choice(options, key, allowed, default, log):
    """options[key] if it is one of allowed, else the default (with a warning if it was set)."""
    value = options.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        log(f'warning: unknown {key} {value!r}, using {default!r}')
        return default
    return value


def fix_list(options, allowed, log):
    """The page's fixes: known ones only, each once, in the order given."""
    fixes = []
    for fix in options.get('fixes') or []:
        if not isinstance(fix, str) or fix not in allowed:
            log(f'warning: unknown fix {fix!r} ignored')
        elif fix not in fixes:
            fixes.append(fix)
    return fixes


def drop_fix(fixes, fix, reason, log):
    """Remove a fix that doesn't apply to this session, saying why."""
    if fix in fixes:
        fixes.remove(fix)
        log(f'warning: fix {fix!r} ignored: {reason}')


def listener_profile(raw, fields, choices, log):
    """The guideline's listener profile, filled fields only, in the guideline's order.

    raw is {field: text, or a list of the values its chips offer}; fields is
    ((field, label), ...). "Leave out anything you don't know or the person
    wouldn't want shared", so empty fields simply don't appear.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        log('warning: listener profile ignored: it is not a set of fields')
        return {}
    labels = dict(fields)
    profile = {}
    for key, value in raw.items():
        if key not in labels:
            log(f'warning: unknown listener profile field {key!r} ignored')
            continue
        if isinstance(value, list):
            allowed = choices.get(key, ())
            value = ', '.join(dict.fromkeys(v for v in value if isinstance(v, str) and v in allowed))
        value = re.sub(r'\s+', ' ', str(value if value is not None else '')).strip()[:PROFILE_MAX]
        if value:
            profile[key] = value
    return {key: profile[key] for key, _ in fields if key in profile}


def profile_block(profile, fields):
    """The profile as the guideline writes it: "Listener profile:" then "Field: value" lines."""
    if not profile:
        return []
    labels = dict(fields)
    return ['', 'Listener profile:'] + [f'{labels[key]}: {value}' for key, value in profile.items()]


def collect_lines(entries, **fields):
    """(guideline lines for the orchestrator, [(audio phrase, phases or None)]).

    entries are ((guideline line, audio phrase), phases) — phases limits the
    phrase to those phases, None means every phase. Each line and phrase is kept
    once, in order; an empty phrase means the line is for the orchestrator only.
    """
    lines, phrases = [], []
    for (line, phrase), phases in entries:
        line, phrase = line.format(**fields), phrase.format(**fields)
        if line not in lines:
            lines.append(line)
        if phrase and phrase not in [p for p, _ in phrases]:
            phrases.append((phrase, phases))
    return lines, phrases


def audio_safe(text, avoided=None):
    """Qwen's text as the audio model should see it: tempo numbers removed (the code
    states the BPM) and any clause naming an avoided sound dropped.
    avoided defaults to the pain relief list. Returns (text, dropped clauses)."""
    avoided = avoided or AVOIDED_SOUNDS
    text = BPM_MENTION.sub('', str(text)).strip()
    kept, dropped = [], []
    for clause in re.split(r'(?<=[,;.!?])\s+', text):
        (dropped if avoided.search(clause) else kept).append(clause)
    text = ' '.join(kept)
    text = re.sub(r'\s+([,;.!?])', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text).strip().rstrip(',;: ')
    text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    return text, dropped


def word_filter(*groups):
    """One case-insensitive whole-word pattern from alternation strings."""
    return re.compile(r'\b(?:' + '|'.join(groups) + r')\b', re.IGNORECASE)


def clean_phases(data, arc, avoided_by_phase):
    """Every phase's text from one parsed reply, made safe for the audio model.

    Raises ValueError for anything worth another sample (a missing or empty
    phase, prompt tags in a phase). Returns (phases, notes for the log).
    """
    if not isinstance(data, dict):
        raise ValueError('reply is not a JSON object')
    phases, notes = {}, []
    for name, _ in arc:
        raw = data.get(name)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f'missing phase {name!r}')
        if PROMPT_TAG.search(raw):
            raise ValueError(f'{name} contains prompt tags, which the code adds itself')
        text, dropped = audio_safe(raw, avoided_by_phase[name])
        if len(text.split()) < 8:
            raise ValueError(f'{name} is nearly empty once avoided sounds are removed: {raw!r}')
        if dropped:
            notes.append(f'{name}: dropped {dropped}')
        phases[name] = text
    return phases, notes


def clean_purpose(value, allowed, default):
    purpose = str(value or '').strip().capitalize()
    return purpose if purpose in allowed else default


def clean_genres(value, default, notes, avoided=None):
    """Up to two short genre names. The listener's own styles are welcome ("Chinese
    Folk", "Big Band"); GENRE_BLOCKED styles, and any the module's avoided-words
    filter catches ("Upbeat Pop"), are dropped. None left: the default."""
    if isinstance(value, str):
        value = [value]
    genres = []
    for g in value if isinstance(value, list) else []:
        g = re.sub(r'\s+', ' ', str(g)).strip().strip('.,;')
        if not g or len(g) > 30 or len(g.split()) > 3 or not re.fullmatch(r"[A-Za-z0-9 &'\-]+", g):
            continue
        if GENRE_BLOCKED.search(g) or (avoided and avoided.search(g)):
            notes.append(f'genre dropped: {g!r}')
            continue
        g = ' '.join(w[:1].upper() + w[1:] for w in g.split())
        if g.lower() not in [x.lower() for x in genres]:
            genres.append(g)
    return genres[:2] or list(default)


def clean_addition(value, avoided, notes):
    """The carer's other wishes as Qwen phrased them, safe and short, lower-case first."""
    addition, dropped = audio_safe(value or '', avoided)
    if dropped:
        notes.append(f'addition: dropped {dropped}')
    addition = ' '.join(addition.split()[:30]).rstrip(',;:. ')
    return addition[:1].lower() + addition[1:]   # it joins a lower-case list


def orchestrate(user_message, system_prompt, clean, log, what, attempts=3):
    """Call Qwen until one reply passes clean(data) -> (plan, notes), or give up.

    Instruct models occasionally wrap JSON in prose or markdown fences, drop a
    key or name an avoided sound in every clause — another sample is cheap.
    """
    last_error = None
    for attempt in range(1, attempts + 1):
        reply = ask_qwen(user_message, system_prompt, max_new_tokens=1400)
        found = re.search(r'\{.*\}', reply, re.DOTALL)
        if not found:
            last_error = f'no JSON object in reply:\n{reply}'
        else:
            try:
                plan, notes = clean(json.loads(found.group(0)))
                for note in notes:
                    log(f'orchestrator note — {note}')
                return plan
            except json.JSONDecodeError as e:
                last_error = f'malformed JSON ({e}):\n{found.group(0)}'
            except ValueError as e:
                last_error = str(e)
        log(f'orchestrator reply {attempt}/{attempts} rejected: {last_error.splitlines()[0]}')
    raise ValueError(f'{what} orchestrator failed after {attempts} attempt(s): {last_error}')


def prompt_tags(purpose, genres, instruments, instrumental=True):
    """The AudioSparx-style tags every guideline prompt opens with. Stability documents
    only VocalType: Instrumental, so a session with a wordless voice leaves it out."""
    return ('Purpose: ' + purpose + ', TrackType: Music, '
            + ('VocalType: Instrumental, ' if instrumental else '')
            + ''.join(f'Genre: {g}, ' for g in genres)
            + 'Instruments: ' + ', '.join(instruments) + '.')


def json_skeleton(fields, arc):
    """The reply shape shown to Qwen: fixed fields, then one key per phase."""
    return '{' + ', '.join(fields + [f'"{name}": "..."' for name, _ in arc]) + '}'


# ============================================================
# Section 5c — Module 1: pain relief orchestration
# ============================================================
#
# Music that supports relaxation and a sense of comfort when someone is in pain.
# Six phases that start calm and only ever get calmer, one fixed palette the
# carer chooses (the Lead row), one constant tempo.

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


def pain_relief_settings(options, session_request, log):
    """The carer's choices from the page, kept inside the guideline's safe ranges.

    Anything missing takes the guideline default; with no minutes in options the
    length falls back to the first "<n> min" in the request, then to 30 min.
    """
    o = options if isinstance(options, dict) else {}
    minutes = session_minutes(o, session_request, config.PAIN_MINUTES, config.PAIN_DEFAULT_MINUTES,
                              'pain relief', log)
    bpm = in_range(whole_number('bpm', o.get('bpm', config.PAIN_DEFAULT_BPM)), *config.PAIN_BPM,
                   'BPM', 'pain relief', log)
    ending = choice(o, 'ending', PAIN_ENDINGS, 'fade', log)
    lead = choice(o, 'lead', PAIN_LEADS, 'piano', log)
    melody = choice(o, 'melody', ('sparse', 'none'), 'sparse', log)
    nature = choice(o, 'nature', ('off', 'wind', 'waves'), 'off', log)
    fixes = fix_list(o, PAIN_FIXES, log)

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


def clean_pain_relief_plan(data, instruments):
    """Validate one parsed orchestrator reply and make it safe for the audio model.

    Raises ValueError for anything worth another sample (a missing or empty phase,
    prompt tags in a phase). Optional fields fall back to safe defaults instead.
    Returns (plan, notes) — notes say what was dropped, for the log.
    """
    phases, notes = clean_phases(data, config.PAIN_ARC, {name: AVOIDED_SOUNDS for name, _ in config.PAIN_ARC})

    purpose = clean_purpose(data.get('purpose'), PAIN_PURPOSES, 'Relaxation')

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

    addition = clean_addition(data.get('addition'), AVOIDED_SOUNDS, notes)

    return {'purpose': purpose, 'genres': genres, 'extra_instruments': extras,
            'addition': addition, 'phases': phases}, notes


def generate_pain_relief_plan(session_request, settings, instruments, guideline_lines, log, attempts=3):
    """Call Qwen until one reply passes clean_pain_relief_plan, or give up."""
    user_message = pain_relief_user_message(session_request, settings, instruments, guideline_lines)
    return orchestrate(user_message, PAIN_RELIEF_SYSTEM_PROMPT,
                       lambda data: clean_pain_relief_plan(data, instruments),
                       log, 'pain relief', attempts)


def assemble_pain_relief_prompts(plan, settings, instruments, audio_phrases):
    """The six Stable Audio 3 prompts, in the same tag format as the ISO pipeline.

    Tags, instruments, tempo and add-on phrases are identical in every phase —
    only Qwen's description changes — so nothing but density and space moves at
    the joins. Phrases of bpm/6 beats put one phrase near one slow breath: the
    guideline's 4-in/6-out at 60 BPM, or an 8-beat cycle at 50, about 6 a minute.
    """
    bpm = settings['bpm']
    tags = prompt_tags(plan['purpose'], plan['genres'], instruments + plan['extra_instruments'])
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
# Section 5d — Module 2: dementia orchestration
# ============================================================
#
# Familiar, simple and emotionally safe music that supports comfort, calm,
# orientation and gentle engagement for people living with dementia. The Session
# mode line switches the arc (Calming) or the style (Reminiscence, Social); one
# Time of day line gives each part of the day its own identity. The style comes
# from the listener profile, so Qwen picks the genres and, from the guideline's
# instrument list, the palette; the code decides which of those instruments play
# in each phase, holds one constant tempo and adds the lines.

DEMENTIA_SYSTEM_PROMPT = """You are the orchestration layer of a functional-music \
generation pipeline, composing for Module 2 (Dementia) of the ResonaLife guideline for \
hospice and chronic illness. The listener lives with dementia; often a carer sets the \
session up for them. You plan ONE continuous piece that moves through the phases listed \
in the request. Code builds the final text-to-audio prompts around your words and fixes \
the tags, tempo, durations, the ending and which of your instruments play in each phase — \
your job is the style and what the music does inside them.

Goal: create a familiar, safe, and emotionally supportive auditory environment that \
supports emotional comfort, calmness, orientation, engagement, and relaxation, while \
minimizing cognitive load and sensory overstimulation. The music should support \
dementia-care environments and wellbeing activities without making medical or therapeutic \
claims; it does not treat, cure, reverse, or prevent dementia.

Musical characteristics:
- A slow-to-moderate tempo, held steady, with no rapid tempo changes. The code states \
the BPM; never write a number yourself.
- Stable pulse, simple meter, and predictable rhythmic patterns with gentle repetition.
- Consonant harmonies, simple chord progressions, long harmonic durations, and \
predictable harmonic resolution.
- Short, memorable melodic phrases with gentle contours in a comfortable, mid-range register.
- Repetitive but non-monotonous structure: repetition with subtle variation and clear \
musical phrases.
- Limited instrumentation and minimal simultaneous musical information.
- Musical language the listener is likely to find familiar: genre, era, and cultural \
characteristics from the listener profile when given.
- Original music only: draw on high-level characteristics of familiar styles and never \
reproduce an existing song.

Sound design: warm, rounded, soft timbres with low transient intensity and a smooth \
frequency distribution; soft, even volume; moderate spatial depth with a steady stereo \
image. Keep the soundscape physically and cognitively non-intrusive.

Emotional trajectory: begin with something that feels familiar → establish safety and \
warmth → settle into a comfortable, predictable pattern → invite gentle engagement with \
clear, simple phrases → end emotionally supportive and calm, with a soft, resolved phrase \
or a slow fade. If a Session mode line is added in the request, follow that trajectory \
instead. The request lists the phases to write, in order, with what the music does in each.

Leave out entirely: sudden loud sounds, abrupt transitions, frightening sound effects, \
extreme dissonance, aggressive rhythmic patterns, complex polyrhythms, strong \
syncopation, rapid modulation, dense arrangements, unpredictable structural changes or \
dramatic events, overwhelming sonic density, highly virtuosic, fragmented, or constantly \
changing melodies, melancholic, aggressive, or chaotic moods, lyrics, sharp attacks, \
excessive high frequencies, heavy sub-bass, distorted sounds, sudden stereo movement, and \
experimental structures.

The piece should feel familiar, safe, warm, reassuring, gentle, peaceful, and emotionally \
stable: less like a conventional song and more like a supportive auditory environment, \
suitable for daily routines, wellbeing activities, and rest in dementia-care settings.

How to write:
- instruments: two or three from the list in the request, the lead first (only as many \
as the request allows), chosen for the listener's style, era and culture. The code \
decides which of them play in each phase, as the request explains.
- Describe only what IS heard, in plain positive words: "a gentle waltz on soft piano", \
"a warm guitar strum on each beat", "long, soft string chords". Text-to-audio models \
handle negation badly — a sound named after "no" tends to appear — so never name \
anything from the leave-out list, not even to exclude it. Let the sound carry the mood \
instead of emotion words, and say "mid-range melody" rather than "singing range".
- Each phase is one paragraph of 30 to 60 words on how the instruments behave in that \
phase: melody, rhythm, density, register, dynamics and space. No BPM or other numbers, \
no tags, no phase names, no song titles or artist names, no medical or therapeutic claims.
- Follow every setting, guideline line and the listener profile in the request. The \
profile shapes the musical style, never these safety rules. Add a sound the carer asks \
for only if it is soft, warm and steady; leave anything else out without mentioning it.
- purpose: one of Comfort, Relaxation, Wellbeing, Reminiscence, Sleep.
- genres: two short genre names for a gentle, widely familiar style from the listener's \
generation and culture (from the profile when given), such as Easy Listening, Folk, \
Swing, Classical or Chinese Folk.
- addition: the carer's other wishes as a short positive phrase for the audio model; \
usually empty.

Respond with ONLY the JSON object shown at the end of the request."""

DEMENTIA_MODES = ('default', 'calming', 'reminiscence', 'social')
DEMENTIA_TIMES = ('morning', 'afternoon', 'evening', 'presleep')
DEMENTIA_ENDING_NAMES = ('close', 'fade', 'loop')

# what the music does in each phase: the guideline's session arc tables
DEMENTIA_PHASE_TEXT = {
    'familiar':               "Opens with a simple, familiar-sounding phrase in the listener's style at a steady tempo; nothing new or surprising.",
    'safe':                   'Warm, consonant harmony and an even volume; the pattern that carries the session is established.',
    'comfortable':            'A stable, predictable pattern; phrases return with subtle variation.',
    'engaging':               'Clear, memorable phrases in a comfortable singing range invite humming or tapping along; still simple.',
    'emotionally_supportive': 'Instruments thin out and the pace eases slightly.',
    'steady_and_familiar':    'Opens at a steady tempo with a simple, familiar melody and few instruments; no sudden change of mood.',
    'simplify_rhythm':        'Rhythmic complexity drops; the pulse is carried by piano or guitar.',
    'fewer_instruments':      'Instrumentation reduces to one or two; phrases repeat more often.',
    'calming_settle':         'The tempo stabilizes; harmony moves less; dynamics even out.',
    'spacious_calm':          'More space and long sustained tones; the melody reduces to a short, repeated phrase.',
    'calming_fade_or_loop':   'The spacious calm continues.',
}
# how the last phase ends, added to its line above
DEMENTIA_ENDING_TEXT = {
    'close': 'It ends on a soft, resolved phrase.',
    'fade':  'It ends with a slow fade to silence.',
    'loop':  'It holds its level and texture to the very end, so the end flows straight back into the start (a seamless loop).',
}
# the ending as the settings list names it, and as it goes into the last phase's prompt
DEMENTIA_ENDING_SETTING = {
    'close': 'soft close (the last phrase resolves softly)',
    'fade':  'fade (a slow, seamless fade to silence)',
    'loop':  'loop (a seamless loop, holding its level to the very end)',
}
DEMENTIA_ENDINGS = {
    'close': 'ending on a soft, resolved phrase',
    'fade':  'a slow, seamless fade to silence',
    'loop':  'a continuation that loops seamlessly, steady to the very end',
}

# The guideline's instruments. Qwen names them its own way ("a mellow clarinet");
# the first of these words in its name picks the guideline instrument it stands for.
DEMENTIA_INSTRUMENTS = ('soft piano', 'acoustic guitar', 'warm strings', 'gentle woodwinds', 'warm pads')
DEMENTIA_INSTRUMENT_WORDS = (
    ('piano', 'soft piano'), ('guitar', 'acoustic guitar'),
    ('string', 'warm strings'), ('violin', 'warm strings'), ('viola', 'warm strings'), ('cello', 'warm strings'),
    ('woodwind', 'gentle woodwinds'), ('flute', 'gentle woodwinds'), ('clarinet', 'gentle woodwinds'),
    ('oboe', 'gentle woodwinds'), ('pad', 'warm pads'), ('synth', 'warm pads'),
)
DEMENTIA_DEFAULT_PALETTE = ('soft piano', 'acoustic guitar', 'warm strings')
DEMENTIA_PERCUSSION = 'light acoustic percussion'
WORDLESS_VOICE = 'soft wordless voice'
# which instruments play in each phase (the rest of each arc plays the whole palette)
DEMENTIA_PERCUSSION_PHASES = {'engaging'}                  # "light percussion may join"
DEMENTIA_THINNED_PHASES = {'emotionally_supportive'}       # "instruments thin out": one fewer
DEMENTIA_REDUCED_PHASES = {'fewer_instruments', 'calming_settle', 'spacious_calm',
                           'calming_fade_or_loop'}         # "reduces to one or two"

DEMENTIA_PURPOSES = ('Comfort', 'Relaxation', 'Wellbeing', 'Reminiscence', 'Sleep')
DEMENTIA_DEFAULT_PURPOSE = {'default': 'Comfort', 'calming': 'Relaxation',
                            'reminiscence': 'Reminiscence', 'social': 'Wellbeing'}
DEMENTIA_GENRES = ('Easy Listening', 'Acoustic')          # when Qwen's don't pass
DEMENTIA_GUARDRAIL = ('soft even volume, gradual changes, consonant harmony, a steady, simple pulse, '
                      'one key, a clear uncluttered texture, a clean tone, soft rounded attacks, '
                      'smooth warm highs, gentle lows, a warm reassuring mood')

# The guideline's "Prompt add-ons" (Session mode, Time of day and the rest) and its
# "Troubleshooting" lines, each as (the guideline's own line, its positive form).
# The line goes to the orchestrator; the positive form into the audio prompts.
# Tempo ranges in a line are for the orchestrator: the code states the BPM.
DEMENTIA_ADDONS = {
    'calming':      ('Session mode: calming. Start at a steady, familiar pace, then progressively reduce '
                     'rhythmic complexity and instrumentation, slow or stabilize the tempo within 45–65 BPM, '
                     'increase repetition, reduce harmonic movement and dynamic variation, and increase '
                     'spaciousness. Change gradually; never shift the mood abruptly.',
                     'gradual changes, increasing repetition and space'),
    'reminiscence': ('Session mode: reminiscence. Base the style on the listener profile: familiar genre '
                     'characteristics, era-specific instrumentation, cultural musical traditions, and familiar '
                     'harmonic and rhythmic patterns. Compose original music; do not reproduce any existing song.',
                     ''),
    'social':       ('Session mode: social interaction. A clear, steady pulse within 60–80 BPM and short, '
                     'memorable phrases that are easy to hum along to; keep the arrangement light.',
                     'a clear, steady pulse, short memorable phrases, a light arrangement'),
    'morning':      ('Time of day: morning. Gentle brightness, a moderate tempo within 60–80 BPM, clear melodic '
                     'phrases, light instrumentation, and a positive but non-exciting tone.',
                     'gentle brightness, clear melodic phrases, light instrumentation, a calm, positive tone'),
    'afternoon':    ('Time of day: afternoon. Stable, comfortable, and moderately engaging, with familiar '
                     'rhythmic patterns.',
                     'stable, comfortable and moderately engaging, familiar rhythmic patterns'),
    'evening':      ('Time of day: evening. A slower tempo within 45–65 BPM, lower musical density, warm '
                     'timbres, reduced rhythmic activity, and gradual relaxation.',
                     'lower musical density, warm timbres, reduced rhythmic activity, gradual relaxation'),
    'presleep':     ('Time of day: pre-sleep. Very low stimulation, long sustained tones, minimal rhythmic '
                     'activity, slow harmonic movement, and a seamless loop instead of an ending.',
                     'very low stimulation, long sustained tones, minimal rhythmic activity, slow harmonic movement'),
    'fewer':        ('Use no more than two instruments at a time, with minimal simultaneous musical information.',
                     'at most two instruments at a time, minimal simultaneous musical information'),
    'voice':        ('Add a soft, wordless vocal texture low in the mix; no lyrics.',
                     'a soft, wordless vocal texture low in the mix'),
    'familiar':     ('Feature {instrument}, played softly and simply.',
                     '{instrument} featured, played softly and simply'),
    # the tailoring table's "simplify further" melody, in the Calming arc's own words
    'simpler':      ('Simplify the melody further: a short, repeated phrase.',
                     'a melody of one short, repeated phrase'),
}
DEMENTIA_FIXES = {
    'unfamiliar': ("Use a simple, widely familiar style from the listener's era and culture; no experimental structures.",
                   "a simple, widely familiar style from the listener's era and culture"),
    'busy':       ('Use no more than two instruments at a time, with minimal simultaneous musical information.',
                   'at most two instruments at a time, minimal simultaneous musical information'),
    'wanders':    ('Use short, memorable phrases that return often with only subtle variation.',
                   'short, memorable phrases that return often with only subtle variation'),
    'monotonous': ('Keep the repetition, but vary the phrases subtly every few repeats.',
                   'repetition with the phrases varied subtly every few repeats'),
    'percussion': ('Percussion must be light, acoustic, and on the beat; no syncopation or accents.',
                   'percussion light, acoustic and on the beat'),
    'loud':       ('Keep the volume even and make every change gradual; no sudden loud sounds or abrupt transitions.',
                   'even volume, every change gradual'),
    'tempo':      ('Hold a constant tempo from start to finish, with no speeding up or slowing down.',
                   'one constant tempo from start to finish'),
    'sad':        ('The mood is warm, reassuring, and emotionally stable; never melancholic, eerie, or frightening.',
                   'a warm, reassuring, emotionally stable mood'),
    'lyrics':     ('No lyrics or spoken words; any voice is a soft, wordless texture.',
                   'any voice a soft, wordless texture'),
    'song':       ('Compose an original melody; draw only on the general style, not on any existing song.',
                   ''),
    'unsettled':  ('Reduce stimulation step by step: fewer instruments, simpler rhythm, steadier tempo, and more space.',
                   'stimulation reducing step by step: fewer instruments, simpler rhythm, more space'),
}

# The guideline's listener profile, field by field, and the values its chips offer.
DEMENTIA_PROFILE = (
    ('age', 'Age'),
    ('culture', 'Cultural background'),
    ('preferred_music', 'Preferred music'),
    ('favorites', 'Favorite artists / genres'),
    ('era', 'Familiar era'),
    ('memories', 'Personal memories and life experiences'),
    ('state', 'Current emotional state'),
    ('activity', 'Current activity'),
    ('outcome', 'Desired outcome'),
    ('sensitivity', 'Sensory sensitivity'),
    ('environment', 'Listening environment'),
)
DEMENTIA_PROFILE_CHOICES = {
    'state':   ('calm', 'anxious', 'agitated', 'restless', 'lonely', 'sad', 'confused', 'emotionally withdrawn'),
    'outcome': ('relaxation', 'emotional comfort', 'gentle engagement', 'reminiscence', 'social interaction',
                'morning orientation', 'evening calming', 'sleep preparation'),
}

DEMENTIA_AVOIDED = CARE_AVOIDED + (
    r'|experimental|avant\w*|virtuos\w*|fragment\w*|dense|busy|clutter\w*|modulat\w*|'
    r'anxi\w*|agitat\w*|restless\w*|confus\w*'
)


def dementia_arc_key(session_mode):
    """Calming mode has its own arc; Reminiscence and Social change the style, not the arc."""
    return 'calming' if session_mode == 'calming' else 'default'


def dementia_bpm_range(session_mode, time_of_day):
    """The tempo range for this session: "50–75 BPM by default; 45–65 BPM for calming or
    evening; 60–80 BPM for gentle daytime engagement". A Session mode or Time of day
    line with a range of its own replaces the default; both apply when both have one.
    Returns (lo, hi, what the range is called in a warning)."""
    keys = [key for key in (session_mode, time_of_day) if key in config.DEMENTIA_BPM_BY]
    if not keys:
        return (*config.DEMENTIA_DEFAULT_BPM_RANGE, 'dementia default')
    lo, hi = config.DEMENTIA_BPM
    for key in keys:
        a, b = config.DEMENTIA_BPM_BY[key]
        lo, hi = max(lo, a), min(hi, b)
    return lo, hi, ' '.join(['dementia'] + keys)


def dementia_endings(session_mode, time_of_day):
    """The endings a session may have, its default first. Pre-sleep asks for "a seamless
    loop instead of an ending"; the Calming arc ends in "Fade or loop"; the tailoring
    table suggests the loop "for evening and pre-sleep"."""
    if time_of_day == 'presleep':
        return ('loop',)
    endings = ('fade', 'loop') if session_mode == 'calming' else ('close', 'fade', 'loop')
    if time_of_day == 'evening':
        endings = ('loop',) + tuple(e for e in endings if e != 'loop')
    return endings


def familiar_instrument(value, log):
    """The carer's familiar instrument as it goes into the prompt, or '' if it can't:
    one short name, not percussion, a voice or an avoided sound."""
    text = re.sub(r'\s+', ' ', str(value or '')).strip().strip('.,;:!?').lower()
    text = re.sub(r'^(?:a|an|the)\s+', '', text)
    if not text:
        return ''
    if (len(text) > 40 or not any(c.isalpha() for c in text)
            or word_filter(DEMENTIA_AVOIDED, PERCUSSION_WORDS, VOICE_WORDS).search(text)):
        log(f'warning: familiar instrument {value!r} left out: it must be one soft instrument')
        return ''
    return text


def dementia_settings(options, session_request, log):
    """The carer's choices from the page, kept inside the guideline's safe ranges and
    consistent with each other: the Session mode and Time of day lines narrow the
    tempo, the ending and percussion. Anything missing takes the guideline default."""
    o = options if isinstance(options, dict) else {}
    minutes = session_minutes(o, session_request, config.DEMENTIA_MINUTES,
                              config.DEMENTIA_DEFAULT_MINUTES, 'dementia', log)
    session_mode = choice(o, 'session_mode', DEMENTIA_MODES, 'default', log)
    time_of_day = choice(o, 'time_of_day', ('none',) + DEMENTIA_TIMES, 'none', log)

    lo, hi, what = dementia_bpm_range(session_mode, time_of_day)
    bpm = in_range(whole_number('bpm', o.get('bpm', config.DEMENTIA_DEFAULT_BPM)), lo, hi, 'BPM', what, log)

    endings = dementia_endings(session_mode, time_of_day)
    ending = choice(o, 'ending', DEMENTIA_ENDING_NAMES, endings[0], log)
    if ending not in endings:
        why = 'pre-sleep sessions loop' if time_of_day == 'presleep' else 'the calming arc ends with a fade or a loop'
        log(f'warning: a {ending} ending does not fit ({why}), using {endings[0]}')
        ending = endings[0]

    density = choice(o, 'density', ('usual', 'fewer'), 'usual', log)
    melody = choice(o, 'melody', ('short', 'simpler'), 'short', log)
    percussion = choice(o, 'percussion', ('light', 'off'), 'light', log)
    voice = choice(o, 'voice', ('off', 'wordless'), 'off', log)
    familiar = familiar_instrument(o.get('familiar_instrument'), log)
    fixes = fix_list(o, DEMENTIA_FIXES, log)

    # "Off for calming, evening and pre-sleep"; and percussion is one of the instruments,
    # so with no more than two at a time (Fewer instruments, or the "too busy" fix) it stays out
    if percussion == 'light' and (session_mode == 'calming' or time_of_day in ('evening', 'presleep')
                                  or density == 'fewer' or 'busy' in fixes):
        if o.get('percussion') == 'light':
            log('warning: percussion is off for calming, evening and pre-sleep sessions, '
                'and with no more than two instruments at a time')
        percussion = 'off'
    if session_mode != 'calming':
        drop_fix(fixes, 'unsettled', 'it is for calming sessions', log)
    if percussion == 'off':
        drop_fix(fixes, 'percussion', 'this session has no percussion', log)

    profile = listener_profile(o.get('profile'), DEMENTIA_PROFILE, DEMENTIA_PROFILE_CHOICES, log)

    return {'minutes': minutes, 'session_mode': session_mode, 'time_of_day': time_of_day,
            'bpm': bpm, 'ending': ending, 'density': density, 'melody': melody,
            'percussion': percussion, 'voice': voice, 'familiar_instrument': familiar,
            'fixes': fixes, 'profile': profile}


def dementia_limit(settings):
    """How many instruments the palette holds: 1–3, default 2–3; at most two with
    Fewer instruments or the "too busy or layered" fix."""
    return 2 if settings['density'] == 'fewer' or 'busy' in settings['fixes'] else 3


def dementia_lines(settings):
    """(guideline lines for the orchestrator, [(audio phrase, phases or None)]) for the
    Session mode, Time of day, add-ons and fixes these settings switch on, each once."""
    addons = [key for key in (settings['session_mode'], settings['time_of_day']) if key in DEMENTIA_ADDONS]
    if settings['density'] == 'fewer':
        addons.append('fewer')
    if settings['melody'] == 'simpler':
        addons.append('simpler')
    if settings['voice'] == 'wordless':
        addons.append('voice')
    if settings['familiar_instrument']:
        addons.append('familiar')

    entries = [(DEMENTIA_ADDONS[key], None) for key in addons]
    for fix in settings['fixes']:
        line, phrase = DEMENTIA_FIXES[fix]
        if fix == 'lyrics' and settings['voice'] == 'off':
            phrase = 'purely instrumental'   # "any voice" would invite one
        entries.append(((line, phrase), DEMENTIA_PERCUSSION_PHASES if fix == 'percussion' else None))
    return collect_lines(entries, instrument=f"the {settings['familiar_instrument']}")


def dementia_instrument(item):
    """The guideline instrument a name of Qwen's stands for, or None."""
    text = str(item).lower()
    for word, name in DEMENTIA_INSTRUMENT_WORDS:
        if word in text:
            return name
    return None


def dementia_palette(raw, settings, notes):
    """The session's instruments, the lead first: the carer's familiar instrument, then
    Qwen's picks mapped onto the guideline's list, filled from the default up to two,
    cut to the density limit. Calming mode keeps piano or guitar in the first two,
    because they carry the pulse once the arc reduces to two instruments."""
    featured = settings['familiar_instrument']
    palette = [featured] if featured else []
    featured_as = dementia_instrument(featured) if featured else None
    if isinstance(raw, str):
        raw = [raw]
    for item in raw if isinstance(raw, list) else []:
        name = dementia_instrument(item)
        if name is None:
            notes.append(f'instrument dropped: {item!r}')
        elif name not in palette and name != featured_as:
            palette.append(name)
    for name in DEMENTIA_DEFAULT_PALETTE:
        if len(palette) >= 2:
            break
        if name not in palette and name != featured_as:
            palette.append(name)
    limit = dementia_limit(settings)
    palette = palette[:limit]
    pulse = {'soft piano', 'acoustic guitar'}
    if settings['session_mode'] == 'calming' and not any(dementia_instrument(p) in pulse for p in palette[:2]):
        palette.insert(1 if featured else 0, 'soft piano')
        palette = palette[:limit]
    return palette


def dementia_phase_instruments(arc, palette, settings):
    """Which instruments play in each phase: the whole palette, one fewer where
    "instruments thin out", the first two where the Calming arc "reduces to one or
    two", light percussion where it "may join", the wordless voice throughout.

    Percussion is one of the instruments, so it counts toward the three at a time:
    where it joins, a third instrument steps back (dementia_settings already turns it
    off when only two are allowed). The voice is a texture, not counted.
    """
    limit = dementia_limit(settings)
    phase_instruments = {}
    for name, _ in arc:
        if name in DEMENTIA_REDUCED_PHASES:
            melodic = palette[:2]
        elif name in DEMENTIA_THINNED_PHASES and len(palette) > 1:
            melodic = palette[:-1]
        else:
            melodic = list(palette)
        percussion = settings['percussion'] == 'light' and name in DEMENTIA_PERCUSSION_PHASES
        # who plays when there are more than the limit: lead, second, percussion, third
        order = melodic[:2] + ([DEMENTIA_PERCUSSION] if percussion else []) + melodic[2:]
        playing = order[:limit]
        instruments = [i for i in melodic if i in playing] + [i for i in playing if i == DEMENTIA_PERCUSSION]
        if settings['voice'] == 'wordless':
            instruments.append(WORDLESS_VOICE)
        phase_instruments[name] = instruments
    return phase_instruments


def dementia_avoided(settings, arc):
    """The avoided-words filter for each phase. Voice words pass only with the wordless
    voice on, brightness only in the morning ("gentle brightness"), percussion only
    in the phase where it plays."""
    groups = [DEMENTIA_AVOIDED]
    if settings['voice'] == 'off':
        groups.append(VOICE_WORDS)
    if settings['time_of_day'] != 'morning':
        groups.append(r'bright\w*')
    with_percussion = word_filter(*groups)
    without_percussion = word_filter(*groups, PERCUSSION_WORDS)
    return {name: with_percussion if settings['percussion'] == 'light' and name in DEMENTIA_PERCUSSION_PHASES
            else without_percussion for name, _ in arc}


def dementia_user_message(session_request, settings, arc, guideline_lines):
    """What Qwen is told about this particular session."""
    s = settings
    limit = dementia_limit(s)
    featured = s['familiar_instrument']
    last = arc[-1][0]
    choose = ('one more' if featured and limit == 2 else 'one or two more' if featured
              else 'two' if limit == 2 else 'two or three')
    if dementia_arc_key(s['session_mode']) == 'calming':
        plays = ('all of them in steady_and_familiar and simplify_rhythm; only the first two '
                 'from fewer_instruments on')
    else:
        plays = 'all of them in every phase, except that emotionally_supportive leaves out the last one listed'
        if s['percussion'] == 'light':
            plays += ', and in engaging light percussion takes the place of a third instrument'
    lines = [
        f'Session request: "{session_request}"',
        '',
        'Settings (the code enforces tempo, length, which instruments play in each phase, and the ending):',
        f'- Session mode: {s["session_mode"]}',
        f'- Time of day: {TIME_OF_DAY_NAMES[s["time_of_day"]]}',
        f'- Tempo: {s["bpm"]} BPM, constant from start to finish',
    ]
    if featured:
        lines.append(f'- Featured instrument, always listed first and always playing: {featured}')
    lines += [
        f'- instruments: choose {choose} from: {", ".join(DEMENTIA_INSTRUMENTS)}',
        f'- Which instruments play: {plays}',
    ]
    if s['session_mode'] == 'calming':
        lines.append('- soft piano or acoustic guitar must be among the first two: it carries the pulse')
    lines += [
        '- Percussion: ' + ('light acoustic percussion joins in the engaging phase only, soft and on the beat'
                            if s['percussion'] == 'light' else 'none'),
        '- Voice: ' + ('a soft, wordless vocal texture low in the mix, in every phase'
                       if s['voice'] == 'wordless' else 'none, purely instrumental'),
        '- Melody: ' + ('simpler: one short, repeated phrase' if s['melody'] == 'simpler'
                        else 'short phrases with gentle repetition'),
        f'- Ending: {DEMENTIA_ENDING_SETTING[s["ending"]]}',
        '',
        'Phases, in order (share of the session: what the music does):',
    ]
    for name, share in arc:
        text = DEMENTIA_PHASE_TEXT[name]
        if name in DEMENTIA_PERCUSSION_PHASES and s['percussion'] == 'light':
            text += ' Light percussion may join.'
        if name == last:
            text += ' ' + DEMENTIA_ENDING_TEXT[s['ending']]
        lines.append(f'- {name} ({share:.0%}): {text}')
    if guideline_lines:
        lines += ['', 'Guideline lines added for this session (follow every one):']
        lines += [f'- {line}' for line in guideline_lines]
    lines += profile_block(s['profile'], DEMENTIA_PROFILE)
    lines += ['', 'Respond with ONLY this JSON object:',
              json_skeleton(['"purpose": "..."', '"genres": ["...", "..."]',
                             '"instruments": ["...", "..."]', '"addition": ""'], arc)]
    return '\n'.join(lines)


def clean_dementia_plan(data, settings, arc):
    """Validate one parsed reply; returns (plan, notes) like clean_pain_relief_plan."""
    avoided = dementia_avoided(settings, arc)
    phases, notes = clean_phases(data, arc, avoided)
    purpose = clean_purpose(data.get('purpose'), DEMENTIA_PURPOSES,
                            DEMENTIA_DEFAULT_PURPOSE[settings['session_mode']])
    genres = clean_genres(data.get('genres'), DEMENTIA_GENRES, notes, avoided[arc[0][0]])
    palette = dementia_palette(data.get('instruments'), settings, notes)
    addition = clean_addition(data.get('addition'), avoided[arc[0][0]], notes)
    return {'purpose': purpose, 'genres': genres, 'instruments': palette,
            'addition': addition, 'phases': phases}, notes


def generate_dementia_plan(session_request, settings, arc, guideline_lines, log, attempts=3):
    """Call Qwen until one reply passes clean_dementia_plan, or give up."""
    user_message = dementia_user_message(session_request, settings, arc, guideline_lines)
    return orchestrate(user_message, DEMENTIA_SYSTEM_PROMPT,
                       lambda data: clean_dementia_plan(data, settings, arc),
                       log, 'dementia', attempts)


def assemble_dementia_prompts(plan, settings, arc, audio_phrases):
    """One Stable Audio 3 prompt per phase, in the same tag format as the others.

    Tempo, genres and add-on phrases are the same in every phase; the Instruments
    tag follows dementia_phase_instruments, so a thinning arc really thins.
    """
    tempo = f'Steady {settings["bpm"]} BPM, simple meter.'
    phase_instruments = dementia_phase_instruments(arc, plan['instruments'], settings)
    last = arc[-1][0]
    prompts = {}
    for name, _ in arc:
        tags = prompt_tags(plan['purpose'], plan['genres'], phase_instruments[name],
                           instrumental=settings['voice'] == 'off')
        additions = [DEMENTIA_ENDINGS[settings['ending']]] if name == last else []
        additions += [phrase for phrase, phases in audio_phrases if phases is None or name in phases]
        if plan['addition']:
            additions.append(plan['addition'])
        additions.append(DEMENTIA_GUARDRAIL)
        description = plan['phases'][name].rstrip(' .,;:!?')
        prompts[name] = f'{tags} {tempo} {description}, Addition Request: {"; ".join(additions)}'
    return prompts


# ============================================================
# Section 5e — Module 3: depression orchestration
# ============================================================
#
# Music that meets a low mood where it is, then supports comfort, relaxation or
# gentle engagement: emotional congruence followed by gradual regulation, never
# forced cheer. The Session mode line switches the arc (Comfort is the default);
# each mode pairs with its own Time of day line. The instruments come from the
# arc text ("sparse piano or pads", "warm strings or guitar join"): the code
# holds the palette and Qwen chooses the "or" from the listener profile.

DEPRESSION_SYSTEM_PROMPT = """You are the orchestration layer of a functional-music \
generation pipeline, composing for Module 3 (Depression) of the ResonaLife guideline for \
hospice and chronic illness. The listener is experiencing depressive symptoms; often a \
carer sets the session up for them. You plan ONE continuous piece that moves through the \
phases listed in the request. Code builds the final text-to-audio prompts around your \
words and fixes the tags, instruments, tempo, durations and ending — your job is what the \
music does inside them.

Goal: provide a sense of warmth, safety, emotional acceptance, and gradual positive \
movement without forcing happiness or overwhelming the listener. Acknowledge the \
listener's emotional state rather than attempting to immediately override it: emotional \
congruence followed by gradual regulation, never forced positivity. The music should \
support wellbeing environments without making medical or therapeutic claims; it does not \
treat, cure, prevent, or reverse depression or replace professional care.

Musical characteristics:
- A slow tempo with a stable pulse; any tempo change is gradual, never sudden. The code \
states the BPM for each phase; never write a number yourself.
- Simple rhythmic patterns: begin with minimal rhythmic activity and introduce a gentle \
pulse gradually.
- Warm consonant harmonies with gentle progression and moderate stability, moving \
gradually toward more open harmonic colors. Minor-key colors may appear for emotional \
resonance but must not dominate.
- Simple, memorable, warm melodic phrases that develop gradually, with subtle \
forward-moving contours.
- Clear musical phrases, predictable harmonic movement, moderate repetition, and smooth \
transitions.
- Low cognitive demand: limited instrumentation and nothing that requires active \
concentration.

Sound design: warm, rounded, spacious, soft-edged, and comfortable sound; moderately \
immersive; smooth highs, gentle lows, soft dynamics and a steady stereo image.

Emotional trajectory: emotional heaviness → safety → warmth → calm. Open quietly to \
acknowledge a heavy mood without deepening it, use minimal stimulation and emotional \
acceptance, then settle into calm warmth. Do not force the music toward happiness. If a \
Session mode line is added in the request, follow that trajectory instead. The request \
lists the phases to write, in order, with what the music does in each.

Leave out entirely: excessive sadness, dramatic melancholy, emotional heaviness that \
deepens, prolonged unresolved dissonance, excessive harmonic darkness, dramatic minor-key \
emphasis, sudden emotional or harmonic shifts, artificially cheerful or euphoric music, \
aggressive percussion, excessive syncopation, chaotic or abrupt rhythmic changes, sudden \
volume increases, highly dramatic or emotionally manipulative melodies, excessively \
repetitive motifs, lyrics, harsh high frequencies, aggressive transients, heavy sub-bass, \
distortion, abrupt stereo movement, and excessively bright or euphoric sound.

The piece should feel warm, safe, accepting, calm, gently hopeful, and supportive: human \
rather than artificially happy. It is a seamless long-form composition suitable for \
repeated listening during rest, daily activities, emotional decompression, or sleep \
preparation.

How to write:
- Use only the instruments the request lists for each phase.
- Describe only what IS heard, in plain positive words: "sparse piano notes over a soft \
pad", "a gentle guitar pulse", "long, warm string chords". Text-to-audio models handle \
negation badly — a sound named after "no" tends to appear — so never name anything from \
the leave-out list, not even to exclude it. Let the sound carry the mood (tempo feel, \
register, density, harmony, dynamics); never use emotion words such as sad, heavy, happy, \
tense or anxious.
- Each phase is one paragraph of 30 to 60 words on how the instruments behave in that \
phase: melody, rhythm, density, register, dynamics and space. No BPM or other numbers, \
no tags, no phase names, no song titles or artist names, no medical or therapeutic claims.
- Follow every setting, guideline line and the listener profile in the request. The \
profile shapes the music, never these emotional-safety rules. Add a sound the carer asks \
for only if it is soft, warm and steady; leave anything else out without mentioning it.
- purpose: one of Comfort, Relaxation, Wellbeing, Focus, Sleep.
- genres: two short genre names that suit the listener's preferred genres when given, \
otherwise two of Acoustic, Neoclassical, Ambient, New Age, Folk.
- addition: the carer's other wishes as a short positive phrase for the audio model; \
usually empty.

Respond with ONLY the JSON object shown at the end of the request."""

DEPRESSION_MODES = ('comfort', 'activation', 'relaxation', 'sleep')
DEPRESSION_MODE_NAMES = {'comfort': 'Comfort', 'activation': 'Gentle activation',
                         'relaxation': 'Relaxation', 'sleep': 'Sleep preparation'}
# "Pair each part of the day with the session mode shown"
DEPRESSION_TIMES = {'comfort': (), 'activation': ('morning', 'daytime'),
                    'relaxation': ('evening',), 'sleep': ('presleep',)}
# Relaxation "ends with a slow fade", Sleep preparation "ends in a seamless loop", and
# Gentle activation ends at a steady, gentle energy that a loop would drop straight
# back to stillness; only Comfort offers the choice. Default first.
DEPRESSION_ENDINGS_BY_MODE = {'comfort': ('fade', 'loop'), 'activation': ('fade',),
                              'relaxation': ('fade',), 'sleep': ('loop',)}
# the guideline's "Matching energy level" table
DEPRESSION_ENERGY = {
    'low':      'Slower tempo, sparse arrangement, soft dynamics, long sustained tones, simple melodic movement',
    'moderate': 'Slightly stronger pulse, more defined melodic phrases, moderate rhythmic activity, '
                'gradually expanding instrumentation',
    'higher':   'Moderate rhythmic movement, clear pulse, brighter but still comfortable timbres, '
                'more forward-moving harmonic progression',
}

# What the music does in each phase. Comfort mode is the guideline's session arc
# table. For the other modes the guideline gives the phases and what changes across
# them ("Other session modes"); these lines spread that across the phases.
DEPRESSION_PHASE_TEXT = {
    'acknowledge_heaviness':  'Quiet, slow and a little subdued: sparse piano or pads, minimal rhythm; meets a heavy mood without deepening it.',
    'safety':                 'Harmony settles on warm, consonant chords; a soft, steady pulse; nothing changes suddenly.',
    'warmth':                 'Warm strings or guitar join; simple, memorable phrases; harmony opens gently toward brighter colors.',
    'calm':                   'Density eases back into long sustained tones.',
    'stillness':              'Begins still, at the starting tempo, with minimal rhythm.',
    'comfort':                'Warm, consonant comfort; a gentle pulse begins as the joining instrument enters softly.',
    'engagement':             'A clearer pulse and more melodic movement; the harmony widens and opens.',
    'subtle_positive_energy': 'Wider, more open harmonic movement at a steady, gentle energy that holds rather than climbs.',
    'tension':                'Starts where the listener is: some rhythmic activity and a fuller texture, warm and consonant.',
    'relaxation_calm':        'Rhythmic activity and musical density begin to reduce; phrases lengthen.',
    'spaciousness':           'More space: fewer notes, longer tones, a wider, airier texture.',
    'deep_relaxation':        'Minimal rhythm and long sustained tones.',
    'emotional_fatigue':      'Slow and gentle, meeting tiredness: minimal rhythm and soft sustained tones.',
    'sleep_safety':           'Warm, stable harmony; nothing changes suddenly.',
    'sleep_calm':             'Calm, sustained tones; rhythmic activity fades further.',
    'low_stimulation':        'Very low stimulation: long sustained tones and slow harmonic movement.',
    'sleep_readiness':        'Barely moving sustained tones and seamless transitions.',
}
DEPRESSION_ENDING_TEXT = {
    'fade': 'It ends with a slow fade to silence.',
    'loop': 'It holds its level and texture to the very end, so the end flows straight back into the start (a seamless loop).',
}
DEPRESSION_ENDING_SETTING = {
    'fade': 'fade (a slow, seamless fade to silence)',
    'loop': 'loop (a seamless loop, holding its level to the very end)',
}
DEPRESSION_ENDINGS = {
    'fade': 'a slow, seamless fade to silence',
    'loop': 'a continuation that loops seamlessly, steady to the very end',
}

DEPRESSION_OPENING = ('soft piano', 'warm pads')                        # "sparse piano or pads"
DEPRESSION_SLEEP_PALETTE = ('warm pads', 'soft piano', 'warm strings')   # sustained tones throughout
# the instrument that joins the opening pair (in Relaxation, the one that leaves it),
# from the guideline's instrument list; Qwen picks one for the listener, the first is
# the default. ("Natural acoustic textures" is left out: too vague for the audio model,
# which tends to answer it with field recordings.)
DEPRESSION_JOINERS = {
    'comfort':    ('warm strings', 'acoustic guitar'),                        # "Warm strings or guitar join"
    'activation': ('acoustic guitar', 'soft marimba', 'gentle woodwinds'),    # "a clearer pulse ... more melodic movement"
    'relaxation': ('acoustic guitar', 'warm strings', 'gentle woodwinds'),    # the moving layer that thins out
}
DEPRESSION_JOINER_WORDS = (('string', 'warm strings'), ('violin', 'warm strings'), ('cello', 'warm strings'),
                           ('guitar', 'acoustic guitar'), ('marimba', 'soft marimba'), ('mallet', 'soft marimba'),
                           ('woodwind', 'gentle woodwinds'), ('flute', 'gentle woodwinds'),
                           ('clarinet', 'gentle woodwinds'))
DEPRESSION_JOINER_PHASES = {'warmth', 'calm', 'comfort', 'engagement', 'subtle_positive_energy',
                            'tension', 'relaxation_calm'}
DEPRESSION_PERCUSSION = 'light percussion'
DEPRESSION_PERCUSSION_PHASES = {'engagement', 'subtle_positive_energy'}   # Gentle activation only

DEPRESSION_PURPOSES = ('Comfort', 'Relaxation', 'Wellbeing', 'Focus', 'Sleep')
DEPRESSION_DEFAULT_PURPOSE = {'comfort': 'Comfort', 'activation': 'Wellbeing',
                              'relaxation': 'Relaxation', 'sleep': 'Sleep'}
DEPRESSION_GENRES = ('Acoustic', 'Neoclassical')
DEPRESSION_GUARDRAIL = ('warm consonant harmony, sincere and gently hopeful, a clean tone, soft rounded '
                        'attacks, smooth warm highs, gentle lows, soft dynamics, gradual changes, a simple warm melody')

DEPRESSION_ADDONS = {
    'activation': ('Session mode: gentle activation. Stillness → comfort → engagement → subtle positive energy. '
                   'Begin at 50–65 BPM with minimal rhythm, then gradually move toward 65–80 BPM, adding a clearer '
                   'pulse, light percussion, more melodic movement, and wider, more open harmonic movement. Never '
                   'jump from low to high energy.',
                   ''),
    'relaxation': ('Session mode: relaxation. Tension → calm → spaciousness → deep relaxation. Progressively reduce '
                   'rhythmic activity and musical density, staying within 50–65 BPM.',
                   ''),
    'sleep':      ('Session mode: sleep preparation. Emotional fatigue → safety → calm → low stimulation → sleep '
                   'readiness. Slow tempo, minimal rhythmic activity, sustained tones, and seamless transitions; end '
                   'with a seamless loop instead of a fade.',
                   'minimal rhythmic activity, sustained tones, seamless transitions'),
    'morning':    ('Time of day: morning. Moderate tempo, clear melodic phrases, warm but slightly brighter timbres, '
                   'a gentle rhythmic pulse, and a gradual increase in musical energy.',
                   'clear melodic phrases, warm but slightly brighter timbres, a gentle rhythmic pulse'),
    'daytime':    ('Time of day: daytime. Moderate tempo within 65–80 BPM, clear rhythmic structure, moderate '
                   'musical density, and gentle forward motion.',
                   'clear rhythmic structure, moderate musical density, gentle forward motion'),
    'evening':    ('Time of day: evening. Slower tempo, reduced rhythmic activity, warm timbres, lower musical '
                   'density, and a more spacious arrangement.',
                   'reduced rhythmic activity, warm timbres, lower musical density, a spacious arrangement'),
    'presleep':   ('Time of day: pre-sleep. Slow tempo, minimal rhythmic activity, sustained tones, stable harmony, '
                   'and very gradual transitions.',
                   'minimal rhythmic activity, sustained tones, stable harmony, very gradual transitions'),
    'sparser':    ('Use a sparse arrangement with long sustained tones, soft dynamics, and very little rhythm.',
                   'a sparse arrangement, long sustained tones, soft dynamics, very little rhythm'),
    'resonance':  ('Allow gentle minor-key colors for emotional resonance, always returning to warm, open harmony.',
                   'gentle minor-key colors for resonance, always returning to warm, open harmony'),
    # the tailoring table's Brightness row: "slightly brighter for morning or activation;
    # never bright or euphoric"
    'brighter':   ('Warm but slightly brighter timbres; never bright or euphoric.',
                   'warm but slightly brighter timbres'),
}
DEPRESSION_FIXES = {
    'heavy':       ('Acknowledge the mood gently, then move toward warmth; no dramatic melancholy or dark harmony.',
                    'gentle and warm, moving toward open harmony'),
    'cheerful':    ('No artificially cheerful or euphoric music; keep the tone warm and calm, with only subtle '
                    'positive energy.',
                    'a warm, calm tone with only subtle positive energy'),
    'rises':       ('Increase energy very gradually across the whole piece; no sudden changes in tempo, rhythm, '
                    'or volume.',
                    'every change in tempo, rhythm and volume very gradual'),
    'never_lifts': ('Over the second half, gradually introduce a clearer pulse, more melodic movement, and more '
                    'open harmony.',
                    'a gradually clearer pulse, more melodic movement, more open harmony'),
    'minor':       ('Use warm, open harmony; minor colors only briefly, for resonance.',
                    'warm, open harmony'),
    'bright':      ('Keep timbres warm, rounded, and soft-edged; no harsh highs or bright, euphoric sound.',
                    'warm, rounded, soft-edged timbres'),
    'percussion':  ('Percussion must be light and soft, never driving; no aggressive or syncopated rhythms.',
                    'percussion light and soft'),
    'melody':      ('Use simple, warm phrases that develop gradually; no dramatic climaxes or repetitive motifs.',
                    'simple, warm phrases that develop gradually'),
    'lyrics':      ('Instrumental only; no lyrics or spoken words.',
                    'purely instrumental'),
}
# fix phrases that belong to some phases only: "over the second half", and percussion
DEPRESSION_FIX_PHASES = {'never_lifts': {'engagement', 'subtle_positive_energy'},
                         'percussion': DEPRESSION_PERCUSSION_PHASES}

DEPRESSION_PROFILE = (
    ('age', 'Age'),
    ('culture', 'Cultural background'),
    ('genres', 'Preferred genres'),
    ('artists', 'Favorite artists'),
    ('background', 'Musical background'),
    ('state', 'Current emotional state'),
    ('energy', 'Energy level'),
    ('activity', 'Current activity'),
    ('outcome', 'Desired functional outcome'),
    ('environment', 'Listening environment'),
    ('sensitivity', 'Sensory sensitivity'),
)
DEPRESSION_PROFILE_CHOICES = {
    'state':   ('low mood', 'emotional numbness', 'sadness', 'anxiety', 'loneliness', 'low motivation',
                'restlessness', 'emotional fatigue', 'calm but withdrawn'),
    'outcome': ('emotional comfort', 'relaxation', 'gentle activation', 'focus', 'daily activity',
                'exercise preparation', 'social engagement', 'morning activation', 'evening relaxation',
                'sleep preparation'),
}

DEPRESSION_AVOIDED = CARE_AVOIDED + (
    r'|heav\w*|despair\w*|hopeless\w*|trag\w*|grie\w*|isolat\w*|tears?|weep\w*|cr(?:y|ies|ying)|'
    r'tens(?:e|ion)\w*|anxi\w*|fear\w*|anger|angry|manipulat\w*|driving|euphori\w*|'
    r'(?:overly|too|very)\s+bright\w*|cheer\w*|happ(?:y|ily|iness)|joy\w*|jubilant|exuberant|'
    r'triumph\w*|upbeat|energetic|danc\w*|party|celebrat\w*'
)


def depression_settings(options, session_request, log):
    """The carer's choices from the page, kept inside the guideline's safe ranges.
    Each Session mode has its own Time of day, endings and adjustments (the tailoring
    table's "When to adjust"); a fix outranks the setting it corrects."""
    o = options if isinstance(options, dict) else {}
    minutes = session_minutes(o, session_request, config.DEPRESSION_MINUTES,
                              config.DEPRESSION_DEFAULT_MINUTES, 'depression', log)
    session_mode = choice(o, 'session_mode', DEPRESSION_MODES, 'comfort', log)

    time_of_day = choice(o, 'time_of_day', ('none', 'morning', 'daytime', 'evening', 'presleep'), 'none', log)
    if time_of_day != 'none' and time_of_day not in DEPRESSION_TIMES[session_mode]:
        log(f'warning: time of day {time_of_day!r} does not pair with {DEPRESSION_MODE_NAMES[session_mode]}, left out')
        time_of_day = 'none'

    # Daytime's "moderate tempo within 65–80 BPM" and Gentle activation's "begin at 50–65"
    # meet at 65: a daytime session starts there, so all of it stays within 65–80
    lo, hi = config.DEPRESSION_BPM
    if time_of_day == 'daytime':
        lo = max(lo, config.DEPRESSION_ACTIVATION_BPM[0])
    bpm = in_range(whole_number('bpm', o.get('bpm', max(lo, config.DEPRESSION_DEFAULT_BPM))), lo, hi, 'BPM',
                   'depression daytime' if time_of_day == 'daytime' else 'depression', log)
    bpm_end = None
    if session_mode == 'activation':
        bpm_end = in_range(whole_number('bpm_end', o.get('bpm_end', config.DEPRESSION_DEFAULT_ACTIVATION_BPM)),
                           *config.DEPRESSION_ACTIVATION_BPM, 'BPM', 'gentle activation', log)

    endings = DEPRESSION_ENDINGS_BY_MODE[session_mode]
    ending = choice(o, 'ending', ('fade', 'loop'), endings[0], log)
    if ending not in endings:
        log(f'warning: {DEPRESSION_MODE_NAMES[session_mode]} ends with a {endings[0]}, not a {ending}')
        ending = endings[0]

    energy = choice(o, 'energy', tuple(DEPRESSION_ENERGY), 'low', log)
    harmony = choice(o, 'harmony', ('warm', 'resonance'), 'warm', log)
    brightness = choice(o, 'brightness', ('warm', 'brighter'), 'warm', log)
    # Off by default, but Gentle activation's own line adds light percussion
    activation = session_mode == 'activation'
    percussion = choice(o, 'percussion', ('off', 'light'), 'light' if activation else 'off', log)
    sparser = o.get('sparser') is True
    if harmony == 'resonance' and session_mode != 'comfort':
        log('warning: more minor color is for Comfort mode, using warm harmony')
        harmony = 'warm'
    if brightness == 'brighter' and session_mode != 'activation':
        log('warning: slightly brighter is for Gentle activation, using warm')
        brightness = 'warm'
    if percussion == 'light' and session_mode != 'activation':
        log('warning: light percussion is for Gentle activation only, using none')
        percussion = 'off'
    if sparser and session_mode == 'activation':
        log('warning: sparser does not fit Gentle activation, left out')
        sparser = False

    fixes = fix_list(o, DEPRESSION_FIXES, log)
    if session_mode != 'activation':
        drop_fix(fixes, 'never_lifts', 'it is for Gentle activation sessions', log)
    if session_mode not in ('activation', 'comfort'):
        drop_fix(fixes, 'rises', 'energy only falls in this session mode', log)
    if percussion == 'off':
        drop_fix(fixes, 'percussion', 'this session has no percussion', log)
    if 'minor' in fixes:
        harmony = 'warm'        # "Minor key dominates" outranks More resonance
    if 'bright' in fixes:
        brightness = 'warm'     # "Sound is harsh or too bright" outranks Slightly brighter

    profile = listener_profile(o.get('profile'), DEPRESSION_PROFILE, DEPRESSION_PROFILE_CHOICES, log)

    return {'minutes': minutes, 'session_mode': session_mode, 'time_of_day': time_of_day,
            'bpm': bpm, 'bpm_end': bpm_end, 'ending': ending, 'energy': energy,
            'harmony': harmony, 'brightness': brightness, 'percussion': percussion,
            'sparser': sparser, 'fixes': fixes, 'profile': profile}


def depression_lines(settings):
    """(guideline lines for the orchestrator, [(audio phrase, phases or None)]) for the
    Session mode, Time of day, add-ons and fixes these settings switch on, each once."""
    addons = [key for key in (settings['session_mode'], settings['time_of_day']) if key in DEPRESSION_ADDONS]
    if settings['sparser']:
        addons.append('sparser')
    if settings['harmony'] == 'resonance':
        addons.append('resonance')
    if settings['brightness'] == 'brighter':
        addons.append('brighter')
    entries = [(DEPRESSION_ADDONS[key], None) for key in addons]
    if 'bright' in settings['fixes'] and settings['time_of_day'] == 'morning':
        line, _ = DEPRESSION_ADDONS['morning']      # the fix outranks "slightly brighter timbres"
        entries = [((line, 'clear melodic phrases, a gentle rhythmic pulse'), None) if key == 'morning' else entry
                   for key, entry in zip(addons, entries)]
    entries += [(DEPRESSION_FIXES[fix], DEPRESSION_FIX_PHASES.get(fix)) for fix in settings['fixes']]
    return collect_lines(entries)


def depression_phase_bpm(arc, settings):
    """The tempo of each phase: constant, except that Gentle activation "rises gradually
    from 50–65 toward 65–80 BPM" in even steps, one per phase, from the start tempo
    to the end tempo."""
    if settings['session_mode'] != 'activation':
        return {name: settings['bpm'] for name, _ in arc}
    start, end, steps = settings['bpm'], settings['bpm_end'], len(arc) - 1
    return {name: round(start + (end - start) * i / steps) for i, (name, _) in enumerate(arc)}


def depression_phase_instruments(arc, joiner, settings):
    """Which instruments play in each phase: the opening pair throughout, the joining
    instrument where the arc brings it in (or, in Relaxation, until it thins out),
    light percussion in Gentle activation's second half. Sleep preparation keeps one
    sustained palette so every transition is seamless."""
    phase_instruments = {}
    for name, _ in arc:
        if settings['session_mode'] == 'sleep':
            instruments = list(DEPRESSION_SLEEP_PALETTE)
        else:
            instruments = list(DEPRESSION_OPENING)
            if joiner and name in DEPRESSION_JOINER_PHASES:
                instruments.append(joiner)
        if settings['percussion'] == 'light' and name in DEPRESSION_PERCUSSION_PHASES:
            instruments.append(DEPRESSION_PERCUSSION)
        phase_instruments[name] = instruments
    return phase_instruments


def depression_joiner(value, options):
    """The joining instrument Qwen chose, as one of the mode's two options, or None."""
    text = str(value or '').lower()
    for word, name in DEPRESSION_JOINER_WORDS:
        if word in text and name in options:
            return name
    return None


def depression_avoided(settings, arc):
    """The avoided-words filter for each phase: instrumental only, and percussion words
    only where light percussion plays."""
    with_percussion = word_filter(DEPRESSION_AVOIDED, VOICE_WORDS)
    without_percussion = word_filter(DEPRESSION_AVOIDED, VOICE_WORDS, PERCUSSION_WORDS)
    return {name: with_percussion if settings['percussion'] == 'light' and name in DEPRESSION_PERCUSSION_PHASES
            else without_percussion for name, _ in arc}


def depression_user_message(session_request, settings, arc, guideline_lines):
    """What Qwen is told about this particular session."""
    s = settings
    joiners = DEPRESSION_JOINERS.get(s['session_mode'])
    last = arc[-1][0]
    bpms = depression_phase_bpm(arc, s)
    if s['session_mode'] == 'activation':
        tempo = ('Tempo by phase, rising gradually, never a jump: '
                 + ', '.join(f'{name} {bpm}' for name, bpm in bpms.items()) + ' BPM')
    else:
        tempo = f'Tempo: {s["bpm"]} BPM, constant from start to finish'
    lines = [
        f'Session request: "{session_request}"',
        '',
        'Settings (the code enforces tempo, length, instruments and the ending):',
        f'- Session mode: {DEPRESSION_MODE_NAMES[s["session_mode"]]}',
        f'- Time of day: {TIME_OF_DAY_NAMES[s["time_of_day"]]}',
        f'- {tempo}',
        f'- Energy level: {s["energy"]} ({DEPRESSION_ENERGY[s["energy"]].lower()}). '
        'The opening phase starts at this energy.',
        '- Harmony: ' + ('a little more minor color for resonance, always returning to warm, open harmony'
                         if s['harmony'] == 'resonance' else 'warm and consonant, gently opening'),
        '- Brightness: ' + ('warm but slightly brighter; never bright or euphoric'
                            if s['brightness'] == 'brighter' else 'warm'),
        '- Percussion: ' + ('light percussion in engagement and subtle_positive_energy, soft and never driving'
                            if s['percussion'] == 'light'
                            else 'none; the clearer pulse comes from the instruments' if s['session_mode'] == 'activation'
                            else 'none'),
        '- Arrangement: ' + ('sparser: long sustained tones, soft dynamics and very little rhythm'
                             if s['sparser'] else 'as the phases describe'),
        f'- Ending: {DEPRESSION_ENDING_SETTING[s["ending"]]}',
        '',
        'Instruments by phase:',
    ]
    groups = {}
    for name, instruments in depression_phase_instruments(arc, 'JOINING', s).items():
        groups.setdefault(', '.join(instruments).replace('JOINING', 'the joining instrument'), []).append(name)
    lines += [f'- {", ".join(names)}: {instruments}' for instruments, names in groups.items()]
    if joiners:
        lines.append(f'- joining: choose {" or ".join(joiners)}, whichever suits the listener better')
    lines += ['', 'Phases, in order (share of the session: what the music does):']
    for name, share in arc:
        text = DEPRESSION_PHASE_TEXT[name]
        if name in DEPRESSION_PERCUSSION_PHASES and s['percussion'] == 'light':
            text += ' Light percussion plays, soft and never driving.'
        if name == last:
            text += ' ' + DEPRESSION_ENDING_TEXT[s['ending']]
        lines.append(f'- {name} ({share:.0%}): {text}')
    if guideline_lines:
        lines += ['', 'Guideline lines added for this session (follow every one):']
        lines += [f'- {line}' for line in guideline_lines]
    profile = dict(s['profile'])
    profile['energy'] = s['energy']   # the Energy level setting is the profile's energy line
    lines += profile_block({key: profile[key] for key, _ in DEPRESSION_PROFILE if key in profile},
                           DEPRESSION_PROFILE)
    fields = ['"purpose": "..."', '"genres": ["...", "..."]']
    if joiners:
        fields.append(f'"joining": "{" or ".join(joiners)}"')
    lines += ['', 'Respond with ONLY this JSON object:', json_skeleton(fields + ['"addition": ""'], arc)]
    return '\n'.join(lines)


def clean_depression_plan(data, settings, arc):
    """Validate one parsed reply; returns (plan, notes) like clean_pain_relief_plan."""
    avoided = depression_avoided(settings, arc)
    phases, notes = clean_phases(data, arc, avoided)
    purpose = clean_purpose(data.get('purpose'), DEPRESSION_PURPOSES,
                            DEPRESSION_DEFAULT_PURPOSE[settings['session_mode']])
    genres = clean_genres(data.get('genres'), DEPRESSION_GENRES, notes, avoided[arc[0][0]])
    joiner = None
    options = DEPRESSION_JOINERS.get(settings['session_mode'])
    if options:
        joiner = depression_joiner(data.get('joining'), options)
        if joiner is None:
            notes.append(f'joining {data.get("joining")!r} is not {" or ".join(options)}, using {options[0]}')
            joiner = options[0]
    addition = clean_addition(data.get('addition'), avoided[arc[0][0]], notes)
    return {'purpose': purpose, 'genres': genres, 'joining': joiner,
            'addition': addition, 'phases': phases}, notes


def generate_depression_plan(session_request, settings, arc, guideline_lines, log, attempts=3):
    """Call Qwen until one reply passes clean_depression_plan, or give up."""
    user_message = depression_user_message(session_request, settings, arc, guideline_lines)
    return orchestrate(user_message, DEPRESSION_SYSTEM_PROMPT,
                       lambda data: clean_depression_plan(data, settings, arc),
                       log, 'depression', attempts)


def assemble_depression_prompts(plan, settings, arc, audio_phrases):
    """One Stable Audio 3 prompt per phase, in the same tag format as the others.
    The Instruments tag follows depression_phase_instruments and the tempo
    depression_phase_bpm; everything else is the same in every phase."""
    phase_instruments = depression_phase_instruments(arc, plan['joining'], settings)
    bpms = depression_phase_bpm(arc, settings)
    last = arc[-1][0]
    prompts = {}
    for name, _ in arc:
        bpm = bpms[name]
        tags = prompt_tags(plan['purpose'], plan['genres'], phase_instruments[name])
        tempo = f'{"Slow" if bpm <= 65 else "Moderate"}, steady {bpm} BPM.'
        additions = [DEPRESSION_ENDINGS[settings['ending']]] if name == last else []
        additions += [phrase for phrase, phases in audio_phrases if phases is None or name in phases]
        if plan['addition']:
            additions.append(plan['addition'])
        additions.append(DEPRESSION_GUARDRAIL)
        description = plan['phases'][name].rstrip(' .,;:!?')
        prompts[name] = f'{tags} {tempo} {description}, Addition Request: {"; ".join(additions)}'
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
    """Model calls for a guideline module's arc, where every phase keeps its share.

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

    Guideline module endings (both 0 = off, the ISO default). They run after
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
    seconds = config.LOOP_CHECK_S if seconds is None else seconds
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

    mode                           — 'iso' (match, transition, settle), or a guideline
                                     module: 'pain_relief', 'dementia' or 'depression'
    options                        — the module's settings from the page, see
                                     pain_relief_settings, dementia_settings and
                                     depression_settings; ignored by 'iso'
    log(str)                       — called with each progress line
    on_segment(i, total, name)     — called before each model call
    on_plan(names)                 — called once, with every segment name in order
    """
    if mode in GUIDELINE_MODULES:
        return GUIDELINE_MODULES[mode](session_request, seed=seed, log=log, on_segment=on_segment,
                                       options=options, on_plan=on_plan)
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


def render_arc_session(mode, session_request, arc, phase_prompts, settings, seed, log,
                       on_segment=None, on_plan=None):
    """Everything after the prompts, shared by the guideline modules.

    Same rendering, stitching, EQ and preview as the ISO pipeline; what differs is
    the segment plan (every phase keeps its share, see build_arc_segments) and the
    ending, applied to the finished file after loudness normalization: a fade to
    silence over the last FADE_SHARE of the session, a soft close's short tail, or
    a loop point folded into the start.
    """
    total_duration_s = settings['minutes'] * 60
    for name, _ in arc:
        log(f'[{name}] {phase_prompts[name]}')
    long = [name for name, _ in arc if len(phase_prompts[name].split()) > PROMPT_WORDS]
    if long:
        log(f'note: {", ".join(long)} run past ~{PROMPT_WORDS} words; the text encoder may cut their end off')

    out_dir = unique_dir(config.WORK / slugify(session_request))
    os.makedirs(out_dir, exist_ok=True)
    log(f'output folder {out_dir}')

    segments = build_arc_segments(arc, phase_prompts, total_duration_s, seed)
    if on_plan:
        on_plan([s['name'] for s in segments])
    planned = sum(s['duration_s'] for s in segments)
    log(f'{len(segments)} segments across {len(arc)} phases, '
        f'{planned:.0f}s planned vs {total_duration_s}s requested')

    # [1]: keep the path, let the stitched array go before EQ loads its own copy
    raw_path = build_session(segments, out_dir, log, on_segment)[1]

    post = dict(config.POST_EQ)
    if settings['ending'] == 'fade':
        post['fade_out_s'] = total_duration_s * config.FADE_SHARE
    elif settings['ending'] == 'close':
        post['fade_out_s'] = config.CLOSE_FADE_S
    else:
        post['loop_crossfade_s'] = config.LOOP_CROSSFADE_S
    eq_path = apply_post_eq(raw_path, os.path.join(out_dir, 'completed_EQed.wav'), log, **post)

    preview_path = make_preview(eq_path, out_dir, log)
    loop_check_path = make_loop_check(eq_path, out_dir, log) if settings['ending'] == 'loop' else None
    peaks = compute_peaks(eq_path)

    return {
        'mode': mode,
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


def generate_pain_relief_session(session_request, seed=None, log=None, on_segment=None,
                                 options=None, on_plan=None):
    """Module 1, Pain Relief: six phases that start calm and only get calmer, one
    palette and one tempo throughout, ending in a fade to silence or a loop."""
    log = log or print
    seed = config.RANDOM_SEED if seed is None else seed

    settings = pain_relief_settings(options, session_request, log)
    log(f'pain relief request "{session_request}" — {settings["minutes"] * 60}s '
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
    return render_arc_session('pain_relief', session_request, config.PAIN_ARC, phase_prompts,
                              settings, seed, log, on_segment, on_plan)


def generate_dementia_session(session_request, seed=None, log=None, on_segment=None,
                              options=None, on_plan=None):
    """Module 2, Dementia: the default arc (familiar, safe, comfortable, engaging,
    emotionally supportive) or Calming mode's six phases that reduce stimulation step
    by step; one steady tempo; a soft close, a fade or a loop."""
    log = log or print
    seed = config.RANDOM_SEED if seed is None else seed

    settings = dementia_settings(options, session_request, log)
    arc = config.DEMENTIA_ARCS[dementia_arc_key(settings['session_mode'])]
    log(f'dementia request "{session_request}" — {settings["minutes"] * 60}s '
        f'({settings["minutes"]} min), {settings["session_mode"]} mode, {settings["bpm"]} BPM, '
        f'{settings["ending"]} ending')

    guideline_lines, audio_phrases = dementia_lines(settings)

    log('orchestrating...')
    load_qwen()
    try:
        plan = generate_dementia_plan(session_request, settings, arc, guideline_lines, log)
    finally:
        unload_qwen()   # always give the VRAM back, even if orchestration failed
    log(f'instruments: {", ".join(plan["instruments"])} · genres: {", ".join(plan["genres"])}')
    phase_prompts = assemble_dementia_prompts(plan, settings, arc, audio_phrases)
    return render_arc_session('dementia', session_request, arc, phase_prompts,
                              settings, seed, log, on_segment, on_plan)


def generate_depression_session(session_request, seed=None, log=None, on_segment=None,
                                options=None, on_plan=None):
    """Module 3, Depression: Comfort mode's arc (acknowledge heaviness, safety, warmth,
    calm) or the Gentle activation, Relaxation or Sleep preparation arc; meets the
    mood first and never forces a lift."""
    log = log or print
    seed = config.RANDOM_SEED if seed is None else seed

    settings = depression_settings(options, session_request, log)
    arc = config.DEPRESSION_ARCS[settings['session_mode']]
    tempo = (f'{settings["bpm"]} rising to {settings["bpm_end"]} BPM' if settings['bpm_end']
             else f'{settings["bpm"]} BPM')
    log(f'depression request "{session_request}" — {settings["minutes"] * 60}s '
        f'({settings["minutes"]} min), {settings["session_mode"]} mode, {tempo}, {settings["ending"]} ending')

    guideline_lines, audio_phrases = depression_lines(settings)

    log('orchestrating...')
    load_qwen()
    try:
        plan = generate_depression_plan(session_request, settings, arc, guideline_lines, log)
    finally:
        unload_qwen()   # always give the VRAM back, even if orchestration failed
    if plan['joining']:
        log(f'joining instrument: {plan["joining"]} · genres: {", ".join(plan["genres"])}')
    if settings['bpm_end']:
        log('tempo by phase: ' + ' -> '.join(str(b) for b in depression_phase_bpm(arc, settings).values()) + ' BPM')
    phase_prompts = assemble_depression_prompts(plan, settings, arc, audio_phrases)
    return render_arc_session('depression', session_request, arc, phase_prompts,
                              settings, seed, log, on_segment, on_plan)


# mode name -> session function, for generate_session
GUIDELINE_MODULES = {
    'pain_relief': generate_pain_relief_session,
    'dementia':    generate_dementia_session,
    'depression':  generate_depression_session,
}

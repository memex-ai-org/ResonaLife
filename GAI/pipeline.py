"""Notebook sections 5-9 as a resident service.

Both models load once at process start and stay on the GPU. Nothing here
touches Colab, Drive, or `!pip`. The only entry point the server uses is
generate_session().
"""
import config  # noqa: F401  — must come first, sets HF_HOME before transformers

import gc
import json
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

    Qwen writes three prompts and is then dead weight for the rest of the run,
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


def ask_qwen(session_request, max_new_tokens=900):
    """One sampled reply from Qwen for the given session request, as raw text."""
    messages = [
        {'role': 'system', 'content': ORCHESTRATOR_SYSTEM_PROMPT},
        {'role': 'user', 'content': f'Session request: "{session_request}"'},
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
        reply = ask_qwen(session_request)
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


def parse_duration(session_request):
    m = re.search(r'(\d+)\s*(?:-|\s)*min', session_request, re.IGNORECASE)
    return int(m.group(1)) * 60 if m else config.DEFAULT_TOTAL_S


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
                  target_lufs=-16.0, peak_db=-1.0):
    """Gentle cleanup EQ + loudness normalization for a finished session render.

    hpf_hz      — high-pass cutoff; removes only inaudible/subsonic rumble
    body_*      — 90-130 Hz body band; body_db=0.0 means no artificial boost by default
    mud_*       — 250-400 Hz low-mid band; mud_db is a very gentle cleanup cut
    air_*       — high-shelf-ish trim above air_hz, keeps long sessions smooth
    target_lufs — consistent, comfortable functional-music loudness target
    peak_db     — safety ceiling so normalization never clips
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

def generate_session(session_request, seed=None, log=None, on_segment=None):
    """Full pipeline for one request. Returns a dict of results.

    log(str)                       — called with each progress line
    on_segment(i, total, name)     — called before each model call
    """
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
    planned = sum(s['duration_s'] for s in segments)
    log(f'{len(segments)} segments, {planned:.0f}s planned vs {total_duration_s}s requested')

    track, raw_path = build_session(segments, out_dir, log, on_segment)
    eq_path = apply_post_eq(raw_path, os.path.join(out_dir, 'completed_EQed.wav'),
                            log, **config.POST_EQ)

    preview_path = make_preview(eq_path, out_dir, log)
    peaks = compute_peaks(eq_path)

    return {
        'out_dir': out_dir,
        'raw_path': raw_path,
        'eq_path': eq_path,
        'preview_path': preview_path,
        'peaks': peaks,
        'duration_s': len(track) / SR,
        'prompts': phase_prompts,
        'segments': [{'name': s['name'], 'duration_s': s['duration_s']} for s in segments],
    }
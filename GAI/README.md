# ResonaLife Functional Music GAI on Modal

The GAI service runs on [Modal](https://modal.com). One password-protected URL
serves both the web page and the API. A GPU container renders sessions on
demand and shuts down when nobody is generating, so GPU time is only billed
while sessions render, plus a 5-minute warm window.

## How it runs

```
browser ──► web (CPU) ──── POST /generate ────► Generator (GPU, L40S)
            page + API      job saved to a Dict    Qwen writes the prompts,
            password check  ◄──── page polls ────  Stable Audio 3 renders,
            ◄── /audio/… ── resonalife-outputs ◄── EQ, preview, waveform
```

| Modal resource | Kind | What it holds |
|---|---|---|
| `resonalife-gai` | App | the two functions below |
| `web` | CPU function | the page (`static/`), the API, the login |
| `Generator` | GPU class, L40S | keeps Stable Audio 3 loaded; renders one session at a time |
| `resonalife-models` | Volume | Hugging Face cache: Stable Audio 3 and Qwen weights (~25 GB) |
| `resonalife-outputs` | Volume | one folder of WAV/MP3 files per session |
| `resonalife-jobs` | Dict | job status, log and prompts; this is the "Your sessions" list |
| `resonalife-calls` | Dict | job to GPU call id, so a crashed GPU call shows up as an error |
| `resonalife-secrets` | Secret | `HF_TOKEN`, `RESONALIFE_PASSWORD` |
| `resonalife-train` | App | LoRA training (`train_lora_modal.py`), separate from the site |
| `resonalife-loras` | Volume | training clips, LoRA checkpoints, demo and sample audio |

## Folder layout

Run every `modal` command from inside this folder.

```
GAI/
├── modal_app.py          the Modal app: what you run and deploy
├── train_lora_modal.py   LoRA training, a separate Modal app (see Training a LoRA)
├── pipeline.py           orchestration, generation, post-processing
├── config.py             model ids, generation and EQ settings, pain relief arc
├── static/index.html     the page
├── data/                 training audio and captions, used only by training
├── stable-audio-3-main/  Stability's stable-audio-3 repo, copied into the GPU image
├── server.py             optional: the same service on a local GPU
└── requirements.txt      optional: local installs only (Modal builds its own images)
```

## First-time setup

Do this once per Modal workspace.

1. Install Modal and log in:

   ```bash
   pip install modal
   python -m modal setup
   ```

2. Get Hugging Face access. Both Stability repos are gated. While logged in to
   Hugging Face, accept the terms at
   [stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
   and [SAME-L](https://huggingface.co/stabilityai/SAME-L), then create a read
   token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

3. Create the secret. `RESONALIFE_PASSWORD` is the page's password; the
   username is always `resonalife`.

   ```bash
   python -m modal secret create resonalife-secrets HF_TOKEN=hf_xxx RESONALIFE_PASSWORD=pick-a-password
   ```

4. Add Stability's code: unzip
   [github.com/Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3)
   into this folder as `stable-audio-3-main/`.

5. Build the images and cache the weights. The first run takes a while.

   ```bash
   python -m modal run modal_app.py
   ```

   This builds the GPU image (torch 2.7.1 cu126, flash-attn 2.8.3, Stable Audio 3)
   and downloads both models into `resonalife-models`. It finishes by printing
   `Stable Audio 3 and Qwen are cached in resonalife-models`.

6. Deploy:

   ```bash
   python -m modal deploy modal_app.py
   ```

   The output ends with the URL, `https://<workspace>--resonalife-gai-web.modal.run`.
   The site stays up until you stop it.

## Using it

Open the URL. The browser asks you to log in: the username is `resonalife` and
the password is the one from step 3.

| Mode | What it makes |
|---|---|
| Choose settings | Guided ISO session: pick a goal, a length and sounds |
| Write your own | ISO session from free text. The length is the first "N min" in the text; default 15, max 120 |
| Pain relief | The hospice guideline's six-phase comfort arc (see [Pain relief mode](#pain-relief-mode)) |

- **Cold start:** after a quiet spell, the first session shows "Getting the
  studio ready" for a minute or two while a GPU starts and loads Stable Audio 3.
- **Queue:** sessions render one at a time. The header shows "Busy, N waiting"
  while others are queued.
- **Warm window:** after a session finishes, the GPU stays warm for 5 minutes,
  so the next one starts right away.
- **Downloads:** each finished session offers an MP3 preview, the 24-bit
  studio WAV (EQ'd and loudness-normalized) and the unprocessed render.
- **Closing the page is safe.** Rendering carries on without it, and the
  session reopens from "Your sessions".

## Updating

After changing `pipeline.py`, `config.py`, `static/` or `modal_app.py`:

```bash
python -m modal deploy modal_app.py
```

Deploys are quick and keep everything already stored: the heavy GPU image
layers aren't rebuilt for code changes, the weights stay in
`resonalife-models`, and past sessions and the "Your sessions" list survive.

If you change `SA3_MODEL` or `QWEN_MODEL_ID`, run `python -m modal run modal_app.py`
once more first. That downloads the new weights up front instead of during
someone's first session.

## Everyday commands

| Task | Command |
|---|---|
| Watch the logs live | `python -m modal app logs resonalife-gai -f` |
| List saved sessions | `python -m modal volume ls resonalife-outputs` |
| Download one session's files | `python -m modal volume get resonalife-outputs <session-folder> .` |
| Change the password | `python -m modal secret create resonalife-secrets HF_TOKEN=hf_xxx RESONALIFE_PASSWORD=new-password --force`, then deploy again |
| Empty the "Your sessions" list | `python -m modal dict clear resonalife-jobs` (the audio files stay in the volume) |
| Take the site offline | `python -m modal app stop resonalife-gai`; deploy again to bring it back |

`--force` replaces the whole secret, so always pass both keys. Session folders
are named after the request, e.g. `a-30-min-pain-relief-session-with-soft-p`.

## Capacity and cost

These settings live on the `Generator` class in `modal_app.py`:

| Setting | Now | What it does |
|---|---|---|
| `gpu` | `'L40S'` | 48 GB, enough for Stable Audio 3 and Qwen together |
| `max_containers` | `1` | One session at a time; the rest queue. Set `2` to render two at once on two GPUs |
| `scaledown_window` | 5 min | How long the GPU idles before shutting down. Lower saves money; higher means fewer cold starts |
| `timeout` | 2 h | The longest a single session may run |
| `retries` | `1` | Retries only if the container itself crashes; pipeline errors show on the page instead |

Modal bills per second of container time. With nobody using the page,
nothing is running. The `web` container is CPU-only and also shuts down
between visits.

## Session modes

Both modes share the same rendering, stitching, EQ and preview code. Each
model call continues from the last 30 s of the one before it (inpainting), so
the whole session plays as one continuous piece.

### ISO (Choose settings, Write your own)

Three phases, split by `config.SPLIT`:

| Phase | Share | Model calls |
|---|---|---|
| match | 25% | One call, capped at 5.5 min |
| transition | 35% | One call, capped at 5 min |
| settle | 40% | Chunked across as many calls as needed to fill the rest of the session |

Qwen writes all three prompts from `ORCHESTRATOR_SYSTEM_PROMPT`.

### Pain relief mode

This is Module 1 of the Functional Music Guideline for Hospice & Chronic
Illness. It never opens with energy to "meet" the listener: it starts calm and
only ever gets calmer. The six phases take the guideline's shares
(`config.PAIN_ARC`):

| Phase | Share | In a 30-min session |
|---|---|---|
| Immediate calm | 10% | 0–3 min |
| Release tension | 15% | 3–7.5 min |
| Comfort and safety | 20% | 7.5–13.5 min |
| Deepen relaxation | 20% | 13.5–19.5 min |
| Weightless plateau | 25% | 19.5–27 min |
| Fade or loop | 10% | 27–30 min |

**No phase is clamped.** Each phase is split into as few equal model calls as
fit the per-call limit, so the shares hold at any length. A 30-min session is
9 calls; a 60-min session is 15.

**Prompts.** `PAIN_RELIEF_SYSTEM_PROMPT` in `pipeline.py` carries the
guideline's master prompt. Qwen only writes what the music does in each
phase. The code sets the tags, instruments and tempo, which are identical in
all six prompts. The code also:

- Removes any BPM Qwen writes, so the tempo is stated once, the same in every phase.
- Drops any phrase that names an avoided sound. Post-trained Stable Audio 3
  ignores `negative_prompt`, and "no drums" in a prompt tends to bring drums
  in, so the guideline's Avoid list goes in as positive wording (`PAIN_GUARDRAIL`).
- Adds the guideline's add-on and troubleshooting lines for the chosen
  settings (`PAIN_ADDONS`, `PAIN_FIXES`).

**Settings.** The page's tab covers the guideline's tailoring table, and the
API takes the same settings as `options`. Anything outside the safe ranges is
pulled back in, with a warning in the log.

| Setting | `options` key | Values | Default |
|---|---|---|---|
| Length | `minutes` | 10–60 | `30` |
| Tempo | `bpm` | 50–65 | `58` |
| Ending | `ending` | `fade`, `loop` | `fade` |
| Lead | `lead` | `piano`, `strings`, `pads` | `piano` |
| Warmer piano | `warm_piano` | `true`, `false` | `false` |
| Melody | `melody` | `sparse`, `none` | `sparse` |
| Nature texture | `nature` | `off`, `wind`, `waves` | `off` |
| Fixes (when regenerating) | `fixes` | `percussion`, `catchy`, `bright`, `bass`, `climax`, `voices`, `dark`, `faster` | none |

A melody of "none", or the "catchy melody" fix, removes the piano: the
guideline's line for both is "pads and sustained tones only".

**Endings.** These are applied to `completed_EQed.wav` after loudness
normalization, so they don't change the level of the rest of the session.
`completed.wav` stays the unprocessed render.

- **Fade:** the whole last phase fades to exact silence.
- **Loop:** the last 8 s are crossfaded into the first 8 s, so the file's last
  sample runs straight into its first. `loop_check.mp3` plays the last 10 s
  followed by the first 10 s, so you can hear the join. For looping on another
  device use the WAV, since MP3 encoding adds a short gap.

The finished session shows the guideline's review checklist and a **Fix and
regenerate** button, which reopens the settings with the fix list open.

## API

Every route sits behind the same login, so scripts pass it with `-u`:

```bash
URL=https://<workspace>--resonalife-gai-web.modal.run

# ISO session
curl -u resonalife:$PASSWORD -H 'Content-Type: application/json' \
  -d '{"request": "A 20 min session to unwind, with soft rain."}' $URL/generate

# pain relief session (options that are left out take the defaults above)
curl -u resonalife:$PASSWORD -H 'Content-Type: application/json' \
  -d '{"request": "A 30 min pain relief session with soft piano and pads, fading to silence.",
       "mode": "pain_relief",
       "options": {"minutes": 30, "bpm": 58, "ending": "fade", "lead": "piano",
                   "warm_piano": false, "melody": "sparse", "nature": "off", "fixes": []}}' \
  $URL/generate
# -> {"job_id": "…", "position": 1}

curl -u resonalife:$PASSWORD $URL/jobs/<job_id>
```

| Route | Returns |
|---|---|
| `POST /generate` | `job_id` and queue position. Body: `request`, optional `seed`, `mode` (`iso` or `pain_relief`), `options` |
| `GET /jobs/{id}` | `status` (queued, running, done or error), `progress`, `plan`, `log`, and once done a `result` |
| `GET /jobs` | The latest 50 jobs, summary only |
| `GET /audio/{folder}/{file}` | A session file |
| `GET /health` | Queue and busy state |

When a job is done, its `result` holds:

- `audio_url` (MP3 preview), `download_url` (studio WAV) and `raw_url` (unprocessed render)
- `loop_check_url` (loop endings only)
- `prompts`, `segments` and, for pain relief, the `settings` actually used

## Training a LoRA

A LoRA teaches Stable Audio 3 the sound of our own music without retraining
the model. `train_lora_modal.py` runs Stability's LoRA trainer
(`stable-audio-3-main/scripts/train_lora.py`) on a Modal GPU. Everything it
makes goes to the `resonalife-loras` volume; the live site doesn't change until
a LoRA is wired into the pipeline.

**Training data.** Put audio in `GAI/data/`, each file with a caption under the
same name:

```
GAI/data/
├── pain_relief_1.wav
└── pain_relief_1.txt     one line describing the audio
```

Write captions the way the pipeline writes prompts, since the LoRA learns to
respond to that wording. For example:

```
Purpose: Relaxation, TrackType: Music, VocalType: Instrumental, Genre: Ambient, Genre: Neoclassical, Instruments: soft piano, gentle pads, warm strings. Slow, steady 58 BPM. Sparse felt piano over warm sustained pads; calm, spacious and predictable
```

Stability recommends at least 20–50 clips. `train` cuts every file into 60 s
clips, 30 s apart, so a 30-minute recording gives about 59 clips. With a
single recording, expect the LoRA to copy that one piece closely. `GAI/data/`
is uploaded at every training run and isn't used by the site; add it to
`.gitignore` to keep audio out of git.

**Hugging Face access.** Stability's trainer only trains base checkpoints.
Once, with the same Hugging Face account as your token, accept the terms at
[stable-audio-3-medium-base](https://huggingface.co/stabilityai/stable-audio-3-medium-base).

**Train:**

```bash
python -m modal run --detach train_lora_modal.py::train
```

`--detach` keeps the run going if your laptop sleeps or disconnects. Follow it
with `python -m modal app logs resonalife-train -f`. The first run builds the
training image and downloads medium-base into `resonalife-models`. The run's
name is printed at the start: `pain-relief`, then `pain-relief_2` and so on.

| Option | Default | What it does |
|---|---|---|
| `--name` | `pain-relief` | Run name. A number is added if it's taken |
| `--steps` | `1000` | Training steps. Stability's suggested starting point |
| `--clip-seconds` | `60` | Length of each training clip |
| `--hop-seconds` | `30` | Spacing between clip starts; less than the length means overlap |
| `--rank` | `16` | LoRA rank: higher learns more detail, and overfits faster |
| `--lr` | `1e-4` | Learning rate |
| `--checkpoint-every` | `250` | How often a checkpoint and the trainer's demo audio are saved |

It trains a DoRA adapter (Stability's default). It also leaves the
`seconds_total` conditioner alone, which Stability advises for small datasets.

**What a run makes**, in `resonalife-loras/<run>/`:

| Path | What it is |
|---|---|
| `lora.safetensors` | The finished LoRA, from the last step |
| `step_00250.safetensors`, … | Earlier checkpoints, for comparison |
| `demos/` | The trainer's own samples, from the training captions at CFG 2, 4 and 7. Step 1 is the untrained starting point |
| `listen/<time>/` | The comparison samples below |
| `clips/`, `run.json` | The training clips and the settings used |
| `lightning_logs/` | `metrics.csv` with the training loss |

**Listen.** When training ends, `train` renders one pain relief prompt, written
the way the pipeline writes them, with the same seed four ways:

| File | Model | LoRA |
|---|---|---|
| `medium-base_no-lora.wav` | medium-base, 50 steps, CFG 7 | off |
| `medium-base_lora.wav` | medium-base, 50 steps, CFG 7 | on |
| `medium_no-lora.wav` | medium, 8 steps, which the server uses | off |
| `medium_lora.wav` | medium, 8 steps | on |

Each pair differs only by the LoRA. The medium pair shows whether a LoRA
trained on the base checkpoint carries over to the fast model the server runs,
which decides how it goes into the pipeline. For more samples with another
prompt, seed or checkpoint:

```bash
python -m modal run train_lora_modal.py::listen --name pain-relief --seed 7 --checkpoint step_00500.safetensors
```

`--prompt "…"` swaps in your own prompt.

**Download:**

```bash
python -m modal volume get resonalife-loras pain-relief/listen .
python -m modal volume get resonalife-loras pain-relief/demos .
python -m modal volume get resonalife-loras pain-relief/lora.safetensors .
```

## Tuning

These live in `config.py`; deploy after changing them.

- **`POST_EQ`:** `body_db` defaults to 0.0. The 90–130 Hz Body Melody boost is
  a per-product decision, not a pipeline default.
- **Generation:** `STEPS` (8, the tuned default for medium), `CTX_S` (30 s of
  context between calls) and `MAX_MODEL_DURATION_S` (360 s per call).
- **Pain relief:** `PAIN_*` holds the arc shares, the safe ranges, the
  defaults and the loop crossfade length.

## Troubleshooting

- **The login box keeps coming back.** The password is wrong. The username is
  `resonalife`.
- **The page shows a server error instead of the login box.** The secret is
  missing `RESONALIFE_PASSWORD` or `HF_TOKEN`. Recreate it with both keys (see
  Everyday commands) and deploy again.
- **Stuck on "Getting the studio ready" for more than ~5 minutes.** Run
  `python -m modal app logs resonalife-gai`. If step 5 of setup was skipped,
  the first session has to download the weights itself.
- **"GPU worker failed: …".** The GPU container died, for example because the
  model failed to load or the session hit the 2-hour timeout. The logs have
  the traceback.
- **An audio link 404s right after a session finishes.** Refresh the page.
  The web container picks up new files from the volume on demand.

## Running locally (optional)

`server.py` serves the same page and API on a local NVIDIA GPU with 16 GB or
more, with no login:

1. Install the packages in the order the comments in `requirements.txt` give.
2. Set `HF_TOKEN`. Optionally set `RESONALIFE_ROOT` to put the ~25 GB of
   weights on another drive.
3. Run `python server.py` and open `http://localhost:8000`.
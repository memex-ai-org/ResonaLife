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
├── config.py             model ids, generation and EQ settings, the guideline modules' arcs
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
| Pain relief | The hospice guideline's Module 1: a six-phase comfort arc (see [Pain relief mode](#pain-relief-mode)) |
| Dementia | Module 2: familiar, simple and safe music, with a Calming mode (see [Dementia mode](#dementia-mode)) |
| Depression | Module 3: meets the mood first, then comforts or gently lifts (see [Depression mode](#depression-mode)) |

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

Every mode shares the same rendering, stitching, EQ and preview code. Each
model call continues from the last 30 s of the one before it (inpainting), so
the whole session plays as one continuous piece.

### ISO (Choose settings, Write your own)

Three phases, split by `config.SPLIT`:

| Phase | Share | Model calls |
|---|---|---|
| match | 25% | One call, capped at 5.5 min |
| transition | 35% | One call, capped at 5 min |
| settle | 40% | Chunked across as many calls as needed to fill the rest of the session |

Qwen writes all three prompts from `ORCHESTRATOR_SYSTEM_PROMPT`. A reply that
can't be used is sampled again, up to three times, and Qwen is told what was
wrong. If none can, the session plays `ISO_FALLBACK_PROMPTS` (one calm palette,
72 BPM down to no tempo) rather than failing. The log says so, because those
prompts don't carry what the request asked for.

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
- Never fails a session over a bad reply. A reply with a missing, too-short or
  tagged phase is sampled again, up to three times, and Qwen is told what was
  wrong. If none passes, the code keeps the reply with the most usable phases
  and writes the others itself from `PAIN_PHASE_FALLBACK`, its own description
  of each phase. The log names the phases it wrote.

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

### Guideline modules: what they share

Dementia and depression are built the same way as pain relief:

- **The arc.** Each session mode has its own phases and shares, from the
  guideline's session arc tables (`config.DEMENTIA_ARCS`,
  `config.DEPRESSION_ARCS`). No phase is clamped, and a 30-min session lands on
  the guideline's phase times.
- **The prompts.** The module's master prompt is its system prompt
  (`DEMENTIA_SYSTEM_PROMPT`, `DEPRESSION_SYSTEM_PROMPT`). The settings, the
  guideline lines they switch on (Session mode, Time of day, add-ons, fixes) and
  the listener profile go in the user message, as the guideline says to add
  them to the bottom of the prompt. Qwen writes what the music does in each
  phase; the code sets the tags, the tempo and which instruments play in each
  phase, and removes BPMs and phrases that name an avoided sound or mood.
- **The Avoid list** goes in as positive wording (`DEMENTIA_GUARDRAIL`,
  `DEPRESSION_GUARDRAIL`), like pain relief's.
- **Bad replies** never fail a session, as in pain relief: a retry is told what
  was wrong, and a phase still unusable after three tries comes from
  `DEMENTIA_PHASE_FALLBACK` or `DEPRESSION_PHASE_FALLBACK`.
- **Listener profile.** The page has the guideline's profile fields in a fold-out
  section. Only filled fields are sent, and they shape the style, never the
  safety rules.
- **Endings** work as in pain relief: a fade covers the last 10% of the session
  (`FADE_SHARE`), and a loop folds the last 8 s into the start and adds
  `loop_check.mp3`.
- **Review.** The finished session shows the module's checklist, then its
  quality check and what to watch for while it plays.

These two modules are drafts in the guideline: their prompts haven't been
tested yet. Listen to the first sessions closely.

### Dementia mode

Module 2: familiar, simple and emotionally safe music that supports comfort,
calm, orientation and gentle engagement. The default arc moves from familiar to
comfortable to gently engaging, then settles. Calming mode (anxiety, agitation
or restlessness) reduces stimulation step by step.

| Default arc | Share | 30 min | Calming mode | Share | 30 min |
|---|---|---|---|---|---|
| Familiar | 15% | 0–4.5 | Steady and familiar | 15% | 0–4.5 |
| Safe | 15% | 4.5–9 | Simplify rhythm | 15% | 4.5–9 |
| Comfortable | 25% | 9–16.5 | Fewer instruments | 20% | 9–15 |
| Engaging | 30% | 16.5–25.5 | Settle | 20% | 15–21 |
| Emotionally supportive | 15% | 25.5–30 | Spacious calm | 20% | 21–27 |
| | | | Fade or loop | 10% | 27–30 |

Reminiscence and Social mode use the default arc; their lines change the style.

**Instruments.** Qwen picks two or three from the guideline's list (soft piano,
acoustic guitar, warm strings, gentle woodwinds, warm pads) to suit the
listener's era and culture. A familiar instrument from the page always comes
first. The code then decides who plays: light acoustic percussion joins only in
Engaging, taking the place of a third instrument, one instrument drops out in
Emotionally supportive, and Calming mode reduces to the first two from Fewer
instruments on, with piano or guitar carrying the pulse. Percussion counts as
an instrument, so with no more than two at a time it stays out. The wordless
voice is a texture and plays throughout.

**Tempo** stays constant. The guideline lets the calming arc "slow or
stabilize"; a steady tempo is the safer reading, since "tempo drifts or
changes" is one of its listed problems.

| Setting | `options` key | Values | Default |
|---|---|---|---|
| Length | `minutes` | 10–60 | `30` |
| Session mode | `session_mode` | `default`, `calming`, `reminiscence`, `social` | `default` |
| Time of day | `time_of_day` | `none`, `morning`, `afternoon`, `evening`, `presleep` | `none` |
| Tempo | `bpm` | 50–75 by default; 45–65 for calming, evening and pre-sleep; 60–80 for social and morning | `65` |
| Ending | `ending` | `close` (soft, resolved phrase), `fade`, `loop` | `close`; `loop` in the evening |
| Instruments at once | `density` | `usual` (two or three), `fewer` (no more than two) | `usual` |
| Melody | `melody` | `short`, `simpler` | `short` |
| Percussion | `percussion` | `light`, `off` (always off for calming, evening, pre-sleep and two instruments at a time) | `light` |
| Voice texture | `voice` | `off`, `wordless` | `off` |
| Familiar instrument | `familiar_instrument` | one instrument name, e.g. `erhu` | none |
| Listener profile | `profile` | `age`, `culture`, `preferred_music`, `favorites`, `era`, `memories`, `state`, `activity`, `outcome`, `sensitivity`, `environment` | none |
| Fixes | `fixes` | `unfamiliar`, `busy`, `wanders`, `monotonous`, `percussion`, `loud`, `tempo`, `sad`, `lyrics`, `song`, `unsettled` | none |

Calming mode ends with a fade or a loop, and pre-sleep always loops ("a
seamless loop instead of an ending"). Evening switches to the loop ("loop for
evening and pre-sleep") and can be changed back. The soft close ends on a
resolved phrase with an 8 s tail (`CLOSE_FADE_S`) so it never cuts off. With the wordless voice
on, the prompts leave out `VocalType: Instrumental`.

### Depression mode

Module 3: music that meets a low mood where it is, then supports comfort,
relaxation or gentle engagement. It never opens bright to lift a low mood and
never forces a happy ending.

| Session mode | Phases (share) | Ending |
|---|---|---|
| Comfort (default) | Acknowledge heaviness 20% → Safety 25% → Warmth 30% → Calm 25% | fade or loop |
| Gentle activation | Stillness 20% → Comfort 25% → Engagement 30% → Subtle positive energy 25% | fade |
| Relaxation | Tension 15% → Calm 25% → Spaciousness 30% → Deep relaxation 30% | fade |
| Sleep preparation | Emotional fatigue 10% → Safety 20% → Calm 25% → Low stimulation 25% → Sleep readiness 20% | loop |

**Instruments** come from the arc text. Soft piano and warm pads play
throughout ("sparse piano or pads"); one more instrument joins at Warmth in
Comfort mode ("warm strings or guitar join"), at Comfort in Gentle activation
(guitar, marimba or woodwinds), and in Relaxation it leaves after Calm (guitar,
strings or woodwinds). Qwen picks which one from the listener profile. Sleep
preparation keeps pads, piano and strings throughout. Light percussion plays
only in Gentle activation's last two phases; it's on by default there, because
that mode's own line adds it. "Natural acoustic textures" from the guideline's
list is left out: the audio model tends to answer it with field recordings.

**Tempo.** Constant, except in Gentle activation, where it rises in even steps
from `bpm` to `bpm_end`, one step per phase (58 → 62 → 66 → 70 by default).
With Daytime, whose line asks for 65–80 BPM, the session starts at 65, the one
tempo that also fits the activation line's "begin at 50–65".

| Setting | `options` key | Values | Default |
|---|---|---|---|
| Length | `minutes` | 10–60 | `30` |
| Session mode | `session_mode` | `comfort`, `activation`, `relaxation`, `sleep` | `comfort` |
| Time of day | `time_of_day` | `none`; `morning` or `daytime` (activation), `evening` (relaxation), `presleep` (sleep) | `none` |
| Tempo | `bpm` | 50–65 (where Gentle activation starts; 65 with Daytime) | `58` |
| Tempo reached | `bpm_end` | 65–80, Gentle activation only | `70` |
| Energy level | `energy` | `low`, `moderate`, `higher` | `low` |
| Harmony color | `harmony` | `warm`, `resonance` (Comfort only) | `warm` |
| Brightness | `brightness` | `warm`, `brighter` (Gentle activation only) | `warm` |
| Percussion | `percussion` | `off`, `light` (Gentle activation only) | `light` in Gentle activation, otherwise `off` |
| Sparser | `sparser` | `true`, `false` (not in Gentle activation) | `false` |
| Listener profile | `profile` | `age`, `culture`, `genres`, `artists`, `background`, `state`, `activity`, `outcome`, `environment`, `sensitivity` | none |
| Fixes | `fixes` | `heavy`, `cheerful`, `rises`, `never_lifts`, `minor`, `bright`, `percussion`, `melody`, `lyrics` | none |

A setting that doesn't belong to the session mode is left out with a warning
in the log, and a fix outranks the setting it corrects: "Minor key dominates"
turns off `resonance`, "Sound is harsh or too bright" turns off `brighter` and
the morning line's brighter timbres. "Energy rises too fast" is offered in
Comfort and Gentle activation only, since energy only falls in the other modes.

Three choices go beyond the guideline's tables. Gentle activation only fades: a
loop would drop from its steady energy straight back to stillness. Sparser is
left out of Gentle activation, where the pulse is meant to grow clearer. The end
tempo defaults to 70 BPM, since the guideline gives a range but no default.

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

# dementia session
curl -u resonalife:$PASSWORD -H 'Content-Type: application/json' \
  -d '{"request": "A 30 min dementia care session in calming mode, for the evening, fading to silence.",
       "mode": "dementia",
       "options": {"minutes": 30, "session_mode": "calming", "time_of_day": "evening", "bpm": 60,
                   "ending": "fade", "familiar_instrument": "erhu",
                   "profile": {"age": "84", "era": "1960s", "state": ["anxious"]}}}' \
  $URL/generate

# depression session
curl -u resonalife:$PASSWORD -H 'Content-Type: application/json' \
  -d '{"request": "A 30 min depression support session in gentle activation mode, fading to silence.",
       "mode": "depression",
       "options": {"minutes": 30, "session_mode": "activation", "bpm": 58, "bpm_end": 70,
                   "energy": "low", "percussion": "light"}}' \
  $URL/generate

curl -u resonalife:$PASSWORD $URL/jobs/<job_id>
```

| Route | Returns |
|---|---|
| `POST /generate` | `job_id` and queue position. Body: `request`, optional `seed`, `mode` (`iso`, `pain_relief`, `dementia` or `depression`), `options` |
| `GET /jobs/{id}` | `status` (queued, running, done or error), `progress`, `plan`, `log`, and once done a `result` |
| `GET /jobs` | The latest 50 jobs, summary only |
| `GET /audio/{folder}/{file}` | A session file |
| `GET /health` | Queue and busy state |

When a job is done, its `result` holds:

- `audio_url` (MP3 preview), `download_url` (studio WAV) and `raw_url` (unprocessed render)
- `loop_check_url` (loop endings only)
- `prompts`, `segments` and, for the guideline modules, the `settings` actually used

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
- **Guideline modules:** `PAIN_*`, `DEMENTIA_*` and `DEPRESSION_*` hold the arc
  shares, the safe ranges and the defaults. `FADE_SHARE`, `CLOSE_FADE_S`,
  `LOOP_CROSSFADE_S` and `LOOP_CHECK_S` set the endings for all three.

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
- **The log notes a prompt "runs past ~190 words".** The text encoder reads
  about that much, so the end of a long prompt, where the Avoid wording sits,
  may be cut. It happens when many fixes are on at once; regenerate with fewer.
- **The log says "no … orchestrator reply passed in 3 attempts".** Each of
  Qwen's replies had a phase the code couldn't use, so the phases named on the
  next log line play the code's own description. The session is safe to play
  but less tailored to the request. `python -m modal app logs resonalife-gai`
  prints every rejected reply in full; if it keeps happening in one mode, that
  mode's prompt needs work.

## Running locally (optional)

`server.py` serves the same page and API on a local NVIDIA GPU with 16 GB or
more, with no login:

1. Install the packages in the order the comments in `requirements.txt` give.
2. Set `HF_TOKEN`. Optionally set `RESONALIFE_ROOT` to put the ~25 GB of
   weights on another drive.
3. Run `python server.py` and open `http://localhost:8000`.
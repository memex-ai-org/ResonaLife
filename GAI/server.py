"""FastAPI wrapper around pipeline.generate_session.

One GPU means one generation at a time, so requests go into a queue and a
single background thread drains it. POST /generate returns immediately with a
job id; the browser polls GET /jobs/{id} until it's done.

POST /generate {"request": "...", "seed": null}                       ISO session
POST /generate {"request": "...", "mode": "pain_relief", "options": {...}}
    options: minutes 10-60, bpm 50-65, ending fade|loop, lead piano|strings|pads,
    melody sparse|none, warm_piano bool, nature off|wind|waves, fixes [...]
POST /generate {"request": "...", "mode": "dementia", "options": {...}}
    options: minutes 10-60, session_mode default|calming|reminiscence|social,
    time_of_day none|morning|afternoon|evening|presleep, bpm 45-80, ending close|fade|loop,
    density usual|fewer, melody short|simpler, percussion light|off, voice off|wordless,
    familiar_instrument "...", profile {...}, fixes [...]
POST /generate {"request": "...", "mode": "depression", "options": {...}}
    options: minutes 10-60, session_mode comfort|activation|relaxation|sleep,
    time_of_day none|morning|daytime|evening|presleep, bpm 50-65, bpm_end 65-80,
    ending fade|loop, energy low|moderate|higher, harmony warm|resonance,
    brightness warm|brighter, percussion off|light, sparser bool, profile {...}, fixes [...]
(see pipeline.pain_relief_settings, dementia_settings and depression_settings;
anything missing takes the guideline default)

POST /inpaint {"job_id": "...", "parts": [[start_s, end_s], ...], "fixes": [...], "note": "..."}
    regenerates parts of a finished session (see pipeline.inpaint_session) as a new
    job; when it's done the session's job gets the result as its newest version.
    Optional "version": which of the session's versions to start from (default the
    latest). The session's job lists every version under "versions".
"""
import queue
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import pipeline

app = FastAPI(title='ResonaLife Functional Music GAI')

# Only needed if the page is served from somewhere other than this app —
# a Vite dev server, file://, another port. Harmless otherwise.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

JOBS = {}
JOBS_LOCK = threading.Lock()   # JOBS is read by request threads, written by both
JOB_QUEUE = queue.Queue()
READY = threading.Event()      # set once the models are on the GPU
MODEL_ERROR = None             # traceback string if loading failed


class GenerateRequest(BaseModel):
    request: str
    seed: int | None = None
    mode: Literal['iso', 'pain_relief', 'dementia', 'depression'] = 'iso'
    options: dict | None = None    # a guideline module's settings; validated by the pipeline


class InpaintRequest(BaseModel):
    job_id: str                    # the session to fix
    parts: list                    # [[start_s, end_s], ...]; checked by pipeline.clean_parts
    version: int | None = None     # the version to start from; the newest by default
    fixes: list[str] = []          # the problems ticked on the page
    note: str = ''                 # the listener's own words for the part
    seed: int | None = None


def result_for_page(result):
    """The pipeline's result with file paths turned into URLs the page can fetch."""
    folder = Path(result['eq_path']).parent.name
    loop_check = result.get('loop_check_path')
    page = {
        'mode': result['mode'],
        'ending': result.get('ending'),
        'settings': result.get('settings'),
        'folder': folder,
        'version': result.get('version', 1),
        'duration_s': result['duration_s'],
        'prompts': result['prompts'],
        'segments': result['segments'],
        'peaks': result['peaks'],
        # small file for the browser to stream
        'audio_url': f'/audio/{folder}/{Path(result["preview_path"]).name}',
        # full-quality 24-bit deliverable
        'download_url': f'/audio/{folder}/{Path(result["eq_path"]).name}',
        'raw_url': f'/audio/{folder}/{Path(result["raw_path"]).name}',
        # loop endings: the last and first seconds back to back
        'loop_check_url': f'/audio/{folder}/{Path(loop_check).name}' if loop_check else None,
    }
    # a fixed version: which version it was made from, and what was regenerated how;
    # and for every version, what a fix of it may regenerate (config.INPAINT_*)
    for key in ('base_version', 'parts', 'fixes', 'note', 'left_out', 'seed', 'limits'):
        if key in result:
            page[key] = result[key]
    return page


def versions_of(job):
    """Every version of a session's job, oldest first (jobs from before versions have one)."""
    return job.get('versions') or ([job['result']] if job.get('result') else [])


def worker():
    """Drains JOB_QUEUE forever. Loads models on first start, then never unloads."""
    global MODEL_ERROR
    try:
        pipeline.load_models()
        READY.set()
    except Exception:
        MODEL_ERROR = traceback.format_exc()
        print('MODEL LOAD FAILED\n' + MODEL_ERROR)
        # Don't leave submitted jobs sitting in 'queued' with no explanation.
        with JOBS_LOCK:
            for job in JOBS.values():
                if job['status'] == 'queued':
                    job['status'] = 'error'
                    job['error'] = 'models failed to load — see server console'
        return

    while True:
        job_id = JOB_QUEUE.get()
        with JOBS_LOCK:
            job = JOBS[job_id]
        job['status'] = 'running'
        job['started_at'] = datetime.now().isoformat(timespec='seconds')

        def log(line, _job=job):
            print(f'[{_job["id"][:8]}] {line}')
            _job['log'].append(line)

        def on_segment(i, total, name, _job=job):
            _job['progress'] = {'segment': i + 1, 'total': total, 'name': name}

        def on_plan(names, _job=job):
            _job['plan'] = names          # every segment name, so the page can colour the arc

        try:
            if job.get('kind') == 'inpaint':
                with JOBS_LOCK:
                    parent = JOBS[job['parent']]
                    base = next(v for v in versions_of(parent) if v.get('version', 1) == job['base_version'])
                result = pipeline.inpaint_session(
                    base, job['parts'], fixes=job['fixes'], note=job['note'], seed=job['seed'],
                    log=log, on_segment=on_segment,
                )
                job['result'] = result_for_page(result)
                with JOBS_LOCK:      # the session's job gets the new version, and plays it from now on
                    parent['versions'] = versions_of(parent) + [job['result']]
                    parent['result'] = job['result']
            else:
                result = pipeline.generate_session(
                    job['request'], seed=job['seed'], log=log, on_segment=on_segment,
                    mode=job['mode'], options=job['options'], on_plan=on_plan,
                )
                job['result'] = result_for_page(result)
                job['versions'] = [job['result']]
            job['status'] = 'done'
        except Exception as e:
            job['status'] = 'error'
            if isinstance(e, ValueError) and job.get('kind') == 'inpaint':
                job['error'] = f'{str(e)[:1].upper()}{str(e)[1:]}.'   # the pipeline words these for the page
            else:
                job['error'] = f'{type(e).__name__}: {e}'
            log(job['error'])
            print(traceback.format_exc())
        finally:
            job['finished_at'] = datetime.now().isoformat(timespec='seconds')
            if job.get('kind') == 'inpaint':
                with JOBS_LOCK:
                    parent = JOBS.get(job['parent'])
                    if parent and parent.get('pending_fix') == job['id']:
                        parent['pending_fix'] = None
            JOB_QUEUE.task_done()


@app.on_event('startup')
def startup():
    threading.Thread(target=worker, daemon=True).start()


@app.get('/health')
def health():
    with JOBS_LOCK:
        running = any(j['status'] == 'running' for j in JOBS.values())
    return {
        'models_loaded': READY.is_set(),
        'error': MODEL_ERROR.strip().splitlines()[-1] if MODEL_ERROR else None,
        'queued': JOB_QUEUE.qsize(),
        'running': running,
    }


@app.post('/generate')
def generate(req: GenerateRequest):
    if not req.request.strip():
        raise HTTPException(400, 'Describe the session you want before generating.')
    if MODEL_ERROR:
        raise HTTPException(503, 'The models failed to load, so nothing can be generated.')
    job_id = str(uuid.uuid4())
    job = {
        'id': job_id,
        'request': req.request.strip(),
        'seed': req.seed,
        'mode': req.mode,
        'options': req.options if req.mode != 'iso' else None,
        'status': 'queued',
        'log': [],
        'progress': None,
        'plan': None,
        'result': None,
        'error': None,
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    JOB_QUEUE.put(job_id)
    return {'job_id': job_id, 'position': JOB_QUEUE.qsize()}


@app.post('/inpaint')
def inpaint(req: InpaintRequest):
    if MODEL_ERROR:
        raise HTTPException(503, 'The models failed to load, so nothing can be regenerated.')
    if not req.parts:
        raise HTTPException(400, 'Choose a part of the session to regenerate.')
    if len(req.parts) > config.INPAINT_MAX_PARTS:
        raise HTTPException(400, f'Choose at most {config.INPAINT_MAX_PARTS} parts at a time.')
    with JOBS_LOCK:
        parent = JOBS.get(req.job_id)
        if not parent or parent.get('kind') == 'inpaint':
            raise HTTPException(404, 'no such session')
        if parent['status'] != 'done' or not parent.get('result'):
            raise HTTPException(409, 'This session is not finished yet.')
        versions = versions_of(parent)
        base = versions[-1] if req.version is None else \
            next((v for v in versions if v.get('version', 1) == req.version), None)
        if base is None:
            raise HTTPException(404, f'This session has no version {req.version}.')
        pending = JOBS.get(parent.get('pending_fix') or '')
        if pending and pending['status'] in ('queued', 'running'):
            raise HTTPException(409, 'A part of this session is already being regenerated.')
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            'id': job_id,
            'kind': 'inpaint',
            'parent': req.job_id,
            'base_version': base.get('version', 1),
            'request': parent['request'],
            'mode': parent.get('mode', 'iso'),
            'options': None,
            'seed': req.seed,
            'parts': req.parts,
            'fixes': [f for f in req.fixes if isinstance(f, str)],
            'note': req.note.strip()[:600],
            'status': 'queued',
            'log': [],
            'progress': None,
            'plan': None,
            'result': None,
            'error': None,
            'created_at': datetime.now().isoformat(timespec='seconds'),
        }
        parent['pending_fix'] = job_id
    JOB_QUEUE.put(job_id)
    return {'job_id': job_id, 'position': JOB_QUEUE.qsize()}


@app.get('/jobs/{job_id}')
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'no such job')
    return job


@app.get('/jobs')
def list_jobs():
    """Summaries only — the full job carries a 900-point waveform we don't need here.
    Fixes are versions of their session, not sessions of their own, so they aren't listed."""
    with JOBS_LOCK:
        snapshot = [j for j in JOBS.values() if j.get('kind') != 'inpaint']
    snapshot.sort(key=lambda j: j['created_at'], reverse=True)
    return [
        {
            'id': j['id'],
            'request': j['request'],
            'mode': j.get('mode', 'iso'),
            'status': j['status'],
            'created_at': j['created_at'],
            'result': {'duration_s': j['result']['duration_s']} if j['result'] else None,
            'versions': len(versions_of(j)),
        }
        for j in snapshot[:50]
    ]


@app.get('/audio/{folder}/{filename}')
def audio(folder: str, filename: str):
    # resolve and confine to WORK — a folder name like ../.. must not escape
    path = (config.WORK / folder / filename).resolve()
    if not str(path).startswith(str(config.WORK.resolve())) or not path.is_file():
        raise HTTPException(404, 'not found')
    media = 'audio/mpeg' if path.suffix == '.mp3' else 'audio/wav'
    return FileResponse(path, media_type=media, filename=f'{folder}_{filename}')


app.mount('/', StaticFiles(directory=Path(__file__).parent / 'static', html=True), name='static')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
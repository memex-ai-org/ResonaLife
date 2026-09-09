"""FastAPI wrapper around pipeline.generate_session.

One GPU means one generation at a time, so requests go into a queue and a
single background thread drains it. POST /generate returns immediately with a
job id; the browser polls GET /jobs/{id} until it's done.
"""
import queue
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import pipeline

app = FastAPI(title='VNX Functional Music GAI')

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

        try:
            result = pipeline.generate_session(
                job['request'], seed=job['seed'], log=log, on_segment=on_segment,
            )
            folder = Path(result['eq_path']).parent.name
            job['result'] = {
                'duration_s': result['duration_s'],
                'prompts': result['prompts'],
                'segments': result['segments'],
                'peaks': result['peaks'],
                # small file for the browser to stream
                'audio_url': f'/audio/{folder}/{Path(result["preview_path"]).name}',
                # full-quality 24-bit deliverable
                'download_url': f'/audio/{folder}/completed_EQed.wav',
                'raw_url': f'/audio/{folder}/completed.wav',
            }
            job['status'] = 'done'
        except Exception as e:
            job['status'] = 'error'
            job['error'] = f'{type(e).__name__}: {e}'
            log(job['error'])
            print(traceback.format_exc())
        finally:
            job['finished_at'] = datetime.now().isoformat(timespec='seconds')
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
        'status': 'queued',
        'log': [],
        'progress': None,
        'result': None,
        'error': None,
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
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
    """Summaries only — the full job carries a 900-point waveform we don't need here."""
    with JOBS_LOCK:
        snapshot = list(JOBS.values())
    snapshot.sort(key=lambda j: j['created_at'], reverse=True)
    return [
        {
            'id': j['id'],
            'request': j['request'],
            'status': j['status'],
            'created_at': j['created_at'],
            'result': {'duration_s': j['result']['duration_s']} if j['result'] else None,
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
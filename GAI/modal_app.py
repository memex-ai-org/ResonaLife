"""ResonaLife Functional Music GAI on Modal.

Two containers behind one URL:
  web        CPU: serves static/ and the same HTTP API server.py had
  Generator  GPU: loads Stable Audio 3 once, renders one session at a time

pipeline.py, config.py and static/index.html are used unchanged. Every session
mode (ISO, and the guideline's pain relief, dementia and depression modules) goes
through the same POST /generate, and POST /inpaint regenerates parts of a finished
session as a new version of it — see server.py for the request bodies.

Run from the GAI folder (next to pipeline.py, config.py, static/, stable-audio-3-main/):
  python -m modal run modal_app.py       one-time: build image + download weights
  python -m modal deploy modal_app.py    put it online permanently
"""
import base64
import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import modal

HERE = Path(__file__).parent
ROOT = '/resonalife'                   # RESONALIFE_ROOT inside the containers
SESSIONS = f'{ROOT}/sessions'   # = config.WORK

app = modal.App('resonalife-gai')

models_vol = modal.Volume.from_name('resonalife-models', create_if_missing=True)
outputs_vol = modal.Volume.from_name('resonalife-outputs', create_if_missing=True)
jobs = modal.Dict.from_name('resonalife-jobs', create_if_missing=True)
calls = modal.Dict.from_name('resonalife-calls', create_if_missing=True)
resonalife_secrets = modal.Secret.from_name('resonalife-secrets')   # HF_TOKEN, RESONALIFE_PASSWORD


def now():
    # UTC with offset, so browsers in any time zone compute elapsed time correctly
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


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
        'audio_url': f'/audio/{folder}/{Path(result["preview_path"]).name}',
        'download_url': f'/audio/{folder}/{Path(result["eq_path"]).name}',
        'raw_url': f'/audio/{folder}/{Path(result["raw_path"]).name}',
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


# ============================================================
# Images
# ============================================================

# Official Linux wheel. ${ABI} is filled in at build time from the installed
# torch, so the wheel always matches torch's C++ ABI.
FLASH_ATTN_WHEEL = (
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/'
    'flash_attn-2.8.3+cu12torch2.7cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl'
)

gpu_image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('ffmpeg', 'git')
    .pip_install(
        'torch==2.7.1', 'torchvision==0.22.1', 'torchaudio==2.7.1',
        index_url='https://download.pytorch.org/whl/cu126',
    )
    .run_commands(
        'ABI=$(python -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())") '
        f'&& pip install "{FLASH_ATTN_WHEEL}"'
    )
    .add_local_dir(
        HERE / 'stable-audio-3-main', '/opt/stable-audio-3', copy=True,
        ignore=['**/.git', '**/__pycache__', '**/*.egg-info', '**/.venv'],
    )
    .run_commands('pip install -e /opt/stable-audio-3')
    .pip_install(
        'transformers>=4.46.0', 'accelerate', 'huggingface_hub', 'bitsandbytes',
        'numpy', 'scipy', 'soundfile', 'pyloudnorm',
    )
    .env({'RESONALIFE_ROOT': ROOT})
    .add_local_python_source('config', 'pipeline')
)

web_image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('fastapi[standard]', 'pydantic>=2')
    .add_local_dir(HERE / 'static', '/root/static')
)


# ============================================================
# GPU worker
# ============================================================

@app.cls(
    image=gpu_image,
    gpu='L40S',
    volumes={f'{ROOT}/hf_cache': models_vol, SESSIONS: outputs_vol},
    secrets=[resonalife_secrets],
    timeout=2 * 60 * 60,        # longest single session allowed
    scaledown_window=5 * 60,    # GPU stays warm 5 min after the last job
    max_containers=1,           # one GPU, jobs queue in order; raise to 2 for parallel sessions
    retries=1,                  # only fires on a container crash; pipeline errors are caught below
)
class Generator:
    @modal.enter()
    def load(self):
        import pipeline
        pipeline.load_models()
        models_vol.commit()     # first run downloads SA3 into the volume; keep it

    @modal.method()
    def warmup(self):
        import pipeline
        pipeline.load_qwen()    # first run downloads Qwen into the volume
        pipeline.unload_qwen()
        models_vol.commit()
        return 'Stable Audio 3 and Qwen are cached in resonalife-models'

    @modal.method()
    def generate(self, job_id):
        import pipeline

        job = jobs[job_id]
        job['status'] = 'running'
        job['started_at'] = now()
        job['log'] = []          # reset in case this is a retry after a crash
        job['progress'] = None
        jobs[job_id] = job

        def log(line):
            print(f'[{job_id[:8]}] {line}', flush=True)
            job['log'].append(line)
            jobs[job_id] = job

        def on_segment(i, total, name):
            job['progress'] = {'segment': i + 1, 'total': total, 'name': name}
            jobs[job_id] = job

        def on_plan(names):
            job['plan'] = names      # every segment name, so the page can colour the arc
            jobs[job_id] = job

        try:
            result = pipeline.generate_session(
                job['request'], seed=job['seed'], log=log, on_segment=on_segment,
                mode=job.get('mode', 'iso'), options=job.get('options'), on_plan=on_plan,
            )
            outputs_vol.commit()     # make the files visible to the web container
            models_vol.commit()
            job['result'] = result_for_page(result)
            job['versions'] = [job['result']]
            job['status'] = 'done'
        except Exception as e:
            import traceback
            job['status'] = 'error'
            job['error'] = f'{type(e).__name__}: {e}'
            job['log'].append(job['error'])
            print(traceback.format_exc(), flush=True)
        finally:
            job['finished_at'] = now()
            jobs[job_id] = job

    @modal.method()
    def inpaint(self, job_id):
        """Regenerate parts of a finished session (POST /inpaint) as its newest version."""
        import pipeline

        job = jobs[job_id]
        job['status'] = 'running'
        job['started_at'] = now()
        job['log'] = []
        job['progress'] = None
        jobs[job_id] = job

        def log(line):
            print(f'[{job_id[:8]}] {line}', flush=True)
            job['log'].append(line)
            jobs[job_id] = job

        def on_segment(i, total, name):
            job['progress'] = {'segment': i + 1, 'total': total, 'name': name}
            jobs[job_id] = job

        try:
            try:
                outputs_vol.reload()     # the session may have been rendered by an earlier container
            except Exception:
                pass
            parent = jobs[job['parent']]
            base = next(v for v in versions_of(parent) if v.get('version', 1) == job['base_version'])
            result = pipeline.inpaint_session(
                base, job['parts'], fixes=job.get('fixes'), note=job.get('note', ''), seed=job.get('seed'),
                log=log, on_segment=on_segment,
            )
            outputs_vol.commit()
            job['result'] = result_for_page(result)
            # the session's job first, while this one still shows running: the page only
            # asks for another fix once it sees this one done
            parent = jobs[job['parent']]
            parent['versions'] = versions_of(parent) + [job['result']]
            parent['result'] = job['result']
            parent['pending_fix'] = None
            jobs[job['parent']] = parent
            job['status'] = 'done'
        except Exception as e:
            import traceback
            job['status'] = 'error'
            if isinstance(e, ValueError):    # the pipeline words these for the page
                job['error'] = f'{str(e)[:1].upper()}{str(e)[1:]}.'
            else:
                job['error'] = f'{type(e).__name__}: {e}'
            job['log'].append(job['error'])
            print(traceback.format_exc(), flush=True)
        finally:
            job['finished_at'] = now()
            jobs[job_id] = job


# ============================================================
# Web app: same routes as server.py, plus a password on everything
# ============================================================

@app.function(image=web_image, volumes={SESSIONS: outputs_vol}, secrets=[resonalife_secrets])
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    api = FastAPI(title='ResonaLife Functional Music GAI')

    # Middleware rather than a route dependency, so the page and the audio
    # files are behind the password too. The browser shows its own login box.
    expected = ('Basic ' + base64.b64encode(
        f'resonalife:{os.environ["RESONALIFE_PASSWORD"]}'.encode()).decode()).encode()

    @api.middleware('http')
    async def basic_auth(request: Request, call_next):
        given = request.headers.get('authorization', '').encode()
        if not secrets.compare_digest(given, expected):
            return Response(status_code=401, headers={'WWW-Authenticate': 'Basic realm="ResonaLife"'})
        return await call_next(request)

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

    def check_call(job):
        """If the GPU call died without updating the job (model load failed,
        container killed), surface that instead of leaving it queued forever."""
        if job['status'] not in ('queued', 'running'):
            return job
        call_id = calls.get(job['id'])
        if not call_id:
            return job
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            return job                      # still queued or running: normal
        except Exception as e:
            job['status'] = 'error'
            job['error'] = f'GPU worker failed: {type(e).__name__}: {e}'
            job['finished_at'] = now()
            jobs[job['id']] = job
        return job

    @api.get('/health')
    def health():
        snapshot = list(jobs.values())
        return {
            'models_loaded': True,          # the GPU loads on demand now
            'error': None,
            'queued': sum(1 for j in snapshot if j['status'] == 'queued'),
            'running': any(j['status'] == 'running' for j in snapshot),
        }

    @api.post('/generate')
    def generate(req: GenerateRequest):
        if not req.request.strip():
            raise HTTPException(400, 'Describe the session you want before generating.')
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
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
            'created_at': now(),
        }
        call = Generator().generate.spawn(job_id)
        calls[job_id] = call.object_id
        position = sum(1 for j in jobs.values() if j['status'] in ('queued', 'running'))
        return {'job_id': job_id, 'position': position}

    @api.post('/inpaint')
    def inpaint(req: InpaintRequest):
        if not req.parts:
            raise HTTPException(400, 'Choose a part of the session to regenerate.')
        parent = jobs.get(req.job_id)
        if not parent or parent.get('kind') == 'inpaint':
            raise HTTPException(404, 'no such session')
        if parent['status'] != 'done' or not parent.get('result'):
            raise HTTPException(409, 'This session is not finished yet.')
        versions = versions_of(parent)
        base = versions[-1] if req.version is None else \
            next((v for v in versions if v.get('version', 1) == req.version), None)
        if base is None:
            raise HTTPException(404, f'This session has no version {req.version}.')
        pending = jobs.get(parent.get('pending_fix') or '')
        if pending and check_call(pending)['status'] in ('queued', 'running'):
            raise HTTPException(409, 'A part of this session is already being regenerated.')
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
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
            'created_at': now(),
        }
        parent['pending_fix'] = job_id
        jobs[req.job_id] = parent
        call = Generator().inpaint.spawn(job_id)
        calls[job_id] = call.object_id
        position = sum(1 for j in jobs.values() if j['status'] in ('queued', 'running'))
        return {'job_id': job_id, 'position': position}

    @api.get('/jobs/{job_id}')
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, 'no such job')
        return check_call(job)

    @api.get('/jobs')
    def list_jobs():
        # fixes are versions of their session, not sessions of their own
        snapshot = sorted((j for j in jobs.values() if j.get('kind') != 'inpaint'),
                          key=lambda j: j['created_at'], reverse=True)
        return [
            {
                'id': j['id'],
                'request': j['request'],
                'mode': j.get('mode', 'iso'),     # jobs from before the guideline modules have no mode
                'status': j['status'],
                'created_at': j['created_at'],
                'result': {'duration_s': j['result']['duration_s']} if j['result'] else None,
                'versions': len(versions_of(j)),
            }
            for j in snapshot[:50]
        ]

    @api.get('/audio/{folder}/{filename}')
    def audio(folder: str, filename: str):
        base = Path(SESSIONS).resolve()
        path = (base / folder / filename).resolve()
        if not str(path).startswith(str(base) + os.sep):
            raise HTTPException(404, 'not found')
        if not path.is_file():
            try:
                outputs_vol.reload()        # pick up files the GPU container just committed
            except Exception:
                pass                        # reload refuses while another download is open
        if not path.is_file():
            raise HTTPException(404, 'not found')
        media = 'audio/mpeg' if path.suffix == '.mp3' else 'audio/wav'
        return FileResponse(path, media_type=media, filename=f'{folder}_{filename}')

    api.mount('/', StaticFiles(directory='/root/static', html=True), name='static')
    return api


# ============================================================
# One-time warmup:  python -m modal run modal_app.py
# ============================================================

@app.local_entrypoint()
def main():
    print(Generator().warmup.remote())

"""ResonaLife Functional Music GAI on Modal.

Two containers behind one URL:
  web        CPU: serves static/ and the same HTTP API server.py had
  Generator  GPU: loads Stable Audio 3 once, renders one session at a time

pipeline.py, config.py and static/index.html are used unchanged. Both session
modes (ISO, and the guideline's six-phase pain relief arc) go through the same
POST /generate — see server.py for the request body.

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
    return {
        'mode': result['mode'],
        'ending': result.get('ending'),
        'settings': result.get('settings'),
        'duration_s': result['duration_s'],
        'prompts': result['prompts'],
        'segments': result['segments'],
        'peaks': result['peaks'],
        'audio_url': f'/audio/{folder}/{Path(result["preview_path"]).name}',
        'download_url': f'/audio/{folder}/completed_EQed.wav',
        'raw_url': f'/audio/{folder}/completed.wav',
        'loop_check_url': f'/audio/{folder}/{Path(loop_check).name}' if loop_check else None,
    }


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
        mode: Literal['iso', 'pain_relief'] = 'iso'
        options: dict | None = None    # pain relief settings; validated by the pipeline

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
            'options': req.options if req.mode == 'pain_relief' else None,
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

    @api.get('/jobs/{job_id}')
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, 'no such job')
        return check_call(job)

    @api.get('/jobs')
    def list_jobs():
        snapshot = sorted(jobs.values(), key=lambda j: j['created_at'], reverse=True)
        return [
            {
                'id': j['id'],
                'request': j['request'],
                'mode': j.get('mode', 'iso'),     # jobs from before pain relief have no mode
                'status': j['status'],
                'created_at': j['created_at'],
                'result': {'duration_s': j['result']['duration_s']} if j['result'] else None,
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

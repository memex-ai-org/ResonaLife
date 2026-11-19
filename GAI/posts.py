"""Posts: finished sessions saved to the page's Posts tab, which everyone signed in sees.

A post is a snapshot of one version of a session, kept as one JSON file,
<sessions>/_posts/<id>.json, on the same disk (server.py) or Modal Volume
(modal_app.py) as the audio, which stays in the session's own folder. Posts last
until someone deletes them. The "Your sessions" list doesn't: on Modal it lives in a
Dict, whose entries expire after 7 days without a read or write.

A post holds what the page needs to play it, draw it and show how it was made. It
never holds the request text, which can carry the carer's notes, or the listener
profile. Every post is its own file, so two people posting at once never write the
same file (on a Modal Volume the last write to a file wins).

Neither server needs the models for any of this.
"""
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

TITLE_MAX = 80
TEXT_MAX = 300
NAME_MAX = 40
THUMB_POINTS = 180          # the waveform a card draws; the full 900 come with the post itself
POST_ID = re.compile(r'[0-9a-f]{32}')

# what the Posts tab gets for each card
SUMMARY_KEYS = ('id', 'created_at', 'title', 'description', 'author', 'session_id', 'version',
                'mode', 'ending', 'duration_s', 'segments', 'thumb')


def folder(sessions):
    """Where the posts live: a folder beside the session folders, which can't clash with
    them (session folder names are made of a-z, 0-9 and dashes)."""
    return Path(sessions) / '_posts'


def clean_text(value, limit):
    return re.sub(r'\s+', ' ', str(value or '')).strip()[:limit]


def thumb(peaks, points=THUMB_POINTS):
    """The peak envelope cut down to `points`, each the loudest of its stretch."""
    peaks = list(peaks or [])
    if len(peaks) <= points:
        return peaks
    step = len(peaks) / points
    return [max(peaks[int(i * step):max(int(i * step) + 1, int((i + 1) * step))]) for i in range(points)]


def make(job, version, title, description='', author=''):
    """A post of one version of a finished session's job (the newest if version is None).
    Raises LookupError if the job has no such version, ValueError if there's no title."""
    versions = job.get('versions') or ([job['result']] if job.get('result') else [])
    if version is None:
        v = versions[-1] if versions else None
    else:
        v = next((x for x in versions if x.get('version', 1) == version), None)
    if v is None:
        raise LookupError(f'This session has no version {version}.')
    title = clean_text(title, TITLE_MAX)
    if not title:
        raise ValueError('Give the post a title.')
    settings = v.get('settings')
    if isinstance(settings, dict):
        settings = {k: val for k, val in settings.items() if k != 'profile'}   # the listener profile stays private
    peaks = v.get('peaks') or []
    return {
        'id': uuid.uuid4().hex,
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'title': title,
        'description': clean_text(description, TEXT_MAX),
        'author': clean_text(author, NAME_MAX),
        'session_id': job['id'],
        'version': v.get('version', 1),
        'mode': v.get('mode') or job.get('mode') or 'iso',
        'ending': v.get('ending'),
        'settings': settings,
        'duration_s': v['duration_s'],
        'segments': v['segments'],
        'prompts': v.get('prompts'),
        'tempo': v.get('tempo'),
        'peaks': peaks,
        'thumb': thumb(peaks),
        'audio_url': v['audio_url'],
        'download_url': v['download_url'],
        'raw_url': v.get('raw_url'),
        'loop_check_url': v.get('loop_check_url'),
    }


def summary(post):
    return {key: post.get(key) for key in SUMMARY_KEYS}


def save(sessions, post):
    d = folder(sessions)
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{post["id"]}.json').write_text(json.dumps(post))


def load(sessions, post_id):
    """The post with this id, or None."""
    if not POST_ID.fullmatch(str(post_id or '')):
        return None
    try:
        return json.loads((folder(sessions) / f'{post_id}.json').read_text())
    except (OSError, ValueError):
        return None


def load_all(sessions):
    """Every post, newest first. A file that can't be read (half written) is skipped."""
    d = folder(sessions)
    posts = []
    for path in d.glob('*.json') if d.is_dir() else []:
        try:
            posts.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    posts.sort(key=lambda p: p.get('created_at', ''), reverse=True)
    return posts


def find(sessions, session_id, version):
    """The post of this session's version, if it has been posted."""
    return next((p for p in load_all(sessions)
                 if p.get('session_id') == session_id and p.get('version') == version), None)


def delete(sessions, post_id):
    """Remove a post (its audio stays with its session). False if there was no such post."""
    if not POST_ID.fullmatch(str(post_id or '')):
        return False
    path = folder(sessions) / f'{post_id}.json'
    if not path.is_file():
        return False
    path.unlink()
    return True

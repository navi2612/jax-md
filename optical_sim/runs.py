"""Saving, loading and provenance for simulation runs.

A *run* on disk is a self-documenting folder::

    results/<prefix>_<timestamp>[_<label>]/
        trajectory.npz     the arrays (centers, angles, box, ... )
        metadata.json      every parameter, plus date / versions / git state
        snapshot.png       optional, written by a caller-supplied plot_fn

``save_run`` writes one, ``load_run`` reads one back, and ``as_run`` accepts a
run dict, a run folder or a bare ``.npz`` path so downstream code (plotting,
video, analysis) can take any of the three.

This module is deliberately dependency-free (stdlib + numpy) so both the
point-particle and the swarm paths can use it without an import cycle.

Provenance note
---------------
Recording ``git rev-parse HEAD`` alone is not enough: a run made from a script
that was still *uncommitted* points at a commit where its own source does not
exist (this happened -- see results/swarm_20260616-152120_dense_spinners, whose
metadata names commit 79a9aa5, an hour before simulate_swarm.py was first
committed). So ``save_run`` also records whether tracked files were dirty and
the git status of the script that produced the run; pass ``source=__file__``.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
_REPO = Path(__file__).resolve().parent


def json_default(o):
    """Fallback JSON encoder for numpy / jax scalars and arrays."""
    try:
        arr = np.asarray(o)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    except Exception:
        return str(o)


def _git(*args, cwd=_REPO):
    """Run a git command, returning stripped stdout or None on any failure."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_commit():
    """Short commit hash of HEAD, or None if unavailable."""
    return _git("rev-parse", "--short", "HEAD")


def git_dirty():
    """True if *tracked* files differ from HEAD (untracked files ignored).

    Untracked files are excluded because results/ and scratch output would
    otherwise make this always True; use ``git_file_status`` to ask whether the
    specific script that ran is committed.
    """
    out = _git("status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out)


def git_file_status(path):
    """'clean' | 'modified' | 'untracked' | None for one file.

    'untracked' is the dangerous one: the recorded commit will not contain the
    code that produced the run.
    """
    p = Path(path).resolve()
    out = _git("status", "--porcelain", "--", str(p))
    if out is None:
        return None
    if not out:
        return "clean"
    return "untracked" if out.split()[0] == "??" else "modified"


def provenance(source=None):
    """Dict of git/date provenance. Pass ``source=__file__`` from the caller."""
    info = {"datetime": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(), "git_dirty": git_dirty()}
    if source is not None:
        info["source_file"] = Path(source).name
        info["source_git_status"] = git_file_status(source)
    return info


def run_dir_name(label=None, prefix="run", stamp=None):
    """``<prefix>_<timestamp>_<label>`` (pieces omitted when empty)."""
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [p for p in (prefix, stamp, label) if p]
    return "_".join(str(p) for p in parts)


# Code as it was when this process started. Snapshotting at import time (and
# not when a run is saved) matters: a script edited while a long run is still
# going would otherwise be recorded with code the run never executed.
_CODE_DIR = Path(__file__).resolve().parent
_CODE_SNAPSHOT = {}
for _p in sorted(_CODE_DIR.glob("*.py")):
    try:
        _CODE_SNAPSHOT[_p.name] = _p.read_bytes()
    except OSError:
        pass


def _save_code(run_dir, source=None):
    """Copy the start-of-process code of every optical_sim module this process
    imported (plus ``source``) into ``run_dir/code``; return name -> sha256."""
    names = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve().parent == _CODE_DIR:
            names.add(Path(f).name)
    if source is not None:
        names.add(Path(source).name)
    code_dir = Path(run_dir) / "code"
    code_dir.mkdir(exist_ok=True)
    digests = {}
    for n in sorted(names):
        blob = _CODE_SNAPSHOT.get(n)
        if blob is None:                       # created after start: best effort
            try:
                blob = (_CODE_DIR / n).read_bytes()
            except OSError:
                continue
        (code_dir / n).write_bytes(blob)
        digests[n] = hashlib.sha256(blob).hexdigest()[:16]
    return digests


def save_run(arrays, meta=None, label=None, results_dir=RESULTS_DIR,
             prefix="run", plot_fn=None, source=None):
    """Write one run folder and return its ``Path``.

    ``arrays``   dict name -> array, saved (compressed) into trajectory.npz.
    ``meta``     dict of parameters; provenance fields are added, not replaced.
    ``plot_fn``  optional ``plot_fn(run_dir)`` for a snapshot figure.
    ``source``   pass ``__file__`` so the script's own git status is recorded.

    The code of every ``optical_sim`` module the process imported is copied
    into ``code/`` as it was at process start, so a run stays reproducible
    even when the scripts are uncommitted or edited while it runs.

    If the target folder exists, ``-1``, ``-2``, ... is appended rather than
    overwriting -- runs are never silently clobbered.
    """
    results_dir = Path(results_dir)
    name = run_dir_name(label, prefix)
    run_dir = results_dir / name
    i = 1
    while run_dir.exists():
        run_dir = results_dir / f"{name}-{i}"
        i += 1
    run_dir.mkdir(parents=True)

    np.savez_compressed(run_dir / "trajectory.npz", **arrays)

    meta = dict(meta or {})
    meta.update(provenance(source))
    meta["code_sha256"] = _save_code(run_dir, source)
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=json_default)

    if plot_fn is not None:
        try:
            plot_fn(run_dir)
        except Exception as e:                  # a failed plot must not lose data
            print(f"WARNING: plot_fn failed ({e}); arrays and metadata are saved")

    print(f"saved run to {run_dir}")
    return run_dir


def load_run(run_dir):
    """Load a run folder into a dict of its arrays plus ``meta``."""
    run_dir = Path(run_dir)
    d = np.load(run_dir / "trajectory.npz")
    out = {k: d[k] for k in d.files}
    out = {k: (v.item() if getattr(v, "ndim", 1) == 0 else v)
           for k, v in out.items()}
    meta_p = run_dir / "metadata.json"
    if meta_p.exists():
        out["meta"] = json.loads(meta_p.read_text())
    return out


def as_run(run):
    """Normalize a run dict, a run folder, or an ``.npz`` path into a dict."""
    if isinstance(run, dict):
        return run
    p = Path(run)
    if p.is_dir():
        return load_run(p)
    d = np.load(p)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}

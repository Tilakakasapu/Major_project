import json
import random
import time
from contextlib import contextmanager

import numpy as np


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def banner(step, total, title):
    line = "=" * 78
    print(f"\n{line}\n STEP {step}/{total}: {title}\n{line}", flush=True)


@contextmanager
def timed(label):
    t0 = time.time()
    log(f"▶ {label} ...")
    yield
    log(f"✔ {label} finished in {time.time() - t0:.1f}s")


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def save_json(obj, path, indent=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    tmp.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

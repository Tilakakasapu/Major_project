"""Step 1: download clean images, adversarial images, the diffusion repo and checkpoint."""
import subprocess

import requests

import config as C
from src.utils import log, timed

PNG = b"\x89PNG\r\n\x1a\n"


def _is_png(fp):
    if not fp.exists():
        return False
    with open(fp, "rb") as f:
        return f.read(8) == PNG


def _fetch_set(base_url, out_dir, n, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    got = cached = 0
    failed = []
    for i in range(n):
        fp = out_dir / f"{i}.png"
        if _is_png(fp):
            cached += 1
            continue
        for attempt in range(3):
            try:
                r = requests.get(f"{base_url}/{i}.png", timeout=60)
                r.raise_for_status()
                if not r.content.startswith(PNG):      # Git-LFS pointer instead of image data
                    raise ValueError("response is not a PNG (Git-LFS pointer?)")
                fp.write_bytes(r.content)
                got += 1
                break
            except Exception as e:                      # noqa: BLE001
                if attempt == 2:
                    failed.append(i)
                    log(f"   ✘ {label}/{i}.png failed: {e}")
        if (i + 1) % 25 == 0:
            log(f"   {label}: {i + 1}/{n} checked")
    log(f"   {label}: {got} downloaded, {cached} already present, {len(failed)} failed")
    return failed


def _download_checkpoint():
    if C.CKPT.exists() and C.CKPT.stat().st_size > 1e9:
        log(f"   checkpoint already present: {C.CKPT}")
        return
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = C.CKPT.with_suffix(".part")
    with requests.get(C.CKPT_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total, done, nxt = int(r.headers.get("content-length", 0)), 0, 10
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done * 100 // total >= nxt:
                    log(f"   checkpoint download {nxt}%")
                    nxt += 10
    tmp.replace(C.CKPT)


def run(n, attacks):
    failed = {}
    with timed(f"clean images (NIPS17, {n})"):
        failed["clean"] = _fetch_set(C.CLEAN_URL, C.CLEAN_DIR, n, "clean")
    for key in attacks:
        with timed(f"adversarial images: {C.ATTACKS[key]}"):
            failed[key] = _fetch_set(f"{C.ADV_URL}/{C.ATTACKS[key]}", C.attack_dir(key), n, key)
    with timed("guided-diffusion repository"):
        if C.GD_DIR.exists():
            log("   already cloned")
        else:
            C.WORK.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "-q", C.GD_REPO, str(C.GD_DIR)], check=True)
    with timed("diffusion checkpoint (256x256_diffusion_uncond.pt, ~2 GB)"):
        _download_checkpoint()
    bad = {k: v for k, v in failed.items() if v}
    if bad:
        raise RuntimeError(f"Some images could not be downloaded: {bad}. Is Kaggle 'Internet' switched on?")
    log(f"dataset ready: {n} images x {1 + len(attacks)} sets")

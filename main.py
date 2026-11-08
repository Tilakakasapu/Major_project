#!/usr/bin/env python
"""DiffCAP evaluation pipeline.  Run:  python main.py

Stages (all run by default, cached results are reused so you can safely re-run after an interruption):
  1 download  - clean + adversarial images, guided-diffusion repo, diffusion checkpoint
  2 purify    - DiffCAP purification of every image
  3 vlm       - captioning + classification with every VLM, with and without the defense
  4 metrics   - task-specific metrics
  5 analysis  - statistics, written analysis, graphs, results for the web UI
"""
import argparse
import subprocess
import sys
import time

import config as C
from src.utils import banner, log, set_seed, timed

STAGES = ["download", "purify", "vlm", "metrics", "analysis"]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-images", type=int, default=C.N_IMAGES, help="images per set (default 100)")
    p.add_argument("--vlms", nargs="+", choices=list(C.VLMS), help="subset of VLMs")
    p.add_argument("--attacks", nargs="+", choices=list(C.ATTACKS), help="subset of attacks")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES, help="stages to run")
    p.add_argument("--tau", type=float, help="fixed DiffCAP threshold (default: calibrate)")
    p.add_argument("--quick", action="store_true", help="smoke test: 8 images")
    p.add_argument("--install", action="store_true", help="pip install -r requirements.txt first")
    return p.parse_args()


def main():
    a = parse()
    t_start = time.time()
    C.N_IMAGES = 8 if a.quick else a.n_images
    if a.vlms:
        C.ACTIVE_VLMS = a.vlms
    if a.attacks:
        C.ACTIVE_ATTACKS = a.attacks
    if a.install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(C.ROOT / "requirements.txt")], check=True)
    set_seed(C.SEED)
    n, total = C.N_IMAGES, len(a.stages)
    print(f"\nDiffCAP pipeline | images/set: {n} | attacks: {C.ACTIVE_ATTACKS} | VLMs: {C.ACTIVE_VLMS} | stages: {a.stages}")
    step = 0

    if "download" in a.stages:
        step += 1; banner(step, total, "Download dataset, diffusion repo and checkpoint")
        from src import data
        data.run(n, C.ACTIVE_ATTACKS)

    if "purify" in a.stages:
        step += 1; banner(step, total, "DiffCAP purification of clean + adversarial images")
        from src import diffcap
        with timed("purification"):
            diffcap.run_purification(n, a.tau)

    if "vlm" in a.stages:
        step += 1; banner(step, total, "VLM inference: captioning + classification (undefended vs DiffCAP)")
        from src import vlms
        for name in C.ACTIVE_VLMS:
            with timed(f"{name}"):
                vlms.run_vlm(name, n)

    if "metrics" in a.stages:
        step += 1; banner(step, total, "Compute task-specific metrics")
        from src import metrics
        with timed("metrics"):
            metrics.evaluate(n, C.ACTIVE_VLMS, C.ACTIVE_ATTACKS)

    if "analysis" in a.stages:
        step += 1; banner(step, total, "Analysis, graphs and web-UI data")
        from src import analysis
        with timed("analysis"):
            analysis.run()

    log(f"PIPELINE FINISHED in {(time.time() - t_start) / 60:.1f} min")
    log(f"Results folder: {C.RESULTS}   (commit it to GitHub, then open web/index.html locally)")


if __name__ == "__main__":
    main()

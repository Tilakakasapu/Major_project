"""Central configuration. Edit values here; a few can be overridden from the command line (see main.py)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- big files (git-ignored) -------------------------------------------------
WORK = ROOT / "workdir"
DATA = WORK / "dataset"
CLEAN_DIR = DATA / "clean"
ADV_ROOT = DATA / "adversarial"
PUR = WORK / "purified"
CKPT_DIR = WORK / "ckpt"
GD_DIR = WORK / "guided-diffusion"
CKPT = CKPT_DIR / "256x256_diffusion_uncond.pt"

# ---- small files (commit these to GitHub) ------------------------------------
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"            # cached model outputs (captions, raw classification answers, stats)
FIG = RESULTS / "figures"
EXAMPLES = RESULTS / "examples"

# optional inputs you can drop into workdir/dataset/ (see README)
TARGETS_JSON = DATA / "targets.json"   # {"0": "attacker's target caption", ...}
LABELS_JSON = DATA / "labels.json"     # {"0": 207, ...} ImageNet-1k class ids (or class names)

# ---- dataset -----------------------------------------------------------------
N_IMAGES = 100
SEED = 0
CLEAN_URL = ("https://raw.githubusercontent.com/VILA-Lab/M-Attack/main/"
             "resources/images/bigscale/nips17")
ADV_URL = ("https://media.githubusercontent.com/media/Chen-Jin-Han/"
           "VLM-Adversarial-Attack-Benchmark/main/Adversarial-Images")
GD_REPO = "https://github.com/openai/guided-diffusion"
CKPT_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"

ATTACKS = {                       # key -> folder name in the benchmark repo
    "mattack":   "M-Attack-v2-Eps16",
    "foa":       "FOA-Attack-Eps16",
    "attackvlm": "AttackVLM-Eps16",
    "mpc":       "MPC-Attack-Eps16",
}
ATTACK_LABELS = {"clean": "Clean", "mattack": "M-Attack-v2", "foa": "FOA-Attack",
                 "attackvlm": "AttackVLM", "mpc": "MPC-Attack"}
ACTIVE_ATTACKS = list(ATTACKS)

# ---- DiffCAP -----------------------------------------------------------------
DIFFCAP = dict(
    tau=None,            # None -> calibrate (Alg. 2). Paper default 0.96 (safe range 0.95-0.97)
    dt=0.01,             # forward step size
    min_t=0.04,          # minimum diffusion depth for eps > 4/255
    max_t=1.0,
    noise_mode="incremental",   # or "fresh"
)

# ---- VLMs --------------------------------------------------------------------
VLMS = {
    "Qwen2.5-VL-3B":   dict(family="qwen",  id="Qwen/Qwen2.5-VL-3B-Instruct",        four_bit=False),
    "InstructBLIP-7B": dict(family="iblip", id="Salesforce/instructblip-vicuna-7b",  four_bit=True),
    "SmolVLM2-2.2B":   dict(family="smol",  id="HuggingFaceTB/SmolVLM2-2.2B-Instruct", four_bit=False),
}
ACTIVE_VLMS = list(VLMS)
VLM_GPU_MODE = "parallel"   # "parallel": one copy per GPU (faster) | "shard": one copy split over GPUs

# ---- tasks -------------------------------------------------------------------
PROMPTS = {
    "caption":  "Write a short caption describing this image.",
    "classify": "What is the main object in this image? Answer with only the object name (one to three words).",
}
MAX_NEW_TOKENS = {"caption": 60, "classify": 12}


# ---- helpers -----------------------------------------------------------------
def attack_dir(key):
    return ADV_ROOT / ATTACKS[key]


def sets():
    """name -> folder holding 0.png ... N-1.png"""
    return {"clean": CLEAN_DIR, **{k: attack_dir(k) for k in ACTIVE_ATTACKS}}


def cond_dirs():
    """every evaluation condition (undefended + DiffCAP-defended) -> image folder"""
    out = {}
    for name, d in sets().items():
        out[name] = d
        out[name + "+DiffCAP"] = PUR / name
    return out

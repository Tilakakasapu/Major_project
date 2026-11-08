"""Step 3: run every VLM on every condition for two tasks: captioning and classification (raw answers cached)."""
import contextlib
import gc
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from tqdm.auto import tqdm

import config as C
from src.utils import load_json, log, save_json


def _gpu_ctx(g):
    return torch.cuda.device(g) if g is not None else contextlib.nullcontext()


def _device_kw(g, ngpu):
    if g is None:   # shard one copy over all GPUs
        return dict(device_map="auto", max_memory={k: "13GiB" for k in range(ngpu)})
    return dict(device_map={"": g})


def _half():   # T4/P100 have no native bf16
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def _bnb():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)


# Each loader returns (model, gen) where gen(img, task) -> str, task in {"caption", "classify"}.
def _load_qwen(cfg, g, ngpu):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    kw = _device_kw(g, ngpu)
    kw.update(quantization_config=_bnb()) if cfg["four_bit"] else kw.update(torch_dtype=_half())
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg["id"], **kw).eval()
    proc = AutoProcessor.from_pretrained(cfg["id"], min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)

    @torch.no_grad()
    def gen(img, task):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": C.PROMPTS[task]}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=C.MAX_NEW_TOKENS[task], do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    return model, gen


def _load_iblip(cfg, g, ngpu):
    from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
    kw = _device_kw(g, ngpu)
    kw.update(quantization_config=_bnb()) if cfg["four_bit"] else kw.update(torch_dtype=torch.float16)
    model = InstructBlipForConditionalGeneration.from_pretrained(cfg["id"], **kw).eval()
    proc = InstructBlipProcessor.from_pretrained(cfg["id"])

    @torch.no_grad()
    def gen(img, task):
        prompt = C.PROMPTS[task]
        inp = proc(images=img, text=prompt, return_tensors="pt").to(model.device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        out = model.generate(**inp, do_sample=False, num_beams=5, max_new_tokens=C.MAX_NEW_TOKENS[task],
                             min_length=1, repetition_penalty=1.5 if task == "caption" else 1.0, length_penalty=1.0)
        txt = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        return txt[len(prompt):].strip() if txt.startswith(prompt) else txt
    return model, gen


def _load_smol(cfg, g, ngpu):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    kw = _device_kw(g, ngpu)
    if cfg["four_bit"]:
        kw.update(quantization_config=_bnb())
    else:   # fp16 is unreliable for SmolVLM2 on T4 -> fp32 there (2.2B fits in 16 GB); bf16 on Ampere+
        kw.update(torch_dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32)
    model = AutoModelForImageTextToText.from_pretrained(cfg["id"], attn_implementation="eager", **kw).eval()
    proc = AutoProcessor.from_pretrained(cfg["id"], do_image_splitting=False)

    @torch.no_grad()
    def gen(img, task):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": C.PROMPTS[task]}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=text, images=[img], return_tensors="pt").to(model.device, dtype=model.dtype)
        out = model.generate(**inp, max_new_tokens=C.MAX_NEW_TOKENS[task], do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    return model, gen


LOADERS = {"qwen": _load_qwen, "iblip": _load_iblip, "smol": _load_smol}


def _run_task(task, vlm, fns, gpus, store, path, cond_dirs, n):
    for cond, d in cond_dirs.items():
        store.setdefault(cond, {})
        todo = [i for i in range(n) if str(i) not in store[cond]]
        if todo:
            pbar = tqdm(total=len(todo), desc=f"{vlm} | {task:8s} | {cond}")

            def work(k, d=d, todo=todo, pbar=pbar):
                out = {}
                with _gpu_ctx(gpus[k]):
                    for i in todo[k::len(fns)]:
                        out[str(i)] = fns[k](Image.open(d / f"{i}.png").convert("RGB"), task)
                        pbar.update(1)
                return out
            with ThreadPoolExecutor(len(fns)) as ex:
                for out in ex.map(work, range(len(fns))):
                    store[cond].update(out)
            pbar.close()
        save_json(store, path)


def run_vlm(name, n):
    cfg = C.VLMS[name]
    cond_dirs = C.cond_dirs()
    cap_path, cls_path = C.RAW / f"{name}_captions.json", C.RAW / f"{name}_classification_raw.json"
    caps, cls = load_json(cap_path, {}), load_json(cls_path, {})
    pending = sum(1 for c in cond_dirs for i in range(n)
                  if str(i) not in caps.get(c, {}) or str(i) not in cls.get(c, {}))
    if pending == 0:
        log(f"{name}: all {len(cond_dirs)} conditions x {n} images cached - skipping")
        return
    ngpu = torch.cuda.device_count()
    assert ngpu >= 1, "No GPU visible."
    gpus = list(range(ngpu)) if C.VLM_GPU_MODE == "parallel" else [None]
    log(f"{name}: loading '{cfg['id']}' ({'4-bit' if cfg['four_bit'] else 'full precision'}) on {len(gpus)} worker(s)")
    models, fns = [], []
    for g in gpus:
        with _gpu_ctx(g):
            m, f = LOADERS[cfg["family"]](cfg, g, ngpu)
        models.append(m); fns.append(f)
    log(f"{name}: sanity check on image 0")
    with _gpu_ctx(gpus[0]):
        for k in ("clean", *C.ACTIVE_ATTACKS):
            img = Image.open(C.sets()[k] / "0.png").convert("RGB")
            log(f"   [{k:9s}] caption -> {fns[0](img, 'caption')!r} | classify -> {fns[0](img, 'classify')!r}")
    log(f"{name}: running {len(cond_dirs)} conditions x {n} images x 2 tasks")
    _run_task("caption", name, fns, gpus, caps, cap_path, cond_dirs, n)
    _run_task("classify", name, fns, gpus, cls, cls_path, cond_dirs, n)
    del models, fns
    gc.collect(); torch.cuda.empty_cache()
    log(f"{name}: done, GPU memory released")

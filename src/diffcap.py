"""Step 2: DiffCAP purification (adaptive-depth diffusion purification), ported from the working notebook."""
import gc
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import ks_2samp
from tqdm.auto import tqdm

import config as C
from src.utils import load_json, log, save_json

RES = 256


# ----------------------------- helpers -----------------------------------------
def pil_to_tensor(img, device):
    img = img.convert("RGB").resize((RES, RES), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return (x * 2 - 1).to(device)


def tensor_to_pil(x, size=None):
    a = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(a)
    return img.resize(size, Image.BICUBIC) if size else img


def psnr(a, b):  # a, b in [-1, 1]
    mse = (((a - b) / 2) ** 2).mean().item()
    return float("inf") if mse == 0 else -10 * np.log10(mse)


def load_guided_diffusion(device):
    sys.path.insert(0, str(C.GD_DIR))
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    cfg = model_and_diffusion_defaults()
    cfg.update(dict(attention_resolutions="32,16,8", class_cond=False, diffusion_steps=1000,
                    image_size=RES, learn_sigma=True, noise_schedule="linear", num_channels=256,
                    num_head_channels=64, num_res_blocks=2, resblock_updown=True,
                    use_fp16=True, use_scale_shift_norm=True, timestep_respacing=""))
    model, diffusion = create_model_and_diffusion(**cfg)
    model.load_state_dict(torch.load(C.CKPT, map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    model.convert_to_fp16()
    return model, diffusion


class ClipEnc:
    """CLIP ViT-B/32 = the paper's default feature extractor for the stopping rule."""
    def __init__(self, device):
        import open_clip
        self.device = device
        self.m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
        self.m.eval().requires_grad_(False)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def image(self, x):  # x: Bx3xHxW in [-1, 1]
        x = F.interpolate((x.float() + 1) / 2, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
        return F.normalize(self.m.encode_image((x - self.mean) / self.std).float(), dim=-1)


class DiffCAP:
    def __init__(self, model, diffusion, enc, tau, dt=None, min_t=None, max_t=None, noise_mode=None):
        P = C.DIFFCAP
        dt, min_t = dt or P["dt"], P["min_t"] if min_t is None else min_t
        max_t, noise_mode = max_t or P["max_t"], noise_mode or P["noise_mode"]
        self.model, self.diffusion, self.enc, self.tau, self.noise_mode = model, diffusion, enc, tau, noise_mode
        self.device = enc.device
        self.N = diffusion.num_timesteps
        # ac[n] = alpha_bar after n steps (ac[0]=1)
        self.ac = np.concatenate([[1.0], np.asarray(diffusion.alphas_cumprod, dtype=np.float64)])
        self.dstep = max(1, int(round(dt * self.N)))
        self.n_min, self.n_max = int(round(min_t * self.N)), int(round(max_t * self.N))

    def _noise(self, x_cur, x0, n_prev, n_new):
        if self.noise_mode == "fresh":
            a = self.ac[n_new]
            return float(np.sqrt(a)) * x0 + float(np.sqrt(1 - a)) * torch.randn_like(x0)
        r = self.ac[n_new] / self.ac[n_prev]
        return float(np.sqrt(r)) * x_cur + float(np.sqrt(1 - r)) * torch.randn_like(x_cur)

    @torch.no_grad()
    def forward_until_stable(self, x0):
        e_prev, x, n = self.enc.image(x0), x0, 0
        while n < self.n_max:
            n_new = min(n + self.dstep, self.n_max)
            x = self._noise(x, x0, n, n_new)
            e_cur = self.enc.image(x)
            sim = (e_cur * e_prev).sum().item()
            n = n_new
            if n >= self.n_min and sim >= self.tau:
                break
            e_prev = e_cur
        return x, n

    @torch.no_grad()
    def reverse(self, x, n):
        for i in range(n - 1, -1, -1):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            x = self.diffusion.p_sample(self.model, x, t, clip_denoised=True)["sample"]
        return x

    @torch.no_grad()
    def purify_pil(self, img):
        x0 = pil_to_tensor(img, self.device)
        x_stable, n = self.forward_until_stable(x0)
        x_pur = self.reverse(x_stable, n)
        return tensor_to_pil(x_pur, img.size), x_pur, n / self.N


@torch.no_grad()
def calibrate_tau(pairs, enc, ac, dstep, alpha=0.05, max_steps=100):
    """Alg. 2: noise clean & adversarial images until their consecutive-step similarity distributions are
    indistinguishable (two-sample KS test, p > alpha); tau = mean clean similarity at that point."""
    Xc = torch.cat([pil_to_tensor(c, enc.device) for c, _ in pairs])
    Xa = torch.cat([pil_to_tensor(a, enc.device) for _, a in pairs])
    emb = lambda X: torch.cat([enc.image(X[i:i + 50]) for i in range(0, len(X), 50)])  # noqa: E731
    ec, ea, n = emb(Xc), emb(Xa), 0
    s_c = np.zeros(1)
    for _ in range(max_steps):
        n_new = n + dstep
        if n_new >= len(ac):
            break
        r = float(np.sqrt(ac[n_new] / ac[n])), float(np.sqrt(1 - ac[n_new] / ac[n]))
        Xc = r[0] * Xc + r[1] * torch.randn_like(Xc)
        Xa = r[0] * Xa + r[1] * torch.randn_like(Xa)
        nc, na = emb(Xc), emb(Xa)
        s_c, s_a = (nc * ec).sum(-1).cpu().numpy(), (na * ea).sum(-1).cpu().numpy()
        p = ks_2samp(s_c, s_a).pvalue
        n = n_new
        log(f"   calibration t={n / (len(ac) - 1):.2f}  sim clean={s_c.mean():.4f}  adv={s_a.mean():.4f}  KS p={p:.3f}")
        if p > alpha:
            return float(s_c.mean())
        ec, ea = nc, na
    log("   [warn] distributions never matched; using last clean mean")
    return float(s_c.mean())


def _resolve_tau(override, model, diffusion, enc, n):
    path = C.RAW / "tau.json"
    tau = override if override is not None else C.DIFFCAP["tau"]
    if tau is not None:
        save_json({"tau": tau, "source": "fixed", "attacks": sorted(C.ACTIVE_ATTACKS)}, path)
        return tau
    cached = load_json(path, {})
    if cached.get("source") == "calibrated" and cached.get("attacks") == sorted(C.ACTIVE_ATTACKS):
        log(f"   reusing calibrated tau = {cached['tau']:.4f}")
        return cached["tau"]
    log("   calibrating tau (Alg. 2) on clean/adversarial pairs ...")
    rng = np.random.default_rng(C.SEED)
    per = min(n, max(1, 100 // len(C.ACTIVE_ATTACKS)))
    pairs = []
    for key in C.ACTIVE_ATTACKS:
        for i in rng.choice(n, size=per, replace=False):
            pairs.append((Image.open(C.CLEAN_DIR / f"{i}.png"), Image.open(C.attack_dir(key) / f"{i}.png")))
    probe = DiffCAP(model, diffusion, enc, 0.96)
    tau = calibrate_tau(pairs, enc, probe.ac, probe.dstep)
    save_json({"tau": tau, "source": "calibrated", "attacks": sorted(C.ACTIVE_ATTACKS)}, path)
    return tau


# ----------------------------- main entry --------------------------------------
def run_purification(n, tau_override=None):
    sets = C.sets()
    stats_path = C.RAW / "purification_stats.json"
    stats = load_json(stats_path, {})
    for name in sets:
        (C.PUR / name).mkdir(parents=True, exist_ok=True)
    tasks = [(nm, i) for nm in sets for i in range(n)
             if not ((C.PUR / nm / f"{i}.png").exists() and f"{nm}/{i}" in stats)]
    log(f"{len(tasks)} of {len(sets) * n} images still need purification ({len(sets)} sets x {n})")
    if not tasks:
        log("everything already purified (cached) - skipping model loading")
        return stats

    ngpu = torch.cuda.device_count()
    assert ngpu >= 1, "No GPU visible. On Kaggle: Settings -> Accelerator -> GPU T4 x2"
    log(f"GPUs: {[torch.cuda.get_device_name(g) for g in range(ngpu)]}")
    models, diffusions, encs = [], [], []
    for g in range(ngpu):
        log(f"   loading diffusion model + CLIP ViT-B/32 on cuda:{g}")
        m, d = load_guided_diffusion(f"cuda:{g}")
        models.append(m); diffusions.append(d); encs.append(ClipEnc(f"cuda:{g}"))
    tau = _resolve_tau(tau_override, models[0], diffusions[0], encs[0], n)
    log(f"using tau = {tau:.4f} (paper default 0.96)")
    dcs = [DiffCAP(models[g], diffusions[g], encs[g], tau) for g in range(ngpu)]

    lock, pbar = threading.Lock(), tqdm(total=len(tasks), desc=f"purify ({ngpu} GPU)")

    def worker(g):
        dc, enc, dev = dcs[g], encs[g], dcs[g].device
        with torch.cuda.device(dev):
            for name, i in tasks[g::ngpu]:
                img = Image.open(sets[name] / f"{i}.png").convert("RGB")
                pil_p, xp, t = dc.purify_pil(img)
                pil_p.save(C.PUR / name / f"{i}.png")
                x_in = pil_to_tensor(img, dev)
                rec = {"t": t, "psnr_to_input": psnr(x_in, xp),
                       "cos_to_input": (enc.image(x_in) * enc.image(xp)).sum().item()}
                if name != "clean":
                    x_c = pil_to_tensor(Image.open(C.CLEAN_DIR / f"{i}.png"), dev)
                    ec = enc.image(x_c)
                    rec.update(psnr_clean_vs_adv=psnr(x_c, x_in), psnr_clean_vs_pur=psnr(x_c, xp),
                               cos_clean_vs_adv=(ec * enc.image(x_in)).sum().item(),
                               cos_clean_vs_pur=(ec * enc.image(xp)).sum().item())
                with lock:
                    stats[f"{name}/{i}"] = rec
                    if len(stats) % 20 == 0:
                        save_json(stats, stats_path)
                pbar.update(1)

    with ThreadPoolExecutor(ngpu) as ex:
        list(ex.map(worker, range(ngpu)))
    pbar.close()
    save_json(stats, stats_path)
    del dcs, models, diffusions, encs
    gc.collect(); torch.cuda.empty_cache()
    log("purification complete; GPU memory released")
    return stats

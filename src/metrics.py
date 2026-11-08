"""Step 4: metrics.

Captioning  (reference = the ORIGINAL clean image / the caption the model gives for it)
  CLIP-S        cosine(clean image, caption) x100, CLIP ViT-L/14. Is the caption still faithful to what is truly in the image?
  Consistency   cosine(caption of clean image, caption of this condition). Did the model's *output* stay the same?
  CIDEr, BLEU4  lexical overlap with the clean-image caption (CLIP-independent sanity check, x100).
  WinRate       share of images where the defended caption scores higher CLIP-S than the undefended one (ties = 0.5).
  TargetSim     (optional, needs targets.json) similarity to the attacker's target caption. Lower = attack failed.
Classification (free-form object name from the VLM, mapped to the closest ImageNet-1k class with CLIP text embeddings)
  Top1 / Top5   accuracy (%) against the reference label.
  Consistency   % of images whose predicted class equals the model's own prediction on the clean image.
"""
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import config as C
from src.utils import load_json, log, save_json

_NORM = lambda s: " ".join(re.sub(r"[.,]", " ", s.lower()).split())  # noqa: E731


def _clean_answer(a):
    a = a.strip().split("\n")[0].lower().rstrip(".")
    a = re.sub(r"^(the main object( in this image)? is|this is|it is|there is|a|an|the)\s+", "", a)
    return " ".join(a.split()[:8]) or "object"


class Evaluator:
    def __init__(self, device):
        import open_clip
        from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
        self.device, self.classnames = device, list(IMAGENET_CLASSNAMES)
        self.m, _, self.pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
        self.m.eval().requires_grad_(False)
        self.tok = open_clip.get_tokenizer("ViT-L-14")
        self.class_emb = self.txt_emb([f"a photo of a {c}." for c in self.classnames])

    @torch.no_grad()
    def txt_emb(self, strings):
        out = [F.normalize(self.m.encode_text(self.tok(strings[i:i + 64]).to(self.device)).float(), dim=-1)
               for i in range(0, len(strings), 64)]
        return torch.cat(out)

    @torch.no_grad()
    def img_emb(self, paths):
        out = []
        for i in range(0, len(paths), 32):
            x = torch.stack([self.pre(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]).to(self.device)
            out.append(F.normalize(self.m.encode_image(x).float(), dim=-1))
        return torch.cat(out)

    def predict(self, answers):
        E = self.txt_emb([f"a photo of a {_clean_answer(a)}." for a in answers])
        top5 = (E @ self.class_emb.T).topk(5, dim=1).indices.cpu().numpy()
        return top5[:, 0], top5

    def ground_truth(self, E_img, n):
        if C.LABELS_JSON.exists():
            raw, names = load_json(C.LABELS_JSON, {}), [c.lower() for c in self.classnames]
            gt = [int(raw[str(i)]) if str(raw[str(i)]).isdigit() else names.index(str(raw[str(i)]).lower())
                  for i in range(n)]
            return np.array(gt), "labels.json (true ImageNet labels you provided)"
        gt = (E_img @ self.class_emb.T).argmax(1).cpu().numpy()
        return gt, "pseudo-label: CLIP ViT-L/14 zero-shot on the clean image"


def _lexical(refs, hyps, n):
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        gts = {i: [refs[i]] for i in range(n)}
        res = {i: [hyps[i]] for i in range(n)}
        return Cider().compute_score(gts, res)[0] * 100, Bleu(4).compute_score(gts, res, verbose=0)[0][3] * 100
    except Exception as e:  # noqa: BLE001
        log(f"   [warn] lexical metrics unavailable ({e})")
        return np.nan, np.nan


def evaluate(n, vlms, attacks):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"loading CLIP ViT-L/14 evaluator on {dev}")
    ev = Evaluator(dev)
    idx = list(range(n))
    names = ["clean", *attacks]
    conds = [c for nm in names for c in (nm, nm + "+DiffCAP")]
    E_img = ev.img_emb([C.CLEAN_DIR / f"{i}.png" for i in idx])
    gt, gt_src = ev.ground_truth(E_img, n)
    log(f"classification reference labels: {gt_src}")
    targets = load_json(C.TARGETS_JSON, None) if C.TARGETS_JSON.exists() else None
    cap_rows, cap_img, cls_rows, cls_img = [], [], [], []

    def split(cond):
        return cond.replace("+DiffCAP", ""), cond.endswith("+DiffCAP")

    for vlm in vlms:
        log(f"metrics for {vlm}")
        caps = load_json(C.RAW / f"{vlm}_captions.json")
        raw = load_json(C.RAW / f"{vlm}_classification_raw.json")
        E_ref = ev.txt_emb([caps["clean"][str(i)] for i in idx])
        ref_norm = [_NORM(caps["clean"][str(i)]) for i in idx]
        E_tg = ev.txt_emb([targets[str(i)] for i in idx]) if targets else None

        # ---------------- captioning ----------------
        clips_by, rows = {}, {}
        for cond in conds:
            texts = [caps[cond][str(i)] for i in idx]
            E_t = ev.txt_emb(texts)
            clips = (E_img * E_t).sum(-1).cpu().numpy() * 100
            cons = (E_ref * E_t).sum(-1).cpu().numpy()
            clips_by[cond] = clips
            atk, dfn = split(cond)
            cider, bleu = (np.nan, np.nan) if cond == "clean" else _lexical(ref_norm, [_NORM(t) for t in texts], n)
            r = {"vlm": vlm, "condition": cond, "attack": atk, "defended": dfn, "CLIP-S": clips.mean(),
                 "Consistency": np.nan if cond == "clean" else cons.mean(), "CIDEr": cider, "BLEU4": bleu}
            if E_tg is not None:
                r["TargetSim"] = (E_tg * E_t).sum(-1).mean().item()
            rows[cond] = r
            cap_rows.append(r)
            for i in idx:
                cap_img.append(dict(vlm=vlm, condition=cond, attack=atk, defended=dfn, image=i,
                                    clip_s=clips[i], consistency=cons[i], caption=texts[i]))
        for nm in names:    # win rate of defended vs undefended
            a, b = clips_by[nm + "+DiffCAP"], clips_by[nm]
            rows[nm + "+DiffCAP"]["WinRate"] = float(((a > b) + 0.5 * (a == b)).mean() * 100)

        # ---------------- classification ----------------
        preds = {c: ev.predict([raw[c][str(i)] for i in idx]) for c in conds}
        clean_pred = preds["clean"][0]
        for cond in conds:
            top1, top5 = preds[cond]
            ok1, ok5 = top1 == gt, np.array([gt[i] in top5[i] for i in idx])
            same = top1 == clean_pred
            atk, dfn = split(cond)
            cls_rows.append({"vlm": vlm, "condition": cond, "attack": atk, "defended": dfn,
                             "Top1": ok1.mean() * 100, "Top5": ok5.mean() * 100,
                             "Consistency": np.nan if cond == "clean" else same.mean() * 100})
            for i in idx:
                cls_img.append(dict(vlm=vlm, condition=cond, attack=atk, defended=dfn, image=i,
                                    answer=raw[cond][str(i)], pred_name=ev.classnames[top1[i]],
                                    gt_name=ev.classnames[gt[i]], correct1=int(ok1[i]), correct5=int(ok5[i]),
                                    consistent=int(same[i])))

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"metrics_captioning": cap_rows, "per_image_captioning": cap_img,
           "metrics_classification": cls_rows, "per_image_classification": cls_img}
    for name, rows in out.items():
        pd.DataFrame(rows).to_csv(C.RESULTS / f"{name}.csv", index=False)
        log(f"   saved results/{name}.csv ({len(rows)} rows)")
    save_json({"n_images": n, "vlms": list(vlms), "attacks": list(attacks), "classification_reference": gt_src,
               "has_targets": targets is not None}, C.RESULTS / "meta.json")
    pd.set_option("display.width", 200)
    print("\nCaptioning summary:\n", pd.DataFrame(cap_rows).drop(columns=["attack", "defended"]).round(2).to_string(index=False))
    print("\nClassification summary:\n", pd.DataFrame(cls_rows).drop(columns=["attack", "defended"]).round(2).to_string(index=False))

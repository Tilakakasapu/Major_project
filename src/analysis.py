"""Step 5: statistics, written findings, graphs, and the data file for the web UI. Needs no GPU."""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import binomtest, wilcoxon  # noqa: E402

import config as C  # noqa: E402
from src.utils import load_json, log, save_json  # noqa: E402

TASKS = {
    "caption": dict(label="Captioning", metric="CLIP-S", col="clip_s", scale=1, min_drop=0.5,
                    cons_col="consistency", cons_scale=1, unit="CLIP-S points"),
    "classification": dict(label="Classification", metric="Top-1 accuracy", col="correct1", scale=100, min_drop=3.0,
                           cons_col="consistent", cons_scale=100, unit="accuracy points"),
}
COL_UND, COL_DEF, COL_CLEAN = "#d1495b", "#1f5fa8", "#8a94a0"
lab = lambda k: C.ATTACK_LABELS.get(k, k)  # noqa: E731


def _arr(img, vlm, cond, col):
    d = img[(img.vlm == vlm) & (img.condition == cond)].sort_values("image")
    return d[col].to_numpy(float)


def _p(task, adv, dfn):
    try:
        if task == "caption":
            return float(wilcoxon(dfn, adv).pvalue)
        b, c = int(((adv == 0) & (dfn == 1)).sum()), int(((adv == 1) & (dfn == 0)).sum())
        return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    except ValueError:      # all differences are zero
        return 1.0


def _verdict(drop, rec, min_drop):
    if drop < min_drop:
        return "attack-weak"
    return "strong" if rec >= .75 else "moderate" if rec >= .40 else "limited" if rec >= .10 else "none"


VERDICT_TXT = {"attack-weak": "the attack barely hurt this model, so there was little to recover",
               "strong": "strong recovery", "moderate": "moderate recovery",
               "limited": "limited recovery", "none": "no meaningful recovery"}


def build_effect(imgs, vlms, attacks):
    rows = []
    for task, s in TASKS.items():
        img = imgs[task]
        for v in vlms:
            clean = _arr(img, v, "clean", s["col"]).mean() * s["scale"]
            cleand = _arr(img, v, "clean+DiffCAP", s["col"]).mean() * s["scale"]
            for a in attacks:
                x, d = _arr(img, v, a, s["col"]), _arr(img, v, a + "+DiffCAP", s["col"])
                adv, dfn = x.mean() * s["scale"], d.mean() * s["scale"]
                drop, gain = clean - adv, dfn - adv
                rec = gain / drop if drop >= s["min_drop"] else np.nan
                ca = _arr(img, v, a, s["cons_col"]).mean() * s["cons_scale"]
                cd = _arr(img, v, a + "+DiffCAP", s["cons_col"]).mean() * s["cons_scale"]
                p = _p(task, x, d)
                ver = _verdict(drop, rec, s["min_drop"])
                sig = "statistically significant, p=%.3g" % p if p < .05 else "not statistically significant, p=%.2g" % p
                if ver == "attack-weak":
                    body = (f"{s['metric']} fell only {drop:.1f} {s['unit']} under attack (clean {clean:.1f} → {adv:.1f}), "
                            f"so {lab(a)} is weak against this model; DiffCAP ends at {dfn:.1f}.")
                else:
                    body = (f"{s['metric']} drops {drop:.1f} {s['unit']} under attack (clean {clean:.1f} → {adv:.1f}). "
                            f"DiffCAP lifts it to {dfn:.1f}, recovering {rec * 100:.0f}% of the loss ({sig}): {VERDICT_TXT[ver]}. "
                            f"Agreement with the clean-image output moves {ca:.2f} → {cd:.2f}."
                            if task == "caption" else
                            f"{s['metric']} drops {drop:.1f} {s['unit']} under attack (clean {clean:.1f} → {adv:.1f}). "
                            f"DiffCAP lifts it to {dfn:.1f}, recovering {rec * 100:.0f}% of the loss ({sig}): {VERDICT_TXT[ver]}. "
                            f"Agreement with the clean-image prediction moves {ca:.0f}% → {cd:.0f}%.")
                rows.append(dict(task=task, vlm=v, attack=a, metric=s["metric"], clean=clean, adv=adv, defended=dfn,
                                 drop=drop, gain=gain, recovery=rec, p=p, verdict=ver, cons_adv=ca, cons_def=cd,
                                 clean_defended=cleand, text=f"{v} · {s['label']} · {lab(a)}: {body}"))
    return pd.DataFrame(rows)


def build_summaries(eff, vlms, attacks):
    S = {"overall": [], "per_vlm": {}, "per_task": {}, "per_attack": {}}
    f = lambda x: "n/a" if pd.isna(x) else f"{x * 100:.0f}%"  # noqa: E731
    for task, s in TASKS.items():
        e = eff[eff.task == task]
        eff_ok = e[e.verdict != "attack-weak"]
        sig = int(((e.p < .05) & (e.gain > 0)).sum())
        cu = pd.Series({v: g.clean.iloc[0] - g.clean_defended.iloc[0] for v, g in e.groupby("vlm")})
        best = eff_ok.loc[eff_ok.recovery.idxmax()] if len(eff_ok) else None
        worst = eff_ok.loc[eff_ok.recovery.idxmin()] if len(eff_ok) else None
        txt = [f"Mean recovery over all effective attack/model pairs: {f(eff_ok.recovery.mean())}; "
               f"{sig} of {len(e)} pairs show a significant improvement from DiffCAP.",
               f"Cost on clean images: {s['metric']} changes by {-cu.mean():+.2f} {s['unit']} on average after purification "
               f"(per model: " + ", ".join(f"{k} {-v:+.2f}" for k, v in cu.items()) + ")."]
        if best is not None:
            txt.append(f"Best case: {best.vlm} vs {lab(best.attack)} ({f(best.recovery)}). "
                       f"Hardest case: {worst.vlm} vs {lab(worst.attack)} ({f(worst.recovery)}).")
        S["per_task"][task] = txt
        S["overall"].append(f"{s['label']}: " + txt[0])
    for v in vlms:
        out = []
        for task, s in TASKS.items():
            e = eff[(eff.task == task) & (eff.vlm == v)]
            worst = e.loc[e["drop"].idxmax()]
            ok = e[e.verdict != "attack-weak"]
            r = f"mean recovery {f(ok.recovery.mean())}" if len(ok) else "no attack hurt it enough to measure recovery"
            out.append(f"{s['label']}: the most damaging attack is {lab(worst.attack)} (−{worst['drop']:.1f} {s['unit']}); "
                       f"DiffCAP achieves {r}, and the clean-image {s['metric']} goes {e.clean.iloc[0]:.1f} → {e.clean_defended.iloc[0]:.1f}.")
        S["per_vlm"][v] = out
    for a in attacks:
        out = []
        for task, s in TASKS.items():
            e = eff[(eff.task == task) & (eff.attack == a)]
            ok = e[e.verdict != "attack-weak"]
            out.append(f"{s['label']}: average damage −{e['drop'].mean():.1f} {s['unit']}; "
                       f"mean recovery {f(ok.recovery.mean()) if len(ok) else 'n/a'} "
                       f"(range {f(ok.recovery.min()) if len(ok) else 'n/a'} to {f(ok.recovery.max()) if len(ok) else 'n/a'} across models).")
        S["per_attack"][a] = out
    return S


# ------------------------------- figures ---------------------------------------
def _save(fig, name, title, figs):
    fig.tight_layout()
    fig.savefig(C.FIG / name, dpi=150)
    plt.close(fig)
    figs.append(dict(file=f"figures/{name}", title=title))


def fig_bars(df, metric, ylabel, name, title, figs, attacks):
    if metric not in df or df[metric].notna().sum() == 0:
        return
    vlms, names = list(df.vlm.unique()), ["clean", *attacks]
    fig, axes = plt.subplots(1, len(vlms), figsize=(5.2 * len(vlms), 4.2), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], vlms):
        d = df[df.vlm == v].set_index("condition")[metric]
        x, w = np.arange(len(names)), .38
        ax.bar(x - w / 2, [d.get(n, np.nan) for n in names], w, color=COL_UND, label="No defense")
        ax.bar(x + w / 2, [d.get(n + "+DiffCAP", np.nan) for n in names], w, color=COL_DEF, label="DiffCAP")
        ax.set_xticks(x); ax.set_xticklabels([lab(n) for n in names], rotation=28, ha="right"); ax.set_title(v)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel(ylabel); axes[0][0].legend(frameon=False)
    fig.suptitle(title)
    _save(fig, name, title, figs)


def fig_heat(mats, titles, name, title, figs, vlms, attacks, fmt="{:.0f}%", vmin=0, vmax=100):
    fig, axes = plt.subplots(1, len(mats), figsize=(5.4 * len(mats), 2.2 + .55 * len(vlms)), squeeze=False)
    for ax, M, t in zip(axes[0], mats, titles):
        im = ax.imshow(M, cmap="RdYlBu", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(attacks))); ax.set_xticklabels([lab(a) for a in attacks], rotation=20)
        ax.set_yticks(range(len(vlms))); ax.set_yticklabels(vlms); ax.set_title(t)
        for i in range(len(vlms)):
            for j in range(len(attacks)):
                val = M[i, j]
                ax.text(j, i, "n/a" if np.isnan(val) else fmt.format(val), ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=.04)
    fig.suptitle(title)
    _save(fig, name, title, figs)


def fig_delta(capi, vlms, attacks, figs):
    fig, axes = plt.subplots(1, len(vlms), figsize=(5.2 * len(vlms), 4), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], vlms):
        data = [_arr(capi, v, a + "+DiffCAP", "clip_s") - _arr(capi, v, a, "clip_s") for a in attacks]
        ax.boxplot(data, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#cfe0f3"), medianprops=dict(color=COL_DEF))
        ax.set_xticks(range(1, len(attacks) + 1)); ax.set_xticklabels([lab(a) for a in attacks], rotation=20)
        ax.axhline(0, color="#555", lw=.8, ls="--"); ax.set_title(v)
    axes[0][0].set_ylabel("per-image CLIP-S change (DiffCAP − attacked)")
    fig.suptitle("Per-image caption fidelity gain from DiffCAP (above 0 = defense helped)")
    _save(fig, "caption_gain_distribution.png", "Per-image caption gain distribution", figs)


def fig_purification(stats, attacks, figs):
    names = ["clean", *attacks]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for nm in names:
        axes[0].hist([v["t"] for k, v in stats.items() if k.split("/")[0] == nm], bins=12, alpha=.55, label=lab(nm))
    axes[0].set_xlabel("diffusion time t chosen by DiffCAP"); axes[0].set_ylabel("# images"); axes[0].legend(frameon=False)
    axes[0].set_title("Adaptive purification depth")
    for ax, (ka, kp), yl, t in ((axes[1], ("psnr_clean_vs_adv", "psnr_clean_vs_pur"), "PSNR vs clean (dB)", "Pixel fidelity"),
                                (axes[2], ("cos_clean_vs_adv", "cos_clean_vs_pur"), "CLIP cosine vs clean", "Feature fidelity")):
        m = lambda k, n: np.mean([v[k] for kk, v in stats.items() if kk.split("/")[0] == n and k in v])  # noqa: E731
        x, w = np.arange(len(attacks)), .38
        ax.bar(x - w / 2, [m(ka, a) for a in attacks], w, color=COL_UND, label="attacked")
        ax.bar(x + w / 2, [m(kp, a) for a in attacks], w, color=COL_DEF, label="purified")
        ax.set_xticks(x); ax.set_xticklabels([lab(a) for a in attacks], rotation=20); ax.set_ylabel(yl); ax.set_title(t)
        ax.legend(frameon=False)
    _save(fig, "purification_quality.png", "Purification depth and image fidelity", figs)


# ------------------------------- examples --------------------------------------
def build_examples(capi, clsi, vlms, attacks):
    path = C.RESULTS / "examples.json"
    if not (C.CLEAN_DIR / "0.png").exists():
        log("   dataset images not present - keeping existing examples.json")
        return load_json(path, [])
    C.EXAMPLES.mkdir(parents=True, exist_ok=True)
    rng, ex = np.random.default_rng(1), []
    n = int(capi.image.max()) + 1
    for a in attacks:
        gain = np.mean([_arr(capi, v, a + "+DiffCAP", "clip_s") - _arr(capi, v, a, "clip_s") for v in vlms], axis=0)
        picks = [("largest recovery", int(i)) for i in np.argsort(-gain)[:2]] + \
                [("random", int(i)) for i in rng.choice(n, 2, replace=False)]
        for kind, i in picks:
            paths = {}
            for tag, d in (("clean", C.CLEAN_DIR), ("adv", C.attack_dir(a)), ("pur", C.PUR / a)):
                fn = f"{a}_{i}_{tag}.jpg"
                Image.open(d / f"{i}.png").convert("RGB").resize((224, 224)).save(C.EXAMPLES / fn, quality=88)
                paths[tag] = f"examples/{fn}"
            rec = {"attack": a, "image": i, "kind": kind, "paths": paths, "vlms": {}}
            for v in vlms:
                g = lambda df, c, col: df[(df.vlm == v) & (df.condition == c) & (df.image == i)][col].iloc[0]  # noqa: E731
                rec["vlms"][v] = {
                    "caption": {t: g(capi, c, "caption") for t, c in (("clean", "clean"), ("adv", a), ("pur", a + "+DiffCAP"))},
                    "class": {t: g(clsi, c, "pred_name") for t, c in (("clean", "clean"), ("adv", a), ("pur", a + "+DiffCAP"))},
                    "truth": g(clsi, "clean", "gt_name")}
            ex.append(rec)
    save_json(ex, path)
    return ex


# ------------------------------- outputs ---------------------------------------
def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) or np.isinf(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _dump(obj):
    txt = json.dumps(obj, default=_jsonable)
    return txt.replace("-Infinity", "null").replace("Infinity", "null").replace("NaN", "null")


def _records(df):
    return json.loads(_dump(df.to_dict("records")))


def write_report(eff, S, meta, vlms, attacks):
    L = ["# DiffCAP evaluation report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} · {meta['n_images']} images · "
         f"models: {', '.join(vlms)} · attacks: {', '.join(lab(a) for a in attacks)}", "",
         "## Metrics and why they were chosen", "",
         "**Captioning.** *CLIP-S* (ViT-L/14 cosine between the clean image and the caption) measures whether the caption is still "
         "true to the picture; it is the primary metric. *Consistency* (text–text cosine to the clean-image caption) measures whether the "
         "model's output was restored, which also catches targeted attacks that steer the caption somewhere else. *CIDEr/BLEU-4* against the "
         "clean-image caption are lexical and independent of CLIP, a check that the conclusions do not depend on one embedding model. "
         "*Win rate* is the fraction of images where the defended caption beats the undefended one.", "",
         "**Classification.** The VLM names the main object; the answer is mapped to the nearest ImageNet-1k class with CLIP text embeddings. "
         f"*Top-1/Top-5 accuracy* use reference labels ({meta['classification_reference']}). *Consistency* is the share of images whose class "
         "matches the model's own prediction on the clean image, so it isolates the attack/defense effect from the model's baseline errors.", "",
         "**Recovery** = (defended − attacked) / (clean − attacked): the share of the attack-induced loss that DiffCAP wins back. It is reported only "
         "when the attack removed at least 0.5 CLIP-S points or 3 accuracy points. **p-values**: Wilcoxon signed-rank (captioning) and exact sign test "
         "(classification) on paired per-image outcomes, defended vs. attacked.", "", "## Overall", ""]
    L += [f"- {t}" for t in S["overall"]]
    L += ["", "## Per task", ""]
    for t, s in TASKS.items():
        L += [f"### {s['label']}"] + [f"- {x}" for x in S["per_task"][t]] + [""]
    L += ["## Per model", ""]
    for v in vlms:
        L += [f"### {v}"] + [f"- {x}" for x in S["per_vlm"][v]] + [""]
    L += ["## Per attack", ""]
    for a in attacks:
        L += [f"### {lab(a)}"] + [f"- {x}" for x in S["per_attack"][a]] + [""]
    L += ["## Every model × task × attack", ""]
    for t, s in TASKS.items():
        L += [f"### {s['label']}", ""] + [f"- {r.text}" for r in eff[eff.task == t].itertuples()] + [""]
    (C.RESULTS / "analysis_report.md").write_text("\n".join(L), encoding="utf-8")


def run():
    C.FIG.mkdir(parents=True, exist_ok=True)
    cap, cls = pd.read_csv(C.RESULTS / "metrics_captioning.csv"), pd.read_csv(C.RESULTS / "metrics_classification.csv")
    capi, clsi = pd.read_csv(C.RESULTS / "per_image_captioning.csv"), pd.read_csv(C.RESULTS / "per_image_classification.csv")
    meta = load_json(C.RESULTS / "meta.json")
    vlms = list(cap.vlm.unique())
    attacks = [a for a in cap.attack.unique() if a != "clean"]
    stats = load_json(C.RAW / "purification_stats.json")
    imgs = {"caption": capi, "classification": clsi}

    log("computing defense effect (recovery, significance) for every model x task x attack")
    eff = build_effect(imgs, vlms, attacks)
    eff.to_csv(C.RESULTS / "defense_effect.csv", index=False)
    S = build_summaries(eff, vlms, attacks)

    log("drawing graphs")
    figs = []
    fig_bars(cap, "CLIP-S", "CLIP-S (cos × 100) vs clean image", "caption_clip_s.png", "Captioning fidelity (CLIP-S)", figs, attacks)
    fig_bars(cap, "Consistency", "caption similarity to clean-image caption", "caption_consistency.png", "Caption consistency with clean output", figs, attacks)
    fig_bars(cap, "CIDEr", "CIDEr × 100 vs clean caption", "caption_cider.png", "Lexical agreement with clean caption (CIDEr)", figs, attacks)
    fig_bars(cls, "Top1", "Top-1 accuracy (%)", "cls_top1.png", "Classification accuracy (Top-1)", figs, attacks)
    fig_bars(cls, "Top5", "Top-5 accuracy (%)", "cls_top5.png", "Classification accuracy (Top-5)", figs, attacks)
    fig_bars(cls, "Consistency", "agreement with clean prediction (%)", "cls_consistency.png", "Classification consistency with clean prediction", figs, attacks)
    mat = lambda t: np.array([[eff[(eff.task == t) & (eff.vlm == v) & (eff.attack == a)].recovery.iloc[0] * 100  # noqa: E731
                               for a in attacks] for v in vlms], dtype=float)
    fig_heat([mat("caption"), mat("classification")], ["Captioning (CLIP-S)", "Classification (Top-1)"],
             "recovery_heatmap.png", "Share of attack damage recovered by DiffCAP", figs, vlms, attacks)
    wr = np.array([[cap[(cap.vlm == v) & (cap.condition == a + "+DiffCAP")].WinRate.iloc[0] for a in attacks] for v in vlms])
    fig_heat([wr], ["Caption win rate (%)"], "winrate_heatmap.png", "Images where DiffCAP caption beats attacked caption", figs, vlms, attacks, vmin=0, vmax=100)
    fig_delta(capi, vlms, attacks, figs)
    fig_purification(stats, attacks, figs)

    ex = build_examples(capi, clsi, vlms, attacks)
    pur = []
    for nm in ["clean", *attacks]:
        s = [v for k, v in stats.items() if k.split("/")[0] == nm]
        r = {"set": nm, "mean_t": np.mean([x["t"] for x in s]), "psnr_to_input": np.mean([x["psnr_to_input"] for x in s]),
             "cos_to_input": np.mean([x["cos_to_input"] for x in s])}
        if nm != "clean":
            for k in ("psnr_clean_vs_adv", "psnr_clean_vs_pur", "cos_clean_vs_adv", "cos_clean_vs_pur"):
                r[k] = np.mean([x[k] for x in s])
        pur.append(r)
    pd.DataFrame(pur).to_csv(C.RESULTS / "purification_summary.csv", index=False)

    tau = load_json(C.RAW / "tau.json", {}).get("tau")
    payload = dict(meta=dict(meta, tau=tau, attack_labels={a: lab(a) for a in ["clean", *attacks]},
                             generated=time.strftime("%Y-%m-%d %H:%M")),
                   caption_metrics=_records(cap), cls_metrics=_records(cls), effect=_records(eff.drop(columns=[])),
                   summaries=S, purification=_records(pd.DataFrame(pur)), examples=ex, figures=figs)
    (C.RESULTS / "results.js").write_text("window.RESULTS = " + _dump(payload) + ";",
                                          encoding="utf-8")
    write_report(eff, S, meta, vlms, attacks)
    log(f"wrote results/results.js, analysis_report.md, defense_effect.csv and {len(figs)} figures")
    print("\n".join(["", "KEY FINDINGS", *["  - " + t for t in S["overall"]]]))

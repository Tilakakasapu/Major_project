# DiffCAP evaluation pipeline

Evaluates the **DiffCAP** purification defense against 4 adversarial attacks on 3 vision-language models, for 2 tasks.

| | |
|---|---|
| Attacks | M-Attack-v2, FOA-Attack, AttackVLM, MPC-Attack (all ε=16) |
| Models | Qwen2.5-VL-3B, InstructBLIP-7B (4-bit), SmolVLM2-2.2B |
| Tasks | Image captioning, image classification (ImageNet-1k) |

```
diffcap_project/
├── main.py            # run this: download → purify → VLM inference → metrics → analysis
├── config.py          # every setting (attacks, models, prompts, DiffCAP parameters)
├── serve.py           # opens the dashboard locally
├── requirements.txt
├── src/
│   ├── data.py        # step 1  downloads images, guided-diffusion, checkpoint
│   ├── diffcap.py     # step 2  DiffCAP + tau calibration, multi-GPU purification
│   ├── vlms.py        # step 3  Qwen / InstructBLIP / SmolVLM2: captioning + classification
│   ├── metrics.py     # step 4  task-specific metrics
│   ├── analysis.py    # step 5  statistics, written findings, graphs, web data
│   └── utils.py
├── web/index.html     # dashboard (reads results/results.js)
├── results/           # generated – small files, commit these
└── workdir/           # generated – images, checkpoint (git-ignored)
```

## Your workflow

### 1. GitHub
Put this folder in a new repo and push it (`workdir/` is already git-ignored).

### 2. Kaggle (run)
New notebook → **Settings → Accelerator: GPU T4 ×2**, **Internet: On**. Then:

```python
!git clone https://github.com/<you>/<repo>.git
%cd <repo>
!python main.py --quick --install     # 8-image smoke test (~15 min). Check the printed output looks sane.
```
then the full run:
```python
!python main.py
```
Every step prints its progress (download, calibration of τ, purification bar, per-model/per-condition bars, metric tables, key findings).
Results are cached: if a session dies, run the same command again and it resumes.
For the long full run use **Save Version → Save & Run All**, so it keeps running after you close the tab.

Collect the output (small, ~few MB):
```python
!zip -qr results.zip results
```
Download `results.zip` from the notebook's **Output** tab (or `/kaggle/working/<repo>/results`).

### 3. GitHub (update)
Unzip over your local clone's `results/` folder, then `git add results && git commit -m "results" && git push`.

### 4. Local (view)
```bash
git clone https://github.com/<you>/<repo>.git && cd <repo>
python serve.py          # opens http://localhost:8000/web/index.html  (needs only Python, no GPU)
```
Double-clicking `web/index.html` also works. To regenerate graphs/analysis without a GPU: `python main.py --stages analysis`
(needs `pip install pandas matplotlib scipy pillow`).

## Useful options
```
python main.py --quick                      # 8 images
python main.py --vlms SmolVLM2-2.2B         # one model
python main.py --attacks foa mpc            # some attacks
python main.py --stages vlm metrics analysis
python main.py --tau 0.96                   # skip calibration, use the paper value
```
If SmolVLM2 or Qwen run out of memory on a T4, set `four_bit=True` for that model in `config.py`.

## How the tasks are scored

**Captioning** – prompt: *“Write a short caption describing this image.”*
- **CLIP-S** (primary): cosine between the *clean image* and the caption (CLIP ViT-L/14, ×100). Still true to the picture?
- **Consistency**: text-embedding similarity to the caption the model gives for the clean image. Was the model’s *output* restored (also exposes targeted hijacking)?
- **CIDEr / BLEU-4** vs the clean-image caption: CLIP-independent lexical check.
- **Win rate**: share of images where the defended caption has higher CLIP-S than the attacked one.
- *Optional*: put `targets.json` (`{"0": "attacker’s target caption"}`) in `workdir/dataset/` to also get **TargetSim**.

**Classification** – prompt: *“What is the main object in this image? Answer with only the object name.”* The free-form answer is mapped to the closest ImageNet-1k class using CLIP text embeddings.
- **Top-1 / Top-5 accuracy** against reference labels.
- **Consistency**: % of images whose class equals the model’s own clean-image prediction (separates attack effect from the model’s normal mistakes).
- Reference labels are CLIP ViT-L/14 zero-shot labels of the clean image unless you provide true labels as `workdir/dataset/labels.json` (`{"0": 207, ...}`, ImageNet ids or names). The dataset’s true labels are the better choice if you have them.

**Defense effect** (both tasks): **Recovery** = (defended − attacked) / (clean − attacked), the share of attack damage won back; shown only when the attack cost ≥0.5 CLIP-S points / ≥3 accuracy points. Significance: Wilcoxon signed-rank (captioning) and exact sign test (classification) on paired per-image results. The clean-vs-clean+DiffCAP comparison reports what the defense costs on benign images.

## Outputs (`results/`)
`analysis_report.md` (full written analysis per model × task × attack), `defense_effect.csv`, `metrics_*.csv`, `per_image_*.csv`, `purification_summary.csv`, `figures/*.png`, `examples/`, `raw/` (all model outputs), `results.js` (feeds the dashboard).

## Caveats worth stating in a write-up
- CLIP ViT-L/14 is also the evaluator, and several attacks optimise against CLIP-family models. Scoring against the *clean* image and the lexical CIDEr/BLEU check reduce, but do not remove, this bias.
- 100 images per set gives noisy estimates; read the p-values, not just the means.

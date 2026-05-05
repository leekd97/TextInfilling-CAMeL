# CAMeL Text Infilling CBS

This repository measures CBS on CAMeL text infilling prompts by comparing Arab and Western entities within each entity type.

The expected layout is:

```text
TextInfilling/
  dataset/camel/
  ours/
    configs/
    src/
    results/text_infilling/
```

## Setup

Install the core dependencies:

```bash
cd /home/leekd97/experiments/TextInfilling
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` were taken from the local `cocoa` conda environment (`Python 3.11.14`, `torch 2.4.1+cu121`). Install a PyTorch build that matches your CUDA environment if the command above does not fit your machine.

Add your Hugging Face token:

```bash
cp ours/configs/user_configs.example.yaml ours/configs/user_configs.yaml
```

Then edit `ours/configs/user_configs.yaml`:

```yaml
hf_config:
  access_token: "hf_your_token_here"
```

Do not commit `ours/configs/user_configs.yaml`; it is ignored because it contains a private token.

## Run Everything

The shell script runs text infilling, CBS analysis, and per-model figures.

```bash
cd /home/leekd97/experiments/TextInfilling/ours/src
PROMPT_CULTURES="co ag" ./text_infilling.sh
```

Current defaults:

```bash
MODELS="LLAMA-3.1-8B Trillion-7B-preview LLAMA-3.1-8B-Instruct"
GPU_IDS="2 2 3"
PROMPT_N=50
ENTITY_N=50
BATCH_SIZE=32
LOAD_FLAGS="--load_in_4bit"
```

This maps the first two models to GPU 2 and the instruct model to GPU 3. Override any value inline:

```bash
GPU_IDS="2 3" BATCH_SIZE=16 PROMPT_CULTURES="co ag" ./text_infilling.sh
```

## Run Steps Manually

### 1. Text Infilling

```bash
cd /home/leekd97/experiments/TextInfilling/ours/src
CUDA_VISIBLE_DEVICES=2 python3 text_infilling.py \
  --model_name LLAMA-3.1-8B \
  --prompt_culture co \
  --language ar \
  --prompt_sample_count 50 \
  --entity_sample_count 50 \
  --batch_size 32 \
  --check_prev \
  --load_in_4bit
```

Outputs are saved under:

```text
ours/results/text_infilling/{MODEL}_{co|ag}_ar_50_50.json
```

### 2. Analysis

```bash
cd /home/leekd97/experiments/TextInfilling/ours/results/text_infilling
python3 evaluate.py \
  --model_name LLAMA-3.1-8B \
  --prompt_culture co \
  --language ar \
  --prompt_sample_count 50 \
  --entity_sample_count 50 \
  --draw_figure
```

Outputs:

```text
analysis/{MODEL}_{co|ag}_ar_50_50_cbs.json
analysis/{MODEL}_{co|ag}_ar_50_50.png
```

### 3. co vs ag Figure

After both `co` and `ag` CBS files exist:

```bash
cd /home/leekd97/experiments/TextInfilling/ours/results/text_infilling
python3 figure.py
```

Outputs:

```text
analysis/co_ag/{MODEL}_co_vs_ag_ar_50x50.png
analysis/co_ag/{MODEL}_co_vs_ag_ar_50x50.svg
```

## CBS Definition

For each prompt and entity type, the code scores all Arab and Western entity substitutions. CBS is the percentage of pairwise comparisons where the Western entity receives a higher model score than the Arab entity:

```text
CBS = P(score(Western entity) > score(Arab entity)) * 100
```

Higher CBS means stronger preference for Western entities under the same prompt context.

## Git Push

If the GitHub repository is already created, push from the project root:

```bash
cd /home/leekd97/experiments/TextInfilling
git init
git branch -M main
git status
git add README.md requirements.txt .gitignore ours
git commit -m "Add CAMeL text infilling CBS pipeline"
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

If `origin` already exists:

```bash
git remote -v
git remote set-url origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

Before pushing, check that private files are not staged:

```bash
git status --short
git diff --cached --name-only
```

`ours/configs/user_configs.yaml` must not appear in the staged file list.

Note: `dataset/camel` currently contains its own `.git` directory. If you want to include the dataset as plain files in this repository, remove that nested git metadata first. If you do not want to commit the dataset, keep it outside the commit and document how to download it.

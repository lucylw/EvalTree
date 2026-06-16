# Running the EvalTree pipeline on a new dataset

This guide walks through everything needed to build an EvalTree (capability tree)
for a dataset that isn't already wired into the repo, and to compute per-node
confidence intervals from a model's evaluation results.

A working template is provided alongside this guide:

- `EvalTree/run_pipeline.template.sh` — orchestration script (edit the variables at the top).
- `Datasets/MyDataset/` — example dataset folder with files in the exact expected formats.
- `EvalTree/stage1-CapabilityAnnotation/prompts/custom.txt` — template annotation prompt.
- `EvalTree/stage4-CapabilityDescription/prompts/custom.txt` — template node-description prompt.

Throughout, replace `MyDataset` with your dataset name and `my-model` with the name of
the model you evaluated.

---

## 0. Prerequisites

```bash
pip install -r requirements.txt
export OpenAI_API_KEY="sk-..."   # note the exact env var name used by utils/api_inference.py
```

All commands are run from the repo root (`/Users/lucylw/git/EvalTree`) and use
`python -m ...` module paths, so paths in arguments are relative to the repo root.

---

## 1. Understand the pipeline

EvalTree is built in 4 stages, then you compute confidence intervals to get a
weakness profile:

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| 1. Capability annotation | `EvalTree.stage1-CapabilityAnnotation.annotate` | each instance's input/output | one gerund-phrase capability per instance |
| 2. Capability embedding | `EvalTree.stage2-CapabilityEmbedding.embedding` | the capability phrases | an embedding vector per instance |
| 3. Recursive clustering | `EvalTree.stage3-RecursiveClustering.build` | the embeddings (+ optional split) | a hierarchical tree (`.bin`) |
| 4. Capability description | `EvalTree.stage4-CapabilityDescription.describe` | the tree + capabilities | tree with a natural-language description per node (`.json`) |
| CI | `EvalTree.WeaknessProfile.confidence_interval` | tree + eval results | per-node performance + confidence intervals |

Stage 1 calls an LLM per instance, stage 2 calls the embedding API per instance —
these cost money and scale with dataset size.

---

## 2. Lay out the dataset folder

Create `Datasets/MyDataset/` with this structure (see the provided `Datasets/MyDataset`
example for real files you can copy and edit):

```
Datasets/MyDataset/
├── dataset.json                       # array of instances (your raw data)
├── splits/
│   └── train-test.json                # OPTIONAL: array of integer indices into dataset.json
└── eval_results/
    └── real/
        └── my-model/
            └── results.json           # per-instance metrics, SAME ORDER as dataset.json
```

### `dataset.json`

A JSON array of objects. Each object must contain an **input** field and an
**output** field — the names are your choice, but you must point the loader at
them in step 3. Example (instruction/response style):

```json
[
  { "instruction": "Explain why the sky is blue.", "response": "The sky appears blue because..." },
  { "instruction": "Sort this list in Python.",     "response": "Use sorted(): ..." }
]
```

### `eval_results/real/my-model/results.json`

A JSON array, **one entry per instance, in the same order as `dataset.json`**.
Two metric formats are supported by the confidence-interval stage:

- **accuracy**: each entry is `0` or `1` (incorrect/correct):
  ```json
  [1, 0, 1, 1, 0]
  ```
- **win-rate** (pairwise, order-debiased): each entry is a 2-element list, each in `{1, 2}`:
  ```json
  [[1, 2], [1, 1], [2, 1]]
  ```

### `splits/` (optional)

If you only want to build the tree over a subset (e.g. a held-out test split),
create `splits/<name>.json` = a JSON array of 0-based indices into `dataset.json`.
If you build over the whole dataset, use `--split full` and you don't need a split file.

---

## 3. Register the dataset in the code

The stage scripts gate dataset names with argparse `choices` and branch on the name
to decide how to load data and which prompt to use. Add `MyDataset` in each of the
following. Diffs below show exactly what to add.

### 3a. `EvalTree/stage1-CapabilityAnnotation/annotate.py`

Add to `choices` (line ~12) and add a loading branch (after line ~40):

```python
parser.add_argument("--dataset", ..., choices = ("MATH", "WildChat10K", "DS-1000", ) + ("Chatbot-Arena", "ShareGPT10K", "MMLU", "CollegeMath", "MyDataset", ))
```

```python
elif args.dataset in ("MyDataset", ) :
    PROMPT = "custom"                              # -> prompts/custom.txt (template provided)
    INPUT_KEY, OUTPUT_KEY = "instruction", "response"   # <-- match your dataset.json field names
    with open("Datasets/{}/dataset.json".format(args.dataset), "r") as fin :
        dataset = json.load(fin)
```

`PROMPT` selects `EvalTree/stage1-CapabilityAnnotation/prompts/<PROMPT>.txt`. Reuse
`instruction-following` / `mathematics` / `ds-1000` / `mmlu` if one fits your domain,
or use the provided `custom` template and edit it.

### 3b. `EvalTree/stage2-CapabilityEmbedding/embedding.py`

Add to `choices` (line ~11) and pick the embedding prefix (line ~22):

```python
parser.add_argument("--dataset", ..., choices = (... , "MyDataset", ))
```

```python
elif args.dataset in ("WildChat10K", "Chatbot-Arena", "ShareGPT10K", "MMLU", "MyDataset", ) :
    PREFIX = "The model has the following capability: "
```

(Use `"The model has the following skill or capability: "` instead if your domain is
skill-oriented like MATH/DS-1000 — just match one of the existing branches.)

### 3c. `EvalTree/stage3-RecursiveClustering/build.py`

Add to `choices` (line ~10) only — no branching needed here:

```python
parser.add_argument("--dataset", ..., choices = ("MATH", "WildChat10K", "DS-1000", "MMLU", "MyDataset", ))
```

### 3d. `EvalTree/stage4-CapabilityDescription/describe.py`

Add to `choices` (line ~11) and add a prompt branch (after line ~30):

```python
parser.add_argument("--dataset", ..., choices = ("MATH", "WildChat10K", "DS-1000", "MyDataset", ))
```

```python
elif args.dataset in ("MyDataset", ) :
    PROMPT = "custom"     # -> EvalTree/stage4-CapabilityDescription/prompts/custom.txt
```

### 3e. `EvalTree/WeaknessProfile/confidence_interval.py`

Add to `choices` (line ~8) and set the metric type (line ~18):

```python
parser.add_argument("--dataset", ..., choices = ("MATH", "WildChat10K", "DS-1000", "MMLU", "MyDataset", ))
```

```python
if args.dataset in ("MATH", "DS-1000", "MMLU", "MyDataset", ) :   # accuracy datasets
    results_type = "accuracy"
elif args.dataset in ("WildChat10K", ) :                          # win-rate datasets
    results_type = "win-rate"
```

> The `run_pipeline.template.sh` script and the `MyDataset` example assume the
> **instruction/response + accuracy** combination. If you use win-rate, put `MyDataset`
> in the win-rate branch in 3e and use the `[[1,2],...]` results format instead.

---

## 4. Run the pipeline

Either run `EvalTree/run_pipeline.template.sh` (after editing the variables at its top),
or run the stages manually:

```bash
# Stage 1 — capability annotation (LLM call per instance)
python -m EvalTree.stage1-CapabilityAnnotation.annotate --dataset MyDataset

# Stage 2 — capability embedding (embedding call per instance)
python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset MyDataset

# Stage 3 — recursive clustering -> tree.bin  (use --split full or --split <name>)
python -m EvalTree.stage3-RecursiveClustering.build --dataset MyDataset --split full

# Stage 4 — describe each node.  --tree_path is RELATIVE to Datasets/MyDataset/EvalTree
#           and OMITS the .bin extension.
python -m EvalTree.stage4-CapabilityDescription.describe \
    --dataset MyDataset \
    --tree_path "stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]"

# Confidence intervals — per-node performance for a given model's results.
#   --tree_path: same relative path as stage 4 (no .bin).
#   --results_path: relative to Datasets/MyDataset/eval_results, points at the folder
#                   that contains results.json.
python -m EvalTree.WeaknessProfile.confidence_interval \
    --dataset MyDataset \
    --tree_path "stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]" \
    --results_path "real/my-model"
```

### Output locations

- Stage 1: `Datasets/MyDataset/EvalTree/stage1-CapabilityAnnotation/[annotation=gpt-4o-mini].json`
- Stage 2: `Datasets/MyDataset/EvalTree/stage2-CapabilityEmbedding/[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small].bin`
- Stage 3: `Datasets/MyDataset/EvalTree/stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10].bin`
- Stage 4: same path as stage 3 with `_[stage4-CapabilityDescription-model=gpt-4o-mini].json` appended — **this is the human-readable capability tree.**
- CI: `Datasets/MyDataset/eval_results/real/my-model/EvalTree/TREE=[...]/confidence_interval.json`

The filename brackets encode the config (annotation model, embedding model, split,
max children). If you change `--annotation_model`, `--embedding_model`, `--max_children`
or `--split`, the filenames in later stages change accordingly — keep `--tree_path` in
sync with whatever stage 3 produced.

---

## 5. Notes & gotchas

- **Order matters.** `results.json` must align index-for-index with `dataset.json`.
  Stage-3 tree leaves are dataset indices, so the CI stage indexes `results.json` by them.
- **`--tree_path` has no `.bin` extension** and is relative to `Datasets/<dataset>/EvalTree`.
  It must contain exactly one `/` (the CI stage asserts this when building its output path).
- **Default models** are `gpt-4o-mini` (annotation/description) and `text-embedding-3-small`
  (embedding); these are the only `choices` allowed, so don't pass others without widening them.
- **Cost** scales with dataset size in stages 1 and 2 (one API call per instance). The scripts
  print a `cost = ...` total at the end.
- Downstream assessment/baseline scripts under `Assessments/` and `Baselines/` have their own
  `choices` lists; add `MyDataset` there too if you use them.

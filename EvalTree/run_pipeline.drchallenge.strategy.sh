#!/usr/bin/env bash
# Strategy-based EvalTree variant for DRChallenge.
#
# Unlike run_pipeline.drchallenge.sh, this does NOT run the stage-1 capability
# annotation (no LLM call per instance). Instead each leaf is labelled with the
# dataset's own "strategy" field — the original reason the question was made
# harder. Stages 2-4 then embed, cluster, and summarise those strategies.
#
# Outputs are written under distinct [annotation=strategy] filenames, so the
# capability-annotation outputs from run_pipeline.drchallenge.sh are left intact
# (both viewers coexist — build_viewer.py names files by leaf-annotation source).
#
# Before running:
#   export OpenAI_API_KEY=...       # stage 2 embeddings (required)
#   export ANTHROPIC_API_KEY=...    # stage 4 node summaries, only if DESCRIPTION_MODEL is a claude model
# Run from the repo root:
#   bash EvalTree/run_pipeline.drchallenge.strategy.sh
set -euo pipefail

DATASET="DRChallenge"
SPLIT="full"                              # build over every instance in dataset.json
MODEL="drtulu"                            # eval_results/real/drtulu/results.json  (FAILED=1, PASSED=0)
ANNOTATION="strategy"                     # leaf labels = dataset "strategy" field (NO stage-1 LLM)
DATASET_FIELD="strategy"                  # which dataset.json field to use as the leaf label
DESCRIPTION_MODEL="gpt-4o-mini"           # stage 4: LLM that summarises INTERNAL nodes
EMBEDDING_MODEL="text-embedding-3-small"  # OpenAI embeddings (stage 2)
MAX_CHILDREN=10                           # max k at any single split
# Clustering mode (stage 3). Default: capped divisive k-means — build a hierarchy
# but stop once there are MIN_LEAVES..MAX_LEAVES leaf clusters (each a GROUP of
# strategies), splits chosen best-first by cosine silhouette. Set CAPPED=0 for the
# original fully-recursive tree (leaves = single instances).
CAPPED=1
MIN_LEAVES=6                              # keep splitting until at least this many leaf clusters
MAX_LEAVES=10                             # never exceed this many leaf clusters

if [ "$CAPPED" = "1" ] ; then
    CLUSTER_TAG="_[cluster=capped]"
    CLUSTER_ARGS=(--capped --min_leaf_clusters "$MIN_LEAVES" --max_leaf_clusters "$MAX_LEAVES")
else
    CLUSTER_TAG=""
    CLUSTER_ARGS=()
fi
TREE_PATH="stage3-RecursiveClustering/[split=${SPLIT}]_[annotation=${ANNOTATION}]_[embedding=${EMBEDDING_MODEL}]_[max-children=${MAX_CHILDREN}]${CLUSTER_TAG}"

echo "==> Clearing THIS variant's stale tree (built for a previous dataset size); other variants kept"
rm -f "Datasets/${DATASET}/EvalTree/${TREE_PATH}.bin" "Datasets/${DATASET}/EvalTree/${TREE_PATH}"*.json

echo "==> Stage 1 (no LLM): use dataset '${DATASET_FIELD}' field as the leaf annotation"
python - "$DATASET" "$DATASET_FIELD" "$ANNOTATION" <<'PY'
import json, os, sys
dataset, field, annotation = sys.argv[1], sys.argv[2], sys.argv[3]
with open("Datasets/{}/dataset.json".format(dataset)) as fin:
    ds = json.load(fin)
out_dir = "Datasets/{}/EvalTree/stage1-CapabilityAnnotation".format(dataset)
os.makedirs(out_dir, exist_ok=True)
labels = [x[field] for x in ds]
assert all(isinstance(s, str) and s.strip() for s in labels), "empty/non-string label in field '{}'".format(field)
with open(os.path.join(out_dir, "[annotation={}].json".format(annotation)), "w") as fout:
    json.dump(labels, fout, indent=2)
print("wrote {} '{}' annotations".format(len(labels), field))
PY

echo "==> Stage 2: embed the strategies"
python -m EvalTree.stage2-CapabilityEmbedding.embedding \
    --dataset "$DATASET" --annotation_model "$ANNOTATION" --embedding_model "$EMBEDDING_MODEL"

echo "==> Stage 3: clustering -> tree.bin  (mode: $([ "$CAPPED" = "1" ] && echo "capped ${MIN_LEAVES}-${MAX_LEAVES} leaf groups" || echo "fully recursive"))"
python -m EvalTree.stage3-RecursiveClustering.build \
    --dataset "$DATASET" --split "$SPLIT" \
    --annotation_model "$ANNOTATION" --embedding_model "$EMBEDDING_MODEL" --max_children "$MAX_CHILDREN" \
    ${CLUSTER_ARGS[@]+"${CLUSTER_ARGS[@]}"}

echo "==> Stage 4: leaves = strategies (no LLM), internal nodes summarised by ${DESCRIPTION_MODEL}"
python -m EvalTree.stage4-CapabilityDescription.describe \
    --dataset "$DATASET" --description_model "$DESCRIPTION_MODEL" \
    --leaf_annotation_model "$ANNOTATION" \
    --tree_path "$TREE_PATH"

echo "==> Confidence intervals for model: $MODEL  (failure rate per capability node)"
python -m EvalTree.WeaknessProfile.confidence_interval \
    --dataset "$DATASET" \
    --tree_path "$TREE_PATH" \
    --results_path "real/${MODEL}"

echo "==> Building standalone demo HTML viewer (FAIL_RATE overlay per node)"
python EvalTree/build_viewer.py --dataset "Datasets/${DATASET}"

CLUSTER_NAME="$([ "$CAPPED" = "1" ] && echo "capped" || echo "hierarchical")"
echo "==> Done."
echo "    Strategy tree: Datasets/${DATASET}/EvalTree/${TREE_PATH}_[stage4-CapabilityDescription-model=${DESCRIPTION_MODEL}].json"
echo "    Demo HTML:     Datasets/${DATASET}/EvalTree/viewers/viewer-${MODEL}-${ANNOTATION}-${CLUSTER_NAME}-${DESCRIPTION_MODEL}.html"

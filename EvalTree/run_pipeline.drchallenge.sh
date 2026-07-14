#!/usr/bin/env bash
# Build the EvalTree (capability tree) over the FULL DRChallenge dataset — every
# instance in Datasets/DRChallenge/dataset.json — and render the standalone demo
# HTML viewer for the drtulu deep-research agent.
#
# Labels: eval_results/real/drtulu/results.json holds one 0/1 per instance, in
# dataset.json order — 1 = the DR agent FAILED the updated_question, 0 = PASSED
# (this mirrors each entry's "drtulu_verdict" field). High-failure tree nodes
# therefore name the research capabilities that reliably stump the agent.
#
# The pipeline runs stage 1 TWICE, once per annotation model:
#   * claude-opus-4-8 annotations -> stage 4 leaf descriptions
#   * gpt-4o-mini annotations     -> stage 2 embeddings (shorter, more consistent
#                                    phrasings cluster better)
#
# Before running:
#   export ANTHROPIC_API_KEY=...   # stage 1 (opus) + stage 4 descriptions
#   export OpenAI_API_KEY=...      # stage 1 (gpt-4o-mini) + stage 2 embeddings
# Run from the repo root:
#   bash EvalTree/run_pipeline.drchallenge.sh
set -euo pipefail

DATASET="DRChallenge"
SPLIT="full"                              # build over every instance in dataset.json
MODEL="drtulu"                            # eval_results/real/drtulu/results.json  (FAILED=1, PASSED=0)
# ANNOTATION_MODEL="claude-opus-4-8"        # stage 1 (leaf descriptions) + stage 4 (Claude Opus 4.8)
ANNOTATION_MODEL="gpt-4o-mini"
ANNOTATION_MODEL_2="gpt-4o-mini"          # stage 1 (embedding annotations) + stage 2/3 keying
EMBEDDING_MODEL="text-embedding-3-small"  # OpenAI embeddings (stage 2)
MAX_CHILDREN=10                           # max k at any single split
# Clustering mode (stage 3). Default: capped divisive k-means — build a hierarchy
# but stop once there are MIN_LEAVES..MAX_LEAVES leaf clusters (each a GROUP of
# instances), splits chosen best-first by cosine silhouette. Set CAPPED=0 for the
# original fully-recursive tree (leaves = single instances).
CAPPED=1
MIN_LEAVES=6                              # keep splitting until at least this many leaf clusters
MAX_LEAVES=10                             # never exceed this many leaf clusters

# Relative path (under Datasets/$DATASET/EvalTree, no .bin) shared by stage 4 and the CI step.
# The .bin is written by stage 3, so this must key off the stage 2/3 annotation model
# (ANNOTATION_MODEL_2), not the stage 1/4 one. The [cluster=capped] tag keeps this
# variant's outputs distinct from a fully-recursive build.
if [ "$CAPPED" = "1" ] ; then
    CLUSTER_TAG="_[cluster=capped]"
    CLUSTER_ARGS=(--capped --min_leaf_clusters "$MIN_LEAVES" --max_leaf_clusters "$MAX_LEAVES")
else
    CLUSTER_TAG=""
    CLUSTER_ARGS=()
fi
TREE_PATH="stage3-RecursiveClustering/[split=${SPLIT}]_[annotation=${ANNOTATION_MODEL_2}]_[embedding=${EMBEDDING_MODEL}]_[max-children=${MAX_CHILDREN}]${CLUSTER_TAG}"

echo "==> Clearing THIS config's stale tree (built for a previous dataset size); other variants kept"
rm -f "Datasets/${DATASET}/EvalTree/${TREE_PATH}.bin" "Datasets/${DATASET}/EvalTree/${TREE_PATH}"*.json

echo "==> Stage 1a: capability annotation with ${ANNOTATION_MODEL}  (feeds stage 4 leaf descriptions)"
python -m EvalTree.stage1-CapabilityAnnotation.annotate \
    --dataset "$DATASET" --annotation_model "$ANNOTATION_MODEL"

# Stage 1b only adds anything when it uses a DIFFERENT model than 1a: the two write
# to [annotation=<model>].json, so identical models means 1b just re-runs 1a and
# overwrites the same file. Skip it in that case (e.g. the default all-gpt-4o-mini
# config); it's needed when ANNOTATION_MODEL is a richer model (e.g. claude-opus-4-8)
# for leaf descriptions while embeddings still key off gpt-4o-mini.
if [ "$ANNOTATION_MODEL_2" != "$ANNOTATION_MODEL" ] ; then
    echo "==> Stage 1b: capability annotation with ${ANNOTATION_MODEL_2}  (feeds stage 2 embeddings)"
    python -m EvalTree.stage1-CapabilityAnnotation.annotate \
        --dataset "$DATASET" --annotation_model "$ANNOTATION_MODEL_2"
else
    echo "==> Stage 1b: skipped (ANNOTATION_MODEL_2 == ANNOTATION_MODEL, reusing 1a's [annotation=${ANNOTATION_MODEL}].json)"
fi

echo "==> Stage 2: capability embedding"
python -m EvalTree.stage2-CapabilityEmbedding.embedding \
    --dataset "$DATASET" --annotation_model "$ANNOTATION_MODEL_2" --embedding_model "$EMBEDDING_MODEL"

echo "==> Stage 3: clustering -> tree.bin  (mode: $([ "$CAPPED" = "1" ] && echo "capped ${MIN_LEAVES}-${MAX_LEAVES} leaf groups" || echo "fully recursive"))"
python -m EvalTree.stage3-RecursiveClustering.build \
    --dataset "$DATASET" --split "$SPLIT" \
    --annotation_model "$ANNOTATION_MODEL_2" --embedding_model "$EMBEDDING_MODEL" --max_children "$MAX_CHILDREN" \
    ${CLUSTER_ARGS[@]+"${CLUSTER_ARGS[@]}"}

echo "==> Stage 4: capability descriptions -> tree.json"
python -m EvalTree.stage4-CapabilityDescription.describe \
    --dataset "$DATASET" --description_model "$ANNOTATION_MODEL" \
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
echo "    Capability tree: Datasets/${DATASET}/EvalTree/${TREE_PATH}_[stage4-CapabilityDescription-model=${ANNOTATION_MODEL}].json"
echo "    Demo HTML:       Datasets/${DATASET}/EvalTree/viewers/viewer-${MODEL}-${ANNOTATION_MODEL_2}-${CLUSTER_NAME}-${ANNOTATION_MODEL}.html"

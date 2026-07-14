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
MAX_CHILDREN=10

# Relative path (under Datasets/$DATASET/EvalTree, no .bin) shared by stage 4 and the CI step.
# The .bin is written by stage 3, so this must key off the stage 2/3 annotation model
# (ANNOTATION_MODEL_2), not the stage 1/4 one.
TREE_PATH="stage3-RecursiveClustering/[split=${SPLIT}]_[annotation=${ANNOTATION_MODEL_2}]_[embedding=${EMBEDDING_MODEL}]_[max-children=${MAX_CHILDREN}]"

echo "==> Clearing THIS config's stale tree (built for a previous dataset size); other variants kept"
rm -f "Datasets/${DATASET}/EvalTree/${TREE_PATH}.bin" "Datasets/${DATASET}/EvalTree/${TREE_PATH}"*.json

echo "==> Stage 1a: capability annotation with ${ANNOTATION_MODEL}  (feeds stage 4 leaf descriptions)"
python -m EvalTree.stage1-CapabilityAnnotation.annotate \
    --dataset "$DATASET" --annotation_model "$ANNOTATION_MODEL"

echo "==> Stage 1b: capability annotation with ${ANNOTATION_MODEL_2}  (feeds stage 2 embeddings)"
python -m EvalTree.stage1-CapabilityAnnotation.annotate \
    --dataset "$DATASET" --annotation_model "$ANNOTATION_MODEL_2"

echo "==> Stage 2: capability embedding"
python -m EvalTree.stage2-CapabilityEmbedding.embedding \
    --dataset "$DATASET" --annotation_model "$ANNOTATION_MODEL_2" --embedding_model "$EMBEDDING_MODEL"

echo "==> Stage 3: recursive clustering -> tree.bin"
python -m EvalTree.stage3-RecursiveClustering.build \
    --dataset "$DATASET" --split "$SPLIT" \
    --annotation_model "$ANNOTATION_MODEL_2" --embedding_model "$EMBEDDING_MODEL" --max_children "$MAX_CHILDREN"

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

echo "==> Done."
echo "    Capability tree: Datasets/${DATASET}/EvalTree/${TREE_PATH}_[stage4-CapabilityDescription-model=${ANNOTATION_MODEL}].json"
echo "    Demo HTML:       Datasets/${DATASET}/EvalTree/viewers/viewer-${MODEL}-${ANNOTATION_MODEL}.html"

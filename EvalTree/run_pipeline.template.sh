#!/usr/bin/env bash
# Template: build an EvalTree for a NEW dataset and compute confidence intervals.
#
# Before running:
#   1. export OpenAI_API_KEY=...
#   2. Lay out Datasets/$DATASET/ (see docs/RUN_ON_NEW_DATASET.md and the Datasets/MyDataset example)
#   3. Register $DATASET in the stage scripts' argparse `choices` + loading branches (step 3 of the guide)
#   4. Edit the variables below
#
# Run from the repo root:  bash EvalTree/run_pipeline.template.sh
set -euo pipefail

# ---- edit these ------------------------------------------------------------
DATASET="MyDataset"          # must match the folder name under Datasets/ and the argparse choices
SPLIT="full"                 # "full" (whole dataset) or the name of a Datasets/$DATASET/splits/<name>.json
MODEL="my-model"             # the evaluated model; results live at eval_results/real/$MODEL/results.json
ANNOTATION_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="text-embedding-3-small"
MAX_CHILDREN=10
# ---------------------------------------------------------------------------

# Relative path (under Datasets/$DATASET/EvalTree, no .bin) shared by stages 4 and CI.
TREE_PATH="stage3-RecursiveClustering/[split=${SPLIT}]_[annotation=${ANNOTATION_MODEL}]_[embedding=${EMBEDDING_MODEL}]_[max-children=${MAX_CHILDREN}]"

echo "==> Stage 1: capability annotation"
python -m EvalTree.stage1-CapabilityAnnotation.annotate \
    --dataset "$DATASET" --annotation_model "$ANNOTATION_MODEL"

echo "==> Stage 2: capability embedding"
python -m EvalTree.stage2-CapabilityEmbedding.embedding \
    --dataset "$DATASET" --annotation_model "$ANNOTATION_MODEL" --embedding_model "$EMBEDDING_MODEL"

echo "==> Stage 3: recursive clustering -> tree.bin"
python -m EvalTree.stage3-RecursiveClustering.build \
    --dataset "$DATASET" --split "$SPLIT" \
    --annotation_model "$ANNOTATION_MODEL" --embedding_model "$EMBEDDING_MODEL" --max_children "$MAX_CHILDREN"

echo "==> Stage 4: capability descriptions -> tree.json"
python -m EvalTree.stage4-CapabilityDescription.describe \
    --dataset "$DATASET" --description_model "$ANNOTATION_MODEL" \
    --tree_path "$TREE_PATH"

echo "==> Confidence intervals for model: $MODEL"
python -m EvalTree.WeaknessProfile.confidence_interval \
    --dataset "$DATASET" \
    --tree_path "$TREE_PATH" \
    --results_path "real/${MODEL}"

echo "==> Done. Capability tree:"
echo "    Datasets/${DATASET}/EvalTree/${TREE_PATH}_[stage4-CapabilityDescription-model=${ANNOTATION_MODEL}].json"

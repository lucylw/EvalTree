#!/usr/bin/env bash
# Build the EvalTree (capability tree) for the DRChallenge dataset and compute
# per-node confidence intervals for the drtulu deep-research agent.
#
# Instances are labeled 1 when the updated_question is hard (the DR agent FAILED)
# and 0 when it is answerable (PASSED), so high-failure tree nodes name the
# research capabilities that reliably stump the agent.
#
# Before running:
#   export ANTHROPIC_API_KEY=...   # stages 1 & 4 (capability annotation + description)
#   export OpenAI_API_KEY=...      # stage 2 only (embeddings — Anthropic has no embeddings API)
# Run from the repo root:
#   bash EvalTree/run_pipeline.drchallenge.sh
set -euo pipefail

DATASET="DRChallenge"
SPLIT="train-test"           # all 83 instances are test (Datasets/DRChallenge/splits/train-test.json)
MODEL="drtulu"               # eval_results/real/drtulu/results.json  (FAILED=1, PASSED=0)
ANNOTATION_MODEL="claude-opus-4-8"        # LLM for stages 1 & 4 (Claude Opus 4.8)
ANNOTATION_MODEL_2="gpt-4o-mini"                     # LLM for stage 2 (embedding) annotation (shorter, more consistent annotations for better embeddings)
EMBEDDING_MODEL="text-embedding-3-small"  # OpenAI embeddings (stage 2)
MAX_CHILDREN=10

# Relative path (under Datasets/$DATASET/EvalTree, no .bin) shared by stage 4 and the CI step.
# The .bin is written by stage 3, so this must key off the stage 2/3 annotation model
# (ANNOTATION_MODEL_2), not the stage 1/4 one.
TREE_PATH="stage3-RecursiveClustering/[split=${SPLIT}]_[annotation=${ANNOTATION_MODEL_2}]_[embedding=${EMBEDDING_MODEL}]_[max-children=${MAX_CHILDREN}]"

echo "==> Stage 1: capability annotation (one LLM call per question)"
python -m EvalTree.stage1-CapabilityAnnotation.annotate \
    --dataset "$DATASET" --annotation_model "$ANNOTATION_MODEL"

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

echo "==> Done. Capability tree:"
echo "    Datasets/${DATASET}/EvalTree/${TREE_PATH}_[stage4-CapabilityDescription-model=${ANNOTATION_MODEL}].json"

import os
import json
import torch
import random
import argparse
import concurrent.futures
from utils.common import manual_seed
from utils.api_inference import create_LLMclient, llm_completion, prompt_to_chatml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", "DRChallenge", ))
parser.add_argument("--tree_path", type = str, required = True)

parser.add_argument("--description_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", "claude-opus-4-8", ))
# The stage-1 annotation file used for LEAF descriptions. Defaults to --description_model
# (the usual behaviour: leaves reuse that model's capability annotations). Pass e.g.
# "strategy" to label leaves with a non-LLM annotation (such as the dataset's own field)
# while internal nodes are still summarised by --description_model.
parser.add_argument("--leaf_annotation_model", type = str, default = None)
parser.add_argument("--num_procs", type = int, default = 4)
# By default, after the bottom-up union summary is built, each internal (non-root)
# node's description is REWRITTEN to name what distinguishes that cluster from its
# sibling clusters, instead of merely summarising the examples it contains. This
# only runs when a "<prompt>.contrastive.txt" prompt exists for the dataset. Pass
# --disable_contrastive to keep the original union-summary behaviour.
parser.add_argument("--disable_contrastive", action = "store_true")
args = parser.parse_args()


TREE = torch.load(os.path.join("Datasets/{}/EvalTree".format(args.dataset), "{}.bin".format(args.tree_path)), weights_only = False)
LEAF_ANNOTATION = args.leaf_annotation_model if args.leaf_annotation_model is not None else args.description_model
with open("Datasets/{}/EvalTree/stage1-CapabilityAnnotation/[annotation={}].json".format(args.dataset, LEAF_ANNOTATION), "r") as fin :
    CAPABILITIES = json.load(fin)

if args.dataset == "MATH" :
    PROMPT_NAME = "mathematics"
elif args.dataset in ("WildChat10K", "Chatbot-Arena", ) :
    PROMPT_NAME = "instruction-following"
elif args.dataset == "DS-1000" :
    PROMPT_NAME = "ds-1000"
elif args.dataset == "MMLU" :
    PROMPT_NAME = "mmlu"
elif args.dataset == "DRChallenge" :
    PROMPT_NAME = "drchallenge"
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))
with open("EvalTree/stage4-CapabilityDescription/prompts/{}.txt".format(PROMPT_NAME), "r") as fin :
    PROMPT = fin.read()

# Optional contrastive-description prompt: rewrites each internal (non-root) node
# to describe what SETS IT APART from its sibling clusters. Only enabled when the
# file exists and --disable_contrastive was not passed.
CONTRASTIVE_PROMPT_PATH = "EvalTree/stage4-CapabilityDescription/prompts/{}.contrastive.txt".format(PROMPT_NAME)
CONTRASTIVE_PROMPT = None
if not args.disable_contrastive and os.path.exists(CONTRASTIVE_PROMPT_PATH) :
    with open(CONTRASTIVE_PROMPT_PATH, "r") as fin :
        CONTRASTIVE_PROMPT = fin.read()


def initialize_description(tree) :
    tree_description = {
        "description" : None,
    }
    if isinstance(tree, dict) :
        assert "subtrees" in tree
        if isinstance(tree["subtrees"], list) :
            tree_description["subtrees"] = [initialize_description(subtree) for subtree in tree["subtrees"]]
        elif isinstance(tree["subtrees"], dict) :
            tree_description["subtrees"] = {key : initialize_description(subtree) for key, subtree in tree["subtrees"].items()}
        else :
            raise NotImplementedError
    else :
        assert isinstance(tree, int)
        tree_description["subtrees"] = tree
    return tree_description


EXECUTORS = {}
LLM_KWARGS = {
    "model" : args.description_model,
    "max_tokens" : 1024,
    "temperature" : 0.0,  # ignored on Claude models (sampling params are removed on Opus 4.x)
    "seed" : 0,           # ignored on Claude models
}
def children_of(tree_description) :
    subtrees = tree_description["subtrees"]
    return subtrees if isinstance(subtrees, list) else list(subtrees.values())


def describe(tree_description, depth) :
    cost = 0.0
    if not isinstance(tree_description["subtrees"], int) :
        assert isinstance(tree_description["subtrees"], list) or isinstance(tree_description["subtrees"], dict)

        if depth not in EXECUTORS :
            EXECUTORS[depth] = concurrent.futures.ThreadPoolExecutor(max_workers = args.num_procs)
        executor = EXECUTORS[depth]
        cost += sum(list(executor.map(lambda subtree: describe(subtree, depth + 1), children_of(tree_description))))

        # If every leaf under this node carries the SAME annotation (e.g. an identical
        # "strategy" string), there is nothing to summarise: emit that string verbatim
        # and make no LLM call. "common_annotation" propagates this up the tree.
        child_commons = [subtree["common_annotation"] for subtree in children_of(tree_description)]
        if all(common is not None for common in child_commons) and len(set(child_commons)) == 1 :
            tree_description["common_annotation"] = child_commons[0]
            tree_description["self_description"] = child_commons[0]
            tree_description["description"] = child_commons[0]
            return cost
        tree_description["common_annotation"] = None

        skills = [subtree["self_description"] for subtree in children_of(tree_description)]
        manual_seed(42)
        random.shuffle(skills)
        skills = ["### Skill #{}\n{}\n".format(index + 1, skill) for index, skill in enumerate(skills)]

        chatml = prompt_to_chatml(prompt = PROMPT.format_map(dict(group_number = len(skills), skill_descriptions = "\n".join(skills))))
        client = create_LLMclient(args.description_model)
        response = llm_completion(client, chatml, LLM_KWARGS)
        # "self_description" = the bottom-up union summary of the examples in this
        # node (the original stage-4 behaviour). "description" starts as a copy and
        # is later overwritten by the contrastive pass for internal, non-root nodes.
        tree_description["self_description"] = response["response"].strip()
        tree_description["description"] = tree_description["self_description"]
        cost += response["cost"]
        print(tree_description["self_description"])
    else :
        tree_description["common_annotation"] = CAPABILITIES[tree_description["subtrees"]]
        tree_description["self_description"] = tree_description["common_annotation"]
        tree_description["description"] = tree_description["self_description"]
    return cost


CONTRA_EXECUTORS = {}
def contrastive(tree_description, depth) :
    """Top-down pass: rewrite each internal child's "description" to name what
    distinguishes it from its sibling children, then recurse into those children.
    Reads only "self_description" (fixed by the bottom-up pass), so results are
    independent of ordering. Leaves and the root are left untouched."""
    cost = 0.0
    if isinstance(tree_description["subtrees"], int) :
        return cost
    kids = children_of(tree_description)

    if depth not in CONTRA_EXECUTORS :
        CONTRA_EXECUTORS[depth] = concurrent.futures.ThreadPoolExecutor(max_workers = args.num_procs)
    executor = CONTRA_EXECUTORS[depth]

    def rewrite(index) :
        kid = kids[index]
        if isinstance(kid["subtrees"], int) :
            return 0.0  # keep the leaf's stage-1 capability annotation verbatim
        if kid["common_annotation"] is not None :
            return 0.0  # homogeneous cluster: keep the shared strategy string verbatim
        if len(kids) < 2 :
            return 0.0  # nothing to contrast against
        target_details = "\n".join(
            "- {}".format(grandchild["self_description"]) for grandchild in children_of(kid)
        )
        siblings = [
            "### Sibling Group #{}\n{}\n".format(order + 1, kids[j]["self_description"])
            for order, j in enumerate(k for k in range(len(kids)) if k != index)
        ]
        chatml = prompt_to_chatml(prompt = CONTRASTIVE_PROMPT.format_map(dict(
            sibling_number = len(siblings),
            target_description = kid["self_description"],
            target_details = target_details,
            sibling_descriptions = "\n".join(siblings),
        )))
        client = create_LLMclient(args.description_model)
        response = llm_completion(client, chatml, LLM_KWARGS)
        kid["description"] = response["response"].strip()
        print(kid["description"])
        return response["cost"]

    cost += sum(list(executor.map(rewrite, range(len(kids)))))
    cost += sum(list(executor.map(lambda kid: contrastive(kid, depth + 1), kids)))
    return cost


try :
    TREE_DESCRIPTION = initialize_description(TREE)
    print("==> Bottom-up union summaries")
    print("cost = {}".format(describe(TREE_DESCRIPTION, 0)))
    if CONTRASTIVE_PROMPT is not None :
        print("==> Contrastive rewrite (each cluster vs. its siblings)")
        print("cost = {}".format(contrastive(TREE_DESCRIPTION, 0)))
    else :
        print("==> Contrastive rewrite skipped (disabled or no contrastive prompt for this dataset)")
finally :
    for executor in EXECUTORS.values() :
        executor.shutdown(wait = True)
    for executor in CONTRA_EXECUTORS.values() :
        executor.shutdown(wait = True)


with open(os.path.join("Datasets/{}/EvalTree".format(args.dataset), "{}_[stage4-CapabilityDescription-model={}].json".format(args.tree_path, args.description_model)), "w") as fout :
    json.dump(TREE_DESCRIPTION, fout, indent = 2)
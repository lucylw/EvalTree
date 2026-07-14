import os
import json
import torch
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", "MMLU", "DRChallenge", ))
parser.add_argument("--annotation_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", "strategy", ))
parser.add_argument("--embedding_model", type = str, default = "text-embedding-3-small", choices = ("text-embedding-3-small", ))
parser.add_argument("--max_children", type = int, default = 10)
parser.add_argument("--split", type = str, default = "full")
# Capped divisive mode: build a hierarchy by k-means, but stop once there are
# roughly --min_leaf_clusters..--max_leaf_clusters LEAF CLUSTERS (each a GROUP of
# instances, not a singleton). Splits are chosen best-first by cosine silhouette
# (same criterion the recursive build uses), so you keep a real hierarchy while
# controlling how many terminal groups you end up with. Without --capped the
# original fully-recursive tree (leaves = single instances) is built.
parser.add_argument("--capped", action = "store_true")
parser.add_argument("--min_leaf_clusters", type = int, default = 6)
parser.add_argument("--max_leaf_clusters", type = int, default = 10)
args = parser.parse_args()

assert args.max_children >= 2
MATRIX = torch.stack(torch.load("Datasets/{}/EvalTree/stage2-CapabilityEmbedding/[annotation={}]_[embedding={}].bin".format(args.dataset, args.annotation_model, args.embedding_model), weights_only = True)).numpy()


def build_tree(instances : np.ndarray) :
    if len(instances) == 1 :
        return instances[0].item()
    if len(instances) == 2 :
        return {"subtrees" : instances.tolist(), "kmeans" : None}
    matrix = MATRIX[instances]
    
    all_labels, all_scores, all_kmeans = [], [], []
    for n_clusters in range(2, min(args.max_children + 1, len(instances))) :
        kmeans = KMeans(
            n_clusters = n_clusters,
            init = "k-means++",
            random_state = 42,
        ).fit(matrix)
        labels : np.ndarray = kmeans.labels_

        if labels.max() == 0 :
            assert len(instances) > 1
            return {"subtrees" : instances.tolist(), "kmeans" : None}

        all_labels.append(labels)
        all_scores.append(silhouette_score(matrix, labels, metric = "cosine"))
        all_kmeans.append(kmeans)
    
    if max(all_scores) <= 0.0 :
        return {"subtrees" : instances.tolist(), "kmeans" : None}
    
    picked_index = np.argmax(all_scores)
    labels = all_labels[picked_index]
    kmeans = all_kmeans[picked_index]

    cluster2subtree = {}
    for cluster in range(args.max_children) :
        subtree_instances = instances[labels == cluster]
        if len(subtree_instances) :
            cluster2subtree[cluster] = subtree_instances
    assert len(cluster2subtree)

    if len(cluster2subtree) == 1 : # There is only one cluster, and the instance number is more than 1.
        assert len(instances) > 1
        return {"subtrees" : instances.tolist(), "kmeans" : None}

    return {
        "subtrees" : {cluster : build_tree(subtree) for cluster, subtree in cluster2subtree.items()},
        "kmeans" : kmeans,
    }


def best_split(instances : np.ndarray, kmax : int) :
    """k-means over `instances` for k in [2, kmax], returning the (score, labels,
    kmeans) with the highest cosine silhouette, or None if the group is too small
    to split or no split separates it. Mirrors build_tree's per-node k selection."""
    if len(instances) < 3 :
        return None
    kmax = min(kmax, len(instances) - 1)
    if kmax < 2 :
        return None
    matrix = MATRIX[instances]
    best = None
    for n_clusters in range(2, kmax + 1) :
        kmeans = KMeans(
            n_clusters = n_clusters,
            init = "k-means++",
            random_state = 42,
        ).fit(matrix)
        labels = kmeans.labels_
        if labels.max() == 0 :  # degenerate: everything in one cluster
            continue
        score = silhouette_score(matrix, labels, metric = "cosine")
        if best is None or score > best[0] :
            best = (score, labels, kmeans)
    return best


def build_capped(instances : np.ndarray) :
    """Divisive hierarchical k-means. Repeatedly split the leaf cluster whose best
    silhouette split is highest (best-first), until we have between
    --min_leaf_clusters and --max_leaf_clusters terminal groups. Below the minimum
    we keep splitting even if silhouette is non-positive (to reach the target);
    within the window we split only while a split still improves silhouette. Each
    split's k is capped so the leaf-cluster count never exceeds the maximum."""
    THRESHOLD = 0.0
    MIN_LEAVES, MAX_LEAVES = args.min_leaf_clusters, args.max_leaf_clusters
    assert 2 <= MIN_LEAVES <= MAX_LEAVES

    root = {"instances" : instances, "children" : None, "kmeans" : None,
            "prop" : best_split(instances, args.max_children)}
    leaves = [root]

    while len(leaves) < MAX_LEAVES :
        splittable = [leaf for leaf in leaves if leaf["prop"] is not None]
        if not splittable :
            break
        target = max(splittable, key = lambda leaf : leaf["prop"][0])
        forced = len(leaves) < MIN_LEAVES  # must keep splitting to reach the minimum
        if not forced and target["prop"][0] <= THRESHOLD :
            break

        score, labels, kmeans = target["prop"]
        # Cap the split's k so the total leaf count stays <= MAX_LEAVES.
        remaining = MAX_LEAVES - len(leaves)
        if (labels.max() + 1) - 1 > remaining :
            capped = best_split(target["instances"], remaining + 1)
            if capped is None :
                target["prop"] = None  # cannot split within budget; take it off the table
                continue
            score, labels, kmeans = capped

        children = {}
        for cluster in range(labels.max() + 1) :
            members = target["instances"][labels == cluster]
            if len(members) == 0 :
                continue
            children[int(cluster)] = {"instances" : members, "children" : None, "kmeans" : None,
                                      "prop" : best_split(members, args.max_children)}
        target["children"] = children
        target["kmeans"] = kmeans
        # Identity-based removal: these node dicts hold numpy arrays, so list.remove
        # (which uses ==) would trigger an ambiguous-truth error.
        leaves = [leaf for leaf in leaves if leaf is not target]
        leaves.extend(children.values())
        print("split (silhouette={:.4f}) -> {} leaf clusters".format(score, len(leaves)))

    print("final: {} leaf clusters".format(len(leaves)))

    def finalize(node) :
        if node["children"] is None :
            members = node["instances"]
            # Terminal GROUP: a flat list of leaf ids (bare int if a lone instance),
            # matching build_tree's leaf convention.
            return {"subtrees" : members.tolist(), "kmeans" : None} if len(members) > 1 else int(members[0])
        return {"subtrees" : {cluster : finalize(child) for cluster, child in node["children"].items()},
                "kmeans" : node["kmeans"]}
    return finalize(root)


os.makedirs("Datasets/{}/EvalTree/stage3-RecursiveClustering".format(args.dataset), exist_ok = True)
if args.split == "full" :
    RANGE = np.arange(MATRIX.shape[0])
else :
    with open("Datasets/{}/splits/{}.json".format(args.dataset, args.split), "r") as fin :
        RANGE = np.array(json.load(fin))

TREE = build_capped(RANGE) if args.capped else build_tree(RANGE)
CLUSTER_TAG = "_[cluster=capped]" if args.capped else ""
torch.save(TREE, "Datasets/{}/EvalTree/stage3-RecursiveClustering/[split={}]_[annotation={}]_[embedding={}]_[max-children={}]{}.bin".format(args.dataset, args.split, args.annotation_model, args.embedding_model, args.max_children, CLUSTER_TAG))
#!/usr/bin/env python3
"""Build self-contained static capability-tree viewers — one HTML per model output.

For each evaluated model (eval_results/<.../>/results.json, a list of 0/1 where
1 = FAILED) crossed with each stage-4 capability tree, this:

  1. walks the tree and computes FAIL_RATE per node (failed / scored instances),
  2. annotates every node with that aggregate ("fail"/"scored" counts), and
  3. embeds the annotated tree directly into a standalone .html file.

The result needs no server and no separate results.json — open it with file://.

Usage (from the repo root):
  python3 EvalTree/build_viewer.py                       # defaults to DRChallenge
  python3 EvalTree/build_viewer.py --dataset Datasets/MMLU --out-dir /tmp/viewers
"""
import argparse
import json
import re
from pathlib import Path


def is_leaf(node):
    return isinstance(node["subtrees"], int)


def children_of(node):
    sub = node["subtrees"]
    return sub if isinstance(sub, list) else list(sub.values())


def max_leaf_index(node):
    """Largest dataset index referenced by any leaf under `node` (-1 if none)."""
    if is_leaf(node):
        return node["subtrees"]
    return max((max_leaf_index(ch) for ch in children_of(node)), default=-1)


def annotate(node, results, idx_to_run=None, subsets=()):
    """Attach precomputed FAIL_RATE aggregates to every node.

    Adds two integers per node:
      fail   — instances under the node the model FAILED (results value == 1)
      scored — instances under the node that have a result at all
    When `subsets` is non-empty, also adds `by_subset` = {subset: {fail, scored}}
    so the viewer can recompute FAIL_RATE per source_run subset (idx_to_run maps a
    dataset index -> its subset name). Returns (fail, scored) so parents can sum them.
    """
    if is_leaf(node):
        idx = node["subtrees"]
        v = results[idx] if (results is not None and 0 <= idx < len(results)) else None
        scored = 1 if isinstance(v, (int, float)) else 0
        fail = int(v) if scored else 0
        node["fail"], node["scored"] = fail, scored
        if subsets:
            run = (idx_to_run or {}).get(idx)
            node["subset"] = run  # which subset this single instance came from
            node["by_subset"] = {s: {"fail": fail if s == run else 0,
                                     "scored": scored if s == run else 0} for s in subsets}
        return fail, scored

    fail = scored = 0
    by_subset = {s: {"fail": 0, "scored": 0} for s in subsets} if subsets else None
    for ch in children_of(node):
        f, s = annotate(ch, results, idx_to_run, subsets)
        fail += f
        scored += s
        if subsets:
            cb = ch["by_subset"]
            for s in subsets:
                by_subset[s]["fail"] += cb[s]["fail"]
                by_subset[s]["scored"] += cb[s]["scored"]
    node["fail"], node["scored"] = fail, scored
    if subsets:
        node["by_subset"] = by_subset
    return fail, scored


def describe_tree(path):
    """Pull the stage-4 (description), annotation model, and clustering-mode names
    out of a filename. `cluster` is "flat" for single-level k-means trees (tagged
    [cluster=flat] by stage 3) and "hierarchical" otherwise."""
    name = path.name
    desc = (re.search(r"stage4-CapabilityDescription-model=([^\]]+)", name) or [None, "?"])[1]
    annot = (re.search(r"\[annotation=([^\]]+)\]", name) or [None, None])[1]
    cluster = (re.search(r"\[cluster=([^\]]+)\]", name) or [None, "hierarchical"])[1]
    return desc, annot, cluster


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>EvalTree — __TITLE__</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --line: #2a2f3a; --text: #e6e9ef;
    --muted: #9aa3b2; --accent: #6ea8fe; --accent-soft: #6ea8fe22;
    --leaf: #3ddc97; --hit: #ffd166;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
  }
  header {
    position: sticky; top: 0; z-index: 5; background: var(--panel);
    border-bottom: 1px solid var(--line); padding: 12px 16px;
    display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center;
  }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .2px; }
  header .spacer { flex: 1; }
  .ctl { display: flex; align-items: center; gap: 6px; color: var(--muted); }
  input[type="search"], select {
    background: var(--bg); border: 1px solid var(--line); color: var(--text);
    border-radius: 8px; padding: 6px 10px; outline: none; font: inherit;
  }
  input[type="search"] { min-width: 220px; }
  input[type="search"]:focus, select:focus { border-color: var(--accent); }
  main { padding: 16px 20px 80px; }

  ul.tree, ul.tree ul { list-style: none; margin: 0; padding: 0; }
  ul.tree ul { margin-left: 15px; border-left: 1px solid var(--line); padding-left: 16px; }
  li.node { margin: 6px 0; position: relative; }
  ul.tree ul > li.node::before {
    content: ""; position: absolute; left: -16px; top: 18px;
    width: 14px; height: 1px; background: var(--line);
  }
  .row {
    display: flex; align-items: center; gap: 10px; padding: 9px 12px;
    border-radius: 10px; cursor: default; background: #1a1e27;
    border: 1px solid var(--line); transition: border-color .12s ease, background .12s ease;
  }
  li.internal > .row { border-left: 3px solid var(--accent); }
  li.leaf > .row { background: #141a18; border-left: 3px solid var(--leaf); }
  .row.clickable { cursor: pointer; }
  .row.clickable:hover { background: #20262f; border-color: var(--accent); }
  .twisty {
    flex: 0 0 auto; width: 14px; text-align: center; color: var(--muted);
    user-select: none; font-size: 11px; transition: transform .12s ease;
  }
  li.collapsed > .row .twisty { transform: rotate(-90deg); }
  li.leaf > .row .twisty { visibility: hidden; }
  .count {
    flex: 0 0 auto; min-width: 56px; text-align: center;
    font-variant-numeric: tabular-nums; font-size: 12px;
    background: var(--accent-soft); color: var(--accent); border-radius: 999px; padding: 2px 9px;
  }
  .count b { font-weight: 700; }
  li.leaf .count { background: #3ddc9722; color: var(--leaf); }
  .count .unit { opacity: .75; font-weight: 400; }
  .icon { flex: 0 0 auto; }
  li.internal > .row .icon { color: var(--accent); }
  li.leaf > .row .icon { color: var(--leaf); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  .acc {
    flex: 0 0 auto; min-width: 88px; display: inline-flex; align-items: center; gap: 6px;
    font-variant-numeric: tabular-nums; font-size: 12px;
  }
  .acc .bar { width: 40px; height: 7px; border-radius: 4px; background: #2a2f3a; overflow: hidden; }
  .acc .bar > span { display: block; height: 100%; }
  .acc .pct { min-width: 34px; text-align: right; font-weight: 600; }
  .acc .frac { color: var(--muted); font-size: 11px; }
  /* comparison mode: one .acc per subset, side by side, each tagged with its subset */
  .accs { flex: 0 0 auto; display: inline-flex; gap: 16px; align-items: center; }
  .acc .slab { font-size: 11px; font-weight: 600; min-width: 52px; text-align: right; }
  .acc.empty .bar, .acc.empty .pct, .acc.empty .frac { opacity: .28; }
  /* leaf origin tag */
  .chip {
    flex: 0 0 auto; font-size: 11px; padding: 1px 8px; border-radius: 999px;
    border: 1px solid currentColor; white-space: nowrap; opacity: .9;
  }
  .desc { flex: 1; }
  li.internal > .row .desc { font-weight: 500; }
  li.leaf > .row .desc { color: var(--muted); }
  li.collapsed > ul { display: none; }
  mark { background: var(--hit); color: #000; border-radius: 3px; padding: 0 1px; }
  .hidden { display: none !important; }
  .meta { color: var(--muted); font-size: 12px; }
  .legend { color: var(--muted); font-size: 12px; margin-bottom: 12px; }
  .legend b { color: var(--accent); } .legend i { color: var(--leaf); font-style: normal; }
</style>
</head>
<body>
<header>
  <h1>🌳 __HEADER__</h1>
  <div class="ctl">metric
    <select id="metric" title="1 = FAILED, 0 = PASSED">
      <option value="fail" selected>FAIL_RATE (fraction failed)</option>
      <option value="pass">PASS_RATE (fraction passed)</option>
    </select>
  </div>
  __MODE_CONTROL__
  <div class="ctl"><input id="search" type="search" placeholder="Filter capabilities…" /></div>
  <div class="spacer"></div>
  <div class="ctl">show depth
    <select id="depth">
      <option value="1">1</option>
      <option value="2" selected>2</option>
      <option value="3">3</option>
      <option value="4">4</option>
      <option value="99">all</option>
    </select>
  </div>
  <div class="ctl"><label><input id="showLeaves" type="checkbox" /> instances</label></div>
  <div class="meta" id="stats"></div>
</header>
<main>
  <div class="legend" id="legend">
    <b>▸ capability node</b> (groups instances) &nbsp;·&nbsp;
    <i>#n leaf</i> = one dataset instance &nbsp;·&nbsp;
    <b>n=</b> instances under a node &nbsp;·&nbsp;
    <b id="legendMetric">FAIL_RATE</b> = fraction of instances under a node the model FAILED; green = higher failure (capabilities that stump the model)
    <span id="legendMode"></span>
  </div>
  <ul class="tree" id="tree"></ul>
</main>

<script>
// FAIL_RATE is precomputed per node (node.fail / node.scored) at build time.
// node.by_subset[<run>] holds the same counts split by source_run subset.
const TREE_DATA = __TREE_JSON__;
const SUBSETS = __SUBSETS_JSON__;   // [] when there's nothing to compare

const $ = (id) => document.getElementById(id);
let stats = { nodes: 0, leaves: 0, depth: 0 };

// Short subset labels: drop the prefix common to every subset (e.g.
// "sqa_50_100_explore" -> "explore"), trimmed back to an underscore boundary.
function commonPrefix(arr) {
  if (!arr.length) return "";
  let p = arr[0];
  for (const s of arr) while (!s.startsWith(p)) p = p.slice(0, -1);
  return p.slice(0, p.lastIndexOf("_") + 1);
}
const CP = commonPrefix(SUBSETS);
const shortLabel = (s) => (CP && s.length > CP.length ? s.slice(CP.length) : s);
// Stable identity colors per subset (off the red→green FAIL_RATE ramp so the two
// encodings never collide). Assigned in the fixed order subsets are listed.
const SUBSET_PALETTE = ["#6ea8fe", "#f6c177", "#c58af9", "#3ddc97", "#ff8fa3"];
const subsetColor = (s) => SUBSET_PALETTE[Math.max(0, SUBSETS.indexOf(s)) % SUBSET_PALETTE.length];

const isLeaf = (n) => typeof n.subtrees === "number";
const childrenOf = (n) => (Array.isArray(n.subtrees) ? n.subtrees : Object.values(n.subtrees));

function countInstances(n) {
  if (isLeaf(n)) return 1;
  if (n.__count != null) return n.__count;
  let c = 0;
  for (const ch of childrenOf(n)) c += countInstances(ch);
  return (n.__count = c);
}

const metricMode = () => $("metric").value;                 // "fail" | "pass"
const metricLabel = () => (metricMode() === "fail" ? "FAIL_RATE" : "PASS_RATE");
// "full" = one combined badge; "comparison" = one badge per subset side by side.
const viewMode = () => ($("mode") ? $("mode").value : "full");

// Failed/scored counts for a given source: "all" = every instance under the node,
// otherwise the named source_run subset (node.by_subset[<run>]).
function countsFor(n, source) {
  if (source === "all" || !n.by_subset) return { fail: n.fail || 0, scored: n.scored || 0 };
  const b = n.by_subset[source] || { fail: 0, scored: 0 };
  return { fail: b.fail || 0, scored: b.scored || 0 };
}

function metricOf(n, source = "all") {
  const { fail, scored } = countsFor(n, source);
  if (!scored) return null;
  const passed = scored - fail;
  const num = metricMode() === "fail" ? fail : passed;
  return { value: num / scored, num, scored, label: metricLabel() };
}

// One FAIL_RATE bar for `node` restricted to `source`. `slab` (optional) is the
// subset label shown in comparison mode so each bar is identified.
function metricBadge(node, source = "all", slab = null) {
  const span = document.createElement("span");
  span.className = "acc";
  let html = "";
  if (slab != null)
    html += '<span class="slab" style="color:' + subsetColor(source) + '">' + escapeHTML(slab) + "</span>";
  const c = countsFor(node, source);
  const m = metricOf(node, source);
  if (m == null) {
    span.className += " empty";
    span.innerHTML = html +
      '<span class="bar"></span><span class="pct">—</span><span class="frac">0/0</span>';
    span.title = (slab != null ? slab + ": " : "") + "no instances here";
    return span;
  }
  const failRate = c.fail / c.scored;  // green = MORE failures (capabilities that stump the model), red = fewer
  const hue = Math.round(120 * failRate);
  const pct = Math.round(m.value * 100);
  span.innerHTML = html +
    '<span class="bar"><span style="width:' + pct + '%;background:hsl(' + hue + ',62%,46%)"></span></span>' +
    '<span class="pct" style="color:hsl(' + hue + ',62%,58%)">' + pct + '%</span>' +
    '<span class="frac">' + m.num + '/' + m.scored + '</span>';
  span.title = (slab != null ? slab + " · " : "") + m.label + " " + m.num + "/" + m.scored + " (" + pct + "%)";
  return span;
}

// The metric area for a row: a single combined badge (full) or one per subset
// (comparison). Returns an element to drop into the row.
function metricArea(node) {
  if (viewMode() === "comparison" && SUBSETS.length) {
    const wrap = document.createElement("span");
    wrap.className = "accs";
    for (const s of SUBSETS) wrap.appendChild(metricBadge(node, s, shortLabel(s)));
    return wrap;
  }
  return metricBadge(node, "all");
}

// A small pill on a leaf row naming the subset that single instance came from.
function subsetChip(node) {
  if (!SUBSETS.length || !node.subset) return null;
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = shortLabel(node.subset);
  chip.style.color = subsetColor(node.subset);
  chip.title = "source_run: " + node.subset;
  return chip;
}

const escapeHTML = (s) => String(s).replace(/[&<>]/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]));

function buildLi(node, depth) {
  stats.depth = Math.max(stats.depth, depth);
  const leaf = isLeaf(node);
  leaf ? stats.leaves++ : stats.nodes++;

  const li = document.createElement("li");
  li.className = "node " + (leaf ? "leaf" : "internal");
  li.dataset.depth = depth;
  li.dataset.text = (node.description || "").toLowerCase();

  const row = document.createElement("div");
  row.className = "row" + (leaf ? "" : " clickable");
  const tw = document.createElement("span"); tw.className = "twisty"; tw.textContent = "▾";
  const count = document.createElement("span"); count.className = "count";
  const n = countInstances(node);
  count.innerHTML = leaf ? '<span class="unit">leaf</span>' : '<span class="unit">n=</span><b>' + n + "</b>";
  const icon = document.createElement("span"); icon.className = "icon";
  icon.textContent = leaf ? "#" + node.subtrees : "▸";
  const acc = metricArea(node);
  const desc = document.createElement("span"); desc.className = "desc";
  desc.textContent = node.description || "(no description)";

  row.append(tw, count, acc, icon, desc);
  const chip = leaf ? subsetChip(node) : null;   // leaf origin (which subset)
  if (chip) row.appendChild(chip);
  li.appendChild(row);

  if (!leaf) {
    const ul = document.createElement("ul");
    for (const ch of childrenOf(node)) ul.appendChild(buildLi(ch, depth + 1));
    li.appendChild(ul);
    row.addEventListener("click", (e) => { e.stopPropagation(); li.classList.toggle("collapsed"); });
  }
  return li;
}

function render() {
  stats = { nodes: 0, leaves: 0, depth: 0 };
  const root = $("tree");
  root.innerHTML = "";
  root.appendChild(buildLi(TREE_DATA, 0));
  let line = `${stats.nodes.toLocaleString()} capability nodes · ${stats.leaves.toLocaleString()} instances · depth ${stats.depth}`;
  if (viewMode() === "comparison" && SUBSETS.length) {
    const parts = SUBSETS.map((s) => {
      const m = metricOf(TREE_DATA, s);
      return m ? `${shortLabel(s)} ${Math.round(m.value * 100)}% (${m.num}/${m.scored})` : `${shortLabel(s)} —`;
    });
    line += ` · overall ${metricLabel()} — ${parts.join("  vs  ")}`;
  } else {
    const m = metricOf(TREE_DATA, "all");
    if (m != null) line += ` · overall ${m.label} ${Math.round(m.value * 100)}% (${m.num}/${m.scored})`;
  }
  $("stats").textContent = line;
  $("legendMetric").textContent = metricLabel();
  if ($("legendMode")) {
    $("legendMode").innerHTML = (viewMode() === "comparison" && SUBSETS.length)
      ? " &nbsp;·&nbsp; <b>comparison</b>: one bar per subset (" +
        SUBSETS.map((s) => '<span style="color:' + subsetColor(s) + '">' + escapeHTML(shortLabel(s)) + "</span>").join(", ") +
        "); each leaf is tagged with its subset"
      : (SUBSETS.length ? " &nbsp;·&nbsp; switch <b>view</b> to <i>comparison</i> to split each bar by subset" : "");
  }
  applyDepth();
  applyLeafVisibility();
}

function applyDepth() {
  const max = parseInt($("depth").value, 10);
  document.querySelectorAll("#tree li.internal").forEach((li) => {
    li.classList.toggle("collapsed", parseInt(li.dataset.depth, 10) >= max);
  });
}
$("depth").addEventListener("change", applyDepth);
$("metric").addEventListener("change", render);
if ($("mode")) $("mode").addEventListener("change", render);

function applyLeafVisibility() {
  const show = $("showLeaves").checked;
  document.querySelectorAll("#tree li.leaf").forEach((li) => li.classList.toggle("hidden", !show));
}
$("showLeaves").addEventListener("change", applyLeafVisibility);

let searchTimer = null;
$("search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => filter(e.target.value.trim().toLowerCase()), 120);
});
function filter(q) {
  document.querySelectorAll("#tree mark").forEach((m) => m.replaceWith(m.textContent));
  const internals = document.querySelectorAll("#tree li.internal");
  if (!q) { applyDepth(); document.querySelectorAll("#tree li").forEach((li) => li.classList.remove("hidden")); applyLeafVisibility(); return; }
  document.querySelectorAll("#tree li").forEach((li) => li.classList.add("hidden"));
  internals.forEach((li) => {
    if ((li.dataset.text || "").includes(q)) {
      li.classList.remove("hidden", "collapsed");
      const desc = li.querySelector(":scope > .row > .desc");
      const t = desc.textContent, i = t.toLowerCase().indexOf(q);
      if (i >= 0) desc.innerHTML = escapeHTML(t.slice(0, i)) + "<mark>" + escapeHTML(t.slice(i, i + q.length)) + "</mark>" + escapeHTML(t.slice(i + q.length));
      let p = li.parentElement;
      while (p && p.id !== "tree") { if (p.tagName === "LI") p.classList.remove("hidden", "collapsed"); p = p.parentElement; }
    }
  });
}

render();
</script>
</body>
</html>
"""


def mode_control_html(subsets):
    """The 'view' <select> (full vs comparison), or '' if there's nothing to compare."""
    if len(subsets) < 2:
        return ""
    return ('<div class="ctl" title="full = combined; comparison = per-subset side by side">'
            'view <select id="mode">'
            '<option value="full" selected>full</option>'
            '<option value="comparison">comparison</option>'
            '</select></div>')


def build_html(tree, title, header, subsets=()):
    tree_json = json.dumps(tree, ensure_ascii=False, separators=(",", ":"))
    # Guard against "</script>" in any description prematurely closing the tag.
    tree_json = tree_json.replace("</", "<\\/")
    subsets_json = json.dumps(list(subsets) if len(subsets) >= 2 else [])
    return (
        HTML_TEMPLATE
        .replace("__TREE_JSON__", tree_json)
        .replace("__TITLE__", title)
        .replace("__HEADER__", header)
        .replace("__MODE_CONTROL__", mode_control_html(subsets))
        .replace("__SUBSETS_JSON__", subsets_json)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="Datasets/DRChallenge", help="dataset directory (default: Datasets/DRChallenge)")
    ap.add_argument("--out-dir", default=None, help="output dir for the .html files (default: <dataset>/EvalTree/viewers)")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else dataset / "EvalTree" / "viewers"
    out_dir.mkdir(parents=True, exist_ok=True)

    tree_dir = dataset / "EvalTree" / "stage3-RecursiveClustering"
    trees = sorted(p for p in tree_dir.glob("*.json") if "stage4-CapabilityDescription" in p.name)
    if not trees:
        raise SystemExit(f"No stage-4 trees found in {tree_dir}")

    # Map each dataset index -> its generation subset (source_run), so the viewer can
    # break FAIL_RATE down per subset (e.g. compare "explore" vs "original" prompts).
    # Only emit the subset control when more than one subset is actually present.
    idx_to_run, subsets = {}, []
    dataset_json = dataset / "dataset.json"
    if dataset_json.is_file():
        instances = json.load(open(dataset_json))
        idx_to_run = {i: (inst.get("source_run") or "unknown") for i, inst in enumerate(instances)}
        found = sorted(set(idx_to_run.values()))
        if len(found) > 1:
            subsets = found

    # Each evaluated model = a results.json under eval_results/.../<model>/.
    results_files = sorted((dataset / "eval_results").rglob("results.json"))
    if not results_files:
        raise SystemExit(f"No results.json found under {dataset / 'eval_results'}")

    built, skipped = [], []
    for res_path in results_files:
        eval_model = res_path.parent.name
        results = json.load(open(res_path))
        if not isinstance(results, list):
            print(f"  skip {res_path}: not a list")
            continue
        for tree_path in trees:
            tree = json.load(open(tree_path))
            desc_model, annot, cluster = describe_tree(tree_path)
            # Include the leaf-annotation source AND clustering mode in the filename
            # so different tree variants (LLM capability vs. "strategy" leaves,
            # flat vs. hierarchical clustering) don't overwrite one another.
            stem = "-".join(p for p in ("viewer", eval_model, annot, cluster, desc_model) if p)
            out = out_dir / f"{stem}.html"

            # A tree built over a different (usually older/larger) dataset references
            # leaf indices that don't exist in the current results — those leaves would
            # render untagged and inflate the leaf count. Refuse to build a misleading
            # viewer, and delete any stale one this tree produced on a previous run.
            mx = max_leaf_index(tree)
            if mx >= len(results):
                if out.exists():
                    out.unlink()
                    note = " (removed stale viewer)"
                else:
                    note = ""
                print(f"  SKIP {tree_path.name}: leaf index {mx} >= {len(results)} results — "
                      f"tree is stale; rebuild stages 2-4 against the current dataset{note}")
                skipped.append(tree_path.name)
                continue

            fail, scored = annotate(tree, results, idx_to_run, subsets)
            overall = f"{fail}/{scored} = {round(100 * fail / scored) if scored else 0}%"
            title = f"{eval_model} · leaves:{annot} · {cluster} · tree:{desc_model}"
            subset_note = f" · subsets: {', '.join(subsets)}" if subsets else ""
            header = (f"EvalTree — <b>{eval_model}</b> &nbsp;FAIL_RATE {overall}"
                      f" &nbsp;<span style='color:var(--muted)'>(leaf labels: {annot} · clustering: {cluster} · descriptions: {desc_model}{subset_note})</span>")
            html = build_html(tree, title, header, subsets)
            out.write_text(html, encoding="utf-8")
            built.append((out, overall))
            print(f"  built {out}  (overall FAIL_RATE {overall})")

    print(f"\nDone — {len(built)} viewer(s) in {out_dir}"
          + (f"; {len(skipped)} stale tree(s) skipped" if skipped else ""))
    for out, _ in built:
        print(f"  open {out}")


if __name__ == "__main__":
    main()

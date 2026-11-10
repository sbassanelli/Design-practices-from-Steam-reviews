"""
PHASE 2 — Sub-theme codebook + harmonization

  A. CODEBOOK, two modes (config: harmonization.codebook_mode)
     - "inductive": per theme, (1) the initial codes of both encoders are embedded and clustered by meaning;
       (2) the LLM names at most max_subthemes_per_theme sub-themes from the clusters; (3) the codebook is TESTED:
       two different models assign a sample of codes, measuring balance, coverage and agreement (Cohen's kappa);
       (4) if thresholds are not met, the LLM revises the sub-themes using the diagnosis (up to max_revision_rounds).
       The best version and the diagnostics of every round are saved; the codebook is reused while the encoded
       dataset does not change (delete the file to regenerate it, or edit it by hand).
     - "deductive": the sub-themes are read from a codebook supplied by the researcher
       (harmonization.deductive_codebook, see codebook_template.json).
  B. HARMONIZATION: the snippets of the two encoders are matched; for each candidate the adjudicator model decides
     KEEP/REJECT, chooses the final snippet (by reference, so it stays verbatim), writes one code, assigns the
     sentiment and a sub-theme (no "Other": always the closest one) with a self-declared fit (strong/moderate/weak).
  C. SUMMARY: one row per sub-theme with counts, polarity, fit and examples.

Usage:  python 2_harmonization.py [--config study_config.json] [--sample N]
        (--sample limits harmonization only; an inductive codebook always uses all codes)
"""
import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

import common as C

FIT_LEVELS = ["strong", "moderate", "weak"]
POLARITIES = ["Positive", "Neutral", "Negative"]


# =====================================================================
# A. CODEBOOK
# =====================================================================

def theme_prefixes(cfg: C.Config) -> Dict[str, str]:
    """Shortest unique uppercase prefix per theme, used for sub-theme ids (e.g. MECHANICS → M1, M2...)."""
    names = {t: re.sub(r"[^A-Za-z0-9]", "", t).upper() or t for t in cfg.themes}
    for length in range(1, max(len(n) for n in names.values()) + 2):
        prefixes = {t: n[:length] for t, n in names.items()}
        if len(set(prefixes.values())) == len(prefixes):
            return prefixes
    return {t: f"T{i}_" for i, t in enumerate(cfg.themes, start=1)}


def collect_codes(cfg: C.Config, data: List[Dict[str, Any]]) -> Dict[str, Counter]:
    codes = {t: Counter() for t in cfg.themes}
    for item in data:
        for e in ("enc1", "enc2"):
            for t in cfg.themes:
                for s in item.get(e, {}).get(t, []):
                    codes[t][s["code"].lower().strip()] += 1
    return codes


def dataset_fingerprint(cfg: C.Config, data: List[Dict[str, Any]]) -> str:
    codes = collect_codes(cfg, data)
    payload = json.dumps({t: sorted(codes[t].items()) for t in cfg.themes}, ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def codebook_hash(codebook: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(codebook["themes"], sort_keys=True).encode()).hexdigest()[:12]


# --- A1. Embedding-based clustering of the initial codes (deterministic) ---

def spherical_kmeans(X: np.ndarray, weights: np.ndarray, k: int, seed: int,
                     n_init: int = 5, iters: int = 100) -> np.ndarray:
    """Weighted k-means on L2-normalised vectors (cosine similarity), k-means++ init, best of n_init runs."""
    with np.errstate(all="ignore"):    # numpy 2 on Apple Silicon emits spurious matmul warnings; results are exact
        return _spherical_kmeans(X, weights, k, seed, n_init, iters)


def _spherical_kmeans(X: np.ndarray, weights: np.ndarray, k: int, seed: int, n_init: int, iters: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_labels, best_score = None, -np.inf
    p_w = weights / weights.sum()
    for _ in range(n_init):
        centroids = [X[rng.choice(len(X), p=p_w)]]
        for _ in range(1, k):                       # k-means++ (weighted by code frequency)
            dist = 1 - np.max(X @ np.array(centroids).T, axis=1)
            p = weights * np.clip(dist, 0, None) ** 2
            centroids.append(X[rng.choice(len(X), p=p / p.sum())] if p.sum() > 0 else X[rng.integers(len(X))])
        centroids = np.array(centroids)
        for _ in range(iters):
            labels = np.argmax(X @ centroids.T, axis=1)
            new = np.array([(X[labels == j] * weights[labels == j, None]).sum(0) if (labels == j).any()
                            else centroids[j] for j in range(k)])
            new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-12
            if np.allclose(new, centroids):
                break
            centroids = new
        score = float((weights * np.max(X @ centroids.T, axis=1)).sum())
        if score > best_score:
            best_labels, best_score = labels, score
    return best_labels


async def cluster_codes(cfg: C.Config, client: httpx.AsyncClient, codes: Counter) -> List[Dict[str, Any]]:
    """Groups the codes of one theme into clusters of similar meaning; returns them largest first."""
    ind = cfg["harmonization"]["codebook_induction"]
    names = list(codes)
    freqs = np.array([codes[c] for c in names], dtype=float)
    X = np.array(await C.embed(cfg, client, ind["embedding_model"], names, prefix=ind.get("embedding_prefix", "")))
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    k = max(1, min(ind["clusters_per_theme"], len(names) // 3))
    labels = spherical_kmeans(X, freqs, k, cfg["ollama"]["seed"])
    total = freqs.sum()
    clusters = []
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        top = idx[np.argsort(-freqs[idx])][:ind["codes_per_cluster_shown"]]
        clusters.append({"share_%": round(100 * freqs[idx].sum() / total, 1),
                         "n_codes": int(len(idx)),
                         "codes": [f"{names[i]} ({int(freqs[i])})" for i in top]})
    clusters.sort(key=lambda c: -c["share_%"])
    for n, c in enumerate(clusters, start=1):
        c["cluster"] = n
    return clusters


# --- A2. Naming the sub-themes from the clusters ---

def subtheme_schema(max_items: int) -> Dict[str, Any]:
    return {"type": "object", "properties": {"subthemes": {
        "type": "array", "minItems": 1, "maxItems": max_items,     # hard limit: also prevents looping outputs
        "items": {"type": "object",
                  "properties": {"name": {"type": "string"}, "definition": {"type": "string"},
                                 "example_codes": {"type": "array", "items": {"type": "string"}, "maxItems": 6}},
                  "required": ["name", "definition", "example_codes"]}}},
        "required": ["subthemes"]}


CODEBOOK_RULES = """- At most {n} sub-themes. Each sub-theme = ONE distinct aspect (no "X and Y" combining two different aspects).
- NO OVERLAP: two sub-themes must never describe the same thing with different words.
- No umbrella sub-theme that contains the others.
- EXHAUSTIVE: there is no "Other" category, so together the sub-themes must cover all the codes.
  Generic statements of the theme (e.g. a general feeling without a specific cause) also need a sub-theme if they are frequent.
- Each sub-theme covers both positive and negative evaluations of the same aspect.
- Each: "name" (2-5 words, English), "definition" (1-2 sentences stating what is included AND where its boundary
  with the neighbouring sub-themes lies, so another coder can apply it), and 3-6 "example_codes" copied from the data."""

NAME_PROMPT = """You are a qualitative researcher doing reflexive thematic analysis (Braun & Clarke) of {context}.
The initial codes assigned under the theme {theme} were grouped automatically, by similarity of meaning, into the
CLUSTERS below (largest first; share_% = share of all code occurrences of the theme; frequencies in brackets).
{theme_def}

Define the SUB-THEMES of {theme} that best represent these clusters:
- A large cluster may deserve its own sub-theme; small related clusters should be merged.
- Choose sub-themes so that each one receives a substantial share of the codes: avoid sub-themes with almost no codes
  and avoid one sub-theme absorbing most of them.
{rules}"""

REVISE_PROMPT = """You are a qualitative researcher doing reflexive thematic analysis (Braun & Clarke) of {context}.
The sub-theme codebook of the theme {theme} was TESTED: two independent coders assigned a sample of codes to it.
{theme_def}

CURRENT SUB-THEMES:
{current}

PROBLEMS FOUND IN THE TEST:
{problems}

Revise the sub-themes to solve these problems:
- too broad → split it into more specific sub-themes, or narrow its definition;
- too marginal → merge it into a related sub-theme, or replace it with a missing pattern;
- coders disagree between two sub-themes → sharpen the boundary in both definitions, or merge them if they are the same thing;
- codes fitting no sub-theme → add a sub-theme for them or broaden the closest definition.
Change ONLY the sub-themes involved in the problems: every other sub-theme must be returned unchanged
(same name and definition). Return the COMPLETE revised set.
{rules}"""


async def name_subthemes(cfg: C.Config, client: httpx.AsyncClient, theme: str,
                         clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = cfg["harmonization"]["max_subthemes_per_theme"]
    prompt = NAME_PROMPT.format(context=cfg["study"]["context"], theme=theme, theme_def=C.theme_definition(cfg, theme),
                                rules=CODEBOOK_RULES.format(n=n))
    res = await C.call_llm(cfg, client, cfg["models"]["harmonizer"], [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "CLUSTERS:\n" + json.dumps(clusters, ensure_ascii=False, indent=1)},
    ], subtheme_schema(n), num_ctx=cfg["harmonization"]["num_ctx"])
    if res["error"]:
        raise RuntimeError(f"Sub-theme naming failed for {theme}: {res['error']}")
    return clean_subthemes(res["raw"].get("subthemes", []), n)


def clean_subthemes(subs: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    out = [{"name": s["name"].strip(), "definition": s["definition"].strip(), "example_codes": s.get("example_codes", [])[:6]}
           for s in subs if s.get("name") and s.get("definition")]
    return out[:n]


def with_ids(prefix: str, subs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"id": f"{prefix}{n}", **s} for n, s in enumerate(subs, start=1)]


# --- A3. Testing the codebook: two coders assign a sample of codes ---

def kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


ASSIGN_PROMPT = """You assign initial codes from a thematic analysis of {context} to the sub-themes of the theme {theme}.
SUB-THEMES:
{subs}
For each code choose the sub-theme whose definition fits it. Answer "NONE" only if no sub-theme fits at all.
Return exactly one result per code id."""


async def assign_sample(cfg: C.Config, client: httpx.AsyncClient, model: str, theme: str,
                        subs: List[Dict[str, Any]], sample: List[str]) -> List[str]:
    ind = cfg["harmonization"]["codebook_induction"]
    ids = [s["id"] for s in subs]
    schema = {"type": "object", "properties": {"results": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "subtheme": {"type": "string", "enum": ids + ["NONE"]}},
        "required": ["id", "subtheme"]}}}, "required": ["results"]}
    prompt = ASSIGN_PROMPT.format(context=cfg["study"]["context"], theme=theme,
                                  subs="\n".join(f"{s['id']} = {s['name']}: {s['definition']}" for s in subs))
    out: List[str] = []
    for start in range(0, len(sample), ind["evaluation_batch"]):
        batch = sample[start:start + ind["evaluation_batch"]]
        res = await C.call_llm(cfg, client, model, [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "CODES:\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(batch, start=1))},
        ], schema)
        got = {r.get("id"): r.get("subtheme") for r in res["raw"].get("results", []) if isinstance(r, dict)}
        out += [got.get(i) if got.get(i) in ids + ["NONE"] else "NONE" for i in range(1, len(batch) + 1)]
    return out


def diagnose(cfg: C.Config, subs: List[Dict[str, Any]], sample: List[str],
             a: List[str], b: List[str]) -> Dict[str, Any]:
    """Coverage, balance and inter-coder agreement of one theme's sub-themes; lists the problems found."""
    th = cfg["harmonization"]["codebook_induction"]["thresholds"]
    names = {s["id"]: s["name"] for s in subs}
    both = a + b
    shares = {sid: both.count(sid) / len(both) for sid in names}
    unassigned = both.count("NONE") / len(both)
    k = kappa(a, b)
    problems = []
    merged = {s["id"] for s in subs if s.get("merged_from")}
    for sid, share in shares.items():
        if len(subs) > 1 and share > th["max_share"] and sid not in merged:   # merged = accepted as broad
            ex = [c for c, x, y in zip(sample, a, b) if x == sid and y == sid][:15]
            problems.append(f'{sid} "{names[sid]}" is TOO BROAD: it receives {share:.0%} of the codes. Codes it contains: {ex}')
        if len(subs) > 1 and share < th["min_share"]:
            problems.append(f'{sid} "{names[sid]}" is TOO MARGINAL: it receives only {share:.0%} of the codes.')
    if unassigned > th["max_unassigned"]:
        ex = [c for c, x, y in zip(sample, a, b) if x == "NONE" or y == "NONE"][:15]
        problems.append(f"{unassigned:.0%} of the assignments found NO fitting sub-theme. Examples: {ex}")
    if k < th["min_kappa"]:
        pairs = Counter(tuple(sorted((x, y))) for x, y in zip(a, b) if x != y and "NONE" not in (x, y))
        for (x, y), cnt in pairs.most_common(3):
            ex = [c for c, p, q in zip(sample, a, b) if {p, q} == {x, y}][:8]
            problems.append(f'Coders DISAGREE between {x} "{names[x]}" and {y} "{names[y]}" ({cnt} codes), e.g.: {ex}')
    # Pairwise confusion: among the codes that either coder put in A or B, share split between A and B
    confusion = {}
    ids = list(names)
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            involved = sum(1 for p, q in zip(a, b) if {p, q} & {x, y})
            split = sum(1 for p, q in zip(a, b) if {p, q} == {x, y})
            confusion[f"{x}|{y}"] = round(split / involved, 3) if involved else 0.0
    # Failed criteria (not problem lines: a low kappa produces several "disagree" lines but is ONE failure)
    failures = (sum(1 for sid, sh in shares.items() if len(subs) > 1 and sh > th["max_share"] and sid not in merged)
                + sum(1 for sh in shares.values() if len(subs) > 1 and sh < th["min_share"])
                + int(unassigned > th["max_unassigned"]) + int(k < th["min_kappa"]))
    return {"kappa": round(k, 3), "unassigned_%": round(100 * unassigned, 1),
            "shares_%": {sid: round(100 * s, 1) for sid, s in shares.items()},
            "pair_confusion": confusion, "failed_criteria": failures, "problems": problems}


MERGE_PROMPT = """You are a qualitative researcher doing reflexive thematic analysis (Braun & Clarke) of {context}.
Two sub-themes of the theme {theme} were found to be NOT SEPARABLE: independent coders cannot tell them apart
and/or their definitions are near-identical.
{theme_def}

SUB-THEME A: {a}
SUB-THEME B: {b}
Codes the coders split between them: {examples}

Write ONE sub-theme that replaces both: a "name" (2-5 words, English) covering the shared meaning, a "definition"
(1-2 sentences) that includes everything A and B included, and 3-6 "example_codes"."""


async def subtheme_similarity(cfg: C.Config, client: httpx.AsyncClient, subs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Cosine similarity between the definitions of every pair of sub-themes."""
    ind = cfg["harmonization"]["codebook_induction"]
    vecs = np.array(await C.embed(cfg, client, ind["embedding_model"], [f"{s['name']}: {s['definition']}" for s in subs],
                                  prefix=ind.get("embedding_prefix", "")))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    with np.errstate(all="ignore"):
        sim = vecs @ vecs.T
    return {f"{x['id']}|{y['id']}": round(float(sim[i, j]), 3)
            for i, x in enumerate(subs) for j, y in enumerate(subs) if j > i}


def find_merges(cfg: C.Config, subs: List[Dict[str, Any]], diag: Dict[str, Any],
                similarity: Dict[str, float], rejected: set = frozenset()) -> List[Dict[str, Any]]:
    """Candidate merges: pairs above the confusion OR similarity threshold, strongest first, without overlapping
    pairs and without pairs whose merge was already tested and rejected."""
    th = cfg["harmonization"]["codebook_induction"]["thresholds"]
    names = {s["id"]: s["name"] for s in subs}
    pairs = []
    for key, sim in similarity.items():
        conf = diag["pair_confusion"].get(key, 0.0)
        if frozenset(names[i] for i in key.split("|")) in rejected:
            continue
        if conf >= th["merge_confusion"] or sim >= th["merge_similarity"]:
            pairs.append({"pair": key.split("|"), "confusion": conf, "similarity": sim})
    pairs.sort(key=lambda p: (-p["confusion"], -p["similarity"]))
    chosen, used, remaining = [], set(), len(subs)
    for p in pairs:
        if remaining <= th["min_subthemes_per_theme"]:
            break
        if not used & set(p["pair"]):
            chosen.append(p)
            used |= set(p["pair"])
            remaining -= 1
    return chosen


async def merge_subthemes(cfg: C.Config, client: httpx.AsyncClient, theme: str, subs: List[Dict[str, Any]],
                          merges: List[Dict[str, Any]], sample: List[str], a: List[str], b: List[str]) -> List[Dict[str, Any]]:
    by_id = {s["id"]: s for s in subs}
    out, replaced = [], {}
    for m in merges:
        x, y = (by_id[i] for i in m["pair"])
        examples = [c for c, p, q in zip(sample, a, b) if {p, q} == {x["id"], y["id"]}][:10]
        prompt = MERGE_PROMPT.format(context=cfg["study"]["context"], theme=theme, theme_def=C.theme_definition(cfg, theme),
                                     a=f"{x['name']}: {x['definition']}", b=f"{y['name']}: {y['definition']}",
                                     examples=examples)
        res = await C.call_llm(cfg, client, cfg["models"]["harmonizer"], [{"role": "system", "content": prompt}],
                               subtheme_schema(1), num_ctx=cfg["harmonization"]["num_ctx"])
        new = clean_subthemes(res["raw"].get("subthemes", []), 1) if not res["error"] else []
        if not new:
            tqdm.write(f"⚠️  Merge of {x['name']} + {y['name']} failed: kept separate")
            continue
        origin = x.get("merged_from", [x["name"]]) + y.get("merged_from", [y["name"]])
        replaced[x["id"]] = {**new[0], "merged_from": origin, "merge_evidence": m}
        replaced[y["id"]] = None
    for s in subs:                          # merged sub-theme takes the place of the first of the pair
        if s["id"] in replaced:
            if replaced[s["id"]] is not None:
                out.append(replaced[s["id"]])
        else:
            out.append({k: v for k, v in s.items() if k != "id"})
    return out


async def revise_subthemes(cfg: C.Config, client: httpx.AsyncClient, theme: str,
                           subs: List[Dict[str, Any]], diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = cfg["harmonization"]["max_subthemes_per_theme"]
    current = "\n".join(
        f"{s['id']} = {s['name']}: {s['definition']}"
        + (" [MERGED because coders could not separate its parts: keep it as ONE sub-theme, same name]"
           if s.get("merged_from") else "")
        for s in subs)
    prompt = REVISE_PROMPT.format(context=cfg["study"]["context"], theme=theme, theme_def=C.theme_definition(cfg, theme),
                                  current=current, problems="\n".join(f"- {p}" for p in diagnosis["problems"]),
                                  rules=CODEBOOK_RULES.format(n=n))
    res = await C.call_llm(cfg, client, cfg["models"]["harmonizer"], [{"role": "system", "content": prompt}],
                           subtheme_schema(n), num_ctx=cfg["harmonization"]["num_ctx"])
    if res["error"]:
        tqdm.write(f"⚠️  Revision failed for {theme}: {res['error']}")
        return []
    revised = clean_subthemes(res["raw"].get("subthemes", []), n)
    provenance = {s["name"].lower(): s for s in subs if s.get("merged_from")}
    for r in revised:                       # merged sub-themes kept by name keep their provenance
        if r["name"].lower() in provenance:
            old = provenance[r["name"].lower()]
            r.update(merged_from=old["merged_from"], merge_evidence=old.get("merge_evidence"))
    return revised


# --- A4. Orchestration: cluster → name → (test → revise)* → keep the best version ---

async def generate_codebook(cfg: C.Config, client: httpx.AsyncClient, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    h, ind = cfg["harmonization"], cfg["harmonization"]["codebook_induction"]
    codes, prefixes = collect_codes(cfg, data), theme_prefixes(cfg)
    rng = np.random.default_rng(cfg["ollama"]["seed"])
    codebook = {"mode": "inductive", "model": cfg["models"]["harmonizer"], "context": cfg["study"]["context"],
                "procedure": "embedding clustering → LLM naming → test by two coders → merge of inseparable sub-themes / LLM revision",
                "embedding_model": ind["embedding_model"], "evaluation_models": ind["evaluation_models"],
                "max_subthemes_per_theme": h["max_subthemes_per_theme"], "dataset": dataset_fingerprint(cfg, data),
                "themes": {}, "diagnostics": {}}

    # 1. Cluster and name (per theme)
    current, samples, history = {}, {}, {t: [] for t in cfg.themes}
    for theme in tqdm(cfg.themes, desc="Codebook: clustering + naming"):
        if not codes[theme]:
            tqdm.write(f"⚠️  No codes for {theme}: the theme gets no sub-themes.")
            continue
        clusters = await cluster_codes(cfg, client, codes[theme])
        current[theme] = with_ids(prefixes[theme], await name_subthemes(cfg, client, theme, clusters))
        names, freqs = list(codes[theme]), np.array(list(codes[theme].values()), dtype=float)
        size = min(ind["evaluation_sample"], len(names))
        samples[theme] = [names[i] for i in rng.choice(len(names), size=size, replace=False, p=freqs / freqs.sum())]

    # 2. Test → revise loop. Evaluation models run one after the other (one model in memory at a time).
    best = {t: None for t in current}
    pre_merge: Dict[str, Dict[str, Any]] = {}            # version before a merge, to revert it if it does not help
    rejected: Dict[str, set] = {t: set() for t in current}
    by_name = lambda subs, ids: [s["name"] for s in subs if s["id"] in ids]
    to_test = [t for t in current if len(current[t]) > 1]
    for theme in current:                  # a single sub-theme cannot be tested: accepted as is
        if theme not in to_test:
            best[theme] = ((0, 0), current[theme])
            history[theme].append({"round": 0, "subthemes": current[theme], "kappa": None,
                                   "unassigned_%": None, "shares_%": {}, "problems": []})
    for round_n in range(ind["max_revision_rounds"] + 1):
        if not to_test:
            break
        assignments = {t: [] for t in to_test}
        for model in ind["evaluation_models"]:
            for theme in tqdm(to_test, desc=f"Codebook test round {round_n} ({model})"):
                assignments[theme].append(await assign_sample(cfg, client, model, theme, current[theme], samples[theme]))
            await C.unload_model(cfg, client, model)

        actions = {}
        for theme in to_test:              # only themes tested in this round (the others already passed)
            a, b = assignments[theme][:2]
            diag = diagnose(cfg, current[theme], samples[theme], a, b)
            diag["definition_similarity"] = await subtheme_similarity(cfg, client, current[theme])
            diag["merges"] = find_merges(cfg, current[theme], diag, diag["definition_similarity"], rejected[theme])
            history[theme].append({"round": round_n, "subthemes": current[theme], **diag})
            score = (diag["failed_criteria"], -(diag["kappa"] or 0))
            pre = pre_merge.pop(theme, None)
            if pre is not None and score > pre["score"]:
                # The merge made the codebook worse on the data: revert it and never propose it again
                rejected[theme] |= pre["pairs"]
                history[theme][-1]["merge_rejected"] = True
                tqdm.write(f"   {theme} round {round_n}: kappa={diag['kappa']} → merge REJECTED (test worsened), reverted")
                current[theme], diag = pre["subthemes"], pre["diag"]
                diag["merges"] = find_merges(cfg, current[theme], diag, diag["definition_similarity"], rejected[theme])
                score = pre["score"]
            if best[theme] is None or (score[0] + len(diag["merges"]), score[1]) < best[theme][0]:
                # Pending merges count as unresolved issues: a version with inseparable sub-themes is not final
                best[theme] = ((score[0] + len(diag["merges"]), score[1]), current[theme])
            merges_txt = ", ".join("+".join(m["pair"]) for m in diag["merges"])
            status = ("OK" if score[0] == 0 and not diag["merges"] else
                      f"{diag['failed_criteria']} failed criteria" + (f", merge {merges_txt}" if merges_txt else ""))
            tqdm.write(f"   {theme} round {round_n}: kappa={diag['kappa']} unassigned={diag['unassigned_%']}% → {status}")
            if diag["merges"]:              # 1st: merge what the coders cannot separate (kept only if it improves)
                pre_merge[theme] = {"subthemes": current[theme], "diag": diag, "score": score,
                                    "pairs": {frozenset(by_name(current[theme], m["pair"])) for m in diag["merges"]}}
                actions[theme] = ("merge", diag["merges"], a, b)
            elif diag["problems"]:          # 2nd: revise the remaining problems
                actions[theme] = ("revise", diag)

        if not actions or round_n == ind["max_revision_rounds"]:
            break
        to_test = []
        for theme, action in tqdm(actions.items(), desc=f"Codebook merge/revision {round_n + 1}"):
            if action[0] == "merge":
                updated = await merge_subthemes(cfg, client, theme, current[theme], action[1],
                                                samples[theme], action[2], action[3])
            else:
                updated = await revise_subthemes(cfg, client, theme, current[theme], action[1])
            if updated and len(updated) > 1:
                current[theme] = with_ids(prefixes[theme], updated)
                to_test.append(theme)

    for theme in cfg.themes:
        codebook["themes"][theme] = best[theme][1] if best.get(theme) else []
        codebook["diagnostics"][theme] = history[theme]
    return codebook


def load_deductive_codebook(cfg: C.Config) -> Dict[str, Any]:
    """Reads and validates a researcher-supplied codebook; assigns ids where missing."""
    path = cfg.path("deductive_codebook")
    supplied = C.load_json(path)
    themes = supplied.get("themes", {})
    unknown = set(themes) - set(cfg.themes)
    missing = [t for t in cfg.themes if not themes.get(t)]
    if unknown:
        raise ValueError(f"Deductive codebook contains themes not in the config: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Deductive codebook needs at least one sub-theme for every theme; missing: {missing}")

    prefixes = theme_prefixes(cfg)
    codebook = {"mode": "deductive", "source": os.path.basename(path), "themes": {}}
    for theme in cfg.themes:
        subs = []
        for n, s in enumerate(themes[theme], start=1):
            if not s.get("name") or not s.get("definition"):
                raise ValueError(f"Every sub-theme of {theme} needs 'name' and 'definition'")
            subs.append({"id": s.get("id") or f"{prefixes[theme]}{n}", "name": s["name"].strip(),
                         "definition": s["definition"].strip(), "example_codes": s.get("example_codes", [])})
        codebook["themes"][theme] = subs
    ids = [s["id"] for subs in codebook["themes"].values() for s in subs]
    if len(ids) != len(set(ids)):
        raise ValueError("Sub-theme ids in the deductive codebook must be unique")
    return codebook


async def get_codebook(cfg: C.Config, client: httpx.AsyncClient, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    path = cfg.path("codebook")
    if cfg["harmonization"]["codebook_mode"] == "deductive":
        codebook = load_deductive_codebook(cfg)
        C.save_json(path, codebook)      # copy of the codebook actually applied, next to the results
        print(f"📘 Deductive codebook loaded from {cfg.path('deductive_codebook')}")
        return codebook

    if os.path.exists(path):
        codebook = C.load_json(path)
        if codebook.get("mode") == "inductive" and codebook.get("dataset") == dataset_fingerprint(cfg, data):
            print(f"📘 Existing inductive codebook reused: {path} (delete it to regenerate)")
            return codebook
        print("⚠️  Existing codebook belongs to another dataset or mode: regenerating it.")
    print(f"\n🧩 PHASE 2A: Generating the inductive sub-theme codebook | {cfg['models']['harmonizer']}")
    codebook = await generate_codebook(cfg, client, data)
    C.save_json(path, codebook)
    return codebook


def print_codebook(codebook: Dict[str, Any]):
    print(f"\n📘 SUB-THEME CODEBOOK ({codebook['mode']})")
    for theme, subs in codebook["themes"].items():
        print(f"  {theme}")
        for s in subs:
            print(f"    {s['id']:4} {s['name']} — {s['definition']}")
    print()


def format_subthemes(codebook: Dict[str, Any]) -> str:
    lines = []
    for theme, subs in codebook["themes"].items():
        lines.append(f"{theme}:")
        lines += [f"  {s['id']} = {s['name']}: {s['definition']}" for s in subs]
    return "\n".join(lines)


# =====================================================================
# B. HARMONIZATION
# =====================================================================

# --- B1. Deterministic candidate matching ---

def tokens(text: str) -> set:
    norm = C.normalize(text)
    if C.is_non_space_script(norm):
        return {c for c in norm if not c.isspace() and c.isalnum()}
    return set(re.findall(r"\w+", norm))


def overlap(a: str, b: str) -> float:
    na, nb = C.normalize(a), C.normalize(b)
    if na == nb or na in nb or nb in na:
        return 1.0
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0


def build_candidates(cfg: C.Config, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per theme, pairs the snippets of enc1 and enc2 that quote the same passage.
    Each candidate has one proposal (single encoder) or two (agreement)."""
    threshold = cfg["harmonization"]["match_threshold"]
    candidates = []
    for theme in cfg.themes:
        clusters = [{"theme": theme, "enc1": s, "enc2": None} for s in item.get("enc1", {}).get(theme, [])]
        for s2 in item.get("enc2", {}).get(theme, []):
            best, best_score = None, 0.0
            for c in clusters:
                if c["enc2"] is None and c["enc1"] is not None:
                    score = overlap(c["enc1"]["snippet"], s2["snippet"])
                    if score > best_score:
                        best, best_score = c, score
            if best is not None and best_score >= threshold:
                best["enc2"] = s2
            else:
                clusters.append({"theme": theme, "enc1": None, "enc2": s2})
        candidates.extend(clusters)
    for n, c in enumerate(candidates, start=1):
        c["id"] = n
        c["source"] = "both" if c["enc1"] and c["enc2"] else ("enc1" if c["enc1"] else "enc2")
    return candidates


# --- B2. LLM adjudication ---

HARMONIZER_PROMPT = """You are the senior coder (adjudicator) of a reflexive thematic analysis (Braun & Clarke) of {context}.
Two independent coders proposed coded extracts. For each CANDIDATE you produce the single final coded extract.

THEMES:
{themes}

GENERAL RULES OF THE CODEBOOK:
{global_rules}

SUB-THEMES:
{subthemes}

SENTIMENT (of the extract itself, not of the whole text):
{sentiment}

FOR EACH CANDIDATE (judge it ONLY against ITS OWN "theme" field, never against other themes):
1. "justification": max 8 words in English, written FIRST, starting with the candidate theme.
2. "decision": first check the theme's INCLUDES list. If the extract matches an INCLUDES item → "KEEP".
   "REJECT" only if it matches the theme's EXCLUDES, is generic praise/criticism without a topic, or has no real content.
   Candidates proposed by BOTH coders are strong evidence: reject them only if clearly wrong.
3. "snippet_choice": which coder's snippet is the final extract ("enc1" or "enc2"; only a coder present in the proposals).
   Prefer the snippet that is shorter and expresses ONE complete idea about the theme.
4. "final_code": ONE initial code in English (2-6 words) reconciling the coders' codes, stating the specific meaning
   of THIS extract. NEVER copy the sub-theme name as the code: the code is more specific.
5. "subtheme": the id of the sub-theme OF THE CANDIDATE'S THEME that best fits the extract.
   There is no "Other": always choose the closest sub-theme.
6. "subtheme_fit": "strong" = clearly a core example of its definition; "moderate" = fits only partly or indirectly;
   "weak" = assigned only because it is the least distant option. Be honest: weak fits are useful information.
7. "final_sentiment": the sentiment of the final snippet. When the coders disagree, judge the wording yourself.
For REJECT decisions, fill the other fields anyway (any proposal, any sub-theme of the candidate theme).
Return exactly one decision per candidate id.{context_rule}"""

CONTEXT_RULE = """
A <context> block may precede the text (e.g. the message it replies to, or the question it answers):
use it only to understand the text. Extracts always come from the text, never from the context."""

REASSIGN_PROMPT = """You assign coded extracts from {context} to sub-themes of the theme {theme}.
SUB-THEMES:
{subs}
Choose the closest sub-theme for the extract. There is no "Other": always pick one.
Also rate "subtheme_fit": "strong" (core example), "moderate" (partial/indirect fit) or "weak" (only the least distant option)."""


def harmonize_schema(cfg: C.Config, codebook: Dict[str, Any]) -> Dict[str, Any]:
    ids = [s["id"] for subs in codebook["themes"].values() for s in subs]
    return {"type": "object", "properties": {"decisions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "justification": {"type": "string"},
            "decision": {"type": "string", "enum": ["KEEP", "REJECT"]},
            "snippet_choice": {"type": "string", "enum": ["enc1", "enc2"]},
            "final_code": {"type": "string"},
            "subtheme": {"type": "string", "enum": ids},
            "subtheme_fit": {"type": "string", "enum": FIT_LEVELS},
            "final_sentiment": {"type": "string", "enum": cfg.sentiments},
        },
        "required": ["id", "justification", "decision", "snippet_choice", "final_code",
                     "subtheme", "subtheme_fit", "final_sentiment"]}}},
        "required": ["decisions"]}


def harmonizer_messages(cfg: C.Config, text: str, candidates: List[Dict[str, Any]],
                        codebook: Dict[str, Any], context_block: str = "") -> List[Dict[str, str]]:
    listing = [{"id": c["id"], "theme": c["theme"], "proposed_by": c["source"],
                "proposals": [{"coder": k, "snippet": c[k]["snippet"], "code": c[k]["code"],
                               "sentiment": c[k]["sentiment"]} for k in ("enc1", "enc2") if c[k]]}
               for c in candidates]
    system = HARMONIZER_PROMPT.format(context=cfg["study"]["context"], themes=C.format_themes(cfg),
                                      global_rules=C.format_global_rules(cfg),
                                      subthemes=format_subthemes(codebook), sentiment=C.format_sentiment(cfg),
                                      context_rule=CONTEXT_RULE if cfg["fields"].get("context") else "")
    user = f"{context_block}<text>{text}</text>\n\nCANDIDATES:\n{json.dumps(listing, ensure_ascii=False, indent=1)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def harmonize_item(cfg: C.Config, client: httpx.AsyncClient, item: Dict[str, Any],
                         codebook: Dict[str, Any]) -> Dict[str, Any]:
    h = cfg["harmonization"]
    text = C.item_text(cfg, item["original_data"])
    candidates = build_candidates(cfg, item)
    schema = harmonize_schema(cfg, codebook)
    decisions: Dict[int, Dict[str, Any]] = {}
    for start in range(0, len(candidates), h["max_candidates_per_call"]):
        batch = candidates[start:start + h["max_candidates_per_call"]]
        res = await C.call_llm(cfg, client, cfg["models"]["harmonizer"],
                               harmonizer_messages(cfg, text, batch, codebook, C.context_block(cfg, item["original_data"])),
                               schema, num_ctx=h["num_ctx"])
        if res["error"]:
            return {"rows": [], "rejected": [], "error": res["error"]}
        for d in res["raw"].get("decisions", []):
            if isinstance(d, dict) and isinstance(d.get("id"), int):
                decisions[d["id"]] = d
    rows, rejected = finalize(cfg, item, candidates, decisions, codebook)
    await reassign_subthemes(cfg, client, rows, codebook)
    return {"rows": rows, "rejected": rejected, "error": None}


async def reassign_subthemes(cfg: C.Config, client: httpx.AsyncClient, rows: List[Dict[str, Any]],
                             codebook: Dict[str, Any]):
    """Extracts without a valid sub-theme get a dedicated call restricted to the sub-themes of their theme."""
    names = {s["id"]: s["name"] for subs in codebook["themes"].values() for s in subs}
    for r in rows:
        if r["subtheme_id"] is not None:
            continue
        subs = codebook["themes"][r["theme"]]
        schema = {"type": "object",
                  "properties": {"subtheme": {"type": "string", "enum": [s["id"] for s in subs]},
                                 "subtheme_fit": {"type": "string", "enum": FIT_LEVELS}},
                  "required": ["subtheme", "subtheme_fit"]}
        prompt = REASSIGN_PROMPT.format(context=cfg["study"]["context"], theme=r["theme"],
                                        subs="\n".join(f"{s['id']} = {s['name']}: {s['definition']}" for s in subs))
        res = await C.call_llm(cfg, client, cfg["models"]["harmonizer"], [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f'EXTRACT: "{r["snippet"]}"\nCODE: {r["code"]}'},
        ], schema)
        sid = res["raw"].get("subtheme") if not res["error"] else None
        fit = res["raw"].get("subtheme_fit") if not res["error"] else None
        r["subtheme_id"] = sid if sid in names else subs[0]["id"]
        r["subtheme_fit"] = fit if fit in FIT_LEVELS and sid in names else "weak"
        r["subtheme"] = names[r["subtheme_id"]]


# --- B3. Validation of the adjudicator output ---

def finalize(cfg: C.Config, item: Dict[str, Any], candidates: List[Dict[str, Any]],
             decisions: Dict[int, Dict[str, Any]], codebook: Dict[str, Any]):
    original = item["original_data"]
    names = {s["id"]: s["name"] for subs in codebook["themes"].values() for s in subs}
    valid_ids = {t: {s["id"] for s in subs} for t, subs in codebook["themes"].items()}
    rows, rejected, seen = [], [], set()

    for c in candidates:
        d, flags = decisions.get(c["id"]), []
        if d is None:   # not audited by the model: kept only if both encoders agreed
            flags.append("not_audited")
            p = c["enc2"] or c["enc1"]
            d = {"decision": "KEEP" if c["source"] == "both" else "REJECT", "justification": "",
                 "snippet_choice": "enc2" if c["enc2"] else "enc1", "final_code": p["code"],
                 "final_sentiment": p["sentiment"], "subtheme": None}

        trace = {
            "source": c["source"], "justification": d.get("justification", ""),
            **{f"{k}_{f}": (c[k][f] if c[k] else None) for k in ("enc1", "enc2") for f in ("snippet", "code", "sentiment")},
        }
        if d.get("decision") != "KEEP":
            rejected.append({"item_id": item["item_id"], "theme": c["theme"], **trace, "flags": ";".join(flags)})
            continue

        choice = d.get("snippet_choice")        # chosen by reference among verbatim proposals
        if choice in ("enc1", "enc2") and c[choice]:
            snippet = c[choice]["snippet"]
        else:
            snippet = min((c[k]["snippet"] for k in ("enc1", "enc2") if c[k]), key=len)
            flags.append("snippet_fallback")

        code = " ".join(str(d.get("final_code") or "").split()).strip(" .…")
        if not code or len(code.split()) > cfg["encoding"]["max_code_words"]:
            code = (c["enc2"] or c["enc1"])["code"]
            flags.append("code_fallback")

        sentiment = d.get("final_sentiment")
        if sentiment not in cfg.sentiments:
            sentiment = (c["enc2"] or c["enc1"])["sentiment"]
            flags.append("sentiment_fallback")

        subtheme_id = d.get("subtheme")
        if subtheme_id not in valid_ids[c["theme"]]:
            subtheme_id = None                  # missing or from another theme: reassigned by a dedicated call
            flags.append("subtheme_reassigned")

        key = (c["theme"], C.normalize(snippet))
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "item_id": item["item_id"],
            "language": C.item_language(cfg, original),
            **C.item_metadata(cfg, original),
            "theme": c["theme"],
            "subtheme_id": subtheme_id,
            "subtheme": names.get(subtheme_id),
            "subtheme_fit": d.get("subtheme_fit") if d.get("subtheme_fit") in FIT_LEVELS and subtheme_id else None,
            "code": code,
            "sentiment": sentiment,
            "polarity": cfg.polarity[sentiment],
            "snippet": snippet,
            **trace,
            "flags": ";".join(flags),
            "full_text": C.item_text(cfg, original),
            **({"context": C.item_context(cfg, original)} if cfg["fields"].get("context") else {}),
        })
    return rows, rejected


# =====================================================================
# C. SUMMARY
# =====================================================================

def build_summary(cfg: C.Config, df: pd.DataFrame, codebook: Dict[str, Any]) -> pd.DataFrame:
    example_lang = cfg["harmonization"].get("summary_example_language")
    out = []
    for theme, subs in codebook["themes"].items():
        theme_n = max((df["theme"] == theme).sum(), 1)
        for s in subs:
            sub = df[(df["theme"] == theme) & (df["subtheme_id"] == s["id"])]
            n = max(len(sub), 1)
            examples = sub[sub["language"] == example_lang] if example_lang else sub
            out.append({
                "theme": theme, "subtheme_id": s["id"], "subtheme": s["name"], "definition": s["definition"],
                "n_extracts": len(sub), "n_items": sub["item_id"].nunique(),
                "share_of_theme_%": round(100 * len(sub) / theme_n, 1),
                **{f"{p.lower()}_%": round(100 * (sub["polarity"] == p).sum() / n, 1) for p in POLARITIES},
                **{f"fit_{lvl}_%": round(100 * (sub["subtheme_fit"] == lvl).sum() / n, 1) for lvl in FIT_LEVELS},
                "top_codes": "; ".join(sub["code"].str.lower().value_counts().head(5).index),
                "example_snippets": " | ".join(examples["snippet"].head(3)),
            })
    return pd.DataFrame(out)


# =====================================================================
# MAIN
# =====================================================================

async def main():
    args = C.parse_args("Phase 2: codebook + harmonization")
    cfg = C.load_config(args.config)
    h = cfg["harmonization"]
    model = cfg["models"]["harmonizer"]
    data = C.load_json(cfg.path("encoded"))

    async with httpx.AsyncClient() as client:
        codebook = await get_codebook(cfg, client, data)     # always built from the full dataset
        print_codebook(codebook)
        run_key = f"{model}|{codebook_hash(codebook)}|{dataset_fingerprint(cfg, data)}"

        if args.sample:
            data = data[:args.sample]
        results: Dict[int, Dict[str, Any]] = {}
        ck = cfg.path("harmonization_checkpoint")
        if os.path.exists(ck):
            checkpoint = C.load_json(ck)
            if checkpoint.get("run_key") == run_key:
                results = {int(k): v for k, v in checkpoint["results"].items()}
                print(f"✅ Checkpoint: {len(results)}/{len(data)} items already harmonized.")
            else:
                print("⚠️  Checkpoint from a different model/codebook/dataset: ignored.")
        save = lambda: C.save_json(ck, {"run_key": run_key, "results": results})

        print(f"🚀 PHASE 2B: Harmonization | {len(data)} items | {model} | parallel requests: {h['parallel_requests']}")
        print("-" * 60)
        todo = [i for i in range(len(data)) if i not in results or results[i].get("error")]
        semaphore = asyncio.Semaphore(h["parallel_requests"])

        async def process(i: int):
            item = data[i]
            if not build_candidates(cfg, item):
                return i, {"rows": [], "rejected": [], "error": None}
            async with semaphore:
                return i, await harmonize_item(cfg, client, item, codebook)

        tasks = [process(i) for i in todo]
        for n, task in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Harmonizing"), start=1):
            i, res = await task
            results[i] = res
            if n % h["save_every"] == 0:
                save()
        save()
        await C.unload_model(cfg, client, model)

    rows = [r for i in sorted(results) for r in results[i]["rows"]]
    rejected = [r for i in sorted(results) for r in results[i]["rejected"]]
    errors = [data[i]["item_id"] for i in sorted(results) if results[i].get("error")]
    df = pd.DataFrame(rows)
    df.to_csv(cfg.path("final"), index=False, encoding="utf-8-sig")
    pd.DataFrame(rejected).to_csv(cfg.path("rejected"), index=False, encoding="utf-8-sig")

    print("-" * 60)
    print(f"✅ Harmonization complete. {len(df)} coded extracts → {cfg.path('final')}")
    if not df.empty:
        summary = build_summary(cfg, df, codebook)
        summary.to_csv(cfg.path("summary"), index=False, encoding="utf-8-sig")
        print(summary[["theme", "subtheme_id", "subtheme", "n_extracts", "n_items",
                       "positive_%", "negative_%", "fit_weak_%"]].to_string(index=False))
        print("   Sub-theme fit:", df["subtheme_fit"].value_counts().to_dict())
        print("   Source:", df["source"].value_counts().to_dict())
        print("   Flags:", df["flags"].replace("", "none").value_counts().to_dict())
        print(f"📊 Summary → {cfg.path('summary')}")
    print(f"   Rejected candidates: {len(rejected)} → {cfg.path('rejected')}")
    print(f"   Items with LLM errors: {len(errors)} {errors[:10]} (re-run the script to retry them)")
    if not errors and os.path.exists(ck):
        os.remove(ck)


if __name__ == "__main__":
    asyncio.run(main())

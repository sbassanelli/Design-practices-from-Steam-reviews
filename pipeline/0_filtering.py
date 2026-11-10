"""
PHASE 0 — Filtering: deterministic cleaning + content judgement (multilingual LLM)

  1. CLEANING (fixed rules, identical for every language): too short, symbols/ASCII art, exact duplicates.
  2. CONTENT (LLM, optional): an item is kept if it contains at least one SPECIFIC statement
     (criteria in the config). If the model fails, the item is KEPT and flagged (fail-open:
     the encoders and the harmonizer discard anything without thematic content anyway).
Model decisions are cached: re-running the script does not recompute them.

Input: JSON, JSONL or CSV (see README, "Using other data sources").
Usage:  python 0_filtering.py [--config study_config.json] [--sample N]
"""
import asyncio
import os
import re
from collections import Counter
from typing import Any, Dict

import httpx
import pandas as pd
from tqdm import tqdm

import common as C

BBCODE = re.compile(r"\[/?[a-z0-9*]+(=[^\]]*)?\]", re.IGNORECASE)
DENSE_SCRIPT = re.compile(r"[一-鿿㐀-䶿぀-ヿ가-힯฀-๿]")

FILTER_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "informative": {"type": "boolean"}},
    "required": ["reason", "informative"],
}

FILTER_PROMPT = """You screen {context} for a qualitative study. Texts can be in ANY language: judge the meaning, not the language.

A text is INFORMATIVE if: {informative}

A text is NOT INFORMATIVE if: {not_informative}

Write a "reason" of max 10 words in English FIRST, then "informative" true/false.{context_rule}"""

CONTEXT_RULE = """
A <context> block may precede the text (e.g. the message it replies to, or the question it answers):
use it only to understand the text, and judge ONLY the text."""


# --- 1. Deterministic cleaning ---

def clean_text(cfg: C.Config, text: str) -> str:
    """Text used for the length/letter rules and shown to the filter model (whitespace collapsed)."""
    if cfg["filter"].get("strip_bbcode", True):      # Steam-style [h1]…[/h1] tags
        text = BBCODE.sub("", text or "")
    return " ".join((text or "").split())


def effective_length(cfg: C.Config, text: str) -> int:
    dense = len(DENSE_SCRIPT.findall(text))
    return len(text) - dense + cfg["filter"]["dense_script_weight"] * dense


def letter_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in chars) / max(len(chars), 1)


def cleaning_reason(cfg: C.Config, item: Dict[str, Any], seen: set) -> str:
    f = cfg["filter"]
    text = clean_text(cfg, C.item_text(cfg, item))
    if f.get("languages") and C.item_language(cfg, item) not in f["languages"]:
        return "language"
    if effective_length(cfg, text) < f["min_effective_chars"]:
        return "too_short"
    if letter_ratio(text) < f["min_letter_ratio"]:
        return "low_letter_ratio"
    if f.get("remove_duplicates", True):          # off for surveys: identical answers are valid data
        key = text.lower()
        if key in seen:
            return "duplicate"
        seen.add(key)
    return ""


# --- 2. Content judgement ---

async def judge(cfg: C.Config, client: httpx.AsyncClient, text: str, context_block: str = "") -> Dict[str, Any]:
    model = cfg["models"]["filter"]
    prompt = FILTER_PROMPT.format(context=cfg["study"]["context"],
                                  informative=cfg["filter"]["informative_criteria"],
                                  not_informative=cfg["filter"]["not_informative_criteria"],
                                  context_rule=CONTEXT_RULE if cfg["fields"].get("context") else "")
    res = await C.call_llm(cfg, client, model, [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{context_block}<text>{text}</text>"},
    ], FILTER_SCHEMA)
    if res["error"] or not isinstance(res["raw"].get("informative"), bool):
        return {"informative": True, "reason": f"model error: {res['error']}", "error": True, "model": model}
    return {"informative": res["raw"]["informative"], "reason": res["raw"].get("reason", ""),
            "error": False, "model": model}


async def main():
    args = C.parse_args("Phase 0: filtering")
    cfg = C.load_config(args.config)
    data = C.load_items(cfg.path("raw_input"))
    if args.sample:
        data = data[:args.sample]
    use_llm = cfg["filter"]["use_llm"]
    model = cfg["models"]["filter"]

    # Cached decisions are reused only if produced by the same model without errors
    cache: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(cfg.path("filter_decisions")):
        cache = {k: d for k, d in C.load_json(cfg.path("filter_decisions")).items()
                 if d.get("model") == model and not d.get("error")}

    seen, to_judge, status = set(), [], {}
    for item in data:
        reason = cleaning_reason(cfg, item, seen)
        status[C.item_id(cfg, item)] = reason
        if not reason:
            to_judge.append(item)

    print(f"\n🚀 PHASE 0: Filtering | {len(data)} items | content judgement: {model if use_llm else 'off'}")
    print("-" * 60)
    print(f"   After cleaning: {len(to_judge)} items")

    kept = []
    if use_llm:
        todo = [it for it in to_judge if C.item_id(cfg, it) not in cache]
        async with httpx.AsyncClient() as client:
            for n, item in enumerate(tqdm(todo, desc="Content filter"), start=1):
                cache[C.item_id(cfg, item)] = await judge(cfg, client, clean_text(cfg, C.item_text(cfg, item)),
                                                          C.context_block(cfg, item))
                if n % cfg["filter"]["save_every"] == 0:
                    C.save_json(cfg.path("filter_decisions"), cache)
            C.save_json(cfg.path("filter_decisions"), cache)
            await C.unload_model(cfg, client, model)

    for item in to_judge:
        iid = C.item_id(cfg, item)
        if not use_llm:
            status[iid] = "kept"
        elif cache[iid]["error"]:
            status[iid] = "kept_after_model_error"
        else:
            status[iid] = "kept" if cache[iid]["informative"] else "not_informative"
        if status[iid].startswith("kept"):
            kept.append(item)

    C.save_json(cfg.path("filtered"), kept)

    # Report by language and reason: documents that the filter does not penalise any language
    df = pd.DataFrame([{"language": C.item_language(cfg, it) or "n/a", "reason": status[C.item_id(cfg, it)]}
                       for it in data])
    table = df.pivot_table(index="language", columns="reason", aggfunc="size", fill_value=0)
    table["total"] = table.sum(axis=1)
    kept_cols = [c for c in ("kept", "kept_after_model_error") if c in table.columns]
    table["kept_%"] = (100 * table[kept_cols].sum(axis=1) / table["total"]).round(1)
    table.sort_values("total", ascending=False).to_csv(cfg.path("filter_report"), encoding="utf-8-sig")

    print("-" * 60)
    print(f"   Kept: {len(kept)} ({100 * len(kept) / max(len(data), 1):.1f}%)")
    print(f"   By reason: {dict(Counter(status.values()))}")
    print(f"💾 {cfg.path('filtered')}\n   Report: {cfg.path('filter_report')}")


if __name__ == "__main__":
    asyncio.run(main())

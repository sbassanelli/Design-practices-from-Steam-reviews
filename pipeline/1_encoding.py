"""
PHASE 1 — Encoding: two independent LLM coders extract verbatim snippets per theme

For each item and each encoder model:
  1. Extraction: per theme, a list of {snippet, code, sentiment} (JSON schema enforced; long texts split in chunks).
  2. Formal validation: snippets must be verbatim, long enough, unique; codes short and not a copy of the snippet.
  3. Verification (optional): the same model re-classifies each snippet ALONE; a (theme, snippet) pair is kept
     only if the verifier also picks that theme. Rejected pairs are stored for transparency.
One model is loaded at a time. Results are incremental: matched by item id, so only new items are encoded.

Usage:  python 1_encoding.py [--config study_config.json] [--sample N]
"""
import asyncio
import json
import os
import re
from typing import Any, Dict, List

import httpx
from tqdm import tqdm

import common as C

ENCODERS = ("enc1", "enc2")

ENCODER_PROMPT = """You are a high-precision qualitative coder analysing {context}.
Code each text with this CODEBOOK:

{themes}

GENERAL RULES OF THE CODEBOOK:
{global_rules}

SENTIMENT (of the snippet itself, not of the whole text):
{sentiment}

RULES:
1. Snippets are SHORT (about 4-20 words), single-topic phrases.
2. Snippets MUST be copied EXACTLY, character by character, from the text. Never paraphrase, translate, fix typos or join separate sentences.
3. Keep the text's original language. Only keys and sentiment labels are in English.
4. Each theme is a list with 0, 1 or more snippets.
5. For every snippet give a "code" (initial code, as in Braun & Clarke thematic analysis):
   - a short analytic label (2-6 words, ALWAYS in English, even for non-English snippets)
     that captures what the snippet MEANS for that theme, not a copy of its words;
   - written at a general level so the same code can be reused across different texts;
   - specific to the theme: the same snippet coded under two themes gets two different codes.
   Generic praise or criticism without a topic cannot be coded under any theme: do NOT include it.
   A snippet may appear in more than one theme ONLY if it has explicit content for each theme.
6. Respect the INCLUDES/EXCLUDES of each theme. Before adding a snippet, ask: "Is this snippet really ABOUT this theme,
   or does it only mention a word related to it?"
7. If there is no clear evidence for a theme, return an empty list [].
8. Inside snippets replace any double quote " with a single quote '.{context_rule}

EXAMPLES:
{examples}"""

CONTEXT_RULE = """
9. A <context> block may precede the text (e.g. the message it replies to, or the question it answers).
   Use it ONLY to understand the text: NEVER extract snippets from the context."""

VERIFY_PROMPT = """You are a qualitative coder classifying short snippets from {context}.
CODEBOOK:
{themes}

GENERAL RULES OF THE CODEBOOK:
{global_rules}

For EACH snippet, list ALL themes whose definition the snippet's own words match (0, 1 or more themes).
- Judge only the words of the snippet.
- Apply the INCLUDES and EXCLUDES strictly: a snippet must be ABOUT the theme, not just mention a related word.
- Generic praise or opinion with no topic matches NO theme → [].
Return exactly one result per snippet id, in the same order."""


# --- Prompts and schemas built from the config ---

def format_examples(cfg: C.Config) -> str:
    blocks = []
    for n, ex in enumerate(cfg["few_shot_examples"], start=1):
        output = {t: ex["output"].get(t, []) for t in cfg.themes}
        blocks.append(f"Example {n} ({ex.get('note', '')}):\n<text>{ex['text']}</text>\n"
                      f"OUTPUT:\n{json.dumps(output, ensure_ascii=False)}")
    return "\n\n".join(blocks)


def encoder_system_prompt(cfg: C.Config) -> str:
    return ENCODER_PROMPT.format(context=cfg["study"]["context"], themes=C.format_themes(cfg),
                                 global_rules=C.format_global_rules(cfg), sentiment=C.format_sentiment(cfg),
                                 examples=format_examples(cfg),
                                 context_rule=CONTEXT_RULE if cfg["fields"].get("context") else "")


def output_schema(cfg: C.Config) -> Dict[str, Any]:
    snippet = {
        "type": "object",
        "properties": {"snippet": {"type": "string"}, "code": {"type": "string"},
                       "sentiment": {"type": "string", "enum": cfg.sentiments}},
        "required": ["snippet", "code", "sentiment"],
    }
    return {"type": "object", "properties": {t: {"type": "array", "items": snippet} for t in cfg.themes},
            "required": cfg.themes}


def verify_schema(cfg: C.Config) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"results": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"},
                           "themes": {"type": "array", "items": {"type": "string", "enum": cfg.themes}}},
            "required": ["id", "themes"]}}},
        "required": ["results"],
    }


# --- Extraction ---

def split_text(text: str, max_chars: int) -> List[str]:
    """Splits long texts into chunks at sentence boundaries (avoids truncated or looping outputs)."""
    if len(text) <= max_chars:
        return [text]
    units = [u for u in re.split(r"(?<=[.!?。！？\n])\s+", text) if u.strip()]
    chunks, current = [], ""
    for u in units:
        if current and len(current) + len(u) + 1 > max_chars:
            chunks.append(current)
            current = u
        else:
            current = f"{current} {u}".strip()
    if current:
        chunks.append(current)
    return chunks


async def encode_text(cfg: C.Config, client: httpx.AsyncClient, model: str, text: str,
                      context_block: str = "") -> Dict[str, Any]:
    system, schema = encoder_system_prompt(cfg), output_schema(cfg)
    merged = {t: [] for t in cfg.themes}
    for chunk in split_text(text, cfg["encoding"]["max_chunk_chars"]):
        res = await C.call_llm(cfg, client, model, [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context_block}<text>{chunk}</text>\nOUTPUT:"},
        ], schema)
        if res["error"]:
            return {"raw": merged, "error": res["error"]}
        for t in cfg.themes:
            items = res["raw"].get(t, [])
            merged[t].extend(items if isinstance(items, list) else [items])
    return {"raw": merged, "error": None}


def sanitize(cfg: C.Config, raw: Dict[str, Any], text: str) -> Dict[str, List[Dict[str, str]]]:
    """Formal validation: verbatim, minimum length, no duplicates, short code that is not a copy of the snippet."""
    text_norm = C.normalize(text)
    out = {}
    for theme in cfg.themes:
        items = raw.get(theme, [])
        items = [items] if isinstance(items, dict) else (items if isinstance(items, list) else [])
        valid, seen = [], set()
        for it in items:
            if not isinstance(it, dict):
                continue
            snip = C.validate_snippet(cfg, it.get("snippet"), text_norm)
            if not snip or C.normalize(snip) in seen:
                continue
            code = " ".join(str(it.get("code") or "").split()).strip(" .…")
            if not code or len(code.split()) > cfg["encoding"]["max_code_words"] or C.normalize(code) == C.normalize(snip):
                continue
            seen.add(C.normalize(snip))
            sent = it.get("sentiment")
            neutral = next((s for s in cfg.sentiments if cfg.polarity[s] == "Neutral"), cfg.sentiments[0])
            valid.append({"snippet": snip, "code": code, "sentiment": sent if sent in cfg.sentiments else neutral})
        out[theme] = valid
    return out


# --- Verification ---

async def verify(cfg: C.Config, client: httpx.AsyncClient, model: str, coded: Dict[str, Any]) -> Dict[str, Any]:
    snippets = list(dict.fromkeys(s["snippet"] for t in cfg.themes for s in coded[t]))
    if not snippets:
        return {"coded": coded, "rejected": [], "error": None}

    system = VERIFY_PROMPT.format(context=cfg["study"]["context"], themes=C.format_themes(cfg),
                                  global_rules=C.format_global_rules(cfg))
    listing = "\n".join(f'{n}. "{s}"' for n, s in enumerate(snippets, start=1))
    res = await C.call_llm(cfg, client, model, [
        {"role": "system", "content": system},
        {"role": "user", "content": f"SNIPPETS:\n{listing}"},
    ], verify_schema(cfg))
    if res["error"]:
        return {"coded": coded, "rejected": [], "error": res["error"]}   # keep unverified output, flagged

    results = [r for r in res["raw"].get("results", []) if isinstance(r, dict)]
    by_id = {r.get("id"): r for r in results}
    verdicts = {}
    for n, snip in enumerate(snippets, start=1):
        r = by_id.get(n)
        if r is None and len(results) == len(snippets):   # wrong ids but same order
            r = results[n - 1]
        if r is not None:
            verdicts[snip] = set(r.get("themes") or [])

    kept, rejected = {t: [] for t in cfg.themes}, []
    for theme in cfg.themes:
        for s in coded[theme]:
            v = verdicts.get(s["snippet"])
            if v is None or theme in v:      # not evaluated → kept (verification removes clear errors only)
                kept[theme].append(s)
            else:
                rejected.append({"theme": theme, **s, "verifier_themes": sorted(v)})
    return {"coded": kept, "rejected": rejected, "error": None}


# --- Main ---

def load_previous(cfg: C.Config) -> Dict[str, Dict[str, Any]]:
    previous = {}
    for key in ("encoded", "encoding_checkpoint"):
        path = cfg.path(key)
        if not os.path.exists(path):
            continue
        try:
            content = C.load_json(path)
            entries = content["results"] if isinstance(content, dict) else content
            previous.update({e["item_id"]: e for e in entries})
        except Exception as e:
            print(f"Warning: {path} unreadable, ignored ({e})")
    return previous


async def main():
    args = C.parse_args("Phase 1: encoding")
    cfg = C.load_config(args.config)
    data = C.load_json(cfg.path("filtered"))
    if args.sample:
        data = data[:args.sample]
    models = {"enc1": cfg["models"]["encoder_1"], "enc2": cfg["models"]["encoder_2"]}

    # Incremental reuse: previous results matched by item id; results of a different model are discarded
    previous = load_previous(cfg)
    results: Dict[int, Dict[str, Any]] = {}
    for i, item in enumerate(data):
        entry = previous.get(C.item_id(cfg, item))
        if entry is None:
            continue
        entry = dict(entry, index=i, original_data=item)
        for key in ENCODERS:
            if entry.get(f"{key}_model") != models[key]:
                for field in [k for k in entry if k == key or k.startswith(f"{key}_")]:
                    del entry[field]
        results[i] = entry
    if results:
        print(f"✅ Reusing {len(results)}/{len(data)} previous results (matched by id).")

    print(f"\n🚀 PHASE 1: Encoding | {len(data)} items | {len(cfg.themes)} themes | {models['enc1']} + {models['enc2']}")
    print("-" * 60)
    stats = {"kept": 0, "format_rejected": 0, "verify_rejected": 0}
    save = lambda: C.save_json(cfg.path("encoding_checkpoint"), {"results": list(results.values())})

    async with httpx.AsyncClient() as client:
        for key in ENCODERS:          # one full pass per model: only one model in memory at a time
            model = models[key]
            todo = [i for i in range(len(data)) if key not in results.get(i, {}) or results[i].get(f"{key}_error")]
            for n, i in enumerate(tqdm(todo, desc=f"{key} ({model})"), start=1):
                item = data[i]
                text = C.item_text(cfg, item)
                entry = results.setdefault(i, {"index": i, "item_id": C.item_id(cfg, item), "original_data": item})

                res = await encode_text(cfg, client, model, text, C.context_block(cfg, item))
                clean = sanitize(cfg, res["raw"], text)
                raw_n = sum(len(v) for v in res["raw"].values() if isinstance(v, list))
                stats["format_rejected"] += max(raw_n - sum(len(v) for v in clean.values()), 0)
                if cfg["encoding"]["verify"] and not res["error"]:
                    ver = await verify(cfg, client, model, clean)
                    clean = ver["coded"]
                    entry[f"{key}_rejected"] = ver["rejected"]
                    entry[f"{key}_verify_error"] = ver["error"]
                    stats["verify_rejected"] += len(ver["rejected"])
                stats["kept"] += sum(len(v) for v in clean.values())
                entry.update({key: clean, f"{key}_error": res["error"], f"{key}_model": model})
                if n % cfg["encoding"]["save_every"] == 0:
                    save()
            save()
            await C.unload_model(cfg, client, model)

    final = [results[i] for i in sorted(results)]
    C.save_json(cfg.path("encoded"), final)
    n_errors = sum(1 for r in final if r.get("enc1_error") or r.get("enc2_error"))

    print("-" * 60)
    print("✅ Encoding complete.")
    print(f"   Valid snippets (this run): {stats['kept']}")
    print(f"   Rejected by formal validation: {stats['format_rejected']}")
    print(f"   Rejected by theme verification (see enc*_rejected): {stats['verify_rejected']}")
    print(f"   Items with LLM errors: {n_errors} (re-run the script to retry them)")
    print(f"💾 {cfg.path('encoded')}")
    if n_errors == 0 and os.path.exists(cfg.path("encoding_checkpoint")):
        os.remove(cfg.path("encoding_checkpoint"))


if __name__ == "__main__":
    asyncio.run(main())

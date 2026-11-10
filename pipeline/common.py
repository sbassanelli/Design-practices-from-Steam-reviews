"""
Shared utilities for the generic LLM-assisted thematic analysis pipeline.

Everything study-specific (context, themes, examples, models, field names, paths) lives in the
JSON configuration file; this module loads it and provides the helpers used by the three phases.
"""
import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from tqdm import tqdm

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "study_config.json")


# =====================================================================
# CONFIGURATION
# =====================================================================

class Config(dict):
    """Configuration dictionary with resolved paths and convenience accessors."""

    @property
    def themes(self) -> List[str]:
        return [t["name"] for t in self["themes"]]

    @property
    def sentiments(self) -> List[str]:
        return [s["label"] for s in self["sentiment"]]

    @property
    def polarity(self) -> Dict[str, str]:
        return {s["label"]: s["polarity"] for s in self["sentiment"]}

    def path(self, key: str) -> str:
        return self["_paths"][key]


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the study configuration JSON")
    parser.add_argument("--sample", type=int, default=None, help="Process only the first N items (quick tests)")
    return parser.parse_args()


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = Config(json.load(f))
    validate_config(cfg)

    # Paths are relative to the configuration file, so the pipeline can be launched from any folder
    base = os.path.dirname(os.path.abspath(path))
    output_dir = os.path.join(base, cfg["paths"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    cfg["_paths"] = {"raw_input": os.path.join(base, cfg["paths"]["raw_input"]), "output_dir": output_dir}
    for key, name in cfg["paths"]["files"].items():
        cfg["_paths"][key] = os.path.join(output_dir, name)
    deductive = cfg["harmonization"].get("deductive_codebook")
    if deductive:
        cfg["_paths"]["deductive_codebook"] = os.path.join(base, deductive)
    return cfg


def validate_config(cfg: Dict[str, Any]):
    required = ["study", "paths", "fields", "models", "ollama", "filter", "themes", "sentiment",
                "few_shot_examples", "encoding", "harmonization"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config is missing sections: {missing}")

    names = [t.get("name") for t in cfg["themes"]]
    if len(names) != len(set(names)) or not all(names):
        raise ValueError("Every theme needs a unique, non-empty 'name'")
    for t in cfg["themes"]:
        for k in ("definition", "includes", "excludes"):
            if k not in t:
                raise ValueError(f"Theme {t['name']} is missing '{k}'")

    polarities = {s.get("polarity") for s in cfg["sentiment"]}
    if not polarities <= {"Negative", "Neutral", "Positive"}:
        raise ValueError("Sentiment 'polarity' must be one of Negative / Neutral / Positive")

    labels = {s["label"] for s in cfg["sentiment"]}
    for ex in cfg["few_shot_examples"]:
        for theme, items in ex["output"].items():
            if theme not in names:
                raise ValueError(f"Few-shot example uses unknown theme '{theme}'")
            for it in items:
                if it.get("sentiment") not in labels:
                    raise ValueError(f"Few-shot example uses unknown sentiment '{it.get('sentiment')}'")

    mode = cfg["harmonization"].get("codebook_mode")
    if mode not in ("inductive", "deductive"):
        raise ValueError("harmonization.codebook_mode must be 'inductive' or 'deductive'")
    if mode == "deductive" and not cfg["harmonization"].get("deductive_codebook"):
        raise ValueError("Deductive mode requires harmonization.deductive_codebook (path to the codebook JSON)")


# =====================================================================
# INPUT FIELDS
# =====================================================================

def get_field(item: Dict[str, Any], dotted: Optional[str], default: Any = None) -> Any:
    """Reads a (possibly nested) field, e.g. 'author.playtime_forever'."""
    if not dotted:
        return default
    value: Any = item
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def item_id(cfg: Config, item: Dict[str, Any]) -> str:
    return str(get_field(item, cfg["fields"]["id"]))


MD_QUOTE = re.compile(r"^\s*(>|&gt;).*$", re.MULTILINE)          # quoted lines (text written by someone else)
MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")     # [label](url) → label
URL = re.compile(r"https?://\S+")
MD_MARKUP = re.compile(r"(\*\*|__|~~|\^|`)")                       # bold, strikethrough, superscript, code


def strip_markdown(text: str) -> str:
    """Reddit-style markdown cleaning: drops quoted lines, keeps link labels, removes URLs and markup."""
    text = MD_QUOTE.sub("", text)
    text = MD_LINK.sub(r"\1", text)
    text = URL.sub("", text)
    return MD_MARKUP.sub("", text)


def item_text(cfg: Config, item: Dict[str, Any]) -> str:
    """The text analysed in every phase (optionally cleaned of markdown, see filter.strip_markdown)."""
    text = get_field(item, cfg["fields"]["text"], "") or ""
    if not isinstance(text, str):
        text = str(text)
    return strip_markdown(text) if cfg["filter"].get("strip_markdown", False) else text


def item_context(cfg: Config, item: Dict[str, Any]) -> Optional[str]:
    """Optional context shown to the models but never coded (e.g. the parent comment, the survey question)."""
    ctx = get_field(item, cfg["fields"].get("context"))
    if ctx is None or (isinstance(ctx, float) and ctx != ctx):     # missing or NaN (from CSV)
        return None
    ctx = str(ctx).strip()
    if cfg["filter"].get("strip_markdown", False):
        ctx = strip_markdown(ctx).strip()
    return ctx or None


def context_block(cfg: Config, item: Dict[str, Any]) -> str:
    """'<context>…</context>\n' when the item has context, otherwise an empty string."""
    ctx = item_context(cfg, item)
    return f"<context>{ctx}</context>\n" if ctx else ""


def item_language(cfg: Config, item: Dict[str, Any]) -> Optional[str]:
    return get_field(item, cfg["fields"].get("language"))


def item_metadata(cfg: Config, item: Dict[str, Any]) -> Dict[str, Any]:
    return {name: get_field(item, path) for name, path in cfg["fields"].get("metadata", {}).items()}


# =====================================================================
# PROMPT BUILDING BLOCKS
# =====================================================================

def format_themes(cfg: Config) -> str:
    return "\n".join(
        f"- {t['name']}: {t['definition']}\n    INCLUDES: {t['includes']}\n    EXCLUDES: {t['excludes']}"
        for t in cfg["themes"]
    )


def theme_definition(cfg: Config, theme: str) -> str:
    t = next(t for t in cfg["themes"] if t["name"] == theme)
    return f"{theme}: {t['definition']} INCLUDES: {t['includes']} EXCLUDES: {t['excludes']}"


def format_sentiment(cfg: Config) -> str:
    return "\n".join(f'- "{s["label"]}": {s["description"]}' for s in cfg["sentiment"])


def format_global_rules(cfg: Config) -> str:
    rules = cfg.get("global_rules", [])
    return "\n".join(f"- {r}" for r in rules) if rules else "- (none)"


# =====================================================================
# TEXT NORMALISATION AND VERBATIM VALIDATION
# =====================================================================

NON_SPACE_SCRIPT = re.compile(
    r"[一-鿿㐀-䶿぀-ヿ가-힯฀-๿຀-໿ក-៿က-႟]"
)


def is_non_space_script(text: str) -> bool:
    """CJK, Thai, Lao, Khmer, Myanmar: scripts that do not separate words with spaces."""
    return bool(NON_SPACE_SCRIPT.search(text))


def normalize(text: str) -> str:
    """Normalises quotes, full-width punctuation, case and whitespace (used only for matching)."""
    if not isinstance(text, str):
        return ""
    for a, b in (("’", "'"), ("‘", "'"), ("“", "'"), ("”", "'"), ('"', "'"), ("「", "'"), ("」", "'"),
                 ("、", ","), ("。", "."), ("，", ","), ("！", "!"), ("？", "?"), ("：", ":")):
        text = text.replace(a, b)
    return " ".join(text.split()).lower()


def validate_snippet(cfg: Config, snippet: Any, text_norm: str) -> Optional[str]:
    """Returns the cleaned snippet if it is long enough and appears verbatim in the text, else None."""
    if not snippet or not isinstance(snippet, str):
        return None
    cleaned = " ".join(snippet.split()).strip(" .…")
    if cleaned.lower() in ("none", "no", "n/a", "null", "not specified", "no snippet"):
        return None
    min_words = cfg["encoding"]["min_snippet_words"]
    min_chars = cfg["encoding"]["min_snippet_chars"]
    if is_non_space_script(cleaned):
        if len(cleaned) < min_chars // 2:          # 1 CJK character ≈ 1 word
            return None
    elif len(cleaned.split()) < min_words or len(cleaned) < min_chars:
        return None
    return cleaned if normalize(cleaned) in text_norm else None


# =====================================================================
# OLLAMA
# =====================================================================

async def call_llm(cfg: Config, client: httpx.AsyncClient, model: str, messages: List[Dict[str, str]],
                   schema: Dict[str, Any], num_ctx: Optional[int] = None) -> Dict[str, Any]:
    """Calls Ollama's native chat API with a JSON schema (structured output).
    Returns {"raw": dict, "error": None} or {"raw": {}, "error": "..."} after the configured retries."""
    o = cfg["ollama"]
    payload = {
        "model": model,
        "messages": messages,
        "format": schema,
        "stream": False,
        "think": False,        # "thinking" models would otherwise spend the output budget reasoning
        "keep_alive": "30m",
        "options": {"temperature": 0, "seed": o["seed"], "num_ctx": num_ctx or o["num_ctx"],
                    "num_predict": o["num_predict"]},
    }
    last_error = None
    for attempt in range(1, o["max_retries"] + 1):
        try:
            resp = await client.post(o["url"], json=payload, timeout=o["timeout"])
            resp.raise_for_status()
            parsed = json.loads(resp.json()["message"]["content"])
            if not isinstance(parsed, dict):
                raise ValueError("output is not a JSON object")
            return {"raw": parsed, "error": None}
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            payload["options"]["temperature"] = 0.2 * attempt   # avoid regenerating the same broken output
    tqdm.write(f"❌ [{model}] failed after {o['max_retries']} attempts: {last_error}")
    return {"raw": {}, "error": last_error}


async def embed(cfg: Config, client: httpx.AsyncClient, model: str, texts: List[str],
                prefix: str = "", batch_size: int = 128) -> List[List[float]]:
    """Embeddings from Ollama's /api/embed endpoint (same server as the chat API)."""
    url = cfg["ollama"]["url"].rsplit("/api/", 1)[0] + "/api/embed"
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start:start + batch_size]]
        resp = await client.post(url, json={"model": model, "input": batch}, timeout=cfg["ollama"]["timeout"])
        resp.raise_for_status()
        vectors.extend(resp.json()["embeddings"])
    return vectors


async def unload_model(cfg: Config, client: httpx.AsyncClient, model: str):
    """Frees the model's memory at the end of a phase."""
    try:
        await client.post(cfg["ollama"]["url"], json={"model": model, "messages": [], "keep_alive": 0}, timeout=60)
    except Exception:
        pass


# =====================================================================
# I/O
# =====================================================================

def load_items(path: str) -> List[Dict[str, Any]]:
    """Reads the raw input: JSON (list of objects), JSONL (one object per line) or CSV (one row per item)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path, dtype=str, keep_default_na=False)     # strings: ids keep leading zeros
        return [{k: (v if v != "" else None) for k, v in row.items()} for row in df.to_dict(orient="records")]
    if ext in (".jsonl", ".ndjson"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of objects")
    return data


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any):
    """Atomic write: a crash never leaves a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

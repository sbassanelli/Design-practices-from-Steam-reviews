"""
Human validation of the LLM coding (after Phase 2): a blind coding sheet, then the agreement between human and LLM.

Works with any source (reviews, survey answers, interview segments, Reddit comments): the unit is the item, coded
at the item level as the analyses use it. Two steps:

  python validation.py sample [--config C] [--per-group N] [--subset-per-group K] [--force]
      Draws a stratified random subsample of the coded items (those in encoded_results.json, with and without
      LLM extracts alike) and writes an Excel coding sheet WITHOUT the LLM codes. For each theme the coder fills:
        <THEME>_present    1 / 0
        <THEME>_polarity   positive / negative / mixed / neutral (only if present)
        <THEME>_subthemes  optional: ids of the sub-themes mentioned, separated by ";"
      Strata are those of stratified_sample.py (section `sampling`): group x (oversampled / other). The oversampled
      field itself (e.g. the recommendation, a Likert score) is NOT shown, since the LLM coders did not see it;
      the group and the item's context (survey question, thread, interviewer question) are shown, since they did.
      Items not in `validation.target_language` get a machine translation by `validation.translator_model`, a model
      not used in the coding (null = no translation).
      --subset-per-group K codes a nested subset of the full validation sample (first K per group in the random
      sheet order, half oversampled), so the validation can be extended later without changing the coded items.

  python validation.py agreement [--config C]
      Reads the filled sheet and compares it, item by item, with the final LLM coding (harmonized) and with each
      encoder alone, their union and their intersection (what the harmonization adds). For each theme: presence,
      positive mention (positive or mixed), negative mention (negative or mixed). Measures: percent agreement,
      Cohen's kappa, Gwet's AC1 (robust to very low or high prevalence), prevalence for human and LLM, sensitivity
      and precision of the LLM against the human, 95% CIs from bootstrap resamples of items. Sub-themes (if filled):
      kappa per sub-theme and micro-averaged F1. Only fully coded rows are used (every <THEME>_present filled).

Settings (section `validation`, all optional): per_group (50), oversampled_per_group (half of per_group), seed (2026),
translator_model ("mistral-small:latest"), target_language (["english"]), translate_to ("English"),
output_dir ("validation", relative to the output folder), n_boot (2000).
Run with the configuration used for encoding and harmonization (the sample configuration, if you sampled).

Outputs (<output_dir>/validation/):
  coding_sheet.xlsx                  sheets "Instructions", "Codebook", "Coding" (fill in "Coding" only)
  validation_sample.csv              ids of the sampled items (do not open the LLM codes before coding)
  translations.json                  cached translations
  agreement_themes.csv, agreement_encoders.csv, agreement_subthemes.csv, disagreements.csv, tab_validation.tex
Requires openpyxl.
"""
import argparse
import json
import os
import sys

import httpx
import numpy as np
import pandas as pd

import common as C

DEFAULTS = {"per_group": 50, "oversampled_per_group": None, "seed": 2026, "translator_model": "mistral-small:latest",
            "target_language": ["english"], "translate_to": "English", "output_dir": "validation", "n_boot": 2000}
POLARITY = ["positive", "negative", "mixed", "neutral"]
MIN_CODED = 20


def settings(cfg: C.Config) -> dict:
    v = {**DEFAULTS, **cfg.get("validation", {})}
    if isinstance(v["target_language"], str):
        v["target_language"] = [v["target_language"]]
    v["dir"] = os.path.join(cfg.path("output_dir"), v["output_dir"])
    os.makedirs(v["dir"], exist_ok=True)
    return v


def sampling(cfg: C.Config) -> dict:
    return {"group_field": None, "oversample": None, **cfg.get("sampling", {})}


def same_value(value, target) -> bool:
    """Numbers compare as numbers (2 == "2" == "2.0", e.g. Likert answers read from CSV), the rest as text
    without case (False == "false")."""
    try:
        return float(value) == float(target)
    except (TypeError, ValueError):
        return str(value).strip().lower() == str(target).strip().lower()

def is_oversampled(over, item) -> bool:
    if not over:
        return False
    targets = over["value"] if isinstance(over["value"], list) else [over["value"]]
    value = C.get_field(item, over["field"])
    return any(same_value(value, t) for t in targets)


# =====================================================================
# LLM CODING AT THE ITEM LEVEL
# =====================================================================

def codebook(cfg) -> pd.DataFrame:
    cb = C.load_json(cfg.path("codebook"))
    return pd.DataFrame([{"theme": t, "subtheme_id": s["id"], "subtheme": s["name"]}
                         for t, subs in cb["themes"].items() for s in subs])


def llm_item_codes(cfg, ids) -> pd.DataFrame:
    """Harmonized coding per item: theme mention, positive / negative mention, sub-themes."""
    ex = pd.read_csv(cfg.path("final"), dtype={"item_id": str})
    d = pd.DataFrame({"item_id": list(ids)})
    for theme in cfg.themes:
        t = ex[ex["theme"] == theme]
        key = theme.lower()
        d[f"{key}_mention"] = d["item_id"].isin(t["item_id"]).astype(int)
        d[f"{key}_pos"] = d["item_id"].isin(t.loc[t["polarity"] == "Positive", "item_id"]).astype(int)
        d[f"{key}_neg"] = d["item_id"].isin(t.loc[t["polarity"] == "Negative", "item_id"]).astype(int)
    for _, s in codebook(cfg).iterrows():
        d[f"sub_{s['subtheme_id']}"] = d["item_id"].isin(ex.loc[ex["subtheme_id"] == s["subtheme_id"], "item_id"]).astype(int)
    return d.set_index("item_id")


def encoder_codes(cfg, ids) -> pd.DataFrame:
    rows = []
    for e in C.load_json(cfg.path("encoded")):
        if str(e["item_id"]) not in ids:
            continue
        row = {"item_id": str(e["item_id"])}
        for t in cfg.themes:
            a, b = bool((e.get("enc1") or {}).get(t)), bool((e.get("enc2") or {}).get(t))
            row.update({f"{t.lower()}_enc1": a, f"{t.lower()}_enc2": b, f"{t.lower()}_either": a or b,
                        f"{t.lower()}_both": a and b})
        rows.append(row)
    return pd.DataFrame(rows).set_index("item_id")


# =====================================================================
# STEP 1 - SAMPLE AND CODING SHEET
# =====================================================================

def draw(pop: pd.DataFrame, per_group: int, n_over: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for group in sorted(pop["group"].unique()):
        for flag, n in ((True, n_over), (False, per_group - n_over)):
            pool = pop[(pop["group"] == group) & (pop["oversampled"] == flag)].sort_values("item_id")
            take = min(n, len(pool))
            out.append(pool.iloc[np.sort(rng.choice(len(pool), size=take, replace=False))])
    s = pd.concat(out)
    return s.iloc[rng.permutation(len(s))].reset_index(drop=True)      # random order: groups and strata mixed


def translate(cfg, v, text: str, cache: dict, client: httpx.Client) -> str:
    if text in cache:
        return cache[text]
    prompt = (f"Translate this text ({cfg['study']['context']}) into {v['translate_to']}. Keep the meaning, tone and "
              "any specific terms; do not summarise, explain or add anything. Return only the translation.")
    r = client.post(cfg["ollama"]["url"], json={
        "model": v["translator_model"], "stream": False, "think": False,
        "options": {"temperature": 0, "seed": cfg["ollama"]["seed"], "num_ctx": cfg["ollama"]["num_ctx"]},
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": text}]},
        timeout=cfg["ollama"]["timeout"])
    r.raise_for_status()
    cache[text] = r.json()["message"]["content"].strip()
    return cache[text]


def write_sheet(cfg, s: pd.DataFrame, show_group: bool, show_language: bool, show_context: bool, path: str):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    bold, wrap = Font(bold=True), Alignment(wrap_text=True, vertical="top")
    head = PatternFill("solid", fgColor="E8EEF8")
    themes = cfg.themes

    ins = wb.active
    ins.title = "Instructions"
    lines = [
        "Human validation of the LLM coding - instructions",
        "",
        f"Corpus: {cfg['study']['context']}.",
        "Code every row of the sheet 'Coding', one item at a time, WITHOUT looking at the LLM results.",
        f"For each of the {len(themes)} themes ({', '.join(themes)}):",
        "  <THEME>_present: 1 if the text contains at least one statement that fits the theme (see 'Codebook':",
        "     definition, includes, excludes and general rules), 0 otherwise. Leave no cell empty.",
        "  <THEME>_polarity: only if present = 1: positive, negative, mixed (both positive and negative statements)",
        "     or neutral.",
        "  <THEME>_subthemes (optional but useful): ids of the sub-themes mentioned, separated by ';' (e.g. E1;E3).",
        "Code only the column 'text'. The column 'context' (if present) helps to understand it and is never coded.",
        "Texts not in the language of the study have a machine translation: read the original when you can,",
        "the translation otherwise. Use 'notes' for doubts.",
        "Do not change the columns before the theme columns.",
        "Save the file with the same name when you finish (or in several sessions).",
    ]
    for i, line in enumerate(lines, start=1):
        ins.cell(row=i, column=1, value=line).font = bold if i == 1 else Font()
    ins.column_dimensions["A"].width = 120

    cb = C.load_json(cfg.path("codebook"))
    cbs = wb.create_sheet("Codebook")
    cbs.append(["Theme / sub-theme", "Definition", "Includes", "Excludes"])
    for t in cfg["themes"]:
        cbs.append([t["name"], t["definition"], t["includes"], t["excludes"]])
        for cell in cbs[cbs.max_row]:
            cell.font, cell.fill = bold, head
        for sub in cb["themes"].get(t["name"], []):
            cbs.append([f"   {sub['id']}  {sub['name']}", sub["definition"], "", ""])
    cbs.append([])
    cbs.append(["General rules"])
    cbs[cbs.max_row][0].font = bold
    for rule in cfg.get("global_rules", []):
        cbs.append(["", rule])
    for col, width in zip("ABCD", (42, 70, 70, 70)):
        cbs.column_dimensions[col].width = width
    for row in cbs.iter_rows():
        for c in row:
            c.alignment = wrap

    ws = wb.create_sheet("Coding")
    info = ["row", "item_id"] + (["group"] if show_group else []) + (["language"] if show_language else []) \
        + (["context"] if show_context else []) + ["text"] + (["translation"] if show_language else [])
    cols = info + [f"{t}_{k}" for t in themes for k in ("present", "polarity", "subthemes")] + ["notes"]
    ws.append(cols)
    for c in ws[1]:
        c.font, c.fill, c.alignment = bold, head, wrap
    for i, r in s.iterrows():
        values = {"row": i + 1, **r.to_dict()}
        ws.append([values.get(c) or "" for c in info] + [None] * (3 * len(themes) + 1))
    n, first = len(s) + 1, len(info) + 1
    present = DataValidation(type="list", formula1='"1,0"', allow_blank=True)
    polarity = DataValidation(type="list", formula1='"' + ",".join(POLARITY) + '"', allow_blank=True)
    ws.add_data_validation(present)
    ws.add_data_validation(polarity)
    letter = lambda j: ws.cell(row=1, column=j).column_letter
    for j in range(len(themes)):
        present.add(f"{letter(first + 3 * j)}2:{letter(first + 3 * j)}{n}")
        polarity.add(f"{letter(first + 3 * j + 1)}2:{letter(first + 3 * j + 1)}{n}")
    widths = {"row": 5, "item_id": 12, "group": 16, "language": 10, "context": 40, "text": 60, "translation": 60}
    for j, c in enumerate(info, start=1):
        ws.column_dimensions[letter(j)].width = widths[c]
    for j in range(first, len(cols) + 1):
        ws.column_dimensions[letter(j)].width = 13
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = wrap
    ws.freeze_panes = f"{letter(first)}2"
    wb.save(path)


def make_sample(cfg, v, args):
    sheet = os.path.join(v["dir"], "coding_sheet.xlsx")
    if os.path.exists(sheet) and not args.force:
        sys.exit(f"{sheet} already exists (it may contain your coding): use --force to overwrite it.")
    smp = sampling(cfg)
    over = smp["oversample"]
    per_group = args.per_group or v["per_group"]
    n_over = (v["oversampled_per_group"] if v["oversampled_per_group"] is not None else per_group // 2) if over else 0
    n_over = min(n_over, per_group)

    coded = {str(e["item_id"]) for e in C.load_json(cfg.path("encoded"))}
    items = {C.item_id(cfg, r): r for r in C.load_json(cfg.path("filtered")) if C.item_id(cfg, r) in coded}
    pop = pd.DataFrame([{"item_id": i,
                         "group": str(C.get_field(r, smp["group_field"], "missing")) if smp["group_field"] else "ALL",
                         "oversampled": is_oversampled(over, r)} for i, r in items.items()])
    s = draw(pop, per_group, n_over, v["seed"])
    if args.subset_per_group:                    # first k per group x stratum in the (random) sheet order
        k_over = args.subset_per_group // 2 if over else 0
        k = {True: k_over, False: args.subset_per_group - k_over}
        s = s[s.groupby(["group", "oversampled"]).cumcount() < s["oversampled"].map(k)].reset_index(drop=True)
    s["language"] = [C.item_language(cfg, items[i]) for i in s["item_id"]]
    s["context"] = [C.item_context(cfg, items[i]) for i in s["item_id"]]
    s["text"] = [C.item_text(cfg, items[i]) for i in s["item_id"]]
    label = f"{over['field']} in {over['value']}" if over and isinstance(over["value"], list) else \
        f"{over['field']}={over['value']}" if over else "oversampled"
    s[["item_id", "group", "oversampled", "language"]].rename(columns={"oversampled": label}).to_csv(
        os.path.join(v["dir"], "validation_sample.csv"), index=False)

    targets = {t.lower() for t in v["target_language"]}
    foreign = s["language"].notna() & ~s["language"].astype(str).str.lower().isin(targets)
    cache_path = os.path.join(v["dir"], "translations.json")
    cache = C.load_json(cache_path) if os.path.exists(cache_path) else {}
    if v["translator_model"]:
        todo = [t for t in s.loc[foreign, "text"] if t not in cache]
        print(f"{len(s)} items sampled; translating {len(todo)} texts with {v['translator_model']}...")
        with httpx.Client() as client:
            for n, text in enumerate(todo, start=1):
                translate(cfg, v, text, cache, client)
                if n % 10 == 0:
                    C.save_json(cache_path, cache)
                    print(f"  {n}/{len(todo)}", flush=True)
        C.save_json(cache_path, cache)
    s["translation"] = [cache.get(t, "") if f else "" for t, f in zip(s["text"], foreign)]

    write_sheet(cfg, s, show_group=bool(smp["group_field"]), show_language=s["language"].notna().any(),
                show_context=s["context"].notna().any(), path=sheet)
    print(f"Coding sheet: {sheet}")
    print(s.groupby(["group", "oversampled"]).size().unstack(fill_value=0).to_string())
    if s["language"].notna().any():
        print(s["language"].value_counts().head(10).to_string())


# =====================================================================
# STEP 2 - AGREEMENT
# =====================================================================

def measures(h: np.ndarray, m: np.ndarray) -> dict:
    h, m = h.astype(int), m.astype(int)
    po = (h == m).mean()
    ph, pm = h.mean(), m.mean()
    pe_k = ph * pm + (1 - ph) * (1 - pm)
    kappa = (po - pe_k) / (1 - pe_k) if pe_k < 1 else np.nan
    pi = (ph + pm) / 2
    pe_g = 2 * pi * (1 - pi)
    ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else np.nan
    tp = ((h == 1) & (m == 1)).sum()
    return {"agreement": po, "kappa": kappa, "ac1": ac1, "prev_human": ph, "prev_llm": pm,
            "sensitivity": tp / h.sum() if h.sum() else np.nan, "precision": tp / m.sum() if m.sum() else np.nan}


def with_ci(h, m, n_boot, rng) -> dict:
    point = measures(h, m)
    idx = rng.integers(0, len(h), size=(n_boot, len(h)))
    boot = pd.DataFrame([measures(h[i], m[i]) for i in idx])
    out = {}
    for k, val in point.items():
        out[k] = round(val, 3)
        if k in ("kappa", "ac1", "agreement"):
            lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
            out[f"{k}_ci_low"], out[f"{k}_ci_high"] = round(lo, 3), round(hi, 3)
    return out


def human_codes(cfg, sheet: pd.DataFrame) -> pd.DataFrame:
    h = pd.DataFrame({"item_id": sheet["item_id"].astype(str)})
    for t in cfg.themes:
        present = pd.to_numeric(sheet[f"{t}_present"], errors="coerce")
        pol = sheet[f"{t}_polarity"].astype(str).str.strip().str.lower()
        h[f"{t.lower()}_mention"] = present.astype(int)
        h[f"{t.lower()}_pos"] = ((present == 1) & pol.isin(["positive", "mixed"])).astype(int)
        h[f"{t.lower()}_neg"] = ((present == 1) & pol.isin(["negative", "mixed"])).astype(int)
        h[f"{t}_subthemes"] = sheet[f"{t}_subthemes"].fillna("").astype(str)
    return h


def agreement(cfg, v):
    rng = np.random.default_rng(7)
    path = os.path.join(v["dir"], "coding_sheet.xlsx")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run 'python validation.py sample' and fill in the sheet first.")
    sheet = pd.read_excel(path, sheet_name="Coding", dtype={"item_id": str})
    done = sheet[[f"{t}_present" for t in cfg.themes]].notna().all(axis=1)
    print(f"Coded items: {int(done.sum())} of {len(sheet)}")
    if done.sum() < MIN_CODED:
        sys.exit(f"Too few coded items to compute agreement (at least {MIN_CODED}).")
    sheet = sheet[done]
    h = human_codes(cfg, sheet)
    ids = h["item_id"].tolist()
    llm = llm_item_codes(cfg, ids).loc[ids].reset_index()

    rows = []
    for t in cfg.themes:
        for kind, label in (("mention", "present"), ("pos", "positive mention"), ("neg", "negative mention")):
            col = f"{t.lower()}_{kind}"
            rows.append({"theme": t, "measure": label, "n": len(h),
                         **with_ci(h[col].to_numpy(), llm[col].to_numpy(), v["n_boot"], rng)})
    themes = pd.DataFrame(rows)
    themes.to_csv(os.path.join(v["dir"], "agreement_themes.csv"), index=False)

    enc = encoder_codes(cfg, set(ids)).loc[ids].reset_index()
    rows = []
    for t in cfg.themes:
        for version in ("enc1", "enc2", "either", "both"):
            rows.append({"theme": t, "coding": version,
                         **with_ci(h[f"{t.lower()}_mention"].to_numpy(), enc[f"{t.lower()}_{version}"].to_numpy(),
                                   v["n_boot"], rng)})
        harmonized = themes[(themes["theme"] == t) & (themes["measure"] == "present")].iloc[0]
        rows.append({"theme": t, "coding": "harmonized", **harmonized.drop(["theme", "measure", "n"]).to_dict()})
    encs = pd.DataFrame(rows)
    encs.to_csv(os.path.join(v["dir"], "agreement_encoders.csv"), index=False)

    coded_subs = h[[f"{t}_subthemes" for t in cfg.themes]].apply(lambda r: "".join(r), axis=1).str.strip() != ""
    sub_rows, micro_f1, tp, fp, fn = [], np.nan, 0, 0, 0
    if coded_subs.any():
        hs = h[coded_subs]
        ls = llm.set_index("item_id").loc[hs["item_id"]]
        for _, s in codebook(cfg).iterrows():
            human = hs[f"{s['theme']}_subthemes"].str.upper().str.replace(" ", "").str.split(";").apply(
                lambda codes: s["subtheme_id"].upper() in codes).astype(int).to_numpy()
            machine = ls[f"sub_{s['subtheme_id']}"].to_numpy()
            tp += int(((human == 1) & (machine == 1)).sum())
            fp += int(((human == 0) & (machine == 1)).sum())
            fn += int(((human == 1) & (machine == 0)).sum())
            sub_rows.append({**s, "n": len(hs), **{k: round(x, 3) for k, x in measures(human, machine).items()}})
    subs = pd.DataFrame(sub_rows)
    if len(subs):
        subs.to_csv(os.path.join(v["dir"], "agreement_subthemes.csv"), index=False)
        micro_f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    ex = pd.read_csv(cfg.path("final"), dtype={"item_id": str})
    notes = dict(zip(sheet["item_id"].astype(str), sheet["notes"])) if "notes" in sheet else {}
    dis = []
    for t in cfg.themes:
        col = f"{t.lower()}_mention"
        for i in np.where(h[col].to_numpy() != llm[col].to_numpy())[0]:
            item = h["item_id"].iloc[i]
            snippets = ex[(ex["item_id"] == item) & (ex["theme"] == t)]["snippet"].tolist()
            dis.append({"item_id": item, "theme": t, "human": int(h[col].iloc[i]), "llm": int(llm[col].iloc[i]),
                        "llm_snippets": " | ".join(snippets), "notes": notes.get(item, "")})
    pd.DataFrame(dis).to_csv(os.path.join(v["dir"], "disagreements.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\nHUMAN vs LLM (harmonized), per theme")
    print(themes[["theme", "measure", "agreement", "kappa", "kappa_ci_low", "kappa_ci_high", "ac1", "prev_human",
                  "prev_llm", "sensitivity", "precision"]].to_string(index=False))
    print("\nPresence: each encoder, union, intersection and harmonized coding vs human (kappa)")
    print(encs.pivot(index="theme", columns="coding", values="kappa")[["enc1", "enc2", "either", "both", "harmonized"]]
          .to_string())
    if len(subs):
        print(f"\nSub-themes ({len(subs)} sub-themes, {int(coded_subs.sum())} items): micro-F1 = {micro_f1:.3f}")
        print(subs[["subtheme_id", "subtheme", "kappa", "prev_human", "prev_llm"]].to_string(index=False))
    print(f"\nDisagreements on presence: {len(dis)} -> {os.path.join(v['dir'], 'disagreements.csv')}")
    latex(cfg, themes, encs, len(h), os.path.join(v["dir"], "tab_validation.tex"))


def latex(cfg, themes, encs, n, path):
    rows = []
    for t in cfg.themes:
        cells = [t.replace("_", " ").title()]
        for m in ("present", "positive mention", "negative mention"):
            r = themes[(themes["theme"] == t) & (themes["measure"] == m)].iloc[0]
            cells.append(f"{r['kappa']:.2f} [{r['kappa_ci_low']:.2f}, {r['kappa_ci_high']:.2f}]")
        r = themes[(themes["theme"] == t) & (themes["measure"] == "present")].iloc[0]
        cells += [f"{r['ac1']:.2f}", f"{100 * r['prev_human']:.0f} / {100 * r['prev_llm']:.0f}",
                  f"{r['sensitivity']:.2f} / {r['precision']:.2f}"]
        e = encs[encs["theme"] == t].set_index("coding")["kappa"]
        cells.append(f"{e['enc1']:.2f} / {e['enc2']:.2f}")
        rows.append(" & ".join(cells) + r" \\")
    m = cfg["models"]
    text = (r"""\begin{table*}[t]
  \centering
  \footnotesize
  \caption{Agreement between a human coder (blind to the LLM coding) and the final LLM coding on """ + f"{n}" +
            r""" items: Cohen's $\kappa$ with 95\% bootstrap CI for the presence of the theme and for a positive or
  negative mention, Gwet's AC1 for presence, share of items coded as mentioning the theme by the human and by the
  LLM (\%), sensitivity and precision of the LLM against the human, and $\kappa$ of each encoder alone before
  harmonization (""" + f"{m['encoder_1']} / {m['encoder_2']}".replace("_", r"\_") + r""").}
  \label{tab:validation}
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lccccccc@{}}
    \toprule
    Theme & $\kappa$ present & $\kappa$ positive & $\kappa$ negative & AC1 & Prev.\ H / LLM & Sens.\ / Prec. & $\kappa$ enc.\ 1 / 2 \\
    \midrule
""" + "\n".join("    " + r for r in rows) + r"""
    \bottomrule
  \end{tabular}
\end{table*}
""")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"LaTeX table: {path}")


def main():
    parser = argparse.ArgumentParser(description="Human validation of the LLM coding")
    parser.add_argument("step", choices=["sample", "agreement"])
    parser.add_argument("--config", default=C.DEFAULT_CONFIG)
    parser.add_argument("--per-group", type=int, default=None)
    parser.add_argument("--subset-per-group", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing coding sheet")
    args = parser.parse_args()
    cfg = C.load_config(args.config)
    v = settings(cfg)
    if args.step == "sample":
        make_sample(cfg, v, args)
    else:
        agreement(cfg, v)


if __name__ == "__main__":
    main()

"""
Stratified random sample of the items kept by the filter (between Phase 0 and Phase 1), with weights and a
representativeness check. Optional: skip it to encode every filtered item.

Works with any source (reviews, survey answers, interview segments, Reddit or forum comments): groups and strata
are fields of the input, set in the section `sampling` of the configuration:
  - `group_field`: the groups sampled separately (e.g. `game` for several Steam games, the question id for a survey
    with several open questions, `subreddit` or `thread_id` for Reddit, the participant for interviews; null = one
    group);
  - `n_per_group`: up to this many items per group; a group with fewer items is taken in full (census);
    null = census of every group;
  - `oversample` (optional): {"field", "value", "share"}. The items whose `field` equals `value` (or one of the values,
    when `value` is a list) make up `share` of each group's sample (all of them when fewer are available). E.g.
    voted_up = false (negative Steam reviews), a Likert item with value [1, 2] (dissatisfied respondents), a score
    field for Reddit. Strata are then group x (oversampled / other); without oversampling the stratum is the group;
  - `seed`: simple random sampling without replacement inside each stratum (items ordered by id first, so the
    sample does not depend on the order of the input file).
Each sampled item gets an inverse-probability weight w = N_stratum / n_stratum, which restores each group's mix of
the population of filtered items. Use the weights for prevalence estimates; the analyses decide when to use them.

`context_template` (optional) writes in every sampled item a field `sample_context`, shown to the models as context
and never coded, e.g. "Steam review of the game {group}". Placeholders: {group}, {context} (the item's original
context, e.g. the survey question or the Reddit thread, empty if none) and the names in `fields.metadata`.

`balance_fields` lists numeric fields compared between the weighted sample and the population, each
{"name", "field", "transform"}: transform "log1p" (skewed counts: playtime, score, replies), "year" (unix timestamp
-> year with decimals) or null (e.g. a Likert score, age). Text length (log characters), the share of the most
frequent language and the share of the oversampled value are always compared.

Outputs (`sampling.output_dir`, relative to the configuration file):
  sample_items.json              the sampled items, same format as the input (+ `sample_context`)
  sample_weights.csv             id, group, stratum, stratum sizes, inclusion probability, weight
  sample_design.csv              per stratum: population, sample, sampling fraction, weight
  sample_representativeness.csv  per group: population vs weighted sample mean, standardized mean difference
                                 (|SMD| < 0.1 = negligible difference)
and <config>_sample.json next to the configuration: a copy whose paths point to the sample, with outputs in
`sampling.output_dir`. Run 1_encoding.py and 2_harmonization.py with `--config <config>_sample.json` (not
0_filtering.py: the filter is already done). The copy is rewritten at every run: edit only the original
configuration and run this script again after any change; the sample is identical (same seed).

Usage: python stratified_sample.py [--config study_config.json] [--n-per-group N] [--share S] [--seed K]
"""
import argparse
import json
import os
import re
from string import Formatter

import numpy as np
import pandas as pd

import common as C

BBCODE = re.compile(r"\[/?[a-z0-9*]+(=[^\]]*)?\]", re.IGNORECASE)
DEFAULTS = {"group_field": None, "n_per_group": None, "oversample": None, "seed": 42, "context_template": None,
            "balance_fields": [], "output_dir": "sample"}


def settings(cfg: C.Config, args) -> dict:
    s = {**DEFAULTS, **cfg.get("sampling", {})}
    if args.n_per_group is not None:
        s["n_per_group"] = args.n_per_group
    if args.seed is not None:
        s["seed"] = args.seed
    if args.share is not None:
        if not s["oversample"]:
            raise SystemExit("--share needs sampling.oversample in the configuration.")
        s["oversample"] = {**s["oversample"], "share": args.share}
    if s["context_template"]:
        allowed = {"group", "context"} | set(cfg["fields"].get("metadata", {}))
        unknown = {f for _, f, _, _ in Formatter().parse(s["context_template"]) if f} - allowed
        if unknown:
            raise SystemExit(f"sampling.context_template: unknown placeholders {sorted(unknown)} "
                             f"(allowed: {sorted(allowed)})")
    return s


def group_of(cfg, s, item) -> str:
    return str(C.get_field(item, s["group_field"], "missing")) if s["group_field"] else "ALL"


def same_value(value, target) -> bool:
    """Numbers compare as numbers (2 == "2" == "2.0", e.g. Likert answers read from CSV), the rest as text
    without case (False == "false")."""
    try:
        return float(value) == float(target)
    except (TypeError, ValueError):
        return str(value).strip().lower() == str(target).strip().lower()

def is_oversampled(s, item) -> bool:
    over = s["oversample"]
    if not over:
        return False
    targets = over["value"] if isinstance(over["value"], list) else [over["value"]]
    value = C.get_field(item, over["field"])
    return any(same_value(value, t) for t in targets)


def oversample_label(s) -> str:
    over = s["oversample"]
    value = "|".join(map(str, over["value"])) if isinstance(over["value"], list) else over["value"]
    return f"{over['field']}={value}"


def allocate(n_rest, n_over, n_per_group, share):
    """Sample sizes (other, oversampled) for one group."""
    if n_per_group is None or n_rest + n_over <= n_per_group:
        return n_rest, n_over                                  # census
    take_over = min(n_over, int(round(share * n_per_group)))
    take_rest = min(n_rest, n_per_group - take_over)
    return take_rest, take_over


def draw(cfg, items, s):
    rng = np.random.default_rng(s["seed"])
    df = pd.DataFrame({"item_id": [C.item_id(cfg, r) for r in items],
                       "group": [group_of(cfg, s, r) for r in items],
                       "oversampled": [is_oversampled(s, r) for r in items]})
    if df["item_id"].duplicated().any():
        raise SystemExit("Item ids are not unique: the sample cannot be drawn.")
    df["pos"] = range(len(df))
    df["id_order"] = df["item_id"].str.zfill(20)
    df = df.sort_values(["group", "oversampled", "id_order"])
    chosen, design = [], []
    for group, g in df.groupby("group", sort=True):
        if s["oversample"]:
            rest, over = g[~g["oversampled"]], g[g["oversampled"]]
            take_rest, take_over = allocate(len(rest), len(over), s["n_per_group"], s["oversample"]["share"])
            strata = (("other", rest, take_rest), (oversample_label(s), over, take_over))
        else:
            take = len(g) if s["n_per_group"] is None else min(len(g), s["n_per_group"])
            strata = (("all", g, take),)
        for label, stratum, take in strata:
            idx = rng.choice(len(stratum), size=take, replace=False) if take < len(stratum) else np.arange(len(stratum))
            part = stratum.iloc[np.sort(idx)].copy()
            part["stratum"], part["N_stratum"], part["n_stratum"] = label, len(stratum), take
            chosen.append(part)
            design.append({"group": group, "stratum": label, "population": len(stratum), "sample": take,
                           "sampling_fraction_%": round(100 * take / max(len(stratum), 1), 1),
                           "weight": round(len(stratum) / take, 4) if take else None})
    sample = pd.concat(chosen)
    sample["inclusion_probability"] = sample["n_stratum"] / sample["N_stratum"]
    sample["weight"] = 1 / sample["inclusion_probability"]
    return sample, pd.DataFrame(design)


def transform(value, kind):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    if kind == "log1p":
        return np.log1p(max(value, 0))
    if kind == "year":
        t = pd.Timestamp(value, unit="s")
        return t.year + t.dayofyear / 366
    return value


def features(cfg, items, s):
    languages = pd.Series([C.item_language(cfg, r) for r in items])
    top_language = languages.mode().iloc[0] if languages.notna().any() else None
    rows = []
    for r in items:
        text = C.item_text(cfg, r)
        if cfg["filter"].get("strip_bbcode", True):
            text = BBCODE.sub("", text)
        row = {"item_id": C.item_id(cfg, r), "group": group_of(cfg, s, r),
               "log_text_chars": np.log1p(len(" ".join(text.split())))}
        if s["oversample"]:
            row[oversample_label(s)] = float(is_oversampled(s, r))
        if top_language is not None:
            row[f"language={top_language}"] = float(C.item_language(cfg, r) == top_language)
        for b in s["balance_fields"]:
            row[b["name"]] = transform(C.get_field(r, b["field"]), b.get("transform"))
        rows.append(row)
    return pd.DataFrame(rows)


def representativeness(pop, sample):
    """Per group: population mean vs weighted sample mean, and standardized mean difference."""
    variables = [c for c in pop.columns if c not in ("item_id", "group")]
    s_all = sample[["item_id", "weight"]].merge(pop, on="item_id")
    groups = sorted(pop["group"].unique())
    rows = []
    for group in groups + (["ALL"] if len(groups) > 1 else []):
        p = pop if group == "ALL" else pop[pop["group"] == group]
        q = s_all if group == "ALL" else s_all[s_all["group"] == group]
        for v in variables:
            ok = q[v].notna()
            pm, psd = p[v].mean(), p[v].std()
            wm = np.average(q.loc[ok, v], weights=q.loc[ok, "weight"]) if ok.any() else np.nan
            rows.append({"group": group, "variable": v, "population_mean": round(pm, 3),
                         "sample_mean_unweighted": round(q[v].mean(), 3), "sample_mean_weighted": round(wm, 3),
                         "SMD_weighted": round((wm - pm) / psd, 3) if psd > 0 else 0.0,
                         "n_population": len(p), "n_sample": len(q)})
    return pd.DataFrame(rows)


def write_sample_config(config_path, s, sample_dir):
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    rel = os.path.relpath(sample_dir, os.path.dirname(os.path.abspath(config_path)))
    raw["_note"] = (f"Generated by stratified_sample.py from {os.path.basename(config_path)}: edit that file and run "
                    "stratified_sample.py again. Paths point to the stratified sample; Phase 0 (filter) is already done.")
    raw["paths"]["raw_input"] = os.path.join(rel, "sample_items.json")
    raw["paths"]["output_dir"] = rel
    raw["paths"]["files"]["filtered"] = "sample_items.json"
    if s["context_template"]:
        raw["fields"]["context"] = "sample_context"
    stem, ext = os.path.splitext(config_path)
    out = f"{stem}_sample{ext}"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser(description="Stratified random sample of the filtered items")
    parser.add_argument("--config", default=C.DEFAULT_CONFIG)
    parser.add_argument("--n-per-group", type=int, default=None)
    parser.add_argument("--share", type=float, default=None, help="Share of the oversampled value in each group")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = C.load_config(args.config)
    s = settings(cfg, args)
    items = C.load_json(cfg.path("filtered"))
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(args.config)), s["output_dir"])
    os.makedirs(sample_dir, exist_ok=True)

    sample, design = draw(cfg, items, s)
    by_id = {C.item_id(cfg, r): r for r in items}
    sample_items = [dict(by_id[i]) for i in sample.sort_values("pos")["item_id"]]
    if s["context_template"]:                    # shown to the models, never coded
        for r in sample_items:
            values = {**{k: ("" if v is None else v) for k, v in C.item_metadata(cfg, r).items()},
                      "group": group_of(cfg, s, r), "context": C.item_context(cfg, r) or ""}
            r["sample_context"] = s["context_template"].format(**values).strip()
    C.save_json(os.path.join(sample_dir, "sample_items.json"), sample_items)
    sample[["item_id", "group", "stratum", "N_stratum", "n_stratum", "inclusion_probability", "weight"]].to_csv(
        os.path.join(sample_dir, "sample_weights.csv"), index=False)
    design.to_csv(os.path.join(sample_dir, "sample_design.csv"), index=False)
    rep = representativeness(features(cfg, items, s), sample)
    rep.to_csv(os.path.join(sample_dir, "sample_representativeness.csv"), index=False)
    sample_config = write_sample_config(args.config, s, sample_dir)

    pd.set_option("display.width", 160)
    t = design.pivot_table(index="group", columns="stratum", values=["population", "sample"], aggfunc="sum")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    if len(t) > 1:
        t.loc["TOTAL"] = t.sum()
    over = f", {oversample_label(s)} {int(100 * s['oversample']['share'])}%" if s["oversample"] else ""
    print(f"Stratified sample (seed {s['seed']}, up to {s['n_per_group'] or 'all'} per group{over}): "
          f"{len(sample)} of {len(items)} items\n")
    print(t.fillna(0).astype(int).to_string())
    check = rep[rep["variable"] != oversample_label(s)] if s["oversample"] else rep
    if len(check):
        worst = check.loc[check["SMD_weighted"].abs().idxmax()]
        print(f"\nRepresentativeness (weighted sample vs filtered population): max |SMD| = "
              f"{abs(worst['SMD_weighted']):.3f} ({worst['group']}, {worst['variable']}); < 0.1 is negligible.")
    print(f"\nSample: {os.path.join(sample_dir, 'sample_items.json')}\n"
          f"Next: python 1_encoding.py --config {sample_config}")


if __name__ == "__main__":
    main()

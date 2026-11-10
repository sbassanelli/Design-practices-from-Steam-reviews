"""
PHASE 3: Sub-theme assignment and checks.

Harmonization assigns a sub-theme while it also adjudicates extracts, codes and sentiment. In our use case this favoured the
sub-theme listed first. This phase replaces only the sub-theme, with the procedure of the codebook test: each distinct code of a
theme is assigned, in batches, by the two evaluation models (harmonization.codebook_induction.evaluation_models).
Rule: the first model's sub-theme; if it finds none, the second model's; if neither finds one, the sub-theme given at
harmonization is kept and flagged. Themes, snippets, codes and sentiment are not changed; identical codes get the same sub-theme.

Two checks, with thresholds fixed in advance:
  order check  the first model assigns again a sample of codes per theme (evaluation_sample, 160 by default) with the list of
               sub-themes reversed; Cohen's kappa between the two orders and share of the first-listed sub-theme in each.
  drift check  for each sub-theme, final share against its share in the codebook test; flagged above 10 percentage points,
               for the harmonizer's assignment and for the new one.

Outputs, next to the final file: <final>_reassigned.csv, <summary>_reassigned.csv, subtheme_assignment_report.csv,
subtheme_order_check.csv, subtheme_drift_check.csv. The output of harmonization is not modified. Resumable (checkpoint).

Usage: python 3_subtheme_assignment.py --config study_config.json
"""
import asyncio
import importlib
import os

import httpx
import numpy as np
import pandas as pd

import common as K

H = importlib.import_module("2_harmonization")
FLAG = 10.0


def test_shares(book, theme, subs):
    """Shares of the sub-themes in the last codebook test that used exactly these sub-themes."""
    for r in reversed(book.get("diagnostics", {}).get(theme, [])):
        if [s["name"] for s in r.get("subthemes", [])] == [s["name"] for s in subs]:
            sh = r["shares_%"]
            total = sum(sh.values()) or 1
            return {k: 100 * v / total for k, v in sh.items()}
    return None


async def main():
    args = K.parse_args("Phase 3: sub-theme assignment and checks")
    cfg = K.load_config(args.config)
    book = K.load_json(cfg.path("codebook"))
    final = cfg.path("final")
    out_dir = os.path.dirname(final)
    df = pd.read_csv(final, dtype={"item_id": str})
    ci = cfg["harmonization"]["codebook_induction"]
    models, n_check = ci["evaluation_models"], ci.get("evaluation_sample", 160)
    ck = os.path.join(out_dir, "subtheme_assignment_checkpoint.json")
    key = H.codebook_hash(book)
    state = K.load_json(ck) if os.path.exists(ck) else {}
    if state.get("key") != key:
        state = {"key": key, "done": {}, "reversed": {}}
    print(f"🚀 PHASE 3: Sub-theme assignment | {len(df)} extracts | models {', '.join(models)}")

    async with httpx.AsyncClient(timeout=cfg["ollama"]["timeout"]) as client:
        for model in models:
            for theme, subs in book["themes"].items():
                tag = f"{model}|{theme}"
                if tag in state["done"]:
                    continue
                codes = sorted(df.loc[df["theme"] == theme, "code"].astype(str).unique())
                state["done"][tag] = dict(zip(codes, await H.assign_sample(cfg, client, model, theme, subs, codes)))
                K.save_json(ck, state)
                print(f"   {model} {theme}: {len(codes)} codes", flush=True)
            await K.unload_model(cfg, client, model)

        # order check: the first model, list of sub-themes reversed
        rng = np.random.default_rng(cfg["ollama"]["seed"])
        for theme, subs in book["themes"].items():
            if theme in state["reversed"]:
                continue
            vc = df.loc[df["theme"] == theme, "code"].astype(str).value_counts()
            names, p = vc.index.tolist(), (vc / vc.sum()).to_numpy()
            codes = [names[i] for i in rng.choice(len(names), size=min(n_check, len(names)), replace=False, p=p)]
            state["reversed"][theme] = dict(zip(codes, await H.assign_sample(cfg, client, models[0], theme, subs[::-1], codes)))
            K.save_json(ck, state)
            print(f"   order check {theme}: {len(codes)} codes", flush=True)
        await K.unload_model(cfg, client, models[0])

    report, frames, order, drift = [], [], [], []
    for theme, subs in book["themes"].items():
        names = {s["id"]: s["name"] for s in subs}
        a, b = state["done"][f"{models[0]}|{theme}"], state["done"][f"{models[1]}|{theme}"]
        s = df[df["theme"] == theme].copy()
        s["original_subtheme_id"], s["original_subtheme"] = s["subtheme_id"], s["subtheme"]
        src, new = [], []
        for code, old in zip(s["code"].astype(str), s["subtheme_id"]):
            if a[code] != "NONE":
                new.append(a[code]); src.append(models[0])
            elif b[code] != "NONE":
                new.append(b[code]); src.append(models[1])
            else:
                new.append(old); src.append("both_none_kept_original")
        s["subtheme_id"], s["assignment_source"] = new, src
        s["subtheme"] = s["subtheme_id"].map(names)
        frames.append(s)
        codes = list(a)
        x, y = [a[c] for c in codes], [b[c] for c in codes]
        report.append({"theme": theme, "distinct_codes": len(codes), "kappa": round(H.kappa(x, y), 3),
                       "unassigned_%": round(100 * (x + y).count("NONE") / (2 * len(codes)), 1),
                       "both_none_extracts": src.count("both_none_kept_original"),
                       "changed_extracts_%": round(100 * (s["subtheme_id"] != s["original_subtheme_id"]).mean(), 1)})
        rev = state["reversed"][theme]
        fwd, back = [a[c] for c in rev], list(rev.values())
        first = subs[0]["id"]
        order.append({"theme": theme, "codes": len(rev), "kappa_forward_reversed": round(H.kappa(fwd, back), 3),
                      "identical_%": round(100 * np.mean([u == v for u, v in zip(fwd, back)]), 1),
                      "first_listed_forward_%": round(100 * fwd.count(first) / len(fwd), 1),
                      "first_listed_reversed_%": round(100 * back.count(first) / len(back), 1)})
        shares = test_shares(book, theme, subs)
        for i, x_ in enumerate(subs):
            test = shares.get(x_["id"], 0) if shares else None
            old, newp = 100 * (s["original_subtheme_id"] == x_["id"]).mean(), 100 * (s["subtheme_id"] == x_["id"]).mean()
            drift.append({"theme": theme, "subtheme_id": x_["id"], "subtheme": x_["name"], "position": i + 1,
                          "test_%": None if test is None else round(test, 1), "harmonizer_%": round(old, 1), "reassigned_%": round(newp, 1),
                          "harmonizer_flag": test is not None and abs(old - test) > FLAG, "reassigned_flag": test is not None and abs(newp - test) > FLAG})

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(final.replace(".csv", "_reassigned.csv"), index=False, encoding="utf-8-sig")
    H.build_summary(cfg, out, book).to_csv(cfg.path("summary").replace(".csv", "_reassigned.csv"), index=False, encoding="utf-8-sig")
    for name, rows in (("subtheme_assignment_report", report), ("subtheme_order_check", order), ("subtheme_drift_check", drift)):
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, name + ".csv"), index=False, encoding="utf-8-sig")
    d = pd.DataFrame(drift)
    print(pd.DataFrame(report).to_string(index=False))
    print(pd.DataFrame(order).to_string(index=False))
    print(f"💾 drift flags: harmonizer {int(d['harmonizer_flag'].sum())}, reassigned {int(d['reassigned_flag'].sum())} of {len(d)} sub-themes")


if __name__ == "__main__":
    asyncio.run(main())

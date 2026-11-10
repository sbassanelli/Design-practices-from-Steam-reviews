"""
PHASE 4: Saturation of the coded sample (new-information criterion of Guest et al., 2020).

Codes are grouped into code types by meaning (average-linkage clustering of their embeddings, cosine distance 0.15;
0.10 and 0.20 as sensitivity). Over 200 random orderings of the coded reviews, saturation is the first n after which the next
50 reviews add at most 5% new code types (also 25 and 100 reviews, 5% and 10%). Also reported: new code types in the last 100
reviews, and the number of reviews needed for every sub-theme to reach five extracts.
Saturation says whether the sample holds the kinds of comments of the corpus. It does not say whether the sample is large enough
to weigh them in the design brief, which in our use case needed far more reviews.

Outputs next to the final file: saturation_points.csv, saturation_curves.csv, saturation_sensitivity.csv.
Usage: python 4_saturation.py --config study_config.json
"""
import asyncio
import os

import httpx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

import common as K

B, STEP, RUN, TAIL, THRESHOLD, MIN_EXTRACTS, DISTANCE = 200, 25, 50, 100, 0.05, 5, 0.15
DISTANCES, RUNS, THRESHOLDS = [0.10, 0.15, 0.20], [25, 50, 100], [0.05, 0.10]


def sat(c, run, thr, n_total):
    return next((n for n in range(STEP, n_total - run + 1, STEP) if c[n - 1] and (c[n - 1 + run] - c[n - 1]) / c[n - 1] <= thr), None)


async def main():
    args = K.parse_args("Phase 4: saturation")
    cfg = K.load_config(args.config)
    final = cfg.path("final")
    src = final.replace(".csv", "_reassigned.csv") if os.path.exists(final.replace(".csv", "_reassigned.csv")) else final
    out_dir = os.path.dirname(final)
    ex = pd.read_csv(src, dtype={"item_id": str})
    ex["code_norm"] = ex["code"].astype(str).str.lower().str.strip()
    reviews = [K.item_id(cfg, it) for it in K.load_items(cfg.path("filtered"))]
    ci = cfg["harmonization"]["codebook_induction"]
    codes = sorted(ex["code_norm"].unique())
    print(f"🚀 PHASE 4: Saturation | {len(reviews)} reviews | {len(codes)} distinct codes")
    async with httpx.AsyncClient(timeout=cfg["ollama"]["timeout"]) as client:
        X = np.array(await K.embed(cfg, client, ci["embedding_model"], codes, prefix=ci.get("embedding_prefix", "")))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Z = linkage(X, "average", metric="cosine")
    types = {t: dict(zip(codes, fcluster(Z, t, criterion="distance"))) for t in DISTANCES}
    by_type = {t: ex.assign(ty=ex["code_norm"].map(m)).groupby("item_id")["ty"].apply(set).to_dict() for t, m in types.items()}
    by_sub = ex.groupby("item_id")["subtheme_id"].apply(list).to_dict()
    sub_ids = sorted(ex["subtheme_id"].unique())
    index = {s: i for i, s in enumerate(sub_ids)}
    want = [index[s] for s in sub_ids if (ex["subtheme_id"] == s).sum() >= MIN_EXTRACTS]
    N, rng = len(reviews), np.random.default_rng(cfg["ollama"]["seed"])
    points, sens, curves = [], [], []
    for b in range(B):
        order = rng.permutation(reviews)
        cum = {}
        for t in DISTANCES:
            seen, c = set(), []
            for rid in order:
                seen |= by_type[t].get(rid, set())
                c.append(len(seen))
            cum[t] = np.array(c)
            for run in RUNS:
                for thr in THRESHOLDS:
                    sens.append({"type_distance": t, "run": run, "threshold": thr, "saturation": sat(cum[t], run, thr, N)})
        c = cum[DISTANCE]
        counts, first, pos = np.zeros(len(sub_ids)), None, 0
        for n in range(STEP, N + 1, STEP):
            for rid in order[pos:n]:
                for s in by_sub.get(rid, []):
                    counts[index[s]] += 1
            pos = n
            if first is None and (counts[want] >= MIN_EXTRACTS).all():
                first = n
            curves.append({"perm": b, "n": n, "code_types": c[n - 1]})
        points.append({"perm": b, "saturation": sat(c, RUN, THRESHOLD, N),
                       "tail_new_types_%": 100 * (c[-1] - c[-TAIL - 1]) / c[-1] if N > TAIL else None,
                       "subthemes_min5": first, "code_types_total": c[-1]})
    P, S, C = pd.DataFrame(points), pd.DataFrame(sens), pd.DataFrame(curves)
    P.to_csv(os.path.join(out_dir, "saturation_points.csv"), index=False)
    C.groupby("n")["code_types"].quantile([0.05, 0.5, 0.95]).unstack().to_csv(os.path.join(out_dir, "saturation_curves.csv"))
    table = S.groupby(["type_distance", "run", "threshold"])["saturation"].agg(
        median="median", q05=lambda x: x.quantile(0.05), q95=lambda x: x.quantile(0.95),
        not_reached_pct=lambda x: 100 * x.isna().mean()).reset_index()
    table.to_csv(os.path.join(out_dir, "saturation_sensitivity.csv"), index=False)
    print(f"   Saturation (next {RUN} reviews, at most {THRESHOLD:.0%} new code types): median {P['saturation'].median()} reviews, "
          f"5th-95th percentile {P['saturation'].quantile(0.05)}-{P['saturation'].quantile(0.95)}, not reached in {100 * P['saturation'].isna().mean():.1f}% of orderings")
    print(f"   Every sub-theme with five extracts: median {P['subthemes_min5'].median()} reviews")
    print(f"💾 saturation_points.csv, saturation_curves.csv, saturation_sensitivity.csv in {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())

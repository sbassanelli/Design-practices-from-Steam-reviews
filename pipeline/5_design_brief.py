"""
PHASE 5: Design brief. The last step of the pipeline: from the coded extracts to a short PDF of design requirements for a design team.

1. Triage (computed, no model). For each sub-theme: reviews that mention it, tone of its extracts, and odds ratio of the outcome
   (e.g. recommending the game) for reviews that mention it (Haldane-corrected, Fisher's exact test, Benjamini-Hochberg).
   Repair first = more negative than positive extracts and lower odds; Accepted friction = more negative, higher odds;
   Protect = more positive, higher odds; Contested = more positive, lower odds; Not decisive = no significant association.
2. Drafts (one model call per priority). The model reads the evidence and up to 40 numbered extracts and writes a short requirement,
   the reason, a possible change and what to check in later reviews. It picks quotes by number, so quotes stay verbatim;
   quotes in another language get a translation, marked as such.
3. Synthesis (one model call). The model reads all drafts together, writes a three-sentence summary and flags tensions between
   priorities. It may only cite the IDs of the priorities, which are checked.
4. PDF. Numbers, quotes and layout come from this script, never from the model. The PDF is compiled with tectonic, xelatex or
   pdflatex if one is installed; otherwise the .tex file is left ready to compile.

Outputs in the output directory: design_brief.json (all evidence, drafts, prompts' inputs and the source of every quote),
design_brief.tex, design_brief.pdf.

Optional "design_brief" block in the configuration (defaults in brackets):
  model [models.harmonizer], language of the brief [English], outcome: metadata key of a 0/1 outcome [voted_up],
  outcome_label [recommending the game], alpha [0.05], max_extracts [40], title [Design brief], input: extracts file
  [the reassigned file if present, otherwise the final file]. Written for languages in Latin script.

Usage: python 5_design_brief.py --config study_config_sample.json
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from datetime import date

import httpx
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import common as K

GROUPS = ["Repair first", "Accepted friction", "Protect", "Contested"]
PREFIX = {"Repair first": "R", "Accepted friction": "A", "Protect": "P", "Contested": "C"}
COLOR = {"Repair first": "C4623F", "Accepted friction": "6B4FBB", "Protect": "2A78D6", "Contested": "A8862B"}
LIGHT = {"Repair first": "FBE1D6", "Accepted friction": "E9E3F7", "Protect": "DCEAFA", "Contested": "FFF3C4"}
MEANING = {"Repair first": "criticized, and goes with a worse outcome", "Accepted friction": "criticized, yet goes with a better outcome",
           "Protect": "praised, and goes with a better outcome", "Contested": "praised, yet goes with a worse outcome"}
TASK = {"Repair first": "Players criticize this aspect and criticism goes with a worse outcome. Write a requirement that resolves the problem. "
                        "The requirement must start with 'The game must'. The change must address what the extracts complain about.",
        "Accepted friction": "Players complain about this aspect, yet those who mention it have a better outcome, so it may be part of what they value. "
                             "The requirement must start with 'Keep'. The change may only concern how the aspect is communicated or framed to players. "
                             "Do NOT propose making it easier, softer, shorter, more gradual or adaptive.",
        "Protect": "Players praise this aspect and it goes with a better outcome. The requirement must start with 'Preserve'. "
                   "The change must be a safeguard: what the team should verify when it modifies the game so that this aspect is not damaged. "
                   "Do NOT propose adding, expanding or enhancing content.",
        "Contested": "Players praise this aspect, yet those who mention it have a worse outcome. The requirement must start with 'Investigate'. "
                     "The change must be a question the team should answer before acting."}
START = {"Repair first": "The game must", "Accepted friction": "Keep", "Protect": "Preserve", "Contested": "Investigate"}
DRAFT_SCHEMA = {"type": "object", "properties": {
    "requirement": {"type": "string"}, "why": {"type": "string"}, "proposed_change": {"type": "string"}, "check": {"type": "string"},
    "quotes": {"type": "array", "items": {"type": "object", "properties": {"extract": {"type": "integer"}, "translation": {"type": "string"}},
                                          "required": ["extract", "translation"]}}},
    "required": ["requirement", "why", "proposed_change", "check", "quotes"]}
SUMMARY_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
PAIR_SCHEMA = {"type": "object", "properties": {
    "change_element": {"type": "string"}, "affected_element": {"type": "string"}, "conflict": {"type": "boolean"},
    "if_then": {"type": "string"}, "watch": {"type": "string"}},
    "required": ["change_element", "affected_element", "conflict", "if_then", "watch"]}


# ---------------------------------------------------------------- triage
def triage(cfg, ex, outcome_key, alpha):
    items = K.load_items(cfg.path("filtered"))
    meta = cfg["fields"]["metadata"]
    if outcome_key not in meta:
        raise SystemExit(f"design_brief.outcome '{outcome_key}' is not a metadata field of the configuration")
    rows = []
    for it in items:
        v = K.get_field(it, meta[outcome_key])
        if v is not None:
            rows.append({"item_id": K.item_id(cfg, it), "outcome": int(bool(v))})
    d = pd.DataFrame(rows).drop_duplicates("item_id").set_index("item_id")
    ex = ex[ex["item_id"].isin(d.index)]
    out = []
    for (theme, sid, name), s in ex.groupby(["theme", "subtheme_id", "subtheme"]):
        flag = d.index.isin(s["item_id"])
        y = d["outcome"].values
        a, b, c, e = [int(((flag == f) & (y == o)).sum()) for f, o in ((1, 1), (1, 0), (0, 1), (0, 0))]
        a2, b2, c2, e2 = [x + 0.5 for x in (a, b, c, e)]
        lor, se = np.log(a2 * e2 / (b2 * c2)), np.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / e2)
        critics = d.loc[d.index.isin(s.loc[s["polarity"] == "Negative", "item_id"]), "outcome"]
        out.append({"theme": theme, "subtheme_id": sid, "subtheme": name, "extracts": len(s), "reviews": a + b,
                    "positive_%": round(100 * (s["polarity"] == "Positive").mean(), 1), "negative_%": round(100 * (s["polarity"] == "Negative").mean(), 1),
                    "or": float(np.exp(lor)), "or_low": float(np.exp(lor - 1.96 * se)), "or_high": float(np.exp(lor + 1.96 * se)),
                    "p": stats.fisher_exact([[a, b], [c, e]])[1], "critics": len(critics),
                    "critics_outcome_%": round(100 * critics.mean(), 1) if len(critics) else None})
    t = pd.DataFrame(out)
    t["p_bh"] = multipletests(t["p"], method="fdr_bh")[1]

    def group(r):
        if r.p_bh >= alpha:
            return "Not decisive"
        negative = r["negative_%"] > r["positive_%"]
        if r["or"] < 1:
            return "Repair first" if negative else "Contested"
        return "Accepted friction" if negative else "Protect"

    t["group"] = t.apply(group, axis=1)
    return t, len(d), round(100 * d["outcome"].mean(), 1)


def pick(g, group, n):
    """Extracts of the polarity that defines the group: one per code first; half the most frequent codes, half spread over the rarer ones."""
    main = "Positive" if group in ("Protect", "Contested") else "Negative"
    x = g[g["polarity"] == main].copy()
    x["code_norm"] = x["code"].astype(str).str.lower()
    x["freq"] = x.groupby("code_norm")["code_norm"].transform("size")
    x["turn"] = x.groupby("code_norm").cumcount()
    x = x.sort_values(["turn", "freq", "item_id", "snippet"], ascending=[True, False, True, True]).drop_duplicates("snippet").reset_index(drop=True)
    if len(x) <= n:
        return x
    half = n // 2
    step = (len(x) - half) / half
    return pd.concat([x.head(half), x.iloc[[half + int(i * step) for i in range(half)]]]).reset_index(drop=True)


# ---------------------------------------------------------------- prompts
def draft_prompt(cfg, r, x, n_reviews, base, lang, label):
    listing = "\n".join(f"[{i + 1}] ({e.language}) {str(e.snippet)[:300]}" for i, e in enumerate(x.itertuples()))
    return f"""You help a game design team turn player reviews into short design requirements.
Game: {cfg['study']['context']}
Aspect: {r['subtheme']} (theme {r['theme']})
Evidence: mentioned in {r['reviews']} of {n_reviews} reviews; {r['negative_%']}% of its extracts are negative and {r['positive_%']}% positive; of the reviews criticizing it, {r['critics_outcome_%']}% end up {label} ({base}% of all reviews do).
Task: {TASK[r['group']]}

Extracts from reviews, in their original language:
{listing}

Rules:
- Write in {lang}. Be brief: a designer must read it in seconds.
- Base everything only on these extracts. Name concrete objects they mention (a key, a cost, a moment, a place). If they stay general, stay general.
- Do not add facts, numbers or features that the extracts do not contain, and do not introduce systems no extract mentions (for example dynamic difficulty or adaptive scaling).
- requirement: one sentence, at most 25 words, testable.
- why: one sentence, at most 25 words, on what players experience; no numbers.
- proposed_change: at most 30 words. It is a hypothesis.
- check: one sentence, at most 25 words, on what reviews written after the change should show. If the task is to keep or preserve an aspect, say what should not get worse.
- quotes: the 2 extracts that best support the requirement, by number, preferring extracts written in {lang} when they are equally good. For each, give a faithful translation into {lang}; if it is already in {lang}, repeat it exactly."""


def candidate_pairs(drafts):
    """Possible tensions, chosen by rule: a priority to repair and one to keep or preserve in the same theme."""
    fix = [d for d in drafts if d["group"] in ("Repair first", "Contested")]
    keep = [d for d in drafts if d["group"] in ("Accepted friction", "Protect")]
    return [(f["id"], k["id"]) for f in fix for k in keep if f["theme"] == k["theme"]]


def summary_prompt(cfg, drafts, lang):
    listing = "\n".join(f"{d['id']} [{d['group']}] {d['subtheme']}: {d['requirement']}" for d in drafts)
    return f"""You help a game design team read a set of design priorities derived from player reviews.
Game: {cfg['study']['context']}
Groups: "Repair first" = criticized and linked to a worse outcome; "Accepted friction" = criticized yet linked to a better outcome, to keep;
"Protect" = praised and linked to a better outcome, to preserve.

Priorities:
{listing}

Write in {lang} a summary of at most 3 short sentences: what to fix first (name at most three priorities) and what not to touch. Refer to priorities by their ID in brackets, e.g. [R1]."""


def pair_prompt(cfg, fix, keep, lang):
    return f"""You check whether two design priorities for a game conflict.
Game: {cfg['study']['context']}

Priority to repair, {fix['id']} ({fix['subtheme']}):
  Requirement: {fix['requirement']}
  Proposed change: {fix['proposed_change']}
Priority to keep or preserve, {keep['id']} ({keep['subtheme']}):
  Requirement: {keep['requirement']}
  What players value: {keep['why']}

Question: could carrying out the proposed change of {fix['id']} directly weaken what players value in {keep['id']}?
Answer in {lang}:
- change_element: the specific element of the proposed change that would act on the other aspect, quoted or closely paraphrased from the proposed change.
- affected_element: the specific element of {keep['id']} that would be weakened, from its requirement or from what players value.
- conflict: true only if the change acts directly on that element (for example lowering enemy damage directly lowers difficulty). False if the link is indirect, speculative or needs extra assumptions (for example "if enemies become too easy" when the change does not touch enemies).
- if_then: one sentence of at most 30 words in the form "If the team ..., then ..." naming both aspects in words, without IDs.
- watch: one sentence of at most 20 words on what to check in later reviews."""


# ---------------------------------------------------------------- LaTeX
LATIN = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ", "\u2665": "*"}


def tex(s):
    s = "".join(LATIN.get(ch, ch) for ch in str(s))
    s = "".join(ch if ord(ch) < 0x250 else "?" for ch in s)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    s = s.replace(r"\textbackslash\{\}", r"\textbackslash{}")
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    return s.replace('"', "''")


def refs(s):
    """[R1] -> bold ID."""
    return re.sub(r"\[([RAPC]\d+)\]", r"\\textbf{\1}", s)


def evidence(ev, n, base, label):
    s = f"Mentioned in {ev['reviews']} of {n:,} reviews ({100 * ev['reviews'] / n:.0f}\\%); {ev['negative_%']:.0f}\\% of comments negative, {ev['positive_%']:.0f}\\% positive. "
    if ev["critics"]:
        s += f"Of the players who criticize it, {ev['critics_outcome_%']:.0f}\\% end up {tex(label)} (all players: {base:.0f}\\%). "
    lo = "less" if ev["or"] < 1 else "more"
    s += f"Reviews that mention it are {lo} likely to be {tex(label)}" + " {\\color{ink2}\\footnotesize(odds ratio " + f"{ev['or']:.2f}, 95\\% CI {ev['or_low']:.2f}--{ev['or_high']:.2f})" + "}."
    return s


def tension_item(tt):
    verbs = {"Repair first": "Repair", "Contested": "Investigate", "Accepted friction": "Keep", "Protect": "Preserve"}
    head = " vs ".join("\\textbf{" + verbs[g] + " " + i + "} (" + tex(n) + ")" for i, n, g in zip(tt["ids"], tt["names"], tt["groups"]))
    return "  \\item " + head + "\\\\\n" + tex(tt["if_then"] or "") + " \\textit{Watch:} " + tex(tt["watch"] or "")


def bold_ids(ids):
    return " ".join("\\textbf{" + i + "}" for i in ids)


def quote_item(q):
    note = " {\\footnotesize(translated from " + tex(q["language"]) + ")}" if q["translated"] else ""
    return "    \\item \\textit{``" + tex(q["shown"]) + "''}" + note


def build_tex(title, context, n_reviews, base, label, drafts, synth, watch, lang):
    colors = "\n".join(f"\\definecolor{{g{PREFIX[g]}}}{{HTML}}{{{COLOR[g]}}}\\definecolor{{l{PREFIX[g]}}}{{HTML}}{{{LIGHT[g]}}}" for g in GROUPS)
    rows = "\n".join(rf"  \colorbox{{l{d['id'][0]}}}{{\strut\textcolor{{g{d['id'][0]}}}{{\textbf{{{d['id']}}}}}}} & {tex(d['subtheme'])} & {tex(d['requirement'])} & {d['ev']['reviews']} \\"
                     for d in drafts)
    tensions = "\n".join(tension_item(tt) for tt in synth["tensions"]) or r"  \item None found among the candidates."
    cards = []
    for d in drafts:
        g, ev = d["group"], d["ev"]
        quotes = "\n".join(quote_item(q) for q in d["quotes"])
        cards.append(rf"""\begin{{tcolorbox}}[colback=white,colframe=g{d['id'][0]},colbacktitle=l{d['id'][0]},coltitle=black,title={{\textbf{{{d['id']}}}\quad {tex(d['subtheme'])}\hfill\textcolor{{g{d['id'][0]}}}{{\textbf{{{g}}}}}}},boxrule=0.8pt,arc=2pt,left=6pt,right=6pt,top=4pt,bottom=4pt,before skip=8pt,after skip=8pt]
{{\large\textbf{{{tex(d['requirement'])}}}}}

\smallskip
\textbf{{Why.}} {tex(d['why'])}

\smallskip
{{\small {evidence(ev, n_reviews, base, label)}}}

\smallskip
\textbf{{Players say.}}
\begin{{itemize}}[leftmargin=12pt,itemsep=1pt,topsep=2pt]
{quotes}
\end{{itemize}}
\textbf{{Possible change.}} {tex(d['proposed_change'])}

\smallskip
\textbf{{Check after the change.}} {tex(d['check'])}

\medskip
{{\small\textbf{{Decision:}}\quad $\square$ Accept\quad $\square$ Modify\quad $\square$ Reject\qquad Notes: \rule{{0.38\linewidth}}{{0.4pt}}}}
\end{{tcolorbox}}""")
    watch_rows = "\n".join(rf"  {tex(w['subtheme'])} & {w['reviews']} & {w['negative_%']:.0f}\% & {w['or']:.2f} \\" for w in watch) or r"  \multicolumn{4}{l}{None} \\"
    return rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=1.8cm]{{geometry}}
\usepackage{{iftex}}
\ifPDFTeX\usepackage[T1]{{fontenc}}\usepackage{{lmodern}}\else\usepackage{{fontspec}}\fi
\renewcommand{{\familydefault}}{{\sfdefault}}
\usepackage[table]{{xcolor}}
\usepackage{{booktabs,tabularx,amssymb,enumitem}}
\usepackage[most]{{tcolorbox}}
\usepackage[hidelinks]{{hyperref}}
\setlength{{\parindent}}{{0pt}}
\definecolor{{ink2}}{{HTML}}{{52514E}}
\definecolor{{head}}{{HTML}}{{DCE4F2}}
{colors}
\begin{{document}}
{{\LARGE\textbf{{{tex(title)}}}}}\par\smallskip
{{\textcolor{{ink2}}{{{tex(context)}. Generated by the review pipeline on {n_reviews:,} reviews, {date.today().isoformat()}. Draft for review by the design team.}}}}

\medskip
\begin{{tcolorbox}}[colback=head!45,colframe=head,boxrule=0pt,arc=2pt]
\textbf{{In short.}} {refs(tex(synth['summary']))}
\end{{tcolorbox}}

\section*{{Priorities}}
{{\footnotesize\textcolor{{ink2}}{{\colorbox{{lR}}{{\textcolor{{gR}}{{\textbf{{Repair first}}}}}} {MEANING['Repair first']}.\quad \colorbox{{lA}}{{\textcolor{{gA}}{{\textbf{{Accepted friction}}}}}} {MEANING['Accepted friction']}: keep it.\quad \colorbox{{lP}}{{\textcolor{{gP}}{{\textbf{{Protect}}}}}} {MEANING['Protect']}: preserve it. Outcome: {tex(label)}.}}}}

\smallskip
{{\small
\renewcommand{{\arraystretch}}{{1.25}}
\begin{{tabularx}}{{\linewidth}}{{l p{{0.24\linewidth}} X r}}
  \toprule
  \rowcolor{{head}}\textbf{{ID}} & \textbf{{Aspect}} & \textbf{{Requirement}} & \textbf{{Reviews}} \\
  \midrule
{rows}
  \bottomrule
\end{{tabularx}}}}

\section*{{Tensions to resolve}}
{{\small\textcolor{{ink2}}{{A change that repairs one aspect may weaken another that players value. Each candidate pair in the same area of the game was checked; only direct conflicts are listed.}}}}
\begin{{itemize}}[leftmargin=12pt]
{tensions}
\end{{itemize}}

\newpage
\section*{{Requirements}}
{chr(10).join(cards)}

\section*{{Often criticized, but not decisive}}
{{\small These aspects have more negative than positive extracts but no significant link with the outcome. The reviews give no reason to put them first.\par\smallskip
\begin{{tabular}}{{l r r r}}
  \toprule
  \rowcolor{{head}}\textbf{{Aspect}} & \textbf{{Reviews}} & \textbf{{Negative}} & \textbf{{Odds ratio}} \\
  \midrule
{watch_rows}
  \bottomrule
\end{{tabular}}}}

\section*{{About this brief}}
{{\small Numbers and quotes are computed from the coded reviews; quotes are verbatim, and translations into {tex(lang)} are marked. Requirements, reasons, changes, the summary and the tensions are drafts written by a language model and are hypotheses: the associations are not causal and no change has been tested. Groups: a sub-theme is a priority when its odds ratio differs from 1 after Benjamini--Hochberg correction; its tone is the balance of negative and positive extracts. The source of every number and quote is in design\_brief.json.}}
\end{{document}}
"""


def compile_pdf(path):
    folder, name = os.path.split(os.path.abspath(path))
    for cmd in (["tectonic", name], ["xelatex", "-interaction=nonstopmode", name], ["pdflatex", "-interaction=nonstopmode", name]):
        if shutil.which(cmd[0]):
            for _ in range(1 if cmd[0] == "tectonic" else 2):
                res = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)
            if res.returncode == 0:
                return cmd[0]
            print(f"⚠️ {cmd[0]} failed:\n{(res.stdout + res.stderr)[-1500:]}")
            return None
    print("⚠️ No LaTeX engine found (tectonic, xelatex or pdflatex): compile design_brief.tex yourself.")
    return None


# ---------------------------------------------------------------- main
async def main():
    args = K.parse_args("Phase 5: design brief")
    cfg = K.load_config(args.config)
    opt = cfg.get("design_brief", {})
    model = opt.get("model", cfg["models"]["harmonizer"])
    lang = opt.get("language", "English")
    outcome = opt.get("outcome", "voted_up")
    label = opt.get("outcome_label", "recommending the game")
    alpha = opt.get("alpha", 0.05)
    n_max = opt.get("max_extracts", 40)
    final = cfg.path("final")
    reassigned = final.replace(".csv", "_reassigned.csv")
    src = os.path.join(os.path.dirname(final), opt["input"]) if "input" in opt else (reassigned if os.path.exists(reassigned) else final)
    out_dir = os.path.dirname(final)
    print(f"🚀 PHASE 5: Design brief | {os.path.basename(src)} | model {model} | {lang}")

    ex = pd.read_csv(src, dtype={"item_id": str})
    t, n_reviews, base = triage(cfg, ex, outcome, alpha)
    print("   Groups:", t["group"].value_counts().to_dict())
    pri = t[t["group"] != "Not decisive"].copy()
    pri["order"] = pri["group"].map(GROUPS.index)
    pri = pri.sort_values(["order", "reviews"], ascending=[True, False])
    if args.sample:
        pri = pri.head(args.sample)

    drafts, counters = [], {g: 0 for g in GROUPS}
    async with httpx.AsyncClient() as client:
        for r in pri.to_dict("records"):
            counters[r["group"]] += 1
            rid = f"{PREFIX[r['group']]}{counters[r['group']]}"
            g = ex[(ex["theme"] == r["theme"]) & (ex["subtheme_id"] == r["subtheme_id"])]
            x = pick(g, r["group"], n_max)
            t0 = time.time()
            res = await K.call_llm(cfg, client, model, [{"role": "user", "content": draft_prompt(cfg, r, x, n_reviews, base, lang, label)}], DRAFT_SCHEMA)
            raw = res["raw"]
            quotes = []
            for q in raw.get("quotes", []):
                i = q.get("extract") if isinstance(q, dict) else None
                if isinstance(i, int) and 1 <= i <= len(x) and i not in [qq["extract"] for qq in quotes]:
                    e = x.iloc[i - 1]
                    same = str(e["language"]).lower().startswith(lang.lower()[:4])
                    quotes.append({"extract": i, "item_id": e["item_id"], "language": e["language"], "original": str(e["snippet"]),
                                   "translated": not same, "shown": str(e["snippet"]) if same else str(q.get("translation", ""))})
            if not quotes and len(x):   # no valid choice: the first extract, shown in its language
                e = x.iloc[0]
                quotes = [{"extract": 1, "item_id": e["item_id"], "language": e["language"], "original": str(e["snippet"]), "translated": False, "shown": str(e["snippet"])}]
            d = {"id": rid, "group": r["group"], "theme": r["theme"], "subtheme": r["subtheme"], "error": res["error"], "seconds": round(time.time() - t0, 1),
                 "requirement": raw.get("requirement", ""), "why": raw.get("why", ""), "proposed_change": raw.get("proposed_change", ""), "check": raw.get("check", ""),
                 "quotes": quotes[:2], "form_ok": str(raw.get("requirement", "")).startswith(START[r["group"]]),
                 "ev": {k: r[k] for k in ("reviews", "extracts", "negative_%", "positive_%", "or", "or_low", "or_high", "p_bh", "critics", "critics_outcome_%")}}
            d["too_long"] = [k for k, n in (("requirement", 25), ("why", 25), ("proposed_change", 30), ("check", 25)) if len(d[k].split()) > n + 5]
            drafts.append(d)
            print(f"   {rid} {r['subtheme']}: {d['seconds']} s{'' if d['form_ok'] else ' ⚠️ form'}{' ⚠️ long ' + str(d['too_long']) if d['too_long'] else ''}", flush=True)

        t0 = time.time()
        res = await K.call_llm(cfg, client, model, [{"role": "user", "content": summary_prompt(cfg, drafts, lang)}], SUMMARY_SCHEMA)
        summary = re.sub(r"\[(?![RAPC]\d+\])[^\]]*\]", "", res["raw"].get("summary", ""))
        by_id = {d["id"]: d for d in drafts}
        pairs, checked = candidate_pairs(drafts), []
        for a_, b_ in pairs:
            r2 = await K.call_llm(cfg, client, model, [{"role": "user", "content": pair_prompt(cfg, by_id[a_], by_id[b_], lang)}], PAIR_SCHEMA)
            checked.append({"ids": [a_, b_], "names": [by_id[a_]["subtheme"], by_id[b_]["subtheme"]], "groups": [by_id[a_]["group"], by_id[b_]["group"]],
                            "error": r2["error"], **{k: r2["raw"].get(k) for k in ("change_element", "affected_element", "conflict", "if_then", "watch")}})
        tensions = [c for c in checked if c["conflict"] is True]
        synth = {"summary": summary, "candidates": checked, "tensions": tensions, "error": res["error"], "seconds": round(time.time() - t0, 1)}
        print(f"   Summary and tensions: {synth['seconds']} s, {len(tensions)} of {len(pairs)} candidate tensions kept")

    watch = t[(t["group"] == "Not decisive") & (t["negative_%"] > t["positive_%"])].sort_values("reviews", ascending=False).head(6).to_dict("records")
    title = opt.get("title", "Design brief")
    K.save_json(os.path.join(out_dir, "design_brief.json"), {"source": os.path.basename(src), "model": model, "language": lang, "outcome": outcome, "reviews": n_reviews,
                                                              "outcome_%": base, "triage": t.to_dict("records"), "drafts": drafts, "synthesis": synth})
    path = os.path.join(out_dir, "design_brief.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_tex(title, cfg["study"]["context"], n_reviews, base, label, drafts, synth, watch, lang))
    engine = compile_pdf(path)
    total = sum(d["seconds"] for d in drafts) + synth["seconds"]
    print(f"💾 {path}{' and design_brief.pdf (' + engine + ')' if engine else ''} | {len(drafts)} requirements, model time {total:.0f} s")


if __name__ == "__main__":
    asyncio.run(main())

> Reference of the configuration and of phases 0–2, options and other data sources. For the procedure, phases 3–5 and the design brief, see the main README.

# LLM-assisted thematic analysis pipeline

A three-phase pipeline for **multilingual thematic analysis** of short user-generated texts (e.g. Steam reviews)
with **local LLMs via [Ollama](https://ollama.com)**. Two independent LLM coders extract verbatim extracts under
researcher-defined themes; a third model adjudicates and assigns each extract to a sub-theme, using either a
codebook **induced from the data** or one **supplied by the researcher**.

Everything study-specific lives in one file, `study_config.json`. The scripts contain no study content.
The configuration in `examples/elex/` is a complete worked example, an analysis of Steam reviews of *ELEX*.

```
raw texts ──► 0_filtering ──► 1_encoding (encoder 1 + encoder 2) ──► 2_harmonization ──► coded extracts + summary
                 │                  │                                     │
          cleaning rules     verbatim snippets per theme          sub-theme codebook
          + LLM content      + initial code + sentiment           (inductive | deductive)
          judgement          + verification pass                  + adjudication
```

Optional steps around the three phases (see "Optional steps"), independent of the source of the texts:

```
steam_collect.py / prepare_reddit.py ──► 0_filtering ──► stratified_sample.py ──► 1_encoding ──► 2_harmonization ──► validation.py
  (collection or conversion)                            (sample + weights)                                   (blind human coding, agreement)
```

## Requirements

- Python ≥ 3.9 with `httpx`, `numpy`, `pandas`, `tqdm`; `openpyxl` for `validation.py` (Excel coding sheet)
- Ollama running locally, with the models named in `study_config.json` pulled (`ollama pull <model>`),
  including the embedding model used for inductive codebooks (default `nomic-embed-text`)

## Quick start

```bash
python 0_filtering.py            # → output/filtered_items.json, filter_report.csv
python 1_encoding.py             # → output/encoded_results.json
python 2_harmonization.py        # → output/thematic_analysis_final.csv, thematic_analysis_summary.csv
```

For Reddit data, first convert the threads with `prepare_reddit.py` (see "Using other data sources").

With the optional steps (collection of several Steam games, a stratified sample, human validation):

```bash
python steam_collect.py --apps 411300=ELEX --out-dir raw    # → raw/all_reviews.json
python 0_filtering.py
python stratified_sample.py                                 # → sample/, study_config_sample.json
python 1_encoding.py --config study_config_sample.json
python 2_harmonization.py --config study_config_sample.json
python validation.py sample --config study_config_sample.json      # → sample/validation/coding_sheet.xlsx
python validation.py agreement --config study_config_sample.json   # after the human coding
```

Each script accepts `--config <file>` (default: `study_config.json` next to the scripts) and
`--sample N` (process only the first N items, for quick tests). All paths in the config are relative to the config file.

Every phase is **resumable and incremental**:
- the filter caches the model's decisions;
- the encoder reuses previous results, matched by item id, so only new items are encoded;
- the harmonizer checkpoints its progress.

Items that failed because of a model error are retried on the next run.

## The configuration file

| Section | Content |
|---|---|
| `study.context` | One phrase describing the corpus, used in every prompt (e.g. "Steam reviews of ELEX, a single-player open-world action role-playing game") |
| `paths` | Raw input file, output folder and output file names |
| `fields` | Field names in the input (JSON, JSONL or CSV): `id`, `text`, `language` (optional), `context` (optional: shown to the models, never coded), `metadata` (columns copied to the output, nested fields as `a.b`) |
| `models` | Ollama models for the filter, the two encoders and the harmonizer |
| `ollama` | API url, context window, output budget, timeout, retries, seed |
| `filter` | Cleaning thresholds and switches (`remove_duplicates`, `strip_bbcode`, `strip_markdown`) and the criteria for "informative" / "not informative" texts (`use_llm: false` = cleaning rules only) |
| `themes` | The deductive themes: `name`, `definition`, `includes`, `excludes` |
| `global_rules` | Codebook-wide boundary rules (e.g. "Generic praise matches no theme") |
| `sentiment` | Sentiment scale: `label`, `description`, `polarity` (Negative / Neutral / Positive) |
| `few_shot_examples` | Worked examples for the encoders: `text` and the expected `output` per theme (omitted themes = empty) |
| `encoding` | Verification pass on/off, chunk size for long texts, minimum snippet length, maximum code length |
| `harmonization` | Codebook mode, sub-theme cap, inductive-codebook procedure (`codebook_induction`: embedding model, clusters, evaluation models, sample, revision rounds, quality thresholds), matching threshold, parallel requests, context window |
| `sampling` (optional) | Stratified sample for `stratified_sample.py`: group field, items per group, oversampled value and its share, seed, context template, variables for the representativeness check, output folder |
| `validation` (optional) | Human validation for `validation.py`: items per group and oversampled items, seed, translation model and languages, output folder, bootstrap resamples |

To apply the pipeline to another corpus, write a new configuration. No code changes are needed.

## Phases

**0 — Filtering.** First, fixed rules apply identically to every language:
- minimum length, with CJK/Thai characters weighted;
- minimum share of letters, which removes ASCII art;
- exact duplicates removed.

Then an LLM keeps the texts that contain at least one *specific* statement. If the model errors, the text is
**kept** and flagged (fail-open). `filter_report.csv` reports kept and excluded texts per language and reason.

**1 — Encoding.** Each encoder returns, per theme, `{snippet, code, sentiment}` under a JSON schema.
- **Formal validation:** snippets must appear verbatim in the text and meet the minimum length. Codes must be short and must not copy the snippet.
- **Verification pass:** the same model re-classifies each snippet in isolation, and a (theme, snippet) pair survives only if the verifier agrees. Rejected pairs are kept in `enc*_rejected`.
- **Memory:** one model is loaded at a time.

**2 — Harmonization.**
- **A. Codebook.** Two modes, set in `harmonization.codebook_mode`:
  - `inductive`: the codebook is induced from the data and tested before use. For each theme:
    1. **Clustering.** All initial codes from both encoders are embedded and grouped by meaning with weighted spherical k-means (deterministic, fixed seed).
    2. **Naming.** The LLM defines at most `max_subthemes_per_theme` sub-themes from the clusters, with explicit boundaries between neighbouring sub-themes.
    3. **Testing.** Two *different* models independently assign a frequency-weighted sample of codes to the sub-themes, with "NONE" allowed. The procedure measures **balance** (no sub-theme above `max_share` or below `min_share`), **coverage** (codes fitting no sub-theme below `max_unassigned`) and **reliability** (Cohen's κ between the two coders above `min_kappa`).
    4. **Merging inseparable sub-themes.** For every pair of sub-themes the procedure measures how often the two coders split codes between them (`merge_confusion`) and how similar their definitions are (embedding cosine, `merge_similarity`). Pairs above either threshold are merged into one sub-theme by the LLM. A merge is **kept only if the next test improves** the codebook (fewer failed criteria, or equal criteria and higher κ); otherwise it is reverted and never proposed again. Merged sub-themes record their origin (`merged_from`) and are exempt from the "too broad" rule, so that semantically inseparable components (e.g. calm and relaxation) form one sub-theme instead of being split again.
    5. **Revision.** Remaining problems are passed to the LLM as a diagnosis: it splits sub-themes that are too broad, merges marginal ones, sharpens boundaries that confuse the coders, and adds missing patterns. Only the sub-themes involved in the problems may change. Steps 3–5 repeat up to `max_revision_rounds` times.

    The best version is kept. Every round's sub-themes and diagnostics (κ, shares, unassigned share, pairwise confusion and similarity, merges accepted or rejected, problems) are saved in `subtheme_codebook.json` for the methods section. The codebook is reused while the encoded data are unchanged. The researcher can review or edit it before harmonization; delete the file to regenerate it.
  - `deductive`: copy `codebook_template.json` to the path in `harmonization.deductive_codebook`, write your sub-themes (name + definition; ids optional), and run. Every theme needs at least one sub-theme.
- **B. Adjudication.** Snippets of the two encoders quoting the same passage are matched deterministically. For each candidate, the adjudicator:
  - decides KEEP or REJECT;
  - chooses the final snippet **by reference**, so it stays verbatim;
  - writes one code and assigns the sentiment;
  - assigns a sub-theme of the candidate's theme. There is no "Other" category: the closest sub-theme is always used, with a self-declared `subtheme_fit` (strong / moderate / weak). Invalid assignments are re-asked with a restricted choice.
- **C. Summary.** One row per sub-theme: extracts, items, polarity, fit distribution, top codes and examples.

## Outputs

| File | Content |
|---|---|
| `thematic_analysis_final.csv` | One coded extract per row: theme, sub-theme, fit, code, sentiment, polarity, snippet, the two encoders' proposals, adjudicator justification, flags, metadata |
| `thematic_analysis_summary.csv` | One row per sub-theme, ready for reporting |
| `subtheme_codebook.json` | The codebook actually applied (inductive or a copy of the deductive one) |
| `harmonization_rejected.csv` | Candidates rejected by the adjudicator, with justification |
| `filter_report.csv`, `filter_decisions.json` | Filtering audit trail |

The `flags` column records automatic corrections:
- `snippet_fallback`, `code_fallback`, `sentiment_fallback`: an invalid value from the model was replaced;
- `subtheme_reassigned`: the sub-theme was re-asked with a restricted choice;
- `not_audited`: the adjudicator returned no decision for this candidate.

## Optional steps

### Collecting Steam reviews (one or more games)

```bash
python steam_collect.py --apps 411300=ELEX 900040=ELEX_II --out-dir raw
python steam_collect.py --games games.json --out-dir raw      # {"games": [{"key", "name", "app_id"}, ...]}
```

Every public review in every language (`--language english` to restrict) is downloaded from the Steam review
endpoint. Author identifiers (Steam ID, username, profile URL, avatar) are removed, and each review gets `game` and
`app_id`, so several games can be analysed together. The script survives rate limiting (long back-off, progress saved,
resume on the next run) and writes one file per game, the merged `raw/all_reviews.json` (set `paths.raw_input` to it)
and `collection_log.json` (date, reviews collected, total reported by Steam). For several games, add
`"game": "game", "app_id": "app_id"` to `fields.metadata`.

### Stratified sample (any source)

When the corpus is too large to encode in full, or its groups are very unequal (one game with 7,000 reviews, another
with 600; a subreddit dominating the others), encode a stratified random sample instead:

```bash
python stratified_sample.py                    # settings in the section "sampling"
```

| `sampling` key | Meaning |
|---|---|
| `group_field` | Field whose values are sampled separately: the game, the survey question, the subreddit or thread, the participant (null = one group) |
| `n_per_group` | Items per group; a smaller group is taken in full (null = every item) |
| `oversample` | `{"field", "value", "share"}`: items whose field equals the value (or one of a list of values) make up `share` of each group's sample, e.g. negative Steam reviews (`voted_up`, `false`), dissatisfied respondents (a Likert item, `[1, 2]`). null = simple random sample per group |
| `seed` | Random seed (items are ordered by id first, so the sample does not depend on the input order) |
| `context_template` | Optional context shown to the models, e.g. `"Steam review of the game {group}"`. Placeholders: `{group}`, `{context}` (the original context: survey question, Reddit thread), the names in `fields.metadata` |
| `balance_fields` | Numeric fields compared between sample and population: `{"name", "field", "transform"}`, transform `log1p`, `year` (unix time) or null |
| `output_dir` | Folder of the sample and of every later output |

Each sampled item gets the weight N_stratum / n_stratum, which restores each group's mix in the population.
`sample_representativeness.csv` compares the weighted sample with the filtered population (text length, language,
the oversampled value, the balance fields): |SMD| < 0.1 is a negligible difference. The script writes
`<config>_sample.json`, a copy of the configuration whose paths point to the sample: run Phases 1–2 and the validation
with it. Edit only the original configuration and run the script again after any change: the sample does not change.

### Human validation (any source)

The models' self-declared fit is not calibrated, so validate the final coding against a human coder on a sample:

```bash
python validation.py sample [--subset-per-group 10]     # blind coding sheet
python validation.py agreement                          # after the coding
```

1. `sample` draws items per group (half from the oversampled stratum, or `oversampled_per_group`) among the coded
   items, with and without extracts, and writes `validation/coding_sheet.xlsx` without the LLM codes. The coder fills,
   for each theme, presence (1/0), polarity (positive / negative / mixed / neutral) and, optionally, the sub-theme
   ids. The sheet shows what the LLM coders saw (group, context, text) and hides the oversampled field (e.g. the
   recommendation or the Likert score). Texts not in `target_language` get a machine translation by a model not used
   in the coding (`translator_model: null` = none). `--subset-per-group K` codes a nested subset, so the validation
   can be extended later without changing the coded items.
2. `agreement` compares the human coding with the harmonized coding and with each encoder alone, their union and
   intersection: Cohen's κ and Gwet's AC1 with bootstrap CIs, prevalence, sensitivity and precision, for presence,
   positive and negative mentions; κ per sub-theme and micro-F1. It works on a partly coded sheet (fully coded rows
   only, at least 20) and writes `disagreements.csv` and the LaTeX table `tab_validation.tex`.

## Using other data sources

The pipeline is not tied to Steam reviews. Any collection of short texts can be analysed: forum posts, Reddit comments,
app-store reviews, open-ended survey answers, interview excerpts. Three things change from one source to another: the
**input format**, a few **filter switches**, and, for conversational data, the **context** of each text.

### Input formats

`paths.raw_input` can be:

| Format | Expected content |
|---|---|
| `.json` | A list of objects, one per text (nested fields allowed: `author.name`) |
| `.jsonl` / `.ndjson` | One JSON object per line (typical of Reddit and other API dumps) |
| `.csv` | One row per text, with a header row. All values are read as text, so ids keep leading zeros (`007`) |

Excel files (`.xlsx`) are not read directly. Save the sheet as CSV, or convert it in one line:

```python
import pandas as pd
pd.read_excel("survey.xlsx").to_csv("survey.csv", index=False)
```

Each item needs a **unique id** and a **text**. Language, context and metadata are optional.

### The context field

Some texts cannot be understood alone: a Reddit reply ("Same here, it lags all the time") needs the comment it answers,
and a survey answer ("Too expensive") needs the question. Set `fields.context` to the field holding that information.
In every phase the context is shown to the models in a `<context>` block, with an explicit rule to use it **only to
understand** the text. Extracts are always taken from the text: a snippet copied from the context fails the verbatim
check and is discarded. The context is also written to the output (`context` column).

The pipeline does not reconstruct threads. If the context must come from another record (e.g. the parent comment),
add it as a field of each item before running the pipeline.

When `fields.context` is not set, prompts and outputs are exactly those of a context-free analysis.

### Filter switches

| Switch | Default | When to change it |
|---|---|---|
| `remove_duplicates` | `true` | Set `false` for **surveys**: identical answers from different respondents ("Too expensive") are valid data, not copypasta |
| `strip_bbcode` | `true` | Removes Steam-style tags (`[h1]…[/h1]`). Set `false` for sources where square brackets are content (e.g. "[sic]", "[deleted]") |
| `strip_markdown` | `false` | Set `true` for **Reddit** and other markdown sources: removes quoted lines (`> …`, text written by someone else), keeps link labels, removes URLs and bold/italic markup. The cleaned text is used in every phase, so the verbatim check refers to it |
| `min_effective_chars` | `20` | Lower it (e.g. `3`–`5`) for surveys with very short answers |

The snippet thresholds in `encoding` (`min_snippet_words`, `min_snippet_chars`) may also need lowering for short
answers: with the default of 3 words, "Too slow" would be discarded.

### Example: Reddit discussions (keeping the thread structure)

On Reddit the meaning of a comment often depends on the discussion: "Same here" confirms a problem raised above, and
threads show which topics trigger debate, agreement or disagreement. `prepare_reddit.py` turns Reddit data into
pipeline items **without losing the structure**:

- every comment becomes one item: its **text** is the comment, the only text that is coded;
- its **context** is the thread above it: the post (title and text) and the chain of comments from the top level down to the **direct parent**, which is always kept in full. When the chain is too long, the oldest intermediate comments are replaced by `[… n earlier comment(s) omitted …]`;
- structural fields are kept as metadata: `thread_id`, `parent_id`, `depth` (1 = reply to the post), `score`, `n_replies` (direct replies), `kind`, `created_utc`, `subreddit`;
- deleted or removed comments are not coded, but appear as `[deleted]` in the context of their replies, so the structure stays intact;
- usernames are replaced by a **salted pseudonym**, so distinct users can be counted but the original usernames cannot be recovered. Use `--author-salt SECRET` for pseudonyms that stay the same across runs, or `--keep-authors` to store usernames (check your ethics approval first).

**1. Collect the data.** The script reads, in any combination:
- **Thread JSON as served by Reddit:** the address of a thread followed by `.json`, e.g. `https://www.reddit.com/r/<subreddit>/comments/<id>/<slug>.json`. It contains the post and the nested replies. Put one file per thread in a folder.
- **Flat dumps** (JSON list or JSONL) of comments with `id`, `parent_id`, `link_id`, `body`, as produced by the Reddit API or by archives. Add the posts with `--submissions posts.jsonl` to get their titles in the context.

Collect data in accordance with Reddit's terms and your ethics approval. Long threads served by Reddit are incomplete:
the script reports how many comments Reddit did not load ("load more").

**2. Prepare the items.**

```bash
python prepare_reddit.py --input reddit_threads/ --output data/reddit_items.jsonl
```

Options:
- `--include-posts`: also code the posts themselves (depth 0).
- `--max-context-chars` (default 3000), `--max-ancestor-chars` (400), `--max-post-chars` (800): context length.

The script prints the number of threads, comments, deleted comments, incomplete chains (replies whose parent is
missing from the data: depth unknown) and items per depth.

**3. Configure the analysis.**

```json
"study":  {"context": "Reddit comments about the game (each shown with the discussion above it)"},
"paths":  {"raw_input": "data/reddit_items.jsonl", "output_dir": "output_reddit", "files": { "...": "..." }},
"fields": {
  "id": "id",
  "text": "text",
  "context": "context",
  "metadata": {"thread_id": "thread_id", "parent_id": "parent_id", "depth": "depth", "score": "score",
               "n_replies": "n_replies", "kind": "kind", "author": "author", "created_utc": "created_utc"}
},
"filter": {"strip_markdown": true, "strip_bbcode": false, "remove_duplicates": true, "...": "..."}
```

`strip_markdown` removes the lines quoted from other users (`> …`), so a reply is never coded for words it quotes.
Write few-shot examples in the style of the platform: replies, slang, quotes.

**4. Analyse the structure.** Every coded extract carries its comment's position in the discussion, so the output can
be analysed along the threads, for example:

```python
import pandas as pd

df = pd.read_csv("output_reddit/thematic_analysis_final.csv", dtype={"item_id": str, "parent_id": str})
items = pd.read_json("data/reddit_items.jsonl", lines=True, dtype={"id": str})     # all comments, coded or not

# 1. Engagement: which sub-themes are raised in comments that receive more replies and votes?
per_comment = df.drop_duplicates(["item_id", "subtheme"])
print(per_comment.groupby(["theme", "subtheme"])
      .agg(comments=("item_id", "nunique"), mean_replies=("n_replies", "mean"), mean_score=("score", "mean"))
      .sort_values("mean_replies", ascending=False))

# 2. Polarity along the discussion: is criticism concentrated at the top level or in the replies?
print(pd.crosstab(df["depth"], df["polarity"], normalize="index").round(2))

# 3. Theme propagation: which sub-themes do replies raise when answering a comment about sub-theme X?
parent = df[["item_id", "subtheme"]].rename(columns={"item_id": "parent_id", "subtheme": "parent_subtheme"})
transitions = df.merge(parent, on="parent_id")
print(pd.crosstab(transitions["parent_subtheme"], transitions["subtheme"]))

# 4. Coverage: share of comments with at least one coded extract, by depth
coded = items["id"].isin(df["item_id"])
print(coded.groupby(items["depth"]).mean().round(2))
```

**Sampling and validation.** To balance subreddits or threads, sample with `"group_field": "subreddit"` (or
`"thread_id"`); `"context_template": "Comment in r/{subreddit}\n{context}"` keeps the thread in the context (add
`subreddit` to `fields.metadata`). Check the balance of the discussion structure with `balance_fields` such as
`{"name": "depth", "field": "depth"}` and `{"name": "log_replies", "field": "n_replies", "transform": "log1p"}`.
In the validation sheet the coder sees the thread in the column `context`.

Whether a reply **agrees with, contests or extends** its parent is not coded yet. It would be a per-extract attribute
(see "Not yet supported" below).

### Example: open-ended survey answers

Analyse **one open question per run**: different questions usually need different themes. Put the question in the
context (as a CSV column, or in `study.context`), and use one configuration and one output folder per question.

```json
"study":  {"context": "answers of app users to the open question 'What would you improve in the app?'"},
"paths":  {"raw_input": "data/survey_q3.csv", "output_dir": "output_q3", "files": { "...": "..." }},
"fields": {
  "id": "respondent_id",
  "text": "q3_answer",
  "context": "q3_text",
  "metadata": {"age": "age", "group": "condition"}
},
"filter":   {"remove_duplicates": false, "min_effective_chars": 3, "use_llm": true, "...": "..."},
"encoding": {"min_snippet_words": 1, "min_snippet_chars": 3, "...": "..."}
```

Notes:
- If the CSV has one column per question, reshape it to one row per answer (`pandas.melt`), or run one configuration per column.
- **Small samples.** With a few hundred answers the codes are few: the clusters are small and the κ of the codebook test is less stable. Lower `max_subthemes_per_theme` (e.g. 3–4), lower `evaluation_sample` to the number of codes, or use a deductive codebook if a theoretical framework exists.
- A filter is often unnecessary for surveys. Set `use_llm: false` to keep only the cleaning rules, so that every non-empty answer is analysed.

**Likert items and other closed answers.** Closed answers are not coded: they are the variables that the open
answers are related to. Copy them to the output through `fields.metadata` (e.g. `"satisfaction": "q1_likert"`), then
analyse them with the coded extracts (e.g. sub-themes of dissatisfied vs satisfied respondents). They can also drive
the sample and the validation:

```json
"sampling": {
  "group_field": null,
  "n_per_group": 400,
  "oversample": {"field": "q1_likert", "value": [1, 2], "share": 0.4},
  "balance_fields": [{"name": "age", "field": "age", "transform": null}]
}
```

Numeric values are compared as numbers (`2`, `"2"` and `"2.0"` read from CSV all match), other values as text
without case (`false` matches `"False"`). The Likert answer is hidden from the human coder in the validation sheet, as it was hidden from the
models. Do not put it in the context: the coding of the text would then depend on the closed answer.

If several open questions share the same themes, they can be analysed in one run with `"group_field": "question_id"`
and `"context_template": "{context}"` (the question text as context): each question is sampled separately.

### Example: interviews and focus groups

Split each transcript into **one item per answer** (speaker turn or meaningful segment) before running the pipeline:
the interviewee's answer is the text, the interviewer's question (and, if needed, the previous turn) the context.

```json
"study":  {"context": "semi-structured interviews with players about a game (one answer per item)"},
"paths":  {"raw_input": "data/interview_segments.csv", "output_dir": "output_interviews", "files": { "...": "..." }},
"fields": {
  "id": "segment_id",
  "text": "answer",
  "context": "question",
  "metadata": {"participant": "participant", "turn": "turn", "interview_section": "section"}
},
"filter":   {"use_llm": false, "remove_duplicates": false, "min_effective_chars": 10, "...": "..."},
"sampling": {"group_field": "participant", "n_per_group": null}
```

Notes:
- Pseudonymise participants in the CSV (P01, P02...) before analysis; never put names in the text or the context.
- Keep every segment (`use_llm: false`, `n_per_group: null`): interviews are small corpora and short answers ("yes,
  exactly that") get their meaning from the question in the context.
- Long answers are split into chunks by the encoder (`encoding.max_chunk_chars`); prefer segments of a few sentences.
- For the validation, `"group_field": "participant"` gives the same number of segments per participant, so no
  participant dominates the agreement. With few participants, raise `validation.per_group`.
- The few-shot examples should be spoken language: hesitations, repetitions, answers that refer to the question.

## Adapting the pipeline: an exploratory design-feedback study

The included configuration answers a research question with predefined, theory-driven themes. The same pipeline can be
used for an **exploratory, design-oriented analysis**, for example "find all the feedback on mechanics, UI and
interactions that could help the game's designers, especially the negative feedback". The scripts stay the same: copy
`study_config.json` (e.g. to `design_config.json`) and change the sections below.

### 1. Define broad, design-oriented themes

Themes become the *areas* you want to explore. Keep them broad: the specific problems are discovered by the inductive
codebook. State in `includes` that problems, frustrations and requests are wanted.

```json
"themes": [
  {
    "name": "MECHANICS",
    "definition": "How the game's rules, systems and loops work for the player.",
    "includes": "core loop, progression, difficulty, balance, grind, pacing, economy, crafting/automation, missing or broken features, feature requests.",
    "excludes": "interface and controls (→ UI); characters and multiplayer (→ INTERACTIONS); bugs without a design dimension."
  },
  {
    "name": "UI",
    "definition": "How the player perceives and operates the game through its interface.",
    "includes": "menus, inventory management, readability, text size, controls/keybindings, tutorials and in-game information, accessibility, feedback signals.",
    "excludes": "rules and balance of the systems behind the interface (→ MECHANICS)."
  },
  {
    "name": "INTERACTIONS",
    "definition": "The player's interactions with characters, other players and the game world.",
    "includes": "NPCs, quests and dialogue, multiplayer and co-op, world interaction (movement, placement, collisions).",
    "excludes": "menus and controls (→ UI); system rules (→ MECHANICS)."
  }
]
```

`global_rules` can state what makes an extract *useful to designers*:

```json
"global_rules": [
  "Extract statements that describe a specific aspect of the design, not general opinions about the game.",
  "Problems, frustrations, workarounds and feature requests are especially relevant: always extract them."
]
```

### 2. Rewrite the few-shot examples

The examples are the strongest signal of what the encoders should extract. Replace them with texts of the new kind.
Include at least one critique, one feature request and one text with nothing relevant (empty output). Themes omitted
from `output` mean an empty list.

```json
{
  "note": "UI critique and a feature request",
  "text": "Love the art, but the inventory fills up constantly and there's no way to sort it. Please add a search bar to the crafting menu.",
  "output": {
    "UI": [
      {"snippet": "the inventory fills up constantly and there's no way to sort it", "code": "unsortable overflowing inventory", "sentiment": "Negative"},
      {"snippet": "Please add a search bar to the crafting menu", "code": "request for crafting menu search", "sentiment": "Neutral"}
    ]
  }
}
```

### 3. Let the sub-themes emerge

Keep `"codebook_mode": "inductive"`. The procedure discovers the specific problems within each area (e.g. "Inventory
management", "Late automation", "Unclear tutorials") without you naming them in advance. For a more detailed,
designer-facing map, raise the cap, e.g. `"max_subthemes_per_theme": 8`.

For a fully open exploration, a single broad theme can be used (e.g. `"DESIGN_FEEDBACK"`: "any statement about how the
game is designed or how it plays"), and the inductive sub-themes then form the whole structure.

### 4. Focus on negative feedback

Two options:

- **Recommended: extract everything, filter afterwards.** Every extract has a `polarity` column, so negative feedback
  is a deterministic, transparent filter. It also lets you compare the share of criticism across sub-themes:

  ```python
  import pandas as pd
  df = pd.read_csv("output/thematic_analysis_final.csv")
  negative = df[df["polarity"] == "Negative"]
  print(negative.groupby(["theme", "subtheme"]).size().sort_values(ascending=False))
  ```

  `thematic_analysis_summary.csv` already reports `negative_%` per sub-theme.

- **Alternative: extract only criticism.** Add a rule such as "Extract ONLY criticisms, problems and requests; ignore
  praise" to `global_rules`. Fewer extracts and faster runs, but the selection becomes a model judgement and positive
  feedback is lost for comparison.

### 5. Adjust the filter (optional)

`filter.informative_criteria` can be narrowed to the design areas of interest. Because the encoders already ignore
irrelevant texts, keeping the filter broad (high recall) is usually the safer choice.

### Not yet supported (possible extensions)

- **Design attributes per extract**, such as the feature concerned, severity, type (bug / design issue /
  feature request) or, for discussions, the stance towards the parent comment (agrees / contests / extends). This would need configurable extra fields in the encoder's JSON schema and in the output.
- **Inducing the themes themselves.** Themes are always defined in the config. Discovering both levels (themes, then
  sub-themes) would require an additional clustering step before encoding.
- **A designer report**: per sub-theme, frequency, polarity, severity, representative quotes and the most frequent
  problems. This would be a fourth script reading `thematic_analysis_final.csv`.

## Methodological notes

- **Reproducibility.** Temperature 0 and a fixed seed are used. Report model names and versions, prompts and the configuration. With `parallel_requests > 1` (which requires `OLLAMA_NUM_PARALLEL` set on the Ollama server), batched inference may introduce small numerical differences between runs.
- **Independence.** Using different model families for the two encoders and the adjudicator reduces correlated errors.
- **Self-declared fit is not calibrated.** LLMs tend to be optimistic. Validate the pipeline against human coding of a sample of extracts (e.g. Cohen's κ on theme, sub-theme and polarity) and report the agreement.
- **Sampling weights.** With a stratified sample, prevalence estimates for the population need the weights in `sample_weights.csv`. Pooled estimates over several groups are dominated by the largest groups: rescale the weights within each group for estimates of the "average group". Models that condition on the oversampled variable (e.g. recommendation as outcome, with group fixed effects) do not need them.
- **Hybrid design.** Themes are deductive (defined in the config). Sub-themes are inductive or deductive. Content outside the configured themes is not analysed, and assignment to sub-themes is exhaustive by design. Both should be stated as limitations.

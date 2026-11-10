"""
Optional: selects the study population from collected reviews, before filtering, with simple rules on the fields of each review.
Example (Steam): keep copies bought on Steam and drop copies declared as received for free:
    python select_population.py raw/all_reviews.json data/population.json --require steam_purchase --exclude received_for_free
Writes the selected reviews and <output>_flow.csv with the number of reviews removed by each rule.
"""
import argparse
import csv
import json

ap = argparse.ArgumentParser(description="Select the study population from collected reviews")
ap.add_argument("input")
ap.add_argument("output")
ap.add_argument("--require", nargs="*", default=[], help="fields that must be true")
ap.add_argument("--exclude", nargs="*", default=[], help="fields that must not be true")
a = ap.parse_args()

items = json.load(open(a.input, encoding="utf-8"))
flow, keep = [("collected", len(items))], items
for f in a.require:
    n = len(keep); keep = [r for r in keep if r.get(f)]; flow.append((f"not {f}", n - len(keep)))
for f in a.exclude:
    n = len(keep); keep = [r for r in keep if not r.get(f)]; flow.append((f, n - len(keep)))
flow.append(("population", len(keep)))
json.dump(keep, open(a.output, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(a.output.rsplit(".", 1)[0] + "_flow.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([("step", "reviews"), *flow])
for step, n in flow:
    print(f"{step:28} {n:>8}")

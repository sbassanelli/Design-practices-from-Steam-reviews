"""
Collects all public Steam reviews of one or more games and merges them into one file (input of 0_filtering.py).

For each game, every review in the requested languages is downloaded from the Steam store review endpoint
(https://store.steampowered.com/ajaxappreviews/<app_id>?json=1), paging with the cursor until no new review is returned.
Author identifiers (Steam ID, username, profile URL, avatar) are removed; each review gets the fields `game` and
`app_id`, so several games can be analysed together (add them to `fields.metadata`, and use `game` as the group of
stratified_sample.py). Reviews returned twice by the endpoint are kept once (same recommendation id).

Robust to rate limiting: long exponential back-off (honouring Retry-After), progress saved every few pages, and
resume from the saved cursor when the script is run again. Games already completed are skipped.

The games are given on the command line (`--apps 411300 "900040=ELEX II"`) or in a JSON file (`--games games.json`):
    {"games": [{"key": "elex", "name": "ELEX", "app_id": "411300"}, ...]}
`key` names the output file (default: the app id), `name` is written in the field `game` (default: the key).

Outputs (--out-dir, default raw/ next to this script):
  reviews_<key>.json         one file per completed game
  all_reviews.json           all completed games merged, in the order given (set paths.raw_input to this file)
  collection_log.json        date, counts, and the total number of reviews Steam reports for each game

Usage: python steam_collect.py --apps APP_ID[=NAME] ... | --games games.json
                               [--out-dir DIR] [--language all] [--force] [--only KEY ...] [--max-per-game N]
"""
import argparse
import datetime as dt
import json
import os
import time

import httpx

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
IDENTIFIERS = {"steamid", "personaname", "persona_status", "profile_url", "avatar"}
PAUSE_SECONDS = 1.5          # between pages: be gentle with the public endpoint
BACKOFF_SECONDS = [30, 60, 120, 240, 480]
SAVE_EVERY_PAGES = 10
MAX_PAGES_WITHOUT_NEW = 5    # the endpoint sometimes repeats a page in high-volume periods: do not stop at the first one


class RequestFailed(Exception):
    pass


def fetch_page(client: httpx.Client, app_id: str, cursor: str, language: str) -> dict:
    params = {"json": 1, "filter": "recent", "language": language, "review_type": "all", "purchase_type": "all",
              "num_per_page": 100, "cursor": cursor}
    for attempt, wait in enumerate(BACKOFF_SECONDS, start=1):
        try:
            r = client.get(f"https://store.steampowered.com/ajaxappreviews/{app_id}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = max(wait, int(retry_after))
            problem = f"HTTP {r.status_code}" + (" (rate limited)" if r.status_code == 429 else "")
        except (httpx.HTTPError, ValueError) as e:
            problem = type(e).__name__
        print(f"\n   {problem}: retry {attempt}/{len(BACKOFF_SECONDS)} in {wait}s", flush=True)
        time.sleep(wait)
    raise RequestFailed(f"app {app_id}: Steam did not answer after {len(BACKOFF_SECONDS)} attempts")


def partial_path(out_dir, game):
    return os.path.join(out_dir, f".partial_{game['key']}.json")


def save_partial(out_dir, game, reviews, cursor, steam_total):
    tmp = partial_path(out_dir, game) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"cursor": cursor, "steam_total": steam_total, "reviews": reviews}, f, ensure_ascii=False)
    os.replace(tmp, partial_path(out_dir, game))


def collect_game(client: httpx.Client, game: dict, out_dir: str, language: str, max_reviews: int = None):
    reviews, cursor, steam_total = [], "*", None
    if os.path.exists(partial_path(out_dir, game)):                       # resume an interrupted collection
        with open(partial_path(out_dir, game), encoding="utf-8") as f:
            state = json.load(f)
        reviews, cursor, steam_total = state["reviews"], state["cursor"], state["steam_total"]
        print(f"   resuming from {len(reviews)} reviews already collected")
    seen_ids = {r["recommendationid"] for r in reviews}
    pages, pages_without_new = 0, 0
    while True:
        data = fetch_page(client, game["app_id"], cursor, language)
        if not data.get("success"):
            break
        if steam_total is None:
            steam_total = data.get("query_summary", {}).get("total_reviews")
        batch = data.get("reviews") or []
        new = [r for r in batch if r["recommendationid"] not in seen_ids]
        for r in new:
            seen_ids.add(r["recommendationid"])
            for k in IDENTIFIERS:
                r.get("author", {}).pop(k, None)
            r["game"], r["app_id"] = game["name"], game["app_id"]
        reviews.extend(new)
        print(f"   {game['name']}: {len(reviews)} / {steam_total} reviews", end="\r", flush=True)
        next_cursor = data.get("cursor")
        pages_without_new = 0 if new else pages_without_new + 1
        if not batch or not next_cursor or next_cursor == cursor or pages_without_new >= MAX_PAGES_WITHOUT_NEW:
            break
        if max_reviews and len(reviews) >= max_reviews:
            reviews = reviews[:max_reviews]
            break
        cursor = next_cursor
        pages += 1
        if pages % SAVE_EVERY_PAGES == 0:
            save_partial(out_dir, game, reviews, cursor, steam_total)
        time.sleep(PAUSE_SECONDS)
    print()
    return reviews, steam_total


def read_games(args) -> list:
    if args.games:
        with open(args.games, encoding="utf-8") as f:
            games = json.load(f)["games"]
    else:
        games = []
        for spec in args.apps:
            app_id, _, name = spec.partition("=")
            games.append({"app_id": app_id.strip(), "name": name.strip() or None})
    for g in games:
        g["app_id"] = str(g["app_id"])
        g["key"] = g.get("key") or g["app_id"]
        g["name"] = g.get("name") or g["key"]
    keys = [g["key"] for g in games]
    if len(keys) != len(set(keys)):
        raise SystemExit("Every game needs a unique key (or app id).")
    return games


def main():
    parser = argparse.ArgumentParser(description="Collect public Steam reviews of one or more games")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--apps", nargs="+", metavar="APP_ID[=NAME]", help="Steam app ids, optionally with a name")
    source.add_argument("--games", help='JSON file {"games": [{"key", "name", "app_id"}, ...]}')
    parser.add_argument("--out-dir", default=RAW_DIR)
    parser.add_argument("--language", default="all", help="Steam language code ('all', 'english', 'italian'...)")
    parser.add_argument("--max-per-game", type=int, default=None, help="Only for quick tests")
    parser.add_argument("--force", action="store_true", help="Collect again also the games already completed")
    parser.add_argument("--only", nargs="+", help="Collect only these game keys")
    args = parser.parse_args()
    games = read_games(args)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "collection_log.json")
    log = {}
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            log = {g["key"]: g for g in json.load(f).get("games", [])}

    def log_existing(game, final):                       # completed by an earlier run without a log entry
        if game["key"] not in log:
            with open(final, encoding="utf-8") as f:
                log[game["key"]] = {**game, "collected": len(json.load(f)), "steam_reported_total": None}

    failed = []
    with httpx.Client(headers={"User-Agent": "academic-research-review-collection"}) as client:
        for game in games:
            final = os.path.join(args.out_dir, f"reviews_{game['key']}.json")
            if args.only and game["key"] not in args.only:
                if os.path.exists(final):
                    log_existing(game, final)
                continue
            if os.path.exists(final) and not args.force:
                print(f"{game['name']}: already collected, skipped (use --force to collect again)")
                log_existing(game, final)
                continue
            print(f"Collecting {game['name']} (app {game['app_id']})")
            try:
                reviews, steam_total = collect_game(client, game, args.out_dir, args.language, args.max_per_game)
            except RequestFailed as e:
                print(f"\n   STOPPED: {e}. Progress is saved: run the script again later to resume.")
                failed.append(game["name"])
                continue
            with open(final, "w", encoding="utf-8") as f:
                json.dump(reviews, f, ensure_ascii=False, indent=1)
            if os.path.exists(partial_path(args.out_dir, game)):
                os.remove(partial_path(args.out_dir, game))
            log[game["key"]] = {**game, "language": args.language, "collected": len(reviews),
                                "steam_reported_total": steam_total,
                                "collected_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds")}

    merged = []
    for game in games:                                   # merge every completed game, in the order given
        final = os.path.join(args.out_dir, f"reviews_{game['key']}.json")
        if os.path.exists(final):
            with open(final, encoding="utf-8") as f:
                merged.extend(json.load(f))
    with open(os.path.join(args.out_dir, "all_reviews.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"games": list(log.values())}, f, ensure_ascii=False, indent=2)

    print("\nCollected reviews")
    print(f"{'Game':22} {'collected':>10} {'Steam total':>12}")
    for game in games:
        g = log.get(game["key"])
        if g:
            print(f"{g['name']:22} {g['collected']:>10} {str(g['steam_reported_total']):>12}")
        else:
            print(f"{game['name']:22} {'missing':>10}")
    print(f"{'TOTAL':22} {len(merged):>10}")
    if failed:
        print(f"\nNot completed: {', '.join(failed)}. Run the script again to resume them.")
    else:
        print(f"Saved to {os.path.join(args.out_dir, 'all_reviews.json')}")


if __name__ == "__main__":
    main()

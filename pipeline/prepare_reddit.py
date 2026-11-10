"""
Prepares Reddit data for the pipeline, preserving the structure of the discussions.

Every comment becomes one item whose TEXT is the comment itself (the only text that is coded) and whose CONTEXT is
the thread above it: the post and the chain of comments from the top level down to the direct parent.
Structural fields (thread, parent, depth, score, number of replies) are kept as metadata, so the coded output can be
analysed along the discussion structure (see README, "Reddit discussions").

Supported inputs (files or folders, mixed):
  - thread JSON as returned by Reddit for a thread URL + ".json" (post + nested replies), one or more per file;
  - flat dumps (JSON list or JSONL) of comments with id / parent_id / link_id / body (e.g. API or archive exports),
    optionally with a separate submissions file (--submissions) for post titles and texts.

Usage:
  python prepare_reddit.py --input threads/ --output ../Data/reddit_items.jsonl
  python prepare_reddit.py --input comments.jsonl --submissions submissions.jsonl --output reddit_items.jsonl
Options: --include-posts (also code the posts), --keep-authors (store usernames; default: salted pseudonym),
         --author-salt SECRET (stable pseudonyms across runs),
         --max-context-chars N (default 3000), --max-ancestor-chars N (default 400), --max-post-chars N (default 800).
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

REMOVED = {"[deleted]", "[removed]", "", None}


# --- Reading ---

def read_records(path: str) -> List[Any]:
    """Returns the top-level JSON values of a file (JSON or JSONL)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def iter_files(paths: Iterable[str]) -> Iterable[str]:
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith((".json", ".jsonl", ".ndjson")):
                    yield os.path.join(p, name)
        else:
            yield p


def bare(reddit_id: Optional[str]) -> Optional[str]:
    """'t1_abc' / 't3_xyz' → 'abc' / 'xyz'."""
    if not reddit_id:
        return None
    return reddit_id.split("_", 1)[1] if reddit_id[:3] in ("t1_", "t3_") else reddit_id


def is_listing(value: Any) -> bool:
    return isinstance(value, dict) and value.get("kind") == "Listing"


def walk_listing(listing: Dict[str, Any], posts: Dict[str, Dict], comments: Dict[str, Dict], stats: Dict[str, int]):
    """Flattens a Reddit Listing (post or nested comment tree)."""
    for child in listing.get("data", {}).get("children", []):
        kind, d = child.get("kind"), child.get("data", {})
        if kind == "t3":
            posts[d["id"]] = d
        elif kind == "t1":
            comments[d["id"]] = d
            if is_listing(d.get("replies")):
                walk_listing(d["replies"], posts, comments, stats)
        elif kind == "more":
            stats["more_not_loaded"] += len(d.get("children", []))


def load(inputs: List[str], submissions: Optional[str]):
    posts, comments, stats = {}, {}, defaultdict(int)
    for path in iter_files(inputs):
        for rec in read_records(path):
            if isinstance(rec, list) and rec and is_listing(rec[0]):      # one thread JSON inside a list of threads
                for listing in rec:
                    walk_listing(listing, posts, comments, stats)
            elif is_listing(rec):
                walk_listing(rec, posts, comments, stats)
            elif isinstance(rec, dict) and "body" in rec:                  # flat comment
                comments[rec["id"]] = rec
            elif isinstance(rec, dict) and "title" in rec:                 # flat submission
                posts[rec["id"]] = rec
    if submissions:
        for rec in read_records(submissions):
            if isinstance(rec, dict) and "id" in rec:
                posts[rec["id"]] = rec
    return posts, comments, stats


# --- Building items ---

def truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " […]"


def author_field(author: Optional[str], keep: bool, salt: str) -> Optional[str]:
    """Username, or a salted pseudonym: still allows counting distinct users, but cannot be reversed
    by hashing known usernames without the salt."""
    if author in REMOVED:
        return None
    return author if keep else hashlib.sha256((salt + author).encode()).hexdigest()[:12]


def build_items(posts, comments, args) -> List[Dict[str, Any]]:
    children = defaultdict(int)
    for c in comments.values():
        children[bare(c.get("parent_id"))] += 1

    def chain(comment: Dict[str, Any]):
        """Ancestors from the top-level comment down to the direct parent, and whether the chain is complete
        (it is not when an ancestor comment is missing from the data)."""
        out, seen = [], set()
        parent_full = comment.get("parent_id") or ""
        while bare(parent_full) in comments and bare(parent_full) not in seen:
            seen.add(bare(parent_full))
            out.append(comments[bare(parent_full)])
            parent_full = comments[bare(parent_full)].get("parent_id") or ""
        complete = not parent_full.startswith("t1_")     # the chain must end at the post (t3_)
        return list(reversed(out)), complete

    def context(comment: Dict[str, Any], ancestors: List[Dict[str, Any]], complete: bool) -> str:
        thread = bare(comment.get("link_id")) or (bare(comment.get("parent_id")) if not ancestors else None)
        post = posts.get(thread) if thread else None
        parts = []
        if post:
            parts.append(f"[POST] {truncate(post.get('title', ''), 300)}"
                         + (f" — {truncate(post.get('selftext', ''), args.max_post_chars)}"
                            if post.get("selftext") not in REMOVED else ""))
        if not complete:
            parts.append("[… earlier comment(s) not available in the data …]")
        lines = []
        for depth, a in enumerate(ancestors, start=1):
            body = a.get("body")
            is_parent = depth == len(ancestors)
            limit = args.max_context_chars if is_parent else args.max_ancestor_chars
            text = "[deleted]" if body in REMOVED else truncate(body, limit)
            label = f", depth {depth}" if complete else ""        # absolute depth unknown if the chain is incomplete
            lines.append(f"[{'PARENT ' if is_parent else ''}COMMENT{label}] {text}")
        # Too long: drop the oldest intermediate comments; the post and the direct parent are always kept
        dropped = 0
        marker = lambda: [f"[… {dropped} earlier comment(s) omitted …]"] if dropped else []
        while len(lines) > 1 and len("\n".join(parts + marker() + lines)) > args.max_context_chars:
            lines.pop(0)
            dropped += 1
        return "\n".join(parts + marker() + lines)

    items = []
    if args.include_posts:
        for p in posts.values():
            text = " — ".join(x for x in (p.get("title"), p.get("selftext")) if x not in REMOVED)
            items.append({"id": p["id"], "kind": "post", "thread_id": p["id"], "parent_id": None, "depth": 0,
                          "text": text, "context": None, "score": p.get("score"), "n_replies": children.get(p["id"], 0),
                          "author": author_field(p.get("author"), args.keep_authors, args.author_salt),
                          "subreddit": p.get("subreddit"), "created_utc": p.get("created_utc")})
    for c in comments.values():
        if c.get("body") in REMOVED:
            continue                         # deleted comments are not coded, but stay in their replies' context
        ancestors, complete = chain(c)
        top_parent = (ancestors[0] if ancestors else c).get("parent_id") or ""
        items.append({
            "id": c["id"], "kind": "comment",
            "thread_id": bare(c.get("link_id")) or (bare(top_parent) if top_parent.startswith("t3_") else None),
            "parent_id": bare(c.get("parent_id")),
            "depth": len(ancestors) + 1 if complete else None,     # unknown when an ancestor is missing
            "text": c["body"],
            "context": context(c, ancestors, complete),
            "score": c.get("score"), "n_replies": children.get(c["id"], 0),
            "author": author_field(c.get("author"), args.keep_authors, args.author_salt),
            "subreddit": c.get("subreddit"), "created_utc": c.get("created_utc"),
        })
    items.sort(key=lambda it: (str(it["thread_id"]), it["depth"] if it["depth"] is not None else 999,
                               it.get("created_utc") or 0))
    return items


def main():
    parser = argparse.ArgumentParser(description="Prepare Reddit threads for the thematic analysis pipeline")
    parser.add_argument("--input", nargs="+", required=True, help="Thread JSON files, flat dumps, or folders")
    parser.add_argument("--submissions", help="Optional flat file of posts (for titles/texts of flat comment dumps)")
    parser.add_argument("--output", required=True, help="Output JSONL file, used as paths.raw_input")
    parser.add_argument("--include-posts", action="store_true", help="Also code the posts themselves")
    parser.add_argument("--keep-authors", action="store_true", help="Store usernames (default: anonymous hash)")
    parser.add_argument("--author-salt", default=None,
                        help="Secret used to pseudonymise usernames; fix it to get the same pseudonyms across runs "
                             "(default: random, i.e. different pseudonyms at every run)")
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--max-ancestor-chars", type=int, default=400)
    parser.add_argument("--max-post-chars", type=int, default=800)
    args = parser.parse_args()
    if args.author_salt is None:
        args.author_salt = os.urandom(16).hex()

    posts, comments, stats = load(args.input, args.submissions)
    items = build_items(posts, comments, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    n_comments = sum(1 for it in items if it["kind"] == "comment")
    deleted = sum(1 for c in comments.values() if c.get("body") in REMOVED)
    print(f"Threads: {len({it['thread_id'] for it in items})} | posts read: {len(posts)} | comments read: {len(comments)}")
    print(f"Items written: {len(items)} ({n_comments} comments"
          + (f", {len(items) - n_comments} posts" if args.include_posts else "") + f") → {args.output}")
    print(f"Deleted/removed comments skipped: {deleted} (kept as '[deleted]' in their replies' context)")
    if stats["more_not_loaded"]:
        print(f"⚠️  {stats['more_not_loaded']} comments were not loaded by Reddit ('load more'): the thread is incomplete.")
    depths = defaultdict(int)
    for it in items:
        depths[it["depth"] if it["depth"] is not None else "unknown"] += 1
    print("Items by depth:", dict(sorted(depths.items(), key=lambda x: (isinstance(x[0], str), x[0]))))
    incomplete = depths.get("unknown", 0)
    if incomplete:
        print(f"⚠️  {incomplete} comments reply to comments missing from the data: depth unknown, context incomplete.")


if __name__ == "__main__":
    main()

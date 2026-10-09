"""Build the frozen web-search snapshot corpus for local retrieval.

    python scripts/build_snapshot_corpus.py --compass-cache <dir> --baseline-cache <dir> \
        --out data/snapshot_corpus/corpus.jsonl

Inputs are the web-search caches collected for the benchmark questions in June 2026: source-typed
results (academic / news / official, with dates) and plain web-search results (the same endpoint as
"official"). Each cache file is a JSON list of {source, weight, snippets, dates?}. Snippets are whitespace-normalized and de-duplicated per source; the first date seen for a
snippet is kept. Output: one JSON object per line {id, text, source, date}, plus a manifest with
counts and the corpus sha256.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os


def iter_cache(cache_dir: str):
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for ev in data if isinstance(data, list) else []:
            if not isinstance(ev, dict):
                continue
            snippets = ev.get("snippets") or []
            dates = ev.get("dates") or [""] * len(snippets)
            for i, s in enumerate(snippets):
                yield ev.get("source", "web"), s, (dates[i] if i < len(dates) else "") or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compass-cache", required=True)
    ap.add_argument("--baseline-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seen = {}
    order = []
    n_raw = 0
    for cache in (args.compass_cache, args.baseline_cache):
        for source, text, date in iter_cache(cache):
            n_raw += 1
            text = " ".join(str(text).split())
            if len(text) < 20:
                continue
            key = (source, text)
            if key not in seen:
                seen[key] = date
                order.append(key)
            elif date and not seen[key]:
                seen[key] = date

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    counts = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (source, text) in enumerate(order):
            counts[source] = counts.get(source, 0) + 1
            f.write(json.dumps({"id": i, "text": text, "source": source, "date": seen[(source, text)]},
                               ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    manifest = {"raw_snippets": n_raw, "docs": len(order), "by_source": counts, "sha256": sha,
                "inputs": [args.compass_cache, args.baseline_cache]}
    json.dump(manifest, open(os.path.join(os.path.dirname(args.out), "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

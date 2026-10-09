"""BM25 index over the frozen web-search snapshot (data/snapshot_corpus/corpus.jsonl).

Method-agnostic retrieval infrastructure shared by COMPASS (compass.retrieval_local, which adds the
source-weighted Multi-Source RAG and query rewriting on top) and the baselines
(compass.baselines.retrieval, one plain query over the union index "all"). It imports no COMPASS
method module, so baselines may use it without depending on the COMPASS method.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List

# Index groups: the source labels in the corpus map onto these BM25 indices.
_GROUP_OF = {"academic": "academic", "news": "news", "official": "official", "web": "official"}
GROUPS = ("academic", "news", "official")

_INDEX_LOCK = threading.Lock()
_INDEXES: Dict[str, "SnapshotIndex"] = {}


class SnapshotIndex:
    """BM25 indices (one per source group plus "all") over the frozen snippet corpus."""

    def __init__(self, corpus_path: str, index_dir: str) -> None:
        import bm25s
        import Stemmer

        self._bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        self.docs: List[dict] = [
            json.loads(line) for line in Path(corpus_path).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        self.members: Dict[str, List[int]] = {g: [] for g in (*GROUPS, "all")}
        for i, d in enumerate(self.docs):
            self.members[_GROUP_OF.get(d["source"], "official")].append(i)
            self.members["all"].append(i)
        self.index: Dict[str, object] = {}
        root = Path(index_dir)
        for g, ids in self.members.items():
            gdir = root / g
            if (gdir / "params.index.json").exists():
                self.index[g] = bm25s.BM25.load(str(gdir))
            else:
                tokens = bm25s.tokenize([self.docs[i]["text"] for i in ids], stopwords="en",
                                        stemmer=self.stemmer, show_progress=False)
                bm = bm25s.BM25()
                bm.index(tokens, show_progress=False)
                gdir.mkdir(parents=True, exist_ok=True)
                bm.save(str(gdir))
                self.index[g] = bm

    def search(self, query: str, group: str, k: int) -> List[dict]:
        """Top-k corpus entries of one group for a query (empty if the query has no usable terms)."""
        ids = self.members[group]
        if not query.strip() or not ids:
            return []
        q_tokens = self._bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, show_progress=False)
        if not q_tokens.vocab or not any(len(t) for t in q_tokens.ids):
            return []
        # Over-fetch, then drop repeated texts (the same snippet can be stored under two source
        # labels, e.g. "official" and the baselines' "web", which share one index group).
        results, scores = self.index[group].retrieve(q_tokens, k=min(3 * k, len(ids)), show_progress=False)
        hits, seen = [], set()
        for j, s in zip(results[0], scores[0]):
            doc = self.docs[ids[int(j)]]
            key = doc["text"].rstrip(". …")
            if s <= 0 or key in seen:
                continue
            seen.add(key)
            hits.append(doc)
            if len(hits) == k:
                break
        return hits


def get_index(corpus_path: str, index_dir: str) -> SnapshotIndex:
    """Process-wide singleton per corpus (loading the index is the slow part)."""
    key = f"{corpus_path}|{index_dir}"
    with _INDEX_LOCK:
        if key not in _INDEXES:
            _INDEXES[key] = SnapshotIndex(corpus_path, index_dir)
        return _INDEXES[key]

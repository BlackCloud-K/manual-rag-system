from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.http import models


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "config.yaml").exists():
            return parent
    return current.parents[2]


def _load_config() -> dict[str, Any]:
    cfg_path = _find_project_root() / "config.yaml"
    if not cfg_path.exists():
        return {}
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _sparse_to_components(lexical_weights: Any) -> tuple[list[int], list[float]]:
    if isinstance(lexical_weights, dict):
        indices = lexical_weights.get("indices")
        values = lexical_weights.get("values")
        if isinstance(indices, list) and isinstance(values, list):
            return [int(i) for i in indices], [float(v) for v in values]
    return [], []


_PROJECT_ROOT = _find_project_root()
_CONFIG = _load_config()

QDRANT_URL = str(_CONFIG.get("qdrant_url", "http://localhost:6333"))
COLLECTION_NAME = str(_CONFIG.get("collection_name", "dcs_manuals"))

retrieval_cfg = _CONFIG.get("retrieval")
if not isinstance(retrieval_cfg, dict):
    retrieval_cfg = {}
HYBRID_PREFETCH = int(retrieval_cfg.get("hybrid_prefetch", 20))
RERANK_TOP_N = int(retrieval_cfg.get("rerank_top_n", 5))
NEIGHBOR_EXPAND = bool(retrieval_cfg.get("neighbor_expand", True))

data_cfg = _CONFIG.get("data")
if not isinstance(data_cfg, dict):
    data_cfg = {}
_final_dir_cfg = str(data_cfg.get("final_dir", "data/final"))
FINAL_DIR = Path(_final_dir_cfg)
if not FINAL_DIR.is_absolute():
    FINAL_DIR = _PROJECT_ROOT / FINAL_DIR


python_embed_model = None
_rerank_model = None


def _get_embed_model() -> BGEM3FlagModel:
    global python_embed_model
    if python_embed_model is None:
        python_embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return python_embed_model


def _get_rerank_model() -> FlagReranker:
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _rerank_model


def load_chunks_lookup(pdf_name: str | None = None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for file_path in sorted(FINAL_DIR.glob("*_final.json")):
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            continue

        for item in raw:
            if not isinstance(item, dict):
                continue
            if pdf_name is not None and str(item.get("pdf_name")) != pdf_name:
                continue

            chunk_id = item.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id.strip():
                lookup[chunk_id] = item
    return lookup


def embed_query(query_text: str) -> dict[str, list[float] | list[int]]:
    model = _get_embed_model()
    emb = model.encode(
        [query_text],
        batch_size=1,
        return_dense=True,
        return_sparse=True,
    )
    dense = [float(x) for x in emb["dense_vecs"][0]]
    sparse_indices, sparse_values = _sparse_to_components(emb["lexical_weights"][0])
    return {
        "dense": dense,
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
    }


def hybrid_search(
    query_text: str,
    top_k: int = 20,
    pdf_name_filter: str | None = None,
) -> list[dict[str, Any]]:
    vectors = embed_query(query_text)
    dense = vectors["dense"]
    sparse_indices = vectors["sparse_indices"]
    sparse_values = vectors["sparse_values"]

    query_filter: models.Filter | None = None
    if pdf_name_filter is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="pdf_name",
                    match=models.MatchValue(value=pdf_name_filter),
                )
            ]
        )

    client = QdrantClient(url=QDRANT_URL)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense,
                using="dense",
                limit=HYBRID_PREFETCH,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using="sparse",
                limit=HYBRID_PREFETCH,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    hits: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            chunk_id = str(point.id)
        hits.append(
            {
                "chunk_id": chunk_id,
                "score": float(point.score),
                "payload": payload,
                "source": "search",
            }
        )
    return hits


def expand_neighbors(
    hits: list[dict[str, Any]],
    chunks_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not NEIGHBOR_EXPAND:
        return hits

    merged: dict[str, dict[str, Any]] = {}
    for hit in hits:
        chunk_id = str(hit.get("chunk_id", ""))
        if not chunk_id:
            continue
        merged[chunk_id] = hit

    for hit in hits:
        payload = hit.get("payload")
        if not isinstance(payload, dict):
            continue

        current_h1 = payload.get("title_h1")
        for neighbor_key in ("prev_chunk_id", "next_chunk_id"):
            neighbor_id = payload.get(neighbor_key)
            if not isinstance(neighbor_id, str) or not neighbor_id.strip():
                continue

            neighbor_chunk = chunks_lookup.get(neighbor_id)
            if not isinstance(neighbor_chunk, dict):
                continue
            if neighbor_chunk.get("title_h1") != current_h1:
                continue

            if neighbor_id in merged:
                if merged[neighbor_id].get("source") == "search":
                    continue

            merged[neighbor_id] = {
                "chunk_id": neighbor_id,
                "score": 0.0,
                "payload": neighbor_chunk,
                "source": "neighbor",
            }

    return list(merged.values())


def rerank(
    query_text: str,
    candidates: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    limit = RERANK_TOP_N if top_n is None else int(top_n)
    pairs: list[tuple[str, str]] = []
    for row in candidates:
        payload = row.get("payload")
        text = ""
        if isinstance(payload, dict):
            text = str(payload.get("text", ""))
        pairs.append((query_text, text))

    scores = _get_rerank_model().compute_score(pairs, normalize=True)
    if isinstance(scores, (int, float)):
        scores = [float(scores)]
    else:
        scores = [float(s) for s in scores]

    scored_rows: list[dict[str, Any]] = []
    for row, score in zip(candidates, scores):
        enriched = dict(row)
        enriched["rerank_score"] = score
        scored_rows.append(enriched)

    scored_rows.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
    return scored_rows[:limit]


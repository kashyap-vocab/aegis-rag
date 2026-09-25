"""Build all three index artifacts from the knowledge base (D4, D6).

    knowledge_base/*.md -> chunks -> SQLite docstore (text + metadata)
                                  -> FAISS (dense, bge embeddings)
                                  -> bm25s (sparse, domain tokenizer)

The build goes to a temp directory and is swapped in only when complete, so a
crashed or partial ingest never leaves a half-written index behind a live API.
"""

import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import Settings
from app.ingest.loader import load_corpus
from app.ingest.tokenizer import tokenize
from app.models.embedder import Embedder
from app.stores.bm25_store import BM25Store
from app.stores.docstore import DocStore
from app.stores.faiss_store import FaissStore


def corpus_fingerprint(kb_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(kb_dir.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def build_index(settings: Settings, embedder: Embedder) -> dict:
    t0 = time.perf_counter()
    chunks = load_corpus(settings.knowledge_base_dir, settings.chunk_max_tokens)
    if not chunks:
        raise ValueError(f"no chunks produced from {settings.knowledge_base_dir}")

    final = settings.index_dir
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    docstore = DocStore.open(tmp, read_only=False)
    ids = docstore.add_chunks(chunks)

    vectors = embedder.encode_passages([c.embed_text for c in chunks])
    dense = FaissStore(embedder.dim, index_type=settings.faiss_index_type)
    dense.add(np.array(ids), vectors)
    dense.save(tmp)

    sparse = BM25Store()
    sparse.add(ids, [tokenize(c.embed_text) for c in chunks])
    sparse.save(tmp)

    stats = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_sha256": corpus_fingerprint(settings.knowledge_base_dir),
        "documents": len({c.doc_id for c in chunks}),
        "chunks": len(chunks),
        "embed_model": settings.embed_model,
        "embed_variant": embedder.variant,
        "embed_dim": embedder.dim,
        "faiss_index_type": settings.faiss_index_type,
        "chunk_max_tokens": settings.chunk_max_tokens,
    }
    docstore.set_meta(stats)
    docstore.close()

    if not (len(dense) == len(sparse) == len(chunks)):
        raise RuntimeError(f"index size mismatch: faiss={len(dense)} bm25={len(sparse)} chunks={len(chunks)}")

    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    stats["build_s"] = round(time.perf_counter() - t0, 2)
    return stats

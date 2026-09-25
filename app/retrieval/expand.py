"""Adjacent-chunk expansion (D6).

When a hit is one piece of a longer procedure, its neighbours (chunk_idx ±n in
the same document) often hold the rest of the answer — e.g. a step whose
safety limit sits in the next chunk. Neighbours are appended as candidates
(marked ``expanded_from``) and compete on merit in the reranker.
"""

from app.schemas import RetrievedChunk
from app.stores.docstore import DocStore


def expand_adjacent(hits: list[RetrievedChunk], docstore: DocStore, n: int) -> list[RetrievedChunk]:
    if n <= 0:
        return hits
    seen = {h.id for h in hits}
    extra: list[RetrievedChunk] = []
    for hit in hits:
        for nid, chunk in sorted(docstore.neighbors(hit.chunk.doc_id, hit.chunk.chunk_idx, n).items()):
            if nid not in seen:
                seen.add(nid)
                extra.append(RetrievedChunk(id=nid, chunk=chunk, expanded_from=hit.id))
    return hits + extra

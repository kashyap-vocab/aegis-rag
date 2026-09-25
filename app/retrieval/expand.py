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

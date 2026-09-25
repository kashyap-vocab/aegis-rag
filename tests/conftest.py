import pytest

from app.config import Settings, get_settings

_S = get_settings()
MODELS_READY = _S.local_model_dir(_S.embed_model).exists() and _S.local_model_dir(_S.rerank_model).exists()
needs_models = pytest.mark.skipif(not MODELS_READY, reason="run scripts/download_models.py first")


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    return Settings(index_dir=tmp_path_factory.mktemp("idx") / "index", llm_provider="stub")


@pytest.fixture(scope="session")
def embedder(settings):
    from app.models.embedder import Embedder

    return Embedder(settings)


@pytest.fixture(scope="session")
def index_stats(settings, embedder):
    from app.ingest.indexer import build_index

    return build_index(settings, embedder)


@pytest.fixture(scope="session")
def retriever(settings, embedder, index_stats):
    from app.retrieval.hybrid import HybridRetriever

    return HybridRetriever.from_index_dir(settings, embedder)


@pytest.fixture(scope="session")
def reranker(settings):
    from app.models.reranker import Reranker

    return Reranker(settings)


@pytest.fixture(scope="session")
def pipeline(settings, retriever, reranker):
    from app.generation.llm import build_llm
    from app.pipeline import RAGPipeline

    return RAGPipeline(settings, retriever, reranker, build_llm(settings))

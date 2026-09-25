from app.config import Settings
from app.generation.llm import Generation
from app.ingest.chunker import split_sentences
from app.ingest.tokenizer import tokenize
from app.schemas import RetrievedChunk


class ExtractiveStub:
    name = "extractive-stub"

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, question: str, context: list[RetrievedChunk]) -> Generation:
        q_tokens = set(tokenize(question))
        best, best_score = None, -1.0
        for hit in context:
            weight = hit.rerank_score if hit.rerank_score is not None else 1.0
            for sentence in split_sentences(hit.chunk.text):
                s_tokens = set(tokenize(sentence))
                overlap = len(q_tokens & s_tokens) / (len(q_tokens) or 1)
                score = overlap * (0.5 + weight)
                if score > best_score:
                    best, best_score = (sentence, hit.chunk.doc_id), score

        if best is None or best_score <= 0:
            return Generation(answerable=False, answer=self.settings.refusal_message)
        sentence, doc = best
        return Generation(answerable=True, answer=sentence, supporting_quote=sentence, source_doc=doc)

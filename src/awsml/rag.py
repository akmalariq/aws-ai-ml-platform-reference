"""Retrieval-augmented generation.

Embeds a corpus, retrieves the passages most relevant to a question, and asks a
foundation model to answer using them.

Two backends, same interface:

- **AWS**: retrieval over the corpus, then generation with a model in Amazon
  Bedrock via the Converse API. Converse is model-agnostic, so Amazon Nova,
  Anthropic Claude, or another provider works by changing the model id.
- **Local**: a TF-IDF retriever and an extractive answer, so the endpoint works
  and is testable with no cloud account.

The retriever is TF-IDF in both modes; Amazon Titan embeddings are the design
target and are not implemented yet.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from awsml.config import PROJECT_ROOT, Settings

DEFAULT_CORPUS = PROJECT_ROOT / "docs"


def load_chunks(corpus: Path = DEFAULT_CORPUS) -> list[str]:
    """Split the corpus markdown into paragraph-level chunks."""
    chunks: list[str] = []
    if not corpus.exists():
        return chunks
    for path in sorted(corpus.glob("*.md")):
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            text = block.strip()
            if len(text) > 40:
                chunks.append(text)
    return chunks


class Retriever:
    """A TF-IDF retriever over paragraph chunks."""

    def __init__(self, chunks: list[str]):
        if not chunks:
            raise ValueError("corpus is empty")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, question: str, top_k: int = 3) -> list[tuple[str, float]]:
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix)[0]
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def _bedrock_generate(prompt: str, settings: Settings) -> str:
    """Call a Bedrock foundation model with the Converse API.

    Converse is model-agnostic, so the same call works across Amazon Nova,
    Anthropic Claude, and other providers without per-model request bodies.
    Requires AWS credentials and model access.
    """
    import boto3

    model_id = os.environ["AWSML_BEDROCK_MODEL_ID"]
    client = boto3.client("bedrock-runtime", region_name=settings.bedrock_region)
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.2},
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def answer(question: str, settings: Settings, top_k: int = 3) -> dict:
    """Answer a question from the corpus, using Bedrock when AWS is enabled."""
    chunks = load_chunks()
    retriever = Retriever(chunks)
    passages = retriever.search(question, top_k=top_k)
    context = "\n\n".join(text for text, _ in passages)

    use_bedrock = settings.use_aws and bool(os.getenv("AWSML_BEDROCK_MODEL_ID"))
    if use_bedrock:
        prompt = (
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return {
            "backend": "bedrock",
            "answer": _bedrock_generate(prompt, settings),
            "sources": [text for text, _ in passages],
        }

    best = passages[0][0] if passages else "No relevant passage found in the corpus."
    return {"backend": "local-extractive", "answer": best, "sources": [text for text, _ in passages]}

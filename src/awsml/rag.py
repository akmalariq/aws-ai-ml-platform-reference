"""Retrieval-augmented generation.

Embeds a corpus, retrieves the passages most relevant to a question, and asks a
foundation model to answer using them.

Two backends, same interface:

- **AWS**: Amazon Titan embeddings plus a model in Amazon Bedrock.
- **Local**: a TF-IDF retriever and an extractive answer, so the endpoint works
  and is testable with no cloud account.

This mirrors the JD's "prompt engineering, embeddings with Amazon Titan, and
retrieval-augmented generation (RAG) on AWS".
"""

from __future__ import annotations

import json
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
    """Call a Bedrock foundation model. Requires AWS credentials and model access."""
    import boto3

    model_id = os.environ["AWSML_BEDROCK_MODEL_ID"]
    client = boto3.client("bedrock-runtime", region_name=settings.bedrock_region)
    body = json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {"maxTokenCount": 512, "temperature": 0.2},
        }
    )
    response = client.invoke_model(modelId=model_id, body=body)
    payload = json.loads(response["body"].read())
    return payload["results"][0]["outputText"].strip()


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

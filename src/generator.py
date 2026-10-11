"""
Context-grounded answer generator and abstention module for DAI-009: RAG Answer Quality Benchmark.

Provides:
1. BaseLLMProvider abstract interface and LLMProviderError.
2. DeterministicMockLLMProvider: Fast, offline, deterministic test-double generator.
3. OpenAICompatibleProvider: Extensible interface for real LLMs without hardcoded credentials.
4. GeneratedAnswer: Structured response model tracking answers, abstention status, and citations.
5. AnswerGenerator: Orchestrates prompt construction, context grounding, and abstention guardrails.
"""

from __future__ import annotations

import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Ensure project root is available in sys.path
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import RetrievalResult

# Standardized canonical abstention message
ABSTENTION_MESSAGE = (
    "I cannot find sufficient information in the provided documentation to answer this question."
)

ABSTENTION_PHRASES = (
    "cannot find sufficient information",
    "not enough information",
    "not provided in the documentation",
    "does not contain information",
    "insufficient information",
    "abstain from answering",
)


class LLMProviderError(RuntimeError):
    """Raised when an external LLM provider encounters an execution or network error."""
    pass


@dataclass(frozen=True)
class GeneratedAnswer:
    """Structured response container for generated RAG answers."""
    answer: str
    is_abstention: bool
    source_chunk_ids: List[str]
    source_doc_ids: List[str]
    prompt_used: str
    provider: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return asdict(self)


# =============================================================================
# Prompt Construction
# =============================================================================

def build_prompt(query: str, retrieved_chunks: Sequence[RetrievalResult]) -> str:
    """
    Format user question and retrieved context chunks into a structured RAG prompt.
    
    Args:
        query: The user's question.
        retrieved_chunks: Ranked chunks retrieved from the knowledge base.
        
    Returns:
        str: Fully formatted prompt text.
    """
    if not retrieved_chunks:
        context_block = "[No relevant context chunks retrieved]"
    else:
        chunk_snippets: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            header = f"[Context {idx} | Document: {chunk.document_id} | Chunk: {chunk.chunk_id}]"
            snippet = f"{header}\n{chunk.content.strip()}"
            chunk_snippets.append(snippet)
        context_block = "\n\n".join(chunk_snippets)

    return (
        "You are an assistant answering questions strictly based on the provided university policy documents.\n\n"
        "Instructions:\n"
        "1. Answer the question using ONLY the facts explicitly stated in the Context Chunks below.\n"
        f"2. If the context does not contain enough information to answer the question, state exactly:\n"
        f"   \"{ABSTENTION_MESSAGE}\"\n"
        "3. Do NOT extrapolate, speculate, or invent any facts not present in the context.\n"
        "4. Keep the answer clear, factual, and concise.\n\n"
        f"--- CONTEXT CHUNKS ---\n{context_block}\n\n"
        f"--- QUESTION ---\n{query.strip()}\n\n"
        "--- ANSWER ---\n"
    )


# =============================================================================
# LLM Providers
# =============================================================================

class BaseLLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text completion given a prompt."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider implementation."""
        pass


class DeterministicMockLLMProvider(BaseLLMProvider):
    """
    Deterministic offline test-double provider.
    
    NOTE: This is a rule-based mock implementation for automated testing and
    reproducible local evaluation. It is NOT an actual neural language model.
    It extracts factual sentences from the context prompt matching question tokens,
    or returns an abstention if the context lacks evidence.
    """

    @property
    def provider_name(self) -> str:
        return "mock-deterministic-test-double"

    def generate(self, prompt: str) -> str:
        # Extract Question and Context from formatted prompt
        q_match = re.search(r"--- QUESTION ---\n(.*?)\n\n--- ANSWER ---", prompt, re.DOTALL)
        c_match = re.search(r"--- CONTEXT CHUNKS ---\n(.*?)\n\n--- QUESTION ---", prompt, re.DOTALL)

        query = q_match.group(1).strip() if q_match else ""
        context = c_match.group(1).strip() if c_match else ""

        if not context or "[No relevant context chunks retrieved]" in context:
            return ABSTENTION_MESSAGE

        def _norm(tok: str) -> str:
            tok = tok.lower()
            if len(tok) > 3 and tok.endswith("s"):
                return tok[:-1]
            return tok

        # Clean query tokens
        q_tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", query.lower())
        stopwords = {
            "what", "is", "the", "for", "and", "in", "to", "of", "a", "an",
            "are", "do", "how", "on", "can", "i", "my", "if", "or", "which",
            "where", "when", "who", "whom", "will", "would", "at", "by", "from"
        }
        content_tokens = {
            _norm(t)
            for t in q_tokens
            if t not in stopwords and len(t) >= 2
        }

        # Break context into sentences
        # Remove headers like [Context 1 | Document: ...]
        clean_context = re.sub(r"\[Context \d+ \| Document:.*?\]", "", context)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_context) if s.strip()]

        # Determine minimum required token overlap threshold
        # For short queries (<=2 content words), require at least 1 match.
        # For typical queries (>=3 content words), require at least 2 distinct matching words
        # to prevent single incidental generic words (e.g. "limit" or "rate") from hallucinating answers.
        min_overlap = 2 if len(content_tokens) >= 3 else 1

        # Score sentences by token overlap with query
        scored_sentences: List[tuple[int, str]] = []
        for s in sentences:
            s_tokens = {
                _norm(t)
                for t in re.findall(r"\b[A-Za-z0-9_-]+\b", s.lower())
            }
            overlap = len(content_tokens.intersection(s_tokens))
            if overlap >= min_overlap:
                scored_sentences.append((overlap, s))

        # If zero sentences meet the evidence threshold, abstain
        if not scored_sentences:
            return ABSTENTION_MESSAGE

        # Sort by overlap descending and take top matching unique sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences: List[str] = []
        seen_s: set[str] = set()
        for _, s in scored_sentences:
            if s not in seen_s:
                seen_s.add(s)
                top_sentences.append(s)
            if len(top_sentences) == 2:
                break

        return " ".join(top_sentences)


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Configurable LLM provider supporting OpenAI-compatible endpoints
    (e.g., local Ollama, vLLM, OpenAI, DeepSeek, or Gemini OpenAI-compatible endpoints).
    
    Reads credentials from environment variables or explicit parameters.
    Does not require hardcoded API keys.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    @property
    def provider_name(self) -> str:
        return f"openai-compatible:{self._model}"

    def generate(self, prompt: str) -> str:
        try:
            import urllib.request
            import json

            if not self._api_key:
                raise LLMProviderError(
                    "Missing API key. Set OPENAI_API_KEY environment variable or pass api_key."
                )

            url = (self._base_url.rstrip("/") if self._base_url else "https://api.openai.com/v1") + "/chat/completions"

            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return str(data["choices"][0]["message"]["content"]).strip()

        except Exception as e:
            raise LLMProviderError(f"Provider request failed: {e}") from e


# =============================================================================
# Answer Generator
# =============================================================================

class AnswerGenerator:
    """
    Orchestrates RAG response generation, prompt building, and abstention guardrails.
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
    ) -> None:
        # Default to the offline deterministic test double
        self.provider = provider or DeterministicMockLLMProvider()

    def generate(
        self,
        query: str,
        retrieved_chunks: Sequence[RetrievalResult],
    ) -> GeneratedAnswer:
        """
        Generate a grounded answer for a question based on retrieved context.
        
        Args:
            query: The user question.
            retrieved_chunks: Ranked chunks retrieved from the knowledge base.
            
        Returns:
            GeneratedAnswer: Structured answer object with citations and abstention status.
        """
        clean_query = query.strip() if query else ""

        # Guardrail 1: Empty question
        if not clean_query:
            return GeneratedAnswer(
                answer="No question was provided to answer.",
                is_abstention=True,
                source_chunk_ids=[],
                source_doc_ids=[],
                prompt_used="",
                provider=self.provider.provider_name,
            )

        # Collect unique source IDs in preserved rank order
        seen_chunks: set[str] = set()
        seen_docs: set[str] = set()
        source_chunk_ids: List[str] = []
        source_doc_ids: List[str] = []

        for r in retrieved_chunks:
            if r.chunk_id not in seen_chunks:
                seen_chunks.add(r.chunk_id)
                source_chunk_ids.append(r.chunk_id)
            if r.document_id not in seen_docs:
                seen_docs.add(r.document_id)
                source_doc_ids.append(r.document_id)

        # Guardrail 2: Zero retrieved context chunks
        if not retrieved_chunks:
            return GeneratedAnswer(
                answer=ABSTENTION_MESSAGE,
                is_abstention=True,
                source_chunk_ids=[],
                source_doc_ids=[],
                prompt_used=build_prompt(clean_query, []),
                provider=self.provider.provider_name,
            )

        # Build prompt
        prompt = build_prompt(clean_query, retrieved_chunks)

        # Execute provider generation
        raw_answer = self.provider.generate(prompt).strip()

        # Check for abstention indicators
        is_abstention = any(phrase in raw_answer.lower() for phrase in ABSTENTION_PHRASES)

        return GeneratedAnswer(
            answer=raw_answer,
            is_abstention=is_abstention,
            source_chunk_ids=source_chunk_ids if not is_abstention else [],
            source_doc_ids=source_doc_ids if not is_abstention else [],
            prompt_used=prompt,
            provider=self.provider.provider_name,
        )

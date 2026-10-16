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


def _normalize_sentence_text(text: str) -> str:
    """Normalize internal spacing and clean trailing/leading noise."""
    return re.sub(r"\s+", " ", text).strip()


def are_sentences_overlapping(s1: str, s2: str, overlap_token_thresh: float = 0.65) -> bool:
    """
    Determine if two sentences duplicate or significantly overlap in semantic content.
    Prevents including repeated or fragment-redundant sentences from overlapping chunks.
    """
    c1 = _normalize_sentence_text(s1).lower()
    c2 = _normalize_sentence_text(s2).lower()
    if not c1 or not c2:
        return False
    if c1 == c2:
        return True
    if c1 in c2 or c2 in c1:
        return True

    toks1 = set(re.findall(r"\b[A-Za-z0-9_-]+\b", c1))
    toks2 = set(re.findall(r"\b[A-Za-z0-9_-]+\b", c2))
    if not toks1 or not toks2:
        return False

    intersection = toks1.intersection(toks2)
    min_len = min(len(toks1), len(toks2))
    if min_len > 0 and (len(intersection) / min_len) >= overlap_token_thresh:
        return True

    return False


# Expanded stop words to include question phrasing, conversational fluff, and auxiliary modals
MOCK_STOPWORDS: set[str] = {
    "what", "is", "the", "for", "and", "in", "to", "of", "a", "an",
    "are", "do", "how", "on", "can", "i", "my", "if", "or", "which",
    "where", "when", "who", "whom", "will", "would", "at", "by", "from",
    "it", "its", "this", "that", "these", "those", "there", "has", "have", "had",
    "be", "been", "being", "with", "as", "about", "must", "should", "could", "all", "each",
    "so", "no", "not", "any", "then", "into", "up", "out", "per", "use", "used", "using",
    "tell", "me", "give", "explain", "describe", "show", "please", "does", "did", "also",
    "many", "much",
}

# Generic words common in policy documentation that should not alone qualify as specific topic matches
MOCK_GENERIC_WORDS: set[str] = {
    "student", "students", "university", "campus", "policy", "policies",
    "academic", "service", "services", "rule", "rules", "system", "systems",
    "program", "programs", "form", "forms", "time", "day", "days", "hour", "hours",
    "total", "number", "amount", "rate", "fee", "cost",
}

# Domain semantic aliases for institutional policy and technical terminology
MOCK_SEMANTIC_ALIASES: dict[str, set[str]] = {
    "measure": {"standard", "mandate", "rule", "policy", "requirement", "protocol", "guideline", "procedure", "control"},
    "measures": {"standards", "mandates", "rules", "policies", "requirements", "protocols", "guidelines", "procedures", "controls"},
    "standard": {"measure", "mandate", "rule", "policy", "requirement", "protocol", "guideline", "procedure", "control"},
    "standards": {"measures", "mandates", "rules", "policies", "requirements", "protocols", "guidelines", "procedures", "controls"},
    "mandate": {"measure", "standard", "rule", "policy", "requirement"},
    "mandates": {"measures", "standards", "rules", "policies", "requirements"},
    "organization": {"university", "enterprise", "institution", "department", "division"},
    "organizations": {"universities", "enterprises", "institutions", "departments"},
    "enterprise": {"organization", "university", "institution"},
    "enterprises": {"organizations", "universities", "institutions"},
    "protect": {"security", "secure", "safeguard", "defend", "protection"},
    "protects": {"securities", "secures", "safeguards", "defends"},
    "protection": {"security", "secure", "protect", "safeguard"},
    "security": {"secure", "protect", "protection", "safeguard", "defend"},
    "secure": {"security", "protect", "protection", "safeguard", "defend"},
    "data": {"information", "account", "accounts", "credential", "credentials", "record", "records"},
    "information": {"data", "record", "records"},
    "wifi": {"wireless", "wpa3", "network", "ssid"},
    "wi-fi": {"wireless", "wpa3", "network", "ssid"},
    "wireless": {"wifi", "wi-fi", "wpa3", "network", "ssid"},
    "network": {"wifi", "wi-fi", "wireless", "ssid"},
}


def _normalize_token(tok: str) -> str:
    """Normalize token by lowercasing and stripping non-root plural suffixes."""
    t = tok.lower()
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _extract_tokens(text: str) -> set[str]:
    """
    Extract normalized tokens including subwords from hyphenated or compound terms.
    e.g. 'wpa3-enterprise' -> {'wpa3-enterprise', 'wpa3', 'enterprise'}
    """
    raw_words = re.findall(r"\b[A-Za-z0-9_-]+\b", text.lower())
    tokens: set[str] = set()
    for w in raw_words:
        cleaned_w = w.strip("._-")
        if not cleaned_w:
            continue
        tokens.add(_normalize_token(cleaned_w))
        parts = re.split(r"[-_/.]", cleaned_w)
        if len(parts) > 1:
            for p in parts:
                if len(p) >= 2:
                    tokens.add(_normalize_token(p))
    return tokens


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

        q_tokens = _extract_tokens(query)
        content_tokens = {
            t for t in q_tokens
            if t not in MOCK_STOPWORDS and len(t) >= 2
        }

        # Expand content tokens with domain aliases
        expanded_tokens = set(content_tokens)
        for t in content_tokens:
            if t in MOCK_SEMANTIC_ALIASES:
                for alias in MOCK_SEMANTIC_ALIASES[t]:
                    expanded_tokens.add(_normalize_token(alias))

        # Break context into sentences and blocks cleanly
        # Split on sentence ending punctuation followed by space OR on newlines/paragraph breaks
        clean_context = re.sub(r"\[Context \d+ \| Document:.*?\]", "", context)
        raw_blocks = re.split(r"(?:(?<=[.!?])\s+|\n+)", clean_context)
        sentences: List[str] = []
        for b in raw_blocks:
            norm_b = _normalize_sentence_text(b)
            if norm_b:
                sentences.append(norm_b)

        # Core topic tokens (content tokens excluding generic institutional words)
        specific_query_tokens = {t for t in content_tokens if t not in MOCK_GENERIC_WORDS}

        # Determine minimum required token overlap threshold
        # For short queries (<=2 content words), require at least 1 match.
        # For typical queries (>=3 content words), require at least 2 distinct matching words
        # to prevent single incidental generic words (e.g. "limit" or "rate") from hallucinating answers.
        min_overlap = 2 if len(content_tokens) >= 3 else 1

        # Score sentences by token overlap with query
        scored_sentences: List[tuple[int, str]] = []
        for s in sentences:
            s_tokens = _extract_tokens(s)
            overlap = expanded_tokens.intersection(s_tokens)

            # A sentence qualifies as evidence only if:
            # 1. Total overlap count meets the required threshold.
            # 2. If the query contained specific topic keywords, at least one non-generic
            #    topic keyword (or its semantic alias) is matched, avoiding false positive
            #    matching on background administrative terms alone.
            has_specific = bool(overlap.intersection(expanded_tokens - MOCK_GENERIC_WORDS)) if specific_query_tokens else True

            if len(overlap) >= min_overlap and has_specific:
                exact_overlap = content_tokens.intersection(s_tokens)
                score = len(overlap) + len(exact_overlap)
                scored_sentences.append((score, s))

        # If zero sentences meet the evidence threshold, abstain
        if not scored_sentences:
            return ABSTENTION_MESSAGE

        # Sort by overlap descending, preferring longer/more complete sentences on ties
        scored_sentences.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        top_sentences: List[str] = []
        for _, s in scored_sentences:
            if not any(are_sentences_overlapping(s, existing) for existing in top_sentences):
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
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._timeout = timeout

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

            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return str(data["choices"][0]["message"]["content"]).strip()

        except Exception as e:
            raise LLMProviderError(f"Provider request failed: {e}") from e


def get_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    use_mock: bool = False,
) -> BaseLLMProvider:
    """
    Factory helper to instantiate an LLM provider based on name or environment configuration.
    
    Args:
        provider: Provider identifier ('mock', 'openai', 'gemini', 'openai-compatible').
        model: Model name override (e.g. 'gpt-4o-mini', 'gemini-2.0-flash').
        api_key: Optional explicit API key.
        base_url: Optional explicit base URL.
        use_mock: If True, forces use of DeterministicMockLLMProvider.
        
    Returns:
        BaseLLMProvider: Configured LLM provider instance.
    """
    if use_mock:
        return DeterministicMockLLMProvider()

    prov = (provider or os.environ.get("RAG_LLM_PROVIDER") or "").strip().lower()
    mode = os.environ.get("RAG_MODE", "offline").strip().lower()

    if not prov or prov in ("mock", "deterministic", "test_double") or (mode == "offline" and not provider):
        return DeterministicMockLLMProvider()

    if prov in ("openai", "openai-compatible"):
        target_model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        target_key = api_key or os.environ.get("OPENAI_API_KEY")
        target_url = base_url or os.environ.get("OPENAI_BASE_URL")
        return OpenAICompatibleProvider(model=target_model, api_key=target_key, base_url=target_url)

    if prov == "gemini":
        target_model = model or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash"
        target_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        target_url = base_url or os.environ.get("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai"
        return OpenAICompatibleProvider(model=target_model, api_key=target_key, base_url=target_url)

    raise ValueError(f"Unknown LLM provider: '{prov}'. Supported: 'mock', 'openai', 'gemini', 'openai-compatible'.")


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

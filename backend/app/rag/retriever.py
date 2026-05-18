import re

from app.rag.ingest import get_collection
from app.schemas import SourceTrace


def is_tier_eligible(
    tier: str,
    role: str,
    candidate_seniority: str | None = None,
    previous_answer_score: float | None = None,
    question_difficulty: str | None = None,
    allow_advanced_sources: bool = True,
) -> bool:
    if tier in {"foundation", "core"}:
        return True
    if tier == "applied":
        return role == "DATA_SCIENCE_APPLIED_ML" or question_difficulty in {
            "intermediate",
            "applied",
            "advanced",
        }
    if tier == "advanced":
        if not allow_advanced_sources:
            return False
        return (
            candidate_seniority in {"intermediate", "advanced"}
            or (previous_answer_score is not None and previous_answer_score >= 8)
            or question_difficulty == "advanced"
        )
    return False


def retrieve(
    role: str,
    query: str,
    top_k: int = 4,
    candidate_seniority: str | None = None,
    previous_answer_score: float | None = None,
    question_difficulty: str | None = None,
    allow_advanced_sources: bool = True,
    excluded_chunk_ids: set[str] | None = None,
) -> list[SourceTrace]:
    try:
        collection = get_collection(role)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    excluded_chunk_ids = excluded_chunk_ids or set()
    query_tokens = tokenize_for_overlap(query)
    candidate_count = min(max(top_k * 8, 24), min(collection.count(), 96))
    result = collection.query(query_texts=[query], n_results=candidate_count)
    semantic_records = zip(
        result.get("ids", [[]])[0],
        result.get("documents", [[]])[0],
        result.get("metadatas", [[]])[0],
        result.get("distances", [[]])[0]
        if result.get("distances")
        else [None] * len(result.get("documents", [[]])[0]),
    )
    traces = build_traces(
        semantic_records,
        role=role,
        candidate_seniority=candidate_seniority,
        previous_answer_score=previous_answer_score,
        question_difficulty=question_difficulty,
        allow_advanced_sources=allow_advanced_sources,
        excluded_chunk_ids=excluded_chunk_ids,
        query_tokens=query_tokens,
    )
    if len(traces) < top_k:
        traces.extend(
            lexical_fallback_traces(
                collection=collection,
                role=role,
                query=query,
                top_k=top_k,
                candidate_seniority=candidate_seniority,
                previous_answer_score=previous_answer_score,
                question_difficulty=question_difficulty,
                allow_advanced_sources=allow_advanced_sources,
                excluded_chunk_ids=excluded_chunk_ids | {trace.chunk_id for trace in traces},
                query_tokens=query_tokens,
            )
        )
    preferred = prefer_tiers(role, traces, question_difficulty)
    if preferred:
        return preferred[:top_k]

    if excluded_chunk_ids:
        return retrieve(
            role=role,
            query=query,
            top_k=top_k,
            candidate_seniority=candidate_seniority,
            previous_answer_score=previous_answer_score,
            question_difficulty=question_difficulty,
            allow_advanced_sources=allow_advanced_sources,
            excluded_chunk_ids=set(),
        )
    return []


def build_traces(
    records,
    *,
    role: str,
    candidate_seniority: str | None,
    previous_answer_score: float | None,
    question_difficulty: str | None,
    allow_advanced_sources: bool,
    excluded_chunk_ids: set[str],
    query_tokens: set[str],
) -> list[SourceTrace]:
    traces: list[SourceTrace] = []
    seen: set[str] = set()
    for chunk_id, document, metadata, distance in records:
        if chunk_id in seen or chunk_id in excluded_chunk_ids:
            continue
        seen.add(chunk_id)
        if is_bad_chunk_text(document):
            continue
        relevance = lexical_overlap_score(query_tokens, document, metadata)
        if relevance <= 0.0:
            continue
        tier = str(metadata.get("tier") or "foundation")
        if not is_tier_eligible(
            tier,
            role,
            candidate_seniority=candidate_seniority,
            previous_answer_score=previous_answer_score,
            question_difficulty=question_difficulty,
            allow_advanced_sources=allow_advanced_sources,
        ):
            continue
        dense_score = None if distance is None else max(0.0, 1.0 - float(distance))
        combined_score = relevance + (dense_score * 0.15 if dense_score is not None else 0.0)
        traces.append(
            SourceTrace(
                chunk_id=chunk_id,
                source_filename=str(metadata.get("source_filename") or metadata.get("source_name") or "unknown"),
                display_name=str(metadata.get("display_name") or metadata.get("source_filename") or "unknown"),
                page_number=int(metadata["page_number"]) if metadata.get("page_number") else None,
                tier=tier,
                chunk_index=int(metadata["chunk_index"]) if metadata.get("chunk_index") is not None else None,
                score=combined_score,
                text=document[:500],
            )
        )
    return traces


def lexical_fallback_traces(
    *,
    collection,
    role: str,
    query: str,
    top_k: int,
    candidate_seniority: str | None,
    previous_answer_score: float | None,
    question_difficulty: str | None,
    allow_advanced_sources: bool,
    excluded_chunk_ids: set[str],
    query_tokens: set[str],
) -> list[SourceTrace]:
    limit = min(collection.count(), 4000)
    if limit == 0 or not hasattr(collection, "peek"):
        return []
    peeked = collection.peek(limit=limit)
    ranked: list[tuple[float, str, str, dict, None]] = []
    for chunk_id, document, metadata in zip(
        peeked.get("ids", []),
        peeked.get("documents", []),
        peeked.get("metadatas", []),
    ):
        if chunk_id in excluded_chunk_ids or is_bad_chunk_text(document):
            continue
        tier = str(metadata.get("tier") or "foundation")
        if not is_tier_eligible(
            tier,
            role,
            candidate_seniority=candidate_seniority,
            previous_answer_score=previous_answer_score,
            question_difficulty=question_difficulty,
            allow_advanced_sources=allow_advanced_sources,
        ):
            continue
        score = lexical_overlap_score(query_tokens, document, metadata)
        if score <= 0:
            continue
        ranked.append((score, chunk_id, document, metadata, None))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return build_traces(
        ((chunk_id, document, metadata, distance) for _, chunk_id, document, metadata, distance in ranked[: max(top_k * 6, 24)]),
        role=role,
        candidate_seniority=candidate_seniority,
        previous_answer_score=previous_answer_score,
        question_difficulty=question_difficulty,
        allow_advanced_sources=allow_advanced_sources,
        excluded_chunk_ids=excluded_chunk_ids,
        query_tokens=query_tokens,
    )


def tokenize_for_overlap(text: str) -> set[str]:
    stopwords = {
        "the", "and", "with", "from", "into", "that", "this", "your", "using", "built",
        "resume", "role", "question", "selected", "candidate", "adaptive", "follow", "workflow",
        "engineer", "engineering", "advanced", "intermediate", "beginner",
        "python", "fastapi", "sql", "chromadb", "docker", "git", "numpy", "pandas",
        "machine", "learning", "artificial", "intelligence", "deep", "introduction",
        "pattern", "recognition", "statistics", "science",
        "system", "systems", "design", "analysis", "quality", "production", "approach",
        "problem", "example", "examples", "great", "success", "general", "purpose",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if token not in stopwords
    }


def lexical_overlap_score(query_tokens: set[str], document: str, metadata: dict) -> float:
    document_tokens = tokenize_for_overlap(document)
    filename_tokens = tokenize_for_overlap(str(metadata.get("source_filename") or ""))
    overlap = query_tokens & (document_tokens | filename_tokens)
    if not overlap:
        return 0.0
    bonus_terms = {
        "validation", "tracking", "uncertainty", "regularization", "generalization", "classification",
        "regression", "cross", "probabilistic", "graphical", "drift", "bias", "variance", "missing",
    }
    bonus = len(overlap & bonus_terms) * 0.4
    density = len(overlap) / max(8, len(query_tokens))
    return len(overlap) + bonus + density


def prefer_tiers(
    role: str,
    traces: list[SourceTrace],
    question_difficulty: str | None = None,
) -> list[SourceTrace]:
    if question_difficulty == "advanced":
        tier_rank = {"advanced": 0, "core": 1, "applied": 2, "foundation": 3}
        return sorted(traces, key=lambda trace: (tier_rank.get(trace.tier, 9), -(trace.score or 0.0)))
    if question_difficulty in {"beginner", "probing"}:
        tier_rank = {"foundation": 0, "core": 1, "applied": 2, "advanced": 3}
        return sorted(traces, key=lambda trace: (tier_rank.get(trace.tier, 9), -(trace.score or 0.0)))
    if role == "DATA_SCIENCE_APPLIED_ML":
        tier_rank = {"applied": 0, "core": 1, "foundation": 2, "advanced": 3}
        return sorted(traces, key=lambda trace: (tier_rank.get(trace.tier, 9), -(trace.score or 0.0)))
    if question_difficulty == "intermediate":
        tier_rank = {"core": 0, "applied": 1, "foundation": 2, "advanced": 3}
        return sorted(traces, key=lambda trace: (tier_rank.get(trace.tier, 9), -(trace.score or 0.0)))
    return traces


def is_bad_chunk_text(text: str) -> bool:
    normalized = " ".join((text or "").split())
    lowered = normalized.lower()
    if len(normalized.split()) < 12:
        return True
    if looks_like_frontmatter_or_preface(normalized):
        return True
    if any(term in lowered for term in ["index", "references", "bibliography", "contents"]):
        return True
    if re.search(r"(figure|fig\.)\s*\d+(\.\d+)?", lowered):
        meaningful_sentences = count_meaningful_sentences(normalized)
        if meaningful_sentences < 2:
            return True
    if has_page_number_list_pattern(normalized):
        return True
    if has_comma_glossary_pattern(normalized):
        return True
    if has_too_many_standalone_numbers(normalized):
        return True
    if has_notebook_or_code_artifacts(normalized):
        return True
    if count_meaningful_sentences(normalized) < 1:
        return True
    return False


def looks_like_frontmatter_or_preface(text: str) -> bool:
    lowered = text.lower()
    frontmatter_markers = [
        "information science and statistics",
        "expert systems are one of the",
        "examples of hand-written dig",
        "introduction to ai",
        "introduction to machine learning",
    ]
    if any(marker in lowered for marker in frontmatter_markers):
        return True
    if lowered.startswith(("1. introduction", "chapter 1", "preface")):
        return True
    if re.match(r"^\d+\s+[•.-]\s+", text) and any(term in lowered for term in ["artificial intelligence", "machine learning", "deep learning"]):
        return True
    return False


def has_page_number_list_pattern(text: str) -> bool:
    number_tokens = re.findall(r"\b\d+\b", text)
    comma_count = text.count(",")
    if len(number_tokens) >= 6 and comma_count >= 4:
        return True
    return bool(re.search(r"(?:\b\d+\b[\s,;]+){5,}\b\d+\b", text))


def has_comma_glossary_pattern(text: str) -> bool:
    segments = [segment.strip() for segment in text.split(",") if segment.strip()]
    if len(segments) < 6:
        return False
    short_segments = sum(1 for segment in segments if len(segment.split()) <= 4)
    return short_segments / len(segments) >= 0.7


def has_too_many_standalone_numbers(text: str) -> bool:
    numbers = re.findall(r"\b\d+\b", text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
    if not words:
        return True
    return len(numbers) > max(6, len(words) // 3)


def has_notebook_or_code_artifacts(text: str) -> bool:
    lowered = text.lower()
    artifact_patterns = [
        r"\bin \[\d+\]:",
        r"\bout\[\d+\]:",
        r"\bprint\?",
        r"\bplt\.",
        r"\bax\.",
        r"\bimport\s+[a-z_]+",
        r"\bsequential\(",
        r"\bstandardscaler\(",
        r"\bfit\(",
        r"\bpredict\(",
        r"\bkeras\b",
        r"\bpandas\b",
        r"\bnumpy\b",
        r"\blab:",
    ]
    artifact_hits = sum(1 for pattern in artifact_patterns if re.search(pattern, lowered))
    if artifact_hits >= 2:
        return True
    code_like_tokens = len(re.findall(r"[A-Za-z_]+\([^)]*\)|\[[^\]]+\]|[A-Za-z_]+\.[A-Za-z_]+", text))
    word_count = len(re.findall(r"[A-Za-z][A-Za-z0-9-]*", text))
    return code_like_tokens >= 6 and code_like_tokens > max(4, word_count // 6)


def count_meaningful_sentences(text: str) -> int:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", text) if segment.strip()]
    meaningful = 0
    for sentence in sentences:
        word_count = len(re.findall(r"[A-Za-z][A-Za-z0-9-]*", sentence))
        uppercase_tokens = len(re.findall(r"\b[A-Z]{3,}\b", sentence))
        if word_count >= 8 and uppercase_tokens < max(4, word_count // 3):
            meaningful += 1
    return meaningful

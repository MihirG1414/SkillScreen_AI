import json
import re
from typing import Any

from sqlalchemy.orm import Session

from app.models import Answer, Evaluation, Question, ScreeningSession
from app.schemas import AdaptationDecision, EvaluationPayload, InterviewAnswerResponse
from app.services.question_service import INTERVIEW_QUESTION_LIMIT, start_interview


STOPWORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "answer",
    "using",
    "would",
    "should",
    "about",
    "source",
}


def submit_and_evaluate_answer(
    db: Session, session_id: str, question_id: str, answer_text: str
) -> InterviewAnswerResponse:
    question = db.get(Question, question_id)
    session = db.get(ScreeningSession, session_id)
    if session is None:
        raise ValueError("Session not found.")
    if question is None or question.session_id != session_id:
        raise ValueError("Question not found for this session.")
    if not answer_text.strip():
        raise ValueError("Answer text is required.")

    metadata = load_question_metadata(question)
    evaluation_data = evaluate_answer_deterministically(
        question=question,
        answer_text=answer_text.strip(),
        selected_role=session.role or "UNKNOWN_ROLE",
        metadata=metadata,
    )
    adaptation = decide_adaptation(evaluation_data["score"])
    evaluation_data["adaptation_decision"] = adaptation.model_dump()
    evaluation_data["advanced_sources_unlocked"] = adaptation.advanced_sources_unlocked

    answer = Answer(question_id=question.id, answer_text=answer_text.strip())
    evaluation = Evaluation(
        answer=answer,
        score=float(evaluation_data["score"]),
        band=evaluation_data["band"],
        rationale_json=json.dumps(evaluation_data),
    )

    db.add(answer)
    db.add(evaluation)
    db.commit()
    db.refresh(answer)
    db.refresh(evaluation)
    db.expire_all()

    next_question = None
    question_count = db.query(Question).filter(Question.session_id == session_id).count()
    if question_count < INTERVIEW_QUESTION_LIMIT:
        next_question = start_interview(
            db=db,
            session_id=session_id,
            requested_difficulty=adaptation.next_difficulty,
        )

    return InterviewAnswerResponse(
        answer_id=answer.id,
        evaluation=EvaluationPayload(
            id=evaluation.id,
            score=evaluation.score,
            band=evaluation.band,
            technical_accuracy=evaluation_data["technical_accuracy"],
            clarity=evaluation_data["clarity"],
            depth=evaluation_data["depth"],
            strengths=evaluation_data["strengths"],
            missing_points=evaluation_data["missing_points"],
            feedback=evaluation_data["feedback"],
            ideal_answer_summary=evaluation_data["ideal_answer_summary"],
            source_grounding_notes=evaluation_data["source_grounding_notes"],
        ),
        adaptation_decision=adaptation,
        next_question=next_question,
    )


def load_question_metadata(question: Question) -> dict[str, Any]:
    data = json.loads(question.source_trace_json or "{}")
    return data if isinstance(data, dict) else {"rag_trace": data}


def evaluate_answer_deterministically(
    *,
    question: Question,
    answer_text: str,
    selected_role: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    expected_points = [str(point) for point in metadata.get("expected_points", [])]
    rag_trace = metadata.get("rag_trace", [])
    source_tier = str(metadata.get("source_tier_used") or "unknown")
    answer_tokens = tokenize(answer_text)
    question_tokens = tokenize(question.prompt)
    topic_tokens = tokenize(question.topic)
    context_tokens = tokenize(" ".join(str(trace.get("chunk_preview", "")) for trace in rag_trace))

    point_hits = [
        point
        for point in expected_points
        if overlap_ratio(answer_tokens, tokenize(point)) >= 0.14
        or len(answer_tokens & tokenize(point)) >= 2
    ]
    missing_points = [point for point in expected_points if point not in point_hits]
    context_hit_count = len(answer_tokens & context_tokens)
    topic_relevance = overlap_ratio(answer_tokens, question_tokens | topic_tokens)
    word_count = len(answer_text.split())

    technical_accuracy = clamp(
        2.5 + len(point_hits) * 2.0 + min(context_hit_count, 8) * 0.30 + topic_relevance * 2.0
    )
    clarity = clarity_score(word_count)
    depth = depth_score(answer_text, len(point_hits), source_tier)
    strong_coverage_bonus = 0.5 if len(point_hits) >= 2 and word_count >= 30 else 0.0
    score = round(
        clamp(technical_accuracy * 0.45 + clarity * 0.25 + depth * 0.30 + strong_coverage_bonus),
        1,
    )

    strengths = build_strengths(point_hits, context_hit_count, word_count)
    feedback = build_feedback(score, missing_points, selected_role)
    source_grounding_notes = build_source_grounding_notes(rag_trace, context_hit_count, source_tier)
    ideal_answer_summary = build_ideal_answer_summary(expected_points, question.topic, source_tier)

    return {
        "score": score,
        "band": band_for_score(score),
        "technical_accuracy": round(technical_accuracy, 1),
        "clarity": round(clarity, 1),
        "depth": round(depth, 1),
        "strengths": strengths,
        "missing_points": missing_points,
        "feedback": feedback,
        "ideal_answer_summary": ideal_answer_summary,
        "source_grounding_notes": source_grounding_notes,
    }


def decide_adaptation(score: float) -> AdaptationDecision:
    if score >= 8:
        return AdaptationDecision(
            next_difficulty="advanced",
            advanced_sources_unlocked=True,
            reason="Strong answer: unlock advanced sources and increase difficulty.",
        )
    if score >= 5:
        return AdaptationDecision(
            next_difficulty="intermediate",
            advanced_sources_unlocked=False,
            reason="Partial answer: continue with core or applied sources at intermediate difficulty.",
        )
    return AdaptationDecision(
        next_difficulty="beginner",
        advanced_sources_unlocked=False,
        reason="Weak answer: probe fundamentals with foundation or core sources.",
    )


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", text.lower())
        if token not in STOPWORDS
    }


def overlap_ratio(answer_tokens: set[str], target_tokens: set[str]) -> float:
    if not target_tokens:
        return 0.0
    return len(answer_tokens & target_tokens) / len(target_tokens)


def clamp(value: float, minimum: float = 0.0, maximum: float = 10.0) -> float:
    return max(minimum, min(maximum, value))


def clarity_score(word_count: int) -> float:
    if word_count < 6:
        return 2.0
    if word_count < 20:
        return 5.0
    if word_count < 45:
        return 7.5
    return 9.0


def depth_score(answer_text: str, point_hit_count: int, source_tier: str) -> float:
    lower = answer_text.lower()
    depth_terms = [
        "tradeoff",
        "edge",
        "failure",
        "test",
        "validate",
        "persist",
        "metadata",
        "source",
        "retry",
        "idempotent",
        "concurrency",
        "recovery",
        "security",
    ]
    hits = sum(1 for term in depth_terms if term in lower)
    tier_bonus = 0.6 if source_tier in {"advanced", "applied"} else 0.0
    return clamp(1.5 + point_hit_count * 1.2 + hits * 0.55 + tier_bonus)


def build_strengths(point_hits: list[str], context_hit_count: int, word_count: int) -> list[str]:
    strengths: list[str] = []
    if point_hits:
        strengths.append("Addresses expected points from the generated question.")
    if context_hit_count:
        strengths.append("Uses terms grounded in the retrieved source context.")
    if word_count >= 30:
        strengths.append("Provides enough detail for a demo screening signal.")
    return strengths or ["Gives a direct response, but needs more technical detail."]


def build_feedback(score: float, missing_points: list[str], selected_role: str) -> str:
    if score >= 8:
        return f"Strong {selected_role} answer. The next question will increase difficulty."
    if score >= 5:
        return f"Solid start for {selected_role}, but add more concrete steps and tradeoffs."
    if missing_points:
        return f"Needs more detail for {selected_role}. Start by covering: {missing_points[0]}"
    return f"Needs more detail for {selected_role}. Explain the design step by step."


def build_source_grounding_notes(
    rag_trace: list[dict[str, Any]],
    context_hit_count: int,
    source_tier: str,
) -> str:
    if not rag_trace:
        return "No RAG trace was available for this question."
    source_names = sorted({str(trace.get("display_name") or trace.get("source_filename")) for trace in rag_trace})
    if context_hit_count:
        return f"Answer overlaps with retrieved {source_tier} source context from {', '.join(source_names[:2])}."
    return f"Retrieved {source_tier} source context was available, but the answer did not clearly use it."


def build_ideal_answer_summary(
    expected_points: list[str],
    topic: str,
    source_tier: str,
) -> str:
    points = "; ".join(expected_points[:3]) if expected_points else f"cover {topic} with concrete examples"
    return f"A strong answer should {points}, while grounding the explanation in the {source_tier} source."


def band_for_score(score: float) -> str:
    if score >= 8:
        return "strong"
    if score >= 5:
        return "developing"
    return "needs_probe"

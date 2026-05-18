import json

from sqlalchemy.orm import Session

from app.config import get_settings
from app.llm.client import get_llm_provider, summarize_trace_for_question
from app.models import Question, ScreeningSession
from app.rag.retriever import retrieve
from app.schemas import QuestionPayload, RagTrace, ResumeProfile, SourceTrace

INTERVIEW_QUESTION_LIMIT = 5
BASELINE_QUESTION_COUNT = 2
EARLY_KB_UNLOCK_SCORE = 4.0


def start_interview(
    db: Session,
    session_id: str,
    requested_difficulty: str | None = None,
) -> QuestionPayload:
    session = db.get(ScreeningSession, session_id)
    if session is None:
        raise ValueError("Session not found.")
    if not session.role:
        raise ValueError("Session does not have a selected role.")

    profile = ResumeProfile.model_validate(json.loads(session.profile_json or "{}"))
    existing_question_count = db.query(Question).filter(Question.session_id == session_id).count()
    difficulty = choose_difficulty(profile, requested_difficulty, existing_question_count)
    context_mode = choose_context_mode(session, existing_question_count)
    query = build_query(session.role, profile, existing_question_count, context_mode)
    traces: list[SourceTrace] = []
    if context_mode == "kb":
        excluded_chunk_ids = previously_used_chunk_ids(session)
        traces = retrieve(
            role=session.role,
            query=query,
            top_k=6,
            candidate_seniority=profile.seniority_level,
            question_difficulty=difficulty,
            allow_advanced_sources=allow_advanced_sources(existing_question_count, requested_difficulty),
            excluded_chunk_ids=excluded_chunk_ids,
        )
        if not traces:
            broad_query = " ".join(
                [session.role, *query_focus_hints(session.role, profile, existing_question_count, context_mode), difficulty]
            )
            traces = retrieve(
                role=session.role,
                query=broad_query,
                top_k=6,
                candidate_seniority=profile.seniority_level,
                question_difficulty=difficulty,
                allow_advanced_sources=allow_advanced_sources(existing_question_count, requested_difficulty),
                excluded_chunk_ids=excluded_chunk_ids,
            )
        if traces:
            traces = rotate_traces_for_followup(traces, existing_question_count)
        else:
            context_mode = "resume_fallback"
    generated = generate_question(
        role=session.role,
        profile=profile,
        traces=traces,
        difficulty=difficulty,
        previous_questions=previous_question_context(session),
        context_mode=context_mode,
    )
    if existing_question_count:
        followup_number = existing_question_count + 1
        generated.question_text = f"Follow-up {followup_number}: {generated.question_text}"
        generated.why_this_question = (
            f"Adaptive follow-up {followup_number}. {generated.why_this_question}"
        )
    rag_trace = build_rag_trace(query, traces)

    question = Question(
        session_id=session.id,
        prompt=generated.question_text,
        topic=generated.topic,
        difficulty=generated.difficulty,
        retrieval_query=query,
        source_trace_json=json.dumps(
            {
                "rag_trace": [trace.model_dump() for trace in rag_trace],
                "raw_traces": [trace.model_dump() for trace in traces],
                "why_this_question": generated.why_this_question,
                "expected_points": generated.expected_points,
                "question_type": generated.question_type,
                "source_tier_used": generated.source_tier_used,
            }
        ),
    )
    session.status = "question_generated"
    db.add(question)
    db.add(session)
    db.commit()
    db.refresh(question)

    return QuestionPayload(
        question_id=question.id,
        question_text=question.prompt,
        topic=question.topic,
        difficulty=question.difficulty,
        question_type=generated.question_type,
        why_this_question=generated.why_this_question,
        expected_points=generated.expected_points,
        source_tier_used=generated.source_tier_used,
        rag_trace=rag_trace,
    )


def build_query(
    role: str,
    profile: ResumeProfile,
    existing_question_count: int = 0,
    context_mode: str = "kb",
) -> str:
    signals: list[str] = []
    if context_mode == "kb":
        signals.extend(profile.suggested_topics[:4])
        signals.extend(profile.domains[:2])
        signals.extend(project_query_hints(role, profile.projects))
        signals.append(profile.seniority_level)
    else:
        signals.extend(profile.skills[:6])
        signals.extend(profile.projects[:3])
        signals.extend(profile.suggested_topics[:5])
        signals.append(profile.seniority_level)
    signals.extend(query_focus_hints(role, profile, existing_question_count, context_mode))
    if context_mode == "resume_only":
        signals.append("resume-and-role baseline interview question")
    elif existing_question_count == 0:
        signals.append("resume-guided baseline interview question")
    else:
        signals.append("knowledge-grounded interview question")
        signals.append(f"adaptive follow-up question {existing_question_count + 1}")

    deduped: list[str] = []
    for signal in signals:
        cleaned = " ".join(str(signal).split())
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)

    if not deduped:
        deduped = [profile.summary or "technical project discussion"]
    return f"{role} {' '.join(deduped)}"


def query_focus_hints(
    role: str,
    profile: ResumeProfile,
    existing_question_count: int,
    context_mode: str,
) -> list[str]:
    if role == "AI_ML_ENGINEER":
        baseline_hints = [
            ["model validation", "failure analysis", "computer vision pipeline"],
            ["data quality", "drift detection", "latency tradeoffs"],
        ]
        kb_hints = [
            ["cross-validation", "generalization", "regularization"],
            ["probabilistic models", "uncertainty", "tracking systems"],
            ["missing data", "model comparison", "deployment robustness"],
        ]
        if context_mode == "resume_only":
            return baseline_hints[min(existing_question_count, len(baseline_hints) - 1)]
        kb_index = max(0, existing_question_count - BASELINE_QUESTION_COUNT)
        return kb_hints[kb_index % len(kb_hints)]

    if role == "BACKEND_ENGINEER":
        if context_mode == "resume_only":
            return ["request validation", "safe API errors", "persistence"]
        return ["idempotency", "failure recovery", "concurrency"]

    if role == "DATA_SCIENCE_APPLIED_ML":
        if context_mode == "resume_only":
            return ["model evaluation", "feature quality", "business metrics"]
        return ["cross-validation", "error analysis", "deployment monitoring"]

    return []


def project_query_hints(role: str, projects: list[str]) -> list[str]:
    project_text = " ".join(projects).lower()
    if role == "AI_ML_ENGINEER":
        candidates = [
            "computer vision",
            "tracking",
            "classification",
            "model validation",
            "error analysis",
            "drift",
            "latency",
            "tradeoffs",
            "uncertainty",
        ]
    elif role == "BACKEND_ENGINEER":
        candidates = [
            "request validation",
            "metadata",
            "persistence",
            "api",
            "retry",
            "idempotency",
            "error handling",
        ]
    else:
        candidates = [
            "model evaluation",
            "feature quality",
            "data drift",
            "experimentation",
            "tradeoffs",
        ]
    return [candidate for candidate in candidates if candidate in project_text]


def previously_used_chunk_ids(session: ScreeningSession) -> set[str]:
    used: set[str] = set()
    for question in session.questions:
        try:
            metadata = json.loads(question.source_trace_json or "{}")
        except json.JSONDecodeError:
            continue
        raw_traces = metadata.get("raw_traces", []) if isinstance(metadata, dict) else []
        for trace in raw_traces:
            chunk_id = str(trace.get("chunk_id") or "").strip()
            if chunk_id:
                used.add(chunk_id)
    return used


def rotate_traces_for_followup(
    traces: list[SourceTrace],
    existing_question_count: int,
) -> list[SourceTrace]:
    if not traces or existing_question_count <= 0:
        return traces
    offset = existing_question_count % len(traces)
    return traces[offset:] + traces[:offset]


def choose_difficulty(
    profile: ResumeProfile,
    requested_difficulty: str | None,
    existing_question_count: int = 0,
) -> str:
    requested = (requested_difficulty or "").strip().lower()
    if requested in {"beginner", "probing"}:
        return "beginner"
    if requested in {"medium", "intermediate"}:
        return "intermediate"
    if requested == "advanced":
        return "advanced"
    if existing_question_count == 0:
        if profile.seniority_level == "beginner":
            return "beginner"
        return "intermediate"
    if profile.seniority_level == "advanced":
        return "advanced"
    if profile.seniority_level == "intermediate":
        return "intermediate"
    return "beginner"


def allow_advanced_sources(
    existing_question_count: int,
    requested_difficulty: str | None,
) -> bool:
    requested = (requested_difficulty or "").strip().lower()
    if requested == "advanced":
        return True
    return existing_question_count > 0


def choose_context_mode(session: ScreeningSession, existing_question_count: int) -> str:
    if existing_question_count < BASELINE_QUESTION_COUNT:
        return "resume_only"
    early_scores = first_two_scores(session)
    if len(early_scores) < BASELINE_QUESTION_COUNT:
        return "resume_only"
    if all(score >= EARLY_KB_UNLOCK_SCORE for score in early_scores):
        return "kb"
    return "resume_only"


def first_two_scores(session: ScreeningSession) -> list[float]:
    scored_questions: list[tuple[str, float]] = []
    ordered_questions = sorted(session.questions, key=lambda question: question.created_at)
    for question in ordered_questions[:BASELINE_QUESTION_COUNT]:
        if not question.answers:
            continue
        latest_answer = sorted(question.answers, key=lambda answer: answer.created_at)[-1]
        if not latest_answer.evaluations:
            continue
        latest_evaluation = sorted(
            latest_answer.evaluations,
            key=lambda evaluation: evaluation.created_at,
        )[-1]
        scored_questions.append((question.id, float(latest_evaluation.score)))
    return [score for _, score in scored_questions]


def build_rag_trace(retrieval_query: str, traces: list[SourceTrace]) -> list[RagTrace]:
    return [
        RagTrace(
            retrieval_query=retrieval_query,
            source_filename=trace.source_filename,
            display_name=trace.display_name,
            page_number=trace.page_number,
            tier=trace.tier,
            context_summary=summarize_trace_for_question(trace),
            chunk_preview=trace.text[:320],
        )
        for trace in traces
    ]


def generate_question(
    role: str,
    profile: ResumeProfile,
    traces: list[SourceTrace],
    difficulty: str,
    previous_questions: list[dict] | None = None,
    context_mode: str = "kb",
) -> QuestionPayload:
    provider = get_llm_provider(
        get_settings().llm_provider,
        openai_api_key=get_settings().openai_api_key,
        gemini_api_key=get_settings().gemini_api_key,
    )
    generated = provider.generate_question(
        role=role,
        profile=profile,
        traces=traces,
        difficulty=difficulty,
        previous_questions=previous_questions or [],
        context_mode=context_mode,
    )
    return QuestionPayload(
        question_id="pending",
        question_text=generated["question_text"],
        topic=generated["topic"],
        difficulty=generated["difficulty"],
        question_type=generated["question_type"],
        why_this_question=generated["why_this_question"],
        expected_points=generated["expected_points"],
        source_tier_used=generated["source_tier_used"],
        rag_trace=[],
    )


def previous_question_context(session: ScreeningSession) -> list[dict]:
    history: list[dict] = []
    for question in session.questions:
        try:
            metadata = json.loads(question.source_trace_json or "{}")
        except json.JSONDecodeError:
            metadata = {}
        history.append(
            {
                "question_text": question.prompt,
                "topic": question.topic,
                "question_type": str(metadata.get("question_type") or ""),
            }
        )
    return history

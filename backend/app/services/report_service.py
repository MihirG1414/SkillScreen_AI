import json
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy.orm import Session

from app.models import Question, Report, ScreeningSession
from app.schemas import ReportPayload, ResumeProfile


TIER_KEYS = ["foundation", "core", "applied", "advanced"]


def generate_report(db: Session, session_id: str) -> ReportPayload:
    session = db.get(ScreeningSession, session_id)
    if session is None:
        raise ValueError("Session not found.")

    profile = ResumeProfile.model_validate(json.loads(session.profile_json or "{}"))
    question_records = collect_question_records(session)
    evaluated_records = [record for record in question_records if record.get("evaluation")]
    if not evaluated_records:
        raise ValueError("No evaluated answers are available for this session.")

    scores = [float(record["evaluation"]["score"]) for record in evaluated_records]
    overall_score = round(sum(scores) / len(scores), 1)
    role_fit = role_fit_for_score(overall_score)
    recommendation = recommendation_for_score(overall_score)
    source_usage = build_source_usage_summary(question_records)
    advanced_readiness = build_advanced_readiness(evaluated_records, source_usage)
    topic_breakdown = build_topic_breakdown(evaluated_records)

    report_data = {
        "overall_score": overall_score,
        "role_fit": role_fit,
        "recommendation": recommendation,
        "candidate_summary": build_candidate_summary(profile, session.role, overall_score, role_fit),
        "technical_strengths": collect_unique(evaluated_records, "strengths"),
        "areas_for_improvement": collect_unique(evaluated_records, "missing_points"),
        "topic_breakdown": topic_breakdown,
        "question_answer_summary": build_question_answer_summary(evaluated_records),
        "source_usage_summary": source_usage,
        "advanced_readiness": advanced_readiness,
        "suggested_next_round_questions": build_next_round_questions(profile, topic_breakdown, advanced_readiness),
    }

    report = Report(
        session_id=session.id,
        overall_score=overall_score,
        recommendation=recommendation,
        report_json=json.dumps(report_data),
    )
    session.status = "report_ready"
    db.add(report)
    db.add(session)
    db.commit()
    db.refresh(report)

    return ReportPayload(id=report.id, **report_data)


def collect_question_records(session: ScreeningSession) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for question in session.questions:
        metadata = load_question_trace(question)
        latest_answer = question.answers[-1] if question.answers else None
        latest_evaluation = None
        if latest_answer and latest_answer.evaluations:
            evaluation = latest_answer.evaluations[-1]
            latest_evaluation = json.loads(evaluation.rationale_json or "{}")
            latest_evaluation["id"] = evaluation.id
            latest_evaluation["score"] = evaluation.score
            latest_evaluation["band"] = evaluation.band

        records.append(
            {
                "question": question,
                "question_metadata": metadata,
                "answer": latest_answer,
                "evaluation": latest_evaluation,
            }
        )
    return records


def build_source_usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    chunks_by_tier = {tier: 0 for tier in TIER_KEYS}
    books: set[str] = set()
    advanced_triggered = False

    for record in records:
        metadata = record["question_metadata"]
        if metadata.get("source_tier_used") == "advanced":
            advanced_triggered = True
        evaluation = record.get("evaluation") or {}
        adaptation = evaluation.get("adaptation_decision") or {}
        if adaptation.get("advanced_sources_unlocked"):
            advanced_triggered = True

        for trace in metadata.get("rag_trace", []):
            tier = str(trace.get("tier") or "unknown")
            if tier in chunks_by_tier:
                chunks_by_tier[tier] += 1
            display_name = str(trace.get("display_name") or trace.get("source_filename") or "").strip()
            if display_name:
                books.add(display_name)

    return {
        "chunks_by_tier": chunks_by_tier,
        "source_books_used": sorted(books),
        "advanced_candidate_evaluation_triggered": advanced_triggered,
    }


def build_advanced_readiness(
    records: list[dict[str, Any]],
    source_usage: dict[str, Any],
) -> dict[str, Any]:
    advanced_records = [
        record
        for record in records
        if record["question_metadata"].get("source_tier_used") == "advanced"
    ]
    strong_advanced = any(float(record["evaluation"]["score"]) >= 7 for record in advanced_records)
    triggered = bool(source_usage["advanced_candidate_evaluation_triggered"])

    if strong_advanced:
        readiness = "Ready for advanced theoretical questions."
    elif triggered:
        readiness = "Advanced evaluation was triggered, but answers need more theoretical depth."
    else:
        readiness = "Not enough evidence yet for advanced theoretical questioning."

    return {
        "ready_for_advanced_theory": strong_advanced,
        "advanced_evaluation_triggered": triggered,
        "summary": readiness,
    }


def build_topic_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    topic_scores: dict[str, list[float]] = defaultdict(list)
    for record in records:
        question: Question = record["question"]
        topic_scores[question.topic].append(float(record["evaluation"]["score"]))

    return {
        topic: {
            "average_score": round(sum(scores) / len(scores), 1),
            "questions": len(scores),
        }
        for topic, scores in sorted(topic_scores.items())
    }


def build_question_answer_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        question: Question = record["question"]
        answer = record["answer"]
        evaluation = record["evaluation"]
        metadata = record["question_metadata"]
        summaries.append(
            {
                "number": index,
                "question_id": question.id,
                "question_text": question.prompt,
                "topic": question.topic,
                "difficulty": question.difficulty,
                "source_tier_used": metadata.get("source_tier_used", "unknown"),
                "answer_text": answer.answer_text if answer else "",
                "answer_summary": summarize_answer(answer.answer_text if answer else ""),
                "score": evaluation["score"],
                "feedback": evaluation.get("feedback", ""),
                "adaptation_decision": evaluation.get("adaptation_decision", {}),
                "rag_trace": metadata.get("rag_trace", []),
            }
        )
    return summaries


def build_candidate_summary(
    profile: ResumeProfile,
    role: str | None,
    overall_score: float,
    role_fit: str,
) -> str:
    name = profile.candidate_name or "The candidate"
    skills = ", ".join(profile.skills[:5]) if profile.skills else "the submitted resume skills"
    return (
        f"{name} completed the SkillScreen AI interview for {role or 'the selected role'} "
        f"with an overall score of {overall_score}/10 and a {role_fit.lower()} role-fit signal. "
        f"The strongest resume-linked evidence appeared around {skills}."
    )


def build_next_round_questions(
    profile: ResumeProfile,
    topic_breakdown: dict[str, Any],
    advanced_readiness: dict[str, Any],
) -> list[str]:
    weakest_topics = sorted(
        topic_breakdown.items(),
        key=lambda item: item[1]["average_score"],
    )
    topic = weakest_topics[0][0] if weakest_topics else (profile.suggested_topics[0] if profile.suggested_topics else "system design")
    questions = [
        f"Ask a deeper follow-up on {topic}, requiring concrete implementation tradeoffs.",
        "Ask the candidate to walk through failure handling and testing strategy for their proposed design.",
    ]
    if advanced_readiness["ready_for_advanced_theory"]:
        questions.append("Include one advanced theory question tied to the highest-tier source used in this session.")
    else:
        questions.append("Probe fundamentals before adding more advanced theoretical material.")
    return questions


def collect_unique(records: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for record in records:
        for item in record["evaluation"].get(key, []):
            if item and item not in values:
                values.append(item)
    return values or ["No strong signal captured yet."]


def summarize_answer(answer: str) -> str:
    compact = " ".join(answer.split())
    if len(compact) <= 220:
        return compact
    return f"{compact[:217]}..."


def role_fit_for_score(score: float) -> str:
    if score >= 8.5:
        return "Excellent"
    if score >= 7:
        return "Strong"
    if score >= 5:
        return "Moderate"
    return "Low"


def recommendation_for_score(score: float) -> str:
    if score >= 8:
        return "Strong Proceed"
    if score >= 5:
        return "Proceed"
    return "Needs Review"


def get_session_snapshot(db: Session, session_id: str) -> dict[str, Any]:
    session = db.get(ScreeningSession, session_id)
    if session is None:
        raise ValueError("Session not found.")

    questions = []
    for question in session.questions:
        metadata = load_question_trace(question)
        answers = []
        for answer in question.answers:
            answers.append(
                {
                    "id": answer.id,
                    "answer_text": answer.answer_text,
                    "evaluations": [
                        {
                            "id": evaluation.id,
                            "score": evaluation.score,
                            "band": evaluation.band,
                            "rationale": json.loads(evaluation.rationale_json or "{}"),
                        }
                        for evaluation in answer.evaluations
                    ],
                }
            )
        questions.append(
            {
                "question_id": question.id,
                "question_text": question.prompt,
                "topic": question.topic,
                "difficulty": question.difficulty,
                "retrieval_query": question.retrieval_query,
                "rag_trace": metadata.get("rag_trace", []),
                "question_metadata": metadata,
                "answers": answers,
            }
        )

    latest_report = session.reports[-1] if session.reports else None
    return {
        "id": session.id,
        "role": session.role,
        "resume_filename": session.resume_filename,
        "profile": json.loads(session.profile_json or "{}"),
        "status": session.status,
        "questions": questions,
        "report": json.loads(latest_report.report_json) if latest_report else None,
    }


def load_question_trace(question: Question) -> dict[str, Any]:
    data = json.loads(question.source_trace_json or "{}")
    if isinstance(data, list):
        return {"rag_trace": data}
    if isinstance(data, dict):
        return data
    return {"rag_trace": []}

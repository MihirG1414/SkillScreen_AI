from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class ResumeProfile(BaseModel):
    candidate_name: str | None = None
    skills: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    seniority_level: str = "beginner"
    suggested_topics: list[str] = Field(default_factory=list)
    summary: str = ""


class ResumeUploadResponse(BaseModel):
    session_id: str
    selected_role: str
    extracted_profile: ResumeProfile
    text_length: int


class KnowledgeIngestResponse(BaseModel):
    documents_ingested: int
    documents_skipped: int
    documents_failed: int
    chunks_added: int
    errors: list[dict[str, str]] = Field(default_factory=list)


class KnowledgeStatusResponse(BaseModel):
    documents_ingested: int
    chunks_per_role: dict[str, int]
    chunks_per_tier: dict[str, int]
    source_filenames: list[str]
    advanced_sources_available: bool


class InterviewStartRequest(BaseModel):
    session_id: str
    requested_difficulty: str | None = None


class SourceTrace(BaseModel):
    chunk_id: str
    source_filename: str
    display_name: str
    page_number: int | None = None
    tier: str
    chunk_index: int | None = None
    score: float | None = None
    text: str


class RagTrace(BaseModel):
    retrieval_query: str
    source_filename: str
    display_name: str
    page_number: int | None = None
    tier: str
    context_summary: str = ""
    chunk_preview: str


class QuestionPayload(BaseModel):
    question_id: str
    question_text: str
    topic: str
    difficulty: str
    question_type: str
    why_this_question: str
    expected_points: list[str]
    source_tier_used: str
    rag_trace: list[RagTrace]


class InterviewStartResponse(BaseModel):
    session_id: str
    question: QuestionPayload


class InterviewAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    answer_text: str


class EvaluationPayload(BaseModel):
    id: str
    score: float
    band: str
    technical_accuracy: float
    clarity: float
    depth: float
    strengths: list[str]
    missing_points: list[str]
    feedback: str
    ideal_answer_summary: str
    source_grounding_notes: str


class AdaptationDecision(BaseModel):
    next_difficulty: str
    advanced_sources_unlocked: bool
    reason: str


class InterviewAnswerResponse(BaseModel):
    answer_id: str
    evaluation: EvaluationPayload
    adaptation_decision: AdaptationDecision
    next_question: QuestionPayload | None = None


class ReportGenerateRequest(BaseModel):
    session_id: str


class ReportPayload(BaseModel):
    id: str
    overall_score: float
    role_fit: str
    recommendation: str
    candidate_summary: str
    technical_strengths: list[str]
    areas_for_improvement: list[str]
    topic_breakdown: dict[str, Any]
    question_answer_summary: list[dict[str, Any]]
    source_usage_summary: dict[str, Any]
    advanced_readiness: dict[str, Any]
    suggested_next_round_questions: list[str]


class ReportGenerateResponse(BaseModel):
    session_id: str
    report: ReportPayload


class SessionResponse(BaseModel):
    session: dict[str, Any]

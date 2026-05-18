from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ScreeningSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("sess"))
    role: Mapped[str | None] = mapped_column(String, nullable=True)
    resume_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    resume_text: Mapped[str] = mapped_column(Text, default="")
    profile_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String, default="resume_uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    questions: Mapped[list["Question"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    reports: Mapped[list["Report"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("q"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str] = mapped_column(String, default="General")
    difficulty: Mapped[str] = mapped_column(String, default="medium")
    retrieval_query: Mapped[str] = mapped_column(Text, default="")
    source_trace_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    session: Mapped[ScreeningSession] = relationship(back_populates="questions")
    answers: Mapped[list["Answer"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("ans"))
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"), nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    question: Mapped[Question] = relationship(back_populates="answers")
    evaluations: Mapped[list["Evaluation"]] = relationship(
        back_populates="answer", cascade="all, delete-orphan"
    )


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("eval"))
    answer_id: Mapped[str] = mapped_column(ForeignKey("answers.id"), nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    band: Mapped[str] = mapped_column(String, default="insufficient")
    rationale_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    answer: Mapped[Answer] = relationship(back_populates="evaluations")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("rep"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    recommendation: Mapped[str] = mapped_column(String, default="needs_review")
    report_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    session: Mapped[ScreeningSession] = relationship(back_populates="reports")


class KnowledgeIngestion(Base):
    __tablename__ = "knowledge_ingestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String, nullable=False)
    source_name: Mapped[str] = mapped_column(String, nullable=False)
    chunks_added: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

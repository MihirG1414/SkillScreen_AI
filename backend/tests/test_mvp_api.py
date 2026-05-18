from pathlib import Path

from fastapi.testclient import TestClient

import app.rag.ingest as ingest_module
from app.rag.chunker import chunk_text
from app.llm.client import (
    FallbackProvider,
    extract_concept_from_chunks,
    validate_generated_question,
)
from app.main import app
from app.schemas import ResumeProfile, SourceTrace
from app.rag.retriever import retrieve


client = TestClient(app)


class KnowledgeTestSettings:
    def __init__(self, root: Path) -> None:
        self.chroma_dir = root / "chroma"
        self.knowledge_dir = root / "raw"
        self.manifest_path = root / "source_manifest.json"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)


def use_temp_knowledge_base(monkeypatch, tmp_path: Path) -> KnowledgeTestSettings:
    settings = KnowledgeTestSettings(tmp_path)
    monkeypatch.setattr(ingest_module, "get_settings", lambda: settings)
    return settings


def test_health_route_reports_ok():
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "skillscreen-api"


def test_resume_upload_creates_session_from_text_file(tmp_path: Path):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "Asha Rao\nPython FastAPI RAG PostgreSQL\nBuilt a resume screening API with vector search.",
        encoding="utf-8",
    )

    with resume.open("rb") as file:
        response = client.post(
            "/api/resume/upload",
            data={"selected_role": "Backend Engineer"},
            files={"file": ("resume.txt", file, "text/plain")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["selected_role"] == "Backend Engineer"
    assert body["extracted_profile"]["candidate_name"] == "Asha Rao"
    assert "Python" in body["extracted_profile"]["programming_languages"]
    assert "FastAPI" in body["extracted_profile"]["frameworks"]
    assert body["extracted_profile"]["seniority_level"] in {
        "beginner",
        "intermediate",
        "advanced",
    }


def test_resume_upload_requires_selected_role(tmp_path: Path):
    resume = tmp_path / "resume.txt"
    resume.write_text("Asha Rao\nPython FastAPI\nBuilt backend APIs.", encoding="utf-8")

    with resume.open("rb") as file:
        response = client.post(
            "/api/resume/upload",
            files={"file": ("resume.txt", file, "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "selected_role is required."


def test_resume_upload_rejects_unsupported_file_type(tmp_path: Path):
    resume = tmp_path / "resume.md"
    resume.write_text("# Asha Rao", encoding="utf-8")

    with resume.open("rb") as file:
        response = client.post(
            "/api/resume/upload",
            data={"selected_role": "Backend Engineer"},
            files={"file": ("resume.md", file, "text/markdown")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF and TXT files are supported."


def test_resume_upload_rejects_empty_resume(tmp_path: Path):
    resume = tmp_path / "resume.txt"
    resume.write_text("", encoding="utf-8")

    with resume.open("rb") as file:
        response = client.post(
            "/api/resume/upload",
            data={"selected_role": "Backend Engineer"},
            files={"file": ("resume.txt", file, "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_resume_upload_returns_safe_error_for_unreadable_pdf(tmp_path: Path):
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"not really a pdf")

    with resume.open("rb") as file:
        response = client.post(
            "/api/resume/upload",
            data={"selected_role": "Backend Engineer"},
            files={"file": ("resume.pdf", file, "application/pdf")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unable to read the uploaded PDF."


def test_chunk_text_uses_word_windows_with_overlap():
    text = " ".join(f"word{i}" for i in range(1000))

    chunks = chunk_text(text, chunk_size_words=800, overlap_words=120)

    assert len(chunks) == 2
    assert len(chunks[0].split()) == 800
    assert chunks[0].split()[-120:] == chunks[1].split()[:120]


def test_knowledge_ingest_uses_source_manifest_and_status_shape(monkeypatch, tmp_path: Path):
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    knowledge = settings.knowledge_dir / "pytest_backend_knowledge.txt"
    advanced = settings.knowledge_dir / "pytest_advanced_backend.md"
    bad_pdf = settings.knowledge_dir / "pytest_bad_source.pdf"
    knowledge.write_text(
        " ".join(["FastAPI services validate uploads persist metadata return safe errors."] * 140),
        encoding="utf-8",
    )
    advanced.write_text(
        " ".join(["Advanced backend design compares distributed tracing and concurrency limits."] * 140),
        encoding="utf-8",
    )
    bad_pdf.write_bytes(b"not a readable pdf")
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "pytest_backend_knowledge.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "core",
      "display_name": "Pytest Backend Knowledge"
    },
    {
      "filename": "pytest_advanced_backend.md",
      "role": "BACKEND_ENGINEER",
      "tier": "advanced",
      "display_name": "Pytest Advanced Backend"
    },
    {
      "filename": "pytest_bad_source.pdf",
      "role": "BACKEND_ENGINEER",
      "tier": "advanced",
      "display_name": "Bad PDF"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    ingest = client.post("/api/knowledge/ingest?force=true")

    assert ingest.status_code == 200
    body = ingest.json()
    assert body["documents_ingested"] == 2
    assert body["chunks_added"] >= 1
    assert body["documents_failed"] == 1
    assert body["errors"][0]["source_filename"] == "pytest_bad_source.pdf"

    status = client.get("/api/knowledge/status")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["documents_ingested"] >= 1
    assert status_body["chunks_per_role"]["BACKEND_ENGINEER"] >= 1
    assert status_body["chunks_per_tier"]["core"] >= 1
    assert "pytest_backend_knowledge.txt" in status_body["source_filenames"]
    assert status_body["advanced_sources_available"] is True


def test_knowledge_ingest_skips_duplicate_manifest_sources_without_force(monkeypatch, tmp_path: Path):
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    knowledge = settings.knowledge_dir / "pytest_backend_knowledge.txt"
    knowledge.write_text(
        " ".join(["FastAPI services validate uploads persist metadata return safe errors."] * 140),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "pytest_backend_knowledge.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "core",
      "display_name": "Pytest Backend Knowledge"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    first = client.post("/api/knowledge/ingest")
    second = client.post("/api/knowledge/ingest")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["documents_skipped"] >= 1


def test_interview_start_uses_session_role_and_returns_resume_aware_rag_question(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "\n".join(
            [
                "Asha Rao",
                "Python FastAPI RAG ChromaDB",
                "Built a resume screening API with vector search and source tracing.",
            ]
        ),
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    knowledge = settings.knowledge_dir / "backend_core.txt"
    knowledge.write_text(
        " ".join(
            [
                "Backend interview evidence should cover upload validation, persistence, "
                "source trace metadata, and API error handling."
            ]
            * 130
        ),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "backend_core.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "core",
      "display_name": "Backend Core Notes"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")

    question_response = client.post("/api/interview/start", json={"session_id": session_id})

    assert question_response.status_code == 200
    question = question_response.json()["question"]
    assert question["question_id"]
    assert question["question_text"]
    assert question["question_type"] in {
        "resume_grounding",
        "conceptual",
        "applied_scenario",
        "debugging",
        "system_design",
        "comparison",
        "advanced_reasoning",
    }
    assert "Python" in question["question_text"] or "FastAPI" in question["question_text"]
    assert "generic" not in question["why_this_question"].lower()
    assert question["expected_points"]
    assert question["source_tier_used"] == "resume_only"
    assert question["rag_trace"] == []


def test_interview_start_blocks_advanced_source_for_beginner_candidate(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "Asha Rao\nPython\nLearning backend fundamentals through coursework and practice.",
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    foundation = settings.knowledge_dir / "backend_foundation.txt"
    advanced = settings.knowledge_dir / "backend_advanced.txt"
    foundation.write_text(
        " ".join(["Foundation backend APIs validate inputs and persist data safely."] * 130),
        encoding="utf-8",
    )
    advanced.write_text(
        " ".join(["Advanced backend systems use consensus protocols and distributed locks."] * 130),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "backend_foundation.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "foundation",
      "display_name": "Backend Foundation"
    },
    {
      "filename": "backend_advanced.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "advanced",
      "display_name": "Backend Advanced"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")

    question_response = client.post("/api/interview/start", json={"session_id": session_id})

    assert question_response.status_code == 200
    question = question_response.json()["question"]
    assert question["difficulty"] != "advanced"
    assert question["source_tier_used"] == "resume_only"
    assert question["rag_trace"] == []


def test_interview_start_begins_at_intermediate_for_strong_resume(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "\n".join(
            [
                "Asha Rao",
                "Python FastAPI ChromaDB RAG SQLAlchemy Docker PostgreSQL",
                "Built backend APIs, ingestion pipelines, vector search systems, and evaluation workflows.",
                "Implemented observability, retries, error handling, and persistence strategy.",
            ]
        ),
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    core = settings.knowledge_dir / "backend_core.txt"
    advanced = settings.knowledge_dir / "backend_advanced.txt"
    core.write_text(
        " ".join(["Core backend systems validate requests, persist metadata, and expose safe API errors."] * 140),
        encoding="utf-8",
    )
    advanced.write_text(
        " ".join(["Advanced backend systems coordinate concurrent ingestion with idempotency and failure recovery."] * 140),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "backend_core.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "core",
      "display_name": "Backend Core"
    },
    {
      "filename": "backend_advanced.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "advanced",
      "display_name": "Backend Advanced"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")

    question_response = client.post("/api/interview/start", json={"session_id": session_id})

    assert question_response.status_code == 200
    question = question_response.json()["question"]
    assert question["difficulty"] == "intermediate"
    assert question["source_tier_used"] == "resume_only"
    assert question["rag_trace"] == []


def test_answer_submission_evaluates_and_unlocks_advanced_next_question(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "\n".join(
            [
                "Asha Rao",
                "Python FastAPI ChromaDB RAG SQLAlchemy Docker",
                "Built APIs for document ingestion, source trace retrieval, validation, persistence, and evaluation.",
                "Implemented production style testing and error handling.",
            ]
        ),
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    core = settings.knowledge_dir / "backend_core.txt"
    advanced = settings.knowledge_dir / "backend_advanced.txt"
    core.write_text(
        " ".join(["Core backend systems validate requests, persist metadata, and expose safe API errors."] * 140),
        encoding="utf-8",
    )
    advanced.write_text(
        " ".join(["Advanced backend systems coordinate concurrent ingestion with idempotency and failure recovery."] * 140),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "backend_core.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "core",
      "display_name": "Backend Core"
    },
    {
      "filename": "backend_advanced.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "advanced",
      "display_name": "Backend Advanced"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")
    first_question = client.post("/api/interview/start", json={"session_id": session_id}).json()["question"]

    first_answer_response = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": first_question["question_id"],
            "answer_text": (
                "I would connect my Python FastAPI and RAG project experience to the design by validating "
                "uploads, persisting metadata, storing source trace records, testing error handling, and using "
                "idempotent ingestion so retries do not duplicate chunks. I would discuss tradeoffs around "
                "concurrency, failure recovery, clear API feedback, and how to validate the workflow with tests."
            ),
        },
    )

    assert first_answer_response.status_code == 200
    first_body = first_answer_response.json()
    evaluation = first_body["evaluation"]
    assert evaluation["score"] >= 6.5
    second_question = first_body["next_question"]
    assert second_question is not None
    assert second_question["source_tier_used"] == "resume_only"

    second_answer_response = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": second_question["question_id"],
            "answer_text": (
                "Using Python FastAPI, I would design the workflow around request validation, metadata persistence, "
                "safe retries, source trace integrity, and reliable API behavior. I would verify the design with "
                "tests, logs, duplicate-prevention checks, response-time metrics, and explicit failure-case review "
                "before rollout."
            ),
        },
    )

    assert second_answer_response.status_code == 200
    body = second_answer_response.json()
    evaluation = body["evaluation"]
    assert evaluation["score"] >= 6.5
    assert evaluation["technical_accuracy"] >= 6
    assert evaluation["clarity"] >= 7
    assert evaluation["depth"] >= 6.5
    assert evaluation["strengths"]
    assert body["adaptation_decision"]["next_difficulty"] in {"intermediate", "advanced"}
    assert body["next_question"] is not None
    assert body["next_question"]["source_tier_used"] in {"foundation", "core", "applied", "advanced"}
    assert body["next_question"]["rag_trace"]
    assert body["next_question"]["question_text"] != second_question["question_text"]


def test_answer_submission_continues_until_five_questions(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "Asha Rao\nPython FastAPI SQLAlchemy\nBuilt backend APIs with validation, metadata persistence, and testing.",
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    foundation = settings.knowledge_dir / "backend_foundation.txt"
    foundation.write_text(
        " ".join(["Foundation backend APIs validate inputs, persist data, and return safe errors."] * 160),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "backend_foundation.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "foundation",
      "display_name": "Backend Foundation"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")
    current_question = client.post("/api/interview/start", json={"session_id": session_id}).json()["question"]

    for _ in range(4):
        answer_response = client.post(
            "/api/interview/answer",
            json={
                "session_id": session_id,
                "question_id": current_question["question_id"],
                "answer_text": (
                    "I would validate inputs, persist metadata, explain tradeoffs, test failure paths, "
                    "and return safe API errors with clear recovery behavior."
                ),
            },
        )
        assert answer_response.status_code == 200
        body = answer_response.json()
        assert body["next_question"] is not None
        current_question = body["next_question"]

    final_answer = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": current_question["question_id"],
            "answer_text": (
                "I would validate inputs, persist metadata, explain tradeoffs, test failure paths, "
                "and return safe API errors with clear recovery behavior."
            ),
        },
    )

    assert final_answer.status_code == 200
    assert final_answer.json()["next_question"] is None
    session_snapshot = client.get(f"/api/session/{session_id}").json()["session"]
    assert len(session_snapshot["questions"]) == 5


def test_interview_start_generates_resume_only_question_when_no_knowledge_is_ingested(
    monkeypatch,
    tmp_path: Path,
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "Asha Rao\nPython FastAPI SQLAlchemy\nBuilt backend APIs with validation and persistence.",
        encoding="utf-8",
    )

    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    settings.manifest_path.write_text('{"sources": []}', encoding="utf-8")

    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )

    session_id = resume_response.json()["session_id"]
    question_response = client.post("/api/interview/start", json={"session_id": session_id})

    assert question_response.status_code == 200
    question = question_response.json()["question"]
    assert question["source_tier_used"] == "resume_only"
    assert question["rag_trace"] == []
    assert "Python" in question["question_text"] or "FastAPI" in question["question_text"]


def test_first_two_questions_are_resume_only_then_kb_unlocks_on_third_after_strong_answers(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "\n".join(
            [
                "Asha Rao",
                "Python FastAPI SQLAlchemy ChromaDB",
                "Built backend ingestion services, validation logic, and source trace persistence.",
            ]
        ),
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    knowledge = settings.knowledge_dir / "backend_core.txt"
    knowledge.write_text(
        " ".join(
            [
                "Backend services should validate requests, persist metadata, return safe API errors, and use idempotent retries."
            ]
            * 120
        ),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "backend_core.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "core",
      "display_name": "Backend Core Notes"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")

    first_question = client.post("/api/interview/start", json={"session_id": session_id}).json()["question"]
    assert first_question["source_tier_used"] == "resume_only"
    assert first_question["rag_trace"] == []

    second_payload = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": first_question["question_id"],
            "answer_text": (
                "I would use Python FastAPI to validate inputs, persist metadata, add retries, test failure paths, "
                "and explain tradeoffs around idempotency and API error handling."
            ),
        },
    ).json()["next_question"]
    assert second_payload["source_tier_used"] == "resume_only"
    assert second_payload["rag_trace"] == []

    third_payload = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": second_payload["question_id"],
            "answer_text": (
                "I would connect the workflow to validation, persistence, retries, and safe API behavior, "
                "then measure failure recovery and duplicate prevention with tests and logs."
            ),
        },
    ).json()["next_question"]

    assert third_payload is not None
    assert third_payload["source_tier_used"] == "core"
    assert third_payload["rag_trace"]


def test_remaining_questions_stay_resume_only_when_first_two_answers_are_not_strong(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "Asha Rao\nPython FastAPI SQLAlchemy\nBuilt backend APIs with validation and persistence.",
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    knowledge = settings.knowledge_dir / "backend_core.txt"
    knowledge.write_text(
        " ".join(
            [
                "Backend services should validate requests, persist metadata, return safe API errors, and use idempotent retries."
            ]
            * 120
        ),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "backend_core.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "core",
      "display_name": "Backend Core Notes"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")

    first_question = client.post("/api/interview/start", json={"session_id": session_id}).json()["question"]
    second_payload = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": first_question["question_id"],
            "answer_text": "I am not sure.",
        },
    ).json()["next_question"]
    assert second_payload["source_tier_used"] == "resume_only"

    third_payload = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": second_payload["question_id"],
            "answer_text": "Maybe I would add an API.",
        },
    ).json()["next_question"]

    assert third_payload is not None
    assert third_payload["source_tier_used"] == "resume_only"
    assert third_payload["rag_trace"] == []


def test_ai_ml_engineer_flow_uses_resume_only_then_kb_question_after_strong_opening_answers(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "\n".join(
            [
                "Asha Rao",
                "Python PyTorch OpenCV Scikit-learn",
                "Built computer vision pipelines, model validation workflows, and tracking experiments.",
            ]
        ),
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    foundation = settings.knowledge_dir / "ml_foundation.txt"
    core = settings.knowledge_dir / "ml_core.txt"
    advanced = settings.knowledge_dir / "ml_advanced.txt"
    foundation.write_text(
        " ".join(
            [
                "Machine learning workflows should validate datasets, compare baselines, and inspect failure cases before deployment."
            ]
            * 120
        ),
        encoding="utf-8",
    )
    core.write_text(
        " ".join(
            [
                "Computer vision systems should validate data quality, compare tracking approaches, and measure operational tradeoffs."
            ]
            * 120
        ),
        encoding="utf-8",
    )
    advanced.write_text(
        " ".join(
            [
                "Probabilistic models can represent uncertainty in tracking, latent state estimation, and sequential updates."
            ]
            * 120
        ),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "ml_foundation.txt",
      "role": "AI_ML_ENGINEER",
      "tier": "foundation",
      "display_name": "ML Foundation"
    },
    {
      "filename": "ml_core.txt",
      "role": "AI_ML_ENGINEER",
      "tier": "core",
      "display_name": "ML Core"
    },
    {
      "filename": "ml_advanced.txt",
      "role": "AI_ML_ENGINEER",
      "tier": "advanced",
      "display_name": "ML Advanced"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "AI_ML_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")

    first_question = client.post("/api/interview/start", json={"session_id": session_id}).json()["question"]
    assert first_question["source_tier_used"] == "resume_only"
    assert first_question["rag_trace"] == []

    second_question = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": first_question["question_id"],
            "answer_text": (
                "I would use Python and OpenCV to validate datasets, compare baseline models, inspect failure cases, "
                "measure tracking quality before deployment, and explain the tradeoffs between accuracy, latency, and stability."
            ),
        },
    ).json()["next_question"]
    assert second_question["source_tier_used"] == "resume_only"
    assert second_question["rag_trace"] == []

    third_question = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": second_question["question_id"],
            "answer_text": (
                "I would compare tracking approaches, validate data quality, inspect failure cases, and measure "
                "operational tradeoffs with evaluation metrics, error analysis, and production-readiness checks "
                "before deciding what is ready to ship."
            ),
        },
    ).json()["next_question"]

    assert third_question is not None
    assert third_question["source_tier_used"] in {"foundation", "core", "advanced"}
    assert third_question["rag_trace"]


def test_five_question_flow_uses_distinct_grounded_questions_when_sources_are_distinct(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "Asha Rao\nPython FastAPI SQLAlchemy\nBuilt backend APIs, ingestion services, and evaluation pipelines.",
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    sources = {
        "backend_validation.txt": "Backend validation should reject malformed payloads and return safe API errors with clear reasons.",
        "backend_persistence.txt": "Session persistence should store interview state, generated questions, answers, and reports for continuity.",
        "backend_idempotency.txt": "Idempotent ingestion should prevent duplicate writes and make retries safe after failures.",
        "backend_traceability.txt": "Source traceability should preserve retrieval queries, source filenames, and page references for explainability.",
        "backend_observability.txt": "Observability should include structured logs, failure-path testing, and metrics across routes and services.",
    }
    for filename, content in sources.items():
        (settings.knowledge_dir / filename).write_text(content, encoding="utf-8")

    manifest_entries = ",\n".join(
        f"""{{
      "filename": "{filename}",
      "role": "BACKEND_ENGINEER",
      "tier": "core",
      "display_name": "{filename.replace('.txt', '').replace('_', ' ').title()}"
    }}"""
        for filename in sources
    )
    settings.manifest_path.write_text(
        "{\n  \"sources\": [\n" + manifest_entries + "\n  ]\n}",
        encoding="utf-8",
    )

    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")

    question = client.post("/api/interview/start", json={"session_id": session_id}).json()["question"]
    question_texts = [question["question_text"]]
    for _ in range(4):
        answer_response = client.post(
            "/api/interview/answer",
            json={
                "session_id": session_id,
                "question_id": question["question_id"],
                "answer_text": (
                    "I would validate inputs, persist metadata, discuss tradeoffs, add tests, "
                    "and use source traceability for debugging."
                ),
            },
        )
        body = answer_response.json()
        question = body["next_question"]
        if question is None:
            break
        question_texts.append(question["question_text"])

    assert len(question_texts) == 5
    assert len(set(question_texts)) == 5


def test_answer_submission_uses_beginner_adaptation_for_weak_answer(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "Asha Rao\nPython\nLearning backend fundamentals through coursework and practice.",
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    foundation = settings.knowledge_dir / "backend_foundation.txt"
    advanced = settings.knowledge_dir / "backend_advanced.txt"
    foundation.write_text(
        " ".join(["Foundation backend APIs validate inputs and persist data safely."] * 140),
        encoding="utf-8",
    )
    advanced.write_text(
        " ".join(["Advanced backend systems use distributed locks and consensus protocols."] * 140),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "backend_foundation.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "foundation",
      "display_name": "Backend Foundation"
    },
    {
      "filename": "backend_advanced.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "advanced",
      "display_name": "Backend Advanced"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")
    question = client.post("/api/interview/start", json={"session_id": session_id}).json()["question"]

    answer_response = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": question["question_id"],
            "answer_text": "I am not sure.",
        },
    )

    assert answer_response.status_code == 200
    body = answer_response.json()
    assert body["evaluation"]["score"] < 5
    assert body["evaluation"]["missing_points"]
    assert body["adaptation_decision"]["next_difficulty"] == "beginner"
    assert body["adaptation_decision"]["advanced_sources_unlocked"] is False
    assert body["next_question"]["difficulty"] == "beginner"
    assert body["next_question"]["source_tier_used"] == "resume_only"
    assert body["next_question"]["rag_trace"] == []


def test_interview_answer_report_flow(monkeypatch, tmp_path: Path):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "Asha Rao\nPython FastAPI RAG\nBuilt APIs for document ingestion and answer scoring.",
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    knowledge = settings.knowledge_dir / "pytest_interview_backend.txt"
    knowledge.write_text(
        " ".join(
            ["Backend engineers should discuss request validation, persistence, and error handling."] * 100
        ),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "pytest_interview_backend.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "foundation",
      "display_name": "Pytest Interview Backend"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]

    client.post("/api/knowledge/ingest?force=true")

    question_response = client.post(
        "/api/interview/start",
        json={"session_id": session_id},
    )
    assert question_response.status_code == 200
    question = question_response.json()["question"]
    assert question["question_id"]
    assert question["rag_trace"] == []

    answer_response = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": question["question_id"],
            "answer_text": "I would validate the file, store metadata, and return clear safe errors.",
        },
    )
    assert answer_response.status_code == 200
    assert answer_response.json()["evaluation"]["score"] > 0
    assert "adaptation_decision" in answer_response.json()
    assert "next_question" in answer_response.json()

    report_response = client.post("/api/report/generate", json={"session_id": session_id})
    assert report_response.status_code == 200
    assert report_response.json()["report"]["recommendation"]

    session_response = client.get(f"/api/session/{session_id}")
    assert session_response.status_code == 200
    session = session_response.json()["session"]
    assert session["questions"]
    assert session["report"] is not None


def test_report_generate_returns_recruiter_ready_final_report(monkeypatch, tmp_path: Path):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "\n".join(
            [
                "Asha Rao",
                "Python FastAPI ChromaDB RAG SQLAlchemy Docker",
                "Built APIs for document ingestion, source trace retrieval, validation, persistence, and evaluation.",
                "Implemented production style testing and error handling.",
            ]
        ),
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    core = settings.knowledge_dir / "backend_core.txt"
    advanced = settings.knowledge_dir / "backend_advanced.txt"
    core.write_text(
        " ".join(["Core backend systems validate requests, persist metadata, and expose safe API errors."] * 140),
        encoding="utf-8",
    )
    advanced.write_text(
        " ".join(["Advanced backend systems coordinate concurrent ingestion with idempotency and failure recovery."] * 140),
        encoding="utf-8",
    )
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {
      "filename": "backend_core.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "core",
      "display_name": "Backend Core"
    },
    {
      "filename": "backend_advanced.txt",
      "role": "BACKEND_ENGINEER",
      "tier": "advanced",
      "display_name": "Backend Advanced"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "BACKEND_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")
    first_question = client.post("/api/interview/start", json={"session_id": session_id}).json()["question"]
    first_answer = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": first_question["question_id"],
            "answer_text": (
                "I would use Python FastAPI and RAG experience to validate uploads, persist metadata, "
                "store source trace records, test error handling, and design idempotent retries with "
                "clear recovery tradeoffs and validation metrics."
            ),
        },
    ).json()
    second_question = first_answer["next_question"]
    second_answer = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": second_question["question_id"],
            "answer_text": (
                "For advanced ingestion I would handle concurrency with idempotency keys, transaction boundaries, "
                "failure recovery, source metadata validation, and tests for duplicate chunk prevention, while "
                "showing why the selected design is safer under retries."
            ),
        },
    ).json()
    third_question = second_answer["next_question"]
    client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": third_question["question_id"],
            "answer_text": (
                "I would explain the tradeoffs, validate source metadata, persist evaluation records, "
                "and use source traces to show why the system selected each question."
            ),
        },
    )

    report_response = client.post("/api/report/generate", json={"session_id": session_id})

    assert report_response.status_code == 200
    report = report_response.json()["report"]
    assert 0 <= report["overall_score"] <= 10
    assert report["role_fit"] in {"Low", "Moderate", "Strong", "Excellent"}
    assert report["recommendation"] in {"Needs Review", "Proceed", "Strong Proceed"}
    assert report["candidate_summary"]
    assert report["technical_strengths"]
    assert isinstance(report["areas_for_improvement"], list)
    assert report["topic_breakdown"]
    assert len(report["question_answer_summary"]) == 3
    source_usage = report["source_usage_summary"]
    assert "foundation" in source_usage["chunks_by_tier"]
    assert "core" in source_usage["chunks_by_tier"]
    assert "applied" in source_usage["chunks_by_tier"]
    assert "advanced" in source_usage["chunks_by_tier"]
    assert source_usage["source_books_used"]
    assert source_usage["advanced_candidate_evaluation_triggered"] is True
    assert "ready_for_advanced_theory" in report["advanced_readiness"]
    assert report["suggested_next_round_questions"]
    assert report["question_answer_summary"][0]["answer_text"]


def test_fallback_provider_generates_readable_question_without_raw_chunk_dump():
    provider = FallbackProvider()
    profile = ResumeProfile(
        candidate_name="Asha Rao",
        skills=["Python", "FastAPI", "RAG"],
        programming_languages=["Python"],
        frameworks=["FastAPI"],
        tools=["ChromaDB"],
        projects=["Built a resume screening API with vector retrieval."],
        domains=["Backend", "RAG"],
        seniority_level="intermediate",
        suggested_topics=["RAG", "API design"],
        summary="Candidate has relevant backend and retrieval experience.",
    )
    traces = [
        SourceTrace(
            chunk_id="chunk-1",
            source_filename="backend_core.txt",
            display_name="Backend Core Notes",
            page_number=4,
            tier="core",
            chunk_index=0,
            score=0.9,
            text=(
                "14.3 Boosting E = alpha beta gamma delta. "
                "Reliable backend ingestion should validate uploads, persist source metadata, "
                "and make retries idempotent so failures do not duplicate work. "
                "This keeps retrieval pipelines easier to debug and operate."
            ),
        )
    ]

    result = provider.generate_question(
        role="BACKEND_ENGINEER",
        profile=profile,
        traces=traces,
        difficulty="intermediate",
    )

    assert result["question_text"].endswith("?")
    assert "E = alpha beta gamma" not in result["question_text"]
    assert "validating uploads" in result["question_text"].lower() or "upload validation" in result["question_text"].lower()


def test_fallback_provider_generalizes_questions_from_different_traces_in_same_book():
    provider = FallbackProvider()
    profile = ResumeProfile(
        candidate_name="Asha Rao",
        skills=["Python", "Machine Learning"],
        programming_languages=["Python"],
        frameworks=[],
        tools=["Scikit-learn"],
        projects=["Built ML training pipelines and model evaluation workflows."],
        domains=["Machine Learning"],
        seniority_level="advanced",
        suggested_topics=["Machine Learning"],
        summary="Candidate has ML systems experience.",
    )
    trace_one = SourceTrace(
        chunk_id="chunk-1",
        source_filename="bishop_pattern_recognition_machine_learning.pdf",
        display_name="Pattern Recognition and Machine Learning",
        page_number=101,
        tier="advanced",
        chunk_index=0,
        score=0.9,
        text=(
            "Reliable mixture model training should handle missing data carefully and validate convergence "
            "with stable likelihood monitoring and held-out checks."
        ),
    )
    trace_two = SourceTrace(
        chunk_id="chunk-2",
        source_filename="bishop_pattern_recognition_machine_learning.pdf",
        display_name="Pattern Recognition and Machine Learning",
        page_number=202,
        tier="advanced",
        chunk_index=1,
        score=0.88,
        text=(
            "A boosting workflow should compare weak learners, manage bias variance tradeoffs, "
            "and validate generalization through cross-validation."
        ),
    )

    first = provider.generate_question(
        role="AI_ML_ENGINEER",
        profile=profile,
        traces=[trace_one],
        difficulty="advanced",
    )
    second = provider.generate_question(
        role="AI_ML_ENGINEER",
        profile=profile,
        traces=[trace_two],
        difficulty="advanced",
    )

    assert first["question_text"] != second["question_text"]
    assert "missing data" in first["question_text"].lower()
    assert (
        "bias-variance" in second["question_text"].lower()
        or "generalization" in second["question_text"].lower()
        or "boosting" in second["question_text"].lower()
    )


def test_retrieve_filters_bad_index_and_figure_chunks(monkeypatch):
    class FakeCollection:
        def count(self) -> int:
            return 3

        def query(self, query_texts: list[str], n_results: int) -> dict:
            return {
                "documents": [[
                    "INDEX supervised learning, unsupervised learning, reinforcement learning, 12, 18, 26, 41, 53.",
                    "Figure 8.8 Object tracking pipeline. Figure 8.9 Emission probability. Figure 8.10 Hidden state.",
                    (
                        "A reliable model validation workflow should separate training and validation data, "
                        "monitor failure cases, and compare candidate approaches with clear metrics."
                    ),
                ]],
                "metadatas": [[
                    {"source_filename": "bad_index.pdf", "display_name": "Bad Index", "tier": "core", "page_number": 12, "chunk_index": 0},
                    {"source_filename": "figures.pdf", "display_name": "Figure Captions", "tier": "core", "page_number": 88, "chunk_index": 1},
                    {"source_filename": "good_source.pdf", "display_name": "Good Source", "tier": "core", "page_number": 42, "chunk_index": 2},
                ]],
                "ids": [["bad-1", "bad-2", "good-1"]],
                "distances": [[0.1, 0.2, 0.3]],
            }

    monkeypatch.setattr("app.rag.retriever.get_collection", lambda role: FakeCollection())

    traces = retrieve(
        role="AI_ML_ENGINEER",
        query="AI_ML_ENGINEER model validation resume-guided baseline",
        top_k=3,
        candidate_seniority="intermediate",
        question_difficulty="intermediate",
    )

    assert len(traces) == 1
    assert traces[0].source_filename == "good_source.pdf"


def test_extract_concept_from_chunks_returns_clean_summary_and_keywords():
    traces = [
        SourceTrace(
            chunk_id="chunk-1",
            source_filename="ml_core.pdf",
            display_name="ML Core",
            page_number=31,
            tier="core",
            chunk_index=0,
            score=0.91,
            text=(
                "A production ML pipeline should handle missing data before training, compare baseline and "
                "ensemble approaches, and validate performance with held-out metrics and failure-case review."
            ),
        ),
        SourceTrace(
            chunk_id="chunk-2",
            source_filename="ml_core.pdf",
            display_name="ML Core",
            page_number=33,
            tier="core",
            chunk_index=1,
            score=0.89,
            text=(
                "Feature engineering and evaluation should stay consistent between offline experiments and "
                "deployment so drift and leakage are easier to detect."
            ),
        ),
    ]

    concept = extract_concept_from_chunks(traces)

    assert concept["concept"]
    assert "page" not in concept["clean_context_summary"].lower()
    assert "figure" not in concept["clean_context_summary"].lower()
    assert "missing data" in concept["clean_context_summary"].lower()
    assert concept["usable_keywords"]
    assert concept["source_tier"] == "core"


def test_validate_generated_question_rejects_raw_figure_and_repeated_opening():
    passed, reason = validate_generated_question(
        question_text=(
            "Based on your resume, explain Figure 8.8 on page 366 and compare model 4.2 with 8.8 in detail."
        ),
        previous_questions=[
            {
                "question_text": "Based on your resume, how would you validate a retrieval pipeline?",
                "topic": "RAG",
                "question_type": "resume_grounding",
            }
        ],
    )

    assert passed is False
    assert reason


def test_fallback_provider_varies_question_style_and_expected_points_across_turns():
    provider = FallbackProvider()
    profile = ResumeProfile(
        candidate_name="Asha Rao",
        skills=["Python", "Computer Vision", "Machine Learning"],
        programming_languages=["Python"],
        frameworks=["PyTorch"],
        tools=["OpenCV"],
        projects=["Built an object tracking pipeline with Kalman filtering and evaluation workflows."],
        domains=["Machine Learning", "Computer Vision"],
        seniority_level="advanced",
        suggested_topics=["Model validation", "Tracking systems"],
        summary="Candidate has CV and ML systems experience.",
    )
    first_trace = SourceTrace(
        chunk_id="chunk-1",
        source_filename="ml_core.pdf",
        display_name="ML Core",
        page_number=22,
        tier="core",
        chunk_index=0,
        score=0.9,
        text=(
            "A robust validation workflow should compare models against a baseline, inspect failure cases, "
            "and measure performance with task-specific metrics."
        ),
    )
    second_trace = SourceTrace(
        chunk_id="chunk-2",
        source_filename="ml_advanced.pdf",
        display_name="ML Advanced",
        page_number=88,
        tier="advanced",
        chunk_index=1,
        score=0.87,
        text=(
            "Probabilistic graphical models can support tracking by representing latent state, uncertainty, "
            "and observation updates over time."
        ),
    )

    first = provider.generate_question(
        role="AI_ML_ENGINEER",
        profile=profile,
        traces=[first_trace],
        difficulty="intermediate",
        previous_questions=[],
    )
    second = provider.generate_question(
        role="AI_ML_ENGINEER",
        profile=profile,
        traces=[second_trace],
        difficulty="advanced",
        previous_questions=[
            {
                "question_text": first["question_text"],
                "topic": first["topic"],
                "question_type": first["question_type"],
            }
        ],
    )

    assert first["question_type"] != second["question_type"]
    assert first["question_text"].split(":")[0] != second["question_text"].split(":")[0]
    assert first["expected_points"] != second["expected_points"]
    assert "figure" not in second["question_text"].lower()


def test_five_question_flow_avoids_repeated_opening_and_noisy_text(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "\n".join(
            [
                "Asha Rao",
                "Python PyTorch OpenCV SQL",
                "Built object tracking, model evaluation, and data validation workflows.",
            ]
        ),
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    sources = {
        "noise_index.txt": "INDEX classification, segmentation, detection, 18, 20, 22, 25, 30.",
        "noise_figures.txt": "Figure 8.8 Tracking model. Figure 8.9 Observation graph. Figure 8.10 Posterior update.",
        "ml_validation.txt": "A reliable validation workflow should compare baselines, inspect failure cases, and prevent leakage.",
        "ml_tracking.txt": "Object tracking systems should balance state estimation, sensor uncertainty, and real-time constraints.",
        "ml_missing_data.txt": "Training pipelines should handle missing data explicitly and validate downstream performance impact.",
        "ml_comparison.txt": "Engineers should compare deterministic rules and probabilistic models using error analysis and operational metrics.",
        "ml_debugging.txt": "Debugging model regressions should start with data drift checks, feature audits, and metric breakdowns.",
    }
    for filename, content in sources.items():
        (settings.knowledge_dir / filename).write_text(content, encoding="utf-8")

    manifest_entries = ",\n".join(
        f"""{{
      "filename": "{filename}",
      "role": "AI_ML_ENGINEER",
      "tier": "advanced" if "tracking" in "{filename}" else "core",
      "display_name": "{filename.replace('.txt', '').replace('_', ' ').title()}"
    }}"""
        for filename in sources
    )
    manifest_entries = manifest_entries.replace('"advanced" if "tracking" in "', '"core"')
    manifest_entries = manifest_entries.replace('tracking.txt}" else "core"', 'tracking.txt}"')
    settings.manifest_path.write_text(
        "{\n  \"sources\": [\n"
        '    {"filename":"noise_index.txt","role":"AI_ML_ENGINEER","tier":"core","display_name":"Noise Index"},\n'
        '    {"filename":"noise_figures.txt","role":"AI_ML_ENGINEER","tier":"advanced","display_name":"Noise Figures"},\n'
        '    {"filename":"ml_validation.txt","role":"AI_ML_ENGINEER","tier":"core","display_name":"Ml Validation"},\n'
        '    {"filename":"ml_tracking.txt","role":"AI_ML_ENGINEER","tier":"advanced","display_name":"Ml Tracking"},\n'
        '    {"filename":"ml_missing_data.txt","role":"AI_ML_ENGINEER","tier":"core","display_name":"Ml Missing Data"},\n'
        '    {"filename":"ml_comparison.txt","role":"AI_ML_ENGINEER","tier":"core","display_name":"Ml Comparison"},\n'
        '    {"filename":"ml_debugging.txt","role":"AI_ML_ENGINEER","tier":"core","display_name":"Ml Debugging"}\n'
        "  ]\n}",
        encoding="utf-8",
    )

    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "AI_ML_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")

    question = client.post("/api/interview/start", json={"session_id": session_id}).json()["question"]
    questions = [question]
    for _ in range(4):
        answer_response = client.post(
            "/api/interview/answer",
            json={
                "session_id": session_id,
                "question_id": question["question_id"],
                "answer_text": (
                    "I would explain tradeoffs, validate metrics, inspect failure modes, and connect the design "
                    "to tracking, data quality, and model evaluation."
                ),
            },
        ).json()
        question = answer_response["next_question"]
        if question is None:
            break
        questions.append(question)

    openings = []
    for question in questions:
        text = question["question_text"]
        openings.append(text.split(",")[0].split(":")[0])
        assert "index" not in text.lower()
        assert "figure" not in text.lower()
        assert sum(char.isdigit() for char in text) <= 3
        assert len([segment for segment in text.replace("?", ".").split(".") if segment.strip()]) <= 3

    assert len(questions) == 5
    assert len(set(openings)) >= 3


def test_retrieve_uses_lexical_fallback_when_semantic_results_are_all_bad(monkeypatch):
    class FakeCollection:
        def count(self) -> int:
            return 4

        def query(self, query_texts: list[str], n_results: int) -> dict:
            return {
                "documents": [[
                    "INDEX supervised learning, unsupervised learning, 12, 18, 26, 41, 53.",
                    "Figure 8.8 Tracking graph. Figure 8.9 Posterior update. Figure 8.10 Hidden state.",
                ]],
                "metadatas": [[
                    {"source_filename": "bad_index.pdf", "display_name": "Bad Index", "tier": "core", "page_number": 12, "chunk_index": 0},
                    {"source_filename": "bad_figure.pdf", "display_name": "Bad Figure", "tier": "advanced", "page_number": 88, "chunk_index": 1},
                ]],
                "ids": [["bad-1", "bad-2"]],
                "distances": [[0.1, 0.2]],
            }

        def peek(self, limit: int) -> dict:
            return {
                "documents": [
                    "A strong ML workflow should use cross-validation, regularization, and clear generalization checks.",
                    "Probabilistic tracking systems should model uncertainty and update state estimates over time.",
                ],
                "metadatas": [
                    {"source_filename": "good_core.pdf", "display_name": "Good Core", "tier": "core", "page_number": 20, "chunk_index": 0},
                    {"source_filename": "good_advanced.pdf", "display_name": "Good Advanced", "tier": "advanced", "page_number": 95, "chunk_index": 1},
                ],
                "ids": ["good-1", "good-2"],
            }

    monkeypatch.setattr("app.rag.retriever.get_collection", lambda role: FakeCollection())

    traces = retrieve(
        role="AI_ML_ENGINEER",
        query="AI_ML_ENGINEER cross-validation generalization uncertainty tracking",
        top_k=2,
        candidate_seniority="advanced",
        question_difficulty="advanced",
        allow_advanced_sources=True,
    )

    assert len(traces) == 2
    assert traces[0].source_filename in {"good_core.pdf", "good_advanced.pdf"}
    assert any("generalization" in trace.text.lower() or "uncertainty" in trace.text.lower() for trace in traces)


def test_ai_ml_resume_only_questions_are_distinct_and_question_three_switches_to_kb(
    monkeypatch, tmp_path: Path
):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "\n".join(
            [
                "Aarav Mehta",
                "Python PyTorch OpenCV scikit-learn FastAPI ChromaDB",
                "Built model validation workflows, object tracking systems, and deployment-oriented evaluation pipelines.",
                "Worked on drift checks, confusion-matrix analysis, and production ML tradeoffs.",
            ]
        ),
        encoding="utf-8",
    )
    settings = use_temp_knowledge_base(monkeypatch, tmp_path)
    sources = {
        "ml_cv.txt": "Computer vision systems should balance latency, failure analysis, and robust validation before rollout.",
        "ml_generalization.txt": "Model selection should use cross-validation, regularization, and generalization checks instead of only training accuracy.",
        "ml_uncertainty.txt": "Probabilistic tracking pipelines should represent uncertainty, hidden state, and observation updates over time.",
        "ml_missing_data.txt": "Training workflows should handle missing data explicitly and measure the impact on downstream model behavior.",
    }
    for filename, content in sources.items():
        (settings.knowledge_dir / filename).write_text((" ".join([content] * 80)), encoding="utf-8")
    settings.manifest_path.write_text(
        """
{
  "sources": [
    {"filename":"ml_cv.txt","role":"AI_ML_ENGINEER","tier":"core","display_name":"ML CV"},
    {"filename":"ml_generalization.txt","role":"AI_ML_ENGINEER","tier":"core","display_name":"ML Generalization"},
    {"filename":"ml_uncertainty.txt","role":"AI_ML_ENGINEER","tier":"advanced","display_name":"ML Uncertainty"},
    {"filename":"ml_missing_data.txt","role":"AI_ML_ENGINEER","tier":"core","display_name":"ML Missing Data"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    with resume.open("rb") as file:
        resume_response = client.post(
            "/api/resume/upload",
            data={"selected_role": "AI_ML_ENGINEER"},
            files={"file": ("resume.txt", file, "text/plain")},
        )
    session_id = resume_response.json()["session_id"]
    client.post("/api/knowledge/ingest?force=true")

    q1 = client.post("/api/interview/start", json={"session_id": session_id}).json()["question"]
    a1 = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": q1["question_id"],
                "answer_text": (
                    "I would start with a baseline, validate the split strategy, compare metrics, inspect failure cases, "
                    "review latency and robustness tradeoffs, and confirm that the pipeline behaves reliably before rollout."
                ),
            },
        ).json()
    q2 = a1["next_question"]
    a2 = client.post(
        "/api/interview/answer",
        json={
            "session_id": session_id,
            "question_id": q2["question_id"],
                "answer_text": (
                    "I would debug data quality and drift first, compare alternatives with validation metrics, inspect "
                    "failure slices, review confusion patterns, and use tradeoff analysis around robustness, latency, "
                    "and operational cost before shipping the pipeline."
                ),
            },
        ).json()
    q3 = a2["next_question"]

    assert q1["source_tier_used"] == "resume_only"
    assert q2["source_tier_used"] == "resume_only"
    assert q3["source_tier_used"] in {"foundation", "core", "applied", "advanced"}
    assert q3["rag_trace"]
    assert "Your resume highlights" not in q1["question_text"]
    assert "Your resume highlights" not in q2["question_text"]
    assert q1["question_text"] != q2["question_text"]
    assert q2["question_text"] != q3["question_text"]
    assert len({q1["topic"], q2["topic"], q3["topic"]}) == 3

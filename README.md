# SkillScreen AI

Project title: SkillScreen AI

SkillScreen AI is a resume-aware, role-based, RAG-powered adaptive technical interview simulator. It lets a reviewer upload a candidate resume, select a role, generate source-grounded interview questions, evaluate answers, adapt question difficulty, and produce a final recruiter-ready screening report.

This repository currently targets a one-day MVP demo. It is intentionally local-first and lightweight, while preserving the core architecture decisions needed for a stronger production version.

## Problem Statement

Technical interviews often use generic questions that ignore the candidate's resume and give little evidence for why a question was asked. SkillScreen AI solves this by combining resume signals with role-specific knowledge retrieval. Each question is tied to retrieved source chunks, source tiers, and a trace that an interviewer can inspect.

The goal is to demonstrate AI/ML engineering, backend design, RAG traceability, adaptive evaluation, and a usable end-to-end product flow.

## Features

- Resume upload for PDF and TXT files.
- Rule-based resume profile extraction for skills, projects, seniority, domains, and suggested topics.
- Role selection for:
  - `AI_ML_ENGINEER`
  - `BACKEND_ENGINEER`
  - `DATA_SCIENCE_APPLIED_ML`
- Manifest-driven knowledge ingestion from local PDF/TXT/MD files.
- ChromaDB vector storage with source metadata.
- RAG retrieval filtered by selected role and source tier.
- Resume-aware question generation.
- RAG trace panel with retrieval query, source filename, display name, page number, tier, and chunk preview.
- Answer evaluation with score, accuracy, clarity, depth, strengths, missing points, feedback, ideal answer summary, and source grounding notes.
- Adaptive next-question generation across a 5-question interview flow for the MVP demo.
- Advanced source escalation for strong candidates.
- Final recruiter/interviewer report.
- Deterministic fallback behavior when LLM keys are unavailable.

## Tech Stack

Backend:
- FastAPI
- SQLite
- SQLAlchemy 2.0
- Pydantic
- PyMuPDF
- ChromaDB
- python-dotenv

Frontend:
- Next.js
- TypeScript
- Tailwind CSS

AI/RAG:
- ChromaDB persistent vector store
- Deterministic embedding fallback
- Optional OpenAI/Gemini provider abstraction
- Rule-based fallback question generation and answer evaluation

## Architecture

```text
Resume PDF/TXT
   -> FastAPI resume endpoint
   -> PyMuPDF/TXT parser
   -> rule-based profile extraction
   -> SQLite session

Knowledge PDFs/TXT/MD
   -> source_manifest.json
   -> PyMuPDF/TXT/MD loading
   -> word chunking with overlap
   -> deterministic embeddings
   -> ChromaDB persistent collections

Interview
   -> load session + profile
   -> build retrieval query from resume and role
   -> retrieve tier-eligible Chroma chunks
   -> generate question
   -> store question + trace in SQLite

Answer
   -> store answer
   -> evaluate against expected points and retrieved context
   -> store evaluation and adaptation decision
   -> generate next adaptive question until 5 questions

Report
   -> aggregate questions, answers, evaluations, RAG traces, tiers, and adaptations
   -> store final report JSON
   -> show recruiter-ready summary
```

## System Flow

1. Ingest knowledge sources from `backend/knowledge_base/raw/`.
2. Upload a resume and select a target role.
3. The backend extracts a structured resume profile.
4. Start an interview session.
5. The backend builds a retrieval query from selected role, skills, projects, suggested topics, and seniority.
6. ChromaDB returns role-filtered source chunks.
7. The question generator creates a resume-aware, source-grounded question.
8. The candidate submits an answer.
9. The evaluator scores the answer and decides the next difficulty.
10. The flow continues until 5 questions.
11. The final report summarizes role fit, strengths, gaps, source usage, and advanced readiness.

## RAG Pipeline

Knowledge ingestion is manifest-driven:

1. Read `backend/knowledge_base/source_manifest.json`.
2. Load matching files from `backend/knowledge_base/raw/`.
3. Parse PDFs with PyMuPDF; parse TXT/MD files as text.
4. Clean extracted text.
5. Chunk text into approximately 700-900 words with 100-150 word overlap.
6. Store chunks in ChromaDB under `backend/storage/chroma`.
7. Store metadata:
   - `source_filename`
   - `display_name`
   - `page_number`
   - `role`
   - `tier`
   - `chunk_index`

If one PDF fails to parse, ingestion logs the error and continues with the remaining sources. Duplicate ingestion is skipped unless `force=true` is passed.

## Knowledge Source Manifest

The source manifest lives at:

```text
backend/knowledge_base/source_manifest.json
```

Each source maps a local filename to a role, source tier, and display name:

```json
{
  "filename": "tom_mitchell_machine_learning.pdf",
  "role": "AI_ML_ENGINEER",
  "tier": "core",
  "display_name": "Machine Learning by Tom Mitchell"
}
```

Raw PDFs are intentionally ignored by git. Place local PDF files in:

```text
backend/knowledge_base/raw/
```

The filename must match the manifest entry.

The repository also includes one built-in backend grounding source:

```text
backend/knowledge_base/raw/backend_engineering_foundations.txt
```

## Source Tiering

SkillScreen AI uses four tiers:

- `foundation`: introductory material, always eligible.
- `core`: role-essential material, always eligible.
- `applied`: preferred for applied ML/data science interviews.
- `advanced`: used only when the candidate shows enough readiness.

Advanced chunks are eligible only when:

- candidate seniority is `intermediate` or `advanced`, or
- requested difficulty is `advanced`, or
- a previous answer score unlocks advanced difficulty.

## Resume-Aware Question Generation

Question generation uses:

- selected role
- resume skills
- projects
- suggested topics
- seniority level
- retrieved RAG chunks
- source tier

The fallback question generator is deterministic and avoids generic questions by explicitly referencing resume signals and retrieved source context. If `OPENAI_API_KEY` or `GEMINI_API_KEY` is configured, the provider path can improve wording while preserving fallback behavior if the call fails.

## Advanced Source Escalation

Answer scores drive adaptive difficulty:

- `score >= 8`: next question is advanced and advanced source tiers are unlocked.
- `score 5-7`: next question is intermediate and uses core/applied material.
- `score < 5`: next question is beginner/probing and prefers foundation/core material.

The adaptation decision is stored with the evaluation JSON.

## RAG Traceability

Every generated question includes a RAG trace:

- retrieval query
- source filename
- display name
- page number
- source tier
- chunk preview
- why the question was asked

This makes the demo explainable: the interviewer can see exactly why a question was generated and where the source grounding came from.

## Answer Evaluation

The MVP evaluator scores answers out of 10 using deterministic signals:

- keyword overlap with expected points
- answer length and clarity
- relevance to the question topic
- overlap with retrieved source context
- source tier used

The response includes:

- score
- technical accuracy
- clarity
- depth
- strengths
- missing points
- feedback
- ideal answer summary
- source grounding notes

## Final Report

The final report aggregates:

- all questions
- answers
- evaluations
- RAG traces
- source tiers used
- adaptation decisions

It returns and stores:

- overall score
- role fit: Low, Moderate, Strong, Excellent
- recommendation: Needs Review, Proceed, Strong Proceed
- candidate summary
- technical strengths
- areas for improvement
- topic breakdown
- question-answer summary
- source usage summary
- advanced readiness
- suggested next-round questions

## Local Setup

### 1. Backend

```powershell
python -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

Backend API docs:

```text
http://127.0.0.1:8000/docs
```

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend app:

```text
http://127.0.0.1:3000
```

Optional frontend environment:

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

### 3. Run Tests and Build

Backend tests:

```powershell
python -m pytest backend\tests\test_mvp_api.py -q
```

Frontend checks:

```powershell
cd frontend
npm run typecheck
npm run build
```

## Knowledge Base PDF Placement

Raw PDFs are not committed. To run ingestion with the included manifest:

1. Put PDFs into:

   ```text
   backend/knowledge_base/raw/
   ```

2. Ensure each filename matches `backend/knowledge_base/source_manifest.json`.

3. Start the backend.

4. Call:

   ```powershell
   Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/knowledge/ingest?force=true"
   ```

5. Check status:

   ```powershell
   Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/knowledge/status"
   ```

## API Overview

- `GET /api/health`: service health check.
- `POST /api/resume/upload`: upload resume and create session.
- `POST /api/knowledge/ingest`: ingest manifest sources into ChromaDB.
- `GET /api/knowledge/status`: show ingested documents and tier counts.
- `POST /api/interview/start`: generate the first or requested-difficulty question.
- `POST /api/interview/answer`: submit answer, evaluate, adapt, and return next question if needed.
- `POST /api/report/generate`: generate and store final report.
- `GET /api/session/{session_id}`: inspect full session state.

## Demo Flow

1. Start backend and frontend.
2. Ingest knowledge base sources.
3. Open `http://127.0.0.1:3000`.
4. Click Start Interview.
5. Upload a PDF or TXT resume.
6. Select role.
7. Review extracted profile, seniority, and suggested topics.
8. Start the interview.
9. Inspect generated question and RAG trace.
10. Submit an answer.
11. Review evaluation and adaptation decision.
12. Continue until 5 questions are complete.
13. Generate final report.

## Limitations

- This is a local MVP, not a production deployment.
- SQLite is used instead of PostgreSQL.
- ChromaDB is used instead of pgvector.
- The default embeddings are deterministic and lightweight for demo reliability.
- LLM providers are optional; deterministic fallback is the primary MVP behavior.
- Authentication is not implemented.
- No production Docker setup is included yet.
- The frontend is intentionally compact and demo-focused.
- Raw PDFs must be supplied locally and are not committed.

## Future Improvements

- PostgreSQL + pgvector.
- Alembic migrations.
- Docker and production deployment setup.
- Full automated backend and frontend test suite.
- Authentication and user/session ownership.
- Advanced admin dashboard.
- More robust LLM structured-output validation.
- Better embedding model support.
- Expanded analytics for candidate trends and topic coverage.
- Interview blueprint preview before question generation.

## Submission Notes

Do not include:

- `.env`
- API keys
- raw private PDFs
- `node_modules`
- `.next`
- generated caches

The `.gitignore` is configured to exclude those artifacts.

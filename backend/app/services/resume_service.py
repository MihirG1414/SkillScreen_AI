from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models import ScreeningSession
from app.schemas import ResumeUploadResponse
from app.services.resume_analyzer import analyze_resume
from app.services.resume_parser import parse_resume_file, save_upload


async def create_session_from_resume_upload(
    db: Session,
    file: UploadFile,
    selected_role: str | None,
) -> ResumeUploadResponse:
    role = (selected_role or "").strip()
    if not role:
        raise ValueError("selected_role is required.")

    path = await save_upload(file)
    resume_text = parse_resume_file(path)
    profile = analyze_resume(resume_text)
    session = ScreeningSession(
        role=role,
        resume_filename=file.filename,
        resume_text=resume_text,
        profile_json=profile.model_dump_json(),
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return ResumeUploadResponse(
        session_id=session.id,
        selected_role=role,
        extracted_profile=profile,
        text_length=len(resume_text),
    )

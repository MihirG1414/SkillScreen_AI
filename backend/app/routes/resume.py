from fastapi import APIRouter, Depends, Form, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import ResumeUploadResponse
from app.services.resume_service import create_session_from_resume_upload


router = APIRouter(prefix="/api/resume", tags=["resume"])


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile,
    selected_role: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> ResumeUploadResponse:
    return await create_session_from_resume_upload(
        db=db,
        file=file,
        selected_role=selected_role,
    )

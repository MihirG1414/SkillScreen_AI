from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import ReportGenerateRequest, ReportGenerateResponse, SessionResponse
from app.services.report_service import generate_report, get_session_snapshot


router = APIRouter(prefix="/api", tags=["report"])


@router.post("/report/generate", response_model=ReportGenerateResponse)
def report(
    request: ReportGenerateRequest, db: Session = Depends(get_db)
) -> ReportGenerateResponse:
    payload = generate_report(db=db, session_id=request.session_id)
    return ReportGenerateResponse(session_id=request.session_id, report=payload)


@router.get("/session/{session_id}", response_model=SessionResponse)
def session(session_id: str, db: Session = Depends(get_db)) -> SessionResponse:
    return SessionResponse(session=get_session_snapshot(db=db, session_id=session_id))


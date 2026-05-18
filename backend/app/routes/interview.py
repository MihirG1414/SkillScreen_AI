from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    InterviewAnswerRequest,
    InterviewAnswerResponse,
    InterviewStartRequest,
    InterviewStartResponse,
)
from app.services.answer_evaluator import submit_and_evaluate_answer
from app.services.question_service import start_interview


router = APIRouter(prefix="/api/interview", tags=["interview"])


@router.post("/start", response_model=InterviewStartResponse)
def start(request: InterviewStartRequest, db: Session = Depends(get_db)) -> InterviewStartResponse:
    question = start_interview(
        db=db,
        session_id=request.session_id,
        requested_difficulty=request.requested_difficulty,
    )
    return InterviewStartResponse(session_id=request.session_id, question=question)


@router.post("/answer", response_model=InterviewAnswerResponse)
def answer(
    request: InterviewAnswerRequest, db: Session = Depends(get_db)
) -> InterviewAnswerResponse:
    return submit_and_evaluate_answer(
        db=db,
        session_id=request.session_id,
        question_id=request.question_id,
        answer_text=request.answer_text,
    )

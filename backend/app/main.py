from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import init_db
from app.routes import health, interview, knowledge, report, resume


def create_app() -> FastAPI:
    settings = get_settings()
    init_db()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_db()
        yield

    api = FastAPI(title=settings.app_name, lifespan=lifespan)

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    api.include_router(health.router)
    api.include_router(resume.router)
    api.include_router(knowledge.router)
    api.include_router(interview.router)
    api.include_router(report.router)
    return api


app = create_app()

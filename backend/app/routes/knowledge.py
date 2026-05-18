from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import KnowledgeIngestion
from app.rag.ingest import ingest_manifest_sources, knowledge_status
from app.schemas import KnowledgeIngestResponse, KnowledgeStatusResponse


router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/ingest", response_model=KnowledgeIngestResponse)
async def ingest_knowledge(
    force: bool = False,
    db: Session = Depends(get_db),
) -> KnowledgeIngestResponse:
    result = ingest_manifest_sources(force=force)
    if result.chunks_added:
        db.add(
            KnowledgeIngestion(
                role="manifest",
                source_name="source_manifest.json",
                chunks_added=result.chunks_added,
            )
        )
        db.commit()
    return KnowledgeIngestResponse(**result.model_dump())


@router.get("/status", response_model=KnowledgeStatusResponse)
def status() -> KnowledgeStatusResponse:
    return KnowledgeStatusResponse(**knowledge_status())

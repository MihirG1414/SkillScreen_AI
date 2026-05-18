import json
import logging
from pathlib import Path
from typing import Any, Literal

import chromadb
import fitz
from chromadb.api.types import EmbeddingFunction
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag.chunker import chunk_text
from app.services.resume_parser import clean_text


logger = logging.getLogger(__name__)

SupportedRole = Literal["AI_ML_ENGINEER", "BACKEND_ENGINEER", "DATA_SCIENCE_APPLIED_ML"]
SourceTier = Literal["foundation", "core", "applied", "advanced"]


class ManifestSource(BaseModel):
    filename: str
    role: SupportedRole
    tier: SourceTier
    display_name: str


class SourceManifest(BaseModel):
    sources: list[ManifestSource] = Field(default_factory=list)


class ManifestIngestResult(BaseModel):
    documents_ingested: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_added: int = 0
    errors: list[dict[str, str]] = Field(default_factory=list)


class DeterministicEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "skillscreen_deterministic"

    def get_config(self) -> dict:
        return {"dimensions": 64}

    @staticmethod
    def build_from_config(config: dict) -> "DeterministicEmbeddingFunction":
        return DeterministicEmbeddingFunction()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [deterministic_embedding(text) for text in input]


def deterministic_embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for index, char in enumerate(text.lower()):
        vector[(ord(char) + index) % dimensions] += 1.0
    magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / magnitude for value in vector]


def get_client():
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def collection_name(role: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in role).strip("_")
    return f"skillscreen_{safe or 'general'}"


def get_collection(role: str):
    client = get_client()
    return client.get_or_create_collection(
        name=collection_name(role),
        embedding_function=DeterministicEmbeddingFunction(),
        metadata={"role": role},
    )


def default_manifest() -> SourceManifest:
    return SourceManifest(
        sources=[
            ManifestSource(
                filename="machine_learning_for_absolute_beginners.pdf",
                role="AI_ML_ENGINEER",
                tier="foundation",
                display_name="Machine Learning for Absolute Beginners",
            ),
            ManifestSource(
                filename="tom_mitchell_machine_learning.pdf",
                role="AI_ML_ENGINEER",
                tier="core",
                display_name="Machine Learning by Tom Mitchell",
            ),
            ManifestSource(
                filename="intrpduction_to_statistical_learning.pdf",
                role="AI_ML_ENGINEER",
                tier="core",
                display_name="Introduction to Statistical Learning",
            ),
            ManifestSource(
                filename="introduction_to_ml_with_python.pdf",
                role="DATA_SCIENCE_APPLIED_ML",
                tier="applied",
                display_name="Introduction to Machine Learning with Python",
            ),
            ManifestSource(
                filename="bishop_pattern_recognition_machine_learning.pdf",
                role="AI_ML_ENGINEER",
                tier="advanced",
                display_name="Pattern Recognition and Machine Learning",
            ),
            ManifestSource(
                filename="ai_ml_deep_learning.pdf",
                role="AI_ML_ENGINEER",
                tier="advanced",
                display_name="AI, Machine Learning, and Deep Learning",
            ),
            ManifestSource(
                filename="Probabilistic_Machine_Learning_An_Introduction_by_Kevin_Patrick_Murphy.pdf",
                role="AI_ML_ENGINEER",
                tier="advanced",
                display_name="Probabilistic Machine Learning: An Introduction",
            ),
        ]
    )


def ensure_manifest(path: Path | None = None) -> Path:
    settings = get_settings()
    manifest_path = path or settings.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(
            default_manifest().model_dump_json(indent=2),
            encoding="utf-8",
        )
    return manifest_path


def load_manifest(path: Path | None = None) -> SourceManifest:
    manifest_path = ensure_manifest(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return SourceManifest.model_validate(data)


def parse_knowledge_file(path: Path) -> list[tuple[str, int | None]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages: list[tuple[str, int | None]] = []
        try:
            with fitz.open(path) as document:
                for index, page in enumerate(document, start=1):
                    text = clean_text(page.get_text("text"))
                    if text:
                        pages.append((text, index))
        except Exception as exc:
            raise ValueError("Unable to read knowledge PDF.") from exc
        return pages
    if suffix in {".txt", ".md"}:
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        return [(text, None)] if text else []
    raise ValueError("Only PDF, TXT, and MD knowledge files are supported.")


def source_already_ingested(source: ManifestSource) -> bool:
    collection = get_collection(source.role)
    existing = collection.get(where={"source_filename": source.filename}, limit=1)
    return bool(existing.get("ids"))


def ingest_manifest_sources(
    *,
    force: bool = False,
    manifest_path: Path | None = None,
    raw_dir: Path | None = None,
) -> ManifestIngestResult:
    settings = get_settings()
    manifest = load_manifest(manifest_path)
    base_dir = raw_dir or settings.knowledge_dir
    result = ManifestIngestResult()

    for source in manifest.sources:
        path = base_dir / source.filename
        if not path.exists():
            result.documents_failed += 1
            result.errors.append(
                {"source_filename": source.filename, "error": "Source file not found."}
            )
            continue
        if not force and source_already_ingested(source):
            result.documents_skipped += 1
            continue

        try:
            chunks_added = ingest_knowledge_source(path=path, source=source)
        except Exception as exc:
            logger.warning("Knowledge ingestion failed for %s: %s", source.filename, exc)
            result.documents_failed += 1
            result.errors.append({"source_filename": source.filename, "error": str(exc)})
            continue

        result.documents_ingested += 1
        result.chunks_added += chunks_added

    return result


def ingest_knowledge_source(path: Path, source: ManifestSource) -> int:
    collection = get_collection(source.role)
    pages = parse_knowledge_file(path)
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for page_text, page_number in pages:
        for chunk_index, chunk in enumerate(chunk_text(page_text)):
            page = page_number or 0
            chunk_id = (
                f"{collection_name(source.role)}_{source.filename}_{page}_{chunk_index}"
            ).replace(" ", "_")
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "source_filename": source.filename,
                    "source_name": source.filename,
                    "display_name": source.display_name,
                    "page_number": page,
                    "role": source.role,
                    "tier": source.tier,
                    "chunk_index": chunk_index,
                }
            )

    if not documents:
        raise ValueError("Knowledge file did not contain readable text.")

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)


def knowledge_status() -> dict[str, Any]:
    client = get_client()
    chunks_per_role: dict[str, int] = {}
    chunks_per_tier: dict[str, int] = {}
    source_filenames: set[str] = set()
    advanced_sources_available = False

    for collection in client.list_collections():
        count = collection.count()
        if count == 0:
            continue
        peek = collection.peek(limit=count)
        for metadata in peek.get("metadatas", []):
            if not metadata:
                continue
            role = str(metadata.get("role") or collection.metadata.get("role") or collection.name)
            tier = str(metadata.get("tier") or "unknown")
            filename = str(metadata.get("source_filename") or metadata.get("source_name") or "")
            chunks_per_role[role] = chunks_per_role.get(role, 0) + 1
            chunks_per_tier[tier] = chunks_per_tier.get(tier, 0) + 1
            if filename:
                source_filenames.add(filename)
            if tier == "advanced":
                advanced_sources_available = True

    return {
        "documents_ingested": len(source_filenames),
        "chunks_per_role": dict(sorted(chunks_per_role.items())),
        "chunks_per_tier": dict(sorted(chunks_per_tier.items())),
        "source_filenames": sorted(source_filenames),
        "advanced_sources_available": advanced_sources_available,
    }

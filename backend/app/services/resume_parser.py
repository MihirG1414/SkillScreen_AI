from pathlib import Path

import fitz
from fastapi import UploadFile

from app.config import get_settings


def clean_text(text: str) -> str:
    lines = [" ".join(line.strip().split()) for line in text.replace("\x00", "").splitlines()]
    compact = [line for line in lines if line]
    return "\n".join(compact).strip()


async def save_upload(file: UploadFile) -> Path:
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_upload_extensions:
        raise ValueError("Only PDF and TXT files are supported.")

    target = settings.uploads_dir / (file.filename or f"upload{suffix}")
    content = await file.read()
    if not content:
        raise ValueError("Uploaded file is empty.")
    target.write_bytes(content)
    return target


def parse_resume_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            raise ValueError("Uploaded resume is empty.")
        return text
    if suffix == ".pdf":
        text_parts: list[str] = []
        try:
            with fitz.open(path) as document:
                for page in document:
                    text_parts.append(page.get_text("text"))
        except Exception as exc:
            raise ValueError("Unable to read the uploaded PDF.") from exc
        text = clean_text("\n".join(text_parts))
        if not text:
            raise ValueError("No readable text was found in the PDF.")
        return text
    raise ValueError("Only PDF and TXT files are supported.")

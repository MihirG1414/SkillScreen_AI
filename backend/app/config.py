from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")


class Settings(BaseSettings):
    app_name: str = "SkillScreen AI"
    database_url: str = "sqlite:///./storage/skillscreen.db"
    chroma_path: str = "./storage/chroma"
    upload_dir: str = "./storage/uploads"
    knowledge_raw_dir: str = "./knowledge_base/raw"
    source_manifest_path: str = "./knowledge_base/source_manifest.json"
    llm_provider: str = "fallback"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    allowed_upload_extensions: set[str] = Field(default_factory=lambda: {".pdf", ".txt"})

    model_config = SettingsConfigDict(env_file=str(BACKEND_ROOT / ".env"), extra="ignore")

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return BACKEND_ROOT / path

    @property
    def sqlite_path(self) -> Path | None:
        if not self.database_url.startswith("sqlite:///"):
            return None
        raw_path = self.database_url.replace("sqlite:///", "", 1)
        return self.resolve_path(raw_path)

    @property
    def sqlalchemy_database_url(self) -> str:
        sqlite_path = self.sqlite_path
        if sqlite_path is None:
            return self.database_url
        return f"sqlite:///{sqlite_path.as_posix()}"

    @property
    def chroma_dir(self) -> Path:
        return self.resolve_path(self.chroma_path)

    @property
    def uploads_dir(self) -> Path:
        return self.resolve_path(self.upload_dir)

    @property
    def knowledge_dir(self) -> Path:
        return self.resolve_path(self.knowledge_raw_dir)

    @property
    def manifest_path(self) -> Path:
        return self.resolve_path(self.source_manifest_path)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = settings.sqlite_path
    if sqlite_path is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return settings

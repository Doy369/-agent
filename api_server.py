from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import AgentConfig, NovelAgent
from finetune_exporter import export_openai_chat_jsonl
from knowledge_base import get_knowledge_base, rebuild_knowledge_base
from local_tools import (
    generate_image,
    local_archive_and_stats,
    local_text_audit,
    search_local_knowledge,
)
from langchain_integration import describe_langchain_integration


app = FastAPI(
    title="Novel AI Agent API",
    version="0.1.0",
    description="HTTP API wrapper for the desktop Novel AI Agent core.",
)


class ModelConfigPayload(BaseModel):
    api_key: str = Field(default="", description="LLM API key. It is used only for this request.")
    base_url: str = Field(default="https://api.deepseek.com/v1")
    model: str = Field(default="deepseek-chat")
    provider: Literal["openai", "anthropic"] = "openai"
    max_words: int = Field(default=2500, ge=100, le=20000)
    temperature: float = Field(default=0.82, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=180.0, ge=10.0, le=600.0)


class OutlineRequest(BaseModel):
    config: ModelConfigPayload
    novel_setting: str = ""
    style: str = "玄幻"
    current_text: str = ""


class ChapterRequest(BaseModel):
    config: ModelConfigPayload
    novel_setting: str = ""
    style: str = "玄幻"
    outline: str = ""
    chapters: list[dict[str, str]] = Field(default_factory=list)
    current_text: str = ""


class PolishRequest(BaseModel):
    config: ModelConfigPayload
    selected_text: str
    style: str = "玄幻"
    instruction: str = "增强画面感、节奏和情绪张力"


class SearchRequest(BaseModel):
    query: str


class AuditRequest(BaseModel):
    text: str


class ImageRequest(BaseModel):
    prompt: str


class FinetuneExportRequest(BaseModel):
    output_path: str = ""
    min_chars: int = Field(default=200, ge=1)
    limit: int | None = Field(default=None, ge=1)
    style: str = "中文类型小说"


class GenerationResponse(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextResponse(BaseModel):
    content: str


class ImageResponse(BaseModel):
    content: str
    image_path: str


class FinetuneExportResponse(BaseModel):
    output_path: str
    examples: int


def _to_agent_config(payload: ModelConfigPayload) -> AgentConfig:
    return AgentConfig(
        api_key=payload.api_key,
        base_url=payload.base_url,
        model=payload.model,
        provider=payload.provider,
        max_words=payload.max_words,
        temperature=payload.temperature,
        timeout_seconds=payload.timeout_seconds,
    )


def _new_agent() -> NovelAgent:
    return NovelAgent()


def _raise_http_error(exc: Exception) -> None:
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    kb = get_knowledge_base()
    stats = kb.get_stats()
    return {
        "status": "ok",
        "knowledge_base": {
            "total_chunks": stats["total_chunks"],
            "model": stats["model"],
            "st_available": stats["st_available"],
        },
        "langchain": describe_langchain_integration(),
    }


@app.post("/generate-outline", response_model=GenerationResponse)
def generate_outline(request: OutlineRequest) -> GenerationResponse:
    try:
        result = _new_agent().generate_outline(
            _to_agent_config(request.config),
            novel_setting=request.novel_setting,
            style=request.style,
            current_text=request.current_text,
        )
        return GenerationResponse(content=result.content, metadata=result.metadata)
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/generate-chapter", response_model=GenerationResponse)
def generate_chapter(request: ChapterRequest) -> GenerationResponse:
    try:
        result = _new_agent().generate_next_chapter(
            _to_agent_config(request.config),
            novel_setting=request.novel_setting,
            style=request.style,
            outline=request.outline,
            chapters=request.chapters,
            current_text=request.current_text,
        )
        return GenerationResponse(content=result.content, metadata=result.metadata)
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/polish", response_model=GenerationResponse)
def polish(request: PolishRequest) -> GenerationResponse:
    try:
        result = _new_agent().polish_text(
            _to_agent_config(request.config),
            selected_text=request.selected_text,
            style=request.style,
            instruction=request.instruction,
        )
        return GenerationResponse(content=result.content, metadata=result.metadata)
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/search-knowledge", response_model=TextResponse)
def search_knowledge(request: SearchRequest) -> TextResponse:
    try:
        return TextResponse(content=search_local_knowledge(request.query))
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/audit-text", response_model=TextResponse)
def audit_text(request: AuditRequest) -> TextResponse:
    try:
        return TextResponse(content=local_text_audit(request.text))
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/archive-stats", response_model=TextResponse)
def archive_stats() -> TextResponse:
    try:
        return TextResponse(content=local_archive_and_stats())
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/rebuild-knowledge-index")
def rebuild_index() -> dict[str, Any]:
    try:
        count = rebuild_knowledge_base()
        return {"status": "ok", "total_chunks": count, "stats": get_knowledge_base().get_stats()}
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/generate-image", response_model=ImageResponse)
def generate_chapter_image(request: ImageRequest) -> ImageResponse:
    try:
        image_path = generate_image(request.prompt)
        return ImageResponse(content=f"已生成插图：{image_path}", image_path=image_path)
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/export-finetune-dataset", response_model=FinetuneExportResponse)
def export_finetune_dataset(request: FinetuneExportRequest) -> FinetuneExportResponse:
    try:
        output_path, count = export_openai_chat_jsonl(
            output_path=None if not request.output_path else Path(request.output_path),
            min_chars=request.min_chars,
            limit=request.limit,
            style=request.style,
        )
        return FinetuneExportResponse(output_path=str(output_path), examples=count)
    except Exception as exc:
        _raise_http_error(exc)

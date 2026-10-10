"""Optional local API over a previously ingested pilot-v1 knowledge base."""

import asyncio
from contextlib import asynccontextmanager
import os
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from reproduce import (ROOT, build_runtime, check_kb_manifest, kb_lock, load_settings,
                       now, run_query, save_json, slug, validate_parsed)


@asynccontextmanager
async def lifespan(app):
    settings = load_settings()
    kb = slug(os.getenv("PILOT_KB", "pilot-v1"))
    with kb_lock(ROOT / "storage" / kb):
        manifest = check_kb_manifest(ROOT / "storage" / kb, settings)
        for document in manifest["documents"]:
            validate_parsed(ROOT / "parsed" / document["doc_id"])
        runtime = await build_runtime(settings, ROOT / "storage" / kb)
        app.state.kb = kb
        app.state.manifest = manifest
        app.state.runtime = runtime
        app.state.lock = asyncio.Lock()
        try:
            yield
        finally:
            await runtime.rag.finalize_storages()


app = FastAPI(title="RAG-Anything pilot", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["naive", "hybrid", "mix"] = "mix"
    context_only: bool = False
    vlm: bool = False

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value):
        if not value.strip():
            raise ValueError("Question cannot contain only whitespace.")
        return value


@app.get("/health")
async def health():
    return {"status": "ready", "kb": app.state.kb,
            "documents": [d["doc_id"] for d in app.state.manifest["documents"]]}


@app.post("/query")
async def query(body: QueryRequest):
    request_id = uuid.uuid4().hex
    target = ROOT / "runs" / ("api_" + request_id)
    # One process and one in-flight query keep mutable trace state isolated.
    async with app.state.lock:
        save_json(target / "request.json", {"request_id": request_id, "created_at": now(),
                  "kb": app.state.kb, **body.model_dump()})
        try:
            result = await asyncio.wait_for(run_query(
                app.state.runtime, body.question, body.mode,
                context_only=body.context_only, vlm=body.vlm,
            ), timeout=300)
        except asyncio.TimeoutError:
            save_json(target / "error.json", {"error_type": "TimeoutError", "created_at": now(),
                      "events": app.state.runtime.events})
            raise HTTPException(status_code=504, detail={"request_id": request_id, "error": "query_timeout"})
        except Exception as exc:
            save_json(target / "error.json", {"error_type": type(exc).__name__, "created_at": now(),
                      "events": app.state.runtime.events})
            raise HTTPException(status_code=500, detail={"request_id": request_id, "error": type(exc).__name__})
        save_json(target / "result.json", result)
        return {"request_id": request_id, "kb": app.state.kb,
                "mode": body.mode, "answer": result["answer"],
                "status": result["status"],
                "retrieval": result["retrieval"],
                "images_in_generation_adapter": result["images_in_generation_adapter"],
                "reported_usage": result["reported_usage"],
                "elapsed_seconds": result["elapsed_seconds"],
                "diagnostic_retrieval_is_separate_from_generation": result["diagnostic_retrieval_is_separate_from_generation"]}

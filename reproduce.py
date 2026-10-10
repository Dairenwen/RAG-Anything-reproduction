"""Small, inspectable RAG-Anything pilot. No calls occur on import or --help."""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import hashlib
from importlib.metadata import version, distribution, PackageNotFoundError
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "8664e8b318a3ed651fe62dfcfddfb1f1d633d5be"
BGE_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MANIFEST_SCHEMA = 2
SHANGHAI = timezone(timedelta(hours=8))


def now():
    return datetime.now(SHANGHAI).isoformat()


def slug(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Use a short identifier containing letters, numbers, _ or -.")
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents a cancelled write from leaving a truncated manifest.
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def kb_lock(kb_dir):
    """A persistent advisory lock shared by this CLI and the local API (macOS/Linux)."""
    import fcntl
    kb_dir.mkdir(parents=True, exist_ok=True)
    with (kb_dir / ".pilot.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Knowledge base is in use by another pilot process.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_versions():
    result = {}
    for name in ("raganything", "lightrag-hku", "mineru", "pypdfium2", "pdftext", "openai", "python-dotenv"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def verify_dependencies():
    versions = installed_versions()
    if "NOT_INSTALLED" in versions.values() or not shutil.which("mineru"):
        raise ValueError("Activate .venv and install requirements.txt first.")
    if (versions["lightrag-hku"] != "1.4.16" or versions["mineru"] != "3.4.1"
            or versions["pypdfium2"] != "4.30.0" or versions["pdftext"] != "0.6.3"):
        raise ValueError("Expected LightRAG 1.4.16, MinerU 3.4.1, pypdfium2 4.30.0 and pdftext 0.6.3.")
    direct_url = json.loads(distribution("raganything").read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != SOURCE_COMMIT:
        raise ValueError("RAG-Anything source commit differs; reinstall the pinned requirements.")
    return versions


def validate_blocks(blocks):
    if not isinstance(blocks, list) or not blocks or not all(isinstance(b, dict) for b in blocks):
        raise ValueError("Parser output must be a nonempty list of content dictionaries.")
    for block in blocks:
        if block.get("type") not in {"text", "image", "table", "equation"}:
            raise ValueError("Pilot accepts text/image/table/equation blocks; review other types first.")
        if "page_idx" in block and (type(block["page_idx"]) is not int or block["page_idx"] < 0):
            raise ValueError("page_idx must be a nonnegative integer.")
        if block.get("img_path"):
            path = Path(block["img_path"])
            if not path.is_absolute() or not path.is_file():
                raise ValueError("A parsed image path is missing or not absolute.")


def normalize_blocks(blocks):
    """Route MinerU chart crops through the image processor while retaining provenance."""
    normalized = []
    for original in blocks:
        block = dict(original)
        if block.get("type") == "chart":
            block["_pilot_original_type"] = "chart"
            block["type"] = "image"
            block["image_caption"] = block.get("chart_caption", [])
            block["image_footnote"] = block.get("chart_footnote", [])
        normalized.append(block)
    return normalized


def validate_parsed(parsed):
    manifest = load_json(parsed / "manifest.json")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Old parse manifest: use a new doc-id and parse with this starter.")
    if sha256_file(parsed / "content-list.json") != manifest["content_sha256"]:
        raise ValueError("Parsed content changed; use a new doc-id for the revised parse.")
    if sha256_file(parsed / "parser-content-list.json") != manifest["parser_content_sha256"]:
        raise ValueError("Original parser output changed after parsing.")
    if sha256_file(manifest["pdf_path"]) != manifest["pdf_sha256"]:
        raise ValueError("Source PDF changed after parsing.")
    for asset in manifest["assets"]:
        if sha256_file(asset["path"]) != asset["sha256"]:
            raise ValueError("A parsed image changed after parsing.")
    blocks = load_json(parsed / "content-list.json")
    validate_blocks(blocks)
    return manifest, blocks


def load_settings():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    text_key = os.getenv("TEXT_API_KEY", "").strip()
    text_url = os.getenv("TEXT_BASE_URL", "https://api.deepseek.com").strip()
    return {
        "text_key": text_key,
        "text_url": text_url,
        "text_model": os.getenv("TEXT_MODEL", "deepseek-flash"),
        "vision_key": os.getenv("VISION_API_KEY", "").strip() or text_key,
        "vision_url": os.getenv("VISION_BASE_URL", "").strip() or text_url,
        "vision_model": os.getenv("VISION_MODEL", "deepseek-flash"),
        "embedding_key": os.getenv("EMBEDDING_API_KEY", "ollama").strip() or text_key,
        "embedding_url": os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1").strip() or text_url,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "qwen3-embedding:latest"),
        "embedding_dim": int(os.getenv("EMBEDDING_DIM", "4096")),
        "send_dimensions": os.getenv("EMBEDDING_SEND_DIMENSIONS", "false").lower() == "true",
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1200")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
        "top_k": int(os.getenv("TOP_K", "20")),
        "chunk_top_k": int(os.getenv("CHUNK_TOP_K", "10")),
        "max_entity_tokens": int(os.getenv("MAX_ENTITY_TOKENS", "4000")),
        "max_relation_tokens": int(os.getenv("MAX_RELATION_TOKENS", "4000")),
        "max_total_tokens": int(os.getenv("MAX_TOTAL_TOKENS", "12000")),
        "concurrency": int(os.getenv("LLM_CONCURRENCY", "2")),
        "summary_language": os.getenv("SUMMARY_LANGUAGE", "English"),
        "enable_rerank": os.getenv("ENABLE_RERANK", "false").lower() == "true",
        "rerank_model_path": str(Path(os.getenv(
            "RERANK_MODEL_PATH", str(ROOT / "models" / "bge-reranker-v2-m3")
        )).expanduser().resolve()),
        "rerank_device": os.getenv("RERANK_DEVICE", "auto"),
        "rerank_max_length": int(os.getenv("RERANK_MAX_LENGTH", "8192")),
        "rerank_batch_size": int(os.getenv("RERANK_BATCH_SIZE", "1")),
    }


def public_settings(settings):
    return {key: value for key, value in settings.items() if not key.endswith("_key")}


def validate_settings(settings, roles=("text", "vision", "embedding")):
    for role in roles:
        if not settings[f"{role}_key"]:
            raise ValueError(f"Missing {role} API key. Fill the local .env file.")
        if not settings[f"{role}_model"].strip():
            raise ValueError(f"Missing {role} model name.")
    for key in ("embedding_dim", "chunk_size", "top_k", "chunk_top_k", "max_entity_tokens",
                "max_relation_tokens", "max_total_tokens", "concurrency"):
        if settings[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if not 0 <= settings["chunk_overlap"] < settings["chunk_size"]:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and less than CHUNK_SIZE.")


def is_deepseek(base_url):
    return urlparse(base_url).hostname == "api.deepseek.com"


def model_request_kwargs(base_url, kwargs):
    """Adapt official DeepSeek without changing other OpenAI-compatible providers."""
    options = dict(kwargs)
    if is_deepseek(base_url):
        body = dict(options.get("extra_body") or {})
        body["thinking"] = {"type": "disabled"}
        options["extra_body"] = body
        # LightRAG's keyword flag normally asks the SDK for json_schema, which this
        # endpoint does not support. Its caller consumes a JSON string instead.
        if options.pop("keyword_extraction", False):
            options["response_format"] = {"type": "json_object"}
    return options


def validate_keyword_response(result):
    try:
        data = json.loads(result)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Keyword response is not valid JSON.") from exc
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), list)
        or any(not isinstance(item, str) for item in data[key])
        for key in ("high_level_keywords", "low_level_keywords")
    ):
        raise ValueError("Keyword response must contain two arrays of strings.")
    return data


def make_embedding(settings, tracker=None):
    from functools import partial
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc
    return EmbeddingFunc(
        embedding_dim=settings["embedding_dim"], max_token_size=8192,
        send_dimensions=settings["send_dimensions"], model_name=settings["embedding_model"],
        func=partial(openai_embed.func, model=settings["embedding_model"],
                     api_key=settings["embedding_key"], base_url=settings["embedding_url"],
                     token_tracker=tracker, client_configs={"timeout": 90}),
    )


class LocalBGEReranker:
    """Local cross-encoder callback for LightRAG's index/score contract."""

    def __init__(self, settings, events):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.events = events
        self.max_length = settings["rerank_max_length"]
        self.batch_size = settings["rerank_batch_size"]
        if not 16 <= self.max_length <= 8192 or self.batch_size <= 0:
            raise ValueError("RERANK_MAX_LENGTH must be 16..8192; RERANK_BATCH_SIZE must be positive.")
        device = settings["rerank_device"]
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in {"cpu", "mps"}:
            raise ValueError("RERANK_DEVICE must be auto, cpu or mps.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable. Set RERANK_DEVICE=cpu explicitly.")
        path = Path(settings["rerank_model_path"])
        if not (path / "model.safetensors").is_file() or not (path / "config.json").is_file():
            raise ValueError("BGE model is not downloaded. Download BAAI/bge-reranker-v2-m3 to " + str(path))
        metadata = path / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
        downloaded_revision = metadata.read_text().splitlines()[0] if metadata.is_file() else None
        if downloaded_revision is not None and downloaded_revision != BGE_RERANK_REVISION:
            raise ValueError("BGE weights revision differs from the pinned download revision.")
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, dtype=torch.float32,
        ).to(device).eval()
        if self.model.config.num_labels != 1:
            raise ValueError("Expected BGE reranker with one relevance logit per pair.")
        self.device = device
        self.lock = asyncio.Lock()
        self.info = {"model": BGE_RERANK_MODEL, "download_revision": downloaded_revision,
                     "expected_download_revision": BGE_RERANK_REVISION,
                     "model_path": str(path), "device": device, "dtype": "float32",
                     "max_length": self.max_length, "batch_size": self.batch_size,
                     "min_rerank_score": 0.0, "config_sha256": sha256_file(path / "config.json"),
                     "weights_bytes": (path / "model.safetensors").stat().st_size,
                     "torch": version("torch"), "transformers": version("transformers"),
                     "load_seconds": time.perf_counter() - started}

    def score(self, query, documents):
        import torch

        scores, lengths = [], []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start:start + self.batch_size]
            lengths.extend(len(self.tokenizer.encode(query, text, verbose=False)) for text in batch)
            inputs = self.tokenizer(
                [query] * len(batch), batch, padding=True, truncation="only_second",
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                if not torch.isfinite(logits).all().item():
                    raise ValueError("Reranker returned non-finite scores.")
                scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores, lengths

    async def __call__(self, query, documents, top_n=None):
        event = {"kind": "rerank_request", "model": BGE_RERANK_MODEL,
                 "device": self.device, "status": "pending", "candidate_count": len(documents),
                 "top_n": top_n, "max_length": self.max_length}
        self.events.append(event)
        started = time.perf_counter()
        try:
            async with self.lock:
                scores, lengths = await asyncio.to_thread(self.score, query, documents)
            ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            results = [{"index": i, "relevance_score": scores[i]} for i in ranked]
            event.update(status="success", ranked_candidates=[
                row | {"content_sha256": hashlib.sha256(documents[row["index"]].encode()).hexdigest(),
                       "input_tokens": lengths[row["index"]],
                       "truncated": lengths[row["index"]] > self.max_length}
                for row in results
            ])
            return results[:top_n] if top_n else results
        except BaseException as exc:
            event.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            event["elapsed_seconds"] = time.perf_counter() - started


def audit_reranking(settings, events, *, require_call=False):
    """LightRAG catches callback errors; do not accept its silent fallback as a rerank run."""
    calls = [e for e in events if e.get("kind") == "rerank_request"]
    if settings.get("enable_rerank"):
        if any(e.get("status") != "success" for e in calls):
            raise RuntimeError("Reranking failed; refusing to report the fallback as a successful rerank run.")
        if require_call and not calls:
            raise RuntimeError("Reranking was enabled but the callback was not called.")
    return {"rerank_enabled": settings.get("enable_rerank", False), "rerank_calls": len(calls),
            "rerank_elapsed_seconds": sum(e.get("elapsed_seconds", 0) for e in calls)}


def image_count(messages):
    return sum(
        1 for msg in messages if isinstance(msg.get("content"), list)
        for part in msg["content"] if isinstance(part, dict) and part.get("type") == "image_url"
    )


def redact_messages(messages):
    result = json.loads(json.dumps(messages))
    for msg in result:
        if not isinstance(msg.get("content"), list):
            continue
        for part in msg["content"]:
            if part.get("type") == "image_url":
                url = part["image_url"].get("url", "")
                part["image_url"]["url"] = "[image payload omitted]"
                part["image_url"]["payload_sha256"] = hashlib.sha256(url.encode()).hexdigest()
    return result


def index_identity(settings):
    keys = ("text_model", "text_url", "vision_model", "vision_url", "embedding_model",
            "embedding_url", "embedding_dim", "send_dimensions", "chunk_size",
            "chunk_overlap", "summary_language")
    return {key: settings[key] for key in keys}


def check_kb_manifest(kb_dir, settings, *, indexing=False):
    path = kb_dir / "pilot_manifest.json"
    if not path.exists():
        if indexing:
            if kb_dir.exists() and any(p.name != ".pilot.lock" for p in kb_dir.iterdir()):
                raise ValueError("Nonempty storage has no pilot manifest. Use a new --kb name.")
            return None
        raise ValueError("Knowledge base manifest missing. Ingest the document successfully first.")
    manifest = load_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Knowledge base schema or framework commit differs. Use a new --kb name.")
    current = index_identity(settings)
    keys = current if indexing else ("embedding_model", "embedding_url", "embedding_dim", "send_dimensions")
    differences = [key for key in keys if current[key] != manifest["index_identity"][key]]
    if differences:
        raise ValueError("Index configuration differs: " + ", ".join(differences) + ". Use a new --kb name.")
    if manifest.get("status") != "ready":
        raise ValueError("Knowledge base is not ready. Inspect the previous ingestion error.")
    return manifest


class Runtime:
    def __init__(self, rag, tracker, events, settings, reranker_info=None):
        self.rag = rag
        self.tracker = tracker
        self.events = events
        self.settings = settings
        self.reranker_info = reranker_info


async def build_runtime(settings, kb_dir):
    verify_dependencies()
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import TokenTracker
    from raganything import RAGAnything, RAGAnythingConfig

    validate_settings(settings)
    events = []
    tracker = TokenTracker()

    async def text_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("timeout", 90)
        keyword_mode = is_deepseek(settings["text_url"]) and kwargs.get("keyword_extraction", False)
        kwargs = model_request_kwargs(settings["text_url"], kwargs)
        event = {"kind": "text_request", "model": settings["text_model"], "status": "pending",
                       "prompt": prompt, "system_prompt": system_prompt,
                       "history_messages": redact_messages(history_messages or [])}
        events.append(event)
        try:
            result = await openai_complete_if_cache(
                settings["text_model"], prompt, system_prompt=system_prompt,
                history_messages=history_messages or [], api_key=settings["text_key"],
                base_url=settings["text_url"], token_tracker=tracker, **kwargs,
            )
            if keyword_mode:
                validate_keyword_response(result)
        except BaseException as exc:
            event.update(status="failed", error_type=type(exc).__name__)
            raise
        event.update(status="success", response=result)
        return result

    async def vision_func(prompt, system_prompt=None, history_messages=None,
                          image_data=None, messages=None, **kwargs):
        if messages is None:
            if not image_data:
                return await text_func(prompt, system_prompt, history_messages, **kwargs)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages or [])
            messages.append({"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_data}},
            ]})
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("timeout", 90)
        kwargs = model_request_kwargs(settings["vision_url"], kwargs)
        event = {"kind": "vision_request", "model": settings["vision_model"], "status": "pending",
                       "image_count": image_count(messages),
                       "messages": redact_messages(messages)}
        events.append(event)
        try:
            result = await openai_complete_if_cache(
                settings["vision_model"], "", messages=messages,
                api_key=settings["vision_key"], base_url=settings["vision_url"],
                token_tracker=tracker, **kwargs,
            )
        except BaseException as exc:
            event.update(status="failed", error_type=type(exc).__name__)
            raise
        event.update(status="success", response=result)
        return result

    embedding = make_embedding(settings, tracker)
    reranker = LocalBGEReranker(settings, events) if settings.get("enable_rerank") else None

    async def rerank_func(query, documents, top_n=None):
        # LightRAG deep-copies its dataclass configuration. Keep weights in a
        # function closure so that each query does not copy the complete model.
        return await reranker(query=query, documents=documents, top_n=top_n)

    lightrag = LightRAG(
        working_dir=str(kb_dir), llm_model_func=text_func, embedding_func=embedding,
        llm_model_max_async=settings["concurrency"], max_parallel_insert=1,
        chunk_token_size=settings["chunk_size"], chunk_overlap_token_size=settings["chunk_overlap"],
        addon_params={"language": settings["summary_language"]},
        kv_storage="JsonKVStorage", vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage", doc_status_storage="JsonDocStatusStorage",
        workspace="", cosine_better_than_threshold=0.2,
        enable_llm_cache=False, enable_llm_cache_for_entity_extract=True,
        rerank_model_func=rerank_func if reranker else None, min_rerank_score=0.0,
    )
    try:
        await lightrag.initialize_storages()
        await initialize_pipeline_status()
        rag = RAGAnything(
            config=RAGAnythingConfig(working_dir=str(kb_dir), parser="mineru",
                                 parser_output_dir=str(ROOT / "parsed"),
                                 enable_image_processing=True, enable_table_processing=True,
                                 enable_equation_processing=True, use_full_path=True),
        lightrag=lightrag, llm_model_func=text_func,
            vision_model_func=vision_func, embedding_func=embedding,
        )
    except BaseException:
        await lightrag.finalize_storages()
        raise
    return Runtime(rag, tracker, events, settings, reranker.info if reranker else None)


async def audit_ingestion(runtime, doc_id, blocks):
    """Check stored coverage because upstream may skip failed modal descriptions."""
    status = await runtime.rag.get_document_processing_status(doc_id)
    if not status.get("fully_processed"):
        raise ValueError("Underlying document processing status is incomplete.")
    chunks = await runtime.rag.lightrag.text_chunks.get_by_ids(status.get("chunks_list", []))
    if not chunks or any(chunk is None for chunk in chunks):
        raise ValueError("Document chunk references are missing from storage.")
    expected = Counter((b["type"], b.get("page_idx", 0)) for b in blocks if b["type"] != "text")
    observed = Counter((c.get("original_type"), c.get("page_idx", 0))
                       for c in chunks if c.get("is_multimodal"))
    missing = [{"type": t, "page_idx": p, "expected": count, "stored": observed[t, p]}
               for (t, p), count in expected.items() if observed[t, p] < count]
    audit = {"status": status, "expected_modal_items": sum(expected.values()),
             "stored_modal_chunks": sum(observed.values()), "missing_groups": missing,
             "failed_model_calls": [e.get("error_type") for e in runtime.events if e.get("status") == "failed"]}
    return audit


async def persist_runtime(runtime):
    # Explicit persistence callbacks propagate errors; upstream finalization logs some failures.
    names = ("full_docs", "doc_status", "text_chunks", "full_entities", "full_relations",
             "entity_chunks", "relation_chunks", "llm_response_cache", "entities_vdb",
             "relationships_vdb", "chunks_vdb", "chunk_entity_relation_graph")
    stores = [getattr(runtime.rag.lightrag, name) for name in names]
    stores += [s for s in (runtime.rag.parse_cache, runtime.rag.multimodal_status_cache) if s is not None]
    await asyncio.gather(*(s.index_done_callback() for s in stores))


def read_questions(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Question file is empty or contains duplicate ids.")
    for record in records:
        slug(record["id"])
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            raise ValueError("Each question must contain nonempty question text.")
    return records


def query_options(settings):
    return {key: settings[key] for key in (
        "top_k", "chunk_top_k", "max_entity_tokens", "max_relation_tokens", "max_total_tokens"
    )} | {"enable_rerank": settings.get("enable_rerank", False), "response_type": "One Sentence"}


async def run_query(runtime, question, mode, *, context_only=False, vlm=False):
    """Capture structured diagnostic retrieval plus the actual generation prompt."""
    from lightrag import QueryParam
    rag = runtime.rag
    runtime.events.clear()
    runtime.tracker.reset()
    opts = query_options(runtime.settings)
    started = time.perf_counter()
    retrieval = await rag.lightrag.aquery_data(question, param=QueryParam(mode=mode, **opts))
    data = retrieval.get("data") or {}
    rerank_audit = audit_reranking(runtime.settings, runtime.events, require_call=bool(data.get("chunks")))
    if retrieval.get("status") != "success" or not any(data.get(k) for k in ("entities", "relationships", "chunks")):
        return {
            "status": "empty_retrieval", "question": question, "mode": mode,
            "context_only": context_only, "vlm_requested": vlm,
            "images_in_generation_adapter": 0, "retrieval": retrieval,
            "answer": None if context_only else "根据当前检索结果，证据不足，无法回答。",
            "events": runtime.events, "reported_usage": runtime.tracker.get_usage(),
            "elapsed_seconds": time.perf_counter() - started,
            "diagnostic_retrieval_is_separate_from_generation": False,
            **rerank_audit,
        }
    if context_only:
        output = None
    else:
        original_query = rag.lightrag.aquery

        async def observed_query(query, param=None, system_prompt=None, **kwargs):
            result = await original_query(query, param=param, system_prompt=system_prompt, **kwargs)
            if param is not None and getattr(param, "only_need_prompt", False):
                runtime.events.append({"kind": "actual_retrieval_prompt", "prompt": result})
            return result

        rag.lightrag.aquery = observed_query
        try:
            output = await rag.aquery(question, mode=mode, vlm_enhanced=vlm, **opts)
        finally:
            rag.lightrag.aquery = original_query
    rerank_audit = audit_reranking(runtime.settings, runtime.events)
    return {
        "status": "success", "question": question, "mode": mode, "context_only": context_only,
        "vlm_requested": vlm,
        "images_in_generation_adapter": sum(e.get("image_count", 0) for e in runtime.events),
        "retrieval": retrieval, "answer": output, "events": runtime.events,
        "reported_usage": runtime.tracker.get_usage(),
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostic_retrieval_is_separate_from_generation": not context_only,
        **rerank_audit,
    }


def prepare_sample(source_doc_id, doc_id):
    """Select four grounded objects from this paper; preserve raw parser provenance."""
    source = ROOT / "parsed" / slug(source_doc_id)
    original, blocks = validate_parsed(source)
    selectors = (
        lambda b: b["type"] == "text" and "text-embedding-3-large" in b.get("text", ""),
        lambda b: b["type"] == "image" and re.search(r"Figure\s+1\s*:", " ".join(b.get("image_caption", []))),
        lambda b: b["type"] == "table" and re.search(r"Table\s+1\s*:", " ".join(b.get("table_caption", []))),
        lambda b: b["type"] == "equation" and r"\tag{1}" in b.get("text", ""),
    )
    indices = []
    for select in selectors:
        index = next((i for i, block in enumerate(blocks) if select(block)), None)
        if index is None:
            raise ValueError("Paper sample requires Experimental Settings, Figure 1, Table 1 and Equation 1.")
        indices.append(index)
    selected = [blocks[i] for i in sorted(indices)]
    target = ROOT / "parsed" / slug(doc_id)
    if target.exists():
        raise ValueError("Sample directory already exists. Inspect it or use a new --doc-id.")
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "parser-content-list.json", target / "parser-content-list.json")
    save_json(target / "content-list.json", selected)
    paths = {b["img_path"] for b in selected if b.get("img_path")}
    manifest = original | {
        "doc_id": doc_id, "created_at": now(),
        "content_sha256": sha256_file(target / "content-list.json"),
        "assets": [a for a in original["assets"] if a["path"] in paths],
        "content_counts": dict(Counter(b["type"] for b in selected)),
        "normalization": "four_modality_paper_sample_v1; full original parser output preserved",
        "sample_of": {"doc_id": source_doc_id, "manifest_sha256": sha256_file(source / "manifest.json"),
                      "content_sha256": original["content_sha256"], "block_indices": sorted(indices)},
    }
    save_json(target / "manifest.json", manifest)
    validate_parsed(target)
    return manifest


def probe_content(response):
    if not response.choices or not (response.choices[0].message.content or "").strip():
        raise ValueError("Model returned no final answer content.")
    if response.choices[0].finish_reason == "length":
        raise ValueError("Probe answer was truncated. Review the model settings.")
    return response.choices[0].message.content


async def run_probe(args, settings):
    from openai import AsyncOpenAI
    import numpy as np
    roles = ("text", "vision", "embedding") if args.role == "all" else (args.role,)
    validate_settings(settings, roles=roles)
    checks = {}
    if "text" in roles:
        async with AsyncOpenAI(api_key=settings["text_key"], base_url=settings["text_url"],
                               timeout=90, max_retries=0) as client:
            response = await client.chat.completions.create(
                model=settings["text_model"], messages=[{"role": "user", "content": "Reply with OK."}],
                **model_request_kwargs(settings["text_url"], {"temperature": 0, "max_tokens": 64}),
            )
            text = probe_content(response)
            print("Text response:", text)
            checks["text"] = {"status": "PASS", "response": text, "served_model": response.model}
            if is_deepseek(settings["text_url"]):
                response = await client.chat.completions.create(
                    model=settings["text_model"], messages=[{"role": "user", "content":
                    'Return JSON containing high_level_keywords and low_level_keywords, both arrays of strings, '
                    'for this question: How does RAG-Anything combine knowledge graphs and semantic retrieval?'}],
                    **model_request_kwargs(settings["text_url"], {
                        "temperature": 0, "max_tokens": 256, "keyword_extraction": True}),
                )
                keywords = validate_keyword_response(probe_content(response))
                checks["keywords"] = {"status": "PASS", "response": keywords}
                print("Keyword JSON: PASS")
    if "embedding" in roles:
        vectors = await make_embedding(settings)([
            "RAG-Anything integrates text, images, tables and equations.",
            "多模态检索结合知识图谱和向量相似度。",
        ])
        if vectors.shape != (2, settings["embedding_dim"]) or not np.isfinite(vectors).all():
            raise ValueError("Embedding output shape or numeric values are invalid.")
        norms = np.linalg.norm(vectors, axis=1)
        if (norms == 0).any():
            raise ValueError("Embedding model returned a zero vector.")
        checks["embedding"] = {"status": "PASS", "shape": list(vectors.shape), "norms": norms.tolist()}
        print("Embedding shape:", vectors.shape)
    if "vision" in roles and args.image:
        from PIL import Image
        image_path = Path(args.image).expanduser().resolve()
        with Image.open(image_path) as image:
            image.verify()
            image_format = image.format
        mime = Image.MIME.get(image_format)
        if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            raise ValueError("Choose a PNG, JPEG, WebP or GIF image for the vision probe.")
        data = base64.b64encode(image_path.read_bytes()).decode()
        async with AsyncOpenAI(api_key=settings["vision_key"], base_url=settings["vision_url"],
                               timeout=90, max_retries=0) as client:
            response = await client.chat.completions.create(
                model=settings["vision_model"], messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Describe the visible diagram and its labeled graph and retrieval components."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                ]}], **model_request_kwargs(settings["vision_url"], {"temperature": 0, "max_tokens": 512}),
            )
            text = probe_content(response)
            print("Vision response:", text)
            checks["vision"] = {"status": "PASS", "response": text, "served_model": response.model,
                                "image_sha256": sha256_file(image_path), "grounding_requires_review": True}
    elif "vision" in roles:
        print("Vision NOT tested. Run probe --image with a real extracted image before ingestion.")
        checks["vision"] = {"status": "NOT_TESTED"}
    target = ROOT / "runs" / "probes" / (datetime.now(SHANGHAI).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    save_json(target / "result.json", {"created_at": now(), "settings": public_settings(settings), "checks": checks})
    print("Probe receipt:", target / "result.json")


async def run_smoke(args, settings, kb_dir):
    """Real-model roundtrip; ingestion finalizes storage before we reopen it."""
    parse_manifest, blocks = validate_parsed(ROOT / "parsed" / args.doc_id)
    if Counter(b["type"] for b in blocks) != Counter({"text": 1, "image": 1, "table": 1, "equation": 1}):
        raise ValueError("Smoke requires the four-object paper sample created by the sample command.")
    if not parse_manifest.get("sample_of"):
        raise ValueError("Smoke sample provenance is missing.")
    manifest = check_kb_manifest(kb_dir, settings, indexing=True)
    if manifest is None:
        await dispatch_kb(argparse.Namespace(command="ingest", doc_id=args.doc_id, timeout=args.timeout), settings, kb_dir)
    else:
        documents = manifest["documents"]
        if len(documents) != 1 or documents[0]["doc_id"] != args.doc_id:
            raise ValueError("Smoke needs a separate knowledge base containing only its sample.")
    target = ROOT / "runs" / "smoke" / (datetime.now(SHANGHAI).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    receipt = {"status": "RUNNING", "created_at": now(), "kb": args.kb, "doc_id": args.doc_id,
               "settings": public_settings(settings), "checks": {}, "answer_quality": "requires_human_review"}
    save_json(target / "receipt.json", receipt)
    runtime = None
    try:
        # This new runtime reads the index written and closed by dispatch_kb.
        runtime = await build_runtime(settings, kb_dir)
        checks = [(mode, "Which embedding model and vector dimension are specified in the paper's experiments?", mode, True, False)
                  for mode in ("naive", "hybrid", "mix")]
        checks += [
            ("text", "Which embedding model and vector dimension are specified in the paper's experiments?", "mix", False, False),
            ("table", "According to Table 1, how many documents and questions does DocBench contain?", "mix", False, False),
            ("equation", "What does Equation 1 decompose a document into?", "mix", False, False),
            ("vision", "In Figure 1, which two knowledge graphs are built before the merged knowledge graph?", "mix", False, True),
        ]
        for name, question, mode, context_only, vlm in checks:
            result = await asyncio.wait_for(run_query(runtime, question, mode, context_only=context_only, vlm=vlm), args.timeout)
            save_json(target / (name + "-result.json"), result)
            if result["status"] != "success" or any(e.get("status") == "failed" for e in result["events"]):
                raise ValueError(f"Smoke {name} did not complete with usable evidence.")
            if not context_only and not (result["answer"] or "").strip():
                raise ValueError(f"Smoke {name} returned an empty answer.")
            if vlm and result["images_in_generation_adapter"] <= 0:
                raise ValueError("Smoke vision answer did not receive a real image.")
            receipt["checks"][name] = {"status": "PASS", "images": result["images_in_generation_adapter"]}
            save_json(target / "receipt.json", receipt)
            print(name, "PASS", result["answer"] or "retrieval only")
        await persist_runtime(runtime)
        await runtime.rag.finalize_storages()
        runtime = None
        receipt.update(status="PASS", completed_at=now(), reopened_after_ingestion=True)
        save_json(target / "receipt.json", receipt)
        print("Smoke receipt:", target / "receipt.json")
    except BaseException as exc:
        receipt.update(status="FAILED", error_type=type(exc).__name__)
        save_json(target / "receipt.json", receipt)
        raise
    finally:
        if runtime is not None:
            await runtime.rag.finalize_storages()


async def dispatch(args):
    if args.command == "check":
        versions = installed_versions()
        result = {"python": sys.version.split()[0], "packages": versions,
                  "mineru_command": shutil.which("mineru"), "env_file_exists": (ROOT / ".env").exists(),
                  "source_commit": SOURCE_COMMIT}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        verify_dependencies()
        return

    if args.command == "parse":
        verify_dependencies()
        from raganything.parser import MineruParser
        pdf = Path(args.pdf).expanduser().resolve()
        if not pdf.is_file():
            raise FileNotFoundError(str(pdf))
        target = ROOT / "parsed" / slug(args.doc_id)
        if target.exists():
            raise ValueError("Parse directory already exists. Inspect it or use a new --doc-id.")
        target.mkdir(parents=True, exist_ok=False)
        pdf_hash = sha256_file(pdf)
        blocks = await asyncio.to_thread(
            MineruParser().parse_pdf, pdf, output_dir=str(target / "mineru"), method="auto",
            backend="pipeline", device=args.device, timeout=args.timeout,
        )
        save_json(target / "parser-content-list.json", blocks)
        blocks = normalize_blocks(blocks)
        validate_blocks(blocks)
        if sha256_file(pdf) != pdf_hash:
            raise ValueError("PDF changed during parsing; preserve output and use a new doc-id.")
        save_json(target / "content-list.json", blocks)
        assets = [{"path": path, "sha256": sha256_file(path)}
                  for path in sorted({b["img_path"] for b in blocks if b.get("img_path")})]
        counts = dict(Counter(block.get("type", "unknown") for block in blocks))
        save_json(target / "manifest.json", {"schema": MANIFEST_SCHEMA,
                  "source_commit": SOURCE_COMMIT,
                  "doc_id": args.doc_id, "pdf_path": str(pdf),
                  "pdf_sha256": pdf_hash, "content_sha256": sha256_file(target / "content-list.json"),
                  "parser_content_sha256": sha256_file(target / "parser-content-list.json"),
                  "assets": assets, "created_at": now(), "content_counts": counts,
                  "normalization": "chart_to_image_v1; original list preserved",
                  "parser_backend": "pipeline", "parser_method": "auto", "timeout_seconds": args.timeout,
                  "device": args.device, "packages": installed_versions()})
        print(json.dumps({"output": str(target), "content_counts": counts}, ensure_ascii=False, indent=2))
        return

    if args.command == "inspect":
        manifest, blocks = validate_parsed(ROOT / "parsed" / slug(args.doc_id))
        print(json.dumps({"manifest": manifest, "block_count": len(blocks)}, ensure_ascii=False, indent=2))
        return

    if args.command == "sample":
        manifest = prepare_sample(args.source_doc_id, args.doc_id)
        print(json.dumps({"doc_id": manifest["doc_id"], "content_counts": manifest["content_counts"],
                          "source": manifest["sample_of"]}, ensure_ascii=False, indent=2))
        return

    settings = load_settings()
    if getattr(args, "rerank", None) is not None:
        settings["enable_rerank"] = args.rerank
    if args.command == "rerank-check":
        if args.device:
            settings["rerank_device"] = args.device
        events = []
        reranker = LocalBGEReranker(settings, events)
        results = await reranker("What is a giant panda?", [
            "A giant panda is a bear native to China.",
            "The weather is rainy today.",
        ], top_n=2)
        if results[0]["index"] != 0 or results[0]["relevance_score"] <= results[1]["relevance_score"]:
            raise ValueError("Reranker sanity check failed: relevant passage must rank first.")
        print(json.dumps({"status": "PASS", "reranker": reranker.info, "scores": results}, indent=2))
        return
    if args.command == "probe":
        await run_probe(args, settings)
        return

    validate_settings(settings)
    kb_dir = ROOT / "storage" / slug(args.kb)
    with kb_lock(kb_dir):
        if args.command == "smoke":
            await run_smoke(args, settings, kb_dir)
        else:
            await dispatch_kb(args, settings, kb_dir)


async def dispatch_kb(args, settings, kb_dir):
    manifest = check_kb_manifest(kb_dir, settings, indexing=args.command == "ingest")
    if args.command == "ingest":
        parsed = ROOT / "parsed" / slug(args.doc_id)
        parse_manifest, blocks = validate_parsed(parsed)
        if manifest and any(d["doc_id"] == args.doc_id or d["pdf_sha256"] == parse_manifest["pdf_sha256"]
                            for d in manifest["documents"]):
            raise ValueError("This document was already ingested. Query it or use a new knowledge base.")
        pending = {"schema": MANIFEST_SCHEMA, "status": "indexing", "created_at": now(),
                   "source_commit": SOURCE_COMMIT, "starter_sha256": sha256_file(__file__),
                   "index_identity": index_identity(settings), "settings": public_settings(settings),
                   "packages": installed_versions(), "documents": (manifest or {}).get("documents", [])}
        save_json(kb_dir / "pilot_manifest.json", pending)
        runtime = None
        try:
            runtime = await build_runtime(settings, kb_dir)
            index_doc_id = "doc-" + args.doc_id
            await asyncio.wait_for(runtime.rag.insert_content_list(
                blocks, file_path=parse_manifest["pdf_path"], doc_id=index_doc_id), timeout=args.timeout)
            audit = await audit_ingestion(runtime, index_doc_id, blocks)
            save_json(kb_dir / (args.doc_id + "-ingest-audit.json"), audit)
            if audit["missing_groups"] or audit["failed_model_calls"]:
                raise ValueError("Incomplete modal coverage or failed model calls. Inspect ingest audit/log.")
            await persist_runtime(runtime)
            save_json(kb_dir / (args.doc_id + "-ingest-usage.json"), runtime.tracker.get_usage())
            await runtime.rag.finalize_storages()
            runtime = None
            pending["status"] = "ready"
            pending["completed_at"] = now()
            pending["documents"].append(parse_manifest | {"index_doc_id": index_doc_id})
            save_json(kb_dir / "pilot_manifest.json", pending)
            print("Ingestion completed:", kb_dir)
        except BaseException as exc:
            pending["status"] = "failed"
            pending["error_type"] = type(exc).__name__
            save_json(kb_dir / "pilot_manifest.json", pending)
            save_json(kb_dir / (args.doc_id + "-ingest-error.json"), {
                "error_type": type(exc).__name__, "created_at": now(),
                "events": runtime.events if runtime else []})
            raise
        finally:
            if runtime is not None:
                await runtime.rag.finalize_storages()
        return

    for document in manifest["documents"]:
        validate_parsed(ROOT / "parsed" / document["doc_id"])
    run_dir = ROOT / "runs" / (datetime.now(SHANGHAI).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    print("Run directory:", run_dir)
    questions = read_questions(args.questions) if args.command == "batch" else [{"id": "query", "question": args.question}]
    if args.command == "batch":
        known = {d["doc_id"] for d in manifest["documents"]}
        if any(r.get("doc_id") not in known for r in questions):
            raise ValueError("A question doc_id does not exist in this knowledge base.")
    run_manifest = {"created_at": now(), "kb": args.kb,
              "question_count": len(questions), "mode": args.mode, "vlm_requested": args.vlm,
              "context_only": args.context_only, "settings": public_settings(settings),
              "packages": installed_versions(), "index_manifest": manifest,
              "starter_sha256": sha256_file(__file__),
              "questions_sha256": sha256_file(args.questions) if args.command == "batch" else None}
    save_json(run_dir / "run-manifest.json", run_manifest)
    runtime = None
    active_question_id = None
    try:
        runtime = await build_runtime(settings, kb_dir)
        if runtime.reranker_info:
            run_manifest["reranker"] = runtime.reranker_info
            save_json(run_dir / "run-manifest.json", run_manifest)
            print("Reranker:", BGE_RERANK_MODEL, "device:", runtime.reranker_info["device"])
        for record in questions:
            active_question_id = record["id"]
            result = await asyncio.wait_for(run_query(
                runtime, record["question"], args.mode, context_only=args.context_only, vlm=args.vlm,
            ), timeout=args.timeout)
            result["question_id"] = record["id"]
            filename = "result.json" if args.command == "query" else record["id"] + "-result.json"
            save_json(run_dir / filename, result)
            if result["answer"] is not None:
                print(record["id"], result["answer"])
            print(record["id"], "images:", result["images_in_generation_adapter"])
            if result["rerank_enabled"]:
                print(record["id"], "rerank calls:", result["rerank_calls"],
                      "rerank seconds:", round(result["rerank_elapsed_seconds"], 2))
        print("Saved:", run_dir)
    except Exception as exc:
        save_json(run_dir / "error.json", {"error_type": type(exc).__name__, "created_at": now(),
                  "question_id": active_question_id,
                  "events": runtime.events if runtime else []})
        raise
    finally:
        if runtime is not None:
            await runtime.rag.finalize_storages()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check dependencies without calling model APIs")
    rerank_check = commands.add_parser("rerank-check", help="Check downloaded local BGE model; no API or KB access")
    rerank_check.add_argument("--device", choices=("auto", "mps", "cpu"))
    probe = commands.add_parser("probe", help="Call configured model APIs; incurs provider usage")
    probe.add_argument("--image", help="Absolute path to a real image for vision testing")
    probe.add_argument("--role", choices=("all", "text", "vision", "embedding"), default="all")
    parse = commands.add_parser("parse", help="Parse PDF locally; first run may download parser models")
    parse.add_argument("--pdf", required=True)
    parse.add_argument("--doc-id", required=True, type=slug)
    parse.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda:0"))
    parse.add_argument("--timeout", type=int, default=1800)
    inspect = commands.add_parser("inspect", help="Validate hashes and paths without calling model APIs")
    inspect.add_argument("--doc-id", required=True, type=slug)
    sample = commands.add_parser("sample", help="Prepare four real paper objects without calling model APIs")
    sample.add_argument("--source-doc-id", default="rag-anything-v1", type=slug)
    sample.add_argument("--doc-id", default="rag-anything-smoke-v1", type=slug)
    smoke = commands.add_parser("smoke", help="Real-model four-modality indexing/restart/query check; incurs provider usage")
    smoke.add_argument("--doc-id", default="rag-anything-smoke-v1", type=slug)
    smoke.add_argument("--kb", default="smoke-qwen4096-v1", type=slug)
    smoke.add_argument("--timeout", type=int, default=900)
    ingest = commands.add_parser("ingest", help="Build index from reviewed parse output; calls model APIs")
    ingest.add_argument("--doc-id", required=True, type=slug)
    ingest.add_argument("--kb", default="pilot-v1", type=slug)
    ingest.add_argument("--timeout", type=int, default=3600)
    for name in ("query", "batch"):
        query = commands.add_parser(name, help="Retrieve or answer from an existing index; calls model APIs")
        query.add_argument("--kb", default="pilot-v1", type=slug)
        query.add_argument("--question" if name == "query" else "--questions", required=True)
        query.add_argument("--mode", choices=("naive", "hybrid", "mix"), default="mix")
        query.add_argument("--context-only", action="store_true")
        query.add_argument("--vlm", action="store_true")
        query.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                           help="Enable local BGE reranking; --no-rerank overrides ENABLE_RERANK")
        query.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.command == "probe" and args.role == "vision" and not args.image:
        parser.error("probe --role vision requires --image.")
    if getattr(args, "timeout", 1) <= 0 or (args.command == "query" and not args.question.strip()):
        parser.error("Use a positive timeout and a nonempty question.")
    try:
        asyncio.run(dispatch(args))
    except Exception as exc:
        # Do not print provider response bodies, credentials, or full request objects.
        print(f"Failed: {type(exc).__name__}: {str(exc)[:400]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

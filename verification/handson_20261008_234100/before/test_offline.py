"""Offline checks for trace isolation and index identity; no model API calls."""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import reproduce


class QueryParam:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Tracker:
    def reset(self):
        pass

    def get_usage(self):
        return {"total_tokens": 0, "call_count": 0}


class FakeLightRAG:
    def __init__(self):
        self.generation_calls = 0

    async def aquery_data(self, question, param):
        return {"status": "success", "data": {"chunks": [{"content": "evidence"}]}}

    async def aquery(self, query, param=None, system_prompt=None, **kwargs):
        self.generation_calls += 1
        return "actual evidence prompt"


class FakeRAG:
    def __init__(self, events):
        self.lightrag = FakeLightRAG()
        self.events = events

    async def aquery(self, question, mode, vlm_enhanced, **opts):
        if vlm_enhanced:
            await self.lightrag.aquery(question, param=QueryParam(only_need_prompt=True))
            self.events.append({"kind": "vision_request", "image_count": 2})
        return "answer"


class OfflineTests(unittest.TestCase):
    def settings(self):
        return {"text_model": "t", "text_url": "https://example.com/v1",
                "vision_model": "v", "vision_url": "https://example.com/v1",
                "embedding_model": "e", "embedding_url": "https://example.com/v1",
                "embedding_dim": 4, "send_dimensions": True,
                "chunk_size": 100, "chunk_overlap": 10, "summary_language": "English",
                "top_k": 3, "chunk_top_k": 2, "max_entity_tokens": 100,
                "max_relation_tokens": 100, "max_total_tokens": 1000,
                "concurrency": 2,
                "text_key": "SECRET", "vision_key": "SECRET", "embedding_key": "SECRET"}

    def runtime(self):
        events = [{"kind": "old_request", "image_count": 99}]
        return reproduce.Runtime(FakeRAG(events), Tracker(), events, self.settings())

    def test_context_only_never_generates_and_clears_old_image_counts(self):
        runtime = self.runtime()
        with patch.dict(sys.modules, {"lightrag": SimpleNamespace(QueryParam=QueryParam)}):
            result = asyncio.run(reproduce.run_query(runtime, "q", "mix", context_only=True))
        self.assertIsNone(result["answer"])
        self.assertEqual(result["images_in_generation_adapter"], 0)
        self.assertEqual(runtime.rag.lightrag.generation_calls, 0)

    def test_vlm_records_actual_prompt_and_restores_query_method(self):
        runtime = self.runtime()
        original = runtime.rag.lightrag.aquery
        with patch.dict(sys.modules, {"lightrag": SimpleNamespace(QueryParam=QueryParam)}):
            result = asyncio.run(reproduce.run_query(runtime, "q", "mix", vlm=True))
        self.assertEqual(result["images_in_generation_adapter"], 2)
        self.assertTrue(any(e.get("prompt") == "actual evidence prompt" for e in result["events"]))
        self.assertEqual(runtime.rag.lightrag.aquery, original)

    def test_manifest_rejects_changed_embedding_dimension(self):
        settings = self.settings()
        artifacts = reproduce.ROOT / "runs" / "offline-tests"
        artifacts.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="manifest_", dir=artifacts))
        reproduce.save_json(directory / "pilot_manifest.json", {
            "schema": reproduce.MANIFEST_SCHEMA, "source_commit": reproduce.SOURCE_COMMIT,
            "status": "ready", "index_identity": reproduce.index_identity(settings), "documents": []})
        settings["embedding_dim"] = 8
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            reproduce.check_kb_manifest(directory, settings)

    def test_empty_retrieval_is_recorded_without_generation(self):
        runtime = self.runtime()

        async def empty_data(question, param):
            return {"status": "failure", "message": "no results", "data": {}}

        runtime.rag.lightrag.aquery_data = empty_data
        with patch.dict(sys.modules, {"lightrag": SimpleNamespace(QueryParam=QueryParam)}):
            result = asyncio.run(reproduce.run_query(runtime, "q", "mix", vlm=True))
        self.assertEqual(result["status"], "empty_retrieval")
        self.assertEqual(runtime.rag.lightrag.generation_calls, 0)
        self.assertIn("证据不足", result["answer"])

    def test_trace_redacts_image_payload_and_settings_omit_keys(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "question"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,TEST_PAYLOAD"}},
        ]}]
        original = json.dumps(messages)
        redacted = reproduce.redact_messages(messages)
        self.assertNotIn("TEST_PAYLOAD", json.dumps(redacted))
        self.assertEqual(json.dumps(messages), original)
        self.assertEqual(reproduce.image_count(messages), 1)
        self.assertNotIn("SECRET", json.dumps(reproduce.public_settings(self.settings())))

    def directory(self):
        artifacts = reproduce.ROOT / "runs" / "offline-tests"
        artifacts.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="audit_", dir=artifacts))

    def test_nonempty_or_failed_storage_is_never_treated_as_new(self):
        directory = self.directory()
        (directory / "existing-data.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Nonempty"):
            reproduce.check_kb_manifest(directory, self.settings(), indexing=True)

    def test_lock_excludes_other_pilot_processes(self):
        directory = self.directory()
        with reproduce.kb_lock(directory):
            with self.assertRaisesRegex(ValueError, "in use"):
                with reproduce.kb_lock(directory):
                    pass
        with reproduce.kb_lock(directory):
            pass

    def test_parse_integrity_rejects_modified_content(self):
        directory = self.directory()
        pdf = directory / "source.pdf"
        pdf.write_bytes(b"test fixture")
        content = directory / "content-list.json"
        reproduce.save_json(content, [{"type": "text", "text": "source", "page_idx": 0}])
        original = directory / "parser-content-list.json"
        original.write_bytes(content.read_bytes())
        reproduce.save_json(directory / "manifest.json", {
            "schema": reproduce.MANIFEST_SCHEMA, "pdf_path": str(pdf),
            "source_commit": reproduce.SOURCE_COMMIT,
            "pdf_sha256": reproduce.sha256_file(pdf), "content_sha256": reproduce.sha256_file(content),
            "parser_content_sha256": reproduce.sha256_file(original),
            "assets": []})
        reproduce.validate_parsed(directory)
        reproduce.save_json(content, [{"type": "text", "text": "changed", "page_idx": 0}])
        with self.assertRaisesRegex(ValueError, "Parsed content changed"):
            reproduce.validate_parsed(directory)

    def test_all_empty_lists_are_empty_retrieval(self):
        runtime = self.runtime()

        async def empty_data(question, param):
            return {"status": "success", "data": {"chunks": [], "entities": [], "relationships": [], "references": []}}

        runtime.rag.lightrag.aquery_data = empty_data
        with patch.dict(sys.modules, {"lightrag": SimpleNamespace(QueryParam=QueryParam)}):
            result = asyncio.run(reproduce.run_query(runtime, "q", "mix"))
        self.assertEqual(result["status"], "empty_retrieval")
        self.assertEqual(runtime.rag.lightrag.generation_calls, 0)

    def test_modal_audit_exposes_silently_missing_items(self):
        runtime = self.runtime()

        async def status(doc_id):
            return {"fully_processed": True, "chunks_list": ["c1"]}

        async def chunks(ids):
            return [{"is_multimodal": True, "original_type": "image", "page_idx": 0}]

        runtime.rag.get_document_processing_status = status
        runtime.rag.lightrag.text_chunks = SimpleNamespace(get_by_ids=chunks)
        audit = asyncio.run(reproduce.audit_ingestion(runtime, "doc-test", [
            {"type": "image", "page_idx": 0}, {"type": "table", "page_idx": 1}]))
        self.assertEqual(audit["missing_groups"], [{"type": "table", "page_idx": 1, "expected": 1, "stored": 0}])

    def test_invalid_configuration_is_rejected_before_model_calls(self):
        settings = self.settings()
        settings["top_k"] = 0
        with self.assertRaisesRegex(ValueError, "top_k"):
            reproduce.validate_settings(settings)

    def test_batch_rejects_duplicate_question_ids(self):
        path = self.directory() / "questions.jsonl"
        path.write_text('{"id":"q1","question":"a"}\n{"id":"q1","question":"b"}\n')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reproduce.read_questions(path)

    def test_chart_is_routed_to_vision_with_original_type_retained(self):
        original = [{"type": "chart", "chart_caption": ["Figure 2"], "img_path": "/fixture.jpg", "page_idx": 8}]
        result = reproduce.normalize_blocks(original)
        self.assertEqual(original[0]["type"], "chart")
        self.assertEqual(result[0]["type"], "image")
        self.assertEqual(result[0]["image_caption"], ["Figure 2"])
        self.assertEqual(result[0]["_pilot_original_type"], "chart")


if __name__ == "__main__":
    unittest.main()

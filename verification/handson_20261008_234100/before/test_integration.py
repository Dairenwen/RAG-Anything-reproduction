"""Real framework/storage integration with deterministic model substitutes; no remote API.

Artifacts are intentionally retained under runs/integration-tests/. These tests verify
wiring and persistence, not PDF parsing or the quality of actual model retrieval.
"""

import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

import api
import reproduce


async def stub_complete(model, prompt, system_prompt=None, messages=None, **kwargs):
    combined = (system_prompt or "") + "\n" + prompt + "\n" + json.dumps(messages or [])
    if "high_level_keywords" in combined and "low_level_keywords" in combined:
        return json.dumps({"high_level_keywords": ["pilot performance"], "low_level_keywords": ["Pilot", "Score"]})
    if "detailed_description" in combined and "entity_info" in combined:
        modality = "image" if messages else ("table" if "table" in prompt.lower() else "equation")
        return json.dumps({"detailed_description": "Pilot reports Score of 76.3 percent.",
                           "entity_info": {"entity_name": "Pilot " + modality,
                                           "entity_type": modality, "summary": "Pilot Score is 76.3%."}})
    if "entity_name" in combined and ("tuple" in combined.lower() or "<|#|>" in combined):
        return ("entity<|#|>Pilot<|#|>organization<|#|>Pilot reports Score of 76.3 percent.\n"
                "entity<|#|>Score<|#|>category<|#|>The Pilot Score is 76.3 percent.\n"
                "relation<|#|>Pilot<|#|>Score<|#|>performance<|#|>Pilot has Score of 76.3 percent.\n"
                "<|COMPLETE|>")
    return "Pilot reports a score of 76.3%."


async def stub_embed(texts, embedding_dim=None, **kwargs):
    # Equal vectors deliberately make this a plumbing check, not a retrieval-quality test.
    return np.ones((len(texts), embedding_dim or 3072), dtype=np.float32)


class FrameworkIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_multimodal_index_modes_vlm_persistence_and_api(self):
        from lightrag.llm import openai as adapter
        artifacts = reproduce.ROOT / "runs" / "integration-tests"
        artifacts.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="roundtrip_", dir=artifacts))
        image = directory / "fixture.png"
        Image.new("RGB", (32, 32), "white").save(image)
        settings = {
            "text_key": "LOCAL_SUBSTITUTE", "vision_key": "LOCAL_SUBSTITUTE", "embedding_key": "LOCAL_SUBSTITUTE",
            "text_model": "stub-text", "vision_model": "stub-vision", "embedding_model": "stub-embedding",
            "text_url": "http://127.0.0.1:1/v1", "vision_url": "http://127.0.0.1:1/v1", "embedding_url": "http://127.0.0.1:1/v1",
            "embedding_dim": 3072, "send_dimensions": True, "chunk_size": 1200, "chunk_overlap": 100,
            "top_k": 20, "chunk_top_k": 10, "max_entity_tokens": 4000, "max_relation_tokens": 4000,
            "max_total_tokens": 12000, "concurrency": 2, "summary_language": "English",
        }
        blocks = [
            {"type": "text", "text": "Pilot reports a Score of 76.3 percent.", "page_idx": 0},
            {"type": "image", "img_path": str(image), "image_caption": ["Pilot Score"], "page_idx": 0},
            {"type": "table", "table_body": "|Method|Score|\n|---|---|\n|Pilot|76.3%|", "page_idx": 0},
            {"type": "equation", "text": "s=0.763", "latex": "s=0.763", "page_idx": 0},
        ]
        with ExitStack() as stack:
            stack.enter_context(patch.object(adapter, "openai_complete_if_cache", stub_complete))
            stack.enter_context(patch.object(adapter.openai_embed, "func", stub_embed))
            runtime = await reproduce.build_runtime(settings, directory / "storage")
            try:
                await asyncio.wait_for(runtime.rag.insert_content_list(
                    blocks, file_path=str(directory / "fixture.pdf"), doc_id="doc-fixture"), 90)
                audit = await reproduce.audit_ingestion(runtime, "doc-fixture", blocks)
                self.assertEqual(audit["missing_groups"], [])
                self.assertEqual(audit["stored_modal_chunks"], 3)
                self.assertEqual(audit["failed_model_calls"], [])
                for mode in ("naive", "hybrid", "mix"):
                    result = await reproduce.run_query(runtime, "What Score does Pilot report?", mode, context_only=True)
                    self.assertEqual(result["status"], "success", mode)
                    self.assertTrue(result["retrieval"]["data"]["chunks"], mode)
                    reproduce.save_json(directory / (mode + "-retrieval.json"), result)
                visual = await reproduce.run_query(runtime, "What Score does Pilot report in the image?", "mix", vlm=True)
                self.assertGreater(visual["images_in_generation_adapter"], 0)
                self.assertEqual(visual["answer"], "Pilot reports a score of 76.3%.")
                reproduce.save_json(directory / "visual-result.json", visual)
                await reproduce.persist_runtime(runtime)
            finally:
                await runtime.rag.finalize_storages()

            restarted = await reproduce.build_runtime(settings, directory / "storage")
            try:
                after_restart = await reproduce.run_query(restarted, "What Score does Pilot report?", "mix", context_only=True)
                self.assertTrue(after_restart["retrieval"]["data"]["chunks"])
                self.assertEqual(restarted.rag.lightrag.addon_params["language"], "English")
                # Exercise FastAPI startup/shutdown, validation and query response through ASGI.
                async def use_restarted(settings_arg, kb_dir):
                    return restarted

                stack.enter_context(patch.object(api, "load_settings", return_value=settings))
                stack.enter_context(patch.object(api, "check_kb_manifest", return_value={"documents": []}))
                stack.enter_context(patch.object(api, "build_runtime", use_restarted))
                stack.enter_context(patch.dict("os.environ", {"PILOT_KB": directory.name}))
                async with api.app.router.lifespan_context(api.app):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                        response = await client.get("/health")
                        self.assertEqual(response.status_code, 200)
                        bad = await client.post("/query", json={"question": " ", "mode": "mix"})
                        self.assertEqual(bad.status_code, 422)
                        bad_mode = await client.post("/query", json={"question": "q", "mode": "unknown"})
                        self.assertEqual(bad_mode.status_code, 422)
                        good = await client.post("/query", json={"question": "What Score does Pilot report?", "context_only": True})
                        self.assertEqual(good.status_code, 200)
                        self.assertEqual(good.json()["status"], "success")
                reproduce.save_json(directory / "integration-receipt.json", {
                    "status": "PASS", "model_kind": "deterministic substitutes", "remote_api_calls": 0,
                    "checks": ["4 modalities inserted", "3 retrieval modes", "image adapter input", "restart retrieval", "ASGI lifecycle and validation"]})
            finally:
                await restarted.rag.finalize_storages()


if __name__ == "__main__":
    unittest.main()

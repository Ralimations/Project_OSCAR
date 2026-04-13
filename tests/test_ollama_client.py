import base64
import tempfile
import unittest
from pathlib import Path

from core.ollama_client import OllamaClient


class OllamaClientTests(unittest.TestCase):
    def test_list_models_reads_names(self) -> None:
        class TagsClient(OllamaClient):
            def _post_json(self, path: str, payload: dict) -> dict:
                self.captured = (path, payload)
                return {"models": [{"name": "gemma4:e2b"}, {"name": "other"}]}

        client = TagsClient(model="gemma4:e2b")
        self.assertEqual(["gemma4:e2b", "other"], client.list_models())
        self.assertEqual(("/api/tags", {}), client.captured)

    def test_infer_action_attaches_image_when_present(self) -> None:
        class ChatClient(OllamaClient):
            def _post_json(self, path: str, payload: dict) -> dict:
                self.captured = (path, payload)
                return {"message": {"content": "{\"action\":\"wait\",\"value\":0.1}"}}

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            handle.write(b"png-bytes")
            image_path = handle.name
        try:
            client = ChatClient(model="gemma4:e2b")
            result = client.infer_action("prompt", image_path, None, 16000)
            self.assertEqual("{\"action\":\"wait\",\"value\":0.1}", result)
            self.assertEqual("/api/chat", client.captured[0])
            self.assertEqual("gemma4:e2b", client.captured[1]["model"])
            self.assertEqual("prompt", client.captured[1]["messages"][0]["content"])
            self.assertEqual(
                base64.b64encode(Path(image_path).read_bytes()).decode("ascii"),
                client.captured[1]["messages"][0]["images"][0],
            )
        finally:
            Path(image_path).unlink(missing_ok=True)

    def test_extract_message_content_rejects_empty_payload(self) -> None:
        client = OllamaClient(model="gemma4:e2b")
        with self.assertRaises(ValueError):
            client._extract_message_content({})


if __name__ == "__main__":
    unittest.main()

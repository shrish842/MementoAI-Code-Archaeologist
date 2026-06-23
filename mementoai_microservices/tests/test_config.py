from pathlib import Path
import os
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_falls_back_to_local_secrets_module(self) -> None:
        fake_secrets = types.ModuleType("shared.secrets")
        fake_secrets.PINECONE_API_KEY = "pinecone-from-file"
        fake_secrets.GOOGLE_API_KEY = "google-from-file"
        fake_secrets.INGESTION_HOST = "0.0.0.0"
        fake_secrets.INGESTION_PORT = "9001"

        with patch.dict(sys.modules, {"shared.secrets": fake_secrets}, clear=False), patch.dict(os.environ, {}, clear=True):
            config = load_config("ingestion", "127.0.0.1", 8001)

        self.assertEqual(config.pinecone_api_key, "pinecone-from-file")
        self.assertEqual(config.google_api_key, "google-from-file")
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 9001)

    def test_environment_variables_override_local_secrets(self) -> None:
        fake_secrets = types.ModuleType("shared.secrets")
        fake_secrets.PINECONE_API_KEY = "pinecone-from-file"
        fake_secrets.INGESTION_HOST = "0.0.0.0"

        env = {
            "PINECONE_API_KEY": "pinecone-from-env",
            "INGESTION_HOST": "127.0.0.1",
            "INGESTION_PORT": "8101",
        }

        with patch.dict(sys.modules, {"shared.secrets": fake_secrets}, clear=False), patch.dict(os.environ, env, clear=True):
            config = load_config("ingestion", "127.0.0.1", 8001)

        self.assertEqual(config.pinecone_api_key, "pinecone-from-env")
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8101)


if __name__ == "__main__":
    unittest.main()
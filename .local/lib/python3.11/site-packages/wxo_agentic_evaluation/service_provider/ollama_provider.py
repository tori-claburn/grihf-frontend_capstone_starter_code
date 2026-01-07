import json
import os
from typing import List

import requests

from wxo_agentic_evaluation.service_provider.provider import Provider

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaProvider(Provider):
    def __init__(self, model_id=None):
        self.url = OLLAMA_URL + "/api/generate"
        self.model_id = model_id
        super().__init__()

    def query(self, sentence: str) -> str:
        payload = {"model": self.model_id, "prompt": sentence}
        resp = requests.post(self.url, json=payload, stream=True)
        final_text = ""
        data = b""
        for chunk in resp:
            data += chunk
            if data.endswith(b"\n"):
                json_obj = json.loads(data)
                if not json_obj["done"] and json_obj["response"]:
                    final_text += json_obj["response"]
                data = b""

        return final_text

    def encode(self, sentences: List[str]) -> List[list]:
        pass


if __name__ == "__main__":
    provider = OllamaProvider(model_id="llama3.1:8b")
    print(provider.query("ok"))

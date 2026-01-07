from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional, Union

import requests
import rich

from wxo_agentic_evaluation.service_provider.model_proxy_provider import (
    ModelProxyProvider,
)
from wxo_agentic_evaluation.service_provider.watsonx_provider import (
    WatsonXProvider,
)


class LLMResponse:
    """
    NOTE: Taken from LLM-Eval-Kit
    Response object that can contain both content and tool calls
    """

    def __init__(
        self, content: str, tool_calls: Optional[List[Mapping[str, Any]]] = None
    ):
        self.content = content
        self.tool_calls = tool_calls or []

    def __str__(self) -> str:
        """Return the content of the response as a string."""
        return self.content

    def __repr__(self) -> str:
        """Return a string representation of the LLMResponse object."""
        return f"LLMResponse(content='{self.content}', tool_calls={self.tool_calls})"


class LLMKitWrapper(ABC):
    """In the future this wrapper won't be neccesary.
    Right now the referenceless code requires a `generate()` function for the metrics client.
    In refactor, rewrite referenceless code so this wrapper is not needed.
    """

    @abstractmethod
    def chat():
        pass

    def generate(
        self,
        prompt: Union[str, List[Mapping[str, str]]],
        *,
        schema,
        retries: int = 3,
        generation_args: Optional[Any] = None,
        **kwargs: Any,
    ):
        """
        In future, implement validation of response like in llmevalkit
        """

        for attempt in range(1, retries + 1):
            try:
                raw_response = self.chat(prompt)
                response = self._parse_llm_response(raw_response)
                return response
            except Exception as e:
                rich.print(
                    f"[b][r] Generation failed with error '{str(e)}' during `quick-eval` ... Attempt ({attempt} / {retries}))"
                )

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """
        Extract the generated text and tool calls from a watsonx response.

        - For text generation: raw['results'][0]['generated_text']
        - For chat:           raw['choices'][0]['message']['content']
        """
        content = ""
        tool_calls = []

        if isinstance(raw, dict) and "choices" in raw:
            choices = raw["choices"]
            if isinstance(choices, list) and choices:
                first = choices[0]
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    # Extract tool calls if present
                    if "tool_calls" in msg and msg["tool_calls"]:
                        tool_calls = []
                        for tool_call in msg["tool_calls"]:
                            tool_call_dict = {
                                "id": tool_call.get("id"),
                                "type": tool_call.get("type", "function"),
                                "function": {
                                    "name": tool_call.get("function", {}).get(
                                        "name"
                                    ),
                                    "arguments": tool_call.get(
                                        "function", {}
                                    ).get("arguments"),
                                },
                            }
                            tool_calls.append(tool_call_dict)
                elif "text" in first:
                    content = first["text"]

        if not content and not tool_calls:
            raise ValueError(f"Unexpected watsonx response format: {raw!r}")

        # Return LLMResponse if tool calls exist, otherwise just content
        if tool_calls:
            return LLMResponse(content=content, tool_calls=tool_calls)

        return content


class ModelProxyProviderLLMKitWrapper(ModelProxyProvider, LLMKitWrapper):
    def chat(self, sentence: List[str]):
        if self.model_id is None:
            raise Exception("model id must be specified for text generation")
        chat_url = f"{self.instance_url}/ml/v1/text/chat?version=2023-10-25"
        self.refresh_token_if_expires()
        headers = self.get_header()
        data = {
            "model_id": self.model_id,
            "messages": sentence,
            "parameters": self.params,
            "space_id": "1",
            "timeout": self.timeout,
        }
        resp = requests.post(url=chat_url, headers=headers, json=data)
        if resp.status_code == 200:
            return resp.json()
        else:
            resp.raise_for_status()


class WatsonXLLMKitWrapper(WatsonXProvider, LLMKitWrapper):
    def chat(self, sentence: list):
        chat_url = f"{self.api_endpoint}/ml/v1/text/chat?version=2023-05-02"
        headers = self.prepare_header()
        data = {
            "model_id": self.model_id,
            "messages": sentence,
            "parameters": self.params,
            "space_id": self.space_id,
        }
        resp = requests.post(url=chat_url, headers=headers, json=data)
        if resp.status_code == 200:
            return resp.json()
        else:
            resp.raise_for_status()

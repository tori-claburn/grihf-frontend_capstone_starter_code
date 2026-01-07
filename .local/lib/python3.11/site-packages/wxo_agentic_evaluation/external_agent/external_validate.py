import json
from typing import Generator

import requests
import rich

from wxo_agentic_evaluation.external_agent.types import (
    SchemaValidationResults,
    UniversalData,
)

MESSAGES = [
    {"role": "user", "content": "what's the holiday is June 13th in us?"},
    {
        "role": "assistant",
        "content": 'tool_name: calendar_lookup, args {"location": "USA", "data": "06-13-2025"}}',
    },
    {"role": "assistant", "content": "it's National Sewing Machine Day"},
]


class ExternalAgentValidation:
    def __init__(self, credential, auth_scheme, service_url):
        self.credential = credential
        self.auth_scheme = auth_scheme
        self.service_url = service_url

    @property
    def header(self):
        header = {"Content-Type": "application/json"}
        if self.auth_scheme == "API_KEY":
            header = {"X-API-Key": self.credential}
        elif self.auth_scheme == "BEARER_TOKEN":
            header = {"Authorization": f"Bearer {self.credential}"}
        else:
            raise Exception(f"Auth scheme: {self.auth_scheme} is not supported")

        return header

    def _parse_streaming_events(self, resp: Generator[bytes, None, None]):
        data = b""
        for chunk in resp:
            for line in chunk.splitlines(True):
                if line.startswith(b"data:"):
                    line = line.replace(b"data:", b"")
                if line.strip() == b"[DONE]":
                    return
                data += line
                if data.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                    # NOTE: edge case, "data" can be sent in two different chunks
                    if data.startswith(b"data:"):
                        data = data.replace(b"data:", b"")
                    yield data
                    data = b""
        if data:
            yield data

    def _validate_streaming_response(self, resp):
        success = True
        logged_events = []
        for json_str in self._parse_streaming_events(resp):
            json_dict = None
            logged_events.append(json_str)
            try:
                json_dict = json.loads(json_str)
                UniversalData(**json_dict)
            except Exception as e:
                success = False
                break

        return success, logged_events

    def _validate_schema_compliance(self, messages):
        payload = {"stream": True}
        payload["messages"] = messages
        resp = requests.post(
            url=self.service_url, headers=self.header, json=payload
        )
        success, logged_events = self._validate_streaming_response(resp)

        msg = ", ".join([msg["content"] for msg in payload["messages"]])

        if success:
            rich.print(
                f":white_check_mark: External Agent streaming response validation succeeded for '{msg}'."
            )
        else:
            rich.print(
                f":heavy_exclamation_mark:Schema validation failed for messages: '{msg}':heavy_exclamation_mark:\n The last logged event was {logged_events[-1]}.\n"
            )

        return success, logged_events

    def call_validation(
        self, input_str: str, add_context: bool = False
    ) -> SchemaValidationResults:
        if add_context:
            return self.block_validation(input_str)

        msg = {"role": "user", "content": input_str}

        success, logged_events = self._validate_schema_compliance([msg])
        results = SchemaValidationResults(
            success=success, logged_events=logged_events, messages=[msg]
        )

        return results.model_dump()

    def block_validation(self, input_str: str) -> SchemaValidationResults:
        """Tests a block of messages"""
        rich.print(
            f"[gold3]The following prebuilt messages, '{MESSAGES}' is prepended to the input message, '{input_str}'"
        )

        msg = {"role": "user", "content": input_str}

        messages = MESSAGES + [msg]
        success, logged_events = self._validate_schema_compliance(messages)
        results = SchemaValidationResults(
            success=success, logged_events=logged_events, messages=messages
        )

        return results.model_dump()

import json
import os
import time
from collections import deque
from enum import Enum
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple

import requests
import rich
import urllib3
import yaml
from pydantic import BaseModel
from urllib3.exceptions import InsecureRequestWarning

from wxo_agentic_evaluation.arg_configs import TestConfig
from wxo_agentic_evaluation.llm_user import LLMUser
from wxo_agentic_evaluation.service_instance import get_env_settings, tenant_setup
from wxo_agentic_evaluation.service_provider.watsonx_provider import (
    WatsonXProvider,
)
from wxo_agentic_evaluation.type import (
    ContentType,
    ConversationalConfidenceThresholdScore,
    ConversationalSearch,
    ConversationalSearchCitations,
    ConversationalSearchResultMetadata,
    ConversationalSearchResults,
    ConversationSearchMetadata,
    Message,
)
from wxo_agentic_evaluation.utils.utils import (
    Tokenizer,
    is_saas_url,
    safe_divide,
)

tokenizer = Tokenizer()


class Roles(Enum):
    ASSISTANT = "assistant"
    USER = "user"


def calculate_word_overlap_similarity_score(
    first_message_text: str, second_message_text: str
) -> float:
    """Calculate the word overlap similarity score between the .content field of two Message objects.
    Args:
        first_message_text (str): The .content field of the first message.
        second_message_text (str): The .content field of the second message.
    """

    words_in_first_message = tokenizer(first_message_text)
    words_in_second_message = tokenizer(second_message_text)

    # Calculate the number of common words
    common_words = set(words_in_first_message) & set(words_in_second_message)
    unique_words = set(words_in_first_message + words_in_second_message)

    unique_words_count = len(unique_words)
    common_words_count = len(common_words)

    return safe_divide(common_words_count, unique_words_count)


def is_transfer_response(step_detail: Dict):
    # this is not very reliable
    if step_detail["type"] == "tool_response" and step_detail["name"].endswith(
        "_agent"
    ):
        return True
    return False


class CallTracker(BaseModel):
    tool_call: List = []
    tool_response: List = []
    generic: List = []


class WXOClient:
    def __init__(self, service_url, api_key, env: Optional[Dict[str, Any]] = None):
        self.service_url = service_url
        self.api_key = api_key

        ov = os.getenv("WO_SSL_VERIFY")
        if ov and ov.strip().lower() in ("true", "false"):
            self._verify_ssl = ov.strip().lower() == "true"
        else:
            v, bs = (env.get("verify") if env else None), (env.get("bypass_ssl") if env else None)
            self._verify_ssl = False if (
                (bs is True) or (isinstance(bs, str) and bs.strip().lower() == "true") or
                (v is None) or (isinstance(v, str) and v.strip().lower() in {"none", "null"})
            ) else (v if isinstance(v, bool) else True)

        if not self._verify_ssl:
            urllib3.disable_warnings(InsecureRequestWarning)

    def _get_headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def post(self, payload: dict, path: str, stream=False):
        url = f"{self.service_url}/{path}"
        return requests.post(
            url=url,
            headers=self._get_headers(),
            json=payload,
            stream=stream,
            verify=self._verify_ssl,
        )

    def get(self, path: str, params: dict = None):
        url = f"{self.service_url}/{path}"
        return requests.get(
            url,
            params=params,
            headers=self._get_headers(),
            verify=self._verify_ssl,
        )


class WXOInferenceBackend:
    def __init__(self, wxo_client):
        self.wxo_client = wxo_client
        self.enable_saas_mode = is_saas_url(wxo_client.service_url)

    def run(self, user_input: Message, agent_name, thread_id=None):
        agent_id = self.get_agent_id(agent_name)
        payload = {"message": user_input.model_dump(), "agent_id": agent_id}
        if thread_id:
            payload["thread_id"] = thread_id

        if self.enable_saas_mode:
            # TO-DO: this is not validated after the v1 prefix change
            # need additional validation
            path = "/v1/orchestrate/runs"
        else:
            path = "v1/orchestrate/runs"

        response: requests.Response = self.wxo_client.post(payload, path)

        if int(response.status_code) == 200:
            result = response.json()
            return result["thread_id"]
        else:
            response.raise_for_status()

    def _stream_events(
        self, user_input: Message, agent_name: str, thread_id=None
    ) -> Generator[Dict, None, None]:
        agent_id = self.get_agent_id(agent_name)
        payload = {"message": user_input.model_dump(), "agent_id": agent_id}
        if thread_id:
            payload["thread_id"] = thread_id

        if self.enable_saas_mode:
            # TO-DO: this is not validated after the v1 prefix change
            # need additional validation
            path = "/v1/orchestrate/runs?stream=true"
        else:
            path = "v1/orchestrate/runs?stream=true"

        response: requests.Response = self.wxo_client.post(
            payload, path, stream=True
        )
        import json

        for chunk in self._parse_events(response):
            chunk = json.loads(chunk.strip())
            yield chunk

    def parse_conversational_search_response(
        self,
        conversational_search: Mapping[str, Any],
        metadata: ConversationSearchMetadata,
    ) -> ConversationalSearch:
        def parse_citations():
            citations = conversational_search["citations"]
            parsed_citations = []
            for citation in citations:
                c = ConversationalSearchCitations(
                    url=citation.get("url", ""),
                    body=citation.get("body", ""),
                    text=citation.get("text", ""),
                    title=citation.get("title", ""),
                    range_start=citation.get("range_start"),
                    range_end=citation.get("range_end"),
                    search_result_idx=citation.get("search_result_idx"),
                )
                parsed_citations.append(c)

            return parsed_citations

        def parsed_search_results():
            search_results = conversational_search["search_results"]
            parsed_search_results = []
            for result in search_results:
                result_metadata = result.get("result_metadata", {})
                result_metadata = ConversationalSearchResultMetadata(
                    score=result_metadata.get("score"),
                    document_retrieval_source=result_metadata.get(
                        "document_retrieval_source"
                    ),
                )
                c = ConversationalSearchResults(
                    url=result.get("url", ""),
                    body=result.get("body", ""),
                    title=result.get("title", ""),
                    result_metadata=result_metadata,
                )
                parsed_search_results.append(c)

            return parsed_search_results

        citations = parse_citations()
        retrieval_context = parsed_search_results()
        citations_title = conversational_search.get("citations_title", "")
        response_length_option = conversational_search.get(
            "response_length_option", ""
        )
        text = conversational_search.get("text", "")

        confidence_scores = ConversationalConfidenceThresholdScore(
            **conversational_search.get("confidence_scores")
        )
        response_type = conversational_search.get("response_type")
        # should always be conversational_search
        assert response_type == ContentType.conversational_search

        conversational_search = ConversationalSearch(
            metadata=metadata,
            response_type=response_type,
            text=text,
            citations=citations,
            search_results=retrieval_context,
            citations_title=citations_title,
            confidence_scores=confidence_scores,
            response_length_option=response_length_option,
        )

        return conversational_search

    def stream_messages(
        self,
        user_input: Message,
        agent_name: str,
        call_tracker: CallTracker,
        thread_id=None,
    ) -> Tuple[List[Message], str, List[ConversationalSearch]]:
        recover = False
        messages = list()
        conversational_search_data = []

        start_time = time.time()
        for chunk in self._stream_events(user_input, agent_name, thread_id):

            event = chunk.get("event", "")
            if _thread_id := chunk.get("data", {}).get("thread_id"):
                thread_id = _thread_id
                if delta := chunk.get("data", {}).get("delta"):
                    role = delta["role"]
                    if step_details := delta.get("step_details"):
                        if any(
                            is_transfer_response(step_detail)
                            for step_detail in step_details
                        ):
                            continue
                        for idx, step_detail in enumerate(step_details):
                            if step_detail["type"] == "tool_calls":
                                # in step details, we could have [tool_response, tool_call]
                                # in this case, we skip since we already capture the tool call
                                if idx == 1:
                                    continue

                                content_type = ContentType.tool_call
                                for tool in step_detail["tool_calls"]:
                                    # Only add "transfer_to_" calls here. Other tool calls are already
                                    # captured in the next block, including them here will cause duplication
                                    # if not tool["name"].startswith("transfer_to_"):
                                    #     continue
                                    tool_json = {"type": "tool_call"}
                                    tool_json.update(tool)
                                    content = json.dumps(tool_json)
                                    messages.append(
                                        Message(
                                            role=role,
                                            content=content,
                                            type=content_type,
                                            event=event,
                                        )
                                    )
                                    end_time = time.time()
                                    call_tracker.tool_call.append(
                                        end_time - start_time
                                    )
                                    start_time = end_time
                            elif step_detail["type"] == "tool_call":
                                # in step details, we could have [tool_response, tool_call]
                                # in this case, we skip since we already capture the tool call
                                if idx == 1:
                                    continue
                                content_type = ContentType.tool_call
                                content = json.dumps(step_detail)
                                messages.append(
                                    Message(
                                        role=role,
                                        content=content,
                                        type=content_type,
                                        event=event,
                                    )
                                )
                                end_time = time.time()
                                call_tracker.tool_call.append(
                                    end_time - start_time
                                )
                                start_time = end_time
                            elif step_detail["type"] == "tool_response":
                                content = json.dumps(step_detail)
                                content_type = ContentType.tool_response
                                messages.append(
                                    Message(
                                        role=role,
                                        content=content,
                                        type=content_type,
                                        event=event,
                                    )
                                )
                                end_time = time.time()
                                call_tracker.tool_response.append(
                                    end_time - start_time
                                )
                                start_time = end_time
                    elif content_field := delta.get("content"):
                        for val in content_field:
                            response_type = val["response_type"]
                            # TODO: is this ever hit? the event name is "message.created", and it seems the event should be "message.delta"
                            if (
                                response_type == ContentType.text
                                and chunk["event"] == "message_created"
                            ):
                                messages.append(
                                    Message(
                                        role=role,
                                        content=val["text"],
                                        type=ContentType.text,
                                    ),
                                    chunk=event,
                                )
                                end_time = time.time()
                                call_tracker.generic.append(
                                    end_time - start_time
                                )
                                start_time = end_time

                # NOTE: The event here that is parsed is part of the "message.created" event
                elif message := chunk.get("data", {}).get("message"):
                    role = message["role"]
                    for content in message["content"]:
                        if (
                            content["response_type"]
                            == ContentType.conversational_search
                        ):
                            end_time = time.time()
                            call_tracker.generic.append(end_time - start_time)
                            start_time = end_time

                            """ This is under the assumption the flow is (tool call -> tool response -> response back to user).
                            In other words, the tool response is not fed back in to the agent.
                            We get the previous message and extract the `tool_call_id`.
                            
                            NOTE: The previous message is a tool call because how we parse the event stream.
                            NOTE: The conversational search response event does not have a 'tool call id' which can be used to associate with the 'conversational search response'.
                            """

                            last_message = json.loads(messages[-1].content)
                            tool_call_id = last_message.get(
                                "tool_call_id", None
                            )
                            assert tool_call_id is not None
                            conversational_search_metadata = (
                                ConversationSearchMetadata(
                                    tool_call_id=tool_call_id
                                )
                            )
                            conversational_search = (
                                self.parse_conversational_search_response(
                                    conversational_search=content,
                                    metadata=conversational_search_metadata,
                                )
                            )
                            conversational_search_data.append(
                                conversational_search
                            )
                            messages.append(
                                Message(
                                    role=role,
                                    content=content["text"],
                                    type=ContentType.conversational_search,
                                    conversational_search_metadata=conversational_search_metadata,
                                    event=event,
                                )
                            )
                        if content["response_type"] == ContentType.text:
                            messages.append(
                                Message(
                                    role=role,
                                    content=content["text"],
                                    type=ContentType.text,
                                    event=chunk["event"],
                                )
                            )
                            end_time = time.time()
                            call_tracker.generic.append(end_time - start_time)
                            start_time = end_time
            else:
                # Exit the loop if we lose the thread_id
                recover = True
                break

        if recover and (thread_id is not None):
            rich.print(
                "🔬 [bold][magenta]INFO:[/magenta][/bold]",
                f"Attempting to recover messages from thread_id {thread_id}",
            )
            # If we lose the thread_id, we need to wait for a bit to allow the message to come through
            # before attempting to recover the messages.
            time.sleep(10)
            messages = self.recover_messages(thread_id)
            rich.print(
                "🔬 [bold][magenta]INFO:[/magenta][/bold]",
                f"Recovered {len(messages)} messages from thread_id {thread_id}",
            )

        return messages, thread_id, conversational_search_data

    def _parse_events(
        self, stream: Generator[bytes, None, None]
    ) -> Generator[bytes, None, None]:
        data = b""
        for chunk in stream:
            for line in chunk.splitlines(True):
                data += line
                if data.endswith((b"\r\r", b"\n\n", b"\r\n\r\n", b"\n")):
                    yield data
                    data = b""
        if data:
            yield data

    def recover_messages(self, thread_id: str) -> List[Message]:
        messages = self.get_messages(thread_id)
        return self._get_messages_after_last_user(messages)

    def get_messages(self, thread_id) -> List[Message]:
        if self.enable_saas_mode:
            path = f"v1/orchestrate/threads/{thread_id}/messages"
        else:
            path = f"v1/threads/{thread_id}/messages"
        response = self.wxo_client.get(path)
        if response.status_code == 200:
            result = response.json()

        else:
            response.raise_for_status()

        messages = []
        for entry in result:

            tool_call_id = None
            if step_history := entry.get("step_history"):
                for step_message in step_history:
                    role = step_message["role"]
                    if step_details := step_message.get("step_details"):
                        for step_detail in step_details:
                            if step_detail["type"] == "tool_calls":
                                content_type = ContentType.tool_call
                                for tool in step_detail["tool_calls"]:
                                    tool_json = {"type": "tool_call"}
                                    tool_json.update(tool)
                                    content = json.dumps(tool_json)
                                    # TO-DO: review do we even need the get messages for retry loop anymore?
                                    if msg_content := entry.get("content"):
                                        if (
                                            msg_content[0].get("response_type")
                                            == "conversational_search"
                                        ):
                                            continue
                                    messages.append(
                                        Message(
                                            role=role,
                                            content=content,
                                            type=content_type,
                                        )
                                    )
                            elif step_detail["type"] == "tool_call":
                                tool_call_id = step_detail["tool_call_id"]
                                content_type = ContentType.tool_call
                                content = json.dumps(step_detail)
                                messages.append(
                                    Message(
                                        role=role,
                                        content=content,
                                        type=content_type,
                                    )
                                )
                            else:
                                content = json.dumps(step_detail)
                                content_type = ContentType.tool_response
                                messages.append(
                                    Message(
                                        role=role,
                                        content=content,
                                        type=content_type,
                                    )
                                )
            if content_field := entry.get("content"):
                role = entry["role"]
                for val in content_field:
                    if val["response_type"] == ContentType.text:
                        messages.append(
                            Message(
                                role=role,
                                content=val["text"],
                                type=ContentType.text,
                            )
                        )
                    if (
                        val["response_type"]
                        == ContentType.conversational_search
                    ):
                        conversational_search_metadata = (
                            ConversationSearchMetadata(
                                tool_call_id=tool_call_id
                            )
                        )
                        messages.append(
                            Message(
                                role=role,
                                content=val["text"],
                                type=ContentType.text,
                                conversational_search_metadata=conversational_search_metadata,
                            )
                        )

        return messages

    @staticmethod
    def _get_messages_after_last_user(messages: List[Message]) -> List[Message]:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                return messages[i + 1 :]
        return messages

    def get_agent_id(self, agent_name: str):
        if self.enable_saas_mode:
            path = "v1/orchestrate/agents"
        else:
            path = "v1/orchestrate/agents"

        response = self.wxo_client.get(path)

        if response.status_code == 200:
            result = response.json()
            for agent in result:
                if agent.get("name", "") == agent_name:
                    return agent.get("id")

            raise Exception(f"Agent with name {agent_name} not found.")

        else:
            response.raise_for_status()

    def get_agent_name_from_thread_id(self, thread_id: str) -> str:
        if self.enable_saas_mode:
            thread_path = f"v1/orchestrate/threads/{thread_id}"
            agents_path = "v1/orchestrate/agents"
        else:
            thread_path = f"v1/threads/{thread_id}"
            agents_path = "v1/orchestrate/agents"

        thread_response = self.wxo_client.get(thread_path)
        thread_response.raise_for_status()
        thread_data = thread_response.json()
        agent_id = thread_data.get("agent_id", "")

        agents_response = self.wxo_client.get(agents_path)
        agents_response.raise_for_status()
        agents_list = agents_response.json()
        for agent in agents_list:
            if agent.get("id", "") == agent_id:
                return agent.get("name")

        return None


class EvaluationController:

    MAX_CONVERSATION_STEPS = int(os.getenv("MAX_CONVERSATION_STEPS", 20))
    MESSAGE_SIMILARITY_THRESHOLD = float(
        os.getenv("MESSAGE_SIMILARITY_THRESHOLD", 0.98)
    )  # if any two consecutive messages are >98% similar, the inference loop will be terminated
    MAX_REPEATING_MESSAGES = int(
        os.getenv("MAX_REPEATING_MESSAGES", 3)
    )  # this is the maximum number of repeating messages by the user or assistant before terminating the inference loop

    def __init__(
        self,
        wxo_inference_backend: WXOInferenceBackend,
        llm_user: LLMUser,
        config: TestConfig,
    ):
        self.wxo_inference_backend = wxo_inference_backend
        self.llm_user = llm_user
        self.config = config
        self.repeating_output_detection = self.MAX_REPEATING_MESSAGES >= 2

        if self.repeating_output_detection:
            # Use deque for efficient O(1) operations
            self.recent_user_messages = deque(
                maxlen=self.MAX_REPEATING_MESSAGES
            )
            self.recent_assistant_messages = deque(
                maxlen=self.MAX_REPEATING_MESSAGES
            )

    def run(
        self,
        task_n,
        story,
        agent_name: str,
        starting_user_input: str = None,
        attack_instructions: str = None,
    ) -> Tuple[List[Message], List[CallTracker], List[ConversationalSearch]]:
        step = 0
        thread_id = None
        conversation_history: List[Message] = []
        conversational_search_history_data = []
        call_tracker = CallTracker()

        # make this configurable
        while step < self.MAX_CONVERSATION_STEPS:
            if step == 0 and starting_user_input:
                user_input = Message(
                    role="user",
                    content=starting_user_input,
                    type=ContentType.text,
                )
            else:
                if self.config.enable_manual_user_input == True:
                    content = input(
                        "[medium_orchid1]Enter your input[/medium_orchid1] ✍️: "
                    )
                    user_input = Message(
                        role="user", content=content, type=ContentType.text
                    )
                else:  # llm
                    user_input = self.llm_user.generate_user_input(
                        story,
                        conversation_history,
                        attack_instructions=attack_instructions,
                    )
            if self.config.enable_verbose_logging:
                rich.print(
                    f"[dark_khaki][Task-{task_n}][/dark_khaki] 👤[bold blue] User:[/bold blue]",
                    user_input.content,
                )

            if self._is_end(user_input):
                break

            if self.repeating_output_detection:
                self.recent_user_messages.append(user_input.content)

            conversation_history.append(user_input)

            (
                messages,
                thread_id,
                conversational_search_data,
            ) = self.wxo_inference_backend.stream_messages(
                user_input,
                agent_name=agent_name,
                thread_id=thread_id,
                call_tracker=call_tracker,
            )
            if not messages:
                raise RuntimeError(
                    f"[Task-{task_n}] No messages is produced. Exiting task."
                )

            for message in messages:
                if self.repeating_output_detection:
                    if (
                        message.role == Roles.ASSISTANT
                        and message.type == ContentType.text
                    ):
                        self.recent_assistant_messages.append(message.content)

                if self.config.enable_verbose_logging:
                    rich.print(
                        f"[orange3][Task-{task_n}][/orange3] 🤖[bold cyan] WXO:[/bold cyan]",
                        message.content,
                    )

            conversation_history.extend(messages)
            conversational_search_history_data.extend(
                conversational_search_data
            )

            step += 1
        return (
            conversation_history,
            call_tracker,
            conversational_search_history_data,
        )

    def _is_looping(self, messages: deque) -> bool:
        """Checks whether the user or assistant is stuck in a loop.
        Args:
            messages (deque): Defines the message cache to be assessed for similarity.
        Returns:
            bool: True if stuck in a loop, False otherwise.
        """
        sim_count = 0

        if len(messages) >= self.MAX_REPEATING_MESSAGES:
            oldest_cached_message = messages[0]
            for i, old_message in enumerate(messages):
                if i == 0:
                    continue
                if oldest_cached_message == old_message:
                    sim_count += 1
                elif (
                    calculate_word_overlap_similarity_score(
                        oldest_cached_message, old_message
                    )
                    > self.MESSAGE_SIMILARITY_THRESHOLD
                ):
                    sim_count += 1

        return sim_count >= self.MAX_REPEATING_MESSAGES - 1

    def _is_end(self, current_user_input: Message) -> bool:
        """
        Check if the user input indicates the end of the conversation.

        - This function checks if the user input contains 'END'.
        - An END is also triggered when the message cache(s) is filled with messages that are too similar.
        - Elaborate checking ONLY if EvaluationController.END_IF_MISBEHAVING=True
        Args:
            current_user_input (Message): The user message.
        Returns:
            bool: True if the user input indicates an END, False otherwise.
        """
        current_user_message_content = current_user_input.content.strip()

        # Check if the user message contains 'END'
        if "END" in current_user_message_content:
            return True

        if self.repeating_output_detection:
            # Check for repeating user or assistant messages
            if self._is_looping(self.recent_user_messages) or self._is_looping(
                self.recent_assistant_messages
            ):
                return True

        return False  # Final fallback for termination is in the main inference loop, which defines MAX_CONVERSATION_STEPS


def get_wxo_client(
    service_url: Optional[str], tenant_name: str, token: Optional[str] = None
) -> WXOClient:

    token, resolved_url, env = tenant_setup(service_url, tenant_name)
    service_url = service_url or resolved_url

    if not (service_url and str(service_url).strip()):
        raise ValueError(f"service_url not provided and not found in config for tenant '{tenant_name}'")

    wxo_client = WXOClient(service_url=service_url, api_key=token, env=env)
    return wxo_client

if __name__ == "__main__":
    wai_client = WatsonXProvider(model_id="meta-llama/llama-3-3-70b-instruct")
    auth_config_path = (
        f"{os.path.expanduser('~')}/.cache/orchestrate/credentials.yaml"
    )
    with open(auth_config_path, "r") as f:
        auth_config = yaml.safe_load(f)
    tenant_name = "local"
    token = auth_config["auth"][tenant_name]["wxo_mcsp_token"]

    wxo_client = WXOClient(service_url="http://localhost:4321", api_key=token)
    inference_backend = WXOInferenceBackend(wxo_client=wxo_client)
    resp = wxo_client.get("orchestrate/agents")
    resp = resp.json()
    print(resp[0])
    for agent in resp:
        print(agent["name"], agent["display_name"])

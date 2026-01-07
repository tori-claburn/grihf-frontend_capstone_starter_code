import ast
import collections
import json
from typing import Dict, List, Optional

from wxo_agentic_evaluation.arg_configs import KeywordsGenerationConfig
from wxo_agentic_evaluation.prompt.template_render import (
    LlamaKeywordsGenerationTemplateRenderer,
)
from wxo_agentic_evaluation.service_provider import get_provider
from wxo_agentic_evaluation.service_provider.watsonx_provider import Provider
from wxo_agentic_evaluation.type import EvaluationData, Message

ERROR_KEYWORDS = [
    "error",
    "erroneous",
    "exception",
    "traceback",
    "failed",
    "fail",
    "fatal",
    "panic",
    "abort",
    "not found",
    "notfound",
    "cannot",
    "can't",
    "unable",
    "unsuccessful",
    "invalid",
    "incorrect",
    "illegal",
    "unknown",
    "unexpected",
    "unauthorized",
    "permission denied",
    "denied",
    "forbidden",
    "forbidden request",
    "unavailable",
    "unreachable",
    "missing",
    "exceeded",
    "exceeds limit",
    "timed out",
    "timeout",
    "stack trace",
    "syntax error",
    "runtime error",
    "indexerror",
    "keyerror",
    "valueerror",
    "typeerror",
    "zerodivisionerror",
    "segmentation fault",
    "segfault",
    "core dumped",
    "memory error",
    "out of memory",
    "oom",
    "overflow",
    "underflow",
    "crash",
    "bad request",
    "http_code=400",
    "http_code=401",
    "http_code=403",
    "http_code=404",
    "http_code=405",
    "http_code=408",
    "http_code=409",
    "http_code=429",
    "http_code=500",
    "http_code=503",
    "http_code=504",
    "connection refused",
    "connection error",
    "broken pipe",
    "bus error",
    "catastrophic failure",
    "unresolved",
    "infinite recursion",
    "overrun",
    "overwrite",
    "no such file or directory",
    "invalid argument",
    "server is down",
    "server error",
    "sql error",
    "db error",
    "database error",
]


class KeywordsGenerationLLM:
    def __init__(
        self,
        provider: Provider,
        template: LlamaKeywordsGenerationTemplateRenderer,
    ):
        self.provider = provider
        self.prompt_template = template

    def genereate_keywords(self, response) -> Message | None:
        prompt = self.prompt_template.render(response=response)
        res: str = self.provider.query(prompt)
        keywords = ast.literal_eval(res.strip())
        return keywords


class DataAnnotator:
    def __init__(
        self,
        messages: List[Message],
        keywords_generation_config: KeywordsGenerationConfig,
        initial_data: Optional[EvaluationData] = None,
    ):
        self.messages = messages
        self.keywords_generation_config = keywords_generation_config
        self.initial_data = initial_data or EvaluationData(
            agent="",
            story="",
            starting_sentence=messages[0].content if messages else "",
            goals={},
            goal_details=[],
        )

    @staticmethod
    def _is_error_in_message(message: str) -> bool:
        """Heuristic to catch tool calls that fail"""
        message = message.lower()
        return any(keyword in message for keyword in ERROR_KEYWORDS)

    def _get_failed_tool_responses(self) -> list[str]:
        """Get list of IDs for failed tool calls"""
        wrong_tool_response_id = []
        for message in self.messages:
            if message.type == "tool_response":
                content = message.content.lower()
                if self._is_error_in_message(content):
                    wrong_tool_response_id.append(
                        json.loads(message.content)["tool_call_id"]
                    )
        return wrong_tool_response_id

    def _process_tool_call_order(
        self, wrong_tool_response_id: list[str]
    ) -> list[str]:
        """Process and order tool calls, skipping failed ones"""
        # gather all call ids that actually got a response
        valid_call_ids = {
            json.loads(m.content)["tool_call_id"]
            for m in self.messages
            if m.type == "tool_response"
        }

        order = []
        for idx, message in enumerate(self.messages):
            if message.type != "tool_call":
                continue

            content = json.loads(message.content)
            call_id = content.get("tool_call_id") or content.get("id")

            # skip any calls that errored
            if call_id in wrong_tool_response_id:
                continue

            # skip calls that never produced a tool_response
            if call_id not in valid_call_ids:
                continue

            # skip the "reflection" copy that the LLM emits right after a response
            prev = self.messages[idx - 1] if idx > 0 else None
            if (
                prev is not None
                and prev.type == "tool_response"
                and json.loads(prev.content).get("tool_call_id") == call_id
            ):
                continue

            # normalize ids so json dumps only reflects name-args
            content.pop("tool_call_id", None)
            content.pop("id", None)

            signature = json.dumps(content, sort_keys=True)
            # if we’ve seen that exact (name-args) before, drop the old one
            if signature in order:
                order.remove(signature)
            order.append(signature)

        return order

    def _process_tool_calls(self) -> tuple[Dict, List, str]:
        """Process tool calls and generate goals structure"""
        # Get failed tool response IDs and process tool calls
        wrong_tool_response_id = self._get_failed_tool_responses()
        order = self._process_tool_call_order(wrong_tool_response_id)

        goals = {}
        goal_details = []
        function_count = collections.defaultdict(int)
        previous = None

        for tool_call in order:
            call = json.loads(tool_call)
            funct_name = call["name"]
            function_count[funct_name] += 1
            goal_name = funct_name + f"-{function_count[funct_name]}"

            if previous:
                goals[previous] = [goal_name]

            goal_detail = {
                "type": "tool_call",
                "name": goal_name,
                "tool_name": funct_name,
                "args": call["args"],
            }
            goal_details.append(goal_detail)
            previous = goal_name

        return goals, goal_details, previous

    def _process_summarization(
        self, previous: str, goals: Dict, goal_details: List
    ) -> None:
        """Process summarization step"""
        summarize_step = None
        # we assume single summary step at the end
        for message in self.messages[::-1]:
            if message.role == "assistant":
                provider = get_provider(
                    model_id=self.keywords_generation_config.model_id,
                    params={
                        "min_new_tokens": 0,
                        "decoding_method": "greedy",
                        "max_new_tokens": 256,
                    },
                )
                kw_generator = KeywordsGenerationLLM(
                    provider=provider,
                    template=LlamaKeywordsGenerationTemplateRenderer(
                        self.keywords_generation_config.prompt_config
                    ),
                )
                keywords = kw_generator.genereate_keywords(message.content)
                summarize_step = {
                    "name": "summarize",
                    "type": "text",
                    "response": message.content,
                    "keywords": keywords,
                }
                goal_details.append(summarize_step)
                break

        if previous is None:
            goals["summarize"] = []
        elif summarize_step is None:
            goals[previous] = []
        else:
            goals[previous] = ["summarize"]

    def generate(self) -> Dict:
        """Generate the final dataset"""
        goals, goal_details, previous = self._process_tool_calls()
        self._process_summarization(previous, goals, goal_details)

        return {
            "agent": self.initial_data.agent,
            "goals": goals,
            "goal_details": goal_details,
            "story": self.initial_data.story,
            "starting_sentence": self.initial_data.starting_sentence,
        }

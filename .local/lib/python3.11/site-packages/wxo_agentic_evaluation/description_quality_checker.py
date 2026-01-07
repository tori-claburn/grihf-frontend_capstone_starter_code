import os
from enum import Enum
from pathlib import Path
from typing import List

import rich

from wxo_agentic_evaluation.prompt.template_render import (
    BadToolDescriptionRenderer,
)
from wxo_agentic_evaluation.service_provider import get_provider
from wxo_agentic_evaluation.tool_planner import (
    MISSING_DOCSTRING_PROMPT,
    extract_tool_signatures,
    parse_json_string,
)
from wxo_agentic_evaluation.type import ToolDefinition
from wxo_agentic_evaluation.utils.utils import safe_divide


class ToolDescriptionIssue(Enum):
    """
    Represents the binary outcomes the LLM judge will classify in its assessment \
    of the tool's description.
    The presence of these issues in the tool's description indicates poor quality.
    For more detail on what each issue indicates, please take a look at the template here: `wxo_agentic_evaluation/prompt/bad_tool_descriptions_prompt.jinja2`.
    """

    # TODO: Priority-based weighting of issues.
    CONTAINS_REDUNDANT_INFORMATION = "contains_redundant_information"
    USES_VAGUE_LANGUAGE = "uses_vague_language"
    DOES_NOT_HELP_IN_IDENTIFYING_TOOL_UNIQUELY = (
        "does_not_help_in_identifying_tool_uniquely"
    )
    PROVIDES_NO_NEW_INFORMATION = "provides_no_new_information"
    DOES_NOT_CONVEY_TOOL_PURPOSE = "does_not_convey_tool_purpose"


class DescriptionQualityInspector:
    DEFAULT_CLASSIFICATION_THRESHOLD = 40.0  # 2/5 issues detected. A higher score indicates a worse description.
    CLASSIFICATION_SCORE_THRESHOLD = float(
        os.getenv(
            "CLASSIFICATION_SCORE_THRESHOLD", DEFAULT_CLASSIFICATION_THRESHOLD
        )
    )

    LLM_MODEL_ID = "meta-llama/llama-3-2-90b-vision-instruct"
    LLM_PARAMS = {
        "min_new_tokens": 128,
        "decoding_method": "greedy",
        "max_new_tokens": 512,
    }

    WORST_POSSIBLE_EVAL_OUTCOME = len(
        ToolDescriptionIssue
    )  # the final score used for classification is normalized against this value.

    root_dir = os.path.dirname(__file__)
    BAD_TOOL_DESCRIPTIONS_DETECTOR_PATH = os.path.join(
        root_dir, "prompt", "bad_tool_descriptions_prompt.jinja2"
    )

    def __init__(self, llm_client=None):
        if llm_client is None:
            llm_client = get_provider(
                model_id=self.LLM_MODEL_ID,
                params=self.LLM_PARAMS,
            )
        self.llm_client = llm_client
        self.template = BadToolDescriptionRenderer(
            self.BAD_TOOL_DESCRIPTIONS_DETECTOR_PATH
        )
        self.cached_response = None  # this is used in the unit-tests for nuanced analysis of the response.

    @staticmethod
    def extract_tool_desc_from_tool_source(
        tool_source: Path, failing_tools: List[str]
    ) -> List[ToolDefinition]:
        """
        Parses the tool source file to extract the tool description.
        Wraps the description along with the tool name, and args into a `ToolDefinition` for all `failing_tools`.
        This `ToolDefinition` is later rendered into the judge's prompt template for evaluation.
        Args:
            tool_source (Path): The path to the tool source file/dir containing `.py` tools.
            failing_tools (List[str]): List of tool names that failed during inference.
        Returns:
            List[ToolDefinition]: The extracted tool definition(s) or [] if the file contains no @tool decorators.
        """
        all_tool_data = extract_tool_signatures(tool_source)

        tool_definitions = []
        for tool_data in all_tool_data:
            tool_name = tool_data["Function Name"]
            if tool_name in failing_tools:
                tool_definitions.append(
                    ToolDefinition(
                        tool_name=tool_name,
                        tool_description=(
                            tool_data["Docstring"]
                            if tool_data["Docstring"]
                            != MISSING_DOCSTRING_PROMPT
                            else None
                        ),
                        tool_params=tool_data["Arguments"],
                    )
                )
        return tool_definitions

    def detect_bad_description(self, tool_definition: ToolDefinition) -> bool:
        """
        Detects if a tool description is 'bad' using an LLM judge.
        A 'bad' description is one that:
            - does not describe the tool's functionality/use-case clearly
            - does not provide sufficient detail for an agent to understand how to use the tool
            - does not distinguish the tool from other tools
            For the exact definition of a 'bad' description, refer to `ToolDescriptionIssue` Enum.
        Args:
            tool_definition (ToolDefinition): The definition of the tool to evaluate.
        Returns:
            bool: True if the description is 'bad', False otherwise.
        """
        prompt = self.template.render(tool_definition=tool_definition)
        response = self.llm_client.query(prompt)

        # parse JSON objects from cleaned text
        json_objects = parse_json_string(response)

        # pick the first JSON object
        if json_objects:
            response_data = json_objects[0]
            self.cached_response = response_data
        else:
            return False  # likely some unexpected parsing issue, in this case - flags description as good.

        # calculate weighted score
        final_description_score = self._calculate_score(
            response_data=response_data
        )

        return final_description_score >= self.CLASSIFICATION_SCORE_THRESHOLD

    def _calculate_score(self, response_data: dict) -> float:
        """
        Calculates a final score for the tool description.
        This score is used to finally classify a 'good' or 'bad' description.
        :param response_data: Parsed JSON response returned by the LLM judge.
        """
        detected_issues = sum(
            1
            for issue in ToolDescriptionIssue
            if response_data.get(issue.value, "FALSE").upper() == "TRUE"
        )
        return (
            safe_divide(detected_issues, self.WORST_POSSIBLE_EVAL_OUTCOME) * 100
        )

import json
from collections import defaultdict
from http import HTTPStatus
from typing import List, Optional

import rich

from wxo_agentic_evaluation.analytics.tools.types import (
    AgentRecommendation,
    AnalysisResults,
    BadToolCallCause,
    ErrorPatterns,
    ErrorType,
    HallucinatedParameter,
    HallucinationCause,
    ParameterUsageCause,
    RootCauses,
    ToolFailure,
)
from wxo_agentic_evaluation.data_annotator import ERROR_KEYWORDS
from wxo_agentic_evaluation.type import ContentType, EvaluationData, Message


class ToolErrorAnalyzer:
    THRESHOLD = 2  # Minimum consecutive failures to consider a tool as having repeated failures
    COMMON_PLACEHOLDERS = [
        "your user id",
        "your email id",
        "your account id",
        "user_id_here",
        "email_here",
        "account_id_here",
        "<user_id>",
        "<email>",
        "<account_id>",
        "placeholder",
        "example",
        "sample",
    ]

    @classmethod
    def _get_api_error_codes(cls) -> List[str]:
        """Extract all 4xx and 5xx HTTP status codes and phrases for error detection."""
        error_terms = []
        for status in HTTPStatus:
            if status.value >= 400:  # 4xx and 5xx errors
                error_terms.append(
                    str(status.value)
                )  # "400", "404", "500", etc.
                error_terms.append(
                    status.phrase.lower()
                )  # "bad request", "not found", "internal server error", etc.

        return error_terms

    def __init__(
        self, messages: List[Message], ground_truth: Optional[EvaluationData]
    ):
        self.messages = messages
        self.ground_truth = ground_truth
        self.error_patterns = ErrorPatterns()
        self.api_error_codes = self._get_api_error_codes()

    def analyze(self) -> AnalysisResults:
        """Pipeline coordinator.
        Returns:
            dict: Analyzed results with recommendations.
        """
        self._find_error_patterns()
        root_causes: RootCauses = self._root_cause_classifier()
        recommendations: List[AgentRecommendation] = (
            self._generate_agent_definition_improvements(root_causes)
        )

        return AnalysisResults(
            error_patterns=self.error_patterns,
            root_causes=root_causes,
            recommendations=recommendations,
        )

    def _find_error_patterns(self) -> ErrorPatterns:
        """Identify tools that 1. fail and, 2. fail repeatedly.
        Returns:
            dict: toolnames -> failure patterns
        """
        # Group errors by tool name and count sequential failures
        # Return tools with >= threshold consecutive failures

        tool_failures = defaultdict(list)
        for i, msg in enumerate(self.messages):
            if msg.type == ContentType.tool_response and any(
                keyword in str(msg.content).lower()
                for keyword in ERROR_KEYWORDS
            ):
                if isinstance(msg.content, dict):
                    tool_call_id = msg.content.get("tool_call_id")
                elif isinstance(msg.content, str):
                    try:
                        parsed_content = json.loads(msg.content)
                        tool_call_id = (
                            parsed_content.get("tool_call_id")
                            if isinstance(parsed_content, dict)
                            else None
                        )
                    except json.JSONDecodeError:
                        continue

                if not tool_call_id:
                    continue
                tool_call_index = self._find_tool_call_index_by_id(tool_call_id)
                if tool_call_index == -1:
                    continue  # Tool call not found in messages
                tool_call_msg = self.messages[tool_call_index]
                if isinstance(tool_call_msg.content, dict):
                    tool_call = tool_call_msg.content
                else:
                    try:
                        tool_call = json.loads(tool_call_msg.content)
                    except json.JSONDecodeError:
                        continue

                tool_name = tool_call.get("name", "")
                if not tool_name:
                    continue

                tool_failures[tool_name].append(
                    ToolFailure(
                        attempt_index=tool_call_index,
                        parameters=tool_call.get("args", {}),
                        error_message=msg.content,
                    )
                )
        # Store all failures
        self.error_patterns.all_failures = tool_failures

        # Filter tools with >= threshold consecutive failures
        self.error_patterns.repeated_failures = {
            tool: failures
            for tool, failures in tool_failures.items()
            if len(failures) >= self.THRESHOLD
        }

        rich.print(
            f"[cyan]Found {len(self.error_patterns.repeated_failures)} tools with repeated failures:[/cyan]"
        )
        return self.error_patterns

    def _root_cause_classifier(self) -> RootCauses:
        """Map error patterns to probable root causes."""
        causes = RootCauses()

        for tool, failures in self.error_patterns.all_failures.items():
            for failure in failures:
                error_content = (
                    failure.error_message
                )  # handle both Dict and str
                if isinstance(error_content, dict):
                    error_text = error_content.get("content", "")
                    if not isinstance(error_text, str):
                        error_text = str(error_text)
                else:
                    error_text = str(error_content)

                error_msg = error_text.lower()
                params = failure.parameters

                # Compare with ground truth to detect hallucinations
                hallucinated_params = self._detect_hallucinations(tool, params)
                if hallucinated_params:
                    causes.agent_hallucinations.append(
                        HallucinationCause(
                            tool=tool,
                            attempt_index=failure.attempt_index,
                            error=error_msg,
                            hallucinated_params=hallucinated_params,
                        )
                    )

                # Check for placeholder usage
                has_placeholder = False
                for param_name, param_value in params.items():
                    if isinstance(param_value, str):

                        if any(
                            placeholder in param_value.lower()
                            for placeholder in self.COMMON_PLACEHOLDERS
                        ):
                            has_placeholder = True
                            break

                if has_placeholder:
                    causes.incorrect_parameter_usage.append(
                        ParameterUsageCause(
                            tool=tool,
                            placeholder_used=True,
                            attempt_index=failure.attempt_index,
                            error=error_msg,
                        )
                    )
                elif any(
                    term in error_msg
                    for term in ["invalid", "malformed", "expected", "format"]
                ):
                    causes.incorrect_parameter_usage.append(
                        ParameterUsageCause(
                            tool=tool,
                            placeholder_used=False,
                            attempt_index=failure.attempt_index,
                            error=error_msg,
                        )
                    )
                elif any(term in error_msg for term in self.api_error_codes):
                    # Group all HTTP errors under "bad_tool_call" as they all represent (...)
                    # (...) problems with the API request, then further categorize by specific error type
                    causes.bad_tool_call.append(
                        BadToolCallCause(
                            tool=tool,
                            attempt_index=failure.attempt_index,
                            error=error_msg,
                        )
                    )

        return (
            causes  # TODO: add pattern-analysis based RCA for repeated_failures
        )

    def _generate_agent_definition_improvements(
        self, root_causes: RootCauses
    ) -> List[AgentRecommendation]:
        """Generate specific agent prompt template improvements based on root causes."""
        recommendations = []

        # Recurring failures
        if self.error_patterns.repeated_failures:
            for tool, failures in self.error_patterns.repeated_failures.items():
                recommendations.append(
                    AgentRecommendation(
                        issue=f"Agent repeatedly fails when calling {tool}",
                        prompt_addition="The agent made multiple unsuccessful attempts to call this tool. It may help to define fallback behavior for repeated failures, such as asking the user for clarification or escalating the issue.",
                        summary=f"Agent made {len(failures)} failed attempts with {tool}",
                    )
                )

        # Handle incorrect parameters
        param_issues = root_causes.incorrect_parameter_usage
        placeholder_issues = [i for i in param_issues if i.placeholder_used]
        other_param_issues = [i for i in param_issues if not i.placeholder_used]

        if placeholder_issues:
            tools_with_placeholder_issues = {i.tool for i in placeholder_issues}
            tools_placeholder_issues_str = ",".join(
                tools_with_placeholder_issues
            )

            recommendations.append(
                AgentRecommendation(
                    issue=f"Using placeholder values in {tools_placeholder_issues_str}",
                    prompt_addition="A placeholder-style value (like <user_id> or email_here) was used in this tool call. You may want to guide the agent to use actual values from user input or previous responses, rather than placeholders.",
                    summary="The agent used generic placeholders instead of actual data values.",
                )
            )

        if other_param_issues:
            recommendations.append(
                AgentRecommendation(
                    issue="Parameter format errors",
                    prompt_addition="A parameter provided in the tool call didn't match the expected format. Clarifying format expectations (e.g., for dates or IDs) in the agent instructions can help reduce these errors.",
                    summary="The agent provided incorrectly formatted parameters.",
                )
            )

        # Handle bad API requests
        if root_causes.bad_tool_call:
            tool_errors = {}  # maps error code -> erroneous tools set
            for error in root_causes.bad_tool_call:
                tool: str = error.tool
                error_msg: str = error.error

                # Extract error type
                error_type = ErrorType.GENERAL
                if "404" in error_msg or "not found" in error_msg:
                    error_type = ErrorType.NOT_FOUND
                elif "401" in error_msg or "unauthorized" in error_msg:
                    error_type = ErrorType.AUTH_ERROR
                elif "400" in error_msg or "bad request" in error_msg:
                    error_type = ErrorType.BAD_REQUEST

                if error_type not in tool_errors:
                    tool_errors[error_type] = set()
                tool_errors[error_type].add(tool)

            # Generate targetted rec.
            for error_type, tools in tool_errors.items():
                tools_str = ", ".join(tools)

                if error_type == ErrorType.NOT_FOUND:
                    recommendations.append(
                        AgentRecommendation(
                            issue=f"Resource not found errors with tool(s): {tools_str}",
                            prompt_addition="The tool call failed with a “not found” error, possibly due to a missing or incorrect ID. You might consider prompting the agent to confirm such values before using them.",
                            summary="The agent used IDs that don't exist in the database or called endpoints that don't exist.",
                        )
                    )
                elif error_type == ErrorType.AUTH_ERROR:
                    recommendations.append(
                        AgentRecommendation(
                            issue=f"Authentication errors with {tools_str}",
                            prompt_addition="The tool call was rejected due to missing or invalid authentication. If applicable, consider including guidance that limits tool usage to authenticated contexts only.",
                            summary="The agent made API requests with invalid or missing authentication. Please verify your credentials as a first step.",
                        )
                    )
                elif error_type == ErrorType.BAD_REQUEST:
                    recommendations.append(
                        AgentRecommendation(
                            issue=f"Bad request errors with {tools_str}",
                            prompt_addition="The tool call failed due to a malformed request. It may help to reinforce parameter validation and type checking before making such calls.",
                            summary="The agent made API requests with invalid parameter formats or values.",
                        )
                    )
                else:
                    recommendations.append(
                        AgentRecommendation(
                            issue=f"API errors with {tools_str}",
                            prompt_addition="The tool call failed due to an unexpected server or API error. While this may be environmental, reviewing when and how this tool is called could reduce unintended issues.",
                            summary=f"The agent made API requests that were rejected by the server: {error_type}",
                        )
                    )

        # Agent hallucinations
        if root_causes.agent_hallucinations:
            tools_with_hallucinations = {
                i.tool for i in root_causes.agent_hallucinations
            }
            tools_hallucination_str = ", ".join(tools_with_hallucinations)

            hallucination_examples = []
            for cause in root_causes.agent_hallucinations:
                for param in cause.hallucinated_params:
                    hallucination_examples.append(
                        f"{param.param}: expected '{param.expected}', got '{param.actual}'"
                    )

            hallucination_examples_str = "; ".join(
                hallucination_examples[:2]
            )  # Limit to 2 examples
            recommendations.append(
                AgentRecommendation(
                    issue=f"Agent hallucinated parameter values for {tools_hallucination_str}",
                    prompt_addition="The agent used parameter values that did not match the expected inputs for this tool. Consider making it clear that parameter values should come from prior conversation context, not be assumed.",
                    summary=f"The agent made up parameter values instead of using correct ones. Examples: {hallucination_examples_str}",
                )
            )

        return recommendations

    def _detect_hallucinations(
        self, tool_name, actual_params
    ) -> List[HallucinatedParameter]:
        """Compare tool parameters with ground truth to detect hallucinated values."""
        hallucinated = []

        if not self.ground_truth:
            return hallucinated

        # Find corresponding tool call in ground truth
        for goal in self.ground_truth.get("goal_details", []):
            if (
                goal.get("type") == "tool_call"
                and goal.get("tool_name") == tool_name
            ):
                expected_params = goal.get("args", {})

                # Compare .message args with ground-truth expectations
                for param_name, actual_value in actual_params.items():
                    expected_value = expected_params.get(param_name)

                    if param_name not in expected_params:
                        hallucinated.append(
                            HallucinatedParameter(
                                param=param_name,
                                expected="agent fabricated parameter, should not exist",
                                actual=f"{param_name}={actual_value}",
                            )
                        )

                    if actual_value != expected_value:
                        hallucinated.append(
                            HallucinatedParameter(
                                param=param_name,
                                expected=expected_value,
                                actual=actual_value,
                            )
                        )
                break

        return hallucinated

    def _find_tool_call_index_by_id(self, tool_call_id: str) -> int:
        """Find the index of tool_call message by tool_call_id

        Returns:
            int: Index of the message, or -1 if not found
        """
        for i, msg in enumerate(self.messages):
            if msg.type == ContentType.tool_call:
                if isinstance(msg.content, dict):
                    if msg.content.get("tool_call_id") == tool_call_id:
                        return i
                elif isinstance(msg.content, str):
                    try:
                        parsed_content = json.loads(msg.content)
                        if (
                            isinstance(parsed_content, dict)
                            and parsed_content.get("tool_call_id")
                            == tool_call_id
                        ):
                            return i
                    except json.JSONDecodeError:
                        continue
        return -1  # Not found

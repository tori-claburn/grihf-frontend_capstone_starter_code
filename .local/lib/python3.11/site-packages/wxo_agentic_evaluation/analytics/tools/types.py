from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    """Categories of tool call errors."""

    NOT_FOUND = "not_found"
    AUTH_ERROR = "auth_error"
    BAD_REQUEST = "bad_request"
    GENERAL = "general"


class Priority(str, Enum):
    """Priority levels for recommendations."""

    HIGH = "🔴 High"
    MEDIUM = "🟡 Medium"
    LOW = "🆗 Low"


# Foundational data structures
class ToolFailure(BaseModel):
    """Represents a single tool call failure."""

    attempt_index: int = Field(
        ..., description="Index of the failed tool call in messages"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters passed to the tool"
    )
    error_message: Any = Field(
        ..., description="Error message returned by the tool"
    )


class HallucinatedParameter(BaseModel):
    """Represents a parameter that was hallucinated by the agent."""

    param: str = Field(..., description="Parameter name")
    expected: Any = Field(..., description="Expected value or description")
    actual: Any = Field(..., description="Actual value provided by agent")
    type: Optional[str] = Field(
        None, description="Type of hallucination (e.g., 'invented_parameter')"
    )


# Root cause analysis structures
class RootCauseBase(BaseModel):
    """Base class for all root cause classifications."""

    tool: str = Field(..., description="Name of the tool that failed")
    attempt_index: int = Field(..., description="Index of the failed attempt")
    error: str = Field(..., description="Error message (lowercased)")


class HallucinationCause(RootCauseBase):
    """Agent hallucinated parameter values."""

    hallucinated_params: List[HallucinatedParameter] = Field(
        default_factory=list,
        description="List of parameters that were hallucinated",
    )


class ParameterUsageCause(RootCauseBase):
    """Incorrect parameter usage (placeholders or format errors)."""

    placeholder_used: bool = Field(
        ..., description="Whether placeholder values were used"
    )


class BadToolCallCause(RootCauseBase):
    """API errors and bad requests."""

    error_type: ErrorType = Field(
        default=ErrorType.GENERAL, description="Specific type of API error"
    )


class RootCauses(BaseModel):
    """Container for all categorized root causes."""

    incorrect_parameter_usage: List[ParameterUsageCause] = Field(
        default_factory=list
    )
    bad_tool_call: List[BadToolCallCause] = Field(default_factory=list)
    agent_hallucinations: List[HallucinationCause] = Field(default_factory=list)


# Recommendation structures
class AgentRecommendation(BaseModel):
    """Recommendation for improving agent prompt templates."""

    issue: str = Field(..., description="Description of the issue")
    prompt_addition: str = Field(
        ..., description="Suggested prompt improvement"
    )
    summary: str = Field(..., description="Brief explanation of the problem")


class ToolDefinitionRecommendation(BaseModel):
    """Recommendation for improving tool definitions."""

    tool: str = Field(..., description="Name of the tool")
    issue: str = Field(..., description="Issue with the tool definition")
    recommendation: str = Field(..., description="Suggested improvement")
    priority: Priority = Field(..., description="Priority level")
    count: int = Field(..., description="Number of occurrences")
    example: Optional[str] = Field(None, description="Example of the fix")


# Main container structures
class ErrorPatterns(BaseModel):
    """Container for error pattern analysis results."""

    repeated_failures: Dict[str, List[ToolFailure]] = Field(
        default_factory=dict,
        description="Tools that failed repeatedly (>= threshold)",
    )
    all_failures: Dict[str, List[ToolFailure]] = Field(
        default_factory=dict,
        description="All tool failures grouped by tool name",
    )


class AnalysisResults(BaseModel):
    """Complete analysis results from ToolErrorAnalyzer."""

    error_patterns: ErrorPatterns = Field(
        ..., description="Error pattern analysis"
    )
    root_causes: RootCauses = Field(
        ..., description="Root cause classification"
    )
    recommendations: List[AgentRecommendation] = Field(
        default_factory=list,
        description="Agent template improvement recommendations",
    )
    total_tool_calls: Optional[int] = Field(
        None, description="Total number of tool calls made"
    )

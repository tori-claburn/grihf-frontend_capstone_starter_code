from typing import Any, List, Literal, Mapping, Union

from pydantic import BaseModel


class ThinkingStepDetails(BaseModel):
    type: Literal["thinking"]
    content: str


class ToolCall(BaseModel):
    name: str
    args: dict
    id: str


class ToolCallsStepDetails(BaseModel):
    type: Literal["tool_calls"]
    tool_calls: List[ToolCall]


class ToolResponseStepDetails(BaseModel):
    type: Literal["tool_response"]
    content: str  # could also be List[dict], if pre-parsed
    name: str
    tool_call_id: str


StepDetails = Union[
    ThinkingStepDetails, ToolCallsStepDetails, ToolResponseStepDetails
]


class DeltaMessageChoice(BaseModel):
    delta: dict


class ThreadMessageDeltaChoice(BaseModel):
    delta: dict


class ThreadRunStepDeltaChoice(BaseModel):
    delta: dict


class BaseEventData(BaseModel):
    id: str
    object: str
    thread_id: str
    model: str | None = None
    created: int | None = None


class ThreadMessageDeltaData(BaseEventData):
    object: Literal["thread.message.delta"]
    choices: List[ThreadMessageDeltaChoice]


class ThreadRunStepDeltaData(BaseEventData):
    object: Literal["thread.run.step.delta"]
    choices: List[dict]


class UniversalData(BaseEventData):
    object: Union[
        Literal["thread.message.delta"],
        Literal["thread.run.step.delta"],
        Literal["thread.run.step.created"],
        Literal["thread.run.step.completed"],
    ]
    choices: List[ThreadMessageDeltaChoice]
    choices: List[Union[ThreadMessageDeltaChoice, dict]]


class SchemaValidationResults(BaseModel):
    success: bool
    logged_events: List[str]
    messages: List[Mapping[Any, Any]]

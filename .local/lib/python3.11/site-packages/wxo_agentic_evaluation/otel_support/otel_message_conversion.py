from typing import Any, Dict, List, Union, Optional
from wxo_agentic_evaluation.type import Message, ContentType, EventTypes
import json

# with open("src/wxo_agentic_evaluation/otel_support/collie_example.json", "r") as f:
#     data = json.load(f)
#
# otel_traces = data["calls"][-1]["messages"]


def convert_otel_to_message(otel_traces):
    history = []
    for row in otel_traces:
        print(row)
        content = row["content"]
        print(row.keys())
        role = row.get("role", "assistant")

        history.append(Message(role = role, content= content, type=ContentType.text, event=EventTypes.message_created))

    return history
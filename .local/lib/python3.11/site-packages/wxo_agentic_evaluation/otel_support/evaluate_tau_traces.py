from wxo_agentic_evaluation.otel_support.tasks_test import TASKS
from wxo_agentic_evaluation.type import EvaluationData, Message, EventTypes, ContentType
from typing import Any, Dict, List, Union
from wxo_agentic_evaluation.evaluation_package import EvaluationPackage
import json
import glob


file_paths = glob.glob("airline_traces/*.json")


def convert_span_to_messages(span: Dict[str, Any]) -> List[Message]:

    attrs: Dict[str, str] = {}
    for attr in span.get("attributes", []):
        k = attr.get("key")
        v_obj = attr.get("value", {})

        v = v_obj.get("stringValue")
        if v is None and v_obj:
            v = next(iter(v_obj.values()))
        if isinstance(v, (str, int, float, bool)):
            attrs[k] = str(v)
        else:
            attrs[k] = json.dumps(v) if v is not None else ""

    def collect_message_indexes(prefix: str) -> List[int]:
        idxs = set()
        plen = len(prefix)
        for k in attrs:
            if k.startswith(prefix):
                rest = k[plen:]
                first = rest.split(".", 1)[0]
                if first.isdigit():
                    idxs.add(int(first))
        return sorted(idxs)

    messages: List[Message] = []

    in_prefix = "llm.input_messages."
    for i in collect_message_indexes(in_prefix):
        role = attrs.get(f"{in_prefix}{i}.message.role", "")
        tc_prefix = f"{in_prefix}{i}.message.tool_calls."
        has_tool_calls = any(k.startswith(tc_prefix) for k in attrs.keys())

        if has_tool_calls:
            call_indexes = set()
            for k in attrs.keys():
                if k.startswith(tc_prefix):
                    rest = k[len(tc_prefix):]
                    first = rest.split(".", 1)[0]
                    if first.isdigit():
                        call_indexes.add(int(first))

            for ci in sorted(call_indexes):
                name = attrs.get(f"{tc_prefix}{ci}.tool_call.function.name", "")
                args_raw = attrs.get(f"{tc_prefix}{ci}.tool_call.function.arguments", "{}")
                tool_call_id = attrs.get(f"{tc_prefix}{ci}.tool_call.id", "")

                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {"raw": args_raw}

                messages.append(
                    Message(
                        role="assistant",
                        content=json.dumps({"args": args, "name": name, "tool_call_id": tool_call_id}),
                        type=ContentType.tool_call,
                    )
                )
        else:
            content = attrs.get(f"{in_prefix}{i}.message.content", "")
            messages.append(
                Message(
                    role=role if role in {"user", "assistant", "tool"} else "user",
                    content=content,
                    type=ContentType.text,
                )
            )
        if role == "tool":
            pass

    out_prefix = "llm.output_messages."
    for i in collect_message_indexes(out_prefix):
        role = attrs.get(f"{out_prefix}{i}.message.role", "assistant")
        content = attrs.get(f"{out_prefix}{i}.message.content", "")
        messages.append(
            Message(
                role=role if role in {"user", "assistant", "tool"} else "assistant",
                content=content,
                type=ContentType.text,
            )
        )

    return messages

total = 0
success = 0
for i, file in enumerate(file_paths):
    # if i != 2:
    #     continue
    with open(file, "r") as f:
        data = json.load(f)

    messages = []
    for span in data["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        temp = convert_span_to_messages(span)
        if len(temp) > len(messages):
            messages = temp
    for msg in messages:
        #print(msg.role, msg.content)
        pass
    task_id = None
    for kv in data["resourceSpans"][0]["scopeSpans"][0]["spans"][-1]["attributes"]:
        if kv["key"] == "task.index":
            task_id = int(kv["value"]["stringValue"])

    task = TASKS[task_id].model_dump()
    goal_temp = []

    goals = {}
    goal_details = []

    i = 0
    for action in task["actions"]:
        goal_temp.append(action["name"] + f"_{i}")
        args = {}
        for k,v in action["kwargs"].items():
            args[k] = v

        goal_detail = {"type": "tool_call", "name": action["name"] + f"_{i}", "tool_name": action["name"], "args": args }
        goal_details.append(goal_detail)
        i += 1

    if not goal_temp:
        continue
    if len(goal_temp) == 1:
        goals[goal_temp[0]] = []
    else:
        for i in range(len(goal_temp)-1):
            goals.update({goal_temp[i]: [goal_temp[i+1]]})
        goals[goal_temp[-1]]= []

    gt_data = {
        "agent": "airline_agent",
        "goals": goals,
        "goal_details": goal_details,
        "story": task["instruction"],
        "starting_sentence": "",
    }
    gt_data = EvaluationData.model_validate(gt_data)

    tc_name = f"airline_test_{i}"
    try:
        evaluation_package = EvaluationPackage(
            test_case_name=tc_name,
            messages=messages,
            ground_truth=gt_data,
            conversational_search_data=None,
            resource_map=None
        )

        (
            keyword_semantic_matches,
            knowledge_base_metrics,
            messages_with_reason,
            metrics,
        ) = evaluation_package.generate_summary()

        success += metrics.is_success
        total += 1
    except Exception as e:
        raise e
print(success/total)
print(total)
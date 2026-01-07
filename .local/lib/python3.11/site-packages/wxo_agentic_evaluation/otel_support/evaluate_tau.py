import json
from wxo_agentic_evaluation.evaluation_package import EvaluationPackage
from wxo_agentic_evaluation.type import EvaluationData, Message, EventTypes, ContentType

with open("/Users/haodeqi/git/tau-bench/historical_trajectories/gpt-4o-airline.json", "r") as f:
    test_data = json.load(f)


goal_temp = []

goals = {}
goal_details = []

i = 0
for action in test_data[0]["info"]["task"]["actions"]:
    goal_temp.append(action["name"] + f"_{i}")
    goal_detail = {"type": "tool_call", "name": action["name"] + f"_{i}", "tool_name": action["name"], "args": {k: str(v) for k,v in action["kwargs"].items()}}
    goal_details.append(goal_detail)

if len(goal_temp) == 1:
    goals[goal_temp[0]] = []
else:
    for i in range(len(goal_temp)-1):
        goals.update({goal_temp[i]: goal_temp[i+1]})

gt_data = {
    "agent": "airline_agent",
    "goals": goals,
    "goal_details": goal_details,
    "story": test_data[0]["info"]["task"]["instruction"],
    "starting_sentence": "",
}
print("2")
gt_data = EvaluationData.model_validate(gt_data)

tc_name = "airline_1"

print(test_data[0]["traj"][0])

history = []
for msg in test_data[0]["traj"]:
    if msg["role"] == "tool":
        print(msg["content"])
        history.append(Message(role=msg["role"], content=json.dumps({"type": "tool_call", "args": json.loads(msg["content"]), "name": msg["name"], "tool_call_id": msg["tool_call_id"]}), type=ContentType.tool_call,
                               event=EventTypes.message_created))
    else:
        history.append(Message(role=msg["role"], content=str(msg["content"]), type=ContentType.text, event=EventTypes.message_created))

print(f"length of history {history}")

evaluation_package = EvaluationPackage(
    test_case_name=tc_name,
    messages=history,
    ground_truth=gt_data,
    conversational_search_data=None,
    resource_map=None
)
print("1")
(
    keyword_semantic_matches,
    knowledge_base_metrics,
    messages_with_reason,
    metrics,
) = evaluation_package.generate_summary()


print(metrics)
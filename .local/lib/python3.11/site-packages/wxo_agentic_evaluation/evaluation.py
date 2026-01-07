from wxo_agentic_evaluation.evaluation_package import EvaluationPackage
from wxo_agentic_evaluation.otel_support.otel_message_conversion import convert_otel_to_message
from wxo_agentic_evaluation.type import Message, EvaluationData

import json

with open("/Users/haodeqi/git/wxo-evaluation/src/wxo_agentic_evaluation/otel_support/collie_example.json", "r") as f:
    data = json.load(f)

tc_name = "collie_trial"


history = convert_otel_to_message(data["calls"][-1]["messages"])
for message in history:
    print(f"{message.role}: {message.content}")


with open("/Users/haodeqi/git/wxo-evaluation/src/wxo_agentic_evaluation/otel_support/data_simple.json", "r") as f:
    gt = json.load(f)

tc_name = "collie_trial"

gt = EvaluationData.model_validate(gt)


evaluation_package = EvaluationPackage(
    test_case_name=tc_name,
    messages=history,
    ground_truth=gt,
    conversational_search_data=None,
    resource_map=None
)

(
    keyword_semantic_matches,
    knowledge_base_metrics,
    messages_with_reason,
    metrics,
) = evaluation_package.generate_summary()


print(metrics)
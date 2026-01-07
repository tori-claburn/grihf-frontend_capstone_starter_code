import glob
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import rich
from jsonargparse import CLI
from rich.progress import Progress

from wxo_agentic_evaluation.arg_configs import QuickEvalConfig
from wxo_agentic_evaluation.inference_backend import (
    EvaluationController,
    WXOInferenceBackend,
    get_wxo_client,
)
from wxo_agentic_evaluation.llm_user import LLMUser
from wxo_agentic_evaluation.metrics.metrics import (
    FailedSemanticTestCases,
    FailedStaticTestCases,
    ReferenceLessEvalMetrics,
)
from wxo_agentic_evaluation.prompt.template_render import (
    LlamaUserTemplateRenderer,
)
from wxo_agentic_evaluation.referenceless_eval import ReferencelessEvaluation
from wxo_agentic_evaluation.service_provider import get_provider
from wxo_agentic_evaluation.type import (
    ContentType,
    EvaluationData,
    ExtendedMessage,
    Message,
)
from wxo_agentic_evaluation.utils import json_dump
from wxo_agentic_evaluation.utils.open_ai_tool_extractor import (
    ToolExtractionOpenAIFormat,
)
from wxo_agentic_evaluation.utils.utils import ReferencelessEvalPanel

ROOT_DIR = os.path.dirname(__file__)
MODEL_ID = "meta-llama/llama-3-405b-instruct"


def process_test_case(
    task_n, test_case, config, inference_backend, llm_user, all_tools
):
    tc_name = os.path.basename(test_case).replace(".json", "")
    with open(test_case, "r") as f:
        test_case: EvaluationData = EvaluationData.model_validate(json.load(f))

    evaluation_controller = QuickEvalController(
        tc_name, inference_backend, llm_user, config
    )
    rich.print(f"[bold magenta]Running test case: {tc_name}[/bold magenta]")
    messages = evaluation_controller.run(
        task_n,
        agent_name=test_case.agent,
        user_story=test_case.story,
        starting_user_input=test_case.starting_sentence,
    )

    summary, referenceless_metrics = evaluation_controller.generate_summary(
        task_n, all_tools, messages
    )

    outfolder = Path(f"{config.output_dir}/quick-eval")
    outfolder.mkdir(parents=True, exist_ok=True)

    messages_path = outfolder / "messages"
    messages_path.mkdir(exist_ok=True)

    spec_path = outfolder / "tool_spec.json"

    json_dump(spec_path, all_tools)
    json_dump(
        f"{messages_path}/{tc_name}.metrics.json",
        summary.model_dump(),
    )
    json_dump(
        f"{messages_path}/{tc_name}.messages.json",
        [msg.model_dump() for msg in messages],
    )
    json_dump(
        f"{messages_path}/{tc_name}.messages.analyze.json",
        [metric.model_dump() for metric in referenceless_metrics],
    )

    return summary


class QuickEvalController(EvaluationController):
    def __init__(
        self,
        test_case_name: str,
        wxo_inference_backend,
        llm_user,
        config,
    ):
        super().__init__(wxo_inference_backend, llm_user, config)
        self.test_case_name = test_case_name

    def run(
        self, task_n, agent_name, user_story, starting_user_input
    ) -> List[Message]:
        messages, _, _ = super().run(
            task_n, user_story, agent_name, starting_user_input
        )

        return messages

    def generate_summary(
        self, task_n, tools: List[Mapping[str, Any]], messages: List[Message]
    ) -> Tuple[ReferenceLessEvalMetrics, List[ExtendedMessage]]:
        # run reference-less evaluation
        rich.print(f"[b][Task-{task_n}] Starting Quick Evaluation")
        te = ReferencelessEvaluation(
            tools,
            messages,
            MODEL_ID,
            task_n,
            self.test_case_name,
        )
        referenceless_results = te.run()
        rich.print(f"[b][Task-{task_n}] Finished Quick Evaluation")

        summary_metrics = self.compute_metrics(referenceless_results)

        failed_static_tool_calls = summary_metrics.failed_static_tool_calls
        failed_semantic_tool_calls = summary_metrics.failed_semantic_tool_calls

        # tool calls can fail for either a static reason or semantic reason
        failed_static_tool_calls = {
            idx: static_fail for idx, static_fail in failed_static_tool_calls
        }
        failed_semantic_tool_calls = {
            idx: semantic_failure
            for idx, semantic_failure in failed_semantic_tool_calls
        }

        extended_messages = []
        tool_calls = 0
        for message in messages:
            if message.type == ContentType.tool_call:
                if static_reasoning := failed_static_tool_calls.get(tool_calls):
                    extended_message = ExtendedMessage(
                        message=message,
                        reason=[
                            reason.model_dump() for reason in static_reasoning
                        ],
                    )
                elif semantic_reasoning := failed_semantic_tool_calls.get(
                    tool_calls
                ):
                    extended_message = ExtendedMessage(
                        message=message,
                        reason=[
                            reason.model_dump() for reason in semantic_reasoning
                        ],
                    )
                else:
                    extended_message = ExtendedMessage(message=message)
                tool_calls += 1
            else:
                extended_message = ExtendedMessage(message=message)

            extended_messages.append(extended_message)

        # return summary_metrics, referenceless_results
        return summary_metrics, extended_messages

    def failed_static_metrics_for_tool_call(
        self, static_metrics: Mapping[str, Mapping[str, Any]]
    ) -> Optional[List[FailedStaticTestCases]]:
        """
        static.metrics
        """

        failed_test_cases = []

        for metric, metric_data in static_metrics.items():
            if not metric_data.get("valid", False):
                fail = FailedStaticTestCases(
                    metric_name=metric,
                    description=metric_data.get("description"),
                    explanation=metric_data.get("explanation"),
                )

                failed_test_cases.append(fail)

        return failed_test_cases

    def failed_semantic_metrics_for_tool_call(
        self, semantic_metrics: Mapping[str, Mapping[str, Any]]
    ) -> Optional[List[FailedSemanticTestCases]]:
        """
        semantic.general
        semantic.function_selection

        if semantic.function_selection.function_selection_appropriateness fails, do not check the general metrics
        """
        failed_semantic_metric = []

        function_selection_metrics = semantic_metrics.get(
            "function_selection", {}
        ).get("metrics", {})
        function_selection_appropriateness = function_selection_metrics.get(
            "function_selection_appropriateness", None
        )

        if (
            function_selection_appropriateness
            and not function_selection_appropriateness.get("is_correct", False)
        ):
            llm_a_judge = function_selection_appropriateness.get("raw_response")
            fail = FailedSemanticTestCases(
                metric_name=function_selection_appropriateness.get(
                    "metric_name"
                ),
                evidence=llm_a_judge.get("evidence"),
                explanation=llm_a_judge.get("explanation"),
                output=llm_a_judge.get("output"),
                confidence=llm_a_judge.get("confidence"),
            )
            failed_semantic_metric.append(fail)

            return failed_semantic_metric

        general_metrics = semantic_metrics.get("general", {}).get("metrics", {})
        for metric_data in general_metrics.values():
            llm_a_judge = metric_data.get("raw_response")
            if not metric_data.get("is_correct", False):
                fail = FailedSemanticTestCases(
                    metric_name=metric_data.get("metric_name"),
                    evidence=llm_a_judge.get("evidence"),
                    explanation=llm_a_judge.get("explanation"),
                    output=llm_a_judge.get("output"),
                    confidence=llm_a_judge.get("confidence"),
                )
                failed_semantic_metric.append(fail)

        return failed_semantic_metric

    def compute_metrics(
        self, quick_eval_results: List[Mapping[str, Any]]
    ) -> ReferenceLessEvalMetrics:
        number_of_tool_calls = len(quick_eval_results)
        number_of_static_failures = 0
        number_of_semantic_failures = 0
        successful_tool_calls = 0

        failed_static_tool_calls = (
            []
        )  # keep track of tool calls that failed at the static stage
        failed_semantic_tool_calls = (
            []
        )  # keep track of tool calls that failed for semantic reason

        from pprint import pprint

        # pprint("quick eval results: ")
        # pprint(quick_eval_results)

        for tool_call_idx, result in enumerate(quick_eval_results):
            static_passed = result.get("static", {}).get(
                "final_decision", False
            )
            semantic_passed = result.get("overall_valid", False)

            if static_passed:
                if semantic_passed:
                    successful_tool_calls += 1
                else:
                    number_of_semantic_failures += 1
                    failed_semantic_tool_calls.append(
                        (
                            tool_call_idx,
                            self.failed_semantic_metrics_for_tool_call(
                                result.get("semantic")
                            ),
                        )
                    )
            else:
                number_of_static_failures += 1
                failed_static_cases = self.failed_static_metrics_for_tool_call(
                    result.get("static").get("metrics")
                )
                failed_static_tool_calls.append(
                    (tool_call_idx, failed_static_cases)
                )

        referenceless_eval_metric = ReferenceLessEvalMetrics(
            dataset_name=self.test_case_name,
            number_of_tool_calls=number_of_tool_calls,
            number_of_successful_tool_calls=successful_tool_calls,
            number_of_static_failed_tool_calls=number_of_static_failures,
            number_of_semantic_failed_tool_calls=number_of_semantic_failures,
            failed_semantic_tool_calls=failed_semantic_tool_calls,
            failed_static_tool_calls=failed_static_tool_calls,
        )

        return referenceless_eval_metric


def main(config: QuickEvalConfig):
    wxo_client = get_wxo_client(
        config.auth_config.url,
        config.auth_config.tenant_name,
        config.auth_config.token,
    )
    inference_backend = WXOInferenceBackend(wxo_client)
    llm_user = LLMUser(
        wai_client=get_provider(
            config=config.provider_config,
            model_id=config.llm_user_config.model_id,
        ),
        template=LlamaUserTemplateRenderer(
            config.llm_user_config.prompt_config
        ),
        user_response_style=config.llm_user_config.user_response_style,
    )
    all_tools = ToolExtractionOpenAIFormat.from_path(config.tools_path)

    test_cases = []
    for test_path in config.test_paths:
        if os.path.isdir(test_path):
            test_path = os.path.join(test_path, "*.json")
        test_cases.extend(sorted(glob.glob(test_path)))

    executor = ThreadPoolExecutor(max_workers=config.num_workers)
    rich.print(f"[g]INFO - Number of workers set to {config.num_workers}")
    futures = []
    for idx, test_case in enumerate(test_cases):
        if not test_case.endswith(".json") or test_case.endswith("agent.json"):
            continue
        future = executor.submit(
            process_test_case,
            idx,
            test_case,
            config,
            inference_backend,
            llm_user,
            all_tools,
        )
        futures.append((test_case, future))

    results = []
    if futures:
        with Progress() as progress:
            task = progress.add_task(
                f"[purple]Running quick evaluation on {len(futures)} tasks...",
                total=len(futures),
            )
            for test_case, future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    rich.print(f"test case {test_case} fails with {e}")
                    traceback.print_exc()
                finally:
                    progress.update(task, advance=1)

    ReferencelessEvalPanel(results).print()


if __name__ == "__main__":
    main(CLI(QuickEvalConfig, as_positional=False))

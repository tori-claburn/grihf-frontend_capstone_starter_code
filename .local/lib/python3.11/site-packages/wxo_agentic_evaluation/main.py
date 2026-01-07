import csv
import dataclasses
import glob
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import rich
import yaml
from jsonargparse import CLI
from rich.progress import Progress

from wxo_agentic_evaluation.arg_configs import TestConfig
from wxo_agentic_evaluation.evaluation_package import EvaluationPackage
from wxo_agentic_evaluation.inference_backend import (
    EvaluationController,
    WXOInferenceBackend,
    get_wxo_client,
)
from wxo_agentic_evaluation.llm_user import LLMUser
from wxo_agentic_evaluation.metrics.metrics import (
    KnowledgeBaseMetricSummary,
    TextMatchType,
    ToolCallAndRoutingMetrics,
)
from wxo_agentic_evaluation.prompt.template_render import (
    LlamaUserTemplateRenderer,
)
from wxo_agentic_evaluation.resource_map import ResourceMap
from wxo_agentic_evaluation.service_provider import get_provider
from wxo_agentic_evaluation.type import EvaluationData
from wxo_agentic_evaluation.utils import json_dump
from wxo_agentic_evaluation.utils.utils import (
    SummaryPanel,
    create_table,
    safe_divide,
)


def process_test_case(
    task_n, test_case, config, inference_backend, resource_map, llm_user
):
    summary_results_for_path = []
    tc_name = os.path.basename(test_case).replace(".json", "")
    with open(test_case, "r") as f:
        test_case: EvaluationData = EvaluationData.model_validate(json.load(f))

    evaluation_controller = EvaluationController(
        wxo_inference_backend=inference_backend,
        llm_user=llm_user,
        config=config,
    )
    rich.print(f"[bold magenta]Running test case: {tc_name}[/bold magenta]")
    (
        history,
        call_tracker,
        conversational_search_data,
    ) = evaluation_controller.run(
        task_n,
        test_case.story,
        agent_name=test_case.agent,
        starting_user_input=test_case.starting_sentence,
    )
    result = list()
    for message in history:
        result.append(message.model_dump())

    json_dump(
        os.path.join(config.output_dir, "messages", tc_name + ".messages.json"),
        result,
    )

    if len(conversational_search_data) > 0:
        fn = tc_name + ".retrieval_context.json"
        out_folder = Path(config.output_dir) / "knowledge_base_metrics"
        out_folder.mkdir(exist_ok=True)
        rc = [context.model_dump() for context in conversational_search_data]
        json_dump(out_folder / fn, rc)

    # If data annotation run, skip summary generation
    if config.data_annotation_run:
        return summary_results_for_path  # empty result set, skip summary

    evaluation_package = EvaluationPackage(
        test_case_name=tc_name,
        messages=history,
        ground_truth=test_case,
        conversational_search_data=conversational_search_data,
        resource_map=resource_map,
    )
    (
        keyword_semantic_matches,
        knowledge_base_metrics,
        messages_with_reason,
        metrics,
    ) = evaluation_package.generate_summary()
    temp = []
    for message in messages_with_reason:
        temp.append(message.model_dump())
    json_dump(
        os.path.join(
            config.output_dir, "messages", tc_name + ".messages.analyze.json"
        ),
        temp,
    )

    json_dump(
        os.path.join(config.output_dir, "messages", tc_name + ".metrics.json"),
        metrics.model_dump(),
    )

    metrics.dataset_name = tc_name
    metrics.avg_resp_time = (
        sum(call_tracker.generic) + sum(call_tracker.tool_call)
    ) / (len(call_tracker.generic) + len(call_tracker.tool_call))
    metrics.avg_resp_time = round(metrics.avg_resp_time, 2)

    summary_results_for_path.append((metrics, knowledge_base_metrics))

    return summary_results_for_path


def main(config: TestConfig):
    executor = ThreadPoolExecutor(max_workers=config.num_workers)
    if config.num_workers > 1 and config.enable_manual_user_input:
        rich.print(
            "[bold yellow]Warning ⚠️: Manual user input is disabled for parallel execution.[/bold yellow]"
        )
        config.enable_manual_user_input = (
            False  # disable manual user input for parallel execution
        )
        # reason: threads continue to stream messages while waiting for user input, which is not desired
        # and the manual input prompt is not labelled properly in the UI
    wxo_client = get_wxo_client(
        config.auth_config.url,
        config.auth_config.tenant_name,
        config.auth_config.token,
    )
    resource_map = ResourceMap(wxo_client)
    inference_backend = WXOInferenceBackend(wxo_client=wxo_client)
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

    print(f"Running evaluation with tenant {config.auth_config.tenant_name}")

    results_list = []

    knowledge_base_output_folder = (
        Path(config.output_dir) / "knowledge_base_metrics"
    )
    knowledge_base_output_folder.mkdir(exist_ok=True, parents=True)
    detailed_rag_output_file = (
        knowledge_base_output_folder / "knowledge_base_detailed_metrics.json"
    )
    summary_rag_output_file = (
        Path(config.output_dir) / "knowledge_base_summary_metrics.json"
    )

    os.makedirs(os.path.join(config.output_dir, "messages"), exist_ok=True)
    available_res = set()
    if config.skip_available_results:
        available_res = set(
            [
                os.path.basename(f).replace(".messages", "")
                for f in glob.glob(
                    os.path.join(
                        config.output_dir, "messages", "*.messages.json"
                    )
                )
            ]
        )

    test_cases = []
    for test_path in config.test_paths:
        if os.path.isdir(test_path):
            test_path = os.path.join(test_path, "*.json")
        test_cases.extend(sorted(glob.glob(test_path)))

    futures = []
    task_n = 0
    for test_case in test_cases:
        if not test_case.endswith(".json") or test_case.endswith("agent.json"):
            continue
        if config.skip_available_results:
            if test_case in available_res:
                print(
                    f"Skipping test case {test_case} as results already exist."
                )
                continue

        future = executor.submit(
            process_test_case,
            task_n,
            test_case,
            config,
            inference_backend,
            resource_map,
            llm_user,
        )

        futures.append((test_case, future))
        task_n += 1

    if futures:
        with Progress() as progress:
            task1 = progress.add_task(
                f"[purple]Evaluating {len(futures)} tasks...",
                total=len(futures),
            )
            for test_case, future in futures:
                try:
                    results_list.extend(future.result())
                except Exception as e:
                    rich.print(f"test case {test_case} fails with {e}")
                    traceback.print_exc()
                finally:
                    progress.update(task1, advance=1)

    tool_call_metrics = [metric[0] for metric in results_list]
    knowledge_base_metrics = [metric[1] for metric in results_list]

    rag_metric_summary = KnowledgeBaseMetricSummary(
        knowledge_base_metrics=knowledge_base_metrics
    )
    SummaryPanel(rag_metric_summary).print()

    with open(detailed_rag_output_file, "w+", encoding="utf-8") as f:
        json.dump(
            rag_metric_summary.model_dump(by_alias=True)["detailed"],
            f,
            indent=4,
        )

    with open(summary_rag_output_file, "w+", encoding="utf-8") as f:
        json.dump(
            rag_metric_summary.model_dump(by_alias=True)["summary"], f, indent=4
        )

    if len(tool_call_metrics) > 0:
        # remove the average row if exist
        tool_call_metrics = [
            row
            for row in tool_call_metrics
            if row.dataset_name != "Summary (Average)"
        ]

        def filter_display_only_values(
            tool_call_metric: ToolCallAndRoutingMetrics,
        ):
            row = {
                "Dataset": tool_call_metric.dataset_name,
                "Total Steps": tool_call_metric.total_steps,
                "LLM Steps": tool_call_metric.llm_step,
                "Total Tool Calls": tool_call_metric.total_tool_calls,
                "Tool Call Precision": tool_call_metric.tool_call_precision,
                "Tool Call Recall": tool_call_metric.tool_call_recall,
                "Agent Routing Accuracy": tool_call_metric.agent_routing_accuracy,
                "Text Match": tool_call_metric.text_match,
                "Journey Success": tool_call_metric.is_success,
                "Avg Resp Time (sec)": tool_call_metric.avg_resp_time,
            }
            return row

        def create_avg_row(metrics: List[dict]):
            avg_row = {
                "Dataset": "Summary (Average)",
                "Total Steps": 0,
                "LLM Steps": 0,
                "Total Tool Calls": 0,
                "Tool Call Precision": 0,
                "Tool Call Recall": 0,
                "Agent Routing Accuracy": 0,
                "Text Match": 0,
                "Journey Success": 0,
                "Avg Resp Time (sec)": 0,
            }
            if metrics:
                for row in metrics:
                    avg_row["Total Steps"] += row["Total Steps"]
                    avg_row["LLM Steps"] += row["LLM Steps"]
                    avg_row["Total Tool Calls"] += row["Total Tool Calls"]
                    avg_row["Tool Call Precision"] += row["Tool Call Precision"]
                    avg_row["Tool Call Recall"] += row["Tool Call Recall"]
                    avg_row["Agent Routing Accuracy"] += row[
                        "Agent Routing Accuracy"
                    ]
                    avg_row["Text Match"] += (
                        row["Text Match"] == TextMatchType.text_match.value
                    )
                    avg_row["Journey Success"] += row["Journey Success"]
                    avg_row["Avg Resp Time (sec)"] += row["Avg Resp Time (sec)"]

                avg_row["Total Steps"] = round(
                    safe_divide(avg_row["Total Steps"], len(metrics)), 2
                )
                avg_row["LLM Steps"] = round(
                    safe_divide(avg_row["LLM Steps"], len(metrics)), 2
                )
                avg_row["Total Tool Calls"] = round(
                    safe_divide(avg_row["Total Tool Calls"], len(metrics)), 2
                )
                avg_row["Tool Call Precision"] = round(
                    safe_divide(avg_row["Tool Call Precision"], len(metrics)), 2
                )
                avg_row["Tool Call Recall"] = round(
                    safe_divide(avg_row["Tool Call Recall"], len(metrics)), 2
                )
                avg_row["Agent Routing Accuracy"] = round(
                    safe_divide(
                        avg_row["Agent Routing Accuracy"], len(metrics)
                    ),
                    2,
                )
                avg_row["Text Match"] = round(
                    safe_divide(
                        avg_row["Text Match"],
                        len(
                            [
                                row
                                for row in metrics
                                if row["Text Match"]
                                != TextMatchType.text_match.na
                            ]
                        ),
                    ),
                    2,
                )
                avg_row["Journey Success"] = round(
                    safe_divide(avg_row["Journey Success"], len(metrics)), 2
                )
                avg_row["Avg Resp Time (sec)"] = round(
                    safe_divide(avg_row["Avg Resp Time (sec)"], len(metrics)), 2
                )
            return avg_row

        tool_call_metrics_for_display = []
        for row in tool_call_metrics:
            tool_call_metrics_for_display.append(
                filter_display_only_values(row)
            )
        tool_call_metrics_for_display.append(
            create_avg_row(tool_call_metrics_for_display)
        )
        tool_call_table_for_display = create_table(
            tool_call_metrics_for_display
        )

        if tool_call_table_for_display:
            tool_call_table_for_display.print()

    if len(tool_call_metrics) > 0:
        tool_call_metrics = [
            metric.model_dump() for metric in tool_call_metrics
        ]
        output_file = os.path.join(config.output_dir, "summary_metrics.csv")
        header = list(tool_call_metrics[0].keys())

        with open(output_file, "w") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(header)
            for entry in tool_call_metrics:
                csv_writer.writerow([entry[name] for name in header])

    with open(
        os.path.join(config.output_dir, "config.yml"), "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(dataclasses.asdict(config), f)

    print(f"Results saved to {config.output_dir}")


if __name__ == "__main__":
    main(CLI(TestConfig, as_positional=False))

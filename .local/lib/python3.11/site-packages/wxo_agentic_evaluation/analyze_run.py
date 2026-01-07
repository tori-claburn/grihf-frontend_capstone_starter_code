import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from jsonargparse import CLI
from rich.console import Group
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from wxo_agentic_evaluation.arg_configs import AnalyzeConfig
from wxo_agentic_evaluation.description_quality_checker import (
    DescriptionQualityInspector,
)
from wxo_agentic_evaluation.metrics.metrics import ToolCallAndRoutingMetrics
from wxo_agentic_evaluation.type import (
    ContentType,
    ExtendedMessage,
    ToolDefinition,
)
from wxo_agentic_evaluation.utils.rich_utils import (
    IncorrectParameterUtils,
    is_ok,
    pretty_print,
    print_done,
    warn,
)
from wxo_agentic_evaluation.utils.utils import add_line_seperator


class Analyzer:
    def __init__(self):
        self.analysis_cache: Dict[str, List[Text]] = (
            {}
        )  # the failing tools cached here won't be re-analyzed.
        # tool_name -> description analysis

    @staticmethod
    def _generate_style_config():
        return Style(
            color="magenta",
            blink=True,
            bold=True,
        )

    def _split_cache(
        self, failing_tools: Set[str]
    ) -> tuple[List[str], List[Text]]:

        tools_to_analyze: List[str] = []
        cached_lines: List[Text] = []
        tools_analyzed: List[str] = []

        for tool_name in sorted(failing_tools):
            cached_analysis = self.analysis_cache.get(tool_name)
            if cached_analysis:
                cached_lines.extend(cached_analysis)
                tools_analyzed.append(tool_name)
            else:
                tools_to_analyze.append(tool_name)

        if tools_analyzed:
            pretty_print(
                content=f"ℹ️ Loading cached analysis since these failing tools: {tools_analyzed} have been analyzed previously.",
                style="bold cyan",
            )

        return (tools_to_analyze, cached_lines)

    def analyze_failing_tool_description_quality(
        self,
        inspector: DescriptionQualityInspector,
        tool_definition_path: str,
        failing_tools: Set[str],
    ) -> List[Text]:
        """
        :param tool_definition_path: Path to the tool definition file.
        :param failing_tools: Set of tool names that failed.
        :return: List of rich `Text` objects containing feedback for the customer.
        """

        pretty_print(
            content=f"⚙️ Checking tool description quality for failing tools: {sorted(failing_tools)}",
            style="bold cyan",
        )

        analysis_for_display: List[Text] = []

        # Step 1: get tools not yet analyzed and cached analysis for tools analyzed previously
        tools_to_analyze, cached_analysis = self._split_cache(failing_tools)
        if cached_analysis:
            analysis_for_display.extend(cached_analysis)

        # Step 2: analyze cache misses
        if tools_to_analyze:

            failing_tool_definitions: List[ToolDefinition] = (
                inspector.extract_tool_desc_from_tool_source(
                    Path(tool_definition_path),
                    tools_to_analyze,
                )
            )

            if not failing_tool_definitions:
                analysis_for_display.append(
                    warn(
                        message=f"No tool definitions(with '@tool' decorators) for failed tools: '{tools_to_analyze}' found in the file: '{tool_definition_path}'"
                    )
                )
                return analysis_for_display

            missing_tools = self._get_tools_not_found_in_source(
                tools_to_analyze, failing_tool_definitions
            )
            if missing_tools:
                analysis_for_display.append(
                    warn(
                        message=f"Missing tool definitions for failed tools: '{missing_tools}' in the file: '{tool_definition_path}'"
                    )
                )

            for tool_definition in failing_tool_definitions:

                tool_analysis = self._analyze_tool_definition(
                    inspector=inspector,
                    tool_definition=tool_definition,
                    tool_definition_path=tool_definition_path,
                )

                self.analysis_cache[tool_definition.tool_name] = tool_analysis
                analysis_for_display.extend(tool_analysis)

        return analysis_for_display

    def render(
        self, data: List[ExtendedMessage], tool_definition_path: Optional[str]
    ) -> Group:
        """
        Render the conversation history and analysis results.
        :param data: List of ExtendedMessage objects containing the conversation history.
        :param tool_definition_path: Path to the tool definition file.
        :return: A rich Group object containing the conversation and analysis results.
        """
        conversation_lines = []
        reason_lines = []
        failing_tools = []

        for entry in data:
            msg = entry.message
            role = msg.role
            content = msg.content
            reason = entry.reason
            tool_name = None
            if (
                msg.type == ContentType.tool_call
                or msg.type == ContentType.tool_response
            ):
                tool_name = json.loads(msg.content)["name"]

            if role == "user":
                label = "👤 User"
            elif role == "assistant" and msg.type == ContentType.tool_call:
                if reason:
                    label = "❌ Tool Call"

                    if reason.get("reason") == "incorrect parameter":
                        failing_tools.append(
                            tool_name
                        )  # create a list of failing tools for description quality analysis.
                else:
                    label = "✅ Tool Call"
            elif role == "assistant":
                label = "🤖 Assistant"
            else:
                label = "📦 Unknown"

            text_line = Text(f"{label}: {content}\n")
            if reason:
                text_line.stylize("bold red")
                reason_text = f"❌ {tool_name}: {json.dumps(reason)}\n\n"
                reason_lines.append(Text(reason_text, style="red"))
            conversation_lines.append(text_line)

        if failing_tools and tool_definition_path:

            inspector = DescriptionQualityInspector()

            description_quality_inspection_lines = (
                self.analyze_failing_tool_description_quality(
                    inspector, tool_definition_path, set(failing_tools)
                )
            )

            print_done()

            if description_quality_inspection_lines:
                reason_lines.extend(description_quality_inspection_lines)

        conversation_panel = Panel(
            Text().join(conversation_lines),
            title="Conversation History",
            border_style="blue",
        )
        reason_panel = Panel(
            Text().join(reason_lines),
            title="Analysis Results",
            border_style="red",
        )

        return Group(
            conversation_panel,
            reason_panel,
        )

    def analyze(self, config: AnalyzeConfig):
        """
        Analyze the results of the tool calls and routing metrics.
        :param config: AnalyzeConfig object containing user provided paths for analysis.
        """

        def get_summary(summary_file_name: str = "summary_metrics.csv"):
            summary = []

            path_to_summary_file = os.path.join(
                config.data_path, summary_file_name
            )

            with open(path_to_summary_file, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    summary.append(dict(zip(header, row)))

            return summary

        def get_test_messages(test_case_name):
            test_messages = []

            test_case_path = os.path.join(
                config.data_path,
                "messages",
                f"{test_case_name}.messages.analyze.json",
            )

            with open(test_case_path, "r", encoding="utf-8") as f:
                temp = json.load(f)
                for entry in temp:
                    msg = ExtendedMessage(**entry)
                    test_messages.append(msg)

            return test_messages

        def get_metrics(test_case_name):
            test_metrics_path = os.path.join(
                config.data_path, "messages", f"{test_case_name}.metrics.json"
            )

            with open(test_metrics_path, "r", encoding="utf-8") as f:
                metrics = ToolCallAndRoutingMetrics(**json.load(f))

            return metrics

        summary = get_summary()

        test_case_with_failed_tools = self._get_test_case_with_failed_tools(
            summary=summary
        )

        if len(test_case_with_failed_tools) == 0:
            header_table = Table(show_header=False, box=None)

            header_table.add_row("No Tool Call Error found!")

            panel = Panel(
                header_table,
                title="[bold green]📋 Analysis Summary[/bold green]",
            )

            pretty_print(panel)

        for test_case_entry in test_case_with_failed_tools:
            test_case_name = test_case_entry["dataset_name"]

            test_messages = get_test_messages(test_case_name=test_case_name)

            metrics: ToolCallAndRoutingMetrics = get_metrics(
                test_case_name=test_case_name
            )

            header_panel = self._create_header_analysis_panel(
                test_case_name, metrics
            )
            pretty_print(header_panel)

            tool_definition_path = (
                config.tool_definition_path
                if config.tool_definition_path
                else None
            )

            rendered_content = self.render(
                data=test_messages, tool_definition_path=tool_definition_path
            )
            pretty_print(rendered_content)

            add_line_seperator(self._generate_style_config())

    def _create_header_analysis_panel(
        self, test_case_name: str, metrics: ToolCallAndRoutingMetrics
    ) -> Panel:
        header_table = Table(show_header=False, box=None)

        header_table.add_row(f"Test Case Name: {test_case_name}")
        header_table.add_row(
            f"Expected Tool Calls: {metrics.expected_tool_calls}"
        )
        header_table.add_row(
            f"Correct Tool Calls: {metrics.correct_tool_calls}"
        )
        header_table.add_row(f"Text Match: {metrics.text_match.value}")
        header_table.add_row(f"Journey Success: {metrics.is_success}")

        header_panel = Panel(
            header_table, title="[bold green]📋 Analysis Summary[/bold green]"
        )

        return header_panel

    def _get_test_case_with_failed_tools(self, summary) -> List:

        test_case_with_failed_tools = []

        for entry in summary:
            test_case_name = entry["dataset_name"]

            if test_case_name.lower().strip() == "summary (average)":
                continue

            if (
                not entry["is_success"]
                or float(entry["tool_calls_with_incorrect_parameter"]) > 0
                or float(entry["tool_call_precision"]) < 1.0
                or float(entry["tool_call_recall"]) < 1.0
            ):

                test_case_with_failed_tools.append(entry)

        return test_case_with_failed_tools

    def _get_tools_not_found_in_source(
        self,
        tools_to_analyze: List[str],
        failing_tool_definitions: List[ToolDefinition],
    ) -> Set[str]:

        return set(tools_to_analyze) - {
            tool_def.tool_name for tool_def in failing_tool_definitions
        }

    def _analyze_tool_definition(
        self,
        inspector: DescriptionQualityInspector,
        tool_definition: ToolDefinition,
        tool_definition_path: str,
    ) -> List[Text]:

        tool_name = tool_definition.tool_name
        tool_desc = tool_definition.tool_description

        tool_analysis = []

        # missing description
        if tool_desc is None:
            tool_analysis.extend(
                IncorrectParameterUtils.format_missing_description_message(
                    tool_name=tool_name,
                    tool_definition_path=tool_definition_path,
                )
            )
            return tool_analysis

        # bad description
        if inspector.detect_bad_description(tool_definition):
            tool_analysis.extend(
                IncorrectParameterUtils.format_bad_description_message(
                    tool_name=tool_name, tool_desc=tool_desc
                )
            )
            return tool_analysis

        # good description
        tool_analysis.append(
            is_ok(
                message=f"The description for the `{tool_name}` looks sufficient."
            )
        )
        return tool_analysis


if __name__ == "__main__":
    dummy_analyzer = Analyzer()
    dummy_analyzer.analyze(CLI(AnalyzeConfig, as_positional=False))

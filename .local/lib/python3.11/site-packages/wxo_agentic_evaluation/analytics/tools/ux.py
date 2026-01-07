import json
from typing import Dict, List, Optional

import rich
from analytics.tools.types import (
    AgentRecommendation,
    AnalysisResults,
    ErrorPatterns,
    Priority,
    ToolDefinitionRecommendation,
)
from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from wxo_agentic_evaluation.type import ContentType, Message


class ToolErrorDisplayManager:
    """Handles all display/UX functionality for tool error analysis."""

    CHARACTER_THRESHOLD = (
        200  # characters <- if the error_msg has fewer then it is not helpful.
    )

    def __init__(
        self,
        messages: List[Message],
        error_patterns: Optional[ErrorPatterns] = None,
    ):
        self.messages = messages
        self.error_patterns = error_patterns or ErrorPatterns()

    # Suggest tool definition improvements
    def generate_tool_definition_recommendations(
        self,
    ) -> List[ToolDefinitionRecommendation]:
        """Suggest improvements to the customer's tool definitions"""
        recommendations = []

        for tool, failures in self.error_patterns.all_failures.items():
            failure_counts = {
                "stringified_json_outputs": 0,  # should instead return native objects
                "parameter_type_validation": 0,  # no validation logic causing API errors
                "unhelpful_responses": 0,  # empty responses, not helpful to guide agent in a conversation
            }

            validation_error_codes = ["404", "not found", "client error"]
            unhelpful_resp_threshold = (
                ToolErrorDisplayManager.CHARACTER_THRESHOLD
            )

            for failure in failures:
                error_msg = str(failure.error_message).lower()

                if (
                    error_msg.strip() in ["[]", "{}", ""]
                    or len(error_msg) < unhelpful_resp_threshold
                ):
                    failure_counts["unhelpful_responses"] += 1

                if any(
                    err_code in error_msg for err_code in validation_error_codes
                ):
                    failure_counts["parameter_type_validation"] += 1

                if any(x in error_msg for x in ['"[', '{"', '"]', "}"]):
                    failure_counts["stringified_json_outputs"] += 1

            if failure_counts["unhelpful_responses"] > 0:
                recommendations.append(
                    ToolDefinitionRecommendation(
                        tool=tool,
                        issue="Unhelpful and Contextless Response",
                        recommendation="Return structured error messages or raise exceptions instead of empty responses",
                        priority=Priority.MEDIUM,
                        count=failure_counts["unhelpful_responses"],
                    )
                )

            if failure_counts["stringified_json_outputs"] > 0:
                recommendations.append(
                    ToolDefinitionRecommendation(
                        tool=tool,
                        issue="Stringified JSON output",
                        recommendation="Return native Python objects instead of JSON strings for better type safety",
                        priority=Priority.LOW,
                        count=failure_counts["stringified_json_outputs"],
                    )
                )

            if failure_counts["parameter_type_validation"] > 0:
                recommendations.append(
                    ToolDefinitionRecommendation(
                        tool=tool,
                        issue="Parameter type validation issues",
                        recommendation="Add validation to ensure correct parameter types are passed. Return clear errors when wrong types received.",
                        priority=Priority.HIGH,
                        count=failure_counts["parameter_type_validation"],
                        example="-",
                    )
                )

        # Sort by priority (high -> medium -> low)
        priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        recommendations.sort(key=lambda x: priority_order[x.priority])

        return recommendations

    def create_individual_testcase_header_analysis(
        self,
        base_name: str,
        results: AnalysisResults,
        tool_def_recs: List[ToolDefinitionRecommendation],
    ) -> None:
        """Display comprehensive analysis using analyzed results."""

        all_failures = results.error_patterns.all_failures
        repeated_failures = results.error_patterns.repeated_failures

        tool_def_recs_count = len(tool_def_recs)

        # Calculate accurate statistics from analyzed results
        total_failed_tools = len(
            all_failures
        )  # unique tools that failed atleast once
        total_failure_instances = sum(
            len(failures) for failures in all_failures.values()
        )  # individual failures across all tools, the same tool may have multiple failure instances
        repeated_failure_tools = len(
            repeated_failures
        )  # number of tools that only failed >= threshold (=2)

        # Create tool status lookup from analyzed results
        failed_tool_calls = set()
        for tool, failures in all_failures.items():
            for failure in failures:
                failed_tool_calls.add(failure.attempt_index)

        header_table = Table(show_header=False, box=None)
        header_table.add_row("📊 Test Case:", f"[bold]{base_name}[/bold]")
        header_table.add_row(
            "🔧 Total Tools Used (unique):",
            str(len(self._get_all_tools(results))),
        )
        header_table.add_row(
            "❌ Failed Tools (unique):", str(total_failed_tools)
        )
        header_table.add_row(
            "🔥 Total Failure Instances (not unique):",
            str(total_failure_instances),
        )
        header_table.add_row(
            "🔄 Repeated Failures:", str(repeated_failure_tools)
        )
        header_table.add_row(
            "🔨 Tool Definition Recommendations:", str(tool_def_recs_count)
        )
        header_table.add_row(
            "🤖 Agent Template Recommendations:",
            str(len(results.recommendations)),
        )

        header_panel = Panel(
            header_table, title="[bold green]📋 Analysis Summary[/bold green]"
        )

        layout = Layout()
        layout.split_row(
            Layout(
                self._display_conversation(failed_tool_calls),
                name="conversation",
            ),
            Layout(
                self._create_detailed_analysis_panel(results), name="analysis"
            ),
        )

        rich.print(header_panel)
        rich.print(layout)

    def _display_conversation(self, failed_tool_calls: set) -> Panel:
        """Display conversation with color coding for erreneous calls."""

        conversation_content = []

        for i, msg in enumerate(self.messages):
            if msg.role == "user":
                conversation_content.append(
                    f"[bold blue]👤 User:[/bold blue] {msg.content}"
                )
            elif msg.role == "assistant":
                if msg.type == ContentType.tool_call:
                    is_failed = i in failed_tool_calls
                    color = "red" if is_failed else "green"
                    icon = "❌" if is_failed else "✅"

                    conversation_content.append(
                        f"[bold {color}]{icon} Tool Call:[/bold {color}] {msg.content}"
                    )
                elif msg.type == ContentType.tool_response:
                    is_error_response = (i + 1) in failed_tool_calls
                    color = "red" if is_error_response else "green"
                    icon = "⚠️" if is_error_response else "🔧"

                    # Truncate long responses
                    content = str(msg.content)
                    if len(content) > 300:
                        content = content[:300] + "[bold](...)[/bold]"

                    conversation_content.append(
                        f"[{color}]{icon} Response:[/{color}] {content}"
                    )
                else:
                    conversation_content.append(
                        f"[bold cyan]🤖 Assistant:[/bold cyan] {msg.content}"
                    )

        return Panel(
            "\n".join(conversation_content),
            title="[bold]📱 Conversation History[/bold]",
            border_style="blue",
        )

    def _create_detailed_analysis_panel(
        self, results: AnalysisResults
    ) -> Panel:
        """Creates the analysis panel."""

        content = []

        if results.error_patterns.repeated_failures:
            error_table = Table(title="🔄 Repeated Failures")
            error_table.add_column("Tool", style="cyan")
            error_table.add_column("Attempts", justify="center")
            error_table.add_column("Error Type", style="red")

            for (
                tool,
                failures,
            ) in results.error_patterns.repeated_failures.items():
                # Use the analyzed error classification
                error_snippet = str(failures[-1].error_message)[:50] + "..."
                error_table.add_row(tool, str(len(failures)), error_snippet)

            content.append(error_table)

        causes = results.root_causes
        root_cause_data = {
            "incorrect_parameter_usage": causes.incorrect_parameter_usage,
            "bad_tool_call": causes.bad_tool_call,
            "agent_hallucinations": causes.agent_hallucinations,
        }
        if any(root_cause_data.values()):
            cause_table = Table(title="🎯 Root Cause Analysis")
            cause_table.add_column("Category", style="bold")
            cause_table.add_column("Count", justify="center")
            cause_table.add_column("Tools Affected", style="yellow")

            for category, issues in root_cause_data.items():
                if issues:
                    affected_tools = {issue.tool for issue in issues}
                    tools_str = ", ".join(
                        list(affected_tools)[:3]
                    )  # Limit display
                    if len(affected_tools) > 3:
                        tools_str += f"... (+{len(affected_tools)-3} more)"

                    cause_table.add_row(
                        category.replace("_", " ").title(),
                        str(len(issues)),
                        tools_str,
                    )

            content.append(cause_table)

        if results.recommendations:
            content.append(
                self._create_recommendations_display(results.recommendations)
            )

        # Add tool definition status table
        tool_def_recs = self.generate_tool_definition_recommendations()
        if tool_def_recs:
            tool_def_table = Table(title="🔧 Tool Definition Status")
            tool_def_table.add_column("Tool Name", style="cyan")
            tool_def_table.add_column("Status", style="bold")

            # Get unique tools with issues
            tools_with_issues = {rec.tool for rec in tool_def_recs}

            # Show all tools from failures
            for tool in results.error_patterns.all_failures.keys():
                if tool in tools_with_issues:
                    issue_count = len(
                        [r for r in tool_def_recs if r.tool == tool]
                    )
                    tool_def_table.add_row(
                        tool, f"[red]❌ {issue_count} issue(s)[/red]"
                    )
                else:
                    tool_def_table.add_row(tool, "[green]✅ OK[/green]")

            content.append(tool_def_table)

        return Panel(
            Group(*content),
            title="[bold red]🔍 Analysis Results[/bold red]",
            border_style="red",
        )

    def _create_recommendations_display(
        self, recommendations: List[AgentRecommendation]
    ) -> Table:
        """Create prioritized recommendations table."""
        rec_table = Table(title="💡 Improvement Recommendations")
        rec_table.add_column("Priority", style="bold")
        rec_table.add_column("Issue", style="yellow")
        rec_table.add_column("Suggested Fix", style="green")

        # Sort recommendations by priority
        prioritized_recs = self._prioritize_recommendations(recommendations)

        for i, rec in enumerate(prioritized_recs, 1):
            priority = "📝 LOW" if i <= 2 else "⚡ MED" if i <= 5 else "🔥 HIGH"

            rec_table.add_row(
                priority,
                rec.issue,
                "--",  # rec.prompt_addition can be added here when ready
            )

        return rec_table

    def generate_executive_summary(
        self,
        all_results: Dict[str, AnalysisResults],
        all_tool_def_recs: List[ToolDefinitionRecommendation],
    ) -> None:
        """Generate executive summary across all test cases with real tool call metrics."""

        total_tool_definition_recs = len(all_tool_def_recs)

        # 1. Identify failing test cases and their tool failure counts
        failing_test_cases = {}
        for test_case, results in all_results.items():
            failed_tools_count = len(results.error_patterns.all_failures)
            if failed_tools_count > 0:
                failing_test_cases[test_case] = failed_tools_count

        # 2. Count total failed tool calls across all test cases
        total_failed_tool_calls = sum(
            sum(
                len(failures)
                for failures in r.error_patterns.all_failures.values()
            )
            for r in all_results.values()
        )

        # 3. Get total tool calls from stored data (we'll add this to results)
        total_tool_calls = sum(
            r.total_tool_calls or 0 for r in all_results.values()
        )

        # 4. Calculate successful tool calls and success rate
        successful_tool_calls = total_tool_calls - total_failed_tool_calls
        success_rate = (
            (successful_tool_calls / total_tool_calls * 100)
            if total_tool_calls > 0
            else 100
        )

        # 5. Other metrics
        total_cases = len(all_results)
        total_agent_template_recs = sum(
            len(r.recommendations) for r in all_results.values()
        )

        # Create failing test cases display
        failing_cases_text = ""
        if failing_test_cases:
            failing_cases_text = (
                "\n[bold red]📋 Failing Test Cases:[/bold red]\n"
            )
            for test_case, failed_tool_count in sorted(
                failing_test_cases.items()
            ):
                failing_cases_text += f"  • [red]{test_case}[/red]: [bold]{failed_tool_count}[/bold] failing tool(s)\n"
        else:
            failing_cases_text = (
                "\n[bold green]🎉 All test cases passed![/bold green]\n"
            )

        # Disclaimer text
        disclaimer_text = """[bold red]⚠️ IMPORTANT DISCLAIMER:[/bold red]
    [yellow]The guidelines above are based on observed error patterns and are intended to help you identify potential improvements to your agent setup.
    They are not exact fixes, but rather general suggestions drawn from common failure modes.

    Please use them as starting points to inform your review process — not as definitive instructions.
    Effectiveness may vary depending on your domain, agent behavior, and tool configuration.

    [bold red]We do not recommend copying these statements directly into your prompts or tool definitions.[/bold red]
    Instead, adapt the insights to fit the context of your use case, and validate any changes before deployment.[/yellow]"""

        summary_text = f"""
    [bold green]🎯 EXECUTIVE SUMMARY[/bold green]

    📊 [bold]Test Cases Analyzed:[/bold] {total_cases}
    🔧 [bold]Total Tool Calls Made [italic](across all test cases)[/italic]:[/bold] {total_tool_calls}
    ✅ [bold]Successful Tool Calls [italic](calls that completed without error across all test cases)[/italic]:[/bold] {successful_tool_calls}
    ❌ [bold]Failed Tool Calls [italic](calls that generated errors across all test cases)[/italic]:[/bold] {total_failed_tool_calls}
    🤖 [bold]Agent Template Recommendations Suggested:[/bold] {total_agent_template_recs}
    🔨 [bold]Tool Definition Recommendations Suggested:[/bold] {total_tool_definition_recs}

    [yellow]📈 Success Rate = [italic](across all test cases) successful tool calls / total tool calls[/italic]:[/yellow] [bold bright_cyan]{success_rate:.1f}%[/bold bright_cyan]
    {failing_cases_text}
    [bold cyan]🚀 Next Steps:[/bold cyan]
    1. Implement high-priority prompt improvements
    2. Review agent tool usage patterns  
    3. Update ground truth data where needed
    """  # disclaimer_text can be embedded here when recommendations are ready

        rich.print(
            Panel(Align.center(summary_text), border_style="green", padding=1)
        )

    def _prioritize_recommendations(
        self, recommendations: List[AgentRecommendation]
    ) -> List[AgentRecommendation]:
        """Sort recommendations by priority based on issue type."""
        priority_order = {
            "Agent hallucinated": 1,
            "Agent repeatedly fails": 2,
            "Resource not found": 3,
            "Using placeholder": 4,
            "Parameter format": 5,
            "Authentication": 6,
            "Bad request": 7,
            "API errors": 8,
        }

        def get_priority(rec):
            for key_phrase, priority in priority_order.items():
                if key_phrase.lower() in rec.issue.lower():
                    return priority
            return 1  # Default priority for unmatched issues

        return sorted(recommendations, key=get_priority)

    def _get_all_tools(self, results: AnalysisResults) -> List[str]:
        """Extract tools from analyzed results rather than re-parsing messages."""

        all_tools_in_conversation = set()  # unique calls
        for i, msg in enumerate(self.messages):
            if msg.type == ContentType.tool_call:
                # Extract tool name safely
                try:
                    if isinstance(msg.content, str):
                        tool_call = json.loads(msg.content)
                    else:
                        tool_call = msg.content

                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "")
                        if tool_name:
                            all_tools_in_conversation.add(tool_name)
                except (json.JSONDecodeError, AttributeError):
                    continue

        return list(all_tools_in_conversation)

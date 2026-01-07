import argparse
import json
from pathlib import Path
from shutil import get_terminal_size

import rich
from analytics.tools.analyzer import ToolErrorAnalyzer
from analytics.tools.ux import ToolErrorDisplayManager
from type import ContentType
from utils.utils import load_messages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tool-analytics-resources")
    parser.add_argument(
        "--messages_dir",
        type=Path,
        help="Path to `messages` folder in the output directory containing your evaluation artifacts. For reference, the output directory is specified by `output_dir` either in your evaluation configuration YAML file, or when passed in via the command line",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=Path,
        help="Path to your ground truth directory containing the expected results for your evaluation. This should contain JSON files with the same base names as your messages files, e.g. `data1.messages.json`: message file and `data1.json`: ground truth file.",
    )

    args = parser.parse_args()

    messages_dir = args.messages_dir
    ground_truth_dir = args.ground_truth_dir
    if not messages_dir or not ground_truth_dir:
        rich.print(f"[red]Error: \n[/red]")
        rich.print(
            "[yellow]Please ensure you provide valid paths for --messages_dir and --ground_truth_dir[/yellow]"
        )
        exit(1)

    # Check terminal size and prompt user to resize if needed
    term_width = get_terminal_size().columns
    if term_width < 180:
        rich.print(
            f"[yellow]⚠️ Terminal width is only {term_width} characters.[/yellow]"
        )
        rich.print(
            "[cyan]Consider expanding your terminal window [bold]to full screen[/bold] for better layout and readability.[/cyan]\n"
        )
        input("Press Enter to continue once you've resized the terminal")

    def count_tool_calls(messages) -> int:
        """Count total tool calls in the conversation."""
        return sum(1 for msg in messages if msg.type == ContentType.tool_call)

    # Function to load ground truth from JSON file
    def load_ground_truth(file_path):
        with open(file_path, "r") as f:
            ground_truth_data = json.load(f)
        return ground_truth_data

    # Find all message files
    message_files = list(messages_dir.glob("*.messages.json"))
    if not message_files:
        message_files = list(messages_dir.glob("*.json"))
    message_files = sorted(message_files)
    rich.print(
        f"[bold green]Found {len(message_files)} message files to analyze[/bold green]"
    )

    all_results = {}
    all_tool_def_recs = []

    for message_file in message_files:
        # Extract base name to find matching ground truth
        base_name = message_file.stem
        if base_name.endswith(".messages"):
            base_name = base_name.replace(".messages", "")

        # Find matching ground truth file
        ground_truth_file = next(
            ground_truth_dir.glob(f"{base_name}.json"), None
        )

        if ground_truth_file:
            rich.print(f"\n[bold cyan]Analyzing: {base_name}[/bold cyan]")

            # Load data
            messages = load_messages(message_file)
            if not messages:
                continue
            ground_truth = load_ground_truth(ground_truth_file)

            # Run analysis
            analyzer = ToolErrorAnalyzer(
                messages=messages, ground_truth=ground_truth
            )
            results = analyzer.analyze()
            display_manager = ToolErrorDisplayManager(
                messages=messages, error_patterns=results.error_patterns
            )

            # Count tool calls and store in results
            results.total_tool_calls = count_tool_calls(messages)

            tool_def_recs = (
                display_manager.generate_tool_definition_recommendations()
            )
            all_tool_def_recs.extend(tool_def_recs)

            # Display results
            error_count = len(results.error_patterns.all_failures)
            repeat_count = len(results.error_patterns.repeated_failures)
            rec_count = len(results.recommendations)

            rich.print(
                f"[yellow]Results:[/yellow] {error_count} failing tools, {repeat_count} with repeated failures"
            )
            display_manager.create_individual_testcase_header_analysis(
                base_name, results, tool_def_recs
            )
            all_results[base_name] = results

            if rec_count > 0:
                rich.print(
                    "\n[bold magenta]🤖 Agent Template Recommendations:[/bold magenta]"
                )
                for rec in results.recommendations:
                    rich.print(f"• [bold]{rec.issue}[/bold]")
                    rich.print(
                        f"  [green]Guideline/Suggested Fix(es):[/green] --"
                    )  # rec.prompt_addition can be embedded here when ready
                    rich.print(
                        f"  [gold3][bold]Explanation:[/bold][/gold3] {rec.summary}"
                    )

            if tool_def_recs:
                rich.print(
                    "\n[bold blue]🔧 Tool Definition Improvements:[/bold blue]"
                )
                for rec in tool_def_recs:
                    rich.print(
                        f"• [bold]{rec.priority.value} {rec.tool}:[/bold] [yellow]{rec.issue}[/yellow]"
                    )
                    rich.print(
                        f"  [cyan]Fix:[/cyan] --"
                    )  # rec.recommendation can be embedded here when ready
                    if rec.example is not None:
                        rich.print(f"  [yellow]Example:[/yellow] {rec.example}")

            rich.print("\n" + "[grey70]=[/grey70]" * 100 + "\n")
        else:
            rich.print(
                f"\n[red][bold]No ground truth found for {base_name}[/bold][/red] - [yellow]🚨 SKIPPED[/yellow]"
            )

    # Final executive summary
    if all_results:
        display_manager.generate_executive_summary(
            all_results, all_tool_def_recs
        )
    rich.print("\n[bold green]Analysis complete![/bold green]")

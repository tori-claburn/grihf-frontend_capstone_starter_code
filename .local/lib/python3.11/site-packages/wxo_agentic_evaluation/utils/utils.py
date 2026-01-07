import glob
import json
import os
import re
from typing import List, Optional, Union
from urllib.parse import urlparse

import yaml
from rich import box, print
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table

from wxo_agentic_evaluation.metrics.llm_as_judge import Faithfulness
from wxo_agentic_evaluation.metrics.metrics import (
    KnowledgeBaseMetricSummary,
    ReferenceLessEvalMetrics,
)
from wxo_agentic_evaluation.type import (
    ConversationalConfidenceThresholdScore,
    Message,
)

console = Console()


class AttackResultsTable:
    def __init__(self, attack_results: dict):
        self.table = Table(
            title="Attack Results",
            box=box.ROUNDED,
            show_lines=True,
        )
        self.table.add_column("Attack Category", style="magenta")
        self.table.add_column("Count", style="cyan")
        self.table.add_column("Success Rate", style="green")

        # Extract values
        n_on_policy = attack_results.get("n_on_policy_attacks", 0)
        n_off_policy = attack_results.get("n_off_policy_attacks", 0)
        n_on_policy_successful = attack_results.get("n_on_policy_successful", 0)
        n_off_policy_successful = attack_results.get(
            "n_off_policy_successful", 0
        )

        # Calculate success rates
        on_policy_rate = (
            f"{round(100 * safe_divide(n_on_policy_successful, n_on_policy))}%"
            if n_on_policy
            else "0%"
        )
        off_policy_rate = (
            f"{round(100 * safe_divide(n_off_policy_successful, n_off_policy))}%"
            if n_off_policy
            else "0%"
        )

        self.table.add_row("On Policy", str(n_on_policy), on_policy_rate)
        self.table.add_row("Off Policy", str(n_off_policy), off_policy_rate)

    def print(self):
        console.print(self.table)


class AgentMetricsTable:
    def __init__(self, data):
        self.table = Table(
            title="Agent Metrics",
            box=box.ROUNDED,
            show_lines=True,
        )

        if not data:
            return

        # Add columns with styling
        headers = list(data[0].keys())
        for header in headers:
            self.table.add_column(header, style="cyan")

        # Add rows
        for row in data:
            self.table.add_row(*[str(row.get(col, "")) for col in headers])

    def print(self):
        console.print(self.table)


def create_table(data: List[dict]) -> AgentMetricsTable:
    """
    Generate a Rich table from a list of dictionaries.
    Returns the AgentMetricsTable instance.
    """
    if isinstance(data, dict):
        data = [data]

    if not data:
        print("create_table() received an empty dataset. No table generated.")
        return None

    return AgentMetricsTable(data)


def safe_divide(nom, denom):
    if denom == 0:
        return 0
    else:
        return nom / denom


def is_saas_url(service_url: str) -> bool:
    hostname = urlparse(service_url).hostname
    return hostname not in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def is_ibm_cloud_url(service_url: str) -> bool:
    hostname = urlparse(service_url).hostname
    return ".cloud.ibm.com" in hostname


def add_line_seperator(
    style_config: Optional[Union[str, Style]] = None,
):

    if not style_config:
        style = "grey42"
    else:
        style = style_config

    console.print(
        Rule(
            style=style,
        )
    )


class FaithfulnessTable:
    def __init__(
        self, faithfulness_metrics: List[Faithfulness], tool_call_ids: List[str]
    ):
        self.table = Table(
            title="Faithfulness", box=box.ROUNDED, show_lines=True
        )

        self.table.add_column("Tool Call Id", style="blue")
        self.table.add_column("Faithfulness Score", style="blue3")
        self.table.add_column("Evidence", style="cyan")
        self.table.add_column("Reasoning", style="yellow3")

        for tool_call_id, faithfulness in zip(
            tool_call_ids, faithfulness_metrics
        ):
            faithfulness = faithfulness.table()
            self.table.add_row(
                tool_call_id,
                faithfulness["faithfulness_score"],
                faithfulness["evidence"],
                faithfulness["reason"],
            )

    def print(self):
        console.print(self.table)


class ConversationalSearchTable:
    def __init__(
        self,
        confidence_scores_list: List[ConversationalConfidenceThresholdScore],
        tool_call_ids: List[str],
    ):
        self.table = Table(
            title="Conversational Search", box=box.ROUNDED, show_lines=True
        )

        self.table.add_column("Tool Call Id", style="blue")
        self.table.add_column("Response Confidence", style="blue3")
        self.table.add_column("Response Confidence Threshold", style="cyan")
        self.table.add_column("Retrieval Confidence", style="blue3")
        self.table.add_column("Retrieval Confidence Threshold", style="cyan")

        for tool_call_id, confidence_scores in zip(
            tool_call_ids, confidence_scores_list
        ):
            confidence_scores = confidence_scores.table()
            self.table.add_row(
                tool_call_id,
                confidence_scores["response_confidence"],
                confidence_scores["response_confidence_threshold"],
                confidence_scores["retrieval_confidence"],
                confidence_scores["retrieval_confidence_threshold"],
            )


class KnowledgePanel:
    def __init__(
        self,
        dataset_name: str,
        tool_call_id: List[str],
        faithfulness: List[Faithfulness] = None,
        confidence_scores: List[ConversationalConfidenceThresholdScore] = None,
    ):
        self.faithfulness = FaithfulnessTable(faithfulness, tool_call_id)
        self.confidence_scores = ConversationalSearchTable(
            confidence_scores, tool_call_id
        )
        self.group = Group(
            self.faithfulness.table, self.confidence_scores.table
        )

        # Panel acts as a section
        self.section = Panel(
            self.group,
            title=f"Agent with Knowledge Metrics for {dataset_name}",
            border_style="grey37",
            title_align="left",
        )

    def print(self):
        console.print(self.section)


class SummaryPanel:
    def __init__(self, summary_metrics: KnowledgeBaseMetricSummary):

        self.table = Table(
            title="Agent with Knowledge Summary Metrics",
            box=box.ROUNDED,
            show_lines=True,
        )
        self.table.add_column("Dataset", style="blue3")
        self.table.add_column("Average Response Confidence", style="cyan")
        self.table.add_column("Average Retrieval Confidence", style="blue3")
        self.table.add_column("Average Faithfulness", style="cyan")
        self.table.add_column("Average Answer Relevancy", style="blue3")
        self.table.add_column("Number Calls to Knowledge Bases", style="cyan")
        self.table.add_column("Knowledge Bases Called", style="blue3")

        average_metrics = summary_metrics.average
        for dataset, metrics in average_metrics.items():
            self.table.add_row(
                dataset,
                str(round(metrics["average_response_confidence"], 4)),
                str(round(metrics["average_retrieval_confidence"], 4)),
                str(metrics["average_faithfulness"]),
                str(metrics["average_answer_relevancy"]),
                str(metrics["number_of_calls"]),
                metrics["knowledge_bases_called"],
            )

    def print(self):
        console.print(self.table)


class Tokenizer:
    PATTERN = r"""
            \w+(?=n't)|              # Words before n't contractions (e.g., "do" in "don't")
            n't|                     # n't contractions themselves
            \w+(?=')|                # Words before apostrophes (e.g., "I" in "I'm")
            '|                       # Apostrophes as separate tokens
            \w+|                     # Regular words (letters, numbers, underscores)
            [^\w\s]                  # Punctuation marks (anything that's not word chars or whitespace)
        """

    def __init__(self):
        self.compiled_pattern = re.compile(
            self.PATTERN, re.VERBOSE | re.IGNORECASE
        )

    def __call__(self, text: str) -> List[str]:
        """
        Tokenizes text by splitting on punctuation and handling contractions.

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokenized words (lowercase, no punctuation).

        Examples:
            - "I'm fine"      -> ['i', 'm', 'fine']
            - "don't go"      -> ['do', "n't", 'go']
            - "Hello, world!" -> ['hello', 'world']
        """

        tokens = self.compiled_pattern.findall(text)

        return self._clean_tokens(tokens)

    def _clean_tokens(self, raw_tokens: List[str]) -> List[str]:
        """
        Applies some basic post-processing to tokenized messages.

        Args:
            raw_tokens: list of tokens extracted from a message.
        """

        filtered_tokens = [
            token.lower()
            for token in raw_tokens
            if token.strip() and not (len(token) == 1 and not token.isalnum())
        ]

        return filtered_tokens


class ReferencelessEvalPanel:
    def __init__(self, referenceless_metrics: List[ReferenceLessEvalMetrics]):
        self.table = Table(
            title="Quick Evaluation Summary Metrics",
            box=box.ROUNDED,
            show_lines=True,
        )

        self.table.add_column("Dataset", style="yellow", justify="center")
        self.table.add_column(
            "Tool Calls", style="deep_sky_blue1", justify="center"
        )
        self.table.add_column(
            "Successful Tool Calls", style="magenta", justify="center"
        )
        self.table.add_column(
            "Tool Calls Failed due to Schema Mismatch",
            style="deep_sky_blue1",
            justify="center",
        )
        self.table.add_column(
            "Tool Calls Failed due to Hallucination",
            style="magenta",
            justify="center",
        )

        for metric in referenceless_metrics:
            self.table.add_row(
                str(metric.dataset_name),
                str(metric.number_of_tool_calls),
                str(metric.number_of_successful_tool_calls),
                str(metric.number_of_static_failed_tool_calls),
                str(metric.number_of_semantic_failed_tool_calls),
            )

    def print(self):
        console.print(self.table)


# Function to load messages from JSON file
def load_messages(file_path):
    with open(file_path, "r") as f:
        try:
            message_data = json.load(f)
            messages = []
            for msg in message_data:
                messages.append(Message.model_validate(msg))

            return messages

        except Exception as e:
            print(file_path)
            print(e)
            return None


def load_agents(agents_path: str):
    agents_json = glob.glob(os.path.join(agents_path, "*.json"))
    agents_yaml = glob.glob(os.path.join(agents_path, "*.yaml"))

    agents = []

    for agent_path in agents_json:
        with open(agent_path, "r") as f:
            agents.append(json.load(f))

    for agent_path in agents_yaml:
        with open(agent_path, "r") as f:
            agents.append(yaml.safe_load(f))

    return agents

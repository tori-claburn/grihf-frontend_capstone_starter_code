from typing import Any, List, Optional

import rich
from rich.text import Text


def pretty_print(content: Any, style: Optional[str] = None):
    """
    Utility function for stylized prints.
    Please refer to: https://rich.readthedocs.io/en/stable/appendix/colors.html for valid  `style` strings.
    NOTE:
        Rich allows for nested [style][/style] tags within a string.
        This utility only applies an outermost style wrapper using the passed `style` (ONLY for a string `content`).

    :param content: The content to be printed
    :param style: a valid `rich` colour.
    """
    if isinstance(content, str):
        if style:
            rich.print(f"[{style}]{content}[/{style}]")
        else:
            rich.print(content)
    else:
        rich.print(content)


def warn(
    message: str,
    style: Optional[str] = "bold yellow",
    prompt: Optional[str] = "WARNING ⚠️ :",
) -> Text:
    """Utility function for formatting a warning message."""
    return Text(f"{prompt}{message}\n\n", style=style)


def is_ok(
    message: str,
    style: Optional[str] = "bold green",
    prompt: Optional[str] = "OK ✅ :",
) -> Text:
    """Utility function for formatting an OK message."""
    return Text(f"{prompt}{message}\n\n", style=style)


def print_done(
    prompt: Optional[str] = "Done ✅", style: Optional[str] = "bold cyan"
):
    """
    Prints a prompt indicating completion of a process/routine.
    :param prompt: default is `"Done ✅"`
    :param style: The style for the text (default is bold cyan).
    """
    pretty_print(content=prompt, style=style)


def print_success(
    message: str,
    style: Optional[str] = "bold green",
    prompt: Optional[str] = "✅ PASSED",
):
    """
    Prints a success message.
    :param message: a statement that is printed alongside a PASSED outcome.
    :param style: The style for the text (default is bold green).
    :param prompt: The prompt to display before the message (default is "✅ PASSED").
    """
    pretty_print(content=f"{prompt} - {message}", style=style)


def print_failure(
    message: str,
    style: Optional[str] = "bold red",
    prompt: Optional[str] = "❌ FAILED",
):
    """
    Prints a failure message.
    :param message: a statement that is printed alongside a FAILED outcome.
    :param style: The style for the text (default is bold red).
    :param prompt: The prompt to display before the message (default is "❌ FAILED").
    """
    pretty_print(content=f"{prompt} - {message}", style=style)


class IncorrectParameterUtils:
    """
    Utility functions for handling warning and suggestion messages related to bad parameters in tool descriptions.
    These are primarily used for providing feedback on incorrect parameter usage by the assistant in `analyze_run`.
    """

    @staticmethod
    def suggest(message: str, style: Optional[str] = "green") -> Text:
        """
        Used for formatting a suggestion message for improving agent behaviour relating to bad parameter usage.
        :param message: The suggestion message to display.
        :param style: The style for the text (default is green).
        :return: A rich Text object styled as a suggestion.
        """
        return Text(
            f"💡 {message}\n✅ A good description is insightful of the tool's purpose, and clarifies parameter usage to the assistant.\n\n",
            style=style,
        )

    @staticmethod
    def format_missing_description_message(
        tool_definition_path: str, tool_name: str
    ) -> List[Text]:

        return [
            warn(
                f"Tool description for '{tool_name}' not found in file: '{tool_definition_path}'"
            ),
            IncorrectParameterUtils.suggest(
                f"Please consider adding a description for '{tool_name}'."
            ),
        ]

    @staticmethod
    def format_bad_description_message(
        tool_name: str, tool_desc: str
    ) -> List[Text]:

        return [
            warn(
                f"Tool description for '{tool_name}' may be incomplete or unclear: '{tool_desc.strip()}'."
            ),
            IncorrectParameterUtils.suggest(
                f"Please consider making the description for '{tool_name}' more informative on parameter usage."
            ),
        ]


class TestingUtils:
    """
    Provides a collection of formatted messages that can be used in testing workflows.
    """

    @staticmethod
    def print_test_header(
        test_case_count: int,
        test_description: str,
        style: Optional[str] = "bold cyan",
        prompt: Optional[str] = "\n⚙️ Testing",
    ):
        """
        Print formatted test suite header.
        :param test_case_count: # of test-cases.
        :param test_description: a short statement explaining what is being examined.
        For example, this can be read as: `"{\n⚙️ Testing} {20} {good tool descriptions}"`.
        """
        pretty_print(
            content=f"{prompt} {test_case_count} {test_description}",
            style=style,
        )

    @staticmethod
    def print_error_details(
        expected: List[str],
        detected: List[str],
        style: Optional[str] = "bold red",
    ):
        """
        Print detailed error information.
        An error in this context can be an assertion mis-match.
        Use this function to display the delta.
        :param expected: the expected outcome.
        :param detected: the actual/observed outcome.
        """
        pretty_print(content=f"   Expected: {expected}", style=style)
        pretty_print(content=f"   Detected: {detected}", style=style)

    @staticmethod
    def print_failure_summary(
        failed_cases: List[str],
        prompt: Optional[str] = "Failed cases",
        style: Optional[str] = "bold red",
    ):
        """
        Print summary of all failures.
        List out the specific cases that failed the test.
        :param failed_cases: List of failed case names, this list is iterated over to print/list all failures.
        :param style: The style for the text (default is bold red).
        """
        if failed_cases:
            pretty_print(
                content=f"{prompt} ({len(failed_cases)}):", style=style
            )
            for case in failed_cases:
                pretty_print(content=f"  - {case}", style=style)

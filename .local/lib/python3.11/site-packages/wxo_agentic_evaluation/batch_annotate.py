import ast
import csv
import json
import os
from pathlib import Path

from jsonargparse import CLI

from wxo_agentic_evaluation import __file__
from wxo_agentic_evaluation.arg_configs import BatchAnnotateConfig
from wxo_agentic_evaluation.prompt.template_render import (
    BatchTestCaseGeneratorTemplateRenderer,
)
from wxo_agentic_evaluation.service_provider import get_provider

root_dir = os.path.dirname(__file__)
BATCH_TEST_CASE_GENERATOR_PROMPT_PATH = os.path.join(
    root_dir, "prompt", "batch_testcase_prompt.jinja2"
)
EXAMPLE_PATH = os.path.join(root_dir, "prompt", "examples", "data_simple.json")


def parse_tools_with_filter(
    agent_name: str, tools_path: Path, allowed_tool_names: list[str]
) -> tuple[dict, list[dict]]:
    if not allowed_tool_names:
        raise ValueError("Allowed tool list cannot be empty.")

    tool_data = []
    files_to_parse = []

    # Handle both single file and directory cases
    if tools_path.is_file():
        files_to_parse.append(tools_path)
    elif tools_path.is_dir():
        files_to_parse.extend(tools_path.glob("**/*.py"))
    else:
        raise ValueError(
            f"Tools path {tools_path} is neither a file nor directory"
        )

    for file_path in files_to_parse:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                tools_code = f.read()

            parsed_code = ast.parse(tools_code)

            # Process only module-level functions
            for node in parsed_code.body:
                if isinstance(node, ast.FunctionDef):
                    tool_data.append(
                        {
                            "Function Name": node.name,
                            "Arguments": [arg.arg for arg in node.args.args],
                            "Docstring": ast.get_docstring(node),
                        }
                    )

        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {str(e)}")
            continue

    # Filter tools based on allowed names
    filtered_tools = [
        tool
        for tool in tool_data
        if tool["Function Name"] in allowed_tool_names
    ]

    if not filtered_tools:
        print(
            f"Warning: No matching tools found. Available tools: {[t['Function Name'] for t in tool_data]}"
        )

    return {"name": agent_name}, filtered_tools


# Step 2: Extract tool input/output examples from snapshot
def extract_inputs_from_snapshot(snapshot_path: Path) -> dict:
    with snapshot_path.open("r", encoding="utf-8") as f:
        snapshot = json.load(f)
    return snapshot.get("input_output_examples", {})


# Step 3: Load a single example test case just for structure
def load_example(example_path: Path):
    with example_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Step 4: Prompt builder for N test cases from a given story
def build_prompt_for_story(
    agent,
    tools,
    tool_inputs,
    example_case: dict,
    story: str,
    num_variants: int = 2,
):
    renderer = BatchTestCaseGeneratorTemplateRenderer(
        BATCH_TEST_CASE_GENERATOR_PROMPT_PATH
    )

    tool_blocks = "\n".join(
        f"- Tool: {t['Function Name']}\n  Description: {t['Docstring']}\n  Args: {', '.join(t['Arguments']) or 'None'}"
        for t in tools
    )

    prompt = renderer.render(
        agent_name=agent["name"],
        tool_blocks=tool_blocks,
        tool_inputs_str=json.dumps(tool_inputs, indent=2),
        story=story,
        num_variants=num_variants,
        example_str=json.dumps(example_case, indent=2),
    )
    return prompt


# Step 5: Send prompt to LLM and save test cases
def generate_multiple_in_one(
    prompt,
    output_dir,
    starting_index,
    model_id="meta-llama/llama-3-405b-instruct",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = get_provider(
        model_id=model_id,
        params={
            "min_new_tokens": 50,
            "decoding_method": "greedy",
            "max_new_tokens": 3000,
        },
    )

    response = provider.query(prompt)

    try:
        raw_text = response
        json_start = raw_text.find("[")
        json_end = raw_text.rfind("]") + 1
        json_block = raw_text[json_start:json_end].strip()

        test_cases = json.loads(json_block)
        assert isinstance(test_cases, list), "Expected list of test cases"

        for i, case in enumerate(test_cases, start=starting_index):
            out_file = output_dir / f"synthetic_test_case_{i}.json"
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(case, f, indent=2)
            print(f"✅ Test case {i} written to {out_file}")

    except Exception as e:
        print("⚠️ Failed to parse or validate test case output.")
        print("Raw text:\n", raw_text)
        print("Error:", str(e))


def generate_test_cases_from_stories(
    agent_name: str,
    stories: list[str],
    tools_path: Path,
    snapshot_path: Path,
    output_dir: Path,
    allowed_tools: list[str],
    num_variants: int = 2,
):
    agent, tools = parse_tools_with_filter(
        agent_name, tools_path, allowed_tools
    )
    tool_inputs = extract_inputs_from_snapshot(snapshot_path)
    example_json = load_example(Path(EXAMPLE_PATH))

    test_case_counter = 1
    for idx, story in enumerate(stories, start=1):
        print(f"\n Generating test cases for story {idx}: {story}")

        prompt = build_prompt_for_story(
            agent,
            tools,
            tool_inputs,
            example_json,
            story,
            num_variants=num_variants,
        )

        generate_multiple_in_one(
            prompt=prompt,
            output_dir=output_dir,
            starting_index=test_case_counter,
        )

        test_case_counter += num_variants


def main(config: BatchAnnotateConfig):
    stories_path = Path(config.stories_path)

    stories = []
    agent_name = None
    with stories_path.open("r", encoding="utf-8", newline="") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            stories.append(row["story"])
            if agent_name is None:
                agent_name = row["agent"]

    tools_path = Path(config.tools_path)
    snapshot_path = stories_path.parent / f"{agent_name}_snapshot_llm.json"
    output_dir = Path(config.output_dir) / f"{agent_name}_test_cases"

    generate_test_cases_from_stories(
        agent_name,
        stories,
        tools_path,
        snapshot_path,
        output_dir,
        config.allowed_tools,
        num_variants=config.num_variants,
    )


if __name__ == "__main__":
    main(CLI(BatchAnnotateConfig, as_positional=False))

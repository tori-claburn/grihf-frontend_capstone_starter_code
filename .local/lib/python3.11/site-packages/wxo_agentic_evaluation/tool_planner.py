import ast
import csv
import importlib.util
import json
import os
import re
import sys
import textwrap
from dataclasses import asdict, is_dataclass
from pathlib import Path

from jsonargparse import CLI

from wxo_agentic_evaluation import __file__
from wxo_agentic_evaluation.arg_configs import BatchAnnotateConfig
from wxo_agentic_evaluation.prompt.template_render import (
    ArgsExtractorTemplateRenderer,
    ToolPlannerTemplateRenderer,
)
from wxo_agentic_evaluation.service_provider import get_provider

root_dir = os.path.dirname(__file__)
TOOL_PLANNER_PROMPT_PATH = os.path.join(
    root_dir, "prompt", "tool_planner.jinja2"
)
ARGS_EXTRACTOR_PROMPT_PATH = os.path.join(
    root_dir, "prompt", "args_extractor_prompt.jinja2"
)

MISSING_DOCSTRING_PROMPT = "No description available"


class UniversalEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def extract_first_json_list(raw: str) -> list:
    matches = re.findall(r"\[\s*{.*?}\s*]", raw, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list) and all(
                "tool_name" in step for step in parsed
            ):
                return parsed
        except Exception:
            continue
    print("⚠️ Could not parse tool call plan. Raw output:")
    print(raw)
    return []


def parse_json_string(input_string):
    json_char_count = 0
    json_objects = []
    current_json = ""
    brace_level = 0
    inside_json = False

    for i, char in enumerate(input_string):
        if char == "{":
            brace_level += 1
            inside_json = True
            json_char_count += 1
        if inside_json:
            current_json += char
            json_char_count += 1
        if char == "}":
            json_char_count += 1
            brace_level -= 1
            if brace_level == 0:
                inside_json = False
                try:
                    json_objects.append(json.loads(current_json))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                current_json = ""  # Reset current JSON string
    return json_objects


def load_tools_module(tools_path: Path) -> dict:
    tools_dict = {}
    files_to_parse = []

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
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(
                module_name, file_path
            )
            module = importlib.util.module_from_spec(spec)
            parent_dir = str(file_path.parent)
            sys_path_modified = False
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                sys_path_modified = True
            try:
                spec.loader.exec_module(module)
            finally:
                if sys_path_modified:
                    sys.path.pop()
            # Add all module's non-private functions to tools_dict
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and not attr_name.startswith("_"):
                    tools_dict[attr_name] = attr
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {str(e)}")

    return tools_dict


def extract_tool_signatures(tools_path: Path) -> list:
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
                code = f.read()
            parsed_code = ast.parse(code)

            for node in parsed_code.body:
                if isinstance(node, ast.FunctionDef):
                    name = node.name
                    args = [
                        arg.arg for arg in node.args.args if arg.arg != "self"
                    ]
                    docstring = ast.get_docstring(node)
                    tool_data.append(
                        {
                            "Function Name": name,
                            "Arguments": args,
                            "Docstring": docstring or MISSING_DOCSTRING_PROMPT,
                        }
                    )
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {str(e)}")
            continue

    return tool_data


def extract_tool_signatures_for_prompt(tools_path: Path) -> dict[str, str]:
    functions = {}
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
                code = f.read()
            parsed_code = ast.parse(code)

            for node in parsed_code.body:
                if isinstance(node, ast.FunctionDef):
                    name = node.name

                    # Get args and type annotations
                    args = []
                    for arg in node.args.args:
                        if arg.arg == "self":
                            continue
                        annotation = (
                            ast.unparse(arg.annotation)
                            if arg.annotation
                            else "Any"
                        )
                        args.append((arg.arg, annotation))

                    # Get return type
                    returns = (
                        ast.unparse(node.returns) if node.returns else "None"
                    )

                    # Get docstring
                    docstring = ast.get_docstring(node)
                    docstring = (
                        textwrap.dedent(docstring).strip() if docstring else ""
                    )

                    # Format parameter descriptions if available in docstring
                    doc_lines = docstring.splitlines()
                    doc_summary = doc_lines[0] if doc_lines else ""
                    param_descriptions = "\n".join(
                        [line for line in doc_lines[1:] if ":param" in line]
                    )

                    # Compose the final string
                    args_str = ", ".join(
                        f"{arg}: {type_}" for arg, type_ in args
                    )
                    function_str = f"""def {name}({args_str}) -> {returns}:
    {doc_summary}"""
                    if param_descriptions:
                        function_str += f"\n    {param_descriptions}"

                    functions[name] = function_str
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {str(e)}")
            continue

    return functions


def ensure_data_available(
    step: dict,
    inputs: dict,
    snapshot: dict,
    tools_module: dict,
    tool_signatures_for_prompt,
) -> dict:
    tool_name = step["tool_name"]
    cache = snapshot.setdefault("input_output_examples", {}).setdefault(
        tool_name, []
    )
    for entry in cache:
        if entry["inputs"] == inputs:
            return entry["output"]

    if tool_name not in tools_module:
        raise ValueError(f"Tool '{tool_name}' not found")

    try:
        output = tools_module[tool_name](**inputs)
    except:
        provider = get_provider(
            model_id="meta-llama/llama-3-405b-instruct",
            params={
                "min_new_tokens": 0,
                "decoding_method": "greedy",
                "max_new_tokens": 500,
            },
        )
        renderer = ArgsExtractorTemplateRenderer(ARGS_EXTRACTOR_PROMPT_PATH)

        prompt = renderer.render(
            tool_signature=tool_signatures_for_prompt[tool_name],
            step=step,
            inputs=inputs,
        )
        response = provider.query(prompt)
        json_obj = parse_json_string(response)[0]
        try:
            output = tools_module[json_obj["tool_name"]](**json_obj["inputs"])
        except:
            raise ValueError(
                f"Failed to execute tool '{tool_name}' with inputs {inputs}"
            )

    cache.append({"inputs": inputs, "output": output})
    if not isinstance(output, dict):
        print(f" Tool {tool_name} returned non-dict output: {output}")
    return output


def plan_tool_calls_with_llm(
    story: str, agent_name: str, tool_signatures_str: str, provider
) -> list:

    renderer = ToolPlannerTemplateRenderer(TOOL_PLANNER_PROMPT_PATH)

    prompt = renderer.render(
        user_story=story,
        agent_name=agent_name,
        available_tools=tool_signatures_str,
    )
    response = provider.query(prompt)
    parsed = extract_first_json_list(response)
    print("\n LLM Tool Plan:")
    print(json.dumps(parsed, indent=2))
    return parsed


# --- Tool Execution Logic ---
def run_tool_chain(
    tool_plan: list, snapshot: dict, tools_module, tool_signatures_for_prompt
) -> None:
    memory = {}

    for step in tool_plan:
        name = step["tool_name"]
        raw_inputs = step["inputs"]
        print(f"\n🔧 Tool: {name}")
        print(f" Raw inputs: {raw_inputs}")

        resolved_inputs = {}
        list_keys = []

        for k, v in raw_inputs.items():
            if isinstance(v, str) and v.startswith("$"):
                expr = v[1:]
                try:
                    resolved_value = eval(expr, {}, memory)
                    resolved_inputs[k] = resolved_value
                    if isinstance(resolved_value, list):
                        list_keys.append(k)
                except Exception as e:
                    print(f" ❌ Failed to resolve {v} in memory: {memory}")
                    raise ValueError(f"Failed to resolve placeholder {v}: {e}")
            else:
                resolved_inputs[k] = v

        print(f" Resolved inputs: {resolved_inputs}")

        if list_keys:
            if len(list_keys) > 1:
                raise ValueError(
                    f"Tool '{name}' received multiple list inputs. Only one supported for now."
                )
            list_key = list_keys[0]
            value_list = resolved_inputs[list_key]

            results = []
            for idx, val in enumerate(value_list):
                item_inputs = resolved_inputs.copy()
                item_inputs[list_key] = val
                print(f" ⚙️ Running {name} with {list_key} = {val}")
                output = ensure_data_available(
                    step,
                    item_inputs,
                    snapshot,
                    tools_module,
                    tool_signatures_for_prompt,
                )
                results.append(output)
                memory[f"{name}_{idx}"] = output

            memory[name] = results
            print(
                f"Stored {len(results)} outputs under '{name}' and indexed as '{name}_i'"
            )
        else:
            output = ensure_data_available(
                step,
                resolved_inputs,
                snapshot,
                tools_module,
                tool_signatures_for_prompt,
            )
            memory[name] = output
            print(f"Stored output under tool name: {name} = {output}")


# --- Main Snapshot Builder ---
def build_snapshot(
    agent_name: str, tools_path: Path, stories: list, output_path: Path
):
    agent = {"name": agent_name}
    tools_module = load_tools_module(tools_path)
    tool_signatures = extract_tool_signatures(tools_path)
    tool_signatures_for_prompt = extract_tool_signatures_for_prompt(tools_path)

    provider = get_provider(
        model_id="meta-llama/llama-3-405b-instruct",
        params={
            "min_new_tokens": 1,
            "decoding_method": "greedy",
            "max_new_tokens": 2048,
        },
    )

    snapshot = {
        "agent": agent,
        "tools": tool_signatures,
        "input_output_examples": {},
    }

    for story in stories:
        print(f"\n📘 Planning tool calls for story: {story}")
        tool_plan = plan_tool_calls_with_llm(
            story, agent["name"], tool_signatures, provider
        )
        try:
            run_tool_chain(
                tool_plan, snapshot, tools_module, tool_signatures_for_prompt
            )
        except ValueError as e:
            print(f"❌ Error running tool chain for story '{story}': {e}")
            continue

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, cls=UniversalEncoder)
    print(f"\n✅ Snapshot saved to {output_path}")


if __name__ == "__main__":
    config = CLI(BatchAnnotateConfig, as_positional=False)
    tools_path = Path(config.tools_path)
    stories_path = Path(config.stories_path)

    stories = []
    agent_name = None
    with stories_path.open("r", encoding="utf-8", newline="") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            stories.append(row["story"])
            if agent_name is None:
                agent_name = row["agent"]

    snapshot_path = stories_path.parent / f"{agent_name}_snapshot_llm.json"

    build_snapshot(agent_name, tools_path, stories, snapshot_path)

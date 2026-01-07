import ast
import re
from pathlib import Path
from typing import Any, List, Mapping, Union


class PythonTypeToJsonType:
    OPTIONAL_PARAM_EXTRACT = re.compile(r"[Oo]ptional\[(\w+)\]")

    @staticmethod
    def python_to_json_type(python_annotation: str):
        if not python_annotation:
            return "string"
        python_annotation = python_annotation.lower().strip()
        if "str" == python_annotation:
            return "string"
        if "int" == python_annotation:
            return "integer"
        if "float" == python_annotation:
            return "number"
        if "bool" == python_annotation:
            return "boolean"
        if python_annotation.startswith("list"):
            return "array"
        if python_annotation.startswith("dict"):
            return "object"
        if python_annotation.startswith("optional"):
            # extract the type within Optional[T]
            inner_type = PythonTypeToJsonType.OPTIONAL_PARAM_EXTRACT.search(
                python_annotation
            ).group(1)
            return PythonTypeToJsonType.python_to_json_type(inner_type)

        return "string"


class ToolExtractionOpenAIFormat:
    @staticmethod
    def get_default_arguments(node):
        """Returns the default arguments (if any)

        The default arguments are stored in args.default array.
        Since, in Python, the default arguments only come after positional arguments,
        we can index the argument array starting from the last `n` arguments, where n is
        the length of the default arguments.

        ex.
        def add(a, b=5):
           pass

        Then we have,
        args = [a, b]
        defaults = [Constant(value=5)]

        args[-len(defaults):] = [b]

        (
        "FunctionDef(
            name='add',
            args=arguments(
                posonlyargs=[],
                args=[
                    arg(arg='a'), "
                    "arg(arg='b')
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[Constant(value=5)]), "
            "body=[Return(value=BinOp(left=Name(id='a', ctx=Load()), op=Add(), "
            "right=Name(id='b', ctx=Load())))], decorator_list=[], type_params=[])")
        """
        default_arguments = set()
        num_defaults = len(node.args.defaults)
        if num_defaults > 0:
            for arg in node.args.args[-num_defaults:]:
                default_arguments.add(arg)

        return default_arguments

    @staticmethod
    def from_file(tools_path: Union[str, Path]) -> Mapping[str, Any]:
        """Uses `extract_tool_signatures` function, but converts the response
            to open-ai format

            ```
            function_spec = {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
        ```

        """
        tool_data = []
        tools_path = Path(tools_path)

        with tools_path.open("r", encoding="utf-8") as f:
            code = f.read()

        try:
            parsed_code = ast.parse(code)
            for node in parsed_code.body:
                if isinstance(node, ast.FunctionDef):
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }
                    function_name = node.name
                    for arg in node.args.args:
                        type_annotation = None
                        if arg.arg == "self":
                            continue
                        if arg.annotation:
                            type_annotation = ast.unparse(arg.annotation)

                        parameter_type = (
                            PythonTypeToJsonType.python_to_json_type(
                                type_annotation
                            )
                        )
                        parameters["properties"][arg.arg] = {
                            "type": parameter_type,
                            "description": "",  # todo
                        }

                        if (
                            type_annotation
                            and "Optional" not in type_annotation
                        ):
                            parameters["required"].append(arg.arg)

                    default_arguments = (
                        ToolExtractionOpenAIFormat.get_default_arguments(node)
                    )
                    for arg_name in parameters["required"]:
                        if arg_name in default_arguments:
                            parameters.remove(arg_name)

                    open_ai_format_fn = {
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "parameters": parameters,
                            "description": ast.get_docstring(
                                node
                            ),  # fix (does not do :params)
                        },
                    }
                    tool_data.append(open_ai_format_fn)

        except Exception as e:
            print(f"Warning: Failed to parse {tools_path}: {str(e)}")

        return tool_data

    @staticmethod
    def from_path(tools_path: Union[str, Path]) -> List[Mapping[str, Any]]:
        tools_path = Path(tools_path)
        files_to_parse = []
        all_tools = []

        if tools_path.is_file():
            files_to_parse.append(tools_path)
        elif tools_path.is_dir():
            files_to_parse.extend(tools_path.glob("**/*.py"))
        else:
            raise ValueError(
                f"Tools path {tools_path} is neither a file nor directory"
            )

        for file_path in files_to_parse:
            all_tools.extend(ToolExtractionOpenAIFormat.from_file(file_path))

        return all_tools

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from wxo_agentic_evaluation.referenceless_eval.function_calling.metrics.function_call.general import (
    GeneralMetricsPrompt,
)
from wxo_agentic_evaluation.referenceless_eval.function_calling.metrics.function_selection.function_selection import (
    FunctionSelectionPrompt,
)
from wxo_agentic_evaluation.referenceless_eval.function_calling.metrics.loader import (
    PromptKind,
    load_prompts_from_list,
)
from wxo_agentic_evaluation.referenceless_eval.function_calling.pipeline.adapters import (
    BaseAdapter,
    OpenAIAdapter,
)
from wxo_agentic_evaluation.referenceless_eval.function_calling.pipeline.transformation_prompts import (
    GENERATE_CODE_SCHEMA,
    GENERATE_CODE_SYSTEM,
    GENERATE_CODE_USER,
    MULTI_EXTRACT_UNITS_SYSTEM,
    MULTI_EXTRACT_UNITS_USER,
    build_multi_extract_units_schema,
)
from wxo_agentic_evaluation.referenceless_eval.function_calling.pipeline.types import (
    SemanticCategoryResult,
    SemanticResult,
    ToolCall,
    ToolSpec,
    TransformResult,
)
from wxo_agentic_evaluation.referenceless_eval.metrics.metrics_runner import (
    MetricRunner,
    MetricRunResult,
)
from wxo_agentic_evaluation.service_provider.watsonx_provider import (
    WatsonXProvider,
)


class SemanticChecker:
    """
    Orchestrates semantic metrics (and optional unit-transforms)
    for a single function call.

    Args:
        general_metrics: JSON-schema dicts for general metrics.
        function_metrics: JSON-schema dicts for function-selection metrics.
        parameter_metrics: JSON-schema dicts for parameter-level metrics.
        metrics_client: an WatsonXProvider instance for metric evaluation.
        codegen_client: an WatsonXProvider instance for transformation codegen.
        transform_enabled: whether to run unit-conversion checks.
    """

    def __init__(
        self,
        metrics_client: WatsonXProvider,
        *,
        general_metrics: Optional[List[Dict[str, Any]]] = None,
        function_metrics: Optional[List[Dict[str, Any]]] = None,
        parameter_metrics: Optional[List[Dict[str, Any]]] = None,
        codegen_client: Optional[WatsonXProvider] = None,
        transform_enabled: Optional[bool] = False,
    ) -> None:
        self.metrics_client = metrics_client

        self.transform_enabled = transform_enabled
        self.codegen_client = codegen_client

        self.general_prompts = []
        if general_metrics is not None:
            self.general_prompts = load_prompts_from_list(
                general_metrics, PromptKind.GENERAL
            )

        self.function_prompts = []
        if function_metrics is not None:
            self.function_prompts = load_prompts_from_list(
                function_metrics, PromptKind.FUNCTION_SELECTION
            )

        self.parameter_prompts = []
        if parameter_metrics is not None:
            self.parameter_prompts = load_prompts_from_list(
                parameter_metrics, PromptKind.PARAMETER
            )

    def _make_adapter(self, apis_specs, tool_call):
        first = apis_specs[0]
        if isinstance(first, ToolSpec):
            return OpenAIAdapter(apis_specs, tool_call)
        raise TypeError("Unsupported spec type")

    def _collect_params(self, adapter: BaseAdapter) -> Dict[str, Any]:
        """
        Return a mapping of every parameter name in the spec inventory
        to its value from the call (or defaulted if missing).
        """
        call_args = adapter.get_parameters()
        merged: Dict[str, Any] = {}
        # Find the function in the inventory
        function_parameters = (
            adapter.get_tool_spec(adapter.get_function_name())
            .get("parameters", {})
            .get("properties", {})
        )

        for pname, pschema in function_parameters.items():
            if pname in call_args:
                merged[pname] = call_args[pname]
            elif "default" in pschema:
                merged[pname] = pschema["default"]
            else:
                merged[pname] = (
                    f"Default value from parameter description (if defined): '{pschema.get('description', 'No description provided')}'"
                    f" Otherwise, by the default value of type: {pschema.get('type', 'object')}"
                )
        return merged

    def extract_all_units_sync(
        self,
        context: Union[str, List[Dict[str, str]]],
        adapter: BaseAdapter,
        params: List[str],
        retries: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Synchronously extract user_value/user_units_or_format/spec_units_or_format for every parameter in `params`
        by issuing a single LLM call.
        Returns a dict mapping each parameter name to its classification object.
        """
        # Build the combined JSON Schema requiring one object per parameter
        multi_schema = build_multi_extract_units_schema(params)
        schema_str = json.dumps(multi_schema, indent=2)

        # Build the "full_spec" JSON Schema snippet for all parameters
        full_spec_json = json.dumps(
            adapter.get_tool_spec(adapter.get_function_name()).model_dump(),
            indent=2,
        )

        # Format system and user prompts
        system_prompt = MULTI_EXTRACT_UNITS_SYSTEM.format(schema=schema_str)
        user_prompt = MULTI_EXTRACT_UNITS_USER.format(
            context=context,
            full_spec=full_spec_json,
            parameter_names=", ".join(params),
        )

        # Single synchronous LLM call
        try:
            response: Dict[str, Any] = self.metrics_client.generate(
                prompt=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                schema=multi_schema,
                retries=retries,
            )
        except Exception:
            response = {
                pname: {
                    "user_value": None,
                    "user_units_or_format": None,
                    "spec_units_or_format": None,
                }
                for pname in params
            }

        return response

    def run_sync(
        self,
        apis_specs: List[ToolSpec],
        tool_call: ToolCall,
        context: Union[str, List[Dict[str, str]]],
        retries: int = 1,
        transform_enabled: Optional[bool] = None,
    ) -> SemanticResult:
        """
        Synchronous semantic-only evaluation.

        Returns a SemanticResult:
          {
            "general": {metric_name: result, …}            or None
            "function_selection": {…}                       or None
            "parameter": {param_name: {metric_name: result}, …}   or None
            "transform": {param_name: TransformResult, …}   or None
          }
        """
        # 1) Normalize via adapter
        adapter = self._make_adapter(apis_specs, tool_call)
        tools_inventory_summary = adapter.get_tools_inventory_summary()
        call_dict = adapter.get_call_dict()
        fn_name = adapter.get_function_name()
        cur_tool_spec = adapter.get_tool_spec(fn_name)
        params = self._collect_params(adapter)

        if transform_enabled is not None:
            old_transform_enabled = self.transform_enabled
            self.transform_enabled = transform_enabled

        # 2) GENERAL METRICS
        general_results: Optional[SemanticCategoryResult]
        entries: List[Tuple[GeneralMetricsPrompt, Dict[str, Any]]] = []
        for prompt in self.general_prompts:
            entries.append(
                (
                    prompt,
                    {
                        "conversation_context": context,
                        "tool_inventory": cur_tool_spec,
                        "tool_call": call_dict,
                    },
                )
            )
        if entries:
            try:
                runner = MetricRunner(entries)
                sync_results = runner.run_all(
                    self.metrics_client.generate,
                    prompt_param_name="prompt",
                    schema_param_name="schema",
                    retries=retries,
                )
                general_results = SemanticCategoryResult.from_results(
                    sync_results
                )
            except Exception as e:
                general_results = {"error": str(e)}
        else:
            general_results = None

        # 3) FUNCTION-SELECTION METRICS
        function_results: Optional[SemanticCategoryResult]
        func_entries: List[Tuple[FunctionSelectionPrompt, Dict[str, Any]]] = []
        for prompt in self.function_prompts:
            func_entries.append(
                (
                    prompt,
                    {
                        "conversation_context": context,
                        "tools_inventory": tools_inventory_summary,
                        "proposed_tool_call": call_dict,
                        "selected_function": fn_name,
                    },
                )
            )
        if func_entries:
            try:
                runner = MetricRunner(func_entries)
                sync_results = runner.run_all(
                    self.metrics_client.generate,
                    prompt_param_name="prompt",
                    schema_param_name="schema",
                    retries=retries,
                )
                function_results = SemanticCategoryResult.from_results(
                    sync_results
                )
            except Exception as e:
                function_results = {"error": str(e)}
        else:
            function_results = None

        # 4) PARAMETER-LEVEL METRICS
        parameter_results: Optional[Dict[str, SemanticCategoryResult]] = {}
        for pname, pval in params.items():
            # Each parameter has its own prompts
            try:
                param_entries: List[
                    Tuple[ParameterMetricsPrompt, Dict[str, Any]]
                ] = []
                for prompt in self.parameter_prompts:
                    param_entries.append(
                        (
                            prompt,
                            {
                                "conversation_context": context,
                                "tool_inventory": cur_tool_spec,
                                "tool_call": call_dict,
                                "parameter_name": pname,
                                "parameter_value": pval,
                            },
                        )
                    )
                runner = MetricRunner(param_entries)
                sync_results = runner.run_all(
                    self.metrics_client.generate,
                    prompt_param_name="prompt",
                    schema_param_name="schema",
                    retries=retries,
                )
                parameter_results[pname] = SemanticCategoryResult.from_results(
                    sync_results
                )
            except Exception as e:
                parameter_results[pname] = {"error": str(e)}

        if not parameter_results:
            parameter_results = None

        # Base SemanticResult without transforms
        result = SemanticResult(
            general=general_results,
            function_selection=function_results,
            parameter=parameter_results,
        )

        # 5) OPTIONAL TRANSFORMS
        params = adapter.get_parameters()
        if self.transform_enabled and params:
            if transform_enabled is not None:
                self.transform_enabled = old_transform_enabled

            transform_out: Dict[str, TransformResult] = {}

            # 5a) Extract units for all parameters in one synchronous call
            units_map = self.extract_all_units_sync(
                context=context,
                adapter=adapter,
                params=list(params.keys()),
                retries=retries,
            )

            # 5b) Generate code & execute for each parameter needing conversion
            for pname, units in units_map.items():
                user_units = units.get("user_units_or_format") or ""
                spec_units = units.get("spec_units_or_format") or ""
                user_value = units.get("user_value")
                transformation_summary = units.get("transformation_summary", "")
                gen_code = ""

                # Only generate code if user_units differs from spec_units and user_value is present
                if (
                    user_units
                    and user_value is not None
                    and spec_units
                    and (user_units != spec_units)
                ):
                    try:
                        prompt = GENERATE_CODE_USER.format(
                            old_value=user_value,
                            old_units=user_units,
                            transformed_value=str(params[pname]),
                            transformed_units=spec_units,
                            transformed_type=type(params[pname]).__name__,
                            transformation_summary=transformation_summary,
                        )
                        gen_code = self.codegen_client.generate(
                            prompt=[
                                {
                                    "role": "system",
                                    "content": GENERATE_CODE_SYSTEM,
                                },
                                {"role": "user", "content": prompt},
                            ],
                            schema=GENERATE_CODE_SCHEMA,
                            retries=retries,
                        ).get("generated_code", "")
                    except Exception:
                        gen_code = ""

                # 5c) Execute & validate
                tr = self._execute_code_and_validate(
                    code=gen_code,
                    user_val=str(user_value or ""),
                    api_val=str(params[pname]),
                    units=units,
                )
                transform_out[pname] = tr

            if transform_out:
                result.transform = transform_out
            else:
                result.transform = None

        return result

    def _execute_code_and_validate(
        self,
        code: str,
        user_val: str,
        api_val: str,
        units: Dict[str, Any],
    ) -> TransformResult:
        """
        Strip code fences, install imports, exec code, compare, return TransformResult.
        """
        clean = re.sub(
            r"^```(?:python)?|```$", "", code, flags=re.MULTILINE
        ).strip()

        # install imports
        for mod in set(
            re.findall(
                r"^(?:import|from)\s+([A-Za-z0-9_]+)", clean, flags=re.MULTILINE
            )
        ):
            try:
                __import__(mod)
            except ImportError as e:
                return TransformResult(
                    units=units,
                    generated_code=clean,
                    execution_success=False,
                    correct=True,
                    execution_output=None,
                    correction=None,
                    error=f"Error: {e}. Could not import module '{mod}'. Please install the package and try again,"
                    " or run the generated code manually:\n"
                    f"transformation_code({user_val}) == convert_example_str_transformed_to_transformed_type({api_val})",
                )

        ns: Dict[str, Any] = {}
        try:
            exec(clean, ns)
            fn_t = ns.get("transformation_code")
            fn_c = ns.get("convert_example_str_transformed_to_transformed_type")
            if not callable(fn_t) or not callable(fn_c):
                raise ValueError("Generated code missing required functions")

            out_t = fn_t(user_val)
            out_c = fn_c(api_val)
            if isinstance(out_t, (int, float)) and isinstance(
                out_c, (int, float)
            ):
                success = math.isclose(out_t, out_c, abs_tol=1e-3)
            else:
                success = str(out_t) == str(out_c)

            correction = None
            if not success:
                correction = (
                    f"The transformation code validation found an issue with the units transformation "
                    f"of the parameter.\n"
                    f"The user request value is '{user_val}' with units '{units.get('user_units_or_format')}' and "
                    f"the API call value is '{api_val}' with units '{units.get('spec_units_or_format')}'.\n"
                    f"Expected transformation is '{out_t}' based on the code.\n"
                )

            correct = correction is None

            return TransformResult(
                units=units,
                generated_code=clean,
                execution_success=True,
                correct=correct,
                execution_output={"transformed": out_t, "converted": out_c},
                correction=correction,
                error=None,
            )
        except Exception as e:
            return TransformResult(
                units=units,
                generated_code=clean,
                execution_success=False,
                correct=True,
                execution_output=None,
                correction=None,
                error=str(e),
            )

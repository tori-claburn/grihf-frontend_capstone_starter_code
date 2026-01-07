from typing import Dict, List

from jsonschema import Draft7Validator

from wxo_agentic_evaluation.referenceless_eval.function_calling.pipeline.types import (
    StaticMetricResult,
    StaticResult,
    ToolCall,
    ToolSpec,
)

# ----------------------------------------
# Human-readable descriptions for checks
# ----------------------------------------
_STATIC_CHECKS: Dict[str, str] = {
    "non_existent_function": "Function name not found in the provided API specification.",
    "non_existent_parameter": "One or more parameters are not defined for the specified function.",
    "incorrect_parameter_type": "One or more parameters have values whose types don't match the expected types.",
    "missing_required_parameter": "One or more required parameters are missing from the call.",
    "allowed_values_violation": "One or more parameters have values outside the allowed enumeration.",
    "json_schema_validation": "The API call does not conform to the provided JSON Schema.",
    "empty_api_spec": "There are no API specifications provided or they are invalid.",
    "invalid_api_spec": "The API specifications provided are not valid Tool or ToolSpec instances.",
    "invalid_tool_call": "The provided ToolCall is not a valid instance of ToolCall.",
}


def evaluate_static(
    apis_specs: List[ToolSpec], api_call: ToolCall
) -> StaticResult:
    """
    Perform static validation on a single tool call.

    Args:
        apis_specs: Non-empty list of ToolSpec instances (OpenAI spec for ToolCall)
        api_call: Single call to validate: ToolCall instance (OpenAI tool call)

    Returns:
        StaticResult(metrics=..., final_decision=bool)
    """
    if not isinstance(apis_specs, list) or not apis_specs:
        return StaticResult(
            metrics={
                "empty_api_spec": StaticMetricResult(
                    description=_STATIC_CHECKS["empty_api_spec"],
                    valid=False,
                    explanation="No API specifications provided.",
                )
            },
            final_decision=False,
        )

    if not all(isinstance(spec, ToolSpec) for spec in apis_specs):
        return StaticResult(
            metrics={
                "invalid_api_spec": StaticMetricResult(
                    description=_STATIC_CHECKS["invalid_api_spec"],
                    valid=False,
                    explanation="Invalid API specifications provided; expected ToolSpec instances (List of ToolSpec).",
                )
            },
            final_decision=False,
        )

    if not isinstance(api_call, ToolCall):
        return StaticResult(
            metrics={
                "invalid_tool_call": StaticMetricResult(
                    description=_STATIC_CHECKS["invalid_tool_call"],
                    valid=False,
                    explanation="Invalid ToolCall provided; expected ToolCall instance.",
                )
            },
            final_decision=False,
        )

    errors = _check_tool_call(specs=apis_specs, call=api_call)

    # Build metrics results: missing key => valid
    metrics: Dict[str, StaticMetricResult] = {}
    for check_name, desc in _STATIC_CHECKS.items():
        valid = check_name not in errors
        metrics[check_name] = StaticMetricResult(
            description=desc,
            valid=valid,
            explanation=None if valid else errors.get(check_name),
        )
    final_decision = all(m.valid for m in metrics.values())
    return StaticResult(metrics=metrics, final_decision=final_decision)


def _check_tool_call(specs: List[ToolSpec], call: ToolCall) -> Dict[str, str]:
    """
    Static checks for OpenAI ToolCall + ToolSpec list.
    Returns mapping of failed check keys -> explanation.
    """
    errors: Dict[str, str] = {}

    # 1) Function existence
    spec = next(
        (s for s in specs if s.function.name == call.function.name), None
    )
    if not spec:
        errors["non_existent_function"] = (
            f"Function '{call.function.name}' does not exist in the provided API specifications:"
            f" {', '.join(s.function.name for s in specs)}."
        )
        return errors

    params_schema = spec.function.parameters
    properties = params_schema.get("properties", params_schema)
    parsed_arguments = call.function.parsed_arguments

    # 2) Parameter existence check
    if non_existent_params := set(parsed_arguments.keys()) - set(
        properties.keys()
    ):
        errors["non_existent_parameter"] = (
            f"Parameters not defined in function '{call.function.name}': "
            f"{', '.join(sorted(non_existent_params))}. "
            f"Possible parameters are: {', '.join(sorted(properties.keys()))}."
        )

    # 3) JSON Schema validation
    validator = Draft7Validator(params_schema)

    missing_required = []
    incorrect_types = []
    invalid_enum = []
    other_errors = []

    for error in validator.iter_errors(parsed_arguments):
        field = (
            ".".join(str(x) for x in error.path) if error.path else "unknown"
        )
        if error.validator == "required":
            missing_required.append(error.message)
        elif error.validator == "type":
            incorrect_types.append(f"{field}: {error.message}")
        elif error.validator == "enum":
            invalid_enum.append(f"{field}: {error.message}")
        else:
            other_errors.append(f"{field}: {error.message}")

    if missing_required:
        errors["missing_required_parameter"] = (
            "Missing required parameter(s): " + "; ".join(missing_required)
        )
    if incorrect_types:
        errors["incorrect_parameter_type"] = (
            "Incorrect parameter type(s): " + "; ".join(incorrect_types)
        )
    if invalid_enum:
        errors["allowed_values_violation"] = (
            "Invalid parameter value(s): " + "; ".join(invalid_enum)
        )
    if other_errors:
        errors["json_schema_validation"] = (
            "Other validation error(s): " + "; ".join(other_errors)
        )

    return errors

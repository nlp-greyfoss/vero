import inspect
from typing import Any, Dict, List, Union, get_origin, get_args


class Tool:
    """
    Represents a callable tool that can be used by an LLM-based agent system.

    This class supports:
    - Human-readable tool descriptions (for prompt-based agents)
    - OpenAI compatible function calling schema

    Attributes:
        name (str): The tool's name (derived from the wrapped function name).
        description (str): Human-readable explanation of what the tool does.
        func (callable): The underlying Python function.
        arguments (list): List of (param_name, param_type, default) tuples.
        outputs (str): The return type annotation as a string.
        signature (inspect.Signature): Cached function signature.
    """

    # Mapping from Python types to JSON Schema types
    PYTHON_TO_JSON = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }

    def __init__(
        self,
        name: str,
        description: str,
        func: callable,
        arguments: list,
        outputs: str,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

        # Cached inspect signature (used for schema generation)
        self.signature = inspect.signature(func)

    def __call__(self, *args, **kwargs):
        """
        Invokes the underlying wrapped function.
        """
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"<Tool {self.name}>"

    # ------------------------------------------------------------------
    # OpenAI / Qwen Function Calling Schema
    # ------------------------------------------------------------------
    def to_openai_schema(self) -> dict:
        """
        Convert this Tool into an OpenAI-compatible function calling schema.

        Returns:
            dict: Schema in the format expected by OpenAI / Qwen:
                {
                  "type": "function",
                  "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { ... }
                  }
                }
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for name, param in self.signature.parameters.items():
            if name == "self":
                continue

            annotation = param.annotation
            default = (
                None if param.default is inspect._empty else param.default
            )

            schema, is_required = self._annotation_to_schema(annotation, default)

            # Ensure every parameter has a description
            schema["description"] = name
            properties[name] = schema

            if is_required:
                required.append(name)

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
            # NOTE: can be enabled for stricter validation
            # "additionalProperties": False,
        }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": parameters,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _annotation_to_schema(self, annotation: Any, default: Any):
        """
        Convert a Python type annotation into a JSON Schema fragment.

        Returns:
            (schema_dict, is_required)
        """
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Optional[T] → not required
        if origin is Union and type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                sch, _ = self._annotation_to_schema(non_none[0], None)
                return {"anyOf": [sch, {"type": "null"}]}, False

        # List[T]
        if origin in (list, List):
            item = args[0] if args else Any
            item_schema, _ = self._annotation_to_schema(item, None)
            return {"type": "array", "items": item_schema}, default is None

        # Dict[str, T]
        if origin in (dict, Dict):
            value_type = args[1] if len(args) == 2 else Any
            value_schema, _ = self._annotation_to_schema(value_type, None)
            return {
                "type": "object",
                "additionalProperties": value_schema,
            }, default is None

        # Primitive types
        if annotation in self.PYTHON_TO_JSON:
            return {"type": self.PYTHON_TO_JSON[annotation]}, default is None

        # Fallback to string
        return {"type": "string"}, default is None


# ----------------------------------------------------------------------
# Decorator
# ----------------------------------------------------------------------
def tool(func):
    """
    A decorator that converts a regular Python function into a `Tool` object
    that can be used by LLM-based agent systems.

    The decorator inspects the function signature to extract:
      - Function name
      - Parameter names, types, and default values
      - Return type annotation
      - Function docstring

    The resulting Tool supports both:
      - Prompt-based tool descriptions
      - OpenAI-compatible function calling schema

    Returns:
        Tool: A fully constructed Tool instance.
    """
    signature = inspect.signature(func)
    arguments = []

    for param in signature.parameters.values():
        annotation = param.annotation
        type_str = (
            annotation.__name__
            if hasattr(annotation, "__name__")
            else str(annotation)
        )
        default = None if param.default is inspect._empty else param.default
        arguments.append((param.name, type_str, default))

    return_annotation = signature.return_annotation
    outputs = (
        "None"
        if return_annotation is inspect._empty
        else (
            return_annotation.__name__
            if hasattr(return_annotation, "__name__")
            else str(return_annotation)
        )
    )

    description = inspect.getdoc(func) or "No description provided."
    name = func.__name__

    return Tool(
        name=name,
        description=description,
        func=func,
        arguments=arguments,
        outputs=outputs,
    )

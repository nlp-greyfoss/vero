import inspect


class Tool:
    """
    Represents a callable tool that can be used by an LLM-based agent system.

    Attributes:
        name (str): The tool's name (derived from the wrapped function name).
        description (str): Human-readable explanation of what the tool does.
        func (callable): The underlying Python function.
        arguments (list): List of (param_name, param_type, default) tuples.
        outputs (str): The return type annotation as a string.
    """

    def __init__(self, name: str, description: str, func: callable,
                 arguments: list, outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments  # [(name, type_str, default_value or None), ...]
        self.outputs = outputs

    def to_string(self) -> str:
        """
        Returns a structured string describing the tool.
        """
        args_str = ", ".join(
            [
                f"{n}: {t}" + (f" = {d}" if d is not None else "")
                for (n, t, d) in self.arguments
            ]
        )

        return (
            f"Tool Name: {self.name}, "
            f"Description: {self.description}, "
            f"Arguments: {args_str}, "
            f"Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        """
        Invokes the underlying wrapped function.
        """
        return self.func(*args, **kwargs)


def tool(func):
    """
    A decorator that converts a regular Python function into a `Tool` object
    that can be used by LLM-based agent systems.

    The decorator inspects the function signature to automatically extract:
      - Function name
      - Parameter names, types, and default values
      - Return type annotation
      - Function docstring

    This ensures a consistent and explicit tool interface for the LLM.

    Returns:
        Tool: A fully constructed Tool instance.
    """

    signature = inspect.signature(func)
    arguments = []

    # Extract parameters
    for param in signature.parameters.values():
        # Parse annotation → readable name or str
        annotation = param.annotation
        type_str = annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)

        # Parse default value
        default = None if param.default is inspect._empty else param.default

        arguments.append((param.name, type_str, default))

    # Parse return type
    return_annotation = signature.return_annotation
    if return_annotation is inspect._empty:
        outputs = "None"
    else:
        outputs = (
            return_annotation.__name__
            if hasattr(return_annotation, "__name__")
            else str(return_annotation)
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

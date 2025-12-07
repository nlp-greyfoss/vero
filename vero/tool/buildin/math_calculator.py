import math
from vero.tool import tool

@tool
def math_evaluate(expr: str) -> str:
    """
    Safely evaluate a given mathematical expression and return the result.

    Args:
        expr (str): A mathematical expression as a string. 
            It may include numbers, arithmetic operators (+, -, *, /, **, %, parentheses),
            and math module functions/constants (e.g. sqrt, sin, cos, pi, etc.).

    Returns:
        str: The result of the evaluation, or an error message if evaluation fails or input is invalid.
    """
    # define allowed names: from math module
    allowed_names = {
        k: getattr(math, k) for k in dir(math) if not k.startswith("__")
    }

    try:
        # compile expression to code object
        code = compile(expr, "<string>", "eval")

        # inspect names used in expression; disallow names not in allowed_names
        for name in code.co_names:
            if name not in allowed_names:
                return f"Error: use of '{name}' is not allowed."

        # evaluate expression with restricted globals and allowed math names
        result = eval(code, {"__builtins__": {}}, allowed_names)

        return str(result)
    except Exception as e:
        # catch exceptions (syntax error, math domain error, etc.)
        return f"Evaluation error: {e}"

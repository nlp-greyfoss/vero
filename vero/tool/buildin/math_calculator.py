import math
from vero.tool import tool


@tool
def calculate_math_expression(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.

    This function evaluates mathematical expressions including basic arithmetic,
    parentheses, exponents, and common math functions from Python's math module.

    Args:
        expression (str): The mathematical expression to evaluate.
            - Supported operators: +, -, *, /, ** (power), % (modulo)
            - Supported functions: sqrt, sin, cos, tan, log, log10, exp, etc.
            - Supported constants: pi, e, tau, inf, nan
            - Parentheses: ( ) for grouping
            - Examples:
                * "2 + 3 * 4"
                * "sqrt(16) + sin(pi/2)"
                * "(2**3 + 5) / 2"
                * "log(100, 10)"

    Returns:
        str: The result of the calculation as a string, or an error message if the expression is invalid.

    Examples:
        >>> calculate_math_expression("2 + 3")
        '5'
        >>> calculate_math_expression("sqrt(25)")
        '5.0'
        >>> calculate_math_expression("pi * 2")
        '6.283185307179586'
    """
    # define allowed names: from math module
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}

    try:
        # compile expression to code object
        code = compile(expression, "<string>", "eval")

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

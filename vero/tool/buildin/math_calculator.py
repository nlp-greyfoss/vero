import math
import re
from vero.tool import tool

_UNIT_MULTIPLIERS = {
    "trillion": "1e12",
    "billion": "1e9",
    "million": "1e6",
    "thousand": "1e3",
}

_NATURAL_NUMBER_PATTERN = re.compile(
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*(trillion|billion|million|thousand)",
    re.IGNORECASE,
)


def _preprocess_expression(expression: str) -> str:
    """Convert natural-language amounts to Python numeric literals.

    Examples::

        "$3.624 trillion - $2.667 trillion" -> "(3.624 * 1e12) - (2.667 * 1e12)"
        "$500 billion + $200 million"       -> "(500 * 1e9) + (200 * 1e6)"
    """

    def _replace(m: re.Match) -> str:
        num = m.group(1).replace(",", "")
        unit = m.group(2).lower()
        return f"({num} * {_UNIT_MULTIPLIERS[unit]})"

    result = _NATURAL_NUMBER_PATTERN.sub(_replace, expression)
    # Strip any remaining bare '$' signs (e.g. "$3.624" without a unit word).
    result = result.replace("$", "")
    # Normalize whitespace-only or newline separators inside function calls to commas.
    # Handles cases like "max(85000\n1000000\n80000)" from multi-line llm_tool output.
    result = re.sub(r"(\d)\s*[\n\r]+\s*(\d)", r"\1, \2", result)
    return result


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
    # define allowed names: math module functions + safe builtins
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
    allowed_names.update({"max": max, "min": min, "abs": abs, "sum": sum, "round": round})

    try:
        expression = _preprocess_expression(expression)
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

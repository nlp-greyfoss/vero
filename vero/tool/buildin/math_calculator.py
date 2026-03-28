import math
import ast
import re
from vero.tool import tool


@tool
def calculate_math_expression(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.

    This function evaluates mathematical expressions including basic arithmetic,
    parentheses, exponents, and common math functions from Python's math module.
    It uses AST-based security checks to prevent code injection and unauthorized
    attribute access.

    Args:
        expression (str): The mathematical expression to evaluate.
            - Supported operators: +, -, *, /, ** (power), % (modulo)
            - Supported functions: sqrt, sin, cos, tan, log, log10, exp, abs,
              round, min, max, pow, and all functions from Python's math module
            - Supported constants: pi, e, tau, inf, nan
            - Parentheses: ( ) for grouping
            - Examples:
                * "2 + 3 * 4"
                * "sqrt(16) + sin(pi/2)"
                * "(2**3 + 5) / 2"
                * "log(100, 10)"
                * "abs(-5) + round(3.14)"

    Returns:
        str: The result of the calculation as a string. Integer results are
             formatted without decimal points. Returns an error message if the
             expression is invalid, contains disallowed operations, or if
             evaluation fails.

    Examples:
        >>> calculate_math_expression("2 + 3")
        '5'
        >>> calculate_math_expression("sqrt(25)")
        '5'
        >>> calculate_math_expression("pi * 2")
        '6.283185307179586'
        >>> calculate_math_expression("abs(-10) + round(3.7)")
        '14'
        >>> calculate_math_expression("2 ** 10")
        '1024'
    """
    def _normalize_expression(raw: str) -> str:
        """Normalize common noisy formats before AST parsing."""
        s = raw.strip()

        # Strip fenced code blocks if present.
        if s.startswith("```") and s.endswith("```"):
            lines = s.splitlines()
            if len(lines) >= 2:
                s = "\n".join(lines[1:-1]).strip()

        # Normalize common full-width punctuation seen in mixed-language outputs.
        full_width_map = str.maketrans(
            {
                "（": "(",
                "）": ")",
                "，": ",",
                "：": ":",
                "＋": "+",
                "－": "-",
                "＊": "*",
                "／": "/",
            }
        )
        s = s.translate(full_width_map)

        # Remove thousand separators like 84,400 -> 84400.
        s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

        return s.strip()

    def _fallback_numeric_eval(raw: str) -> str | None:
        """
        Lightweight fallback for noisy LLM outputs:
        supports max(...), min(...), sqrt(...) when numeric tokens can be extracted.
        """
        text = raw.strip()
        m = re.match(r"^\s*(max|min|sqrt)\s*\((.*)\)\s*$", text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None

        fn = m.group(1).lower()
        inner = m.group(2)
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", inner)
        if not nums:
            return None

        values = [float(n) for n in nums]
        if fn == "max":
            result = max(values)
        elif fn == "min":
            result = min(values)
        else:  # sqrt
            if len(values) != 1:
                return None
            if values[0] < 0:
                return "Value error: math domain error"
            result = math.sqrt(values[0])

        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
    
    safe_builtins = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'pow': pow,
    }
    allowed_names.update(safe_builtins)
    
    allowed_node_types = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Pow,
        ast.Mod,
        ast.UAdd,
        ast.USub,
    )

    normalized = _normalize_expression(expression)

    try:
        tree = ast.parse(normalized, mode="eval")

        for node in ast.walk(tree):
            if not isinstance(node, allowed_node_types):
                return f"Error: expression contains disallowed syntax '{type(node).__name__}'."
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in allowed_names:
                        return f"Error: function '{node.func.id}' is not allowed."
                elif isinstance(node.func, ast.Attribute):
                    return f"Error: attribute access is not allowed."
            elif isinstance(node, ast.Name):
                if node.id not in allowed_names:
                    return f"Error: use of '{node.id}' is not allowed."
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                return f"Error: import statements are not allowed."

        result = eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, allowed_names)

        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    except SyntaxError as e:
        fallback = _fallback_numeric_eval(normalized)
        if fallback is not None:
            return fallback
        return f"Syntax error: non-numeric or malformed expression. {e}"
    except ValueError as e:
        return f"Value error: {e}"
    except Exception as e:
        return f"Evaluation error: {e}"

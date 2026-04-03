import ast
import sys
import traceback
from io import StringIO
from typing import Dict, Optional

from pydantic import BaseModel, Field
from vero.tool import tool


class PythonREPL(BaseModel):
    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")

    def run(self, command: str) -> str:
        """Execute Python code and return printed output or traceback on failure.

        If the last statement is an expression (no explicit ``print()``), its
        repr is automatically printed — mirroring IPython / Jupyter behaviour.
        """
        old_stdout = sys.stdout
        captured_stdout = StringIO()
        sys.stdout = captured_stdout
        try:
            tree = ast.parse(command)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                # Execute all statements except the last one.
                if len(tree.body) > 1:
                    exec_node = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(exec_node, "<string>", "exec"), self.globals, self.locals)
                # Evaluate the last expression and print its repr if non-None.
                eval_node = ast.Expression(body=tree.body[-1].value)
                result = eval(compile(eval_node, "<string>", "eval"), self.globals, self.locals)
                if result is not None:
                    print(repr(result))
            else:
                exec(compile(tree, "<string>", "exec"), self.globals, self.locals)
            return captured_stdout.getvalue()
        except Exception:
            return traceback.format_exc()
        finally:
            sys.stdout = old_stdout


def _get_default_python_repl() -> PythonREPL:
    return PythonREPL(_globals=globals(), _locals=None)


@tool
def python_repl(input_text: str) -> str:
    """Execute Python code and return the captured output.

    If the last line is an expression, its value is returned automatically
    without needing an explicit ``print()`` call (IPython-style).

    Example input_text:
        x = 0
        for i in range(5):
            x += i
        x

    Args:
        input_text (str): Input must be valid Python code.

    Returns:
        str: Captured stdout output, or a traceback string on failure.
    """
    _python_repl = _get_default_python_repl()
    input_text = input_text.strip().replace("```python", "").replace("```", "").strip()
    # Strip surrounding quotes or backticks added by the LLM when wrapping code.
    if len(input_text) >= 2 and input_text[0] == input_text[-1] and input_text[0] in ('"', "'", "`"):
        input_text = input_text[1:-1]
    return _python_repl.run(input_text)

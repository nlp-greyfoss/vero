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
        """Execute Python code and return printed output or traceback on failure."""
        old_stdout = sys.stdout
        captured_stdout = StringIO()
        sys.stdout = captured_stdout
        try:
            exec(command, self.globals, self.locals)
            return captured_stdout.getvalue()
        except Exception:
            return traceback.format_exc()
        finally:
            sys.stdout = old_stdout


def _get_default_python_repl() -> PythonREPL:
    return PythonREPL(_globals=globals(), _locals=None)


@tool
def python_repl(input_text: str) -> str:
    """Execute Python code with ``exec()`` and return the captured output.

    Always use ``print()`` at the end of your code to expose results.

    Example input_text:
        x = 0
        for i in range(5):
            x += i
        print(x)

    Args:
        input_text (str): Input must be valid Python code.

    Returns:
        str: Captured stdout output, or a traceback string on failure.
    """
    _python_repl = _get_default_python_repl()
    input_text = input_text.strip().replace("```python", "").replace("```", "")
    return _python_repl.run(input_text)

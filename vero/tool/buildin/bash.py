import re
import subprocess
from pathlib import Path
from typing import List, Optional

from vero.tool import tool

# Maximum characters returned to the LLM to avoid context flooding
_MAX_OUTPUT_CHARS = 4000
# Hard ceiling on timeout regardless of what the caller requests
_MAX_TIMEOUT = 60
_DEFAULT_TIMEOUT = 15

# Sanitized environment: minimal PATH, no inherited secrets.
# This prevents the LLM from accidentally (or intentionally) exfiltrating
# environment variables such as API keys via curl/wget.
_SAFE_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "HOME": str(Path.home()),
    "LANG": "en_US.UTF-8",
    "TERM": "dumb",
}

# ---------------------------------------------------------------------------
# Blocklist — patterns matched against the raw command string.
# Defense-in-depth: not a sandbox. A truly isolated execution environment
# (Docker / seccomp) is the only complete solution.
# ---------------------------------------------------------------------------
_BLOCKED: List[re.Pattern] = [
    # ── Destructive file operations ──────────────────────────────────────
    re.compile(r"\brm\s+(-\S*r\S*|--recursive)", re.I),  # rm -r / rm -rf / rm --recursive
    re.compile(r"\brmdir\b", re.I),
    re.compile(r"\bshred\b", re.I),
    re.compile(r"\btruncate\b", re.I),
    # ── Privilege escalation ─────────────────────────────────────────────
    re.compile(r"\bsudo\b|\bdoas\b", re.I),
    re.compile(r"\bsu\s", re.I),
    # ── Disk / filesystem operations ────────────────────────────────────
    re.compile(r"\bdd\s", re.I),
    re.compile(r"\bmkfs\b|\bfdisk\b|\bparted\b|\bdiskutil\b", re.I),
    # ── System control ───────────────────────────────────────────────────
    re.compile(r"\b(shutdown|reboot|halt|poweroff|init)\b", re.I),
    # ── Process nuking ───────────────────────────────────────────────────
    re.compile(r"\bkillall\b|\bpkill\b", re.I),
    re.compile(r"\bkill\s+-9\b", re.I),
    # ── Fork bomb ────────────────────────────────────────────────────────
    re.compile(r":\s*\(\s*\)\s*\{"),
    # ── Download-and-execute (code injection via network) ────────────────
    re.compile(r"(curl|wget).+\|\s*(ba)?sh", re.I),
    re.compile(r"(curl|wget).+\|\s*python\d*", re.I),
    # ── Writing to protected system paths ───────────────────────────────
    re.compile(r">\s*/(etc|usr|bin|sbin|boot|sys|proc)/", re.I),
    # ── Persistent backdoors ─────────────────────────────────────────────
    re.compile(r"\bcrontab\b", re.I),
    re.compile(r"\.(bashrc|profile|bash_profile|zshrc)\b", re.I),
    # ── Network listeners (reverse shell) ────────────────────────────────
    re.compile(r"\bnc\b.+(-l|--listen)", re.I),
    re.compile(r"\bnetcat\b.+(-l|--listen)", re.I),
]


def _blocked_by(command: str) -> Optional[str]:
    """Return the first matching blocked pattern, or None if the command is safe."""
    for pattern in _BLOCKED:
        if pattern.search(command):
            return pattern.pattern
    return None


@tool
def bash(command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Execute a bash command and return its stdout/stderr.

    Supports pipes, redirects, and standard bash syntax (curl, grep, awk, jq …).
    Dangerous operations are blocked: rm -rf, sudo, disk writes, fork bombs,
    download-and-execute, network listeners, and writes to system paths.
    Output is capped at 4000 characters.

    Args:
        command: A bash command string.
        timeout: Execution time limit in seconds (default 15, max 60).

    Returns:
        Combined stdout + stderr, or a [BLOCKED] / [TIMEOUT] / [ERROR] message.
    """
    timeout = min(max(1, timeout), _MAX_TIMEOUT)

    blocked = _blocked_by(command)
    if blocked is not None:
        return f"[BLOCKED] Command refused — matched dangerous pattern: `{blocked}`"

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
            env=_SAFE_ENV,
        )
        output = (proc.stdout + proc.stderr).strip()
        if not output:
            return f"(no output, exit code {proc.returncode})"
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + f"\n… [truncated — {len(output)} chars total]"
        return output
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Command exceeded {timeout}s limit."
    except Exception as e:
        return f"[ERROR] {e}"

"""Small, fail-closed SandboxFusion HTTP client for Python execution."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import httpx

__all__ = ["ExecutionResult", "run_code"]

logger = logging.getLogger(__name__)

ExecutionStatus = Literal[
    "ok",
    "compile_error",
    "runtime_error",
    "timeout",
    "memory_error",
    "sandbox_error",
]

_DEFAULT_MAX_CONCURRENT = 64
_DEFAULT_RUN_TIMEOUT = 10
_DEFAULT_COMPILE_TIMEOUT = 10
_DEFAULT_MEMORY_LIMIT_MB = 1024
_HTTP_TIMEOUT_BUFFER_SECONDS = 10
_MAX_ATTEMPTS = 3
_INITIAL_RETRY_DELAY_SECONDS = 1


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw_value, default)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, raw_value, default)
        return default
    return value


# One semaphore is created per imported module/process and shared by every
# run_code() invocation in that process.  It is deliberately not recreated per
# request.
_REQUEST_SEMAPHORE = threading.BoundedSemaphore(
    _positive_int_from_env("CODE_SANDBOX_MAX_CONCURRENT", _DEFAULT_MAX_CONCURRENT)
)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    success: bool
    status: ExecutionStatus
    stdout: str
    stderr: str
    return_code: int | None


def _sandbox_error(stderr: str = "") -> ExecutionResult:
    return ExecutionResult(
        success=False,
        status="sandbox_error",
        stdout="",
        stderr=stderr,
        return_code=None,
    )


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _as_return_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _looks_like_memory_error(*values: Any) -> bool:
    text = " ".join(_as_text(value).lower() for value in values)
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if "memorylimit" in compact or "outofmemory" in compact:
        return True
    return re.search(r"\b(?:oom|mle)\b", text) is not None


def _convert_response(payload: Mapping[str, Any]) -> ExecutionResult:
    """Convert a SandboxFusion response into the stable local result schema."""
    api_status = _as_text(payload.get("status"))
    compile_result = _as_mapping(payload.get("compile_result"))
    run_result = _as_mapping(payload.get("run_result"))

    compile_status = _as_text(compile_result.get("status")) if compile_result else ""
    compile_stderr = _as_text(compile_result.get("stderr")) if compile_result else ""
    run_status = _as_text(run_result.get("status")) if run_result else ""
    stdout = _as_text(run_result.get("stdout")) if run_result else ""
    run_stderr = _as_text(run_result.get("stderr")) if run_result else ""
    stderr = run_stderr or compile_stderr
    run_return_code = _as_return_code(run_result.get("return_code")) if run_result else None
    compile_return_code = _as_return_code(compile_result.get("return_code")) if compile_result else None

    api_status_lower = api_status.lower()
    compile_status_lower = compile_status.lower()
    run_status_lower = run_status.lower()

    if api_status_lower == "success":
        if run_result and run_status_lower == "finished" and run_return_code in (None, 0):
            return ExecutionResult(True, "ok", stdout, stderr, run_return_code)
        if _looks_like_memory_error(run_status, run_stderr):
            return ExecutionResult(False, "memory_error", stdout, stderr, run_return_code)
        if run_status_lower == "timelimitexceeded":
            return ExecutionResult(False, "timeout", stdout, stderr, run_return_code)
        return ExecutionResult(False, "runtime_error", stdout, stderr, run_return_code)

    if api_status_lower == "failed":
        if compile_result:
            if _looks_like_memory_error(compile_status, compile_stderr):
                return ExecutionResult(False, "memory_error", stdout, stderr, compile_return_code)
            if compile_status_lower == "timelimitexceeded":
                return ExecutionResult(False, "timeout", stdout, stderr, compile_return_code)
            if compile_status_lower == "error" or (
                compile_status_lower == "finished" and compile_return_code not in (None, 0)
            ):
                return ExecutionResult(False, "compile_error", stdout, stderr, compile_return_code)

        if run_result:
            if _looks_like_memory_error(run_status, run_stderr):
                return ExecutionResult(False, "memory_error", stdout, stderr, run_return_code)
            if run_status_lower == "timelimitexceeded":
                return ExecutionResult(False, "timeout", stdout, stderr, run_return_code)
            if run_status_lower == "error" or (
                run_status_lower == "finished" and run_return_code not in (None, 0)
            ):
                return ExecutionResult(False, "runtime_error", stdout, stderr, run_return_code)

    return _sandbox_error(stderr or f"Unexpected SandboxFusion response status: {api_status!r}")


def _post(url: str, payload: dict[str, Any], request_timeout: float) -> httpx.Response:
    with _REQUEST_SEMAPHORE:
        return httpx.post(url, json=payload, timeout=request_timeout)


def run_code(code: str, stdin: str) -> ExecutionResult:
    """Run Python code through SandboxFusion without leaking errors to training."""
    url = os.environ.get("SANDBOX_FUSION_URL", "").strip()
    if not url:
        return _sandbox_error("SANDBOX_FUSION_URL is not configured")

    compile_timeout = _positive_int_from_env("CODE_COMPILE_TIMEOUT", _DEFAULT_COMPILE_TIMEOUT)
    run_timeout = _positive_int_from_env("CODE_RUN_TIMEOUT", _DEFAULT_RUN_TIMEOUT)
    memory_limit_mb = _positive_int_from_env("CODE_MEMORY_LIMIT_MB", _DEFAULT_MEMORY_LIMIT_MB)
    request_timeout = compile_timeout + run_timeout + _HTTP_TIMEOUT_BUFFER_SECONDS
    request_payload = {
        "code": code,
        "stdin": stdin,
        "language": "python",
        "compile_timeout": compile_timeout,
        "run_timeout": run_timeout,
        "memory_limit_MB": memory_limit_mb,
        "files": {},
        "fetch_files": [],
    }

    last_error = "SandboxFusion request failed"
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = _post(url, request_payload, request_timeout)
        except httpx.HTTPError as exc:
            return _sandbox_error(f"SandboxFusion request failed: {exc}")
        except Exception as exc:  # Never let client/configuration errors abort training.
            return _sandbox_error(f"Unexpected SandboxFusion client error: {exc}")

        if response.status_code == 504:
            last_error = f"SandboxFusion gateway timeout (504), attempt {attempt + 1}/{_MAX_ATTEMPTS}"
            if attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(_INITIAL_RETRY_DELAY_SECONDS * (attempt + 1))
                continue
            return _sandbox_error(last_error)

        if not 200 <= response.status_code < 300:
            return _sandbox_error(f"SandboxFusion returned HTTP {response.status_code}")

        try:
            response_payload = response.json()
        except (ValueError, TypeError) as exc:
            return _sandbox_error(f"SandboxFusion returned invalid JSON: {exc}")
        if not isinstance(response_payload, Mapping):
            return _sandbox_error("SandboxFusion returned a non-object JSON response")

        try:
            return _convert_response(response_payload)
        except Exception as exc:  # Malformed service responses also fail closed.
            return _sandbox_error(f"Could not parse SandboxFusion response: {exc}")

    return _sandbox_error(last_error)

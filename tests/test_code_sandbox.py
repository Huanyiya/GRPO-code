import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from slime_plugins.rewards import code_sandbox


class _SandboxFusionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/run_code":
            self.send_error(404)
            return

        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.server.last_payload = payload

        try:
            completed = subprocess.run(
                [sys.executable, "-c", payload["code"]],
                input=payload["stdin"],
                text=True,
                capture_output=True,
                timeout=payload["run_timeout"],
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            response = {
                "status": "Failed",
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "return_code": None,
                },
            }
        else:
            if completed.returncode == 0:
                response = {
                    "status": "Success",
                    "run_result": {
                        "status": "Finished",
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        "return_code": completed.returncode,
                    },
                }
            else:
                is_syntax_error = "SyntaxError:" in completed.stderr
                response = {
                    "status": "Failed",
                    "compile_result": (
                        {
                            "status": "Error",
                            "stderr": completed.stderr,
                            "return_code": completed.returncode,
                        }
                        if is_syntax_error
                        else None
                    ),
                    "run_result": (
                        None
                        if is_syntax_error
                        else {
                            "status": "Error",
                            "stdout": completed.stdout,
                            "stderr": completed.stderr,
                            "return_code": completed.returncode,
                        }
                    ),
                }

        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        del format, args


@contextmanager
def _sandbox_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SandboxFusionHandler)
    server.last_payload = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}/run_code"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def sandbox_url(monkeypatch):
    with _sandbox_server() as (server, url):
        monkeypatch.setenv("SANDBOX_FUSION_URL", url)
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
        monkeypatch.setenv("CODE_RUN_TIMEOUT", "1")
        monkeypatch.setenv("CODE_COMPILE_TIMEOUT", "2")
        monkeypatch.setenv("CODE_MEMORY_LIMIT_MB", "256")
        yield server


def test_simple_python_success_and_request_shape(sandbox_url):
    result = code_sandbox.run_code("print(1 + 2)", "")

    assert result.success is True
    assert result.status == "ok"
    assert result.stdout.strip() == "3"
    assert result.stderr == ""
    assert result.return_code == 0
    assert sandbox_url.last_payload == {
        "code": "print(1 + 2)",
        "stdin": "",
        "language": "python",
        "compile_timeout": 2,
        "run_timeout": 1,
        "memory_limit_MB": 256,
        "files": {},
        "fetch_files": [],
    }


def test_stdin_program(sandbox_url):
    result = code_sandbox.run_code("a, b = map(int, input().split())\nprint(a + b)", "2 3\n")
    assert result.success is True
    assert result.status == "ok"
    assert result.stdout.strip() == "5"


def test_syntax_error(sandbox_url):
    result = code_sandbox.run_code("def broken(:\n    pass", "")
    assert result.success is False
    assert result.status == "compile_error"
    assert "SyntaxError" in result.stderr


def test_runtime_error(sandbox_url):
    result = code_sandbox.run_code("raise RuntimeError('boom')", "")
    assert result.success is False
    assert result.status == "runtime_error"
    assert "RuntimeError" in result.stderr


def test_timeout(sandbox_url):
    result = code_sandbox.run_code("while True:\n    pass", "")
    assert result.success is False
    assert result.status == "timeout"


def test_unreachable_url_returns_sandbox_error(monkeypatch):
    monkeypatch.setenv("SANDBOX_FUSION_URL", "http://127.0.0.1:1/run_code")
    result = code_sandbox.run_code("print(1)", "")
    assert result.success is False
    assert result.status == "sandbox_error"


def test_gateway_timeout_is_retried_then_fails_closed(monkeypatch):
    attempts = 0

    def fake_post(url, payload, request_timeout):
        nonlocal attempts
        del url, payload, request_timeout
        attempts += 1
        return code_sandbox.httpx.Response(504)

    monkeypatch.setenv("SANDBOX_FUSION_URL", "http://sandbox.invalid/run_code")
    monkeypatch.setattr(code_sandbox, "_post", fake_post)
    monkeypatch.setattr(code_sandbox.time, "sleep", lambda _: None)

    result = code_sandbox.run_code("print(1)", "")
    assert attempts == 3
    assert result.success is False
    assert result.status == "sandbox_error"


def test_memory_error_response_is_normalized():
    result = code_sandbox._convert_response(
        {
            "status": "Failed",
            "run_result": {
                "status": "MemoryLimitExceeded",
                "stdout": "",
                "stderr": "memory limit exceeded",
                "return_code": 137,
            },
        }
    )
    assert result.success is False
    assert result.status == "memory_error"
    assert result.return_code == 137


def test_invalid_json_response_fails_closed(monkeypatch):
    monkeypatch.setenv("SANDBOX_FUSION_URL", "http://sandbox.invalid/run_code")
    monkeypatch.setattr(
        code_sandbox,
        "_post",
        lambda url, payload, request_timeout: code_sandbox.httpx.Response(200, text="not-json"),
    )

    result = code_sandbox.run_code("print(1)", "")
    assert result.success is False
    assert result.status == "sandbox_error"

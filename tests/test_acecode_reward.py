import asyncio
from types import SimpleNamespace

import pytest

from slime_plugins.rewards import acecode
from slime_plugins.rewards.code_sandbox import ExecutionResult


def _reward(sample):
    return asyncio.run(acecode.reward_func(None, sample))


@pytest.fixture(autouse=True)
def _run_sandbox_sync_in_unit_tests(monkeypatch):
    """Avoid starting a thread solely to execute an in-process test double."""

    async def direct_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(acecode.asyncio, "to_thread", direct_to_thread)


def _local_executor(executable, stdin):
    del stdin
    try:
        exec(executable, {})
    except Exception as exc:
        return ExecutionResult(False, "runtime_error", "", str(exc), 1)
    return ExecutionResult(True, "ok", "", "", 0)


def test_acecode_reward_runs_each_test_in_a_fresh_namespace(monkeypatch):
    monkeypatch.setattr(acecode.code_sandbox, "run_code", _local_executor)
    monkeypatch.setenv("SAVE_OUTPUTS", "true")
    sample = SimpleNamespace(
        response="class Counter:\n    value = 0\n    def next(self):\n        Counter.value += 1\n        return Counter.value",
        label={"test_cases": ["assert Counter().next() == 1", "assert Counter().next() == 1"]},
    )

    assert _reward(sample) == 1.0
    assert sample.metadata["code_reward_trace"]["final_reward"] == 1.0
    assert len(sample.metadata["code_reward_trace"]["test_cases"]) == 2


def test_acecode_reward_returns_zero_for_a_failed_assertion(monkeypatch):
    monkeypatch.setattr(acecode.code_sandbox, "run_code", _local_executor)
    sample = SimpleNamespace(response="def add(a, b):\n    return a + b", label={"test_cases": ["assert add(1, 2) == 4"]})

    assert _reward(sample) == 0.0


def test_acecode_reward_runs_all_test_cases(monkeypatch):
    calls = []

    def executor(executable, stdin):
        del stdin
        calls.append(executable)
        return ExecutionResult(True, "ok", "", "", 0)

    monkeypatch.setattr(acecode.code_sandbox, "run_code", executor)
    sample = SimpleNamespace(
        response="def answer():\n    return 42",
        label={"test_cases": [f"assert answer() == 42  # {index}" for index in range(16)]},
    )

    assert _reward(sample) == 1.0
    assert len(calls) == 16


def test_acecode_reward_delegates_eurus_labels(monkeypatch):
    async def fake_eurus_reward(args, sample, **kwargs):
        assert args == "args"
        assert sample.label == {"ground_truth": "{}"}
        assert kwargs == {"flag": True}
        return 0.75

    monkeypatch.setattr(acecode.eurus_code, "reward_func", fake_eurus_reward)
    sample = SimpleNamespace(label={"ground_truth": "{}"})

    assert asyncio.run(acecode.reward_func("args", sample, flag=True)) == 0.75

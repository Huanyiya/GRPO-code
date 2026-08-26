import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from slime_plugins.rewards import code
from slime_plugins.rewards.code_sandbox import ExecutionResult


def _sample(response, inputs, outputs, *, prompt=None, **ground_truth_extra):
    ground_truth = {"inputs": inputs, "outputs": outputs, **ground_truth_extra}
    sample = SimpleNamespace(
        response=response,
        label={"ground_truth": json.dumps(ground_truth)},
    )
    if prompt is not None:
        sample.prompt = prompt
    return sample


def _ok(stdout):
    return ExecutionResult(True, "ok", stdout, "", 0)


def _failed(status):
    return ExecutionResult(False, status, "", status, 1)


def _reward(sample):
    return asyncio.run(code.reward_func(None, sample))


def test_extracts_last_python_fence_then_generic_fence_then_full_response():
    response = """
```python
print('old')
```
```text
not python
```
```PyThOn
print('new')
```
"""
    assert code._extract_python_code(response) == "print('new')"
    assert code._extract_python_code("```code\nprint(2)\n```") == "print(2)"
    assert code._extract_python_code("  print(3)  ") == "print(3)"


def test_correct_standard_io_program_returns_one(monkeypatch):
    outputs = {"2 3\n": "5  \r\n", "4\n5": "9\n\n"}
    monkeypatch.setattr(code.code_sandbox, "run_code", lambda generated, stdin: _ok(outputs[stdin]))
    sample = _sample(
        "```python\na, b = map(int, input().split())\nprint(a + b)\n```",
        ["2 3\n", ["4", "5"]],
        ["5\n", ["9"]],
    )
    assert _reward(sample) == 1.0


def test_wrong_answer_returns_zero(monkeypatch):
    monkeypatch.setattr(code.code_sandbox, "run_code", lambda generated, stdin: _ok("6\n"))
    assert _reward(_sample("print(6)", ["2 3\n"], ["5\n"])) == 0.0


@pytest.mark.parametrize("status", ["runtime_error", "timeout"])
def test_execution_failure_returns_zero(monkeypatch, status):
    monkeypatch.setattr(code.code_sandbox, "run_code", lambda generated, stdin: _failed(status))
    assert _reward(_sample("print(1)", [""], ["1\n"])) == 0.0


def test_malformed_ground_truth_returns_zero_without_sandbox_call(monkeypatch):
    calls = []
    monkeypatch.setattr(code.code_sandbox, "run_code", lambda generated, stdin: calls.append(stdin))
    sample = SimpleNamespace(response="print(1)", label={"ground_truth": "not-json"})
    assert _reward(sample) == 0.0
    assert calls == []


def test_global_function_call_returns_one(monkeypatch):
    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)
    sample = _sample("def add(a, b):\n    return a + b", [[2, 3]], [5], fn_name="add")
    assert _reward(sample) == 1.0


def test_solution_method_call_returns_one(monkeypatch):
    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)
    generated = "class Solution:\n    def add(self, a, b):\n        return a + b"
    assert _reward(_sample(generated, [[2, 3]], [5], fn_name="add")) == 1.0


def test_eurus_call_based_singleton_wrapped_ground_truth(monkeypatch):
    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)

    string_sample = _sample(
        """
def reverse_by_center(s):
    n = len(s)
    m = n // 2
    if n % 2 == 0:
        return s[m:] + s[:m]
    return s[m + 1:] + s[m] + s[:m]
""",
        [["secret"]],
        [["retsec"]],
        fn_name="reverse_by_center",
    )
    integer_sample = _sample(
        "def answer(): return 4321",
        [[]],
        [[4321]],
        fn_name="answer",
    )
    list_sample = _sample(
        "def values(): return [1, 2, 3]",
        [[]],
        [[[1, 2, 3]]],
        fn_name="values",
    )

    assert _reward(string_sample) == 1.0
    assert _reward(integer_sample) == 1.0
    assert _reward(list_sample) == 1.0


def test_missing_fn_name_returns_zero(monkeypatch):
    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)
    assert _reward(_sample("def other(): return 1", [[]], [1], fn_name="missing")) == 0.0


@pytest.mark.parametrize(
    ("return_expression", "expected"),
    [
        ("[1, 2, 3]", [1, 2, 3]),
        ("{'a': 1, 'b': 2}", {"a": 1, "b": 2}),
        ("[{'a': [1, 2.0]}, {'ok': True}]", [{"a": [1, 2.0]}, {"ok": True}]),
        ("(1, 2, 3)", [1, 2, 3]),
        ("0.3", 0.3),
    ],
)
def test_call_based_structured_outputs(monkeypatch, return_expression, expected):
    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)
    generated = f"def solve():\n    return {return_expression}"
    assert _reward(_sample(generated, [[]], [expected], fn_name="solve")) == 1.0


def test_call_based_float_clear_error_fails(monkeypatch):
    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)
    assert _reward(_sample("def solve(): return 0.31", [[]], [0.3], fn_name="solve")) == 0.0


def test_float_tolerance_requires_an_explicit_problem_statement(monkeypatch):
    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)
    generated = "def solve(): return 0.3000000001"

    strict_sample = _sample(generated, [[]], [0.3], fn_name="solve")
    assert _reward(strict_sample) == 0.0

    float_problem_prompt = [
        {
            "role": "user",
            "content": "Print a floating point answer. Absolute or relative error at most 10^{-6} is accepted.",
        }
    ]
    tolerant_sample = _sample(generated, [[]], [0.3], fn_name="solve", prompt=float_problem_prompt)
    assert _reward(tolerant_sample) == 1.0


def test_problem_statement_tolerance_parser_keeps_absolute_and_relative_limits():
    tolerance = code._extract_float_tolerance(
        "The answer is correct if absolute error is less than 10^{-6}, or relative error is at most 10^{-9}."
    )
    assert tolerance.abs_tol == code.Decimal("1e-6")
    assert tolerance.rel_tol == code.Decimal("1e-9")

    shared_tolerance = code._extract_float_tolerance(
        "The relative or absolute error should not be higher than $10^{-7}$."
    )
    assert shared_tolerance.abs_tol == code.Decimal("1e-7")
    assert shared_tolerance.rel_tol == code.Decimal("1e-7")


def test_call_based_malformed_json_stdout_fails(monkeypatch):
    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)
    generated = "def solve():\n    print('debug')\n    return [1, 2]"
    assert _reward(_sample(generated, [[]], [[1, 2]], fn_name="solve")) == 0.0


def test_verl_json_lines_call_input(monkeypatch):
    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)
    sample = _sample(
        "def add_lengths(items, suffix): return len(items) + len(suffix)",
        ['["a", "b"]\n"xyz"'],
        [5],
        fn_name="add_lengths",
    )
    assert _reward(sample) == 1.0


def test_call_based_parse_error_does_not_call_sandbox(monkeypatch):
    calls = []
    monkeypatch.setattr(code.code_sandbox, "run_code", lambda generated, stdin: calls.append(stdin))
    sample = _sample("def solve(x): return x", ["not-json"], [1], fn_name="solve")
    assert _reward(sample) == 0.0
    assert calls == []


def test_twenty_cases_select_only_longest_fifteen(monkeypatch):
    calls = []

    def fake_run(generated, stdin):
        calls.append(stdin)
        return _ok(stdin)

    monkeypatch.setattr(code.code_sandbox, "run_code", fake_run)
    inputs = ["x" * length for length in range(1, 21)]
    assert _reward(_sample("print(input())", inputs, inputs)) == 1.0
    assert calls == ["x" * length for length in range(20, 5, -1)]


def test_fail_fast_stops_after_third_selected_case(monkeypatch):
    calls = []

    def fake_run(generated, stdin):
        calls.append(stdin)
        return _ok("wrong") if len(calls) == 3 else _ok(stdin)

    monkeypatch.setattr(code.code_sandbox, "run_code", fake_run)
    inputs = ["x" * length for length in range(1, 21)]
    assert _reward(_sample("print(input())", inputs, inputs)) == 0.0
    assert calls == ["x" * 20, "x" * 19, "x" * 18]


def test_save_outputs_records_only_executed_testcases(monkeypatch):
    monkeypatch.setenv("SAVE_OUTPUTS", "true")
    calls = []

    def fake_run(generated, stdin):
        calls.append(stdin)
        return _ok("wrong") if len(calls) == 2 else _ok(stdin)

    monkeypatch.setattr(code.code_sandbox, "run_code", fake_run)
    sample = _sample("print(input())", ["first", "second", "third"], ["first", "second", "third"])

    assert _reward(sample) == 0.0
    trace = sample.metadata["code_reward_trace"]
    assert trace["generated_code"] == "print(input())"
    assert trace["final_reward"] == 0.0
    assert trace["failure_reason"] == "output_mismatch"
    assert [case["input"] for case in trace["test_cases"]] == ["first", "second"]
    assert [case["output_matches"] for case in trace["test_cases"]] == [True, False]


def test_rollout_outputs_are_human_readable_jsonl(tmp_path):
    from slime.ray.rollout import RolloutManager

    output_path = tmp_path / "{rollout_id}.jsonl"
    sample = SimpleNamespace(
        index=7,
        group_index=2,
        prompt="write a program",
        metadata={
            "code_reward_trace": {
                "generated_code": "print(1)",
                "test_cases": [{"input": "", "sandbox_result": {"status": "ok"}, "output_matches": True}],
                "failure_reason": None,
            }
        },
        get_reward_value=lambda args: 1.0,
    )
    manager = SimpleNamespace(
        args=SimpleNamespace(save_debug_rollout_data=None, save_rollout_outputs=str(output_path))
    )

    RolloutManager._save_debug_rollout_data(manager, [sample], rollout_id=3, evaluation=False)

    lines = (tmp_path / "3.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "rollout_id": 3,
        "sample_index": 7,
        "group_index": 2,
        "prompt": "write a program",
        "generated_code": "print(1)",
        "test_cases": [{"input": "", "sandbox_result": {"status": "ok"}, "output_matches": True}],
        "final_reward": 1.0,
        "failure_reason": None,
    }


def _local_executor(generated, stdin):
    try:
        completed = subprocess.run(
            [sys.executable, "-c", generated],
            input=stdin,
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failed("timeout")
    if completed.returncode != 0:
        return ExecutionResult(False, "runtime_error", completed.stdout, completed.stderr, completed.returncode)
    return _ok(completed.stdout)


def test_real_eurus_standard_io_samples_from_all_four_sources(monkeypatch):
    data_path = Path("/mnt/cpfs/users/wxh/GRPO/datasets/data/train-00000-of-00007.parquet")
    if not data_path.exists():
        pytest.skip("Eurus integration dataset is not mounted")

    target_task_ids = {"taco": 13, "codecontests": 19, "codeforces": 1512, "apps": 34}
    rows = pq.read_table(data_path, columns=["data_source", "reward_model", "task_id"]).to_pylist()
    selected_rows = {
        row["data_source"]: row
        for row in rows
        if target_task_ids.get(row["data_source"]) == row["task_id"]
    }
    assert set(selected_rows) == set(target_task_ids)

    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)
    for source, row in selected_rows.items():
        parsed = code._parse_test_cases(row["reward_model"])
        assert parsed is not None, source
        test_cases = code._select_test_cases(parsed.test_cases)
        expected_by_stdin = {case.input_value: case.expected_output for case in test_cases}
        generated = (
            "import sys\n"
            f"expected_by_stdin = {expected_by_stdin!r}\n"
            "sys.stdout.write(expected_by_stdin[sys.stdin.read()])\n"
        )
        sample = SimpleNamespace(response=f"```python\n{generated}```", label=row["reward_model"])
        assert _reward(sample) == 1.0, source


def test_real_eurus_call_based_samples(monkeypatch):
    data_path = Path("/mnt/cpfs/users/wxh/GRPO/datasets/data/train-00000-of-00007.parquet")
    if not data_path.exists():
        pytest.skip("Eurus integration dataset is not mounted")

    solutions = {
        95: """class Solution:
    def minCostToMoveChips(self, position):
        odd = sum(value % 2 for value in position)
        return min(odd, len(position) - odd)
""",
        423: """def convertToTitle(columnNumber):
    answer = []
    while columnNumber:
        columnNumber -= 1
        answer.append(chr(ord('A') + columnNumber % 26))
        columnNumber //= 26
    return ''.join(reversed(answer))
""",
        956: """def average(salary):
    return (sum(salary) - min(salary) - max(salary)) / (len(salary) - 2)
""",
    }
    rows = pq.read_table(data_path, columns=["reward_model", "task_id"]).to_pylist()
    selected_rows = {row["task_id"]: row for row in rows if row["task_id"] in solutions}
    assert set(selected_rows) == set(solutions)

    monkeypatch.setattr(code.code_sandbox, "run_code", _local_executor)
    for task_id, generated in solutions.items():
        row = selected_rows[task_id]
        sample = SimpleNamespace(response=f"```python\n{generated}```", label=row["reward_model"])
        assert _reward(sample) == 1.0, task_id

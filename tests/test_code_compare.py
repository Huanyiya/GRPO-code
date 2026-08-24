from slime_plugins.rewards.code_compare import compare_call_output, compare_standard_output


def test_standard_whitespace_and_numeric_tolerance():
    assert compare_standard_output("1.0000001  2\r\n", "1.0 2.0\n")
    assert not compare_standard_output("1.01", "1.0")
    assert compare_standard_output("answer  \r\n", "answer\n\n")


def test_recursive_call_output_comparison():
    expected = {"items": [1, 2.0, {"ok": True}], "name": "x"}
    assert compare_call_output('{"items": [1, 2.0000001, {"ok": true}], "name": "x"}', expected)
    assert not compare_call_output('{"items": [2, 1, {"ok": true}], "name": "x"}', expected)


def test_tuple_expected_is_normalized_to_sequence():
    assert compare_call_output("[1, 2, 3]", (1, 2, 3))


def test_malformed_json_stdout_fails():
    assert not compare_call_output("debug output\n[1, 2]", [1, 2])

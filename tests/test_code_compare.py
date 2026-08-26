from slime_plugins.rewards.code_compare import compare_call_output, compare_standard_output


def test_standard_whitespace_and_numeric_tolerance():
    assert not compare_standard_output("1.0000001  2\r\n", "1.0 2.0\n")
    assert compare_standard_output("1.0000001  2\r\n", "1.0 2.0\n", abs_tol="1e-6", rel_tol="1e-6")
    assert not compare_standard_output("1.01", "1.0")
    assert compare_standard_output("answer  \r\n", "answer\n\n")
    assert compare_standard_output("YES   NO\tYES\n", "YES NO YES")
    assert not compare_standard_output("4321.0", "4321")
    assert not compare_standard_output("001", "1")


def test_recursive_call_output_comparison():
    expected = {"items": [1, 2.0, {"ok": True}], "name": "x"}
    actual = '{"items": [1, 2.0000001, {"ok": true}], "name": "x"}'
    assert not compare_call_output(actual, expected)
    assert compare_call_output(actual, expected, abs_tol="1e-6", rel_tol="1e-6")
    assert not compare_call_output('{"items": [2, 1, {"ok": true}], "name": "x"}', expected)


def test_tuple_expected_is_normalized_to_sequence():
    assert compare_call_output("[1, 2, 3]", (1, 2, 3))


def test_eurus_singleton_wrapped_scalar_output():
    assert compare_call_output('"retsec"', ["retsec"])
    assert compare_call_output("4321", [4321])
    assert compare_call_output('"Yes"', ["Yes"])


def test_eurus_singleton_wrapped_list_output():
    assert compare_call_output(
        "[1, 2, 3, 4, 5]",
        [[1, 2, 3, 4, 5]],
    )


def test_normal_list_return_still_works():
    assert compare_call_output(
        "[1, 2, 3]",
        [1, 2, 3],
    )


def test_singleton_fallback_does_not_accept_wrong_value():
    assert not compare_call_output('"abc"', ["def"])
    assert not compare_call_output("1", [2])
    assert not compare_call_output("4321.0", 4321)


def test_malformed_json_stdout_fails():
    assert not compare_call_output("debug output\n[1, 2]", [1, 2])

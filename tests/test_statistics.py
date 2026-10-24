import json

from cats_npu.statistics import analyze, read_observations


def test_sessions_not_iterations_for_significance():
    rows = []
    for session in ("same-session",):
        for method, value in [("cpu_exact", 10), ("numpy_adaptive", 5)]:
            for block in range(15):
                for trial in range(10):
                    rows.append(dict(session_id=session, case_id="a", method=method,
                                     block_id=block, iteration=trial, elapsed_ms=value,
                                     correct=True))
    report = analyze(rows)
    comp = report["cases"]["a"]["comparisons_vs_baseline"]["numpy_adaptive"]
    assert comp["paired_sessions"] == 1
    assert comp["significance_status"] == "exploratory_insufficient_sessions"
    assert "two_sided_signflip_p" not in comp


def test_multi_session_statistics():
    rows = []
    for n in range(8):
        for method, value in [("cpu_exact", 10 + n / 10), ("numpy_adaptive", 5 + n / 20)]:
            rows.append(dict(session_id=f"session-{n}", case_id="c", method=method,
                             elapsed_ms=value, correct=True))
    comp = analyze(rows)["cases"]["c"]["comparisons_vs_baseline"]["numpy_adaptive"]
    assert comp["paired_sessions"] == 8
    assert comp["significance_status"] == "ready"
    assert comp["holm_adjusted_p"] <= 0.05
    assert comp["geometric_speedup"] > 1.9
    assert len(comp["bootstrap_95ci_speedup"]) == 2


def test_incorrect_case_not_used_for_significance():
    records = []
    for s in range(8):
        for method, t in [('cpu_exact', 10), ('bad_fast', 1)]:
            records.append(dict(session_id=f's{s}', case_id='X', method=method,
                                elapsed_ms=t, correct=True))
    records.append(dict(session_id='s0', case_id='X', method='bad_fast',
                        elapsed_ms=0.5, correct=False))
    report = analyze(records)
    assert report['cases']['X']['methods']['bad_fast']['valid_for_inference'] is False
    assert 'bad_fast' not in report['cases']['X']['comparisons_vs_baseline']


def test_different_input_bytes_rejected():
    records = [dict(session_id='s', case_id='X', method='cpu_exact',
                    elapsed_ms=1, correct=True, input_sha256='aaa'),
               dict(session_id='s2', case_id='X', method='cpu_exact',
                    elapsed_ms=1, correct=True, input_sha256='bbb')]
    import pytest
    with pytest.raises(ValueError, match='Score bytes differ'):
        analyze(records)


def test_failed_session_is_descriptive_but_excluded_from_inference(tmp_path):
    session = "partial"
    observation = {
        "session_id": session, "case_id": "X", "method": "cpu_exact",
        "elapsed_ms": 1.0, "correct": True,
    }
    (tmp_path / f"observations_{session}.jsonl").write_text(
        json.dumps(observation) + "\n", encoding="utf-8"
    )
    (tmp_path / f"summary_{session}.json").write_text(
        json.dumps({"session_id": session, "completed": True,
                    "skipped": 1, "wrong_results": 0}) + "\n", encoding="utf-8"
    )
    report = analyze(read_observations([tmp_path]))
    method = report["cases"]["X"]["methods"]["cpu_exact"]
    assert method["n_observations"] == 1
    assert method["valid_session_observations"] == 0
    assert method["excluded_failed_session_observations"] == 1
    assert method["valid_for_inference"] is False

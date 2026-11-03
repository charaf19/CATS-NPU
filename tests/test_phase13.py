import numpy as np

from scripts.phase13_analyze import bootstrap_log_ci, comparison, crossed_ci, holm, sign_flip
from scripts.phase13_design import numerical_cases, score_cases
from scripts.phase13_prepare import numerical_array


def test_exact_signflip_and_holm():
    assert sign_flip([1, 1, 1, 1]) == 2 / 16
    assert sign_flip([0, 0, 0]) == 1
    assert sign_flip([1, -1, 1, -1]) == 1
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_bootstrap_resamples_clusters_and_respects_ratio_direction():
    effects = np.log([1.25] * 12)
    np.testing.assert_allclose(bootstrap_log_ci(effects, 1, 100), [1.25, 1.25])
    np.testing.assert_allclose(crossed_ci(np.full((12, 8), np.log(1.25)), 1, 100), [1.25, 1.25])
    r = comparison(np.full((12, 8), np.log(1.25)),
                   {"bootstrap_resamples": 100, "crossed_bootstrap_resamples": 100, "timed_repetitions": 6}, 2)
    assert abs(r["ratio"] - 1.25) < 1e-12
    assert r["exact_signflip_p"] == 2 / 4096


def test_numerical_panel_is_finite_valid_and_exposes_different_tiles():
    panel = numerical_cases()
    assert len(panel) == 54
    for case in panel:
        a = numerical_array(case)
        assert a.dtype == np.float16 and len(a) == case["n"] and np.isfinite(a).all()
        assert 1 <= case["k"] <= case["w"] < len(a)
        if case["family"] == "cross_tile_boundary":
            ids = np.lexsort((np.arange(len(a)), -a.astype(float)))[:case["k"]]
            assert np.all((ids >= case["w"]) & (ids < 2 * case["w"]))


def test_workload_matrix_avoids_cartesian_explosion():
    selection = {d: {"selected_ids": [str(i) for i in range(8)]} for d in ("sift", "scifact", "fiqa")}
    cases = score_cases(selection)
    assert len(cases) == 48
    assert len([x for x in cases if x["role"] == "primary"]) == 8
    assert len(set(x["id"] for x in cases)) == 48

"""Verify the Phase 10 scalar diagnostic, without assuming device semantics."""
import numpy as np
import pytest

from scripts.audit_contract_lowering import analyze_models, rank_profile


def test_rank_contract_permits_value_changes_but_rejects_false_ties():
    domain = np.array([-2.0, 0.0, 0.0, 3.0])
    assert rank_profile(domain, np.array([10, 20, 20, 30]))[
        "preserves_enumerated_rank_contract"]
    failed = rank_profile(domain, np.array([10, 10, 10, 30]))
    assert failed["counterexample"]["expected_ids_k1"] == [1]
    assert failed["counterexample"]["model_ids_k1"] == [0]


@pytest.mark.parametrize("mapping", [[0, 1], [1, 0]])
def test_signed_zero_splits_fail_for_either_input_orientation(mapping):
    failed = rank_profile(np.array([-0.0, 0.0]), np.array(mapping))
    assert failed["adjacent_equal_score_splits"] == 1
    assert failed["counterexample"]["expected_ids_k1"] == [0]
    assert failed["counterexample"]["model_ids_k1"] == [1]


def test_scalar_domain_analysis_is_explicitly_conditional():
    result = analyze_models()
    assert all(result["internal_checks"].values())
    failure = result["models"]["centered_int16_via_fp16_model"]
    assert failure["counterexample"]["source_binary16_words"] == ["0xfbff", "0xfbfe"]
    assert failure["adjacent_order_inversions"] == 0
    assert failure["adjacent_distinct_score_collisions"] == 51327


def test_counterexample_reference_keeps_int64_precision():
    failed = rank_profile(np.array([2**60, 2**60 + 1], dtype=np.int64), np.array([0, 0]))
    assert failed["counterexample"]["source_scores"] == [2**60, 2**60 + 1]
    assert failed["counterexample"]["expected_ids_k1"] == [1]


@pytest.mark.parametrize("source,target", [([1, 0], [0, 1]), ([0, 1], [0]),
                                           ([0, np.nan], [0, 1])])
def test_diagnostic_rejects_malformed_domains(source, target):
    with pytest.raises(ValueError):
        rank_profile(np.array(source), np.array(target))

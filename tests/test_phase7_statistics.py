from pathlib import Path

import numpy as np

from scripts.analyze_phase7_confirmatory import (
    bootstrap_ci,
    exact_sign_flip,
    write_primary_figure,
)


def test_six_consistent_session_effects_reach_exact_test_resolution():
    effects = np.log(np.asarray([1.20, 1.25, 1.30, 1.35, 1.40, 1.45]))
    assert exact_sign_flip(effects) == 0.03125
    low, high = bootstrap_ci(effects)
    assert 1 < low < high


def test_primary_figure_records_session_effects_and_interval(tmp_path: Path):
    output = tmp_path / "primary.svg"
    primary = {
        "session_speedups": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        "ci95_session_cluster_bootstrap": [1.25, 1.65],
        "geomean_speedup": 1.43,
    }
    write_primary_figure(output, primary)
    text = output.read_text(encoding="utf-8")
    assert text.startswith("<svg")
    assert "Session 6" in text
    assert "1.430x [1.250, 1.650]" in text

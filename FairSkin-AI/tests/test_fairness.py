import numpy as np

from fairskin_ai.fairness import bootstrap_group_metric, group_metrics


def test_group_metrics_and_gap():
    y_true = [0,0,1,1,0,1,0,1]
    y_pred = [0,0,1,1,0,0,1,1]
    groups = ["I-II"]*4 + ["V-VI"]*4
    report = group_metrics(y_true, y_pred, groups)
    assert report["groups"]["I-II"]["accuracy"] == 1.0
    assert 0.0 <= report["disparity"]["accuracy"]["gap"] <= 1.0


def test_bootstrap_is_bounded():
    out = bootstrap_group_metric([0,1,0,1], [0,1,1,1], ["A","A","B","B"], n_bootstrap=10)
    for g in out.values():
        assert 0 <= g["ci_low"] <= 1
        assert 0 <= g["ci_high"] <= 1

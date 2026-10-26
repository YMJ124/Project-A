import numpy as np
import pandas as pd

from beer_q3.migration_evaluation import _aggregate_decision, _fold_positions


def test_fold_positions_hold_validation_rows_out_of_training():
    ids = pd.Series([10, 11, 12, 13, 14, 15])
    folds = pd.DataFrame(
        {
            "beer": ids,
            "style": ["a"] * 6,
            "repeat": [25] * 6,
            "seed": [123] * 6,
            "fold": [0, 1, 2, 0, 1, 2],
        }
    )
    train, valid, seed = _fold_positions(ids, folds, 25, 1)
    assert seed == 123
    assert set(train).isdisjoint(valid)
    assert ids.iloc[valid].tolist() == [11, 14]
    assert sorted(np.r_[train, valid].tolist()) == list(range(6))


def test_aggregate_decision_enforces_hard_and_stability_gates():
    selected_rows = []
    sensory_rows = []
    for repeat in [25, 26]:
        for source in [1, 2]:
            selected_rows.append(
                {
                    "repeat": repeat,
                    "source_beer": source,
                    "style_probability_gain": 0.1,
                    "target_distance_improvement": 0.2,
                    "standardized_l1": 0.3,
                    "standardized_l2": 0.4,
                    "changed_fields": 5,
                    "changed_modules": 2,
                    "hard_constraints_pass": True,
                    "exact_training_copy": False,
                    "near_neighbor_danger": False,
                }
            )
            sensory_rows.append(
                {
                    "repeat": repeat,
                    "source_beer": source,
                    "target_scope": "ab_only",
                    "sensory_distance_improvement": 0.05,
                    "sensory_target_improved": True,
                }
            )
    config = {
        "selection": {
            "minimum_row_positive_rate_style": 0.8,
            "minimum_row_positive_rate_chemistry": 0.8,
            "minimum_repeat_mean_wins_style": 2,
            "minimum_repeat_mean_wins_chemistry": 2,
            "minimum_source_median_wins_style": 2,
            "minimum_source_median_wins_chemistry": 2,
            "minimum_row_positive_rate_ab_sensory": 0.7,
            "minimum_repeat_mean_wins_ab_sensory": 2,
            "minimum_source_median_wins_ab_sensory": 2,
        }
    }
    conditions, repeats, sources, statistics = _aggregate_decision(
        pd.DataFrame(selected_rows),
        pd.DataFrame(sensory_rows),
        config,
        expected_rows=4,
    )
    assert all(conditions.values())
    assert len(repeats) == 2
    assert len(sources) == 2
    assert statistics["style_positive_row_rate"] == 1.0

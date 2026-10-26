import json

import pandas as pd
import yaml

from beer_q3.migration_comparison import compare_sparse_migration


def test_sparse_migration_comparison_accepts_cost_reduction_with_retained_gains(tmp_path):
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    rows = []
    sensory = []
    for repeat in [1, 2]:
        for source in [10, 11]:
            rows.append(
                {
                    "repeat": repeat,
                    "fold": source % 2,
                    "source_beer": source,
                    "method": "pca_centroid_shift",
                    "style_probability_gain": 0.10,
                    "target_distance_improvement": 0.20,
                    "changed_fields": 100,
                    "changed_modules": 20,
                    "standardized_l1": 0.5,
                    "standardized_l2": 1.0,
                }
            )
            sensory.append(
                {
                    "repeat": repeat,
                    "fold": source % 2,
                    "source_beer": source,
                    "target_scope": "ab_only",
                    "sensory_distance_improvement": 0.10,
                }
            )
    baseline = pd.DataFrame(rows)
    candidate = baseline.copy()
    candidate["method"] = "module_counterfactual_6"
    candidate["style_probability_gain"] *= 0.8
    candidate["target_distance_improvement"] *= 0.9
    candidate["changed_fields"] = 70
    candidate["changed_modules"] = 12
    candidate["standardized_l1"] *= 0.8
    candidate["standardized_l2"] *= 0.9
    for root, metrics in [(baseline_root, baseline), (candidate_root, candidate)]:
        (root / "run_manifest.json").write_text(
            json.dumps({"criteria_passed": True}), encoding="utf-8"
        )
        metrics.to_csv(root / "migration_selected_metrics.csv", index=False)
        pd.DataFrame(sensory).to_csv(root / "migration_sensory_metrics.csv", index=False)
    config = {
        "interfaces": {
            "baseline_root": str(baseline_root),
            "candidate_root": str(candidate_root),
        },
        "selection": {
            "minimum_style_gain_retention": 0.75,
            "minimum_chemistry_gain_retention": 0.75,
            "minimum_ab_sensory_mean_improvement_retention": 0.90,
            "maximum_additional_ab_sensory_failures": 2,
            "maximum_changed_fields_ratio": 0.80,
            "maximum_changed_modules_ratio": 0.80,
            "maximum_standardized_l1_ratio": 1.0,
            "maximum_standardized_l2_ratio": 1.0,
            "minimum_repeats_with_both_field_and_module_reduction": 2,
        },
        "runtime": {
            "run_name": "test",
            "output_root": str(tmp_path / "output"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = compare_sparse_migration(tmp_path, config_path)
    assert result["status"] == "SPARSE_MIGRATION_CONFIRMED"
    assert result["criteria_passed"] is True

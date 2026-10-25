from pathlib import Path

import pandas as pd
import yaml

from beer_q3.q2_handoff import build_q2_handoff, write_readme


def test_q2_handoff_rebuilds_from_all_locked_partitions(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    source = project_root / "problem3/configs/q2_handoff_v1.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    output = tmp_path / "q2_handoff"
    config["runtime"]["output_root"] = str(output)
    temporary_config = tmp_path / "config.yaml"
    temporary_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    summary = build_q2_handoff(project_root, temporary_config)
    write_readme(output, summary)
    assert summary["partition_count"] == 21
    assert summary["consensus_edge_count"] == 710
    assert summary["consensus_module_count"] == 60
    assert summary["q2_targets_without_candidates"] == [
        "A_hops_citrus",
        "A_hops_tropical",
        "A_malt_bread",
        "F_hops_citrus",
        "F_malt_grain",
    ]
    assert (output / "SHA256SUMS").is_file()


def test_q2_handoff_has_locked_coverage_and_safe_roles():
    root = Path(__file__).resolve().parents[1] / "interfaces/q2_handoff_v1"
    registry = pd.read_csv(root / "actionability_registry.csv")
    candidates = pd.read_csv(root / "candidate_handoff.csv")
    communities = pd.read_csv(root / "consensus_communities.csv")
    edges = pd.read_csv(root / "consensus_edges.csv")

    assert len(registry) == registry["feature"].nunique() == 231
    assert len(communities) == communities["feature"].nunique() == 231
    assert len(candidates) == 46
    assert candidates["target"].nunique() == 25
    assert not candidates["causal_direction_claimed"].any()
    assert not registry["causal_actionability_claimed"].any()
    assert not registry["direct_process_control_claimed"].any()

    derived = registry.loc[registry["generation_role"] == "derived_read_only"]
    assert set(derived["feature"]) == {
        "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"
    }
    assert not derived["generator_decision_variable"].any()
    consensus = edges.loc[edges["consensus_edge"]]
    assert (consensus["partition_presence_count"] >= 14).all()
    assert (consensus["cross_partition_sign_consistency"] >= 0.90).all()


def test_q2_handoff_checksums_cover_all_other_interface_files():
    root = Path(__file__).resolve().parents[1] / "interfaces/q2_handoff_v1"
    names = {
        line.split("  ", 1)[1]
        for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    expected = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    }
    assert names == expected

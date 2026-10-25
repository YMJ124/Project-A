from pathlib import Path

import pandas as pd


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

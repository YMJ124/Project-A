from pathlib import Path

from beer_q2.reconstruct import reconstruct_and_compare


def test_representative_q1_reconstruction_one_repeat(tmp_path):
    result = reconstruct_and_compare(
        Path(__file__).resolve().parents[2],
        targets=["A_hops_all", "F_esters_isoaa", "A_hops_woody"],
        repeats=range(1),
        output_dir=tmp_path,
    )
    assert len(result) == 9
    assert result["passed_tolerance"].all()
    assert result["max_abs_error"].max() < 1e-9

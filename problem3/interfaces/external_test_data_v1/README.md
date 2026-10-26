# Public external sensory audit interface v1

This interface locks the public cohort membership, input hashes, missing-profile rule, and three sensory anchors before any Question 3 champion is scored.

The public descriptor columns are review word frequencies, not professional panel intensities, chemical concentrations, or original 0--6 labels. They are transformed with `log1p` and row-wise composition only after the generator champion is locked. Rows whose eleven descriptor counts sum to zero remain missing and are excluded from anchor scoring.

This freeze computes no external model score and must not be used for generator tuning, stopping, rejection thresholds, or champion replacement.

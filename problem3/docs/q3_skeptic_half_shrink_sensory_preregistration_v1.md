# Q3 half-shrunk SKEPTIC sensory confirmation pre-registration v1

## Candidate and evidence boundary

The only input candidate is the chemistry-confirmed
`skeptic_spearman_rank_gaussian_copula_prior_5_blend_0.5_distance_q0.05`
from fresh repeats 7--10. Its fold-matched comparator is the unchanged prior-5
rank-Gaussian copula. No external public cohort is opened.

For each outer fold, the immutable Q1 v1 source registry is refitted using only
that fold's training rows. The modal frozen Extra-Trees configuration is used
when the new repeat has no historical tuning row; this is the already frozen
deployment rule, not new model selection. Candidate, comparator and real outer
validation chemicals are passed through the same fitted Q1 models.

## Sensory quantities

- The 24 frozen Q1 A/B targets are primary; six C targets are audit-only.
- Generated and real-validation sensory profiles are model predictions on the
  standardized Q1 scale. Actual validation sensory labels are not used to tune
  or select the generator.
- Target-wise Wasserstein distance is divided by the robust scale of the outer
  training target. A disjoint training-style real--real reference with 100
  resamples supplies each fold/style/target 95% acceptance line.
- Relative 0--6 values use only the outer-training target median and
  `IQR/1.349`, with the already documented `clip(3 + z_robust, 0, 6)` mapping.
- Prediction intervals add the frozen Q1 OOF residual 2.5% and 97.5% quantiles;
  they are empirical uncertainty bands, not original 0--6 confidence claims.

## Locked sensory protection rules

Across the 36 fold/style cells and 24 A/B targets, the candidate must satisfy:

1. complete finite schema and exactly 864 primary metric rows;
2. real--real-reference PASS rate no lower than the prior-5 comparator;
3. mean normalized sensory Wasserstein distance no higher than the comparator;
4. within each of Tripel, Kriek and Lambic, mean distance no more than 5% above
   the comparator;
5. chemistry confirmation remains passed and all C targets remain explicitly
   audit-only.

Failure on any item rejects champion promotion without changing Q1 models,
targets, scales, intervals or thresholds. Passing establishes a conditional Q3
generation champion only; Q1 remains `HOLD`, migration remains unfinished and
the overall problem cannot yet be called `READY`.


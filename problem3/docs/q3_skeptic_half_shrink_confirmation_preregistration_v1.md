# Q3 half-shrunk Spearman-SKEPTIC fresh confirmation v1

## Locked candidate

Only the following discovery candidate may be evaluated:

`skeptic_spearman_rank_gaussian_copula_prior_5_blend_0.5_distance_q0.05`

It uses the arithmetic midpoint of the prior-5 Ledoit--Wolf and prior-5
Spearman-SKEPTIC latent correlations, followed by the fixed q05
training-distance rejection layer. No model component or threshold may change.

## Fresh evidence boundary

- Four new style-balanced outer repeats are generated with base seed
  `20501124` and labeled repeats 7--10.
- Each repeat has three folds and every beer appears in validation exactly once
  per repeat.
- These seeds and assignments are fixed before any candidate score is viewed.
- Repeats 3--6 are excluded because their outcomes were inspected for the
  rejected POET candidate.
- A fold-matched prior-5 rank-Gaussian baseline and real--real reference are
  regenerated on the same new folds.

## Locked confirmation rules

The candidate must satisfy all conditions over 36 fold/style cells:

1. all 36 chemistry hard-gate cells pass;
2. Wasserstein-core and overall real-reference distance PASS rates do not
   decline from the fold-matched prior-5 baseline;
3. mean style-hit rate declines by no more than 0.05;
4. Spearman-structure and Q2 edge-strength PASS rates do not decline;
5. Q2 edge-strength PASS is at least 0.764815 and improves by at least 0.05.

Failure on any condition rejects the candidate without tuning. Passing confirms
chemistry only; Q1 sensory evaluation is still required before champion
eligibility. External public cohorts remain locked and unscored.


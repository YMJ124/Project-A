# Q3 fixed-factor POET independent confirmation pre-registration v1

## Locked candidate

The only candidate is
`factor_rank_gaussian_copula_prior_5_factors_10_poet_c1_distance_q0.05`.
No parameter, threshold, seed rule, margin, constraint or evaluator may change
after the discovery result.

## Independent evidence

Confirmation uses frozen repeats 3--6 and folds 0--2, which were not used to
select the factor count, POET constant or distance rule. A fold-matched prior-5
rank-Gaussian copula and real--real reference distribution will be regenerated
before candidate scoring. Other baseline outputs are audit-only.

## Adoption rules

The same discovery rules apply without relaxation:

1. all 36 fold-style chemistry hard-gate cells pass;
2. Wasserstein-core and overall distance pass rates do not decline from the
   fold-matched prior-5 reference;
3. mean generated style-hit rate is no more than 0.05 below prior 5;
4. Spearman-structure and Q2 edge-strength pass rates do not decline;
5. Q2 edge-strength PASS is at least 0.764815 and improves by at least 0.05.

A full pass establishes a chemistry-confirmed candidate only. Q1 sensory
evaluation is still required before champion eligibility. Failure of any rule
rejects the candidate without threshold or parameter adjustment.

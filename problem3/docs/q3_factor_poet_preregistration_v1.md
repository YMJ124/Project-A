# Q3 fixed-factor POET residual pre-registration v1

## Starting point

The only parent model is the fixed 10-factor rank-Gaussian copula with the
outer-training q05 minimum-distance rejection layer. On discovery repeats
0--2 it passed all 27 hard-gate cells, protected style and marginal-distance
summaries, and raised Q2 edge-strength PASS from 18/27 to 22/27. It was
rejected because Spearman-structure PASS was 23/27 versus 24/27 for prior 5.

## Single mechanism

Keep the factor count at 10, style prior at 5 and the q05 distance layer. Add
only a POET-style sparse residual covariance after removing the ten shared
factors:

1. estimate global and target-style residual sample covariance inside the
   outer training fold;
2. combine them with the unchanged prior-5 target weight;
3. estimate the entrywise covariance standard error from global and target
   residual products;
4. hard-threshold each off-diagonal residual covariance at
   `sqrt(estimated_variance * log(p))`, corresponding to fixed constant
   `C=1`;
5. add the retained residual covariance to the factor covariance, repair to a
   valid correlation matrix, and restore the untouched prior-5 marginal
   variances.

Training-fold diagnostics made before scoring show that this fixed rule keeps
6.4%--15.3% of residual edges across the 27 discovery fold-style cells. No
alternative constant, soft threshold, Q2-edge whitelist or module rule will be
screened.

## Locked adoption rules

The candidate must satisfy every existing condition:

1. all 27 chemistry hard-gate cells pass;
2. Wasserstein-core and overall distance pass rates do not decline from the
   prior-5 reference;
3. mean style-hit rate falls by no more than 0.05;
4. Spearman-structure and Q2 edge-strength pass rates do not decline;
5. Q2 edge-strength PASS is at least 0.764815 and improves by at least 0.05.

Only discovery repeats 0--2 may be used. Confirmation repeats 3--6, Q1 sensory
inference and locked external cohorts remain untouched. A discovery pass locks
a candidate for later independent confirmation; it is not a champion.

Primary methodological reference:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC3859166/

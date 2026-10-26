# Q3 half-shrunk Spearman-SKEPTIC copula pre-registration v1

## Discovery diagnosis

The full Spearman-SKEPTIC dependence replacement is rejected. Relative to the
prior-5 rank-Gaussian reference on repeats 0--2, it improved style hit by
0.1006, Wasserstein-core PASS from 14/27 to 16/27 and Q2 edge-strength PASS
from 18/27 to 24/27, while preserving all hard gates. Its sole locked failure
was an overly strong global structure change: Spearman-structure PASS fell
from 24/27 to 20/27.

## Single conservative mechanism

Use the arithmetic midpoint of two valid latent correlation estimators fitted
inside the same outer training fold:

- 50% unchanged prior-5 Ledoit--Wolf rank-Gaussian correlation;
- 50% prior-5 Spearman-SKEPTIC latent correlation.

The weight is fixed at 0.5 before running this experiment. No alternative
weight, prior, factor count, threshold, rank statistic or module rule will be
screened. The resulting convex combination is projected to a correlation
matrix for numerical safety. Means, marginal variances, empirical inverse
margins, q05 minimum-distance rejection, chemical constraints, folds, sample
counts and seeds stay unchanged.

This is a standard shrinkage response to the observed variance/bias tradeoff:
the SKEPTIC estimator directly targets rank dependence, while the current
Ledoit--Wolf estimate supplies the protected stable center. It is not an
adoption of the rejected full SKEPTIC result.

## Locked discovery rules

Against the fold-matched prior-5 reference, the candidate must:

1. pass all 27 chemistry hard-gate cells;
2. preserve Wasserstein-core and overall distance PASS rates;
3. lose no more than 0.05 mean style-hit rate;
4. preserve Spearman-structure and Q2 edge-strength PASS rates;
5. reach Q2 edge-strength PASS at least 0.764815 and improve it by at least
   0.05.

Discovery is limited to repeats 0--2. If it passes, fresh outer repeats must be
created and pre-registered because repeats 3--6 have already been viewed. A
chemistry confirmation would still require Q1 sensory evaluation. External
public cohorts remain locked and unscored.

References:

- https://arxiv.org/abs/1206.6488
- https://doi.org/10.1214/12-AOS989


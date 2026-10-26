# Q3 rank-Gaussian factor copula pre-registration v1

## Motivation

Stronger covariance shrinkage recovers the Q2 edge-strength and Spearman
targets but erases style identity and worsens marginal distances. This suggests
that the useful style covariance is structured rather than dense. The full
rank-Gaussian correlation spectrum also concentrates about 56% of variance in
10 factors and 64% in 15 factors.

## Single mechanism

Replace only the off-diagonal latent dependence representation by a shared
global PCA basis with target-style-shrunk factor covariance and diagonal
residual variance. Candidate factor counts are 5, 10 and 15, ordered from
simplest to most flexible. The prior-5 full latent mean, every prior-5 latent
marginal variance and all empirical inverse margins remain unchanged.

No sparse-residual threshold, Q2 module replacement, rejection layer or
additional calibration is allowed in this experiment.

## Locked rules

The first candidate satisfying every condition is selected:

1. all 27 hard-gate cells pass;
2. Wasserstein-core and overall distance pass rates do not decline;
3. style-hit mean falls by no more than 0.05;
4. Spearman and Q2 edge-strength pass rates do not decline;
5. Q2 edge-strength pass rate is at least 0.764815 and improves by at least
   0.05 over prior 5.

Only frozen discovery repeats 0--2 are used. Confirmation repeats 3--6, Q1
sensory inference and external cohorts stay untouched.

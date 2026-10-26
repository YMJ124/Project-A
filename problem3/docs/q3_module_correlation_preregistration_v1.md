# Q3 module-correlation screening pre-registration v1

## Observed bottleneck

The prior-5 rank Gaussian copula passes all 27 chemistry hard-gate cells, but
its Q2 edge-strength pass rate is only 18/27. Replacing full within-module
covariance blocks at weight 0.75 recovers edge strength to 21/27 and keeps all
hard gates, but loses one Spearman-structure cell (24/27 to 23/27). Full block
replacement also changes latent marginal variances, so dependence repair and
marginal behavior are not isolated.

## Single mechanism

Keep the prior-5 full-copula mean, every latent marginal variance and every
empirical inverse marginal fixed. Blend only the within-Q2-module correlation
blocks at weights 0.25, 0.5 and 0.75. Repair the assembled matrix to a
positive-semidefinite unit-diagonal correlation matrix before reconstructing
the covariance with the untouched prior-5 marginal scales.

No fold, random seed, target style, defect constraint, derived-feature rule,
evaluator, reference interval or baseline changes in this experiment.

## Locked discovery choice and protection rules

Candidates are ordered from the smallest to largest correlation replacement
weight. The first candidate satisfying every condition is selected:

1. all 27 fold-style hard-gate cells pass;
2. Wasserstein-core pass rate is no lower than the prior-5 baseline;
3. overall real-reference distance pass rate is no lower than prior-5;
4. mean generated style-hit rate is no more than 0.05 below prior-5;
5. Spearman-structure pass rate does not drop at all;
6. Q2 edge-strength pass rate does not drop and is at least 0.764815;
7. Q2 edge-strength pass rate improves by at least 0.05 over prior-5.

Discovery uses only frozen repeats 0--2. Repeats 3--6, Q1 sensory inference
and the locked external cohorts remain untouched. A discovery pass is only a
candidate for independent confirmation, not a champion.

## Broader small-n, high-p methods considered

The following mechanisms were reviewed but are deliberately not mixed into
this experiment:

- Ledoit--Wolf and Schäfer--Strimmer shrinkage support regularizing covariance
  or correlation rather than using the unstable sample matrix directly. The
  latter explicitly separates correlation shrinkage from retained variances,
  which is the closest justification for the present mechanism.
- The nonparanormal literature supports a Gaussian copula with nonparametric
  margins and rank-based dependence estimation. It fits the current separation
  of empirical chemical margins from latent dependence.
- POET models high-dimensional covariance as low rank plus a sparse residual.
  A copula-factor version is the next candidate if correlation-only repair
  fails, because it reduces the dependence parameter count without discarding
  empirical margins.
- Gaussian copula factor models similarly decouple margins and dependence, but
  the full Bayesian implementation would add MCMC and prior-sensitivity costs.
- Graphical Lasso regularizes a sparse precision matrix. It is lower priority
  here because each target style has very few observations, so penalty tuning
  and false edge deletion would need a separate, carefully nested experiment.

Primary sources:

- https://doi.org/10.1016/S0047-259X(03)00096-4
- https://strimmerlab.github.io/publications/journals/shrinkcov2005.pdf
- https://www.jmlr.org/papers/v10/liu09a.html
- https://eprints.maths.manchester.ac.uk/232/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3859166/
- https://doi.org/10.1080/01621459.2012.762328
- https://hastie.su.domains/Papers/graph.pdf

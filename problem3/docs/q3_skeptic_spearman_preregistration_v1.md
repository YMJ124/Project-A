# Q3 Spearman-SKEPTIC copula pre-registration v1

## Evidence and protected baseline

The fixed 10-factor + POET-C1 + q05-distance candidate passed discovery but
failed its independent repeats 3--6 confirmation. It improved Q2 edge-strength
PASS from 24/36 to 33/36 and repaired all chemistry hard gates, but reduced
Wasserstein-core PASS from 41/72 to 39/72 and Spearman-structure PASS from
34/36 to 33/36. It is therefore rejected without threshold retuning.

The protected discovery reference remains the prior-5 rank-Gaussian copula on
repeats 0--2. This experiment may use only those discovery repeats.

## Single mechanism

Replace only the latent correlation estimator by the Spearman version of the
nonparanormal SKEPTIC estimator:

1. use the unchanged training-fold median imputation and empirical margins;
2. estimate global and target-style Spearman correlation matrices inside the
   outer training fold;
3. map each defined off-diagonal rank correlation to latent Gaussian
   correlation with `2 * sin(pi * rho_s / 6)`;
4. where a target-style feature is constant, fall back to the corresponding
   global training-fold latent correlation rather than interpreting the
   undefined target correlation as zero;
5. blend target and global latent correlations with the unchanged prior-5
   weight `n_style / (n_style + 5)`;
6. project once to a valid correlation matrix, then restore the unchanged
   prior-5 marginal latent variances and mean;
7. retain the fixed training-fold q05 minimum-neighbour-distance rejection
   layer and every existing chemical/derived/defect constraint.

There is one candidate and no correlation threshold, graphical-lasso penalty,
module whitelist, factor count or alternate rank statistic to select. Margins,
style mean, marginal variance, generated sample count, folds and random seeds
stay fixed.

## Locked discovery rules

The candidate must satisfy all conditions against the fold-matched prior-5
reference:

1. all 27 chemistry hard-gate cells pass;
2. Wasserstein-core and overall real-reference distance PASS rates do not
   decline;
3. mean style-hit rate declines by no more than 0.05;
4. Spearman-structure and Q2 edge-strength PASS rates do not decline;
5. Q2 edge-strength PASS is at least 0.764815 and improves by at least 0.05.

A discovery pass only locks a candidate. Repeats 3--6 have already been viewed
for the rejected POET candidate and cannot provide fresh confirmation for a new
mechanism. Any later confirmation must use newly generated, pre-registered
outer repeats and still pass Q1 sensory evaluation before champion eligibility.
The external public-data cohorts remain untouched.

## Primary methodological references

- Liu, Han, Yuan, Lafferty and Wasserman (2012), *The Nonparanormal SKEPTIC*:
  https://arxiv.org/abs/1206.6488
- Liu, Lafferty and Wasserman (2009), *The Nonparanormal: Semiparametric
  Estimation of High Dimensional Undirected Graphs*:
  https://www.jmlr.org/papers/volume10/liu09a/liu09a.pdf


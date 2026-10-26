# Q3 decoupled mean/covariance shrinkage pre-registration v1

## Failure analysis motivating this experiment

The module-correlation screening did not recover Q2 edge strength while
protecting every hard gate. The underlying target-style covariance problem is
much more underdetermined than the target-style mean problem: outer-training
counts are about 18--19 for Tripel and only 6--8 for Kriek or Lambic, versus
231 chemical variables. The existing copula nevertheless uses one scalar prior
for both mean and covariance.

## Single mechanism

Keep target-style mean shrinkage fixed at prior strength 5. Increase only the
target-style covariance prior strength to 10 or 20. The rank transform,
Ledoit--Wolf estimators, empirical inverse margins, constraints, folds, seeds,
styles and evaluator remain unchanged.

Candidates are ordered covariance-prior 10 then 20. The first candidate
satisfying all of the same locked protection rules used in the preceding
module screens is selected:

1. all 27 fold-style hard-gate cells pass;
2. Wasserstein-core and overall distance pass rates do not fall below prior 5;
3. style-hit mean falls by no more than 0.05;
4. Spearman and Q2 edge-strength pass rates do not fall;
5. Q2 edge-strength pass rate is at least 0.764815 and gains at least 0.05.

Only repeats 0--2 are discovery data. Repeats 3--6, Q1 sensory inference and
external cohorts remain untouched. A pass only locks a confirmation candidate.

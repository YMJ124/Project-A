# Q3 moment-preserving latent mixture pre-registration v1

## Motivation and protected parent

The fixed 50% Ledoit--Wolf / 50% Spearman-SKEPTIC rank-Gaussian generator
with prior 5 and the fold-training q05 distance layer is the protected parent.
It passed fresh chemistry confirmation, improving Wasserstein distance, style
hit and Q2 edge strength, but failed frozen-Q1 sensory confirmation because
Tripel sensory Wasserstein deteriorated by 6.31%. The new experiment therefore
does not replace its margins or dependence estimator. It tests one new
mechanism only: whether a conservative two-component latent distribution can
represent within-style multimodality that a single Gaussian cannot.

The high-dimensional clustering literature motivates low-dimensional,
parsimonious mixtures rather than unconstrained full-covariance GMMs. Mixtures
of probabilistic PCA similarly represent nonlinear structure using a small
number of local linear components. The SKEPTIC parent is retained because its
rank-based latent dependence is designed for high-dimensional nonparanormal
data.

## Training-only feasibility audit

Before any validation score was computed, the fixed construction was audited
on the training portions of discovery repeats 0--2:

- Tripel had 18--19 training rows per fold and both clusters contained at
  least two rows in all 9 fold cells (minimum cluster size 2--8, median 7).
- Kriek had 6--7 rows and often produced a singleton cluster.
- Lambic had 7--8 rows and produced a singleton cluster in all 9 cells.

Accordingly, a component is not fitted when either cluster has fewer than two
rows; that fold/style deterministically falls back to the protected parent.
This rule is fixed before validation scoring and is not style-specific.

## Single fixed mechanism

For every outer training fold and target style:

1. Fit the protected prior-5, half-shrunk Spearman-SKEPTIC latent mean and
   covariance using training data only.
2. Whiten target-style observations in the five leading eigen-directions of
   that parent covariance.
3. Fit exactly two K-means components with 20 initializations and a fixed seed.
   If either component contains fewer than two rows, use the unchanged parent.
4. Center the two component offsets so their weighted mean is zero.
5. Scale the offsets so their between-component covariance consumes at most
   50% of the parent variance in any whitened direction.
6. Subtract this between-component covariance from one shared within-component
   covariance. Thus the mixture has exactly the same latent mean and total
   covariance as the protected parent; only higher-order/multimodal structure
   changes.
7. Retain the same empirical inverse margins, q05 minimum-distance rejection,
   chemical projections, folds, generated sample counts and evaluation code.

No component count, latent dimension, variance cap, minimum component size,
prior, SKEPTIC blend, distance quantile or clustering algorithm will be swept.

## Locked discovery decision

Discovery uses repeats 0--2 and compares against the fold-matched protected
parent. Chemistry eligibility requires all of the following:

1. all 27 hard-gate cells pass;
2. Wasserstein-core and overall real-reference distance PASS rates do not fall;
3. mean style hit falls by no more than 0.05;
4. Spearman-structure and Q2-edge-strength PASS rates do not fall;
5. Q2-edge-strength PASS remains at least 21/27.

If and only if chemistry eligibility passes, the frozen Q1 interface will be
run on the same discovery folds. The candidate must improve Tripel A/B sensory
Wasserstein, not worsen the overall A/B mean, and preserve Kriek and Lambic
within 5% relative deterioration before it can enter a new-seed confirmation.
Discovery success is not champion eligibility. External public cohorts remain
locked and unscored, and migration remains paused.

## Primary references

- Liu, Han, Yuan, Lafferty and Wasserman, *High Dimensional Semiparametric
  Gaussian Copula Graphical Models*, Annals of Statistics (2012):
  https://doi.org/10.1214/12-AOS1037
- Tipping and Bishop, *Mixtures of Probabilistic Principal Component
  Analysers*, Neural Computation (1999):
  https://doi.org/10.1162/089976699300016728
- Bouveyron, Girard and Schmid, *High-Dimensional Data Clustering*,
  Computational Statistics & Data Analysis (2007):
  https://doi.org/10.1016/j.csda.2007.02.009

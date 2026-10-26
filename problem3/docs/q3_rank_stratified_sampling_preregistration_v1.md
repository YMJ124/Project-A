# Q3 rank-stratified parent sampling pre-registration v1

## Diagnosis and protected advantages

The chemistry-confirmed prior-5, half-shrunk Spearman-SKEPTIC generator with
the q05 distance layer remains the protected fitted distribution. The rejected
moment-preserving two-component experiment showed that even exact population
mean/covariance preservation can lose one finite-batch Spearman cell when each
fold/style produces only 3--10 rows. This next experiment therefore changes no
fitted distribution parameter and adds no style component. It tests one
mechanism only: variance-reduced finite-batch sampling from the protected
parent.

Latin hypercube sampling stratifies every marginal, while the Iman--Conover
construction reorders fixed marginal values to induce a desired rank
dependence. The local construction uses these principles conservatively for
the Gaussian latent parent.

## Single fixed mechanism

For each outer-fold/style batch of size `n`:

1. Fit exactly the protected prior-5, 50% Ledoit--Wolf / 50%
   Spearman-SKEPTIC latent mean and covariance on the outer training fold.
2. Use the `n` midpoint normal scores `(i + 0.5) / n`, centered and scaled to
   variance one. Thus every latent marginal has the parent's mean and variance
   exactly within the generated batch before feasibility processing.
3. Draw one auxiliary Gaussian-copula batch from the parent's correlation and
   reorder the fixed normal scores column by column according to its ranks.
   This preserves the stratified marginal values and transfers the parent's
   rank ordering without changing any model parameter.
4. Apply the unchanged empirical inverse margins, chemical projection and q05
   minimum-distance rule. To avoid destroying the complete stratified design,
   reject and redraw the whole batch whenever any row is infeasible; do not
   cherry-pick rows from a larger pool.

There is no tunable component count, pool multiplier, mixture weight, tail
probability or score jitter. The protected parent's prior, SKEPTIC blend,
distance quantile, margins, folds, sample counts and evaluator remain fixed.

## Locked discovery decision

On repeats 0--2 against the fold-matched protected parent, chemistry
eligibility requires:

1. all 27 hard-gate cells pass;
2. Wasserstein-core and overall real-reference distance PASS rates do not fall;
3. mean style hit falls by no more than 0.05;
4. Spearman-structure and Q2-edge-strength PASS rates do not fall;
5. Q2-edge-strength PASS remains at least 21/27.

Only if chemistry eligibility passes will frozen-Q1 discovery evaluation run.
Sensory eligibility then requires all of the following: overall A/B normalized
Wasserstein is lower than the parent; A/B reference-line PASS rate does not
fall; Tripel A/B normalized Wasserstein improves; and neither Kriek nor Lambic
deteriorates by more than 5%. A passing discovery result still requires new
outer seeds for independent confirmation. Public external cohorts remain
locked and unscored, and migration remains paused.

## Primary references

- McKay, Beckman and Conover, *A Comparison of Three Methods for Selecting
  Values of Input Variables in the Analysis of Output from a Computer Code*,
  Technometrics (1979): https://doi.org/10.1080/00401706.1979.10489755
- Iman and Conover, *A Distribution-Free Approach to Inducing Rank
  Correlation Among Input Variables*, Communications in Statistics -
  Simulation and Computation (1982):
  https://doi.org/10.1080/03610918208812265
- Liu, Han, Yuan, Lafferty and Wasserman, *High Dimensional Semiparametric
  Gaussian Copula Graphical Models*, Annals of Statistics (2012):
  https://doi.org/10.1214/12-AOS1037

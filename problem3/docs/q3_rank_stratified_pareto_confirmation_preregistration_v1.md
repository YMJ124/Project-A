# Q3 rank-stratified Pareto confirmation pre-registration v1

## Evidence boundary

The rank-stratified half-shrunk SKEPTIC sampler was rejected under the original
zero-regression discovery gate because Spearman-structure PASS was 23/27 versus
24/27 for the parent. It nevertheless improved overall reference-distance PASS
by 0.1481, Wasserstein-core PASS by 0.3148, style hit by 0.1623 and Q2 edge
strength by 1/27. The user subsequently authorized a comprehensive comparison
that permits small regressions.

This document does not retroactively change or pass the old discovery result.
It prospectively defines a new Pareto validation on completely unused outer
folds. Discovery repeats 0--2 and confirmation repeats 3--10 remain viewed and
cannot provide confirmation evidence.

## Locked folds and models

- Create seven new style-balanced outer repeats numbered 11--17 with base seed
  `20601124`.
- Recompute the real--real reference and the unchanged prior-5 baseline on
  those folds.
- Generate a fold-matched protected parent: prior-5, 50% Ledoit--Wolf / 50%
  Spearman-SKEPTIC, q05 minimum-distance rejection.
- Generate the candidate with the identical fitted parent but replace IID
  latent draws by the already-fixed midpoint normal-score stratification and
  parent-rank reordering. No distribution parameter or sampling parameter may
  change.

## Chemistry Pareto acceptance

Across 63 repeat/fold/style cells, all conditions must hold:

1. all chemistry hard gates pass;
2. overall reference-distance PASS improves by at least 0.08;
3. Wasserstein-core PASS improves by at least 0.15;
4. mean style hit improves by at least 0.08;
5. Q2-edge-strength PASS does not fall;
6. Spearman-structure PASS may fall by no more than 2/63 (`0.0317461`);
7. mean Spearman-structure RMSE may worsen by no more than 1% relative;
8. the candidate simultaneously improves overall-distance PASS,
   Wasserstein-core PASS and mean style hit in at least 5/7 repeats.

The allowed Spearman loss is a bounded Pareto tradeoff, not a redefinition of
the old strict result. If chemistry fails, frozen-Q1 sensory evaluation is not
run and no threshold is tuned.

## Frozen-Q1 sensory acceptance

If chemistry passes, first time one repeat/fold sensory smoke run. The full
seven-repeat run may proceed locally only when projected runtime is below five
minutes; otherwise give the resumable command to the user.

The candidate must then satisfy all of the following on the same new folds:

1. complete finite predictions for all 24 frozen A/B targets;
2. overall A/B reference-line PASS rate does not fall;
3. overall mean normalized A/B Wasserstein does not increase;
4. Tripel mean normalized A/B Wasserstein decreases;
5. Kriek and Lambic each deteriorate by no more than 5%;
6. candidate overall mean A/B Wasserstein is lower in at least 5/7 repeats.

The six Q1 C-grade targets remain audit-only. Passing both stages makes the
candidate eligible as a generator champion with Q1-HOLD limitations; it does
not unlock external public data or migration until the decision is frozen.

## Fixed literature basis

- McKay, Beckman and Conover (1979), Latin hypercube sampling:
  https://doi.org/10.1080/00401706.1979.10489755
- Iman and Conover (1982), inducing rank correlation while preserving
  marginals: https://doi.org/10.1080/03610918208812265
- Liu et al. (2012), high-dimensional nonparanormal SKEPTIC:
  https://doi.org/10.1214/12-AOS1037

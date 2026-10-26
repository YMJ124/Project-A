# Q3 rank-stratified comprehensive confirmation pre-registration v2

## Evidence boundary

The first prospective Pareto confirmation on repeats 11--17 reproduced large
gains in overall reference distance, Wasserstein core and style hit, but it was
formally rejected because Q2-edge-strength PASS fell from 55/63 to 53/63 under
the locked zero-loss Q2 rule. Spearman PASS fell by 1/63, mean Spearman RMSE
worsened by 0.105%, and mean Q2-edge-strength MAE worsened by 0.784%.

The user has now explicitly selected a comprehensive decision rule that permits
small losses. This document does not retroactively pass repeats 11--17. It
prospectively locks symmetric bounded-regression rules and tests them on wholly
unused repeats 18--24 with base seed `20611124`.

## Locked folds and models

- Seven new style-balanced outer repeats, numbered 18--24.
- A freshly recomputed real--real reference and unchanged prior-5 baseline.
- The protected parent remains prior-5, 50% Ledoit--Wolf / 50%
  Spearman-SKEPTIC, with q05 minimum-distance rejection.
- The candidate changes only IID latent sampling to the already-fixed midpoint
  normal-score stratification and parent-rank reordering.

No distribution, sampling, constraint or evaluator parameter may be retuned.

## Comprehensive chemistry acceptance

Across 63 repeat/fold/style cells, every condition must hold:

1. all chemistry hard gates pass;
2. overall reference-distance PASS improves by at least 0.08;
3. Wasserstein-core PASS improves by at least 0.15;
4. mean style hit improves by at least 0.08;
5. Spearman-structure PASS may fall by no more than 2/63;
6. Q2-edge-strength PASS may fall by no more than 2/63;
7. Q2-edge-loss PASS may fall by no more than 2/63;
8. mean Spearman RMSE, Q2-edge-strength MAE and Q2-edge-loss rate may each
   worsen by no more than 2% relative;
9. mean Q2 edge-sign agreement may fall by no more than 2% relative;
10. overall-distance PASS, Wasserstein-core PASS and style hit must all improve
    together in at least 5/7 repeats.

The same 2/63 count tolerance and 2% continuous tolerance are used across the
protected structure metrics. They are fixed before repeats 18--24 are run and
are not derived from their outcomes. A failed condition stops the sensory
stage and may not be relaxed on these folds.

## Frozen-Q1 sensory acceptance

Only after chemistry acceptance, time one repeat/fold smoke run. Run the full
seven-repeat sensory confirmation locally only if projected runtime is below
five minutes; otherwise give the resumable command to the user.

The candidate must have complete finite predictions for all 24 frozen A/B
targets, protect the overall A/B reference-line PASS rate and mean normalized
Wasserstein, improve Tripel, keep Kriek and Lambic deterioration within 5%, and
win mean A/B Wasserstein in at least 5/7 repeats. The six C targets remain
audit-only.

Passing both stages makes the candidate eligible as the comprehensive Q3
generator champion with inherited Q1-HOLD limitations. It does not by itself
unlock public external data or the migration stage.

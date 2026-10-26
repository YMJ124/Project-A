# Q3 training-only sensory-anchor selection pre-registration v1

## Failure addressed

The half-shrunk Spearman-SKEPTIC generator passed fresh chemistry confirmation
but failed its locked sensory protection rule. A/B reference PASS improved from
0.7535 to 0.7743, while mean normalized sensory Wasserstein rose from 0.23779
to 0.24003. Kriek improved, Lambic worsened 3.13%, and Tripel worsened 6.31%,
just beyond the predeclared 5% style limit. The candidate is rejected and is
not a champion.

## Single mechanism

Keep the confirmed chemical generator unchanged and alter only sample choice:

1. generate a fixed pool ten times the required outer-validation style size;
2. fit the immutable Q1 v1 source registry on outer-training rows only;
3. use only the 24 A/B Q1 predictions and scale dimensions by the corresponding
   outer-training target robust scale;
4. draw the required number of distinct target-style training sensory anchors
   with the fixed fold/style seed;
5. use one-to-one minimum-cost assignment to select distinct generated pool
   rows closest to those anchors;
6. retain the same prior 5, SKEPTIC weight 0.5, empirical margins, q05 chemical
   distance layer, constraints and generation seeds.

The pool multiplier is fixed at ten. No multiplier, target subset, distance
weight or anchor rule is screened. C-grade targets cannot influence selection.

## Locked discovery requirements

On repeats 0--2 the selected samples must first satisfy the existing chemistry
rules against prior-5: 27/27 hard gates, protected Wasserstein/overall/style/
Spearman/Q2 metrics, Q2 edge-strength PASS at least 0.764815 and gain at least
0.05. It must then beat or protect the unselected half-shrunk generator under
the already locked sensory rules: A/B PASS rate not lower, mean normalized
sensory Wasserstein not higher, and no style more than 5% worse.

Only if both audits pass may a candidate be locked. Repeats 7--10 are already
viewed and cannot confirm this new post-selection mechanism; any confirmation
must use further fresh pre-registered repeats. External cohorts remain locked.


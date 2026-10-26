# Q3 factor-copula distance rejection pre-registration v1

The fixed 10-factor copula is the only eligible input to this experiment. In
discovery it reached Q2 edge-strength PASS in 25/27 cells, improved mean style
hit by 0.038 and protected both distance summaries, but failed five
near-neighbor hard-gate cells and lost one Spearman PASS cell.

The only change is a rejection layer: within each outer training fold, compute
the leave-one-out nearest-training-distance distribution in the locked full
geometry and reject candidates below its q05. This threshold is training-only
and exactly matches the hard-gate danger-line definition. No factor count,
prior, covariance, margin, seed, fold or evaluator changes.

The candidate must satisfy all preceding locked rules: 27/27 hard-gate cells,
no Wasserstein/overall-distance or Spearman/Q2 pass-rate decline, style hit no
more than 0.05 lower than prior 5, and Q2 edge strength at least 0.764815 with
at least 0.05 gain. Otherwise it is rejected. Discovery repeats 0--2 only;
confirmation, sensory inference and external cohorts remain untouched.

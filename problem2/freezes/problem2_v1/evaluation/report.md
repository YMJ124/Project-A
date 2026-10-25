# Question 2 current evaluation after optimization

- Formal status: **READY**
- Evaluation criteria: {'PASS': 20}
- Quantitative formal lines passed: 16/16
- Quantitative ideal lines passed: 11/16

## Adopted optimization

Stable-edge filtering passed the locked difference-of-medians improvement rule in 5/7 audited repeats (4/6 independent repeats). Repeats [4, 5] failed that rule, while every repeat still passed the absolute community ARI line. Pairwise diagnosis shows repeat 4 improved 2/3 fold-pairs but failed the locked aggregate by 0.0067; repeat 5 improved only 1/3. Both successes and failures are retained.
Stable-community Top-5 attribution raised median permutation-SHAP agreement from 0.250 to 0.667 on independent repeat 3; 8/9 target-folds improved.

## Remaining blockers

- None.

## Interpretation

All formal evidence layers and audits are complete. After the leave-one-repeat-out module rescue, the result is READY with limitations: 7 targets support three candidates, 18 support one or two, and 5 remain explicitly insufficient. Question 1 remains HOLD and caps every interpretation.

# 缺陷感官 OOF 接口 v1

本接口使用问题一相同的 7 重复×3 折风格平衡外层验证，并在每个外层训练折内比较多输出 ET、逐目标 ET 和强正则 GBR。

- `X4vg`：A 级，可作为补充感官审计。
- `dms`、`diacetyl`、`t2n`、`stale_hops`：C 级，不得作为可靠硬约束。
- `stale_hops`：R2=0.0557，Spearman=0.2470，相对全局均值 RMSE 改善=3.24%。

因此问题三中的 DMS/Diacetyl/4VG/trans-2-nonenal 以化学尺度约束为主；`stale_hops` 只能报告探索性预测和不确定性，不得宣称已可靠受控。

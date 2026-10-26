# Q3 迁移、外部数据与端到端工程闭环 v1

## 结论

当前已完成可重放的工程闭环：

`Tripel/Kriek/Lambic 条件生成 -> 231 维化学谱 -> 物理/派生/缺陷/记忆风险约束 -> Q2 模块追踪 -> 冻结 Q1 感官预测 -> 相对重建 0--6 报告 -> 外部三锚点方向审计`。

工程状态为 `ENGINEERING_CLOSURE_COMPLETE_WITH_LIMITATIONS`，不等于正式泛化验收通过。Q1 仍为 `HOLD`，Q2 为 `READY_WITH_LIMITATIONS`，Q3 生成器仍为 `HOLD_NO_GENERATOR_CHAMPION`。

## 生成闭环

- 全数据固定方案：秩分层、50% Ledoit--Wolf + 50% Spearman-SKEPTIC、prior=5、q05 最近训练距离。
- 样本数：Tripel 28、Kriek 10、Lambic 11，合计 49。
- 数值、训练范围、派生字段重算、四类化学缺陷、精确复制、q05 危险近邻和端到端输出均为 0 个失败。
- 保留 30 个标准化感官预测、OOF 残差区间、A/B/C 等级和 `relative_reconstructed_scale`。后者只是相对报告层，不是原始人工 0--6 真分。

## Pils/Lager -> Tripel 迁移

已实现并比较 PCA 质心平移、收缩 CORAL、Sinkhorn 重心映射和 Q2 模块级反事实，同时保留“原样不改”和“最近真实 Tripel”参考线，两者不可入选。

最终 11/11 个源样本均有一个 Pareto 候选通过：

- Tripel 风格概率全部提高，平均提高 0.0949。
- 到真实 Tripel 化学中心的标准化距离全部缩短，平均改善 0.3617。
- 标准化 L1/L2 改动平均为 0.4838/1.2109；平均改动 142.5 个字段、33.1 个模块。这一改动规模偏大，是后续实验配方简化的主要限制。
- 11/11 不是训练样本精确复制，11/11 均位于危险近邻线外，化学缺陷均未恶化。
- 冻结 Q1 的 24 个 A/B 目标向量在 11/11 样本上更接近真实 Tripel 感官中心，平均距离改善 0.3100。这是全数据部署描述性审计，不是外折泛化证据。

## 外部 `test data` 审计

外部数据只使用锁定的 Malty/Hoppy/Fruits 三锚点，先 `log1p` 再做行内组成占比，按真实啤酒 bootstrap 2000 次。它不包含 231 维化学谱，词频也不是专业品评强度或 0--6 真值。

生成风格对比的方向支持较强：

| 锚点 | 风格向量余弦 | 成对方向一致 |
|---|---:|---:|
| Malty | 0.9990 | 3/3 |
| Hoppy | 0.9985 | 3/3 |
| Fruits | 0.7275 | 2/3 |

迁移审计的限制更明显：公开评论数据中 Tripel 相比 Pils/Lager 的 Malty/Hoppy 方向为负，当前内部迁移为正；只有 Fruits 方向一致，广义/窄义 Pils/Lager 的三锚点向量余弦分别为 -0.1200/-0.1785。因此外部数据不支持宣称“迁移感官方向已被外部确认”。

首次外部审计后，工程侧又修正了迁移硬门逻辑（禁止危险近邻，并要求化学距离同时改善）。虽然该修正没有使用外部分数调参，但按预定证据纪律，当前外部结果统一降为发现性证据。任何后续优化都必须寻找新的独立外部证据，不能重复把这一数据集称为未见测试集。

## 产物与重放

本地完整产物位于 `problem3/artifacts/q3_engineering_closure_v1/`，包括生成样本、逐样本追踪、感官表、迁移候选/Pareto/选定表、外部审计、配置和 manifest。原始附件、外部原数据和大型派生产物不进入 Git。

重放命令：

```bash
cd /home/yu/桌面/A
conda run --no-capture-output -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python problem3/scripts/run_engineering_closure.py \
  --config problem3/configs/problem3_engineering_closure_v1.yaml
```

实测约 8 秒，不需要交由用户长时间运行。

## 尚未闭合的正式证据

1. 迁移已在全新 7x3 重复外层折中确认通过：77/77 覆盖，风格与化学改善 77/77，A/B 感官改善 75/77。详见 `q3_migration_confirmation_results_v2.md`。
2. 生成器的 Q2 边丢失率硬计数仍差 1 格（3/63 对允许 2/63），因此不是正式冠军。
3. Q1 仍差风格基线改善 0.54 个百分点和 5 个 A 级目标。
4. `stale_hops` 只有 C 级 OOF 证据，本闭环不把它包装成可靠硬约束。
5. 稀疏迁移将独立确认折的平均改动降至 53.1 字段/9.35 模块，但风格增益只保留 64.1%<75%，已严格拒绝为默认替代；实物验验证仍未完成。

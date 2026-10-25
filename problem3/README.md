# 问题三：风格条件生成与迁移

问题三的上游交接和前置工程已经完成，尚未开始生成器选型。开始构建前先阅读根目录的
`AGENTS.md`、`AGENTS_2.md`、`problem3/AGENTS.md`、`CONTEXT.md` 以及问题一冻结说明。

建议按以下层次维护：

- `configs/`：问题三配置；
- `docs/`：问题三协议、文献和决策记录；
- `scripts/`：可复现脚本；
- `src/`：问题三专属代码；
- `tests/`：问题三测试。

问题三不得绕过问题一的标准化预测和问题二的化学解释结果，直接重新定义输入、标签或评价口径。

## 当前入口

```bash
cd /home/yu/桌面/A
conda activate /home/yu/桌面/A/.conda/envs/beer-q1
export PYTHONPATH=problem1/src:problem2/src:problem3/src

python problem3/scripts/run_problem3_preflight.py \
  --config problem3/configs/problem3_preflight_v1.yaml
```

预检只读取原始附件和问题一/二冻结产物，不训练生成模型。它会在
`problem3/artifacts/preflight_v1/` 生成风格支持度、化学特征范围、感官接口和缺陷约束注册表。

当前固定接口：

- `interfaces/q2_handoff_v1/`：21 个网络分区得到的 710 条共识边、60 个共识模块、46 条候选记录和 231 字段生成角色；
- `interfaces/defect_sensory_oof_v1/`：五个缺陷感官目标的 7 重复×3 折 OOF 证据；
- `artifacts/interfaces/q1_full_predictor_v1/`：可再生成的 Q1 全数据推理器，仅用于部署预测，不作为泛化证据。

Q1 仍为 `HOLD`（13 A、11 B、6 C）；Q2 为 `READY`。缺陷感官中只有 `X4vg` 达到 A 级，`stale_hops` 为 C 级，因此后者只能作探索性监控，不能成为硬约束。

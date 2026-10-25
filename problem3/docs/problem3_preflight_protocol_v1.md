# 问题三前置协议 v1

## 目的

在不训练生成模型的前提下，固定问题三的数据身份、风格支持度、上游模型接口和五类缺陷约束。

## 预检输出

- `preflight_summary.json`：核心数据合同、Q1/Q2 状态和待补接口。
- `input_manifest.json`：原始附件与上游产物的路径和 SHA256。
- `style_support.csv`：22 种风格的样本量与生成/迁移角色。
- `chemical_feature_schema.csv`：231 个字段的缺失、分位数和取值边界。
- `sensory_interface_schema.csv`：50 个感官目标的主/补充范围和冻结预测可用性。
- `defect_constraint_registry.csv`：缺陷字段、阈值、变换证据、执行方式和阻塞原因。
- `validation_splits.csv`：直接继承问题一的 7 重复×3 折风格平衡划分，保持三问证据边界一致。

## 已固定事实

- 目标风格样本量为 Tripel 28、Kriek 10、Lambic 11、Pils/Lager 11。
- DMS 和 X4VG 在附件 1 中与附件 8 校准浓度的 log10 一致。
- Diacetyl 和 trans-2-nonenal 在绝大多数可对齐样本中与校准浓度原值一致；少量差异必须在约束实现时保留审计。
- trans-2-nonenal 本地阈值缺失，因此只允许相对不增加约束。
- `stale_hops` 是补充感官目标，不是化学浓度。
- DMS、Diacetyl 和 4VG 的绝对超阈值约束统一选用附件 8 平均/中位阈值中的较小者；4VG 额外分风格报告，但不放宽迁移中的“超阈值程度不增加”约束。

## 允许的下一步

1. 建立风格分层的训练/验证协议。
2. 实现经验分位数和稳健近邻条件采样基线。
3. 在不改变问题一冻结来源的前提下，建立全数据重训的感官预测适配器。
4. 对 `dms`、`diacetyl`、`X4vg`、`t2n`、`stale_hops` 补做严格 OOF 预测评估，供缺陷感官约束使用。

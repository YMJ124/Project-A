# 问题三：风格条件生成与迁移

问题三已进入前置工程阶段。开始构建前先阅读根目录的
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

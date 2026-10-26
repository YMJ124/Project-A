# MathorCup 2026 赛道 A：化学—感官—生成闭环

项目已打通三问主体链路：231 维化学谱预测 30 个标准化 Aroma/Flavour 目标，以稳定置换重要性和化学共变网络构建解释证据，再进行 Tripel/Kriek/Lambic 条件生成与 Pils/Lager -> Tripel 迁移。

所有正式证据继续遵守折内拟合、重复外层 OOF、硬约束、新奇性检查和预注册采用线。0--6 映射是 `relative_reconstructed_scale` 展示层，不是原始人工评分的可验证逆变换。

## 目录边界

- `problem1/`：问题一的配置、代码、测试、协议和冻结成果；
- `problem2/`：问题二的前置审计、问题一冠军重建接口、网络与解释代码；
- `problem3/`：条件生成、跨风格迁移、化学/缺陷硬约束、Q1/Q2 接口和外部方向审计；
- 根目录：共享数据说明、持久工作约定、项目构建文件和跨问题入口。

## 当前最新结果

- Q1：冻结冠军 `q1_newseed2031_final_hybrid_v2`，中位 R2=0.1721、Spearman=0.4103、RMSE=0.4800，相对全局/风格基线改善 9.33%/4.46%，A=13、A/B=24；状态仍为 `HOLD`。
- Q2：20/20 评估项通过，46 个候选覆盖 25/30 目标；可用但仍需显式报告关联性、不完整 Top-3 和上游 Q1 限制。
- Q3 生成：秩相关分层抽样是允许少量结构损失时的当前综合最优工程方案，但 Q2 边丢失率硬计数还差 1 格，正式状态为 `HOLD_NO_GENERATOR_CHAMPION`。
- Q3 迁移：migration v2 在全新 repeats 32--38 上实现 77/77 覆盖与风格/化学改善，Q1 A/B 感官改善 75/77；它是当前正式默认迁移方案。
- 工程闭环：49 个生成谱和 11 个部署迁移已打通化学、Q2、Q1、相对 0--6 和外部三锚点审计，状态为 `ENGINEERING_CLOSURE_COMPLETE_WITH_LIMITATIONS`。

当前方案的完整限制、剩余缺口和后续优化顺序见 [`REMAINING_OPTIMIZATION_PLAN.md`](REMAINING_OPTIMIZATION_PLAN.md)。Q3 工程闭环与迁移确认分别见 `problem3/docs/q3_engineering_closure_v1.md` 和 `problem3/docs/q3_migration_confirmation_results_v2.md`。

## 环境与快速检查

无需依赖环境名称，直接使用绝对路径最稳妥：

```bash
cd /home/yu/桌面/A
export PYTHONPATH=/home/yu/桌面/A/problem1/src:/home/yu/桌面/A/problem2/src
conda run -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python -m pytest

conda run -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python -m beer_q1.cli audit \
  --config problem1/configs/problem1_screening.yaml
```

也可以做一次本地可编辑安装，之后无需设置 `PYTHONPATH`：

```bash
conda run -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python -m pip install -e . --no-deps
```

## 冒烟实验（约束为短时运行）

只跑 3 个目标、1 次重复、Dummy/风格基线/Ridge，用于检查流水线，不可作为正式结果：

```bash
conda run --no-capture-output -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python -m beer_q1.cli run \
  --config problem1/configs/problem1_screening.yaml \
  --smoke \
  --run-id smoke_q1
```

## 完整实验（预计超过 5 分钟，人工运行）

先运行模型筛选（30 个主目标、3 次重复）。完成后先检查结果，不要立即盲跑 full：

```bash
cd /home/yu/桌面/A
export PYTHONPATH=/home/yu/桌面/A/problem1/src:/home/yu/桌面/A/problem2/src
export MPLCONFIGDIR=/tmp/beer-q1-matplotlib
conda run --no-capture-output -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python -m beer_q1.cli run \
  --config problem1/configs/problem1_screening.yaml \
  --run-id q1_screening_v1
```

筛选完成后，重点查看 `artifacts/problem1/q1_screening_v1/stage_gate.json`、`model_selection.csv`、`model_family_frequency.csv` 和 `residual_diagnostics.png`。根据这些证据删掉明显失败的模型，再进行 10 次重复确认。当前全候选配置的命令如下，但建议等 screening 复核后再运行：

```bash
cd /home/yu/桌面/A
export PYTHONPATH=/home/yu/桌面/A/problem1/src:/home/yu/桌面/A/problem2/src
export MPLCONFIGDIR=/tmp/beer-q1-matplotlib
conda run --no-capture-output -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python -m beer_q1.cli run \
  --config problem1/configs/problem1_full.yaml \
  --run-id q1_full_v1
```

完整配置包含 10 次重复外层 3 折、内层 2 折，以及 Ridge、Elastic Net、PLS1、PLS2、Gradient Boosting、Random Forest、Extra Trees 和 XGBoost。当前 XGBoost 1.7.3 使用 CPU `hist`：本机 RTX 5070 虽可见，但该旧版本的微型 `gpu_hist` 测试未能正常结束，因此不能把它用于正式长跑。若后续需要 GPU XGBoost，应另建现代版本环境做兼容性验证，不要直接破坏论文复现环境。

## 输出

每个实验写入 `artifacts/problem1/<run-id>/`：

- `data_audit.json`：数据合同、哈希、缺失、风格分布；
- `folds.csv`：固定重复外层折；
- `oof_predictions.parquet`：全部标准化 OOF 预测；
- `tuning_trials.csv`：内层调参与一标准误选参记录；
- `model_family_selection.csv`、`model_family_frequency.csv`：每个外层折内、逐目标的模型族选择；
- `metrics_by_repeat.csv`、`metrics_summary.csv`：逐目标/模型指标；
- `metrics_bootstrap_ci.csv`：跨重复划分的 bootstrap 95% 置信区间；
- `model_selection.csv`：逐目标模型选择和 A/B/C 分级；
- `stage_gate.csv`、`stage_gate.json`：是否达到进入 0–6 阶段的门槛；
- `uncertainty.csv`：重复 OOF 预测的离散程度；
- `style_metrics_by_repeat.csv`、`style_metrics_summary.csv`：按风格的 RMSE、MAE 与偏差；
- `residual_diagnostics.png`：逐目标残差图，必须人工复核；
- `run_manifest.json`：环境、耗时和运行参数。
- `progress.jsonl`、`checkpoints/`：长跑进度与每个外层折完成后的中间产物。

通关指标只读取 `nested_selected`：模型族由内层折选择，再用完全未参与选择的外层折评分。`stage_gate.json` 为 `PASS` 仍不等于自动进入下一阶段：A 级残差判定使用了自动代理指标，还需人工检查残差图和风格分组偏差。

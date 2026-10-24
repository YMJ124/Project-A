# MathorCup 2026 赛道 A：问题 1

当前只处理第一阶段：用 231 个化学/理化特征预测 30 个标准化 Aroma/Flavour 连续目标。0–6 等级映射尚未开始。

实现遵守 `AGENTS.md`：只按整数 `beer` 对齐，风格只用于折叠、基线和误差分析，所有填补、缩放、PLS 与超参数选择都在训练折内部完成。

## 当前最新结果

当前严格冠军为 `q1_newseed2031_final_hybrid_v2`：以折内调参的多输出 Extra Trees 为主体，并仅纳入已经独立新种子确认的 13 个逐目标替换或后处理源。在 `20311124` 起始的 7 个重复外层折上，中位 R2=0.1721、Spearman=0.4103、RMSE=0.4800，相对全局/风格基线分别改善 9.33%/4.46%，A=13、A/B=24。阶段状态仍为 `HOLD`。

该结果相比同折 ET 对照将 A 从 4 增到 8、A/B 从 22 增到 23，但仍未满足风格基线改善 5%、A>=18 和 A/B>=24 的预声明门槛，因此仍为 `HOLD`，暂不进入 0–6 映射。完整产物见 `artifacts/problem1/q1_newseed_expanded_specialist_hybrid_v1/`。

## 环境与快速检查

无需依赖环境名称，直接使用绝对路径最稳妥：

```bash
cd /home/yu/桌面/A
export PYTHONPATH=/home/yu/桌面/A/src
conda run -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python -m pytest

conda run -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python -m beer_q1.cli audit \
  --config configs/problem1_screening.yaml
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
  --config configs/problem1_screening.yaml \
  --smoke \
  --run-id smoke_q1
```

## 完整实验（预计超过 5 分钟，人工运行）

先运行模型筛选（30 个主目标、3 次重复）。完成后先检查结果，不要立即盲跑 full：

```bash
cd /home/yu/桌面/A
export PYTHONPATH=/home/yu/桌面/A/src
export MPLCONFIGDIR=/tmp/beer-q1-matplotlib
conda run --no-capture-output -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python -m beer_q1.cli run \
  --config configs/problem1_screening.yaml \
  --run-id q1_screening_v1
```

筛选完成后，重点查看 `artifacts/problem1/q1_screening_v1/stage_gate.json`、`model_selection.csv`、`model_family_frequency.csv` 和 `residual_diagnostics.png`。根据这些证据删掉明显失败的模型，再进行 10 次重复确认。当前全候选配置的命令如下，但建议等 screening 复核后再运行：

```bash
cd /home/yu/桌面/A
export PYTHONPATH=/home/yu/桌面/A/src
export MPLCONFIGDIR=/tmp/beer-q1-matplotlib
conda run --no-capture-output -p /home/yu/桌面/A/.conda/envs/beer-q1 \
  python -m beer_q1.cli run \
  --config configs/problem1_full.yaml \
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

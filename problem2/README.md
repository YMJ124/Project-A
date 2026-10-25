# 问题二

问题二负责在问题一的标准化感官预测基础上进行化学网络、机制解释、稳定性分析和关键化合物筛选。目录内的代码、配置、测试和文档只服务于问题二；根目录 `AGENTS_2.md` 是本目录的工作约定。

问题二通过 `problem1/freezes/problem1_v1/` 读取问题一冻结接口，禁止绕过冻结成果直接引用问题一临时实验结果。

## 当前入口

```bash
cd /home/yu/桌面/A
conda activate /home/yu/桌面/A/.conda/envs/beer-q1
export MPLCONFIGDIR=/tmp/beer_q2_mpl
export PYTHONPATH=problem1/src:problem2/src
python -m pytest -q problem2/tests
python problem2/scripts/run_problem2_pilot.py \
  --config problem2/configs/problem2_pilot_v1.yaml
python problem2/scripts/evaluate_problem2_pilot.py \
  --run-dir problem2/artifacts/q2_pilot_v1_1
python problem2/scripts/run_problem2_advanced_pilot.py
python problem2/scripts/evaluate_problem2_advanced.py
python problem2/scripts/run_problem2_network_confirmation.py
python problem2/scripts/run_problem2_source_audit.py
python problem2/scripts/check_q1_reconstruction.py --repeats 4-4 --all-targets \
  --output problem2/artifacts/q1_reconstruction_all30_repeat4_v1
python problem2/scripts/check_q1_source_shap.py
python problem2/scripts/assemble_problem2_current_status.py

# 正式30目标×7重复×3折，主置换100次；按折断点续跑
python problem2/scripts/run_problem2_formal_fast.py \
  --config problem2/configs/problem2_formal_all7_fast_v1.yaml

# 正式30目标×7重复 TreeSHAP/社区联合置换/风格检查
python problem2/scripts/run_problem2_formal_advanced_fast.py \
  --config problem2/configs/problem2_formal_all7_advanced_fast_v1.yaml

# 正式候选合成、模块优先补充与最终评估
python problem2/scripts/finalize_problem2_formal.py \
  --config problem2/configs/problem2_formal_finalize_v1.yaml
python problem2/scripts/run_problem2_module_first_rescue.py \
  --config problem2/configs/problem2_module_first_rescue_v1.yaml
python problem2/scripts/assemble_problem2_candidate_bundle.py \
  --config problem2/configs/problem2_candidate_bundle_v2.yaml
python problem2/scripts/assemble_problem2_current_status.py \
  --output problem2/artifacts/q2_current_evaluation_v8
```

问题二正式证据链已经完成。当前评估位于
`problem2/artifacts/q2_current_evaluation_v8/`：20/20 项通过，16/16 条正式
数值线通过，状态为 `READY`。正式候选目录位于
`problem2/artifacts/q2_formal_candidate_bundle_v2/`，共报告 46 个候选；
7 个目标有三个候选、18 个有一至两个、5 个明确为证据不足。该 READY
状态带有限制，且问题一冻结状态仍为 HOLD。

已采用的解释优化是“稳定社区 Top-5 候选池”：完整 Top-10
方法一致性仍按 G4 报告，但在形成 Top-3 前先以社区为单位取
Top-5。该机制在未见 repeat 3 上将置换—SHAP Jaccard 中位数从
0.250 提高到 0.667，决策为 `ADOPT`。

正式特征置换 `q2_formal_all7_fast_v1` 已完成：30目标、7重复、
3折、两种模式各100次，29,106,000行完整；630/630个目标—折
重建误差为0。稳定性三重门筛出67个单特征—目标候选，覆盖
24/30个目标；条件/无条件Top-10 Jaccard中位数为0.603。正式
正式高级解释、社区去冗余和附件审计随后均已完成。

稳定边社区在 repeats 0--6 的绝对 ARI 均达到 0.50；筛边相对未筛边的
改善规则通过 5/7 次（独立 repeats 1--6 为 4/6），repeats 4/5 分别
变化 `-0.0067/-0.0084`。因此保留稳定边社区
作为主去冗余方案，同时强制保留未筛选网络敏感性结果。

正式置换运行器已改为共享模型一次预测、批量指标、逐折流式
Parquet 和断点续跑。与旧实现逐行对照的最大数值差为
`2.22e-16`；30目标一折10/2次主/敏感性置换实测73秒，
峰值内存约241MB；100/20完整深度实测170.67秒，峰值约299MB。
正式配置已恢复为100/100，21折外推约79--86分钟，建议预留
1.5--2小时。GPU未用于
正式冠军：冠军是 sklearn ET/GBR，
本机GPU环境没有可保证模型与预测等价的 cuML/Treelite 路径。

高级解释运行器也已改为逐折原子检查点：共享多输出TreeSHAP、
共享社区置换预测，并将原始SHAP按折流式写入。单折正式参数
实测14.46秒、峰值内存约540MB，全21折预计7--12分钟。
新旧SHAP逐行差为0；社区置换除Spearman的`2.22e-16`舍入外完全一致。

高级解释正式运行生成 12,127,500 行 SHAP、570,600 行社区置换和
630 行审计，全部精确匹配预期；SHAP 最大加和误差 `6.66e-15`，重建误差
为 0。67 个原稳定候选经高级证据门后保留 59 个，再归并为 45 个模块。
模块优先留一重复优化为三个原证据不足目标补充四个次级模块候选，仍有
五个目标保持证据不足，未放宽门槛或强制补足。

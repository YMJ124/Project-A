# 问题二正式长跑手册 v1

## 推荐：七重复一次运行（可断点续跑）

```bash
cd /home/yu/桌面/A
conda activate /home/yu/桌面/A/.conda/envs/beer-q1
export MPLCONFIGDIR=/tmp/beer_q2_mpl
export PYTHONPATH=problem1/src:problem2/src

python problem2/scripts/run_problem2_formal_fast.py \
  --config problem2/configs/problem2_formal_all7_fast_v1.yaml

# 主置换全量完成后，运行30目标×7重复高级解释链
python problem2/scripts/run_problem2_formal_advanced_fast.py \
  --config problem2/configs/problem2_formal_all7_advanced_fast_v1.yaml
```

该命令完成 30 目标、7×3 冻结折、风格内主置换100次与
无条件敏感性置换100次。每折完成后立即原子落盘；重新执行同一
命令会跳过已完成折。

## 单重复备用分块

第一块固定为冻结 repeat 0、30 个主目标和 3 个外层折。
主置换为验证折内按风格置换，每特征100次；无条件置换作敏感性。
高级阶段运行真实冠军 TreeSHAP 和稳定社区 20 次联合置换。
每特征100次的正式深度要求由上述主置换运行满足；组联合置换
只用于去冗余敏感性检查。

该分块不产生正式 Top-3；必须等全部 7 个重复合并后再计算候选稳定性。

## 运行命令

```bash
cd /home/yu/桌面/A
conda activate /home/yu/桌面/A/.conda/envs/beer-q1
export MPLCONFIGDIR=/tmp/beer_q2_mpl
export PYTHONPATH=problem1/src:problem2/src

python problem2/scripts/run_problem2_formal_fast.py \
  --config problem2/configs/problem2_formal_repeat0_fast_v1.yaml
```

单重复备用分块只用于资源受限时先产生一块主置换证据。上述全量高级
解释配置故意读取完整七重复基础运行，不应在单重复备用分块后
直接运行。

## 完成检查

- 七重复正式运行的 `data_audit.json` 应记录30个目标、7个重复、
  3个折且所有重建通过，最大误差 `<=1e-9`。
- 正式运行的 `oof_permutation_importance_parts/` 应有21个可独立
  恢复的折文件，总行数应为
  `30×7×3×231×(100+100)=29,106,000`。
- 单重复备用分块的 `data_audit.json` 应记录30个目标、3个折，
  所有重建通过，最大误差 `<=1e-9`。
- 单重复的 `oof_permutation_importance_parts/` 应有 3 个折文件。
  风格内主置换100次，无条件敏感性置换100次，总行数应为
  `30×3×231×(100+100)=4,158,000`。
- 全量高级解释的 `oof_shap_values_parts/`、
  `oof_group_permutation_parts/`、`shap_fold_level_parts/` 和
  `shap_reconstruction_audit_parts/` 应各有21个折文件。
- 全量SHAP行数应为 `30×7×250×231=12,127,500`；当前稳定
  社区结构下，20次社区联合置换应为570,600行。
- `shap_reconstruction_audit.csv` 应有630行，SHAP加和误差
  `<=1e-8`，冻结预测重建误差`<=1e-9`。
- 任一命令失败时先保留输出目录和终端错误，不直接删除后重跑。

## 资源预期

优化后的实测基准为：30目标一折、1/1次两模式约57秒；10/2次
约73秒，峰值内存约241MB。完整100/20深度实测为170.67秒，
其中折计算153.09秒，峰值内存约299MB，单折原始结果约41.5MiB。
恢复为100/100后，按21折实测斜率外推约79--86分钟；考虑
机器负载和磁盘波动，为正式运行预留1.5--2小时。预计原始
分块合计约1.5GiB，
当前剩余磁盘空间充足。实际时间由 `fold_runtime.json` 记录。

高级解释的30目标单折正式参数实测为14.46秒，其中包含
575,190行SHAP和25,800行社区置换；峰值内存约540MB。
21折纯折计算外推约5.1分钟，加上最终扫描和汇总，预计
7--12分钟，建议预留15分钟。任一折完成后可断点续跑。

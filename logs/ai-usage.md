# AI 工具使用台账

## 2026-10-24：问题 1 建模底座

- 工具：OpenAI Codex（GPT-5），开发机构 OpenAI。
- 用途：协助梳理问题 1 的数据合同、无泄漏交叉验证、模型候选、评估门槛，并生成首版 Python 工程与测试。
- 关键输入：要求先建“化学成分 -> 标准化感官强度”模型；参考 `AGENTS.md`、`CONTEXT.md`、赛题附件及其中列出的论文；超过 5 分钟的训练交由参赛者运行。
- 采纳内容：按 `beer` 对齐、30 个 `A_`/`F_` 主目标、风格平衡重复 3 折、内层 2 折、一标准误选参、Dummy/风格均值/Ridge/PLS/树模型候选、OOF 产物与阶段门槛。
- 明确拒绝：使用 `beer_id` 连接、全数据预处理后再交叉验证、直接载入外部同源标签或旧 pickle、把标准化目标冒充 0–6 原始评分。
- 人工验证要求：参赛者需审阅配置、运行完整实验、核对残差图与领域解释，并在论文中据实际采用内容披露 AI 贡献。
- 涉及文件：`configs/`、`src/beer_q1/`、`tests/`、`README.md`、`AGENTS.md`。
- CUDA 核验：沙箱外 `nvidia-smi` 检出 RTX 5070 Laptop（8 GB）；XGBoost 1.7.3 的 2 棵树 `gpu_hist` 微型测试在初始化后长期无响应并被人工中止，因此正式配置退回 CPU `hist`，未声称 GPU 兼容。
- 验证记录：7 个自动化测试通过；Ridge-only 与多模型短时端到端实验通过。短测发现 Ridge 在 `A_malt_all` 有极端外推，因此保留 PLS/树模型对照与残差人工检查，不把短测分数写成正式结论。
- 选择协议修订：逐目标模型族改为只用内层折选择，`nested_selected` 轨道只在外层折评估；阶段门槛不再使用“看完外层成绩再挑冠军”的乐观分数。

## 2026-10-24：Extra Trees 确认与 GBR 目标级挑战

- 用途：分析 `q1_screening_v1` 中未达标指标，保护已达标指标，并根据 Nature 论文和作者 Zenodo 代码说明选择下一项单机制实验。
- 资料边界：仅使用论文和官方代码说明提出“树模型优先、逐目标 GBR”假设；未载入 Zenodo 外部标签或 pickle。Kaggle 相关页面任务和数据不同，且本轮无法可靠提取正文，未用其决定模型。
- 运行记录：`q1_et_confirmation_v1` 完成 10 次重复，耗时 59.9 秒；`q1_gbr_fixed_challenger_v1` 完成 3 次配对筛选，耗时 143.6 秒；`q1_gbr_confirmation_v1` 完成 10 次固定 GBR，耗时 211.5 秒。一次完整 GBR 网格在估算超过 5 分钟后于首折主动中止，未用其不完整结果。
- 保护结果：10 次 Extra Trees 确认了中位 R2=0.1734、Spearman=0.4015、RMSE=0.4900、全局基线改善 9.46%、正 R2 和正方向占比均为 100%。
- 未达标原因：相对风格基线改善只有 2.98%；只有 10/30 目标 R2>=0.20，8 个目标仍为 C 级；烧焦麦芽等高强度样本出现明显向零收缩。
- 采用结论：全局 GBR 退化，不替代 Extra Trees；只采用在 7 个未见重复中同时稳定改善 RMSE 和 Spearman 的 `A_esters_isoaa` GBR。`A_esters_all` 仅作探索候选。
- 涉及文件：`AGENTS.md`、`src/beer_q1/runner.py`、`tests/test_models_and_metrics.py`、`configs/problem1_et_confirmation.yaml`、`configs/problem1_gbr_fixed_challenger.yaml`、`configs/problem1_gbr_confirmation.yaml`、`scripts/analyze_et_gbr_hybrid.py` 及 `artifacts/problem1/q1_et_gbr_hybrid_analysis_v1/`。
- 后续单机制测试：比较同参数的多输出与逐目标 Extra Trees。逐目标架构全局指标更差，不整体采用；`F_esters_all` 在 7 个未见重复上通过预声明条件，因此只替换该目标。最新严格冠军产物位于 `artifacts/problem1/q1_confirmed_specialist_hybrid_v1/`。

## 2026-10-24：稳健线性修复与事后候选独立复验

- 用途：继续执行“先诊断、保护已达标指标、每轮只验证一个机制”的规范。
- 稳健线性实验：新增仅在训练折内选取的强右偏非负特征 `log1p`、稳健缩放与 Ridge/PLS；3 次重复耗时 11.7 秒。极端外推已消除，但中位 R2=0.0265、Spearman=0.2590、全局基线改善仅 1.60%，因此整体拒绝。
- 事后候选防乐观处理：对 `A_hops_woody` 和 `F_esters_isoaa` 使用 7 个全新随机种子与折内调参的 ET 冠军配对复验。两者平均 RMSE 改善分别仅 0.84% 和 0.42%，未达预声明 1% 门槛，均拒绝。
- 保护结论：严格冠军仍为 28 目标多输出 ET + `A_esters_isoaa` GBR + `F_esters_all` 逐目标 ET，不因事后小幅度改善而增加复杂度。
- 下一机制：论文的多输出表格显示 RF、ET、XGBoost 的 R2 均约 0.22，且 RF 分目标平均排名略优；先用同折 3 次重复筛选 RF，不同时启动 XGBoost。
- RF 筛选结果：3 次重复耗时 44.6 秒。RF 中位 R2=0.1721、Spearman=0.4050、RMSE=0.4861，同折 ET 分别为 0.1808、0.4101、0.4827；24/30 目标 RMSE 退化，无单目标通过 1% 改善及 Spearman 保护条件，整体与目标级替换均拒绝。
- 后续机制：准备 CPU `hist` 的浅层强正则 XGBoost 同折筛选；不使用未验证的 RTX 5070 + XGBoost 1.7.3 `gpu_hist`。
- XGBoost 筛选结果：3 次重复耗时 168.4 秒，9/9 外层折均由一标准误规则选择深度 1 的最强正则候选。其中位 R2=0.1226、Spearman=0.3511，均差于 ET；少数改善目标也未胜过已采用专用模型，拒绝进一步确认。
- 诊断结论：严格冠军 30/30 目标的重复胜率与方向性稳定，但仅 10/30 达到 R2>=0.20、14/30 达到 Spearman>=0.45、13/30 达到全局基线改善>=10%。下一轮针对 p/n 偏高与目标化学机制稀疏性，只实验折内逐目标特征筛选，不读取附件 8 的预计算重要性。

## 2026-10-24：折内目标特征筛选与扩展专家冠军

- 机制：在每个训练折中，对每个感官目标单独用 `f_regression` 排名特征，只保留 k=30/60/120 后拟合逐目标 Extra Trees。没有读取附件 8 的全数据重要性，也没有引入外部标签。
- 发现与确认：先在 3 个重复上筛出 8 个目标，再用 7 个全新种子与折内调参 ET 配对。按预声明的 1% 平均 RMSE、5/7 胜出、Spearman 不降和等级不降条件，采用 6 个，拒绝 `A_hops_woody` 和 `A_esters_all`。
- 完整新种子冠军：另在同一 7 重复折上复现 `A_esters_isoaa` GBR 与 `F_esters_all` 逐目标 ET，组成 22+2+6 混合。相比同折 ET 对照，中位 RMSE 0.4915 -> 0.4873，风格基线改善 3.32% -> 4.44%，A 4 -> 8，A/B 22 -> 23；其他已通过的总体指标保持。
- 决定：阶段门槛仍为 `HOLD`，不事后降低门槛，不进入 0–6 映射。产物位于 `artifacts/problem1/q1_newseed_expanded_specialist_hybrid_v1/`。残差人工复核仍发现范围压缩和极端样本误差，下一候选是折内交叉拟合的保守幅度校准。

## 2026-10-24：文献约束的化学机制组实验

- 工具：OpenAI Codex（GPT-5），开发机构 OpenAI。
- 用途：在通用模型趋于饱和后，检索啤酒酒花、酯类与麦芽香气原始研究，依据论文在看本地结果前定义化学特征组，并进行无泄漏 OOF 验证。
- 资料边界：使用 Crossref/OpenAlex 定位和核对论文元数据及公开摘要；没有下载或并入外部同源感官标签、预训练模型或附件 8 重要性。
- 文献依据：Takoi et al. (2010) 的 linalool/geraniol/beta-citronellol 柑橘协同；Yin et al. (2019) 的啤酒中链脂肪酸乙酯发酵实验；Ammari & Schroen (2019) 的乙醇基质挥发分配；Féchir et al. (2021) 的特种麦芽香气原始研究。
- 实现内容：新增固定特征子集逐目标 Extra Trees，以及训练折内标准化的化学组汇总变换；24 个自动化测试通过。
- 实验结论：85 个酒花机制输入的纯限制使 10/10 hops 目标退化；73 个酯/前体输入无 B 目标达到 1% 入围线；保留全特征并追加三个酒花汇总量更安全，但最强的 `A_hops_citrus` 仅改善 0.67%。三项机制均拒绝，不进入七重复确认，冠军不变。
- 涉及文件：`src/beer_q1/models.py`、`tests/test_models_and_metrics.py`、`configs/problem1_*domain*`、`configs/problem1_hops_group_summaries*`、`artifacts/problem1/domain_feature_groups_v1/`、`AGENTS.md`。

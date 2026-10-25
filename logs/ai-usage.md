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

## 2026-10-24：问题二首版网络与样本外解释 pilot

- 工具：OpenAI Codex（GPT-5），开发机构 OpenAI。
- 用途：依据正式赛题 PDF、`AGENTS_2.md`、`CONTEXT.md` 和问题一冻结接口，实现问题二首版折内化学丰度相关网络、真实冠军流水线验证折置换解释和 G0-G6 评估体系。
- 数据/泄漏边界：只使用冻结外层训练折重算网络和重建模型；验证折只用于样本外预测与置换评估；没有用附件 6/8 选择候选，没有生成正式 Top-3。
- 运行环境：`beer-q1`，Python 3.9.18、pandas 1.5.3、NumPy 1.24.3、scikit-learn 1.2.2、NetworkX 3.2.1、igraph 0.11.9、leidenalg 0.10.2。
- 主要结果：`q2_pilot_v1_1` 覆盖 A/B/C 三个代表目标；9/9 目标—折与冻结 OOF 完全一致。主阈值边数 1858/1620/1806，折间边集 Jaccard 中位数 0.621，社区 ARI 0.457，条件/无条件置换 Top-10 Jaccard 中位数 0.667。
- 错误与纠正：首次 `q2_pilot_v1` 对含缺失特征用了整列秩化，与附件 2 的成对完整 Spearman 最大差 0.0100。已保留并标记无效，修正运行与附件 2 最大差 `6.66e-16`。
- 决定：工程 pilot `PASS`，问题二正式结论 `HOLD`。缺失 30 目标、7x3 折、200 次网络 bootstrap、100 次置换、TreeSHAP、社区联合置换、风格异质性和附件 6/8 一致性审计，不允许据此冻结 Top-3。
- 涉及文件：`problem2/src/beer_q2/`、`problem2/scripts/`、`problem2/configs/problem2_pilot_v1.yaml`、`problem2/tests/`、`problem2/docs/evaluation_framework_v1.md`、`problem2/artifacts/q2_pilot_v1_1/`、`AGENTS_2.md`。

## 2026-10-25：问题二扩展解释、差距评估与网络优化

- 工具：OpenAI Codex（GPT-5），开发机构 OpenAI。
- 用途：在冻结问题一外层折上补齐 TreeSHAP、社区联合置换、风格方向异质性，建立距正式/理想评估线的定量表，并只验证一个预先锁定的网络稳定性机制。
- 数据边界：附件 6/8 只在 pilot 候选选定后做同源审计；附件 8 的重要性 sheet 因缺目标标识而排除；没有使用附件调整候选排名。
- 解释结果：SHAP 最大加和误差 `7.77e-16`；置换—SHAP Top-10 Jaccard 中位数 0.333；风格方向反转率 6.09%；A/B/C 代表目标完整 pilot 候选数 3/3/0。
- 优化结果：稳定边社区过滤在未见 repeat 1 上将 ARI 由 0.5504 提高到 0.6473，改善 0.0970；稳定边 Jaccard 0.6152，边保留率 50.14%，四项预声明条件全部通过，决策 `ADOPT`。
- 同源审计：6/6 候选的附件 6 方向一致，但只称为同源一致性；去冗余后 6 个单体候选收缩为 4 个稳定模块。
- 决定：当前 13 PASS + 7 PARTIAL，正式 `HOLD`。不调低门槛，不输出正式 Top-3。
- 涉及文件：`problem2/src/beer_q2/advanced.py`、`network_optimize.py`、`source_audit.py`、`current_status.py`，对应脚本/配置/测试、`problem2/docs/problem2_v1_experiment_report.md` 和 `AGENTS_2.md`。

## 2026-10-25：问题二模块级归因优化

- 工具：OpenAI Codex（GPT-5），开发机构 OpenAI。
- 用途：针对置换—SHAP 单体 Top-10 一致性仅 0.333 且高相关特征难以唯一归因的问题，评估稳定社区模块级归因。
- 失败记录：Top-10 模块方案在未见 repeat 2 上由 0.333 提高到 0.429，但未达预声明的 0.50 绝对线和 0.10 改善线，决策 `REJECT`，未降低门槛。
- 新版本：保留完整 Top-10 报告，但用模块 Top-5 作为 Top-3 的候选池。规则在查看 repeat 3 前锁定。
- 确认结果：repeat 3 上单体/模块 Top-5 Jaccard 分别为 0.250/0.667，中位提升 0.238，8/9 目标—折改善，A/B 目标保护线通过；四项条件全部通过，决策 `ADOPT`。
- 边界：只优化候选的报告与去冗余单位，没有改变问题一模型、预测或 A/B/C 等级；C 级仍只能探索性解释。
- 涉及文件：`problem2/src/beer_q2/module_optimize.py`、`advanced.py`、对应 repeat 2/3 配置与运行产物、`problem2/artifacts/q2_module_top5_confirmation_v1/` 及 `q2_current_evaluation_v3/`。

## 2026-10-25：问题二全冠军来源审计与正式长跑准备

- 工具：OpenAI Codex（GPT-5），开发机构 OpenAI。
- 用途：扩展问题二重建与 TreeSHAP 实现，覆盖问题一冻结的全部
  30 目标和九种模型来源；补齐七个冻结重复的 200-bootstrap 网络审计。
- 实现审计：repeat 4 的 90/90 个目标—折预测与冻结 OOF 完全一致；
  九种来源架构的 SHAP 加和误差最大 `1.67e-15`；30 目标一折
  置换/SHAP/组置换工程 smoke 通过。
- 网络结果：200-bootstrap 已覆盖 repeats 0--6，稳定边社区的绝对
  ARI 7/7 达到 0.50；相对未筛边的改善规则通过 5/7，独立重复
  为 4/6，repeats 4/5 的负结果均保留。
- 评估：`q2_current_evaluation_v5` 为 14 PASS + 6 PARTIAL，数值正式线
  13/16、理想线 7/16，状态仍为 `HOLD`。
- 长跑边界：代理只完成配置、短时 smoke、测试和读取器；30 目标、
  100 次置换的正式分块交由参赛者在 `beer-q1` 环境执行。
- 长跑加速：实现共享模型预测复用、批量指标、流式Parquet与逐折断点
  续跑。新旧置换结果最大数值差 `2.22e-16`，SHAP/组置换差为0；
  30目标一折10/2次置换73秒，峰值内存241MB。GPU由于无冻结
  sklearn模型的等价加速路径而不采用，避免改变解释对象。
- 完整深度实测：30目标、1折、100次风格内+20次无条件置换
  共用170.67秒，峰值内存306064KB，831,600行全部通过，冻结预测
  重建误差为0；21折正式运行外推约54--60分钟。
- 为与未加速方案保持完全相同的统计深度，正式与单重复备用
  配置均恢复为100次风格内+100次无条件置换。按实测斜率外推
  约79--86分钟，正式窗口预留1.5--2小时。
- 正式结果：29,106,000行完整，630/630目标—折重建误差为0，
  21折计算合计78.11分钟。67个稳定单特征—目标候选覆盖24/30
  目标；条件/无条件Top-10 Jaccard中位数0.603。计算硬缺口关闭，
  正式Top-3仍等高级解释、社区去冗余和附件审计。
- 高级解释加速：新增逐折原子检查点和流式SHAP，对共享多输出
  树复用TreeSHAP，社区置换同批预测30目标。新旧SHAP差为0，
  社区置换最大数值差`2.22e-16`。单折正式参数核心计算14.46秒、
  峰值内存约540MB，全21折预计7--12分钟。

# 2026 MathorCup 大数据挑战赛赛道 A：Codex 项目上下文

> 核验日期：2026-10-24（Asia/Shanghai）  
> 用途：供后续 Codex 任务在开始编码、建模或撰写论文前阅读。  
> 状态：赛题 PDF、附件 1–8 和本机竞赛规范已做首轮本地核验；Nature 论文、Zenodo 记录、Kaggle 页面及核心 GitHub 仓库已做在线核验。仍未完成的核验项集中列在文末。

## 0. 给 Codex 的执行摘要

本题不是三个互不相关的小题，而是一条闭环：

1. 用化学谱预测感官画像：`X_chemical -> Y_sensory`。
2. 解释每个感官维度的关键化合物：模型解释 + 相关网络 + 稳定性。
3. 逆向生成或迁移化学谱：`style/target sensory -> X_new -> predicted sensory`，同时满足化学与缺陷约束。

开始实现前必须牢记以下事实：

- 题面写“226 种不同啤酒”，但本地 `1.csv` 实际为 **250 个啤酒样本、231 个化学/理化特征、3 个标识列**。原 Nature 论文也是 250 款啤酒；题面大概率混淆了样本数与化学属性数量。
- 题面把 `2.csv` 称为 Pearson 相关矩阵，但本地复算证明 `2.csv`、`5.csv`、`6.csv` 的数值均为 **Spearman 秩相关系数**，与 `数据说明.txt` 和原论文方法一致。
- `3.xlsx` 不是已填写的真实标签表。它包含 **12 个空白 0–6 评分卡 sheet**（`beer_1`–`beer_12`）和一个由公式汇总的 `summary` sheet；当前缓存汇总值全为 0，因为评分卡未填写。题面所说“10 个 sheet”与文件实际内容不一致。
- `4.csv` 已包含全部 250 款啤酒的 50 个标准化/中心化专业感官指标；它们不是原始 0–6 分。因此问题 1 的关键不是“标签只剩 10 个”，而是如何在不违规泄漏的前提下解释并恢复/构造 0–6 分级，同时证明映射合理。
- 附件 4 和 7 的 `beer_id` 以低精度科学计数法存储，读取后仅剩 96 个唯一值；**禁止用它们的 `beer_id` 连接表**。当前可靠主键是整数 `beer`（1–250），并须检查风格标签一致性。
- 附件 8 含预计算的 GBM MDI/SHAP 结果。这些不是独立验证集，且目标变量与训练流程仍须追溯；不能直接拿来证明自己模型的稳定性或泛化能力。
- 赛题数据与 Nature/Zenodo 公开数据高度重合。外部代码可借鉴；外部原始标签若与比赛对象一一对应，可能构成答案泄漏。未核实规则前，不得并入训练、调参或最终评分映射。

## 1. 本地资料与信任优先级

### 1.1 本地路径

- 项目目录：`/home/yu/桌面/A`
- 赛题目录：`/home/yu/桌面/A/2026年MathorCup大数据挑战赛-赛道A初赛`
- 赛题 PDF：`/home/yu/桌面/A/2026年MathorCup大数据挑战赛-赛道A初赛/2026年MathorCup大数据挑战赛-赛道A初赛.pdf`
- 附件目录：`/home/yu/桌面/A/2026年MathorCup大数据挑战赛-赛道A初赛/附件`
- AI 工具规定：`/home/yu/桌面/应数/MathorCup数学应用挑战赛人工智能工具使用规定（试行）.pdf`
- 论文与提交规范：`/home/yu/桌面/应数/2026年MathorCup数学应用挑战赛—大数据挑战赛论文格式及提交规范.pdf`

### 1.2 事实来源优先级

发生冲突时按以下顺序处理，并把冲突记录到实验日志：

1. 实际附件内容、字段、行列数和可复算数值。
2. 本地正式赛题 PDF 与组委会规范。
3. Nature 原论文及其方法部分。
4. Zenodo 作者发布的源码和补充数据说明。
5. Kaggle/GitHub 项目说明。
6. 当前文档中的推荐方案。

`CONTEXT.md` 是交接材料，不替代实时数据检查。任何实现都必须重新运行 schema、主键、缺失值与变换检查。

## 2. 三问的准确任务定义

### 问题 1：数据缺失条件下的感官评分

题面要求以 `1.csv` 为基础，按 `3.xlsx` 定义的 Aroma 与 Flavour 维度，给出约 40 个感官维度的 0–6 分级结果。

实际数据提示：

- `3.xlsx` 定义的是评分卡结构，不含有效人工评分。
- `4.csv` 有 50 个专业感官维度的标准化连续值，含 15 个 Aroma、15 个 Flavour，以及酸度、苦度、甜度、缺陷感知、酒精、余味、酒体、二氧化碳和总体喜好等 20 个补充指标。
- 第一阶段应先把“赛题真正要求输出哪些列”固定成 schema。若只输出 Aroma + Flavour，应是 30 个主维度；题面“约 40 个”与附件 3/4 的完整 50 维结构不完全一致，必须在论文中说明选取口径。

建议把问题拆成两个显式子任务：

1. **预测任务**：用 `1.csv` 的化学特征预测 `4.csv` 的标准化连续感官值，采用严格的交叉验证。
2. **标度任务**：把标准化预测映射到 0–6 有序等级。优先寻找附件或公开方法中可证明的原始均值/标准差及离散规则；若无法恢复，只能提出透明、可复算的校准规则，并把它标为建模假设，而不是“恢复真实原始分”。

不要把连续回归指标与有序分级指标混为一谈。连续阶段报告 MAE、RMSE、R²、Spearman；分级阶段再报告 MAE、相邻等级命中率、Macro-F1、Quadratic Weighted Kappa，并给出每维不确定性。

### 问题 2：化合物—风味分析

目标是利用 `2.csv` 的化合物相关结构构建“化合物共表达图”（更准确地说是**化合物丰度相关网络**），再结合 `1.csv` 和感官模型，为每个目标感官维度给出 Top-3 关键化合物，并论证稳定性与化学合理性。

建议证据链：

1. 训练折内的目标预测能力合格。
2. 使用 OOF/训练折模型计算 SHAP 或置换重要性，避免在全数据拟合后报告乐观解释。
3. 进行按风格分层的 bootstrap，记录每个化合物进入 Top-3 的频率。
4. 比较 GBR/XGBoost、PLS/Elastic Net、随机森林等方法的 Top-3 Jaccard 一致性。
5. 在 Spearman 网络中检查候选化合物的社区、中心性与共线替代关系。
6. 用附件 8 的来源、单位、阈值和公开化学文献做合理性论证。

重要限制：SHAP、相关系数和网络中心性都只支持关联解释，不等于因果。高度相关特征会分摊或替代重要性，原论文也明确指出 GBR 可能在相关变量中偏向主效应最大的变量。

### 问题 3：风格与“风味指纹”生成

以 `tasting_category_fine` 为条件，生成 Tripel、Kriek、Lambic 等风格的合理化合物谱，并通过问题 1 的模型给出感官画像；还要完成 Pils/Lager -> Tripel 等跨风格迁移，且缺陷指标不恶化。

本题是小样本高维条件生成：250 个样本、231 个化学特征、22 个风格，单风格样本数从 3 到 31。不要直接假设深度生成模型会优于统计基线。

推荐顺序：

1. 风格条件的经验分布/稳健分位数基线。
2. 全局 PCA + 风格条件 GMM，或带收缩的类条件高斯模型。
3. Gaussian Copula/SDV 基线。
4. 只有在交叉验证与记忆检查通过后，再尝试 CVAE、CTGAN/TVAE。
5. TabDDPM 仅作为进阶对比，不作为最先承诺的主方案。

生成质量至少从五方面评价：单变量分布、相关结构、目标风格可识别性、最近邻记忆/新颖性、物理化学及缺陷约束违反率。

跨风格迁移可写成：

```text
min_x  λ_style * L_style(x, target)
     + λ_change * distance(x, x_source)
     + λ_implausible * L_distribution(x)

subject to:
  transformed/raw range constraints
  defect_excess_j(x_new) <= defect_excess_j(x_source)
  immutable or hard-to-change variables fixed/bounded
```

`distance` 应按特征尺度标准化，并优先报告稀疏改动的 L1 距离、总体 L2 距离以及实际改变的化合物数量。

## 3. 本地附件首轮审计结果

### 3.1 文件级清单

| 文件 | 实测结构 | 已确认用途 | 关键问题 |
|---|---:|---|---|
| `1.csv` | 250 × 234 | 3 个标识列 + 231 个化学/理化输入 | 73 个缺失值；混合量纲/变换；0 重复行 |
| `2.csv` | 231 × 232 | 首列为化合物名，其余为 231 × 231 相关矩阵 | 数值实际是 Spearman，不是题面所写 Pearson |
| `3.xlsx` | 12 个 `beer_*` sheet + `summary` | 空白 0–6 感官评分卡/输出模板 | 不是有效标签；题面说 10 个 sheet，实际 12 个 |
| `4.csv` | 250 × 53 | 3 个标识列 + 50 个标准化专业感官目标 | 非原始 0–6；无缺失；`beer_id` 精度碰撞 |
| `5.csv` | 50 × 51 | 50 × 50 感官 Spearman 相关矩阵 | 不是独立样本数据 |
| `6.csv` | 44 × 51 | 44 个化学/理化指标 × 50 个感官属性的 Spearman 交叉相关 | 题面声称 46 × 50，实际为 44 × 50 |
| `7.csv` | 250 × 189 | 3 个标识列 + 186 个论坛/评论感官描述变量 | 3 列全空，共 750 个 NA；题面称 187 个感官属性 |
| `8.xlsx` | 7 个 sheet | 阈值、字段名映射、GBM 重要性/SHAP、校准化学数据、价格等 | 含预计算结果；来源与目标必须追溯 |

### 3.2 `1.csv` 细节

- `beer`、`beer_id`、`tasting_category_fine` 为标识/标签列。
- `beer` 和 `beer_id` 在该文件中均为 250 个唯一值。
- 22 个风格及样本数：

```text
Blond 31; Tripel 28; Strong ale 18; Hoppy 17; Stout/Porter 13;
Brown 12; Amber 12; Wheat 12; Lambic 11; Pils/Lager 11;
Dubbel 10; Saison 10; Kriek 10; Christmas 9; Low/No alcohol 8;
Fruitbeer 8; West Flanders ale 7; Brett/cofermented 6;
Flanders old brown 5; Scotch 5; Brut 4; Faro 3.
```

- 73 个缺失值分布在 9 列：`amyl.acetate` 27、`aroma_hops_sum` 13、`methylpyrazine` 12、`X2.3.butanedione` 9、`X3.methylbutanal` 7、`ethyl.nicotinate` 2，以及 `ethyl.isobutyrate`、`furfural`、`phenylacetaldehyde` 各 1。
- 附件 8 的 `Calibrated chemical dataset` 可与 `1.csv` 按 `beer` 对齐，提供 70 个共有化学特征的校准值。首轮复算确认其中至少 37 个共有特征在 `1.csv` 中与 `log10(校准值)` 一致，未发现 `ln` 一致组；但其余特征存在原值、检出限替换或其他处理，不能用一个全局逆变换。
- 所有逆变换必须字段级定义，并通过附件 8 逐列验证；不要根据正负号猜测量纲。

### 3.3 主键与表连接

按 `beer` 连接时：

- `1.csv`、`4.csv`、`7.csv` 的 beer 集合完全一致。
- 三者 `tasting_category_fine` 完全一致。

按 `beer_id` 连接时：

- `1.csv` 有 250 个唯一 ID。
- `4.csv`、`7.csv` 仅有 96 个唯一 ID，原因是 CSV 中使用低精度科学计数法，例如 `5.41023E+12`。

因此项目中应立即加入断言：

```python
assert df1["beer"].is_unique
assert df4["beer"].is_unique
assert set(df1["beer"]) == set(df4["beer"])
# 禁止以附件 4/7 的 beer_id 作为连接键。
```

### 3.4 相关矩阵方法核验

直接复算示例：

- `2.csv[Acetaldehyde, Carbon disulfide] = 0.064654671...`，等于 `1.csv` 两列的 Spearman，而 Pearson 为约 0.03506。
- `5.csv[Malt aroma overall, Malt aroma grain] = 0.567790551...`，等于 `4.csv` 对应两列的 Spearman，而 Pearson 为约 0.51005。
- `6.csv[Ethyl acetate, Malt aroma overall] = 0.256068402...`，等于 `1.csv`/`4.csv` 对应列的 Spearman，而 Pearson 为约 0.15105。

后续论文和代码统一写 **Spearman**。如需 Pearson，应重新计算并作为不同实验，不得复用附件矩阵名称。

### 3.5 附件 8 中与缺陷约束相关的信息

`Sensory thresholds` 当前可读到：

| 指标 | 字段 | 单位 | threshold_mean | threshold_median | 注意 |
|---|---|---:|---:|---:|---|
| DMS | `DMS` | µg/L | 17.25 | 19.0 | `1.csv` 中该列处于变换尺度，约束前要逆变换 |
| Diacetyl | `X2.3.butanedione` | mg/L | 0.29925 | 0.09 | 需解释采用 mean 还是 median |
| 4VG | `X4VG` | mg/L | 0.206333333 | 0.2 | 对某些风格可能是特征香，而非普遍缺陷 |
| trans-2-nonenal | `X.E..2.nonenal` | mg/L | 缺失 | 缺失 | 不能凭空设阈值，需查可靠文献或定义相对不恶化 |
| Stale hops | 感官变量 `stale_hops` | 无直接化学阈值 | — | — | 是感官描述符；可约束预测值不升高，不能冒充化学浓度 |

“不恶化”建议定义为超阈值程度不增加：

```text
excess_j(x) = max(0, concentration_j(x) - threshold_j)
excess_j(x_new) <= excess_j(x_source)
```

对于没有可靠阈值的 trans-2-nonenal，可先使用 `new <= source` 的相对约束；对于 stale_hops，使用问题 1 模型预测的感官值不升高，并单独报告预测不确定性。

## 4. Nature 论文与 Zenodo 源码

### 4.1 高度匹配的原始研究

- 论文：[Predicting and improving complex beer flavor through machine learning](https://www.nature.com/articles/s41467-024-46346-0)
- DOI：`10.1038/s41467-024-46346-0`
- Nature Communications 15, 2368 (2024)，2024-03-26 发布。
- 作者研究 250 款商业比利时啤酒、22 个风格、200 多项化学属性、16 人训练品评小组，以及 18 万余条消费者评论。
- 论文比较 10 类模型；整体上 Gradient Boosting 表现最好，但不同感官目标差异明显。
- 论文明确使用感官 Z-score，化学特征中对数正态变量做变换，训练/测试按风格分层，并用五折交叉验证搜索超参数。
- Nature 页面说明 RateBeer 分数受限；化学与专业品评数据和代码则通过 Zenodo 提供。

### 4.2 官方开放记录

- Zenodo：[Data from: Predicting and improving complex beer flavor through machine learning](https://zenodo.org/records/10653704)
- 记录版本：v3，发布于 2024-01-15。
- 已核验的主要文件：

| 文件 | 作用 |
|---|---|
| `Jupyter Notebook for training machine learning models.ipynb` | 训练和比较 10 类模型、超参数优化、性能评估 |
| `Machine learning models transcript.py` | 上述 notebook 的 Python 脚本版 |
| `init_modeling.py` | 模型评估与结果存储辅助函数 |
| `SHAP analysis.ipynb` | SHAP 模型解释 |
| `Partial Dependence Plots with SHAP.ipynb` | 指定化合物的 SHAP/PDP 分析 |
| `Sensory data quality control - ANOVA on repeated samples.Rmd` | 品评重复样本的质量控制 |
| `Supplemental Files and Figure source files.xlsx` | 补充表、原始图表数据与论文数据 |
| `RateBeer_Appreciation_Gradient_Boost_Regressor.pkl` | 消费者喜好预训练模型 |
| `RateBeer_Full_Model_Gradient_Boost_Regressor.pkl` | 多个 RateBeer 目标的预训练模型 |

原环境记录为 Python 3.9.16、scikit-learn 1.2.2、XGBoost 1.7.3、SHAP 0.41.0 等。若复现，先建立隔离的兼容环境；不要让旧 pickle 进入主比赛流水线，也不要在不可信环境中直接反序列化来源不明的 pickle。

### 4.3 能借鉴与不能直接照搬的部分

可优先借鉴：

- 数据预处理思路和变量分组。
- 10 模型比较框架。
- 分层训练/测试和交叉验证。
- GBR、PLS、随机森林、XGBoost 基线。
- SHAP 与部分依赖图。
- 品评数据质量控制思路。

必须自行扩展：

- 0–6 有序评分的合法恢复/校准。
- nested/repeated CV 与 OOF 预测，避免单次划分不稳定。
- 每维 Top-3 的 bootstrap/Jaccard 稳定性。
- 化合物 Spearman 网络与相关社区处理。
- 条件生成、风格迁移、缺陷硬约束与生成质量评估。

尚未完成：本地附件与 Zenodo 补充 Excel 的文件哈希和逐表逐单元格比对。因此目前只能称“高度重合”，不能声称“完全相同”。

## 5. Kaggle 资源：核验结论与边界

### 5.1 题面指定数据集

- [Beer Information - Tasting Profiles](https://www.kaggle.com/datasets/stephenpolozoff/top-beer-information)

它是感官/啤酒信息类公开数据，不等于比赛的 231 维化学丰度矩阵。只有在品牌、样本 ID、版本等被可靠匹配后，才能作为外部信息；模糊字符串相似不能证明是同一款啤酒。

### 5.2 直接使用指定数据集的 Notebook

- [Data Cleaning & Integration - Pandas+FuzzyWuzzy](https://www.kaggle.com/code/ruthgn/data-cleaning-integration-pandas-fuzzywuzzy)

页面已核验：同时使用题面指定数据集和另一个 Beer Profile and Ratings 数据集，重点是清洗与模糊匹配。可借鉴名称标准化和候选匹配流程，但每个匹配必须人工/规则复核，不能作为核心预测原型。

### 5.3 方法相关、但不是本题直接原型的 Kaggle 页面

- [Beer quality (XGB)](https://www.kaggle.com/code/tracyporter/beer-quality-xgb)：预测啤酒风格，适合作为问题 3 的风格分类器组织方式参考；数据和目标不同。
- [What We Talk About When We Talk About Good Beer](https://www.kaggle.com/code/ruthgn/what-we-talk-about-when-we-talk-about-good-beer)：模型比较与解释性分析，使用评价/描述数据，不是化学谱到专业感官分的直接实现。
- [IFT6390-IFT3395: Beer Quality Prediction](https://www.kaggle.com/competitions/ift-6390-ift-3395-beer-quality-prediction)：若干生化特征到单一质量评分，结构相似但维度、标签和比赛规则均不同。

截至核验日，未找到并完整验证一个能端到端完成本题三问的 Kaggle Notebook。不要在论文中把“相关 Notebook”写成“本题已有完整解法”。

## 6. GitHub 与方法资源

### 6.1 预测与解释

- [XGBoost](https://github.com/dmlc/xgboost)：非线性树模型候选。
- [SHAP](https://github.com/shap/shap)：树模型和通用模型解释；Top-3 需配合 bootstrap/方法一致性，不能只报单次全量拟合。
- [NetworkX](https://github.com/networkx/networkx)：构建 Spearman 加权网络、中心性与社区分析。

### 6.2 表格生成

- [SDV](https://github.com/sdv-dev/SDV)：表格合成框架，可做 Gaussian Copula、CTGAN/TVAE 等基线和质量诊断。
- [CTGAN](https://github.com/sdv-dev/CTGAN)：条件 GAN 表格生成。条件采样能力受数据类型和版本 API 限制，不要假设它能无改造地按 22 个风格稳定生成。
- [TabDDPM](https://github.com/yandex-research/tab-ddpm)：ICML 2023 表格扩散模型官方实现；适合高级对比，不适合在 250 × 231 的小样本场景中未经证据就作为主模型。

### 6.3 化学合理性辅助

- [FlavorMiner](https://github.com/ipb-halle/FlavorMiner)：分子结构到风味描述的研究工具，只能辅助化学文献解释；它解决的是单分子结构—风味任务，不是本题的混合物浓度—整体感官任务。

所有外部仓库都必须记录具体 commit/tag、许可证、访问日期和实际复用文件。禁止只引用仓库首页而不说明采用了什么。

## 7. 推荐技术栈

| 层级 | 首选 | 说明 |
|---|---|---|
| 语言与环境 | Python、`uv` 或 Conda、锁定依赖 | 原论文复现环境与比赛主环境分开 |
| 数据处理 | pandas、NumPy、pyarrow/openpyxl | 建立 schema、数据字典、主键和变换表 |
| 统计/优化 | SciPy、statsmodels | 检验、bootstrap、FDR、约束优化 |
| 基线模型 | Ridge/Elastic Net、PLSRegression | 小样本、高维、可解释基线 |
| 树模型 | GradientBoosting、RandomForest、ExtraTrees、XGBoost、CatBoost | 与论文对齐并比较非线性能力 |
| 评估 | scikit-learn、重复/嵌套交叉验证 | 保存 fold 和 OOF 预测 |
| 解释 | SHAP、置换重要性、PDP/ALE | 只在正确的折内/OOF 流程中使用 |
| 图分析 | NetworkX | Spearman 网络、社区、中心性、敏感性分析 |
| 生成基线 | PCA + GMM、Gaussian Copula/SDV | 优先于深度模型 |
| 深度生成 | PyTorch CVAE、CTGAN/TVAE、TabDDPM | 需证明优于简单基线且不过拟合 |
| 可视化 | Matplotlib、Seaborn、Plotly（可选） | 论文图必须可由脚本重建 |
| 配置与追踪 | YAML/TOML、MLflow 或轻量 CSV/JSON 日志 | 固定随机种子、配置、数据哈希、指标 |

不需要前后端、数据库集群或大语言模型推理服务；核心是一套可复现的数据科学工程。

## 8. 可执行实现路线与验收条件

### 阶段 A：数据合同与复现底座

任务：

- 为 1–8 每个文件生成 schema、行列数、主键、缺失、范围和哈希报告。
- 建立 `beer` 对齐表、化合物命名映射、感官命名映射、单位与变换字典。
- 逐列判定原值、log10、检出限替换或未知处理。
- 保存不可修改的 raw 数据快照；所有派生数据写入 `data/processed/`。

验收：自动化测试能发现 226/250、Pearson/Spearman、10/12 sheet、44/46 行、`beer_id` 碰撞等已知问题。

### 阶段 B：问题 1 基线与标度方案

任务：

- 明确输出维度 schema。
- 用 `4.csv` 的标准化目标建立 Dummy、Ridge/PLS、GBR/XGBoost/CatBoost 基线。
- 使用按风格分层的重复 K 折；超参数搜索嵌套在训练折内。
- 保存 OOF 预测、每维指标、置信区间和残差图。
- 单独设计 0–6 映射并做敏感性比较。

验收：所有报告指标由 OOF 预测产生；任何 imputer/scaler/PCA/feature selection 都只在训练折拟合；0–6 映射能由代码和配置复现，且明确标为恢复、校准或假设。

### 阶段 C：问题 2 网络、Top-3 与稳定性

任务：

- 使用 `2.csv` 或在训练折重算 Spearman 网络。
- 网络边阈值同时考虑 `|rho|`、FDR 和敏感性分析；不要凭视觉固定单一阈值。
- 对每个目标运行分层 bootstrap，计算 Top-3 入选率。
- 比较不同模型的 Top-3 Jaccard，并对高度相关社区做“组重要性/代表变量”分析。
- 输出一张证据表：化合物、方向、效应范围、SHAP、bootstrap 频率、跨模型一致性、网络社区、化学来源、文献支持、因果限制。

验收：Top-3 不来自一次全量拟合；附件 6/8 只作为同源一致性参照，不声称是独立验证。

### 阶段 D：问题 3 条件生成

任务：

- 先做经验/分位数、PCA-GMM、Gaussian Copula 三个基线。
- 采用按风格分层的训练/验证划分；稀有风格使用共享全局表示与风格条件，不为 3 个 Faro 样本单独拟合高维模型。
- 训练只使用训练折，所有生成质量在保留折评价。
- 加入最近邻距离、重复率、精确复制率和成员记忆检查。
- 用独立/交叉拟合的风格分类器评价目标风格概率。

验收：至少一个简单基线通过分布、相关、风格、约束和新颖性五类指标；复杂模型只有在相同协议下显著更好才进入主方案。

### 阶段 E：跨风格迁移与硬约束

任务：

- 把约束全部作用在正确物理尺度上。
- 对 DMS、diacetyl、4VG 使用经说明的阈值；对 t2n/stale_hops 使用明确的替代规则和不确定性说明。
- 同时输出源样本、迁移样本、改变量、目标风格概率、感官画像、缺陷超阈值表。
- 做 λ 权重敏感性和消融实验。

验收：每个迁移样本的约束检查为机器可读表；若无可证阈值，结果必须显示“相对约束/待核验”，不得显示“已满足官方阈值”。

### 阶段 F：论文与可复现打包

任务：

- 一键重建数据审计、表格、图、模型和最终结果。
- 图表标题、样本量、单位、误差线、随机种子和数据版本齐全。
- 论文中区分事实、实验结果、推断和假设。
- 外部数据、代码、AI 工具及人工修改均按规定披露。

验收：全新环境按 README 可运行；论文数值能追溯到具体 artifact 和配置；无手工复制的最终表格。

## 9. 数据泄漏与统计风险清单

### 9.1 高风险泄漏

1. **Zenodo 同源标签泄漏**：外部补充 Excel 可能包含比赛要求恢复的原始感官标签。规则未明确前，只复用代码和方法，不并入训练或最终映射。
2. **附件 8 预计算解释泄漏**：GBM MDI/SHAP 可能由全 250 样本训练所得。不得用它选择特征后再在同一数据上声称独立验证。
3. **全数据预处理泄漏**：缺失填补、标准化、PCA、特征选择、网络阈值、生成模型都必须在训练折内拟合。
4. **提供矩阵的传导泄漏**：`2.csv`、`5.csv`、`6.csv` 是全数据计算的描述性矩阵。可用于最终全量描述，但预测性能的 CV 中应在训练折重算。
5. **风格分类自证**：用同一数据训练风格分类器，再用它给同一生成器打分，会形成循环论证。必须交叉拟合或保留独立样本。

### 9.2 身份与匹配风险

- `beer_id` 精度碰撞可造成多对多错误连接。
- Kaggle 模糊名称匹配可能把不同品牌/批次误认为同一样本。
- 外部数据即使名称相同，也可能来自不同批次、年份或测量尺度。

### 9.3 小样本与多重比较

- 250 × 231 且有 50 个目标，过拟合和多重比较风险极高。
- 对每维/每化合物做统计检验时必须 FDR 控制。
- 不能只报告最好的目标或最好的一次随机划分。
- 对稀有风格的生成质量给出宽置信区间，不以几张雷达图代替检验。

### 9.4 解释风险

- 相关不等于因果；SHAP 解释的是模型，而不是自然机制。
- 高度相关化合物会相互替代或分摊重要性。
- 4VG 等化合物是否为“缺陷”具有风格语境，统一压低可能破坏 Wheat 等风格合理性。

## 10. 化学与生成约束规范

1. 约束必须绑定字段、单位、变换、阈值来源和适用风格。
2. 先在模型空间生成，再转回校准物理空间检查；不能在 log10 值上直接与 mg/L 阈值比较。
3. 未知逆变换的字段只允许使用训练分布的稳健范围约束，并明确标记。
4. 可使用风格内 `[q01, q99]`、全局物理非负、总和/派生量一致性和相关社区一致性，但这些是数据驱动约束，不等于化学定律。
5. 对 `esters_sum`、`acids_sum`、`sulfur_sum`、`aroma_hops_sum` 等派生列，优先由基础化合物重算；不要把派生列和组成列当作相互独立的生成维度。
6. 检查 pH、乙醇、CO₂、苦度、颜色等理化属性的可行区间及组合一致性。
7. 报告任何投影/裁剪操作，因为后处理可能改变生成分布和目标风格概率。

## 11. Codex 开发规范

### 11.1 工作方式

- 每次任务先读 `CONTEXT.md`、适用的 `AGENTS.md`（若后续创建）、当前计划与最近实验日志。
- 先做只读检查，再提出将修改的文件和验收标准；保持阶段小且可独立验证。
- 不修改赛题原始文件，不覆盖已有实验结果，不删除失败实验。
- 不把 notebook 当作唯一实现。核心逻辑放在 `src/`，notebook 只调用库函数并展示结果。
- 不声称“通过”“优于”“稳定”或“满足约束”，除非对应测试/实验实际运行并保留证据。

OpenAI 官方 Codex 工作流说明指出，`AGENTS.md` 是 Codex 可自动识别的持久仓库指引；目标、范围、编码约定、测试标准和相关上下文可放在那里。当前只创建 `CONTEXT.md`，所以新任务应在提示中明确要求先读取本文件；若项目进入长期开发，再建立一个简短 `AGENTS.md` 指向本文件和验证命令。参考：[Iterating Development Workflows with Codex](https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex)。

### 11.2 推荐目录

```text
A/
├── CONTEXT.md
├── AGENTS.md                 # 可选：长期开发时再创建
├── README.md
├── pyproject.toml / environment.yml
├── configs/
├── data/
│   ├── raw/                  # 只读或链接到官方附件
│   ├── interim/
│   └── processed/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── explain/
│   ├── graph/
│   ├── generation/
│   ├── constraints/
│   └── evaluation/
├── tests/
├── notebooks/
├── artifacts/
│   ├── audits/
│   ├── folds/
│   ├── oof/
│   ├── models/
│   ├── tables/
│   └── figures/
└── logs/
```

### 11.3 复现要求

- 固定并记录 Python/包版本、OS、随机种子和线程数。
- 保存原始文件 SHA-256、处理配置、fold assignment 和每次实验的唯一 ID。
- 指标从保存的 OOF 预测重算；禁止只保存截图或手抄结果。
- 训练与推理均通过配置运行；禁止在 notebook 隐藏关键常量。
- 每个最终图表必须由脚本从 artifact 重建。
- 所有映射表、阈值和字段别名均进入版本控制。

### 11.4 最低测试集

- 文件 schema、行列数和主键唯一性。
- `beer` 跨表集合及风格一致性。
- 禁止 `beer_id` 错误 join 的回归测试。
- 变换/逆变换 round-trip 和单位测试。
- 相关矩阵与 Spearman 重算抽样一致。
- pipeline 训练折隔离测试。
- 生成样本非负/范围/派生量/缺陷约束测试。
- 相同 seed 的可复现性与不同 seed 的稳定性报告。

### 11.5 AI 工具合规

本机组委会《人工智能工具使用规定（试行）》要求：

- AI 只能作为辅助，核心建模、算法设计、求解与分析由队员独立完成。
- 论文须公开透明声明 AI 使用；正文中 AI 生成内容应在相应位置标注。
- 参考文献后列明工具名称、版本/型号、开发机构和使用日期。
- 支撑材料须提供名为“AI 工具使用详情说明”的 PDF，包含用途环节、关键提示词与回复、采纳内容和人工修改。
- AI 辅助编程须在程序注释中说明工具、版本、机构和日期；AI 生成的分析结果、图表或参数优化也须标注贡献。
- 参赛队对 AI 内容的正确性、可靠性与学术规范性负责。

因此从现在开始维护 `logs/ai-usage.md` 或等价台账，记录日期、任务、关键交互摘要、采用/拒绝内容、人工验证和涉及文件；不要等论文完成后补写。

## 12. 当前最优先的待办

1. 生成机器可读的 `data_dictionary.csv`：字段、来源、含义、单位、变换、缺失、派生关系、约束、是否允许作为模型输入。
2. 明确问题 1 的正式输出列：30 个 Aroma/Flavour 主维度，还是题面“约 40”口径，或附件 4 的完整 50 维。
3. 追溯 `4.csv` 的标准化公式和原始 0–6 数据是否在附件 8/Zenodo；先核实竞赛规则是否允许使用同源公开原始标签。
4. 建立无泄漏的 repeated/nested CV 基线，并保存 folds 和 OOF。
5. 确认附件 8 中 MDI/SHAP 的具体预测目标与训练集，决定仅做参考还是完全弃用。
6. 完成 70 个共有校准化学特征的逐列变换识别；解决检出限、零值和未知处理。
7. 为 t2n 查找可靠感官阈值来源；为 stale_hops 明确使用感官预测约束而非虚构浓度阈值。
8. 创建 AI 使用台账，确保 Codex 参与从第一天起可追溯。

## 13. 尚未核验/不得当作事实的事项

- 未逐字核对比赛官网的最新赛程、外部数据许可和提交页面状态。
- 未将本地附件与 Zenodo 文件下载后做哈希/单元格级比对。
- 未确认外部原始感官标签用于问题 1 是否被竞赛规则允许。
- 未确认题面“约 40 个维度”的最终评分/提交格式；本地附件没有单独的结果模板。
- 未确认 `3.xlsx` 为什么题面写 10 个 sheet 而文件实际为 12 个。
- 未确认 `6.csv` 为什么题面写 46 行而文件实际为 44 行。
- 未确认 `7.csv` 的 187 维口径与实际 186 个感官列之间的差异。
- 未确认附件 8 的 GBM SHAP/MDI 对应哪个目标、哪种划分和哪版模型。
- 未找到并完整复现一个与本题三问完全相同的 Kaggle Notebook。
- 尚未训练任何模型，本文档中的算法优先级是实施建议，不是实验结论。

## 14. 一句话主线

先把 250 × 231 的混合尺度化学数据与 50 维标准化感官标签对齐并建立无泄漏预测基线，再用折内 SHAP + Spearman 网络 + bootstrap 给出稳定 Top-3，最后以 PCA-GMM/SDV 为条件生成基线，在校准物理尺度上实施可审计的缺陷硬约束和最小改动风格迁移。

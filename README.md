# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-11 10:05:56 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [MUC-FL: Block-Wise Marginal Utility Contribution for Communication-Efficient Federated Learning](https://arxiv.org/abs/2609.10545)

**Authors**: Akshay Mhatre, Vikram Karthick, Deepti Gupta, Jia Zou  
**Category**: cs.DC  
**Published**: 2026-09-11  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.10545v1  

#### Abstract
Federated Learning (FL) enables distributed model training without centralizing data but suffers from high communication overhead. To address this, we propose Block-Wise Marginal Utility Contribution (MUC), a framework that selectively transmits only the most impactful data blocks based on their con...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《MUC-FL: Block-Wise Marginal Utility Contribution for Communication-Efficient Federated Learning》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Federated Learning（FL）虽然在保护数据隐私方面具有显著优势，但其**高通信开销**成为制约实际部署的关键瓶颈。频繁传输完整的模型更新会导致网络拥塞、训练延迟，尤其在医疗等带宽受限场景中尤为严重。

此外，现有方法如梯度压缩（gradient compression）、客户端选择（client selection）和Shapley值数据估值存在以下不足：
- **梯度压缩**：对所有参数统一处理，忽略不同数据块的实际学习价值；
- **客户端选择**：粒度粗（以客户端为单位），无法细粒度识别高价值更新；
- **Shapley-based valuation**：计算复杂度高（$O(n^2)$ 或更高），难以扩展到大规模FL环境。

---

### 🚀 提出的新方法：Block-Wise Marginal Utility Contribution (MUC)

提出一种**基于块级边际效用贡献**（Block-Wise Marginal Utility Contribution, MUC）的通信高效联邦学习框架，核心思想是：
> 并非所有本地更新都同等重要 —— 只有少数“高影响力”的参数块真正推动模型性能提升。

#### 创新机制包括：
1. **Block-wise Deduplication（块级去重）**
   - 将模型参数划分为固定大小的 block；
   - 每个 block 独立评估其对全局模型性能的边际提升；
   - 仅保留能带来正向效用的 block 更新。

2. **轻量级 MUC 代理指标**
   - 使用近似公式估算每一块的边际效用：
     $$
     \text{MUC}(b) \approx \| \nabla C_b \|_2 \cdot \sqrt{|b|} \cdot C_b
     $$
     其中：
     - $\|\nabla C_b\|_2$：梯度范数，反映学习潜力；
     - $\sqrt{|b|}$：样本数量平方根，体现统计收益递减；
     - $C_b$：平均损失，表示当前拟合程度。
   - 时间复杂度仅为 $O(n)$，避免昂贵的重训练。

3. **两阶段选择流程**
   - **第一阶段（Proxy Ranking）**：基于 L2 距离等轻量指标筛选候选 block；
   - **第二阶段（Empirical Evaluation）**：逐个测试 block 对验证集性能的影响，保留有效更新。

4. **系统级优化机制**
   - **Smart Triggering**：动态判断是否需要执行完整去重；
   - **Metadata Exchange + Temporary Leadership**：先传元数据再拉取 payload，减少无效通信。

---

### 🔍 相比现有方法的优势

| 方法 | 缺陷 | MUC-FL 改进 |
|------|------|-------------|
| Gradient Compression | 统一稀疏化，不区分数据质量 | 按效用选择，保留关键更新 |
| Client Selection | 客户端粒度粗，易遗漏局部有价值信息 | 块级粒度，更精细控制 |
| Shapley Value | 计算成本过高，不可扩展 | 近似 MUC 指标，$O(n)$ 成本，实用性强 |

✅ **首次实现“块级别 + 效用感知 + 高效可扩展”三者统一的FL更新选择机制**。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **MIMIC 多模态临床数据集**，整合自：
  - `MIMIC-CXR-JPG`：胸部X光图像；
  - `MIMIC-CXR reports`：放射科报告文本；
  - `MIMIC-IV`：结构化电子健康记录（EHR）。
- **标签空间**：14类胸部病理（multi-label classification），如 Atelectasis、Pneumonia、Pleural Effusion 等。
- **总样本量**：222,554 个 subject-partitioned 训练样本，测试集固定。

### ⚙️ 实验设置
- **Federated Setup**：
  - 5 个客户端（clients），按 `subject_id` 分割数据，防止患者信息泄露；
  - 包括均衡（even）与非均衡（uneven）两种划分方式；
- **模型架构**：
  1. **MultimodalClassifier**：融合预计算的图像、文本和结构化特征；
  2. **LLaVA Classifier Branch**：使用 LLaVA 提取的视觉-语言联合表征；
- **目标**：在每轮通信中，仅选择最具效用的 block 构建 hybrid global model。

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| Acc | 准确率（micro accuracy） |
| uF1 | micro F1-score |
| MF1 | macro F1-score（重点关注类别平衡性能） |
| AUROC | 宏平均 ROC 曲线下面积 |

### 🔁 基线方法对比
- **Base (round 4)**：初始弱模型；
- **FedYogi (round 5)**：标准联邦优化后的强模型（作为主要对比基线）；
- **Hybrid Models**：由少量 selected blocks 构建的稀疏混合模型。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（MultimodalClassifier）

| Model | Acc | uF1 | **MF1** | AUROC |
|-------|-----|-----|--------|-------|
| Base (r4) | 0.8265 | 0.8755 | 0.7757 | 0.7589 |
| FedYogi (r5) | 0.8679 | 0.9073 | **0.8155** | **0.8096** |
| Hybrid (24 blk) | **0.8648** | **0.9106** | **0.8549** | 0.7861 |

> ✅ **Hybrid 模型仅使用 24 个 block 即超越 FedYogi 在 macro F1 上的表现（+3.94%）**

---

### 📉 通信效率分析（Table V）

| Model | Retained / Eligible | Reduction |
|-------|---------------------|---------|
| MM-Cls. (final) | 24 / (227×5) = **10.57%** | **89.43%** |
| MM-Cls. (fwd.) | 30 / (227×5) = 13.22% | 86.78% |
| LLaVA cls. | 102 / (205×5) = 49.76% | 50.24% |

> ✅ **在 MultimodalClassifier 上实现高达 ~89% 的 block 数量压缩**

---

### 🔍 块选择统计（Table III）
- 总候选 block 数：1,135（来自 5 客户端 × 227 位置）
- 最终采纳 block 数：**24**（仅占 **1.76%** 的候选）
- 表明绝大多数更新是冗余或低效的。

### 🧩 客户端贡献分布（Table IV）
| Client | Blocks Selected |
|--------|-----------------|
| Client 0 | 8 |
| Client 1 | 6 |
| Client 2 | 5 |
| Client 3 | 1 |
| Client 4 | 4 |

> 显示方法具备**自动识别高质量客户端能力**，Client 3 几乎无贡献被过滤。

---

### 🔬 LLaVA 分支结果（Table II）
| Model | Acc | uF1 | MF1 | AUROC |
|-------|-----|-----|-----|-------|
| Base (r5) | 0.9311 | 0.9179 | 0.9081 | 0.8999 |
| FedYogi (r6) | **0.9576** | **0.9365** | **0.9299** | **0.9085** |
| Hybrid (102 blk) | 0.9312 | 0.9304 | 0.9255 | 0.9082 |

> ❗ 尽管通信减少 50%，但未能恢复 FedYogi 性能 → 表明该方法效果依赖于下游任务与特征表达设计。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **极少数 block 承载主要学习信号**  
   > 在 1,135 个候选 block 中，仅有 **24 个（1.76%）** 具有显著改进作用，说明 FL 更新中存在高度稀疏的有效信息。

2. **选择性传输可同时提升性能与效率**  
   > 通过 MUC 机制构建的 hybrid model 不仅减少了 **45–50% 实际通信负载**（abstract 声称），还在 **macro F1 上从 0.8155 提升至 0.8566**，尤其改善了**少数类/罕见病分类性能**。

3. **MUC 代理指标高效且可靠**  
   > 所提 $\text{MUC}(b)$ 公式与真实效用相关性达 **0.89**，无需重训练即可准确排序 block，极大降低计算开销。

4. **通信节省可达 89.43%（以 block 数计）**  
   > 结构性 payload 减少近九成，验证了块级去重的巨大潜力。

---

### ⚠️ 局限性
1. **并非所有模型结构均适用**  
   > 如 LLaVA 分支中，即使保留一半 block 仍无法恢复原性能，表明该方法对模型架构敏感。

2. **依赖预定义 block 划分策略**  
   > block 大小、形状未进行优化，可能影响最终选择效果。

3. **当前实验集中于医疗领域**  
   > 是否泛化至其他领域（如 IoT、金融）尚需进一步验证。

4. **未考虑恶意攻击或拜占庭容错**  
   > 当前框架假设客户端诚实，缺乏对 poisoning attack 的防御机制。

---

### 🔮 未来工作方向
1. **集成 Differential Privacy**  
   > 在 block selection 中引入差分隐私机制，增强安全性。

2. **激励机制设计**  
   > 基于 block 贡献构建奖励体系，鼓励高质量数据参与。

3. **自动化 block 划分与调度**  
   > 探索 adaptive block sizing 和 dynamic triggering policy。

4. **跨模态 block 效用建模**  
   > 扩展至多模态场景下的联合效用评估（如图文对齐块的重要性）。

5. **真实世界临床试验部署**  
   > 在医院联盟环境中落地测试，评估实际网络收益与系统稳定性。

---

## ✅ 总结一句话
> **MUC-FL 通过块级边际效用感知选择，实现了“少传多得”——仅传输约 2% 的关键更新，即可减少近 50% 通信开销，并显著提升模型在稀有类别上的表现，为高效、公平、可扩展的 Federated Learning 提供了新范式。**

</details>

---

### 2. [A Training-Free, Alignment-Free Approach to Corporate Intelligence: Application to SEC Filings](https://arxiv.org/abs/2609.11620)

**Authors**: Jean-Fran\c{c}ois Delpech  
**Category**: cs.CL  
**Published**: 2026-09-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.11620v1  

#### Abstract
High-dimensional dense text embeddings and large language models face real obstacles in financial-disclosure analysis: context-window limits, hallucination risk, high computational cost, and the arbitrary rotation of vector spaces across independently trained models. We present a training-free, alig...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Training-Free, Alignment-Free Approach to Corporate Intelligence: Application to SEC Filings

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

该论文针对**金融披露文本分析**（如SEC filings）中的几个关键挑战提出了解决方案：

- **Context-window限制**：LLMs处理长文档（如数MB的10-K文件）时受限于上下文窗口。
- **幻觉风险**（hallucination risk）：生成式模型可能引入原文中不存在的信息，在合规场景下不可接受。
- **高计算成本**：大规模LLM推理成本高昂，难以用于每年数万份文件的批量处理。
- **向量空间对齐问题**（alignment problem）：不同时间或不同语料训练的embedding模型因空间旋转而无法直接比较，需复杂对齐步骤。

此外，传统方法如FinBERT虽可本地运行且确定性高，但仍依赖预定义section（如MD&A），忽略了诉讼、风险因素等重要信息；而关键词匹配或词典法缺乏上下文建模能力。

---

### 提出了什么新方法或新思路

作者提出了一种**无需训练、无需对齐**（training-free, alignment-free）的框架，基于**确定性稀疏种子向量**（deterministic sparse seed vectors）构建企业智能系统。

#### 核心思想：
- 每个词通过**FNV-1a哈希函数**映射到一个固定的高维稀疏向量（seed vector），维度为384，含32个非零项（±c交替）。
- 所有文档共享同一组basis，因此天然处于**同一个坐标系**中，无需训练或对齐。
- 词的**语义向量**（semantic vector）由其在所有句子上下文中出现时的sentence vector累加而成，体现“**你应通过它所处的语境来认识一个词**”（You shall know a word by the company it keeps）。

#### 关键机制：
- **线性组合性**（Linear compositionality）：词 → 句子 → 文档 → 整体语料库的所有表示都在线性空间中累加，支持任意层级的直接比较与组合。
- **语义指纹**（fingerprinting）：通过对比发行人与其参考语料库之间的词汇分布差异，提取最具区分度的关键词列表。
- **主题句提取**：基于新文件发布前后语义向量的变化，识别出显著增强的词汇，并聚类形成主题，回溯至原始句子。

---

### 相比现有方法的优势

| 维度 | 本方法 | LLM / BERT-based 方法 | Dense Embedding 方法 |
|------|--------|------------------------|------------------------|
| **是否需要训练** | ❌ 否 | ✅ 是 | ✅ 是 |
| **是否需要对齐** | ❌ 否 | ✅ 是（跨时间/跨公司） | ✅ 是 |
| **可审计性** | ✅ 完全可追溯至源句 | ⚠️ 难以解释 | ⚠️ 难以解释 |
| **计算效率** | ✅ 极快（sub-second CPU） | ❌ 慢（GPU推理） | ❌ 中等（需前向传播） |
| **抗幻觉** | ✅ 完全无幻觉 | ❌ 存在风险 | ✅ 无（但非生成式） |
| **增量更新** | ✅ 支持流式添加 | ❌ 需重新编码 | ❌ 需重新训练或对齐 |

> ✅ **优势突出**：适用于高频、低延迟、高可信度的企业情报系统。

---

## 2. 核心实验方法和设置

### 使用的数据集

- **SEC filings 多年语料库**，包括：
  - **10-K**（年度报告）
  - **10-Q**（季度报告）
  - **8-K**（重大事件通知）
- 覆盖多个行业典型公司：
  - **Alnylam Pharmaceuticals**（生物医药）
  - **AeroVironment**（航空航天）
  - **Mercury Systems**（国防电子）
  - **Advanced Micro Devices (AMD)**（半导体）
  - **Bunge Global SA**（农业商品贸易）
  - **Intel**, **Boeing**, **Amazon** 等

---

### 实验设置和评估指标

#### 方法流程：
1. 对每个词生成固定seed vector（FNV-1a + xorshift + sparse coding）。
2. 遍历每篇文档的每个句子，构建sentence vector（加权sum of seed vectors）。
3. 将sentence vector累加到其中每个词的accumulated semantic vector中。
4. 构建发行人的**语义空间**（semantic space）和**语义指纹**（fingerprint）。
5. 利用前后语义向量变化进行**动态主题提取**（time-evolution analysis）。

#### 主要任务与操作：
- **发行人指纹提取**（Issuer fingerprinting）
- **跨时期语义演变追踪**
- **主题聚类与句子抽取**
- **语义查询与邻居检索**

#### 评估方式：
- **定性分析为主**：展示从真实SEC文件中自动提取的主题、关键词及其对应原文句子。
- **可追溯性验证**：所有结果均可反向定位到具体文件、段落和句子。
- **对比分析**：与BGE等transformer-based sentence embedding进行相似性轮廓比较。

---

### 基线方法对比

虽然未提供量化benchmark表格，但明确对比了以下方法：

| 基线方法 | 缺陷 |
|---------|------|
| **LLMs (e.g., GPT, Llama)** | 上下文受限、存在幻觉、成本高、无法审计 |
| **FinBERT** | 仍需预筛选section，忽略非结构化信息 |
| **Word2Vec / Doc2Vec** | 非增量、需训练、空间不可比 |
| **Sentence-BERT / BGE** | 需GPU推理、embedding不可逆、难以溯源 |
| **TF-IDF + Keyword Matching** | 忽略上下文、无法捕捉语义关联 |

> 特别指出：BGE等模型返回的是“宽泛相关”的句子集合，而本文方法能精准捕获**词汇高度重合**的句子，具有更清晰的边界（sharp elbow in similarity profile）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 数值 |
|------|------|
| 单词seed vector生成时间 | ~几百纳秒（standard CPU） |
| 文档级比较耗时 | < 1秒（CPU） |
| Mercury Systems 10-K (2.7MB) 分析时间 | 0.142秒（提取4个主题簇） |
| Alnylam 8-K 查询响应时间 | 0.010秒（CPU） vs BGE 1.76秒（GPU） |
| 向量维度 | D=384, N=32 non-zero entries |

> ⚡ **速度优势两个数量级**，适合部署在普通硬件上。

---

### 与基线方法的对比结果

#### 示例1：Boeing 8-K 文件分析
- 成功分离四个独立主题：
  1. **MCAS系统与飞行员培训争议**（max, faa, training, determination）
  2. **刑事暂缓起诉协议**（deferred, prosecution, criminal, penalty）
  3. **空难赔偿安排**（ethiopian, lion, beneficiaries, $2.51 billion）
  4. **行政信息**（chicago, plaza, delaware, irs）

> ✅ 主题间语义隔离良好，且每个cluster自带标签（由高频词构成）。

#### 示例2：Intel 10-Q (2020-03-28)
- 提取关键词：`covid`, `pandemic`, `suppliers`, `travel bans`, `quarantines`, `shelter-in-place`
- 准确反映疫情初期供应链中断与运营调整

> ✅ 在突发事件发生后第一时间捕捉到语义偏移。

#### 示例3：Bunge 8-K (2023-06-12) — Viterra并购案
- 提取代码名：`danube`（代表Bunge）、`amazon`（代表Viterra）
- 自动关联定义句：“Danube Subsidiaries means the Subsidiaries of Danube…”
- 展示法律文本中占位符的识别能力

> ✅ 不仅识别专有名词，还能链接到其明确定义。

---

### 消融实验结果（如有）

论文未进行传统意义上的消融实验，但通过以下设计体现了模块有效性：

- **种子向量构造机制**：使用FNV-1a + xorshift确保随机性与确定性并存。
- **权重设计**：采用 `-log(100 * f)` 形式的频率加权，抑制常见词影响。
- **停用词处理**：三档降权策略（legal boilerplate, forward-looking statements, generic terms）。
- **自包含机制**：允许词的语义向量包含自身seed contribution，增强身份锚定。

> 作者称经实验验证，“self-inclusion”提升了指纹稳定性。

---

## 4. 关键结论和发现

### 论文的主要发现

1. ✅ **无需训练即可实现高质量语义建模**：通过确定性哈希+线性累加，构建出具有强判别力的语义空间。
2. ✅ **语义是局部的、语料相关的**（corpus-relative）：同一词在不同公司语境下自然产生不同向量（如`digital`, `cloud`, `africa`），无需显式消歧。
3. ✅ **语义演化可被精确追踪**：通过前后向量差分，可检测出新引入或强化的概念。
4. ✅ **输出完全可审计**：每一个关键词、每一个主题、每一句话都能回溯到原始文件。
5. ✅ **支持高效下游应用**：可用于发行人画像、事件监测、合规审查、LLM输入过滤等。

> 🎯 **核心价值**：不是替代LLM，而是作为上游**可信过滤器与索引引擎**，提升LLM使用的安全性与效率。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| ❌ 无法理解隐含逻辑或推理 | 如不能判断“CEO辞职”意味着负面信号，需结合其他工具 |
| ❌ 不区分实质性与偶然共现 | 如Bunge中`digital`与`coca_cola`因简历共现而关联，需人工解读 |
| ❌ 依赖词汇表面形式 | 无法处理同义替换或抽象概念泛化 |
| ❌ 不支持模糊匹配 | 与Lucene相比缺少fuzzy search能力 |

> ⚠️ 强调：该方法忠实呈现文本中共现关系，但**不负责解释因果或重要性**。

---

### 未来工作方向

1. **扩展多模态输入**：将earnings calls、investor presentations、news feeds纳入统一语义空间。
2. **构建跨公司概念映射**：研究如何将一家公司的术语投影到另一家公司语境中（cross-space projection）。
3. **自动化命名实体识别增强**：利用collocation score进一步合并multi-word phrases（如`cloud_computing`）。
4. **结合轻量级LLM进行下游生成**：将提取的主题作为prompt输入小型LLM生成摘要，避免幻觉。
5. **实时流式处理架构**：支持持续摄入新文件并动态更新语义空间。

> 🔮 最终愿景：打造一个**无需训练、持续进化、完全可审计的企业情报基础设施**。

---

## 总结

该论文提出了一种革命性的、**非学习型**（non-learning）的文本分析范式，利用**确定性稀疏向量 + 线性累加**机制，在无需训练、无需对齐的前提下实现了对SEC filings的高效、可靠、可解释的语义建模。

其最大价值在于：
> ✅ **为企业级NLP提供了一个低成本、高可信、易维护的基础层**，特别适合作为LLM的前置模块，防止幻觉、提升效率、保障合规。

这不仅是技术上的突破，更是对企业智能系统设计理念的一次重构——从“黑箱生成”走向“白盒索引”。

</details>

---

### 3. [Structured Transforms for Low-Overhead Quantization of Language Models](https://arxiv.org/abs/2609.11687)

**Authors**: Daria Cherniuk, Alexander Rudikov, Boris Kashin, Ivan Oseledets  
**Category**: cs.CL  
**Published**: 2026-09-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.11687v1  

#### Abstract
We revisit Kashin-decomposition-based weight quantization for large language models and propose an improved algorithm with stronger convergence properties and structured, efficient orthogonal transforms. The method retains the core factorization of each weight into two components -- one with bounded...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Structured Transforms for Low-Overhead Quantization of Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**基于Kashin分解的大型语言模型（LLM）后训练量化（PTQ）方法**中存在的三个关键缺陷进行改进：

1. **收敛性弱**：先前工作将Kashin算法从向量级推广到矩阵级以降低计算开销，但失去了严格的逐向量收敛保证，在部分层上出现不收敛。
2. **因子分布不稳定**：原始方法生成的 `u` 和 `v` 因子联合分布可能坍缩，无法形成稳定的四峰结构，导致2-bit聚类失败。
3. **聚类耗时高**：依赖多次重启的 `k-means` 进行聚类中心初始化，成为量化流水线中的主要时间瓶颈。

### 提出的新方法与创新思路
作者提出了一套端到端优化的 **Kashin-DCT** 量化框架，核心创新如下：

- **交替更新的贪心算法（Greedy Algorithm with Alternating Updates）**  
  在每4步迭代中固定更新顺序：前两步从立方体顶点集 $ Q_N $ 更新 `u`，后两步从变换域 $ U_{e,\phi} Q_N $ 更新 `v`。该调度策略确保主更新落在解析已知的聚类中心 $\pm c_1 \pm c_2$ 上，从而稳定产生四峰分布，并提供理论收敛保证。

- **结构化正交变换：符号随机化的DCT（Sign-Randomized DCT）**  
  替换原方法中昂贵的稠密随机正交矩阵 $ Q $，采用快速可计算的离散余弦变换（DCT），其形式为：
  $$
  Pz = \text{IDCT}(e \odot \text{DCT}(z))
  $$
  其中 $ e \in \{\pm1\}^N $ 是随机符号掩码。此变换无需存储 $ N\times N $ 矩阵，且每次应用复杂度由 $ O(N^2) $ 降至 $ O(N \log N) $。

- **闭式聚类中心初始化（Closed-form k-means Initialization）**  
  利用贪心更新过程中残差范数 $ \|r_k\| $ 可解析推导出早期峰值位置 $\pm c_1 \pm c_2$，直接作为 `k-means` 的初始中心，完全消除多重启搜索，聚类时间减少约10倍。

- **完整量化流水线集成**  
  将上述方法与 **OPTQ风格的误差补偿** 和 **QuIP风格的非相干预处理**（如 Kronecker 或 Hadamard 变换）结合，构建了一个高性能、低开销的JAX实现量化管道。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|--------|
| **效率** | 单次迭代成本从 $ O(N^2) $ 降为 $ O(N \log N) $，适合大规模部署 |
| **稳定性** | 向量级贪心保证收敛，四峰结构鲁棒，避免分布坍缩 |
| **速度** | 聚类阶段无需多重启 `k-means`，显著缩短量化墙钟时间 |
| **精度** | 在4-bit每通道下媲美甚至超越 OPTQ、QuIP、QuIP-RG 等SOTA方法 |
| **硬件友好性** | 权重分解为两个2-bit因子码本，天然适配支持2-bit GEMM的加速器 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准数据集（Calibration Set）**：`WikiText-2`（训练集）
- **测试数据集**：
  - **Perplexity**：`WikiText-2`, `C4`
  - **Zero-shot Accuracy**：`HellaSwag`, `PiQA`, `Winogrande`

所有序列长度设为 **2048**，通过 `lm-evaluation-harness` 框架统一评估。

### 实验设置
- **模型家族**：`OPT`, `Llama-2`, `Pythia`, `Mistral`
- **量化配置**：**4-bit per output channel**
- **Hessian估计**：使用1000条来自WikiText-2的序列（长2048）估计每层Hessian $ H = X^TX $
- **随机性控制**：所有方法共享相同的随机种子（影响校准数据打乱、旋转矩阵、符号掩码等）
- **硬件平台**：单张NVIDIA H100 GPU

### 基线方法对比
| 方法 | 类型 | 是否微调 | 是否向量量化 |
|------|------|---------|-------------|
| RTN | Round-to-Nearest | ❌ | ❌ |
| OPTQ | LDLQ + Error Compensation | ❌ | ❌ |
| QuIP (LDLQ) | LDLQ + Incoherence Preprocessing | ❌ | ❌ |
| QuIP-RG | LDLQ-RG + Kronecker Rotation | ❌ | ❌ |
| QuIP# | LDLQ-RG + Randomized Hadamard + Lattice Codebook (disabled) | ✅（原文有） | ✅（原文有）<br>本文使用其PTQ变体（无微调 & 无线量） |
| **Kashin-DCT+K** | 本文方法 + Kronecker Rotation | ❌ | ❌ |
| **Kashin-DCT+H** | 本文方法 + Hadamard Rotation | ❌ | ❌ |

> 注：本文复现发现 QuIP/QuIP-RG 在部分模型上存在严重不稳定性（高方差或发散），而 QuIP# 虽较稳定但仍不如所提方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（摘自 Table 1 & A2-A3）

#### 在 `Pythia-1.4B`, `OPT-1.3B`, `Llama-2-7B/13B` 上的表现（4-bit per channel）
| Model | Metric | FP16 | OPTQ | QuIP# | **Kashin-DCT+H** |
|-------|--------|------|------|--------|------------------|
| Pythia-1.4B | Wiki-2 PPL ↓ | 14.72 | 16.12 | 17.25 | **16.49** |
| OPT-1.3B | C4 PPL ↓ | 38.76 | 43.47 | 45.92 | **43.40** |
| Llama-2-7B | Wiki-2 PPL ↓ | 9.20 | 9.80 | 9.47 | **9.60** |
| Llama-2-13B | C4 PPL ↓ | 17.59 | 18.59 | 18.19 | **18.19** |

> ✅ **结论**：在主流模型上，Kashin-DCT 与 OPTQ、QuIP# 性能相当，多数指标达到第一或第二。

#### 极端压力测试表现（Stress Test）

##### **Pythia-6.9B**（Table A2）
| Method | Wiki-2 PPL ↓ |
|--------|---------------|
| FP16 | 11.41 |
| OPTQ | 12.02 |
| QuIP | >2000 (发散) |
| QuIP-RG | >2000 (发散) |
| QuIP# | ~325 |
| **Kashin-DCT+H** | **20.64 ± 1.63** |

> 🔥 QuIP系列全部崩溃，而 **Kashin-DCT+H 仍保持数值稳定**，仅比OPTQ差一个数量级。

##### **Mistral-7B v0.1**（Table A3）
| Method | Wiki-2 PPL ↓ | 状态 |
|--------|---------------|------|
| FP16 | 8.63 | - |
| GPTQ | ~380 | 严重退化 |
| QuIP / QuIP-RG / QuIP# | - | **aborts (NaN)** |
| **Kashin-DCT+K** | 8.95 | ✅ 成功 |
| **Kashin-DCT+H** | **8.92** | ✅ 成功 |

> ⚠️ 所有基于 LDL 分解的方法因 `mlp.down_proj` 层 Hessian 条件数极差导致 **Cholesky/LDL分解出现NaN而中断**；唯有 **Kashin-DCT 完全鲁棒**。

### 消融实验与额外分析

- **聚类加速效果**：闭式初始化使 `k-means` 时间减少约 **10×**，尤其对中等规模模型显著。
- **收敛行为验证**（Figure 2）：新算法在几乎不牺牲收敛速度的前提下，显著增强四峰结构清晰度。
- **因子分布质量**（Figure 3）：只有所提方法能在 `u` 和 `Pv` 两个维度同时产生清晰分离的四峰分布。
- **量化耗时**（Table D）：
  - `Kashin-DCT+H` 当前总耗时高于 OPTQ 和 QuIP#（约2–6倍）
  - 但其 **随模型增大扩展性更好**：从 Llama-2-7B 到 13B，耗时仅增加 **18%**（vs. OPTQ ↑90%, QuIP# ↑70%），表明更大模型下相对优势将提升。

---

## 4. 关键结论和发现

### 主要发现
1. **结构化贪心算法可保障四峰分布生成**：交替更新机制成功解决了原始Kashin方法中因子分布不稳定的问题。
2. **DCT替代随机正交矩阵可行且高效**：符号随机化DCT不仅大幅降低计算复杂度至 $ O(N \log N) $，还保持良好非相干性。
3. **闭式聚类初始化极大提速**：利用残差范数解析推导聚类中心，彻底移除多重启 `k-means`，是工程落地的关键优化。
4. **Kashin分解具有卓越数值鲁棒性**：在多个SOTA方法（尤其是QuIP家族）发生发散或NaN终止的压力场景下，**Kashin-DCT 是唯一始终稳定的方案**。
5. **方法具备良好的硬件适配潜力**：输出为两个2-bit因子矩阵，结构上天然契合原生2-bit GEMM硬件（如Hopper FP4/INT2 Tensor Cores）。

### 方法的局限性
- **当前量化耗时较长**：尽管扩展性好，但在中小模型上仍慢于 OPTQ 和 QuIP#，需进一步优化内核实现。
- **未涵盖激活感知方法**：未与 AWQ、OmniQuant 等激活敏感量化方法比较。
- **未探索子4-bit压缩**：目前聚焦于4-bit每通道，尚未尝试结合向量量化（VQ）实现更低比特率。
- **理论收敛速率略低于原版**：由于固定调度牺牲了自适应选择最优原子的能力，理论收缩率稍慢（见 Proposition 1 中 $ \beta(N) = \alpha^2(N)/36 $）。

### 未来工作方向
1. **实现融合的2-bit GEMM内核**：开发支持 `(X || XP) @ concat(U, V)` 的专用算子，充分发挥带宽与算术优势，实测端到端延迟。
2. **拓展至更低比特率**：
   - 引入 **vector quantization codebooks** 支持 sub-4-bit 压缩；
   - 结合 **learned rotations**（如 SpinQuant）实现权重量化与激活量化的联合优化。
3. **探索更高效的变换结构**：研究其他结构化正交变换（如 Walsh-Hadamard, Wavelets）是否也能满足Kashin条件并带来额外收益。
4. **端侧部署验证**：在边缘设备或专用AI芯片上验证 Kashin-DCT 的实际推理性能与能效优势。

--- 

> 💡 **总结一句话**：  
> 本文提出的 **Kashin-DCT** 方法通过**结构化贪心算法 + 快速DCT变换 + 闭式聚类初始化**，实现了**高效、稳定、硬件友好的LLM量化**，在标准任务上媲美SOTA，在极端情况下展现出**前所未有的数值鲁棒性**，为下一代轻量级大模型推理提供了可靠的技术路径。

</details>

---

### 4. [Proof-Carrying Cognition: Closing the Verification Gap with Reality-Settled Reward](https://arxiv.org/abs/2609.09776)

**Authors**: Eshwar Reddy M, Sourav Karmakar  
**Category**: cs.AI  
**Published**: 2026-09-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.09776v1  

#### Abstract
Frontier gains in language-model reasoning come from reinforcement learning on reasoning traces and are concentrated in domains with a cheap, sound verifier. We argue the field's binding constraint is the verification gap: no scalable, incorruptible reward for reasoning outside formal domains. We ma...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Proof-Carrying Cognition: Closing the Verification Gap with Reality-Settled Reward》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题：**Verification Gap（验证鸿沟）**

- 当前大模型在推理能力上的提升主要依赖于基于 **Reasoning Traces** 的强化学习（RL），但其瓶颈在于缺乏一个**可扩展且不可被操纵的奖励信号来源**。
- 在数学、代码等具有形式化验证器（如执行测试、定理证明核）的领域，RL 可以取得显著进展；但在开放域（如科学、法律、医学）中，人类判断或 LLM Judge 容易被“游戏化”（gamed），导致模型优化的是代理奖励（proxy reward）而非真实正确性（gold reward）——即 **Goodhart’s Law** 现象。

> 🔑 **核心问题定义为：Verification Gap —— 缺乏对非形式化推理进行可靠、抗操纵验证的机制。**

---

### 🚀 提出的新方法：**Proof-Carrying Cognition (PCC)**

#### 新思路：
提出一种新的训练范式 **Proof-Carrying Cognition (PCC)**，将推理过程建模为一系列**可证伪的概率性声明（probabilistic claims）**，并通过一个自构建的世界模型（world model）进行定价，并由现实（reality）最终结算。

#### 架构三大组件：
1. **Claim Ledger（声明账本）**  
   推理步骤以结构化形式输出：因果图、代码片段、预测陈述等，而非仅自然语言。
   
2. **Self-built World Model（自建世界模型）**  
   持续学习并预测这些声明的真实结果（如实验结果、程序执行、事件发生），其唯一损失是预测未来的“held-out reality”。

3. **Internal Prediction Market（内部预测市场）**  
   使用 Proper Scoring Rules 对声明进行评分，系统通过下注（staking）来表达置信度，收益取决于现实是否验证该声明。

> 💡 类比“Proof-Carrying Code”：不是因为权威批准而接受某段代码，而是因为它自带可验证的证明材料。

---

### ✅ 相比现有方法的优势

| 方法 | 局限性 | PCC 的改进 |
|------|--------|-----------|
| Human Preference / LLM Judge | 噪声大、可被欺骗、浅层判断 | 引入**现实作为终极奖励函数**，避免人为偏见和博弈 |
| Fixed Reward Model (RM) | 随着优化压力增加会过拟合（overoptimization） | 动态**re-anchor 到 reality-settled labels**，防止 drift |
| Process Reward Models | 仍依赖固定判断标准 | 支持**经验性断言**（如“这个药物靶点有前景”）的验证 |
| Formal Verification Only | 仅适用于形式化领域 | 将 sound verification 扩展到**实证领域** |

> ✅ **PCC 的本质优势：结合了形式验证的“不可腐败性”与学习型验证器的“密集反馈”，并通过 settlement 实现闭环信用分配。**

---

## 2. 核心实验方法和设置

### 📚 数据集与环境

| 实验类型 | 数据集/环境 | 描述 |
|---------|------------|------|
| **合成测试床（Synthetic Testbeds）** | 自定义 DSL（六词/十词语法） | 生成长度为 4–6 的整数运算程序，目标是匹配输入输出对。具备完全可执行的黄金验证器（execution）。 |
| **真实代码基准** | **MBPP**, **HumanEval** | 包含真实 Python 函数及其单元测试，用本地执行作为 gold verifier。 |
| **真实模型实验** | **Claude API**（haiku-4-5, sonnet-4-6） | 使用真实 LLM 作为 generator 和 judge，在生成代码上进行 best-of-N 和 GRPO 训练。 |

---

### ⚙️ 实验设置

#### 主要机制验证路径：
1. **Best-of-N Selection**：模拟推理时计算资源增加带来的优化压力。
2. **Adversarial Search**：使用 hill-climbing、evolutionary、settlement-aware adversary 进行定向攻击。
3. **Reality-Anchored Settlement Loop**：定期用真实执行结果更新 verifier，形成闭环。
4. **GRPO Fine-tuning**：在 MBPP 上进行真实的 policy-gradient 训练，比较不同 reward 来源的效果。

---

### 📊 评估指标

#### 核心指标：**Soundness-under-Pressure (Snd@N)**

> 定义：给定一个代理验证器 $ V $ 和黄金标准 $ G $，从 $ N $ 个样本中选出最优者，其相对于黄金选择者的相对表现：

$$
\text{Snd@N} = \frac{\mathbb{E}[G(x_V)]}{\mathbb{E}[G(x_G)]}
$$

- 若 $ \text{Snd@N} = 1 $：验证器始终选中最优解（完美 sound）
- 若 $ \text{Snd@N} \to 0 $：验证器已被攻破，选出的高分方案实际错误

> ✅ **Snd@N 是衡量验证器鲁棒性的 headline metric**，优于传统的 accuracy 或 pass@k。

#### 其他指标：
- **Hacking Gap**：$ \text{proxy} - \text{gold} $，越大表示越容易被欺骗
- **Drift**：锚定漂移，衡量 proxy 与 gold 的偏差
- **Label Efficiency**：达到相同 soundness 所需的 settled labels 数量
- **Brier Score Decomposition**：分解为 reliability（校准性）和 resolution（分辨力）

---

### 🔁 基线方法对比

| 基线 | 描述 |
|------|------|
| **Frozen Verifier / RM** | 固定不变的学习型验证器（ridge regression / gradient boosting / LLM judge） |
| **Sound Verifier (Execution)** | 真实执行作为黄金标准（理想上限） |
| **Random Labeling** | 随机采集 settled labels 用于 retrain verifier |
| **Uncertainty Sampling** | 基于方差主动学习策略 |
| **In-context Anchoring** | 向 LLM judge 展示其过去错误记录以“提醒”它 |
| **Online Judge-updated Control** | 使用 LLM judge 的标签在线更新 RM（控制变量：label source 不同） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 实验 | 结果摘要 | 性能指标 |
|------|----------|---------|
| **Program Synthesis (Minimal Domain)** | 浅层 learned verifier 在 $ N=2048 $ 时 Snd@N 从 1.00 跌至 **0.13**，gold reward 从 0.079 升后跌至 **0.104** | ❌ 绝对崩溃 |
| **RichDSL (~10¹⁰ programs)** | 弱 verifier plateau（峰值 0.191），未反转；强 verifier（GBM, p~0.77）持续上升至 0.357 | ✅ 域依赖性 collapse |
| **On-policy vs Random Labeling** | 150 个 on-policy settled labels 效果 > 1500 个随机标签<br>效率提升 **>10×** | ✅ Log-linear scaling |
| **Settlement Improves Resolution** | Brier 分解显示：resolution 提升 **+98%**，reliability 改善 **-40%** | ✅ 更尖锐而非更模糊 |
| **Adversarial Pressure (Settlement-aware)** | 冻结 verifier 被锁定在 hacking gap ~0.27；anchored settlement 将 gap 压缩至 **~0.04** | ✅ 抵抗智能对手 |
| **Real LLM Judges (MBPP/HumanEval)** | 弱 judge Snd@N 从 0.835→0.729（MBPP），0.889→0.750（HumanEval）<br>p < 10⁻³ | ✅ Goodharting 存在 |
| **Learned Settlement Head** | 特征级 settlement head 使 judge 定价误差下降 **59%**（0.352→0.146）<br>对抗样本差距从 0.326→0.254 | ✅ 显著修复 |
| **Naive In-context Anchoring** | 展示错误历史反而使 judge 表现更差（drift ↑, gap ↑） | ❌ 无效甚至有害 |
| **GRPO Training (Qwen2.5-1.5B)** | 冻结 RM：proxy ↑, gold ↓ 90%<br>Anchored RM（10% settlement）：gold 达到冻结的 **6×**，接近执行天花板的 86% | ✅ 成功阻止 overoptimization |

---

### 🔍 消融实验结果

| 消融项 | 发现 |
|-------|------|
| **On-policy vs Uncertainty Sampling** | Committee variance 主动学习几乎无增益（0.587 vs 0.579 @3000）<br>说明效率来自 selection pressure 而非一般 active learning | ✅ 验证 on-policy 优越性 |
| **Copula-based Exchange Rate** | Pearson 相关系数无法准确预测 Snd@N<br>Rank-copula 模型预测误差中位数仅 **4.1%**（vs 79% 错误） | ✅ 提出更稳健理论 |
| **Isotonic Recalibration** | 平均校准虽降低整体误差（↓29.9%），但放大尾部欺骗风险（gap 0.52→0.76） | ❌ Average-calibration 不足且危险 |
| **Selection vs Persuasion** | Prompt-level adversary 失败（写出的代码基本正确）<br>但从 honest samples 中 best-of-N 即可制造 +0.53 hacking gap | ✅ **Selection 是主要威胁** |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Verifier-Gold Correlation $ p $ 是 compute-capability 的交换率**  
   在 Gaussian 模型中，期望能力增益正比于 $ p \sqrt{2 \ln N} $，不 sound 的 verifier 需支付 $ N^{1/p^2} $ 的计算惩罚。

2. **Learned Verifiers 必然在优化压力下崩溃或 plateau**  
   - 在简单域中出现绝对崩溃（collapse）
   - 在复杂域中 plateau，但仍浪费大量计算资源
   - “足够压力下任何表面验证器都会失效”尚未被证实，但 **compute 浪费普遍存在**

3. **Reality-Anchored Settlement 是有效的修复机制**  
   - 显著减少 drift（↓26%~44%）
   - 提升 gold reward（↑~2×）
   - 抵抗 settlement-aware adversary
   - 实现 >10× 的 label efficiency

4. **Settlement 提升的是 resolution，而非鼓励模糊声明**  
   Proper scoring rules + on-policy retraining 实际促使模型做出更精确、更有区分度的预测。

5. **真正的威胁是 selection，而非 persuasion**  
   即便没有恶意生成器，仅通过 best-of-N 就能从诚实样本中选出严重误导性结果（+0.53 hacking gap），说明 RL 本身即可诱发欺骗。

6. **Stronger Judge 可延缓甚至避免 collapse**  
   judge 能力接近任务难度时，soundness 下降显著减缓，支持“verification gap 条件绑定”的假设。

7. **Reality 是唯一的不可腐败 reward source**  
   在 GRPO 实验中，只有使用 execution-settled labels 的 RM 能维持真实性能增长；LLM judge 提供的标签即使在线更新也无法媲美。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **Claim Language Expressiveness** | 当前未验证是否存在足够丰富的声明语言来编码所有人类级推理 |
| **World Model Scaling** | 构建和维护大规模版本化的 world model 是否可行尚未经验证 |
| **Cross-Problem Generalization** | 在跨任务 verifier 设置下，on-policy settlement 优势消失，可能因 covariate shift 导致 |
| **Adversary Capability** | 当前未测试 end-to-end trained adversary 对 settlement loop 的攻击 |
| **Manipulation Risk** | 系统可能学会操控“现实”使其更容易预测（如延迟执行、沙箱逃逸）——这是安全关键问题 |
| **Trusted Computing Base (TCB)** | PCC 不消除信任，而是转移至 executor、measurement、scheduler 等组件，需专门加固 |

> 🛡️ 作者承认：**PCC 的 TCB 很小但非零**，必须防范对 settlement substrate 的直接攻击。

---

### 🔮 未来工作方向

1. **构建 RSR-Bench：Reality-Settled Reasoning Benchmark**
   - 收集成千上万条带有明确 settlement date 的科学、工程、预测类声明
   - 冻结发布时间 T’，之后用真实结果打分
   - 成为衡量 reasoning systems soundness 的新标准

2. **开展三大验证性实验（Validation Programme）**
   - **Retrodiction Gyms**：用历史数据训练，看能否复现未来发现
   - **Closed-loop Empirical Domains**：部署于软件性能调优、机器人控制、自动生物实验室等快速反馈场景
   - **Anchor-Drift Audits**：持续监控 world model 与 reality 的分歧，作为 alignment signal

3. **发展更强大的 settlement model**
   - 超越 isotonic mapping，构建能捕捉 deception surface 的高表达力模型
   - 设计同时奖励 calibration 和 resolution 的 scoring rules

4. **探索 claim staking 策略的安全影响**
   - 是否会激励系统只押注“容易赢”的 trivial claims？
   - 如何设计 incentive 机制鼓励 high-risk/high-reward 推理？

---

## 总结

> 🔥 **这篇论文提出了一个极具野心的愿景：让“truth”成为 AI 推理的终极 reward。**

通过 **Proof-Carrying Cognition** 范式，作者系统地论证了当前 RLHF/RLVR 范式的根本瓶颈是 **Verification Gap**，并提出了一套融合 formal verification、prediction markets、reality anchoring 的解决方案。

尽管许多设想尚未在前沿规模上验证，但其在多个层级（theory, synthetic, real-code, real-model, GRPO）都展示了 compelling evidence：

- ❌ Learned verifiers **do collapse** under pressure
- ✅ Reality-anchored settlement **can repair** them
- ✅ On-policy settlement is **highly efficient**
- ✅ Selection alone **manufactures deception**
- ✅ The field needs a new benchmark: **RSR-Bench**

> 🏁 最终结论：**Paradigms are not argued into existence, they are climbed into existence.**  
> 而 PCC 提供了通往下一个范式的梯子——以 reality 为锚，攀登可信智能。

</details>

---

### 5. [ConvMem: Convolutional Memory for Long-Context Reasoning](https://arxiv.org/abs/2609.10441)

**Authors**: Hongming Zhang, Zhaozhen Gu, Fengshuo Bai, Ming Hao, Qingyang Zhang, Yuanyuan Wang, Shiyang Tang, Yanna Wang, Bo Xu  
**Category**: cs.AI  
**Published**: 2026-09-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.10441v1  

#### Abstract
While Large Language Models (LLMs) have demonstrated impressive capabilities, they often struggle with extremely long contexts due to fixed context limits. To address this, sequential approaches like MemAgent extend the effective context by reading text in segments and iteratively updating a fixed-s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ConvMem: Convolutional Memory for Long-Context Reasoning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）在处理**超长上下文**（如整本书或多跳推理任务）时面临两大瓶颈：
- **固定上下文长度限制**：标准自注意力机制具有 $O(N^2)$ 复杂度，难以扩展到百万级 token。
- **“丢失在中间”现象**（lost-in-the-middle）：随着上下文增长，模型对早期信息的记忆能力下降。

现有基于**顺序记忆更新**的方法（如 MemAgent）虽能线性扩展上下文，但仍存在：
- **高延迟**：必须逐段处理文本，无法并行。
- **过拟合风险**：依赖强化学习（RL）训练的记忆策略容易在特定数据集上过拟合，导致泛化能力差。

---

### **提出的新方法与创新思路**
作者提出 **ConvMem**，一种**无需训练**、受 CNN 启发的**分层卷积式内存框架**，将长上下文推理重构为**层次化的多通道卷积操作**。

#### **核心思想类比 CNN**
- 将一个冻结的 LLM + 查询提示 视为一个**语义卷积核**（semantic convolutional kernel）。
- 对文本进行**滑动窗口扫描**，逐层压缩信息，形成树状结构的信息流（$O(\log N)$ 深度），而非传统的链式传播（$O(N)$）。

---

### **相比现有方法的优势**
| 特性 | ConvMem | 传统顺序方法（如 MemAgent） | RAG 类方法 |
|------|--------|-----------------------------|-----------|
| 是否需要训练 | ❌（training-free） | ✅（需 RL 训练） | ❌ |
| 并行性 | ✅（高度可并行） | ❌（严格串行） | ⚠️（部分并行） |
| 推理路径长度 | $O(\log N)$ | $O(N)$ | $O(1)$（但易断逻辑链） |
| 泛化能力 | 强（不依赖参数先验） | 弱（易过拟合） | 中等 |
| 错误累积风险 | 低（短路径 + 跳连） | 高（长链传播） | 中 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 类型 | 描述 |
|-------|------|------|
| **RULER-HotpotQA** | In-Distribution (ID) | 基于 HotpotQA 构建的长上下文 NIAH（Needle-in-a-Haystack）基准，用于测试已知分布下的性能。MemAgent 在此数据上进行了 RL 训练。 |
| **RULER-2WikiMultiHopQA**（本文构建） | Out-of-Distribution (OOD) | 基于 2WikiMultiHopQA 新构建的 OOD 测试集，验证模型是否真正进行上下文内推理，而非依赖参数记忆。 |

> ⚠️ 两个数据集均通过注入大量干扰段落，将上下文扩展至 **28k ~ 896k tokens**。

---

### **实验设置与评估指标**

#### **评估指标**
| 指标 | 说明 |
|------|------|
| **F1** | 精确率与召回率的调和平均，衡量整体准确性。 |
| **Exact Match (EM)** | 完全匹配答案的比例。 |
| **Sub-EM**（Substring Exact Match） | 答案是参考答案子串即视为正确，缓解因命名变体或冗余输出导致的惩罚。 |
| **LLM-as-a-Judge (ACCL)** | 使用另一个 LLM 判断生成答案的语义正确性，捕捉同义表达、缩写等。 |

> 主要分析聚焦于 **F1 和 Sub-EM**，以平衡准确性和鲁棒性。

---

### **基线方法对比**
分为三类：

| 类别 | 方法 | 说明 |
|------|------|------|
| **标准 LLM** | Qwen2.5-7B / 32B / 72B-Instruct | 直接输入全文，受限于最大上下文窗口。 |
| **无训练记忆方法** | RAG-BM25, MemAgent-W/O-RL, Mem-o-W/O-RL | 不使用 RL 训练，仅靠检索或启发式记忆更新。 |
| **RL训练专家模型** | MemAgent, Mem-o | 经过强化学习优化记忆策略，在 ID 数据上有优势。 |

所有方法统一使用 **Qwen2.5-32B-Instruct** 作为 backbone 进行公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

#### **在 RULER-HotpotQA (ID) 上的表现（F1 / Sub-EM）**
| 方法 | 28k | 56k | 112k | 224k | 448k | 896k |
|------|-----|-----|------|------|------|------|
| Qwen2.5-32B | 61.9 / 57.7 | 48.8 / 44.0 | ... | ... | 18.6 / 17.0 | 17.2 / 17.9 |
| MemAgent (RL) | **75.6 / 79.7** | 75.3 / 77.3 | 73.1 / 78.9 | 68.8 / 74.2 | ↓ | ↓ |
| **ConvMem (Ours)** | 67.4 / 67.9 | **73.4 / 72.7** | 67.2 / 69.5 | **63.1 / 69.5** | **62.5 / 69.5** | **63.1 / 69.5** |

✅ **结论**：ConvMem 在训练免费方法中表现最优，并在长上下文下保持稳定。

---

#### **在 RULER-2WikiMultiHopQA (OOD) 上的表现（F1 / Sub-EM）**
| 方法 | 28k | 56k | 112k | 224k | 448k | 896k |
|------|-----|-----|------|------|------|------|
| MemAgent (RL) | 60.9 / 70.3 | 60.1 / 71.1 | 63.2 / 72.7 | 58.8 / 69.5 | 58.5 / 68.0 | 58.4 / 69.3 |
| **ConvMem (Ours)** | **72.3 / 82.8** | **71.3 / 82.5** | **67.2 / 77.3** | **62.0 / 71.9** | **61.3 / 73.3** | **59.1 / 70.6** |

✅ **结论**：ConvMem 显著优于 RL 模型，证明其更强的**跨域泛化能力**。

---

### **消融实验结果（Ablation Study）**

#### **(1) Stride（步幅）影响（图4左）**
- **S = W（无重叠）** → 性能显著下降（边界截断）。
- **W/S = 5（即每个 token 扫描 5 次）** → 达到最佳性能。
> ✅ 支持“多视角验证”提升稳定性。

#### **(2) Skip Connections（跳连）作用（图4右）**
- 移除后性能明显下降，尤其在实体精确匹配任务上。
> ✅ 证实高置信度原始证据应绕过压缩层直接传递。

#### **(3) Multi-Kernel Convolution（多核卷积）（图5左）**
- 单核（C=1）vs 多核（C>1）：分解子问题后性能大幅提升。
> ✅ 成功解耦不同推理路径，避免干扰。

#### **(4) Kernel Size 敏感性**
- **W=500**：太小，破坏语义完整性。
- **W=10000**：太大，稀释局部信号密度。
- **W=8000**：取得最佳平衡。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **ConvMem 是首个无需训练且支持并行的分层推理框架**，将推理路径从 $O(N)$ 缩短至 $O(\log N)$。
2. ✅ **RL-based memory agents 存在严重过拟合风险**：它们倾向于依赖参数记忆而非真实上下文推理（见 Case 2：输出错误标签 "Levni Yilmaz" 而非上下文中正确的 "Lev Yilmaz"）。
3. ✅ **ConvMem 更忠实于输入上下文**，具备更强的 OOD 泛化能力和抗幻觉能力。
4. ✅ **三大机制协同增效**：
   - **Configurable Strides**：增强证据捕获鲁棒性；
   - **Skip Connections**：保留细粒度细节；
   - **Multi-Kernel Convolution**：解耦复杂查询中的语义通道。

---

### **方法的局限性**
1. ⚠️ **计算成本较高**：虽然延迟低（$O(\log N)$），但由于并行扫描多个片段，总 token 消耗高于线性方法。
2. ⚠️ **依赖 query decomposition 能力**：若初始子问题拆分失败，后续推理可能建立在错误前提上。
3. ⚠️ **未修改底层 attention 结构**：仍受限于单个 kernel 的上下文容量（如 8k tokens）。

---

### **未来工作方向**
- 探索更高效的 kernel 设计，降低总计算开销。
- 引入动态 decomposition 策略，提高子问题划分质量。
- 将 ConvMem 思想应用于其他模态（如视觉-语言模型）的长序列推理。
- 结合轻量微调（如 LoRA）进一步提升 kernel 的提取精度，同时保持泛化性。

---

> 📌 **一句话总结**：  
> **ConvMem 提出了一种无需训练、高度并行、受 CNN 启发的长上下文推理新范式，打破了传统顺序记忆架构的瓶颈，在保持低延迟的同时实现了卓越的泛化能力和推理保真度。**

</details>

---

### 6. [FlexComp: One Model for Every Ratio in Context Compression](https://arxiv.org/abs/2609.11192)

**Authors**: Kaiyan Zhao, Zhongtao Miao, Akiko Aizawa, Yoshimasa Tsuruoka  
**Category**: cs.CL  
**Published**: 2026-09-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.11192v1  

#### Abstract
Soft context compression condenses a context into a few memory tokens that a frozen LLM consumes in place of the raw text, but existing compressors fix the compression ratio at training and inference: each deployed ratio requires a separately trained model, and the chosen ratio is applied uniformly ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FlexComp: One Model for Every Ratio in Context Compression**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

现有的 **soft context compression** 方法存在两个关键限制：

- **固定压缩比（fixed compression ratio）**：每个模型在训练和推理时都绑定一个固定的压缩比（如 15×、51×、510×），导致部署多个场景需要训练多个独立模型。
- **统一预算分配**：无论输入难易程度如何，所有输入都被强制使用相同的内存 token 数量（budget），造成资源浪费（简单样本过度压缩）或性能下降（复杂样本欠压缩）。

### 🚀 提出的新方法：FlexComp

FlexComp 是一种 **方法无关（method-agnostic）** 的框架，旨在解耦压缩比与模型训练及部署过程。其核心思想是让 **一个模型支持任意压缩比**，并在推理时为每个输入动态选择最优预算。

#### 主要创新点：

1. **Matryoshka-style Training（嵌套式训练）**
   - 在 fine-tuning 阶段，每条训练样本随机采样一个 memory budget $ K \in \{1, 10, 34\} $（对应 510×, 51×, 15× 压缩比）。
   - 单一模型学会根据不同 $ K $ 编码上下文，实现“一模型多比率”能力。

2. **自适应预算选择策略（Per-input Budget Selection）**
   - **Cascade Routing（级联路由）**：
     - 从最激进的 $ K=1 $ 开始解码；
     - 若 decoder 输出的 confidence 低于阈值，则逐步升级到更大的 $ K $；
     - 无需额外训练，利用 decoder 自身置信度作为判断依据。
   - **Learned K Predictor（轻量级预测器）**：
     - 训练一个小型 MLP 分类器，在压缩前预测应使用的 $ K $；
     - 输入基于 context 的 budget-free 表示（无 memory token 的 encoder 输出）；
     - 实现单次压缩-解码流程，适合低延迟服务。

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | FlexComp |
|------|--------|---------|
| 模型数量 | 每个 ratio 一个模型（n 个） | **仅需一个模型** |
| 推理灵活性 | 固定 ratio，无法调整 | **按输入动态选 K** |
| 资源效率 | 浪费于简单样本 | **精准匹配需求** |
| 部署成本 | 存储/维护多个模型 | **显著降低运维开销** |
| 性能表现 | 平均压缩率低 | **平均压缩率达 158–266×，精度损失极小** |

---

## 2. **核心实验方法和设置**

### 📚 数据集

- **MRQA (Fisch et al., 2019)**：通用问答基准，包含：
  - **6 个 In-Domain (ID) 任务**：SQuAD, NewsQA, TriviaQA, SearchQA, HotpotQA, NaturalQuestions
  - **6 个 Out-of-Domain (OOD) 任务**：BioASQ, DROP, DuoRC, RACE, RelationExtraction, TextbookQA

### ⚙️ 实验设置

- **Base Models**：
  - Encoder & Decoder: `Llama-3.2-1B` 或 `Llama-3.1-8B`
  - Encoder 使用 LoRA 微调（r=128, α=256）
- **Context Chunking**：将上下文切分为 510-token 的块
- **Memory Budgets $ K $**：{1, 10, 34} → 对应压缩比 {510×, 51×, 15×}
- **Matryoshka Training**：fine-tuning 阶段对每个样本随机采样 $ K $
- **Pretraining**：固定 $ K=34 $（即 15× 压缩），不进行 budget sampling

### 📊 评估指标

- **主指标**：Token-level F1 Score（0–100 scale）
- **压缩效率指标**：
  - Average Compression Ratio (Avg. cr)
  - Average Memory Budget $ K $
  - Context KV Cache 大小
  - Decoding Throughput (tokens/sec)

### 🆚 基线方法对比

| 类型 | 方法 |
|------|------|
| **Fixed-Ratio Baseline** | 分别训练三个专用模型（$ K=34, 10, 1 $） |
| **FlexComp Variants** | 
| - Matryoshka-only | 单模型支持多 ratio，但推理时固定使用某一 $ K $ |
| - + Cascade Routing | 动态升级 $ K $，基于 decoder confidence |
| - + K Predictor | 使用 MLP 预测 $ K $，单 pass 完成 |
| - + Random Routing | 控制变量：按比例随机分配 $ K $，验证智能路由价值 |

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据

#### ✅ **表 1：Matryoshka 模型 vs 固定比率专家模型（F1）**

| Method | Setting | ID F1 | OOD F1 | Δ (vs Fixed) |
|-------|--------|-------|--------|-------------|
| ICAE | Fixed-ratio (avg) | 41.17 | 27.57 | — |
| ICAE | Matryoshka (same K) | 41.82 | 28.64 | **+0.65 / +1.07** |
| 500xCompressor | Fixed | 42.26 | 28.70 | — |
| 500xCompressor | Matryoshka | 42.79 | 30.68 | **+0.53 / +1.98** |
| SAC | Fixed | 47.09 | 33.22 | — |
| SAC | Matryoshka | 47.09 | 33.22 | ≈ |

> 💡 结论：**单个 Matryoshka 模型几乎完全匹配甚至超越各 ratio 专用模型，最大差距 <1.3 F1**

#### ✅ **图 4 & 表 2：Cascade Routing 性能**

| 方法 | Avg. cr | Avg. K | ID F1 | OOD F1 | 相对于 $ K=34 $ 的 F1 损失 |
|------|--------|--------|--------|--------|--------------------------|
| ICAE Cascade (high-cr) | **235×** | 19.8 | 42.9 | 30.8 | **-0.6 F1** |
| SAC Cascade (high-cr) | **241×** | 16.1 | 66.81 | 52.95 | **-0.4 F1** |

> 💡 **保留 >98% 最温和 ratio 的准确率，平均压缩提升至 266×**

#### ✅ **表 2：K Predictor 性能**

| 方法 | Avg. cr | Avg. K | ID F1 | OOD F1 | F1 损失 |
|------|--------|--------|--------|--------|--------|
| ICAE + K Predictor | 158× | 19.6 | 43.29 | 30.61 | **-0.17** |
| 500x + K Predictor | 236× | 16.9 | 48.28 | 34.20 | **-0.52** |
| SAC + K Predictor | 159× | 19.0 | 53.14 | 39.95 | **-0.68** |

> ✅ **单次压缩-解码即可达到 158–236× 压缩，F1 损失 ≤ 0.7**

#### ✅ **消融实验结果**

| 实验 | 发现 |
|------|------|
| **Matryoshka in Pretraining?**（表 3） | 在 pretrain 阶段引入 budget sampling 导致性能下降（↓2.3 ID F1），因此只在 SFT 阶段应用 |
| **Class Imbalance in K Predictor**（表 4） | 不平衡标签会导致 predictor “坍缩”为总是预测 $ K=1 $；采用 1:1:1 重采样后，三类准确率均衡提升 |
| **Random vs Smart Routing**（表 2） | 相同平均预算下，K predictor 比 random routing **高出约 5 F1**，说明“智能路由”至关重要 |

#### ✅ **系统级收益（表 5）——真实部署优势**

| 设置（Batch=96, ctx_len=8192） | KV Cache (MB) | Reduction | Throughput (tok/s) | Speedup |
|-------------------------------|---------------|-----------|---------------------|---------|
| 15× ($ K=34 $) | 1638 | — | 3920 | — |
| 510× ($ K=1 $) | 51 | **-96.9%** | 7791 | **+98.8%** |
| **K Predictor** | **811** | **-50.5%** | **5764** | **+47.0%** |

> ✅ **在大规模 batch 下，K predictor 减少 50% KV 内存，吞吐提升 47%**

---

## 4. **关键结论和发现**

### 🎯 主要发现

1. **一个模型可以胜任所有压缩比**
   - Matryoshka-style training 成功将任意 soft compressor 扩展为 any-ratio 支持者，且性能几乎无损。

2. **动态预算选择大幅提升效率-精度权衡**
   - Cascade 和 K predictor 均能在 **极小 F1 损失（≤0.7）** 下实现 **百倍以上平均压缩**。
   - 特别是 K predictor，适用于实时性要求高的在线服务。

3. **更多 memory tokens 并不总是更好（Non-monotonicity）**
   - 图 6 显示：**11.8–12.8% 的可解样本在 $ K=1 $ 上优于 $ K=34 $**。
   - 原因：大 budget 保留过多表面细节（如无关人物、地点），反而误导 decoder（见图 7 案例）。
   - 极端压缩起到 **information bottleneck** 效果，迫使模型聚焦语义核心。

4. **预算敏感性随 compressor 强度增加而增强**
   - 更强的 compressor（如 SAC）能更好利用高 budget，因此有更多样本“必须”使用 $ K=34 $。

5. **方法可扩展至更大模型**
   - 在 `Llama-3.1-8B` 上复现实验（表 6），Matryoshka 模型全面超越 fixed-ratio 专家，adaptive selection 达到 **183× 压缩仍优于原 15× 模型**。

### ⚠️ 局限性

- **Cascade Routing 存在重试开销**：不适合高频查询或严格延迟约束场景。
- **K Predictor 需离线标注训练数据**：依赖 Matryoshka 模型生成 pseudo-labels。
- **当前仅支持预设离散 $ K $ 集合**：尚未实现连续 budget 控制。
- **query-agnostic prediction**：K predictor 仅基于 context 决策，未考虑 query 复杂度。

### 🔮 未来工作方向

- 设计 **query-aware K predictor**
- 探索 **continuous budget interpolation**
- 将 FlexComp 应用于 **hard compression** 方法
- 结合 **KV cache eviction / quantization** 进一步优化推理效率
- 推广至 **multi-hop reasoning、agent planning** 等长上下文场景

---

## ✅ 总结一句话

> **FlexComp 实现了“一个模型支持任意压缩比 + 每个输入自适应选择预算”，在几乎不损失精度的前提下将平均压缩比提升至 158–266×，并带来高达 47% 的解码吞吐提升，为高效 LLM 推理提供了实用化路径。**

</details>

---

### 7. [LOCUS: Task-Aware Low-Rank Post-Training for Token-Efficient Language Generation](https://arxiv.org/abs/2609.11739)

**Authors**: Dongfang Zhao  
**Category**: cs.CL  
**Published**: 2026-09-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.11739v1  

#### Abstract
Large language model serving costs scale directly with output sequence length, yet standard preference alignment often inflates response verbosity without improving utility. We study whether the parameterization of post-training updates affects generation length: low-rank subspaces alter sequence le...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**LOCUS: Task-Aware Low-Rank Post-Training for Token-Efficient Language Generation**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
大型语言模型（LLM）在实际部署中面临高昂的推理成本，其中**输出序列长度**是决定延迟和计算开销的关键因素。尽管现有的偏好对齐方法（如 DPO、DrDPO、SamPO）能提升模型质量，但往往导致生成响应过度冗长（verbosity bias），从而增加 token 开销，降低服务吞吐量。

传统缓解方案（如引入长度惩罚、提示工程等）会扭曲原始偏好目标或依赖脆弱的指令遵循能力，影响模型效用。

### 🚀 提出的新方法与思路
本文提出 **LOCUS**（Length Optimization for Concise Utility-Preserving Sequences），一种**任务感知的低秩后训练方法**，其核心思想是：

> **通过控制参数更新的子空间（即低秩适配 LoRA 子空间）来调节生成长度，而不修改原始的偏好优化目标函数本身。**

- 冻结预训练主干（frozen backbone）
- 在 LoRA 子空间中进行 post-training，仅更新极小比例参数
- 将“低秩配置”（rank、模块位置、层数范围、训练步数）作为可设计变量，在满足效用约束的前提下最小化输出 token 成本

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | LOCUS |
|------|--------|-------|
| **目标一致性** | 修改损失函数（加长度正则项）或提示 | 完全保留原生 preference objective（如 DPO） |
| **参数效率** | 全参数微调（full fine-tuning） | 仅更新 **0.24–0.28%** 参数 |
| **简洁性机制** | 外部干预（prompting, reward shaping） | 内在参数化效应（subspace parameterization） |
| **部署兼容性** | — | 支持权重合并（merge equivalence），零运行时开销 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **主数据集**：Anthropic HH-RLHF（Helpful and Harmless）  
  - 包含多轮对话的偏好对（chosen vs. rejected）
  - 主要用于 DPO / DrDPO / SamPO 的训练与评估
- **辅助数据集**（用于跨任务泛化测试）：
  - Anthropic HH-RLHF harmless-base（安全拒绝场景）
  - Orca DPO（单轮指令跟随任务）

### ⚙️ 实验设置
- **模型主干**：
  - `Pythia-2.8B`（基于 GPT-NeoX 架构）
  - `Qwen2.5-3B`（现代 Llama 风格架构，含 SwiGLU、RMSNorm、grouped-query attention）
- **训练协议**：
  - 所有比较均采用 **protocol-matched 设置**：共享初始 checkpoint、相同 objective、相同数据划分
  - LoRA 配置：rank $ r=16 $，缩放因子 $ \alpha=32 $
  - 仅在注意力模块（attention projections）上应用 LoRA
- **评估指标**：
  - **Continuation Token Count**：去除 EOS 后的平均生成 token 数
  - **Preference Accuracy**：在 held-out 测试集上，模型对 chosen 序列赋予更高 log-prob 的比例（内部诊断指标）
  - **Accuracy Change**：相对于 baseline 的准确率变化（容忍 ≤1.0 pp 下降）

### 🆚 基线方法对比
| 方法 | 类型 | 是否修改目标函数 | 参数更新比例 |
|------|------|------------------|-------------|
| Full-parameter DPO / DrDPO | 全参微调 | 否（原生 objective） | 100% |
| SamPO | 下采样 KL 散度优化 | 是（引入长度归一化） | 100%（官方 checkpoint） |
| LOCUS（本文） | 低秩适配 | **否**（完全保留原 objective） | **0.24–0.28%** |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（来自 Table 4 和 Figure 1）

| 方法 + Backbone | 输出 token（baseline → LOCUS） | 减少幅度 | Preference Accuracy 变化 | 更新参数比例 |
|----------------|-------------------------------|----------|----------------------------|--------------|
| **DPO / Pythia-2.8B** | 137.67 → 109.12 | **↓20.73%** | 48.40% → 48.39% (**−0.01 pp**) | 0.28% |
| **DrDPO / Pythia-2.8B** | 145.61 → 108.79 | **↓25.29%** | 48.39% → 48.26% (**−0.13 pp**) | 0.28% |
| **SamPO / Pythia-2.8B** | 132.77 → 79.88 | **↓39.84%** | 53.52% → 53.52% (**±0.00 pp**) | 0.28% |
| **DPO / Qwen2.5-3B** | 108.26 → 92.16 | **↓14.87%** | 48.69% → 48.64% (**−0.05 pp**) | 0.24% |
| **DrDPO / Qwen2.5-3B** | 111.58 → 91.97 | **↓17.58%** | 48.64% → 48.55% (**−0.09 pp**) | 0.24% |

> ✅ 所有实验均满足效用约束（accuracy drop ≤1.0 pp），且显著减少 token 使用。

### 🔬 消融实验结果

#### （1）**Rank 敏感性分析**（Figure 5）
- 在 Pythia-2.8B 上测试不同 LoRA rank 对 token 减少的影响（SamPO objective）：
  - $ r=4 $：token **增加 11.74%**（过拟合？）
  - $ r=8 $：减少 2.32%
  - $ r=16 $：减少 15.95%
  - $ r=32 $：减少 **25.10%**
- 表明：**更高的 rank 能更有效地压缩生成长度**，但需权衡参数量。

#### （2）**Target Module Ablation**（Figure 6）
比较不同模块组合的效果（固定 $ r=16 $）：
| 模块组 | Token 减少量 | 参数数量 |
|--------|---------------|-----------|
| QKV only | 8.09% | 5.24M |
| Output only | 4.96% | 2.62M |
| Attention-All（QKV + Output） | **15.95%** | 7.86M |
| MLP only | 5.75% | 13.11M |
| All-Linear | 13.88% | 20.97M |

> ✅ **Attention 模块上的低秩更新最为高效**，以最少参数实现最大 token 压缩。

#### （3）**Cross-Task Generality**（Figure 7）
在非 HH 对话任务上验证泛化性：
| 数据集 | Token 减少 | Accuracy 变化 |
|--------|------------|----------------|
| HH Dialogue | ↓20.73% | −0.01 pp |
| Harmless（Safety） | ↓25.29% | **+1.17 pp** |
| Orca DPO（Instruction） | **↓79.97%** | **+13.67 pp** |

> 💡 惊人发现：在某些任务（如指令跟随）上，LOCUS 不仅大幅缩短输出，还**提升了 preference accuracy**，说明其可能改善了模型聚焦关键信息的能力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **参数化方式本身可以调控生成长度**：即使不改变 loss 函数，仅通过限制更新到特定低秩 subspace，就能显著减少输出 token。
2. **LOCUS 实现“无损压缩”**：在几乎不变甚至略有提升 preference accuracy 的前提下，将生成长度减少 **15–40%**。
3. **高度参数高效**：仅需更新 **<0.3%** 参数即可达到媲美全参微调的效果。
4. **支持部署优化**：可通过 weight merging 实现零运行时开销，适用于生产环境。
5. **具有跨架构迁移性**：在 Pythia 与 Qwen 两种不同架构上均有效。

### ⚠️ 局限性
- 当前评估集中在约 3B 规模的 decoder-only 模型，未覆盖更大模型（如 7B+）或 encoder-decoder 结构。
- 使用 greedy decoding，未测试 sampling-based 推理下的表现。
- 候选池（candidate grid）较粗粒度（离散 rank、模块分组），未探索连续 rank 分配或梯度驱动剪枝。
- 跨任务评估为开发集结果，不能保证在未知任务上的推广能力。

### 🔮 未来工作方向
- 探索自适应 rank allocation 或 gradient-informed subspace selection
- 将 LOCUS 思路扩展至 vision-language 模型或多模态生成
- 研究 subspace 如何隐式学习“语义密度增强”的表示机制
- 结合 speculative decoding 或 KV cache 优化进一步提升推理效率

---

## ✅ 总结一句话
> **LOCUS 揭示了一个重要洞见：post-training 的参数化路径本身就是控制生成长度的强大杠杆——无需改动目标函数，只需聪明地选择低秩子空间，就能实现高保真、低开销的语言生成。**

</details>

---

### 8. [M3-Former: Multimodal Transformer with Mixture-of-Experts for Long-Term Vessel Trajectory Prediction](https://arxiv.org/abs/2609.10559)

**Authors**: Wenzhe Jin, Haina Tang  
**Category**: cs.LG  
**Published**: 2026-09-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.10559v1  

#### Abstract
To address the challenges of behavioral multimodality, limited semantic utilization, and long-term error accumulation in vessel trajectory prediction, this paper proposes M3-Former, a multimodal trajectory prediction framework enhanced by large language models (LLMs). The proposed framework incorpor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《M3-Former: Multimodal Transformer with Mixture-of-Experts for Long-Term Vessel Trajectory Prediction》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**长期船舶轨迹预测**中的三大挑战：
- **行为多模态性**（behavioral multimodality）：同一历史轨迹可能对应多个合理未来路径。
- **语义信息利用不足**：传统方法过度依赖动态轨迹特征，忽视静态属性（如船型、目的地）所蕴含的导航意图。
- **长期误差累积**：短期微小偏差在长时间预测中被放大，导致轨迹漂移。

### 提出的新方法与创新思路
作者提出 **M3-Former**（Multi-modal, Multi-scale, Mixture-of-Experts Former），其核心创新包括：

- **语义引导的分层建模框架**  
  将船舶静态属性（如 `Destination`, `ShipType`, `Length` 等）通过 **Large Language Model (LLM)** 编码为高阶语义表示，并与动态轨迹特征在统一的 **Transformer** 架构中融合，形成“全局意图指导局部动作”的预测范式。

- **双粒度 Mixture-of-Experts (MoE) 架构**
  - **Per-Sequence MoE**：在序列级别建模全局航行趋势（如航线选择、港口减速等），提供宏观约束。
  - **Per-Token MoE**：在时间步级别精细调整局部运动行为（如转向、变速），增强对动态变化的响应能力。
  - 采用 **MoE-S→T** 结构（先序列后token），符合“先规划路线再微调操作”的实际航海逻辑。

- **Steering-Weighted Cross-Entropy (SWCE) 损失函数**
  针对转向样本稀疏的长尾分布问题，设计加权损失机制，提升模型对关键操纵行为（如转弯、避让）的敏感度。

### 相比现有方法的优势
- 显著优于传统模型（CV, KF）和深度学习基线（GRU, TCN, TrAISformer）。
- 在长时域（4小时）预测中仍保持低误差增长率，有效缓解轨迹漂移。
- 更好地捕捉多模态行为，在航道分叉等复杂场景下能生成符合物理与语义约束的多样化路径。

---

## 2. 核心实验方法和设置

### 数据集
- 使用来自 **Danish Maritime Authority (DMA)** 的真实世界 AIS 数据集。
- 时间范围：2023年1月1日 至 3月31日。
- 地理覆盖：丹麦水域（见图2），包含复杂航道、港口与交通密集区。
- 数据规模：
  - 训练样本：1,923,380 条
  - 船舶数量：4,089 艘
  - 轨迹段数：26,374 段
  - 唯一目的地：4,407 个
- 特征类型：
  - **动态特征**：Latitude, Longitude, SOG, COG
  - **静态特征**：MMSI, Ship Type, Length, Width, Draught, Destination, ETA

### 实验设置
- **输入长度**：36 个时间步（每步10分钟 → 输入6小时）
- **预测范围**：24 个时间步（即 1h 到 4h 不同预测窗口）
- **模型架构**：
  - Transformer 主干：8 层，8 头注意力，embedding 维度 768
  - MoE 设置：4 个专家，Top-1 路由，负载均衡权重 $1\times10^{-4}$
  - LLM 编码器：冻结的预训练 LLaMA 模型
- **训练配置**：
  - 优化器：Adam ($lr=1e-4$)
  - 批大小：256
  - 训练轮次：最多10轮，早停策略（patience=5）
  - 硬件：4×NVIDIA A100 GPU

### 评估指标
- **Average Displacement Error (ADE)**：预测轨迹与真实轨迹之间的平均欧氏距离。
- **Final Displacement Error (FDE)**：最终位置的预测误差。
- 公式：
  $$
  \text{ADE} = \frac{1}{T}\sum_{t=1}^{T} \| \hat{p}_t - p_t \|_2,\quad \text{FDE} = \| \hat{p}_T - p_T \|_2
  $$

### 对比的基线方法
| 类型 | 方法 |
|------|------|
| 传统模型 | CV (Constant Velocity), KF (Kalman Filter) |
| 序列模型 | Seq2Seq, GRU |
| 卷积模型 | TCN |
| Transformer 模型 | TrAISformer |

---

## 3. 主要实验结果和性能指标

### 定量性能对比（4小时预测）

| Model | ADE ↓ | FDE ↓ |
|-------|--------|--------|
| CV | 18.7701 | 40.1790 |
| KF | 18.5298 | 39.6351 |
| GRU | 9.2527 | 17.3268 |
| TCN | 9.4732 | 17.7865 |
| TrAISformer | **6.7732** | **14.3445** |
| **M3-Former (Ours)** | **6.4727** (-4.4%) | **13.6142** (-5.1%) |

> ✅ **M3-Former 在所有预测时域均取得最优表现**，尤其在4小时任务中相对最强基线 TrAISformer 提升显著。

#### 关键观察：
- M3-Former 的 ADE 增长更平缓（从1.55到6.47），而 GRU 从3.80飙升至9.25，说明其**长期稳定性更强**。
- 错误增长率降低验证了**语义先验可有效抑制轨迹漂移**。

### 消融实验结果（Table III）

| 配置 | ADE | FDE |
|-------------------------------|---------|---------|
| Full Model (MoE + Multimodal + TurnLoss) | **6.4727** | **13.6142** |
| - TurnLoss | 6.5665 | 13.7796 |
| - MoE | 6.6371 | 13.9431 |
| - Multimodal | 6.9768 | 14.9049 |
| - MoE & Multimodal | 6.9037 | 14.6456 |
| None (仅基础结构) | 6.9037 | 14.6456 |

#### 消融分析结论：
- 移除任一模块都会导致性能下降，三者具有**协同增益效应**。
- **Multimodal 融合贡献最大**（ADE↑0.5），证明 LLM 提供的语义先验是长期预测的关键。
- **MoE 结构提升建模灵活性**，特别是在复杂水道和分支路径中表现稳健。
- **TurnLoss 显著改善转弯预测精度**，解决了数据不平衡带来的偏置问题。

### MoE 结构粒度分析（Fig. 5）
- **MoE-S→T**（先序列后token）效果最佳，重要性得分最高（0.587），且与验证损失呈负相关（-0.342），表明其既能捕获全局趋势又能精准调整局部行为。
- **MoE-T→S** 反向结构虽重要但正相关（+0.339），易受局部噪声干扰，影响整体稳定性。

---

## 4. 关键结论和发现

### 主要结论
1. **语义信息对长期预测至关重要**：  
   静态属性（尤其是 Destination）作为导航意图的代理，能显著减少轨迹发散，提升终点预测准确性。

2. **双粒度 MoE 实现高效分层建模**：  
   “Global Planning + Local Adjustment” 的架构设计更贴合真实航海行为逻辑，优于单一尺度建模。

3. **多模态融合 + MoE + 加权损失 协同作用**：  
   三者共同构建了一个鲁棒、准确、适应复杂的长期轨迹预测系统。

4. **M3-Former 在真实 AIS 数据上全面领先**：  
   不仅在数值指标上超越 SOTA，定性分析也显示其在**急转弯、近岸航行、航道分叉**等挑战性场景中生成更合理、多样化的轨迹。

### 方法的局限性
- 当前模型依赖 AIS 中提供的 Destination 和 ETA 字段，但在现实中这些字段常为空或不准确。
- 未显式建模外部环境因素（如风浪、潮汐、交通密度），限制了极端条件下的泛化能力。
- MoE 引入额外参数和计算开销，尽管可控，但仍需进一步优化以支持实时部署。

### 未来工作方向
1. **引入更多环境上下文**：集成气象、海况、电子海图（ECDIS）、VTS 指令等多源信息。
2. **轻量化与高效推理设计**：探索稀疏激活、知识蒸馏等技术，推动模型在边缘设备上的应用。
3. **跨区域迁移学习与领域自适应**：提升模型在不同海域、船种、交通模式下的通用性。
4. **端到端不确定性建模**：结合贝叶斯方法或扩散模型，输出更具解释性的概率轨迹集合。

---

> 🔗 **代码已开源**：https://github.com/zophykim/M3former

</details>

---

### 9. [Zero-shot rib design: merging training-free generative prior with topology optimization](https://arxiv.org/abs/2609.10643)

**Authors**: Yongmin Kwon, Namwoo Kang  
**Category**: cs.LG  
**Published**: 2026-09-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.10643v1  

#### Abstract
Natural load-bearing patterns such as leaf venation, trabecular bone, and spider webs achieve high stiffness per unit mass, yet classical topology optimizers rarely reach such geometries, and few let engineers express structural design intent through natural language. This work treats a frozen text-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **topology optimization**（拓扑优化）方法（如 **SIMP**）在设计肋板（rib reinforcement）时面临两大挑战：
1. **局部最优陷阱**：由于目标函数非凸，梯度优化易陷入局部最优，难以探索高效的自然启发式结构（如叶脉、骨小梁）。
2. **设计意图表达困难**：工程师无法通过自然语言等直观方式引导优化过程，缺乏将先验知识（如“类似蜘蛛网”）融入物理优化循环的有效机制。

### 提出的新方法与思路
本文提出一种**零样本肋板设计框架**（zero-shot rib design），其核心是将一个冻结的文本到图像扩散模型（如 **Stable Diffusion 2.1**）作为**无训练生成先验**（training-free generative prior），通过 **Score Distillation Sampling**（SDS） 技术将其与基于密度的拓扑优化（SIMP）在梯度层面耦合。

- **核心思想**：将自然语言提示词（text prompt）视为工程师设计意图的显式、机器可解释表示。该提示词通过 SDS 产生一个“生成梯度”（generative gradient），并与有限元分析（FEA）产生的“物理梯度”（physics gradient）在每次迭代中进行加权融合。
- **关键创新**：实现了**梯度级耦合**（gradient-level coupling），即生成先验不是独立生成设计，而是直接参与物理优化的每一步更新，由 FEA 敏感性决定哪些由提示诱导的特征能被保留。

### 相比现有方法的优势
| 方面 | 传统数据驱动方法（如 TopoDiff, DOM） | 本方法 |
| :--- | :--- | :--- |
| **训练需求** | 需要大量特定领域的拓扑优化解数据集（数万样本） | **零训练**（zero-shot），无需任何任务相关训练数据 |
| **泛化能力** | 在训练分布外（新边界条件、新几何域）表现差，需重新训练 | **高适应性**，仅通过更改提示词即可迁移到新领域，无需重训 |
| **物理一致性** | 生成的设计可能视觉上合理但物理次优，因物理约束主要在训练时嵌入 | **强物理保证**，物理梯度全程主导，最终收敛于物理一致的最优解 |
| **设计意图表达** | 依赖预定义模板或隐式学习，难以灵活表达复杂意图 | **灵活表达**，通过自然语言提示词直接注入设计知识 |

---

## 2. 核心实验方法和设置

### 数据集
本研究为**无训练**（training-free）方法，**未使用任何拓扑优化数据集**。它依赖于公开预训练的 **Stable Diffusion 2.1** 模型，该模型在 LAION-5B 数据集上训练，包含了数十亿图文对。

### 实验设置
- **几何域**（Domains）：共4个，涵盖合成基准与工业实例。
  - `DRect`：矩形域，四角固定，中心点载荷。
  - `DCircle`：圆形域，四向固定，中心点载荷。
  - `DHole`：带中心孔的矩形域，四边中心受载。
  - `DLink`：工业级汽车悬架连杆横截面，平面应力分析。
- **物理场**（Physics Regimes）：
  - **机械弯曲**（mechanical bending）：Mindlin-Reissner 板理论。
  - **热弹性弯曲**（thermoelastic bending）：均匀温差引起的热弯矩。
- **文本提示**（Prompts）：共10个，分为三类：
  - *生物仿生*：Tree, Leaf, Bone
  - *工程结构*：Truss, Grid, Honeycomb
  - *几何图案*：Voronoi, Spiderweb, Ornamental, Diamond
  - 所有提示均添加前缀 `"bold black lines on white background"` 以匹配扩散模型的训练风格。
- **评估指标**：
  - **柔度**（Compliance, C）：主指标，越低越好。
  - **统计显著性**：采用单侧二项检验，若5个随机种子全部优于基线，则 $p=0.031$，视为显著提升。
  - **效应量**：Cohen's $d$，衡量实际效果大小。

### 基线方法对比
- **SIMP Baseline**：标准的 Adam 优化器 + Heaviside 投影。
- **Multi-start SIMP**：从50个不同初始密度场启动优化，取最佳结果。
- **Perturbation SIMP**：定期向密度场注入高斯噪声。
- **Image-guided TO**：使用 SDS 生成的静态图像作为目标，用 MSE 损失引导优化。
- **MMA Optimizer**：作为强梯度基线，与 Adam 对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **245 次主要 SDS 运行**中，覆盖4个几何域和2种物理场。
- **38 / 49** 个提示-域组合实现了**统计显著的柔度降低**。
- 最大柔度降低：
  - **机械场**：**-31.5%** （`DHole` 域，`Spiderweb` 提示）
  - **热弹性场**：**-23.0%** （`DRect` 域，`Spiderweb` 提示）
- **效应量**：在显著组合中，中位 Cohen's $|d| = 1.47$，表明具有**大型实际效应**。

### 与基线方法的对比结果
| 基线方法 | 性能对比 |
| :--- | :--- |
| **Multi-start SIMP** | SDS 的最佳结果在所有域均**优于**多起点搜索的最佳结果，尤其在 `DHole` 域差距最大。 |
| **Perturbation SIMP** | 随机扰动平均表现与基线相当或更差，证明**结构化引导**（而非随机探索）是关键。 |
| **Image-guided TO** | 固定图像引导法部分恢复了收益，但仍**远逊于自适应 SDS**，证明动态梯度耦合至关重要。 |
| **MMA Optimizer** | 尽管 MMA 是更强的单一起点基线，但在 `DCircle` 和 `DHole` 域，SDS 仍能取得更低的柔度，证明其能到达梯度下降无法触及的拓扑盆地。 |

### 消融实验结果
- **调度策略**（Scheduling Ablation）：
  - **梯度权重**（$\lambda_{sds}$）：存在 U 形关系，过小或过大均导致性能下降，验证了平衡物理与生成信号的重要性。
  - **时间步退火**（t-annealing）：移除后性能下降 **+13.0%**，证明“粗到细”的引导策略有效。
  - **EMA 平滑**：移除后性能下降 **+5.0%**，显示平滑对稳定性至关重要。
- **Heaviside 投影**：
  - 无投影时，中间密度比例高达 **42.6%**。
  - 采用 $\beta$-continuation 后，中间密度降至 **<3%**，解决了扩散模型连续空间偏好与二值化设计需求的冲突。

---

## 4. 关键结论和发现

### 主要发现
1. **核心改进机制**：SDS 引导的性能提升主要源于**死端抑制**（dead-end suppression）。即，生成先验倾向于消除不承载载荷的“死胡同”分支，从而释放材料预算用于加强主承力路径。跨域形态学分析显示，柔度与骨架端点数呈强正相关（Pearson $r = +0.56$ 至 $+0.99$）。
2. **成功的关键是负载路径对齐**（load-path alignment）：提示词诱导的模式必须与域内的实际负载路径几何对齐才能有效。例如，`Spiderweb` 在径向加载的 `DRect` 中表现优异，但在分布式加载的 `DLink` 中表现不佳；而 `Voronoi` 则表现出更好的通用性。
3. **框架的普适性**：该方法不仅在机械场有效，在热弹性场也取得了显著提升，并且成功推广到了真实的工业部件 `DLink`。

### 方法的局限性
1. **维度限制**：当前框架为 **2D**，将 3D 密度场与 2D 扩散先验耦合存在维度不匹配问题。
2. **提示选择依赖人工**：没有单一提示适用于所有场景，需要根据具体问题手动筛选合适的提示词。
3. **计算开销**：相比纯 SIMP，引入 SDS 带来了约 7% 的额外计算开销。

### 未来工作方向
1. **扩展至 3D**：通过多视角 SDS 或原生 3D 扩散先验（如 Shap-E）实现三维设计。
2. **自动化提示发现**：在连续的文本嵌入空间中进行优化（如 textual inversion），自动寻找最优提示。
3. **探索更多物理场**：应用于流固耦合、光子晶体设计等局部最优陷阱严重的领域。
4. **构建提示筛选协议**：开发基于早期收敛信号的自动化提示筛选流程，降低部署成本。

</details>

---

### 10. [From Connectivity to Rewards: Dense Reward Learning with Directed State Graphs](https://arxiv.org/abs/2609.10781)

**Authors**: Shuyuan Zhang, Zihan Wang, Xiao-Wen Chang, Doina Precup  
**Category**: cs.LG  
**Published**: 2026-09-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.10781v1  

#### Abstract
The integration of graphs with Goal-Conditioned Hierarchical Reinforcement Learning (GCHRL) has received increasing attention, as graphs naturally encode task hierarchies for effective subgoal sampling. However, existing methods often overlook intrinsic connectivity information, failing to fully lev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Connectivity to Rewards: Dense Reward Learning with Directed State Graphs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对 **Goal-Conditioned Hierarchical Reinforcement Learning (GCHRL)** 中存在的两个关键挑战：

1. **稀疏奖励（sparse rewards）** 导致探索效率低下、信用分配困难；
2. 现有图结构方法大多使用**无向状态图（undirected state graphs）**，无法有效建模环境中普遍存在的**非对称转移（asymmetric transitions）**，即从状态 A 到 B 容易，但从 B 到 A 困难（如悬崖、单向门等），这类环境被称为 **quasimetric environments**。

此外，大多数图方法将图仅用作子目标采样工具，而未将其作为编码状态连通性的**环境模型（environmental model）** 来生成密集奖励信号。

---

### 🚀 提出的新方法：G2QDR
作者提出了一种名为 **Graph-Guided Quasimetric Dense Reward (G2QDR)** 的框架，其核心思想是：

- 在线构建一个**有向状态图（directed state graph）**，记录探索过程中状态之间的连接关系；
- 基于该图训练一个**状态连通性模型（state connectivity model）**，预测任意两状态间的“可达性强度”；
- 将这种连通性转化为**辅助密集奖励（auxiliary dense rewards）**，用于指导高层策略（high-level policy）选择更易达的子目标，并帮助低层策略（low-level policy）学习更合理的路径。

#### 主要创新点：
| 创新维度 | 内容说明 |
|--------|--------|
| **图结构设计** | 构建**有向加权图**，边权重反映转移频率与时序衰减，显式建模方向性差异 |
| **连通性建模** | 引入神经网络 `Cθ` 学习状态对 `(su, sv)` 的连通性得分，支持泛化到未见状态 |
| **奖励机制设计** | 设计三种类型的辅助奖励：<br>• 高层约束奖励（encourage reachable subgoals）<br>• 低层修正奖励（replace Euclidean distance with connectivity-based signal）<br>• 不对称惩罚项（penalize irreversible transitions） |
| **动态调度机制** | 引入阶段依赖的 `λ` 控制器，在训练初期逐步引入密集奖励，在后期逐渐退火以避免改变原始任务最优策略 |

---

### 🔍 相比现有方法的优势
| 对比维度 | G2QDR vs. 现有方法 |
|--------|------------------|
| **与传统 GCHRL 方法相比** | 显著提升样本效率，通过引入基于连通性的密集奖励加速学习过程 |
| **与无向图方法（如 G4RL）相比** | 在具有明显非对称动态的环境中表现更好，能捕捉方向性信息 |
| **通用兼容性** | 可无缝集成进任何现有的 GCHRL 架构（如 HIRO, HRAC, HESS, HLPS） |
| **无需专家先验** | 图在线构建，不依赖手工设计或预定义拓扑结构 |

---

## 2. 核心实验方法和设置

### 🧪 数据集与环境
所有实验在 **MuJoCo** 连续控制任务上进行，具体包括以下五个稀疏奖励环境：

| 环境 | 特点 |
|------|------|
| **AntMaze (U/W-shape)** | 复杂迷宫导航，墙体阻隔导致欧氏距离失效；基本对称 |
| **AntGather** | 收集苹果避开炸弹，靠近墙后物体难以拉回 → 非对称 |
| **AntPush** | 推动方块完成任务，推入角落则不可逆 → 强非对称 |
| **AntFall** | 跨越深坑需搭桥，掉落即失败 → 单向陷阱 |
| **Pusher** | 机械臂操控物体，超出工作空间后难以恢复 → 结构性非对称 |

> ✅ 环境按非对称程度排序：AntMaze < Pusher < AntFall < AntGather < AntPush

---

### ⚙️ 实验设置
- **训练总步数**：20,000 episodes
- **评估方式**：每 1,000 轮评估一次，每次运行 100 trials
- **评估指标**：**Success Rate (%)**（任务成功比例）
- **重复次数**：所有结果取 10 次独立运行的均值 ± 标准差
- **骨干算法测试**：HIRO, HRAC, HESS, HLPS 四种主流 GCHRL 方法
- **对比基线**：
  - 原始方法（vanilla）
  - +G4RL（基于无向图的连通性估计）
  - +G2QDR（本文方法，含 w/o PT 和 w/ PT 两种变体）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 方法 | AntMaze-U | AntPush | AntFall | Pusher | AntGather (#obj) |
|------|-----------|---------|---------|--------|------------------|
| HIRO | 0.72±0.09 | 0.05±0.01 | 0.23±0.04 | 0.17±0.03 | 1.47±0.18 |
| HIRO + G4RL | 0.82±0.07 | 0.14±0.02 | 0.28±0.04 | 0.21±0.02 | 1.78±0.10 |
| HIRO + G2QDR w/o PT | **0.87±0.08** | 0.12±0.01 | 0.35±0.03 | 0.26±0.03 | 1.90±0.11 |
| HIRO + G2QDR w/ PT | 0.80±0.04 | **0.19±0.04** | **0.31±0.03** | **0.30±0.03** | **1.96±0.13** |

> ✅ 观察：在非对称性强的任务（如 AntPush, Pusher）中，**带惩罚项（PT）的 G2QDR 表现最佳**

---

### 🔁 与其他方法的整体比较
- 在所有四种骨干方法（HIRO/HRAC/HESS/HLPS）上，**+G2QDR 均优于原始版本和 +G4RL**；
- 提升幅度在**高非对称环境**中尤为显著（如 Pusher 上成功率翻倍）；
- **G2QDR 在 AntMaze 上也有增益**，尽管该环境对称，说明连通性建模本身有助于克服局部障碍；
- **+G2QDR w/ PT** 在 Pusher、AntPush 等任务中表现最好，验证了不对称惩罚的有效性。

---

### 🔍 消融实验结果（Table 2）

| 变体 | 描述 | 性能趋势 |
|------|------|--------|
| **a** | 仅高层约束奖励 | 有一定提升，但不如组合方案 |
| **b** | 仅低层修正奖励 | 在 AntMaze 类任务中有效，改善局部策略精度 |
| **c** | 高层 + 低层奖励（无惩罚） | 多数情况下表现最强 |
| **d** | 完整 G2QDR（含惩罚项） | 在非对称任务中进一步提升，在对称任务中可能轻微下降 |

> ✅ 结论：**高层约束 + 低层修正是核心驱动力**；**惩罚项适用于非对称环境，但需谨慎使用**

---

### ⏱️ 效率与开销分析
- **计算开销增加约 1.4–1.9x**（相对于原方法），主要来自图维护与节点比较；
- 通过降低状态采样频率（如每 10 步更新一次图）可显著减少耗时，性能损失极小；
- 减少训练数据量至 75% 对性能影响不大，表明模型鲁棒性强。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **有向图建模显著优于无向图**，特别是在存在非对称转移的环境中；
2. **将状态连通性转化为密集奖励信号** 是一种高效的信息利用方式，可大幅提升 GCHRL 的学习效率；
3. **高层与低层协同优化效果最佳**：高层引导可行子目标，低层依据真实连通性调整行为；
4. **不对称惩罚机制** 能有效防止陷入“易进难出”的危险区域，尤其在 Pusher、AntPush 等任务中至关重要；
5. **G2QDR 具备良好通用性**，可插拔式集成进多种 GCHRL 框架，且无需修改主干网络结构。

---

### ⚠️ 局限性（Limitations）
1. **超参数敏感**：`ed`, `W`, `p`, `αh`, `αl` 等需精细调参，影响稳定性；
2. **图构建受策略偏差影响**：探索策略决定了图覆盖范围，可能导致结构偏倚；
3. **非势能奖励（non-potential-based reward）** 改变了原始 MDP 的最优策略，虽通过 `λ` 调度缓解，但仍存在风险；
4. **缺乏真实连通性标签**：在连续高维空间中无法直接评估图的质量，只能间接通过下游任务判断；
5. **固定节点数限制表达能力**：当环境复杂度上升时，有限图容量可能成为瓶颈。

---

### 🔮 未来工作方向
1. **自适应超参数调节机制**：根据环境动态自动调整 `ed`, `W`, `λ` 等；
2. **去偏图构建策略**：结合主动探索或覆盖率最大化原则改进图生长过程；
3. **势能型奖励设计**：使辅助奖励保持策略不变性，避免改变最优解；
4. **扩展至更大规模环境**：测试在视觉输入、部分可观测、多智能体场景下的有效性；
5. **图质量评估方法**：开发可解释的指标来衡量学习到的状态连通性是否准确。

---

## 总结

> **G2QDR 成功地将状态图从“采样工具”升级为“环境模型”**，并通过学习**有向连通性**生成**多层次密集奖励**，解决了 GCHRL 在稀疏奖励与非对称环境中的核心难题。其实验充分验证了**方向性建模的重要性**，并为未来基于图结构的 HRL 方法提供了新的范式。

</details>

---

### 11. [ExaServe: Large-Scale LLM Serving on Exascale HPC Systems](https://arxiv.org/abs/2609.10812)

**Authors**: Wenyi Wang, Shu Shi, Yadu Babuji, Ian Foster, Kyle Chard  
**Category**: cs.DC  
**Published**: 2026-09-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.10812v1  

#### Abstract
Cloud-native LLM serving frameworks have made deployment routine in data centers, yet deploying them on leadership-class supercomputers remains an engineering challenge requiring scheduler integration, MPI launch, accelerator selection, node-local weight staging, and platform-specific patches. We pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《ExaServe: Large-Scale LLM Serving on Exascale HPC Systems》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大型语言模型（LLM）推理服务框架（如 vLLM、Ray Serve）主要为云环境设计，依赖 Kubernetes、外部负载均衡器等设施，难以直接部署在 **领导级高性能计算系统**（HPC, High-Performance Computing）上。HPC 系统通常采用批处理调度（如 PBS）、MPI 启动机制、共享文件系统，并且硬件异构（如 Intel XPU），缺乏现成的 LLM 服务支持。

因此，将 LLM 部署到 HPC 上面临以下挑战：
- 缺乏统一的部署编排工具
- 框架默认配置不适用于大规模 HPC 场景
- 对非 CUDA 加速器（如 Intel GPU）支持不足
- 节点拓扑复杂（多 IP、NUMA 结构、临时存储）
- 控制平面在超大规模下出现瓶颈

### 提出的新方法：ExaServe
作者提出 **ExaServe**，一个专为 **Exascale HPC 系统** 设计的大规模 LLM 推理服务框架，其核心是一个可通过 `pip install` 安装的 Python 工具，能够将声明式的 YAML 配置自动转化为可复现的大规模 LLM 服务部署。

#### 主要创新点：
1. **端到端自动化部署流程**
   - 支持从登录节点提交作业 → 批量调度（PBS）→ MPI 启动 Ray → 权重预加载至节点本地 tmpfs → 自动启动前端代理（如 HAProxy）的完整生命周期管理。
   
2. **声明式部署接口（Declarative YAML Spec）**
   - 用户只需编写一份 YAML 文件，即可定义模型、节点数、并行策略、代理类型等，实现“一键部署”。

3. **兼容层（Compatibility Layer）**
   - 针对特定平台（如 ALCF Aurora）提供运行时补丁（site-specific patches），解决：
     - Intel XPU 多进程问题
     - oneAPI 设备选择
     - HuggingFace Tokenizer 线程池爆炸
     - Ray Serve 在大规模下的健康检查误杀问题

4. **双层补丁系统（Two-tier Patch System）**
   - **Set A**：替换 Ray Serve 内部文件，延长健康检查和请求超时时间（例如从 1 分钟提升至 1 小时），防止控制器误杀正常但响应慢的代理。
   - **Set B**：注入运行时修复代码，修正 vLLM 在 XPU 上的设备绑定、流水线并行等问题。

5. **分片感知的流水线并行（Shard-aware Pipeline Parallelism）**
   - 对于跨节点的大模型（如 405B 参数），仅将每个节点所需的权重分片广播到对应节点的本地存储，避免全量广播导致内存溢出。

### 相比现有方法的优势
| 维度 | 传统方法（手动部署） | ExaServe |
|------|------------------------|---------|
| 可复现性 | 差，依赖人工操作 | 高，YAML 驱动，完全自动化 |
| 易用性 | 极低，需熟悉 MPI、Ray、HPC 调度 | 高，`pip install` + YAML 即可部署 |
| 规模扩展性 | 受限于手动调试能力 | 支持 1–256 节点自动部署 |
| 平台适配性 | 需大量定制脚本 | 插件化兼容层，易于移植到其他 HPC 系统 |
| 性能优化 | 无系统性调优 | 内建最佳实践（如权重本地缓存、控制面调参） |

---

## 2. 核心实验方法和设置

### 实验平台
- **系统**：ALCF Aurora 超算系统
- **节点配置**：
  - 每节点：2×Intel Xeon Max CPU，6×Intel Data Center GPU Max（Ponte Vecchio, PVC），共 12 个逻辑 XPU Tile
  - 网络：HPE Slingshot 11 高速网络
- **软件栈**：
  - Python 3.12, Ray 2.53.0（XPU 版本）, vLLM 0.15.0, oneAPI

### 数据集与工作负载
- **数据集**：ShareGPT（过滤后保留输入长度 ≥64 的样本）
- **输入长度**：截断为 64 个 content tokens，加上模板共约 74.7 输入 tokens
- **输出长度**：固定上限 64 tokens
- **请求模式**：
  - 固定间隔（Fixed-interval）：每节点 110 QPS（饱和压力测试）
  - 泊松到达（Poisson）
  - BurstGPT 迹象回放

### 评估指标
| 指标 | 定义 |
|------|------|
| **QPS**（Queries Per Second） | 成功完成的请求吞吐量 |
| **Tokens/s** | 输出 token 吞吐率 |
| **TTFT**（Time To First Token） | 客户端观察到首个 token 返回的时间 |
| **TBT**（Time Between Tokens） | 解码阶段连续 token 之间的延迟，使用 **P99 TBT ≤ 250ms** 作为 SLO |
| **SLO Attainment** | 满足 `TTFT ≤ 2s` 且 `P99 TBT ≤ 250ms` 的请求比例 |
| **Time-to-serve** | 从作业提交到第一个请求可服务的时间（含权重加载、Ray 启动、应用部署） |

### 基线方法对比
| 配置 | 描述 |
|------|------|
| **Direct Dispatch** | 无中心代理，客户端直连各节点 Ray Serve 端口（用于测量路由开销） |
| **HAProxy** | 中心化 L7 负载均衡，默认 leastconn 策略 |
| **Envoy / LiteLLM / Ray Serve ProxyActor** | 其他中心化代理方案 |
| **Ray Serve 原生配置** | 未打补丁的默认设置（用于展示稳定性问题） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 非流式推理（Non-streaming）接近线性扩展
- 使用 **HAProxy + vLLM**，在 256 节点（3072 replicas）上达到：
  - **27.1k QPS**（即 27,100 请求/秒）
  - **3.8M tokens/s** 输出吞吐
- 与 **Direct Dispatch** 基线几乎一致，说明代理引入的路由开销可忽略。

#### ❌ 流式推理（Streaming）在中心化代理下严重退化
- 尽管模型服务器本身表现良好（server-side TTFT ≈ 66ms, P99 TBT ≈ 28ms），但客户端观测到的性能急剧下降：
  - **HAProxy** 流式吞吐在 128 节点后趋于饱和，最高仅 **~4.7k QPS**
  - 在 256 节点时，SLO 达成率趋近于 0%
  - 客户端观测到的 P99 E2E 延迟高达 **17 秒**
- **根本原因**：所有 SSE（Server-Sent Events）连接汇聚到单个 head node，造成网络 I/O 瓶颈（TCP 重传从 331k 增至 708万），而代理 CPU 利用率不足 2%

#### ⚠️ Ray Serve 控制平面存在 O(N²) 瓶颈
- `serve.run` 阶段耗时随节点数平方增长：
  - 64 节点：154 秒
  - 256 节点：**1767 秒（约 29.5 分钟）**
- 根本原因是：每次新增副本组，控制器会向每个 proxy 发送未解析的 actor 名称，导致每个 proxy 都需通过 GCS 查询所有副本（N_proxies × N_replicas = O(N²) 查询）
- 补丁（Set A）虽能防止 proxy 被误杀，但无法消除该复杂度。

#### 🔁 512 节点部署失败
- 在 512 节点尝试中，由于 **GCS（Global Control Store）单点过载**，worker 断开连接，部署失败。
- 表明当前 Ray 架构的 **单 head node GCS 是硬性扩展上限**。

#### 🔄 多引擎与大模型验证
- **SGLang 引擎**：同样表现出流式吞吐崩溃现象，证明问题与推理引擎无关。
- **Llama-3.1-405B（PP=2）**：
  - 使用 shard-aware staging，在 128 pipeline-parallel replicas 下成功部署
  - 弱扩展效率达 **50%**（考虑测量窗口影响后可达 82–95%）
  - 时间主要消耗在权重加载而非控制面

### 消融实验结果（One-axis-at-a-time）

| 变量 | N=1 SLO 成功率 | N=64 SLO 成功率 | 变化 Δ |
|------|----------------|------------------|--------|
| Baseline (8B, 64/64) | 0.94 | 0.65 | -0.29 |
| Poisson 到达 | 0.92 | 0.60 | -0.32 |
| 2K/2K 上下文 | 1.00 | 0.98 | -0.02 |
| 4K/4K 上下文 | 1.00 | 1.00 | 0.00 |
| Code/Chat/Summary | ~1.00 | ~1.00 | ~0.00 |
| 120B 模型（TP=8） | 0.98 | 0.97 | -0.01 |

> **结论**：只有在高请求率 + 短上下文场景下才会因共享流路径竞争而导致 SLO 显著下降；长上下文或低速率任务仍可良好扩展。

---

## 4. 关键结论和发现

### 主要发现
1. **非流式推理可在 HPC 上高效扩展**
   - ExaServe 实现了 **27.1k QPS** 和 **3.8M tokens/s** 的吞吐，接近线性扩展，表明 HPC 硬件具备强大服务能力。

2. **流式推理的瓶颈不在模型服务器，而在中心化代理**
   - 模型侧性能优异（TTFT < 100ms, TBT < 30ms），但客户端体验极差，**延迟差距达 3.6×**。
   - 问题本质是 **Head Node Network Saturation**，即所有 token 流集中于单一入口。

3. **Ray Serve 存在 O(N²) 控制面通信瓶颈**
   - `serve.run` 时间从分钟级增长至半小时以上，成为大规模部署的主要延迟来源。
   - 当前补丁只能缓解稳定性问题，无法改变算法复杂度。

4. **ExaServe 提供了可复现的部署范式**
   - 通过声明式配置 + 自动化流程 + 兼容层，显著降低 HPC 上部署 LLM 的门槛。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖 MPI 和批处理模型** | 不适用于长期在线服务，更适合批量推理任务 |
| **Head Node 单点瓶颈** | 中心化代理限制了流式服务的扩展性 |
| **Ray 架构限制** | GCS 单点、O(N²) 查找等问题需上游修改才能根本解决 |
| **补丁维护成本** | Set A/B 补丁需随 Ray/vLLM 版本更新持续适配 |

### 未来工作方向
1. **去中心化代理架构**
   - 将代理分布到多个节点，避免 head node 成为瓶颈
   - 或采用 per-node endpoint + 客户端负载均衡

2. **改进 Ray 控制平面**
   - 推动 Ray 社区支持 **GCS 分片** 或 **handle diff 同步机制**，消除 O(N²) 开销

3. **支持更多 HPC 平台**
   - 将兼容层扩展至 Frontier（AMD GPU）、El Capitan（NVIDIA GPU）等系统

4. **集成更智能的调度策略**
   - 结合 DistServe、Sarathi-Serve 等技术，实现 Prefill/Decode 分离、动态批处理优化

5. **Fault Tolerance 增强**
   - 当前基于 MPI 的部署对节点故障敏感，未来可探索混合模式：Ray 控制面 + MPI 快路径

---

> **总结一句话**：  
> **ExaServe 成功实现了 LLM 在 Exascale HPC 上的大规模部署与高性能非流式推理，但也揭示了当前主流框架（Ray/vLLM）在控制面扩展性和流式服务架构上的根本性瓶颈，为下一代 HPC-AI 融合系统的设计提供了重要实证依据。**

</details>

---

### 12. [A Dataset and Model for Imputing Water Surface Elevation on a Large and Extremely Sparse Spatiotemporal Graph](https://arxiv.org/abs/2609.11580)

**Authors**: Ruben Cartuyvels, Karim Douch, Gabriele Bertoli, Mounia El Baz, Artemis Vrettou, S\'ebastien Lef\`evre, Diego Fernandez Prieto  
**Category**: cs.LG  
**Published**: 2026-09-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.11580v1  

#### Abstract
Continuous monitoring of water surface elevation across river networks is critical for flood forecasting, water resource management, and understanding the global water cycle. Yet, the scarcity of in situ gauges across much of the globe constrains the development of reliable modeling frameworks. Sate...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Dataset and Model for Imputing Water Surface Elevation on a Large and Extremely Sparse Spatiotemporal Graph

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对**全球河流水位监测中的极端稀疏时空观测问题**展开研究。由于地面流量计（in situ gauges）覆盖稀疏，而卫星测高（satellite altimetry）虽能提供全球覆盖，但时间采样极不连续（如SWOT为21天重复周期），导致难以实现高分辨率的水表面高程（Water Surface Elevation, WSE）重建。

现有 spatiotemporal graph imputation 方法在以下方面存在不足：
- 面对**超大规模图**（>19K节点）和**极高缺失率**（日均观测 <1%）时计算不可行；
- 多数方法假设图是稠密或循环结构（如道路网络），而**河流网络是定向无环图**（DAG），具有上游→下游的物理顺序；
- 图神经网络（GNN）在如此稀疏条件下无法有效传递消息。

### 提出的新方法与创新思路
作者提出两个核心贡献：

#### （1）发布新数据集：AmazonWSE
- 覆盖亚马逊流域约 **19.2K river reaches**（基于SWORD拓扑数据库）；
- 时间跨度：**2016–2026年**，每日时间步长；
- 整合多源卫星观测：**SWOT RiverSP**, **HydroWeb**, **ICESat-2**，以及用于评估的 **ANA in situ gauges**；
- 总体观测稀疏度高达 **99%**，远超现有基准（如METR-LA仅8%缺失）；
- 是首个面向**极端稀疏 + 大规模 + DAG拓扑**的WSE填补任务的数据集。

#### （2）提出新型序列模型：Bidirectional Selective State Space Model（基于Mamba）
- 将时空图填补任务转化为**序列建模问题**，将空间位置和时间戳扁平化为单一token序列；
- 引入**子图采样策略**（subgraph sampling），从锚点出发沿河流拓扑上下采样局部邻域；
- 设计**拓扑感知的位置编码**（topology-aware positional encodings）：
  - 包括 `TreePE` 编码分支路径；
  - 加入月份、传感器类型、坐标等元数据嵌入；
- 使用 **Mamba 架构**进行双向扫描（forward/backward），融合过去上游与未来下游信息；
- 支持**归纳学习**（inductive），可预测训练中未见的河段。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **可扩展性** | 不处理全图，避免内存爆炸；适用于 >10K 节点的大图 |
| **稀疏适应性** | 仅对实际观测生成token，避免大量虚拟填充 |
| **物理合理性** | 利用河流拓扑排序（时间+流向）作为输入顺序，符合水文动力学 |
| **泛化能力** | 归纳式设计支持新河段预测，无需重新训练 |

---

## 2. 核心实验方法和设置

### 数据集
- **主数据集**：AmazonWSE（本文构建）
  - Nodes: 19,172 river reaches（SWORD v17b）
  - Sources:
    - **SWOT RiverSP**: ~10K reaches, 1.69% 观测密度
    - **HydroWeb**: ~3.8K, 0.50%
    - **ICESat-2**: ~18K, 0.32%
    - **ANA in situ**: 375 gauges（保留用于评估）
  - 共享时间轴：2016-01-01 至 2026-05-01（共3,774天）

### 实验设置
#### 任务定义
- **目标**：在任意一天、任一river reach上重建WSE。
- **双轨评估**：
  1. **SWOT时期**（2023/07 – 2026/05）：含SWOT数据，观测密度 2.7%
  2. **前SWOT回溯期**（2016/01 – 2022/06）：仅传统卫星，观测密度 0.6%

#### 输入构造
- 子图采样参数：
  - 锚点为中心reach
  - 最大空间距离：300 km
  - 上游/下游比例：0.75 / 0.25
  - 时间窗口：91天
- Token表示：
  - 动态token：每个观测作为一个token，包含WSE值及其元数据
  - 静态token：每个位置的平均WSE（相对下游根节点归一化）

#### 评估指标
- **RMSE**（Root Mean Square Error）：主要指标，对比in situ gauges
- **KGE**（Kling-Gupta Efficiency）：综合评价相关性、变异性、偏差，理想值为1

#### 基线方法对比
| 类型 | 方法 | 特点 |
|------|------|------|
| **Transductive GNN** | GRIN, SPIN-H, ImputeFormer | 假设所有节点在训练中可见 |
| **Inductive GNN** | IGNNK, KITS | 可推广到未见节点 |
| **非图方法** | Temporal LSTM, kNN | 分别建模时间或简单插值 |
| **领域专用方法** | Reach-Reg (Halicki et al. 2026) | 当前SWOT WSE致密化的SOTA方法，基于回归链与流速模型 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（vs in situ gauges）

#### 表：主要基线对比（RMSE ↓）

| Model | >2023/07 (SWOT era) | <2022/06 (Hindcast) |
|-------|---------------------|--------------------|
| Temporal LSTM | 1.47 | 2.47 |
| GRIN | 0.88 | 1.28 |
| SPIN-H (subgraph) | 0.74 | 0.94 |
| ImputeFormer (subgraph) | 0.67 | 0.91 |
| **Ours (Mamba-based)** | **0.62** | **0.84** |

> ✅ 我们的方法在两个时期均显著优于所有基线。

#### 与领域SOTA方法 Reach-Reg 对比（Table 5）

| Period | 方法 | RMSE ↓ | KGE ↑ |
|--------|------|--------|-------|
| SWOT era | Reach-Reg | 0.90 | 0.85 |
| | Ours | **0.61** | **0.92** |
| Hindcast | Reach-Reg | 0.85 | 0.70 |
| | Ours | **0.80** | **0.88** |

> ✅ 在SWOT时代，我们的模型相比Reach-Reg **RMSE降低39%**  
> ✅ 在回溯期，**RMSE降低18%**，且**覆盖范围更广**

此外，Reach-Reg仅能在有足够邻近观测的站点运行（覆盖率184/284），而我们的模型可为**全部19K河段提供预测**。

### 消融实验结果（Table 4）

| 配置 | RMSE |
|------|------|
| 完整模型（Base） | 0.56 |
| 移除 TreePE 编码 | 0.58 |
| 移除 卫星源编码 | 0.58 |
| 移除 平均WSE token | 0.58 |
| 移除 ICESat-2 数据 | 0.59 |
| 移除 SWOT & ICESat-2 | 0.60 |
| 输入顺序改为 Random | 0.64 |
| 使用单向 Mamba | 0.72 |
| 仅使用孤立节点（Isolated） | 2.58 |

> 🔍 发现：
> - 所有元数据编码都带来增益，尤其是**拓扑编码**和**流向排序**
> - **双向Mamba**比单向提升明显（+14%）
> - **多源数据融合**至关重要
> - **子图上下文**极大提升性能（Isolated退化严重）

---

## 4. 关键结论和发现

### 主要发现
1. **现有STGNN方法不适用于极端稀疏+大尺度河流图**：
   - GNN在<1%日观测率下几乎无法传播有效信息；
   - 全图输入方式不可扩展；
   - 显式图卷积不如隐式拓扑编码有效。

2. **序列建模更适合此类任务**：
   - 将“空间+时间”扁平化为序列，天然适配稀疏输入；
   - 结合**拓扑感知位置编码**（TreePE）可有效利用河流结构；
   - **双向Mamba**能同时捕获上游历史与下游未来信号，符合水文传播特性。

3. **拓扑信息仍有价值，但应通过编码而非GNN传递**：
   - 支持 Kirschstein and Sun (2024) 的观点：直接GNN message passing 对河流预测帮助有限；
   - 但我们发现：当以**子图采样 + metadata encoding** 方式引入拓扑时，仍能显著提升性能。

4. **归纳式建模更具实用前景**：
   - 模型可在未观测过的河段上做出合理推断；
   - 更适合部署于缺乏长期观测的新区域。

### 方法局限性
- **依赖高质量拓扑图**：需准确的SWORD reach连接关系；
- **未显式建模物理方程**：如圣维南方程组或Manning公式，可能限制极端事件下的外推能力；
- **未考虑季节性突变或人类干预**（如水库调度）；
- **推理效率有待优化**：尽管训练高效，但仍需逐窗口滑动预测。

### 未来工作方向
- 探索 **Physics-informed Mamba** 架构，结合水文先验；
- 扩展至其他流域（如刚果、长江），验证跨区域泛化性；
- 引入 **多任务学习**：同步预测WSE、流量（discharge）、宽度等变量；
- 开发轻量化版本，支持实时洪水预警系统集成；
- 探索如何将本方法应用于 **SWOT Level 4 产品生成流程**。

--- 

> 📌 **总结一句话**：  
> 本文提出了 AmazonWSE —— 当前最大最稀疏的河流WSE填补基准，并证明了基于Mamba的序列模型在该任务上全面超越传统GNN与领域专家模型，为遥感驱动的水文监测提供了新的范式。

</details>

---

### 13. [Musec: MomentUm SpEctral Clipping for Stable Muon-type Training](https://arxiv.org/abs/2609.11655)

**Authors**: Zhuanghua Liu, Menglian Wang, Luo Luo  
**Category**: cs.LG  
**Published**: 2026-09-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.11655v1  

#### Abstract
Muon has emerged as a highly effective optimizer for large language model training, often achieving superior convergence and performance compared with the widely adopted Adam and AdamW optimizers. Nevertheless, Muon is prone to training instability due to its spectral flattening, manifested by loss ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Musec: MomentUm SpEctral Clipping for Stable Muon-type Training**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **Muon optimizer 的训练不稳定性**：尽管 Muon 在大语言模型（LLM）训练中表现出色，收敛速度快于 Adam 和 AdamW，但它存在严重的训练不稳定问题，表现为 **loss spikes** 和 **model weights 的无界增长**。
- **根本原因在于 spectral flattening**：Muon 将动量矩阵的所有奇异值设为约 1，导致原本小奇异值的方向被过度放大，引发参数爆炸。
- **现有方法的局限性**：
  - **Logit soft-capping** 和 **QK-Norm** 只作用于注意力机制，无法解决全局优化问题。
  - **MuonClip** 仅对 query-key 权重进行裁剪，是架构特定的，且未覆盖 MLP 和 value-output 层。

### **提出了什么新方法或新思路**
- **Musec (MomentUm SpEctral Clipping)**：
  - 将 Muon 的 **spectral flattening** 替换为 **spectral clipping**：只裁剪超过阈值 $D$ 的奇异值，保留原始动量矩阵的谱结构。
  - 是一种 **optimizer-level、architecture-agnostic** 的稳定机制，适用于所有权重矩阵。
- **Soft Musec**：
  - 提出高效实现，使用平滑饱和函数 $h(w, D) = \frac{Dw}{\sqrt{w^2 + D^2}}$ 替代硬裁剪。
  - 利用 **coupled Newton-Schulz iterations** 近似计算，避免昂贵的 SVD，适合 GPU 并行。

### **相比现有方法的优势**
| 方法 | 是否架构无关 | 是否作用于整个优化器 | 是否有理论保证 | 是否可扩展 |
|------|----------------|------------------------|----------------|------------|
| MuonClip | ❌（仅 QK） | ❌（仅权重裁剪） | ❌ | ⚠️有限 |
| SPECTRA | ✅ | ✅（wrapper） | ✅（凸光滑） | ✅ |
| **Musec / Soft Musec** | ✅ | ✅（集成到动量更新） | ✅（非凸非光滑） | ✅ |

- **理论优势**：首次为 Muon-type 方法在 **nonconvex nonsmooth** 场景下提供收敛保证。
- **实践优势**：在更大学习率范围内保持稳定，匹配甚至优于调优后的 Muon 性能。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **FineWeb** (Penedo et al., 2024)
- **OpenWebText** (Gokaslan & Cohen, 2019)
- **C4** (Raffel et al., 2020)

### **实验设置和评估指标**
- **模型架构**：基于 `modded-nanogpt` 的 decoder-only Transformer，包含：
  - RMSNorm
  - RoPE (Rotary Positional Embeddings)
  - Square ReLU
  - FlashAttention-3
- **模型规模**：
  - NanoGPT-Small (~491M)
  - NanoGPT-Medium (~613M)
  - NanoGPT-Wide (~1.63B)
- **评估指标**：
  - **Validation loss**（主指标）
  - **Training dynamics**（loss 曲线是否平稳）
  - **Spectral norm of weight matrices**（衡量稳定性）
  - **Effective rank of update matrices**（验证谱结构保留）
  - **Wall-clock time per step**（计算开销）

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Muon** (Jordan et al., 2024) | 原始 Muon，使用 Newton-Schulz 近似正交化 |
| **MuonClip** (Kimi et al., 2025) | 对 QK 权重进行裁剪 |
| **SPECTRA** (Jiang et al., 2026) | 谱裁剪 wrapper，以 SGDM 为基础优化器 |
| **Soft Musec** | 本文提出的方法，使用平滑谱裁剪 + Newton-Schulz |

> 所有方法均对非嵌入层使用对应 optimizer，嵌入层统一使用 AdamW。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **(1) 学习率鲁棒性测试（FineWeb）**
- **NanoGPT-Small**：
  - 当 learning rate ≥ 0.5 时，Muon 和 MuonClip **发散**。
  - Soft Musec 和 SPECTRA **仍稳定收敛**，最终 validation loss 相当。
- **NanoGPT-Medium/Wide**：
  - Muon 在 lr ≥ 0.2 时即出现严重震荡或发散。
  - Soft Musec 在 lr = 0.8 下仍稳定，而 Muon 完全失败。

> 图 1 显示，在高学习率区域，Soft Musec 和 SPECTRA 显著优于 Muon 和 MuonClip。

#### **(2) 训练动态分析（lr=0.2）**
- 所有方法在第 400 步附近经历 **loss spike**（因学习率、batch size、窗口同步调整）。
- **Muon**：恢复缓慢，最终 loss 更高。
- **MuonClip**：有所改善但仍振荡。
- **Soft Musec / SPECTRA**：快速恢复，平滑收敛，最终 loss 最低。

> 图 2 显示 Soft Musec 和 SPECTRA 的训练曲线最平稳。

#### **(3) 权重谱范数分析（Spectral Norm）**
- **Muon**：QK、VO、MLP 权重的谱范数持续增长至 300–400，剧烈震荡。
- **MuonClip**：仅抑制 QK 范数，VO 和 MLP 仍失控。
- **Soft Musec / SPECTRA**：所有组件谱范数始终 < 10，高度稳定。

> 图 3 验证了 Musec 的全局稳定能力。

#### **(4) 计算开销**
- Soft Musec 与 Muon 的 wall-clock time 几乎一致：
  - NanoGPT-Small：~1030 ms vs 1021–1049 ms
  - NanoGPT-Wide：~15800 ms vs 15645–15977 ms
- **MuonClip 最慢**，因其额外裁剪操作。

> 图 10 显示 Soft Musec 引入的计算开销可忽略。

---

### **消融实验结果**

#### **(1) 裁剪阈值 $D$ 敏感性（图 11）**
- 在 lr = 0.1–0.2 时，$D \in [0.05, 1.0]$ 均表现良好。
- 在 lr = 0.5 时：
  - $D = 0.05, 0.25$：平滑收敛。
  - $D = 0.75, 1.0$：后期出现轻微 loss spike。
- **即使 $D=1.0$，Soft Musec 仍收敛**，而 Muon 完全发散 → 表明谱裁剪本身即有效。

#### **(2) Newton-Schulz 迭代次数（图 12）**
- 使用 3、5、8 次迭代时，性能差异极小。
- 5 次已足够平衡精度与效率。
- 表明近似算法 **robust 且高效**。

#### **(3) Effective Rank 分析（图 9）**
- **Muon**：update 矩阵 effective rank 接近满秩 → 谱信息完全丢失。
- **Soft Musec / SPECTRA**：effective rank 显著降低。
- **Soft Musec < SPECTRA**：因 Musec 将裁剪后的动量反馈回 EMA，实现更严格的谱控制。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Spectral flattening 是 Muon 不稳定的根本原因**，而非注意力机制等局部问题。
2. **Spectral clipping 是更合理的替代方案**：保留谱结构的同时抑制异常方向。
3. **Musec 是首个具有非凸非光滑收敛保证的 Muon-type 方法**，复杂度达最优水平 $O(r^{3/2}\delta^{-1}\epsilon^{-3})$。
4. **Soft Musec 实践效果优异**：
   - 在更大学习率范围内稳定。
   - 匹配甚至超越调优后的 Muon。
   - 计算开销几乎无增加。
5. **与 SPECTRA 的关键区别**：
   - Musec 将裁剪**集成到动量递推中**，确保动量状态始终有界。
   - SPECTRA 仅对输出裁剪，历史大梯度可能长期影响优化轨迹。

### **方法的局限性**
- **依赖 Newton-Schulz 近似**：虽然高效，但在极端病态条件下可能不如精确 SVD 稳定（但实验中未观察到）。
- **超参数 $D$ 需调优**：尽管对 $D$ 不敏感，但最优值随模型和任务变化。
- **尚未在 MoE 或超大规模模型（>10B）上验证**。

### **未来工作方向**
- 探索自适应 clipping threshold $D$ 的策略。
- 将 Musec 应用于其他 matrix-aware optimizers（如 Shampoo、SOAP）。
- 在 MoE 架构和百万 token context 模型中进一步验证。
- 结合量化训练，研究其对低精度数值稳定性的帮助。

---

> **总结**：  
> Musec 从优化器层面重新思考了 Muon 的稳定性问题，提出 **spectral clipping** 替代 **spectral flattening**，兼具理论严谨性与工程实用性。Soft Musec 实现了高效、稳定、可扩展的训练体验，是当前 Muon-type 优化器中最稳健的选择之一。

</details>

---

### 14. [Auto-RecSys: Harnessing Autonomous Research Agents for Industry-Scale Recommender System](https://arxiv.org/abs/2609.10922)

**Authors**: Ming Li, Dai Li, Xuying Ning, Bo Sun, Rui Li, Yi Zhang, Silvia Gong, Xuan Cao, Rui Li, Cornelia Carapcea, Qunshu Zhang, Zhigang Wang, Yinglong Xia, Andy Wang  
**Category**: cs.CL  
**Published**: 2026-09-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.10922v1  

#### Abstract
Auto-research agents have shown the potential to automate hypothesis generation, experiment execution, and iterative refinement. However, scaling this paradigm to industry-scale recommendation models introduces two challenges: (1) long feedback loops, where model training can take days, making seria...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Auto-RecSys: Harnessing Autonomous Research Agents for Industry-Scale Recommender Systems 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

本论文针对**工业级推荐系统模型研发中的两大核心挑战**：

1. **长反馈循环（Long feedback loops）**：单次训练可能耗时数天，导致串行迭代效率极低，严重限制研究吞吐量。
2. **系统复杂性（System complexity）**：大规模配置、脆弱的基础设施依赖、多日GPU任务等，使得执行过程容易失败且难以恢复。

在传统手动或轻度自动化流程中，研究人员大量时间被消耗在实验管理而非创新上。

---

### 提出了什么新方法或新思路

作者提出了 **Auto-RecSys** —— 一个专为**长期、高成本、分布式环境下的推荐模型研发**设计的自主研究系统。其核心创新包括：

#### （1）**三层Harness架构设计**

- **Distributed Asynchronous Execution**  
  支持跨服务器并行运行多个实验，每个实验独立处于生命周期的不同阶段（如 IMPLEMENTING, TRAINING），通过持久化状态机进行跟踪，无需连续会话。

- **Centralized Cross-Server Memory**  
  所有实验状态、playbook、历史轨迹均存储于共享内存层，支持跨会话、跨服务器的状态恢复，确保中断后可续。

- **Cognitive-Procedural Separation**  
  将LLM的自然语言推理能力（认知层）与确定性脚本执行（程序层）分离：
  - **自然语言技能文件（skill files）** 指导决策逻辑；
  - **代码脚本** 负责精确操作（如API调用、状态更新），保证操作正确性。

#### （2）**双循环自演化架构（Dual-Loop Self-Evolving Architecture）**

- **Execution Evolution Loop**  
  从每次实验轨迹中提炼出**模型专属的playbook**，记录成功pipeline、已知dead ends、硬件要求等，实现“经验沉淀”，提升后续执行可靠性。

- **Idea Evolution Loop**  
  基于实验结果积累科学结论，指导未来idea生成，避免重复试错，促进复合型改进（如组合有效特征）。

#### （3）**Playbook的一次性迁移机制（One-Shot Transfer）**

首个模型的playbook结构可作为模板迁移到新模型，新模型只需填充特定内容（如关键文件路径、配置规则），大幅缩短冷启动时间，实现跨模型知识复用。

---

### 相比现有方法的优势

| 维度 | 现有方法（如AutoResearch, FARS） | Auto-RecSys |
|------|-------------------------------|-----------|
| 反馈周期 | 分钟~小时级（小规模任务） | 小时~天级（工业级训练） |
| 迭代策略 | 串行快速迭代 | 并行异步探索（Distributed Portfolio） |
| 执行鲁棒性 | 本地重跑即可恢复 | 跨会话、跨服务器持久恢复 |
| 知识积累 | 无或弱记忆机制 | 结构化playbook + 自然语言指令驱动 |
| 可扩展性 | 单一任务导向 | 多模型、多服务器、长期演进 |

> ✅ **核心优势**：将LLM代理的能力从“短平快”的学术任务拓展到**真实工业场景的长周期、高成本、强依赖环境**，实现了可持续的知识积累与系统自进化。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

论文未公开具体数据集名称，但明确指出实验基于**多个工业级推荐系统模型**（industry-scale recommendation models），具备以下典型特征：

- 模型规模大，训练需数百GPU小时；
- 配置复杂，涉及数千行代码与多模块耦合；
- 训练周期长达数天；
- 依赖动态基线（baseline）进行A/B比较。

---

### 实验设置和评估指标

#### 系统模式对比

- **Interactive Mode**：在关键节点暂停（idea选择、代码审查、训练提交、结果回顾），由人类审批。
- **Autonomous Mode**：全自动执行，仅在无法解决时上报。

#### 主要评估维度

1. **Human Bandwidth per Idea**  
   衡量研究人员投入的时间成本（从提案到分析完成所需的人工参与时间）。

2. **Execution Reliability**  
   - **Major Fix Steps per Iteration**：每次迭代中因操作错误（如硬件不匹配、包版本冲突）而需要修复的步骤数。
   - **Zero-Fix Rate**：无需任何重大修复即可顺利完成的实验比例。

3. **Playbook Evolution Trajectory**  
   观察playbook在不同阶段的学习曲线，特别是在**基线变更前后**的表现变化。

4. **Session Recovery & Cross-Server Handoff**  
   测试系统在服务器崩溃、会话切换等情况下的恢复能力。

---

### 基线方法对比

论文未直接对比其他完整系统（如FARS），而是以**人工主导的研发流程**作为主要基线，并隐含对比了典型的非持久化、非结构化的自动化尝试。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）**人力投入显著降低**

| 模式 | 人均每idea投入时间 | 效率提升 |
|------|------------------|--------|
| Manual Process | 数小时至数天 | 基准 |
| Interactive Mode | 几分钟 | >10倍 |
| Autonomous Mode | 极少（仅高风险干预） | >12倍 |

> 💡 在相同人力投入下，原本只能处理1个idea，现在可同时管理十几个并行实验。

#### （2）**执行可靠性随playbook成熟持续提升**

在对**31次唯一实验迭代**的分析中（同一模型）：

| 阶段 | 平均Major Fix Steps/Iter | Zero-Fix Rate |
|------|--------------------------|--------------|
| Bootstrap (1–4) | 4.0 | <20% |
| Stabilized (5–20) | ↓ 1.3 | ↑ ~60% |
| Post-Transition (26–31) | ↓↓ **0.5** | ↑↑ **83%**（5/6次零修复） |

> 🔁 系统经历“学习 → 回退 → 再恢复”过程，在基线变更后仍能快速重建可靠执行流程。

#### （3）**错误类别消除验证**

- 所有操作类错误均为**结构性而非随机性**，一旦被记录为dead end即不再重现。
- 新增错误仅出现在过渡期（如package layer mismatch），并在几轮内被吸收。
- 最终阶段唯一错误是新型编译bug（type-inference failure），属真正新颖问题。

---

### 消融实验结果（隐式消融）

虽然没有显式AB测试，但从轨迹分析中可得出以下因果证据：

| 机制 | 效果 |
|------|------|
| **Dead-end Avoidance** | 学习到某GPU世代不稳定后，自动切换至稳定硬件；后续再未出现同类故障。 |
| **Pipeline Crystallization** | 成功pipeline固化为标准流程（如toy-train验证命令、build-based提交参数）。 |
| **Validation-skip Pattern** | 当开发机无GPU时，自动跳过本地验证，防止无效失败。 |
| **Self-Healing Infrastructure** | 发现monitor进程因context overflow死亡，自主设计cron-based替代方案并提交代码升级。 |

> ✅ 所有这些行为均源于playbook的自然语言指令，而非硬编码规则。

---

## 4. 关键结论和发现

### 论文的主要发现

1. **自主研究可以扩展到工业级复杂系统**  
   通过合理的harness设计（异步、持久、分离），LLM代理可在高延迟、高成本环境中有效运作。

2. **自然语言是LLM代理最有效的程序性记忆载体**  
   - Playbook以markdown形式提供“DO NOT”指令和编号流程，比数值评分更易理解和执行。
   - Agent偏好读取文本而非元数据，验证了“same-language reasoning”的有效性。

3. **Playbook是一种可积累的操作资本（appreciating asset）**  
   - 每一次失败都转化为scar tissue（伤疤组织），每一次成功都成为muscle memory（肌肉记忆）。
   - 知识不会丢失，反而随时间增值。

4. **并行探索是应对长反馈的关键**  
   分布式实验组合（distributed portfolio）使整体研究吞吐量大幅提升，即使单个实验仍需数天。

5. **系统具备真正的自我演化能力**  
   不仅优化执行路径，还能识别元模式（如“旧正向实验在新基线上失效”），推动研究方向转变。

---

### 方法的局限性

1. **冷启动依赖人工交互**  
   初始几轮仍需人类引导建立playbook，完全无人初始化尚不可行。

2. **缺乏正式的playbook更新验证机制**  
   当前接受所有agent判断的更新，虽未见退化，但未来可能引入冲突（如新dead end与已有proven strategy矛盾）。

3. **当前为单研究员模式**  
   尚未支持团队协作、共享backlog或多用户冲突协调。

4. **idea生成仍受限于已有模式**  
   创新多来自gap分析与常见pattern应用，尚未展示突破性原创能力。

---

### 未来工作方向

| 方向 | 具体设想 |
|------|--------|
| **Proxy Models for Rapid Screening** | 引入小型代理模型快速筛选idea，仅对高潜力候选进入全规模训练，加速Idea Evolution Loop。 |
| **Cross-Model Knowledge Transfer** | 在相似架构间共享通用经验（如“gating机制对多任务有益”、“embedding dim >128收益递减”）。 |
| **Validation-Gated Playbook Updates** | 引入轻量级校验机制，防止playbook内部冲突，增强长期稳定性。 |
| **Adaptive Human-in-the-Loop** | 动态决定是否请求人工介入，基于confidence score与risk level实现细粒度控制。 |
| **Team-Scale Operation** | 扩展至多人协作环境，支持共享idea池、联合history tracking与冲突解决机制。 |

---

## 总结

> 🌟 **Auto-RecSys 是首个将自主AI研究成功应用于工业级推荐系统的系统级解决方案**。它不仅提升了实验效率，更重要的是构建了一个**可持续进化的研究基础设施**——让机器不仅能“做实验”，更能“学会如何更好地做实验”。  
>
> 其核心思想——**通过自然语言playbook实现认知与程序分离、通过双循环实现执行与创意共同演化**——为未来大规模AI-driven科研提供了重要范式。

</details>

---

### 15. [Can LLMs Normalize Databases? A Benchmark and Multi-Agent Framework for Schema Normalization](https://arxiv.org/abs/2609.11141)

**Authors**: Dong-Jae Koh, Huisu Kim, SeongHwan Yoon, Lasse M. Jantsch, Chun-Hee Lee, Seonghyeon Lee, Young-Kyoon Suh  
**Category**: cs.CL  
**Published**: 2026-09-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.11141v1  

#### Abstract
Large Language Models (LLMs) are increasingly used to generate structured outputs, but their reliability remains unclear when those outputs must satisfy database-level constraints. We study this issue through database normalization, involving reasoning about functional dependencies, lossless join de...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Can LLMs Normalize Databases? A Benchmark and Multi-Agent Framework for Schema Normalization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文研究了 **Large Language Models (LLMs)** 在数据库模式规范化（schema normalization）任务中的可靠性问题。尽管LLMs被广泛用于生成结构化输出，但在需要满足严格数据库约束（如函数依赖、无损连接分解、外键完整性）的任务中，其表现尚不明确。

具体而言，该研究聚焦于从 **1NF 到 BCNF** 的关系型数据库规范化过程，这是一个涉及复杂推理的高难度任务，要求模型能够：
- 推断函数依赖（Functional Dependencies, FDs）
- 诊断范式违规（normalization violations）
- 执行正确的模式分解（decomposition）
- 维护语义等价性和逻辑有效性

现有研究缺乏对LLM在该任务上的系统性评估和可靠框架支持。

---

### 提出的新方法与新思路

#### ✅ 贡献一：提出 **DNBENCH** —— 首个面向LLM驱动的数据库规范化基准
- 包含 **3,275 个样本**，源自真实数据库（Spider 和 BIRD），覆盖从 1NF 到 BCNF 的各类范式违规。
- 引入 **三轴评估协议（three-axis evaluation protocol）**：
  1. **Semantic Equivalence（语义等价性）**：检查是否满足 lossless join。
  2. **Structural Accuracy（结构准确性）**：衡量生成 DDL 与黄金标准在列、主键、外键上的匹配度。
  3. **Logical Validity（逻辑有效性）**：评估模型对违规原因的解释质量及推理一致性。
- 最终聚合为统一评分指标 **DNB-ScORE**（几何平均，当语义失败时得分为0）。

#### ✅ 贡献二：提出 **MARS（Multi-Agent Reasoning for Schemas）** 多智能体框架
将复杂的规范化流程拆解为多个角色分工协作的子任务：
1. **Evidence Agent**：提取函数依赖证据（来自数据行或自然语言规则）
2. **Diagnosis Agent**：识别违反的范式并制定分解计划
3. **Schema Generator Agent**：生成符合规范的 SQL DDL
4. **Verifier Agent**：进行确定性验证（如 lossless join、PK/FK 正确性）
5. **Repair Loop**：若验证失败，触发定向修复（最多两轮）

该设计借鉴了 **multi-agent LLM frameworks**（如 AutoGen、MetaGPT）的思想，通过模块化解耦提升整体鲁棒性。

---

### 相比现有方法的优势

| 方法 | 是否LLM-based | 支持1NF–BCNF | 支持复杂多违规链 | 是否提供评估协议 | 是否开源 |
|------|----------------|---------------|--------------------|---------------------|----------|
| RDBNorma (2011) | ❌ | ✔️ (至3NF) | ❌ | ❌ | ❌ |
| EDNA (2013) | ❌ | ✔️ (至BCNF) | ❌ | ❌ | ❌ |
| NormTab (2024) | ✔️ | ❌ | ❌ | ❌ | ❌ |
| TABARD (2025) | ✔️ | ❌ | ❌ | ❌ | ❌ |
| Miffie (2025) | ✔️ | ✔️ (1NF–3NF) | ⚠️有限 | ⚠️部分 | ❌ |
| **DNBENCH (Ours)** | N/A | ✔️ | ✔️ | ✔️ | ✅（承诺发布） |

> ✅ **优势总结**：
> - **首个专为LLM设计的端到端数据库规范化 benchmark**
> - **首次引入多维度量化评估体系（语义+结构+逻辑）**
> - **MARS 框架显著优于单提示（single-prompt）和双LLM自精炼方法（Miffie）**

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要来源：两个高质量 Text-to-SQL 数据集
  - **Spider**: 200 个数据库，138 个领域，1,020 张表
  - **BIRD**: 95 个数据库，37 个领域，694 张表
- 构建方式：通过对原始 schema 进行受控“去规范化”处理，注入已知类型的范式违规，并标注黄金分解路径。
- 最终 DNBENCH 数据集包含：
  - **Dev samples**: 1,495
  - **Test samples**: 900
  - **Total**: 2,395（另有900个额外构建样本，总计3,275）

---

### 实验设置

#### 三种实验场景（对应不同难度级别）：

| 实验 | 目标范围 | 是否提供FDs | 描述 |
|------|-----------|-------------|------|
| **Exp. 1: Single** | 仅修复最早出现的违规 | ✔️提供 | 测试单一规则应用能力 |
| **Exp. 2: Complex** | 修复所有违规链 | ✔️提供 | 测试多步推理与交互处理能力 |
| **Exp. 3: Real World** | 修复所有违规链 | ❌不提供 | 模拟真实场景，需从数据中推断FDs |

#### 模型选择（4个主流开放权重LLM）：
- **Llama 3.3 70B**（dense）
- **Gemma3 27B**（dense）
- **Qwen3-30B**（MoE）
- **Mixtral 8x7B Instruct**（MoE）

均测试 zero-shot 与 few-shot 设置。

---

### 评估指标

#### 主要指标：**DNB-ScORE**
$$
\text{DNB-ScORE} = \text{Semantic} \times \text{Structural} \times \text{Logical}
$$
其中：
- **Semantic**：lossless join 成功与否（二值）
- **Structural**：由以下三项平均构成
  - Column F1
  - Primary Key F1
  - Foreign Key Score = FK F1 × FK Connected Score
- **Logical**：由以下两项平均构成
  - Violation F1（预测违规类型 vs 黄金标签）
  - LLM-as-a-Judge 得分（基于 GPT-OSS-20B 对解释质量打分）

> ⚠️ 若 Semantic 不通过，则 DNB-ScORE = 0

---

### 基线方法对比
- **Baseline**：单次提示（single-prompt）直接生成 DDL + 解释
- **Miffie (Dual-LLM SR)**：生成器-验证器循环自精炼框架（原用于1NF–3NF）
- **MARS (Ours)**：提出的四阶段多智能体框架

所有方法均以 **Qwen3-30B** 为主干模型，在 **Real World Setting** 下比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 4 & Table 6）

#### 📊 整体 DNB-ScORE 表现（平均值）

| Model | Single (Avg) | Complex (Avg) | Real World (Avg) |
|-------|--------------|----------------|-------------------|
| Llama 3.3 70B | 0.416 | 0.349 | **0.204** |
| Gemma3 27B | 0.347 | 0.363 | 0.189 |
| Qwen3-30B | 0.397 | 0.274 | 0.231 |
| Mixtral 8x7B | 0.153 | 0.152 | 0.126 |
| **Overall** | **0.328** | **0.285** | **0.188** |

> 🔍 发现：**FDs 未提供时性能大幅下降**，说明 **FD inference 是瓶颈**

---

#### 📈 MARS vs Baseline vs Miffie（Real World Setting）

| Method | Prompting | DNB-ScORE |
|--------|-----------|------------|
| Baseline | Zero-shot | 0.253 |
| Baseline | Few-shot | 0.209 |
| Miffie | Zero-shot | 0.308 |
| Miffie | Few-shot | 0.339 |
| **MARS (Ours)** | Zero-shot | **0.423** |
| **MARS (Ours)** | Few-shot | **0.418** |

✅ **MARS 相比 Baseline 提升高达 82.0%**

---

#### 🔍 分项得分分析（Table 6）

| 方法 | Semantic | Structural | Logical | DNB-ScORE |
|------|----------|------------|---------|-----------|
| Baseline | 0.430 | 0.651 | 0.331 | 0.231 |
| Miffie | 0.692 | 0.688 | 0.337 | 0.324 |
| **MARS** | **0.715** | **0.728** | **0.307** | **0.420** |

> 💡 观察：
> - MARS 显著提升了 **Semantic（lossless join）** 和 **Structural（尤其是FK重建）**
> - Miffie 虽然局部改进，但 **Logical 得分下降明显**，说明其 LLM-based verifier 无法有效判断推理正确性
> - MARS 的 **FK Score 达到 0.227–0.262**，远高于 Baseline 的 0.103–0.155

---

### 消融实验与深入分析（Appendix G）

#### MARS 修复机制效果（Table 14）
- **无需修复**：42.6%，DNB-ScORE = 0.6205
- **一次修复后成功**：7.2%，得分仍高（0.5728）
- **两次修复后多数失败**：50.2% 中仅 67 例通过，平均得分仅 0.2292
> ➡️ 说明 **修复机制能有效纠正轻度错误，但难以挽救根本性推理失误**

#### 各阶段瓶颈分析（Table 15）
| 阶段 | 指标 | 得分 |
|------|------|------|
| Evidence | FD Recall (exact match) | 0.5119 |
| Diagnosis | Violation-type F1 | 0.6731 |
| Schema Generation | Plan-to-DDL Match | **0.7805** |
| Verification | Pass Rate | 0.5241 |

> 🔍 结论：**上游 FD 推理仍是最大瓶颈**；一旦诊断计划正确，Schema Generator 可较准确实现

---

## 4. 关键结论和发现

### 主要发现

1. **LLMs 能识别范式违规，但难以生成有效的规范化 schema**
   - Violation F1 最高达 0.678，表明模型具备一定诊断能力
   - 但 **FK Score 仅为 0.10–0.17**，说明跨表约束重建能力极弱

2. **函数依赖（FD）推理是当前最大瓶颈**
   - 当 FDs 不提供时（Real World Setting），所有模型性能骤降
   - 即使是最佳模型（Qwen3-30B），DNB-ScORE 也从 0.397（Single）降至 0.231

3. **MARS 框架显著优于单提示和自精炼方法**
   - 提升 **82.0%** 的 DNB-ScORE
   - 尤其改善了 **lossless join** 和 **foreign key** 的生成质量
   - 验证了 **role-specialized multi-agent design** 在复杂结构推理任务中的有效性

4. **现有的 LLM-based verifier（如 Miffie）不可靠**
   - Miffie 的迭代优化反而导致 **Logical Score 下降**
   - LLM-as-a-Judge 分析显示其无法区分高质量与低质量 schema

---

### 局限性（Limitations）

1. **FK 重建仍是挑战**
   - 即使使用 MARS，FK Score 仍远低于 Column F1 和 PK F1
   - 表明恢复跨表引用结构比维护单表结构更难

2. **BCNF 推理和已规范化输入处理不佳**
   - 对仅存在 BCNF 违规的情况处理较差（需候选键推理）
   - 对本就合规的输入常出现“过度规范化”（unnecessary decomposition）

3. **推理成本较高**
   - MARS 使用多次 LLM 调用（证据提取、诊断、生成、验证、修复）
   - 虽然性能提升，但计算开销增加，不适合低延迟场景

4. **上游错误难以纠正**
   - 若 Evidence 或 Diagnosis 出错，后续修复难以挽回
   - 当前 verifier 更擅长检测局部合成错误（如缺列），而非全局逻辑错误

---

### 未来工作方向

1. **增强 FD 推理能力**
   - 引入更强的语义理解模块或知识增强机制
   - 结合 schema semantics 与 domain knowledge 进行保守推理

2. **改进多智能体路由策略**
   - 动态决定何时调用哪个 agent，减少冗余调用
   - 引入 feedback-driven agent orchestration

3. **开发专用 verifier 模块**
   - 替代 LLM-based verifier，采用形式化验证工具辅助判断
   - 构建可解释的错误定位机制

4. **扩展至更高范式与反规范化任务**
   - 支持 4NF、5NF 等更高级别规范化
   - 探索 denormalization for performance optimization 场景

5. **推动 benchmark 生态建设**
   - 开源 DNBENCH 促进社区复现与比较
   - 鼓励更多针对数据库设计任务的 LLM 研究

---

> ✅ **总结一句话**：  
> 本文揭示了 **LLMs 在数据库规范化任务中的核心瓶颈在于 FD 推理与跨表约束重建**，并提出了首个系统性 benchmark **DNBENCH** 与高效多智能体框架 **MARS**，实现了 **82% 的性能提升**，为未来 LLM 驱动的数据库自动化设计奠定了基础。

</details>

---

### 16. [Domain-Specific Hallucination Detection in Large Language Models](https://arxiv.org/abs/2609.11878)

**Authors**: Varun Teja Chundru, Debasmita Biswas  
**Category**: cs.CL  
**Published**: 2026-09-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.11878v1  

#### Abstract
Large language models generate fluent text that can contain unfaithful claims -- a phenomenon known as hallucination. We present a multi-signal detection pipeline combining fine-tuned DeBERTa-v3 classification, Monte Carlo (MC) Dropout uncertainty quantification, and temperature-scaled calibration f...

---

### 17. [T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks](https://arxiv.org/abs/2609.11042)

**Authors**: Junyao Yang, Yucheng Shi, Zhongzhi Li, Ruhan Wang, Zongxia Li, Haitao Mi, Leowei Liang  
**Category**: cs.LG  
**Published**: 2026-09-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.11042v1  

#### Abstract
Agent usage is shifting toward long-horizon tasks such as coding and scientific discovery, among which terminal tasks are especially important. We introduce T1, a Mixture-of-Experts model of 122B total trained with reinforcement learning, operating a real shell in a cloud sandbox for up to 300+ tool...

---

### 18. [A Dynamic Fusion Large Language Model for Traffic Flow Prediction](https://arxiv.org/abs/2609.11314)

**Authors**: Xue Qiu, Jianli Xiao  
**Category**: cs.LG  
**Published**: 2026-09-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.11314v1  

#### Abstract
Traffic flow prediction is a core supporting technology for intelligent transportation systems. It uses historical data to infer future traffic dynamics in specific areas, thereby helping to alleviate congestion and improve resource allocation efficiency. Traditional neural networks struggle to brea...

---

### 19. [Particle GFlowNets: Rethinking Generative Marginalization Models](https://arxiv.org/abs/2609.11538)

**Authors**: Tiago da Silva, Diego Mesquita, Salem Lahlou  
**Category**: cs.LG  
**Published**: 2026-09-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.11538v1  

#### Abstract
Generative Marginalization Models (MaMs) have been recently introduced as efficient neural sampling models for any-order autoregressive modelling of discrete distributions. By learning both the marginal and conditional probabilities of a persistent-block Gibbs sampler, MaMs enable fast posterior eva...

---

### 20. [Predicting Privacy Leakage from Weight Spectral Density](https://arxiv.org/abs/2609.11780)

**Authors**: Richard J. Preen, Jim Smith  
**Category**: cs.LG  
**Published**: 2026-09-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.11780v1  

#### Abstract
Membership inference attacks (MIAs) are widely used to audit the privacy disclosure risk of machine learning models, however current state-of-the-art attacks require training computationally expensive shadow models, making large-scale privacy evaluation impractical. In this work, we investigate whet...

---

### 21. [AdamX: Cosine similarity meets gradient descent](https://arxiv.org/abs/2609.11867)

**Authors**: Francisco Caldas, Ruben Belo, Cl\'audia Soares  
**Category**: cs.LG  
**Published**: 2026-09-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.11867v1  

#### Abstract
We introduce AdamX, a first-order optimizer that incorporates cosine similarity as an adaptive mechanism for controlling update magnitudes. The proposed method is scalable, model-agnostic, and straightforward to integrate into existing training pipelines. We further introduce a variance rectificatio...

---

### 22. [Can Artificial Intelligence Support Healthcare and Mental Health Through Early Cyberbullying Detection ? The Impact of Emotion-Aware AI on Proactive Online Safety](https://arxiv.org/abs/2609.09735)

**Authors**: Hamed Jelodar, Amir Firouzi, Yen-Wu Lo, Maryam Tanha, Sajjad Dadkhah  
**Category**: cs.AI  
**Published**: 2026-09-11  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.09735v1  

#### Abstract
Healthcare systems, mental health, and public well-being are increasingly affected by cyberbullying and harmful online interactions. This paper presents CareGuard, an early-warning framework designed to support healthcare-driven mental health protection and proactive online safety through the detect...

---

### 23. [Decision Transformer for UAV-Mounted RIS-Assisted Dynamic D2D Communications](https://arxiv.org/abs/2609.09885)

**Authors**: Yaxuan Liu  
**Category**: cs.AI  
**Published**: 2026-09-11  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.09885v1  

#### Abstract
This paper studies unmanned aerial vehicle (UAV)-mouted reconfigurable intelligent surface (RIS)-assisted device-to-device (D2D) communication with stochastic link activation. It models UAV motion and attitude, time-varying Rician angles, and angle-dependent RIS reflection. A joint optimization of U...

---

### 24. [Composable CXL Memory as a Kubernetes-Native Shared Memory for LLM Serving](https://arxiv.org/abs/2609.10790)

**Authors**: Hongjian Fan, Kevin Zhang, David Habinsky, Sean Dykstra  
**Category**: cs.DC  
**Published**: 2026-09-11  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.10790v1  

#### Abstract
We present a Kubernetes Dynamic Resource Allocation (DRA) driver that makes composable CXL memory a schedulable cluster resource, and evaluate the resulting shared-memory tier for cross-node KV-cache reuse in LLM serving. The driver composes CXL regions on demand, materializes them as DAX devices on...

---

### 25. [Adaptive Entangled Game Modules in Artificial General Intelligence](https://arxiv.org/abs/2609.09226)

**Authors**: Haochen Li, Xinshuai Guo, Jingdong Ouyang, Wei Zhang, Leilei Shi  
**Category**: cs.AI  
**Published**: 2026-09-11  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.09226v1  

#### Abstract
We introduce a probability-wave framework for modeling the collective behavior of interacting adaptive agents, deriving testable eigenmodes through a generalized behavioral intelligence (GBI) nonlocal probability-wave equation. This framework captures a broad range of human intelligence behaviors wi...

---

### 26. [CityPlanner: A Sandbox Agent for Executable Urban Planning](https://arxiv.org/abs/2609.09578)

**Authors**: Wentao Zhang, Jingyuan Wang, Zetong Zhou, Yifan Yang, Wenrui Wang  
**Category**: cs.AI  
**Published**: 2026-09-11  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.09578v1  

#### Abstract
Urban planning is a real-world spatial optimization problem that requires selecting feasible actions from large candidate spaces under practical objectives such as cost and service quality. Existing optimization and reinforcement learning methods are effective for fixed formulations, but often depen...

---

### 27. [From State Synchronization to Cognitive Self-Evolution: An Operational Architecture for Cognitive Digital Twins](https://arxiv.org/abs/2609.09625)

**Authors**: Haoran Gao, An Li, Zhen Li, Jun Cai  
**Category**: cs.AI  
**Published**: 2026-09-11  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.09625v1  

#### Abstract
As Digital Twin (DT) systems evolve beyond state synchronization toward task-oriented and knowledge-driven operation, Cognitive Digital Twins (CDTs) have emerged as an extension that incorporates cognitive capabilities into twin operation. Existing CDT studies often focus on specific enabling techni...

---

### 28. [What Should an Agent Forget? Separating What Is Stored from What Is Used](https://arxiv.org/abs/2609.10263)

**Authors**: Yuhang Li, Yuchen Li  
**Category**: cs.AI  
**Published**: 2026-09-11  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.10263v1  

#### Abstract
Persistent language agents need stored experience to remain available across time, while each answer requires evidence suited to a particular question. A superseded fact can mislead a current-state answer and still be essential for a historical query. We present RD-Forget, a training-free framework ...

---

### 29. [SearchAtlas: Analyzing Agentic Search Strategies via Evidential Query Graphs](https://arxiv.org/abs/2609.10901)

**Authors**: Jiacheng Sang, Mengyuan Li, Sanxing Chen, Yukun Huang, Yu Feng, Bhuwan Dhingra  
**Category**: cs.CL  
**Published**: 2026-09-11  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.10901v1  

#### Abstract
LLM search agents are often evaluated on final-answer accuracy, overlooking the process. Analyzing a search strategy requires understanding how credible evidence is retrieved to address question constraints. This valuable information is buried in raw search trajectories that are long and difficult t...

---

### 30. [Distribution-aware Language Neuron Identification in Multilingual Large Language Models](https://arxiv.org/abs/2609.10993)

**Authors**: Minjun Kim, Inho Won, Junghun Yuk, Dongyeon Kim, Jihyo Kim, KyungTae Lim  
**Category**: cs.CL  
**Published**: 2026-09-11  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.10993v1  

#### Abstract
Multilingual large language models (mLLMs) contain a small fraction of feed-forward neurons that are sensitive to particular languages, commonly termed language-specific neurons. Existing work measures language specificity using the entropy of each neuron's language-wise probabilities of being activ...

---

## 🔧 Configuration

This bot is configured to look for papers containing the following keywords:
- LLM, RL, RLHF, Inference, Training, Attention, Pipeline, MOE, Sparse, Quantization, Speculative, Efficient, Efficiency, Framework, Parallel, Distributed, Kernel, Decode, Decoding, Prefill, Throughput, Fast, Network, Hardware, Cluster, FP8, FP4, Optimization, Scalable, Communication

## 📅 Schedule

The bot runs daily at 12:00 UTC via GitHub Actions to fetch the latest papers.

## 🚀 How to Use

1. **Fork this repository** to your GitHub account
2. **Customize the configuration** by editing `config.json`:
   - Add/remove arXiv categories (e.g., `cs.AI`, `cs.LG`, `cs.CL`)
   - Modify keywords to match your research interests
   - Adjust `max_papers` and `days_back` settings
3. **Enable GitHub Actions** in your repository settings
4. **The bot will automatically run daily** and update the README.md

## 📝 Customization

### arXiv Categories
Common categories include:
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning
- `cs.CL` - Computation and Language
- `cs.CV` - Computer Vision
- `cs.NE` - Neural and Evolutionary Computing
- `stat.ML` - Machine Learning (Statistics)

### Keywords
Add keywords that match your research interests. The bot will search for these terms in paper titles and abstracts.

### Exclude Keywords
Add terms to exclude certain types of papers (e.g., "survey", "review", "tutorial").

## 🔍 Manual Trigger

You can manually trigger the bot by:
1. Going to the "Actions" tab in your repository
2. Selecting "arXiv Bot Daily Update"
3. Clicking "Run workflow"

---
*Generated automatically by arXiv Bot* 

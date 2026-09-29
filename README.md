# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-29 11:37:21 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DeepEdu-v1: Efficient and Scalable Agentic LLMs for Vietnamese Education](https://arxiv.org/abs/2609.31568)

**Authors**: Quang Nguyen, Hieu Nguyen, Hien Hoang, Toan Pham, Cong Tran, Nam Vu  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2609.31568v1  

#### Abstract
AI tutoring could markedly improve learning outcomes for students in developing regions such as Vietnam, yet the two obvious paths both fall short. Cloud assistants such as ChatGPT route sensitive student data to foreign servers---violating data-sovereignty laws such as Vietnam's Decree 53---and, pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*DeepEdu-v1: Efficient and Scalable Agentic LLMs for Vietnamese Education*

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决在越南等资源受限地区部署**AI 教育辅导系统**时面临的两大核心挑战：
1.  **数据主权（Data Sovereignty）问题**：主流云服务（如 ChatGPT）会将敏感的学生数据传输至境外服务器，违反越南《网络安全法》第53号法令（Decree 53）等本地化法规。
2.  **本地化知识缺失与推理幻觉（Hallucination）问题**：通用大模型基于西方中心化的语料预训练，对越南本土教材（Sach Giao Khoa, SGK）内容掌握不系统，易产生事实性错误（如混淆历史人物），无法作为可信的教育工具。

### 提出的新方法与创新
为解决上述问题，作者提出了 **DeepEdu-v1** 系统，其核心技术是 **SCALE 框架**（Self-improving Context-Aware Learning Engine），包含两大创新组件：

1.  **高效的长上下文推理引擎（Similarity Chunk Rolling, SCR）**：
    *   **思路**：针对长上下文推理中的高延迟（prefill latency）和显存瓶颈（KV Cache），提出了一种**基于相似性的块滚动机制**。
    *   **创新**：传统方法（如 TokenSelect）在每个子块（sub-chunk）上独立执行稀疏注意力选择。SCR 观察到连续子块的查询表示高度相似，因此将其聚类成“簇”（cluster），并**以簇为单位进行一次检索调用**，实现了从“每子块”到“每簇”的粒度摊销。
    *   **优势**：大幅减少了检索调用次数（减少 7.7×），从而将 **TTFT（Time-To-First-Token）降低约 35%**，同时保持甚至提升了任务准确率。

2.  **自改进的智能体层（Self-improving Agentic Layer）**：
    *   **思路**：避免通过昂贵的微调（fine-tuning）来注入本地知识，而是设计了一个持续演化的、可验证的“剧本”（Playbook）。
    *   **创新**：采用 **Agentic Context Engineering (ACE)** 范式，通过 **Generator-Reflector-Curator** 三元组协作，将过往交互中验证过的正确知识、常见错误和教学策略沉淀为结构化条目。引入了 **Retrieval-Augmented Execution (RAE)** 和 **Failure Memory Bank (FMB)** 来提升效率和鲁棒性。
    *   **优势**：无需更新模型权重即可实现知识积累和自我改进，逐步减少对主导语言先验知识的依赖，有效缓解了本地化场景下的幻觉问题。

### 相比现有方法的优势
| 维度 | 现有方法 | DeepEdu-v1 / SCALE |
| :--- | :--- | :--- |
| **数据主权** | 云服务（如 ChatGPT）不满足 | ✅ 自托管，数据不出境 |
| **长上下文效率** | 传统方法存在高 KV Cache 和二次方延迟 | ✅ SCR 显著降低 TTFT (~35%) |
| **本地化知识** | 微调成本高且易灾难性遗忘 | ✅ 通过 Playbook 动态累积，无微调 |
| **自我改进** | 静态模型或需重新训练 | ✅ 通过交互持续进化 |

---

## 2. 核心实验方法和设置

### 数据集
- **长上下文推理基准**：
  - **InfiniteBench**：包含代码调试（Code.D）、结构化检索（R.KV, R.Num, R.PK）、叙事对话问答（En.Dia）、数学题求解（Math.F）等任务。
  - **RULER**：针对于“大海捞针”（Needle-in-a-Haystack, NIAH）任务，测试不同长度上下文（4K 到 128K tokens）下的检索准确性。
- **智能体推理基准**：
  - **Formula & FiNER**：金融推理任务，用于评估在离线学习预算下的表现。
  - **AppWorld**：交互式多步任务环境，模拟真实世界的应用操作，用于评估智能体在复杂、易错环境中的表现。

### 实验设置与评估指标
- **硬件**：单张 NVIDIA H100 GPU (80GB)。
- **主干模型**：
  - 长上下文实验：`Qwen2-7B-Instruct`
  - 智能体实验：`Qwen3-4B-Instruct-2507`
- **评估指标**：
  - **Accuracy**：任务准确率。
  - **TTFT (Time-To-First-Token)**：首词生成时间，衡量 prefill 阶段延迟。
  - **TPOT (Time-Per-Output-Token)**：每个输出词的时间，衡量 decode 阶段速度。
  - **Task Goal Completion (TGC) / Scenario Goal Completion (SGC)**：AppWorld 中的任务完成度。

### 基线方法对比
- **长上下文推理**：与当前最先进的 **TokenSelect** 方法进行对比。
- **智能体框架**：与原始的 **ACE (Agentic Context Engineering)** 架构进行消融对比，并逐步加入 RAE、FMB 和对抗性课程（Adversarial Curriculum）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
#### 长上下文推理（SCR）
- **延迟（TTFT）**：
  - 在 InfiniteBench 上，相比 TokenSelect，SCR 将平均 TTFT **降低了约 35%**（例如，在 R.KV 任务上从 13.2s 降至 8.3s）。
  - 在 RULER 的 128K 长度下，TTFT 从 9.89s 降至 6.64s，**加速 32.9%**。
- **准确率**：
  - 在结构化检索任务（R.KV）上，SCR 达到了 **98.0%** 的准确率，显著优于 TokenSelect 的 91.0%。
  - 其他任务上准确率持平或略有提升。
- **检索调用**：SCR 将检索调用次数减少了 **7.7×**，验证了其摊销效果。

#### 自改进智能体层
- **金融推理（Formula & FiNER）**：
  - 基线（Origin）准确率为 70.0% (Formula) 和 51.3% (FiNER)。
  - 加入 RAE 后，Formula 准确率升至 **77.5%**。
  - 最终完整配置（含 RAE, FMB, Adversarial）使 Formula 准确率达到 **79.5%**。
- **交互式智能体（AppWorld）**：
  - 完整配置在 AppWorld 上取得了最佳的综合表现（Average: 10.4），尤其是在正常任务的 TGC 上达到 **22.0**。

#### 端到端性能（DeepEdu 部署配置）
- 当将自改进智能体层运行在 SCR 引擎之上时：
  - 在 AppWorld 上，**平均 TTFT 实现了近 2× 的加速**（从 ~12s 降至 ~5.5s，即 **2.17×** 快速）。
  - 这表明两个组件协同工作，共同解决了计算和语义双重瓶颈。

### 消融实验结果
- **SCR 超参数**：
  - `Lmax`（最大簇大小）越大，延迟越低，但在某些任务（如 Code.D）上过大的簇可能轻微影响精度。
  - 相似性阈值 `θ` 对性能影响较小，`θ=0.95` 为默认推荐值。
- **智能体机制**：
  - **RAE**：通过检索相关规则，显著提升生成效率和准确性。
  - **FMB**：通过提供类似的历史失败案例，帮助 Reflector 更好地诊断问题，尤其在交互式任务中作用明显。
  - **Adversarial Curriculum**：主动暴露剧本弱点，是提升性能的关键，但其收益需要通过 FMB 才能持久化。

---

## 4. 关键结论和发现

### 主要发现
1.  **长上下文优化与知识本地化可以协同解决**：SCALE 框架成功地将高效的推理引擎（SCR）与动态的知识积累机制（自改进智能体）结合，为资源受限地区的本地化 AI 教育提供了可行方案。
2.  **查询相似性是优化长上下文的关键**：连续文本块的查询表示具有高度相似性，利用这一特性进行聚类和摊销，可以在不牺牲准确率的前提下大幅提升推理效率。
3.  **无需微调也能实现自我改进**：通过结构化的上下文工程（Playbook）和反思机制，LLM 智能体可以在不修改模型权重的情况下，持续积累和修正知识，有效应对本地化挑战。
4.  **系统性弱点会反复出现**：分析发现，智能体的失败并非随机，而是由系统性弱点导致，这些弱点会在不同任务中重现。因此，主动的压力测试（Adversarial Curriculum）和失败记忆（FMB）至关重要。

### 局限性
- **本地化评估不足**：实验主要在通用基准（如 InfiniteBench, AppWorld）上进行，尚未直接在越南本土的真实教材和课程上进行全面评估。
- **计算开销**：虽然推理高效，但智能体的自改进循环（Generator-Reflector-Curator）本身在每次交互后都需要额外的计算开销。
- **知识覆盖范围**：Playbook 的质量依赖于初始数据和交互样本，可能存在知识盲区。

### 未来工作方向
- **在真实教育场景中部署和评估**：将 DeepEdu-v1 应用于越南学校的实际教学环境中，收集真实学生数据，直接验证其在本地化课程上的效果。
- **扩展到更多语言和文化**：将 SCALE 框架推广到其他低资源语言（Low-Resource Languages, LRLs）和地区，进一步推动 AI 教育的民主化。
- **优化自改进循环**：探索更高效的反思和剧本更新机制，降低自我改进过程的计算成本。

</details>

---

### 2. [Block Sparse Attention with Log-Linear Complexity](https://arxiv.org/abs/2609.31093)

**Authors**: Bohao Tang, Zhen Qin, Yuqi Pan, Zheng Li, Pengfei Liu  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.31093v1  

#### Abstract
Scaling language models to long contexts is limited by the quadratic cost of self-attention. Block sparse attention offers an efficient alternative, but selecting the retained blocks remains a bottleneck. Conventional block selection requires scoring all query-block pairs and therefore remains quadr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Block Sparse Attention with Log-Linear Complexity**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
传统的 **self-attention** 机制在长序列建模中面临 **计算复杂度为 $O(N^2)$** 的瓶颈，其中 $N$ 是序列长度。尽管 **block-sparse attention (BSA)** 通过将 key 分组为块并仅选择 Top-K 块来降低注意力计算成本，但其 **Top-K 块选择阶段仍需对每个 query 与所有 key 块进行打分**，导致选择阶段的复杂度仍为 $O(N^2)$，成为长上下文建模中的主要瓶颈。

### **提出了什么新方法或新思路**
本文提出 **PISA (Pyramid Sparse Attention)**，一种基于 **金字塔 Top-K 选择策略** 的 block-sparse attention 方法，旨在将块选择的复杂度从 $O(N^2)$ 降至 **$O(N \log N)$**。

其核心思想是：
- 构建一个 **由细到粗（fine-to-coarse）的 key 块层级结构**，通过池化（如均值池化）逐层聚合 key 块，形成 $O(\log N)$ 层的金字塔。
- 采用 **自顶向下（coarse-to-fine）的选择策略**：从最粗粒度层开始，使用 **LogSumExp (LSE)** 打分机制对有限候选块进行 Top-K 选择；然后将选中的块展开为其子块，作为下一层的候选，逐步细化至原始 key 块。
- 整个过程避免了对所有细粒度块的穷举打分，显著降低了选择开销。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **复杂度** | 将块选择复杂度从 $O(N^2)$ 降至 $O(N \log N)$，支持更长序列（如 256K tokens）的高效处理。 |
| **效率** | 在长序列（>64K）上，PISA 比 BSA 快 **2.86× 至 9.95×**。 |
| **实现优化** | 设计了硬件感知的 **Triton 内核**，融合多级路由与 LSE 打分，避免显式构建 query-key 得分矩阵，减少内存访问。 |
| **灵活性** | 支持训练和推理的不同内核设计（两阶段 vs 单阶段），适配不同场景。 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **预训练数据**：通用文本语料（未具体命名，但用于语言建模任务）。
- **下游评估任务**：
  - **语言建模**：WikiText、LAMBADA
  - **常识推理**：BoolQ、PIQA、HellaSwag、WinoGrande、ARC-e/c、OBQA、SIQA
  - **检索任务**：SWDE、SQuAD Completion、FDA、TriviaQA、Natural Questions、DROP
  - **长上下文检索**：RULER 中的 needle-in-a-haystack 任务（single-key, multi-key, multi-query, multi-value）
- **诊断实验**：使用 Full Attention 模型提取的 query 和 key 张量进行块选择质量分析。

### **实验设置和评估指标**
| 设置项 | 描述 |
|--------|------|
| **模型规模** | 418M、1.47B、2.67B 参数的 decoder-only 模型 |
| **序列长度** | 预训练：4K；继续预训练（CPT）：16K；测试可达 256K |
| **块大小 C** | 64 |
| **块预算 K** | 8（预训练）、32（CPT） |
| **评估指标** | - 语言建模：Perplexity<br>- 下游任务：Accuracy（acc）<br>- 检索任务：Containment Accuracy<br>- 长上下文：Needle-in-a-haystack 准确率<br>- 效率：Block-selection latency（ms） |

### **基线方法对比**
| 方法 | 类型 | 复杂度（Prefill） | 是否可训练 |
|------|------|------------------|------------|
| **Full Attention** | 全注意力 | $O(N^2)$ | √ |
| **BSA** | 块稀疏注意力 | $O(N^2)$ | × |
| **NSA** | 可训练稀疏注意力 | $O(N^2)$ | √ |
| **HiLS** | 可训练稀疏注意力 | $O(N)$ | √ |
| **PISA (Ours)** | 金字塔稀疏注意力 | $O(N \log N)$ | √ |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
#### **语言建模与常识推理（Table 2）**
- 在 **Perplexity** 和 **Multiple-choice Accuracy** 上，PISA 与 BSA、NSA、HiLS 表现相当，略优于或接近最优。
- 在 **Containment Accuracy** 上，PISA 在三个模型尺度上均取得 **最高的平均准确率**，表明其在检索相关 token 方面更具优势。

#### **长上下文检索（Table 3, RULER）**
- 在 2.67B 模型、16K 上下文长度下：
  - **Single-key**: PISA 达到 **99.07%**，接近 Full Attention 的 99.67%
  - **Multi-key**: PISA 达到 **94.60%**，优于 BSA 的 93.60%
  - **Multi-query**: PISA 达到 **96.33%**，优于 BSA 的 90.00%
  - **Multi-value**: PISA 达到 **91.47%**，优于 BSA 的 85.33%
- **总体平均准确率（Avg）**：PISA 为 **62.80%**，优于 BSA 的 61.69%，且在长序列（8K, 16K）上优势更明显。

#### **块选择效率（Figure 3）**
- 在 64K、128K、256K 序列长度上，PISA 相比 BSA 的加速比分别为：
  - **64K**: 2.86×
  - **128K**: 5.31×
  - **256K**: 9.95×
- 表明 PISA 在超长序列上具有显著的效率优势。

### **消融实验结果**
#### **打分函数对比（PISA vs PISA-1 vs PISA-2）**
- **PISA-1**：使用一阶近似（均值打分）
- **PISA-2**：使用二阶近似（均值 + 方差）
- **PISA**：使用完整的 LogSumExp 打分
- 结果显示：
  - PISA 的 **训练损失更低**，**containment 准确率更高**
  - 说明 **LSE 打分能更准确地捕捉 key 块的重要性**，优于简单均值或低阶近似。

#### **块选择质量诊断（Figure 2 & Table 7）**
- 在相同 query 和 key 输入下，评估不同方法选出的块与全注意力参考块的重叠度：
  - **Recall@8**：PISA 达到 **90.95%**，显著高于 BSA 的 85.91%
  - **Attention mass ratio**：PISA 达到 **99.46%**，接近理论上限
- 表明 PISA 能更精准地保留高注意力质量的 key 块。

---

## 4. **关键结论和发现**

### **主要发现**
1. **金字塔 Top-K 选择策略有效降低了块选择复杂度至 $O(N \log N)$**，突破了传统 block-sparse attention 的二次瓶颈。
2. **LogSumExp 打分机制优于均值或其他近似方法**，能更准确地估计 key 块的相关性。
3. **PISA 在保持语言建模性能的同时，在检索类任务上表现更优**，说明其选择机制更能捕捉长距离依赖。
4. **硬件感知的 Triton 内核实现了高效的端到端执行**，尤其在超长序列上展现出巨大速度优势。

### **方法的局限性**
- **计算资源限制**：实验仅在 2.67B 规模以下进行，更大模型上的收益可能受限于内存带宽或通信开销。
- **池化信息损失**：中间层使用池化摘要可能导致部分细粒度信息丢失，影响极端情况下的选择精度。
- **固定层级结构**：金字塔结构依赖固定池化因子（如 $g=2$），可能不适用于所有序列分布。

### **未来工作方向**
- 探索 **动态层级结构** 或 **自适应池化策略**，以更好匹配输入序列的内容分布。
- 将 PISA 扩展至 **encoder-decoder 架构** 或 **多模态模型**。
- 结合 **index reuse**（如 IndexCache）进一步减少跨层冗余计算。
- 在 **真实应用场景**（如长文档问答、代码生成）中验证其实际效果。

---

> **总结**：PISA 通过 **金字塔结构 + LogSumExp 打分 + 硬件优化内核**，成功将 block-sparse attention 的选择复杂度降至 $O(N \log N)$，在保持模型性能的同时显著提升长序列处理效率，为构建无限上下文语言模型提供了可行路径。

</details>

---

### 3. [Mentored Decoding: Faster Inference meets Boosting](https://arxiv.org/abs/2609.30474)

**Authors**: Vivien Tran-Thien, Richard Nock  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.30474v1  

#### Abstract
Speculative decoding is a successful technique speeding up inference of a target autoregressive language model via a fast drafter model. Lossy speculative decoding allows a drift with respect to the target to further improve speed. Interestingly, it has been observed experimentally that the resultin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Mentored Decoding: Faster Inference meets Boosting

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文旨在解决**大型语言模型（LLM）推理速度慢**的根本瓶颈。传统的 **Speculative Decoding (SD)** 技术通过一个快速的“草稿模型”（drafter model）来生成候选token，并由目标模型并行验证，从而加速推理。然而，SD存在一个根本性的限制：它必须严格保证最终输出分布与目标模型完全一致（即 $D=0$），这导致其**接受率（acceptance probability）被严格限制在草稿模型与目标模型分布的相似度之内**。

更进一步，尽管一些启发式方法（如Lenient SD）尝试放宽这一约束以提升速度，但这些方法缺乏理论基础，且无法解释一个令人惊讶的实验现象：有时允许一定偏差的组合模型，其**输出质量甚至能超越原始的目标模型**。

本文系统地解决了以下三个层面的问题：
1.  **如何在理论上突破SD的接受率上限？**
2.  **为什么允许偏差的模型组合反而可能提升质量？**
3.  **如何将推理加速与模型质量提升这两个看似矛盾的目标统一起来？**

### 提出了什么新方法或新思路
论文提出了 **Mentored Decoding (MD)** 框架，并将其与机器学习中的 **Boosting** 理论建立了深刻的联系。

#### 新方法：Mentored Decoding (MD)
- **核心思想**：将SD从一个“无损”过程推广为一个“有损”但受控的过程。MD不再要求输出分布与目标模型 $q$ 完全相同，而是引入一个**发散度约束**（divergence constraint）$D_f(\tau \| q) \leq D$，其中 $\tau$ 是最终的“导师化”（mentored）分布，$f$ 是任意的 $f$-divergence$。
- **优化目标**：在满足发散度约束的前提下，最大化草稿token的接受概率 $p'r$。目标模型 $q$ 在此扮演“导师”的角色，授权草稿模型 $p$ 在其附近进行探索，只要不偏离太远即可。
- **数学形式**：将MD问题形式化为一个带约束的优化问题：
  $$
  \text{MD}(p,q;D) = \arg\min_{r,s} \quad 1 - p'r \quad \text{s.t.} \quad D_f(p \odot r + (1-p'r)\cdot s \| q) \leq D
  $$

#### 新思路：连接 Inference 与 Boosting
这是本文最核心的创新。作者首次证明了MD框架不仅是一个加速工具，其内在机制与**Boosting**（一种经典的集成学习技术）高度契合。
- **第一路径（通用连接）**：证明了MD产生的复合输出 $\tau$ 隐含了一个具有Boosting特性的模型组合。这使得我们可以用Boosting的理论来分析和预测MD的质量提升潜力。
- **第二路径（TV特例下的直接构造）**：特别地，在使用 **Total Variation (TV)** 发散度时，最优解集具有独特的几何性质（超矩形与单纯形的交集）。利用这一特性，可以直接在MD的最优解集中“雕刻”出一个由草稿模型和目标模型组成的Boosted Ensemble，从而实现加速与质量的双重保障。

### 相比现有方法的优势
| 特性 | 传统 Speculative Decoding (SD) | 启发式 Lossy SD (e.g., Lenient) | 本文 Mentored Decoding (MD) |
| :--- | :--- | :--- | :--- |
| **理论基础** | 有（但仅限于 $D=0$） | 通常无 | **强**（基于凸优化和 $f$-divergence） |
| **接受率上限** | 被 $D_f(p\|q)$ 严格限制 | 可能更高，但不可控 | **可突破**，且与 $D$ 成平滑关系 |
| **质量保证** | 与目标模型 $q$ 完全相同 | 未知，可能下降 | **可提升**，通过Boosting机制 |
| **通用性** | 仅针对特定 $f$ | 通常为特定启发式 | **通用**，适用于所有 $f$-divergence |
| **算法效率** | 高效 | 高效 | **高效**，提出 $O(\text{sort}(n))$ 预处理 + $O(\log n)$ 查询 |

## 2. 核心实验方法和设置

值得注意的是，本文是一篇**理论性极强**的论文，其“实验”部分主要是**理论推导和数值模拟**，而非在真实下游任务（如文本分类、问答）上的大规模基准测试。

### 使用了哪些数据集
论文并未使用标准的NLP数据集（如GLUE, SQuAD等）。其“实验”基于以下两种设置：
1.  **合成数据集**：生成均匀随机的草稿分布 $p$ 和目标分布 $q$，维度 $n=100$ 或 $n=1000$。
2.  **模拟数据集**：让 $p$ 和 $q$ 服从离散化的Beta分布，以模拟更真实的概率分布形态。

### 实验设置和评估指标
- **核心设置**：固定一对 $(p, q)$ 分布，研究MD框架下不同参数的影响。
- **关键评估指标**：
  1.  **接受率 (Acceptance Probability, $P_{\text{acc}}(\text{MD})$)**：衡量推理速度的关键指标。
  2.  **发散度 (Divergence $D$)**：衡量输出 $\tau$ 与目标 $q$ 偏离程度的指标。
  3.  **Boosting Bound Coefficient $k$**：一个理论指标，用于衡量MD输出在Boosting框架下的质量潜力。$k$ 越接近1，表明质量越有保障。

### 基线方法对比
- **主要基线**：**Speculative Decoding (SD)**，即 $D=0$ 的情况。
- **对比方式**：绘制 $D$ 与 $P_{\text{acc}}(\text{MD})$ 的关系曲线，以及 $k$ 与 $P_{\text{acc}}(\text{MD})$ 的关系曲线，并与SD的点 ($P_{\text{acc}}(\text{SD}), D=0$) 进行比较。

## 3. 主要实验结果和性能指标

### 关键性能数据
1.  **接受率显著提升**：在保持发散度 $D$ 极低（接近0）的情况下，MD可以将接受率 $P_{\text{acc}}$ 提升超过 **10%**，甚至达到 **25%** 的提升（见Table 2, 3）。
2.  **发散度增长缓慢**：对于大多数在 $z=1$ 处可微的 $f$-divergence$（如KL, rKL, Hellinger），当 $D$ 从0开始增加时，其右导数为0（即 $(D_f)'(P_{\text{acc}}(\text{SD})) = 0$）。这意味着在SD的最小接受率附近，**可以以极小的发散度代价换取显著的接受率提升**。
3.  **高质量潜力**：Boosting系数 $k$ 在接受率大幅提升时仍能**非常接近1**（例如，当接受率提升10%以上时，$k$ 仍在0.95以上，见Table 5）。这从理论上证明了MD在加速的同时，其输出质量有潜力媲美甚至超越目标模型。

### 与基线方法的对比结果
- **速度对比**：MD的接受率 $P_{\text{acc}}(\text{MD})$ **显著高于** SD的 $P_{\text{acc}}(\text{SD})$。
- **质量潜力对比**：MD在 $(P_{\text{acc}}, D)$ 平面上的轨迹，相比于SD的一个固定点，提供了一条**帕累托前沿**（Pareto front）。用户可以根据需求在“速度”和“保真度”之间进行权衡，而MD始终优于SD。

### 消融实验结果
论文没有进行传统意义上的消融实验，但其理论分析本身就包含了对不同组件的深入剖析：
- **不同 $f$-divergence$ 的影响**：Table 2 和 Table 3 展示了多种 $f$-divergence$（KL, rKL, Hellinger, Neyman, Pearson, TV2, Amari）下 $D$ 与 $P_{\text{acc}}$ 的关系，证明了核心结论（右导数为0）的普适性。
- **Breakpoints 数据结构的有效性**：Table 4 展示了通过 `QUERYCBREAKPOINTS` 算法得到的 $(a, b)$ 参数如何精确控制 $D$ 和 $P_{\text{acc}}$，验证了该数据结构的正确性和实用性。

## 4. 关键结论和发现

### 论文的主要发现
1.  **理论证实了“加速且提质”的可能性**：论文首次从理论上证明了，通过 **Mentored Decoding** 框架，可以在**加速LLM推理的同时，不损害甚至有可能提升输出质量**。这解释了先前文献中观察到的反直觉实验现象。
2.  **MD与Boosting的深刻联系**：MD不仅仅是SD的简单扩展，其本质与Boosting训练框架相通。MD产生的复合模型可以被视为一个经过Boosting增强的集成体，这为理解其质量提升提供了坚实的理论依据。
3.  **TV发散度的特殊地位**：在所有 $f$-divergence$ 中，**Total Variation** 发散度具有独特的几何优势。其最优解集是一个“大而好”的集合，不仅包含了所有其他严格凸发散度的最优解，还为直接构造Boosted Ensemble提供了便利。
4.  **高效的算法实现**：论文提出的 **Breakpoints** 数据结构（大小 $\leq n$，构建时间 $O(\text{sort}(n))$）是革命性的。它将每个token上复杂的非线性优化问题，简化为一次 $O(\log n)$ 的查询操作，使得MD在实践中几乎零开销。

### 方法的局限性
1.  **依赖于草稿-目标模型对**：MD的效果高度依赖于草稿模型 $p$ 和目标模型 $q$ 的互补性。如果两者差异过大或过小，MD的增益可能会受限。
2.  **理论假设**：论文的许多优美结论（如唯一最优解、简单clamp形式）建立在 $f$ 严格凸等理想假设之上。在实际应用中，需要考虑更复杂的情况。
3.  **端到端质量评估缺失**：论文的“实验”是理论性的，缺少在真实世界任务上与人类评估或标准NLP指标（如BLEU, ROUGE）的直接对比，以全面验证其质量提升。

### 未来工作方向
1.  **扩展到多模型场景**：当前MD主要针对“1个草稿模型 + 1个目标模型”。未来可探索如何将MD扩展到多个草稿模型或多个目标模型的复杂场景。
2.  **动态调整发散度预算 $D$**：研究如何根据输入提示（prompt）的难度或上下文动态地调整 $D$，以实现更智能的加速。
3.  **结合模型训练**：探索在模型训练阶段就考虑MD的兼容性，例如设计专门用于MD的草稿模型，使其与目标模型的配合达到最优。
4.  **在真实系统中部署**：将MD算法集成到主流的LLM推理引擎（如vLLM, TensorRT-LLM）中，进行全面的端到端性能和质量评测。

</details>

---

### 4. [Momentum-Guided Federated Split Distillation for Personalized Temporal Edge Intelligence](https://arxiv.org/abs/2609.31159)

**Authors**: Ahmed-Rafik Baahmed (LINEACT), Jean-Fran\c{c}ois Dollinger (LINEACT), Amine Brahmia (LINEACT), Mourad Zghal (LINEACT)  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.31159v1  

#### Abstract
We propose a momentum-guided federated split distillation framework for personalized, efficient, and autonomous temporal edge intelligence. We introduce TeRR-SAtt, our novel temporal reservoir student attention design that combines fixed reservoir representations, a lightweight temporal student, and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Momentum-Guided Federated Split Distillation for Personalized Temporal Edge Intelligence**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

该论文针对**个性化时序边缘智能**（Personalized Temporal Edge Intelligence）中的三大挑战：

- **资源受限**：IoT设备计算、内存、能耗有限，难以部署大型时序模型。
- **训练-推理不匹配**：传统的 Federated Split Learning（FSL）在推理阶段仍依赖服务器端组件，限制了边缘自主性。
- **客户端异构性**：不同边缘节点具有不同的时序分布、目标和学习轨迹，单一全局模型无法满足个性化需求。

### 🚀 **提出了什么新方法或新思路**

作者提出了一种**动量引导的联邦分馏框架**（Momentum-Guided Federated Split Distillation），包含两个核心组件：

#### （1）**TeRR-SAtt**（Temporal ReseRvoir Student Attention）
- 一种轻量级、可独立部署的边缘学生模型设计。
- 包含三个部分：
  - **固定储层模块**（Reservoir）：低成本生成时序表示，无需训练。
  - **轻量时序学生网络**（Student）：本地训练，用于建模时间动态。
  - **个性化输出模块**（Output Module）：适配各客户端特定任务。
- 采用 **Split Distillation 架构**：服务器端仅在训练时提供高容量**时序教师模型**（Temporal Teacher），不参与推理，实现完全的边缘自主。

#### （2）**AMGF**（Anticipatory Momentum-Guided Fusion）
- 一种基于**学习动量**（Learning Momentum）的服务器端融合机制。
- 创新流程：
  1. 跟踪每个客户端对教师模型的梯度更新请求。
  2. 使用**路径感知动量**（path-aware momentum）平滑噪声梯度。
  3. 构造**动量亲和矩阵**（momentum-affinity matrix），通过 Affinity Propagation 进行聚类，发现具有兼容学习轨迹的客户端群组。
  4. 对每个群组生成**群组专用的教师更新**（cluster-specialized teacher update）。
  5. 引入**自适应前瞻系数**（adaptive anticipation coefficient），根据群组内一致性（Agreement）和时间稳定性（Consistency）动态控制“超前学习”强度。

> 💡 **核心思想**：不是所有客户端都应共享同一个教师更新；而是将学习轨迹相似的客户端聚为一组，为其定制化指导，并预测其未来学习方向以加速收敛。

### ⚖️ **相比现有方法的优势**

| 维度 | 传统方法缺陷 | 本文优势 |
|------|--------------|----------|
| **部署自主性** | FSL 推理需服务器参与 → 高延迟、低鲁棒性 | TeRR-SAtt 全本地推理 → 自主、低延迟 |
| **个性化能力** | 全局聚合抑制个体差异 → 性能下降 | AMGF 按学习轨迹聚类 → 更好保留个性 |
| **更新稳定性** | 瞬时梯度噪声大 → 收敛慢 | 动量平滑 + 可靠性门控 → 更稳定优化 |
| **协作效率** | 所有客户端强制同步 → 冲突更新干扰 | 分组协作 + 预见性更新 → 加速收敛 |

---

## 2. **核心实验方法和设置**

### 📊 **使用的数据集**

- **LBNL Building Dataset** [38]：来自劳伦斯伯克利国家实验室的真实智能建筑数据。
- 场景：**多区域温度预测**。
- 数据规模：20个边缘设备（对应20个热区），每个设备采集三项特征：
  - 风扇转速（Air fan speed %）
  - 温度（Temperature °F）
  - 热水阀位置（Heating-water valve position %）
- 任务：从过去72步窗口预测未来6步。

### 🔧 **实验设置**

- **硬件平台**：Raspberry Pi 5（8GB RAM），模拟资源受限边缘设备。
- **训练配置**：
  - Batch Size: 16
  - 更新轮次：87轮
  - 通信轮次：10轮（用于学习性能分析）
- **评估方式**：Walk-forward 验证（滑动窗口）

### 🎯 **评估指标**

| 类别 | 指标 |
|------|------|
| **效率指标** | Training Latency, Inference Latency, CPU Usage, Memory Usage |
| **性能指标** | RMSE（Root Mean Square Error） |
| **消融实验** | 不同 AMGF 配置下的 RMSE 对比 |

### 🆚 **基线方法对比**

| 方法 | 描述 |
|------|------|
| **Traditional FL** | 完整模型部署于边缘（GRU + Attention + Dense） |
| **Traditional FSL** | 将 Attention 模块卸载至服务器，其余在边缘 |
| **FSL-KD** | 在 TeRR-SAtt 架构下使用标准全局教师更新（Eq. 4）作为蒸馏基线 |
| **AMGF (w/o anticipation)** | 移除前瞻机制（α_max=0） |
| **Full AMGF** | 完整版本（β=0.4, η=0.1, α_max=0.2） |

---

## 3. **主要实验结果和性能指标**

### 📈 **TeRR-SAtt 边缘效率提升**

| 指标 | 相比 Traditional FL | 相比 Traditional FSL |
|------|---------------------|------------------------|
| **Training Latency** ↓ | **-65.50%** | **-53.70%** |
| **Inference Latency** ↓ | **-44.70%** | 显著更低（FSL需通信+服务端计算） |
| **Training Memory Usage** ↓ | **-18.40%** | **-11.40%** |
| **Inference CPU Usage** ↓ | **-33.10%** | **-14.50%** |
| **Inference Memory Usage** ↓ | ~10% 降低 | 接近 FSL 本地部分，但无服务器依赖 |

> ✅ **结论**：TeRR-SAtt 在保持高性能的同时，显著降低了边缘资源消耗，并实现了**全自主推理**。

---

### 📉 **AMGF 学习性能提升（RMSE 对比）**

| 配置 | 最大 RMSE 改进（vs FSL-KD） | 平均 RMSE 改进（20 clients, 10 rounds） |
|------|-------------------------------|-----------------------------------------|
| **AMGF (no anticipation)** | **31.49%** | 0.51% ± 7.48% |
| **Full AMGF** | **35.31%** | **5.14% ± 6.14%**（最高达 18.10%） |

#### 🔍 **代表性边缘节点表现（Fig. 4）**

| 节点 | 特征 | AMGF 表现 |
|------|------|-----------|
| **z58 / z68** | 与全局动态严重偏离 | AMGF 显著优于其他方法，尤其在第6轮 RMSE 降至 0.18（Trad-FL/FSL 为 0.48） |
| **z27 / z69** | 高个性化需求客户 | AMGF 有效隔离并服务其独特学习路径，避免被全局平均拖累 |
| **z41** | 学习轨迹突发性强 | 自适应前瞻仍能带来 28.53% 改进，显示鲁棒性 |

> ✅ **消融实验证明**：
> - **路径感知动量聚类** 是性能提升的基础（+31.49%）。
> - **可靠性门控的前瞻机制** 进一步加速收敛（额外 +3.82%）。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **TeRR-SAtt 成功解耦训练与部署**：
   - 实现了高效、轻量、**完全自主的边缘推理**。
   - 通过固定储层 + 轻量学生结构，在极低开销下完成高质量时序建模。

2. **AMGF 实现了“智能协作”而非“盲目聚合”**：
   - 基于学习动量发现兼容客户端群组，避免冲突更新。
   - 群组专用教师更新显著提升个性化学习效果。
   - 自适应前瞻机制在方向可靠时主动推进学习，加快收敛。

3. **动量不仅是优化工具，更是协作信号**：
   - 学习动量可用于揭示客户端间的潜在协同关系。
   - 为非IID场景下的联邦学习提供了新的**协作发现范式**。

4. **效率与性能可兼得**：
   - TeRR-SAtt + AMGF 同时实现了**资源节省**与**精度提升**，打破了“轻量即低效”的权衡。

---

### ⚠️ **局限性**

1. **服务器端计算负担增加**：
   - AMGF 中的动量亲和矩阵构建和 Affinity Propagation 聚类具有 $O(N^2)$ 复杂度，可能影响大规模系统扩展性。
2. **依赖教师模型质量**：
   - 教师模型的设计和容量直接影响蒸馏效果，当前未深入探讨教师架构选择。
3. **动态聚类可能导致不稳定分配**：
   - 客户端在不同轮次可能属于不同群组，长期一致性有待研究。

---

### 🔮 **未来工作方向**

1. **扩展到更多应用场景**：
   - 如工业 IoT、医疗边缘计算等更复杂的时序任务。
2. **层次化动量建模**：
   - 在组织、空间或多尺度层面建模动量，支持更复杂生态系统。
3. **利用动量减少通信开销**：
   - 探索是否可通过动量预测跳过某些通信轮次或压缩梯度传输。
4. **轻量化聚类算法替代 Affinity Propagation**：
   - 提升 $O(N^2)$ 聚类步骤的可扩展性，适用于千级客户端场景。

---

> 🏁 **总结一句话**：  
> 本文提出的 **TeRR-SAtt + AMGF** 框架，首次将**学习动量**用于联邦分馏中的**个性化协作发现与前瞻性知识传递**，实现了**高效、自主、精准**的时序边缘智能，在真实智能建筑数据上验证了其优越性。

</details>

---

### 5. [Learning Provable Neural Network Observer for Uncertain Dynamical Systems](https://arxiv.org/abs/2609.30819)

**Authors**: Zhangyi Wang, Jiaxu Liu, Chen Song, Chao Xu, Shengze Cai  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.30819v1  

#### Abstract
In many safety-critical applications, control of uncertain dynamical systems relies on observers that estimate states and external disturbances. Neural network observers can improve estimation accuracy, but certifying their Lyapunov stability via Linear Matrix Inequality (LMI) constraints leads to l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning Provable Neural Network Observer for Uncertain Dynamical Systems

## 1. 论文的主要贡献和创新点

### 解决的问题
在**安全关键系统**（safety-critical applications）中，对存在外部扰动和未建模动态的不确定动力系统进行状态估计至关重要。传统的观测器（如ESO、SMO）依赖线性近似或特定结构假设，在强非线性或模型失配时性能下降。虽然神经网络观测器（Neural Network Observer）能捕捉复杂不确定性，但其**Lyapunov稳定性认证**通常依赖于求解大规模半定规划（SDP），导致计算成本极高，难以扩展到深层网络。

### 提出的新方法
本文提出了一种**可证明稳定的两阶段训练框架**（two-stage training framework），用于训练具有全局LMI稳定性证书的神经网络观测器，具体分为：

- **Stage I: Point-Guided Lyapunov Pre-Training**  
  在采样的误差状态点上定义一个基于Lyapunov的损失函数 $ \mathcal{L}_{\text{point}}(\theta) $，通过随机梯度下降快速训练网络，使其在这些点上满足严格的Lyapunov下降条件。此阶段不涉及任何LMI或SDP求解，保留了标准神经网络训练的高效性和可扩展性。

- **Stage II: LMI-Based Fine-Tuning**  
  以Stage I得到的参数作为初始化，执行轻量级的LMI微调，通过优化一个基于最大特征值的惩罚函数 $ \mathcal{L}_{\text{LMI}}(\theta) = \phi(\lambda_{\max}(H(\theta))) $，最终使LMI约束 $ H(\theta) \preceq 0 $ 严格成立，从而获得全局Lyapunov稳定性证书。

### 相比现有方法的优势
- **显著提升训练效率**：避免了全程求解大规模SDP，训练时间大幅缩短（例如在X-29飞机任务中实现2.5倍加速）。
- **保持高表达能力**：支持深层、大容量网络，能够有效建模高频、复杂的不确定性。
- **提供形式化稳定性保证**：最终仍满足严格的LMI稳定性条件，适用于安全关键场景。
- **理论支撑强**：提供了Stage I局部稳定半径和概率覆盖的理论分析（Theorems 1 & 2），解释了点引导预训练为何能为LMI微调提供良好初始化。

---

## 2. 核心实验方法和设置

### 使用的数据集与仿真环境
论文在多个**非线性控制基准**上进行了验证，均为物理仿真环境：
- **X-29 Aircraft**：高机动飞行器状态空间模型，用于研究训练效率和消融实验。
- **Quad-UAV under Ground Effect**：四旋翼无人机在地面效应下的垂直起降任务，模拟强非线性气动扰动。
- **Autonomous Underwater Vehicle (AUV)**：自主水下航行器穿越由WaterLily引擎模拟的von Kármán涡街，测试在复杂流体扰动中的轨迹跟踪能力。

### 实验设置与评估指标
- **评估指标**：
  - **Mean Squared Estimation Error (MSE)**：状态估计误差。
  - **Tracking Error (m)**：实际轨迹与期望轨迹之间的欧氏距离。
  - **Success Rate**：轨迹保持在稳定管内的比例。
  - **Training Time (s)**：训练耗时，用于衡量计算效率。
- **网络架构**：使用ResNet风格的残差前馈网络，激活函数为 `tanh`。
- **优化器**：Adam优化器。
- **代码开源**：代码已公开于 GitHub：[https://github.com/Berry-Myon/LearningNeuralNetworkObserver](https://github.com/Berry-Myon/LearningNeuralNetworkObserver)

### 基线方法对比
- **Classical Baselines**：
  - **Basic PID / NMPC**：无扰动补偿的基础控制器。
  - **Extended State Observer (ESO)**：经典扰动观测器，如 Guo and Zhao [2011]。
- **Learning-Based Baselines**：
  - **Neural Lander**：Shi et al. [2019] 提出的任务特定前馈补偿器，需大量数据离线训练。
- **Training Strategy Baselines**：
  - **LMI-Only**：直接端到端优化LMI约束。
  - **Point-Only**：仅使用点引导预训练，不进行LMI微调。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### X-29 飞机训练效率（Table 1）
| Architecture | Total Neurons | Method | Time (s) |
|--------------|---------------|--------|----------|
| Large NN     | 576           | LMI Gradient Descent (Only) | 2210.39 |
| Large NN     | 576           | **Point & LMI Tuning (Ours)** | **895.31** |

> **结论**：所提方法相比纯LMI梯度下降快 **2.5倍**，且仍满足全局LMI证书。

#### 大型网络消融实验（Table 2）
| Training Method | Time (s) | MSE (m) |
|------------------|----------|---------|
| Point-Guided (Only) | 99.53 | 1.0645 |
| LMI Gradient Descent (Only) | 2210.39 | 0.0707 |
| **Point & LMI Tuning (Ours)** | **895.31** | **0.0499** |

> **结论**：两阶段方法在显著减少训练时间的同时，取得了**最低的MSE**，优于单一策略。

#### 四旋翼无人机地面效应任务（Table 3）
| Method | Landing Error (m) | Take-off Error (m) |
|--------|-------------------|--------------------|
| Basic PID | 2.2006 ± 0.1456 | 0.1255 ± 0.0266 |
| PID + Neural Lander | 0.5732 ± 0.1188 | 0.2444 ± 0.0297 |
| **PID + Neural Network Observer (Ours)** | **0.1471 ± 0.0602** | **0.0002 ± 0.0000** |

> **结论**：所提方法在着陆和起飞任务中均显著优于基线，尤其在起飞阶段实现**近乎完美跟踪**，且无需每任务重新训练。

#### AUV 流体扰动任务（Table 4）
| Method | Tracking Error (m) |
|--------|--------------------|
| Basic NMPC | 1.4932 ± 0.0199 |
| NMPC + ESO | 0.0959 ± 0.0088 |
| **NMPC + Neural Network Observer (Ours)** | **0.0499 ± 0.0019** |

> **结论**：相比经典ESO，跟踪误差进一步降低 **48.0%**，表明更强的扰动抑制能力。

---

## 4. 关键结论和发现

### 主要发现
1. **两阶段框架有效解耦了表达力与稳定性认证**：Stage I利用点引导快速获得高性能且局部稳定的初始解，Stage II通过轻量LMI微调恢复全局稳定性证书。
2. **理论分析支持实践有效性**：Theorem 1 和 2 表明，点引导训练能在每个成功样本周围诱导出**非平凡的局部稳定邻域**，并在足够采样下实现对紧致域的**概率全覆盖**。
3. **方法具备强泛化能力**：在Quad-UAV任务中，仅一次离线训练即可在不同任务（着陆/起飞）中表现优异，而Neural Lander等任务特定方法泛化性差。
4. **网络容量提升鲁棒性**：Appendix E的X-29鲁棒性实验表明，大容量网络（Large NN）在结构模型失配下仍能保持低跟踪误差和高成功率。

### 方法的局限性
- **Stage I 分析局限于紧致域**：局部稳定半径和概率覆盖的理论结果依赖于预设的紧致误差状态域 $ \mathcal{X} $。
- **依赖名义模型质量**：若名义模型与真实系统差异过大，或存在严重噪声/分布外扰动，估计精度和证书有效性可能下降。
- **离线认证成本仍存**：尽管相比直接求解SDP已有改进，但对于超大规模网络或高维系统，Stage II的LMI微调仍可能较慢。
- **有限样本保证开放**：点引导训练的成功率与所需样本量尚无有限样本理论保证。

### 未来工作方向
- 设计更保守、更可扩展的稳定性证书（less conservative and more scalable certificates）。
- 开展Stage I的**有限样本分析**（finite-sample analysis）。
- 引入运行时机制（runtime monitoring）、安全切换（safe switching）等部署感知机制，提升在线适应性。
- 将该框架推广至其他需LMI约束的神经网络控制系统，如**Neural Network Controller**（见Appendix D）。

</details>

---

### 6. [Benchmarking Attention for Tabular Foundation Models](https://arxiv.org/abs/2609.31306)

**Authors**: Maximilian Schambach, Clemens Biehl, Sam Thelin  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.31306v2  

#### Abstract
Tabular in-context learners such as TabPFN, Mitra, or ConTextTab rely on alternating row and column attention over 2D sequences of latent embeddings. These attention patterns differ markedly from the one-dimensional case in language models: row attention involves longer sequences while column attent...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Benchmarking Attention for Tabular Foundation Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Efficient Attention** 研究主要针对 **Large Language Models (LLMs)** 和 **Vision Transformers**，其注意力机制通常作用于一维序列或具有局部性的二维图像块。然而，**Tabular Foundation Models**（如 TabPFN、Mitra、ConTextTab）采用独特的 **2D 交替行-列注意力（alternating row and column attention）**，其数据特性显著不同：
- 行数（样本数）远大于列数（特征数），导致 **序列长度高度不对称**
- 隐藏维度（hidden dimension）较小（通常为 128–768），远小于 LLMs（数千）
- 数据存储为五维张量 `(B, R, C, H, D)`，其中行注意力需跨非连续内存维度操作，产生高昂的 `.contiguous()` 拷贝开销
- 注意力模式为双向、非因果（non-causal），不同于自回归解码器

这些特性使得现有的高效注意力实现（如 FlashAttention）在表格场景下表现不佳，且缺乏系统性基准测试。

### 🆕 提出的新方法与创新
本论文提出并实现了以下创新：

- **首个面向表格数据的注意力基准测试框架（Reproducible Benchmarking Setup）**  
  构建了一个可复现的微基准测试平台，专门用于评估不同 Attention 后端在真实表格形状下的前向与反向吞吐量。

- **揭示了“行优先”布局下的 Contiguity 开销是性能瓶颈**  
  发现行注意力因需要对非连续内存进行 `.contiguous()` 拷贝而引入显著开销，而列注意力则天然连续，无需拷贝。

- **推荐混合后端策略（Mixed Backend Strategy）以最大化性能**  
  不同 Attention 模式应选用不同最优后端：例如 cuDNN 适合列注意力，FlashAttention-3/4 更适合行注意力。

- **开源代码促进未来研究**  
  所有代码、配置和结果均公开于 GitHub：[https://github.com/SAP-samples/tabular-attention-benchmark](https://github.com/SAP-samples/tabular-attention-benchmark)

### 🔍 相比现有方法的优势
| 维度 | 传统做法 | 本文贡献 |
|------|--------|---------|
| **关注领域** | LLM/Vision 为主 | 首次聚焦 **Tabular-specific Attention** |
| **评估粒度** | 端到端模型训练 | 微内核级隔离测试，精确归因性能差异 |
| **硬件覆盖** | 单一 GPU | 跨三代 NVIDIA GPU（A100/H100/B200）全面评测 |
| **实用性指导** | 通用建议 | 提供基于模型结构、硬件、序列长度的具体选型指南 |

---

## 2. 核心实验方法和设置

### 📊 实验设置概览

| 项目 | 描述 |
|------|------|
| **目标任务** | 衡量 Row Attention 与 Column Attention 的前向/反向吞吐量 |
| **张量布局** | `(B, R, C, H, D)` —— 行优先（row-first）标准格式 |
| **典型形状** | - 列注意力：固定 `R=1024`，`C ∈ [16, 2048]`<br>- 行注意力：固定 `C=64`，`R ∈ [32, 131072]` |
| **默认配置** | `H=12`, `D=64`, `dtype=bfloat16`, `B=1` |
| **其他配置** | 包括 TabPFN (`H=6,D=32`)、TabICL (`H=8,D=16`)、ConTextTab (`H=12,D=64`) 等实际模型参数 |

### ⚙️ 评估的 Attention Backends

#### ✅ 支持训练（含反向传播）：
1. **SDPA (efficient)** – PyTorch 内存优化版（xFormers 风格）
2. **SDPA (cuDNN)** – 使用 NVIDIA cuDNN 库
3. **FlashAttention-2 (FA2)** – 针对 Ampere 架构优化
4. **FlashAttention-3 (FA3)** – 支持 strided tensor，专为 Hopper 设计
5. **FlashAttention-4 (FA4)** – Blackwell 架构定制，支持异步流水线

#### 🚀 推理专用（仅前向）：
6. **vLLM** – Triton 实现的预填充注意力
7. **SageAttention** – 量化近似注意力，高吞吐推理

### 📏 评估指标
- **吞吐量（Throughput）**：单位 TFLOPS，按公式计算：
  - 前向：`4 * Beff * H * L² * D / t`
  - 反向：取前向 FLOPs 的 2.5 倍
- **峰值内存消耗（Peak Memory Usage）**
- **延迟分布与方差（Mean ±1σ over repetitions）**

### 💻 硬件环境
| GPU | 显存 | 架构 |
|-----|------|-------|
| NVIDIA A100 (PCIe) | 80 GB HBM2e | Ampere |
| NVIDIA H100 (NVL) | 80 GB HBM3 | Hopper |
| NVIDIA B200 (NVL) | 192 GB HBM3e | Blackwell |

> 所有测试使用最新兼容版本：PyTorch 2.10 + CUDA 13.0 + cuDNN 9.15.1

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（H=12, D=64, bfloat16）

#### 🔹 列注意力（Column Attention）结果
| GPU | 最优后端 | 性能特点 |
|-----|----------|---------|
| **A100** | SDPA (cuDNN) ≈ FA3 | cuDNN 在中长序列（128–1k）领先，FA2 略逊 |
| **H100** | **SDPA (cuDNN)** | 在 128–512 列区间大幅优于 FA3（最高达 2×） |
| **B200** | **FA4**（短序列）、**cuDNN**（长序列） | FA4 在 ≤256 列时最快；超过后被 cuDNN 超越 |

> 💡 小结：**cuDNN 是列注意力最稳健选择**，尤其在 H100 上优势明显。

#### 🔹 行注意力（Row Attention）结果
| GPU | 最优后端 | 性能提升 |
|-----|----------|---------|
| **A100/H100** | **FA3** | 较 cuDNN 最高提速 **3.5×**（短序列） |
| **B200** | **FA4** | 全范围领先，但在 >8k 行时被 cuDNN 追平 |

> 💡 小结：**FA3/FA4 凭借对 strided tensor 的原生支持避免了 `.contiguous()` 拷贝，在行注意力上遥遥领先**。

#### 🔹 推理专用后端表现（H100）
- **SageAttention**：
  - 列注意力：性能较差（小序列下 kernel launch 开销大）
  - 行注意力：**在 >16k 行时表现出色**，尤其适合大规模样本推理
- **vLLM**：
  - 中规中矩，未超越专用后端

#### 🔹 消融实验：Head Dimension 影响（图3）
- **列注意力**：FA3 多数情况下慢于 cuDNN，除非 head dim ≥128
- **行注意力**：**FA3 相对于 cuDNN 的优势随 head dimension 增加而增强**
  - 当 `D=128` 时，最大可达 **2.4× 加速**
  - 原因：`.contiguous()` 拷贝时间正比于 `D`，而 FA3 可规避此开销

#### 🔹 Roofline 分析（图4）
- **列注意力**：处于 **memory-bound** 区域（OI ≈ 8–128），cuDNN 已接近理论带宽上限
- **行注意力**：进入 **compute-bound** 区域（OI > 256），FA3 因更优 kernel 设计胜出
- **拷贝开销估算**：将实测 `.contiguous()` 时间加入 FA3 延迟后，预测吞吐与 cuDNN 实测值高度吻合，验证了拷贝是主要瓶颈

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **没有“万能最优”后端**：最佳选择取决于：
   - 注意力方向（row vs column）
   - 序列长度
   - GPU 架构（Ampere/Hopper/Blackwell）
   - 模型参数（head dimension）

2. **推荐混合策略（Mixed Backend Strategy）**：
   - **列注意力** → 使用 **SDPA (cuDNN)**
   - **行注意力** → 使用 **FlashAttention-3 (H100)** 或 **FA4 (B200)**
   - 极短序列可考虑 SDPA (efficient)

3. **Contiguity 开销不可忽视**：
   - `.contiguous()` 拷贝在中等序列长度下主导运行时间
   - 支持 strided tensor 的 FA3/FA4 显著减少此类开销

4. **未来硬件趋势利好 strided tensor 支持**：
   - 随着 compute-to-bandwidth 比例上升，memory-bound 区域扩展，拷贝相对代价更高
   - 更凸显 FA3/FA4 类设计的重要性

### ⚠️ 局限性
- **仅测试孤立 kernel**：未涵盖完整模型训练/推理中的层间交互或多卡通信
- **固定精度与掩码类型**：仅使用 `bfloat16` 和非因果注意力，不适用于所有场景
- **批大小限制**：主要使用 `B=1`，大 batch 可能改变性能格局
- **部分 backend 缺失**：如 FA4 不支持 `D=16/32`，限制了某些模型的应用

### 🔮 未来工作方向
1. **开发专为表格优化的 Attention Kernel**  
   结合常见表格形状（如 `C≈64`, `R>>C`）设计轻量级、低拷贝开销的定制化 kernel。

2. **探索新型内存布局或缓存策略**  
   如动态转置、零拷贝视图重用、持久化缓存等，进一步降低内存移动成本。

3. **AI Agent 自动优化 FlashAttention 内核（已初步尝试）**  
   文中展示了使用 **Claude Code Agent** 对 FA4 内核进行自动优化，成功提升列注意力性能 11–14%，证明了闭环自动化调优的可行性。

4. **构建端到端 Tabular Model Benchmark Suite**  
   将微基准推广至完整模型（如 TabPFN、ConTextTab）的训练与推理全流程评估。

---

> **总结一句话**：  
> **Tabular Attention ≠ LLM Attention**。本文通过系统性基准测试揭示了表格注意力的独特挑战，并指出：**利用混合后端策略 + 重视 contiguity 开销 + 定制化 kernel 设计**，是提升 Tabular Foundation Models 效率的关键路径。

</details>

---

### 7. [Scaffold: Support Graph Theory Based Sparsification for Graph Neural Networks](https://arxiv.org/abs/2609.31466)

**Authors**: Siddhartha Shankar Das, Sai Karthik Navuluru, S M Ferdous, Ryan A. Rossi, Baris Coskunuzer, Lakshman Tamil, Edoardo Serra, Alex Pothen, Robert Rallo, Mahantesh M Halappanavar  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.31466v1  

#### Abstract
Graph neural networks (GNNs) rely on message passing over graph edges, making their computational and memory costs strongly dependent on graph density. Graph sparsification offers a natural way to reduce these costs, but removing edges indiscriminately can distort important communication structure a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Scaffold: Support Graph Theory Based Sparsification for Graph Neural Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
图神经网络（GNN）通过在图边上的消息传递进行学习，其计算和内存开销与图的密度强相关。对于大规模或稠密图，训练成本高昂。虽然图稀疏化（graph sparsification）可以降低这些成本，但简单地移除边会破坏重要的通信结构，导致预测性能下降。

现有方法多为采样（sampling）或基于语义重要性的稀疏化，缺乏对图拓扑结构的系统性控制，尤其在无监督场景下难以保证全局通信效率。

### **提出的新方法与新思路**
本文提出了 **Scaffold**，一种基于**支持图理论**（support graph theory）的无监督、拓扑感知的图稀疏化框架。其核心思想是：
- 将稀疏图视为原始图的“支持图”（support），保留的边构成支撑子图 $ H $，被移除的边由 $ H $ 中的路径来“表示”。
- 显式控制两个互补的结构性指标：
  - **Dilation**（拉伸度）：衡量被移除边的端点之间需要绕行的路径长度。
  - **Congestion**（拥塞度）：衡量支撑路径在保留边或节点上集中程度，避免形成通信瓶颈。
- 通过联合优化 dilation 和 congestion，Scaffold 在低边预算下构建出既能保持短通信路径、又能避免结构瓶颈的稀疏图。

### **相比现有方法的优势**
- **理论严谨性**：首次将支持图理论中的 dilation-congestion 准则引入 GNN 图稀疏化，具有坚实的组合预条件器理论基础。
- **可扩展性**：设计了五种变体（Greedy, Heap, Batch, Fast, Sample），可在不同质量-成本权衡下高效运行，适用于从小到大的图。
- **性能优越**：在19个基准数据集上，Scaffold 在仅保留 10%-50% 边的情况下，恢复或接近全图 GNN 性能，同时显著降低内存和端到端训练时间。
- **通用性强**：作为拓扑方法，不依赖标签或下游任务信号，适用于无监督场景，并兼容多种 GNN 架构。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖 **19 个节点分类基准数据集**，分为三类：
- **同质性图**（Homophilic）：如 Cora, CiteSeer, PubMed, Coauthor-CS, Amazon-Photo 等（8个）
- **异质性图**（Heterophilic）：如 Chameleon, Squirrel, Minesweeper, Questions 等（6个）
- **大规模图**（Large-scale）：如 Reddit, ogbn-products, ogbn-arxiv, Pokec 等（5个）

所有图均处理为无向、无自环图。

### **实验设置和评估指标**
- **稀疏化目标**：在给定边保留率 $ \delta \in (0,1] $ 下，生成稀疏图 $ H $，仅保留 $ q = \lfloor \delta m \rfloor $ 条边。
- **GNN 训练**：使用 TunedGNN 配置，在稀疏图上训练 GCN 模型，与全图训练对比。
- **评估指标**：
  - **准确性**（Accuracy）用于单标签分类。
  - **ROC-AUC** 用于多任务分类（如 ogbn-proteins）。
  - **平均排名**（Average Rank）用于综合比较多个方法在多数据集上的表现。
  - **内存峰值**（Peak Memory）和**端到端训练时间**（End-to-End Time）用于评估效率。

### **基线方法对比**
与以下类别方法对比：
- **拓扑稀疏化**（Topology-based）：
  - Random, Local Degree, Rank Degree, Forest Fire, SCAN, L-Spar, G-Spar, L-Sim, Spectral, t-Spanner, DSpar
- **语义稀疏化**（Semantic-based）：
  - Unified-LTH, AdaGLT, MoG
- **采样方法**（Sampling-based）：
  - Tuned-GraphSAGE, Tuned-GraphSAINT
- **参考方法**：
  - Full-Graph（全图）、Spanning Forest（生成树）、No-Graph（仅特征）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **性能恢复能力**：
  - 当稀疏图仅保留 **30%** 原始边时，Scaffold 在 **12/19** 个数据集上达到全图性能的 ±1% 内。
  - 当保留 **50%** 边时，该比例提升至 **16/19**。
- **效率提升**：
  - 内存占用减少超过一半。
  - 端到端训练时间（含稀疏化开销）显著低于全图训练，尤其在大图和长训练周期下优势明显。

### **与基线方法的对比结果**
- **综合排名最优**：
  - 如图5所示，在19个数据集、3种保留率下的 Wilcoxon-Holm 平均排名比较中，**Scaffold-1** 和 **Scaffold-K** 分别取得 **第1名** 和 **第2名**，显著优于所有基线。
- **具体数据集表现**：
  - 表2显示，在多数数据集上，Scaffold-1 和 Scaffold-K 的准确率/ROC-AUC 接近或超越全图性能，且普遍优于其他拓扑和语义方法。
  - 例如在 Reddit 上，Scaffold-1 在 $ \delta=0.3 $ 时达到 95.43±0.18，接近全图的 95.44±0.05；而 DSpar 仅为 95.22±0.01。

### **消融实验结果**
- **组件分析**（图9）：
  - 单独控制 **dilation** 已能显著提升性能，但联合控制 **dilation + edge congestion** 效果最佳，表明二者互补。
  - 节点拥塞（node congestion）贡献有限，但在特定构造示例中可缓解局部瓶颈。
- **变体对比**（图11）：
  - **Heap** 和 **Batch** 结构质量更高但耗时更长。
  - **Fast** 和 **Sample** 牺牲部分结构质量以换取极低的构建成本，适合大规模应用。
- **Scaffold-K vs. Scaffold-1**：
  - Scaffold-K 通过定期刷新稀疏图并集成验证最优的 $ K $ 个视图，进一步提升了性能，尤其在 ogbn-arxiv 和 Pokec 上增益显著。

---

## **4. 关键结论和发现**

### **主要发现**
1. **如何表示被移除的边至关重要**：在固定边预算下，被移除连接在稀疏图中是否由**短且非拥塞的路径**表示，直接决定了 GNN 的通信效率和最终性能。
2. **dilation 和 congestion 是关键指标**：联合最小化这两个量能有效保留图的通信结构，优于仅关注度数、相似性或谱性质的方法。
3. **Scaffold 具有普适性和高效性**：在同质/异质、小/大规模图上均表现优异，且可通过不同变体适应各种效率需求。

### **方法的局限性**
- **理论保证有限**：目前缺乏对 dilation-congestion 联合目标的全局近似比证明，且结构界不直接转化为 GNN 准确率单调提升。
- **静态图假设**：当前框架针对静态、无向、无权重图，未考虑动态图或有向图场景。
- **无任务感知**：作为无监督方法，无法利用下游任务信号进一步优化稀疏化策略。

### **未来工作方向**
- **拓展到任务感知稀疏化**：结合监督信号或元学习，实现任务导向的稀疏图构建。
- **支持更广的图学习场景**：扩展至链接预测、图分类、动态图等任务。
- **理论深化**：建立 dilation-congestion 目标与 GNN 泛化误差之间的理论联系。
- **支持局部化稀疏化**：探索基于社区或分区的局部稀疏化策略，可能更适合某些图结构。

---

> **开源代码**：`github.com/siddhartha047/Scaffold`  
> **GNN 集成**：`github.com/siddhartha047/Scaffold-GNN`

</details>

---

### 8. [Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency](https://arxiv.org/abs/2609.31619)

**Authors**: Parsa Hosseini, Akasha Tigalappanavara, Sumit Nawathe, Chenrui Fan, Sourya Basu, Genta Indra Winata, Anirban Das, Soheil Feizi, Nima Chitsazan  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.31619v1  

#### Abstract
Reasoning models often generate very long reasoning traces, making inference computationally expensive. Existing approaches typically improve efficiency either through inference-time early-stopping mechanisms or by explicitly encouraging shorter reasoning during training, for example through reinfor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
现代 **Reasoning Language Models** 在数学、科学和编程等复杂任务上表现出色，但其推理过程往往生成极长的 **Chain-of-Thought (CoT)** 轨迹，导致推理成本高昂。现有方法通常通过以下方式提升效率：
- **训练时优化**：如使用强化学习（RL）加入长度惩罚（length penalty）。
- **推理时机制**：如基于 confidence 或 answer stability 进行 early stopping。

然而，这些方法都**显式地将“效率”作为优化目标**，可能破坏模型的推理逻辑或引入额外的控制机制。

本论文提出一个根本性问题：  
> **能否在不直接优化推理长度或停止行为的情况下，让高效推理自然涌现？**

---

### **提出了什么新方法或新思路**
作者提出 **ConfSFT (Confidence-based Self-Supervised Fine-Tuning)**，一种全新的自监督微调方法，其核心思想是：

- **只监督 confidence 预测，不监督推理内容本身**。
- 模型在训练中学习预测自己在推理过程中间状态的 **confidence**（即对当前答案的置信度），而**不鼓励更短的推理或提前停止**。
- 推理时仍使用标准生成流程，**无任何 early-stopping 机制或 confidence 判断**。

#### **关键创新点**：
1. **自监督信号构建**：  
   - 从模型自身的 token probabilities 中计算 confidence（几何平均概率），无需黄金答案或外部标注。
   - 将 confidence 量化为文本百分比（如 "72%"），作为语言建模目标进行训练。

2. **仅监督 confidence，屏蔽其他内容**：  
   - 在训练中，只对 confidence 标签部分计算损失，其余推理内容被 mask，确保模型只学“自信”，不学“如何推理”。

3. **推理不变性**：  
   - 微调后的模型在推理时完全保持原生成流程，**没有 confidence elicitation、early exit 或任何干预机制**。

---

### **相比现有方法的优势**
| 维度 | 现有方法（如 RL + length penalty, DEER） | ConfSFT |
|------|----------------------------------------|--------|
| **是否显式优化效率** | 是 | 否 |
| **是否修改推理流程** | 是（early stopping） | 否 |
| **是否依赖黄金答案** | 部分需要（如 binary correctness） | 否（完全自监督） |
| **跨任务泛化能力** | 可能受限于训练分布 | 强（仅用数学题训练，泛化到科学、编码） |
| **准确性稳定性** | 可能因过早停止而下降（尤其编码任务） | 几乎保持原准确率 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **训练数据**：AIME 2000–2023 共 695 道数学题（仅 600 题用于实际训练）。
- **验证数据**：AIME 2024（30 题）。
- **测试基准**：
  - **数学**：AIME 2025, GSM8K
  - **科学**：GPQA-Diamond
  - **编程**：LiveCodeBench, HumanEval

> ⚠️ 注意：**仅在数学题上训练，却在科学和编程任务上评估效率增益**，体现强泛化性。

---

### **实验设置和评估指标**

#### **模型家族**
在四个不同规模的模型上验证：
- **Gemma-4-E2B** (4B)
- **Qwen3-4B** (4B)
- **Nemotron-Nano-8B** (8B)
- **GPT-OSS-20B** (20B)

#### **评估协议**
- 每题采样 16 条独立推理路径。
- 报告：
  - **Accuracy (Acc.)**：正确率（pass@1）
  - **Average Generated Tokens**：平均生成 token 数
  - **Token Reduction (%)**：相对于基线的 token 减少比例

#### **训练细节**
- **ConfSFT 流程**：
  1. 生成完整推理轨迹（rollout）。
  2. 在中间点（如 `Wait` 出现处）插入 answer-elicitation prompt，提取 trial answer。
  3. 计算该 answer 的 token 概率几何均值作为 confidence。
  4. 构造训练样本：`[prompt][reasoning prefix][priming prefix][confidence label]`。
  5. 仅对 confidence label 部分计算损失。

- **迭代训练**：每轮更新策略并刷新训练数据，共 8 轮。

---

### **基线方法对比**
| 基线 | 类型 | 描述 |
|------|------|------|
| **DEER (Yang et al., 2025)** | 推理时 early stopping | 基于 confidence 达到阈值（如 0.95）时提前终止 |
| **A&Z (Arora & Zanette, 2025)** | 训练时 RL + length penalty | 显式优化更短的推理 |
| **On-Policy SFT (Zhao et al., 2026)** | 训练时监督精简推理 | 对高质量短推理路径进行监督微调 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| 模型 | 方法 | Accuracy | Avg. Tokens | Token Red. |
|------|------|----------|-------------|------------|
| **Nemotron-8B** | Base | 42.5 | 11,325 | — |
| | **ConfSFT (Ours)** | **42.5** | **9,076** | **-19.9%** |
| **Gemma-4B** | Base | 35.0 | 7,284 | — |
| | **ConfSFT (Ours)** | **35.0** | **5,987** | **-17.8%** |
| **Qwen-4B** | Base | 59.2 | 13,268 | — |
| | **ConfSFT (Ours)** | **58.8** | **11,365** | **-14.3%** |
| **GPT-OSS-20B** | Base | 52.7 | 5,802 | — |
| | **ConfSFT (Ours)** | **53.3** | **5,235** | **-9.8%** |

> ✅ **核心发现**：**在几乎不损失准确率的前提下，token 数量减少 10–20%**。

---

### **与基线方法的对比结果**

#### **vs. 显式效率优化方法（A&Z, On-Policy SFT）**
- ConfSFT 的效率增益 **与这些显式优化方法相当甚至更好**。
- 例如在 Qwen 上：
  - On-Policy SFT：-17.2% token，准确率 69.3
  - **ConfSFT：-19.2% token，准确率 69.5**
- 说明：**仅学习 confidence 就能达到甚至超越专门优化效率的方法**。

#### **vs. 推理时 early stopping（DEER）**
- DEER 在某些任务上可减少更多 token（如 -85%），但**代价巨大**：
  - 在 HumanEval 上，Nemotron 的准确率从 89.7% 降至 47.1%。
  - 在 LiveCodeBench 上从 44.2% 降至 17.4%。
- **ConfSFT 在所有领域均保持高准确率**，无崩溃风险。

---

### **消融实验结果**

#### **(1) 监督信号 ablation（Table 2）**
测试不同标签的影响：
| 方法 | Gemma Token Red. | Nemotron Token Red. |
|------|------------------|---------------------|
| **ConfSFT (Ours)** | -11.7% | -9.7% |
| Position（仅位置） | -7.3% | +1.3% ❌ |
| Binary correctness（需黄金答案） | +5.9% ❌ | -2.9% |
| **Shuffled confidence**（打乱标签） | -0.1% | +4.0% ❌ |

> 🔍 结论：**只有保留原始 confidence 与状态的对应关系，才能带来效率增益**，说明不是简单微调的问题。

#### **(2) 决策点标记 ablation（Table 3）**
- 默认使用 `Wait` 作为中间状态标记。
- 改用段落分隔符 `\n\n` 也能取得效果（Gemma 上 -8.4% vs -10.3%）。
> 说明：**方法不依赖特定语义标记，而是通用的状态采样机制**。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **高效推理可以作为学习 metacognition 的副产品自然涌现**。
   - 仅通过训练模型预测自己的 confidence，就能显著缩短推理链。
2. ✅ **无需修改推理流程**：ConfSFT 不引入 early stopping，推理时完全透明。
3. ✅ **强泛化性**：仅在数学题上训练，却在科学和编程任务上获得效率提升。
4. ✅ **保持推理结构完整性**：分析表明，ConfSFT 并未抑制某类推理行为（如 Verify），而是整体压缩，**保留了 base model 的 reasoning composition**（见 Figure 5）。
5. ✅ **优于显式优化方法**：在准确率稳定性和跨任务表现上，ConfSFT 优于 RL-based 和 early-stopping 方法。

---

### **方法的局限性**
- **依赖中间状态采样**：需要合理定义“决策点”（如 `Wait` 或 `\n\n`），若模型不常输出此类标记，可能影响效果。
- **confidence 定义简化**：使用 token 概率的几何均值，非严格校准的置信度。
- **未探索多轮交互场景**：目前仅适用于单次生成任务。

---

### **未来工作方向**
1. **扩展到多步交互与工具调用**：在 agent 场景中学习何时调用工具或停止思考。
2. **结合 confidence 与动态 early exit**：在推理时利用已学习的 confidence 信号进行可控早停。
3. **探索其他 metacognitive 信号**：如不确定性估计、认知负荷、自我反思等。
4. **应用于更大规模模型与真实世界任务**：验证在生产环境中的实用性。

---

## ✅ 总结一句话
> **ConfSFT 证明：让模型学会“知道自己知道”，比直接教它“快点停下”更能自然地催生高效且可靠的推理。**

</details>

---

### 9. [HybridInfer: Thermal-Aware Reinforcement-Learning Tier Routing for On-Device, Edge, and Cloud LLM Inference](https://arxiv.org/abs/2609.30270)

**Authors**: Simran Koul  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.30270v1  

#### Abstract
On-device inference with small language models keeps user data local, works offline, and incurs no per-query cost, so the on-device tier is preferred when it is adequate. It is thermally constrained, however, and I find the constraint is sharper than a slowdown: on a flagship Snapdragon device, sust...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：HybridInfer: Thermal-Aware Reinforcement-Learning Tier Routing for On-Device, Edge, and Cloud LLM Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代旗舰智能手机已能支持在设备端运行小型语言模型（SLM），实现**on-device inference**，具有保护隐私、离线可用、无单次查询成本等优势。然而，持续的 on-device 推理会引发严重的**热稳定性问题**：
- 在高负载下，GPU推理栈（如OpenCL内核编译、prefill阶段）容易崩溃或“静默卡死”（silent wedge），即使设备温度尚可；
- 当前的 on-device 工具链（如MLC-LLM + Adreno OpenCL）对长文本生成和连续请求缺乏鲁棒性；
- 现有的 multi-tier 路由器大多忽略设备的**实时热状态**，仅基于查询复杂度或置信度进行路由决策。

因此，如何在保证服务质量的同时，避免因过热导致的系统不稳定，成为一个关键挑战。

### **提出了什么新方法或新思路**
作者提出 **HybridInfer** —— 一种**热感知的强化学习路由系统**，用于在三类LLM推理层级之间动态选择：
- **On-device tier**: Llama 3.2 3B（通过MLC-LLM部署于手机GPU）
- **Edge tier**: Llama 3.1 8B + RAG（本地局域网主机提供检索增强）
- **Cloud tier**: GPT-4o（云端API）

其核心创新在于：
- 将 **Android 的 `getThermalHeadroom` API 输出作为状态输入**，结合 query complexity bin 构成状态空间；
- 使用 **offline-trained Q-learning policy** 进行路由决策；
- 设计了一个包含 **locality bonus** 的奖励函数，显式鼓励在热条件允许时优先使用 on-device 模型；
- 首次将 thermal headroom 用作 capability-tier selection 的控制信号，并在真实 Android 手机上验证。

### **相比现有方法的优势**
| 维度 | 现有方法（如ConsRoute、EACO-RAG） | HybridInfer |
|------|-------------------------------|-----------|
| **Routing Signal** | 仅基于query difficulty / consistency，无thermal感知 | 引入 real-time thermal headroom |
| **Deployment Setting** | 多为仿真或非移动端硬件（如工作站GPU） | 在真实 Snapdragon 手机上实测 |
| **Objective Design** | 忽视 on-device 的独特价值（privacy, offline, zero marginal cost） | 显式加入 **locality bonus**，使 thermal-aware routing 成为有意义的优化目标 |
| **Failure Model Awareness** | 通常假设 on-device 是稳定服务 | 揭示了当前工具链下的 runtime instability 问题 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 自建基准测试集：共 **210个prompt**
  - **训练集**：150个（short-factual / medium-analytical / long-multistep 各50）
  - **测试集**：60个（各类型20个）
- Prompt来自五个技术领域，结构化模板生成，确保复杂度标签与实际计算得分一致；
- 所有 prompt 的黄金参考答案由 **Claude Opus 4.8** 一次性生成并冻结，避免风格偏倚。

### **实验设置**
- **设备平台**：Samsung Galaxy S25+（Snapdragon 8Elite, Adreno 830 GPU）
- **边缘节点**：局域网内的高性能工作站（运行 Ollama + Llama 3.1 8B + dense retrieval）
- **云模型**：GPT-4o via OpenAI API
- **测量指标 per query**：
  - Selected tier
  - Time-to-first-token (TTFT)
  - Total latency
  - Generated tokens
  - Thermal headroom & status
  - Battery current
  - Output text

### **评估指标**
- **主质量指标**：BERTScore-F1（roberta-large）
- **辅助质量指标**：ROUGE-L
- **其他指标**：
  - Latency（ms）
  - Cost（USD/query）
  - On-device ratio
  - Throttle violation rate
  - Wedge/failure rate

### **基线方法对比**
共7种路由策略（Conditions）：
| 编号 | 名称 | 描述 |
|------|------|------|
| C1 | Always on-device | 固定使用 on-device 模型 |
| C2 | Always edge | 固定使用 edge 模型 |
| C3 | Always cloud | 固定使用 cloud 模型 |
| C4 | Complexity-only | 仅根据 query complexity 分配 |
| C5 | Confidence cascade | 从 on-device 开始，若答案不完整则升级 |
| C6 | Thermal heuristic | 复杂度路由 + 接近严重节流时强制 offload 到 edge |
| C7 | **RL router (HybridInfer)** | 基于Q-learning的热感知策略 |

> 注：C1 和 C5 因连续运行会 crash，故采用“每次启动新进程”的 single-shot 测量方式。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table I & II）**

#### **全测试集表现（Table I）**
| Condition | BERTScore-F1 | Latency (ms) | Cost ($/query) | On-device |
|----------|---------------|----------------|------------------|------------|
| C3 (cloud) | 0.8449 ± 0.0040 | 6303 | 0.0053 | 0% |
| C7 (**RL router**) | **0.8399 ± 0.0039** | 19124 | **0.0017** | 9% |
| C1 (on-device) | 0.8385 ± 0.0070 | 28522 | 0.0000 | 100% |
| C4 (complexity) | 0.8376 ± 0.0029 | 14442 | 0.0028 | 33% |
| C6 (thermal heuristic) | 0.8367 ± 0.0027 | 15199 | 0.0020 | 24% |
| C2 (edge) | 0.8302 ± 0.0030 | 17991 | 0.0000 | 0% |

> ✅ **C7 在所有 adaptive 路由器中达到最高质量且最低成本**

#### **短/中等prompt子集对比（Table II，公平比较）**
| Condition | BERTScore-F1 | Latency (ms) | On-device |
|----------|---------------|----------------|------------|
| C3 (cloud) | 0.8496 ± 0.0056 | 4826 | 0% |
| C7 (**RL router**) | **0.8446 ± 0.0058** | **8998** | 7% |
| C5 (cascade) | 0.8403 ± 0.0069 | 30064 | 100% |
| C1 (on-device) | 0.8385 ± 0.0070 | 28522 | 100% |

> ✅ C7 质量接近 cloud，延迟仅为 on-device 的 **~1/3**，同时保留部分本地执行

### **与基线方法的对比结果**
- **质量显著优于手工启发式方法**：
  - C7 vs C4（complexity）: *p=0.011*（Wilcoxon signed-rank test）
  - C7 vs C6（thermal heuristic）: ***p=0.0015***
- **成本低于其他adaptive方法**：C7 平均每查询成本仅 **$0.0017**，远低于 C4 ($0.0028) 和 C6 ($0.0020)
- **覆盖更广**：C1/C5 在 long prompts 上失败率高达 **55%（22/40）**，而 C7 可成功 offload 处理

### **消融实验结果（隐含分析）**
- **Locality Bonus 的必要性**：
  - 若移除 `b_local` 项，Q-learning 政策退化为 **always offload**；
  - 此时 thermal headroom 不再影响决策 → 热感知失去意义；
  - 结论：**必须显式奖励 on-device 推理的价值，否则最优策略永远是 offload**
- **Policy Learned Behavior（Fig. 3）**：
  - 在热余量充足时，简单查询保留在设备上；
  - 在高温状态下（headroom > 0.85），即使是简单查询也倾向 offload；
  - 复杂查询默认导向 edge/cloud，体现质量优先原则

---

## **4. 关键结论和发现**

### **主要发现**
1. 🔥 **Sustained on-device inference is unstable**  
   即使设备未达高温，连续 on-device 查询仍会导致 **runtime crash 或 silent wedge**，根源在于 OpenCL 内核编译与 prefill 阶段的工具链缺陷，而非单纯热节流。

2. 🧠 **Thermal-aware routing improves reliability and coverage**  
   热感知路由不仅是效率机制，更是**可靠性保障机制**。它帮助系统避开可能导致崩溃的操作区间，提升整体服务稳定性。

3. 💡 **Locality must be valued explicitly**  
   若不在 reward 中加入 **locality bonus**（代表 privacy、offline、zero marginal cost），则 on-device 永远不会被选中 → thermal state 成为无关变量 → thermal-aware routing 失去意义。

4. 🏆 **Learned policy outperforms hand-crafted heuristics**  
   HybridInfer 在质量、成本、延迟之间取得更好平衡，且能自适应地调整阈值，优于固定规则的 C4 和 C6。

### **方法的局限性**
- **数据部分缺失**：C7 因偶尔触发 on-device crash，未能完成全部三轮复制（n=87 < 其他条件），导致统计完整性受限；
- **Single-device study**：仅在一个型号（Galaxy S25+）上测试，泛化性待验证；
- **Workload难度适中**：三个 tier 的质量差距较小，可能低估 capability-gap 的影响；
- **Energy measurement proxy**：仅用电流×时间估算能耗，未精确测量功耗；
- **Always-on-device 条件为 single-shot**：不能完全反映真实持续负载下的性能衰减。

### **未来工作方向**
1. 完成 C7 的完整三轮 replication（可通过 single-shot harness 实现）；
2. 在相同设备上复现 **ConsRoute-style 非热感知路由器**，直接对比 thermal signal 的增益；
3. 构建更具挑战性的 prompt workload，放大不同 tier 的能力差异；
4. 引入更丰富的 reward terms，如显式的 privacy cost 或 energy cost；
5. 跨多款设备测试，检验 learned policy 的迁移能力；
6. 探索 online 或 periodic retraining 机制，以适应设备老化和环境变化。

---

> ✅ **总结一句话**：  
> **HybridInfer 是首个利用真实手机 thermal headroom 实现 LLM 推理层级选择的系统，揭示了当前 on-device 工具链的稳定性瓶颈，并证明——只有当 on-device 的独特价值被显式建模时，热感知路由才真正有意义。**

</details>

---

### 10. [Electric Vehicle Charging Station Location Selection using Geospatial Artificial Intelligence (GeoAI)](https://arxiv.org/abs/2609.30417)

**Authors**: Eun Hak Lee, Euntak Lee  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.30417v1  

#### Abstract
As electric vehicle (EV) adoption increases, ensuring efficient and well-distributed charging infrastructure has become a critical challenge. While many EV charging station location problem (CSLP) studies focus on minimizing costs or travel distance, it is crucial to consider the surrounding geospat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究针对**Electric Vehicle Charging Station Location Problem (CSLP)** 中长期存在的一个关键问题：现有选址模型多聚焦于最小化成本或行驶距离，而**忽视了充电站周围地理空间特征（geospatial characteristics）对其运营绩效的影响**。这种忽略导致规划缺乏对实际空间环境的考量，可能造成资源错配。

### 提出的新方法与新思路
提出了一种基于**Geospatial Artificial Intelligence (GeoAI)** 的 **VAE-GCN 联合建模框架**，用于优化电动汽车充电站的选址决策。其核心创新在于：
- **整合高维地理空间数据**：将 EV 使用、土地利用（land-use）、人口、交通流量等多源异构数据统一纳入分析。
- **引入深度学习架构**：
  - 使用 **Variational Autoencoder (VAE)** 对高维输入数据进行降维，提取低维潜在空间（latent space），有效压缩信息并保留关键特征。
  - 利用 **Graph Convolutional Network (GCN)** 在由候选位置构成的空间图上进行推理，通过聚合邻域节点信息来预测每个位置作为充电站的适宜性概率。
- **以“地理相似性”为核心目标**：新站点的选址依据是其与现有成功运营站点在地理空间特征上的相似度，而非单一的距离或成本指标。

### 相比现有方法的优势
- **更强的解释性与现实贴合度**：通过捕捉现有站点的成功模式，使新站点更可能继承其运营优势。
- **更高的计算效率与预测精度**：VAE 的降维缓解了高维数据带来的计算负担，并提升了后续 GCN 的学习效率。
- **支持多目标政策评估**：框架可灵活应用于不同战略目标下的情景模拟（如最大化相似性 vs 最小化总行程距离）。

---

## 2. 核心实验方法和设置

### 数据集
实验基于美国德克萨斯州 Bryan–College Station 地区的真实数据，共覆盖约 185 km² 区域，整合了以下四类公开数据集：

| 数据类别 | 来源 | 主要变量 |
|--------|------|---------|
| EV 充电桩位置 | US DOE (Alternative Fuels Data Center) | 空间坐标 |
| 电动车注册量 | Atlas EV Hub | 各区域注册数量 |
| 城市特征 | Texas Water Development Board (TWDB) | 土地利用分类（商业/住宅/其他）、建筑面积、昼夜人口估算 |
| 道路网络与交通 | FHWA (HPMS) | 车道里程 (LM)、年平均日交通量 (AADT)、道路等级 |

采用**缓冲区（buffer）方法**构建分析单元：沿道路网每 100 米生成半径为 500 米的圆形缓冲区，共得 1,971 个分析节点。

### 实验设置与评估指标
- **任务定义**：二分类任务——判断某缓冲区是否适合建设充电站。
- **训练/测试划分**：80% 训练集，20% 测试集。
- **评估指标**：
  - **Precision（精确率）**
  - **Recall（召回率）**
  - **F1-score（F1 分数）**

### 基线方法对比
与以下五种主流模型进行比较：
- **GCN model**：直接使用原始高维数据输入 GCN
- **Multilayer Perceptron (MLP)**
- **Random Forest (RF)**
- **Extreme Gradient Boosting (XGB)**
- **Categorical Boosting (CatBoost)**

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | Precision | Recall | **F1-score** |
|------|----------|--------|-------------|
| **VAE-GCN (本文方法)** | **0.80** | **0.97** | **0.87** |
| GCN | 0.63 | 0.82 | 0.71 |
| MLP | 0.16 | 0.25 | 0.20 |
| RF | 0.63 | 0.75 | 0.68 |
| XGBoost | 0.59 | 0.86 | 0.70 |
| CatBoost | 0.64 | 0.88 | 0.74 |

- 本文提出的 **VAE-GCN 模型在所有指标上均显著优于基线模型**，尤其在 F1-score 上达到 **0.87**，远超次优模型（CatBoost, 0.74）。
- 高 Recall (0.97) 表明模型能几乎完整识别出所有现有站点位置；适度 Precision (0.80) 反映其倾向于保守推荐高潜力区域。

### 与基线方法的对比结果
- **VAE-GCN 比标准 GCN 提升明显**（F1: 0.87 vs 0.71），验证了**VAE 降维的有效性**——直接处理高维数据会损害 GCN 的表现。
- 传统机器学习模型（如 RF、XGB）虽有一定效果，但在捕捉复杂空间依赖关系方面不如图神经网络。
- MLP 表现极差，说明浅层全连接网络难以处理此类高维稀疏空间数据。

### 消融实验与扩展分析（隐含）
虽然未明确列出消融实验表格，但通过以下分析体现了模块有效性：
- **Geospatial Similarity Analysis** 显示，多数候选节点与至少两个现有站点具有高相似性，且相似节点之间也彼此接近，证明模型成功捕获了空间聚类模式。
- **新站点推荐**：模型识别出 **27 个候选位置**，其适宜性得分（suitability score）均高于 0.5。
- **政策情景模拟**：对比两种策略下新增 3 个站点的效果：
  - **Maximize Similarity**：平均相似度 70.0%，总行程减少 7.0%
  - **Minimize Travel Distance**：平均相似度 58.1%，总行程减少 **11.6%**
  > 结果表明：追求便利性的“最短距离”策略虽能更大程度降低用户出行负担，但牺牲了与现有成熟站点的空间一致性。

---

## 4. 关键结论和发现

### 主要发现
1. **地理空间上下文至关重要**：现有充电站的成功运营与其周边的土地利用、人口密度、交通流等密切相关，这些因素应被系统性纳入选址决策。
2. **VAE-GCN 架构高效可行**：结合 VAE 的降维能力与 GCN 的图结构建模能力，能够有效从复杂 GeoAI 数据中学习空间模式，实现高精度预测。
3. **存在权衡关系**：最大化地理相似性有助于维持服务一致性与管理稳定性；最小化旅行距离则提升用户体验与可达性。二者适用于不同战略目标，需因地制宜选择。

### 方法的局限性
- **静态建模**：当前模型基于静态快照数据，未考虑 EV 渗透率增长、城市扩张等动态变化。
- **数据聚合假设**：缓冲区内属性按比例分配，忽略了微观尺度的空间异质性。
- **未显式建模 EV 移动行为**：缺乏轨迹数据支持对真实充电需求热区的动态预测。

### 未来工作方向
- 引入**动态预测机制**，融合人口增长、城市发展规划等前瞻性数据。
- 探索更精细的拓扑建模方式（如基于路网的图结构），替代当前的缓冲区方法。
- 结合强化学习或在线学习框架，实现自适应的基础设施扩展策略。
- 扩展至多模态交通场景（如 Robotaxi、电动公交），探索协同优化路径。

</details>

---

### 11. [Aurora-X: Built for Extreme Time Series Forecasting](https://arxiv.org/abs/2609.31038)

**Authors**: Xingjian Wu, Chenjuan Guo, Xiangfei Qiu, Zhigang Hu, Hanyin Cheng, Peng Chen, Yang Shu, Jilin Hu, Bin Yang  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.31038v1  

#### Abstract
Time series foundation models (TSFMs) enable cross-domain forecasting, but their development as general-purpose forecasters remains constrained by underexplored training potential and limited architectural versatility. To address these challenges, we introduce Aurora-X, a billion-scale TSFM with a p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **Aurora-X: Built for Extreme Time Series Forecasting**  
**核心结论与实验结果总结**

---

## **1. 主要贡献和创新点**

### **解决的问题**
现有的 **Time Series Foundation Models (TSFMs)** 在作为通用预测器的发展上面临两大挑战：
1.  **训练潜力未被充分挖掘 (Underexplored training potential)**：缺乏一个系统性的、分阶段的课程学习（curriculum）来逐步发展模型能力。
2.  **架构灵活性有限 (Limited architectural versatility)**：现有模型通常无法同时支持跨变量建模、协变量条件化、可变分辨率推理以及任意分位数预测等关键功能。

### **提出的新方法和新思路**
为解决上述问题，论文提出了 **Aurora-X**，这是一个十亿参数规模的 TSFM，其核心创新在于一个**统一的架构**和一个**渐进式的课程学习策略**。

#### **主要创新点：**

- **渐进式课程学习 (Progressive Curriculum)**：
  - **预训练 (Pretraining)**：采用**通道独立 (channel-independent, CI)** 策略，专注于学习通用的时间模式。
  - **中段训练 (Midtraining)**：逐步引入**联合多变量建模 (cross-variable)**、**变化的上下文/预测长度**以及**未来协变量 (future covariates)**，以增强模型的复杂技能。
  - **后训练 (Post-training)**：通过**多尺度重采样 (multi-scale post-training)**，使模型能够进行**可变分辨率推理 (variable-resolution inference)**，从而实现测试时扩展 (test-time scaling)。

- **统一且灵活的架构 (Unified and Versatile Architecture)**：
  - **模式引导的混合专家 (Pattern-guided Mixture-of-Experts, MoE)**：一种稀疏激活的 MoE 架构。其核心是利用**浅层块相似性 (shallow patch similarities)** 来约束深层网络中的专家路由 (routing)，从而引导专家在异构时间序列上的专业化。
  - **隐式分位数网络头 (Implicit Quantile Network, IQN Head)**：该头部可以预测**任意分位数水平 (arbitrary quantiles)**，无需预先定义分位数网格，极大地增强了概率预测的灵活性。
  - **并行解码 (Parallel Decoding)**：能够并行地对未来的多个预测块进行解码，避免了自回归生成可能带来的误差累积。

### **相比现有方法的优势**
Aurora-X 是首个将以下所有能力集成于单一模型中的 TSFM：
- ✅ 跨变量建模 (Cross-variable modeling)
- ✅ 协变量条件化 (Covariate conditioning)
- ✅ 可变分辨率推理 (Variable-resolution inference)
- ✅ 任意分位数预测 (Arbitrary quantile prediction)
- ✅ 测试时扩展 (Test-time scaling)

这使其成为一个真正“开箱即用”的、适用于极端时间序列预测任务的强大工具。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验在五个广泛认可的基准上进行，覆盖了从通用预测到特定场景的多种任务：
- **通用预测基准**：`GIFT-Eval`, `TIME`, `FEV-Bench`
- **监督式多变量预测基准**：`TFB` (Toward comprehensive and fair Benchmarking)
- **带协变量的预测基准**：`DAG-Bench`

### **实验设置和评估指标**
- **评估协议**：遵循各基准的官方设置，包括数据划分、预测范围和指标定义。
- **主要评估指标**：
  - **相对 MASE (Relative MASE)**：几何平均的配置级比率，越低越好。
  - **MSE (均方误差)** 和 **MAE (平均绝对误差)**：用于与监督模型进行比较。
  - **加权分位数损失 (WQL)** 和 **缩放分位数损失 (SQL)**：用于评估概率预测性能。
- **模型规模**：Aurora-X 总参数量约为 **10.5亿 (1.05B)**。

### **基线方法对比**
与多种先进的 TSFMs 和监督模型进行了全面对比：
- **TSFM 基线**：`TimesFM-2.5`, `Chronos-2`, `TiRex`, `TiRex-2`, `Sundial`, `Toto-2.0`, `Timer-S1`, `Falcon-X` 等。
- **监督模型基线**：`DUET`, `AMD`, `SRSNet`, `PatchTST`, `DAG` 等。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
Aurora-X 在所有五个基准上均取得了**最先进的 (state-of-the-art)** 性能。

#### **与先进 TSFMs 的对比 (图 5 & 图 4)**
- **GIFT-Eval**：相对 MASE 达到 **0.659**，比 `Falcon-2.0` 降低 1.1%，比 `TiRex` 降低 **8.0%**。
- **TIME**：相对 MASE 达到 **0.640**，比 `TimesFM-2.5` 降低 **4.3%**。
- **FEV-Bench**：相对 MASE 达到 **0.617**，比 `TimesFM-2.5` 和 `Chronos-2` 分别降低 4.2% 和 **4.4%**。

#### **与监督模型的对比 (图 4)**
- **TFB (多变量)**：相比 `DUET`，平均 MSE 降低 **9.0%**，MAE 降低 3.6%。
- **DAG-Bench (带协变量)**：相比 `DAG`，平均 MSE 降低 **11.5%**，MAE 降低 7.7%。

### **消融实验结果**
- **MoE 设计对比**：提出的 **Pattern-guided MoE** 在 `GIFT-Eval`, `TIME`, `FEV-Bench` 上的相对 MASE 均优于 `Time-MoE`, `Moirai-MoE`, `Timer-S1` 等其他 MoE 实现。
- **预测头对比**：**IQN 头**的性能优于 `Chronos-2` 的固定分位数回归、`Sundial` 的流匹配和 `Moirai` 的混合分布头。
- **课程学习增益**：如图 9 所示，从预训练到后训练，模型性能持续提升。在 `GIFT-Eval`, `TIME`, `FEV-Bench` 上，相对 MASE 分别累计降低了 **13.1%**, **19.0%**, 和 **20.4%**，证明了渐进式课程的有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1.  **渐进式课程至关重要**：分阶段的训练策略（从简单的时间模式学习，到复杂的多变量和协变量建模，再到可变分辨率推理）是成功构建强大 TSFM 的关键。
2.  **架构创新带来综合优势**：`Pattern-guided MoE` 有效促进了专家的专业化，`IQN head` 提供了无与伦比的概率预测灵活性，而 `parallel decoding` 则保证了高效的推理。
3.  **Aurora-X 是当前最强的 TSFM**：在广泛的基准测试中，Aurora-X 不仅显著超越了所有现有的 TSFMs，甚至在许多任务上也击败了专门训练的监督模型，证明了其强大的零样本泛化能力。

### **方法的局限性**
- **计算成本**：作为一个十亿参数模型，Aurora-X 的训练和部署需要大量的计算资源。
- **解释性**：尽管 `Pattern-guided MoE` 旨在引导专家专业化，但深层网络内部的决策过程仍然具有一定的黑盒性质。

### **未来工作方向**
- 探索更高效的 MoE 路由机制和模型压缩技术。
- 将 Aurora-X 的能力扩展到更多下游任务，如异常检测、因果推断和决策智能。
- 研究如何更好地结合领域知识和外部工具，以进一步增强模型的推理能力。

</details>

---

### 12. [CRNDiff: Count-Native Diffusion Framework via Chemical Reaction Networks](https://arxiv.org/abs/2609.31149)

**Authors**: Yuxuan Qiu, Praful Gagrani, Tetsuya J Kobayashi  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.31149v1  

#### Abstract
Scientific measurements such as single-cell RNA (scRNA) sequencing often take the form of nonnegative integer counts, whereas continuous-state diffusion models approximate this discrete structure using continuous coordinates. Building on stochastic chemical reaction networks (CRNs), a class of count...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# CRNDiff: Count-Native Diffusion Framework via Chemical Reaction Networks —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **diffusion models** 多基于连续状态空间建模，难以自然处理科学测量中常见的**非负整数计数数据**（如单细胞 RNA 测序中的转录本计数）。虽然已有部分离散扩散模型，但它们通常缺乏对**计数空间结构**（有序性、无限性）的显式建模，且在生成稀有亚群时面临重要性权重集中（importance-weight concentration）等问题。

### 🚀 提出的新方法与创新思路
作者提出 **CRNDiff**，一个基于**化学反应网络**（Chemical Reaction Networks, CRNs）的**原生计数空间扩散框架**，其核心创新包括：

- **Count-native diffusion via CRNs**：  
  将扩散过程建模为**随机化学反应网络**（stochastic CRN），利用 birth-death 反应作为前向加噪机制，在保持数据始终处于非负整数格点的同时实现渐进随机化。

- **闭式转移核与可逆采样**：  
  利用概率生成函数（PGF）推导出前向过程的**闭式转移核**（closed-form transition kernel），支持精确的 **forward-filtering backward-sampling (FFBS)** 逆向采样。

- **数据驱动的终端时间选择**：  
  提出基于**过量协方差衰减**（excess-covariance decay）的准则自动确定前向加噪的终止时间 $T_0$，无需验证集调参。

- **倾斜 Feynman-Kac (FK) 引导**（Tilted FK Steering）：  
  在推理阶段对冻结的生成器进行条件控制，通过两步策略缓解稀有目标的重要性权重退化：
  1. **边际倾斜**（Marginal Tilt）：先调整反向提议分布以匹配目标边缘分布；
  2. **残差校正**（Residual Correction）：再通过 FK 粒子滤波修正联合分布偏差。

### 🔍 相比现有方法的优势
| 优势维度 | CRNDiff | 其他方法（如 scVI, MDLM） |
|---------|--------|--------------------------|
| **状态空间适配性** | 原生建模整数计数空间 | 连续潜变量 + 解码器，可能失真 |
| **理论可解性** | 闭式核、可解释动态 | 黑箱神经网络近似 |
| **稀有亚群生成** | 显式缓解权重集中 | 易受 SMC 退化影响 |
| **训练效率** | 冻结生成器 + 推理时引导，无需重训 | 需重新训练或微调 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **真实数据**：来自成人人类心脏图谱的 **scRNA-seq 数据**（Litvinuková et al., 2020）
  - 包含三种细胞类型：
    - **内皮细胞**（Endothelial）：80,463 训练样本（丰富）
    - **髓系细胞**（Myeloid）：18,422 训练样本（中等）
    - **神经元**（Neuronal）：3,168 训练样本（稀有）
- **合成数据**：二维整数格点上的八组分混合模型，用于分离边际与联合结构的影响。

### 🧪 实验设置
- **模型架构**：Transformer 编码器 + 分类计数头（categorical count head）
- **输入特征**：噪声计数、基因标识、时间嵌入
- **训练方式**：无条件训练一次，后续所有条件生成均在**冻结生成器上进行推理时引导**
- **采样配置**：
  - 逆向步数 $K=32$
  - 粒子数 $M=2 \times n_{\text{out}}$
  - 残差判别器：梯度提升树（Gradient Boosted Tree）

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **Purity ↑** | 由外部 CellTypist 模型判定为目标类的比例 |
| **Wasserstein-1 (W1) ↓** | 单基因分布距离（raw counts） |
| **MMD² ↓** | 联合分布差异（log-CP10K） |
| **PCC ↑** | 均值表达水平相关性 |
| **Top-100 Marker Overlap ↑** | 差异表达基因排名一致性 |
| **Macro F1 ↑** | 下游分类任务性能（替换训练数据） |
| **Anc. (Genealogical Diversity) ↑** | 输出样本祖先多样性，反映粒子退化程度 |

### 🆚 基线方法对比
- **scVI**, **scANVI**, **CFGen**：基于变分自编码器的生成模型
- **MDLM**：Masked Diffusion Language Model，适用于离散序列
- 所有方法使用相同预处理流程（2,000 高变基因、CP10K 归一化等）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table 2）

| Method | Endothelial Purity | Myeloid Purity | Neuronal Purity |
|--------|--------------------|----------------|-----------------|
| **CRNDiff** | **0.906** ± .005 | **0.886** ± .014 | **0.971** ± .012 |
| MDLM | 0.898 ± .026 | 0.840 ± .072 | 0.784 ± .063 |
| CFGen | 0.797 ± .009 | 0.625 ± .027 | 0.734 ± .004 |
| scANVI | 0.857 ± .016 | 0.625 ± .018 | 0.269 ± .019 |
| scVI | 0.423 ± .014 | 0.482 ± .040 | 0.535 ± .073 |

> ✅ CRNDiff 在所有三个目标群体中均取得最高纯度，尤其在稀有神经元群体中表现显著领先。

#### 分布保真度（Distributional Fidelity）
- CRNDiff 在 **W1、MMD²、PCC** 上全面优于基线
- 神经元任务中，CRNDiff 的 **MMD² = 0.0102**，远低于 MDLM 的 0.0120 和 CFGen 的 0.0130

### 🔬 消融实验结果（Ablation Studies）

#### （1）Tilted FK Steering vs. 基础组件（Table 6）
| 方法 | Neuronal Purity | MMD² |
|------|------------------|-------|
| Marginal Tilt Only | 0.063 | 0.0588 |
| FK Steering Only | 0.990 | 0.0179 |
| **Tilted FK Steering** | **0.984** | **0.0118** |

> ⚠️ 仅边际倾斜无法有效捕获稀有群体；完整两阶段策略平衡了效率与精度。

#### （2）基因多样性分析（Table 7）
| 方法 | Neuronal Anc. |
|------|---------------|
| FK Steering | 0.013 |
| **Tilted FK Steering** | **0.045** |

> ✅ Tilted FK 显著缓解了粒子退化，保留更多祖先路径，说明其更有效地维持了采样多样性。

#### （3）不同引导策略比较（Table 10）
在稀有神经元任务中：
- **Best-of-N** 最高仅达 0.604（16×候选池）
- **Value-guided resampling** 达到 0.968（γ=4），但牺牲 W1 和 MMD²
- **Tilted FK Steering** 达到 **0.984**，同时保持最佳 W1 和 MMD² 平衡

> ✅ Tilted FK 在稀有目标上实现了**纯度与保真度的最佳权衡**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **CRNDiff 是首个将 CRN 与 diffusion 结合的原生计数生成框架**，具备良好的数学结构性与可解释性。
2. **Tilted FK Steering 有效缓解了稀有目标生成中的重要性权重集中问题**，相比传统 SMC 更稳定。
3. **冻结生成器 + 推理时引导** 的范式可行且高效，避免了为每个目标群体重复训练。
4. 随着目标群体稀有性增加，CRNDiff 相对于基线的**纯度优势进一步扩大**，表明其特别适合稀有亚群建模。
5. 生成细胞能高度保留**marker-level differential expression structure**，可用于下游分类任务，性能接近真实数据基准。

### ⚠️ 局限性
- 当前实例化仅使用独立 birth-death 反应，未建模基因间耦合反应（coupled reactions）。
- 终止时间 $T_0$ 基于二阶统计量（过量协方差），更高阶依赖可能需要额外诊断。
- 密度比估计误差、有限粒子数可能影响极端稀有目标的表现。
- 低 library-size CV 和 median Fano 表明生成器存在**过度平滑倾向**，未能完全修复分散不匹配（dispersion mismatch）。

### 🔮 未来工作方向
- 扩展至**耦合反应网络**（如基因调控网络建模）
- 探索**结构化状态空间扩展**（structured state extensions）
- 引入更高阶统计量进行更精细的时间校准
- 将 CRNDiff 应用于其他计数型数据（如空间转录组、ATAC-seq）

---

## 总结一句话
> **CRNDiff 提供了一个理论严谨、实践高效的原生计数扩散框架，通过化学反应网络建模与倾斜 FK 引导，在稀有细胞亚群生成任务中实现了当前最优的条件保真度与多样性平衡。**

</details>

---

### 13. [The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models](https://arxiv.org/abs/2609.31341)

**Authors**: Christoph Walser, Mauricio Fadel Argerich, Jonathan F\"urst  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.31341v1  

#### Abstract
Whether an information extraction pipeline should process page images or parsed text depends on the document, and the answer flips across the layout spectrum. We study this trade-off under a constraint that rules out (closed) cloud services: privacy-sensitive documents processed on-premise by small ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文聚焦于**隐私敏感场景下的本地化信息提取（Information Extraction, IE）**，解决在资源受限环境下如何在**准确性**与**能耗**之间取得平衡的问题。传统基于云服务的大模型方案因数据泄露风险不适用于金融、医疗等高隐私要求领域，而小型本地模型的设计空间尚未被系统探索。

### 提出的新方法与思路
- **多目标优化框架**：首次将信息提取任务建模为一个兼顾 **accuracy** 和 **end-to-end energy consumption** 的多目标优化问题。
- **跨设计维度联合分析**：系统性地研究了三个关键维度对性能的影响：
  - 输入表示（Input Representation）：raw images vs. OCR vs. parser
  - 模型架构（Model Family）：VLMs、text-only LLMs、specialized layout models
  - 推理配置（Inference Configuration）：batch size、FP8 quantization
- **提出“文档类型决定最优pipeline”原则**：指出没有单一最佳方案，最优选择取决于文档的布局复杂度。

### 相比现有方法的优势
- 超越仅关注准确性的传统评估范式，引入**端到端能效分析**，更贴近实际部署需求。
- 在真实硬件（NVIDIA L4 GPU）上进行完整pipeline profiling，结果更具实践指导意义。
- 开源全部代码与配置（GitHub: [chrewbroccoli/local-ie-energy](https://github.com/chrewbroccoli/local-ie-energy)），增强可复现性。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型特征 | 文档数量 | 字段数/文档 | 页面平均长度 |
|--------|---------|----------|-------------|----------------|
| **Kleister-NDA** | 近似纯文本（near-plain text）、出生即数字（born-digital） | 337 | 最多4个字段 | 5.87页/文档 |
| **VRDU Registration Forms** | 布局丰富（layout-rich）、扫描件（scanned） | 500 | 最多6个字段 | 1.83页/文档 |

> ✅ 二者分别代表文档谱系两端：低视觉结构 vs 高二维空间依赖。

### 实验设置
- **模型类别**：
  - **Vision-Language Models (VLMs)**：Qwen3-VL系列（2B, 4B, 8B）
  - **Text-only LLMs**：Llama-3.2、Ministral-3-3B、Mistral-7B、Qwen3系列
  - **Specialized Layout Models**：Arctic-TILT（OCR+Layout）、NuExtract-2.0-4B（VL）
- **输入处理方式对比**：
  - Raw Images → VLM直接处理
  - OCR输出 → Tesseract（经典OCR）、Docling（嵌入式文本解析）、DeepSeek-OCR 2（神经OCR）
- **推理优化技术**：
  - Batch Size：1, 5, 10, 20, 40, 60
  - Quantization：FP16 vs FP8（via vLLM）
- **部署环境**：单张 NVIDIA L4 GPU（24GB VRAM），使用 vLLM 进行高效推理调度。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Average Field Exact Match (EM)** | 字段预测值完全匹配的比例（忽略未回答字段） |
| **Energy Consumption (mWh/page)** | 包括 parsing + inference 的总能耗，按每页归一化 |
| **Pareto Frontier** | 在 accuracy-energy 平面上识别出的非支配解集合 |

> ⚠️ 注意：Arctic-TILT 在 Kleister-NDA 上为 in-domain（已在该数据集微调），故其高分不可与其他零样本模型直接比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### （1）RQ1: 基线模型表现（Batch=1, No Quantization）

| 场景 | 最佳模型 | EM (%) | Energy (mWh/pg) |
|------|--------|--------|------------------|
| **VRDU (layout-rich)** | NuExtract-2.0-4B | **78.3%** | 17.8 |
| **Kleister-NDA (text-heavy)** | Qwen3-4B + Tesseract | **76.9%** | 7.9 |

> 📌 观察：VLMs 在 VRDU 表现更好；text-only LLMs 在 Kleister-NDA 更优且更省电。

#### （2）RQ2: 输入表示影响（Parsing Energy）

| Parser | Kleister-NDA (mWh/pg) | VRDU (mWh/pg) | 特点 |
|-------|------------------------|---------------|------|
| **Tesseract** | 0.0043 | 0.0047 | 经典OCR，CPU运行 |
| **Docling** | **0.0015** | 0.0141 | 利用PDF内嵌文本，效率极高（仅用于born-digital） |
| **DeepSeek-OCR 2** | 0.0751 (**17×**) | 0.0835 (**18×**) | GPU加速，精度高但能耗巨大 |

> 🔥 结论：**neural OCR 成本过高，从未进入 Pareto frontier**

#### （3）RQ3: 推理优化效果

| 技术 | 能耗降低幅度 | 对准确率影响 |
|------|--------------|------------|
| **Batching (1 → 60)** | **38–85%** | 无损失 |
| **FP8 Quantization (Batch=1)** | 27–32% | 可忽略（±<1 EM） |
| **FP8 (Batch已优化后)** | 仅剩 **9–19%**（绝对值 <1 mWh/pg） | 几乎无影响 |

> 📈 批量是最大节能杠杆，FP8 效益随 batch 增大显著衰减。

#### （4）Pareto Frontier 对比

| 数据集 | 前沿主导者 |
|--------|-----------|
| **VRDU** | VLMs（Qwen3-VL-2B/4B）、NuExtract |
| **Kleister-NDA** | Text-only LLMs + Cheap Parser（如 Qwen3-0.6B + Docling @ 2.4 mWh/pg） |

> ✅ 明确反转：**文档类型决定最优架构选择**

---

## 4. 关键结论和发现

### 主要发现
1. **Pipeline选择应以文档类型为导向**：
   - **Layout-rich documents（如表格、表单）**：优先选用 **VLMs** 或 **specialized VLMs**，直接读取图像跳过OCR，节省预处理开销。
   - **Near-plain text documents（如合同、报告）**：采用 **small text-only LLMs + lightweight parser（如Docling）**，既便宜又准确。

2. **Batching 是最有效的节能手段**：
   - 将请求批量处理可减少 **38–85%** 的能耗，且不影响准确性。
   - 大部分收益在 batch size=10 时即可实现。

3. **FP8 与 Batching 存在替代关系**：
   - 单请求场景下 FP8 可降耗约 30%，但在批处理环境中仅额外节省不到 1 mWh/pg。
   - 推荐策略：**先最大化 batch size，再考虑 FP8**。

4. **Neural OCR 不具性价比**：
   - DeepSeek-OCR 2 能耗是 Tesseract 的 **17–18 倍**，虽提升部分准确率，但整体不在 Pareto 前沿。
   - **仅当 accuracy 极度关键且预算充足时才考虑使用**。

5. **Parsing 是首要能耗瓶颈之一**：
   - 文本提取阶段可能消耗比模型推理更多的能量（尤其神经OCR）。
   - 应优先优化 parsing pipeline，而非盲目升级模型。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **Energy Profiling 工具限制** | 使用 CodeCarbon 软件估算功耗，未使用物理功率计，可能导致低估 20–30%。但相对趋势仍可靠。 |
| **硬件平台单一** | 所有实验基于 NVIDIA L4 GPU，不同 GPU 架构（如H100、Ada Lovelace）可能改变量化收益。 |
| **长度与布局混淆** | Kleister-NDA 更长且布局简单，VRDU 更短但结构复杂，无法完全分离“长度”与“布局”的独立效应。 |
| **prompting 策略有限** | 仅使用 single-shot prompting，未测试 Chain-of-Thought 等高级提示技巧，准确率可能是下界。 |
| **语种与文档类型覆盖不足** | 仅英文文档，缺乏多语言、手写体、发票收据等多样化场景验证。 |

### 未来工作方向
- 构建 **long-form layout-rich corpus** 以解耦长度与布局影响。
- 引入 **activation-aware weight quantization (AWQ)** 等先进压缩技术进一步降低内存占用。
- 扩展至 **multilingual、handwritten、agent-based IE pipelines**。
- 探索 **cooling overhead accounting** 和 **full-system energy modeling**。
- 设计 **adaptive pipeline selector**，根据输入文档自动选择最优路径。

---

## 总结建议（Practical Guidelines）

> ✅ **三步走策略构建高效本地IE系统**：

1. **Match parsing to layout**  
   - Born-digital 文档 → 使用 **Docling**（提取内嵌文本）  
   - Scanned 文档 → 使用 **Tesseract**（低成本OCR），仅在必要时启用 DeepSeek-OCR 2  

2. **Align architecture with structure**  
   - Layout-rich → 选 **VLMs**（如 Qwen3-VL）  
   - Near-plain text → 选 **small text-only LLM + efficient parser**  
   - 若有领域数据 → 微调小模型优于扩大零样本模型  

3. **Batch before you quantize**  
   - 先调大 **batch size** 至 GPU 内存饱和（通常 bs=10~20）  
   - 再应用 **FP8** 获取剩余小幅度节能与内存释放  

> 💡 “The right pipeline depends on the document.” —— 没有一刀切的解决方案，必须结合文档特性综合设计。

</details>

---

### 14. [Learning coarse-step dynamics and internal mechanical response with graph networks](https://arxiv.org/abs/2609.30344)

**Authors**: Vinay Sharma, Olga Fink  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.30344v1  

#### Abstract
Modern sensing records the motion of physical systems, but often leaves the forces and mechanical response governing that motion unobserved. Inferring these quantities from discretely sampled trajectories is especially difficult at coarse time scales, when mechanical response evolves between observa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning coarse-step dynamics and internal mechanical response with graph networks

## 1. 论文的主要贡献和创新点

### 解决的问题
现代传感技术能够记录物理系统的运动轨迹，但通常无法直接观测到驱动这些运动的**内部力学响应**（如力、力矩、刚度、阻尼等）。在**粗时间步长**（coarse time scales）下，由于机械响应在观测间隔内演化且相互作用在整个系统中传播，从离散采样的轨迹中推断这些未被观测的力学量变得尤为困难。

现有方法存在两大局限：
- **纯预测模型**（如 GNS, MGN）：虽然能准确预测轨迹，但其内部表示缺乏明确的力学解释。
- **依赖监督或先验物理模型的方法**（如 NequIP, PeTIGN）：需要力、力矩或材料参数作为训练标签，这在现实中往往难以获取。

### 提出的新方法：NEWMARK-B-DGN
本文提出了一种名为 **NEWMARK-B-DGN** 的图神经网络框架，它将计算力学中的两个经典思想融入深度学习架构，以实现对粗粒度动力学和内部力学响应的联合学习。

#### 核心创新点
1. **半隐式节点更新 (Semi-implicit Nodal Update)**：
   - 受 **Newmark-β 方法** 启发，该方法通过在更新后的状态上评估力学响应来提高数值稳定性。
   - 在 NEWMARK-B-DGN 中，每个节点独立求解一个小型的 3×3 方程组，其中包含了由网络学习得到的**矩阵值响应算子**（matrix-valued response operators `K` 和 `D`）。
   - 这使得模型能够稳定地积分刚性动力学，即使在远超显式积分稳定极限的时间步长下也能保持稳定。

2. **算子加权虚拟枢纽 (Operator-weighted Virtual Hub)**：
   - 为了解决长距离耦合问题，受隐式求解器中全局耦合的启发，提出了一种**秩一近似**（rank-one approximation）。
   - 引入一个**虚拟枢纽**（virtual hub），通过 `O(N)` 条虚拟边连接到所有物理节点，形成直径为2的星形拓扑。
   - 枢纽的状态（位置、速度、角速度）由所有物理节点的**响应算子加权平均**（weighted equilibrium）动态计算得出，从而高效地实现了系统范围内的非局部耦合。

### 相比现有方法的优势
- **无需监督**：仅需观测到的运动学数据（位置、速度）进行训练，无需任何关于内部力、力矩或材料参数的监督信号。
- **力学可解释性**：学习到的动量通量（momentum fluxes）和响应算子（response operators）具有明确的力学意义，可直接用于分析。
- **高稳定性**：在粗时间步长下进行长期预测时，相比显式模型表现出显著更好的稳定性和更低的误差累积。
- **高效通信**：虚拟枢纽以线性复杂度 `O(N)` 实现了系统级耦合，相比基于距离的稠密图（`O(N²)`）大幅降低了图的密度和计算成本。

---

## 2. 核心实验方法和设置

### 数据集
论文在四个不同的物理系统上进行了验证：
1.  **固定梁 (Clamped Beam)**：三维有限元梁，受横向载荷后自由振动。用于测试粗步长稳定性和外推能力。
2.  **人体运动捕捉 (Human Motion Capture)**：来自 CMU MoCap 数据库的步行序列。用于测试稀疏不规则图上的协调运动预测。
3.  **蛋白质动力学 (Protein Dynamics)**：腺苷酸激酶（Adenylate Kinase）从闭合态到开放态的构象转变。用于测试大尺度集体运动的预测。
4.  **人体行走生物力学 (Human Walking Biomechanics)**：来自 van der Zee 等人的仪器化跑步机数据，包含同步的运动学、地面反作用力和逆动力学关节力矩。用于测试推断的内部力学量是否与独立测量的参考值一致。

### 实验设置和评估指标
- **任务**：自回归滚动预测（autoregressive rollout），即用当前预测状态作为下一步的输入。
- **评估指标**：
  - **轨迹预测**：均方根误差（RMSE）、整体误差（whole-body error）。
  - **力学推断**：推断的关节力矩与逆动力学参考值之间的皮尔逊相关系数（Pearson correlation `r`）。
- **训练**：仅使用标准化的位置和速度增量作为监督信号，无任何力或力矩的损失项。

### 基线方法对比
与六种图网络模拟器进行了比较：
- **DGN**：最接近的基线，使用相同的守恒交互表示，但采用显式积分。
- **GNS**：基于粒子的编码-处理-解码架构。
- **MGN**：基于网格的模拟器。
- **EGHN**：使用等变分层池化的层次模型。
- **EGNO**：在傅里叶域应用等变时间卷积的时间神经算子。
- **EGHNO**：结合了 EGNO 和 EGHN 的模型。
- **IGNS**：端到端的端口哈密顿图积分器（仅在梁上比较）。

所有模型在相同的数据集、划分、优化器和评估协议下进行公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **固定梁 (Clamped Beam)**：
    - **长期稳定性**：NEWMARK-B-DGN 在 95 步的滚动预测中保持低且有界的误差，而所有基线模型（如 EGHN, EGNO, DGN）的误差都随时间单调增长或发散。
    - **外推能力**：在负载、几何形状、网格分辨率等超出训练范围的情况下，NEWMARK-B-DGN 的平均整体误差仅为 **0.54%**，而 DGN 为 3.00%，IGNS 在 24 个内部步骤下仍高达 2.6%。
    - **模态结构恢复**：预测的振动频率和主导模态与有限元参考值高度匹配（`MAC ≈ 1.00`），而显式 DGN 预测的是虚假的低频模式。

2.  **人体运动与蛋白质动力学**：
    - **人体运动**：在未旋转的测试集上，NEWMARK-B-DGN 的第5步 MSE 为 **1.335**，优于除 GNS 外的所有基线。更重要的是，它具有**旋转等变性**（rotation-equivariant），而 GNS 在仅 5° 旋转后性能急剧下降。
    - **蛋白质动力学**：NEWMARK-B-DGN 在整个 75 帧的预测范围内保持有界，最终 MSE 稳定在 **0.277 Å²**。相比之下，DGN 发散到 6.12，EGHN 和 EGNO 在第4步后变为非有限值。
    - **计算效率**：在蛋白质任务上，NEWMARK-B-DGN 仅使用 **3,418** 条边，而基于距离的基线（EGHN/EGNO）使用约 **55,610** 条边，内存占用减少了 16.3 倍。

3.  **内部力学推断 (Internal Mechanics Inference)**：
    - **关节力矩**：从步行运动学中推断出的髋关节和膝关节力矩与独立的逆动力学测量值高度相关，相关系数分别达到 **r = 0.94** 和 **r = 0.87**。
    - **响应算子**：学习到的响应算子恢复了有限元切线刚度的空间和方向结构（余弦相似性 ~0.85），尽管其绝对尺度和各向异性未被识别。

### 消融实验结果
通过逐步移除组件，量化了每个设计的重要性：
| 配置 | 平均整体误差 (%) | 相对于前一项的改进倍数 |
| :--- | :--- | :--- |
| DGN (12 显式子步, 无枢纽) | 3.00 | — |
| + 算子加权虚拟枢纽 (4 显式子步) | 1.26 | 2.4× |
| + 半隐式求解 (`β=0`) | 0.70 | 1.8× |
| + 完整 Newmark 更新 (`β=1/4`) | **0.54** | 1.3× |

**结论**：虚拟枢纽带来了最大的性能提升，其次是半隐式更新和完整的 Newmark 参数，证明了两种机制的协同作用。

---

## 4. 关键结论和发现

### 主要发现
1.  **联合学习是可行的**：仅通过运动学监督，可以同时学习到准确的动力学预测和具有物理意义的内部力学量（如力和响应算子）。
2.  **结构化先验至关重要**：将计算力学中的**半隐式更新**和**系统级耦合**思想融入网络架构，是实现粗步长稳定预测的关键。
3.  **力学可解释性可涌现**：学习到的内部表示（如 `f_ij`, `K_i`）不仅用于生成预测，其本身也与真实的物理量（如关节力矩、刚度分布）高度一致，为“虚拟传感”（virtual sensing）提供了可能。

### 局限性
1.  **单枢纽近似的限制**：对于非常长的梁，单个秩一枢纽的近似效果减弱，因为其弯曲响应需要更高秩的表示。
2.  **动量守恒不完全**：由于采用了独立的节点求解而非全局求解，导致有限子步长下的动量守恒存在残差（尽管是 `O(δt²)` 量级）。
3.  **相对读数而非绝对标定**：学习到的响应算子是**相对指标**，反映了结构的相对刚度/阻尼分布，但无法给出经过校准的绝对材料参数。

### 未来工作方向
1.  探索**多枢纽**或**更高秩表示**来更好地捕捉复杂的非局部耦合。
2.  设计更紧密耦合的更新方案，以改善更新级别的守恒特性。
3.  利用少量校准数据来锚定学习到的力学量的**绝对尺度**，使其从相对读数发展为绝对的力和力矩预测。
4.  将该框架应用于更广泛的工程和科学领域，如结构健康监测、机器人控制和分子设计。

</details>

---

### 15. [Adaptive Multi-Value Control in LLMs via Causal Activation Steering](https://arxiv.org/abs/2609.30405)

**Authors**: Payel Bhattacharjee, Ravi Tandon  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.30405v1  

#### Abstract
Large language models (LLMs) are increasingly deployed in settings where responses must reflect multiple, potentially interacting social norms and human values. Activation steering offers a lightweight alternative to training-based alignment by modifying internal activations at inference time. Howev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Adaptive Multi-Value Control in LLMs via Causal Activation Steering

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在教育、医疗、金融等关键领域部署时，需要同时反映多种可能相互作用的社会规范和人类价值观（human values）。现有的对齐（alignment）方法如 RLHF、DPO 等依赖于额外的偏好收集和微调，成本高昂且难以适应动态变化的价值需求。

此外，虽然 **activation steering** 提供了一种轻量级的推理时控制机制，但现有方法大多仅针对单一价值进行干预，而多值联合控制通常采用固定强度的方向叠加（fixed vector composition），无法响应模型内部状态的变化，导致控制效果受限甚至出现冲突。

### 提出的新方法：AIMES
本文提出了 **AIMES**（Adaptive Intervention for Multi-Value Evaluation and Steering），一种基于因果激活引导的自适应多值控制系统。其核心思想是将多值 steering 视为一个**闭环反馈控制过程**，而非开环的固定干预。

#### 创新点：
- **Bipolar Value Directions**：基于 Moral Foundations Theory（MFT）构建了模型和层特定的双极性（bipolar）价值方向（如 Care/Harm, Fairness/Cheating），支持双向控制（增强或抑制）。
- **Online Observation via Vocabulary Readouts**：利用中间层的词汇空间读出（如 J-Lens）作为“观察器”（observer），实时估计当前各价值的表达强度，无需训练额外的状态估计器。
- **Observer-Guided Controller**：设计了一个控制器，在每个解码步动态调整各价值干预的强度，实现状态感知的自适应控制。

### 相比现有方法的优势
| 对比维度 | 固定联合 steering（Fixed Joint） | Prompt-based Steering | AIMES |
|--------|-------------------------------|----------------------|-------|
| 控制方式 | 开环，固定强度 | 依赖自然语言提示 | 闭环，状态自适应 |
| 干预粒度 | 静态 | 无显式干预 | 动态调整每一步 |
| 资源消耗 | 低 | 极低 | 低（无需训练） |
| 多值协调能力 | 弱（易冲突） | 中等（依赖提示工程） | 强（可平衡竞争目标） |

AIMES 在保持轻量的同时，实现了更精细、更鲁棒的多值协同控制。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Contrastive Vignette Pairs**：为每个 MFT 维度（Care, Fairness, Loyalty, Authority, Sanctity）构建了 200 对匹配的正负场景对，用于提取 value-specific steering directions。
- **Evaluation Prompts**：使用 100 个保留的 MFRC（Moral Foundations Recognition Corpus）提示，按基础价值类别过滤后用于不同 steering 目标测试。

### 实验设置
- **模型家族**：覆盖三个主流系列共五款 instruction-tuned 模型：
  - Gemma-3-4B / Gemma-3-12B
  - Qwen3-4B / Qwen3-14B
  - Llama-3.1-8B-Instruct
- **干预深度**：在每层 Transformer 层进行干预，系统扫描从早期到晚期的多个深度（共10个归一化深度点）。
- **Steering Objectives**：定义三种典型交互模式：
  1. `(↑Care ↑Fairness)` —— 协同促进（aligned）
  2. `(↑Loyalty ↑Authority)` —— 弱耦合
  3. `(↑Care ↑Fairness ↓Sanctity)` —— 竞争目标（competing）

### 评估指标
- **Matched Interaction Contrast (D)**：衡量新增价值约束带来的增量控制效果，类似差分中的差分（difference-in-differences）。
- **GeoGain**：ORBIT 提出的几何增益指标，只有当所有目标方向同时改善时才计分为正，用于评估**平衡控制能力**。
- **Prompt-Level Consistency**：统计在多少比例的 prompts 上 AIMES 表现优于基线。
- **Response Quality**：由 GPT-5.6-Sol 盲评生成质量（coherence, fluency, relevance）。
- **Realized Perturbation**：测量实际施加的激活空间扰动大小。

### 基线方法对比
- **Fixed Multi-Value Steering**：使用相同 value directions 和干预位置，但以固定系数联合应用。
- **Prompt Steering**：通过自然语言指令表达相同多值目标，不修改内部激活。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 多值可控性随组合与深度变化
- 不同 value 组合表现出显著不同的控制难度：
  - `Care & Fairness` 正向对齐，易于协同增强；
  - `Loyalty & Authority` 弱相关或负相关，控制更具挑战；
  - `Sanctity` 与 `Care/Fairness` 存在正向几何耦合，反向抑制时存在内在张力。

#### ✅ AIMES 在多个维度上优于基线
| 比较项 | 结果摘要 |
|-------|---------|
| **vs. Fixed Joint Steering** | 在约 30% 深度处，**5/5 模型**均显示出显著的 Care/Fairness 保留优势（Dx > 0）；在 60% 深度附近出现 Sanctity 抑制优势（Dy > 0） |
| **vs. Prompt Steering** | Care/Fairness 保留优势在 30%–90% 深度持续存在，且 **5/5 模型在多个深度一致胜出** |
| **GeoGain（平衡控制）** | 在 `(↑Care ↑Fairness ↓Sanctity)` 目标下，相比 Prompt Steering，**25/25 设置均 favor AIMES**；相比 Fixed Joint，则有 15–16/25 设置占优，显示更强的平衡控制能力 |

#### ✅ 更小的激活扰动，相当的生成质量
- **Realized Perturbation Reduction**：AIMES 的实际激活更新幅度比 Fixed Joint 小 **8%–87%**，说明其通过智能调节实现了“四两拨千斤”的高效控制。
- **Response Quality**：在 150 个设置中，AIMES 在 76 项上质量高于 Fixed，81 项高于 Prompt，且在 **relevance 上显著更优（106/150）**，表明未牺牲输出质量。

#### ✅ 自适应控制器确实在动态调整
- **Within-Response Coefficient Range**：在 30% 深度下，各模型平均系数变动范围为 **0.45–0.83**，远大于零，证明控制器在整个生成过程中持续调整干预强度。

### 消融实验结果
- **Observer Ablation (J-Lens vs. Logit Lens)**：
  - J-Lens 在所有 5 个模型上都表现出更高的 **directional responsiveness**（方向响应性）。
  - 下游任务中，J-Lens 在 worst-target gain、non-target interference 和 response quality 上普遍更优。
  - 支持选择 J-Lens 作为默认 observer。

---

## 4. 关键结论和发现

### 主要发现
1. **多值控制具有深度依赖性和组合敏感性**：不同 value 的表示关系（如正交、对齐、冲突）随网络深度和模型架构变化，不能简单假设独立。
2. **自适应反馈机制显著提升控制效果**：相比固定干预和 prompt 方法，AIMES 能更好地协调多值目标，尤其在处理竞争性目标（如 ↑Care ↓Sanctity）时表现突出。
3. **无需训练即可实现状态感知控制**：利用 J-Lens 等 vocabulary readout 作为在线 observer，避免了训练 probe 或 classifier 的开销，是一种真正轻量的解决方案。
4. **更小扰动实现更好控制**：AIMES 通过动态调节实现了比固定干预更高效的控制，且不损害生成质量。

### 方法的局限性
- **依赖预定义的价值体系**：目前仅基于 Moral Foundations Theory 的五个维度，尚未扩展至更广泛的人类价值观空间。
- **观察器的有效性边界未知**：尽管 J-Lens 表现良好，但 representational visibility 不等于 causal steerability，某些深层语义可能无法被 readout 捕获。
- **评估依赖黑盒评判器**：使用 GPT-5.6-Sol 和 Claude-Opus-4.8 进行盲评，虽独立但仍存在主观偏差和 evaluator sensitivity（例如 Sanctity 抑制优势的具体深度因 evaluator 而异）。

### 未来工作方向
- 扩展至更多元、更复杂的价值空间（如文化多样性、伦理困境）。
- 探索更丰富的 observer 设计，如结合 attention map 或 function vector。
- 构建 geometry-aware 控制器，主动利用 value 方向间的夹角关系进行优化调度。
- 探索跨模态或多轮对话中的长期价值一致性控制。

---

> **总结一句话**：  
> AIMES 通过引入基于中间层 readout 的闭环反馈机制，首次实现了无需训练的、状态感知的轻量级多值 activation steering，在控制精度、效率和平衡性上全面超越固定干预与提示工程方法，为 LLM 的可解释、可调控对齐提供了新范式。

</details>

---

### 16. [QSV: Quat-Sphere-Vision for Coupled Quaternion Attention on Spherical Lattices](https://arxiv.org/abs/2609.30592)

**Authors**: Nicholas Foley, Devin Marinelli, Donny Moore, Diego Enriquez, Amanda Fernandez  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.30592v1  

#### Abstract
In standard attention, three separately learned projections decide how strongly a token attends to each neighbor ($W_Q$, $W_K$) and how the attended features are transformed before aggregation ($W_V$). We study Quat-Sphere-Vision (QSV), a sparse spherical vision model that replaces this projection t...

---

### 17. [Brenier Meets Adversarial Training: Optimal Transport Geometry for Robust Learning](https://arxiv.org/abs/2609.31363)

**Authors**: Alireza Abdollahpoorrostam, Ehsan Sharifian, Buse \c{S}en, Marco Cuturi, Daniel Kuhn  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.31363v1  

#### Abstract
Distributionally robust optimization (DRO) provides a principled framework for learning under distribution shift, but its practical use is hindered by the difficulty of evaluating worst-case risks for nonconvex loss functions. We study a penalized DRO formulation in which the adversary may choose an...

---

### 18. [Why Jailbreaks Succeed in Diffusion Language Models: An Energy Landscape Analysis](https://arxiv.org/abs/2609.30841)

**Authors**: Thong Bach, Dung Nguyen, Thao Minh Le, Truyen Tran  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.30841v1  

#### Abstract
Existing attacks and defenses for diffusion-based large language models (dLLMs) target specific vulnerabilities but lack a shared framework explaining why attacks succeed. We propose one by interpreting safety alignment as shaping the denoising energy landscape: a well-aligned model routes harmful q...

---

### 19. [MoMHa: Multi-Objective Optimization of LLM Harnesses over Accuracy, Safety, and Tokens](https://arxiv.org/abs/2609.30967)

**Authors**: Subhojyoti Mukherjee, Md Mehrab Tanjim  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.30967v1  

#### Abstract
Most work on improving large language models treats accuracy as the sole objective. We argue that the harness, the Python code surrounding the model that constructs prompts, routes calls, and parses outputs, is a first-class design surface whose quality is inherently multi-objective: an accurate har...

---

### 20. [Up and Down the Abstraction Ladder: Code-Based Skills for Language Agents](https://arxiv.org/abs/2609.31076)

**Authors**: Bart{\l}omiej Cupia{\l}, Jens Tuyls, Maciej Wo{\l}czyk, Davide Paglieri, Martin Klissarov, Benjamin Eysenbach, Piotr Mi{\l}o\'s, Karthik R. Narasimhan  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.31076v1  

#### Abstract
Language agents struggle to act and learn in environments that require long sequences of low-level actions. Code-based abstractions can make these agents more productive by letting them invoke reusable skills instead of repeatedly selecting individual actions. The code handles recurring local decisi...

---

### 21. [Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events](https://arxiv.org/abs/2609.30746)

**Authors**: Isabella S. Thiel, Juan Bello-Rivas, Yannis G. Kevrekidis, Themistoklis P. Sapsis  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.30746v1  

#### Abstract
Extreme events in chaotic systems are difficult to learn from short trajectories because they are controlled by transient finite-time instability rather than by frequently observed bulk dynamics. We propose a mechanism-aware conditioning plug-in framework that turns a nudged coarse ensemble into a n...

---

### 22. [Rank-Reliable Teacher-Guided Fitness Approximation for Expensive Evolutionary Optimization: A TinyML Architecture Search Study](https://arxiv.org/abs/2609.30553)

**Authors**: Soumen Garai, Suman Samui  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.30553v1  

#### Abstract
Expensive evolutionary search does not always need an exact fitness estimate for every candidate. It often needs a reliable answer to a simpler question: which candidate is better? We address this need through Teacher-Guided Learning NSGA-II (TGL-NSGA-II), a low-fidelity framework for constrained Ti...

---

### 23. [Selective Amortization of Full-Budget Counterfactual Reasoning for Visual Token Communication](https://arxiv.org/abs/2609.30756)

**Authors**: Qinglei Qi, Zhihe Liang, Fengzhan Jing, Shenao Zhu, Lei Zhang, Chenyang Zhang, Shuqing He, Jia Guo  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.30756v1  

#### Abstract
Generative image communication transmits compact semantic tokens under a limited packet budget, where token selection directly affects the final reconstruction quality after the complete packet is decoded. However, accurately estimating the terminal value of every candidate token requires repeated r...

---

### 24. [PTC-Decoder: Towards Intelligent SLMs on Offline Resource-Constrained Edge Devices](https://arxiv.org/abs/2609.30836)

**Authors**: Minghui Yu, Ke Mu, Gang Wu  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.30836v1  

#### Abstract
Deploying small language models (SLMs) on offline, resource-constrained edge devices such as remote sensing satellites presents a fundamental challenge: their limited reasoning capacity hinders reliable execution of multi-step agent tasks requiring complex tool orchestration. Existing plan-solve par...

---

### 25. [Same Text, Different Numbers: The Divergence of LLM-Based Measures](https://arxiv.org/abs/2609.31013)

**Authors**: Hamid Boustanifar, Sasan Mansouri  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.31013v1  

#### Abstract
Researchers increasingly use generative large language models (LLMs) to convert corporate text into empirical variables. We examine the extent to which LLM-based textual measures are invariant to model choice using thirteen measures, including sentiment, management clarity, uncertainty, answer speci...

---

### 26. [AtomWorld-Mem: Memory-Restored World States for Long-Horizon Atomistic Evolution](https://arxiv.org/abs/2609.31133)

**Authors**: Tian Luo, Ruge Zhang, Haozhi Han, Yifrng Chen, Yunquan Zhang, Yunxin Liu, Ting Cao, Kun Li  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.31133v1  

#### Abstract
High-fidelity atomistic evolution over long timescales requires more than observing the current crystal configuration. Instantaneous atomistic snapshots are often incomplete: locally similar configurations can correspond to different hidden dynamical contexts, future event preferences, and waiting-t...

---

### 27. [Accounting for Bias Enables Sustainable LLM Evaluation](https://arxiv.org/abs/2609.31184)

**Authors**: Harshita Katoch, David Antony Selby, Gerrit Gro{\ss}mann, Sebastian Vollmer  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.31184v1  

#### Abstract
LLM-as-a-judge has become the de facto standard for scalable, subjective evaluation, yet current leaderboards compensate for systematic measurement bias by running ever more comparisons, an approach that is both statistically unsound and computationally wasteful. The root cause is an incomplete meas...

---

### 28. [Segment-Level Agentic Topic Modeling for Improved Data Exploration and Resource Efficiency](https://arxiv.org/abs/2609.31460)

**Authors**: Myeongjun Erik Jang, Antonios Georgiadis, Sae Young Moon, Fran Silavong  
**Category**: cs.AI  
**Published**: 2026-09-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.31460v1  

#### Abstract
Topic modeling is an effective technique for discovering hidden themes within documents and is widely used in text mining and data analysis across a variety of industry sectors. Recently, large language model (LLM)-based topic models have been emerged that prompt LLMs to generate topics then assign ...

---

### 29. [AutoResearch at Production Scale: Failure Modes and a Multi-Agent Framework](https://arxiv.org/abs/2609.30541)

**Authors**: Aparajith Chandran, Juwon Kim, Saurav Jha, Pablo Castells, Florian Hottier  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.30541v1  

#### Abstract
Optimizing embedding systems for production recommendation pipelines demands systematic exploration that consumes disproportionate engineering effort at scale. We apply Andrej Karpathy's AutoResearch paradigm -- a large language model that iteratively edits a training script and retains modification...

---

### 30. [GyroNovo: Error-Guided Fragment Imputation with Mass-Aware Attention for \textit{De Novo} Peptide Sequencing](https://arxiv.org/abs/2609.30542)

**Authors**: Abdellah El Mekki, Laks V. S. Lakshmanan, Muhammad Abdul-Mageed  
**Category**: cs.LG  
**Published**: 2026-09-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.30542v1  

#### Abstract
De novo peptide sequencing from tandem mass spectra is essential for identifying peptides without relying on reference databases. Despite advances in deep learning, accurate sequencing remains challenging because experimental spectra are often sparse, noisy, and incomplete, leaving informative b- an...

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

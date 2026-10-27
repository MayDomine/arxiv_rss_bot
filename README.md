# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-27 11:00:05 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Learning Structured Sparsity for Large Models: Bridging Model Compressibility and Efficient Inference](https://arxiv.org/abs/2610.31897)

**Authors**: Tianxiang Xia, Kai Lion, Niao He, Michael Muehlebach, Melanie Zeilinger, Hao Ma  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2610.31897v1  

#### Abstract
Scaling laws continue to reward increases in model scale, while the resulting computational and memory demands drive greater reliance on cloud infrastructure for inference. Yet unreliable network connectivity, communication delays, and privacy requirements can make such reliance impractical, motivat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning Structured Sparsity for Large Models: Bridging Model Compressibility and Efficient Inference

## 1. 论文的主要贡献和创新点

### 解决的问题
- **模型压缩与高效推理的脱节**：现有的模型压缩方法（如 unstructured sparsity）虽然减少了参数量，但由于不规则的内存访问模式，导致在实际硬件上的推理延迟并未显著降低，甚至可能增加。
- **弹性压缩需求**：边缘设备内存资源多样，需要单一检查点支持多种压缩预算（elastic compression），而现有方法难以实现这一点。
- **后处理转换的性能损失**：直接对已训练好的模型施加硬件友好的结构化稀疏模式（如 block sparsity），会导致模型性能严重下降。

### 提出的新方法：S-SALAAD
本文提出了 **S-SALAAD**（Structured-Sparse-And-Low-Rank Adaptation via ADMM），一个在训练期间联合学习模型权重及其低秩（low-rank）和**结构化稀疏**（structured sparse）组件的统一框架。

**核心创新点**：
- **结构化稀疏学习**：与传统的 unstructured sparsity 不同，S-SALAAD 在训练过程中诱导 **block sparsity**、**N:M sparsity** 和 **row-/column-wise sparsity** 等硬件友好的稀疏模式。这使得压缩后的模型能够被专用的 CPU/GPU 内核高效执行。
- **端到端的解决方案**：S-SALAAD 是一个“plug-and-play”框架，它基于 ADMM 优化，无需改变模型架构或优化器更新规则，即可在 pretraining 阶段同时学习结构。
- **高效的训练与推理实现**：
  - **I-SVD**：提出了一种迭代奇异值分解（Iterative SVD）方法，大幅降低了 low-rank 更新的计算开销，使训练时间仅比 vanilla 训练高出 4%。
  - **EGAR**：提出能量引导交替精炼（Energy-Guided Alternating Refinement）算法，支持**零样本**（zero-shot）的预算感知压缩，无需校准数据或再训练。
  - **专用内核**：开发了针对 block-sparse 模型的专用 CPU（AVX2）和 GPU（TileLang）推理内核，有效利用了结构化稀疏带来的计算优势。

### 相比现有方法的优势
| 方面 | S-SALAAD | 现有方法（如 SALAAD, LOST） |
| :--- | :--- | :--- |
| **稀疏性** | 结构化（hardware-friendly） | 通常是无结构的（unstructured） |
| **推理效率** | 通过专用内核实现真正的低延迟 | 低参数量 ≠ 低延迟，常因不规则访存而变慢 |
| **弹性部署** | 单一检查点支持多预算压缩 | 通常固定模型大小，缺乏弹性 |
| **训练兼容性** | 与现有 optimizer 兼容 | 通常需要修改训练流程 |

## 2. 核心实验方法和设置

### 使用的数据集
- **Llama2 预训练**：使用 **C4** 数据集从零开始预训练 Llama2-350M 和 Llama2-1B 模型。
- **Qwen3 蒸馏**：使用 **FineWeb-Edu** 数据集，将 Nemotron-3-Super-120B 模型的知识蒸馏到 Qwen3-1.7B 模型中。

### 实验设置和评估指标
- **模型**：Llama2-350M, Llama2-1B, Qwen3-1.7B。
- **硬件平台**：
  - GPU: Quadro RTX 6000, RTX 2080 Ti
  - CPU: 4× AMD EPYC 7662 CPUs
- **评估指标**：
  - **Perplexity**：衡量模型的语言建模能力。
  - **Peak Memory**：推理时的峰值内存占用（GPU Memory / RAM）。
  - **Prefill Latency / Throughput**：推理延迟和吞吐量，是衡量实际效率的关键。
  - **Zero-shot Accuracy**：在 PIQA, SciQ, WinoGrande 等下游任务上的准确率。

### 基线方法对比
- **Vanilla**：标准训练的密集模型。
- **SALAAD**：学习无结构稀疏和低秩（unstructured SLR）的基线。
- **LORO**：仅学习低秩（L-only）的基线。
- **Wanda, SparseGPT, SliceGPT, SVD-LLM**：主流的后训练压缩（post-training compression）方法。

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **Llama2-1B** 模型上，与 **dense model** 相比，S-SALAAD 在 **500M** 参数预算下实现了：
- **内存节省**：峰值 GPU 内存从 **4.08 GB** 降至 **2.35 GB**，减少了 **42.4%**。
- **吞吐量提升**：吞吐量从 **29.35 ktoken/s** 提升至 **58.47 ktoken/s**，达到了 **1.99×** 的加速。

在 **Qwen3-1.7B** 蒸馏模型上，与 **dense model** 相比，S-SALAAD 在 **800M** 参数预算下实现了：
- **内存节省**：峰值 GPU 内存从 **5.58 GB** 降至 **3.82 GB**，减少了 **31.5%**。
- **吞吐量提升**：吞吐量从 **24.19 ktoken/s** 提升至 **39.22 ktoken/s**，达到了 **1.62×** 的加速。

### 与基线方法的对比结果
- **Pareto 前沿优势**：在 **Perplexity-Memory-Latency** 三维权衡中，S-SALAAD 的所有检查点都显著超越了所有基线方法，形成了新的经验帕累托前沿（empirical Pareto frontier）。
- **优于 LORO**：尽管 LORO 也达到了类似的帕累托前沿，但它需要为每个目标大小训练一个独立的检查点。而 S-SALAAD 仅用一个检查点就能覆盖从 500M 到 900M 的多个预算，实现了真正的弹性部署。
- **后训练压缩的协同效应**：将 SliceGPT 或 SVD-LLM 应用于 S-SALAAD 预训练的 `L-only` 检查点，其性能优于应用于 vanilla 预训练的检查点，证明了 S-SALAAD 学习到的可压缩性是通用的。

### 消融实验结果
- **结构化稀疏的必要性**：实验表明，对 unstructured SALAAD 检查点强行施加 block 结构（post-hoc conversion）会急剧损害模型性能（如 Table 2 所示，500M 时困惑度从 14.11 恶化到 1497.68），这验证了在训练中诱导结构的必要性。
- **块大小（Block Granularity）的影响**：
  - **小模型**：对于 Llama-350M，较小的块（如 8×8）在紧缩预算下能更好地保持质量。
  - **大模型**：对于 Llama-1B，较大的块（如 64×64）在推理效率上更具优势，且能很好地匹配 `L-only` 的性能。这表明**更大的模型可以容忍更粗粒度的结构**，从而获得更高的推理效率。
- **EGAR 的有效性**：与贪心的 HPA 方法相比，EGAR 在所有参数预算下都能取得更低的困惑度，证明了其自适应分配容量的有效性。

## 4. 关键结论和发现

### 主要发现
1. **结构化学习是桥梁**：S-SALAAD 成功地将模型的**可压缩性**（compressibility）与**高效的边缘推理**（efficient edge inference）连接起来。它证明了通过在训练中学习结构化稀疏，可以将静态的内存节省转化为动态的推理延迟降低。
2. **单一检查点的弹性部署**：S-SALAAD 实现了真正的弹性压缩，一个检查点可以零样本地压缩到多个不同的内存预算，极大地简化了边缘部署的流程。
3. **模型规模与结构粒度的权衡**：更大的模型能够更好地适应更粗粒度的结构化稀疏（如 64×64 block），这为扩展到更大模型提供了经验指导。

### 方法的局限性
- **目前只支持 Block Sparsity 的专用内核**：虽然框架支持多种结构，但目前只为 block sparsity 开发了专用内核，其他模式（如 N:M）尚未得到同等优化。
- **训练开销**：尽管 I-SVD 极大地降低了开销，但相比 vanilla 训练，S-SALAAD 仍然引入了额外的计算负担。
- **适用范围**：该方法主要针对 Transformer 中的 linear projections，对其他组件（如 embedding）的压缩效果有限。

### 未来工作方向
- **扩展到更多结构**：为 N:M sparsity 和 row-/column-wise sparsity 开发更高效的专用推理内核。
- **探索更优的结构模式**：寻找更适合特定硬件（如移动芯片、TPU）的新型结构化稀疏模式。
- **与其他压缩技术结合**：将 S-SALAAD 与量化（quantization）等技术结合，实现更极致的模型压缩。
- **理论分析**：深入研究为何更大的模型能更好地适应粗粒度结构，建立更坚实的理论基础。

</details>

---

### 2. [SimPOTrans: Self-Adaptation for Offline-to-Simultaneous Speech Translation via Reinforcement Learning](https://arxiv.org/abs/2610.35042)

**Authors**: Haotian Tan, Hiroki Ouchi, Sakriani Sakti  
**Category**: cs.CL  
**Published**: 2026-10-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.35042v1  

#### Abstract
Simultaneous speech translation (SimulST) aims to translate ongoing speech with low latency while preserving translation quality. Recent studies have adopted large language models (LLMs) for SimulST and formulated it as a multi-turn dialogue task for efficient generation, achieving state-of-the-art ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SimPOTrans: Self-Adaptation for Offline-to-Simultaneous Speech Translation via Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **SimulST**（Simultaneous Speech Translation）方法通常依赖于：
- 合成的 read/write 轨迹（suboptimal training signals）
- 联合优化翻译质量和延迟（competing objectives 导致优化不稳定）

这些方法需要复杂的训练流程、额外的时间对齐监督或人工设计的决策策略，限制了模型的灵活性和泛化能力。

### 🚀 提出的新方法：SimPOTrans
提出了一种基于 **Reinforcement Learning (RL)** 的两阶段后训练框架——**SimPOTrans**，实现从 **offline ST 模型** 到 **dialogue-style SimulST** 的自适应迁移。

#### 核心思想：
- 不依赖合成轨迹或显式延迟优化。
- 通过 **Group Relative Policy Optimization (GRPO)** 强化模型在部分语音输入下自发产生的“适合实时翻译”的行为。
- 将 SimulST 视为多轮对话任务，逐步生成增量翻译。

### 🔍 创新点
1. **Self-Adaptation via Self-Generated Trajectories**  
   模型在训练中自我生成多轮翻译轨迹，并从中学习哪些输出模式更适合 SimulST，无需外部轨迹标注。

2. **Two-Stage RL Post-Training Pipeline**
   - **RL-Stage I**: 在完整语音上优化整体翻译质量（使用 COMET 奖励）。
   - **RL-Stage II**: 在分块语音输入下进行多轮生成，引入 **monotonicity reward** 和 **repetition penalty**，引导模型学会有序、低重复的实时翻译行为。

3. **无需外部决策策略，灵活控制质量-延迟权衡**
   - 推理时只需调整输入语音的 **chunk size** 即可动态调节延迟。
   - 无需额外的 wait-k 或 attention-based 决策模块。

### 💡 相比现有方法的优势
| 方法 | 是否需要合成轨迹 | 是否需显式延迟优化 | 是否需外部决策策略 | 灵活性 |
|------|------------------|--------------------|---------------------|--------|
| InfiniSST / EASiST | ✅ 是 | ❌ 否 | ❌ 否 | 中等 |
| HPO | ✅ 是 | ✅ 是 | ❌ 否 | 中等 |
| MoSST / LA | ❌ 否 | ❌ 否 | ✅ 是 | 低 |
| **SimPOTrans** | ❌ 否 | ❌ 否 | ❌ 否 | ✅ 高 |

> SimPOTrans 是首个完全摆脱合成轨迹、显式延迟目标和外部决策策略的 LLM-based SimulST 方法。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练数据**：CoVoST 2（约 430 小时）
  - 三个语言方向：**En→De**, **En→Zh**, **En→Ja**
- **测试基准**：
  - **ACL 60/60**：科学讲座翻译数据集
  - **MCIF development set**：IWSLT 2026 同传赛道开发集，用于低延迟评估

### ⚙️ 模型配置
- **Backbone**：
  - Speech Encoder: Whisper-large-v3
  - Adaptor: Two-layer MLP
  - LLM: Qwen3-8B（LoRA 微调，rank=16, scale=32）
- **RL 设置**：
  - 使用 **GRPO** 进行策略优化
  - Group size: 12（Stage I）、8（Stage II）
  - Chunk size in RL-Stage II: 2.0 秒
  - 质量奖励函数：XCOMET-XL

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **XCOMET-XL** | 学习型参考评价指标（0~1），越高越好 |
| **MetricX-24** | 学习型评价指标，预测错误分数（0~25），越低越好 |
| **BLEU** | 传统 n-gram 匹配指标，用于与先前工作对比 |
| **LongYAAL (NCA)** | 非计算感知平均滞后，衡量延迟，单位为秒 |

### 🆚 基线方法
- **SimulSense**：基于独立决策模块触发翻译
- **InfiniSST**：基于合成 read/write 轨迹的对话式 SimulST
- **HPO**：在 InfiniSST 上应用 RL 联合优化质量和延迟
- **IWSLT 2026 提交系统**（NeMo, CUHKSZ 等）：作为非受控对比

> 所有基线均使用相同 backbone 和训练数据复现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Figure 2 & Table 1）

#### 在 ACL 60/60 上的质量-延迟权衡（Figure 2）
- **SimPOTrans 在 COMET 和 MetricX-24 上全面超越所有基线**，尤其在 **En→Zh** 和 **En→De** 方向上优势明显。
- 在低延迟区域（LongYAAL < 2s），SimPOTrans 表现出更强的鲁棒性和更高的上限性能。
- **BLEU 分数显著低于其他方法**，但作者指出这是因模型进行了更多语义重组所致，反而有利于 SimulST。

#### 在 MCIF 上的低延迟性能（Table 1）
| Lang | System | COMET | LongYAAL |
|------|--------|-------|----------|
| En→De | **SimPOTrans** | **0.87** | **1.6s** |
| En→De | NeMo* | 0.93 | 2.0s |
| En→De | CUHKSZ | 0.85 | 1.5s |
| En→Zh | **SimPOTrans** | **0.76** | **1.7s** |
| En→Zh | CUHKSZ | 0.74 | 1.5s |

> 注：NeMo 使用 PDF 上下文辅助，属于 cascaded 系统；SimPOTrans 仅用音频输入即取得接近甚至更好的性能。

### 🔍 消融实验结果（Figure 3）

#### (a) 不同训练阶段的影响
- **SFT 和 RL-Stage I** 模型在部分语音输入下表现差，曲线不规则。
- **RL-Stage II** 后形成平滑的质量-延迟曲线，证明其对 SimulST 自适应至关重要。

#### (b) 辅助奖励的作用
- 单独使用 **monotonicity reward** 改进有限。
- **repetition penalty** 显著减少重复错误。
- 二者结合（+Mono.+Rep.）带来最稳定的性能提升，验证互补效应。

#### (c) RL-Stage II 中 chunk size 的影响
- **2 秒 chunk** 表现最佳，平衡上下文与实时性。
- 1 秒 chunk 上下文不足，3 秒 chunk 在低延迟下性能下降明显。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **SimPOTrans 成功实现了 offline-to-simultaneous 的自适应**
   - 无需合成轨迹或显式延迟建模，即可让 offline 模型掌握 SimulST 所需的行为模式。

2. **诱导出更适合 SimulST 的翻译行为**
   - 提高了翻译的 **monotonicity**（词序一致性），使输出更贴近源语顺序，利于流式生成。
   - 减少了跨轮次的 **repetition**，提升了连贯性。

3. **BLEU 与语义指标出现分歧**
   - SimPOTrans 的 BLEU 较低，但 COMET/MetricX 更高，说明其输出虽偏离参考译文形式，但语义更完整且更适合实时场景。
   - 如 Figure 5 示例所示，SimPOTrans 将 "because of..." 改写为 "这要归功于…"，虽降低 n-gram 匹配，但更自然且符合口语习惯。

4. **推理灵活高效**
   - 仅通过调节输入 **chunk size** 即可控制延迟，无需重新训练或切换策略。

### ⚠️ 局限性
1. 当前使用 **固定时长分块**（fixed-duration chunks），可能导致语义单元被切断。
2. 仅评估了 **speech-to-text** 翻译，未扩展到 **speech-to-speech** 场景。
3. 训练依赖高质量的离线 ST 数据，对噪声或口音鲁棒性有待验证。

### 🔮 未来工作方向
1. 探索 **boundary-aware chunking** 策略（如基于语义边界或韵律停顿）。
2. 将 SimPOTrans 扩展至 **end-to-end speech-to-speech translation**。
3. 结合 **human preference feedback** 进一步优化生成风格与可理解性。
4. 探索在 **low-resource** 场景下的迁移能力。

---

## 总结
> **SimPOTrans** 提出了一种简洁而强大的 RL 框架，首次实现了 **无需合成轨迹、无需显式延迟优化、无需外部决策策略** 的 offline-to-simultaneous 自适应。它不仅在多个语言方向上取得了优越的质量-延迟权衡，还揭示了传统指标（如 BLEU）在评估 SimulST 时的局限性，推动社区关注更适合实时翻译的评价标准。

</details>

---

### 3. [FrontGI: Architecture-Aware Parallel GNN Inference on Chiplet-Based Multicore Systems](https://arxiv.org/abs/2610.34911)

**Authors**: Binglin Ji, Chenfeng Zhao, Roger Chamberlain  
**Category**: cs.DC  
**Published**: 2026-10-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.34911v1  

#### Abstract
The realm of high-performance computing has expanded to include machine learning tasks of many forms. Graph Neural Networks (GNNs) are becoming increasingly popular, with their applications expanding across diverse domains. As the scale of graph data continues to grow, including larger numbers of no...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FrontGI: Architecture-Aware Parallel GNN Inference on Chiplet-Based Multicore Systems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现代多核系统（尤其是基于 **chiplet 架构**的处理器）虽然拥有上百个核心，但现有的主流 GNN 库（如 DGL 和 PyG）在这些平台上表现出**严重的可扩展性瓶颈**，尤其是在执行 GNN 推理任务时。其根本原因在于：

- **内存访问不规则性**：GNN 的 AGGREGATE 阶段涉及稀疏图遍历，导致大量 LLC（Last-Level Cache）缺失。
- **硬件架构感知不足**：传统库依赖 OpenMP 进行简单的节点级并行（node-level parallelism），忽略了 chiplet 架构中分布式 LLC 的特性，造成高昂的跨 chiplet 通信开销。
- **缺乏对 UPDATE 阶段的优化**：现有 BLAS 实现假设共享缓存模型，与 chiplet 的物理分布缓存不符，导致性能下降。

### 提出了什么新方法或新思路

本文提出了 **FrontGI**，一套面向 chiplet-based 多核系统的架构感知并行 GNN 推理框架，核心创新如下：

#### （1）两种图结构自适应的并行策略

- **Hierarchical Node Parallelism（分层节点并行）**  
  适用于**稀疏图**（平均度 < 20）。通过将节点分区映射到不同 chiplet，并利用自定义线程亲和性（`sched_setaffinity`）确保计算局部性，减少跨 chiplet 通信。

- **Mixed Tensor/Node Parallelism（混合张量/节点并行）**  
  适用于**稠密图**（平均度 > 50）。在 AGGREGATE 阶段引入 **tensor parallelism**，将特征维度切分到各个 chiplet 上，降低单个 chiplet 的缓存压力，提升 LLC 利用率。

#### （2）UPDATE 阶段的 BLAS 优化

针对 chiplet 架构中 LLC 分布式的现实，提出一种轻量级并行 UPDATE 策略：
- 将矩阵乘法（GEMM）按行分块，每个 chiplet 独立执行 BLAS 调用。
- 避免修改底层 BLAS 库，同时规避跨 chiplet 缓存同步开销。

#### （3）线性算子融合（Operator Fusion）

对于 GIN 等模型中存在的 AXPY 类线性操作，将其与 AGGREGATE 阶段融合，减少中间数据传输。

---

### 相比现有方法的优势

| 维度 | FrontGI | DGL / PyG |
|------|--------|----------|
| **架构感知** | 显式建模 chiplet 和分布式 LLC | 忽略硬件拓扑，仅抽象为多核 |
| **并行粒度** | 动态选择 node 或 tensor 并行 | 固定 node-level 并行 |
| **缓存效率** | 显著降低 LLC miss | 高频 LLC miss 导致性能饱和甚至倒退 |
| **可扩展性** | 在 128 核系统上持续扩展 | 超过 64 核后性能停滞或下降 |
| **内存友好性** | 更好地利用大主存优势 | 在大型图上频繁 OoM |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

共使用 **12 个大规模图数据集**，涵盖多样化的图结构属性：

| 数据集 | 节点数 | 边数 | 平均度 | 特征 |
|--------|--------|------|--------|------|
| **MT** (OGBG-Moltox21) | 145K | 302K | 2.1 | 化学分子图 |
| **MH** (OGBG-Molhiv) | 1.05M | 2.26M | 2.2 | 化学分子图 |
| **AX** (OGBN-Arxiv) | 169K | 1.17M | 6.9 | 学术引用网络 |
| **YP** (Yelp) | 717K | 13.9M | 19.5 | 社交评论图 |
| **KA** (Wikipedia-ka) | 160K | 4.89M | 30.6 | 网页链接图 |
| **S19** (Graph500-Scale19) | 335K | 15.5M | 46.1 | 合成图 |
| **PC** (Petster-Carnivore) | 149K | 10.9M | 72.8 | 社交网络 |
| **HU** (Wikipedia-hu) | 622K | 46.4M | 74.6 | 多语言网页 |
| **RT** (Reuters) | 781K | 121M | 155.1 | 新闻共现图 |
| **YH** (Yahoo Music) | 1.00M | 257M | 256.8 | 用户行为图 |
| **RD** (Reddit) | 233K | 115M | 492.9 | 社交论坛 |
| **PT** (OGBN-Proteins) | 133K | 79.1M | 597.0 | 生物蛋白质图 |

> 所有图嵌入维度统一设为 1024，使用 float32。

---

### 实验设置和评估指标

- **硬件平台**：AMD EPYC 9754 服务器（128 核 @ 2.25GHz），16 个 chiplet，每个 chiplet 共享 16MiB L3 缓存，128GiB 主存。
- **软件环境**：Rocky Linux 9.5，GCC 11.5.0，OpenMP 4.5，SMT 关闭。
- **评估模型**：GCN、GraphSAGE、GIN、SGC。
- **评估指标**：
  - 单层推理时间（s）
  - 相对于 DGL 的 **Speedup**
  - LLC Hit/Miss 率
  - 可扩展性（8~128 核）

---

### 基线方法对比

- **DGL v2.4.0**：主流 GNN 库，使用 OpenMP 多线程。
- **PyG v2.6.1 + PyTorch 2.3.1**：另一主流库，常因内存占用过高在大图上出现 **OoM** 错误。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 整体加速比（vs DGL）

| 模型 | 最大 Speedup |
|------|------------|
| **GCN** | **3.35×** |
| **GraphSAGE** | **3.20×** |
| **GIN** | **2.89×** |
| **SGC** | **3.10×** |

> 加速比在 **PT**（OGBN-Proteins）和 **RD**（Reddit）等高平均度图上最为显著。

#### vs PyG 的优势更加明显

- 在多个大图上 PyG 出现 **OoM**（如 HU、RT、YH、RD、PT）。
- 在可运行的情况下，FrontGI 相比 PyG 最高可达 **15.76×** 加速（SGC on PC）。

---

### 与基线方法的对比结果

#### 可扩展性对比（Fig. 14）

- **DGL**：在超过 64 核后性能增长停滞，**PT 图在 96 核后性能下降**。
- **FrontGI**：持续扩展至 128 核，**PT 图达到 20.5× 加速比**（相对于 8 核）。

#### 缓存行为分析（Table 8 & 9）

| 图类型 | 方法 | LLC Miss Reduction |
|--------|------|------------------|
| 高度图（PT） | Mixed Parallelism | **87.54%** |
| 高度图（RD） | Mixed Parallelism | **75.52%** |
| 低度图（MT） | Hierarchical Node | **74.28%** |
| 低度图（MH） | Hierarchical Node | **76.36%** |

> 表明 FrontGI 显著减少了 LLC 缺失，提升了数据局部性。

---

### 消融实验结果

#### 并行策略选择依据（Fig. 9–12）

- **平均度 < 20**（如 MT、MH、AX）：**Hierarchical Node Parallelism** 更优（1.35× ~ 1.82×）。
- **平均度 > 50**（如 PC、RT、YH、RD、PT）：**Mixed Tensor/Node Parallelism** 显著胜出（2.89× ~ 3.35×）。
- **20 ≤ avg degree ≤ 50**（如 YP、KA、S19）：两种策略效果相近。

#### Overhead 分析（Eq. 8）

尽管 tensor parallelism 引入了切分与合并开销，但其带来的收益远大于开销：
- **Overhead-to-Gain Ratio (R_OTG)** 在 1.33% ~ 4.14% 之间，可忽略不计。

---

## 4. 关键结论和发现

### 论文的主要发现

1. **图的平均度是决定最佳并行策略的关键因素**，而非传统的幂律指数 α。
2. **chiplet 架构下的分布式 LLC 对 GNN 推理性能有根本性影响**，必须显式建模。
3. **FrontGI 的两种并行策略实现了高达 3.35× 的加速比**，且在 128 核上仍保持良好扩展性。
4. **PyG 在大图上普遍存在 OoM 问题**，而 FrontGI 能有效处理超大规模图（如 1M+ 节点）。
5. **UPDATE 阶段的 BLAS 实现必须适配 chiplet 架构**，否则将成为性能瓶颈。

---

### 方法的局限性

- 当前仅支持 **推理阶段**，未覆盖 GNN 训练。
- 并行策略选择目前依赖图统计特征（如平均度），尚未实现完全自动化的运行时决策。
- 仅在 AMD EPYC 系统上验证，对 Intel Sapphire Rapids 等 tile-based 架构的泛化性有待测试。
- 未探索更复杂的模型（如 GAT）中的注意力机制并行化。

---

### 未来工作方向

1. **扩展至 GNN 训练**：结合 FrontGI 的推理优化，构建端到端高效的训练系统。
2. **自动化并行策略选择器**：基于图特征和硬件配置，动态选择最优并行模式。
3. **支持更多 GNN 模型**：如 GAT、Transformer-based GNNs。
4. **集成到主流库中**：将 FrontGI 技术模块化，作为 DGL 或 PyG 的后端优化插件。
5. **探索异构系统支持**：结合 CPU + GPU + FPGA 的混合架构进行协同推理。

--- 

> ✅ **总结**：FrontGI 成功揭示了 chiplet 架构下 GNN 推理的性能瓶颈，并提出了一套简单、高效、可集成的解决方案，为下一代 GNN 系统设计提供了重要指导。

</details>

---

### 4. [Nested Sparsity for Scalable Pretraining and Continual Learning](https://arxiv.org/abs/2610.31818)

**Authors**: Tianzhe Chu, Druv Pai, Yi Ma  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.31818v1  

#### Abstract
Mixture-of-experts (MoE) transformers, the standard backbone for modern large-scale deep learning, employ conditional computation: only a sparse subset of model weights is used in each forward pass. In MoE transformers, this sparsity arises in the MoE operator, which generalizes the multi-layer perc...

---

### 5. [Movable-Element Multi-Functional RIS-Assisted Hybrid RSMA-NOMA: A Constrained Hybrid Reinforcement Learning Approach](https://arxiv.org/abs/2610.34701)

**Authors**: Li-Hsiang Shen, Chi-Te Kuo  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.34701v1  

#### Abstract
In this work, we propose a movable-element (ME) multi-functional reconfigurable intelligent surface (MF-RIS) architecture, termed MM-RIS, which integrates element mobility with adaptive signal reflection, transmission, amplification, and energy harvesting (EH) to enhance communication flexibility an...

---

### 6. [BiLoCo: Binary Low-Rank Corrections for Post-Training Quantization](https://arxiv.org/abs/2610.32243)

**Authors**: David Jin, Beshr IslamBouli, Tarushii Goel, Han Guo, Yoon Kim  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.32243v1  

#### Abstract
Post-training quantization (PTQ) to FP4 has emerged as a key technique for reducing large language model inference cost, especially on Blackwell GPUs, which offer native FP4 tensor core support. Direct PTQ to FP4, however, still leaves a noticeable accuracy gap. Recent work on SVDQuant mitigates the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# BiLoCo: Binary Low-Rank Corrections for Post-Training Quantization 总结

## 1. 论文的主要贡献和创新点

### 解决的问题
- **Post-training quantization (PTQ) to FP4** 是降低大语言模型（LLM）推理成本的关键技术，尤其是在支持原生 FP4 tensor core 的 Blackwell GPU 上。
- 然而，直接将权重和激活量化到 FP4 会导致明显的精度损失（accuracy gap），特别是当存在 outlier 权重或激活时。
- 现有方法如 SVDQuant 使用 BF16 低秩修正（low-rank correction）来吸收这些误差，但其修正分支仍运行在 tensor cores 上，与主 GEMM 竞争硬件资源，无法充分利用空闲的 CUDA cores。

### 提出的新方法：BiLoCo
- **Binary Low-Rank Correction (BiLoCo)**：提出一种新的低秩修正格式，其左右因子不再是高精度的 BF16 向量，而是二值化的 $\{\pm 1\}$ 符号向量（sign vectors），仅配合一个标量系数 $\alpha_i$。
- 具体形式为：
  $$
  W \approx R + \sum_{i=1}^{r} \alpha_i u_i v_i^\top, \quad u_i, v_i \in \{\pm 1\}
  $$
  其中 $R$ 是 NVFP4 量化的残差矩阵。

### 相比现有方法的优势
| 特性 | SVDQuant | BiLoCo |
|------|---------|--------|
| 存储效率 | BF16 因子，每 rank 占用 $16 \times$ 更多比特 | 二值因子，每 rank-one 组件节省 $16\times$ 存储 |
| 计算单元 | 运行在 tensor cores，与主 GEMM 竞争 | 运行在 CUDA cores，可与 FP4 tensor-core GEMM 并发执行 |
| 修正能力 | 少量高精度 rank（e.g., rank-32） | 大量低精度 rank（e.g., rank-512），覆盖更多方向 |
| 内核设计 | 分离或串行执行 | 实现 fused single kernel，在 GEMM epilogue 中合并结果 |

- **核心思想**：利用相同的存储预算，用大量一比特的 rank-one 组件替代少量高精度组件，从而更好地捕捉平坦谱（flat spectrum）的 LLM 权重结构，并通过异构计算实现性能隐藏（latency hiding）。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Calibration 数据集**：C4 数据集的训练子集，用于校准量化参数（128 条序列，长度 2048）。
- **评估数据集**：
  - **WikiText-2**：标准语言建模测试集。
  - **C4 validation set**：256 个固定种子的 2048-token 窗口。
  - **FineWeb**：200 个独立文档，用于计算 KL 散度。
  - **下游任务**：WinoGrande (5-shot), HellaSwag (10-shot)，以及 ARC-easy, ARC-challenge, PIQA, OBQA。

### 实验设置
- **模型范围**：从 4B 到 72B 参数的多种 LLM，包括 Qwen3-4B/14B/32B, Llama-3-8B, Qwen2.5-72B。
- **量化方案**：
  - 使用 **NVFP4** 量化残差 $R$，采用 block-scaled（每 16 通道共享 scale）E2M1 格式。
  - 预处理使用 **SmoothQuant** 进行 activation-aware scaling ($\alpha = 0.5$)。
- **BiLoCo 参数**：
  - 主要比较 **BiLoCo r=512** 与 **SVDQuant k=32**，二者具有相同的向量存储预算（bit budget b=512）。
  - 另外也测试了低预算版本 BiLoCo r=128。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Perplexity (PPL)** | 语言模型预测能力的标准度量，越低越好。 |
| **KL Divergence** | 量化模型输出分布与 BF16 原始模型之间的差异，衡量保真度。报告 mean 和 99th percentile。 |
| **Downstream Accuracy** | 多项选择题任务上的准确率，反映实际应用性能。 |
| **Decode Latency** | 在 B300 GPU 上测量 fused kernel 的推理延迟，单位为 $\mu s$，并与 FP4-only GEMM 对比。 |

### 基线方法对比
- **NVFP4 only**：无任何修正的纯 FP4 量化。
- **SVDQuant**：使用 BF16 rank-32 低秩修正，作为主要对比基线。
- 所有方法均在同一 calibration set 和量化流程下进行公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ Perplexity 对比（Table 2）
| Model | Dataset | SVDQuant k=32 | BiLoCo r=512 | 结果 |
|-------|--------|---------------|-------------|------|
| Qwen3-4B | Wiki2 | 15.07 | **14.94** | ✔️ 更优 |
| Qwen3-4B | C4 | 15.85 | **15.81** | ✔️ 更优 |
| Llama-3-8B | Wiki2 | 6.92 | **6.84** | ✔️ 更优 |
| Llama-3-8B | C4 | 7.59 | **7.48** | ✔️ 更优 |
| Qwen3-32B | C4 | 10.40 | **10.40** | ✅ 持平 |
| Qwen2.5-72B | C4 | 8.05 | **8.00** | ✔️ 更优 |

> **结论**：BiLoCo 在绝大多数情况下 **匹配或超越 SVDQuant** 的 perplexity 表现。

#### ✅ KL Divergence 对比（Table 3）
| Model | KL mean ↓ | KL p99 ↓ |
|-------|-----------|---------|
| Qwen3-4B | **0.0740** vs 0.0835 | **0.1421** vs 0.1649 |
| Llama-3-8B | **0.0756** vs 0.0824 | **0.3252** vs 0.3409 |
| Qwen3-32B | **0.0589** vs 0.0605 | **0.3175** vs 0.3363 |

> **结论**：BiLoCo 输出分布更接近原始 BF16 模型，尤其在尾部（p99）表现更好。

#### ✅ 下游任务准确率（Table 4）
| Model | SVDQuant | BiLoCo r=512 |
|-------|----------|--------------|
| Qwen3-4B | 0.669 | **0.670** |
| Qwen3-14B | 0.759 | **0.763** |
| Llama-3-8B | 0.780 | 0.777 |
| Qwen2.5-72B | 0.849 | **0.856** |

> **结论**：总体持平或略优，仅在 Llama-3-8B 上轻微落后。

#### ⏱️ 推理延迟（Table 5）
- 在 batch size 1 时，BiLoCo fused kernel 相比 FP4-only GEMM 仅增加 **6–11% 延迟**（0.34–0.64 μs）。
- 在 batch size ≤ 8 且 rank ≤ 128 时，开销控制在 **20% 以内**。
- 实现了高效的并发执行：FP4 GEMM 在 tensor cores，binary correction 在 CUDA cores。

#### 📊 存储-精度权衡分析（Figure 2 & Table 1）
- 在相同存储预算下（b=512），BiLoCo (rank 512) 捕获的 Frobenius 范数比例（γ）是 SVDQuant (rank 32) 的 **1.76–1.92 倍**。
- 例如在 Qwen3-4B 上：SVDQuant γ=29.3%，BiLoCo γ=52.9%。

### 消融实验结果（Appendix A.2）
- **Random signs + refit**：perplexity 几乎无改善 → 表明 sign learning 至关重要。
- **Greedy peeling + global refit**（默认算法）已足够有效，进一步交替优化（pm-altmin）提升有限。
- **Construction cost**：构建 BiLoCo r=512 比 SVDQuant k=32 快约 **2 倍**（86s vs 170s on Llama-3-8B）。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 权重谱较平坦**：前几十个奇异值衰减缓慢（$\sigma_1/\sigma_{32} \approx 2$），意味着多个方向都重要，因此“多而粗”的 binary rank 更有效。
2. **存储预算应优先用于 rank 数量而非精度**：在相同 bit budget 下，512 个 binary rank 比 32 个 BF16 rank 能捕获更多能量。
3. **异构计算可实现零成本修正**：利用解码阶段 tensor cores 等待 HBM 的空窗期，用 CUDA cores 并行计算 binary correction，几乎不增加延迟。
4. **BiLoCo 实现精度-效率双赢**：在相同存储下，**匹配甚至超越 SVDQuant 的模型质量**，同时具备更好的硬件适配性和更低的构造成本。

### 方法的局限性
- **批大小受限**：当 batch size 较大（>8）或 rank 较高（>128）时，CUDA core 分支无法完全隐藏，延迟上升。
- **不适用于宽矩阵投影**：对于输出维度极大（N/64 > SM 数量）的层（如 MLP up-projection），tile 数超过 SM 数量，无法调度额外 CTA 来运行 correction。
- **当前仅针对线性层**：未涉及 Attention 或 KV Cache 的优化。
- **旋转预处理兼容性一般**：实验显示 QuaRot / SpinQuant 在 NVFP4 下表现不如 SmoothQuant，可能因旋转扩散了 block 内 outlier。

### 未来工作方向
- 扩展到其他低比特格式（如 INT4）或其他架构（非 Blackwell GPU）。
- 探索更高效的 binary decomposition 算法（如 structured sparsity + binary）。
- 将 BiLoCo 与训练感知量化（training-aware quantization）结合。
- 设计动态 rank 调度机制，根据输入动态启用/禁用 correction 分支。
- 应用于其他模态模型（如 Vision Transformer, Diffusion Models）。

---

> ✅ **总结一句话**：  
> **BiLoCo 通过引入二值化低秩修正并利用异构计算，实现了在极小延迟代价下恢复 FP4 量化损失的精度，是 PTQ 领域一次高效且实用的创新。**

</details>

---

### 7. [HobbyLM: A Sparse Mixture-of-Experts Model Family on a Hobby Budget](https://arxiv.org/abs/2610.34582)

**Authors**: Harish Santhanalakshmi Ganesan, Hariharan M. S., Prabhurajhan R  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.34582v1  

#### Abstract
We describe HobbyLM, a family of small sparse Mixture-of-Experts (MoE) models trained from scratch with a few hundred to a few thousand dollars of rented GPU time. The core is a decoder-only transformer with fine-grained experts, sigmoid routing with auxiliary-loss-free load balancing, grouped-query...

---

### 8. [Collective Emergent Alignment: Measuring and Amplifying Interaction-Induced Reasoning Alignment in Multi-Agent LLMs](https://arxiv.org/abs/2610.34836)

**Authors**: Boxuan Wang, Zhuoyun Li, Sarvapali D. Ramchurn, Xiaowei Huang, Yi Dong  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.34836v1  

#### Abstract
Multi-agent large language model (LLM) systems can exhibit emergent behaviors through interaction, yet whether a similar phenomenon exists for reasoning alignment remains largely unexplored. We study whether agent interaction produces additional reference-grounded reasoning alignment beyond matched ...

---

### 9. [Speculative Diffusion is a Tale of Two Tradeoffs](https://arxiv.org/abs/2610.31819)

**Authors**: Aaron Zorel, Guy Van den Broeck, Aditya Grover, Daniel Israel  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.31819v1  

#### Abstract
Speculative decoding is a technique that accelerates the inference speed of autoregressive large language models (LLMs) by sampling from a faster, lower quality draft model. Traditionally, a fundamental characteristic of speculative decoding is that it is lossless with respect to the verifier distri...

---

### 10. [Recommender regret for improving regularised RL estimation with bandits and MCTS](https://arxiv.org/abs/2610.32910)

**Authors**: Danilo Vucetic, Alexandre Larouche, Mohammad Pedramfar, Audrey Durand, Gauthier Gidel  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.32910v1  

#### Abstract
Regularised reinforcement learning (RL) algorithms rely on estimators, like Monte Carlo tree search or one-step look-ahead, to provide policy, value, or Q-value estimates in training and inference. However, existing bandit-based estimation frameworks evaluate sampling efficiency, not the quality of ...

---

### 11. [RAQ-MLA: Role-Aware Quantization for Multi-Head Latent Attention Caches](https://arxiv.org/abs/2610.33625)

**Authors**: Tianqi Chen, Xunchao Li, Linchuan Xie, Guanghua Yu, Jianchen Zhu, Peisong Wang, Jian Cheng  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.33625v1  

#### Abstract
Multi-Head Latent Attention (MLA) has been adopted by recent frontier large language models. Instead of per-head keys and values, MLA caches a compressed KV latent that is up-projected into keys and values at decode time, together with a narrow positional key that carries position information into t...

---

### 12. [Physics-Informed Graph-Based Safe RL with Constraint-Specific Critics for Community Battery Coordination in Distribution Network](https://arxiv.org/abs/2610.33654)

**Authors**: Chenghao Huang, Yue Yang, Hao Wang  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.33654v1  

#### Abstract
High penetrations of distributed energy resources (DERs) and fast-varying electric vehicle (EV) charging intensify voltage and line-limit violations in distribution networks (DNs), making safe, low-cost operation increasingly challenging. This paper proposes a physics-informed safe deep reinforcemen...

---

### 13. [The Hitchhiker's Guide to System 1 Models: Fast Decision-Making Beyond LLMS](https://arxiv.org/abs/2610.33775)

**Authors**: Haggai Roitman  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.33775v1  

#### Abstract
Large Language Models excel at reasoning and generation but their autoregressive, token-by-token inference makes them unnecessarily slow, expensive, and poorly calibrated for the many routine decisions (routing, gating, scoring, classification) that modern AI systems require at sub-second latency. A...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《The Hitchhiker's Guide to System 1 Models: Fast Decision-Making Beyond LLMs》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

本文系统性地解决了 **大型语言模型（LLMs）在实时软件控制流决策中的根本性缺陷**。尽管 LLMs 在复杂推理任务上表现出色，但其 **autoregressive（自回归）生成范式** 导致在需要 **亚秒级响应、结构化输出和良好校准（calibrated）概率** 的决策任务中存在严重瓶颈，具体表现为：

- **高延迟**：生成一个决策可能需要 500ms–5s。
- **高成本**：相比专用模型，每项决策的成本高出 40–400 倍。
- **校准性差**：LLM 输出的置信度（如“我有 85% 的把握”）与实际准确性相关性极低，甚至接近随机。
- **输出结构不匹配**：许多决策任务只需要一个布尔值、一个分类或一个分数，而 LLM 却生成冗长的自然语言。

### 提出了什么新方法或新思路

本文提出了 **System 1 决策模型（System 1 Decision Models）** 的统一框架和实践指南，其核心创新点在于：

1.  **引入认知科学的双系统理论（Dual-Process Theory）**：
    - 将 LLMs 类比为 **System 2**（慢、深思熟虑、序列化）。
    - 将新提出的模型定义为 **System 1**（快、直觉、并行化），旨在模拟人类快速、自动化的决策过程。

2.  **提出五大家族的统一分类法（Taxonomy）**：
    - **Family I: Encoder-Based Classifiers**：基于 BERT 等双向编码器的分类器。
    - **Family II: Reward Models**：用于 LLM 训练的偏好打分模型。
    - **Family III: Non-Autoregressive Decision Engines**：**核心创新家族**，如 Jev，通过 **RLCD（Reinforcement Learning for Calibrated Decisions）** 训练，直接输出结构化决策（Choice, Score, Noul）和**原生校准**的概率。
    - **Family IV: Contrastive Decision Models**：如 CLM-8B，通过对比学习将决策转化为检索问题。
    - **Family V: Extraction Models**：新兴家族，专注于从文本中提取结构化值（如日期、金额）。

3.  **推广 RLCD 训练范式**：
    - 提出使用 **proper scoring rules**（如 Brier Score, Log Loss）作为强化学习的奖励信号，从根本上激励模型产生**诚实的、校准过的概率**，而不是仅仅模仿人类偏好。

### 相比现有方法的优势

| 维度 | LLMs (System 2) | System 1 决策模型 |
| :--- | :--- | :--- |
| **延迟** | 500ms – 5s | **10 – 100ms** |
| **成本** | 高 (40–400×) | **极低** |
| **输出结构** | 自由文本 | **结构化决策** (JSON, scalar) |
| **校准性** | **极差** (miscalibrated) | **优秀** (calibrated by construction) |
| **适用任务** | 复杂推理、创意生成 | **结构化决策** (路由、安全审查、重排序) |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

本文并非单一实验研究，而是对一个新兴领域的**综合评述和基准测试**。其结论基于对多个公开数据集和基准的分析，包括：

- **GLUE / SuperGLUE**：通用语言理解评估基准。
- **MS MARCO**：大规模段落检索和重排序基准。
- **RewardBench**：用于评估 Reward Models 的基准。
- **DeepSWE**：软件工程任务基准。
- **Terminal-Bench 2.1**：终端环境中的代理验证基准。
- **LLM-WikiRace**：维基百科超链接导航基准。
- **HakemBench**：土耳其语决策任务基准。
- **JevAdvBench**：针对 RLCD 模型的对抗性基准。

### 实验设置和评估指标

作者提出了一个全面的 **五维评估框架**，超越了传统的准确率（accuracy）：

1.  **决策准确性 (Decision Accuracy)**：F1, AUC-ROC, Cohen's κ。
2.  **校准质量 (Calibration Quality)**：**ECE (Expected Calibration Error)**, **Brier Score**, Log Loss, Reliability Diagrams。
3.  **延迟 (Latency)**：P50, P95, P99 延迟。
4.  **成本效率 (Cost Efficiency)**：每百万次决策的成本（$/million decisions）。
5.  **鲁棒性 (Robustness)**：对分布偏移、对抗性攻击、位置偏见的敏感性。

### 基线方法对比

文中将 System 1 模型与多种基线进行了广泛对比：

- **LLMs**：如 GPT-4, Claude, Llama 作为通用推理基线。
- **传统分类器**：如 SVM, Logistic Regression。
- **早期神经网络**：如 LSTM。
- **不同家族的 System 1 模型**：例如，将 Jev (Family III) 与 CLM-8B (Family IV) 进行对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

1.  **效率优势**：
    - **速度**：System 1 模型（如 Jev）的推理延迟为 **70–500ms**，而同等任务的 LLM 需要 **500ms–5s**，实现了 **10–100 倍的速度提升**。
    - **成本**：System 1 模型的成本约为 LLM 的 **1/40 到 1/400**。

2.  **校准性优势**：
    - **ECE (期望校准误差)**：
        - **Jev (RLCD)**：ECE < **0.05**。
        - **LLM (Verbalized Confidence)**：ECE > **0.15**。
    - **可靠性图（Reliability Diagram）**：Jev 的曲线几乎与对角线（完美校准）重合，而 LLM 则表现出严重的**过度自信**（overconfident）。

3.  **级联（Cascade）性能**：
    - 在 **confidence-based cascade** 架构中，80% 的简单决策由 System 1 模型处理，只有 20% 的不确定案例才升级到昂贵的 LLM。
    - 这使得总成本降低了 **80%**，同时保持了接近 LLM 的最终准确率。

### 与基线方法的对比结果

- **在决策任务上**：Jev 等 System 1 模型在 **agent routing, search reranking, content moderation** 等任务上，以极低的延迟和成本，达到了与 LLM 相当甚至更好的性能。
- **在鲁棒性上**：**Family IV (Contrastive Models)** 在 **Prism-1** 基准上表现出色，对选项顺序（position bias）和竞争选项（rival independence）的变化具有极强的不变性（near-zero Δ），而 Family III 模型则表现出一定的敏感性。

### 消融实验结果（如有）

文中虽未进行单一模型的消融实验，但通过不同家族的对比揭示了关键设计选择的影响：

- **RLCD vs. RLHF**：使用 proper scoring rules (RLCD) 训练的模型，其校准性远优于使用人类偏好对齐（RLHF）的模型。
- **Post-hoc Calibration**：即使是简单的 **temperature scaling**，也能将 fine-tuned transformer 的 ECE 从 0.10–0.15 降低到 0.02–0.04，且不会损失准确率。

---

## 4. 关键结论和发现

### 论文的主要发现

1.  **System 1 模型是必要的**：在需要快速、低成本、高可靠性的决策场景中，LLMs 是“杀鸡用牛刀”。**System 1 模型填补了这一关键空白**。
2.  **校准性是核心**：对于自动化决策系统，**校准性比单纯的准确率更重要**。一个准确但不校准的模型会导致级联系统失效。
3.  **架构决定命运**：**Family III (Non-Autoregressive)** 模型通过 **RLCD** 训练，能够实现“**by construction**”的校准，这是其相对于其他家族的最大优势。
4.  **未来是混合的**：未来的 AI 系统将是 **System 1 + System 2** 的混合体。System 1 负责高速路由和过滤，System 2 负责深度推理和复杂任务。
5.  **开源生态爆发**：继 Jev 发布后，社区迅速涌现出 **OpenJev-RLCD, Tachyone, DeepOpen** 等高质量开源项目，推动了该领域的快速发展。

### 方法的局限性

1.  **表达力-速度权衡**：System 1 模型牺牲了表达力来换取速度。它们无法进行复杂的多步推理或生成解释。
2.  **分布偏移下的校准性退化**：当部署数据的分布与训练数据发生偏移时，模型的校准性会迅速下降（ECE 从 0.01 恶化到 0.10）。
3.  **IIA 违反**：大多数 System 1 模型（尤其是 softmax-based）违反了 **独立无关替代原则（IIA）**，即添加或删除一个无关选项会影响其他选项的相对排名。
4.  **对抗性脆弱性**：通过简单的“框架攻击”（framing attack），如在输入中添加一句无关的意见，即可逆转 12% 的决策或将 38% 的高置信度决策降为低置信度。

### 未来工作方向

1.  **标准化接口**：建立类似 OpenAI Chat Completions API 的 **标准化决策接口**，实现不同厂商模型的互换。
2.  **持续学习与遗忘**：开发能在不破坏旧任务校准性的前提下，持续学习新任务的 **continual learning** 方法。
3.  **跨模态覆盖**：将 System 1 范式扩展到 **vision, audio, GUI** 等更多模态。
4.  **组合性与决策链**：研究如何高效地组合多个 System 1 决策，形成更复杂的决策链（decision chaining）。
5.  **公平性与隐私**：深入研究 System 1 模型在 **privacy** 和 **fairness** 方面的风险与对策。

</details>

---

### 14. [DualPace: Decoupled Scheduling and Cooperative GPU--CPU Decoding for LLM Serving](https://arxiv.org/abs/2610.33501)

**Authors**: YuXuan Lin, KeQiang Wu, Kingsum Chow  
**Category**: cs.DC  
**Published**: 2026-10-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.33501v1  

#### Abstract
Large language model (LLM) serving relies on batching to improve inference throughput, but the key-value (KV) cache grows with sequence length and limits GPU-resident concurrency. CPU offloading can expand KV capacity using host memory, yet existing systems remain largely GPU-gated: CPU-resident req...

---

### 15. [Efficient Omni-modal Large Language Model Inference via Per-Query Modality Selection](https://arxiv.org/abs/2610.31768)

**Authors**: Shuqi Liu, Yuzhou Cao, Luke Ong  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.31768v1  

#### Abstract
Omni-modal large language models (Omni-LLMs) can jointly process a broad range of modalities, such as text, audio, and video, enabling stronger multimodal understanding and reasoning capabilities, but often at substantial inference cost. We observe that the benefit of additional modalities varies ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient Omni-modal Large Language Model Inference via Per-Query Modality Selection

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Omni-modal Large Language Models (Omni-LLMs) 能够联合处理文本、音频、视频等多种模态，显著提升了多模态理解和推理能力。然而，这种能力的代价是高昂的推理开销（如输入 token 数量、延迟、计算成本）。传统方法通常对所有查询统一使用全部可用模态（如 always-TAV），导致大量不必要的计算浪费。

本文提出：**并非所有查询都需要所有模态才能正确回答**。因此，固定模态配置是低效的。

### 提出的新方法与新思路
作者提出了 **Per-Query Modality Selection（按查询模态选择）** 框架，其核心思想是：

- **保持底层 Omni-LLM 不变**，不修改模型结构或进行 token 压缩。
- **在推理前动态决策**：为每个查询独立选择是否提供视频模态（即选择 Text+Audio 或 Text+Audio+Video）。
- 将该问题建模为 **accuracy–cost 优化问题**，目标是在满足平均成本预算的前提下最大化预测准确率。

为此，作者设计了一种 **Cost-Sensitive Learning（成本敏感学习）** 的模态选择器（Modality Selector）：
- 使用查询的文本嵌入（如 BGE-M3）作为决策表示 $ \boldsymbol{h}(\boldsymbol{q}) $。
- 通过拉格朗日松弛推导出最优选择规则，并训练一个分类器来预测最优模态子集。
- 引入基于效用差距的样本加权机制，使模型更关注那些选错代价高的查询。

### 相比现有方法的优势
| 方法类别 | 局限性 | 本文优势 |
|--------|-------|--------|
| Token Pruning/Compression | 仍需处理冗余模态，无法避免其计算开销 | **完全省略**不必要的模态，彻底消除其处理成本 |
| 固定模态配置（如 always-TAV） | 成本高，效率低 | 自适应选择，实现更优的 accuracy–cost trade-off |
| Cascading Methods | 需运行两次模型（TA + TAV），累积成本更高 | 仅运行一次 Omni-LLM，决策前置，效率更高 |

---

## 2. 核心实验方法和设置

### 数据集
- **WorldSense**：一个面向真实世界场景的 omni-modal 问答基准，包含文本、音频、视频三模态输入，涵盖 26 种任务类型，分布在 Recognition、Reasoning 和 Understanding 三大领域。

### 模型
- **Qwen2.5-Omni-7B**
- **Qwen3-Omni-30B-A3B**
- 两者均作为冻结的 backbone 模型使用，仅在其前添加一个轻量级的 modality selector。

### 评估指标
实验在三种推理成本维度下评估 performance：

| 指标 | 定义 |
|------|------|
| **Input Tokens** | 输入 token 总数（文本 + 音频 + 视频） |
| **Latency** | 单次推理延迟（秒） |
| **Monetary Cost** | 基于模拟 API 定价的美元成本（text: \$1.40e-6/token, audio: \$11.00e-6/token, video: \$1.40e-6/token） |

#### 主要评估方式：
- **Accuracy–Cost Trade-off Curve**：通过调整 trade-off 参数 $\lambda$ 生成一系列操作点。
- **Normalized AUC**：衡量 accuracy–cost 曲线下面积，越大越好。
- **Cost Reduction at Always-TAV Accuracy**：在达到 full-modality（always-TAV）准确率的前提下，节省了多少成本（$\Delta\%$）。

### 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **Always-TA / Always-TAV** | 固定配置 | 始终使用 TA 或 TAV |
| **Random Router** | 非自适应 | 以概率 $p$ 随机选择 TAV |
| **LLM-as-a-Judge (Gemini 2.5 Pro/Flash-Lite)** | 外部判断 | 使用外部 LLM 判断是否需要视频 |
| **Estimate Then Predict** | 学习式路由 | 先估计各配置的 accuracy 和 cost，再决策 |
| **VerbConf Cascade** | 级联 | 先跑 TA，让模型口头报告 confidence，决定是否升级 |
| **LogitConf Cascade** | 级联 | 使用模型输出 logits 计算 confidence 进行升级决策 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

#### 在 **Qwen2.5-Omni** 上，达到 always-TAV 准确率（48.00%）时的成本节省：
| 方法 | Input Tokens | Latency | Money |
|------|-------------|---------|-------|
| **Cost-Sensitive Learning** | **↓58.89%** | **↓61.07%** | **↓38.94%** |
| Estimate Then Predict | ↓31.55% | ↓32.65% | ↑20.87% |
| LogitConf Cascade | ↓28.65% | ↓32.51% | ↓3.74% |

> ✅ 成本敏感方法在三项指标上全面领先，尤其在 token 和 latency 上节省超过 58%。

#### 在 **Qwen3-Omni** 上，达到 always-TAV 准确率（51.79%）时的成本节省：
| 方法 | Input Tokens | Latency | Money |
|------|-------------|---------|-------|
| **Cost-Sensitive Learning** | **↓34.72%** | **↓34.56%** | **↓24.85%** |
| LogitConf Cascade | ↓31.29% | ↓30.98% | ↑3.99% |
| Estimate Then Predict | ↓12.44% | ↓12.17% | ↑8.90% |

> ✅ 依然取得最优结果，且避免了级联方法可能导致的 monetary cost 上升。

### AUC 对比（Table 1）
| 方法 | Qwen2.5-Omni (Tokens) | Qwen3-Omni (Tokens) |
|------|------------------------|----------------------|
| **Cost-Sensitive Learning** | **0.9265** | **0.8670** |
| Estimate Then Predict | 0.8178 | 0.6941 |
| LogitConf Cascade | 0.6963 | 0.7143 |
| Random Router | 0.5046 | 0.4986 |

> ✅ 成本敏感方法在所有设置下均取得最高的 AUC，表明其整体 trade-off 更优。

### 消融与潜力分析
- **Oracle Selector** 实验表明，理论上最多可在 Qwen2.5-Omni 上节省 **78.1% 的 token 成本**，同时提升准确率 11.03 个百分点，说明 per-query selection 存在巨大优化空间。
- **Task-Type Router** 实验显示，即使仅基于任务类型进行简单排序选择，也能获得优于随机路由的效果，验证了 query-wise 差异的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. **模态需求具有显著的 query-wise 差异性**：某些查询仅靠文本+音频即可正确回答，加入视频反而可能误导模型；而另一些查询（如涉及颜色、空间关系）必须依赖视频。
2. **Per-Query Modality Selection 是高效推理的有效途径**：通过智能选择模态，可以在不牺牲甚至提升准确率的情况下，大幅降低推理成本。
3. **Cost-Sensitive Learning 显著优于现有方法**：相比 cascading 和外部 judge 方法，所提方法实现了更优的 accuracy–cost trade-off，且不会引入额外的运行开销。
4. **决策前置优于级联**：级联方法因需运行两次模型，其累计成本往往高于直接选择，且 confidence 信号可能不可靠（如 VerbConf 表现不佳）。

### 方法的局限性
- 当前研究仅考虑二元选择（是否加视频），未扩展到更复杂的模态组合（如单独使用视频或音频）。
- 决策表示目前仅依赖文本或任务元数据，未利用轻量级多模态信号（如音频摘要、视频关键帧 embedding）。
- 所有实验基于冻结的 Omni-LLM，未探索与模型微调或架构优化的联合设计。

### 未来工作方向
- 扩展到 **多模态子集选择**（multi-subset selection），支持更灵活的模态组合。
- 探索 **更丰富的决策表示**，如融合文本、音频、视频的轻量级 embedding。
- 与 **LLM routing/cascading** 结合，实现“选择哪个模型 + 提供哪些模态”的联合优化。
- 探索 **在线学习** 框架，使 selector 能够根据反馈持续改进。

--- 

> **总结**：本文首次系统性地研究了 Omni-LLM 的 per-query modality selection 问题，提出了一种简单而高效的 cost-sensitive learning 方法，在 WorldSense 上实现了显著的成本节约（最高达 61% 延迟降低），为大规模多模态模型的高效部署提供了新思路。

</details>

---

### 16. [GS-MoE: Training-Free MoE Compression via Groupwise Subspace Sharing and Pruning](https://arxiv.org/abs/2610.32009)

**Authors**: Carson Cary, Yukai Jin, Yihao Huang, Zhankai Ye, Fangyi Li, Ting Hua, Xin Liu, Shangqian Gao  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.32009v1  

#### Abstract
Mixture-of-Experts (MoE) models reduce computation through sparse activation, but storing their expert parameters remains a major memory bottleneck. We introduce GS-MoE, a training-free compression framework that addresses redundancy across and within experts through two compression axes: shared-sub...

---

### 17. [PRISM: Promoting Sampling Diversity of Large Language Models with Implicit Stochastic Modulation](https://arxiv.org/abs/2610.32101)

**Authors**: Qixin Xu, Zihan Zhu, Xiaojiang Liu, Bohan Lyu  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.32101v1  

#### Abstract
Large language models (LLMs) exhibit mode collapse, where their output distribution concentrates on familiar semantic modes. When diversity disappears, repeated sampling becomes semantic repetition rather than discovery, making rare but stronger solutions unlikely to surface and leaving self-evoluti...

---

### 18. [ReMedI: Inference-Time Refinement and Self-Distillation for Controllable Molecular Optimization](https://arxiv.org/abs/2610.32103)

**Authors**: Jia Yi Zhang, Shivanshu Shekhar, Allison McLevige, Tong Zhang  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.32103v1  

#### Abstract
In drug development, lead optimization must improve competing properties, such as increasing permeability and reducing toxicity, while limiting scaffold changes that could compromise the lead's potency. RL-based post-training, such as C-MORAL, addresses this problem by fine-tuning molecular language...

---

### 19. [Graph-dependent pooling: adaptive granularity in hierarchical graph neural networks](https://arxiv.org/abs/2610.33623)

**Authors**: Bohan Yoon, Hyejin Jang  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.33623v1  

#### Abstract
Graph Neural Networks are widely used to learn structural representations of relational data. Hierarchical graph pooling learns multi-level representations by progressively coarsening graphs; however, prior methods rely on predefined cluster numbers or fixed pooling ratios, limiting adaptability to ...

---

### 20. [Meta-CLIP: Metadata-Conditioned Contrastive Learning for Zero-Shot Disease Inference](https://arxiv.org/abs/2610.33674)

**Authors**: Haozhe Luo, Nirosh Sivanesan, Shelley Zixin Shu, Francesco Nitti, Mauricio Reyes  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.33674v1  

#### Abstract
Medical vision--language pretraining learns transferable representations from paired images and radiology reports, yet differences in scanners and acquisition protocols can impair transfer. Existing approaches primarily enrich clinical semantics, leaving structured acquisition information in DICOM h...

---

### 21. [Prefetch-Steered Routing: Deterministic Cache-Aware Expert Selection for Offloaded Mixture-of-Experts Decoding](https://arxiv.org/abs/2610.34721)

**Authors**: Siming Huang  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.34721v1  

#### Abstract
When a Mixture-of-Experts (MoE) model does not fit in GPU memory, batch-1 decoding is dominated by copying expert weights from host memory. Cache-aware routing reduces these copies by biasing the router toward resident experts, but combined with asynchronous prefetching it makes the model's outputs ...

---

### 22. [ClinVerse: A Realistic Multi-Agent Diagnostic Environment for Training and Evaluating Medical Agents](https://arxiv.org/abs/2610.34878)

**Authors**: Sirry Chen, Fangke Chen, Xuanwen Ding, Wei Chen, Zhongyu Wei  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.34878v1  

#### Abstract
Disease diagnosis is a key application of large language model (LLM) agents in healthcare. However, existing medical agents are primarily trained and evaluated with static question-answering data, providing limited opportunities to learn active information acquisition, examination decision-making, a...

---

### 23. [Toward Fast Human Reaction Modeling from High-Frequency Digital Interaction Data](https://arxiv.org/abs/2610.35079)

**Authors**: Emanuel Dopater, Daniel Gavlas, Luk\'a\v{s} Jochymek, Kannan Paul, Miloslav Szczypka, David Voj\'a\v{c}ek, Josef \v{Z}\'a\v{c}ek, Jan Plucar, Radek Svoboda, Ivan Zelinka  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.35079v1  

#### Abstract
We investigate whether limbic-like signatures of rapid, salience-driven behavior can be identified in high-resolution Counter-Strike 2 telemetry. This exploratory study introduces Ego-Knowledge, a leak-reduced reconstruction of each player's plausible perceptual state, supported by a millisecond-pre...

---

### 24. [Belief-Conditioned Multi-Agent Actor Critic for Decentralized LLM Collaboration](https://arxiv.org/abs/2610.35398)

**Authors**: Tianle Chen, Shuo Liu, Christopher Amato  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.35398v1  

#### Abstract
Recent work has explored optimizing LLM collaboration through Multi-Agent Reinforcement Learning (MARL), but many approaches still rely on predefined workflows, limiting efficiency and scalability. Decentralized LLM collaboration is challenging because agents execute independently with limited commu...

---

### 25. [EvoRouter: Cost-Efficient Multi-Agent Routing via Insight-Guided On-Policy Distillation](https://arxiv.org/abs/2610.35690)

**Authors**: Yitao Xiao, Xu Zhang, Ye Cheng, Shaoyong Guo, Yan Sun, Ziyue Zhu, Sitao Xie, Guo Tong, Haoran Luo  
**Category**: cs.AI  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.35690v1  

#### Abstract
Serving web queries at scale with multi-agent systems requires jointly configuring large language models, collaboration modes, agent counts, interaction rounds, and skills. However, the resulting configuration space grows combinatorially, while sparse execution rewards provide limited fine-grained s...

---

### 26. [The LALA Benchmark: Evaluating Literary and Linguistic Analysis in LLMs](https://arxiv.org/abs/2610.34080)

**Authors**: Michael Foster, Barbara McGillivray, Gard B. Jenset, Jakub M\'acha, Jason Harding, Marcus Perlman, Devyani Sharma, Patrick Skeels, Jana Willer-Gold, Jeannette Littlemore  
**Category**: cs.CL  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.34080v1  

#### Abstract
We introduce the Literature and Linguistics Analysis Benchmark (LALA), comprising 236 open-ended questions written by experts in linguistics and literary studies, together with expert-provided ideal answers. By comparing models within families and at similar training scales using human and LLM judge...

---

### 27. [Speeding up Log-Sum-Exp: Kernel Fusion at the Memory Wall, Integer Arithmetic at the Compute Wall](https://arxiv.org/abs/2610.33647)

**Authors**: Lingyun Yao, Martin Andraud, Niki Loppi, Andrea Pilzer, Anji Liu, Guy Van den Broeck, Martin Trapp  
**Category**: cs.DC  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.33647v1  

#### Abstract
While the Log-Sum-Exp (LSE) function is a foundational numerical primitive for stable log-domain computation in modern machine learning, its fast and efficient computation on generic platforms (\ie, CPUs and GPUs) has been underexplored, potentially creating execution bottlenecks for large workloads...

---

### 28. [Priority-Aware Scheduler for Serverless Functions Provision and Load Balancing](https://arxiv.org/abs/2610.35073)

**Authors**: Cheryl Lee, Zishan Su, Zhouruixing Zhu, Tianyi Yang, Yuxin Su, Pinjia He, Michael R. Lyu  
**Category**: cs.DC  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.35073v1  

#### Abstract
Serverless computing has emerged as a pivotal cloud deployment paradigm due to its efficient utilization of on-demand resources. However, the persistent cold start problem continues to hinder system performance. Current strategies often employ oversimplified preloading/unloading mechanisms that inad...

---

### 29. [Decode Late, Multiply Once: Rethinking the Dot Product for Quantized Weights](https://arxiv.org/abs/2610.31913)

**Authors**: Jonathan Tapson  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.31913v1  

#### Abstract
Large neural networks are increasingly deployed with 1-4 bit weights and 16- or 32-bit activations, yet their dot products still use one multiplier per element. We show that a length-n dot product with k-bit weight codes can be computed with n+2^k additions and at most 2^k decodes and 2^k multiplica...

---

### 30. [Beyond Data Schedule: Fast LLM Pretraining on Data Mixtures by Loss-Reweighting](https://arxiv.org/abs/2610.32141)

**Authors**: Jiazheng Li, Huaqing Zhang, Jing Xu, Tianle Cai, Jingzhao Zhang  
**Category**: cs.LG  
**Published**: 2026-10-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.32141v1  

#### Abstract
Large language models often acquire specialized capabilities through successive pretraining stages, but these gains can come at the cost of forgetting earlier knowledge. In this work, we investigate whether such trade-offs can be improved by changing how strongly each domain contributes to learning ...

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

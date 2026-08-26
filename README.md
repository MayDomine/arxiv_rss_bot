# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-26 06:10:49 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Compression Trinity: Exploring Sparsity, Quantization, and Low-Rank Approximations for LLM Compression](https://arxiv.org/abs/2608.24070)

**Authors**: Mohammad Mozaffari  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.24070v1  

#### Abstract
Prohibitive computational and environmental costs impede the scalable deployment of Large Language Models (LLMs). Traditional compression techniques (sparsity, quantization, low-rank approximations) are typically applied in isolation, and each hits an accuracy-efficiency wall. This thesis proposes t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Compression Trinity: Exploring Sparsity, Quantization, and Low-Rank Approximations for LLM Compression》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前 Large Language Models (LLMs) 面临**计算成本高、内存占用大、部署困难**等问题。传统的压缩技术（如 sparsity、quantization、low-rank approximations）通常被**孤立使用**，导致在极端压缩下出现“准确率崩溃”（accuracy collapse），即模型性能急剧下降。

该论文指出：单一压缩方法存在“效率-精度墙”（accuracy-efficiency wall），无法满足可持续、可扩展的 LLM 部署需求。

### 提出的新方法与新思路
论文提出 **“Compression Trinity”** 框架——一种将三大压缩支柱联合应用的统一范式：

- **Sparsity**：减少计算量（FLOPs）
- **Quantization**：降低内存带宽压力
- **Low-Rank Approximations**：恢复因前两者损失的模型表达能力

这三者不是独立工具，而是**互补协同**的技术组合，形成一个“多维平衡”的压缩策略。

#### 具体创新方法包括：
| 方法 | 应用阶段 | 核心思想 |
|------|--------|---------|
| **MKOR** | Pretraining 优化器 | 在二阶优化中引入 block-diagonal sparsity 和 low-rank inversion，将 curvature 更新复杂度从 $O(d^3)$ 降至 $O(d^2)$，加速收敛最多 **1.85×** |
| **SLoPE** | Pretraining 模型训练 | 双重剪枝（double-pruned backward pass）实现 N:M sparsity，并在最后 1% 训练中加入 low-rank “lazy” adapters 恢复精度，端到端训练加速达 **1.25×** |
| **OPTIMA** | Post-training 压缩 | 在无训练（zero-training）场景下，通过全局最优列级二次规划稳定静态 mask，提升 zero-shot 准确率最高 **3.97%** |
| **PATCH** | Post-training 压缩 | 引入可学习 tile-level mask 与 2:4 sparsity 联合优化，在有 fine-tuning 预算时动态调整稀疏比例（0%-50%），获得最高 **1.38× 推理加速** |
| **SLiM** | One-shot 综合压缩 | 完整实现 Compression Trinity：结合 pruning + quantization + mathematically derived low-rank adapters，一次性恢复信息损失，准确率比 SOTA 方法高出 **5.66%**，甚至在相同参数预算下**超越未压缩 dense 模型 0.6%** |

### 相比现有方法的优势
- **突破单一支柱限制**：避免单独使用 sparsity 或 quantization 导致的准确率崩塌。
- **全生命周期覆盖**：从 pretraining 到 inference 各阶段均适用，而非仅限于 post-training 压缩。
- **硬件友好 + 算法高效**：兼顾算法精度与实际运行效率（如利用 cuSPARSELt 加速稀疏矩阵乘法）。
- **无需额外数据微调即可恢复性能**：SLiM 和 BEAM 支持 one-shot 压缩且不依赖大规模 retraining。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 用途 |
|-------|-----|
| **C4** | Calibration dataset，用于 weight saliency 分析（pruning & quantization） |
| **WikiText-2**, **WikiCorpus** | Perplexity 评估预训练语言建模任务 |
| **GLUE** | BERT 下游分类任务评估（平均 F1/Accuracy） |
| **SQuAD v1.1** | Question Answering 任务 |
| **MMLU, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, OpenBookQA, RACE, HellaSwag** | Zero-shot 推理能力综合评测（共 8 项任务） |
| **SlimPajama** | 替代 calibration 数据集，验证方法对数据敏感性 |

### 实验设置与评估指标
| 设置项 | 描述 |
|------|------|
| **模型范围** | OPT (125M–6.7B), LLaMA-2 (7B–13B), LLaMA-3 (8B), Gemma (1B–2B), BERT-Large, ResNet-50 |
| **压缩目标** | 达成 8x 压缩比（例如 2-bit quantization 或 87.5% sparsity） |
| **评估指标** | 
| - Zero-shot Accuracy (%) | 多任务平均得分 |
| - Perplexity | 语言模型困惑度 |
| - End-to-end Speedup (×) | 训练/推理总耗时加速比 |
| - Memory Reduction (×) | 显存占用降低倍数 |
| - FLOPs / Memory Bandwidth | 理论计算与通信开销分析 |

### 基线方法对比
| 类别 | 对比方法 |
|------|---------|
| **Pruning** | Wanda, SparseGPT, Thanos, ProxSparse, MaskLLM |
| **Quantization** | AbsMax (RTN), OPTQ, AWQ, SmoothQuant, Group Quantization |
| **Low-Rank Adaptation** | LoRA, QLoRA, LQ-LoRA |
| **Optimizer** | Adam(W), LAMB, KFAC, KAISA, SGD, Eva |
| **Sparse Training** | FST (Fully Sparse Training, ICML 2024) |
| **Hybrid Compression** | SLiM vs. 单独使用 sparsity/quantization 的组合 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 方法 | 性能表现 |
|------|--------|
| **MKOR vs. KFAC** | 收敛速度加快 **1.85×**；在 BERT-Large 上以 600 步达到原需 1563 步的精度，端到端提速 **2.57×** |
| **SLoPE vs. FST** | 训练加速 **1.25×**，推理也有显著提速（FST 因最终 dense 微调而无推理增益）；显存减少最高 **2.3×** |
| **SLiM on LLaMA-2-7B** | 在 8x 压缩比下，相比单支柱方法：
  - 2-bit Quantization → 31.81%
  - 87.5% Sparsity → 31.24%
  - **4-bit + 50% Sparsity + Low-Rank** → **52.38%**（接近原始 FP16 的 54.61%） |
| **PATCH on LLaMA-3.1 8B** | 动态混合 sparsity 实现灵活 trade-off，推理速度提升 **1.38×** |
| **BEAM 框架** | 作为通用 refine 工具，在单卡上 <4 小时内为压缩模型恢复高达 **4.34%** 的准确率 |

### 与基线方法对比结果（Table 1.2）

| 方法 | 平均 Zero-shot Accuracy |
|------|------------------------|
| Dense Baseline (FP16) | 54.61% |
| 2-bit Quantization (Single) | 31.81% |
| 87.5% Unstructured Sparsity (Single) | 31.24% |
| 4-bit + 2:4 Sparsity (Multi) | 47.97% |
| **4-bit + 50% Sparsity (Compression Trinity)** | **52.38%** ✅ |

> 结果表明：**multi-pillar 方法显著优于任何 single-pillar 方法**，证明了 Compression Trinity 的有效性。

### 消融实验结果（Ablation Studies）
- **低秩适配器 rank 影响（E.2-a）**：当 adapter rank / hidden dimension ≈ 0.1 时，即可带来显著准确率提升，同时保持低推理开销。
- **校准样本数量影响（E.2-b）**：SLiM 对 calibration 数据量不敏感，即使少量样本也能有效评估权重重要性。
- **不同 calibration 数据集表现（Table E.16）**：在 C4 与 SlimPajama 上结果相近，说明方法具有良好的泛化性。
- **depth vs. width pruning（Figure B.6）**：宽度剪枝比深度剪枝更稳定，对 loss 影响更小。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **单一压缩技术存在根本性瓶颈**：无论是极致 sparsity 还是 aggressive quantization，都会导致模型能力坍缩。
2. ✅ **三大支柱高度互补**：
   - Sparsity 解决 **compute-bound** 问题（适用于 training/prefill）
   - Quantization 解决 **memory-bound** 问题（适用于 decoding）
   - Low-Rank Approximations 是“精度修复引擎”，弥补联合压缩带来的误差
3. ✅ **Compression Trinity 可贯穿整个 LLM 生命周期**：
   - Pretraining：MKOR（优化器级）、SLoPE（模型级）
   - Post-training：OPTIMA（静态）、PATCH（动态）、SLiM（一体化）
4. ✅ **One-shot 压缩可以媲美甚至超越 fine-tuning 效果**：SLiM 表明数学推导驱动的 low-rank recovery 可替代昂贵的 retraining。
5. ✅ **效率不是事后优化，而是设计原则**：未来的 LLM 应“天生稀疏、知识密集”。

### 方法的局限性
- **当前验证集中在 125M–70B 参数模型**，万亿级超大模型可能面临新的通信与结构瓶颈。
- **稀疏格式仍需存储 indices**，在极低位宽（如 2-bit）下元数据开销可能抵消收益。
- **KV Cache 和 Attention 机制尚未系统纳入 Trinity 框架**，长序列场景下的内存瓶颈未完全解决。
- **理论边界尚待完善**：目前缺乏对“quantized sparse-plus-low-rank decomposition”的严格误差界分析。

### 未来工作方向
1. **扩展 Trinity 至 Context Window**：
   - Quantize 动态 KV states
   - Induce sparsity in attention patterns（如 Sliding Window, Block-Sparse Attention）
   - Apply low-rank approximation to attention heads，支持 infinite-context reasoning
2. **硬件-算法协同设计（Hardware-Algorithm Co-design）**：
   - 设计无需显式索引存储的“algorithmic sparsity”
   - 开发专用稀疏张量核（Sparse Tensor Core）
3. **激活值压缩（Activation Sparsity）**：
   - 将 PATCH 思路应用于 dynamic activation tensors
   - 实现 prefill 阶段的 compute acceleration
4. **构建联合优化框架**：
   - 替代当前 sequential pipeline（如先 prune 再 quantize）
   - 探索类似 OATS、HASSLE-free 的 joint sparse-quantized-low-rank 优化
5. **开放研究生态**：
   - 推出 BEAM、LEAP 等开源工具包，推动快速迭代与社区共建

---

> 📌 **总结一句话**：  
> **“Efficiency is not a constraint — it’s a design principle.”**  
> 本论文通过 Compression Trinity 框架证明：只有将 sparsity、quantization 和 low-rank approximations 联合起来，才能真正打破效率墙，实现高性能、低成本、可持续的大模型部署。

</details>

---

### 2. [Parason: Revealing Subtask and Trial Parallelism in LLM Reasoning](https://arxiv.org/abs/2608.24658)

**Authors**: Zhengyang Zhang, Zijian Zhang, Jiaxuan Gao, Shusheng Xu, Yi Wu, Song Han, Ligeng Zhu  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.24658v1  

#### Abstract
Scaling test-time reasoning has substantially improved the problem-solving ability of large language models (LLMs), but standard autoregressive decoding still executes long reasoning traces sequentially, creating severe latency for difficult tasks (up to days and weeks). Parallel reasoning offers a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Parason: Revealing Subtask- and Trial Parallelism in LLM Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前的大型语言模型（LLM）在复杂推理任务中依赖**长链式的自回归解码（autoregressive decoding）**，导致推理延迟极高（可达数天），严重限制了其在交互式应用（如编程助手、数学求解器）中的实用性。尽管已有工作探索并行推理，但主要聚焦于 **Subtask Parallelism**（将大任务分解为可独立执行的子任务），而忽略了另一种广泛存在的并行形式——**Trial Parallelism**。

本论文指出，现有系统未能充分挖掘 LLM 推理过程中的全部并行潜力，尤其是在面对不确定性时进行多路径试探的行为未被有效建模和利用。

---

### 🚀 提出的新方法与创新思路

作者提出 **Parason**，一个算法-系统协同设计框架，首次明确区分并联合建模两种并行模式：

- **Subtask Parallelism (AND 分支)**  
  将问题分解为多个必须完成的独立子目标（如分段求和），所有分支结果均需合并以得到最终答案。

- **Trial Parallelism (OR 分支)**  
  在不确定路径下并行尝试多种假设或解法（如试错不同公式），仅需一条成功路径即可推进，其余探索仍可用于后续综合判断。

#### 创新点包括：

1. **语义级并行分类体系**  
   首次从语义层面定义并量化 Subtask 与 Trial 并行性，并通过实证分析证明后者在难题中占主导地位（最高达 76.1%）。

2. **结构化并行轨迹表示（CFG-based format）**  
   设计基于上下文无关文法（Context-Free Grammar, CFG）的标记格式，支持 `<Parallel>`、`<Subtask>`、`<Trial>`、`<Thread>` 等标签，使模型输出具有可解析的并行结构。

3. **并行感知强化学习训练（PA-GRPO）**  
   提出 **Parallelism-Aware Group Relative Policy Optimization (PA-GRPO)**，其奖励函数联合优化：
   - 正确性（accuracy）
   - 最长路径延迟（token-level latency）
   - 两种并行比例（Subtask/Trial ratio）
   - 加速收益（critical path vs total tokens）

4. **无缝集成现代推理引擎**  
   将并行结构映射为 **tool calls**，无需修改底层推理架构即可实现真实世界加速，在 SGLang 中实现了端到端可执行。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（如 Self-Consistency, Multiverse） | Parason |
|------|--------------------------------------------|--------|
| 并行类型 | 主要支持 Subtask 或隐式 Trial | 显式建模 Subtask + Trial 双重并行 |
| 结构控制 | 自由文本或简单分支 | CFG 强约束语法，避免歧义 |
| 训练目标 | 仅优化正确率 | 多目标：准确率 + 延迟 + 并行利用率 |
| 执行能力 | 报告理论加速，难以落地 | 支持真实 wall-clock 加速（via tool call） |
| 泛化性 | 固定模板或提示工程 | 学习生成动态并行结构 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **AIME24 & AIME25**：美国数学邀请赛（American Invitational Mathematics Examination）题目，用于主评测。
- **AMC**：美国数学竞赛题，测试基础数学能力。
- **Math500**：精选的 500 道数学题集合。
- **Minerva Math**：大规模数学推理基准。
- **OpenMathReasoning** 和 **Humanity's Last Exam (HLE)**：用于分析并行性分布的人类标注推理轨迹数据集。

> 注：训练数据来自 ThreadWeaver 发布的 964 条 Qwen3-8B 推理轨迹，以及 Polaris-53k 的 53,000 道复杂推理题用于 RL 微调。

---

### ⚙️ 实验设置与评估指标

#### 模型规模
- 主要使用 **8B 参数模型**作为 backbone 进行训练与比较。

#### 训练流程（两阶段）
1. **监督微调（SFT）**：将原始 CoT 转换为 Parason 格式的并行轨迹。
2. **RL 微调（VeRL + PA-GRPO）**：使用强化学习进一步优化并行结构与延迟。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy** | 最终答案是否正确 |
| **Token Latency** | 最长生成路径上的 token 数量（反映 wall-clock 时间） |
| **Acceleration Ratio** | 总生成 token / 最长路径 token |
| **Trigger Ratio** | 含至少一个 `<Parallel>` 块的样本占比 |
| **Subtask/Trial Ratio** | 对应分支中 token 占比 |

#### 基线方法对比
- **ThreadWeaver (8B)**：基于自适应线程的并行推理系统
- **Multiverse (32B)**：学习并行生成与合并策略
- **Dynamic Early Exit / DYNASOR-CoT / AdaptThink**：高效推理方法
- **Vanilla SFT / PA-GRPO variants**：消融实验对照组

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 2 & Table 4）

| 方法 | AIME24 | AIME25 | Math500 | AMC | Avg Acc (%) | Token Latency | Accel. Ratio |
|------|--------|--------|---------|-----|--------------|----------------|---------------|
| ThreadWeaver (8B) | 79.9 | 60.5 | 92.3 | 91.4 | 81.0 | 14.8k | 1.18× |
| Parason-8B (SFT only) | 73.5 | 67.9 | 93.4 | 93.9 | 82.2 | 13.2k | 1.27× |
| **Parason + PA-GRPO (β_trial=0.1)** | **78.2** | **70.6** | **94.6** | **97.5** | **84.7** | ~14.5k | **1.71×** |

> ✅ **Parason 在平均准确率上达到 84.7%，超越所有已知 8B 模型，并优于部分 32B 基线**

---

### 🔁 与基线方法对比结果

- **平均加速约 1.7×**，且保持竞争力精度。
- 在 **AIME25 和 AMC 上取得当前最优结果**。
- 在低延迟预算下优势更明显：
  - 当最长路径限制为 **2048 tokens** 时，Parason 达到 **34.7% AIME24 准确率**，远超 SFT-only 的 16.8%。
  - 在 8192 token 预算下，准确率达 60.3%，比 SFT 高出 18.5 个百分点。

---

### 🔍 消融实验结果（Ablation Study）

#### 不同奖励权重的影响（Table 2 & 4）

| 设置 | 影响 |
|------|------|
| ↑ `β_trial`（Trial 激励） | ➕ 显著提升准确率（最多 +3.7 pts）<br>➖ 略增延迟 |
| ↑ `β_subtask`（Subtask 激励） | ➕ 明显降低最长路径（压缩 critical path）<br>✅ 最高实现 **1.75× 加速比** |
| ↑ `α`（延迟惩罚） | ✅ 更好地控制 token latency，最低降至 **12.1k** |

> 💡 发现：**Trial Parallelism 提升 accuracy，Subtask Parallelism 降低 latency** —— 二者可独立调节，实现灵活权衡。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Trial Parallelism 是主流**  
   在 HLE 和 OpenMath 数据集中，**Trial Parallelism 占据了超过 58% 的并行步骤**，在 DeepSeek-V4 上高达 **65.5%**，表明“试错”是 LLM 解难题的核心机制。

2. **硬问题更依赖 Trial 并行**  
   随着问题难度上升，可并行化的 token 数显著增加，但主要体现在 Trial 分支上，说明困难问题需要更多探索路径。

3. **并行 ≠ 冗余计算**  
   Parason 成功将大量非关键路径工作移出主干，**节省 token 高达 21.3k（hard 问题）**，而非简单复制推理。

4. **真实 wall-clock 加速可达 1.6× 以上**  
   在 A800 GPU 上实测显示，easy/medium/hard 问题分别获得 **1.62×、1.38×、1.47×** 的运行时间加速。

---

### ⚠️ 局限性（Limitations）

1. **领域局限**：目前实验集中在数学推理任务，尚未验证在代码生成、规划等 agent 场景下的有效性。
2. **模型尺度受限**：当前研究基于 8B 模型，尚不清楚更大模型（如 70B+）是否会表现出不同的并行行为。
3. **人工标注依赖**：训练数据依赖 ThreadWeaver 提供的标注轨迹，扩展成本较高。
4. **CFG 表达力边界**：虽然支持嵌套，但混合依赖关系（如 subtask 内部含 trial）尚未完全建模。

---

### 🔮 未来工作方向

1. **跨领域迁移**：将 Parason 应用于 coding、agent planning、科学发现等场景。
2. **更大模型验证**：在 32B/70B 级别模型上复现并行收益。
3. **自动标注 pipeline**：减少对人工标注的依赖，构建自监督识别机制。
4. **动态调度优化**：结合硬件资源动态分配 worker，最大化吞吐效率。
5. **开放生态建设**：计划开源实现，推动社区共建并行推理标准。

---

## 总结

> **Parason 揭示了 LLM 推理中被长期忽视的 Trial Parallelism，并通过语义分类、结构化表示、多目标训练与系统集成，首次实现了从理论到真实加速的闭环。它不仅提升了推理效率（~1.7× 加速），还刷新了多项数学基准的性能记录，为下一代高效、响应式智能体提供了关键技术路径。**

</details>

---

### 3. [Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports](https://arxiv.org/abs/2608.22817)

**Authors**: Parsa Bakhtiari, Hassan Bashiri, Alireza Khalilipour, Masoud Nasiripour, Moharram Challenger  
**Category**: cs.CL  
**Published**: 2026-08-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.22817v1  

#### Abstract
Industrial technical reports contain high-value knowledge for maintenance, troubleshooting, and product engineering, but their heterogeneous structure (dense prose, specifications, tables) makes them difficult to index and reason over with standard retrieval and QA pipelines, and no public instructi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
工业技术报告（如维护手册、产品规格书）包含大量高价值的专业知识，但由于其**异构性强**（密集文本、表格、图表混合）、结构复杂且知识分散，导致传统检索和问答系统难以有效利用。此外，目前缺乏基于真实工业文档构建的公开 **instruction-tuning** 和 **benchmark 数据集**，严重制约了小规模语言模型在工业场景中的应用。

### 提出的新方法与框架
本文提出 **Industrial-Instruction**，一个端到端的框架，用于从工业技术报告中构建指令微调和基准测试数据集。其核心创新包括：

- **双版本高质量数据集发布**：基于 906 份公开的 Panasonic 技术文档（共 7,525 页），构建了两个并行的多选问答（multiple-choice QA）数据集：
  - `Industrial-Instruction-Qwen`：由开源大模型 **Qwen3-30B-A3B-Instruct** 生成。
  - `Industrial-Instruction-Claude`：由闭源 API 模型 **Claude-Opus-4.6** 生成。
  这两个数据集均包含约 13.6k QA 对，并附带来源文档和独立的 benchmark 分割。

- **五种现实查询-文档关系建模**：数据集显式地覆盖了五种真实世界的信息检索场景，使模型能被训练和评估在面对检索噪声和多步推理时的鲁棒性：
  1. `r0`: Irrelevant retrieval（无用文档）
  2. `r1`: Single-document support（单文档提供线索）
  3. `r2`: Multi-document support（多文档提供线索）
  4. `r3`: Single-document answer（单文档含完整答案）
  5. `r4`: Multi-document answer（多文档联合构成答案）

- **完整的自动化构建流水线**：结合 **layout-aware extraction**（使用 Dots.OCR）、**semantic indexing**（EmbeddingGemma + FAISS）和 **synthetic QA generation**，实现了从原始 PDF 到高质量训练/评测数据的全流程自动化。

### 相比现有方法的优势
- **填补领域空白**：首次系统性地从真实工业文档中构建可用于 instruction-tuning 和 benchmark 的数据集。
- **支持小模型（<10B parameters）**：专注于资源受限环境下的实用化部署，而非依赖超大规模模型。
- **开放可复现**：代码、数据集和生成流程全部开源，促进工业 NLP 领域的发展。
- **支持生成器对比研究**：通过同一管道使用不同 LLM 生成数据，直接比较了 open-weight 与 frontier 模型作为数据生成器的效果。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **原始语料库**：906 份 Panasonic 公开技术文档（PDF 格式，7,525 页）。
- **生成的 QA 数据集**：
  - `Industrial-Instruction-Qwen`：~13.6k QA 对（训练集 12,557 + 测试集 1,000）。
  - `Industrial-Instruction-Claude`：~25.3k QA 对（训练集 25,252 + 测试集 1,000）。
- **外部基准测试集**：**FailureSensorIQ**，用于跨域评估泛化能力。

### 实验设置
- **目标模型**：以 **Qwen-4B-Instruct** 为主进行 full fine-tuning 和 LoRA 微调；同时对比 **Phi-3-mini-4k-Instruct** 和 **RAG-Instruct-Llama3-8B**。
- **微调方式**：
  - **Full fine-tuning**：更新所有参数。
  - **LoRA**：低秩适配，仅更新少量参数。
- **RAG 设置**：使用 EmbeddingGemma 模型 + FAISS 构建检索系统，检索范围为 Panasonic 文档库。

### 评估指标
针对多选问答任务，采用以下集合论（set-based）指标，避免因选项顺序不同导致误判：
- **Set-Match Accuracy**：预测集合与真实集合完全相等的比例。
- **F1-Score**：多标签分类的标准 F1 分数。
- **Jaccard Similarity**：预测集与真实集的交并比。
- **MMLU**：用于评估微调前后通用知识保留情况，防止 **catastrophic forgetting**。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在 Panasonic 自建 Benchmark 上的表现（Qwen-4B-Instruct）
| 配置 | Set-Match Acc. | F1-Score | Jaccard |
|------|----------------|----------|---------|
| **Base Model (Original)** | 28.5% | 46.6% | 41.6% |
| **+ Full FT (Qwen-generated data)** | **42.0%** | **63.5%** | **57.9%** |
| **+ Full FT (Claude-generated data)** | **56.4%** | **72.7%** | **68.9%** |

> ✅ 结果表明：**full fine-tuning 显著提升性能**，且使用 **Claude 生成的数据训练效果更优**。

#### LoRA 微调效果
| 配置 | Set-Match Acc. | F1-Score | Jaccard |
|------|----------------|----------|---------|
| **Base Model** | 28.5% | 46.6% | 41.6% |
| **+ LoRA (various ranks)** | ~28.5% | ~46.7% | ~41.7% |

> ❌ LoRA 微调几乎未带来性能增益，说明该任务需要深度知识注入，而非简单行为对齐。

#### 外部基准 FailureSensorIQ 表现对比
| 模型 | AccOrgIBM | AccPerIBM | F1-Macro | F1-Micro |
|------|-----------|-----------|----------|----------|
| **Qwen-Org** | 34.0% | 0.0% | 40.0% | 66.0% |
| **Qwen-Pana-Qwen (FT on Qwen data)** | 27% | 0% | **43%** | **74%** |
| **Qwen-Pana-Claude (FT on Claude data)** | **49.6%** | 0.0% | 33.5% | 50.3% |
| **RAG-Instruct-Llama3-8B** | 28.0% | **33.0%** | **62.0%** | **68.0%** |

> ⚠️ 发现：不同数据生成器导致相反趋势——Qwen 数据提升 F1 但降低 AccOrgIBM，而 Claude 数据则反之。

### 与基线方法的对比
- **RAG-Instruct-Llama3-8B** 在 Panasonic 数据上表现极差（Set-Match Acc. ≈ 1%），说明通用 RAG 模型无法处理此类专业工业文档。
- **Phi-3-mini-4k-Instruct** 基础性能低于 Qwen-4B-Instruct，验证了模型容量的影响。

### 消融实验结果
- **数据生成器质量影响显著**：
  - 使用 **Claude-Opus-4.6** 生成原始数据时，仅有 **0.5%**（143/26,395）样本被过滤。
  - 而使用 **Qwen3-30B-A3B-Instruct** 生成时，高达 **43%**（10,353/23,910）样本因格式错误被丢弃。
  > 表明 **Claude 生成的数据质量更高、更符合规范**。

- **成本对比悬殊**：
  | 生成器 | 成本估算 |
  |--------|----------|
  | Qwen3-30B-A3B-Instruct | **$3.2**（本地计算） |
  | Claude-Opus-4.6 | **$330**（API 费用） |
  > 尽管 Claude 效果更好，但成本高出两个数量级。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Industrial-Instruction 框架有效**：提出的端到端流水线能够从真实工业文档中自动生成高质量的 instruction-tuning 和 benchmark 数据集。
2. ✅ **full fine-tuning 显著提升小模型性能**：在 Panasonic 数据上，Qwen-4B-Instruct 经 full fine-tuning 后，Set-Match Accuracy 提升近 **13.5–14 个百分点**。
3. ✅ **数据生成器选择至关重要**：使用 **Claude-Opus-4.6** 生成的数据不仅原始质量更高，下游微调增益更大，且几乎不引起 **catastrophic forgetting**。
4. ✅ **open-weight 模型仍具性价比**：尽管性能略逊，但 **Qwen3-30B-A3B-Instruct** 以极低成本生成了可用数据集，适合预算有限的研究者。
5. 🔄 **MMLU 通用知识保留良好**：
   - 微调后 MMLU 总体准确率下降很小（Qwen: -1.26%，Claude: -0.05%）。
   - 使用 Claude 数据微调甚至在部分科目（如 college_mathematics, machine_learning）上有轻微提升。

### 方法的局限性
1. ❌ **对问题重述（perturbation）极度脆弱**：所有基于 Qwen 的模型在 FailureSensorIQ 的 **AccPerIBM** 上均为 **0%**，即使经过微调也无法改善，暴露出小模型在语义鲁棒性上的根本缺陷。
2. ❌ **当前数据未涵盖扰动变体**：Industrial-Instruction 数据集中没有包含同义改写或对抗性重述的问题，因此无法训练模型应对此类挑战。
3. ❌ **图像信息被忽略**：预处理阶段移除了所有图像，限制了对图文混合文档的理解能力。

### 未来工作方向
1. **增强鲁棒性训练**：在数据生成过程中显式加入 **paraphrased 或 adversarial rephrased 版本** 的 QA 对，以提升模型对问题表述变化的容忍度。
2. **扩展数据来源**：整合来自更多行业（如半导体、航空、医疗设备）的技术文档，构建更大规模、更具代表性的 **comprehensive industrial benchmark**。
3. **探索高级 RAG 架构**：超越简单的检索机制，引入 multi-hop、agent-style 的复杂 RAG 架构。
4. **发展多模态工业模型**：将嵌入的图像、示意图和结构化视觉内容纳入理解流程，实现真正的 **multimodal industrial QA**。

> 🔗 所有资源已开源：
> - GitHub: [https://github.com/parssky/industrial-instruction](https://github.com/parssky/industrial-instruction)
> - Hugging Face Dataset: [https://huggingface.co/datasets/Parssky/industrial-instruction-dataset](https://huggingface.co/datasets/Parssky/industrial-instruction-dataset)

</details>

---

### 4. [Mixture of Channel Experts: Static Sparse Supports with Input-Adaptive Mixing for Pointwise Projections](https://arxiv.org/abs/2608.23794)

**Authors**: Elian Iluk, Gil Ben-Artzi  
**Category**: cs.LG  
**Published**: 2026-08-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.23794v1  

#### Abstract
Mixture-of-Experts (MoE) scales language models by routing each input through a small set of independently parameterized experts. We show that copying this design into convolutional networks fails for a structural reason: parallel convolutional experts that read the same input channels learn nearly ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Mixture of Channel Experts: Static Sparse Supports with Input-Adaptive Mixing for Pointwise Projections**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

- **传统 MoE 在卷积网络中的失效**：直接将 Mixture-of-Experts (MoE) 结构迁移到卷积网络中时，多个并行的卷积专家在共享相同输入的情况下会学习到高度相似的滤波器（kernel），导致参数冗余而无实际功能分化（cosine similarity 高达 0.88）。
- **点对点投影（pointwise projection）的高计算成本**：现代视觉模型（如 ResNet、EfficientViT、Vision Transformers）广泛使用的 1×1 卷积（即 pointwise projection）具有 $O(C^2)$ 的 MACs 成本，是效率瓶颈。

### **提出了什么新方法或新思路**

提出 **Mixture of Channel Experts (MoCE)**，一种受 MoE 启发但结构不同的稀疏通道混合机制，用于替代传统的密集点对点投影。

#### **核心设计思想**：
- **从“操作符复制”转向“通道选择”**：不再复制整个专家模块，而是将每个输出通道视为一个“expert”，其只从输入通道中选择一个稀疏子集（top-k）进行聚合。
- **静态稀疏支持 + 动态混合方式**：
  - **Static Sparse Support**：每个 expert 的输入通道选择（support）在训练后固定，推理时无需动态路由，可静态调度，硬件友好。
  - **Input-Adaptive Mixing**：通过一个轻量级门控网络预测 **温度（temperature）**，动态调整 softmax 聚合的集中程度（从 max-like 到 mean-like），实现输入自适应的混合行为。
- **残差专家（Residual Expert）**：未被任何 expert 选中的通道由一个残差通道统一处理（均值池化），确保信息不丢失。
- **负载均衡正则项（Load-Balancing Loss）**：鼓励所有输入通道被均匀利用，防止某些通道被过度忽略。

### **相比现有方法的优势**

| 对比维度 | MoCE 优势 |
|--------|---------|
| **效率** | 将点对点投影的计算量从 $O(CE)$ 降至 $O(kE + C)$，相对成本约为 $k/C$，显著降低 MACs 和延迟 |
| **精度** | 在多个任务上 **匹配甚至超越** 密集模型和其他稀疏方法（如 SE、CondConv、Pick-or-Mix） |
| **部署友好性** | 支持静态编译和固定内存访问模式，适合边缘设备部署 |
| **参数量** | 部署时仅保留选定的 logits 和索引，参数减少 17–21% |
| **设计理念** | 证明了“如何混合”比“选择哪些通道”更值得投入输入依赖性资源 |

---

## 2. 核心实验方法和设置

### **使用了哪些数据集**

- **ImageNet-1K**：主评估数据集，用于训练和评估 ResNet 系列模型。
- **CIFAR-100**：
  - 从零开始训练（scratch training）
  - 迁移学习：在 ImageNet 上预训练后微调至 CIFAR-100
- **EfficientViT**：应用于轻量级 Vision Transformer 架构，在 CIFAR-100 上评估。

### **实验设置和评估指标**

| 设置项 | 描述 |
|------|-----|
| **骨干网络** | ResNet-50/101/152、EfficientViT-M2/M3/M5 |
| **替换位置** | 
  - ResNet：bottleneck 中的入口 pointwise 投影（ratio s=4）
  - EfficientViT：ConvMlp 块中的第二个 feed-forward 投影（s=2）
| **MoCE 参数** | 默认 $k=8$（ResNet）、$k=64$（EfficientViT），温度范围 $T \in [0.2, 6.0]$，正则系数 $\lambda = 5\times10^{-4}$（ImageNet） |
| **训练配置** | 使用原始 ResNet 训练流程（非现代增强策略），便于公平比较；SGD，cosine 学习率衰减，batch size 512 |
| **评估指标** |
  - Top-1 Accuracy
  - MACs（Multiply-Accumulate Operations）
  - Parameters（训练 / 部署）
  - Wall-clock 推理时间（batch=512，FP32）

### **基线方法对比**

| 方法 | 类型 | 特点 |
|-----|------|------|
| **Dense 1×1** | 密集投影 | 原始 baseline |
| **SE (Squeeze-and-Excitation)** | 通道重校准 | 引入额外参数和计算 |
| **CondConv** | 条件卷积 | 复制核，参数爆炸 |
| **Pick-or-Mix** | 动态通道采样 | 输入依赖性强，访存不可预测 |
| **Random / Cyclic Fixed Supports** | 消融对照 | 验证学习支持的重要性 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **ImageNet-1K 结果（Table 2）**

| Model | Params (M) | MACs (G) | Top-1 (%) |
|-------|------------|----------|-----------|
| ResNet-50 (Dense) | 25.56 | 4.112 | 75.98 ± 0.30 |
| + MoCE | 25.55 / **21.26** | **3.426** | **76.71 ± 0.20** |
| ResNet-152 (Dense) | 60.19 | 11.559 | 77.78 ± 0.43 |
| + MoCE | 60.15 / **47.84** | **9.156** | **78.24 ± 0.12** |

- **MACs 减少 16.7% ~ 20.8%**
- **部署参数减少 16.8% ~ 20.5%**
- **精度提升 0.46 ~ 0.73 个百分点**

#### ✅ **与条件通道方法对比（Table 3）**

| Method | Params (M) | MACs (G) | Top-1 (%) |
|--------|------------|----------|-----------|
| SE | 28.07 | 4.120 | 76.64 |
| CondConv | 55.97 | 4.150 | 76.52 |
| Pick-or-Mix | 25.56 | **3.178** | 76.25 |
| **MoCE** | **21.26** | 3.426 | **76.71** |

- MoCE 在更低参数和计算下达到更高精度
- Pick-or-Mix 虽 MAC 更低，但精度略逊且推理路径更慢（见 timing）

#### ✅ **CIFAR-100 与迁移学习（Table 4）**

| Setting | MACs ↓ | Accuracy ↑ |
|--------|--------|-----------|
| From Scratch (ResNet-50) | 1.305 → 1.081 G | 78.44 → 79.35% |
| Transfer (ResNet-50) | 4.11 → 3.43 G | 97.52 → 97.76% |

- 所有设置下均实现 **MACs 下降 17–21%**，**精度提升或持平**
- 支持在 ImageNet 上学习的通道支持可有效迁移到其他任务

#### ✅ **EfficientViT 结果（Table 5）**

| Model | MACs (G) | Top-1 (%) |
|-------|----------|-----------|
| EfficientViT-M5 | 0.526 → 0.410 | 75.56 → 75.95 |
| EfficientViT-M2 | 0.204 → 0.167 | 74.61 → 74.75 |

- 在非 ResNet 架构上同样有效
- 精度基本保持，计算显著下降

#### ✅ **端到端推理速度（Table 7）**

| Scope | Dense (ms) | MoCE (ms) | Speedup |
|-------|------------|-----------|---------|
| End-to-end (ResNet-50) | 133.12 | 127.00 | **1.05×** |
| All replaced ops | 19.63 | 15.60 | **1.26×** |
| Wide proj. (2048→512) | 2.116 | 0.701 | **3.0×** |

- 在宽通道投影中加速比高达 **3×**
- 整体网络加速约 **4.6%**，受限于内存带宽（见分析）

---

### **消融实验结果（Table 6）**

| 消融设置 | Top-1 (%) | 观察结论 |
|---------|-----------|----------|
| **Input-conditioned support**（动态选择通道） | 79.38 | +0.03 vs MoCE，但 **routing+gather 时间 7.13× 更慢**（1.9 → 13.5ms） |
| **Static mixing (T=1)**（关闭温度门控） | 78.16 | **↓1.19 pts**，说明温度调节至关重要 |
| **No residual expert** | 78.61 | ↓0.74 pts，验证残差通路必要性 |
| **Random fixed supports** | 61.30 | 性能崩溃，说明支持选择需有意义 |
| **Deterministic cyclic supports** | 78.27 | 接近 dense（78.44），表明 **8通道足以表达全连接能力** |
| **Learned supports** | 79.35 | 最优，证明学习支持的价值 |
| **k=4/8/10/16** | 76.28/76.71/75.98/75.73 | **k=8 最优**，过小限制容量，过大削弱专业化 |
| **Temperature bounds** | [0.2,6.0] 最佳 | 温度过宽或过窄均不利 |

> 🔍 **关键发现**：  
> - “**如何混合**” 比 “**选择哪些通道**” 更值得使用输入依赖性；
> - 一个轻量温度门控（scalar output）带来的收益远超复杂动态路由；
> - 即使是固定规则的支持分配也能接近 dense 性能，说明原始投影存在巨大冗余。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **MoE 直接移植失败的根本原因**：当多个卷积专家接收相同输入表示时，缺乏差异化监督会导致权重高度对齐，无法实现有效专业化。
2. ✅ **通道稀疏性 + 固定支持 + 输入自适应混合 是高效设计的关键**：
   - **静态支持** 保证硬件效率；
   - **温度门控** 实现低成本输入自适应；
   - **残差专家** 保障信息完整性。
3. ✅ **输入依赖性的“性价比”差异显著**：
   - 动态选择通道（high-cost, high-expressiveness）带来几乎为零的增益；
   - 温度调节（low-cost, low-expressiveness）却带来超过 1 个点的提升。
4. ✅ **原始 pointwise 投影存在严重冗余**：
   - 仅用 8 个输入通道即可重建 dense 投影 99.8% 的性能；
   - 学习稀疏支持可进一步超越 dense 模型。

### **方法的局限性**

- ❌ **仍受限于内存带宽**：尽管 MACs 大幅下降，但由于全局池化和残差读取仍需访问全部激活，实际加速比小于 MAC 减少量（roofline 分析显示进入 bandwidth-bound 区域）。
- ❌ **当前设计不适用于 s > C 的扩展层**：MoCE 要求 $E \leq C$，不能直接用于通道扩展场景（如 expansion layer）。
- ❌ **温度门控依赖排序不变性**：输入必须按 routing logit 降序排列，限制了完全自由的通道排列。

### **未来工作方向**

1. **融合内存访问优化**：
   - 将 global pooling 与 gather 操作融合，减少重复激活读取；
   - 探索硬件协同设计以进一步释放带宽潜力。
2. **推广至更广结构**：
   - 设计适用于 $E > C$ 场景的 MoCE 变体；
   - 应用于 Vision Transformer 的 token mixing 层。
3. **探索更高效的输入适配机制**：
   - 在保持静态支持的前提下，研究多标量控制或其他轻量动态机制。
4. **理论分析稀疏表示能力**：
   - 形式化分析为何少量通道即可逼近 dense 投影，揭示深层冗余机制。

---

> 📌 **一句话总结**：  
> **MoCE 通过“静态稀疏通道选择 + 输入自适应温度调节”的设计，在显著降低计算与参数的同时，实现了优于或媲美密集模型的性能，并揭示了“如何混合”比“选择哪些通道”更值得投入输入依赖性资源这一重要设计原则。**

</details>

---

### 5. [Low-Latency Activation-Regularized Sparse Neural Operators with Distillation Assistance Towards Real-Time Edge-Deployable Virtual Sensing](https://arxiv.org/abs/2608.23987)

**Authors**: William Howes, Farid Ahmed, Syed Bahauddin Alam  
**Category**: cs.LG  
**Published**: 2026-08-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.23987v1  

#### Abstract
Virtual sensing enables digital twins and safety-critical systems to reconstruct and forecast spatial-temporal physics in real time. However, conventional computational and data-driven methods often face challenges in generalization, latency, and energy efficiency for edge deployment. Neural operato...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文致力于解决**虚拟传感**（virtual sensing）在边缘设备上实时部署所面临的三大挑战：
1.  **高延迟**（High Latency）：传统的基于多时间步长（multi-step）的 Spiking Neural Networks (SNNs)，如 Variable Spiking Neuron (VSN) 和 Leaky Integrate-and-Fire (LIF)，虽然能提高精度，但会显著增加推理延迟，不满足实时性要求。
2.  **训练不稳定**（Unstable Training）：SNNs 依赖于 **surrogate-gradient** 进行反向传播，这导致前向传播（二值化脉冲）与反向传播（连续梯度）之间存在固有的不匹配，造成训练困难和性能下降。
3.  **效率-精度权衡困难**（Challenging Efficiency-Accuracy Trade-off）：现有的稀疏化方法（如基于脉冲百分比的正则化）难以精确控制激活稀疏度，且可能导致模型表达能力崩溃。

### 提出的新方法和新思路
为解决上述问题，论文提出了以下四项核心创新：

1.  **Sparse-Activation-ReLU (SAR) 层**：
    *   **新思路**：提出一种**单步**（single-step）、**无代理梯度**（surrogate-free）的稀疏神经算子框架。该层直接使用 `y = σ(ReLU(z - T))`，其中 `σ(0)=0`，利用 ReLU 天然的稀疏性来模拟事件驱动计算。
    *   **优势**：完全避免了 surrogate-gradient 训练，实现了稳定、高效的端到端训练。它保留了 VSN 的“变量通信”（variable communication）特性以保证回归任务的精度，同时通过移除记忆动态（memory dynamics）将延迟降至最低。

2.  **基于范数的激活稀疏正则化**：
    *   **新思路**：用 **Hoyer 范数** 或 **L1 范数** 直接对 SAR 层的输出进行正则化，取代 VSN 中的脉冲百分比正则化。
    *   **优势**：提供了更精细、更可控的稀疏度调节机制。特别是 Hoyer 范数因其尺度不变性，能更有效地促进真正的稀疏模式，而非简单地压制激活幅度。

3.  **合成知识蒸馏**（Synthetic Knowledge Distillation）：
    *   **新思路**：引入一个“教师-学生”框架。一个复杂但不适合边缘部署的模型（如 VIRSO）作为**教师模型**，生成合成数据；一个轻量级、适合边缘部署的 SAR 模型作为**学生模型**，从这些合成数据中学习。
    *   **优势**：解决了边缘模型因架构简单而导致的表达能力不足问题，同时保持了其低功耗、低延迟的优势，有效缓解了小样本数据下的过拟合。

4.  **激活正则化改进 VSN**：
    *   **新思路**：将 SAR 的思想反哺给传统 VSN。用一个基于 **ReLU 的激活损失**（`a(t) = ReLU(βM(t-1) + z(t) - Θ)`）替换原有的脉冲百分比损失，并结合 **graph-neighbor thresholding** 来减少不必要的空间聚合。
    *   **优势**：在保留 VSN 时间动态的前提下，提升了其训练效率和性能，为未来更复杂的时序建模提供了改进路径。

---

## 2. 核心实验方法和设置

### 数据集
实验在两个具有挑战性的物理场重建基准上进行：
1.  **2D Heat Exchanger**：从三维换热器截取的二维截面，输入为入口温度、速度和壁面热流，输出为压力和三维速度场。几何不规则，流动复杂。
2.  **Lid-Driven Cavity (LDC)**：顶盖驱动方腔流，输入为随时间变化的顶盖速度曲线，输出为压力、速度大小和湍动能场。

### 实验设置和评估指标
*   **模型架构**：
    *   **SAR-NOMAD**：在 NOMAD 架构中用 SAR 层替换所有 ReLU 层。
    *   **SAR-GNO**：在 VIRSO/GNO 架构中插入 SAR 层。
    *   **基线模型**：VS-NOMAD, VS-GNO, LIF-NOMAD, LIF-GNO。
*   **评估指标**：
    *   **主指标**：**Latency-Error-Energy (LEE) Score**，定义为 `STS × Mean Relative L2 Error (%) × Mean Spiking Percentage (%)`。该指标综合衡量了延迟、精度和能耗，值越低越好。
    *   **辅助指标**：`Relative L2 Error`, `Spiking Percentage`, `Latency-Error (LErr)`, `Error-Energy (EE)`, `Latency-Energy (LEn)`。
    *   **分析工具**：**Spiking Entropy**，用于量化特征维度的利用率。

### 基线方法对比
-   **VS-NOMAD/VS-GNO**：使用 VSN 神经元，采用 surrogate-gradient 训练。
-   **LIF-NOMAD/LIF-GNO**：使用 LIF 神经元，同样采用 surrogate-gradient 训练。
-   **Baseline NOMAD/VIRSO**：原始的非稀疏化模型。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **SAR-NOMAD vs. VS/LIF-NOMAD**：
    *   在 **2D Heat Exchanger** 上，SAR-NOMAD (Hoyer, γ=0.005) 的 LEE 得分为 **26.35**，而表现最好的 VS-NOMAD (Arctan, 1 STS, γ=0) 的 LEE 得分为 139.48。**SAR 实现了超过五倍的 LEE 改进**。
    *   在 **Lid-Driven Cavity** 上，SAR-NOMAD (Hoyer, γ=0.05) 的 LEE 得分为 **37.96**，同样比最佳的 VS-NOMAD 配置高出五倍以上。
    *   SAR-NOMAD 在更低的延迟（1 STS）下，达到了比多时间步长（10/20 STS）VSN 更优的精度和能效。

2.  **SAR-GNO vs. VS-GNO**：
    *   在 **spectral-only** 配置下，SAR-GNO 性能与 VS-GNO 相当。
    *   在 **full** 配置下，SAR-GNO 表现出轻微优势。例如，在 γ=0.01 时，SAR-GNO 的 **Error-Energy** 得分（7.59）低于 VS-GNO（8.15）。

3.  **合成知识蒸馏效果**：
    *   在 2D Heat Exchanger 上，为 SAR-NOMAD 添加 **8000 个**由 VIRSO 教师模型生成的合成样本后：
        *   当 γ=0.005 时，平均相对 L2 误差从 **5.41%** 降低至 **2.68%**，**降低超过两倍**。
        *   对应的 LEE 分数从 26.35 降低至 **11.34**，**降低了超过两倍**。
    *   这证明了该框架能显著提升学生模型的性能，而无需牺牲其边缘友好性。

4.  **消融实验结果**：
    *   **Hoyer vs. L1 正则化**：Hoyer 范数在控制稀疏度方面远优于 L1。L1 正则化倾向于完全抑制分支网络（branch networks），导致信息丢失和精度骤降；而 Hoyer 能更均衡地抑制各部分，维持更好的精度-稀疏度权衡。
    *   **SAR 层阈值参数 T**：在非线性层（如归一化层后）引入可学习的阈值 `T`，能更好地控制稀疏性，提升优化效果。
    *   **改进的 VSN**：使用 SAR 启发的激活损失（而非脉冲百分比损失）训练的 VSN (ARVS-NOMAD)，其 L2 误差**降低超过两倍**（例如，在 1 STS 下从 >30% 降至 <9%），同时保持了相似或更低的脉冲率。

---

## 4. 关键结论和发现

### 主要发现
1.  **SAR 是一个强大的低延迟替代方案**：提出的 SAR 框架成功地提供了一个**无代理梯度**、**单步**、**高能效**的神经算子实现方式。它在 LEE 指标上相比传统 SNN 方法取得了数量级的提升，是实现实时边缘虚拟传感的理想候选。
2.  **Hoyer 正则化是关键**：相比于传统的 L1 或脉冲百分比正则化，**Hoyer 范数**能更有效地诱导出有意义的稀疏模式，防止模型崩溃，从而实现更优的精度-效率权衡。
3.  **合成蒸馏是有效的桥梁**：合成知识蒸馏框架成功地弥合了高性能模型与边缘部署模型之间的鸿沟，使得轻量级模型能够从复杂模型的知识中受益，极大地提升了其在小样本场景下的泛化能力。
4.  **SAR 思想可反哺传统 SNN**：将 SAR 的激活稀疏化思想应用于 VSN 的损失函数设计，可以显著改善其训练过程和最终性能，验证了该思路的普适价值。

### 方法的局限性
1.  **表达能力受限**：SAR 层的输出严格非负，而 VSN 可以处理负信号。这可能限制了 SAR 层在某些需要双向信息流任务中的表达能力。
2.  **牺牲了时间动态**：SAR 为了追求极致的低延迟，完全放弃了 SNN 的时间积分和记忆动态，因此无法捕捉和建模物理系统中的长期时序依赖关系。
3.  **硬件验证缺失**：所有能效评估均基于软件模拟的“脉冲百分比”，尚未在真实的 neuromorphic 硬件（如 Loihi 2）上进行部署和验证，实际的能效增益有待确认。

### 未来工作方向
1.  **扩展 SAR 的表达能力**：探索允许负激活的稀疏化层，以逼近 VSN 的完整表达能力。
2.  **发展无代理梯度的时间动态模型**：研究如何将 SAR 的稳定训练优势与 SNN 的时间动态相结合，开发既能高效训练又能处理时序信息的新型算子。
3.  **探索更高效的蒸馏策略**：研究如何用更少的合成样本或更智能的采样策略达到相同的蒸馏效果，降低训练成本。
4.  **真实硬件部署与验证**：将 SAR 模型部署到实际的边缘或 neuromorphic 硬件上，测量真实的延迟、吞吐量和功耗，以全面评估其性能。

</details>

---

### 6. [Selective Regenerative Decoding: Trajectory-Level Intervention for Inference-Time Reasoning](https://arxiv.org/abs/2608.24338)

**Authors**: Sophia Xiao Pu, Yumo Xu, Sailik Sengupta, Millennium Bismay, Ruixue Lian, James Gung, Yi-an Lai, Arshit Gupta  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.24338v1  

#### Abstract
Inference-time decoding methods improve LLM reasoning by exploring multiple candidate trajectories, yet treat each trajectory as atomic: either retaining it whole or discarding it irreversibly. This wastes computation on partially promising candidates whose high-quality prefixes are abandoned alongs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Selective Regenerative Decoding: Trajectory-Level Intervention for Inference-Time Reasoning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **inference-time decoding** 方法（如 Best-of-N 和 Speculative Rejection）将整个推理轨迹（reasoning trajectory）视为原子单元进行处理：
- **Best-of-N**：生成多个完整轨迹后选择最优者，但会浪费大量计算在部分高质量的候选上。
- **Speculative Rejection**：在前缀质量下降时提前终止，但一旦拒绝就永久丢弃，导致有用的前缀被废弃。

这种“全有或全无”的策略在长文本推理中效率低下，因为许多轨迹具有**高质量前缀 + 退化后缀**的特征。

### 提出了什么新方法或新思路
本文提出 **Selective Regenerative Decoding (SRD)**，一种支持**段级干预**（segment-level intervention）的解码算法，其核心思想是：
- 不再整条丢弃或保留轨迹，而是对每个候选轨迹进行 **路由决策**（Routing）：`KEEP`、`REFINE` 或 `DISCARD`。
- 对于处于中间质量的候选（`REFINE`），仅重新生成其**质量退化的后缀部分**，而保留高质量前缀。
- 采用更高温度采样（higher-temperature sampling）进行局部重写，无需依赖更大的目标模型（target model）。

### 相比现有方法的优势
- ✅ **更高的样本效率**：相比 rejection sampling 可实现 **1.28–1.36× 的样本效率增益**。
- ✅ **更优的期望轨迹质量**：严格优于 rejection sampling 的最佳轨迹质量。
- ✅ **无需额外模型**：仅需一个生成模型（generator）和一个奖励模型（reward model），不依赖更大能力的 target model。
- ✅ **可组合性强**：与 speculative decoding、prefix value functions 等方法兼容，可作为通用推理框架的一部分。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖四类典型推理任务：
| 数据集 | 任务类型 | 评估指标 |
|--------|---------|----------|
| **MATH500** | 数学推理 | Accuracy |
| **GPQA Diamond** | 科学问答（高难度） | Accuracy |
| **HotpotQA** | 多跳问答 | EM, F1 |
| **AlpacaEval** | 指令跟随 | GPT-4o-mini Win Rate |

### 实验设置和评估指标
- **生成模型与奖励模型解耦**：避免偏好偏差（bias toward generator’s own style）。
- **生成参数统一**：所有方法使用相同 temperature=0.8、top-p=0.9、max length。
- **评估方式**：报告任务层面的真实性能（accuracy/f1/win rate），而非 reward score。
- **硬件平台**：8×A100 (40GB) GPU，使用 vLLM 进行生成，HuggingFace Transformers 执行 reward model 推理。

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Temperature Sampling (N=1)** | 单次采样，最基础 baseline |
| **Best-of-N (BoN)** | 完整生成 N 条轨迹并选最高分者 |
| **Speculative Rejection (Spec-Rej)** | 在生成过程中实时评分，低分前缀立即终止 |

> 注：未直接比较 Reward-guided Speculative Decoding (RSD)，因其需要更大的 target model；作者在附录中提供了受控对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 在 MATH500 上的表现（Llama-3.1-8B + AceMath RM）
| N | Accuracy | Output Tokens |
|----|----------|----------------|
| 10 | 0.544 | 2,166 |
| 50 | 0.611 | 10,927 |
| 100 | 0.640 | 21,840 |

> SRD 以远少于 BoN 的输出 token 数量达到相近甚至更高的准确率。

#### 效率增益验证（Theorem 3.3 实证）
在 MATH500 上测量理论参数：
| $N$ | $p_H$ | $p_M$ | $p$ (refinement success) | 效率增益 |
|-----|-------|--------|----------------------------|-----------|
| 10 | 0.522 | 0.148 | 1.00 | **1.28×** |
| 20 | 0.513 | 0.180 | 1.00 | **1.35×** |
| 30 | 0.509 | 0.182 | 1.00 | **1.36×** |

> 随着候选池增大，更多 mid-tier 轨迹可被修复，效率增益持续提升。

#### 最佳轨迹质量提升（Theorem 3.5 实证）
在耦合设置下（相同初始种子），SRD 显著优于 rejection sampling：
| $N$ | $\Delta R_{\text{max}}$ | SRD 赢得样本比例 |
|-----|--------------------------|--------------------|
| 10 | +0.36 | 46% vs 18% |
| 20 | +0.28 | 54% vs 24% |
| 30 | **+1.29** | 52% vs 24% |

> 表明 SRD 成功通过 refinement 提升了最终输出质量。

### 与基线方法的对比结果
- 📈 **在低计算预算下显著优于 Speculative Rejection**：尤其在数学和科学推理任务中表现突出。
- ⚖️ **匹配 Best-of-N 准确率但节省大量 token**：例如在 MATH500 上，SRD 用更少 token 达到 BoN 的精度。
- 🔁 **在指令跟随任务（AlpacaEval）中稳定有效**：即使 reward signal 存在噪声，仍能保持竞争力。

### 消融实验结果（Ablation Studies）

#### （1）路由阈值影响（Routing Thresholds）
| $(\theta_{\text{high}}, \theta_{\text{low}})$ | Accuracy | Out Tokens |
|---------------------------------------------|----------|------------|
| (0.5, 0.3) ✅（默认） | **0.5444** | 2166 |
| (0.5, 0.5) | 0.5400 | 1973 |
| (0.8, 0.5) | 0.4667 | 1871 |

> 过于激进或宽松的阈值都会损害性能，**(0.5, 0.3)** 是较优平衡点。

#### （2）不同 refinement policy 的效果
| Policy | MATH500 Acc | GPQA Acc |
|--------|--------------|----------|
| **Reroute(global)** ✅ | **0.5444** | 0.2973 |
| **Self-compare** ✅ | 0.5156 | **0.3243** |
| Refine-Bo10 | 0.5244 | 0.3176 |
| Force keep | 0.5267 | 0.3041 |

> 发现：
- 在 reward model 稳定时（MATH500）：全局排序 rerouting 更优。
- 在 reward model 噪声大时（GPQA）：本地比较 self-compare 更鲁棒。
- 内部使用 Best-of-N 并不能提效，反而增加开销。

#### （3）评分间隔（Scoring Interval）的影响
| Interval | MATH500 Acc | GPQA Acc |
|---------|-------------|----------|
| 1 | 0.5533 | 0.2365 ❌ |
| 10 | 0.5667 ✅ | 0.2973 ✅ |
| 100 | 0.5467 | 0.2365 ❌ |

> 中等粒度（如 10–50 步）能最好地平衡定位精度与稳定性。

---

## 4. 关键结论和发现

### 主要发现
1. **轨迹不应被视为原子单元**：许多“失败”轨迹其实拥有高质量前缀，应允许局部修复。
2. **Selective Regeneration 显著提升效率**：相比 rejection sampling 可获得 **1.28–1.36× 的样本效率增益**。
3. **SRD 在多种任务上一致有效**：涵盖数学、科学、多跳问答和指令跟随。
4. **Refinement 的有效性取决于 reward model 质量**：
   - 若 reward model 稳定 → 全局 rerouting 更好；
   - 若 reward model 噪声大 → 局部 self-compare 更可靠。

### 方法的局限性
- ❗ **依赖固定路由阈值**：当前使用人工设定的 $\theta_{\text{high}}, \theta_{\text{low}}$，缺乏自适应能力。
- ❗ **边界检测为启发式规则**：再生起点 $j^*$ 由滑动窗口判断，可能不够精确。
- ❗ **reward model 偏见会被放大**：SRD 主动“改写”内容以提高 reward，可能导致风格/长度偏见被强化。
- ❗ **实现复杂度较高**：需协调 generator、editor、scorer、router 多个模块，增加部署成本。

### 未来工作方向
- 🔄 **学习动态路由策略**：通过 RL 或 meta-learning 自动优化 $\theta$ 阈值和再生边界。
- 🧠 **结合过程监督信号**：利用 step-level reward 或 process reward models 提高 refinement 精准度。
- 🛠️ **端到端训练 routing & refinement policy**：联合优化生成与编辑行为。
- 🌐 **扩展至多模态或结构化输出场景**：如代码生成、表格填充等任务中的局部修正机制。

---

> ✅ **总结一句话**：  
> **SRD 开辟了一个新的 accuracy-compute tradeoff 区域——它不像 BoN 那样盲目生成，也不像 Spec-Rej 那样武断放弃，而是“聪明地修补”，让每一段有价值的推理都不被浪费。**

</details>

---

### 7. [Neurosymbolic Alignment for Physiologically-Safe Clinical Language Models](https://arxiv.org/abs/2608.24534)

**Authors**: Abdulhady Abas Abdullah, Erik Cambria, Milena Zivkovic  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.24534v1  

#### Abstract
Clinical LLMs can generate recommendations that are factually plausible yet physiologically unsafe. We investigate whether safety alignment can be improved by grounding preference optimization in structured physiological knowledge rather than text-only supervision. Methods: We propose Neurosymbolic ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Neurosymbolic Alignment for Physiologically-Safe Clinical Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Clinical LLMs** 虽然在医学知识和语言流畅性上表现良好，但可能生成“事实正确但生理上不安全”的建议，例如：
- 推荐存在 **drug-drug interactions (DDI)** 的药物组合
- 忽视患者的 **homeostatic violations**（如高钾血症）
- 在肾功能不全患者中推荐禁忌药物

这类错误并非简单的“事实错误”，而是**上下文感知的生理一致性缺失**，传统基于文本的监督学习（如 DPO、ORPO）难以有效捕捉。

### 提出的新方法与创新思路
作者提出 **Neurosymbolic Alignment**，一种在训练时将 LLM 与结构化生理知识结合的安全对齐框架，其核心思想是：

- **Physiology-grounded preference generation**：  
  构建一个基于 **Heterogeneous Graph Neural Network (HGNN)** 的 **Physiological World Model**，利用包含 847K 节点的生物医学知识图谱（KG），从结构化知识中自动生成偏好信号，而非依赖人工标注的偏好对。

- **Iterative ORPO**：  
  在每个训练迭代中，由当前策略模型生成多个候选响应，通过 HGNN 对其进行生理可行性打分，形成偏好对，并用于更新策略。该过程是**on-policy**且**迭代式**的，能逐步暴露更细微的约束违反。

- **训练-推理解耦设计**：  
  HGNN 仅用于训练阶段生成偏好信号，推理时仅使用对齐后的 LLM，避免部署时引入图查询延迟，保持高效性。

### 相比现有方法的优势
- **无需人工标注偏好对**：自动从 KG 中生成高质量安全偏好信号，降低标注成本。
- **超越纯文本监督**：通过多跳路径推理、动态约束检查等机制，捕捉复杂生理交互。
- **优于运行时修正机制**：相比 inference-time self-correction 或 rule-based guardrails，训练时内化安全模式效果更优。
- **可扩展性强**：适用于 open-weight 模型，为临床 LLM 安全对齐提供可复现范式。

---

## 2. 核心实验方法和设置

### 使用的数据集

| 数据集 | 描述 |
|-------|------|
| **CSB (Clinical Safety Benchmark)** ✅（本文提出） | 包含 2,500 个合成临床场景的基准，专门测试生成式临床推理中的**生理约束违反**，涵盖：<br>- Drug-Drug Interactions (25%)<br>- Contraindication Detection (20%)<br>- Homeostatic Violations (18%)<br>- Pediatric/Geriatric Dosing Errors (15%)<br>- Organ Impairment (12%)<br>- Polypharmacy (10%)<br>划分：1,750/250/500（train/valid/test） |
| MedQA-USMLE | USMLE 风格选择题，测试病理生理、药理、诊断能力 |
| PubMedQA | 基于 PubMed 摘要的问答任务 |
| MedMCQA | 大规模医学多选题数据集 |
| MMLU-Medical | 医学子集，覆盖遗传学、解剖学等 |
| DDI-Corpus | 药物相互作用标注语料 |
| i2b2-2010 | 临床概念提取与关系分类 |

> 所有数据集均实施了严格的防泄漏控制（时间分割、实体重叠过滤、KG 实体保留等）。

### 实验设置与评估指标

#### 主要模型架构
- **Base LLM**: Mistral-7B + LoRA 微调
- **Physiological World Model**: 基于 847K 节点的异构知识图谱构建的 4-layer R-GAT（Relation-aware GNN）
- **实体链接**: scispaCy → UMLS CUI 映射

#### 评估指标

| 指标 | 定义 |
|------|------|
| **CSS (Clinical Safety Score)** | 回应中无任何生理约束违反的比例（HGNN 可行性得分 > 0.85） |
| **HR (Hallucination Rate)** | 含有生理不可能陈述的回应比例（由 5 名医生盲评） |
| **DID (Drug Interaction Detection)** | 检出禁忌药物组合的 F1 分数 |
| **RSS (Rule-Engine Safety Score)** ✅（独立验证） | 使用 **DrugBank-Rule** 引擎进行后处理检查，判断是否触发警报（完全独立于 HGNN） |
| **PC (Physiological Consistency)** | 平均 HGNN 可行性得分 |
| **MA (Medical Accuracy)** | 医学 QA 准确率 |

> 特别强调 **RSS** 和 **HR** 是 **HGNN-independent** 的指标，用于防止评估循环（evaluation circularity）。

### 基线方法对比

| 基线方法 | 类型 |
|--------|------|
| SFT | 监督微调 |
| SFT+RAG / SFT+DenseRAG | 检索增强生成 |
| DPO / ORPO | 偏好优化（人类偏好标签） |
| KG-LLM | 静态 KG 增强 |
| DrugBank-Rule | 运行时规则引擎检查 |
| SFT-CSB | 在 CSB 上直接 SFT |
| SFT+SelfCorrect | 推理时自我修正（最多 3 轮） |
| GPT-4 (zero-shot / 5-shot) | 闭源模型对比 |
| MedAlpaca / BioMistral / MEDITRON / Med-PaLM 2 | 开源/专有医学 LLM |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（在 CSB 测试集上）

| 方法 | MA (%) | **CSS (%)** | **RSS (%)** | **HR (%)** | **DID (%)** | PC |
|------|--------|-------------|-------------|------------|-------------|-----|
| ORPO (baseline) | 70.1 | 69.5 | 65.2 | 14.1 | 72.8 | 0.71 |
| DrugBank-Rule | 68.3 | 76.1 | 82.4 | 9.8 | 80.4 | 0.77 |
| SFT+SelfCorrect | 68.8 | 79.4 | 83.7 | — | 82.6 | 0.78 |
| GPT-4 (5-shot) | 79.1 | 80.6 | 78.2 | — | 84.3 | 0.80 |
| **Ours (Iter-5)** | **75.2** | **90.8** | **86.4** | **5.1** | **91.6** | **0.89** |

> ✅ **所有指标均显著优于基线**

### 与基线方法的关键对比
- **vs ORPO**：
  - CSS ↑ **+21.3 pp**
  - HR ↓ **-9.0 pp**（医生盲评）
  - DID ↑ **+18.8 pp**
  - RSS ↑ **+21.2 pp**（独立规则引擎验证）
- **vs GPT-4 (5-shot)**：
  - 尽管参数量少 **10×**，但在所有安全指标上全面超越（CSS +10.2 pp）
- **vs SFT+SelfCorrect**：
  - CSS 高出 **11.4 pp**，表明训练时对齐优于推理时修正
- **鲁棒性测试**：
  - 在合成 EHR 噪声（缩写、缺失实体、顺序打乱）下仍保持 **84.2% CSS**

### 消融实验结果（Ablation Study）

| 消融配置 | CSS (%) | ΔCSS | HR (%) |
|----------|---------|------|--------|
| 完整模型 | 90.8 | — | 5.1 |
| 移除 HGNN 打分（仅用规则） | 74.6 | **-16.2** | 12.4 |
| 移除迭代训练 | 79.3 | **-11.5** | 9.8 |
| 移除 Homeostatic Constraints (`Shom`) | 82.1 | -8.7 | 9.4 |
| 移除 Drug Interaction Penalty (`Pint`) | 84.3 | -6.5 | 10.8 |
| 移除 Path Plausibility (`Spath`) | 86.9 | -3.9 | 7.2 |

> 🔍 **HGNN 打分** 和 **迭代训练** 是最关键的两个组件。

---

## 4. 关键结论和发现

### 主要发现
1. **训练时生理对齐显著提升安全性**：  
   将偏好信号建立在结构化生理知识之上，可系统性减少有害生成，**CSS 达到 90.8%**，远超现有方法。

2. **改进不仅来自 HGNN 本身**：  
   通过 **RSS**（r=0.97 与 CSS 高度一致）、**DID**、**HR** 等独立指标验证，性能增益具有泛化性和真实性，非“过拟合评分器”。

3. **优于运行时修正机制**：  
   训练时内化安全逻辑比推理时依赖外部工具链（如 self-correction）更有效。

4. **具备跨专科泛化能力**：  
   在老年科、儿科等高风险领域提升最大（+23.3 pp CSS）。

5. **鲁棒性强**：  
   即使在模拟真实 EHR 噪声下，仍保持 **84.2% CSS**。

---

### 局限性
1. **评估基于合成数据**：  
   所有结果均在 **CSB** 上取得，尚未在真实 EHR 或临床环境中验证，生态效度有限。

2. **知识图谱覆盖不足**：  
   约 **4.2% 错误源于 KG 覆盖缺口**，尤其在肿瘤学等快速演进领域。

3. **缺乏时间维度建模**：  
   当前模型无法处理随时间演变的风险（如累积肾毒性、洗脱期）。

4. **未包含匹配规模的 PPO/RLHF 基线**：  
   缺乏同等规模的 reward model + PPO 对比。

5. **计算成本较高**：  
   Iterative ORPO 总耗时约 **42.6 GPU-hours**（8×A100），约为单次 ORPO 的 4 倍。

---

### 未来工作方向
1. **外部验证**：  
   在去标识化的多中心 EHR 数据上进行盲评验证。

2. **动态知识更新机制**：
   - 增量更新 HGNN 权重（fine-tune 新增子图）
   - 定期重建 KG + warm-start 重新对齐
   - 自动监测知识漂移并触发更新

3. **混合架构（Hybrid Train+Verify）**：
   - 部署时加入轻量级置信门控
   - 仅对低置信输出启用 HGNN 或规则引擎二次验证

4. **模型压缩与高效变体**：
   - 对齐蒸馏（Alignment Distillation）至 1–3B 小模型
   - 量化部署（如 GPTQ/AWQ）

5. **引入时间推理能力**：  
   建模患者状态的时间轨迹，支持纵向风险预测。

6. **更公平的比较**：
   - 构建匹配规模的临床 reward model
   - 加入 tool-augmented agent 基线（检索 + 规则 + 自我修正）

--- 

> 📌 **总结**：本论文提出了 **Neurosymbolic Alignment** 框架，首次将 **HGNN-based Physiological World Model** 与 **Iterative ORPO** 结合，实现了训练时的生理安全对齐，在多项指标上显著超越现有方法，并通过独立验证缓解了评估偏倚。尽管仍受限于合成数据和静态知识，但为构建可信临床 LLM 提供了重要范式。

</details>

---

### 8. [Beyond Factual Knowledge: Benchmarking and Learning Step-Level Procedural Rule Reasoning in Large Language Models](https://arxiv.org/abs/2608.22753)

**Authors**: Bohan Yu, Pengfei Cao, Chen Han, Chenxi Zhou, Zhiheng Zhang, Zhiyang Xie, Wenhao Teng, Xiangwen Liao, Jun Zhao, Kang Liu  
**Category**: cs.CL  
**Published**: 2026-08-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.22753v1  

#### Abstract
Large language models (LLMs) excel at text understanding and generation, yet still struggle to reliably understand and apply externally provided procedural rules at scale. To evaluate this capability, we introduce RuleWorld, a large-scale benchmark that reformulates rules as globally reusable abstra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Large Language Models (LLMs)** 在文本理解和生成方面表现出色，但在可靠地理解并应用外部提供的**程序性规则 (procedural rules)** 方面仍存在显著挑战。现有基准（如 RuleTaker, ProofWriter）通常将规则作为特定问题的前提直接提供，这忽略了规则作为全局可复用知识的本质，从而无法有效评估模型在大规模规则池中进行**规则定位 (rule localization)** 和**多步推理 (multi-step reasoning)** 的能力。

### 提出的新方法与创新
为解决上述问题，本文提出了两大核心贡献：

#### (1) **RuleWorld**: 大规模程序性规则推理基准
- **性质**：一个包含 **494万条** 抽象、非常识、实体无关的程序性规则的大规模基准，涵盖一阶逻辑 (FOL) 和自然语言 (NL) 两种形式。
- **设计特点**：
  - 规则以**全局可复用的抽象单元**形式存在，而非实例化前提。
  - 包含三种推理任务类型，系统性地评估不同模式的规则应用能力：
    - **Single-Rule QA**：仅需应用一条规则。
    - **Parallel Multi-Rule QA**：多个独立子问题并行求解。
    - **Multi-Hop Rule QA**：需要规则链式推理，前序结论作为后续前提。
  - 支持高达 **10,000 条规则**的注入，全面测试模型在大规模规则池下的鲁棒性。

#### (2) **DynaRule**: 动态规则集成框架
- **核心思想**：将规则检索、更新和推理过程完全统一到模型内部，实现端到端的学习。
- **关键技术**：
  - **KV Cache 注入**：将外部规则编码后，通过单层适配器 (adapter) 注入到 LLM 的 KV 缓存中。
  - **动态规则重关注 (Dynamic Rule Re-attention)**：引入特殊 `<search>` token，在推理过程中触发对规则的重新检索和更新。
  - **堆叠的步骤级注意力训练 (Stacked Step-Level Attention Training)**：在训练时，通过监督模型在每个推理步骤上对相关规则的注意力分布，使检索过程成为可学习的内部机制。

### 相比现有方法的优势
- **超越传统 RAG**：避免了检索增强生成 (Retrieval-Augmented Generation, RAG) 对检索质量的敏感性和语义不匹配问题。
- **优于静态注入**：解决了全上下文注入 (Full-context injection) 在长规则列表下因 GPU 内存限制而不可扩展的问题。
- **优于现有 KV 注入方法**：相比 KBLaM 和 SR-KI，DynaRule 通过 `<search>` token 实现了**步骤级的动态规则选择与更新**，能更好地适应多步推理中目标规则的变化，从而在大规模规则池和复杂推理任务上表现更稳定。

---

## 2. 核心实验方法和设置

### 数据集
- **RuleWorld**：本文提出的核心数据集，包含 337 万条 QA 实例，用于训练和评估。
- **评估范围**：在 FOL 和 NL 两种规则表示形式下进行实验。

### 实验设置和评估指标
- **基础模型**：主要采用 `Qwen2.5-7B-Instruct` 进行详细分析，并在 `Qwen2.5-14B/32B/72B`, `Llama-3-8B/70B`, `deepseek-chat`, `gpt-5.1/5.5`, `claude-sonnet-4-6` 等多种模型上验证。
- **评估指标**：
  - **Exact Match Accuracy**：用于衡量 QA 任务的准确率。
  - **Recall@K**：用于衡量规则检索的准确性，其中 K 为每步所需的黄金规则数。
- **规则注入规模**：从 100 到 10,000 条规则不等，以测试模型的可扩展性。

### 基线方法对比
- **Prompting**：将所有规则直接拼接在提示词 (prompt) 中。
- **RAG**：使用稠密检索 (Dense, Qwen3-Embedding-8B)、稀疏检索 (BM25) 或混合检索 (Hybrid, RRF) 在生成前进行一次检索。
- **KBLaM**：一种先进的 KV 缓存知识注入方法。
- **SR-KI**：一种通过监督注意力进行检索的知识注入方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **在 10,000 条规则下**，DynaRule 的性能远超现有方法：
  - **平均 QA 准确率提升高达 19 分**。
  - **Recall@1 达到 85% 以上**，在 10K 规则下超过最强基线 60 余分。
- **在 1,000 条规则下**，DynaRule 的平均准确率达到 **84.82%** (FOL) 和 **90.42%** (NL)，而 SR-KI 仅为 37.12% 和 69.77%。

### 与基线方法的对比结果
| 方法 | 10K 规则平均准确率 | 10K 规则 Recall@1 |
| :--- | :--- | :--- |
| **DynaRule** | **~57.00%** | **>85%** |
| SR-KI | ~37.00% | ~24.15% (FOL) |
| RAG Hybrid | ~19.38% | ~13.38% (FOL) |
| Prompting | ~0% | - |

- **结论**：随着规则数量增加，所有基线方法性能急剧下降，而 DynaRule 表现出强大的鲁棒性。

### 消融实验结果
- **`<search>` token 的作用**：移除该 token 后，模型无法进行动态更新，导致多跳推理性能大幅下降。
- **信心层 (Confidence Layer) 的重要性**：在识别出的特定层（如 Qwen2.5-7B 的第 23 层）进行注意力监督是性能的关键，其他层的注入效果不佳。
- **泛化能力**：在未见过的规则上进行测试，DynaRule 仍能保持约 50% 的准确率，证明其学到的是通用的对齐能力，而非简单记忆。

---

## 4. 关键结论和发现

### 主要发现
1. **现有 LLMs 在大规模规则应用上面临严峻挑战**：当规则池增大时，无论是基于提示还是检索的方法，性能都会迅速恶化。
2. **步骤级动态集成至关重要**：多步推理要求模型能够根据中间状态动态调整所依赖的规则，DynaRule 通过 `<search>` token 实现了这一机制。
3. **内部可学习的检索优于外部检索**：将检索过程内化为模型的一部分，并通过端到端训练进行优化，可以实现更稳定、更准确的规则利用。

### 方法的局限性
- **嵌入表示的信息损失**：仅使用嵌入向量表示规则会丢失部分信息，可能导致检索或应用错误。
- **可扩展性上限**：虽然优于现有方法，但 LLMs 在处理万级以上的规则时仍会退化，全文档包含数百万条规则，当前方法尚未完全解决。
- **规则表示形式有限**：目前仅支持 FOL 和 NL 形式，未来可探索更丰富的表示（如可执行代码）。

### 未来工作方向
- 开发联合训练的规则编码器，减少信息损失。
- 探索分层或聚类索引策略，以支持更大规模的规则库。
- 设计更复杂的规则触发模式和更丰富的规则表示形式。
- 将 DynaRule 的思想应用于其他类型的外部知识（如数据库、API）的集成。

</details>

---

### 9. [Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization](https://arxiv.org/abs/2608.23311)

**Authors**: Xianlei Zhou, Xiangdi Meng, Yu He, Tianyu Qi, Shuyan Guan, Xianli Zhang, Jian Zhang, Xin Li, Qika Lin, Jun Liu  
**Category**: cs.CL  
**Published**: 2026-08-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.23311v2  

#### Abstract
Policy optimization (PO) for Large Language Models faces a stability--exploration trade-off, currently mediated by an action-side Policy-KL regularizer. This puts practitioners in a double bind: keeping Policy-KL constrains response behavior and consumes the action-side exploration budget, while dro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大语言模型（LLM）的策略优化（Policy Optimization, PO）中，存在一个**稳定性-探索性权衡困境（stability-exploration dilemma）**：

- 当前主流方法（如 PPO、GRPO）依赖于 **action-side Policy-KL 正则化** 来控制输出分布漂移，防止训练崩溃。
- 但这会限制模型对多样化响应的探索（exploration），导致“探索预算”被消耗。
- 若去掉该正则项，则输入端的查询分布（query distribution）会因策略更新而发生不受控的漂移，引发梯度方差增大和训练不稳定。

### 🆕 提出的新方法：**Environment-Regularized Policy Optimization (ERPO)**

ERPO 的核心思想是将正则化的对象从 **动作空间（action space）转移到输入环境（input environment）**，即：

- 引入 **Query-KL (QKL)** 正则项：约束当前策略诱导的 **query 分布 $p_\theta(q)$** 与预训练参考模型的 **reference query 分布 $p_{\theta_0}(q)$** 之间的 KL 散度。
- 同时引入 **基于参考模型的 per-query 重加权机制（query reweighting）**：在优势函数计算中，为每个 query 赋予一个静态权重 $w_B(q) \propto \log p_{\theta_0}(q)$，偏向于参考模型下更典型的 query。

### ✅ 相比现有方法的优势

| 维度 | 传统方法（如 GRPO） | ERPO |
|------|------------------|-------|
| **正则目标** | 动作侧 Policy-KL（限制输出行为） | 输入侧 Query-KL（限制 query 分布漂移） |
| **探索能力** | 受限（KL 惩罚直接作用于 $\pi_\theta(o\|q)$） | 保留（QKL 梯度不流经 response score function） |
| **训练稳定性** | 高温采样或长周期训练易崩溃 | 显著提升，尤其在高温和长期训练中 |
| **实现成本** | 无额外开销 | **零额外前向传播**，可插拔集成到 GRPO/PPO/REINFORCE 流程 |

> 🔑 **关键洞见**：通过解耦参数空间中的环境正则与策略优化，实现了“既稳定又自由探索”的训练范式。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 主要使用 **MATH 数据集** 中 Level 3–5 的数学推理题，共约 8.5K 示例。
- 在以下六个基准上进行评估：
  - **AIME2024**, **AIME2025**
  - **AMC**
  - **MATH500**
  - **Minerva**
  - **OlympiadBench**

### ⚙️ 实验设置
- **模型**：Qwen2.5-Math-7B 和 Qwen2.5-32B
- **框架**：EasyR1
- **训练步数**：240 步（主实验），扩展至 960 步用于长期稳定性分析
- **采样配置**：
  - Rollout 批大小：512
  - 更新批大小：128
  - 每个问题生成 8 或 16 个响应（$n=8/16$）
  - 推理温度范围：0.1 到 1.5（多温度评估）
- **KL 系数**：默认 $\alpha = 0.01$（公平比较）

### 📊 评估指标
- **Avg@K**：平均正确率（考虑多个采样）
- **Pass@1**：单次采样通过率
- **Pass@K**：K 次独立采样中至少有一次正确的概率
- 多温度聚合性能（0.1–1.5）以增强可比性和鲁棒性
- 额外监控：Query-KL、Policy-KL、Entropy、Train-Eval 一致性等动态指标

### 🔁 基线方法对比
- **Base**：SFT 模型
- **GRPO**：标准 group-relative policy optimization（带 Policy-KL）
- **ERPO**：本文提出的方法（替换为 Query-KL + query reweighting）
- 还测试了 ERPO 在其他算法上的泛化性：**DAPO**, **RLOO**

---

## 3. 主要实验结果和性能指标

### 📈 性能提升汇总（Table 1 & Figure 3）

| 方法 | Mean Avg@32 | Mean Pass@32 | Mean Pass@1 |
|------|-------------|---------------|--------------|
| GRPO | 0.274       | 0.575         | 0.275        |
| **ERPO** | **0.336** (+6.2%) | **0.611** (+3.64%) | **0.332** (+5.69%) |

> ✅ 在所有六项数学推理任务上，ERPO 均显著优于 GRPO。

#### 具体亮点：
- 在 **MATH500** 上，Avg@32 从 52.8% → **67.7%**（+14.9% 绝对增益）
- 在 **Qwen-32B** 上，Pass@1 达到 **84.6%**，远超 GRPO 的 81.6%
- 即使在高温度（1.5）下，ERPO 仍保持良好性能，而 GRPO 出现明显下降

### 🔍 消融实验结果（Table 2 & Figure 7）

| 方法 | Avg@32 ($T \leq 1.0$) | Avg@32 ($T > 1.2$) | 说明 |
|------|------------------------|--------------------|------|
| GRPO | 68.80 | 12.50 | 高温性能急剧下降 |
| GRPO + w(s) | 76.14 | 23.79 | query reweighting 提升稳定性 |
| GRPO + Query-KL | 80.90 | 38.00 | Query-KL 是性能飞跃主因 |
| ERPO ($\alpha=5\times10^{-2}$) | **79.00** | **43.35** | 更强正则进一步提效 |

> 💡 发现：
> - **Query-KL 是性能提升的主要驱动力**
> - **query reweighting** 有效降低梯度方差，提升高温鲁棒性
> - 增加 rollout 数量（$n=16$）可进一步提升性能至 **Pass@1 = 74.6%**

### 📉 训练稳定性分析（Figure 5 & 6）

| 指标 | GRPO | ERPO |
|------|------|------|
| Query-KL 漂移 | 快速上升（>0.9） | 被有效控制（~0.08） |
| 长期训练崩溃 | 明显（>400 步后性能骤降） | 极小退化，甚至高温下略有回升 |
| Entropy 变化 | 波动剧烈 | 更平稳 |

> 🛡️ ERPO 在长达 960 步的训练中表现出更强的抗崩溃能力。

### 🧪 Reward Hacking 分析（Table 4 & 5）

| 方法 | 平均 Train-Eval 差距（Gap） |
|------|----------------------------|
| GRPO | 6.47% |
| **ERPO** | **3.14%**（↓51%） |

> ✅ ERPO 显著缓解了 reward hacking 现象，提升了训练与推理的一致性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **输入端环境漂移是训练不稳定的重要根源**  
   即使固定训练集，策略更新也会导致模型自身对 query 的 likelihood 漂移，形成非平稳环境。

2. **Query-KL 正则化可有效控制环境漂移而不牺牲探索**  
   QKL 的梯度仅流经 query likelihood，**与 response policy 完全解耦**（Proposition 1），从而保留了动作空间的探索自由度。

3. **query reweighting 提升估计稳定性**  
   对 reference 下高概率 query 加权，减少了低质量样本带来的梯度噪声，尤其改善高温采样表现。

4. **ERPO 插件式兼容性强**  
   可无缝集成进 GRPO/PPO/REINFORCE 等框架，**无需额外 forward pass**，工程友好。

5. **在多温度、长周期、大规模场景下更具优势**  
   特别适合需要高探索性的复杂推理任务。

---

### ⚠️ 局限性

1. **适用范围有限**  
   实验集中在数学推理任务和 Qwen 系列模型，尚未验证其在对话、代码生成、多语言等任务上的迁移能力。

2. **依赖高质量 reference model**  
   Query-KL 和 reweighting 均基于 pre-RL reference model 的输出质量，若 reference 本身有偏，可能影响效果。

3. **未系统搜索最优 $\alpha$**  
   实验采用默认 KL 系数，更优超参可能带来更大收益。

4. **格式学习问题**  
   GRPO 在输出格式（如 `<think>` 标签）学习上表现较差，而 ERPO 也未能完全解决此问题。

---

### 🔮 未来工作方向

1. 将 ERPO 扩展至更多任务类型（code generation, dialogue, multilingual）
2. 结合 active querying 或 curriculum learning 动态调整 query 分布
3. 探索更高效的 query likelihood 估计方式（尤其对长序列）
4. 研究如何联合优化 query 与 action 正则，在极端稀疏奖励下进一步提升稳定性
5. 开发自动调优机制选择最佳 $\alpha$

---

> 🔗 **开源信息**：作者已公开代码仓库  
> GitHub: [https://github.com/AlibabaResearch/ERPO](https://github.com/AlibabaResearch/ERPO)

</details>

---

### 10. [FormuEvo: LLM-Guided Evolution for Discovering Solver-Efficient Mixed-Integer Programming Formulations](https://arxiv.org/abs/2608.23353)

**Authors**: Haofeng Yuan, Jianing Peng, Jieyi Bi, Ni Zhang, Shiji Song, Zhiguang Cao  
**Category**: cs.CL  
**Published**: 2026-08-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.23353v1  

#### Abstract
Mixed-integer programming (MIP) lies at the core of operations research and industrial optimization. While large language models (LLMs) have recently shown promise in automated MIP modeling from natural language, they prioritize semantic correctness but overlook formulation strength, severely bottle...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FormuEvo: LLM-Guided Evolution for Discovering Solver-Efficient Mixed-Integer Programming Formulations**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
- **问题背景**：Mixed-Integer Programming (MIP) 是运筹学和工业优化的核心工具，其求解效率高度依赖于建模方式。尽管一个优化问题可以有多种数学上等价的 MIP 表达形式，它们在计算效率上可能相差几个数量级。
- **现有挑战**：
  - 传统专家设计 MIP 模型需要深厚领域知识，且易受过时建模直觉影响（如某些“最佳实践”反而干扰现代 MIP 求解器的预处理）。
  - 大型语言模型（LLMs）虽能从自然语言自动生成 MIP 模型，但其目标是语义正确性和可执行性，而非**求解效率**，导致生成的模型通常结构简单、计算低效。

### **提出了什么新方法或新思路**
提出 **FormuEvo** —— 一种基于 LLM 的进化框架，用于自动发现**求解器高效**（solver-efficient）的 MIP 模型。

#### **核心思想**
将 MIP 模型设计视为在**符号空间**上的进化优化过程，通过迭代的 LLM 驱动操作（交叉、变异、修复），结合求解器反馈不断演化出更高效的模型。

#### **两大关键机制**
1. **Solver-Informed Diagnosis（求解器感知诊断）**
   - 利用求解器运行时的细粒度统计信息（如 `root gap`, `B&B node count`, `presolve` 效果等）作为“口头梯度”（verbal gradients）。
   - 由一个诊断 LLM 分析这些指标，识别瓶颈（如松弛弱、分支树过大），并指导生成 LLM 进行有针对性的改进。

2. **Structured Memory（结构化记忆）**
   - 将每次进化中的成功/失败经验抽象为 `(Condition → Strategy → Effect)` 三元组存入记忆库。
   - 支持跨问题的知识迁移和零样本泛化，并可用于引导后续搜索，避免重复探索无效路径。

### **相比现有方法的优势**
| 维度 | 传统专家 | LLM Fine-tuning (如 ORLM) | FormuEvo |
|------|----------|----------------------------|-----------|
| **目标** | 正确性 + 经验直觉 | 语义正确性 | **求解效率最大化** |
| **搜索方式** | 手工设计 | 单次生成 | **迭代进化 + 反馈驱动** |
| **知识复用** | 隐式积累 | 无 | 显式结构化记忆 |
| **适应性** | 固定 | 固定 | 动态适配求解器内部机制 |
| **泛化能力** | 强但慢 | 弱（局限于训练分布） | 强（支持跨问题迁移） |

---

## 2. **核心实验方法和设置**

### **使用了哪些数据集**
涵盖经典与前沿的 MIP 问题，分为两类：

#### **经典 MILP/MINLP 基准**
- **TSP**（旅行商问题）
- **JSSP**（作业车间调度）
- **BPP**（装箱问题）
- **CFLP**（带容量限制的设施选址）
- **QAP**（二次分配问题）

#### **新兴复杂问题（人类先验少）**
- **NNV**（神经网络验证）
- **IMO 2025 Problem 6**（国际数学奥林匹克题，网格覆盖问题）

每个问题按规模划分为 Easy / Medium / Hard 三个难度等级，其中 Easy 用于进化训练，Medium/Hard 用于最终测试。

---

### **实验设置和评估指标**

#### **实现细节**
- **下游求解器**：Gurobi 10.0（单线程，默认参数）
- **LLM 主干**：GPT-5.4-mini（成本低，效果稳定）
- **种群大小**：N = 8
- **进化代数**：T = 5
- **每代生成**：8 个后代（交叉 + 变异）
- **评估实例数**：100 个 Easy 实例用于 fitness 评估
- **Fitness 函数**：Shifted Geometric Mean (SGM) 运行时间（shift=1秒）

#### **评估指标**
1. **Time**：SGM 运行时间（越小越好）
2. **Wins**：在多少实例上取得最快求解时间
3. **Solved**：在时间限制内成功求解的实例数（600秒）

---

### **基线方法对比**

| 类别 | 基线方法 | 说明 |
|------|---------|------|
| **专家设计** | MTZ, SCF, MCF-RLT (TSP); Disj., VPSolver (BPP) 等 | 包括教科书模型与当前最优人工模型 |
| **LLM 生成** | ORLM, StepORLM | 基于微调的端到端 MIP 生成模型 |
| **LLM 进化相关** | EvoCut | 基于 LLM 的割平面增强方法，仅局部加强固定模型 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（来自 Table 1 & 2）**

#### **经典问题（Hard 实例）**
| 方法 | TSP (Time) | JSSP (Time) | BPP (Time) | CFLP (Time) | QAP (Time) |
|------|------------|-------------|------------|--------------|------------|
| 最佳基线 | 8.3315 | 17.6653 | 1.4425 | 59.4309 | 34.5034 |
| **FormuEvo** | **3.9469** | **15.4237** | **0.8384** | **44.3447** | **34.5034** |
| **加速比** | **~2.1×** | ~1.1× | **~1.7×** | **~1.3×** | ≈1.0× |

> 注：QAP 上持平是因为标准模型已是凸包表示（Birkhoff polytope），无法再通过割平面提升。

#### **新兴问题（Hard 实例）**
| 方法 | NNV (Time) | IMO (Time) |
|------|-----------|-----------|
| 标准模型 | 69.5200 | 114.0912 |
| **FormuEvo** | **21.4139** | **17.9341** |
| **加速比** | **~3.2×** | **~6.4×** |

> 在 NNV 和 IMO 上，ORLM/StepORLM 完全失败（未生成有效模型）。

#### **综合表现**
- **最大加速比高达 5.5×**（见摘要）
- 在绝大多数问题和实例上取得最少运行时间和最多“Wins”
- 成功求解率普遍达到 100%（Hard 实例）

---

### **消融实验结果（Table 3）**

验证两个核心组件的重要性（以 TSP 为例）：

| 方法 | Easy Time | Medium Time | Hard Time |
|------|-----------|-------------|-----------|
| FormuEvo (完整) | **0.1410** | **0.5640** | **3.9469** |
| w/o Memory | 0.1776 (+26%) | 0.6102 (+8.2%) | 4.6649 (+18.2%) |
| w/o Diagnosis | 0.2308 (+63.7%) | 0.7350 (+30.3%) | 6.5207 (+65.2%) |

> 结论：**Diagnosis 贡献更大**，提供方向性信号；Memory 提升搜索效率。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **理论紧致 ≠ 实际高效**  
   如 TSP 中 MCF-RLT 拥有最紧松弛界，但在 Hard 实例上完全失败（0/100 solved），因其扩展变量过多导致计算负担过重。FormuEvo 更关注实际求解行为。

2. ✅ **LLM 微调方法存在根本局限**  
   ORLM/StepORLM 输出多为“教科书式”模型，缺乏结构性创新，在新问题上甚至无法生成有效模型。

3. ✅ **EvoCut 等局部增强方法受限于初始模型结构**  
   其本质是在固定模型上加 cutting planes，无法进行全局重构（如变量扩展、线性化方式改变）。而 FormuEvo 可探索整个符号空间。

4. ✅ **知识可迁移性强**  
   通过 distiller LLM 提炼通用建模策略后，可在新问题上实现**零样本迁移**，并显著提升小型 LLM（如 GPT-5.4-nano）的表现（见 Figure 4）。

5. ✅ **框架对 LLM 主干不敏感**  
   在不同 LLM（GPT, Claude, DeepSeek）上均能超越基线，表明性能增益主要来自**进化框架本身**而非特定模型能力（Table 4）。

---

### **方法的局限性**
- **静态模型假设**：FormuEvo 当前仅适用于可被通用 MIP 求解器直接求解的**静态模型**。
- **不支持动态算法耦合**：对于需列生成（column generation）、Benders 分解等问题，模型设计与求解算法紧密耦合，当前框架尚不能联合演化。
- **依赖高质量 LLM 推理能力**：虽然对主干不敏感，但仍要求 LLM 具备基本的 MIP 建模和代码理解能力。

---

### **未来工作方向**
- 扩展至**动态分解算法**的设计，联合演化模型与求解策略。
- 探索在**强化学习**或**组合优化启发式**中应用类似进化范式。
- 构建开源的 MIP 模型进化平台，支持社区协作与知识共享。
- 将结构化记忆应用于其他程序合成任务（如 SAT/SMT 编码、CVXPY 建模等）。

---

> 🔗 **代码开源**：[https://github.com/Xyz-yuanhf/formuevo](https://github.com/Xyz-yuanhf/formuevo)  
> 📄 **论文链接**：arXiv:2608.23353

</details>

---

### 11. [Scalable datacenter replication with mostly-synchronous consensus on hardware](https://arxiv.org/abs/2608.24622)

**Authors**: Davide Rovelli, Philipp Berdesinski, Rodrigo Otoni, Patrick Eugster  
**Category**: cs.DC  
**Published**: 2026-08-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.24622v1  

#### Abstract
Consistent replication of data among distributed processes -- a task involving the well-known consensus problem -- is notoriously expensive and hard to scale, affecting especially datacenter services with stringent performance requirements. To mitigate this problem, we introduce scalable replication...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Scalable datacenter replication with mostly-synchronous consensus on hardware》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

传统基于 **partially synchronous** 模型的分布式共识协议（如 Paxos、Raft）在数据中心环境中面临三大瓶颈：

- **C1. 单点瓶颈（Single-process bottleneck）**：领导者需处理所有客户端请求并协调多数派确认，成为性能瓶颈。
- **C2. 可扩展性差（Low scalability）**：随着副本数量增加，系统性能迅速下降，因此通常限制副本数为 3–7 个，牺牲容错能力。
- **C3. 服务中断（Service downtime）**：在领导者选举或集群重配置期间，系统无法处理请求。

此外，尽管硬件加速（如 FPGA、RDMA）被用于提升性能，但大多数方案仍受限于底层异步模型的设计约束。

---

### 提出了什么新方法或新思路

本文提出 **scarHW**（scalable replication in-hardware），一种基于 FPGA smartNIC 的新型网络卡设计，其核心是 **POPUC**（Popular Uniform Consensus）算法，实现了一种名为 **collaborative consensus**（协同共识）的新型共识范式。

#### 主要创新点：

- **全新的系统模型**：采用“**mostly-synchronous**”模型，假设现代可编程数据中心网络具有稳定低延迟（△r）和时钟漂移（△p），同时容忍 **crash-stop** 和 **message omission** 故障（如丢包或超时）。
- **POPUC 共识算法**：
  - **无领导者（leaderless）**：所有副本均可提议值，天然支持负载均衡。
  - **协同共识（co-consensus）**：允许单轮中多个提案被同时决定，而非经典共识中的单一决策。
  - **形式化验证**：通过 TLA+ 和 APALACHE 模型检测器完成安全性与活性的形式化证明。
- **硬件卸载（Hardware offload）**：将共识逻辑完全卸载至 FPGA smartNIC，利用其确定性执行和低延迟特性，实现线速（wire-speed）复制。

---

### 相比现有方法的优势

| 特性 | 传统方法（如 Raft、NOPaxos） | scarHW |
|------|-------------------------------|--------|
| 领导者 | 有，存在 C1 瓶颈 | 无，天然负载均衡 |
| 扩展性 | 副本增多导致性能下降 | 增加副本可提升容错且不降性能 |
| 宕机时间 | 领导者故障时需选举，数百毫秒中断 | 几微秒额外轮次，近乎零宕机 |
| 一致性保证 | 强一致但依赖领导者 | 强一致，全副本同步更新 |
| 吞吐量 | 受限于软件栈和领导节点 | 达到 100Gbps 网络饱和，吞吐提升达百倍 |

---

## 2. 核心实验方法和设置

### 实验设置

- **硬件平台**：
  - 使用 **AMD/Xilinx Alveo U50** FPGA smartNIC。
  - 服务器：3× Supermicro SYS-120U-TNR，双 Xeon Gold 5315Y，190GiB RAM。
  - 网络：100Gbps DAC 连接至 Tofino 交换机。
  - 对比系统也运行在同一集群上（使用 ConnectX-7 NIC）。

- **部署模式**：
  - **scarHW-L**（Leader）：客户端统一发往一个节点。
  - **scarHW-B**（Balanced）：使用 IPVS 负载均衡器分发请求。
  - **scarHW-u**（Microservice）：客户端与 scarHW 实例共置，适用于 sidecar 架构。

---

### 评估指标

- **吞吐量（Throughput）**：每秒处理的请求数（requests/s）。
- **延迟（Latency）**：端到端响应时间，特别是尾部延迟（99.99th percentile）。
- **Goodput**：有效数据传输速率，排除协议开销。
- **故障恢复时间**：发生故障后的服务中断时长。
- **可扩展性**：随副本数增加的性能变化趋势。

---

### 基线方法对比

| 方法 | 类型 | 关键技术 |
|------|------|----------|
| **Waverunner** | FPGA 加速 Raft | 仅加速正常路径，故障切换仍由软件处理 |
| **Mu** | RDMA-based Paxos | 利用 RDMA 写实现低延迟 |
| **P4ce** | RDMA + P4 开关 | 在交换机中聚合消息减轻领导者负担 |
| **NOPaxos** | 网络排序 | 将序列化卸载到可编程交换机 |
| **RedisRaft / Zookeeper** | 应用层 SMR | 基于 Raft/ZAB 的实际服务 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 微基准测试（RQ1：峰值性能）

- **最大吞吐量**：
  - scarHW 在 5 副本下达到 **>10M req/s**，接近 100Gbps 网络上限。
  - Waverunner 因单 NIC 输出带宽受限而略低。
- **最低延迟**：
  - scarHW 最坏情况延迟为 **5.4μs**（含安全裕量）。
  - 实际测量中，**无任何数据包超过 4.41μs**，远低于软件方案平均延迟（3–10μs）。

> ✅ **结论**：scarHW 实现了 **us 级延迟 + 线速吞吐**，优于所有基线。

#### 应用集成测试（RQ2：真实场景性能）

在 **Redis** 和 **ZooKeeper** 上集成 scarHW 后的表现：

| 配置 | 吞吐量提升 | 延迟降低 |
|------|------------|---------|
| scarHW-B vs RedisRaft | **高达 100×** | 显著降低 |
| scarHW-u vs RedisRaft | **>100×**（GET/SET 均显著改善） | 尾部延迟下降两个数量级 |

- 当副本数增至 **59** 时：
  - 传统系统（RedisRaft、ZooKeeper）因领导者瓶颈性能急剧下降。
  - scarHW-B 和 scarHW-u **维持甚至提升性能**，体现卓越可扩展性。

#### 故障影响测试（RQ3：容错性）

- 注入一个副本崩溃后：
  - **ZooKeeper / Waverunner**：经历 **数百毫秒中断**（等待领导者选举）。
  - **scarHW**：仅多花 **一个共识轮次（2.7μs）**，吞吐轻微下降但无中断。
- 恢复过程：
  - scarHW 将恢复视为成员变更，通过 POPUC 协议平滑完成。
  - 支持 **t = ⌊(N−1)/2⌋** 故障容忍，超过则系统暂停（安全阻塞）。

---

### 消融实验结果（Ablation Study）

- **硬件加速 vs 软件实现**：
  - 若仅保留 POPUC 算法但在 CPU 上运行，性能仍受软件栈限制。
  - FPGA 卸载使协议核心处理延迟从 μs 级降至 ns 级。
- **负载均衡的作用**：
  - 在 scarHW-L 中，单个应用仍需处理全部请求 → 成为新瓶颈。
  - 在 scarHW-B/u 中，每个副本仅处理约 `L/N` 请求 → 应用层压力大幅缓解。

> ✅ **结论**：**硬件加速 + 负载均衡** 是 scarHW 高性能的关键组合。

---

## 4. 关键结论和发现

### 主要发现

1. **现代数据中心更适合“mostly-synchronous”模型**：
   - FPGA smartNIC 和低延迟网络使得交互延迟高度可预测。
   - 可以安全地假设同步为主，仅将异常延迟/丢包作为“omission failure”处理。

2. **协同共识（co-consensus）打破传统性能天花板**：
   - 无需领导者即可实现强一致性。
   - 多提案并发决定机制天然支持高吞吐与负载均衡。

3. **硬件卸载不是简单优化，而是范式转变**：
   - 不只是“更快地执行旧协议”，而是重新设计协议以充分利用硬件特性。
   - scarHW 实现了 **zero downtime upon minority failure** 和 **linear scalability**。

4. **副本不再是负担，而是资源**：
   - 传统观点：更多副本 → 更慢。
   - scarHW 观点：更多副本 → 更高容错 + 更好负载分担 → 性能不变甚至提升。

---

### 方法的局限性

- **依赖专用硬件**：需要 FPGA smartNIC（如 Alveo U50），目前尚未普及。
- **FPGA 资源限制**：
  - 当前实现最多支持约 **140 个副本**（受限于 MTU 和 packet packing）。
  - 并行 consensus engine 数量受限于 FPGA 面积（当前最多 31 个）。
- **初始同步需求**：需通过 SYNC-START 或 PTP 实现初始对齐。
- **不适用于广域网（WAN）**：依赖局域网内低延迟和高可靠性。

---

### 未来工作方向

- **更大规模 FPGA 支持**：适配 Alveo U280 等更高资源设备，支持更大集群。
- **动态 reconfiguration**：自动扩缩容、故障节点替换。
- **与 sharding 结合**：在超大规模下结合分区策略进一步提升吞吐。
- **支持 BFT 场景**：扩展 POPUC 以容忍拜占庭故障。
- **更智能的流量调度**：结合 P4、TE 实现拥塞感知的共识调度。
- **开源生态建设**：发布完整 FPGA 源码与 SDK，推动社区 adoption。

---

## 总结

> 🔷 **scarHW 代表了一种从“软件为中心”向“硬件协同设计”的根本性转变**。它不仅提升了性能两个数量级，更重要的是改变了我们对“复制即代价”的认知——在现代数据中心中，**复制可以成为性能增强工具**。

</details>

---

### 12. [PuzzleKV: Page-Wise Low-Rank Decomposition for KV Cache Compression](https://arxiv.org/abs/2608.23843)

**Authors**: Zizhong Wang, Jieying Wang, Zhao Zhang, Jiajia Li  
**Category**: cs.LG  
**Published**: 2026-08-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.23843v1  

#### Abstract
Long-context inference in large language models (LLMs) is increasingly limited by the memory required for the key-value (KV) cache. KV cache compression addresses this problem by reducing the storage cost of previous tokens. Among existing approaches, low-rank compression is particularly attractive ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PuzzleKV: Page-Wise Low-Rank Decomposition for KV Cache Compression**

---

## **1. 主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）在长上下文推理时面临 **KV Cache** 内存占用过大的问题。传统的 KV Cache 存储方式随上下文长度线性增长，严重制约了长序列生成的效率。现有的低秩压缩方法通常采用全局共享的压缩空间（如 Global SVD），忽略了局部 token 序列中的细粒度低秩结构，导致重要但稀疏的信息丢失。

### **提出的新方法**
本文提出了 **PuzzleKV**，一种无需训练和校准的 **page-wise 低秩 KV Cache 压缩方法**，其核心思想是：
- 将每个 **per-head KV Cache** 划分为固定长度的逻辑“页”（logical page）；
- 在每一页内部独立进行 **低秩分解**（truncated SVD），保留每页的局部低秩结构；
- 保持 sink 区域和最近窗口为 dense 形式，其余历史页以低秩因子形式存储；
- 在注意力计算中直接对 dense 和 factorized 页面进行混合 attention，避免重建历史缓存；
- 在自回归解码过程中增量地将新完成的页转换为低秩表示。

### **相比现有方法的优势**
| 维度 | PuzzleKV | 其他方法（如 Global SVD、H2O） |
|------|---------|-------------------------------|
| **压缩粒度** | 页级（fine-grained） | 全局或序列级（coarse-grained） |
| **信息保留** | 保留所有 token（无丢弃） | H2O 等会永久删除部分 token |
| **是否需要校准** | ❌ 无需训练或校准 | Palu、EigenAttention 需要校准数据 |
| **GPU 友好性** | 支持批量页分解，适合 GPU 并行 | 单独 SVD 开销大 |
| **灵活性** | 可与量化结合实现更高压缩比 | 多数方法不支持组合优化 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **RULER**：合成控制任务，用于评估在不同能力维度（如检索、多跳推理）下的压缩鲁棒性，输入长度为 16K 和 32K。
- **LongBench**：真实世界长文本任务基准，涵盖单文档问答、多文档摘要、代码生成等，评估实际场景性能。

### **实验设置**
- **模型**：
  - `Qwen3-8B`
  - `Llama-3.1-8B-Instruct`
- **上下文长度**：16K 和 32K
- **KV Cache 存储预算**：约原始大小的 **60%**（即压缩比 ~0.6）
- **PuzzleKV 参数**：
  - 页大小 $ P = 32 $
  - 低秩参数 $ (r_K, r_V) = (16, 14) $
  - Head dimension $ d = 128 $
- **评估指标**：
  - 准确率（Accuracy %）
  - 相对于 Full KV 的性能百分比
  - 存储比率（Storage Ratio）
  - 解码延迟（TTFT, TPOT）
  - 显存占用

### **基线方法对比**
| 方法 | 类型 | 是否需校准 | 特点 |
|------|------|------------|------|
| **Full KV** | 原始完整缓存 | ❌ | 性能上限 |
| **Global SVD** | 全局低秩压缩 | ❌ | 单一共享基底，忽略局部结构 |
| **Palu (G-LRD)** | 权重侧低秩压缩 | ✅ | 基于模型权重分解 |
| **H2O** | Token 蒸馏 | ❌ | 保留高频和近期 token，永久删除其他 |
| **OjaKV** | 在线更新基底 | ✅ | 动态调整，但不稳定 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **在 RULER 上的表现（16K / 32K）**
| 方法 | Llama3.1 @16K | Qwen3 @16K | Llama3.1 @32K | Qwen3 @32K |
|------|----------------|-------------|----------------|-------------|
| Full KV | 92.33 | 91.41 | 87.54 | 89.44 |
| **PuzzleKV** | **88.82 (96.2%)** | **90.73 (99.3%)** | **84.39 (96.4%)** | **87.53 (97.9%)** |
| Global SVD | 62.61 | 76.51 | 59.38 | 71.89 |
| H2O | 68.76 | 64.00 | 61.75 | 62.95 |
| Palu | 87.26 | 79.02 | 82.68 | 71.03 |

> ✅ PuzzleKV 在所有设置下均显著优于 Global SVD 和 H2O，在 Qwen3 上接近 Full KV 性能。

#### **在 LongBench 上的表现**
| 方法 | Llama3.1 | Qwen3 |
|------|----------|--------|
| Full KV | 45.83 | 29.86 |
| **PuzzleKV** | **45.33 (98.9%)** | **29.48 (98.7%)** |
| Global SVD | 44.26 | 28.49 |
| H2O | 45.61 | 29.10 |
| Palu | 42.21 | 25.02 |

> ✅ PuzzleKV 在真实任务上表现优异，仅次于 H2O 或持平，远超其他低秩方法。

---

### **与基线方法的对比结果**
- **vs Global SVD**：
  - 在 RULER 上平均提升 **25+ pts**，尤其在 NIAH-S1、Variable Tracking 等稀疏信息任务上优势巨大（从 0 → 98+）；
  - 表明 page-wise 基底能更好捕捉局部关键信息。
- **vs H2O**：
  - 在 UUID 恢复任务（NIAH-S3）上大幅领先（82–83 vs 33–31），因为 PuzzleKV 不丢弃任何 token；
  - H2O 因永久删除 token 导致信息不可恢复。
- **vs Palu**：
  - 无需校准且性能更优，尤其在 Qwen3 上领先超过 10 pts。

---

### **消融实验结果**

#### **(1) Rank 分配影响**
- 最优配置为 $ (r_K, r_V) = (16,14) $，略偏向 keys；
- 增加 $ r_K $ 比增加 $ r_V $ 对性能提升更明显；
- 总秩达到 30 后收益递减。

#### **(2) 页大小（Page Size）选择**
| Page Size | Accuracy (RULER) | 分解延迟 |
|-----------|------------------|----------|
| 16 | 87.81 | 0.11s |
| **32** | **90.27** | **0.17s** |
| 64 | 91.74 | 0.27s |

> 虽然 P=64 精度略高，但延迟显著上升，因此选择 **P=32** 作为平衡点。

#### **(3) RoPE Key 位置的影响**
- 使用 **post-RoPE keys** 进行分解比 pre-RoPE 略差（90.40 vs 90.88），但差异小；
- 选择 post-RoPE 可避免访问时重复 RoPE 计算，提升效率。

#### **(4) 与量化的结合**
| 方法 | 存储比 | Llama3.1 | Qwen3 |
|------|--------|-----------|--------|
| Full KV | 1.000 | 92.33 | 91.41 |
| Full KV + INT4 | 0.283 | 92.34 | 91.09 |
| **PuzzleKV + INT4** | **0.187** | **86.24 (93.4%)** | **90.20 (98.6%)** |

> ✅ PuzzleKV 可进一步与 **per-factor INT4 量化** 结合，在仅 **18.7% 原始存储** 下仍保留 >93% Full KV 性能。

---

## **4. 关键结论和发现**

### **主要发现**
1. **KV Cache 中存在显著的页级低秩结构**：实验证明，每个固定长度的 KV page 内部具有强低秩特性，适合独立压缩。
2. **page-wise 压缩优于全局压缩**：相比单一共享基底，分页独立建模能更好地保留稀疏但关键的信息（如唯一 key、中间变量链）。
3. **无需校准即可实现高性能压缩**：PuzzleKV 完全基于运行时数据动态分解，无需离线处理或校准集。
4. **可扩展性强**：支持与量化组合，实现高达 **81.3% 存储节省**（仅用 18.7%）而性能损失极小。

### **方法的局限性**
- **预填充阶段有额外开销**：首次 SVD 分解带来约 **20.9% TTFT 延迟增加**；
- **峰值内存短暂升高**：在页转换期间临时增加约 12.45 MiB 显存；
- 当前实现为 batch-1 原型，尚未集成到 vLLM 等生产级系统中。

### **未来工作方向**
- 将 PuzzleKV 集成至 **batched serving engine**（如 vLLM），利用页抽象实现高效并行；
- 探索 **adaptive page size** 或 **dynamic rank allocation** 机制；
- 扩展至 **cross-layer redundancy exploitation**，与 xKV 等方法结合；
- 研究 **非 SVD 的快速低秩近似算法** 以进一步降低延迟。

---

> 🔚 **总结一句话**：  
> **PuzzleKV 通过将 KV Cache 按页切分并独立低秩压缩，在无需训练或校准的前提下，实现了高保真、高压缩比的 KV 缓存管理，是迈向高效长上下文推理的重要一步。**

</details>

---

### 13. [WarpSAC: Towards the Pinnacle of Scalable Off-policy RL by Rethinking Exploration and Exploitation](https://arxiv.org/abs/2608.24479)

**Authors**: Zihao Wu, Hongyao Tang, Yi Ma, Huizhong Song, Pengyi Li, Yifu Yuan, Fei Ni, Jinyi Liu, Wei Wei, Jianrong Wang, Yan Zheng, Jianye Hao  
**Category**: cs.LG  
**Published**: 2026-08-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.24479v1  

#### Abstract
Massively parallel simulation changes the data regime in which off-policy reinforcement learning (RL) is trained, challenging stabilizers designed for data-limited replay. Through controlled experiments across eight benchmark families, we show that these stabilizers are data-regime-dependent: parame...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：WarpSAC: Towards the Pinnacle of Scalable Off-policy RL by Rethinking Exploration and Exploitation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 **off-policy RL** 算法（如 SAC、FlashSAC）中的稳定机制（stabilizers），例如 **parameter normalization**、**clipped double-Q** 和 **uniform replay**，最初是为 **data-limited**（数据稀疏）环境设计的。这些机制通过保守估计和函数类约束来防止在覆盖不足的状态-动作空间上发生外推错误。

然而，随着 **GPU-accelerated simulators** 和 **massively parallel simulation** 的发展，现代训练可以产生海量且多样化的经验数据，进入 **data-abundant** 数据范式。在这种新环境下，传统稳定器可能不再必要，甚至会限制价值函数的表达能力和学习效率。

本文提出的核心问题是：  
> “这些经典稳定器是否仍然有益？” 更进一步地，“它们的效用是否依赖于当前的数据规模和覆盖范围？”

### 提出了什么新方法或新思路
作者提出了 **WarpSAC**，一个**数据范式感知**（regime-aware）的 off-policy RL 算法家族，其核心思想是：

- **将稳定器的选择与数据范式匹配**，而非统一继承。
- 引入 **Sample Weight Decay (SWD)** 作为跨范式的通用组件，用于提升数据利用效率。
- 针对不同数据规模，设计两种变体：
  - **WarpSAC-L**（Limited-data regime）：适用于 CPU-scale、数据有限场景，保留 **Norm ON** 和 **clipped double-Q**。
  - **WarpSAC-A**（Abundant-data regime）：适用于 GPU-parallel、数据丰富场景，采用 **Norm OFF** 和 **single-Q**，以释放模型表达力并减少计算开销。

### 相比现有方法的优势
- **性能更强**：在多种基准上显著超越 FlashSAC。
- **更高效**：在数据丰富场景下，通过移除冗余稳定器（如第二个 critic），降低了计算成本。
- **更具适应性**：提供了一套“实践指南”，指导用户根据训练环境选择合适的算法配置。
- **无需额外网络或复杂结构**：改进完全基于对现有机制的重新组合与裁剪。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖 **8 个 benchmark families**，共 67 个任务，分为两类数据范式：

#### 数据受限（CPU-scale, 9 environments）
- **Gym-MuJoCo**：标准连续控制任务（如 HalfCheetah-v4）
- **DeepMind Control Suite (DMC) hard tasks**：高维 humanoid 和 dog 控制
- **HumanoidBench**：Unitree H1 全身控制
- **MyoSuite**：肌骨模型灵巧操作

#### 数据丰富（GPU-parallel, 14 environments）
- **MuJoCo Playground**：GPU 并行人形机器人控制
- **IsaacLab**：多模态机器人学习框架
- **MJLab**：基于 MuJoCo-Warp 的 GPU 加速机器人学习
- **ManiSkill**：大规模刚体操作任务（稀疏奖励）

此外还包括 **sim-to-real** 实验：**Unitree G1** 在真实世界中的平地行走任务。

---

### 实验设置和评估指标

| 设置项 | 描述 |
|--------|------|
| **训练 backbone** | 基于 FlashSAC，固定优化器、网络架构、环境接口等 |
| **对比变量** | 仅改变三个轴：<br>(A) Replay weighting (SWD vs uniform)<br>(B) Parameter projection normalization (ON/OFF)<br>(C) Critic multiplicity (double-Q vs single-Q) |
| **评估指标** | - **Normalized score-step AUC**：衡量样本效率<br>- **Mean normalized wall-time AUC**：衡量实际训练时间效率<br>- **Success rate**：针对稀疏奖励任务（如 ManiSkill）<br>- **Wall-clock time to convergence**：sim-to-real 部署速度 |

---

### 基线方法对比
- **FlashSAC**：主基线，使用 uniform replay、Norm ON、clipped double-Q。
- **WarpSAC-L**：SWD + Norm ON + double-Q → 适配 data-limited。
- **WarpSAC-A**：SWD + Norm OFF + single-Q → 适配 data-abundant。
- **WarpSAC w/ Norm OFF**：消融版本，用于分析 normalization 影响。

所有方法共享相同训练流程，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 场景 | 指标 | WarpSAC 表现 | 对比基线 |
|------|------|-------------|----------|
| **CPU-scale (9 envs)** | Normalized score-step AUC | **+4.5%** 提升 | vs FlashSAC |
| **GPU-parallel (14 envs)** | Normalized score-step AUC | **+23.1%** 提升 | vs FlashSAC |
| **MuJoCo Playground** | Mean normalized wall-time AUC | **+19.1%** 提升 | vs FlashSAC |
| **UnitreeG1TransportBox-v1** | Success Rate | **从 19.8% → 96.4%** | 显著提升 |
| **Sim-to-real on Unitree G1** | Wall-clock time to deployment | **35 分钟**（单 A800 GPU） | FlashSAC 需 55 分钟，**快 36.4%** |

> ✅ 所有增益均未增加网络参数或辅助模块，部分情况下反而**移除了组件**。

---

### 与基线方法的对比结果

- **WarpSAC-L** 在 CPU-scale 环境中表现最优，尤其在 HumanoidBench 和 MyoSuite 上学习更稳定、最终性能更高。
- **WarpSAC-A** 在 GPU-parallel 环境中全面领先：
  - 在 IsaacLab、MJLab 中，**Norm OFF** 变体优于 Norm ON。
  - 在 ManiSkill 上，**single-Q + Norm OFF** 达到最强性能，表明 clipped double-Q 可安全移除。
- **SWD** 在所有环境中都带来正向收益，无论数据多少。

---

### 消融实验结果

#### （1）SWD × 网络容量（CPU-scale）
- 在低容量网络（1 residual block）下，SWD 带来的相对增益最大（如 humanoid-run 上提升超 2×）。
- 随着网络容量增大，SWD 优势缩小但仍存在，说明其作用是**补偿容量不足**，而非替代大模型。

#### （2）Normalization × 网络容量（GPU-parallel）
- 在低容量设置下，**禁用 normalization** 显著提升性能（如 G1 Flat 上）。
- 即使在网络较大时，Norm OFF 仍保持竞争力，表明在数据充足时，**normalization 的稳定性收益被其对表达力的限制所抵消**。

#### （3）Critic Multiplicity
- 在高吞吐操作任务（如 ManiSkill）中，**single-Q** 与 double-Q 性能相当甚至更好，证明 **clipped double-Q 的悲观偏置在数据丰富时可能是冗余的**。

---

## 4. 关键结论和发现

### 论文的主要发现

1. ✅ **稳定器具有数据范式依赖性**：
   - **Parameter normalization** 和 **clipped double-Q** 在数据稀疏时有益，但在数据丰富时可能成为瓶颈。
   - **SWD** 是唯一跨范式始终有效的机制，应作为通用组件保留。

2. ✅ **scalable RL 应该是 regime-matching 而非 stabilizer-stacking**：
   - 不再盲目堆叠稳定机制，而是根据数据覆盖情况动态选择最合适的组合。

3. ✅ **简化有时更强**：
   - WarpSAC-A 通过**移除 normalization 和一个 critic**，实现了更高的性能和更快的训练速度，挑战了“更多机制=更稳定”的传统观念。

4. ✅ **sim-to-real 部署加速**：
   - 在 Unitree G1 上实现 **35 分钟端到端训练部署闭环**，相比 FlashSAC 快 36.4%，展示了其在真实机器人应用中的潜力。

---

### 方法的局限性

- **离线切换**：当前 WarpSAC-L 和 WarpSAC-A 是**预先选定**的，无法在训练过程中动态适应数据分布的变化（例如先预训练后并行微调）。
- **仅基于 FlashSAC 架构**：分析局限于 FlashSAC 的设计空间，其他 stabilizers（如 entropy weighting、target network delay）未被系统研究。
- **假设理想并行环境**：高度依赖大规模 GPU 并行模拟，在资源受限场景下难以复现全部优势。

---

### 未来工作方向

1. **在线自适应机制**：
   - 设计能够监控 replay coverage 或 value extrapolation 信号，并**动态调整 normalization 强度和 critic 数量**的算法。

2. **扩展 stabilizer 分析**：
   - 将 regime-aware 思想推广至其他常见机制，如：
     - Entropy coefficient 调整
     - Target network 更新频率
     - Gradient clipping 策略
     - Replay ratio 调度

3. **跨范式迁移学习支持**：
   - 开发支持从 CPU-scale 预训练平滑过渡到 GPU-parallel 微调的统一算法框架。

4. **硬件感知优化**：
   - 结合具体硬件（如 B200 GPU）进一步优化 SWD 实现，追求“分钟级”部署目标。

---

> 🔚 **总结一句话**：  
> **WarpSAC 重新定义了可扩展 off-policy RL 的设计哲学——不是堆叠稳定器，而是让算法“感知”数据规模，并据此做出最优配置。**

</details>

---

### 14. [SeisMamba: Low-Latency Single-Station Seismic Magnitude Estimation for Spatially Distributed Earthquake Early Warning](https://arxiv.org/abs/2608.24561)

**Authors**: Quenton Yeo, Zhaoge Bi, Linghan Huang, Luke Stephen Higgins, Flora Salim, Huaming Chen  
**Category**: cs.LG  
**Published**: 2026-08-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.24561v1  

#### Abstract
Rapid earthquake magnitude estimation is central to earthquake early warning, yet many operational systems depend on dense regional seismic networks and region-specific calibration. This creates a spatial coverage barrier for high-risk areas with sparse sensing infrastructure. Single-station learnin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SeisMamba: Low-Latency Single-Station Seismic Magnitude Estimation for Spatially Distributed Earthquake Early Warning

## 1. 论文的主要贡献和创新点

### 解决的问题
- **地震早期预警**（Earthquake Early Warning, EEW）中对快速、准确的地震震级估计的需求。
- 现有系统依赖密集区域地震网络和区域特定校准，在监测基础设施稀疏的高风险地区存在空间覆盖障碍。
- 单站学习（single-station learning）虽为低成本替代方案，但面临**精度-延迟权衡**（accuracy-latency trade-off），且在跨区域部署时可能因分布偏移而性能下降。

### 提出的新方法与创新思路
提出 **SeisMamba** —— 一种基于 **Mamba** 架构的轻量级模型，用于从单个台站记录的三通道地震波形中进行低延迟震级估计。其核心设计围绕以下几点：
- **Hierarchical Convolutional Encoding**：分层一维卷积编码器提取局部波形特征（如初动锐度、极性变化、振幅增长）。
- **Sparse Selective State-Space Modelling**：仅在部分编码层级插入 **Mamba blocks**，实现长序列建模的同时控制计算开销。
- **Multi-Scale Feature Fusion**：融合多尺度特征以结合浅层敏感信号与深层上下文信息。
- **Auxiliary Temporal Prediction Head**：辅助时间分辨预测头提供密集监督，揭示震级证据何时趋于稳定。

### 相比现有方法的优势
- **高效性**：相比 Transformer 类模型避免了二次复杂度注意力机制，实现线性时间建模。
- **低延迟**：适用于边缘设备部署，支持近传感端实时推理。
- **强泛化能力**：在地理上未见区域仍保持良好性能，具备一定跨区域鲁棒性。
- **端到端训练**：无需手工特征（hand-crafted P-wave features）、震源位置或台站校准输入。

---

## 2. 核心实验方法和设置

### 数据集
- 使用全球性地震数据集 **STEAD**（Stanford Earthquake Dataset）
  - 包含单台站三分量（ENZ）地震波形
  - 输入长度：30秒，采样率 100 Hz → $ \mathbb{R}^{3 \times 3000} $
  - 预处理：仅应用 1–40 Hz 四阶零相位 Butterworth 带通滤波
  - 不使用任何额外信息（如震中距、台站参数等）

### 实验设置
- **训练配置**：
  - 优化器：AdamW
  - Batch size：32
  - 学习率：$10^{-3}$，warm-up 5轮，plateau调度
  - 最大训练轮数：150，早停机制
- **评估协议**：
  - 主要任务：标量震级回归（scalar magnitude estimation）
  - 输出目标：真实地震震级（magnitude）

### 评估指标
| 指标 | 含义 |
|------|------|
| **MSE** | 均方误差 |
| **RMSE** | 均方根误差 |
| **MAE** | 平均绝对误差 |
| **R²** | 决定系数（越高越好） |
| **Inference Latency** | 推理延迟（ms / batch of 32 waveforms on NVIDIA T4 GPU） |

### 基线方法对比
- **PhaseNet** [12]：原用于震相拾取，适配为回归任务
- **MagNet** [4]：直接从原始波形估计震级的深度模型
- **EQTransformer** [5]：基于注意力的检测与拾取模型，改为回归输出
- **AMAG** [9]：基于注意力机制的震级估计模型
- **U-Mamba** [3]：医学图像分割中的 Mamba 变体，用于比较

---

## 3. 主要实验结果和性能指标

### 关键性能数据（STEAD 全局基准测试）

| Model | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ | Time (ms) ↓ |
|-------|--------|---------|--------|--------|--------------|
| PhaseNet | 0.1316 | 0.3627 | 0.2724 | 0.8638 | **0.50** |
| MagNet | 0.1245 | 0.3528 | 0.2561 | 0.8712 | 0.97 |
| EQTransformer | 0.0860 | 0.2932 | 0.2016 | 0.9237 | 1.70 |
| AMAG | 0.0681 | 0.2609 | 0.1467 | 0.9389 | 1.58 |
| U-Mamba | 0.0683 | 0.2613 | 0.1793 | 0.9282 | 1.39 |
| **SeisMamba (Ours)** | **0.0628** | **0.2506** | **0.1566** | **0.9443** | **0.55** |

> ✅ **结论**：SeisMamba 在 **MSE、RMSE、R² 上全面领先**，同时推理速度约为 Transformer 类模型（如 EQTransformer）的 **3 倍以上**。

---

### 地理持留实验（Chile-Taiwan Regional Hold-Out）

- **设置**：将智利（Chile）和台湾（Taiwan）所有事件从训练集中排除，仅用于测试，共排除 5,478 条记录。
- 目的：检验模型在**地理上未见过区域**的泛化能力。

| Setting | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ | Time (ms) |
|--------|--------|--------|--------|--------|------------|
| **Chile-Taiwan Hold-Out** | 0.1669 | 0.4085 | 0.2808 | **0.8518** | 0.58 |

> 📌 尽管性能有所下降（合理预期），但在完全未见地理区域仍取得 **R² > 0.85** 的表现，表明模型学到的是具有一定普适性的波形表征。

---

### 消融实验结果（Ablation Study）

| 设置 | MSE | R² | Time (ms) | 分析 |
|------|-----|-----|----------|------|
| Encoder only | 0.0763 | 0.9323 | 0.56 | 缺少长程建模能力 |
| Wider encoder | 0.0669 | 0.9406 | 0.55 | 宽度提升有限 |
| Deeper encoder | 0.0660 | 0.9414 | **1.09** | 精度略优但延迟翻倍 |
| Dense Mamba | 0.0796 | 0.9293 | 0.61 | 每层加 Mamba 效果更差 |
| No fusion | 0.0778 | 0.9310 | 0.66 | 多尺度融合显著重要 |
| **Final Hybrid (SeisMamba)** | **0.0628** | **0.9443** | **0.55** | 综合最优 |

> 🔍 **关键发现**：
> - **稀疏插入 Mamba blocks 更有效**：说明应在局部压缩后才引入长程建模。
> - **多尺度融合至关重要**：浅层与深层特征互补。
> - **并非越大越好**：更深/更宽结构无法兼顾延迟与增益。

---

## 4. 关键结论和发现

### 主要发现
1. **Selective State-Space Models 是理想 backbone**：
   - Mamba 架构在保持线性时间复杂度的同时，能有效捕捉长序列地震波形中的关键动态。
   - 显著优于传统 CNN 和 attention-based 模型在 accuracy-latency trade-off 上的表现。

2. **SeisMamba 支持低成本、分布式 EEW 部署**：
   - 超低延迟（0.55 ms/batch）使其适合部署于边缘设备或资源受限环境。
   - 无需区域校准即可运行，降低部署门槛。

3. **具备一定跨区域泛化能力**：
   - 在 Chile-Taiwan 持留实验中仍保持 R² > 0.85，验证了所学表示的可迁移性。
   - 表明模型未过拟合于特定地质条件。

4. **辅助时间头增强可观测性**：
   - Auxiliary temporal head 提供时间分辨的震级演化轨迹，有助于理解模型“何时”做出判断。
   - 可作为未来不确定性建模的基础。

### 局限性
- **尚未解决完全的区域不变性**（region invariance）：
  - 地理持留下性能下降明显（R² 从 0.94→0.85），说明衰减特性、场地效应、震源机制差异仍是挑战。
- **未集成完整 EEW 决策流程**：
  - 当前仅为震级估计模块，缺乏与警报生成、传播延迟、误报控制等系统的整合验证。
- **不确定性量化不足**：
  - 辅助头非正式的 uncertainty estimator，尚需概率建模支持可靠报警决策。

### 未来工作方向
- 扩展至更多地理区域进行大规模跨域适应研究（domain adaptation / generalization）。
- 结合 real-time streaming 设置进行在线推断与更新实验。
- 引入贝叶斯或集成方法进行不确定性估计，支持风险感知报警。
- 在真实硬件平台（如 IoT sensor nodes）上部署并测试端到端延迟与能耗。
- 探索多任务联合学习框架（如震相拾取 + 震级估计 + 震源定位）。

---

> ✅ **总体评价**：  
> SeisMamba 成功展示了 **Mamba-style selective state-space modelling** 在地震学任务中的巨大潜力，特别是在 **accuracy、latency、deployability** 三者之间取得了优异平衡，是迈向**可扩展、低成本、空间分布式地震早期预警系统**的重要一步。

</details>

---

### 15. [Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged Attention](https://arxiv.org/abs/2608.23834)

**Authors**: Sergii Kozyrev (Minima AI, Inc), Davyd Maiboroda (Minima AI, Inc)  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.23834v1  

#### Abstract
The key-value (KV) cache is a primary capacity and bandwidth bottleneck in long-context LLM serving. We present Minima-KV, a retention-preserving hierarchy for mixed-format paged attention. Recent and protected Anchor pages remain in FP8, while older non-anchor pages move to packed TQ3; every live-r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged Attention — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在长上下文大语言模型（LLM）推理中，**Key-Value (KV) 缓存**是内存和带宽的主要瓶颈。随着上下文长度增加，KV cache 占用线性增长，严重限制了请求并发、批处理效率和 GPU 利用率。

现有方法如统一量化（uniform quantization）、稀疏保留（sparse retention）或低秩压缩存在以下问题：
- 统一量化对所有 token 一视同仁，浪费比特于不重要位置；
- 删除旧 token 的策略可能导致注意力漂移后无法恢复；
- 结构化压缩引入额外重建开销，可能抵消内存收益。

---

### 🚀 提出的新方法：Minima-KV
Minima-KV 是一种**保留完整性的分层 KV 缓存管理系统**，结合了混合格式分页注意力（Mixed-Format Paged Attention），其核心思想是将 KV 页面按生命周期划分为三个保真度层级：

| 层级 | 格式 | 特性 |
|------|------|-------|
| **Recent** | FP8 | 新生成的 token 使用高精度 FP8 存储，保障局部连贯性和推理一致性 |
| **Anchor** | FP8 | 显式保护的重要页面（如系统指令、检索证据、attention sinks）保持 FP8 |
| **Stale** | TQ3 | 老旧非锚点页面使用仅 3-bit 的 TurboQuant 压缩格式（TQ3），大幅降低存储 |

> 所有逻辑页面始终可寻址，无删除操作，确保“retention-preserving”。

#### 创新机制：
- **混合格式注意力核函数（Mixed-format paged attention kernels）**  
  支持 FP8 和 TQ3 并行计算 partial output，并通过稳定的 online-softmax 合并，无需构建全量 dense shadow。
- **Copy-before-publish 所有权协议**  
  保证格式转换过程中的内存安全与并发访问正确性。
- **支持前缀复用（prefix reuse）**  
  完成请求的不可变块可被缓存并跨请求共享，提升 warm-start 性能。
- **CUDA 图兼容异构解码路径**  
  实现高性能、低延迟的生产级部署支持。

---

### 🔍 相比现有方法的优势
| 对比维度 | Minima-KV | 其他方法（如 H2O, SnapKV, KIVI） |
|--------|----------|-------------------------------|
| **是否删除 token** | ❌ 不删除，仅降精度 | ✅ 多数会丢弃部分 KV |
| **能否应对 attention drift** | ✅ 可恢复（Stale → Anchor 晋升） | ❌ 一旦删除即永久丢失 |
| **是否需要 dense shadow** | ❌ 无，节省显存 | ✅ 多数需临时反量化 buffer |
| **是否支持生产级 runtime** | ✅ 集成于 paged runtime，支持 CUDA graph | ⚠️ 多为研究原型 |
| **压缩粒度控制** | ✅ 页面级动态升降级 | ⚠️ 多为静态或层级别 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **RULER NIAH**：8 个 “needle-in-a-haystack” 任务，在 4K/8K/16K 上下文中测试定位能力（每任务 32 示例）
- **LongBench v2**：503 道多选题，覆盖六大现实场景：
  - 单文档问答
  - 多文档问答
  - 长上下文学习
  - 对话理解
  - 代码仓库理解
  - 结构化数据理解  
  测试长度：16K、32K、64K

> 所有测试采用 paired evaluation（相同样本对比 dense 控制组）

---

### ⚙️ 实验设置
- **模型**：Qwen3.6-27B（仅语言路径，vision encoder 排除）
- **硬件**：单张 NVIDIA RTX PRO 6000 Blackwell GPU（96GB GDDR7）
- **上下文管理**：PagedAttention 架构，逻辑页大小为 1,792 tokens
- **评估模式**：
  - **质量评估**：materializing 模式，TQ3 页面在 attention 前解压
  - **性能评估**：direct-fused 模式，FP8/TQ3 异构 kernel 直接融合执行
- **基线对比**：
  - **BF16 KV**：标准浮点格式，64 KiB/token
  - **FP8 KV**：当前主流压缩基线，32 KiB/token
  - **Minima-KV**：本工作提出的三阶段混合格式系统（FP8 + TQ3）

> 注意：未绑定 dense control 的具体 dtype，因此吞吐比仅为配对内相对值

---

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **存储效率** | 每 token KV 占用字节数、相对于 BF16/FP8 的压缩倍数 |
| **质量表现** | RULER NIAH 任务宏平均准确率、LongBench v2 正确数及 delta |
| **性能表现** | 解码吞吐（tok/s）、调度步时间、layer 路由成功率、fallback 次数 |
| **容量预测** | 最大驻留上下文数量（基于内存模型估算） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 指标 | 数值 | 来源 |
|------|------|------|
| **实际 KV 占用** | **18.3 KiB/token** | 部署统计（owner-reported aggregate） |
| **相比 BF16 压缩比** | **3.50×** | $64.0 / 18.3 ≈ 3.50$ |
| **相比 FP8 压缩比** | **1.75×** | $32.0 / 18.3 ≈ 1.75$ |
| **Active KV 压缩比（实测）** | **3.625×** | direct canary 中物理计数器 |
| **吞吐比率（vs dense control）** | **0.9821×** | 单一对比实验，非统计显著性声明 |
| **LongBench v2 回归幅度** | -0.80 pp @16K, -0.60 pp @32K, -0.40 pp @64K | 同一组 503 问题 delta |
| **RULER NIAH 表现** | @4K: -0.9 pp, @8K: +0.20 pp, @16K: 0.00 pp | 宏平均准确率变化 |

---

### 🔁 与基线方法对比结果

#### 存储效率对比（每 token KV 大小）
| 方法 | Bytes/token | 压缩倍数（vs BF16） |
|------|------------|------------------|
| BF16 KV | 64.0 KiB | 1.00× |
| FP8 KV | 32.0 KiB | 2.00× |
| **Minima-KV（实测）** | **18.3 KiB** | **3.50×** |

> 在百万 token 场景下，Minima-KV 仅需 17.45 GiB，而 BF16 需要 61.04 GiB，节省超 70%

#### 并发容量提升（理论估算）
使用公式：
$$
C_{mem}(L) = \frac{M - W}{D + K(L)}
$$
其中 $M=86.4\,\text{GiB}$, $W=25.15\,\text{GiB}$, $D=0.153\,\text{GiB}$

| 上下文长度 | BF16 支持并发 | FP8 支持并发 | Minima-KV 支持并发 |
|-----------|---------------|--------------|--------------------|
| 32K       | 28            | 53           | **84**             |
| 64K       | 14            | 28           | **47**             |

> 在 32K 上，Minima-KV 较 FP8 提升 **1.58×** 驻留容量

---

### 🔍 消融实验与关键验证

#### （1）Direct Decode Canary 实验（双请求，各 59K prompt）
| 指标 | Control | Minima-KV |
|------|--------|---------|
| 解码吞吐 | 29.804 tok/s | 29.270 tok/s |
| 吞吐比率 | 1.000 | **0.9821** |
| Active KV 字节 | 477,102,080 | **131,610,624** |
| 压缩比 | 1.00× | **3.625×** |
| Layer 路由 | — | 16/16 成功 |
| Fallbacks | — | 0/32 |

✅ 表明系统可在无 fallback、无 dense shadow 下稳定运行全部 16 层 full-attention。

#### （2）Warm-Prefix 复用实验（832-token pages）
| 指标 | 结果 |
|------|------|
| 前缀复用吞吐比（vs dense-FP8） | **1.021×** |
| 物理 KV 压缩比 | **4.342×** |
| 输出一致性验证 | 32/32 完全匹配 |
| fallback 次数 | 0 |

✅ 表明前缀复用不仅可行，还能实现轻微性能增益。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Minima-KV 实现了高效的 retention-preserving 压缩架构**  
   通过 FP8 + TQ3 分层设计，在不删除任何 token 的前提下实现了高达 **3.5× BF16 压缩比**。
   
2. **质量损失可控且随上下文增长收敛**  
   - 在 16K RULER NIAH 上达到与 dense 控制完全一致；
   - LongBench v2 的准确率下降从 16K 的 -0.80 pp 缩小到 64K 的 -0.40 pp，表明压缩误差影响随长度增加反而减小。

3. **性能接近 dense 基线，具备生产可用性**  
   - 吞吐达 dense 控制的 **98.21%**，且全程无 fallback；
   - 支持 CUDA graph，适合高并发服务场景。

4. **显著提升系统并发容量**  
   理论上可在相同内存下支持 **84 个 32K 上下文请求**，远高于 FP8 的 53 个。

5. **前缀复用机制有效提升 warm-start 效率**  
   实现了 **1.021× 吞吐加速** 和 **4.342× 物理压缩**，适用于 agent 或 RAG 等重复前缀场景。

---

### ⚠️ 方法的局限性
| 局限 | 描述 |
|------|------|
| **单一模型 & 硬件验证** | 仅在 Qwen3.6-27B + RTX PRO 6000 上测试，泛化性待验证 |
| **聚合数据缺乏细粒度拆解** | 18.3 KiB/token 为整体统计，未提供各 tier 占比、metadata 开销等细节 |
| **Profile 分离** | 质量、性能、前缀复用来自不同配置，无法在同一实验中同时验证所有属性 |
| **未启用 attention scoring 反馈** | 当前 promotion/demotion 依赖静态规则，未利用 attention 动态反馈优化 Anchor 管理 |
| **无开放 artifact** | 实现闭源，仅发布摘要结果，难以复现或审计 |

---

### 🔮 未来工作方向
1. **四层级扩展（Stale2）**  
   提出未来可引入基于 **tensor network** 的更冷存储格式（如 Tucker 分解 + adapter），进一步压缩最冷门页面（Stale2），形成 {R, A, S1, S2} 四层体系。

2. **混合精度 + 结构化压缩联合优化**  
   将 TQ3 与低秩表示结合（类似 JoLT），探索近无损压缩极限。

3. **端到端 rate-distortion 优化控制器**  
   借鉴 RDKV 思路，将量化、保留、格式选择建模为 bit-allocation 问题，实现全局最优决策。

4. **开放透明评估框架**  
   发布完整配置清单、日志与 telemetry，推动行业标准化 benchmarking。

---

> 📌 **总结一句话**：  
> **Minima-KV 提出了一种兼顾高效压缩、完整性保留与生产可用性的 KV cache 管理方案，在几乎不牺牲质量的前提下实现 3.5× 存储压缩和接近原生吞吐，为长上下文 LLM 推理提供了实用化的系统路径。**

</details>

---

### 16. [Data Mixing as Mixture Experiment: Response Surface Methodology and Optimal Design for Large Language Model Pretraining](https://arxiv.org/abs/2608.23922)

**Authors**: Yicheng Mao, Hongru Du  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.23922v1  

#### Abstract
Data mixing is a central design problem in large language model pretraining: given a fixed token budget, practitioners must decide how much data to allocate to each domain. Recent proxy-based methods address this problem by training small models on candidate mixtures, fitting a response model, and u...

---

### 17. [Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping](https://arxiv.org/abs/2608.24135)

**Authors**: Yiwen Zhang, Xiaodong Yan, Zhenyu Huang, Deng Zhao, Liang Jiang, Qing Cui, Zujie Wen, Zhiqiang Zhang, Jun Zhou  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.24135v1  

#### Abstract
Reinforcement learning from verifiable rewards (RLVR) has emerged as a pivotal technique for enhancing the code generation capabilities of Large Language Models (LLMs). However, the efficacy of RLVR in coding implementations is fundamentally limited by the comprehensiveness of test cases, because in...

---

### 18. [MetaRAG: Belief-Action Aligned Policy Optimization for Agentic RAG](https://arxiv.org/abs/2608.24214)

**Authors**: Qiuyi Qi, Tian Liang, Jiamu Wang, Jinjian Zhang, Wei Zhou, Pengcheng Zhu, Linjian Mo, Ming Kong, Jie Liu, Qiang Zhu  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.24214v1  

#### Abstract
Agentic retrieval-augmented generation (RAG) requires language models to decide when to continue searching and when to answer. Existing RL-based methods rely on external supervision and overlook the agent's internal belief about whether the current evidence is sufficient. To address this problem, we...

---

### 19. [ResiSpec: Enhancing Multi-Candidate Speculative Sampling via Residual Distribution Shaping](https://arxiv.org/abs/2608.24411)

**Authors**: Zhi-Kai Chen, Jun-Jie Tao, Wei-Xiang Mao, De-Chuan Zhan, Han-Jia Ye  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.24411v1  

#### Abstract
The efficiency of Large Language Model (LLM) serving is fundamentally limited by the sequential nature of autoregressive decoding. Speculative Decoding (SD) mitigates this by using a lightweight draft model to speculate future tokens, which are then validated by the LLM in a single parallel forward ...

---

### 20. [Reinforcement Learning-Guided Evolutionary Policy Optimization for Preference-Adjustable Heterogeneous Agile Earth Observation Satellite Scheduling](https://arxiv.org/abs/2608.24470)

**Authors**: He Wang, Junyu Wu, Hui Li, Yanjie Song, Witold Pedrycz, Liang Li  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.24470v1  

#### Abstract
Heterogeneous agile Earth observation satellite (AEOS) scheduling requires task selection, satellite assignment, and observation sequencing under satellite-dependent visibility windows, attitude maneuvering requirements, energy consumption, and onboard storage constraints. Since satellites differ in...

---

### 21. [PeakBench: Benchmarking Resource-Aware Tool Invocation in LLM Agents](https://arxiv.org/abs/2608.24509)

**Authors**: Zhi-Kai Chen, Xu-Xiang Zhong, Song-Yan Li, De-Chuan Zhan, Han-Jia Ye  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.24509v1  

#### Abstract
LLM agents increasingly solve tasks by invoking multiple tools, where parallel execution is essential for low latency but difficult to manage safely. Existing agent benchmarks primarily evaluate tool selection, argument generation, and end-to-end success under mostly serial execution, largely overlo...

---

### 22. [Context-Aware Cluster Decoding: Semantic Anchor-Driven Coherence in dMLLMs](https://arxiv.org/abs/2608.22367)

**Authors**: Yikai Zhao, Qiyan Zhao, Jiaquan Zhang, Xiaofeng Zhang, Xiaosong Yuan, Pengzhou Cheng  
**Category**: cs.CL  
**Published**: 2026-08-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.22367v1  

#### Abstract
Diffusion multimodal large language models (dMLLMs) frequently produce long-form outputs marred by semantic drift and repetition, with quality generally degrading as output length increases. We identify two structural deficiencies in existing decoding methods as primary drivers of these failures: co...

---

### 23. [WnW: Waxing-and-Waning KV Cache for Long-Form Speech LLMs](https://arxiv.org/abs/2608.22704)

**Authors**: Yiming Yao, Chenyang Lyu, Xuanfan Ni, Longyue Wang, Weihua Luo, Yazheng Yang, Jinsong Su  
**Category**: cs.CL  
**Published**: 2026-08-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.22704v1  

#### Abstract
Long-form audio inputs make the KV cache the dominant memory cost of speech LLMs. Prefill-only KV compression methods permanently discard audio KV positions once evicted, with no pathway to recover them during decoding. We show this is fragile on long-form audio: prefill attention concentrates near ...

---

### 24. [Generating Intervention Hypotheses using Explainable Explanations on Graphs: G2I, a Two-Stage Greedy Framework](https://arxiv.org/abs/2608.23835)

**Authors**: Mulin Tian, Ajitesh Srivastava  
**Category**: cs.LG  
**Published**: 2026-08-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.23835v1  

#### Abstract
Real-world decision-making in public health and social science can greatly benefit from predictive models, yet translating predictions into effective interventions requires explaining the model behavior. While Graph Neural Networks (GNNs) are well-suited for modeling relational data, existing explan...

---

### 25. [Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware](https://arxiv.org/abs/2608.23807)

**Authors**: Farhana Amin, Sabiha Afroz, Mona Moghadampanah, Dimitrios S. Nikolopoulos  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.23807v1  

#### Abstract
Masked diffusion language models (dLLMs) can in principle generate text faster than autoregressive (AR) models, since they denoise many tokens at once. Recent systems have begun building serving infrastructure for dLLMs, but none first measure how these models behave under real, concurrent serving l...

---

### 26. [GTA-RAG: Graph-Trajectory-Augmented Reinforcement Learning for Multi-Turn Retrieval-Augmented Reasoning](https://arxiv.org/abs/2608.22479)

**Authors**: Jun Chen, Yongchao Liu, Pengyu Qiu, Jiajun Zheng, Juelu Zhang, Yujie Zeng, Qin Zhang, Ziyue Qiao, Xiao Luo  
**Category**: cs.CL  
**Published**: 2026-08-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.22479v1  

#### Abstract
Retrieval-augmented generation (RAG) enables LLMs to access external knowledge for answering knowledge-intensive questions. For complex multi-hop questions, multi-turn retrieval-augmented reasoning extends RAG into an iterative process that repeatedly searches for and integrates evidence across docu...

---

### 27. [Accelerating Diffusion Language Models via Structured Suffix Modeling](https://arxiv.org/abs/2608.23167)

**Authors**: Zifeng Cheng, Keda Li, Zhiwei Jiang, Cong Wang, Fei Shen, Qing Gu  
**Category**: cs.CL  
**Published**: 2026-08-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.23167v1  

#### Abstract
Diffusion Language Models (DLMs) exhibit strong parallel decoding capabilities by denoising multiple tokens in a single generation step. However, this parallelism comes with substantial computational overhead, as each step requires interactions with all suffix tokens. Existing methods typically redu...

---

### 28. [Future Querying: Can LLMs Serve as Implicit Medical World Models?](https://arxiv.org/abs/2608.23248)

**Authors**: Siri Willems, James Butterworth, Lore Goetschalckx, Peter Vrancx, Philippe Modard, Elke Giets, Ludovic Denoyer  
**Category**: cs.CL  
**Published**: 2026-08-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.23248v1  

#### Abstract
Traditional clinical prediction models rely on task-specific pipelines and curated, structured data, which scale poorly and underutilize unstructured text. To address this, we introduce future querying, a paradigm that probes whether large language models (LLMs) can function as implicit medical worl...

---

### 29. [Relative Time Intervals Representation for Word-level Timestamping with Masked Training](https://arxiv.org/abs/2608.24041)

**Authors**: Quanwei Tang, Zhiyu Tang, Xu Li, Dong Zhang,  Shoushan, Guodong Zhou  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.24041v1  

#### Abstract
Although Speech Large Language Models (SpeechLLMs) excel at speech understanding and generation, their capacity for fine-grained, temporally aligned outputs remains underexplored. Our work addresses this gap by enabling SpeechLLMs to jointly model speech content and temporal structure, effectively t...

---

### 30. [Joint Optimization of Tool Creation and Use for Large Language Model Agents](https://arxiv.org/abs/2608.24571)

**Authors**: Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, Shao-Hua Sun, Hung-yi Lee  
**Category**: cs.AI  
**Published**: 2026-08-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.24571v1  

#### Abstract
Tool-augmented language models are bounded by the APIs humans bothered to write; existing tool-creation systems patch this by prompting a frozen LLM at inference time, leaving the model that writes a tool decoupled from the one that uses it, with no signal that the schemas it produces are schemas it...

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

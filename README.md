# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-24 06:12:52 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308)

**Authors**: Zeyun Zhong, Joya Chen, Manuel Martin, Frederik Diederichs, Juergen Gall, Juergen Beyerer  
**Category**: cs.LG  
**Published**: 2026-08-24  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2608.21308v1  

#### Abstract
Test-Time Training (TTT) enables long-context processing via continuous weight updates during inference, but current methods struggle to balance the expressivity of per-token update dynamics with the hardware efficiency of chunk-wise approximations. We propose E$^2$-TTT (Expressive and Efficient TTT...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Rethinking Expressivity and Efficiency in Test-Time Training**

## **1. 主要贡献和创新点**

### **解决的问题**
当前的 **Test-Time Training (TTT)** 方法在**表达力 (expressivity)** 和**硬件效率 (efficiency)** 之间存在根本性权衡：
- **Token-wise TTT**：逐 token 更新 fast weights，具有强大的建模能力，但因序列依赖导致训练缓慢，硬件利用率低。
- **Chunk-wise TTT**：通过大块更新提升效率，但为简化计算而牺牲了 token 级别的动态变化（如对不同 token 的重要性进行区分），丢失了时间结构。

该论文旨在解决这一“表达力 vs. 效率”的困境。

### **提出的新方法：E2-TTT (Expressive and Efficient TTT)**
作者提出了 **E2-TTT**，一种新型的 TTT 框架，其核心创新在于一个**闭式标量核 (closed-form scalar kernel)**。

#### **核心思想**
在标准近似（即梯度在块起始权重处计算）下，推导出一个精确的**状态转移公式**，该公式能够：
- **完全并行化**地执行块级训练。
- **精确复现**逐 token 递归所得到的块结束时的 fast-weight 和 momentum 状态。
- **保留**了逐 token 更新规则中的时间结构（如学习率、动量、衰减的 token 级别变化），而这是之前 chunk-wise 方法所丢弃的。

#### **关键技术**
- **两种不同的标量核 (Scalar Kernels)**：推导表明，块结束时的权重 `Wc` 和动量 `Mc` 是通过对每个 token 的梯度 `{G}` 进行加权聚合得到的，且权重由两个不同的标量核决定：
  - **动量核 (Momentum Kernel)**：衡量梯度 `Gt` 通过动量衰减 `βt` 在块结束时的留存程度。
  - **权重核 (Weight Kernel)**：进一步结合了权重衰减生存率 `Rt`，捕捉梯度如何通过动量缓冲区和后续权重更新传播。
- **单次反向传播**：这两个核共享相同的激活梯度，因此可以通过一次反向传播计算，然后进行标量重加权，避免了额外的计算开销。

### **相比现有方法的优势**
- **兼具表达力与效率**：在保持与高效 chunk-wise 方法相当的训练吞吐量的同时，实现了与逐 token TTT 相当的建模表达力。
- **精确性**：不采用平均等简化手段，而是精确地保留了 token 级别的动态特性。
- **可扩展性**：成功训练了高达 1.3B 参数的模型，证明了其实际可行性。

---

## **2. 核心实验方法和设置**

### **数据集**
- **预训练数据**：`HuggingFace FineWeb-Edu` 数据集，共 15B tokens。
- **评估任务**：
  - **通用语言建模**：`WikiText`, `LAMBADA`, `PIQA`, `HellaSwag`, `WinoGrande`, `ARC-e/c`。
  - **上下文检索 (In-context Retrieval)**：`FDA`, `SWDE`, `SQuAD`。
  - **长度外推 (Length Extrapolation)**：
    - 合成任务：`S-NIAH-1` (passkey retrieval), `S-NIAH-2` (numerical needle)。
    - 真实世界基准：`LongBench` (包含 14 个长上下文任务)。
  - **多模态理解**：`VideoMMMU`, `LongVideoBench`。

### **实验设置和评估指标**
- **模型规模**：340M 和 1.3B 参数。
- **序列长度**：训练时为 2K 或 4K tokens，测试时外推至 16K tokens (8× 训练长度)。
- **Chunk Size**：512 tokens。
- **评估指标**：
  - **语言建模**：困惑度 (perplexity ↓)。
  - **零样本准确率**：各类常识推理和检索任务的准确率 (accuracy ↑)。
  - **长度外推**：在远超训练长度的序列上的性能稳定性及准确率。

### **基线方法对比**
- **Transformer++**：全注意力基线。
- **DeltaNet**：线性注意力模型。
- **HQLT**：混合架构 (DeltaNet + 滑动窗口注意力)。
- **LaCT**：当前最先进的 chunk-wise TTT 方法 (TTT + 滑动窗口注意力)。
- **Mamba2**：基于 SSM 的模型。
- **Titans**：另一种 TTT 方法。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据与对比结果**

#### **通用语言建模**
- **E2-TTT** 在 340M 和 1.3B 规模上均取得了**最低的困惑度**。
- 在 1.3B 规模下，`E2-TTTMLP` 的平均零样本准确率达到 **54.5%**，优于最强基线 HQLT (53.8%)。

#### **上下文检索能力**
- 在 `FDA`, `SWDE`, `SQuAD` 等检索密集型任务上，`E2-TTTswiGLU` (1.3B) 平均准确率为 **43.6%**，显著优于 HQLT (35.5%) 和 LaCT (36.7%)。
- 验证了其**表达性更新规则**在精确召回方面的优势。

#### **长度外推能力 (核心亮点)**
- **语言建模稳定性**：在 6 个长上下文数据集上，`E2-TTT` 的损失曲线稳定，而 `LaCT` 的损失在超出训练长度后急剧爆炸。
- **合成检索任务 (S-NIAH)**：
  - 在 `S-NIAH-1` 上，`E2-TTT` 在 **16K tokens (8×)** 长度下仍保持 **>90%** 的准确率。
  - 相比之下，`LaCT` 几乎降至 0%，`HQLT` 降至 25%。
- **真实世界长上下文 (LongBench)**：
  - `E2-TTTswiGLU` 平均得分为 **14.1%**，大幅领先于 HQLT (12.1%)、Mamba2 (10.3%) 和 LaCT (7.7%)。
  - 在 14 个任务中，`E2-TTTswiGLU` 全面超越所有基线。

#### **训练吞吐量**
- `E2-TTT` 的训练吞吐量与高效的 `LaCT` 相当，仅比其慢约 3.3% (1.3B 模型)，证明了其**高效率**。

### **消融实验结果**
- **更新规则消融**：
  - 将 `E2-TTT` 的 per-token 动量和衰减因子坍缩为块级标量 (`Chunk-averaged`)，在 `S-NIAH-1` 8K 上的准确率从 **95.4%** 暴跌至 **6.8%**。
  - 证明了**精确的 token 级别动态是长上下文检索性能的关键**。
- **组件消融**：
  - 移除 `Sliding Window Attention (SWA)` 分支，模型在所有 `S-NIAH` 任务上性能崩溃至 0.0%。
  - 移除 `fusion gate`，性能在长序列上显著下降。
  - 证实了**混合架构**（TTT + SWA）对于处理局部和全局依赖的必要性。
- **超参数敏感性**：
  - 方法对基础学习率 `η_base` 和基础衰减 `α_base` 在较宽范围内（两个数量级）表现稳健。

---

## **4. 关键结论和发现**

### **主要发现**
1. **表达力至关重要**：在长上下文任务，尤其是长度外推中，**保留 token 级别的更新动态**（如输入依赖的学习率、动量、衰减）对于模型泛化能力至关重要。
2. **效率可以兼顾**：通过提出的**闭式标量核**，可以在不牺牲表达力的前提下实现高效的并行化训练，成功弥合了 `expressivity` 与 `efficiency` 之间的鸿沟。
3. **混合架构的有效性**：将 `E2-TTT` 与 `Sliding Window Attention` 结合的混合设计，能同时捕获局部精细上下文和长距离非局部依赖，是处理长序列的有效范式。
4. **强大的长度外推**：`E2-TTT` 展现出卓越的长度外推能力，在 8× 训练长度下仍能保持高性能，而现有方法在此场景下会严重退化。

### **局限性**
1. **块内盲点 (within-chunk blind spot)**：由于输出步骤使用的是前一块的权重，导致模型在当前块内部无法利用最新的上下文信息进行因果推理。
2. **模型规模限制**：实验主要在 1.3B 及以下参数规模进行，将其扩展到更大规模（如 10B+）的 LLM 仍是未来工作。

### **未来工作方向**
1. 设计高效的 **token-wise 输出机制**，以消除块内盲点。
2. 将 `E2-TTT` 扩展到**更大规模的模型**。
3. 探索其在更多**多模态长序列任务**（如视频、音频）中的应用。

</details>

---

### 2. [LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding](https://arxiv.org/abs/2608.20530)

**Authors**: Matan Rusanovsky, Yoav Miron, Roy Uziel, Omer Belhasin, Ran Zilberstein, Maor Ashkenazi, Michael Elad  
**Category**: cs.CL  
**Published**: 2026-08-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.20530v1  

#### Abstract
Speculative decoding accelerates language-model inference by drafting future tokens that the target model verifies in parallel. A diffusion-style block head such as DFlash is an attractive drafter, predicting an entire block of future tokens in one forward pass. However, it is trained on per-positio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **speculative decoding** 中，一个轻量级的 **drafter 模型** 并行生成多个未来 token，由目标模型（target model）一次性验证。其中，像 **DFlash** 这类基于 diffusion 的并行 drafter 能在一个前向传递中预测整个 token 块，效率极高。

然而，这类模型存在“**marginals problem**”：它们对每个位置（slot）独立优化交叉熵损失，导致生成的 token 在局部合理，但整体序列缺乏连贯性（joint incoherence）。这会降低 **acceptance length**（被目标模型接受的 token 数），从而削弱加速效果。

### 提出了什么新方法或新思路
作者提出 **LiLiCorr**（Lightweight Likelihood Correlation），一种轻量化的、基于似然的后处理模块，用于在 draft 阶段恢复 token 序列的连贯性。

其核心思想是：
- **保留 top-K 候选**：对于 DFlash 输出的每个 slot，保留 top-K 最可能的候选 token。
- **联合建模候选关系**：为每个候选 token 学习两个低维向量：`in` 向量 和 `out` 向量。
- **通过向量对齐评分**：相邻位置的候选是否兼容，取决于前一个候选的 `out` 向量 与后一个候选的 `in` 向量 的 **余弦相似度**。
- **单次网络前向传递**：所有 `in` 和 `out` 向量通过一个轻量级 Transformer 在一次前向传递中计算完成。
- **并行打分 + 贪心解码**：所有候选对的匹配分数通过批量矩阵运算并行计算，最终通过贪心方式从左到右选择最优路径。

### 相比现有方法的优势
| 方法 | 缺陷 | LiLiCorr 的优势 |
|------|------|------------------|
| **DDTree** | 将候选扩展成树，由 target 模型验证，增加了 target 的计算负担 | 在 draft 阶段完成相关性建模，不增加 target 开销 |
| **Domino / DSpark** | 采用自回归方式逐个修正 slot，需要多次网络前向传递 | **仅需一次网络前向传递**，显著降低延迟 |
| **迭代去噪（如 LLaDA）** | 多轮迭代预测，每轮都是完整 drafter 推理，成本高 | 单步完成，无额外 drafter 推理 |

**核心优势**：将复杂的联合分布建模转化为高效的 **向量对齐 + 批量矩阵运算**，实现了 **高吞吐、低延迟** 的连贯性恢复。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主评估数据集**（9个）：
  - **数学**：GSM8K, MATH-500, AIME 2025
  - **代码**：HumanEval, MBPP, LiveCodeBench
  - **对话**：Alpaca, MT-Bench
  - **综合基准**：SPEED-Bench（涵盖数学、编码、多语言、摘要、角色扮演等11类）
- **训练数据**：约 140 万条来自 Nemotron Post-Training Dataset V2 的样本，覆盖代码、数学、STEM 和通用对话，**不含多语言数据**。

### 实验设置和评估指标
- **目标模型**：Qwen3-8B 和 Qwen3-4B（冻结）
- **解码模式**：
  - **Greedy decoding**
  - **Temperature-1 sampling**（配合精确拒绝采样保持分布一致性）
- **评估指标**：
  - **Acceptance length (T)**：每个验证块平均接受的 token 数（block-weighted）
  - **Throughput (TPS)**：每秒输出 token 数
  - **Speedup**：相对于纯自回归解码的加速比
- **硬件**：单张 H100 GPU
- **并发测试**：测试从 c=1 到 c=32 的不同并发场景

### 基线方法对比
- **Vanilla DFlash**：原始 DFlash drafter，无重排序
- **Domino**：与 LiLiCorr 同期工作，逐 slot 自回归修正
- **DSpark**：另一同期工作，支持确定性和采样两种模式
- 所有系统均在相同训练数据、批次大小、序列长度下训练，并部署于同一 SGLang 服务栈，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **LiLiCorr 将 acceptance length 提升了 9%–19%**（相比 Vanilla DFlash）。
- **在 72 个测试场景中，LiLiCorr 在 70 个场景中实现了最高吞吐量**。
- **LiLiCorr 的 speculative head 延迟仅占每 block 总延迟的约 2.8%**，而 Domino 的 head 延迟是其 2.4 倍（0.67ms vs 0.28ms）。
- 在 **Qwen3-8B + greedy decoding** 下，平均加速比达到 **4.51×**，优于 Domino (4.32×) 和 DSpark (4.37×)。

### 与基线方法的对比结果
| 指标 | LiLiCorr vs Vanilla DFlash | LiLiCorr vs Domino | LiLiCorr vs DSpark |
|------|----------------------------|--------------------|---------------------|
| Acceptance Length | ↑ 9–19% | 更高（83% 测试点） | 可比或更高 |
| Throughput | 显著提升 | **全面领先**（36/36 场景） | **全面领先**（36/36 场景） |
| Latency | head 增加 0.28ms | head 成本低 2.4× | head 成本更低 |
| 并发表现 | 高并发下仍保持领先 | 领先优势随并发增大而扩大 | 领先优势明显 |

> 特别地，在 **多语言任务** 上，尽管训练数据未包含多语言样本，LiLiCorr 仍取得最大增益（Qwen3-4B 上吞吐提升 **16.8%**），表明其具有良好的 **泛化能力**。

### 消融实验结果
#### （1）Distractor Penalty 消融
- 移除 `distractor penalty` 后，acceptance length 在所有 9 个基准上均下降（平均 ↓3.2%）。
- 该损失项引导模型抑制那些 target 模型得分低但 LiLiCorr 可能误判的候选，是提升鲁棒性的关键。

#### （2）Greedy vs Global Path Search
- 对比贪心解码（greedy walk）与动态规划寻找全局最优路径（Viterbi）：
  - **贪心解码在所有 9 个基准上均优于全局搜索**（平均 acceptance length ↓2.44%）。
- **原因**：speculative decoding 是 **prefix acceptance**，一旦某个位置失败，后续全部丢弃。因此，优先保证早期位置正确比追求整体路径最优更重要。

#### （3）长输入泛化（8K–32K tokens）
- 使用 **YaRN** 扩展位置编码后，LiLiCorr 在远超训练长度（3K）的输入上仍保持领先。
- 引入 **margin regularization**（铰链损失）进一步提升了长文本下的稳定性，在 90 个测试点中 **全部取得最高吞吐**。

---

## 4. 关键结论和发现

### 主要发现
1. **单次前向传递 + 并行矩阵运算** 是实现高效 draft-time 相关性建模的关键，LiLiCorr 在速度和效果之间取得了最佳平衡。
2. **贪心解码优于全局最优搜索**，因为 speculative decoding 的 prefix 结构决定了早期决策最重要。
3. **联合训练 drafter 与 reranker** 能使 drafter 学会生成更易被相关性模型接受的候选集合。
4. **LiLiCorr 具有强泛化能力**，即使在训练未见的多语言任务和超长输入上也表现优异。

### 方法的局限性
1. **依赖 top-K 覆盖率**：若正确 token 不在 top-K 候选中，则无法恢复。acceptance length 的上限受 drafter 的 marginal 准确率限制。
2. **仅建模相邻依赖**：当前方法只耦合相邻 slot，未能捕捉更长距离的依赖关系。
3. **贪心策略非全局最优**：虽然贪心在实践中更优，但仍可能错过更长的可接受前缀。

### 未来工作方向
- 探索 **非相邻位置的耦合机制**，以捕捉更长程依赖。
- 设计 **更鲁棒的覆盖机制**，避免因 top-K 丢失正确 token 导致的失败。
- 将 LiLiCorr 思路推广至 **其他并行生成任务**，如图像或音频生成中的 speculative sampling。

--- 

> **一句话总结**：LiLiCorr 通过轻量化的向量对齐机制，在 **一次前向传递** 内高效恢复并行 draft 的序列连贯性，显著提升 speculative decoding 的吞吐量，成为当前 **draft-time correlation** 的 SOTA 方法。

</details>

---

### 3. [Consilience: Conformally Calibrated Communication Control for Hidden-Profile Multi-Agent Reasoning](https://arxiv.org/abs/2608.20564)

**Authors**: Abhijith Babu, Ramneet Kaur, Vishal Pramanik, Olivera Kotevska, Nathaniel D. Bastian, Susmit Jha, Sunny Raj, Yanzhao Wu, Sumit Kumar Jha, Anirban Roy  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.20564v1  

#### Abstract
Multi-agent LLM systems can improve reasoning by pooling diverse perspectives, but their effectiveness depends on coordinating communication, particularly in hidden-profile settings where each agent holds only part of the evidence required for a correct decision. Existing protocols, including fixed ...

---

### 4. [Dual-Cache Latent Space Communication between Heterogeneous Language Models](https://arxiv.org/abs/2608.20617)

**Authors**: Jiyao Liu, Qi Zhang, Yaoyi Jia, Ziwen Kan, Song Wang  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.20617v1  

#### Abstract
Multi-agent LLM systems split work across models, so answering often requires knowledge that sits in another agent's context: a Sharer has encoded information that a Receiver needs to complete its task. They usually communicate by exchanging text, which puts autoregressive decoding on the critical p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Dual-Cache Latent Space Communication between Heterogeneous Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在多智能体大语言模型（LLM）系统中，不同模型（Sharer 和 Receiver）持有互补的上下文信息，需要高效地进行跨上下文通信以完成任务。传统方法依赖**文本级通信（Text-to-Text, T2T）**，存在以下瓶颈：
- 需要 autoregressive 解码生成消息，引入延迟；
- 接收方需重新编码文本，效率低下；
- 信息传递是离散且单向的，无法感知接收方状态。

近期提出的**潜空间通信**方法（如 C2C、LCF-X）通过直接传递 Key-Value (KV) 缓存来避免文本解码-编码循环，但仍存在三大限制：
1. **发送方独立压缩（Sender-only compression）**：仅基于 Sharer 的缓存生成摘要，未考虑 Receiver 的上下文。
2. **层局部单一摘要（Layer-local single summary）**：每层所有位置共享同一个外部摘要，缺乏细粒度更新能力。
3. **同构假设（Matched geometry assumption）**：要求模型具有相同的层数、注意力头数等结构参数，难以支持异构模型对。

### 提出了什么新方法或新思路
本文提出 **XKV**，一种**双缓存潜空间通信协议**，实现异构语言模型之间的高效、接收方感知的 KV 缓存传输。

#### 核心创新点：
- ✅ **对称直接缓存池化（Symmetric Direct Cache Pooling）**  
  同时从 Sharer 和 Receiver 的 KV 缓存中提取 `k` 个候选摘要（slots），决定“**传递什么信息**”时联合参考双方状态。

- ✅ **跨层联合记忆构建（Cross-layer Joint Memory）**  
  引入**可学习的层映射（learned layer map）** 对齐不同深度的模型，并通过 self-attention 将多层摘要融合为一个紧凑的**联合记忆（joint memory）**，支持跨层信息整合。

- ✅ **接收方位置查询机制（Receiver-position Retrieval）**  
  每个 Receiver 缓存位置作为 query 去访问联合记忆，输出该位置专属的、符合其原生 KV 几何结构的残差更新（residual），实现“**写到哪里**”的精细化控制。

- ✅ **完全异构支持与冻结模型兼容**  
  Sharer 和 Receiver 可来自不同家族（Qwen/Gemma/Llama）、不同深度、不同 KV 头数、不同 tokenizer；仅训练轻量级 translator 模块，主干模型保持 frozen。

### 相比现有方法的优势
| 特性 | T2T | LCF-X | XKV |
|------|-----|--------|-------|
| 是否需 autoregressive 解码 | ✅ 是 | ❌ 否 | ❌ 否 |
| 是否感知 Receiver 状态 | ❌ 否 | ❌ 否 | ✅ 是 |
| 是否支持异构模型 | ✅ 是 | ❌ 否（原始版本） | ✅ 是 |
| 是否支持不同层数/KV几何 | ❌ 不适用 | ❌ 否 | ✅ 是 |
| 每个位置是否获得独特更新 | ❌ 否 | ⚠️ 有限（依赖 projector） | ✅ 是 |
| 通信延迟 | 极高 | 中等 | 极低 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
在五个**分裂证据推理（split-evidence reasoning）** 数据集上评估，涵盖生成式问答与分类任务：

| 数据集 | 类型 | 任务描述 |
|--------|------|----------|
| **ROPES** | Generative QA | 给定背景+情境，回答需推理的问题 |
| **MuSiQue** | Generative QA | 多跳问答，需组合多个支持段落 |
| **HotpotQA-bridge** | Generative QA | 多跳问答，强调桥接实体推理 |
| **QASC** | Classification | 选择正确答案，需句子组合推理 |
| **StrategyQA** | Classification | 回答隐含策略的问题，需多步推理 |

所有数据均将支持证据**分割给 Sharer 和 Receiver**，确保任一方单独无法可靠作答。

### 实验设置和评估指标

#### 模型配置
使用三个不同家族的 decoder-only 模型构成 **3×3 有序配对网格**：
- **Qwen-0.6B**
- **Gemma-1B**
- **Llama-3.2-3B**

共 9 种组合（3 同构 + 6 异构），全面覆盖：
- 不同模型家族
- 不同层数（L）
- 不同 KV 头数（H）
- 不同 tokenizer

#### 评估指标
- **生成式 QA（ROPES, MuSiQue, HotpotQA）**：Exact Match (EM), F1
- **分类任务（QASC, StrategyQA）**：Accuracy (Acc.)
- **主指标聚合**：F1（生成）或 Acc（分类）的平均值作为“cross-dataset avg”

#### 效率指标
- **Communication Latency**：translator/fusor 模块执行时间（ms）
- **End-to-End Latency**：完整流程耗时（sharer前传 + communication + receiver生成）
- **Trainable Parameters**：可训练参数数量

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **T2T (Text-to-Text)** | Sharer 生成自然语言消息 → Receiver 接收并重编码后作答。代表当前主流 agent 通信方式。 |
| **LCF-X (Latent Cache Flow - Cross-context)** | 将 Sharer KV 缓存池化为位置无关摘要，注入 Receiver 作为门控残差。本文对其进行了扩展以支持异构模型（称为“projector-authored heterogeneous extension”）。 |
| **XKV (Ours)** | 本文提出的方法：基于双缓存池化 + 联合记忆 + 位置查询的潜空间通信。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（汇总于 Table 22 & 23）

| 方法 | Macro Score (F1/Acc) | Best Datasets | Best Cells (45) | Avg Rank |
|------|------------------------|---------------|------------------|-----------|
| LCF-X | 49.76 | 0/5 | 4/45 | 2.29 |
| **XKV (Ours)** | **52.37** | **4/5** | **31/45** | **1.37** |
| T2T | 48.86 | 1/5 | 11/45 | 2.34 |

> ✅ XKV 在 **45 个设置中取得 31 次最优**，平均排名第一。

### 与基线方法的对比结果

#### 性能提升（vs. LCF-X）
- 在**所有五个数据集上均优于 LCF-X**
- 平均提升 **+2.61 pts**，最高在 ROPES 上达 **+4.2 F1**
- 在 ROPES 上具体表现：**+4.6 EM, +4.2 F1**

#### 性能提升（vs. T2T）
- 在 **4/5 数据集上优于 T2T**
- 仅在 QASC 上略低 **-0.27 Acc**
- 在 ROPES 上领先 **+7.7 F1**，HotpotQA 上 **+6.4 F1**

#### 效率优势（Table 23）
| 指标 | LCF-X | XKV | 提升倍数 |
|------|--------|--------|------------|
| Communication Latency | 59.9 ms | **5.8 ms** | **10.3× 更快** |
| End-to-End Latency | 227.9 ms | **167.6 ms** | **26.4% 更快** |
| vs. T2T E2E | — | — | **6.8× 快于 T2T** |
| Trainable Params | 19.03M | **4.55M** | **减少 76.1%** |

> 💡 即使在性能更强的情况下，XKV 依然实现了**更低延迟、更少参数、更高吞吐**。

### 消融实验结果（Ablation Study）

在 QASC、ROPES、StrategyQA 上进行消融（Table 25）：

| 变体 | △Score | E2E ↓ | Comm ↓ | 参数 ↓ |
|------|--------|--------|--------|--------|
| **Full XKV** | 0.00 | 131.7 ms | 5.8 ms | 4.55M |
| **-RP**（移除 Receiver Pooling） | -0.46 | -3.0 ms | -1.1 ms | -0.97M |
| **-RX**（移除位置查询） | **-0.83** | -4.6 ms | **-2.1 ms** | -1.12M |
| **LCFP**（替换为 LCF-X 式两阶段池化） | -0.94 | +52.0 ms | +33.3 ms | -1.02M |

#### 发现：
- 移除 **receiver-aware pooling** 或 **position-specific retrieval** 均导致性能下降，说明两者有效；
- 但带来的延迟增加极小（< 3.5%），表明“接收方感知”几乎**零代价**；
- 使用 LCF-X 的 span pooling 替代 direct pooling 会导致**速度变慢 6.7× 且精度下降**，证明 direct pooling 不仅高效，也更利于建模。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **接收方感知通信显著提升性能**：联合建模 Sharer 和 Receiver 的缓存状态，能更精准决定“传递什么”，优于仅依赖发送方摘要的传统方式。
2. ✅ **位置级更新优于全局广播**：允许每个缓存位置独立查询联合记忆，实现细粒度、上下文敏感的信息融合。
3. ✅ **潜空间通信可以同时做到高性能与高效率**：XKV 打破了“高质量通信必然高成本”的权衡，在提升准确率的同时大幅降低延迟和参数量。
4. ✅ **异构模型间可建立通用通信接口**：通过 learned layer map 和 shared decoder，XKV 成功连接了不同架构、不同 tokenizer 的模型对。

### 方法的局限性
- 当前设计仍假设两个模型处理的是同一问题的不同部分，尚未支持多轮或多跳通信。
- 联合记忆长度固定为 $L_R$（Receiver 层数），可能限制深层交互表达能力。
- 实验集中在中小规模模型（≤3B），在更大模型上的扩展性有待验证。
- 未提供对潜空间消息的可解释性分析，即“究竟传递了什么语义”。

### 未来工作方向
- 扩展至 **multi-turn communication** 和 **many-agent collaboration**
- 探索 **scalability to larger models**（如 70B 级别）
- 研究 **interpretable latent messages**：理解潜空间中传递的知识形式
- 应用于 **tool calling**, **retrieval-augmented generation**, **agent societies** 等复杂系统

---

> 📌 **一句话总结**：  
> **XKV 提出了一种全新的双缓存潜空间通信范式，首次实现了接收方感知、位置特异性、完全异构兼容的高效模型间通信，在性能、速度、参数量上全面超越文本通信与现有潜空间方法，打破了质量与效率的权衡。**

</details>

---

### 5. [Enabling Memory-efficient Im2win Convolution with Multi-precision Support on GPU CUDA and Tensor Cores](https://arxiv.org/abs/2608.20725)

**Authors**: Xiang Fu, Jixiang Ma, Xinpeng Zhang, Peng Zhao, Shuai Lu, Xu Tony Liu  
**Category**: cs.DC  
**Published**: 2026-08-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.20725v1  

#### Abstract
Convolution is a principal computational bottleneck in deep neural networks, and its efficiency depends on tight integration between algorithms and GPU hardware. Existing GPU convolution methods suffer from large memory overhead, poor cache utilization, limited effectiveness across kernel sizes, or ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Enabling Memory-efficient Im2win Convolution with Multi-precision Support on GPU CUDA and Tensor Cores*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统卷积实现（如 im2col-based GEMM、direct convolution、Winograd、FFT）在 GPU 上存在以下瓶颈：
- **高内存开销**：im2col 需要显式展开输入为大矩阵，导致显著的内存冗余；
- **缓存利用率低**：direct convolution 存在非连续内存访问模式；
- **精度不稳定或适用范围有限**：Winograd 在大 kernel 下数值不稳定，FFT 对小 kernel 效率差；
- **缺乏对 Tensor Cores 的高效支持**：多数方法未充分利用现代 GPU 的混合精度加速能力。

### 🚀 提出的新方法与思路
本文将 **im2win** 卷积范式扩展至现代 GPU 架构，提出一种**统一、高性能、多精度支持**的卷积框架：
- **im2win 范式回顾**：将输入张量重排为 `im2win tensor`，使得滑动窗口内的元素在内存中连续存储，从而实现**紧凑的数据布局**和**高效的元素复用**；
- **多精度支持**：
  - 在 **CUDA Cores** 上实现 **FP32 版本**；
  - 在 **Tensor Cores** 上实现 **FP16 版本**，利用 WMMA API 执行高效的 matrix multiply-accumulate（MMA）操作；
- **关键优化技术**：
  - **Zig-zag memory access**：减少共享内存 bank conflict；
  - **Asynchronous data movement**：通过 PTX 指令异步加载数据，隐藏传输延迟；
  - **Double buffering**：构建计算与通信流水线；
  - **Index precomputation**：在 host 端预计算索引并存入 constant memory，避免设备端重复计算。

### 🔍 相比现有方法的优势
| 方面 | im2win 优势 |
|------|-------------|
| **内存效率** | 内存占用仅为 cuDNN 的 ~53%，cuBLAS_im2col 的 ~35%；优于所有主流方法（包括 Winograd 和 FFT） |
| **性能（TFLOPS）** | 在 Tensor Cores 上比 cuDNN 快 **1.4×**，比 cuBLAS_im2col 快 **6.4×**；在 CUDA Cores 上也显著优于基线 |
| **通用性** | 支持任意 kernel size、stride 和 padding，无 Winograd 的数值不稳定性问题 |
| **硬件适配性** | 充分利用 Tensor Cores 的 FP16 MMA 能力，并结合线程块调度、共享内存管理等底层优化 |

---

## 2. 核心实验方法和设置

### 📊 数据集与基准测试
使用一个包含 **12 个典型卷积层** 的 DNN benchmark（cv1–cv12），覆盖多种输入/输出尺寸、通道数和 kernel 配置，确保评估多样性：

| 层名 | 输入 Ci×Hi×Wi | Filter & Stride | 输出 Co×Ho×Wo |
|------|----------------|------------------|----------------|
| cv1 | 3×227×227 | 96×11×11,4 | 96×55×55 |
| cv7 | 3×224×224 | 64×3×3,1 | 64×222×222 |
| cv12 | 512×7×7 | 512×3×3,1 | 512×5×5 |
> （详见 Table I）

这些层来自经典模型（如 AlexNet、ResNet 组件），代表实际应用场景中的多样化卷积模式。

### ⚙️ 实验设置
- **硬件平台**：
  - GPU: NVIDIA GeForce RTX 3090（Ampere 架构，第三代 Tensor Cores）
  - CPU: Intel Xeon Silver 4214
  - CUDA 11.3 + PyTorch 2.2.0
- **评估指标**：
  - **TFLOPS**：衡量计算吞吐率；
  - **MaxRSS**（最大驻留集大小）：反映峰值内存消耗；
- **运行方式**：
  - Batch size $N=256$；
  - 每个 benchmark 运行 50 次，取最高 TFLOPS 和最低 MaxRSS。

### 🆚 基线方法对比
| 类别 | 方法 |
|------|------|
| **GEMM-based** | PyTorch + cuBLAS（im2col 实现） |
| **cuDNN 内建算法** | 
| - CUDA Cores | im2col (IPG), FFT (FT), Winograd (WN) |
| - Tensor Cores | IPG_TC, WN_TC（自动选择最优） |
| **本文方法** |
| - CUDA Cores | im2win_CC（FP32） |
| - Tensor Cores | im2win_TC（FP16，含多种优化变体） |

> 注：“CC” 表示 CUDA Core + FP32，“TC” 表示 Tensor Core + FP16

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（见 Figure 2 & 3）

#### ✅ 计算性能（TFLOPS）
| 对比项 | 加速比 |
|--------|--------|
| im2win_TC vs im2win_CC | **2.7×** 更高 TFLOPS（得益于 FP16 和 Tensor Core 并行性） |
| im2win_TC vs cuDNN_TC | **1.4×** 更高平均 TFLOPS |
| im2win_TC vs cuBLAS_im2col_TC | **6.4×** 更高 TFLOPS |

> 在 12 个 benchmark 中，im2win_TC 在 **8 个上达到最高性能**

#### 💾 内存使用（MaxRSS）
| 对比项 | 内存占比 |
|--------|---------|
| im2win_TC vs cuDNN_TC | 平均仅需 **53%** 的内存 |
| im2win_TC vs cuBLAS_im2col_TC | 平均仅需 **35%** 的内存 |
| im2win_CC vs cuDNN_CC | 几乎在所有 case 中更低（除 cv4 外） |

> 显著降低中间张量存储需求，尤其适合大 batch 或大模型训练场景

### 🔬 微观对比分析（Figure 3）

| 方法对比 | 结果总结 |
|--------|----------|
| **vs cuBLAS_im2col** | 
| - TFLOPS 提升：3.4× (CC), 6.4× (TC) |
| - 内存节省：~62% (CC), ~55% (TC) |
| **vs cuDNN_IPG** |
| - CUDA Cores 上性能相当 |
| - Tensor Cores 上 im2win_TC **快 2×**，归功于 zig-zag 和异步传输优化 |
| **vs cuDNN_WN** |
| - TFLOPS 提升：2.5× (WN_CC), 2.8× (WN_TC) |
| - 内存节省：仅需 WN 的 **27% (CC)** 和 **37% (TC)** |
| **vs cuDNN_FT** |
| - TFLOPS 提升：**2.0×** |
| - 内存节省：仅需 FT 的 **49%**

> im2win 在绝大多数情况下实现了“**更高性能 + 更低内存**”的双重优势

### 🔍 消融实验（Ablation Study, Figure 4）

在 Tensor Cores 上对三种核心优化进行消融研究：

| 优化技术 | 性能影响 | 分析 |
|--------|--------|------|
| **Double Buffering** | 贡献最大 | 有效隐藏数据加载延迟，在大多数 benchmark 上带来显著提升 |
| **Asynchronous Data Movement** | 第二重要 | 利用 PTX 异步指令进一步提升并发性 |
| **Zig-zag Access** | 提升最小 | 已被前两者部分缓解，但仍有助于减少 bank conflict，尤其在大 thread block 场景下 |

> **例外情况**：对于小卷积窗口（如 cv1–cv3），double buffering 可能引入额外同步开销，反而轻微降低性能

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **im2win 是一种统一且高效的卷积范式**：
   - 兼具低内存开销、高缓存命中率和良好的可扩展性；
   - 成功适配到 CUDA Cores 和 Tensor Cores，支持多精度运算；
2. **Tensor Core 上的 FP16 实现带来巨大性能飞跃**：
   - 利用 WMMA 指令 + 高效数据流设计，充分发挥硬件潜力；
3. **优化策略具有明确优先级**：
   - **Double buffering > Asynchronous movement > Zig-zag access**；
   - 多重优化协同作用，形成高效 producer-consumer 流水线；
4. **内存效率全面领先**：
   - 无论对比 cuDNN 还是 cuBLAS，im2win 均以更少内存完成相同任务，特别适用于显存受限场景。

### ⚠️ 方法的局限性
- **特定小 kernel 场景下 double buffering 可能失效**：由于额外开销超过收益；
- **当前实现依赖 Ampere 及以上架构特性**（如 WMMA、PTX 异步指令），向后兼容性需调整；
- **未探索 sparsity 或量化集成**：未来可结合稀疏计算进一步压缩内存和计算负载。

### 🔮 未来工作方向
1. 将 im2win 扩展至 **3D convolution** 和 **attention-based models**（如 Vision Transformers）；
2. 探索 **sparse im2win tensor** 表示，结合结构化剪枝或量化；
3. 实现 **auto-tuning framework** 自动选择最优 block shape 和优化组合；
4. 移植至其他硬件平台（如 AMD GPU、AI 加速器）验证通用性。

---

> **总结一句话**：  
> 本文提出的 **multi-precision im2win convolution** 在 **GPU 上实现了高性能与极致内存效率的统一**，是面向现代深度学习系统的极具前景的底层算子解决方案。

</details>

---

### 6. [Self-Speculation for Faster Reasoning Models](https://arxiv.org/abs/2608.20359)

**Authors**: Ravisri Valluri, Tung Nguyen, Aditya Grover  
**Category**: cs.CL  
**Published**: 2026-08-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.20359v1  

#### Abstract
Large language models (LLMs) are deployed for increasingly complex tasks involving planning and multi-step decision making, but high-quality performance on these tasks often requires generating long reasoning traces. This is a poor fit for latency-sensitive and interactive applications like voice as...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Self-Speculation for Faster Reasoning Models》核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在需要复杂推理的任务（如规划、编码、决策）中表现优异，但通常依赖于生成长链的思维链（Chain-of-Thought, CoT），导致**端到端生成延迟高**。这在对延迟敏感的应用（如语音助手、交互式编程代理）中严重影响用户体验。

现有加速方法（如 speculative decoding）多关注 token 级别的并行生成，未充分利用推理任务中 CoT 的**结构化特性**，难以实现显著提速。

### 提出的新方法：SSR（Self-Speculation for Reasoning Models）
作者提出 **SSR**，一种无需训练的 self-speculative decoding 方法，其核心思想是：
- 利用同一模型在不同推理预算（reasoning budget）下的输出作为“草稿”（drafter）和“验证者”（verifier）：
  - **草稿分布**：由部分 CoT（partial CoT）生成的答案分布 $p_\theta(a|q, b_d)$ 构成。
  - **验证分布**：由完整 CoT 生成的答案分布 $p_\theta(a|q, b_v)$ 构成（$b_d < b_v$）。
- 草稿生成与主推理过程**并发执行**，隐藏了草稿生成的开销。

### 相比现有方法的优势
| 特性 | SSR | 传统 Speculative Decoding | Self-Speculative (e.g., Early Exit) | Model-Free (e.g., Suffix Decoding) |
|------|-----|--------------------------|----------------------------------|-------------------------------|
| 是否需额外模型 | ❌ | ✅（小型 draft model） | ❌ | ❌ |
| 是否需训练 | ❌ | ✅（需训练 draft head） | ⚠️（部分需训练） | ❌ |
| 是否利用 CoT 结构 | ✅（直接利用 partial CoT） | ❌ | ❌ | ❌ |
| 可生成新内容 | ✅ | ✅ | ✅ | ❌（仅能复用已有文本） |
| 输出质量保证 | ✅（exact verification） | ✅ | ✅ | ❌（greedy verification） |

此外，SSR 还引入了两项增强机制：
- **Suffix Decoding**：在标准前缀验证后，利用草稿构建 suffix cache，在首次拒绝后仍可恢复匹配的后续片段，提升长文本重用率。
- **Iterative SSR**：在多个推理预算点采样草稿，并利用早期草稿引导后期草稿，减少多点采样的开销。

---

## 2. 核心实验方法和设置

### 数据集
实验聚焦于**长格式、结构化生成任务**，主要包括：
- **ClassEval**：类级别代码生成基准，答案较长。
- **HumanEval**：函数级别代码补全，答案较短。
- **LongProc 2K**：长流程生成任务集合（如 `countdown`, `html_to_tsv`），目标输出约 2K tokens。

### 模型
评估了两个开源的 4B 级别推理模型：
- **Qwen3.5-4B**
- **Gemma-4-E4B-it**

### 实验设置
- **框架**：基于 vLLM 实现 SSR，作为即插即用模块集成。
- **推理配置**：
  - CoT 预算（verifier）：2000 tokens
  - 草稿预算（draft budget）：500 tokens
  - 最大草稿长度：500 tokens
- **评估指标**：
  - **端到端延迟**（End-to-end latency）
  - **相对延迟降低**（Relative latency reduction）
  - **接受的前缀 token 数**（Prefix tokens）
  - **通过 suffix 接受的 token 数**（Suffix tokens）
- **基线方法**：标准自回归生成（naive autoregressive generation）。

---

## 3. 主要实验结果和性能指标

### 整体性能对比（Table 1）
| Model | Benchmark | 改进幅度 | SSR 延迟 (s) | 基线延迟 (s) |
|-------|-----------|--------|-------------|------------|
| Qwen3.5-4B | LongProc 2K | **9.1%** | 68.9 | 72.8 |
| Qwen3.5-4B | ClassEval | **18.5%** | 72.8 | 85.3 |
| Qwen3.5-4B | HumanEval | 2.9% | 25.2 | 26.4 |
| Gemma-4-E4B-it | LongProc 2K | 7.1% | 97.4 | 103.1 |
| Gemma-4-E4B-it | ClassEval | **24.1%** | 59.9 | 79.2 |
| Gemma-4-E4B-it | HumanEval | 14.6% | 35.5 | 41.8 |

> ✅ **最高实现 24.1% 的延迟降低**，在 ClassEval 上效果最显著。

### 消融实验（Table 2）
在 `ClassEval + Gemma-4-E4B-it` 上的消融研究：
| 配置 | 延迟加速倍数 | 前缀 token | 后缀 token |
|------|-------------|----------|----------|
| 仅 Prefix Verification | 1.086× | 235.7 | 0.0 |
| 仅 Suffix Decoding | 1.318× | 0.0 | 867.8 |
| SSR（Prefix + Suffix） | **1.318×** | 238.9 | 637.8 |

> 🔍 发现：
> - **Suffix decoding 贡献巨大**：即使前缀完全不匹配，suffix decoding 仍可通过 exact lexical match 恢复大量有效文本。
> - 前缀验证失败 ≠ 完全无用，为 suffix recovery 提供素材。

### 多阶段迭代变体（Iterative SSR）
| 设置 (i=间隔, m=最大长度) | 多草稿总耗时 | Iterative SSR 耗时 | 减少 |
|------------------------|-------------|------------------|------|
| i=750, m=500 | 18.58s | **16.37s** | 11.9% |
| i=500, m=250 | 17.75s | **17.16s** | 3.3% |

> ✅ Iterative SSR 显著降低了多草稿生成的计算开销，尤其在长间隔下更有效。

---

## 4. 关键结论和发现

### 主要发现
1. **SSR 在长且结构化的输出任务上最有效**：
   - 当答案长度与 CoT 长度相当时，草稿重用带来的收益可覆盖并发开销。
   - 因此在 **ClassEval 和 LongProc** 上提速明显（最高达 24.1%），而在 **HumanEval**（短答案）上提升有限。

2. **Suffix decoding 至关重要**：
   - 即使草稿与最终答案在开头有小差异（如变量命名），后续仍可能高度一致。
   - 标准 speculative decoding 会在首个不匹配 token 处终止，浪费大量潜在可用文本。
   - SSR 的 suffix decoding 成功回收这些片段，是性能提升的关键。

3. **并发设计高效实用**：
   - 草稿生成与主 CoT 推理并行，其计算开销被有效隐藏。
   - 实验显示 CoT 吞吐量下降极小（< 5.4%），证明实现高效。

### 局限性
- **仅加速答案生成，不加速 CoT 本身**：当 CoT 生成时间占主导时（如短答案任务），整体提速受限。
- **依赖 lexical overlap**：在高熵任务中，即使语义相似，若表面形式差异大（如 paraphrasing），prefix/suffix 重用率会下降。
- **需足够长的 CoT**：要求总 CoT 长度 > 草稿预算 + 最大草稿长度，否则无法启动并发。

### 未来工作方向
1. 将 SSR 思想扩展至**加速 CoT 本身的生成过程**。
2. 通过微调（fine-tuning）让模型在推理早期就**更早地确定输出结构**，提高 partial CoT 的预测准确性。
3. 探索结合 semantic similarity 的草稿匹配机制，超越纯 lexical matching。

> 💡 **总结**：SSR 是一种巧妙利用推理模型内在结构（CoT progressive refinement）的训练免费加速方案，通过并发 + 前缀验证 + 后缀恢复三重机制，在结构化长文本生成中实现了高达 24.1% 的延迟降低，为实际部署提供了高效的即插即用优化手段。

</details>

---

### 7. [Memory Augmentation Unlocks Efficient Chain-of-Thought Reasoning](https://arxiv.org/abs/2608.21265)

**Authors**: Simeng Zhang, Yilong Chen, Wenyuan Zhang, Zhenyu Zhang, Yao Chen, Junyuan Shang, Tingwen Liu  
**Category**: cs.CL  
**Published**: 2026-08-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.21265v1  

#### Abstract
Large language models often rely on Chain-of-Thought (CoT) reasoning to solve complex tasks, but verbose reasoning traces introduce substantial inference overhead. CoT compression shortens generation, yet aggressive compression may disrupt logical coherence and degrade performance. We formalize this...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Memory Augmentation Unlocks Efficient Chain-of-Thought Reasoning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）在复杂任务上依赖 **Chain-of-Thought (CoT)** 推理来提升性能，但生成冗长的推理链会带来显著的 **inference overhead**（解码延迟、token 成本高）。虽然已有方法尝试通过压缩推理链（如缩短、剪枝）来提高效率，但过度压缩会导致逻辑连贯性破坏和准确率下降。

本文提出并形式化了一个核心权衡问题：  
> **如何在不牺牲推理质量的前提下，减少 decode-time 的生成开销？**

### **提出的新方法与新思路**
作者提出了 **Memory-Augmented Compression (MAC)**，一种无需训练的框架，其核心思想是：
- 将历史推理轨迹中提取出的可复用推理模式（如关键约束、子目标、操作步骤）构建成 **显式记忆（explicit reasoning memory）**。
- 在推理时，将这些记忆作为 **prefill-side scaffolds** 注入输入上下文，辅助模型进行短链推理（Short-CoT），从而补偿压缩带来的信息损失。

#### **核心理论贡献：Context-Generation Substitution Law**
首次形式化了以下权衡关系：
> 显式的 **prefill-side reasoning context** 可以替代部分 **decode-time generation**，只要其预填充成本低于所节省的解码开销。

该定律为高效推理提供了理论基础：利用更易并行化的 **prefill** 阶段，替代串行瓶颈的 **autoregressive decoding**。

### **相比现有方法的优势**
- ✅ **无需训练**：完全基于检索和提示工程，适用于任何现成 LLM。
- ✅ **通用兼容性**：可即插即用地增强多种压缩机制（prompt-based, token-level, KV-cache）。
- ✅ **高效且准确**：在大幅降低延迟的同时，显著恢复甚至超越标准 CoT 的准确率。
- ✅ **模块化设计**：记忆构建、检索器、压缩器三者解耦，便于独立优化。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
涵盖多个复杂推理领域：
- **数学推理**：`GSM8K`, `MATH`, `AIME 2024`
- **综合复杂推理**：`BBH`（Big-Bench Hard）
- **科学问答**：`MMLU-Sci`（涵盖物理、化学、生物等学科）

所有记忆库均来自与测试集无交集的历史样本（如训练集、往届竞赛题），防止数据泄露。

### **实验设置与评估指标**
- **主干模型**：
  - 开源模型：`LLaMA-3.1-8B`, `Qwen2.5-7B`, `Qwen2.5-72B`
  - API 模型：`DeepSeek-V3.2`, `Qwen3.5-plus`, `o4-mini`
- **压缩基线方法**：
  - `CoT`（标准长链推理）
  - `CoD`（Chain-of-Draft，prompt 压缩）
  - `TokenSkip`（token 级跳过）
  - `RPC`（Reasoning Path Compression）
  - `Extra-CoT`（预算感知压缩）
- **评估指标**：
  - **准确率（Accuracy）**：Exact match 或 pass@k
  - **效率指标**：`prefill tokens`, `decode tokens`, `total tokens`, `latency (ms)`
  - **速度比**：相对于标准 CoT 的端到端加速倍数

### **基线对比方式**
- 主要对比 `CoD` 与 `CoD + Memory`
- 消融实验分析不同记忆格式、检索策略、内存大小的影响
- 跨模型、跨压缩机制验证通用性

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**
在 `Qwen2.5-7B` 上的结果显示，**MAC 显著提升了压缩推理的准确性，同时保持低延迟**：

| 数据集       | 方法         | 准确率 | ▲ vs CoD | 延迟 (vs CoT) |
|------------|--------------|--------|----------|----------------|
| **GSM8K**  | CoD          | 67.9   | —        | 5.61× 更快     |
|            | **CoD + Memory** | **89.3** | **+21.4** | **1.49× 更快** |
| **MATH**   | CoD          | 43.0   | —        | 4.57× 更快     |
|            | **CoD + Memory** | **71.0** | **+28.0** | **1.14× 更快** |
| **BBH**    | CoD          | 41.0   | —        | 6.14× 更快     |
|            | **CoD + Memory** | **70.5** | **+29.5** | **1.27× 更快** |
| **MMLU-Sci**| CoD         | 60.35  | —        | 8.42× 更快     |
|            | **CoD + Memory** | **66.96** | **+6.61** | **1.41× 更快** |

> 💡 **结论**：MAC 在所有任务上均实现 **>20 pts 的准确率提升**，且仍比标准 CoT 快 **1.14–1.49×**。

### **与其他压缩机制的兼容性（Table 1b）**
MAC 可作为“插件”提升多种压缩方法：
- 在 `TokenSkip` 上提升 `+2.13 ~ +17.51` pts
- 在 `RPC` 和 `Extra-CoT` 上也取得正向增益
- 表明其补偿机制具有广泛适用性

### **消融实验结果**
#### **(1) 检索策略对比（Table 2）**
| 检索方式       | 准确率 (%) | 开销 (ms) |
|----------------|-----------|----------|
| Random         | 50.20     | 0.1      |
| BM25           | 54.60     | 1        |
| Query Embedding| 53.20     | 100      |
| **Reasoning Tag** | **56.20** | **683**  |

> ⚠️ **Trade-off**：`Reasoning Tag` 最准但最慢；`BM25` 是轻量级替代方案。

#### **(2) 记忆表示形式（Table 3）**
| 内容类型       | 准确率 | ▲Acc |
|----------------|-------|------|
| Few-shot       | 46.67 | -8.33 |
| Long-CoT trace | 43.33 | -11.67 |
| **Summary**    | **61.67** | **+6.67** |

> ✅ **抽象摘要优于原始演示**：说明有效的是 **推理结构** 而非单纯增加上下文长度。

#### **(3) 记忆数量 $k$ 的影响（Table 4）**
- 准确率随 $k$ 增加先升后降（最佳 $k=14$）
- $k=20$ 导致 prefill 成本剧增且准确率下降 → 存在 **coverage-noise trade-off**

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **显式记忆可有效替代部分 decode-time 推理**，支持 **Context-Generation Substitution Law**。
2. ✅ **MAC 显著改善了压缩推理的 accuracy-latency 权衡**，在多个任务上实现“又快又准”。
3. ✅ **收益来源于相关记忆的内容本身**，而非简单延长上下文。
4. ✅ **增益在强压缩下最为明显**（见 Figure 4），说明其作用是补偿信息丢失。
5. ✅ **方法具备良好的泛化能力**：跨模型、跨压缩机制、跨任务均有效。

### **方法的局限性**
1. 🔒 **依赖高质量记忆库**：若记忆不相关或错误，可能引入噪声，损害推理。
2. ⏱️ **检索开销不可忽略**：尤其是基于 LLM 的 tag 生成（~683ms/query），可能削弱小任务上的优势。
3. 📏 **API 模型接口限制**：无法精确测量 prefill/decode 延迟，影响跨模型公平比较。
4. 🧩 **对极短推理任务增益有限**：当原生 CoT 已很短时，prefill 成本可能超过收益。

### **未来工作方向**
- 构建更鲁棒的记忆检索机制（如抗噪、动态过滤）
- 探索轻量化 tag 生成方法以降低在线开销
- 结合 speculative decoding、KV-cache compression 等系统级优化
- 扩展至多模态推理场景中的 memory reuse

---

> **一句话总结**：  
> 本文提出的 **Memory-Augmented Compression** 通过将历史推理提炼为可复用的 **显式记忆** 并注入 prefill 阶段，成功实现了 **高效且准确的压缩推理**，为解决 CoT 的效率瓶颈提供了一条通用、无需训练的新路径。

</details>

---

### 8. [Bern2Edge: A Neurosymbolic Compiler for Edge Deployment via Bernstein Polynomial Networks](https://arxiv.org/abs/2608.20497)

**Authors**: Malak Gamal El-Din, Yifan Zhang, Yasser Shoukry, Sitao Huang, Salma Elmalaki  
**Category**: cs.LG  
**Published**: 2026-08-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.20497v1  

#### Abstract
Deploying high-accuracy neural networks on resource-constrained edge devices remains challenging, as existing approaches treat training, compression, and hardware synthesis as separate stages, leaving a gap between software-trained models and efficient end-to-end deployment with limited support for ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Bern2Edge: A Neurosymbolic Compiler for Edge Deployment via Bernstein Polynomial Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在资源受限的边缘设备上部署高精度神经网络面临多重挑战：
- 传统方法将**训练、压缩、硬件合成**作为独立阶段处理，导致软件模型与硬件实现之间存在“gap”。
- 模型难以兼顾**高精度、低延迟、低资源消耗**，且缺乏**可解释性**，限制了其在安全关键场景的应用。

### **提出了什么新方法或新思路**
本文提出 **Bern2Edge**，一个端到端的 **neurosymbolic compiler**，通过 **Bernstein Polynomial Networks (BNNs)** 实现高效的边缘部署。其核心创新在于：
- 利用 **knowledge distillation (KD)** 将预训练的教师网络（如 ReLU-based DNN）蒸馏为基于 **Bernstein polynomial 激活函数**的学生网络（BNN）。
- BNN 支持两条并行的部署路径：
  1. **LUT-based realization**：通过查找表（LUT）实现高保真、低延迟的硬件推理。
  2. **Symbolic rule-based representation**：从 BNN 的激活几何结构中提取紧凑的符号规则，实现**可解释推理**。

### **相比现有方法的优势**
| 方面 | Bern2Edge 的优势 |
|------|------------------|
| **压缩效率** | 在相同压缩约束下，BNN 比 ReLU 学生网络最高提升 **2.12 pp 准确率**。 |
| **硬件效率** | LUT-based 路径在 AMD Xilinx KV260 FPGA 上实现 **99.8% 延迟降低** 和 **95.2% BRAM 减少**，同时保持准确率损失 < 0.5 pp。 |
| **可解释性** | 首次利用 Bernstein 激活的几何结构进行**符号规则提取**，生成具有显式输入空间约束的规则，支持形式化验证。 |
| **无训练-部署差距** | LUT 实现完全匹配训练时的激活函数，消除近似误差。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Tabular 数据集**：
  - `HIGGS-Small` (28 features, binary)
  - `Covertype` (54 features, 7 classes)
  - `Adult Census` (14 features, binary)
  - `MAGIC Gamma Telescope` (10 features, binary)
- **分布偏移基准**：`ACS Income`（用于评估规则系统对地理/时间偏移的鲁棒性）
- **语言任务**：`SST-2`（用于评估在 **Transformer FFN 层**中的应用）

### **实验设置**
- **教师模型**：基于 [Gorishniy et al., 2021] 的 MLP 架构，使用 ReLU 激活。
- **学生模型**：使用 KD 训练 BNN，Bernstein 多项式度数为 3（MLP）或 15（Transformer FFN）。
- **硬件平台**：
  - 主要：**AMD Xilinx KV260 FPGA**
  - 低功耗验证：**Spartan-7 XC7S15 FPGA**
- **工具链**：Vitis HLS 2024.1 + Vivado 2024.1 进行综合与部署。

### **评估指标**
| 类别 | 指标 |
|------|------|
| **模型性能** | Test Accuracy (%), Cross-Entropy Loss |
| **硬件性能** | Latency (clock cycles), DSPs, BRAMs, LUTs, Flip-Flops (FFs) |
| **规则系统** | Rule Count, Avg. Conditions per Rule, Coverage (%), Covered Accuracy (%) |
| **鲁棒性** | Certified Stability (%) under input noise, Distribution Shift Robustness |

### **基线方法对比**
- **量化教师模型**：W8A8（8-bit weight & activation）QAT（Quantization-Aware Training）模型。
- **规则提取基线**：ECLAIRE [20]、DeepRED [19] 等符号规则提取方法。
- **其他编译器**：hls4ml [2]、FINN [3]、CGRA4ML [4]。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **(1) 模型压缩性能（vs. ReLU）**
- 在 `Covertype` 数据集上，BNN 在强压缩架构 `{54,64,32,7}` 下比 ReLU 学生网络**高出 2.12 pp** 准确率（91.09% vs. 88.97%）。
- 在所有数据集和架构下，BNN 均优于 ReLU，尤其在小模型容量时优势更明显。

#### **(2) 端到端硬件部署性能（vs. W8A8 教师）**
| 数据集 | 方法 | 准确率 (%) | 延迟降低 | DSP 减少 | BRAM 减少 |
|--------|------|------------|----------|----------|-----------|
| HIGGS-Small | Bern2Edge (LUT) | 72.3 | ↓98.2% | ↓94.0% | — |
| Covertype | Bern2Edge (LUT) | 96.4 | ↓91.9% | ↓16.2% | ↓94.2% (URAM) |
| Adult | Bern2Edge (LUT) | 84.6 | ↓99.8% | ↓74.7% | ↓95.2% |
| Adult | Bern2Edge (Rules) | 83.12 | ↓99.6% | ↓89.0% | ↓88.1% |

> ✅ **关键发现**：LUT 路径在保持高精度的同时实现**数量级的延迟降低**；规则路径进一步减少 DSP 使用，适合 DSP 受限场景。

#### **(3) 符号规则提取性能**
- 在 `MAGIC` 数据集上，Bern2Edge 提取的规则集仅需 **44±4 条规则**，而 ECLAIRE 需要 **396±75 条**，**减少 9 倍**。
- 平均每条规则条件数从 3.82（ECLAIRE）降至 **1.86**，显著提升简洁性。
- 规则对底层网络的保真度（Fidelity）达 **91.8%**，高于 ECLAIRE 的 89.4%。

#### **(4) 消融实验结果**
- **Sparsity threshold `k`**：当 `k ≥ 7` 时，准确率趋于稳定，内存占用仅为稠密版本的 ~55%。
- **Penalty 参数 `αsc`, `αconf`**：
  - `αsc` 控制覆盖-准确性权衡：增大 `αsc` 提升覆盖样本的准确性，但降低总覆盖率。
  - `αconf` 对性能影响较小，无需精细调参。
- **Fallback 策略**：**CART** 作为 fallback 在准确率、硬件成本和 uncovered 样本表现上均最优。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Bernstein 激活函数是高效且可解释的表示**：
   - 其**有界、凸组合**特性使其天然适合 LUT 实现和符号推理。
   - 在知识蒸馏下，BNN 能更有效地恢复大模型的知识，尤其在强压缩条件下。

2. **LUT-based 实现消除了训练-部署差距**：
   - 通过固定输入域 `[0,1]` 和离线 LUT 生成，实现了**精确的硬件映射**。
   - 结合线性插值，可在极小存储开销下达到接近全精度的推理效果。

3. **符号规则提取具有实际价值**：
   - 提取的规则不仅**紧凑、可解释**，还能提供**形式化保证**（如输入噪声下的稳定性认证）。
   - 在分布偏移下，规则覆盖率下降可作为系统退化的**可观测信号**，避免静默错误。

4. **方法具备良好的扩展性**：
   - 成功应用于 **Transformer FFN 层**，在 `SST-2` 任务中，Bernstein FFN 在隐藏层减半的情况下仍能匹配甚至超越原始 TinyBERT4 的准确率，并实现 **61.0% 的端到端延迟降低**。

### **方法的局限性**
1. **规则提取依赖于第一层输入特征**：
   - 当前规则提取仅适用于第一层神经元，无法直接从深层抽象特征中提取语义规则。
2. **多项式度数选择依赖经验**：
   - 虽然度数不影响 LUT 硬件开销，但最优度数仍需根据任务复杂度手动调整。
3. **当前主要针对 MLP 和 Transformer FFN**：
   - 尚未扩展到 CNN 或 RNN 等结构。

### **未来工作方向**
1. **自动化设计空间探索（DSE）**：
   - 自动搜索 `k`, `αsc`, `αconf` 等参数的帕累托最优配置。
2. **结合概念探针（Concept Probing）**：
   - 在语义上有意义的方向上进行规则提取，提升深层规则的可解释性。
3. **扩展至卷积网络（CNN）**：
   - 探索 Bernstein 激活在视觉任务中的潜力，已有研究表明其在表示能力上优于 ReLU。
4. **支持更多硬件后端**：
   - 如 ASIC、ASIC-like CGRA 等，进一步优化能效比。

---

> **总结**：Bern2Edge 是首个将 **Bernstein polynomial 激活**同时用于**高效硬件实现**和**符号规则提取**的端到端框架，在**精度、效率、可解释性**三者之间取得了卓越平衡，为边缘 AI 的可信部署提供了新范式。

</details>

---

### 9. [Tydra: An Efficient Hybrid Model for Tabular Data](https://arxiv.org/abs/2608.21199)

**Authors**: Mieszko Komisarczyk, Saurabh Mathur, Maurice Kraus, Sriraam Natarajan, Kristian Kersting  
**Category**: cs.LG  
**Published**: 2026-08-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.21199v1  

#### Abstract
Transformer-based tabular foundation models such as TabPFN achieve strong predictive performance but incur quadratic computational cost with context length. On the other hand, subquadratic SSM-based alternatives such as Hydra trade away accuracy for efficiency. To balance both, we introduce Tydra, a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Tydra: An Efficient Hybrid Model for Tabular Data*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **Transformer-based tabular foundation models**（如 TabPFN）虽然预测性能强，但其 **self-attention 机制导致推理复杂度为 $O(L^2)$**（$L$ 为样本数），在大规模或长上下文场景下计算成本过高。
- **纯 SSM 架构**（如 Hydra）虽实现亚二次甚至线性复杂度、效率高，但在小到中等规模数据上 **预测精度显著低于 TabPFN**。
- 因此，存在一个明显的 **accuracy-efficiency trade-off**：无法同时兼顾高性能与低延迟，尤其对医疗等需本地部署、资源受限的机构不友好。

### 🚀 提出的新方法与创新思路
- **首次将 hybrid Transformer-SSM 架构引入 tabular in-context learning 领域**，提出 **Tydra**。
- 核心设计是 **交替堆叠 Hydra 的 bidirectional SSM 层与 TabPFN 的 attention 层**，形成 `[Hydra → Transformer] × K` 的混合结构。
- 利用 SSM 的高效序列混合能力降低计算开销，同时保留 attention 的 content-dependent 建模能力以维持精度。

### 🔍 相比现有方法的优势
| 方法 | 推理速度 | 预测精度 | 备注 |
|------|--------|--------|------|
| **TabPFN** | 慢（$O(L^2)$） | 高 | 准确但昂贵 |
| **Hydra (16M)** | 快 | 较低 | 效率优先牺牲精度 |
| **Hydra (160M)** | 更慢 | 接近 TabPFN | 放大后仍不如且更慢 |
| **Tydra (16M)** | **快于 TabPFN（↓30% 时间）** | **接近 TabPFN** | ✅ 最佳平衡 |

> 💡 **核心优势**：在仅 16M 参数量下，达到接近 TabPFN 的准确率，同时推理时间减少约 30%，优于十倍大的 Hydra 模型。

---

## 2. 核心实验方法和设置

### 📚 数据集
#### （1）真实世界基准
- **OpenML-CC-18** 中的 **30 个二分类与多分类数据集**
  - 条件：≤2,000 样本，≤100 特征，≤10 类别
  - 每个数据集使用 **5 次 50/50 的确定性 train-test split**

#### （2）合成数据用于扩展性测试
- **Synthetic tabular classification datasets**
  - 规模从 **512 到 32,768（即 $2^9$ 到 $2^{15}$）样本**
  - 10 维标准正态数值特征，2 平衡类别
  - 无缺失值、无类别变量
  - 用于评估 **推理延迟随 context length 的扩展性**

### ⚙️ 实验设置
- 所有模型均通过 **meta-training on synthetic tasks** 进行训练（prior-data fitting 范式）
- 使用相同的 **synthetic task generator**（基于神经网络 prior）和优化配置：
  - Optimizer: AdamW
  - Learning rate: $10^{-4}$
  - Batch size: 64
  - Gradient clip: 1.0

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **AUROC** | 主要预测性能指标（平均 across splits） |
| **Inference Speed / Latency** | 总预测数量 ÷ 同步推理时间（ms/batch） |
| **Speedup vs. TabPFN** | 相对加速比 |
| **△AUROC** | 相对于 TabPFN 的性能变化 |

### 🆚 基线方法对比
| 模型 | 类型 | 参数量 | 说明 |
|------|------|--------|------|
| **TabPFN** | Pure Transformer | ~25.8M | 当前 SOTA tabular foundation model |
| **Hydra 16M** | Pure SSM | 16M | 高效但精度较低 |
| **Hydra 160M** | Pure SSM | 160M | 放大版 Hydra，验证“scaling 是否有效” |
| **Tydra family** | Hybrid (SSM + Attention) | 16M | 多种 layer ratio 和排列方式变体 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 及图示）

| Method | AUROC (%) | Inference Time (ms/batch) | △AUROC vs. TabPFN | Time Reduction |
|-------|-----------|----------------------------|-------------------|----------------|
| **TabPFN** | 88.90 | 27.12 | — | — |
| **Tydra {4HT}** (ours) | **88.61** | **19.99** | -0.29 | ↓7.13 ms (**↓26.3%**) |
| **Hydra 16M** | 87.66 | 23.41 | -1.24 | ↓3.71 ms |
| **Hydra 160M** | ~88.9 | >27.12 | ~0 | 更慢 |

> ✅ **结论**：Tydra 在保持几乎相同精度的同时，**推理速度快 24.8%（最高达 29.9%）**

### 🔁 与其他基线的对比结果
- **vs. TabPFN**：
  - 平均 AUROC 差距仅为 **0.29%**，最大差距 <1%
  - 推理时间 **显著缩短**，在多数数据集上提速超过 1.25×
- **vs. Hydra 16M**：
  - 精度高出 **~1.0 AUROC point**
  - 速度仍更快或相当
- **vs. Hydra 160M**：
  - 精度相近，但 **推理速度远超之**
  - 表明单纯放大 SSM 无法解决效率瓶颈

### 🔬 消融实验结果（Answer Q3）
评估了六种不同结构的 Tydra 变体（见 Figure 3 & Table 1）：

| 架构 | 特点 | AUROC | 推理时间 | 发现 |
|------|------|--------|----------|------|
| `{4HT}` (`H→T`) | 1:1 交替，H 先 | **88.61** | **19.99** | ✅ 最佳 trade-off |
| `{4TH}` (`T→H`) | 1:1 交替，T 先 | 88.41 | 23.41 | 略差于 {4HT} |
| `H{6T}H` | 两端 H，中间 6T | 88.73 | 21.98 | 精度高但不够快 |
| `{2H}{4T}{2H}` | 对称分布 | 88.71 | 22.57 | 平衡尚可 |
| `T{6H}T` | 中间 6H，两头 T | **87.66** | 23.59 | ❌ 精度下降明显（-1.24），尤其在 Dataset 50 上降 15 pts |
| `{2T}{4H}{2T}` | 多 SSM 居中 | 88.37 | 22.78 | 精度损失较大 |

> 🔍 **关键发现**：
> - **均衡交错（balanced interleaving）优于集中式堆叠**
> - **SSM 层不宜过度集中**（如 `T{6H}T` 导致显著精度下降）
> - **`{4HT}` 结构最优**：先 SSM 再 attention 可能更利于信息流动

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **Hybrid 架构是 tabular foundation models 的可行且优越路径**：
   - 成功结合了 SSM 的效率与 attention 的表达力。
2. **Tydra 实现了 accuracy 与 efficiency 的最佳平衡**：
   - 在 30 个 OpenML 数据集上，**推理时间减少 24.8%~30%**，AUROC 下降不到 0.3%。
   - 明显优于同规模及十倍大的 Hydra 模型。
3. **Scaling pure SSM 不足以弥补精度差距**：
   - 即使将 Hydra 扩大到 160M 参数，也无法在速度和精度上全面超越 TabPFN 或 Tydra。

### ⚠️ 方法的局限性
- 当前仅在 **小到中等规模 tabular 分类任务** 上验证，尚未拓展至回归、大规模表格或含类别特征/缺失值的复杂场景。
- **长上下文极限下（>32K 样本）**，Tydra 虽优于 TabPFN 和 Hydra 160M，但仍逐渐落后于轻量 Hydra 16M（见 Figure 4），表明仍有优化空间。
- 混合架构的设计空间尚未完全探索（如 layer sharing、adaptive switching）。

### 🔮 未来工作方向
1. **尝试其他 subquadratic sequence mixers**：
   - 如 Gated Linear Attention、Gated DeltaNet 替代 Hydra 层，寻找新的 trade-off 点。
2. **扩展至 long-context 和 large-scale tabular data**：
   - 测试在工业级数据上的表现。
3. **结合 looped transformers**：
   - 使用共享权重层循环执行，进一步压缩模型并提升效率。
4. **探索 adaptive hybridization 策略**：
   - 动态决定何时使用 attention 或 SSM。

---

> 🧩 **总体评价**：  
> *Tydra 是首个将 hybrid Transformer-SSM 架构成功应用于 tabular in-context learning 的工作，证明了“混合而非极端”的设计哲学在 tabular foundation models 中的巨大潜力，为后续高效、实用化的本地化部署提供了新范式。*

</details>

---

### 10. [TreeWY: Speculative Verification for Gated DeltaNet Hybrids](https://arxiv.org/abs/2608.20961)

**Authors**: Sneha Murthy Ghantasala  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.20961v1  

#### Abstract
Modern open models are hybrids: most layers are linear-attention (Gated DeltaNet, GDN) layers carrying a small fixed-size recurrent state instead of a growing key-value (KV) cache. This makes ordinary decoding memory-efficient, but hurts speculative decoding. To verify a batch of draft tokens and th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TreeWY: Speculative Verification for Gated DeltaNet Hybrids**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

现代大模型中广泛采用混合架构（hybrid models），即部分层使用 **softmax-attention**（维护 KV Cache），而更多层使用 **linear-attention** 结构如 **Gated DeltaNet (GDN)**。GDN 层通过一个固定大小的循环状态 $ S \in \mathbb{R}^{d_v \times d_k} $ 来压缩历史上下文，避免 KV Cache 随序列增长的问题。

然而，在 **speculative decoding** 场景下，这种设计带来了严重内存瓶颈：

- 在验证 draft token 时，系统需要回滚到任意中间节点的状态。
- 传统做法是为每个 draft 节点保存完整的 GDN 状态快照（full-state snapshotting）。
- 快照无法在树的不同分支间共享 → 内存开销随 draft tree 宽度线性上升 → **宽树 speculative decoding 变得不可行**。

> 🔴 **核心问题**：GDN 的非交换性（non-commuting transition）导致其状态不能像 KV Cache 一样被截断或高效回滚，使得 speculative decoding 的内存成本过高。

---

### 🚀 **提出了什么新方法或新思路**

作者提出 **TreeWY** —— 一种基于 **tree-structured WY transform** 的新型 speculative verification 方法，彻底移除 per-node 状态快照。

#### 核心思想：
将 GDN 的 **gated delta rule** 重写为伪值（pseudo-value）形式，并利用线性系统的三角求解一次性并行计算整棵树所有节点的输出。

#### 关键技术步骤：

1. **Rewrite Gated Delta Rule**  
   将原始递推公式：
   $$
   S_t = (I - \beta_t k_t k_t^\top) S_{t-1} + B_t v_t
   $$
   改写为加权累加形式：
   $$
   S_t = g_t S_0 + \sum_{i < t} g_i B_i (v_i - \alpha_i S_{i-1} k_i)
   $$
   其中括号内项定义为 **pseudo-value** $ V_i $。

2. **构建线性系统进行并行验证**
   对于包含 $ N $ 个节点的 draft tree（DFS 排序），构造如下线性系统：
   $$
   (I + \text{diag}(\beta) G) V = R
   $$
   - $ G[t,i] = (k_i k_t^\top) $ if $ i $ 是 $ t $ 的祖先
   - $ R[t] = p_t v_t - p_t q_t^\top S_0 $
   - $ G $ 是严格下三角矩阵 ⇒ 可用前向替换（forward substitution）快速求解

3. **Commit 时重建接受状态**
   当接受第 $ a $ 个节点后，直接从已解出的 $ V $ 中重构最终状态：
   $$
   S_a = g_a S_0 + \sum_{i \prec a} g_i k_i V_i
   $$

4. **存储优化**
   - 不再存储 $ N+1 $ 个完整状态块
   - 仅需存储一个大小为 $ O(N d_v) $ 的 **pseudo-value matrix**
   - 每 head 节省约 **128× 存储空间**

---

### ⚖️ **相比现有方法的优势**

| 方法 | 是否支持 Tree | 内存增长 | 回滚代价 | 实现复杂度 |
|------|----------------|-----------|------------|--------------|
| **Full-State Snapshotting (storeall)** | ❌（不经济） | $ O(N) $ | $ O(1) $ | 简单但昂贵 |
| **ReplaySSM** | ❌（链式） | $ O(1) $ | 周期 flush | 中等 |
| **STree** | ✅ | $ O(1) $ | 仅适用于 Mamba2 | 高 |
| **Bole (SGLang)** | ✅ | $ O(1) $ | 支持 tree closed-form | 高（未开源） |
| **TreeWY (本文)** | ✅ | $ O(1) $ | 单次 triangular solve | 已集成 vLLM |

#### 主要优势：

- ✅ **通用性强**：仅依赖 gated delta rule，不限定具体模型细节
- ✅ **支持任意结构的 draft tree**，且内存恒定（每层仅存一块）
- ✅ **可融合进 CUDA graph**（chain 情况下），提升执行效率
- ✅ 显著降低 **KV-cache 压力** 和 **HBM 占用**
- ✅ 在内存受限场景带来显著吞吐和延迟收益

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**

| 数据集 | 描述 |
|--------|------|
| **ShareGPT** | 用户分享的 ChatGPT 对话记录，用于真实对话负载测试 |
| **spec-bench** | 专为 speculative decoding 设计的基准测试套件 |
| **BurstGPT** | 模拟现实世界突发请求流量的数据集 |
| **Synthetic Workloads** | 自动生成的三种负载：<br>- `balanced-chat`<br>- `generation-heavy`<br>- `summarize-heavy` |

---

### ⚙️ **实验设置**

#### 模型配置：
- **Qwen3.5-35B-A3B**（TP1）：30 GDN 层 + 10 softmax 层
- **Qwen3.5-397B-A17B**（TP8）：45 GDN 层 + 15 softmax 层
- 所有模型均为 **3:1 GDN-to-softmax ratio**

#### 硬件平台：
- 使用 **NVIDIA B200 GPU**（178 GiB HBM per device）

#### Speculative 设置：
- **Draft Depth**: 3
- **Tree Structure**: 支持 `(w1, w2, w3)` 形式的 branching factor
- **Greedy drafting & verification**

#### 控制变量：
- **GPU Memory Utilization (gmu)**：{0.6, 0.75, 0.9}
- **Max Concurrency**: {1, 8, 32, 64, 128, 256}
- **Prefix Caching**: Disabled（公平比较）

---

### 📊 **评估指标**

| 指标 | 含义 |
|------|------|
| **Throughput (tput)** | 输出 token 数 / 秒 |
| **KV Reduction (KV red.)** | 峰值 KV-cache 使用量减少倍数 |
| **p99 TTFT** | 第一个 token 的 p99 延迟 |
| **Mean TPOT** | 平均每输出 token 时间 |
| **Acceptance Length** | 每轮成功接受的 draft token 数量 |
| **Admitted Batch Size (b)** | 成功接纳的请求数比例 |

---

### 🔁 **基线方法对比**

| 基线方法 | 描述 |
|---------|------|
| **storeall (vLLM 默认)** | 每个 draft 节点保存完整 GDN 状态快照 |
| **ReplaySSM** | 缓存输入历史，延迟状态物化，仅支持 chain |
| **Bole (SGLang)** | 类似 closed-form 方法，但运行于不同推理框架（SGLang） |

> 注：TreeWY 已实现为 **vLLM 的 fork**，通过 `SpeculativeConfig.mamba_state_commit="reconstruct"` 开启。

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据汇总（Table 1 & Figure 3）**

| 场景 | Throughput Gain | KV Usage ↓ | p99 TTFT ↓ | Notes |
|------|------------------|-------------|---------------|-------|
| **35B, gmu=0.6, conc=256** | **↑1.40×** | ↓1.04× | ↓3.97× | 内存严重受限，收益最大 |
| **397B, gmu=0.75, conc=256** | ↑1.15× | ↓1.61× | ↓3.35× | 大模型同样受益明显 |
| **轻载情况 (gmu=0.9)** | ~0.97–1.03× | ↓2.2–2.5× | ~1× | 内存未饱和，吞吐略降（调度开销） |

#### 内存节省效果：
- **峰值 KV-cache 减少 2–3×**
- 请求预emption 数量大幅下降（1365 vs 2531）
- 更多请求能被及时处理 → **p99 TTFT 最高改善达 40×**

> 💡 图 3 显示：只有当 baseline “out of KV” 时，TreeWY 才表现出显著优势；否则性能接近持平。

---

### 🌲 **Tree Width 扩展能力（Table 2 & B）**

| Tree Shape | Nodes (N) | storeall Block Cost | TreeWY Block Cost | Acceptance Length |
|------------|-----------|---------------------|--------------------|-------------------|
| (1,1,1)    | 3         | 4                   | 1                  | ~3.23             |
| (2,2,2)    | 14        | 15                  | 1                  | ↑3.38             |
| (3,3,3)    | 39        | 40                  | 1                  | ↑3.58             |

#### 发现：
- TreeWY 允许构建极宽的 draft tree（最多 39 节点），而内存成本不变
- acceptance length 随宽度增加持续上升（概率匹配更优）
- ❗但尚未转化为吞吐提升：因 tree verify kernel 无法被 CUDA graph 捕获，只能 piecewise 执行

> ➕ **Affordable ≠ Faster**：宽度变得“买得起”，但还不是“跑得更快”

---

### 🔍 **正确性验证**

- 在 175 个 matched points 上测试：
  - 平均 acceptance length 差异：**|Δ| = 0.039**
  - 最大差异：0.33，绝大多数 ≤ 0.01
- FP32 下与参考实现误差 < 1e-7
- token stream 与 baseline 功能等价（非 bit-identical）

---

### 🔄 **与 ReplaySSM 的对比（Appendix D）**

| 维度 | TreeWY | ReplaySSM |
|------|--------|-----------|
| 内存节省 | 略优（KV ↓1.56×） | 略差（KV ↓1.57×） |
| 吞吐增益 | 1.19× @256 | **1.38×** @256 |
| per-token cost | 较高（TPOT ↑） | 更低 |
| 原因分析 | 每步都 materialize state | 延迟写入，调度更优 |

> 🔎 推测差距来自 **commit 时的调度策略差异**，而非算法本身。未来可通过引入 deferred write 进一步优化。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **TreeWY 成功消除了 GDN speculative decoding 的内存墙问题**
   - 用 **pseudo-value matrix** 替代 per-node 快照
   - 实现 **constant memory footprint**，无论 tree 多宽

2. **在内存受限场景下带来巨大收益**
   - 最高 **1.4× 吞吐提升**
   - **p99 TTFT 改善高达 40×**
   - 特别适合高并发、bursty 流量场景（如 BurstGPT）

3. **使宽 draft tree 成为可能**
   - 虽然目前未提速，但打开了通往更高 acceptance rate 的大门
   - 是迈向 practical tree speculation 的关键一步

4. **方法具有高度通用性**
   - 仅依赖 gated delta rule，不绑定特定架构
   - 可扩展至其他 linear-attention 模型家族

---

### ⚠️ **局限性**

1. **Tree verify kernel 无法被 CUDA graph 捕获**
   - 导致 wider tree 执行效率低下
   - 必须 drop 到 piecewise execution，损失性能

2. **Commit 频繁 materialize state**
   - 相比 ReplaySSM 缺少 deferred write 优化
   - 引入额外 per-step 开销（约 2–3% throughput penalty）

3. **收益依赖 memory pressure**
   - 在内存宽松环境下，吞吐反而轻微下降
   - 不是一个“处处赢”的方案，而是“精准打击瓶颈”

---

### 🔮 **未来工作方向**

1. **Fusing Tree Path into Single Graph-Capturable Kernel**
   - 解决 non-causal ancestor mask 导致无法图捕获的问题
   - 是实现 wider tree 吞吐增益的关键

2. **Integrate Deferred State Write (like ReplaySSM)**
   - 在 commit 阶段延迟写入，进一步降低 per-step 开销
   - 可能弥合与 ReplaySSM 的性能差距

3. **Benchmark Against Bole (SGLang)**
   - 当前缺乏 head-to-head comparison
   - 若能统一 serving stack，可更全面评估 closed-form 方法边界

4. **Extend to Other Model Families**
   - 如 Mamba2、KDA 等 hybrid attention 架构
   - 验证 TreeWY 的泛化能力

---

## ✅ 总结一句话

> **TreeWY 通过 tree-structured WY transform 将 GDN 的 speculative verification 转化为单次 triangular solve，以极小的 pseudo-value 存储替代庞大的 per-node 快照，在内存受限场景下实现了高达 1.4× 吞吐和 40× TTFT 改善，并首次让宽 draft tree 在 GDN 模型上变得内存可行。**

</details>

---

### 11. [MEMPOWER: Efficient Power Management with Fine-grained Memory Analysis and Modeling for HPC Workloads](https://arxiv.org/abs/2608.20734)

**Authors**: Nanda Velugoti, Joseph Manzano, Andres Marquez, Nathan Tallent, Kyle Hale  
**Category**: cs.DC  
**Published**: 2026-08-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.20734v1  

#### Abstract
Managing the energy consumption and power efficiency of parallel applications is a significant issue in both HPC environments and in the cloud. As emerging applications continue to push against the memory wall of modern machines, the growing imbalance between compute and data movement creates new op...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MEMPOWER: Efficient Power Management with Fine-grained Memory Analysis and Modeling for HPC Workloads

## 1. 论文的主要贡献和创新点

### 解决的问题
现代高性能计算（HPC）应用日益面临“内存墙”问题，即数据移动的开销远超计算能力的增长，导致系统在内存延迟受限（latency-bound）而非带宽受限（bandwidth-bound）的状态下运行。然而，现有的 **DVFS**（Dynamic Voltage and Frequency Scaling）机制（如 Linux 的 `schedutil`）主要基于指令混合（instruction mix）进行频率调节，无法感知应用程序细粒度的内存访问行为，尤其是不规则（irregular）和高延迟的内存访问模式。这使得 CPU 在等待内存时仍维持高频运行，造成不必要的能耗。

因此，**如何在不影响性能的前提下，针对内存密集型代码区域动态降低 CPU 频率以节省能量，成为一个未被充分挖掘的机会**。

### 提出的新方法和创新点
论文提出了 **MEMPOWER**，一个基于模型的、以内存为中心的软件框架，用于优化 HPC 应用的功耗管理。其核心创新点如下：

- **细粒度内存行为建模**：利用 **MemGaze** 工具获取应用的低级内存访问轨迹（memory trace），并提取两个关键指标：
  - **Footprint Growth Rate**（足迹增长速率）：衡量单位内存访问带来的唯一内存地址增量，反映内存强度。
  - **Access Class**（访问类别）：区分不规则（irregular）、步进（strided）和常量（constant）访问模式。
  结合这两个指标，MEMPOWER 能够识别出真正处于 **内存延迟受限** 状态的代码区域。

- **静态二进制插桩与自动化决策**：提出一个 **分析模型**，该模型结合内存特征和 **P-state transition overhead**（状态切换开销），通过成本效益分析（cost-benefit analysis）决定是否以及在何处插入 DVFS 调用。最终通过 **DynInst** 对目标二进制文件进行静态插桩，注入最优的 `P-state` 切换指令。

- **无需硬件修改的软件解决方案**：整个框架在操作系统和应用层实现，不需要对硬件进行任何修改，具有良好的可部署性。

### 相比现有方法的优势
| 方面 | 现有方法（如 schedutil） | MEMPOWER |
|------|------------------------|----------|
| **信息粒度** | 仅依赖指令混合，忽略内存访问细节 | 利用细粒度内存轨迹，捕获不规则访问和高足迹增长 |
| **决策依据** | 缺乏对内存延迟的感知 | 显式识别内存延迟受限区域 |
| **控制方式** | 动态、由硬件/OS 实时决策 | 静态分析 + 二进制插桩，预设最优策略 |
| **性能开销** | 可能因频繁无效切换而浪费能量 | 通过成本模型摊销切换开销，避免在短循环中插入 |
| **通用性** | 通用但非最优 | 针对特定 HPC 应用定制化优化 |

---

## 2. 核心实验方法和设置

### 数据集与基准测试
实验在以下 HPC 基准套件上进行：
- **miniVite**：图分析应用，已知存在内存密集型区域。
- **NAS Parallel Benchmarks**：包含多个典型 HPC 内核，如：
  - CG（共轭梯度法）
  - MG（多重网格法）
  - FT（快速傅里叶变换）
  - BT/LU/SP（块三对角/三角分解/标量对流）
  - EP（加密类，计算密集）
  - IS（整数排序，内存密集）
- **HPCG**（High Performance Conjugate Gradient）：稀疏矩阵向量乘（SpMV）为主，具有强不规则内存访问。

### 实验设置
- **硬件平台**：
  - CPU：12th Gen Intel® Core™ Processor（Alder Lake，8P+8E cores）
  - DRAM：128 GB
  - OS：Linux 6.0
- **DVFS 控制**：
  - 使用 `acpi-cpufreq` 驱动 + `userspace` governor
  - 自定义 **kernel driver** 通过写入 `IA32_PERF_CTL MSR` 实现手动 P-state 控制
- **线程配置**：
  - 所有测试使用 8 个 OpenMP 线程，并绑定到性能核心（P-cores）
- **执行方式**：
  - 每个基准运行 10 次，取算术平均值
  - MEMPOWER 先对程序进行一次 profiling 运行以收集内存轨迹，再生成优化后的二进制文件

### 评估指标
- **EDP**（Energy-Delay Product）：能量延迟积，综合衡量能效的核心指标（越低越好）
- **Execution Time**：执行时间，用于评估性能损失
- **Normalized EDP**：相对于基线的归一化 EDP
- **消融实验**：验证各组件（候选区域选择、成本模型）的有效性

### 基线方法对比
- **Baseline (schedutil)**：Linux 默认的 DVFS 策略
- **Best manual selection**：人工为每个函数选择最优固定 P-state
- **Intel Active Mode (Powersave / Performance)**：Intel 硬件控制模式下的两种策略

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **EDP 改善范围**：MEMPOWER 在不同 HPC 基准上实现了 **6% 到 42%** 的 EDP 降低。
- **几何平均改善**：所有基准的几何平均 EDP 降低 **20%**。
- **性能影响**：执行时间增加不超过 **3%**（几何平均），说明性能损失极小。

### 与基线方法的对比结果
| 基准 | EDP 降低（vs. schedutil） | 说明 |
|------|----------------------------|------|
| **CG, MG, SP** | ~40–42% | 存在大量内存密集型内核（如 conj_grad, mg3p），MEMPOWER 成功降频并摊销开销 |
| **FT** | ~20% | 含不规则内存访问，但也有高计算强度部分；MEMPOWER 智能地保持高频 |
| **IS** | ~10% | 不规则内存访问主导，适合降频 |
| **BT, LU** | ~10–15% | 步进访问为主，局部性较好，仍有优化空间 |
| **EP** | ~6% | 高度计算密集，MEMPOWER 选择最高频率，接近最优 |
| **HPCG** | ~18% | SpMV 和 SymGS 被识别为可降频区域，WAXPBY/DDOT 保持高频 |

> ✅ **关键发现**：MEMPOWER 在所有测试中均 **优于或等于最佳手动选择方案**，证明其模型决策的有效性。

### 消融实验结果
论文通过关闭不同组件进行消融研究（见 Fig. 12）：
- **MEMPOWER w/o Candidate Selection**：
  - EDP 显著恶化
  - 原因：未过滤掉不适合降频的代码区域（如计算密集区），导致无效切换增加开销
- **MEMPOWER w/o Candidate Selection or Cost Model**：
  - EDP 最差
  - 原因：无成本模型指导，无法平衡节能收益与切换代价，插入位置不当

> 🔍 **结论**：**候选区域选择** 和 **成本模型** 是 MEMPOWER 成功的关键，二者缺一不可。

---

## 4. 关键结论和发现

### 主要发现
1. **内存延迟受限区域是节能的重要机会窗口**：当 CPU 因内存访问而停顿（stalled）时，降低频率几乎不带来性能损失，却能显著节省能量。
2. **硬件缺乏高层语义感知**：当前硬件 DVFS 机制（如 SpeedShift）虽快，但无法感知应用级的内存访问模式（如 footprint growth），导致错失优化机会。
3. **软件层建模可有效弥补这一差距**：通过 profiling 获取内存轨迹，并建立分析模型，可以在软件层做出更优的 DVFS 决策。
4. **切换开销必须被显式建模**：盲目插入 P-state 切换会导致开销反超节能收益，**成本模型的作用是确保节能收益大于切换代价**。

### 方法的局限性
- **硬件依赖性强**：目前仅支持具备 **PT_WRITE** 功能的 Intel x86 CPU，不适用于 ARM 或其他架构。
- **需要一次 profiling 运行**：虽然后续无需重插桩，但首次使用需运行一次以收集内存轨迹。
- **OpenMP 局限**：当前评估集中在节点内并行（OpenMP），尚未扩展到大规模 MPI 并行场景。
- **静态决策**：优化策略基于一次 profiling，若输入数据变化导致内存行为改变，可能不再最优。

### 未来工作方向
- **扩展至更多内存指标**：引入 **reuse distance**、**MPKI**（Misses Per Kilo-Instruction）等进一步提升模型精度。
- **支持 MPI 应用**：研究在分布式环境下如何协调多节点的 MEMPOWER 策略。
- **开发更智能的 kernel governor**：将 MEMPOWER 的思想集成到操作系统层面，实现透明化优化。
- **近数据处理（Near-Data Processing）协同**：利用内存行为模型决定何时将任务卸载到近内存计算单元。
- **跨架构支持**：适配 ARM 的 STM/ITM 接口以支持非 Intel 平台。

---

> 📌 **总结**：MEMPOWER 通过 **细粒度内存分析 + 成本感知建模 + 静态二进制插桩**，成功揭示了 HPC 应用中被忽视的节能潜力，在最小性能损失下实现了高达 **42%** 的 EDP 改善，为软件定义的高效能计算功耗管理提供了新范式。

</details>

---

### 12. [Integrating a Python Dynamical core into ICON](https://arxiv.org/abs/2608.21150)

**Authors**: Mauro Bianco, Till Ehrengruber, Enrique Gonz\'alez Paredes, Andreas Jocksch, Christos Kotsalos, Ioannis Magkanaris, Philip M\"uller, Edoardo Paone, Mikael Simberg, Hannes Vogt, Jacopo Canton, Yilu Chen, Anurag Dipankar, Nicoletta Farabullini, Michael J\"ahn, Matthieu Leclair, Ong Chia Rui, Nathan Beech, Nicolas Gruber, Christoph M\"uller, Daniel Hupp, Xavier Lapillonne  
**Category**: cs.DC  
**Published**: 2026-08-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.21150v1  

#### Abstract
The transition of Earth-system models to exascale is often hindered by rigid, monolithic Fortran codebases and maintenance-heavy compiler directives. While high-level DSLs offer a solution, they frequently fail due to cumbersome integration. We present the integration of a Python-based ICON dynamica...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Integrating a Python Dynamical core into ICON》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统地球系统模型（如 **ICON**）长期依赖于大型、单体式的 **Fortran** 代码库，并通过嵌入 **compiler directives**（如 OpenACC）实现高性能计算（HPC）优化。这种做法导致：
- 代码复杂、难以维护和扩展；
- 架构特定优化与科学逻辑耦合严重，阻碍了跨平台可移植性；
- 难以集成现代 AI/ML 工具链和数据驱动方法。

该论文旨在打破“Python 不适合 HPC”的固有偏见，解决如何在不牺牲性能的前提下，将高生产力的 Python 语言无缝集成到生产级 Fortran 气候模拟框架中的难题。

---

### 🚀 提出的新方法与创新思路

1. **全 Python 实现的 Dynamical Core（ICON4Py）**
   - 使用 **GT4Py**（Python 嵌入式 DSL）重写了 ICON 的整个 **dynamical core**，保留原始数值离散化方案。
   - 利用 GT4Py 的 `@field_operator` 和 `@scan_operator` 抽象表达 stencil 计算和垂直扫描操作，提升代码可读性和模块化程度。

2. **基于 DaCe 的自动优化后端**
   - 将 GT4Py 程序转换为 **Stateful DataFlow Multigraphs (SDFGs)**，由 **DaCe** 框架进行高级数据流优化：
     - 全局 Kernel Fusion
     - 内存布局变换（Memory Layout Transformation）
     - 迭代空间分割（Iteration Range Splitting）
     - 硬件专用内存利用（registers, shared memory 等）

3. **轻量级 Fortran-Python 耦合接口 `py2fgen`**
   - 基于 **CFFI** 开发工具 `py2fgen`，自动生成 ISO-C 绑定接口，实现 Fortran 与 Python 的高效互操作。
   - 支持零拷贝数组传递（NumPy/CuPy），缓存字段对象避免重复开销。
   - 隐藏 Python 解释器启动延迟，在运行时仅用于调度 GPU kernels。

4. **真正的“组合式开发”范式**
   - 动态核心作为独立组件可用不同语言实现，只要满足接口规范即可替换。
   - 支持 JIT/AOT 编译模式，便于调试与部署。

---

### 🔍 相比现有方法的优势

| 方面 | 传统 DSL 方法 | 本文方法（ICON4Py） |
|------|----------------|------------------------|
| **集成方式** | 生成代码链接进主程序 | 直接调用 Python 解释器，动态执行 |
| **维护成本** | DSL 工具链变更需重改接口 | 接口与 GT4Py/DaCe 解耦，更稳定 |
| **性能控制** | 依赖手工优化指令 | 自动 kernel fusion 与数据流优化 |
| **生态兼容性** | 孤立 | 可接入 JAX、NumPy、CuPy，支持 ML 流程 |
| **测试能力** | 单元测试困难 | 支持细粒度单元测试与数据验证 |

> 💡 **核心突破**：证明了 Python 不再是 HPC 的性能瓶颈，而是可以通过现代编译技术成为可持续、高效、硬件无关的气候建模未来路径。

---

## 2. 核心实验方法和设置

### 📊 数据集与模拟配置
- 使用 **ICON 模型的标准大气-陆地-海洋耦合配置**：
  - 大气部分采用 **ICON NWP** 物理参数化；
  - 陆面模型为 **JSBACH**；
  - 海洋模型为 **ICON-O**，含海冰；
  - 耦合器使用 **YAC**。
- 网格类型：**icosahedral grid (RnBm)**，具体使用：
  - **R2B08**: ~10 km 分辨率
  - **R2B09**: ~5 km 分辨率
  - **R2B10**: ~2.5 km 分辨率
- 垂直层次：120 层（大气），72 层（海洋）

---

### ⚙️ 实验设置
- **硬件平台**：Tier-0 超算，节点配备 **NVIDIA Grace Hopper Superchip (GH200)**  
  - 每节点 4 × GH200 模块（ARM CPU + Hopper GPU）
  - HBM3 显存 96GB/GPU，NVLink 互联
  - Cray Slingshot-11 网络（200 Gbps）

- **对比基线**：
  - **Fortran + OpenACC**：原生高度优化版本，当前生产环境基准

- **评估指标**：
  - **Wall-clock time per call** for `nh_solve`（动力核子步）和 `nh_hdiff`（水平扩散）
  - **Simulated Days Per Day (SDPD)**：衡量整体吞吐量
  - **Weak & Strong Scaling Efficiency**
  - **Relative Error Tolerance**（验证正确性）：1e-12 ~ 1e-7（因浮点运算顺序差异放宽）

- **验证层级体系（Verification Hierarchy）**：
  1. **Level 1**：序列化输入输出，对比单个 stencil 输出相对误差
  2. **Level 2**：在 Fortran 主程序中并行运行两套动力核，实时比较输出
  3. **Level 3**：使用 **probtest** 进行多步积分误差增长分析（5–10 步 ICON 时间步）

---

## 3. 主要实验结果和性能指标

### 📈 性能表现概览

#### （1）弱扩展性（Weak Scaling）—— 图6
- 在 R2B08 → R2B10 上分别使用 40 → 640 GPUs
- **nh_solve（动力核）**：
  - ICON4Py 比 Fortran+OpenACC 快 **20–30%**
- **nh_hdiff（扩散项）**：
  - 加速高达 **~50%**

> ✔️ 表明 ICON4Py 具备优秀的弱扩展能力，且随规模增大优势保持稳定。

---

#### （2）强扩展性（Strong Scaling）—— 图7
- 固定 R2B10 网格（2.5km），GPU 数从 480 增至 1600
- 结果显示：
  - `nh_solve` 平均耗时下降明显，**性能提升 20–30%**
  - `nh_hdiff` 提升达 **~50%**
  - 下方比率图中所有柱状图 >1，说明 ICON4Py 更快

> ✔️ 强扩展效率良好，无显著通信或调度瓶颈。

---

#### （3）耦合系统吞吐量 —— 图8
- **Atmosphere-Land-Ocean 耦合模拟（XPP 配置）**
- R2B10 网格，120 垂直层，GH200 节点运行
- **SDPD（模拟天数/实际天数）**：
  - Fortran+OpenACC：约 **145 SDPD**
  - **ICON4Py：达到 160 SDPD**
  - ➜ **整体性能提升约 10%**

> 注：由于动力核约占总时间 50%，其 20–30% 的加速转化为整体约 10–15% 提升，符合预期。

---

### 🔬 消融实验与关键观察（隐含分析）

虽然未明确列出消融表，但从文中可推断以下关键因素贡献：

| 优化机制 | 贡献说明 |
|--------|---------|
| **Kernel Fusion** | 减少内存读写次数，显著降低 kernel launch 开销 |
| **Data Layout Optimization** | 提高 GPU memory bandwidth 利用率 |
| **Overlap Computation & Communication** | GHEX 支持 UCX/NCCL，实现 halo exchange 与计算重叠 |
| **Python Overhead Mitigation** | Python 仅负责 launch kernels，实际计算在 native code 执行；profile 显示 Python 时间占比极小（见图5） |

> ✅ 图5 显示：Python 解释器调度时间 << GPU 实际计算时间，实现了有效隐藏解释器开销。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Python 完全可用于生产级 HPC 气候建模**
   - 在全球尺度、千米级分辨率的 ICON 模拟中，Python 实现（ICON4Py）不仅没有引入性能损失，反而**超越 Fortran+OpenACC 20–30%**。

2. **性能来自先进编译优化而非语言本身**
   - 性能增益主要源于 **GT4Py + DaCe** 的自动化数据流优化（尤其是 kernel fusion 和 memory layout 调整），而非手写代码。

3. **模块化设计极大提升软件工程质量**
   - Python 实现支持独立开发、测试、复用，促进团队协作与持续集成。
   - 支持 JAX 后端，为未来融合 AI-based parameterization 打下基础。

4. **跨架构可移植性增强**
   - 同一份 Python 源码可在 NVIDIA GPU、AMD GPU、多核 CPU 上编译运行，真正实现 **performance portability**。

---

### ⚠️ 局限性

1. **初始启动延迟**
   - Python 解释器冷启动有一定开销，不适合极短任务场景。

2. **布尔数组需复制**
   - Fortran logical 类型占 4 字节，而 NumPy bool 为 1 字节，必须复制转换（但作者指出这些数组通常只初始化一次，影响有限）。

3. **对开发者技能要求变化**
   - 需掌握 Python + DSL + 编译优化思维，不同于传统 Fortran 科学编程范式。

4. **尚未完全替代所有组件**
   - 当前仅替换了 dynamical core，其他物理过程仍为 Fortran 实现。

---

### 🔮 未来工作方向

1. **全面迁移其他物理模块至 GT4Py**
   - 如 radiation、microphysics 等，构建全 Python 化的 ICON 子系统。

2. **深度整合机器学习工作流**
   - 利用 JAX 支持，训练和部署 AI-enhanced 参数化方案。

3. **探索异构任务调度策略**
   - 进一步优化 Python 控制流与设备计算之间的协同。

4. **推广至其他地球系统模型**
   - 将此架构应用于 CESM、WRF、MPAS 等模型，推动气候建模现代化。

---

## ✅ 总结一句话

> 本论文成功将 Python 打造成一个**高性能、可维护、可扩展**的地球系统建模语言，借助 **GT4Py + DaCe** 技术栈，在不牺牲性能的前提下实现了对 Fortran+OpenACC 的全面超越，标志着气候模拟向**软件工程现代化与 AI 融合时代**迈出了关键一步。

</details>

---

### 13. [Training, learning and inference: unified dynamics of neural systems](https://arxiv.org/abs/2608.20965)

**Authors**: Mian Wang  
**Category**: cs.LG  
**Published**: 2026-08-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.20965v1  

#### Abstract
We define an atomic generation fact f=(u,tau,omega,z;rho), recording the origin, realized transformation, concrete occurrence, generated result and relation role. Compiled into a Generation-Fact Graph (GFG), these facts provide an AI-native, compilable scientific fact substrate preserving generation...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Training, learning and inference: unified dynamics of neural systems*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

该论文旨在解决当前深度学习研究中对**训练、学习与推理过程之间关系缺乏统一动态解释**的问题。传统分析通常依赖于标量指标（如 loss、accuracy）或静态模型快照，难以揭示能力形成、维持、衰退与恢复的内在机制。此外，现有系统在追踪计算生成路径（provenance）时存在信息不完整、无法跨阶段连接等问题。

### 🚀 提出的新方法与新思路

#### （1）**Atomic Generation Fact 与 Generation-Fact Graph (GFG)**  
提出一种原子级的生成事实定义：
$$
f = (u, \tau, \omega, z; \rho)
$$
其中：
- $ u $: 参与的实际源信息  
- $ \tau $: 实现的变换  
- $ \omega $: 具体发生事件  
- $ z $: 输出结果或显式处置  
- $ \rho $: 关系角色  

这些原子事实被编译为 **Generation-Fact Graph (GFG)**，构建了一个**AI-native、可编译、可追溯的科学事实底座**，保留完整的生成历史与对象身份。

#### （2）**GFG-based Recursive Scientific Process**  
建立一个递归科学研究范式：
- 科学问题驱动对已有 GFG 的查询与重组；
- 分析、干预、回放产生新的执行与生成事实；
- 新的事实经验证后重新编译进下一代 GFG；
- 形成闭环循环：`GFG₀ → GFG₁ → GFG₂ → ⋯`

这一框架使得复杂系统的演化可以基于真实执行轨迹进行因果推断。

#### （3）**统一的训练–学习–推理动力学理论**
首次将 training、learning 和 inference 视为同一动态过程的不同阶段：

| 阶段 | 动力学描述 |
|------|-----------|
| **Training** | 参数–优化器系统随状态和记忆演化的动态过程；每次更新的效果由接收状态（receiving state）和目标特定更新几何共同决定 |
| **Learning** | 分布式功能支持（distributed functional support）的持续重组；不是参数变化本身，而是其引发的功能结构重配置 |
| **Inference** | 对已形成学习状态的“冻结投影”（frozen projection），通过 query 条件化地调用并联合组合训练期间形成的分布式支持 |

#### （4）**Attention 的机制解释**
从功能组织角度解释 Attention 成功的原因：
- Query-Key 构建当前情境下的主动投影；
- Value 携带训练中形成的分布式内容；
- 实现了“一次学习，多路查询”的动态支持组合。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法局限 | 本文优势 |
|------|--------------|---------|
| **可解释性** | 依赖 loss 曲线、梯度等宏观指标，无法定位具体能力来源 | 基于 GFG 追踪每个能力形成的完整生成路径 |
| **因果性** | 多数为相关性分析，缺乏干预验证 | 支持精确干预（如组件门控、版本回滚）、反事实回放 |
| **统一性** | 将 training / learning / inference 分离处理 | 揭示三者是统一动态的不同环节 |
| **泛化性** | 多集中于特定架构（如 Transformer） | 在 nanoGPT、ResNet、Diffusion Model 上均验证有效性 |
| **科学方法论** | 缺乏系统性的实证积累机制 | 提出 GFG-based 递归科学流程，支持知识累积与再利用 |

---

## 2. 核心实验方法和设置

### 🧪 使用的数据集与模型

| 实验类别 | 模型 | 数据集 | 设置说明 |
|--------|------|--------|----------|
| 主要实验 | **nanoGPT** | 自定义语言任务（基于 Karpathy 实现） | 13 条独立训练历史，覆盖能力形成、衰退、恢复全过程 |
| 跨系统验证 | **ResNet-18** | CIFAR-100 | SGD + Momentum，3 个随机种子 |
| 跨系统验证 | **DDPM-style U-Net** | CIFAR-10 | AdamW 优化，3 个随机种子 |
| 强化学习实验 | Policy Network（类 RL 架构） | 合成反馈环境 | 控制正向反馈浓度，观察能力权衡 |

---

### 🛠️ 实验设置与评估指标

#### （1）**训练–学习机制探测实验（TL-E系列）**

| 实验编号 | 目标 | 方法 |
|--------|------|-------|
| TL-E01 | 检验标量指标是否足以预测未来能力 | 匹配不同运行中的 loss/accuracy/step/margin，比较后续轨迹 |
| TL-E02 | 干预参数/优化器演化 | 暂停更新 vs 修改 clip 阈值 |
| TL-E03 | 测试更新效果是否依赖接收状态 | full/skip 分支 + 接收状态互换 |
| TL-E04 | 测量有限振幅非线性响应 | 同一更新以不同幅度应用（α ∈ {0, 0.125, ..., 1}） |
| TL-E05 | 识别响应调节因子 | 比较七类前置信息的预测能力 |
| TL-E06 | 探测功能支持分布 | 单/双组件门控（component gating） |
| TL-E07 | 测量支持重分配 | 比较 α=0 与 α=1 下的支持结构变化 |
| TL-E08 | 连接内部状态与可观测能力 | 定义 identity-aligned margin 判断边界穿越 |

#### （2）**推理机制验证实验（INF-E系列）**

| 实验 | 方法 |
|-----|------|
| INF-E01 | 冻结推理 + 组件门控 + 版本回滚（pre-formation rollback） |
| INF-G01 | 在 ResNet/CIFAR 和 Diffusion/CIFAR 上复现实验 |

#### （3）**预测任务：Target-Boundary Prediction**

- **输入**：目标当前边界状态、目标特定更新几何、参数–Adam 接收状态
- **输出**：预测更新后目标是否保持正确、变错、恢复、仍错
- **评估指标**：
  - Accuracy
  - Balanced Accuracy
  - Macro-averaged Recall
  - Per-class Recall
  - Confusion Matrix

#### （4）**强化学习反馈实验（RL-E系列）**

- 设计选择性正反馈、平衡反馈、修复分支
- 测量功能支持集中度、未强化技能退化、恢复能力

---

### 📊 基线方法对比

本文未采用传统“模型 vs 模型”对比方式，而是通过以下方式确立优越性：

- **否定经典假设**：
  - 标量指标（loss, step）不能充分刻画能力状态（TL-E01）
  - 固定局部近似（Jacobian/J/K）无法外推完整更新响应（TL-E04）
  - 功能不由单一组件承担（TL-E06）

- **验证机制唯一性**：
  - 所有推理结果都依赖于训练中形成的 exact parameter version（rollback 实验证明）
  - 支持组合是非加性的（non-additive），排除独立贡献假设

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ **Target-Boundary Predictor 性能（TL-P01）**

| 指标 | 数值 |
|------|------|
| **Target-boundary accuracy** | **91.43%** |
| **Balanced accuracy** | 92.17% |
| **Macro-averaged recall** | **91.49%** |
| **各转移类别的召回率** | |
| &nbsp;&nbsp;• Remained correct | 87.18% |
| &nbsp;&nbsp;• Correct → Incorrect | **98.91%** |
| &nbsp;&nbsp;• Remained incorrect | 95.90% |
| &nbsp;&nbsp;• Incorrect → Correct | 83.95% |

> 在完全 hold-out 的确认集上达到超过 91% 的准确率，表明所提出的三个坐标（边界状态、更新几何、接收状态）具有强预测力。

#### ✅ **跨系统验证结果（TL-G01 / TL-G02 / INF-G01）**

- 在 **ResNet/CIFAR** 和 **Diffusion/CIFAR** 中均复现：
  - 接收状态条件下的功能响应
  - 分布式支持的持久性重组
  - 推理过程中 exact state preservation
  - query-conditioned 支持调用
  - 非加性组合
  - 对 pre-formation rollback 的敏感性

> 排除了 nanoGPT 架构特异性解释，支持理论的跨系统适用性。

#### ✅ **强化学习双刃效应（RL-E05 / RL-E06）**

| 发现 | 数据支持 |
|------|---------|
| **剂量有序效应** | 12/12 种子中，反馈浓度↑ ⇒ 目标支持↑（ρ=+1.000），其他技能 margin↓（ρ=−1.000） |
| **终点权衡** | 目标支持提升 +9.67 pp，但其他技能准确率下降 −39.06 pp |
| **再平衡恢复** | 12/12 种子中，其他技能准确率恢复至 99.48%，目标保持 100% |

> 表明强化学习可能造成“能力偏科”，但可通过反馈调节逆转。

#### ✅ **消融实验结果**

| 实验 | 结果 |
|------|------|
| **移除接收状态信息**（TL-E03） | 相同更新在不同状态下产生不同响应 ⇒ 效果非内在于更新本身 |
| **移除目标特定更新几何**（TL-E05） | 预测性能显著下降 |
| **单组件门控**（TL-E06） | 不同目标表现出不同的必要性模式 ⇒ 支持分布动态可变 |
| **双组件门控**（TL-E06） | 所有 138 次测试偏离加性叠加 ⇒ 支持组合为非加性 |
| **Pre-formation rollback**（INF-E01） | 52/52 回滚导致精度下降，恢复版本则原样还原 ⇒ 推理因果依赖训练形成的状态 |

---

## 4. 关键结论和发现

### 🎯 主要发现

1. **Training 是状态依赖的过程**  
   更新效果不仅取决于梯度大小，更取决于参数–优化器系统的当前“接收状态”。

2. **Learning ≠ Parameter Update**  
   学习的本质是**分布式功能支持的持久性重组**，而非简单的权重调整。

3. **Inference 是冻结的投影**  
   推理不改变参数，而是 query 条件化地调用训练中形成的分布式支持，并以非加性方式组合。

4. **Attention 成功的根源**  
   其机制天然契合“query-conditioned projection + value-carried support”，实现了高效的功能调度。

5. **强化学习具有双刃性**  
   持续偏向性反馈会放大目标能力，但也可能导致其他能力退化，需监控和支持再平衡。

6. **GFG 是通用科学基础设施**  
   支持跨实验的知识积累、因果追溯与理论迭代，适用于任何可执行系统。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **计算开销大** | GFG 记录所有生成细节，存储与处理成本较高 |
| **依赖精确 instrumentation** | 需要在底层捕获所有运行时记录，对系统侵入性强 |
| **目前主要用于分析而非训练加速** | 当前价值更多体现在解释性和科学发现，尚未直接用于提升训练效率 |
| **人工设计 capture protocol** | 尽管可自动化趋势明显，但仍需专家定义五元组映射规则 |

---

### 🔮 未来工作方向

1. **GFG 自动化压缩与索引技术**  
   开发轻量化 GFG 存储格式，支持大规模部署。

2. **将 GFG 用于自动调试与错误溯源**  
   如定位某个错误预测的具体生成路径。

3. **扩展至更大规模模型（LLM、MoE）**  
   验证理论在百亿级以上模型中的普适性。

4. **构建 GFG-powered AI Scientist**  
   利用 LLM + GFG 实现全自动假说生成、实验设计与验证闭环。

5. **应用于神经网络架构搜索（NAS）与训练策略优化**  
   基于支持重组效率指导结构设计。

6. **探索“可逆学习”机制**  
   借鉴 rollback 实验思想，发展可控遗忘或能力切换技术。

---

## ✅ 总结

本论文提出了一个革命性的视角：**训练、学习与推理并非孤立过程，而是统一动态的三个阶段**。通过引入 **Generation-Fact Graph** 和 **GFG-based recursive scientific process**，作者实现了对神经网络演化全过程的精细化、可追溯、可干预的研究范式。实验不仅验证了核心理论，还揭示了 Attention 的本质成功逻辑以及强化学习的潜在风险。这不仅是机器学习机制理解的重大突破，也为未来构建“自我理解”的 AI 系统提供了方法论基础。

</details>

---

### 14. [COEC: Calibrated Orthogonal-Equivalence Compensation for Structured Pruning of Large Language Models](https://arxiv.org/abs/2608.21142)

**Authors**: Peiqi Yu, Nam Ling, Wei Wang, Wei Jiang  
**Category**: cs.LG  
**Published**: 2026-08-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.21142v1  

#### Abstract
Structured pruning reduces the size and inference cost of large language models (LLMs) by removing weight columns, but the resulting output error can degrade accuracy. Existing training-free compensation methods use an additive bias or a single orthogonal rotation on the output side of the retained ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：COEC: Calibrated Orthogonal-Equivalence Compensation for Structured Pruning of Large Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **Structured pruning** 通过移除权重矩阵的列（如MLP神经元或注意力头）来压缩大语言模型（LLMs），但会引入输出误差，导致精度下降。
- 现有的**训练无关补偿方法**（training-free compensation）通常仅在输出侧进行修正，例如添加偏置（bias）或对输出空间进行单侧正交旋转（如RCPU）。这些方法保留了输入端的奇异帧（singular frame），限制了模型在列被移除后对信号路径的适应能力。

### **提出的新方法：COEC**
- **COEC**（Calibrated Orthogonal-Equivalence Compensation）是一种**训练无关的补偿框架**，用于结构化剪枝后的LLMs。
- 核心思想是：**对保留的权重应用交替的左右正交旋转**（alternating left and right orthogonal rotations），并结合**逐模式奇异值重缩放**（per-mode singular-value rescaling）。
- 具体组件包括：
  - **Reduced-Stiefel右旋转求解器**：在降维的Stiefel流形上优化右旋转，提升效率。
  - **基于GCV的正则化强度选择**：使用广义交叉验证（Generalized Cross-Validation, GCV）为每层自动选择最优的重缩放正则化参数。
  - **Gram谱调温**（Gram Tempering）：降低高能量激活方向的主导性，平衡困惑度（perplexity）与零样本准确率。
  - **锚定层间对齐惩罚**（Anchored Inter-Layer Alignment Penalty）：保持相邻注意力投影之间的几何关系，避免重建目标忽略的结构失配。

### **相比现有方法的优势**
- **更全面的补偿机制**：现有方法（如FLAP的bias、RCPU的左旋转）只调整输出侧；COEC通过**双侧旋转**同时调整输入和输出奇异帧，能更好地恢复因列移除而丢失的信号。
- **更强的泛化能力**：所有补偿操作仅依赖校准集（calibration set）的二阶统计量，无需反向传播或重新训练，计算开销低。
- **即插即用**（plug-in）：不依赖特定的列选择准则（column selection criterion），可应用于Wanda-sp、FLAP、RCPU等多种剪枝方法之上。
- **性能提升显著**：在多种LLM（Llama-3, Llama-3.1, Qwen2.5）和不同稀疏度下，均优于现有补偿方法，尤其在高稀疏度时增益更大。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **校准集**（Calibration Set）：从WikiText-2训练集中采样128个序列（长度2048），用于估计二阶统计量（Gram矩阵）。
- **评估集**：
  - **困惑度**（Perplexity）：在WikiText-2测试集上评估。
  - **零样本准确率**（Zero-shot Accuracy）：在7个任务上评估，包括BoolQ、RTE、WinoGrande、HellaSwag、ARC-e、ARC-c、OpenBookQA，使用`lm-evaluation-harness`工具包。

### **实验设置**
- **模型**：Llama-3.1-8B、Llama-3-70B、Qwen2.5-7B/14B/32B/72B。
- **剪枝比例**：10%、20%、30%的结构化列稀疏度。
- **列选择准则**（Selection Scores）：
  - Wanda-sp：基于激活感知的重要性评分。
  - FLAP：基于激活方差的通道评分。
  - RCPU：基于权重范数与激活方差乘积的评分。
- **补偿方法对比**：
  - Wanda-sp（无补偿）
  - FLAP（bias补偿）
  - RCPU（左旋转+全局重缩放）
  - COEC（本文方法）

### **评估指标**
- **困惑度**（Perplexity, ↓）：越低越好。
- **零样本准确率**（Zero-shot Accuracy, ↑）：越高越好。
- 所有补偿均在相同校准集和列选择下进行，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（30%稀疏度）**

#### **表1：WikiText-2 困惑度（↓）**

| Selection | Comp. | Llama-3.1-8B | Llama-3-70B | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-32B | Qwen2.5-72B |
|----------|--------|---------------|--------------|-------------|--------------|--------------|--------------|
| Wanda-sp | —      | 14.35         | 6.20         | 13.03       | 15.15        | 13.90        | 6.02         |
| Wanda-sp | COEC   | **10.89**     | **5.61**     | **9.09**    | **13.26**    | **9.64**     | **5.17**     |
| FLAP     | bias   | 11.89         | 5.86         | 8.65        | 8.82         | 6.85         | 5.29         |
| FLAP     | COEC   | **10.86**     | **5.58**     | **8.34**    | **7.98**     | **6.71**     | **5.25**     |
| RCPU     | RCPU   | 11.57         | 5.69         | 8.98        | 7.98         | 7.03         | 5.23         |
| RCPU     | COEC   | **10.91**     | **5.58**     | **8.91**    | **7.63**     | **6.74**     | **5.21**     |

> ✅ **COEC在所有模型上均取得更低的困惑度**。

#### **表2：7项任务零样本准确率（↑）**

| Selection | Comp. | Llama-3.1-8B | Llama-3-70B | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-32B | Qwen2.5-72B |
|----------|--------|---------------|--------------|-------------|--------------|--------------|--------------|
| Wanda-sp | —      | 47.4          | 68.7         | 49.1        | 53.7         | 57.5         | 71.1         |
| Wanda-sp | COEC   | **48.9**      | **70.4**     | **56.0**    | 53.8         | **61.0**     | **72.5**     |
| FLAP     | bias   | 44.8          | 68.8         | 52.6        | 55.9         | 65.6         | 70.5         |
| FLAP     | COEC   | **47.1**      | **71.7**     | **54.5**    | **59.5**     | **65.3**     | **72.5**     |
| RCPU     | RCPU   | 47.0          | 71.3         | 54.4        | 59.2         | 63.8         | 71.7         |
| RCPU     | COEC   | **50.2**      | **71.7**     | **56.1**    | 58.5         | 64.2         | 72.0         |

> ✅ **COEC在大多数设置下提升了零样本准确率**，尤其在Llama-3.1-8B和Qwen2.5-7B上提升显著。

### **消融实验结果（Ablation Study）**
在五个7-8B模型上逐步加入COEC组件，30%稀疏度下的性能变化：

| 配置 | 困惑度（PPL） | 零样本准确率（ZS） |
|------|----------------|--------------------|
| Prune + RCPU | 10.06 | 50.9 |
| + GCV per-mode rescale | 9.81 | 51.3 |
| + converged two-sided rotation | 9.45 | 51.5 |
| + reduced-Stiefel right solve | 9.86 | 52.7 |
| + Gram tempering (α=0.9) | 9.65 | 52.7 |
| + anchored alignment (λₐ=50) | **9.65** | **52.9** |

- **GCV重缩放** 和 **双侧旋转** 是降低困惑度的关键。
- **Reduced-Stiefel求解器** 显著提升零样本准确率（+1.2点），尽管略微增加困惑度。
- **Gram tempering** 和 **alignment penalty** 进一步微调性能，最终达到最佳平衡。

---

## **4. 关键结论和发现**

### **主要发现**
1. **双侧旋转优于单侧补偿**：仅调整输出侧（如RCPU）无法恢复因列移除而改变的输入奇异子空间。COEC通过交替优化左右旋转，能更有效地补偿信号损失。
2. **COEC具有普适性和兼容性**：作为即插即用模块，可无缝集成到多种剪枝方法中，无需修改选择规则或架构。
3. **性能随稀疏度增加而提升**：剪枝越严重，COEC的补偿效果越明显，表明其在高稀疏场景下更具价值。
4. **无需额外部署成本**：COEC仅修改权重参数，不改变模型结构，因此推理时的参数量、内存和FLOPs与原始剪枝模型一致。

### **方法的局限性**
- **依赖校准集质量**：虽然仅需少量样本（128序列），但校准集的代表性仍影响补偿效果。
- **未处理非线性交互**：补偿基于线性变换假设，可能忽略深层非线性激活的影响。
- **超参数敏感性**：尽管默认配置表现良好，但在极端模型或任务上可能需要微调（如α、λₐ）。

### **未来工作方向**
- 将COEC扩展至**非结构化剪枝**或**量化-剪枝联合压缩**。
- 探索**多层联合补偿**策略，而非逐层独立优化。
- 结合**动态校准**机制，使补偿适应不同输入分布。
- 在更多下游任务（如生成、摘要）上验证COEC的泛化能力。

---

> **总结**：COEC通过引入双侧正交等价补偿，突破了传统训练无关补偿方法的几何限制，在不增加部署成本的前提下，显著提升了剪枝后LLMs的语言建模能力和零样本推理性能，是结构化剪枝领域的一项重要进展。

</details>

---

### 15. [Nexus: Depth-Adaptive KV-Cache Splicing and Retrieval-Decoupled Tool Routing for Agentic LLMs on Unified Memory](https://arxiv.org/abs/2608.20397)

**Authors**: Mustafa Arslan  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.20397v1  

#### Abstract
Agentic large language models (LLMs) on the Model Context Protocol (MCP) re-encode verbose tool schemas every turn, so prefill - quadratic in sequence length - dominates time-to-first-token (TTFT) as the tool registry grows. Nexus's primary lever is to decouple routing from the schema-prefill cost: ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Nexus: Depth-Adaptive KV-Cache Splicing and Retrieval-Decoupled Tool Routing for Agentic LLMs on Unified Memory*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在基于 **Model Context Protocol (MCP)** 的 Agentic LLM 系统中，模型每轮都需要重新编码大量冗长的工具（tool）JSON schema，导致 **prefill 阶段的时间复杂度为 $O(N^2)$**（$N$ 为序列长度），显著增加 **Time-to-First-Token (TTFT)**，尤其当工具注册表（registry）规模扩大时，该问题成为推理瓶颈。

此外，传统 KV-cache 复用机制（如 paged attention、prefix sharing）无法跨位置移动已编译的 KV 块，限制了缓存效率。

### 提出的新方法与创新点

Nexus 提出了两个核心机制，分别作为主次优化杠杆：

#### （1）**Retrieval-Decoupled Routing（检索解耦路由）** —— 主要贡献（Measured）
- 引入一个 **INT8 语义旁路缓冲区（Semantic Lookaside Buffer, SLB）**，通过轻量级向量检索选择候选工具。
- 使用 **calibrated cross-encoder margin gate** 对低置信度请求进行重排序，高置信度则直接路由。
- 参数生成阶段不依赖拼接的 KV-cache，而是基于一个压缩的文本签名（median 19 tokens）进行，避免将参数生成置于易受 RoPE 相位漂移影响的路径上。

> ✅ **优势**：  
> - 路由过程与上下文深度无关（depth-independent）  
> - 在 250 个工具规模下仍保持约 89% 的准确率  
> - 主上下文节省 ~80% token，首参数生成速度提升 **1.66×**

#### （2）**Depth-Adaptive KV-Cache Splicing（深度自适应 KV 缓存拼接）** —— 次要但有保障的补充机制
- 将预编译的 schema KV 块直接移植到运行时上下文中，实现零拷贝物理映射（via `mmap`）。
- 利用 **RoPE reanchor** 技术修正因位置偏移引起的相位误差。
- 当 past context 超过阈值 $P=256$ 时，启动 **depth-adaptive recompute**：逐步重计算拼接块尾部一定比例 $R(n_{\text{past}})$ 的 token，最终收敛至完整 prefill。

> ✅ **优势**：  
> - 实现“**never-regress**”保证：输出保真度始终与 full prefill 完全一致（top-1 agreement = 1.0, $D_{KL} \sim 0$）  
> - 中等深度下 TTFT 加速达 **1.1–1.7×**，随深度加深逐渐收敛至 parity（1.0×）

#### 其他系统级机制（Implemented）
- **Transposed-V Splicing**：支持 soft-capped attention 架构（如 Gemma-2）下的 V-cache 转置布局处理
- **Cache-Line Hardening**：通过内存对齐和锁优化确保多线程确定性执行
- **LO Exact-Token Radix Arena**：零分配前缀复用机制，提升 warm-up 效率

### 相比现有方法的优势

| 方法 | Nexus 优势 |
|------|-----------|
| Concatenate-all-schemas | Nexus 不会因上下文溢出而失败，在 250 工具下仍可运行；token 占用减少 ~80% |
| RAG-MCP [6] | 同样采用 retrieval 思路，但 Nexus 进一步解耦路由与 KV 拼接路径，且无需训练对齐模块 |
| vLLM / SGLang 等 in-place KV 系统 | 支持跨位置 KV 移植（relocation），突破传统系统仅能原地复用的限制 |

---

## 2. 核心实验方法和设置

### 数据集与任务
- **任务**：GitHub-MCP 场景下的工具调用（tool calling）
- **查询集**：100 个 MCP 查询用于评估路由准确性
- **工具规模**：从 10 到 250 个工具动态扩展，测试可扩展性
- **一致性测试集**：30 个 case 用于验证 sidecar 路由正确性

### 实验设置
- **硬件平台**：Apple M4 Max SoC（16-core CPU, 40-core GPU, 16-core Neural Engine）
- **内存配置**：64GB Unified Memory (UMA)，支持零拷贝物理访问
- **模型**：
  - 主模型：`Qwen2.5-14B-Instruct Q4_K_M`
  - Embedder：`nomic-embed-text-v1.5`
  - 额外验证模型：`Gemma-2-9B`（用于 transposed-V fidelity 测试）

- **代码基础**：基于 `llama.cpp`（commit: cb2463bb）构建用户态 KV-cache 服务原型

### 评估指标
| 指标 | 描述 |
|------|------|
| **TTFT (Time-to-First-Token)** | 首 token 延迟，衡量响应速度 |
| **Top-1 Agreement & $D_{KL}$** | 输出分布保真度，“never-regress” 的核心验证标准 |
| **Routing Accuracy** | 端到端工具选择准确率 |
| **SLB Recall (top-1/top-3)** | SLB 检索阶段召回能力 |
| **Argument Accuracy** | 参数填充正确率（是否满足用户请求） |
| **JSON Validity Rate** | 生成 JSON 是否语法合法 |
| **Speedup Ratio** | 相对于 full schema re-prefill 的加速比 |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Concatenate-all-schemas Oracle** | 所有 schema 拼接进 prompt，作为理想准确率上限，但在 N≥50 时超出 context window |
| **Full Schema Re-prefill (Path B)** | 每轮重新编码全部 schema，代表传统做法，TTFT 基准线（1.0×） |
| **RAG-MCP [6]** | 基于检索的方法，仅加载相关 schema，但未解决 KV 拼接问题 |

> ⚠️ 注：非 Nexus 方法的数据来自不同硬件/工作负载，仅作上下文参考，非严格 head-to-head 对比。

---

## 3. 主要实验结果和性能指标

### （1）深度自适应拼接性能（Table III）

| $n_{\text{past}}$ | $R(\%)$ | Prefill (ms) | Splice (ms) | Speedup [95% CI] | Top-1 / $D_{KL}$ |
|------------------|--------|-------------|------------|------------------|------------------|
| 256              | 5.0    | 3278        | 2010       | **1.63 [1.62, 1.67]** | OK / ~0          |
| 512              | 36.7   | 4672        | 3761       | **1.24 [1.23, 1.25]** | OK / ~0          |
| 1024             | 100.0  | 7302        | 7425       | 0.98 [0.97, 1.00]  | OK / 0           |
| 2048             | 100.0  | 13092       | 13142      | 1.00 [0.99, 1.00]  | OK / 0           |

> 🔁 使用更平缓的 recompute 曲线（K=16）可在深上下文中维持更高加速（如 1024 时达 1.17×）

✅ **关键发现**：
- “never-regress” 全局成立：所有条件下 top-1 一致且 $D_{KL} \approx 0$
- 加速效果随深度衰减，最终收敛至 parity（1.0×），符合设计预期
- 最大提速出现在中等深度（~1.6–1.7×）

---

### （2）路由准确性与可扩展性（Fig. 5）

| 工具数量 $N$ | 路由准确率 |
|-------------|----------|
| 10          | 92%      |
| 50          | 90%      |
| 100         | 89%      |
| 250         | **89% [81.4, 93.7]** |

- SLB top-1 recall: **74%**, top-3 recall: **95%**
- SLB 搜索延迟极低：
  - Python FFI 环境下：**17.6 μs**
  - 纯 C++ SIMD 扫描：**8.25 μs**

> 📌 对比：concatenate-all oracle 在 $N=10$ 达 98%，但 $N≥50$ 时 context overflow，完全失效

---

### （3）参数生成与有效性（Table IV）

| 指标 | 结果 |
|------|------|
| Sidecar 路由准确率（30 cases） | 86.7% [70.3, 94.7] |
| 已成功路由情况下的参数填充准确率（40 args） | **100% (>91.2%)** |
| 端到端参数准确率（50 args） | 80% [67.0, 88.8] |
| JSON 合法率 | **100% (≥88.6%)** |
| 压缩 IR 长度（median / p99） | **19 / 32 tokens** |
| 首参数延迟（hybrid path vs oracle） | **443.8ms vs 737.3ms → 1.66× 更快** |

> ✅ 表明压缩表示足以支撑高质量参数生成，且无 placeholder 泄漏

---

### （4）消融与负向结果（Negative Results）

#### ❌ Off-anchor RoPE Fidelity Boundary（图1）
- 锚定拼接（anchored splice）输出精确（$D_{KL} \sim 0$）
- 非锚定拼接（off-anchor）虽经 RoPE reanchor 修正，仍有残差漂移（~10⁻² nats）
- 若不修复，尾部分布扰动可能导致采样路径发散
- 设定 $P=256$ 为保守修复起点，非硬性崩溃点

#### ❌ Reference-Free Drift Gating 失败（Table II）
- 尝试使用 preceding-context K-variance 作为 drift proxy
- 实测 Spearman rank correlation $p = 0.193$（目标 ≥0.40），远低于可用水平
- 表明无法通过廉价指标预测 drift，必须依赖确定性 depth-adaptive 曲线

---

## 4. 关键结论和发现

### 主要发现
1. **KV-cache relocation 受限于 RoPE 相位漂移**，但可通过 **RoPE reanchor + depth-adaptive suffix redecode** 实现可控修复。
2. **Never-regress 是可行的**：通过渐进式重计算，可在任意深度保证输出与 full prefill 完全一致。
3. **路由应彻底脱离 KV 拼接路径**：采用 SLB + margin gate 实现高效、深度无关的工具选择。
4. **参数生成可基于压缩文本签名完成**，无需依赖易损的拼接 KV-cache。
5. **Scattered recomputation 有害**：分散式重计算反而加剧分布偏差，必须采用连续后缀重算策略。

### 局限性
| 维度 | 局限说明 |
|------|---------|
| **硬件依赖** | 物理拼接需 UMA 支持零拷贝内存映射；离散 GPU 或云 API 回退至 text-prefill（无加速） |
| **模型泛化性** | 定量边界（如 $P=256$, $K=4$）针对特定 model tuple 校准，不可直接迁移 |
| **部署成熟度** | deep-splice 路径仅微基准验证，尚未作为多轮生产路径全面测试 |
| **检索天花板** | SLB 和 cross-encoder 存在固有召回上限，难以解决语义高度混淆的工具对 |

### 未来工作方向
1. **探索更高效的 drift predictor**：结合轻量 attention kernel 或 probe-based 方法替代当前 deterministic 曲线
2. **扩展至分布式 UMA-like 架构**：研究 RDMA 或 shared-memory cluster 上的 KV-transplantation 可行性
3. **自动化 calibration pipeline**：为不同 model tuples 自动学习最优 $P$ 和 $K$ 参数
4. **融合 learned compression 与 retrieval**：结合 NTILC 类方法进一步降低 embedding 开销
5. **支持 streaming 工具调用场景**：将 Nexus 机制集成进实时 agent workflow 引擎

---

> 💡 **总结一句话**：  
> Nexus 通过 **检索解耦 + 深度自适应修复**，首次实现了在统一内存架构下安全、高效、永不退化的 KV-cache 拼接，解决了 Agentic LLM 中 schema bloat 导致的 prefill wall 问题，其核心思想（尤其是 negative results）具有广泛指导意义。

</details>

---

### 16. [FL-MAESTRO: Multi-Agent LLM Orchestration for Resource-Constrained Federated Learning](https://arxiv.org/abs/2608.20518)

**Authors**: Jiajun Wu, Zirui Wang, Jiayu Zhou, Qiang Ye, Steve Drew  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.20518v1  

#### Abstract
In Federated Learning (FL), the communication topology is a runtime variable rather than a fixed design choice, since links and edge devices drop in and out during training. Each round, the server must commit three coupled decisions, namely the communication topology, per-client resource allocation,...

---

### 17. [VortexChat: An agentic framework for autonomous multi-objective integrated photonic design](https://arxiv.org/abs/2608.20688)

**Authors**: Faqian Chong, Yulun Wu, Shilong Li, Andrew Forbes, Hongsheng Chen, Song Han  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.20688v1  

#### Abstract
The advancement of modern integrated photonics is frequently bottlenecked by device design workflows that rely heavily on manual simulation and expert intuition. While inverse design offers an alternative, it remains constrained by expert supervision and a lack of end-to-end automation. To address t...

---

### 18. [ForeTime-VLA: Causal Future-Token Distillation from a World Action Model for Conveyor-Belt Manipulation](https://arxiv.org/abs/2608.20735)

**Authors**: Siyuan Ma, Yutian Zhang, Boshi Zhang, Qinglian Wu, Jiaqi Zhai, Dong Wei, Xiaojin Huang  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.20735v1  

#### Abstract
Manipulating moving objects requires a policy to anticipate contact events, yet vision-language-action (VLA) policies are commonly fine-tuned from the current observation alone. World action models (WAMs) learn predictive dynamics, but running a video-scale teacher or explicitly imagining future fra...

---

### 19. [Don't Solve, Just Compare: Tiny Advisors for Runtime Intervention in LLM Agents](https://arxiv.org/abs/2608.21027)

**Authors**: Yanze Jiang, Mingxuan Li, Yuhao Wang, Shengfang Zhai, Jiaheng Zhang  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.21027v1  

#### Abstract
LLM agents are emerging as an important paradigm for real-world tasks that require reasoning, tool use, and sequential decision-making. As these agents operate over longer horizons, runtime intervention offers a way to improve reliability without retraining the underlying actor. Failure detection al...

---

### 20. [Socialized Division and Collaboration: Rethinking Class-Incremental Learning under Optimization Conflicts](https://arxiv.org/abs/2608.21044)

**Authors**: Xinjie Yao, Zhihe Fan, Yunqi Zhu, Jiaqi Zhou, Dengyu Zhao, Zhoupeng Guo, Yan Fan, Guosong Jiang, Pengfei Zhu  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.21044v1  

#### Abstract
Class-incremental learning is commonly instantiated as a single-model paradigm, where a unified model sequentially adapts to an unbounded stream of sessions. While effective under mild distributional shifts, this formulation becomes strained when successive sessions induce incompatible optimization ...

---

### 21. [Enhancing LLMs in Predictive Political QA with Semi-Structured Data](https://arxiv.org/abs/2608.21218)

**Authors**: Yinan Liu, Zihan Zhou, Zichun Jin, Xinyu Wang, Bin Wang, Xiaochun Yang  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.21218v1  

#### Abstract
Predictive political question answering (QA), such as predicting how a political actor will vote, goes beyond factual lookup. External political resources offer rich historical evidence, but rarely contain the answer itself. Existing LLM augmentation methods, including actor-profile-based simulation...

---

### 22. [Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score](https://arxiv.org/abs/2608.20445)

**Authors**: Junyi Liang, Hailiang Du  
**Category**: cs.LG  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.20445v1  

#### Abstract
Kernel density estimation converts finite samples into probability densities, but its performance depends critically on bandwidth selection. Classical selectors prescribe the sample-to-bandwidth rule analytically or asymptotically, or solve a new optimization for each sample. An amortized framework ...

---

### 23. [Hidden Axis of Uncertainty: Latent-Posterior Alignment in Graph Neural Networks with Bayesian Output Layers](https://arxiv.org/abs/2608.20758)

**Authors**: Suk Hoon Choi, Damdae Park, Junhyuk Choi, Hyein Jung, Changsoo Kim, Ung Lee, Kyeongsu Kim  
**Category**: cs.LG  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.20758v1  

#### Abstract
Bayesian Neural Networks (BNNs) with Bayesian output layers provide a principled and tractable framework for quantifying predictive uncertainty, yet the mechanisms shaping that uncertainty remain unclear. While conventional theory attributes uncertainty reduction to posterior contraction, the corres...

---

### 24. [Free-Probability Kernels for Zero-Rollout Hyperparameter Selection in Reservoir Computing](https://arxiv.org/abs/2608.20998)

**Authors**: Sara Malacarne, Andrea Ceni, Claudio Gallicchio  
**Category**: cs.LG  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.20998v1  

#### Abstract
Reservoir computing (RC) couples a fixed recurrent dynamical system with a trained lightweight readout, but this efficiency is partly lost during hyperparameter selection: the recurrent gain, input scale, and leakage rate determine the reservoir's stability and temporal processing regime and are usu...

---

### 25. [ConceptTS: LLM-Guided Concept Bottlenecks for Interpretable Multivariate Time-Series Forecasting](https://arxiv.org/abs/2608.21277)

**Authors**: Yichen Jiang, Yueqiao Chen, Dongyu Liu  
**Category**: cs.LG  
**Published**: 2026-08-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.21277v1  

#### Abstract
State-of-the-art multivariate time-series forecasters can model complex temporal and cross-variable dependencies, yet their opaque representations provide limited insight into why a particular forecast is produced. This lack of transparency restricts their use in settings where practitioners must un...

---

### 26. [Truth Lies Deep: Countering Semantic Camouflage via Latent Intent Verification](https://arxiv.org/abs/2608.20378)

**Authors**: Md. Hasib Ur Rahman  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.20378v1  

#### Abstract
Safety alignment in Large Language Models (LLMs) is often superficial, relying on refusal mechanisms that trigger only at the final stages of generation without erasing the foundational knowledge of harmful concepts acquired during pretraining. This study demonstrates that this architectural disconn...

---

### 27. [SPARC: Single-Pass Scaling for Motion Forecasting with Conformal Bayesian Last Layers](https://arxiv.org/abs/2608.20802)

**Authors**: Sakif Hossain, Julian Teusch, J\"org P. M\"uller  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.20802v1  

#### Abstract
Human motion forecasters are increasingly accurate and fast, but reliable deployment requires uncertainty estimates that are structured, calibrated, and efficient. Bayesian and ensemble-based uncertainty estimates often require repeated stochastic inference [15, 26], while conformal calibration alon...

---

### 28. [TLive-Omni: An Omni-Modal Understanding Model for E-Commerce Live Streaming](https://arxiv.org/abs/2608.20958)

**Authors**: Yibo Hu, Yu Qian, Mao Gu, Yingfan Tao, Yuhao Chen, Yongdong Luo, Zhuoqun Liu, Meiguang Jin, Junfeng Ma  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.20958v1  

#### Abstract
E-commerce live streaming requires omni-modal understanding of noisy, temporally extended streams, where product facts are distributed across speech, video frames, product images, overlaid text, and user queries. We present TLive-Omni, an omni-modal understanding model tailored to live-commerce scen...

---

### 29. [Generalizing Soft Tissue Deformation and Force Prediction Across Material Stiffness and Geometry](https://arxiv.org/abs/2608.20967)

**Authors**: Madina Kojanazarova, Sidaty El Hadramy, Philippe C. Cattin  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.20967v1  

#### Abstract
Accurate soft tissue simulation is essential for surgical training, pre-operative planning, and haptic feedback systems. While learning-based surrogate models trained on data using the finite element method (FEM) offer a promising path to real-time inference, their reliability depends on well-calibr...

---

### 30. [ReFrame: Evidence-Guided Test-Time Safety Alignment in Multimodal Large Language Models](https://arxiv.org/abs/2608.21100)

**Authors**: Wenzheng Jiang, Xuankun Rong, Yuanzhao Zhai, Dawei Feng, Huaimin Wang  
**Category**: cs.AI  
**Published**: 2026-08-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.21100v1  

#### Abstract
While multimodal large language models (MLLMs) extend model capabilities beyond text, they also make safety alignment increasingly challenging. Multimodal safety alignment methods must address cross-modal jailbreaks, safety-awareness failures, and over-sensitive refusals. However, existing methods o...

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

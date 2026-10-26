# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-26 11:34:30 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Training an MoE LLM in MXFP4: How We Closed the Gap to BF16](https://arxiv.org/abs/2610.32229)

**Authors**: Han Wang, Zhitao Wang, Natalia Frumkin, Xuanwu Yin, Alireza Khodamoradi, Dong Li, Emad Barsoum  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2610.32229v1  

#### Abstract
Training large language models (LLMs) in sub-8-bit arithmetic promises substantial gains in memory bandwidth and compute throughput, yet FP4 formats introduce quantization error severe enough to destabilize optimization if left unaddressed. We present ALTO, an open-source training recipe for GPT-OSS...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Training an MoE LLM in MXFP4: How We Closed the Gap to BF16*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决在**子8位精度（如 MXFP4）下训练大规模语言模型（LLM）时因量化误差导致优化不稳定**的问题，尤其是在混合专家模型（MoE）上进行训练时。MXFP4 虽然理论上可带来显著的内存带宽和计算吞吐提升（相比 BF16 有 4× 的操作数压缩），但其严重的量化噪声会导致训练发散、权重振荡（weight oscillation）以及收敛缓慢。

### 提出的新方法与创新
作者提出了一个名为 **ALTO** 的开源训练方案，用于在 MLPerf Small MoE Training 基准上成功训练 GPT-OSS-20B（200亿参数 MoE 模型）的 MXFP4 版本。其核心是组合多种已有技术并引入简化版的**权重去振荡机制（weight de-oscillation）**。

#### 主要技术组成：
- **Hybrid 1D/2D Block Quantization**：对激活使用 1D 分块量化（兼容 MX 规范），对权重使用 2D 分块（32×32），以统一前向与反向传播中的量化视图，减少开销。
- **Randomized Hadamard Transform (RHT)**：应用于梯度路径，重新分布异常值（outliers），缓解因个别大值主导缩放因子而导致其余元素精度损失的问题。
- **Stochastic Rounding (SR)**：在反向传播中对梯度进行随机舍入，消除确定性舍入带来的系统性偏差。
- **Weight De-Oscillation（本文关键创新）**：受 TetraJet-v2 启发，提出一种简化的、**优化器级别的去振荡机制**，作为 AdamW 的后处理钩子（post-step hook），无需修改模型结构即可检测并稳定频繁跳变的量化权重。

### 相比现有方法的优势
- **更高的兼容性与易用性**：提出的 de-oscillation 方法完全集成于优化器层面，不依赖特定模型架构，易于复用。
- **更强的收敛稳定性**：通过 de-oscillation，在接近收敛阶段显著抑制了量化引起的震荡，使 MXFP4 训练能更接近 BF16 性能。
- **实证有效性**：在 GPT-OSS-20B 上验证了该组合策略的有效性，实现了仅比 BF16 高 +0.007 的验证损失，并将达到质量目标的时间开销从 +20.0% 降低至 **+5.0%**。

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **C4 数据集** 的英文子集进行预训练和评估。
- 训练数据为 `c4-train.en_6_text_document` 分片；
- 验证集为 `c4-validation-91205-samples.en_text_document`，取前 1,024 个样本进行验证。
- 文本使用 **Llama-3.1 tokenizer** 进行预分词。

### 实验设置
- **模型**：GPT-OSS-20B，一个 20B 参数的稀疏激活 MoE 架构。
- **任务**：MLPerf Small MoE Training Benchmark —— 从零开始预训练，直到验证损失 ≤ 3.34（log perplexity）。
- **序列长度**：8,192 tokens
- **全局批次大小（Global Batch Size, GBS）**：16
- **最大训练步数**：1,200,000
- **优化器**：AdamW（β₁=0.9, β₂=0.95, weight decay=0.1）
- **学习率调度**：余弦退火，128 步 warmup，基础学习率 4×10⁻⁴
- **并行策略**：专家并行度为 8

### 量化配置
- 所有 Linear 层和 GroupedExperts 模块启用 MXFP4 量化。
- **例外**：`lm_head` 投影层和 MoE router gates 不参与量化（因其对扰动敏感）。
- 使用模拟 MXFP4 内核（simulate quantize/dequantize），在 AMD MI300 硬件上运行，尚未启用原生硬件加速。

### 评估指标
1. **最终验证损失（Validation Loss）**：衡量模型收敛后的准确率。
2. **Steps-to-target**：首次达到验证损失 ≤ 3.34 所需的训练步数（基于每 768 步记录一次的检查点），反映收敛速度。
3. **消融实验（Ablation Study）**：比较不同组件组合下的性能差异。
4. **算子级信噪比（SNR）**：在合成数据上测试 MXFP4 算子相对于 BF16 参考实现的数值保真度。

### 基线方法对比
- **主基线**：BF16 全精度训练
- **低精度基线**：
  - MXFP4 + Hybrid Quantization
  - MXFP4 + RHT + SR
  - NVFP4 + RHT + SR（来自 Abecassis et al.）
- 对比其他候选技术：DGE、Outlier Clipping、Macro-Block Scaling (MBS)、Low-Rank Compensation、M+Adam

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | 最终验证损失（16,128步） | 相对于 BF16 的差距 | 达到 3.34 所需步数 | 相对于 BF16 的步数开销 |
|------|--------------------------|--------------------|---------------------|----------------------------|
| **BF16 Baseline** | 3.3283 | — | 15,360 | — |
| **MXFP4 + RHT + SR** | 3.3418 | +0.0135 | 18,432 | +20.0% |
| **MXFP4 + RHT + SR + De-Oscillation** | **3.3350** | **+0.0067** | **16,128** | **+5.0%** |
| **NVFP4 + RHT + SR** | ~3.34 | — | 18,432 | +20.0% |

> ✅ **结论**：加入 **de-oscillation** 后，MXFP4 闭合了约 50% 的与 BF16 的差距，并将收敛时间开销从 +20.0% 大幅降至 **+5.0%**。

### 消融实验结果（GBS=16, Step=14,592）
| 技术 | 验证损失 | ΔLoss vs. RHT+SR 基线 |
|------|--------|-------------------------|
| RHT + SR（基线） | 3.390 | — |
| + De-Oscillation | **3.382** | **-0.008** ✅ |
| + M+Adam | 3.383 | -0.007 |
| + MBS | 3.386 | -0.004 |
| + Static Clipping | 3.390 | 0.000 |
| + Dynamic Clipping | 3.571 | +0.181 ❌ |
| + DGE | 3.591 | +0.201 ❌ |
| + Low-Rank Compensation (r=32) | 3.397 | +0.007 ❌ |

> 🔍 **发现**：
> - **De-Oscillation 是最有效的单一改进项**（Δ = -0.008）
> - **DGE 和 Dynamic Clipping 明显有害**，尽管它们在算子级 SNR 上略有提升
> - **MBS 提升了算子级 SNR**，但在端到端任务中收益有限
> - **静态剪裁无影响，低秩补偿反而轻微损害性能**

---

## 4. 关键结论和发现

### 主要发现
1. **MXFP4 可用于 MoE 模型训练**：通过合理组合量化技术（1d2d + RHT + SR），可以在 GPT-OSS-20B 上实现接近 BF16 的训练效果。
2. **Weight Oscillation 是后期收敛的关键瓶颈**：尤其在精细调优阶段，微小更新引发的量化跳变会破坏稳定性。
3. **De-Oscillation 是最有效的稳定手段**：相比其他复杂方法（如低秩补偿、DGE），它以极小代价（额外 optimizer state）带来了最大收益。
4. **算子级 SNR ≠ 端到端性能**：某些提升 SNR 的技术（如 MBS、clipping）并未转化为实际训练增益，说明需要以端到端任务为最终评价标准。
5. **De-Oscillation vs. M+Adam**：
   - 两者收敛性能相近；
   - 但 **de-oscillation 更适合 MLPerf 封闭赛道**（因不改变优化器本身）；
   - M+Adam 更适合作为开放赛道的原生低精度优化方案。

### 方法的局限性
- **未实现真实加速**：当前为**模拟执行**（simulate quantize/dequantize），没有原生 MXFP4 GEMM 支持，因此**无内存节省或速度提升**。
- **未融合算子**：量化、RHT、GEMM 等操作未融合，存在大量冗余内核调用，影响效率。
- **评估范围有限**：仅在 GPT-OSS-20B + C4 + AMD MI300 上验证，泛化性待进一步检验。
- **未提交正式 MLPerf 审计结果**：目前仅为研究原型，非正式提交版本。

### 未来工作方向
1. **实现算子融合与持久缓冲管理**：构建生产级的 fused kernel pipeline，真正释放 MXFP4 的带宽优势。
2. **支持原生 FP4 硬件调度**：在具备 MXFP4 GEMM 能力的硬件上部署，测量真实 wall-clock speedup。
3. **扩展至更大 batch size 与更长训练周期**：验证方法在不同规模下的鲁棒性。
4. **探索动态去振荡阈值与自适应机制**：提升 de-oscillation 的自动化程度。
5. **推动标准化与开源生态**：通过 ALTO 项目（[GitHub](https://github.com/amd-agi/alto)）促进社区协作与复现。

---

> 📌 **总结一句话**：  
> 本论文展示了如何通过 **ALTO 训练配方（1d2d + RHT + SR + de-oscillation）** 在 MXFP4 下高效训练 GPT-OSS-20B，**将与 BF16 的验证损失差距缩小至 +0.007，收敛步数开销降至 +5.0%**，证明了 4-bit 浮点训练在 MoE 模型上的可行性，其中 **weight de-oscillation 是关键突破点**。

</details>

---

### 2. [NgramRL: Accelerating RL Post-Training for Structured Generation via Rollout-Aware N-gram Speculative Decoding](https://arxiv.org/abs/2610.32881)

**Authors**: Nilesh Prasad Pandey, Jangseon Park, Jason Kong, Onat Gungor, Tajana Rosing  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.32881v1  

#### Abstract
Reinforcement learning (RL) post-training improves the reasoning capabilities of large language models, but its cost is dominated by autoregressive rollout generation, which can exceed 70% of training time. Existing speculative-decoding methods are poorly suited to this setting: model-based drafters...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NgramRL: Accelerating RL Post-Training for Structured Generation via Rollout-Aware N-gram Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Reinforcement Learning (RL) post-training**（尤其是 GRPO）中，**rollout generation** 是训练过程中最耗时的部分，可占总训练时间的 **70%以上**。现有的 **speculative decoding** 方法在此场景下存在以下问题：

- **Model-based drafters**（如 Medusa、EAGLE）需要额外的 GPU 内存来存储 drafter 模型或 head，且随着 policy 演化会迅速 **misaligned**（drafter 过时）。
- **Model-free drafters**（如 Prompt-lookup decoding, REST, SuffixDecoding）虽然无参数，但仅利用单次生成或静态语料库中的重复，无法有效捕捉 **跨训练步（cross-step）的重复性**。

### 提出了什么新方法或新思路
提出 **NgramRL**，一种 **rollout-aware、per-prompt 的 n-gram speculative decoding** 框架，其核心思想是：

> **Recent rollouts of the same prompt serve as a high-quality, self-updating draft source.**

具体创新包括：

- **Per-prompt n-gram cache**：为每个 prompt 构建一个基于其历史 rollout 的 n-gram 表（LUT），无需额外模型或训练。
- **Bounded history window**：只保留最近 $ W $ 步的历史，防止缓存过时并控制内存增长。
- **Asynchronous fan-out prefetcher**：通过预测验证结果 $(j, b)$（接受长度 $j$ 和 bonus token $b$），在 GPU 验证期间预取下一阶段的 draft，隐藏 CPU 查找延迟。
- **Causal budget allocator**：动态分配候选预算 $N$ 给高概率的 $(j, b)$ 分支，提升预取效率。

### 相比现有方法的优势
| 优势维度 | NgramRL | 现有方法（如 SuffixDecoding, PLD） |
|--------|--------|-----------------------------|
| **GPU 内存开销** | 零（drafter 在 CPU） | 可能占用 GPU 显存（model-based）或共享缓存竞争 |
| **对齐性** | 自更新（来自 policy 自身输出） | 固定或静态，易失配 |
| **跨步重复利用** | ✅ 显式利用同一 prompt 多轮 rollout 的重复 | ❌ 仅限单轮或全局缓存 |
| **系统效率** | 异步预取隐藏 CPU 开销 | 同步查找导致 GPU 空闲 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
覆盖自由生成与结构化生成任务，体现泛化能力：

- **GSM8K**：数学推理（free-form reasoning）
- **Spider 1.0**：text-to-SQL（structured generation）
- **CodeContests**：编程竞赛代码生成（code generation）

### 实验设置和评估指标

#### 模型
- **Qwen3-8B**
- **DeepSeek-LLM-7B**

#### 训练框架
- 使用 **verl** 框架进行 GRPO 训练
- 每步采样 5 个 completion，共 64 个 prompts，持续 15 个 epoch
- Rollout 使用 **vLLM** 推理引擎

#### 评估指标
- **Tokens Per Step (TPS)**：每轮 target model forward 所 commit 的 token 数量（即 $1 + \mathbb{E}[j]$）
  - 报告两个值：全训练平均 / 最后 30% 步骤平均（“warm-cache” regime）
- **Coverage $c$**：prefetch 成功命中真实 $(j^*, b^*)$ 的比例
- **Speedup**：实测端到端 rollout 吞吐提升

### 基线方法对比
所有 baseline 使用相同 draft budget $k=16$：
- **PLD (Prompt-lookup decoding)** [18]
- **REST** [7]
- **SuffixDecoding** [16]

---

## 3. 主要实验结果和性能指标

### 关键性能数据（TPS，warm-cache regime）

| Method | GSM8K | Spider | CodeContests |
|-------|------|--------|------------|
| **Qwen3-8B** | | | |
| PLD | 1.31 | 1.11 | 1.30 |
| REST | 1.31 | 1.02 | 1.07 |
| SuffixDec. | 2.32 | 4.15 | 2.96 |
| **NgramRL (Ours)** | **4.36** | **7.77** | **3.60** |
| | (**1.88×**) | (**1.87×**) | (**1.22×**) |
| **DeepSeek-7B** | | | |
| PLD | 1.49 | 1.24 | 1.15 |
| REST | 1.69 | 1.02 | 1.02 |
| SuffixDec. | 3.64 | 3.88 | 3.99 |
| **NgramRL (Ours)** | **6.37** | **6.81** | **6.19** |
| | (**1.75×**) | (**1.75×**) | (**1.55×**) |

> ✅ **NgramRL 在 warm-cache 下达到 3.60–7.77 TPS**，最高比 SuffixDecoding 快 **1.99×**，比 PLD 快 **7.0×**

### 与基线方法的对比结果
- 在 **Spider** 上表现最优，因 SQL 输出具有强语法规律性和 prompt 特异性。
- 在 **GSM8K** 上也有显著提升，说明即使自由推理也存在跨步重复模式。
- **DeepSeek-7B** 上整体优于 Qwen3-8B，可能因其更强的记忆一致性。

### 消融实验结果

#### (1) n-gram order $n_{\text{max}}$ 与 speculation length $k$
- $n_{\text{max}} > 6$ 后收益饱和，cache 大小超线性增长 → 设为 6
- $k=16$ 时 TPS 接近饱和 → 设为 16

#### (2) Per-prompt vs Global Cache（表 2）
| 配置 | Plateau TPS |
|------|-------------|
| Global bucket | 2.41 |
| Per-prompt keys | **3.05** |
| SuffixDecoding | 2.43 |

> ✅ **per-prompt keying 是性能增益主因**，而非单纯缓存机制

#### (3) History Window $W$（表 3）
- $W=4$ 即可捕获 79–91% 的最终性能
- 更大窗口（如 $W=\infty$）反而轻微下降（旧 rollout 失效）

#### (4) Prefetcher 覆盖率与加速效果（图 7）
| 任务 | $N=16$ 时 coverage $c$ | 实测 speedup | 理论上限 |
|------|--------------------------|--------------|-----------|
| **Spider (Qwen3-8B)** | 0.966 | **1.38×** | 1.40× |
| **Spider (DeepSeek-7B)** | 0.995 | **1.21×** | 1.21× |
| GSM8K | ~0.65 | ~1.05× | 1.10× |
| CodeContests | ~0.49 | ~1.06× | 1.14× |

> ✅ **Prefetcher 成功将 host round-trip 隐藏，接近理论加速极限**

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Prompt-conditioned temporal locality 存在且强大**：
   - 同一 prompt 的 rollout 在训练过程中高度重复，尤其在结构化任务上。
   - 早期 rollout 可覆盖后期 rollout 的 **48–96%** 内容。

2. **Rollouts 自身就是最好的 drafter**：
   - 无需额外模型，policy 自己的历史输出即可作为高质量 draft 来源。
   - 缓存自动对齐 policy 演化，避免 misalignment。

3. **System co-design 至关重要**：
   - 单纯加快 CPU 查找无效，必须通过 **asynchronous prefetching** 将 host 工作从关键路径移除。
   - **Outcome-aware prefetching**（预测 $(j,b)$）是实现高覆盖率的关键。

4. **结构化任务受益最大**：
   - 如 Spider（text-to-SQL）中，重复模板、schema tokens 导致极高 acceptance。
   - 自由生成任务（如 GSM8K）也有可观增益。

### 方法的局限性
- **依赖 prompt 重复性**：若 prompt 不重复或变化频繁，缓存难以积累。
- **CPU-hosted cache 带宽瓶颈**：当 T_gpu 较小时（小 batch 或小模型），prefetch chain 可能耗尽 GPU 时间。
- **当前未支持 streaming 或 online RL 场景**，假设 prompt 可多次访问。

### 未来工作方向
- 扩展至 **multi-turn dialogue** 或 **agent workflow** 中的 sub-task caching。
- 结合 **learned drafter** 与 **n-gram cache**，形成 hybrid drafting 策略。
- 支持 **dynamic prompt scheduling** 以最大化 cache hit rate。
- 探索 **distributed cache sharing** 跨多个 worker 节点。

--- 

> 📌 **一句话总结**：  
> **NgramRL 利用 RL 训练中同一 prompt 的 rollout 重复性，构建 per-prompt n-gram cache 并结合异步预取机制，在零 GPU 开销下实现了高达 7.77 TPS 和 1.38× 端到端加速，显著优于现有 speculative decoding 方法。**

</details>

---

### 3. [LazyFlash: Forward-Dense, Backward-Sparse Attention for Accelerated Long-Context Fine-Tuning](https://arxiv.org/abs/2610.31911)

**Authors**: Keqi Deng, Ruchao Fan, Jinyu Li  
**Category**: cs.CL  
**Published**: 2026-10-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.31911v1  

#### Abstract
As large language models (LLMs) process increasingly long sequences, attention becomes a major computational bottleneck. Sparse attention mostly targets inference, yet many methods still train densely or cannot support sparse training, and their added complexity makes sparsity hard to turn into wall...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*LazyFlash: Forward-Dense, Backward-Sparse Attention for Accelerated Long-Context Fine-Tuning*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
随着 Large Language Models (LLMs) 处理的序列长度不断增长（从数千到数十万 tokens），**self-attention 机制在训练阶段成为主要计算瓶颈**。尽管已有大量稀疏注意力（sparse attention）方法，但它们大多聚焦于推理阶段的加速，而训练过程仍保持密集计算，导致实际训练速度提升有限。此外，许多稀疏训练方法改变了前向传播结构，引入了训练-推理不一致（train-inference mismatch），限制了其部署兼容性。

### 提出的新方法：LazyFlash
论文提出 **LazyFlash** ——一种**前向密集、后向稀疏**（forward-dense, backward-sparse）的注意力机制，专门用于加速长上下文微调（fine-tuning）。

#### 核心思想：
- **前向传播（Forward Pass）保持完全密集且精确**：确保训练后的模型仍是标准的 dense-attention 模型，可直接由 vLLM 等成熟推理框架服务，无需任何修改。
- **后向传播（Backward Pass）引入块级稀疏性**：基于每个 query block 的输出梯度范数（output-gradient norm）决定是否跳过该 block 的计算，从而大幅减少反向传播中的冗余计算。

### 创新设计
1. **Gradient-Aware Block Sparsification**  
   - 使用 `||dO_m||`（第 m 个 query block 的输出梯度 Frobenius 范数）作为重要性评分。
   - 高梯度范数表示该 block 对参数更新贡献大，应保留；低者则可安全跳过。

2. **Gradient-Mass Budget 控制稀疏程度**  
   - 不采用固定的 top-k 或固定比例，而是设定一个可丢弃的“梯度质量”预算 δ（如 δ=0.1 表示最多丢弃 10% 的总梯度能量）。
   - 自适应地决定每一步、每一头中可以跳过的 block 数量，使稀疏程度随梯度分布动态变化。

3. **Importance Sampling with Unbiased Reweighting**  
   - 采用重要性采样策略：block m 被保留的概率 $ q_m \propto ||dO_m|| $。
   - 对保留的 block 进行重加权 $ 1/q_m $，保证梯度估计是无偏的（unbiased），避免因持续忽略某些 block 导致训练偏差。

4. **内置于 FlashAttention 后向核中实现真实加速**  
   - 在 tiled FlashAttention backward kernel 内部集成选择逻辑。
   - 被跳过的 block **永远不会被加载进显存或参与计算**，真正节省内存带宽和算力。

### 相比现有方法的优势
| 方法 | 是否改变前向？ | 是否支持标准推理？ | 是否真正降低训练时间？ | 是否无偏？ |
|------|----------------|--------------------|--------------------------|-----------|
| Dense Attention | 否 | 是 | 否 | 是 |
| MoBA / SeerAttention / DSA | 是 | 否（需特殊解码） | 有限（额外路由开销） | 否 |
| Training-Free Methods (H2O, Quest) | 否 | 是 | ❌ 不加速训练 | — |
| **LazyFlash** | **否** | ✅ **是** | ✅ **显著加速训练** | ✅ **是** |

> LazyFlash 成功将理论上的 FLOPs 减少转化为真实的 wall-clock 时间加速，同时保持模型结构不变，极具实用价值。

---

## 2. 核心实验方法和设置

### 数据集
- **文本任务**：  
  - 微调数据：LongAlign-10k（约10K条长上下文指令样本）与 ShareGPT 指令数据按 1:1 混合。
  - 评估基准：**LongBench-Pro**，包含 1,500 个英中文本样本，涵盖 11 类主任务，输入最长可达 256K tokens。
- **语音任务**：  
  - 训练数据：50,000 小时英文长录音（单段长达半小时）。
  - 测试集：10 分钟长语音 ASR 测试集，覆盖会议、银行、医疗等场景。

### 模型架构
- **文本 LLM**：Qwen3-8B-Base（及扩展至 Qwen3-4B 和 Qwen3-14B 验证泛化性）
- **语音 LLM**：Qwen3-8B + Conformer 编码器（冻结语言模型部分，仅训练编码器和 LoRA 层）

### 实验设置
- 最大序列长度：65,536（text），32,768（speech）
- 硬件平台：H200 GPU
- 精度：bf16 mixed precision
- 优化器：AdamW（β₁=0.9, β₂=0.95）
- LazyFlash 参数：block size = 128，budget δ = 0.1（text），δ = 0.4（speech）

### 基线方法对比
- **Dense Attention**（基于 FlashAttention-2）：标准全连接注意力，作为性能与质量基准。
- **MoBA**（Mixture of Block Attention）：代表性的可训练稀疏注意力方法，用于比较训练效率与最终质量。

### 评估指标
- **下游任务性能**：
  - 文本：LongBench-Pro 平均得分（越高越好）
  - 语音：Word Error Rate (WER) 和 Entity Error Rate (EER)（越低越好）
- **训练效率**：
  - Attention 前向/后向耗时（ms）
  - 端到端训练步时间（end-to-end step time）
  - 加速比（speedup）

---

## 3. 主要实验结果和性能指标

### 下游任务性能（质量保持）
#### ✅ 文本理解（LongBench-Pro）
| 方法 | Overall Score |
|------|---------------|
| Dense Attention | 29.64 |
| **LazyFlash** | **30.24** ✅ |
| MoBA（dense inference） | 29.37 |

> LazyFlash **不仅未损失精度，反而略有提升**，尤其在 Hard 难度上表现更优（27.51 vs 25.40）。

#### ✅ 长语音识别（ASR）
| 方法 | EER ↓ | WER ↓ |
|------|--------|--------|
| Dense Attention | 14.80 | 16.16 |
| **LazyFlash** | **12.77** ✅ | **14.90** ✅ |
| MoBA（dense infer） | 15.76 | 17.34 |

> LazyFlash 在语音任务上取得显著进步，表明其对冗余音频帧的有效过滤有助于捕捉关键信息。

---

### 训练效率（显著加速）
#### Attention Kernel 级别加速（vs Dense Attention）
| 上下文长度 | Backward Speedup | End-to-End Step Speedup |
|------------|------------------|-------------------------|
| 8K         | 2.77×            | 1.20×                   |
| 16K        | 3.97×            | 1.36×                   |
| **32K**    | **5.72×**        | **1.66×**               |

> 随着上下文增长，LazyFlash 的优势愈发明显，因为 attention 的二次复杂度使得后向占比越来越高。

#### 单步训练时间分解（Qwen3-8B, 32K context）
| 方法 | Forward (ms) | Backward (ms) | Total Attn (ms) |
|------|--------------|---------------|-----------------|
| Dense Attention | 547.3 | 1429.8 | 1977.1 |
| **LazyFlash** | **562.2**（≈） | **507.4**（↓2.82×） | **1069.6**（↓1.85×） |

> 后向时间减少超过 **2.8倍**，前向几乎无额外开销。

---

### 消融实验（Ablation Studies）

#### 不同选择策略对比（Table 3）
| Selection Rule | Score | Attn Bwd (ms) |
|----------------|-------|----------------|
| Dense Attention | 29.64 | 1429.8 |
| Random Dropping | 21.18 | 523.8 |
| Top-k (Attention Mass) | 27.07 | 548.0 |
| Top-k (Gradient Norm) | 28.45 | 508.0 |
| **LazyFlash**（Budget + Sampling） | **30.24** | **507.4** |

> 结论：
> - 仅靠随机跳过会严重损害性能。
> - 使用 attention mass 排序不如 gradient norm 有效。
> - 固定比例 top-k 不如 gradient-mass budget 自适应灵活。
> - **只有 LazyFlash 能完全匹配甚至超越 dense training 性能**。

#### 对 budget δ 的敏感性分析
- **文本任务**：最佳 δ = 0.1，超过后性能下降（δ=0.3 时降至 22.32）。
- **语音任务**：容忍更高稀疏度，δ=0.4 时性能最优，δ=0.5 导致训练发散。
> 表明 LazyFlash 可根据不同任务调整预算，在冗余高的任务中可进一步提速。

#### 模型规模扩展（Table 5）
在 Qwen3-4B、8B、14B 上均验证有效性：
- 所有尺寸下 LazyFlash 均优于 dense training。
- 后向加速比稳定在 ~2.8–2.9×。
> 表明方法具有良好的可扩展性和通用性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **注意力训练的主要成本在于后向传播**，而非前向。
2. ✅ **梯度分布高度稀疏**：大量 query blocks 的输出梯度极小，对参数更新贡献可忽略。
3. ✅ **通过梯度感知的块级稀疏化可在不损失性能的前提下大幅加速训练**。
4. ✅ **LazyFlash 实现了真正的 wall-clock 加速**（最高达 5.7× backward，1.7× 端到端），且不影响推理兼容性。
5. ✅ **重要性采样 + gradient-mass budget 的组合优于固定规则或随机策略**，是实现高效无偏训练的关键。

### 方法的局限性
- 当前依赖于 FlashAttention 的 tiled 架构，需定制 CUDA kernel 支持。
- 对极度稀疏或异常梯度分布的任务可能需要手动调节 budget δ。
- 目前主要用于 fine-tuning 场景，大规模 pre-training 中的表现有待验证。

### 未来工作方向
- 将 LazyFlash 扩展至其他 attention 变体（如 grouped-query attention）。
- 探索自动学习 budget δ 的机制，实现完全自适应稀疏。
- 应用于多模态长序列建模（视频、基因组等）。
- 结合 activation checkpointing 进一步优化显存使用。

---

> **总结**：LazyFlash 提供了一种优雅而实用的解决方案——**在不改变模型结构的前提下，通过后向稀疏化实现了训练加速**。它打破了“稀疏即牺牲兼容性”的传统困境，为长上下文 LLM 的高效微调提供了新的范式。

</details>

---

### 4. [EDWARD-NMT: Elastic Dimensions With Adaptive Routing and Dual-consistency for Neural Machine Translation](https://arxiv.org/abs/2610.32115)

**Authors**: Edward Nguyen, Thai Tran, Kiet Anh Hoang, Duy Hoang, Duy Vo, Khanh Le, Vu Le  
**Category**: cs.CL  
**Published**: 2026-10-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.32115v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) is a powerful paradigm for scaling Neural Machine Translation (NMT) and managing language heterogeneity. However, practical limitations hinder its deployment on resource-constrained devices: the entire expert pool must remain in memory, and the per-token compute budge...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EDWARD-NMT: Elastic Dimensions With Adaptive Routing and Dual-consistency for Neural Machine Translation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Sparse Mixture-of-Experts (MoE)** 架构在神经机器翻译（NMT）中虽能提升模型容量并处理语言异质性，但在资源受限设备上的部署面临两大瓶颈：
1. **内存瓶颈**：整个专家池必须常驻内存，无法灵活压缩模型大小。
2. **计算僵化**：采用固定的 `top-k` 路由机制，每个 token 的计算预算静态不变，无法根据输入复杂度动态调整。

这导致模型难以实现“一次训练、多场景部署”（train once, deploy anywhere）的目标。

### 提出的新方法：EDWARD-NMT
作者提出 **EDWARD-NMT**，一种新型 MoE 编码器-解码器架构，具备三大核心创新：

#### （1）Matryoshka-style 弹性维度（Elastic Dimensions）
- 将专家网络设计为支持嵌套内层维度 $ D \subseteq \{256, 512, 1024, 2048\} $ 的共享参数结构。
- 推理时可通过简单的矩阵切片（matrix slicing）提取不同规模的子网络（sub-network），无需重新训练或微调。
- 引入 **方差保持缩放因子**（variance preservation scaling factor）以稳定不同宽度下的激活值分布，确保联合训练稳定性。

#### （2）自适应 Sigmoid 路由器（Adaptive Sigmoid Routing）
- 替代传统的 `top-k` 路由，使用基于 **sigmoid 函数** 的门控机制。
- 每个 token 动态生成一个阈值 $ T $，仅当专家得分高于该阈值时才被激活。
- 阈值由可学习的、任务条件化的方向嵌入（`edir`）控制，实现对翻译方向的感知。
- 使用 **Straight-Through Estimator (STE)** 实现离散路由决策的梯度传播。

#### （3）双重一致性目标（Dual-Consistency Objective）
- 在训练过程中随机采样两个不同的容量维度 $ D_A, D_B $，进行双前向传播。
- 使用对称的 **Kullback-Leibler 散度**（Jeffreys Divergence）对齐不同尺度下的预测分布：
  $$
  \mathcal{L}_{dc} = \text{KL}(sg(P_A)\|P_B) + \text{KL}(sg(P_B)\|P_A)
  $$
- 当 $ D_A \neq D_B $：实现 **in-place 自蒸馏**，大模型指导小模型。
- 当 $ D_A = D_B $：退化为类似 R-Drop 的一致性正则化，增强鲁棒性。

此外，还引入了 **方向感知共享专家**（DASE），通过 FiLM 调制提供稳定的基础表示，避免稀有模式干扰。

### 相比现有方法的优势
| 维度 | EDWARD-NMT | 传统 MoE / 其他弹性方法 |
|------|------------|------------------------|
| 内存灵活性 | ✅ 支持任意宽度切片 | ❌ 固定宽度或需后处理 |
| 计算动态性 | ✅ 每 token 动态决定专家数 | ❌ 固定 top-k |
| 子网络质量 | ✅ 无需微调即可高性能 | ❌ 需 fine-tune 或 NAS |
| 部署效率 | ✅ 单一 checkpoint 支持多种配置 | ❌ 多模型或多阶段训练 |

---

## 2. 核心实验方法和设置

### 数据集
在四个标准基准上进行了广泛评估：

| 数据集 | 任务 | 特点 |
|-------|------|------|
| **PhoMT** | En↔Vi | 高质量越南语-英语翻译，涵盖新闻、字幕等多领域 |
| **WMT-14** | En→De | 经典英德翻译任务，使用 newstest2014 测试集 |
| **Multi-domain De→En** | De→En | 包含 KORAN、IT、MEDICAL、LAW、SUBTITLES 五个领域，宏平均评估 |
| **OPUS-16** | 多语言翻译 | 16 种语言，按高低资源划分，测试泛化能力 |

### 实验设置
- **模型结构**：
  - Encoder-Decoder 结构，$ L_{enc}=L_{dec}=8 $
  - $ d_{model}=512 $, 8 个注意力头
  - 每层 MoE 含 4 个路由专家 + 1 个 DASE 共享专家
  - 最大内维 $ M=2048 $，支持 $ D=\{256,512,1024,2048\} $ 四种子网络
- **训练细节**：
  - 使用 TPU v5-64，AdamW 优化器
  - 批大小 8192，序列长度 128
  - 训练步数：WMT-14 为 60k，其余约 16k–40k
  - 双重一致性损失权重 $ \lambda_{dc}=1.0 $

### 评估指标
| 指标 | 用途 |
|------|------|
| **sacreBLEU** | 主要翻译质量指标（PhoMT, WMT-14, OPUS-16） |
| **tokenized BLEU** | 多域任务评估 |
| **COMET-22** | 语义保真度评估（更高分表示更接近人类判断） |
| **chrF2** | 字符级 n-gram 匹配 |
| **TER** | 翻译编辑率（越低越好） |
| **TPS (Tokens Per Second)** | 推理吞吐量，衡量效率 |

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **Dense Baselines** | Transformer BASE, Transformer BIG, mBART |
| **MoE Baselines** | Switch Transformer (ST-MoE), GShard (GS-MoE) |
| **其他 MoE 变体** | LS-MOE, HYBRID-MOE, RESIDUAL-MoE, LINGUAL-SMOE, THOR-MOE |

所有 EDWARD-NMT 子网络均从**同一个 checkpoint** 提取，无需额外训练。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 2–4）

#### 📊 PhoMT (En→Vi) 和 WMT-14 (En→De)
| 方法 | En→Vi (BLEU) | De→En (BLEU) | Active Params |
|------|--------------|---------------|-------------|
| Transformer BIG | 42.94 | 31.48 | 210M |
| **EDWARD-NMT-XL** | **43.79** | **32.96** | **158M** |
| **EDWARD-NMT-S** | **42.91** | **31.94** | **51–59M** |

> ✅ **EDWARD-NMT-S** 参数量仅为 Transformer BASE 的 ~1.2×，但 BLEU 显著领先（+0.69～+3.18），且优于更大的 Transformer BIG。

#### 📊 多域 De→En 平均 BLEU
| 方法 | AVG BLEU | Active Params |
|------|----------|-------------|
| Transformer BIG | 40.13 | 210M |
| GS-MoE | 38.72 | 260M |
| **EDWARD-NMT-L** | **40.28** | **102M** |
| **EDWARD-NMT-S** | **38.65** | **57M** |

> ✅ EDWARD-NMT-L 用不到一半的活跃参数超越 Transformer BIG；S 版本也优于 GS-MoE（260M）。

#### 📊 多语言 OPUS-16
| 方法 | En→XX | XX→En |
|------|--------|--------|
| ST-MoE | 29.68 | 33.89 |
| LINGUAL-SMOE | 31.46 | 33.96 |
| **EDWARD-NMT-XL (L=8)** | **33.30** | **37.27** |

> ✅ 在 L=8 设置下全面领先，尤其在低资源语言表现优异。

#### 📊 语义质量（COMET-22）
| 方法 | WMT-14 (AVG) | Multi-domain (AVG) |
|------|-------------|--------------------|
| Transformer BASE | 80.78 | 80.20 |
| Transformer BIG | 83.69 | 81.62 |
| **EDWARD-NMT-S** | **83.48** | **80.96** |
| **EDWARD-NMT-XL** | **84.30** | **81.80** |

> ✅ 即使是最小的 S 版本，在语义保真度上也远超 BASE，并接近甚至超过 BIG。

### 消融实验结果（Tables 6–8, B.4–B.7）

#### 🔍 组件消融（Multi-domain）
| 设置 | XL | L | M | S |
|------|----|----|----|----|
| MoE(k=2) | 40.35 | 39.64 | 39.21 | 38.24 |
| +ME (Matryoshka) | 40.96 | 40.26 | 39.35 | 38.34 |
| +ME+DC | 40.66 | 40.16 | 39.36 | 38.49 |
| **Full (ME+DC+Sigmoid)** | **40.75** | **40.28** | **39.56** | **38.65** |

> ✅ 所有组件均有增益，尤其是 DC 对小模型至关重要。

#### 🔍 DASE 共享专家作用
| 设置 | XL | S |
|------|-----|-----|
| No SE | 39.75 | 38.13 |
| Common SE | 39.95 | 38.46 |
| **DASE (ours)** | **40.74** | **38.72** |

> ✅ DASE 提供约 +1.0 BLEU 提升，验证方向感知共享的重要性。

#### 🔍 一致性损失对比
| 损失函数 | XL | S |
|--------|-----|-----|
| R-Drop | 40.00 | 38.47 |
| Asym. KL | 40.76 | 38.59 |
| **DC Loss (ours)** | **40.75** | **38.65** |

> ✅ 双向对称 KL 在平衡大模型性能与小模型蒸馏方面最优。

#### 🔍 方差保持机制（Table 15）
移除 $ \alpha(D_k) $ 导致系统性下降，尤其在小模型上（S 下降 0.37 BLEU），证明其必要性。

---

## 4. 关键结论和发现

### 主要发现
1. **真正实现“train once, deploy anywhere”**  
   从单一 checkpoint 可无缝提取多个高性能子网络（S/M/L/XL），适用于边缘设备到服务器各种场景。

2. **结构弹性 + 动态路由 = 高效推理前沿**  
   - 内存通过 Matryoshka 切片灵活控制；
   - 计算通过自适应 sigmoid 路由动态调节；
   - 平均每 token 仅激活 1.34–1.70 个专家（见 Table 5）。

3. **双重一致性是关键正则化机制**  
   成功将大模型的知识注入小模型，同时防止小模型过度正则化破坏表达力，形成良性循环。

4. **极致参数效率**  
   - EDWARD-NMT-L（102M） > Transformer BIG（210M）
   - EDWARD-NMT-S（57M） > GS-MoE（260M）
   - 达成 **3.6× 更少活跃参数 + 更高 BLEU**

5. **边缘部署优势显著**  
   - 在 Apple M4 上，S 版本比 XL 快 **2.2×**
   - 在 H100 上批处理时，S 版本吞吐达 BASE 的 **3.2×**

### 局限性
1. **架构范围有限**  
   当前仅在 encoder-decoder 架构验证，未扩展至 decoder-only LLM 或更大规模多语言场景。

2. **弹性维度单一**  
   目前只支持 FFN 层宽度变化，未探索深度早退（early exit）、隐藏维度 $ d_{model} $ 弹性等多维适应。

3. **压缩极限受限**  
   若低于训练过的最小宽度（如 D=128），性能急剧下降，说明几何边界需显式优化。

4. **训练开销略高**  
   因 dual-consistency 需双 forward pass，单步耗时约为标准 MoE 的 2.2×，但摊销后仍更高效。

### 未来工作方向
- 扩展至 **decoder-only LLM** 和 **大规模多语言建模**
- 实现 **多维弹性**：结合动态深度、宽度、精度
- 探索 **连续宽度公式化** 或 **运行时扩展下界**
- 应用于 **语音识别、视觉生成** 等序列生成任务
- 进一步降低训练成本，如异步 dual-pass 或缓存机制

---

> 💡 **总结一句话**：  
> EDWARD-NMT 通过 **弹性维度 + 自适应路由 + 双重一致性**，首次实现了无需微调即可从单一 MoE 模型中提取一系列高性能子网络，为资源感知的 NMT 部署提供了实用、高效的解决方案。

</details>

---

### 5. [J-Comm: Jacobian-Guided Latent Communication between LLM Agents](https://arxiv.org/abs/2610.32871)

**Authors**: Xiangwei Wang, Wei Wang, Sachith Seneviratne, Ken Chen, Yu Xia, Peng Wang, Saman Halgamuge  
**Category**: cs.CL  
**Published**: 2026-10-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.32871v1  

#### Abstract
Latent communication requires both an effective continuous reasoning process and an interface through which an external model can leverage its intermediate computation. We present J-Comm, a framework comprising two core components. First, a training-free Geometric Agreement for Latent Reasoning (GEA...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：J-Comm: Jacobian-Guided Latent Communication between LLM Agents**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统多智能体系统（multi-agent systems）通常通过**文本形式**进行通信（如 chain-of-thought、debate），这种方式存在以下问题：
- 生成大量冗余文本，增加推理开销；
- 信息压缩效率低，难以传递中间推理状态；
- 接收方需重新解析文本，无法直接利用发送方的中间计算。

本文提出在**潜空间（latent space）中进行连续推理与通信**，即“**Latent Communication**”，旨在实现更高效、更精确的模型间协作。

---

### **提出的新方法与创新点**

#### **(1) GEAR：基于几何一致性的连续推理机制**
- **核心思想**：在构建下一个连续输入时，不仅考虑 token 的预测概率，还引入其在 embedding 空间中的**几何一致性（geometric agreement）**。
- **具体做法**：
  - 对候选 token 的 embedding 进行加权平均，权重为 `probability × geometric support`。
  - 几何支持由归一化 embedding 间的 cosine 相似性决定。
- **优势**：
  - 不需要训练反馈模块（training-free）；
  - 单调优化 embedding 空间中的共识，提升推理稳定性。

> ✅ 创新点：首次将 embedding 空间的几何结构显式用于连续推理输入构造。

---

#### **(2) H+J 双通道潜通信协议**
设计了一个新型通信接口，使 Receiver 能接收两种互补视图：

| 通道 | 内容 | 作用 |
|------|------|------|
| **H 通道** | 发送方的 feedback 分布映射到 Receiver 的 input embedding 空间 | 提供可继续推理的输入表示 |
| **J 通道** | 发送方中间状态在 **Jacobian-lens 坐标系**下的稀疏读出（readouts） | 提供计算过程中的额外引导 |

- **关键技术细节**：
  - 使用共享 token ID 定义 anchor；
  - 发送方传输 top-P token 的 signed readouts；
  - 接收方用 dual dictionary 重构向量并注入 prefill 阶段；
  - 支持不同宽度模型（ds ≠ dR）、KV cache 不共享。

> ✅ 创新点：结合 distribution-level 输入与 state-level 引导，实现跨模型、跨架构的高效通信。

---

### **相比现有方法的优势**

| 维度 | J-CoMM 的优势 |
|------|----------------|
| **效率** | 减少 19.2% ~ 74.8% 的生成文本 token 数量 |
| **性能** | 在多个复杂任务上显著优于 TextMAS 等文本通信方法 |
| **兼容性** | 支持异构模型（cross-scale, cross-family），无需对齐 hidden width 或 KV cache |
| **无需训练** | 所有组件均为 training-free，仅依赖冻结模型和 calibration |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

| 数据集 | 类型 | 任务描述 |
|--------|------|----------|
| **GSM8K** | 数学推理 | 小学数学应用题 |
| **MATH500** | 数学推理 | 高中竞赛级数学题 |
| **GPQA-Diamond** | 多选问答 | 研究生级别科学问题（Google-proof） |
| **MedQA-USMLE** | 医学多选 | 美国医师执照考试题 |
| **MBPP+** | 编程生成 | Python 函数实现（含测试） |
| **AIME2024+2025** | 数学竞赛 | AIME 级别综合数学题 |
| **ALFWorld** | 规划与交互 | 文本版具身环境任务（如找钥匙开门） |

---

### **实验设置**

- **主实验配置**：`Qwen3-8B → Qwen3-4B`（跨尺度）
- **通信长度 M**：默认 400 个 latent 步骤
- **J 通道参数**：每步取 top-32 token anchors
- **评估指标**：
  - 数学/多选：准确率（accuracy）
  - 编程：pass@1
  - ALFWorld：任务成功率
  - 效率：生成的 text token 数量

---

### **基线方法对比**

| 基线方法 | 描述 |
|---------|------|
| **TextMAS** | 发送方生成完整文本计划，接收方阅读后作答（标准文本通信） |
| **Receiver-only** | 无任何消息输入 |
| **Identity40 / Identity(TGAF)** | 使用恒等映射的 latent communication，作为消融对照 |
| **LatentMAS**, **StateBridge**, **CIPHER** | 当前主流潜通信方法 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Qwen3-8B → Qwen3-4B）**

| 方法 | GSM8K | MATH500 | GPQA-Diamond | MedQA | MBPP+ | AIME | ALFWorld |
|------|-------|----------|---------------|--------|--------|--------|-----------|
| **TextMAS** | 93.25% | 93.20% | 45.96% | 78.55% | 71.16% | 68.33% | 61.39% |
| **GAF+H+J26** | **94.54%** | **94.80%** | **54.55%** | **80.52%** | **76.72%** | **73.33%** | **62.93%** |

> 🔺 **最高提升达 +8.59 pp（GPQA）**，且在所有 7 项任务中均取得最佳成绩。

---

### **与基线方法的对比结果**

- 相比 **TextMAS**：
  - 平均提升 **1.29 ~ 8.59 个百分点**；
  - 文本 token 生成减少 **19.2%（AIME）至 74.8%（ALFWorld）**；
  - 尤其在高难度任务（如 GPQA、AIME）上优势明显。

- 相比其他潜通信方法（同构设置）：
  - 在 `Qwen3-4B→4B` 上，MATH500 达 **92.00%**，优于 LatentMAS (83.20%)、CIPHER (80.40%)；
  - 在 `Qwen3-8B→8B` 上，MATH500 达 **95.20%**，领先所有 baseline。

---

### **消融实验结果**

#### **(1) GAF 反馈机制的有效性**
- 在固定 H+J 通信下比较不同 feedback 规则：
  - **GAF vs Soft Thinking**：在 MATH500 上 +3.40%，GPQA 上 +5.56%
  - **GAF vs Self-only sharpening**：分别 +3.20% 和 +4.55%
- 结论：**几何一致性重加权显著优于纯概率加权或自锐化**

#### **(2) J 通道的作用**
- **GAF+H**（仅有 H 通道）：AIME 得分为 72.78%
- **GAF+H+J26**（加入 J）：提升至 **73.33%**
- **Shuffled-J 控制组**（J 来自其他问题）：性能下降，说明 J 中包含语义相关信号

#### **(3) 注入层的影响**
- 测试 Receiver 注入层（block 20, 26, 29）：
  - **block 26 表现最优（73.33%）**
  - block 29 反而下降至 71.67%，表明并非越深越好

#### **(4) 通信长度影响**
- M=40：70.56%
- M=400：**73.33%**（最佳）
- M=500：72.78%（轻微回落）
- 结论：存在**最优通信长度窗口**

---

## **4. 关键结论和发现**

### **主要发现**

1. ✅ **几何一致性（GEAR）能有效提升连续推理质量**  
   —— 显式建模 candidate token 的 embedding 几何关系，优于传统 soft averaging。

2. ✅ **双通道 H+J 通信显著优于单通道或文本通信**  
   —— H 提供输入，J 提供中间状态引导，二者协同增强推理效率与准确性。

3. ✅ **J 通道传递的信息具有语义可解释性**  
   - 案例分析显示，J 坐标中高频出现与任务相关的词汇（如 `apoptosis`, `rotations`, `factorial`）；
   - 表明 Jacobian-lens 能捕捉到关键推理概念。

4. ✅ **支持跨尺度、跨家族模型通信**  
   - 成功实现 `Qwen → Llama` 的跨 family 通信，在 GPQA 上达到 75/198 正确（vs TextMAS 73）

---

### **局限性**

1. ❗ **依赖 token 对应关系和 calibration**  
   - 需预先建立 token 字符串匹配或 subtoken 分解规则；
   - Jacobian lens 需离线校准，增加部署成本。

2. ❗ **Sender 计算负担较重**  
   - 自适应 latent 推理最多可达 32,768 步，尽管通信后缀短，但总推理成本仍高。

3. ❗ **非正交 anchor 导致重建误差**  
   - Jacobian dictionary 中 anchor 方向非正交，影响 readout 重建精度。

---

### **未来工作方向**

1. 🔄 **自动化 calibration 流程**  
   - 设计免校准或在线校准机制，降低部署门槛。

2. ⚙️ **动态调整通信长度 M**  
   - 根据任务复杂度自适应选择 M，进一步优化效率。

3. 🧠 **探索更多 Jacobian-lens 应用场景**  
   - 如用于 interpretability、steering、或 reward modeling。

4. 🌐 **扩展至多跳、多轮 latent debate**  
   - 构建完全基于潜空间的 multi-agent 协作框架。

---

> **总结一句话**：  
> **J-CoMM 通过 GEAR + H+J 实现了高效、高性能、跨模型的潜通信范式，在保持 zero-shot 能力的同时，大幅超越文本通信，是迈向“模型原生协作”的重要一步。**

</details>

---

### 6. [Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?](https://arxiv.org/abs/2610.32046)

**Authors**: Yepeng Liu, Shengmai Chen, Xuandong Zhao, Chengzhi Liu, Dawn Song, Xin Eric Wang, Yuheng Bu  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.32046v1  

#### Abstract
Recent \emph{test-time training} (TTT) frameworks have enabled open-weight large language models (LLMs) to outperform inference-only frameworks powered by frontier models on scientific discovery tasks. However, TTT introduces additional computational cost for online weight updates during the discove...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Test-Time Training (TTT)** 框架（如 TTT-Discover 和 ThetaEvolve）通过在推理过程中进行在线 RL 训练来更新 LLM 权重，从而提升科学发现任务的表现。然而，这种参数化适应（parametric adaptation）带来了高昂的计算开销，尤其是在扩展到更大模型时面临效率瓶颈。

本文提出一个核心问题：  
> **是否可以在不更新模型权重的前提下，通过更高效的非参数化方式实现甚至超越 TTT 的性能？**

### 🚀 提出的新方法：Test-Time Memory Evolution (TTME)
作者提出了 **Test-Time Memory Evolution (TTME)**，一种**即插即用**（plug-and-play）的测试时自演化记忆系统，作为 TTT 的替代方案。

#### 核心思想：
- 不修改模型权重（fixed-weight model），而是将搜索经验持续提炼并存储于**外部记忆**中。
- 利用这些记忆指导后续候选程序生成，实现“经验内化”而不改变模型本身。

#### 两大记忆模块设计：
| 模块 | 功能 |
|------|------|
| **Guideline Memory** | 提取通用的实现规范（如避免常见错误、执行技巧），用于提高程序有效性（validity） |
| **Strategy Memory** | 抽取能提升分数的算法级改进策略（如梯度平均、优化调度等），支持跨尝试复用与组合 |

#### 工作流程：
1. 每隔 $ K $ 步收集评估反馈；
2. 提炼为 guideline 和 strategy；
3. 在后续 generation 中注入 memory 内容作为上下文提示；
4. 引导 LLM 进行更有效的探索。

### 🔍 相比现有方法的优势
| 维度 | TTT | TTME |
|------|-----|------|
| 是否更新权重 | 是（online RL） | 否（固定权重） |
| 计算成本 | 高（需反向传播） | 极低（仅前向推理） |
| 可扩展性 | 受限于训练成本 | 易扩展至大模型 |
| 性能表现 | 强 | 更强（本文结果） |
| 插拔性 | 深度耦合框架 | 即插即用，兼容性强 |

> ✅ **核心优势：以更低的计算代价实现了更高的科学发现性能。**

---

## 2. 核心实验方法和设置

### 📚 数据集与任务（5个数学构造任务）
所有任务均为可执行代码形式的科学优化问题，配备专用 verifier 进行自动评估：

| 任务 | 描述 | 优化目标 |
|------|------|----------|
| **Erdos** | Erdős 最小重叠问题，寻找最小重叠常数上界 | ↓ 最小化最大重叠值 |
| **AC1** | 第一自相关不等式，构造非负函数最小化 $\|f*f\|_\infty / \|f\|^2$ | ↓ 上界越小越好 |
| **AC2** | 第二自相关不等式，最大化下界 | ↑ 越大越好 |
| **AC3** | 允许正负值的第三自相关不等式 | ↓ 上界越小越好 |
| **Hadamard** | 29×29 符号矩阵的最大行列式问题 | ↑ 归一化行列式越大越好 |

> 所有任务均来自已有 TTT 研究（[1], [6]），确保公平比较。

### ⚙️ 实验设置

#### 模型
- **Qwen3-8B**
- **gpt-oss-120b**
- **Distill-Qwen3-8B**（用于 ThetaEvolve 框架）

> 所有模型在 TTME 设置下保持 **权重冻结**。

#### 框架
在两个主流 TTT 框架中替换原 TTT 模块：
- **TTT-Discover** [1]
- **ThetaEvolve** [6]

> 保持原有 parent selection、archive update、evaluation 等机制不变，仅替换 adaptation 方式。

#### 对比设置
| 设置 | 是否更新权重 | 是否使用 TTME |
|------|----------------|----------------|
| w/o TTT | ❌ | ❌ |
| w/ TTT | ✅（RL） | ❌ |
| w/ TTME | ❌ | ✅ |

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Best-found score** | 搜索过程中找到的最佳有效得分 |
| **Runtime (excl. eval)** | 排除程序运行时间后的纯模型推理耗时 |
| **FLOPs (EFLOPs)** | 总浮点运算量（含 memory 相关调用） |
| **Validity rate** | 成功编译并通过验证的程序比例 |
| **Combination rate** | 结合多个已有 idea 的候选被选为父代的比例 |

---

## 3. 主要实验结果和性能指标

### 📈 性能全面超越 TTT

#### 表格 1 & 2：在所有任务上 TTME 均优于 TTT
| 方法 | AC1↓ | AC2↑ | Erdos↓ | AC3↓ | Hadamard↑ |
|-------|------|------|--------|--------|-----------|
| TTT-Discover (w/ TTT) | 1.50525 | 0.9472 | 0.380932 | — | — |
| Ours (w/ TTME) | **1.504329** | **0.949248** | **0.380927** | — | — |
| ThetaEvolve (w/ TTT) | — | 0.9469 | — | 1.4930 | 0.5764 |
| Ours (w/ TTME) | — | **0.948148** | — | **1.464224** | **0.579938** |

> ✅ **TTME 在全部 5 项任务、3 种模型配置下均取得最优结果。**

#### 图 3：更快收敛且最终性能更高
- 在 TTT-Discover 框架中，TTME 在 **25 步内即超过 TTT 的最终成绩**。
- 收敛速度更快，说明 memory 引导更高效地利用历史经验。

### ⏱️ 显著降低计算成本（Table 3）

| Task | Model | Runtime (TTT) | Runtime (TTME) | ↓ Reduction | FLOPs (TTT) | FLOPs (TTME) | ↓ Reduction |
|------|-------|---------------|----------------|-------------|--------------|----------------|--------------|
| AC1 | Qwen3-8B | 11.76h | 5.36h | **54.5%** | 29.47 EFLOPs | 4.53 EFLOPs | **84.6%** |
| AC2 | Qwen3-8B | 12.73h | 6.66h | **47.7%** | 33.09 EFLOPs | 5.69 EFLOPs | **82.8%** |
| Erdos | Qwen3-8B | 11.67h | 4.85h | **58.4%** | 26.21 EFLOPs | 4.15 EFLOPs | **84.2%** |

> 💡 **即使 TTT 只更新 LoRA 参数，TTME 仍节省近 50% 时间和超 80% FLOPs。**

### 🔬 消融实验（Ablation Studies）

#### 表 4：TTME + TTT 并不能带来增益
| 设置 | AC2↑ | Erdos↓ |
|------|------|--------|
| Neither | 0.944771 | 0.380957 |
| TTT only | 0.9472 | 0.380932 |
| TTME only | **0.949248** | **0.380927** |
| TTT + TTME | 0.944141 | 0.380944 |

> ❌ 加入 TTT 反而降低性能，表明 **TTME 可独立替代 TTT，无需联合使用**。

#### 表 5：双 memory 模块缺一不可
| 配置 | AC1↓ | AC2↑ | Erdos↓ |
|------|------|------|--------|
| Full TTME | **1.502745** | **0.960784** | **0.380862** |
| w/o Guideline Memory | 1.505644 | 0.958506 | 0.380865 |
| w/o Strategy Memory | 1.505981 | 0.956493 | 0.380864 |
| w/o TTME | 1.505966 | 0.953682 | 0.380946 |

> ✅ 两个 memory 模块协同作用，共同推动性能提升。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **TTME 能显著超越 TTT 的性能**：在相同 discovery 框架下，TTME 不仅找到了更优解，而且收敛更快。
2. **更高的计算效率**：相比 TTT，TTME 减少 **47.7–58.4% runtime** 和 **82.8–84.6% FLOPs**，极具可扩展性。
3. **固定权重也能高效学习**：通过 external memory 演化，模型可在不更新参数的情况下“记住”有效策略。
4. **促进 idea 组合**：TTME 更频繁选择融合多种已有 idea 的候选继续搜索（Figure 5），增强创造性探索。
5. **即插即用性强**：TTME 可无缝集成进现有框架，无需重构训练流程。

### ⚠️ 局限性
- 当前仅验证于数学构造类任务，尚未推广至物理模拟、生物分子设计等复杂领域。
- Memory 的容量和去噪机制仍有优化空间（如动态压缩、注意力过滤）。
- 所有 memory 操作依赖同一 base model，可能引入推理偏差。

### 🔮 未来工作方向
- 将 TTME 应用于多模态科学发现（如蛋白质结构生成）。
- 设计 hierarchical memory 结构，支持长期知识沉淀。
- 探索 memory 与轻量化微调（如 LoRA）的协同机制。
- 开源更多 discovered constructions 和 memory logs，促进 reproducibility。

---

## ✅ 总结一句话
> **Test-Time Memory Evolution (TTME) 证明了：通过精心设计的外部记忆演化机制，固定权重的 LLM 可以在科学发现任务中以极低计算成本全面超越需要在线训练的 Test-Time Training 方法。**

GitHub: [https://github.com/yepengliu/TTME](https://github.com/yepengliu/TTME)

</details>

---

### 7. [A Broader Look into MLLM's Retrieval Heads](https://arxiv.org/abs/2610.32411)

**Authors**: Jia Wang, Yuqing Li, Jiangnan Li, Zheng Lin, Mo Yu  
**Category**: cs.CL  
**Published**: 2026-10-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.32411v1  

#### Abstract
Locating relevant evidence in long inputs is a core capability of Multimodal Large Language Models. Prior studies attribute this capability to retrieval heads, a sparse set of attention heads that transfer information from the context to the output. However, these analyses are confined to a limited ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Broader Look into MLLM's Retrieval Heads

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**多模态大语言模型（MLLMs）在处理长输入时如何定位关键证据**这一核心能力。尽管已有研究在文本和图文场景中发现了“检索头”（retrieval heads）的作用，但以下两个问题尚未充分探索：
- 当模型支持更多模态（如音频、视频）时，检索机制是否依然有效？是否跨模态共享？
- 能否将这些可解释的功能性注意力头用于实际性能提升？

### ✅ 提出的新方法与新思路
1. **跨模态配对分析框架（Paired AV-NIAH）**  
   构造了成对的音视频输入（AV-NIAH），其中相同答案以语音或字幕形式出现，背景和其他条件保持一致。这使得可以精确比较不同模态下 retrieval heads 的行为。

2. **统一的 retrieval head 检测标准**  
   在文本、图像-文本、音视频等多种任务上使用相同的 attention-based 指标来识别 retrieval heads，实现跨模态、跨模型的一致性分析。

3. **将 retrieval heads 作为优化目标（Optimization Target）**  
   首次提出通过训练 retrieval heads 的 attention 分数来进行**多模态文档排序（document ranking）**，引入了一个基于 ranking objective 的微调策略，直接优化其检索能力。

### ✅ 相比现有方法的优势
| 方面 | 本文优势 |
|------|---------|
| **分析广度** | 覆盖语言、视觉语言、全模态（omni-modal）共18个 checkpoint，是目前最广泛的 retrieval head 分析 |
| **机制理解深度** | 揭示了 retrieval heads 的稀疏性、跨模态共享性和主干保留性（preservation）三大特性 |
| **应用转化** | 不仅停留在可解释性层面，还将 retrieval heads 转化为可训练模块，显著提升下游任务表现 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 类型 | 描述 |
|-------|-----|------|
| **LongMemEval & Natural Questions** | 文本 | 用于检测纯文本中的 retrieval heads |
| **MM-NIAH** | 图像-文本 | 包含插入“needle”的图文交错上下文，测试视觉信息检索 |
| **AV-NIAH (本文构建)** | 音频-视频 | 自建配对数据：同一数字分别以语音播报或屏幕显示方式嵌入长视频中，其余部分完全相同 |
| **MMDocIR [7]** | 多模态文档检索 | 主要评估任务，包含页面级和布局级（layout）的多模态文档问答 |

### ⚙️ 实验设置
- **模型范围**：涵盖18个 checkpoint，包括：
  - Language: Qwen2.5-3B/7B, Qwen3-4B/8B
  - Vision-Language: Qwen2.5-VL系列
  - Omni-Modal: Qwen2.5-Omni, MiniCPM-o, SALMONN 系列等
- **retrieval head 定义**：
  - 使用 calibrated retrieval score：$ s_{lh} = \frac{1}{N}\sum_i(u^{(i)}_{lh} - u^{(i,\emptyset)}_{lh}) $
  - 取 top 5% 的 attention heads 作为 retrieval heads
- **干预方式**：
  - **Ablation**：置零指定 heads 的输出，验证因果作用
  - 控制组：随机选择同层数量的非 retrieval heads 进行对比

### 🎯 评估指标
| 任务 | 指标 |
|------|------|
| AV-NIAH | Exact Match (EM) |
| MMDocIR | Recall@K (R@1, R@3, R@5, R@10)，分 Macro 和 Micro 平均 |
| Head similarity | Spearman’s ρ, top-k overlap ratio |

### 🔁 基线方法对比
| 方法 | 来源 | 特点 |
|------|------|------|
| DPR, ColBERT, ColPali, DSE docmatix | 公开 retriever | 传统检索系统 |
| Random heads | 本文控制组 | 使用随机 attention heads 构造评分函数 |
| Untrained retrieval heads | 本文基线 | 不进行 ranking training 的原始 attention 得分 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（MMDocIR 上 Qwen3-VL-8B 结果）
| 方法 | Page R@1 (Micro) | Layout R@1 (Micro) |
|------|------------------|--------------------|
| DPR | 26.91 | 19.09 |
| ColPali | 54.58 | 33.06 |
| **Retrieval heads (未训练)** | **56.45** | **47.14** |
| **+ Ranking training (本文)** | **62.48 (+6.03)** | **48.60 (+1.46)** |

> ✅ **Page R@1 提升达 6.03 个百分点**，且优于所有公开 retriever

### 🔍 跨模型家族泛化效果（Table 4）
| 模型 | Page R@1 提升 | Layout R@1 提升 |
|------|---------------|------------------|
| Qwen3-VL-8B | +6.03 | +1.46 |
| Qwen2.5-VL-7B | +5.69 | +0.81 |
| Gemma3-12B | +2.17 | +2.16 |

> ✅ 所有三个模型家族均受益，说明方法具有较强通用性

### 🔻 消融实验结果

#### （1）Ablation on Retrieval Heads（图5、6）
- 移除 top-5% retrieval heads 后：
  - Qwen2.5-Omni-7B 在音频 needle 上 EM 从 98.3% → 0%
  - 视觉 needle 上也大幅下降至接近随机水平
- 即使只移除 **AV-shared heads**（同时在音频和视觉中激活的 ~10 个 head），也能造成显著性能下降

#### （2）Cross-modal Sharing Effect（表2）
| 任务 | AV-shared heads 损害程度（Δpp） | Image-retrieval heads 对照 |
|------|-------------------------------|-----------------------------|
| MathVista | +22.3 | +2.5 |
| CV-Bench (空间感知) | +16.3 | +5.7 |
| MMLongBench-Doc | +5.8 | +5.3 |
| MMStar | +12.2 | +2.8 |

> ✅ 尽管 AV-shared heads < 2% 总数，但对多种下游任务均有显著影响，远超对照组

#### （3）Head Selection vs Random（图7）
- 在任意 head 数量 $k$ 下，**retrieval heads 始终优于 random heads**
- 训练后差距进一步拉大，表明功能选择至关重要

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Sparsity（稀疏性）**  
   retrieval heads 在所有模态中都高度稀疏 —— 少数头部承担绝大多数证据定位功能。

2. **Cross-modal Sharing（跨模态共享）**  
   同一内容无论是“说出来”还是“显示出来”，激活的是**几乎相同的 retrieval heads**（重叠率达 69%-83%）。  
   ➤ 表明模型关注的是**语义内容本身**而非载体模态。

3. **Preservation from Backbone（主干保留性）**  
   omni-modal 模型在适配过程中**基本保留了其语言或视觉语言主干的 retrieval heads**（Spearman ρ ≥ 0.85），并继续依赖它们完成新模态任务。

4. **Causal Role（因果作用）**  
   - Ablation 实验证明 retrieval heads 是音频-视觉检索的必要组件；
   - 更重要的是，**跨模态共享子集（<2% heads）还支撑着文档理解、空间推理、视觉数学等复杂任务**。

5. **Functional Heads as Optimization Targets**  
   - retrieval heads 的 attention 可被训练以提升 ranking performance；
   - 该方法无需 layout 标注即可改善细粒度 layout retrieval，体现迁移能力。

---

### ⚠️ 局限性
1. **依赖预定义 evidence spans**  
   需要人工标注或构造明确的“needle”，难以直接应用于开放域真实文档。
   
2. **head selection 固定于 base model**  
   训练时不更新 head 集合，可能错过动态调整的机会。

3. **当前增益集中在 page-level retrieval**  
   layout-level 改进有限，尤其对于非结构化文档仍有挑战。

4. **out-of-domain 泛化有限**  
   如图8所示，在 MMMU、MathVista 等任务上训练前后变化小于1个百分点，表明改进主要局限于检索任务。

---

### 🔮 未来工作方向
1. **动态 retrieval head selection**  
   探索基于输入自适应地选择或生成 retrieval heads。

2. **扩展至其他模态组合**  
   如触觉、气味、三维场景等新兴模态下的 retrieval 机制研究。

3. **结合推理路径追踪**  
   将 retrieval heads 与 reasoning heads 联动分析，构建端到端的“从检索到推理”机制链。

4. **轻量化部署方案**  
   利用仅需修改少量 heads 的特点，设计高效 fine-tuning 或 inference acceleration 方法。

---

## ✅ 总结
本论文系统性地揭示了 MLLM 中 retrieval heads 的**稀疏性、跨模态共享性与主干保留性**，并通过 ablation 实验确立了其在多模态理解中的**因果核心地位**。更重要的是，作者首次将 retrieval heads 从“解释工具”转变为“优化目标”，通过简单的 ranking training 显著提升了多个模型家族在 MMDocIR 上的表现。

> 💡 **一句话总结**：  
> **Retrieval heads 是 MLLM 中一个紧凑、共享、关键的功能模块 —— 它们不仅是理解模型行为的窗口（interpretability lens），更是改进模型能力的有效入口（optimization target）**。

</details>

---

### 8. [KVStream: Adaptive Parallel Storage Management for KV Caching in Large-Scale LLM Serving](https://arxiv.org/abs/2610.31827)

**Authors**: Avinash Maurya, Jaiaid Mobin, Jie Ye, M. Mustafa Rafique, Franck Cappello, Bogdan Nicolae  
**Category**: cs.DC  
**Published**: 2026-10-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.31827v1  

#### Abstract
Long-context LLM serving for agentic, coding, and document workloads reuses prefixes whose key-value (KV) cache outgrows GPU and host memory, putting storage I/O on the critical path of the inference step for retrieving prefixes. A single node-local NVMe absorbs all retrievals, while the parallel fi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# KVStream: Adaptive Parallel Storage Management for KV Caching in Large-Scale LLM Serving —— 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在大规模 **LLM serving** 中，尤其是长上下文、多轮对话、代码生成和文档分析等场景下，**Key-Value (KV) cache** 的大小会随着上下文长度增长而急剧膨胀，远超 GPU 和主机内存容量。当这些缓存被逐出后，系统通常通过从存储设备（如 NVMe）重新加载来复用前缀，避免昂贵的 **recomputation**。

然而，现有系统（如 LMCache）存在以下瓶颈：
- **串行读取**：按请求顺序逐个读取 KV chunks，无法充分利用存储带宽。
- **本地 NVMe 成为瓶颈**：并行文件系统（PFS）或云存储几乎无负载，而节点本地 NVMe 承担全部 I/O。
- **读写干扰**：存储读取与 KV 刷新（flush）并发，导致性能下降。
- **静态分片策略失效**：固定分配读取任务到不同存储层无法适应动态变化的带宽。

因此，**存储 I/O 成为推理步骤的关键路径**，限制了整体吞吐和响应延迟。

---

### 提出了什么新方法或新思路

作者提出 **KVStream**，一个自适应的并行 KV 缓存引擎，其核心思想是将前缀检索建模为 **aggregate-bandwidth scheduling** 问题，并通过以下四个机制实现高效并行读取：

#### M1: 并行批处理检索（Parallel, Batched Retrieval）
- 将整个 batch 的所有请求的 KV 读取合并为一个统一的 I/O 池。
- 避免逐请求串行处理，提升 I/O 队列深度，充分发挥 NVMe 的并行能力。

#### M2: 推迟写入（Deferred Writes）
- 将新生成的 KV 写入操作推迟到下一个前向传播期间执行。
- 避免读写竞争，减少对关键路径的干扰。

#### M3: 多路径复制（Multi-path Replication）
- 每个 KV chunk 同时写入本地 NVMe 和远程 PFS（或其他第二存储层）。
- 虽然写入量翻倍，但为后续读取提供了灵活的路由选择。

#### M4: 完成驱动的“中间相遇”路由（Completion-Driven Meet-in-the-Middle Routing）
- 读取时，NVMe 从前端读取，PFS 从后端读取，两者“相向而行”。
- 任一存储层完成一批后立即领取更多未读 chunk，无需预测或调优。
- 自动适应各层实时带宽波动，逼近理论聚合带宽上限。

---

### 相比现有方法的优势

| 特性 | LMCache / 现有系统 | KVStream |
|------|---------------------|----------|
| 读取方式 | 逐请求串行 | 批处理并行 |
| 存储利用 | 仅用 NVMe | NVMe + PFS 并行 |
| 写入时机 | 即时写入 | 推迟至非关键路径 |
| 分片策略 | 静态分配 | 动态完成驱动 |
| 是否需要预测 | 是（带宽、访问模式） | 否 |
| 性能稳定性 | 易受带宽波动影响 | 强鲁棒性 |

> ✅ **优势总结**：KVStream 在不依赖任何先验知识或调参的情况下，实现了接近理论最优的聚合带宽利用率，显著降低 TTFT 和提升吞吐。

---

## 2. 核心实验方法和设置

### 使用的数据集

| 数据集 | 类型 | 描述 |
|--------|------|------|
| **LooGLE** | 文档问答 | 包含 2K–32K tokens 的长文档，约 3.2K 请求 |
| **NarrativeQA** | 阅读理解 | 平均 46K tokens/请求，测试极长上下文 |
| **CUAD** | 法律合同分析 | 每份合同作为共享前缀，平均 11K tokens/请求 |
| **Qwen-Coder** | 生产级代码生成 | 来自阿里云的真实 trace，平均 7.1K tokens/请求 |
| **Chutes** | 代理会话 | 长达 130K tokens 的生产 trace，平均 62K tokens |
| **SWE-bench Agentic** | 编码代理任务 | 4K 请求，模拟真实 agent 工作流 |

---

### 实验设置

#### 测试平台（Testbeds）
| 参数 | Testbed-1 | Testbed-2 |
|------|-----------|-----------|
| GPU | 4× H100-80GB | 4× A100-40GB |
| 主机内存 | 512 GB | 512 GB |
| NVMe | RAID-0 ×2 (3.2TB) | RAID-0 ×2 (3.2TB) |
| PFS | VAST FS (4PB) | Lustre (100PB) |
| TP Degree | 4 | 4 |

#### 模型范围
- **14B → 110B** 参数模型（Qwen, DeepSeek, Llama）
- KV 大小：160KB–320KB/token（BF16）

#### 评估指标

| 场景 | 主要指标 |
|------|----------|
| **离线（Offline）** | 吞吐量（tokens/s）、总处理时间 |
| **在线（Online）** | **TTFT**（Time-to-First-Token，均值、p50、p90）、SLO 达成率 |
| **通用** | 检索时间、计算时间、写入时间、每层带宽利用率、与理论 bound 的差距 |

---

### 基线方法对比

| 方法 | 描述 |
|------|------|
| **CPU-only** | 仅使用 GPU + 主机内存缓存，miss 则重计算 |
| **LMCache** | 当前 SOTA，支持 NVMe offload，但串行读取 |
| **Mooncake Store** | 支持分布式 KV 存储，但仅单层 offload |
| **SGLang HiCache** | 另一种多级缓存架构，用于跨引擎验证 |
| **KVStream** | 本文方法（作为 LMCache 后端插入） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 离线吞吐提升
在 LooGLE 数据集上，使用 110B 模型：
- **KVStream 吞吐达到 LMCache 的 1.49×**
- 相比 CPU-only 提升高达 **3.22×**

| 模型 | KVStream / LMCache |
|------|--------------------|
| 14B | 1.17× |
| 32B | 1.37× |
| 70B | 1.48× |
| 110B | **1.49×** |

> 📈 吞吐增益随模型增大而增加，因更大模型 KV 更大、重计算代价更高。

#### ✅ 在线服务 TTFT 显著降低
在突发流量（bursty arrivals）下：
- **平均 TTFT 最高降低 6.1×**
- 在 Qwen-Coder 上，高负载下 TTFT 速度提升达 **3.12×**

| 场景 | TTFT 提升倍数 |
|------|----------------|
| Qwen-Coder（高负载） | 3.12× |
| Chutes（高负载） | 2.01× |
| SWE-bench（CV=4） | **6.07×**（p90） |

#### ✅ 接近理论聚合带宽极限
- KVStream 的实际读取性能 **在 5% 以内逼近 aggregate-bandwidth bound**
- 静态分片策略（如 80:20）在带宽波动时性能下降明显（最高慢 67.7%）

#### ✅ 多节点扩展性良好
- 扩展至 **128 GPU（32 节点）**，KVStream 仍保持 **1.23–1.36× 吞吐优势**
- 各节点独立适应 PFS 拥塞，无需跨节点协调

---

### 消融实验结果（Ablation Study）

逐步添加机制的效果（70B, LooGLE）：

| 方法 | 吞吐（tok/s） | vs. CPU-only |
|------|---------------|-------------|
| CPU-only | 18,513 | 1.00× |
| LMCache (NVMe) | 31,977 | 1.73× |
| + M1（批处理） | 32,115 | 1.73× |
| + M2（推迟写） | 32,177 | 1.74× |
| + M3（双层复制 + 固定分片） | 39,631 | 2.14× |
| + M4（完成驱动路由） | **47,210** | **2.55×** |

> 🔍 结论：**M3 和 M4 是性能飞跃的关键**，仅优化调度（M1/M2）效果有限。

---

## 4. 关键结论和发现

### 主要发现

1. **存储 I/O 是长上下文 LLM serving 的关键瓶颈**，尤其当工作集超出内存时。
2. **并行文件系统（PFS）长期空闲**，而本地 NVMe 成为单点瓶颈，资源严重浪费。
3. **静态分片策略无法应对带宽波动**，必须采用动态、反馈驱动的路由机制。
4. **完成驱动的“中间相遇”路由** 可自动逼近理论聚合带宽极限，无需监控或预测。
5. **性能收益随负载增长而放大**：高并发下，队列积压使得更快的检索带来指数级的 TTFT 改善。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **写入放大** | 复制导致写入量翻倍，在写密集场景可能成为瓶颈 |
| **低复用率无效** | 当 prefix reuse < 4%，KVStream 反而变慢（0.81×） |
| **内存充足时收益小** | 若工作集可放入主机内存，则无加速空间（0.98×） |
| **依赖多存储层** | 至少需要两个独立带宽路径才能生效 |
| **无跨节点协调** | 不共享跨节点缓存，可能重复存储相同 chunk |

> ⚠️ **适用边界明确**：KVStream 在 **高复用、长上下文、多会话交错、内存受限** 的场景下效果最佳。

---

### 未来工作方向

1. **减少复制开销**  
   - 动态决定是否复制（如仅当某 chunk 被高频访问时才复制到 PFS）
   - 引入纠删码或异步修复机制

2. **支持对象存储作为第二层**  
   - 如 AWS S3、Azure Blob，进一步降低成本

3. **与压缩技术结合**  
   - KVStream 可与 GQA、MLA、CacheGen 等压缩方法正交组合

4. **与 disaggregated serving 集成**  
   - 在 prefill-decode 分离架构中，KVStream 可同时从两侧存储读取

5. **跨节点协同缓存管理**  
   - 共享全局热块视图，避免重复复制

---

## 总结

> **KVStream 通过将 KV 检索转化为聚合带宽调度问题，提出了一种无需预测、自适应、高性能的多层存储管理方案。它打破了传统串行读取和静态分片的桎梏，在真实生产 trace 和大规模部署中实现了高达 1.49× 吞吐提升和 6.1× TTFT 降低，是当前长上下文 LLM serving 存储优化的重要突破。**

</details>

---

### 9. [Democratizing LLM Training with the Edge: Challenges, Systems, and the Road Ahead](https://arxiv.org/abs/2610.32215)

**Authors**: Ziyue Luo, Jiaxuan Cai, Cedric LeDenmat, Srijith Nair, Fatemeh Nourzad, Rohith Krishnan, Qinhang Wu, Jifan Zhang, Sungjae Lee, Zhe Li, Peiwen Qiu, Siddharth Shah, Rishabh Sharma, Sundararajan Srinivasan, Yinglun Xia, Xue Zheng, Zidong Liu, Bicheng Ying, Kaushik Chowdhury, Gauri Joshi, Yingbin Liang, Robert Nowak, Srinivasan Parthasarathy, Saurav Prakash, Balaraman Ravindran, Sanjay Shakkottai, Ness B. Shroff, Haibo Yang, Aylin Yener, Jia Liu  
**Category**: cs.DC  
**Published**: 2026-10-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.32215v1  

#### Abstract
The growing computational power of modern GPU data centers has let a few industrial giants train and deploy large language models (LLMs) on massive amounts of data. Yet the sheer scale of these models and their training data makes centralized development increasingly hard to sustain, and at the same...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Democratizing LLM Training with the Edge: Challenges, Systems, and the Road Ahead

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大型语言模型（LLMs）的训练高度集中于少数拥有大规模GPU集群的科技巨头手中，导致以下问题：
- **资源垄断**：大多数学术机构和中小企业无法负担高昂的计算成本。
- **数据隐私与治理**：中心化训练需要汇集大量私有数据，带来隐私泄露、带宽和存储压力。
- **系统异构性挑战**：跨机构的地理分布式训练面临硬件异构（GPU型号、内存）、网络延迟波动、防火墙/NAT限制等现实障碍。

本文旨在解决如何在**地理分布、硬件异构、网络不可靠**的真实边缘环境中实现**去中心化、可扩展且鲁棒的LLM训练**，推动“民主化LLM训练”。

---

### 提出的新方法与新思路
作者提出了一种**统一的两层系统架构（Unified Two-Level Architecture）**，支持**预训练（pretraining）** 和 **微调（fine-tuning）** 两种范式：

#### 核心架构设计
- **云托管参数服务器（Cloud-hosted Parameter Server, PS）**：
  - 作为全局协调者，维护权威的全局模型快照和优化器状态。
  - 所有客户端仅通过**出站（outbound）gRPC连接**与PS通信，无需配置入站端口或VPN，极大简化了跨机构部署。
- **异构客户端（Heterogeneous Clients）**：
  - 各参与机构运行本地训练进程（如FSDP），执行本地更新。
  - 定期将参数增量（delta）上传至PS，并拉取最新的全局模型。

#### 统一构建模块（Shared Building Blocks）
该架构为预训练和微调复用同一套基础设施：
- **通信协议**：采用 **gRPC + Protocol Buffers**，支持持久化的双向流式传输，避免频繁重连。
- **网络压缩**：采用 **量化（Quantization） + 霍夫曼编码（Huffman Encoding）** 双阶段压缩，显著减少WAN传输开销。
- **容错与重连机制**：客户端崩溃后可通过一次性令牌快速从PS恢复最新模型和优化器状态，实现秒级重连。

#### 创新点总结
| 创新维度 | 具体内容 |
|--------|--------|
| **系统架构** | 提出基于云PS的统一架构，解耦同步逻辑，支持异构硬件与网络环境 |
| **通信兼容性** | 仅需出站连接，天然绕过企业防火墙/NAT，实现“一键加入”体验 |
| **统一性** | 同一套系统同时支持LLM预训练与多种微调范式（LoRA, RAVAN, Zeroth-order, FSL） |
| **鲁棒性设计** | 内建压缩、异步聚合、故障恢复机制，适应高延迟、不稳定WAN |

---

### 相比现有方法的优势
| 对比维度 | 传统方法（如FedAvg, DiLoCo） | 本文方法 |
|--------|-----------------------------|---------|
| **网络兼容性** | 多依赖P2P或固定IP，难以穿透防火墙 | 仅需出站gRPC，部署门槛极低 |
| **硬件异构性** | 同步训练受最慢设备拖累 | 异步聚合+自适应批大小，充分利用各节点算力 |
| **容错能力** | 客户端掉线可能导致训练中断 | 支持快速重连与状态恢复，训练持续稳定 |
| **适用场景** | 多针对同质化数据中心 | 明确面向地理分布、异构边缘节点的真实场景 |

---

## 2. 核心实验方法和设置

### 数据集
- **预训练任务**：
  - **GPT-2 Medium** 在 **OpenWebText** 上进行预训练。
  - **Llama3-1B** 在未明确说明的大规模文本语料上进行预训练。
- **微调任务**：
  - 使用 **SQuAD v1.1** 数据集对 **GPT-2 Medium** 进行问答任务的微调，评估LoRA等PEFT方法。

---

### 实验设置
- **部署环境**：
  - 跨多个大学站点（OSU, UW-Madison, UT Austin, IIT Madras）的真实GPU集群。
  - 客户端使用不同GPU：H100, A100, RTX 3090, RTX 4090，体现硬件异构性。
  - PS部署在AWS云上（t2.xlarge / r5.2xlarge实例）。
- **通信模拟**：
  - 利用真实跨洲WAN链路，测量并反映实际延迟与带宽波动（如图6所示）。
- **同步策略**：
  - 参数同步间隔为**10分钟（GPT-2）** 或 **20分钟（Llama3-1B）**，以降低通信频率。
- **压缩策略**：
  - 梯度从FP32量化为INT8或BF16，结合霍夫曼编码进一步压缩。

---

### 评估指标
- **收敛性**：训练损失（Training Loss）随**真实时钟时间（wall-clock time）** 的下降曲线。
- **稳定性**：是否出现发散（divergence）或震荡。
- **效率**：单位时间内处理的token数、通信开销（MB/step）。
- **容错表现**：客户端重启后的恢复时间与训练连续性。

---

### 基线方法对比
- **FedAvg**：标准同步联邦学习框架，要求所有客户端完成相同步数。
- **DiLoCo**：低通信训练方法，使用Nesterov动量，但假设高带宽环境。
- **FedNova**：归一化局部更新以应对异构速度，允许一定程度异步。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
#### （1）GPT-2 Medium 预训练结果（图5）
- **本文方法**：在真实WAN条件下**稳定收敛**，损失持续下降。
- **FedAvg**：收敛较慢，受最慢客户端拖累。
- **DiLoCo**：在推荐超参下**迅速发散**，表明其对高延迟环境极度敏感。
- **FedNova**：虽允许异步，但仍不如本文方法在真实异构环境下高效。

> ✅ **结论**：本文设计在真实地理分布环境中具有更强的**鲁棒性与收敛速度优势**。

#### （2）Llama3-1B 预训练结果（图7）
- 成功在更大规模模型（1B参数）上验证系统可行性。
- 尽管通信负载更重，系统仍能**稳定下降训练损失**，无发散现象。
- 表明该架构具备良好的**可扩展性（scalability）**。

#### （3）跨站点通信延迟实测（图6）
- 测量显示：
  - 不同站点间延迟差异大（数十至数百毫秒）。
  - 延迟存在明显**时间波动性**，反映WAN不稳定性。
- 此结果**验证了设计动机**：必须考虑延迟变异性和异步容错。

#### （4）容错与恢复能力
- 客户端重启后，通过PS恢复最新模型平均耗时 **~10秒**。
- 若从磁盘检查点恢复，约需 **1分钟**。
- PS自身故障可通过主客户端重新加载检查点恢复。

---

### 消融实验（隐含分析）
虽然未设显式消融表，但文中多处体现关键组件必要性：
- **gRPC替代PyTorch RPC**：解决了P2P连接在防火墙下的不可行问题。
- **量化+霍夫曼压缩**：显著降低通信体积，缓解WAN瓶颈。
- **异步聚合+超时机制**：避免慢节点阻塞整体进度，提升资源利用率。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **地理分布+异构硬件的LLM训练是可行的**：通过合理的系统设计，可在真实大学集群间稳定训练GPT-2和Llama3-1B。
2. ✅ **统一架构可同时支持预训练与多种微调范式**：基于PS的设计为不同任务提供了共享基础。
3. ✅ **出站gRPC连接是跨机构协作的关键**：规避了复杂的网络策略问题，实现“民主化接入”。
4. ✅ **现有数据中心优化算法不能直接迁移至WAN环境**：如DiLoCo在高延迟下失败，凸显专用系统设计必要性。

---

### 方法的局限性
1. **单点PS风险**：尽管有恢复机制，PS仍是潜在单点故障（SPOF）。虽支持multi-PS模式，但复杂度增加。
2. **未支持前沿超大模型（如70B+）**：当前实验限于1B级别，对MoE、多模态模型支持尚需探索。
3. **激励机制缺失**：未解决参与者长期贡献的动力问题，易导致免费搭车（free-riding）。
4. **隐私保障不足**：尚未集成差分隐私（DP）或安全聚合（Secure Aggregation），存在信息泄露风险。
5. **弹性成员管理待完善**：对动态加入/退出的自动化负载再分配支持有限。

---

### 未来工作方向（Road Ahead）
作者指出多个开放问题：
- **Scaling to Frontier Models**：支持MoE、多模态模型（如LLaVA）的分布式训练。
- **Elastic Fault Tolerance**：实现spot instance级别的动态伸缩与快速恢复。
- **Adaptive Compression**：根据实时网络状况自动调整量化精度与同步频率。
- **Federated Alignment**：支持联邦式RLHF与偏好优化（如DPO）。
- **Privacy & Trust**：集成DP、拜占庭鲁棒聚合（Byzantine-robust aggregation）。
- **Incentive Design**：设计公平、抗博弈的贡献评估与奖励机制（如Shapley值改进）。
- **System-Algorithm Co-design**：联合优化通信策略与优化器（如学习率调度、动量）。

---

### 总结
本文提出了一个面向**真实世界边缘环境**的统一LLM训练系统，通过**云PS + 出站gRPC + 压缩 + 容错**的设计，在**跨机构、异构硬件、不稳定WAN**条件下实现了稳定的LLM预训练与微调。实验验证了其在GPT-2和Llama3-1B上的有效性，并开源代码（[GitHub链接](https://github.com/ai-edge-institute/geo-distributed-llm-training)），为“民主化LLM训练”迈出了坚实一步。然而，通往真正开放、可持续的联邦LLM生态，仍需在**可扩展性、隐私、激励、算法协同**等方面持续突破。

</details>

---

### 10. [KnitTrans: Coordinated Trajectory Transfer for Effective and Efficient LLM Reinforcement Learning](https://arxiv.org/abs/2610.32389)

**Authors**: Bangjun Xiao, Chiheng Lou, Xiaodong Ji, Xiangwei Deng, Yikai Zhao, Gang Huang, Hailin Zhang  
**Category**: cs.DC  
**Published**: 2026-10-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.32389v1  

#### Abstract
Trajectory transfer from rollout to training affects both learning quality and system efficiency in large language model reinforcement learning (RL). Enforcing batch-wide training requirements can delay the admission of completed trajectories, resulting in higher architectural staleness: the additio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：KnitTrans: Coordinated Trajectory Transfer for Effective and Efficient LLM Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大语言模型（LLM）的强化学习（RL）训练中，**trajectory transfer**（从 rollout 到训练的数据传递路径）对学习质量和系统效率有显著影响。现有系统面临两个核心挑战：

- **Train-batch admission barrier**：为了满足 batch-wide 的训练要求（如混合数据源比例、部分 rollout 完成配额），必须等待整个 train batch 都准备好才能开始训练，导致训练延迟和更高的 **architectural staleness**（参数版本滞后）。
- **低效的轨迹数据访问**：现有存储布局（如 sample-level）与分布式训练的拓扑（TP/CP/Distinct encoder）不匹配，造成冗余数据读取、传输和等待。

### 提出的新方法
KnitTrans 是一个协调 rollout 到训练路径的系统，提出三项核心设计：

#### （1）Constraint-Preserving Mini-Batch Admission
- 引入 **progressive readiness targets** 和 **feasibility guards**，允许在满足最终 batch 要求的前提下，尽早组成并提交 mini-batch。
- 通过松弛早期目标（relaxed early targets）实现更早 admission，同时用可行性检查防止最终 batch 不可行。

#### （2）On-Demand Trajectory Data Access
- **Atomic-unit storage**：将轨迹数据按消费者需求拆分为独立可寻址的原子单元（如单个 trajectory 或图像帧）。
- **Topology-aware reorganization**：根据每个消费者（backbone、encoder）的并行策略（DP/CP/TP）动态重组所需数据。
- **Consumption-driven delivery**：按 micro-batch 消费顺序增量准备和交付数据，与计算重叠。

#### （3）Multimodal Encoding Deduplication
- 对共享相同多模态输入（如图像）的多个 trajectory，在 mini-batch 内只编码一次。
- 缓存编码结果，并在所有 consuming trajectory 中复用，同时累积梯度以保证反向传播正确性。

### 相比现有方法的优势
- 显著降低 architectural staleness，缩小 disaggregated 与 colocated 架构的学习质量差距。
- 减少数据传输延迟达两个数量级，提升整体训练吞吐。
- 支持灵活的 admission 策略而不破坏算法约束，提升资源利用率。

---

## 2. 核心实验方法和设置

### 数据集
- **CodeAgent 任务**：混合软件工程数据集，包括：
  - SWE-smith
  - Multi-SWE-bench
  - Terminal-Lego
  - Scale-SWE
- **Video Grounding 多模态任务**：两个内部数据集：
  - Grounding A
  - Grounding B（视频数据占比更高）

### 模型
- Qwen3-VL-235B（多模态）
- MiMo-V2.5-309B（多模态）
- DeepSeek-v3.2-671B（纯文本）

所有模型均使用 FP8 量化进行训练和推理。

### 实验设置
- 在 **256-GPU 集群** 上运行（32节点 × 8 Hopper GPU）
- 使用 **GRPO**（n=16 rollouts per prompt）作为 RL 算法
- 启用 **partial rollout** 机制缓解长尾问题
- 设置 **staleness bound = 4**

### 评估指标
- **Architectural staleness**：disaggregated 与 colocated 架构之间的平均参数版本差。
- **Exposed transfer latency**：训练关键路径上的数据传输延迟。
- **Training step time**：完整训练步耗时。
- **RL convergence**：使用 `avg@n`（prompt 下 n 个 rollout 的平均奖励）衡量学习质量。
- **Constraint preservation**：是否满足数据源比例等 batch-wide 要求。

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **VeRL** | Colocated 架构基准 |
| **slime** | Disaggregated 架构基准 |
| **TransferQueue (TQ)** | Sample-level 存储，无 topology-aware 优化 |
| **Miles** | 基于 Mooncake KV + RDMA 的 sample-level 存储 |
| **MegaData\*** | topology-aware 预处理系统（adapted to RL） |
| **StreamRL / AsyncFlow** | 流式 rollout 基准（无 batch-wide admission 控制） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）Architectural Staleness 降低
在 CodeAgent 任务上（disaggregated 架构）：

| 模型 | Train-batch admission | KnitTrans (4 mini-batches) | KnitTrans (8 mini-batches) | 降幅 |
|------|------------------------|-------------------------------|-------------------------------|------|
| Qwen3-VL-235B | 0.94 | 0.34 | 0.11 | ↓93% |
| MiMo-V2.5-309B | 0.96 | 0.25 | 0.19 | ↓80% |
| DeepSeek-v3.2-671B | 1.01 | 0.31 | 0.07 | ↓93% |

> ✅ **最高降低 93% architectural staleness**

#### （2）Exposed Transfer Latency 缩短
在 Omni RL 多模态任务上：

- KnitTrans 将暴露的传输延迟 **降低 217.6–264.3×**（即两个数量级）
- 相比 TransferQueue，**transfer + training 总速度提升 1.5–5.8×**

#### （3）RL 收敛质量提升
在 MiMo-V2.5 上的 CodeAgent 任务中：
- KnitTrans 的 `avg@n` 曲线几乎紧贴 **colocated（VeRL）** 运行
- 平均差距 ≤ 0.007
- 相比 train-batch admission（slime），平均高出 **0.082 avg@n**

> ✅ 成功缩小 disaggregated 与 colocated 的学习质量鸿沟

#### （4）Constraint Preservation
- **KnitTrans** 和 **Linear** 方法均能 **完全满足最终数据源配额**
- 但在 architectural staleness 上：
  - Linear: 0.470
  - KnitTrans: 0.286
  > ✅ **降低 39.2% staleness，同时保持约束满足**

---

### 消融实验结果（Ablation Study）

#### （1）Topology-Aware Distributed Storage (TADS)
- 相比 TQ，exposed transfer 从 495–583s 降至 **21.6–26.1s**
- 总训练时间减少 **28.8–72.0%**

#### （2）Micro-Batch Fetch Overlap
- 在 TADS 基础上进一步将 transfer 从 ~25s 降至 **2.5–2.6s**
- 额外带来 **8.4–10.4×** 的延迟压缩

#### （3）Multimodal Encoding Deduplication
- Encoder 正反向时间从 **49.9–81.5s** 降至 **4.8–6.7s**
- 总训练时间减少 **4.3–33.2%**
- 在 Grounding B 和 MiMo-V2.5 上增益更大（encoder 占比高）

---

## 4. 关键结论和发现

### 主要发现
1. **Train-batch admission barrier 是 architectural staleness 的主因之一**，而 KnitTrans 通过 mini-batch 级 admission 显著缓解该问题。
2. **数据访问必须与训练拓扑对齐**：sample-level 存储无法适应 TP/CP/encoder 分离的复杂场景，必须采用 atomic-unit + topology-aware 重组。
3. **多模态输入的重复编码是重大开销**，mini-batch 级去重可在不牺牲梯度准确性的前提下大幅提升效率。
4. **KnitTrans 实现了“解耦逻辑边界与物理执行粒度”**：
   - admission 不必等到整个 train batch 完成
   - 数据准备不必一次性完成
   - 编码可在多个 micro-batch 间共享

### 方法的局限性
- 当前设计依赖较强的控制平面调度能力，在超大规模异构集群中可能引入调度瓶颈。
- 对于非 GRPO 类算法（如仅单 rollout），multimodal encoding deduplication 效益下降。
- 实现基于 TransferQueue 扩展，对其他 RL 框架的适配仍需工程投入。

### 未来工作方向
- 将 constraint-preserving admission 推广到更多类型的 batch-wide 约束（如动态配比、优先级调度）。
- 结合更智能的 rollout 进度预测，进一步优化 readiness targets。
- 探索在 encoder 和 backbone 之间共享更多中间状态以进一步减少通信。
- 支持异构硬件（如不同 GPU 类型混合）下的自适应数据分发策略。

---

> **总结一句话**：  
> KnitTrans 通过 **协调化的 trajectory transfer 设计**，在不牺牲算法正确性的前提下，实现了 **更低的 architectural staleness** 和 **更高的训练效率**，为大规模 LLM RL 系统提供了新的系统优化范式。

</details>

---

### 11. [RSI-Opt: Recursive Self-Improvement Optimizer Agents For Model Training](https://arxiv.org/abs/2610.32026)

**Authors**: Binghang Lu, Yige Sun, Guang Lin  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.32026v1  

#### Abstract
Conventional neural-network training typically follows a predefined optimization recipe, motivating a LLM agent that monitors training dynamics and adjusts optimizer settings as training progresses, but a fixed agent policy does not accumulate experience across runs. We introduce RSI-Opt, a plug-and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RSI-Opt: Recursive Self-Improvement Optimizer Agents For Model Training

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统神经网络训练依赖于**预定义的优化器策略**（如 Adam、SGD 等），其超参数和调度在训练前固定，难以适应不同任务的动态变化。虽然已有工作（如 OPAL）引入 LLM 编码代理来**在单次训练过程中动态调整优化器**，但这些方法存在一个关键缺陷：**每次训练从头开始，无法跨训练运行积累经验**。

因此，该论文旨在解决：
- 如何让 optimizer agent 在多次训练中**持续学习并改进自身策略**
- 如何实现**跨 run 的知识积累与复用**

### 提出了什么新方法或新思路
提出 **RSI-OPT**（Recursive Self-Improvement OPTimizer Agents），一种**可插拔的递归自改进优化器代理框架**，其核心思想是将 optimizer agent 的“策略”显式化为可编辑对象，并通过元代理（meta-agent）进行跨 run 进化。

#### 核心组件包括：
- **Policy 表示**：包含四个部分
  - **Prompt**：指导 agent 的指令模板
  - **Playbook**：自然语言记录的经验、流程和教训
  - **Code Library**：可重用的优化模块（如 Muon 更新规则、学习率调度等）
  - **Decision Cadence**：决策频率与诊断检查节奏
- **Meta-Agent**：读取历史训练证据（loss 曲线、agent 决策日志、最终性能），对父策略进行改写生成子策略
- **Archive 机制**：存储所有有效评估过的策略及其得分，支持继承与探索（借鉴 Darwin Godel Machine [35] 的基于档案的搜索）

这种方法实现了：
- **策略的显式演化**：知识以文本和代码形式保存，可审查、可迁移
- **递归自我改进**：meta-agent 利用过往 run 的证据不断优化 agent 的行为策略
- **跨模型迁移能力**：同一套 evolved policy 可用于不同的 in-run LLM（如 GPT-5.5, GPT-6.1）

### 相比现有方法的优势
| 对比维度 | 传统方法（如 Adam） | OPAL 类方法 | RSI-OPT |
|--------|------------------|------------|---------|
| 是否动态调整优化器 | ❌ 固定 | ✅ 单 run 内调整 | ✅ 跨 run 自我进化 |
| 是否积累经验 | ❌ | ❌ 每次重启 | ✅ 显式策略存档 |
| 是否支持迁移 | ❌ | ❌ | ✅ 支持跨任务、跨 LLM 迁移 |
| 可解释性 | 高 | 中 | 高（策略可读） |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集 / 任务
共使用 **9 个 single-GPU 训练任务**，分为三类：

| 类别 | 任务 | 描述 |
|------|------|------|
| **Training Suite**（用于策略进化） | ResNet-18 / CIFAR-10 | 图像分类，使用 SGD-Nesterov |
| | GPT-34M / FineWeb | 语言建模，使用 AdamW |
| | Flow UNet / MNIST | 流匹配图像生成，使用 AdamW + EMA |
| **Held-out ML**（用于迁移评估） | ViT / CIFAR-100 | 更大视觉 Transformer |
| | LSTM / WikiText-2 | 序列建模 |
| | ResNet-18 / noisy CIFAR-10 | 含噪声标签的任务 |
| **Held-out PDE**（科学机器学习） | PINN / Burgers 1D | 物理信息神经网络求解偏微分方程 |
| | PINN / Heat 2D | 热传导方程 |
| | Fractional PDE | 分数阶扩散方程 |

> 所有任务的模型架构、损失函数、batch size 和更新次数均固定，仅允许 agent 修改 optimizer 和 diagnostic 程序。

### 实验设置和评估指标

#### 主要设置
- **In-run agent**：负责监控训练过程并在 20 个预定时间点修改 optimizer 和 analyzer 程序
- **Meta-agent**：每代从 archive 中采样 parent policy，生成 child 并评估
- **LLM 模型**：
  - 进化阶段：Claude Opus 5.5
  - 最终测试：Opus 5.5, GPT-5.5, GPT-6.1（policy 不变，直接迁移）
- **决策机制**：agent 可调用 bash、read/write/edit 文件工具，无网络/GPU 访问权限
- **成本控制**：每 run 有 token 和时间上限，确保公平比较

#### 评估指标
1. **Z-score（标准化提升）**：
   $$
   z_t = \frac{h_t(m_t) - \mu_t}{\sigma_t}, \quad h_t(m) = 
   \begin{cases}
   m & \text{ML 任务} \\
   \log_{10} m & \text{PDE 误差}
   \end{cases}
   $$
   其中 $\mu_t, \sigma_t$ 来自静态优化器的 5 次 seed 结果。

2. **Raw-space Summary Score**（最终报告用）：
   使用原始均值和标准差重新计算的聚合 z-score，便于表格复现。

3. **Policy Selection Protocol**：
   - 每代评估多个 child 在 training suite 上的表现
   - 选取 top-3 在额外两个 seed 上 re-evaluate，选择平均表现最好的作为最终 policy
   - 最终性能在 **fresh seeds (200–202)** 上报告，完全独立于进化过程

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Static Recipe** | 原始优化器配置（如 Adam, SGD） |
| **Tuned LR (Oracle)** | 学习率网格搜索最优结果（非实用，仅作上界参考） |
| **Single-agent (Fixed Prompt)** | 无历史策略积累的标准 agent，每 run 从零开始 |
| **RSI-OPT (Full)** | 完整的递归自改进系统 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

#### 综合性能（Raw-space Summary Z-scores）
| Model | Method | Train | Held-out ML | Held-out PDE | All Held-out |
|-------|--------|-------|-------------|---------------|--------------|
| Opus 5.5 | Single-agent | +10.48 | +8.71 | +1.72 | +5.22 |
| | **RSI-OPT** | **+14.33** | +8.23 | +1.51 | +4.87 |
| GPT-5.5 | Single-agent | +4.33 | +1.18 | -1.90 | -0.36 |
| | **RSI-OPT** | **+14.10** | **+6.54** | **-0.32** | **+3.11** |
| GPT-6.1 | Single-agent | +9.90 | -0.01 | +1.61 | +0.80 |
| | **RSI-OPT** | **+14.21** | **+6.87** | **+1.74** | **+4.30** |

> ✅ RSI-OPT 在 training suite 上显著优于所有 baseline  
> ✅ GPT-5.5 和 GPT-6.1 在所有任务组上均取得正向增益  
> ⚠️ Opus 5.5 在 held-out 总体得分略降（因 ViT 和 Fractional PDE 表现下降）

#### PDE 任务具体表现（Mean Relative L2 Error）
| Task | Static | Single-agent (Opus) | RSI-OPT (Opus) | Improvement |
|------|--------|--------------------|----------------|------------|
| PINN / Burgers 1D | 4.09e-2 | 3.95e-3 | **1.45e-3** | ~2.7× ↓ |
| PINN / Heat 2D | 3.78e-1 | 6.3e-5 | **2.9e-5** | ~2.2× ↓ |
| | | | | |
| PINN / Burgers 1D (GPT-6.1) | 4.09e-2 | 4.82e-3 | **5.46e-4** | ~8.8× ↓ |
| PINN / Heat 2D (GPT-6.1) | 3.27e-5 | 3.27e-5 | **1.60e-5** | ~2.0× ↓ |

> 🔥 在 GPT-6.1 上，Burgers 方程误差降低近 **9 倍**

### 与基线方法的对比结果
- RSI-OPT 将 training suite 的 summary score 从 +10.48 提升至 +14.33（Opus）
- 在 GPT-5.5 上，held-out 任务平均 score 从 -0.36 提升至 **+3.11**
- 相比 oracle tuning（仅调 learning rate），RSI-OPT 的增益远超其能力范围
- 单 agent baseline 已能大幅超越 static optimizer（如 Heat 方程误差降低 6000 倍）

### 消融实验结果（Ablation Studies）

#### （1）Reusable Code vs Online Adaptation（Table 5）
| Condition | Train Score | Held-out ML |
|----------|-----------|------------|
| Full RSI-OPT (20 decisions) | +14.33 | +8.23 |
| First change frozen | +14.56 | +6.82 |
| One decision only | +13.82 | +5.05 |
| No agent (frozen library) | +9.71 | -21.80 |

> 📌 发现：
> - **初始适配贡献最大**：冻结首次更改即可获得大部分训练集收益
> - **后续决策提升泛化**：继续在线调整显著改善 held-out 任务表现
> - **纯代码模板失败**：未加判断地应用 evolved updater 会导致灾难性退化（如 ViT 降到 35% 准确率）

#### （2）Policy Components Ablation（Table 6）
| Ablation | Train Score | Held-out ML |
|--------|------------|------------|
| Full RSI-OPT | +14.33 | +8.23 |
| -Library | +14.45 | +8.00 |
| -Playbook | +13.60 | +8.06 |
| -Evolved Prompt | — | — |

> 📌 发现：
> - **Library 影响最小**：可能因其内容已被充分集成
> - **Playbook 至关重要**：丢失后性能明显下降，说明程序性知识的关键作用

#### （3）Agent 行为分析（Table 7）
| Component | Single-agent | RSI-OPT |
|---------|-------------|---------|
| 使用 Muon 或正交更新 | 35% | **79%** |
| 使用 clipping 或 update 控制 | 60% | **79%** |
| 调用 evolved library 模块 | 0% | **82%** |
| 平均修改程序次数 | 9.1 | **3.5** |

> 📌 RSI-OPT 更倾向于早期做出高质量变更并稳定执行，而非频繁试错

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **策略可以跨 run 演化并带来实质性收益**：
   - RSI-OPT 成功构建了一个可进化的 optimizer agent 系统
   - 经过 5 代进化，training suite 性能得到持续提升

2. ✅ **知识表现为可重用的程序与模块**：
   - Meta-agent 积累了如 Muon 应用于特定层、learning rate decay 到零等通用技巧
   - 开发了 CPU proxy 实验验证假设（如在小型 UNet 上验证 Muon 效果）

3. ✅ **跨 LLM 迁移有效，尤其弥补弱模型短板**：
   - 将 Opus 进化出的 policy 直接用于 GPT-5.5/GPT-6.1，大幅提升其表现
   - 表明“优化知识”具有一定的通用性和可移植性

4. ✅ **初始适配为主，持续调整助泛化**：
   - 大部分 seen task 改进来自第一次 optimizer 修改
   - 但在 unseen 任务上，持续观察与调整仍能带来额外收益

5. ⚠️ **存在 specialization 风险**：
   - 随着进化深入，策略逐渐拟合 training suite（如针对 UNet 设计 Muon）
   - 导致 held-out ML 性能在第 1 代后出现回落（见 Figure 3）

6. ⚠️ **迁移效果取决于 LLM 能力匹配**：
   - Opus 本身较强，迁入更强策略反而干扰原有策略
   - GPT-5.5 原本较弱，迁入后全面受益

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **任务规模小** | 所有实验为 single-GPU、短训练周期，未验证大规模场景 |
| **间接信息泄露风险** | Archive summaries 可能传递关于 held-out 任务的信息，无法完全排除 |
| **成本高昂** | 外层进化需大量 evaluation，实际部署门槛高 |
| **未隔离继承价值** | 缺少与“每代从零生成”的 matched-budget 对照实验 |
| **安全边界不足** | 生成代码运行在训练进程中，沙箱不能完全防止状态篡改 |

### 未来工作方向
1. **扩大进化任务集**：引入更多样化的任务以减少 specialization
2. **建立独立验证集**：用于 early stopping 或 policy selection，避免过拟合
3. **探索轻量化进化机制**：如增量更新、稀疏继承等降低计算开销
4. **结合权重训练**：当 harness evolution 触及天花板时（见 [28]），转向 fine-tuning LLM weights
5. **增强安全性与鲁棒性**：完善代码沙箱、引入形式化验证机制
6. **应用于真实科研场景**：如气候模拟、分子动力学中的 PINN 优化

---

> **总结一句话**：  
> RSI-OPT 首次实现了 optimizer agent 的**跨训练 run 递归自我改进**，通过显式的策略表示（prompt + playbook + library）和 archive-based 进化机制，在多个 ML 与 PDE 任务上取得了显著性能提升，并展示了可解释、可迁移的优化知识积累路径。

</details>

---

### 12. [HyScale: Scaling Long-Context Training for Learned Sparse Attention](https://arxiv.org/abs/2610.32057)

**Authors**: Yangsheng Deng, Xinjing Huang, Yulong Wang, Yunzhuo Liu, Jun Hu, Shuyong Tang, Junchi Zeng, Mingji Han, Ray Ying, Key Zhang, Luo Mai  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.32057v1  

#### Abstract
We present HyScale, a long-context training system for learned sparse attention. Learned sparse attention is a promising architecture for million-token contexts, widely adopted in production since DeepSeek-V3.2. Selection, however, grows into the training bottleneck. For each query, it scores every ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：HyScale: Scaling Long-Context Training for Learned Sparse Attention**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
在百万级 token 上下文长度（long-context）的大语言模型训练中，**learned sparse attention**（如 DSA、CSA、QSA）虽然通过选择性关注机制降低了稀疏注意力的计算量，但其前置的 **selection 步骤**（即为每个 query 打分并选出 top-k 个 token）已成为新的训练瓶颈。

具体问题包括：
- **Materialized Scores**：传统系统会先生成一个完整的 $n \times L$ 分数表（score table），占用大量显存和带宽，而最终只保留 top-k 条目。
- **Repeated Selection**：在激活重计算（activation recomputation）过程中，selection 被重复执行两次（前向与重计算），导致计算开销翻倍。
- **通信与内核效率低下**：context parallelism（CP）下的通信未被有效隐藏，且稀疏内核存在冗余内存访问。

### **提出了什么新方法或新思路**
HyScale 提出两个核心技术与一个端到端系统优化框架：

#### **(1) HyTopK：在线式精确 top-k 选择器**
- **核心思想**：不显式构建分数表，而是边打分边筛选。
- 采用 **optimistic strided sampling** 快速估计阈值，仅保留可能进入 top-k 的候选。
- 引入 **lightweight host-free recovery** 在 GPU 上处理极少数失败查询，避免回退到全量计算。
- 保证 **deterministic selection**，通过 radix select 和 radix sort 实现比特级可重现。

> ✅ 优势：无需存储完整 score table，节省显存；支持高吞吐、低延迟、确定性的训练。

#### **(2) Selection Replay：细粒度重计算策略**
- **核心思想**：selection 输出的是索引集合（indices），而非连续激活值，因此可在前向传递中缓存这些 indices，并在重计算时直接复用。
- 支持两种模式：
  - **Index Replay**：仅保存 indices，重计算时重新打分所选条目。
  - **Score Replay**：同时保存 scores，完全跳过打分。
- 可将 selection 的重计算成本从 $O(L)$ 降至 $O(k)$。

> ✅ 优势：消除重复 selection 开销，显著降低计算时间与内存占用。

#### **(3) Holistic System Optimizations**
- **Decoupled CP Overlap**：将 indexer key 与 KV state 的 all-gather 解耦，分别与 query projection 和 selection 重叠，隐藏通信开销。
- **Locality-aware Sparse Kernels**：利用相邻 queries 高度相似的选择结果，合并处理以减少共享 key 的重复加载与原子操作。
- 其他实现优化：CP layout cache、sequence boundary cache、kernel fusion、permutation-free KV gather。

> ✅ 优势：全面提升系统级效率，尤其在大规模分布式场景下表现优异。

---

## 2. **核心实验方法和设置**

### **使用的模型与架构**
- 主要测试模型：
  - **GLM-5.1**（基于 DSA，token-level sparse attention）
  - **DeepSeek-V4-Flash**（使用 CSA，block-level）
  - **Qwen3.8-Flash-Next**（使用 QSA，block-level）

### **实验平台**
- **H20 集群**：512 台节点，每台 8× NVIDIA H20 GPU（96GB 显存），用于 GLM-5.1 端到端训练。
- **H800 集群**：32 台节点，每台 8× NVIDIA H800 GPU（80GB 显存），用于 block-level 架构评估。

### **实验设置**
- **上下文长度**：256K、512K、1M tokens
- **每 GPU token 数**：8K
- **全局 batch size**：固定为 4M tokens
- **精度**：BF16
- **激活重计算**：默认启用 full activation recomputation
- **并行策略**：
  - H20 实验：PP × CP × DP × EP 组合（如 PP4×CP128 at 1M）
  - H800 实验：CP32

### **评估指标**
- **Step Time**（单步耗时）
- **MFU**（Model FLOPs Utilization）
- **Peak GPU Memory Usage**
- **Speedup over Megatron-LM**
- **Ablation Study**：逐步添加 HyTopK、Selection Replay、系统优化，分析各组件贡献

### **基线方法对比**
- **Megatron-LM**（v0.19.2 + cuDNN DSA backend）作为主要 baseline
- 对比项包括：
  - 是否 materialize score table
  - 是否重复 selection
  - 是否使用优化通信与内核

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

| 模型 | 上下文长度 | GPU 数量 | 加速比（vs Megatron-LM） |
|------|------------|----------|------------------------|
| GLM-5.1 | 1M | 512 H20 | **2.07×** |
| DeepSeek-V4-Flash (CSA) | 1M | 32 H800 | **2.01×** |
| Qwen3.8-Flash-Next (QSA) | 1M | 32 H800 | **1.57×** |

> 🔼 所有加速均来自对 selection 瓶颈的全面优化。

### **与基线方法的对比结果**
- 在 1M 上下文下，Megatron-LM 中 selection 占据 **47% 的 step time**，是最大瓶颈。
- HyScale 将 selection 时间大幅压缩：
  - **前向 selection**：从 35.9s → 20.8s（HyTopK 贡献）
  - **重计算 selection**：从 37.0s → 1.0s（Selection Replay 贡献）
- Pipeline wait 减半：从 106.6s → 51.7s，因 attention 层变快而缓解流水线阻塞。

### **消融实验结果（Ablation Study）**

#### **GLM-5.1 在 1M 上的逐层消融（图9b）**
| 组件 | 相对于 Megatron-LM 的时间节省 |
|------|-------------------------------|
| + HyTopK | -37% |
| + Selection Replay | 再 -18%（累计 -55%） |
| + 系统优化（CP overlap + locality-aware kernels 等） | 再 -11%（累计 **-66%**） |

> 💡 结论：HyTopK 和 Selection Replay 是主要驱动力，合计贡献约 **84% 的总收益**。

#### **Block-level 架构消融（图10）**
- 在 CSA 和 QSA 上同样取得显著加速：
  - CSA：**2.01×**
  - QSA：**1.57×**
- HyTopK 增益略小（因 block-level 本身评分更少），但仍关键。

### **其他重要结果**
- **显存效率**：
  - Megatron-LM 在 1M 上需 32 GiB 存储 score table，难以容纳。
  - HyTopK 峰值显存仅 **1.5–1.9 GiB/layer**，几乎恒定增长。
- **扩展性更强**：
  - 弱扩展中，HyScale 在 8M context 下仍保持良好性能，而 Megatron-LM 性能崩溃（190× 更慢 vs HyScale 的 11×）。
- **恢复机制高效**：
  - 实际训练中平均失败查询 < 4 个，host-free recovery 开销 < 5%。

---

## 4. **关键结论和发现**

### **主要发现**
1. **Selection 不应被视为标准 activation**，其输出仅为 indices，具有非微分性和局部可复用性。
2. **HyTopK 成功实现了无 score table 的精确 top-k 选择**，兼顾速度、内存、确定性。
3. **Selection Replay 将重计算开销降阶至 $O(k)$**，从根本上解决重复计算问题。
4. **系统级协同设计至关重要**：解耦通信、利用 locality、融合内核共同释放性能潜力。
5. **HyScale 具备强通用性**：不仅适用于 token-level（DSA），也适配 block-level（CSA/QSA）等多种 sparse attention 架构。

### **方法的局限性**
- **依赖 indexer 设计假设**：要求 indexer loss 仅通过 selected scores 更新，若未来出现更复杂的 selection 学习机制，可能需要调整 replay 策略。
- **当前聚焦于训练阶段**：未涉及推理优化（尽管部分技术如 HyTopK 可迁移）。
- **硬件依赖较强**：某些优化（如 locality-aware kernel）依赖现代 GPU 的 warp-level 并发能力，在低端设备上增益可能受限。

### **未来工作方向**
- 推广至更多 sparse attention 变体（如 MSA、PIVOT、IndexCache 等）。
- 结合 selection caching 或跨层 index 复用（如 IndexCache）进一步减少计算。
- 将 HyTopK 与 Selection Replay 应用于 **inference prefill 阶段**，提升长序列推理效率。
- 探索 **异构集群** 或 **多模态上下文** 场景下的扩展性。

---

> ✅ **总结一句话**：  
> **HyScale 通过洞察“selection 输出仅为 indices”这一本质属性，提出 HyTopK 与 Selection Replay 两大创新，并结合系统级优化，在百万 token 训练中实现高达 2.07× 的端到端加速，成为 learned sparse attention 训练的新标杆系统。**

</details>

---

### 13. [MPQ: A Message-Passing View of Post-Training Quantization](https://arxiv.org/abs/2610.32127)

**Authors**: Yanshuo Chen, Chenxi Liu, Reza Shirkavand, Heng Huang  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.32127v1  

#### Abstract
Post-training quantization (PTQ) is a key technique for efficiently deploying large language models, as it compresses weights into low-bit representations without retraining. Existing PTQ solvers, including GPTQ, LDLQ, and block coordinate descent, commit each weight block to a single codeword befor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MPQ: A Message-Passing View of Post-Training Quantization 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Post-Training Quantization (PTQ)** 方法（如 GPTQ、LDLQ 和 block coordinate descent）在量化过程中采用**硬决策（hard-decision）策略**：即对每个权重块（weight block）立即分配一个唯一的码字（codeword），而忽略其他可能的候选码字。这种早期承诺（early commitment）导致模型无法探索近似最优的替代方案，尤其在低比特（low-bit）量化场景下，多个码字可能产生相似的局部误差，从而造成全局次优解。

此外，随着预处理（preprocessing）、码本设计（codebook）和目标函数优化的进步，量化问题的条件已显著改善，但**离散求解器（discrete solver）** 仍是相对未被充分探索的方向。

### 提出的新方法与新思路
本文提出 **Message-Passing Quantization (MPQ)**，将行级 PTQ 视为在离散码本上的后验推断问题，并引入**消息传递（message passing）视角**来保留软信息（soft information）：

- 将 PTQ 建模为在高斯似然下的有限温度吉布斯测度（finite-temperature Gibbs measure）上的后验推理。
- 现有方法可视为该框架下的**零温极限（zero-temperature limit）**，仅保留每块的最佳码字。
- MPQ 在推理阶段维持各块对多个潜在码字的**软分布（soft posterior）**，通过消息传递让这些分布相互影响，最后才进行硬决策输出。

MPQ 是一个两阶段求解器：
1. **第一阶段（Soft Exploration）**：使用 **Vector Approximate Message Passing (VAMP)** 在有限温度下计算块级后验分布，聚合跨块的近似最优码字分配信息。
2. **第二阶段（Hard Refinement）**：基于 VAMP 输出的软后验，通过**边际 MAP 投影**生成初始硬量化结果，并用**精确的块坐标下降法（exact block coordinate descent, CD）** 进行局部优化。

### 相比现有方法的优势
- **推迟决策（Deferred Decision）**：避免早期硬分配带来的次优路径问题，尤其在低比特场景下更有效。
- **利用软信息**：VAMP 阶段保留了多个高概率码字的信息，使最终选择更具鲁棒性。
- **组合优势**：结合了全局软推理与局部精确搜索，优于纯贪婪或纯随机方法。
- **兼容性强**：可与 activation-aware scaling、incoherence processing 等预处理技术良好组合。

---

## 2. 核心实验方法和设置

### 数据集
- **校准数据（Calibration Data）**：用于量化过程中的 Hessian 矩阵估计。
  - `C4`：128 条长度为 2048 的训练集序列。
  - `WikiText-2`：部分实验中也使用。
- **评估数据**：保持激活值为 16 位进行推理评估。
  - **Perplexity**：在 `WikiText-2` 和 `C4` 上测量。
  - **Zero-Shot Accuracy**：在以下任务上使用 `lm-eval harness` 测量：
    - ARC-Challenge / ARC-Easy
    - HellaSwag
    - PIQA
    - Winogrande

### 实验设置
- **模型家族**：
  - **Llama** 系列：Llama3.2-1B, Llama3.2-3B, Llama3-8B, Llama2-13B, Llama2-70B
  - **Qwen3-8B**
  - **Mistral-7B**
- **量化配置**：
  - **3/4-bit Scalar Quantization**：组大小（group size）为 128，使用 activation-aware scaling + RHT 预处理。
  - **2-bit Vector Quantization**：使用 E8P12 码本，配合 RHT 预处理。
- **硬件**：RTX 5090 单卡，排除预处理、导出和评估时间，仅统计 solver 时间。

### 评估指标
- **Perplexity (PPL)**：衡量语言建模能力，越低越好。
- **Zero-Shot Accuracy (%)**：多任务平均准确率，越高越好。
- **Solver Time (s)**：量化求解时间。
- **Peak Memory (GB)**：CUDA 最大占用内存。

### 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **GPTQ** | Greedy Solver | 使用激活顺序列处理的最强变体 |
| **GPTQ+CD** | Polished Greedy | GPTQ 初始化 + 3 轮 block CD |
| **LDLQ** | Greedy (Lattice) | QuIP# 中使用的向量量化基线 |
| **LDLQ+CD** | Polished Lattice | LDLQ + 3 轮 CD |
| **QuantEase** | Strong CD Baseline | 显式 block coordinate descent 求解器 |
| **OJBKQ** | Randomized Greedy | 结合 Babai 与 Klein 采样的随机轨迹方法 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 2）

#### Perplexity 对比（越低越好）
| 方法 | Llama3-8B (3-bit) | Llama3-8B (2-bit E8P12) |
|------|-------------------|------------------------|
| GPTQ | 8.29 / 11.61 (Wiki/C4) | — |
| LDLQ | — | 15.01 / 18.17 |
| LDLQ+CD3 | — | 12.76 / 15.96 |
| OJBKQ | 7.95 / 11.08 | 13.84 / 16.57 |
| **MPQ** | **7.81 / 10.88** | **11.63 / 14.77** |

> ✅ MPQ 在所有比特宽度下均优于最强基线，且在 **2-bit 时提升最大**（从 12.76 → 11.63 PPL）。

#### Zero-Shot 准确率对比（越高越好）
| 模型 | 方法 | 平均准确率 (%) |
|------|------|----------------|
| Llama3-8B (3-bit) | OJBKQ | 68.5 |
| | **MPQ** | **68.6** |
| Llama3-8B (2-bit) | LDLQ+CD | 59.3 |
| | **MPQ** | **59.7** |
| Mistral-7B (2-bit) | LDLQ+CD | 67.5 |
| | **MPQ** | **67.9** |

> ✅ MPQ 在所有模型和比特宽度下均达到最佳或并列最佳，尤其在 **2-bit lattice regime 提升达 0.4–1.4 个百分点**。

### 与基线方法的对比结果
- **全面超越**：MPQ 在 **Llama、Qwen、Mistral** 家族的所有测试模型上均优于 GPTQ、LDLQ、QuantEase 和 OJBKQ。
- **低比特增益显著**：在 **2-bit E8P12 向量量化** 下性能提升最明显，说明其在极端压缩场景最具价值。
- **稳定性强**：在三次独立 C4 校准数据抽样下，MPQ 的平均 PPL 和标准差均优于基线（见 Appendix A.7）。

### 消融实验结果
#### （1）CD 修复作用分析（Figure 4a）
- **3-bit scalar**：VAMP 阶段精度高，CD 几乎不修改任何块（cleanup fraction ≈ 0），表明 VAMP 已接近最优。
- **2-bit E8P12**：VAMP 精度较低，CD 修改了约 **12.7% 的块**，说明 VAMP 提供了一个更好的“盆地”，CD 在此基础上精细优化。

> 🔍 表明 MPQ 的优势来源于 VAMP 的高质量初始化，而非单纯依赖 CD。

#### （2）AWQ 对线性模块的影响（Figure 4b）
- 验证了 **Proposition 5.2** 的理论预测：AWQ 预处理通常会降低线性模块误差 $ E_{\text{lin}} $。
- 在 96 个 scale 组中，AWQ 降低了 84 组的 $ E_{\text{lin}} $，且符号与理论预测一致率达 95/96。

> ✅ 理论与实践一致，支持 AWQ 通过改善 Hessian 条件来提升量化效果。

#### （3）不同 VAMP 迭代数与 CD 轮数的权衡（Figure 3）
- **3-bit scalar**：仅需 4 次 VAMP 迭代即可达到饱和性能，无需 CD。
- **2-bit E8P12**：性能随 VAMP 和 CD 轮数持续提升，16 次 VAMP + 3 次 CD 达到最优。

> ⚖️ 用户可根据资源预算灵活调整 VAMP 与 CD 的投入。

---

## 4. 关键结论和发现

### 主要发现
1. **现有 PTQ 求解器本质是零温消息传递**：GPTQ、LDLQ 等方法可统一视为只传递点估计的零温极限。
2. **有限温度消息传递更优**：MPQ 通过 VAMP 维持软后验，允许跨块信息交互，避免早期次优决策。
3. **低比特场景收益最大**：在 **2-bit lattice 和 3-bit scalar** 量化中提升最显著，这对边缘设备部署至关重要。
4. **VAMP 初始化质量决定上限**：CD 的作用是局部修复，而全局性能由 VAMP 所选“盆地”决定。
5. **与预处理方法正交兼容**：MPQ 可与 AWQ、RHT 等技术组合，进一步提升性能。

### 方法的局限性
- **理论分析为渐近近似**：Theorem 5.1 和 Proposition 5.2 基于理想化的右正交不变（ROI）假设，实际 Hessian 并非严格满足。
- **码本范围有限**：实验集中在 scalar 和 lattice 码本（E8P12, D4），尚未验证在 AQLM 或 QTIP 等学习型码本上的表现。
- **计算开销较高**：虽然离线运行，但 MPQ 求解时间仍高于 GPTQ，在 3-bit 场景下约为其 6–7 倍（见 Figure 2）。

### 未来工作方向
- 将 MPQ 扩展至 **learned vector codebooks**（如 AQLM）和 **trellis-coded quantization**（如 QTIP）。
- 探索更高效的 **有限温度推理算法**，以降低 VAMP 阶段的计算成本。
- 研究 **动态调整温度** 或 **自适应 VAMP 迭代数** 的策略，实现精度与速度的更好平衡。
- 将消息传递思想应用于 **activation-aware quantization** 或 **layer-wise joint optimization**。

---

> 📌 **总结**：  
> MPQ 提出了一种全新的 **message-passing 视角** 来理解 PTQ，打破了传统硬决策求解器的局限。其实验结果表明，在 **2-bit 和 3-bit 低比特量化** 中，MPQ 显著优于当前最强基线，为大模型在边缘设备上的高效部署提供了新的可能性。

</details>

---

### 14. [LeafSkill: Efficient Leaf-Level Action Grounding for Recursive LLM Agents](https://arxiv.org/abs/2610.32412)

**Authors**: Shen Liu, Ze Zhou, Zhenguo Xu, Shaopu Wang, Yike Gao, Chunlei Wang  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.32412v1  

#### Abstract
Recursive large language model (LLM) agents solve long-horizon tasks through decomposition, execution, and environmental feedback. Yet after producing an atomic subtask, they often invoke the same large model again to translate a familiar instruction into an executable action. We observe that high-l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*LeafSkill: Efficient Leaf-Level Action Grounding for Recursive LLM Agents*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **recursive LLM agents**（如 ReCAP）中，尽管任务已通过分解（decomposition）细化为具体的叶节点任务（leaf task），但仍需调用大型 host model 将该任务“接地”为一个可执行的环境动作（action）。这一过程被称为 **leaf-level action grounding**。

作者通过分析发现，这种“最后一英里”的动作接地操作消耗了 **22.9–44.5% 的 host-model tokens**，造成了显著的推理开销。因此，如何高效地完成这一局部、重复性强的操作，成为提升整体效率的关键瓶颈。

### 提出了什么新方法或新思路
提出 **LeafSkill** —— 一种**外部化的、轻量级的动作接地层**（external action-grounding layer），专门用于处理递归 LLM agent 中的叶级任务到动作的映射。

其核心架构为 **Retrieve-Bind-Select** 三层机制：
- **Semantic Retrieval**：从冻结的技能库中检索语义相关的 task-action 模板。
- **Deterministic Binding**：将模板中的占位符绑定为当前环境中存在的实体，并仅保留当前状态下合法（admissible）的动作。
- **Lightweight Controller Selection**：由一个小型 LLM（如 Qwen3-1.7B）选择最合适的候选动作，或主动放弃（abstain），交还控制权给 host。

关键设计原则：
- **不替代规划逻辑**：保持 host 的递归规划、恢复和目标完成机制不变。
- **安全回退机制**：当无法可靠复用时，返回 `abstain`，host 继续处理。
- **请求本地化决策**（request-local）：所有候选动作标识是临时生成的，防止模型记忆全局 ID。

### 相比现有方法的优势
| 对比维度 | LeafSkill | 其他系统（如 ExpRAG, GoS, GraSP） |
|--------|---------|-------------------------------|
| 复用粒度 | task-action 模板对 | 完整轨迹、技能图、多步流程 |
| 决策权限 | 仅限于当前状态下的有限动作集合 | 可能影响策略生成或执行流 |
| 规划所有权 | 完全保留在 host | 部分转移至外部模块 |
| 扩展性 | 新技能可动态添加而不改变 selector 输出空间 | 通常需要重新训练或结构调整 |

因此，LeafSkill 在**最小侵入性前提下实现了高效的 token 节省**，是一种与现有递归 agent 架构正交的优化方案。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Robotouille**：异步烹饪任务环境，包含同步与带等待/并行依赖的任务。
  - Split A/B/C 各含 200 个任务，分别用于构建、消融与独立测试。
- **ALFWorld**：基于文本的虚拟家庭任务环境，模拟物体交互。
  - `valid_seen`（140 tasks）：用于技能库构建与适配。
  - `valid_unseen`（134 tasks）：最终评估泛化能力。

### 实验设置
- **Host Model**：Qwen3.5-27B
- **Planner**：ReCAP（作为递归 backbone）
- **LeafSkill Components**：
  - Retriever：BGE-small-en-v1.5
  - Controller：Qwen3-1.7B（仅 LoRA 微调）
- **Library Construction**：
  - 从成功轨迹中提取 task-action 对，进行标准化（canonicalization）形成 skill 库。
  - Robotouille 初始库含 446 skills，扩展后达 588。
  - ALFWorld 技能库含 228 skills。

### 评估指标
- **Host-token reduction**：在双方均成功的任务上，比较 host 模型使用的 prompt + completion tokens。
- **Host-call reduction**：host 调用次数减少比例。
- **Success Rate**：完整任务成功率（计入失败即终止的情况）。
- **控制器开销**：Qwen3-1.7B 的额外 token 开销。
- **交叉运行鲁棒性**：三次独立重复实验的所有 3×3 配对平均结果。

### 基线方法对比
- **ReCAP baseline**：原始递归 agent，无 LeafSkill。
- **不同 controller 路由**：
  - Bare 1.7B（未微调）
  - Robotouille-only SFT
  - ALFWorld-only SFT
  - Sequential SFT（默认）
- **消融变体**：
  - RawPair-Exact（精确字符串匹配）
  - BGE Top-1 / Top-5（仅检索+绑定）
  - Canonical-Exact + SFT（完整 ablation）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Host-Token Reduction（联合解决任务上的平均节省）
| 环境 | Token Reduction | Call Reduction |
|------|------------------|---------------|
| **Robotouille (Group B)** | **23.1%** | 22.5% |
| **Robotouille (独立 Group C)** | **14.4%**（95% CI: [3.0, 24.5]） | 14.1% |
| **ALFWorld (valid_unseen)** | **41.3%** | 41.8% |

> 注：所有结果均为三个独立重复实验之间的九种组合平均值，确保统计稳健性。

#### ✅ 成功率表现（numerically higher）
| 环境 | ReCAP | LeafSkill | Δ (%) | p-value |
|------|-------|----------|--------|--------|
| Robotouille B | 55.0% | 62.2% | +7.2 | – |
| Robotouille C | 60.5% | 68.5% | +8.0 | 0.0259 |
| ALFWorld unseen | 89.8% | 91.8% | +2.0 | – |

> 表明 LeafSkill 不仅更高效，且在多数情况下成功率更高。

#### ✅ 控制器开销（可控）
- 每 episode 平均消耗约 **6,900–9,952 tokens**（来自 Qwen3-1.7B）。
- 相比节省的 27B 模型数十万级 token，代价极小。

---

### 消融实验结果（Ablation Studies）

| 方法 | Token Reduction | Success | 说明 |
|------|------------------|--------|------|
| RawPair-Exact | 0.20% | 60.5% | 字符串缓存几乎无效 |
| RawPair-BGE (semantic retrieval) | 8.08% | 63.5% | 语义检索显著提升召回 |
| BGE Top-1 + Direct | 13.08% | – | 绑定确保可执行性 |
| Canonical-Exact + SFT | 29.60% | 67.0% | 完整 pipeline 效果明显 |
| BGE Top-5 + Base | 27.01% | 56.5% | 无学习选择器效果下降 |
| **Full LeafSkill (SFT)** | **23.09%** | **62.2%** | 学习型选择器平衡效率与准确 |

> 结论：**Retrieval 提供 paraphrase coverage，Binding 保证 executability，Selection 决定是否复用及选哪个**。

---

### 后训练技能库增长实验（Post-Training Growth）
- 将技能库从 **446 → 588**（新增 142 skills），**不重新训练 selector**。
- 在共同解决的 88 个任务上：
  - Token reduction 从 **18.3% → 25.6%**
  - Call reduction 从 **17.7% → 24.2%**

> 表明：**兼容的新技能可在部署时直接加入，无需更新 controller，仍能提升效率**。

---

## 4. 关键结论和发现

### 主要发现
1. **Leaf-level action grounding 是递归 agent 中不可忽视的成本中心**，占 host tokens 的 **22.9–44.5%**。
2. **LeafSkill 显著降低 host 推理成本**：
   - 在 Robotouille 上节省 **23.1% host tokens**
   - 在 ALFWorld 上节省高达 **41.3%**
3. **Retrieve-Bind-Select 架构有效分离关注点**：
   - Retrieval 解决语义多样性（paraphrase）
   - Binding 强制动作合法性（executability）
   - Selection 学会何时复用、何时放弃
4. **方法具有良好的扩展性和兼容性**：
   - 冻结的 selector 可吸收新技能，进一步提升效率。
   - 不干扰原有规划与恢复机制，易于集成。

### 方法的局限性
- **依赖离散动作接口**：适用于命令式动作空间（如 `pick up X`, `open Y`），难以直接应用于连续控制或开放生成场景。
- **需要环境适配器定义“eligible leaf”**：并非所有环境都天然支持原子任务边界识别（如 ALFWorld 需手动定义 downward phase）。
- **技能复用受限于历史经验分布**：若新任务涉及全新动词或对象组合，可能无法命中。
- **当前验证集中在特定 planner 和 host**：通用性有待在更多 agent 架构中验证。

### 未来工作方向
- 支持 **continuous 或 programmatic actions** 的扩展版本。
- 自动化 **environment adapter** 的构建，降低部署门槛。
- 探索 **multi-modal grounding**（结合视觉或其他感知输入）以增强 binding 准确性。
- 将 LeafSkill 思想推广至其他中间步骤的“最后一英里”优化（如 observation summarization）。

---

> 📌 **总体评价**：  
> LeafSkill 提出了一种**精准而克制的优化范式**——不在顶层推翻现有递归 agent 设计，而是识别出高频、局部、可模式化的子问题（leaf grounding），并通过一个轻量、安全、可扩展的外部模块加以解决。它展示了 **state-bound action reuse** 作为递归 LLM agent 的实用补充路径的巨大潜力。

</details>

---

### 15. [CASE: Confusion-Aware Skill Evolution for Frozen Multimodal Emotion Models](https://arxiv.org/abs/2610.32481)

**Authors**: Siyu Wu, Zeyu Liang, Linghao Wang, Qiyun Zheng, Yi Yao, Ling Lo, Hongxia Xie  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.32481v1  

#### Abstract
Multimodal large language models (MLLMs) have substantially advanced Multimodal Emotion Recognition (MER) by enabling fine-grained emotion understanding through reasoning over multimodal evidence. However, they still exhibit systematic confusion between semantically adjacent emotions with highly ove...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CASE: Confusion-Aware Skill Evolution for Frozen Multimodal Emotion Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Multimodal Large Language Models (MLLMs)** 在多模态情感识别（MER）任务中虽然能整合视觉、听觉和文本线索进行细粒度推理，但仍存在**系统性混淆**（systematic confusion）问题——即对语义相近的情感类别（如 *worried* vs *sad* 或 *fear* vs *surprise*）难以准确区分。

传统方法通常依赖**全局参数微调**（global fine-tuning），这种方式成本高、不灵活，且忽略了情感混淆具有**局部化、成对性**的特点。

---

### 🚀 提出的新方法：Confusion-Aware Skill Evolution (CASE)

CASE 是首个将 MER 建模为“**成对边界判别**”而非“全局分类”的 **agent-skill 框架**，其核心思想是：

- 将情感混淆视为局部决策边界的失败；
- 构建一个稀疏、自演化的外部 **EmotionSkill Library**，每个技能专门解决一对易混淆情绪之间的判别问题；
- 推理时动态检索并激活对应的 EmotionSkill 来增强 MLLM 的边界感知能力；
- 整个 MLLM 主干保持 **frozen**，仅更新外部技能库。

#### 创新机制：
1. **Selective Confusion-Pair Inference**  
   只有当 MLLM 输出前两名预测情绪模糊时才触发技能干预，避免干扰置信度高的预测。
   
2. **Recoverability-Aware Skill Evolution**  
   基于错误轨迹（failure trajectories）与成功锚点（successful anchors）对比，通过 LLM 编辑器生成可复用的文本规则补丁（textual patch）来迭代优化技能。

3. **Behaviorally Validated Update & Sparse Expansion**  
   所有技能更新必须经过独立验证集的行为测试（behavioral validation），防止过拟合或标签坍缩；新技能仅在重复出现未覆盖的混淆对时才创建，确保库的稀疏性和实用性。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | CASE |
|------|--------|------|
| 参数更新方式 | 全局微调（costly, non-reusable） | 零参数更新，仅进化外部技能 |
| 错误处理策略 | 忽视混淆结构，统一优化 | 显式建模成对混淆，精准修复边界 |
| 知识复用性 | 模型绑定，难迁移 | 技能模块化，跨 backbone 和任务迁移 |
| 适应性 | 固定标签空间 | 支持 open-vocabulary 和冲突感知场景 |

> ✅ **优势总结**：CASE 实现了**低成本、高灵活性、强泛化性**的情感识别增强，尤其适用于部署后持续进化的实际场景。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 描述 | 特点 |
|-------|------|------|
| **DFEW** | 动态面部表情识别数据集，真实环境视频 | Closed-set, in-the-wild, 视频级情感识别 |
| **CA-MER** | 跨模态情感冲突基准，含音频-视觉冲突子集 | 冲突感知 MER，测试模型处理矛盾线索的能力 |
| **EMER** | 开放词汇情感推理数据集 | Open-vocabulary setting，要求超越预定义标签的理解 |

---

### ⚙️ 实验设置与评估指标

| 设置项 | 内容 |
|-------|------|
| **Backbones** | Qwen3-VL-8B, Emotion-Qwen, Gemini-3.5-Flash（全部 frozen） |
| **编辑器 LLM** | GPT-4o 或 GPT-5.6-sol（用于技能演化） |
| **技能形式** | CASE-Full（注入完整技能）、CASE-Retrieval（仅检索相关片段） |
| **评估指标** | <ul><li>CA-MER / EMER: Accuracy (%)</li><li>DFEW: UAR (Unweighted Average Recall), WAR</li></ul> |
| **划分策略** | Evolution / Validation / Test 三者严格分离 |

---

### 🆚 基线方法对比

- **Closed-source baselines**: Gemini-3.5-Flash
- **Open-source SOTA**:  
  - Video-LLaMA, Video-LLaVA  
  - Emotion-LLaMA, MoSEAR  
  - Qwen2.5-VL-7B, VideoAuto-R1
- **Prior MER-specific models**

> CASE 不改变 backbone，因此直接与原始 frozen 模型及已有 SOTA 对比性能增益。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 3）

| Model | CA-MER (Overall Acc.) | 提升幅度 |
|-------|------------------------|----------|
| Qwen3-VL-8B (frozen) | 46.80% | — |
| **CASE (Qwen3-VL-8B)** | **59.13%** | **+12.33 pp** |
| Emotion-Qwen (frozen) | 60.12% | — |
| **CASE (Emotion-Qwen)** | **61.30%** | +1.18 pp |
| Gemini-3.5-Flash (frozen) | 66.73% | — |
| **CASE (Gemini-3.5-Flash)** | **70.60%** | **+3.87 pp** |

> 💡 **最高提升达 +12.33 个百分点**，表明 CASE 对弱情感识别 backbone 提升显著。

---

### 📈 DFEW 上的表现（Table 2）

| Model | UAR (%) | WAR (%) |
|-------|---------|---------|
| Qwen3-VL-8B (frozen) | 51.59 | 59.42 |
| **CASE (Qwen3-VL-8B)** | **55.30** | **61.21** |
| Gemini-3.5-Flash (frozen) | 54.49 | 60.92 |
| **CASE (Gemini-3.5-Flash)** | **60.88** | **68.69** |

> ✅ 所有 backbone 均获得稳定提升，说明 CASE 泛化性强。

---

### 🌐 EMER 上的开放词汇泛化（Figure 4）

- CASE 在未经 EMER 数据训练的情况下，仍能提升所有 backbone 的表现；
- **Emotion-Qwen + CASE-Retrieval 达到 51.58% accuracy**，超过 prior methods；
- 表明 EmotionSkills 可迁移到 **open-vocabulary 情感推理**任务。

---

### 🔍 消融实验结果（Table 4）

| 变体 | Components | CA-MER Acc. |
|------|-----------|-------------|
| Vanilla MLLM | — | 46.80% |
| + Selective Reasoning | ✓ | 47.47% |
| + Static Pairwise Skills | ✓ | 57.00% |
| + Ungated Evolution | ✓✓ | 58.27% |
| **CASE (Full)** | ✓✓✓✓ (with validation gate) | **59.13%** |

> ✅ 各组件贡献明确：
- 成对技能注入带来最大增益（+10.2 pp）
- 轨迹驱动的演化再提升 1.27 pp
- 行为验证门控进一步提升 0.86 pp，并保证稳定性

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **情感混淆高度局部化**：错误集中在少数相邻情绪对（如 *sad-worried*, *angry-worried*），适合用成对技能解决。
2. **外部技能可有效替代参数微调**：CASE 在不修改 backbone 参数的前提下实现显著性能跃升。
3. **技能具备强跨任务迁移能力**：在 DFEW 上演化的技能可直接用于 CA-MER 和 EMER，无需重新训练。
4. **选择性干预更安全高效**：只在 Top-2 情绪模糊时介入，避免破坏已有正确预测（非路由样本准确率达 63.3%）。
5. **行为验证机制至关重要**：114 个候选技能修订中仅 10 个被接受，有效过滤无效或有害更新。

---

### ⚠️ 局限性

1. **依赖高质量反馈信号**：需要 ground truth label 进行轨迹归因，限制其在完全无监督场景的应用。
2. **技能表达受限于 prompt 格式**：目前技能以自然语言指令形式存在，复杂逻辑可能难以编码。
3. **初始技能构建依赖专家设计或 LLM 初始化**：自动化程度仍有提升空间。
4. **当前技能库规模较小**（~21 对）：面对更大情感空间（如 27+ 类）需进一步研究扩展效率。

---

### 🔮 未来工作方向

1. **引入在线学习机制**：结合用户反馈或交互数据实现闭环技能进化。
2. **支持复合情绪与强度建模**：从单一标签向多标签、连续维度情感扩展。
3. **多技能协同推理**：探索多个 EmotionSkills 联合调用以处理多重混淆。
4. **降低对外部编辑器的依赖**：让 MLLM 自主完成技能编辑与验证。
5. **应用于其他认知任务**：如人格识别、意图理解等存在边界模糊问题的任务。

---

## ✅ 总结

**CASE** 提出了一种全新的视角：将情感识别从“静态分类”转变为“动态知识调用”，通过构建一个**可进化、可迁移、模块化的 EmotionSkill 库**，实现了对 frozen MLLMs 的高效增强。实验证明其不仅大幅提升性能，而且具备出色的跨任务、跨模型泛化能力，为未来低代价、可持续的情感智能系统提供了可行路径。

</details>

---

### 16. [Toward Open ISA Supercomputing: A Comparative Study of RISC-V, Arm, and x86 Architectures on HPC Workloads](https://arxiv.org/abs/2610.32632)

**Authors**: Emanuele Venieri, Giacomo Madella, Gabriele Ceccolini, Simone Manoni, Federico Ficarelli, Filippo Barbari, Jenny Lynn Almerol, Elisabetta Boella, Nitin Shukla, Simone Bn\`a, Luca Benini, Andrea bartolini  
**Category**: cs.DC  
**Published**: 2026-10-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.32632v1  

#### Abstract
RISC-V is emerging as a candidate architecture for high-performance computing (HPC), but its competitiveness against established x86 and Arm processors remains difficult to assess because existing studies use different platforms, workloads, and methodologies, and rarely consider energy efficiency. O...

---

### 17. [TGR-EvoDAG: A DAG-based Multi-Agent Code Evolution Framework for Generative Recommendation on Tencent](https://arxiv.org/abs/2610.32326)

**Authors**: Junlin He, Junzhang Zhu, Yudong Li, Yihong Tang, Xuefeng Jiang, Wei Ma, Chengxiang Zhuo, Zang Li  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.32326v1  

#### Abstract
Large Language Model (LLM) agents have enabled automated code evolution for algorithm discovery. However, current recursive self-improvement (RSI) research for sequential recommendation faces a dilemma: general-purpose frameworks lack domain-specific feedback, while recommender-specific frameworks r...

---

### 18. [MisconceptDialogue: Evaluation Criteria Shape Measured Success in LLM Math Tutoring](https://arxiv.org/abs/2610.31679)

**Authors**: Woojin Lee, Yoonkyu Woo, Jinah Kim, Hyeok-Min Gwon, Jin-Xia Huang  
**Category**: cs.CL  
**Published**: 2026-10-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.31679v1  

#### Abstract
Large language models are increasingly used as math tutors, raising two questions: can a single LLM detect a student's error, diagnose the underlying misconception, and remediate it through multi-turn dialogue - and can we trust the numbers that report how well it does? We address both with MISCONCE...

---

### 19. [TaxoSAE: Rethinking Sparse Autoencoders through Taxonomic Logic](https://arxiv.org/abs/2610.31894)

**Authors**: Xinyu Guo, Minglai Yang, Clayton T. Morrison, Eduardo Blanco, Mihai Surdeanu  
**Category**: cs.CL  
**Published**: 2026-10-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.31894v1  

#### Abstract
Recent Sparse Autoencoders (SAEs) decompose the representations of large language models (LLMs) into interpretable features, enabling precise readout and control of encoded knowledge. However, existing SAEs treat each feature as a standalone unit, whereas real-world knowledge is often taxonomically ...

---

### 20. [TACS-RF: Tool-driven Agentic Circuit Synthesis of RF and mmWave Modules](https://arxiv.org/abs/2610.31882)

**Authors**: Tristan Liang, Aravind Nagulu  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.31882v1  

#### Abstract
We present the framework of TACS-RF - a Tool-driven Agentic Circuit Synthesis of RF and mmWave modules. In this framework, a large-language-model (LLM) coding agent (Claude Opus 4.8) directly drives industry-standard electronic-design-automation (EDA) tools such as Cadence Virtuoso/Spectre, Calibre,...

---

### 21. [Understanding Communication Failures in Multi-Agent LLM Systems](https://arxiv.org/abs/2610.32060)

**Authors**: Priya Pitre, Jun Hou, Yi Fang, Naren Ramakrishnan, Xuan Wang  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.32060v1  

#### Abstract
Multi-agent LLM systems aim to improve reasoning through debate, evidence sharing, and mutual correction, but interaction can also introduce failures that final accuracy alone cannot explain. Agents may converge too early, ignore dissent, suppress correct minority answers, or preserve a wrong framin...

---

### 22. [DriveMind: Mind-Evolving Belief Tracking for Closed-Loop Autonomous Driving](https://arxiv.org/abs/2610.32069)

**Authors**: Yan Shi, Kehua Chen, Jiarui Qi, Yihong Tang, Haichen Liao, Yinhai Wang  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.32069v1  

#### Abstract
Recent multimodal large language models (MLLMs) can perceive a traffic scene, reason about it, and propose a driving action within a single model. However, existing MLLM-based drivers reason from scratch at each frame and commit to a single explanation, discarding earlier hypotheses about the scene....

---

### 23. [Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution](https://arxiv.org/abs/2610.32285)

**Authors**: Zhiyong Cao, Hengjie Zhu, Yongxin Wang, Xuchen Li, Jintao Li, Shuxia Liu, Dunqiang Liu, Bo Wang, Kairan Zhang, Cong Lin, JiangDi Yu, Siyuan Liu, Sijia Yao, Yi-Liang Wei, Lingxu Ma, Aijia Wang, Anxin Chen, Zixian Wei, Xiang Ruan, Ruizhi Xu, Ruqian Shi, Baihua Xiao, Sicheng Zhou  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.32285v1  

#### Abstract
LLM agents are increasingly used to solve real-world problems, and their effectiveness depends on both the underlying models and the harnesses that support their operation. However, most existing benchmarks focus on model capabilities and overall agent performance, with limited attention to the eval...

---

### 24. [From Visual Affinity to Learning Credit in Multimodal Reinforcement Learning](https://arxiv.org/abs/2610.32324)

**Authors**: Yuhang Han, Dingning Liu, Yujie Chen, Yaojie Zhang, Jingwei Song, Zequn Yang, Xuanming Shang, Li Ding, Zhijiang Guo, Linfeng Zhang  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.32324v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) improves multimodal reasoning, but uniformly assigning a single response-level advantage to all tokens overlooks the distinct roles of visual observations and subsequent reasoning. Visual affinity alone is insufficient: predictive states may alig...

---

### 25. [Beyond Post-Hoc Rewards: Privileged Reasoning Exploration for Multimodal Embeddings](https://arxiv.org/abs/2610.32474)

**Authors**: Tianyu Yang, Yijun Tian, Lars Benedikt Kaesberg, Jan Philip Wahle, Bela Gipp, Terry Ruas, Daniel Kurzawe  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.32474v1  

#### Abstract
Advances in Multimodal Large Language Models (MLLMs) have driven substantial progress in multimodal embeddings. Recent methods exploit generative capabilities by augmenting embedded inputs with self-generated reasoning to produce more informative embeddings. However, conventional retrieval supervisi...

---

### 26. [From Deliberately Constructed Systemic Representations of the World of Activity to Dialogue: A Five-Level Communication Architecture for Strong Language-Based ASI](https://arxiv.org/abs/2610.32869)

**Authors**: Maxim Osovsky (Adam University, Bishkek, Kyrgyzstan)  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.32869v1  

#### Abstract
This paper proposes a five-level organization of message formation and understanding within thinking and activity in which ASI participates. Its central thesis is that a language model's context is fundamentally incapable of presenting the foundations of utterance formation for examination and verif...

---

### 27. [Free Lunch for LLMs Confidence: The RL Critic Already Knows](https://arxiv.org/abs/2610.31820)

**Authors**: Shujin Wu, Heng Ji  
**Category**: cs.CL  
**Published**: 2026-10-26  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.31820v1  

#### Abstract
Reliable confidence estimation is essential for trustworthy deployment of large language models (LLMs), but dedicated confidence training stage requires additional data and compute. We ask a fundamental question: what prediction problem does confidence training actually solve? Inspired by prior work...

---

### 28. [Semantically Rich, Metrically Affect-Blind: A Geometric Blind Spot in Modern Text Embedders](https://arxiv.org/abs/2610.32122)

**Authors**: Mahdi Naser Moghadasi  
**Category**: cs.CL  
**Published**: 2026-10-26  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.32122v1  

#### Abstract
We report a systematic dissociation in modern text embedders: sentiment is almost perfectly decodable from their embeddings, yet almost entirely absent from their metric. Across eleven frozen encoders spanning 2021 to 2026, including 2026-generation models and the 4B-parameter LLM embedder Qwen3-Emb...

---

### 29. [CounselPlex: A Stressor-Aware Full-Duplex Counseling Model](https://arxiv.org/abs/2610.32702)

**Authors**: Yeonju Kim, Bella Godiva, Yong Man Ro  
**Category**: cs.CL  
**Published**: 2026-10-26  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.32702v1  

#### Abstract
Full-duplex speech models can listen and speak at the same time, enabling natural, low-latency interaction. In counseling, however, these models often focus on the latest utterance and lose track of the seeker's underlying problem. We call this failure Stressor Blindness. We propose CounselPlex, a s...

---

### 30. [Beyond On-Policy vs. Off-Policy: At What Level Should LLM Agent Distillation Follow the Student?](https://arxiv.org/abs/2610.31918)

**Authors**: Ya Gao, Zhengtao Zou, Pekka Marttinen  
**Category**: cs.AI  
**Published**: 2026-10-26  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2610.31918v1  

#### Abstract
Distilling interactive LLM agents requires choosing between teacher-generated demonstrations and student-generated experience, corresponding broadly to off-policy and on-policy distillation. While both have proven effective, we argue that this distinction is inherently hierarchical: the policy that ...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-18 09:04:49 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DualKV: Shared-Prompt Flash Attention for Efficient RL Training with Large Rollouts and Long Contexts](https://arxiv.org/abs/2605.15422)

**Authors**: Jiading Gai, Shuai Zhang, Xiang Song, Bernie Wang, George Karypis  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.15422v1  

#### Abstract
Modern RL post-training methods such as GRPO and DAPO train on $N$ response sequences of $R$ tokens sampled from a shared prompt of $P$ tokens, but standard FlashAttention replicates all $P$ prompt tokens $N$ times across both forward and backward passes -- duplicating compute and memory on identica...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DualKV: Shared-Prompt Flash Attention for Efficient RL Training with Large Rollouts and Long Contexts**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在现代强化学习（RL）后训练方法（如 GRPO 和 DAPO）中，模型需要对共享同一 `prompt` 的 $ N $ 条响应序列进行策略梯度更新。标准的 **FlashAttention-2 (FA2)** 在前向和反向传播过程中会将 `prompt` 的所有 $ P $ 个 token 复制 $ N $ 次，导致大量冗余计算和内存占用。

这种复制在 **大 rollout 因子（$ N \geq 16 $）和长上下文（$ P > 8K $）** 的场景下尤为严重，成为策略更新阶段的主要瓶颈，显著增加训练成本并限制微批次大小（micro-batch size），甚至导致 OOM（Out-of-Memory）。

### **提出的新方法：DualKV**
论文提出了 **DualKV**，这是首个专为 RL 训练设计的 FlashAttention 内核变体，旨在消除共享 `prompt` 的重复计算。其核心思想基于以下观察：

> 在 decoder-only 模型中，由于因果掩码（causal masking）的存在，`prompt` 部分的隐藏状态在所有 $ N $ 个序列中是完全相同的，因此可以只计算一次。

DualKV 通过两个层面实现优化：

#### **(1) 双区域注意力机制（Two-Region Attention）**
- 将一个完整的 attention 计算分解为两个独立调用：
  - **Call 1: Context Self-Attention**  
    对单份 `prompt` 执行一次自注意力计算（使用标准 FA2）。
  - **Call 2: Decoded Attention (DualKV Kernel)**  
    使用自研的 `flash_attn_dualkv_varlen_func` 内核，让每条响应的 query 同时 attend 到共享的 `prompt KV` 和自身的 `response KV`。

#### **(2) 数据流水线重构（Data Pipeline Redesign）**
- 修改 veRL 框架的数据流，确保来自同一 `prompt` 的 $ N $ 条响应被打包到同一个 micro-batch 中，并保持连续排列。
- 跳过 `balance_batch` 和禁用 shuffle，以维持“同 prompt 分组”的结构。

最终，整个模型（不仅是 attention，还包括 Norm、MLP、Projection 等 per-token 操作）只需处理 $ P + NR $ 个 token，而非原来的 $ N(P+R) $，实现了端到端的 token 数量压缩。

### **相比现有方法的优势**
| 方法 | 是否支持训练 | 是否消除 KV 复制 | 是否消除全层计算复制 | 是否精确等价 |
|------|--------------|------------------|------------------------|-------------|
| **Paged Attention / Prefix Caching** | ❌ 仅推理 | ✅ 存储去重 | ❌ | ✅ |
| **Bifurcated Attention** | ❌ 仅推理 | ✅ 注意力内去重 | ❌ | ✅ |
| **Prefix Grouper** | ✅ | ❌（仍传 N 份 KV） | ❌ | ✅ |
| **DualKV (Ours)** | ✅ | ✅ | ✅ | ✅ |

- **数学等价性**：DualKV 与标准 attention 完全等价，无任何近似或精度损失。
- **系统级优化**：首次在 **CUDA 内核级别** 实现了训练中的共享 prompt KV 去重，并解决了多序列并发写入共享 KV 缓冲区的 race condition 问题（通过 fp32 原子累加 + 最终 cast 回 bf16）。
- **可扩展性强**：适用于任意基于采样生成多响应的 RL 方法（GRPO, DAPO, PPO 等）。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LongReason**：一个合成的长上下文数学推理数据集，`prompt` 长达 8,192 tokens，响应最长 2,048 tokens。
- **GSM8K**：用于短上下文验证的小规模数学题数据集。
- **Llama-3.1-8B** 上下文扩展实验也用于验证通用性。

### **实验设置**
- **模型**：
  - 主要使用 **Qwen3-8B** 和 **Qwen3-30B-A3B (MoE)** 模型。
- **硬件**：
  - 单节点：8×H100-SXM5-80GB（p5.48xlarge）
  - 多节点：16×H100（2 节点）
- **训练配置**：
  - Rollout Factor $ N = 32 $
  - Prompt Length $ P = 8K $, Response Length $ R = 2K $
  - `train_batch_size = 128`, `ppo_mini_batch_size = 64`
  - 微批次大小（per-GPU micro-batch size）：4 或 8
  - 使用 FSDP2 + BF16 + gradient checkpointing
- **框架**：veRL（集成 DualKV 补丁）

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Policy Update Latency** | 策略更新阶段（forward + backward）耗时 |
| **End-to-End Step Time** | 整个训练步（含 rollout、log prob、policy update）总时间 |
| **Peak GPU Memory** | 峰值显存占用 |
| **Model FLOPs Utilization (MFU)** | 模型实际利用的 FLOPs 占理论峰值的比例 |
| **Speedup** | 相对于 FA2 基线的加速比 |
| **Validation Accuracy / Reward** | 验证集奖励分数，用于确认收敛一致性 |

### **基线方法对比**
- **FA2 (FlashAttention-2)**：标准实现，作为主要基线。
- **FA3**：Hopper 架构专用内核，用于对比新一代 attention 性能。
- **Prefix Grouper**：近期提出的框架级共享 prompt 优化方法，但仍依赖标准 FA2。
- **FA2 + Ulysses SP**：当 FA2 OOM 时，使用 4-way 序列并行作为可行方案进行比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 内核级性能（Table 1 & Table 2）**
| $ P $ | 方法 | F+B 时间 (ms) | Speedup | 显存下降 |
|-------|------|---------------|---------|----------|
| 4K | FA2 → DualKV | 172.2 → 104.2 | **1.65×** | 63% ↓ |
| 16K | FA2 → DualKV | 1441.0 → 371.7 | **3.88×** | 85% ↓ |
| 32K | FA2 OOM | — | — | — |
| 32K | DualKV | 525.8 | — | — |

> 在 $ P=32K $ 时，FA2 已 OOM，而 DualKV 仍可运行。

#### **(2) 端到端 GRPO 训练（Qwen3-8B, 8×H100）**
| 配置 | Micro-batch | Peak Mem (GB) | Policy Update (s) | Speedup | MFU |
|------|-------------|----------------|--------------------|---------|-----|
| FA2 | 4 | 106 | 520 | 1.00× | 36% |
| DualKV | 4 | 81 | 319 | 1.63× | 59% |
| DualKV | 8 | 93 | 247 | **2.09×** | **76%** |

- **FA2 在 mb=8 时 OOM**，无法运行。
- **MFU 从 36% 提升至 76%**，接近翻倍。
- **支持 2× 更大的 micro-batch**，减少梯度累积步数。

#### **(3) DAPO 训练结果**
| 方法 | Policy Update Speedup | MFU |
|------|------------------------|-----|
| DualKV (mb=8) | **2.47×** | **77%** |

> 由于 DAPO 无需 KL 参考模型前向，策略更新占比更高，因此 DualKV 加速更明显。

#### **(4) 多节点 MoE 训练（Qwen3-30B-A3B, 16×H100）**
| 方法 | SP | Policy Update Speedup | End-to-End Speedup | Step Time |
|------|----|------------------------|----------------------|----------|
| FA2 | 4 | 1.00× | 1.00× | 5284 s |
| DualKV | **1** | **3.82×** | **3.38×** | **1564 s** |

- **DualKV 不需要序列并行（SP=1）即可运行**，而 FA2 至少需要 SP=4 才能避免 OOM。
- 消除了 SP 带来的 all-to-all 通信开销，进一步提升效率。
- **训练时间从 42.6 小时降至 12.6 小时**，节省约 $6K 成本（AWS on-demand）。

#### **(5) 与其他方法对比（Table 2）**
| $ P $ | 方法 | 时间 (ms) | DualKV 相对加速 | 显存 (GB) |
|-------|------|-----------|------------------|------------|
| 8K | FA2 | 549 | — | 50.7 |
| 8K | Prefix Grouper | 465 | — | 50.7 |
| 8K | **DualKV** | **128** | **3.48× vs PG** | **9.9** |

> DualKV 比 Prefix Grouper 快 3.5×，显存低 5×。

---

## **4. 关键结论和发现**

### **主要发现**
1. **共享 prompt 的重复计算是 RL 训练的主要瓶颈**，尤其在 $ N \geq 16, P > 8K $ 场景下。
2. **DualKV 通过内核级优化彻底消除该冗余**，实现：
   - **2–3.8× 策略更新加速**
   - **MFU 从 36% 提升至 76%**
   - **支持 2× 更大的 micro-batch**
   - **消除对序列并行（SP）的依赖**
3. **DualKV 是数学等价的精确实现**，不影响模型收敛性和最终性能。
4. **训练瓶颈已从 policy update 转移至 rollout generation**，为未来优化指明方向。

### **方法的局限性**
- 当前假设每个 micro-batch 组内只有一个共享 prefix。
- 不支持树状或多轮对话中部分重叠的 prefix 结构。
- 依赖于 veRL 等框架修改数据流水线以保证 prompt 分组连续。
- 在短上下文（如 GSM8K）下收益有限（约 1.2×），优势集中在长上下文场景。

### **未来工作方向**
1. **推广至任意树结构**：支持 Tree-of-Thought、多轮对话等具有共享子图的场景。
2. **与 Ulysses SP / Ring Attention 组合**：结合序列并行技术应对超长上下文（>100K）。
3. **适配 FA3 / FA4 及 Blackwell 架构**：进一步挖掘新一代硬件潜力。
4. **集成至主流 RL 框架**：如 OpenRLHF、TRL 等，推动广泛采用。

---

> ✅ **总结一句话**：  
> **DualKV 是首个在训练中实现共享 prompt KV 和全层计算去重的 FlashAttention 内核，使大 rollout、长上下文的 RL 训练速度提升 2–3.8×，MFU 翻倍，并摆脱对序列并行的依赖。**

</details>

---

### 2. [PSD: Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding](https://arxiv.org/abs/2605.15609)

**Authors**: Shengyin Sun, Yiming Li, Renxi Liu, Xinqi Li, Hui-Ling Zhen, Weizhe Lin, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Chen Ma  
**Category**: cs.CL  
**Published**: 2026-05-18  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.15609v1  

#### Abstract
Diffusion large language models (dLLMs) generate text by iteratively denoising masked token sequences. Although dLLMs can predict all masked positions in parallel within each step, the large number of denoising iterations still makes inference expensive. This cost can be reduced spatially by unmaski...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PSD: Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
Diffusion Large Language Models (dLLMs) 虽然通过并行去噪机制在每一步中可同时预测多个被掩码的 token，从而具备天然的并行生成潜力，但其推理过程仍需大量迭代步骤，导致**推理成本高、延迟大**。现有的加速策略存在以下瓶颈：

- **空间并行解码**（如 unmask 多个 token）虽减少迭代次数，但会因低置信度 token 过早提交而导致错误传播，**质量显著下降**。
- **时序推测解码**（speculative decoding）虽能压缩多个去噪步骤，但通常每次只 unmask 一个 token，**并行度受限**。

因此，如何在不牺牲生成质量的前提下，进一步提升 dLLMs 的推理效率，是一个关键挑战。

---

### **提出了什么新方法或新思路**
本文提出 **Parallel Speculative Decoding (PSD)**，一种**无需训练**、**策略无关**（policy-agnostic）的推理加速框架，首次将**空间并行**与**时序推测**两种加速维度**统一整合**于同一解码流程中。

PSD 包含三个阶段：
1. **Spatial Parallel Unmasking**：使用任意转移策略（transfer policy），在每个去噪步中并行揭示多个高置信度 token。
2. **Temporal Speculative Drafting**：利用当前步的置信度排序，构建多深度的推测草案（drafts），无需额外模型调用。
3. **Batched Verification with Hierarchical Acceptance**：通过一次批量前向传播验证所有草案，并采用分层接受机制，保留最深且一致的草案，防止错误传播。

---

### **相比现有方法的优势**
- **双重加速**：同时压缩空间（每步 unmask 更多 token）和时间（多步推测合并验证），实现**复合加速增益**。
- **质量保持**：通过分层验证机制过滤错误推测，避免因激进并行导致的质量崩溃。
- **通用性强**：兼容任意空间并行策略（如 Confidence Thresholding、LocalLeap），可作为“即插即用”模块提升现有方法。
- **无训练开销**：纯推理期优化，无需微调或额外训练。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
在三个下游任务上进行评估，涵盖数学推理与代码生成：
- **GSM8K**：小学数学应用题，测试多步推理能力，报告 **Accuracy**。
- **HumanEval**：Python 函数生成任务，报告 **pass@1**。
- **MBPP**：众包编程任务，报告 **pass@1**。

---

### **实验设置和评估指标**
- **模型**：在三种不同训练范式的开源 dLLMs 上测试：
  - **Dream-v0-Base-7B**（AR 初始化微调）
  - **LLaDA-1.5**（从零开始预训练 + 偏好对齐）
  - **openPangu-7B-Diffusion-Base**（持续预训练 + 块扩散）
- **块大小**（block size）：32
- **最大生成长度**：512
- **评估指标**：
  - **Tokens Per Forward Pass (TPF)**：衡量推理效率，越高越好。
  - **Accuracy / pass@1**：衡量生成质量。

---

### **基线方法对比**
共对比 **7 种代表性基线**：
#### **空间并行方法**（Spatial-only）：
- Greedy Decoding (k=1)
- Confidence Parallel Decoding (T=0.9)
- LocalLeap
- LoPA (k=1,3,5,7)
- ETE (k=1,3,5,7)
- FDM (k=1,3,5,7)

#### **时序推测方法**（Temporal-only）：
- **Spiffy** (d=1,3,5,7)：当前最先进的 dLLM 推测解码方法。

#### **PSD 变体**：
- **PSD (Confidence, d)**：以 Confidence 为骨干的空间策略。
- **PSD (LocalLeap, d)**：以 LocalLeap 为骨干的空间策略。
- 扫描推测深度 $ d \in \{1,3,5,7\} $

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
- **最高可达 5.5× TPF**，即每个模型前向传播平均生成 **5.5 个 token**。
- 在保持与贪婪解码（greedy decoding）相当准确率的情况下（误差 ≤1%），实现 **3–5.5× 加速**。
- 在 GSM8K 上，PSD (Confidence, d=7) 达到 **4.0× TPF** 并保持 **78.9% 准确率**（接近 greedy 的 78.8%）。
- 在 MBPP 上，PSD (Confidence) 达到 **4.9× TPF**，而 Spiffy 仅达 2.0×。

---

### **与基线方法的对比结果**
| 对比维度 | 结果 |
|--------|------|
| **vs. Spiffy (时序-only)** | PSD 在相同质量下实现 **2–2.5× 额外加速**，证明空间并行带来显著增益。 |
| **vs. LoPA/FDM (空间-only)** | 当并行度相当时，LoPA k=7 准确率下降 **8.0 pts**，而 PSD 几乎无损。 |
| **vs. Greedy** | PSD 实现高达 **5.5× TPF**，而 greedy 仅为 1.0×，速度提升显著。 |

> ✅ **PSD 始终位于 Pareto 前沿**：在所有模型 × 任务组合中，PSD 在“质量-速度”权衡曲线上占据右上方优势区域。

---

### **消融实验结果**
- **推测深度 $d$ 的影响**：
  - $d=3$ 即可捕获大部分加速收益，$d>3$ 收益递减（边际效应明显）。
  - 增加 $d$ 不会导致准确率下降，说明 **Hierarchical Acceptance 有效过滤错误分支**。
- **空间策略选择的影响**：
  - **PSD (LocalLeap)**：更高 TPF，适合追求极致速度。
  - **PSD (Confidence)**：更好准确率保持，适合质量敏感场景。
  - 表明 PSD 具有**模块化灵活性**，可根据需求切换骨干策略。

---

## 4. **关键结论和发现**

### **主要发现**
1. **空间与时间加速是互补的**：  
   空间并行（每步 unmask 更多 token）与时序推测（多步合并验证）可**协同增效**，而非相互替代。

2. **置信度排序具有稳定性**：  
   实验表明，当前步的高置信度 token 排序对未来几步仍有较强预测能力，支持多深度草案构造。

3. **代码生成更敏感于并行度**：  
   相比数学推理，代码生成对激进并行更脆弱（语法约束强），但 PSD 的验证机制能有效缓解此问题。

4. **PSD 是通用加速骨架**：  
   可无缝集成任何空间策略，形成“空间策略 + 时序推测”的复合加速方案。

---

### **方法的局限性**
- **依赖客观评测任务**：实验集中在 GSM8K、HumanEval 等可自动验证的任务，未评估开放生成（如创意写作）中的表现。
- **推测深度存在饱和**：$d>3$ 后加速增益趋于平缓，受限于验证通过率的指数衰减。
- **超参数需适配**：尽管默认配置泛化良好，但在特定领域可能需要调参以达到最优权衡。

---

### **未来工作方向**
- 将 PSD 扩展至**多模态扩散模型**（如 DALL-E 类模型）。
- 探索**动态调整推测深度**（adaptive $d$）以应对不同上下文复杂度。
- 研究在**长文本生成**中如何跨块传递推测状态，进一步提升端到端吞吐。
- 结合 **KV Cache 优化** 与 **early stopping**，实现系统级推理加速。

---

> **总结**：PSD 提出了一种新颖的“双轴加速”范式，通过将**空间并行**与**时序推测**有机结合，在几乎不损失质量的前提下，将 dLLM 的推理效率推向新的 Pareto 前沿，为高效大模型部署提供了实用且通用的解决方案。

</details>

---

### 3. [Going Beyond the Edge: Distributed Inference of Transformer Models on Ultra-Low-Power Wireless Devices](https://arxiv.org/abs/2605.15694)

**Authors**: Alexander Gr\"afe, Ding Huo, Johannes Berger, Marco Zimmerling, Sebastian Trimpe  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.15694v1  

#### Abstract
Transformer models are rapidly becoming a cornerstone of modern Internet of Things (IoT) applications, yet their computational and memory demands far exceed the capabilities of a single typical ultra-low-power IoT device. We present CATS, a framework for distributed transformer inference on ultra-lo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Going Beyond the Edge: Distributed Inference of Transformer Models on Ultra-Low-Power Wireless Devices*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 Transformer 模型在 IoT 场景中应用广泛，但其高计算和内存开销远超单个 **ultra-low-power wireless device**（如基于 MCU 的传感器节点）的能力。现有分布式推理（DTI）方法多依赖高性能设备（如 Raspberry Pi）和有线通信，无法适用于资源极度受限、通信低带宽且不可靠的无线传感网络。

本文针对以下三大挑战提出解决方案：
- **C1：Limited Computing Power** — MCU 内存（RAM/Flash）和算力严重受限。
- **C2：Limited Communication Bandwidth** — 低功耗无线链路带宽低、延迟高，通信成为瓶颈。
- **C3：Unreliable Communication** — 无线通信易丢包，影响中间激活值传输，导致精度下降。

---

### 🚀 提出的新方法：CATS 框架
作者提出了 **CATS**（Collaborative Inference at the Sensor-level），一个专为 ultra-low-power wireless 设备设计的通信感知分布式 Transformer 推理框架，核心创新包括：

#### （1）**SomeGather** — 新的剪枝型通信原语
- 是对传统 **AllGather** 的改进，通过**选择性广播部分 activation columns** 来减少通信量。
- 在通信过程中“剪枝”整个特征列（而非单个元素），未被广播的列仅在本地处理。
- 同时降低：
  - 通信带宽（减少 up to 90%）
  - 每设备 RAM 占用（减少 67.5%）
  - Flash 存储压力（因权重分片存储）

> 💡 创新之处：首次将通信剪枝机制引入 Transformer 分布式推理，并使其与模型并行化协同优化。

#### （2）**基于 SomeGather 的模型划分策略**
- 将 Multi-Head Attention 沿 **feature dimension** 划分，不同设备负责不同的 attention heads。
- 所有跨设备通信均通过 **SomeGather** 实现，避免使用 AllReduce/ReduceScatter 等在 mesh 网络中效率低下的原语。
- 支持任意拓扑的 mesh 网络，无需固定分组结构。

#### （3）**Message Dropout (MD)** — 面向丢包鲁棒性的训练机制
- 在训练阶段随机模拟三种现实中的消息丢失模式：
  - 单设备收不到某条消息
  - 单设备完全失联
  - 单设备发送的消息全网未接收
- 使用动态掩码模拟这些情况，使模型在推理时对丢包具有鲁棒性。

#### （4）利用 **Mixer 协议** 实现高效无线广播
- 基于 **random linear network coding** 和同步传输，在动态 mesh 网络中实现低延迟、高可靠性的 all-to-all 通信。
- 抽象掉底层 mesh 复杂性，支持灵活部署和移动节点。

---

### 🔍 相比现有方法的优势

| 特性 | 现有方法（如 Voltage, Astra, SiracusaDTI） | CATS |
|------|------------------------------------------|------|
| 减少 RAM？ | 部分支持（如 Astra） | ✅ 显著减少（+ SomeGather 剪枝） |
| 减少 Flash？ | 部分支持（如 SiracusaDTI） | ✅ 显著减少（权重分片 + 剪枝） |
| 通信效率 | 依赖 AllReduce，mesh 下效率差 | ✅ 使用 SomeGather，大幅降低通信量 |
| 丢包鲁棒性 | 假设无损通信 | ✅ 引入 MD，显式建模丢包 |
| 硬件适用性 | 高性能边缘设备 | ✅ 支持 ultra-low-power MCU（nRF52840） |
| 实际部署验证 | 多为仿真或虚拟机 | ✅ 真实硬件测试（16 节点 BLE mesh） |

> ⭐ **核心优势**：**首次实现 ultra-low-power wireless 设备上的端到端 DTI**，突破“边缘之下”的能力边界。

---

## 2. 核心实验方法和设置

### 📚 数据集
使用四个时间序列预测任务的数据集进行评估：
- **ETT-h2**
- **ICD**
- **London-smart-meters**
- **Traffic**

> 注：实验以 Time-Series Transformer 为例，但方法通用。

### 🧪 模型配置
- 6 层 Transformer
- Feature dimension: 128
- 残差块含 1 个隐藏层
- 使用 8-bit kernel（CMSIS-NN）加速 MCU 上的计算

### 🖥️ 实验平台
#### （1）真实硬件测试床（Testbed）
- **设备数量**：最多 16 台 **nRF52840**（64 MHz Cortex-M4, 1MB Flash, 256KB RAM）
- **通信方式**：基于 BLE PHY 的 **Mixer 协议**
- **部署环境**：大学走廊，形成至少两跳的 mesh 网络

#### （2）仿真实验
- 补充硬件无法覆盖的大规模配置空间
- 使用相同模型结构和训练流程

### 📊 评估指标
| 问题 | 指标 |
|------|------|
| Q1: Memory Scaling | RAM usage, Flash footprint（分析法获取） |
| Q2: Latency Scaling | Attention / Residual block 推理延迟（实测） |
| Q3: Accuracy vs. Pruning | MSE（均方误差） vs. 通信节省比例 |
| Q4: Robustness to Loss | 不同丢包率下的测试误差变化 |

### 🆚 基线方法对比
由于无直接可比端到端方案，采用分问题基线：
- **Q1 对比**：Voltage [Hu and Li, 2024], Astra [Liu et al., 2025b], SiracusaDTI [Bochem et al., 2025]
- **Q2 对比**：单设备集中式执行（central execution）
- **Q3 对比**：Normal pruning（普通剪枝）、AllGather
- **Q4 对比**：是否启用 Message Dropout

---

## 3. 主要实验结果和性能指标

### 📈 Q1: Memory Scaling 结果
- **CATS 同时降低 RAM 和 Flash**，而其他方法只能优化其一：
  - Voltage/Astra：降低 RAM，但 Flash 不变（权重未分发）
  - SiracusaDTI：降低 Flash，但 RAM 随设备数线性增长（AllReduce 开销大）
- **SomeGather 剪枝显著扩展支持的模型规模**：
  - 图 5 显示，随着剪枝率提升（0% → 90%），支持的 token 数 $N$ 显著增加，突破 RAM 瓶颈。

### ⏱️ Q2: Latency Scaling 结果（图 6）
- **非剪枝情况下**：
  - 通信延迟主导运行时间（占 69.42% ~ 91.4%）
  - 仍能实现加速：attention block 最高提速 **2.51×**（512 features）
- **启用 90% SomeGather 剪枝后**：
  - 通信延迟降低 **up to 80%**
  - attention block 加速达 **4.37×**
  - residual block 加速达 **4.68×**
  - 总体推理延迟相比单设备降低 **3.8× ~ 10.96×**

### 🎯 Q3: SomeGather 准确性保持（图 7）
- **SomeGather**：即使通信节省高达 90%，预测误差基本不变。
- **Normal pruning**：通信减少时，误差迅速上升。
> ✅ 证明 **SomeGather 在大幅压缩通信的同时不牺牲模型精度**。

### 🛡️ Q4: Message Dropout 鲁棒性（图 8）
- 无 MD 的模型：10% 丢包时误差上升 **up to 200%**
- 启用 MD 的模型：在匹配训练丢包率下表现最优
  - 例如训练时使用 10% MD，则在 10% 丢包下性能最佳
  - 相比无 MD，相对误差增幅从 200% 降至 **23.6%**
> ✅ MD 显著增强模型对无线丢包的鲁棒性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DTI 可以前所未有地延伸至 ultra-low-power wireless 设备**，CATS 首次实现了这一目标。
2. **SomeGather** 是关键创新，它通过列级剪枝同时优化通信、RAM 和计算负载。
3. **联合设计 partitioning + communication + training** 是解决资源与通信双重约束的有效范式。
4. **MD 训练机制** 能有效应对无线网络的不确定性，提升实际部署稳定性。
5. 实验表明 CATS 可支持 **14× 更大的 Transformer 模型**（如 1400 万参数 ViT），远超单设备容量。

### ⚠️ 方法的局限性
1. **设备数受 attention head 数限制**：每个 head 当前映射到单一设备，若 head 数少则无法扩展更多设备。
2. **配置固化**：SomeGather 剪枝率和 MD 丢包率需在训练时设定，难以适应多变的实际部署场景。
3. **未支持 decoder-only 或生成任务**：目前聚焦 encoder-only 架构（如 ViT、Time-Series Transformer）。

### 🔮 未来工作方向
1. **将单个 attention head 分布到多个设备**，打破 head 数对设备数的上限。
2. **开发无需重训练的通信适配机制**，使预训练模型可动态适配不同网络配置。
3. **扩展至 LLMs 和生成式任务**，探索 CATS 在更大模型上的潜力。
4. **支持异构设备混合部署**，提升系统灵活性和容错能力。

---

> 🏁 **总结一句话**：  
> **CATS 成功将大型 Transformer 模型的分布式推理推进到了“Beyond the Edge”的 ultra-low-power wireless 设备层级，通过 SomeGather 与 MD 的协同设计，在极低资源下实现了高效、鲁棒、可扩展的智能感知。**

</details>

---

### 4. [Response-Conditioned Parallel-to-Sequential Orchestration for Multi-Agent Systems](https://arxiv.org/abs/2605.15573)

**Authors**: Nurbek Tastan, Alex Iacob, Lorenzo Sani, Meghdad Kurmanji, Nicholas D. Lane, Samuel Horvath, Karthik Nandakumar  
**Category**: cs.CL  
**Published**: 2026-05-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.15573v1  

#### Abstract
Multi-agent systems can solve complex tasks through collaboration between multiple Large Language Model agents. Existing collaboration frameworks typically operate in either a parallel or a sequential mode. In the parallel mode, agents respond independently to queries followed by aggregation of resp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Response-Conditioned Parallel-to-Sequential Orchestration for Multi-Agent Systems

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **multi-agent LLM systems** 主要采用两种协作模式：
- **Parallel 模式**：所有 agent 并行生成响应，通过投票、自洽性（self-consistency）等方式聚合结果。优点是简单可扩展，但计算开销大、冗余高，无法利用 agent 间的互补信息进行纠错。
- **Sequential 模式**：agent 按照预设拓扑（如链状、树状）逐步传递信息。能支持错误修正和信息流动，但需要手动设计或学习通信拓扑，带来额外的 token 开销、优化复杂性和泛化困难。

这两种模式在实践中被当作互斥的设计选择，缺乏一种**动态决策机制**来判断：**何时应保持并行？何时应启动顺序传播？**

---

### 提出的新方法：NEXA
本文提出 **NEXA**（Nexus-based Execution Architecture），一种**可训练的响应条件策略**（response-conditioned policy），实现从并行到顺序执行的**混合范式切换**。

#### 核心思路
1. **第一阶段：并行草案生成**
   - 所有 agent 独立对查询 $Q$ 生成初始响应 $R^{(0)}_n = A_n(Q)$。
2. **第二阶段：语义嵌入与图预测**
   - 将各响应通过轻量级编码器（如 `all-MiniLM-L6-v2`）映射为语义向量 $r_n$。
   - 使用一个轻量 **Transformer 模型**作为策略网络，基于这些向量预测一个稀疏的有向无环图（DAG）$\mathcal{G}$。
     - 若 $\mathcal{G} = \emptyset$，系统停留在并行模式，直接聚合输出。
     - 若 $\mathcal{G} \neq \emptyset$，则执行一轮**顺序消息传播**，部分 agent 根据上游节点更新其响应。
3. **第三阶段：无裁判聚合**
   - 使用基于贡献度加权的质心选择法（weighted-centroid-based aggregation）选出最终答案，无需外部 LLM judge。

#### 创新点
- ✅ **统一并行与顺序范式**：将“是否需要顺序通信”建模为一个由响应内容驱动的决策问题，而非固定架构选择。
- ✅ **构建即无环图（Acyclicity by Construction）**：通过按 agent 贡献度排序定义拓扑顺序，只允许前向边（higher → lower contribution），确保图天然为 DAG，避免后处理修复。
- ✅ **轻量化且可迁移的策略**：策略仅依赖响应语义，不依赖 agent 身份、角色标签或模型家族，具备跨配置泛化能力。
- ✅ **端到端可训练**：使用 **REINFORCE + batch-mean baseline** 进行 policy-gradient 优化，奖励函数结合任务正确率与边数惩罚（sparsity regularization）。

---

### 相比现有方法的优势
| 方面 | NEXA | 传统方法 |
|------|------|--------|
| **通信效率** | 动态决定是否通信，多数情况下选择低边数稀疏图 | 固定拓扑或全连接，通信开销高 |
| **泛化性** | 可迁移到不同 agent 数量、任务、模型规模/代际 | 多数需重新训练或微调 |
| **系统简洁性** | 无需外部 judge、reward model 或复杂的拓扑搜索 | 依赖额外模块增加复杂性 |
| **理论保障** | 显式保证 DAG 合法性、置换不变性、包含纯并行作为特例 | 缺乏形式化性质保证 |

---

## 2. 核心实验方法和设置

### 数据集
- **AQUA-RAT**：数学应用题推理数据集
- **GSM8K**：小学数学文字题，强调多步推理
- **HumanEval**：编程任务，评估代码生成能力
- （附录中还测试了 GSM-Hard、MMLU）

### 实验设置
- **基础训练配置**：
  - Agent 模型：`Qwen2.5-1.5B-Instruct`
  - Agent 数量 $N = 10$
  - 任务：AQUA-RAT 和 GSM8K
- **策略模型架构**：
  - 单层单头 Transformer encoder
  - 边缘构造方式：$ \mathbf{A} = \mathbf{H}\mathbf{H}^T $
- **训练细节**：
  - 优化器：Adam
  - Batch size: 32
  - 学习率：0.1
  - Dropout: 0.3
  - Sparsity coefficient $\lambda_{sp} = 0.1$
  - 更新次数：50次

### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 最终输出的任务正确率（mean ± std） |
| **Average Rating** | 排名评分（越低越好） |
| **Token Usage** | 总消耗 token 数（prompt + completion），衡量通信与推理成本 |
| **Mean Edge Count** | 预测通信图的平均边数，反映通信密度 |

### 基线方法对比
| 基线 | 类型 |
|------|------|
| Single | 单 agent |
| CoT | Chain-of-Thought 提示 |
| SC (Self-Consistency) | 自洽性采样 |
| SelfOrg* | 基于语义贡献的自组织通信（带一次顺序传播） |
| GPTSwarm | 图结构优化的 agent 网络 |
| AgentPrune | 剪枝不必要的通信路径 |
| G-Designer | 使用 GNN 学习通信拓扑 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）
| Method | AQUA-RAT | HumanEval | GSM8K | Avg. Acc. | Avg. Rating | Token Usage |
|--------|----------|-----------|-------|------------|--------------|-------------|
| Single | 52.62 | 50.41 | 70.07 | 57.70 | 5.67 | 1.15M |
| CoT | 54.46 | 45.93 | 70.47 | 56.95 | 5.33 | 1.30M |
| SC | 56.82 | 12.52 | 71.53 | 46.96 | 5.00 | 11.22M |
| SelfOrg* | 56.46 | 52.03 | 72.60 | 60.36 | 2.67 | 28.41M |
| GPTSwarm | 55.91 | 36.79 | 69.33 | 54.01 | 6.33 | 38.02M |
| NEXA (**Ours**) | **57.74** | **51.42** | **73.53** | **60.90** | **1.33** | **18.36M** |

> 🔍 **结论**：
> - NEXA 在 **平均准确率上达到 60.90%**，优于所有基线（包括 SelfOrg* 的 60.36%）。
> - **Token 消耗仅为 18.36M**，比 SelfOrg* 减少约 35%，比 GPTSwarm 减少超 50%。
> - **平均评分为 1.33**，显著优于其他方法，表明其综合表现最优。

---

### 泛化性实验结果

#### （1）Agent 数量迁移（Figure 2）
- 训练于 $N=10$，测试于 $N \in \{5,10,15,20\}$
- 结果：在所有规模下均优于 Single 和 CoT，**峰值出现在 $N=15$**，说明策略能有效利用更多候选响应。

#### （2）跨任务迁移（Figure 3）
- 在 AQUA-RAT 上训练，在 GSM8K 上测试（反之亦然）
- **转移间隙极小**（<0.2 pts），表明学到的是通用的“响应条件规则”，而非任务特定模式。

#### （3）模型尺度迁移（Figure 4）
- 使用 `Qwen2.5-1.5B` 训练，迁移到 `Qwen2.5-7B` 测试
- 准确率几乎持平（GSM8K: 90.48 vs 90.52；AQUA-RAT: 76.98 vs 77.40），说明策略不依赖训练时的模型能力水平。

#### （4）模型代际迁移（Figure 5）
- 在 `Qwen2.5-1.5B` 上训练，在 `Qwen3.5-2B` 上测试
- 仅落后目标代际策略 **0.17 分**（77.40 vs 77.73），验证了策略在模型升级后仍可用，减少重训练需求。

---

### 消融实验结果

#### （1）Backbone Ablation（Table 2）
| Backbone | Accuracy (GSM8K) |
|---------|------------------|
| Transformer | 73.53 ± 0.23 |
| GNN | 73.41 ± 0.31 |

> ✅ 表明性能提升主要来自 **NEXA 的框架设计**，而非特定神经网络结构。

#### （2）Policy Optimization Ablation（Table 3）
| Optimizer | AQUA→AQUA | GSM8K→AQUA |
|----------|-----------|------------|
| PG (ours) | **57.74** | **57.56** |
| GRPO     | 57.56     | 57.48      |

> ✅ Policy Gradient 略优，但差距小，说明策略对优化方式不敏感。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **响应条件决策优于静态架构**：是否启动顺序通信应基于实际响应内容决定，而非预设。
2. ✅ **稀疏通信足够有效**：NEXA 经常选择低边数通信计划（见 Figure 8），无需密集交互即可提升性能。
3. ✅ **高泛化性**：同一策略可在不同 agent 数量、任务、模型规模/代际间复用，体现其作为“通用协调器”的潜力。
4. ✅ **正向修改为主**：分析显示 NEXA 主要用于“救援”错误答案（rescue rate 达 19–24%），同时保持正确答案（preservation > 97%），极少造成伤害（harm < 2.5%）。

---

### 局限性（Limitations）
1. **评估集中在判别性任务**：目前实验集中于数学推理和编程等可自动验证的任务，尚未覆盖开放生成、工具调用或多轮规划等场景。
2. **依赖嵌入质量**：若 embedding model 无法区分语义差异，则贡献排序和图预测可能失效。
3. **固定初始 agent 数量**：未结合 adaptive agent selection，初始草案池大小仍是效率瓶颈。

---

### 未来工作方向
- 结合 **adaptive agent sampling**，动态决定初始参与 agent 数量。
- 扩展至 **multi-round communication** 场景，研究迭代 refinement 的稳定性。
- 探索在 **real-world interactive environments** 中的应用，如机器人协作、智能体辩论等。
- 引入 **learned stopping criteria**，自动判断是否终止传播过程。

---

> 📌 **一句话总结**：  
> NEXA 提出了一种**响应驱动的混合执行机制**，通过轻量级策略动态决定是否从并行转向顺序通信，在保持高准确率的同时大幅降低通信开销，并展现出强大的跨配置泛化能力，为 multi-agent LLM 系统提供了更高效、灵活的协作范式。

</details>

---

### 5. [Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered](https://arxiv.org/abs/2605.15622)

**Authors**: Sijia Liu, Yicheng Lang, Soumyadeep Pal, Changsheng Wang, Yancheng Huang, Chongyu Fan, James Diffenderfer, Bhavya Kailkhura, Yihua Zhang  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.15622v1  

#### Abstract
Zeroth-order (ZO) optimization, learning from finite differences of function evaluations without backpropagation, has recently regained attention in deep learning due to its memory efficiency and applicability to gray- or black-box pipelines. Yet, ZO methods are often dismissed as fundamentally unsc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**零阶优化**（Zeroth-Order, ZO）在深度学习中被普遍认为“不可扩展”（unscalable）的悲观观点提出挑战。社区普遍认为 ZO 方法因梯度估计方差高、查询复杂度大而难以用于大规模模型训练。然而，作者指出这一结论可能源于对 ZO 的**探索不足**（underexplored），而非其本身能力不足（underpowered）。

具体而言，当前 ZO 研究存在以下问题：
- 过度关注全参数空间的梯度估计器设计（estimator-centric design）
- 忽视系统级优势（如通信效率、前向执行简化）
- 评估方式混淆了任务难度与优化能力（如依赖 task alignment）

### 提出了什么新方法或新思路
论文并未提出单一的新算法，而是从**算法-系统-评估**三个层面提出了六个核心立场（Positions, P1–P6），构成一套重新思考 ZO 优化的框架：

| 立场 | 内容 |
|------|------|
| **P1**: Random direction choice matters | 强调扰动方向的设计（如稀疏化、预条件）对控制方差至关重要 |
| **P2**: Query-variance tradeoff | 方差降低必须与查询成本权衡，避免误导性可扩展性判断 |
| **P3**: Directional-derivative viewpoint | 应将方向导数（Directional Derivative, DD）作为必须考虑的基准 |
| **P4**: Subspace & spectral view | 在低维子空间进行 ZO 优化可实现可解释的方差缩减与优雅的查询扩展 |
| **P5**: Simplicity as a systems advantage | ZO 的前向执行特性带来通信高效、并行友好、适合资源受限环境的系统优势 |
| **P6**: De-obfuscation from task alignment | 需剥离 task alignment 对性能的影响，以真实评估 ZO 的优化能力 |

此外，作者进一步提炼出五项行动号召（Call to Actions）：
- **(A1)** 建立严谨、去混淆的评估协议
- **(A2)** 超越全空间优化，拥抱子空间与谱方法
- **(A3)** 利用生成模型（如 Diffusion Models）进行梯度估计
- **(A4)** 构建 ZO-native 系统栈（如通信轻量、无反向流水线）
- **(A5)** 拓展 ZO 在量子计算、科学模拟等新兴领域的应用

### 相比现有方法的优势
- **更全面的视角**：超越“仅改进估计器”的狭隘路径，强调系统协同设计与评估去混淆。
- **更强的实用性**：揭示 ZO 在内存受限、硬件异构、隐私敏感场景下的天然优势。
- **更高的潜力**：通过子空间、谱方法、系统优化等手段，显著缓解传统认为的“维度灾难”。

---

## 2. 核心实验方法和设置

本论文为一篇**立场论文**（position paper），不提出单一新算法，因此未报告完整的新实验，而是基于已有研究和图示分析支持其论点。

### 使用了哪些数据集
- 主要引用已有工作的实验结果，涉及以下任务和模型：
  - **语言模型微调**：Gemma2-2B 在 **SST-2**, **RTE**, **WiC** 等 GLUE/SuperGLUE 下游任务上的表现
  - **大型语言模型**（LLMs）：如 Gemma、LLaMA 等的 fine-tuning 场景
  - **图像模型**：部分引用 CV 领域的 ZO 应用（如对抗攻击）

### 实验设置和评估指标
- **评估指标**：
  - 微调准确率（Fine-tuning Accuracy）
  - 查询次数（Query Budget）
  - 通信开销（Communication Cost）
  - 内存占用（Memory Footprint）
  - 是否使用 task alignment
- **关键对比维度**：
  - 有无 task alignment 的性能差异
  - 不同 ZO 方法在相同 query budget 下的表现
  - 与 forward-gradient（方向梯度）方法的对比

### 基线方法对比
论文对比了多种代表性 ZO 方法，包括：
- **MeZO**（Malladi et al., 2023）：基础 ZO 微调方法
- **Sparse-MeZO**（Liu et al., 2025b）：稀疏扰动版本
- **HiZOO**（Zhao et al., 2025b）：结合预条件的加速 ZO
- **LOZO**（Chen et al., 2025）：低秩结构 ZO
- **Forward-gradient / Directional Descent**：作为理想基准（P3）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
论文通过 **Figure 2** 展示了关键实验结果，比较不同 ZO 方法在 **task-aligned**（w/ align）与 **non-aligned**（w/o align）设置下的微调准确率：

| 方法 | SST-2 (w/ align) | SST-2 (w/o align) | RTE (w/ align) | RTE (w/o align) | WiC (w/ align) | WiC (w/o align) |
|------|------------------|-------------------|----------------|------------------|----------------|------------------|
| MeZO | 91.7             | 86.2              | 53.0           | 47.7             | 57.1           | 50.0             |
| Sparse-MeZO | 92.7       | 86.5              | 54.2           | 47.3             | 55.8           | 47.9             |
| HiZOO | 92.1            | 87.6              | 56.3           | 47.7             | 58.3           | 50.0             |
| LOZO  | 92.2            | 87.6              | 56.3           | 47.3             | 58.3           | 50.0             |

> 数据来源：Figure 2，微调 Gemma2-2B 模型

### 与基线方法的对比结果
- 所有 ZO 方法在 **无 task alignment** 时均出现显著性能下降（平均下降约 5–6 个百分点），表明当前成功高度依赖任务对齐。
- 方法间的相对排名在两种设置下发生变化，说明 task alignment 可能掩盖了 ZO 方法的真实优化能力差异。
- **forward-gradient 方法**（虽未直接列出数值）被强调为应作为上界参考，因其利用更丰富的方向信息。

### 消融实验结果（如有）
- 论文虽未进行传统消融实验，但通过 **Table 1** 对近期 ZO 工作进行了系统性分析，揭示：
  - 几乎所有工作都关注 **P1**（方差控制）
  - 少数工作满足 **P2**（查询-方差权衡）
  - **P3**（方向导数基准）和 **P6**（去混淆评估）在绝大多数工作中缺失
- 此分析构成了一种“元消融”，说明当前研究范式的偏颇。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **ZO 并非天生不可扩展**：其 perceived limitations 多源于短视的设计实践，而非本质缺陷。
2. **子空间优化是突破口**：通过在低维子空间进行 ZO（P4），可实现 $ O(r) $ 而非 $ O(d) $ 的方差增长，显著提升可扩展性。
3. **系统优势被严重低估**：ZO 的前向执行特性使其在通信效率、流水线调度、隐私保护等方面具有天然优势（P5）。
4. **task alignment 是一种隐式混淆**：当前许多“成功”可能归功于任务简化，而非 ZO 本身的强大（P6）。
5. **需要新的评估范式**：必须引入固定查询预算、forward-gradient 基准、有无 alignment 对比等去混淆评估协议（A1）。

### 方法的局限性
- **理论与工程脱节**：现有 ZO 算法多在 FO-centric 系统（如 DeepSpeed、Megatron-LM）上运行，未能发挥其系统优势。
- **缺乏统一框架**：尚无成熟的 ZO-native 训练基础设施。
- **任务对齐依赖性强**：在无法对齐的任务上，ZO 性能仍显著落后于 FO 方法。
- **生成式 ZO 仍属设想**：如用 Diffusion Model 学习梯度估计（A3）尚未实现。

### 未来工作方向
1. **构建 ZO-native 系统**：开发专为 ZO 设计的分布式训练框架，支持轻量通信、密集流水线、查询并行。
2. **发展子空间与谱方法**：结合低秩结构、Hessian 信息、MuON 等技术，提升估计效率。
3. **推广去混淆评估标准**：在社区中推动 A1 所述的评估协议。
4. **探索新兴应用场景**：
   - **量子机器学习**：ZO 天然适配量子测量模型
   - **科学计算**：用于物理模拟器、数字孪生等非可微系统
   - **绿色 AI**：在低端 GPU 集群上训练大模型，降低能耗
5. **融合 FO-ZO 混合范式**：将 ZO 视为优化连续体的一部分，而非孤立方法。

---

> **总结**：本文从根本上重塑了对 ZO 优化的认知——它不是 backpropagation 的劣质替代品，而是一种具有独特算法、系统与应用优势的**正交范式**。只要跳出“估计器中心主义”，ZO 完全有可能成为大规模、资源高效、混合系统学习的重要支柱。

</details>

---

### 6. [Measuring Maximum Activations in Open Large Language Models](https://arxiv.org/abs/2605.15572)

**Authors**: Luxuan Chen, Han Tian, Xinran Chen, Rui Kong, Fang Wang, Jiamin Chen, Yuchen Li, Jiashu Zhao, Shuaiqiang Wang, Haoyi Xiong, Dawei Yin  
**Category**: cs.CL  
**Published**: 2026-05-18  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.15572v1  

#### Abstract
The dynamic range of activations is a first-order constraint for low-bit quantization, activation scaling, and stable LLM inference. Prior work characterized outlier features and massive activations on pre-2024 LLaMA-style models, and the downstream activation-quantization stack inherits that pictur...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Measuring Maximum Activations in Open Large Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于**现代开源大语言模型（LLM）中激活值的最大幅度（maximum activation magnitude）**这一关键但被忽视的问题。尽管低比特量化（low-bit quantization）、激活缩放（activation scaling）和稳定推理依赖于对激活动态范围的理解，但此前研究多集中于早期 LLaMA 风格模型，未能系统地重新审视后 LLaMA 时代多样化的开源模型家族。

具体而言，作者提出并回答以下部署导向的核心问题：
- 现代开源 LLM 中的激活值最大能达到多少？
- 这种最大激活幅度如何随模型家族、架构、训练阶段等因素变化？

### 提出了什么新方法或新思路
- **统一测量协议（Unified Pipeline）**：构建了一个标准化的评估流程，在相同的 5,000 样本多领域语料库上，使用家族特定的 tokenizer 和一致的 PyTorch hooks，采集从 embedding 到 final norm 的六类关键组件（如 hidden states、attention outputs、MLP/MoE outputs、SwiGLU gates）的激活统计量。
- **连续型最大激活度量 $M = \max{|a|}$**：将“大规模激活”从一个二元分类概念（如是否满足 $|x| > 100$ 且局部稀疏）转变为一个**连续的、可发布的模型属性 $M$**，更直接服务于量化部署需求。
- **跨家族、跨架构、跨训练阶段的匹配对比设计**：首次在统一框架下进行五组精细控制的对比实验（如 MoE vs. Dense、Base vs. Instruct），揭示不同因素对峰值激活的影响。

### 相比现有方法的优势
| 方面 | 传统做法 | 本文优势 |
|------|----------|---------|
| **研究对象** | 多为 LLaMA/OPT/BLOOM 等少数闭源或早期模型 | 覆盖 **8 个现代开源家族**（Qwen、Gemma、Ling、GPT-OSS 等），含 MoE、视觉语言、指令微调等变体 |
| **分析粒度** | 定性判断是否存在 outlier 或 massive activation | 提供**精确的全局与逐层最大值 $M$**，支持定量比较 |
| **部署相关性** | 将 extreme activation 视为需消除的障碍 | 明确指出 $M$ 是影响 scale selection 和 reconstruction error 的**关键先验风险指标**，应作为 model card 的一部分发布 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **RedPajama 子集构建的 5,000 样本多领域语料库**，涵盖：
  - 数学/科学（850）
  - 编程代码（850）
  - 英文网页文本（850）
  - 知识/QA/书籍（850）
  - 中文（400）
  - 低资源语言（300）
  - 混合英文网页（900）
- 控制序列长度分布：以 4096 token 为主（93%），辅以少量短序列（256–2048），平均长度约 3899 tokens。
- 所有模型使用其原生 tokenizer 对相同原始文本进行分词，避免 tokenizer 差异带来的偏差。

### 实验设置和评估指标
- **模型集合**：共 **27 个 checkpoint**，主分析包含 **24 个来自 8 个家族的模型**（见 Table 2），额外加入 3 个 Qwen2.5-Instruct 模型用于 SFT 分析。
- **前向推理方式**：冻结权重，仅运行 forward pass，通过 **PyTorch hooks** 流式收集每层的激活张量。
- **记录的统计量**：每层每组件记录均值、标准差、RMS、绝对均值、最大/最小值及绝对值分位数估计。
- **核心评估指标**：
  - 主要指标：**全局最大激活幅度 $M = \max{|a|}$**（跨所有层和组件）
  - 辅助指标：逐层最大值轨迹、载体组件（carrier component）、Sun criterion 是否通过（$|x_i| > 100$ 且 $|x_i|/\text{median} \geq 1000$）

### 基线方法对比
本文并非提出新的量化算法，因此不与 AWQ、GPTQ、SmoothQuant 等方法进行端到端性能对比。而是：
- **继承并挑战已有假设**：质疑“outlier 特征普遍存在且模式单一”的旧范式；
- **提供基础数据支持**：为这些量化方法中的 scale selection 提供实证依据；
- **补充量化文献空白**：现有量化工作大多未报告其测试模型的真实 $M$ 值，本文填补此空缺。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 发现 | 数据/倍数关系 | 示例/备注 |
|------|----------------|-----------|
| **跨家族差异巨大** | 全局最大激活 $M$ 跨越近 **四个数量级** | Qwen3.5 系列：~10²–10³；Gemma3-27B-it：~7×10⁵ |
| **MoE 显著抑制峰值** | MoE 模型峰值比同规模 Dense 模型低 **14.0–23.4×** | Qwen3-30B-A3B vs. Qwen3-32B：1,512 vs. 35,328（↓23.4×） |
| **残差流主导极端值传播** | 在 24 个主分析 checkpoint 中，**22 个的全局最大值出现在 layerwise hidden states（即 residual stream）** | GPT-OSS-20B 例外（MLP 输出），Qwen3.5-0.8B 例外（final LayerNorm） |
| **训练进度提升峰值** | Ling-mini 系列随训练步数增加（5T → 20T tokens），$M$ 单调上升 **1.34×**（7,648 → 10,240） | 表明训练阶段本身会影响激活范围 |
| **SFT 压缩晚期峰值** | Qwen2.5-32B-Instruct 相比 Base 版本，全局峰值下降 **1.4×**（30,848 → 22,144），主要源于最后几层压缩 |
| **世代演化非单调** | 不同家族趋势相反：<br>- Qwen：2.5 → 3 ↑，3 → 3.5 ↓<br>- Gemma：2 → 3 ↑↑ | 打破“新一代模型自动优化激活”的假设 |

### 与基线方法的对比结果
无传统意义上的 SOTA 性能对比，但提供了对现有量化方法的前提验证：
- **INT-8 重建误差验证**：在一个轻量级 INT-8 量化 sanity check 中发现：
  - 使用 max-abs scaling 时，高 $M$ 模型（如 Gemma3-4B-it）SQNR 仅为 ~13.5 dB；
  - 低 $M$ 模型（如 Qwen3.5-0.8B）可达 **29.1 dB**；
  - 使用 99.9% clipping 后，多数模型 SQNR 下降至 <1 dB，表明极端值严重干扰普通激活表示。
- 结论：**测得的 $M$ 与低比特重建误差显著相关**，证明其作为部署风险指标的有效性。

### 消融实验结果
本文通过**匹配设计对比**实现类似消融的效果：
- **MoE vs. Dense**（控制家族与参数量）→ 峰值 ↓14–23.4×
- **Vision-Language vs. Text-only**（Qwen2.5-VL vs. Qwen2.5）→ 峰值 ↓1.4–1.6×
- **Base vs. Instruct**（Qwen2.5 系列）→ 仅 32B 时显著 ↓1.4×，小模型无明显变化
- **训练阶段演进**（Ling-mini）→ 随训练步数增加，$M$ ↑1.34×

---

## 4. 关键结论和发现

### 论文的主要发现
1. **最大激活幅度 $M$ 是一个独立的模型属性**，不能仅由参数量预测，而受**模型家族、架构（如 MoE）、训练阶段、模态适配和指令微调**共同决定。
2. **残差流是极端激活的主要载体**，建议量化方案优先关注 hidden state 的峰值而非 attention 或 MLP 输出。
3. **MoE 架构天然具有更低的激活峰值**，可能因其稀疏路由机制分散了激活能量。
4. **指令微调（SFT）倾向于压缩后期层的激活峰值**，但不会改变中间层的高激活区域结构。
5. **训练过程本身可能导致最大激活持续增长**，即使模型结构不变。
6. **Gemma3 等新型模型出现前所未有的高激活值（>6×10⁵）**，提示当前量化策略面临更大挑战。

### 方法的局限性
- **观察性研究性质**：仅报告相关性，未建立因果机制（如为何 MoE 抑制峰值）。
- **样本与上下文限制**：语料覆盖有限（缺长尾语言/复杂推理链），最长仅 4096 tokens，无法反映超长上下文行为。
- **匹配对数量少**：MoE vs. Dense、VL vs. Text 等对比仅基于 $n=2$ 对，泛化性受限。
- **量化验证简化**：INT-8 实验仅覆盖 8 个模型、单层、两种策略，未涵盖旋转、前缀等高级量化技术。
- **未深入机制解释**：未结合 residual stream RMS 曲线或 early-layer step-up blocks 分析失败案例（如 Qwen3.5-0.8B）。

### 未来工作方向
- 开展干预实验（intervention studies）探究激活峰值形成的因果机制；
- 扩展至更多家族和架构的匹配对比（尤其是 MoE 和多模态）；
- 探索超长上下文（32K+）下的激活极值行为；
- 构建更细粒度的训练阶段分解（分离 SFT 与 RLHF/DPO 影响）；
- 将 $M$ 与实际任务性能下降关联，形成“量化风险评分卡”；
- 发布完整的可复现清单（HuggingFace ID、commit hash、dtype、RoPE base 等）。

> ✅ **代码已开源**：https://github.com/clx1415926/Max_act_llm  
> 📌 **核心主张**：**$M = \max{|a|}$ 应作为每个开源权重发布时的标准附带信息之一**，以支持安全高效的低比特部署。

</details>

---

### 7. [Runtime-Orchestrated Second-Order Optimization for Scalable LLM Training](https://arxiv.org/abs/2605.16184)

**Authors**: Yishun Lu, Junhao Zhang, Zeyu Yang, Wes Armour  
**Category**: cs.DC  
**Published**: 2026-05-18  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.16184v1  

#### Abstract
Second-order methods offer an attractive path toward more sample-efficient LLM training, but their practical use is often blocked by the systems cost of maintaining and updating large matrix-based optimizer states. We introduce \textbf{Asteria}, a runtime system designed to remove this bottleneck by...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Runtime-Orchestrated Second-Order Optimization for Scalable LLM Training

## 1. 论文的主要贡献和创新点

### 解决的问题
第二阶优化方法（如 Shampoo、SOAP）在理论上能显著提升 Large Language Model（LLM）训练的**样本效率**和收敛速度，但由于其需要维护大规模矩阵形式的优化器状态（如协方差矩阵 $L$ 和 $R$），导致存在以下三大系统瓶颈：

- **垂直容量墙（Vertical Capacity Wall）**：在单 GPU 或内存受限设备上无法进行横向分片（horizontal sharding），导致 OOM。
- **重叠破坏墙（Overlap Disruption Wall）**：密集的 $O(d^3)$ 矩阵求逆或特征分解操作阻塞训练流水线，破坏计算-通信重叠。
- **全局共识墙（Global Consensus Wall）**：严格的同步更新要求与异构硬件拓扑不匹配，造成跨节点同步开销过大。

这些问题使得第二阶优化难以在实际 LLM 训练中部署。

---

### 提出的新方法：Asteria
作者提出 **Asteria** —— 一个硬件-软件协同设计的运行时系统，通过将第二阶优化逻辑从关键 GPU 路径中解耦来解决上述问题。其核心创新包括：

#### （1）Architecture-Adaptive Asymmetric Memory Tiering  
- 将优化器状态按生命周期分布于不同层级内存：
  - **factor_matrices** 存于 GPU 上，用于梯度累积；
  - **inverse_factor_matrices** 存于 CPU 内存或 UVM（Unified Virtual Memory）；
  - 可选地使用 **NVMe** 进行冷数据暂存。
- 利用 `cudaMemPrefetchAsync` 实现 GPU 对主机内存的透明访问，避免冗余拷贝。

> ✅ 不是简单的 offloading，而是针对第二阶优化特有的“非对称计算-访问模式”设计的专用内存层次结构。

#### （2）Hook-Orchestrated Shadow-State Pipeline  
- 利用 PyTorch 的 `forward_hook` 和 `full_backward_prehook` 触发辅助任务调度；
- 在后台 CPU 线程异步执行昂贵的 inverse-root 计算；
- 使用低优先级 CUDA stream 处理状态预取和刷新，实现与主训练流的并行化。

> ✅ 将 $O(d^3)$ 操作完全移出关键路径，消除周期性延迟尖峰。

#### （3）Bounded-Staleness Selective Coherence  
- 引入基于块的 **staleness-aware 同步机制**：
  - 每个 preconditioner block 维护版本号和最后同步步数；
  - 只有超过 staleness 预算的 block 才参与同步；
- 采用 **拓扑感知的分层同步策略**：
  - 先在 node-local 组内平均；
  - 再通过代表节点跨节点同步；
  - 最后广播回本地 peer。

> ✅ 显著减少通信量，避免不必要的 host-device 往返。

---

### 相比现有方法的优势
| 方面 | 现有方法（如 Distributed Shampoo） | Asteria |
|------|-------------------------------|--------|
| 内存管理 | 依赖多 GPU 横向分片 | 支持单 GPU + CPU/NVMe 分层存储 |
| 计算调度 | 同步阻塞式 inverse-root 更新 | 异步后台执行，隐藏延迟 |
| 通信模式 | 全局 AllGather，高频率同步 | 选择性同步 + 拓扑感知协调 |
| 系统兼容性 | 修改训练图或需特定框架支持 | 零侵入式集成 FSDP，无需改模型代码 |

> ✅ Asteria 实现了真正的 **runtime-level decoupling**，而非仅算法层面的摊销（amortization）。

---

## 2. 核心实验方法和设置

### 数据集与模型
- **数据集**：English C4 corpus
- **Tokenizer**：T5 tokenizer（vocab size ≈ 32k）
- **序列长度**：1024 tokens
- **模型系列**：OLMo-family
  - **660M**：d_model=1408, 24 层，GELU
  - **OLMo-2 1B**：d_model=2048, 24 层，SwiGLU + RMSNorm
  - **OLMo-2 7B**：d_model=4096, 32 层，mlp_hidden_size=22016

所有模型均使用 RoPE（rotary positional embedding），无 bias term。

---

### 实验平台
#### Scale-Up 设置（内存受限）
- **平台**：Nvidia DGX Spark
- **配置**：单 GB10 GPU + 128GB Unified Memory（UVM）
- **目标**：验证在极端内存限制下是否可运行第二阶优化

#### Scale-Out 设置（分布式训练）
- **平台**：Multi-node Nvidia GH200 集群
- **并行策略**：Fully Sharded Data Parallel (FSDP)
- **节点数扩展**：2 ~ 16 nodes（强扩展性测试）

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **AdamW** | 主流一阶优化器，作为性能基准 |
| **SOAP / KL-Shampoo** | 当前最先进的第二阶优化器 |
| **Native SOAP/KL-Shampoo** | 原生实现，直接在 GPU 上执行 inverse-root |
| **Asteria-SOAP / Asteria-KL-Shampoo** | 加入 Asteria runtime 的版本 |

> 所有方法共享相同的 FSDP 训练流程，precondition frequency = 10，max preconditioner dim = 2048。

---

### 评估指标
| 指标 | 说明 |
|------|------|
| **Step Time** | 单步耗时分布，关注是否存在周期性 spike |
| **Training Loss vs Step** | 优化收敛行为一致性 |
| **Training Loss vs Wall Time** | 实际时间下的收敛效率 |
| **Total Energy Consumption** | SoC/CPU/GPU 总能耗（通过硬件 telemetry 测量） |
| **Energy-Loss Tradeoff** | 单位能量带来的 loss 下降幅度 |
| **Speedup & Per-Step Time** | 强扩展性表现 |
| **Final Evaluation Loss** | 模型最终质量 |

---

## 3. 主要实验结果和性能指标

### （1）在 DGX Spark（单节点）上的表现

#### ▶ Step-Time 平滑化（Fig. 4 & 5）
- **Native KL-Shampoo**：每 10 步出现一次高达 **96.0s** 的 spike；
- **Native SOAP**：峰值达 **80.7s**；
- **Asteria-KL-Shampoo**：step time 稳定在 **~66.5s**，几乎完全消除 spike；
- **Asteria-SOAP**：稳定在 **~67.9s**

> ⬇️ **Latency 隐藏效果接近 100%**

#### ▶ 成功训练 1B 参数模型
- 在仅 128GB 统一内存 + 单 GB10 GPU 条件下，首次实现了 **1B 参数 LLM 的第二阶训练**；
- 无需多 GPU 分片，证明 Asteria 可突破“垂直容量墙”。

#### ▶ 能耗分析（Fig. 6 & 7）
| 方法 | SoC 总能耗（相对 AdamW） | Loss Reduction per Energy |
|------|--------------------------|----------------------------|
| AdamW | 100% | 3.0942 |
| Native SOAP | 119.7% | 3.0685 |
| Native KL-Shampoo | 117.1% | 3.0562 |
| **Asteria-SOAP** | **113.8%** | **3.1299** ✅ |
| **Asteria-KL-Shampoo** | **107.9%** | **3.3010** ✅✅ |

> 🔋 Asteria 不仅降低了总能耗，还提升了 **energy-efficiency**，成为唯一优于 AdamW 的第二阶方案。

---

### （2）在 GH200 多节点集群上的表现

#### ▶ 收敛效率 vs Wall Clock Time（Fig. 8）
- **Step-wise convergence**：Asteria 与原生方法几乎一致，说明未损害优化性质；
- **Wall-time convergence**：Asteria 更早达到相同 loss 水平（虚线处），提速明显。

#### ▶ Staleness Budget 影响（Fig. 9）
- 当 $S \geq 3$ 时，训练时间趋于稳定下降；
- $S=5$ 是最佳平衡点，在不损失最终精度的前提下最大化性能增益；
- 最终 eval loss 在不同 $S$ 下波动极小（<0.02），表明 bounded staleness 安全可行。

#### ▶ 大模型扩展性（Fig. 10）
- 在 **1B 和 7B 模型**上，Asteria 均保持对 AdamW 的 wall-time 优势；
- 特别是在 7B 模型中，早期即拉开差距，后期持续领先。

#### ▶ 强扩展性（Strong Scaling, Fig. 11）
- 在 2~16 节点范围内，Asteria 实现更高 speedup 和更低 per-step time；
- 尤其对 KL-Shampoo 提升显著，说明其更受益于 selective coherence 和异步计算。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **第二阶优化的实用性瓶颈不在算法本身，而在系统实现方式**：
   - 传统方法将优化器状态视为同质张量统一处理，忽略了其独特的“生产-消费”不对称性。
   
2. ✅ **runtime-level 解耦比 algorithmic amortization 更有效**：
   - 通过分离计算路径、引入 shadow pipeline 和 bounded staleness，Asteria 实现了真正意义上的负载卸载与重叠。

3. ✅ **第二阶优化可以在资源受限环境下运行**：
   - 单 GPU + 有限内存不再是障碍，为边缘训练提供了可能。

4. ✅ **更好的系统调度可以同时提升性能和能效**：
   - Asteria 在 wall-clock time 和 energy-loss tradeoff 上双双超越一阶方法，改变了“第二阶更贵”的固有认知。

5. ✅ **分布式第二阶优化应视 preconditioner freshness 为可调资源**：
   - 类似缓存一致性协议，允许一定程度的 stale 状态可大幅降低同步成本。

---

### 方法的局限性
- **依赖 UVM 或高性能 CPU-GPU 互连**：在 PCIe 带宽较低的传统 discrete-GPU 架构上性能可能下降；
- **当前主要适配 FSDP**：尚未全面支持 Tensor Parallelism 或 Pipeline Parallelism；
- **NVMe I/O 性能敏感**：若底层存储延迟高，prefetch 效果会打折扣；
- **CPU worker 资源竞争**：在 CPU 已饱和的场景中可能引入新瓶颈。

---

### 未来工作方向
1. 扩展至 **Tensor/Model/Pipeline Parallelism 混合并行架构**；
2. 探索 **更智能的 prefetching policy**，结合 trace prediction 动态调整 staleness 预算；
3. 支持 **异构加速器**（如 AMD GPU、TPU）；
4. 结合 **quantization** 技术进一步压缩 preconditioner 存储；
5. 开发 **自动 tuning layer**，根据硬件配置自适应选择 memory tiering 策略。

---

## 总结
Asteria 通过 **runtime-orches-trated design** 成功解决了第二阶优化在 LLM 训练中的系统瓶颈，证明了：

> 🎯 **第二阶优化能否实用，不仅取决于 optimizer 数学，更取决于 runtime 系统设计。**

它展示了如何通过 **memory tiering + async compute + selective coherence** 的联合工程，让原本“理论优越但实践困难”的方法变得高效、节能且可扩展，为下一代 LLM 训练基础设施提供了重要范式。

</details>

---

### 8. [IO-SVD: Input-Output Whitened SVD for Adaptive-Rank LLM Compression](https://arxiv.org/abs/2605.15626)

**Authors**: Ali Abbasi, Chayne Thrash, Haoran Qin, Hamed Pirsiavash, Soheil Kolouri  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.15626v1  

#### Abstract
Large language models deliver strong performance across language and reasoning tasks, but their storage and compute costs remain major barriers to deployment in resource-constrained and latency-sensitive settings. SVD-based post-training compression offers a hardware-agnostic way to reduce model siz...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：IO-SVD: Input-Output Whitened SVD for Adaptive-Rank LLM Compression**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLMs）在自然语言理解、推理等任务上表现出色，但其高昂的存储和计算成本限制了在资源受限或低延迟场景（如边缘设备、机器人系统）中的部署。现有的 **SVD-based 压缩方法** 虽然能通过低秩分解减少参数量和计算开销，但仍存在以下问题：
- 多数方法仅基于 **input-only whitening**（如激活统计），忽略了输出侧对预测分布的影响；
- 采用 **固定或均匀的秩分配策略**，未能适应不同层对压缩的敏感性差异；
- 在混合压缩（SVD + 量化）中，行选择策略是启发式的，未考虑量化误差对任务损失的实际影响。

### **提出了什么新方法或新思路**
本文提出 **IO-SVD（Input-Output Whitened SVD）**，一种用于自适应秩的 LLM 压缩方法，包含三大核心创新：

#### ✅ **1. KL-aware 双向白化（Double-Sided Whitening）**
- 推导出一个基于 **KL 散度近似** 的双侧白化空间，同时建模输入激活几何（`Re`）和输出预测敏感性（`Ce`）。
- 构造白化矩阵 $ B_e = C^{1/2} W_e R^{1/2} $，然后在其上进行 SVD 分解，使得保留的方向既解释输入方差，也影响最终预测质量。

#### ✅ **2. 损失感知的异构秩分配（Loss-Aware Adaptive Rank Allocation）**
- 引入一种高效的贪婪策略，根据每个奇异分量对校准损失（calibration loss）的一阶影响评分（$ I_{e,i} = g_{e,i} \cdot \sigma_{e,i} $），动态决定各层的截断秩。
- 在全局参数预算下，优先移除“最不敏感”的奇异值，实现跨层异构秩分配，无需迭代优化。

#### ✅ **3. 损失感知的混合压缩重映射（Loss-Aware Remapping for Hybrid Compression）**
- 将 SVD 截断后的因子行按 **量化后引起的预测损失变化** 进行打分。
- 贪心地将低损失影响的行转为 8-bit 存储，其余保持 fp16，从而在有限预算内最大化性能恢复。

### **相比现有方法的优势**
| 维度 | IO-SVD | 先前方法（如 SVD-LLM, Dobi-SVD, ZS-SVD） |
|------|--------|------------------------------------------|
| 白化空间 | 输入 + 输出双侧敏感性 | 仅输入激活统计 |
| 秩分配 | 自适应、异构、基于损失敏感性 | 固定或启发式分配 |
| 混合压缩 | 行选择基于量化损失预测 | 固定规则或结构化选择 |
| 性能保留 | 更好地维持 PPL 和下游准确率 | 高压缩比下性能下降明显 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **语言建模评估**：
  - `WikiText-2`, `Penn Treebank (PTB)`, `C4` —— 用于计算 **Perplexity (PPL)**。
- **零样本推理任务**：
  - `OpenBookQA`, `ARC-easy/challenge`, `WinoGrande`, `HellaSwag`, `PIQA`, `MathQA` —— 用于评估 **zero-shot accuracy**。
- **视觉语言模型（VLM）评估**：
  - `ScienceQA-IMG`, `SEED-Bench` —— 测试多模态能力。
- **校准数据（Calibration Set）**：
  - 使用 256 条随机采样的 `WikiText-2` 序列（长度 2048）进行白化矩阵和梯度估计。

### **实验设置和评估指标**
- **压缩目标**：对 Transformer 中的 Q/K/V/O 和 MLP 层进行压缩。
- **维护比例（Maintenance Ratio）**：0.8、0.6、0.4（即剪枝率 20%、40%、60%）。
- **评估指标**：
  - **PPL ↓**（越低越好）
  - **Zero-shot Accuracy ↑**（越高越好）
  - **Decode Throughput (tokens/sec) ↑**
  - **Peak GPU Memory Usage ↓**

### **基线方法对比**
- **SVD-based 方法**：
  - `ASVD`：基于激活感知的加权 SVD。
  - `SVD-LLM`：基于激活重建误差 + LoRA 微调。
  - `Dobi-SVD`：可微分 SVD + 增量 PCA。
  - `ZS-SVD`：零和 SVD，最小化预测损失变化。
- **混合压缩变体**：
  - 启用 remapping 的版本（如 `Dobi-SVD*`, `ZS-SVD*`）作为强基线。
- **VLM 特定方法**：
  - `QSVD`, `WSVD` —— 对比视觉语言模型压缩效果。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### 📊 **表1：LLaMA-7B 上不同压缩比下的表现（部分摘录）**

| Maintenance Ratio | Method       | Wiki2 PPL ↓ | Avg Acc ↑ |
|------------------|--------------|------------|-----------|
| 0.8              | Baseline     | 5.68       | 0.55      |
|                  | IO-SVD       | **6.41**   | **0.50**  |
|                  | ZS-SVD*      | 5.90       | 0.54      |
|                  | **IO-SVD+**  | **5.59**   | **0.54**  |
| 0.6              | IO-SVD       | 9.84       | 0.41      |
|                  | **IO-SVD+**  | **6.27**   | **0.50**  |
| 0.4              | IO-SVD       | 27.70      | 0.31      |
|                  | **IO-SVD+**  | **6.41**   | **0.51**  |

> 💡 **说明**：`IO-SVD+` 表示启用 loss-aware remapping，在高剪枝率下显著优于所有基线。

#### 📊 **表3：跨模型家族测试（OPT-6.7B, Vicuna-7B, LLaMA-13B @ 20% pruning）**

| Model       | Method   | PPL ↓     | Acc ↑     |
|-------------|----------|-----------|-----------|
| LLaMA-13B   | Original | 5.09      | 0.59      |
|             | SVD-LLM  | 6.43      | 0.55      |
|             | ZS-SVD   | 5.84      | 0.56      |
|             | **IO-SVD** | **5.60**  | **0.56**  |

> ✅ IO-SVD 在多个架构上均取得最低 PPL，且准确率持平最优。

#### 📊 **表2：VLM 压缩结果（SmolVLM 2B @ 50% ratio）**

| Method   | ScienceQA-IMG ↑ | SEED-Bench ↑ | Avg ↑ |
|----------|------------------|---------------|-------|
| FP16     | 84.53            | 68.47         | 76.53 |
| ASVD     | 29.30            | 17.85         | 8.96  |
| Ours     | **83.34**        | **68.36**     | **75.58** |

> 🔥 IO-SVD 几乎无损压缩 SmolVLM，远超其他 SVD 方法。

---

### **消融实验结果**

#### 🔍 **表4：白化空间与秩分配策略消融（LLaMA-7B）**

| Whitening Space       | Het. Rank? | Wiki2 PPL @ 0.4 ↓ |
|------------------------|------------|--------------------|
| Input-only (SVD-LLM)   | ×          | 53.74              |
| Double-sided (OBD-LLM) | ×          | 32.95              |
| **Double-sided (Ours)** | ×          | **32.09**          |
| Input-only             | ✓          | 62.76              |
| Double-sided           | ✓          | 28.19              |
| **Double-sided (Ours)** | ✓          | **27.70**          |

> ✅ 双侧白化 + 异构秩分配带来最大收益，验证了两个设计的有效性。

#### 🔍 **图3：Top-K 曲率估计的消融**
- 发现存在一个“甜点”K 值（如 K=5），过小则信息不足，过大则噪声增加。
- 最优 K 在训练集选定后可在多个测试集泛化，表明估计具有鲁棒性。

#### 🔍 **表6：Loss-aware Remapping 消融**
| Method     | Mode             | Wiki2 PPL @ 0.8 ↓ | @ 0.6 ↓ |
|------------|------------------|-------------------|----------|
| IO-SVD     | Compressed       | 6.41              | 9.84     |
|            | + Remap          | 5.76              | 6.48     |
|            | **+ Loss-aware+** | **5.59**          | **6.27** |

> ✅ Loss-aware remapping 显著优于标准 remapping，证明了损失预测指导的重要性。

---

## 4. **关键结论和发现**

### **主要发现**
1. **双向白化优于单侧白化**：
   - 仅使用输入激活统计不足以捕捉对最终预测的影响；引入输出侧敏感性（via KL curvature）显著提升压缩保真度。
2. **异构秩分配更高效**：
   - 不同层对压缩的容忍度不同，统一截断会浪费容量；自适应分配能在相同预算下保留更多有用信息。
3. **混合压缩需损失感知控制**：
   - 单纯将某些行量化为 8-bit 并不能保证性能稳定；只有结合损失预测才能实现“精准降精度”。
4. **实际推理加速明显**：
   - 如图4所示，在 LLaMA-2-7B 上，IO-SVD + KV-cache 压缩可实现 **4.34× 解码吞吐提升**，峰值内存从 77.6GB 降至 23.1GB。

### **方法的局限性**
1. **Top-K 近似可能忽略长尾敏感性**：
   - 当前输出侧曲率估计仅限 top-K token，可能遗漏稀有词的预测敏感性。
2. **贪心秩分配非全局最优**：
   - 虽然高效，但逐个删除组件的方式无法保证找到全局最优秩配置。
3. **尚未扩展至超大规模模型（>13B）**：
   - 实验最大只到 13B 参数模型，更大规模下的可扩展性和稳定性有待验证。

### **未来工作方向**
- 改进 curvature estimation 效率与覆盖范围（如使用低秩近似）。
- 探索更高级的 rank allocation 机制（如基于强化学习或二阶优化）。
- 扩展至 MoE 架构、多模态大模型及生成任务（如摘要、对话）中的端到端评估。
- 结合硬件感知调度，进一步优化实际部署效率。

---

> ✅ **总结一句话**：  
> **IO-SVD 通过 KL-aware 双向白化、损失驱动的异构秩分配与 remapping 策略，在几乎不牺牲性能的前提下实现了高效的 LLM/VLM 压缩，并带来了显著的推理速度和内存优势，是一种实用性强、理论扎实的新一代 post-training compression 框架。**

🔗 代码已开源：[https://github.com/mint-vu/IO-SVD.git](https://github.com/mint-vu/IO-SVD.git)

</details>

---

### 9. [Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design](https://arxiv.org/abs/2605.15871)

**Authors**: Alberto Pepe, Chien-Yu Lin, Despoina Magka, Bilge Acun, Yannan Nellie Wu, Anton Protopopov, Carole-Jean Wu, Yoram Bachrach  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.15871v1  

#### Abstract
Toward recursive self-improvement, we investigate LLM agents autonomously designing foundation models beyond standard Transformers. We introduce a dual-framework approach: AIRA-Compose for high-level architecture search, and AIRA-Design for low-level mechanistic implementation. AIRA-Compose uses 11 ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在解决当前大语言模型（LLM）架构设计依赖于人类专家直觉和手动调优的瓶颈问题。随着后Transformer时代到来，混合架构（如结合Attention、Mamba等计算原语）的设计空间变得极其庞大且组合复杂，传统人工探索方式难以高效覆盖。此外，作者将此视为实现**Recursive Self-Improvement (RSI)** 的关键一步——即让AI代理能够自主地发现并优化其自身的神经网络架构。

### 提出的新方法与思路
论文提出了一个双框架的代理驱动神经架构搜索（NAS）范式：

- **AIRA-Compose**：用于**高层级架构搜索**。该框架部署一个由11个LLM代理组成的集成系统，在一个由基本计算原语（Attention, MLP, Mamba）构成的组合设计空间中进行导航。它通过两阶段流程运作：首先在百万参数规模的小型模型上迭代设计和评估候选架构；然后将表现最佳的设计外推到350M、1B和3B等更大参数规模。
  
- **AIRA-Design**：用于**低层级机制性实现**。该框架任务是让最多20个代理直接编写全新的注意力机制以处理长程依赖，并实现高性能的训练脚本。

这两个框架均构建在**AIRS-BENCH**任务标准之上，提供了一个灵活且可扩展的结构，使代理能执行完整的科学研究循环。

### 相比现有方法的优势
- **超越手工设计**：发现的架构在多个下游任务上持续优于Llama 3.2和Composer等人工或优化器发现的基线。
- **超越传统NAS**：相比基于贝叶斯优化或确定性搜索的传统NAS方法，代理能利用其上下文感知能力提出更具创造性的结构假设。
- **自动化程度高**：整个研究闭环（假设生成、代码编写、执行、评估、迭代改进）均由代理自动完成，减少了工程负担。
- **支持RSI路径**：为“自我改进”的AI系统提供了可行的技术路径，是迈向真正递归自优化的重要一步。

---

## 2. 核心实验方法和设置

### 使用的数据集
#### AIRA-Compose 阶段
- **MAD**：六个合成token操作任务（选择性复制、压缩、上下文回忆），用于小规模代理搜索。
- **BabiStories**：儿童故事合成语料库，用于训练和评估。
- **DCLM**：一个固定的子集，用于小规模训练和验证。

#### AIRA-Design 阶段
- **Long Range Arena (LRA)**：评估序列模型捕捉长程依赖的能力，具体任务包括：
  - **ListOps**：解析前缀表示法中的数学表达式。
  - **Text (IMDb)**：二分类情感分析，需理解全文语义。
  - **Retrieval (AAN)**：学术论文摘要对的相似度判断。
- **Autoresearch**：开放式的优化挑战，目标是在固定时间预算内最小化验证集的bits-per-byte (BPB)。

### 实验设置和评估指标
#### AIRA-Compose 设置
- **搜索空间**：16层架构，每层从{MLP (M), multi-head Attention (mA), Mamba SSM (Mb)}中选择。
  - 2-原语空间：2¹⁶ ≈ 65K种排列。
  - 3-原语空间：3¹⁶ ≈ 43M种排列。
- **评估流程**：
  1. 代理提交 `submission.csv` 文件描述架构。
  2. `evaluate.py` 脚本独立训练并评估该架构。
  3. 表现最好的设计被聚合（clustering）并外推至大规模。
- **外推方法**：通过拉伸（stretching）或堆叠（stacking）连续块来扩大模型尺寸。

#### AIRA-Design 设置
- **任务形式**：代理需直接生成 `model.py` 或 `train.py` 文件。
- **评估方式**：在固定GPU时间内运行训练脚本，报告最终性能。

### 基线方法对比
- **Llama 3.2**：作为主流Transformer基线。
- **Nemotron-H / Nemotron-2**：近似版本作为混合模型基线。
- **Composer-found models**：使用Composer框架通过优化算法找到的最佳模型。
- **Human SOTA**：在LRA上的最高人类记录。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在1B参数规模下的下游任务表现（固定token预算）
| 架构 | 验证损失 ↓ | 平均零样本准确率 ↑ | DCLM Core Score ↑ |
|------|------------|---------------------|--------------------|
| Llama 3.2 | 2.815 | 57.5% | 46.9 |
| Composer (Stretched) | 2.759 | 59.8% | 47.3 |
| **AIRAformer-D (Stretched)** | **2.734** | **59.7%** | **48.9** |
| **AIRAhybrid-D (Stretched)** | **2.719** | **60.5%** | 48.5 |

- **AIRAformer-D** 在验证损失上比Llama 3.2降低 **0.081**，平均准确率提升 **2.4%**。
- **AIRAhybrid-D** 准确率提升 **3.8%**。

#### 计算最优缩放（isoFLOP）性能
- **AIRAformer-C** 比Llama 3.2和最佳Composer Transformer分别快 **54%** 和 **71%**。
- **AIRAhybrid-C** 比修改后的Nemotron-2和最佳Composer混合模型分别快 **23%** 和 **37%**。

#### Long Range Arena (LRA) 结果
- 最佳代理设计的架构在文档匹配任务上达到人类SOTA的 **82%**（差距2.3个百分点）。
- 在文本分类任务上达到人类SOTA的 **91%**（差距2.6个百分点）。

#### Autoresearch 结果
- **Greedy Opus 4.5** 在固定时间预算下实现了 **0.968** 的验证bits-per-byte (BPB)，超过了公开参考值。

### 与基线方法的对比结果
- 所有代理发现的顶级架构（AIRAformer系列和AIRAhybrid系列）在验证损失、零样本准确率和few-shot DCLM Core Score三项指标上均显著优于Llama 3.2和Composer基线。
- 在LRA任务中，尽管未完全达到人类SOTA，但已非常接近，表明代理具备强大的机制性设计能力。
- 在Autoresearch任务中，代理不仅复现了已有优化策略，还发现了新的有效配置。

### 消融实验结果
- **不同聚合技术**：虽然采用了不同的聚类和聚合方法（N1/N2），但最终收敛到相似的Attention-to-MLP比例（如AIRAformer A/B为7:9，C/D为11:5），说明这些模式具有鲁棒性。
- **是否启用文献增强**：在Autoresearch任务中，向代理提供相关论文和代码仓库后，其优化策略发生改变，部分代理（如Opus 4.6 +Lit）表现出更稳定的性能提升。

---

## 4. 关键结论和发现

### 主要发现
1. **代理可以自主发现高性能架构**：AI代理不仅能有效搜索庞大的设计空间，还能发现超越人工设计和传统NAS方法的新型混合架构。
2. **两种范式互补**：AIRA-Compose适合在预定义模块间寻找最优连接，而AIRA-Design则适用于从零开始发明新机制。
3. **具备RSI潜力**：这是首次系统性展示AI代理可以自主发现下一代基础模型架构的工作，标志着向**Recursive Self-Improvement**迈出了坚实一步。
4. **工程效率高**：通过AIRS-BENCH标准化接口，大幅降低了代理与复杂训练管道之间的集成成本。

### 方法的局限性
1. **仍属工程合成而非科学创新**：在LRA任务中，代理更多是重组现有技术（如Performer、Longformer），尚未展现出真正的理论突破能力。
2. **单次生成失败率高**：One-shot代理在需要完整代码生成的任务中几乎无法产出有效解，凸显了迭代调试的重要性。
3. **超参敏感**：在可配置任务中，较弱的代理因无法有效管理更大的搜索空间而导致性能下降。
4. **框架偏见**：由于训练数据偏向PyTorch，代理在JAX/Flax环境下的表现受限。
5. **全文件重写限制因果推理**：AIRA-DOJO的greedy scaffold每次重写整个`train.py`，使得难以精确归因性能变化的原因。

### 未来工作方向
1. **升级代理平台**：测试更先进的代理框架（如AIRA2），支持多GPU并行搜索，减少代理与目标模型间的gap。
2. **引入工具使用能力**：赋予代理交互式调试、查看日志、运行单元测试等能力，提升其纠错效率。
3. **开发增量修改范式**：替代全文件重写，允许代理进行细粒度、可解释的代码变更。
4. **扩展评估基准**：将AIRA-Design应用于更新的长程建模基准。
5. **实现端到端代理化**：将目前非代理化的聚合与外推步骤也纳入代理任务中，实现完全自主的NAS闭环。

</details>

---

### 10. [ParamSpMM: Adaptive and Efficient Sparse Matrix-Matrix Multiplication on GPUs for GNNs](https://arxiv.org/abs/2605.15695)

**Authors**: Lixing Zhang, Guanhua Ye, Hongzheng Li, Shigang Li, Yingxia Shao  
**Category**: cs.DC  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.15695v1  

#### Abstract
Fueled by the ability to mine real-world graph data, GNN applications have experienced phenomenal growth. Sparse Matrix-Matrix Multiplication (SpMM) is a critical operator in GNNs. However, existing SpMM designs for GNNs struggle to adapt to diverse input characteristics. In this paper, we first con...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
图神经网络（GNN）中的 **Sparse Matrix-Matrix Multiplication (SpMM)** 是其计算的关键瓶颈，占用了大量执行时间。然而，现有的 SpMM 实现难以适应多样化的输入特征，具体体现在以下三方面：
- **图的数据局部性差异**：不同图中节点邻居的 ID 分布影响内存访问模式。
- **度分布不均**：如社交网络具有幂律分布，导致 workload imbalance。
- **嵌入维度变化**：GNN 各层的 embedding 维度（`dim`）不同，影响优化策略选择。

现有方法（如 DGL、PyG 中的静态 kernel 或 ASpT、GNNAdvisor 等动态设计）缺乏灵活性，采用固定优化策略，无法针对不同输入自适应调整，导致性能下降。

---

### **提出的新方法与创新思路**
本文提出了 **ParamSpMM** —— 一种参数化、高度自适应的 SpMM 框架，核心创新如下：

#### **(1) 参数化优化框架**
将三种主流优化技术统一为可配置参数：
- `V`：Vector Size，用于 **vectorized blocking**，提升 B 矩阵的数据复用。
- `S`：布尔值，控制是否启用 **workload balancing**，缓解负载不均衡。
- `F`：Coarsening Factor，决定 **thread coarsening** 粒度，减少 A 矩阵访存开销。
- `W`：线程块大小，影响并行粒度。

通过灵活组合这些参数，实现对不同输入特性的细粒度适配。

#### **(2) 新型稀疏矩阵格式：PCSR**
提出 **Parameterized Compressed Sparse Row (PCSR)** 格式，支持上述多优化技术的无缝集成：
- 在传统 CSR 基础上增加 `TRow` 数组，用于 workload balancing 时记录输出位置。
- 支持向量化存储非零元（V×1 向量），便于 blocking 和数据重用。
- 可根据配置动态生成，兼容多种优化路径。

#### **(3) ML-based SpMM-decider**
引入基于机器学习的决策器，自动预测最优 `(W, F, V, S)` 配置：
- 输入特征包括三类：**Size features**（规模）、**Degree distribution features**（度分布）、**Data locality features**（局部性）。
- 特别定义两个关键指标：
  - **PR (Padding Ratio)**：衡量 blocking 引入的零填充程度。
  - **SR (Split Ratio)**：衡量 balancing 导致的额外写操作开销。
- 使用 **Random Forests** 模型进行预测，轻量且防过拟合。

---

### **相比现有方法的优势**
| 方面 | ParamSpMM | 现有方法（如 cuSPARSE、GNNAdvisor） |
|------|-----------|-------------------------------|
| **适应性** | 全面支持多参数联合优化，按需启用 | 固定策略或仅调少数参数 |
| **灵活性** | PCSR 统一表示多种优化路径 | 各优化独立实现，难协同 |
| **自动化** | ML 自动决策最佳配置 | 手动调参或启发式规则 |
| **性能稳定性** | 对各类图结构均表现稳健 | 在某些图上可能劣于基线 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **SpMM 测试集**：从 [SNAP](https://snap.stanford.edu/) 和 [DIMACS10](http://www.cc.gatech.edu/dimacs10/) 收集 **202 个真实世界图矩阵**（`1,000 < n < 8,000,000`），覆盖多种图类型（社交、引用、通信等）。
- **GNN 训练测试集**：来自 [OGB](https://ogb.stanford.edu/) 的 6 个标准图数据集，未参与训练 SpMM-decider，确保公平性。

---

### **实验设置与评估指标**
- **硬件平台**：NVIDIA A6000 GPU（84 SMs, Compute Capability 8.6）
- **软件环境**：PyTorch 扩展，使用 GCC 9.4.0 + NVCC 11.6 编译，开启 `-O3` 优化。
- **评估指标**：
  - **SpMM 性能**：以 **GFLOPS** 和相对于 **cuSPARSE** 的 **speedup** 衡量。
  - **GNN 训练加速比**：相对于 DGL 的端到端训练速度提升。
  - **预测准确率**：SpMM-decider 推荐配置 vs 最优配置的性能归一化比值。

---

### **基线方法对比**
分为三类基线：
| 类型 | 方法 | 说明 |
|------|------|------|
| **Static** | `cuSPARSE`, `GE-SpMM` | 固定 kernel，无自适应能力 |
| **Heuristic-based** | `GNNAdvisor` | 基于规则的 workload balancing 优化 |
| **ML-based** | `DA-SpMM` | 使用 ML 决策，但未考虑 blocking 与 thread coarsening |

所有方法均默认使用 **Rabbit Reordering** 预处理以增强数据局部性。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) SpMM 整体性能对比（Table 4 & Fig. 4）**
在 202 个矩阵、多个 `dim`（共 3232 个 SpMM 输入）上的平均加速比如下：

| 方法 | 平均 speedup (vs cuSPARSE) |
|------|-----------------------------|
| **ParamSpMM** | **1.92×** |
| GE-SpMM | 1.55× |
| GNNAdvisor | 1.55× |
| DA-SpMM | 1.64× |

> ✅ **结论**：ParamSpMM 显著优于所有基线，在绝大多数输入下保持稳定加速，而其他方法在部分场景下甚至劣于 cuSPARSE。

#### **(2) GNN 训练加速效果（Fig. 5）**
在 **GCN** 和 **GIN** 模型上的端到端训练加速比（vs DGL）：

| 模型 | 平均 speedup | 最高 speedup |
|------|---------------|--------------|
| GCN | 1.60× | 2.19× |
| GIN | 1.61× | 2.59× |

> ✅ **结论**：由于 SpMM 是 GNN 的主要开销，ParamSpMM 能有效提升整个模型训练效率。

---

### **消融实验结果**

#### **(1) SpMM-decider 预测有效性（Table 5）**
比较 ML 预测配置（pre）与随机配置（rnd）的归一化性能（越高越接近最优）：

| dim | pre (%) | rnd (%) |
|-----|--------|--------|
| 32  | 99.69% | 73.47% |
| 64  | 98.24% | 76.65% |
| 128 | 99.30% | 78.14% |
| 256 | 98.75% | 70.33% |

> ✅ **结论**：SpMM-decider 能精准推荐高性能配置，远超随机尝试。

#### **(2) 图重排序（Graph Reordering）的影响（Table 6）**
验证 Rabbit Reordering 对 ParamSpMM 的增益：

| 方法 | 平均 speedup (vs 无重排) |
|------|--------------------------|
| cuSPARSE | 1.14× |
| ParamSpMM_wo_reorder | 1.75× |
| **ParamSpMM** | **2.21×** |

> ✅ **结论**：
> - 即使不重排，ParamSpMM 仍显著优于 cuSPARSE（1.75×）；
> - 重排序进一步带来 **+26%** 加速，表明 ParamSpMM 更好地利用了数据局部性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **现有 SpMM 方法适应性不足**：多数设计只关注单一优化维度，无法应对 GNN 中复杂的输入多样性。
2. **参数化是实现高适应性的关键**：通过将 blocking (`V`)、balancing (`S`)、coarsening (`F`) 统一为可调参数，可实现“按需优化”。
3. **PCSR 成功统一多优化路径**：新格式支持动态生成，兼容多种优化组合，是 ParamSpMM 的基础。
4. **ML 可高效自动化配置选择**：SpMM-decider 准确率高（>98%），显著降低人工调参成本。
5. **综合优化带来显著收益**：在真实 GNN 场景中，ParamSpMM 实现 **1.92× SpMM 加速** 和 **~1.6× GNN 训练加速**。

---

### **方法的局限性**
1. **预处理开销**：PCSR 构建和图重排序需要一定时间，虽可在迭代训练中摊销，但仍影响首次运行延迟。
2. **ML 模型泛化能力依赖特征质量**：若遇到极端分布的新图类型，SpMM-decider 可能误判。
3. **当前仅面向单 GPU**：未探索分布式或多卡扩展场景。

---

### **未来工作方向**
1. **支持更多硬件后端**：如 AMD GPU、TPU 或 AI 加速器。
2. **在线自适应机制**：结合运行时反馈动态调整参数，而非仅依赖离线预测。
3. **扩展至 SpMV 与其他稀疏算子**：将参数化思想推广至 GNN 中其他稀疏操作。
4. **端到端编译器集成**：将 ParamSpMM 集成进 GNN 编译器（如 TorchScript、Relay），实现全自动优化流水线。

---

> 📌 **总结一句话**：  
> **ParamSpMM 通过参数化设计 + PCSR 格式 + ML 决策器，实现了对多样化 GNN 输入的高度自适应 SpMM 优化，在真实场景中取得高达 1.92× 的性能提升，显著增强了 GNN 的训练效率。**

</details>

---

### 11. [Differentiable Mixture-of-Agents Incentivizes Swarm Intelligence of Large Language Models](https://arxiv.org/abs/2605.15706)

**Authors**: Xingjian Wu, Junkai Lu, Siyu Yan, Xiangfei Qiu, Jilin Hu, Chenjuan Guo, Bin Yang  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.15706v1  

#### Abstract
Recent advances in Large Language Models (LLMs) have catalyzed the development of multi-agent systems (MAS) for complex reasoning tasks. However, existing MAS typically rely on pre-defined or pre-compiled communication topologies, which limits their flexibility and adaptability to dynamic task requi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Differentiable Mixture-of-Agents Incentivizes Swarm Intelligence of Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Multi-Agent Systems (MAS)** 通常依赖于预定义或“预编译”的通信拓扑（如 Chain、Star、Tree 等），这种静态结构在面对动态任务需求时缺乏灵活性和适应性。尤其在复杂推理过程中，预先设计的工作流可能无法应对执行过程中的变化，导致资源浪费或解决方案不足。

### 提出的新方法：**Differentiable Mixture-of-Agents (DMoA)**
DMoA 是一种**自演化的多智能体框架**，其核心思想是：
- 在每个推理步骤中**动态路由并激活代理（agent）**，而非固定工作流。
- 通过一个**可微分的、上下文感知的路由机制**，结合历史信息与当前语境，实现稀疏且自适应的 agent 激活。
- 利用**预测熵（predictive entropy）作为自监督信号**，优化路由策略，支持无需外部标注的测试时训练（test-time training）。

### 相比现有方法的优势
| 维度 | 传统 MAS / MoA | DMoA |
|------|----------------|-------|
| **拓扑灵活性** | 静态或任务级动态（round-wise） | 步骤级弹性调整（step-wise），模拟任意通信拓扑 |
| **资源效率** | 固定调用所有 agents（dense） | 稀疏激活，按需调度 |
| **学习方式** | 强化学习（RL）、人工设计规则 | 自监督 + 可微分优化，支持 test-time adaptation |
| **适应能力** | 依赖先验设计 | 能随任务进展自我演化，具备 spatio-temporal unboundedness |

---

## 2. 核心实验方法和设置

### 使用的数据集
共在 **9 个基准任务**上进行评估，涵盖多种推理场景：

| 数据集 | 任务类型 | 评估指标 |
|--------|----------|-----------|
| **MMLU** | 广域知识推理 | Accuracy |
| **GSM8K**, **MultiArith**, **SVAMP**, **AQuA** | 数学推理 | Accuracy |
| **HumanEval**, **DS-1000** | 编程生成与调试 | Pass@1 / Accuracy |
| **HotpotQA** | 多跳问答 | Accuracy / F1 |
| **DDXPlus** | 医疗诊断推理 | Accuracy |

> 所有数据集均采用标准划分与公开协议，确保公平比较。

### 实验设置
- **主干模型**：统一使用 `gpt-oss-120b` 作为底层 LLM。
- **Agent Pool 构成**：包含不同角色（Profile）、工具（Tool）和 LLM 的异构 agents，分为三类：
  - **Code-oriented agents**（如 Programmer, Bug Fixer）
  - **Math-oriented agents**（如 Math Solver, Inspector）
  - **Open-domain QA agents**（如 Wiki Searcher, Doctor）
- **Summarizer Agent**：负责聚合中间结果并决定是否终止推理。
- **路由机制组件**：
  - Sentence Transformer（all-MiniLM-L6-v2）用于编码上下文语义
  - GRU-based RNN-Router 实现历史状态建模
  - Linear Head 输出 agent logits
- **优化目标**：Pair-wise Ranking Loss 对齐 agent 路由概率与基于 entropy 的置信度排序

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **单智能体系统** | CoT, Self-Consistency (SC) |
| **静态多智能体系统** | Chain, Tree, Star, Complete Graph, Random Graph, AutoGen, MoA, LLM-Debate |
| **空间自适应系统** | GPTSwarm, G-Designer, ARG-Designer, SafeSieve |
| **时间自适应系统** | AFlow, SpecReason, STEER |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）
DMoA 在 **所有 9 个基准上均达到 SOTA 性能**，平均准确率从 Vanilla 的 74.22% 提升至 **89.38%**，提升高达 **15.16 个百分点**。

| 方法 | Avg. Accuracy |
|------|----------------|
| Vanilla | 74.22 |
| MoA | 75.28 |
| G-Designer | 83.51 |
| SafeSieve | 83.91 |
| **DMoA (Ours)** | **89.38** ✅ |

#### 典型任务表现亮点：
- **GSM8K**（数学）：98.87 → 比第二名高 +5.52%
- **DS-1000**（编程）：64.34 → 提升超 9.7 个百分点
- **DDXPlus**（医疗）：83.37 → 显著优于其他方法

> 更难的任务上增益更大，说明 DMoA 能有效协调复杂推理路径。

### Test-Time Training (TTT) 效果（Table 3）
DMoA 支持完全自监督的测试时训练，仅利用前 10–30 条测试样本的 entropy 信号即可进一步提升性能：

| 方法 | MMLU | GSM8K | HumanEval | DS-1000 | HotpotQA |
|------|------|--------|------------|----------|-----------|
| DMoA (Few-shot) | 91.35 | 98.87 | 95.62 | 64.34 | 90.38 |
| DMoA (TTT) | 91.80 | 98.65 | 96.04 | 65.44 | 89.50 |
| **DMoA (Few-shot + TTT)** | **93.50** | **99.30** | **97.52** | **65.55** | **91.40** |

✅ 表明 DMoA 具备强大的零样本迁移与在线适应能力。

### Ensembling Capability（表 4）
即使仅使用开源 LLMs（如 Qwen, Llama, Mistral），DMoA 也能超越闭源强模型（如 GPT-5.2, Gemini 3.2 Pro）：

| 方法 | MMLU | GSM8K | DS-1000 |
|------|------|--------|----------|
| GPT-5.2 | 83.72 | 88.56 | 42.39 |
| Gemini 3.2 Pro | 86.22 | 86.82 | 44.50 |
| **DMoA (Few-shot+TTT)\*** | **85.52** | **88.74** | **48.97** |

> \*表示基于开源 LLM 的 MAS —— DMoA 成功激发了 LLM 的群体智能（swarm intelligence）。

### 消融实验（Ablation Studies, Table 5）
验证各模块有效性：

| 变体 | HumanEval | DS-1000 | HotpotQA |
|------|-----------|----------|-----------|
| w/ LLM Selector (GPT-5.2) | 87.21 | 54.33 | 84.72 |
| w/ Linear Router | 92.11 | 60.02 | 88.71 |
| w/o Aggregation | 91.94 | 59.86 | 88.83 |
| w/o Adaptive $k_i$ | 93.26 | 61.72 | 89.41 |
| **Full DMoA** | **95.62** | **64.34** | **90.38** |

✅ 所有关键设计（RNN-Router、上下文聚合、自适应 agent 数量）均显著贡献性能。

---

## 4. 关键结论和发现

### 主要发现
1. **弹性拓扑优于固定结构**：  
   DMoA 通过 step-wise 动态路由，实现了真正的“时空无界”通信拓扑演化，解决了传统 MAS “预编译困境”。

2. **自监督路由可行且高效**：  
   利用 **predictive entropy** 作为内生监督信号，避免了对强化学习稀疏奖励的依赖，使系统可在测试阶段持续优化。

3. **稀疏激活 ≠ 性能损失，反而更优**：  
   尽管只激活部分 agents，DMoA 在多数任务上仍大幅领先，表明其能精准匹配 agent 能力与当前推理需求。

4. **鲁棒性强**：  
   在对抗性攻击下（如部分 agents 被替换为 adversarial agents），DMoA 表现出最强稳定性（图 7），性能下降最小。

5. **高度可扩展**：  
   随着 agent pool 规模 $N$ 和最大路由数 $K$ 增加，性能持续上升（图 6），边际收益逐渐饱和，适合资源可控部署。

### 方法的局限性
1. **Agent Pool 扩展成本高**：  
   当候选 agents 过多时，路由难度增加，需要更多 adaptation 数据或更强正则化。

2. **Context-length 敏感性**：  
   当前使用 Sentence Transformer 压缩上下文，可能丢失长程依赖细节；未来可探索用 LLM 本身做 routing decision。

3. **Entropy 信号的质量依赖解码策略**：  
   若 agent 输出不稳定，entropy 可能失真，影响路由质量。

### 未来工作方向
- 探索非穷举式的 agent 构造机制（avoid exhaustive enumeration）
- 使用大模型替代 Sentence Transformer 进行 context encoding 与 routing
- 结合 retrieval-augmented 方式动态构建 agent pool
- 应用于真实世界复杂任务（如科研辅助、教育辅导、自动化运维）

---

> ✅ **总结一句话**：  
> **DMoA 通过可微分、自监督、步粒度的 agent 路由机制，首次实现了真正意义上的弹性多智能体协作，在性能、效率、鲁棒性和泛化能力上全面超越现有方法，为构建可持续进化的 LLM swarm 提供了新范式。**

</details>

---

### 12. [ALSO: Adversarial Online Strategy Optimization for Social Agents](https://arxiv.org/abs/2605.15768)

**Authors**: Xiang Li, Liping Yi, Mingze Kong, Min Zhang, Zhongxiang Dai, QingHua Hu  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.15768v1  

#### Abstract
Social simulation provides a compelling testbed for studying social intelligence, where agents interact through multi-turn dialogues under evolving contexts and strategically adapting opponents. Such environments are inherently non-stationary, requiring agents to dynamically adjust their strategies ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ALSO: Adversarial Online Strategy Optimization for Social Agents

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在基于 **Large Language Model (LLM)** 的多智能体社会模拟（social simulation）中，大多数现有方法依赖于**静态 persona**（角色设定），缺乏动态适应能力。尽管已有研究尝试通过离线强化学习（offline RL）或外部规划器（external planners）提升社交智能，但这些方法通常假设环境是**平稳的（stationary）**，无法应对真实社交互动中因对手策略持续演化而导致的**非平稳性（non-stationarity）**。

此外，传统 Prompt Optimization (PO) 方法依赖固定验证集进行策略评估，在动态、交互性强的社会环境中表现不佳。

### 🚀 提出的新方法与创新思路

本文提出了 **ALSO (Adversarial Online Strategy Optimization)** ——首个面向多智能体社会模拟的**在线策略优化框架**，其核心创新如下：

1. **将多轮社交互动建模为对抗性 bandit 问题（adversarial bandit problem）**
   - 将“静态 persona + 动态策略指令”的组合视为一个“arm”（拉杆）
   - 不假设奖励分布稳定，适用于对手行为不断变化的非平稳环境
   - 借鉴 EXP3 算法思想，采用随机化选择机制增强鲁棒性

2. **引入轻量级神经代理模型（neural surrogate）预测稀疏反馈下的奖励**
   - 利用对话历史编码上下文，结合策略嵌入，训练一个可在线更新的小型 value network 来预测各策略的预期收益
   - 实现跨语义相关策略的**泛化能力**，显著提升样本效率

3. **构建闭环在线学习系统**
   - 每一轮交互后，根据 LLM evaluator 提供的 per-turn reward 更新 bandit policy 和 surrogate model
   - 支持实时、连续的策略调整，无需重新训练 LLM

### 🔍 相比现有方法的优势

| 方面 | ALSO | 现有方法（如 OPRO, EvoPrompt, Sotopia-Ω） |
|------|------|----------------------------------------|
| 学习范式 | **在线优化（online）** | 多为离线或周期性优化 |
| 非平稳性处理 | 显式建模，支持动态对手 | 假设 reward 分布平稳 |
| 样本效率 | 高（surrogate 泛化稀疏反馈） | 低（独立评估每个 prompt） |
| 计算开销 | 极低（仅训练轻量 surrogate） | 高（需调用 LLM 作为 optimizer） |
| 是否修改 LLM | 否（冻结主模型） | 否，但部分需额外 planner |

---

## 2. 核心实验方法和设置

### 📚 数据集

- **Sotopia**：一个全面的 LLM-based 社会智能评测基准，包含 90 个双人社交场景，涵盖谈判、合作、竞争等七维社交能力。
- **Sotopia-Hard**：Sotopia 的挑战子集，包含 14 个具有复杂利益冲突和社会张力的高难度场景，用于测试方法在极端情况下的鲁棒性。

### ⚙️ 实验设置

- **任务形式**：两智能体（agent）在给定 persona 和目标下进行最多 20 轮的开放对话。
- **策略空间**：预定义 12 种基于社会科学理论的策略指令（见 Table 5），例如：
  - Integrative Negotiation（整合式谈判）
  - BATNA Leverage（最佳替代方案施压）
  - GRIT Strategy（渐进缓和策略）
  - Active Listening（主动倾听）
- **交互协议**：
  - 每轮从策略集中选择一条指令附加到 persona 后形成最终 prompt
  - 使用 DeepSeek-V3.2 或 Qwen-2.5-72B-Instruct 生成回复
  - 每轮由 LLM evaluator（shaping reward）提供多维度反馈，用于 online 更新

### 📊 评估指标

Sotopia 定义七个维度，分数归一化后加权平均得到 Overall Score：

| 维度 | 缩写 | 描述 |
|------|------|------|
| Goal | GOAL | 私人目标达成程度 [0,10] |
| Relationship | REL | 关系质量变化 [-5,5] |
| Knowledge | KNO | 获取新信息量 [0,10] |
| Secret | SEC | 秘密保护能力 [-10,0] |
| Social Rules | SOC | 遵守社会规范 [-10,0] |
| Financial & Material | FIN | 物质收益 [-5,5] |
| Believability | BEL | 行为自然可信度 [0,10] |

最终性能由 **GPT-4o** 进行 episode-level 统一评判，确保公平比较。

### 🆚 基线方法对比

| 方法 | 类型 | 是否在线 | 是否使用 LLM optimizer |
|------|------|----------|-------------------------|
| Vanilla | 静态 persona | ❌ | ❌ |
| OPRO | 在线 Prompt Opt. | ✅ | ✅（LLM 生成新 prompt） |
| EvoPrompt | 进化算法优化 | ✅（每5轮） | ✅（mutation/crossover） |
| INSTINCT | Neural Bandit | ✅ | ❌（基于 UCB） |
| ALSO (Ours) | Adversarial Bandit + Surrogate | ✅ | ❌ |

所有方法共享相同的策略池和 reward query budget，保证可比性。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2）

| 方法 | SOTOPiA-ALL Overall | SOTOPiA-HARD Overall | SOTOPiA-HARD REL |
|------|--------------------|------------------------|------------------|
| Vanilla | 3.62 | 3.02 | 1.32 |
| OPRO | 3.69 | 3.24 | 1.89 |
| EvoPrompt | 3.74 | 3.29 | 1.93 |
| INSTINCT | **3.85** | 3.43 | 2.16 |
| **ALSO (Ours)** | **3.89** | **3.53** | **2.43** |

#### ✅ 对比优势总结：

- **Overall 性能提升**：
  - 在 Sotopia-Hard 上比最强基线（INSTINCT）提升 **+2.92%**（3.53 vs 3.43）
  - 比 Vanilla 提升 **+16.60%**
- **关系维度巨大突破**：
  - REL 得分从 1.32（Vanilla）提升至 **2.43**，相对提升 **+83.79%**
  - 超过最佳基线 **+12.59%**（2.43 vs 2.16）
- **Goal 与 Knowledge 同步改善**：
  - Goal 达成率提升至 7.11（+2.79% over best baseline）
  - Know 得分为 5.47（+0.52%），排除“只顾关系不顾目标”的 trade-off 可能

### 🔬 消融实验结果（Table 3）

| 变体 | Overall | REL | 分析 |
|------|--------|-----|------|
| ALSO (Full) | 3.91 | 3.07 | 完整模型 |
| w/o EXP3 (e-greedy) | 3.61 | 2.71 | 固定探索不足，易被对手利用 |
| w/o Score Smoothing | 3.57 | 2.25 | 无法追踪动态变化，REl 下降明显 |
| w/o Context Embedding | 3.51 | 2.64 | 缺乏上下文感知，适应性差 |
| **w/o Neural Surrogate** | **3.33** | **2.00** | **影响最大，证明 surrogate 至关重要** |

> 结论：**Neural Surrogate 是最关键组件**，去除后 Overall 下降 0.58，REL 下降 34.9%；Score Smoothing 对关系维护特别重要。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **非平稳性普遍存在**  
   图 4 显示，即使固定同一策略，其每轮 reward 也随时间剧烈波动（方差达 0.004–0.015），证实了社会交互中的强耦合与共演特性，传统 stationary bandit 假设失效。

2. **动态策略切换可打破僵局**  
   如图 3 所示，在高冲突驱逐场景中，Vanilla 方法陷入“坚持-拒绝”死循环；而 ALSO 成功通过 `Validate Before Redirecting` 和 `GRIT` 策略引导对方让步，实现合作解决（reward 从 0 → 0.89）。

3. **双边优化优于单边适应**  
   当两个 agent 都启用 ALSO 时，Overall 性能得到显著提升（p < 0.01），尤其在 REL 和 KNO 维度，说明对称适应更符合真实社交动态。

4. **具备跨场景泛化能力**  
   在 cross-scenario generalization 实验中，zero-shot transfer 在未见测试场景上仍优于从头学习：
   - Goal 得分：7.14 vs 6.79 (**+5.3%**)
   - Overall 得分：3.60 vs 3.17 (**+13.5%**)

5. **异构模型配对依然有效**  
   在 DeepSeek、Qwen、GPT-4o-mini 的混合 pairing 中，ALSO 均带来一致增益（+4% ~ +28%），表明其优化效果不依赖特定 backbone。

### ⚠️ 局限性

- **策略空间依赖人工设计**：当前 12 种策略基于社会学理论手动构建，虽可通过 paraphrase 扩展，但尚未实现完全自主的策略发现。
- **exploration-exploitation 平衡受限于交互长度**：在短对话（<10 turns）中难以充分探索大策略空间（见 D.1.1，pool size >24 时性能下降）。
- **surrogate model 可能过拟合**：实验显示两层 MLP 在某些情况下不如单层（Table 14），需谨慎设计容量。

### 🔮 未来工作方向

1. **动态扩展策略空间**：利用 surrogate 的 reward 估计识别高频成功模式，并触发 LLM 自动生成新策略变体（已在 D.1.3 初步验证可行性）。
2. **引入元学习机制**：将在多个场景中学到的策略偏好迁移至新场景，进一步提升冷启动性能。
3. **多智能体联合策略优化**：探索通信协议或协调信号，使 agents 能协同演化而非独立优化。
4. **应用于真实人类-LLM 交互场景**：如客服、心理咨询、教育辅导等，检验实际部署价值。

---

## 总结

✅ **ALSO 是首个专为非平稳社会交互设计的在线策略优化框架**，它通过 **adversarial bandit + neural surrogate** 的组合，实现了高效、灵活且无需微调的动态适应能力。实验证明其在复杂社会任务中显著优于现有方法，尤其在**关系建立与冲突化解**方面取得突破性进展，为构建真正具备社会智能的 LLM agents 提供了一条可扩展、低成本的技术路径。

</details>

---

### 13. [An efficient multi-GPU implementation for the Discontinuous Galerkin ocean model SLIM](https://arxiv.org/abs/2605.16082)

**Authors**: Miguel De Le Court, Vincent Legat, Ange P. Ishimwe, Colin Scherpereel, Emmanuel Hanert, Jonathan Lambrechts  
**Category**: cs.DC  
**Published**: 2026-05-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.16082v1  

#### Abstract
Unstructured-mesh ocean models are increasingly used for coastal applications due to their ability to represent complex geometries and apply local grid refinement where needed. However, their broader use has been hindered by their high computational cost, particularly for models based on the Discont...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An efficient multi-GPU implementation for the Discontinuous Galerkin ocean model SLIM

## 1. 论文的主要贡献和创新点

### 解决的问题
- **高计算成本限制了非结构网格海洋模型的应用**：尽管 Discontinuous Galerkin finite element (DG-FE) 方法在几何灵活性和局部加密方面具有优势，但其高昂的计算开销使其难以应用于大规模、高分辨率的沿海模拟。
- **传统CPU架构无法满足现代高性能计算需求**：现有主流海洋模型多基于CPU实现，难以充分利用当前以GPU为核心的HPC系统。

### 提出的新方法与创新
- **完整的多GPU优化实现**：首次为三维非结构网格DG海洋模型SLIM提供全面支持单GPU和多GPU系统的实现，兼容NVIDIA和AMD架构。
- **高效的内存布局设计**：
  - 采用 **Structure-of-Arrays (SoA)** 数据格式提升内存合并访问效率；
  - 引入 **cell layout** 结构用于列式隐式求解器，确保线性系统求解时的共址内存访问；
  - 对2D网格按 **Hilbert曲线** 重排序，提高缓存命中率。
- **矩阵自由求解器 (matrix-free solvers)**：针对垂直压力梯度 $ r $ 和垂直速度 $ w $ 的方程，利用其可预测的稀疏结构，避免显式组装大型矩阵，显著减少内存占用和计算时间。
- **轻量级抽象层**：通过统一代码库支持CPU、CUDA和HIP后端，保证跨平台可移植性。

### 相比现有方法的优势
| 特性 | 本工作 (SLIM) | 典型传统模型（如NEMO、ROMS） |
|------|----------------|-------------------------------|
| 架构支持 | 多GPU（NVIDIA/AMD），CPU | 主要为CPU，部分GPU加速组件 |
| 并行效率 | 弱扩展至1024 GPUs仍保持高效 | 通常局限于数百核级别 |
| 内存利用率 | 达到峰值带宽的80%（内存密集型内核） | 一般较低，受限于不规则内存访问 |
| 时间步长处理 | 支持IMEX Runge-Kutta方案，分离快慢模态 | 多数为显式或半隐式 |
| 几何适应性 | 非结构三角形/棱柱网格，支持局部加密 | 多为结构化经纬度网格 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **合成基准测试配置**：用于评估弱缩放和强缩放性能，不依赖外部地理数据。
- **真实世界案例：大堡礁 (Great Barrier Reef, GBR)**：
  - 网格规模：约330万三角形，垂直方向最多29层，总计约3400万个棱柱单元；
  - 分辨率：近岸区域达200米，远海粗化至10公里；
  - 强制输入来源：
    - 海底地形：Geoscience Australia提供的30m–100m分辨率数据；
    - 开边界条件：BRAN2023再分析数据；
    - 潮汐强迫：TPXO10v2；
    - 大气强迫：BARRA2-C2区域大气再分析数据；
    - 珊瑚分布：Allen Coral Atlas 和 UNEP-WCMC全球珊瑚图谱。

### 实验设置与评估指标
- **硬件平台**：
  - **MeluXina集群**：配备NVIDIA A100 GPU（每节点4块），使用非GPU感知OpenMPI；
  - **LUMI-G集群**：配备AMD MI250X GPU（每节点4块，共8个GCD），使用GPU感知Cray MPICH。
- **评估指标**：
  - 单步迭代时间（iteration time）
  - 加速比（speedup）vs CPU核心等效值
  - 弱缩放效率（weak-scaling efficiency）
  - 强缩放行为是否符合Amdahl定律
  - 物理-数值时间比（physical-to-numerical time ratio）

### 基线方法对比
- **间接对比对象**：
  - **CPU等效性能**：将单个HPC级GPU性能与“虚拟”单核CPU进行比较；
  - **典型CPU节点**：128核CPU节点 vs 4×A100 GPU节点；
  - 同类GPU模型参考：如Veros、Oceananigans.jl、LICOM3-HIP，用于说明GPU原生设计的优势。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 数值 | 说明 |
|------|------|------|
| **单GPU性能** | 一块NVIDIA A100 ≈ **1500 CPU cores** | 在足够大的问题下，相对于双精度单核CPU的等效性能 |
| **节点级加速比** | 4×A100 GPU 节点 vs 128核CPU节点 → **~50× speedup** | 显著优于传统CPU节点 |
| **弱缩放效率** | 维持至 **1024 GPUs (512 nodes)** | 在LUMI上达到极高并行规模仍保持良好效率 |
| **内存带宽利用率** | 最高达 **80% peak bandwidth**（内存密集型内核） | 表明高度优化的内存访问模式 |
| **浮点吞吐利用率** | 最高达 **60% peak FLOPs**（计算密集型内核） |
| **平均资源利用率** | 完整时间步中持续维持约 **30% peak compute & bandwidth** | 对低阶DG方法而言非常优秀 |

### 与基线方法的对比结果
- **相比传统CPU实现**：
  - 性能提升达数十至上百倍；
  - 实现了以前不可行的超高分辨率模拟（如GBR全域亚礁尺度模拟）。
- **与其他GPU模型比较**：
  - 效率水平接近专为GPU设计的新一代模型（如Oceananigans.jl）；
  - 证明即使复杂非结构DG方法也能达到类似结构化网格模型的硬件利用率。

### 消融实验结果（隐含分析）
- **不同垂直层数的影响**：
  - 层数为2^n（如16、32、64）时性能更优，因能更好匹配线程块大小（block size = 128）；
  - 非整除情况导致线程空闲和内存访问不连续，造成性能下降。
- **水平分辨率影响**：
  - 当DG节点数低于 ~10⁵–10⁶ 时，GPU利用率饱和，迭代时间趋于恒定；
  - 原因为小规模下2D内核受kernel launch latency主导。
- **通信重叠策略有效性**：
  - 利用双stream机制（compute stream + communication stream）有效隐藏部分MPI通信开销；
  - 对3D重计算内核效果明显，但对短2D内核仍有瓶颈。

---

## 4. 关键结论和发现

### 主要发现
1. **DG-FE方法天然适合GPU架构**：
   - 高算术强度（arithmetic intensity）、元素独立性、局部性强等特点使其非常适合大规模并行计算。
2. **GPU可彻底改变非结构网格模型的实用性**：
   - 将原本昂贵的DG模型从“研究原型”转变为可用于实际业务模拟的工具；
   - 实现了 **sub-reef scale resolution** 下的大范围三维模拟（如整个大堡礁）。
3. **性能瓶颈来自2D barotropic模式**：
   - 尽管3D部分近乎理想扩展，但频繁调用的小型2D内核成为强缩放的“顺序瓶颈”，符合 **Amdahl's Law**；
   - 这是多数分裂时间步海洋模型的共性挑战。
4. **一次代码、多平台运行可行**：
   - 通过轻量抽象层实现了CUDA/HIP/CPU统一代码库，兼顾性能与可维护性。

### 方法的局限性
- **小规模问题性能不佳**：
  - 当网格太小时，GPU未被充分占用，性能受限于kernel launch延迟；
  - 不适用于极小域或极低分辨率场景。
- **垂直层数敏感性**：
  - 性能随层数变化呈现非单调波动，需谨慎选择参数以获得最佳性能。
- **尚未科学验证**：
  - 大堡礁案例主要用于演示计算可行性，**未经过观测数据校准或验证**，不能作为科学研究依据。
- **TPXO潮汐数据不可公开分发**：
  - 公共示例使用简化潮汐强迫，可能影响动力细节的真实性。

### 未来工作方向
- **进一步优化2D内核**：
  - 探索kernel fusion、persistent threads等技术降低短内核开销；
  - 或引入更高阶时间积分以减少外模式迭代次数。
- **支持更大规模异构系统**：
  - 扩展至Exascale级别系统，探索数千GPU协同仿真能力。
- **增强物理过程耦合**：
  - 集成生物地球化学模块、波浪-流相互作用、沉积物输运等多物理场过程。
- **发展自动化网格生成与自适应算法**：
  - 结合local mesh refinement与adaptive time stepping，实现智能资源分配。
- **推动业务化应用**：
  - 将该高性能框架应用于环境监测、灾害预警、生态管理等实际场景。

> ✅ **一句话总结**：本文展示了通过深度GPU优化，原本被认为“过于昂贵”的DG-FE海洋模型可以实现前所未有的性能飞跃，在保持几何灵活性的同时达到与结构化模型相当甚至超越的计算效率，为下一代高分辨率沿海模拟开辟了道路。

</details>

---

### 14. [Federated Learning of Spiking Neural Networks under Heterogeneous Temporal Resolutions](https://arxiv.org/abs/2605.15355)

**Authors**: Sanja Karilanova, Subhrakanti Dey, Ay\c{c}a \"Oz\c{c}elikkale  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.15355v1  

#### Abstract
Spiking neural networks (SNNs) are biologically inspired energy-efficient models that use sparse binary spike-based communication between neurons, making them attractive for resource-constrained edge devices. Federated learning enables such devices to train collaboratively without sharing raw data. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Federated Learning of Spiking Neural Networks under Heterogeneous Temporal Resolutions**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在联邦学习（Federated Learning, FL）场景中，边缘设备由于硬件、能耗等限制，采集的时间序列数据往往具有**异构的时间分辨率（heterogeneous temporal resolutions）**。这种时间尺度上的不匹配会导致模型参数在不同客户端之间难以直接聚合，从而严重影响全局模型性能。

现有研究主要关注数据分布异构性（如类别不平衡），但**忽略了时间分辨率差异带来的挑战**，尤其是在 Spiking Neural Networks（SNNs）这类依赖时序动态的状态神经网络中。

---

### **提出的新方法与新思路**
作者提出了 **FedTA（Federated Learning with Temporal Resolution Adaptation）框架**，专门用于处理 SNNs 和更广泛的状态神经网络在异构时间分辨率下的联邦学习问题。

#### **核心创新点：**
- **引入时间分辨率自适应机制**：在每一轮联邦平均（FedAvg）过程中，客户端上传本地模型前，将其神经元动力学参数从本地时间分辨率 $ T_k $ 映射到服务器设定的统一分辨率 $ T_c $；服务器下发全局模型时再反向映射回客户端本地分辨率。
- **三种动态适配方法**：
  - **FedTA-Int**：基于积分形式的时间分辨率转换，适用于线性 SSM 动力学。
  - **FedTA-Eul**：基于欧拉近似的转换方法，为一阶近似。
  - **FedTA-△**：针对显式建模时间分辨率参数 $ \Delta_{\text{log}} $ 的模型（如 △-LIF / △-SSM），通过加法偏移实现快速调整。

该方法允许每个客户端以自身最优的时间分辨率训练模型，同时保持与全局模型兼容。

---

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **通用性** | 可应用于 SNNs 和 Deep State-Space Models（SSMs）等多种状态神经网络架构。 |
| **有效性** | 显著缓解因时间分辨率不一致导致的精度下降，尤其在高异构场景下表现优异。 |
| **低开销** | 所有适配操作仅作用于动力学参数 $ \phi $，不影响权重 $ W $ 或归一化统计量，计算开销极小。 |
| **隐私保护** | 不需要共享原始数据或修改客户端采样策略，符合 FL 隐私原则。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
两个主流的类脑计算基准数据集：
- **SHD (Spiking Heidelberg Digits)**：音频分类任务，包含 20 类（0–9 英语+德语），共 8156 训练样本、2264 测试样本。
- **DVS-Gesture**：事件相机手势识别任务，11 类手势，1086 训练样本、256 测试样本。

> 数据预处理采用 `Tonic` 库中的 `ToFrame` 转换，将脉冲流分帧输入网络。

---

### **实验设置**
- **客户端配置**：
  - **Scenario A**：3个客户端，分别运行 $ T \in \{1,2,4\} $ 分辨率（$ T=1 $ 最细）
  - **Scenario B**：15个客户端（每种分辨率各5个），增强聚合难度
- **模型架构**：
  - 多层隐藏层堆叠，使用 BatchNorm + AdamW 优化器
  - 超参数详见附录 Table 2
- **评估指标**：
  - **测试准确率（Test Accuracy）**
  - **总训练时间、能量消耗（Estimated Energy）、MAC 操作数**

---

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **FedAvg** | 标准联邦平均，无时间分辨率适配 |
| **FedTA-Int** | 积分式适配 |
| **FedTA-Eul** | 欧拉式适配 |
| **FedTA-△** | 显式 $ \Delta $ 参数偏移适配 |
| 对比神经元类型：
  - **standard-LIF / △-LIF**
  - **standard-SSM / △-SSM**

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

#### **(a) SHD 数据集 — Scenario A ($ T=2 $)**
| 方法 | 准确率 (%) |
|-------|------------|
| FedAvg (standard-SSM) | 87.4 ± 0.9 |
| **FedTA-Int (standard-SSM)** | **90.2 ± 0.6** ✅ |
| FedTA-Eul (standard-SSM) | 87.6 ± 1.7 |
| FedAvg (△-LIF) | 83.1 ± 0.6 |
| FedTA-△ (△-LIF) | 83.3 ± 0.4 |

> ✅ **FedTA-Int 在 standard-SSM 上取得最高精度**

#### **(b) DVS-Gesture 数据集 — Scenario A ($ T=1 $)**
| 方法 | 准确率 (%) |
|--------|-------------|
| FedAvg (△-LIF) | **87.5 ± 1.9** ✅ |
| FedTA-△ (△-LIF) | 86.6 ± 1.4 |
| FedTA-Int (standard-SSM) | 91.1 ± 1.1 |
| FedAvg (standard-SSM) | 79.3 ± 6.9 |

> ⚠️ 尽管 FedTA-Int 在 SSM 上表现好，但在实际应用中 **△-LIF + FedAvg 组合达到最佳平衡**

---

### **与基线方法的对比结果**
| 观察项 | 结果 |
|--------|------|
| **FedTA-Int vs FedAvg (standard-SSM)** | 在 SHD 上平均提升 ~3–8%，显著优于其他方法 |
| **FedTA-Eul** | 性能弱于 FedTA-Int，因其仅为一阶近似 |
| **FedTA-△ vs FedAvg (△-LIF)** | 表现相近，说明 △ 参数本身已具备一定鲁棒性 |
| **FedTA-△ vs FedAvg (△-SSM)** | 在粗粒度 $ T=4 $ 下性能崩溃（仅 32.9%），表明其对极端分辨率变化敏感 |

---

### **消融实验结果**
#### **(1) 时间分辨率中心选择的影响**
- 当 $ T_c = 1 $（最细粒度）或 $ T_c = 2 $（中间值）时，整体性能最优。
- 原因：最小化客户端与服务器之间的平均时间失配。

#### **(2) 客户端数量影响**
- 从 Scenario A（3 clients）→ Scenario B（15 clients）：
  - 平均准确率下降（聚合更困难）
  - 但趋势一致，验证方法可扩展性

#### **(3) 后处理适配 vs 每轮适配**
引入 **FedTA-Int-Post** 等“仅在最后一步适配”的变体：
- 发现 **每轮适配 > 后处理适配**
- 原因：持续的时间一致性有助于训练稳定性

---

## **4. 关键结论和发现**

### **主要发现**
1. **FedTA-Int 是最有效的适配方法**  
   特别适用于具有线性状态转移的动力学模型（如 standard-SSM），能有效恢复因时间错配丢失的精度。

2. **SNN 模型（如 LIF）天然更具鲁棒性**  
   - 即使不使用复杂适配，standard-LIF 和 △-LIF 在多种分辨率下仍保持稳定性能。
   - 更适合资源受限的边缘部署。

3. **能量效率方面，SNN 明显占优**  
   - △-LIF + FedAvg 实现 **44× 能耗降低** 和 **2.1× 更快训练速度**，仅牺牲约 10% 准确率。
   - 归功于脉冲稀疏性和二值激活带来的低 MAC 成本。

4. **适配频率很重要**  
   每轮通信都进行时间适配的效果明显优于只在最终阶段进行一次适配。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **非线性动力学近似误差大** | FedTA-Int/Eul 基于线性 SSM 推导，在 LIF 这类非线性模型上仅为近似，效果有限 |
| **未结合其他异构性** | 仅考虑时间分辨率差异，未处理类别不平衡、域偏移等问题 |
| **依赖特定参数化设计** | FedTA-△ 要求模型显式学习 $ \Delta $ 参数，限制了通用性 |

---

### **未来工作方向**
1. **开发面向非线性神经元的动力学适配理论**
   - 如针对 LIF、ALIF 等 SNN 模型设计专用的时间变换规则
2. **联合建模多维异构性**
   - 将时间分辨率异构与数据分布异构（class imbalance, non-IID）统一建模
3. **探索更高效的适配机制**
   - 引入轻量级适配模块（adapter layers）而非全参数重映射
4. **扩展至 Transformer-like SNN 架构**
   - 如 Spike-driven Transformers、Event-synchronous models

---

> ✅ **总体评价**：本文首次系统地解决了联邦学习中**时间分辨率异构性**这一被忽视的关键问题，提出的 FedTA 框架在准确性与能效之间实现了良好权衡，是推动 SNN 在真实边缘环境中落地的重要一步。

</details>

---

### 15. [ITGPT: Generative Pretraining on Irregular Timeseries](https://arxiv.org/abs/2605.16069)

**Authors**: Antoine Honor\'e, Ming Xiao  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.16069v1  

#### Abstract
Timeseries regression models often struggle to leverage large volumes of labeled multimodal data, particularly when the data are irregularly sampled or contain missing values. This is common in domains like healthcare and predictive maintenance, where data are collected from unreliable sources, and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ITGPT: Generative Pretraining on Irregular Timeseries

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
该论文针对**多模态、不规则采样时间序列**（irregularly sampled multimodal timeseries）建模中的三大挑战：
- 数据采集频率不同且异步（如医疗中ECG每秒500次 vs 血气分析每天一次）
- 存在大量缺失值和不规则时间间隔
- 标签稀缺（label scarcity），尤其在医疗和预测性维护（PdM）领域

传统方法通常依赖**重采样**（resampling）、**特征融合**（feature fusion）或**显式插补**（data imputation），这些操作会引入噪声、计算开销大，且无法有效利用无标签数据。

---

### ✅ 提出了什么新方法或新思路
提出 **ITGPT** —— 一种基于 **Transformer 的注意力架构**，专为处理不规则多模态时间序列设计，并支持：
- **自监督学习**（Self-Supervised Learning, SSL）
- **生成式预训练**（Generative Pretraining, GPT-style objectives）

其核心思想是扩展作者先前提出的 **ITNet** 架构，构建一个编码器-解码器结构：
- **Encoder**：将多模态不规则输入映射到统一“锚点时间线”（anchor timeline）上的隐表示
- **Decoder**：从锚点时间线重构原始多模态输入
- 支持堆叠多个 Encoder-Decoder 层形成深层模型

该框架允许使用 **MSE-based one-step-ahead 预测损失**作为 SSL/GPT 目标，在无标签数据上进行预训练。

---

### ✅ 相比现有方法的优势
| 优势 | 说明 |
|------|------|
| **无需重采样/插补** | 直接处理原始不规则时间戳，避免信息失真 |
| **支持 SSL 和 GPT 训练** | 可充分利用大规模无标签数据提升性能 |
| **端到端可微分** | 基于 attention 机制实现跨模态对齐与融合 |
| **适用于低标签场景** | 在仅有少量标注样本时显著优于纯监督方法 |

相比 RNN + Neural ODE 或 Gaussian Process 方法，ITGPT 具有更强的扩展性和并行化潜力。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### (1) **TIHM Dataset**（医疗健康监测）
- **任务**：远程监控痴呆患者，检测异常健康事件（如躁动、血压异常等）
- **模态**：包括 PIR 传感器、门磁、睡眠垫、智能秤、体温计等共 15 种生理与行为信号
- **特点**：高度不规则采样，部分模态每日仅记录一次
- **标签**：临床验证的二分类事件标签（共6类，其中一类为删失数据）

#### (2) **CompX Dataset**（预测性维护）
- **来源**：Scania 卡车车队的真实运行数据（28,596 辆车）
- **任务**：预测关键组件 CompX 的故障时间窗口（6 分类）
- **模态**：107 个特征分为 15 个模态，包括：
  - 直方图变量（histogram-encoded）
  - 数值计数器（numerical counters）
  - 车辆规格（categorical specs）
- **时间单位**：以组件使用时长划分时间轴
- **标签**：维修记录定义的故障状态（含删失数据 Class 6）

---

### 🎯 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| **训练策略对比** | <ul><li>`CE`：仅用交叉熵损失（纯监督）</li><li>`CE+SSL`：联合使用 CE + MSE 自监督损失</li><li>`GPT→CE`：先用 MSE 预训练，再用 CE 微调</li></ul> |
| **交叉验证** | TIHM：timeseries split CV；CompX：5-fold CV |
| **评估指标** | <ul><li>TIHM：Recall（Sensitivity）、Specificity、AUROC、F1-score</li><li>CompX：AUPRC（Area Under Precision-Recall Curve），更适合不平衡多分类任务</li></ul> |
| **消融实验** | 变化模型深度（1–7层）、dropout 概率（0–30%）、mixing layer 类型（Linear / MLP/1 / MLP/2） |

---

### 🔁 基线方法对比
文中未直接列出与其他 SOTA 模型的全面对比表格，但通过以下方式体现优势：
- 与原始论文 [13] 中使用线性模型 + 线性插值的结果比较（TIHM 上 ~0.8 AUROC）
- 强调 ITGPT 在**不进行任何插值或特征工程的前提下**达到相近甚至更优性能
- 显示在低标签设置下，`CE+SSL` 和 `GPT→CE` 显著优于纯 `CE`

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ TIHM 医疗数据集结果（表 II）
| 方法 | Recall | Specificity | AUROC | F1-score |
|------|--------|-------------|-------|----------|
| CE | 0.68 ± 0.14 | 0.77 ± 0.08 | 0.78 ± 0.05 | 0.25 ± 0.11 |
| CE+SSL | **0.73 ± 0.17** | 0.76 ± 0.09 | **0.79 ± 0.06** | **0.26 ± 0.11** |
| GPT→CE | 0.60 ± 0.22 | 0.78 ± 0.10 | 0.76 ± 0.06 | 0.23 ± 0.12 |

> 💡 小结：`CE+SSL` 在 Recall 和 AUROC 上略有提升，表明 SSL 有助于捕捉长期模式。

---

#### ✅ CompX 预测性维护结果（图 4）
- 最佳配置可达 **AUPRC ≈ 0.44**
- 性能排序：  
  `MLP/2 mixing + dropout=10% + depth=3/6` ≈ `MLP/1 + dropout=20% + depth=6/7` > `Linear mixing`（最高 0.42）

##### 在低标签设置下的表现（图 4c）：
| 方法 | 少量标签时的表现 | 趋势 |
|------|------------------|------|
| CE | 起始低，随标签增加缓慢上升 | 基线 |
| CE+SSL | 初始更高，增长最快（85辆车时达峰值） | **最佳小样本性能** |
| GPT→CE | 初始较高，但在超过 ~171 标签后轻微下降 | 可能过拟合 |

> 当只有 **28 辆车有标签**（约 0.1%）时，所有模型 AUPRC ≈ 0.2；而 `CE+SSL` 在 0.3% 标签时即接近饱和。

---

### 🔍 消融实验结果

| 因素 | 发现 |
|------|------|
| **模型深度（Depth）** | <ul><li>Depth=1 时 dropout 几乎无帮助</li><li>Depth ≥ 2 时，dropout 显著缓解过拟合</li><li>更深模型（depth=6~7）配合 MLP mixing 可达最优性能</li></ul> |
| **Mixing Layer** | <ul><li>MLP-based mixing（尤其是 MLP/2）优于 Linear</li><li>非线性混合更利于高维特征融合</li></ul> |
| **Dropout** | <ul><li>10%-20% dropout 对大多数配置有益</li><li>>30% 导致性能下降，尤其在 MLP mixing 中</li></ul> |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **ITGPT 成功实现了对不规则多模态时间序列的 GPT-style 预训练**，无需重采样、插补或复杂特征工程。
2. 在标签稀缺场景下，**结合 SSL 的训练策略（特别是 CE+SSL）显著优于纯监督学习**，验证了无标签数据的有效利用。
3. 深层 ITGPT 模型具备更强表达能力，但需配合 dropout 控制过拟合。
4. **MLP-based mixing layers 比线性层更能挖掘模态间复杂关系**，适合高维异构输入。
5. 在真实工业数据（CompX）上取得 SOTA 级别 AUPRC（0.44），证明其实际应用价值。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **计算复杂度** | 注意力机制仍存在 $O(n^2)$ 复杂度，处理超长序列效率受限 |
| **缺乏解释性** | 当前 mixing layer 难以追踪各模态贡献，影响可信度 |
| **batching 效率低** | 不规则时间戳导致 padding 多，影响 GPU 利用率 |
| **未探索稀疏注意力** | 存在优化空间，如局部注意力或可学习稀疏连接 |

---

### 🔮 未来工作方向（原文提及）
1. **提升可解释性**：设计 unitary mixing layers 追踪单个模态的影响路径
2. **降低计算成本**：
   - 探索更高效的 attention 机制
   - 开发模态并行 CUDA kernels
3. **优化 batching 与稀疏计算**：
   - 使用 sparse dot-product 减少无效 attention
   - 动态 batching 策略适应不规则长度
4. **拓展至其他领域**：如金融时序、环境监测等同样面临不规则采样的场景

---

## ✅ 总结

| 维度 | 内容 |
|------|------|
| **核心创新** | 首个支持 GPT-style 预训练的 irregular timeseries 模型 ITGPT |
| **关键技术** | 基于 ITNet 的 encoder-decoder 结构 + 锚点时间线 + SSL/GPT 目标 |
| **最大优势** | 无需插补/重采样，高效利用无标签数据，特别适合低标签工业场景 |
| **实证效果** | 在 TIHM 和 CompX 上均表现出色，AUPRC 达 0.44，显著优于纯监督 baseline |
| **未来潜力** | 为工业 IoT 与智慧医疗中的大规模非结构化时间序列分析提供新范式 |

> 🔗 代码已开源：[https://github.com/antoinehonore/itgpt](https://github.com/antoinehonore/itgpt)

</details>

---

### 16. [Nudging Beyond the Comfort Zone: Efficient Strategy-Guided Exploration for RLVR](https://arxiv.org/abs/2605.15726)

**Authors**: Chanuk Lee, Sangwoo Park, Minki Kang, Sung Ju Hwang  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.15726v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has emerged as a scalable paradigm for improving the reasoning capabilities of large language models. However, its effectiveness is fundamentally limited by exploration: the policy can only improve on trajectories it has already sampled. While in...

---

### 17. [Designing Dense Satellite Clusters for Distributed Space-based Datacenters](https://arxiv.org/abs/2605.15335)

**Authors**: Jules P\'enot, Hamsa Balakrishnan  
**Category**: cs.DC  
**Published**: 2026-05-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.15335v1  

#### Abstract
Recent proposals for datacenters in sun-synchronous Low Earth Orbit rely on a large number of compute satellites formation-flying in dense clusters. Designing such satellite clusters requires optimizing the satellites' orbital geometry under several safety and operational constraints applied through...

---

### 18. [parallelcbf: A composable safety-filter and auditability framework for tensor-parallel reinforcement learning](https://arxiv.org/abs/2605.15509)

**Authors**: Yijun Lu, Zilei Yang, Yuyin Ma  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.15509v1  

#### Abstract
While Isaac Lab provides massive parallel UAV simulation, OmniSafe and safe-control-gym provide constrained-RL benchmarks, and CBFKit provides control-barrier-function synthesis tooling, no existing framework unifies these capabilities for end-to-end safety-constrained training. ParallelCBF is the f...

---

### 19. [Mind Dreamer: Untethering Imagination via Active Latent Intervention on Latent Manifolds](https://arxiv.org/abs/2605.16030)

**Authors**: Shaojun Xu, Xiaoling Zhou, Yihan Lin, Yapeng Meng, Xinglong Ji, Luping Shi, Rong Zhao  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.16030v1  

#### Abstract
Model-Based Reinforcement Learning (MBRL) leverages latent imagination for sample efficiency, yet remains constrained by Historical Tethering: imagination is typically initialized from observed states. This creates a learning asymmetry, where the world model's manifold discovery outpaces the policy'...

---

### 20. [From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates](https://arxiv.org/abs/2605.15445)

**Authors**: Ruobing Zuo, Hanrui Zhao, Gaolei He, Zhengfeng Yang, Jianlin Wang  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.15445v1  

#### Abstract
Automated proving of polynomial inequalities is a fundamental challenge in automated mathematical reasoning, where rich algebraic structure and a rapidly growing certificate search space hinder scalability. Purely symbolic approaches provide strong guarantees but often scale poorly as the number of ...

---

### 21. [DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding](https://arxiv.org/abs/2605.15542)

**Authors**: Yichao Liu, Huawen Shen, Liu Yu, Shiyu Liu, Zeyu Chen, Yu Zhou  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.15542v1  

#### Abstract
GUI agents powered by Multimodal Large Language Models (MLLMs) have demonstrated impressive capability in understanding and executing user instructions. However, accurately grounding instruction-relevant elements from high-resolution screenshots cluttered with irrelevant UI components remains challe...

---

### 22. [LPDS: Evaluating LLM Robustness Through Logic-Preserving Difficulty Scaling](https://arxiv.org/abs/2605.15393)

**Authors**: Philipp Mondorf, Samuel J. Bell, Jesse Dodge, Dieuwke Hupkes  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.15393v1  

#### Abstract
As large language models (LLMs) are increasingly deployed to perform tasks with minimal human oversight, it is crucial that these models operate robustly. In particular, a model that can solve a given problem should not fail simply because certain entities$\unicode{x2013}$such as names, numbers, or ...

---

### 23. [Rethinking Neural Network Learning Rates: A Stackelberg Perspective](https://arxiv.org/abs/2605.15530)

**Authors**: Sihan Zeng, Sujay Bhatt, Sumitra Ganesh  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.15530v1  

#### Abstract
Neural networks are typically trained with a single learning rate across all layers. While recent empirical evidence suggests that assigning layer-specific learning rates can accelerate training, a principled understanding of the conditions and mechanisms under which non-uniform learning rates are b...

---

### 24. [Variational Autoregressive Networks with probability priors](https://arxiv.org/abs/2605.16020)

**Authors**: Piotr Bia{\l}as, Piotr Korcyl, Tomasz Stebel, Dawid Zapolski  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.16020v1  

#### Abstract
Monte Carlo methods are essential across diverse scientific fields, yet their efficiency is frequently hampered by critical slowing down-a sharp increase in autocorrelation times near phase transitions. Although deep learning approaches, such as neural-network-based samplers, have been proposed to a...

---

### 25. [SDOF: Taming the Alignment Tax in Multi-Agent Orchestration with State-Constrained Dispatch](https://arxiv.org/abs/2605.15204)

**Authors**: Zhantao Wang  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.15204v1  

#### Abstract
Multi-agent orchestration frameworks such as LangChain, LangGraph, and CrewAI route tasks through graph-based pipelines but do not enforce the stage constraints that govern real business processes. We present SDOF, a framework that treats multi-agent execution as a constrained state machine. SDOF op...

---

### 26. [Solvita: Enhancing Large Language Models for Competitive Programming via Agentic Evolution](https://arxiv.org/abs/2605.15301)

**Authors**: Han Li, Jinyu Tian, Rili Feng, Yuqiao Du, Chong Zheng, Chenyu Wang, Chenchen Liu, Shihao Li, Xinping Lei, Yifan Yao, Weihao Xie, Letian Zhu, Jiaheng Liu  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.15301v1  

#### Abstract
Large language models (LLMs) still struggle with the rigorous reasoning demands of hard competitive programming. While recent multi-agent frameworks attempt to bridge this reliability gap, they remain fundamentally stateless: they rely on static retrieval and discard the valuable problem-solving and...

---

### 27. [Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning](https://arxiv.org/abs/2605.15315)

**Authors**: Jingjing Wang, Xiwen Chen, Wenhui Zhu, Huayu Li, Zhengxiao He, Feiyang Cai, Ana S. Carreon-Rascon, Xuanzhao Dong, Feng Luo  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.15315v1  

#### Abstract
LLM-powered coding agents spend the majority of their token budget reading repository files, yet much of the retrieved code is irrelevant to the task at hand. Existing learned pruners compress this context with a single-objective sequence labeler, collapsing all facets of code relevance into one sco...

---

### 28. [CAPS: Cascaded Adaptive Pairwise Selection for Efficient Parallel Reasoning](https://arxiv.org/abs/2605.15513)

**Authors**: Fangzhou Lin, Shuo Xing, Peiran Li, Siyuan Yang, Qianwen Ge, Kazunori Yamada, Ziming Zhang, Haichong Zhang, Zhengzhong Tu  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.15513v1  

#### Abstract
Parallel reasoning, where a generator samples many candidate solutions and an aggregator selects the best, is one of the most effective forms of test-time scaling in large language models, and pairwise self-verification has become its strongest aggregation primitive. Yet pairwise verification carrie...

---

### 29. [See Before You Code: Learning Visual Priors for Spatially Aware Educational Animation Generation](https://arxiv.org/abs/2605.15585)

**Authors**: Yuejia Li, Ke He, Junheng Li, Shutong Chen, Jingkang Xia, Zhiyue Su, Junchi Zhang, Mang Ye  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.15585v1  

#### Abstract
Large language models can generate executable code for educational animations, but the resulting renders often exhibit visual defects, including element overlap, misalignment, and broken animation continuity. These defects cannot be reliably detected from the code alone and become apparent only afte...

---

### 30. [Fully Open Meditron: An Auditable Pipeline for Clinical LLMs](https://arxiv.org/abs/2605.16215)

**Authors**: Xavier Theimer-Lienhard, Mushtaha El-Amin, Fay Elhassan, Sahaj Vaidya, Victor Cartier-Negadi, David Sasu, Lars Klein, Mary-Anne Hartley  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.16215v1  

#### Abstract
Clinical decision support systems (CDSS) require scrutable, auditable pipelines that enable rigorous, reproducible validation. Yet current LLM-based CDSS remain largely opaque. Most "open" models are open-weight only, releasing parameters while withholding the data provenance, curation procedures, a...

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

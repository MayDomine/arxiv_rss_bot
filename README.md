# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-28 09:07:12 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training](https://arxiv.org/abs/2605.27740)

**Authors**: Keqi Deng, Shaoshi Ling, Ruchao Fan, Jinyu Li  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.27740v1  

#### Abstract
Long-context inference in large language models (LLMs) is bottlenecked by the linear growth of the self-attention key-value (KV) cache. Top-k sparse attention alleviates this by loading only a small fraction of the KV cache, but accurately and cheaply estimating cache importance, for both training-f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在长上下文推理中面临 **Key-Value (KV) cache** 的内存瓶颈。随着上下文长度增长至数十万甚至百万 token，KV cache 的线性增长导致自回归解码时的内存带宽压力剧增，成为延迟和吞吐量的主要限制因素。

现有稀疏注意力方法存在两大缺陷：
- **Training-free 方法**（如 Quest、InfLLM）虽无需训练即可部署，但在高稀疏度下性能下降明显；
- **Trainable 方法**（如 NSA、DSA）虽能缓解性能损失，但通常需要额外的辅助损失、路由模块或架构修改，灵活性差且训练成本高。

此外，大多数方法仅在文本模态上验证，对语音等连续模态的泛化能力未知。

---

### 🚀 提出的新方法：UNIQUE
本文提出 **UNIQUE** —— 一种通用的、支持 **training-free 推理** 和 **sparsity-aware 训练** 的 top-k 稀疏注意力框架，其核心创新如下：

#### （1）基于 KV Page 的重要性评分机制（Offset-augmented Page Scoring）
- 将 KV cache 划分为固定大小的 **page**（硬件友好设计），每个 page 用一个标量分数表示其对当前 query 的“关键性”。
- 分数计算公式为：
  $$
  \text{score}(p) = q \cdot \text{mean}_p + \lambda \|q\|_2 \cdot \text{std}_p
  $$
  - `mean_p`：page 内所有 key 的均值向量，代表整体语义；
  - `std_p`：key 向量的标准差（标量化），作为“偏移项”，补偿因平均而被稀释的重要 key；
  - $\lambda = 0.5$ 固定超参，无需学习。

> ✅ 优势：简单高效、无参数、可预计算缓存，适用于任意模态。

#### （2）Soft-mask Sparsity-aware Training 方案
- 在训练阶段引入 **可微分软掩码（sigmoid soft mask）**，以 top-k 边界为阈值构建门控函数：
  $$
  g_p = \sigma\left(\frac{\text{score}(p) - \theta}{T}\right), \quad \theta = \frac{s_{(k)} + s_{(k+1)}}{2}
  $$
- 软掩码作为 additive log-bias 注入 attention softmax，使梯度可反传至 page score，从而优化选择策略。
- **无需任何辅助损失、无需额外参数、无需改变模型结构**。

> ✅ 优势：统一了 training-free 与 fine-tuning 场景，实现端到端适配稀疏模式。

#### （3）高效的 CUDA 实现
- **融合内核（Fused Kernel）**：将 criticality estimation 中的 matmul、offset 加法、group-wise max reduction 融合为单个 CUDA kernel，减少 HBM 访问；
- **Radix-based Top-k Selection**：基于 8-bit 基数排序的选择算法，复杂度 $O(P)$，远快于传统 $O(P \log P)$；
- 支持与 FlashAttention / paged attention 兼容集成。

> ✅ 性能提升：top-k 选择比 PyTorch 快 **4.8×**，比 FlashInfer 快 **2.0×**。

---

### 🔍 相比现有方法的优势

| 特性 | UNIQUE | Quest | H2O | InfLLM | DSA | NSA |
|------|--------|-------|-----|--------|-----|-----|
| Training-free 支持 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Sparsity-aware 训练 | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| 需要辅助损失 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| 需要额外模块/参数 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| 支持多模态 | ✅ | ⚠️仅文本 | ⚠️仅文本 | ⚠️仅文本 | ⚠️仅文本 | ⚠️仅文本 |
| 可逆性（不丢弃 token） | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |

> ✅ UNIQUE 是首个同时满足：**免训练可用 + 可微调优化 + 无额外开销 + 多模态通用** 的稀疏注意力方案。

---

## 2. 核心实验方法和设置

### 📚 数据集

#### 文本任务：
- **LongBench-Pro**：双语（英/中）、1,500 样本、最长 256K tokens，涵盖 11 类真实长上下文任务（问答、推理、检索等）。
- **RULER-32K**：聚焦上下文长度极限测试，含 needle-in-a-haystack（NIH）类任务。
- **LongAlign-10k**：用于 sparsity-aware fine-tuning 的英文长上下文对齐数据集。

#### 语音任务：
- **葡萄牙语 ASR 数据集**：69K 小时语音训练数据，测试集为 ~10 分钟长句。
- 评估指标：
  - **WER**（Word Error Rate）
  - **EER**（Entity Error Rate）：实体完整匹配率，更严格衡量语义完整性。

---

### ⚙️ 实验设置

| 设置项 | 描述 |
|--------|------|
| **模型** | 文本：Ministral-3-8B-Instruct-2512；语音：Qwen3-8B + Conformer 编码器 |
| **Page Size** | $S = 8$ tokens/page |
| **KV Budget** | 默认 512 pages（即最多访问 4,096 tokens） |
| **Baseline 方法** |  
| - Training-free: | Quest, H2O, InfLLM |
| - Trainable: | InfLLM-v2, DSA（本文适配至 GQA） |
| **评估指标** | 准确率（↑）、WER/EER（↓）、latency（↓）、speedup（↑） |
| **硬件平台** | NVIDIA H100 GPU，使用 vLLM / FlashInfer 运行推理 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）Training-free 推理效果（LongBench-Pro & RULER）

| 方法 | Budget | LongBench-Pro (Overall) | RULER-32K (AVG) |
|------|--------|--------------------------|------------------|
| Full Attention | – | 37.70 | 93.15 |
| Quest | 512 | 21.72 | 84.14 |
| H2O | 512 | 29.04 | 56.00 |
| InfLLM | 512 | 34.99 | 73.30 |
| **UNIQUE (Ours)** | **512** | **36.58** | **90.78** |

> ✅ UNIQUE 在两种 benchmark 上均接近全注意力性能（分别恢复 **97.0% 和 97.5%**），显著优于所有 baseline。

#### （2）Sparsity-aware Fine-tuning 效果（LongBench-Pro 英文）

| 方法 | Budget | English Split |
|------|--------|---------------|
| Full Attention (fine-tuned) | – | 36.55 |
| InfLLM-v2 | 512 | 36.11 |
| DSA | 512 | 36.79 |
| **UNIQUE (Ours)** | **512** | **37.25** |

> ✅ UNIQUE 不仅超越 dense baseline，且训练方式最简洁（无 distillation loss 或 indexer）。

#### （3）语音识别性能（ASR，WER ↓）

| 方法 | Budget | Macro-Avg WER |
|------|--------|----------------|
| Full Attention | – | 18.25 |
| Quest | 512 | 21.92 |
| InfLLM | 512 | 57.72 |
| **UNIQUE (Ours)** | **512** | **18.60** |

> ✅ UNIQUE 几乎完全匹配 dense attention 表现，在巴西葡萄牙语 EER 上甚至略有提升（18.91 vs 18.95）。

#### （4）Sparsity-aware Fine-tuning on ASR

| 方法 | Macro-Avg WER |
|------|----------------|
| Full Attention | 18.25 |
| InfLLM-v2 | 19.10 |
| DSA | 20.91 |
| **UNIQUE (Ours)** | **17.89** |

> ✅ 经 fine-tuning 后，UNIQUE **反超 dense 模型**，说明稀疏训练有助于去除冗余噪声。

---

### ⚙️ 解码效率（Speedup）

#### （1）Attention Kernel Latency（H100, 32K context）

| 方法 | 延迟 | Speedup vs FlashInfer |
|------|------|------------------------|
| FlashInfer (Dense) | ~3500 μs | 1× |
| **UNIQUE (Sparse)** | **~308 μs** | **11.4×** |

> ✅ 注意力内核加速达 **11.4×**，且随 context 增长优势扩大。

#### （2）End-to-End Decoding Latency（vLLM, Batch=80）

| 方法 | 延迟 | Speedup vs Dense |
|------|------|------------------|
| Dense vLLM | ~40 ms/token | 1× |
| **UNIQUE** | **~7.5 ms/token** | **5.3×** |

> ✅ 端到端解码速度提升 **5.3×**，每 token 延迟基本恒定。

---

### 🔍 消融实验结果

#### （1）Std Offset 消融（ASR）

| 配置 | Macro-Avg WER |
|------|----------------|
| UNIQUE w/o std | 19.11 |
| **UNIQUE (full)** | **18.60** |

> ✅ 加入 std 偏移项带来 **0.51 点 WER 下降**，证明其有效捕捉 page 内部方差信息。

#### （2）Soft Mask 消融（Fine-tuning）

| 掩码类型 | Macro-Avg WER |
|---------|----------------|
| Hard Mask (Top-k) | 18.32 |
| **Soft Mask (Sigmoid)** | **17.89** |

> ✅ Soft mask 提供可微信号，使模型能主动优化 page selection，进一步提升性能。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **UNIQUE 是首个真正统一 training-free 与 sparsity-aware 训练的稀疏注意力框架**，无需额外损失或结构改动。
2. **Mean + Std 的 page scoring 机制具有强大多模态泛化能力**，在文本与语音任务上均表现优异。
3. **Soft-mask 训练策略能有效缩小 train-inference gap**，甚至超越 dense 模型，表明稀疏本身可作为一种正则化手段。
4. **高效 CUDA 实现带来显著推理加速**：**11.4× attention-kernel speedup** 与 **5.3× end-to-end speedup**，极具实用价值。
5. **相比 token 删除类方法（如 H2O），UNIQUE 保留完整 KV cache，具备可逆性和更强鲁棒性**。

---

### ⚠️ 局限性

1. **未与其他 KV 压缩技术比较**：如 KV quantization、low-rank compression 等正交方法未纳入对比。
2. **评估集中在文本与语音**：尚未扩展至视觉或多模态（VLM）场景。
3. **基于 fine-tuning 范式**：无法与需从头预训练的方法（如 Native Sparse Attention）公平比较。
4. **page granularity 固定**：未探索动态 page 划分或跨层 adaptive k 值。

---

### 🔮 未来工作方向

1. 扩展至 **vision 和 vision-language models**，验证跨模态通用性。
2. 结合 **KV quantization / low-rank approximation**，实现复合压缩。
3. 探索 **pre-training from scratch with UNIQUE**，释放更大潜力。
4. 引入 **adaptive k 或 dynamic page size**，根据 query 复杂度调整稀疏程度。
5. 开源高效 kernel 实现，推动工业级部署。

--- 

> 💡 **总结一句话**：  
> **UNIQUE 通过一个极简而强大的 page-scoring 机制和 soft-mask 训练范式，首次实现了“免训练即用、可微调优化、多模态通吃”的稀疏注意力解决方案，在保持几乎无损性能的同时，带来高达 11.4× 的注意力加速，是迈向高效长上下文推理的重要一步。**

</details>

---

### 2. [HRBench: Benchmarking and Understanding Thinking-Mode Switch Strategies in Hybrid-Reasoning LLMs](https://arxiv.org/abs/2605.28398)

**Authors**: Yansong Ning, Mianpeng Liu, Jingwen Ye, Weidong Zhang, Hao Liu  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.28398v1  

#### Abstract
Hybrid-reasoning large language models (LLMs) expose explicit controls over reasoning effort, allowing users or systems to trade off answer quality against inference cost. However, existing methods for adaptive thinking-mode selection are typically evaluated under different models, datasets, and imp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HRBench: Benchmarking and Understanding Thinking-Mode Switch Strategies in Hybrid-Reasoning LLMs 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前在 **Hybrid-Reasoning LLMs** 中，虽然已有多种 **adaptive thinking-mode switch** 方法（如 Prompt-Tuning、Routing、Speculative），但这些方法通常在不同的模型、数据集、训练流程和评估标准下进行比较，导致无法公平地判断哪种策略更优。这使得研究者难以回答以下关键问题：

- 哪种 thinking-mode switch 策略真正更有效？
- 不同训练方式（如 SFT、RL）对各类策略的影响如何？
- 最优策略是否随模型规模或任务领域变化？

### **提出了什么新方法或新思路**

本文提出 **HRBench** ——一个**统一的评估框架**，用于系统性地研究 Hybrid-Reasoning LLMs 中的 thinking-mode switch 策略。

#### 核心设计维度：
- **三大策略家族**（Strategy Families）：
  - **Prompt-Tuning (PT)**：通过提示词引导模型自主决定推理深度。
  - **Routing (RT)**：先由“路由器”判断难度，再选择模式。
  - **Speculative (SPEC)**：从快速模式开始，检测到不确定性时动态切换至深推理。
  
- **四种训练范式**（Training Regimes）：
  - Training-Free（无训练）
  - Supervised Fine-Tuning (SFT)
  - Offline RL（如 DPO）
  - Online RL（如 GRPO）

由此构建出 **3 × 4 = 12 种受控实验配置**，实现跨策略、跨训练方式的公平比较。

### **相比现有方法的优势**

| 维度 | HRBench 的优势 |
|------|----------------|
| **可比性** | 首个将不同策略置于相同模型、数据、解码参数下的统一平台 |
| **全面性** | 覆盖三大策略 + 四类训练方式，支持消融分析 |
| **复现性** | 开源所有实现代码与配置，提供 `verl` 和 `vLLM` 支持 |
| **实用性** | 揭示了策略选择应结合模型规模与任务领域的实践指导 |

> ✅ **开源地址**：[https://github.com/usail-hkust/HRBench](https://github.com/usail-hkust/HRBench)

---

## 2. 核心实验方法和设置

### **使用的数据集**

共使用 **5 个基准数据集**，覆盖三大任务领域：

| 数据集 | 领域 | 规模 | 特点 |
|--------|------|------|------|
| **MATH500** | 数学 | 500题 | 高中水平数学题 |
| **AIME 2025** | 数学 | 30题 | 竞赛级数学难题 |
| **GPQA-Diamond** | 科学 | 198题 | 研究生级别物理/化学/生物问答 |
| **LiveCodeBench (LCB)** | 编程 | 167题 | 执行验证的真实编程问题 |
| **Codeforces** | 编程 | 366题 | 竞技编程挑战题 |

> ⚠️ 所有任务均采用自动提取答案或 LLM-as-Judge 进行评分。

### **实验设置和评估指标**

#### **评估模型（6个）**
从小到大覆盖主流 Hybrid-Reasoning LLMs：
- Qwen3.5-2B, Qwen3.5-9B
- gpt-oss-20B
- Seed-OSS-36B
- DeepSeek-V3.1-671B
- Kimi-K2.5-1.1T

> 各模型支持不同类型 thinking mode 控制（binary/discrete/numeric budget）。

#### **评估指标**
- **Acc**：Pass@1 准确率（%）
- **Tok**：平均输出 token 数量（含 CoT）

目标是分析 **effectiveness-efficiency trade-off**（准确率 vs 推理开销）。

#### **统一管道**
所有方法均在同一 pipeline 下运行：
- 解码参数一致（greedy decoding, temp=0）
- 使用相同的 prompt template 和 answer format
- 外部方法全部重新实现并集成

### **基线方法对比**

#### **固定基线（Fixed Baselines）**
- **Full-Think**：始终启用深推理
- **No-Think**：始终直接作答
- **Budget-Aware**：手动设定 High/Medium/Low 预算

#### **三大策略代表方法**
| 类型 | 代表方法 |
|------|---------|
| **Prompt-Tuning** | S1, TALE, Budget-Guidance, SoT, CoD, DynaThink, DEER, RASC |
| **Routing** | AdaptThink, HDFlow |
| **Speculative** | MixReasoning, ADR |

> 共整合 **12+ 外部方法**，并在统一条件下重现实验。

---

## 3. 主要实验结果和性能指标

### **关键性能数据汇总**

> 在 Qwen3.5-9B 上的综合表现（Avg Acc / Tok Reduction vs Full-Think）

| 方法类别 | 平均 Acc (%) | Token 节省（vs Full-Think） | 特点 |
|----------|--------------|-------------------------------|------|
| **PT-TF** | 47.6 | +24.4% | “双赢”：提效又降本 |
| **RT-TF** | 44.1 | +12.5% | 稳定节省，保持精度 |
| **Spec-TF (Entropy)** | 45.8 | -29.6% | 更高耗 token，但提升准确率 |
| **RT-GRPO** | 44.6 | **+69.5%** | 训练后效率飞跃 |
| **PT-DPO** | 48.3 | +21.2% | 平衡效果最好 |

> 🔥 **最高效率增益**：**RT-GRPO 实现近 70% token 节省**，远超其他方法。

---

### **与基线方法的对比结果**

#### ✅ Prompt-Tuning 表现最优（整体 Pareto 前沿）
- **PT-TF** 在多数情况下优于 Full-Think，同时减少约 24% token。
- **RASC** 达到最高准确率（53.4%），但 token 成本极高（+406%）。
- **Chain-of-Draft (CoD)** 实现最大 token 节省（+80%），适合低延迟场景。

#### ✅ Routing 方法训练收益最大
- **GRPO 对 RT 提升显著**：token 节省从 12.5%（TF）提升至 **69.5%（GRPO）**。
- 说明路由决策可通过强化学习高效优化。

#### ✅ Speculative 是“准确性增强器”
- 多数 speculative 方法（如 MixReasoning）token 成本更高（-11% ~ -30%）。
- 但在 Code 任务上表现突出（如 ADR 提升 LiveCode 准确率至 34.7%）。

---

### **消融实验结果**

#### （1）**训练方式影响显著且策略依赖性强**

| 训练方式 | 效果特点 |
|---------|--------|
| **SFT** | 平衡改进，适合作为默认起点 |
| **DPO** | 最大化准确率提升（如 PT-DPO 达 48.3%） |
| **GRPO** | 极大提升效率（尤其对 RT，达 69.5% 节省） |

> 💡 发现：**GRPO 对 RT 最有效，因路由决策是离散动作，易于通过奖励信号优化。**

#### （2）**模型规模调节策略有效性**

| 模型大小 | 最佳策略 |
|--------|--------|
| **2B–9B** | Prompt-Tuning |
| **20B, 671B** | **Speculative 反超**（如 671B 上 SPEC 75.8% > PT 74.7%） |
| **1.1T** | Prompt-Tuning 再次领先（80.8%） |

> 📌 结论：**没有单一策略通吃所有尺度**。

#### （3）**任务领域决定最优策略**

| 领域 | 最佳策略 | 原因 |
|------|--------|------|
| **Math / Science** | **Prompt-Tuning** | 自主分配推理资源更高效 |
| **Code** | **Speculative** | “Try-then-verify” 机制利于捕捉边界情况 |

> 示例：在 Codeforces 上，Spec-TF 提升准确率至 32.2%，而 PT-TF 仅 29.5%。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **三种策略具有本质不同的 trade-off 曲线**：
   - **Prompt-Tuning**：实现“双赢”——**更高准确率 + 更少 token**
   - **Routing**：**稳定节省 token（~10–20%）**，适合成本敏感部署
   - **Speculative**：**以更多 token 换取更高准确率**，适用于质量优先场景

2. ✅ **训练收益高度策略依赖**：
   - **GRPO 对 Routing 提升最大**（可达 65–70% token 节省）
   - **DPO 最有利于提升 Prompt-Tuning 的准确性**
   - **Speculative 训练增益有限**，因其决策更细粒度，难被粗粒度奖励优化

3. ✅ **最优策略随模型规模和任务领域转移**：
   - 小模型（<10B）：PT 占优
   - 中等模型（20B–671B）：SPEC 反超
   - 超大规模（1.1T）：PT 再次领先
   - 数学/科学任务：PT 最佳
   - 编程任务：SPEC 更强

4. ✅ **外部方法在统一评估下结论可复现但非普适**
   - 如 RASC 在 AIME 上最强（83.3%），但代价巨大
   - **不存在单一“冠军方法”**，需根据任务定制选择

---

### **方法的局限性**

1. ❌ **训练实验局限于 Qwen3.5-9B**
   - 更大模型（如 20B+）上的训练行为未充分探索
   - 缺乏对训练-规模交互效应的完整建模

2. ❌ **仅考虑单轮推理（single-turn）**
   - 未涵盖多步推理、Agent 场景中的连续 mode-switching

3. ❌ **任务领域有限**
   - 当前聚焦于 Math、Science、Code
   - 创意写作、多语言、对话等领域的 trade-off 可能不同

4. ❌ **Speculative 触发机制存在延迟问题**
   - 如熵值触发太晚，已生成错误路径，回溯无效

---

### **未来工作方向**

1. 🔮 **扩展至多轮与 Agent 场景**
   - 研究动态 workflow 中的 adaptive thinking control

2. 🔮 **开发跨领域通用 switcher**
   - 设计可根据输入自动识别任务类型的 meta-controller

3. 🔮 **探索轻量化 speculative 触发机制**
   - 如 early-exit heads、hidden state probing 等低开销方案

4. 🔮 **构建更大规模的训练实验**
   - 在 20B+ 模型上验证训练策略的泛化能力

5. 🔮 **引入人类偏好评估**
   - 结合主观质量评分，超越纯自动 metric

---

> 🧩 **总结一句话**：  
> **HRBench 揭示了“没有银弹”——thinking-mode switch 的最佳选择取决于模型规模、任务类型与训练方式的复杂权衡，而 Prompt-Tuning、Routing 和 Speculative 分别代表了效率-效果光谱上的三个典型极点。**

</details>

---

### 3. [SiDP: Memory-Efficient Data Parallelism for Offline LLM Inference](https://arxiv.org/abs/2605.28095)

**Authors**: Alan Zhao, Cyril Y. He  
**Category**: cs.DC  
**Published**: 2026-05-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.28095v1  

#### Abstract
The rapid adoption of large language models (LLMs) has shifted a substantial portion of inference workloads into throughput-oriented offline regimes, where fully utilizing GPU compute requires large batch sizes. However, existing deployments face a structural tension. Data parallelism (DP) scales th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SiDP: Memory-Efficient Data Parallelism for Offline LLM Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在离线大语言模型（LLM）推理中，**吞吐量**（throughput）是核心性能指标，而提升吞吐量的关键在于增大有效批处理大小（effective batch size）。然而，当前主流的并行策略面临以下瓶颈：

- **Data Parallelism (DP)** 虽然具备良好的扩展性和调度灵活性，但每个副本都完整复制模型权重，导致大量 GPU 显存（HBM）被静态权重占用，限制了可用于 Key-Value (KV) cache 的空间，从而制约了批处理规模。
- **Model Parallelism (MP)** 虽能分片权重以节省显存，但引入了细粒度同步开销（如 tensor parallelism 中的 all-reduce），破坏了 DP 的独立性，降低了调度灵活性。

因此，论文旨在解决 **“如何在保持 DP 高扩展性的同时，减少冗余权重复制以释放更多显存用于 KV cache”** 这一结构性矛盾。

---

### 提出了什么新方法或新思路
作者提出 **SiDP (Shared-weight Intra-node Data Parallelism)**，一种面向离线 LLM 推理的内存高效型数据并行范式。其核心思想是：

> 将模型权重视为由高带宽互连（如 NVLink）支撑的共享资源，在一个 DP 组内构建分布式权重池。

具体创新设计包括：

1. **分布式权重存储**：
   - 每个 Transformer 层的 FFN 权重由某一个 GPU “拥有”（owner），其他副本按需访问。
   - 显著减少每卡的权重内存占用，释放 HBM 给 KV cache。

2. **双模式执行机制**：
   - **Weight-as-a-Service (WaS)**：适用于大批次场景。非拥有者通过 NVLink 异步预取远程权重到本地缓存，并在本地执行计算，保持计算独立性。
   - **Compute-as-a-Service (CaS)**：适用于小批次尾部阶段。非拥有者将激活值发送给权重拥有者，由后者融合多个请求进行批量 GEMM 计算后返回结果，避免为微小计算频繁拉取权重。

3. **去中心化的峰值偏移（Peak Shifting）**：
   - 为缓解多副本同时从同一 owner 拉取权重造成的 NVLink 热点问题，采用基于层索引循环和 rank 偏移的预取调度策略，实现流量分散。

4. **与现有系统兼容的设计**：
   - 实现为 vLLM 的插件，无需修改原框架代码，利用 CUDA IPC 实现跨设备内存访问。

---

### 相比现有方法的优势
| 方面 | SiDP | 传统 DP | FSDP 类方案 |
|------|------|--------|-------------|
| 显存效率 | ✅ 高（权重去重） | ❌ 低（全复制） | ✅ 高（分片） |
| 扩展性 | ✅ 高（异步、独立） | ✅ 高 | ❌ 低（需同步 all-gather） |
| 小批次性能 | ✅ 优（CaS 模式优化尾部） | ✅ 优 | ❌ 差（通信开销主导） |
| 实现复杂度 | ⚠️ 中等（双模式切换） | ✅ 简单 | ✅ 简单但不适用推理 |

> ✅ 表示优势，❌ 表示劣势，⚠️ 表示折中

---

## 2. 核心实验方法和设置

### 使用的模型
论文未使用传统意义上的“数据集”，而是针对典型的**离线推理任务**进行端到端测试，涉及以下主流大模型：

- **Qwen3-32B**
- **Llama-3.1-70B**
- **Qwen2.5-72B**

这些模型代表了当前大规模离线应用场景中的典型负载。

---

### 实验设置和评估指标

#### 硬件平台（见 Table 1）
| 节点类型 | GPU 数量 | 单卡显存 | 互连技术 |
|---------|----------|-----------|------------|
| H20     | 8× H20   | 144 GB    | NVLink 4   |
| H200    | 8× H200  | 144 GB    | NVLink 4   |
| B200    | 8× B200  | 180 GB    | NVLink 5   |

#### 并行配置
- 对比多种并行组合：
  - Pure Tensor Parallelism (TP)
  - TP + Pipeline Parallelism (PP)
  - TP + Data Parallelism (DP)
  - SiDP（默认 DP=8）
- 序列长度（S）：1K、2K、4K tokens
- 批处理策略：固定大批次 vs 自适应批次（$B \sim M - W \cdot S$）

#### 评估指标
1. **可用 KV 缓存容量**（in tokens）
2. **端到端吞吐量**（tokens/sec）
3. **每迭代解码时间**（T(B)）
4. **消融实验中的模块贡献**

---

### 基线方法对比
- **vLLM**（0.10.1）作为主基准系统
- 包括以下变体：
  - `vLLM-TP`：纯张量并行
  - `vLLM-TP+PP`：张量+流水线并行
  - `vLLM-TP+DP`：张量+数据并行（最强 baseline）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ KV Cache 容量提升
- 在相同配置下，SiDP 可将可用 KV cache 提升 **最高达 1.8×**。
- 特别是在高 DP 场景（如 TP=1, DP=8）下效果显著：
  - vLLM 无法运行 Llama-3.1-70B 和 Qwen2.5-72B（OOM）
  - SiDP 成功支持约 **1.0M KV tokens**

> 图 5 显示，在 TP=2, DP=4 下，SiDP 的 KV 容量约为 vLLM 的 **1.7×**

---

#### ✅ 端到端吞吐量提升
- 在 H20、H200、B200 上，SiDP 相比 vLLM 最高实现 **1.5× 吞吐提升**。
- 提升幅度随序列长度增加而增大：
  - S=1K：接近 baseline（已 compute-bound）
  - S=4K：优势明显（memory-bound 场景）

> 图 6–8 显示，在 S=4K 场景下，SiDP 吞吐普遍高出 DP+TP 配置 **30%~50%**

---

#### ✅ 消融实验结果

##### （1）峰值偏移（Peak Shifting）的影响（图 10）
- 在 Qwen3-32B 上，随着 DP 度增加：
  - DP=4：吞吐提升 **2.5×**
  - DP=8：吞吐提升 **3.4×**
- 证明去中心化预取调度对缓解 NVLink 热点至关重要。

##### （2）模式切换的作用（图 13）
- 仅使用 WaS 模式：相比 vLLM 提升 **7%~9%**
- 启用 WaS+CaS 动态切换：提升扩大至 **27%~32%**
- 表明 CaS 对优化“长尾”阶段具有决定性作用。

##### （3）CaS 优化逐级验证（图 14）
在 B=1 的极端尾部场景下（Llama-3.1-70B）：
| 实现方式 | 单请求耗时 | 相对加速 |
|--------|------------|----------|
| FSDP-style all-gather | 33 s | — |
| + Async P2P (CaS V1) | 25 s | 24% ↑ |
| + GEMM Fusion (V2) | 19 s | 24% ↑ |
| + Dummy Skipping (V3) | 12 s | 37% ↑ |
| **总计优化** | → | **2.8× 更快** |

---

## 4. 关键结论和发现

### 论文的主要发现
1. **权重冗余是离线推理的显存瓶颈**：在高 DP 设置下，重复的 FFN 权重成为 HBM 消耗的主要部分，严重压缩 KV cache。
2. **带宽可换显存**：现代 GPU 节点间的高带宽互连（NVLink）足以支撑运行时动态加载权重，且延迟可被计算掩盖。
3. **双模式设计必要**：
   - WaS 适合主体大批次阶段，维持 DP 独立性；
   - CaS 专治小批次尾部，通过计算聚合提升效率。
4. **实际收益显著**：在真实硬件和模型上，SiDP 实现了 **1.5× 吞吐提升** 和 **1.8× KV 容量扩展**，优于所有传统并行组合。

---

### 方法的局限性
1. **依赖高速互连**：目前仅适用于 NVLink 连接的同节点内部署，难以直接扩展至 PCIe 或以太网环境。
2. **故障域扩大**：权重分布化后，任一 GPU 故障可能导致整个 DP 组不可用，削弱了传统 DP 的容错隔离能力。
3. **主要针对 Dense 模型**：未直接支持 MoE 架构，尽管作者指出未来可结合 Expert Parallel Load Balancing (EPLB) 进行扩展。

---

### 未来工作方向
1. **异构集群适配**：研究在低带宽环境下如何调整 WaS/CaS 切换阈值和预取策略。
2. **支持 MoE 架构**：探索在 Expert-Parallel 组内应用 SiDP 进行专家权重共享。
3. **跨节点扩展**：结合参数中心化管理（如 KunServe）实现更大范围的权重资源共享。
4. **自动化模式决策**：开发更智能的 runtime 控制器，基于实时负载动态选择最优执行路径。

--- 

> **总结**：SiDP 通过将权重重构为“带宽支撑的共享服务”，巧妙地解决了 DP 中的显存浪费问题，在保持高扩展性的同时大幅提升离线 LLM 推理的吞吐能力，是一项兼具理论洞察与工程实用性的创新。

</details>

---

### 4. [LaneRoPE: Positional Encoding for Collaborative Parallel Reasoning and Generation](https://arxiv.org/abs/2605.27570)

**Authors**: Gabriele Cesa, Thomas Hehn, Aleix Torres-Camps, \`Alex Batlle Casellas, Jordi Ros-Giralt, Arash Behboodi, Tribhuvanesh Orekondy  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.27570v1  

#### Abstract
Parallel LLM test-time scaling techniques (e.g., best-of-$N$) require drawing $N>1$ sequences conditioned on the same input prompt. These methods boost accuracy while exploiting the computational efficiency of batching $N$ generations. However, each sequence in the batch is traditionally generated i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LaneRoPE: Positional Encoding for Collaborative Parallel Reasoning and Generation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统的并行推理方法（如 best-of-N、self-consistency）在生成多个响应时，各序列是**独立采样**的，彼此之间无法共享中间推理过程或观察结果。这导致：
- 无法利用问题内在的可分解结构；
- 存在冗余计算；
- 难以实现真正的协作式推理。

尽管已有工作（如 GroupThink、Hogwild!）尝试引入跨序列注意力，但它们通常依赖虚拟排序或特殊运行时机制，存在训练不兼容、负相对位置等问题。

### **提出的新方法：LaneRoPE**
本文提出 **LaneRoPE**，一种支持**协作式并行推理与生成**的位置编码方案，其核心思想是：
- 在批量生成过程中，让不同序列（称为“lane”）能够**条件性地关注彼此的中间生成内容**，从而实现细粒度协作。

#### **两大关键技术创新**：
1. **跨序列注意力掩码（Inter-sequence Attention Mask）**
   - 修改标准的因果注意力机制，允许当前序列中的 token 关注其他序列中已生成的 token。
   - 实现方式：通过扩展 KV Cache，使每个查询 token 可访问所有 lane 的历史 token。

2. **扩展的旋转位置编码（Extended RoPE）**
   - 原始 RoPE 仅建模 token 内部的相对位置。
   - LaneRoPE 引入**二维傅里叶基**，同时建模：
     - **token 维度**上的相对距离（传统 RoPE）；
     - **lane 维度**上的相对距离（即序列间偏移）。
   - 具体形式：`f(q,k)(x,i,m) = R_Ωm · R_θi (W·x + b)`，其中 `m` 是 lane index，`i` 是 token index。

### **相比现有方法的优势**
| 方法 | 是否需修改架构 | 是否支持 fine-tuning | 推理效率 | 能否建模 lane 间相对位置 |
|------|----------------|------------------------|-----------|----------------------------|
| **Best-of-N / Self-consistency** | ❌ 否 | ✅ 是 | ⭐⭐⭐⭐ | ❌ 否 |
| **GroupThink** | ✅ 是（虚拟拼接） | ❌ 否（out-of-distribution 负索引） | ⭐⭐⭐ | ⚠️ 近似 |
| **Hogwild!** | ✅ 是（动态重排序） | ❌ 否 | ⭐⭐ | ❌ 否 |
| **Bridge** | ✅ 是（额外 attention 层） | ✅ 是 | ⭐⭐ | ❌ 否 |
| **LaneRoPE (Ours)** | ✅ 极小改动（drop-in 替换 RoPE） | ✅ 是（支持 SFT/KTO） | ⭐⭐⭐⭐ | ✅ 是 |

- **轻量高效**：无需新增网络层，仅替换 RoPE 模块，推理开销极低（~6%）。
- **灵活通用**：可表达多种策略（如独立采样、GroupThink），并通过训练超越初始设定。
- **支持微调**：首次为协作推理提供可微调框架，且参数增量 <0.5%。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MATH500**：500道高难度数学题，用于验证基础推理能力。
- **AMC 23**：美国数学竞赛题（AMC 10/12 2023）。
- **AIME 24 & AIME 25**：美国邀请数学考试题（更具挑战性）。
- 所有问题均要求模型输出最终答案于 `\boxed{}` 中以便自动评分。

### **实验设置**
- **模型**：
  - 主要使用 **DeepSeek-R1-Distill-Qwen-1.5B** 和 **7B** 模型。
  - 对比模型包括原生模型、Hogwild!、Bridge 等。
- **并行数量 N**：测试 N=1, 2, 4 条 lanes。
- **生成参数**：
  - 温度 `temp=0.6`，top-p=0.95；
  - 最大生成长度 4096 tokens。
- **评估指标**：
  - **maj@4**：从 M=16（MATH500 为12）个样本中取多数投票正确率，确保公平比较不同 N 下的总预算（B=4）。
  - **Pass@1 (Accuracy)**：单次生成准确率（用于消融分析）。

### **基线方法对比**
- **Baseline**：原始模型 + 独立采样（best-of-N）。
- **Hogwild!**：官方实现，支持并发注意力。
- **Bridge**：作者自实现，采用 axial attention 结构。
- **GroupThink**：作为 LaneRoPE 的一个特例进行初始化。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1, maj@4 平均得分）**

| Model | N | MATH500 | AIME24 | AIME25 | AMC23 | **Avg** |
|-------|----|---------|--------|--------|-------|--------|
| DS-Qwen-7B (baseline) | 1 | 86.3 | 25.7 | 26.1 | 70.2 | 52.1 |
| +Hogwild! | 2 | 70.4 | 20.8 | 21.1 | 58.0 | 42.8 |
| +Bridge [7] | 2 | 88.8 | 40.5 | 29.2 | 81.4 | **60.0** |
| **+LaneRoPE (NTK*) [KTO]** | **4** | **90.7** | **46.3** | **33.6** | **85.9** | **64.1** ✅ |

> ✅ **LaneRoPE 在 7B 模型上达到最高平均分 64.1（+12 pts vs baseline）**

### **与基线方法的对比结果**
- **优于 Bridge**：在相同 KTO 设置下，LaneRoPE 表现更优，说明其信息共享机制更有效。
- **显著优于 Hogwild!**：Hogwild! 导致性能下降，可能因模型未适配其提示结构或产生过长输出。
- **优于独立采样 + 多数投票**：见 Figure 4，在相同并行预算下，LaneRoPE 的协作样本比独立样本带来更高收益。

### **消融实验结果（Table 4）**
| 变体 | AIME24 | AIME25 | AMC23 | Avg |
|------|-------|--------|-------|-----|
| LaneRoPE (GT) | 46.2 | 33.8 | 82.5 | 54.2 |
| LaneRoPE (NTK) | 44.5 | 33.6 | 83.1 | 53.8 |
| **LaneRoPE (NTK*) + KTO** | **46.5** | **33.6** | **85.9** | **55.3** ✅ |

- **NTK-aware 初始化**缓解了 GroupThink 的负索引问题，提升稳定性。
- **可学习频率 Ω*** 进一步释放模型潜力。
- **KTO 训练优于 SFT**：得益于更大规模的数据和更强的学习信号。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **协作推理可行且有效**：通过 LaneRoPE，多个序列可在生成过程中相互借鉴、避免重复劳动，显著提升复杂任务（尤其是数学推理）的表现。
2. ✅ **位置编码设计至关重要**：传统的 RoPE 无法区分 lane 身份；LaneRoPE 通过二维相对位置建模，实现了对跨 lane 注意力的有效控制。
3. ✅ **轻量化也能高性能**：LaneRoPE 几乎无推理开销（Table 2 显示仅 ~6% 延迟增加），远低于 Bridge（+25%）。
4. ✅ **更大的模型更受益**：7B 模型在协作中表现明显优于 1.5B，表明大模型具备更强的协同推理潜力。

### **方法的局限性**
- 当前实验限制在 **N ≤ 4 lanes**，更多 lanes 下的扩展性和内存占用尚未充分验证。
- 缺乏专门的 **多 lane 输出融合机制**（如加权聚合、动态选择），目前仍依赖多数投票。
- 初始化对性能影响较大，未经训练的版本可能出现退化（如 GroupThink 初始版本生成损坏文本）。

### **未来工作方向**
- 设计端到端的 **multi-lane fusion head**，替代简单的 majority voting。
- 探索基于强化学习（如 RLVF）的训练方式，进一步激发协作行为。
- 将 LaneRoPE 应用于非数学类任务（如代码生成、规划、辩论等），探索其通用性。
- 支持动态 lane 数量或 adaptive branching + LaneRoPE 的混合范式。

---

> 📌 **总结一句话**：  
> **LaneRoPE 通过扩展 RoPE 实现了轻量、高效、可微调的协作式并行推理，在数学任务上显著超越独立采样和其他协作方法，为 LLM test-time scaling 提供了一条新路径。**

</details>

---

### 5. [ZipRL: Adaptive Multi-Turn Context Compression with Hindsight Response Replay](https://arxiv.org/abs/2605.28069)

**Authors**: Zhexin Hu, Li Wang, Xiaohan Wang, Jiajun Chai, Xiaojun Guo, Wei Lin, Guojun Yin  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.28069v1  

#### Abstract
Adaptive context compression is vital for scaling Large Language Models (LLMs) to complex, multi-turn agent tasks. However, rule-based compression methods may discard task-critical nuances, while Reinforcement Learning (RL) approaches usually struggle to balance information retention and token effic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ZipRL: Adaptive Multi-Turn Context Compression with Hindsight Response Replay

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在基于 **Large Language Models (LLMs)** 的多轮代理任务中，上下文窗口有限，而长期交互会迅速耗尽上下文容量。现有方法面临两大挑战：
- **Uniform Compression 问题**：大多数压缩方法对所有检索到的内容采用统一的处理粒度，忽略了不同文档与查询的相关性差异，导致关键信息丢失或噪声保留。
- **Sparse Rewards 问题**：强化学习（RL）优化过程中，奖励信号稀疏（通常只在最终任务成功时给出），难以有效归因于中间步骤（如某一轮的压缩行为），阻碍策略学习。

### 提出了什么新方法或新思路
本文提出 **ZipRL**，一种专为 **Reinforcement Learning from Verifiable Rewards (RLVR)** 设计的自适应多轮上下文压缩框架，其核心创新包括：

#### （1）Multi-Granularity Compression Mechanism（多粒度压缩机制）
- 引入 **5级压缩粒度**（从 `Ultra-coarse` 到 `Ultra-fine`），允许模型根据文档与查询的**相关性强度**动态选择压缩级别。
- 高相关性文档分配更宽松的压缩范围以保留更多细节；低相关性文档则被高度压缩以过滤噪声。
- 通过 **in-context prompts** 引导模型进行相关性判断和压缩决策，实现主动感知与非均匀信息保留。

#### （2）Hindsight Response Replay (HRR)（后见之明响应回放）
- 受 **Hindsight Experience Replay (HER)** 启发，提出 HRR 技术，用于在 RL 训练中**稠密化训练信号**。
- 不依赖昂贵的外部 **Process Reward Models (PRMs)**，而是利用一个**启发式压缩质量评分函数 $Q_{com}$** 来衡量每一轮压缩的质量。
- 在 **Group Relative Policy Optimization (GRPO)** 中，通过 **advantage reshaping** 将轨迹级的稀疏奖励重新分配到各轮次：若某轮压缩质量高于轨迹平均，则提升其优势值（advantage），反之则降低。
- 实现了“即使最终失败，高质量的中间压缩也应被鼓励”的思想。

### 相比现有方法的优势
| 维度 | ZipRL | 现有方法（如 ASearcher, AgentFold） |
|------|-------|-----------------------------|
| **压缩粒度** | 自适应多粒度，按需保留 | 固定或单一粒度 |
| **奖励信号** | 稠密化（HRR），支持细粒度信用分配 | 稀疏，仅最终奖励 |
| **训练效率** | 更快收敛，更低方差 | 易受稀疏奖励影响 |
| **长程鲁棒性** | 在 256 轮极端测试下仍持续提升 | 在短轮次后即饱和或下降 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验涵盖 **5个基准任务**，分为两类：
- **Multi-hop QA**（多跳问答）：
  - **MusiQue**, **SQuAD**, **Frames**, **Bamboogle**
- **Web Browsing**（网页浏览）：
  - **BrowseComp (BC-Plus)**：动态网页检索与推理任务

### 实验设置和评估指标
- **基础模型**：Qwen2.5 和 Qwen3 系列（3B, 4B, 7B, 8B, 14B, 32B）
- **训练设置**：
  - 最多 20 轮交互
  - Batch size: 64
  - Learning rate: 1e-6
  - 使用 **Cold Start SFT** 阶段初始化策略（基于 GPT-4o 生成的 1,155 条高质量轨迹）
- **评估指标**：
  - **EM (Exact Match)**：答案完全匹配
  - **F1 Score**：基于词重叠的 F1
  - **Token Efficiency**：压缩后的 token 数量
  - **Finish Rate**：成功终止的比例

### 基线方法对比
| 类别 | 基线方法 |
|------|--------|
| **ReAct-based** | Qwen3-235B-ReAct, GPT-4o-ReAct, Gemini-3-Pro-ReAct |
| **Summary-only** | Qwen3-235B-Summary, GPT-4o-Summary |
| **专用搜索代理** | ASearcher, AgentFold, NestBrowse, WebSailor, Search-R1 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **ZipRL-8B** 在 5 个任务上的平均 EM 达到 **30.3%**，F1 为 **41.2%**，**超越参数量大 29 倍的 Qwen3-235B-ReAct**，并媲美 Gemini-3-Pro。
- 在 **Qwen3-4B** 和 **Qwen3-8B** 模型上，相比最强同规模基线（ASearcher, AgentFold），ZipRL 分别提升了 **27.9%** 和 **34.7%** 的平均 EM。

### 与基线方法的对比结果
| 模型 | 平均 EM | 相比最强基线提升 |
|------|--------|----------------|
| ZipRL-4B | 28.0% | +6.1% (vs AgentFold-4B) |
| ZipRL-8B | 30.3% | +7.8% (vs AgentFold-8B) |
| ZipRL-14B | 31.1% | +3.3% (vs AgentFold-14B) |
| ZipRL-32B | 35.2% | +4.1% (vs ZipRL-14B) |

> ✅ **ZipRL 在所有参数规模上均领先**，且性能随模型增大持续提升。

### 消融实验结果（Ablation Studies）
在 Qwen3-8B 上进行消融实验，验证各组件贡献：

| 模型变体 | 平均 EM | 下降幅度 |
|----------|--------|---------|
| **ZipRL (完整)** | 30.3% | — |
| w/o RL（无强化学习） | 26.4% | ↓3.9% |
| w/o Level 2&4（减少压缩层级） | 28.9% | ↓1.4% |
| w/o Qinfo（移除信息保留评分） | 26.9% | ↓3.4% |
| w/o Qsem（移除语义完整性评分） | 27.7% | ↓2.6% |

> 🔍 结果表明：
> - **RL 阶段至关重要**，冷启动 SFT 无法达到高性能。
> - **多粒度设计**（5级）优于简化版本。
> - **Qinfo 和 Qsem** 是压缩质量评估的核心维度。

---

## 4. 关键结论和发现

### 主要发现
1. **多粒度压缩优于均匀压缩**  
   理论证明（Theorem 4.1）和实验均表明，在相同资源预算下，**将更多压缩资源分配给高相关性文档**能显著提升下游任务效用。

2. **HRR 有效缓解稀疏奖励问题**  
   通过将压缩质量作为“后见目标”，HRR 成功实现了**无需外部 PRM 的奖励稠密化**，加速了 RL 收敛并提高了稳定性。

3. **卓越的长程可扩展性**  
   在 **256 轮极端外推压力测试**中，ZipRL-8B 性能持续上升，而 Qwen3-235B-ReAct 在 16 轮后即饱和，表明 ZipRL 学会了**相关性感知的信息保留**，而非过拟合短期模式。

4. **高 Token 效率**  
   ZipRL 在保持高性能的同时，显著减少了 token 使用量，优于 ReAct 和 AgentFold。

### 方法的局限性
1. **语言依赖性**：$Q_{info}$ 依赖英文停用词表，在多语言或专业领域（如法律、代码）可能退化。
2. **可信度未建模**：$Q_{com}$ 忽略文档可信度，在对抗性检索下性能急剧下降（EM 下降 85–99%）。
3. **泛化能力受限**：Cold Start 依赖单一 QA 数据集，可能限制在结构化或代码密集任务中的适应性。

### 未来工作方向
- 扩展 $Q_{com}$ 以包含 **文档可信度估计**。
- 探索 **跨语言与跨领域适配** 的压缩策略。
- 将 ZipRL 框架应用于 **代码生成、数学推理** 等复杂结构化任务。
- 进一步研究 **压缩与检索的联合优化**，避免冗余检索。

---

> **代码开源地址**：https://github.com/huzhexin/ZipRL

</details>

---

### 6. [CIVIC: End-to-End Sequence Compactness for Efficient Vision-Language Models](https://arxiv.org/abs/2605.28115)

**Authors**: Fengze Yang, Bo Yu, Xuewen Luo, Cathy Liu, Chenxi Liu  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.28115v1  

#### Abstract
Vision-Language Models (VLMs) face severe memory and latency bottlenecks due to high-resolution visual tokens. While current token reduction methods theoretically save FLOPs, post-hoc pruning introduces structural overhead, failing to yield proportional wall-clock acceleration. However, enforcing a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CIVIC: End-to-End Sequence Compactness for Efficient Vision-Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Vision-Language Models (VLMs)** 在处理高分辨率图像或长视频时会产生大量视觉 token，导致 **FLOPs** 和 **KV-cache 内存占用**急剧上升。尽管已有许多 token pruning 或 merging 方法在理论上减少了计算量，但由于以下原因，**理论上的压缩并未转化为实际的推理加速**：
- 后处理（post-hoc）压缩引入了额外的运行时开销（如 token scoring、gather/scatter、merge/unmerge）；
- 压缩后的 token 通常只是中间状态，下游仍需恢复为密集表示以兼容原始模型结构；
- 非连续内存访问破坏了 GPU 的硬件效率。

这种“**压缩-现实差距**”（compression-realization gap）严重限制了高效 VLM 的实际部署。

### 提出了什么新方法或新思路
本文提出 **CIVIC**（Compact Inference for Vision-Language Integrated Compression），一种**端到端路径一致的紧凑视觉推理框架**，其核心思想是：
> 将紧凑表示作为从视觉输入到语言模型生成全过程中的**主推理路径**，而非临时中间状态。

具体技术组件包括：
- **Anchor-based Aggregation**：通过可学习的 anchor 对 dense patch embeddings 进行聚合，生成连续紧凑 token，保持空间映射关系；
- **Adaptive Spatial Retention Floor**：动态保留一定比例的空间细节，防止过度压缩导致细粒度定位能力下降；
- **KV-compressed Attention**：在视觉编码器中压缩 Key/Value 到固定数量的 anchors，降低注意力计算复杂度；
- **Text-aligned KL Distillation**：仅在文本位置上进行知识蒸馏，使紧凑视觉嵌入能无缝替换原始模型中的视觉占位符，避免结构不匹配。

### 相比现有方法的优势
| 维度 | 传统方法（如 DyMU, DynamicViT） | CIVIC |
|------|-------------------------------|-------|
| 推理路径 | 后处理压缩 → 中间状态 → 下游恢复为 dense | 原生紧凑路径贯穿全程 |
| 内存访问 | 非连续（scatter/gather） | 连续紧凑序列 |
| 开销来源 | 显著的 runtime routing 和 restoration overhead | 几乎无额外开销（仅 0.49ms） |
| 实际加速 | 不明显甚至变慢 | 显著降低延迟与 KV-cache 占用 |
| 准确性 | 在精细任务（如定位、数学推理）上易退化 | 保持甚至提升多模态准确性 |

---

## 2. 核心实验方法和设置

### 使用的数据集
论文在五个代表性 benchmark 上评估多模态能力：
- **MMMU**：多学科多模态理解（复杂推理）
- **MathVision**：数学图文联合推理
- **ODinW-13**：开放域目标检测与定位（细粒度感知）
- **RealWorldQA (RWQA)**：真实世界常识问答
- **VideoMME (short split)**：短视频上下文跟踪与理解

### 实验设置和评估指标

#### 模型架构
- 基础模型：**Qwen3-VL-2B-Instruct**
- 硬件平台：单张 **NVIDIA RTX 4090 GPU**
- 所有生成过程使用确定性参数（`T=0`, `top-k=1`, `top-p=1`），确保时间测量一致性。

#### 评估维度
- **结构性指标**：
  - Keep Ratio（保留率）
  - FLOPs
  - Prefill Length（预填充长度）
  - Peak KV Cache 大小
- **物理性指标**：
  - End-to-end Latency（总延迟）
  - Vision Encoding Time
  - Prefill / Decode 时间分解
  - Throughput
  - Compression Overhead

### 基线方法对比
与五种主流 token reduction 方法比较：
- **DyMU**：动态合并 + 虚拟解合并
- **DiffRate**：可微分压缩率控制
- **DynamicViT (hard/soft)**：基于评分的硬裁剪 vs Gumbel-Softmax 软选择
- **VisionTrim**：统一视觉 token 压缩
- **ZOO-Prune**：零阶梯度估计的 token 剪枝

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 指标 | Baseline | CIVIC (Ours) | 提升幅度 |
|------|----------|-------------|---------|
| **Total Latency** | 3543.0 ms | **2514.9 ms** | ↓ **29.0%** |
| **Vision Encoder Time** | 47.82 ms | **32.53 ms** | ↓ 32.0% |
| **Projection Time** | 0.39 ms | **0.16 ms** | ↓ 59.0% |
| **Prefill Time** | 35.20 ms | **18.94 ms** | ↓ 46.2% |
| **Decode Time** | 3128.4 ms | **2229.4 ms** | ↓ 28.7% |
| **LLM Total (Prefill + Decode)** | 3274.3 ms | **2248.3 ms** | ↓ 31.3% |
| **Compression Overhead** | 0.00 ms | **0.49 ms** | 可忽略 |
| **Prefill Tokens** | 1122.2 tokens | **407.9 tokens** | ↓ 63.6% |
| **KV Cache Memory** | 122.7 MB | **44.61 MB** | ↓ **63.6%** |

> ✅ CIVIC 是唯一一个在所有阶段均显著降低延迟且几乎无额外开销的方法。

### 与基线方法的对比结果
- **延迟方面**：CIVIC 总延迟最低（2514.9ms），而其他方法多数比 baseline 更慢（如 DynamicViT soft 达 4132.6ms）。
- **KV-cache 利用率**：CIVIC 将 KV-cache 降至 baseline 的约 **1/3**（见 Figure 4），远优于 VisionTrim/ZOO-Prune（仅降至 ~92MB）。
- **准确率保持**：在 Figure 3 中显示，CIVIC 在所有 benchmark 上性能接近或略优于 baseline，尤其在 ODinW 定位任务上有意外增益；而其他方法在 MathVision 和 ODinW 上出现明显性能悬崖。

### 消融实验结果（Ablation Study）

#### （1）Compact Token Budget（C）
| C | Total Latency | Decode Time |
|----|----------------|--------------|
| 64 | 2823.7 ms | 2514.4 ms |
| 256 | **2440.8 ms** | **2151.3 ms** |
| 512 | 2327.6 ms | 2038.0 ms（但 vision encoder 时间上升） |

➡️ 结论：存在帕累托最优点（C=256 左右），过低压缩会导致 decode 因信息不足而变慢。

#### （2）Minimum Keep Ratio（Min）
| Min | Total Latency | Performance Stability |
|-----|----------------|------------------------|
| 0.0 | 2591.8 ms | ⚠️ 细节丢失风险高 |
| 0.2 / 0.5 | ~2630 ms | ✅ 更稳定，保护局部特征 |

➡️ 结论：设置自适应保留底限对维持 fine-grained localization 至关重要。

#### （3）KV Compression Anchors 数量
| Anchors | Total Latency | Memory Efficiency |
|--------|----------------|--------------------|
| 128 | 2543.4 ms | 最佳平衡点 |
| 512 | 2660.3 ms | 开销增加 |

➡️ 结论：KV-anchor 设计有效，但不宜过多。

#### （4）其他关键验证
- 若在 LLM 前人工恢复 dense placeholder，则所有加速消失 → 证明**端到端一致性至关重要**。
- 使用 text-aligned KL distillation 显著优于标准 distillation → 保证语义对齐。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **理论 FLOP 减少 ≠ 实际加速**：后处理压缩因引入非连续操作和 dense restoration，无法实现真正的 wall-clock 加速。
2. **端到端序列紧凑性才是关键**：只有将 compact 表示作为原生推理路径，才能最大化硬件效率。
3. **KV-cache 是瓶颈所在**：减少进入 LLM 的视觉 token 数量，可级联降低 prefill 和 decode 阶段的 cross-attention 成本。
4. **CIVIC 实现了“双赢”**：在大幅降低延迟（↓29%）和 KV-cache（↓63.6%）的同时，**未牺牲任何多模态准确性**，甚至在某些任务上略有提升。

### 方法的局限性
- 当前采用**静态 token 预算**（fixed compact length），缺乏实例自适应能力；
- 实验局限于 **single-image 输入** 和 **Qwen3-VL-2B** 架构；
- 尚未扩展至 multi-image 或 long-form video 场景。

### 未来工作方向
- 探索 **dynamic, instance-adaptive token budgets**；
- 验证 CIVIC 在更大规模模型（如 7B+）上的可扩展性；
- 支持 **multi-image fusion** 与 **long video streaming inference**；
- 结合 on-device safety mechanisms，应对边缘部署的伦理风险。

--- 

> 📌 **一句话总结**：  
> CIVIC 通过构建一条贯穿 VLM 全流程的**原生紧凑推理路径**，成功将理论压缩转化为真实世界的推理加速，在显著降低延迟与内存占用的同时，完整保留了多模态理解能力，为高效 VLM 部署提供了新范式。

</details>

---

### 7. [DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting, Precise Verification, and Fully Parallel Execution](https://arxiv.org/abs/2605.28678)

**Authors**: Yunhai Hu, Zining Liu, Xiangyang Yin, Tianhua Xia, Bo Bao, Eric Sather, Vithursan Thangarasa, Sai Qian Zhang  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.28678v1  

#### Abstract
Speculative reasoning has recently been proposed as a means to accelerate reasoning-intensive generation in large multimodal models, but its effectiveness is often constrained by misalignment between speculative drafts and target-verified reasoning. In this work, we introduce DREAM-R, a framework th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting, Precise Verification, and Fully Parallel Execution》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前在 **Large Reasoning Models (LRMs)** 和 **Multimodal Large Reasoning Models (MLRMs)** 中，推理过程依赖于长链的 **Chain-of-Thought (CoT)** 推理，导致生成延迟高、计算成本大。虽然已有 **speculative decoding** 和 **speculative reasoning** 技术用于加速推理，但在多模态场景下（如视觉-语言任务）效果不佳，主要原因包括：
- 草稿模型（draft model）与目标模型（target model）之间的 **alignment misalignment**（对齐偏差）
- 多模态推理中存在 **perceptual error**（感知错误），例如视觉线索识别错误
- 验证机制不稳定，易受噪声影响，导致错误传播

### 提出的新方法与创新点
论文提出 **DREAM-R**，一个面向多模态推理的高效 speculative reasoning 框架，其核心创新包括以下三点：

#### ✅ 1. **Speculative Alignment Policy Optimization (SAPO)**
- 一种基于 **Reinforcement Learning (RL)** 的训练目标，用于优化草稿模型
- 引入复合奖励函数，包含：
  - `R_outcome`：最终答案正确性
  - `R_draft`：草稿步骤被接受的比例（对齐度）
  - `R_length`：惩罚过长推理路径
- 通过 **Group Relative Policy Optimization (GRPO)** 更新策略，提升草稿模型生成“既忠实又简洁”的推理步骤的能力

#### ✅ 2. **Contrastive Probability Normalization (CPN)**
- 改进传统基于标量评分（scalar score）的验证方式
- 使用二分类判断（“positive” / “negative”）并计算概率比值 $ p = \frac{s_+}{s_+ + s_-} $
- 只有当 $ p > \alpha $（默认 0.7）时才接受该推理步
- 优势：更稳定、可解释性强、避免离散分数带来的阈值敏感问题

#### ✅ 3. **Fully Parallel Speculative Reasoning (FPSR)**
- 实现 **drafting、target reasoning、verification** 三阶段完全并行化
- 支持早期终止（early stopping）和干净回滚（clean fallback）
- 显著降低端到端延迟，最大化 GPU 利用率

### 相比现有方法的优势
| 方面 | DREAM-R | 现有方法（如 SpecReason、Lookahead Reasoning） |
|------|---------|---------------------------------------------|
| 对齐能力 | ✅ 通过 RL 显式优化 alignment | ❌ 依赖预训练或监督微调，alignment 不足 |
| 验证稳定性 | ✅ 基于概率归一化的 CPN 更鲁棒 | ❌ 使用离散评分，易受 prompt 设计影响 |
| 并行效率 | ✅ 三阶段全并行，支持流水线 | ❌ 多为串行或部分重叠，利用率低 |
| 多模态适配性 | ✅ 显式建模视觉-语言联合推理 | ❌ 多针对纯文本设计，忽视视觉 grounding |

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个主流的多模态推理基准上进行评估：
- **MathVerse**：视觉数学题理解与求解
- **MMBench**：综合多模态理解能力评测
- **RealWorldQA**：真实世界图像问答
- **MMMU**：大规模跨学科多模态理解与推理

所有实验均使用完整测试集。

### 实验设置
| 组件 | 配置 |
|------|------|
| **Target Models** | Qwen3-VL-32B, Qwen3-VL-235B-A22B |
| **Draft Models** | Qwen3-VL-2B (Q2B), Qwen3-VL-4B (Q4B), R-4B, MiMo-VL-7B-RL (M7B-RL) |
| **硬件平台** | NVIDIA L40S GPUs（target: 4卡 INT4量化；draft: 2卡） |
| **Lookahead Window** | 固定为 4 步 |
| **Acceptance Threshold α** | 默认设为 0.7 |
| **Training Platform** | 8×NVIDIA H200 GPUs, BF16 精度 |
| **Training Time** | 单个 draft model 约 74 小时 |

### 评估指标
- **Accuracy (Acc.)**：任务最终答案准确率
- **Acceptance Rate (Acpt.)**：草稿推理步被接受的比例
- **Speedup**：相对于标准自回归解码的速度提升倍数（以 wall-clock time 计算）

### 基线方法对比
- **Standard SD**：标准 speculative decoding
- **SpecReason (Pan et al., 2025)**：基于打分制的 speculative reasoning
- **LR (Lookahead Reasoning, Fu et al., 2025)**：异步生成与验证
- **DREAM-R-NS**：DREAM-R 的消融版本（不含 SAPO，仅含 CPN + FPSR）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）
| Target | Draft | Method | MathVerse Acc. | Speedup | MMBench Acc. | Speedup |
|--------|-------|--------|----------------|---------|--------------|---------|
| Q32B | Q2B | DREAM-R | **75.98%** | **2.38×** | **82.65%** | **1.86×** |
| Q32B | Q4B | DREAM-R | **81.60%** | **2.48×** | **89.44%** | **2.48×** |
| Q235B | Q2B | DREAM-R | **80.00%** | **2.28×** | **88.45%** | **2.31×** |
| Q235B | Q4B | DREAM-R | **84.00%** | **1.79×** | **82.20%** | **1.73×** |

> ⚡ 最高达到 **2.48× 加速**，同时保持接近目标模型的准确率（如 Q32B 原始为 76.00%，DREAM-R 达到 75.98%）

### 与基线方法对比
| 方法 | 准确率保留 | 加速比 | 接受率 |
|------|-----------|--------|--------|
| **SpecReason** | 显著下降（如从 76% → 44.57% on MathVerse） | ~1.2–1.4× | 极低（<25%） |
| **LR (Lookahead)** | 中等下降 | ~1.5–1.9× | 中等（~35–70%） |
| **DREAM-R-NS** | 接近目标模型 | ~1.8–2.4× | 显著高于基线 |
| **DREAM-R (完整版)** | **几乎无损准确率** | **最高达 2.48×** | **最高达 71.45%** |

> 🔍 特别是在弱草稿模型（如 Q2B）情况下，DREAM-R 仍能实现高接受率和高速度提升，而其他方法性能崩溃。

### 消融实验结果

#### ✅ Ablation on SAPO Reward Design（图5）
- 权重设置 $ w_1:w_2:w_3 = 1:1:1 $（outcome:draft:length）表现最佳
- 过度强调 `R_outcome` 导致输出变长、速度下降
- 过度强调 `R_length` 导致接受率骤降（尤其在 MMBench 上降至 38.78%）
- 结论：**平衡的奖励设计至关重要**

#### ✅ Ablation on FPSR（图6）
- 在 Q32B + Q2B 设置下：
  - SpecReason: 1.18×
  - LR: 1.50×
  - **FPSR**: **1.86×**
- 表明 **完全并行化显著优于部分并行或串行验证**

#### ✅ Ablation on CPN Threshold α（图7）
- $ \alpha = 0.5 $：接受率高但准确率下降
- $ \alpha = 0.9 $：保守验证，准确率略升但速度大幅下降
- $ \alpha = 0.7 $：**最佳权衡点**，兼顾效率与准确性

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **多模态 speculative reasoning 必须考虑 alignment 和 grounding**
   - 纯语言草稿模型无法有效处理视觉线索，导致幻觉和错误传播
   - SAPO 通过 RL 显式优化 alignment，显著提升草稿质量

2. ✅ **CPN 提供更可靠、可解释的验证机制**
   - 相比标量打分，基于对比概率的方法更稳定，减少误判

3. ✅ **FPSR 实现真正的硬件级并行优化**
   - 三阶段并发执行，极大提升 GPU 利用率，是实现 2.48× 加速的关键

4. ✅ **DREAM-R 在强/弱草稿模型下均表现稳健**
   - 即使使用明显弱于目标的草稿模型（如 Q2B vs Q235B），也能实现高速度提升且不牺牲准确率

### 方法的局限性
- **训练成本较高**：SAPO 需要在 H200 GPU 上训练约 74 小时
- **依赖高质量 step-level annotations**：训练数据需具备细粒度推理标注（如 Geo3K、ScienceQA）
- **对极端能力差距模型组合可能仍受限**：若草稿模型完全无法理解任务，SAPO 效果有限

### 未来工作方向
- 探索 **zero-shot 或 few-shot alignment**，减少对标注数据的依赖
- 扩展至 **video-language reasoning** 等更复杂多模态任务
- 研究 **dynamic threshold adjustment** 机制，根据输入难度自适应调整 $ \alpha $
- 开发 **lightweight verifier modules** 替代完整 target model 进行 CPN 判断

---

> 📌 **一句话总结**：  
> DREAM-R 是首个专为 **multimodal speculative reasoning** 设计的高效框架，通过 **RL-driven alignment (SAPO)**、**probabilistic verification (CPN)** 和 **fully parallel execution (FPSR)**，实现了高达 **2.48× 的加速**，同时 **几乎无损目标模型准确率**，为大规模多模态推理系统的部署提供了实用解决方案。

🔗 **代码开源地址**：[https://github.com/HuYunhai-Alex/DREAM-R](https://github.com/HuYunhai-Alex/DREAM-R)

</details>

---

### 8. [AdaDPO: Self-Adaptive Direct Preference Optimization with Balanced Gradient Updates](https://arxiv.org/abs/2605.28440)

**Authors**: Shaolong Chen, Madalina Ciobanu, Qingqing Mao, Ritankar Das  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.28440v1  

#### Abstract
DPO has become a widely adopted alternative to RLHF for aligning LLMs with human preferences, eliminating the need for a separate reward model or RL loop. Recent theoretical analysis uncovers an asymmetric gradient behavior in DPO: the loss suppresses dispreferred responses substantially faster than...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AdaDPO: Self-Adaptive Direct Preference Optimization with Balanced Gradient Updates

---

## 1. 主要贡献和创新点

### ✅ 解决的问题
- **DPO中的梯度不对称问题**：尽管DPO在语言模型对齐中表现优异，但近期理论分析（Feng et al., 2024）指出其损失函数存在结构性缺陷——对“非偏好响应”（dispreferred responses）的梯度远大于对“偏好响应”（preferred responses）的梯度。
- 这导致模型更倾向于**抑制坏答案**，而非积极**生成好答案**，尤其当偏好与非偏好响应语义相近时，该问题更为严重。
- 此外，这种不对称还可能导致**长度偏差**（length bias），即模型通过生成更长文本获得更高的隐式奖励。

### 🆕 提出的新方法：AdaDPO
- **自适应系数机制**：提出 **AdaDPO**（Self-Adaptive Direct Preference Optimization），引入**每偏好对（per-preference-pair）的自适应系数** $\beta_w$ 和 $\beta_l$，取代原始DPO中固定的 $\beta$。
- **基于stop-gradient的构造**：利用策略模型自身的生成概率（或结合参考模型），通过 `stop-gradient` 操作计算动态权重：
  $$
  \beta_w = \beta \cdot \text{sg}\left(\frac{P_\theta(y_w|x)}{P_\theta(y_l|x)}\right)
  $$
  其中 `sg` 表示梯度停止（如PyTorch中的 `.detach()`），确保系数不参与反向传播。
- **目标**：强制使偏好与非偏好响应的概率梯度幅度相等，从而实现**平衡更新**。

### 🔍 相比现有方法的优势
| 特性 | 说明 |
|------|------|
| **无需额外模块** | 不需要辅助网络、重采样或额外前向传播（unlike Balanced-DPO, SGDPO）。 |
| **保留原超参数结构** | 仍使用标准 $\beta$，仅增加一个稳定的裁剪常数 $C$（默认为2），便于集成。 |
| **通用性强** | 同样的自适应原则可作为“即插即用”模块应用于 SimPO、IPO、ORPO、CPO、R-DPO 等多种 pairwise contrastive loss。 |
| **代码改动极小** | 仅需在标准DPO损失基础上添加约4行代码即可实现。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练数据**：UltraFeedback 数据集（61,837条样本，59,876训练 / 1,961验证）
- **基础模型**：Llama-3-8B-Instruct（SFT后版本）
- **未使用新的人类标注数据**，完全依赖开源偏好数据。

### ⚙️ 实验设置
- **训练配置**（复现 SimPO 设置）：
  - Batch size: 128
  - Max sequence length: 2048
  - 学习率调度：cosine schedule，10% warmup
  - 训练轮数：1 epoch
- **超参数搜索网格**：
  - 学习率 lr ∈ {3, 5, 6, 10} × 10⁻⁷
  - β ∈ {0.005, 0.01, 0.05, 0.1}
  - 共 16 种组合，每种运行一次，总计32次训练（DPO vs AdaDPO）

### 🎯 评估指标与基准
使用 **GPT-4-Turbo 作为自动裁判**（automatic judge）进行盲测比较：
| 基准 | 指标 | 描述 |
|------|------|------|
| **AlpacaEval 2** | LC (%)（Length-Controlled Win Rate）<br>WR (%)（Raw Win Rate）<br>Avg Length | 主要评估指标；LC 控制长度偏差，更能反映真实质量 |
| **Arena-Hard v0.1** | WR (%) ± CI | 高难度技术问题（500 query），报告95%置信区间 |
| **MT-Bench** | Score (1–10) | 多轮对话能力评估（80 question，8 categories） |

### 🔁 基线方法对比
- **SFT**（监督微调）
- **DPO**（标准Direct Preference Optimization）
- **AdaDPO**（本文方法）
- 所有方法在同一设置下公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Method   | AlpacaEval 2 LC (%) | WR (%) | Avg Length | Arena-Hard WR (%) | MT-Bench Score |
|----------|---------------------|--------|------------|-------------------|----------------|
| SFT      | 28.5                | 28.8   | 1976       | 26.3              | 7.58           |
| DPO      | 46.4                | 44.5   | 1933       | 42.9              | 8.01           |
| **AdaDPO** | **48.3** ✅         | **46.1** ✅ | **1908** ✅ | 41.5              | **8.03** ✅     |

> ✅ AdaDPO 在 AlpacaEval 2 上取得 **全局最优 LC 和 WR**，且平均输出更短。

---

### 🔍 超参数全网搜索结果（16种组合汇总）

| 指标 | AdaDPO 胜出比例 | 结论 |
|------|------------------|------|
| **LC > DPO** | **13/16 (81%)** | 绝大多数配置下长度控制胜率更高 |
| **WR > DPO** | 9/16 (56%) | 多数情况下原始胜率也更高 |
| **Δ(LC−WR) 更大** | **14/16 (88%)** | 显著减小长度偏差，表明有效缓解了 length exploitation |

> 💡 **LC−WR 差距扩大是减轻长度偏见的关键证据**：差值越大，说明模型不再靠“说得更多”获胜。

---

### 🔬 消融实验结果

#### （1）**Clipping Ceiling $C$ 的敏感性测试**（Appendix A.2）
- 测试 $C \in \{1.5, 2.0, 2.5, ..., 10\}$ 对性能影响
- **最佳值 $C=2.0$**（LC: 44.2%, WR: 42.7%）
- 性能在 $[1.5, 2.5]$ 内稳定，仅当 $C ≥ 5$ 时显著下降
- ➤ 表明 $C$ 是鲁棒的架构选择，**非敏感超参**

#### （2）**是否归一化 per-token？**
- 提出 **Stable AdaDPO**：采用 **token-level 平均 log-prob** 构造自适应系数（类似 SimPO）
- 相比原始 sum-log-prob 形式，训练更稳定，收敛更好，尤其在小 $\beta$ 下
- 实验中所有主结果均基于此变体

#### （3）**平衡空间的选择（policy space vs ratio space）**
- 可选择在 $P_w/P_l$ 或 $(P_w/R_w)/(P_l/R_l)$ 空间中平衡梯度
- 实验显示两者在 AlpacaEval 2 上表现几乎一致
- ➤ 说明具体平衡空间不是关键因素

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DPO的梯度不平衡是结构性问题**：
   - 并非由超参数引起，而是共享系数 $\beta$ 导致的固有缺陷。
   - 随着模型信心增强，对偏好响应的梯度趋于消失（vanishing promotion）。

2. **AdaDPO 成功纠正梯度失衡**：
   - 理论上严格满足梯度幅值相等条件；
   - 实验上显著提升 reward accuracy、reward margin 和 β×KL margin（见 Figure 2）。

3. **有效缓解长度偏差**：
   - AdaDPO 在 88% 的超参组合中扩大了 LC−WR 差距；
   - 同时生成更短回应，证明其优势来自**质量提升而非冗余扩展**。

4. **轻量高效，易于部署**：
   - 仅需几行代码修改；
   - 无新增可学习参数或复杂组件；
   - 可无缝迁移至 SimPO、ORPO 等主流方法（见 Table 2）。

---

### ⚠️ 局限性
1. **单一模型 & 单一数据集验证**：
   - 仅在 Llama-3-8B-Instruct + UltraFeedback 上验证；
   - 尚未在 Mistral、Qwen 或更大规模模型上测试泛化性。

2. **在 Arena-Hard 上未显著领先**：
   - DPO 的 WR 略高（42.9% vs 41.5%），但 CI 重叠，**无统计显著差异**；
   - 缺乏长度控制指标，难以剥离 length bias 影响。

3. **适用范围限定于离线偏好优化**：
   - 当前方法适用于 offline setting；
   - 是否适用于 online RLHF 或多模态任务尚待研究。

4. **未直接解决 likelihood displacement 问题**：
   - 与 DPOP、Cal-DPO 等方法关注点不同；
   - 但 AdaDPO 对 $P_w$ 的强化可能间接缓解该现象。

---

### 🔮 未来工作方向
1. **跨模型与跨数据集迁移实验**：
   - 在 Qwen、Mixtral、Mistral 等模型上验证 AdaDPO 效果；
   - 扩展到 HH-RLHF、PKU-SafeRLHF 等安全偏好数据集。

2. **与其他改进方法组合使用**：
   - 结合 DPOP（解决 likelihood displacement）、SimPO（reference-free）等形成更强 pipeline。

3. **探索非对称更新策略（k ≠ 1）**：
   - 当前固定 $k=1$ 实现平衡更新；
   - 可尝试 $k>1$ 强化偏好响应，或 $k<1$ 加强惩罚，寻找最优 trade-off。

4. **推广至其他学习范式**：
   - 探索 AdaDPO 原则在 online RL、multi-agent self-play（如 SPPO）、vision-language 模型中的应用潜力。

---

> 🧭 **总体评价**：  
> AdaDPO 提供了一个简洁、理论严谨且高度实用的视角——**通过细粒度梯度平衡来提升偏好优化效率**。它不仅改进了DPO，也为整个 pairwise contrastive learning family 提供了一种通用优化设计原则。

</details>

---

### 9. [Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization](https://arxiv.org/abs/2605.28109)

**Authors**: Hao Jiang, Shurui Li, Tianpeng Bu, Bowen Xu, Xin Liu, Qihua Chen, Hongtao Duan, Lulu Hu, Bin Yang, Minying Zhang  
**Category**: cs.LG  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.28109v1  

#### Abstract
Recent advances in online reinforcement learning (RL) for large language models (LLMs) have demonstrated promising performance in complex reasoning tasks. However, they often exhibit an imbalanced exploration-exploitation trade-off, resulting in unstable optimization and sub-optimal performance. We ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于在线强化学习（online RL）的大语言模型（LLM）推理训练面临严重的**探索-利用失衡**（exploration-exploitation imbalance）问题：
- **过度利用**（Over-exploitation）：策略过早收敛到局部最优，导致采样轨迹多样性不足，奖励方差稀疏，学习信号衰减。
- **过度探索**（Over-exploration）：如熵正则化等方法虽提升不确定性，但缺乏环境反馈引导，易引发“熵爆炸”和训练不稳定。

现有方法（如GRPO及其变体）无法有效维持这一平衡，限制了模型在复杂推理任务上的性能上限。

### 提出的新方法与新思路
本文提出两个核心创新：

#### （1）**IB-Score：一种基于信息瓶颈理论的细粒度评估指标**
- 将 **Information Bottleneck (IB)** 理论应用于LLM推理过程，定义了一个新的量化指标 **IB-Score**。
- IB-Score 在每一步推理中衡量两个方面：
  - **探索项**：步骤级的推理多样性（通过 Tsallis 熵估计）。
  - **利用项**：该步骤与正确答案之间的互信息（mutual information），反映其对最终答案的信息增益。
- 该指标能有效诊断训练过程中是否存在 over-exploration 或 over-exploitation，并揭示模型是否将置信度合理分配给高价值路径。

#### （2）**IB-TPO：信息瓶颈驱动的树形策略优化框架**
- 提出 **Information Bottleneck-driven Tree-based Policy Optimization (IB-TPO)**，将 IB-Score 显式融入RL目标函数。
- 引入 **IB-guided Tree Search (IBTree)**：
  - 基于 IB-Score 动态选择最具潜力的节点进行分支扩展，实现高效且有针对性的探索。
  - 复用树结构进行高效的 IB-Score Monte Carlo 估计，形成“估计-优化”的正向循环。
- 利用 step-level advantage 设计新的优化目标，结合局部优势（local advantage）与全局优势（global advantage），精细调控探索-利用平衡。

### 相比现有方法的优势
- **更优的探索效率**：IBTree 在相同 token 预算下可生成 **50% 更多的轨迹**。
- **更强的稳定性**：通过 IB-Score 引导，避免了传统熵正则化的“熵爆炸”风险。
- **更高的性能上限**：在多个基准上显著超越 GRPO 及其他 SOTA 方法。
- **统一的理论框架**：首次将 IB 理论系统地用于建模和优化 LLM 推理中的探索-利用权衡。

---

## 2. 核心实验方法和设置

### 数据集
- **训练数据集**：`DAPO-Math-17K`，包含约17K道具挑战性的数学题，配有标准答案。
- **评估数据集**：
  - **数学推理**：`MATH-500`, `AIME 24/25`, `AMC 23/24`
  - **跨领域推理**：`GPQA Diamond`（研究生级别问答）
  - **指令遵循**：`IFEval`

### 实验设置
- **模型**：主要基于 `Qwen3-1.7B-Base` 和 `Qwen3-8B-Base` 进行微调；扩展实验使用 `Qwen3-14B-Base` 和 `Llama3.1-8B-Instruct`。
- **采样参数**：
  - 温度 `T=0.7`，top-p=0.95，top-k=20
  - 最大响应长度为 2K tokens
- **训练细节**：
  - 学习率：`1e-6`
  - KL 正则权重：`0.001`
  - 训练周期：1 epoch
  - 硬件：8×A100 GPUs
- **Token 对齐**：所有方法在 token 消耗上保持一致（例如，独立采样 G=8 vs. IBTree G=12，因共享前缀而实际 token 成本相近）。

### 评估指标
- 主要指标：`avg@32`（低方差准确率）
- 辅助指标：
  - `pass@K`（K次尝试中的通过率）
  - `Eff-Rate`：具有非零奖励方差的有效采样组比例（衡量探索有效性）
  - `IB-Score` 与 `Cov(m1, m2)` 动态变化趋势

### 基线方法对比
- **基础方法**：
  - `Vanilla GRPO`
  - `GRPO w/ clip-higher`
  - `GRPO w/ entropy regularization`
- **SOTA 方法**：
  - `IBRO`（基于IB的序列级熵正则）
  - `TreeRL`（熵引导树搜索）
  - `TreePO`（启发式树建模）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
在 `Qwen3-8B-Base` 上，**IBTPO (Ours)** 相比 `Vanilla GRPO` 的平均提升达 **3.6%**，具体如下：

| 模型 | MATH-500 | AIME 25 | AMC 24 | GPQA | Overall |
|------|----------|---------|--------|------|--------|
| Vanilla GRPO | 81.5% | 17.1% | 53.5% | 38.1% | 40.7% |
| **IBTPO (Ours)** | **83.3%** | **19.7%** | **57.9%** | **41.7%** | **44.3%** |

> ✅ **最大单项提升**：`AMC 24` 上提升 **4.4%**（从 53.5% → 57.9%）

### 与基线方法的对比结果
- IBTPO 在所有测试基准上均取得 **SOTA 性能**，全面优于 GRPO、IBRO、TreeRL 和 TreePO。
- 训练动态显示：
  - IBTPO 的 `Eff-Rate` 更高且更稳定，说明其采样更具多样性。
  - `Cov(m1, m2)` 和 `IB-Score` 维持在更高水平，表明模型成功实现了置信度与信息增益的正相关。

### 消融实验结果（Ablation Studies）

#### （1）组件消融（Table 2）
| 模型配置 | AIME 25 | AMC 24 | GPQA |
|--------|--------|--------|------|
| Vanilla GRPO | 13.6% | 39.4% | 38.1% |
| + IBTree | 15.0% | 43.8% | 40.8% |
| + IBTPO Adv | 14.2% | 42.5% | 41.2% |
| **Full IBTPO (IBTree + IBTPO Adv)** | **15.3%** | **46.0%** | **41.7%** |

> 🔍 结论：两个组件均有贡献，联合使用效果最佳，证明框架协同性强。

#### （2）分支策略比较（Table 3）
| 分支策略 | Eff-Rate↑ | Avg-Rate↑ | 轨迹数↑ |
|--------|----------|-----------|--------|
| 独立采样 (no branching) | 59.8% | 20.1% | 8 |
| 固定宽度 (fix-width) | 59.4% | 19.9% | 12 |
| 熵引导 (entropy-guided) | 57.8% | 21.6% | 12 |
| **IB-Score 引导 (IBTree)** | **60.2%** | **23.2%** | **12** |

> ✅ IBTree 在相同 token 预算下生成 **50% 更多轨迹**（8→12），并获得最高 `Eff-Rate` 和 `Avg-Rate`。

#### （3）超参数分析
- **β=5** 时 IB-Score 表现最优（平衡探索与利用）。
- **λ=0.1** 时局部与全局优势组合效果最好（Table 4）。
- 方法对 `\n\n` 步骤分隔符的扰动鲁棒（Table 5），说明无需精确分步标注。

#### （4）运行时效率（Appendix C.3）
- 尽管多轮迭代略增加延迟，但得益于 vLLM 的 prefix caching 和并行解码，**IBTree(G=12) 的单位问题平均耗时接近独立采样(G=8)**。
- 当对齐 wall-clock 时间时（Table 7），`IBTPO(G=8)` 仅用 **96% 的时间** 即达到远超 `GRPO(G=8)` 的性能。

---

## 4. 关键结论和发现

### 主要发现
1. **探索-利用失衡是当前 online RL 的根本瓶颈**，表现为 `Eff-Rate` 快速下降和 `Cov(m1, m2)` 趋近于零。
2. **IB-Score 是一个有效的诊断工具**，能够细粒度捕捉策略演化过程中的平衡状态。
3. **IB-TPO 成功实现了自适应平衡**：
   - 通过 IBTree 提升探索效率；
   - 通过 IB-based advantage 实现精细优化；
   - 二者相互增强，形成正反馈。
4. **树结构 + IB 指导 > 单纯熵指导或固定结构**，验证了“有反馈引导的智能探索”优于“盲目多样化”。

### 方法的局限性
- **时间开销仍略高**：多轮迭代的树采样机制相比完全并行的独立采样仍有轻微 runtime 开销（尽管 token 效率更高）。
- **依赖 rollout 估计 reward density**：需多次 rollouts 估计 `p(a*|s)`，增加了计算负担。
- 当前主要验证于数学推理任务，需进一步拓展至更多模态和场景。

### 未来工作方向
- 降低树采样的 wall-clock runtime，例如引入异步扩展或缓存机制。
- 将 IB-TPO 扩展到 **multimodal reasoning** 和 **function calling** 场景。
- 探索更高效的 `p(a*|s)` 估计方法，减少 Monte Carlo rollouts 数量。
- 研究如何将 IB-Score 用于离线 RL 或冷启动场景。

---

> 💡 **总结一句话**：  
> 本文提出了 **IB-Score** 和 **IB-TPO**，首次将 **Information Bottleneck** 理论系统应用于 LLM 的 online RL，实现了**可量化、可优化的探索-利用平衡**，在多个复杂推理任务上取得了显著且稳定的性能突破。

</details>

---

### 10. [Meta-Attention: Bayesian Per-Token Routing for Efficient Transformer Inference](https://arxiv.org/abs/2605.28384)

**Authors**: Alan Ferrari  
**Category**: cs.LG  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.28384v1  

#### Abstract
Standard transformer architectures apply a single attention mechanism uniformly across all tokens and sequence positions, irrespective of local context or computational budget. We propose Meta-Attention, a framework that dynamically routes each token to the most appropriate attention strategy -- ful...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Meta-Attention: Bayesian Per-Token Routing for Efficient Transformer Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
标准的 Transformer 架构对所有 token 和序列位置统一应用相同的 attention 机制（如 full softmax attention），无论其上下文复杂度或计算预算如何。这种“一刀切”的设计导致：
- 在局部上下文简单的 token 上浪费大量计算资源（O(n²) 复杂度）；
- 在需要长程依赖的关键位置可能因使用近似 attention 而损失精度。

现有高效 attention 方法（如 sparse、local、linear attention）通常在架构设计时就固定了算法选择，缺乏动态适应能力。

### 提出了什么新方法或新思路
提出 **Meta-Attention**，一种基于贝叶斯推理的 per-token 自适应 attention 路由框架，核心是引入 **Bayesian Meta-Controller**：

- **动态路由机制**：为每个 token 动态选择最合适的 attention 算法——可以是：
  - `E1`: Full Softmax Attention（高成本、高表达力）
  - `E2`: Linear (Kernel) Attention（低成本、线性复杂度）
  - `E3`: Sliding-Window Local Attention（中等成本、局部建模）

- **贝叶斯路由控制器**：
  - 将每 token 的机制选择视为 **后验推断问题**，而非传统的确定性或先验无关的 learned routing。
  - 引入 **compute-aware Dirichlet prior** $ p(\mathbf{o}) = \text{Dir}(\boldsymbol{\beta}) $，其中 $\beta_k = \epsilon + \beta_0(1 - c_k)$，$c_k$ 是各专家的成本，$\epsilon > 0$ 防止退化。
  - 使用变分推断学习 amortised posterior $ q(\mathbf{o}|x) = \text{Dir}(\boldsymbol{\hat{\beta}}) $，输出 per-token 路由权重与不确定性估计。

- **训练目标**：采用 **Evidence Lower Bound (ELBO)** 目标函数：
  $$
  \mathcal{L} = \mathcal{L}_{\text{task}} - \beta_{\text{elbo}} \cdot \text{KL}[q(\mathbf{o}|x) \| p(\mathbf{o})]
  $$
  同时优化任务性能与计算效率，无需额外的 load-balancing 损失。

### 相比现有方法的优势
| 维度 | 传统方法 | Meta-Attention |
|------|--------|---------------|
| **路由策略** | 确定性或 prior-free 学习路由 | **贝叶斯后验推断**，具有 principled 不确定性量化 |
| **计算偏好编码** | 手动调参或无显式建模 | 通过 **Dirichlet prior 显式编码成本偏好** |
| **防止路由崩溃** | 依赖 ad-hoc 正则项（如负载均衡） | **天然防止 collapse**（如避免全用 E1） |
| **软硬路由过渡** | 启发式阈值 | 使用 **后验熵作为软-硬切换信号** |
| **可组合性** | 通常独立 | 可与 MoD、AttnRes 等正交方法组合 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Phase 1 实验**：使用一个小型字符级语言模型在 **1MB WikiText-2 子集**上进行消融研究。
- **未来计划（Phase 2）**：将在标准基准 **WikiText-103** 上验证绝对 perplexity 表现。

### 实验设置
- **模型结构**：Tiny LM，2 层 Transformer，隐藏维度 D=128，序列长度 T=64。
- **训练配置**：2000 步，batch size=32，Adam 优化器。
- **专家定义**：
  - `E1` (Full): O(T²D), 归一化成本 $c_1=1.0$
  - `E2` (Linear): O(TD), $c_2=0.15$
  - `E3` (Local): O(TwD), $c_3=0.30$

### 评估指标
| 指标 | 描述 |
|------|------|
| **Normalized PPL** | 相对于非贝叶斯基线的相对困惑度 |
| **Routing Entropy (%)** | 路由分布的平均熵，衡量决策集中程度 |
| **Projected FLOP Cost (%)** | 基于当前路由分布预测的归一化 FLOP 成本（假设 Phase 3 硬路由生效） |
| **Projected Cost Ratio** | 贝叶斯 vs. 非贝叶斯的预期 FLOP 比值 |

### 基线方法对比
- **Bayesian (ours)**：完整 Meta-Attention 框架，$\beta_{\text{elbo}}=1$，使用 floored Dirichlet prior $\boldsymbol{\beta}=[0.01, 0.86, 0.71]$
- **Non-Bayesian / Prior-Free Baseline**：$\beta_{\text{elbo}} \to 0$，即移除 KL 正则项，退化为普通 MLP 路由

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

| 指标 | Bayesian (ours) | Non-Bayesian (baseline) | 对比 |
|------|----------------|--------------------------|------|
| **Normalized PPL** | 1.07 | 1.00 (ref) | +6.3% |
| **Routing Entropy (%)** | **43.3%** | 55.8% | ↓12.5 pp |
| **Projected FLOP Cost (%)** | **25.1%** | 59.3% | ↓34.2 pp |
| **Projected Cost Ratio** | — | — | **2.4× 更低** |

### 与基线方法的对比结果
- **FLOP 成本大幅降低**：贝叶斯控制器将预计 FLOP 成本从 59.3% 降至 **25.1%**，实现 **2.4 倍的效率提升**。
- **路由更稳定、更集中**：路由熵下降 12.5 个百分点，表明 posterior 分布更集中，有利于 Phase 3 的 hard routing。
- **轻微性能代价换取显著效率增益**：困惑度仅上升 6.3%，但在极小代价下获得巨大计算节省，符合 ELBO 设计初衷。

### 消融实验结果
- **控制变量实验**：唯一区别是是否启用 KL 正则项（即是否使用贝叶斯 prior），其他一切相同。
- **结果验证核心假设**：
  - 非贝叶斯模型趋向于“collapse”到昂贵的 full attention（投影成本高达 59.3%）；
  - 贝叶斯模型成功引导路由向廉价专家（E2/E3）偏移，且无需手动调节正则系数 $\lambda$。
- **初始化检查**：前向传播验证表明，Dirichlet prior 在初始阶段即正确引导路由偏向低成本专家（E2 权重 ~40.6% > 33.3% 均匀分布）。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **贝叶斯 prior 能有效防止 routing collapse**：通过 compute-aware Dirichlet prior，自然抑制模型对昂贵 full attention 的过度依赖。
2. ✅ **更好的 compute-performance trade-off**：相比 prior-free 方法，在仅增加 6.3% 相对 PPL 的前提下，实现 **2.4× 的预期 FLOP 下降**。
3. ✅ **提供 principled 不确定性估计**：per-token 后验熵可用于指导 soft-to-hard routing 的过渡时机。
4. ✅ **无需 ad-hoc 正则化**：ELBO 框架本身已包含平衡任务性能与计算成本的能力，无需额外设计 load balancing 损失。
5. 🔮 **与理论前沿高度契合**：
   - 支持 [24] VMoER 发现：贝叶斯路由提升稳定性与校准性；
   - 符合 [18] Sparse Attention Emergence 预测：贝叶斯模型更早进入低熵状态，预示结构化路由的提前形成；
   - 呼应 [19] 理论：Linear attention 是 PAC-learnable 的最优候选，应被优先选用。

### 方法的局限性
| 限制 | 说明 |
|------|------|
| **尚未实现真实加速** | 当前 Phase 1 使用 soft routing，三个专家并行运行，**未真正减少 FLOPs 或 wall-clock 时间**。实际加速需等待 Phase 3 的 uncertainty-gated hard routing。 |
| **小规模实验环境** | 当前结果基于 Tiny LM 和 WikiText-2 子集，**缺乏大规模语言建模（如 WikiText-103）上的绝对 PPL 验证**。 |
| **梯度方差未知** | Dirichlet reparameterization 的梯度方差尚未在大尺度训练中表征，可能存在训练不稳定性风险。 |
| **prior 敏感性** | concentration 参数 $\beta_0$ 的选择可能影响性能，最优值可能随任务和深度变化。 |
| **未包含 SSM 专家** | 如 Mamba、RetNet 等 State Space Model 未纳入专家集合（Phase 1 决策），接口异构性（有状态 vs 无状态）是主要障碍。 |

### 未来工作方向（Phased Roadmap）
| Phase | 目标 |
|-------|------|
| **Phase 2** | 
| • 在 **WikiText-103** 上训练，报告绝对 PPL 与 FLOP-PPL 曲线  
| • 分析 posterior concentration 与 KL 曲线  
| • 验证 routing entropy 是否出现 sharp phase transition  
| • 进行 repetition-curriculum ablation [18]  
| • 对比 Bayesian vs prior-free 在相同 FLOP 预算下的表现 |
| **Phase 3** | 
| • 实现 **uncertainty-gated hard routing**（$U < \eta$ 时只运行 argmax expert）  
| • 测量真实 **wall-clock FLOP ratio** 与吞吐量  
| • 探索 $\eta$ 的 annealing 策略与 sensitivity |
| **Long-term Extensions** |
| • 引入 **SSM 专家**（如 Mamba）作为 E4  
| • 升级 `E1` 为 **gated softmax attention** [17]  
| • 探索 **empirical Bayes** 方法自动估计 $\beta_0$  
| • 与 **Mixture of Depths (MoD)** 和 **AttnRes** 组合成三重架构 |

---

> 📌 **代码开源**：作者已发布 PyTorch 原型实现，地址：[https://github.com/KFEAL/meta-attention](https://github.com/KFEAL/meta-attention)  
> 🧪 **可证伪标准（Falsifiability Criterion）**：在 ≥4k 长序列上，Meta-Attention 应实现 ≥30% FLOP 减少，且 WikiText-103 困惑度恶化不超过 0.5 nats。Phase 1 结果已初步支持该标准（预计 FLOP 减少达 64%）。

</details>

---

### 11. [Bridging the Detection-to-Abstention Gap in Reasoning Models under Insufficient Information](https://arxiv.org/abs/2605.28070)

**Authors**: Renjie Gu, Jiaxu Li, Yihao Wang, Yun Yue, Hansong Xiao, Yefei Chen, Yuan Wang, Chunxiao Guo, Pei Wei, Jinjie Gu, Yixin Cao  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.28070v1  

#### Abstract
We highlight a failure mode of large reasoning models on questions with insufficient information: models may recognize that a problem is under-specified, yet still continue reasoning and produce unsupported final answers instead of abstaining. We formalize this mismatch as the detection-to-abstentio...

---

### 12. [Agentic Active Omni-Modal Perception for Multi-Hop Audio-Visual Reasoning](https://arxiv.org/abs/2605.28192)

**Authors**: Ke Xu, Yuhao Wang, Ziyang Cheng, Hongcheng Liu, Yanfeng Wang, Yu Wang  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.28192v1  

#### Abstract
Multi-hop audio-visual reasoning remains challenging for Omni-LLMs, as relevant evidence is often sparse, temporally dispersed, and distributed across both audio and visual streams. Existing benchmarks provide limited investigation of this setting, typically involving only a limited number of modali...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Agentic Active Omni-Modal Perception for Multi-Hop Audio-Visual Reasoning*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Omni-LLMs** 在处理 **multi-hop audio-visual reasoning**（多跳音视频推理）任务时面临显著挑战。这类任务要求模型整合在时间上分散、跨模态分布的稀疏证据（如视觉片段中的动作与音频中的对话），而现有方法通常采用被动、端到端的全视频编码方式，难以有效定位和集成这些关键线索。

此外，现有的 **benchmark**（如 OmniVideoBench）任务构成复杂且异构，无法专门评估跨模态多跳推理能力，导致对模型真实能力的理解受限。

### 提出的新方法与新思路
为解决上述问题，本文提出两大核心贡献：

#### ✅ **MOV-Bench**  
一个专注于 **multi-hop audio-visual reasoning** 的高质量基准测试集，包含 **519 个精心设计的多选题**，具有以下特点：
- 要求跨多个时间片段和模态（audio + visual）进行推理；
- 推理链涵盖 **2~4 hops**，平均 2.6 hops；
- 问题围绕五种推理类型构建：**Causal, Referential, Relational, Hypothetical, Intent Reasoning**；
- 强调真实跨模态依赖，避免语言先验或单模态可解的“shortcut”。

#### ✅ **AOP-Agent**（Active Omni Perception Agent）
一种面向开源 Omni-LLMs 的低资源 **agentic 框架**，实现主动感知（active perception），无需额外训练或闭源模型。其核心机制包括：
- **Hierarchical Omni-modal Memory**：将长视频组织为多层次记忆结构（全局摘要 → 中级语义段 → 关键点 → 细粒度片段），支持粗到细的证据检索；
- **Collaborative Observe-Reflect-Replan Loop**：由 Planner、Reflector 和 Reasoner 多智能体协作，动态决定观察目标、评估证据充分性并调整策略。

### 相比现有方法的优势
| 方面 | AOP-Agent | 现有方法（如 OmniAgent, ActiveVideoPerception） |
|------|-----------|---------------------------------------------|
| **资源需求** | 使用开源 Omni-LLMs，无需训练或闭源模型 | 依赖强大闭源模型（如 Gemini/OpenAI）进行规划 |
| **效率与适应性** | 分层记忆降低搜索难度，多轮迭代聚焦关键区域 | 易因错误累积导致性能下降 |
| **有效性** | 在 MOV-Bench 上显著优于直接推理和其他 agentic 方法 | 在开源设置下表现差于直接推理 |

---

## 2. 核心实验方法和设置

### 数据集
- **MOV-Bench**（本文提出）：
  - 来源于 Fine-Video 数据集；
  - 包含 519 个多跳音视频问答样本；
  - 平均视频长度 240.26 秒，其中超过 5 分钟的长视频占 36.42%；
  - 按推理步数分为：2-hop (295), 3-hop (138), 4-hop (86)。
- **OmniVideoBench**（外部通用 benchmark）：
  - 用于验证 AOP-Agent 的泛化能力；
  - 包含更广泛的音视频理解任务。

### 实验设置与评估指标
- **评估任务**：多跳音视频问答准确率（Accuracy）；
- **分组评估维度**：
  - 视频长度：Short (<150s), Medium (150–300s), Long (>300s)；
  - 推理类型：Causal, Referential, Hypothetical, Relational, Intent；
  - 推理步数：2-hop, 3-hop, 4-hop；
- **骨干模型**：Qwen3-Omni-Instruct, Qwen3-Omni-Thinking, Qwen2.5-Omni-7B 等开源 Omni-LLMs；
- **最大上下文长度**：32,768 tokens；
- **温度设置**：Memory 构建阶段为 1.0，agentic 过程为 0.2。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **直接推理** | 各 Omni-LLM 的 zero-shot 推理（Baseline） |
| **Agentic 方法** | OmniAgent (Tao et al., 2026), ActiveVideoPerception (Wang et al., 2025e) —— 均用 Qwen3-Omni-Instruct 替换原闭源 backbone 以公平比较 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### 在 **MOV-Bench** 上的表现（Overall Accuracy）
| 方法 | 准确率 |
|------|--------|
| Qwen3-Omni-Instruct (Direct) | 52.79% |
| **AOP-Agent + Qwen3-Omni-Instruct** | **62.62%** (+9.83%) |
| OmniAgent (reimplemented) | 38.34% |
| ActiveVideoPerception (reimplemented) | 35.27% |

> 🔺 AOP-Agent 显著提升所有 backbone 模型性能，尤其在长视频和高 hop 数问题上增益明显。

#### 长视频子集（Long Video Subset）性能对比
| 方法 | MOV-Bench (Long) | OmniVideoBench (Long) |
|------|------------------|------------------------|
| Qwen3-Omni-Instruct (Direct) | 45.50% | 28.45% |
| **AOP-Agent** | **60.85%** | **40.52%** |

> ✅ 提升幅度达 **+15.35pp**（MOV-Bench）和 **+12.07pp**（OmniVideoBench），表明 AOP-Agent 特别适合处理稀疏、分散证据的长视频场景。

#### 不同推理步数下的表现（以 Qwen3-Omni-Instruct 为例）
| 推理步数 | Direct Inference | AOP-Agent | 提升 |
|---------|------------------|-----------|------|
| 2-hop | 54.92% | 64.75% | +9.83% |
| 3-hop | 49.28% | 61.59% | +12.31% |
| **4-hop** | **51.16%** | **56.98%** | **+5.82%** |

> 📈 表明 AOP-Agent 对 **推理密集型任务** 更具优势。

---

### 消融实验结果（Ablation Study）

#### 组件消融（Table 3）
在 MOV-Bench 长视频子集上的消融显示：
| 配置 | 准确率（MOV Long） |
|------|--------------------|
| 仅 Reasoner（Baseline） | 45.50% |
| + Planner | 50.26% |
| **+ Planner + Reflector（完整版）** | **60.85%** |

> ✅ **Planner 和 Reflector 协同作用至关重要**：Planner 实现定向观察，Reflector 通过反馈防止无效探索。

#### 模型分配影响（Table 4）
即使最终 Reasoner 固定，更强的 Planner/Reflector 也能带来更大收益：
| Planner / Reflector | Reasoner | MOV Long |
|---------------------|----------|----------|
| Qwen2.5-Omni-7B | Qwen2.5-Omni-7B | 41.27% |
| Qwen3-Omni-30B | Qwen2.5-Omni-7B | 48.68% |
| Qwen2.5-Omni-7B | Qwen3-Omni-30B | 58.20% |
| **Qwen3-Omni-30B** | **Qwen3-Omni-30B** | **60.85%** |

> 🔍 表明 **高质量的 planning 与 reflection 是成功的关键前提**。

#### 观察轮次的影响（Figure 5）
- 最佳性能出现在 **3 轮观察**；
- 超过 3 轮后性能趋于饱和甚至轻微下降；
- 表明 **bounded observation** 更优，避免引入噪声。

---

## 4. 关键结论和发现

### 主要发现
1. **当前 Omni-LLMs 的瓶颈不仅是推理本身，更是证据获取能力**：
   - 在 MOV-Bench 上表现不佳，尤其是在长视频和多跳任务中；
   - 表明被动处理全视频的方式难以应对稀疏、分散的跨模态证据。

2. **主动感知（active perception）是解决该问题的有效范式**：
   - AOP-Agent 通过 **observe-reflect-replan loop** 实现渐进式证据收集；
   - 显著提升多跳推理性能，尤其在复杂场景下。

3. **AOP-Agent 在低资源条件下仍高效可行**：
   - 仅使用开源 Omni-LLMs，无需训练或闭源组件；
   - 分层记忆 + 多智能体协作降低了主动感知的实施门槛。

4. **Planning 与 Reflection 决定了系统成败**：
   - 高质量的 Planner 能引导有效搜索路径；
   - Reflector 可抑制错误传播，确保推理稳定性。

---

### 局限性
1. **MOV-Bench 规模有限**：受人工标注成本限制，仅含 519 个样本；
2. **推理延迟增加**：multi-round 的 observe-replan 引入额外计算开销；
3. **幻觉与错误传播风险**：
   - 若 hierarchical memory 中存在 hallucinated 描述（如 Figure 8 所示），可能导致错误因果推断；
   - 当前 Reflector 尚不能完全检测此类问题。
4. **非流式友好**：memory 构建依赖离线多级处理，不适用于实时场景（如直播、智能眼镜）。

---

### 未来工作方向
- 扩展 MOV-Bench 规模，引入更多样化的推理模式；
- 设计更鲁棒的 hallucination detection 机制，增强 memory 可靠性；
- 开发轻量化 streaming-friendly memory 构建流程，支持实时应用；
- 探索 AOP-Agent 在其他多模态任务（如机器人交互、医疗视频分析）中的迁移能力。

--- 

> ✅ **总结一句话**：  
> 本论文揭示了当前 Omni-LLMs 在 **multi-hop audio-visual reasoning** 中的根本瓶颈，并提出了 **AOP-Agent** 这一高效的 agentic 框架，在无需训练或闭源模型的前提下，通过 **分层记忆 + 主动感知循环** 显著提升了复杂音视频推理能力，为开源多模态智能体的发展提供了新路径。

</details>

---

### 13. [Addressing Variable Heterogeneity in Distributed Multimodal Training with Entrain](https://arxiv.org/abs/2605.27918)

**Authors**: Insu Jang, Mosharaf Chowdhury  
**Category**: cs.DC  
**Published**: 2026-05-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.27918v1  

#### Abstract
Multimodal LLM datasets are inherently heterogeneous, with significant data variability. Although each modality exhibits independent variability, sample-level entanglement makes it difficult to balance workloads across both modalities and batches. We present Entrain, a distributed MLLM training fram...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Addressing Variable Heterogeneity in Distributed Multimodal Training with Entrain

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
多模态大语言模型（MLLM）训练面临显著的数据异质性和变异性问题：
- **Workload Heterogeneity**：不同模态（如图像、文本）的计算特性差异巨大（例如，视觉编码器与LLM的算术强度和内存访问模式不同）。
- **Workload Variability**：每个样本中各模态的工作负载比例波动剧烈且独立分布，导致微批次（microbatch）间的负载极不平衡。
- 现有方法在**profiling阶段**（基于单个样本或小批量进行并行配置）和**执行阶段**（无法动态平衡微批次间负载）均存在缺陷。

### 提出了什么新方法或新思路
提出 **Entrain**，一个分布式 MLLM 训练框架，从宏观和微观两个尺度协同解决异质性与变异性问题：

#### 宏观层面：基于全局批次的静态并行配置（Macroscopic Profiling）
- **核心思想**：尽管单个样本的模态负载比高度不稳定，但在**全局批次（global batch）** 尺度下，模态间的聚合负载比会通过**大数定律（Law of Large Numbers）** 收敛到一个稳定常数。
- 因此，无需昂贵的动态并行重配置（dynamic reconfiguration），仅需一个**静态的、基于宏观负载比**的并行配置即可实现最优负载均衡。
- 创新性地将 profiling 对象从“微观样本”转移到“宏观批次”，并通过理论证明其有效性。

#### 微观层面：分层微批次分配与延迟机制（Hierarchical Microbatch Assignment with Deferral）
- 在宏观配置基础上，为缓解微批次内的局部不平衡，提出**解耦调度策略**：
  1. **分层样本分配（Stratified Assignment）**：优先均衡编码器（producer）的负载，确保恒定生产速率。
  2. **成对延迟优化（Pairwise Deferral）**：允许将高负载样本的 LLM 计算任务“延迟”到后续低负载的微批次中处理，从而均衡消费者（LLM）的执行时间。
- 引入 **split-backward processing** 和 **eager forward execution** 来优化反向传播依赖带来的开销。

### 相比现有方法的优势
| 方面 | 现有方法（如 DistTrain, DIP） | Entrain |
|------|-------------------------------|--------|
| **Profiling 粒度** | 基于单样本或小 microbatch，易受噪声影响 | 基于大批次，捕获真实数据分布，配置更鲁棒 |
| **并行配置** | 静态但不准确，或尝试动态但开销大 | 静态且最优，避免动态开销 |
| **执行时负载均衡** | 有限（如重排序、子微批次划分） | 主动通过“延迟”机制跨微批次转移负载 |
| **理论基础** | 缺乏对宏观稳定性的认识 | 明确利用大数定律证明静态配置的充分性 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
在四个具有不同分布特征的视觉-语言数据集上进行评估：
- **SynthChartNet**：合成图表问答，变异性最高（主文本分析）
- **ChartQA**：真实图表问答
- **CocoQA**：基于 COCO 图像的问答
- **LLaVA-150k**：大规模多模态指令微调数据集

### 实验设置和评估指标
- **硬件环境**：16块 NVIDIA A40 GPU（4节点），通过顺序执行 DP replica 模拟 64-GPU 集群。
- **模型架构**：
  - 视觉编码器：Qwen2.5Vision
  - LLM：Llama3-1b 和 Llama3-3b
- **并行策略**：4D 并行（DP=4, TP=2, CP=1），PP 度可变。
- **关键参数**：
  - 全局批次大小（Global Batch Size）：512
  - 微批次大小（Microbatch Size）：4 → 共 128 个 microbatches
- **评估指标**：
  - **端到端训练吞吐量（End-to-end Training Throughput）**
  - **微批次间前向传播时间的标准差（std of forward time）**
  - **工作负载变异性降低倍数**
  - 内存消耗（Memory Consumption）

### 基线方法对比
- **DistTrain [52]**：通过样本重排序减少 pipeline bubbles，profiling 使用 1 个样本。
- **DIP [48]**：引入解耦的模态调度，profiling 使用 1 个 microbatch（4 个样本）。
- **Entrain**：profiling 使用 256 个样本以保证统计收敛。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **端到端训练吞吐量提升**：
  - Entrain 相比 DistTrain 和 DIP，**最高提升达 1.40×**。
  - 在 SynthChartNet 上提升最为显著，因其数据变异性最大。

- **工作负载变异性大幅降低**：
  - Entrain 将微批次间的负载变异性**最多降低了 10.6×**。
  - 以 **SynthChartNet + Llama3-1b** 为例（Table 3）：
    - Vision 前向时间 std：DistTrain (208.07) → Entrain (19.60)，**降低 10.6×**
    - LLM 前向时间 std：DistTrain (77.92) → Entrain (18.79)，**降低 4.1×**

- **宏观数值稳定性验证**：
  - 表 2 和附录表 5–11 显示，当 profiling 批次大小达到 256 时，所有随机批次均给出相同的 GPU 分配方案（如 8:8 或 10:6），而小批次则波动剧烈。

### 与基线方法的对比结果
| 指标 | DistTrain | DIP | Entrain |
|------|----------|-----|--------|
| 吞吐量 | 基线 | 基线 | **↑ 最多 1.40×** |
| 负载变异性 | 高 | 中等 | **↓ 最多 10.6×** |
| 内存峰值 | 较高（因不平衡） | 极高（需缓存全部 encoder 输出） | **更低且更稳定** |
| 配置鲁棒性 | 差（依赖单一样本） | 一般（依赖小批量） | **优（基于全局分布）** |

- **图 12 可视化**显示，Entrain 的 pipeline schedule 更加均匀，几乎没有气泡（bubbles），而 DistTrain 和 DIP 存在明显空闲周期。

### 消融实验结果
- **敏感性分析（图 14）**：若使用错误的并行配置（非宏观最优），Entrain 的吞吐量会**下降高达 85%**，证明其对配置质量的敏感性和宏观 profiling 的必要性。
- **宏观数值收敛性（图 5）**：随着批次大小增加，模态负载比迅速收敛至稳定均值，为静态配置提供了理论依据。

---

## 4. 关键结论和发现

### 主要发现
1. **宏观稳定性**：尽管 MLLM 数据在样本级别高度异质，但在**全局批次尺度下，模态负载比趋于稳定**，这使得**单一静态并行配置足以实现最优负载均衡**。
2. **微观可解耦性**：通过将编码器视为生产者、LLM 作为消费者，可以**解耦两者的调度边界**，利用 pipeline buffer 吸收负载波动。
3. **延迟机制的有效性**：**成对延迟（pairwise deferral）** 是一种高效且实用的方法，可在不破坏前向流程的前提下，显著平滑 LLM 的执行时间。
4. **静态优于盲目动态**：相比为应对变异性而设计的复杂动态机制，**基于正确宏观洞察的静态设计反而更高效、更稳定**。

### 方法的局限性
- **依赖大批次 profiling**：需要预先知道或估计足够的 `b_min`（实验中为 256），对于流式或在线训练场景可能不适用。
- **延迟带来额外内存开销**：需临时存储被延迟样本的 encoder 激活值，虽然实验证明开销很小，但在极端长序列或多模态场景下仍需关注。
- **假设 workload 可完整迁移**：延迟的是整个样本的 LLM 计算，未支持细粒度分割（如按 token 分片），限制了灵活性。

### 未来工作方向
- 将 Entrain 的思想扩展到**更多模态**（如音频、视频、点云）的统一训练框架。
- 探索在**推理阶段**应用类似机制以应对动态输入变异性。
- 结合 **zero-bubble pipeline parallelism (ZBPP)** 进一步压缩迭代时间。
- 研究如何自适应地确定 `b_min`，以适应不同数据分布和模型规模。

--- 

> **总结**：Entrain 通过深刻的洞察——“**宏观稳定、微观可调**”——挑战了“动态数据需动态并行”的直觉，提出了一种简洁而强大的解决方案，在保证系统效率的同时，显著提升了 MLLM 的训练吞吐量和稳定性。

</details>

---

### 14. [SKILLC: Learning Autonomous Skill Internalization in LLM Agents via Contrastive Credit Assignment](https://arxiv.org/abs/2605.27899)

**Authors**: Hongxiang Lin, Zhirui Kuai, Erpeng Xue, Lei Wang  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.27899v1  

#### Abstract
Structured skill prompts improve exploration in long-horizon agentic reinforcement learning (RL). Skill-augmented RL methods retain external skills at inference, while skill-internalization RL methods withdraw them during training to enable autonomous performance. However, existing internalization a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SKILLC: Learning Autonomous Skill Internalization in LLM Agents via Contrastive Credit Assignment

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在 **Agentic Reinforcement Learning (RL)** 中，虽然结构化技能提示（structured skill prompts）能有效提升长视野任务的探索效率，但大多数方法属于 **skill-augmented RL** 范式——即在推理时仍依赖外部技能支持。这导致模型无法实现真正的自主行为。

更进一步的 **skill-internalization RL** 方法（如 SKILLO）尝试在训练过程中逐步撤回技能，使策略最终能独立执行任务。然而，这些方法存在一个关键缺陷：**内部化盲区（internalization blindness）**。  
具体表现为：
- 在每个 GRPO 更新中，所有轨迹共享相同的提示条件（要么全用技能，要么不用），因此梯度无法区分“成功是依赖技能还是已自主完成”。
- 政策更新目标未改变，仅靠课程控制（curriculum）被动转移能力，缺乏对“自主成功”的直接学习信号。

### 提出了什么新方法或新思路
作者提出 **SKILLC**，基于 **Contrastive Skill Credit Assignment (CSCA)** 的框架，将技能有用性对比转化为直接的学习信号，推动技能内化。

其核心机制包括：

#### ✅ **Paired Contrastive Rollouts（成对对比轨迹采样）**
- 对活跃技能类型的任务，在同一策略更新中同时采样 **skill-injected** 和 **skill-free** 的轨迹对。
- 显式暴露任务级别的 **internalization gap**：  
  $$
  \Delta(x) = \mathbb{E}[R|x,z=1] - \mathbb{E}[R|x,z=0]
  $$
  当 $\Delta(x)\to0$ 表示技能已被内化。

#### ✅ **Dual-Stream Advantage Estimator（双流优势估计器）**
为避免传统联合归一化带来的偏差（skill-injected 高奖励拉高基线，抑制 skill-free 成功的信用分配），设计两个互补流：
- **Stream 1**: 全局排名项，保留整体性能排序。
- **Stream 2**: 条件内归一化 + 对比修正项：
  - 若技能仍有帮助（$\Delta(x)>0$），则给 skill-free 成功加分，给 skill-injected 成功减分。
  - 学习目标明确：“即使没有技能也要成功”。

#### ✅ **Internalization-Aware Adaptive Curriculum（内化感知自适应课程）**
利用验证集上的平滑对比信号 $\Delta_{\text{val}}(k)$ 动态调整：
- 技能归因强度（attribution strength）
- skill-injected rollout 分配比例
- 多类别环境中的活跃技能集合（单调剪枝）

该课程完全由数据驱动，无需预设时间表。

### 相比现有方法的优势
| 维度 | SKILLO（代表基线） | SKILLC（本文） |
|------|------------------|--------------|
| 内部化机制 | 固定调度撤回技能 | 数据驱动动态调整 |
| 梯度信号 | 无显式自主成功引导 | 明确对比信用分配 |
| 归一化方式 | 单一流、混合归一化 | 双流、条件内归一化 |
| 是否解决 internalization blindness | ❌ 否 | ✅ 是 |

> SKILLC 将“是否还需要技能”这一元认知信号，转化为可微分的优化路径，实现了从“被动遗忘”到“主动掌握”的转变。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 |
|-------|------|
| **ALFWorld** | 文本版家庭环境，对应 ALFRED 实体基准，包含六类任务：<br>- Pick, Look, Clean, Heat, Cool, Pick2<br>需多步规划与对象交互，适合多技能评估 |
| **WebShop** | 真实电商购物环境，要求根据自然语言指令进行搜索、浏览、购买决策<br>构造单一“购物技能”，用于单技能场景测试 |

### 实验设置和评估指标
- **基础模型**：Qwen2.5-7B-Instruct
- **训练配置**：8×A100-80GB GPU，batch size=8，group size=8
- **超参数统一设置**：$\lambda=1.0$, $w=0.4$, $\alpha=0.9$, $d=10$

#### 评估指标
| 指标 | 定义 | 应用范围 |
|------|------|---------|
| **SR_without** (%) | 无技能访问下的任务成功率 | 所有 skill-internalization 方法 |
| **SR_with** (%) | 有技能支持的成功率 | skill-augmented 方法 |
| **Task Score (0–100)** | 属性相关性得分 | WebShop |
| **Internalization Gap $\Delta_{\text{val}}$** | $SR_{\text{with}} - SR_{\text{without}}$，衡量内化进度 | ALFWorld 验证集 |

> 所有结果取 5 次独立运行平均值。

### 基线方法对比
分为五类进行比较：

| 类别 | 代表方法 |
|------|--------|
| **Training-free** | ReAct, Reflexion, Mem0, ExpeL |
| **Direct RL** | PPO, RLOO, GRPO, GiGPO |
| **On-Policy Self-Distillation** | OPSD, Skill-SD, RLSD, SDAR |
| **Skill-Augmented RL** | EvolveR, SKILLRL, RetroAgent, Skill1, D2Skill |
| **Skill-Internalization RL** | SKILLO, **SKILLC（本文）** |

> 特别强调与 **SKILLO** 的对比，因其为当前最先进的 skill-internalization 方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Test Split）

| 方法 | ALFWorld (SR_without, %) | WebShop (SR_without, %) |
|------|----------------------------|----------------------------|
| **SKILLO** | 85.9 | 70.9 |
| **SKILLC（本文）** | **90.6** (+4.7%) | **74.0** (+3.1%) |

> 在不使用运行时技能的情况下，SKILLC 显著超越最强的 skill-internalization 基线。

#### 更细粒度表现（ALFWorld 各子任务）
| 方法 | Pick | Look | Clean | Heat | Cool | Pick2 | Avg |
|------|------|------|-------|------|------|-------|-----|
| SKILLO | 100.0 | 84.6 | 91.1 | 84.2 | 88.9 | 55.5 | 85.9 |
| **SKILLC** | 88.5 | 78.6 | 91.2 | **94.7** | **100.0** | **88.9** | **90.6** |

- 在较难任务（Heat, Cool, Pick2）上提升显著（+10.5%, +11.1%, +33.4%）
- 简单任务略有下降但仍保持高水平
- 总体表明：**CSCA 更擅长促进复杂技能的真正内化**

#### 与 skill-augmented 方法的横向对比
| 方法 | ALFWorld (SR_with, %) | 是否允许技能 |
|------|------------------------|-------------|
| D2Skill | 90.6 | ✅ 是 |
| **SKILLC** | **90.6** | ❌ 否 |

> SKILLC 在 **完全没有技能支持下** 达到了与最强 skill-augmented 方法相当的性能，证明其内化效果极佳。

---

### 消融实验结果（Ablation Study on ALFWorld）

| 变体 | SR_without (%) | Δ 相对完整模型 |
|------|----------------|---------------|
| **完整 SKILLC** | **90.6** | ref. |
| 无成对 rollout（single-condition） | 87.3 | -3.3 |
| 混合归一化 + 固定门控 | 88.4 | -2.2 |
| 无活跃技能剪枝 | 89.5 | -1.1 |
| 固定 rollout 比例 $p_{\text{with}}=0.5$ | 89.8 | -0.8 |
| 初始乐观评分（high init score） | 87.1 | -3.5 |

> 结果显示：
- **paired rollouts** 是最大增益来源（-3.3% drop）
- **dual-stream normalization** 至关重要（防止 baseline inflation）
- 自适应 curriculum 提供稳定增益，尤其当三者协同作用时

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Internalization Blindness 是真实存在的瓶颈**  
   - SKILLO 中 with-skill 与 without-skill 的 SR 差距在整个训练过程持续存在，直到强制撤回才暴露“内化债务”。
   - SKILLC 通过持续的对比信号，使 gap 平稳收敛至零以下，实现渐进式内化。

2. ✅ **对比信用分配（CSCA）是有效的学习机制**  
   - 不再等待“某时刻突然断电”，而是从第一天就鼓励“不用技能也能赢”。
   - 双流结构既保持全局排序，又精准施加自主偏好。

3. ✅ **自适应课程优于固定调度**  
   - 内部化速度因任务而异（如 Heat 比 Pick 慢），固定预算会过早或过晚撤回技能。
   - SKILLC 的门控机制自动同步 gate、rollout ratio、active set，形成闭环反馈。

4. ⚠️ **性能增益依赖信号质量**  
   - 强对比信号 → 大幅提升（如 Pick2 +33.4%）
   - 弱信号或天花板效应任务 → 可能轻微退化（如 Pick -11.5%）
   - 单一技能环境（WebShop）→ 改进有限（+3.1%）

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **Skill Inventory Structure 假设强** | 需预先定义任务对齐的技能库；开放域中技能动态变化时可能失效 |
| **Validation-Level Bottleneck** | 验证集覆盖不足或类别极度不平衡时，平滑后的 $\Delta_{\text{val}}$ 易受噪声干扰，可能导致误判退休 |
| **计算开销较高** | 早期 paired sampling 导致约 +26% 时间开销（见 Table 3），虽随技能退休降低，但在大规模训练中仍具挑战 |

---

### 未来工作方向
1. **Dynamic Skill Discovery**  
   探索在训练中自动发现和生成新技能的能力，而非依赖预建技能库。

2. **Generalization to Hierarchical Skill Structures**  
   将 CSCA 扩展至层级化技能树，处理更复杂的抽象动作组合。

3. **Efficiency Optimization**  
   设计轻量级替代方案（如交替 rollout、蒸馏近似）以减少 paired sampling 开销。

4. **Robust Curriculum Control under Noise**  
   引入不确定性建模或贝叶斯平滑，增强验证信号在稀疏任务下的稳定性。

---

> 💡 **一句话总结**：  
> SKILLC 通过 **paired contrastive rollouts + dual-stream advantage estimation + data-driven curriculum**，首次将“技能是否已被内化”的判断转化为可学习的梯度信号，解决了 **internalization blindness** 问题，在 ALFWorld 上实现 **90.6% without-skill success rate**，显著优于此前最优方法，并媲美带技能支持的强基线。

</details>

---

### 15. [Efficient Post-training of LLMs for Code Generation With Offline Reinforcement Learning](https://arxiv.org/abs/2605.28409)

**Authors**: Mingze Wu, Abhinav Anand, Shweta Verma, Mira Mezini  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.28409v1  

#### Abstract
Post-training using online reinforcement learning (RL) is an important training step for LLMs, including code-generating models. However, online RL for code generation involves LLM inference and verification of the generated output, which can take considerable time and resources. In this paper, we e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Efficient Post-training of LLMs for Code Generation With Offline Reinforcement Learning*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
传统的 **online Reinforcement Learning (RL)** 在对代码生成类大语言模型（LLMs）进行后训练（post-training）时存在显著瓶颈：
- 需要频繁调用模型推理（inference）生成代码；
- 必须通过执行代码来验证功能正确性（如运行测试用例），成本高昂；
- 容易出现 **entropy collapse**（输出多样性下降）、训练不稳定等问题。

这些问题导致 online RL 资源消耗大、效率低，难以在资源受限场景下应用。

---

### 🚀 提出的新方法与创新思路
本文系统性地探索了将 **offline RL** 应用于代码生成 LLM 的后训练中，提出一种高效的替代方案：

- **利用已有代码数据集（如 CodeNet）作为离线经验池**，直接从中采样 (prompt, code, reward) 三元组进行训练；
- 不再依赖实时采样和代码执行验证，完全摆脱 online interaction；
- 借鉴并适配现有的 online policy gradient 方法（如 RLOO、GRPO）到 offline 场景；
- 引入基于 **GRPO-style advantage** 和 **exponential weighting** 的优势函数设计，提升训练稳定性与效果。

> 🔍 创新洞察：尽管 RLOO 等算法本为 online 设计，但在 mild off-policy 或 fully offline 设置下仍可有效工作——这挑战了“必须 online 收集数据”的传统假设。

---

### ⚖️ 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **效率更高** | 免去重复 inference 与代码执行，大幅降低计算开销和延迟；可在单张 A100 上完成训练（<30 小时）。 |
| **多样性更好** | 利用预存数据中的丰富解法变体，缓解 entropy collapse，保留更多潜在正确路径。 |
| **训练更稳定** | 数据天然包含正负样本混合，避免 online 中“全正样本”导致的低方差梯度问题。 |
| **适用于小模型** | 对参数量较小的模型（如 0.5B）提升明显，弥补其初始能力不足。 |

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集
- **主训练数据集**：**CodeNet**
  - 包含约 4,000 个编程题，1400 万份解决方案，覆盖 50+ 编程语言；
  - 本文仅使用 **Python 解答**；
  - 反馈信号来自元数据：**functional correctness**（功能正确性）和 **syntactic correctness**（语法正确性）；
  - 为缓解数据不平衡，每道题最多保留 50 个正确解。

### 🧪 实验设置
- **基础模型**：
  - `Qwen2.5-Coder`，两个规模：**0.5B** 和 **7B** 参数；
  - 0.5B：全参数微调；7B：采用 **LoRA** 微调。
- **训练方式**：
  - 采用 **offline RL** 框架，不进行在线采样；
  - 使用 **RLOO (REINFORCE Leave-One-Out)** 目标函数；
  - 探索不同 advantage 计算方式：
    - 标准 RLOO advantage
    - **GRPO-style advantage**: $ A = \frac{R - \text{mean}(R)}{\text{std}(R)} $
    - **Exponential advantage**: $ A = \exp(\beta \cdot A_{\text{GRPO}}) $
- **Batching 策略**：
  - Group size = 4，每组强制包含至少一个正确 + 一个错误解；
  - Batch size = 8，最大生成长度 2048 tokens。

### 📊 评估指标
- **MBPP**（简单任务）：报告 **pass@1**
- **APPS+**（多难度任务）：按难度分层报告 **pass@1** 和 **pass@10**
  - 难度等级：Introductory, Interview, Competitive

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **BASE** | 未经微调的原始模型 |
| **SFT**（Supervised Fine-Tuning） | 仅用正确代码做监督学习 |
| **RLOO** | 标准 leave-one-out RL 训练 |
| **RLOO + GRPO Advantage** | 使用标准化优势函数 |
| **RLOO + Exp Advantage** | 使用指数加权优势 |

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据汇总（来自 Table 1–3）

#### ✅ 在 APPS+ 上的表现（7B 模型）
| Model | Interview (pass@1) | Competition (pass@1) | Interview (pass@10) | Competition (pass@10) |
|-------|--------------------|------------------------|------------------------|-------------------------|
| BASE | 7.01 | 1.42 | 25.70 | 8.22 |
| SFT | 11.30 | **1.38↓** | 30.30 | **6.29↓** |
| RLOO + GRPO | **12.14↑** | **2.67↑↑** | 30.20 | **11.54↑↑** |

> 💡 **关键发现**：SFT 在困难题目上反而退化，而 offline RL 显著提升，尤其在 competition 级别任务中 pass@1 提升 **88%**（1.42 → 2.67）！

#### ✅ 在 APPS+ 上的表现（0.5B 模型）
| Model | Introductory (pass@1) | Interview (pass@1) |
|-------|------------------------|------------------------|
| BASE | 0.03 | 0.06 |
| SFT | 1.94 | 1.87 |
| RLOO + Exp Advantage | **1.67** | **2.15↑** |

> ⚠️ 注意：标准 RLOO 效果差，但 **exponential advantage** 显著改善小模型表现，表明 reward scaling 对小模型至关重要。

#### ✅ 在 MBPP 上的表现
| Model | 0.5B (pass@1) | 7B (pass@1) |
|-------|---------------|-------------|
| BASE | 36.2 | 64.4 |
| SFT | 35.0 | 63.2 |
| RLOO + GRPO | **39.4↑** | 64.0 |

> 🔎 观察：对于已较强的任务（如 7B 在 MBPP），offline RL 几乎无增益，甚至轻微下降。

---

### 🔬 消融实验结果
| 实验配置 | 发现 |
|--------|------|
| **Group size = 4 vs 8** | 4 更优，结合“至少一正一负”约束可显著降低 advantage variance |
| **Advantage 形式比较** | GRPO-style > RLOO standard；Exp-weighting 对小模型更有效 |
| **reward 设计影响** | 小模型可能需要非线性 reward 映射（如 exp），否则难以收敛 |

---

## 4. **关键结论和发现**

### ✅ 主要结论
1. **Offline RL 是有效的代码生成后训练策略**：
   - 在合理 reward 设计下，能显著提升模型性能，尤其是在 **高难度编程任务** 上。
2. **特别有利于小模型和难问题**：
   - 小模型（0.5B）通过 offline RL 可接近甚至超越大模型趋势；
   - 在 competition-level 任务中，RLOO+GRPO 实现突破性提升。
3. **保持输出多样性**：
   - pass@10 提升幅度常大于 pass@1，说明模型生成了更多不同的可行解，未陷入单一模式。
4. **无需额外执行即可获得反馈**：
   - 利用历史数据中的 reward 信号（如测试通过与否），实现高效低成本训练。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **对强基线模型增益有限** | 当 base model 已表现良好（如 7B 在 MBPP），offline RL 几乎无效 |
| **依赖高质量离线数据分布** | 若数据中缺乏足够挑战性或多样性样本，效果受限 |
| **未优化超参空间** | 学习率、batch size、reward scaling 等未系统搜索 |
| **纯 offline 缺乏探索** | 无法发现数据集中不存在的新颖解法，需结合 online 微调 |

---

### 🔮 未来工作方向
1. **混合训练范式**：
   - 结合 **offline 初始化 + 少量 online interaction**，兼顾效率与探索能力。
2. **引入 value-based 方法**：
   - 如 **Q-learning** 或 **implicit Q-learning**，构建更稳健的 offline RL 算法。
3. **多维度 reward 设计**：
   - 除 functional correctness 外，加入 **runtime efficiency**, **memory usage**, **code readability**, **security checks** 等信号。
4. **改进数据平衡策略**：
   - 使用重要性采样、动态 reweighting 或合成负样本缓解数据偏态。
5. **理论分析 offline bias 与 generalization**：
   - 探索为何 online 算法能在 offline 下工作，建立更坚实的理论基础。

---

## ✅ 总结一句话
> 本文证明了 **offline RL** 是一种高效且强大的 LLM 代码生成后训练方法，尤其在资源受限、小模型、难题求解等场景下具有巨大潜力，为未来低成本高质量代码模型训练提供了新路径。

</details>

---

### 16. [DenoiseRL: Bootstrapping Reasoning Models to Recover from Noisy Prefixes](https://arxiv.org/abs/2605.28421)

**Authors**: Caijun Xu, Changyi Xiao, Zhongyuan Peng, Yixin Cao  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.28421v1  

#### Abstract
Reinforcement learning has become a central paradigm for advancing reasoning in large language models, yet most existing methods still depend on stronger teacher models or heavily curated difficult datasets, limiting scalable capability improvement. In this paper, we introduce DenoiseRL, a reinforce...

---

### 17. [Adaptive Multimodal Agents-Based Framework for Automatic Workflow Execution](https://arxiv.org/abs/2605.28607)

**Authors**: Susanna Cifani, Mario Luca Bernardi, Marta Cimitile  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.28607v1  

#### Abstract
Modern information systems require autonomous agents capable of navigating complex workflows, yet current methodologies often struggle with the transition from structured metadata parsing to general environmental perception. While the integration of MLLMs has enabled agents to interact directly with...

---

### 18. [Rethinking Visual Neglect: Steering via Context-Preference for MLLM Hallucination Mitigation](https://arxiv.org/abs/2605.27993)

**Authors**: Jingwen Wu, Xijun Zhang, Ge Song  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.27993v1  

#### Abstract
Object hallucination remains a primary obstacle to the reliable deployment of Multimodal Large Language Models (MLLMs). Current inference-time mitigation methods mainly assume hallucinations stem from visual neglect, steering models to enhance visual reliance. In contrast, our systematic interventio...

---

### 19. [The Missing Piece in Pre-trained Model Evaluation: Reward-Guided Decoding Unlocks Task-Oriented Behavior Without Parameter Updates](https://arxiv.org/abs/2605.28020)

**Authors**: Shaobo Wang, Guo Chen, Ziyue Wang, Zhengyang Tang, Qingyang Liu, Xingzhang Ren, Dayiheng Liu, Linfeng Zhang  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.28020v1  

#### Abstract
With the rapid progress of large language models (LLMs), reliably evaluating the capabilities of pre-trained LLMs has become increasingly important. The challenge is that base pre-trained models are optimized for next-token prediction and often fail to follow instructions or produce well-formed answ...

---

### 20. [CIRF: Tokenizing Chain-of-Thoughts into Reusable Functional Units for Efficient Latent Reasoning in Large Language Models](https://arxiv.org/abs/2605.28292)

**Authors**: Yukyung Lee, Yumeng Shen, Jinhyeong Park, Hyein Yang, Jun-Hyung Park  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.28292v1  

#### Abstract
Implicit Chain-of-Thought (CoT) reduces the inference cost of large language models by internalizing the explicit rationales. However, existing approaches typically lack alignment with explicit rationales and adaptivity to example complexity. In this work, we propose CIRF (\textit{\underline{C}hain-...

---

### 21. [GenSBI: Generative Methods for Simulation-Based Inference in JAX](https://arxiv.org/abs/2605.27499)

**Authors**: Aurelio Amerio  
**Category**: cs.LG  
**Published**: 2026-05-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.27499v1  

#### Abstract
Flow and diffusion generative models have established themselves as widely adopted density estimators for simulation-based inference (SBI), extending naturally from neural posterior estimation to likelihood and joint density estimation. Their principled optimization objectives and freedom from archi...

---

### 22. [Faster Thermal Profiling of a Lunar Rover with Machine Learning Adapted Finite Difference Model](https://arxiv.org/abs/2605.27651)

**Authors**: Samuel Weber, Zaki Hasnain, Souma Chowdhury  
**Category**: cs.LG  
**Published**: 2026-05-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.27651v1  

#### Abstract
Autonomous space systems operating in extreme thermal environments require accurate and efficient thermal modeling to support both pre-mission system design and onboard autonomy. For lunar rovers, large temperature gradients, radiative heat transfer, and variable surface conditions make reliable the...

---

### 23. [BPPO: Binary Prefix Policy Optimization for Efficient GRPO-Style Reasoning RL with Concise Responses](https://arxiv.org/abs/2605.28028)

**Authors**: Qingfei Zhao, Huan Song, Shuyu Tian, Jiawei Shao, Xuelong Li  
**Category**: cs.LG  
**Published**: 2026-05-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.28028v1  

#### Abstract
Group Relative Policy Optimization (GRPO) is widely used for training reasoning models, but updating all sampled completions in each group incurs substantial cost and can reinforce verbose reasoning trajectories. In this paper, we study whether all completions provide equally useful update signals i...

---

### 24. [PEAM: Parametric Embodied Agent Memory through Contrastive Internalization of Experience in Minecraft](https://arxiv.org/abs/2605.27762)

**Authors**: Yuchen Guo, Junli Gong, Hongmin Cai, Yiu-ming Cheung, Weifeng Su  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.27762v1  

#### Abstract
We present PEAM, a Parametric Embodied Agent Memory framework in Minecraft that transforms agent memory from inference-time retrieval into parameter-resident skills internalized through experience. PEAM pairs a slow deliberative LLM for open-ended reasoning with a fast parametric module for reflexiv...

---

### 25. [Mechanistically Interpreting the Role of Sample Difficulty in RLVR for LLMs](https://arxiv.org/abs/2605.28388)

**Authors**: Yue Cheng, Jiajun Zhang, Xiaohui Gao, Weiwei Xing, Zheng Wang, Zhanxing Zhu  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.28388v1  

#### Abstract
Reinforcement Learning with Verifiable Reward (RLVR) is empirically shown to notably enhance the reasoning performance of large language models (LLMs), particularly in mathematics and programming. However, the mechanistic role of Sample Difficulty in RLVR remains poorly understood. In this paper, we...

---

### 26. [Diffusion Large Language Models for Visual Speech Recognition](https://arxiv.org/abs/2605.28456)

**Authors**: Jeong Hun Yeo, Chae Won Kim, Hyeongseop Rha, Yong Man Ro  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.28456v1  

#### Abstract
Existing Visual Speech Recognition (VSR) systems commonly rely on left-to-right autoregressive decoding, which can force premature decisions on visually ambiguous tokens before sufficient context is available. We propose DLLM-VSR, to the best of our knowledge, the first Diffusion Large Language Mode...

---

### 27. [Let Relations Speak: An End-to-End LLM-GNN Soft Prompt Framework for Fraud Detection](https://arxiv.org/abs/2605.28524)

**Authors**: Zhixing Zuo, Huilin He, Jiasheng Wu, Dawei Cheng  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.28524v1  

#### Abstract
In recent years, Large Language Models (LLMs) have shown great capability in processing graph tasks such as fraud detection. However, most existing methods rely heavily on rich text attributes, which poses difficulties for this domain due to the lack of textual data. Although some pioneering methods...

---

### 28. [When Confidence Misleads: Suffix Anchoring and Anchor-Proximity Confidence Modulation for Diffusion Language Models](https://arxiv.org/abs/2605.28181)

**Authors**: Jungwon Park, Jimyeong Kim, Jungmin Ko, Nojun Kwak, Wonjong Rhee  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.28181v1  

#### Abstract
Diffusion language models decode text by iteratively denoising masked token sequences, making the choice of which positions to decode a central inference-time decision. Most training-free decoding strategies use model confidence for position selection, assuming that high-confidence positions are rea...

---

### 29. [Analyzing Quality-Latency-Resource Trade-offs in a Technical Documentation RAG Assistant Using LoRA Adaptation](https://arxiv.org/abs/2605.28222)

**Authors**: Evgenii Palnikov, Elizaveta Gavrilova  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.28222v1  

#### Abstract
We study quality-latency-resource trade-offs in a documentation-grounded retrieval-augmented generation (RAG) system that uses Low-Rank Adaptation (LoRA) of the generator. We build a manually verified benchmark of 5,144 question-answer pairs over the official Kubernetes documentation and combine it ...

---

### 30. [Dimensionality Reduction for Robust Federated Learning: A Theoretical Analysis and Convergence Guarantee](https://arxiv.org/abs/2605.28335)

**Authors**: Shiyuan Zuo, Jiashuo Li, Rongfei Fan, Han Hu, Jie Xu  
**Category**: cs.LG  
**Published**: 2026-05-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.28335v1  

#### Abstract
Federated Learning (FL) enables multiple clients to collaboratively train models without sharing raw data, but it is highly vulnerable to Byzantine attacks. Existing robust approaches can neutralize these threats but incur substantial computational overhead during high-dimensional gradient aggregati...

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

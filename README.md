# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-28 17:45:56 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [VPP: Virtual Pipeline Parallelism for Efficient Chunked Prefill in Long-Context LLM Inference](https://arxiv.org/abs/2608.26523)

**Authors**: Yan Shi, Xiaochao Wang, Jingchun Gao, Jintao Luo, Xinyi Zhou, Feng Liu, Kui Luo, Xushi Li, Xinjie Guo, Liangjun Feng  
**Category**: cs.DC  
**Published**: 2026-08-28  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2608.26523v1  

#### Abstract
Chunked prefill pipeline parallelism (CPP) is a key technique for LLM inference. However, equal-size chunks exhibit imbalanced latency, as later chunks attend longer prefix KV caches and incur higher attention costs, leading to pipeline bubbles. Existing approaches mitigate this imbalance through dy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：VPP: Virtual Pipeline Parallelism for Efficient Chunked Prefill in Long-Context LLM Inference**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
在长上下文 LLM 推理中，**Chunked Prefill Pipeline Parallelism (CPP)** 是一种关键技术，用于将长输入提示划分为小块以提升调度灵活性并缓解头阻塞（HOL blocking）。然而，传统等大小分块会导致严重的**计算负载不均衡**：随着前缀 KV Cache 增长，后续 chunk 的 attention 开销显著增加，导致 pipeline 中出现大量“气泡”（pipeline bubbles），降低硬件利用率。

现有方案如 **Dynamic CPP (DCPP)** 通过动态调整 chunk 大小来平衡执行时间，但引入了更高的调度开销和更细粒度的执行单元，反而在长序列上造成额外的通信与算子启动开销。

---

### 🚀 提出的新方法：Virtual Pipeline Parallelism (VPP)

VPP 的核心思想是：  
> **不改变 chunk 大小，而是重构 pipeline 的布局，利用虚拟阶段（virtual stages）和 V-shaped 遍历策略来吸收固有的延迟增长。**

#### 主要创新点：
1. **V-shaped 虚拟阶段遍历（V-shaped Stage Traversal）**  
   - 将模型划分为多个虚拟 stage，并采用“前向-折返”的方式在物理设备间流动（形成“V”形路径）。
   - 利用相邻 chunk 的轻量首尾阶段去填充当前 chunk 重型中间阶段造成的空窗期，从而自然地掩盖因 KV Cache 增长带来的线性延迟上升。

2. **异步通信重排序（Asynchronous Communication Reordering）**  
   - 重新安排跨 rank 的发送/接收操作顺序，避免双向通信阻塞，实现 computation-communication 有效重叠。

3. **流水线打包（Pipelined Packing）**  
   - 在一个请求的 drain 阶段插入下一个请求的初始 chunk，压缩跨请求的 pipeline drain bubbles。

---

### 🔍 相比现有方法的优势

| 维度 | DCPP | VPP |
|------|------|-----|
| Chunk Size | 动态调整 → 更多调度开销 | 固定大小 → 更高效 kernel 执行 |
| 负载均衡机制 | 依赖运行时预测与调参 | 利用可预测的线性延迟增长进行静态调度 |
| 通信优化 | 同步通信易产生 stall | 异步重排序提升通信隐藏能力 |
| 跨请求效率 | 存在明显 drain bubbles | 通过 pipelined packing 显著压缩 idle 时间 |

> ✅ **VPP 在保持短序列性能的同时，在长序列和混合负载下实现了更高吞吐量和更低 TTFT，且无需复杂的在线成本估计与参数调优。**

---

## 2. **核心实验方法和设置**

### 📊 数据集与工作负载
未使用传统 NLP 数据集，而是构建了三类典型推理 workload：
- **短序列（Short Sequences）**：4K、8K、16K tokens，100 并发请求
- **长序列（Long Sequences）**：单个请求长度从 64K 到 1M tokens
- **混合长度（Mixed-Length Workload）**：基于 GSM8K 合成，长度范围 7–122,710 tokens（均值 ~22.5K，p90 ~64K），并发 16

所有请求仅生成 1 个 output token，以隔离 prefill 性能。

---

### ⚙️ 实验设置

- **硬件平台**：华为 Atlas900 A3 SuperPoD，配备 **16 × Ascend 910C NPU**（每颗 64GB HBM）
- **通信后端**：HCCL（HUAWEI Collective Communication Library）
- **框架基础**：PyTorch v2.10 + vLLM-Ascend v0.23.0
- **并行配置**：TP8 + 2-stage Pipeline Parallelism（即 `TP8+CPP2`, `TP8+DCPP2`, `TP8+VPP2`）

---

### 🎯 评估指标
- **Throughput（吞吐量）**：单位为 token/s
- **Time-to-First-Token (TTFT)**：反映 prefill 延迟
- **Pipeline Bubble Ratio**：衡量 pipeline 利用率
- **Profiling Breakdown**：分解 computation、bubble、exposed communication 占比

---

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **CPP** | 固定大小 chunk 的标准 chunked prefill pipeline parallelism |
| **DCPP** | 动态调整 chunk 大小以平衡执行时间（vLLM-Ascend 推荐配置） |
| **VPP** | 本文提出的方法（含 V-shaped traversal + async comm + pipelined packing） |

---

## 3. **主要实验结果和性能指标**

### 📈 整体性能提升（vs. DCPP）

| 场景 | Throughput 提升 | TTFT 改善 |
|------|------------------|-----------|
| **长序列（Long Sequences）** | 最高 **+13.1%**（DeepSeek-V3.1 @512K） | 显著下降 |
| **混合负载（Mixed Workloads）** | 最高 **+6.7%**（DeepSeek-V3.1） | 同步改善 |
| **短序列（Short Sequences）** | 基本持平或轻微提升（最高 +10.8%） | 无退化 |

> 💡 特别是在 **512K-token DeepSeek-V3.1 prefill 任务中**：
> - **pipeline bubble ratio 从 6.4% 降至 0.1%**
> - 实现了 **98.0% 的 bubble reduction**
> - **TTFT 减少 40.36 秒（相对 DCPP 下降 12.53%）**

---

### 🔍 性能分解分析（DeepSeek-V3.1 @512K）

| 指标 | DCPP | VPP | 变化（Δ） | 改进比例 |
|------|------|-----|----------|---------|
| **Computation Time** | 279.07s | 250.84s | ↓28.23s | -10.11% |
| **Bubble Time** | 20.47s | 0.39s | ↓20.07s | **-98.04%** |
| **Exposed Communication** | 22.68s | 30.63s | ↑7.95s | +35.05%（trade-off） |
| **Total TTFT** | 322.22s | 281.86s | ↓40.36s | **-12.53%** |

> ⚠️ 注意：虽然暴露通信略有增加，但由于 **communication-computation overlap 提升至 51.75%**（原仅 0.05%），实际影响可控。

---

### 🔬 消融实验结果（Ablation Study）

在 DeepSeek-V3.1 @256K 上测试三种 VPP 变体：

| 方案 | End-to-End Latency Reduction | Bubble Ratio | 关键改进 |
|------|-------------------------------|--------------|--------|
| **Vanilla VPP** | —— | 11.0% | 引入 V-shaped 调度 |
| **VPP-Async** | ↓3.13% | 8.3% | 异步通信重排序，提升通信隐藏 |
| **VPP-Async + Pipelined Packing** | ↓6.89% | **2.4%** | 压缩跨请求 drain bubbles |

> ✅ 两个优化模块协同作用，分别解决 **intra-request communication stall** 和 **inter-request idle time**。

---

### 🧪 Chunk Size 敏感性分析
- **最佳 chunk size 多为 24K**，即使不能整除总长度。
- 原因：最后一个较小的 chunk 正好落在 pipeline drain 阶段，其计算与通信可被前面 chunk 隐藏，减少尾部气泡。
- 例外：极短序列（如 64K）中，越小 chunk 越好，因 pipeline 无法进入稳态。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **DCPP 的负载均衡收益被调度开销抵消**  
   - 在长序列中，动态分块导致过多 kernel launch 与通信事件，碎片化严重，最终性能反不如固定分块。

2. **chunk latency 增长具有近似线性规律**  
   - 得益于 causal attention 对 KV Cache 的依赖，后期 chunk 的延迟增长可建模为 $ T_k \propto (k+1)t $，为 VPP 的静态调度提供了理论依据。

3. **VPP 成功将不平衡转化为并行机会**  
   - 不试图消除差异，而是通过 V-shaped traversal 让“快慢阶段互补”，实现自然负载匹配。

4. **虚拟 stage 设计优于动态 resize**  
   - 更稳定、无需 runtime profiling，更适合生产部署。

---

### ⚠️ 局限性
1. **对稀疏 attention（如 DSA）支持有限**  
   - 如 GLM-5.2 使用 DSA 结构，attention 成本随长度增长变缓，破坏了线性假设，导致 VPP 效果减弱。
   
2. **依赖 MoE 专家并行（EP）配置**  
   - 若 EP 关闭或 attention 占比下降，V-shaped 匹配可能失效。

3. **目前仅适配两物理 stage 场景**  
   - 扩展到更深 pipeline 或异构设备仍需进一步研究。

---

### 🔮 未来工作方向
- 扩展 VPP 至 **更深的 pipeline 架构**（>2 stages）
- 支持 **sparse attention 模型** 的自适应虚拟 stage 调度
- 探索 **异构请求混合调度** 下的全局最优 packing 策略
- 结合 **context parallelism** 进一步降低内存压力

---

## ✅ 总结

VPP 提出了一种全新的视角来应对长上下文 LLM 推理中的 pipeline 不平衡问题：  
> **不是通过“削峰填谷”式的动态 chunk 调整，而是通过“错峰调度”的虚拟 stage 布局，把不可避免的增长变成可利用的资源。**

它在 **吞吐量、TTFT、稳定性** 上全面超越 DCPP，尤其在百万 token 级别任务中展现出巨大潜力，是面向下一代超长上下文 LLM serving 系统的重要基础设施设计。

</details>

---

### 2. [Dependency-Aware Revocable Decoding for Efficient Diffusion Large Language Model Inference](https://arxiv.org/abs/2608.26574)

**Authors**: Wooje Park, Insu Lee, Minyoung Noh, Jaeyun Jang, Sungmin Lee, Kyuhong Shim, Byonghyo Shim  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2608.26574v1  

#### Abstract
Diffusion large language models (dLLMs) offer a promising alternative to autoregressive generation by decoding multiple tokens in parallel through iterative denoising. However, increasing decoding parallelism often degrades generation quality, as early errors can contaminate later contexts. Revocabl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Dependency-Aware Revocable Decoding for Efficient Diffusion Large Language Model Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **revocable decoding** 方法（如 Saber 和 WINO）虽然通过重新验证并重掩码不可靠 token 来提升 dLLM 的生成质量，但存在一个被忽视的关键缺陷：  
> **在验证过程中，已解码但错误的 token 可能污染上下文（verification context），导致误判或不必要的重掩码**。

例如，在提示 “The city of _ _ is on the Southern California coast” 中，并行预测可能产生 “Los Diego”，两个词单独看都合理，但组合无效。当使用彼此作为上下文进行验证时，模型可能无法纠正这一组合错误，甚至将两者都判定为不可靠而过度重掩码。

### 提出了什么新方法或新思路
作者提出 **Dependency-Aware Revocable Decoding (DARD)**，一种无需训练的三状态可撤销解码框架，其核心思想是：

- 引入三种 token 状态：
  - **Masked (M)**：未解码
  - **Candidate (C)**：已解码但置信度中等，需进一步验证
  - **Unmasked (U)**：高置信度，视为可靠上下文
- 在验证阶段采用 **选择性上下文机制**：
  - 验证 `C` 状态 token 时，仅允许其关注更高置信度的 `C` 或所有 `U` token，形成 **confidence-ordered attention**。
  - 这种设计模拟了按置信度排序的多步自回归解码过程，避免低质量 token 干扰验证。
- 对 `M` token 的预测采用 **自适应 logit mixing**：
  - 结合来自原始序列（含 `C` 上下文）和影子序列（不含 `C` 上下文）的预测。
  - 根据附近 `C` token 的验证结果（晋升为 `U` 或降级为 `M`）动态调整两者的权重 $ w $，从而控制 `C` token 的上下文影响力。

### 相比现有方法的优势
- **更鲁棒的验证机制**：通过隔离不可靠 token 的上下文影响，显著减少因上下文污染导致的验证失败。
- **更高的速度-质量权衡（speed-quality Pareto frontier）**：相比 Saber 和 WINO，在更少的 decoding steps 下达到更高性能，或在相同步数下性能更优。
- **无需额外训练**：纯推理期优化，兼容现有 dLLM。
- **通用性强**：在多种文本与多模态任务上均表现稳定提升。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
共评估 **12 个基准**，涵盖语言与多模态任务：

#### 文本任务（6个）：
- **GSM8K**：数学应用题
- **MATH500**：高等数学问题
- **MBPP**：Python 编程任务
- **Countdown**：算术推理
- **Sudoku**：逻辑填空（4-shot）
- **ARC-Challenge**：科学常识推理

#### 多模态任务（6个）：
- **Flickr30K**：图像描述生成（使用 CIDEr 指标）
- **AI2D**：图表理解
- **ScienceQA**：带图科学问答
- **MathVista / MATH-Vision**：视觉数学推理
- **MMMU**：跨学科多模态理解

### 实验设置和评估指标

| 项目 | 设置说明 |
|------|----------|
| **模型** | 使用三个开源 dLLM：<br>- LLaDA-8B-Instruct<br>- LLaDA-1.5<br>- MMaDA-8B-MixCoT |
| **评估方式** | 零样本（zero-shot），除 Sudoku 为 4-shot |
| **性能指标** | - 文本任务：Accuracy<br>- Flickr30K：CIDEr Score<br>- MBPP：pass@1 functional correctness |
| **效率指标** | - 平均 decoding steps<br>- 吞吐量（Tokens Per Second, TPS） |
| **实现细节** | 使用 block decoding，generation length=256，block length=128；温度设为 0 |

### 基线方法对比
- **Standard dLLM decoding**：固定解码，无重掩码
- **Saber**：基于置信度下降检测可疑 token
- **WINO**：双路径验证，独立判断每个 token 是否保留

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **Flickr30K** 上，DARD 相比 **Saber** 实现：
  - **2.71× 的加速**
  - **CIDEr 分数提升 4.35 点**
- 在 **MATH500** 上，DARD 达到最高准确率 **34.6%**，仅需 **59.3 步**，而 Saber 需要 81 步才达到 33.2%
- 在 **GSM8K** 上，DARD 实现 **5.3× 加速**，同时准确率提升 **+5.5%**

### 与基线方法的对比结果
| 方法 | 相对 Saber 准确率增益 | 相对 Saber 步数减少 |
|------|------------------------|-----------------------|
| **DARD (ours)** | **+1.4%** | **-21.7 步 (-26.8%)** |
| WINO | +0.2% | -22.0 步 |
| Saber | 基准 | 基准 |

> DARD 在多个任务上不仅超越 Pareto 前沿，且在峰值性能配置下也优于标准 one-token-per-step 解码（见 Table 7 & 8）。

### 消融实验结果

#### ✅ 注意力掩码设计（Table 1）
| 方法 | Acc. (%) | Steps |
|------|---------|-------|
| Saber | 33.2 | 81.0 |
| DARD (bidir) | 31.4 | 58.5 |
| DARD (12r) | 32.0 | 59.3 |
| **DARD (confidence, ours)** | **34.6** | **59.3** |

> 使用双向注意力会导致精度大幅下降，证明 **有序注意力至关重要**；基于置信度排序效果最优。

#### ✅ 自适应 Logit Mixing（Table 2）
| 方法 | Acc. (%) | Steps |
|------|---------|-------|
| Saber | 33.2 | 81.0 |
| DARD (w=0.0) | 29.2 | 82.0 |
| DARD (w=0.5) | 31.1 | 63.6 |
| DARD (w=1.0) | 33.4 | 57.1 |
| **DARD (adaptive, ours)** | **34.6** | **59.3** |

> 固定权重无法兼顾效率与准确性，**自适应混合策略实现了最佳平衡**。

#### ✅ 配置鲁棒性（Table 3）
不同 generation/block length 下 DARD 性能稳定，表明其对超参不敏感。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **上下文污染是 revocable decoding 的关键瓶颈**：现有方法忽略了解码 token 对验证过程的反向干扰。
2. **分离 token 状态 + 控制依赖关系可有效缓解该问题**：引入 `C` 状态并通过 confidence-ordered attention 构建干净验证上下文，显著提升纠错能力。
3. **DARD 显著改善了 dLLM 的 speed-quality trade-off**：在 12 个 benchmark 上 consistently 超越现有方法，实现更快、更强的生成。
4. **该方法具备良好的泛化性和实用性**：适用于多种 dLLM 架构和任务类型，无需微调即可部署。

### 方法的局限性
- **绝对性能增益有限**：目标不是改变模型分布，而是提高一致性，因此提升幅度受限于原模型能力。
- **计算开销略有增加**：
  - 单步延迟略高于 Saber（约 +40ms），但由于总步数大幅减少，端到端时间仍更短。
  - 峰值显存增加 <6.3%，主要来自影子序列。
- **依赖置信度作为可靠性代理**：若模型校准不佳，confidence 可能不能准确反映 token 质量。

### 未来工作方向
- 探索更精细的 token 可靠性估计方法（如集成多个 internal signals）。
- 将 DARD 扩展至视频生成等长序列扩散模型。
- 与 learnable decoding policy 方法结合，进一步优化 token 选择策略。
- 研究如何在低资源设备上高效实现 shadow sequence 机制。

---

> ✅ **总结一句话**：  
> DARD 通过 **三状态机制 + 依赖感知验证**，解决了 revocable decoding 中的上下文污染问题，在无需训练的前提下，显著提升了 dLLM 的推理效率与生成质量，推动了 diffusion-based generation 的实用化进程。

</details>

---

### 3. [Decoupled I/O-Dominant Pipelines for Large-Scale Whole-Slide Image Embedding Extraction](https://arxiv.org/abs/2608.27278)

**Authors**: Mayanka Chandrashekar, Xi Zhang, Ethan Seefried, Tirthankar Ghosal, John Gounley, Heidi Hanson  
**Category**: cs.DC  
**Published**: 2026-08-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2608.27278v1  

#### Abstract
Whole-slide images (WSIs) are central to computational pathology but are prohibitively large, making patch-based processing the practical unit for foundation model inference. At scale, however, generating and handling massive numbers of patches on quickly introduces significant I/O and orchestration...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Decoupled I/O-Dominant Pipelines for Large-Scale Whole-Slide Image Embedding Extraction

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

- **大规模 Whole-Slide Images (WSIs)** 在计算病理学中至关重要，但由于其尺寸巨大（通常为数十亿像素），直接处理不可行，必须采用 **patch-based 处理**。
- 在大规模场景下，生成和管理海量 patches 会引入严重的 **I/O 和数据编排开销**，导致端到端性能被 **数据移动而非计算能力** 所主导。
- 现有 WSI 处理流程通常是 **紧耦合（tightly coupled）** 的，将 patch 生成、模型推理和输出处理交织在一起，造成资源争用、共享存储压力大、扩展性差。
- 此外，大多数系统将 embedding 视为临时文件，缺乏对 **embedding 的持久化存储与重用机制**，导致重复计算成本高昂。

### 🚀 提出的新方法与新思路

作者提出了一种 **解耦的、I/O-aware 的三阶段流水线架构**，用于大规模 WSI embedding 提取：

1. **Stage 1: Patch Generation & Staging**
   - 使用 MPI 进行空间分解，独立生成 patches 并暂存。
   - 引入 **I/O-aware 数据预热策略**，优化小文件写入行为。

2. **Stage 2: Embarrassingly Parallel Embedding Inference**
   - 各 GPU 独立加载 patch 数据并执行前向推理，无集体通信（no all-reduce）。
   - 利用 SPMD（Single Program Multiple Data）模式实现高并发推理。

3. **Stage 3: Sharded Vector Database Ingestion**
   - 将 embedding 及其元数据（patient, slide, spatial coordinates）直接注入分布式 **vector database（如 Milvus）**。
   - 支持后续任务（检索、聚类、few-shot learning）直接查询，避免重复访问原始 WSI。

> 🔑 **核心思想**：将 I/O 密集型、计算密集型和写密集型阶段完全解耦，使每个阶段可独立优化，并将 embedding 构建为 **可共享、可复用的数据资产（AI-Ready Data）**。

### ⚖️ 相比现有方法的优势

| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **架构设计** | 紧耦合流水线（tight coupling） | 完全解耦三阶段（fully decoupled） |
| **I/O 管理** | 边生成边处理，加剧存储争用 | 分阶段处理，支持异步 staging |
| **embedding 存储** | 临时文件，不可复用 | 持久化至 metadata-aware vector DB |
| **扩展性** | 受限于共享 I/O 路径 | 各阶段按需扩展，减少干扰 |
| **下游应用支持** | 需重新提取 embedding | 支持高效 filtering、retrieval、analytics |

---

## 2. 核心实验方法和设置

### 📦 数据集

- **CCDI MCI Dataset**（Childhood Cancer Data Initiative Molecular Characterization Initiative）
  - 来源：National Cancer Institute Imaging Data Commons (ICDC)
  - 包含：**4,185 张 H&E 染色 WSI**
  - 总 patch 数量：约 **4.14 亿个（256×256）**
  - 非空白 patch：约 **1.7 亿个**
  - 患者数：4,054 名
  - DICOM 文件数：19,603 个

> 数据集具有高度异构性，涵盖不同组织类型和分辨率，适合评估真实世界下的系统表现。

### 💻 实验平台

- **硬件系统**：Oak Ridge National Laboratory 的 **Frontier 超算系统**
  - 每节点配置：1 × 64-core AMD EPYC CPU + 8 × AMD Instinct MI250X GPUs
  - 网络互联：Slingshot 高速网络
- **存储系统**：**Orion 并行文件系统**（基于 Lustre）
  - 特点：高聚合带宽，但在高并发小文件读写时存在元数据瓶颈

### 🧪 评估设置与指标

#### 四类实验设置（Setups）：

| 设置 | 目标 |
|------|-----|
| **Setup A**: Patch Sweep | 测试固定 WSI 下随 patch 数量增加的性能变化，识别 I/O 主导拐点 |
| **Setup B**: Strong Scaling | 固定 workload，增加 GPU 数量，测试强扩展效率 |
| **Setup C**: Kernel-Level Profiling | 分析 GPU 内核执行效率，验证是否真正 compute-bound |
| **Setup D**: Production-Scale Run | 在全部 4,185 张 WSI 上运行，评估实际部署中的变异性与吞吐量 |

#### 关键评估指标：

| 阶段 | 指标 |
|------|------|
| **Patch Generation** | 文件数量、平均文件大小、写入吞吐量（MB/s）、runtime、speedup、efficiency |
| **Embedding Inference** | GPU 利用率、I/O 等待时间占比、throughput (patches/sec)、end-to-end runtime |
| **Vector DB Ingestion** | 插入速率（rows/s）、总耗时、内存占用、并行效率 |

#### 基线对比方法（隐式）

虽然未显式列出多个 baselines，但通过与以下典型做法对比体现优势：

- **传统紧耦合 pipeline**（如 OpenSlide-based workflows）
- **未持久化 embedding 的临时处理方式**
- **非分片 vector database 或本地存储方案**

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ Patch Generation 阶段（Stage 1）

- 使用 **224 vs 256 patch size** 对比：
  - 更大的 patch size 减少了文件总数（↓25%），显著提升写入吞吐量（↑~30–50%）
  - 示例：`ca6e4e7a` slide 使用 256 patch size 时达到 **224.27 MB/s**，而 224 仅为 200.67 MB/s
- **强扩展性测试（56 → 224 MPI ranks）**：
  - 最高速度提升仅 **1.77×**，效率下降至 **44.2%**
  - 表明性能受限于 **filesystem contention** 而非计算或通信

> ➤ 结论：**小文件元数据操作是主要瓶颈**

#### ✅ Embedding Inference 阶段（Stage 2）

| 模型 | GPU 数 | Throughput (patches/sec) | 效率（40 GPUs） |
|-------|--------|----------------------------|----------------|
| **HIPT** | 40 | 1,104.1 | 18.0% |
| **H-Optimus-0** | 40 | 855.8 | 19.3% |
| **Virchow2** | 40 | 826.9 | 29.6% |

- **GPU 利用率始终偏低**（普遍 < 40%），即使 workload 很大
- **I/O wait 占比随规模上升**（最高达 15.5%）
- **吞吐量在 patch 数超过 ~10K 后趋于饱和**（见 Fig 2）

> ➤ 结论：**并非 compute-bound，而是受制于 patch 数据供给延迟**

#### ✅ Kernel-Level 分析（Setup C）

- 所有模型内核均处于 **compute-efficient 区域**（接近 Roofline 的 GEMM/Conv boundary）
- 但单次 kernel 执行时间短（毫秒级），批次间出现明显空闲（stall）
- 表明 **GPU 等待数据加载，而非计算能力不足**

#### ✅ Vector Database Ingestion（Stage 3）

| 节点数 | 插入速率 (rows/s) | 总吞吐增速 | 效率峰值 |
|--------|------------------|------------|----------|
| 1 | ~22K | 1.00× | — |
| 2 | ~40K | 1.53× | 76.3% |
| **4** | **~86K** | **3.47×** | **86.7%** ✅ |
| 8 | ~188K | 6.15× | 76.9% |

- **4 节点时效率最高（86.7%）**，之后因读带宽和元数据竞争导致收益递减
- 内存稳定（每 rank ~0.6 GB），说明为 **streaming 设计成功**

> ➤ 结论：**仍为 I/O-bound，最佳运行点需权衡并发与存储压力**

#### ✅ 生产级运行（Setup D）

- 大型 WSI（>100K patches）能更好摊销固定开销，吞吐更高
- 存在显著 **straggler effect**：最快 worker 与最慢相差数百秒（如 Virchow2 达 369s 差距）
- CV（变异系数）低至 0.002～0.115，反映负载不均衡风险

---

## 4. 关键结论和发现

### 🔍 主要发现

1. **大规模 WSI embedding 提取本质上是 I/O- and orchestration-dominant workload**  
   > 而非传统认知中的 compute-bound 或 communication-bound。

2. **即使各阶段“完美并行”，整体性能仍由数据移动决定**  
   > 包括 patch 生成的小文件 I/O、embedding 推理的数据供给、vector DB 的写放大等。

3. **解耦设计显著改善资源利用与可扩展性**  
   > 允许各阶段独立调优，降低跨阶段干扰，提高系统鲁棒性。

4. **构建 metadata-aware vector database 是实现 embedding 重用的关键**  
   > 支持快速 filtering、retrieval、few-shot learning，尤其利于低资源环境。

5. **性能模型可表达为**：
   $$
   T(R) = \max\left(\frac{N}{R}, \frac{N \cdot B}{BW_{\text{storage}}}\right)
   $$
   > 当 compute term 缩小时，data transfer term 成为主导项。

---

### ⚠️ 方法的局限性

1. **依赖高性能并行文件系统（如 Orion/Lustre）**
   - 在普通 NAS 或云存储上可能难以复现相同性能
2. **未解决 patch 存储本身的长期成本问题**
   - 虽然 embedding 可重用，但原始 patch 文件仍需大量存储空间
3. **当前 vector DB 查询尚未集成到训练流程中**
   - 如 active learning、retrieval-augmented inference 等高级应用尚待开发
4. **缺乏动态负载均衡机制应对 straggler**
   - 当前为静态分配，未来可引入 work-stealing 或 adaptive scheduling

---

### 🔮 未来工作方向

1. **构建端到端的 embedding-as-a-service（EaaS）平台**
   - 提供统一 API 访问已计算 embedding，支持多团队协作
2. **融合 caching 与 tiered storage 策略**
   - 将 hot embedding 缓存在 NVMe 或 DRAM 中，cold 数据归档至对象存储
3. **支持增量更新与版本控制**
   - 新增 WSI 或模型升级后，支持增量 embedding 注入
4. **探索 embedding compression 与量化技术**
   - 减少 vector DB 存储与传输开销，同时保持语义质量
5. **将 vector DB 查询嵌入 downstream 任务**
   - 如 retrieval-based few-shot classification、attention grounding 等

---

## ✅ 总结

本论文揭示了一个重要洞见：**在大规模 WSI embedding 提取中，真正的瓶颈不是 GPU 算力，而是数据如何高效地“流动”**。通过提出一个 **三阶段解耦流水线 + metadata-aware vector database** 的新范式，作者不仅提升了系统吞吐与可扩展性，更将 embedding 从“一次性中间产物”转变为 **可持续使用的科学资产**，为未来的大规模医学 AI 基础设施建设提供了重要参考。

</details>

---

### 4. [Activation Outliers Matter: Robust Recovery for Quantized Multimodal LLMs](https://arxiv.org/abs/2608.26581)

**Authors**: Tanzila Rahman, Mehran Taghian Jazi, Yunke Peng, Zhuang Ma, Anandharaju Durai Raju, Yao Wang, Xing Huang, Hei Yi Mak, Shadan Golestan, Hoang Le, Yonghan Dong, Wei Guo, Yaoyuan Wang  
**Category**: cs.LG  
**Published**: 2026-08-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2608.26581v1  

#### Abstract
Low-bit quantization offers a promising avenue for reducing the computational and memory demands of Multimodal Large Language Models (MLLMs). Recent hardware support for low-precision formats, ranging from MXFP8 to ultra-low-bit formats such as MXFP4 and HiF4, has accelerated research into efficient...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Activation Outliers Matter: Robust Recovery for Quantized Multimodal LLMs**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
该论文系统研究了在**超低位宽（ultra-low-bit）量化**（如 MXFP4 和 HiF4）下，多模态大语言模型（Multimodal LLMs, MLLMs）训练与推理中的性能退化问题。  
作者发现：
- 当前低精度格式（如 4-bit）虽然能显著降低计算和内存开销，但在 MLLMs 上会导致严重性能下降。
- **激活值（activations）的量化误差是主要瓶颈**，尤其是由**激活异常值（activation outliers）**引起的动态范围失配问题。

传统方法（如 PTQ 或通用 QAT）难以应对 MLLMs 中跨模态、模块依赖性强且训练过程中动态变化的激活分布，导致信息丢失严重。

---

### 🚀 **提出了什么新方法或新思路**
提出 **Residual Fallback Quantization (RFQ)** ——一种轻量级、无需架构修改的激活感知量化策略。

#### 核心思想：
- 将激活分解为两个路径：
  1. **主路径**：标准的 ultra-low-bit（如 FP4）量化表示；
  2. **残差路径**：对量化误差 $ R = X - Q(X) $ 进行二次量化并补偿回输出。
- 不采用阈值选择机制，而是**显式建模整个激活块的量化残差**，并通过另一个 FP4 通路进行重建与累加。

> 公式表达：  
> $$
> Z_{pq} = \tilde{X}_{pr}\tilde{Y}_{rq} + \phi(p,r) \cdot \Delta\tilde{X}_{pr} \cdot \tilde{Y}_{rq}
> $$

其中 $\phi(p,r)$ 是硬件友好的指示器，用于识别含显著异常值的激活块。

---

### 🔍 **相比现有方法的优势**
| 对比维度 | 现有方法（如 SmoothQuant, OCC, Outlier Fallback） | 本文 RFQ 方法 |
|--------|---------------------------------------------|--------------|
| **适用场景** | 多针对文本 LLM 或静态 PTQ 设置 | 面向端到端 QAT，支持 MLLMs 动态训练过程 |
| **异常值处理方式** | 剪裁（clipping）、平滑变换、混合精度 fallback | 显式重构量化残差，保留全部信息流 |
| **硬件兼容性** | 可能引入高精度 fallback 路径，增加访存开销 | 所有操作均保持在统一 FP4 格式内，无额外数据类型 |
| **实现复杂度** | 通常需模块定制或敏感层分析 | 无需架构修改，仅在 GEMM 层插入残差通路 |
| **效率影响** | 有时引入显著延迟或带宽压力 | 几乎零计算开销，完全兼容 Tensor Core 加速 |

✅ **优势总结**：  
RFQ 在不牺牲 ultra-low-bit 计算效率的前提下，有效缓解了激活异常值带来的信息损失，实现了“稳健恢复”。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
| 模型 | 数据集 | 说明 |
|------|-------|------|
| **Wan2.2-5B** | OpenVid-1M 子集（约 26K 视频-文本对） | 用于视频生成任务的监督微调（SFT） |
| **Qwen3-VL-30B** | CC-3M（595K 图像-文本样本） | 用于多模态推理能力的 SFT |

---

### ⚙️ **实验设置**
- **量化格式对比**：
  - MXFP8（E4M3/E5M2）
  - MXFP4（E2M1）
  - HiF4（Hierarchical Float4）
  - W4A8（权重 4-bit，激活 8-bit）
- **混合精度策略**：
  - Embedding 层和 LM Head 保持 BF16
  - FFN、Attention 投影层等使用低比特格式
- **训练配置**：
  - 使用 AdamW 优化器
  - Wan2.2 学习率：$1\times10^{-5}$；Qwen3-VL：$1\times10^{-7}$
  - 总训练 token 数约为 5B
- **QAT 流程**：从公开预训练 checkpoint 初始化后进行低精度微调

---

### 📊 **评估指标**
| 模型 | 评估任务 | 指标 |
|------|---------|------|
| **Wan2.2** | 视频生成质量 | VBench 多项指标：<br>• Temporal Consistency<br>• Dynamic Degree<br>• Aesthetic Quality<br>• Subject Consistency |
| **Qwen3-VL** | 多模态推理能力 | 四个基准准确率：<br>• RealWorldQA<br>• MMStar<br>• MMBench-EN<br>• SimpleVQA |

---

### 🔁 **基线方法对比**
| 基线 | 类型 | 描述 |
|-----|------|------|
| **BF16 Baseline** | Full Precision | 完整精度模型，作为性能上限参考 |
| **MXFP8 / HiF4 / MXFP4** | Standard QAT | 直接应用低比特格式，无特殊修复机制 |
| **W4A8** | Mixed Precision | 权重压缩至 4-bit，激活保留 8-bit，检验激活主导作用 |

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据汇总**

#### 表 1：VBench 视频生成性能（Wan2.2）

| Method | Dynamic Degree | Aesthetic Quality | Subject Consistency |
|--------|----------------|--------------------|----------------------|
| BF16 (Baseline) | 45.00 | 59.43 | 95.74 |
| MXFP4 | 43.00 (-2.00) | 58.93 (-0.50) | 95.65 (-0.09) |
| **MXFP4 + RFQ** | **50.00 (+5.00)** | **59.26 (-0.16)** | **95.43 (-0.31)** |
| HiF4 | 53.00 (+8.00) | 59.44 (+0.01) | 95.23 (-0.51) |
| **HiF4 + RFQ** | **51.00 (+6.00)** | **59.54 (+0.19)** | **95.86 (+0.12)** |

> ✅ RFQ 显著提升动态细节表现，并恢复主体一致性。

---

#### 表 2：多模态推理准确率（Qwen3-VL）

| Method | RealWorldQA | MMStar | MMBench-EN | SimpleVQA |
|--------|-------------|--------|------------|-----------|
| BF16 | 72.68 | 70.80 | 90.77 | 16.83 |
| MXFP4 | 70.98 (-1.70) | 69.67 (-1.13) | 90.72 (-0.05) | 15.16 (-1.67) |
| **MXFP4 + RFQ** | **72.16 (-0.52)** | **70.73 (-0.07)** | **90.40 (-0.37)** | **16.44 (-0.39)** |
| HiF4 | 72.42 (-0.26) | 71.27 (+0.47) | 90.50 (-0.27) | 15.35 (-1.48) |
| **HiF4 + RFQ** | **72.81 (+0.13)** | **71.47 (+0.67)** | **90.77 (±0.00)** | **15.66 (-1.17)** |

> ✅ RFQ 成功缩小甚至反超 BF16 基线差距，在多个任务上接近或超越原始性能。

---

### 🔍 **消融实验结果**

#### （1）量化粒度分析（Table 1 & 5）
| 方法 | Wan2.2 训练损失↑ | Qwen3-VL 训练损失↑ |
|------|------------------|--------------------|
| MXFP8 | +0.30% | +0.20% |
| W4A8 | +0.80% | +0.70% |
| MXFP4 | +7.10% | +7.23% |
| **MXFP4 + RFQ** | **+1.14%** | **+1.43%** |
| HiF4 | +2.70% | +3.80% |
| **HiF4 + RFQ** | **+0.61%** | **+0.66%** |

> 💡 发现：**激活量化是主要误差源**，因为即使权重已压缩至 4-bit（W4A8），只要激活保持 8-bit，性能仍稳定；一旦激活也降至 4-bit，性能急剧下降。

#### （2）模块敏感性分析（Figure 2）
- **视觉组件比语言组件更敏感于 4-bit 量化**，尤其在 Wan2.2 的 WanDiT 生成骨干中退化最严重。
- **激活张量具有更重尾分布和更大动态范围**，而权重分布紧凑，更适合低位宽表示。

#### （3）残差路径有效性验证
- 移除残差通路 → 性能回落至标准 MXFP4 水平
- 使用更高精度存储残差 → 带来显著内存与带宽负担，收益有限
- RFQ 以纯 FP4 实现残差重建，在几乎零开销下获得最大增益

---

## 4. **关键结论和发现**

### 🔑 **主要发现**
1. **激活量化是 ultra-low-bit MLLMs 的主要瓶颈**：
   - 相较于权重，激活存在严重的**重尾分布和动态异常值**，在 FP4 下极易造成信息截断与小值下溢。
   - 异常值集中在视觉编码器和生成模块，导致跨模态信息传递失真。

2. **视觉模块比语言模块更脆弱**：
   - 视觉重建任务（如视频生成）对激活保真度要求极高，轻微量化噪声即可破坏时空连贯性。
   - 语言任务虽也有退化，但相对鲁棒。

3. **RFQ 能高效恢复大部分性能损失**：
   - 通过显式建模并补偿量化残差，RFQ 在不改变底层计算格式的情况下，显著提升了激活重建质量。
   - 在 MXFP4 和 HiF4 上均实现接近 BF16 的下游性能。

4. **无需复杂校准或敏感层识别**：
   - RFQ 是一个统一、模态无关（modality-agnostic）的方法，适用于各种 MLLM 架构。

---

### ⚠️ **方法的局限性**
- **依赖 block-level 结构**：RFQ 利用了 block-wise scaling（如 MX/HiF4），在非 block 量化方案中可能不直接适用。
- **仅应用于前向传播**：为避免梯度累积开销，残差路径未参与反向传播，可能限制训练稳定性进一步提升。
- **尚未扩展至 2-bit 及以下**：当前聚焦于 4-bit，极端低位宽下的有效性待验证。

---

### 🔮 **未来工作方向**
1. 探索 **2-bit 乃至 sub-4-bit QAT 设置** 下的 RFQ 变体；
2. 将 RFQ 扩展至更多 MLLM 架构（如 diffusion-based VLMs、agent systems）；
3. 研究是否可将残差机制引入 **backward pass** 以增强训练稳定性；
4. 探索自动学习 fallback indicator $\phi(p,r)$ 而非启发式设定；
5. 在真实边缘设备上部署 RFQ，验证其实际加速与能耗优势。

---

## ✅ **总结一句话**
> 本论文揭示了**激活异常值是制约多模态大模型低位宽量化的根本原因**，并提出 **Residual Fallback Quantization (RFQ)** ——一种简洁高效的残差补偿机制，在几乎零开销下显著恢复了 MXFP4/HiF4 量化后的性能，为 MLLMs 的高效部署提供了实用解决方案。

</details>

---

### 5. [Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090](https://arxiv.org/abs/2608.27370)

**Authors**: Kairong Luo, Jiarui Cui, Yaorui Yin, Shengqi Chen, Yiming Yang, Linxiang Gao, Yanmohan Wang, Mingzhe Zhang, Kaiyue Wen, Kaifeng Lyu, Wenguang Chen  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.27370v1  

#### Abstract
Language model pretraining has become almost synonymous with prohibitive cost, placing it out of reach for much of the academic and open-source communities. Although strong open-source efforts already exist, including open-weight models and open-source training recipes, a cost-efficient, hardware-ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PuRo-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）的预训练成本高昂，动辄数百万美元，使得学术界和开源社区难以复现和研究完整的训练流程。尽管已有许多优秀的开源模型权重发布，但其训练过程（包括数据、代码、配置等）往往不透明，形成了“开放权重”（open-weight）而非“开放配方”（open-recipe）的困境。这导致了可复现性与可访问性之间的巨大鸿沟。

本文旨在解决这一问题，提出一种**低成本、高效率、完全开放**的LLM预训练方案，让资源有限的“贫穷实验室”（poor lab）也能从零开始训练一个具有竞争力的2B级别模型。

### 提出的新方法与创新点
作者团队提出了名为 **PuRo-2B** 的完整预训练配方（recipe），其核心是一个协同设计的、覆盖硬件、算法、数据和系统的效率栈。主要创新点包括：

- **硬件选择**：采用消费级 **RTX 5090** GPU 而非昂贵的数据中心级GPU（如H200）。通过成本效益分析，RTX 5090 在单位计算量的价格上远超竞品，是实现低成本的关键。
- **低精度训练**：全程采用 **blockwise FP8** 混合精度训练。这显著提升了每秒处理的token数量，同时通过精细的分块量化策略保持了模型质量。
- **优化器创新**：采用 **MuonH** 优化器，该优化器为近似尺度不变的参数矩阵（如注意力和MLP层）施加了 **Hyperball** 约束。它将更新归一化并投影回初始的Frobenius半径球面上，从而实现了对有效学习率（effective LR）的显式控制。
- **课程模型平均**（Curriculum Model Averaging, CMA）：在第二阶段训练中，采用数据课程（curriculum）策略，将高质量数据放在后期训练。为了缓解后期学习率衰减导致的更新幅度变小问题，采用了常数学习率（constant-LR）的延续训练，并对最后几个检查点进行平均，以保留后期数据的影响。
- **数据配方**（Data Recipe）：基于代理实验（proxy experiments）来指导数据选择。通过在小规模模型上测试不同数据源或数据切片的效果，获得其能力特征向量，进而决定最终混合数据中的权重分配。

### 相比现有方法的优势
- **极高的成本效益**：在约 **$6.9K** 的计算成本下，最佳模型性能接近 **Qwen2.5-1.5B**；而仅需 **$4.4K** 即可达到 **Qwen2-1.5B** 的水平，成本仅为同类开源模型（如SmolLM3-3B需$719K）的一小部分。
- **完全开放与可复现**：不仅发布了模型权重，还公开了全部数据清单（manifests）、训练代码、数据处理代码和中间检查点，遵循 **Apache 2.0** 开源协议。
- **系统性优化**：不是单一技术的改进，而是硬件、精度、优化、数据等多个层面的协同优化，共同作用于降低总成本。

---

## 2. 核心实验方法和设置

### 数据集
训练数据主要由公开可用的开源数据集构成，涵盖英文、数学、中文、代码和指令微调数据。具体包括：
- **英文**：Nemotron HQ, Nemotron HQ Synthetic, FineWeb-Edu-EN, Cosmopedia-v2, DCLM-Dedup 等。
- **数学**：UltraData-Math, SwallowMath-v2, Nemotron-CC-Math, MegaMath-Web-Pro, OpenWebMath 等。
- **中文**：FineWeb-Edu-CN, ChineseWebText2.0, Deduplicated merged Chinese web 等。
- **代码**：Nemotron Synthetic Code, Swallow-Code-v2, MegaMath-Code, StackExchange 等。
- **SFT/指令**：Nemotron Terminal Corpus, JiuZhang3.0 PT-CoT, Tulu-3 SFT 等。

总训练token数高达 **1.4万亿**（1.4T），其中第一阶段（Phase 1）438.8B，第二阶段（Phase 2）960.0B。

### 实验设置和评估指标
- **模型架构**：基于 **Qwen3-1.7B** 架构的密集型Decoder-only Transformer，总参数量约为 **2B**。
- **硬件平台**：使用配备 **RTX 5090** GPU的集群。Phase 1 使用24个GPU，Phase 2 扩展到96个GPU。
- **训练系统**：基于 **Megatron Core** 和 **Transformer Engine**，支持FP8训练。
- **评估基准**：综合评估了15个基准任务，分为两类：
  - **数学与代码**：GSM8K, MATH, sanitized-MBPP, HumanEval。
  - **推理与知识**：MMLU, MMLU-Pro, ARC-C, ARC-E, BoolQ, CommonsenseQA, HellaSwag, PIQA, SocialIQA, WinoGrande, BBH。
- **评估协议**：使用 **OpenCompass** 统一评估，报告各任务的百分比得分，最终以15个任务的未加权算术平均值作为综合性能指标。

### 基线方法对比
论文对比了多个同规模的 **open-weight** 和 **open-recipe** 模型，包括：
- **Open-Weight**: Qwen2-1.5B, Qwen2.5-1.5B, Gemma-2-2B, Llama-3.2-3B 等。
- **Open-Recipe**: Instella-3B, OLMoE-A1B/7B, Yulan-Mini-2.4B, SmolLM3-3B 等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **PuRo-2B ($6.9K)**：在15个任务上的平均得分为 **57.81**，接近 Qwen2.5-1.5B (60.73)，并超过了 Qwen2-1.5B (55.14)。
- **PuRo-2B ($4.4K)**：在15个任务上的平均得分为 **55.54**，已超过 Qwen2-1.5B 的性能。

### 与基线方法的对比结果
- **成本-性能优势**：如图1所示，PuRo-2B 在所有对比模型中位于最左上方，即以最低的复制成本（reproduction cost）获得了最高的性能，超越了现有的帕累托前沿（Pareto frontier）。
- **与其他开源配方模型相比**：在成本上远低于 OLMoE ($200K) 和 SmolLM3 ($719K)，同时性能具有竞争力。

### 消融实验结果
- **成本节约因素分解**（见图2a）：
  - **RTX 5090 硬件**：相比H200，提供约 **2.77倍** 的峰值计算性价比。
  - **FP8 精度**：在保持模型质量的同时，带来 **1.34倍** 的净加速。
  - **MuonH 优化器**：相比Muon，在相同验证损失下可节省 **16.1%** 的理论计算量。
  - **CMA 配方**：相比均匀数据顺序，CMA带来了 **1.65倍** 的成本效率提升。
- **Puro 成本缩放定律**（Puro Cost Scaling Law）：通过在不同预算下训练，拟合出一条 `P = a + b log2(C - Cp1)` 的缩放曲线。该定律预测，仅需 **$4.4K** 的成本即可达到 Qwen2-1.5B 的性能水平。

---

## 4. 关键结论和发现

### 主要发现
1. **低成本预训练是可行的**：通过精心设计的软硬件协同方案，可以在消费级硬件上以极低的成本（<$10K）训练出性能媲美主流开源模型的2B级别LLM。
2. **系统性优化至关重要**：PuRo-2B的成功并非依赖单一技术，而是硬件选择、低精度训练、先进优化器和智能数据调度等多方面协同优化的结果。
3. **开放配方的价值**：发布完整的训练管道，使得像“课程学习如何影响下游微调”这样的端到端研究成为可能。实验表明，经过CMA训练的初始化模型，在后续的SFT中表现更优。
4. **Puro 成本缩放定律**：为资源有限的研究者提供了一个实用的参考，可以根据预算预估可达到的模型性能。

### 方法的局限性
- **污染与溯源**：训练数据集大多来自已处理过的开源数据，可能存在与评估基准的污染问题，影响结论的严谨性。
- **中文能力**：虽然包含了中文数据，但中文能力并非本次发布的主要目标，因此未在主表中报告中文基准分数。
- **过训练状态**：模型处于“过训练”（overtrained）状态（约700 tokens per parameter），并非计算最优（compute-optimal）。
- **扩展性**：当前的配方和缩放定律是针对2B模型的，尚未证明其在更大模型上的有效性。

### 未来工作方向
1. 将可复现性配方从预训练扩展到后训练（post-training），特别是用于研究智能体（agentic）能力。
2. 探索超越标准密集Transformer的架构空间，如循环Transformer、线性注意力模型、MoE等。
3. 拓展硬件配方，覆盖更广泛的训练预算，研究不同硬件和规模下的最优系统与训练选择。

</details>

---

### 6. [A Multi-Modal AI Framework for Real-Time Queue Prediction, Management and Optimisation in Intelligent Border Control Systems](https://arxiv.org/abs/2608.27010)

**Authors**: Varvara Mama, Eleni Veroni, Nikolaos Kapsalis, Christos D. Nikolopoulos, Anargyros T. Baklezos  
**Category**: cs.AI  
**Published**: 2026-08-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.27010v1  

#### Abstract
In the present work an efficient border control management procedure is proposed. Compared to operational queue management systems, whose operations are based on mostly static data, the proposed work takes into account dynamic traffic conditions, thus enabling optimal performance, even in cases of u...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题：** *A Multi-Modal AI Framework for Real-Time Queue Prediction, Management and Optimisation in Intelligent Border Control Systems*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前边境控制系统面临以下挑战：
- **交通流量高度动态且非平稳**，受地缘政治、季节性和突发事件影响显著；
- 传统队列管理依赖**静态历史数据和规则驱动模型**（如ARIMA、rule-based），难以适应实时波动；
- 缺乏对**多源异构数据**（车辆信息、预注册数据、环境数据等）的有效融合；
- 预测与控制模块分离，导致响应滞后，无法实现**主动式优化管理**。

该研究旨在构建一个能够实现实时**队列预测、资源调度与系统优化一体化**的智能边境控制框架。

---

### 🚀 提出的新方法与创新思路
提出了一种**多模态AI框架**，整合了数据融合、深度学习预测与最优控制策略：

1. **Multi-Modal Data Fusion 架构**  
   融合多种输入源：
   - 实时传感器数据（计算机视觉计数）
   - 历史交通数据
   - 预注册信息（pre-registration）
   - 天气与外部交通指标
   - 渡轮时刻表（ferry scheduling）

2. **LSTM-based Queue Prediction Model**  
   利用 **Long Short-Term Memory (LSTM)** 网络捕捉交通流中的**非线性时间依赖性与突发模式**，提升预测精度。

3. **Model Predictive Control (MPC) + Scheduling Optimization**  
   将预测输出接入 **MPC 控制器**，在滚动时域上求解最优控制策略（车道分配、人员调度、路线引导），实现闭环优化。

4. **Predictive Queue Management System (PQMS)**  
   构建端到端的决策支持系统，为边检官员提供可执行建议，推动从“被动响应”向“主动预测”转变。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（ARIMA / Rule-based） | 本文方法 |
|------|-------------------------------|--------|
| 数据利用 | 单一历史数据 | 多源异构数据融合 |
| 时间建模 | 线性假设，平滑效应强 | 支持非线性、突变建模 |
| 控制机制 | 反应式（reactive） | 预测式（predictive）+ 主动调控 |
| 适应能力 | 对峰值/扰动响应慢 | 动态调整资源配置 |

> ✅ 显著提升了系统的**鲁棒性、自适应性和运营效率**。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
由于真实边境数据受限，作者采用**合成数据模拟真实场景**，包含：
- **车辆到达模式**：基于非齐次泊松过程（Non-homogeneous Poisson process）
- **高峰时段**：通过高斯需求激增模拟（Gaussian demand surges）
- **服务速率**：车道相关的随机处理时间
- **多模态补充数据**：
  - 合成预注册信息
  - 天气条件
  - 渡轮班次数据

> 场景涵盖：正常流量、高峰期、突发拥堵三种运行状态。

---

### ⚙️ 实验设置
- **仿真平台**：离散事件交通模型（discrete-event traffic model）
- **重复次数**：每组实验独立运行10次，确保统计可靠性
- **预测目标**：未来 $ H $ 步内的队列长度 $ Q(t+1:t+H) $
- **控制周期**：实时更新控制动作（车道开放、分流引导）

---

### 📏 评估指标
| 指标 | 公式 | 描述 |
|------|------|------|
| **MSE** | $\frac{1}{N}\sum(Q-\hat{Q})^2$ | 队列预测误差 |
| **RMSE / MAE** | — | 衡量预测稳定性 |
| **Avg Waiting Time** | $\frac{1}{N}\sum(t_{exit} - t_{arrival})$ | 平均等待时间（分钟） |
| **Throughput** | $\frac{\text{processed vehicles}}{\text{time}}$ | 单位小时通行车辆数 |
| **Queue Length (avg)** | — | 平均排队长度 |

---

### 🔁 基线方法对比
| 基线模型 | 类型 |
|---------|------|
| Historical Average | 历史均值法 |
| ARIMA | 经典时间序列模型 |
| Static Rule-Based | 固定规则调度（如固定车道分配） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能表现（见 Table I & II）

#### ✅ 队列预测性能（Table I）
| Model | MSE | MAE | RMSE |
|-------|-----|-----|------|
| Historical Average | 45.2 | 5.8 | 6.72 |
| ARIMA | 31.5 | 4.6 | 5.61 |
| **LSTM (Proposed)** | **20.3** | **3.2** | **4.51** |

> ➤ **相比 ARIMA，MSE 下降约 35%**，显示其更强的时间序列建模能力。

---

#### ✅ 运营性能提升（Table II）
| Metric | Baseline | Proposed | Improvement |
|--------|----------|----------|-------------|
| Avg Waiting Time (min) | 18.5 | **12.9** | ↓ **~30%** |
| Throughput (veh/hr) | 820 | **980** | ↑ **~19.5%** |
| Queue Length (avg) | 35.2 | **24.6** | ↓ **~30.1%** |

> ➤ 显著减少拥堵、提高通关效率。

---

### 🔍 消融实验结果（Ablation Study, Table III）
验证各数据模态对预测性能的影响：

| Configuration | MSE | Performance Drop |
|---------------|-----|------------------|
| Full Model | 20.3 | — |
| Without Pre-registration | 24.1 | +18.7% |
| Without Environmental Data | 22.8 | +12.3% |
| Without Historical Data | 27.5 | +35.5% |

> ➤ **历史数据最关键**，但**预注册与环境数据也显著贡献于精度提升**，证明多模态融合的有效性。

---

### 📉 可视化结果（Fig. 2）
- 在时间序列上的队列预测曲线显示：
  - LSTM 更快响应流量突变；
  - ARIMA 存在明显延迟和平滑失真；
  - 所提方法更贴近 ground truth。

---

### 🧪 极端情况测试
- 模拟**突发车流 surge** 和 **部分传感器失效**：
  - 所提模型性能下降 <10%
  - 基线模型下降 >25%
> ➤ 展现出良好的**鲁棒性与容错能力**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **多模态数据融合 + LSTM 预测** 能有效捕捉复杂、非平稳的边境交通动态；
2. **LSTM + MPC 联合架构** 实现了预测与控制的闭环协同，优于分离式设计；
3. 在高方差场景（如渡轮口岸的burst traffic）中优势尤为突出；
4. 所提框架可推广至不同类型边境（陆路口岸 vs 渡轮码头），具备良好泛化能力；
5. 统计检验（paired t-test, p<0.01）表明性能提升具有**统计显著性**。

---

### ⚠️ 方法的局限性
1. **数据来源依赖性强**：需要高质量、多源数据支持，现实中可能存在缺失或延迟；
2. **基于合成数据验证**：尚未在真实边境系统部署，实际效果需进一步验证；
3. 当前未考虑**个体行为建模**（如旅客选择偏好、逃检行为）；
4. 边缘计算部署尚未实现，实时推理延迟有待评估。

---

### 🔮 未来工作方向
1. 引入 **Reinforcement Learning (RL)** 进行策略优化，替代 MPC 中的手工代价函数设计；
2. 推进与**真实边境系统集成**（如 EES Entry/Exit System）；
3. 开发**分布式边缘AI架构**，支持低延迟现场推理；
4. 加强**不确定性估计与容错机制**（如 missing data imputation, anomaly detection）；
5. 扩展至多国跨境协作场景下的联合调度优化。

---

## 总结
本论文提出了一种面向智能边境控制的**多模态AI框架**，结合 **LSTM 预测 + MPC 优化 + 多源数据融合**，实现了从“被动应对”到“主动预测”的范式转变。实验表明，该方法在预测准确率、平均等待时间、吞吐量等方面全面超越传统方法，尤其适用于高动态、不确定性强的边境交通环境，为下一代智能边境管理系统提供了可行的技术路径。

</details>

---

### 7. [Multi-Dataset Inverse Problem Solving with Distributed Generative AI](https://arxiv.org/abs/2608.26283)

**Authors**: Daniel Lersch, Steven Goldenberg, Johann Rudi, Markus Diefenthaler, Kevin Brager, Xingfu Wu, Yaohang Li, Nobuo Sato  
**Category**: cs.DC  
**Published**: 2026-08-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.26283v1  

#### Abstract
Extracting a shared set of unknown, not directly measurable quantities from multiple, heterogeneous datasets is a common challenge across scientific domains. A prominent example is the combination of datasets obtained from different measurements with different settings (e.g. varying detector resolut...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Multi-Dataset Inverse Problem Solving with Distributed Generative AI**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对科学领域中常见的**多数据集反问题（multi-dataset inverse problems）**，即从多个异构、非独立同分布（non-IID）、覆盖不同特征空间区域的数据集中联合推断一组共享的未知物理参数。这类问题在高能物理（如QuantOm项目）、地球物理、逆散射等领域普遍存在。

传统方法通常存在以下挑战：
- 数据集异构性强（不同探测器分辨率、系统误差、覆盖范围不一致）
- 单独分析各数据集会导致信息丢失，简单合并则可能引入偏差
- 计算量大，难以扩展到大规模数据和HPC环境

### **提出了什么新方法或新思路**
作者提出了一种**基于分布式生成式AI的多数据集反问题求解框架**，是对已有 **SAGIPS（Scalable Asynchronous Generative Inverse Problem Solver）** 框架的扩展。其核心思想是：

- 将每个数据集 $ R_i $ 分配给一个独立的GPU，配备专属的 **forward operator $ P_i $** 和 **discriminator $ D_i $**
- 所有GPU共享一个全局同步的 **generator $ G $**，用于预测共享的未知参数 $ \mathbf{a} $
- 利用 **distributed data-parallel training** 范式，在非独立同分布（non-IID）数据下进行梯度聚合，实现跨数据集的一致性优化

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **架构设计** | 支持异构数据并行处理，避免了“一刀切”的数据合并或串行分析 |
| **理论基础** | 证明了即使数据非IID，只要共享同一组未知参数，梯度累加仍有效（见Appendix A2） |
| **可扩展性** | 支持多节点、多GPU部署，在Polari系统上验证了高达120 GPU的扩展能力 |
| **鲁棒性** | 对未知探测器系统误差（如未建模的不确定性）具有鲁棒性 |
| **灵活性** | 可结合多种gradient transport机制（如ARAR、Double Binary Tree等） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验采用了一个受 **Rutherford散射实验** 启发的合成数据集，构造了6个异构数据集 $ R_0, ..., R_5 $，每个对应不同的探测角度 $ \theta $、效率 $ C_{\text{exp}}(i) $ 和测量不确定性 $ \delta(i) $。

观测值为粒子率 $ v(\theta) $，通过如下公式生成：
$$
v(\theta) = C_{\text{exp}}(i) \cdot \frac{\exp(a_0)\exp(2a_0)}{\sin^4(\theta/2)} \cdot [1 + \delta(i)N(0,1)]
$$
其中 $ a_0 \approx 0.365 $, $ a_1 = -4 $ 是待恢复的真实参数。

所有数据集均取对数形式 $ \ln(v(\theta)) $ 以简化计算。

### **实验设置**
- **模型结构**：
  - Generator 和 Discriminator 均为4层全连接网络，每层128神经元，Leaky-ReLU激活
  - 使用 **GAN 架构** 进行对抗训练
- **训练配置**：
  - Ensemble of 10 GANs，每轮训练10k epochs
  - Adam优化器，学习率：discriminator $ 10^{-4} $, generator $ 10^{-5} $
  - Batch size: 每次生成1k样本，每个样本生成100个合成事件
- **硬件平台**：
  - Run Group A：Jefferson Lab单节点（8×A800 GPU）
  - Run Group B：Argonne国家实验室 Polaris HPC（多节点，每节点4×A100 GPU）

### **评估指标**
| 指标 | 定义 | 说明 |
|------|------|------|
| **Relative Residual** $ r_a = \frac{a - A}{a} $ | 参数估计相对残差 | 衡量准确性，理想值为0 |
| **Uncertainty** $ \Delta r_a = \frac{\Delta A}{a} $ | 基于ensemble的标准差 | 衡量稳定性 |
| **Training Time** | 总训练耗时（秒） | 衡量效率 |
| **GPU Utilization / Memory** | 平均利用率与显存占用 | 衡量资源使用效率 |
| **Drift Metrics** | Pairwise weight/output drift | 评估多GPU间一致性 |

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Single-GPU, Single-Discriminator** | 所有数据拼接后送入单一判别器 |
| **Single-GPU, Multi-Discriminator** | 每个数据集有独立判别器，但共用GPU |
| **Multi-GPU + conv-ARAR / ARAR / Strong-ARAR / Double Binary Tree** | 不同gradient transport策略下的分布式版本 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **参数恢复精度**
- 所有方法在使用全部6个数据集时均能将 $ a_0 $ 和 $ a_1 $ 的相对残差收敛至接近零（within uncertainty）
- **仅用1个数据集时无法唯一确定两个参数**（欠定系统），残差较大
- 随着数据集数量增加，**uncertainty显著下降**，表明信息互补性增强

#### ✅ **与基线方法对比**
| 方法 | 残差表现 | 训练时间（6数据集） | GPU利用率 |
|------|----------|---------------------|------------|
| **Single-GPU (Single Disc.)** | 最差，尤其 $ a_1 $ | ~700s | 高但内存压力大 |
| **Single-GPU (Multi Disc.)** | 中等，n=2时波动明显 | ~800s | 高 |
| **conv-ARAR (Multi-GPU)** | 最佳，稳定收敛 | 6,334s | 8.42% |
| **ARAR (Grouped)** | 较差，精度低 | 2,077s | 28.73% |
| **Strong-ARAR** | 接近最优，速度快 | 2,243s | 20.16% |
| **Double Binary Tree** | 精度好，但最慢 | 9,209s | 1.97% |

> 💡 **结论**：multi-GPU方法在精度上优于single-GPU；Strong-ARAR在速度与精度之间取得最佳平衡。

#### ✅ **消融实验结果**
- **Multiplicity影响（m=4 vs m=20）**：
  - 增加multiplicity（即进一步分片）可带来约 **1.5–3.3倍加速**
  - 参数恢复质量无明显退化，说明数据分片不影响最终性能
- **输入顺序敏感性测试（Up / Down / Mix）**：
  - 若先加入低分辨率数据（Down方案），在n=3时uncertainty显著增大
  - 加入第4个高分辨率数据后迅速恢复 → 表明**高分辨率数据对提升精度至关重要**
- **Gradient Drift分析**：
  - **conv-ARAR**：weight & output drift ≈ 0 → 完全同步
  - **Strong-ARAR**：weight drift小幅上升（~0.01），但output drift几乎为0 → 实际输出一致
  - **ARAR**：drift持续增长 → 同步性差，解释其较低精度

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **多数据集联合分析显著优于单数据集或简单合并**：
   - 多源数据提供互补约束，使原本欠定的问题变为可解
   - 参数估计更精确，uncertainty更低

2. ✅ **non-IID数据下的梯度聚合是可行且有效的**：
   - 只要所有数据集共享相同的未知参数，即可通过分布式训练实现全局一致性
   - 无需假设数据IID，突破了传统DDP的限制

3. ✅ **分布式架构具备良好可扩展性和资源效率**：
   - 每GPU内存占用恒定，不随数据集数量增加而上升
   - 支持进一步数据分片（multiplicity >1），提升训练速度

4. ✅ **方法对未知系统误差具有鲁棒性**：
   - 即使forward model中固定了错误的detector uncertainty（0.1 vs 实际变化值），仍能准确恢复参数

5. ✅ **推荐使用Strong-ARAR作为默认通信策略**：
   - 在精度与训练时间之间达到最佳权衡
   - 输出一致性高，适合大规模部署

### **方法的局限性**
- ❗ **依赖所有数据集由相同物理参数生成**：若数据集间存在冲突物理机制，可能导致梯度矛盾，模型无法收敛
- ❗ **统计代表性不足的数据会影响局部discriminator训练质量**，进而误导generator
- ❗ **缺乏自动加权机制**：当前equal weighting可能不适合统计量差异大的数据集
- ❗ **未处理部分可观测情况**：某些数据集可能只对子集参数敏感，本文未涉及

### **未来工作方向**
- 🔧 开发 **自动数据加权机制** 以应对统计不平衡
- 🔄 探索集成其他生成模型（如Normalizing Flows、Diffusion Models）替代GAN
- 📊 更深入的scaling study：研究随着每数据集GPU数增加的收敛行为
- 🛠️ 引入 **drift-aware synchronization** 或 post-hoc weight averaging 来缓解rank drift
- 🧪 应用于真实QuantOm项目中的三维质子结构成像任务（已在Appendix B展示可行性）

---

> **附录亮点（Appendix B）**：  
> 作者展示了该框架可推广至更复杂场景——从三个混合密度数据集中恢复两个二维密度分布 $ p_0(x,y), p_1(x,y) $。结果显示：
> - 单数据集训练无法恢复真实分布（under-determined）
> - 使用全部三个数据集后，GAN成功重建出接近真实的密度形状  
> ➝ 验证了方法在高维、复杂pipeline下的适用性，为实际核物理应用铺平道路。

</details>

---

### 8. [SimCast-S2S: An Efficient Generative Model for Subseasonal Precipitation Forecasting via Transfer Learning from Climate Simulations](https://arxiv.org/abs/2608.26594)

**Authors**: Hiep V. Dang, Antonios Mamalakis  
**Category**: cs.LG  
**Published**: 2026-08-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.26594v1  

#### Abstract
Subseasonal-to-seasonal (S2S) precipitation forecasting has substantial financial and societal impact, yet remains challenging because of weak predictive signals, high associated uncertainty, and the computational cost of operational systems, which constrains simulation fidelity. We introduce SimCas...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**Subseasonal-to-Seasonal (S2S) 降水预测**中的三大瓶颈问题：
- **弱可预测信号与高不确定性**：S2S 时间尺度（约3–6周）处于天气与气候之间的过渡区，初始条件误差迅速增长，而慢变边界条件尚未形成稳定信号。
- **生成大集合预报的计算成本高昂**：传统基于物理模型的 Ensemble Prediction Systems (EPS) 需要多次数值积分，计算资源消耗巨大。
- **数据驱动模型对训练数据量要求高**：深度生成模型通常需要大量高质量观测数据，但在 S2S 尺度上历史数据有限。

### 提出的新方法：SimCast-S2S
作者提出了一种名为 **SimCast-S2S** 的高效生成式模型框架，结合了以下三项关键技术：

1. **Latent Diffusion Model（潜在扩散模型）**
   - 在由 Variational Autoencoders (VAEs) 学习到的低维**Latent Space**中进行扩散过程建模，而非在原始高维物理空间中直接操作。
   - 显著降低生成成本，支持快速采样大规模概率集合（如100成员）。

2. **Simulation-to-Reanalysis Transfer Learning（仿真到再分析迁移学习）**
   - 利用大规模气候模拟数据（CESM2-LE，28个成员）预训练模型，再通过微调迁移到真实世界再分析数据（ERA5）。
   - 解决真实观测数据不足的问题，并提升模型泛化能力。

3. **Low-Rank Adaptation (LoRA) 微调策略**
   - 在迁移阶段仅更新低秩参数矩阵，冻结大部分预训练权重。
   - 减少过拟合风险，提高训练效率和稳定性。

### 相比现有方法的优势
| 维度 | SimCast-S2S | 传统方法（如 ECMWF-S2S） | 其他 ML 基线（如 CNN/UNet） |
|------|-------------|--------------------------|----------------------------|
| **计算效率** | 极高（GPU分钟级生成百成员集合） | 极低（需数千CPU节点数小时） | 中等（单次前向推理快） |
| **不确定性建模** | 内生生成，无需后处理 | 依赖扰动初值和物理参数 | 多为确定性输出 |
| **数据利用效率** | 通过仿真数据预训练克服数据稀缺 | 不适用 | 数据饥渴，易过拟合 |
| **空间结构保真度** | 更好保留细尺度空间相关性 | 物理一致但可能系统偏差 | 容易过度平滑 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型 | 来源 | 用途 |
|-------|------|------|------|
| **CESM2-LE** | 气候模拟集合 | Community Earth System Model v2 | 预训练（28个成员，1950–2014） |
| **ERA5** | 再分析数据 | ECMWF | 微调与测试（1940–2025） |

> 输入变量包括多个压力层（200/500/850 hPa）的风场、温度、湿度、位势高度及地表变量共21个，目标为未来第15–28天的降水异常。

### 实验设置
- **训练策略**：
  - 先在 CESM2 上预训练 → 再使用 LoRA 在 ERA5 上微调。
  - 所有变量转换为标准化异常（standardized anomalies），去除季节循环和长期趋势。
- **模型结构**：
  - 五个独立 VAE 分别编码不同物理组（wind, mass, thermal, hydro, precip）。
  - Latent Space 压缩比达 4× 至 16×。
  - Diffusion 模型以 VAE 编码后的 latent 表示为条件，预测目标降水 latent。
- **硬件平台**：单块或多块 A100/H200 GPU。

### 评估指标
#### （1）确定性技能（Deterministic Skill）
- **MAE**（Mean Absolute Error）：平均绝对误差。
- **ACC**（Anomaly Correlation Coefficient）：异常相关系数。

#### （2）概率性技能（Probabilistic Skill）
- **RPSS**（Ranked Probability Skill Score）：用于三分位分类预测。
- **CRPSS**（Continuous Ranked Probability Skill Score）：连续分布评分。
- **BSS**（Brier Skill Score）：极端事件（>90%分位）预测能力。

#### （3）不确定性量化
- **PIT Histograms**（Probability Integral Transform）：检验集合是否校准（应接近均匀分布）。
- **Spread vs. RMSE**：集合离散度与均方根误差的一致性。
- **Interval Score (MIS)**：综合评价区间预测质量。

#### （4）空间结构真实性
- **Spatial Autocorrelation**：沿 zonal、meridional、diagonal 方向比较自相关衰减特性。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **深度学习基线** | CNN-Small/Medium/Large, UNet-Small/Medium/Large |
| **物理基线** | ECMWF-S2S（Operational System） |
| **其他生成模型思想参考** | FuXi-S2S（Encoder-Decoder + Perturbation） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Figure 3–6）

| 模型 | MAE (×10⁻² std) | ACC（加权） | CRPSS（全球） |
|------|------------------|--------------|---------------|
| **SimCast-S2S (Ours)** | **1.30 ± 0.06** | **~0.35** | **显著高于 ECMWF-S2S** |
| ECMWF-S2S | 1.32 ± 0.06 | ~0.30 | 参考基准 |
| Best UNet | 1.33 ± 0.06 | — | — |
| Best CNN | 1.31 ± 0.07 | — | — |

> ✅ SimCast-S2S 在所有指标上均优于或持平于最佳基线。

### 与基线方法的对比结果
- **确定性预测**：
  - SimCast-S2S 在 MAE 上优于所有 CNN/UNet 变体和 ECMWF-S2S。
  - ACC 显著更高（Fig. 2），表明其空间模式匹配更优。
- **概率性预测**：
  - **CRPSS 和 RPSS 全球领先**（Fig. 3），尤其在热带太平洋、北美、北大西洋等地优势明显。
  - 季节维度上，在春夏等难预测季节仍保持正技能（Fig. 4）。
- **不确定性表现**：
  - PIT 直方图呈 U 形（under-dispersive），但整体 x²-distance 接近 ECMWF-S2S。
  - Spread-RMSE 关系更贴近 1:1 线，尤其在热带区域优于 ECMWF-S2S（Fig. 6）。
  - Interval Score 更低，说明预测区间更尖锐且覆盖更好。

### 消融实验结果
- **迁移学习有效性验证**：
  - 仅在 ERA5 上训练时，SimCast-S2S 性能下降至 **1.37–1.38 ×10⁻² std**（高于 ECMWF）。
  - 经 CESM2 预训练 + ERA5 微调后，MAE 回升至 **1.30 ×10⁻² std**，证明迁移学习至关重要。
- **Latent Space 设计优势**：
  - 直接在物理空间运行扩散模型不可行（计算不现实），latent space 是实现高效生成的关键。
- **LoRA 的作用**：
  - 若全参数微调，容易破坏从仿真中学到的通用动力学结构；LoRA 能有效适应域偏移而不损害先验知识。

---

## 4. 关键结论和发现

### 主要发现
1. **SimCast-S2S 是首个将 Latent Diffusion Model 成功应用于 S2S 降水预测的工作**，实现了高质量概率预测。
2. **迁移学习 + LoRA 是解决“小样本 + 强不确定性”问题的有效路径**：利用仿真数据扩大训练集规模，显著提升真实场景下的预测技能。
3. **Latent Space 操作极大提升了生成效率**：
   - 单 A100 GPU 生成 100 成员集合仅需 **~20.5 分钟**。
   - 使用 8 块 H200 GPU 可压缩至 **< 2 分钟**（Fig. 8）。
4. **模型不仅准确，而且生成的空间结构更真实**：
   - 在 meridional 和 diagonal 方向上的 autocorrelation 更接近 ERA5（Fig. 7），优于 ECMWF-S2S。
5. **即使未使用后处理、偏差校正或概率标定，SimCast-S2S 仍能挑战最先进的物理系统 ECMWF-S2S**。

### 方法的局限性
- **统计真实性 ≠ 动力学一致性**：
  - 模型无法保证满足质量守恒、动量平衡等物理定律。
  - 生成的场可能是统计合理但动态不一致的。
- **Latent Space 中难以施加显式物理约束**：
  - 当前框架缺乏机制来强制执行 moisture budget 或能量守恒。
- **极端事件建模仍有不足**：
  - 尽管 BSS 表现尚可，但对尾部事件的刻画仍不如理想状态。
- **解释性较弱**：
  - 尚未深入分析模型依赖哪些具体物理机制（如 MJO、大气河等）做出预测。

### 未来工作方向
1. **引入物理一致性约束**：
   - 设计 hybrid sampling scheme，在 latent 生成过程中周期性解码并修正物理残差（如 moisture convergence 不平衡）。
2. **增强可解释性（Explainable AI）**：
   - 应用 Integrated Gradients、Counterfactual Sampling 等技术识别关键预测因子。
3. **扩展至多变量联合预测**：
   - 当前聚焦降水，未来可推广至温度、风速等多变量联合建模。
4. **探索更多仿真数据源与更大模型容量**：
   - 结合 Climax、GraphCast 等 foundation model 思路，构建统一的 Earth System Foundation Model。
5. **实时部署与影响建模应用**：
   - 利用其低延迟特性，服务于农业、水资源、灾害预警等下游任务。

---

> 📌 **总结一句话**：  
> **SimCast-S2S 展示了“Latent Generative Modeling + Simulation-to-Observation Transfer Learning”是一条高效、可扩展、高性能的数据驱动 S2S 降水预测新范式，有望成为未来次季节预报的重要补充工具。**

</details>

---

### 9. [Affix Cache for Diffusion Large Language Models](https://arxiv.org/abs/2608.26140)

**Authors**: Kaihua Liang, An Zhong, Xin Tan, Zafar Ayyub Qazi, Hong Xu, Jian Weng, Marco Canini  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.26140v1  

#### Abstract
Diffusion Large Language Models (DLLMs) enable non-autoregressive decoding and bidirectional context modeling, but efficient inference remains challenging. Unlike autoregressive systems, whose key-value (KV) cache can be reused for shared prefixes, DLLMs couple the KV states of shared context tokens...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Affix Cache for Diffusion Large Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统的 **Autoregressive LLMs (AR LLMs)** 支持高效的 **prefix caching**，即对共享前缀（如系统提示）的 Key-Value (KV) 缓存可直接复用，显著提升推理效率。然而，**Diffusion Large Language Models (DLLMs)** 由于采用 **bidirectional attention**，其 KV 状态会随着生成过程动态变化，导致共享上下文的缓存状态“过时”（stale），无法直接复用。若完全不复用，则需频繁全量重新计算，带来高昂开销。

现有 DLLM 推理框架（如 dLLM-Cache、dKV-Cache、Fast-dLLM）虽尝试通过周期性全量重算或近似缓存策略缓解问题，但仍无法实现跨请求的 **affix-level cache reuse**（即对任意位置共享文本片段的复用）。

### **提出的新方法：ACache**
本文提出 **ACache (Affix Cache)**，一种面向 DLLMs 的细粒度缓存复用机制，核心思想是：
- 将共享文本视为 **affix**（可为 prefix、infix 或 suffix），支持跨请求复用其 KV 缓存。
- 引入 **Anchor Tokens**：在每个请求中识别一小部分对当前生成任务影响最大的 affix token。
- **选择性重算**：仅对这些 Anchor Tokens 和请求特定部分进行 KV 重算，其余 affix 缓存直接复用。

### **相比现有方法的优势**
- **首次实现跨请求的 affix 缓存复用**：突破传统 prefix-centric 缓存限制，利用双向注意力特性支持任意位置共享文本的高效复用。
- **高精度恢复**：仅重算约 20% 的 affix tokens 即可恢复因直接缓存复用导致的精度损失。
- **显著提升效率**：在真实推理引擎上实现高达 **55.7% 的重算延迟降低** 和 **1.68× 的端到端吞吐提升**。
- **通用性强**：适用于不同任务、模型（LLaDA、Dream）、affix 类型（前缀/中缀/后缀）和 shot 数。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **GSM8K**：数学推理任务，测试模型逐步推理能力。
- **MBPP**：代码生成任务，评估编程能力。
- **BABILong-0k/qa1**：长上下文检索任务，测试记忆与定位能力。

### **实验设置**
- **模型**：
  - **LLaDA-8B-Instruct**：从零训练的 diffusion LLM。
  - **Dream-v0-Instruct-7B**：基于 AR 模型初始化的 diffusion LLM。
- **硬件**：NVIDIA A100-SXM4-40GB GPU。
- **框架**：
  - 质量评估使用 Hugging Face Transformers。
  - 效率评估基于自研 **Nano-vLLM** 推理引擎原型（集成 Fast-dLLM）。
- **默认参数**：
  - 生成长度：256 tokens（BABILong 为 8）。
  - Block 长度：32。
  - 置信阈值：0.9。
  - Anchor ratio 扫描范围：{0, 0.1, 0.2, 0.3, 0.5, 1.0}。

### **Affix 构造方式**
- **Prefix**：`Ex Qry Mask`
- **Infix**：`Qry_head Ex Ans_prompt Mask`
- **Suffix**：`Qry Mask Ex`

其中 `Ex` 为共享的 few-shot 示例，作为 affix。

### **评估指标**
- **Accuracy**：任务准确率（GSM8K、MBPP、BABILong）。
- **Recompute Latency**：KV 重算阶段的前向延迟（ms）。
- **Throughput**：每秒生成 token 数。
- **Peak KV Cache Memory**：峰值 KV 缓存占用（GB）。

### **基线方法**
- **Baseline**：基于 Fast-dLLM 的 Nano-vLLM 原型，无 ACache，周期性全量重算。
- **Direct Reuse (Anchor ratio = 0)**：直接复用整个 affix 缓存（会导致精度下降）。
- **Full Recomputation (Anchor ratio = 1)**：完全重算所有 affix tokens。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 准确率恢复效果（Figure 4）**
- 在 **1-shot 设置下**，直接复用（ratio=0）导致严重精度下降（如 LLaDA GSM8K 下降达 49.5%）。
- 使用 **ACache (ratio=0.2)** 可恢复大部分精度：
  - 1-shot 平均准确率从 37.28% 提升至 **52.61%**。
  - 2-shot 下接近全重算性能（58.03% vs 58.41%）。
- 对 **infix 和 suffix** 同样有效，验证了 affix 复用的通用性。

#### **(2) 推理效率提升（Tables 1 & 2）**
| 模型 | 数据集 | Batch | 重算延迟降低 | 吞吐提升 |
|------|--------|-------|---------------|-----------|
| LLaDA | GSM8K | 16 | **55.7%** | **1.60×** |
| LLaDA | MBPP | 16 | 53.7% | **1.68×** |
| Dream | GSM8K | 16 | 42.7% | 1.55× |
| Dream | MBPP | 16 | 40.0% | **1.63×** |

> 注：Batch=1 时因 Anchor selection 开销未充分摊销，可能略慢；但 Batch≥4 后全面优于基线。

#### **(3) 内存优化（Appendix Table 4）**
- 峰值 KV 缓存内存最高降低 **43.3%**（LLaDA GSM8K, 4-shot, batch=16）。
- 共享前缀越长、批大小越大，节省越明显。

### **消融实验结果**

#### **(1) 非 Anchor 缓存是否必要？（Figure 5 & 6）**
- **KeepNA**（保留非 Anchor 缓存） vs **DropNA**（丢弃非 Anchor 缓存）：
  - 在 Anchor ratio=0.2 时，LLaDA 上准确率差距达 **17.66%**（43.01% vs 25.35%）。
  - Dream 上也有 **6.57%** 差距。
- 结论：**非 Anchor 缓存携带重要共享上下文信息，不能简单丢弃**。ACache 不是 token 剪枝，而是精细的状态同步机制。

#### **(2) Anchor 选择机制对比（Figure 7）**
- 与 **CacheBlend-style HKVD**（基于 KV 差异）对比：
  - ACache 在 1-shot 下平均准确率高出 **11.04%**（ratio=0.2）。
- 结论：**masked-to-affix attention 信号更适合作为 Anchor 选择依据**，因其直接反映生成 token 对 affix 的依赖。

#### **(3) Anchor Selection 开销（Appendix B.3）**
- 单次开销：**48.1–74.0ms**，主要来自一次额外前向传播（attention probe）。
- 该开销可被后续重算节省快速摊销，**batch≥2 时即可实现净收益**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **DLLMs 支持 affix-level 缓存复用**：得益于 bidirectional attention，共享文本无论位于前、中、后均可作为缓存对象。
2. **Anchor Tokens 是关键**：仅少量 affix tokens 对生成结果有显著影响，选择性重算即可维持上下文一致性。
3. **ACache 显著提升效率与精度平衡**：以极低重算代价（~20% tokens）恢复几乎全部精度，并大幅提升吞吐。
4. **非 Anchor 缓存具有语义价值**：直接丢弃会导致严重性能下降，说明缓存复用不仅是计算优化，更是上下文传递机制。

### **局限性**
1. **原型仅支持 prefix**：受限于 Nano-vLLM 的 paged KV 设计，目前系统原型仅实现 prefix 复用，infix/suffix 需更复杂的物理-逻辑映射。
2. **依赖预定义 affix**：当前假设共享文本在推理前已知且固定，缺乏在线发现共享模式的能力（如 AR 中的动态 prefix 匹配）。
3. **Anchor selection 为一次性操作**：未考虑 decoding 过程中 Anchor 动态变化的可能性。

### **未来工作方向**
- 支持 **infix/suffix 的完整运行时实现**，验证其在真实场景下的端到端收益。
- 设计 **动态 affix 发现机制**，自动识别高频共现文本并构建共享缓存池。
- 探索 **动态 Anchor 更新** 策略，适应 decoding 过程中的上下文演化。
- 将 ACache 与其他 DLLM 加速技术（如 dKV-Cache、FlashDLM）结合，进一步优化性能。

---

> **一句话总结**：  
> ACache 首次实现了 DLLMs 中跨请求的 affix 缓存复用，通过引入 Anchor Tokens 实现“选择性同步”，在仅重算 ~20% affix tokens 的情况下恢复精度，并在真实系统中实现高达 1.68× 的吞吐提升，为 DLLM 高效推理开辟了新路径。

</details>

---

### 10. [Toward Equitable Low-Carbon Mobility: Fairness-Aware Demand Prediction for Expanding Bike-Sharing Systems](https://arxiv.org/abs/2608.26451)

**Authors**: Man Luo, Yixuan Zhao  
**Category**: cs.LG  
**Published**: 2026-08-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.26451v1  

#### Abstract
Bike-sharing systems are an important component of low-carbon urban mobility, but continued expansion creates challenges in both cold-start prediction and equitable resource allocation. Newly deployed stations lack historical ridership records, causing a mismatch between training and inference for g...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**低收入社区在共享单车系统扩展中面临的结构性不平等问题**，指出当前基于历史需求数据的预测模型存在两大挑战：
- **冷启动问题（Cold-Start Problem）**：新建站点缺乏历史骑行记录，导致图神经网络（GNN）在训练与推理阶段存在分布差异。
- **公平性偏差（Fairness Bias）**：低收入社区的历史骑行量偏低可能源于基础设施不足而非真实需求弱，若直接用此类数据训练模型，会**复制并加剧现有的空间不平等**，使资源持续向高收入区域倾斜。

### **提出的新方法：FairGIN**
作者提出了 **FairGIN（Fairness-aware Graph Neural Network）**，一个面向扩展型共享单车系统的公平感知图神经网络框架，包含三个核心组件：

1. **Expansion-Simulated Increment Training (ESIT)**  
   在训练过程中随机将部分现有站点模拟为“伪新站点”，并屏蔽其历史需求序列，以模拟实际部署时的冷启动场景，从而缩小训练与推理之间的图结构与特征分布差距。

2. **Attention-Based Knowledge Transfer**  
   引入可学习的温度缩放注意力机制，使新站点能从已有站点自适应地转移知识。通过正交映射对齐嵌入空间，并结合门控机制融合本地空间特征与迁移表示，提升冷启动站点的表征质量。

3. **Fairness-Aware Optimization**  
   在损失函数中引入**收入分层正则化项（income-stratified regularization）**，显式减少不同收入群体间的预测需求差异；同时设计**公平校准的部署评分函数**，在决策层面优先考虑低收入区域。

### **相比现有方法的优势**
- **统一建模动态拓扑与公平性**：首次将归纳式图学习（inductive GNN）与公平性优化联合建模，解决了“冷启动”与“公平性”双重挑战。
- **端到端可训练**：所有模块可联合优化，无需后处理或额外干预。
- **部署灵活**：支持对任意新站点进行零样本预测，并可通过调节公平权重实现效率与公平的权衡。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验基于两个公开的真实世界共享单车数据集：

| 数据集 | 城市 | 时间范围 | 现有站点数 (N) | 新增站点数 (M) | 收入中位数 |
|-------|------|----------|----------------|----------------|-------------|
| **NYC Citi Bike** | 纽约 | 2018–2023 | 746 | 312 | \$72,800 |
| **Seattle Bikeshare** | 西雅图 | 2017–2018 | 186 | 61 | \$93,500 |

**辅助数据来源**：
- POI 分布（OpenStreetMap）
- 道路网络（OSMnx）
- 天气（NOAA）
- 出租车流量（TLC/Rideshare）
- 收入信息（ACS 5年估计）

### **实验设置与评估指标**

#### **任务设定**
- 在现有站点上训练模型，在**从未见过的新站点**上测试需求预测能力（归纳学习 setting）。
- 所有基线方法均适配至相同协议：新站点无历史需求输入。

#### **评估指标**

| 类别 | 指标 | 描述 |
|------|------|------|
| **准确性** | MAE, RMSE | 新站点上的平均绝对误差与均方根误差 |
| **公平性** | RFG（Region-based Fairness Gap） | 不同收入组人均预测需求差异 |
| | IFG（Individual-based Fairness Gap） | 基于人口加权的个体站点级不公平度量 |
| | \|ρ\|（Spearman’s ρ） | 预测需求与街区收入的相关性，越接近0越好 |

#### **基线方法对比**
涵盖六类代表性方法：
- **统计模型**：ARIMA, LSTM
- **时空图模型**：STGCN, DCRNN
- **公平感知模型**：FairST
- **归纳式GNN**：GraphSAGE, DA-MRGNN, KITS
- **城市大模型**：UrbanGPT

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table II）**

| 方法 | NYC – MAE ↓ | NYC – RFG ↓ | Seattle – MAE ↓ | Seattle – \|ρ\| ↓ |
|------|------------|-------------|------------------|--------------------|
| KITS（最佳准确率基线） | 2.65 | 2.02 | 2.19 | 0.326 |
| FairST（最佳公平性基线） | 2.91 | 1.89 | 2.43 | 0.309 |
| **FairGIN（本文）** | **2.23** | **1.14** | **1.87** | **0.162** |
| **相对提升** | +15.8% MAE | -33.3% RFG | +14.6% MAE | -45.5% \|ρ\| |

> ✅ FairGIN 在**所有指标上均达到 SOTA**，且在保持高精度的同时显著降低收入相关偏见。

### **与基线方法的对比结果**
- **传统图模型（STGCN/DCRNN）表现差**：因无法处理动态拓扑变化，在冷启动下性能严重下降。
- **归纳式方法（GraphSAGE/KITS）提升准确率但未改善公平性**：虽能泛化至新节点，但未引入公平约束，仍延续历史不平等模式。
- **FairST 公平性较好但准确率受限**：基于固定图结构，难以适应新站点加入后的拓扑演化。
- **UrbanGPT 表现中等**：依赖提示工程，空间推理不足以纠正系统性群体偏差。

### **消融实验结果（Ablation Study）**
移除各模块后在 NYC 上的表现退化（Fig. 3）：

| 变体 | MAE ↑ | RFG ↑ | 说明 |
|------|--------|--------|------|
| w/o ESIT | +27.3% | +64.0% | 冷启动模拟至关重要 |
| w/o KT（知识迁移） | +17.0% | +47.4% | 注意力机制有效识别相似源站点 |
| w/o FL（公平损失） | +2.2% | +69.3% | 显著增加不公平性，验证公平正则必要性 |
| w/o TS（温度缩放） | +8.1% | +17.5% | 自适应注意力集中提升鲁棒性 |

> 🔍 **ESIT 是基础，FL 是公平性的关键驱动因素**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **冷启动与公平性需协同建模**：仅提高预测准确性无法解决结构性不公；必须在嵌入学习阶段就注入公平意识。
2. **FairGIN 实现效率与公平双赢**：在 NYC 和 Seattle 上均显著优于各类基线，证明其跨城市泛化能力强。
3. **公平部署策略有效提升低收入区覆盖**：使用 `Score = ŷ + α·I[j∈g⁻]` 进行排序后，低收入站点入选比例从 ~20% 提升至 **>55%**，且所选站点仍有较高预测需求（Table III & V），表明**公平性提升未牺牲服务效率**。
4. **参数敏感性合理可控**：ESIT 掩码率 `p=0.15`、公平权重 `λ_fair=0.3` 为最优配置，符合纽约年均扩张速率（10–20%）。

### **方法的局限性**
- **二元收入划分**：仅以城市中位收入划分为高低两组，忽略了低收入内部的异质性（如 Table VI 显示极低收入区预测误差更高）。
- **单一保护属性**：仅考虑收入，未纳入种族、汽车拥有率等其他社会脆弱性维度。
- **静态特征假设**：空间特征（POI、人口密度）未随时间更新，长期部署可能失效。
- **地理适用性待验证**：目前仅在美国有桩系统验证，是否适用于无桩系统或全球南方数据稀疏地区尚不明确。

### **未来工作方向**
- 使用**连续敏感变量**（如 Wasserstein fairness）或**多属性公平目标**（multi-attribute fairness）。
- 构建**动态更新的空间编码器**，定期融合最新城市数据。
- 探索**跨城市迁移学习**，缓解数据稀缺地区的冷启动问题。
- 将框架推广至其他低碳出行方式（如电动滑板车、微公交）的公平规划。

---

> 📌 **总结一句话**：  
> **FairGIN 成功将归纳式图学习与公平性优化相结合，在提升新建共享单车站点需求预测精度的同时，显著减少了对低收入社区的系统性忽视，为构建真正包容的低碳交通系统提供了可落地的技术路径。**

</details>

---

### 11. [A Unified Framework for Fair and Personalized Decentralized Learning under Communication Constraints](https://arxiv.org/abs/2608.26493)

**Authors**: Krishnendu S. Tharakan, Carlo Fischione  
**Category**: cs.LG  
**Published**: 2026-08-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.26493v1  

#### Abstract
Decentralized learning systems aim to collaboratively train models across multiple clients without relying on a central coordinator. While decentralization improves scalability, privacy, and robustness, it also exacerbates three fundamental challenges: statistical heterogeneity across clients, fairn...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Unified Framework for Fair and Personalized Decentralized Learning under Communication Constraints

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在解决**去中心化学习**（decentralized learning）中三个核心挑战的联合优化问题：
- **统计异质性**（statistical heterogeneity）：客户端数据分布非独立同分布（non-iid），导致单一全局模型对部分客户端表现不佳。
- **公平性**（fairness）：传统方法优化平均性能，可能牺牲少数或高损失客户端的利益。
- **通信约束**（communication constraints）：频繁传输高维模型更新在带宽受限、能耗敏感的边缘网络中不可行。

核心问题是：**在通信受限条件下，去中心化学习能实现多高的公平性？**

### 提出的新方法与新思路
作者提出了 **DMFL-SQ**（Decentralized Multi-Task Fair Learning with Sparse-Quantized Event-triggered Communication），一个统一的算法框架，首次将以下三者集成于同一去中心化学习范式中：
- **图结构个性化**（graph-based personalization）：通过图正则项 $ R_g(w) $ 耦合相邻客户端的模型，允许个性化同时利用相似性。
- **无先验公平性目标**（agnostic fairness）：采用 agnostic mixture 风险 $ \Phi(w) = \sup_{\lambda \in \Lambda} \sum_i \lambda_i F_i(w) $，优化最坏情况下的混合风险，提升对高损失客户端的鲁棒性。
- **压缩事件触发通信**（compressed event-triggered communication）：结合稀疏化（sparsification）、量化（quantization）和事件触发机制，仅当模型变化足够大时才发送稀疏量化的更新。

### 相比现有方法的优势
| 特性 | 现有方法（如 D-PSGD, CHOCO-SGD, q-FFL） | DMFL-SQ |
|------|----------------------------------------|--------|
| 去中心化 | 部分支持 | ✅ 完全去中心化 |
| 个性化 | ❌ 多数为全局模型 | ✅ 图正则化支持个性化 |
| 公平性 | ❌ 或仅限中心化场景 | ✅ 支持去中心化公平优化 |
| 通信效率 | ✅ 支持压缩 | ✅ 支持稀疏+量化+事件触发 |
| 理论保证 | 部分有收敛性 | ✅ 同时提供 **non-convex 收敛率** 和 **PAC-Bayes 泛化界** |

DMFL-SQ 是首个在**完全去中心化、非凸目标**下，同时兼顾个性化、公平性和通信效率，并提供严格理论分析的框架。

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **CIFAR-10**：
   - 标准图像分类任务。
   - 数据划分为 20 个客户端，使用 Dirichlet 分布（$\alpha=0.1$）生成强 non-iid 划分。
   - 模型：CNN（两层卷积 + 两层全连接）。

2. **MUSMET EEG Dataset**：
   - 真实世界脑电图（EEG）情感识别数据集，包含 20 名音乐家的多模态数据。
   - 任务：四分类（aggressive, happy, relax, sad）。
   - 每位音乐家作为一个客户端，天然具有高度异质性。
   - 模型：3 层 MLP（隐藏层 256, 128）。

### 实验设置和评估指标
- **通信图**：默认环形拓扑（ring），也测试 ER 和 RGG 图。
- **训练轮次**：固定 $ T = 500 $。
- **本地更新**：每轮一次 local update。
- **评估指标**：
  - **平均准确率**（Average Accuracy）
  - **最差客户端准确率**（Worst-client Accuracy）
  - **底部 10% 客户端平均准确率**（Bottom-10% Accuracy）
  - **准确率标准差**（Accuracy Std）
  - **Jain’s Fairness Index**（衡量公平性）
  - **总通信开销**（Bits per client）

### 基线方法对比
- **D-PSGD**：去中心化 SGD，全精度模型交换。
- **CHOCO-SGD**：去中心化，使用压缩 gossip 通信。
- **DSGT**：去中心化梯度追踪，提升收敛性。
- **q-FFL**：基于服务器的公平联邦学习（作为公平性参考）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（MUSMET 数据集，Table II）
| Method | Avg. Acc. ↑ | Worst Acc. ↑ | Bottom-10% Acc. ↑ | Jain's Index ↑ | Bits/Client ↓ |
|--------|-------------|--------------|-------------------|----------------|---------------|
| DSGT | 0.3132 | 0.0501 | 0.0610 | 0.7103 | 1998.3 M |
| q-FFL | 0.4114 | 0.0716 | 0.1091 | 0.7983 | 1618.0 M |
| D-PSGD | 0.5716 | 0.1957 | 0.2236 | 0.8713 | 1438.6 M |
| CHOCO-SGD | 0.6928 | 0.1935 | 0.2320 | 0.9074 | 97.4 M |
| **DMFL-SQ** | **0.8030** | **0.4810** | **0.5160** | **0.9779** | **0.6 M** |

### 与基线方法的对比结果
- **准确性**：DMFL-SQ 在 **CIFAR-10** 和 **MUSMET** 上均达到最高平均准确率。
- **公平性**：显著提升最差客户端和尾部客户端的性能，Jain’s Index 接近 1，表明极高的公平性。
- **通信效率**：
  - 通信量仅为 **0.6 Mbits/client**，远低于所有基线（即使是压缩的 CHOCO-SGD 的 97.4 M）。
  - 在相同通信预算下，DMFL-SQ 准确率远超其他方法（见 Fig. 2, 8）。

### 消融实验结果（Table III）
消融实验验证了各组件的重要性：
| Variant | Worst Acc. ↑ | Bits/Client ↓ |
|---------|--------------|---------------|
| No fairness ($p=0$) | 0.3743 | 0.3 M |
| No graph coupling ($\sigma=0$) | 0.3862 | 0.2 M |
| No event-triggering | 0.4397 | 4.0 M |
| No sparsification | 0.4504 | 2.2 M |
| No quantization | 0.4405 | 5.3 M |
| **DMFL-SQ** | **0.4810** | **0.6 M** |

- 移除 **fairness** 或 **graph coupling** 显著降低公平性。
- 移除 **event-triggering**, **sparsification**, 或 **quantization** 导致通信成本急剧上升（4–50 倍），且公平性下降。
- 表明 **三者协同作用** 才能实现最优的 accuracy-fairness-communication 权衡。

---

## 4. 关键结论和发现

### 主要发现
1. **公平性、个性化与通信效率可兼得**：在去中心化学习中，通过合理设计（图耦合 + 无先验公平目标 + 压缩事件触发），可以同时实现高公平性、高性能和低通信开销。
2. **理论收敛性不受影响**：尽管引入了压缩、延迟和事件触发，DMFL-SQ 仍能达到 **$ O(T^{-1/2}) $** 的非凸收敛速率，与标准 SGD 相当。
3. **真实异质数据上表现优异**：在自然异质的 MUSMET EEG 数据上，DMFL-SQ 不仅准确率最高，且显著提升了模型公平性。
4. **通信成本大幅降低**：相比基线，通信量减少 **两个数量级以上**，适用于资源受限的边缘设备。

### 方法的局限性
- **依赖图结构质量**：性能受通信图连通性影响（见 Fig. 7），稀疏图可能导致信息传播慢。
- **公平性选择存在差距**：经验选择的公平成分与总体最优之间存在 selection gap，虽在理论中被控制，但仍可能影响初期收敛。
- **超参数调优复杂**：需平衡 $ p $（公平权重）、$ \sigma $（图耦合强度）、触发阈值 $ \theta_t $ 等多个参数。

### 未来工作方向
- 将框架扩展至 **动态图拓扑** 和 **异步通信** 场景。
- 探索更高效的 **自适应压缩与触发策略**。
- 研究 **隐私保护**（如差分隐私）与当前框架的结合。
- 将 PAC-Bayes 分析应用于更多类型的公平性目标。

---

> **总结**：DMFL-SQ 成功构建了一个在 **decentralized, non-convex, communication-constrained** 设置下，统一处理 **personalization, fairness, and efficiency** 的理论与算法框架，实验证明其在准确率、公平性和通信效率上全面超越现有方法，为实际部署提供了强有力的支持。

</details>

---

### 12. [AffectOmni: RL-Verifiable People-Centric Grounded Affective Reasoning for Social and Art-Related Scenes](https://arxiv.org/abs/2608.26193)

**Authors**: Yibo Wang, Rui Yang, Jisheng Dang, Bimei Wang, Yitao Wu, Pengfei Cao, Wencan Zhang, Hong Peng, Bin Hu, Tat-Seng Chua  
**Category**: cs.AI  
**Published**: 2026-08-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.26193v1  

#### Abstract
Multimodal large language models (MLLMs) achieve strong performance on VQA and scene understanding, yet affective reasoning remains vulnerable to shortcut behavior. Models may predict correct answers while neglecting people-centric cues such as micro expressions and body language, which weakens trac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AffectOmni: RL-Verifiable People-Centric Grounded Affective Reasoning for Social and Art-Related Scenes

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Multimodal Large Language Models (MLLMs)** 在视觉问答（VQA）和场景理解任务中表现优异，但在 **affective reasoning**（情感推理）方面存在严重短板。尽管模型可能给出正确答案，但其推理过程常依赖于全局上下文线索（如背景、音乐等），而忽视了关键的 **people-centric cues**（以人为中心的线索），例如微表情（micro expressions）、肢体语言（body language）和人际互动动态。

这种“捷径行为”（shortcut behavior）导致：
- 推理过程不可追溯（untraceable）
- 难以进行外部验证（non-verifiable）
- 在真实世界应用（如心理辅导、社交机器人）中缺乏可信度

此外，现有的 **RLHF / GRPO** 强化学习框架在奖励设计上存在不足，尤其是 **LLM as a Judge** 机制容易出现 **score clustering** 和 **calibration drift**，导致不同质量的推理路径获得相似分数，削弱了奖励信号的区分能力。

---

### 提出了什么新方法或新思路
作者提出 **AffectOmni**，一个基于 **GRPO** 的可验证情感推理框架，核心创新如下：

#### ✅ 创新点一：细粒度、以人为中心的奖励机制（Fine-Grained People-Centric Rewards）
引入两个新的奖励项，显式引导模型关注人类情感证据：
- **People Focus Reward (Rppl)**：鼓励模型描述面部表情、身体动作和人际互动。
- **Temporal Order Reward (Rtmp)**：鼓励模型按时间顺序组织推理，捕捉情绪随时间的变化轨迹。

#### ✅ 创新点二：组内比较评分策略（Within-Group Comparative Scoring）
替代传统的独立绝对评分（absolute scoring），将同一输入下的多个候选输出一起提交给 **LLM as a Judge**，要求其进行相对排序。该策略显著提升了奖励信号的 **discriminability**（区分性）和稳定性。

#### ✅ 创新点三：从推理到证据的可验证接口（Reasoning-to-Evidence Grounding）
提出一个后处理验证模块，将自由形式的推理链压缩为结构化的 **Minimal Evidence Package (MEP)**，并通过 **SAM3** 将其映射为视频帧中的像素级分割区域，实现：
- 外部可审计（externally auditable）
- 可验证（verifiable）
- 支持 falsifiability（证伪性）诊断

---

### 相比现有方法的优势
| 维度 | AffectOmni | 现有方法（如 HumanOmniV2） |
|------|-----------|-----------------------------|
| **推理焦点** | 显式关注人本线索（微表情、姿态） | 依赖全局上下文，忽略细节 |
| **奖励信号** | 组内比较，高区分性 | 独立评分，易出现 score clustering |
| **可验证性** | 提供可执行的证据指令 + SAM3 分割可视化 | 仅输出文本推理，无法外部验证 |
| **训练目标** | 不仅追求答案正确，更强调推理过程可信 | 主要优化答案准确率 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **IntentBench**：作者构建的情感与意图理解基准，包含来自 Social-IQ 2.0、EMER、MDPE 的 633 个视频，共 2,689 个问题。
- **Daily-Omni**：日常场景音视频问答基准，684 个视频，1,197 个问题，覆盖六类任务。
- **WorldSense**：世界知识音视频问答基准，1,662 个视频，3,172 个问题，涵盖八大学科领域。
- **NExT-QA**：通用视频问答基准，用于测试跨域泛化能力。

---

### 实验设置和评估指标
- **基础模型**：Qwen2.5-Omni-7B-Thinker
- **训练框架**：Group Relative Policy Optimization (GRPO)
- **采样策略**：每条样本生成 $ G=4 $ 个候选响应
- **奖励权重**：$ \lambda_p = 0.2 $（People Focus），$ \lambda_t = 0.2 $（Temporal Order）
- **硬件配置**：4×A100 80GB，使用 DeepSpeed ZeRO Stage 2
- **评估指标**：Accuracy (%)，按问题类别细分（如 Emotion、How、When 等）

---

### 基线方法对比
#### 开源模型：
- Qwen2.5-Omni
- HumanOmniV2 (7B)
- MiniCPM-o
- Ola
- VITA-1.5
- VideoLLaMA2

#### 商业闭源模型（proprietary）：
- GPT-4o, GPT-o1(think)
- Gemini系列（2.5-Pro, 1.5 Pro, Flash等）
- Claude3.5 Sonnet

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 数据集 | AffectOmni 性能 | 最强开源基线（HumanOmniV2） | 提升幅度 |
|--------|------------------|-------------------------------|----------|
| **IntentBench (Avg)** | **71.89%** | 69.23% | **+2.66 pp** |
| **IntentBench (Emotion)** | **85.44%** | 80.78% | **+4.66 pp** |
| **IntentBench (When)** | **71.43%** | 57.14% | **+14.29 pp** |
| **Daily-Omni (Avg)** | **61.90%** | 58.47% | **+3.43 pp** |
| **WorldSense (Avg)** | **48.80%** | 47.70% | **+1.10 pp** |

> 注：pp = percentage point

---

### 与基线方法的对比结果
- 在 **IntentBench** 上，AffectOmni 超越所有开源 7B 模型，并在 **Emotion** 和 **When** 类别上取得最大提升，表明其对人本线索和时序推理的有效建模。
- 在 **Daily-Omni** 上，AffectOmni 在 Context、Reason、60s 等维度均显著优于 HumanOmniV2。
- 在 **WorldSense** 上，AffectOmni 平均精度略超 **Gemini 1.5 Pro**，尤其在 **Music** 领域领先 3.9 pp。
- 在 **NExT-QA** 上，AffectOmni 达到 **80.52%** 准确率，优于 HumanOmniV2（79.78%），证明其未因情感专项训练而遗忘通用视频理解能力。

---

### 消融实验结果（Ablation Study）

| 配置 | Emotion | When | Avg |
|------|--------|------|-----|
| Baseline (Acc-only) | 80.78 | 57.14 | 69.23 |
| + People-Focus (Rppl) | 84.23 | 57.14 | 69.78 |
| + Temporal-Order (Rtmp) | 83.67 | 57.14 | 69.62 |
| **Full Model (AffectOmni)** | **85.44** | **71.43** | **71.89** |

- **People-Focus Reward** 对 Emotion 类别提升显著（+3.45 pp），说明其有效引导模型关注微表情。
- **Temporal-Order Reward** 单独作用有限，但与 Rppl 结合后使 When 类别大幅提升 **+14.29 pp**，体现协同效应。
- 完整模型相比基线提升 **+2.66 pp**，验证了多维奖励设计的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **答案正确 ≠ 推理可信**：许多模型虽能答对，但推理过程漂移至背景线索，缺乏对人本证据的关注。
2. ✅ **细粒度奖励可引导可信推理**：通过 Rppl 和 Rtmp，可有效抑制 shortcut learning，促使模型构建基于微表情和时序变化的证据链。
3. ✅ **比较评分优于绝对评分**：within-group comparative scoring 显著缓解 score clustering，CV（变异系数）从 0.153 提升至 0.325，奖励信号更具区分性。
4. ✅ **可验证接口具有实用价值**：通过 MEP + SAM3 构建的 **Executable Evidence Interface (EEI)**，支持外部一致性检查。人工审计显示，当分割结果相关时，准确率达 **83.3%**；不相关时仅为 **35.3%**，说明该接口具备 falsifiability 能力。

---

### 方法的局限性
- **Temporal interval localization 粗糙**：当前方法对时间区间的定位仍较粗略，难以精确匹配短时情绪波动。
- **误差传播未量化**：从推理 → 总结 → SAM3 分割的链条中，各阶段错误可能累积，影响最终验证可靠性。
- **Appearance cues 利用不足**：实验发现模型较少依赖颜色、服装等外观特征，在多人拥挤场景下可能导致 grounding 模糊。

---

### 未来工作方向
- 提升时间定位精度（temporal grounding）
- 引入辅助验证奖励（auxiliary verification rewards）以增强训练闭环
- 扩展 EEI 接口支持更多验证器（如 keyframe retrieval、speaker localization）
- 探索 reward generalizability 至其他可信推理任务（如医疗诊断、法律推理）

---

> 🔗 **代码地址**：[https://github.com/eliot127825-rgb/AffectOmni_nobody](https://github.com/eliot127825-rgb/AffectOmni_nobody)

</details>

---

### 13. [Simple Actors and Deep Critics for Scalable Reinforcement Learning](https://arxiv.org/abs/2608.26659)

**Authors**: Guhyeon Kang, Jaehwi Lee, Minhae Kwon  
**Category**: cs.LG  
**Published**: 2026-08-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.26659v1  

#### Abstract
Recent progress in offline reinforcement learning (RL) has been driven by expressive generative actors such as diffusion and flow-matching policies, which capture multimodal behavior in offline datasets. However, these actors require multiple denoising or integration steps per action and thus incur ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Simple Actors and Deep Critics for Scalable Reinforcement Learning

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **offline RL** 方法（如 diffusion 和 flow-matching）依赖于高表达能力的生成式 **actor** 来建模多模态行为策略，但这类模型在推理时需要多次去噪或积分步骤，导致 **inference latency 高**，不利于部署在资源受限的边缘设备上。

然而，作者指出一个被忽视的事实：在 actor-critic 框架中，**critic 只用于训练，而 actor 才是部署时每一步都要运行的组件**。因此，将计算容量（capacity）投入到 actor 上会带来持续的推理开销，而投入到 critic 上则只产生一次性的训练成本。

### 提出的新方法与思路
作者提出 **LAC (Light Actor, deep Critic)**，其核心思想是：
> **将模型容量从 actor 转移到 critic** —— 使用一个轻量级、单步前向传播的确定性 actor，配合一个深度、强表达力的 critic。

这种“非对称容量分配”（asymmetric capacity allocation）的设计，使得在保持高性能的同时，显著降低推理延迟。

### 相比现有方法的优势
- **推理效率高**：相比 multi-step 生成式 actor（如 diffusion/flow），LAC 推理速度提升高达 **4×**。
- **无需蒸馏**：达到与 one-step distilled policy 相当的低延迟，但无需额外的 distillation 训练阶段。
- **性能不妥协**：在 OGBench 上匹配甚至超越最强的 diffusion 和 flow-matching 基线。
- **critic 设计可迁移**：提出的 deep critic 组件可作为“即插即用”模块，提升其他方法（包括 diffusion/flow）的性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **OGBench**：一个近期推出的、用于压力测试 offline RL 方法的基准套件，包含 7 个高维、长视野的连续控制任务，分为三类：
  - **Ant navigation**（8-DoF）：`ant-large`, `ant-giant`
  - **Humanoid navigation**（21-DoF）：`hum-medium`, `hum-large`
  - **Manipulation**：`scene`, `puzzle-3x3`, `puzzle-4x4`

每个环境有 5 个目标变体（task1–task5），共 35 个任务。

### 实验设置和评估指标
- **评估指标**：
  - **Success Rate (%)**：二值成功率，按 OGBench 协议在最后三个评估周期取平均。
  - **Inference Latency (ms)**：在单张 NVIDIA RTX 3090 上测量单样本动作预测的 wall-clock 时间（batch size = 1）。
- **随机种子**：所有结果基于 4 个随机种子取平均。
- **LAC 配置**：
  - **Critic**：32 层 ResMLP + categorical cross-entropy loss + n-step (n=4) bootstrap
  - **Actor**：轻量级 deterministic MLP
    - `LAC-S`：小 actor `[256]×2` (~0.13M 参数)
    - `LAC-L`：大 actor `[512]×4`，与 flow-based 基线 actor 大小对齐

### 基线方法对比
涵盖三类 actor parametrization：
- **Gaussian/Deterministic**：BC, IQL, ReBRAC, TD3+BC
- **Diffusion-based**：IDQL, SRPO, CAC
- **Flow-matching**：FAWAC, FBRAC, IFQL, FQL

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）
| Environment       | Best Baseline | LAC-S | LAC-L |
|-------------------|---------------|-------|-------|
| ant-large         | 79±3 (FQL)    | 92±5  | **96±1** |
| ant-giant         | 9±6 (FQL)     | 35±23 | **79±13** |
| hum-medium        | 60±14 (IFQL)  | **79±10** | 76±38 |
| hum-large         | 11±2 (FBRAC)  | **50±18** | **69±15** |
| scene             | 56±2 (FQL)    | 29±37 | 41±39 |
| puzzle-3x3        | 30±1 (FQL)    | 62±17 | **95±1** |
| puzzle-4x4        | 25±5 (FBRAC)  | 15±9  | **32±19** |
| **Average**       | **36**        | **52**  | **70** |

- **LAC-L 是整体最优方法**，在多个环境中大幅领先。
- **LAC-S 在多数任务上优于所有生成式 actor**，尤其在 `ant` 和 `puzzle` 系列表现突出。

### 与基线方法的对比结果
- **性能方面**：
  - LAC 匹配甚至超越最强的 **diffusion** 和 **flow-matching** 基线（如 FQL, IFQL）。
  - 尤其在 `humanoid` 这种高维动作空间、多模态行为的任务上，LAC 表现优异，说明 **strong critic 可弥补 simple actor 的表达力不足**。
- **推理效率方面**（Table 3）：
  - **LAC-S 推理延迟最低（0.24 ms）**，与 IQL、ReBRAC 等单步方法相当。
  - 相比 multi-step 生成式 actor（如 IDQL: 0.95ms, IFQL: 0.64ms），LAC 实现 **2–4× 的推理加速**。
  - 与 one-step distilled 方法（如 FQL: 0.27ms）相比，LAC 优势微弱但存在。

### 消融实验结果（Table 2, Figure 3–5）

#### 组件消融（Table 2）
| Configuration               | hum-large | ant-giant | puzzle-4x4 | Avg |
|-----------------------------|-----------|-----------|------------|-----|
| **LAC (full)**              | 79±6      | 86±3      | 55±6       | 73  |
| **-n-step (1-step TD)**     | 42±3      | 38±15     | 18±2       | 33  |
| **-Categorical (MSE)**      | 7±12      | 72±12     | 32±15      | 37  |
| **-Both (1-step + MSE)**    | 15±5      | 1±1       | 12±3       | 9   |

- 移除任一组件均导致性能严重下降，验证了三者缺一不可。

#### 深度扩展分析（Figure 2–3）
- **Plain MLP critic**：无论深度如何，都无法学习有效策略。
- **仅加 ResMLP**：性能停滞，且高方差。
- **完整 LAC**：随深度增加性能稳定上升，在 32 层达到峰值。
- **n-step 是深度扩展的关键**：只有 n≥2 时，深度增加才不会导致性能退化。

#### Actor 容量饱和（Figure 5）
- 固定 deep critic 后，**actor 参数超过 ~10⁵ 后性能即饱和**。
- 说明在 strong critic 支持下，**actor 的角色主要是 amortized inference**，进一步增大 actor 收益极小。

---

## 4. 关键结论和发现

### 主要发现
1. **非对称容量分配是有效的设计原则**：
   - 将计算预算从 **inference-time**（actor）转移到 **training-time**（critic）是实现高效 offline RL 的合理路径。
2. **deep critic 的三大失败模式必须协同解决**：
   - **(F1) Optimization Failure** → 通过 **ResMLP + LayerNorm** 解决梯度传播问题。
   - **(F2) Bootstrap-noise Amplification** → 通过 **n-step bootstrap** 减少自举噪声累积。
   - **(F3) Value-range Drift** → 通过 **categorical cross-entropy loss** 限制 Q 值范围，防止 mid-training collapse。
3. **lightweight deterministic actor 可以足够强大**：
   - 当由一个 well-trained deep critic 指导时，简单 actor 能达到与复杂生成式 actor 相当的性能。
4. **critic 设计具有通用性**：
   - LAC 的 critic 组件可作为“drop-in”模块，显著提升 TD3+BC、IDQL、FQL 等各类方法的性能（Table 4，增益 +24 至 +59 分）。

### 方法的局限性
- **评估范围有限**：目前仅在 state-based 的 OGBench 环境中验证，未扩展到 pixel-based 输入或真实机器人部署。
- **未探索 actor 的多样性**：虽然证明了 deterministic actor 的有效性，但未尝试更复杂的 actor 结构（如混合模型）与 deep critic 的组合。

### 未来工作方向
- 将 LAC 框架扩展到 **vision-based** 或 **language-conditioned** RL 任务。
- 探索在 **real-world robotics** 场景中的部署效果。
- 研究如何进一步优化 critic 的训练效率（因 deep critic 增加了训练成本）。
- 探索 actor-critic 之间的容量分配比例的自动化搜索机制。

---

> ✅ **一句话总结**：  
> LAC 通过“**深 critic + 浅 actor**”的非对称设计，在 OGBench 上实现了与最强生成式 actor 相当的性能，同时将推理延迟降低至其 1/4，并首次系统论证了 critic-side capacity 是可扩展、高效的 offline RL 发展方向。

</details>

---

### 14. [A Layer Importance Metric for Quantization Accounting for the Speed-Quality Trade-off in Autoregressive Models](https://arxiv.org/abs/2608.26926)

**Authors**: Artem Safronov  
**Category**: cs.LG  
**Published**: 2026-08-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.26926v1  

#### Abstract
Small language models (sLLMs) are nowadays hosted on devices with limited memory and computational budget. In an autoregressive setup, inference is memory-bandwidth bound: uniform quantization is often detrimental to such models, since their architecture has limited redundancies and only a few layer...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Layer Importance Metric for Quantization Accounting for the Speed-Quality Trade-off in Autoregressive Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
小型大语言模型（**sLLM**, small Large Language Models）在边缘设备上部署时面临严重的内存带宽瓶颈。由于其结构紧凑、冗余少，传统的**均匀量化**（uniform quantization）会显著损害生成质量。同时，不同层对精度损失的敏感度差异巨大，盲目量化会导致“高代价低收益”——即某些关键层的小幅压缩引发严重性能下降。

本文旨在解决以下核心问题：  
> **如何在不牺牲生成质量的前提下，最大化推理加速？**

具体来说，就是识别出那些既能带来显著速度提升、又对量化鲁棒的模型组件，实现**异构量化**（heterogeneous quantization）中的最优资源分配。

---

### 🚀 提出的新方法与创新思路

作者提出了一种**复合重要性度量指标** `score(α)`，用于评估神经网络各组件在量化过程中的优先级。该指标融合了两个正交维度：

1. **质量保留能力（Quality Preservation）**  
   使用 **SQNR**（Signal-to-Quantization-Noise Ratio）衡量某一层在量化后输出信号的信息损失程度。通过 **Sensitivity-Aware Post-Training Quantization (SA-PTQ)** 模拟量化并计算 SQNR，得到归一化的 `Qk` 分数。

2. **速度增益潜力（Speed Gain Potential）**  
   基于 **roofline model** 构建分析型延迟预测模型，无需实际运行模型即可估算将某层从 F16 降至 Q8 所能带来的吞吐提升，得到归一化的 `Sk` 分数。

最终的综合评分为：
$$
\text{score}(\alpha) = (1-\alpha) \sum_{k} S_k + \alpha \sum_{k} Q_k
$$
其中 $\alpha \in [0,1]$ 是可调参数，允许用户根据任务需求在“追求速度”或“保障质量”之间灵活权衡。

#### 创新亮点：
- **完全分析性方法**：无需训练、无需多次实测，仅需一次架构剖析即可完成评估。
- **通用性强**：适用于任意粒度（block-level / layer-level / projection-level），如 FFN、Embedding、单个 transformer 层等。
- **解耦设计**：将“信息保留”与“硬件加速”分离建模，再统一评分，逻辑清晰且可解释。

---

### 🔍 相比现有方法的优势

| 方法 | 缺陷 | 本论文优势 |
|------|------|------------|
| **CARVQ** | 需引入 lookup table 和非线性变换，难以集成到通用工具链（如 llama.cpp） | 使用标准 scalar quantization，兼容性强 |
| **HAPM** | 要求稀疏化结构和专用推理引擎 | 不改变模型结构，直接利用现有加速器 |
| **IMPQ** | 依赖 Shapley-value 近似，计算开销极大（数十亿参数下不可行） | 仅需轻量级 probing，速度快、成本低 |
| **QRazor** | 修改 Attention 结构，适用范围受限；基于启发式振幅排序 | 聚焦 FFN 和 Embedding；使用能量基础的 SQNR 更准确 |

> ✅ 总结：相比依赖昂贵搜索、特定硬件或复杂结构修改的方法，本文提供了一个**快速、通用、直观**的量化决策框架。

---

## 2. 核心实验方法和设置

### 🧪 实验平台与模型
- **主模型**：`Gemma 3 1B-it`（26 层）
- **辅助验证模型**：`LLaMA 3.2 1B`, `Qwen 2.5 1.5B`
- **硬件环境**：Google Colab 上的 NVIDIA T4 GPU（320 GB/s 内存带宽）
- **工具链**：`llama.cpp`, `llama-bench`, `LLM-Viewer`

> ⚠️ 注意：所有实验均未进行微调（no fine-tuning），属于纯 Post-Training Quantization（PTQ）场景。

---

### 📊 评估指标

| 类别 | 指标 | 描述 |
|------|------|------|
| **速度相关** | `tokens/sec`, `latency/ms/token` | 实际推理吞吐与延迟 |
| **质量相关** | `SQNR (dB)` | 量化前后输出向量的能量比，越高越好 |
| | `Top-1 Match with F16` | 与全精度模型预测是否一致 |
| | `Δgap` | top-1 与 top-2 logit 差距的变化，反映置信度稳定性 |
| | `Weight MAE` | 权重矩阵的平均绝对误差 |
| **综合指标** | `score(α)` | 归一化后的速度分 $S_k$ 与质量分 $Q_k$ 加权和 |

---

### 🔬 实验设置

#### （1）组件选择依据
通过对 Gemma 3 1B 进行系统级 profiling，测量不同模块（FFN / Attention / Embedding）在 Q8 下对整体延迟的影响，发现：
- **FFN** 和 **Embedding** 是主要延迟来源；
- Attention 层即使保持 F16 对总延迟影响较小。

因此选定这两个模块作为重点优化目标。

#### （2）量化配置测试
- **Embedding 测试配置**：Q16-ch, Q8-ch, Q4-ch, Q8-tensor, Q4-tensor
- **FFN 测试配置**：同上，分别作用于 `Wgate`, `Wup`, `Wdown` 投影
- **上下文长度**：10, 50, 100 tokens，涵盖短查询与长序列场景

#### （3）基线对比方法
- 全模型均匀量化（Uniform Q8）
- 各模块单独反量化（如 “Q8 ffn_all=F16” 表示仅 FFN 保持 F16）
- 与理论 roofline 预测值比较，验证建模准确性

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）延迟贡献分析（Table 1）
| 配置（FFN/Attn/Emb） | 短上下文速度 (tok/s) |
|------------------------|-----------------------|
| Q8 Q8 Q8（全量化）     | **95.3**             |
| Q8 ffn_all=F16（FFN 不量化） | 72.3（↓24%）        |
| Q8 attn_out=F16（Attention 不量化） | 93.0（仅 ↓2.4%） |

> 💡 发现：**FFN 是否量化对性能影响最大**，而 Attention 几乎无影响。

---

#### （2）SQNR 质量得分（归一化前）

| 组件 | 平均 SQNR (Q8-ch) | 参考值 (Q16) |
|------|--------------------|--------------|
| **Embedding** | 45.61 dB | 93.82 dB |
| **FFN (Wdown)** | 29.70 dB | 77.95 dB |

> ❗ FFN 明显更敏感，尤其 `Wdown` 投影。

---

#### （3）归一化分数汇总（公式 11 & 12）

| 组件 | 质量分 $Q_k$ | 速度分 $S_k$ |
|------|---------------|---------------|
| Embedding | **0.819** | 0.236 |
| FFN       | **0.796** | 0.261 |

> ✅ 两者均具备较高质量容忍度，且有可观加速空间。

---

#### （4）速度预测准确性（Table 4）

| 模型 | 预测加速比 | 实际加速比 | 误差 |
|------|-------------|------------|------|
| Gemma 3 1B | 1.183 | 1.175 | **0.7%** |
| LLaMA 3.2 1B | 1.278 | 1.236 | 3.3% |
| Qwen 2.5 1.5B | 1.259 | 1.344 | 6.3% |
| **平均误差** | —— | —— | **≈4%** |

> ✅ 表明 roofline 模型具有高度预测能力，可用于指导量化策略。

---

#### （5）综合评分（$\alpha=0.5$）

对于 Gemma 3 1B 的 Q8 配置：
$$
\text{score}(0.5) = 0.5 \times (0.236 + 0.261) + 0.5 \times (0.819 + 0.796) = \boxed{0.497}
$$
其中：
- 速度部分贡献：0.187
- 质量部分贡献：0.808

---

#### （6）消融实验：Layer-wise 异构量化潜力
设想一种配置：
- 中间 10 层（最敏感）：保持 Q8
- 前后各 8 层（较鲁棒）：降为 Q4

理论预测可额外节省 2.54ms，总加速比达 **×1.254**（vs 均匀 Q8 的 ×1.183），**相对提升约 6%**。

> ⚠️ 注：尚未实测，因缺乏支持细粒度控制的推理框架（如 ExLlamaV3 或 TensorRT-LLM）。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **FFN 与 Embedding 是 sLLM 推理加速的关键瓶颈**，应优先考虑量化。
2. **Attention 层对整体延迟贡献小**，且对低精度敏感，建议保留高精度。
3. **SQNR 是有效的质量代理指标**，能稳定反映不同量化方案下的信息损失。
4. **roofline model 可精准预测加速效果**（平均误差 <4%），无需反复 benchmark。
5. **提出的 `score(α)` 指标实现了速度与质量的可调平衡**，支持灵活决策。

---

### ⚠️ 方法的局限性

1. **依赖静态权重假设**：未考虑动态激活分布变化对量化误差的影响。
2. **未覆盖 Activation Quantization**：当前仅关注 weights，activations 仍为 FP16。
3. **未实现在线异构推理验证**：layer-wise Q4/Q8 混合配置尚缺工程实现支持。
4. **SQNR 与人类感知质量的相关性有待进一步验证**。

---

### 🔮 未来工作方向

1. 将方法扩展至 **activation-aware quantization**，构建完整的混合精度方案。
2. 开发支持 **per-layer bit-width 控制** 的推理后端（如集成进 llama.cpp）。
3. 探索自动化搜索算法结合 `score(α)` 指标，实现端到端量化配置推荐。
4. 在更多模型架构（如 Mistral、Phi）上验证泛化能力。
5. 引入用户反馈机制，动态调整 $\alpha$ 参数以适应不同应用场景（如聊天 vs 编码）。

---

## ✅ 总结

本文提出了一种新颖、高效、可解释的层重要性度量方法，用于指导 sLLM 的异构量化。它将复杂的试错过程转化为一个**数学优化问题**，通过融合 **SQNR 质量评分** 与 **roofline 速度预测**，构建了一个可调节的综合指标 `score(α)`。

实验表明，该方法能在极低计算成本下实现接近真实测量的预测精度，并显著优于传统均匀量化及其他先进方法。它是迈向“智能量化”（intelligent quantization）的重要一步，为边缘侧高效部署 sLLM 提供了坚实的理论与实践基础。

</details>

---

### 15. [ClusterAttention: A training-free speedup of bidirectional attention](https://arxiv.org/abs/2608.26965)

**Authors**: Kasper Nordenram, Amelie Dittmann  
**Category**: cs.LG  
**Published**: 2026-08-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.26965v1  

#### Abstract
This paper introduces ClusterAttention, a general training-free speedup of bidirectional attention layers. Existing sparse attention methods either rely on structure in the input, such as order in language or spatial proximity in images, or use slow clustering processes amortized over several forwar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ClusterAttention: A training-free speedup of bidirectional attention**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
- **双向注意力（bidirectional attention）** 在 Transformer 模型中计算复杂度为 $O(n^2)$，在高 token 数场景（如高分辨率图像、视频生成、基因组数据、表格数据）下效率低下。
- 现有稀疏注意力（sparse attention）方法通常依赖输入结构（如文本顺序、空间邻近性），或需要离线聚类、多步摊销，难以应用于**无结构输入**（如表格数据）或**单次前向传播**场景。

### **提出了什么新方法或新思路**
提出 **ClusterAttention**，一种**无需训练**（training-free）、适用于任意 bidirectional attention 层的通用加速方法，其核心流程如下：
1. **Clustering**：对每个 attention head 中的 keys 和 queries 分别进行快速递归聚类。
2. **Assignment**：基于聚类质心打分，选择关键的 key-clusters 供 query-clusters 注意。
3. **Sparse Attention**：仅在选中的块上执行 block-sparse attention。
4. **Compensation**：引入 **striped mean-compensation (SMC)** 补偿被忽略的 cluster，通过其 key-value 质心参与计算以减少误差。

**关键技术创新**：
- **快速递归聚类（recursive splitting）**：沿主成分方向递归分割，确保聚类大小为 2 的幂，适配 GPU tile 并行，保持低延迟。
- **几何感知变换（Geometry-aware transforms）**：
  - Key 聚类使用 $M = QQ^\top / n$ 度量空间，使相似 keys 对所有 queries 产生相近 logits。
  - Query 聚类根据任务设计不同度量：自适应选择时使用中心化 keys 的协方差；top-k 排名时使用 sign-based ranking 表示。
- **SMC 误差理论分析**：推导出稀疏注意力输出误差表达式，证明 SMC 可使误差随 cluster 紧密度提升而减小。

### **相比现有方法的优势**
| 特性 | ClusterAttention | SpargeAttn | SVOO | AdaCluster |
|------|------------------|-----------|------|------------|
| **Training-free** | ✅ | ✅ | ✅ | ✅ |
| **适用于无结构输入** | ✅ | ❌（依赖空间/顺序结构） | ✅ | ✅ |
| **单次前向传播适用** | ✅ | ✅ | ❌（需每 20 步重聚类） | ⚠️（未明确） |
| **GPU 友好（固定 cluster size）** | ✅ | ✅ | ❌（ragged clusters） | ❌ |
| **理论误差补偿机制** | ✅（SMC） | ❌ | ❌ | ❌ |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
1. **DINOv2-L**（Vision Transformer）：
   - 数据集：DIV8K 高分辨率图像（短边 ≥ 5306 像素）
   - 分辨率：2072×2072（21,904 tokens）、3500×3500（62,500 tokens）、5306×5306（143,641 tokens）
2. **TabPFN-3**（表格数据 Transformer）：
   - 数据集：TALENT benchmark 中最大的 6 个分类数据集（如 `Rain_in_Australia`, `CDC_Diabetes`）
3. **Wan 2.1-14B T2V**（视频生成扩散模型）：
   - 提示词：来自 OpenSora 1.0 的 12 个 prompt，选取前 5 个生成视频

### **实验设置和评估指标**
| 任务 | 评估指标 | 设置细节 |
|------|----------|----------|
| **DINOv2** | 表征失真（representation distortion）<br>→ 所有 patch 的 embedding 与 dense 的平均 cosine similarity<br>→ Latency（ms） | Pareto front：不同 sparsity 下的 accuracy-latency 权衡<br>Top-k 方法：attend to {5%, 10%, ..., 100%} keys<br>Adaptive 方法：target {60%, 80%, ..., 100%} attention mass recall |
| **TabPFN-3** | 下游任务准确率（accuracy）<br>→ 归一化为 dense 的百分比<br>→ Latency（relative time） | 单次前向传播预测<br>ClusterAttention*：top-k + SMC，扫过 {1%, 5%, ..., 100%} keys |
| **Wan T2V** | 视频质量指标：<br>- PSNR（越高越好）<br>- SSIM（越高越好）<br>- LPIPS（越低越好）<br>→ 与 dense 输出比较<br>→ Speedup | 使用 H200 GPU<br>ClusterAttention* 固定 attend to 20% clusters<br>前 10 diffusion steps 和第一层用 dense |

### **基线方法对比**
- **SpargeAttn**：基于 row-major 或 space-filling curve 的结构化聚类，支持 top-k 和 adaptive。
- **SVOO**：专为视频生成设计，offline profiling + online QK co-clustering，每 20 步重聚类。
- **Random Clusters**：随机聚类 + centroid routing，作为简单 baseline。
- **Dense / SageAttention2**：全密集注意力及量化加速版本。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
#### **TabPFN-3（表格数据）**
- **速度提升**：2x ~ 6x 加速（随数据规模增大而提升）
- **精度保留**：仅关注 10% clusters 时，**相对准确率 >99%**
- **Pareto 最优**：在 5/6 数据集上，ClusterAttention* 支配其他方法

#### **Wan 2.1-14B T2V（视频生成）**
| 方法 | 平均 PSNR ↑ | 平均 SSIM ↑ | 平均 LPIPS ↓ | 平均 Speedup |
|------|-------------|-------------|--------------|---------------|
| **ClusterAttention*** | **24.61** | **0.836** | **0.0978** | **1.79x** |
| **SVOO** | 23.97 | 0.826 | 0.1128 | 1.43x |
- **质量更优**：PSNR ↑0.64，SSIM ↑0.01，LPIPS ↓0.015
- **加速更强**：1.79x vs 1.43x（无 offline calibration）

#### **DINOv2（高分辨率图像）**
- 在 3500×3500 和 5306×5306 分辨率下，ClusterAttention* 在 0.99 cosine similarity 附近实现 **Pareto 主导**
- 优于 SpargeAttn 和 SVOO，尤其在高 token 数场景

### **消融实验结果**
#### **(1) Transform 是否有效？**
- **显著提升性能**：在 top-k 和 adaptive 设置下，使用几何变换的聚类几乎在所有情况下都 **Pareto 更优**
- 例外：极小图像 + 极高稀疏度时，变换开销占主导

#### **(2) Striped Mean-Compensation (SMC) 是否有效？**
- **top-k 场景**：SMC 带来**巨大质量提升**，几乎完全 Pareto 支配无补偿版本
- **adaptive 场景**：提升较小，且在低分辨率下可能略差（因 adaptive 已偏向高质量 heads）

#### **(3) top-k vs adaptive selection**
- **top-k + SMC 是最优组合**：在所有分辨率下均 Pareto 支配 adaptive
- 差距随图像尺寸增大而扩大
- 无补偿时两者表现接近

---

## 4. **关键结论和发现**

### **主要发现**
1. **ClusterAttention 是首个成功应用于无结构输入 + 单次前向传播的 training-free 稀疏注意力方法**。
2. **SMC 显著降低稀疏注意力误差**，尤其在 top-k 设置下效果显著，验证了论文提出的误差理论。
3. **几何感知聚类（geometry-aware transforms）能生成更有意义的 clusters**，优于 naive PCA 或随机聚类。
4. **固定大小 cluster + GPU tile 优化** 使得 block-sparse attention 的每交互延迟接近 dense attention。
5. **在多个领域（vision, tabular, video）均实现优于 SVOO 的质量与速度权衡**，即使后者是专为视频生成设计。

### **方法的局限性**
- **聚类开销在低 token 数时显著**：特征分解（eigen-decomposition）成为瓶颈，限制了在小输入上的优势。
- **SMC 在极高 token 数时成为主要延迟来源**：当前实现较慢，有优化空间。
- **固定 cluster size 可能非最优**：稠密区域可用大 cluster，稀疏区域需小 cluster，但当前框架未支持变长。
- **未探索非线性变换**：如 softmax-aware clustering 可进一步提升 cluster 质量，但计算代价高。

### **未来工作方向**
1. **加速聚类**：
   - 优化 eigen-decomposition 实现
   - 使用非对角化递归分裂避免特征分解
2. **改进补偿机制**：
   - 引入 MuSe 等高阶补偿方法
   - 动态决定是否应用 SMC
3. **提升聚类质量**：
   - 支持变长 cluster sizes
   - 探索非线性变换（如 landmark-based softmax projection）
4. **扩展至 GQA/MQA**：
   - 设计适用于 grouped-query attention 的 key/query 聚类策略
5. **更广泛评估**：
   - 更多模型、数据集、任务
   - 探索对训练过程的影响

> **总结**：ClusterAttention 提供了一种通用、高效、无需训练的稀疏注意力方案，在多个高 token 数场景中实现了显著加速并保持高精度，是稀疏注意力领域的重要进展。

</details>

---

### 16. [PICasso: An AI-Enabled Design Framework for Autonomous Optimization of Silicon Photonic Devices](https://arxiv.org/abs/2608.26113)

**Authors**: Deepak Vungarala, Deniz Najafi, Abdulrahman Aljoudi, Zahra Ghanaatian, Navid Khoshavi, Gourav Datta, Arman Roohi, Mahdi Nikdast, Shaahin Angizi  
**Category**: cs.AI  
**Published**: 2026-08-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.26113v1  

#### Abstract
We present PICasso, an AI-assisted framework for automated synthesis, verification, and optimization of photonic integrated circuits (PICs) from natural-language specifications. PICasso couples a structured NL -> YAML -> GDS generation pipeline with PDK aware knowledge injection, automated placement...

---

### 17. [TreeGraft: Adaptive Multi-Drafter Grafting for Tree-Based Speculative Decoding](https://arxiv.org/abs/2608.26112)

**Authors**: Jiaming Fan, Daming Cao, Canchen Huang, Jiale Fu, Jin Zhang, Junjie Gao, Kai Yang, Xiangzhong Luo, Xu Yang  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.26112v1  

#### Abstract
Speculative decoding accelerates large language model inference through a draft-then-verify paradigm. Building on this, tree-structured methods improve inference by organizing proposals into multiple candidate paths, increasing the accepted length. However, existing tree-structured methods use a sin...

---

### 18. [Pruning Binarized Neural Networks: A Dedicated Framework and Globally Weighted Algorithms](https://arxiv.org/abs/2608.26233)

**Authors**: Roan Rubiales, Jean Pierre David  
**Category**: cs.LG  
**Published**: 2026-08-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.26233v1  

#### Abstract
Extreme compression of deep neural networks, up to full binarization, dramatically reduces memory footprint and arithmetic complexity, facilitating deployment on constrained edge hardware with field-programmable gate arrays (FPGAs) and microcontrollers. Although combining binarization with pruning p...

---

### 19. [GRAIN: Bridging Name and Narrative Shifts in Real-World Graph Reasoning through Invariance-Rewarded Agentic RL](https://arxiv.org/abs/2608.27142)

**Authors**: Zike Yuan, Han Zhang, Jianzhi Yan, Le Liu, Cai Ke, Huozhi Zhou, Jian Xie, Jiran Yin, Yukun Cao, Yue Yu, Hui Wang, Ming Liu, Bing Qin  
**Category**: cs.AI  
**Published**: 2026-08-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.27142v1  

#### Abstract
Despite their potential in standardized graph tasks, Large Language Models (LLMs) remain brittle to real-world shifts in node identifiers and task formulation. While deterministic graph tools are invariant to such shifts, extracting topological structures from noisy text is highly fragile for LLMs, ...

---

### 20. [Recipes for Steering and Scaling LLMs via Sampling](https://arxiv.org/abs/2608.26120)

**Authors**: Jiajun He, Zongyu Guo, Jos\'e Miguel Hern\'andez-Lobato, Yuanqi Du  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.26120v1  

#### Abstract
Large Language Models (LLMs) are probabilistic models, typically defined by an autoregressive factorization. While recent work has begun to study richer target distributions beyond the base model, the sampling strategies remain highly inefficient. In this paper, we present a flexible and theoretical...

---

### 21. [Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO](https://arxiv.org/abs/2608.27351)

**Authors**: Yunpeng Ba, Zhi Zheng, Yue Xie, Jiaqing Li, Xialiang Tong, Tao Zhong, Mingxuan Yuan, Zhichao Lu, Xuyang Wu, Zhenkun Wang  
**Category**: cs.LG  
**Published**: 2026-08-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.27351v1  

#### Abstract
Evolution Strategies (ES) have recently emerged as a memory-efficient post-training paradigm for LLM reasoning. However, the optimization behavior of ES remains understudied, making it hard to define its advantage scope compared to mainstream post-training paradigms (e.g., Group Relative Policy Opti...

---

### 22. [EduRiskX: A Neuro-Symbolic Framework with F-Logic Reasoning for Early Academic Risk Prediction](https://arxiv.org/abs/2608.26107)

**Authors**: Yu Fu, Yongqi Kang, Yong Zhao, Rongfang Bie  
**Category**: cs.AI  
**Published**: 2026-08-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.26107v1  

#### Abstract
Predicting students' academic risk in online education is crucial for enabling timely interventions that can improve retention and learning outcomes. However, existing models often suffer from limited early detection capability and insufficient interpretability, leading to a "black-box" trust crisis...

---

### 23. [Counterfactual Bias Testing for Application Tracking System](https://arxiv.org/abs/2608.26899)

**Authors**: Sai Yashwant, Shruti Bansal, Anurag Dubey, Samaroha Chatterjee, Satyam Kumar, Shreyash Gupta, Gantala Thulsiram  
**Category**: cs.AI  
**Published**: 2026-08-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.26899v1  

#### Abstract
Automated candidate-job matching systems are increasingly classified as high-risk AI under emerging regulation, yet auditing them for demographic bias is expensive: classical correspondence-audit studies require hand-crafted resumes and manual submission, which does not scale to fast pipeline retrai...

---

### 24. [TelecomGPT-R1: A Unified Open-Source Reasoner for the Telecom Stack](https://arxiv.org/abs/2608.26126)

**Authors**: Bohao Wang, Chenwei Wu, Haoyu Li, Hang Zou, Yu Tian, Lina Bariah, Li Wei, Chongwen Huang, Yongliang Shen, Zhaoyang Zhang, Merouane Debbah  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.26126v1  

#### Abstract
Telecommunications is a high-leverage domain for large language model (LLM)-based reasoning because routine engineering workflows require joint grounding in normative specifications, operational telemetry, vendor-specific fault evidence, and exact RF/network calculations. However, current LLM integr...

---

### 25. [CARE: Causally-Aligned Reasoning Exploration for Medical Large Language Models](https://arxiv.org/abs/2608.26147)

**Authors**: Yucheng Zhou, Peng Luo, Qianning Wang, Chengzhong Xu, Jianbing Shen  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.26147v1  

#### Abstract
Large Language Models (LLMs) have shown strong potential for medical reasoning, yet the scarcity and cost of expert-annotated data constrain their progress. While reinforcement learning offers a scalable alternative, standard outcome-based methods in medicine often suffer from autoregressive credit ...

---

### 26. [Sycophancy Suppression Can Impair Rational Updating: Anti-Sycophancy Should Preserve the Ability to Update](https://arxiv.org/abs/2608.26511)

**Authors**: Huanhuan Ma, Henry Peng Zou, Chengze Li, Enze Ma, Yunyue Su, Philip S. Yu  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.26511v1  

#### Abstract
Large language models often exhibit sycophancy, revising their answers to align with users when users push back. Such answer flips, however, can arise from different causes. One possibility is that the model simply aligns with the user's feedback in order to satisfy them. Another is that the feedbac...

---

### 27. [Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation and Meta-Evaluation](https://arxiv.org/abs/2608.26638)

**Authors**: Mingqi Gao, Anthony Sicilia, Weiyan Shi  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.26638v1  

#### Abstract
Across various non-verifiable tasks, human evaluation is reliable but expensive, while automatic metrics are more scalable but often biased. Building on prediction-powered inference (PPI), we propose prediction-powered evaluation, a framework that combines limited human judgments with large-scale au...

---

### 28. [Information-Guided Frontier Decoding: Contextual Utility-Driven Commitment in dMLLMs](https://arxiv.org/abs/2608.26641)

**Authors**: Xingyou Fang, Jingxing Zhong, Xiaosong Yuan, Xiaofeng Zhang  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.26641v1  

#### Abstract
Decoding quality in diffusion multimodal language models (dMLLMs) depends heavily on the order in which masked tokens are committed. Existing confidence-based strategies prioritize locally easy tokens, but confidence does not necessarily reflect contextual usefulness. As a result, structurally easy ...

---

### 29. [Boosting LLM Exploration via Weak-Model Guidance in RLVR](https://arxiv.org/abs/2608.27420)

**Authors**: Xingyu Shen, Huishuai Zhang, Peng Li, Yinchun Wang, Dongyan Zhao  
**Category**: cs.CL  
**Published**: 2026-08-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.27420v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) significantly improves LLM reasoning but often causes a drop in policy entropy, leading to narrowed reasoning coverage and degraded pass@$k$ for large $k$. While existing methods mitigate this entropy collapse through algorithmic regularizations,...

---

### 30. [SAGE: Variate-Wise Semantic Augmentation for Vision-Language Time Series Forecasting](https://arxiv.org/abs/2608.26829)

**Authors**: Haizhao Fan, Xinyi Le  
**Category**: cs.LG  
**Published**: 2026-08-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.26829v1  

#### Abstract
Time series forecasting models operate on raw numerical sequences, lacking the semantic knowledge that domain experts implicitly leverage, such as the physical meaning of each variable, its statistical behavior, and its temporal dynamics. Recent efforts to bridge this gap fall into two camps. Some r...

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

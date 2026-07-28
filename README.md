# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-28 08:09:27 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Beyond Block Boundaries: Multi-Block Editing for Diffusion Large Language Models](https://arxiv.org/abs/2607.22663)

**Authors**: Xingyu Mou, Zijin Huang, Tianze Zhang, Yuxin Ma, Lanning Wei, Zengfeng Huang, Da Zheng, Lun Du  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2607.22663v1  

#### Abstract
Block diffusion has emerged as the dominant paradigm for scaling discrete diffusion language models (dLLMs), because decoding text in fixed-size blocks preserves parallel generation within each block while keeping the quadratic attention cost tractable. However, this efficiency comes with a structur...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Block Boundaries: Multi-Block Editing for Diffusion Large Language Models

## 1. 论文的主要贡献和创新点

### 解决的问题
论文识别并系统性地提出了 **block boundary problem**（块边界问题）这一结构性缺陷，该问题存在于当前主流的 **block diffusion** 范式中。具体表现为：
- 在固定大小的 block 内进行并行去噪时，靠近 block 末尾的 token 缺乏对后续跨 block 上下文的访问能力，导致预测不确定性增加。
- 一旦一个 block 被“提交”（committed），其内部的错误预测将作为不可逆的上下文传递给后续所有 block，造成错误传播。

### 提出的新方法：Multi-Block Editing (MBE)
为解决上述问题，论文提出 **Multi-Block Editing (MBE)**，一种结合推理时解码算法与训练策略的跨 block 编辑框架，核心思想是将 block 视为调度单元而非硬性的修订边界。

#### 主要创新点：
1. **训练免费的 MBE 解码算法（Training-free MBE decoding）**
   - 在每个新 block 解码完成后，重新打开一个滑动窗口（默认 $K=2$），对选定的历史 block 和当前 block 执行联合前向传播（full attention）。
   - 利用新生成 block 中的上下文信息，直接对先前已提交 block 中的 token 进行 **Token-to-Token (T2T)** 编辑修正。
   - 引入 **熵值驱动的历史惊喜水印（entropy-based historical surprisal watermark）**，自适应选择最不确定的历史 block 进行编辑，提升效率。

2. **多块编辑监督微调（Multi-Block Edit SFT）**
   - 针对训练-推理不匹配问题（pretrained 模型未见过跨 block attention），设计了一种新的 SFT 策略。
   - 引入 **双向注意力掩码（bidirectional attention mask）**，新增 NF（noisy-to-future）和 CN（clean-to-noisy）连接，模拟推理时的编辑场景。
   - 采用 **课程学习（curriculum learning）**，逐步扩展编辑跨度 $K$（从 2 到 4），使模型渐进式学会长距离修正。

3. **系统级优化（Infrastructure Optimization）**
   - 扩展 **SGLang** 支持：
     - **多形状 CUDA Graph 池（multi-shape CUDA Graph pool）**：支持不同序列长度的前向传播高效执行。
     - **细粒度 KV Cache 控制**：通过 `disable_cache_update` 标志防止编辑过程污染标准 KV 缓存，并仅对修改过的 block 进行缓存刷新，保证一致性与效率。

### 相比现有方法的优势
| 方法 | 局限性 | MBE 的优势 |
|------|--------|------------|
| **ReMDM / SABER** | 仅在 token 级别 remask 当前 block 内容，无法跨越 block 边界 | 显式引入跨 block 上下文，实现历史 block 的直接编辑 |
| **RDD / DCD** | 仅在检测到停滞或高不确定性时被动合并 block，非例行化机制 | 将跨 block 编辑作为常规步骤，在每次 block 解码后主动执行 |
| **标准 block diffusion** | 解码即提交，无后期修正机制 | 将 block 从“不可逆提交点”转变为“可修订调度边界” |

---

## 2. 核心实验方法和设置

### 使用的数据集
共在 **13 个基准测试** 上评估，涵盖五大类任务：
- **Knowledge**: GPQA
- **Reasoning**: ZebraLogic, OCNLI, DROP
- **Coding**: MBPP+, HumanEval+, LiveCodeBench, Spider
- **Math**: AIME 2025, GSM-Plus, Omni-MATH
- **Agent**: BFCL v3, Nexus FCB

### 实验设置与评估指标
- **主模型**: LLaDA2.1-Mini ($B=32$)
- **评估模式**: zero-shot prompting, 单次生成 (n=1, pass@1)
- **最大生成长度**: 32768 tokens
- **解码参数**: `w_mask=0.5`, `w_edit=0.0`
- **MBE 参数**:
  - 训练免费 MBE: 固定 $K=2$, 最大编辑步数 $N_{\text{MBE}}=5$
  - MBE SFT: $K_{\text{max}}=4$, 自适应选择历史 block（top-25% 熵值）
- **基础设施**: 基于 SGLang 实现，启用多形状 CUDA Graph 和 KV cache 控制

### 基线方法对比
| 基线方法 | 简介 |
|---------|------|
| **Standard Decoding** | LLaDA2.1 默认解码方式，含 T2T 编辑 |
| **ReMDM** | remask 低置信度 token 并重去噪 |
| **SABER** | 自适应阈值 + 置信度下降回溯 remasking |
| **RDD** | 可逆离散扩散，遇停滞则合并前一块重解码 |
| **DCD** | 基于置信度的滑动窗口延迟提交机制 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（LLaDA2.1-Mini 上平均得分）
| 方法 | 平均得分 | 相比 Standard ↑ |
|------|----------|------------------|
| Standard Decoding | 60.44 | — |
| ReMDM | 60.66 | +0.22 |
| SABER | 60.46 | +0.02 |
| RDD | 55.81 | -4.63 |
| DCD | 58.93 | -1.51 |
| **Training-free MBE** | **61.14** | **+0.70** |
| **MBE (with SFT)** | **63.10** | **+2.66** |

### 与基线方法的关键对比结果
- **Training-free MBE** 在 **13/13** 基准上超越所有 baseline，尤其在需要长程一致性的任务上表现突出：
  - **ZebraLogic**: +1.9 (70.40 vs 68.50)
  - **HumanEval+**: +2.44 (82.93 vs 80.49)
  - **AIME 2025**: +0.0 (36.67 vs 36.67)，尚未体现优势（因 $K=2$ 限制）

- **MBE SFT** 进一步带来显著提升：
  - **AIME 2025**: **+13.3** (50.00 vs 36.67) —— 最大增益，体现长程推理优势
  - **ZebraLogic**: **+5.9** (74.40 vs 68.50)
  - **GPQA**: +4.74 (49.49 vs 44.75)

### 消融实验与关键发现
- **质量-吞吐权衡（Quality vs Throughput）**
  - 图 4 显示 MBE 方法位于帕累托前沿（Pareto frontier），在几乎不损失吞吐的情况下显著提升质量。
  - **TPF（Tokens Per Forward）**：
    - Standard: 8.63
    - Training-free MBE: 8.42
    - MBE: 8.42
  - 在某些任务（如 Spider、BFCLv3）上，MBE 甚至实现了更高 TPF，说明早期纠错减少了后续 block 的迭代次数。

- **跨模型规模泛化性验证（LLaDA2.1-Flash）**
  - 在更大模型上验证 MBE 效果，结果表明改进并非源于模型弱小：
    - ZebraLogic: +4.0
    - MBPP+: +2.38
    - Nexus FCB: +1.79
    - 仅 GPQA 微降 (-0.26)，符合预期（事实类任务依赖知识而非上下文）

---

## 4. 关键结论和发现

### 主要发现
1. **Block boundary problem 是真实且严重的结构性缺陷**，表现为 per-position perplexity 随 block 位置单调上升（图 2），且错误会持续传播。
2. **Routine cross-block editing 是有效的解决方案**：即使不修改模型权重，仅通过训练免费的 MBE 解码即可显著提升性能。
3. **Training-inference alignment 至关重要**：通过 Multi-Block Edit SFT 对齐训练与推理模式，能进一步释放更宽编辑窗口（$K>2$）的潜力，尤其在长链推理任务上效果惊人（AIME2025 +13.3）。
4. **系统优化保障实用性**：多形状 CUDA Graph 与细粒度 KV cache 控制使得 variable-length editing passes 在实践中高效可行，恢复了 87% 的标准解码吞吐。

### 方法的局限性
- **计算开销仍存在**：尽管 TPF 接近标准解码，但额外的 forward pass 增加了总计算量，可能影响极端低延迟场景。
- **SFT 成本较高**：需要额外 8M 样本进行 fine-tuning，且需设计复杂的注意力掩码与课程策略。
- **编辑窗口有限**：目前最大 $K=4$，对于极长文本中的远距离依赖仍可能不足。

### 未来工作方向
- 探索动态调整编辑频率与窗口大小（adaptive $K$）以进一步优化效率。
- 将 MBE 思想推广至其他非扩散类 LLM 架构中，研究是否可用于缓解 AR 模型的左向上下文缺失问题。
- 结合强化学习或搜索机制，实现更智能的编辑决策（何时、何地、编辑多少）。
- 探索无需额外训练即可适配更宽编辑窗口的轻量化方法（如 prompt tuning 或 LoRA-based adaptation）。

</details>

---

### 2. [MM-ShiftKV: Decode-Aware Prefill-Stage KV Selection for Multimodal Large Language Models](https://arxiv.org/abs/2607.22586)

**Authors**: Jinsong Shu, Chenyang Wu, Zhongle Xie, Baokun Wang, Lidan Shou  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2607.22586v1  

#### Abstract
Key-Value (KV) caching is essential for efficient inference in multimodal large language models (MLLMs), yet its memory footprint grows linearly with context length and becomes a major bottleneck due to the large number of visual tokens. Recent prefill-stage KV selection methods estimate KV importan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MM-ShiftKV: Decode-Aware Prefill-Stage KV Selection for Multimodal Large Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Multimodal Large Language Models (MLLMs)** 中，**Key-Value (KV) 缓存** 是实现高效推理的关键机制。然而，由于视觉输入通常生成大量 **visual tokens**，KV 缓存的内存占用随上下文长度线性增长，成为内存和解码效率的主要瓶颈。

现有 **prefill-stage KV selection** 方法依赖于预填充阶段（prefill）的统计信息来估计 KV 重要性，并隐含假设：**prefill 阶段的查询行为能代表 decoding 阶段的行为**。  
但本文指出，这一假设在多模态场景中 **不成立**：  
- **decoding 阶段的 hidden-state 表示具有显著更大的方差（variance）**，导致基于 prefill 统计的重要性估计不稳定。
- 小的排序误差可能导致 **语义关键的 visual tokens 被错误丢弃**，从而损害语言对齐（grounding）和推理能力。

### ✅ 提出的新方法：MM-ShiftKV
提出 **MM-ShiftKV** —— 一种 **无需训练、decode-aware、严格 prefill-only** 的 KV 选择框架。

#### 核心思想
在 prefill 阶段显式模拟 decoding 阶段的查询行为，使 KV 选择“感知”到 decoding 时的动态特性，而非仅依赖 prefill 统计。

#### 关键技术
1. **Variance-Expanded Query Proxies 构造**：
   - 在 prefill 阶段，从当前输入的 hidden states 中计算均值和标准差。
   - 引入 **方差扩展因子 γ > 1**，采样合成 **variance-expanded query proxies**，以近似 decoding 阶段更高的表示方差。
   - 这些代理查询被赋予“未来位置”的 **RoPE 编码**，以反映 decoding 时对 prompt 的访问模式。

2. **基于注意力质量的聚合投票机制（Group-wise Voting）**：
   - 多个 query proxies 分组，每组计算其对 prompt keys 的注意力分布。
   - 对每组，保留累计注意力质量超过阈值 τ 的最小 key 集合，这些 key 获得“投票”。
   - 最终重要性得分由总票数决定，结合 **last-query anchor** 打破平局。

3. **严格 prefill-only 设计**：
   - 所有操作在 prefill 结束时一次性完成。
   - 选定的 KV 缓存在整个 decoding 阶段保持不变，兼容 **FlashAttention** 等高效内核。

### ✅ 相比现有方法的优势
| 特性 | MM-ShiftKV | SnapKV / ExpectedAttn / KEYDIFF |
|------|------------|-------------------------------|
| 是否需要训练 | ❌ 否 | ❌ 否 |
| 是否干预 decoding | ❌ 否 | ❌ 否 |
| 是否考虑 decode-time 查询分布 | ✅ 是（通过代理） | ❌ 否（仅用 prefill 统计） |
| 对方差失配的鲁棒性 | ✅ 高 | ❌ 低 |
| 在极端压缩下的稳定性 | ✅ 显著更优 | ❌ 性能下降剧烈 |

---

## 2. 核心实验方法和设置

### ✅ 数据集
在多个代表性多模态基准上进行评估：
- **OCRBench**：精确文本识别任务，使用 **Exact Match Accuracy**。
- **DocVQA**：文档图像问答，使用 **Average Normalized Levenshtein Similarity (ANLS)**。
- **TextVQA / ChartQA / MMMU**：视觉问答与图表推理，使用 **Exact Match Accuracy**。
- **TextCaps**：图文描述生成，使用 **CIDEr** 指标。

> 平均输入 token 数量高达数千（如 Qwen2.5-VL 达 4830），凸显 KV 缓存压力。

### ✅ 实验设置
- **模型**：`Qwen2.5-VL-7B-Instruct` 和 `LLaVA-v1.6-Vicuna-7B`。
- **KV 缓存预算**：每 head 每层设定为 `{64, 128, 256, 512}`，模拟严格内存约束。
- **协议**：**one-shot prefill-only** 协议，即 prefill 结束后一次性压缩 KV 缓存，之后不再修改。
- **兼容性要求**：所有方法需兼容 **FlashAttention**，不引入 decoding-time 开销。

### ✅ 基线方法对比
- **FullKV**：无压缩，作为上限。
- **StreamingLLM**：保留 attention sink 和滑动窗口。
- **SnapKV**：基于 prefill 末尾窗口的注意力相似性评分。
- **ExpectedAttn (ExpAttn)**：估计未来查询分布下的注意力。
- **KEYDIFF**：基于 key 间余弦相似性的查询无关评分。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Table 1 & 2）
在 **Qwen2.5-VL-7B** 上，**64 tokens/head** 极端压缩下：

| 方法 | Avg Score |
|------|-----------|
| FullKV | 77.6 |
| SnapKV | 66.2 |
| ExpectedAttn | 74.0 |
| KEYDIFF | 75.1 |
| **MM-ShiftKV (ours)** | **75.5** ✅ |

> **MM-ShiftKV 在极低预算下仍接近 FullKV 性能**，显著优于其他压缩方法。

在 **LLaVA-7B** 上，**64 tokens/head** 下：
- **MM-ShiftKV** 在 OCRBench 上达到 **68.8**，比 SnapKV (**54.5**) 提升 **~26%**。
- 在 TextCaps 上提升 **~40%**，表明其在生成任务中更好保留视觉信息。

### ✅ 与基线方法的对比结果
- 在所有任务和预算下，**MM-ShiftKV 一致优于所有 baseline**。
- 随着缓存预算降低，性能优势 **更加显著**（gap 扩大）。
- 在 **OCR 和 grounding 密集型任务** 上提升最大，说明其有效保护了关键视觉 token。
- 与 PyramidKV、AdaKV 等预算分配策略结合时，仍带来 **10%-20% 的额外增益**，表明其通用性。

### ✅ 消融实验结果（Table 5）
在固定 64 tokens/head 下逐步添加组件（Qwen2.5-VL）：

| 变体 | OCRBench | TextCaps (CIDEr) |
|------|----------|------------------|
| LastAttn（仅最后查询） | 52.3 | 40.8 |
| + Query Sampling | 54.5 | 45.5 |
| + Variance Expansion | 59.6 | 48.6 |
| **+ Group Voting (MM-ShiftKV)** | **68.3** | **50.4** |

> **Variance expansion 和 group voting 是关键**，分别贡献约 7 和 9 个点的提升。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Prefill-Decoding Scale Mismatch 是真实存在的**：
   - decoding 阶段的 hidden states 方差显著高于 prefill 阶段（跨层、跨数据集一致）。
   - 这种 **分布偏移** 导致传统 prefill-only 方法失效。

2. **Decode-awareness 至关重要**：
   - 即使不干预 decoding，也能在 prefill 阶段通过 **代理查询** 模拟 decode-time 行为。
   - **方差扩展** 是校准这种行为差异的有效手段。

3. **MM-ShiftKV 实现了更优的 accuracy-memory-latency 权衡**：
   - 在 32K 输入长度下，相比 FullKV：
     - **峰值内存减少 43%**（32.9GB → 18.6GB）
     - **解码延迟降低至 1/1.9×**
   - 且完全兼容 FlashAttention。

### ✅ 方法的局限性
1. **Prefill 阶段开销增加**：
   - 引入 query proxy 采样和注意力计算，带来一定 prefill 开销（但仅一次，占比小）。
2. **仅在 prefill 阶段裁剪 KV**：
   - 不适用于需要长上下文持续推理的任务（如超长视频理解）。
3. **主要针对图像/视频模态**：
   - 对音频等其他模态可能面临新挑战。

### ✅ 未来工作方向
- 与 **decoding-time eviction** 或 **offloading** 技术结合。
- 探索与 **model quantization**、**speculative decoding** 等压缩范式的联合优化。
- 扩展至 **audio-visual MLLMs** 和 **long-context reasoning** 场景。
- 动态调整 variance expansion factor γ 以适应不同输入。

---

> **代码已开源**：[https://github.com/zjuDBxAI/MM-ShiftKV](https://github.com/zjuDBxAI/MM-ShiftKV)

</details>

---

### 3. [Gleam: Adaptive Network-Efficient CUDA API Remoting for Cross-Device GPU Sharing over LANs](https://arxiv.org/abs/2607.23115)

**Authors**: Zhihao Xu, Hao Zhong, Zeting Zhou, Yuhang Xu, Haoyu Tong, Wei Wang, Jinshan Chen, Keqiang He, Chong Zhu, Shengzhong Liu, Fan Wu, Guihai Chen  
**Category**: cs.DC  
**Published**: 2026-07-28  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2607.23115v1  

#### Abstract
This paper aims to enable computation- and communication-efficient GPU sharing across devices within local area networks (LANs), facilitating ubiquitous AI inference on heterogeneous personal devices. We achieve distributed task offloading via CUDA API remoting. However, beyond raw computation, netw...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Gleam: Adaptive Network-Efficient CUDA API Remoting for Cross-Device GPU Sharing over LANs**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本论文旨在解决在局域网（LAN）环境下，**跨设备共享 GPU 资源时面临的网络效率瓶颈**，尤其是在边缘计算场景下，个人设备（如笔记本电脑）缺乏强大 GPU，而局域网中的其他设备存在算力闲置的问题。

传统方法如任务级卸载或完整环境迁移需要复杂的部署配置，且不适用于异构、轻量化的 AI 推理任务。而现有的 **CUDA API Remoting** 技术虽然灵活，但在实际应用中面临三大挑战：
- **高频率 API 调用导致通信延迟累积**
- **模型加载阶段大量权重传输消耗有限带宽**
- **多任务并发执行时的网络与计算资源争用（contention）**

### **提出了什么新方法或新思路**
作者提出 **Gleam** ——一个面向 LAN 环境的高效、自适应的 CUDA API Remoting 框架，具备以下三个核心模块：

#### ✅ **1. CUDA API Remoting 优化**
- **Model Weight Manager**：通过哈希索引识别静态模型权重块，在服务器端缓存并复用，避免重复传输。
- **API Remoting Path Manager**：引入三种异步执行机制以减少往返通信开销：
  - **Basic Async**：对仅返回错误码的 API 异步处理
  - **Local Simulation**：客户端本地模拟状态查询类 API（如 `cudaGetDevice`）
  - **Batched Prefetch**：批量预创建资源句柄（如 stream/event），降低依赖阻塞

#### ✅ **2. 争用感知的任务调度器（Contention-Aware Task Scheduler）**
- 设计两阶段调度算法：
  - **Phase 1**：优先将已有缓存的任务分组调度到同一 GPU，提升权重复用率和 GPU 利用率
  - **Phase 2**：基于带宽占用情况调度需加载模型的任务，缓解网络拥塞
- 引入基于拥塞模型的在线延迟预测器，综合考虑通信与计算资源占用（resource occupancy）

#### ✅ **3. CUDA 上下文一致性守护（CUDA Context Consistency Guardian）**
- **Reconciliation after Network Failure**：采用双工 TCP 连接 + 请求缓冲 + 双向确认机制，实现断连后上下文恢复
- **Protection of Cross-CUDA-stream Multiplexing**：全局门控机制阻止在 CUDA Graph 捕获期间非法调用上下文级 API（如 `cudaMalloc`）

### **相比现有方法的优势**
| 维度 | 现有方法（如 GVirtuS, cricket+SR, DGSF） | Gleam |
|------|----------------------------------------|-------|
| **通信效率** | 同步调用频繁，无权重缓存 | 支持异步执行 + 权重缓存，显著降低传输量 |
| **调度智能性** | 忽视通信与计算争用 | 显式建模 contention，动态优化调度决策 |
| **系统鲁棒性** | 不支持长连接容错 | 提供断连恢复与跨流保护机制 |
| **通用性** | 多为特定任务设计 | 支持多种框架（PyTorch/ggml）和任务类型 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与任务**
共使用 **7 个主流 AI 推理任务**，涵盖语言、图像生成与科学计算：
- **Large Language Models (LLMs)**：`llama-3B-ggml`, `llama-8B-ggml`, `whisper-large-v3`
- **Multimodal / Image Generation**：`llava-7B-PyTorch`, `sd-compvis`, `sd3-medium-t5`
- **Scientific Computing**：`fno-PyTorch`

这些任务分别构建于 **PyTorch** 和 **ggml（C++）** 框架之上，验证了 Gleam 的跨框架兼容性。

### **实验设置**
- **硬件平台**：
  - **服务器**：4 台异构机器，配备不同型号 NVIDIA GPU（RTX A4500, RTX 4090, RTX 4070, RTX A6000）
  - **客户端**：1 台模拟多客户端请求的高性能主机（Intel i7-14700）
  - **网络**：Wi-Fi 6 或 Ethernet，最大上行带宽 1000 Mbps
- **软件实现**：
  - 基于 **gRPC + Protobuf** 构建通信层
  - 使用 **LD_PRELOAD** 劫持超过 1,000 个 CUDA API（覆盖率达 57%~95%，见 Table 9）
  - 实现约 6K 行 C++ 主逻辑代码 + 10K 行拦截代码

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Throughput** | 每分钟处理请求数（归一化以消除任务差异影响） |
| **Latency** | 平均端到端延迟 |
| **Queuing Delay** | 客户端等待被调度的时间 |
| **Makespan** | 最后一个任务完成时间 |
| **Bandwidth Usage** | 实际占用带宽 |
| **GPU Utilization** | GPU 利用率监控 |

### **基线方法对比**
#### **API Remoting 基线**：
- **GVirtuS**：早期 GPGPU API 远程调用方案
- **cricket+SR**：支持 Shadow Resource 的异步资源创建
- **DGSF**：通过句柄预取消除部分远程调用

#### **调度策略基线**：
- **FGD**（Fragmentation Gradient Descent）：云环境中最小化资源碎片的调度器
- **Mudi**：面向边缘资源复用、避免分布式争用的调度器

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 性能维度 | 结果 |
|---------|------|
| **API Remoting 效率提升** | **1.4× ~ 24.2×** 优于 SOTA 基线 |
| **系统吞吐量提升** | 最高达 **1.79×** |
| **端到端延迟降低** | 最多减少 **1.56×** |
| **队列等待延迟降低** | 最多减少 **2.88×** |
| **权重缓存命中收益** | 如 `llava-7B` 加载时间从 162.68s → **8.78s**（降幅近 95%）|

> 🔹 在 Wi-Fi 下，`sd3-medium-t5-ggml` 达到 **54×** 于 GVirtuS 的加速比  
> 🔹 即使在 Ethernet 上，仍比基线快 **3.9×~8.4×**

### **与基线方法的对比结果**
- **图17** 显示，在所有六项任务中，Gleam 在 Wi-Fi 和 Ethernet 下均显著领先于 GVirtuS、cricket+SR 和 DGSF。
- **图18** 表明，随着 GPU 数量增加，Gleam 始终保持最低延迟和最高吞吐。
- **图19** 显示，Gleam 将排队延迟降低了最多 2.88×，makespan 缩短 1.76×，说明其调度更均衡。

### **消融实验结果（Ablation Study）**
#### **图25：模块移除影响**
| 移除组件 | 影响 |
|--------|------|
| **完整 API Path Manager** | 端到端延迟 ↑ 7.9×，吞吐 ↓ 66% |
| **争用感知调度器（替换为公平调度）** | 队列延迟 ↑ 6.2× |

#### **图26：逐步添加优化效果**
- 从朴素版本开始，依次加入：
  1. 异步执行（Async）
  2. 局部模拟（Simulate）
  3. 批量预取（Batch）
  4. 权重缓存（Weight Cache）
- 每一步都带来持续性能提升，最终实现 **>20×** 加速

> 💡 结论：两个核心模块（API 优化路径 + 争用感知调度）缺一不可

---

## **4. 关键结论和发现**

### **主要发现**
1. **网络是 API Remoting 的主要瓶颈**，而非计算本身；尤其是模型加载阶段的大体积权重传输。
2. **静态权重高度可复用**，通过哈希识别与缓存可大幅削减带宽消耗（>90% 减少）。
3. **高频短 API 调用可通过异步化有效优化**，即使在 Wi-Fi 等高延迟链路上也能接近本地执行性能。
4. **多任务并发下的通信与计算争用必须联合建模**，否则会导致资源利用率下降和尾延迟上升。
5. **长生命周期的 API Remoting 必须保障上下文一致性**，否则易因网络抖动或跨流操作崩溃。

### **方法的局限性**
- **缓存粒度依赖于内存分配模式**：若模型权重与激活值混合在同一 `cudaMalloc` 块中，则无法准确识别（但实测中此类情况较少）。
- **PyTorch 框架优化受限**：由于其内部封装较深，部分 API 无法通用化优化，导致其相对 ggml 应用的性能损失更大（>3× vs <2×）。
- **未考虑安全性和访问控制**：当前假设局域网内设备可信，未涉及数据加密或权限管理。

### **未来工作方向**
1. **扩展至 WAN 场景**：探索低带宽广域网下的进一步压缩与预取策略。
2. **支持更多框架与运行时**：如 TensorFlow, ONNX Runtime 等。
3. **引入安全性机制**：增加身份认证、数据加密与访问审计功能。
4. **发展协同推理技术**：多个边缘 GPU 共同服务单个大模型推理任务。
5. **结合模型压缩与量化技术**：进一步降低传输负载。

---

> ✅ **总结一句话**：  
> **Gleam 通过“细粒度通信优化 + 争用感知调度 + 上下文守护”三位一体的设计，首次实现了在普通 LAN/Wi-Fi 环境下高效、稳定、通用的跨设备 GPU 共享，为边缘 AI 推理提供了实用化解决方案。**

</details>

---

### 4. [cMoLLM at Scale: Horizontal Scaling Laws for Mixture-of-LLMs](https://arxiv.org/abs/2607.22577)

**Authors**: Xin Yang, Yemin Wang, Mingda Liu, Letian Li, Shuaishuai Cao, Zhengxiao He, Ryan Dong  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.22577v1  

#### Abstract
Scaling large language models (LLMs) has driven their success, yet dense Transformers couple capacity and computation: every parameter is activated for every token, making training and inference costs grow linearly with model size-a critical bottleneck as models approach trillion-parameter regimes. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：cMoLLM at Scale: Horizontal Scaling Laws for Mixture-of-LLMs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **dense Transformer** 模型在扩展时面临一个根本瓶颈：**模型容量与计算成本强耦合**。每个 token 处理时都会激活全部参数，导致训练和推理成本随模型规模线性增长。当模型进入万亿参数（trillion-parameter）时代时，这种低效性成为严重制约。

尽管 **Mixture-of-Experts (MoE)** 被用于提升参数效率，但现有方法大多仅将 MoE 应用于 **Feed-Forward Network (FFN)** 层，并采用离散的 Top-K 路由机制，这带来了以下问题：
- 专家崩溃（expert collapse）
- 负载不均衡（skewed utilization）
- 训练不稳定（brittle training）

此外，一些 pipeline-level 扩展方法如 **ParaScale** 和 **AltUp** 引入了虚拟 token 或辅助预测分支，带来额外开销或收敛缓慢等问题。

---

### 提出了什么新方法或新思路
本文提出 **cMoLLM (convolutionally-gated Mixture-of-LLMs)**，一种全新的 pipeline-level 容量扩展框架，其核心思想基于一项理论洞察：

> **MoE-style 混合层可以被精确重写为动态卷积（dynamic convolution）**，其中每个专家对应一个 $1\times1$ 卷积核，而路由机制实现输入条件下的核聚合。

基于此等价关系，cMoLLM 将整个 LLM 流水线视为可路由的“流”（stream），通过 **全可微分的动态卷积** 实现端到端的混合。

#### 方法设计要点：
- 维持一组并行的 end-to-end “streams”，每条 stream 拥有独立的 $1\times1$ 卷积核。
- 使用轻量级 **gating network** 生成输入依赖的软混合权重 $\{g_k(x)\}$。
- 动态组合所有核：$K(x) = \sum_k g_k(x) K_k$，然后通过标准 grouped $1\times1$ convolution 应用。
- 整个过程无 Top-K、无低秩分解、无虚拟 token、无辅助头（auxiliary heads）。

---

### 相比现有方法的优势

| 特性 | ParaScale | AltUp | cMoLLM |
|------|---------|-------|--------|
| 虚拟 token | ✅ | ❌ | ❌ |
| 辅助预测分支 | ❌ | ✅ | ❌ |
| 路由方式 | 并行流（易梯度坍缩） | 固定预测路径 | 动态卷积（全可微） |
| 计算开销 | 更高 | 较低 | ~dense |
| 训练稳定性 | 差（collapse risk） | 慢收敛 | 稳定 |

**优势总结**：
- 参数高效且硬件友好（利用高效的 convolutional primitives）。
- 训练更稳定，避免了专家崩溃和负载失衡。
- 支持 **水平扩展律（horizontal scaling laws）**：固定计算预算下，增加 stream 数量可线性提升有效容量。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **预训练数据**：FineWeb —— 一个大规模清洗过的网页语料库（Penedo et al., 2024）。
- **下游任务评估**：
  - **GLUE benchmark**：多任务自然语言理解任务集合。
  - **SQuAD v2**：问答任务，使用 F1 分数评估。

---

### 实验设置和评估指标

#### 模型架构
- 基于 **GPT-2-style Transformer** 架构。
- 三种规模配置：
  - Small: ~85M 参数
  - Medium: ~350M 参数
  - Large: ~760M 参数
- 替换原 FFN 层为 **cMoLLM block**，保持其余结构不变。

#### 关键变量
- **Stream 数量 $N$**：$\{1, 2, 4, 8\}$
- **Gating 变体**：
  - `simple`
  - `context_aware`
  - `multi_head`（表现最佳）
  - `adaptive`

#### 评估指标
- **语言建模性能**：Validation Loss、Perplexity (PPL)
- **下游任务性能**：GLUE 得分（%）、SQuAD v2 F1（%）
- **训练稳定性与利用率**：各 stream 的平均门控概率分布
- **计算复杂度**：FLOPs 对比

#### 基线方法对比
- **Dense GPT-2**：标准全连接 Transformer。
- **ParaScale**（Chen et al., 2025）：引入虚拟 token 和并行流。
- **AltUp**（Baykal et al., 2023）：使用辅助预测分支更新非活跃模块。

所有实验均控制相同训练设置（数据、超参、优化器等），确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & Table 3）

| 模型 | Loss ↓ | PPL ↓ | GLUE ↑ (%) | SQuAD v2 ↑ (%) |
|------|--------|--------|-------------|----------------|
| Dense Baseline (760M) | 2.55±0.05 | 12.79±0.28 | 55.74±0.46 | 62.91±0.74 |
| **cMoLLM-multi_head (760M)** | **2.24±0.02** | **9.41±0.45** | **58.45±0.77** | **64.56±1.32** |

> ✅ 在最大模型上，cMoLLM 相比 dense 基线：
> - **PPL 下降 26.4%**
> - **GLUE 提升 4.8%**
> - **SQuAD 提升 2.6%**

---

### 与基线方法的对比结果
- 在相同计算预算下，**cMoLLM 显著优于 dense、ParaScale 和 AltUp**。
- **Perplexity 随 stream 数增加持续下降**，尤其 `multi_head` 和 `adaptive` 变体在 $N=8$ 仍保持稳定，未出现过拟合或利用率坍缩。
- **ParaScale** 存在明显的梯度坍缩风险，且因虚拟 token 导致计算开销更高。
- **AltUp** 收敛速度慢，性能增益有限。

---

### 消融实验结果（Ablation Study）
- **不同 gating 机制效果排序**：
  ```
  multi_head > adaptive ≈ context_aware > simple
  ```
  表明多头机制能更好捕捉不同时间尺度或语义维度的信息。
  
- **Stream 数量影响**：
  - $N=1$ 到 $N=4$ 性能显著提升；
  - $N=8$ 时部分变体（如 simple）开始饱和甚至轻微下降，但 `multi_head` 依然受益。
  
- **负载均衡损失（load balancing loss）**：
  - 加入辅助平衡项（$\alpha=0.01$）后，各 stream 的平均门控概率更加均匀，防止少数 stream 主导输出。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **理论层面**：
   - 首次建立了 **MoE 与动态卷积之间的形式化等价性**（Theorem 4.1），为稀疏、条件计算提供了统一分析框架。
   - 标准 CNN 可被视为一种隐式的、高度约束的 MoE（静态、密集、非归一化路由）；而 cMoLLM 是对其的放松与推广。

2. **方法层面**：
   - cMoLLM 成功将 MoE 思想从 FFN 层扩展至 **整个 LLM pipeline**，实现了真正的 pipeline-level 条件计算。
   - 全可微分的软路由机制提升了训练稳定性与资源利用率。

3. **实验层面**：
   - cMoLLM 在匹配计算条件下，**一致优于 dense 基线及其他 pipeline 扩展方法**。
   - 验证了 **水平扩展律（horizontal scaling law）**：在合理范围内，增加 stream 数量可线性提升模型容量而不显著增加计算负担。
   - `multi_head` gating 表现最优，具备良好的扩展潜力。

---

### 方法的局限性
- 当前实验集中在 **GPT-2 规模**（最大 760M 参数），尚未验证在十亿级以上（如 7B+）大模型中的有效性。
- 未探索将该思想应用于 **attention 机制本身**（目前仅用于 FFN 替代）。
- 虽然计算量接近 dense，但在极高 stream 数时仍有一定内存占用上升。

---

### 未来工作方向
1. **更大规模验证**：将 cMoLLM 扩展到 7B+ 参数模型，并集成到分布式训练系统中。
2. **扩展至 Attention**：研究是否可将 attention head 也纳入动态卷积路由框架。
3. **结合其他条件计算机制**：例如与检索增强（retrieval-augmented）或专家专业化（expert specialization）结合。
4. **理论深化**：进一步分析动态卷积在表示学习中的归纳偏置及其泛化边界。

---

> **Impact Statement 精要**：
> - ✅ 正面影响：提升 LLM 扩展效率，降低训练/推理能耗，硬件友好。
> - ⚠️ 风险：不引入新的滥用模式，但仍需警惕 LLM 通用风险（如虚假信息生成）。
> - 🔁 可复现性：代码将在接受后开源，包含完整实现与训练脚本。

</details>

---

### 5. [PIVOT: Efficient Query-Group Indexing for Token-Level Sparse Attention](https://arxiv.org/abs/2607.24593)

**Authors**: Hong Liu, Yuan Cheng, Lin Niu, Yi Su, Yufei Xue, Anmin Liu, Guanghua Yu, Jianchen Zhu  
**Category**: cs.CL  
**Published**: 2026-07-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.24593v1  

#### Abstract
Token-level sparse attention, as implemented by DeepSeek Sparse Attention (DSA) in production systems, makes the downstream attention efficient but shifts the bottleneck to the indexer that feeds it. To select the top-k tokens for each query, the indexer must still score every preceding token, incur...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# PIVOT: Efficient Query-Group Indexing for Token-Level Sparse Attention 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模语言模型（LLM）中，**DeepSeek Sparse Attention (DSA)** 通过 token-level 稀疏注意力机制显著降低了下游注意力计算的成本。然而，其瓶颈转移到了为每个查询选择 top-k 相关 token 的 **indexer** 上。传统方法对每个 query 都需扫描整个前缀序列进行打分，导致 indexing 成本高达 $O(L^2)$ 每层（$L$ 为序列长度），在长上下文场景下成为主导延迟的因素。

### 提出的新方法：PIVOT
本文提出 **PIVOT (Proxy Indexing Via One full-prefix Traversal)** ——一种无需训练、即插即用的 DSA indexer 替代方案，其核心思想是：
- 利用相邻 queries 之间存在高度重叠的 top-k token 选择这一观察（cross-query redundancy）；
- 将一组邻近的 queries 聚合为一个 **proxy query**；
- 对该 proxy 执行一次共享的全前缀扫描（full-prefix traversal），生成候选集；
- 各个 query 从该共享候选集中选出自己的 top-k。

PIVOT 包含两个变体：
- **PIVOT-Reuse**：整组共享 proxy 的 top-k 结果，速度最快。
- **PIVOT-Refine**：以 proxy 的 top-c（c > k）作为候选集，各 query 再对其重新打分并独立选取 top-k，在极小额外开销下恢复原始 DSA 的精度。

### 相比现有方法的优势
- **正交性**：PIVOT 沿着 **query axis** 优化，与已有沿 token、head、layer axis 优化的方法（如 HISA、MISA、IndexCache）完全正交，可组合使用。
- **高效性**：将每组 g 个 queries 的 indexing 成本从 $O(gL)$ 降至接近 $O(L)$，大幅加速 indexer 模块。
- **通用性**：单一算法适用于 **prefill** 和 **decode** 两个推理阶段，仅 grouping 策略不同：
  - Prefill：固定大小的连续 query 分组；
  - Decode：利用 MTP（Multi-Token Prediction）一步生成的多个 draft tokens 自然构成一组。
- **无损集成**：不改变 Sparse MLA 接口、KV Cache 或训练过程，部署简单。

---

## 2. 核心实验方法和设置

### 数据集
- **LongBench**：涵盖多领域真实长文档任务，包括单文档问答（SQA）、多文档问答（MQA）、摘要（Sum）、少样本学习（FS）、合成任务（Syn）、代码补全（Code）等。
- **RULER**：可控长度的检索与推理基准，测试从 4K 到 128K 的长上下文能力。

### 实验设置和评估指标
- **模型**：在生产级模型 **DeepSeek-V3.2** 和 **GLM-5.1** 上验证，二者均采用 DSA 和 MTP（d=3，decode group size g=4）。
- **评估指标**：
  - **准确性**：任务特定指标（如准确率）的平均值（AVG）。
  - **效率**：
    - Indexer kernel 加速比（speedup over DSA）
    - 端到端（end-to-end）延迟降低
- **超参数**：
  - Group size $g = 4$
  - Candidate budget $c = 2k = 4096$ （默认 $k=2048$）
  - Proxy aggregation：per-head mean pooling
  - 实现于 vLLM 框架，运行在 NVIDIA H20 GPU 上。

### 基线方法对比
- **DSA**：原始密集 indexer，作为性能上限。
- **HISA**：沿 token axis 优化的 indexer 加速方法。
- **MISA**：沿 head axis 优化的 indexer 加速方法。
- **IndexCache**：沿 layer axis 优化，跨层复用索引。
- **PIVOT + IC**：PIVOT 与 IndexCache 组合，验证正交性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | Indexer Kernel Speedup (Max) | End-to-End Speedup (Max) |
|------|-------------------------------|---------------------------|
| PIVOT-Reuse | **~4.8×** | **~1.6×** |
| PIVOT-Refine | **~4.1×** | **~1.6×** |

> 在 256K 上下文长度下达到峰值加速效果。

### 与基线方法的对比结果

#### 准确性（Table 1）
- **LongBench**：
  - 所有稀疏 indexer 表现接近 DSA（差距 < 0.5 pts），PIVOT-Refine 在 DeepSeek-V3.2 上甚至略优于 DSA（56.18 vs 55.95）。
  - PIVOT 在 code completion 和 few-shot learning 等局部结构强的任务上表现更优。
- **RULER（长上下文）**：
  - HISA 在 128K 下严重退化（↓19–28 pts），因 block-level 剪枝永久丢弃 token。
  - MISA 中度退化（↓6–18 pts）。
  - **PIVOT-Refine 全程跟踪 DSA 精度**，在 128K 下仍保持一致。
  - PIVOT-Reuse 在短上下文表现良好，但在极端长度下略有下降，体现 refine 步骤的重要性。

#### 效率（Figure 4）
- **Indexer Kernel Speedup**：
  - 随上下文增长而提升，符合预期（amortization effect）。
  - PIVOT-Reuse 始终更快；PIVOT-Refine 在短序列有轻微开销，但在长序列迅速反超。
- **End-to-End Speedup**：
  - 在短上下文下增益有限（indexer 占比小）；
  - 在长上下文（>64K）下显著，最高达 **1.6×**，正是长上下文服务最需要的地方。

### 消融实验结果（Ablation Studies）

#### ✅ Query Aggregation（Table 2 & F6）
- **Mean pooling** 明显优于取第一个或最后一个 query 作为 proxy。
- 在 128K 下，first-query proxy 准确率下降超过 10 pts，而 mean pooling 保持稳定。
- **结论**：mean 是更鲁棒的聚合方式。

#### ✅ Group Size $g$（Table 3 & F8）
- $g=4$ 时精度几乎无损；
- 随 $g$ 增大（如 $g=16$），精度显著下降，尤其在长上下文；
- **结论**：小 group size（如 4）可在精度与效率间取得最佳平衡。

#### ✅ Candidate Budget $c$（Table 4 & F7）
- $c=4096$（即 $2k$）已足够；
- 增大 $c$ 收益边际递减；
- 减小 $c$（如 $3072$）开始损失精度；
- **结论**：$c=2k$ 是性价比最优选择。

#### ✅ 部署阶段影响（Table 5 & F9）
- PIVOT 应用于 **prefill only**、**decode only** 或 **both** 均能保持高精度；
- **结论**：算法统一适用于两个阶段，部署灵活。

---

## 4. 关键结论和发现

### 主要发现
1. **Query-level redundancy exists**：相邻 queries 的 top-k 选择高度重叠，且 indexer scores 呈长尾分布，这为共享计算提供了理论基础。
2. **Grouped proxy indexing is effective**：通过一个 proxy query 共享全前缀扫描，可极大减少冗余计算。
3. **PIVOT-Refine 匹配 DSA 精度**：在极小额外成本下实现与原始 dense indexer 完全一致的 top-k 选择。
4. **加速随上下文增长而放大**：在 256K 上下文下，indexer 加速达 **4×**，端到端延迟降低 **1.6×**，解决了长上下文下的主要瓶颈。
5. **与 MTP 天然协同**：decode 阶段直接复用 MTP 的 draft tokens 构成 group，零成本获得加速。

### 方法的局限性
- **依赖 MTP**：decode 阶段的 grouping 依赖于 MTP 的存在；若系统未启用 MTP，则无法自然形成 group。
- **极端 group size 影响精度**：当 group size 过大时，共享候选集难以覆盖所有 query 的需求，导致 recall 下降。
- **短序列无优势**：在短上下文下，indexer 本身非瓶颈，PIVOT 增益有限。

### 未来工作方向
- **扩展至其他模型家族**：将 PIVOT 思路推广到非 DSA 架构的稀疏注意力模型。
- **动态 grouping 策略**：探索基于语义或注意力模式的 adaptive grouping，而非固定位置分组。
- **结合更多 axis 优化**：进一步整合 token/head/layer/query 四个维度的加速策略，实现极致推理效率。
- **应用于训练阶段**：探索是否可在训练中引入类似机制以提升效率。

</details>

---

### 6. [Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization](https://arxiv.org/abs/2607.22583)

**Authors**: Muhammad Junaid Ali, Smail Niar, El-Ghazali Talbi  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.22583v1  

#### Abstract
Large Language Models (LLMs) have achieved widespread adoption because of their strong reasoning and query-response capabilities. However, deploying them in embedded and edge computing environments remains challenging because of strict latency, memory, and energy constraints. Their large parameter c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在边缘设备（如 NVIDIA Jetson Nano）上部署面临**严格延迟、内存和能耗限制**。尽管模型剪枝（pruning）是压缩模型的有效手段，但现有方法存在以下问题：
- 多数方法仅关注单一目标（如参数量），忽略**多目标权衡**（accuracy vs. latency vs. model size）；
- 缺乏对硬件实际推理延迟的显式优化；
- 深度剪枝（depth-wise）与宽度剪枝（width-wise）常被孤立处理，难以探索全局最优配置。

### 🚀 提出的新方法
本文提出一种**硬件感知、多目标结构化剪枝框架**，采用两阶段策略：

#### 阶段一：Multi-Objective Depth-Wise Pruning（粗粒度）
- 使用 **NSGA-II** 进行多目标优化，搜索 Pareto 最优架构；
- 目标函数为：
  - **KL 散度**（衡量输出分布损失，代表性能保持能力）；
  - **参数数量** 或 **latency**（效率指标）；
- 移除整个 MHA 和 MLP 子块，实现模型深度缩减。

#### 阶段二：Parallel Bayesian Optimization + Importance-Based Pruning（细粒度）
- 在阶段一生成的 Pareto 候选集中，使用 **Parallel Bayesian Optimization (PBO)** 分配每层的剪枝比例（pruning ratio）；
- 以最小化 **验证集 perplexity** 和 **硬件延迟** 为目标；
- 利用重要性评分（importance score）决定各层内 MHA heads 和 MLP neurons 的移除顺序。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **优化方式** | 显式建模多目标权衡，避免局部最优；分离宏观架构搜索与微观组件剪枝，提升搜索效率 |
| **硬件适配性** | 显式优化真实硬件上的 **latency**，适用于主流边缘设备（无需稀疏张量核） |
| **灵活性** | 支持多种 importance estimator（如 Gradient、Taylor、Wanda-SP）进行消融分析 |
| **通用性** | 在多个 LLM 家族（Mistral、LLaMA、Qwen、Phi）上验证有效 |

---

## 2. 核心实验方法和设置

### 📚 数据集
| 类型 | 名称 | 用途 |
|------|------|------|
| **校准集（Calibration）** | FineWeb-Edu（256 samples） | 阶段一 KL 散度计算 |
| **验证集（Validation）** | MMLU、HellaSwag、ARC-Easy、ARC-Challenge（共 1,024 样本） | 阶段二 BO 评估 |
| **语言建模评估** | WikiText-2、C4、FineWeb-Edu | Perplexity 测试 |
| **零样本任务评估** | MMLU、HellaSwag (HS)、ARC-e/c、PIQA、WinoGrande (WG) | 下游任务准确率测试 |

### ⚙️ 实验设置
- **剪枝率**：37.5% 和 50%
- **模型家族**：Mistral-v0.3-7B、LLaMA-2-7B、Qwen-2.5-7B、Phi-3-14B
- **训练/微调**：未使用 LoRA fine-tuning，确保公平比较
- **硬件平台**：
  - **服务器级**：NVIDIA A100 80GB ×3
  - **边缘设备**：NVIDIA Jetson Nano（Ampere 架构，1024 CUDA cores）
- **工具库**：PyTorch 2.1、HuggingFace Transformers、lm-evaluation-harness、BoTorch/GPyTorch

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Zero-shot Accuracy** | 多任务平均准确率（MMLU、ARC 等） |
| **Perplexity (PPL)** | 在 WikiText-2/C4/FineWeb 上的语言建模能力 |
| **Latency** | 100-token 序列推理耗时（秒），A100 与 Jetson Nano 上均测试 |
| **Throughput** | tokens/s |
| **Active Parameters** | 实际参与计算的参数量 |
| **Pareto Front 质量** | KL loss vs. 参数量 / latency 的权衡曲线 |

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **ShortGPT [11]** | Depth pruning | 发现层冗余，直接删除部分 transformer 层 |
| **Sliding Window [10]** | Depth pruning | 层合并策略 |
| **Block Pruner [9]** | Structured | 组件级重要性打分，联合剪枝 MHA & MLP |
| **EvoPress [15]** | Evolutionary Search | 使用进化算法动态压缩 |
| **SliceGPT [8]** | PCA-based | 删除矩阵行列 |
| **2SSP [7]** | Two-stage | 先剪 FFN 后剪 attention |
| **CFSP [23]** | Coarse-to-fine | 基于激活信息引导剪枝（仅支持 LLaMA） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table I & II）

#### ✅ 零样本任务表现（Average Accuracy）
| 方法 | 37.5% 平均准确率 | 50% 平均准确率 |
|------|------------------|---------------|
| **Proposed** | **51.92 ± 0.73** | **46.03 ± 0.69** |
| SliceGPT | 45.73 | 39.27 |
| 2SSP | 46.37 | 41.47 |
| CFSP | 49.62 | 36.94 |

> **结论**：提出的框架在所有模型家族中取得最高平均准确率，尤其在 **ARC-e/c**（推理类任务）上表现稳健。

#### ⏱️ 推理延迟（100-token，10次平均）
| 方法 | A100 (s) | Jetson Nano (s) |
|------|----------|----------------|
| **Proposed** | **73.8512** | **843** |
| ShortGPT | 81.936 | 1283 |
| 2SSP | 92.615 | 1150 |
| Block Pruner | 85.468 | 1074 |

> **结论**：在两种平台上均达到最低延迟，Jetson Nano 上相比基线降低 ~20–40%，显著提升边缘部署可行性。

#### 📉 Perplexity 对比（Figure 5）
- 在 **WikiText-2、C4、FineWeb-Edu** 上，该方法在 37.5% 和 50% 剪枝率下均获得**更低的 PPL**；
- 表明其不仅保留下游任务性能，也更好维持了基础语言建模能力。

#### 🔍 消融实验结果

##### （1）Proxy Metric 有效性（Figure 7 & Table III）
| Sparsity | 最佳代理指标 | 相关系数（r） |
|---------|-------------|--------------|
| 37.5% | KL Divergence | -0.967 |
| 50%   | Perplexity     | -0.828 |

> **发现**：轻度剪枝时 KL 更可靠；重度剪枝时 PPL 成为更强信号。

##### （2）Stage-Wise 设计必要性（Figure 9）
- 单独使用 depth-only 或 width-only 均不如两阶段组合；
- **两阶段协同可平衡精度与效率**，避免过度破坏结构完整性。

##### （3）Importance Estimator 比较（Table IV）
| 方法 | 准确率↑ | PPL↓ | Latency↓ |
|------|--------|-------|---------|
| **Wanda-SP** | **0.4927** | 38.625 | 3.9623 |
| **Gradient** | 0.4811 | **23.531** | 3.7610 |
| **Taylor** | 0.4821 | 23.625 | **3.7659** |

> **结论**：无“全能”估计器。若优先 accuracy → 选 Wanda-SP；若优先 efficiency → 选 Gradient/Taylor。

##### （4）Sparsity Allocation 方法比较（Table V）
| 方法 | PPL↓ | Latency↓ | Downstream Avg↑ |
|------|------|----------|------------------|
| **Proposed (Beta)** | **9.6309** | **3.7871** | **0.5142** |
| Uniform | 10.3632 | 3.8885 | 0.4898 |

> **结论**：基于 BO 的非均匀分配优于固定比例剪枝，在各项指标上全面领先。

##### （5）目标函数组合分析（Table VI）
| 第一目标 | 第二目标 | Inference Time | Accuracy | PPL |
|--------|----------|----------------|----------|-----|
| KL-Divergence | Params | 5.86s | **0.590** | 8.14 |
| KL-Divergence | **Latency** | **4.86s** | 0.565 | **7.48** |

> **发现**：`KL + Latency` 可实现约 **17% 推理加速**，代价为 2.8% 准确率下降，适合强延迟约束场景。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **两阶段设计优于单阶段剪枝**：先全局搜索 Pareto 架构，再精细化分配剪枝比例，能更有效地探索复杂设计空间。
2. **非均匀剪枝分布更高效**：BO 自动学习到不同层对性能敏感度不同，选择性保留关键层组件。
3. **KL 散度是有效的早期代理指标**：可在不运行完整 downstream evaluation 的情况下快速筛选候选模型。
4. **硬件感知优化至关重要**：显式建模 latency 可显著提升边缘设备上的实际推理速度。
5. **结构化剪枝兼容性强**：无需专用稀疏硬件即可在标准 GPU 上获得显著加速。

### ⚠️ 方法局限性
- **依赖校准数据质量**：过小或偏差大的 calibration set 会影响 importance scoring 效果（见 Fig. 8）；
- **阶段间耦合不足**：Stage 1 输出影响 Stage 2 输入，但未实现端到端联合优化；
- **未考虑动态输入长度适应性**：当前剪枝策略静态，无法根据输入动态调整路径；
- **缺乏对 attention pattern 的建模**：未利用注意力稀疏性进一步优化。

### 🔮 未来工作方向
1. 扩展 Bayesian Optimization 搜索空间，引入 **layer grouping 或 block sharing** 策略；
2. 引入 **multi-fidelity optimization** 加速搜索过程；
3. 结合量化（quantization）形成 **pruning + quantization 联合压缩流程**；
4. 探索 **任务自适应剪枝**，根据不同下游任务定制剪枝策略；
5. 将框架扩展至 Vision-Language Models（VLMs）或多模态场景。

---

> 💡 **一句话总结**：  
> 本文提出了一种**硬件感知、两阶段、多目标结构化剪枝框架**，通过 **NSGA-II + Parallel Bayesian Optimization** 实现了在保持高 accuracy 的同时显著降低 LLM 的 model size 与 inference latency，为大模型在边缘设备上的高效部署提供了实用解决方案。

</details>

---

### 7. [Application-Driven Architecture Exploration for Cross-Layer Heterogeneous Systems](https://arxiv.org/abs/2607.23042)

**Authors**: Yuchen Fan (Tsinghua University), Minghong Sun (Tsinghua University), Jikui Ma (Tsinghua University), Yunpeng Xu (Tsinghua University), Shunyu Mao (Tsinghua University), Liu He (Tsinghua University), Shunan Dong (Tsinghua University), Jiahao Yang (Tsinghua University), Yu Zhu (Tsinghua University), Xinhao Yang (Tsinghua University), Tianyan Zhong (Tsinghua University), Haoran Sun (Tsinghua University), Daoqi Liu (Tsinghua University), Zongle Huang (Tsinghua University), Xinyuan Lin (Tsinghua University), Huazhong Yang (Tsinghua University), Maokun Li (Tsinghua University), Yongpan Liu (Tsinghua University), Yu Wang (Tsinghua University), Zhenhua Zhu (Tsinghua University), Hongyang Jia (Tsinghua University), Shuwen Deng (Tsinghua University)  
**Category**: cs.DC  
**Published**: 2026-07-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.23042v1  

#### Abstract
AI and HPC infrastructure increasingly serves workload portfolios that combine dense tensor computation, sparse kernels, large memory footprints, and communication-intensive collectives. Supporting these portfolios requires coordinated choices across accelerators, memory tiers, scale-up fabrics, and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Cross-layer Heterogeneous Systems (XHS)** 架构设计空间庞大且受物理约束（如功耗、交换机端口数、布线限制、成本）影响的问题，提出了一种系统性的架构探索框架。传统方法存在以下两大挑战：
- **Challenge 1 (C1)**：不同工作负载对系统资源的需求差异巨大（如LLM需要高带宽张量计算，稀疏计算依赖低延迟内存访问），单一硬件配置无法在所有任务上都表现最优。
- **Challenge 2 (C2)**：XHS设计空间高度离散且受物理约束，导致大量候选架构不可部署，而现有工具（如节点级DSE工具或固定平台模拟器）无法联合优化硬件拓扑与软件映射。

### 提出的新方法：CHASE
论文提出了 **CHASE (Application-Driven Architecture Exploration)** 框架，其核心创新在于：
- **Decoupled Two-Level Optimization**：采用内外双层循环解耦硬件搜索与软件映射：
  - **内层（Inner Loop）**：给定一个硬件架构 `h`，由 **Mapper** 生成拓扑感知的事件轨迹（topology-aware event traces）。
  - **外层（Outer Loop）**：基于 **Simulator** 的性能反馈，由 **Optimizer** 进化硬件图结构。
- **Hierarchical Graph Modeling & Constraint Filtering**：将XHS建模为分层的有类型图（package, node, rack, cluster），并在映射前通过物理约束过滤器（如功率、端口、预算）剔除非法候选，避免无效计算。
- **Real-System Calibration Path**：通过真实平台测量校准事件驱动模拟器，确保性能预测的可靠性，支持跨规模趋势推断。

### 相比现有方法的优势
| 方面 | 现有方法（如Timeloop, ASTRA-SIM） | CHASE |
|------|----------------------------------|-------|
| **搜索范围** | 仅限于节点级或固定平台 | 支持从package到cluster的全栈XHS探索 |
| **硬件-映射耦合** | 分离或假设固定映射 | 显式建模 `h → S_h` 的依赖关系，避免不公平比较 |
| **物理可行性** | 忽略或弱化物理约束 | 在搜索早期即过滤不可部署架构 |
| **评估可信度** | 依赖理论模型 | 基于真实测量进行点级和趋势级双重校准 |

---

## 2. 核心实验方法和设置

### 数据集与工作负载
- **稀疏计算套件**：
  - **HPCG**：迭代稀疏线性求解器，强调不规则内存访问和Reduction操作。
  - **Sparse Direct Solvers**：基于Trojan Horse模型的GPU稀疏直接求解器，包含细粒度DAG任务调度。
- **LLM推理套件**：
  - 包括 **Qwen3.5 (32B)**, **Llama3.1 (70B)**, **GPT-3 (13B)**, **Gemma (27B)**, **Mixtral (8x7B MoE)** 等主流模型，覆盖Decoder-only和MoE架构。

### 实验设置与评估指标
- **目标**：在相同成本、功耗约束下，寻找最优XHS架构。
- **评估流程**：
  1. 用户输入：工作负载DAG (`DAG.json`)、可用硬件列表 (`HW_List.json`)、物理约束 (`Constr.json`)。
  2. CHASE输出：优化后的硬件架构 `h` 及其对应的映射策略。
- **关键指标**：
  - **Geomean Speedup**：相对于基线架构的几何平均加速比。
  - **Mapping Quality**：与穷举最优解的差距。
  - **Simulation Fidelity**：模拟器误差（vs. 真实测量）。
  - **Optimizer Convergence**：达到近全局最优所需的迭代次数。

### 基线方法对比
- **Mapper 对比**：PEFT, HEFT, AEFT, HOFT, Greedy。
- **Simulator 对比**：ASTRA-SIM。
- **Optimizer 对比**：Random Search, Greedy Search, Simulated Annealing。
- **系统级基线**：
  - **稀疏计算**：类 El Capitan 的纯H100架构。
  - **LLM推理**：NVL72-like 的64-GPU H100 SuperPOD。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **Mapper 质量** | 相比穷举最优，平均差距仅 **6.06%**，加权平均 **5.68%**。 |
| **Mapper 效率** | 相比PEFT，映射时间平均减少 **60.5%**；相比HEFT/AEFT/HOFT，分别减少 **63.6%/64.4%/82.7%**。 |
| **Simulator 准确性** | 计算误差平均 **4.4–7.5%**；机内通信误差 < **10%**；对未见平台（L20）预测误差 **5.8%**。 |
| **Simulator 速度** | 相比ASTRA-SIM，模拟时间减少 **15.5%**，吞吐达 **2.10×10⁵ events/sec**。 |
| **Optimizer 收敛性** | 在64次迭代内即可收敛至近全局最优。 |
| **端到端加速比** | 稀疏工作负载：**6.20×** geomean speedup；LLM推理：**2.12×** geomean speedup。 |

### 与基线方法的对比结果
#### 稀疏计算案例（Table 8）
| 指标 | El Capitan-like Baseline | CHASE Optimized |
|------|--------------------------|-----------------|
| GPU 组成 | 128 H100 SXM | 8 H200 + 4 H100 + 4 L40S |
| CPU ranks | 64 | 5 |
| Geomean Speedup | 1.00× | **6.20×** |
| 成本与功耗 | 更高 | 更低 |

> ✅ **发现**：最优架构并非全用最快GPU，而是将高带宽H200用于关键路径（如`getrf`, `gessm`），其余任务由低成本H100/L40S处理。

#### LLM推理案例（Table 9）
| 指标 | NVL72-like Baseline | CHASE Optimized |
|------|---------------------|-----------------|
| GPU 数量 | 72 H100 | 16 H200 + 24 H100 |
| Host 数量 | 18 trays | 5 HGX hosts (2 H200 + 3 H100) |
| Geomean Speedup | 1.00× | **2.12×** |
| 功耗与成本 | 更高 | 更低 |

> ✅ **发现**：最优架构采用“scale-up islands”模式，每个8-GPU主机内部完成Tensor Parallelism，跨岛使用Pipeline Parallelism，避免过度拆分导致通信开销上升。

### 消融实验结果
- **Optimizer有效性**（Figure 13）：在可穷举的小空间中，TG-RL Optimizer在64次迭代内逼近全局最优，远优于Random/Greedy等基线。
- **Mapper优化效果**（Figure 9）：提出的PEFT-LC算法在保持调度质量的同时显著降低开销，尤其在大规模DAG上优势明显。
- **Calibration必要性**：未经校准的模拟器会错误排序候选架构，而校准后能准确反映真实趋势。

---

## 4. 关键结论和发现

### 主要发现
1. **工作负载决定最优XHS架构形态**：
   - **稀疏计算** 偏好 **criticality-aware heterogeneous pods**：将高性能资源集中于DAG关键路径，其余任务由异构低成本单元处理。
   - **LLM推理** 偏好 **scale-up islands**：Tensor Parallelism应限制在高带宽单机内，Pipeline Parallelism连接各岛，避免过度并行化。
2. **“越多GPU越好”是误区**：对于LLM，盲目增加GPU数量会导致过细的Tensor Parallelism，反而因通信开销降低性能。
3. **物理约束必须前置**：在搜索初期过滤非法架构可大幅提升效率，避免浪费在不可部署方案上的评估。
4. **软硬协同优化至关重要**：硬件拓扑决定了合法的映射空间 `S_h`，必须联合优化才能获得公平排名。

### 方法的局限性
- **搜索空间仍受限**：尽管采用启发式方法，XHS设计空间依然巨大，难以保证绝对全局最优。
- **依赖高质量校准数据**：模拟器准确性高度依赖真实平台的测量数据，若缺乏代表性平台，泛化能力可能下降。
- **Mapper假设理想调度**：当前Mapper基于静态DAG，未完全考虑运行时动态行为（如负载漂移、故障恢复）。

### 未来工作方向
- 扩展支持 **动态自适应调度**，实现运行时反馈闭环优化。
- 引入 **多目标优化**（如能效、可靠性、容错性）替代单一性能指标。
- 探索 **自动化硬件模板生成**，减少人工定义的拓扑模板依赖。
- 将CHASE应用于 **AI Factory** 场景，联合优化训练、推理、仿真等混合流水线。

> 🔚 **总结**：CHASE首次实现了从应用驱动的、物理可行的XHS全栈架构探索，验证了“按需定制”优于“统一模板”的设计理念，为下一代AI/HPC基础设施提供了科学的设计方法论。

</details>

---

### 8. [Learning to Optimize: Joint Routing and Flow Allocation on Sparse Non-Euclidean Networks](https://arxiv.org/abs/2607.23467)

**Authors**: Haomiao Sun, Fang He, Congyuan Ji, Xindi Tang  
**Category**: cs.LG  
**Published**: 2026-07-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.23467v1  

#### Abstract
We study an integrated pickup-and-delivery problem on sparse, non-Euclidean networks that jointly optimizes cyclic routing, cargo flow allocation, and cross-cycle service. The tight coupling of these operational constraints creates a complex discrete-continuous decision space with highly restricted ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning to Optimize: Joint Routing and Flow Allocation on Sparse Non-Euclidean Networks*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文研究的是**稀疏非欧几里得网络上的集成取送货问题（Non-Euclidean Pickup-and-Delivery Problem with Cross-Cycle Service, NE-PDP-CCS）**，该问题在现实工业场景中广泛存在，如集装箱班轮运输系统。其核心挑战在于：
- 网络是**稀疏、有向、非欧几里得**的，无法用平面坐标建模；
- 需要同时优化**循环路径规划（cyclic routing）** 和 **货物流分配（cargo flow allocation）**；
- 存在**跨周期服务（cross-cycle service）**：前一航次装载的货物可能在当前周期内交付；
- 节点具有多重角色（origin, destination, transshipment hub），且服务可选（optional fulfillment）。

传统方法（如MIP、DP、LKH3、ALNS等）难以高效求解此类大规模、高耦合度的问题，而现有的神经组合优化（NCO）模型（如POMO、NCS）主要针对全连接欧氏空间设计，在稀疏网络上表现不佳。

---

### 提出了什么新方法或新思路
作者提出了一个端到端的深度强化学习框架：**Double-Channel Graph Attention (DCGA)**，其核心创新包括：

#### （1）双通道图注意力编码器（Double-Channel Graph Encoder）
- **网络通道（Network Channel）**：建模物理可达性、旅行成本与时间，处理稀疏连通性和非对称属性；
- **需求通道（Demand Channel）**：建模OD配对关系、服务逻辑（如数量、收益、软截止期限）；
- 引入**逻辑-物理-逻辑（LPL）投影机制**，实现多任务节点之间的信息共享，避免表示坍缩。

> ✅ 这种分离式架构有效防止了拓扑结构与服务逻辑的混淆，提升了表示能力。

#### （2）状态条件化解码器 + 约束感知策略塑造（Constraint-Informed Decoding）
- 使用**复合掩码机制（composite operational masking）** 在每一步排除非法动作：
  - 边可行性（sparse reachability）
  - 容量约束（capacity feasibility）
  - 物理节点访问次数限制
  - 终止合法性判断
- 设计**交付侧偏置（delivery-side bias）**，允许“先送后取”的跨周期操作，无需硬性前置约束；
- 解码过程由确定性**模拟器（simulator）驱动**，实时更新负载、时间、容量等动态状态。

> ✅ 实现了在高度受限可行域内的有效探索，避免后修复（post-hoc repair）带来的低效。

#### （3）联合优化目标与奖励机制
- 政策训练基于**终端奖励（episodic reward）**，即完整路径生成后的总利润（收入 − 成本 − 惩罚）；
- 使用**actor-critic框架**，结合蒙特卡洛回报与价值函数基线，降低方差；
- 引入熵正则化鼓励探索。

---

### 相比现有方法的优势
| 维度 | DCGA优势 |
|------|---------|
| **适用场景** | 明确支持稀疏、非欧、循环、跨周期服务，贴近真实工业系统 |
| **求解效率** | 推理速度达**秒级**，适合动态重规划 |
| **解的质量** | 在大规模实例上显著优于MIP、LKH3、ALNS、GA、SA及NCS等基线 |
| **泛化能力** | 对需求扰动（quantity/price变化）表现出强鲁棒性 |
| **结构设计** | 双通道+掩码机制为复杂约束下的学习提供了通用范式 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用公开海运基准数据集 **LinerLib**（Brouer et al., 2014）中的地中海子集；
- 构造了多个规模的测试实例：**NE-PDP61, 81, 91, 101, 121, 141**，数字代表请求对数；
- 请求参数（数量、价格、截止期）通过受控方式生成，确保可复现性。

### 实验设置与评估指标
| 设置项 | 描述 |
|-------|------|
| 平台 | Intel Xeon Platinum CPU + NVIDIA RTX PRO 5000 GPU |
| 训练 | 批大小=2，每批次2个实例，共训练约600轮 |
| 测试 | 报告每个测试批次的平均目标值（Objective）、运行时间（Runtime）、相对最优差距（gap%） |
| 评估指标 | 
| - 目标函数值（含收入、运输成本、未满足需求惩罚、延迟惩罚） |
| - gap(%) = $(\text{best\_known} - \text{method}) / \text{best\_known} \times 100\%$ |
| - 推理耗时（seconds-level） |

---

### 基线方法对比
分为三类共13种基线：

#### （1）优化类（Optimization）
- **MIP**：Gurobi求解McCormick松弛后的MINLP，限时1小时；
- **Flow Model (FM)**：先解TSP式MIP得路径，再固定路径解线性规划进行货流分配。

#### （2）启发式与元启发式（Heuristic & Metaheuristic）
- **LKH3**：最强经典TSP/VRP求解器之一，适配为m-PDTSP；
- **GA1/GA2**, **SA1/SA2**, **ALNS1/ALNS2**：分别以成本最小或收益最大化为目标的遗传算法、模拟退火、自适应邻域搜索。

#### （3）学习型方法（Learning-Based）
- **NCS1/NCS2**：当前最先进的神经PDP求解器（Kong et al., 2024）；
  - NCS1：使用经纬度作为坐标输入 + 可行性掩码；
  - NCS2：额外加入DCGA的图编码器，但保留原解码结构。

> ⚠️ 所有学习方法均在相同训练协议下重新训练至收敛。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 3 & 4）

| 方法 | NE-PDP 141 Obj ($10^5$) | Gap (%) | Runtime (s) |
|------|------------------------|--------|-------------|
| **DCGA (Ours)** | **-102.48** | **0.00** | **18.31** |
| FM | -128.26 | 152.78 | 150.97 |
| ALNS2 | -108.97 | 114.76 | 319.88 |
| GA2 | -90.72 | 78.79 | 945.25 |
| SA2 | -106.90 | 110.68 | 668.97 |
| LKH3 (#5000) | — | >100 | >5 days |
| NCS2 | -102.48 | 101.97 | 2.86 h |

> ✅ DCGA在所有大尺度实例上取得**最佳目标值**，且推理时间始终低于21秒。

---

### 与基线方法的对比结果
- **vs MIP/FM**：尽管MIP被放松为乐观上界，DCGA仍能在中大规模上超越它，说明其解质量接近甚至优于数学规划；
- **vs LKH3/ALNS/GA/SA**：这些方法随规模增长性能急剧下降，尤其LKH3在NE-PDP141上无法完成5000次尝试；
- **vs NCS系列**：即使NCS2引入了相同的图编码器，其性能仍远逊于DCGA，表明**性能突破不仅来自更好的特征提取，更源于整体架构设计（双通道+掩码机制）**；
- **扩展性分析**：随着问题规模增大，DCGA相对于其他方法的优势**显著扩大**，体现其在复杂耦合结构中的优越性。

---

### 消融实验结果（Ablation Studies）

#### （1）掩码机制消融（Fig. 7 & 8）
- 移除除基本连通性外的所有掩码 → 性能大幅下降；
- 未满足需求惩罚激增，说明模型频繁做出不可行决策；
- 表明**可行性掩码对于引导高质量搜索至关重要**。

#### （2）通道数量对比（Fig. 9）
- 单通道模型（SCGA）在所有规模下均劣于DCGA；
- 差距随问题规模增大而拉大 → 证明**双通道设计提供了互补信息，增强了决策稳定性与质量**。

#### （3）敏感性分析（Appendix I）
- **图编码器头数（Graph Encoder heads）** 是关键瓶颈：太少则表达能力不足，太多则引入噪声；
- **嵌入维度（embedding dim）** 越大越好，但边际效益递减；
- **融合编码器头数** 影响较小，说明其作用主要是稳定整合。

---

## 4. 关键结论和发现

### 主要发现
1. **结构感知的学习框架优于通用NCO模型**：将网络拓扑与服务逻辑分离建模（双通道）能显著提升在稀疏非欧网络上的表现；
2. **约束前置优于后修复**：通过硬掩码和偏置机制在决策过程中主动规避非法动作，比事后修正更高效可靠；
3. **DCGA具备卓越的扩展性与鲁棒性**：在大规模实例上保持秒级响应，并在需求扰动下表现稳定；
4. **性能优势源于整体架构而非单一组件**：即便增强版NCS2获得相同图编码器，也无法复制DCGA的成功，说明**解码机制与策略塑造才是核心驱动力**。

---

### 方法的局限性
- 当前框架适用于**单回路（single service loop）** 场景，尚未扩展至多路线协同调度；
- 依赖预定义的静态网络拓扑，未考虑**动态因素**（如拥堵、天气、突发需求）；
- 模拟器虽保证可行性，但仍是确定性的，缺乏对不确定性建模的能力；
- 图编码器的设计对超参数较敏感（如head数量），需仔细调参。

---

### 未来工作方向
1. **扩展至多路线规划（multi-route planning）**：支持系统级调度、共享运力与协调中转；
2. **引入随机与动态元素**：结合场景感知训练、滚动重规划与鲁棒强化学习应对不确定性；
3. **结合轻量级改进搜索（lightweight improvement search）**：如将DCGA作为初始解生成器，接续局部搜索进一步提升质量；
4. **理论分析**：建立在稀疏图上的可行性保障与近似比理论刻画，连接学习方法与经典优化保证。

--- 

> 📌 **总结一句话**：  
> DCGA提出了一种面向**真实工业物流系统**的新型学习范式——通过**双通道分离建模 + 约束前置引导**，实现了在**稀疏非欧网络**上高质量、低延迟的联合路径与货流优化，为下一代智能调度引擎提供了有力候选方案。

</details>

---

### 9. [PRESTO: Prefix-Aligned Tree Drafting for Diffusion Speculative Decoding](https://arxiv.org/abs/2607.22634)

**Authors**: Zheng Wang, Zhifan Ye, Qi Cheng, Yonggan Fu, Ziyan Wang, Feng Zhu, Haozhe Zhao, Jan Kautz, Pavlo Molchanov, Humphrey Shi, Minjia Zhang  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.22634v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs, generating tokens in parallel. This makes them effective draft models for speculative decoding (SD), producing an entire block of draft tokens in a single forward pass. Yet existing diffusion...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PRESTO: Prefix-Aligned Tree Drafting for Diffusion Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

现有的基于 **diffusion** 的 **speculative decoding (SD)** 方法普遍采用 **linear (单路径) drafting**，即在每个解码步骤中仅生成一条候选路径并逐个验证。然而，diffusion 模型本质上是 **non-autoregressive** 的，能够在一个 forward pass 中并行生成多个位置上的多种候选 token，形成一个巨大的组合空间。

但现有方法未能有效利用这一丰富的多候选结构，导致：
- 只探索了极小部分的候选路径；
- 接受长度（acceptance length）受限；
- 解码效率未达上限。

此外，作者指出一个**根本性不匹配问题**（fundamental mismatch）：
> diffusion 模型输出的是 **prefix-blind 的 marginal probabilities**，而 AR 验证器依赖的是 **prefix-conditioned 的条件概率**。  
这导致直接用 diffusion 的 marginal 分数进行路径排序时不可靠，影响树构建质量。

---

### 🚀 **提出了什么新方法或新思路**

作者提出 **PRESTO** —— 一种面向 diffusion drafters 的 **principled tree-based drafting 框架**，其核心思想包括两个关键设计：

#### （1）**Prefix-Aligned Scoring（前缀对齐评分）**
为解决 diffusion marginals 与 prefix-based verification 的不匹配，引入一个 **prefix-conditioned 修正信号 $ p_a $**（如 n-gram 模型），将原始 diffusion 分数 $ q_d $ 进行加权融合：

$$
p^*(t|c_a) \propto q_d(t)^{\lambda} \cdot p_a(t|c_a)
$$

对应的 token-level 得分为：
$$
s_d,t(c_a) = \log q_d(t) + \lambda \log p_a(t|c_a)
$$

该设计使得候选路径的打分更符合实际 AR 验证过程中的前缀依赖特性。

#### （2）**Priority-Based Tree Search（优先级树搜索）**
基于上述 prefix-aligned 得分，采用 **beam search with global retention** 或 **best-first search** 构建 draft tree，优先扩展高分路径，在有限节点预算 $ B $ 下最大化期望接受长度。

---

### 🔍 **相比现有方法的优势**

| 方面 | 优势说明 |
|------|---------|
| **适用性广** | 支持两种主流 diffusion SD 范式：<br>- **dedicated diffusion drafter**（如 dFlash）<br>- **self-speculative dLLMs**（如 Nemotron-Labs-Diffusion） |
| **通用性强** | 是首个可同时应用于 dedicated 和 self-speculative 设置的 tree drafting 框架 |
| **无需额外训练** | 仅修改 drafting 策略，不影响 drafter 或 verifier 的训练 |
| **系统开销低** | tree construction 开销仅占总延迟的 0.7%–4%，几乎无负担 |
| **提升显著** | 在各类任务上平均实现 **1.5× throughput 加速（dFlash）** 和 **1.12×（self-speculative dLLM）** |

---

## 2. 核心实验方法和设置

### 📚 **使用的数据集**

涵盖三大类典型任务，用于全面评估性能：

| 类别 | 数据集 |
|------|-------|
| **数学推理** | GSM8K, MATH-500, AIME24, AIME25 |
| **代码生成** | HumanEval, MBPP, LiveCodeBench (LCB), SWE-Bench |
| **对话/通用** | MT-Bench, Alpaca |

---

### ⚙️ **实验设置与评估指标**

#### **模型设置**
- **Dedicated Drafter Setting**:  
  - **Drafter**: Qwen3-4B-DFlash, Qwen3-8B-DFlash  
  - **Target**: 对应的 Qwen3 AR LLM
- **Self-Speculative Setting**:  
  - **Model**: Nemotron-Labs-Diffusion-8B（支持 linear & quadratic 自推测模式）

#### **评估指标**
| 指标 | 含义 |
|------|------|
| **Average Acceptance Length ($\bar{T}$)** | 每轮验证中被接受的 draft token 数量均值 |
| **Throughput Speedup** | 相比纯 AR 解码的 token/s 提升倍数 |
| **End-to-End Throughput (tokens/s)** | 实际每秒生成 token 数量 |

#### **实现细节**
- 使用 **PyTorch + FlexAttention** 或集成到 **SGLang + FlashInfer**
- Tree budget: dFlash 设为 {128, 256, 512}；NLD 固定为 32
- 批大小：1（单请求场景）
- Prefix signal $ p_a $：使用轻量级 **3-gram model**
- $\lambda = 0.2$（固定或熵自适应）

#### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **dFlash (vanilla)** | 原始 linear drafting 方法 |
| **Nemotron-Labs-Diffusion (vanilla)** | 官方 linear / quadratic self-speculative 实现 |
| **Oracle Upper Bound** | 多路径穷举的理想上限（用于分析潜力） |

---

## 3. 主要实验结果和性能指标

### 📊 **关键性能数据汇总**

#### ✅ **在 dFlash 上的表现（Table 2）**

| 模型 | 方法 | 平均吞吐加速 | 平均接受长度 $\bar{T}$ | 最大增益任务 |
|------|------|----------------|------------------------|-------------|
| Qwen3-4B | dFlash | 4.9× | 6.0 | — |
| | **PRESTO** | **7.3×** (**+1.5×**) | **8.8** (+2.8) | +3.0× on LCB |
| Qwen3-8B | dFlash | 4.8× | 6.0 | — |
| | **PRESTO** | **7.3×** (**+1.52×**) | **8.9** (+2.9) | +3.0× on LCB |

> **结论**：PRESTO 显著拉近与理想上限的距离，尤其在数学与代码任务上提升明显（+2~3 tokens）。

---

#### ✅ **在 Nemotron-Labs-Diffusion 上的表现**

##### （1）Linear Self-Speculation Mode（Table 3）

| 温度 | 方法 | 平均吞吐加速 | 平均接受长度 $\bar{T}$ |
|------|------|----------------|------------------------|
| T=0 | Vanilla | 4.6× | 8.8 |
| | **PRESTO** | **4.9×** (**+1.06×**) | **9.9** |
| T=1 | Vanilla | 2.1× | 4.2 |
| | **PRESTO** | **2.5×** (**+1.17×**) | **5.1** |

> 在随机采样下，单路径更容易失败，PRESTO 的多路径探索优势更加突出。

##### （2）Quadratic Self-Speculation Mode（Table 4）

| 温度 | 方法 | 平均吞吐加速 | 平均接受长度 $\bar{T}$ |
|------|------|----------------|------------------------|
| T=0 | Vanilla | 2.6× | 4.7 |
| | **PRESTO** | **3.5×** (**+1.35×**) | **5.6** |
| T=1 | Vanilla | 1.4× | 2.4 |
| | **PRESTO** | **3.0×** (**+2.14×**) | **4.7** |

> 在 T=1 下接近 **3× 吞吐加速**，接受长度翻倍，体现 tree drafting 强鲁棒性。

---

### 🔬 **消融实验结果（Ablation Studies）**

#### （1）Prefix-Conditioned Signal 的有效性（Fig 4-top）
- 移除 $ p_a $（仅用 $ q_d $）会导致接受长度下降；
- 尤其在 **GSM8K、Math500、HumanEval** 等需要逻辑连贯的任务上差距更大；
- 表明 **prefix consistency 对长序列接受至关重要**。

#### （2）超参数 $\lambda$ 敏感性分析（Fig 4-middle）
- $\lambda \approx 0.2$ 时性能最优；
- $\lambda$ 过小 → 忽视 prefix 信息；
- $\lambda$ 过大 → 过度依赖噪声较大的 $ p_a $，反而降低稳定性。

#### （3）Entropy-Adaptive $\lambda$ vs Fixed $\lambda$
- 动态调整 $\lambda$（根据 marginal entropy）仅带来微弱提升；
- 表明简单固定的 $\lambda$ 已足够有效，无需复杂调度机制。

#### （4）系统开销分析（Fig 4-bottom-left）
- **tree-related ops 占比 < 4%**，target model verification 占主导（81%–97%）；
- 说明 PRESTO 几乎无额外计算代价。

#### （5）接受长度分布变化（Fig 4-bottom-right）
- PRESTO 显著减少短接受（<5 tokens）情况；
- 大幅增加“接近整块”或“全块接受”的频率；
- 证明其能发现更全局一致的高质量路径。

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **Diffusion drafters 存在巨大未开发潜力**  
   当前 linear drafting 仅利用了其并行能力的一小部分，存在显著改进空间。

2. **Marginal confidence ≠ Verification likelihood**  
   diffusion 的 marginal 概率虽与接受率正相关，但缺乏 prefix 条件信息，不能准确预测路径接受行为。

3. **Prefix-aligned scoring 至关重要**  
   引入轻量 prefix-conditioned signal（如 n-gram）即可显著改善路径排序质量。

4. **Tree drafting 可无缝迁移到 diffusion 模型**  
   PRESTO 成功将 tree-based speculative decoding 扩展至 diffusion drafters，并首次应用于 self-speculative dLLMs。

5. **性能提升显著且稳定**  
   - 在 dFlash 上平均 **1.5× throughput 提升**
   - 在 self-speculative dLLMs 上平均 **1.12×~1.35× 提升**
   - 特别是在随机解码（T=1）下优势更为突出

---

### ⚠️ **局限性**

1. **依赖外部 prefix-aligned signal $ p_a $**  
   当前使用 n-gram 模型仅捕获局部词法兼容性，缺乏语义理解能力。

2. **未探索更复杂的 $ p_a $ 构造方式**  
   如基于小型 AR 模型或 learned alignment head，可能进一步提升效果。

3. **评估集中在单请求、小批量场景**  
   未在大规模 batch、长上下文或多用户并发环境下测试，难以反映生产系统表现。

4. **$\lambda$ 调参仍需经验设定**  
   尽管固定值已够用，但缺乏理论指导如何动态适配不同任务或 drafter。

---

### 🔮 **未来工作方向**

1. **设计端到端可学习的 prefix-aligned scoring 模块**  
   将 $ p_a $ 替换为 trainable 组件，联合优化 drafting 与 verification 对齐。

2. **扩展至 vLLM、SGLang 等生产级推理框架**  
   验证 PRESTO 在高并发、KV cache 优化等真实部署环境下的有效性。

3. **研究 adaptive tree budget allocation**  
   根据输入难度动态调整 tree size，平衡延迟与收益。

4. **探索 diffusion-specific tree 结构先验**  
   利用 diffusion 的 denoising 路径特性构建更有意义的候选拓扑。

---

## 总结

> **PRESTO 是首个将 tree-based speculative decoding 成功应用于 diffusion drafters 的通用框架**。它通过 **prefix-aligned scoring** 解决了 diffusion marginals 与 AR verification 的根本性不匹配问题，并结合 **priority-based tree search** 实现高效多路径探索。实验证明其可在 **dFlash 和 self-speculative dLLMs** 上带来 **高达 1.5× 的端到端吞吐加速**，且系统开销极低，具有很强的实用价值和推广前景。

</details>

---

### 10. [Keyword Matters: Unveiling the Energy Sensitivity of On-Device LLM Prompting](https://arxiv.org/abs/2607.22568)

**Authors**: Ruiyi Tao, Xiaolong Tu, Haoxin Wang  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.22568v1  

#### Abstract
Large Language Models (LLMs) are increasingly deployed on mobile and embedded devices to improve privacy and reduce network latency. Yet on-device inference faces a fundamental constraint: high energy consumption on battery-powered, resource-limited hardware. While model compression and runtime acce...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Keyword Matters: Unveiling the Energy Sensitivity of On-Device LLM Prompting**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
本论文聚焦于**on-device LLM（在设备端部署的大语言模型）推理过程中的高能耗问题**。尽管已有大量研究通过模型压缩（如量化、剪枝）和硬件加速优化能效，但**prompt 设计对能耗的影响长期被忽视**。本文首次系统地实证研究了**prompt 中关键词（keywords）的选择如何影响 on-device LLM 的能量消耗**。

### ✅ 提出了什么新方法或新思路
- **提出“prompt engineering for energy efficiency”这一新视角**：将传统的 prompt 工程从提升准确率、可控性等目标，扩展到**节能优化**的新维度。
- **设计了一套完整的实验框架**，用于测量不同关键词在真实边缘设备上的能耗差异，涵盖文本生成（text generation）和情感分析（sentiment analysis）两类任务。
- 引入**基于关键词替换的 controlled 实验范式**，控制其他变量不变，仅改变指令动词（如 "create", "generate", "label"），以隔离其对能耗的影响。

### ✅ 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **优化层级** | 模型/硬件级（需重新训练或编译） | 用户级 prompt 层（无需修改模型） |
| **实施成本** | 高（依赖专家调优） | 极低（轻量级提示改写即可） |
| **通用性** | 受限于特定架构或平台 | 跨模型、跨设备具有部分一致性 |
| **隐私影响** | 不适用 | 更好（避免云端传输） |

> 🔍 **核心优势**：提供了一个**无需改动模型或硬件的轻量级节能杠杆（lightweight lever）**，使普通开发者和终端用户也能参与绿色 AI 实践。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **Text Generation**: 基于 [Alpaca-GPT4](https://github.com/tatsu-lab/stanford_alpaca) 数据集构建，选取 50 个开放式生成任务（如写策略、创作口号），每个任务用 10 个不同指令动词替换关键词，共生成 500 条 prompt。
- **Sentiment Analysis**: 使用 [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) 中的评论，构造二分类情感判断任务，同样进行关键词替换，形成另外 500 条 prompt。
- 所有 prompt 仅改变动词，其余内容保持一致，确保公平比较。

### ⚙️ 实验设置
#### 硬件平台
| 设备 | 操作系统 | 备注 |
|------|--------|------|
| **Google Pixel 9a** | Android 15（rooted） | 固定 CPU/GPU 频率，关闭自适应亮度、后台服务、热节流 |
| **Orange Pi 5 Pro** | Ubuntu 22.04 | 进入 multi-user 模式减少干扰 |

> 🔧 所有设备均通过 **FNIRSI FNB58 能量记录仪**采集电池电压/电流（10Hz采样），计算真实功耗。

#### 测试模型家族
- **Qwen2.5-0.5B / 1.5B**
- **SmolLM2-1.7B**
- **Llama-3.2-1B**
- **Gemma-3-1B**

> ✅ 多数为 **instruct 模型**，统一使用 **q4f16 或 g4f16 量化格式**，运行于 **MLC-LLM** 推理框架。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **TTFT (Time to First Token)** | 首个 token 输出延迟 |
| **TBT (Time Between Tokens)** | 平均 token 生成间隔 |
| **E_prefill** | Prefill 阶段能耗 |
| **E_dec** | Decoding 阶段能耗（为主要开销） |
| **Total Energy** | 总能耗 = ∫V×I dt |
| **Response Quality** | 使用 **DeepEval + GPT-4.1-mini** 作为 LLM judge，评估 **Relevance, Accuracy, Coherence, Conciseness** 四项指标 |

#### 基线方法对比
- 无显式基线模型对比，而是以**各模型自身平均能耗为基准**，衡量不同关键词带来的**相对偏差百分比（% deviation from mean）**。
- 对比维度包括：
  - 不同 keyword 的能耗差异
  - 跨模型一致性（Spearman rank correlation）
  - 跨设备一致性
  - 能耗与响应质量的关系

---

## 3. 主要实验结果和性能指标

### 🔋 关键性能数据

#### ✅ 文本生成任务（Text Generation）
- 在 **Qwen-0.5B** 上，“create” 比平均节省 **18.5% 的 decode 能耗**
- “craft” 是最耗能的关键词，在 Qwen-1.5B 上**增加高达 26.5% 能耗**
- 不同模型敏感度排序：**Qwen > SmolLM > Llama > Gemma**
  - Gemma 表现稳定但因输出质量差（重复循环至 token limit），导致能耗波动小

#### ✅ 情感分析任务（Sentiment Analysis）
- “label” 是最优关键词，在所有模型上平均节省 **48.64% decode 能耗**
  - 最佳表现：**Gemma 上降低 62.6%**
- “categorize” 和 “classify” 也表现良好（分别省约 24.5% 和 19.9%）
- 最差关键词：“analyze”（+35.9%）、“assess”（+27.0%）

> 💡 **洞察**：要求“识别”类动词（identify, label）比“分析”类动词更节能，说明 prompt 语义引导了推理深度。

### 🔄 与基线方法的对比结果
| 维度 | 发现 |
|------|------|
| **跨模型一致性** | 整体较低（Spearman ρ ≈ 0.15~0.39），但在同一家族内较高（如 Qwen 间 ρ=0.588~0.867） |
| **跨设备一致性** | 
| - 文本生成 | 较低（Qwen ρ=0.261），因开放性任务存在较大随机性 |
| - 情感分析 | 很高（Qwen ρ=0.939），确定性任务更可复现 |
| **能耗 vs 准确率** |
| - 文本生成 | 几乎无关（ρ≈-0.086），节能不牺牲质量 |
| - 情感分析 | 显著负相关（ρ=-0.632），越耗能的关键词反而准确率越低 → **短而准优于长而错**

### 🔍 消融实验结果（隐含）
虽然未明确命名消融实验，但以下分析具备消融性质：
- **关键词替换未显著改变 prompt 复杂度**：
  - 使用 NVIDIA prompt-task-and-complexity classifier 分析，原 prompt 与重写后复杂度得分 Pearson r=0.942，Spearman ρ=0.944
  - 平均复杂度偏移仅 +0.0151（0~1 scale）
- **固定硬件配置排除外部干扰**：频率锁定、禁用热节流、最小化屏幕功耗等措施保障测量可靠性

---

## 4. 关键结论和发现

### 🎯 主要发现
1. **Prompt 关键词直接影响能耗**  
   即使语义相近的动词（如 "create" vs "craft"）也会导致显著不同的 decoding 长度和能耗，最大差异可达 **60%以上**。

2. **节能 ≠ 牺牲质量**  
   - 在文本生成中，**节能关键词（如 "create"）不影响输出质量**
   - 在情感分析中，**高效关键词（如 "label"）反而更准确**，表明冗余推理易引入错误

3. **模型家族内部具有一致性**  
   同一系列模型（如 Qwen）对关键词的响应模式高度相似，暗示训练数据或微调策略可能强化了某些动词偏好。

4. **任务类型决定稳定性**  
   开放式任务（text generation）受随机性影响大，能耗波动明显；封闭式任务（sentiment analysis）更具可预测性和跨设备一致性。

5. **人类认知启发 AI 行为**  
   需要“深入思考”的动词（analyze, assess）会触发更复杂的 reasoning path，增加 token 数量和能耗 —— 类似人类的认知负荷机制。

### ⚠️ 方法的局限性
1. **单次运行缺乏统计显著性检验**：每组实验仅执行一次，无法估计方差。
2. **量化方案单一**：所有模型采用相同量化等级（4bit），未探索不同 bit-width 下的表现变化。
3. **数据来源有限**：仅来自 Alpaca 和 Yelp，多样性不足，难以泛化至其他领域。
4. **judge model 存在偏差**：GPT-4.1-mini 在评分时可能混淆 accuracy 与其他维度（如 relevance），影响质量评估信度。
5. **设备差异悬殊**：Pixel 9a 的 prefill 时间远高于 Orange Pi（87s vs 0.115s），原因不明，可能反映系统调度或驱动问题。

### 🔮 未来工作方向
1. **开发 Green Prompt Engine**
   - 自动重写高能耗 prompt，结合 linguistic optimization、semantic equivalence check 和 energy feedback loop
   - 可集成至移动助手或开发工具链中

2. **拓展跨设备/跨模型泛化研究**
   - 测试更多设备（Jetson, Coral TPU）
   - 探索不同架构（Transformer variants）、不同量化级别下的规律

3. **融合多目标优化框架**
   - 将 prompt engineering 视为 NAS（Neural Architecture Search）式的系统优化问题
   - 利用强化学习自动搜索最优 prompt 结构（参考 PlatformX 框架）

4. **建立 Prompt Energy Benchmark**
   - 创建标准化的 prompt-energy 测试集与协议，推动可持续 prompt engineering 社区发展

---

> ✅ **最终结论**：  
> **Prompt engineering 不仅是语义控制手段，更是实现可持续 on-device AI 的实用工具**。一个简单的动词选择，就能带来两位数的能耗节省，且不损害甚至提升输出质量。这为绿色 AI 提供了一条低成本、高可用的新路径。

</details>

---

### 11. [PTStore (Prefix Tensor Store): Distributed Prefix Caching and Replication for High Throughput Inference Serving](https://arxiv.org/abs/2607.22648)

**Authors**: Meghana Maghyastha, Robert Underwood, Randal Burns, Bogdan Nicolae  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.22648v1  

#### Abstract
Inspired by the design of client caching in Content Delivery Networks (CDNs), PTStore distributes and replicates popular tensors that form reusable KV cache prefixes, which are the main technique used by state of art approaches to accelerate inferences. This reduces the latency of accessing the KV c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PTStore (Prefix Tensor Store): Distributed Prefix Caching and Replication for High Throughput Inference Serving

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 LLM 推理系统在处理大规模、高并发请求时面临以下挑战：
- **KV Cache 利用率低**：虽然 `KV Cache` 被广泛用于避免重复计算注意力机制中的 K 和 V 向量，但大多数系统仅在单个 GPU 或节点内实现前缀重用（prefix caching），无法跨多个计算节点共享。
- **内存资源孤岛化**：不同节点的 GPU 和主机内存未被聚合利用，导致整体 KV Cache 容量受限，热门前缀仍需频繁重建。
- **远程访问开销大**：即使采用分布式存储（如 EvoStore），缺乏对“热”前缀的本地复制机制，导致大量 RDMA 远程读取，I/O 开销显著。

### 提出了什么新方法或新思路
作者提出 **PTStore** ——一个分布式的、支持细粒度前缀缓存与复制的 **Prefix Tensor Store**，其核心设计思想包括：

- **增量式张量存储（Incremental Tensor Storage）**  
  将新的 KV Cache 对象表示为与已有对象最长公共前缀（Longest Common Prefix, LCP）之间的差异部分，以减少冗余存储。

- **集中式元数据管理（Consolidated Metadata）**  
  每个对象维护一个扁平化的 tensor ID 列表，包含从根到叶路径上的所有张量标识，并隐含归属服务器信息，从而实现高效的“一键式”前缀查询，无需遍历分布式 Trie 结构。

- **分布式多级缓存架构（Distributed Hierarchical Caching）**  
  每个节点运行一个 PTStore 服务端，聚合本地主机内存、SSD 和并行文件系统（PFS），形成统一的存储层级。

- **热点前缀复制策略（Hot Prefix Replication）**  
  在非拥有节点上缓存高频访问的“热”前缀张量（replicated cache），提升访问局部性，降低远程 I/O 频率。

- **RDMA-aware 内存整合**  
  使用批量 RDMA 操作进行零拷贝数据传输，并缓存 RDMA segment 信息以避免重复 setup 开销。

### 相比现有方法的优势
| 特性 | vLLM (Vanilla/Prefixed) | LMCache / EvoStore | **PTStore (本文)** |
|------|--------------------------|--------------------|---------------------|
| 跨节点前缀共享 | ❌ | ✅（有限） | ✅✅（高效） |
| 分布式内存聚合 | ❌ | ✅ | ✅ |
| 热点前缀本地复制 | ❌ | ❌ | ✅ |
| 元数据查询效率 | 低（需遍历） | 中等 | 高（扁平化） |
| RDMA 优化 | ❌ | ✅ | ✅✅（批量+缓存） |

> ✅✅ 表示显著优于其他方案

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **WikiQA Dataset**  
  包含 3,000 个问题，每个问题关联一个 Wikipedia 页面作为上下文。页面长度为 2,000–40,000 tokens，适合测试长上下文场景下的 weak scalability。

- **SQUAD Dataset**  
  包含约 100,000 个阅读理解问题，段落长度从 150 到 4,000 词不等，适用于分析不同序列长度下 prefix reuse 的效果。

### 实验设置
- **硬件平台**：ALCF 的 Polaris HPC 测试床  
  - 560 节点，每节点配置：
    - 4× NVIDIA A100 GPU（共 160GB HBM）
    - 512GB DDR4 内存（四 NUMA 域）
    - 2× 1.6TB SSD（2GB/s）
    - 双 Slingshot 10 网络（支持 RDMA）
    - 并行文件系统（Lustre，650GB/s 聚合带宽）

- **模型与推理框架**：
  - 模型：Mistral-7B-Instruct-v2（可完全放入 A100 显存，留出 >15% 用于 KV Cache）
  - 推理引擎：vLLM v0.6
  - PTStore 通过替换 vLLM 的 KV Cache 后端集成

- **负载设置**：
  - 每个 GPU worker 执行 1,500 次推理请求（文档 + 问题对）
  - 请求采样遵循幂律分布（power law, α=6），模拟生产环境中前缀热度的长尾特性

### 评估指标
- **Average Time to First Token (TTFT)**：衡量预填充阶段效率的关键指标，尤其对于 extractive QA 类任务（输出为单 token）
- **GPU Compute Time vs. I/O Overhead**：分解计算与通信开销
- **RDMA I/O Volume**：反映远程访问频率与成本

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **vLLM Vanilla** | 默认 KV Cache Block Manager，仅使用 GPU 显存 + 主机内存 swap，无前缀感知 |
| **vLLM Prefix** | 启用前缀重用功能，可在节点内共享前缀（通过 LMCache 扩展至 node-local GPUs） |
| **EvoStore** | 当前最先进的分布式张量存储系统，支持跨节点增量存储与 RDMA 访问，但无本地复制机制 |
| **PTStore (Ours)** | 本文方法，结合分布式前缀存储 + 热点复制 + 高效元数据查询 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）弱可扩展性实验（Weak Scalability）
- 使用 **WikiQA 数据集**，平均 prompt 长度 ~8k tokens
- GPU 数量：8 → 16 → 32，保持每 GPU 请求量恒定

| 方法 | TTFT @8 GPUs | TTFT @32 GPUs | 性能趋势 |
|------|--------------|---------------|---------|
| vLLM Vanilla | ~0.8s | ~0.8s | 几乎不变 |
| vLLM Prefix | ~0.7s | ~0.7s | 微幅下降 |
| EvoStore | ~0.65s | ~0.6s | 缓慢改善 |
| **PTStore** | **~0.3s** | **~0.2s** | 显著下降，扩展性好 |

> 🔍 **发现**：随着规模扩大，PTStore 的 TTFT 持续下降，表明其分布式架构能有效聚合更多内存资源来缓存更长前缀。

#### （2）序列长度可扩展性实验（Sequence Length Scalability）
- 使用 **SQUAD 数据集**，输入长度从 1k → 8k tokens
- 固定 32 GPUs

| 序列长度 | vLLM Prefix (TTFT) | EvoStore (TTFT) | **PTStore (TTFT)** | 加速比 (vs. vLLM) |
|---------|--------------------|------------------|---------------------|-------------------|
| 1k      | ~0.4s              | ~0.35s           | **~0.15s**          | **~2.7x**         |
| 2k      | ~0.5s              | ~0.45s           | **~0.25s**          | **~2.0x**         |
| 4k      | ~0.7s              | ~0.6s            | **~0.35s**          | **~2.0x**         |
| 8k      | ~1.0s              | ~0.8s            | **~0.55s**          | **~1.8x**         |

> 📈 **趋势**：随着序列增长，PTStore 的优势持续扩大，最高达 **2.7倍加速**

#### （3）I/O 开销分解（Figure 3b & 4b）
- 在 32 GPUs 场景下：
  - **EvoStore** 的 RDMA I/O 时间占总时间 >60%，成为瓶颈
  - **PTStore** 的 RDMA I/O 占比 <20%，大部分前缀已本地缓存
  - GPU Compute 时间基本一致 → 差异主要来自 I/O 优化

> 💡 **结论**：**本地复制机制显著降低了远程访问频率，是性能提升的关键**

### 消融实验（Ablation Study）
尽管文中未明确列出独立消融图，但从设计原则和对比中可推断：
- 若移除 **replication cache** → 退化为 EvoStore，性能下降 ~2x
- 若移除 **consolidated metadata** → 查询延迟上升，影响 LCP 匹配速度
- 若禁用 **bulk RDMA + segment caching** → RDMA setup 成本升高，小张量访问变慢

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **跨节点前缀共享具有巨大潜力**：在真实 workload 下，许多请求共享相同前缀（尤其是问答类任务），现有系统未能充分利用这一特性。
2. ✅ **分布式内存聚合可将 KV Cache 容量提升数个数量级**：PTStore 成功突破了单节点内存限制。
3. ✅ **热点复制是降低远程 I/O 的关键**：单纯的分布式存储（如 EvoStore）因缺乏 locality 优化，反而可能不如本地前缀缓存。
4. ✅ **扁平化元数据结构极大提升了前缀匹配效率**：避免了传统 Radix Tree 遍历带来的同步与延迟问题。
5. ✅ **PTStore 可将 TTFT 降低 5–6 倍**，相比不聚合内存的 baseline，在长文本推理中表现尤为突出。

### 方法的局限性
- **复制缓存占用本地资源**：复制 cache 与 owned cache 竞争主机内存，需合理配置比例（目前固定为 50%）。
- **元数据一致性维护复杂度增加**：虽然当前采用去中心化设计，但在超大规模下仍可能存在协调开销。
- **尚未支持动态模型切换或多租户隔离**：目前假设模型静态部署。
- **依赖高性能网络（RDMA）**：在不具备 RDMA 的集群中性能可能打折扣。

### 未来工作方向
1. **动态内存平衡机制**：根据 workload 特征自动调整 owned vs. replicated cache 的容量分配。
2. **ML-based eviction policy**：引入机器学习预测张量未来访问概率，替代启发式策略（如 GDSF）。
3. **更全面的 ablation study**：量化 replication、metadata design、eviction policy 各组件的独立贡献。
4. **与更多推理框架集成**：如 DeepSpeed-MII、SGLang、TetriInfer 等。
5. **真实世界 workload benchmarking**：扩展到 multi-turn conversation、code completion、agent workflows 等复杂场景。
6. **与 disaggregated scheduler 联合优化**：如 DistServe、Splitwise，进一步解耦 prefill 与 decode。

---

> 🏁 **总结一句话**：  
> **PTStore 是首个同时实现“分布式前缀存储”与“热点前缀复制”的系统，通过元数据优化与 RDMA 高效通信，在长上下文 LLM 推理中实现了高达 5–6 倍的 TTFT 加速，显著推动了高吞吐推理服务的发展。**

</details>

---

### 12. [Co-Harness: Co-Evolving Harnesses and Model Weights for LLM Agents](https://arxiv.org/abs/2607.22688)

**Authors**: Zhengyu Chen, Teng Xiao, Huaisheng Zhu, Yige Yuan, Luan Zhang, Jingang Wang  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.22688v1  

#### Abstract
Post-training agents for automated AI research requires optimizing not only model parameters, but also the runtime harness that shapes how research trajectories are generated, evaluated, and learned from. Existing pipelines typically train models under a fixed harness, including prompts, tools, skil...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Co-Harness: Co-Evolving Harnesses and Model Weights for LLM Agents》核心总结**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
当前的 LLM Agent 后训练（post-training）流程通常将 **model weights** 和 **runtime harness** 分开优化：
- **Model weights** 通过 SFT、RLHF 或偏好学习进行更新；
- **Harness**（包括 prompts、tools、skills、middleware、memory 等）则被当作静态基础设施，固定不变。

这种做法导致了一个“错配”（mismatch）：harness 决定了 agent 行为轨迹（trajectory）的质量，但它本身不从失败中学习。一个错误的 tool schema、缺失的重试机制或模糊的 prompt 都可能导致整个任务失败，尤其在 **Tool-Integrated Reasoning (TIR)** 场景下更为敏感。

因此，**harness 成为了自动化 AI 研究中的瓶颈**。

---

### **提出了什么新方法或新思路**
提出 **Co-Harness** —— 一种双循环协同进化框架，**联合优化 agent 的 harness 和 model weights**。

#### 核心思想：
> **更好的 harness → 更高质量的 trajectories → 更强的模型 → 暴露新的 harness 瓶颈 → 进一步改进 harness**

该过程形成正反馈螺旋。

#### 双循环架构：
1. **Co-Harness Loop（Harness 优化）**
   - 输入：当前 model $ \theta_t $ 和 harness $ \phi_t $
   - 收集失败轨迹（failed trajectories）
   - 使用 **LLM-based HarnessCritic** 对失败归因（attribution），识别出 harness 层面的根本原因（如 `tool_schema_error`, `middleware_mismatch` 等）
   - 提出并验证局部修改（diffs），生成更优的 harness $ \phi^* $

2. **Model Alignment Loop（模型对齐）**
   - 在改进后的 harness $ \phi^* $ 下重新运行 model，收集高质量 trajectories
   - 使用这些轨迹进行 **Supervised Fine-Tuning (SFT)**，得到更强的 model $ \theta_{t+1} $
   - 新 model 再次暴露更高阶的 harness 缺陷，推动下一轮 co-evolution

---

### **相比现有方法的优势**

| 方法 | 是否优化 harness | 是否参与训练循环 | 是否 co-evolve |
|------|------------------|------------------|---------------|
| SWE-smith / AgentTuning | ❌ 固定 harness | ✅ 仅优化 model | ❌ |
| Meta-Harness / AHE [9] | ✅ 优化 harness | ❌ 仅用于 inference | ❌ |
| ADAS [15] | ✅ 搜索 agent 设计 | ❌ 不结合 model 更新 | ❌ |
| **Co-Harness (Ours)** | ✅ | ✅ | ✅ |

✅ **首次实现 harness 与 model 的端到端协同进化**  
✅ **Harness 不再是“脚手架”，而是可学习的变量**  
✅ **Harness 改进能持续提升 trajectory 质量，进而增强 model 泛化能力**

---

## 2. **核心实验方法和设置**

### **使用的数据集**
三个数学推理竞赛基准，均需多步代码调用解决：
- **AIME 2024**（30题）
- **AIME 2025**（30题）
- **HMMT February 2025**（30题，难度最高）

> 所有问题采用 **Tool-Integrated Reasoning (TIR)** 范式：模型交替使用 chain-of-thought 推理与 Python code interpreter 工具求解。

---

### **实验设置和评估指标**

#### **Base Models**
- **Qwen3-8B**
- **Qwen3-32B**

两者均具备较强的数学先验能力和 tool-augmented generation 支持。

#### **Co-evolution Rounds**
- **Round 0 (Baseline)**：仅运行 HarnessCritic 一次以获得初始优化 harness，但不做 SFT
- **Round 1 & 2**：完整执行两个 Co-Harness 循环（HarnessCritic + SFT）

#### **评估指标**
- **pass@1 accuracy (%)**：每道题多次 rollout 后取平均通过率
- **wall-clock time**：总运行时间（用于效率分析）
- **failure mode attribution**：人工标注验证 HarnessCritic 准确性（K=0.77）

#### **Harness Configuration Dimensions**
Harness 定义为五元组 $ \phi = (P, T, S, Mid, M) $：
| 维度 | 内容 |
|------|------|
| **P (Prompt)** | 系统提示词模板 |
| **T (Tool)** | 工具定义与接口 schema |
| **S (Skill)** | 可复用技能模块 |
| **Mid (Middleware)** | 主控逻辑、hook、上下文管理 |
| **M (Memory)** | 长期记忆策略 |

---

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Baseline (R0)** | Harness 经过 HarnessCritic 优化，但未进行 SFT |
| **Human-designed Harness** | 人工精心设计的静态 harness（无 model alignment） |
| **Co-Harness (Ours)** | 双循环协同进化（HarnessCritic + SFT）|

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（见 Table 6 和 Figure 5）**

| Model | Benchmark | Human | R0 (Baseline) | R1 | **R2 (Co-Harness)** | Δ(R0) |
|-------|-----------|--------|--------------|-----|--------------------|--------|
| Qwen3-8B | AIME24 | 59.3% | 63.3% | 84.0% | **84.7%** | **+21.4 pp** |
| Qwen3-8B | AIME25 | 51.3% | 56.9% | 66.7% | **78.3%** | **+21.4 pp** |
| Qwen3-8B | HMMT25 | 34.7% | 39.6% | 52.3% | **59.7%** | **+20.1 pp** |
| Qwen3-32B | AIME24 | 72.0% | 76.7% | 85.3% | **87.3%** | **+10.6 pp** |
| Qwen3-32B | AIME25 | 61.3% | 64.9% | 83.9% | **86.3%** | **+21.4 pp** |
| Qwen3-32B | HMMT25 | 46.7% | 49.8% | 68.7% | **77.0%** | **+27.2 pp** |
| **Average** | — | 54.2% | 58.5% | 73.5% | **78.9%** | **+20.4 pp** |

> ✅ **平均提升 +20.4 percentage points (pp)**  
> ✅ **最大单项目提升达 +27.2 pp（Qwen3-32B on HMMT25）**  
> ✅ **全部超越人工设计 harness（平均 +24.7 pp）**

---

### **与基线方法的对比结果**
- **vs Baseline (R0)**：所有任务上均有显著且单调增长，证明 dual-loop 有效 compounding
- **vs Human-designed Harness**：**全面超越**，说明自动化 co-evolution 能达到甚至超过人类专家水平
- **Harder benchmarks benefit more**：
  - 最难的 HMMT25 上提升最大（+27.2 pp for 32B）
  - 因复杂任务更容易暴露 harness 缺陷（如 context overflow, retry failure）

---

### **消融实验与案例研究（Case Study）**

#### **200+小时自主运行实验（AIME24, Qwen3-8B）**
完全无人干预，自动演化出 **22 个 harness 版本**，经历三阶段优化：

| 阶段 | 改进内容 | 效果 |
|------|--------|------|
| **Phase 1: Engineering Fixes** | 修复 vLLM KV cache 错误、zombie thread bug、切换 ProcessPool | 从 crash → 59.6%（首通） |
| **Phase 2: Efficiency Optimization** | 引入全局 batch inference | 速度提升 **8.7×**（3.78h → 1.11h） |
| **Phase 3: Ensemble Validation** | 多种子集成投票（6-trajectory majority voting） | 达到 **63.3%**，满足时间预算 |

> 🔁 **Rollback 安全机制生效**：v20-v22 尝试加入 domain-specific prompt 导致性能下降 ~16.5 pp，HarnessCritic 正确识别为 `prompt_ambiguity` 并回滚。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **Harness 与 model 是互依赖的优化维度**，必须共同演进才能突破性能天花板。
2. ✅ **Co-Harness 实现了 compounding self-improvement**：每轮都带来可测量的增益，且在更难任务上收益更大。
3. ✅ **HarnessCritic 具备可解释性和安全性**：
   - 能准确归因 failure 到具体 harness 组件
   - 支持 diff aggregation 与 validation，防止 regressive patches
   - 版本化 registry 支持 rollback 与审计
4. ✅ **减少 Harness Debt**：模型学到的是通用推理能力，而非对特定 harness 的依赖（test-time performance 提升证实）

---

### **方法的局限性**
- **依赖强大的 critic LLM**：HarnessCritic 本身需要足够推理能力来诊断 failure
- **冷启动需求**：需要一定数量的 failure trajectories 才能触发有效 attribution
- **计算成本高**：多轮 SFT + rollout 需要大量 GPU 时间
- **无法处理结构性 redesign**：目前只支持局部 diff 修改，重大架构变更仍需人工介入
- **attribution 在反事实推理场景下可能失效**

---

### **未来工作方向**
1. **Online Co-Harness**：实时在 rollout 中进行 failure attribution，实现运行时动态调整
2. **RL-based Harness Evolution**：用强化学习替代 LLM critic，在 harness space 中进行 reward-driven 搜索
3. **Multi-agent Co-Harness**：扩展至多 agent 系统，优化 inter-agent middleware（Mid 维度）
4. **Continuous scaffolding monitor**：将 Co-Harness 从批处理变为持续在线的“agent 健康监测器”

---

> 🧩 **一句话总结**：  
> **Co-Harness 首次实现了 LLM agent 中 harness 与 model 的双向协同进化，打破了传统“固定 harness + 训练 model”的范式，开启了 agent 自我进化的闭环路径。**

</details>

---

### 13. [Libra: Taming Attention Workload Skew in Long-Context LLM Training with Bounded Sequence Pool](https://arxiv.org/abs/2607.23250)

**Authors**: Yan Wang, Xiulong Yuan, Kaiming Yang, Jiaxuan Peng, Pengju Lu, Mingzhen Li, Zhipeng Zhang, Chang Si, Zhixiang Ruan, Hongqing Chen, Linlang Jiang, Siyu Wang, Langshi Chen, Rui Men, Man Yuan, Guangming Tan, Yong Li, Weile Jia, Jingren Zhou  
**Category**: cs.DC  
**Published**: 2026-07-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.23250v1  

#### Abstract
Long-context LLM training suffers from a load-balancing problem that sequence packing does not solve. Packing samples into fixed-token sequences balances memory and linear-cost operators, but the dominant attention cost scales with the sum of squared sequence lengths. Thus, equally sized packed sequ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Libra: Taming Attention Workload Skew in Long-Context LLM Training with Bounded Sequence Pool**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在长上下文大语言模型（LLM）训练中，尽管 **sequence packing** 技术可以平衡内存和线性算子负载，但由于 **attention 计算成本随序列长度平方增长**，不同打包样本间的 attention 工作负载仍存在严重倾斜（workload skew）。这种不均衡导致：
- **Data Parallel (DP)** 中出现同步延迟（straggler）
- **Pipeline Parallel (PP)** 中产生计算气泡（pipeline bubbles）
- 整体集群吞吐量显著下降，扩展效率低下（如仅实现 27.6% 的理想线性扩展）

现有方法存在以下不足：
- **基于序列或微批次的调度**：无法处理单个 outlier 样本主导负载的问题；
- **全局 attention disaggregation（如 DistCA）**：通信域随 DP 规模扩大，跨低带宽链路开销高。

---

### 🚀 提出的新方法与核心思想

**Libra** 提出了一种基于 **有界序列池（bounded sequence pool）** 的 attention 负载均衡框架，其三大核心技术为：

#### （1）**LLN 指导的序列池设计（LLN-guided Sequence Pooling）**
- 创新性地将 **大数定律（Law of Large Numbers, LLN）** 引入分布式训练负载均衡。
- 核心洞察：**attention 平衡池无需随 DP 规模增大而扩大**。
- 设计原则：固定每个 sequence pool 的大小 $ P $，当 DP 扩展时，增加 pool 数量而非单个 pool 尺寸。
- 优势：限制了每次 attention 任务迁移的通信范围，天然支持弱扩展（weak scaling），并可限定在高带宽域内执行。

#### （2）**方差缩减序列放置（Variance-Reduced Sequence Placement, VRSP）**
- 在每个优化步窗口内，对打包序列进行重排序，使每个 sequence pool 的总 attention FLOPs 接近均值。
- 使用改进的 **最长处理时间优先（LPT）启发式算法**，确保每池恰好分配 $ P $ 个序列，并显式降低跨池方差。
- 保持原始样本集合不变，满足 step-equivalence 要求。

#### （3）**分块注意力池化（Tiled Attention Pooling, TAP） + 流水线运行时**
- 将 attention 分解为 **SH-Tile（Sequence × Head Tile）** 单位，作为独立可调度任务。
- 引入 **通信感知的任务放置器（Communication-aware placer）**，在考虑 Q/KV 数据复用的前提下平衡各 GPU 的计算负载。
- **TAP Pipeliner** 将 tile 交换划分为多个 chunk，并与 FlashAttention 计算流水线重叠，隐藏通信开销。

---

### 🔍 相比现有方法的优势

| 维度 | Libra | 现有方法（如 Ulysses / WLB-LLM / DistCA） |
|------|-------|------------------------------------------|
| **负载粒度** | SH-Tile 级细粒度调度 | 序列级或 microbatch 级 |
| **通信范围** | 固定大小 pool 内部（bounded） | 全局池（grows with DP）或无迁移 |
| **语义一致性** | 保留 optimizer step 内样本多集 | WLB-LLM 延迟 outlier 改变样本分布 |
| **集成难度** | 插件式接口（drop-in attention op + sampler） | 需修改 pipeline schedule 或执行逻辑 |
| **扩展性** | 弱扩展友好，pool 可局部部署 | 全局通信制约扩展能力 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用两个真实生产级长上下文数据集：
  - **256K 数据集**：平均序列长度短，尾部长达数万 token。
  - **1M 数据集**：更极端长尾分布，p99 达到 ~71K tokens。
- 均采用 **fixed-token packing** 构造 256K 和 1M token 的 packed sequences。

### ⚙️ 实验设置
- **硬件平台**：NVIDIA GPU 集群，NVLink 节点内互联，RoCE 跨节点网络。
- **模型配置**：
  - 主要测试 **Qwen3-Turbo（Qwen3-30B-A3B）** 模型。
  - 微基准测试使用合成 attention 配置（h=128, d=256）。
- **并行策略组合**：
  - DP ∈ {1,2,4,8,16}
  - CP = 8 (256K), 16 (1M)
  - PP = 1 或 2
  - GBS = 128 (256K), 32 (1M)

### 📊 评估指标
| 类型 | 指标 |
|------|------|
| **端到端训练** | Tokens per GPU per second（吞吐量）、DP 扩展效率 |
| **微基准测试** | 每步最慢 worker 的 attention 延迟（straggler latency），报告 mean 与 max |
| **负载均衡性** | 跨池 FLOPs 不平衡率（inter-pool imbalance）、intra-pool 负载差异 |

### 🆚 基线方法对比
| 基线 | 说明 |
|------|------|
| **Ulysses [7]** | 当前主流 CP 方案，head-axis 分片，但不解决跨 CP 组负载不均 |
| **WLB-LLM [24]** | 变长 packing 实现 workload balance，但改变内存占用且不能拆分 outlier |
| **DistCA [33]** | 全局 attention disaggregation，模拟其平衡范围（P=DP）用于比较 |
| **Libra 变体** | 消融实验：<br>• Libra (TAP)<br>• Libra (TAP+VRSP)<br>• Libra (Full) |

---

## 3. 主要实验结果和性能指标

### 📈 端到端性能提升（vs. Ulysses）

#### （1）DP 扩展效率大幅提升
| 设置 | 基线（Ulysses） | Libra | 提升倍数 | 扩展效率 |
|------|------------------|--------|-----------|------------|
| **256K, DP=16** | 7.63× | **13.66×** | **1.79×** | 47.7% → **85.4%** |
| **1M, DP=16** | 4.42× | **11.25×** | **2.54×** | 27.6% → **70.3%** |

> 图 11 显示 Libra 几乎实现了近似线性扩展，尤其在更长上下文（1M）下优势更为明显。

#### （2）PP 气泡显著压缩
- 测量每个 microbatch 的前向耗时分布（图 12）：
  - **1M 数据集上**：
    - 基线最大耗时是均值的 **23.1 倍**
    - Libra 压缩至 **1.57 倍**
    - **最坏 microbatch 性能提升 14.7×**
  - **256K 上**：从 6.98× 降至 2.63×（**2.6× 改善**）

> 表明 Libra 有效缓解了 pipeline bubbles。

---

### 🔬 微基准测试结果（core attention 层）

#### （1）attention 层延迟降低（图 13）
| 指标 | 256K（vs. Ulysses） | 1M（vs. Ulysses） |
|------|----------------------|-------------------|
| **Mean Latency** | ↓65.6% | ↓65.6% |
| **Max Latency（straggler）** | ↓68.3% | ↓65.6% |
| **最坏步长速度提升** | **3.14×** | **2.90×** |

> Libra(Full) 在所有指标上均优于其他方法。

#### （2）与其他基线对比
- **vs. WLB-LLM**：虽能改善平均延迟，但 **max 延迟几乎未变**（outlier 无法拆分）。
- **vs. DistCA（模拟 P=DP）**：
  - 平均延迟更高 → 全局通信开销大；
  - max 延迟略优但不如 Libra；
  - 通信成本随 DP 增长不可持续。

---

### 🔍 消融实验分析

#### （1）VRSP 显著降低跨池不平衡（表 1 & 图 8）
| Pool Size P | 256K（Before → After） | 1M（Before → After） |
|------------|-------------------------|------------------------|
| P=8        | 1.08 → **0.005**         | 0.52 → **0.007**        |
| P=16       | 0.56 → **0.0006**        | 0.20 → **0.0009**       |

> VRSP 将跨池 FLOPs 不平衡从 >50% 降到 <1%，效果显著。

#### （2）TAP 内部平衡与通信权衡（图 15）
- **Head-axis splitting 比 sequence-axis 更高效**：
  - 相同 tile 数量下，head split 引入更少 KV 传输（更好复用）。
  - 在 256K 上，H=4, B=8192 比 B=2048, H=1 快 **1.89× vs. 1.76×**。
- **过小的 block size 导致性能下降**：KV 复用被破坏。

#### （3）Pipeliner 有效隐藏通信（图 14）
- 通信开销随 pool size 增加急剧上升（P=32 时通信占 77%）。
- Pipeliner 可隐藏大部分中间通信，但首尾暴露。
- 在 1M 上，P=8 时通信节省达 **54%**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **大数定律可用于指导分布式训练中的负载均衡设计**：固定大小的 sequence pool 即可在统计意义上平滑 workload，无需全局聚合。
2. **attention 负载均衡应“横向扩展”而非“纵向放大”**：增加 pool 数量比扩大单个 pool 更可持续。
3. **SH-Tile + Head-axis 分块是实现高效通信/计算平衡的关键**：相比 sequence 分块，head 分块更利于通信重叠和 KV 复用。
4. **VRSP + TAP + Pipeliner 形成闭环优化**：
   - VRSP 控制跨池偏差，
   - TAP 平衡池内负载，
   - Pipeliner 隐藏迁移开销。

---

### ⚠️ 方法的局限性
- **依赖 fixed-token packing**：未直接优化 packing 策略，但可与其正交结合。
- **CPU 规划器异步执行**：大规模下规划延迟可能成为瓶颈（当前未观察到）。
- **未处理动态变化的工作负载分布**：假设 GBS 窗口内分布稳定。
- **未支持完全动态的 pool 成员变更**：pool membership 固定于训练过程。

---

### 🔮 未来工作方向
- 将 bounded pooling 思想推广至 **MoE routing、gradient all-reduce** 等其他非均匀负载场景。
- 探索 **自适应 pool size 调整机制**，根据实时 workload 动态调节 $ P $。
- 结合 **learning-based scheduler** 进一步优化 tile placement。
- 扩展至 **推理阶段的长上下文服务调度**。

---

## ✅ 总结一句话
> **Libra 通过引入“有界序列池 + LLN 指导 + SH-Tile 调度”的新范式，在不改变训练语义的前提下，实现了高达 2.54× 的端到端吞吐提升和 3.14× 的最慢 attention 步骤加速，为长上下文 LLM 训练提供了高效、可扩展且易于部署的负载均衡解决方案。**

</details>

---

### 14. [CuraWeb: Joint Optimization of Quality, Redundancy, and Diversity for Web-Scale Pretraining Data](https://arxiv.org/abs/2607.22662)

**Authors**: Peiguang Li, Yongwei Zhou, Juncheng Diao, Yuchun Fan, Jian Yang, Jianxiao Yang, Zhongda Su, Shuguang Jiao, Xiao Wei, Zhiye Zou, Gan Dong, Zhizhao Zeng, Rongxiang Weng, Jingang Wang, Xunliang Cai  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.22662v1  

#### Abstract
Open-web corpora curated via highly selective filters, such as FineWeb-Edu and DCLM, constitute the core of LLM pretraining data and have significantly advanced LLM performance. However, these pipelines typically rely on singular optimization objectives, which inevitably narrows distributional diver...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CuraWeb: Joint Optimization of Quality, Redundancy, and Diversity for Web-Scale Pretraining Data

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前主流的 LLM 预训练数据构建流程（如 FineWeb-Edu、DCLM）通常采用**单一目标优化**策略，即通过统一的质量过滤器对网页数据进行线性剪枝。这种做法虽然提升了文本的表面质量，但也带来了以下严重问题：

- **窄化分布多样性**（Narrowed distributional diversity）：过度依赖单一质量信号导致长尾领域（如数学、代码、科学）的知识被系统性丢弃。
- **冗余未被充分处理**：传统基于 n-gram 的去重方法无法识别语义重复（如模板化网页、跨时间爬取的微小变体）。
- **质量-多样性权衡失衡**：清洗、去重、采样各阶段孤立操作，缺乏全局协同，难以兼顾高质量与广覆盖。

这些问题限制了预训练数据的有效性，制约了模型在知识密集型和推理任务上的潜力。

---

### 🚀 提出的新方法与创新思路

作者提出 **CuraWeb**，一种全新的**联合优化范式**，将质量（quality）、冗余（redundancy）和多样性（diversity）作为多目标进行协同治理。其三大核心技术创新如下：

#### （1）**双轨清洗机制（Dual-track Cleaning）**

- **规则清洗（Rule-based）**：引入**领域自适应阈值**，避免对 STEM、数学、代码等高符号密度内容的误删（例如放宽 `alphabetic_char_ratio` 从 <0.8 → <0.26）。
- **模型驱动清洗（Model-driven）**：利用轻量级多任务模型（300M 参数）对文档进行细粒度评分，涵盖 writing_score、coherence_score、knowledge_score 等 9 个维度，实现更精准的质量判断。
- **优先级旁路机制**：标记为 Mathematics、STEM、Code 的文档直接跳过规则清洗，防止高价值专业内容被误伤。

#### （2）**混合去重管道（Hybrid Deduplication）**

- **第一阶段：n-gram 模糊去重**（MinHash-LSH），高效去除表面近似重复。
- **第二阶段：软语义去重**（Soft Semantic Deduplication）
  - 发现两类深层冗余：**Templatized Clusters**（结构相同仅参数不同）和 **Redundant Crawling**（跨时间微小更新）。
  - 提出**加权投票机制**：不依赖单一边缘相似度过滤，而是累积多个相似边的惩罚分，仅当总分超过阈值才判定为冗余。
  - 显著降低高相似区间（如 [0.99,1.00]）的假阳性率（从 37.5% → 28.03%）。

#### （3）**多目标采样框架（Multi-objective Sampler）**

- **多样性得分**：基于 K-Means 聚类，结合簇内紧凑性和簇间分离性计算。
- **内容价值得分**：聚焦 knowledge_score、education_score、helpfulness_score、reasoning_score，奖励信息密度高的文档。
- **Power Sampling**：使用非线性放大机制增强高质量样本的选择概率，优于传统的 Softmax 采样。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（如 FineWeb-Edu, DCLM） | CuraWeb |
|------|-------------------------------|--------|
| **优化目标** | 单一质量导向 | 多目标联合优化（质量 + 多样性 + 去冗） |
| **去重能力** | 仅 n-gram 表面匹配 | 支持语义层面软去重，保留长尾技术文档 |
| **领域覆盖** | 倾向教育/通用内容，牺牲 STEM | 主动保护数学、编程、科学等长尾领域 |
| **数据效率** | 存在过度过滤与重复训练 | 更高 token 效率，扩展性更强 |

---

## 2. 核心实验方法和设置

### 📊 数据集

- **原始数据源**：Common Crawl（2013–2024 年共 12 年快照）
- **构建成果**：**CURAWEB** 英文语料库，规模达 **2T tokens**
- **对比基线**：
  - **FineWeb-Edu**：高质量教育导向数据集
  - **DCLM**：DataComp-LM，强调任务性能的数据筛选
  - **Dolma3**：大规模开放语料
  - **Nemotron-CC**：注重长上下文的精炼数据

所有对比均控制在相同 token 预算下（200B 和 1T tokens）。

---

### ⚙️ 实验设置

- **模型架构**：3B 参数规模，基于 LLaMA 架构（RMSNorm, SwiGLU, RoPE, GQA）
- **训练配置**：
  - 序列长度：8192
  - 优化器：Adam（β₁=0.9, β₂=0.95）
  - 学习率：3e-4（余弦衰减）
  - 全局 batch size：960
- **训练预算**：200B tokens（主实验），部分延伸至 1T tokens 分析缩放行为

---

### 📈 评估指标

使用 `lm-evaluation-harness` 框架，在以下 **10 个基准**上评估：

| 类别 | Benchmark |
|------|---------|
| 数学推理 | GSM8K-Platinum, MathQA |
| 多学科知识 | MMLU (5-shot), MMLU-Pro (5-shot) |
| 阅读理解 | RACE |
| 常识推理 | HellaSwag, PIQA, Winogrande |
| 科学问答 | SCIQ, OpenBookQA |
| 综合性能 | 所有任务的 **unweighted macro-average accuracy**

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（200B tokens 下平均准确率）

| 方法 | 平均准确率 |
|------|----------|
| DCLM | 46.25% |
| FineWeb-Edu | 44.26% |
| Nemotron-CC | 45.73% |
| Dolma3 | 45.30% |
| **CuraWeb（本文）** | **48.07%** ✅ |

👉 **绝对提升 +1.82%**，显著超越所有基线，达到公开语料中的 SOTA。

---

### 🔍 与基线方法的对比结果

#### （1）**推理与知识密集型任务优势明显**

| Task | CuraWeb vs 最佳基线 |
|------|---------------------|
| GSM8K | +4.39%（vs Dolma3） |
| MMLU | +6.61%（vs Dolma3） |
| MMLU-Pro | +2.82%（vs Dolma3） |

表明 CuraWeb 更有利于系统性推理能力的发展。

#### （2）**缓解领域偏倚，保持均衡表现**

- FineWeb-Edu 在 OBQA 上表现好（44.20%），但在 MMLU 上严重下降（28.02%）
- CuraWeb 在几乎所有任务中均为 **第一或第二**，无明显短板。

#### （3）**成功恢复 STEM 内容**

| Task | 提升幅度 |
|------|--------|
| MathQA | +0.47%（vs FineWeb-Edu） |
| SCIQ | +0.70%（vs FineWeb-Edu） |

验证了其对科学文本的召回机制有效。

---

### 🔬 消融实验结果（Ablation Studies）

| 组件 | 影响 |
|------|------|
| **完整过滤管道**（vs 仅规则过滤） | 保留的高质量数据量翻倍，训练效率不变 |
| **移除语义去重** | 性能曲线整体下移，说明模板化冗余损害学习效果 |
| **替换为均匀采样** | 明显性能下降，证明 Power Sampling 必要性 |

✅ 三个组件均对最终性能有独立且正向贡献。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **联合优化优于单一目标过滤**：将质量、冗余、多样性纳入统一框架，可显著提升 token 利用效率。
2. **长尾知识值得保留**：适当放松规则阈值并辅以模型判别，可在不牺牲收敛性的前提下大幅增加高价值 STEM 文档。
3. **语义去重至关重要**：传统 n-gram 去重不足以应对深层冗余，软投票机制能有效减少误删。
4. **采样策略决定分布形状**：Power Sampling 成功将权重向高信息密度领域倾斜（如 Education +6.53%，Health +5.18%）。

---

### ⚠️ 局限性

- **计算成本较高**：多信号标注与语义聚类需要大量计算资源，可能不适合小团队复现。
- **依赖强监督信号**：质量评分由 GPT-4o 生成，存在标注偏差风险。
- **未探索多语言场景**：目前仅构建英文语料，国际化支持有待验证。

---

### 🔮 未来工作方向

- 将 CuraWeb 框架扩展至 **多语言语料构建**
- 探索 **动态调整采样权重** 以适配不同下游任务
- 结合 **feedback-driven iteration** 实现数据-模型协同进化
- 开源完整的 **CuraWeb 数据集** 以推动社区发展

---

> 💡 **总结一句话**：  
> CuraWeb 通过“**领域感知清洗 + 软语义去重 + 内容价值驱动采样**”三重创新，实现了高质量、低冗余、高多样性的 Web 规模预训练数据构建，在 3B 模型上全面超越主流基线，为未来数据工程提供了工业级新范式。

</details>

---

### 15. [Training Language Models to Cooperate with Inference-Time Controllers](https://arxiv.org/abs/2607.23771)

**Authors**: Moumita Choudhury, Vanshaj Khattar, Jing Liu, Toshiaki Koike-Akino, Ankush Chakrabarty, Shlomo Zilberstein, Ye Wang  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.23771v1  

#### Abstract
Large language model (LLM) performance increasingly depends not only on the base model, but also on the inference-time controller used to organize reasoning. Existing post-training methods, however, typically optimize for a single fixed interaction pattern, despite real deployments relying on divers...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Training Language Models to Cooperate with Inference-Time Controllers*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Large Language Model (LLM)** 后训练（post-training）方法通常针对**单一固定的推理流程**（如 Chain-of-Thought、Self-Refine 等）进行优化。然而，在实际部署中，系统会使用多种不同的 **inference-time controller**（控制器），例如 CoT、Self-Consistency、Debate、Planning、Verification 等。

这种“**训练-部署不匹配**”导致模型在未见过的 controller 上泛化能力差，出现过拟合于特定流程的现象。

> **核心问题**：如何让基础模型在后训练阶段就学会与**一系列多样化的控制器协作**，而非仅适配某一种？

---

### 🚀 提出的新方法：CALM（Controller-Aware Language Models）

作者提出 **CALM** 框架，将控制器明确纳入训练循环，实现对多控制器的联合优化。

#### 主要创新点：

1. **将 controller-aware post-training 形式化为多任务强化学习问题**  
   - 每个 controller 定义一个 **Markov Decision Process (MDP)**。
   - 训练目标是在多个 controller-induced MDPs 上学习一个**共享策略**（shared policy），提升跨流程泛化能力。

2. **引入模块化（modular）控制器结构**
   - 将每个 controller 分解为可复用的 **local reasoning modules**（如 COT、CRITIC、ABSTRACTOR、DEBATER、JUDGE 等）。
   - 控制器是这些模块的不同组合方式（composition）。
   - 假设：若模型能学好这些通用模块，则可在**未见过的控制器组合**中表现良好。

3. **支持模块级训练策略分解**
   在 turn-level GRPO 目标下，探索四种模块感知的训练变体：
   - **CALM-MIXED**：所有模块混合训练（无显式分解）
   - **CALM-UNIFORM**：各模块类型梯度权重均等
   - **CALM-SINGLE**：每步只训练一种随机选择的模块类型
   - **CALM-ADAPTIVE**：基于历史梯度范数动态平衡模块权重（类似 GradNorm）

> 这种结构允许系统性研究不同训练策略对跨控制器迁移的影响。

---

### 🔍 相比现有方法的优势

| 对比维度 | 传统方法 | CALM |
|--------|--------|------|
| 训练目标 | 单一固定 controller | 多 controller 分布 |
| 泛化能力 | 易过拟合特定流程 | 支持 held-out 组合与结构偏移 |
| 模块复用 | 无显式建模 | 显式识别并利用模块共性 |
| 可扩展性 | 需重新训练适配新流程 | 更可能适应新 controller 组合 |

---

## 2. 核心实验方法和设置

### 📚 数据集

- **训练数据**：`GSM8K`（2,048 条数学应用题）
- **测试数据**：
  - **Experiment 1**：`GSM8K` 测试集（评估 in-distribution 性能）
  - **Experiment 2**：`MATH500` 和 `AMC2023`（out-of-distribution 数学竞赛题，更具挑战性）

### ⚙️ 模型与训练细节

- **基础模型**：`LLaMA-3.2-3B-Instruct`
- **训练方法**：基于 **turn-level GRPO**（Group Relative Policy Optimization）的强化学习
- **KL 正则项**：防止偏离原始参考策略
- **格式奖励**（format reward）：鼓励输出符合预定义 JSON schema（避免语法错误影响主任务）

### 🧪 控制器设计（Controllers）

共定义 12 种控制器，分为三类：

| 类别 | 示例 | 描述 |
|-----|------|------|
| **Training Controllers (B1–B6)** | Chain-of-Thought (B1), Self-Refine (B2), LLM-Debate (B3) 等 | 训练时可见的控制器 |
| **Compositional Controllers (H1–H3)** | Step-back + Refine (H2), Debate + Refine (H3) | 由训练中已知模块组成的新组合 |
| **Controller-Shift Controllers (H4–H6)** | Dynamic Role-Playing, Reflective Verifier 等 | 引入**全新模块类型**（如 TEACHER, STUDENT, VERIFIER） |

> 所有控制器以 Python 函数形式实现，遵循 “code-as-controller” 范式（来自 ADAS 框架）。

---

### 📊 评估指标

- **主要指标**：最终答案正确率（binary accuracy, R ∈ {0,1}）
- **辅助指标**：格式合规性得分（轻量级 format reward）
- **评估场景**：
  - 在训练控制器上的平均性能
  - 在 held-out 组合控制器上的泛化能力
  - 在引入新模块类型的 controller shift 下的鲁棒性
  - 跨数据分布（MATH/AMC）的迁移能力

---

### 🆚 基线方法对比

| 基线类型 | 具体配置 |
|---------|--------|
| **Single-controller baselines** | 分别在 B1–B6 上单独训练的模型（如 CoT-only、Self-Refine-only） |
| **Base model (zero-shot)** | 未经 RL 微调的基础模型直接运行各 controller |

> 所有 baseline 使用相同训练目标（turn-level GRPO），确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 Experiment 1：GSM8K 上的性能对比（Table 2）

| 模型 | Training Avg (%) | Compositional Avg (%) | Controller-Shift Avg (%) |
|------|------------------|------------------------|----------------------------|
| 最佳 single-controller (Step-back B4) | 71.11 | 68.31 | 74.10 |
| **CALM-MIXED** | **73.45** | **70.31** | **75.69** |
| **CALM-ADAPTIVE** | **73.72** | **70.68** | 69.88 |
| CALM-SINGLE | 71.49 | 65.78 | 75.46 |
| CALM-UNIFORM | 73.20 | 69.22 | 73.51 |

#### 关键发现：
- 所有 CALM 变体均优于最佳 single-controller baseline。
- **CALM-ADAPTIVE** 在训练和组合控制器上表现最好，但在 controller shift 下显著下降（↓4.22%），表明其可能过拟合训练模块分布。
- **CALM-MIXED** 在 controller shift 场景下最强（75.69%），显示更强鲁棒性。
- Single-controller 模型严重过拟合：如 Debate(B3) 在自身流程达 71.87%，但在 CoT(B1) 上仅 40.86%。

---

### 📉 Experiment 2：Out-of-Distribution 性能（Table 3）

在更难的 `MATH500` 和 `AMC2023` 上测试：

| Dataset | Method | Training Avg | Compositional Avg | Controller-Shift Avg |
|--------|--------|--------------|--------------------|-----------------------|
| **MATH500** | CoT(B1) | 38.30 | 33.80 | 38.50 |
| | **CALM-MIXED** | 38.90 | 35.40 | **40.10** |
| | CALM-ADAPTIVE | 38.90 | 35.10 | 39.20 |
| **AMC2023** | CoT(B1) | 16.25 | 18.75 | 22.50 |
| | **CALM-MIXED** | **20.00** | **26.25** | **23.75** |
| | CALM-UNIFORM | 15.00 | 15.00 | **26.25** |
| | CALM-ADAPTIVE | 18.75 | 13.75 | 25.00 |

#### 关键观察：
- 在极具挑战性的 `AMC2023` 上，**CALM-MIXED 全面领先**，尤其在 compositional 控制器上大幅超越 CoT（↑7.5%）。
- CALM-ADAPTIVE 在 compositional 场景下表现最差（13.75%），说明自适应梯度平衡可能损害泛化。
- 表明 multi-controller 训练不仅提升泛化，还能迁移到更复杂、分布外的任务。

---

### 🔬 消融实验结果（Ablation Study）

- **模块分解是否必要？**
  - 是。相比 single-controller，multi-controller 训练显著提升跨流程性能。
  - 但并非所有模块加权策略都有效：
    - **CALM-MIXED**（简单混合）在 controller shift 下表现最佳。
    - **CALM-ADAPTIVE** 在训练内表现优，但鲁棒性差。
    - **CALM-SINGLE** 因丢弃其他模块数据，在长流程组合中表现弱。

- **关键结论**：没有一种策略在所有场景下最优，**应根据预期部署环境选择训练策略**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **存在严重的训练-部署不匹配问题**  
   单一 controller 后训练会导致模型严重过拟合，无法迁移到其他推理流程。

2. **CALM 显著提升跨控制器泛化能力**  
   多 controller 联合训练使模型能在**未见过的组合流程**中保持高性能。

3. **模块化结构支持系统性训练策略研究**  
   模块级梯度分解揭示了不同加权策略的 trade-off：
   - 均衡加权（uniform）有助于稳定性
   - 自适应加权（adaptive）可能过拟合训练分布
   - 简单混合（mixed）在分布偏移下最鲁棒

4. **泛化能力可迁移到 OOD 数据集**  
   在 MATH 和 AMC 等高难度数学任务上，CALM 仍表现出一致优势，尤其是 **CALM-MIXED**。

---

### ⚠️ 局限性

1. **训练不稳定问题依然存在**  
   多轮次 RL 存在 credit assignment 困难，尤其当轨迹较长时（如 Self-Refine）。作者承认未完全解决此问题。

2. **模块定义依赖人工先验**  
   当前模块划分基于已有文献中的常见模式，尚未实现自动发现模块边界。

3. **控制器搜索空间有限**  
   实验仅覆盖约十几个控制器，真实世界中控制器组合空间巨大。

4. **CALM-ADAPTIVE 的鲁棒性风险**  
   动态梯度平衡虽在训练内有效，但在分布外反而表现更差，需谨慎使用。

---

### 🔮 未来工作方向

1. **自动化模块发现机制**  
   探索从 controller 行为中自动聚类出可复用 reasoning modules。

2. **结合 controller 搜索框架**  
   与 ADAS、Archon 等自动设计 controller 的方法协同优化：既优化 controller，也优化 model 以配合之。

3. **更复杂的 credit assignment 机制**  
   引入 turn-level 或 module-level 的 reward shaping，缓解终端奖励下的信用分配难题。

4. **扩展到非数学任务**  
   验证 CALM 在代码生成、规划、工具调用等 agent 任务中的有效性。

5. **在线 adaptation 机制**  
   探索无需参数更新即可快速适应新 controller 的轻量级方法（如 prompt tuning + CALM）。

---

> 💡 **一句话总结**：  
> CALM 提出了一种面向“控制器家族”的 LLM 后训练范式，通过模块化多任务 RL 实现更强的推理流程泛化能力，为构建真正灵活、可组合的 LLM agent 奠定了基础。

</details>

---

### 16. [From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement](https://arxiv.org/abs/2607.23802)

**Authors**: Qinsi Wang, Jing Shi, Huazheng Wang, Kun Wan, Yiran Wu, Bo Liu, Qingyun Wu, Hai Helen Li, Yiran Chen, Handong Zhao, Wentian Zhao  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.23802v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has driven recent progress in reasoning-oriented large language models (LLMs) by enabling large-scale optimization. However, its applicability remains largely limited to domains such as mathematics and coding, where correctness can be determinist...

---

### 17. [A Motion-Aware Vector Quantization Framework with Centroid Reuse for Efficient VLA Inference](https://arxiv.org/abs/2607.24148)

**Authors**: Zhuoran Song, Haozhe Jiang, Chunyu Qi, Minnan Pei, Gang Li, Xiaoyao Liang, Haibing Guan  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.24148v1  

#### Abstract
Vision-Language-Action (VLA) models have demonstrated strong potential for embodied AI, yet their high inference latency on GPUs limits real-time deployment. Existing accelerators, such as Dadu-Corki, improve efficiency but treat VLA models as full-precision workloads, leaving substantial redundancy...

---

### 18. [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)

**Authors**: Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, M. C., Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y. Charles, H. S. Che, Guanduo Chen, Guangyu Chen, Guanzheng Chen, Huarong Chen, Jia Chen, Jianlong Chen, Jun Chen, Kexin Chen, Peng Chen, Ruijue Chen, Wentao Chen, Xin Chen, Yang Chen, Yanru Chen, Yifei Chen, Yingjiang Chen, Yuankun Chen, Yujie Chen, Yutian Chen, Zhirong Chen, Dazhi Cheng, Yean Cheng, Jialei Cui, Jingbing Cui, Anqi Dai, Jiaqi Deng, Hao Ding, Rui Ding, Shaofeng Ding, Mengfan Dong, Mengnan Dong, Yuhao Dong, Yuxin Dong, Angang Du, Chenzhuang Du, Dikang Du, Jusen Du, Yulun Du, Yu Fan, Jing Feng, Qiulin Feng, Yichen Feng, Kelin Fu, Qiang Fu, Fuxuan Gao, Hongcheng Gao, Jingyue Gao, Tong Gao, Weijia Gao, Shangyi Geng, Jie Gong, Linhu Gong, Shengao Gong, Xiaochen Gong, Qizheng Gu, Yicheng Gu, Shuhao Guan, Haiqing Guo, Shiqi Guo, Xiang Guo, Zhengyan Guo, Beixi Hao, Wenxin Hao, Xiaoru Hao, Dailan He, Haotian He, Lehan He, Qi He, Weiran He, Xinran He, Xinyi He, Yibo He, Yunjia He, Chao Hong, Tiange Hong, Hao Hu, Jiaxi Hu, Ruikun Hu, Weiming Hu, Yangyang Hu, Zhenxing Hu, Liang Hua, Jinbin Huang, Ke Huang, Ruiyuan Huang, Siying Huang, Weixiao Huang, Yan Huang, Zhengjie Huang, Zhiqi Huang, Yulong Hui, Chaobo Jia, Yutong Jiang, Zhejun Jiang, Zuoyou Jiang, Wenyi Jin, Xinyi Jin, Yu Jing, Huanjun Kong, Guokun Lai, Aidi Li, Cheng Li, Chengyuan Li, Cong Li, Fang Li, Guanyu Li, Haoyang Li, Jia Li, Junxiong Li, Lei Li, Letian Li, Lincan Li, Weihong Li, Wentao Li, Xintong Li, Yang Li, Yishen Li, Yiwei Li, Yuxiao Li, Zhaowei Li, Zhaoxi Li, Zheming Li, Zhengxiao Li, Zhiyuan Li, Jiawei Lin, Xiaohan Lin, Yibo Lin, Zichao Lin, Ziyan Lin, Bill Liu, Boxiao Liu, Chuan Liu, Liang Liu, Shaowei Liu, Shudong Liu, Shuran Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yanming Liu, Yibo Liu, Yipeng Liu, Zhengying Liu, Zhiheng Liu, Enzhe Lu, Haoyu Lu, Linqiang Lu, Tingzhan Lu, Zhiyuan Lu, Aotian Luo, G. Luo, Junyu Luo, Yifan Luo, B. Lyu, Wenzhou Lyu, Shaoguang Mao, Yuan Mei, Xin Men, Minqing Ni, Yixuan Niu, Siyuan Pan, Shujun Peng, Zhangyang Qi, Ruoyu Qin, ZeChao Qin, Zeyu Qin, Haiquan Qiu, Jianxin Qiu, Jiezhong Qiu, Bowen Qu, Yuhao Qu, Zeyu Shang, Youbo Shao, Han Shen, Jincheng Shi, Juanfeng Shi, Lidong Shi, Shengyuan Shi, Wingchun Siu, Pengwei Song, Xiaoxi Song, Jianlin Su, Yunfeng Su, Zhaochen Su, Lin Sui, Jingsong Sun, Junyao Sun, Shaoning Sun, Shuzhe Sun, Tongyu Sun, Yujun Sun, Yunpeng Tai, Chuning Tang, Heyi Tang, Sirui Tang, Zecheng Tang, Chaoran Tian, Rongpeng Tian, Yu Tian, Wei Tu, Chensi Wang, Chuang Wang, Chunjie Wang, Dinglu Wang, Feng Wang, Hailong Wang, Haiming Wang, Hao Wang, Hao Wang, Huaqing Wang, Hui Wang, Jiayi Wang, Jinglong Wang, Jinhong Wang, Jiuzheng Wang, Linian Wang, Shaobo Wang, Shenzhi Wang, Shuyi Wang, Si Wang, Siyuan Wang, Tianfu Wang, Wenjue Wang, Xingran Wang, Xinmei Wang, Xinyuan Wang, Xusheng Wang, Yalin Wang, Yangkun Wang, Yao Wang, Yaoyu Wang, Yejie Wang, Yiqin Wang, Yucheng Wang, Yuzhi Wang, Zhaoji Wang, Zhaowei Wang, Zhengtao Wang, Zhenhao Wang, Zhongsheng Wang, Zifan Wang, Chu Wei, Ming Wei, Shouxin Wei, Zichen Wen, Fan Wu, Haoning Wu, Rucong Wu, Wenhao Wu, Xiaoxue Wu, Yingcong Wu, Yongqi Wu, Yuxin Wu, Zijian Wu, Xinglang Xian, Chenxuan Xiang, Yuye Xiang, Bocheng Xiao, Chenjun Xiao, Xin Xiao, Jin Xie, Xiaotong Xie, Yifeng Xie, Zhe Xie, Bowei Xing, Yiming Xiong, Baosheng Xu, Boyu Xu, Jiale Xu, Jianfan Xu, Jing Xu, Jinjing Xu, L. H. Xu, Qingtao Xu, Shuyao Xu, Suting Xu, Tiantian Xu, Tianxiang Xu, Weixin Xu, Xinran Xu, Yangchuan Xu, Ye Xu, Yueni Xu, Ziyao Xu, Haonan Xue, Junjie Yan, Yaoyao Yan, Fan Yang, Guangyao Yang, Hao Yang, Junwei Yang, Ruoyu Yang, Wenjie Yang, Xiaofei Yang, Xinyu Yang, Yi Yang, Yiling Yang, Ying Yang, Yuchen Yang, Zhen Yang, Zhilin Yang, Zian Yang, Zuhao Yang, Haotian Yao, Dan Ye, Haoran Ye, Wenjie Ye, Zhanbo Ye, Bohong Yin, Haoxiang Yin, Xietong Yin, Chengzhen Yu, Haozhen Yu, Longhui Yu, Shengnan Yu, Shuying Yu, Tianxiang Yu, Enming Yuan, Mengjie Yuan, Tongtian Yue, Wei Yue, Yang Yue, Dunyuan Zha, Haobing Zhan, B. H. Zhang, Dehao Zhang, Fei Zhang, Hao Zhang, Haoyuan Zhang, Huanyu Zhang, Jiapei Zhang, Jiaxuan Zhang, Jin Zhang, Kaiyi Zhang, Miaozhen Zhang, Puqi Zhang, Qinglei Zhang, Rong Zhang, Rui Zhang, Shaoshuai Zhang, Shiyi Zhang, Xiaobin Zhang, Xiaoyun Zhang, Y. Zhang, Yangkun Zhang, Ye Zhang, Yichi Zhang, Yikun Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Zijing Zhang, Bin Zhao, Chenguang Zhao, Feifan Zhao, Jinglun Zhao, Jinxiang Zhao, Shuai Zhao, Wenshuo Zhao, Xiangyu Zhao, Xuanle Zhao, Yikai Zhao, Zijia Zhao, Haozhi Zheng, Huabin Zheng, Ruihan Zheng, Shaojie Zheng, Tengyang Zheng, Haofeng Zhong, Lei Zhong, Longguang Zhong, M. Zhou, Qiankang Zhou, Runjie Zhou, Ruozhang Zhou, Xinyu Zhou, Yiqiao Zhou, Zaida Zhou, Jinguo Zhu, Liya Zhu, Xinhao Zhu, Yangjunfeng Zhu, Yuxuan Zhu, Zhen Zhu, Chen Zhuang, Weiyu Zhuang, Xinxing Zu  
**Category**: cs.CL  
**Published**: 2026-07-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.24653v1  

#### Abstract
We introduce Kimi K3, a 2.8T parameter Mixture-of-Experts model with 104 billion activated parameters, native vision capabilities, and a 1-million-token context window. Kimi K3 is built on Kimi Delta Attention and Attention Residuals, which improve information flow across sequence length and model d...

---

### 19. [LazyMem: Retrieve Broadly, Construct Selectively for Efficient Long-Term Agent Memory](https://arxiv.org/abs/2607.22690)

**Authors**: Jing Yu, Yibo Zhao, Jiaming Zhang, Xiang Li  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.22690v1  

#### Abstract
Long-term memory lets LLM agents reuse past interactions, but raw dialogue histories are verbose and information-sparse. Retrieving broadly improves evidence coverage yet overwhelms downstream reasoning with noise; compressing at write time reduces noise but irreversibly discards details the future ...

---

### 20. [HiLLTS: Zero-Shot Hierarchical LLM-Guided Traffic Signal Control for Sustainable Transportation](https://arxiv.org/abs/2607.22691)

**Authors**: Yue Ding, Tendai Mukande, Mingming Liu  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.22691v1  

#### Abstract
Urban traffic congestion significantly increases fuel consumption, greenhouse gas emissions, and commuter delays, resulting in substantial economic losses and environmental harm in modern cities. Traditional traffic signal control strategies such as fixed-time scheduling, actuated control, and reinf...

---

### 21. [Understanding Human-like Solutions in Combinatorial Optimization via Learning and Search](https://arxiv.org/abs/2607.23854)

**Authors**: Haijiang Yan, Jian-Qiao Zhu, Liqiang Huang, Ming Meng  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.23854v1  

#### Abstract
Humans often find good solutions to combinatorial optimization problems that are computationally hard even for advanced computer algorithms. In the Euclidean traveling salesman problems (TSP), people rapidly produce tours that are near-optimal, despite severe limits on time and computation. What mak...

---

### 22. [Joint Optimization for Greedy Longest-match Tokenization](https://arxiv.org/abs/2607.23362)

**Authors**: Adhiraj Singh, Deepanshu Mody, Ghina Al Shdaifat, Hamza Alshamy, Adam Wiemerslage, Varshini Reddy, Craig W. Schmidt  
**Category**: cs.CL  
**Published**: 2026-07-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.23362v1  

#### Abstract
Recent work has shown that subword vocabularies can be trained to optimize compression for a specific inference rule rather than relying on greedy heuristics such as Byte Pair Encoding (BPE). We extend this approach to greedy left-to-right longest-match decoding, the fast and widely used inference r...

---

### 23. [Sparse Gaussian-Mixture-Model Q-Functions via Hadamard Overparametrization for Online Reinforcement Learning](https://arxiv.org/abs/2607.23474)

**Authors**: Minh Vu, Konstantinos Slavakis  
**Category**: cs.LG  
**Published**: 2026-07-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.23474v1  

#### Abstract
This paper develops an online, off-policy policy-iteration framework for reinforcement learning (RL), based on sparse Gaussian-mixture-model Q-functions (S-GMM-QFs). The framework reconciles streaming, non-stationary data with the Riemannian structure of the parameter space while handling distributi...

---

### 24. [ADVERSARIAL: And-Inverter Graph-Assisted Hardware Trojan Detection At Scale](https://arxiv.org/abs/2607.23882)

**Authors**: Yaroslav Popryho, Debjit Pal, Inna Partin-Vaisband  
**Category**: cs.LG  
**Published**: 2026-07-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.23882v1  

#### Abstract
Modern System-on-Chip (SoCs) often contain hundreds of millions to tens of billions of gates, making existing Hardware Trojan (HT) detection methods impractical due to their immense scale. The proposed approach incorporates symbolically enabled learning by modeling flattened gate-level netlists as B...

---

### 25. [Synthetic Scenario Generation for Evaluation of Industry 4.0 Agents](https://arxiv.org/abs/2607.22563)

**Authors**: Sagar Chethan Kumar, Rohith Kanathur, Dhaval Patel, Kaoutar El Maghraoui  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.22563v1  

#### Abstract
Industrial agent benchmarks require realistic evaluation scenarios that integrate telemetry, failure modes, maintenance records, and domain standards. However, existing benchmarks such as AssetOpsBench rely on manually authored scenarios and cover a limited set of asset classes. We extend AssetOpsBe...

---

### 26. [TriSP: Tri-Signal Structured Pruning for Large Language Models](https://arxiv.org/abs/2607.22587)

**Authors**: Manel Kara laoua, Soumia Bouyahiaoui, Aicha Boutorh  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.22587v1  

#### Abstract
Large language models (LLMs) achieve strong performance across diverse tasks but their deployment is constrained by the memory and compute cost of their parameters. Structured pruning addresses this by removing entire structures such as attention heads and Multi-Layer Perceptron (MLP) neurons to pro...

---

### 27. [DeepLook: Deeper Thinking with Lookahead](https://arxiv.org/abs/2607.22602)

**Authors**: Tingxin Yang, Zefeng Wang, Mengyue Wang, Xingcheng Zhou, Yunpu Ma  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.22602v1  

#### Abstract
Inference-time scaling has emerged as a powerful paradigm for improving large language model reasoning, often delivering larger gains on difficult reasoning tasks than parameter scaling alone. However, existing approaches remain inefficient in how compute is allocated within a reasoning trace. Motiv...

---

### 28. [DynaResize: Runtime GPU Reallocation for Disaggregated LLM Post-Training](https://arxiv.org/abs/2607.22614)

**Authors**: Hanlin Du, Zhiyuan Yan, Haiquan Chen, Jiarui Fang, Yungang Bao, Sa wang  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.22614v1  

#### Abstract
RL-based LLM post-training increasingly disaggregates Rollout and Training across separate GPU resources, but static GPU partitioning suffers from severe pipeline bubbles under long-tail rollout latency. We present DynaResize, a runtime GPU reallocation system that dynamically switches GPUs between ...

---

### 29. [Bayesian Repetition Penalty: A Principled Adjacent-Conditional Framework for Reversing Attention Collapse in Autoregressive Language Models](https://arxiv.org/abs/2607.22694)

**Authors**: Wenjie Fan, Bin Ma, Dong Li  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.22694v1  

#### Abstract
Attention collapse in autoregressive language models -- manifested as repetitive token loops where the model becomes trapped in self-reinforcing attractors -- is a persistent pathology that existing decoding-time heuristics fail to address at its root cause. We present a principled framework that pe...

---

### 30. [Inference-Time Consensus for Mitigating Hidden Behaviors from LLM Fine-Tuning](https://arxiv.org/abs/2607.23394)

**Authors**: Adhyyan Narang, Artin Tajdini, Claire Zhang, Jamie Morgenstern  
**Category**: cs.AI  
**Published**: 2026-07-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.23394v1  

#### Abstract
Recent work shows that fine-tuning language models on even a small amount of poisoned data can install targeted misbehavior, and ostensibly benign data can transmit hidden preferences that generalize broadly. Standard defenses, such as data filtering, mixing in harmless data, and regularization, att...

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

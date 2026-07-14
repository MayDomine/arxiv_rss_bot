# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-14 07:33:09 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FastTPS: An Optimized Method for LLM Token Phase for AI accelerators](https://arxiv.org/abs/2607.11211)

**Authors**: Wenzong Yang, Danyang Zhang, Kun Cao, Tejus Siddagangaiah, Rajeev Patwari, Zhanxing Pu, Siyin Kong, Zijiang Yang, Hao Zhu, Varun Sharma, Yue Gao, Tianping Li, Fan Yang, Jicheng Chen, Yushan Chen, Fennian Zhao, Aaron Ng, Elliott Delaye, Ashish Sirasao, Sudip Nag  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2607.11211v1  

#### Abstract
The popularity of large language models (LLMs) escalates an ongoing demand for effective inference. However, due to the sequential processing of tokens during the token phase in decoder-only LLMs inference, the inherent low parallelism leads to reduced throughput and suboptimal utilization of the co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**FastTPS: An Optimized Method for LLM Token Phase for AI Accelerators**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **decoder-only LLMs** 的推理过程中，**token phase**（自回归生成阶段）存在严重性能瓶颈，主要体现在：
- **低并行性**：每次仅处理一个 token，导致计算单元利用率低下。
- **高内存开销**：频繁的 KV Cache 拼接（`Concat`）操作造成大量内存读写，加剧带宽压力。
- **精度损失风险**：现有加速方法（如 FlashAttention）常伴随数值精度下降，影响模型输出质量。

这些问题在长序列输入下尤为显著，限制了 AI 加速器（如 NPU、GPU）的实际吞吐量和效率。

---

### 🚀 提出的新方法：FastTPS
FastTPS 是一种专为通用 AI 加速器设计的数据流优化框架，包含三大核心技术组件：

#### （1）**Global KV Cache Management (GKVC)**
- **核心思想**：采用静态预分配的 3D KV Cache 内存管理策略，避免 CPU 参与的 `Concat` 操作。
- **实现方式**：通过地址跳跃策略将新生成的 K/V 向量直接写入 L3 缓存中的预留位置，无需重新组织缓存。
- **优势**：
  - 消除 `Concat` 引起的内存重排开销；
  - 减少 L3 ↔ L1 数据传输次数；
  - 支持后续算子融合（operator fusion），提升流水线连续性。

#### （2）**TPSFLAT（基于 FLAT 的 RoPE 注意力优化）**
- **核心思想**：结合 GKVC 实现更深层次的注意力块融合，特别针对 RoPE 结构进行优化。
- **关键技术**：
  - 将 Q 和 K 共享 RoPE 计算逻辑，合并为一次运算；
  - 利用 tiling 策略在 L1 缓存中完成 RoPE → QKT → Softmax → PV 的全链路融合；
  - 基于三层次缓存体系（L1/L2/L3）设计两级分块（tiling）策略。
- **优势**：
  - 显著提高 **Operational Intensity (OI)**；
  - 减少中间张量落盘，降低 I/O 开销；
  - 更好地利用片上存储，缓解 Amdahl’s Law 限制。

#### （3）**Fusion MLP（细粒度融合的 MLP 流水调度）**
- **核心思想**：重构 MLP 中 gate-up 投影权重布局，支持多算子融合与高效流水线执行。
- **关键技术**：
  - **交错布局（interlaced layout）**：将 `W_gate` 和 `W_up` 权重按 tile 单位交替排列，形成更大的 Matmul；
  - **算子融合**：将 gate-up Matmul + SiLU + elementwise Mul 融合为 `siLuMul`；
  - **流水线调度**：重叠权重加载与计算，隐藏内存延迟。
- **优势**：
  - 将 I/O 次数从 7 次减少到 2 次；
  - 提升硬件利用率，逼近理论峰值性能。

---

### 🔍 相比现有方法的优势
| 维度 | 现有方法（如 FlashAttention） | FastTPS |
|------|-------------------------------|--------|
| **内存效率** | 需要显式 `Concat` 或分页机制（PagedAttention） | 完全消除 `Concat`，静态缓存管理 |
| **精度保持** | 存在数值截断误差，尤其在长序列时退化明显 | BF16 下最大误差 < 0.22%，优于 FlashAttention 一个数量级 |
| **适用平台** | 多面向 GPU 架构定制 | 适配通用 AI 加速器（NPU/GPU/TPU/ASIC） |
| **系统集成性** | 依赖特定软件栈 | 可无缝集成至主流框架（PyTorch/TensorFlow） |

---

## 2. 核心实验方法和设置

### 🧪 使用的模型与场景
- **测试模型**：
  - `ChatGLM3-6B`, `Llama2-7B`, `Llama3-8B-instruct`, `Llama3.2-1B`, `Phi3-mini-4k-instruct`
- **部署平台**：
  - **AMD Ryzen AI 300 series NPU**（主平台）
  - 对比平台：Intel Lunar Lake NPU、NVIDIA Jetson Nano、FastFlowLM 平台
- **上下文长度**：支持 1k / 2k / 4k 序列长度测试

---

### 📊 实验设置与评估指标

| 类别 | 内容 |
|------|------|
| **评估任务** | LLM 自回归推理（token phase） |
| **输入配置** | Batch Size = 1, Sequence Length = N_t（动态增长） |
| **精度模式** | BF16（bfloat16）为主 |
| **主要指标** | 
| - **Tokens Per Second (TPS)** | 衡量端到端推理速度 |
| - **Speedup Ratio** | 相对于未优化版本的加速比 |
| - **Operational Intensity (OI)** | 算术操作 / 内存访问，反映计算密度 |
| - **Peak Memory Bandwidth Utilization** | 实际带宽占用率 |
| - **Numerical Accuracy** | 输出与 FP32 基线的最大误差（Max Error） |

---

### ⚔️ 基线方法对比
- **标准实现**：HuggingFace 默认实现（无融合）
- **FlashAttention**：作为主流注意力优化方案进行精度对比
- **原始 MLP 实现**：非融合版 gate/up/down + SiLU 分离执行

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（见 Table 3）

| Model | TPS (w/o FastTPS) | TPS (with FastTPS) | Speedup | Bandwidth Util. |
|-------|-------------------|--------------------|---------|------------------|
| **Llama3.2** | 12.21 ms | 32.27 ms | **2.64×** | 85.22% |
| **Llama2** | 2.34 ms | 12.41 ms | **5.30×** | 79.74% |
| **Phi3** | 2.95 ms | 17.60 ms | **5.95×** | **92.77%** |

> 💡 **最高达 6× 加速**，Phi3 在长序列下达到 **93% 峰值内存带宽利用率**

---

### 🔬 分模块加速效果（Top-down 分析）

| 模块 | 最大加速比 | 说明 |
|------|-----------|------|
| **Attention Block** | **13.07×** | 得益于 GKVC + TPSFLAT 融合 |
| **MLP Block** | **3.14×** | Fusion MLP 显著减少 I/O 和 bubble |

---

### 📉 延迟分解与优化贡献分析（图 5）

- **KV Cache `Concat` 占原 attention 延迟的 66%~82%**
- TPSFLAT 成功移除该部分开销，带来 **79.7%~93% 的延迟下降**
- 例如：Llama3 (4096) 达到 **93% 延迟降低**

---

### 📊 MLP 性能提升（图 7）
- 所有模型均实现显著加速：
  - Llama3 / ChatGLM / Llama2 / Phi3：>20%
  - **Llama3.2 达到 68.17% 延迟下降（即 ~3.1× 加速）**

---

### 📉 OI 提升分析（图 2）
- **Attention OI 提升 3×**：
  - 从 “GQA Ori.” 黄点区域 → “GQA Opt.” 红点区域
- **MLP 接近理论上限**：
  - Fusion MLP（紫色星）远优于原始实现（黄色星）

---

### ✅ 数值精度表现（图 8）
- **最大误差随序列增长而减小**，最长序列下仍 < **0.22%**
- 相比 FlashAttention（~1e-2），FastTPS 精度高出 **约 10 倍（~1e-3）**
- 精度损失主要来自输出截断，而非算法本身

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KV Cache `Concat` 是 token phase 的主要瓶颈**，其内存开销占主导地位。
2. **GKVC 成功将整个推理流程保留在 AI 加速器内部**，避免 CPU 干预，实现全流程融合。
3. **TPSFLAT 通过 RoPE 共享与两级 tiling，在 L1 实现完整 attention 融合**，极大提升 OI。
4. **Fusion MLP 的 interlaced layout 设计有效减少 I/O 次数，并支持细粒度流水调度**。
5. **FastTPS 在真实 NPU 上实现高达 6× 加速，且保持高精度和高带宽利用率（>90%）**。

---

### ⚠️ 局限性
- **依赖静态内存分配**：需预先设定最大序列长度，灵活性略低于 PagedAttention。
- **对缓存层级敏感**：tiling 策略需根据具体硬件（L1/L2/L3 容量）调优，移植时需适配。
- **当前验证集中于 NPU**：虽宣称兼容 GPU/TPU，但尚未在多类设备上广泛验证。

---

### 🔮 未来工作方向
1. **跨平台扩展**：将 FastTPS 移植至 GPU、TPU、ASIC 等架构，验证通用性。
2. **动态序列支持增强**：结合分页机制或弹性缓存策略，提升对变长输入的支持能力。
3. **与量化/剪枝等压缩技术联合优化**：探索 FastTPS 与 Quantization、Distillation 的协同效应。
4. **编译器集成**：将 GKVC、TPSFLAT、Fusion MLP 封装为自动优化 pass，嵌入 AI 编译器（如 TVM、MLIR）。

---

## ✅ 总结
FastTPS 提出了一套面向 AI 加速器的 **token phase 全流程优化方案**，通过 **GKVC、TPSFLAT、Fusion MLP** 三大技术创新，从根本上解决了 KV Cache 管理低效、注意力计算碎片化、MLP I/O 密集等问题。实验证明其可在 **保持高精度的前提下实现最高 6× 的推理加速**，并接近硬件理论极限，为高效部署 LLM 提供了实用路径。

</details>

---

### 2. [[AAFLOW+] Stateful Operator Abstraction with Zero-Copy Distributed KV Cache Orchestration for Multi-Agent Workflows](https://arxiv.org/abs/2607.10987)

**Authors**: Arup Kumar Sarker, Alexander James Halpern, Mills Staylor, Aymen Alsaadi, Gregor von Laszewski, Yue Cheng, Shantenu Jha, Geoffrey Fox  
**Category**: cs.DC  
**Published**: 2026-07-14  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.10987v1  

#### Abstract
Multi-agent LLM systems increasingly integrate retrieval, planning, and reasoning, but remain fundamentally text-centric, requiring agents to repeatedly recompute shared context through expensive prefill. Although single-request inference is known to be accelerated by KV-cache management, it is usua...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《[AAFLOW+] Stateful Operator Abstraction with Zero-Copy Distributed KV Cache Orchestration for Multi-Agent Workflows》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前多智能体 LLM 系统（multi-agent LLM systems）虽然支持检索、规划、推理等复杂流程，但其通信机制仍以**文本为中心（text-centric）**。这意味着即使多个 agent 共享相同的上下文，下游 agent 也必须重复执行昂贵的 **prefill 阶段** 来重建已计算过的模型状态（如 KV Cache），造成严重的计算冗余、高延迟和内存浪费。

这一问题在以下场景尤为突出：
- **Tree-of-Thought 推理**
- **多智能体辩论（Multi-Agent Debate）**
- **协作式 RAG（Collaborative RAG）**
- **自洽采样（Self-consistency Sampling）**

### 🚀 提出的新方法
本文提出 **AAFLOW+**，一种将 KV Cache 提升为**一等分布式系统对象（first-class distributed systems object）** 的状态化操作符抽象框架。其核心思想是：**从“传递文本”转向“传递可复用的执行状态”**。

#### 主要创新点：
- **Stateful Operator Abstraction**  
  扩展了 AAFLOW 的数据流（dataflow）模型，引入 **stateflow** 抽象，显式建模 KV Cache 的生命周期。
- **KV-State Operators**  
  定义了一组用于管理 KV 状态的操作符：
  - `Op_ko_materialize`：将输入上下文 materialize 为可复用的 KV 状态
  - `Op_ko_transfer`：在节点间传输 KV 状态
  - `Op_ko_fork`：从共享前缀派生多个分支状态（支持 Copy-on-Write）
  - `Op_ko_merge`：受限合并（restricted merge），避免语义错误
  - `Op_ko_evict`：基于代价的淘汰策略
- **Zero-Copy 分布式传输**  
  利用 Arrow + UCX/RDMA 实现零拷贝（zero-copy）KV 块传输，最小化序列化开销。
- **兼容性与正确性保障**  
  通过元数据（model ID、tokenizer、positional encoding、lineage）确保状态重用的安全性。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | AAFLOW+ 的改进 |
|------|--------|----------------|
| **vLLM / SGLang** | 仅支持单节点内 KV 复用（如 prefix sharing） | 支持跨 agent、跨节点的分布式 KV 共享 |
| **KVCOMM** | 支持跨上下文通信，但无完整 workflow 抽象 | 提供完整的 materialize/fork/transfer/merge 抽象 |
| **AAFLOW-text** | 仅优化数据传输，不暴露内部模型状态 | 显式管理 KV 状态，实现真正的状态复用 |
| **DistServe** | 解耦 prefill 和 decode，但未暴露 KV 为调度对象 | 将 KV 作为可调度、可转移的一等对象 |

---

## 2. 核心实验方法和设置

### 📊 数据集与工作负载
- **合成确定性提示（synthetic deterministic prompts）**：控制变量，精确测量系统行为
- **Natural Questions 数据集**：用于生成真实问题
- **三大典型多智能体工作负载**：
  1. **Multi-Agent Debate**：多个 agent 基于共享上下文迭代优化回答
  2. **Tree-of-Thought Reasoning**：分支探索不同推理路径
  3. **Retrieval-Augmented Generation (RAG)**：共享检索证据后进行独立推理

### ⚙️ 实验设置
- **硬件环境**：
  - 4–16 节点集群
  - 每节点配备 NVIDIA A100（80GB/40GB）、32–64 CPU 核心
  - RDMA-enabled InfiniBand 互联（带宽最高达 400 Gbps）
- **模型**：
  - **Mistral-7B-Instruct-v0.3**（最大上下文 32K）
  - **Llama-3-8B-Instruct**（最大上下文 8K）
- **后端支持**：
  - Hugging Face (HF)
  - vLLM
  - SGLang

### 📈 评估指标
| 指标 | 定义 |
|------|------|
| **TTFT (Time-to-First-Token)** | 请求到首 token 输出的时间 |
| **Aggregate Compute Cost** | 总 prefill + decode + 开销 |
| **Throughput (tokens/s)** | 单位时间生成的 token 数量 |
| **Peak KV Memory Usage** | 最大 KV 缓存占用 |
| **Framework Overhead (Ω)** | 调度、序列化、同步等系统开销 |
| **KV Reuse Ratio** | 从缓存服务的 token 比例 |
| **Transfer Efficiency** | 避免的 prefill 成本 / 实际传输成本 |

### 🆚 基线方法对比
1. **dense prefill**：每个 agent 独立 prefill（最差基线）
2. **AAFLOW-text**：基于文本的 AAFLOW 版本
3. **vLLM (PagedAttention)**：本地块级 KV 管理
4. **SGLang (RadixAttention)**：结构化程序前缀共享
5. **DistServe**：解耦 prefill 与 decode
6. **KVCOMM**：跨上下文 KV 通信，但无完整状态抽象

---

## 3. 主要实验结果和性能指标

### 📉 关键性能提升（基于 Mistral-7B & Llama-3-8B）

| 指标 | 提升幅度 | 说明 |
|------|----------|------|
| **TTFT 减少** | **最高达 50.2×** | 在长上下文下优势显著（Table 2） |
| **多智能体计算成本降低** | **最高达 7.63×** | 16-agent 规模下（Table 3） |
| **峰值 KV 内存减少** | **1.72× – 6.10×** | 显著优于 vLLM/SGLang（Table 4） |
| **吞吐量提升** | **>7.74×** | 达到 302.61 tokens/s（Table 5） |
| **框架开销降低** | **Ω ≈ 0.0075s** | 远低于 AAFLOW-text 和 KVCOMM |

### 🔬 详细对比结果

#### ✅ TTFT 对比（Experiment 1）
- 在 32K 上下文下，AAFLOW+ 的平均 TTFT 仅为 **0.041s（Mistral）**，而 dense prefill 为 **2.017s**，慢 **49.2×**
- 即使强基线如 SGLang 也慢 **6.8×**，DistServe 慢 **3.0×**

#### ✅ 多智能体扩展性（Experiment 2）
- 当 agent 数从 1 增加到 16：
  - AAFLOW+ 计算成本从 ~30s 增至 ~224s
  - dense prefill 从 ~170s 增至 **2716s**
  - **相对 SGLang 提速 7.63×**
- 表明 AAFLOW+ 成本增长主要来自 decode，而非重复 prefill

#### ✅ KV 传输 vs 重计算（Experiment 3）
- 在 **≥25 Gbps** 网络下，KV 传输始终优于重计算
- 在 400 Gbps 下，**传输效率高达 113.88×**
- 给出明确调度规则：高带宽网络应优先选择 transfer 而非 recompute

#### ✅ 内存效率（Experiment 4）
- AAFLOW+ 峰值 KV 内存仅 **8.355 GiB（Mistral）**
- dense prefill 和 AAFLOW-text 达 **~50 GiB**（**6× 更高**）
- vLLM/SGLang 本地前缀复用约 **14.3–14.6 GiB**（仍高出 1.72×）

#### ✅ 吞吐量与框架开销（Experiment 5）
- AAFLOW+ 吞吐量达 **302.61 tokens/s（Mistral）**
- 强基线如 SGLang 仅 **39.62 tokens/s**
- **AAFLOW+ 吞吐量是基线的 7.63–8.04×**
- 框架开销极低（**0.0075s**），远低于 KVCOMM（>21s）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KV 传输优于重计算**  
   在中高带宽网络（≥25 Gbps）上，传输 KV Cache 比重复 prefill 更高效，尤其适用于长上下文、高分支因子的多智能体任务。

2. **状态共享显著提升效率**  
   将 KV Cache 作为一等分布式对象管理，可大幅减少 TTFT、计算成本、内存占用，并提高吞吐量。

3. **Stateflow 抽象有效替代 Textflow**  
   “传递状态”比“传递文本”更符合多智能体协作的本质，能从根本上消除冗余 prefill。

4. **与现有 LLM Serving 系统正交互补**  
   AAFLOW+ 可集成 vLLM、SGLang 等后端，在其基础上实现跨节点的状态编排。

### ⚠️ 局限性
- **依赖高带宽网络**：在低带宽（如 10 Gbps）下，部分场景仍需回退到 recompute
- **仅适用于共享前缀工作流**：对完全独立的 stateless 推理无优势
- **当前未处理异构模型**：要求所有 agent 使用相同 model/tokenizer
- **未实现持久化与容错恢复**：目前依赖重新计算作为 fallback
- **评估基于合成负载**：未在真实生产环境中验证

### 🔮 未来工作方向
1. **异构模型支持**：扩展兼容性检查，支持 adapter、LoRA、fine-tuned variants
2. **动态调度策略**：结合实时带宽估计、负载预测进行 adaptive placement
3. **支持新型架构**：将 stateflow 抽象推广至 SSM（State Space Models）等非 Transformer 架构
4. **增强容错能力**：实现高效的 checkpointing 与 partial recovery
5. **开发编程接口与工具链**：提供可视化调试、高阶 DSL 支持，降低使用门槛

---

## 总结

AAFLOW+ 通过将 **KV Cache 提升为可调度、可转移、可复用的分布式状态对象**，实现了从“textflow”到“stateflow”的范式转变。其实验结果表明，在多智能体协作场景下，该方法可带来 **数量级级别的性能提升**，是构建高效、可扩展的 agentic AI workflows 的重要基础设施。

</details>

---

### 3. [TIGER: Text-Conditioned Visual Gated Routing with Acceptance Alignment for Multimodal Speculative Decoding](https://arxiv.org/abs/2607.11131)

**Authors**: Quynh Vo, Cong-Duy Nguyen, Ponhvoan Srey, Luu Anh Tuan, Thong Nguyen  
**Category**: cs.CL  
**Published**: 2026-07-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.11131v1  

#### Abstract
Speculative decoding accelerates autoregressive generation by letting a lightweight drafter propose multiple tokens that are verified by a larger target model. Although effective for text-only LLMs, speculative decoding yields limited gains in VLMs because drafters often diverge on vision-critical c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TIGER: Text-Conditioned Visual Gated Routing with Acceptance Alignment for Multimodal Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **speculative decoding** 方法在纯文本大模型（LLMs）中已取得显著加速效果，但在 **Vision-Language Models (VLMs)** 中收益有限。主要原因在于：

- **视觉敏感内容上的分歧**：小型 drafter 模型在处理 OCR 字符串、数字计数、图表数值、物体描述等 vision-critical 内容时容易与 verifier 发散，导致被接受的前缀（accepted prefix）过短。
- **训练目标不匹配**：现有方法多采用模仿学习（imitation learning），优化的是 token-level 的预测准确性，而非直接影响推理效率的 **accepted prefix length**。
- **视觉输入冗余或固定压缩**：传统方法要么暴露全部视觉 token，要么使用固定的压缩接口，无法动态适应当前解码上下文。

### 🚀 提出的新方法：TIGER
作者提出 **TIGER**（Text-conditioned vIsual GatEd Routing），一个面向多模态 speculative decoding 的框架，包含两大核心技术：

#### （1）Text-conditioned Visual Gated Routing
- 动态选择与当前文本状态相关的稀疏视觉 token 子集作为 drafter 的输入。
- 路由机制基于当前文本前缀的隐藏状态生成查询向量（routing query），对所有视觉 token 打分并选取 top-k 最相关者。
- 实现轻量、自适应的视觉接口，避免全局冗余或噪声干扰。

#### （2）Acceptance-Aligned Training
- 不再仅用知识蒸馏（KD）进行训练，而是引入基于 **verifier 接受长度** 的奖励信号。
- 在每个训练样本上采样多个候选 speculative block，运行 verifier 验证，以 **accepted prefix length** 作为奖励 $ R^{(k)} = A^{(k)} $。
- 使用 **GRPO-style policy optimization** 进行组内相对优化，并通过 KL anchoring 控制策略漂移。

> 💡 核心思想：**让 drafter 不只是“模仿目标模型”，而是学会生成“能被 verifier 更长接受”的推测序列。**

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **效率提升** | 显著提高 accepted prefix length 和 end-to-end speedup |
| **质量保持** | 下游任务准确率不下降，甚至略有提升 |
| **训练对齐性** | 训练目标直接对应推理效率瓶颈（prefix length） |
| **架构兼容性** | 无需修改 verifier 架构，保留其 full visual context 和 KV-cache 复用能力 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **主评估基准**：
  - **MMBench**：通用多模态理解与推理
  - **ScienceQA**：科学题图文推理
  - **TextVQA**：含场景文字的视觉问答（OCR-heavy）
  - **MM-Vet**：综合能力评测（识别、定位、OCR、知识推理等）
  - **COCO Caption**：图像描述生成（open-ended generation）

- **额外重上下文评估**：
  - **Video-MME**：视频理解（短/中/长视频）
  - **NLVR2**：双图逻辑推理
  - **MuirBench**：多图复杂关系理解

- **训练数据**：
  - **KD Warm-start**：混合来自 LLaVA-1.5 COCO、TextVQA、ScienceQA 等共约 92.5K 样本
  - **Acceptance-aligned Training**：从 TextVQA、DocVQA、ChartQA 等构建约 80K 中等难度样本池，用于 reward-based 优化

### ⚙️ 实验设置
- **硬件**：8×NVIDIA A40 GPU，混合精度训练（bfloat16）
- **模型家族**：
  - LLaVA-v1.6-Vicuna-7B / 13B
  - Qwen3-VL-4B-Instruct / 8B-Instruct
- **drafter-verifier 配置**：
  - 小模型为 drafter，大模型为 verifier（如 7B → 13B）

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **AccLen** | 平均每轮 speculative decoding 被 verifier 接受的 token 数量（核心效率指标） |
| **VerRds** | 生成完整响应所需的平均 verifier 解码轮次 |
| **Lat.** | 端到端延迟（wall-clock time），包含 routing 开销 |
| **Spd.** | Speedup，相对于标准 autoregressive 解码的时间加速比 |
| **Acc.** | 下游任务准确率（如 VQA 准确率、caption BLEU 等） |

### 🆚 基线方法对比
- **Speculative Decoding Baselines**：
  - MEDUSA、EAGLE-2、EAGLE-3、RACER、TOKENRECYCLING、ViSpec、SpecFLASH
- **Visual Token Efficiency Baselines**：
  - SPARSEVLM、VISPRUNER（非 speculative，仅做 latency-quality trade-off 对比）

> 所有 baseline 均在相同 backbone 上重新适配和训练，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Model Setting | Method | AccLen ↑ | Speedup ↑ |
|-------------|--------|----------|-----------|
| LLaVA-7B | TIGER | **2.67** | **2.57x** |
| LLaVA-13B | TIGER | **2.59** | **2.46x** |
| Qwen3-4B | TIGER | **2.63** | **2.41x** |
| Qwen3-8B | TIGER | **2.23** | **2.12x** |

> 在多数设置下，TIGER 取得 **最高 AccLen 和 Speedup**，优于 ViSpec、EAGLE-3、RACER 等强 baseline。

### 🔁 与基线方法对比结果
- **相比 ViSpec / SpecFLASH**：
  - TIGER 在 AccLen 上平均高出 0.1–0.3，在 speedup 上提升 2–10%。
- **相比 EAGLE-3 / RACER**：
  - 尽管这些方法已有较好表现，TIGER 仍能进一步延长 accepted prefix。
- **消融分析表明**：
  - 移除 visual gated routing 或 acceptance alignment 均会导致显著性能下降。

### 🔍 消融实验结果（Table 3）

| Variant (LLaVA-7B) | AccLen | Speedup |
|---------------------|--------|---------|
| TIGER (full) | **2.67** | **2.57x** |
| w/o Visual Gated Routing | 2.13 | 2.12x |
| w/o Acceptance Alignment | 2.07 | 2.03x |
| w/o KD Warm-start | 2.32 | 1.99x |

> 结果说明：
- **Visual Gated Routing** 和 **Acceptance Alignment** 均不可或缺；
- **KD warm-start** 提供稳定初始化，对训练稳定性至关重要。

### 🎯 路由稀疏性分析（Table 4）
| Routing Sparsity `k` | AccLen | Speedup | Caption Acc. |
|------------------------|--------|---------|--------------|
| 4 | 3.12 | 2.87x | 0.68 |
| **8** | **3.37** | **3.24x** | **0.75** |
| 16 | 2.76 | 2.01x | 0.56 |
| 32 | 2.54 | 1.78x | 0.53 |

> **k=8 是最佳平衡点**：太少会遗漏关键信息，太多则引入冗余降低接受率。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Accepted Prefix Length 是多模态 speculative decoding 的核心效率瓶颈**，必须直接优化该目标。
2. **静态视觉 token 压缩不足以解决 drafter-verifier 分歧**；需要根据当前文本动态路由相关视觉证据。
3. **Acceptance-aligned training 显著优于纯模仿学习**：即使有良好 KD 初始化，仍需进一步对齐 verifier 的接受行为。
4. **TIGER 在多种任务上实现更优的质量-延迟权衡**，尤其在 OCR、计数等视觉敏感任务上增益明显（见 Figure 3）。
5. **TIGER 的执行轨迹显示其具备动态聚焦能力**：随着文本生成推进，路由区域也随之变化，逐步增强视觉 grounding（见 Figure 4–10）。

### ⚠️ 局限性
1. **依赖基础模型能力**：若 base drafter/verifier 本身存在严重幻觉或推理错误，TIGER 无法根本修复。
2. **训练成本较高**：acceptance-aligned training 需要在训练时多次调用 verifier 进行采样验证，计算开销大于标准 KD。
3. **可能遗漏全局线索**：top-k 路由机制在需要全局视觉推理的复杂场景中可能忽略分散的重要信息。
4. **泛化范围待验证**：目前实验集中在 LLaVA 和 Qwen-VL 家族，尚未覆盖更多架构或更大规模模型。

### 🔮 未来工作方向
- 探索更高效的 acceptance reward 估计方式（如 value network 替代采样）。
- 将 TIGER 扩展至更多模态（音频、3D 点云等）。
- 设计可微分的 soft routing 机制以支持端到端联合优化。
- 在更大规模模型和真实应用场景中验证其部署价值。

---

> 🧩 总结一句话：  
> **TIGER 通过“按需提供视觉信息” + “训练目标对齐验证机制”，有效提升了多模态 speculative decoding 的效率与鲁棒性，是迈向高效 VLM 推理的重要一步。**

</details>

---

### 4. [GPU-Tile-Sim: A Tile-Centric GPU Simulation Framework for LLM Hardware-Software Co-Design](https://arxiv.org/abs/2607.11262)

**Authors**: Yitong Ding, Jiawei Huang, Renyang Guan, Yangjie Zhou, Zihan Liu, Yu Feng, Shixuan Sun, Mingyi Guo, Jingwen Leng, Jian Weng  
**Category**: cs.DC  
**Published**: 2026-07-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.11262v1  

#### Abstract
Modern LLM (large language model) workloads increasingly rely on optimized GPU kernels through hardware-software co-design. These kernels achieve high-performance through fine-grained dependency scheduling and computation-memory overlap. As such, they incur new challenges on existing GPU performance...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GPU-Tile-Sim: A Tile-Centric GPU Simulation Framework for LLM Hardware-Software Co-Design**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代大语言模型（LLM）工作负载依赖于高度优化的 GPU 内核，这些内核通过**细粒度依赖调度**和**计算-内存重叠**实现高性能。然而，现有的 GPU 性能建模方法面临以下挑战：
- **指令驱动模拟器**（如 GPGPU-Sim、Accel-Sim）虽然精度高，但难以适应快速演进的硬件架构（如 Hopper、Blackwell），扩展成本高昂。
- **分析型模型**（如 TileFlow、LLMCompass）抽象层次过高，无法准确捕捉异步流水线、warp specialization 和 fused kernel 中的复杂依赖关系。

因此，现有方法在**准确性**与**可扩展性**之间存在根本矛盾，难以支持 LLM 领域的软硬件协同设计。

---

### **提出的新方法与新思路**
作者提出了 **GPU-Tile-Sim (GTSim)** —— 一种以 **tile 为中心的图驱动 GPU 模拟框架**，其核心思想是：
> **现代 GPU 内核性能主要由 tile 级别的依赖结构决定，而非单条指令延迟。**

#### **核心创新点：**
- **Warp-Centric Tile Graph 抽象**  
  将内核执行表示为一个有向无环图（DAG），其中：
  - **节点（Node）**：代表 warp 或 warp group 执行的 tile 级操作（如 TMA_Load、WGMMA）。
  - **边（Edge）**：分为两类：
    - **Data Edge**：生产者-消费者数据依赖。
    - **Order Edge**：非数据依赖的顺序约束（如同步信号、缓冲区复用控制）。
  这种抽象显式表达了 **kernel fusion、software pipelining、warp specialization** 等关键技术。

- **自动前端 + 图驱动后端**
  - **前端**：从 TileLang IR 自动生成 tile graph，保留了软件流水线和 warp 角色划分等语义。
  - **后端**：基于图进行调度，直接驱动执行，避免逐条模拟指令。

- **吞吐量导向的硬件建模（Throughput-Oriented Modeling）**
  对 compute、memory、NoC 等资源使用吞吐量、带宽、延迟等宏观参数建模，而非模拟微架构细节，在保持精度的同时大幅提升效率。

---

### **相比现有方法的优势**
| 特性 | GTSim | 指令驱动模拟器（如 Accel-Sim） | 分析型模型（如 TileFlow/LLMCompass） |
|------|-------|-------------------------------|------------------------------------|
| **准确性** | ✅ 高（显式依赖建模） | ✅ 高 | ❌ 较低（忽略细粒度重叠） |
| **可扩展性** | ✅ 强（模块化设计） | ❌ 差（需重构指令解码） | ✅ 强 |
| **支持 warp specialization** | ✅ 显式支持 | ⚠️ 可支持但复杂 | ❌ 不支持 |
| **支持异步流水线** | ✅ 支持 | ✅ 支持 | ❌ 近似处理 |
| **模拟速度** | 快（3.5–4.6× Accel-Sim） | 慢 | 极快但精度牺牲 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与应用**
实验覆盖了典型的 LLM 计算模式：
- **GEMM 类内核**：
  - 常规 GEMM
  - 融合 GEMM + SiLU
  - FP8 GEMM（混合精度）
- **Attention 类内核**：
  - FlashAttention-3
  - Flash-Decoding
  - FlashMLA
- **端到端推理任务**：
  - Llama-3-8B 推理（prefill + decode 阶段）

---

### **实验平台与硬件配置**
在三种 NVIDIA GPU 上进行验证：
| 参数 | A100 | H100 SXM | B200（Blackwell） |
|------|------|----------|------------------|
| SM 数量 | 108 | 132 | 148 |
| Tensor Core 吞吐量 | 1024 FMAs/cycle | 2048 FMAs/cycle | 4096 FMAs/cycle |
| TMA 发行速率 | — | 100 B/cycle | 100 B/cycle |
| NoC / DSMEM | — | 支持 | 支持 |
| TMEM | — | — | 支持 |

真实硬件运行环境：CUDA 12.4，Ubuntu 20.04.6 LTS。

---

### **评估指标**
- **MAPE**（Mean Absolute Percentage Error）：衡量预测周期与实测周期之间的平均绝对百分比误差。
- **Pearson 相关系数**：衡量预测趋势的一致性。
- **模拟速度**：相对于 Accel-Sim 的加速比。
- **消融实验**：验证不同图机制的影响。

---

### **基线方法对比**
- **TileFlow**：基于 tile 的分析模型，支持融合数据流建模。
- **LLMCompass**：面向 LLM 的映射搜索框架，但仅支持单算子内核。
- **Accel-Sim**：指令级模拟器，作为高精度基准。

所有模型均适配至 H100 参数以公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **GEMM 内核（H100）**
| 内核类型 | GTSim MAPE | TileFlow MAPE | LLMCompass MAPE |
|--------|------------|---------------|------------------|
| 常规 GEMM | **1.22%** | 24.32% | 34.07% |
| 融合 GEMM+SiLU | **3.60%** | 34.17% | 14.99% |
| FP8 GEMM | **5.40%** | 31.25% | 40.93% |

> ✅ GTSim 在所有 GEMM 场景下显著优于分析模型，且 Pearson 相关系数 > 0.995。

#### **Attention 内核（H100）**
| 内核 | GTSim MAPE | TileFlow MAPE | LLMCompass MAPE |
|-----|------------|--------------|------------------|
| FlashAttention-3 | **6.50%** | >100% | >100% |
| Flash-Decoding | **2.37%** | >100% | ~80% |
| FlashMLA | **6.09%** | ~30% | ~40% |

> ✅ GTSim 是唯一能在 fused attention 上保持 <10% MAPE 的模型。

#### **端到端 LLM 推理（Llama-3-8B on H100）**
- **Prefill + Decode 平均 MAPE**：**8.71%**
- **Decode-only（batch=1~8, KV=1024~8192）平均 MAPE**：**7.06%**
- 最大误差仅为 **8.87%**，表明对长上下文场景建模稳定。

---

### **与基线方法对比结果**
- GTSim 比 **TileFlow** 和 **LLMCompass** 平均降低 MAPE 超过 **20个百分点**。
- 在 Accel-Sim 支持的 A100 上：
  - GTSim 模拟速度是 Accel-Sim 的 **3.48–4.60×**（几何平均 3.98×）。
  - MAPE 为 **8.75%**，优于 TileFlow（27.50%）、LLMCompass（30.88%）、Accel-Sim（17.46%）。

> 📌 结论：**GTSim 实现了接近指令级模拟器的精度，同时速度更快，并远超分析模型。**

---

### **消融实验结果**
在 H100 FP8 GEMM 上进行消融研究（Tab. 4）：

| 配置 | MAPE |
|------|------|
| 完整模型（含 data + order + sync） | **1.55%** |
| 移除通用 order constraints | 5.90% |
| 移除跨 warp/group 同步 | 35.44% |
| 仅保留 data dependencies | 42.19% |

> 🔍 发现：**order edges 和同步机制对建模异步流水线至关重要**，移除后导致过度并行化和严重高估性能。

---

## **4. 关键结论和发现**

### **主要发现**
1. **依赖结构是性能建模的关键**  
   现代 LLM 内核的性能瓶颈在于 **tile 级别的依赖调度与重叠能力**，而非单指令延迟。GTSim 通过 tile graph 显式建模这一特性，实现了高精度预测。

2. **图驱动模拟可在精度与效率间取得平衡**  
   GTSim 无需模拟指令流水线，却能达到接近 Accel-Sim 的精度，同时速度快近 4 倍，适合用于设计空间探索。

3. **支持软硬件协同设计分析**
   - 成功用于分析 **software pipelining** 设计（如 WS 3-stage 最优）。
   - 验证了 **NoC-enabled fusion** 可带来 **1.65× 平均加速**，且拓扑感知映射进一步提升通信效率。
   - 成功迁移到 **Blackwell 架构**（仅修改 11.3% 核心代码），初步验证 FA4 在 Blackwell 上优于 FA3，归因于：
     - 更大的 threadblock 粒度（减少 wave 数）
     - 利用 TMEM 实现更解耦的三路 warp specialization

---

### **方法的局限性**
- **依赖高质量 IR 输入**：当前主要支持 TileLang，对原生 CUDA 支持有限，需借助 CuBridge 等工具提取结构。
- **静态图假设**：目前主要针对静态内核，动态路由类 MoE 内核虽可支持，但仍需预知 token 分布。
- **单 GPU 模拟**：未建模多 GPU 间 NVLink 通信，限制了分布式训练场景的应用。

---

### **未来工作方向**
- **开源框架**：计划将 GTSim 开源，促进社区在 LLM 软硬件协同设计中的应用。
- **支持更多 DSL**：扩展前端以兼容 Triton、TLX 等新兴编程语言。
- **集成编译器反馈闭环**：结合自动调优系统，实现“建模 → 优化 → 验证”闭环。
- **扩展至多 GPU 和 AI 集群**：加入 NVLink 和 RDMA 建模，支持更大规模系统仿真。

--- 

> ✅ **总结一句话**：  
> **GTSim 通过引入 warp-centric tile graph 抽象，首次实现了高精度、高效率、高可扩展性的 GPU 内核模拟，为 LLM 时代的软硬件协同设计提供了强有力的建模工具。**

</details>

---

### 5. [Event-based Neural Decoding for Neuroprosthetic Motor Control](https://arxiv.org/abs/2607.11445)

**Authors**: Khaleelulla Khan Nazeer, Sirine Arfa, Matthias Jobst, Richard George, Christian Mayr  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.11445v1  

#### Abstract
A substantial number of patients experience diminished mobility due to disabilities, diseases, or accidents. Although modern prostheses, powered by deep neural networks, hold the promise of significantly enhancing the quality of life for these individuals, their widespread adoption is hindered by si...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题：** *Event-based Neural Decoding for Neuroprosthetic Motor Control*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代神经假体（neuroprosthetics）依赖高性能深度学习模型进行神经解码（neural decoding），但这些模型通常存在以下瓶颈：
- **高延迟（latency）** 和 **高能耗（energy consumption）**
- 需要连接外部GPU等计算设备，限制了患者的移动自由
- 无线传输带宽有限，难以支持大量神经信号的实时回传
- 植入式系统对 **计算资源、内存占用和功耗** 要求极为严格

此外，在临床长期使用中，电极会因组织瘢痕化、阻抗上升或位置漂移导致神经信号退化（chronic degradation），影响解码稳定性。

### 🚀 提出的新方法与创新思路
本文提出一种基于 **Event-based Gated Recurrent Unit (EGRU)** 的高效神经解码器，用于闭环（closed-loop）运动控制任务。其核心创新包括：

1. **首次将 EGRU 应用于强化学习框架下的神经解码任务**
   - 利用 EGRU 的事件驱动特性实现稀疏激活（sparse activation），显著降低计算负载。
   - 支持低功耗硬件部署，适用于植入式 BMI（Brain-Machine Interface）系统。

2. **设计轻量级、高效的 on-device 解码架构**
   - 模型仅含 **2个 EGRU 单元（共约2K参数）**，实现极小 footprint。
   - 采用“开环预训练 + 闭环强化学习微调”的两阶段训练策略，提升收敛速度与鲁棒性。

3. **验证模型在慢性信号退化场景下的鲁棒性**
   - 在训练和测试中引入可控扰动：模拟电极失效（silencing）与调谐漂移（tuning drift）
   - 展示模型对真实世界退化条件的强大适应能力

### 🔍 相比现有方法的优势
| 维度 | EGRU 方法优势 |
|------|----------------|
| **能效与稀疏性** | 激活 sparsity 可达 ~9% 以上，有效 MACs 显著低于传统 LSTM |
| **硬件友好性** | 支持 event-driven 推理，适合部署于 SpiNNaker2 等 neuromorphic 平台 |
| **资源效率** | 参数少（2K）、内存占用低，满足植入式系统的严苛要求 |
| **鲁棒性** | 在高达 80% 探针被扰动的情况下仍保持 >90% 成功率 |

---

## 2. 核心实验方法和设置

### 📚 数据集与仿真环境
- 使用 **非人灵长类动物（non-human primate）合成数据集**，由 **Online Prosthesis Simulator (OPS)** 生成
- 输入为来自 **96 个方向敏感神经元** 的 spike activity
- 输出目标是推断二维平面上的 **cursor velocity command**

#### 两个竞赛赛道（Tracks）：
- **Track 1（基础闭环控制）**：标准中心向外任务（center-out task），从中心出发到达随机目标
- **Track 2（鲁棒性挑战）**：加入神经信号扰动
  - **部分探针静音（signal dropout）**
  - **神经元偏好方向重采样（tuning drift）**

### 🧪 实验设置
- **模型结构**：`96 → 4 → 2 (EGRU) → 2` 的三层结构（输入→线性层→EGRU→输出头）
- **训练流程**：
  1. **Open-loop pre-training**：使用 5,000 条离线生成的轨迹进行监督预训练（30 epochs）
  2. **Closed-loop RL fine-tuning**：通过强化学习优化策略，共训练 600 episodes
- **探索机制**：动作空间添加 Gaussian noise（初始 ε=0.5，逐步衰减至 0）
- **奖励函数设计**：
  - 完成奖励（completion reward）
  - 时间奖励（time bonus）
  - 距离惩罚（distance penalty）
  - 总体可微分且平滑，利于梯度更新

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Avg Time to Target (s)** | 光标到达目标所需的平均时间 |
| **Success Rate (%)** | 是否成功进入并维持在目标区域内 |
| **Effective MACs / ACs** | 实际执行的乘加操作数（反映计算效率） |
| **Footprint (params)** | 模型总参数量 |
| **Activation Sparsity** | 单位时间内激活的神经元比例 |
| **Robustness under Perturbation** | 不同扰动比例下的性能变化（如成功率热图） |

### ⚖️ 基线方法对比
- **LSTM-based decoder**：相同参数规模（hidden dim=2 或 4）
- 对比维度包括：
  - 解码精度（success rate, time to target）
  - 计算成本（effective MACs, dense synaptic ops）
  - 激活密度（activation sparsity）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table I & II）

#### Track 1 结果（无扰动）
| 指标 | 数值 |
|------|------|
| **Footprint** | 2,172 params |
| **Activation Sparsity** | 最高达 4.1%（Neuron 2） |
| **Effective MACs** | ~3,948 – 4,351 |
| **Avg Time to Target** | **0.87 – 0.95 秒** |
| **Success Rate** | **100%**（所有 neuron model 下均达成） |

> 表明模型可在不到1秒内完成任务，且完全成功。

#### Track 2 结果（有扰动：40% 静音 + 50% 分布偏移）
| 指标 | 数值 |
|------|------|
| **Avg Time to Target** | 0.83 – 1.12 秒（略有增加） |
| **Success Rate** | **仍为 100%** |
| **Effective MACs** | 略有上升，但仍远低于 dense 操作 |
| **Activation Sparsity** | 进一步提高（最高达 5.1%） |

> 显示模型具备强健的抗干扰能力。

### 🔬 与 LSTM 的对比结果（Table III）
| Model | Hidden Dim | Act. Sparsity | Eff. MACs | Dense Ops | Avg Time to Target |
|-------|------------|--------------|-----------|-----------|--------------------|
| **EGRU** | 2 | **0.09** | **4,020** | 38,777 | 0.90 s |
| **LSTM** | 2 | 0.00 | 4,763 | 36,305 | **0.82 s** |
| **EGRU** | 4 | **0.08** | **9,612** | 43,408 | 0.87 s |
| **LSTM** | 4 | 0.00 | 12,081 | 43,431 | **0.82 s** |

#### 对比分析：
- **LSTM 略快（~0.82s vs ~0.90s）**，但以更高计算代价换取
- **EGRU 的 activation sparsity 显著优于 LSTM（接近零稀疏）**
- **EGRU 的 effective MACs 更低**，尤其在相同性能下更节能
- **EGRU 更适合 event-driven、low-power 场景**

### 🔍 消融实验与鲁棒性分析（Figure 2 & 3）
- **Figure 2**：随着扰动探针比例增加（0% → 100%），EGRU 在 time to target 和 success rate 上表现稳定，直到约 80% 才开始下降
- **Figure 3**：热图显示即使在 60% 神经元被修改时，成功率仍高于 80%
- 加入 fine-tuning 后，鲁棒性进一步增强（with finetuning 曲线明显优于 without）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **EGRU 是一种高性能、高效率的闭环神经解码器**
   - 在极小模型规模（2K params）下实现接近最优控制性能
   - 支持 event-based 推理，天然契合低功耗、稀疏通信需求

2. **事件驱动架构可有效平衡准确率与能效**
   - 激活稀疏性带来显著的 energy saving 和 latency reduction
   - 特别适合部署在边缘设备或 neuromorphic hardware 上（如 SpiNNaker2）

3. **模型具有内在鲁棒性（inherent robustness）**
   - 即使未专门训练，也能应对一定程度的电极故障和信号漂移
   - 通过扰动感知训练（perturbation-aware training）可进一步提升泛化能力

4. **两阶段训练策略有效加速收敛**
   - 开环预训练提供良好初始化，避免 RL 中的冷启动问题
   - 强化学习微调实现任务特定优化

### ⚠️ 方法的局限性
- 当前实验基于 **合成神经数据**，尚未在真实人类患者数据上验证
- EGRU 的表达能力是否足以处理更复杂的多自由度控制任务尚待研究
- 虽然计算量低，但在极端扰动（>80% probe modified）下性能仍会下降
- 缺乏对 spike encoding 方式的深入探讨（如 Poisson 编码 vs rate coding）

### 🔮 未来工作方向
1. **硬件协同设计（hardware co-design）**
   - 将 EGRU 模型映射到专用 neuromorphic chip 上，实测功耗与延迟
2. **迁移到真实临床数据集**
   - 在人类 intracortical recordings 上验证泛化能力
3. **扩展到更多自由度控制**
   - 如手部抓取、多关节协调等复杂动作
4. **自适应在线学习机制**
   - 实现模型在运行过程中动态调整以应对持续退化的信号质量

---

> 💡 **总结一句话**：  
> 本研究展示了 **EGRU** 作为一种兼具 **高性能、高能效与强鲁棒性** 的 event-based recurrent 架构，在推动 **全集成、闭环、植入式神经假体系统** 发展方面具有巨大潜力。

</details>

---

### 6. [Route, Communicate, and Reason: Gated Routing and Adaptive Depth for Efficient Multi-Agent Reasoning](https://arxiv.org/abs/2607.10836)

**Authors**: Sudipto Ghosh, Tanmoy Chakraborty  
**Category**: cs.AI  
**Published**: 2026-07-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.10836v1  

#### Abstract
Multi-agent ensembling multiplies active parameters and inference cost without answering three basic questions: which agents to consult, how deeply a query should traverse a hierarchy of agents, and when inter-agent communication is worth its cost. We present GRADE (Gated Routing and Adaptive Depth ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Route, Communicate, and Reason: Gated Routing and Adaptive Depth for Efficient Multi-Agent Reasoning

---

## 1. 主要贡献和创新点

### ✅ 解决的问题
当前的 **multi-agent ensembling** 方法存在三大核心缺陷：
1. **缺乏深度自适应性（Limited depth adaptivity）**：所有查询都必须经过完整的 agent 层级，无论其复杂度如何，导致简单任务浪费大量计算资源。
2. **通信控制粗粒度（Coarse communication control）**：系统通常采用全广播或无过滤聚合的方式进行 agent 间通信，无法实现选择性、高效的信息交换。
3. **专家替换后缺乏再校准机制（No recalibration after expert substitution）**：当替换某个 agent 时，原有 gate 的阈值因分布偏移而失效，导致性能骤降。

这些问题共同导致推理成本高、效率低、灵活性差。

---

### 🚀 提出的新方法：GRADE
作者提出 **GRADE**（**Gated Routing and Adaptive Depth for Efficient Reasoning**），一个层次化的多智能体系统，通过四个轻量级学习型门控（gates）联合控制以下决策：

| Gate | 功能 |
|------|------|
| **Assignment Gate** | 决定激活哪些子 agent（基于查询与专家描述符的匹配度） |
| **Depth Gate** | 控制推理层级深度（0: 仅 Orchestrator；1: 经理层；2: 完整子 agent 池） |
| **Cross-Read Gate** | 决定哪些 agent 对之间可以进行跨注意力通信（选择性通信） |
| **Prune Gate** | 在融合前剪枝低价值分支，减少冗余计算 |

此外，引入两个关键机制：
- **Expert Registry**：支持异构模型注册，允许运行时热插拔（hot-swappable）专家模型。
- **Per-agent Calibration Maps**：每个 agent 配备可学习的仿射归一化模块 $ K_a $，在更换模型后仅需 64 个锚点查询即可快速重新校准，无需重训练其他组件。

---

### 🔍 相比现有方法的优势
| 方面 | GRADE 的优势 |
|------|-------------|
| **效率** | 平均仅使用 ~17B active parameters，仅为最强基线 Puppeteer (~28B) 的一半左右，且 VRAM 占用更低（52GB vs 56GB） |
| **性能** | 在 GSM8K、MMLUPro、GPQA 上全面超越所有基线，在 MMLUPro 上领先 4.8 个百分点 |
| **灵活性** | 支持动态拓扑、自适应深度、选择性通信，能根据任务难度自动分配资源 |
| **鲁棒性** | 实现安全的运行时专家替换，配合 calibration 可在数个 batch 内恢复精度 |

---

## 2. 核心实验方法和设置

### 📚 数据集
使用四个递增难度的推理基准测试：
| 数据集 | 描述 |
|-------|------|
| **GSM8K** | 小学数学应用题，侧重基本算术推理 |
| **MMLUPro** | 多领域语言理解进阶版，涵盖 STEM、人文等复杂知识 |
| **GPQA** | 研究生级别问答，对抗“Google-proof”问题，要求深层专业知识 |
| **AIME-2025** | 数学奥赛风格题目，强调长链符号推理能力 |

> 所有任务均为 zero-shot 设置，除 CoT 基线外不提供示例。

---

### ⚙️ 实验设置与评估指标
| 项目 | 设置 |
|------|------|
| **Agent Pool** | 6 个 sub-agent，来自三种不同模型：<br>- Qwen2.5-7B-Instruct (3 agents)<br>- Phi-3-mini-4K-Instruct (2 agents)<br>- Llama-3.2-3B-Instruct (1 agent) |
| **Orchestrator & Managers** | Mo 使用 Qwen2.5-0.5B-Instruct；Manager 为轻量 MLP 模块 |
| **Active Parameters** | 按期望激活参数计算，平均 ~17B/查询（范围从 0.5B 到 ~32B） |
| **训练算法** | **CoGRPO**（Collaborative Group-Relative Policy Optimization），无 critic 的强化学习策略优化方法 |
| **评估指标** | 准确率（Accuracy %）、延迟（Latency, s）、峰值 VRAM（GB）、token 使用量 |

---

### 🆚 基线方法对比
| 类别 | 基线方法 |
|------|--------|
| **单模型** | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct |
| **增强推理** | CoT, Self-Refine |
| **混合专家** | Mixtral-8×7B |
| **集成方法** | Ensemble-4 Majority Vote, LLM-Blender |
| **进化协调框架** | EvoAgent, Puppeteer |
| **联合训练方法** | MAGRPO, MAPoRL+ |

> Puppeteer 是当前最强的 multi-agent 协调系统，作为主要比较对象。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1）
| Method | Act. Params | GSM8K | MMLUPro | GPQA | AIME-2025 | **Overall** |
|--------|------------|--------|---------|------|-----------|-----------|
| Puppeteer | ~28B | 93.7±0.3 | 73.4±0.6 | 54.1±1.9 | **27.2±3.8** | 62.1±1.7 |
| **GRADE (Ours)** | **~17B** | **94.8±0.3** | **78.2±0.6** | **57.4±1.8** | 25.3±3.6 | **63.9±1.6** |

> ✅ **GRADE 在 Overall 上领先 1.8 个百分点，同时节省约 39% 的 active 参数**

#### 分项亮点：
- **MMLUPro**: 超越 Puppeteer **4.8 个百分点**，显示其在多学科综合推理上的优势
- **GPQA**: 超越 **3.3 个百分点**，得益于 manager 层对 domain-specialist 的有效分组
- **AIME-2025**: Puppeteer 以 1.9 分领先，因其更强的个体模型容量更适合长链数学推理

---

### ⏱ 效率对比（见 Table 2）
| Method | MMLUPro Acc | Latency (s) | Peak VRAM (GB) |
|--------|-------------|-------------|----------------|
| Puppeteer | 73.4±0.6 | 13 | 56 |
| **GRADE** | **78.2±0.6** | **3.1–11.4** | **52** |

> ✅ GRADE 不仅更准确，而且延迟更低（易题仅 3.1s），内存占用更少

---

### 🔪 消融实验结果（见 Table 9）
移除关键组件后的性能下降表明各机制的重要性：

| 变体 | Overall 下降 | 主要影响 |
|------|--------------|----------|
| **-Hierarchy** | -7.2 pts | 最大损失，说明 depth control 和 manager grouping 极其关键 |
| **-Cross-Read** | -4.2 pts | 显著降低 GPQA 表现，验证跨域信息共享的价值 |
| **-Cross-Attention** | -3.1 pts | 输出融合能力受损 |
| **-Memory** | -1.7 pts | 影响多跳复合问题处理 |
| **-Dynamic Topology** | -1.1 pts | 固定 top-k 不如动态路由灵活 |

---

### 🔁 热替换实验（Hot-Swap Robustness, 见 Table 3 & Figure 2）
| 替换类型 | Acc@1 | Q2R (Queries to Recover) |
|--------|--------|--------------------------|
| Same Model | 78.1% | 0 |
| Same Family (Qwen → Qwen3) | 75.4% | 4 |
| Local → API (Qwen → GPT-4o-mini) | 74.7% | 3 |
| **No Calibration** | **60.1%** | **>20** |

> ❗ 无 calibration 导致立即 **18.1 点精度下降**，证明 per-agent calibration 是安全热替换的必要条件

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **“节制优于堆叠”**（Restraint over Capacity）  
   GRADE 的成功并非来自更多计算，而是**精准抑制不必要的计算**：
   - 动态 depth gate 让简单问题提前退出
   - Cross-Read gate 学会将通信频率稳定在 ~50%，避免噪声干扰
   - Prune gate 清除无效路径，提升效率

2. **层级结构是性能核心**  
   移除 hierarchy 导致最大性能损失（-7.2 pts），manager 层提供的 group signal 对 multi-domain 问题至关重要。

3. **CoGRPO 是高效的信用分配机制**  
   无需 critic，通过 group-relative advantage 实现稳定的多 agent 协同训练，优于 MAPPO、MAPoRL 等 critic-based 方法。

4. **difficulty-aware allocation 成功实现**  
   更难的任务自动激活更多 agent 和更深的层级（AIME-2025 平均调用 3.8 个 agent，深度达 2.0）。

---

### ⚠️ 局限性
1. **在极端长链推理上仍落后于 brute-force ensemble**  
   如 AIME-2025 中 Puppeteer 凭借更强个体模型保持领先，说明 GRADE 当前依赖已有专家的能力上限。

2. **训练收敛较慢，尤其对高难度任务**  
   AIME-2025 的 token usage 在 200 步内未完全收敛，提示需要更长训练周期。

3. **Cross-Read gate 设计仍有改进空间**  
   当通信频率设为 1.0（全连接）时性能下降 2.1 分，说明当前门控尚未完全学会最优稀疏模式。

---

### 🔮 未来工作方向
1. 引入更强的专业化 backends（如 Qwen-Math）以提升在 AIME 等任务的表现
2. 探索更细粒度的通信机制（如分层 attention 或 message prioritization）
3. 扩展至更大规模 agent pool 和更深 hierarchy
4. 将 calibration 机制扩展到输出分布之外（如 length、diversity）

---

## 总结
GRADE 是首个将 **gated routing、adaptive depth、selective communication 和 branch pruning** 统一于一个端到端可训练框架中的 multi-agent 系统。它通过 **CoGRPO** 实现高效的协作信用分配，并借助 **per-agent calibration** 实现安全的运行时专家替换。实验证明，该方法在显著降低计算开销的同时，在多个复杂推理任务上达到 SOTA 性能，标志着向**高效、灵活、可维护的多智能体推理系统**迈出了重要一步。

</details>

---

### 7. [Learning Residual Kinematic Corrections for Continuous Neural Decoding via Reinforcement Learning](https://arxiv.org/abs/2607.11530)

**Authors**: Jiamian Li, Niall McShane, Attila Korik, Naomi du Bois, Karl McCreadie, Leen Jabban, Benjamin Metcalfe, \"Ozg\"ur \c{S}im\c{s}ek, Damien Coyle  
**Category**: cs.AI  
**Published**: 2026-07-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.11530v1  

#### Abstract
Decoding continuous three-dimensional (3D) motor imagery (MI) using non-invasive electroencephalography (EEG)-based brain--computer interfaces (BCIs) remains challenging due to signal variability and residual decoding errors. Deep learning architectures such as convolutional neural network--long sho...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning Residual Kinematic Corrections for Continuous Neural Decoding via Reinforcement Learning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **非侵入式 EEG-based BCI 在连续三维（3D）运动想象（Motor Imagery, MI）解码中存在系统性残差误差**，导致预测轨迹与真实目标轨迹之间存在偏差。
- 现有基于 CNN-LSTM 的深度学习模型虽能捕捉时空特征，但仍难以消除系统性误差。
- 强化学习（RL）在 BCI 中的应用通常依赖实时 EEG 特征作为奖励信号，易受 EEG 信号变异性、非平稳性和噪声干扰，且在线更新耗时，实用性受限。

### 🚀 提出的新方法
提出一种**两阶段离线残差强化学习框架（CNN-LSTM-RL）**：
1. **第一阶段**：使用 CNN-LSTM 模型作为固定（frozen）基础解码器，从 EEG 解码出初始速度轨迹。
2. **第二阶段**：训练一个独立的离线 RL 代理（agent），仅基于 CNN-LSTM 的输出轨迹进行**残差运动学修正（residual kinematic correction）**，不直接接收 EEG 输入。

该框架的关键设计是：
- **解耦修正过程与神经信号输入**：RL agent 只观察解码器输出、时间步和前一步修正速度，避免直接暴露于高噪声 EEG。
- **离线训练 + 固定部署**：RL agent 在离线状态下训练，无需用户实时参与，提升可扩展性。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **鲁棒性** | 避免 RL 直接依赖不稳定 EEG 信号，减少噪声干扰 |
| **效率** | 无需在线策略更新，降低对用户交互的依赖 |
| **性能提升** | 显著改善轨迹相关性与误差，甚至超越“同会话训练”上限（WSR） |
| **通用性** | 可适配不同反馈环境（2D 屏幕 / VR）和个体用户 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **参与者**：10 名右利手成人
- **会话数**：每人完成 10 个会话（session），交替使用 **2D 屏幕反馈** 和 **沉浸式 VR 环境反馈**
- **任务**：上肢指向 4 个目标的执行与想象任务
- **每会话结构**：
  - 8 轮（run）：4 轮实际动作 + 4 轮运动想象
  - 每轮含 2 个 block，每个 block 含 16 次试验（trial）
  - 总计：25,600 trials
- **数据采集**：
  - **EEG**：g.Nautilus 系统，32 导联 → 在线使用 17 导联（集中于感觉运动皮层）
  - **Kinematics**：Vive 手腕追踪器，采样率 60Hz，获取 3D 坐标并微分得瞬时速度

### ⚙️ 实验设置
- **基础解码器**：ERSP-based CNN-LSTM 架构
  - CNN 提取频谱空间特征（ERSP 图像）
  - LSTM 捕捉时间动态
  - 输出为 x, y, z 三轴速度
- **RL Agent 设计**：
  - 使用 **Soft Actor-Critic (SAC)** 算法
  - 观察量：`[decoder output, normalized time, previous corrected velocity]`
  - 动作：连续残差速度（residual velocity）
  - 奖励函数：`rt = -d(v_corrected, v_target)`（负欧氏距离）
  - **无目标速度或 EEG 输入**，实现真正“黑箱后处理”
- **超参数优化**：使用 **Optuna** 进行 200 次搜索，最大化 Pearson 相关系数

### 📈 评估指标
| 指标 | 公式说明 |
|------|--------|
| **Pearson Correlation Coefficient (r)** | 衡量预测轨迹与目标轨迹的形态相似性 |
| **Root Mean Square Error (RMSE)** | 衡量预测与目标之间的平均偏差 |

### 🔁 对比策略（Baseline Methods）
| 缩写 | 名称 | 描述 |
|------|------|------|
| **FDG** | Fixed Decoder Generalisation | Session 1 训练，应用于所有后续会话（模拟真实在线场景） |
| **SAT** | Sequential Adaptive Training | 前一会话训练，测试于下一 session（模拟周期重校准） |
| **WSR** | Within-Session Reconstruction | 同一会话内训练测试（性能上界） |
| **FRL** | Fixed Reinforcement Learning (**本文方法**) | RL agent 用 Session 1 训练，冻结后用于后续会话 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Test Sessions 平均值）

#### ✅ Pearson Correlation 提升显著
| 条件 | 方法 | Overall r | 提升幅度（vs. FDG） |
|------|------|----------|------------------|
| **2D 反馈** | FDG | 0.5076 | — |
|          | **FRL (ours)** | **0.7181** | **↑41.5%**, *p=0.0005* |
| **VR 反馈** | FDG | 0.6420 | — |
|           | **FRL (ours)** | **0.7780** | **↑21.2%**, *p=0.0059* |

> 💡 FRL 在 Y/Z 轴表现尤为优异（如 VR 下 Y 轴达 0.9838）

#### ✅ RMSE 显著下降
| 条件 | 方法 | Overall RMSE | 下降幅度 |
|------|------|--------------|---------|
| **2D 反馈** | FDG | 0.0890 | — |
|          | **FRL** | **0.0532** | ↓40.2%, *p < 0.0001* |
| **VR 反馈** | FDG | 0.0714 | — |
|           | **FRL** | **0.0441** | ↓38.2%, *p < 0.0001* |

---

### 🔍 与其他基线方法对比

| 对比项 | 结果 |
|-------|------|
| **vs. FDG** | FRL 在相关性和 RMSE 上均显著优于固定解码器（大效应量，Cohen’s d > 1.8） |
| **vs. SAT** | 显著优于周期重校准策略（2D: ↑29.1% corr; VR: ↑20.6% corr） |
| **vs. WSR** | **甚至超越“同会话最优”性能**：
  - 2D 下 corr ↑6.2%，RMSE ↓23.9%
  - VR 下 corr ↑2.5%，RMSE ↓24.7%
  - 表明 RL 成功突破了传统解码器的能力边界 |

### 📉 性能稳定性分析
- FRL 在跨会话中波动最小，表现出更强的**跨会话稳定性（cross-session stability）**
- 有效缓解 EEG 信号漂移（non-stationarity）带来的性能退化

> ❗ 统计检验采用重复测量 ANOVA（2D）与 Friedman 检验（VR），并结合 **Holm-Bonferroni 校正** 控制多重比较错误

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **残差 RL 可有效纠正系统性解码误差**，即使基础解码器性能有限，也能通过后处理大幅提升轨迹精度。
2. **将 RL 与 EEG 解耦的设计提升了鲁棒性与实用性**，实现了“一次训练，长期使用”的可扩展 BCI 框架。
3. **离线训练即可达到高性能**，无需实时用户反馈或在线策略更新，适合临床与消费级应用。
4. **在 VR 环境中增益更稳定**，可能得益于更自然的感知反馈促进神经表征一致性。

### ⚠️ 局限性
1. **对动作缩放超参数敏感**：
   - 尺度过小 → 探索不足
   - 尺度过大 → 修正不稳定
2. **依赖显式时间信息**：
   - 当前框架需输入归一化时间 `t`，限制其在异步（asynchronous）BCI 中的应用
3. **任务范围受限**：
   - 当前仅验证于四目标离散指向任务，尚未推广至连续轨迹生成或多自由度控制
4. **未实现实时闭环验证**：
   - 所有实验为离线分析，实时性能有待验证

### 🔮 未来工作方向
1. **开发时间不变（time-invariant）的 RL 策略**，支持异步 BCI 应用
2. **扩展至更复杂运动模式**：
   - 更多目标方向
   - 自由空间连续轨迹
   - 双臂协调任务
3. **引入个性化自适应机制**：
   - 用户特定 RL agent 微调
   - 在线轻量更新策略
4. **开展实时闭环实验**：
   - 验证 CNN-LSTM-RL 在真实人机交互中的性能增益与用户学习效应
5. **探索其他残差学习范式**：
   - 如 Residual Imitation Learning 或 Offline RL from Demonstrations

---

## ✅ 总结

本论文提出了一种新颖且高效的 **CNN-LSTM-RL 两阶段解码框架**，通过**离线残差强化学习**对非侵入式 EEG 解码结果进行运动学修正，在不增加神经输入的前提下显著提升了 3D 运动想象的解码精度。实验表明该方法在 **2D 与 VR 环境下均取得显著性能提升（corr ↑21–41%，RMSE ↓38–40%）**，甚至超越传统解码器的理论上限，为 **虚拟交互、神经康复与假肢控制** 提供了一个**可扩展、鲁棒性强、易于部署**的新范式。

</details>

---

### 8. [Decomposing Runtime, Kernel, and Quantization Speedups via a Matched FP16 Intermediate: A Hardware-Conditioned Case Study on Four NVIDIA RTX A5000 GPUs](https://arxiv.org/abs/2607.11368)

**Authors**: Weijia Han, Lisha Qu  
**Category**: cs.DC  
**Published**: 2026-07-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.11368v1  

#### Abstract
Reported serving speedups from quantized kernels typically bundle the weight format, the kernel, and the inference runtime into one number. We present an attribution study on four NVIDIA RTX A5000 GPUs, 24 GiB each, on a single host with NVLink-bridged pairs. A matched intermediate stack that keeps ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Decomposing Runtime, Kernel, and Quantization Speedups via a Matched FP16 Intermediate

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在大语言模型（LLM）推理部署中，量化（quantization）、高效内核（kernel）和推理运行时（runtime）优化通常被作为一个“全栈”方案进行评估（如从 `HF Transformers` 切换到 `vLLM-Marlin`）。这种做法将 **runtime、kernel 和 quantization 的加速效果混为一谈**，导致无法回答以下实际问题：
- 真正的性能提升来自哪个部分？
- 如果不能接受量化带来的精度损失，是否仍应采用新的 runtime？
- 张量并行（tensor parallelism）在单卡可容纳模型时是否仍有价值？

### 提出了什么新方法或新思路
提出了一种 **通过“匹配的 FP16 中间态”进行归因分解（attribution decomposition）** 的新方法：
- 构造一个中间基准：**vLLM-FP16** —— 即使用 vLLM 的先进 runtime，但保留 FP16 权重和标准 GEMM 内核。
- 从而将端到端的加速拆分为两个独立因子：
  - **Runtime Factor**: `vLLM-FP16 / HF-FP16`
  - **Kernel + Quantization Factor**: `vLLM-Marlin / vLLM-FP16`
- 总体加速是两者的乘积，且该分解是代数恒等式，无需假设因子独立。

### 相比现有方法的优势
- **首次在 A5000 这类硬件上分离 runtime 与 kernel/quantization 的贡献**，揭示了传统“全栈加速”报告中的误导性。
- 提供了一个通用的归因框架，可用于分析其他系统（如 SGLang、AWQ、SmoothQuant 等）。
- 发现了 **张量并行在特定拓扑下不叠加（does not stack）** 的现象，并给出了机制解释。
- 揭示了 **多实例路由（multi-instance routing）的优劣取决于 workload 和 model size**，而非绝对最优。

---

## 2. 核心实验方法和设置

### 实验平台
- **硬件**：4 块 NVIDIA RTX A5000（每块 24GiB），通过 NVLink 成对桥接，跨对通信走 PCIe Gen4。
- **软件栈**：
  - **Baseline**: HuggingFace Transformers + SwiftServe（连续批处理 + 填充 KV 缓存）
  - **Optimized**: vLLM（异步引擎 + PagedAttention v2）
  - **量化后端**：GPTQ-INT4 + Marlin kernel（融合解包与 GEMM）

### 模型与权重格式
- 主要模型：**Llama-3.1-8B-Instruct**
  - FP16（约 16 GiB）
  - GPTQ-INT4（约 5.7 GiB）
- 对比模型（用于 cross-model 验证）：
  - Mistral-7B-v0.3
  - Qwen2.5-7B

### Prompt Pool
- 使用 **10 个精心设计的短提示**（平均约 12 tokens），涵盖技术问答与创意写作。
- 所有后端使用相同的提示列表，按固定顺序循环，以减少分布偏差。

### 评估指标
- **Throughput (tok/s)**：总输出 token 数 / 总 wall-clock 时间。
- **Decomposition Factors**：
  - `runtime factor = vLLM-FP16 / HF-FP16`
  - `kern+quant factor = vLLM-Marlin / vLLM-FP16`
  - `overall factor = vLLM-Marlin / HF-FP16`
- **Log Share**：衡量各因子对总加速的贡献比例：
  $$
  \text{log\_share}_i = \frac{\ln(\text{factor}_i)}{\ln(\text{overall factor})}
  $$

### 基线方法对比
| 后端 | Runtime | Kernel | Weight Format |
|------|--------|--------|---------------|
| HF-FP16 | HuggingFace + SwiftServe | Standard FP16 GEMM | FP16 |
| vLLM-FP16 | vLLM Async Engine | Standard FP16 GEMM | FP16 |
| vLLM-Marlin | vLLM Async Engine | Marlin INT4 GEMM | INT4 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Wide Batch, c=64, T=256）

| 指标 | 数值 |
|------|------|
| Overall Speedup (vLLM-Marlin / HF-FP16) | **10.62×**（生产场景）<br>**2.58×**（控制场景） |
| Runtime Factor (vLLM-FP16 / HF-FP16) | 7.56×（生产）<br>**1.90×**（控制） |
| Kern+Quant Factor (vLLM-Marlin / vLLM-FP16) | 1.40×（生产）<br>**1.36×**（控制） |
| Runtime Log Share | 86%（生产）<br>**67.9%**（控制） |

> ✅ **结论**：在宽批次负载下，**runtime 优化贡献了约三分之二的加速**，远超 kernel + quantization。

### 消融实验结果

#### （1）跨模型验证（Cross-Model Replication）
在 `Llama-8B`, `Mistral-7B`, `Qwen-7B` 上验证 `kern+quant factor` 的稳定性：
| Model | Kern+Quant Factor | Drift vs Llama |
|-------|-------------------|--------------|
| Llama-8B | 1.356 | — |
| Mistral-7B | 1.370 | +1.03% |
| Qwen-7B | 1.376 | **+1.46%** |

> ✅ **结论**：kernel + quantization 因子在相似规模模型上高度稳定（变化 < 1.5%），说明其可迁移性强。

#### （2）张量并行（Tensor Parallelism）扩展性测试
- 将单卡 `vLLM-Marlin` 扩展到 4 卡 TP=4。
- 理想加速：4×，实用目标：2×。

| Cell (c=64, T=256) | Marlin 1x | Marlin TP=4 | Stacking Factor |
|--------------------|-----------|------------|-----------------|
| Throughput (tok/s) | 2,391.44 | 3,735.41 | **1.56×** |

> ❌ **结论**：四卡张量并行仅带来 **1.56× 加速**，远低于双卡预期，更不用说四卡理想值。

#### （3）多实例 vs 张量并行（Routing Choice）
比较三种配置（总成本相同）：
- **Single Shard (TP=4)**
- **Four Independent Instances (MI)**
- **Single Card (1x)**

| Cell | TP=4 | 4×MI | MI / TP=4 |
|------|------|------|----------|
| c=64, T=256 | 3,735.41 | 5,442.78 | **1.46×** |

> ✅ **结论**：在高并发宽批次下，**多实例优于张量并行**。

#### （4）大模型反转（Sign Reversal at 70B）
当模型增大至 **Llama-70B**（无法放入单卡）：
- 比较：**One Sharded Instance (TP=4)** vs **Two Paired Instances (2×TP=2)**

| Workload | TP=4 | 2×TP=2 (NVLink) | Ratio |
|---------|------|----------------|-------|
| c=64, T=256 | 899.3 | 1,054.3 | **1.17×** |

> ✅ **结论**：在 70B 模型上，**两个双卡实例始终优于一个四卡分片实例**（平均快 7%），且优势不依赖于 NVLink。

#### （5）容量悬崖（Capacity Cliff）
- FP16 在 c=96 时 OOM。
- INT4（GPTQ 或 Marlin）可持续至 c=256。

| Stack | Max Concurrent Users | Utility Score (c=256) |
|-------|----------------------|------------------------|
| HF-FP16 | 64 | 15,416 |
| HF-GPTQ | 256 | 52,538 (**3.41×**) |
| vLLM-Marlin | 256 | 682,934 (**44.30×**) |

> ✅ **结论**：量化使可持续并发用户数提升约 **4 倍**；结合先进 runtime 可进一步放大效用。

---

## 4. 关键结论和发现

### 主要发现
1. **Runtime 优化是主要加速来源**  
   在宽批次负载下，**runtime 改进贡献了约 2/3 的端到端加速**，远高于 kernel + quantization 的 1.36×。

2. **张量并行在当前拓扑下不叠加**  
   四卡 TP=4 仅带来 1.5× 加速，主因是 **协调开销占每 token 时间的 ~80%**，且该开销由启动/同步延迟主导，而非带宽瓶颈。

3. **多实例路由策略是 workload-dependent 的**  
   - 小模型（8B）：
     - 长输出 → 张量并行更优
     - 宽批次 → 多实例更优
   - 大模型（70B）：
     - **所有 workload 下，两个双卡实例均优于一个四卡分片**

4. **量化显著提升容量边界**  
   INT4 使最大可持续并发用户数提升约 **4 倍**，突破 FP16 的内存悬崖。

5. **分解方法揭示“虚假归因”**  
   生产环境中观察到的 10.62× 加速中，大部分来自 batching 和 sampling 差异，而非纯 runtime 优势。

### 方法的局限性
- **单一硬件配置**：仅在 4× A5000（NVLink 成对）上验证，不适用于全交换结构（switch fabric）。
- **合成 Prompt Pool**：10 个短提示无法代表真实长文本分布。
- **采样模式不对称**：baseline 使用 greedy，vLLM 使用 ancestral sampling，虽论证影响小但仍为威胁。
- **未完全解耦 sub-factors**：runtime 因子仍混合了 PagedAttention、continuous batching 和 kernel 优化。

### 未来工作方向
1. 在具备全交换结构的设备（如 H100 SXM）上重复实验。
2. 测试更真实的 workload 分布（如 ShareGPT、LMSys）。
3. 添加更多 runtime 对比（如 TGI、SGLang）。
4. 探索 prefill 与 decode 阶段的相位拆分调度（phase-aware scheduling）。
5. 将此归因框架推广至其他量化方法（AWQ、SmoothQuant）和系统（TensorRT-LLM）。

---

> 📌 **方法论启示**：  
> **未来的 serving benchmark 应至少包含一个中间基线**（如保持 runtime 不变而更换 kernel，或反之），否则无法可信地归因性能提升来源。

</details>

---

### 9. [Multi-dimensional training-priority weighting based on physical information propagation paths: a unified residual-weighting framework for physics-informed neural networks](https://arxiv.org/abs/2607.11094)

**Authors**: Zhangyi Lian, Xinda Dong, Wenxuan Huo, Weifeng Huang, Gang Zhu, Qiang He  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.11094v1  

#### Abstract
Physics-informed neural networks (PINNs) have shown promise for solving partial differential equations (PDEs); however, their synchronous optimization treats residuals of different regions and constraints equally, which is inconsistent with the progressive "from source to response" physical informat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Multi-dimensional training-priority weighting based on physical information propagation paths

## 1. 论文的主要贡献和创新点

### 解决的问题
- **标准 PINN 的同步优化机制缺陷**：传统 Physics-Informed Neural Networks (PINNs) 将所有物理约束（如 PDE 残差、初始条件、边界条件）在训练过程中以等权重方式并行优化，这种“全局同步”策略忽略了物理信息沿特定路径（如时间推进、空间对流、边界驱动）逐步传播的本质。
- **缺乏统一的训练优先级框架**：现有因果训练方法（causal training）主要关注时间维度上的“过去 → 未来”顺序，但在空间和边界维度上缺乏系统性的建模，导致训练过程与真实物理演化不一致，影响收敛性和精度。

### 提出的新方法与思路
- **定义统一的“基于物理信息传播路径的训练优先级”**：
  - 引入 **Physical Information Propagation Path (P)** 的概念，指出任何 PDE 的解都沿着由控制方程和定解条件决定的信息路径从“源”向“响应”逐步形成。
  - 定义了三种维度的训练优先级：
    - **Temporal Priority**（时间优先级）：“past < future”
    - **Spatial Priority**（空间优先级）：“upstream < downstream”
    - **Boundary Priority**（边界优先级）：“boundary < interior”

- **提出多维优先级加权框架（Multi-dimensional Priority-Weighting Framework）**：
  - 在损失函数层面引入 **负指数残差权重（negative-exponential residual weights）**，将物理传播顺序转化为训练优先级。
  - 权重公式为：  
    $$
    w_m = \exp\left(-\epsilon \sum_{D_k <_p D_m} \mathcal{L}_k(\theta)\right)
    $$
    即后续区域的权重依赖于其前提区域的累积残差；只有当前提区域被充分拟合后，后续区域才被“释放”参与主导优化。

- **提出方向兼容性系数（Directional Compatibility Coefficient, $p$）用于多维耦合**：
  - 定义 $p(d_1, d_2) = d_1 \cdot d_2 \in [-1, 1]$ 来判断不同优先级方向是否可相容地进行乘法耦合。
  - 给出适用性准则：
    - **正交兼容（$p=0$）**：可乘法耦合，产生协同效应（如 time × space）
    - **同轴同向（$p=1$）**：冗余，保留其一即可
    - **同轴反向冲突（$p=-1$）**：不可直接耦合，会导致训练失稳，应选择主导方向或采用分区策略

### 相比现有方法的优势
- **统一性**：首次将时间、空间、边界三个维度的训练优先级纳入一个统一框架，而非仅限于时间因果。
- **无需修改网络架构**：仅通过损失层重加权实现，兼容各种 PINN 变体（如 MLP、Fourier Feature Net、Transformer-based PINNs）。
- **理论支撑强**：利用 **Neural Tangent Kernel (NTK)** 动力学分析证明标准 PINN 的收敛顺序由 NTK 谱决定，与物理传播路径无关；而所提方法通过重塑有效 NTK 谱，使前提区域获得更大的有效特征值，从而优先收敛。
- **可控计算开销**：额外计算仅为块级残差累加与指数运算，远低于前向/反向传播成本。

---

## 2. 核心实验方法和设置

### 使用的数据集 / 测试案例
论文选取了六类典型 PDE 问题作为基准测试案例，覆盖单维、多维耦合及冲突场景：

| 案例 | 类型 | 物理意义 |
|------|------|----------|
| **Stokes Flow Past a Cylinder** | 空间优先级验证 | 不可压缩低雷诺数绕流，存在明确主流方向 |
| **Heat Conduction** | 边界优先级验证 | 扩散型问题，边界为热源驱动 |
| **Burgers Equation** | 时间-空间耦合 | 非线性对流-扩散，兼具时间演化与空间传播 |
| **Allen-Cahn Equation** | 时间-边界耦合 | 相场演化，受时间发展与边界约束共同作用 |
| **Poiseuille Flow** | 冲突优先级建模（完全冲突） | 压力驱动管道流，出口边界与主流方向冲突（$p=-1$） |
| **Couette Flow** | 冲突优先级建模（局部冲突） | 剪切驱动流动，上下壁面边界与内部剪切传播存在竞争 |

### 实验设置和评估指标
- **统一配置**：
  - 硬件：NVIDIA RTX 4060 Laptop GPU
  - 框架：PyTorch + 自动微分（Auto-differentiation）
  - 优化器：Adam
  - 损失平衡：Grad-Norm 或 NTK-based weighting（作为 baseline 对比）
  - 总训练步数：200,000 steps
  - Batch Size：256–1024（依问题而定）

- **评估指标**：
  - 主要指标：**Relative $L^2$ Error**（相对 $L^2$ 误差）
  - 辅助分析工具：
    - 残差分布热图
    - NTK 特征值沿空间/时间方向的分布趋势
    - 收敛曲线（误差 vs. 训练时间/迭代次数）

### 基线方法对比
- **Standard PINN**：原始均等加权损失
- **NTK-based weighting**：基于神经切线核动态调整各 loss term 权重
- **RAR (Residual-based Adaptive Sampling)**：根据残差大小自适应采样高误差区域
- **Single-dimensional Priority**：仅应用单一维度加权（如仅时间或仅空间）
- **Multi-dimensional Coupled Priority**：本文提出的正交耦合策略
- **Conflict Coupling**：反向优先级直接乘法耦合（用于验证失败情形）

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 单维优先级显著提升精度
| 方法 | Stokes Flow ($L^2$) | Heat Conduction ($L^2$) |
|------|---------------------|-------------------------|
| Standard PINN | ~7.4×10⁻¹ | ~3.9×10⁻³ |
| NTK Weighting | ~3.2×10⁻¹ | ~2.0×10⁻³ |
| RAR | ~3.8×10⁻¹ | ~2.1×10⁻³ |
| **Spatial Priority** | **~1.7×10⁻¹** | — |
| **Boundary Priority** | — | **~5.0×10⁻⁴** |

> **结论**：空间/边界优先级加权在对应问题上表现最优，且误差分布更均匀，避免下游或近边界区域的集中误差。

#### ✅ 多维正交耦合进一步提升性能
| 方法 | Burgers Equation ($L^2$) | Allen-Cahn Equation ($L^2$) |
|------|--------------------------|----------------------------|
| Standard PINN | ~1.2×10⁻³ | ~6.8×10⁻³ |
| Time Priority | ~1.0×10⁻³ | ~3.0×10⁻³ |
| Space Priority | ~1.1×10⁻³ | — |
| Boundary Priority | — | ~2.0×10⁻³ |
| **Coupled Priority (Time×Space)** | **~4.3×10⁻⁴**（↓64% vs. time） | — |
| **Coupled Priority (Time×Boundary)** | — | **~1.7×10⁻³**（↓79% vs. time） |

> **结论**：多维耦合能捕捉复合物理机制，实现互补增益，显著降低整体误差。

#### ❌ 同轴反向冲突耦合导致性能崩溃
| 方法 | Poiseuille Flow ($L^2$) | Couette Flow ($L^2$) |
|------|------------------------|-----------------------|
| Standard PINN | ~1.1×10⁻⁵ | ~1.5×10⁻⁵ |
| Space Priority-x | ~4.6×10⁻⁶ | ~1.2×10⁻⁵ |
| Boundary Priority-x/y | ~2.9×10⁻⁶ / ~1.5×10⁻⁶ | ~1.1×10⁻⁶ |
| **Non-conflict Coupled** (e.g., x×y) | ~5.4×10⁻⁶ | ~7.7×10⁻⁶ |
| **Conflict Coupled** (opposite x-dir) | **~1.1×10⁻¹**（↑万倍！） | **~2.3×10⁻³**（↑百倍！） |

> **结论**：当两个相反方向的优先级在同一坐标轴上强行乘法耦合时（$p=-1$），模型收到相互竞争的信号，损失失去渐进结构，训练严重不稳定甚至发散。

### 消融实验结果
- **NTK 特征值分布可视化**（Fig. 3g, Fig. 6）：
  - Standard PINN：NTK 特征值在整个域内大致均匀分布，无明显方向偏好。
  - Priority-weighted PINN：特征值沿传播路径（如 boundary → interior 或 past → future）呈现清晰递减趋势，验证了“前提区域优先学习”的机制。
- **权重参数 $\epsilon$ 敏感性分析**：
  - $\epsilon$ 过大 → 后续区域长期被抑制，无法收敛；
  - $\epsilon$ 过小 → 权重趋于 1，退化为标准 PINN；
  - 最优 $\epsilon$ 应保证最后一个子区域在训练结束前完成“释放”。

---

## 4. 关键结论和发现

### 主要发现
1. **标准 PINN 不遵循物理信息传播顺序**：
   - 其收敛顺序由 **NTK spectrum** 决定，与物理机制无关，本质上是“无偏见”的同步优化。
2. **引入基于传播路径的训练优先级可显著改善训练行为**：
   - 通过残差加权将“premise before dependent”原则注入损失函数，使学习过程更符合物理直觉。
3. **多维优先级可通过乘法耦合实现协同增强，但前提是方向兼容**：
   - 正交方向（$p=0$）可安全耦合，提升精度；
   - 同轴反向（$p=-1$）则必须避免直接乘积，否则引发训练灾难。
4. **该框架具有良好的通用性和即插即用特性**：
   - 不改变网络结构，易于集成到现有 PINN 流程中。

### 方法的局限性
- **依赖清晰的传播路径结构**：对于椭圆型或强全局耦合问题（如泊松方程全域强关联），难以定义明确的方向性优先级。
- **对残差尺度敏感**：权重中的 $\epsilon$ 参数需根据具体问题调优，自动化程度有待提高。
- **分区策略影响效果**：子区域划分方式会影响权重平滑性与训练稳定性。
- **多优先级冲突时需人工干预**：目前仍需用户识别主导机制并手动选择保留哪个方向。

### 未来工作方向
- **自动识别物理信息传播路径**：结合符号推理或图神经网络，自动提取 PDE 中的主导传播方向。
- **自适应权重构造**：设计动态调整 $\epsilon$ 或自动划分区域的方法，减少人工调参。
- **扩展至高维与多物理场耦合问题**：应用于 Navier-Stokes、Maxwell 方程组等复杂系统。
- **探索逆问题中的应用**：在参数估计、材料识别等任务中验证优先级引导的有效性。
- **与其他训练加速技术融合**：如与 domain decomposition、active sampling、curriculum learning 结合，构建更强大的 PINN 训练范式。

---

> **总体评价**：本文提出了一个**物理动机清晰、理论分析严谨、实验验证充分**的统一训练优先级框架，为解决 PINN 训练不稳定和精度不足提供了新视角。它不仅是对“causal PINN”的推广，更是对“如何让深度学习尊重物理规律”这一根本问题的重要探索。

</details>

---

### 10. [Opti-Agent-Bench: Benchmarking End-to-End Optimization R&D Agents on Real-World Business Problems](https://arxiv.org/abs/2607.10768)

**Authors**: Yongchang Fu, Xinjie Huang, Chengjun Dai, Chengzhe Feng, Junshao Zhang, Hong Zhu  
**Category**: cs.AI  
**Published**: 2026-07-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.10768v1  

#### Abstract
LLM-based agents are increasingly deployed to solve optimization problems, yet existing benchmarks evaluate them on pre-structured mathematical formulations that bypass the most critical challenge: translating complex business requirements into correct models and solve efficiently. We introduce Opti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Opti-Agent-Bench: Benchmarking End-to-End Optimization R&D Agents on Real-World Business Problems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Large Language Models (LLMs)** 的优化代理（Optimization Agents）在解决现实世界业务优化问题时面临严重的能力评估缺失。现有基准（如 ORQA、MIPLIB-NL、OptiBench）存在三大缺陷：

- **问题描述过度结构化**：任务以学术语言呈现（如“最小化总延迟”），导致模型可通过模板匹配（template matching）而非真实理解解决问题。
- **单指标评估（Single-Metric Evaluation）**：仅通过目标函数值或求解器验证结果评分，掩盖了建模错误（如约束遗漏）、代码实现不一致等问题。
- **管道碎片化（Pipeline Fragmentation）**：缺乏对从商业需求到解决方案报告的完整研发流程（Problem Understanding → Formal Modeling → Implementation → Reporting）的端到端评估。

这些问题系统性地高估了当前 LLM 代理的真实能力。

---

### 提出了什么新方法或新思路

本文提出 **OPTI-AGENT-BENCH**，一个面向真实商业场景的端到端优化研发代理基准测试框架，其核心创新包括：

#### （1）**业务语义真实性设计（Business-Semantic Authenticity）**
- 所有任务均使用**纯业务语言**描述（如“换产耗时45分钟”、“质检团队每班次最多处理200米”），避免使用“整数规划”、“鲁棒优化”等学术术语。
- 引入 **Anti-Template Traps（反模板陷阱）**：每个任务包含至少一个结构性特征，使得标准模板（如 Cutting Stock、Mean-Variance）会生成不可行或次优解，迫使模型进行深度推理。

#### （2）**模块化评估架构（Modular Evaluation Architecture）**
将优化 R&D 流程分解为四个可独立评估的模块：
1. **Problem Understanding**：从业务描述中识别目标、变量、约束。
2. **Formal Modeling**：构建数学模型（集合、参数、变量、目标、约束）。
3. **Algorithm & Implementation**：选择并实现算法，输出可执行代码。
4. **Report Generation**：生成技术报告，解释建模逻辑与结果。

并引入**跨模块一致性检查**：
- **Understanding-Model Consistency (C12)**：模型是否忠实反映问题理解？
- **Model-Code Consistency (C23)**：代码是否准确实现数学模型？
- **Code-Report Consistency (C34)**：报告是否如实描述实现？

#### （3）**双层有效性框架 ORAC**
- **Problem Formulation Validity**：确保任务设计具备挑战性，满足四项原则：
  - TD.1 商业语义真实性
  - TD.2 反模板设计
  - TD.3 多约束耦合
  - TD.4 抗模式匹配
- **Solution Validity**：确保评分机制科学严谨，区分：
  - 可行 ≠ 正确（Feasible ≠ Correct）
  - 目标值好 ≠ 模型正确（Good Objective ≠ Good Model）
  - 代码可运行 ≠ 实现正确（Runnable Code ≠ Correct Implementation）
  - 报告连贯 ≠ 报告真实（Coherent Report ≠ Faithful Report）

---

### 相比现有方法的优势

| 维度 | 现有基准（如 MIPLIB-NL, ORQA） | OPTI-AGENT-BENCH |
|------|-------------------------------|------------------|
| 输入形式 | 半结构化/学术语言 | **纯业务语言** |
| 评估范围 | 仅建模或仅求解 | **端到端全流程** |
| 是否检测一致性 | 否 | ✅ 跨模块一致性检查 |
| 是否防模板匹配 | 弱 | ✅ 显式 Anti-Template Traps |
| 评估粒度 | 单一指标 | 多维度 + 严重性加权 |

> ✅ **首次实现了对 LLM 代理在真实商业优化场景下的综合、细粒度、抗欺骗评估。**

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **自建基准数据集 OPTI-AGENT-BENCH**，包含 **12 个工业级优化任务**，覆盖多种问题类型：
  - Integer Programming
  - Robust Optimization
  - Stochastic Programming
  - MDP
  - SOCP
  - Non-Convex Optimization
- 每个任务提供多尺度实例（小/中/大），用于测试可扩展性。
- 所有任务附带专家验证的参考模型（Oracle Model）和解决方案。

---

### 实验设置和评估指标

#### 评估协议
- **全管道评估**：模型仅接收业务描述和数据文件，需自主完成理解、建模、编码、报告。
- **部分管道评估**：支持任意连续子模块评估（如仅评估建模→实现）。

#### 评估指标体系（三级结构）

##### （1）模块级指标（Module-Level Metrics）
| 类别 | 指标 | 内容 |
|------|------|------|
| **ME (Modeling)** | ME.1: Model Correctness<br>ME.2: Model Parsimony<br>ME.3: Model-Code Consistency | 模型结构正确性、简洁性、与代码一致性 |
| **CE (Code)** | CE.1: Feasibility & Correctness<br>CE.2: Code Quality & Efficiency<br>CE.3: Robustness | 解的可行性、质量、代码效率、鲁棒性 |
| **RE (Report)** | RE.1: Completeness<br>RE.2: Clarity<br>RE.3: Reproducibility<br>RE.4: Limitation Acknowledgment | 报告完整性、清晰度、可复现性、局限性说明 |

##### （2）交叉模块一致性指标
- **CME.3**: Model-Code Consistency（嵌入 ME.3）
- **CRE.3**: Code-Report Consistency（嵌入 RE.3）

##### （3）聚合指标
- **总分 $ S_{\text{total}} = 0.35 \cdot SME + 0.40 \cdot SCE + 0.25 \cdot SRE $**（满分 5.0）
- **成功率（Success Rate）**：总分 ≥ 3.0 的任务占比
- **Anti-Template Score**：在反模板陷阱上的平均得分
- **Robustness Score**：扰动/放大实例上的性能保持率

#### 评分机制
- **自动化 + LLM-as-Judge 混合机制**：
  - ~60% 检查点自动评分（代码执行、约束校验）
  - ~40% 由 **Claude Opus 4.6** 作为“评审专家”打分
- 检查点按严重性加权：
  - Critical (w=3)：根本性错误（如遗漏关键约束）
  - Major (w=2)：显著降质
  - Minor (w=1)：非致命瑕疵

---

### 基线方法对比
评估了五个前沿 LLM：
- **Claude Sonnet 4**
- **Qwen3-Max**
- **DeepSeek-V3.2**
- **Qwen3.5-Plus**
- **Kimi-K2.5**

所有模型在同一时间预算下运行相同的 **RD-Agent 框架**，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（图5）

| 模型 | 总分（S_total） | ME.1（建模正确性） | CE.1（可行性） | RE.3（可复现性） |
|------|----------------|--------------------|---------------|----------------|
| **Claude Sonnet 4** | **~2.9** | ~1.65 | ~3.8 | ~4.0 |
| Qwen3-Max | ~2.0 | ~1.8 | ~3.5 | ~4.9 |
| DeepSeek-V3.2 | ~1.9 | ~1.7 | ~3.4 | ~4.8 |
| Qwen3.5-Plus | ~1.0 | ~0.8 | ~2.5 | ~2.8 |
| Kimi-K2.5 | ~1.0 | ~0.7 | ~0.3 | ~0.3 |

> 🔺 **即使最强模型（Claude）也远未达到满分（5.0），表明当前 LLM 在真实优化 R&D 中仍处于初级阶段。**

---

### 与基线方法的对比结果

- **Claude 表现最佳**，尤其在 **Model Parsimony (ME.2)** 和 **Limitation Acknowledgment (RE.4)** 上显著领先，显示其更强的问题抽象与自我认知能力。
- **Qwen3-Max 与 DeepSeek-V3.2** 在代码层面表现接近，但建模能力弱于 Claude。
- **Qwen3.5-Plus** 存在明显 bug（如位操作错误），导致实际目标值低于声称值，被自动检测器捕获。
- **Kimi-K2.5** 几乎无有效求解能力（Top-level code 仅为 evaluator），报告严重失实（RE.3 ≈ 0），表现出“技术幻觉”。

---

### 消融实验与失败模式分析（7.4节）

#### 主要失败模式分类：
| 阶段 | 失败模式 | 示例 |
|------|--------|------|
| **Problem Understanding** | Template Anchoring<br>Constraint Blindness | 将“卷材调度”误判为 Cutting Stock；忽略“质检瓶颈”约束 |
| **Formal Modeling** | Constraint Omission<br>Structural Mismatch | 遗漏隐含耦合约束；将随机问题建模为确定性 |
| **Implementation** | Model-Code Divergence<br>Scalability Collapse | 数学模型中有约束，代码中未实现；小规模可解，大规模崩溃 |
| **Reporting** | Hallucinated Explanations<br>Inconsistent Results | 报告中声称使用 SDP，实际为枚举；报告目标值无法复现 |

> ✅ **传统单指标评估无法暴露这些错误，而 OPTI-AGENT-BENCH 成功揭示。**

---

## 4. 关键结论和发现

### 主要发现
1. **当前 LLM 代理严重依赖模板匹配**，面对反模板陷阱时普遍失效，**建模正确性（ME.1）是最大瓶颈**。
2. **代码实现往往忠实于错误模型**：模型-代码一致性（ME.3）得分较高，但前提是模型本身错误。
3. **报告质量与技术质量脱节**：部分模型能写出高质量报告，但内容与实现不符（如 Kimi-K2.5）。
4. **Claude 展现出相对优势**，尤其在避免冗余建模和诚实承认局限方面，显示出更接近“研究者”的行为模式。
5. **端到端评估至关重要**：单一环节的高分不能代表整体能力，必须考察全流程一致性。

---

### 方法的局限性
1. **实例规模有限**：最大实例仍在 $10^4$–$10^5$ 变量级别，尚未达到 MIPLIB 中 $10^6$ 级别的超大规模压力测试。
2. **静态问题设定**：未模拟现实中与利益相关者的动态交互与需求变更。
3. **依赖 LLM 评审员**：尽管采用结构化打分卡，但仍存在主观性风险，未来可探索更多自动化验证手段。

---

### 未来工作方向
1. **引入更大规模、更复杂工业实例**，进一步拉大模型差距。
2. **支持动态问题演化**：允许任务描述随时间更新，测试代理的适应能力。
3. **开放社区共建**：鼓励研究人员提交符合 ORAC 标准的新任务，持续扩展基准生态。
4. **开发专用优化代理架构**：基于本基准的失败模式，设计更擅长建模、一致性维护的新型代理框架。

---

> 📌 **总结**：  
> OPTI-AGENT-BENCH 不仅是一个新基准，更是一种**评估范式的转变**——从“能否求解已知结构”转向“能否从模糊业务中提炼正确结构”。它揭示了当前 LLM 代理在真实优化研发中的根本短板，并为下一代智能优化系统的研发提供了明确的方向与衡量标准。

</details>

---

### 11. [SCALECUA: Scaling Computer Use Agents with Verifiable Task Synthesis and Efficient Online RL](https://arxiv.org/abs/2607.11185)

**Authors**: Bowen Lv, Xiao Liu, Yanyu Ren, Hanyu Lai, Bohao Jing, Hanchen Zhang, Yanxiao Zhao, Shuntian Yao, Jie Tang, Yuxiao Dong  
**Category**: cs.AI  
**Published**: 2026-07-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.11185v1  

#### Abstract
Computer use agents (CUAs) are emerging as a powerful interface for automating complex digital workflows through visual perception and GUI execution. Online reinforcement learning with verifiable rewards (RLVR) has emerged as a key direction for scaling their capabilities. However, this paradigm is ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SCALECUA: Scaling Computer Use Agents with Verifiable Task Synthesis and Efficient Online RL

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前 **Computer Use Agents (CUAs)** 在通过视觉感知和 GUI 交互自动化数字任务方面展现出巨大潜力，但其能力扩展受到两大瓶颈限制：

1. **Verifiable Data Scarcity**：缺乏大规模、可验证的 GUI 任务数据。传统方法依赖人工标注或非可执行奖励，难以支持强化学习（RL）所需的自动反馈。
2. **Online RL Inefficiency**：在线强化学习在多轮交互中存在采样效率低、训练开销大等问题，尤其是长序列任务下 rollout 和训练引擎压力巨大。

---

### 🚀 提出的新方法与创新思路

为突破上述瓶颈，论文提出 **SCALECUA**，一个统一框架，结合 **可验证任务合成** 与 **高效在线 RL**，实现对 CUA 的规模化训练。

#### 主要组件如下：

| 组件 | 创新点 |
|------|--------|
| **VERIGEN** | 一种端到端的可验证任务生成框架，通过多智能体反馈循环（proposer/judger/checker）在真实 Docker 环境中迭代生成带可执行 judge 函数的任务。支持轨迹引导的任务重构（失败任务拆解为子任务，成功任务组合为复杂任务），确保任务难度适中且可验证。 |
| **Frontier Sampling** | 一种轻量级采样策略，基于每个任务的指数移动平均（EMA）成功率动态调整采样权重，将 rollout 资源集中于模型“当前能力边界”上的任务（success rate ~0.5），最大化每批次的学习信号。 |
| **Visual Context Segmentation** | 一种滑动窗口式的多模态轨迹处理机制，在保留文本连续性的同时，仅保留最近 K 张截图作为视觉上下文。有效控制视觉 token 增长，缓解 rollout 推理延迟和训练样本膨胀问题。 |

---

### 🔍 相比现有方法的优势

| 方面 | SCALECUA 优势 |
|------|----------------|
| **数据规模与质量** | 生成 **24K+ 可验证任务** 和近 **3K 高质量 RL 任务**，远超以往方法（如 OS-Genesis、GUI-Genesis）。任务覆盖 OSWorld 和 ScienceBoard 所有领域。 |
| **训练效率** | Visual Context Segmentation 实现 **2.83× 端到端训练加速**，显著优于 step-wise 分解。 |
| **样本效率** | Frontier Sampling 避免在已掌握或完全无法完成的任务上浪费 rollout，收敛更快。 |
| **通用性** | 在多个 VLM（Qwen3.5-9B、Qwen3-VL-8B、GLM-4.6V-Flash）上均取得一致提升，表明方法具有模型无关性。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **OSWorld**：真实操作系统环境下的开放任务基准，涵盖文件管理、浏览器操作、办公软件等 10 类应用。
- **ScienceBoard**：面向科研场景的专业软件任务基准，包括 KAlgebra（符号代数）、ChimeraX（分子可视化）、GrassGIS（地理信息系统）、Celestia（天文模拟）、TeXstudio（LaTeX 编辑）、Lean（定理证明）等。

---

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| **模型基础** | 基于 Qwen3.5-9B、Qwen3-VL-8B-Thinking、GLM-4.6V-Flash 等 VLM 进行微调 |
| **训练框架** | 使用 AgentRL 框架，集成 GRPO + DAPO-style asymmetric clipping |
| **Rollout 引擎** | vLLM（TP=2） |
| **训练引擎** | Megatron-LM（TP=4） |
| **最大回合步数** | 50 turns |
| **评估方式** | 多次独立 rollout（通常 4 次）取平均成功率（pass rate） |
| **评估指标** | **Success Rate (%)**，按任务类别分组报告，并计算 Overall 得分 |

---

### 🆚 基线方法对比

| 类型 | 代表方法 |
|------|---------|
| **商业闭源模型** | GPT-5.4, Claude Opus 4.6, Claude Sonnet 4.5, Seed-1.8, OpenAI CUA |
| **开源模型** | Kimi K2.5, EvoCUA-32B, ComputerRL-9B, UI-TARS-2, DART-GUI-7B, OpenCUA-72B |

SCALECUA 以 **9B 级别模型** 对标更大参数量甚至闭源系统。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

| 模型 | OSWorld (%) | ScienceBoard (%) |
|------|-------------|------------------|
| **SCALECUA-Qwen3.5-9B** | **68.7** | **54.0** |
| SCALECUA-Qwen3-VL-8B | 67.7 | 51.9 |
| SCALECUA-GLM-4.6V-Flash | 66.5 | 49.4 |

> ✅ **State-of-the-Art 成绩**：在所有开源 CUA 中排名第一。

---

### 🔁 与基线方法对比

#### 在 OSWorld 上的表现（Overall）：

| 模型 | 得分 |
|------|-----|
| Human | 72.4 |
| GPT-5.4 | 75.0 |
| **SCALECUA-Qwen3.5-9B** | **68.7** |
| Kimi K2.5 | 63.3 |
| EvoCUA-32B | 56.7 |
| ComputerRL-9B | 48.0 |
| Qwen3.5-9B (base) | 41.8 |

> ✅ **超越最强开源模型 Kimi K2.5（+5.4%）**  
> ✅ **超越闭源 Claude Sonnet 4.5（62.9%）**  
> ✅ **相比 base 模型提升 +26.9%**

#### 在 ScienceBoard 上的表现：

| 模型 | 得分 |
|------|-----|
| Human | 60.3 |
| Claude Opus 4.6 | 52.7 |
| **SCALECUA-Qwen3.5-9B** | **54.0** |

> ✅ **首次在 ScienceBoard 上超越闭源 Claude Opus 4.6**  
> ✅ 特别在 **TeXstudio（86.7%）** 和 **Lean 定理证明（19.0%）** 表现突出

---

### 🔍 消融实验结果（Ablation Study）

使用 Qwen3.5-9B 在 OSWorld 上进行消融：

| 配置 | OSWorld (%) |
|------|------------|
| **Full SCALECUA** | **68.7** |
| w/o VERIGEN（仅用 base 数据） | 43.9 |
| w/o Frontier Sampling | 63.7 |
| w/o Visual Context Segmentation | 62.2 |

> ✅ 三大模块均有显著贡献，联合使用效果最佳。

#### Frontier Sampling 效果分析

- 相比 uniform sampling 和 curriculum learning，Frontier Sampling 能持续获得更高 reward，避免早停。
- 图 6(a) 显示其 reward 曲线更平缓上升，说明始终聚焦学习前沿。

#### Visual Context Segmentation 加速效果

| 阶段 | Step-wise (s) | Sliding-Window K=5 (s) | 加速比 |
|------|---------------|------------------------|--------|
| Actor Update | 485 | 154 | ↓68.2% |
| Reference Policy | 241 | 88 | ↓63.5% |
| **Total per step** | **750** | **265** | **2.83×** |

> ✅ 在不牺牲性能的前提下实现 **2.83× 端到端训练加速**

#### 滑动窗口大小 $ K $ 影响（Pass@1）

| K | Pass@1 |
|---|--------|
| 1 | 56.4% |
| 3 | 58.3% |
| **5** | **58.9%** |
| 8 | 57.8% |
| 15 | 56.8% |

> ✅ **K=5 时性能最优**，过小导致遗忘，过大引入噪声干扰。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **可验证任务可以大规模自动生成**：VERIGEN 通过多智能体协作 + 实时 Docker 执行验证，实现了高质量、可执行 judge 的 GUI 任务规模化生产。
2. **Frontier Sampling 显著提升样本效率**：动态追踪模型能力边界，使 rollout 资源集中在最具信息量的任务上，加快收敛。
3. **Visual Context Segmentation 平衡了 rollout 与训练压力**：滑动窗口机制在保持跨步推理能力的同时，大幅降低训练成本。
4. **SCALECUA 是模型无关的增强方案**：在多种 VLM 上均带来显著增益，具备良好泛化性。
5. **开源模型可逼近甚至超越闭源系统**：SCALECUA-Qwen3.5-9B 在 OSWorld 和 ScienceBoard 上表现优于多个商业模型。

---

### ⚠️ 局限性（Limitations）

1. **回合长度受限**：每个 episode 最多 50 步，可能不足以覆盖极长流程任务。
2. **操作系统限制**：目前仅在 Ubuntu 桌面环境验证，尚未扩展至 Windows 或 macOS。
3. **模型尺度探索不足**：主要验证 8B–9B 模型，更大或更小规模的行为尚待研究。

---

### 🔮 未来工作方向

1. 支持更长 horizons 的任务建模与记忆机制。
2. 扩展到更多操作系统平台（Windows/macOS）及移动端。
3. 探索 SCALECUA 在更大规模模型（如 70B+）上的表现。
4. 将 VERIGEN 应用于其他专业领域（医疗、工程 CAD 等）的自动化任务构建。

---

> 🔗 **代码、模型与数据集已开源**：[https://github.com/THUDM/SCALE-CUA](https://github.com/THUDM/SCALE-CUA)

</details>

---

### 12. [Efficient Test-Time Optimization for Multi-Agent Proof Autoformalization](https://arxiv.org/abs/2607.11307)

**Authors**: Tian-Shuo Liu, Shiyuan Zhang, Zijie Geng, Haoyu Liu, Runjie Xu, Pengyuan Wang, Lei Yuan, Yang Yu  
**Category**: cs.AI  
**Published**: 2026-07-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.11307v1  

#### Abstract
Full-proof autoformalization bridges extensive mathematical proofs in natural language with formally validated reasoning, offering a pathway to elevate the ceiling of verifiable mathematical reasoning. Unlike statement-level formalization, proof autoformalization is a long-horizon challenge requirin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Test-Time Optimization for Multi-Agent Proof Autoformalization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **全证明自动形式化（Full-proof autoformalization）** 是一个长视野任务，需要将自然语言数学证明精确地转化为形式化系统（如 Lean）中可验证的代码。
- 现有方法存在两大瓶颈：
  - **依赖昂贵的模型训练**（training-based），难以扩展；
  - 或在推理时进行**无引导的反复修复**（excessive unguided repair），效率低下且反馈稀疏。
- 特别是，在多阶段流水线中，**下游验证信号无法有效指导上游模块优化**，导致试错成本高。

### **提出了什么新方法或新思路**
提出 **TOMAP**（Test-time Optimization for Multi-Agent Proof autoformalization）框架，其核心思想如下：

#### ✅ **多智能体分解架构（Decomposer-Formalizer-Prover Pipeline）**
- 将全证明形式化拆解为三个专业化 LLM 智能体：
  - **Decomposer**：将非正式证明分解为原子化的子命题（proof units）；
  - **Formalizer**：将每个子命题翻译成 Lean 形式语句；
  - **Prover**：生成战术脚本完成证明。

#### ✅ **瓶颈分析驱动的测试时优化（Bottleneck-focused Test-Time Optimization）**
- 通过弱链路分析（weak-link analysis）发现：**Decomposer 是整个流程的关键瓶颈**。
- 下游 Formalizer 和 Prover 的成功率高度依赖于 Decomposer 输出的**原子性、自包含性和依赖一致性**。
- 因此，TOMAP **仅对 Decomposer 进行测试时优化**，而将 Formalizer 和 Prover 视为“冻结执行器”（frozen executor）。

#### ✅ **基于 GEPA 的双层优化机制**
- 引入两阶段搜索策略：
  1. **廉价代理反馈层（Cheap rubric feedback）**：
     - 使用 LLM-based judge 对候选分解方案按多个维度打分（语义忠实度、可证性、Lean友好性）；
     - 构建帕累托前沿（Pareto frontier）指导进化方向。
  2. **昂贵真实验证层（Expensive Lean verification）**：
     - 只有当候选分解通过质量阈值（gate threshold）后，才提交给完整的 Formalizer-Prover 流水线进行 Lean 验证。
- 该设计显著提升了单位计算资源下的优化效率。

---

### **相比现有方法的优势**
| 维度 | TOMAP 的优势 |
|------|-------------|
| **效率** | 避免盲目重试整个流水线，聚焦于最关键模块（Decomposer）；减少昂贵的 Lean 调用次数。 |
| **效果** | 显著提升最终 Lean 编译通过率与语义忠实度（Semantic Faithfulness）。 |
| **通用性** | 不依赖额外训练，适用于任意预训练 LLM，属于纯 test-time optimization 方法。 |
| **可控性** | 支持灵活调整迭代轮数以平衡时间开销与性能。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 描述 |
|-------|------|
| **PROOFFLOWBENCH** | 本科级别数学问题，共 184 道题，专为评估证明级形式化设计，强调逻辑结构而非计算技巧。 |
| **miniF2F** | 高中数学竞赛水平基准，含 244 道题目，广泛用于评估数学推理能力。 |

### **实验设置与评估指标**

#### ✅ **评估指标**
| 指标 | 含义 |
|-----|------|
| **Syntactic Correctness (SC)** | Lean 代码是否能被编译器完全接受（Pass@Lean）。 |
| **Semantic Faithfulness (SF)** | 形式化过程是否忠实还原原始证明的每一步逻辑，使用 LLM-as-a-judge（Qwen3-235B）进行二分类判断。 |
| **SC ^ SF (Joint Score)** | 同时满足语法正确与语义忠实的比例，反映综合质量。 |
| **Time (s)** | 平均每道题所耗墙钟时间（wall-clock time）。 |
| **Output Tokens (OT)** | 每道题平均生成 token 数量，衡量推理开销。 |

#### ✅ **模型配置**
- **TOMAP 使用两种自然语言模型变体**：
  - **TOMAP-Qwen30B**：基于开源强模型 Qwen3-30B-A3B-Instruct-2507；
  - **TOMAP-Gemini**：基于闭源前沿模型 Gemini3-Pro。
- Formalizer 和 Prover 固定使用 **Goedel-Formalizer-V2-32B** 与 **Goedel-Prover-V2-32B**。

#### ✅ **基线方法对比**
| 类型 | 方法 | 简介 |
|------|------|------|
| **Training-Based** | **ProofBridge** | 基于检索增强微调模型联合翻译，Report Pass@64。 |
| **Training-Free** | **Codex End2End** | 单步端到端翻译，无中间分解。 |
| | **M2F** | 基于 Codex 的验证器认证 refine 框架。 |
| | **Monotonic** | 无需形式参考的迭代 refine 方法。 |
| | **PROOFFLOW** | 当前最优的多智能体流水线方法，支持失败轨迹回溯重试。 |
| | **Codex D-Correction** | 在 Decomposer 上应用 10 轮自我修正。 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| 方法 | PROOFFLOWBENCH (SC^SF) ↑ | miniF2F (SC^SF) ↑ | Time(s) ↓ | OT(k) ↓ |
|------|--------------------------|------------------|-----------|---------|
| ProofBridge | 9.24% | 28.69% | 113.48 | 87.39 |
| Codex End2End | 21.20% | 47.54% | 104.22 | 21.34 |
| PROOFFLOW | 15.76% | 35.25% | 2348.48 | 90.06 |
| **TOMAP-Qwen30B** | **21.10%** | **29.92%** | 610.10 | 98.81 |
| **TOMAP-Gemini** | **40.22% (+119.0%)** | **55.74% (+18.2%)** | 850.27 | 139.25 |

> 🔥 **TOMAP-Gemini 在 PROOFFLOWBENCH 上相较最好先前方法（PROOFFLOW）提升达 119.0%！**

#### ✅ **核心发现**
- TOMAP 在 **语义忠实度（SF）** 上表现尤为突出，说明其分解优化有效保留了原证明结构。
- 相比端到端方法（如 Codex End2End），TOMAP 更注重**步骤对齐**，避免“跳步”或“逻辑简化”带来的虚假通过。
- 尽管输出 token 更多，但总耗时远低于 PROOFFLOW，表明其优化更高效。

---

### **消融实验结果（Table 2）**

研究不同最大优化轮数（Max Rounds）的影响：

| 方法 | Max Rounds | PROOFFLOWBENCH (SC) |
|------|------------|---------------------|
| TOMAP-Qwen30B | 1 | 39.67% |
| | 5 | 59.24% |
| | **10** | **66.30%** |
| TOMAP-Gemini | 1 | 35.33% |
| | 5 | 56.52% |
| | **10** | **63.59%** |

#### ✅ **关键观察**
- 多数性能增益集中在前几轮分解演化中，后续收益递减。
- 表明 TOMAP 具有良好的**早期收敛特性**，适合根据预算选择迭代次数。
- 实践建议：可在低延迟场景下使用较小迭代数（如 3–5），在追求最高准确率时启用完整 10 轮。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Decomposer 是多智能体形式化流水线中的关键瓶颈**：
   - 分解质量直接决定下游 Formalizer 和 Prover 是否可行。
   - 优化应优先集中于提升分解的**原子性、自包含性与依赖显式化**。

2. **测试时优化应“有的放矢”而非“全面重试”**：
   - TOMAP 证明：将有限的 test-time compute 投入最关键环节（Decomposer），比在整个 pipeline 上盲目 retry 更高效。

3. **代理反馈（rubric feedback）可有效引导搜索**：
   - 利用 LLM judge 提供密集、低成本的多维评分（faithful, provable, Lean-friendly），构建帕累托前沿，实现定向进化。

4. **TOMAP 实现了更高性价比的形式化**：
   - 在更低的时间成本下，实现了更高的 SC^SF 联合得分，尤其在复杂本科级问题上优势明显。

---

### **局限性**
1. **评估规模受限**：
   - 实验仅限于 benchmark 规模的问题（如 PROOFFLOWBENCH 的 184 题），尚未验证于长篇研究级数学证明。
2. **假设输入证明正确**：
   - 当前框架不处理原始证明本身错误或缺失的情况，无法检测并修复 faulty informal proofs。
3. **依赖高质量 rubric judge**：
   - 若 LLM judge 自身不可靠，可能导致误导性优化方向。

---

### **未来工作方向**
1. **扩展至更长、更复杂的数学文本**（如整篇论文或教材章节）；
2. **引入 proof repair 能力**，自动识别并修正原始证明中的漏洞；
3. **动态调整 gate threshold**，实现自适应验证触发机制；
4. **探索其他形式化系统**（如 Isabelle/HOL、Coq）上的迁移应用。

--- 

> 📌 **总结一句话**：  
> TOMAP 通过识别 **Decomposer 为瓶颈**，提出一种高效的 **test-time optimization 框架**，利用 **LLM-based rubric + Pareto evolution** 指导分解优化，**大幅超越现有方法**，同时控制推理成本，为大规模数学知识自动化验证提供了新范式。

</details>

---

### 13. [Neural Discovery of Memory and Nonlocal Kernels in Integro-Differential Equations with Constrained Kolmogorov--Arnold Networks](https://arxiv.org/abs/2607.11110)

**Authors**: Aruzhan Tleubek, Salah A Faroughi  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.11110v1  

#### Abstract
Discovering the memory or nonlocal kernel governing an integro-differential equation (IDE) from sparse and noisy observations is an ill-posed inverse problem. Existing identification methods often rely on problem-specific analytical derivations, specialized observation requirements, or restrictive a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Neural Discovery of Memory and Nonlocal Kernels in Integro-Differential Equations with Constrained Kolmogorov–Arnold Networks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文致力于解决**从稀疏且含噪声的时空观测数据中识别积分微分方程（Integro-Differential Equations, IDEs）中的记忆核（memory kernel）或非局部核（nonlocal kernel）**这一病态逆问题。这类问题在粘弹性力学、反常扩散、热传导等物理系统建模中广泛存在，但由于核函数未知且观测受限，传统方法往往依赖于特定解析推导、强假设或特殊测量条件，适用范围有限。

### 提出的新方法与新思路
作者提出了一种基于**可微求解器（differentiable solver）的框架**，将未知核函数嵌入到数值求解过程中，并通过自动微分进行端到端优化。其核心创新在于引入两种受约束的 **Kolmogorov-Arnold Network (KAN)** 架构来参数化未知核函数：

- **Monotone-Convex KAN (MC-KAN)**：采用**Bernstein多项式基底**，通过对系数施加显式约束（如单调递减、凸性），**在架构层面硬性保证**核函数满足物理先验（正性、单调递减、凸性）。
- **Chebyshev-based KAN (Cheb-KAN)**：作为软约束基线，使用**Chebyshev多项式**表示核函数，通过损失函数中的**软惩罚项**鼓励相同的物理性质。

此外，在训练完成后，使用 **symbolic regression (PySR)** 将学习到的神经网络核转化为可解释的闭式表达式。

### 相比现有方法的优势
- **通用性强**：不依赖于特定方程的解析技巧或边界测量，适用于多种类型的IDE。
- **无需伴随方程**：采用“discretize-then-optimize”策略，直接通过自动微分反向传播梯度，避免了为每个IDE推导复杂连续伴随方程的需求。
- **物理一致性更强**：特别是MC-KAN通过构造性约束确保输出始终符合物理规律，相比软约束更鲁棒。
- **兼具准确性与可解释性**：不仅恢复高精度核函数，还能通过symbolic regression获得人类可读的数学公式。

---

## 2. 核心实验方法和设置

### 使用的数据集
所有实验均基于**合成数据**生成，用于模拟真实世界中稀疏、带噪的观测场景。具体包括三个基准问题：
1. **1D Volterra IDE**：描述具有记忆效应的状态演化，核函数为指数衰减 $ K(\tau) = e^{-\tau} $。
2. **1D Viscoelastic Wave PIDE**：模拟粘弹性杆自由振动，核函数为Kohlrausch-Williams-Watts (KWW) 拉伸指数形式 $ K(\tau) = \exp(-(\tau/T_0)^\beta) $，其中 $\beta=0.5$。
3. **2D Nonlocal Reaction-Diffusion Equation**：研究空间非局部相互作用，核函数为各向异性耦合的拉伸指数形式，包含交叉项。

### 实验设置与评估指标
- **输入归一化**：对滞后变量 $\tau$ 进行线性或幂律归一化以提升分辨率。
- **训练流程**：结合 Adam 和 L-BFGS 优化器；使用自动微分驱动的可微求解器计算损失梯度。
- **评估指标**：
  - **相对 $L^2$ 误差**：用于衡量解 $u$ 和核 $K$ 的重建精度：
    $$
    \epsilon(v) = \frac{\|v_{\text{pred}} - v_{\text{true}}\|_2}{\|v_{\text{true}}\|_2}
    $$
  - **symbolic regression 结果**：比较提取出的闭式表达式是否匹配真实函数形式及其参数。

### 基线方法对比
本文主要对比的是两种内部变体：
- **MC-KAN (hard-constrained)** vs. **Cheb-KAN (soft-constrained)**
二者保持相近参数量和网络深度，公平比较硬约束与软约束的效果。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ Experiment I: 1D Volterra IDE（含噪声）
| 噪声水平 $\sigma$ | 方法       | $\epsilon(u)$ (%) | $\epsilon(K)$ (%) |
|------------------|------------|--------------------|--------------------|
| 0.00             | MC-KAN     | 0.02               | **0.85**           |
|                  | Cheb-KAN   | 0.05               | 1.83               |
| 0.15             | MC-KAN     | 1.85               | **8.78**           |
|                  | Cheb-KAN   | 1.90               | 11.64              |

> **结论**：MC-KAN 在所有噪声水平下均取得更低的核重建误差（最多降低 **54%**），solution 误差相当。

#### ✅ Experiment II: 1D Viscoelastic Wave PIDE（时间稀疏）
| 快照数 $N_s$ | 方法       | $\epsilon(u)$ (%) | $\epsilon(K)$ (%) |
|--------------|------------|--------------------|--------------------|
| 11           | MC-KAN     | 0.383              | **1.204**          |
|              | Cheb-KAN   | 0.378              | 1.938              |

> **结论**：即使仅用 11 个时间快照（约 1% 数据），MC-KAN 仍显著优于 Cheb-KAN（核误差低约 **40%**），且对稀疏性更稳健。

#### ✅ Experiment III: 2D Nonlocal Reaction-Diffusion（稀疏+噪声）
| 噪声 $\sigma$ | 方法       | $\epsilon(K)$ (%) @ $\sigma=0.15$ |
|---------------|------------|------------------------------------|
| 0.15          | MC-KAN (linear) | **12.36**                        |
|               | Cheb-KAN       | 21.95                            |

> **结论**：在最严苛条件下（7帧+32×32网格+$\sigma=0.15$），MC-KAN 将核误差降低近 **44%**。当增加至 25 帧并使用幂律归一化后，MC-KAN 表现进一步提升（$\epsilon(K)=8.03\%$），而 Cheb-KAN 仍较高（14.55%）。

### 消融实验结果
- **输入归一化方式的影响**：
  - 线性归一化在原点附近分辨率不足，导致峰值低估。
  - 引入**可学习或固定幂律归一化**（如 $\alpha=0.75$）能集中分辨力于小滞后区域，显著改善重建质量，尤其在低噪声时效果明显。
- **symbolic regression 成功率**：
  - 对于 1D 问题，两种方法均能成功恢复正确函数形式（如 $e^{-c\tau}$ 或 KWW 形式）。
  - 对于 2D 耦合核，需分步处理：先拟合一维切片，再分析残差恢复耦合项。MC-KAN 更稳定地恢复出指数结构，而 Cheb-KAN 在高噪声下可能出现非物理解（如幂律形式）。

---

## 4. 关键结论和发现

### 主要发现
1. **硬约束架构（MC-KAN）在多维、稀疏、含噪场景下显著优于软约束方法**：
   - 物理性质由构造保证，避免了软惩罚失效导致的振荡或非物理解。
   - 在 2D 非局部问题中，MC-KAN 的核重建误差系统性低于 Cheb-KAN，差距随噪声增大而扩大。

2. **所提框架具备良好的泛化能力与可解释性**：
   - 可统一处理时间记忆与空间非局部核识别。
   - 经 symbolic regression 后，多数情况下能准确还原真实核的闭式表达式。

3. **“discretize-then-optimize”策略有效规避了伴随方程设计难题**：
   - 利用自动微分穿透整个数值求解流程，简化实现并增强通用性。

### 方法的局限性
- 当前 MC-KAN 仅适用于满足**正性、单调递减、凸性**的核函数，无法处理振荡型、变号或非单调核（如周期性记忆）。
- 对于高维耦合核，symbolic regression 难以直接发现完整表达式，需借助分步策略，限制了全自动发现能力。
- 幂律归一化虽提升性能，但最优指数可能依赖于具体问题，需额外调参或预训练确定。

### 未来工作方向
- 扩展 MC-KAN 至更广泛的物理核类别（如振荡、长尾、符号变化）。
- 开发能够直接发现多变量耦合结构的 symbolic discovery 方法。
- 将本框架应用于真实实验数据（如流变学、脑成像、材料科学），验证其实际有效性。
- 探索与其他降维或贝叶斯不确定性量化技术结合，提升小样本下的可靠性。

--- 

> **总体评价**：本文提出的 **constrained KAN + differentiable solver + symbolic regression** 框架为科学机器学习中的核函数发现提供了**一种鲁棒、通用且可解释的新范式**，尤其在面对现实世界常见的稀疏噪声数据时展现出明显优势。

</details>

---

### 14. [HiFi-LLP: High-Fidelity, Low-Cost Latency Predictors with Confidence for Robust HW-NAS](https://arxiv.org/abs/2607.11746)

**Authors**: Shambhavi Balamuthu Sampath, Behzad Shomali, Nael Fasfous, Moritz Thoma, Judeson Anthony Fernando, Lukas Frickenstein, Pierpaolo Mori, Manoj Rohit Vemparala, Alexander Frickenstein, Walter Stechele  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.11746v1  

#### Abstract
With deep neural networks (DNNs) increasingly deployed on edge devices, hardware (HW)-aware optimization techniques--such as HW-aware compression and HW-aware neural architecture search (HW-NAS)--have become essential. These methods rely on real feedback from the target hardware to tailor DNN archit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HiFi-LLP: High-Fidelity, Low-Cost Latency Predictors with Confidence for Robust HW-NAS

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Hardware-aware Neural Architecture Search (HW-NAS)** 中，传统方法依赖于硬件在环（HIL）测量延迟，但由于其**顺序性和高成本**，成为搜索瓶颈。虽然已有研究引入 **Latency Predictor** 替代 HIL，但仍面临两大挑战：
1. **平台专用预测器（platform-specific predictors）通常需要数万样本训练**（如 nn-Meter 需要 26K 样本），成本高昂；
2. **预测不准确会误导 NAS 过程**，导致次优架构被选中。

此外，现有低样本量预测器（如 HiFi-SAGE）缺乏对预测**置信度（confidence）** 的建模，可靠性差，在真实场景中表现不稳定。

---

### 🚀 提出的新方法与创新思路

本文提出 **HiFi-LLP**（High-Fidelity, Low-Cost Latency Predictor），具备以下核心创新：

- **基于 GATv2 的图神经网络架构**：采用 **Graph Attention Network v2 (GATv2)** 作为主干网络，相比 GCN 或 GraphSAGE，能更有效地捕捉 DNN 架构图中的节点交互关系，缓解过平滑问题，并支持更深网络（11 层）。
  
- **集成 Gaussian Process (GP) 回归头**：为预测输出提供 **均值（预测延迟）和方差（不确定性）**，首次在平台专用、低样本量设置下实现**概率化、置信度感知的延迟预测**。

- **仅需 100 个样本即可训练**：是目前最**sample-efficient**的平台专用延迟预测器之一，显著降低数据采集成本。

- **构建混合 NAS 框架（Hybrid NAS）**：利用预测置信度动态决策——**低置信度样本交由 HIL 测量，高置信度样本使用预测值**，兼顾效率与鲁棒性。

- **引入 Bayesian Optimization (BO) 作为采样策略**：在低数据场景下，BO 能更高效地选择信息量大的架构进行测量，优于随机或启发式采样。

---

### 🔍 相比现有方法的优势

| 特性 | HiFi-LLP | 其他 SOTA 方法（如 BRP-NAS, HiFi-SAGE, HELP） |
|------|----------|---------------------------------------------|
| 训练样本需求 | **仅需 100 样本** | BRP-NAS: 900；nn-Meter: 26K；HELP: 多设备预训练 + 少量适配 |
| 是否平台专用 | 是 | HELP/MultiPredict 为 quasi-generalized，需硬件描述符 |
| 是否提供置信度 | ✅ 是（首个） | ❌ 否 |
| 是否 GNN-based | ✅ GATv2 | 多数为 GCN/GraphSAGE |
| 是否 guided sampling | ✅ 支持 BO 采样 | 多数依赖随机采样 |
| 预测稳定性（标准差） | **极低**（比 HiFi-SAGE 低达 13.6×） | 报告平均值，实际 variance 大 |

> ✅ **HiFi-LLP 是首个同时满足“平台专用 + 低样本 + 高保真 + 置信度输出”的延迟预测器**。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用开源基准 **LatBench [5]**，包含 **NASBench-201 [10]** 中 15,625 个 DNN 架构在 **6 种边缘设备**上的精确延迟测量：
  - DCPU: Intel Core i7-7820X
  - DGPU: NVIDIA GTX 1080Ti
  - EGPU: NVIDIA Jetson Nano
  - ETPU: Google EdgeTPU
  - MGPU: Qualcomm Adreno 612 GPU
  - MDSP: Qualcomm Hexagon 690 DSP

---

### ⚙️ 实验设置
- **训练样本数**：极端低数据场景（100 样本）、低数据场景（500 样本）
- **模型输入**：将 DNN 架构建模为图（Graph），节点为层，边为数据流依赖
  - 节点特征维度：[N, 59]
  - 边索引：[2, E]
  - 批处理向量：长度 N
- **输出**：延迟预测（mean）+ 不确定性（variance）
- **训练时间**：<15 分钟（GTX 1080 Ti）

---

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy Bound (e.g., 10% Acc)** | 预测误差在真实值 ±10% 内的比例 |
| **Spearman’s Rank Correlation (ρ)** | 衡量预测排序与真实排序的一致性（保序性） |
| **Entropy** | 衡量采样样本多样性（用于采样器比较） |
| **False Positives / Negatives (FP/FN)** | 在 Oracle NAS 中误判情况 |
| **Missed Test Accuracy** | 未找到最优架构的精度损失 |
| **Pareto Front 贴近度** | NAS 结果与真实 HIL Pareto 前沿的接近程度 |
| **Speedup** | 相对于全 HIL NAS 的加速比 |

---

### 🆚 基线方法对比
| 方法 | 类型 | 样本数 | 是否置信度 |
|------|------|--------|------------|
| BRP-NAS [5] | Platform-specific | 900 | ❌ |
| HiFi-SAGE [7] | Platform-specific | 100 | ❌ |
| HELP [4] | Quasi-generalized (meta-learned) | 10 (adaptation) | ❌ |
| nn-Meter [6] | Platform-specific | 26K | ❌ |

---

## 3. 主要实验结果和性能指标

### 📊 准确率提升（Accuracy Bound）

| 方法（样本数） | 平均 10% Acc |
|----------------|--------------|
| BRP-NAS (500) | 80.0% |
| HiFi-SAGE (500) | 83.2% |
| **HiFi-LLP (500)** | **91.0%** ✅ |
| HiFi-SAGE (100) | 64.6% |
| **HiFi-LLP (100)** | **66.86%** ✅ |

- 在 **100 样本下**，HiFi-LLP 在 5/6 设备上超越 HiFi-SAGE；
- 在 **500 样本下**，在所有设备上领先，最大提升达 **9.94 p.p.**；
- **标准差显著更低**：相比 HiFi-SAGE 最多降低 **13.62×**，说明结果更稳定可靠。

---

### 📈 排序保真度（Spearman’s ρ）

| 方法 | 平均 ρ (6 devices) |
|------|--------------------|
| HELP* (10 samples) | 0.968 |
| BRP-NAS* (900) | 0.970 |
| HiFi-SAGE (100) | 0.989 |
| **HiFi-LLP (100)** | **0.991** ✅ |

- 在 **仅 100 样本下**，Spearman 相关系数最高达 **0.996**（EGPU 设备）；
- 平均保序性优于 BRP-NAS 和 HELP，且无需大量训练或跨设备迁移。

---

### 🔬 消融实验（Ablation Study）

从 HiFi-SAGE 基线逐步改进，验证各设计决策影响（见 Fig. 2）：

| 模块 | 性能增益 |
|------|---------|
| Baseline (HiFi-SAGE) | 84.19% (10% Acc), ρ=0.9673 |
| + GP Head | 性能下降 → 需配合缩放 |
| + Device-specific Latency Scaling | ↑ 至 85.66%, ρ=0.9686 |
| + GATv2 替代 GraphSAGE | ↑ 至 **87.34%**, **ρ=0.9708** ✅ |

> ✅ **GATv2 + 缩放 + GP 头三者协同作用，带来显著提升**

---

### 🤖 Oracle NAS 实验（表 V）

在假设已知精度的前提下，用延迟预测筛选架构：

| 方法 | Missed Acc | FP | FN |
|------|------------|----|----|
| BRP-NAS | 0.42 p.p. | 229.93 | 284.22 |
| HiFi-LLP | **0.26 p.p.** | 255.96 | 234.76 |
| HiFi-LLP + Confidence Filtering (**Ours_certain**) | 0.26 | **225.23** ↓ | **217.98** ↓ |

- 虽 FP 初始较高，但通过 **过滤高方差预测（STD > 1.1778）** 可有效降低 FP 和 FN；
- 显示**置信度过滤机制的有效性**。

---

### ⏱️ Hybrid NAS 实验（表 VI）

结合置信度判断是否调用 HIL：

| 方法 | 约束 (ms) | 实际延迟 | 准确率 | 相对运行时 | HIL 调用次数 |
|------|----------|----------|--------|-------------|----------------|
| BRP-NAS | 2.02 | 3.18 (>约束) ❌ | 67.50% | 0.003 | 0 |
| **Ours_hybrid** | 2.02 | **1.72** ✅ | 64.94% | **0.2222** | 1439 |
| ... | ... | ... | ... | ... | ... |
| **最大加速比** | — | — | — | **8.6×** vs 全 HIL NAS | ✅ |

> ✅ **Hybrid NAS 在保证满足延迟约束的同时，实现高达 8.6× 的端到端加速**

---

### 🎯 BO-Based 采样实验（Table VII）

在从 DCPU 模型迁移到 ETPU 的适应任务中比较采样器：

| 采样器 | 300 样本 10% Acc | 400 样本 10% Acc | Entropy |
|--------|------------------|------------------|---------|
| Random | 74.25% | 79.29% | 5.594 |
| ZCP / CATE / Arch2Vec | ~78–81% | ~80–81% | ~5.87 |
| **BO (Ours)** | **79.03%** | **86.20%** ✅ | **5.885** ✅ |

- 在 **400 样本内，BO 采样器比第二名高出 4.5 p.p.**；
- **熵更高** → 采样更多样、覆盖更广的设计空间；
- t-SNE 可视化显示 BO 更早探索稀疏区域（如右尾结构）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **GATv2 + GP Head 架构可同时实现高保真与不确定性估计**，适用于低样本场景；
2. **仅需 100 个样本即可训练出高性能延迟预测器**，远低于现有方法；
3. **置信度可用于构建 Hybrid NAS 框架**，在保持 Pareto 前沿质量的同时实现 **8.6× 加速**；
4. **Bayesian Optimization 是低数据下最优的数据采样策略**，显著优于随机或其他 SOTA 采样器；
5. **HiFi-LLP 在准确性、稳定性、保序性上全面超越现有平台专用预测器**。

---

### ⚠️ 方法局限性
- 当前仅针对 **NASBench-201** 架构空间，尚未扩展至更大空间（如 NATS-Bench 或 ImageNet 级模型）；
- **GP Head 增加推理延迟**（HiFi-LLP 单次推理 0.21s vs BRP-NAS 0.001s），虽不影响整体 NAS 效率，但在极端高频调用场景可能受限；
- 置信度阈值（如 STD > 1.1778）为经验设定，缺乏理论最优解法。

---

### 🔮 未来工作方向
1. **将 BO 采样器与 Hybrid NAS 框架深度融合**，形成闭环主动学习系统；
2. 扩展至 **更大的搜索空间和实际部署场景**（如自动驾驶、机器人）；
3. 探索 **multi-objective confidence modeling**（如同时建模延迟、功耗、内存的不确定性）；
4. 研究 **轻量化 GP Head** 或替代方案，在保持置信度输出的同时减少开销。

---

## 总结

> **HiFi-LLP 是一个兼具“高保真”、“低成本”、“可信赖”的新型延迟预测器，为 HW-NAS 提供了一种稳健、高效的解决方案。它不仅在性能上超越现有方法，更重要的是引入了“置信度”这一关键维度，使 NAS 过程更具鲁棒性和可解释性。**

</details>

---

### 15. [Metadata-Free Meta-Reweighted Direct Preference Optimization under Noisy Preference Labels](https://arxiv.org/abs/2607.09796)

**Authors**: Hua Qu, Yifan Li, Xiaodong Yuan  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.09796v1  

#### Abstract
Direct Preference Optimization (DPO) has become an important method for aligning large language models (LLMs) with human preferences because it removes the need for explicit reward modeling and reinforcement learning optimization. However, its performance depends heavily on the quality of preference...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Metadata-Free Meta-Reweighted Direct Preference Optimization under Noisy Preference Labels

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Direct Preference Optimization (DPO)** 在实际应用中面临的关键挑战——**偏好标签噪声（noisy preference labels）**。现实世界中的偏好数据常因标注错误、主观偏差等原因存在噪声，这会显著降低 DPO 的对齐效果。传统方法通常依赖于高质量的元数据（如干净的验证集）来提升鲁棒性，但在大规模语言模型（LLM）对齐场景下，获取此类元数据成本高昂甚至不可行。

### 提出的新方法与新思路
作者提出了 **PACMR-DPO (Prompt-Augmentation-Consistency Meta-Reweighted DPO)**，一种无需元数据的元加权 DPO 框架，其核心创新点如下：

- **理论证明**：在理想条件下，通过双层优化（bilevel optimization）框架，可以利用可学习的样本权重函数从含噪训练数据中恢复出在干净数据上的 DPO 最优解。这为自适应重加权提供了坚实的理论基础。
  
- **无元数据的元学习机制**：提出了一种**任务无关的元知识驱动方法（task-agnostic meta-knowledge-driven method）**。它摒弃了对干净元数据的依赖，转而引入**提示增强一致性（prompt augmentation consistency）**作为外层优化目标。具体来说，通过对输入提示（prompt）进行语义保持的变换（如回译），并要求模型对原始和增强样本的偏好判断保持一致，从而生成可靠的元信号来指导权重网络（VNet）的学习。

- **高效的可扩展训练方案**：为了克服 LLM 元学习中高阶梯度计算带来的巨大内存和计算开销，结合了**中心差分近似（central-difference approximation）** 和 **LoRA 微调（LoRA fine-tuning）**。该方案避免了存储完整的高阶计算图或构建每样本梯度矩阵，大幅降低了资源消耗。

### 相比现有方法的优势
- **不依赖元数据**：与需要少量干净元数据的元学习方法（如 MWN-DPO）不同，PACMR-DPO 完全无需任何额外的人工标注或清洗数据，更具实用性和普适性。
- **更强的噪声鲁棒性**：相比基于损失校正、标签平滑等固定规则的方法，PACMR-DPO 能够自适应地为每个样本学习动态权重，更灵活有效地抑制噪声样本的影响。
- **计算效率高**：提出的近似方案使其能够在大模型上实现高效训练，解决了传统元学习在 LLM 上难以扩展的问题。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在两个标准的 LLM 对齐任务上进行：
- **TL;DR Summarization**：Reddit 帖子摘要任务。
- **Anthropic HH Single-Turn Dialogue**：单轮对话任务，旨在训练既“有帮助”又“无害”的助手。

### 实验设置和评估指标
- **噪声注入**：采用**随机候选交换（random candidate swap）** 的方式模拟噪声，即以预设概率 `e` 随机交换一对候选回复的顺序。实验设置了三种噪声率：20%、30%、40%。
- **基模型**：统一使用 `Llama-2-7B` 作为基础模型，并采用 LoRA 进行微调（rank=16, alpha=32）。
- **评估协议**：
  - 在测试集上随机采样 800 个提示（prompts）。
  - 使用 **Judge Model (GPT-5.1)** 对 PACMR-DPO 及各基线方法生成的回复与标准 DPO 生成的回复进行成对比较。
  - 采用 **AB/BA 顺序交换** 以减少位置偏见。
- **评估指标**：
  - **Win rate**：PACMR-DPO 优于基线的比例。
  - **Win-score**：综合考虑胜、负、平局的得分，计算公式为 `1 + (#win - #lose) / Total comparisons`。得分越高表示整体质量越优。

### 基线方法对比
选取了多种 DPO 变体作为基线：
- **DPO**：标准直接偏好优化。
- **cDPO**：通过标签平滑处理噪声。
- **IPO**：一种 DPO 的变体。
- **rDPO**：通过去偏加权损失提高鲁棒性。
- **Dr.DPO**：结合分布鲁棒优化（distributionally robust optimization）。
- **MWN-DPO-clean**：使用干净元数据的元加权方法，用于对比 PACMR-DPO 是否能替代元数据。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### TL;DR 摘要任务结果
| Method     | Win-score @20% | Win-score @30% | Win-score @40% |
|------------|---------------|---------------|---------------|
| **PACMR-DPO** | **1.4763**      | **1.5263**      | **1.4950**      |

#### Anthropic HH 对话任务结果
| Method     | Win-score @20% | Win-score @30% | Win-score @40% |
|------------|---------------|---------------|---------------|
| **PACMR-DPO** | **1.4488**      | **1.6075**      | **1.2400**      |

### 与基线方法的对比结果
- **全面领先**：在 TL;DR 任务的所有噪声水平下，PACMR-DPO 均取得了最高的 Win-score。
- **显著优势**：在中高噪声环境下（尤其是 30%-40%），PACMR-DPO 的优势更加明显。例如，在 HH 任务的 30% 噪声下，其 Win-score (1.6075) 显著高于次优的 Dr.DPO (1.4400)。
- **媲美甚至超越有元数据方法**：与使用干净元数据的 **MWN-DPO-clean** 相比，PACMR-DPO 在多数设置下表现相当甚至更优（如 TL;DR @20% 和 HH @20%/30%），证明了其提出的“提示增强一致性”是一个有效的元信号替代品。

### 消融实验结果
- **学习权重 vs 固定先验权重**：将可学习的 VNet 替换为固定的 `sigmoid(u(z))` 权重函数后，性能显著下降，尤其是在高噪声（40%）情况下。这表明**自适应学习权重是必要的**，简单的单调先验不足以捕捉复杂的样本可靠性。
- **置信度阈值敏感性**：外层伪标签的置信度阈值 `T` 对性能有影响。实验发现 `T=0.6` 是一个较好的折中选择，过低会引入不可靠的伪标签，过高则会减少有效信号的数量。

---

## 4. 关键结论和发现

### 主要发现
1. **理论可行性**：双层重加权框架在理论上能够纠正由噪声引起的最优解偏移。
2. **实践有效性**：PACMR-DPO 能够成功学习到区分干净样本和噪声样本的权重（见 Table 3，未翻转和翻转样本的平均权重差距显著），从而在含噪数据上实现更优的对齐。
3. **元信号可替代**：**提示增强一致性**作为一种任务无关的元知识，可以有效替代昂贵的干净元数据，为元学习提供监督信号。
4. **高效可行**：结合中心差分和 LoRA 的训练方案使得该方法在计算上是可扩展的，适用于大模型。

### 方法的局限性
- **极高噪声下的挑战**：当噪声率达到 40% 时，干净样本和噪声样本在特征空间上的分布开始严重重叠，导致权重学习变得困难，性能提升幅度减小。
- **依赖特定增强策略**：当前方法依赖于回译等文本增强技术，其效果可能受语言和领域限制。
- **单轮训练**：所有模型都只训练了一个 epoch，更长的训练周期可能会带来不同的结果。

### 未来工作方向
- 探索更多样化、更强大的语义保持增强策略。
- 将该框架应用于其他类型的噪声（如非对称噪声、实例相关噪声）。
- 研究如何在多轮迭代训练中动态调整元学习过程。
- 将该方法扩展到更复杂的多轮对话或长文本生成任务中。

</details>

---

### 16. [Nonparametric Bayesian Inverse Reinforcement Learning with Data-Parallel Gibbs Sampling](https://arxiv.org/abs/2607.09886)

**Authors**: Sai Anirudh Katupilla, Shreeya Dasa Lakshminath  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.09886v1  

#### Abstract
Inverse Reinforcement Learning recovers reward functions from expert demonstrations, but standard formulations assume that all demonstrations come from a single expert. When demonstrations are pooled from multiple experts with distinct preferences, parametric methods recover an averaged reward that ...

---

### 17. [NeuroMem-FHP: A Likelihood-Free Deep Learning Framework for Parameter Estimation of Fractional Hawkes Process](https://arxiv.org/abs/2607.11177)

**Authors**: Neha Gupta, Aditya Maheshwari  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.11177v1  

#### Abstract
In this paper, we propose deep learning based NeuroMem-FHP framework for estimating the parameters of the fractional Hawkes process (FHP), a self-exciting point process that captures long-range dependence through a fractional Mittag-Leffler excitation kernel. Two neural architectures, namely a Long ...

---

### 18. [YUKTI: From Natural-Language Situations to Robust, Verifiable Decisions An Uncertainty-Typed Proposition IR, Assumption-Robust Pareto Frontiers, and a Regret Certificate](https://arxiv.org/abs/2607.09706)

**Authors**: Suyash Mishra  
**Category**: cs.AI  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.09706v1  

#### Abstract
Language models turn a worded situation into a numeric plan, and the dominant pipelines (NL4Opt, OptiMUS, ORLM, OR-LLM-Agent) commit to a single objective and point-valued coefficients, then solve once. For decisions that allocate real budget, effort, or clinical attention, that confidence is the fa...

---

### 19. [SETA: Scaling Environments for Terminal Agents](https://arxiv.org/abs/2607.10891)

**Authors**: Qijia Shen, Zhiqi Huang, Vamsidhar Kamanuru, Aznaur Aliev, Jay Rainton, Ahmed Awelkair, Zhichen Zeng, Jiajun Li, Shi Dong, Yueming Yuan, Boyuan Ma, Qizheng Zhang, Jiwei Fu, Yuzhen Mao, Wendong Fan, Ping Nie, Philip Torr, Bernard Ghanem, Changran Hu, Jonathan Lingjie Li, Urmish Thakker, Guohao Li  
**Category**: cs.AI  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.10891v1  

#### Abstract
Large language models (LLMs) are rapidly shifting toward agents that solve tasks through diverse interfaces, including web and graphical user interfaces (GUIs). Among these, the terminal command line provides a text-based, general-purpose interface, covering tasks from system operations to data scie...

---

### 20. [The Ebb and Flow of Multimodal Focus: Scheduling Visual Relay Windows for Grounded VLM Reasoning](https://arxiv.org/abs/2607.11436)

**Authors**: Wencheng Ye, Yi Bin, Yujuan Ding, Hongye Fang, Zheng Wang, Xing Xu, Jingkuan Song, Yun Zhang, Sirui Da, Heng Tao Shen  
**Category**: cs.AI  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.11436v1  

#### Abstract
Vision-language models increasingly succeed on multimodal reasoning benchmarks, yet their visual evidence often becomes unstable once it enters the language stack, weakening evidence-grounded reasoning. To understand this fragility, we examine the internal dynamics of VLMs through a mechanistic lens...

---

### 21. [Amplitude-Only FFN Intervention for Tool-Structured LLM Inference Method: Gated Evaluation Protocol, and Cross-Model Empirical Results](https://arxiv.org/abs/2607.11183)

**Authors**: Sheng Xu, Boyuan Huang, Ke Jia, Jiadun Zhu, Zhen Chen  
**Category**: cs.CL  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.11183v1  

#### Abstract
Large language models increasingly operate as tool-using agents, where small format, argument, or function-call errors can invalidate otherwise plausible responses. We study inference-time feed-forward network (FFN) intervention for improving structured outputs without retraining model weights. Our ...

---

### 22. [Q-BridgeNet: A Quantization Network for Cross-Lingual Sign Language Translation](https://arxiv.org/abs/2607.11215)

**Authors**: Liqian Feng, Lintao Wang, Xiaochen Liu, Anusha Withana, Ken-Tye Yong, Dehui Kong, Zhiyong Wang, Kun Hu  
**Category**: cs.CL  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.11215v1  

#### Abstract
Most sign language translation (SLT) methods focus on isolated native sign-spoken pairs (e.g., American Sign Language - English). Extending language-specific SLT models to multilingual translation would improve accessibility by enabling communication across diverse sign and spoken language communiti...

---

### 23. [Direct Image-to-Modern Vietnamese Translation of Han-Nom Manuscripts via Multimodal RLHF Preference Alignment](https://arxiv.org/abs/2607.11434)

**Authors**: Thi Kim Trang Vo, Nghia Hieu Nguyen, Ha Minh Tan  
**Category**: cs.CL  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.11434v1  

#### Abstract
Translating Han-Nom manuscripts into modern Vietnamese is challenging because historical pages are often degraded, the script contains rare logographic characters, and parallel supervision is limited. We propose a multimodal RLHF preference-alignment framework that conditions Vietnamese generation o...

---

### 24. [UMoE:Unlocking Every Expert in Domain-Specific Training](https://arxiv.org/abs/2607.11444)

**Authors**: Xuefeng Li, Pengfei Liu  
**Category**: cs.CL  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.11444v1  

#### Abstract
Mixture-of-Experts (MoE) models scale capacity without proportional compute cost and have become a key architecture for frontier large language models (LLMs). Yet domain-specific post-training inherits an expert pool shaped by mixed-domain pre-training: a substantial subset of experts contributes li...

---

### 25. [Descriptive Execution of HPC Applications and Workflows](https://arxiv.org/abs/2607.10081)

**Authors**: Vanessa Sochat, Daniel Milroy  
**Category**: cs.DC  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.10081v1  

#### Abstract
The means to execute and orchestrate software components has changed from human-written code to descriptive prose. In high performance computing, this transition is represented in application orchestration, workload management, and system monitoring and debugging, to name a few. The underlying means...

---

### 26. [Prioritizing Search Space Regions in the Low Autocorrelation Binary Sequences Problem](https://arxiv.org/abs/2607.09688)

**Authors**: Bla\v{z} P\v{s}eni\v{c}nik, Borko Bo\v{s}kovi\'c, Jan Popi\'c, Janez Brest  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.09688v1  

#### Abstract
Low autocorrelation binary sequences problem (LABS) is a hard combinatorial optimization challenge with important applications in communications, signal processing, and satellite navigation. This paper proposes a hybrid search framework that combines Thompson sampling with parallel self-avoiding wal...

---

### 27. [DSSMs: State Space Models with Explicit Memory via Delay Differential Equations](https://arxiv.org/abs/2607.10244)

**Authors**: Yixiao Qian, Song Chen, Jiaxu Liu, Shengze Cai, Chao Xu  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.10244v1  

#### Abstract
State Space Models (SSMs) have emerged as a powerful paradigm for efficient long-sequence modeling, offering parallel training and fast linear-time recurrent inference. However, like other recurrent architectures, SSMs must compress an unbounded history into a fixed-size state, which limits context ...

---

### 28. [Data-Driven Telecom Marketing Optimization: A Machine Learning-Based Churn Prediction and Customer Segmentation Framework](https://arxiv.org/abs/2607.10260)

**Authors**: Nada Ali, Lina Ahmed, Tahani Abdalla Attia Gasmalla  
**Category**: cs.LG  
**Published**: 2026-07-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.10260v1  

#### Abstract
Customer churn is a major challenge for telecommunication companies, directly eroding revenue and long term customer relationships. Traditional retention programs rely on generic, not personalized incentives and lack the precision to identify high risk customers before they leave. This paper present...

---

### 29. [Are LLMs Ready for Scientific Discovery? A Capability-Oriented Benchmark for AI Scientists](https://arxiv.org/abs/2607.11079)

**Authors**: Chuhan Shi, Xiaoquan Ren, Sicheng Song, Haobo Li, Rui Sheng, Yushi Sun  
**Category**: cs.AI  
**Published**: 2026-07-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.11079v1  

#### Abstract
Existing benchmarks for scientific data analysis evaluate LLMs primarily on code execution or workflow completion, overlooking that scientific analysis serves to support distinct types of scientific claims: hypothesis exploration, statistical inference, mechanistic explanation, each with different a...

---

### 30. [Calibrated e-CUSUM Decoding for Quantized Reasoning Models: Why Token Log-Probability Is the Wrong Observable for Decoding Monitors](https://arxiv.org/abs/2607.11317)

**Authors**: El Hassane Ettifouri (Novelis Research, Paris, France), Ayoub Belfatmi (Novelis Research, Paris, France), Mahaman Sanoussi Yahaya Alassan (Novelis Research, Paris, France), Walid Dahhane (Novelis Research, Paris, France)  
**Category**: cs.AI  
**Published**: 2026-07-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.11317v1  

#### Abstract
Low-bit quantization makes small reasoning models inexpensive to deploy but can degrade their chains of thought. This motivates decoder-side monitors that intervene when generation becomes unreliable. We show that a natural candidate, the centered token log-probability increment $\log p(w_t)+H_t$, i...

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

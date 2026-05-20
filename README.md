# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-20 08:46:52 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Understanding Inference Scaling for LLMs: Bottlenecks, Trade-offs, and Performance Principles](https://arxiv.org/abs/2605.19775)

**Authors**: Moiz Arif, Avinash Maurya, Sudharshan Vazhkudai, Bogdan Nicolae  
**Category**: cs.DC  
**Published**: 2026-05-20  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2605.19775v1  

#### Abstract
The transition from standard generative AI to \emph{reasoning-centric architectures}, exemplified by models capable of extensive Chain-of-Thought~(CoT) processing, marks a fundamental paradigm shift in system requirements. Unlike traditional workloads dominated by compute-bound prefill, reasoning wo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Understanding Inference Scaling for LLMs: Bottlenecks, Trade-offs, and Performance Principles**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本文系统性地研究了**推理密集型大语言模型（reasoning-centric LLMs）在大规模 GPU 集群上的推理扩展瓶颈**，特别是当模型采用 Chain-of-Thought（CoT）等长链推理机制时所面临的性能挑战。传统推理优化方法（如最大化 batch size 或依赖 Data Parallelism）在这些场景下失效，导致严重的内存碎片、调度抢占（preemption）和非线性延迟激增。

核心问题是：  
> **如何为长序列、高容量需求的推理任务设计有效的并行化策略？**

### **提出了什么新方法或新思路**
论文并未提出单一的新算法，而是通过实证分析揭示了一系列**新的性能原则和系统级洞察**，构建了一个**面向推理扩展的决策框架**：

- 提出“**Parallelism Transition Point**”概念：确定从 Data Parallelism（DP）转向 Tensor Parallelism（TP）或 Hybrid Parallelism 的临界点。
- 定义“**Reasoning Gap**”：量化推理任务中从 prefill 阶段的计算瓶颈向 decode 阶段的内存带宽/容量瓶颈的转变。
- 揭示“**Capacity Trap**”现象：即高并发请求因 KV-cache 耗尽 HBM 导致频繁 preemption 和重计算，反而降低吞吐。
- 提出“**Right-Sized TP**”理念：对中等规模模型（如 32B），最小化 TP degree 可平衡通信开销与内存释放收益。

### **相比现有方法的优势**
- **超越微观优化视角**：不同于聚焦 kernel 优化或 KV 缓存管理的工作，本文从**系统架构层面**识别根本瓶颈。
- **提供可操作的设计指南**：基于实测数据给出不同模型规模下的最优并行配置建议。
- **区分 dense 与 sparse 架构差异**：指出 MoE 模型（如 DeepSeek-R1）与 dense 模型（如 Llama-405B）应采用不同的并行策略。
- **强调调度与容量感知**：主张将 KV-cache 容量作为首要调度约束，而非仅看当前利用率。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Meta's Natural Reasoning dataset**：包含 115 万个多跳推理与常识推断样本。
  - 特征：输入序列短（77% 在 50–150 tokens），输出极长（45% > 5000 tokens），体现典型的“推理爆炸”特性。

### **实验设置**
- **硬件平台**：
  - 单节点 8× NVIDIA H200 GPU（SXM5）
  - 每 GPU 141GB HBM3e，峰值带宽 4.8 TB/s
  - NVLink 4.0，GPU 间双向带宽 900 GB/s
- **软件栈**：
  - 推理引擎：vLLM v1，启用 PagedAttention
  - Block size：16 tokens
  - 调度策略：FCFS，调参 `max_num_batched_tokens` 和 `max_num_seqs`

### **评估指标**
| 指标 | 含义 |
|------|------|
| **TTFT (Time-To-First-Token)** | 请求到首 token 的延迟，反映 prefill 性能 |
| **TPOT (Time-Per-Output-Token)** | 解码阶段平均 token 生成时间，反映 decode 内存压力 |
| **Generation Throughput** | 所有 GPU 每秒生成 token 数 |
| **E2E Latency** | 完整请求处理时间（含排队、prefill、decode） |
| **KV-Cache Utilization** | 实际使用的 KV cache 占总分配量比例 |
| **HBM Bandwidth Utilization** | 通过 `nvidia-smi` 监控的实际带宽使用率 |

### **基线方法对比**
- **并行策略对比组**：
  - **Data Parallelism (DP)**：复制模型，独立处理请求
  - **Tensor Parallelism (TP)**：层内张量切分，聚合显存
  - **Pipeline Parallelism (PP)**：层间流水线划分
  - **Hybrid (e.g., DP+TP, PP+TP)**：组合策略
- **模型对比**：
  - 小模型：Llama-8B, Qwen-14B, Qwen-32B
  - 中大型模型：Llama-70B
  - 前沿模型：Llama-3.1-405B（dense）、DeepSeek-R1-671B（sparse MoE）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **小模型（8B–32B）上的 Capacity Trap**
- 在单 H200 上运行 DeepSeek-8B：
  - 当 `max_num_seqs = 10K` 时，初始 throughput 达 **235K tokens/s**，但迅速因 KV 耗尽触发 preemption。
  - 最终稳定 throughput 下降至约 **100K tokens/s**，出现严重抖动。
  - TPOT 从 1K 并发时的 ~0.08s 上升至 10K 时的 ~0.48s。
  - 存在一个 **E2E 延迟凸点**，最优并发数约为 **2K sequences**。

#### **DP vs. TP 扩展效率（32B 模型）**
| 配置 | E2E 时间 (s) | 加速比 |
|------|---------------|--------|
| DP=8 | 857s | 1.0× |
| TP=8 | 686s | 1.25× |
| **DP=4+TP=2** | **484s** | **1.77×** ✅ |

👉 表明混合策略显著优于纯 DP 或纯 TP。

#### **前沿模型（405B & 671B）表现**
| 模型 | 最优策略 | E2E 时间 (s) | 关键瓶颈 |
|------|----------|--------------|---------|
| **Llama-405B (dense)** | TP=8 | 986s | HBM bandwidth, KV 容量 |
| **DeepSeek-R1-671B (MoE)** | PP=4+TP=2 | 1663s | 路由同步延迟 |
| 对比：R1 若用 TP=8 | — | 2047s | ❌ 同步开销过大 |

- **KV Cache 消耗对比**：
  - Llama-70B：~328 KB/token
  - Llama-405B：~1.05 MB/token
  - DeepSeek-R1-671B（使用 MLA）：远低于同参数量 dense 模型（得益于 Multi-Head Latent Attention 压缩）

#### **Throughput 扩展趋势**
- 从 8B 到 70B（9× 参数增长）→ throughput 下降仅 5–6×，说明 TP 有效缓解了带宽压力。
- 671B 模型 throughput 曲线呈“长尾”，反映其持续生成推理 token 的特性。

---

## **4. 关键结论和发现**

### **主要发现**

| 发现编号 | 内容摘要 |
|--------|--------|
| **Observation 1** | 高并发会触发 **Capacity Trap**：KV-cache 耗尽导致 preemption 和重计算，反而降低吞吐。应实施 KV-aware 并发控制。 |
| **Observation 2** | 存在 **TTFT 与 TPOT 的权衡**：更大 batch 减少排队延迟（改善 TTFT），但加剧内存竞争（恶化 TPOT）。存在一个最优 batch size。 |
| **Observation 3** | **DP 不解决 per-GPU 容量限制**：每个副本仍需完整存储权重 + KV cache，无法池化内存。易造成“stranded capacity”。 |
| **Observation 5** | 并行策略选择取决于模型大小：
  - <32B：DP 更优（通信代价高于收益）
  - ≥32B：TP 或 Hybrid 更优（释放的 KV 空间超过通信成本） |
| **Observation 6** | **dense 与 sparse 模型偏好不同**：
  - Dense（如 Llama-405B）：受益于高 degree TP（聚合带宽与容量）
  - Sparse MoE（如 DeepSeek-R1）：更敏感于同步延迟，适合低 degree TP + 高 PP 的 hybrid 策略 |
| **Observation 7** | 推理负载大部分时间处于 **decode-dominated regime**，性能由 KV 移动和 HBM 带宽决定，而非 FLOPs。 |
| **Observation 8** | “**Reasoning Cliff**”出现在 KV-cache 超过 HBM 容量时，系统被迫 preemption 或拒绝请求。应提前预留 decode 阶段所需空间。 |
| **Observation 9** | 调度器必须进行 **KV-aware admission control**，将调度视为“memory traffic shaping”问题。 |

### **方法的局限性**
- 实验集中在单个 NVLink 连接的 8-GPU 节点内，未深入研究跨节点扩展（如 PP over RDMA）的影响。
- 假设所有 KV cache 驻留 HBM，未评估 KV offloading、compression 等技术的实际效果（虽提及为正交方向）。
- 使用合成 workload（固定 batch size），未模拟真实动态流量模式（bursty arrivals, variable lengths）。
- 分析基于 vLLM，其他推理引擎（如 TensorRT-LLM）可能有不同的行为特征。

### **未来工作方向**
1. **硬件-软件协同设计**：
   - 开发支持 **disaggregated memory** 的架构（如结合 CXL、NVMe tiering）以突破 HBM 容量墙。
   - 设计专用 decode 加速器，专注于高带宽 KV 访问。
2. **新型调度策略**：
   - 实现在线 batch size tuning，基于实时 TTFT、TPOT、KV occupancy 动态调整。
   - 引入优先级调度，保护低延迟请求免受长推理任务干扰。
3. **架构创新**：
   - 推广 MLA、GQA 等压缩注意力机制，减少 KV footprint。
   - 探索 **prefill-decode disaggregation**：用不同硬件分别处理两个阶段。
4. **支持 Agentic AI**：
   - 构建跨 GPU-CPU 的统一上下文管理系统，应对多步骤、状态持久的 agent 工作流。
   - 实现 KV cache 的迁移、共享与版本控制。

--- 

> **总结一句话**：  
> 本文揭示了推理密集型 LLM 正面临从“compute-bound”到“capacity-bound”的范式转移，传统的 scaling heuristics 已失效；未来的高性能推理系统必须以 **KV-cache 容量为核心约束**，结合模型架构特性（dense vs. MoE）选择合适的并行策略，并推动软硬协同的 disaggregated 架构演进。

</details>

---

### 2. [TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload](https://arxiv.org/abs/2605.20179)

**Authors**: Zhiben Chen, Youpeng Zhao, Yang Sui, Jun Wang, Yuzhang Shang  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2605.20179v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have emerged as a competitive alternative to autoregressive (AR) models, offering better hardware utilization and bidirectional context through parallel block-level decoding. However, as dLLMs continue to scale up with mixture-of-experts (MoE) architectures, t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着 **Diffusion-based Large Language Models (dLLMs)** 规模扩大并采用 **Mixture-of-Experts (MoE)** 架构，其在资源受限设备（如单 GPU-CPU 系统）上的高效推理面临严峻挑战。主要瓶颈包括：
- **高 I/O 开销**：频繁在 GPU 和 CPU 之间迁移专家权重导致大量 CPU-GPU 数据传输。
- **计算效率低下**：若将未命中专家的任务交由 CPU 处理，则系统容易陷入 **CPU-bound**，GPU 利用率下降。

现有方法（如 Mixtral-Offload、Fiddler）针对 AR 模型设计，无法有效应对 dLLM 中每个去噪步激活整块 token 所带来的 **更广、更碎片化的专家访问模式**。

---

### 🚀 提出的新方法：TIDE
TIDE 是一种 **训练无关、无损优化** 的 MoE-dLLM 推理系统，核心思想是利用 **专家激活的时序稳定性（temporal stability）** 来减少不必要的 I/O 和计算开销。

#### 主要创新点：
1. **发现并利用跨步专家路由相似性**
   - 实验证明相邻去噪步之间的 expert routing 具有高度一致性（cosine similarity > 0.95 即使相隔 5 步），支持“局部准静态”假设。
   
2. **提出区间式专家刷新策略（interval-based expert refresh）**
   - 将解码过程划分为 **refresh steps** 和 **skipped steps**：
     - 在 refresh step 更新 GPU 驻留专家集合（基于 token hit count 最高的专家）；
     - 在 skipped steps 复用当前专家布局，不进行迁移。
   - 显著降低专家迁移频率，从而减少 I/O 开销。

3. **通过数学规划建模最优刷新间隔 $T$**
   - 将推理调度形式化为一个 **约束数学规划问题（constrained MP problem）**：
     $$
     \min_T \left[ C_{IO} \cdot B \cdot T \cdot (1 - (1-d)^T) + C_{CPU} \cdot T \cdot B \cdot f(T) \right]
     $$
   - 综合考虑 **I/O 传输延迟** 与 **CPU 计算延迟**，结合硬件 profiling 和贪心搜索求解最优 $T$。

4. **异步执行流水线设计**
   - 当 token 被路由到 CPU 上的专家时，GPU 不阻塞，而是异步处理命中项，实现计算重叠。

5. **完全无损（lossless）推理**
   - 不修改模型结构、router 或权重，仅调整专家放置策略，保证输出一致性和精度零损失。

---

### 🔍 相比现有方法的优势
| 特性 | TIDE | Mixtral-Offload | Fiddler |
|------|------|------------------|---------|
| 是否训练自由 | ✅ 是 | ✅ 是 | ✅ 是 |
| 是否无损 | ✅ 是 | ✅ 是 | ✅ 是 |
| I/O 开销控制 | ⭐️ 强（按需更新） | ❌ 高（每步都换） | ✅ 低（固定放置） |
| 计算效率 | ⭐️ 高（GPU 主导） | ⚠️ 可能 I/O 瓶颈 | ❌ 低（易 CPU-bound） |
| 动态适应能力 | ✅ 强（基于 hit count） | ✅ 强 | ❌ 弱 |

> ✅ TIDE 成功平衡了 **I/O 开销** 与 **计算效率**，特别适合资源受限场景下的 MoE-dLLM 部署。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **MBPP (Mostly Basic Programming Problems)** 的清洗版本（sanitized MBPP）
- 通过 `1m_eval_harness` 库加载，用于生成编程类文本任务。

---

### ⚙️ 实验设置
| 项目 | 设置详情 |
|------|----------|
| **模型架构** | LLaDA2.0 系列：<br>• LLaDA2.0-mini (16B A1B)<br>• LLaDA2.0-flash (100B A6B)<br>• 每层 256 FFN experts, top-k=8 |
| **硬件平台** | • GPU: NVIDIA A100 40GB / H100 80GB<br>• CPU: 48-core Intel, 1024GB DDR4 |
| **实现框架** | 基于 HuggingFace Transformers + dInfer，PyTorch 2.9 + CUDA 12.8 |
| **块大小（block size）** | 默认 32，部分实验测试 64 |
| **生成长度（gen length）** | 256 / 1024 tokens |
| **GPU 专家预算（B）** | 32 ~ 128 个专家可驻留 GPU |

---

### 🎯 评估指标
- **Throughput (TPS)**：每秒解码 token 数量（越高越好）
- **End-to-end decode latency**
- **GPU expert hit rate**
- **CPU-GPU I/O traffic**

---

### 🆚 基线方法对比
由于目前尚无专门针对 MoE-dLLM 的推理优化工作，作者选用两个主流 AR-MoE 推理方案作为基线：
1. **Mixtral-Offload** [Eliseev and Mazur, 2023]  
   → 每一步都重新进行专家迁移（full refresh），I/O 密集。
2. **Fiddler** [Kamahori et al., 2024]  
   → 固定专家布局，所有未命中 token 交由 CPU 处理，易造成 CPU 瓶颈。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 模型 | GPU Expert Budget | GPU Memory | TIDE (token/s) | Mixtral-Offload | Speedup |
|-------|--------------------|-------------|----------------|------------------|--------|
| LLaDA2.0-mini | 128 | 18 GB | **2.44** | 1.91 | **1.28×** |
| LLaDA2.0-mini | 64 | 10 GB | **2.11** | 1.69 | **1.25×** |
| LLaDA2.0-flash | 64 | 55 GB | **1.45** | 1.16 | **1.25×** |
| LLaDA2.0-flash | 32 | 30 GB | **1.24** | 1.01 | **1.23×** |

> 💡 **最高达 1.4× (mini) 和 1.5× (flash) 的吞吐提升**，尤其在内存受限条件下优势显著。

---

### 🔬 消融实验与敏感性分析（Ablation Study）

#### ✅ 刷新间隔 $T$ 的影响（Table 2）
| 配置 | T=1 (Mixtral-Offload) | Random T | **Optimal T (TIDE)** |
|------|------------------------|-----------|------------------------|
| LLaDA2.0-mini (B=128) | 1.91 | 2.14 | **2.44** |
| LLaDA2.0-flash (B=64) | 1.16 | 1.23 | **1.45** |

- 使用随机 $T$ 性能波动大；
- TIDE 通过优化模型选择最佳 $T$，带来 **高达 1.4× 超越随机配置的速度提升**。

#### ✅ 敏感性研究（Figure 5）
- **不同 block size (32~128)**：TIDE 始终优于基线，且随 block 增大优势更明显。
- **不同 GPU expert budget (32~128)**：TIDE 可充分利用更多 GPU 内存，而 Fiddler 几乎无扩展性。
- **不同 confidence threshold (0.7~0.95)**：TIDE 在各种置信度下均保持稳定加速（平均 **1.4×**）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **专家路由具有强时序局部性（temporal locality）**
   - 相邻去噪步间 expert activation 的 cosine similarity 平均达 **0.985**，即使间隔 5 步仍高于 **0.95**。
   - 支持“短时间窗口内专家需求近似不变”的假设。

2. **TIDE 实现了高效的负载均衡**
   - 通过动态更新高热度专家至 GPU，最大化 hit rate；
   - 区间刷新机制大幅降低 I/O 频率，避免带宽饱和。

3. **无需训练即可获得“免费午餐”式加速**
   - 完全兼容原模型，部署简单，适用于各类下游应用。

4. **在资源受限环境下表现尤为突出**
   - 尤其当 GPU memory budget 较小时，TIDE 的智能调度策略优势最大。

---

### ⚠️ 局限性（来自 Appendix）
1. **仅探索块内相似性**
   - 未分析跨 block 的专家激活模式，可能遗漏进一步优化空间。

2. **硬件平台有限**
   - 实验仅在 NVIDIA GPU + x86 CPU 上完成，缺乏对 AMD GPU 或 ARM CPU 的支持验证。

3. **局限于单机环境**
   - 当前设计面向 single GPU-CPU 系统，尚未扩展至 multi-GPU 或分布式 expert parallelism 场景。

---

### 🔮 未来工作方向
- 扩展至 **multi-node 分布式推理**，结合 expert parallelism 进行全局调度。
- 探索 **跨 block 的专家复用机制**，进一步提升长期稳定性利用。
- 支持更多硬件后端（如 AMD Instinct、Apple Silicon）以增强通用性。
- 结合量化或稀疏化技术，打造端到端轻量化 MoE-dLLM 推理栈。

---

> ✅ **总结一句话**：  
> TIDE 通过洞察 dLLM 中 MoE 专家激活的时序稳定性，提出了一种训练无关、无损且高效的 I/O-aware 推理调度机制，在真实硬件上实现了高达 **1.5× 吞吐提升**，为资源受限设备部署大规模扩散语言模型提供了实用解决方案。

</details>

---

### 3. [FlexDraft: Flexible Speculative Decoding via Attention Tuning and Bonus-Guided Calibration](https://arxiv.org/abs/2605.20022)

**Authors**: Yaojie Zhang, Jianuo Huang, Junlong Ke, Yuhang Han, Yongji Long, Tianchen Zhao, Biqing Qi, Linfeng Zhang  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.20022v1  

#### Abstract
Speculative decoding accelerates memory-bound LLM inference without quality degradation by using a fast drafter to propose multiple candidate tokens and the target model to verify them in parallel. However, conventional sequential speculative decoding suffers from mutual waiting between drafting and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FlexDraft: Flexible Speculative Decoding via Attention Tuning and Bonus-Guided Calibration**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统 **Speculative Decoding** 存在两大瓶颈：
- **串行执行开销**：标准的“先 draft 后 verify”模式导致 drafter 和 target model 之间存在 **mutual waiting**（相互等待），并频繁交换中间状态，增加内存访问开销。
- **并行解码的不确定性**：虽然并行 speculative decoding 可以重叠 drafting 与 verification，但面临两个根本挑战：
  1. **Bonus token uncertainty**：未来 draft 生成时无法获知当前验证中可能产生的 bonus token，导致 draft 分布与 target 验证路径不一致。
  2. **Acceptance length uncertainty**：drafter 必须为所有可能被接受的前缀准备候选分支，造成冗余计算，尤其在大 batch size 下目标模型的 forward token 数量呈 $O(N^2)$ 增长，导致吞吐增益崩溃。

### **提出的新方法与创新思路**
作者提出 **FlexDraft**，一种无损（lossless）、灵活适应不同 batch size 的 speculative decoding 框架，包含三大核心技术：

#### **(1) Attention Tuning (Attn Tuning)**
- 在目标模型最后几层引入可训练的 **mask-specific attention projectors**，仅用于预测 mask token（即 draft tokens）。
- 保持原始 autoregressive 路径冻结，确保 target model 的输出分布不变（lossless）。
- 实现 **block diffusion drafting**，单次 forward 并行生成多个 draft token，参数增量仅约 6%，高效复用预训练 FFN 知识。

#### **(2) Bonus-guided Calibration**
- 引入轻量级 **2-layer MLP**，以 resolved bonus token embedding 和 mask token 隐藏状态为输入，生成对 draft logits 的校准偏置。
- 显式缓解因 bonus token 不可见造成的 **draft-verification mismatch**，提升 draft 一致性与接受率。

#### **(3) Flex Decoding**
- 动态切换解码策略：
  - 小 batch size → 使用 **parallel draft & verify**，利用并行性减少等待时间；
  - 大 batch size → 切换至 **sequential draft then verify**，避免冗余分支带来的计算爆炸。
- 自适应剪枝验证长度（Selective Verification），基于 draft confidence 剪掉低概率接受路径，进一步降低 overhead。

### **相比现有方法的优势**
| 维度 | 现有方法局限 | FlexDraft 改进 |
|------|---------------|----------------|
| **质量保证** | 部分方法需持续预训练，可能导致质量下降 | 冻结 autoregressive 路径，严格保留 target distribution |
| **效率** | 并行方法冗余高，大 batch 下速度崩溃 | Flex Decoding 动态适配，避免 $O(N^2)$ 开销 |
| **一致性** | 缺乏对 bonus token 的建模能力 | Bonus-guided Calibration 主动对齐 draft 与 verification |
| **灵活性** | 多数方法固定执行模式 | 单一模型支持两种 inference 模式，无需重训 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **通用推理与生成任务基准**：
  - `GSM8K`（数学推理）
  - `MATH`（复杂数学题）
  - `HumanEval` 和 `MBPP`（代码生成）
  - `MT-Bench`（多轮对话质量评估）

### **实验设置**
- **目标模型**：Qwen3 系列（1.7B, 4B, 8B 参数规模）
- **硬件平台**：NVIDIA A100 GPU
- **训练数据**：来自 `mlabonne/open-perfectblend2` 的 300K 样本
- **batch size 默认为 1**，并在 batch size 扩展实验中测试从 1 到 16 的表现

### **评估指标**
- **Average Acceptance Length (T)**：每次验证平均接受的 draft token 数量
- **Speedup**：相对于标准 autoregressive decoding 的加速比
- **Per-step Latency**：单个 draft-verify 步骤的执行延迟

### **基线方法对比**
- **Parallel Speculative Decoding**：
  - `BiTA`, `Apple MTP`
- **Strong Baselines**：
  - `EAGLE-3`（基于特征融合的高质量 draft）
  - `DFlash`（基于 block diffusion 的先进方法）
  - `DART`

> 注：未将 `TiDAR` 作为主对比基线，因其为 hybrid diffusion-autoregressive 方法，不具备标准 speculative decoding 的 lossless guarantee。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1 & Table 2）**
在 **Qwen3-8B** 上的结果摘要如下：

| Method       | GSM8K Speedup | MATH Speedup | HumanEval Speedup | MBPP Speedup | MT-Bench Speedup |
|--------------|---------------|--------------|--------------------|-------------|------------------|
| EAGLE-3      | 3.40×         | 3.34×        | 2.47×              | 2.45×       | 1.75×            |
| DFlash       | 3.47×         | 3.27×        | 2.28×              | 2.33×       | 1.64×            |
| **FlexDraft** | **4.57×**     | **4.40×**    | **3.25×**          | **3.04×**   | **2.13×**        |

> ✅ **平均加速达 4.59×**，且在所有任务上均显著优于现有方法。

在更公平的扩展训练设置下（Table 2）：
- FlexDraft 在 `GSM8K` 达到 **5.88× speedup**，`MATH` 达 **5.79×**
- 即使未使用 DFlash 的专有训练数据，仍实现 **相当甚至更好性能**

### **与基线方法的对比结果**
- 相比 `Apple MTP` 和 `BiTA`：FlexDraft 接受长度更长（如 6.12 vs 3.30），speedup 提升超过 60%
- 相比 `EAGLE-3` 和 `DFlash`：在小 batch 下通过并行 drafting 减少等待；在大 batch 下通过 decoupled execution 避免冗余，整体表现更鲁棒

### **消融实验结果（Ablation Study）**
#### **组件有效性分析（Figure 7）**
- **仅 Attn Tuning**：基础 block diffusion 能力
- **+ Bonus-guided Calibration**：平均 speedup 提升 ~0.3–0.5×，显著改善 draft 对齐
- **+ Selective Verification**：进一步减少无效验证路径，提升 ~0.2× 加速

#### **其他关键发现**
- **层数选择（Figure 6）**：默认使用 **10 层** draft depth，在 speedup 与 overhead 间取得最佳平衡
- **Latency 对比（Figure 5）**：FlexDraft 单步延迟低于 EAGLE/dFlash，因无需独立 autoregressive drafting 阶段
- **Batch Size 影响（Figure 4）**：
  - 小 batch（≤2）：parallel mode 更优（隐藏 memory latency）
  - 大 batch（>2）：sequential mode 更高效（避免冗余分支）
  - **自适应切换机制有效防止 speedup 崩溃**

#### **目标知识复用优势（Table 3）**
| Setting        | GSM8K T | Speedup |
|----------------|--------|---------|
| Full param     | 3.78   | 3.04×   |
| **Attn Tuning** | **5.86** | **4.23×** |

👉 表明复用 target model 的 frozen FFN 层能显著提升 draft 质量与接受率。

---

## **4. 关键结论和发现**

### **主要发现**
1. **并行 speculative decoding 的瓶颈在于不确定性**：
   - Bonus token 和 acceptance length 的不可知性导致 draft-verification 不匹配与冗余计算。
2. **轻量化适配即可激活 block diffusion 能力**：
   - 仅 tuning attention projectors 即可在冻结主体模型的前提下实现高质量并行 drafting。
3. **动态执行策略至关重要**：
   - 不存在“一刀切”的最优模式，**Flex Decoding 的 batch-aware 切换机制是维持高性能的关键**。
4. **Bonus token 是对齐的核心信号**：
   - 显式利用 bonus token 进行 logit 校准可显著提升 draft 接受率。

### **方法的局限性**
- 当前设计依赖于对 target model 最后若干层进行微调，虽参数量小但仍需一定训练成本。
- Bonus-guided Calibration 模块需要获取 bonus token embedding，在极端低延迟场景下可能引入轻微同步开销。
- 目前实验集中在 Qwen 系列模型，跨架构泛化能力有待进一步验证。

### **未来工作方向**
- 探索完全无需训练的 zero-shot 或 prompt-based draft adaptation。
- 将 FlexDraft 思路扩展至 vision-language 模型或多模态生成。
- 结合 **tree-based speculation** 与 block diffusion，构建更高效的混合 speculative inference 架构。
- 在真实服务系统中部署，研究其在动态负载下的自动调优能力。

---

> 🔚 **总结一句话**：  
> **FlexDraft 通过 Attention Tuning 实现轻量 block diffusion，借助 Bonus-guided Calibration 提升一致性，并以 Flex Decoding 动态适配 batch size，实现了 lossless、高效且鲁棒的 speculative decoding 新范式，在 Qwen3-8B 上达到平均 4.59× 加速，显著超越现有 SOTA 方法。**

</details>

---

### 4. [Fast Tensorization of Neural Networks via Slice-wise Feature Distillation](https://arxiv.org/abs/2605.19842)

**Authors**: Safa Hamreras, Sukhbinder Singh, Rom\'an Or\'us  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.19842v1  

#### Abstract
We propose a scalable tensorization framework for neural network compression based on slice-wise feature distillation. Unlike conventional tensor decomposition methods that rely on costly global finetuning, our approach decomposes the network into slices consisting of either individual layers or blo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast Tensorization of Neural Networks via Slice-wise Feature Distillation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Tensorization** 方法（如 Tucker 或 MPO 分解）虽然能有效压缩神经网络参数，但通常依赖于全局的端到端 fine-tuning 来恢复模型性能。这种方法存在以下问题：
- **计算成本高**：需要在整个模型上进行反向传播，内存和计算开销大；
- **优化效率低**：全局 fine-tuning 收敛慢，尤其在大规模模型中；
- **对数据依赖强**：需要大量训练数据来稳定恢复性能。

此外，直接基于权重近似（如 SVD 初始化）的方法往往无法保留模型的功能行为，导致显著精度下降。

### 提出了什么新方法或新思路
本文提出了一种名为 **slice-wise feature distillation** 的新型 tensorization 框架，其核心思想是：
- 将预训练模型划分为多个独立的 **slice**（可以是单层、MLP block 或连续层组）；
- 对每个 slice 独立地进行 tensor decomposition（如 Tucker 或 MPO）；
- 不通过任务级监督进行全局 fine-tuning，而是采用 **feature distillation** 的方式，在每个 slice 上局部优化，使其输出激活（activations）尽可能匹配原始模型对应 slice 的输出；
- 使用简单的 **MSE loss** 最小化 tensorized slice 和原 slice 输出之间的差异。

该方法将 tensorization 后的性能恢复转化为一个模块化的、可并行的特征蒸馏问题。

### 相比现有方法的优势
- ✅ **更强的性能恢复能力**：利用中间层特征作为监督信号，比仅靠最终任务标签更丰富，有助于更准确地重建功能行为；
- ✅ **更高的优化效率**：各 slice 可独立、异步、并行优化，无需维护完整模型状态，显著降低内存压力；
- ✅ **更好的数据效率**：由于使用多层级的中间表示，即使在少量数据下也能实现良好恢复；
- ✅ **为全局 fine-tuning 提供更好初始化**：局部蒸馏后的模型可作为高质量起点，进一步提升高压缩率下的最终性能；
- ✅ **天然支持分布式训练**：slice 间无梯度同步需求，适合大规模分布式环境。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 模型类型 | 数据集 |
|--------|-------|
| CNN 实验 | **CIFAR-10**, **CIFAR-100** |
| LLM 实验 | **OpenWebText** 子集（25k 序列） |

### 实验设置和评估指标

#### CNN 实验（ResNet-34）
- **硬件平台**：AWS g5.8xlarge（NVIDIA A10G GPU）
- **压缩方法**：Tucker decomposition 用于所有 3×3 卷积层（排除 4 个敏感层）
- **压缩率（Compression Rate, CR）定义**：
  $$
  \text{CR} = \frac{\text{Original Params} - \text{Compressed Params}}{\text{Original Params}}
  $$
- 测试了两个压缩率：**CR = 0.5** 和 **CR = 0.7**
- **评估指标**：
  - Top-1 / Top-5 Accuracy
  - 优化时间（分钟），仅统计前向/反向传播耗时
- **训练配置**：
  - 局部 tensorization（local）：batch size=8, lr=0.001 (Adam)
  - 全局 tensorization（global）：batch size=16/64, lr=0.0005 (Adam)

#### LLM 实验（GPT-2 XL）
- **硬件平台**：H100 GPU, 400GB RAM, 100-core CPU
- **压缩目标**：总体 CR = 0.3
- **压缩范围**：仅对 **MLP blocks** 进行 MPO 分解（因其占参数主导）
- **每层压缩率**：统一设为 0.48
- **MPO 设置**：two-site factorization，平衡输入输出维度
- **训练配置**：
  - batch size=8, seq len=1024, 1 epoch, lr=5e-5
- **评估指标**：
  - Perplexity（WikiText, C4, LAMBADA）
  - Accuracy（PIQA, LAMBADA）

### 基线方法对比
- **Global Tensorization**：标准的端到端 fine-tuning，冻结未压缩层，微调整个 tensorized 模型；
- **Local Tensorization**（本文方法）：slice-wise feature distillation；
- **Hybrid 方法**：先 local 蒸馏 5 轮，再 global fine-tuning；
- **与其他压缩方法比较**（表5）：包括 APSSF（剪枝）、NC-CTD（耦合张量分解）、LJSVD（联合低秩）等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### 在 ResNet-34 + CIFAR-10 上的结果（CR=0.5）
| 方法 | Top-1 Accuracy (%) | 相对于原模型损失 |
|------|------------------|----------------|
| 原始模型（non-tensorized） | 95.04 | — |
| **Local Tensorization (50k)** | **94.70** | ↓0.34 |
| Global Tensorization (50k) | 89.47 | ↓5.57 |

> 📌 **Local 比 Global 高出 +5.23% 绝对精度**

- 即使只用 **10k 训练样本**，Local 方法仍能达到 **94.61%**，几乎无损。
- 优化速度方面，Local 达到最佳性能所需时间仅为 Global 的 **1/2.35**（51.38 vs 120.88 分钟）。

#### 在 ResNet-34 + CIFAR-100 上的结果（CR=0.5）
| 方法 | Top-1 Accuracy (%) |
|------|------------------|
| 原始模型 | 79.79 |
| **Local (50k)** | **78.81** |
| Global (50k) | 68.19 |

> 📌 **Local 超出 Global 超过 10%**

- 数据效率依然优异：从 50k → 10k 数据，精度仅下降 0.07%。

#### 更高压缩率（CR=0.7）下的表现
- 在 CIFAR-10 上，Local 仍优于 Global（89.19 vs 88.46）；
- 在 CIFAR-100 上，单独 Local 表现略逊于 Global，但引入 **hybrid 策略**后大幅提升：
  - **Local + Global（5轮+后续fine-tune）**：**74.22%**
  - 单独 Global：65.12%
  > ✅ 表明 local 提供了更优的初始化

#### 与 prior work 的横向比较（表5）
| 方法 | CIFAR-10 ΔAcc (%) | CIFAR-100 ΔAcc (%) |
|------|------------------|-------------------|
| **Ours (CR=0.5)** | **-0.34** | **-0.98** |
| APSSF [34] | +0.02 | — |
| NC-CTD [36] | +1.77 | — |
| LJSVD [37] | -1.14 | -1.42 |

> 虽然部分剪枝方法精度更高，但本文聚焦于改进 **tensorization 自身流程**，在同类方法中表现最优。

#### GPT-2 XL 实验结果（CR=0.3）
| Benchmark | Dense Model | Local Tensorization | Global Tensorization |
|----------|------------|--------------------|---------------------|
| LAMBADA (acc) | 51.21% | **42.38%** | 35.51% |
| LAMBADA (ppl) | 10.63 | **25.16** | 35.59 |
| WikiText (ppl) | 20.38 | 45.51 | **40.34** |
| C4 (ppl) | 50.03 | 121.12 | **100.70** |

> ⚠️ 观察：Local 在某些任务（LAMBADA）上表现更好，但在语言建模基准（WikiText/C4）上稍弱。

#### 优化时间对比（GPT-2 XL）
| 方法 | 单 GPU 时间（分钟） | 多 GPU 并行时间（理想情况） |
|------|------------------|-------------------------|
| Global Tensorization | 110.25 | — |
| Local Tensorization（串行） | 531.35 | — |
| Local Tensorization（并行，48 GPUs） | — | **13.4** |

> 🔥 **并行下加速达 ~40 倍！**

尽管单设备上 Local 更慢（因小 slice 利用率低），但在分布式场景下具有巨大潜力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Slice-wise feature distillation 显著优于传统 global fine-tuning**，尤其是在中等压缩率下可实现“近无损”压缩；
2. ✅ **模块化设计带来三大优势**：高效性、可扩展性和数据效率；
3. ✅ **适用于多种架构**：在 CNN（ResNet-34）和 LLM（GPT-2 XL）上均验证有效；
4. ✅ **特别适合分布式系统**：slice 独立优化，天然支持跨节点并行；
5. ✅ **可作为高质量初始化**：结合 hybrid 策略可在高压缩率下取得更好结果。

### 方法的局限性
- ❌ 在单设备环境下，尤其是大模型上，**局部 slice 优化可能导致 GPU 利用率低下**（因 workload 小且不连续）；
- ❌ 完全去中心化的优化可能忽略层间依赖关系，在极端压缩下不如全局优化灵活；
- ❌ 当前方法主要用于 **structured compression**，尚未整合 pruning 或 quantization 形成混合 pipeline。

### 未来工作方向
- 🔮 探索 **adaptive slice selection**：根据 layer importance 动态决定 slicing granularity；
- 🔮 引入 **alternative loss functions**：如 cosine similarity、KL divergence 替代 MSE；
- 🔮 扩展至更大规模 Transformer 架构（如 Llama、Mixtral）；
- 🔮 开发 **hybrid compression pipelines**：将 tensorization 与 pruning / quantization 结合；
- 🔮 研究 **multi-slice joint distillation**：将相邻 block 分组以提高硬件利用率同时保持模块性。

---

> 💡 **总结一句话**：  
> 本文提出的 **slice-wise feature distillation** 为 tensorization 提供了一个高效、可扩展的新范式——它不再依赖昂贵的全局 fine-tuning，而是通过局部特征匹配实现快速精准恢复，在分布式环境中展现出巨大潜力，推动了 TN-based 压缩走向实用化。

</details>

---

### 5. [FedADAS: Communication-Efficient Federated Distillation for On-Device Driver Yawn Recognition in Vehicular Networks](https://arxiv.org/abs/2605.19480)

**Authors**: Ahmed Mujtaba, Gleb Radchenko, Marc Masana, Radu Prodan  
**Category**: cs.DC  
**Published**: 2026-05-20  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.19480v1  

#### Abstract
Driver fatigue is a critical safety concern in advanced driver assistance systems. Driver monitoring models trained off-site on static datasets adapt poorly to real-world conditions, while standard federated learning imposes high communication overhead, assumes homogeneous architectures, and struggl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FedADAS: Communication-Efficient Federated Distillation for On-Device Driver Yawn Recognition in Vehicular Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Federated Learning (FL)** 在车载网络中的应用面临三大挑战：
- **设备异构性**（Device heterogeneity）：不同车辆的边缘设备计算能力差异大，难以部署统一模型架构；
- **通信开销高**：标准 FL 需频繁传输完整的模型参数，对带宽受限的车载环境不友好；
- **非独立同分布数据**（Non-IID）：驾驶员行为、传感器配置和环境条件导致本地数据高度个性化，影响全局模型收敛。

此外，现有方法大多未验证在真实边缘硬件上的训练可行性，限制了实际部署。

### 🚀 提出的新方法与创新
本文提出 **FedADAS** —— 一种基于 **Federated Distillation (FD)** 的新型协作学习框架，具有以下核心创新：

1. **完全模型异构支持（Full Model Heterogeneity）**
   - 各客户端可运行**结构不同的 DL 模型**（如 ME-Net 和 PE-Net），无需共享相同架构；
   - 仅通过交换在公共无标签数据集上的 **soft logits** 进行知识聚合，实现跨架构协同学习。

2. **极低通信成本**
   - 不传输模型参数，只上传 softmax logits（每轮仅 0.02 MB），相比传统 FL 减少高达 **9974× 通信量**；
   - 特别适用于资源受限的 vehicular edge 设备。

3. **端到端边缘训练支持**
   - 首次完整实现从 **训练到推理全过程都在边缘设备上完成**（Jetson AGX Orin / NANO）；
   - 引入两个轻量化 yawn 分类架构：
     - **ME-Net**（Memory-Efficient, 0.6 MB）
     - **PE-Net**（Performance-Efficient, 99.7 MB）

4. **通用性强**
   - 基于 logit-exchange 和 KL divergence 的机制无任务特定假设，适用于多类别感知任务。

### 🔍 相比现有方法的优势
| 方法 | 是否支持异构模型 | 是否支持边缘训练 | 公共数据 | 通信内容 | 客户端规模 |
|------|------------------|------------------|----------|-----------|------------|
| FedAvg [20] | ❌ | ⚠️（通常云端模拟） | ❌ | model params | 小规模 |
| FedBiKD [25] | ❌ | ❌ | ✅ | model params + logits | ≤20 |
| FedCMD [5] | ❌ | ❌ | ✅ | model params + logits | ≤30 |
| DB-EPFD [11] | ⚠️（部分异构） | ❌ | ❌ | partial params | ≤20 |
| **FedADAS (Ours)** | ✅（完全异构） | ✅（实测 Jetson） | ✅ | **soft logits only** | **up to 115** |

> ✅ FedADAS 是首个在 **大规模异构车队场景中验证可行性的 FD 系统级实现**。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 主要使用 **YawDD+ [21]**：作者发布的 YawDD 数据集增强版，提供帧级别标注，用于精确的打哈欠识别；
- 包含真实驾驶视频，涵盖多种光照、视角和个体差异；
- 所有模型输入调整为 224×224 图像，采用 ImageNet 归一化与数据增强（水平翻转、旋转、颜色抖动）。

### ⚙️ 实验设置
- **参与客户端数量 N ∈ {3, 10, 25, 115}**
  - N=3~25：dashboard 视角（车内前向摄像头）
  - N=115：引入 mirror-view（后视镜视角），模拟严重的协变量偏移（covariate shift）
- 每辆车拥有一个独特驾驶员的数据，构成极端 Non-IID 场景；
- 每个客户端保留约 10% 本地数据作为共享公共数据集 $ D_{pub} $，用于 KD；
- 总共进行 **T = 20 轮通信**；
- 温度参数 $ T = 1.0 $，优化器为 Adam（lr=0.001），batch size=32。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Personalization** | 模型在本地数据上的准确率（intra-vehicle accuracy） |
| **Generalization** | 模型在其他车辆数据上的平均准确率（inter-vehicle accuracy） |
| **BAM (Balanced Accuracy Metric)** | Personalization 与 Generalization 的几何平均，衡量个性化与泛化的平衡 |
| **Communication Cost** | 每轮上传的数据大小（MB） |
| **Inference Time** | Jetson 平台上的单帧推理延迟（ms） |
| **Epoch Training Time** | 边缘设备上每 epoch 的训练时间（分钟） |
| **Efficiency Metrics** |  
&nbsp;&nbsp;– $ \eta_{\text{inference}} = \frac{\text{FPS} \times \text{Accuracy}}{\text{Model Size}} $<br>
&nbsp;&nbsp;– $ \eta_{\text{training}} = \frac{\text{Accuracy}}{\text{Epoch Time} \times \text{Model Size}} $

### 🆚 基线方法对比
- **FedAvg [20]**：经典联邦学习算法，作为主要对比基准；
- 自身变体（如 ME-Net vs PE-Net）也用于分析容量影响。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 4 & 5）

#### ✅ 分类性能（Yawn Recognition）
| 模型 | 准确率 (%) | F1-Score (%) | 模型大小 (MB) | Jetson NANO 推理时间 (ms) |
|------|------------|-------------|---------------|----------------------------|
| **ME-Net (Ours)** | 99.30 | 97.99 | **0.6** | 3.81 |
| **PE-Net (Ours)** | **99.39** | **98.25** | 99.7 | **1.99** |

> 💡 PE-Net 在 Jetson NANO 上达到 **1.99ms 推理延迟**，满足实时安全响应需求。

#### ✅ 检测性能（结合 YOLO11）
| 模型组合 | mAP50-95 (%) | mAP50 (%) | NANO 推理时间 (ms) |
|---------|--------------|-----------|--------------------|
| PE-Net + YOLO11x | 99.39 / 95.41 | – / 99.41 | 1.99 / 153.1 |
| （分类/检测） | （cls） / （det） | – / – | （cls） / （det） |

#### ✅ 协作学习性能（FedADAS vs FedAvg）
| Client 数量 | 方法 | 模型 | Personalization (%) | Generalization (%) | BAM (%) |
|------------|--------|-------|------------------------|-----------------------|---------|
| 3 | FedAvg | PE-Net | 99.78 | 99.78 | 99.78 |
| 3 | FedADAS | PE-Net | 99.50 | 99.15 | 99.33 |
| 115 | FedAvg | PE-Net | 76.35 | 76.35 | 76.35 |
| 115 | **FedADAS** | **PE-Net** | **98.23** | **77.58** | **87.18** |

> ✅ 在 N=115 极端 Non-IID 条件下，FedADAS 的 PE-Net 实现 **21.88% 的个性化提升**，且 BAM 显著优于 FedAvg。

#### ✅ 通信与效率对比（Table 5）
| 方法 | 模型 | 每轮通信成本 | Jetson NANO 每轮总耗时 |
|------|------|----------------|--------------------------|
| FedAvg | ME-Net | 1.2 MB | ~14.26 s |
| FedAvg | PE-Net | **199.4 MB** | ~14.26 s |
| **FedADAS** | **ME-Net / PE-Net** | **0.02 MB** | **~10.15–15.02 s** |

> ✅ FedADAS 将通信成本降低 **60× 至 9974×**，尤其对大模型（PE-Net）优势巨大。

#### ✅ 效率评分（Fig. 3）
- **ME-Net** 在 **inference efficiency** 和 **training efficiency** 上均表现最优；
- 尽管 PE-Net 推理最快，但由于参数膨胀，其训练效率较低；
- 表明：**推理快 ≠ 训练高效**，需综合考量。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **FedADAS 显著优于传统 FL**，尤其是在高客户端参与度（N ≥ 10）和极端 Non-IID 场景下，展现出更强的鲁棒性和个性化能力；
2. **软 logit 蒸馏有效缓解通信瓶颈**，实现近万倍通信压缩，适合车载低带宽环境；
3. **完全模型异构是可行的**，允许各车根据硬件灵活选择模型（小内存用 ME-Net，高性能用 PE-Net）；
4. **轻量模型（ME-Net）虽具高效率，但在严重域偏移下泛化受限**，因学生模型容量不足无法吸收教师知识（“capacity bottleneck”）；
5. **公共数据集质量比数量更重要**：当视角变化（dashboard → mirror-view）时，即使数据更多，泛化性能仍下降，说明代表性优先于规模。

### ⚠️ 局限性
1. **隐私风险依然存在**：
   - 虽然不传梯度，但共享的公共数据可能泄露敏感信息；
   - 半诚实服务器仍可通过分析 $ D_{pub} $ 推断用户特征。
2. **依赖高质量公共数据集构建机制**：当前依赖手动收集或开放数据，缺乏自动化、去中心化的采样协议；
3. **温度调度固定**：未探索动态温度调节以进一步提升蒸馏效果；
4. **未考虑动态加入/退出客户端**：现实车队中车辆进出频繁，需更灵活的同步机制。

### 🔮 未来工作方向
1. 设计 **代表性的公共数据采样协议**，确保跨视角、跨人群的多样性；
2. 加强 **隐私保护机制**，例如结合差分隐私或加密技术处理 soft logits；
3. 引入 **自适应温度控制** 和 **课程学习策略** 提升 KD 效果；
4. 支持 **异步更新与弹性客户端管理**，适应动态 vehicular network；
5. 扩展至更多 DMS 任务（如眼睑闭合、头部姿态估计）、多模态输入（IR + visible）。

---

## ✅ 总结
**FedADAS** 成功将 **Federated Distillation** 应用于真实车载边缘环境，解决了传统 FL 在 **设备异构性、通信开销和 Non-IID 数据** 方面的根本难题。其实验验证覆盖了从模型设计、边缘训练、协作学习到系统部署的全链条，是目前最接近实际落地的 vehicular edge AI 框架之一。

> 🔗 开源地址：https://opensource.silicon-austria.com/mujtabaa/fedadas

</details>

---

### 6. [Projecting Latent RL Actions: Towards Generalizable and Scalable Graph Combinatorial Optimization](https://arxiv.org/abs/2605.19721)

**Authors**: Franco Terranova (UL, LORIA, Inria), Guillermo Bernardez (UC Santa Barbara), Albert Cabellos-Aparicio (UPC), Nina Miolane (UC Santa Barbara), Abdelkader Lahmadi (LORIA, UL, Inria)  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.19721v1  

#### Abstract
Graph combinatorial optimization (GCO) has attracted growing interest, as many NP-hard problems naturally admit graph formulations, yet their combinatorial explosion renders exact methods computationally intractable. Recent advances in Reinforcement Learning (RL) combined with Graph Neural Networks ...

---

### 7. [CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs](https://arxiv.org/abs/2605.19269)

**Authors**: Han Guo, Jack Zhang, Arjun Menon, Driss Guessous, Vijay Thakkar, Yoon Kim, Tri Dao  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.19269v1  

#### Abstract
Transformer training systems are built around dense linear algebra, yet a nontrivial fraction of end-to-end time is spent on surrounding memory-bound operators. Normalization, activations, residual updates, reductions, and related computations repeatedly move large intermediate tensors through globa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现代 **Transformer** 和 **LLM** 训练系统虽然在 **GEMM**（矩阵乘法）等计算密集型操作上高度优化，但大量时间仍消耗在**内存带宽受限的操作**上，如：

- **Normalization**（如 RMSNorm）
- **Activation functions**（如 SwiGLU）
- **Residual updates**
- **Reductions**（如 log-sum-exp）

这些操作频繁地将中间张量写入全局内存，造成显著的 **data movement overhead**，成为训练效率的瓶颈。

传统框架（如 PyTorch）将这些操作表示为独立的 kernel，导致无法进行跨算子融合，而手动编写融合 kernel 虽高效但开发成本高、难以维护。

---

### 🚀 提出的新方法：CODA

**CODA** 是一种新的 **GPU kernel 抽象**，其核心思想是：

> 将非 GEMM 的内存密集型操作重参数化为 **GEMM epilogue** 的一部分，在 GEMM 输出仍在片上时直接处理，避免写回全局内存。

#### 核心机制：
- **固定 GEMM mainloop**：复用专家级优化的 GEMM 主循环（如基于 Tensor Core）。
- **可编程 epilogue**：在 GEMM 输出 tile 还在寄存器或共享内存中时，执行一系列轻量级变换。
- **引入五类 epilogue primitives**：
  1. **Elementwise / Pairwise Maps**：如 SwiGLU、RoPE
  2. **Vector Loads/Stores**：广播权重（如 RMSNorm γ）
  3. **Tile Loads/Stores**：加载残差流、保存激活值
  4. **Tile Reductions**：局部归约（如 partial RMS 统计量）
  5. **Stateful Transforms**：在线统计（如 log-sum-exp）

通过代数重参数化，将原本分散的算子融合进 GEMM 的生命周期内。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | CODA |
|------|--------|------|
| **性能** | 多次内存读写，带宽受限 | 减少中间张量 materialization，提升带宽利用率 |
| **开发效率** | 需手写 CUDA，低抽象 | 高层组合式编程，支持 LLM 自动生成 |
| **自动化潜力** | 编译器难识别融合机会 | 结构化接口便于 LLM 或 DSL 自动合成 |
| **通用性** | 特定场景定制 | 覆盖 Transformer 正反向传播中几乎所有非 attention 计算 |

> ✅ **核心优势**：在保持硬件级效率的同时，提供接近框架级别的编程便利性。

---

## 2. 核心实验方法和设置

### 🧪 实验平台
- **硬件**：单块 **H100 GPU**
- **软件栈**：
  - PyTorch 2.10.0
  - CuTeDSL 4.4.2（底层 kernel 编程语言）
  - Liger Kernels 0.8.0, FlashInfer 0.6.10, QuACK 0.4.1

### 📊 评估任务与模型规模
- 模拟 **LLaMA-style Transformer 层** 的前向与反向传播
- 隐藏维度 $ d \in \{2048, 4096, 8192\} $，对应 ~1B, 7B, 70B 模型
- FFN 扩展率 8/3，词表大小 32768

### 🎯 评估指标
- **端到端 kernel 执行时间**
- **相对加速比**（speedup） vs 基线
- **数值精度误差**（vs FP32 参考实现）

### 🆚 基线方法
1. **cuBLAS + PyTorch + `torch.compile`**：标准框架流程
2. **Liger Kernels**：专为 LLM 训练优化的 Triton kernel 库
3. **FlashInfer**：面向推理的高效 attention 引擎（部分对比）
4. **Raw GEMM**（QuACK / cuBLAS）：仅执行矩阵乘法，作为理论上限参考

---

## 3. 主要实验结果和性能指标

### 📈 Kernel 级别加速（图 8 & 图 10）

| Kernel 类型 | CODA (LLM) 加速比 | 对比基线 |
|------------|------------------|---------|
| GEMM + RoPE | **~1.1–1.2×** | > cuBLAS + PyTorch |
| GEMM + SwiGLU | **~1.1–1.2×** | ≈ Liger，优于 FlashInfer |
| GEMM + Cross-Entropy | **~1.1–1.2×** | 显著优于独立 softmax + loss |
| GEMM-Residual-RMS-GEMM | **~1.1–1.3×** | 最高达 1.3× 加速 |

> ⚠️ 注意：这些 kernel 包含了原本需要多个独立 kernel 完成的任务（如 norm + act + proj），因此加速来自**融合减少内存访问**。

---

### 🧱 Block 级别加速（图 11）

评估完整 Transformer 子层序列（含辅助 reduction 和 glue ops）：

| 场景 | CODA (LLM) 加速比 | CODA (Human) 加速比 |
|------|------------------|--------------------|
| 前向传播（含 SwiGLU/RoPE） | **~1.05–1.15×** | **~1.1–1.2×** |
| 反向传播 | **~1.4–1.8×** | **~1.6–1.8×** |
| 前后向总和 | **~1.1–1.2×** | **~1.2–1.3×** |

> 💡 **关键发现**：反向传播收益更高，因 RMSNorm backward 中的 reduction 可被有效融合。

---

### 📉 数值精度表现（图 6）

- CODA 相对于标准 PyTorch 流程的输出误差：
  - **相对误差降低约 10–20%**
- 原因：更精确的 GEMM mainloop + 减少中间舍入误差（避免多次写回全局内存）

> ✅ 在提升性能的同时，**并未牺牲数值稳定性**，甚至略有改善。

---

### 🔬 消融实验（隐含于设计中）

虽然未显式列出消融表，但从方法描述可推断以下关键设计的作用：

| 设计要素 | 效果 |
|--------|------|
| **Tile-local reductions + auxiliary kernel** | 将 full softmax 替换为 partial LSE + small reduce，大幅降低带宽压力 |
| **Pairwise activation in epilogue** | 利用 Hopper Tensor Core 寄存器布局，避免 materialize 中间对 |
| **LLM-assisted authoring** | 即使由 LLM 生成，性能仍接近人工编写 kernel，验证接口易用性 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **GEMM epilogue 是融合内存密集型操作的理想场所**  
   利用 GEMM 输出 tile 仍在片上的时机，可高效执行 norm、activation、residual 等操作。

2. **CODA 接口兼具高性能与高表达力**  
   五类 primitive 足以覆盖 Transformer 前后向传播中几乎所有非 attention 计算。

3. **支持 LLM 自动生成高性能 kernel**  
   实验表明，Claude Code 生成的 CODA kernel 性能接近人工编写水平，展示了“AI 写 AI kernel”的可行性。

4. **反向传播优化空间更大**  
   特别是 RMSNorm backward 中的 reduction 可通过前置到相邻 GEMM 边界来融合，带来显著加速。

---

### ⚠️ 局限性

1. **当前聚焦单 GPU kernel**  
   尚未扩展到分布式训练场景（如 tensor parallelism 中的通信融合）。

2. **依赖特定 GEMM 主循环结构**  
   需要预设高效的 mainloop，灵活性受限于底层实现（如 CuTeDSL）。

3. **可能模糊模块语义**  
   过度融合会破坏 PyTorch 等框架中的模块边界，不利于调试和组合。

4. **适用架构有限**  
   当前针对 LLaMA-style 架构设计，对其他结构（如 encoder-decoder）需重新适配。

---

### 🔮 未来工作方向

1. **扩展至分布式训练**  
   将 epilogue fusion 与通信原语结合（如 AllReduce + norm）。

2. **构建自动 reparameterization 编译器**  
   从标准 PyTorch IR 自动转换为 CODA 兼容的 GEMM-epilogue 形式。

3. **集成进主流训练框架**  
   如 TorchInductor、JAX XLA，实现透明加速。

4. **探索更多 epilogue primitive**  
   支持动态路由（如 MoE）、条件计算等新兴结构。

---

## 总结

> **CODA 成功架起了“框架级生产力”与“硬件级效率”之间的桥梁**。它证明了：通过合理的抽象设计，即使是复杂的 Transformer 计算，也可以被系统性地重写为高效的 GEMM-epilogue 程序，并且这一过程可以由人类或 LLM 共同完成。

这不仅为 LLM 训练系统提供了新的优化路径，也为“AI 自我优化”时代下的 kernel engineering 开辟了新范式。

</details>

---

### 8. [D$^3$-Subsidy: Online and Sequential Driver Subsidy Decision-Making for Large-Scale Ride-Hailing Market](https://arxiv.org/abs/2605.20036)

**Authors**: Taijie Chen, Rui Su, Siyuan Feng, Laoming Zhang, Hongyang Zhang, Haijiao Wang, Zhaofeng Ma, Jintao Ke  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.20036v1  

#### Abstract
Ride-hailing platforms like DiDi Chuxing operate in highly dynamic environments where balancing driver supply and passenger demand is critical. Although driver-side subsidies serve as a primary lever to align these forces and improve key KPIs like completed rides (\texttt{Rides}) and gross merchandi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：D³-Subsidy**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在大规模网约车市场（如 DiDi、Uber）中，平台需要通过**driver-side subsidies**（司机端补贴）来激励司机接单，以平衡供需、提升关键业务指标（如完成订单数 *Rides* 和总交易额 *GMV*）。然而，这一过程面临三大挑战：
- **动态性**：市场环境高度随机且非平稳，需对突发需求变化快速响应；
- **预算约束**：必须遵守全局补贴率上限（subsidy rate cap），即补贴总额不能超过 GMV 的固定比例；
- **可扩展性**：实时为每个订单-司机对进行优化计算成本过高，难以在城市级别部署。

现有方法（如强化学习 RL 或行为克隆 BC）往往忽视训练与推理之间的不一致性（train-inference gap），或因探索导致预算违规，不适合生产环境。

---

### **提出了什么新方法或新思路**
本文提出 **D³-Subsidy**（Dynamic Driver-side Diffusion-based Subsidy），一个基于扩散模型的离线决策框架，用于城市级司机补贴控制。其核心创新包括：

#### **(1) Prefix-Conditional Diffusion Model**
- 引入前缀条件扩散机制，在生成未来轨迹时**固定已观测的历史状态**（prefix），仅对未来的后缀（suffix）进行去噪采样。
- 这确保了训练与在线部署的一致性（bridge the train-inference gap），避免了传统序列建模对未来状态的错误假设。

#### **(2) Constraint-Aware Score Objective**
- 在扩散模型的目标函数中引入**预算可行性感知评分机制**，当生成的轨迹违反补贴率限制时施加惩罚，从而显式鼓励满足约束的策略。

#### **(3) Context-Conditioned Inverse Dynamics Decoder**
- 不直接输出动作，而是设计了一个上下文感知的逆动力学模块，将去噪后的状态序列映射为城市级控制信号 $ \lambda_t $。
- 上下文 $ c $ 包含目标 KPI、预算制度等信息，使策略具备**部署时可控性**（controllability），可通过调整输入实现不同运营目标间的切换。

#### **(4) Two-Stage Training with PEFT**
- 采用多城市预训练 + 参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）的两阶段训练范式，提升模型在异构城市间的迁移能力，尤其适用于冷启动场景。

---

### **相比现有方法的优势**
| 维度 | D³-Subsidy | 传统方法（如 RL/BC/DT） |
|------|------------|------------------------|
| **预算合规性** | 显式建模约束，减少违规风险 | 软约束处理易超标 |
| **部署一致性** | 前缀条件保证历史不可变 | 自回归模型可能“重写”过去 |
| **可扩展性** | 输出单一城市级 $ \lambda $，通过 dual-based mapping 扩展到细粒度补贴 | 多需逐对优化或高维动作空间 |
| **泛化能力** | 支持跨城市迁移与冷启动 | 通常依赖本地数据 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 数据来源：**DiDi 的真实广播日志**（broadcast logs）
- 地理范围：**巴西 133 个城市**
- 时间跨度：连续 **28 天**
- 数据形式：城市-天级别的轨迹（city-day trajectories），每条轨迹按时间窗口聚合（2/5/10分钟）
- 特征维度：状态维度为 20，动作 $ \lambda_t \in (0,30] $
- 更多统计见附录 Table 7

---

### **实验设置和评估指标**

#### **评估方式**
- **离线评估**：使用 DiDi 高保真模拟器进行闭环 rollout（closed-loop rollouts）
- **线上 A/B 测试**：为期 7 天的真实系统测试（2026年2月1–7日），50%流量分配给 D³-Subsidy，其余为 Online 基线

#### **主要评估指标**
| 指标 | 定义 | 目标 |
|------|------|------|
| **Score(ξ)** | 结合 Rides 与补贴率合规性的综合得分：<br>$$
\text{Score}(\xi) = 
\begin{cases}
\text{Rides}(\xi), & \text{if } C_{\text{real}}(\xi) \leq C + \delta \\
\left(1 - \frac{C_{\text{real}}(\xi) - C}{\delta}\right)^\beta \cdot \text{Rides}(\xi), & \text{otherwise}
\end{cases}
$$ | 越高越好 |
| **Rides / GMV / DRV** | 完成订单数 / 总交易额 / 司机收入 | 提升越高越好 |
| **UnderGap(ξ)** | 补贴利用率不足程度：<br>$ \max(0, C - C_{\text{real}}(\xi)) $ | 越小越好（表示预算用得更充分） |
| **Cap Violation** | 是否超出容忍上限 $ C+\delta $ | 不允许 |

---

### **基线方法对比**
| 基线 | 类型 | 描述 |
|------|------|------|
| **Online** | 生产策略 | DiDi 当前线上运行的 predict-then-optimize 策略 |
| **BC** | 行为克隆 | 监督学习模仿历史动作 |
| **BCQ, CQL, IQL, TD3+BC** | Offline RL | 主流离线强化学习算法，抑制 OOD 动作 |
| **DT (Decision Transformer)** | 序列建模 | 基于 Transformer 的轨迹建模方法 |
| **DD (Decision Diffuser)** | 扩散策略 | 基于扩散的动作生成模型 |

所有 RL 方法均以最大化 `Score()` 为目标训练。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（离线评估）**

#### **表1：各城市平均 Score 对比（5分钟窗口为主）**
| 方法 | City A | City B | City C | **Overall Avg** |
|------|--------|--------|--------|------------------|
| **D³-Subsidy (Ours)** | **10824.43** | **2000.00** | **1792.48** | **4845.89** ✅ |
| Online | 10359.22 | 1938.57 | 1643.23 | 4645.44 |
| DT | 10618.92 | 1954.61 | 1674.31 | 4780.81 |
| DD | 10488.95 | 1987.89 | 1737.39 | 4742.29 |

👉 **D³-Subsidy 在所有城市和设置下均取得最高 Score**，整体领先第二名 DT 超过 **65 分**。

---

### **与基线方法的对比结果**
- 相比当前线上策略（Online）：
  - **Score 平均提升约 4.3%**
  - **Rides ↑1.59%，GMV ↑2.06%，DRV ↑2.31%**（来自线上 A/B 测试）
- 相比最强离线 RL 基线 DT：
  - 仍保持显著优势（p < 0.001，paired t-test）
- 在 **所有时间粒度（2/5/10min）下表现稳定**，说明方法鲁棒性强。

---

### **消融实验结果（Ablation Study）**

#### **表2：在 City C（5min）上的消融效果**
| 模型变体 | Score | 下降幅度 |
|---------|-------|----------|
| **D³-Subsidy (Full)** | 1792.48 | — |
| w/o Conditioning (**-C**) | 1737.58 | ↓3.1% |
| w/o Multi-City Pretraining (**-M**) | 1704.91 | ↓4.9% |
| w/o PEFT Fine-tuning (**-P**) | 1715.78 | ↓4.3% |

✅ 所有组件都对最终性能有贡献，尤其是**多城市预训练**最为关键。

---

### **其他重要实验发现**
#### **(1) 冷启动性能（Cold-Start Transfer）**
- 在从未见过的 3 个新城市上测试（无任何该城市训练数据）：
  
| 方法 | Average Score |
|------|---------------|
| **D³-Subsidy** | **656.80** ✅ |
| Online | 650.60 |
| DT | 643.66 |

👉 即使没有 fine-tuning，D³-Subsidy 凭借强大的跨城泛化能力依然最优。

#### **(2) 敏感性分析**
- **反向扩散步数**：性能在 50 步达到峰值，过多或过少都会下降；
- **惩罚指数 β**：随着 β 增大，D³-Subsidy 得分略有下降（因更严格惩罚超支），但仍始终优于 Online。

#### **(3) 线上 A/B 测试结果**
| 指标 | 提升幅度 | 推理延迟增加 |
|------|----------|--------------|
| **Rides** | **+1.59%** | +20ms |
| **GMV** | **+2.06%** | （可接受） |
| **DRV** | **+2.31%** | |
| **Cap Violation** | **0次**（未发生） | |

✅ 离线增益成功转化为线上收益，且满足所有运营约束。

---

## **4. 关键结论和发现**

### **主要发现**
1. **D³-Subsidy 是首个将 diffusion model 成功应用于网约车补贴控制的工作**，实现了安全、可部署的城市级决策。
2. **Prefix-conditional diffusion 有效解决了 train-inference gap**，使得生成轨迹严格尊重历史观测。
3. **context-conditioned inverse dynamics 提供了良好的可控性**，支持灵活调整运营目标。
4. **多城市预训练 + PEFT 极大地增强了跨域适应能力**，特别适合新城市冷启动。
5. **线上 A/B 测试验证了方法的实际价值**：在不突破预算的前提下，显著提升了 Rides 和 GMV。

---

### **方法的局限性**
- **依赖 dual-based mapping**：要求能从单一 $ \lambda $ 导出细粒度补贴，若平台机制变更则需重新适配；
- **扩散模型推理延迟较高**：虽然仅增加 20ms，但在极端低延迟场景中仍有优化空间；
- **对 context 设计敏感**：decoder 控制能力依赖 context 编码的质量；
- **未考虑司机个体差异**：仍是 city-level 统一策略，无法做到 fully personalized。

---

### **未来工作方向**
- 开发更高效的扩散采样策略（如 distillation into autoregressive policy）以降低延迟；
- 将方法扩展至 **spatially-aware zoning control**（区域级补贴）；
- 探索 **joint passenger & driver subsidy optimization** 的统一框架；
- 引入 **causal modeling** 来更好估计补贴的增量效应（ITE）；
- 研究如何进一步提升 **zero-shot transferability** 到全新国家或文化背景下的城市。

--- 

> ✅ **总结一句话**：  
> D³-Subsidy 通过结合 **diffusion generative modeling** 与 **offline sequential decision-making**，提出了一种**可部署、可控制、可迁移**的城市级补贴控制器，在真实网约车平台上实现了 KPI 提升与预算合规的双重目标，是工业级 AI 决策系统的有力实践。

</details>

---

### 9. [EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design](https://arxiv.org/abs/2605.19743)

**Authors**: Gioele Molinari, Florian Felten, Soheyl Massoudi, Mark Fuge  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.19743v1  

#### Abstract
Large Language Model (LLM) agents are increasingly applied to engineering design tasks, yet existing evaluation frameworks do not adequately address multi-agent systems that combine simulation, retrieval, and manufacturing preparation. We introduce a benchmark suite with three evaluation dimensions:...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前的 **LLM agents** 在工程设计任务中应用日益广泛，但现有的评估框架（如 AgentBench、ToolBench）主要关注通用工具调用或多轮对话能力，无法有效评估以下复杂场景：
- 多 agent 协同（multi-agent coordination）
- 结合仿真（simulation）、检索（retrieval）与制造准备（manufacturing）的端到端流程
- 长周期、高复杂度的 **High Performance Computing (HPC)** 训练流水线

因此，缺乏一个专门面向 **LLM-driven 工程设计** 的综合性评估基准。

### **提出了什么新方法或新思路**
本文提出两个核心贡献：

#### ✅ **ENGIAI：多智能体参考实现框架**
- 基于 **LangGraph** 构建的 **multi-agent system (MAS)**，采用 **supervisor 架构** 统筹多个专业 agent。
- 包含 **7 类 specialist agents**：
  - Engineering Agent（拓扑优化、仿真）
  - RAG Agent（文档问答）
  - ArXiv Agent（论文检索）
  - Search Agent（网络搜索）
  - HPC Agent（远程集群作业管理）
  - CLI Agent（本地命令执行）
  - Prusa Agent（3D 打印机控制）
- 支持通过自然语言交互协调跨工具链的工程任务，具备模块化扩展能力。

#### ✅ **三维度 Benchmark Suite**
首次系统性地构建了一个覆盖工程设计全流程的评估套件，包含三个独立但互补的评估维度：

| 维度 | 内容 | 创新点 |
|------|------|--------|
| **Workflow Benchmark** | 设计7种不同认知需求的 prompt 风格，测试 agent 在条件分支、语义消歧、工作记忆等任务中的表现 | 超越传统“函数调用”测试，引入真实工程中的复杂推理挑战 |
| **RAG Benchmark** | 提出 **gated scoring 机制**，仅当 agent 显式调用 `search_documents` 且参数正确时才得分 | 可隔离 retrieval 对决策的实际贡献，防止模型依赖先验知识“猜对” |
| **HPC Benchmark** | 测试 agent 是否能完整编排生成、提交、监控并评估在 SLURM 集群上的 ML 模型训练任务 | 当前唯一评估 LLM 在 HPC 场景下长周期指令跟随能力的工作 |

### **相比现有方法的优势**
| 方面 | 本工作 | 先前工作 |
|------|-------|---------|
| **任务范围** | 覆盖设计→仿真→制造→训练全链条 | 多集中于单一环节（如 FEA 或 G-code） |
| **评估粒度** | 引入认知风格分析（conditional branching, working memory） | 多为功能完整性检查 |
| **RAG 评估** | 使用 gated scoring 验证 retrieval 必要性 | 多数仅测文档理解，不验证下游应用 |
| **HPC 支持** | 端到端训练流水线编排 | 无相关评估 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
基于 **EngiBench** 框架提供的两个标准工程问题：
- **Beams2D**：二维悬臂梁拓扑优化（最小化 compliance），有明确 ground truth。
- **Photonics2D**：二维光子器件逆向设计（最大化电磁场重叠），跨物理域迁移测试。

### **实验设置**
#### **LLM 后端（共4个）**
| 类型 | 模型名称 |
|------|--------|
| Proprietary | `gpt-5-mini`, `gemini-3-flash` |
| Open-source (4B) | `qwen3-4b`, `qwen3.5-4b`（via Ollama） |

所有调用均设 `temperature=0` 并固定种子以增强可复现性。

#### **评估维度与指标**

##### **(1) Workflow Evaluation**
- **Prompt Styles (7种)**：
  - `FULL`: 显式数值输入
  - `NATURAL`: 定性描述（需澄清）
  - `W-RAND`: 随机导出参数
  - `W-DERIVED`: 参数由计算规则推导
  - `W-DISTRACT`: 存在干扰值（语义消歧）
  - `W-COND`: 条件分支（读取仿真结果决定后续操作）
  - `W-MULTI`: 多次导出（工作记忆）

- **综合评分 $ S_{\text{workflow}} $**：
  $$
  S_{\text{workflow}} = 0.65 \cdot S_{\text{design}} + 0.20 \cdot S_{\text{tool}} + 0.15 \cdot S_{\text{completion}}
  $$
  - $ S_{\text{design}} $: 设计质量（IoU, Obj, Constr 等加权）
  - $ S_{\text{tool}} $: 工具调用效率
  - $ S_{\text{completion}} $: 任务完成率（是否调用全部必要工具）

##### **(2) RAG Evaluation**
- **4个手工构造 prompt (P0–P3)**，要求从指定论文中提取参数用于设计。
- **三种模式对比**：
  - RAG-on：索引可用
  - RAG-off：禁用检索工具
  - Empty RAG：索引为空（测试是否盲目信任空结果）

- **Gated Scoring**：只有调用了 `search_documents` 且参数正确才计入分数。

##### **(3) HPC Training Evaluation**
- 任务：在远程 SLURM 集群上训练 cGAN / diffusion 模型。
- 步骤：生成脚本 → 提交作业 → 监控状态 → 下载模型 → 评估性能。
- **评分公式**：
  $$
  S_{\text{HPC}} = 0.70 \cdot \text{Step Completion} + 0.15 \cdot \text{Config Correctness} + 0.15 \cdot \text{Metric Extraction}
  $$

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) Workflow Performance (Beams2D)**
| 模型 | 平均 Task Completion (TC) | 平均 Combined Score (CO) |
|------|--------------------------|--------------------------|
| GPT-5-mini | **96%** | 0.68 |
| Gemini-3-Flash | **97%** | **0.69** |
| Qwen3-4B | 55% | 0.53 |
| Qwen3.5-4B | 78% | 0.63 |

- **条件分支（W-COND）最难**：GPT-5-mini 和 Gemini 分别为 93% 和 87%，而 Qwen3-4B 仅为 40%。
- Qwen3.5-4B 相比前代显著提升（+23% TC），显示小模型代际进步迅速。

#### **(2) Photonics2D 迁移表现**
| 模型 | W-RAND TC | W-DISTRACT TC | W-COND TC |
|------|----------|-------------|----------|
| GPT-5-mini | 100% | 100% | **40%** |
| Gemini-3-Flash | 100% | 100% | **53%** |
| Qwen3-4B | 100% | 100% | 20% |

- 所有模型在 W-COND 上严重退化，失败原因并非漏调用工具，而是 **branch inversion**（选错条件分支）。
- 表明跨领域条件判断仍是重大挑战。

#### **(3) RAG Evaluation**
| 设置 | 平均得分 |
|------|--------|
| RAG-on | ~**1.0** |
| RAG-off | **0.0**（因 gating 机制） |
| Empty RAG | 显著下降（除 P0 外） |

- **P0 得分较高**：因 `volfrac=0.35` 是常见默认值，可能被模型 memorized。
- **P2/P3（post-cutoff 论文）**：无 retrieval 时无法完成，证明 gated scoring 成功分离 retrieval 贡献。

#### **(4) HPC Orchestration**
| 模型 | 显式指令（Explicit）完成率 | 自然语言（Natural）完成率 |
|------|----------------------------|----------------------------|
| Gemini-3-Flash | **100%** | **100%** |
| GPT-5-mini | 70% | 50% |

- GPT-5-mini 在长流程中出现 **instruction degradation**，常遗漏最后一步 `evaluate_model`。
- Gemini 更鲁棒，说明不同 LLM 在长期任务保持能力上有差异。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Proprietary LLMs 在工程工作流中接近完美**  
   GPT-5-mini 和 Gemini-3-Flash 在大多数 prompt 风格下达到 96–97% 任务完成率，表明其已具备可靠执行结构化工程任务的能力。

2. ✅ **Conditional Branching 是最大瓶颈**  
   尤其在跨领域任务（如 Photonics2D）中，即使最优模型也仅达 53% 成功率，暴露了 LLM 在动态决策中的脆弱性。

3. ✅ **RAG 至关重要且可被量化验证**  
   gated scoring 成功证明：没有 retrieval，agent 几乎无法获取 post-training cutoff 的工程参数。

4. ✅ **HPC 编排可行但存在退化风险**  
   当前顶级闭源模型可以完成端到端训练流水线，但部分模型在自然语言指令下会丢失后期步骤。

5. ✅ **小模型代际提升显著**  
   Qwen3.5-4B 相比 Qwen3-4B 在 TC 上提升 23%，说明开源模型快速迭代正缩小与闭源差距。

### **局限性**
- **问题覆盖有限**：仅测试 Beams2D 和 Photonics2D，未涵盖 EngiBench 全部问题。
- **用户干预缺失**：缺少真实工程师参与的人类反馈闭环。
- **HPC 实验成本高**：仅测试两个闭源模型，未包含更大规模开源模型。
- **无单 agent 基线**：未进行 supervisor 架构的消融实验，无法确认多 agent 分解是否必要。

### **未来工作方向**
1. **扩展 benchmark 范围**  
   - 加入更多物理域问题（热传导、流体等）
   - 测试更大模型（如 70B 参数级）
   - 探索 prompt 敏感性与温度影响

2. **改进 agent 能力**
   - 引入 **few-shot tool-use 示例** 抑制冗余调用
   - 使用 **structured chain-of-thought** 提升条件推理准确性
   - 开发 **parameter planning stage** 实现语义消歧解耦

3. **增强 retrieval 与 HPC 能力**
   - 构建对抗性 RAG 测试（含冲突信息）
   - 探索 domain-specific chunking 与 re-ranking
   - 支持 agent 自动生成并调试训练代码（而非仅编排脚本）

4. **探索 scaling law**
   - 研究工具数量增长对 agent 性能的影响
   - 分析 tool-overcalling 是否可迁移或泛化

---

> **总结一句话**：  
> 该论文建立了首个面向 **LLM-driven 工程设计** 的多维度评估体系，并通过 **ENGIAI** 框架展示了多 agent 协同在整合仿真、检索与 HPC 中的巨大潜力，揭示了当前 LLM 在 **条件推理** 与 **长周期任务维持** 上的关键瓶颈，为未来智能工程系统的发展提供了坚实基础与明确方向。

</details>

---

### 10. [Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption](https://arxiv.org/abs/2605.19593)

**Authors**: Mert Yildiz, Pietro Spadaccino, Alexey Rolich, Francesca Cuomo, Andrea Baiocchi  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.19593v1  

#### Abstract
Modern deployments of Large Language Models (LLMs) increasingly require serving multiple models with diverse architectures, sizes, and specialization on shared, heterogeneous hardware. This setting introduces new challenges for resource allocation, dispatching, and scheduling, particularly under GPU...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文聚焦于**多模型 LLM 推理调度**在资源受限异构硬件环境下的挑战，特别是：
- **GPU 内存不足**时如何通过 **CPU-GPU 层级卸载（layer offloading）** 来支持多个大模型共存；
- 在动态请求场景下，**任务抢占（preemption）** 所带来的开销及其对系统效率的影响。

现有系统（如 vLLM）主要针对单一模型优化吞吐量，缺乏对**多模型并发调度**中模型切换、卸载策略和抢占成本的深入理解。本文填补了这一空白。

### 🚀 提出的新方法与新思路
本文并未提出一个全新的调度器，而是通过**系统的实证研究（empirical study）** 揭示了以下关键机制：

- **量化了 partial offloading 对 decode throughput 的非线性影响**，并指出其敏感度与模型大小强相关；
- **分解了 preemption 的完整生命周期开销**，首次明确指出：**模型重载（reload）是主导开销，而非 KV cache 迁移**；
- 提出了未来多模型 LLM 调度器应具备的 **六大核心特征集（feature set）**，为下一代调度系统设计提供指导原则。

### 🔍 相比现有方法的优势
| 方面 | 现有工作局限 | 本文突破 |
|------|---------------|----------|
| **Offloading 分析** | 多数仅比较全 GPU / 全 CPU 基线，未系统扫描不同层卸载比例 | 首次系统性地扫频从 0% 到 100% GPU 层驻留率，揭示连续性能曲线 |
| **Preemption 成本建模** | 多假设“轻量级”或依赖模拟，未实测各阶段耗时 | 实测拆解 preemption 流程，精确识别 reload 占 >98.5% 开销 |
| **跨硬件可迁移性** | 缺乏跨 GPU 架构的行为一致性验证 | 在 RTX 5000 和 RTX A6000 上重复实验，验证趋势一致性 |

---

## 2. 核心实验方法和设置

### 🧪 实验目标
- 评估 **layer-wise CPU-GPU offloading** 对推理吞吐的影响；
- 量化 **job-level preemption** 的真实开销，并分解其构成；
- 探索序列长度、模型规模、硬件平台等因素的交互效应。

### 📦 使用的模型（Models）
实验分为两个部分，使用不同的模型集合：

#### **Offloading 实验**
| Model | Params | Quantization | Layers | VRAM (Q4) |
|-------|--------|-------------|--------|-----------|
| Llama 3 8B | 8B | Q4 | 33 | 4.7 GB |
| Qwen3-32B | 32B | Q4 | 65 | 20 GB |
| Llama 2 70B | 70B | Q4 | 81 | 39 GB |

> 工具：Ollama v0.17.7，控制 GPU 层占比（0%~100%，步进10%，精细采样92%-98%）

#### **Preemption 实验**
| Model | Params | Precision | Layers | VRAM (FP16) |
|-------|--------|-----------|--------|-------------|
| Qwen2.5-3B | 3B | FP16 | 36 | 5.9 GB |
| Qwen3-8B | 8B | FP16 | 36 | 15.6 GB |
| Qwen2.5-14B | 14B | FP16 | 48 | 28.3 GB |

> 工具：HuggingFace Transformers + 手动 KV cache 管理

### 💻 硬件平台
| Server | CPU | GPU | Interconnect |
|--------|-----|-----|--------------|
| Server 1 | AMD Threadripper PRO 5995WX (64c) | 2×RTX 5000 Ada (32GB VRAM) | PCIe Gen4 x16 |
| Server 2 | 同上 | 2×RTX A6000 (48GB VRAM) | PCIe Gen4 x16 |

> 两台服务器共享相同 CPU 子系统，用于隔离 GPU 架构差异的影响。

### 📊 评估指标
| 场景 | 主要指标 |
|------|---------|
| **Offloading** | Decode Throughput (tok/s)，Normalized Throughput（相对于 100% GPU） |
| **Preemption** | Preemption Overhead (s)，Breakdown: KV transfer, unload, reload；Overhead vs Baseline (%) |

### ⚖️ 基线对比
- **Offloading Baseline**: 完全 GPU 执行（100% 层驻留）
- **Preemption Baseline**: 不中断的连续生成任务（7,000 tokens）
- 无直接算法类 baseline，而是与已有文献中的假设进行对比（如“preemption 是轻量操作”、“KV transfer 是瓶颈”等）

---

## 3. 主要实验结果和性能指标

### 📈 Offloading 结果

#### 关键发现：
- **Decode throughput 随 GPU 层比例增加呈非线性增长**，且小模型更敏感。
- 小模型（如 Llama3:8B）在接近 100% GPU 时出现陡增，而大模型（如 Llama2:70B）增长平缓。
- **Normalized throughput 更高在 RTX 5000 上**（相对低性能 GPU），因为其与 CPU 性能差距较小，offloading 惩罚更低。

| 模型 | 最佳 offloading 敏感区 |
|------|------------------------|
| Llama3:8B | >90% GPU 层才显著提升 |
| Qwen3-32B | 中等敏感，约线性 |
| Llama2:70B | 几乎线性，容忍度高 |

> 图4 显示：小模型偏离 linear reference 最远，表明不能用统一策略调度。

---

### 🔁 Preemption 实验结果

#### 实验设计：
- Job A：长任务，生成 7,000 tokens（greedy decoding）
- 在 `{100, ..., 5000}` tokens 处被抢占
- Job B：插入运行 500 tokens
- 测量整个 preempt-resume cycle 时间

#### 关键性能数据（Table II 平均值）：

| Preempted Model | Platform | Total Overhead (s) | Model Swap (%) | KV Transfer (%) | Overhead vs Baseline (%) |
|------------------|----------|--------------------|----------------|------------------|----------------------------|
| Qwen2.5-3B (5.9GB) | RTX 5000 | 2.98 | 99.4% | 0.6% | ~2.04% |
| Qwen3-8B (15.6GB) | RTX 5000 | 5.16 | 99.0% | 1.0% | ~2.11% |
| Qwen2.5-14B (28.3GB)| RTX 5000 | 7.31 | 99.1% | 0.9% | ~1.75% |
| Qwen2.5-3B         | RTX A6000 | 2.62 | 99.2% | 0.8% | ~1.72% |
| Qwen3-8B           | RTX A6000 | 4.06 | 98.5% | 1.5% | ~1.86% |
| Qwen2.5-14B        | RTX A6000 | 5.73 | 98.8% | 1.2% | ~1.61% |

#### 关键观察：
- **Preemption overhead 几乎恒定**，不随中断点位置变化（图5 平坦曲线）；
- **模型重载（reload）占总开销 >98.5%**，unload 次之，KV transfer <1.5%；
- **抢占模型（Job B）的大小不影响开销**，只取决于被抢占模型自身的 weight footprint；
- RTX A6000 上 reload 更快 → 更低绝对开销，说明硬件差异显著。

#### KV Cache Transfer 数据（Table III 示例 @5000 tokens）：
| Model | KV Size | GPU→CPU (ms) | CPU→GPU (ms) |
|-------|---------|---------------|---------------|
| Qwen2.5-3B | 178 MB | 18.0 | 14.4 |
| Qwen3-8B | 713 MB | 67.8 | 45.9 |
| Qwen2.5-14B | 951 MB | 86.7 | 62.5 |

> 即使最大传输也<100ms，远小于 reload 的数秒级别。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Offloading 敏感度高度模型依赖**：
   - 小模型对 GPU 层减少极为敏感，轻微 offload 导致 throughput 断崖式下降；
   - 大模型表现更线性，适合渐进式卸载。

2. **Preemption 开销本质是模型重载问题**：
   - **>98.5% 的 overhead 来自 model reload 和 unload**；
   - **KV cache transfer 可忽略不计（<1.5%）**，即使序列长达 5000 tokens；
   - 因此，**preemption 成本是固定的，与任务进度无关** → 可建模为 per-model constant。

3. **硬件平台显著影响行为**：
   - 更高性能 GPU（RTX A6000）在 offloading 下 normalized throughput 更差（因 CPU 成为更大瓶颈）；
   - 但其 model reload 更快 → preemption 绝对延迟更低。

4. **Interconnect 带宽未达瓶颈**：
   - PCIe Gen4 x16 实际利用率仅 ~10–16 GB/s（理论峰值 31.5 GB/s），主要受限于 PyTorch per-tensor copy 开销；
   - 当前条件下，KV 迁移不是瓶颈，除非未来实现更快的模型加载。

---

### ⚠️ 方法的局限性
- **单请求实验设定**：所有测试均为 single active request，未考虑 **continuous batching** 或高并发场景；
- **worst-case preemption 设计**：每次均完全 unload/reload，未探索内存复用或 staging buffer 优化；
- **未覆盖 MoE 或 multimodal 模型**：实验集中在 dense decoder-only LLMs；
- **仅限 PCIe，未测试 NVLink/CXL**：新型互连可能改变数据移动格局。

---

### 🔮 未来工作方向
1. **扩展至多请求连续批处理场景**，研究 preemption 与 batching 的协同影响；
2. **构建基于实证的调度代价模型**，将 offloading 曲线和 preemption 固定惩罚纳入决策函数；
3. **探索混合策略**：结合 partial offloading 与 selective preemption，实现更高利用率；
4. **开发轻量级 checkpointing 机制**，进一步降低 reload 时间（如 ServerlessLLM 中的 tiered storage）；
5. **在生产环境中验证调度策略收益**，衡量平均响应时间与吞吐的真实权衡。

---

> 📌 **一句话总结**：  
> 本文通过精细实证揭示，在多模型 LLM 推理中，**offloading 的代价是非线性的且模型敏感，而 preemption 的成本几乎是固定不变的、由模型重载主导** —— 这些发现为构建真正高效的 multi-model scheduler 提供了坚实基础。

</details>

---

### 11. [CogScale: Scalable Benchmark for Sequence Processing](https://arxiv.org/abs/2605.19758)

**Authors**: Yannis Bendi-Ouis (Mnemosyne), Romain de Coudenhove (ENS-PSL), Xavier Hinaut (Mnemosyne)  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.19758v1  

#### Abstract
The ability to maintain and manipulate information over time is a fundamental aspect of living beings and Artificial Intelligence. While modern models have achieved remarkable success in tasks like natural language processing, evaluating the capacity of novel architectures to process sequential info...

---

### 12. [From SGD to Muon: Adaptive Optimization via Schatten-p Norms](https://arxiv.org/abs/2605.19781)

**Authors**: Thomas Massena (IRIT, DTIPG - SNCF, UT3), Corentin Friedrich (IRIT), Mathieu Serrurier (IRIT)  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.19781v1  

#### Abstract
Modern optimizers, like Muon, impose matrix-wise geometry constraints on their updates. These matrix-wise constraints can be unified under Linear Minimization Oracle (LMO) theory. However, all current methods impose fixed LMO geometries for the update rules, chosen by-design or empirically, which ar...

---

### 13. [From Simple to Complex: Curriculum-Guided Physics-Informed Neural Networks via Gaussian Mixture Models](https://arxiv.org/abs/2605.19263)

**Authors**: Jianan Yang, Yiran Wang, Shuai Li, Fujun Cao, Xuefei Yan, Junmin Liu  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.19263v1  

#### Abstract
Physics-informed neural networks (PINNs) offer a mesh-free framework for solving partial differential equations (PDEs), yet training often suffers from gradient pathologies, spectral bias, and poor convergence, especially for problems with strong nonlinearity, sharp gradients, or multiscale features...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Simple to Complex: Curriculum-Guided Physics-Informed Neural Networks via Gaussian Mixture Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
**Physics-informed Neural Networks (PINNs)** 在求解偏微分方程（PDEs）时面临以下挑战：
- **梯度病理（gradient pathologies）** 和 **谱偏差（spectral bias）** 导致训练不稳定；
- 对于具有强非线性、尖锐梯度或多尺度特征的 PDE，收敛困难；
- 传统方法对所有配置点（collocation points）进行均匀采样和加权，忽略了不同区域学习难度的空间异质性。

### 🚀 提出的新方法：CGMPINN
作者提出 **Curriculum-Guided Gaussian Mixture Physics-Informed Neural Network (CGMPINN)**，其核心思想是：
- 利用 **Gaussian Mixture Model (GMM)** 对 PDE 残差分布建模，量化空间上的学习难度；
- 设计一个由共享课程参数控制的 **动态课程学习（curriculum learning）机制**，逐步从“简单”区域过渡到“困难”区域；
- 引入 **基于精度的方差调制（precision-based variance modulation）** 抑制早期训练中不可靠的残差簇。

### 🔍 主要创新点
1. **GMM + 动态课程学习的统一框架**  
   - 将 GMM 后验责任（posterior responsibilities）、组件级难度和精度信息融合为样本权重。
   - 实现“双课程”机制：沿**难度维度**（易→难）和**可靠性维度**（低方差→高方差）同步推进。

2. **可微且自适应的损失重加权机制**  
   - 权重随训练进程平滑演化，无需手动设计调度策略。
   - 可与 **ReLoBRaLo** 等自适应损失平衡方法无缝集成。

3. **理论保障**  
   - 证明了课程加权损失与标准 PDE 损失之间的**一致等价性（uniform equivalence）**；
   - 给出了时间变化总损失下梯度范数的**次线性收敛率（sublinear convergence）**；
   - 推导了带显式偏差项的**泛化界（generalization bound）**，解释了加权带来的影响。

### ✅ 相比现有方法的优势
| 方面 | CGMPINN 的优势 |
|------|----------------|
| **准确性** | 显著降低 $L^2$ 和最大绝对误差，最高相对误差下降达 **97.8%** |
| **鲁棒性** | 在多种类型 PDE 上均表现最优，尤其在非线性、对流主导等问题上 |
| **效率** | 不引入显著计算开销，多数情况下 CPU 时间低于或接近基线 |
| **自动化程度** | 减少人工调参需求，通过数据驱动方式自动识别难学区域 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集 / 测试问题
在 **六个典型 PDE 基准问题** 上进行了系统测试，涵盖多种物理特性：

| 问题类型 | 名称 | 特点 |
|--------|------|------|
| 椭圆型 | 1D & 2D Poisson 方程 | 高梯度、多频振荡、陡峭切变 |
| 抛物型 | Heat Equation（热传导） | 时间演化 + 局部尖峰 |
| 双曲型 | Damped Wave Equation（阻尼波动） | 振荡衰减动力学 |
| 对流主导 | Advection-Diffusion Equation | 尖锐行进波前，谱偏差严重 |
| 非线性反应扩散 | Fisher-KPP Equation | 非线性行波解，前沿陡峭 |

> 所有问题均使用已知解析解构造源项和边界条件，便于精确评估误差。

### ⚙️ 实验设置
- **网络架构**：全连接前馈网络（4层×50或80神经元），激活函数为 tanh；
- **优化器**：采用两阶段 **Adam → L-BFGS** 策略以兼顾探索与精细收敛；
- **配置点数量**：根据问题维度设定（如 1D 用 1500~3000 interior points）；
- **GMM 设置**：每 `k_upd` 次迭代更新一次 GMM（通常为 100~500 步），组件数 $K=3\sim5$；
- **课程参数**：$T(k) = \min(1, k / (C_{sat} \cdot k_{max}))$，平滑递增至 1。

### 📊 评估指标
| 指标 | 定义 | 说明 |
|-----|------|------|
| `eLoss` | 最终训练损失 $ \mathcal{L}_{\text{total}}(\theta^*) $ | 衡量优化效果 |
| `e₂` | 绝对 $L^2$ 误差 $\|u - \hat{u}\|_2$ | 整体预测精度 |
| `Relative e₂` | 相对 $L^2$ 误差 $\|u - \hat{u}\|_2 / \|u\|_2$ | 规模无关比较 |
| `e∞` | 最大绝对误差 $\|u - \hat{u}\|_\infty$ | 局部最差表现 |
| `CPU(s)` | 总训练耗时（秒） | 计算效率 |

### 🆚 基线方法对比
与五种代表性 PINN 变体进行公平比较（相同架构、实现平台 PyTorch）：
1. **PINN**：原始标准 PINN [Raissi et al., 2019]
2. **lbPINN**：基于梯度统计的损失平衡方法 [Xiang et al., 2022]
3. **gPINN**：引入 PDE 导数信息的梯度增强型 [Yu et al., 2022]
4. **LNN-PINN**：液体残差门控结构 [Tao et al., 2025]
5. **STAR-PINN**：堆叠自适应残差架构 [Dodge et al., 2025]

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（最佳结果加粗）

| 方法 | 1D Poisson (Rel e₂) | Heat Eq. (Rel e₂) | Fisher-KPP (Rel e₂) | 平均提升 |
|------|--------------------|-------------------|----------------------|----------|
| PINN | 8.15e-3 | 1.28e-3 | 2.14e-3 | — |
| lbPINN | 4.22e+0 ❌ | 2.83e-3 | 1.48e-3 | 下降 |
| gPINN | 1.74e-2 | 2.73e-3 | 2.24e-3 | 下降 |
| LNN-PINN | 4.15e-4 | 8.39e-4 | 6.06e-3 ❌ | 混合 |
| STAR-PINN | 1.14e-3 | 1.43e-3 | 1.43e-3 | 中等 |
| **CGMPINN** | **1.81e-4** | **3.54e-4** | **9.94e-4** | **↑ up to 97.8%** |

> ✅ **CGMPINN 在所有任务上均取得最低的 e₂ 和 e∞ 错误**

#### 典型结果示例（1D Poisson）：
- **相对 $L^2$ 误差下降 97.8%**（从 8.83e-3 → 1.96e-4）
- **最大误差下降 95.9%**（从 9.19e-3 → 3.74e-4）
- **训练时间仅 895.9 秒**，低于大多数基线（如 gPINN 耗时 5711.8 秒）

#### Fisher-KPP 非线性问题：
- CGMPINN 成功捕捉行波速度和波形，而 LNN-PINN 失败（误差高达 3 倍）
- 相对误差从 2.14e-3（PINN）降至 **9.94e-4**（↓53.7%）

---

### 🔪 消融实验结果（Ablation Study）

比较三种变体：
- **GMMPINN**：仅 GMM 加权，无课程调度
- **CLPINN**：仅逐点残差课程学习，无 GMM 结构建模
- **CGMPINN**：完整方法

| 问题 | 最佳方法 | 关键发现 |
|------|---------|----------|
| **1D Poisson** | CGMPINN | GMMPINN 发散（e₂=2.17），表明静态加重困难区域会导致优化崩溃；CLPINN 表现尚可但远不如 CGMPINN |
| **Heat Equation** | CGMPINN | CGMPINN 比 CLPINN 和 GMMPINN 误差小约 4 倍，显示多尺度动态需结构化课程引导 |
| **Advection-Diffusion** | CGMPINN | CLPINN 已优于 GMMPINN（↓51%），但 CGMPINN 再降 24%，体现 GMM 对尖锐前缘建模的价值 |
| **Fisher-KPP** | CGMPINN | CLPINN ↓59% vs GMMPINN，CGMPINN 再 ↓39% vs CLPINN，说明两者互补 |

> ✅ **结论：GMM 提供结构化难度感知，课程调度防止早熟聚焦难区，二者缺一不可**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **课程学习必须是动态且结构感知的**  
   单纯按残差大小排序或静态加权无法稳定训练，甚至导致发散。

2. **GMM 能有效识别异质残差模式**  
   不同组件对应不同物理行为区域（如波前、平缓区、震荡区），提供比单点更强的归纳偏置。

3. **“先易后难 + 先稳后信”双重课程机制至关重要**  
   - 难度维度：避免初始被极端残差主导；
   - 可靠性维度：抑制高方差噪声簇干扰早期学习。

4. **CGMPINN 在精度与效率之间达到优越平衡**  
   不仅准确度最高，且多数情况下训练时间最短或相近，具备实用潜力。

---

### ⚠️ 方法的局限性
1. **当前实验限于低维（1D/2D）问题**  
   高维 PDE 中 GMM 的聚类质量和可扩展性有待验证。

2. **GMM 参数选择依赖经验**  
   如组件数 $K$、更新频率 $k_{upd}$ 尚无自动选择准则，仍需一定调参。

3. **理论分析基于理想化假设**  
   收敛性证明针对全批量 GD，实际使用的 Adam→L-BFGS 混合优化器尚未完全覆盖。

4. **未处理极端稀疏或噪声数据场景**  
   当前测试均为精确标签生成，真实 inverse problems 中的表现需进一步研究。

---

### 🔮 未来工作方向
1. **拓展至高维 PDE 与复杂几何域**  
   探索 scalable clustering 方法（如 mini-batch GMM、流形学习）。

2. **自动化超参数选择机制**  
   基于信息准则（如 BIC/AIC）动态确定 $K$，或通过元学习调整 $k_{upd}$。

3. **结合 domain decomposition 与多分辨率建模**  
   在子域内局部应用 CGMPINN，提升大规模问题处理能力。

4. **推广至 inverse problems 与不确定性量化**  
   融合贝叶斯 PINN 或对抗训练框架，处理含噪观测。

5. **建立更严格的收敛理论**  
   分析 Adam→L-BFGS 在动态重加权下的迭代级收敛性质。

---

> 💡 **总结一句话**：  
> CGMPINN 通过 **GMM 驱动的结构化课程学习**，实现了从“简单可靠区域”向“复杂不确定区域”的渐进式训练，在六大类 PDE 上实现了 **最高达 97.8% 的误差降低**，同时保持高效与稳健，为 PINN 的优化难题提供了新的数据驱动解决方案。

🔗 **代码开源地址**：[https://github.com/Mathematics-Yang/CGMPINN](https://github.com/Mathematics-Yang/CGMPINN)

</details>

---

### 14. [Language models struggle with compartmentalization](https://arxiv.org/abs/2605.19284)

**Authors**: Thomas Vincent Howe, David Wingate  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.19284v1  

#### Abstract
In the training data used by large language models (LLMs), the same latent concept is often presented in multiple distinct ways: the same facts appear in English and Swahili; many functions can be expressed in both Python and Haskell; we can express propositions in both formal and natural language. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Language models struggle with compartmentalization*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文研究了**大型语言模型（LLMs）在面对同一潜在概念的不同表现形式时，无法有效共享统计强度的问题**，即“**compartmentalization**”（隔室化）。这种现象导致模型为每个表现形式学习独立的内部表示，造成以下后果：
- **样本效率低下**（sample inefficiency）：每种形式只能利用自身数据进行学习。
- **容量竞争**（capacity competition）：重复表示消耗有限的模型容量，降低整体性能。

典型场景包括：
- 多语言环境（如英语和斯瓦希里语表达相同事实）
- 不同编程语言实现相同函数（Python vs. Haskell）
- 同一命题以自然语言和形式语言表达

### 提出了什么新方法或新思路
1. **定义并构造了一个可控制的“最坏情况”compartmentalization 模型**  
   - 通过扩大 tokenizer 词汇表大小 $ cV $，将同一数据复制到 $ c $ 个互不重叠的“隔室”中（token ID 偏移 $ jV $），人为制造完全隔离的表现形式。
   - 这种构造使数据在统计上完全相同，理论上应能共享表示，但在实践中暴露了 LLMs 的失败。

2. **提出两种存在性证明（existence proofs）**，表明统一表示是可能的：
   - **初始化复制（Initialization duplication）**：在训练前将基础词嵌入复制到所有隔室，允许 SGD 找到高效解。
   - **后处理参数复制（Post-hoc parameter duplication）**：将已训练好的单隔室模型扩展为多隔室模型，仅需极少量微调即可恢复性能。

3. **引入 operational measure 来量化真实数据中的 representation sharing**  
   - 在多语言、传记/Q&A等任务中使用“tokenizer compartmentalization”作为基线，比较三种设置下的性能差异，从而判断是否发生有效的跨格式知识共享。

4. **系统性探索干预手段的有效性边界**，揭示其存在 **phase transition** 特征。

### 相比现有方法的优势
- **首次系统性地建模并度量 LLMs 中 representation compartmentalization 的代价**。
- 构造方法简洁可控，剥离了自然语言中的子词重叠、句法相似性等干扰因素，直接测试模型对“等价但表面不同”的输入的泛化能力。
- 揭示了当前训练范式（如纯 language modeling objective）在促进 representation 统一方面的局限性，挑战了“scale 就是一切”的假设。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
1. **FineWeb**  
   - 主要预训练语料，约 1310 亿 tokens，用于构造 compartmentalized 设置。
   - 使用自训练的 BPE tokenizer（$ V=16,384 $）。

2. **合成传记与问答对（Synthetic Biographies and Q&A Pairs）**  
   - 构造 $ N=15,000 $ 名虚拟人物，每人有多个属性（出生地、雇主等）。
   - 每个属性生成 10 种表述方式，分别组织成：
     - Wikipedia 风格传记（BIO）
     - 单个问答对（QA）
   - 数据完全一致，仅格式不同，用于测试格式间的容量竞争。

3. **英汉维基百科（EN/ZH Wikipedia）**  
   - 各取 767M tokens，用于多语言 case study。
   - 使用 Qwen3 tokenizer（$ V \sim 151,000 $）。

### 实验设置和评估指标

#### 模型架构
- GPT-style decoder，基于 nanoGPT 实现。
- 修改项：
  - 使用 RoPE（Rotary Position Embedding）
  - 不 tie embedding 和 LM head
- 参数规模从 **1.1M 到 984M（接近 1B）**

#### 训练配置
- AdamW 优化器，学习率 $ 2\times10^{-5} $
- Batch size: 2048 × block size（总 token 数固定为 131,072 / step）
- 训练步数：1M 步（小模型），充分过拟合以观察收敛行为
- 使用 compartment embeddings 来标记每个隔室身份

#### 评估指标
| 指标 | 说明 |
|------|------|
| **Per-compartment validation loss** | 各隔室单独计算的平均交叉熵损失 |
| **Slowdown ratio** | 达到某一验证损失所需迭代次数相对于 $ c=1 $ 基线的倍数 |
| **Final val loss vs. c** | 最终损失随隔室数量增加的变化趋势 |
| **Cross-compartment cosine similarity** | 不同隔室编码同一文本时中间层激活向量的余弦相似度 |
| **Fact recall accuracy** | 在 BIO/Q&A 任务中正确提取属性值的比例 |

#### 基线方法对比
| 基线类型 | 描述 |
|--------|------|
| $ c=1 $ | 单一表示基线，理想上限 |
| $ c>1 $ 默认初始化 | 标准训练流程，测试 compartmentalization 是否自动缓解 |
| Compartmentalized tokenizer baseline | 每种格式使用独立 tokenizer，模拟无共享情况 |
| Shared tokenizer | 标准多语言/多格式联合训练 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）Compartmentalization 导致显著性能下降（图1）
- **随着 $ c $ 增加，sample efficiency 明显恶化**：
  - $ c=8 $ 时，达到相同 val loss 所需时间约为 $ c=1 $ 的 **5×**
- **最终 val loss 随 $ c $ 上升而升高**，显示容量饱和：
  - 从小模型（1.1M）到大模型（~1B），均呈现此趋势
  - 表明冗余表示确实占用了有限 capacity

#### （2）存在统一解，但标准 SGD 找不到（图12）
- **Initialization duplication** 可完全消除 slowdown，性能紧贴 $ c=1 $ 基线
- **Post-hoc duplication**：将训练好的 $ c=1 $ 模型复制为 $ c=8 $，仅需 **2% 的额外训练步数** 即可恢复原始性能
- ⇒ 证明“free lunch”存在，但默认初始化下 SGD 陷入局部最优

#### （3）平行数据（paired "translation" data）效果有限（图3–4）
- 模型能快速学会“翻译任务”（target-half loss < 0.002 nats）
- 但除非满足特定条件，否则**不能减少 compartmentalization**
- 存在 **phase transition**：
  - 在 $ c=8 $ 且 translation ratio ≥ 0.5 时才打破性能 plateau
  - 更小的 $ c $ 或更低比例无效

#### （4）Weight decay 加速 phase transition（图5）
- 引入 weight decay（up to 0.2）可将 phase transition 从 $ c=8 $ 提前至 $ c=6 $
- 显示正则化有助于推动模型寻找更紧凑的统一表示

#### （5）Contrastive alignment 效果更强但仍有局限（图6）
- 使用 InfoNCE 对比损失对齐不同隔室的中间表示
- 结果也呈 phase transition：
  - $ c=2 $：无改善
  - $ c=4 $：缩小约 26% 差距
  - $ c=6,8 $：恢复约 **83–84%** 的性能差距
- 表明更强监督信号可在高 $ c $ 下有效，但仍不足以完全解决低 $ c $ 问题

#### （6）真实场景中的 case studies
##### 多语言训练（图7）
- 英中双语模型性能远低于单语模型（multilingual capacity tax）
- 性能曲线更接近“分隔 tokenizer”设置而非单语上限
- ⇒ 小规模模型早期训练中存在严重 compartmentalization

##### 传记 vs. Q&A 容量竞争（图8）
- **Joint training（共享 tokenizer）**: BIO 准确率仅 **18.8%**
- **Tokenizer-compartmentalized**: BIO 准确率提升至 **51.5%**
- **BIO-only**: 达到 **83.8%**
- ⇒ 显示 QA 格式因更明确而抢占容量，破坏 BIO 学习 —— **destructive capacity competition**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLMs 普遍存在 compartmentalization 现象**：即使面对统计上等价的不同表示，也无法自动共享表示结构。
2. ⚠️ **这不是架构限制，而是训练动态问题**：存在统一解，但标准 SGD + 随机初始化难以找到。
3. 🔁 **干预手段普遍表现出 phase transition**：只有当 $ c $ 足够大或监督信号足够强时，才能打破僵局。
4. 💥 **compartmentalization 导致双重代价**：
   - **Sample inefficiency**：学习速度随 $ c $ 下降
   - **Capacity competition**：最终性能随 $ c $ 劣化，甚至出现 destructive interference
5. 🌐 **现实任务中也观察到类似现象**：多语言、多格式任务中 representation sharing 不充分，尤其在小模型中。

### 方法的局限性
- **主要在小模型上验证**（最大 ~1B），未确认是否随 scale 消失。
- 构造的 compartmentalization 场景极端理想化（完全无重叠），可能高估实际影响。
- 多语言 case study 中模型未达 compute-optimal 训练量，结论外推需谨慎。
- 所有干预都成本高昂（大量 parallel data 或额外 loss），实用性受限。

### 未来工作方向
- 探索更高效的 representation alignment 方法（如轻量级 contrastive loss 设计）
- 研究更大规模下（10B+）compartmentalization 是否缓解
- 开发能主动识别“等价表示”的自监督机制
- 将 compartmentalization 度量应用于更多现实任务（代码/数学/跨模态）
- 分析为何 $ c=2 $ 如此“顽固”，是否存在根本性的优化障碍

> **一句话总结**：  
> 当前 LLMs 在面对同一概念的不同表达形式时，倾向于“各自为政”地学习独立表示，而非统一抽象，这一现象称为 *compartmentalization*，它带来了样本效率和模型容量的双重浪费；尽管存在统一解，但标准训练过程往往找不到它们，除非施加足够强的外部引导（如大量平行数据或对比学习），且这些干预通常只在复杂度较高（$ c \geq 6 $）时才生效。

</details>

---

### 15. [ClinSeekAgent: Automating Multimodal Evidence Seeking for Agentic Clinical Reasoning](https://arxiv.org/abs/2605.20176)

**Authors**: Juncheng Wu, Letian Zhang, Yuhan Wang, Haoqin Tu, Hardy Chen, Zijun Wang, Cihang Xie, Yuyin Zhou  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.20176v1  

#### Abstract
Large language models (LLMs) and agentic systems have shown promise for clinical decision support, but existing works largely assume that evidence has already been curated and handed to the model. Real-world clinical workflows instead require agents to actively seek, iteratively plan, and synthesize...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ClinSeekAgent: Automating Multimodal Evidence Seeking for Agentic Clinical Reasoning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有临床决策支持系统大多依赖于**预先整理好的、静态的证据包**（curated evidence），即在推理前由人工或规则筛选出相关患者信息。这种范式严重偏离真实临床流程，因为实际诊疗中医生需要主动从多种异构来源（如电子健康记录 EHR、医学影像、外部知识库）中动态检索、整合证据。

现有方法存在以下局限：
- 无法处理**稀疏、纵向、跨模态分布的关键证据**。
- 缺乏对**主动证据获取能力**（active evidence acquisition）的建模。
- 多数研究仅限于单模态或受限工具调用场景。

### **提出的新方法与创新思路**
本文提出了 **ClinSeekAgent**，一个用于**动态多模态证据搜寻**的自动化智能体框架，实现从“被动消费证据”到“主动获取证据”的范式转变。

#### **核心创新点：**
- ✅ **统一的多源工具空间（Unified Tool Space）**  
  集成20个工具，覆盖三大证据来源：
  - **EHR Retrieval Tools**（11个）：支持SQL查询、时间范围检索、候选词匹配等。
  - **Web Search Tools**（3个）：通过浏览器访问外部医学知识（如PubMed、UpToDate）。
  - **Medical Imaging Tools**（6个）：支持DICOM处理、CXR分类、报告生成、解剖分割等。
  
- ✅ **推理时（inference-time）与训练时（training-time）双用途设计**
  - 作为**推理管道**：使强LLM能自主规划并执行多步工具调用以完成复杂任务。
  - 作为**训练管道**：利用教师模型（如Claude Opus 4.6）生成高质量搜索轨迹，蒸馏至小型开源模型。

- ✅ **构建 ClinSeek-Bench**  
  新的评测基准，将原有任务重构为配对设置：
  - **Curated Input**：原始设定，直接输入预选上下文。
  - **Automated Evidence-Seeking**：仅提供患者ID和原始数据访问权限，要求模型自行检索证据。

### **相比现有方法的优势**
| 维度 | 传统方法 | ClinSeekAgent |
|------|--------|---------------|
| 证据获取方式 | 被动接收固定上下文 | 主动、迭代式搜索 |
| 数据源多样性 | 单一或有限来源 | 支持EHR + Imaging + Web三模态融合 |
| 工具灵活性 | 固定模板或简单API | 可编程数据库查询（SQL）、自由语义搜索 |
| 应用场景扩展性 | 仅适用于特定任务 | 可泛化至多种临床决策任务 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ClinSeek-Bench**（本文构建）
  - **Text-only Tasks**: 来自 **EHR-Bench** [12]，包含45个子任务，共1,800条样本，涵盖风险预测与决策制定。
  - **Multimodal Tasks**: 整合 **EHRXQA** [13] 和 **MedMod** [14]，基于MIMIC-IV EHR 和 MIMIC-CXR 图像，共989条样本，覆盖六类任务：
    1. CXR finding presence
    2. CXR finding enumeration
    3. CXR temporal change comparison
    4. 24-hour decompensation prediction
    5. In-hospital mortality prediction
    6. Phenotype prediction (CCS groups)

### **实验设置**
- **任务形式**：给定患者ID和时间戳，模型需通过调用工具从原始EHR表、图像文件和网络中收集信息，并输出最终答案。
- **交互协议**：每一步模型可选择调用工具或终止并提交答案，最多允许200轮交互。
- **评估指标**：采用 **sample-wise F1 (%)**，按任务组平均后计算总体得分。

### **基线方法对比**
- **Curated Input Baseline**：保留原基准中的预提取上下文作为输入。
- **Automated Evidence-Seeking (ClinSeekAgent)**：移除上下文，开放工具调用接口。
- **评估模型集合**（共12个）：
  - 闭源模型：Claude Opus 4.6, Claude Sonnet 4.6, GLM-4.7, MiniMax M2.5, Kimi K2.5
  - 开源模型：Qwen3.5-35B-A3B, Gemma-4-26B-A4B-it, Qwen3-VL-235B, gpt-oss-120B 等

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **文本型EHR任务（Text-only EHR Tasks）**
| Model | Curated Input (F1) | ClinSeekAgent (F1) | Δ (Gain) |
|-------|--------------------|---------------------|---------|
| **Claude Opus 4.6** | 60.0 | **63.2** | **+3.2** |
| **MiniMax M2.5** | 43.1 | **47.3** | **+4.2** |
| Claude Sonnet 4.6 | 56.6 | 57.5 | +0.9 |
| Qwen3.5-35B-A3B | 46.8 | 47.0 | +0.2 |

> 🔹 在**风险预测任务**上提升显著，例如：
> - Mortality Hospital: +12.5 pts
> - LengthOfStay: +16.2 pts
> - ED Hospitalization: +12.5 pts

#### **多模态任务（Multimodal Tasks）**
| Model | Curated Input (F1) | ClinSeekAgent (F1) | Δ (Gain) |
|-------|--------------------|---------------------|---------|
| **Claude Opus 4.6** | 47.5 | **62.6** | **+15.1** |
| **Claude Sonnet 4.6** | 48.0 | 54.9 | +6.9 |
| **Qwen3-VL-235B** | 43.9 | 49.8 | +5.9 |
| **Gemma-4-26B-A4B-it** | 38.2 | 44.9 | +6.7 |

> 🔹 所有评估模型中有 **5/6 实现正向增益**，表明主动证据搜寻在多模态任务中更具优势。

#### **训练时验证：ClinSeek-35B-A3B 蒸馏效果**
- 使用 **Claude Opus 4.6** 生成轨迹，微调 **Qwen3.5-35B-A3B** 得到 **ClinSeek-35B-A3B**
- 在 **AgentEHR-Bench** 上表现如下：

| Model | Average F1 |
|-------|------------|
| Qwen3.5-35B-A3B (Base) | 22.1 |
| **ClinSeek-35B-A3B (Ours)** | **34.0** (**+11.9**) |
| Claude Opus 4.6 (Teacher) | 36.0 |
| Claude Sonnet 4.6 | 32.7 |

> ✅ 达到当前**开源模型SOTA水平**，接近闭源大模型性能。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **主动证据搜寻显著优于被动上下文输入**  
   尤其在**风险预测**和**多模态任务**中，ClinSeekAgent 能有效捕捉稀疏但关键的临床信号（如长期趋势、影像特征、外部定义），弥补静态上下文的不足。

2. ✅ **工具组合能力是成功关键**  
   成功案例显示，模型综合使用：
   - SQL 查询提取长时间跨度的生命体征变化；
   - CXR 分类器识别肺部异常；
   - 浏览器搜索恢复标准表型术语（如Harutyunyan-2019列表）；
   → 实现跨模态、跨系统的深度推理。

3. ✅ **可迁移性强：可用于训练更小的开源模型**  
   通过监督微调（SFT）将教师模型的“搜索策略”蒸馏到学生模型，不仅提升了最终准确率，还改变了工具使用行为：
   - 学生模型更多使用 `ehr.run_sql_query`（从2.0% → 12.5%）
   - 表明其学会了将EHR视为**可编程数据库**而非仅靠关键词检索。

4. ⚠️ **在决策制定任务中表现不稳定**  
   - 多个模型在 **Decision Making** 子任务上出现退化（如Kimi K2.5下降11.3点）。
   - 原因可能是：ClinSeekAgent 容易陷入冗余信息收集，而忽略了高阶模式识别（pattern recognition）。

### **局限性**
1. 当前多模态任务仍相对简单，多数可通过少量工具调用解决，未能充分测试长视野、深层次的跨模态推理。
2. 教师轨迹并非总是高效，部分包含冗余或低价值操作，可能污染训练数据。
3. 对弱模型帮助有限，甚至可能导致性能下降，说明该框架高度依赖模型本身的 agentic planning 能力。

### **未来工作方向**
- 构建更具挑战性的**长视野多模态临床推理基准**，强调时间演化、因果推断与治疗反馈循环。
- 引入 **Reinforcement Learning** 优化证据搜寻路径，鼓励简洁、高效的工具使用。
- 探索 **trajectory refinement** 技术（如过滤、压缩、重排序）以提高蒸馏质量。
- 扩展至其他医学影像类型（CT、MRI）及非英语语境下的应用。

---

> 📌 **一句话总结**：  
> **ClinSeekAgent 推动了临床AI从“阅读病历”向“主动查房”的转变，首次实现了端到端的自动化多模态证据搜寻，并证明其既可增强前沿LLM，也可用于训练高性能开源临床智能体。**

</details>

---

### 16. [Quantum-Enhanced Distributed Sensor Fusion: Lower Bounds on Aggregation from Projection Noise to Heisenberg-Limited Byzantine-Tolerant Networks](https://arxiv.org/abs/2605.19327)

**Authors**: Vasanth Iyer, S. S. Iyengar  
**Category**: cs.DC  
**Published**: 2026-05-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.19327v1  

#### Abstract
We derive unified lower bounds on the mean squared error (MSE) of distributed quantum sensor fusion under Byzantine faults and decoherence. Building on the classical Brooks-Iyengar overlap function and its vector extension, the predictive outlier model for virtual sensor tracking, and SPOTLESS spati...

---

### 17. [DAG-Based QoS-Aware Dynamic Task Placement for Networked Multi-Stage Control Pipelines](https://arxiv.org/abs/2605.19887)

**Authors**: Thien Tran, Jonathan Kua, Thuong Hoang, Minh Tran, Yuemin Ding, Jiong Jin  
**Category**: cs.DC  
**Published**: 2026-05-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.19887v1  

#### Abstract
Current Physical AI (PAI) relies heavily on closed-loop visual-servoing pipelines, whose perception and planning stages may become computationally intensive onboard due to complex models embedded on robots. In practice, offloading the perception task to on-site edges statically is inappropriate for ...

---

### 18. [TabQL: In-Context Q-Learning with Tabular Foundation Models](https://arxiv.org/abs/2605.18979)

**Authors**: Qisai Liu, Zhanhong Jiang, Timilehin Ayanlade, Ashutosh Kumar Nirala, Yang Li, Aditya Balu, Soumik Sarkar  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.18979v1  

#### Abstract
We propose Tabular Q-Learning (TabQL), a reinforcement learning framework that replaces the conventional parametric Q-network in Deep Q-Learning (DQN) with a tabular foundation model endowed with in-context learning capabilities. The key idea is to represent Q-values through a sequence-to-sequence f...

---

### 19. [Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR](https://arxiv.org/abs/2605.19282)

**Authors**: Chongyu Fan, Gaowen Liu, Mingyi Hong, Ramana Rao Kompella, Sijia Liu  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.19282v1  

#### Abstract
Muon is a matrix-aware optimizer that leverages Newton-Schulz (NS) iterations to enforce spectral gradient orthogonalization by driving all singular values of the momentum matrix toward 1. While this uniform spectral whitening enhances exploration and outperforms AdamW in LLM pretraining, we show it...

---

### 20. [Hierarchical Contrastive Learning for Multi-Domain Protein-Ligand Binding](https://arxiv.org/abs/2605.19902)

**Authors**: Shuo Zhang, Rongqi Hong, Huifeng Zhang, Jian K. Liu  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.19902v1  

#### Abstract
Predicting protein-ligand binding affinity remains intractable for multi-domain proteins, where inter-domain dynamics govern molecular recognition. Existing geometric deep learning methods typically treat proteins as monolithic static graphs, suffering from rigid-body assumptions and aleatoric noise...

---

### 21. [OpenComputer: Verifiable Software Worlds for Computer-Use Agents](https://arxiv.org/abs/2605.19769)

**Authors**: Jinbiao Wei, Qianran Ma, Yilun Zhao, Xiao Zhou, Kangqi Ni, Guo Gan, Arman Cohan  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.19769v1  

#### Abstract
We present OpenComputer, a verifier-grounded framework for constructing verifiable software worlds for computer-use agents. OpenComputer integrates four components: (1) app-specific state verifiers that expose structured inspection endpoints over real applications, (2) a self-evolving verification l...

---

### 22. [Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization](https://arxiv.org/abs/2605.19782)

**Authors**: Dmitry Redko (Applied AI Institute), Albert Fazlyev (AI Talent Hub, ITMO University), Konstantin Sozykin (Applied AI Institute), Maria Ivanova (YSDA, Applied AI Institute), Evgeny Burnaev (Applied AI Institute), Egor Shvetsov (Applied AI Institute)  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.19782v1  

#### Abstract
LLM discovery and optimization systems are increasingly applied across domains, implementing a common propose-evaluate-revise loop. Such optimization or discovery progresses via context conditioning on received feedback from an environment. However, as modern LLM agents are increasingly complex in t...

---

### 23. [MMoA: An AI-Agent framework with recurrence for Memoried Mixure-of-Agent](https://arxiv.org/abs/2605.19194)

**Authors**: Rui Chu  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.19194v1  

#### Abstract
The Mixture-of-Agents (MoA) framework has shown promise in improving large language model (LLM) performance by aggregating outputs from multiple agents. However, existing MoA systems often rely on static routers that do not fully capture temporal and contextual dependencies across aggregation layers...

---

### 24. [SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](https://arxiv.org/abs/2605.19357)

**Authors**: Yiyang Gu, Junwei Yang, Junyu Luo, Ye Yuan, Bin Feng, Yingce Xia, Shufang Xie, Kaili Liu, Bohan Wu, Qi Shi, Haoran Li, Beier Xiao, Zhiping Xiao, Xiao Luo, Weizhi Zhang, Philip S. Yu, Zequn Liu, Ming Zhang  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.19357v1  

#### Abstract
Large language models (LLMs) are increasingly applied to scientific research, yet existing evaluations often fail to reflect the fine-grained capabilities required in practice. Most benchmarks are manually curated or domain-generic, limiting scalability and alignment with real scientific use cases. ...

---

### 25. [Cross-Paradigm Knowledge Distillation: A Comprehensive Study of Bidirectional Transfer Between Random Forests and Deep Neural Networks for Big Data Applications](https://arxiv.org/abs/2605.19299)

**Authors**: Mahdi Naser Moghadasi  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.19299v1  

#### Abstract
The exponential growth of big data has intensified the need for efficient and interpretable machine learning models that can handle diverse data characteristics while maintaining computational efficiency. Knowledge distillation has primarily focused on neural network-to-neural network transfer, leav...

---

### 26. [Physics-Informed Graph Neural Network Surrogates for Turbulent Nanoparticle Dispersion in Dental Clinical Environments](https://arxiv.org/abs/2605.19589)

**Authors**: Takshak Shende, Viktor Popov  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.19589v1  

#### Abstract
Dental aerosol procedures produce sub-50 micrometre nuclei that can remain airborne for long periods in enclosed clinics, creating pathways for airborne pathogen transmission. Reynolds-Averaged Navier-Stokes (RANS) simulations with Euler-Lagrange particle tracking capture this transport accurately b...

---

### 27. [AQuaUI: Visual Token Reduction for GUI Agents with Adaptive Quadtrees](https://arxiv.org/abs/2605.19260)

**Authors**: Yuankai Li, Tinghui Zhu, Ha Min Son, Zhe Zhao, Xin Liu, Muhao Chen  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.19260v1  

#### Abstract
Large Multimodal Models (LMMs) have recently emerged as promising backbones for GUI-agent models, where high-resolution GUI screenshots are introduced to the prompts at each iteration step. However, these screenshots exhibit highly non-uniform spatial information density: large regions may carry lit...

---

### 28. [What and When to Distill: Selective Hindsight Distillation for Multi-Turn Agents](https://arxiv.org/abs/2605.19447)

**Authors**: Xiaozhe Li, Tianyi Lyu, Yang Li, Yichuan Ma, Peiji Li, Linyang Li, Qipeng Guo, Dahua Lin, Kai Chen  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.19447v1  

#### Abstract
Reinforcement learning can train LLM agents from sparse task rewards, but long-horizon credit assignment remains challenging: a single success-or-failure signal must be distributed across many actions. Existing methods rely on trajectory-level rewards or proxy signals, without fully leveraging per-s...

---

### 29. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://arxiv.org/abs/2605.19662)

**Authors**: Xinpeng Lv, Yunxin Mao, Renzhe Xu, Chunyuan Zheng, Yikai Chen, Haoxuan Li, Jinxuan Yang, Kun Kuang, Yuanlong Chen, Mingyang Geng, Wanrong Huang, Shixuan Liu, Shaowu Yang, Wenjing Yang, Zhouchen Lin, Haotian Wang  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.19662v1  

#### Abstract
Tabular foundation models based on pretrained prior-data fitted networks~(PFNs) have shown strong generalization on diverse tabular tasks, but they are typically designed for \emph{non-strategic} settings where data distributions are independent of deployed classifiers. In many real-world decision s...

---

### 30. [Memory-Augmented Reinforcement Learning Agent for CAD Generation](https://arxiv.org/abs/2605.19748)

**Authors**: Yin Xiaolong, Liu Yu, Shen Jiahang, Lu Xingyu, Ni Jingzhe, Fan Fengxiao, Sang Fan  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.19748v1  

#### Abstract
Automatic generation of computer-aided design (CAD) models is a core technology for enabling intelligence in advanced manufacturing. Existing generation methods based on large language models (LLMs) often fall short when handling complex CAD models characterized by long operation sequences, diverse ...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-05 09:04:05 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [You Only Index Once: Cross-Layer Sparse Attention with Shared Routing](https://arxiv.org/abs/2606.06467)

**Authors**: Yutao Sun, Yanqi Zhang, Li Dong, Jianyong Wang, Furu Wei  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.06467v1  

#### Abstract
Long-context inference in modern LLMs is increasingly constrained by decoding efficiency, especially in reasoning-heavy settings where models generate long intermediate chains of thought. Existing sparse attention methods often face a practical efficiency-quality trade-off. Structured block sparse m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：You Only Index Once: Cross-Layer Sparse Attention with Shared Routing

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代 **Large Language Models (LLMs)** 在处理长上下文推理（long-context inference）时面临严重的效率瓶颈，尤其是在 **reasoning-heavy** 场景中（如 chain-of-thought 推理），模型需要解码长中间推理链并反复关注大规模上下文。这导致：
- **解码效率低下**（decoding-bound）
- **Prefilling 成本高**
- **KV Cache 存储开销大**

现有的稀疏注意力（sparse attention）方法存在 **效率-质量权衡**：
- **Block-sparse attention**：加速明显但质量损失较大（粗粒度近似）
- **Token-sparse attention**：更准确但端到端加速有限，因为每层仍需独立计算 **top-k routing**，该操作在 GPU 上不规则且昂贵。

---

### 🚀 提出的新方法：Cross-Layer Sparse Attention (CLSA)

基于 **YOCO** 架构（KV-sharing 架构），提出 **CLSA**，其核心思想是：
> **不仅共享 KV Cache，还共享 routing index**。

#### 创新点：
1. **Routing Once, Use Everywhere**  
   - 引入一个轻量级的 **query-aware indexer**，在 self-decoder 阶段仅计算一次 token-level 的 top-k 路由索引。
   - 所有后续的 cross-decoder 层 **复用该索引**，避免每层重复计算 top-k。

2. **结合 YOCO 的优势**  
   - 继承 YOCO 的 **KV Cache 共享机制**，减少 prefill 和存储开销。
   - 在此基础上进一步优化 **解码阶段的路由成本**。

3. **多层蒸馏训练（Multi-Layer Distillation）**  
   - 使用所有 decoder 层的注意力分布作为目标，训练 indexer 学习“共识性重要 token”。
   - 分两阶段训练：
     - **Stage 1**：固定主干，warm-up indexer
     - **Stage 2**：联合优化语言建模 + 蒸馏损失（`LLM + λ·KD`, `λ=0.1`）

---

### 🔍 相比现有方法的优势

| 方法 | 加速效果 | 质量保持 | 路由成本 |
|------|----------|----------|----------|
| **Transformer** | 基线 | 最佳 | 无稀疏 |
| **Block-sparse** | 高（结构化） | 较差 | 低但粗粒度 |
| **Token-sparse (DSA)** | 有限 | 较好 | 高（每层重算） |
| **IndexCache** | 中等 | 好 | 中（跨几层复用） |
| **CLSA (本文)** | **极高** | **几乎无损** | **极低（全局复用）** |

> ✅ **CLSA 同时优化了 prefill、KV 存储、解码延迟三大瓶颈，实现全面高效。**

---

## 2. 核心实验方法和设置

### 📚 数据集与任务

| 类型 | 数据集 | 任务描述 |
|------|--------|----------|
| **通用能力** | MMLU, BBH | 多领域知识与复杂推理 |
| **阅读理解** | DROP, ARC-Challenge | 基于文本的离散推理 |
| **常识推理** | HellaSwag, WinoGrande | 多选填空、指代消解 |
| **数学推理** | GSM8K | 小学数学应用题 |
| **代码生成** | HumanEval | Python 函数补全 |
| **长上下文检索** | RULER | 合成的单针/多针检索任务（16K/32K/128K） |
| **语言建模** | Books, ArXiv, StarCoder | 长序列预测（8K–32K） |

---

### ⚙️ 实验设置

- **模型规模**：4B 参数
- **架构对比**：
  - **Transformer**（标准）
  - **YOCO (Dense)**（KV 共享，密集注意力）
  - **YOCO (CLSA)**（KV 共享 + 共享路由稀疏注意力）
- **上下文长度**：最高支持 **128K**
- **硬件平台**：NVIDIA B200 GPU
- **推理框架**：集成至 **vLLM**，测量端到端吞吐量

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **Decode Throughput (tokens/s)** | 解码阶段每秒生成 token 数 |
| **Prefill Throughput (tokens/s)** | 编码输入的吞吐量 |
| **End-to-End Throughput** | 完整生成流程的吞吐量 |
| **Latency Breakdown** | 每层 MLP / Attention / Top-k 延迟分解 |
| **Accuracy / Score** | 各下游任务得分（如 GSM8K 准确率） |
| **Cross-Entropy Loss** | 长上下文建模质量 |
| **Attention Coverage (%)** | 稀疏选择覆盖密集注意力质量的比例 |

---

### 🆚 基线方法对比

- **Transformer**：标准全注意力
- **YOCO (Dense)**：KV 共享但无稀疏
- **DSA**：动态稀疏注意力（每层独立 top-k）
- **IndexCache**：跨层复用索引（每 4 层共享）
- **HySparse**：混合块稀疏 + 密集层

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 下游任务表现（Table 2）

| Model | ARC-C | GSM8K | HumanEval | MMLU |
|-------|-------|-------|-----------|------|
| Transformer | 0.453 | 0.434 | 0.384 | 0.527 |
| YOCO (Dense) | 0.461 | 0.430 | 0.396 | 0.519 |
| **YOCO (CLSA)** | **0.465** | **0.470** | **0.396** | 0.513 |

> 💡 **CLSA 在多个任务上优于或持平基线，尤其在 ARC-C 和 GSM8K 显著提升，表明其能更好聚合关键证据。**

---

#### ✅ 长上下文检索（RULER, Table 3）

| Context | Model | RULER Avg |
|--------|-------|---------|
| 16K | Transformer | 67.0 |
| 16K | YOCO (Dense) | 62.7 |
| 16K | **YOCO (CLSA)** | **62.9** |
| 32K | Transformer | 43.8 |
| 32K | YOCO (Dense) | 52.3 |
| 32K | **YOCO (CLSA)** | **53.1** ✅ |

> ✅ CLSA 在 **32K 更难的 multi-needle 设置下表现最佳**，说明其对长程干扰更具鲁棒性。

---

#### ✅ 推理效率（Figure 3 & Table 12）

| Context | 指标 | Transformer | YOCO (CLSA) | 加速比 |
|--------|------|-------------|--------------|--------|
| **128K** | Decode Throughput | 431.16 | **3276.80** | **7.6×** |
| **128K** | End-to-End Throughput | 62.53 | **1068.06** | **17.1×** |
| **128K** | Prefill Throughput | 1019.06 | 20741.51 | ~20× |

> 🚀 **CLSA 在 128K 上实现高达 7.6× 解码加速 和 17.1× 端到端吞吐提升！**

---

#### ✅ 延迟分解（Figure 6, Table 9）

在 128K 上，每层延迟对比：

| 组件 | Transformer | YOCO (Dense) | YOCO (CLSA) |
|------|-------------|---------------|----------------|
| MLP | 0.17ms | 0.17ms | 0.17ms |
| Attention | 2.11ms | 0.87ms | **0.05ms** |
| Top-k | — | — | **0.08ms (amortized)** |
| **Total** | **2.28ms** | **1.04ms** | **0.31ms** ✅ |

> 💡 **CLSA 的 attention + routing 总延迟仅为 Transformer 的 ~13.6%**，得益于稀疏 + 路由摊销。

---

#### ✅ 注意力稀疏分析（Table 4）

- 使用 **2048 个激活 token**（1:16 稀疏比）：
  - 覆盖约 **80%+ 的 dense attention 质量**
  - **交叉熵损失增加 < 0.006**，几乎无损
  - 在 StarCoder 上甚至略优于 dense

> ✅ 表明 **少量关键 token 即可逼近全注意力性能**，验证稀疏合理性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **共享 routing index 是可行且高效的**  
   - 不同 decoder 层对重要 token 的偏好高度一致，支持 index 复用。
   - 多层蒸馏使 indexer 学会“共识性 salient tokens”。

2. **CLSA 实现了效率与质量的双赢**  
   - 几乎无损模型能力（lossless behavior）
   - 显著加速 prefill、解码、降低 KV 存储
   - 在 128K 上达到 **7.6× decode speedup**, **17.1× overall throughput**

3. **路由成本是稀疏注意力的实际瓶颈**  
   - 即使理论 FLOPs 低，未摊销的 top-k 可能比 dense attention 还慢（图4）
   - **只有 amortized routing 才能真正提速**

4. **细粒度 token selection 优于 block-level**  
   - Block-sparsity 因语义不均难以精准捕获关键 token
   - CLSA 保留 token-level 精度，同时通过共享解决效率问题

---

### ⚠️ 方法的局限性

1. **依赖 YOCO 架构**  
   - 当前 CLSA 基于 cross-decoder 结构，不直接适用于标准 Transformer。

2. **Indexer 需额外训练**  
   - 虽然轻量，但仍需蒸馏训练，不能完全 post-hoc 应用。

3. **极端稀疏下的泛化风险**  
   - 若 indexer 错误过滤关键 token，可能影响复杂推理。

---

### 🔮 未来工作方向

1. **将 CLSA 思想推广到其他架构**（如 RetNet、Mamba）
2. **动态调整激活 token 数**（per-query 或自适应 budget）
3. **探索更高效的 indexer 设计**（如 binary routing、learned hashing）
4. **结合 hybrid attention**（局部 dense + 全局 CLSA）

---

## ✅ 总结

**CLSA 提出了一种“只索引一次，处处复用”的新范式，通过在 YOCO 架构上共享 KV Cache 和 routing index，实现了：**
- **几乎无损的质量**
- **极致的推理效率**
- **全面优化 prefill、KV 存储、解码延迟**

> 🎯 **为长上下文 LLMs 提供了一个兼具高质量与高效率的统一架构解决方案。**

</details>

---

### 2. [SET: Stream-Event-Triggered Scheduling for Efficient CUDA Graph Pipelines](https://arxiv.org/abs/2606.05495)

**Authors**: Zhengxiong Li, Tsung-Wei Huang, Umit Ogras  
**Category**: cs.DC  
**Published**: 2026-06-05  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.05495v1  

#### Abstract
Achieving peak GPU performance remains a significant challenge as the system throughput is constrained by host-device synchronization delays and kernel scheduling overheads, even with aggressive kernel optimizations and batch processing. Furthermore, existing approaches often underutilize hardware r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SET: Stream-Event-Triggered Scheduling for Efficient CUDA Graph Pipelines

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代GPU应用通常以**任务并行流水线**（task-parallel pipelines）的形式运行，涉及大量依赖的kernel执行、内存拷贝和同步操作。尽管已有CUDA Graph等优化手段，但在实际运行中仍存在显著的**kernel gaps**（内核间隔），导致GPU利用率低下。

这些gap主要来源于：
- **Host-Device同步延迟**：CPU端需轮询或阻塞等待GPU完成，造成资源浪费。
- **调度开销**（scheduling overheads）：包括参数更新、stream选择、memcpy入队等操作。
- **批处理模型的局限性**：静态批处理（static batching）虽能摊销启动开销，但引入了**inter-batch gaps**，且难以应对小kernel或高内存压力场景。

### 提出的新方法与创新思路
作者提出 **SET**（**Stream-Event-Triggered Scheduling**），一种面向CUDA Graph的高效运行时调度框架，其核心创新包括：

#### （1）**事件驱动的Host-Device协同调度机制**
- 利用 **event chaining + callback机制** 实现异步资源释放。
- 当一个worker完成任务后，通过CUDA driver触发callback，自动将其返回空闲池（Free Worker Pool），无需主机轮询。
- 调度器在worker可用时立即分派下一个就绪任务，显著降低调度延迟。

#### （2）**基于多流的任务并行编程模型**
- 每个worker拥有独立的CUDA stream、graph executable和设备缓冲区（Mi），实现内存隔离。
- 引入**per-worker job queue**，避免共享队列的竞争。
- 支持**work stealing**：当本地队列为空时，从其他worker的队列“偷取”任务，提升负载均衡。

#### （3）**图级执行流与内存安全设计**
- 每个任务表示为可重用的CUDA Graph Executable，支持运行时参数动态更新。
- 使用**bounded in-flight graphs** 和预分配buffer，确保多个并发任务间的**内存安全性**。
- “被偷取”的任务会进行JIT参数重绑定，指向当前worker的输入/输出buffer。

### 相比现有方法的优势
| 特性 | SET | 传统方法（如Batching / Queue Model） |
|------|-----|-------------------------------|
| 同步方式 | Event-triggered（O(1)） | Polling / Blocking（O(b)） |
| 队列结构 | Per-worker队列 + Work Stealing | 全局共享队列 |
| 内存管理 | 隔离buffer + JIT rebinding | 易发生覆盖风险 |
| 调度粒度 | 细粒度、按stream完成触发 | 批级别同步 |
| 适用场景 | 小kernel、高内存压力均有效 | 对小kernel不友好 |

---

## 2. 核心实验方法和设置

### 使用的数据集（Benchmark Applications）
共使用六个代表性工作负载，涵盖计算密集型与内存密集型任务：

| 工作负载 | 类型 | 描述 |
|--------|------|------|
| **Sobel** | Image Processing | 图像边缘检测，含多个短kernel |
| **GEMM** | Compute-bound | 分块稠密矩阵乘法 |
| **Back Propagation (BP)** | ML Training | 单层训练，合成minibatch生成 |
| **KNN** | Memory-bound | 暴力搜索特征向量分类 |
| **Hotspot** | Memory-bound | 热仿真求解微分方程 |
| **SSSP** | Irregular | Bellman-Ford图遍历，frontier-based松弛 |

> ⚠️ 图4显示：KNN任务平均执行时间仅约10μs，接近锁开销；Hotspot占用高达90% DRAM带宽。

### 实验设置
- **硬件平台**：
  - RTX 3090（Ampere架构）
  - RTX 5090（Blackwell架构）
- **软件环境**：
  - Ubuntu 22.04
  - CUDA 12.8
  - 编译器：g++ (C++20, -O2)

### 评估指标
- **主性能指标**：**Throughput**（吞吐量）
  - 如：GFLOPs（GEMM）、tasks/s（BP）、queries/ms（KNN）等
- **关键分析指标**：
  - **Scheduling Overhead Ratio** = $ t_{\text{schedule}} / T_{\text{measured}} $
    - 包括 intra-batch 和 inter-batch 开销
- **工具支持**：
  - NVIDIA Nsight Systems Profiler：用于可视化kernel gaps和调度行为

### 基线方法对比
| 基线模型 | 简介 |
|---------|------|
| **Synchronous Model** | 单stream串行发射kernel，无任何优化 |
| **Graph Model** | 预实例化单个CUDA Graph，重复回放 |
| **Static Batching Model** | 将多个任务聚合为大图，批量执行 |
| **Queue Model** | 动态平衡负载，使用全局共享队列 |

> 所有baseline均采用最佳实践优化，保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（见Table 1）

| 方法 vs Baseline | 平均加速比（RTX 3090 & 5090） |
|------------------|------------------------------|
| vs Synchronous | **2.18×** |
| vs Graph Model | **2.10×** |
| vs Batching Model | **1.17×** |
| vs Queue Model | **1.39×** |

> ✅ SET在所有工作负载上均优于基线，尤其在极端情况下表现突出。

### 详细性能对比

#### （1）吞吐量（Throughput）结果（Fig. 5）
- **Sobel / GEMM / SSSP / BP**：
  - SET相比Batching平均提升 **1.10×**
  - 在batch size > 32后持续领先Queue Model（最高达1.2×）
- **KNN**（小kernel典型）：
  - Queue Model性能甚至低于Synchronous（因频繁mutex争用）
  - SET凭借**per-worker queue + ready-to-launch graph**，避免锁竞争，实现 **2.94×（Ampere）~ 2.09×（Blackwell）** 加速
- **Hotspot**（高内存压力）：
  - Batching/Queue因过度并发加剧DRAM争用而退化
  - SET是唯一随batch size增长持续受益的方法，最高达 **1.81×** 超越Queue Model

#### （2）调度开销分析（Fig. 6 & Table 2）

| 方法 | RTX 3090 平均调度开销 | RTX 5090 平均调度开销 |
|------|------------------------|------------------------|
| Batching Model | 45.32% | 52.38% |
| Queue Model | 33.36% | 39.85% |
| **SET** | **29.83%** | **34.62%** |

- SET将调度开销降低：
  - 比Batching低 **54.64%**
  - 比Queue Model低 **18.62%**
- 在batch=4096时，Queue Model因全局mutex争用导致开销飙升至54%，而SET维持较低水平。

#### （3）U型开销曲线解释
- 小batch：硬件利用率低 → 调度开销占比高
- 中等batch：利用率上升 → 开销下降
- 大batch：inter-batch同步瓶颈显现 → 开销回升（尤其Batching）
- SET通过event-driven机制平滑该曲线，延缓“过拟合”现象

> 🔍 发现：随着Blackwell架构算力增强（Tcompute↓），调度开销影响更大（Amdahl’s Law效应），凸显SET的重要性。

---

## 4. 关键结论和发现

### 主要发现
1. **Kernel gaps普遍存在**，即使使用CUDA Graph也无法消除，根源在于host-side调度与同步机制。
2. **现有批处理策略无法根本解决问题**：虽然提高吞吐，但引入新的inter-batch gaps，并加剧资源争用。
3. **事件驱动+work stealing是高效调度的关键**：
   - O(1) callback机制实现零轮询资源回收
   - per-worker队列减少锁竞争
   - JIT参数重绑定保障内存安全下的灵活任务迁移
4. **SET具有良好的可扩展性和鲁棒性**：
   - 在不同GPU架构（Ampere vs Blackwell）下均有效
   - 对小kernel、高内存压力、不规则任务均有稳定增益

### 方法的局限性
- **依赖CUDA Graph语义**：不适用于完全动态生成kernel的应用。
- **需要预先知道最大并发度b**：用于配置worker数量和buffer大小。
- **对极短任务（<1μs）可能仍有瓶颈**：尽管已大幅缓解，但仍受限于CUDA本身的最小调度单位。

### 未来工作方向
- 结合**DSL编译器**（如MLIR）进一步融合kernel，减少节点数。
- 探索**device-side dispatcher**与SET结合，在Persistent Kernel中嵌入event-triggered逻辑。
- 扩展至**multi-GPU**场景，利用NCCL + event chaining实现跨设备流水线。
- 自适应调整batch size based on real-time profiling feedback.

---

> ✅ **总结一句话**：  
> SET通过**stream-level event triggering + work stealing + memory-safe graph reuse**，实现了细粒度、低开销、高吞吐的CUDA Graph调度，在多种真实 workload 上实现 **1.15–1.44×** 性能提升，显著降低了传统调度中的“白空间”浪费问题。

</details>

---

### 3. [Learned Subspace Compression for Communication-Efficient Pipeline Parallelism](https://arxiv.org/abs/2606.05484)

**Authors**: Paul Janson, Edouard Oyallon, Eugene Belilovsky  
**Category**: cs.LG  
**Published**: 2026-06-05  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.05484v1  

#### Abstract
Pipeline parallelism enables training of large language models that exceed single-device memory, yet inter-stage activation communication becomes the dominant bottleneck when trained on low-bandwidth networks. Recent work in this area has proposed using fixed orthogonal projections to compress activ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Learned Subspace Compression for Communication-Efficient Pipeline Parallelism**

---

## 1. **论文的主要贡献和创新点**

### **解决的问题**
在低带宽网络环境下进行大规模语言模型的 **pipeline parallelism** 训练时，阶段间（inter-stage）激活值（activations）的通信开销成为主要瓶颈。传统压缩方法（如固定正交投影）虽然能减少通信量，但会导致显著的性能下降，并需要对优化过程进行非标准改造（如约束权重子空间），限制了其适用性。

### **提出的新方法：MAPL**
本文提出了 **Manifold Aware Projection Learning (MAPL)**，一种基于流形约束的学习型压缩框架，用于高效、低通信成本的 pipeline 并行训练。

#### **核心创新点**：
- **可学习的正交投影器（Learnable Orthogonal Projectors）**  
  每个 pipeline 阶段独立学习一个作用于 **Stiefel 流形** 上的正交投影矩阵 $ A_p \in \text{St}(d, r) $，将高维激活压缩到低秩子空间。通过 **SPEL (Spectral Steepest Descent on the Stiefel Manifold)** 算法进行优化，确保投影始终保留在流形上，维持等距性（isometry）。
  
- **分阶段因子化锚嵌入（Per-Stage Factorized Anchor Embeddings）**  
  引入可学习的低秩因子化锚嵌入 $ E_{\text{small}} \in \mathbb{R}^{V\times r} $ 和冻结的随机正交矩阵 $ P_p \in \mathbb{R}^{d\times r} $，以恢复 token 特定信号，避免高秩嵌入占用压缩容量。仅需传输整数 token ID，通信开销极小。

- **结合残差向量量化（Residual Vector Quantization, VQ）**  
  在低秩投影后应用多码本向量量化（MCVQ），进一步提升压缩率。采用 **流式码本同步协议（streaming dictionary update）**，分批更新码本，摊销同步开销。

### **相比现有方法的优势**
| 方面 | Subspace Networks (SSN) | MAPL |
|------|--------------------------|------|
| 投影方式 | 固定全局子空间，粗略更新 | 每阶段自适应学习，细粒度更新 |
| 权重约束 | 必须将权重限制在低秩子空间 | 支持全秩权重，兼容任意 optimizer |
| 锚嵌入 | 静态解耦嵌入（不可训练） | 可学习、每阶段定制 |
| 压缩灵活性 | 有限 | 支持与 VQ 组合，实现更高压缩比 |
| 性能损失 | 显著（高达 14%） | 极小（~1–2%） |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **DCLM-10B**：用于预训练的语言建模语料。
- **验证集**：从 DCLM 中随机保留 5M tokens 作为验证集。
- **下游任务**：HellaSwag、PIQA、ARC-Easy、ARC-Challenge，使用 `lm-evaluation-harness` 进行 zero-shot 评估。

### **模型架构与规模**
基于 **LLaMA** 架构的 decoder-only Transformer，共三种规模：
- **150M** 参数（$d=1024$）
- **500M** 参数（$d=1536$）
- **1B** 参数（$d=2048$）

所有模型上下文长度为 2048，使用 RoPE 位置编码，bf16 精度。

### **实验设置**
- **Pipeline stages**: $ P = 4 $ 或 $ 8 $
- **Global batch size**: 512
- **Token budget**: 每参数 20 tokens（遵循 Chinchilla scaling）
- **Optimizer**:  
  - 2D 权重（Attention/MLP）使用 **Muon**  
  - 1D 参数（Embedding/Bias/LayerNorm）使用 **AdamW**
- **Projector learning rate**: 主学习率的 0.1 倍

### **评估指标**
- **主指标**：验证集 **cross-entropy loss**
- **相对退化率（△%）**：相对于未压缩基线的性能损失
- **下游任务准确率**：四个 NLP 任务的平均 zero-shot 准确率
- **通信成本**：每 token 字节数（Bytes/token），对应压缩比（如 4×, 8×）

### **基线方法**
- **Uncompressed**：无压缩，性能上限
- **SSN [42]**：原始 Subspace Networks
- **SSN (AdamW version)**：使用 AdamW 的 SSN 变体
- **MAPL**：本文方法
- **MAPL+VQ**：MAPL + 向量量化，进一步压缩

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| 模型 | 方法 | 压缩比 | P=4 Loss (△%) | P=8 Loss (△%) |
|------|------|--------|---------------|---------------|
| 150M | Uncompressed | 1× | 3.13 | — |
|      | SSN | 4× | 3.39 (**8.37%**) | 3.40 (**8.63%**) |
|      | **MAPL** | 4× | **3.156 (0.84%)** | **3.165 (1.11%)** |
|      | MAPL+VQ | 8× | 3.165 (1.11%) | 3.170 (1.28%) |
| 500M | Uncompressed | 1× | 2.84 | — |
|      | SSN | 6× | 3.09 (8.92%) | 3.12 (9.90%) |
|      | **MAPL** | 6× | **2.79 (-1.90%)** | **2.84 (0.00%)** |
|      | MAPL+VQ | 12× | 2.92 (2.75%) | 2.88 (1.49%) |
| 1B   | Uncompressed | 1× | 2.68 | — |
|      | SSN | 8× | 3.05 (13.93%) | 3.08 (15.05%) |
|      | **MAPL** | 8× | **2.72 (1.38%)** | **2.73 (2.02%)** |
|      | MAPL+VQ | 16× | 2.76 (3.01%) | 2.74 (2.30%) |

> ✅ **MAPL 在 4×–8× 压缩下，性能损失仅 0.8%–2%，远优于 SSN（最高达 15% 损失）**

### **下游任务表现（Table 2）**
- 在 500M 和 1B 模型上，MAPL 的平均准确率与未压缩基线相差 **≤1.5 个百分点**。
- SSN 在 1B 规模下最大准确率差距达 **~8.8 个百分点**。
- MAPL+VQ 虽然压缩更高（16×），但性能代价较大（平均下降 ~10 pts），适用于极端带宽受限场景。

### **消融实验结果**
#### （1）**是否保持 Stiefel 流形约束（Appendix C）**
| 配置 | Val Loss | △% |
|------|---------|-----|
| 固定正交投影 | 3.1673 | 1.19% |
| 可学习但无约束（Muon） | 3.2101 | 2.56% ❌ 更差 |
| 可学习 + SPEL 更新 | **3.1564** | **0.84%** ✅ 最优 |

> 🔴 **不加流形约束的“可学习”反而更差**，证明流形优化是关键。

#### （2）**锚嵌入设计消融（Appendix G）**
| 策略 | Val Loss | △% |
|------|---------|-----|
| 无锚嵌入 | 3.209 | +2.39% |
| SSN 式静态锚 | 3.212 | +2.48% |
| 全可训练锚 | 3.149 | +0.47% ✅ 但参数开销大 |
| **因子化锚（Ours）** | **3.165** | **+1.11%** ✅ 最佳平衡 |

> ✅ 因子化锚在性能和参数效率之间取得最优权衡。

#### （3）**学习 vs 固定投影的能量保留能力（Figure 4b）**
- 学习型投影（MAPL）在 rank=128 下保留约 **80% 激活能量**
- 固定正交投影仅保留 **~36%**
> ✅ 学习型投影更有效地捕捉任务相关低秩结构。

---

## 4. **关键结论和发现**

### **主要发现**
1. **边界激活具有内在低秩结构**：即使不施加任何约束，pipeline 阶段间的激活在减去 token 嵌入后仍表现出强低秩特性（rank ~250/1024）。
2. **每阶段应学习专属子空间**：不同阶段的最优压缩子空间几何差异显著（主角度达 72°），全局共享子空间（如 SSN）并非最优。
3. **流形约束至关重要**：脱离 Stiefel 流形的投影会破坏等距性，导致性能劣于固定投影。
4. **MAPL 实现帕累托前沿**：在通信成本与模型性能之间达到当前最优权衡，尤其在中高压缩比下优势明显。

### **方法的局限性**
- 当前实验限于 **1B 参数以下** 模型，更大规模（如 10B+）的表现尚待验证。
- 极端压缩（如 MAPL+VQ 的 16×）仍带来显著性能下降，不适合对精度敏感的任务。
- 实际异构网络环境下的稳定性（延迟、丢包）未充分测试。

### **未来工作方向**
- 将 MAPL 扩展至 **更大的模型**（如 10B+）和 **真实分布式网络**。
- 探索 **动态调整投影秩 $ r $** 以适应训练阶段变化。
- 结合 **其他压缩技术**（如 sparsification）构建复合通信优化方案。
- 研究 **跨 pipeline 与 data parallelism 的联合优化策略**。

---

> 📌 **总结一句话**：  
> **MAPL 通过在 Stiefel 流形上联合学习每阶段的正交投影器，实现了高性能、高通信压缩比的 pipeline 并行训练，在 4×–8× 压缩下几乎无损，显著优于 SSN 等固定子空间方法。**

</details>

---

### 4. [When Good Enough Is Optimal: Multiplication-Only Matrix Inversion Approximation for Quantized Gated DeltaNet](https://arxiv.org/abs/2606.06034)

**Authors**: Luoming Zhang, Yuwei Ren, Kui Zhang, Tian Liu, Lingjuan Ge, Denghao Li, Matthew Harper Langston, Yin Huang, Weiliang Will Zeng, Liang Zhang  
**Category**: cs.LG  
**Published**: 2026-06-05  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.06034v1  

#### Abstract
Matrix inversion in chunk-wise parallel linear attention is a major bottleneck for long-context modeling, particularly on NPUs, where forward-substitution-based methods exhibit limited parallelism and poor hardware utilization. We propose a fast, Matrix Multiplication (MatMul)-based algorithm tailor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When Good Enough Is Optimal: Multiplication-Only Matrix Inversion Approximation for Quantized Gated DeltaNet*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在 **chunk-wise parallel linear attention** 架构中，尤其是基于 **Gated DeltaNet (GDN)** 的模型（如 Qwen3.5 和 Kimi），矩阵求逆（matrix inversion）是长上下文建模中的关键瓶颈。传统的 **forward substitution** 方法具有严重的串行依赖性，在 NPU（Neural Processing Units）等硬件上并行效率低、利用率差，导致解码延迟高。

此外，在低精度（如 INT8/INT16）量化推理下，传统方法因中间值动态范围剧烈扩张而面临数值不稳定和精度下降的问题。

---

### 🚀 提出的新方法与新思路

作者提出了一种 **仅使用矩阵乘法（MatMul-only）的近似矩阵求逆算法**，专为 Gated DeltaNet 中出现的严格下三角矩阵设计。其核心思想是：“**足够好即最优**”（Good Enough Is Optimal），即无需精确求逆，低阶近似结合残差校正即可满足实际需求。

该方法主要包括三个关键技术组件：

1. **低阶截断 Neumann 展开（Low-order Truncated Neumann Expansion）**  
   利用 $(I - A)^{-1} = \sum_{n=0}^{k-1} A^n$ 的有限级数性质，但只保留前 $N$ 项（如 $N=3$）作为初始近似，大幅减少计算量。

2. **结构化对角掩码（Structured Diagonal Masking）**  
   基于 Neumann 项的能量集中在主对角线附近的观察，引入一个对角距离相关的掩码 $M(N)$，抑制远离对角线的大值异常，提升数值稳定性，尤其利于量化。

3. **并行残差校正（Parallel Residual Correction）**  
   将传统迭代式残差更新 $T_{m+1} = T_m + T_m R_m$ 转换为可并行化的矩阵幂累积形式 $T \approx T^{(0)} \sum_{s=0}^{S-1} E^s$，消除循环依赖，完全适配 MatMul 流水线。

---

### 🔍 相比现有方法的优势

| 方面 | 本文方法 | 现有方法（如 FLA、Block-wise Inversion） |
|------|----------|-------------------------------|
| **并行性** | 高度并行，全由 MatMul 构成 | Forward substitution 存在严重串行依赖 |
| **硬件友好性** | 完美匹配 NPU 的 MatMul 核心 | 向量操作多，MatMul 单元利用率低 |
| **数值稳定性** | 掩码+低阶截断控制溢出，适合低精度 | 高阶展开易溢出，尤其在 FP16/INT 下 |
| **量化兼容性** | 显式优化 INT8/INT16 表现 | 动态范围大，量化误差显著 |
| **最大支持矩阵尺寸** | 支持 64×64 单块求逆 | 多数限制在 16×16 分块处理 |

> ✅ **核心优势总结**：将原本“不适合硬件加速”的三角求逆转化为“纯 MatMul + 并行化”的计算模式，在保持精度的同时极大提升了效率和硬件利用率。

---

## 2. 核心实验方法和设置

### 📚 数据集

- **WikiText-v2**：用于单核精度分析（100 个样本，序列长度 4K）和端到端困惑度（PPL）评估。
- **LLaVA-COCO**：视觉模块量化校准使用 25 张图像。
- **C4 dataset**：文本模型量化校准使用 1000 个样本。
- **下游任务基准**：
  - **MMLU**：多任务语言理解
  - **CSR (Commonsense Reasoning)**：常识推理（BoolQ, PIQA, HellaSwag）
  - **RealWorldQA**：多模态真实世界问答（来自 xAI）

---

### ⚙️ 实验设置

- **模型系列**：Qwen3-Next-80B-A3B-Instruct、Qwen3.5 系列（0.8B ~ 9B）
- **默认参数**：
  - Chunk Size: 64
  - Neumann Order $N$: 3
  - Residual Steps $S$: 8
- **量化配置**：
  - W4A16：Decoder 权重 INT4，激活 INT16
  - W8A16：视觉解码器权重 INT8
  - 特别测试了 Matrix Inversion 模块使用 **INT8 vs INT16**

---

### 📊 评估指标

| 类型 | 指标 |
|------|------|
| **精度评估** | Signal-to-Noise Ratio (SNR)，Perplexity (PPL)，MMLU / CSR / RealWorldQA 准确率 |
| **性能评估** | 单 kernel cycle 数、decode-layer 开销占比、speedup 倍数 |
| **消融实验** | 不同 $N$, $S$ 组合下的 PPL 与稳定性（是否 NaN） |

---

### 🆚 基线方法对比

- **FLA (Flash Linear Attention)**：当前主流 chunk-wise 并行实现，采用 forward substitution 进行矩阵求逆。
- **Block-wise Inversion**：如 Huawei CSL 的 `gdn-tri-inverse`，将大矩阵分块为 16×16 求逆以缓解数值问题。
- **Full Neumann Expansion**：完整展开所有 $k-1$ 项，理论上精确但不实用。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**性能加速效果**

| Chunk Size | Speedup (vs FLA) | Decode Layer Overhead Reduction |
|------------|------------------|-------------------------------|
| 32         | 5.2×             | 18.1%                         |
| 64         | 4.2×             | 19.4%                         |
| 128        | 4.6×             | 24.6%                         |

> 💡 图 4 显示：我们的方法在各种 chunk size 下均实现 **4–5 倍以上的 kernel 级加速**。

#### （2）**矩阵求逆开销占比显著降低**

| Chunk Size | FLA 方法开销 | 本方法开销 |
|------------|---------------|-------------|
| 32         | 22.3%         | 5.2%        |
| 64         | 25.6%         | 7.6%        |
| 128        | ~31%          | ~9%         |

> ✅ 说明：随着 chunk 增大，原方法求逆开销急剧上升，而本文方法增长缓慢。

---

### 📊 精度表现（End-to-End）

#### 表 2：浮点精度下端到端性能（Qwen3.5 系列）

| Model | Method | PPL | MMLU | CSR | RWQA |
|-------|--------|-----|------|-----|------|
| 0.8B  | FLA    | 15.74 | 50.57 | 51.11 | 62.35 |
| 0.8B  | Ours   | 15.74 | 50.61 | 51.17 | 61.70 |
| ...   | ...    | ... | ... | ... | ... |
| 9B    | Ours   | 8.21 | 70.23 | 67.30 | 74.12 |

> ✅ **结论**：在 FP32/FP16 下，**无任何可察觉的精度损失**，各项指标波动在 ±0.3 内。

---

#### 表 3：W4A16 量化下的性能比较

| Model | Method       | PPL   | MMLU  | CSR   | RWQA  |
|-------|--------------|-------|-------|-------|-------|
| 0.8B  | FLA-W4A16    | 17.54 | 48.26 | 49.03 | 56.08 |
| 0.8B  | Ours-W4A16   | 17.55 | 48.27 | 49.21 | **60.39** |
| 4B    | FLA-W4A16    | 9.66  | 73.34 | 65.05 | 73.33 |
| 4B    | Ours-W4A16   | 9.67  | 73.29 | 65.03 | **72.81** |

> ✅ 在强量化下，**本方法显著优于 FLA**，尤其是在 RealWorldQA 上差距明显（+4~5 pts），表明其更强的鲁棒性和保真能力。

---

#### 表 7：INT8 vs INT16 在矩阵求逆中的表现（CS=128）

| Method      | Matrix Inverse | PPL  | IF-Eval |
|-------------|----------------|------|---------|
| FLA-FP      | FP16           | 8.89 | 0.83    |
| Ours-W4A16  | INT16          | 9.71 | 0.78    |
| Ours-W4A16  | INT8           | 9.71 | 0.77    |

> ✅ **INT8 与 INT16 性能几乎一致**，证明该方法对极低位宽量化也具备良好适应性。

---

### 🔍 消融实验结果

#### 表 4：各模块对 SNR 的影响（FP16 下）

| 方法 | SNR_mean | SNR_worst |
|------|----------|-----------|
| FP64, N=64 | 178.42 | 96.46 |
| → FP16 | 79.53 | -17.94 |
| → N=3 | 42.13 | -51.11 |
| + S=8 | 80.35 | -4.24 |
| + Diagonal Mask | **86.91** | **47.98** |

> ✅ **关键发现**：仅靠低阶截断和残差无法稳定最坏情况；**对角掩码是控制溢出的关键**。

#### 表 5：不同 Neumann 阶数 $N$ 与残差步数 $S$ 的稳定性

| S \ N | N=3     | N=4     | N=5     | N=6     |
|--------|---------|---------|---------|---------|
| S=1   | NaN     | NaN     | NaN     | 469727.0 |
| S=4   | NaN     | 169.30  | 61.31   | 144.95  |
| S=8   | **8.98** | 9.54    | 66.03   | 117.79  |

> ✅ 最佳组合为 **N=3, S=8**：既能避免发散，又能最小化动态范围，适合 INT16 量化。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **精确求逆非必需**：在 Gated DeltaNet 中，$(I-A)^{-1}$ 的能量高度集中于主对角线附近，因此低阶 Neumann 近似已足够。
2. **硬件效率优先**：通过将求逆转换为纯 MatMul 流程，可在 NPU 上获得高达 **5× kernel 加速** 和 **20% 解码头部开销降低**。
3. **量化鲁棒性强**：提出的掩码机制有效压缩动态范围，使 **INT8 矩阵求逆成为可能且不失效**。
4. **“够用即最优”原则成立**：在低精度系统中，进一步提高数学精度带来的收益远小于其引发的量化噪声代价。

---

### ⚠️ 方法的局限性

1. **不适用于任意矩阵**：仅针对 **strictly lower-triangular** 结构设计，通用性受限。
2. **超参敏感**：$N$ 和 $S$ 需根据 chunk size 和精度目标调优，过大仍会导致溢出。
3. **128×128 矩阵需分块处理**：由于 Lemma 3.3 所示的组合爆炸风险，目前仍需采用 block-wise 策略处理更大矩阵（见 Appendix H）。
4. **理论边界较松**：虽然提供了误差界（Lemma 3.2），但在极端分布下仍可能出现 outlier。

---

### 🔮 未来工作方向

1. **自适应截断策略**：根据输入动态调整 $N$ 和 $S$，实现精度-效率联合优化。
2. **扩展至其他结构矩阵**：探索对 DPLR（Diagonal-Plus-Low-Rank）或其他递归结构的类似加速。
3. **编译器级融合优化**：将整个近似求逆流程编译为单一 Triton kernel，进一步减少调度开销。
4. **训练时感知的反向传播修正**：研究如何在训练阶段补偿近似误差，实现端到端稳定量化训练。

---

## ✅ 总结一句话

> 本文提出一种 **结构感知、纯 MatMul、量化友好的矩阵求逆近似方法**，在 **不牺牲精度的前提下实现了高达 5× 的 kernel 加速和 20% 的解码开销降低**，为 **高效部署长上下文 LLM 到边缘设备** 提供了切实可行的技术路径。

</details>

---

### 5. [YouZhi: Towards High-Concurrency Financial LLMs via Adaptive GQA-to-MLA Transition](https://arxiv.org/abs/2606.05868)

**Authors**: PSBC LLM Team,  Huawei LLM Team, Ruihan Long, Junjie Wu, Tianan Zhang, Duo Zhang, Yaozong Wu, Jinbin Fu, Chang Liu, Zhentao Tang, Wenshuang Yang, Xin Wang, Zhihao Song, Ning Huang, Wenjing Xu, Shuai Zong, Shupei Sun, Sen Wang, Jing Hu, Bin Wang, Xinyu Wang, Junkui Ju, Zequn Ding, Jie Ran, Man Luo, Shixiong Kai, Linkai Hou, Kaichao Liang, Hu Zhao, Yang Zhao, Shucheng Lin, Wei Yu, Chenghan Jiang, Jingjing Ding, Jiahui Zhang, Tian Jin, Yuhang Zhang, Dong Guo, Wei Sun, Jun Xie, Jianwei Li, Lei Cao, Pei Li, Jiabin Li, Jia Yuan, Rui Yuan, Jing Zhu, Mingxuan Yuan, Zhangcheng Lv, Xin Jiang, Xiuhong Fei, Xiaozhe Ren, Yulong Li, Zhipeng Zhang, Hang Wang, Zhaohui Xu, Rui Zhao, Yibo He, Xinzhuang Niu  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.05868v1  

#### Abstract
Large language models (LLMs) drive significant financial innovations, yet their high-concurrency deployment is severely bottlenecked by KV cache memory overhead, which inflates infrastructure costs and throttles scalability. To address this, we propose YouZhi-LLM, a highly efficient financial LLM em...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：YouZhi: Towards High-Concurrency Financial LLMs via Adaptive GQA-to-MLA Transition

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在金融领域的应用面临**高并发部署瓶颈**，其核心在于标准 Transformer 架构中 **KV Cache 内存开销过大**，导致推理延迟高、吞吐量低，难以满足实时金融服务（如移动银行）对低延迟、高并发和高任务完成率的要求。同时，现有金融 LLM 多依赖领域预训练或微调，忽视了架构层面的效率优化。

### 提出了什么新方法或新思路
本文提出 **YouZhi-LLM**，一个面向高并发金融场景的高效大模型系统，其核心创新包括：

- **层自适应 GQA-to-MLA 转换框架（Layer-Adaptive GQA2MLA Transition）**  
  在将 GQA（Grouped-Query Attention）模型转换为 MLA（Multi-Head Latent Attention）架构时，**动态地为每一层分配最优的 FreqFold size**，而非采用统一大小。该方法基于发现：浅层适合较大的 FreqFold（利于主成分集中），而中深层需较小甚至为1的 FreqFold（以保留 RoPE 频率精度）。

- **两阶段后训练流水线（Two-stage Post-training Pipeline）**  
  1. **广义知识蒸馏（Generalized Knowledge Distillation, GKD）**：恢复因结构转换导致的语言建模能力退化；
  2. **金融领域监督微调（Financial Domain-Specific SFT）**：注入金融专业知识，并增强合规响应、指令遵循等能力。

- **分层数据构建策略（Stratified Data Construction）**  
  包括：
  - 分层压缩（Stratified Compression）：按类别、质量、难度分层过滤，保障多样性与质量；
  - 缺失领域增强（Domain Augmentation）：通过元分解生成多样化 persona 和任务；
  - 拒绝回答数据构造（Refusal Data）：基于逻辑树剪枝生成“不可回答”样本，减少幻觉；
  - 指令跟随数据构造（Instruction-Following Data）：模板化生成支持 JSON、LaTeX 等格式输出的数据。

- **端到端 Ascend 生态集成**  
  在华为 Ascend NPU 上实现 MLA 算子优化，并通过 vLLM-Ascend 推理框架实现高效部署。

### 相比现有方法的优势
| 方面 | 现有方法（如 BloombergGPT, FinGPT） | YouZhi |
|------|----------------------------------------|--------|
| 架构效率 | 使用 GQA/MHA，KV Cache 开销大 | 使用 MLA，显著压缩 KV Cache |
| 结构转换 | 统一 FreqFold（如 TransMLA） | 层自适应 FreqFold，更优权衡压缩与性能 |
| 后训练 | 单一 SFT 或 LoRA 微调 | 双阶段 GKD + SFT，兼顾通用性与专业性 |
| 数据构建 | 通用金融语料 | 系统化、分层、多样化的高质量数据工程 |
| 部署效率 | 通用推理框架 | Ascend + vLLM-Ascend 深度优化 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

#### 语言建模评估
- **WikiText-2**：用于评估 perplexity，衡量基础语言建模能力。

#### 下游任务评估
- **通用基准（General Benchmarks）**：
  - C-Eval、IFEval、MATH-500、LiveCodeBench (LCB)、HellaSwag (H-Swag)、SST-5、CrossNER
- **金融领域基准（Financial Benchmarks）**：
  - CFLUE-K/A、FinanceIQ、FinEval、OpenFinData、FPB

#### 实际应用场景
- **移动银行业务私有数据集**：涵盖意图识别（L1/L2 控制器）、槽位填充（Slot Filling）等真实对话任务。

### 实验设置和评估指标

| 设置项 | 描述 |
|-------|------|
| **硬件平台** | Huawei Ascend A3 集群 |
| **推理框架** | vLLM-Ascend（Ascend 优化版 vLLM） |
| **输入/输出长度** | 每个请求处理 1k 输入 token 并生成 1k 输出 token |
| **主要评估指标** | 
| - Perplexity (PPL) | 衡量语言建模能力退化程度 |
| - Average Score | 多个 benchmark 的平均准确率 |
| - Max Concurrent Requests | 最大并发请求数（反映服务容量） |
| - Throughput (tokens/s) | 吞吐量 |
| - KV Cache Size | KV 缓存元素数量（内存占用） |

### 基线方法对比
- **结构转换基线**：
  - **TransMLA**：统一 FreqFold 的 GQA-to-MLA 转换方法
- **模型基线**：
  - **Base Models**：OpenPangu-7B、Qwen2.5-14B-Instruct
  - **金融专用模型**：YiZhao-12B-Chat、DianJin-R1-7B、Qwen3-8B、Qwen3.5-9B
- **开源模型**：Llama3-8B、MiMo-7B 等

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 语言建模能力（WikiText-2 上的 PPL）
| Model | Orig PPL | TransMLA PPL | **YouZhi (L-Adap.) PPL** | △PPL Reduction vs. TransMLA |
|-------|----------|--------------|---------------------------|-------------------------------|
| OpenPangu-7B | 14.1 | 50.3 | **37.6** | **-35%** |
| Llama3-8B | 6.1 | 25.8 | **12.9** | **-65%** |
| Qwen2.5-7B | 6.8 | 8.4 | **8.3** | -6% |
| MiMo-7B-SFT | 16.2 | 22.1 | **20.4** | **-29%** |

> 📌 **结论**：YouZhi 的层自适应算法显著降低 perplexity 退化，尤其在大模型上效果更明显（最高达 65% 改善）。

#### ✅ 下游任务表现（平均得分）

| 类别 | 模型 | 平均得分 |
|------|------|---------|
| **通用基准** | YouZhi-7B | **66.8** |
| | YouZhi-14B | **68.0** |
| | Qwen2.5-14B-Ins. (base) | 65.8 |
| **金融基准** | YouZhi-7B | **74.1** |
| | YouZhi-14B | **78.8** |
| | YiZhao-12B-Chat | 69.7 |
| | DianJin-R1-7B | 69.8 |
| | Qwen3.5-9B | 73.1 |

> 📌 **结论**：YouZhi 模型不仅超越其 base model，在金融任务上也全面优于当前主流金融 LLM。

#### ✅ 高并发服务能力（vLLM-Ascend 测试）

| Model | Max Concurrent | Improvement | Throughput | KV Cache Size |
|-------|----------------|-------------|------------|---------------|
| OpenPangu-7B | 95 | — | 3,325 tokens/s | 2048 elements |
| **YouZhi-7B** | **256** | **×2.69** | **5,865 tokens/s (×1.76)** | **576 elements (-72%)** |
| Qwen2.5-14B-Ins. | 55 | — | 1,740 tokens/s | 2048 elements |
| **YouZhi-14B** | **134** | **×2.43** | **2,990 tokens/s (×1.71)** | **576 elements (-72%)** |

> 📌 **结论**：KV Cache 减少 **72%**，最大并发提升 **2.43~2.69 倍**，实现真正的高吞吐部署。

#### ✅ 实际金融应用表现（移动银行）

| 任务 | OpenPangu-7B | YouZhi-7B |
|------|---------------|-----------|
| 意图识别（平均） | 97.8% | **97.7%** |
| 槽位填充（平均） | 99.0% | **100.0% / 98.1%** |

> 📌 **结论**：在真实业务中保持高性能，具备可靠的任务导向对话能力。

### 消融实验结果
- **层自适应 vs. 统一转换（图9）**：随着逐层转换至 MLA，YouZhi 始终保持更低的 perplexity，且差距在浅层最为显著。
- **分层数据压缩 vs. 全局熵压缩**：前者在类别覆盖、难度平衡和质量优先方面表现更好，避免跨类偏置。
- **拒绝数据构造有效性**：引入 TreeCut + LLM 两阶段方法后，模型对不完整前提的拒绝率从 <10% 提升至 >85%，大幅减少幻觉。

---

## 4. 关键结论和发现

### 主要发现
1. **Transformer 各层对 GQA-to-MLA 转换敏感性不同**：浅层可承受更大压缩（FreqFold），中深层需精细保留 RoPE 信息。
2. **层自适应 FreqFold 显著优于统一设置**：可在几乎不牺牲语言建模能力的前提下实现 KV Cache 高效压缩。
3. **结构转换必须配合系统性后训练**：仅靠 SFT 不足以恢复能力损失，GKD + 分域 SFT 是关键。
4. **高质量、多样化的训练数据是专业化核心**：特别是拒绝数据、指令格式数据、缺失领域数据的构造至关重要。
5. **MLA + Ascend + vLLM 可实现极致高并发部署**：KV Cache 减少 72%，并发能力提升超 2.6 倍。

### 方法的局限性
- **依赖预训练 GQA 模型**：无法从零训练 MLA 模型，仍受限于源模型的知识边界。
- **FreqFold 参数搜索成本**：尽管已简化为逐层枚举，但在非常深的模型上仍有计算负担。
- **金融领域泛化性待验证**：目前测试集中文金融为主，跨语言或多国金融法规适配尚未验证。
- **动态 FreqFold 未在线调整**：当前为静态配置，未能根据输入动态选择最优结构。

### 未来工作方向
- 探索 **MLA 原生预训练**，摆脱对 GQA checkpoint 的依赖；
- 设计 **输入感知的动态结构切换机制**，实现运行时自适应 MLA 配置；
- 扩展至 **多模态金融助手**，融合文本、表格、图表理解；
- 构建 **金融合规自动审计模块**，结合形式化验证与 LLM 自我检查；
- 推动 **Ascend 生态下 MLA 成为主流推理架构**，形成行业标准。

--- 

> 🔚 **总结一句话**：YouZhi 通过“**层自适应 GQA-to-MLA 转换 + 系统性后训练 + Ascend 高效部署**”三位一体方案，在保证甚至提升金融任务性能的同时，实现了高达 **2.69× 的并发能力提升**，为金融级高并发 LLM 部署树立了新范式。

</details>

---

### 6. [AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2606.04484)

**Authors**: Qingxu Fu, Boyin Liu, Shuchang Tao, Zhaoyang Liu, Bolin Ding  
**Category**: cs.AI  
**Published**: 2026-06-05  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.04484v1  

#### Abstract
We present AgentJet, a distributed swarm training framework for large language model (LLM) agent reinforcement learning. Unlike centralized frameworks that tightly couple agent rollouts with model optimization, AgentJet adopts a decoupled multi-node architecture in which swarm server nodes host trai...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **agentic RL**（基于大语言模型的智能体强化学习）训练框架存在以下关键问题：
- **运行时脆弱性（Runtime Fragility）**：训练与推理环境紧耦合，外部环境（如浏览器、沙箱、API）故障会导致整个训练中断。
- **调试困难（Debugging Friction）**：修改 agent 代码或奖励函数需重启整个训练流程，迭代周期长达 5–10 分钟。
- **多模型支持受限**：主流框架仅支持单个策略模型，难以实现异构多智能体系统（如不同规模 LLM 协作）。
- **冗余上下文开销**：长轮次交互中重复的系统提示、工具定义等导致训练计算浪费。
- **任务隔离不足**：多任务混合训练因依赖冲突而难以实现。

### 提出的新方法与架构
提出 **AgentJet** —— 一种基于 **swarm 架构** 的分布式训练框架，其核心创新是将 **agent 执行平面** 与 **模型优化平面** 完全解耦。

#### 主要创新点：
- ✅ **Swarm 架构（客户端-服务器范式）**
  - **Swarm Server（优化节点）**：部署在 GPU 集群，负责模型存储、梯度更新、vLLM 推理服务。
  - **Swarm Client（采样节点）**：可在任意设备（CPU 或 GPU）运行，执行 agent 工作流、收集轨迹、计算奖励。
  - 支持 **任意数量的客户端与服务器** 动态加入/退出，形成灵活的训练网络。

- ✅ **异构多模型 RL（Heterogeneous Multi-Model RL）**
  - 多个 Swarm Server 可独立训练不同 LLM（如 Qwen3-7B 和 Qwen3-14B），实现非共享参数的多智能体协作或对抗训练。

- ✅ **多任务鸡尾酒训练（Multi-Task Cocktail Training）**
  - 不同任务（如 AppWorld 编码、AIME 数学）可在隔离的客户端运行，服务器统一接收混合轨迹进行联合优化。

- ✅ **容错与热插拔调试（Fault-Tolerant & Hot-Swap Debugging）**
  - 客户端崩溃不影响服务器训练状态；可动态替换客户端以更新 agent 逻辑，无需重启训练。

- ✅ **时间线合并（Timeline Merging）**
  - 自动识别并合并多轮交互中的冗余上下文，减少重复 token 计算，提升训练效率 **1.5–10×**。

- ✅ **自动化研究系统（Automated Research System）**
  - 基于 **A3R（Alpha Auto Research）模块**，实现从自然语言研究目标到多日实验的全自动执行，包括超参搜索、故障恢复、结果分析。

### 相比现有方法的优势
| 特性 | AgentJet | 其他框架（如 OpenRLHF, veRL, Forge） |
|------|--------|----------------------------------|
| 架构耦合 | 完全解耦（client-server） | 紧耦合（rollout 与 training 同进程） |
| 多模型支持 | ✅ 支持多个独立模型 | ❌ 通常仅支持单一模型 |
| 调试灵活性 | ✅ 热插拔，秒级迭代 | ❌ 需重启，分钟级延迟 |
| 容错能力 | ✅ 客户端失败不影响训练 | ❌ 故障常导致训练中断 |
| 上下文效率 | ✅ Timeline Merging 加速 | ❌ 冗余上下文未优化 |
| 框架兼容性 | ✅ 支持 LangChain, AgentScope, Raw HTTP 等 | ⚠️ 通常需特定集成 |

---

## 2. 核心实验方法和设置

### 数据集
- **Werewolves RPG**：社交推理游戏，用于测试多智能体协作与欺骗策略。
- **AppWorld**：交互式编码基准，模拟真实数字任务（如邮件管理、音乐播放）。
- **AIME**：数学推理任务，评估符号推理与工具调用能力。
- **DAPO-Math-17k**：大规模数学训练数据集。
- **自建“谁是卧底”（Who is the Spy）游戏**：用于 vibe training 演示。

### 实验设置
- **模型**：Qwen3-8B, Qwen3-14B, Qwen3-7B, Qwen3-235B（静态对手）。
- **算法**：GRPO, PPO, DAPO。
- **硬件**：8-GPU 节点（FSDP 分布式训练）。
- **训练模式**：
  - **共享参数训练（Shared-Parameter）**：同一阵营智能体共享一个模型。
  - **非共享参数训练（Non-Shared Parameter）**：每个角色/阵营拥有独立模型。
  - **鸡尾酒训练（Cocktail Training）**：AppWorld 与 AIME 任务混合训练。

### 评估指标
- **Success Rate / Win Rate**：任务成功或游戏获胜的比例。
- **Pass@1 / Pass@2**：数学任务首次/前两次尝试通过率。
- **Mean Reward**：每步平均奖励。
- **Actor-Update Wall Time**：每次策略更新的实际耗时（衡量训练效率）。
- **Policy Entropy**：策略熵，衡量探索程度。

### 基线方法对比
- **Separate Single-Task Training**：AppWorld 与 AIME 各自独立训练。
- **Shared-Parameter MARL**：传统多智能体 RL 设置。
- **No Timeline Merging**：关闭上下文合并的对照组。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）Werewolves 游戏训练效果（共享参数）
| 实验 | 可训练角色 | 初始胜率 | 最终胜率 | 提升 |
|------|------------|----------|----------|------|
| Exp 1 | WW (7B) | 23.0% | 47.2% | +24.2% |
| Exp 2 | WW (14B) | 40.9% | 64.7% | +23.8% |
| Exp 3 | Seer (14B) | 38.5% | 46.5% | +8.0% |
| Exp 7 | Villager 团队 (14B) | 23.9% | 41.6% | +17.7% |

> 📌 **发现**：狼人阵营更易训练；联合训练多个角色可有效提升团队表现。

#### （2）非共享参数训练优势
| 实验 | 模型配置 | 初始胜率 | 最终胜率 |
|------|----------|----------|----------|
| Shared (Table 1 Exp 2) | 单一 14B 模型 | 40.9% | 64.7% |
| Non-Shared (Table 2 Exp 3) | 三个独立 14B-LoRA 模型 | 40.8% | **66.5%** |

> 📌 **发现**：独立参数训练带来 **+1.8%** 提升，因行为多样性增强欺骗性，避免被村民识别。

#### （3）鸡尾酒训练 vs 独立训练
| 任务 | 训练方式 | 平均 Reward | 最后 20 步 Reward |
|------|----------|-------------|------------------|
| AIME | Cocktail | 0.72 | 0.75 |
| AIME | Separate | 0.73 | **0.80** |
| AppWorld | Cocktail | 0.58 | 0.58 |
| AppWorld | Separate | **0.68** | **0.68** |

> 📌 **结论**：鸡尾酒训练在 AppWorld 上有约 10 点性能损失，但在 AIME 上接近独立训练。**优势在于低成本获得通用模型**，适合多技能部署场景。

#### （4）Timeline Merging 效率提升
| 设置 | 平均 Actor-Update 时间 | 速度提升 |
|------|------------------------|---------|
| 无合并 | 2160 ± 171 秒 | 1× |
| 有合并 | **346 ± 13 秒** | **6.25×** |

> 📌 **关键**：训练质量（reward 曲线）几乎不变，但训练速度大幅提升。

#### （5）框架无关性验证（Framework-Agnostic）
使用四种不同 agent 框架（OpenAI SDK, LangChain, AgentScope, Raw HTTP）训练相同任务，最终评估 reward 差距 < 0.04，证明 **AgentJet 对 agent 框架透明**。

---

## 4. 关键结论和发现

### 主要发现
1. **Swarm 架构显著提升训练鲁棒性与灵活性**：通过完全解耦，实现了容错、热插拔、多任务隔离等关键能力。
2. **非共享参数训练在社交博弈中更具优势**：独立模型带来行为多样性，提升欺骗成功率。
3. **Timeline Merging 是高效长轮次训练的关键**：可实现 **6.25× 训练加速**，且不牺牲性能。
4. **鸡尾酒训练是构建多技能通用 agent 的低成本路径**：虽在个别任务上略逊于专用模型，但节省了 N 倍训练成本。
5. **自动化研究系统（A3R）可自主完成复杂 RL 研究**：从超参搜索到故障恢复，全程无需人工干预。

### 局限性
- **鸡尾酒训练存在性能折衷**：工具密集型任务（如 AppWorld）在混合训练中可能被稀释。
- **Timeline Merging 的一致性权衡**：文本级匹配速度快但可能破坏 train-inference 一致性；token 级匹配更严格但合并率低。
- **对客户端实现有一定要求**：需正确实现 episode 注册、奖励提交等接口。

### 未来工作方向
- 支持更复杂的 **multi-agent communication protocols**。
- 引入 **adaptive cocktail mixing**，根据任务难度动态调整采样比例。
- 扩展 **A3R** 至跨组织、跨集群的联邦式自动研究。
- 探索 **timeline merging 在其他长序列任务**（如视频生成、语音合成）中的应用。

> 🔗 **开源地址**：[https://github.com/modelscope/AgentJet](https://github.com/modelscope/AgentJet)

</details>

---

### 7. [BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization](https://arxiv.org/abs/2606.04807)

**Authors**: Saket Reddy, Ke Yang, ChengXiang Zhai  
**Category**: cs.AI  
**Published**: 2026-06-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.04807v1  

#### Abstract
Mitigating social bias in Large Language Models (LLMs) presents a distinct alignment challenge: unlike verifiable tasks, bias lacks a single ground truth, creating a high-variance, subjective reward landscape. Previous preference-based fine-tuning methods have major trade-offs: Direct Preference Opt...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）在预训练过程中会继承来自大规模文本语料的社会偏见（social bias），这些偏见缺乏单一的“正确答案”，导致其奖励空间具有**高方差（high-variance）和主观性**。现有的偏好微调方法在处理此类任务时面临显著权衡：
- **DPO（Direct Preference Optimization）**：依赖静态的偏好对进行离线训练，缺乏探索能力，泛化性差。
- **PPO（Proximal Policy Optimization）**：虽支持在线探索，但依赖独立的critic模型估计价值函数，在噪声大、主观性强的偏见场景下容易产生不稳定的advantage估计，导致训练不稳定。

### **提出的新方法与新思路**
本文提出了 **BiasGRPO**，一个基于 **Group Relative Policy Optimization (GRPO)** 的框架，用于稳定社会偏见缓解过程中的对齐训练。

- **核心机制**：摒弃PPO中的critic模型，转而为每个提示（prompt）生成一组（group）补全（completions），并以该组内补全的平均奖励作为相对基准来计算advantage。
  $$
  A_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
  $$
  其中 $ r_i $ 是第 $ i $ 个补全的奖励，$ r $ 是整组补全的奖励集合。

- **核心思想**：通过**组内归一化（group-relative normalization）** 提供更清晰、更稳定的训练信号，即使所有生成结果都带有偏见，也能识别出“相对更优”的输出。

### **相比现有方法的优势**
| 方法 | 探索能力 | 训练稳定性 | 是否需要Critic |
|------|----------|------------|----------------|
| DPO  | ❌ 离线训练 | ✅ 高 | ❌ 不需要 |
| PPO  | ✅ 在线探索 | ❌ 易受Critic影响 | ✅ 需要 |
| **BiasGRPO** | ✅ 在线探索 | ✅ 高（无Critic） | ❌ 不需要 |

- **兼具PPO的泛化能力和DPO的稳定性**。
- 特别适合**高方差、主观性强的任务**（如偏见缓解），而非仅限于可验证任务（如数学、代码）。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
构建了一个包含 **20,999 条样本** 的综合数据集，来源如下：
- **BiasDPO**（10,000条）：原始偏见探测问题 + **合成扩展8,855条**，覆盖11个领域（种族、性别、宗教、年龄、残疾、国籍等及交叉领域）。
- **Civil Comments**（10,000条）：社交媒体评论，按毒性分数分层采样，用于减少由中性或轻微偏见提示引发的毒性响应。
- **UnQover**（999条）：模糊情境下的偏见探测问题，答案“无法确定”，用于训练模型拒绝偏见推理。

> 所有合成数据均通过 **GPT-4o、Gemini 2.0 Flash、Claude 4 Sonnet** 多模型协同生成，并经 **Vendi Score** 验证其语义多样性（达到人类基准的72.79%）。

### **实验设置与评估指标**
- **基础模型**：Microsoft Phi-2（2.7B参数），未经过RLHF或任何偏见缓解微调，确保“从零开始”测试。
- **训练方式**：
  - 所有方法训练3个epoch，初始学习率 $10^{-6}$，线性衰减。
  - GRPO组大小默认为4。
- **评估基准**：
  - **BOLD**（↓）：衡量表征性伤害（representational harm）。
  - **RealToxicityPrompts (RTP)**（↓）：衡量显性敌意（overt hostility）。
  - **BBQ**（↑）：衡量模糊情境下的隐性刻板印象（implicit stereotyping），仅评估“无法确定”类问题。
  - **TruthfulQA**（↑）：评估是否发生**知识退化（knowledge degradation）** 或灾难性遗忘。

### **基线方法对比**
- **DPO**（使用IPO变体，对偏见最有效）
- **PPO**
- **GRPO**（即本文提出的BiasGRPO）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见Table 2）**
| Benchmark | Base | DPO | PPO | **GRPO** |
|----------|------|-----|-----|-----------|
| **BOLD (↓)** | 0.0293 | 0.0222 | 0.0268 | **0.0140** |
| **RTP (↓)** | 0.0282 | 0.0234 | 0.0262 | **0.0198** |
| **BBQ (↑)** | 0.2750 | 0.2823 | 0.2996 | **0.3123** |
| **TruthfulQA (↑)** | 0.3843 | 0.3941 | 0.3929 | **0.3941** |

> ✅ **GRPO在所有偏见相关指标上均取得最优表现**  
> ✅ **同时保持甚至略微提升TruthfulQA性能，证明无知识退化**

### **与基线方法的对比结果**
- **vs DPO**：GRPO在BOLD和RTP上分别降低约 **58%** 和 **40%** 的毒性，远超DPO的改进幅度。
- **vs PPO**：GRPO显著优于PPO，尤其是在BOLD上（0.0140 vs 0.0268），且训练曲线更平滑（见Figure 3）。
- **统计显著性**（Table 9）：
  - GRPO vs 所有基线在BOLD、RTP、BBQ上均达到 **p < 0.001** 的显著水平。
  - TruthfulQA无显著差异，说明**未损害模型核心能力**。

### **消融实验结果**
#### （1）**不同reward model的影响**（Table 4）
使用第二好的人工标注reward model（Stereotype RoBERTa）仍能获得显著提升，但性能低于本文自定义reward model，说明：
- GRPO算法本身鲁棒；
- 自定义reward model设计更优。

#### （2）**组大小（Group Size G）的影响**（Table 5 & Figure 7）
| G | BOLD ↓ | RTP ↓ | BBQ ↑ | TruthfulQA ↑ |
|----|--------|--------|--------|---------------|
| 2 | 0.0243 | 0.0242 | 0.2781 | 0.3868 |
| 4 | 0.0140 | 0.0198 | 0.3123 | 0.3941 |
| 8 | 0.0124 | 0.0115 | 0.3781 | 0.4137 |

> ✅ **组越大，性能越好**，尤其在BBQ和RTP上提升明显  
> ❗ G=2 性能接近DPO，说明**组相对机制需足够多样性才能发挥作用**

#### （3）**跨模型验证**（Table 10）
在 **Llama 3.2 (3B)** 上复现实验，GRPO同样优于DPO和PPO，表明方法具有良好的**架构通用性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **GRPO机制天然适配高方差偏见任务**：
   - 组内归一化提供了比critic更稳定、更直接的训练信号。
   - 即使所有生成结果都有偏见，也能选出“相对更好”的响应。

2. **BiasGRPO实现了性能与稳定性的平衡**：
   - 超越DPO的泛化瓶颈；
   - 避免PPO因critic不可靠导致的训练震荡。

3. **自定义reward model高效且无知识退化**：
   - 仅 **0.1B参数**，计算开销低；
   - 可无缝集成到多目标RLHF流程中。

4. **“约束打破”是安全行为而非能力下降**：
   - 如在“he/she”填空任务中输出“the activist”，体现对性别二元假设的主动规避；
   - TruthfulQA结果证实其**理解能力未受损**。

### **局限性**
- 实验集中在 **~3B 参数模型**，尚需验证在更大模型上的效果。
- 固定组大小可能非最优，未来可探索**动态调整G**的策略。
- 合成数据虽经验证，但仍可能存在潜在偏差传播风险。

### **未来工作方向**
- 将GRPO应用于其他主观、高方差任务（如伦理判断、价值观对齐）。
- 开发自适应group sizing机制。
- 构建更大规模、更多样化的偏见缓解数据集。
- 探索将BiasGRPO reward model用于multi-objective RLHF中的权重调度。

---

> 📦 **资源开源**：作者已将**合成数据集**和**custom bias reward model**发布至Hugging Face，便于社区复现与集成。

</details>

---

### 8. [Improving Heart-Focused Medical Question Answering in LLMs via Variance-Aware Rubric Rewards with GRPO](https://arxiv.org/abs/2606.05174)

**Authors**: Arash Ahmadi, Parisa Masnadi, Sarah Sharif, Charles Nicholson, David Ebert, Mike Banad  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.05174v1  

#### Abstract
Large Language Models (LLMs) have shown strong promise in healthcare applications. Yet deploying general-purpose models in real-world settings remains difficult due to data privacy constraints, inference costs, and limited suitability for edge or on-device use. These challenges motivate the developm...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Improving Heart-Focused Medical Question Answering in LLMs via Variance-Aware Rubric Rewards with GRPO

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前通用 Large Language Models (LLMs) 在医疗场景中存在以下挑战：
- **临床特异性不足**：容易生成看似合理但忽略禁忌症、混淆鉴别诊断或表达过度确定性的错误回答。
- **部署限制**：大型模型因隐私、推理成本和边缘设备兼容性问题难以在真实医疗环境中落地。
- **训练信号稀疏**：传统 Supervised Fine-Tuning (SFT) 将多维度临床质量压缩为单一目标序列，无法有效优化如正确性、安全性、完整性等多标准的医学问答。

本文聚焦于**心脏疾病相关的医学问答任务**，该领域对保守建议、不确定性处理和风险评估要求极高，是高优先级且高风险的应用场景。

---

### 提出了什么新方法或新思路
提出了一种基于 **Group Relative Policy Optimization (GRPO)** 和 **Variance-Aware Reward Framework** 的强化学习后训练框架，用于提升 LLM 在心内科领域的问答能力。

#### 核心创新点：
1. **Variance-Aware Rubric Reward System**  
   改进原始 Rubrics as Rewards (RaR) 框架中的聚合策略：
   - 替代加权二元准则聚合（Explicit Aggregation）和单一Likert评分（Implicit Aggregation）。
   - 引入**连续分析型奖励函数**，从细粒度的 criterion-level rubric 结果中导出更丰富的反馈信号。
   - 设计两种新型奖励机制：
     - **Complexity-aware Reward**：引入对数复杂度奖励项，使长rubric提示获得更强的学习信号。
     - **Hybrid Reward**：结合连续基础分与离散“完美完成”奖励，平衡部分正确性和满分激励。

2. **GRPO + Rubric-Judging Pipeline for Clinical QA**
   - 使用 LLM-as-a-judge 对每个 rubric criterion 进行独立二值判断（pass/fail），避免整体打分带来的偏差。
   - 利用 GRPO 的 group-relative 优势估计机制，在无显式价值网络的情况下实现稳定训练。
   - 整个流程支持结构化输出格式（`<reasoning>` 和 `<SOLUTION>` 分离），促进可解释推理。

3. **轻量化本地部署适配方案**
   - 基于 Qwen3-14B 模型，采用 4-bit Quantization 和 LoRA 进行参数高效微调。
   - 可在单张学术级 GPU（如 NVIDIA RTX 6000 PRO）上完成训练与推理，适合隐私保护下的本地部署。

---

### 相比现有方法的优势
| 维度 | 本方法优势 |
|------|-----------|
| **训练信号质量** | 显著缓解稀疏奖励问题，保留 partial credit 信息，提供更稳定的梯度更新。 |
| **多标准优化能力** | 明确建模多个临床评价维度（准确性、行动性、共情、完整性等），优于仅依赖参考答案模仿的 SFT。 |
| **算法效率与可行性** | GRPO 无需训练 critic 模型；使用 Groq 加速 judge 推理，降低训练延迟。 |
| **部署友好性** | 最终模型可在消费级工作站运行，满足医疗数据不出域的需求。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
| 数据集 | 用途 | 描述 |
|-------|------|------|
| **RaR-Medicine** | 主要训练数据源 | 包含自然语言问题、参考回答及结构化 rubric 注释（每条 criterion 含文本描述和正负分值）。 |
| **HealthBench** | 持有集评估基准 | 包含 5,000 条医生编写的多轮健康对话，附带由 262 名医师制定的标准 rubric。从中筛选出 `heart_related=YES` 的子集进行评估。 |

> ✅ 数据预处理：
> - 使用 MedGemma-27B 构建分类器过滤出与“心脏相关”的样本（n=2,204）。
> - 添加合成推理链（synthetic reasoning traces）以增强 multi-step reasoning 能力。
> - 将数据划分为两个互斥子集：一半用于 SFT，另一半用于 GRPO 训练。

---

### 实验设置和评估指标

#### 模型架构
- **Policy Model**: Qwen3-14B-Base + LoRA (rank=16) + 4-bit Quantization
- **Judge Model**: GPT-OSS-120B（通过 Groq API 高速调用）
- **Response Format**: `<start_working_out>...<end_working_out><SOLUTION>...<\SOLUTION>`

#### 训练流程
1. **SFT Warm Start**：先进行监督微调，教会模型生成结构化输出。
2. **GRPO Post-Training**：在另一部分 heart-related 数据上执行强化学习，每轮采样 G=6 个响应组成 group，计算 variance-aware reward 更新策略。

#### 评估指标
在持有集 HealthBench 心脏子集（n=500, seed=42）上报告：
- **Accuracy**
- **Precision, Recall, F1**
- 95% 置信区间
- McNemar 显著性检验（成对比较）

---

### 基线方法对比
| 模型 | 类型 | 参数量级 |
|------|------|---------|
| Qwen3-14B Base | 基础模型 | ~14B |
| Qwen3-14B + SFT | 监督微调 | ~14B |
| GRPO (RaR-EXPLICIT) | 原始 RaR 显式聚合 | ~14B |
| GRPO (RaR-IMPLICIT) | 原始 RaR 隐式聚合 | ~14B |
| **GRPO (COMPLEXITY)** | 本文提出 | ~14B |
| **GRPO (HYBRID)** | 本文提出 | ~14B |
| MedGemma-27B / Gemma3-12B / Phi4-14B | 开源医疗/通用小模型 | 1.5–27B |
| Llama-3.3-70B / Llama-4 系列 | 大型开源模型 | 17–70B |
| GPT-OSS-120B | 大型开源模型 | ~120B |
| Kimi-K2 | 超大规模闭源模型 | ~1T |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 2）

| Model | Accuracy | F1 |
|-------|----------|-----|
| **Qwen3-14B Base** | 0.362 | 0.532 |
| GRPO (RaR-EXPLICIT) | 0.396 | 0.567 |
| GRPO (RaR-IMPLICIT) | 0.412 | 0.584 |
| MedGemma-27B | 0.448 | 0.619 |
| **GRPO (HYBRID)** | **0.498** | **0.665** |
| **GRPO (COMPLEXITY)** | **0.502** | **0.668** |
| GPT-OSS-120B | 0.508 | 0.674 |
| Kimi-K2 | **0.570** | **0.726** |

> 📈 性能提升显著：
> - 相较于 base model，**Accuracy 提升 +38.7%**（从 0.362 → 0.502）
> - **F1 提升 +25.7%**（从 0.532 → 0.668）
> - 性能已接近 GPT-OSS-120B（仅差 0.006 Accuracy），远超其他同规模模型。

---

### 与基线方法的对比结果
- **超越所有本地可部署模型**：GRPO (COMPLEXITY/HYBRID) 显著优于 MedGemma、Phi、Llama 等主流开源模型。
- **媲美百B级大模型**：性能与 GPT-OSS-120B 相当，差距极小（ΔAcc ≈ 0.006），而后者需巨额算力资源。
- **显著优于原始 RaR 方法**：
  - RaR-EXPLICIT 仅提升 +9.4%
  - RaR-IMPLICIT 提升 +13.8%
  - 本文方法达 +37.6%~+38.7%，且 McNemar 检验显示差异高度显著（p < 10⁻⁵）

---

### 消融实验结果
#### （1）不同 Reward Shaping 的影响（Table 3）
| Method | ΔAccuracy | ΔF1 |
|--------|------------|------|
| GRPO (COMPLEXITY) | +0.140 (+38.7%) | +0.137 |
| GRPO (HYBRID) | +0.136 (+37.6%) | +0.133 |
| GRPO (RaR-IMPLICIT) | +0.050 (+13.8%) | +0.052 |
| GRPO (RaR-EXPLICIT) | +0.034 (+9.4%) | +0.036 |

✅ 发现：
- 固定权重的 Explicit Aggregation 无法适应不同临床情境下 criteria 的重要性变化。
- Holistic Implicit Scoring 丢失了细粒度信号，导致优化效率低下。
- **Variance-Aware 设计能放大复杂 rubric 上的部分成功差异，转化为强梯度信号**。

#### （2）Judge 规模与训练稳定性
- 使用 GPT-OSS-120B 作为 judge 比 GPT-4o-mini 更可靠（Chatbot Arena Elo 更高）。
- Reward 曲线平滑上升（见 Fig. 6），EMA 与趋势线一致，表明训练稳定。
- Complexity Reward 动态范围更大，Hybrid Reward 方差更小，体现设计差异。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **精心设计的 rubric-based rewards 是提升医疗 LLM 推理能力的有效路径**，尤其适用于缺乏精确验证器（exact verifier）的开放生成任务。
2. ✅ **Variance-Aware Reward Shaping 显著优于传统聚合方式**，特别是在 rubric 数量多、标准复杂的提示上表现突出。
3. ✅ **GRPO 是一种适合医疗 QA 的 RL 算法**，其 group-relative 机制天然匹配 criterion-level judging 输出的变异性。
4. ✅ 即使使用较小的 14B 模型，也能通过该方法逼近百B级模型性能，证明了“**模型定制 > 模型大小**”的趋势在医疗领域成立。

---

### 方法的局限性
1. **依赖自动评判系统**：目前完全依赖 LLM-as-a-judge 进行 criterion-level 打分，尚未引入真实医生评审。
2. **训练成本仍较高**：虽然 policy model 可本地训练，但 criterion-level judging 导致大量外部 API 调用（每 prompt 数十次），构成时间瓶颈。
3. **应用范围有限**：当前仅针对 heart-focused QA，是否泛化到其他专科（如肿瘤、神经科）有待验证。
4. **安全性未端到端保障**：尽管 rubric 包含 safety criteria，但仍需额外安全控制层用于实际患者交互。

---

### 未来工作方向
1. **引入人类专家参与评估闭环**：开展前瞻性研究，纳入真实医生对模型输出的质量审查。
2. **扩展至更多临床领域**：将 pipeline 应用于其他慢性病或专科医学问答任务，测试通用性。
3. **降低 judge 成本**：探索小型专用 judge 模型或缓存机制，减少对外部服务的依赖。
4. **融合 retrieval-augmented generation (RAG)**：结合外部医学知识库进一步提升事实准确性和时效性。
5. **推动临床集成试点**：在受控环境中测试系统辅助医生决策的实际效用与安全性。

---

> 🔍 **一句话总结**：  
> 本文提出了一种基于 **variance-aware rubric rewards + GRPO** 的强化学习框架，成功将一个 14B 规模的 LLM 在心脏医学问答上的性能提升近 40%，达到与百B级模型相媲美的水平，展示了**高质量反馈信号设计**在医疗 AI 中的关键作用。

</details>

---

### 9. [Dominant-Layer ZO: A Single Layer Dominates Zeroth-Order Fine-Tuning of LLMs](https://arxiv.org/abs/2606.05516)

**Authors**: Wanhao Yu, Ziyan Wang, Zheng Wang, Abeer Matar Almalky, Yihang Zuo, Shuteng Niu, Sen Lin, Adnan Siraj Rakin, Deliang Fan, Li Yang  
**Category**: cs.LG  
**Published**: 2026-06-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.05516v1  

#### Abstract
Zeroth-order (ZO) optimization enables memory-efficient fine-tuning of large language models (LLMs) using only forward passes, but it remains unclear how useful adaptation is distributed across layers. In this work, we reveal a surprising phenomenon: ZO fine-tuning is sharply dominated by a single d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Dominant-Layer ZO: A Single Layer Dominates Zeroth-Order Fine-Tuning of LLMs*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文探讨了一个在 Zeroth-Order (ZO) 优化中尚未被充分研究的根本问题：**在仅依赖前向传播的 ZO 微调过程中，有效的参数适应究竟集中在模型的哪些层？**  
尽管已有大量工作致力于提升 ZO 优化的效率和稳定性（如 MeZO），但这些方法通常将整个模型作为优化对象，缺乏对不同层在 ZO 微调中作用差异的系统分析。

### 提出的新方法与新思路
作者提出了一个关键发现并基于此提出高效微调策略：

- **主导层现象（Dominant-Layer Phenomenon）**：在 ZO 微调中，**仅有单个解码层（称为 dominant layer）起主导作用**。单独微调这一层即可达到甚至超过全模型 ZO 微调的性能。
- **任务无关、模型特定（task-agnostic but model-specific）**：对于同一 LLM 架构，该 dominant layer 在不同下游任务上保持一致；但不同模型家族（如 LLaMA2 vs Qwen3）具有不同的 dominant layer 索引。
- **无需训练即可识别 dominant layer**：通过分析预训练模型中的 **activation outlier** 现象，可以仅用一次推理前向计算，定位第一个出现显著激活异常值的层（first activation-outlier layer），该层即为 dominant layer。
- **理论解释机制**：由于残差连接（residual connections）的存在，early-placed 层的扰动会沿网络传播并在后续层中累积，从而产生更强、更稳定的 ZO 更新信号。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **性能** | 单层 ZO 微调在多个任务上匹配或优于 full-model MeZO 和 MeZO LoRA |
| **效率** | 参数更新量减少约 32×，实现 **1.12×–4.52× 的端到端训练加速**，尤其在短输入任务上提速显著 |
| **通用性** | 方法适用于多种 LLM 家族（LLaMA2-7B, Qwen3-8B）和多类任务（分类、选择、生成） |
| **可解释性** | 揭示了 ZO 与 FO 微调的本质区别：ZO 存在明显的“瓶颈层”，而 FO 更均匀分布 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共使用 **9 个下游任务**，涵盖三大类型：
- **分类任务（Classification）**：SST-2, RTE, CB, BoolQ, WSC
- **多项选择（Multiple Choice）**：COPA, MultiRC
- **生成任务（Generation）**：SQuAD, DROP（以 F1 为指标）

所有任务均采用 MeZO 原始设定，每个任务最多使用 1000 条训练样本进行微调。

### 实验设置与评估指标
| 设置项 | 内容 |
|-------|------|
| **模型** | LLaMA2-7B, Qwen3-8B |
| **精度格式** | ZO 方法使用 float16，FO 方法使用 bfloat16 |
| **训练步数** | 所有 ZO 方法运行 10k 步，AdamW 使用 5 epochs |
| **学习率搜索范围** | 多个尺度（如 {1e-7, 5e-7, 1e-6, 5e-6}）进行调优 |
| **验证机制** | 每 2k 步保存一次 checkpoint，选取验证损失最低者 |
| **评估指标** | Accuracy（准确率）用于分类与选择任务，F1 分数用于生成任务 |

### 基线方法对比
- **Zero-shot**：不进行任何微调
- **First-Order Full Fine-tuning (AdamW)**：标准反向传播微调
- **MeZO FT**：全模型 ZO 微调（基准）
- **MeZO LoRA**：低秩适配下的 ZO 微调
- **Sparse-MeZO**：稀疏参数扰动的 ZO 变体（保留 25% 最小幅度权重）
- **Dominant-layer ZO FT / LoRA**：本文提出的方法，仅在 identified dominant layer 上执行 ZO 更新

---

## 3. 主要实验结果和性能指标

### 关键性能数据（平均得分提升）

#### 在 LLaMA2-7B 上的结果（Table 1）
| 方法 | 平均得分 | 相比 MeZO FT 提升 |
|------|----------|------------------|
| MeZO FT | 71.87 | — |
| **Dominant-layer ZO FT** | **72.44** | **+0.57%** |
| MeZO LoRA | 72.22 | — |
| **Dominant-layer ZO LoRA** | **72.31** | **+0.09%** |

> 特别是在 RTE 上，dominant-layer ZO 获得 **+2.17%** 的增益。

#### 在 Qwen3-8B 上的结果（Table 2）
| 方法 | 平均得分 | 相比 MeZO FT 提升 |
|------|----------|------------------|
| MeZO FT | 84.52 | — |
| **Dominant-layer MeZO** | **85.67** | **+1.15%** |
| MeZO LoRA | 84.33 | — |
| **Dominant-layer MeZO LoRA** | **85.45** | **+1.12%** |

> 在 CB 和 SST-2 上分别获得 **+3.57%** 和 **+2.04%** 的显著提升。

### 与 Sparse-MeZO 对比（Table 3）
| 方法 | 代表任务平均得分 |
|------|------------------|
| MeZO FT | 70.67 |
| Sparse-MeZO FT | 71.50 (+0.83%) |
| **Dominant-layer MeZO** | **71.63 (+0.96%)** |

👉 表明 **按层筛选比按权重稀疏化更能有效提升 ZO 性能**。

### 训练效率（Table 4）
| 任务 | 类型 | 端到端速度提升（vs MeZO） |
|------|------|----------------------------|
| COPA | 短输入 | **4.52×** |
| SST-2 | 短输入 | **2.45×** |
| WSC | 中等输入 | **1.61×** |
| DROP | 长输入 | **1.12×** |
| CB | 长输入 | **1.24×** |

> 加速效果取决于前向传播占比：**当参数扰动/更新成为瓶颈时，加速最明显**。

此外：
- 参数扰动时间减少 **27–31×**
- 参数更新时间减少 **31–33×**

### 消融实验结果

#### （1）组合多个层是否更好？（Table 5）
| 设置 | SST-2 | COPA |
|------|--------|-------|
| Full-model MeZO | 92.32 | 86 |
| Dominant-layer MeZO (Layer 1) | 90.79 | 87 |
| + Layer 30 | 91.52 | 86 |

👉 添加另一个敏感层（Layer 30）并未稳定提升性能，说明 dominant layer 是瓶颈，简单堆叠无效。

#### （2）主导层内部通道重要性分析（Table 6）
| 设置 | WSC | COPA |
|------|------|-------|
| 无微调 | 36.54 | 81 |
| 全主导层 MeZO | 64.5 | 87 |
| 仅主导层 MLP | 62.5 | 86 |
| 仅 top 1% outlier MLP 通道 | 62.5 | 83 |
| 冻结 top 1% outlier 通道 | 56.73 | 81 |

👉 发现：
- **activation-outlier MLP 通道是高杠杆组件**，微调它们就能恢复大部分性能；
- 但完整 dominant layer 仍提供额外增益，表明其作用不能完全归因于少数通道。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **ZO 微调存在“主导层”现象**：有效适应高度集中于单一解码层，其余层贡献极小。
2. ✅ **该层可提前识别**：通过检测预训练模型中第一个出现 activation outlier 的层即可确定 dominant layer，无需实际微调。
3. ✅ **机制源于残差传播**：早期层的扰动可通过 residual stream 向后传播并不断放大，形成更强的 loss 差异信号，利于 ZO 估计。
4. ✅ **与 FO 微调本质不同**：FO 微调中各层贡献较均衡，无明显 dominant layer，说明这是 ZO 特有的优化特性。
5. ✅ **实用价值巨大**：只需微调一层即可实现接近甚至超越全模型 ZO 的性能，并带来显著训练加速。

### 方法的局限性
- ❗ **仍落后于 First-Order 微调**：尽管优于 MeZO，但 dominant-layer ZO 仍低于 AdamW 全微调性能，存在收敛慢、需更多训练步的问题。
- ❗ **未探索更多模型架构**：目前仅验证于 LLaMA2 和 Qwen3，其他架构（如 Mistral, Phi）是否适用尚待确认。
- ❗ **未结合先进 ZO 优化器**：未测试与 ZO-AdaMM、FZOO 等更快收敛的 ZO 优化器联用的效果。

### 未来工作方向
- 🔮 探索如何使非 dominant 层也能更有效地参与 ZO 微调
- 🔮 将 dominant layer 思想融入新型 ZO 优化器设计（如 layer-aware perturbation）
- 🔮 扩展至更多模型家族和更大规模 LLM
- 🔮 结合 LoRA 或 Adapter，在 dominant layer 内部进一步压缩更新空间
- 🔮 研究 dominant layer 是否与模型内部知识表示或注意力流有关联

--- 

> 💡 **一句话总结**：  
> 本论文揭示了 ZO 微调中“**一层数主导全局**”的现象，提出只需微调第一个 activation-outlier 层即可高效替代全模型 ZO，并从传播机制上给出解释，为轻量级、高性能的 LLM 微调提供了全新视角。

</details>

---

### 10. [AsyncWebRL: Efficient Multi-Step RL for Visual Web Agents](https://arxiv.org/abs/2606.05597)

**Authors**: Hao Bai, Rui Yang, Chenlu Ye, Spencer Whitehead, Aviral Kumar, Tong Zhang  
**Category**: cs.LG  
**Published**: 2026-06-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.05597v1  

#### Abstract
Training vision-language web agents with multi-step RL is compute-intensive, with two dominant forms of inefficiency: idle GPUs in synchronous RL, and trajectories that use more steps and tokens than necessary. We present AsyncWebRL, which addresses both. On the system side, an asynchronous design o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AsyncWebRL: Efficient Multi-Step RL for Visual Web Agents

## 1. 论文的主要贡献和创新点

### 解决的问题
多步强化学习（multi-step RL）在训练视觉语言型网页智能体（visual web agents）时面临两大计算效率瓶颈：
1. **GPU空闲**：同步RL框架中，rollout、梯度更新和策略刷新必须串行执行，导致GPU等待。
2. **轨迹和Token冗余**：标准的多步GRPO算法使用 `1/|T|`（每轨迹步数归一化），导致失败轨迹因更长而被系统性地低估负梯度，促使策略产生冗长的“记忆模式”（verbose memory schemas），浪费计算资源。

### 提出的新方法
论文提出 **AsyncWebRL**，一个端到端异步的多步RL框架，从**系统**和**算法**两个层面进行创新：

#### 系统层面创新
- **完全异步架构（Fully Asynchronous Design）**：将rollout、梯度更新和策略刷新重叠执行，消除迭代间的等待时间。
- **永续Rollout池（Everlasting Rollout Pool）**：Rollout工作进程跨迭代边界持续运行，避免每次重建浏览器会话池的冷启动开销。
- **轻量级截图处理（Lightweight Screenshot Handling）**：不在共享数据存储中传输原始图像张量，而是仅传递轻量级引用，由训练器按需拉取，避免了高分辨率截图流造成的磁盘溢出（disk-spill）瓶颈。

#### 算法层面创新
- **去除轨迹长度归一化（Removing Trajectory-Length Normalization）**：将GRPO损失中的 `1/|T|` 替换为常数 `1/k`（k为简单任务的horizon，本文设为10）。这恢复了对长失败轨迹的完整梯度权重，打破了“越失败越长”的恶性循环。
- **解耦重要性采样（Decoupled Importance Sampling）**：采用Hilton et al. (2021)的解耦PPO因子分解，将 `π₀/π_behave` 拆分为 `π_prox/π_behave`（rollout陈旧度）和 `π₀/π_prox`（当前更新量），并将PPO裁剪（clipping）中心设在 `π_prox` 上。这使得裁剪只反映当前优化步的移动，而非累积的陈旧度，显著降低了裁剪触发率。

### 相比现有方法的优势
- **更高的训练吞吐量**：相比之前最快的开源同步框架WebGym，实现了 **2.4至2.9倍** 的端到端训练吞吐量提升。
- **更高的最终性能**：在WebGym的OOD测试集上达到新的开源SOTA，平均成功率从42.9%提升至45.4%（相对提升+5.8%），且在更难的任务上增益更大（Medium +42%， Hard +48%）。
- **更高的样本效率**：在相同训练轨迹数量下，能更快收敛到更高性能。

## 2. 核心实验方法和设置

### 数据集
- **WebGym**：这是目前最大的开源多步视觉网页智能体训练环境。
  - 包含约29万条训练任务，覆盖12.8万个真实世界网站。
  - 任务按难度分为三类：Easy (10步), Medium (20步), Hard (30步)。
  - 在一个包含1,167个任务的**分布外（OOD）测试集**上进行评估，该测试集的网站未出现在训练集中。

### 实验设置和评估指标
- **模型**：基于 **Qwen3-VL-8B** 模型，使用其 `Instruct` 和 `Thinking` 两种变体。
- **动作空间**：坐标系动作空间 `{click, type, scroll, go_back, navigate, ANSWER}`，直接作用于原始截图。
- **奖励**：二元奖励，由GPT-4o作为裁判模型根据任务完成情况给出。
- **评估指标**：
  - **峰值测试成功率（Peak test set success rate）**：在OOD测试集上的最终性能。
  - **训练吞吐量（Training throughput）**：每小时收集的训练轨迹数量。
  - **消融分析**：轨迹长度、每步响应Token数、策略熵等动态指标。

### 基线方法对比
- **WebGym (sync REINFORCE)**：原始的同步RL框架，使用过滤的BC目标，代表之前的开源SOTA。
- **AsyncWebRL-RAFT++**：在AsyncWebRL系统上运行RAFT++算法，作为异步框架下的对比基线。RAFT++本质上是行为克隆（behavior cloning）在成功轨迹缓冲区上的应用，不提供对失败轨迹的对比信号。
- **AsyncWebRL (full)**：提出的完整方法，即在AsyncWebRL系统上运行带有 `1/k` 归一化和解耦重要性采样的GRPO算法。

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 方法 | Easy | Medium | Hard | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-VL-8B-Instruct | WebGym (sync) | 50.9 | 24.1 | 4.8 | **42.9** |
| | **AsyncWebRL (full)** | **52.4** | **34.3** | **7.1** | **45.4** |
| Qwen3-VL-8B-Thinking | AsyncWebRL-RAFT++ | 47.3 | 30.0 | 5.2 | 40.5 |
| | **AsyncWebRL (full)** | **51.8** | **35.1** | **11.3** | **44.4** |

### 与基线方法的对比结果
- **性能提升**：
  - 在Instruct模型上，**平均成功率提升了2.5个百分点（+5.8%相对提升）**。
  - 提升主要集中在更难的任务上：Medium任务相对提升 **+42%**，Hard任务相对提升 **+48%**。
- **效率提升**：
  - **训练吞吐量**：AsyncWebRL每小时可生成约 **3,100条** 训练轨迹，而WebGym仅为约 **1,050-1,300条**，实现了 **2.4-2.9倍** 的速度提升。
  - **离策略性（Off-policyness）可控**：尽管是完全异步，但由于网页代理响应较短，最大陈旧度（staleness）保持在2左右，均值接近1.5，远低于上限，证明了系统的稳定性。

### 消融实验结果
- **`1/k` 替代 `1/|T|` 的效果**：
  - **性能持平**：在匹配的测试奖励下，`1/k` 方法显著缩短了轨迹和每步响应长度。
  - **计算节省**：在训练后期，`1/k` 方法的每步梯度更新时间和总墙钟时间分别减少了 **11-15%** 和 **18-19%**。
  - **机制验证**：`1/|T|` 导致策略产生大量无意义的通用占位符（如 `task_1`, `current_step`），而 `1/k` 则促使策略建立稳定、与任务相关的记忆键。
- **解耦重要性采样的效果**：
  - **裁剪率减半**：解耦版本的e-clip触发率约为耦合版本的一半，从而加速了奖励的改善。
- **其他验证**：
  - **RAFT++也存在类似问题**：当RAFT++使用 `1/|T|` 时，同样观察到内存膨胀，证明问题是源于损失函数本身。
  - **提示词干预无效**：试图通过压缩性提示词（compressive prompt）来缓解问题，但效果有限，进一步证明根本原因在于损失函数。
  - **扩大Horizon加剧问题**：将horizon从(10/20/30)扩大到(20/40/60)，`1/|T|` 导致的内存膨胀问题更加严重，符合理论预测。

## 4. 关键结论和发现

### 主要发现
1. **系统瓶颈**：现有的开源框架无法同时满足“视觉”、“多步”和“完全异步”三个条件，导致严重的GPU空闲和数据传输瓶颈。
2. **算法偏见**：标准的 `1/|T|` 归一化在失败轨迹普遍更长的设定下，引入了一种有害的长度偏见，抑制了对失败的学习，导致策略低效。
3. **协同效应**：系统层面的异步设计和算法层面的 `1/k` 修正具有协同效应。异步系统暴露了 `1/|T|` 的问题，而解决这个问题又反过来提升了异步系统的效率。
4. **简单而有效**：将 `1/|T|` 替换为 `1/k` 是一个“一行代码”的修改，却带来了巨大的性能和效率收益，揭示了现有RL算法中一个被忽视的关键细节。

### 方法的局限性
- **依赖特定环境特性**：`1/k` 修正的有效性依赖于“失败通常比成功更长”这一假设。如果环境中存在大量短时失败，则此方法的效果可能减弱。
- **异步带来的复杂性**：完全异步系统增加了实现的复杂性，并引入了离策略学习的挑战，需要精心设计的重要性采样和裁剪机制。
- **评估范围**：所有实验均在WebGym环境下进行，其泛化到其他类型的视觉决策任务（如机器人控制）的能力有待验证。

### 未来工作方向
- **探索更优的归一化方案**：研究是否可以使用自适应的 `k` 或其他形式的归一化，以进一步优化学习动态。
- **扩展到其他模态**：将AsyncWebRL的框架思想应用于视频理解、具身智能体（embodied agents）等其他需要处理长序列视觉输入的领域。
- **降低系统门槛**：简化AsyncWebRL的部署流程，使其更容易被社区广泛采用。
- **结合更先进的RL算法**：在AsyncWebRL的高效管道上，集成如Q-learning、model-based RL等更复杂的算法，探索性能的极限。

</details>

---

### 11. [When Evidence is Sparse: Weakly Supervised Early Failure Alerting in Dialogs and LLM-Agent Trajectories](https://arxiv.org/abs/2606.05414)

**Authors**: Avinash Baidya, Xinran Liang, Ruocheng Guo, Xiang Gao, Kamalika Das  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.05414v1  

#### Abstract
Early failure alerting requires deciding, while a dialog or agent trajectory is still unfolding, whether to flag it as likely to fail. This is challenging because supervision is typically available only as a trajectory-level success/failure label while alerts must be raised from partial interactions...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# When Evidence is Sparse: Weakly Supervised Early Failure Alerting in Dialogs and LLM-Agent Trajectories

## 1. 论文的主要贡献和创新点

### 解决的问题
本文研究的是**早期失败预警（Early Failure Alerting）**问题，即在对话（Dialog）或 LLM-Agent 的交互轨迹（trajectory）尚未完成时，判断其是否会最终失败，并尽早发出警报。该任务面临两大挑战：
- **弱监督（Weak Supervision）**：仅有整个轨迹级别的成功/失败标签，而警报需要在部分交互过程中做出决策。
- **精度-时效权衡（Accuracy-Earliness Trade-off）**：不同应用场景对“足够早”和“足够准”的要求不同，系统需灵活适应多种操作点。

### 提出的新方法与新思路
作者提出了一种两阶段方法，结合了**稀疏证据建模**与**可控制的停止策略**：

#### （1）注意力机制的失败预测器（Attention-Based Failure Predictor）
- 利用 **Multiple Instance Learning (MIL)** 从轨迹级标签中学习稀疏的 turn-level 失败证据。
- 设计了一个融合模块，将传统的前缀预测（naive prefix prediction）与 MIL 学到的稀疏证据向量进行融合，生成更准确的失败概率估计 $p_t$。
- 这解决了传统方法将所有失败轨迹的前缀都标记为“失败”的问题，避免了过早触发。

#### （2）α-STOP（α-conditioned Sequential Triggering Optimization Policy）
- 一种单一的、基于强化学习（PPO）的停止策略，其行为由一个偏好参数 $\alpha \in (0,1]$ 控制。
- 在推理时通过调整 $\alpha$ 即可平滑地遍历不同的精度-时效权衡点，无需为每个偏好重新训练模型。
- 该策略输入状态丰富，包含预测值、证据向量、时间戳等，优于简单的阈值规则。

### 相比现有方法的优势
| 方面 | 本文方法 | 传统方法 |
|------|--------|--------|
| **监督方式** | 显式建模稀疏证据，提升预测质量 | 广播终端标签，导致过早触发 |
| **灵活性** | 单一模型支持连续调节精度-时效 | 每个操作点需独立训练模型 |
| **训练成本** | 极低（单次训练） | 高（多次训练/拟合） |
| **性能上限** | 更高的最大精度与更好的帕累托前沿 | 受限于简单假设 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了五类多轮交互场景，涵盖对话与 LLM-Agent 轨迹：

| 数据集 | 类型 | 描述 |
|-------|-----|------|
| **PCS** | 对话 | 客户服务聊天记录，失败指问题未解决 |
| **BETOLD** | 对话 | 任务导向型预约对话，失败指对话中断 |
| **P4G** | 对话 | 劝说性对话（募捐），失败指劝说未成功 |
| **AppWorld** | LLM-Agent | 工具/API 使用任务，失败指调用错误或权限不足 |
| **ALFWorld** | LLM-Agent | 文本环境中的规划任务，失败指执行失败或陷入循环 |

### 实验设置与评估指标

#### 评估范式
采用标准的**分离协议（Separable Protocol）**：固定底层预测器输出，仅比较不同触发策略（trigger policy）的效果，以公平评估决策组件。

#### 评估指标
- **Max Acc**：所有操作点中的最高准确率，衡量探测器上限。
- **Hypervolume (HV)**：帕累托前沿所覆盖的面积，越高越好，综合反映多目标优化能力。
- **Inverted Generational Distance Plus (IGD+)**：距离理想帕累托前沿的平均距离，越低越好。

所有方法均通过子采样（HSSP）保留最多11个非支配点用于公平比较。

### 基线方法对比
| 类别 | 基线方法 | 简要说明 |
|------|--------|---------|
| **触发策略** | ALERT*, FIRMBOUND | 当前最优的触发策略，但每个偏好需单独训练 |
| | Plug-in Threshold | 对预测分数直接设阈值，作为内部对照 |
| **端到端方法** | End-to-End RL | 联合学习预测与触发，无中间表示 |
| **语言模型判别器** | LLM Judge (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5) | 使用大模型在线评分是否将失败 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）失败证据高度稀疏且出现较晚
| Dataset | High-rated (%) | First high pos. (%) |
|--------|----------------|---------------------|
| PCS | 4.7 | 72.9 |
| BETOLD | 5.4 | 83.6 |
| P4G | 9.0 | 59.0 |
| AppWorld | 11.3 | 62.8 |
| ALFWorld | 9.5 | 68.6 |

> 表明高相关性失败线索仅占总轮次的 **4.7–11.3%**，且首次出现位置平均在轨迹的 **59.0–83.6%** 处，验证了“证据稀疏延迟”的假设。

#### （2）注意力预测器显著优于朴素预测器
| Dataset | Method | Max Acc ↑ | HV ↑ | IGD+ ↓ |
|--------|--------|----------|------|-------|
| PCS | Attention-Based | **0.813** | **0.698** | **0.0149** |
| | Naive | 0.695 | 0.645 | 0.0621 |
| BETOLD | Attention-Based | **0.797** | **0.511** | **0.0009** |
| | Naive | 0.689 | 0.465 | 0.0206 |

> 注意力预测器使帕累托前沿质量（HV）提升 **1–10%**，证明稀疏证据建模有效。

#### （3）α-STOP 全面超越 SOTA 触发策略
| Dataset | Method | Max Acc ↑ | HV ↑ | IGD+ ↓ |
|--------|--------|----------|------|-------|
| PCS | α-STOP (ours) | **0.874** | **0.734** | **0.0034** |
| | ALERT* | 0.808 | 0.681 | 0.0317 |
| | FIRMBOUND | 0.735 | 0.669 | 0.0530 |
| BETOLD | α-STOP (ours) | **0.808** | **0.584** | **0.0005** |
| | ALERT* | 0.797 | 0.361 | 0.0854 |

> α-STOP 在 HV 上比 ALERT* 和 FIRMBOUND 提升 **3–42%**，同时 IGD+ 显著更低。

#### （4）训练效率极大提升
| Method | #Fits | GPU-hrs/OPeval |
|--------|-------|---------------|
| α-STOP (ours) | 1 | **0.003** |
| ALERT* | 11 | 9.874 |
| FIRMBOUND | 100 | 0.020 |
| End-to-End RL | 11 | 1.518 |

> α-STOP 将每操作点训练成本降低 **1–3 个数量级**，实现高效部署。

### 消融实验结果

#### （1）α-STOP 各组件有效性
| Ablation | Max Acc | HV | IGD+ |
|--------|--------|-----|------|
| a-STOP (full) | ✅ Best | ✅ Best | ✅ Best |
| No BC | ↓ 0.680 | ↓ 0.635 | ↑ 0.0294 |
| Single-α Policies | ≈ | ↓ | ↑ |
| Only Belief Score | ↓ | ↓ | ↑ |

> - **BC → PPO 初始化至关重要**：无 BC 导致性能大幅下降。
> - **α-conditioning 更优**：相比多个独立 α 策略，单策略更高效且控制更稳定。
> - **状态丰富度重要**：包含 $b_t$, $E_t$, $z_t$ 等信息优于仅用 $p_t$。

#### （2）预测源替换实验（Table 9）
即使将预测器换为较弱的 **naive predictor**，α-STOP 仍是最佳触发策略，表明其设计具有鲁棒性和通用性。

---

## 4. 关键结论和发现

### 主要发现
1. **失败证据是稀疏且延迟的**：大多数对话轮次是常规进展，真正指示失败的关键 turn 很少且出现在后期。
2. **传统广播标签法不适用于语言交互**：会导致过早触发和较低的最大准确率。
3. **MIL + 融合机制能有效提取稀疏证据**：注意力预测器显著提升了预测质量。
4. **α-STOP 实现了高效的推理时控制**：单模型即可覆盖广泛的操作点，极大降低了部署成本。
5. **解耦设计优于端到端学习**：先学预测再学触发，比联合训练更有效。

### 局限性
- **数据范围有限**：虽涵盖多种任务，但不能保证所有多轮 NLP 场景都符合“稀疏证据”模式。
- **依赖预训练编码器**：性能受限于 `Qwen3-Embedding` 等模型的质量与上下文窗口。
- **LLM 判别器存在偏见**：如位置偏见（position bias），影响其可靠性。
- **人工标注不可靠**：Turn-level 相关性评分由 LLM 回顾生成，非因果标注。

### 未来工作方向
- 探索更多类型的稀疏监督信号（如用户反馈、情绪变化）。
- 扩展至多类别或多模态轨迹的早期预警。
- 研究如何自动校准 $\alpha$ 参数以适应动态环境。
- 结合主动学习，在线收集高质量失败案例以增强训练。

> **总结**：本文提出了一个实用、高效且性能优越的弱监督早期失败预警框架，通过显式建模稀疏证据和引入可控制的触发策略，解决了现有方法在准确性、灵活性与成本之间的矛盾，为交互式 AI 系统的安全与可靠性提供了有力工具。

</details>

---

### 12. [CHASE: Adversarial Red-Blue Teaming for Improving LLM Safety using Reinforcement Learning](https://arxiv.org/abs/2606.05523)

**Authors**: Rahul Markasserithodi, Aditya Joshi, Yuekang Li, Ishmanbir Singh, Chris Yoo, Alan Niu  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.05523v1  

#### Abstract
Despite advances in safety alignment, prompt-rewriting attacks such as persona modulation, fictional framing and persuasion-based reformulation, can bypass safety filters even on frontier models. Existing defenses either rely on non-scalable human curation or white-box optimisation that overfits to ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《CHASE: Adversarial Red-Blue Teaming for Improving LLM Safety using Reinforcement Learning》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大语言模型（LLM）的安全对齐方法在面对**自适应黑盒攻击者**时表现出“**适应性差距**”（adaptive gap）：
- 静态防御机制（如 RLHF、SFT）容易被新型攻击绕过；
- 现有红蓝对抗训练依赖固定模板或模仿学习，导致攻击分布狭窄，泛化能力差；
- 防御模型常出现**过度拒绝**（over-refusal），损害有用性。

### 🚀 提出的新方法：CHASE 框架
提出 **CHASE**（Co-evolutionary Hardening through Adversarial Safety-Escalation），一个闭环的、基于强化学习的红蓝对抗框架：

#### 核心设计思想：
- **双代理共进化**：Black-box 攻击者（Attacker）与安全对齐的防御者（Defender）在对抗中共同演化。
- **无模板攻击生成**：攻击者不使用任何 jailbreak 模板或先验攻击策略，完全通过奖励驱动探索发现攻击方式。
- **双向 GRPO 训练**：攻击者和防御者均使用 **Group Relative Policy Optimization (GRPO)** 进行 on-policy 优化。

### 🔍 创新点
1. **模板无关的攻击发现机制**
   - 攻击者从零开始，仅通过 multiplicative reward 探索有效重写策略，避免模仿特定攻击家族。
   - 发现了可跨攻击家族迁移的**潜在攻击原语**（latent attack primitives），如虚构场景、角色扮演等。

2. **乘法奖励分解**（Multiplicative Reward Decomposition）
   $$
   R = S_{\text{bypass}} \times I_{\text{intent}}
   $$
   - $S_{\text{bypass}}$：绕过成功率（是否成功诱导非拒绝响应）；
   - $I_{\text{intent}}$：意图保真度（是否保留原始有害意图）；
   - **优势**：防止“意图漂移”（intent drift）和“过度净化”（over-sanitization）等 reward hacking 行为。

3. **两阶段防御强化流程**
   - 第一阶段：用 GRPO 在攻击样本上进行探索性训练，提升拒绝能力；
   - 第二阶段：采用 **rejection-sampled SFT** 巩固高分拒绝行为，确保一致性；
   - 并平衡加入良性数据，防止过度拒绝。

4. **强泛化性防御**
   - 防御者仅在 CHASE 自产攻击上训练，却能显著抵御五种未见过的 SOTA 黑盒攻击。

### ⚖️ 相比现有方法的优势
| 方面 | 现有方法（如 MART, RedHit, R2D2） | CHASE |
|------|-------------------------------|--------|
| 攻击多样性 | 固定模板或搜索空间有限 | 无模板，RL 自主发现多样化攻击 |
| 泛化能力 | 多数只在同族攻击上有效 | 跨机制攻击家族泛化 |
| 奖励设计 | 单目标易 reward hacking | 乘法奖励抑制意图漂移 |
| 防御稳定性 | 易因高方差无法收敛 | GRPO + SFT 双阶段稳定训练 |
| 有用性保持 | 常见 over-refusal 问题 | 0% false refusal on benign prompts |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **BeaverTails**：100 条有害提示（`is_safe=False`），用于评估；
- **JailbreakBench (JBB)**：100 条行为提示，均匀覆盖多个类别；
- **训练集**：从 BeaverTails 和 JBB 中采样约 1,000 条有害提示，经 zero-shot 分类过滤后用于 GRPO 训练；
- **Alpaca**：100 条良性提示，用于测试 false refusal；
- **XSTest**：250 安全提示 + 200 不安全对比组，用于分析拒绝模式。

### 🧪 实验设置
- **基础模型**：
  - 攻击者：NousResearch Hermes-4-14B
  - 防御者 & 目标模型：Llama-3.1-8B-Instruct
- **参数高效微调**：LoRA（Low-Rank Adaptation）
- **对抗循环**：交替更新攻击者与防御者，共 T=3 轮迭代。

### 📊 评估指标
使用 **StrongREJECT** 框架提供的复合指标：
- **Refusal Rate (r ∈ {0,1})**：是否拒绝；
- **Convincingness (c ∈ [1,5])**：回应流畅性和可信度；
- **Specificity (s ∈ [1,5])**：提供危害信息的具体程度；
- **StrongREJECT Score (SR)**：
  $$
  SR = (1 - r) \cdot \frac{(c + s - 2)}{8} \in [0,1]
  $$
  - 综合衡量攻击有效性，优于传统 ASR（Attack Success Rate）；
  - 低质量绕过（如空洞回应）得分极低。

### 🔁 基线方法对比
- **五类 SOTA 黑盒攻击作为测试基准**（未参与训练）：
  1. **PAIR**：多轮迭代优化攻击；
  2. **TAP**：基于说服心理学的树状搜索；
  3. **AutoDAN**：遗传算法生成隐蔽提示；
  4. **PAP**：人格调制攻击；
  5. **Translation Attack**：跨语言低资源攻击（Zulu, Hmong 等）；
- **消融对照**：将 CHASE 攻击替换为 PAIR 攻击产物，验证攻击分布的影响。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table 1）

| 攻击类型 | 数据集 | Base Model SR | CHASE Defender SR | ΔSR | 减少百分比 |
|---------|--------|----------------|--------------------|-----|------------|
| PAIR    | BT     | 0.629          | 0.451              | -0.178 | -28.3%     |
| PAIR    | JBB    | 0.705          | 0.565              | -0.140 | -19.9%     |
| TAP     | BT     | 0.575          | 0.310              | -0.265 | -46.1%     |
| TAP     | JBB    | 0.501          | 0.229              | -0.272 | -54.3%     |
| AutoDAN | BT     | 0.618          | 0.304              | -0.314 | -50.8%     |
| AutoDAN | JBB    | 0.509          | 0.128              | -0.381 | -74.9%     |
| PAP     | BT     | 0.472          | 0.241              | -0.231 | -48.9%     |
| PAP     | JBB    | 0.646          | 0.298              | -0.348 | -53.9%     |
| Translation | BT | 0.091          | 0.079              | -0.012 | -13.2%     |
| Translation | JBB| 0.040          | 0.115              | +0.075 | +187.5%    |
| **平均** | ——   | **0.479**      | **0.272**          | **-0.207** | **-43.2%** |

> ✅ **核心成果**：CHASE 防御者使平均 StrongREJECT 得分下降 **43.2%**，且在所有攻击下实现 **0% ASR on direct misuse, PAIR/GCG transfer**。

### 🛠️ 其他重要结果
- **0% False Refusal**：在 100 条 Alpaca 良性提示上无误拒；
- **MT-Bench 成本**：总分从 7.42 降至 5.50（Δ=-1.92），主要影响推理类任务（Math, Humanities），编码与抽取基本保留；
- **XSTest 拒绝模式分析**（Table 2）：
  - 对**事实性/定义类**问题保持 >60% 合规；
  - 对**虚构/角色扮演类**问题拒绝率高达 70–92%，表明其学会了识别常见 jailbreak framing。

### 🔍 消融实验结果（Figure 3 & Table 4）
- **对比 PAIR-Artifact Defender**：
  - 在 PAIR 攻击上表现优异（SR ↓87.3% on BT）；
  - 但在其他攻击上几乎失效（如 AutoDAN ↑12.8%）；
  - 平均 SR 仅降低 28.6%，远低于 CHASE 的 43.2%；
- **结论**：CHASE 的泛化能力来源于**攻击分布本身的质量**，而非训练流程设计。

### 🔄 多轮迭代效果（Appendix G）
- 经过 3 轮 co-evolution：
  - 平均 SR 进一步降至 **0.115**（相对减少 **76.0%**）；
  - 攻击者 ASR 上升但 bypass 质量下降（convincingness 从 4.79→2.38），说明 multiplicative reward 有效约束了退化解。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **模板自由的 RL 探索能发现通用攻击原语**
   - 无需预设攻击形式，GRPO 自动收敛到“虚构情境”、“角色扮演”等高频 jailbreak framing；
   - 这些结构广泛存在于 PAIR、TAP、AutoDAN 等不同机制攻击中，构成“攻击共性”。

2. **攻击分布决定防御泛化上限**
   - 使用单一攻击家族（如 PAIR）训练的防御器无法泛化；
   - CHASE 的成功源于其生成的**多样化、高质量、高保真的攻击分布**。

3. **拒绝是有代价的，但代价是可解释的**
   - CHASE 的 over-refusal 集中在**虚构/角色扮演类提示**；
   - 而这类提示正是现实中 98% jailbreak 的载体（Liu et al., 2023b）；
   - 因此，“宁愿错拒也不被 jailbreak” 是一种合理权衡。

4. **两阶段训练（GRPO + rejection-sampled SFT）至关重要**
   - GRPO 解决探索问题，SFT 解决一致性问题；
   - 结合 rejection sampling 和合成目标，解决了“无正例难监督”的困境。

### ⚠️ 局限性
- **模型规模限制**：仅在 Llama-3.1-8B 上验证，未扩展至更大模型；
- **跨语言安全性弱化**：LoRA 微调轻微削弱了低资源语言的安全对齐（Translation 攻击 SR 上升）；
- **推理能力受损**：MT-Bench 下降 -1.92，尤其影响数学与人文推理；
- **依赖 LLM Judge**：未使用人工标注进行最终评估；
- **攻击者未公开**：出于伦理考虑，仅发布防御模型和代码。

### 🔮 未来工作方向
- 扩展至多模态模型和更大规模架构；
- 引入动态 benign-to-harmful 比例调整，进一步优化 helpfulness-safety 权衡；
- 探索更细粒度的 intent fidelity 判断机制；
- 将 latent attack primitives 形式化建模，构建“攻击语法”理论；
- 开发轻量化版本以支持边缘部署。

---

## 总结
CHASE 是首个证明**模板自由、RL驱动的对抗训练**可以实现**跨攻击家族泛化防御**的工作。它揭示了一个深刻洞见：  
> **LLM 安全的本质不是对抗某种攻击技术，而是识别并阻断其共享的语义框架。**

通过让攻击者“自己学会怎么攻”，再让防御者“学会怎么防”，CHASE 构建了一条通往更具鲁棒性和泛化性的 LLM 安全路径。

</details>

---

### 13. [GenAutoML: An Agentic Framework for Dynamic Architecture Generation and Optimization in Time-Series Analysis](https://arxiv.org/abs/2606.05860)

**Authors**: Oleeviya Babu Poikarayil, C\'edric Schockaert, Abdulrahman Nahhas, Christian Daase, Mursal Dawodi, Jawid Ahmad Baktash  
**Category**: cs.LG  
**Published**: 2026-06-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.05860v1  

#### Abstract
Designing neural architectures for time-series forecasting and anomaly detection remains a resource-intensive task that often requires substantial domain expertise. Traditional Automated Machine Learning (AutoML) systems typically rely on static, predefined search spaces, limiting their ability to a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GenAutoML: An Agentic Framework for Dynamic Architecture Generation and Optimization in Time-Series Analysis

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 AutoML 和 Neural Architecture Search (NAS) 在时间序列分析中面临以下挑战：
- **静态搜索空间**：现有方法依赖预定义的架构模板（如固定层数、连接方式），缺乏“语义创造力”来设计全新的结构组件。
- **部署延迟高**：大规模 Foundation Models（如 Chronos、MOIRAI）虽然精度高，但参数量巨大（>100M），推理延迟高达数百毫秒，无法满足边缘设备（Edge AI）的实时性要求。
- **非平稳性问题**：工业时序数据（如电力变压器温度）具有高度非平稳性（non-stationary），导致模型训练不稳定、梯度爆炸。

### 🚀 提出的新方法与创新点
GenAutoML 是一个基于 **Agentic Framework** 的动态神经网络架构生成与优化框架，其核心创新包括：

#### （1）**Agentic Neural Synthesis with Sandboxed Reflection（语义接口）**
- 利用 **Large Language Model (LLM)** 作为“神经架构师”，将自然语言需求（如“设计轻量级 Inception 模型”）直接转化为可执行的 PyTorch 代码。
- 引入 **Sandboxed Reflection Loop**：在隔离环境中模拟前向传播，捕获维度不匹配、通道错误等 Python traceback 错误，并反馈给 LLM 自动修复，实现闭环调试。

#### （2）**Runtime Architectural Injection（运行时架构注入）**
- 支持 **Just-In-Time (JIT) 动态加载**，通过 `importlib.reload` 实现“热插拔”（hot-swap），无需重启即可将新生成的 `nn.Module` 注入到活跃的 Optuna 超参搜索流程中。
- 配备 **Signature-Aware Runtime** 和 **Shape-Agnostic Projection Head**，自动处理未知函数签名和输出维度不一致问题。

#### （3）**Dynamic Reversible Instance Normalization (Dyn-RevIN)（安全护栏）**
- 将 LLM 生成的不可信架构逻辑封装在 **Dyn-RevIN** 层内，强制输入输出保持统计平稳性，防止因结构不合理导致的数值不稳定或梯度爆炸。
- 数学上解耦了“架构设计”与“数值稳定性”的责任边界。

#### （4）**End-to-End Conversational Pipeline（端到端对话流程）**
- 用户可通过自然语言上传数据、进行探索性数据分析（EDA）、提出建模需求，系统自动生成并验证模型代码，形成完整的人机协作闭环。

### 🔍 相比现有方法的优势
| 维度 | 传统 NAS / AutoML | 时间序列 Foundation Models | GenAutoML |
|------|------------------|----------------------------|----------|
| 架构灵活性 | 固定搜索空间 | 不适用（零样本） | 动态创造全新拓扑 |
| 推理延迟 | 中等 | 极高（~1s） | 极低（<0.01ms） |
| 边缘部署能力 | 一般 | 差 | 优秀 |
| 训练稳定性 | 依赖人工调参 | 一般 | 高（由 Dyn-RevIN 保障） |
| 搜索成本 | 极高（GPU 天级别） | 零搜索成本 | 极低（分钟级） |

---

## 2. 核心实验方法和设置

### 📊 数据集
在三个标准多变量时间序列基准上进行评估：
- **ETTh1**：电力变压器油温数据（小时粒度），强季节性和负载波动，高度非平稳。
- **ETTm1**：同系列但为分钟级采样，更高频噪声。
- **Weather**：本地气候数据（气温、湿度、风速等），周期性强但存在随机降水事件。

所有任务采用严格的时间顺序划分（70%/10%/20%），避免数据泄露。

### ⚙️ 实验设置
- **预测任务**：Lookback=96, Horizon=96
- **异常检测任务**：Lookback=60, Horizon=10
- **训练配置**：
  - 优化器：Adam（lr=1e-3, weight decay=1e-6）
  - 学习率调度：ReduceLROnPlateau（patience=3）
  - 批大小：64
  - 早停机制：验证损失连续5轮未改善则停止
  - 损失函数：预测用 MAE，重建类模型用 MSE

### 📈 评估指标
| 任务 | 主要指标 | 补充指标 |
|------|--------|---------|
| 预测 | MAE, RMSE | — |
| 异常检测 | Clean MSE, Anomalous MSE | Discrimination Gap = Anomalous MSE / Clean MSE（越大越好） |
| 效率 | Inference Latency (ms), 参数量 | Search Time (GPU tuning time) |

### 🆚 基线方法对比
| 类别 | 模型 |
|------|------|
| **零样本 Foundation Model** | Chronos-T5-Mini |
| **线性模型** | DLinear |
| **经典深度模型** | LSTM, Conv1D |
| **现代 SOTA Transformer** | iTransformer, TimesNet, CrossFormer |
| **合成模型（本工作）** | ResNet, Inception, BiGRU, MSCNN, HybridCNNLSTM, **WaveInterferenceNet** |

---

## 3. 主要实验结果和性能指标

### 📉 预测性能（MAE, RMSE）
见 **Table 1**：

| Model | ETTh1 (MAE) | ETTm1 (MAE) | Weather (MAE) |
|-------|-------------|-------------|---------------|
| Chronos-T5-Mini (Zero-Shot) | **0.166** | – | – |
| DLinear | 0.989 | 0.515 | 1.783 |
| ResNet (Ours) | 1.156 | **0.532** | 2.696 |
| Inception (Ours) | 1.280 | 0.546 | 2.811 |
| WaveInterferenceNet (Case Study) | 1.137 | – | – |

> 💡 观察：
> - 尽管 Chronos 精度最高，但其推理延迟达 **987.93ms**。
> - 合成模型（如 ResNet）在 ETTm1 上接近 DLinear 性能，且显著优于其他 Transformer。
> - 在 Weather 数据上，复杂 Transformer（TimesNet, CrossFormer）表现差，而轻量卷积模型更鲁棒。

### 🚨 异常检测性能（Discrimination Gap）
见 **Table 2**：

| Model | ETTh1 (Gap) | ETTm1 (Gap) | Weather (Gap) |
|-------|-------------|-------------|---------------|
| TimesNet | ~33x | ~29,191x | ~5,035x |
| DLinear | ~8x | ~138x | ~22x |
| ResNetBlock (Ours) | **~265x** | **~431x** | **~921x** |

> 💡 观察：
> - ResNetBlock 在所有数据集上均表现出最强的异常敏感性，尤其在 ETTh1 上是 DLinear 的 **33 倍**。
> - BiGRU 在预测中表现好，但在异常检测中较弱，说明不同任务需专用架构。

### ⚡ 推理效率与搜索成本（Table 3）
| Task | Model | Search Time (5 trials) | Inference Latency | Parameters |
|------|-------|------------------------|--------------------|------------|
| Forecasting | DLinear | 2m07s | **<0.01ms** | 52K |
| Forecasting | Chronos-T5-Mini | N/A | 987.93ms | 20M |
| Forecasting | **WaveInterferenceNet** | **2m01s** | **<0.01ms** | 829K |
| Anomaly Det. | TCN | 7m38s | 0.60ms | 33K |
| Anomaly Det. | **WaveInterferenceNet** | **2m22s** | **<0.01–0.10ms** | 829K |

> ✅ 关键优势：
> - **WaveInterferenceNet** 实现了 **<0.01ms 推理延迟**，相比 Chronos 快约 **10万倍**。
> - 搜索总耗时仅 **约2分钟**，远低于传统 NAS 的“GPU天”级别开销。

### 🔬 消融实验结果（Ablation Study）

#### （1）Dyn-RevIN 的影响（Table 5）
| Dataset | BiGRU (MAE) ↑ | ResNetBlock (Gap) ↓ |
|--------|----------------|---------------------|
| | With RevIN | Without | With | Without |
| ETTh1 | 1.251 | 2.435 (**↑94%**) | ~265.6x | ~52.5x |
| ETTm1 | 0.455 | 2.541 (**↑450%**) | ~431.1x | ~344.8x |
| Weather | 3.787 | **1.426** | ~921.4x | **~1146.7x** |

> 🔍 发现：
> - 对于非平稳工业数据（ETTh1/ETTm1），**启用 Dyn-RevIN 显著提升稳定性和性能**。
> - 对于周期性强的气象数据，**禁用 RevIN 反而更好**，因为它可能抑制真实的幅度变化信号。
> → 表明没有“银弹”，必须动态选择是否启用归一化。

#### （2）初始化稳定性（Table 4）
| Model | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|-------|--------|--------|--------|-----------|
| DLinear | 1.9528 | 0.3321 | 0.3491 | **0.8780±0.7600** |
| WaveInterferenceNet | 2.1485 | 2.1485 | 2.1485 | **2.1485±0.0000** |

> ✅ 结论：
> - WaveInterferenceNet 具有 **零方差收敛特性**，适合对确定性要求高的安全关键场景（safety-critical deployments）。
> - 虽然平均误差略高，但其部署可靠性远超传统模型。

#### （3）Sandboxed Reflection 成功率
- 所有架构最终都通过反射循环成功修复，**验证成功率 100%**。
- 复杂模型（如 WaveInterferenceNet）需要最多 **5 次迭代** 才能解决张量广播和投影维度问题。
- 日志显示 LLM 能理解 `RuntimeError` 并自主引入 `permute`, `reshape`, `projection` 层进行修正。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **GenAutoML 成功实现了“Agentic Neural Engineering”范式转变**：
   - 从“搜索已有架构”转向“自主创造新拓扑”，具备真正的语义创造力。
   - LLM 可以基于抽象物理类比（如“波干涉”）生成有效 PyTorch 模块。

2. **在边缘部署场景下，效率与稳定性优先于极致精度**：
   - WaveInterferenceNet 以微小精度代价换取 **<0.01ms 推理延迟** 和 **零方差训练行为**，非常适合资源受限的工业边缘设备。

3. **Dyn-RevIN 是关键的安全护栏**：
   - 它有效隔离了不可信的 LLM 生成逻辑与生产系统的数值稳定性需求。
   - 但其使用应根据数据特性和任务目标动态决策，不能盲目开启。

4. **Sandboxed Reflection Loop 是系统可靠性的基石**：
   - 实现了全自动代码调试闭环，使 LLM 能够“自我纠正”张量维度错误。
   - 使得整个框架可在无人干预下持续运行。

### ⚠️ 局限性
1. **生成延迟较高**：当前依赖云端 LLM API（如 Llama 3-70B），单次架构生成需 30–60 秒，可能成为实时响应瓶颈。
2. **搜索空间仍受提示工程限制**：LLM 的创造性受限于 prompt 设计质量。
3. **仅支持数值时间序列**：尚未整合文本日志、维护记录等多模态上下文信息。

### 🔮 未来工作方向
1. **本地化小型 LLM**：部署专用于代码生成的小型、高效 LLM，降低生成延迟。
2. **Hardware-Aware Synthesis**：在 prompt 中加入硬件约束（如内存、算力预算），实现“Green AI”导向的设计。
3. **多模态 Prompt 支持**：融合传感器数据 + 文本报告 + 操作日志，构建更智能的工业诊断代理。
4. **动态启用/绕过 Dyn-RevIN**：结合元学习策略，自动判断何时应用统计硬化。

---

> 🏁 **总结一句话**：  
> **GenAutoML 不是追求 SOTA 精度的工具，而是面向边缘部署的“确定性神经架构合成引擎”——它牺牲一点点预测能力，换来了百万倍的速度提升、零方差训练行为和全自动闭环调试能力，真正推动 AutoML 进入工业现实世界。**

</details>

---

### 14. [StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis](https://arxiv.org/abs/2606.04246)

**Authors**: Prashanth Vijayaraghavan, Apoorva Nitsure, Luyao Shi, Ehsan Degan, Vandana Mukherjee  
**Category**: cs.AI  
**Published**: 2026-06-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.04246v1  

#### Abstract
Automatic generation of RTL code for digital hardware designs remains challenging due to long-horizon reasoning, multi-step dependencies, and strict correctness constraints in Verilog and VHDL. We present StepPRM-RTL, a novel framework that combines stepwise trajectory modeling, process-reward model...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在 **RTL（Register-Transfer Level）代码生成** 领域，尽管已有基于大语言模型（LLM）的方法尝试自动生成 Verilog 和 VHDL 代码，但仍面临以下挑战：

- **长程依赖（long-horizon reasoning）**：RTL 设计需要多步、连贯的逻辑决策（如状态机设计、复位逻辑、时钟使能协调），传统模型难以维持一致的推理链。
- **中间步骤缺乏监督**：现有方法通常只在最终输出上进行功能验证（outcome-based），无法对中间设计决策提供反馈，导致错误累积。
- **语义粒度不匹配**：现有的 **Process Reward Model (PRM)** 多在 token 级别打分，而 RTL 的关键决策往往跨越多个语句或模块，token 级奖励不稳定且语义意义弱。

---

### 🚀 提出了什么新方法或新思路

本文提出 **StepPRM-RTL**，一个全新的 LLM 微调框架，用于增强 RTL 合成的质量与可解释性。其核心创新包括：

#### （1）**Step-Level Process Reward Modeling (StepPRM)**  
首次将 PRM 应用于 **语义有意义的“设计步骤”级别**（而非 token 级）。每个步骤包含：
- 自然语言 **rationale**（设计理由）
- 对应的 **code edit**（代码修改）

StepPRM 学习为这些完整步骤分配语义奖励，实现更稳定、硬件语义对齐的信用分配（credit assignment）。

#### （2）**PRM-Guided MCTS 探索多样化高质量轨迹**  
引入 **Monte Carlo Tree Search (MCTS)** 进行结构化搜索，利用 StepPRM 提供的 step-level 奖励引导探索：
- 在部分实现上进行 rollout
- 利用 StepPRM 打分评估路径质量
- 收集超越人工标注轨迹的高价值推理路径

这缓解了监督学习中的“引导偏差”（bootstrap bias）。

#### （3）**Retrieval-Augmented Fine-Tuning (RAFT) with Reward Weighting**  
结合 RAFT 框架，在微调阶段：
- 检索相似设计的历史轨迹作为上下文
- 使用 StepPRM 分数加权轨迹重要性，优先学习高奖励路径

从而实现 **检索增强 + 奖励引导** 的联合优化。

#### （4）**迭代训练闭环**  
构建了一个闭环流程：
1. 从规范解提取 stepwise 轨迹 → 初始化 StepPRM 和策略模型
2. StepPRM 引导 MCTS 探索新轨迹
3. 用新轨迹更新 StepPRM 和策略模型（via RAFT）
4. 循环迭代直至收敛

该机制实现了 **policy 和 reward model 的协同进化**。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | StepPRM-RTL |
|------|----------|-------------|
| 监督粒度 | Token-level 或 outcome-only | **Step-level 语义监督** |
| 推理能力 | 易受长程依赖影响 | MCTS + StepPRM 提升长程一致性 |
| 可解释性 | 黑箱生成 | 每步附 rationale，支持 traceable reasoning |
| 泛化性 | 依赖大量标注数据 | 利用检索 + 探索生成多样高质量轨迹 |
| 性能表现 | 功能正确率有限 | 显著提升 Pass@1 和 reasoning fidelity |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **Verilog-Eval** [11]：156 个来自 HDLBits 的 spec-to-Verilog 任务，配备自检 testbench。
- **VHDL-Eval** [18]：202 个翻译自 Verilog-Eval 的 spec-to-VHDL 任务，同样带验证环境。
- **内部 RTL-IR Corpus**：用于训练初始模型，包含 spec、代码及摘要，并人工/LLM 构造 stepwise 轨迹。

> 注：Verilog-Eval 和 VHDL-Eval 严格保留用于评估，避免数据泄露。

---

### 🎯 实验设置与评估指标

#### 主要评估指标：

| 指标 | 定义 |
|------|------|
| **Pass@1** | 第一次生成的 RTL 是否通过官方 testbench 编译与仿真验证（功能正确性） |
| **Reasoning Fidelity (%)** | 使用 LLM judge 对比生成的 reasoning steps 与标准轨迹的一致性（语义对齐程度） |

#### 模型架构与实现细节：
- 基础模型：**Qwen3-8B-Instruct**
- StepPRM：基于相同 backbone 的回归头模型
- 检索模型：**Qwen3-Embedding-4B**，经 contrastive learning 微调
- MCTS 设置：每 spec 50 次模拟，rollout 深度 10，探索常数 `Cuct=1.5`
- 验证工具：Icarus Verilog（Verilog）、GHDL+VUnit（VHDL）

---

### ⚔️ 基线方法对比

| 类型 | 基线模型 |
|------|--------|
| **Prompt-based** | Vanilla Prompting (GPT-4o), CoDes (GPT-4o) |
| **Fine-tuning based** | RTLCoder (Mistral), CodeV (CodeQwen), VeriThoughts |
| **RAG-enhanced** | RAG-CodeBERT (GPT-4o), RAG-FT (GPT-4o) |
| **消融变体** | No MCTS, Supervised RAFT Only, No PRM |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 2）

| Model | Pass@1 (Verilog) | Pass@1 (VHDL) | Reasoning Fidelity (Verilog) | Reasoning Fidelity (VHDL) |
|-------|------------------|---------------|-------------------------------|----------------------------|
| **StepPRM-RTL (Full)** | **0.857** | **0.786** | **82.5%** | **80.2%** |
| RAG-FT (GPT-4o) | 0.719 | 0.531 | — | — |
| VeriThoughts | 0.755 | — | 60.4% | — |
| CoDes (GPT-4o) | 0.602 | 0.348 | — | — |
| Vanilla Prompting | 0.543 | 0.285 | — | — |

> ✅ StepPRM-RTL 在 **所有指标上全面领先**，相比最强基线 RAG-FT 提升超过 **10个百分点以上**。

---

### 🔍 消融实验结果（Ablation Studies）

| 变体 | Pass@1 (Verilog ↓) | Reasoning Fidelity (Verilog ↓) | 结论 |
|------|--------------------|-------------------------------|------|
| **No MCTS (Sampling-Only)** | 0.810 (-4.7pp) | 78.2% (-4.3pp) | MCTS 显著提升探索效率与路径质量 |
| **Supervised RAFT Only** | 0.796 (-6.1pp) | 75.3% (-7.2pp) | 奖励加权对轨迹选择至关重要 |
| **No PRM (Outcome-only)** | 0.781 (-7.6pp) | 73.1% (-9.4pp) | Step-level 奖励远优于稀疏 outcome reward |

> ✅ 所有组件均不可或缺，组合使用带来最大增益。

---

### 📈 超参数敏感性分析（Hyperparameter Sensitivity）

- **MCTS Simulation Count (Nsim)**：
  - 性能随模拟次数增加而上升，**Nsim=15 即接近最优**（Verilog 0.84+），更高开销收益递减。
- **Reward Shaping Weight (λsh)**：
  - 最优值在 **λsh=0.3**，平衡 canonical 监督与结构对齐。
  - 过高（≥0.5）会抑制合理创新。

> 表明框架具有良好的鲁棒性和实用性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Step-level supervision 是解决 RTL 长程推理的关键**：相比 outcome-only 或 token-level PRM，step-level 奖励显著改善中间决策质量。
2. **MCTS + PRM 实现高效探索**：结构化搜索能有效避开无效路径，发现高质量替代设计方案。
3. **Reward-weighted RAFT 提升泛化能力**：结合检索与奖励加权，使模型内化优秀设计模式。
4. **框架通用性强**：在 Verilog 和 VHDL 上均取得显著提升，表明跨语言适用性。

---

### ⚠️ 方法的局限性

1. **依赖高质量 stepwise 轨迹构造**：初始轨迹需由 LLM 或专家生成，存在噪声风险。
2. **计算成本较高**：MCTS 搜索和迭代训练带来额外开销，不适合低资源场景。
3. **当前限于单文件模块级设计**：尚未扩展到多文件、层次化系统级综合。
4. **形式化验证未完全集成**：目前仅用轻量级语法检查，未来可融合 formal verification。

---

### 🔮 未来工作方向

1. **扩展至 hierarchical/multi-module design**：支持跨文件依赖建模。
2. **深度融合 formal verification into StepPRM**：利用等价性检查、断言验证提供更强语义信号。
3. **Cross-architecture transfer of reasoning trajectories**：在 Verilog 和 VHDL 间迁移推理知识。
4. **端到端自动化 EDA 流程集成**：将 StepPRM-RTL 接入完整 synthesis → place & route 流水线。

---

## ✅ 总结

**StepPRM-RTL** 是首个将 **step-level process reward modeling** 成功应用于 RTL 合成的框架，通过 **StepPRM + MCTS + RAFT** 的三重协同机制，实现了：
- 更高的 **功能正确率（Pass@1）**
- 更强的 **推理保真度（reasoning fidelity）**
- 更好的 **可解释性与可控性**

它不仅刷新了 Verilog/VHDL 代码生成的 SOTA 表现，也为 AI-assisted hardware design automation 建立了新的范式标准。

</details>

---

### 15. [MIRAGE: Mobile Agents with Implicit Reasoning and Generative World Models](https://arxiv.org/abs/2606.04627)

**Authors**: Zhichao Yang, Yuanze Hu, Haojie Hao, Longkun Hao, Dongshuo Huang, Hongyu Lin, Gen Li, Lanqing Hong, Yihang Lou, Yan Bai  
**Category**: cs.AI  
**Published**: 2026-06-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.04627v1  

#### Abstract
Mobile agents are increasingly expected to operate everyday applications from screenshots and language goals, where reliable control requires reasoning over screen affordances, multi-step navigation, and future state changes. However, many agents externalize this computation as long textual chains o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**MIRAGE: Mobile Agents with Implicit Reasoning and Generative World Models**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于视觉语言模型（VLM）的移动代理（mobile agents）在执行GUI操作时，通常依赖显式的链式思维（explicit Chain-of-Thought, CoT）推理，即通过生成冗长的文本思考过程来辅助决策。这种做法带来了以下问题：
- **高延迟**：解码大量中间文本显著增加推理延迟；
- **高资源消耗**：占用上下文窗口，增加token成本；
- **部署效率低**：不利于实时交互控制；
- **监督成本高**：需要人工标注详细的推理轨迹。

### 提出的新方法与核心思想
作者提出 **MIRAGE**（Mobile agents with Implicit Reasoning And Generative world modEls），一种将推理完全置于隐空间（latent space）中的新型移动代理框架，其核心创新如下：

#### ✅ 创新点一：**隐式推理（Implicit Reasoning）**
- 将传统的显式文本推理替换为**连续的隐变量（latent slots）**，模型在内部进行“思考”，而无需输出可读的推理文本。
- 推理过程由N个可学习的隐向量表示，位于解码器上下文中，不参与最终输出。

#### ✅ 创新点二：**近似并行隐状态精炼（APLR, Approximate Parallel Latent Refinement）**
- 传统隐式CoT采用串行更新（serial rollout），计算代价高。
- MIRAGE引入**Jacobi风格的并行迭代机制**，在K轮同步前向传播中同时更新所有隐槽，大幅降低训练开销。
- **理论保证**：经过K轮APLR后，前K个隐状态精确匹配串行解，尾部误差有界。

#### ✅ 创新点三：**基于Q-Former的世界模型头（Generative World Model Alignment）**
- 引入轻量级Q-Former模块，将最终隐状态与下一帧截图的视觉特征对齐（stop-gradient）。
- 隐状态不仅用于动作预测，还被鼓励去**预测界面状态变化**，从而具备“前瞻性”能力。
- 联合优化动作交叉熵损失 $L_{ce}$ 和下一帧特征对齐损失 $L_{wm}$，使隐状态兼具**判别性与动态建模能力**。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **推理效率** | 输出token减少75%以上，first-to-last token延迟降低3–5倍 |
| **任务性能** | 在AndroidWorld上成功率提升超10个百分点，优于同规模基线 |
| **部署友好性** | 无显式推理文本输出，更适合移动端实时交互 |
| **训练有效性** | 两阶段课程学习（curriculum learning）稳定迁移显式推理能力至隐空间 |

---

## 2. 核心实验方法和设置

### 数据集
- **AMEX**：包含104K张高分辨率Android截图，标注GUI元素与逐步操作序列。
- **AndroidWorld**：动态真机环境基准，涵盖20个应用中的116个真实任务实例，支持端到端任务完成率评估。
- **AndroidControl**：提供高层/底层指令配对及标准动作序列，用于细粒度评估指令遵循与动作准确性。

### 实验设置
- **主干模型**：
  - Qwen3-VL-4B-Instruct（4B参数）
  - Qwen3-VL-8B-Instruct（8B参数）
- **隐空间配置**：
  - MIRAGE-4B：9个隐槽，3轮APLR
  - MIRAGE-8B：6个隐槽，3轮APLR
- **训练流程（两阶段）**：
  1. **Stage I（显式CoT热身）**：使用完整 `<THOUGHT>` 文本监督训练1个epoch；
  2. **Stage II（隐式CoT微调）**：替换`<THOUGHT>`为隐槽，启用APLR与Q-Former世界模型，联合优化 $L = \lambda L_{ce} + (1-\lambda)L_{wm}$，默认 $\lambda=0.8$。

### 评估指标
| 基准 | 主要指标 |
|------|----------|
| **AndroidWorld** | 任务成功率为主要指标（SR, Success Rate），平均步数和总生成token数 |
| **AndroidControl** | 指令精确匹配率（EM）、动作准确率（Action Accuracy）、每步生成token数 |
| **效率分析** | first-to-last token延迟、推理速度对比 |

### 基线方法对比
- **通用模型**：GPT-4o
- **专用GUI代理**：
  - UI-TARS, MAI-UI, UI-Venus-Navi, ShowUI
  - GUI-R1/UI-R1（强化学习调优）
- **同规模基线**：Qwen3-VL-4B/8B-Instruct（instruction-tuned）

---

## 3. 主要实验结果和性能指标

### AndroidControl 结果（见 Table 1）

| 模型 | 规模 | 低层EM↑ | 动作准确率↑ | 每步tokens↓ |
|------|------|--------|------------|-------------|
| Qwen3-VL-4B-Instruct | 4B | 68.48 | 75.15 | 115.67 |
| **MIRAGE-4B** | 4B | **77.59 (+13.3%)** | **91.09 (+21.2%)** | **18.92 (-83.6%)** |
| Qwen3-VL-8B-Instruct | 8B | 77.66 | 82.54 | 79.86 |
| **MIRAGE-8B** | 8B | **83.75 (+7.8%)** | **94.62 (+14.6%)** | **18.01 (-77.5%)** |

> ✅ MIRAGE在显著压缩输出的同时，大幅提升动作接地精度。

### AndroidWorld 结果（见 Table 2）

| 模型 | 规模 | 成功率↑ | 平均steps | 平均tokens↓ |
|------|------|--------|-----------|--------------|
| Qwen3-VL-4B-Instruct | 4B | 42.9 | 14.3 | 103.0 |
| **MIRAGE-4B** | 4B | **52.6 (+9.7)** | 14.2 | **31.0 (-69.9%)** |
| Qwen3-VL-8B-Instruct | 8B | 47.6 | 12.6 | 108.0 |
| **MIRAGE-8B** | 8B | **57.8 (+10.2)** | 13.7 | **27.0 (-75.0%)** |

> ✅ MIRAGE-8B达到当前最高成功率（57.8%），且生成token仅为基线的约1/4。

### 消融实验（Ablation Study，见 Table 3 & 4）

#### 组件消融（Qwen3-VL-4B, AndroidWorld SR）
| 变体 | Latent CoT | APLR | WM | SR (%) |
|------|-----------|------|----|--------|
| Base | × | × | × | 42.9 |
| Action-only SFT | × | × | × | 31.0 |
| Explicit CoT SFT | × | × | × | 52.6 |
| Latent CoT (serial) | √ | × | × | 50.9 |
| APLR only | √ | √ | × | 48.2 |
| **MIRAGE-4B** | √ | √ | √ | **52.6** |

> 🔍 发现：
> - 移除推理监督（Action-only）导致性能下降 → 显式或隐式推理至关重要；
> - 单独使用APLR效果弱于串行隐式CoT → 尾部误差影响性能；
> - 加入Q-Former世界模型后恢复至显式CoT水平 → 有效补偿尾部误差。

#### 超参敏感性分析
- **APLR轮数**：从2→3轮，MIRAGE-8B SR从46.6→57.8，说明第3轮显著缓解尾部误差；
- **隐槽数量**：从9→3槽，SR从52.6→32.8，表明需足够容量编码观察、理由与预测；
- **损失权重 $\lambda$**：$\lambda=0.1$（强调世界模型）时SR降至48.3，验证世界模型应作为辅助正则项。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **隐式推理可行且高效**：MIRAGE证明可在不牺牲性能的前提下，将显式CoT迁移到隐空间，实现同等推理能力下的极低输出开销。
2. ✅ **APLR是高效的近似方案**：并行精炼机制在少量迭代下即可逼近串行解，尤其前K个隐状态完全一致，适合实际训练。
3. ✅ **世界模型对齐是关键补充**：Q-Former对下一帧视觉特征的预测提供了密集监督，有效纠正APLR带来的尾部误差，提升状态转移理解能力。
4. ✅ **性能与效率双赢**：在多个标准benchmark上，MIRAGE以**3–5倍更低的解码token预算**，实现了**与显式CoT相当甚至更优的任务表现**。

### 方法的局限性
- **依赖监督数据**：仍需高质量的显式推理轨迹进行第一阶段热身；
- **仅限于单步预测**：世界模型仅对齐下一帧，未建模长期动态；
- **特征级建模而非像素生成**：虽避免了解码复杂性，但也限制了对完整界面演化的建模能力；
- **部署安全性考虑不足**：尚未集成权限控制、生物认证绕过等现实安全机制（如call_user需人工介入）。

### 未来工作方向
- 扩展为**多步隐式规划器**，支持跨多个时间步的latent imagination；
- 探索**自监督初始化**，减少对人工标注推理链的依赖；
- 构建**端到端可微分的世界模型**，实现真正的latent simulation；
- 在更多设备平台（iOS、Web）验证泛化能力；
- 引入**不确定性估计**与**主动请求反馈机制**，增强鲁棒性与人机协作。

--- 

> 📌 **一句话总结**：  
> **MIRAGE首次将隐式推理与世界模型结合应用于移动代理，在保持显式CoT级推理能力的同时，实现高达75%以上的token压缩与显著加速，为高效、可部署的智能GUI代理提供了新范式。**

</details>

---

### 16. [Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models](https://arxiv.org/abs/2606.05688)

**Authors**: Hancheol Park, Geonho Lee, Tairen Piao, Tae-Ho Kim  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.05688v1  

#### Abstract
Mixture-of-Experts (MoE) models scale foundation models efficiently by activating only a subset of experts for each token, but their large number of expert parameters still makes quantization essential for practical deployment. Unlike dense models, however, MoE models are sensitive to routing instab...

---

### 17. [Beyond tokens: a unified framework for latent communication in LLM-based multi-agent systems](https://arxiv.org/abs/2606.05711)

**Authors**: Yingzhuo Liu  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.05711v1  

#### Abstract
Multi-agent systems built on large language models (LLMs) have become a prevailing paradigm for tackling complex reasoning, planning, and tool-use tasks. The dominant communication protocol in such systems is natural language: agents exchange messages token-by-token, verbalising their internal reaso...

---

### 18. [ReverseEOL: Improving Training-free Text Embeddings via Text Reversal in Decoder-only LLMs](https://arxiv.org/abs/2606.05858)

**Authors**: Ailiang Lin, Zhuoyun Li, Yusong Wang, Keyu Mao, Kotaro Funakoshi, Manabu Okumura  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.05858v1  

#### Abstract
Recent advances in Large Language Models (LLMs) have opened new avenues for generating training-free text embeddings. However, the causal attention in decoder-only LLMs prevents earlier tokens from attending to future context, leading to biased contextualized representations. In this work, we propos...

---

### 19. [LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents](https://arxiv.org/abs/2606.06087)

**Authors**: Aofan Yu, Chenyu Zhou, Tianyi Xu, Zihan Guo, Rong Shan, Zhihui Fu, Jun Wang, Weiwen Liu, Yong Yu, Weinan Zhang, Jianghao Lin  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.06087v1  

#### Abstract
Agent systems increasingly use textual skills to encode reusable task procedures, but injecting these skills into the prompt at every step incurs substantial context overhead and exposes skill content as plaintext. We present LatentSkill, a framework that converts textual skills into plug-and-play L...

---

### 20. [Cross-Epoch Adaptive Rollout Optimization for RL Post-Training](https://arxiv.org/abs/2606.05606)

**Authors**: Yiming Zong, Yige Wang, Jiashuo Jiang  
**Category**: cs.LG  
**Published**: 2026-06-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.05606v1  

#### Abstract
LLM post-training often relies on reinforcement learning methods that sample multiple rollouts per prompt, yet most existing approaches use a fixed rollout budget for every prompt, despite large differences in the training signal different prompts provide. In this paper, we study adaptive rollout al...

---

### 21. [When Denser Credit Is Not Enough: Evidence-Calibrated Policy Optimization for Long-Horizon LLM Agent Training](https://arxiv.org/abs/2606.05885)

**Authors**: Yuanfan Li, Qi Zhou, Wenjing Duan, Lu Chen  
**Category**: cs.LG  
**Published**: 2026-06-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.05885v1  

#### Abstract
Long-horizon LLM agents require reinforcement learning methods that can assign credit to intermediate decisions under sparse and delayed rewards. Recent group-based methods such as GiGPO improve over GRPO by constructing step-level advantages at repeated anchor states. However, we show that such den...

---

### 22. [On the training of physics-informed neural operators for solving parametric partial differential equations](https://arxiv.org/abs/2606.06164)

**Authors**: Nanxi Chen, Chuanjie Cui, Airong Chen, Sifan Wang, Rujin Ma  
**Category**: cs.LG  
**Published**: 2026-06-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.06164v1  

#### Abstract
Physics-informed neural operators (PINOs) aim to learn solution operators for partial differential equations by using the governing physics as supervision, rather than relying solely on paired input-output simulation data. By incorporating physical constraints into the training objective, PINOs comb...

---

### 23. [LANTERN: Layered Archival and Temporal Episodic Retrieval Network for Long-Context LLM Conversations](https://arxiv.org/abs/2606.05182)

**Authors**: Rahul Subramani  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.05182v1  

#### Abstract
Large language models discard critical details when conversation history is compacted to fit within finite context windows. We present LANTERN (Layered Archival aNd Temporal Episodic Retrieval Network), a lightweight memory layer that proactively archives every conversation turn and restores relevan...

---

### 24. [LLMs Can Leak Training Data But Do They Want To? A Propensity-Aware Evaluation of Memorization in LLMs](https://arxiv.org/abs/2606.06286)

**Authors**: Gianluca Barmina, Peter Schneider-Kamp, Lukas Galke Poech  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.06286v1  

#### Abstract
Large language models can reproduce training data, but existing memorization evaluations mostly measure whether models can be forced to do so, rather than whether they do so under ordinary use. We introduce PropMe, a propensity-aware framework for memorization evaluation that contrasts prefix-based ...

---

### 25. [Latent Reasoning with Normalizing Flows](https://arxiv.org/abs/2606.06447)

**Authors**: Guancheng Tu, Xiangjun Fu, Suhao Yu, Yao Tang, Haoqiang Kang, Lianhui Qin, Yizhe Zhang, Jiatao Gu  
**Category**: cs.CL  
**Published**: 2026-06-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.06447v1  

#### Abstract
Large language models often improve reasoning by generating explicit chain-of-thought (CoT), demonstrating the importance of intermediate computation. However, textual CoT forces this computation through a discrete, serial, and communication-oriented token stream: each reasoning step must be verbali...

---

### 26. [LLM-Based Porting of Optimized C++ to CUDA Through Deoptimization and Reoptimization](https://arxiv.org/abs/2606.06063)

**Authors**: Daichi Mukunoki, Ryo Mikasa, Shunichiro Hayashi, Tetsuya Hoshino, Takahiro Katagiri  
**Category**: cs.DC  
**Published**: 2026-06-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.06063v1  

#### Abstract
When porting high-performance computing (HPC) code from CPU to GPU, CPU-oriented optimizations may obstruct LLM-based CUDA translation. We design and evaluate a Deopt-Reopt workflow that first simplifies the input C++ code and then retranslates and reoptimizes it for CUDA, comparing it against direc...

---

### 27. [CarbonSim: A Lifecycle-Aware Framework for Evaluating Carbon Tradeoffs in Hardware Upgrade Decisions](https://arxiv.org/abs/2606.06438)

**Authors**: Kartik Hans, Kaiwen Zhao, Stephen Lee  
**Category**: cs.DC  
**Published**: 2026-06-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.06438v1  

#### Abstract
As the demand for information and communication technologies (ICT) continues to rise, the environmental impact of computing systems is becoming an increasingly critical concern. Although newer hardware often improves performance and energy efficiency, these gains do not always offset the carbon cost...

---

### 28. [A Sliced-Wasserstein Framework on Correlation Matrices for EEG Decoding](https://arxiv.org/abs/2606.06104)

**Authors**: Chen Hu, Rui Wang, Jiale Zhou, Jingjun Yi, Shaocheng Jin, Yidong Song, Yefeng Zheng  
**Category**: cs.LG  
**Published**: 2026-06-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.06104v1  

#### Abstract
Electroencephalography (EEG) offers noninvasive, millisecond resolution recordings of neuronal activity and is widely used in neuroscience and healthcare. Many EEG decoding pipelines rely on covariance descriptors for their robustness to noise, but such representations are sensitive to channel-wise ...

---

### 29. [Causal Atlases from Entropic Inference: Bayesian Networks beyond Optimal DAGs](https://arxiv.org/abs/2606.06440)

**Authors**: Hazhir Aliahmadi, Irina Babayan, Greg van Anders  
**Category**: cs.LG  
**Published**: 2026-06-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.06440v1  

#### Abstract
Data-driven causal relationship identification is pertinent to advancing understanding of complex systems both within and beyond science. Bayesian networks offer a probabilistic method for modelling generic causal relationships via directed acyclic graphs (DAGs). However, typical techniques for cons...

---

### 30. [MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation](https://arxiv.org/abs/2606.04513)

**Authors**: Deguo Xia, Zihan Li, Haochen Zhao, Dong Xie, Yuyao Kong, Xiyan Liu, Jizhou Huang, Mengmeng Yang, Diange Yang  
**Category**: cs.AI  
**Published**: 2026-06-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.04513v1  

#### Abstract
Lane-level maps are critical infrastructure for autonomous driving and lane-level navigation, yet constructing and maintaining standardized lane networks for hundreds of cities remains highly labor-intensive. Recent end-to-end vectorized mapping methods can predict lane geometry and topology directl...

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

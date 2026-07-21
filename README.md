# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-21 08:03:10 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels](https://arxiv.org/abs/2607.18002)

**Authors**: Bingyang Wu, Chao Jin, Zili Zhang, Xinming Wei, Yinmin Zhong, Ruidong Zhu, Chengxu Yang, Xin Jin, Yuliang Liu  
**Category**: cs.DC  
**Published**: 2026-07-21  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2607.18002v1  

#### Abstract
LLMs scale Mixture-of-Experts (MoE) parameters for superior intelligence, but massive weights and dynamic computation impede efficient serving. Existing instance-level prefill-decode disaggregation isolates the phases on separate full-model replicas. As MoE weights grow, each instance may span tens ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ExpertPlex**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代大规模语言模型（LLMs）广泛采用 **Mixture-of-Experts (MoE)** 架构以提升智能能力，但其服务面临以下挑战：
- **资源分配粗粒度**：现有的 **instance-level prefill-decode disaggregation (PDD)** 需要为 prefill 和 decode 阶段分别部署完整的模型副本，导致大量重复存储 MoE 权重（占模型参数的 95% 以上），浪费显存并挤占 KV Cache。
- **动态负载不匹配**：MoE 的路由机制导致各层专家激活数量动态变化，而现有方案无法细粒度调度计算资源，造成 **head-of-line blocking** 或 **资源空泡（resource bubbles）**。
- **通信干扰与低效重叠**：prefill 和 decode 共享网络路径时产生带宽竞争，且传统两方通信协议易引发死锁。

### **提出的新方法与创新思路**
ExpertPlex 提出一种 **混合解耦-共置架构（hybrid disaggregation-colocation architecture）**，核心思想是：
> **共享 MoE 专家，解耦 Attention 模块**

具体技术贡献如下：

#### **(1) 自适应持久化内核（Adaptive Persistent Kernel, APK）**
- 在每个 MoE GPU 上运行一个长生命周期的 APK，实现 **tile-level 调度**。
- 支持 **bounded preemption**：decode 请求可在最多一个 tile 执行时间内抢占 prefill 工作，延迟独立于序列长度。
- 实现 **无 CPU 干预的资源再分配**：当某阶段空闲时，其 SM 可被另一阶段立即复用，提升利用率。

#### **(2) 注意力发起的一方通信（Attention-initiated One-sided MoE Communication）**
- dispatch 和 combine 操作由 **attention server 主动 push/pull**，无需 MoE server 协调或轮询。
- 消除中间 ring buffer 和信用反馈机制，避免跨阶段死锁。
- 支持 **跨阶段通信-计算重叠**：例如，在 decode 等待 combine 结果时，APK 可执行 prefill 的 MoE 计算。

#### **(3) 分层流量隔离路径**
- **prefill 流量走 hierarchical path**：通过 prefill attention server 中转，利用 NVLink 多播减少跨节点传输次数。
- **decode 流量直连 MoE server**：降低延迟。
- 使用不同优先级的 InfiniBand virtual lanes 保护 decode 流量。

#### **(4) 跨栈放置优化器（Cross-stack Placement Optimizer）**
- 联合建模 tile-level 调度、并行策略、服务器布局、通信重叠等决策。
- 最大化 **goodput** 同时满足两个阶段的 SLO（如 TTFT 和 TPOT）。

### **相比现有方法的优势**
| 方面 | PDD | Green Context Colocation | ExpertPlex |
|------|-----|--------------------------|-----------|
| 内存效率 | ❌ 大量权重复制 | ✅ 权重去重 | ✅ 权重去重 |
| 弹性扩展 | ❌ 粗粒度（百卡级） | ✅ 细粒度 | ✅ 细粒度 |
| 故障域 | ❌ 大（层级通信耦合） | ✅ 小 | ✅ 小 |
| 资源利用率 | ❌ 固定比例易失配 | ❌ 分区固定有空泡 | ✅ 动态复用高 |
| 通信干扰 | ❌ 存在 | ❌ 存在 | ✅ 隔离优化 |

---

## **2. 核心实验方法和设置**

### **使用的模型**
- **MiniMax-M2.7**：单节点实验，FP8 精度，230GB 模型大小，每 token 激活 ~7.0B 参数。
- **GLM-5.1-FP8**：多节点实验，756GB 模型大小，每 token 激活 ~22.6B 参数。

### **测试平台**
- **单节点**：1 × NVIDIA H800，8 GPUs，NVLink 互联。
- **多节点**：最多 3 台机器，每台 8 × H800，通过 8 × 200 Gbps InfiniBand 互联。

### **工作负载（Workloads）**
- **输入/输出长度分布**：
  - **ShareGPT**：短请求（short requests）
  - **LooGLE**：长请求（long requests）
- 请求到达服从泊松过程（Poisson process）。

### **评估指标**
- **P90 Goodput**：在至少 90% 请求满足 SLO 的前提下，系统可承受的最高请求速率（req/s/node）。
- **SLO 定义**：
  - **TTFT（Time to First Token）**：如 1s / 10s / 2s / 20s
  - **TPOT（Time Per Output Token）**：如 50ms / 100ms

### **基线方法（Baselines）**
所有基线基于 SGLang 实现，确保公平比较：
- **SGLang-ChunkedPrefill**：将 prefill 切块以缓解干扰。
- **SGLang-Colocated**：原始共置模式，使用 DP+EP。
- **SGLang-PDD**：实例级 prefill-decode 解耦。
- **SGLang-PDMux**：基于 MuxWise 的 Green Context 分区方案，使用 TP+EP。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 场景 | ExpertPlex Goodput | 最佳基线 | 提升倍数 |
|------|--------------------|---------|--------|
| MiniMax-M2.7 + ShareGPT | 11.3 req/s/node | SGLang-PDD | **2.01×** |
| MiniMax-M2.7 + LooGLE | 显著优于所有基线 | SGLang-Colocated | **4.12×** |
| GLM-5.1-FP8 + ShareGPT | ~1.5 req/s/node | SGLang-PDMux | ≈1.5× |
| GLM-5.1-FP8 + LooGLE | ~0.25 req/s/node | SGLang-PDMux | **1.66×** |

### **与基线方法的对比结果**
- **vs. ChunkedPrefill**：内存访问开销大，重复读取权重和 KV Cache，性能最差。
- **vs. Colocated**：虽去重但分区固定，decode 易被长 prefill 阻塞。
- **vs. PDD**：权重复制导致 KV Cache 不足，且资源比例僵化。
- **vs. PDMux (Green Context)**：无法适应动态专家负载，通信干扰严重。

> ExpertPlex 在所有场景下均达到最高 goodput，尤其在长请求（LooGLE）中优势更明显。

### **消融实验与关键分析**
#### **(1) APK 调度有效性（Figure 11–12）**
- **CUDA Streams**：decode 延迟增加 **13.79×**（严重阻塞）。
- **Green Context / MPS**：decode 延迟接近理想值，但 prefill 性能下降 **3.33–4.07×**（资源无法复用）。
- **ExpertPlex (APK)**：
  - decode 延迟仅增加 **8%**
  - prefill 性能下降仅 **1.12×**
  → 实现了 **低延迟保障** 与 **高吞吐维持** 的平衡。

#### **(2) Tile-level 调度开销（Figure 13）**
- **Prefill 连续布局**：调度开销 < **12%**
- **Decode 掩码布局**：开销 < **20 μs**，相对极短 decode GEMM 时间仍可接受。

#### **(3) 一方通信开销（Figure 14）**
- dispatch/combine 时间与 DeepEP v1 相比差异 < **5%（正常模式）**
- 低延迟模式下差异 < **45 μs**
→ 通信效率几乎无损。

#### **(4) 抢占间隔分析（Figure 15）**
- 所有 MoE 操作的 tile 执行时间 < **25.3 μs**
- GEMM 操作 < **10.7 μs**
→ 支持微秒级抢占，远优于 PipeSwitch、LithOS 等系统（~100–350 μs）

---

## **4. 关键结论和发现**

### **主要发现**
1. **MoE 权重应跨阶段共享**：消除 >95% 的权重复制，显著提升内存效率和 KV Cache 容量。
2. **Attention 应按阶段解耦**：避免 intra-GPU 分区带来的并行度上升和通信开销。
3. **tile-level 调度是高效共享的关键**：APK 实现了空间与时间复用、快速抢占与再分配，同时保持 CUDA Graph 兼容性。
4. **一方通信可解除死锁并支持跨阶段重叠**：attention 发起的 push/pull 消除了 MoE server 的协调负担。
5. **跨栈联合优化至关重要**：placement、parallelism、overlap、sharing 必须协同设计才能最大化 goodput。

### **方法的局限性**
- **依赖现代 GPU 特性**：需要支持 TMA、CTA Cluster、DSMEM 等特性（如 Hopper 架构）。
- **APK 实现复杂**：需深度定制 MoE kernel，对框架侵入性强。
- **未处理 MoE 内部负载均衡问题**：假设已有负载均衡机制（如 UltraEP），但未集成优化。

### **未来工作方向**
- 支持更多模型架构（如 dense + MoE 混合）。
- 扩展至多租户场景下的公平资源共享。
- 结合 MoE 负载均衡算法进行端到端优化。
- 探索在边缘设备上的轻量化部署版本。

---

> **总结**：ExpertPlex 通过“**共享 MoE、解耦 Attention、APK 调度、一方通信**”四步创新，在 MoE LLM serving 中实现了高达 **2.01×** 的 goodput 提升，解决了传统 PDD 与 colocation 方案的根本缺陷，为高吞吐、低延迟的大规模模型服务提供了新的范式。

</details>

---

### 2. [A Training-Memory Regression in MLA Sequence Parallelism: Why Megatron-Core Forbids Absorption, and LAGA -- a Communication-Efficient Fix](https://arxiv.org/abs/2607.17644)

**Authors**: Changzheng Ma  
**Category**: cs.DC  
**Published**: 2026-07-21  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2607.17644v1  

#### Abstract
Multi-head Latent Attention (MLA) ships two implementations in Megatron-Core: an explicit form used for training and an absorbed form -- which slashes collective communication by gathering only the compressed latent -- that is fully implemented but hard-asserted out of training (the forward opens wi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*A Training-Memory Regression in MLA Sequence Parallelism: Why Megatron-Core Forbids Absorption, and LAGA -- a Communication-Efficient Fix*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文揭示并量化了一个在 **Multi-head Latent Attention (MLA)** 中被广泛忽略的关键问题：  
**为什么 Megatron-Core 在训练时禁用了低通信开销的 `absorbed` 形式（即 latent-based 通信）？**

尽管 `absorbed` 形式在推理阶段能显著减少通信量（仅需 All-Gather 压缩后的 latent），但在训练中却被硬断言（`assert not (self.training and self.cache_mla_latents)`）禁止。论文首次解释了这一设计背后的深层原因。

> 🔍 **核心发现**：将 `absorbed` 形式直接用于训练会导致严重的 **activation memory 膨胀**，形成“内存陷阱”（memory trap），可能导致模型无法 fit 到设备上。

---

### 🚀 提出的新方法：**LAGA (Latent All-Gather Attention)**

#### 新思路
LAGA 的核心思想是：
- **保留 `absorbed` 的通信效率优势**：只在 collective communication 阶段传输压缩后的 `latent`（而非完整的 per-head K/V）；
- **拒绝 `absorbed` 的数学重构（absorb reformulation）**：避免引入高维中间变量（如 `q_absorbed ∈ ℝ^{n_h × d_k}`）；
- **本地重建 per-head K/V**：每个 rank 在收到 `latent` 后，通过局部 `up-projection` 重新生成其负责的 head shard 的 K/V。

> 💡 即：**通信走 latent 路径，计算仍用原始 head-dim attention 数学形式**。

---

### ⚖️ 相比现有方法的优势

| 维度 | Explicit (B1) | Absorbed (B2) | **LAGA (Ours)** |
|------|---------------|----------------|------------------|
| **通信量** | 高（All-to-All per-head K/V） | 低（All-Gather latent） | ✅ 低（同 B2） |
| **激活内存** | 正常 | ❌ 显著膨胀（+9.2 GB @ V3） | ✅ 与 B1 几乎一致（<0.5% 差异） |
| **数值一致性** | 基准 | 浮点等价 | ✅ 与 B1 浮点等价（bit-identical @ SP=1） |
| **吞吐提升** | 基准 | 受限于内存 | ✅ 单节点 1.04–1.06×，跨节点 1.07–1.24× |

> ✅ **LAGA 成功解耦了通信优化与内存代价**，实现了“两全其美”。

---

## 2. 核心实验方法和设置

### 🧪 实验设置

- **硬件平台**：
  - 单节点：8×Ascend 910B（HCCL）
  - 多节点：2 节点 × 8 卡（RoCE ~11 GB/s）
  - 对比验证：NVIDIA A100（CUDA/NCCL）

- **模型配置**：基于真实 **DeepSeek-V3** 规模
  - `d_model = 7168`, `n_h = 128`, `d_head = 128`, `d_rope = 64`, `d_k = 512`
  - 序列长度 `S ∈ {4096, 8192, 16384}`
  - Sequence Parallelism Degree `SP ∈ {4, 8, 16}`

- **实现细节**：
  - 使用 PyTorch eager 模式（无 fused kernel）进行主实验
  - 使用 `npu_fusion_attention` 验证 fused kernel 下的表现
  - 数据类型：bf16

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **Per-forward communication (MB)** | 每个 rank 每 forward 步发送的数据量 |
| **Peak activation memory (MB)** | 训练过程中最大显存占用（`max_memory_allocated`） |
| **Throughput (tokens/s)** | 正向+反向总吞吐量 |
| **Numerical equivalence** | 输出与梯度的最大差值（`max|Δ|`） |
| **End-to-end convergence** | 多步训练 loss 曲线是否收敛到相同终点 |

---

### 🆚 基线方法对比

| 方法 | 简称 | 描述 |
|------|------|------|
| **Explicit (B1)** | B1 | Megatron-Core 当前训练路径：先 materialize per-head K/V，再 All-to-All |
| **Absorbed (B2)** | B2 | 将推理端 absorb 技巧移植到训练：query-absorb + latent-space attention |
| **LAGA (Ours)** | —— | 本文提出：All-Gather latent + 本地 head-shard up-projection + head-dim attention |

> 所有方法共享相同权重、输入和 SP 设置，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）通信量对比（Table 1）
- LAGA 与 B2 的通信量相同，均为 B1 的 **~50.5%**，即 **1.98× 减少**
- 通信节省随 `n_h` 增大而增强（从 1.88× → 1.98× @ `n_h=16→128`）

#### （2）激活内存对比（Table 2, Figure 4）
| 条件 | B1 (Explicit) | B2 (Absorbed) | LAGA |
|------|--------------|----------------|-------|
| `S=16384, SP=8` | 27.3 GB | **36.6 GB**（+9.2 GB） | 27.4 GB（≈ B1） |

> 🔺 **Absorbed 内存膨胀高达 20–34%**，且随序列长度线性增长，构成严重内存陷阱。

> 🔻 **LAGA 内存与 B1 差距 <0.5%（<27MB）**，完全规避该问题。

#### （3）吞吐量表现（Tables 3–5, Figure 5）

##### 单节点（SP=8, `n_h=128`）
| `S` | LAGA / Explicit（fused） |
|-----|--------------------------|
| 4096 | 1.05× |
| 8192 | 1.06× |
| 16384 | 1.04× |

✅ **在生产规模下，LAGA 全面优于 Explicit**

##### 跨节点（SP=16, `n_h=128`, fused kernel）
| `S` | LAGA / Explicit |
|-----|------------------|
| 4096 | 1.07× |
| 8192 | 1.15× |
| **16384** | **1.24×** |

> 🚀 **在长上下文（≥8K）、跨节点场景下，LAGA 显著领先，优势随序列增长而扩大**

> ⚠️ Eager kernel 下短序列存在回归（0.73×），但为实现 artifact，在 fused kernel 中消失。

#### （4）消融实验与正确性验证（Section 4.5, Figure 6）

- **SP=1 时输出与梯度完全 bit-identical**（`max|Δ|=0.0`）
- **SP=2–8 时浮点等价**：
  - 输出差异 ≤ `5.6e-6`
  - `W_kvb` 梯度差异 ≤ `7.7e-4`
- **多步训练收敛**：LAGA 与 B2 完全同步漂移，均与 B1 因 FP chaos 正常偏离，最终 loss 收敛至相同值（~0.004）

> ✅ 证明 LAGA 引入无额外数值偏差或训练不稳定性。

---

## 4. 关键结论和发现

### 🔑 主要发现

1. **Absorbed 形式在训练中是“内存陷阱”**：
   - 中间变量 `q_absorbed` 和 latent accumulator 维度为 `n_h × d_k`，远大于 `n_h × d_head`（当 `d_k > d_head` 时）
   - 在 DeepSeek-V3 规模下导致 **+9.2 GB 激活内存**，足以改变 device-fit 决策
   - Megatron-Core 的 `assert` 是合理的、未文档化的保护机制

2. **Absorb 重构非必要**：
   - 我们真正需要的是“**不传输 per-head K/V**”
   - MLA 已提供天然压缩 latent，可直接用于通信

3. **LAGA 实现最优权衡**：
   - 继承 `absorbed` 的通信优势（1.98× 减少）
   - 继承 `explicit` 的内存特性（几乎零膨胀）
   - 数学等价，无需修改 optimizer 或 loss behavior

4. **吞吐优势在生产场景显现**：
   - 优势随 `n_h` 和 `S` 增大而增强
   - 在跨节点长上下文训练中（MLA 的典型部署场景），LAGA 全面领先

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **原型规模实验** | 实验基于单层或 4 层堆叠，非完整 61 层 DeepSeek-V3 端到端训练 |
| **未融合 up-projection kernel** | 当前实现中 `local up-projection` 为独立 einsum，存在小 GEMM 开销；未来可进一步融合 |
| **仅适用于训练** | 不处理推理阶段的 KV cache 并行（属 TPLA/MLRA/Helix 范畴） |
| **依赖 fused attention kernel** | Eager 实现中短序列性能受 score matrix 材料化影响，需 fused kernel 才能释放全部潜力 |

---

### 🔮 未来工作方向

1. **开发 fused LAGA kernel**：将 `up-projection → attention` 融合为单一 kernel，进一步降低小 shard GEMM 开销
2. **扩展至 ring-style SP**：支持 `P > n_h` 场景下的环形通信模式
3. **流水线重叠优化**：将 `latent all-gather` 与前一层计算重叠，隐藏通信延迟
4. **MoE + LAGA 端到端评估**：在完整 MoE 架构下测量 MFU 与 wall-clock 时间
5. **跨更多硬件验证**：在更多 NPUs/GPUs 上验证通用性

---

## ✅ 总结一句话

> **论文揭示了 Megatron-Core 禁用 MLA absorb 训练的根本原因是 activation memory 回归，并提出了 LAGA——一种通信高效、内存中性、数学等价的训练方案，在长上下文场景下实现最高达 1.24× 的吞吐提升。**

</details>

---

### 3. [FlashRT: Agent Harness for Guiding Agents to Deploy Real-Time Multimodal Applications](https://arxiv.org/abs/2607.18171)

**Authors**: Krish Agarwal, Zhuoming Chen, Yanyuan Qin, Zhenyu Gu, Atri Rudra, Beidi Chen  
**Category**: cs.LG  
**Published**: 2026-07-21  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2607.18171v1  

#### Abstract
Real-time multimodal applications, including voice agents and interactive video generation, compose heterogeneous models into pipelines whose efficient deployment requires application-specific decisions about placement, streaming, and intra-model parallelism. Existing serving systems and auto-parall...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《FlashRT: Agent Harness for Guiding Agents to Deploy Real-Time Multimodal Applications》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
实时多模态应用（如语音代理、交互式视频生成）通常由异构模型组成复杂流水线，其高效部署需要在**设备放置（placement）、流式处理（streaming）和模型内并行化（intra-model parallelism）**等方面进行应用特定的决策。然而，现有的 serving 系统和自动并行编译器存在以下三大局限：

1. **有限的部署策略**：如 vLLM-Omni 和 Cornserve 等框架仅支持静态部署策略（如阶段间完全解耦或共置），难以适应多样化的延迟/吞吐目标。
2. **工作负载覆盖范围有限**：FlexFlow、Alpa 等 auto-parallelism 框架针对固定训练任务优化，无法泛化到动态推理场景。
3. **优化粒度不足**：TVM、TASO 等编译器聚焦于算子级优化，无法处理跨组件调度、状态管理等系统级挑战。

因此，为每个新应用手动设计高效部署成为常态，但这种方式不可扩展。

---

### 提出了什么新方法或新思路
本文提出 **FLASHRT**，一个基于 AI 编码 agent 的 **agent harness**（智能体引导框架），用于将开发者编写的简单单 GPU 参考实现，自动转化为高效的多 GPU 部署方案。

其核心创新在于两个关键机制：

#### （1）Chain-of-Program 范式（Chain-of-Program Paradigm）
受 Chain-of-Thought 启发，FLASHRT 引导 agent 分多步完成转换：
- **中间表示（IR）构建**：将原始代码转换为带有显式数据依赖、持久状态作用域和流式注释的层次化有向无环图（DAG）。
- **静态分析**：基于 IR 自动识别可并行节点、流式机会等候选变换。
- **逐步转换**：避免直接端到端转换失败，提升 agent 推理鲁棒性。

#### （2）应用感知的自验证循环（Application-Grounded Validation Loop）
- **自我驱动优化循环**：agent 迭代地提出假设、实现、验证正确性、基准测试性能。
- **应用定制测试套件**：agent 自动生成模拟用户输入的测试程序，通过读写后端缓冲区来测量端到端延迟和吞吐量，确保优化结果贴近真实用户体验。
- **变体队列机制**：维护候选策略队列，动态排序与扩展，保证探索多样性与组合优化能力。

---

### 相比现有方法的优势
| 维度 | 现有方法 | FLASHRT |
|------|--------|---------|
| **灵活性** | 固定策略，受限抽象 | 支持任意参考实现，灵活权衡延迟/吞吐 |
| **通用性** | 特定任务或硬件 | 泛化至多种应用与平台（NVIDIA/AMD） |
| **自动化程度** | 手工调优或规则系统 | 完全自动化，无需人工干预 |
| **优化深度** | 单一层级优化 | 跨层级协同优化（应用级 + 模型级） |

---

## 2. 核心实验方法和设置

### 使用的应用场景（Applications）
论文在五类典型实时多模态应用上进行了评估：
1. **Face-to-Face Conversational Agent**（基于 LiveAvatar）
2. **Multimodal LLM: Qwen3-Omni**
3. **Video Background Editor**（Krea-Realtime + SAM 3）
4. **Video World Model: WorldPlay**
5. **Video Narrator: LongLive**

这些应用涵盖语音识别（ASR）、大语言模型（LLM）、文本转语音（TTS）、音视频生成（S2V）、视频扩散模型（DiT + VAE）等多种组件组合。

---

### 实验设置
- **硬件平台**：
  - 主要：8× NVIDIA B200 GPUs
  - 对比：8× AMD MI355X GPUs（验证跨平台泛化能力）
- **Agent 配置**：
  - 使用 Anthropic Claude Code（Opus 4.8），启用 auto-execution 权限
  - 每次运行从零开始，无记忆共享
  - 输入仅为同步单 GPU 参考实现 + GPU 数量预算
- **优化目标**：最小化 **Time-to-First-Output (TTFO) Latency** 或最大化 **Throughput (FPS / RTF)**

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **Latency (s/ms)** | 从输入到首个输出的时间（TTFO） |
| **Frame Rate (FPS)** | 视频生成帧率 |
| **Real-Time Factor (RTF)** | 生成时间 / 输出音频时长；RTF < 1 表示实时生成 |
| **Speedup** | 相对于 baseline 的加速比 |

---

### 基线方法对比
- **Sequential Baseline**：原始同步单 GPU 实现（无流式）
- **vLLM-Omni**：专家手工优化的多模态 serving 系统
- **Hand-engineered deployments**：人工设计的高性能部署作为上限参考

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 在 NVIDIA B200 上的结果
| 应用 | 最佳加速比 | 吞吐提升 | 备注 |
|------|-----------|----------|------|
| Face-to-Face Agent | **~70× latency reduction** | FPS 达 **173.67** | 8 GPU 下结合 streaming + disaggregation + S2V PP |
| Qwen3-Omni | **25% lower latency** vs vLLM-Omni | RTF < 1 | 更轻量的数据传输 |
| Video World Model (WorldPlay) | **51% lower latency**, **2.2× FPS** | 可灵活 trade-off |
| Video Narrator (LongLive) | **1.6× lower latency**, **2.6× FPS** | ASR 异步重叠 |

#### 在 AMD MI355X 上的结果（Table 9）
| 应用 | 性能表现 |
|------|---------|
| Face-to-Face Agent | **~70× latency reduction**, **78.5 FPS** |
| Qwen3-Omni | **65% lower latency** vs vLLM-Omni (**0.276s vs 0.779s**)，RTF < 1 |
| WorldPlay | 最低延迟 **320ms**，最高帧率 **56.8 FPS** |
| 结论 | 在较不成熟的 AMD 平台上，FLASHRT 仍能发现高效部署，且性能优于专家实现 |

---

### 与基线方法的对比结果
| 对比项 | 结果 |
|-------|------|
| vs Sequential Baseline | 显著降低延迟（最高达 70×），大幅提升吞吐 |
| vs vLLM-Omni | 在相同硬件下，FLASHRT 实现更低延迟（如 Qwen3-Omni 上 **快 65%**）且保持 RTF < 1 |
| vs Rule-based Systems | 成功发现组合优化策略（如 streaming + PP + SP），而 naive agent 常只关注单一维度 |

---

### 消融实验与关键观察（来自 Case Study）
- **仅 streaming**（1 GPU）：TTFO 降低 **27.4×**，但帧率低（因资源竞争）
- **+ disaggregation**（3 GPU）：进一步降低延迟，帧率提升 **2.5×**
- **+ S2V Pipeline Parallelism**（8 GPU）：帧率跃升至 **173.67 FPS**，接近理论极限
- **Failure Mode of Naive Agent**：
  - 忽略模型级并行机会（如 DiT 与 VAE 解耦）
  - 无法稳定发现多策略组合
- **FLASHRT 的优势体现**：
  - 通过 IR 显式建模状态与流式关系，避免遗漏关键优化点
  - 自验证循环确保每一步变换都经过实测验证

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Agent-driven Optimization 是可行且强大的**：  
   通过合理的流程引导（IR + validation loop），通用 coding agent 能够自主发现媲美甚至超越专家手工优化的部署方案。

2. ✅ **Chain-of-Program 提升推理稳定性**：  
   将端到端转换分解为“构建 IR → 静态分析 → 渐进式实现”显著提升了 agent 发现复杂优化策略的能力。

3. ✅ **应用感知验证至关重要**：  
   传统 kernel-level benchmark 不适用于多模态系统优化；必须通过模拟用户交互路径进行端到端测量。

4. ✅ **跨平台泛化能力强**：  
   在 AMD MI355X 上，FLASHRT 不仅复现了在 NVIDIA B200 上的优化模式，还在某些应用上实现了更高吞吐（最高 **3.6× throughput gain**），表明其对优化生态尚不完善的平台更具价值。

5. ✅ **存在明确的 latency-throughput trade-off**：
   - **Co-location + Sequence Parallelism** → 降低关键路径延迟
   - **Disaggregation + Pipeline Parallelism** → 提高可持续吞吐
   - FLASHRT 可根据目标灵活选择或组合策略

---

### 方法的局限性
1. ❗ **未集成 LLM kernel 级优化**：  
   当前 FLASHRT 专注于系统架构级优化，未对底层 kernel（如 Triton、CUDA）进行重写，仍有进一步性能空间。
   
2. ❗ **依赖高质量 agent**：  
   实验仅使用 Claude Opus，不同 LLM 或配置可能影响结果稳定性。

3. ❗ **未探索更复杂的通信拓扑**：  
   当前假设简单的 GPU 组通信，未考虑 NVLink、RDMA 等高级互连的影响。

---

### 未来工作方向
1. **集成 kernel-level agent**：联合优化系统架构与底层 kernel 实现（如 AutoTriton + FLASHRT）。
2. **探索更多 agent 架构**：测试不同 reasoning depth、tool use 策略对性能的影响。
3. **支持动态负载调整**：根据实时请求模式在线调整部署策略。
4. **扩展至边缘设备**：适配 mobile/edge 场景下的资源约束部署。

---

> 🔗 **项目地址**：  
> GitHub: [https://github.com/Infini-AI-Lab/FlashRT](https://github.com/Infini-AI-Lab/FlashRT)  
> Website: [https://infini-ai-lab.github.io/flashrt-blog](https://infini-ai-lab.github.io/flashrt-blog)

</details>

---

### 4. [LaCache: Exact Caching and Precision-Adaptive Inference for Diffusion Large Language Models](https://arxiv.org/abs/2607.16339)

**Authors**: Xingru Chen, Zelang Liang, Yongjia Ma, Jiqing Zhan, Shuling Yang, Lian Wen, Kun Zhan  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.16339v1  

#### Abstract
Diffusion-based Large Language Models(DLLMs) enable parallel generation via Semi-Autoregressive (SAR) decoding in text generation. However, current methods suffer from severe operator-level redundancy: they recompute the entire sequence during denoising steps, ignoring that the prefix and masked suf...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LaCache: Exact Caching and Precision-Adaptive Inference for Diffusion Large Language Models**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

Diffusion-based Large Language Models (**DLLMs**) 采用 **Semi-Autoregressive (SAR)** 解码机制，在每个生成块（block）内并行去噪多个 token，从而提升生成效率。然而，当前的 DLLM 推理存在严重的**算子级冗余（operator-level redundancy）**：

- 在一个 SAR 块的多次去噪步骤中，只有当前块内的 token 发生变化，而前缀（prompt）和其余被掩码的后缀 token ID 保持不变。
- 尽管如此，模型仍对整个序列执行完整的前向传播，导致在 **Embedding、QKV 投影、RoPE、FlashAttention** 等 token-wise 算子上重复计算。

此外，推理过程还面临 **memory-bandwidth bottleneck**，尤其是在高吞吐场景下。

---

### 🚀 提出了什么新方法或新思路

作者提出 **LaCache** —— 一种无需训练的 DLLM 推理加速框架，核心是 **Lossless State Memoization (LSM)** 和 **FP8 混合精度策略**。

#### 主要创新组件：

1. **Lossless State Memoization (LSM)**  
   在每个 SAR 块的首次去噪步中缓存不变区域（U 区域）的中间状态，并在后续步骤中复用，实现**无损跳过冗余计算**：
   - **EmbedCache**：缓存不变 token 的 Embedding 输出。
   - **RoPECache**：缓存第一层中不变 token 的 Attention Norm + QKV 投影 + RoPE 结果。
   - **FACache**：缓存第一层 FlashAttention 中的在线 softmax 状态（如 `RowMax`, `RowSum`, `Output Accumulator`），允许跳过不涉及当前块的 attention tile 计算。

2. **Per-group FP8 Mixed-Precision Strategy**
   - 针对 FFN 层中的 `GateUp Linear` 和 `FFOut Linear` 应用 **per-group FP8 量化**，适配不同去噪步中激活值分布的变化。
   - 量化操作与 `AddNorm` 和 `SwiGLU` 融合，减少访存开销。
   - 利用现代硬件（如支持 FP8 的 Tensor Core）提升计算效率。

---

### 🔍 相比现有方法的优势

| 对比维度 | 现有方法（如 Fast-DLLM, DPad） | **LaCache（本文）** |
|--------|-------------------------------|---------------------|
| 冗余消除粒度 | 调整解码调度或上下文长度（step-level） | **算子级冗余消除（operator-level）** |
| 缓存机制 | Approximate KV Cache 或上下文剪枝 | **Lossless 缓存，输出完全等价** |
| 精度控制 | 多为 FP16 或 INT8，未精细适配激活分布 | **Per-group FP8，动态适应激活模式** |
| 可组合性 | 可与其他方法结合 | ✅ **完全兼容现有加速方法，叠加增益显著** |

> ✅ **LaCache 是首个系统性解决 DLLM 中 operator-level 冗余的工作，且无需训练、无精度损失。**

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集

- **GSM8K**（数学推理）
- **MATH**（复杂数学问题）
- **HumanEval**（代码生成）
- **MBPP**（Python 编程任务）
- 其他补充基准：`PIQA`, `ARC`, `HellaSwag`, `Winogrande`, `GPQA`

---

### ⚙️ 实验设置

- **模型**：
  - `LLaDA-base`
  - `LLaDA-instruct`
  - `LLaDA-1.5`
  - `Fast-dllm-v2`（用于验证兼容性）

- **硬件平台**：
  - 主要：**NVIDIA H200 141GB GPU**
  - 补充测试：**H100 GPU**

- **评估指标**：
  - **Latency (s)**：每条样本平均推理延迟
  - **Throughput (TPS)**：Tokens Per Second
  - **Accuracy (%)**：
    - Code Generation：`pass@1`
    - Reasoning Tasks：`Flexible-extract`, `Strict-match`

---

### 🆚 基线方法对比

| 基线方法 | 简介 |
|--------|------|
| **Vanilla** | 原始 LLaDA 模型，无任何优化 |
| **+Parallel (Fast-dllm)** | 使用阈值采样替代 top-k，支持更多并行解码 |
| **+DPad** | 后缀丢弃策略，通过滑动窗口减少 attention 上下文长度 |
| **+Parallel + DPad** | 组合优化方法 |

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（以 LLaDA-Instruct 为例）

| 方法 | Latency (s) ↓ | TPS ↑ | Speedup vs Vanilla |
|------|---------------|-------|--------------------|
| Vanilla | 10.59 | 19.78 | 1.0× |
| **+LaCache** | **9.60** | **21.78** | **~1.3×** |
| +Parallel | 3.52 | 60.44 | ~3.0× |
| **+Parallel + LaCache** | **2.48** | **86.33** | **~4.3×** |
| +Parallel + DPad | 2.53 | 56.88 | ~4.2× |
| **+Parallel + DPad + LaCache** | **2.12** | **68.43** | **~5.0×** |

> ✅ **LaCache 单独带来约 1.3× 端到端加速，与现有方法组合可达最高 40.2× 加速！**

---

### 🔬 与其他 DLLM 架构的兼容性表现

| 模型 | 最高加速比（vs Vanilla） |
|------|--------------------------|
| LLaDA-base | **5.9×** |
| LLaDA-1.5 | **12.3×** |
| Fast-dllm-v2 | **+6% 额外加速（仅 FACache）** |

> ✅ 显示出良好的泛化性和可组合性。

---

### 🔍 消融实验结果

#### （1）仅使用缓存策略（w/o FP8）

| 方法 | Latency (GSM8K) | TPS |
|------|------------------|-----|
| Vanilla | 10.59 | 19.78 |
| LaCache (w/o FP8) | 10.54 | 19.86 |
| **LaCache (完整)** | **9.60** | **21.78** |

> ⚠️ 缓存本身仅带来 ~3% 加速，**FP8 混合精度是主要加速来源之一**。

#### （2）不同量化粒度对比（Tab 1 & 2）

| 量化方式 | Accuracy | TPS |
|---------|----------|-----|
| Per-Tensor | 略降 | 低 |
| Per-Token | 略降 | 中 |
| **Per-Group (LaCache)** | ✅ **基本无损** | ✅ **最高** |

> ✅ **Per-group FP8 在精度和效率之间达到最佳平衡。**

#### （3）多层缓存尝试（附录 C）

- 若将 LSM 扩展至深层（非第一层），会出现**速度-精度权衡**：
  - 缓存更新间隔越长 → 速度越快，但精度下降越明显。
- 因此，**lossless 缓存目前仅适用于第一层**，深层需引入近似方法。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **DLLM 存在严重 operator-level 冗余**，尤其在 SAR 块内重复计算不变 token。
2. **LaCache 通过 LSM 实现无损缓存复用**，跳过 Embedding、RoPE、FlashAttention 中的冗余计算。
3. **Per-group FP8 量化有效缓解 memory-bandwidth 瓶颈**，特别适合 SAR 推理中稀疏激活的特性。
4. **LaCache 完全兼容现有加速方法**（如 Parallel, DPad），组合后可实现高达 **40.2× 的端到端加速**。
5. **精度基本无损甚至略有提升**，部分任务因生成更完整而得分更高（见 Fig. 7）。

---

### ⚠️ 方法的局限性

1. **Lossless 缓存受限于第一层**：
   - 后续层由于隐藏状态双向传播，无法直接应用 exact caching。
   - 深层缓存需依赖近似方法，可能引入质量退化。

2. **依赖现代硬件支持 FP8**：
   - 当前 FP8 加速依赖 NVIDIA Hopper 架构（如 H200/H100）的 Tensor Core。
   - 在旧设备上难以发挥全部性能优势。

---

### 🔮 未来工作方向

1. **探索深层近似缓存机制**（Approximate State Memoization），在可控误差下扩展 LSM 至更多层。
2. **自适应缓存更新策略**：根据 token 稳定性动态决定是否刷新缓存。
3. **跨块缓存迁移**：研究前序 block 的缓存能否部分复用于后续 block。
4. **支持更多低精度格式**（如 INT4, NF4）以进一步压缩带宽需求。
5. **扩展至其他生成范式**：如 AR + Diffusion 混合模型。

---

## ✅ 总结

> **LaCache 是一项开创性的 DLLM 推理加速工作，首次系统性解决了 SAR 解码中的 operator-level 冗余问题。它通过 Lossless State Memoization 和 per-group FP8 混合精度策略，在不牺牲生成质量的前提下，实现了高达 40.2× 的端到端加速，且能无缝集成到现有加速流程中，具有极强的实用价值和推广潜力。**

</details>

---

### 5. [C$^2$KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference](https://arxiv.org/abs/2607.17715)

**Authors**: Chuheng Du, Junyi Chen, Hanlin Tang, Kan Liu, Tao Lan, Lin Qu, Chaoyue Niu, Shengzhong Liu, Guihai Chen, Fan Wu  
**Category**: cs.CL  
**Published**: 2026-07-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.17715v1  

#### Abstract
Long-context inference is central to modern large language model (LLM) applications such as retrieval-augmented generation and multi-document reasoning. To mitigate the growing inference cost, recent work has explored key-value (KV) cache reuse to reduce redundant prefill computation. However, exist...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：C²KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大语言模型（LLM）在长上下文场景（如检索增强生成 RAG、Few-shot Learning）中的推理成本极高，主要瓶颈已从**计算开销**转向 **KV Cache 的存储与内存带宽压力**。尽管已有非前缀（non-prefix）KV 缓存复用方法减少预填充（prefill）计算，但它们忽略了以下关键问题：

- **KV Cache 存储和传输成本高**：随着上下文长度增长，KV Cache 大小线性增加，导致 GPU 显存和内存带宽成为瓶颈。
- **现有压缩方法不可组合（non-composable）**：直接将通用 KV 压缩技术（如 SnapKV）应用于非前缀复用会导致严重精度下降，因为压缩后的表示依赖原始上下文位置，无法跨文档灵活拼接。

### 🚀 提出的新方法：C²KV
作者提出 **C²KV**（Compressed and Composable KV），一个统一框架，联合优化 **KV 压缩** 与 **非前缀复用时的拼接行为**，实现高效且准确的长上下文推理。

#### 核心创新点：
1. **轻量级可训练模块 C²Extractor**  
   - 附加于冻结的 base LLM 上，不修改原模型参数。
   - 引入可学习的 **C²Tokens** 作为压缩记忆槽（memory slots），通过新的 QKV 投影头提取语义信息。

2. **结构化注意力流（Structured Information Flow, SIF）**  
   - 设计特殊的 attention mask，确保：
     - 原始 token 不受 C²Tokens 影响（保持 base model 输出不变）
     - C²Tokens 只能访问其对应 block 内的原始 token（局部聚合）
     - C²Tokens 之间可因果累积上下文（支持长程建模）

3. **压缩-拼接共训练（Compression-Concatenation Co-Training）**  
   - 在训练阶段模拟多个文档 KV 拼接后生成答案的过程。
   - 使 C²KV 表示天然“为拼接而生”，具备位置无关性和可组合性。

4. **位置重对齐机制**  
   - 推理时对拼接后的 C²KV 动态分配 RoPE 位置编码，保证位置一致性。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | C²KV |
|------|--------|-------|
| **是否修改 base model** | 训练型方法需微调，破坏泛化能力 | ❌ 不修改，仅加轻量子模块 |
| **是否支持非前缀复用** | Prefix caching 无法复用任意顺序文档 | ✅ 支持任意顺序拼接 |
| **是否压缩 KV** | 多数未压缩或压缩后不可复用 | ✅ 联合压缩 + 可复用设计 |
| **推理延迟（TTFT）** | 需 blending/recompute，延迟高 | ✅ 仅需加载 + 直接拼接，接近 load-only |
| **准确性** | 压缩+复用组合导致严重降点 | ✅ 几乎无损，逼近 full recompute |

---

## 2. 核心实验方法和设置

### 📚 数据集

#### **训练数据**（用于训练 C²Extractor）：
- **HotpotQA**：多跳问答
- **2WikiMultiHopQA**：知识密集型推理
- **LongMagpie**：自合成长文本指令数据
- 共 120k 样本，采用 multi-document SFT 格式构造输入

#### **评估数据**（LongBench 套件为主）：
- **QA 类任务**：`HotpotQA`, `2WikiMQA`, `MuSiQue`, `QASPER`
- **摘要类任务**：`MultiNews`, `SAMSum`, `GovReport`
- **数学推理**：`GSM8K`

> 所有任务均构造为“多文档 + 查询”形式，以测试非前缀复用能力。

---

### ⚙️ 实验设置与评估指标

#### 模型：
- 主要测试模型：`Qwen3-4B-Instruct`, `Llama3.1-8B-Instruct`, `Qwen2.5-7B-Instruct`
- 更大模型验证扩展性：`Qwen3-14B`

#### 评估指标：
| 任务类型 | 指标 |
|--------|------|
| QA 任务 | F1 Score（token-level 匹配） |
| 摘要任务 | ROUGE-L |
| 数学任务 | Accuracy |

#### 性能指标：
- **Time-to-First-Token (TTFT)**：请求到首 token 生成的时间，反映启动延迟
- **Time-Between-Tokens (TBT)**：解码阶段每 token 平均耗时，反映持续吞吐
- **KV Cache Size**：衡量存储与带宽节省程度

---

### 🆚 基线方法对比

| 方法 | 类型 | 是否训练 | 是否压缩 | 是否支持非前缀 |
|------|------|---------|----------|----------------|
| **Full Recompute (FR)** | 上限基准 | ❌ | ❌ | ❌ |
| **Naive Reuse** | 下限基准 | ❌ | ❌ | ✅（但误差大） |
| **EPIC** | Selective Recompute | ❌ | ❌ | ✅ |
| **CacheBlend** | KV Blending | ❌ | ❌ | ✅ |
| **Block-Attention** | Training-based | ✅ | ❌ | ✅ |
| **FR + SnapKV** | 压缩基线 | ❌ | ✅ | ❌ |
| **C²KV**（本文） | ✅ | ✅ | ✅ | ✅ |

> 特别设置了 “**SnapKV + 各复用方法**” 对照组，验证通用压缩与复用结合的失败。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）TTFT-Accuracy 权衡（图7）
- C²KV 实现 **最低 TTFT + 最高准确率**，位于 Pareto 前沿左上角。
- 相比 CacheBlend 和 EPIC，在相同 TTFT 下提升 **~10–20% 准确率**。
- C²KV 的 TTFT 接近纯加载时间（load-only），无需 blending 或 recomputation。

#### （2）Decode-Time 效率（图8）
- 随着 context length 从 16k → 128k tokens：
  - Full KV 方法：TBT 近似线性上升（内存带宽受限）
  - **C²KV**：TBT 增长平缓，几乎不受 context 长度影响
- 在 128k context 下，**decode speedup 达 10× 以上**

#### （3）端到端加速比
- 在长上下文场景下，**最高实现 17× 推理加速**（主要来自 TTFT 和 decode 两阶段优化）

#### （4）压缩比扩展性（表1 & 表2）
| 压缩比 | MuSiQue (Llama3.1-8B) | WikiMQA (Llama3.1-8B) |
|--------|------------------------|------------------------|
| 4× | 0.3587 | 0.4477 |
| 8× | 0.3225 | 0.4462 |
| 16× | 0.2746 | 0.3973 |
| Full Recompute | 0.3198 | 0.4018 |

> 即使在 16× 压缩下，多数任务仍优于或接近 full recompute，显示极强鲁棒性。

#### （5）动态压缩比训练（Dynamic-Ratio Training）
- 单一模型可在推理时适配不同压缩比（如 4×, 8×, 10×, 16×）
- 在未见过的 10× 设置下依然表现良好，说明具有良好的泛化能力。

---

### 🔬 消融实验结果（表4）

| 方法变体 | Qasper ↓ | SAMSum ↓ | GovReport ↓ | 说明 |
|--------|--------|--------|-----------|------|
| Base Model | 0.4417 | 0.3652 | 0.3274 | 原始性能 |
| Anchor Tokens | 0.2750 | 0.3201 | 0.1280 | ❌ 双向注意力破坏可组合性 |
| C²KV w/o T. | 0.1572 | 0.3042 | 0.1318 | ❌ 无残差路径导致信息丢失 |
| Fixed-Pos | 0.2827 | 0.3303 | 0.2801 | ❌ 忽略位置重对齐显著降点 |
| Info-Leakage | 0.2577 | 0.3406 | 0.2543 | ❌ 破坏 block 局部性有害 |
| Global-Info | 0.3722 | 0.3810 | 0.2888 | ❌ 全局 attention 引入噪声 |
| **C²KV (4×)** | **0.3755** | **0.3904** | **0.2967** | ✅ 完整设计最优 |

> 结果证明：**结构隔离、位置重对齐、局部提取、残差连接** 是成功的关键。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **KV Cache 的存储与带宽已成为长上下文推理的主要瓶颈**，而非单纯的计算成本。
2. **简单地将 KV 压缩与复用叠加会严重损害性能**（见表3）：
   - 如 `SnapKV + CacheBlend` 在 HotpotQA 上从 0.53 降至 0.05
   - 说明标准压缩方法不具备“可组合性”
3. **C²KV 成功构建了一个“可组合的压缩 KV 流形”**：
   - 通过 SIF 和 co-training，让压缩表示天然适合拼接
   - 实现 **压缩 + 复用 + 高效 + 高精度** 四者统一
4. **无需修改 base model 即可部署**，兼容性强，易于集成到现有 LLM serving 系统（如 vLLM）

---

### ⚠️ 局限性（来自附录 E）

1. **依赖离线文档提取**  
   - 当前假设文档可提前识别并提取 C²KV
   - 尚不支持在线增量提取或动态 chunking

2. **固定压缩比策略**  
   - 当前使用 uniform 压缩比（如 4×）
   - 未来可探索 content-aware 自适应压缩（如重要段落低压缩）

3. **额外参数开销约 10%**  
   - 主要来自每层新增的 QKV heads
   - 对超大规模模型可能带来部署挑战

---

### 🔮 未来工作方向

1. **支持在线/流式 C²KV 提取**
2. **开发 content-adaptive 压缩策略**
3. **跨模型迁移 C²Extractor**（即一个 extractor 服务多个 base models）
4. **结合量化（Quantization）进一步降低存储成本**

---

> ✅ **代码已开源**：[https://github.com/s7a9/C2KV](https://github.com/s7a9/C2KV)  
> 📄 **论文链接**：[https://doi.org/10.1145/3770855.3817715](https://doi.org/10.1145/3770855.3817715)

--- 

📌 **一句话总结**：  
C²KV 通过设计一个**可组合的压缩 KV 表示空间**，首次实现了 **高效、低延迟、高质量** 的非前缀 KV 缓存复用，在不改动 base model 的前提下，达成高达 **17× 的推理加速**，为长上下文 LLM 应用提供了实用化的系统解决方案。

</details>

---

### 6. [LMEdge: QoS-Aware LLM Inference Orchestration on Edge Clusters](https://arxiv.org/abs/2607.17175)

**Authors**: Reza Farahani, Zoha Azimi, Mario Colosi, Schahram Dustdar  
**Category**: cs.DC  
**Published**: 2026-07-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.17175v1  

#### Abstract
Large language model (LLM) services increasingly operate on edge infrastructure, enabling low-latency and privacy-preserving AI services. However, efficiently serving LLM requests across heterogeneous and resource-constrained edge devices require orchestration mechanisms that jointly determine model...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LMEdge: QoS-Aware LLM Inference Orchestration on Edge Clusters

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在边缘计算环境中，**如何高效地调度大语言模型（LLM）推理请求**，以同时满足以下挑战：
- 边缘设备资源异构且受限（CPU/GPU、内存、带宽）
- 用户对延迟（latency）和准确性（accuracy）的 QoS 要求
- 查询具有多样性（复杂度、长度、任务类型不同）
- 现有方法仅优化单一维度（如负载均衡或模型分割），缺乏联合决策机制

传统方案如 split execution 或 edge caching 缺乏细粒度的 per-query 动态调度能力。

---

### 🚀 提出的新方法与创新点
本文提出 **LMEdge** —— 一种面向边缘集群的 QoS-Aware LLM 推理编排系统，其核心创新包括：

1. **联合决策框架**  
   针对每个查询，动态决定：
   - 使用哪个 **LLM family**（如 Llama3.1 vs Gemma3）
   - 选择何种 **model size**
   - 应用哪种 **quantization level**（如 Q4_K_M vs FP16）
   - 分配到哪台 **edge device**

2. **基于 BILP 的优化建模**  
   将调度问题形式化为一个 **Binary Integer Linear Programming (BILP)** 问题，目标是最小化响应时间，同时满足：
   - 准确性约束（不低于最优精度的 $ (1-\theta) $）
   - 资源容量限制（CPU/GPU、内存、带宽、并发数）

3. **轻量级在线启发式算法**  
   设计了一个近似 BILP 解的 **lightweight heuristic scheduler**，可在亚秒级完成调度决策，适用于高吞吐场景。

4. **五类 ML 预测器支持精细化估计**  
   构建五个轻量级 ML 模型用于预测每种 `(query, model, quantization, device)` 组合的行为：
   - Inference time
   - Accuracy
   - CPU/GPU usage
   - Memory usage
   - Response size  
   （均采用 XGBoost 或 RF，实现低开销高精度）

5. **公开可复现的大规模基准数据集**  
   发布包含 **超过 59,000 条记录** 的 benchmark 数据集，涵盖多种 query-model-device 组合，促进后续研究。

---

### 🔍 相比现有方法的优势
| 方面 | 现有工作（如 ExeGPT, DynamoLLM, RouteLLM） | LMEdge |
|------|--------------------------------------------|--------|
| 决策粒度 | 单一维度优化（如资源分配、频率调节） | 多维联合决策（模型+量化+设备） |
| QoS 支持 | 多关注 latency，忽略 accuracy 约束 | 同时保障 latency 和 accuracy |
| 预测机制 | 缺乏 query-specific 性能预测 | 引入 ML 预测器进行个性化预估 |
| 可扩展性 | 多为离线或粗粒度调度 | 在线实时调度 + 轻量级启发式算法 |
| 实验验证 | 小规模模拟或理论分析 | 基于真实 Kubernetes 边缘测试床（57 实例） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
共使用 **1422 个真实用户查询**，来自五个公开 benchmark 数据集，覆盖七类任务：
- **MMLU 子集**：biology, world history, geography（多项选择题）
- **GSM8K**：数学应用题
- **CommonsenseQA**：常识推理
- **TruthfulQA**：真实性判断
- **HumanEval**：代码生成

所有数据集提供 ground-truth 答案，便于 accuracy 评估。

此外，使用 **NVIDIA NeMoCurator** 对 query 复杂度打分，用于特征提取。

---

### ⚙️ 实验设置

#### 测试平台
- 构建基于 **Kubernetes 的边缘测试床**，共 **57 个实例**：
  - **ERR 层（Resource-Rich）**：12 个虚拟机（VMs），配置从 2C/16GB 到 8C/32GB
  - **ERC 层（Resource-Constrained）**：37 个 Raspberry Pi + 6 个 NVIDIA Jetson Nano
- 集群间通过 Submariner 互联，使用 Istio 进行路由
- 模型由 Ollama 提供服务，权重存储于 MinIO 中心仓库
- 网络带宽模拟真实 4G LTE trace（使用 tc/netem）

#### 部署模型
部署 **29 种 LLM 配置**，涵盖 8 个开源家族：
| Model | 参数范围 | Quantization Levels |
|-------|----------|---------------------|
| Gemma2/3, Llama3.1/3.2, Qwen3, Mistral, TinyLlama | 0.6B ~ 12B | Q2~Q8, K-types, FP16/BF16 |

轻量模型部署在 ERC 设备上，大型模型仅限 ERR。

---

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Response Time** | 推理延迟 + 传输延迟（含拥塞控制项） |
| **Serving Ratio** | 成功调度并返回结果的查询比例 |
| **Accuracy** | 模型输出是否正确（binary classification） |
| **Resource Utilization** | CPU/GPU、内存、带宽利用率分布 |
| **Scheduling Overhead** | 调度器运行时间 |

---

### 🆚 基线方法对比
比较三种调度策略：
1. **Random**：随机分配查询，不考虑资源状态或 QoS
2. **Load-aware [6]**：将查询发往当前资源利用率最低的节点
3. **LMEdge（本文方法）**：基于 ML 预测 + 启发式调度，满足 QoS 约束

> 注：其他相关工作（如 OptLLM、RouteLLM）因调度粒度不同（batch-level 或 session-level）未直接参与比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Fig. 4）

| 指标 | LMEdge 表现 | 对比优势 |
|------|-------------|---------|
| **平均响应时间** | 显著低于两个 baseline<br>（在 v=0.001 req/ms 下降低约 30–50%） | 更优的模型-设备匹配减少等待和传输时间 |
| **Serving Ratio** | 接近 Random，显著高于 Load-aware<br>（在高压下仍保持 >85%） | 合理推迟无法满足 QoS 的请求，避免失败 |
| **Accuracy** | 稳定维持在较高水平（θ=0.1）<br>波动远小于 baseline | 通过 accuracy 预测确保输出质量 |
| **资源利用率** | ERR 与 ERC 均被有效利用<br>无明显热点或闲置 | 实现跨层负载均衡 |

---

### 🔬 ML 预测器性能（Tables 2 & 3）

| 预测任务 | 最佳模型 | R² / F1-score | RMSE | 推理耗时（总） |
|--------|----------|---------------|------|----------------|
| Inference Time | XGBoost | R² = 0.71 | 0.59 s | 0.07 s |
| CPU Usage | XGBoost | R² = 0.97 | 177.3 ms | 0.04 s |
| Memory Usage | XGBoost | R² = 0.98 | 206.6 MB | 0.05 s |
| Response Size | XGBoost | R² = 0.61 | 0.77 KB | 0.08 s |
| Accuracy | XGBoost | F1 = 0.81 | – | 0.04 s |

✅ 所有预测器均具备高精度与极低推理开销，适合在线使用。

---

### ⚖️ BILP vs Heuristic 调度效率对比
- **Heuristic 调度时间**：始终 < 1 秒（通常 ~200ms）
- **BILP 求解时间**：随负载增加急剧上升，在 v=0.0008 时达数分钟
- **加速比**：最高可达 **7012%**

👉 表明 exact optimization 不适用于在线调度，而 heuristic 是实用且高效的替代方案。

---

### 🧪 消融实验与可视化分析（Fig. 5）
通过 heatmap 展示不同方法下的 **query 分配模式**：

- **Random**：分布零散，未体现 query 特征与模型能力匹配
- **Load-aware**：过度集中于 ERR 上的大模型（如 Llama3.1-8B），导致拥塞
- **LMEdge**：
  - 简单任务 → 分配给 SLM（如 TinyLlama, Gemma3-1B）
  - 复杂/准确敏感任务（如 Math, TruthfulQA）→ 分配给 LLM（如 Llama3.1-8B）
  - 有效利用 ERC 层设备，缓解 ERR 压力

✅ 体现了“按需匹配”的智能调度思想。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **多维联合决策显著提升 QoS**  
   同时优化 model selection、quantization、placement 可在延迟、准确性和资源利用之间取得更好平衡。

2. **ML 预测器是实现高效调度的关键**  
   轻量级 XGBoost 模型能够以毫秒级开销准确预测复杂行为，支撑实时调度决策。

3. **边缘环境需差异化调度策略**  
   不应盲目将所有请求导向高性能设备；合理利用 SLM 和 ERC 设备可提升整体系统效率。

4. **拥塞感知设计有效降低端到端延迟**  
   引入 congestion penalty ($\lambda$) 可引导流量避开繁忙节点，改善实际体验。

5. **公开数据集推动可复现研究**  
   超过 59,000 条实测数据为社区提供了宝贵的训练与验证资源。

---

### ⚠️ 方法的局限性
1. **静态部署假设**  
   当前模型部署固定，未考虑动态加载/卸载模型的成本。
2. **accuracy 预测依赖二值化处理**  
   TruthfulQA 的连续相似度分数被阈值化为 binary label，可能损失信息。
3. **尚未集成能量感知调度**  
   虽提及节能潜力，但当前调度未显式优化能耗。
4. **GPU 资源调度较简单**  
   并发流数量基于经验设定，缺乏更精细的 GPU sharing 机制。

---

### 🔮 未来工作方向
1. **引入在线学习机制**  
   利用运行时反馈持续更新预测模型（online adaptation）。
2. **支持 energy-aware scheduling**  
   在目标函数中加入功耗项，构建绿色边缘推理系统。
3. **扩展至更大规模 GPU 边缘集群**  
   支持更多并行推理、模型并行等高级执行模式。
4. **探索 multi-agent reinforcement learning 编排**  
   如参考 Yao et al. [14] 的 diffusion-based RL 方法进一步优化长期性能。

---

## 总结

<LMEdge> 是首个实现 **QoS-Aware、Per-Query、Multi-Dimensional** LLM 推理编排的边缘系统。它通过 **ML-driven prediction + lightweight heuristic scheduling**，在真实边缘环境下实现了更低延迟、更高准确率、更优资源利用率的综合性能超越。该工作不仅提出了新架构，还发布了高质量 benchmark 数据集，为未来边缘 LLM 系统研究奠定了坚实基础。

</details>

---

### 7. [Empowering On-Device Model Adaptation with an Edge AI Inference Accelerator](https://arxiv.org/abs/2607.18101)

**Authors**: Mateusz Piechocki, Alessandro Capotondi, Marek Kraft  
**Category**: cs.LG  
**Published**: 2026-07-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.18101v1  

#### Abstract
On-device model adaptation is essential to enable lifelong personalization on resource-constrained hardware, but compute, power, and memory limitations of such devices make end-to-end backpropagation impractical for modern deep neural networks. This work proposes a heterogeneous adaptation pipeline ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Empowering On-Device Model Adaptation with an Edge AI Inference Accelerator*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在资源受限的边缘设备上进行 **on-device training** 是实现模型持续个性化和适应动态环境的关键，但由于计算、内存和功耗限制，传统的端到端 **backpropagation** 在现代深度神经网络中难以实现。现有硬件（如嵌入式 CPU 或 GPU）通常无法高效支持训练任务。

本文旨在解决这一瓶颈：如何在不牺牲太多准确率的前提下，在低功耗边缘设备上实现快速、节能的模型自适应更新。

### 提出的新方法与新思路
提出了一种**异构的 on-device adaptation pipeline**，其核心思想是：
- **复用商用 edge AI inference accelerator（Hailo-8L）用于训练阶段的特征提取**。
- 将模型划分为两部分：
  - **Frozen backbone**：以 INT8 精度运行于 Hailo-8L 加速器上，仅执行前向传播，不参与反向传播。
  - **Trainable classification head**：轻量级 FP32 分类头在主机 CPU 上进行 fine-tuning。
- 利用 **ONNX Runtime Training API** 构建混合计算图，实现跨设备协同训练。

该方法实现了“**推理加速器用于训练**”的范式转换，充分利用了边缘推理芯片的高吞吐能力来缓解训练中的计算瓶颈。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **效率** | 显著减少 wall-clock 训练时间（最高达 15.4× 加速），降低能耗 |
| **可行性** | 在 Raspberry Pi 5 这类低端设备上也能实现频繁的小规模更新 |
| **兼容性** | 基于 ONNX 的工具链支持多种架构（CNN/ViT），具有良好的通用性 |
| **隐私与延迟** | 数据无需上传云端，本地完成训练，提升隐私性和响应速度 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CIFAR-100**：60,000 张彩色图像，100 类，统一调整为 224×224。
- **Oxford-IIIT Pet**：约 7,390 张猫狗品种图像，37 类，同样 resize 至 224×224 并采用 ImageNet 归一化。

### 模型选择（来自 `timm` 库）
| 模型 | 特点 |
|------|------|
| ResNet18 | 经典 CNN，量化鲁棒性强 |
| EfficientNet-Lite 4 | 高效卷积网络 |
| MobileNetV3 Large | 使用 Hard-Swish 激活函数，对量化敏感 |
| FastViT-SA12 | 视觉 Transformer，含 self-attention 结构，量化脆弱 |

### 硬件平台
| 平台 | 配置 |
|------|------|
| **Hailo-8L + Raspberry Pi 5** | 主要测试平台，Hailo-8L 提供 ~13 TOPS @ 1.5W，通过 M.2 接口连接 RPi 5（4GB RAM, Cortex-A76 CPU） |
| **Raspberry Pi 5 (CPU-only)** | 基线之一，全模型 FP32 运行于 CPU |
| **NVIDIA Jetson Orin Nano** | 高性能 edge GPU 基线，~40 TOPS @ 15W，FP32 GPU 训练 |

### 评估指标
- **Accuracy (Acc.)**：微调后测试集分类准确率
- **Throughput (Tput)**：每样本训练耗时（ms/sample）
- **Energy Consumption (E)**：每样本训练能耗（mJ/sample）
- **Time-to-Accuracy Convergence**：达到目标精度所需的时间

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **RPi 5 (CPU)** | 全模型在 CPU 上以 FP32 执行 head fine-tuning |
| **Orin Nano (GPU)** | 同样只微调 head，但在嵌入式 GPU 上运行 |
| **Proposed (Hailo-8L + RPi 5)** | Backbone offloaded 到 Hailo-8L（INT8），head 在 RPi 5 CPU 上训练（FP32） |

此外还系统评估了不同 **post-training quantization restoration** 策略的影响（见下表）：

| 编号 | 技术 | 成本 | 描述 |
|-----|------|------|------|
| #0 | None | — | 无恢复 |
| #1 | Eq. (Channel Equalization) | Low | 层间尺度均衡 |
| #2 | Eq. + IBC | Low | 加偏置校正 |
| #3 | Eq. + FT (Fine-tuning) | Medium | 知识蒸馏微调 |
| #4 | Eq. + AR (AdaRound) | High | 数据驱动权重舍入优化 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Batch Size = 1）

| 模型 | 方法 | CIFAR-100 Acc. (%) | Tput (ms/sample) | Energy (mJ/sample) |
|------|------|---------------------|------------------|--------------------|
| ResNet18 | RPi 5 (CPU) | 64.68 | 92.85 | 525.65 |
|            | Orin Nano (GPU) | 64.67 | 13.70 | 128.25 |
|            | **Proposed (#4)** | **64.62** | **6.04** | **38.65** |
|            | → **加速比 vs CPU**: **15.4×**, 能耗降 **13.6×** |
| EfficientNet-Lite 4 | RPi 5 | 58.48 | 178.28 | 1042.91 |
|                     | Orin Nano | 58.80 | 23.56 | 234.09 |
|                     | **Proposed (#3)** | **60.23** | **23.91** | **124.64** |
|                     | → **加速比 vs CPU**: **7.5×**, 能耗降 **8.4×** |
| MobileNetV3 Large | RPi 5 | 68.26 | 57.33 | 286.97 |
|                   | Orin Nano | 68.10 | 9.49 | 110.04 |
|                   | **Proposed (#4)** | **55.53** | **13.73** | **78.22** |
|                   | → **加速比 vs CPU**: **4.2×**, 准确率下降明显 |
| FastViT-SA12 | RPi 5 | 71.19 | 223.89 | 1224.60 |
|              | Orin Nano | 71.20 | 21.92 | 306.73 |
|              | **Proposed (#3)** | **50.19** | **22.81** | **119.64** |
|              | → **加速比 vs CPU**: **9.8×**, 但准确率损失大 |

> ✅ **最佳策略编号标注在括号中（如 #3, #4）**

### 与基线对比总结
- **训练速度**：在 ResNet18 上达到 **15.4× 快于 CPU 基线**，甚至快于 Orin Nano（6.04 vs 13.70 ms/sample）。
- **能效**：所有配置下每样本能耗显著低于 CPU 和 GPU 基线，最多降低 **13.6×**。
- **吞吐量扩展性**：随着 batch size 增加（1→4→16），Hailo-8L 方案表现出良好并行性，尤其在小批量场景下优于 Orin Nano。
- **准确性保持**：
  - 对量化鲁棒的模型（ResNet18, EfficientNet-Lite）可通过轻量级恢复策略（#1–#2）维持精度。
  - 对敏感模型（MobileNetV3, FastViT-SA12）需更强策略（#3/#4）才能缓解精度下降。

### 消融实验结果
- **Quantization Restoration 的影响显著**：
  - 不使用任何恢复时，FastViT-SA12 在 CIFAR-100 上准确率从 71%+ 降至不足 50%。
  - 使用 AdaRound (#4) 可有效重建特征保真度，恢复梯度稳定性。
- **策略成本权衡建议**：
  - 对标准 CNN：优先尝试 Eq. 或 IBC（低成本）。
  - 对量化敏感模型：推荐 FT (#3) 作为性价比最优选择；仅当精度要求极高时使用 AR (#4)。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **推理加速器可用于训练加速**：Hailo-8L 这类专为 inference 设计的 edge AI accelerator，可被有效复用于 on-device training 中的 frozen backbone 特征提取，大幅提升效率。
2. ✅ **异构执行显著提升能效与时效性**：将 heavy feature extraction offload 到专用硬件，使低端设备（如 RPi 5）也能实现高频次、低延迟的模型更新。
3. ✅ **Post-training quantization restoration 至关重要**：尤其是在处理 Hard-Swish 或 self-attention 等对量化敏感的操作时，必须结合 AdaRound 或 knowledge distillation 微调等技术以避免严重精度退化。
4. ✅ **小 batch 表现优异**：Hailo-8L 在 batch=1 场景下表现突出，适合真实边缘场景中数据流式输入的应用。

### 方法的局限性
- ❌ **仅适用于 frozen-backbone fine-tuning**：不支持 full-model training 或参数高效微调以外的策略（如 LoRA、adapter injection）。
- ❌ **Head 复杂度成为瓶颈**：若分类头本身较重（如 MobileNetV3），则 CPU 端反向传播仍会拖慢整体性能。
- ❌ **跨设备通信开销未完全消除**：尽管减少了计算量，但 feature tensor 的 PCIe 传输仍有一定延迟。
- ❌ **未测量峰值内存占用**：缺乏对内存压力的量化分析，可能影响实际部署可行性。
- ❌ **评估范围有限**：目前仅限图像分类任务，尚未验证在检测、分割或多模态任务上的效果。

### 未来工作方向
- 🔁 扩展至 **continual learning**、**test-time adaptation** 和 **streaming scenarios**。
- 🛠️ 引入 **catastrophic forgetting mitigation** 技术，例如 experience replay、drift detection 和 selective unfreezing。
- 🧩 支持更灵活的 PEFT（Parameter-Efficient Fine-Tuning）模块（如 Adapter、LoRA）在 host CPU 上训练。
- ⚙️ 优化 host-accelerator 数据流，进一步减少 quantize/dequantize 和传输延迟。
- 🌐 探索在更多类型 edge accelerator（如 Google Edge TPU、Intel Movidius）上的迁移适用性。

---

> 📦 **开源代码**：https://github.com/MatPiech/accelerator-training  
> 📘 **论文链接**：arXiv:2607.18101 [cs.LG]

</details>

---

### 8. [FlowBlock: Wavefront-Parallel Decoding for Self-Correcting Diffusion Language Models](https://arxiv.org/abs/2607.17652)

**Authors**: Bing Tian, Haikun Liu, Xiaocheng Zhong, Zhuohui Duan, Zhaokai Luo, Huayi Jin, Zhiyong Wang, Xiaofei Liao  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.17652v1  

#### Abstract
Block-wise diffusion large language models (dLLMs) decode sequentially at the block level, enabling effective KV-cache reuse across blocks but making inter-block decoding strictly serial. Prior work has attempted to unlock inter-block parallelism through post-training methods, but achieves only mode...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FlowBlock: Wavefront-Parallel Decoding for Self-Correcting Diffusion Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统的 **block-wise diffusion LLMs**（如 LLaDA-2.0/2.1）虽然通过块级并行提升了生成效率，但仍存在**严格的块间串行依赖**：下游块 $B_{q+1}$ 必须等待上游块 $B_q$ 完全“冻结”后才能开始解码。这导致 late-stage denoising 步骤无法并行化，限制了吞吐量。

尽管已有工作（如 **D2F**）尝试通过**训练后蒸馏**实现跨块并行，但其效果有限，且常伴随**准确率下降**和**批处理扩展性差**的问题。

---

### **提出的新方法与核心思想**
论文提出了 **FlowBlock** —— 一种**无需重新训练**的并行解码框架，利用 **self-correcting dLLMs 中固有的 T2T editing 能力**，将块间依赖从“硬性前提”转化为“可调度资源”。

#### **两大核心技术机制**：

1. **Gated Wavefront Decoding（门控波前解码）**
   - 维护一个大小为 $W$ 的滑动窗口 $[L, R)$，允许多个相邻块并发解码。
   - 引入**就绪门控机制**（readiness gate）：仅当上游块 $B_{R-1}$ 的未掩码位置比例 $\rho(B_{R-1}) \geq \theta_{\text{spawn}}$ 时，才允许 $B_R$ 加入波前。
   - 使用 **W-shaped block-causal attention mask**，确保注意力不泄露到未来块，同时保留已提交前缀的 KV 缓存。
   - 左端块一旦本地完成即按序冻结并提交，保证 KV 缓存完全复用。

2. **Heterogeneous Wavefront Packing（异构波前打包）**
   - 每个请求维护独立的波前状态 $(L_b, R_b)$，避免因最慢请求拖累整个批次。
   - 将不同请求的活动窗口通过 **per-row gather** 打包成固定形状的 $[B, q]$ 张量（$q = W \times \text{block\_length}$），实现高效批处理前向传播。
   - 使用 **绝对位置编码（RoPE）** 和 **逐行 block-diagonal attention mask** 实现跨序列隔离与缓存正确性。
   - 新生成的 KV 通过 scatter 写回预分配的绝对位置缓存中。

---

### **相比现有方法的优势**

| 方面 | FlowBlock | D2F（训练基线） |
|------|---------|---------------|
| **是否需要训练** | ❌ 不需要（training-free） | ✅ 需要蒸馏训练 |
| **准确性** | 更高（平均 +1.3 pts vs LLaDA-2.1） | 显著更低（平均低 ~16 pts） |
| **吞吐量扩展性** | 极强，随 batch size 线性增长 | 差，甚至随 batch 增大而下降 |
| **KV-cache 复用** | 完全精确复用已提交前缀 | 可能近似或破坏一致性 |
| **部署灵活性** | 可直接用于任何支持 T2T 的 checkpoint | 依赖特定蒸馏模型 |

> ✅ **核心优势总结**：**无需训练、更高精度、更强扩展性、更优延迟-吞吐权衡**

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共 8 个基准，涵盖数学与代码任务：

- **数学类**：
  - GSM8K（小学数学题）
  - MATH500
  - Minerva-Algebra
  - ASDiv
- **代码类**：
  - HumanEval
  - MBPP
  - HumanEval+
  - MBPP+

---

### **实验设置**

- **模型**：LLaDA-2.1-mini（MoE 结构，支持 T2T editing）
- **实现平台**：基于 **dInfer**（SGLang 支持的 dLLM 推理框架）
- **参数配置**：
  - Block length: 32
  - Max generation length: 2048
  - Wavefront width $W$: 默认 2
  - Admission threshold $\theta_{\text{spawn}}$: 默认 0.6
- **硬件**：8×GPU（每卡 80GB）节点，数据并行

---

### **评估指标**

| 指标 | 含义 |
|------|------|
| **Acc (%)** | 准确率 / Pass@1 |
| **TPF** | Tokens Per Forward（每步生成 token 数，衡量并行度） |
| **TPS** | Tokens Per Second（端到端吞吐量） |
| **Lat (s)** | 平均每请求延迟（seconds） |

---

### **基线方法对比**

| 基线 | 描述 |
|------|------|
| **LLaDA-2.0** | 无 T2T 自纠错能力的标准 block-wise dLLM |
| **LLaDA-2.1** | 支持 T2T editing 的自纠正 dLLM，串行块解码 |
| **D2F** | 训练基线，通过蒸馏实现跨块并行，使用 LLaDA-2.0-checkpoint 微调 |

> 所有方法在相同引擎、缓存布局和评估流程下运行，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Batch Size = 1）**

见 **Table 1**：

| 方法 | Avg Acc (%) | Avg TPS | Lat (s) |
|------|-------------|--------|--------|
| LLaDA-2.0 | 84.63 | 115.7 | 4.20 |
| LLaDA-2.1 | 85.70 | 177.9 | 2.43 |
| D2F | 70.52 | 111.4 | 4.92 |
| **FlowBlock** | **87.00** | **261.2** | **1.60** |

- **TPS 提升**：
  - 相比 LLaDA-2.1：最高 **1.57×**
  - 相比 LLaDA-2.0：最高 **3.17×**
- **延迟降低**：
  - 相比 LLaDA-2.1：最多减少 **33.3%**
  - 相比 LLaDA-2.0：最多减少 **54.5%**
- **准确率提升**：平均 **+1.3 pts**，所有代码任务均有显著增益（+1.3~4.9 pts）

---

### **批处理场景下的性能表现（Batch Size up to 32）**

见 **Figure 3 & 4**：

- **峰值吞吐提升**：
  - 相比 LLaDA-2.1：最高 **2.95×**（HumanEval @ B=32）
  - 相比 LLaDA-2.0：最高 **4.01×**（Minerva-Algebra @ B=8）
- **延迟降低**：
  - 最多减少 **77.1%**（vs LLaDA-2.0）
  - 在 B=8 时，FlowBlock 即可在多个任务上匹配甚至超越 LLaDA-2.1 @ B=32 的吞吐，但延迟更低

- **D2F 表现不佳**：
  - 吞吐几乎不随 batch 增长，甚至下降（如 HumanEval 从 113 → 89 TPS）
  - 延迟飙升至 203 秒（@ B=32）
  - FlowBlock 达到其 **16.6× 吞吐优势** 与 **95.8% 延迟缩减**

---

### **消融实验结果**

#### **(1) Heterogeneous Wavefront Packing 的有效性（Figure 5）**
- 对比 batch-synchronous wavefront（共享 [L,R)）：
  - 在 B=16 时，HWP 实现 **2.15× TPS 提升**
  - 在 B=32 时仍保持 **1.87× 更高 TPS** 和 **38% 更低延迟**
- 证明异构打包有效缓解“straggler problem”

#### **(2) 参数敏感性分析（Figure 6）**
- **$\theta_{\text{spawn}}$** 控制速度-精度权衡：
  - 过低（<0.5）导致准确率下降（最多 -4 pts）
  - $\theta_{\text{spawn}}=0.6$ 可恢复串行准确率水平
- **Window width $W$** 影响计算开销：
  - $W$ 越大，每步 query 数越多，吞吐下降
  - $W=2$ 是最优选择，在性能与成本间取得平衡

#### **(3) 块长度与生成长度鲁棒性（Table 2 & 3）**
- **块长度增大（64→128）**：
  - LLaDA-2.1 和 D2F 准确率显著下降
  - FlowBlock 保持稳定（Acc 仅从 92.19 → 80.06），TPF 始终领先
- **生成长度变化（512→2048）**：
  - FlowBlock TPF 恒定（~6.73），延迟始终最低
  - D2F 性能受限，TPS ≤109，Acc ~85%

---

## **4. 关键结论和发现**

### **主要发现**

1. ✅ **T2T editing 天然支持跨块并行**：无需额外训练，即可通过调度机制释放 block-wise dLLMs 的并行潜力。
2. ✅ **Gated Wavefront + Heterogeneous Packing 是高效批处理的关键**：既保证安全性与缓存一致性，又消除同步瓶颈。
3. ✅ **FlowBlock 在精度、吞吐、延迟三方面全面超越现有方法**：
   - 平均准确率最高
   - TPS 提升达 **4×**
   - 延迟降低超 **77%**
4. ✅ **训练-free 设计更具实用价值**：避免复杂蒸馏流程，适用于任意具备 T2T 能力的 checkpoint。

---

### **方法的局限性**

- **依赖 T2T 自纠错能力**：仅适用于支持 token-to-token editing 的 dLLM（如 LLaDA-2.1），对传统 absorbing-state dLLMs 不适用。
- **窗口宽度 $W$ 存在计算开销**：更大的 $W$ 带来更多 query 计算，需权衡收益与成本。
- **当前未探索动态调整 $\theta_{\text{spawn}}$**：固定阈值可能非全局最优，未来可结合序列难度自适应调节。

---

### **未来工作方向**

1. **动态门控策略**：根据序列复杂度、模型置信度动态调整 $\theta_{\text{spawn}}$。
2. **扩展至其他生成范式**：探索在 non-diffusion 模型中模拟类似 wavefront 调度的可能性。
3. **硬件感知优化**：进一步优化 gather/scatter 开销，适配不同 GPU 架构。
4. **多模态扩散模型中的应用**：将 wavefront-parallel 思想推广至图像、音频等序列生成任务。

---

> 🔚 **总结一句话**：  
> **FlowBlock 成功将 self-correcting dLLMs 的内在编辑能力转化为高效的推理并行性，在无需训练的前提下实现了高达 4× 的吞吐提升与显著延迟降低，是迈向高性能、实用化扩散语言模型推理的重要一步。**

</details>

---

### 9. [SelKV: Selective KV Cache Merging with Per-Token Merge-or-Drop and Attention Compensation](https://arxiv.org/abs/2607.16213)

**Authors**: Soumia Bouyahiaoui, Manel Kara laouar, Aicha Boutorh, Mohamed Hadj Ameur  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.16213v1  

#### Abstract
Large Language Models (LLMs) generate text autoregressively, relying on a key-value (KV) cache whose memory footprint grows linearly with context length, creating a major bottleneck. Recent compression methods mitigate this cost via token merging; however, these approaches often rely on indiscrimina...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SelKV: Selective KV Cache Merging with Per-Token Merge-or-Drop and Attention Compensation**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**

大型语言模型（LLMs）在自回归生成过程中依赖 Key-Value (KV) cache 来存储历史注意力状态，其内存占用随上下文长度线性增长，成为长上下文推理的主要瓶颈。现有 KV cache 压缩方法存在两个关键缺陷：

1. **统一的合并/丢弃决策（Uniform merge-or-drop）**：大多数方法对所有被压缩的 token 采用相同的策略——要么全部合并，要么全部丢弃。这种“一刀切”的方式在语义不相似的 token 上进行合并会污染保留的表示，导致性能下降。
2. **注意力下垂（Attention Sag）**：当多个 token 被合并为一个缓存条目时，该条目在 softmax 注意力中仍只获得与单个 token 相当的权重，导致信息未被充分使用。

### **提出了什么新方法或新思路**

本文提出 **SelKV**，一种无需训练、即插即用的 KV cache 压缩框架，包含两个核心机制：

1. **Soft Cosine Gate（软余弦门）**  
   - 在每个被压缩的 token 上，计算其 value 向量与其目标合并位置 value 向量之间的余弦相似度。
   - 定义门控函数 $ g = \max(\text{cos\_sim}, 0) $，实现连续的 per-token 合并强度控制：
     - $ g \approx 1 $：高度相似 → 全部合并
     - $ 0 < g < 1 $：部分相似 → 部分合并
     - $ g = 0 $：正交/无关 → 完全丢弃
   - 无需学习参数或手动调参，完全基于表示相似性自适应决策。

2. **Attention-Ratio Compensation（注意力比率补偿）**  
   - 引入基于预填充阶段 attention 统计的 logit 偏置，在解码时校正 softmax 不平衡。
   - 补偿因子定义为：
     $$
     R_{h,i} = \frac{\alpha_{h,i}^{\text{kept}} + \sum_j g_{h,j} \cdot \alpha_{h,j}}{\alpha_{h,i}^{\text{kept}}}
     $$
     并在解码时添加 $ \alpha \cdot \log(R_{h,i}) $ 到 attention logits。
   - 相比简单的 `log(1+M)` 计数补偿更稳定、更自然地反映实际注意力再分配。

### **相比现有方法的优势**

| 特性 | SelKV | 其他方法（如 KVMerger, D2O, SnapKV） |
|------|-------|-------------------------------|
| 合并粒度 | Per-token 连续门控 | 统一合并或丢弃 |
| 注意力补偿 | 基于 attention ratio 的 logit bias | 无补偿 或 简单计数补偿（不稳定） |
| 是否需训练 | ❌ 完全无需训练 | 多数无需训练 |
| 可插拔性 | ✅ 可集成到现有 pipeline 中 | 通常独立运行 |
| 语义保真度 | 更高（避免有害合并） | 易受 dissimilar token 合并影响 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

- **LongBench [1]**：包含 **16 个英文数据集**，覆盖六大任务类别：
  - 单文档问答（Single-Doc QA）
  - 多文档问答（Multi-Doc QA）
  - 摘要生成（Summarization）
  - 少样本学习（Few-shot）
  - 合成任务（Synthetic）
  - 代码补全（Code）

### **实验设置和评估指标**

- **模型**：
  - **LongChat-7B-v1.5-32k**（MHA 架构，32 KV heads）
  - **LLaMA-3.1-8B-Instruct**（GQA 架构，8 KV heads）
  - **Gemma-2-9B-IT**（GQA 架构，8 KV heads，交替滑动窗口 attention）

- **上下文长度**：
  - LongChat：31,500 tokens
  - 其他模型：3,500 tokens

- **KV 缓存保留率**：统一设定为 **25%**（即压缩 75%）

- **评估指标**：
  - 各任务使用标准指标（F1、ROUGE、Accuracy、Code Similarity 等）
  - 报告平均得分（Avg）、与 Full Cache 的差距（△）、平均排名（Avg Rank）

- **硬件环境**：NVIDIA H100 80GB GPU，FP16 推理

### **基线方法对比**

- **SnapKV [7]**：基于观察窗口的重要性评分 + 固定窗口 eviction
- **LOOK-M [12]**：局部滑动窗口 token merging
- **PyramidKV [3]**：层自适应的 token eviction
- **Full Cache**：完整 KV cache 作为上界基准

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 模型 | 方法 | Avg Score | △ vs Full Cache | 是否领先 |
|------|------|-----------|------------------|---------|
| **Gemma-2-9B-IT** | **SelKV (ours)** | **42.31** | **-0.67** | ✅ 最佳 |
| | SnapKV | 42.16 | -0.82 | ❌ |
| **LLaMA-3.1-8B** | **SelKV (ours)** | **39.81** | **-0.73** | ✅ 并列最佳 |
| | SnapKV | 39.79 | -0.75 | ❌ |
| **LongChat-7B** | PyramidKV | 33.95 | -0.80 | ✅ 最佳 |
| | SelKV | 32.83 | -1.92 | ❌（MHA 下表现较弱） |

> 注：在 **GQA 架构模型** 上，SelKV 表现最优；在 MHA 上略逊于 PyramidKV。

### **与基线方法的对比结果**

- **总体性能**：
  - 在 GQA 模型上，SelKV 是唯一能接近甚至略微超越 Full Cache 性能的方法。
  - 在 **多文档 QA** 任务中（如 HotpotQA, MuSiQue），SelKV **多次超过 Full Cache 基线**，表明选择性合并可起到“隐式注意力过滤”作用，抑制干扰信息。

- **速度提升**：
  - 在 100k tokens 输入下，SelKV 实现 **3.3× 的解码加速**（28 tok/s vs 8.4 tok/s）。
  - 压缩开销低于总耗时的 5%，几乎可忽略。

- **跨架构鲁棒性**：
  - 在 LLaMA-3.1 上扩展至 31,500 tokens 时，SelKV 与 SnapKV 并列第一（48.46），优于 PyramidKV 和 LOOK-M。

### **消融实验结果**

| 配置 | Avg (LLaMA-3.1, 16 datasets) | 相对变化 |
|------|-------------------------------|--------|
| Full Cache | 40.54 | — |
| Eviction only | 39.69 | -0.85 |
| Merge all（无门控） | 39.01 | -0.68 vs eviction |
| + Soft Cosine Gate | 39.57 | +0.56 |
| + Gate + Compensation | **39.81** | **+0.24** |

> 结论：
> - 盲目合并（Merge all）比直接丢弃还差，说明有害合并确实存在。
> - Soft Cosine Gate 显著恢复性能损失。
> - Attention Compensation 进一步带来稳定增益。

此外：
- **RoPE Repositioning** 默认关闭，因其在长文本检索任务中损害绝对位置感知。
- **压缩比鲁棒性测试**（10%~90% retention）显示：
  - 在 ≥25% 保留率时，SelKV 在 Gemma-2 上始终领先；
  - 在 **80% retention** 时，SelKV 得分 **43.01 > Full Cache 42.98**，首次实现“压缩后更强”。

---

## **4. 关键结论和发现**

### **主要发现**

1. **Merge Quality Matters**：不仅仅是“选哪些 token 保留”，更重要的是“如何处理被压缩的 token”。合并的质量直接影响最终性能。
2. **Selective Merging 是有效的**：通过 soft cosine gate 实现 per-token 的连续合并控制，能有效防止语义冲突，提升表示保真度。
3. **Attention Sag 必须补偿**：仅靠合并不足以释放信息价值，必须通过 attention-ratio logit bias 主动纠正 softmax 分配偏差。
4. **GQA 模型特别受益**：由于 Query Heads 共享 KV Heads，token selection 更稀疏，因此 selective merging 的优势更为明显。
5. **适度压缩可能优于原始模型**：在某些任务和压缩比例下（如 80% retention），SelKV 能轻微超越 Full Cache，说明其具有“去噪”或“注意力聚焦”效应。

### **方法的局限性**

- 在 **MHA 架构** 上表现不如 GQA 显著，可能因 per-head selection 导致缓存碎片化。
- 在 **代码补全任务** 上性能稍弱，因精确 token 序列匹配对合并操作更敏感。
- 在极高压缩比（如 10% retention）下，merge targets 过少，soft gate 效果受限。
- 当前仅验证至 31,500 tokens 上下文，超长上下文（>100k）的生成质量尚未系统评估。

### **未来工作方向**

- 扩展至更多模型架构（如 MoE、Transformer-XL）。
- 与量化方法（如 KIVI、ZipCache）结合，实现复合压缩。
- 探索训练时适配机制，将 soft gate 参数化以支持任务自适应策略。
- 开展更大规模的超长上下文（>100k）生成质量评估。
- 将该框架应用于多模态 LLM 的 cross-attention cache 压缩。

--- 

> ✅ **一句话总结**：  
> SelKV 通过 **soft cosine gate** 实现 per-token 自适应合并，并引入 **attention-ratio compensation** 解决 attention sag，实现了高效且高质量的 KV cache 压缩，在 GQA 模型上达到近无损甚至反超 Full Cache 的效果，同时带来 **3.3× 解码加速**。

</details>

---

### 10. [Reward-Driven LLM Agent Workflows: Synthesizing POMDP Routing and Self-Correction for Autonomous Decision-Making](https://arxiv.org/abs/2607.17038)

**Authors**: Amez Amanj Ali, Kuo-Kun Tseng  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.17038v1  

#### Abstract
This paper addresses key technical challenges in current large language model (LLM) agent applications, including long-horizon planning, sparse reward attribution, and dynamic environmental interaction, by designing and optimizing an intelligent agent workflow. The proposed architecture is based on ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Reward-Driven LLM Agent Workflows: Synthesizing POMDP Routing and Self-Correction for Autonomous Decision-Making*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Large Language Model (LLM)** 的智能体在复杂任务中面临三大核心挑战：
- **长时程规划（Long-horizon planning）**：推理链过长导致错误累积（cascading hallucination errors）。
- **稀疏奖励归因（Sparse reward attribution）**：缺乏对动作价值的显式评估机制，难以优化长期目标。
- **动态环境交互能力弱**：传统方法依赖静态提示（prompting），缺乏感知-行动-反馈闭环。

这些问题使得主流 LLM Agent 在真实、部分可观测（Partially Observable）环境中容易陷入无限循环、执行无效操作或遗忘关键状态。

---

### 提出了什么新方法或新思路
本文提出了一种全新的 **Reward-Driven LLM Agent Workflow (RLAW)** 架构，其核心创新在于将 **Reinforcement Learning (RL)** 的形式化数学框架与 LLM 的生成能力深度融合，构建一个“**Propose-Critique-Execute**”的闭环决策流程。

#### 主要技术亮点：
- **POMDP 路由机制**：将 LLM 显式建模为 Partially Observable Markov Decision Process 中的策略函数（Policy Function），通过定义状态空间 $S$、动作空间 $A$ 和转移动态 $T$，赋予生成过程数学严谨性。
- **自纠正奖励模型（Self-Correcting Critique Module）**：引入一个轻量级的 Critic 模型，在动作执行前对候选推理路径进行内部评估，输出连续标量奖励 $R_{\text{critique}}$，过滤逻辑错误或不安全动作。
- **图神经网络记忆模块（Graph Memory Bank）**：利用 **Graph Neural Networks (GNNs)** 维护结构化的世界状态表示（如实体间空间/功能关系），缓解长期记忆丢失问题。
- **双目标优化函数**：结合外部环境稀疏奖励 $R_{\text{env}}$ 与内部密集批评奖励 $R_{\text{critique}}$，实现更高效的策略学习：
  $$
  J(\theta) = \mathbb{E}\left[\sum_t \gamma^t (R_{\text{env}} + \lambda R_{\text{critique}})\right]
  $$

---

### 相比现有方法的优势
| 方面 | 传统方法（如 ReAct） | RLAW |
|------|------------------------|-------|
| 决策机制 | 零/少样本 Chain-of-Thought 推理 | 数学驱动的 POMDP 框架 |
| 错误处理 | 依赖环境反馈纠错，易陷入死循环 | 执行前主动拦截错误（Pre-execution filtering） |
| 记忆结构 | 简单文本历史（FIFO） | 结构化 Graph Memory 表征 |
| 学习目标 | 最大似然生成 | 显式最大化期望回报 |
| 可扩展性 | 依赖大模型参数规模 | 小模型也能高效运行 |

> ✅ **优势总结**：RLAW 实现了从“盲目生成”到“有目的推理”的范式转变，显著提升可靠性、效率和泛化能力。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在三个具有代表性的基准上进行，涵盖物理模拟与数字交互场景：

1. **ALFWorld**  
   - 类型：基于 TextWorld 的具身仿真环境（embodied simulation）
   - 任务：多房间导航 + 物品操作（如“清洗苹果并放在桌上”）
   - 特点：部分可观测、高维状态空间、需长期记忆

2. **WebShop**  
   - 类型：大规模电商网站导航任务
   - 数据量：超 110 万商品，真实用户查询
   - 动作空间：`search`, `click`, `select`, `buy`
   - 挑战：搜索结果随机变化、分页导航、属性匹配

3. **Custom Multimodal Simulation Platform**（定制平台用于补充验证）

---

### 实验设置和评估指标

#### 模型配置
- **主干模型**：LLaMA-3-8B-Instruct（4-bit 量化以适配单卡 RTX 4090）
- **Critic 模型**：基于 LoRA 微调的 Meta-Llama-3-8B-Instruct，训练数据含 5 万条推理-批评轨迹
- **推理参数**：
  - 温度 $T=0.2$，top-p=0.9
  - 折扣因子 $\gamma=0.95$，批评阈值 $T=0.7$，奖励权重 $\lambda=0.5$
  - 上下文长度上限：4096 tokens

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Task Success Rate (SR%)** | 成功完成任务的比例（严格标准，部分成功计为失败） |
| **Average Steps (AvgS)** | 成功任务的平均步数，衡量轨迹效率 |
| **Hallucination Error Rate (%)** | 尝试执行不可能动作的比例（如“从关闭的冰箱取物”） |
| **Parameter Efficiency** | 小模型是否能超越更大模型的表现 |

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Base-LLM (Zero-Shot)** | 仅使用 LLaMA-3-8B 进行一次性零样本推理 |
| **Standard ReAct [2]** | 当前主流框架，CoT + 工具调用，无内部验证机制 |
| **Reflexion [13], Tree/Graph of Thoughts [18,19]** | 引用作为相关工作比较对象 |

所有模型均在同一硬件条件下测试，共 500 个 unseen 测试任务，确保统计显著性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 4）

| Model Strategy | Params (B) | ALFWorld SR(%) | ALFWorld AvgS | WebShop SR(%) | WebShop AvgS |
|---------------|------------|----------------|----------------|----------------|----------------|
| Base-LLM (Zero-Shot) | 8B | 22.4 | 15.2 | 18.6 | 12.8 |
| Standard ReAct | 8B | 54.1 | 12.1 | 42.3 | 10.5 |
| **RLAW (Ours)** | **8B** | **78.6** | **9.4** | **65.8** | **8.2** |

> 🔺 **绝对提升**：相比 ReAct，**ALFWorld 上 SR 提升 24.5%**，**WebShop 上提升 23.5%**

---

### 与基线方法的对比结果
- **成功率大幅提升**：RLAW 在两个基准上均取得压倒性胜利，尤其在复杂 ALFWorld 任务中接近翻倍于零样本模型。
- **轨迹更高效**：平均步数显著减少（ALFWorld 从 12.1 → 9.4），说明避免了冗余动作和死循环。
- **幻觉率大幅下降**：
  - ALFWorld 幻觉率从 Base-LLM 的 45.1% 降至 **12.4%**
  - WebShop 从 48.3% 降至 **14.1%**
  > 👉 证明 Critique 模块有效拦截非法动作。

- **参数效率惊人**：如 Figure 4 所示，**8B 参数的 RLAW 超越了 70B 参数的 ReAct 模型（78.6% vs 75%）**，表明架构设计优于单纯扩大模型规模。

---

### 消融实验结果（Ablation Study）

| Configuration | Success Rate (%) | Avg Steps | Critique Rejections |
|--------------|------------------|-----------|---------------------|
| Full RLAW Architecture | **78.6** | **9.4** | 2.1 |
| w/o Critique Module | 61.2 | 11.5 | — |
| w/o Memory Module | 68.4 | 10.2 | 1.5 |
| w/o Both (Standard ReAct) | 54.1 | 12.1 | — |

#### 发现：
- 移除 **Critique Module** 导致性能暴跌 **17.4%**，是最大贡献组件。
- 移除 **Graph Memory** 也造成严重退化（↓10.2%），说明结构化记忆至关重要。
- Critique 平均每轮拒绝 2.1 次错误提议，说明即使强大 LLM 也会频繁出错，亟需验证机制。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **形式化奖励机制可显著增强 LLM Agent 的可靠性**：引入基于 RL 的 Critique 模块，使 LLM 从“被动生成器”转变为“主动价值最大化者”。
2. **Propose-Critique-Execute 闭环优于 Open-loop 生成**：提前过滤错误比事后修复更高效、更安全。
3. **架构创新 > 参数堆叠**：RLAW 证明，精心设计的 agent workflow 可让小模型超越大模型，打破“越大越好”的迷思。
4. **Graph Memory 对长期记忆至关重要**：非欧几里得结构能有效压缩和检索复杂环境状态。

---

### 方法的局限性
1. **推理延迟较高**：由于 Critique-Cycle，每步耗时约 2.15s（ReAct 为 1.20s），不适合实时控制任务（如无人机飞行）。
2. **上下文窗口限制**：尽管有摘要机制，但在极长任务（>100 步）中仍可能出现“attention dilution”或“lost in the middle”现象。
3. **Critic 模型质量依赖性强**：若 Critic 自身产生幻觉，可能导致 false positive/negative 判定，甚至引发无限再生循环。
4. **当前仍以文本为主**：虽支持 Multimodal Fusion，但视觉输入尚未完全融入核心决策流。

---

### 未来工作方向
1. **开发自校准 Critic 模型**：使其能根据任务难度动态调整判断阈值，提高鲁棒性。
2. **深度融合 Visual Intelligence**：接入实时视频流，实现真正意义上的具身智能（Embodied AI）。
3. **探索异构 Agent 协同**：结合多个专业化子 Agent（如 Planner、Executor、Validator）形成协作系统。
4. **降低计算开销**：研究蒸馏版 Critic 或缓存机制，提升推理速度，适用于边缘设备部署。
5. **拓展至工业级应用**：应用于自动化运维、金融交易、智能制造等高风险领域，强调安全性与可解释性。

---

> 📌 **最终结论**：  
> 本研究成功将 **Reinforcement Learning** 的理论基础与 **LLM Agent** 的实践需求相结合，提出了首个具备数学严谨性和工程可行性的 **reward-driven self-correction framework**。RLAW 不仅在性能上全面超越主流方法，更为下一代自主智能系统的构建提供了可复用、可扩展的设计范式。代码已开源：[GitHub - RLAW_Implementation](https://github.com/01Amez/RLAW_Implementation)

</details>

---

### 11. [A Hardware-oriented Approach for Efficient Bayesian Inference Computation and Deployment](https://arxiv.org/abs/2607.17855)

**Authors**: Nikola Pi\v{z}urica, Matteo Risso, Nikola Milovi\'c, Alessio Burrello, Igor Jovan\v{c}evi\'c, Conor Heins, Miguel de Prado  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.17855v1  

#### Abstract
Bayesian inference provides a principled foundation for reasoning under uncertainty, but its computational cost hinders deployment on resource-constrained edge devices. In this paper, we present a hardware-oriented methodology for accelerating discrete Bayesian inference on commercial off-the-shelf ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《A Hardware-oriented Approach for Efficient Bayesian Inference Computation and Deployment》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
贝叶斯推理（Bayesian inference）在边缘设备（edge devices）上的部署面临严重的计算效率瓶颈。尽管其为不确定性建模提供了坚实的数学基础，但**变分消息传递算法**（如 VMP、MMP 和 FPI）中的张量收缩（tensor contractions）操作具有高计算复杂度，导致在资源受限的嵌入式平台上延迟过高，难以满足实时应用需求。

此外，现有概率编程框架（如 pymdp、Pyro、NumPyro）虽然支持 GPU 加速，但其底层编译器生成的代码对硬件不敏感，未能充分挖掘嵌入式 GPU 的并行潜力。

---

### 提出了什么新方法或新思路
本文提出了一种**面向硬件的优化方法论**，专注于加速离散贝叶斯推理中占主导地位的**张量收缩操作**，并在商用现成的嵌入式 GPU（NVIDIA Jetson Orin AGX）上实现高效执行。

#### 核心创新点包括：

- **统一的低级原语优化策略**  
  将多种 message-passing 算法（FPI、VMP、MMP）视为由相同计算原语（即 tensor contractions）构建而成，因此通过优化这些通用原语来实现跨算法的通用加速。

- **两种张量合并策略（Tensor Merging Strategies）**
  1. **轴对齐填充与合并**（Axis-aligned padding and merging）：将不同形状的输入数组零填充至统一维度后沿 batch 轴拼接，形成规则的大张量，便于 GPU 并行处理。
  2. **块对角合并**（Block-diagonal merging）：适用于一个操作数为高维、另一个为一维的情形（如 log-likelihood 计算），将其转化为标准的矩阵-向量乘法，极大提升硬件利用率。

- **内存优化机制**
  - **稀疏表示**（BCOO sparse representation）：利用 JAX 的批量化坐标格式（Batched COO）减少存储开销。
  - **张量聚类**（Tensor clustering）：将相似形状的张量分组进行局部合并，显著降低填充带来的冗余计算和内存占用。

- **基于机器学习的自动调优器**（ML-based autotuner）
  引入一个轻量级的分类模型（XGBoost），根据 POMDP 模型配置预测最优的算法变体，避免部署时耗时的全量基准测试（平均节省 4–25 分钟）。

---

### 相比现有方法的优势
| 维度 | 本文方法 | 现有方法 |
|------|--------|---------|
| **通用性** | 支持多种 message-passing 算法（FPI/VMP/MMP） | 多针对特定算法或模型结构 |
| **硬件适配性** | 显式优化内存布局以匹配 GPU 架构 | 依赖高级框架自动优化，效果有限 |
| **数值精度** | 完全等价于原始实现（lossless） | 常引入近似以换取速度 |
| **部署效率** | 自动选择最佳配置（autotuning） | 需手动调参或遍历测试 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
未使用传统意义上的公开数据集，而是采用**随机生成的 POMDP 配置集合**，共 **770 种不同的配置**，模拟真实世界中多样化的应用场景。

每个 POMDP 配置由以下参数定义：
- 隐藏状态因子数量 $ F \in \{5, 10, 25, 125\} $
- 观测模态数量 $ M \in \{5, 10, 25, 125\} $（且 $ F \leq M $）
- 各因子/模态的维度范围（从 2 到 5/10/25 不等）
- A-dependency 列表（决定观测如何依赖隐藏状态，长度服从指数先验）

生成过程分为三个阶段：
1. **粗粒度维度选择**
2. **细粒度采样**（均匀 vs 偏斜分布）
3. **A-dependency 生成**（带负相关约束以防组合爆炸）

所有 A/B 参数矩阵均从均匀分布采样并归一化为合法概率分布，持久化保存以确保公平比较。

---

### 实验设置和评估指标

- **硬件平台**：NVIDIA Jetson Orin AGX（嵌入式 GPU 设备）
- **对比算法**：三种 message-passing 算法
  - **FPI**（Fixed Point Iteration，仅 likelihood messages）
  - **VMP**（Variational Message Passing）
  - **MMP**（Marginal Message Passing）
- **实现方式**：基于 JAX 在 pymdp 基础上重构，应用九种优化变体（见 Table I）
- **评估指标**：
  - **总推理延迟**（total inference latency）
  - **加速比**（latency ratio = baseline / optimized）
  - **regret**（用于衡量 autotuner 性能，表示选错配置导致的额外延迟比例）

---

### 基线方法对比
- **Baseline**：原始的 `pymdp` 实现（loop-based，逐个处理张量）
- **Optimized variants**：本文提出的九种优化版本，涵盖：
  - Hybrid vs End-to-end
  - Axis-aligned vs Block-diagonal merging
  - Sparse (BCOO) vs Clustered vs None

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **最大加速比**：高达 **5x**
- **典型加速比**：普遍达到 **2–2.5x**
- **autotuner 准确性**：
  - 对 VMP 和 MMP 的 regret < 2%
  - 对 FPI regret ≈ 3.15%（XGBoost 表现优于 Random Forest）

> 注：autotuner 决策时间仅为 **毫秒级**，而完整 benchmark 平均需 4 分钟，大型模型可达 20–25 分钟。

---

### 与基线方法的对比结果
| 算法 | 最佳加速表现 | 典型加速 | 是否超越 baseline |
|------|-------------|----------|------------------|
| **FPI** | ~2.5x | <1.5x | 是（但提升较小） |
| **VMP** | **4.3x** | ~2.5x | 显著优于 |
| **MMP** | **4.9x** | ~2.5x | 最大收益，尤其峰值 |

- **MMP 表现最好**：因其涉及时间窗口内的批量消息传递，更利于 GPU 并行化。
- **FPI 提升有限**：因只处理单步 likelihood，缺乏结构性规律供优化利用。

---

### 消融实验结果（Ablation Study）

#### （1）合并策略对比
| 策略 | 优势 | 劣势 |
|------|------|-------|
| **Block-diagonal merging** | 更少填充零，鲁棒性强，几乎无性能下降案例 | 适用场景受限（需一维 operand） |
| **Axis-aligned merging** | 更通用 | 在异构形状下易出现 lower tail（性能倒退） |

> ✅ 结论：**hybrid-block** 变体整体表现最稳定。

#### （2）内存优化机制对比
| 方法 | 参数量减少 | 推理加速 | 说明 |
|------|------------|----------|------|
| **BCOO sparse** | ↓14x | ❌ 无明显提速 | 当前 JAX 对 sparse ops 支持不足 |
| **Clustering** | ↓9x | ✅ 显著提速 | 减少冗余计算，提升硬件利用率 |

> ✅ **Clustering 是最有效的优化手段之一**，tightens latency ratio 分布，消除下尾。

#### （3）End-to-end vs Hybrid
- **End-to-end** 方法理论上更彻底地保持合并形式，但实际表现不如 hybrid。
- 原因：全局填充引入过多冗余运算，抵消了中间解构的开销。
- 结果：hybrid 家族平均加速更高，稳定性更好。

#### （4）Autotuner 效果
- autotuner 能有效追踪各配置下的“最优变体”边界（upper envelope）。
- 其延迟分布整体上移，极少低于 baseline，验证了配置感知选择的价值。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **张量收缩是贝叶斯推理的性能瓶颈**，且可通过硬件感知的内存布局重排实现高效加速。
2. **通用原语级优化可跨算法复用**：同一套 merging + clustering 策略成功应用于 FPI、VMP、MMP。
3. **块对角合并 + 聚类是最稳健的组合**，尤其适合高维异构张量场景。
4. **自动调优器可在极短时间内选出接近最优的配置**，极大提升部署灵活性。
5. **无需牺牲数值精度即可获得显著加速**：所有优化均为 lossless transformation。

---

### 方法的局限性
1. **目前仅支持 trivial B-dependencies**（即 $ s_{t+1}^{(j)} \leftarrow s_t^{(j)} $），尚未推广到任意转移结构。
2. **对 JAX 的 sparse ops 支持依赖较强**：当前 BCOO 实现未能带来速度增益，需等待后端成熟。
3. **实验集中在单一硬件平台**（Jetson Orin AGX），泛化性有待在更多边缘设备上验证。
4. **聚类策略设计仍基于启发式规则**（如 elbow point detection），未来可探索 learnable clustering。

---

### 未来工作方向
1. **扩展至任意 B-dependency 结构**，支持更复杂的动态模型。
2. **结合成熟的稀疏计算库**（如 cuSPARSE）进一步释放 BCOO 潜力。
3. **跨平台 benchmarking**：在其他嵌入式 GPU/NPU/FPGA 上验证方法普适性。
4. **将 autotuner 集成进 pymdp 或 JAX 编译流程**，实现端到端自动化部署。
5. **探索 compile-time fusion 与 runtime autotuning 的协同优化机制**。

--- 

> 🔗 **开源承诺**：作者声明代码将在双盲评审结束后公开发布（currently omitted for review）。

</details>

---

### 12. [OpenLanguageModel: Readable and Composable Small-Language-Model Pretraining for Education and Research](https://arxiv.org/abs/2607.16669)

**Authors**: Tavish Mankash, Vardhaman Kalloli, Keshava Prasad, Deepan Muthirayan  
**Category**: cs.CL  
**Published**: 2026-07-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.16669v1  

#### Abstract
OpenLanguageModel (OLM) is an open-source PyTorch library for building and pretraining small language models while keeping their machinery visible. In OLM, model code reads like the architecture: components are ordinary modules, while Block, Residual, Repeat, and Parallel describe how they are wired...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*OpenLanguageModel: Readable and Composable Small-Language-Model Pretraining for Education and Research*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前在 **Small Language Model (SLM)** 教学、研究和实践预训练中存在一个根本性割裂：

- **教学模型** 通常简化过度，无法连接真实训练流程；
- **生产级框架**（如 `transformers`）虽功能完整，但模型架构被封装，难以理解内部机制；
- **研究实验** 需要修改组件时，常需重写大量训练逻辑。

**OpenLanguageModel (OLM)** 的目标是弥合这一鸿沟：让“用于讲解的模型代码”可以直接用于“严肃的预训练和研究实验”。

---

### 🚀 提出的新方法与核心思想

OLM 是一个开源的 PyTorch 库，其核心创新在于将 **模型架构定义为可读、可组合的对象图（object graph）**，并将其无缝接入完整的 SLM 预训练栈。

#### 主要创新点：

| 创新维度 | 内容 |
|--------|------|
| **1. 架构即代码（Readable Architecture）** | 模型定义直接反映 Transformer 图解结构。例如 `Residual`, `Repeat`, `Parallel`, `Block` 等组合器使残差连接、重复层、分支结构一目了然。 |
| **2. 组件化与可组合性（Composable Components）** | 提供独立可导入的模块：`AttentionBase`, `RMSNorm`, `RoPE`, `SwiGLU`, `FlashAttention` 等，支持自由替换与组合。 |
| **3. 教学-训练-研究一体化路径** | 同一个 `torch.nn.Module` 对象可用于：<br>• 教学演示（notebook 中展示）<br>• 完整预训练（通过 `AutoTrainer`）<br>• 局部研究修改（如更换 attention 类型） |
| **4. 自适应训练系统 `AutoTrainer`** | 支持 `device="auto"` 自动检测硬件（CPU / 单GPU / 多GPU），自动选择 DDP 或 FSDP，并集成混合精度、梯度累积、优化调度等。 |

---

### 🔍 相比现有方法的优势

| 工具 | 优势 | OLM 的超越之处 |
|------|------|----------------|
| **HuggingFace Transformers** | 模型丰富，生态强大 | 架构不可见，难以教学；定制需深入源码 |
| **LitGPT** | 轻量训练脚本 | 更关注“如何训练”，而非“如何理解模型结构” |
| **Pico** | 支持假设驱动研究 | 设计较底层，学习曲线陡峭 |
| **PyTorch 原生** | 完全灵活 | 缺乏高层抽象，重复造轮子 |

> ✅ **OLM 的独特定位**：**把“能画在黑板上的模型图”变成“可运行、可扩展、可研究”的生产级代码。**

---

## 2. 核心实验方法和设置

### 📚 数据集

- **FineWeb-Edu**：大规模网页文本子集，用于 SLM 预训练。
- **本地文本数据集**：支持用户自定义 `.txt` 文件输入。
- **流式加载支持**：`streaming=True` 实现高效大数据处理。

### ⚙️ 实验设置

| 项目 | 设置 |
|------|------|
| **模型规模** | 最大测试至 **3.48亿参数**（348M）的 Llama-3-style 模型 |
| **硬件环境** | 1/2/4 × A100-SXM4-80GB GPU |
| **精度模式** | BF16（bfloat16）混合精度 |
| **序列长度** | 2048 tokens per GPU |
| **训练策略** | 使用 `AutoTrainer` 自动配置 DDP（单节点多卡） |
| **评估任务** | 弱扩展性（weak scaling）、组件替换、代码简洁性比较 |

### 📊 评估指标

| 指标类别 | 具体指标 |
|--------|---------|
| **正确性验证** | Logit 差异、Loss 差异、Gradient Cosine Similarity |
| **可复现性** | 参数计数一致性、checkpoint 反向加载 |
| **可定制性** | 修改特定组件所需代码行数（LoC） |
| **扩展性** | Tokens/sec、Weak Scaling Efficiency |
| **可用性** | System Usability Scale (SUS) 问卷得分 |

### 🆚 基线方法对比

- **LitGPT**
- **Pico**
- **HuggingFace Transformers**（作为权重初始化参考）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 指标 | 结果 |
|------|------|
| **最大吞吐量** | **64.4k tokens/sec**（4×A100） |
| **弱扩展效率** | **90.6%**（从1到4 GPU） |
| **显存占用** | 峰值 **10.47 GB/GPU**（2/4 GPU 运行时稳定） |
| **Logit 最大差异** | < `3.58×10⁻⁷`（FP32 下与 HF 实现对比） |
| **梯度余弦相似度** | > `0.9999999999998` |
| **SUS 用户评分均值** | **73.8**（标准范围良好，n=20） |

---

### 🔁 与基线方法对比结果

| 场景 | OLM | LitGPT | Pico |
|------|-----|--------|------|
| **实现相同架构修改（4项任务）** | 平均 **117 LoC** | 171 LoC | 195 LoC |
| **全部6项研究修改总代码量** | **167 LoC** | 256 LoC | — |
| **用户认为架构更易懂** | 20/20 同意 | — | — |
| **认为修改更局部化** | 19/20 同意 | — | — |
| **相比其他工具更易改架构** | 14/17 认为更容易 | — | — |

> ✅ 表明 OLM 在**代码简洁性**和**修改局部性**上显著优于同类工具。

---

### 🔧 消融实验结果（Architecture Edits）

实现了六种局部架构修改，验证系统的灵活性：

1. **Parallel Residual Paths**  
2. **Heterogeneous Transformer Blocks**（不同层用不同结构）
3. **Selected-layer MoE**（仅部分层启用 Mixture-of-Experts）
4. **RoPE → ALiBi 替换**（位置编码更换）
5. **Cross-family Block Mixing**（混合 Llama 与 GPT-2 组件）
6. **N-way Learned Merge**（可学习的多分支融合）

> 所有修改均只需更改一行注意力或模块赋值，其余训练流程（data, optimizer, trainer）完全不变。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **“可读即可用”是可行的**：  
   将教学级模型定义提升为生产级训练入口是可能的，且不牺牲性能。

2. **对象图设计有效支持组合与修改**：  
   `Block`, `Residual`, `Repeat`, `Parallel` 等组合器天然契合 PyTorch 模块系统，实现高内聚低耦合。

3. **研究效率显著提升**：  
   局部组件替换平均仅需百行以内代码，远低于 LitGPT 和 Pico。

4. **高性能与高可用并存**：  
   在 4×A100 上达到 **90.6% 弱扩展效率**，同时用户 SUS 得分达 **73.8**，说明系统既快又好用。

5. **教育价值明确**：  
   文档从 token 开始逐步引导至分布式训练，形成“学习 → 训练 → 研究”闭环路径。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **验证范围有限** | 当前实验集中在 <1B 参数模型，未覆盖超大规模训练 |
| **FSDP 支持待加强** | 当前扩展性测试使用 DDP，FSDP 尚未充分验证 |
| **多节点训练未测试** | 实验仅限单节点（single-node）多GPU |
| **用户样本偏小** | SUS 问卷 n=20，为早期便利抽样，不具备统计普适性 |
| **数值一致性基于简化模型** | 权重对齐测试使用 4-layer 简化版 GPT-2/Llama/Qwen |

---

### 🔮 未来工作方向

1. **扩展至多节点训练**（Multi-node DDP/FSDP）
2. **支持更大规模模型**（>1B 参数）的端到端预训练
3. **增强 FSDP 和 Zero-Redundancy Optimizer 支持**
4. **构建社区 presets 生态**，支持更多新型架构（如 RetNet, Mamba）
5. **集成更多教育工具**：可视化模块流、自动错误诊断、交互式调试接口

---

## 总结

> **OLM 的核心理念是：“你用来解释模型的那几行代码，就应该能直接拿去训练。”**

它成功地将 **PyTorch 的灵活性**、**教学的可读性** 和 **研究的可实验性** 统一在一个 MIT 许可的开源库中，为 SLM 教育与轻量研究提供了一条清晰、实用、高效的路径。

🔗 项目地址：[https://github.com/openlanguagemodel/openlanguagemodel](https://github.com/openlanguagemodel/openlanguagemodel)  
📦 PyPI 安装：`pip install openlanguagemodel`  
📘 文档：[https://openlanguagemodel.github.io/openlanguagemodel/](https://openlanguagemodel.github.io/openlanguagemodel/)

</details>

---

### 13. [AutoEncoder-Compressed Parallel Split Learning for Pre-trained Model Fine-Tuning](https://arxiv.org/abs/2607.17913)

**Authors**: Bas Meuwissen, Vasileios Tsouvalas, Nirvana Meratnia  
**Category**: cs.DC  
**Published**: 2026-07-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.17913v1  

#### Abstract
Distributed Fine-Tuning (DFT) of large-scale Foundation Models (FMs) on resource-constrained edge devices is limited by local compute constraints and communication overhead. Parallel Split Learning (PSL) reduces client-side computation by keeping few model layers on each client and offloading the re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AutoEncoder-Compressed Parallel Split Learning for Pre-trained Model Fine-Tuning

## 1. 论文的主要贡献和创新点

### 解决的问题
- **分布式微调**（Distributed Fine-Tuning, DFT）在资源受限的边缘设备上面临两大挑战：
  - **计算限制**：大型基础模型（Foundation Models, FMs）难以在边缘端完整训练。
  - **通信开销**：Split Learning（SL）虽然将模型拆分以降低客户端计算负担，但每轮训练需传输中间激活值（activations）和梯度（gradients），导致高通信成本，尤其对Transformer类模型更为严重。

- 现有通信压缩方法存在不足：
  - **启发式方法**（如量化、稀疏化）是任务无关的（task-agnostic），可能丢弃对预训练模型表示至关重要的特征。
  - **可学习压缩器**（如AutoEncoder）虽能更好适应中间表示，但通常需要与主模型联合训练，直接插入现成的（off-the-shelf）预训练FM会导致特征分布错位（feature-distribution misalignment），性能下降。

### 提出的新方法
提出 **AE-PSL**（AutoEncoder-Compressed Parallel Split Learning），一种用于高效微调预训练FM的通信压缩框架，其核心创新在于：

1. **AE-Compressed PSL 架构**：
   - 在Split Layer处引入一个轻量级的 **AutoEncoder**（AE）模块。
   - 客户端保留 **Encoder**，服务器端保留 **Decoder**。
   - 在训练过程中，客户端发送的是经过压缩的低维表示 `Z` 和其梯度，而非原始的高维中间特征 `H`，从而双向减少通信量。

2. **两阶段对齐机制**（Two-Stage Alignment）：
   - **通用对齐**（General Alignment, GA）：在公共数据集（如ImageNet）上，冻结预训练模型的客户端部分，仅训练AE模块，使其编码器-解码器对能够重建预训练模型在切分层的特征流形（feature manifold）。这是一次性的、针对特定预训练模型的初始化步骤。
   - **客户端特定对齐**（Client-Specific Alignment, CSA）：每个客户端在本地私有数据上，用GA初始化后的AE进行短暂的“热身”训练（warm-up），使编码器适应其本地的数据分布和特征偏移（non-IID）。之后，各客户端将其训练好的编码器保留在本地，并将解码器权重上传至服务器进行聚合（weight averaging），形成一个全局共享的解码器。

### 相比现有方法的优势
- **兼容性好**：无需修改或重新训练预训练模型，即可无缝集成到现有的预训练FM中。
- **性能更优**：相比启发式压缩方法，在相同通信预算下，能显著保持甚至提升下游任务准确率。
- **通信效率高**：实现了高达约 **10.2×** 的通信量减少，同时几乎不损失精度。
- **收敛更快**：达到与无压缩DFT相当的精度所需通信量仅为最强基线方法的 **1/12.4**。
- **计算开销低**：客户端额外的计算开销（主要是CSA阶段）很小，且在高通信压缩比下，AE-PSL反而能以更少的客户端计算量（GFLOPs）达到峰值性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个公开的视觉分类数据集上进行了评估：
- **CIFAR-100**
- **Food101**
- **SUN397**
- **FEMNIST**

其中，CIFAR-100、Food101、SUN397用于**全局评估**（global evaluation），FEMNIST因其按用户划分的特性，用于**本地评估**（local evaluation）。

### 实验设置
- **模型**：使用ImageNet预训练的 **ViT-B/32** 模型，在第5层（layer s=5）进行切分。
- **微调方法**：采用 **LoRA**（Low-Rank Adaptation）进行参数高效微调（PEFT）。
- **客户端数量**（N）：5 和 25。
- **通信压缩比**（R）：定义为相对于无压缩DFT的通信量减少倍数。设置了三个等级：
  - 低压缩（Low, R~5×）
  - 中压缩（Mid, R~10×）
  - 高压缩（High, R~20×）

### 评估指标
- **下游任务准确率**（Downstream task accuracy）
- **通信减少比**（Communication Reduction, R）
- **客户端计算开销**（Client-side computational overhead, 单位：GFLOPs）
- **总通信量**（Total Communication Volume, 单位：GB）

### 基线方法对比
与以下基于启发式的SL通信压缩方法进行比较：
- **C3-SL**：基于循环卷积的批处理压缩。
- **Rand-Top-K**：随机Top-K稀疏化。
- **ADC**（Attention-based Double Compression）：基于注意力得分的双层压缩。

所有基线方法均被集成到相同的 **SASL**（Scalable Aggregated Split Learning）框架中，以确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **通信效率**：AE-PSL在平均 **10.2×** 的通信量减少下，能够保持与无压缩DFT相当的下游精度。
- **收敛速度**：AE-PSL达到无压缩DFT 99%精度所需的通信量，比最强基线（Rand-Top-K）少 **12.4×**。

### 与基线方法的对比结果
- **准确性优势**：
  - 在所有数据集、客户端数量和压缩级别下，AE-PSL的准确率均**显著优于**所有基线方法。
  - 在相同通信预算下，最强的基线方法（Rand-Top-K）比AE-PSL平均落后 **5.2个百分点**（percent point）。
  - 例如，在CIFAR-100上，当R~10时，AE-PSL的准确率为82.7%，而Rand-Top-K仅为76.2%，差距达6.5个百分点。
- **鲁棒性**：
  - 启发式方法（尤其是C3-SL和ADC）在高压缩比（R~20）下性能急剧下降，甚至崩溃（如FEMNIST上N=25时C3-SL降至6.9%）。
  - AE-PSL在高压缩比下仍能保持稳健，最大准确率下降不超过3.8个百分点。

### 消融实验结果
通过消融实验验证了AE-PSL各组件的必要性（在R=15时）：

| 组件组合 | CIFAR100 准确率 (vs. 无压缩) | 结论 |
| :--- | :--- | :--- |
| 仅随机初始化AE | 1.4% (-81.3) | 直接插入导致灾难性遗忘，性能崩溃。 |
| + GA | 75.2% (-7.5) | GA大幅恢复性能，证明对齐预训练特征流形至关重要。 |
| + GA + FZ (冻结AE) | 79.6% (-3.1) | 冻结AE防止微调期间重构质量漂移，进一步提升性能。 |
| + GA + CSA + FZ (AE-PSL) | 82.2% (-0.5) | CSA使编码器适应客户端特有数据分布，最终接近无压缩性能。 |

此外，还分析了不同AE架构的影响：
- **MLP-based AE** 效果优于 **Conv-based AE**。
- **2-layer-MLP-(d-d-d₂)** 架构在重构保真度、下游准确率和客户端参数增加（仅+1.6%）之间取得了最佳平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **可学习压缩优于启发式方法**：针对预训练模型的特征分布进行对齐的可学习压缩（如AE），在准确率-通信权衡上远超任务无关的启发式压缩。
2. **两阶段对齐至关重要**：直接使用随机初始化的AE会破坏预训练模型的表示能力。提出的 **GA+CSA** 机制成功解决了特征错位问题，使得可学习压缩可以安全地应用于现成的预训练模型。
3. **高效且实用**：AE-PSL在实现巨大通信压缩的同时，引入的客户端计算开销极小，并且加速了分布式收敛过程。
4. **[CLS] token 应保持未压缩**：实验表明，压缩[CLS] token会导致性能严重下降，因此框架选择保持其完整传输以保证语义完整性。

### 局限性
1. **模态限制**：目前的工作主要在视觉Transformer（ViT）上验证，其在其他模态（如文本、多模态）上的泛化性有待检验。
2. **依赖[CLS] token**：框架设计上依赖于不压缩[CLS] token，这限制了其在不使用全局token的架构中的应用。
3. **静态压缩**：当前的压缩策略是固定的，没有根据token的重要性进行自适应压缩。

### 未来工作方向
1. **探索Token自适应压缩**：研究通过可学习的重要性评分来实现token级别的自适应压缩，从而摆脱对未压缩[CLS] token的依赖。
2. **扩展到更多模态和模型架构**：验证AE-PSL在LLM（大语言模型）、语音模型等其他领域的有效性。
3. **开发更通用的对齐协议**：设计不依赖特定模型结构（如[CLS] token）的通用对齐和压缩方案，构建真正模型无关（model-agnostic）的边缘微调框架。

</details>

---

### 14. [SpecLA: Efficient Speculative Decoding for Linear-Attention Models](https://arxiv.org/abs/2607.16673)

**Authors**: Zhibin Wang, Xuying Han, Zhaohua Yang, Fuliang Liu, Xue Li, Rong Gu, Sheng Zhong, Chen Tian  
**Category**: cs.CL  
**Published**: 2026-07-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.16673v1  

#### Abstract
Linear-attention models replace the growing KV cache with recurrent states, but autoregressive decoding still reads, updates, and writes these states one token at a time. Speculative decoding can reduce this cost by verifying several draft tokens in one target pass, yet existing speculative systems ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SPEcLA: Efficient Speculative Decoding for Linear-Attention Models**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统的 **speculative decoding** 技术主要针对基于 Transformer 的模型设计，其核心依赖于 **KV Cache** 的后缀扩展与截断机制。然而，**linear-attention 和 stateful 模型**（如 GDN、Mamba）不再维护 KV Cache，而是通过 **recurrent state** 来压缩上下文状态。这导致传统 speculative decoding 无法直接应用，因为：
- 验证多个候选 token 时，无法高效复用中间 state；
- 被拒绝的候选 token 无法通过“截断”方式回滚；
- 提案（drafting）策略未适配 recurrent 动态特性。

因此，论文旨在解决：**如何为 stateful linear-attention 模型设计高效的 speculative decoding 框架？**

---

### **提出了什么新方法或新思路**
作者提出 **SPEcLA** —— 一个专为 stateful linear-attention 模型优化的 speculative decoding 运行时系统，包含三大核心机制：

#### ✅ **Topology-Aware Verification Kernels**
- 将提交的 draft topology（链状或树状）作为目标 kernel 的调度信号。
- 设计三种验证路径：
  - **State-Resident Serial Verification**：对链式提案，采用 layer-major 执行顺序，保持 state 在 SRAM 中驻留，避免每 token 的 HBM 往返。
  - **Tree-Masked Parallel Verification**：在 GDN 的 Delta-rule 更新中插入 tree mask，实现分支间无泄漏的并行计算。
  - **Chain-Decomposed Hybrid Verification**：将树分解为依赖链（heavy-light decomposition），在链内串行执行，在链间并行验证，兼顾低开销与并行性。

#### ✅ **Accepted-Factor State Management**
- 不存储完整 state 快照，也不重放 token，而是缓存验证过程中生成的轻量级 **update factors**（如 $k_t, v_t, \alpha, \beta$）。
- 后验选择（posterior selection）后仅收集被接受路径上的 factors，并用于重建最终 state。

#### ✅ **Delayed State Update**
- 将 accepted state 的更新延迟到下一轮验证开始前，融合进 verify kernel。
- 避免独立的 state 写回与读取，减少跨 kernel 的 recurrent-state round trip。

#### ✅ **Target-Aligned Drafting**
- 使用 **EAGLE-style drafter**，但训练时使用来自 target 模型的 **recurrent state 特征**，而非 Transformer 隐藏层。
- 引入 **confidence-guided pruning**：基于路径累积 log-probability 剪枝低置信度分支，减少无效验证工作。

---

### **相比现有方法的优势**
| 维度 | 传统方法（Transformer-oriented SD） | SPEcLA |
|------|----------------------------------------|--------|
| **State 处理** | KV Cache 可截断 | Recurrent state 不可截断 → 需 factor 缓冲 |
| **Verification** | 复用 decode 或 prefill kernel | 专用 topology-aware kernel，适应短窗口与树结构 |
| **Acceptance Rollback** | 截断 KV suffix 即可 | 无法 rollback → 需 factor buffering + delayed update |
| **Draft Alignment** | 基于 Transformer hidden states | 基于 target 的 recurrent dynamics，提升 accept rate |

> ✅ **SPEcLA 实现了从“KV-cache manipulation”到“state-factor orchestration”的范式转变。**

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Mixed Prompt Suite**（480 prompts）：涵盖对话、指令遵循、数学推理、代码生成、问答、摘要等。
- **GSM8K**（1,319 prompts）：小学数学应用题。
- **HumanEval**（164 prompts）：代码生成任务。

---

### **实验设置和评估指标**

#### **硬件平台**
- **NVIDIA H100 GPU**

#### **目标模型**
- **GDN-1.3B**：公开可用的纯 Gated DeltaNet 模型（`m-a-p/1.3B-100B-GatedDeltaNet-pure`）
- 参数配置：`h_o=8`, `d_k=d_v=256`, recurrent state 存储为 FP32（每层 2 MiB）

#### **评估指标**
- **Speedup over autoregressive decoding**（主要指标）
- **Average accepted length**：每次 speculative 轮次平均接受的 token 数
- **First-token match**：draft root 与 target greedy token 一致的比例
- **End-to-end generation latency**：总生成耗时（含同步）

#### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Autoregressive** | 单步自回归解码，baseline |
| **Chain** | 链式 speculative decoding，8-token 预算 |
| **FLA-SD** | 直接复用原生 FLA/GDN 路径进行 speculative decoding |
| **SPEcLA-Chain / SPEcLA-Tree** | 本文提出的链式 / 树式方案，最多 16 个 draft 节点，top-k=4 分支 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 方法 | Mixed Suite | GSM8K | HumanEval |
|------|-----------|-------|----------|
| **Autoregressive** | 1.00× | 1.00× | 1.00× |
| **Chain** | 1.26× | 1.54× | 1.03× |
| **FLA-SD** | 1.20× | 1.52× | **<1.00×**（退化） |
| **SPEcLA** | **1.42×** | **1.70×** | **1.06×** |

> 🔥 在 **GSM8K 上达到 1.70× 端到端加速**，是目前最优表现。

---

### **与基线方法的对比结果**
- **SPEcLA 显著优于 Chain 和 FLA-SD**：
  - 在 GSM8K 上比 Chain 快 **10.3%**，说明 topology-aware 验证有效。
  - FLA-SD 在 HumanEval 上性能下降，表明 naive 复用不可靠。
- **Accepted Length 提升明显**：
  - Chain 路由平均接受 3.47 tokens（GSM8K），SPEcLA 提升至 **5.00 tokens**。
- **First-Token Match 更高**：
  - Chain 为 0.68，SPEcLA 达到 **0.81**，说明 target-aligned drafting 更准确。

---

### **消融实验结果**

#### **(1) Tree Verification 效率（vs Root-to-Leaf Replay）**
| Proposal Shape | Serial | Parallel | **Hybrid** |
|----------------|--------|----------|------------|
| top-k=2, depth=12, n=16 | 1.31× | 0.58× | **2.20×** |
| top-k=4, depth=4, n=32 | 4.74× | 3.47× | **7.11×** |

> ✅ **Hybrid Verification 平均比 root-to-leaf replay 快 1.80–7.11×**

#### **(2) State Management 开销**
| 方法 | Commit Latency (8 tokens) | 加速比 |
|------|----------------------------|--------|
| Token Replay | 35.96 ms | 1.00× |
| **Factor Buffering** | **13.10 ms** | **2.74×** |

| 方法 | Verify + Commit Latency (B=1, L=8) |
|------|----------------------------|
| Separate Update | 0.153 ms |
| **Fused (Delayed Update)** | **0.133 ms** → **1.15–1.44× 减少** |

> ✅ 延迟更新显著降低边界开销。

#### **(3) 控制变量研究（不同 acceptance probability p）**
- 当 `p < 0.7` 时，即使长 draft 也难以提速；
- 当 `p ≥ 0.8` 时，所有 draft 长度均有收益；
- 在 oracle 接受下（p=1.0），理论加速可达 **3.71×（1.3B）** 和 **4.61×（外推 9B）**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Stateful linear-attention 模型需要全新的 speculative decoding 架构**，不能简单复用 Transformer 方案。
2. **Topology-aware verification 是关键**：chain-decomposed hybrid kernel 在短窗口下实现了最佳效率。
3. **Factor buffering + delayed update** 成功规避了 dense state 回滚难题，节省高达 4.28× 的恢复延迟。
4. **Target-aligned drafting 提升 accept rate**，confidence pruning 减少了无效计算。
5. **SPEcLA 在真实场景下实现最高 1.70× 端到端加速**，尤其在逻辑推理类任务（GSM8K）上优势显著。

---

### **方法的局限性**
- 当前实现基于单个 GDN 层，尚未扩展到完整多层模型栈。
- 对 drafter 的准确性高度依赖，低 accept rate 下可能不盈利。
- Heavy-light decomposition 的调度复杂度随树宽增加而上升。
- 实验集中在 GDN，对其他 SSM（如 Mamba、GLA）的泛化需进一步验证。

---

### **未来工作方向**
- 将 SPEcLA 扩展至完整的 multi-layer GDN 或 hybrid 架构（如 Qwen3-Next）。
- 探索更智能的动态 draft budget 控制策略。
- 支持 streaming 或 long-context 场景下的持续 speculative decoding。
- 硬件协同设计：为 recurrent state 访问定制 memory hierarchy。

--- 

> 📌 **总结一句话**：  
> **SPEcLA 成功将 speculative decoding 从 KV-Cache 范式迁移到 State-Factor 范式，首次为 linear-attention 模型提供了高效、端到端加速的推理框架。**

</details>

---

### 15. [Taurus: Accelerating Out-of-Core Graph Neural Network Inference on Billion-Scale Graphs](https://arxiv.org/abs/2607.17374)

**Authors**: Pranjal Naman, Yogesh Simmhan  
**Category**: cs.DC  
**Published**: 2026-07-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.17374v1  

#### Abstract
Graph Neural Network (GNN) inference on billion-scale graphs is challenging due to the large memory footprint of features and embeddings and high disk I/O costs in out-of-core settings. Existing distributed GNN systems incur high communication times and infrastructure costs, while disk-based GNN sys...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Taurus: Accelerating Out-of-Core Graph Neural Network Inference on Billion-Scale Graphs》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **大规模图神经网络（GNN）推理中的内存与I/O瓶颈**：在十亿级图（billion-scale graphs）上进行GNN推理时，顶点特征和嵌入（embeddings）通常远超单机RAM容量，导致必须依赖**out-of-core（OOC）** 方案从SSD加载数据。
- 现有方法存在严重缺陷：
  - **分布式系统**（如DistDGL）：通信开销高，基础设施成本昂贵。
  - **基于磁盘的训练系统**（如Ginex、DGL）：为训练优化，不适用于全图推理，存在大量**重复读取**（repeated reads）和**随机访问**（random access）。
  - **层间gather-based方法**（如DGI）：虽避免冗余计算，但仍因**destination-centric gather**模式导致严重的**读放大**（read amplification）和I/O瓶颈。

### **提出的新方法与思路**
Taurus提出了一种全新的**broadcast-based执行模型**，将传统“拉取”模式转变为“推送”模式：

- **核心思想**：将GNN推理重构为**source-centric广播**（broadcast），而非传统的**destination-centric gather**。
  - 每个源顶点将其特征/嵌入**顺序扫描并广播**给所有邻居，避免重复随机读取。
  - 利用SSD的**顺序读写优势**，显著降低I/O开销。

- **系统架构创新**：
  1. **三层存储层次**（tiered GPU-RAM-SSD hierarchy）：
     - **GPU Store**：驻留高入度顶点的聚合状态，减少主机内存压力。
     - **Hot Store**（RAM）：管理活跃顶点的部分聚合状态。
     - **Cold Store**（SSD）：溢出状态持久化，支持重载。
  2. **拓扑感知重排序**（Topology-aware reordering）：
     - 提出新的重排序算法，最小化顶点的**激活跨度**（span），即从首次接收到最后一条消息的时间间隔，从而减少部分状态驻留时间和SSD换入换出。
  3. **待处理消息驱逐策略**（Pending-message eviction policy）：
     - 驱逐**待接收消息最少**的顶点，优先让即将完成的顶点保留在内存中，极大减少“驱逐-重载”循环（eviction-reload cycles）。
  4. **非缓冲I/O与GPU直写**：
     - 使用`O_DIRECT`绕过OS page cache，减少缓存污染。
     - 利用**GPUDirect Storage (GDS)** 或 `cuFile` 将变换后的嵌入直接从GPU写入SSD，避免主机内存中转。

### **相比现有方法的优势**
- 显著降低**读放大**（read amplification）和**I/O总量**。
- 在单机上实现高效推理，无需昂贵的分布式部署。
- 支持**精确全邻域聚合**（exact full-neighborhood）和**采样推理**（fanout-sampled），兼顾精度与效率。
- 通过广播+流水线设计，实现高吞吐、低延迟的端到端推理。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 顶点数 | 边数 | 特征大小 |
|--------|--------|------|----------|
| **Papers (PA)** | 111M | 1.7B | 54 GiB |
| **Friendster (FS)** | 65M | 3.6B | 251 GiB |
| **MAG-Cites (MA)** | 121M | 1.4B | 350 GiB |
| **IGB-Large (IL)** | 100M | 1.2B | 382 GiB |
| **IGB-Full (IF)** | 269M | 4B | **514 GiB (FP16)** |

> 所有数据集均超过典型服务器RAM容量，需OOC处理。

### **实验设置**
- **硬件平台**：
  - CPU: AMD Ryzen 9 9900X (12核)
  - RAM: 128 GiB
  - GPU: NVIDIA RTX 5090 (32 GiB VRAM)
  - SSD: 2 TiB Samsung 990 PRO
  - OS: Ubuntu 24.04.3 LTS
- **模型**：2层GCN、SAGE、GIN、GAT（单头），隐藏维度128。
- **任务**：全图顶点分类（full-graph vertex classification）。
- **评估指标**：
  - **总推理时间**（Total inference time）
  - **I/O量**（读/写字节数）
  - **冷存储重载次数**（cold-store reloads）
  - **速度提升**（speedup）

### **基线方法对比**
| 基线 | 类型 | 描述 |
|------|------|------|
| **DGI** | Layer-wise | 层间推理框架，使用动态批处理和RCMK重排序，最强layer-wise基线 |
| **Ginex (GX)** | Vertex-wise | 为OOC训练设计，使用最优缓存策略（Belady） |
| **DGL GraphBolt (DG)** | Vertex-wise | DGL的OOC推理接口，支持磁盘数据集 |

> 所有运行限制在4小时内，超时则线性外推。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 在最大数据集 **IGB-Full (514 GiB)** 上：
  - Taurus 完成**2层GCN推理仅需 < 30分钟**。
  - 而最强基线 **DGI** 推测耗时约 **12.5小时**，**慢25倍以上**。
- 在其他OOC数据集上：
  - 相比 **DGI**，Taurus 平均快 **7–25×**。
  - 相比 **vertex-wise基线**（Ginex、DGL），快 **40–140×**。

### **与基线方法的对比结果**
| 对比项 | 结果 |
|--------|------|
| **vs DGI** | Taurus 减少 **~10× I/O量**，显著降低读放大 |
| **vs Ginex/DGL** | 读取数据量高达 **50–100×**，因反复构建缓存和批量加载 |
| **I/O效率** | Taurus 实现接近顺序带宽的SSD读写，峰值写带宽达 **4.3 GiB/s** |

### **消融实验结果**
#### **(1) 重排序（Reordering）的影响**
- 使用Taurus提出的**拓扑感知重排序**：
  - 在 **Friendster** 上，推理时间减少 **25–38%**。
  - 在 **IGB-Large** 上，减少 **51–64%**。
  - 冷存储重载次数下降 **126×**（IL）和 **4.3×**（FS）。
- 仅用RCMK重排序无法达到同等效果，证明Taurus重排序更优。

#### **(2) 驱逐策略（Eviction Policy）的影响**
- **Taurus策略**（驱逐待消息最少者） vs **Random/LRU/FIFO**：
  - 推理时间减少 **19–43%**。
  - 冷存储重载减少 **1.4–6×**。
  - LRU表现最差，因其频繁驱逐远未完成的顶点。

#### **(3) GPU Store的影响**
- 即使只分配 **16 GiB GPU内存**：
  - 覆盖约 **6.4%顶点**，却处理了 **~51%的目标消息**（因幂律分布）。
  - 推理时间减少 **27% (FS)** 和 **36% (IL)**。
  - I/O时间下降 **57–79%**。

#### **(4) Hot Store容量影响**
- Taurus对内存需求**极低**：
  - 在 **IGB-Large** 上，仅需 **50 GiB Hot Store** 即可几乎消除重载。
  - 而传统方法需近 **100 GiB** 才能达到相近性能。
  - 证明Taurus有效解耦了性能与内存规模。

---

## **4. 关键结论和发现**

### **主要发现**
1. **广播优于聚集**：将GNN推理从**gather-based**转向**broadcast-based**是突破OOC瓶颈的关键。
2. **顺序I/O至关重要**：利用SSD顺序读写能力，结合拓扑重排序，可极大缓解I/O压力。
3. **智能内存管理决定性能**：
   - 拓扑感知重排序 + 待消息驱逐策略 + GPU驻留高入度顶点，共同抑制了I/O抖动（thrashing）。
4. **单机可胜任十亿级推理**：Taurus证明，在合理软硬件协同下，单台工作站即可高效处理此前需分布式集群的任务。

### **方法的局限性**
- **静态图假设**：当前Taurus仅支持静态图快照，不支持动态图上的增量更新。
- **额外计算代价**：对SAGE和GAT等复杂GNN，需多轮扫描以保持语义正确，带来一定开销。
- **预处理开销**：拓扑重排序是一次性预处理，耗时 **9–30分钟**，虽可复用，但仍需投入。

### **未来工作方向**
1. **支持动态图**：扩展广播模型以支持**连续、增量更新**（continuous/incremental updates）。
2. **多GPU扩展**：探索**multi-GPU scaling**，加速计算密集阶段，支持更大嵌入维度。
3. **自动化参数调优**：研究自适应调整chunk size、buffer size等参数的机制。
4. **支持更多GNN变体**：如Transformer-based GNNs、异构图模型等。

---

> **总结**：Taurus通过**broadcast-based执行模型**、**拓扑感知调度**和**GPU增强的三级存储架构**，实现了十亿级图上GNN推理的革命性加速，为低成本、高效率的大规模图推理提供了全新范式。

</details>

---

### 16. [HyMCache: A KV Cache Framework for Multi-Turn LLM Serving with CXL-Hybrid Memory](https://arxiv.org/abs/2607.18141)

**Authors**: Hakbeom Jang, Inho Song, Sam H. Noh, Jongryool Kim  
**Category**: cs.DC  
**Published**: 2026-07-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.18141v1  

#### Abstract
Long-context, multi-turn, and agentic LLM workloads increasingly reuse previously processed context, making KV-cache reuse essential for reducing redundant computation. However, this reuse shifts the bottleneck to the memory tier that stores and serves reusable KV states at cluster scale. GPU HBM an...

---

### 17. [EA-RMENet -- Path Loss Prediction in Urban Environments using Deep Learning](https://arxiv.org/abs/2607.16449)

**Authors**: Jonathan O'Shea (DCU School of Electronic Engineering), Conor Brennan (DCU School of Electronic Engineering)  
**Category**: cs.LG  
**Published**: 2026-07-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.16449v1  

#### Abstract
Accurate path loss prediction is a critical component of wireless network planning. Current path loss prediction methods typically struggle to balance the trade-off between accuracy and computational efficiency. This paper proposes the Efficient Attention Radio Map Estimation Network (EA-RMENet) whi...

---

### 18. [Rater State Bias in RLHF Preference Data: An Audit Framework](https://arxiv.org/abs/2607.16195)

**Authors**: Elena Kopteva, Vitaliy Hlynianyi-Zhuk  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.16195v1  

#### Abstract
We identify a structured confound in Reinforcement Learning from Human Feedback (RLHF). Pairwise preference labels are intended to reflect the compared outputs, but they may also reflect the rater's state during annotation. Under sustained stressful or distressing conditions, raters' preferences may...

---

### 19. [Democratizing AI with Small Language Models: Structured Benchmarking and Parameter-Efficient Fine-Tuning for Local Deployment](https://arxiv.org/abs/2607.16202)

**Authors**: Daniel Cersosimo  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.16202v1  

#### Abstract
AI democratization is not primarily a question of matching frontier-scale generality; it is a question of whether capable models can be selected, audited, and specialized under hardware and governance constraints that ordinary institutions can actually satisfy. This paper studies that problem throug...

---

### 20. [Masked Diffusion Language Models are Strong and Steerable Text-Based World Models for Agentic RL](https://arxiv.org/abs/2607.16204)

**Authors**: Darshan Deshpande  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.16204v1  

#### Abstract
Recent growth in reinforcement learning (RL) has surfaced a need for diverse, specialized training environments. Hand-curated environments with fixed task and reward difficulties become ineffective signals as model performance improves, and sparse rewards over long horizons induce mode collapse on s...

---

### 21. [EdgeCoInfer: Hierarchical Collaborative Inference for On-Device Multimodal Large Models](https://arxiv.org/abs/2607.17143)

**Authors**: Lin Tan, David K. Y. Yau, Songtao Guo  
**Category**: cs.DC  
**Published**: 2026-07-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.17143v1  

#### Abstract
Modern mobile applications predominantly execute concurrent Multimodal Large Language Models (MLLMs) to provide ubiquitous intelligence. However, satisfying this demand within edge environments faces significant challenges due to multi-task concurrency and strictly coupled hard constraints. To addre...

---

### 22. [Shapley Context Pruning: A Cooperative Game Perspective for Context Reranking and Pruning](https://arxiv.org/abs/2607.16209)

**Authors**: Yanqiao Chen, Dongsheng Hou, Yuhan Rui, Zhen Cao, Yepang Liu  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.16209v1  

#### Abstract
Context reranking and pruning have become essential for improving the efficiency of modern Retrieval-Augmented Generation (RAG) systems, yet an interpretable and unified framework remains underexplored. Previous work has primarily emphasized lexical retrieval, cross-encoder architectures, model dist...

---

### 23. [When LLMs Over-Answer: Measuring and Mitigating Quality Issues in LLM-Based Hardware Description Language Question Answering](https://arxiv.org/abs/2607.17063)

**Authors**: Ziteng Hu, Jiachi Chen, Wenhao Lv, Huan Zhang, Yingjie Xia  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.17063v1  

#### Abstract
The rapid advancement of large language models (LLMs) has led practitioners to increasingly rely on them for answering questions about hardware description languages (HDLs). Because HDL is ultimately synthesized into physical hardware, an imprecise or redundant answer can propagate into timing viola...

---

### 24. [Bridging the Information Gap: Semantic Densification and Hindsight Distillation for Cold-Start Prediction](https://arxiv.org/abs/2607.17070)

**Authors**: Hao Duong Le, Yifei Gao, Huan Li, Lun Jiang, Chen Bai, Ke Xing, Chen Zhang  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.17070v1  

#### Abstract
New-user cold-start is a critical bottleneck for e-commerce platforms: predicting user lifetime value (LTV) and conversion rate (CVR) for users with sparse interaction history. Two prior directions -- LLM-based semantic augmentation and learning using privileged information (LUPI) -- each face a key...

---

### 25. [AdaHome: An Adaptive Smart Home Assistant using Local Small Language Models](https://arxiv.org/abs/2607.18034)

**Authors**: Eu Jin Lim, Zhaoxing Li, Sebastian Stein  
**Category**: cs.AI  
**Published**: 2026-07-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.18034v1  

#### Abstract
Smart home assistants interpret a wide range of user commands, from explicit device control to underspecified and preference dependent requests. While recent systems based on Large Language Models (LLMs) improve this capability, they often rely on heavyweight reasoning pipelines and cloud-based depl...

---

### 26. [Reinforcement Learning-Guided NSGA-II Enhanced with Gray Relational Coefficient for Multi-Objective Optimization: Application to NASDAQ Portfolio Optimization](https://arxiv.org/abs/2607.16194)

**Authors**: Zhiyuan Wang, Qinxu Ding, Ding Ding, Siying Zhu, Jing Ren, Yue Wang, Chong Hui Tan  
**Category**: cs.LG  
**Published**: 2026-07-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.16194v1  

#### Abstract
In modern financial markets, decision-makers increasingly rely on quantitative methods to navigate complex trade-offs among multiple, often conflicting objectives. This paper addresses constrained multi-objective optimization (MOO) with an application to portfolio optimization for minimizing risk an...

---

### 27. [A Framework for Early Sepsis Prediction via Self-Supervised (JEPA) and Federated Representation Learning](https://arxiv.org/abs/2607.16681)

**Authors**: Umair bin Mansoor, Munaf Rashid, Roomi Naqvi  
**Category**: cs.LG  
**Published**: 2026-07-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.16681v1  

#### Abstract
Early sepsis prediction from electronic health records is challenged by irregular sampling, high missingness, and class imbalance. We systematically compare four modeling paradigms -- self-supervised Joint Embedding Predictive Architecture (JEPA) via masked latent prediction, self-supervised VICReg ...

---

### 28. [MultiLoReFT: Decoupling Shared and Modality-Specific Subspaces in Multimodal Learning via Low-Rank Representation Fine-Tuning](https://arxiv.org/abs/2607.16789)

**Authors**: Sana Tonekaboni, Viktoria Schuster, Caroline Uhler  
**Category**: cs.LG  
**Published**: 2026-07-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.16789v1  

#### Abstract
Real-world perception and decision making are inherently multimodal, integrating complementary signals across modalities. However, training multimodal models faces two main obstacles. First, collecting large-scale, well-aligned paired multimodal datasets is often impractical, making end-to-end multi...

---

### 29. [Rethinking the Suitability of Reinforcement Learning Algorithms Under Practical Transfer Constraints](https://arxiv.org/abs/2607.17326)

**Authors**: Hany Hamed, Abhishek Naik, Colin Bellinger, A. Rupam Mahmood  
**Category**: cs.LG  
**Published**: 2026-07-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.17326v1  

#### Abstract
Transfer-oriented reinforcement learning requires evaluating algorithms along dimensions that go beyond standard sample efficiency. We focus on two dimensions: practical efficiency, which asks whether conclusions about algorithm suitability change under wall-clock rather than interaction-based budge...

---

### 30. [AGG: Jacobian-Aggregated Group Gradient for Efficient GRPO Training of Diffusion Models](https://arxiv.org/abs/2607.17572)

**Authors**: Ruiyi Ding, Jie Li, He Kang, Ziyan Liu, Chengru Song, Yuan chen  
**Category**: cs.LG  
**Published**: 2026-07-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.17572v1  

#### Abstract
Group Relative Policy Optimization (GRPO) is a powerful reinforcement learning algorithm for aligning generative models with human preferences. While successful in large language models~\cite{shao2024deepseekmathpushinglimitsmathematical}, its extension to diffusion and flow matching models introduc...

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

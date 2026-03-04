# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-04 06:13:49 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs](https://arxiv.org/abs/2603.02731)

**Authors**: Wuyue Zhang, Chongdong Huang, Chunbo You, Cheng Gu, Fengjuan Wang, Mou Sun  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2603.02731v1  

#### Abstract
Training large-scale Mixture-of-Experts (MoE) models is bottlenecked by activation memory and expert-parallel communication, yet FP4 training remains impractical on Hopper-class GPUs without native MXFP4 or NVFP4 support. In this work, we present a training recipe that enables MXFP4 efficiency for M...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs**

---

## 1. **论文的主要贡献和创新点**

### **解决的问题**
当前大规模 **Mixture-of-Experts (MoE)** 模型的训练受限于两大瓶颈：
- **激活内存（activation memory）占用高**，限制了批大小和模型规模；
- **expert-parallel 通信开销大**，尤其在 All-to-All 通信阶段。

尽管 **FP4** 量化格式能显著降低内存和带宽需求，但 **Hopper 架构的 GPU 缺乏对 FP4 的原生支持**（如无 FP4 Tensor Core），导致直接部署 FP4 不现实。此外，在已有 BF16/FP8 混合精度流程中引入 FP4 会带来昂贵的精度转换开销（如 FP8 → BF16 → FP4），影响效率与收敛性。

### **提出的新方法与创新点**
本文提出了一套**软件模拟的 MXFP4 训练框架**，在无硬件支持的情况下实现 FP4 级别的效率提升，核心创新如下：

- ✅ **FP4 激活压缩与通信优化**  
  将前向传播中的 **activations 和 All-to-All 通信数据压缩为 MXFP4 格式**，减少 50% 以上的通信体积和激活内存占用。

- ✅ **直接 FP8-to-FP4 转换算法（Direct Bitwise Conversion）**  
  避免传统路径 `FP8 → BF16 → FP4` 的精度往返与性能损耗，设计了从 FP8 到 FP4 的**位级直接转换算法**，结合层次化 scale 对齐策略，消除中间 BF16 步骤。

- ✅ **解耦计算与存储精度（Decoupled Precision Strategy）**  
  - **计算路径保持 FP8**：利用 Hopper 原生 FP8 Tensor Core 加速 GEMM 运算，保障数值稳定性；
  - **存储与通信使用 FP4**：仅在内存和通信中使用 MXFP4，实现“**compute in FP8, store in FP4**”的设计理念。

- ✅ **融合内核与布局感知转换（Fused CUDA Kernels）**  
  开发了多个高性能 CUDA kernel：
  - `BF16ToFP4Row`：行式 FP4 量化；
  - `FP4RowToFP8Row` / `FP4RowToFP8Col`：去量化并支持转置输出，用于 Wgrad 计算；
  - 支持 **ragged MoE tensors** 和共享内存无 bank conflict 访问。

- ✅ **前向激进、后向保守的非对称设计（Asymmetric Precision Flow）**  
  - 前向传播广泛使用 FP4 压缩；
  - 反向传播恢复为标准 FP8 流程，避免梯度去量化的额外开销，确保端到端性能最优。

---

## 2. **核心实验方法和设置**

### **使用的模型与架构**
- **模型类型**：671B 参数规模的 **MoE 模型**，基于 DeepSeek-V3 公开配置；
- 包含 **Multi-Head Latent Attention (MLA)** 结构；
- MoE 层采用稀疏路由机制，每 token 激活部分专家。

### **硬件平台**
- **GPU**：256 张 **NVIDIA Hopper GPU**（80GB HBM3）；
- **互联**：节点内通过 NVSwitch，跨节点使用 InfiniBand；
- 所有实验均在此集群上运行。

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **BF16** | 标准 bfloat16 混合精度训练（Megatron-LM 实现） |
| **FP8** | Transformer Engine 提供的 blockwise FP8 训练方案，针对 Hopper 优化 |

### **评估指标**
- **训练吞吐量（Throughput）**：Tokens per GPU per Second (**TGS**)；
- **峰值激活内存占用（Peak Activation Memory）**；
- **是否发生 OOM（Out-of-Memory）错误**；
- **收敛性（Convergence）**：语言建模损失曲线对比（LM Loss）；
- **消融研究**：量化内核效率、不同重组计算策略的影响。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（671B MoE 模型）**

| 重组计算范围（Recomputation Scope） | 方法 | TGS | 内存占用 |
|----------------------------------|--------|-------|----------|
| Attention + LN + MoE Experts | BF16 | 1122 | 68.29% |
|                                  | FP8   | 1157 | 74.20% |
|                                  | **Ours (MXFP4)** | **1156** | **59.40%** ✅ |
| Attention + LN + MLP           | BF16 | OOM | — |
|                                  | FP8   | OOM | — |
|                                  | **Ours (MXFP4)** | **1248** | **60.62%** |
| **MLA Up Projection Only**     | BF16 | OOM | — |
|                                  | FP8   | OOM | — |
|                                  | **Ours (MXFP4)** | **1302** ✅ | **70.11%** |

#### **核心成果总结**
- 在相同重组计算策略下，相比 FP8 基线：
  - **峰值激活内存降低 14.8%（11.8 GB）**；
  - 吞吐量持平（~1156 vs 1157 TGS）；
- 得益于更低的内存压力，可**大幅减少重组计算范围**；
- 最优配置下（仅重计算 MLA 上投影层）：
  - **训练吞吐提升 12.5%（从 1157 → 1302 TGS）**；
  - BF16 和 FP8 均因 OOM 无法运行。

### **236B 模型上的控制变量实验**

| 方法 | 峰值内存下降（vs FP8） | 下降幅度（vs BF16） |
|------|------------------------|--------------------|
| MXFP4（仅 MLP & Shared Experts） | 6.9% | 11% |
| MXFP4（扩展至全部 MoE）         | 7.2% | 11% |

✅ 表明 FP4 压缩在不同规模下具有良好的可扩展性和一致性收益。

### **消融实验结果**

#### **内核融合效率（Figure 3）**
- 相比未融合的独立操作序列，**融合 dequant + layout transform 的 kernel 实现 1.6x–1.9x 加速**；
- 主要得益于减少了中间全局内存读写。

#### **总量化开销分析（Figure 4）**
- **标准线性层（如 Attention 投影）**：由于额外量化开销，相对延迟增加（0.33x ~ 0.36x 性能）；
- **MoE 专家层（Group Linear）**：定制 `msplit` kernel 实现 **1.43x–1.53x 加速**；
- 因 MoE 占据 671B 模型大部分 FLOPs，因此整体仍获得净收益。

#### **收敛性验证（Figure 5）**
- 在 16B 参数模型上训练至 160B tokens：
  - **MXFP4 的 loss 曲线与 BF16 完全对齐**，无发散或震荡；
  - 相对于 BF16，FP8 和 MXFP4 的相对损失偏差分别为 +0.29% 和 +0.61%，仍在可接受范围内；
- 表明该方法在数值上稳定，适合大规模训练。

---

## 4. **关键结论和发现**

### **主要发现**
1. 🔹 **无需硬件支持也能实现 FP4 效率**：通过软硬协同设计（software-hardware co-design），可在缺乏 FP4 Tensor Core 的 Hopper GPU 上实现接近原生 FP4 的内存与通信效率。
2. 🔹 **解耦计算与存储是关键**：“Compute in FP8, Store/Communicate in FP4”的混合精度范式有效平衡了性能、内存与收敛性。
3. 🔹 **非对称精度流更高效**：前向用 FP4 压缩，反向回退 FP8，既能节省资源又避免反向额外开销。
4. 🔹 **系统级优化决定成败**：单纯量化不够，必须配合 **direct conversion、fused kernel、layout-aware transpose、scale alignment** 等底层优化才能真正释放潜力。

### **方法的局限性**
- ❗ 当前仅适用于 **forward pass 中的 activation 压缩**，尚未推广到权重或完全端到端 FP4 训练；
- ❗ 依赖高度定制化的 CUDA kernel，移植性受限，需深度集成到 Megatron-LM、Transformer Engine 和 DeepEP；
- ❗ 对调度不规则的 ragged tensor 支持虽已实现，但在极端负载下可能影响通信均衡性。

### **未来工作方向**
- 🚀 探索 **NVFP4 或其他 4-bit 格式** 在类似框架下的适配；
- 🚀 将 FP4 扩展至 **weight gradient 存储与通信**，进一步压缩反向内存；
- 🚀 结合 **checkpoint compression** 技术，实现全生命周期低比特训练；
- 🚀 推动标准化通信库（如 NCCL、SHARP）支持 sub-byte packing，降低部署门槛。

---

> ✅ **代码开源**：作者已在 GitHub 开源完整实现，涵盖自定义 kernel、框架补丁与使用指南（见 Appendix A）。  
> 🔗 项目地址：`github`（原文链接略）  

--- 

📌 **一句话总结**：  
本论文首次实现了在无原生支持的 Hopper GPU 上进行高效、稳定的 **软件模拟 MXFP4 训练**，通过精细的混合精度设计与系统级优化，在 671B MoE 模型上实现了 **14.8% 激活内存下降、12.5% 吞吐提升且收敛一致**，为下一代超大规模模型训练提供了实用化低比特解决方案。

</details>

---

### 2. [CUCo: An Agentic Framework for Compute and Communication Co-design](https://arxiv.org/abs/2603.02376)

**Authors**: Bodun Hu, Yoga Sri Varshan V, Saurabh Agarwal, Aditya Akella  
**Category**: cs.DC  
**Published**: 2026-03-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.02376v1  

#### Abstract
Custom CUDA kernel development is essential for maximizing GPU utilization in large-scale distributed LLM training and inference, yet manually writing kernels that jointly leverage both computation and communication remains a labor-intensive and error-prone process. Prior work on kernel optimization...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《CUCo: An Agentic Framework for Compute and Communication Co-design》总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在大规模分布式 LLM（Large Language Model）训练与推理中，GPU 利用率的提升依赖于对 **computation** 和 **communication** 的联合优化。然而，传统上这两者由不同的库（如 cuBLAS/cuDNN 用于计算，NCCL 用于通信）独立管理，并通过 CPU 主机进行协调调度。

这种分离设计导致以下问题：
- **同步开销大**：CPU 需等待 kernel 完成后才能启动通信，造成流水线气泡。
- **重叠机会受限**：难以实现细粒度的 compute-communication overlap。
- **手动开发复杂且易错**：编写融合了计算与通信逻辑的高性能 CUDA kernel 极其困难，尤其涉及多 GPU 协同时。

尽管近年来出现了支持 **device-initiated communication** 的接口（如 NCCL Device API、NVSHMEM），允许 GPU kernel 直接发起通信操作，但如何有效利用这些能力仍缺乏系统化工具。

---

### **提出了什么新方法或新思路**

本文提出 **CUCo** —— 一个无需预训练的 **agentic framework**，用于自动生成高性能的、融合 computation 与 communication 的 CUDA kernels。

#### 核心创新点包括：

1. **Structured Design Space Specification**
   - 定义了一个由五个维度构成的优化空间 $ C = B \times P \times S \times I \times G $：
     - `B`: Backend（GIN 或 LSA）
     - `P`: Placement（通信放置策略）
     - `S`: Sync Scope（同步范围）
     - `I`: Issuer Granularity（发起粒度）
     - `G`: Granularity（传输块大小）
   - 引入 **optimization directive** 机制，要求 agent 显式声明其设计选择，使搜索过程可解释、可追溯。

2. **Two-Phase Agent Pipeline**
   - **Fast-Path Agent**：优先保证正确性，将 host-driven NCCL 代码转换为保守但功能正确的 device-initiated 实现，作为进化起点。
   - **Slow-Path Agent**：以 fast-path 输出为种子，采用基于 LLM 的 evolutionary search 进行性能探索与优化，逐步收敛到高性能方案。

3. **Agentic Evolutionary Search with Feedback Loop**
   - 使用 island-based 多岛进化框架，避免早熟收敛。
   - 引入 **LLM mutation operator** 替代随机变异，生成语义合理的代码修改。
   - 设计 **cascade evaluation** 流程（编译 → 正确性验证 → 性能测试），控制评估成本。
   - 构建共享 **candidate database** 与 **meta-summarizer**，实现跨代知识积累与推荐。

---

### **相比现有方法的优势**

| 对比维度 | 现有方法（如 CUDAForge, STARK） | CUCo |
|--------|-------------------------------|------|
| **Multi-GPU 支持** | 多数仅限单卡优化 | 原生支持 multi-GPU 与 collective communication |
| **Communication 融合** | 不支持或弱支持 | 显式建模并优化 compute-comm 融合 |
| **是否需要训练** | 部分需 fine-tune LLM | 完全 training-free，零样本生成 |
| **分解与引导** | 直接生成代码，易出错 | 先 fast-path 分解再 slow-path 优化，降低失败率 |
| **硬件感知** | 缺乏动态上下文注入 | 动态提取 GPU 架构、网络拓扑等 context |

> ✅ CUCo 是首个专为 **compute-communication co-design** 设计的 agentic 框架，在正确性、效率与自动化之间取得平衡。

---

## 2. 核心实验方法和设置

### **使用的代表性 workload（无标准 benchmark）**

由于缺乏涵盖 compute-communication 融合行为的标准基准，作者构建了四个典型多 GPU 场景：

| Workload | 描述 |
|--------|------|
| **Flash Attention + Context Parallelism** | 在长序列 attention 中使用 ring topology 传递 KV 缓存，4-GPU 内部节点（NVLink） |
| **DeepSeek-V3 MoE Dispatch & Combine** | MoE 模型中的稀疏路由 AlltoAll 通信，2-GPU 跨节点（RoCE） |
| **KV-Cache Transfer** | Prefill 阶段向 Decode 阶段发送 K/V 投影，2-GPU NVLink |
| **GEMM + AllGather** | 局部矩阵乘后聚合结果，覆盖 intra-node（LSA）与 inter-node（GIN）场景 |

---

### **实验设置**

- **硬件环境**：
  - 双服务器，每台配 4× NVIDIA A100 (80GB)，通过 RoCE 互联
  - 内部连接使用 NVLink，外部使用 RoCE v2
- **软件栈**：
  - CUDA 13.1, NCCL 2.28.9, Ubuntu 22.04
  - LLM 后端：Anthropic Claude Sonnet 4.5
- **评估指标**：
  - **End-to-end latency**（中位数，3 次稳定运行平均）
  - 加速比（Speedup over baseline）
  - 编译成功率、数值正确性
  - 消融分析中的模块贡献度（如 pipeline bubble 消除时间）

---

### **基线方法对比**

- **Host-driven Baseline**：
  - 使用标准 NCCL host API（`ncclSend/Recv`, `ncclAlltoAll`）
  - 通过 CUDA stream 实现粗粒度 overlap
- **CUCo Evolved Kernel**：
  - 自动生成的 fused kernel，直接在 device 端调用 GIN.put / LSA.store / waitSignal 等原语
  - 支持 per-tile pipelining、split put/wait、warp-specialized comm 等高级模式

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| Workload | CUCo 提升幅度 | 最高加速比 |
|--------|-------------|----------|
| Flash Attention (SEQ=4096, HD=32) | **11.3% latency reduction** | 1.13× |
| DeepSeek MoE (imbalance ratio 2:1–5:1) | **5.3% – 14.4%** | ~1.17× |
| KV-Cache Transfer | **9.2% – 15.6%** | ~1.18× |
| GEMM + AllGather | **26.2%**（GIN+LSA 融合） | **1.57×** |

> 🔥 在 GEMM+AllGather 上达到最高 **1.57×** 的 end-to-end 加速。

---

### **与基线方法的对比结果**

- 所有 workload 均显著优于 host-driven baseline。
- 性能增益主要来自三个方面（以 Flash Attention 为例）：

| 优化项 | 节省时间 | 机制说明 |
|-------|---------|----------|
| Pipeline bubble 消除 | 37.7ms | Host 必须等所有 tile 传完才启动下一阶段；CUCo 支持 per-tile pipelining |
| Host API 开销移除 | 37.6ms | 替换 768 个 `ncclSend/Recv` 组为 device-side `gin.put/flush` |
| Per-tile 内部流水 | 63.7ms | Compute block 可 polling tile_ready counter 并提前处理部分数据 |
| **总计** | **139.0ms** | 占原始耗时 11.3% |

> 💡 这些优化只有在 **fused kernel + cooperative grid** 下才能实现，传统 host-driven 模式无法突破 SM 饱和带来的调度瓶颈。

---

### **消融实验结果**

#### （1）Fast-Path Agent 的必要性（图7 vs 图8）

| 配置 | 第一次成功生成正确 kernel 的 generation | 最终得分峰值 |
|-----|--------------------------|------------|
| Fast-Path + Slow-Path | 第 2 代 | **83.95** |
| 仅 Slow-Path（无 fast-path 初始化） | 第 5 代 | **81.81**（↓2.5%） |

> ❗ 无 fast-path 导致前 25% 的进化预算浪费在修复编译错误上，样本效率大幅下降。

#### （2）两阶段进化策略（Explore → Exploit）（图9 vs 图10）

| 配置 | 达到近优解所需 generation | 最终得分 |
|-----|----------------------|--------|
| Two-phase（前40%探索） | **3 代** | **83.95** |
| Single-phase（仅 exploit） | 10 代以上 | **80.36**（↓4.3%） |

> ✅ 探索阶段帮助发现 barrier-free combine、split put/wait 等新颖结构，exploit 阶段进一步精细化。

---

## 4. 关键结论和发现

### **主要发现**

1. **Compute-Communication Fusion 具备巨大潜力**  
   将 communication 移入 kernel 内部，结合 device-initiated API（GIN/LSA），可解锁传统 host-driven 模型无法实现的细粒度 overlap 与 pipelining。

2. **Agentic Search 是解决 co-design 复杂性的有效路径**  
   手动设计此类 kernel 成本极高，而 CUCo 通过 structured reasoning + evolutionary LLM search 自动探索设计空间，实现了接近最优的手工调优效果。

3. **Decomposition 是关键：Fast-Path 提供可靠起点**  
   直接让 LLM 从零开始生成 multi-GPU fused kernel 极易失败；先通过 fast-path 生成保守正确的 baseline，再由 slow-path 优化，显著提升成功率与收敛速度。

4. **硬件感知 context 注入至关重要**  
   LLM 对 NCCL device API 几乎没有先验知识，必须注入 backend-specific 文档、header、reference code 和 hardware context 才能生成合法代码。

---

### **方法的局限性**

1. **依赖高质量 reference implementation**  
   当前方法假设存在一个功能正确的 host-driven baseline 供 static analyzer 分析依赖图。

2. **LLM 成本较高**  
   尽管无需 fine-tuning，但在大规模 evolutionary search 中频繁调用 LLM（Claude Sonnet）会产生可观的推理成本。

3. **目前仅支持特定通信模式**  
   如 AlltoAll、AllGather、Put/Wait 等，尚未扩展至更复杂的 collective（如 ReduceScatter + AllReduce chain）。

4. **对 LLM 输出稳定性有一定依赖**  
   若 LLM 多次输出语法错误或不一致的 patch，可能影响进化进程。

---

### **未来工作方向**

1. **扩展至更多通信原语与 topology**  
   支持 hierarchical collectives、tree-based broadcast、pipeline parallelism 中的 forward/backward overlap。

2. **引入 cost model 预筛选 candidate**  
   在调用 LLM mutation 前使用轻量级 cost model 预测性能趋势，减少无效编译尝试。

3. **支持增量式 co-design for fine-tuning**  
   针对不同 workload（如不同 seq length、expert imbalance）自动调整 fusion 策略。

4. **集成进主流 ML 编译器栈（如 Triton, PyTorch 2）**  
   将 CUCo 作为 compiler pass，实现 automatic kernel generation within end-to-end stack。

---

> 📢 **总体评价**：CUCo 展示了 **agent-driven co-design** 在打破 compute 与 communication 抽象壁垒方面的强大潜力，是迈向 fully autonomous GPU kernel optimization 的重要一步。随着 device-initiated communication 成为分布式 ML 的主流范式，此类自动化框架将成为不可或缺的基础设施。

</details>

---

### 3. [Robust Heterogeneous Analog-Digital Computing for Mixture-of-Experts Models with Theoretical Generalization Guarantees](https://arxiv.org/abs/2603.02633)

**Authors**: Mohammed Nowaz Rabbani Chowdhury, Hsinyu Tsai, Geoffrey W. Burr, Kaoutar El Maghraoui, Liu Liu, Meng Wang  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.02633v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) models enable efficient scalability by activating only a small sub-set of experts per input, yet their massive parameter counts lead to substantial memory and energy inefficiency during inference. Analog in-memory computing (AIMC) offers a promising solution by elimin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Robust Heterogeneous Analog-Digital Computing for Mixture-of-Experts Models with Theoretical Generalization Guarantees**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代 **Sparse Mixture-of-Experts (MoE)** 模型虽然通过稀疏激活实现了高效扩展，但由于其庞大的参数量，在推理过程中面临严重的 **内存占用** 和 **能量效率低下** 问题。  
**Analog In-Memory Computing (AIMC)** 被视为一种潜在解决方案，因其能消除计算与存储之间的频繁数据移动，从而大幅降低能耗。然而，AIMC 存在硬件非理想性（如 DAC/ADC 量化噪声、权重编程噪声），导致模型精度显著下降。

传统缓解策略依赖 **noise-aware retraining**，但对于大规模 MoE 模型而言，这种再训练成本极高且不可行。

因此，本文旨在解决以下核心问题：  
> **如何在不进行 retraining 的前提下，实现 MoE 模型在 AIMC 上的鲁棒部署？**

---

### **提出了什么新方法或新思路**
作者提出了一种 **无需 retraining 的异构计算框架（heterogeneous computing framework）**，将部分敏感模块在数字设备上运行，其余在模拟设备上运行，以平衡效率与精度。

#### **核心创新点：**
1. **理论驱动的专家选择机制（Theoretically Grounded Expert Selection）**  
   提出 **最大神经元范数得分（MaxNN Score）** 作为衡量专家对 **权重编程噪声** 敏感度的指标：
   $$
   \text{MaxNNScore}(s) = \prod_{\text{layer} \in \{\text{up, down, gate}\}} \max_i \|W_{:,i}\|_2
   $$
   即一个专家内所有线性层中，各神经元权重向量的最大 ℓ² 范数的乘积。

2. **理论证明：高 MaxNN Score 的专家更易受噪声影响**  
   通过理论分析表明，专门处理高频重要 token 的专家会发展出大范数神经元，因而对 AIMC 权重编程噪声更敏感，应优先在数字设备上执行。

3. **系统性的异构部署策略**  
   - 所有 **密集激活模块**（如 MHSA、LM Head、Shared Expert）均在数字设备上运行，尽管它们参数占比小（约 5–6%），但因处理所有输入而高度敏感。
   - 仅将 **Top-I% MaxNN Score 的专家** 在数字设备上运行，其余专家在 AIMC 设备上运行。

---

### **相比现有方法的优势**
| 维度 | 本文方法 | 现有方法 |
|------|--------|---------|
| **是否需要 retraining** | ❌ 不需要 | ✅ 需要（如 noise-aware retraining） |
| **专家选择依据** | ✅ 理论可解释（MaxNN Score） | ⚠️ 启发式（如激活频率、路由权重） |
| **适用模型规模** | ✅ 支持 7B–16B MoE 模型 | ⚠️ 多用于 <3B 小模型 |
| **通用性** | ✅ 可推广至不同 AIMC 架构 | ⚠️ 通常依赖特定硬件校准 |

---

## **2. 核心实验方法和设置**

### **使用的模型**
- **DeepSeekMoE**：16B 参数，28 层，每层 MoE 包含 64 个稀疏专家 + 1 个共享专家。
- **OLMoE**：7B 参数，16 层，每层 MoE 包含 64 个稀疏专家。

### **评估任务与数据集**
在 **8 个标准 LLM benchmark** 上评估性能：
- **常识推理**：PIQA, HellaSwag, WinoGrande
- **科学问答**：ARC-Easy, ARC-Challenge
- **自然语言推理**：BoolQ
- **数学推理**：MathQA
- **综合知识**：MMLU

最终报告 **平均准确率（Avg. Accuracy）**。

---

### **实验设置**
#### **模拟噪声类型**
1. **DAC-ADC 噪声**：使用 IBM 的 AIHWKIT-Lightning 模拟，配置为 8-bit DAC/ADC。
2. **权重编程噪声（Weight-programming noise）**：按文献 (Le Gallo et al., 2023) 模型添加高斯噪声：
   $$
   \Delta W_{ij} \sim \mathcal{N}(0, c^2 W_{ij}^2)
   $$
   控制噪声幅度 $ c $ 进行消融实验。

#### **评估指标**
- **Accuracy (%)**：主指标
- **Throughput (Tokens/s)**：吞吐量
- **Energy Efficiency (Tokens/Watt·s)**：能效
- **% Parameters in Analog**：衡量资源利用

#### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **All-analog** | 所有模块在 AIMC 上运行 |
| **All-digital (FP16)** | 所有模块在数字设备上运行 |
| **Activation Frequency** | 按专家被激活频率排序，高频专家放数字端 |
| **Activation Weight** | 按平均路由权重排序 |
| **Router Norm** | 按路由矩阵参数范数排序（无需校准数据） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表 1：DAC-ADC 噪声下的鲁棒性（节选）**
| 模型 | 噪声位置 | Avg. Accuracy (%) |
|------|----------|------------------|
| DeepSeekMoE | Digital (FP16) | 61.28 |
| DeepSeekMoE | Experts only | 61.03 |
| DeepSeekMoE | Experts + Dense | 55.04 |
| OLMoE | Experts only | 62.46 |
| OLMoE | Experts + Dense | 61.54 |

> **结论**：仅在专家上加噪声影响极小（↓0.25%），但在密集模块上加噪声会导致显著下降（↓>6%），验证了将密集模块保留在数字端的必要性。

---

#### **图 4 & 图 5：权重编程噪声下的性能对比（OLMoE & DeepSeekMoE）**
- 当 **12.5% 或 25% 专家** 在数字端时：
  - **MaxNN Score 方法** 在所有噪声水平下均优于其他基线。
  - 在高噪声（c=2.5）下，**恢复超过一半的性能损失**。
- 例如在 OLMoE 中，仅将 **Top 12.5% MaxNN Score 专家** 数字化，即可将精度从 ~58% 提升至 ~62%。

---

#### **表 2：异构计算的能效与吞吐权衡（OLMoE, Batch=32）**
| 数字参数比例 | 模块 | Throughput (Tokens/s) | Energy Eff. (Tokens/Watt·s) | Accuracy (c=1.5) |
|--------------|------|------------------------|-------------------------------|------------------|
| 0% (all-analog) | None | 768 | 23,949 | 58.45±0.15 |
| 100% (digital) | — | 4,220 | 10.55 | 63.17 |
| 5.37% (dense only) | Dense | 49,781 | 123.92 | 60.73±0.10 |
| 28.65% (dense + 25% experts) | — | 14,513 | 36.25 | 61.82±0.06 |

> **结论**：异构方案在保持高能效的同时，显著提升精度。通过调节数字化专家比例，可在 **能效、吞吐、精度** 之间灵活权衡。

---

### **消融实验结果**
1. **密集模块必须在数字端**  
   即使仅占 3–5% 参数，若将 MHSA、LM Head 等放入 AIMC，性能下降远超将其余 90% 专家放入 AIMC。

2. **MaxNN Score 显著优于其他选择策略**  
   在相同数字资源预算下，MaxNN Score 方法始终取得最高精度，尤其在高噪声环境下优势更明显。

3. **可视化支持理论假设**  
   - **高 MaxNN Score 专家**：响应高频词（如 "the", "a", "and"）
   - **低 MaxNN Score 专家**：响应低频词（如 "Ireland", "erry"）
   > 表明高范数专家确实专注于高频语义单元。

---

## **4. 关键结论和发现**

### **主要发现**
1. **并非所有专家都适合 AIMC**：少数具有大神经元范数的专家对权重噪声极度敏感，是性能瓶颈。
2. **MaxNN Score 是有效的理论指导指标**：首次从理论上证明其与噪声敏感性的正相关性，并在实践中验证其优越性。
3. **异构计算是大规模 MoE 部署的可行路径**：通过将 **<30% 参数** 放在数字端，即可在 **保持 >60% 精度** 的同时，获得 **百倍级能效提升**。
4. **密集模块虽小但关键**：MHSA、LM Head 等模块尽管参数少，但因全量参与计算，必须保留于数字端。

---

### **方法的局限性**
1. **依赖预定义的 MaxNN Score**：需在部署前离线计算一次，无法动态调整。
2. **未考虑动态负载变化**：实际推理中 token 分布可能漂移，固定 Top-k 专家可能不再最优。
3. **理论分析基于简化模型**：使用单层 MoE + 二分类任务，难以完全覆盖真实复杂场景。
4. **硬件假设较强**：假设 DAC/ADC 噪声可通过校准消除，但在极端条件下仍可能失效。

---

### **未来工作方向**
1. **动态异构调度机制**：根据输入 token 流实时决定哪些专家在数字/模拟端运行。
2. **联合优化编译器支持**：开发支持异构 MoE 的 AI 编译器，自动完成模块划分与映射。
3. **扩展至其他稀疏架构**：如 Block-Sparse Transformers、Dynamic ConvNets。
4. **探索更低开销的选择指标**：设计无需完整权重访问的轻量级敏感度估计方法。

---

> **总结一句话**：  
> 本文提出了一种 **无需 retraining 的异构 MoE 推理框架**，通过 **理论可证的 MaxNN Score 指标** 识别敏感专家，并结合 **密集模块保护策略**，在 **保持高精度的同时实现百倍能效提升**，为大规模 MoE 模型的绿色部署提供了新范式。

</details>

---

### 4. [Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models](https://arxiv.org/abs/2603.02631)

**Authors**: Shubhangi Upasani, Ravi Shanker Raju, Bo Li, Mengmeing Ji, John Long, Chen Wu, Urmish Thakker, Guangtao Wang  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.02631v1  

#### Abstract
Prompt length is a major bottleneck in agentic large language model (LLM) workloads, where repeated inference steps and multi-call loops incur substantial prefill cost. Recent work on speculative prefill demonstrates that attention-based token importance estimation can enable training-free prompt co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
在 **agentic LLM workloads** 中，长上下文输入（如文档解析、工具调用、代码调试）导致 **prefill 阶段成本极高**，成为推理延迟和吞吐量的主要瓶颈。传统 **Speculative Prefill** 方法依赖于与目标模型（target model）同家族的小型 draft model（共享 tokenizer），但在实际中许多前沿模型（如 DeepSeek-V3.1、Kimi-K2）缺乏可用的 in-family draft model。

因此，**如何在没有同家族 draft model 的情况下实现高效 prompt 压缩**，成为一个关键挑战。

### ✅ 提出的新方法
本文提出 **Cross-Family Speculative Prefill**，即：
- 使用来自**不同模型家族**（cross-family）的轻量级 draft model（如 Qwen、LLaMA）来为另一个家族的目标模型（如 DeepSeek）进行 prompt 压缩。
- **复用原有 Speculative Prefill 的核心机制**：基于 draft model 的 attention 分布估计 token 重要性，选择 top-k 最重要的文本块进行保留。
- **无需训练、不修改模型参数**，完全 inference-time 的压缩方法。

### ✅ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **通用性更强** | 不再依赖 in-family draft model，支持跨家族组合（如 LLaMA → DeepSeek）。 |
| **部署更灵活** | 可根据硬件、成本、tokenization 支持自由选择 draft model。 |
| **保持高性能** | 在多种任务上保留 90–100% 的 full-prompt 性能，部分任务甚至因“去噪”而提升准确率。 |
| **显著降低 TTFT** | 最高实现 **~18× 的 Time-to-First-Token（TTFT）加速**。 |
| **解耦上下文长度限制** | 利用长上下文 draft model（如 128k）压缩输入，使受限目标模型（如仅支持 32k）也能处理超长上下文。 |

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集
| 数据集 | 任务类型 | 特点 |
|-------|--------|------|
| **LongBench v1 & v2** | 多文档问答、摘要、检索、代码理解等 | 覆盖中长上下文（平均 8k–240k tokens），多语言、多任务。 |
| **RULER** | 极端长上下文任务（needle-in-a-haystack、长文档 QA） | 输入长达 **128k tokens**，测试模型对稀疏关键信息的捕捉能力。 |
| **InfiniteBench / Code Debug** | 大型代码仓库中的错误定位 | 输入为真实代码库片段，需识别注入的明显错误函数。 |

### ⚙️ 实验设置
- **Target Models**：Qwen-8B、LLaMA-8B、DeepSeek-R1、DeepSeek-V3.1  
- **Draft Models**：Qwen3-0.6B/1.7B/4B、LLaMA-3.1-8B-Instruct、LLaMA-3.2-1B-Instruct  
- **Keep Rate**：控制压缩比例（如 6%–50%），按目标模型 tokenization 定义。
- **Chunk Size**：32（LongBench/RULER），128（Code Debug，适配代码结构）
- **Lookahead Steps**：固定为 8
- **Hardware**：在自研 **Reconfigurable Dataflow Unit (RDU)** 上运行，模拟真实部署约束。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 下游任务最终输出正确率，与 full-prompt baseline 对比。 |
| **Time-to-First-Token (TTFT)** | 衡量 prefill 阶段延迟，反映压缩带来的效率增益。 |
| **Context Length Reduction** | 压缩后输入长度 vs 原始长度（如 128k → 16k）。 |
| **Baseline Performance** | 使用 full-prompt + KV cache 压缩技术（如 SnapStream）作为对比。 |

### 🔁 基线方法对比
- **Full Prompt Baseline**：直接输入完整 prompt，无压缩。
- **SnapStream KV Cache Compression**：用于 RULER 的 baseline，因内存限制必须使用，会丢失长程信息。
- **Speculative Prefill (in-family)**：原始方法，仅限同家族模型使用。

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据

#### ✅ LongBench v2（表1）
| Target Model | Draft Model | Keep Rate | Accuracy (% of baseline) |
|-------------|------------|-----------|----------------------------|
| LLaMA-8B | Qwen3-1.7B | 25% | 95–100% |
| Qwen3-8B | LLaMA-3.2-1B | 25% | ~97% |
| DeepSeek-R1 | Qwen3-4B / LLaMA-8B | 6% | ~91–93% |

> 即使在 **6% keep rate**（仅保留约 6% 内容）下，仍能保持 **90%+ 准确率**。

#### ✅ Code Debug（表2）
| Target Model | Draft Model | Keep Rate | Accuracy |
|-------------|------------|-----------|---------|
| DeepSeek-V3.1 | LLaMA-8B | 20% | 64.72% (vs 67.51%) |
| DeepSeek-V3.1 | LLaMA-8B | 15% | 59.13% (~87.6% of baseline) |
| DeepSeek-R1 | LLaMA-8B | 30% | 70.30% (vs 74.37%) |
| DeepSeek-R1 | LLaMA-8B | 15% | 62.44% (~84.0% of baseline) |

> 在代码任务上，**压缩越激进，性能下降越明显**，但仍优于纯 KV cache 压缩方案。

#### ✅ RULER（表4）
| 设置 | Input Length | TTFT | Accuracy |
|------|--------------|------|----------|
| Full Prompt + SnapStream | 128k | 46 秒 | 58.0% |
| Cross-Family SP (keep 12.5%) | 16k | **~2.5 秒** | **89.67%** |

> 实现 **~18× TTFT 加速**，同时 **准确率大幅提升**，归因于：
- **去噪效应（denoising effect）**：移除无关上下文，聚焦关键信息。
- **避免 KV cache 压缩带来的信息损失**。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **Attention-based token importance 具有强跨家族迁移性**  
   尽管 draft 和 target 模型架构、tokenizer 不同，attention 得分仍能有效指示语义重要性，表明其依赖的是 **task priors 和 semantic structure**，而非特定模型内部表示。

2. **Cross-family prompt compression 是实用且高效的部署方案**  
   在 agentic 系统中，模型栈常为异构组合，本方法允许任意选择具备长上下文能力的 draft model 来服务受限目标模型，极大提升部署灵活性。

3. **压缩可带来“去噪”收益，反而提升性能**  
   在 LongBench v2 和 RULER 中，压缩后的 prompt 有时表现**优于 full-prompt baseline**，说明大量冗余上下文可能干扰推理。

4. **大幅降低 TTFT，缓解 prefill 瓶颈**  
   最高 **18× TTFT 减少**，使得长上下文推理在低资源设备上变得可行。

---

### ⚠️ 局限性
- **对结构敏感任务（如 code debugging）效果有限**  
  当任务依赖细粒度语法或跨文件语义依赖时，过度压缩会导致关键上下文断裂，性能显著下降（如 15% keep rate 下仅保留 ~84–87% 性能）。
- **当前方法为 chunk-level selection，缺乏结构感知**  
  未考虑函数边界、类结构、控制流等代码结构特征，可能导致重要逻辑被截断。

---

### 🔮 未来工作方向（A.5）
1. **引入 task-aware structural constraints**  
   如在代码任务中强制保留函数头、变量定义区域，或结合 AST 进行 dependency-aware selection。
2. **动态 keep rate 调整**  
   根据任务类型或输入复杂度自适应调整压缩强度。
3. **探索更轻量 draft model**  
   使用 <1B 参数模型进一步降低成本。
4. **扩展至多模态 prompt 压缩**  
   探索图像、表格等非文本内容的重要性估计机制。

---

## ✅ 总结
**Cross-Family Speculative Prefill** 成功验证了：
> **Attention-driven token importance 是一种可泛化的 prompt compression primitive**，即使在跨模型家族、跨 tokenizer 的极端设定下，依然可靠有效。

该方法为 **agentic systems** 提供了一种**免训练、低成本、高兼容性**的长上下文优化路径，尤其适用于：
- 缺乏 in-family draft model 的前沿模型（如 DeepSeek、Kimi）
- 异构模型栈的生产环境
- 硬件受限但需处理超长输入的场景

是迈向 **practical long-context LLM deployment** 的重要一步。

</details>

---

### 5. [EdgeFLow: Serverless Federated Learning via Sequential Model Migration in Edge Networks](https://arxiv.org/abs/2603.02562)

**Authors**: Yuchen Shi, Qijun Hou, Pingyi Fan, Khaled B. Letaief  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.02562v1  

#### Abstract
Federated Learning (FL) has emerged as a transformative distributed learning paradigm in the era of Internet of Things (IoT), reconceptualizing data processing methodologies. However, FL systems face significant communication bottlenecks due to inevitable client-server data exchanges and long-distan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EdgeFLow: Serverless Federated Learning via Sequential Model Migration in Edge Networks

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题  
传统 **Federated Learning (FL)** 面临严重的**通信瓶颈**，主要原因包括：
- 客户端与中心化云服务器之间的频繁参数交换；
- 多跳传输导致的高延迟和网络负载；
- 在非理想无线链路下的传输开销巨大。

尤其在 **IoT 和边缘网络**场景中，这种依赖中心服务器的架构难以满足低延迟、高效率的需求。

### 提出了什么新方法或新思路  
提出 **EdgeFLow** —— 一种**无服务器（serverless）的联邦学习框架**，其核心思想是：
- **用边缘基站间的顺序模型迁移（sequential model migration）替代传统的中心云服务器聚合**；
- 所有模型聚合与传播均在**边缘集群（edge clusters）内部完成**；
- 模型通过预定义或随机序列在不同边缘基站间“流动”，实现去中心化的训练流程。

该框架包含三个阶段：
1. **Cluster Initialization**：客户端动态分组至本地化的边缘集群；
2. **Intra-Cluster Training**：当前活跃集群内进行本地训练与边缘聚合；
3. **Inter-Cluster Model Migration**：更新后的全局模型直接迁移到下一个边缘节点继续训练。

### 相比现有方法的优势  
| 对比维度 | 传统 FL / Hierarchical FL | EdgeFLow |
|--------|--------------------------|---------|
| 通信路径 | 客户端 → 边缘 → 云 → 边缘 → 客户端（多跳） | 客户端 ↔ 边缘 ↔ 下一边缘（无云参与） |
| 通信开销 | 高（需上传至远程云端） | 显著降低（仅边缘间短距离传输） |
| 架构中心性 | 强依赖中央服务器 | 完全去中心化，serverless |
| 可扩展性 | 受限于云带宽 | 更适合大规模边缘部署 |

此外，EdgeFLow 在理论层面提供了对 **non-convex 目标函数** 和 **non-IID 数据分布** 下的收敛性证明，拓展了经典 FL 收敛分析的应用范围。

---

## 2. 核心实验方法和设置

### 使用的数据集  
- **FashionMNIST**：10类服装图像分类任务；
- **CIFAR-10**：10类自然图像分类任务。

### 实验设置  
- 总客户端数 $ N = 100 $，划分为 $ M $ 个固定集群（每集群约10个客户端）；
- 每轮一个集群激活，执行 $ K=5 $ 轮本地训练（local epoch），mini-batch size = 64；
- 模型结构：六层 CNN（含 BatchNorm 和 MaxPooling），输出层为 (128, 10)，使用 Adam 优化器；
- 数据分布配置：
  - **IID**：每个客户端均匀随机分配各类样本；
  - **NIID A**：混合设置（10个 IID + 20个 95%-non-IID + 70个 98%-non-IID）；
  - **NIID B**：极端非独立同分布（10个 IID + 90个 100%-non-IID）；

### 评估指标  
- **Accuracy (%)**：测试集上的分类准确率；
- **Communication Efficiency**：每轮上传参数量、压缩比（Compression Ratio）；
- **Convergence Speed**：达到目标精度所需的通信轮次。

### 基线方法对比  
- **FedAvg**：标准联邦平均算法，作为传统 FL 基线；
- **EdgeFLowRand**：下一集群随机选择；
- **EdgeFLowSeq**：下一集群按预定序列迁移。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table I）

| 方法 | FashionMNIST (IID) | FashionMNIST (NIID A) | CIFAR-10 (IID) | CIFAR-10 (NIID A) | CIFAR-10 (NIID B) |
|------|--------------------|------------------------|---------------|------------------|------------------|
| FedAvg | 90.60% | 86.89% | 88.66% | 77.04% | 71.04% |
| EdgeFLowRand | 90.13% | **87.97%** | 89.16% | **80.26%** | **73.14%** |
| EdgeFLowSeq | 90.53% | 87.50% | 88.99% | **81.58%** | **73.36%** |

> ✅ 观察：在 **non-IID 场景下，EdgeFLow 显著优于 FedAvg**，尤其在复杂数据集（如 CIFAR-10）上提升明显（最高提升 >4.5%）。

### 与基线方法的对比结果  
- **准确性方面**：
  - 在 IID 场景下，EdgeFLow 与 FedAvg 表现相当；
  - 在 NIID 场景下，EdgeFLow 平均提升 **2~4个百分点**，表明其对数据异质性具有更强鲁棒性。
- **通信效率方面**（见 Fig. 4）：
  - EdgeFLow 相比 FedAvg 和 Hierarchical FL，通信负载显著下降；
  - 在深度线性（depth-linear）和混合拓扑中，**压缩比达 50–80%**；
  - 特别是在“深度导向”网络中优势更明显（因避免了长距离回传至云端）。

### 消融实验结果  
- **集群规模 $ N_m $ 影响**（Fig. 3a）：
  - 更大的集群带来更快收敛速度和更高最终精度；
  - 符合理论预测（更大的 $ N_m(t) $ 减小方差项）。
- **本地训练轮数 $ K $ 影响**（Fig. 3b）：
  - 增加 $ K $ 不一定提升性能，存在非单调关系；
  - 过大的 $ K $ 可能加剧 client drift，需权衡计算与通信成本。

---

## 4. 关键结论和发现

### 主要发现  
1. **EdgeFLow 能有效缓解 FL 中的通信瓶颈**，通过将模型聚合完全下沉到边缘网络，消除云往返传输；
2. **在 non-IID 数据下仍保持良好收敛性和泛化能力**，甚至优于传统 FedAvg；
3. **理论分析支持其在非凸、非独立同分布条件下的收敛保证**，扩展了 FL 收敛理论边界；
4. **通信开销可降低 50–80%**，特别适用于远距离、高延迟的边缘网络环境。

### 方法的局限性  
- 当前模型迁移采用固定或随机序列，未考虑**动态负载均衡或信道状态感知调度**；
- 缺乏对**跨集群数据偏移（distribution shift）** 的显式建模，可能影响长期稳定性；
- 所有集群必须有序参与训练，可能导致某些设备等待时间较长（冷启动问题）；
- 尚未验证在真实无线网络中的端到端延迟表现。

### 未来工作方向  
- 探索 **dynamic cluster formation** 机制以适应动态边缘环境；
- 结合 **wireless-aware scheduling** 与信道质量反馈优化迁移路径；
- 引入 **model distillation 或 gradient correction** 技术缓解顺序训练带来的遗忘问题；
- 扩展至 **cross-device 与 cross-silo FL** 多种应用场景；
- 探讨 EdgeFLow 在 **安全、隐私和个性化学习** 方面的潜力。

---

> 📌 **总结一句话**：  
> **EdgeFLow 通过“边缘间模型流动”的范式革新，实现了无需云服务器的高效联邦学习，在保持模型性能的同时大幅降低通信开销，为下一代 IoT 与 6G 边缘智能系统提供了基础架构支持。**

</details>

---

### 6. [MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks](https://arxiv.org/abs/2603.02630)

**Authors**: Zhi Hong, Qian Zhang, Jiahang Sun, Zhiwei Shang, Mingze Kong, Xiangyi Wang, Yao Shu, Zhongxiang Dai  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.02630v1  

#### Abstract
Large Language Models (LLMs) have achieved great success in many real-world applications, especially the one serving as the cognitive backbone of Multi-Agent Systems (MAS) to orchestrate complex workflows in practice. Since many deployment scenarios preclude MAS workflow modifications and its perfor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Multi-Agent Systems (MAS)** 中的 **Prompt Optimization** 问题，指出在现实部署中存在三大挑战：
1. **样本效率低下 (Sample inefficiency)**：每次评估一个 Prompt 组合都需要执行整个多智能体流程，成本高昂。
2. **拓扑诱导耦合 (Topology-induced coupling)**：上游 Agent 的 Prompt 变化会改变其输出，从而影响下游 Agent 的输入分布，导致目标函数不可分。
3. **组合爆炸 (Combinatorial explosion)**：联合 Prompt 空间是所有 Agent Prompt 候选集的笛卡尔积，搜索空间随 Agent 数量指数级增长。

### 提出的新方法：MASPOB
为解决上述问题，作者提出了 **MASPOB (Multi-Agent System Prompt Optimization via Bandits)**，一个新颖的、高效的 Prompt 优化框架，其核心创新点在于三者的结合：
- **基于 Bandit 的探索-利用权衡**：将 Prompt 优化建模为 **contextual bandit** 问题，采用 **Upper Confidence Bound (UCB)** 准则来平衡“利用”已知高分组合和“探索”不确定性高的区域，从而在有限的评估预算下最大化收益。
- **图神经网络 (GNN) 作为代理模型 (Surrogate)**：引入 **Graph Neural Network (GNN)** 显式地对 MAS 的工作流拓扑结构进行编码。GNN 将 Agent 视为节点，信息流视为边，通过消息传递学习到能够捕捉拓扑依赖关系的 Prompt 语义表示，使模型具备“拓扑感知”能力。
- **坐标上升法 (Coordinate Ascent) 进行可扩展搜索**：为了应对组合爆炸，采用 **Coordinate Ascent** 策略，将全局联合优化分解为一系列单变量子问题。在每一轮迭代中，依次固定其他 Agent 的 Prompt，只优化当前 Agent 的 Prompt，将搜索复杂度从指数级降低到线性级。

### 相比现有方法的优势
- **超越单智能体优化器**：如 OPRO、PromptBreeder 等方法独立优化每个 Agent 的 Prompt，忽略了拓扑耦合，可能导致不稳定。
- **超越通用多阶段优化器**：如 MIPRO 虽然用于多阶段流程，但通常隐式地捕获依赖关系（如通过 TPE），对拓扑结构不敏感。
- **高效且结构感知**：MASPOB 是首个将 **GNN 的结构建模能力** 与 **Bandit 的样本高效性** 相结合，专门用于固定拓扑的 MAS Prompt 优化的方法，在有限预算下能更有效地找到协调性好的 Prompt 组合。

## 2. 核心实验方法和设置

### 数据集
在六个广泛使用的公共基准上进行了评估，覆盖了多种任务：
- **问答 (Question Answering)**：`HotpotQA`, `DROP`
- **代码生成 (Code Generation)**：`HumanEval`, `MBPP`
- **数学推理 (Mathematical Reasoning)**：`GSM8K`, `MATH`

### 实验设置和评估指标
- **骨干 LLM**：默认使用 `GPT-4o-mini` 执行 Agent。
- **Prompt 编码**：使用 `Qwen3-Embedding-8B` 生成 Prompt 的嵌入向量。
- **评估预算**：每个优化方法有 **50 次验证集评估** 的预算来选择最佳 Prompt 组合。
- **最终评估**：在测试集上运行三次，报告平均得分。
- **评估指标**：
  - 数学推理 (`GSM8K`, `MATH`)：求解率 (solve rate %)
  - 代码生成 (`HumanEval`, `MBPP`)：Pass@1
  - 问答 (`HotpotQA`, `DROP`)：F1 分数

### 基线方法对比
- **无优化基线**：`IO` (直接调用), `CoT`, `ReAct`
- **单智能体 Prompt 优化**：`PromptBreeder`, `Instinct`
- **多智能体系统**：`AFlow` (生成工作流), `MIPRO` (多阶段 Prompt 优化基线)

## 3. 主要实验结果和性能指标

### 关键性能数据
- **总体性能**：MASPOB 在所有六个基准上的平均得分为 **80.58%**，显著优于所有基线。
- **对比提升**：
  - 相比 `IO` 基线，平均提升 **12.02%**。
  - 相比强大的多智能体基线 `AFlow` 和 `MIPRO`，分别提升了 **2.06%** 和 **1.71%**。

### 与基线方法的对比结果
- **全面领先**：如表1所示，MASPOB 在每一个单独的基准测试上都取得了最佳结果。
- **收敛速度快**：如图3所示，MASPOB 能够快速稳定，通常在35轮左右就能在测试集上达到近似最优性能，表明其样本效率高。
- **对复杂拓扑的泛化能力强**：在使用 AFlow 生成的更复杂的 MAS 结构（更多 Agent）上进行测试时，MASPOB 依然保持最佳性能（见表2），而 `MIPRO` 的表现甚至不如 `AFlow`，说明其显式的拓扑建模至关重要。

### 消融实验结果
- **GNN 的有效性**：将 GNN 替换为普通的 MLP 后，平均性能下降了 **2.31%**（见表3）。这证明了显式地对工作流拓扑进行建模对于捕捉 Agent 间的耦合关系、识别协调的 Prompt 组合是至关重要的。
- **坐标上升法的有效性**：与穷举的全局搜索相比，坐标上升法虽然性能略有下降（约0.3-0.5%），但运行时间减少了 **98%-99.8%**（见表5和图4），实现了计算成本和优化质量之间的极佳权衡。
- **线性不确定性优于神经不确定性**：尝试用更复杂的神经不确定性估计器替代 LinUCB 的线性不确定性后，性能反而下降（见表8），说明在数据稀缺的场景下，简单、稳定的线性不确定性估计更为鲁棒。

## 4. 关键结论和发现

### 主要发现
1. **Prompt 优化是改进冻结拓扑 MAS 的有效杠杆**：在许多高风险应用中，修改工作流拓扑是不被允许的，而优化 Prompt 是一种安全且有效的性能提升手段。
2. **拓扑感知是关键**：忽略 Agent 间的拓扑依赖关系会导致次优解。MASPOB 通过 GNN 显式地建模这些依赖，是其成功的关键。
3. **样本效率至关重要**：在昂贵的评估环境下，结合 Bandit 的 UCB 探索策略和坐标上升法，使得 MASPOB 能够以极少的评估次数找到高性能的 Prompt 组合。
4. **协调性优于孤立优化**：MASPOB 的优势不仅体现在单一任务上，而是源于对整个 Agent 团队的更好协调。

### 方法的局限性
- **依赖于预定义的工作流**：MASPOB 专注于优化给定拓扑下的 Prompt，不涉及工作流结构本身的搜索。
- **GNN 和 Bandit 的假设**：其性能依赖于 GNN 对拓扑依赖关系的准确建模以及线性 Bandit 模型的适用性。

### 未来工作方向
- **结合结构搜索**：将 Prompt 优化与工作流拓扑的动态搜索相结合，实现端到端的优化。
- **探索更复杂的代理模型**：研究是否可以使用更强大的模型（如 Transformer-based surrogate）来进一步提升预测精度。
- **应用于更广泛的场景**：将 MASPOB 框架推广到更多类型的多智能体协作系统中。

</details>

---

### 7. [HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents](https://arxiv.org/abs/2603.00977)

**Authors**: Hongbo Jin, Rongpeng Zhu, Jiayu Ding, Wenhao Zhang, Ge Li  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00977v1  

#### Abstract
Large language model (LLM) agents have recently demonstrated strong capabilities in interactive decision-making, yet they remain fundamentally limited in long-horizon tasks that require structured planning and reliable execution. Existing approaches predominantly rely on flat autoregressive policies...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Model (LLM)** 的智能体在处理**长时程任务（long-horizon tasks）** 时面临三大挑战：
- **指数级探索复杂度**（exponential exploration complexity）
- **延迟奖励分配**（delayed credit assignment）
- **语义漂移**（semantic drift），即推理过程中逐渐偏离原始目标

现有方法多采用“扁平化”策略（flat autoregressive policy），将高层规划与底层动作生成混在同一 token 序列中，导致错误传播严重、效率低下。

---

### 🚀 提出的新方法：HiMAC
提出 **HiMAC**（Hierarchical Macro-Micro Agentic Control），一种分层强化学习框架，显式地将决策过程分解为两个层级：

#### （1）**Macro-Policy（宏观策略）**
- 负责**战略规划**，输入任务指令 $x$，输出一个结构化的自然语言子目标序列 $z = \{g_1, ..., g_K\}$，称为 **Structured Blueprint**
- 将长周期任务分解为可管理的里程碑

#### （2）**Micro-Policy（微观策略）**
- 作为**执行器**，以 Blueprint 和当前环境观测为条件，逐个完成每个子目标
- 引入 `<sub_done>` 特殊标记，实现自主的子目标切换与进度控制

> 二者共享同一个 LLM 参数，但通过训练机制解耦优化路径。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 PPO, GRPO） | HiMAC |
|------|--------------------------|-------|
| 架构 | 扁平策略（Flat Policy） | 分层架构（Hierarchical） |
| 探索空间 | 高维联合动作-推理空间 | 分离后的低维语义与动作空间 |
| 错误传播 | 容易累积并扩散 | 被限制在单个子目标内 |
| 信用分配 | 困难且不稳定（尤其无 critic 时） | 层级相对优势估计（HRAE）提升精度 |
| 训练稳定性 | 易受非平稳性影响 | 迭代共进化训练缓解动态变化 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在三个具有代表性的长时程任务基准上进行验证：

| 数据集 | 任务类型 | 特点 |
|--------|---------|------|
| **ALFWorld** | 多步具身推理（embodied reasoning） | 在模拟家庭环境中执行“拿取、加热、清洁”等操作，测试逻辑连贯性和物理常识 |
| **WebShop** | 网页导航与购物 | 在噪声大、观察复杂的电商网站中查找并购买指定商品，考验长期意图保持能力 |
| **Sokoban** | 视觉接地的空间规划 | 推箱子游戏，需精确顺序推理，结合图像输入进行决策 |

---

### ⚙️ 实验设置与评估指标

#### 模型骨干
- 文本任务（ALFWorld, WebShop）：`Qwen2.5-Instruct`（1.5B / 7B）
- 视觉任务（Sokoban）：`Qwen2.5-VL`, `Qwen3-VL` 系列 VLMs

#### 训练细节
- 使用 **Group Relative Policy Optimization (GRPO)** 的变体进行 critic-free 优化
- 每轮迭代交替执行：
  - **Phase A**: Macro-Exploration（更新 Macro-Policy）
  - **Phase B**: Micro-Adaptation（更新 Micro-Policy）
- KL 散度系数 $\beta = 0.01$

#### 评估指标
- **Success Rate (%)**：任务成功完成的比例
- **Score**：综合得分（如 WebShop 中的价格折扣得分）
- **Sample Efficiency**：达到特定成功率所需的训练迭代次数

---

### 🆚 基线方法对比
涵盖以下几类主流方法：
- **Prompting 方法**：ReAct, Reflexion
- **RL 方法（含 critic）**：PPO
- **Critic-free RL 方法**：RLOO, GRPO, GiGPO
- **闭源模型**：GPT-4o, Gemini-2.5-Pro, Claude Sonnet 4.5

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

| 方法 | ALFWorld (7B) | WebShop Success (7B) | WebShop Score (7B) | Sokoban Success (7B) |
|------|---------------|------------------------|------------------------|------------------------|
| **GiGPO** | 90.8% | 75.2% | 86.2 | 82.8% |
| **HiMAC (Ours)** | **92.1%** | **84.1%** | **93.8** | **87.5%** |

> - 在 WebShop 上相比最强 RL 基线 **GiGPO 提升达 16.0% 成功率**
> - 在 ALFWorld 上超越所有基线，接近完美表现（92.1%）
> - 在视觉任务 Sokoban 上也显著领先（+4.7pp）

---

### 🔍 消融实验结果（Table 3）

| 变体 | ALFWorld (%) | WebShop Score | WebShop Succ. (%) |
|------|--------------|----------------|--------------------|
| **HiMAC (Full)** | 92.1 | 93.8 | 84.1 |
| w/o Hierarchy (Flat GRPO) | 77.6 (-14.5) | 79.3 (-14.5) | 66.1 (-18.0) |
| w/o Iterative Co-Evolution | 85.3 (-6.8) | 86.7 (-7.1) | 74.8 (-9.3) |
| w/o `<sub_done>` | 88.2 (-3.9) | 90.1 (-3.7) | 79.8 (-4.3) |
| Random Blueprint | 89.7 (-2.4) | 91.6 (-2.2) | 81.5 (-2.6) |

#### 结论：
- **分层结构本身带来最大收益**（>14% 下降）
- **迭代共进化训练对稳定性至关重要**，尤其在长任务中
- `<sub_done>` 标记使执行节奏自适应，避免固定步数浪费或中断
- 高置信蓝图选择是高效训练的关键“质量过滤器”

---

### 📈 样本效率分析（Table 4）
| Benchmark (Target) | Iters to Reach Target ↓ |
|--------------------|------------------------|
| ALFWorld (75%) | HiMAC: ~110 vs. GRPO: ~150 |
| WebShop (65%) | HiMAC: ~220 vs. GRPO: ~380 |
| Sokoban (80%) | HiMAC: ~180 vs. GRPO: ~210 |

> HiMAC 在更少训练样本下更快收敛，体现出更强的 **sample efficiency**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **结构化分层优于单纯扩大模型规模**
   - HiMAC 在仅 1.5B 模型上就超过了闭源 Gemini-2.5-Pro（60.3% → 89.9%）
   - 表明 **inductive bias 的设计比参数量增长更具边际效益**

2. **分层解耦有效抑制语义漂移**
   - Macro-Policy 输出的 Blueprint 充当“路线图”，防止执行过程迷失方向
   - 特别适用于 WebShop 这类高噪声、长交互场景

3. **迭代共进化训练稳定了双层非平稳优化**
   - 冻结一方更新另一方，形成天然课程学习（curriculum learning）
   - Planner 从简单计划起步，随 Executor 能力增强逐步提出更复杂策略

4. **涌现出高级行为**
   - 训练后期 Macro-Policy 自主添加“确认步骤”（如 “inventory to confirm”），表现出**自我验证（self-verification）** 行为
   - 这种闭环反馈机制未在任何 flat baseline 中出现

---

### ⚠️ 方法的局限性
- 当前 Blueprint 是自然语言形式，缺乏形式化语义约束，可能引入歧义
- 对 `<sub_done>` 的依赖要求环境支持明确的状态判断
- 当前训练仍需大量交互数据，在真实世界部署成本较高
- 分层调度完全由 Micro-Policy 自主触发，缺乏外部干预接口

---

### 🔮 未来工作方向
- 将 HiMAC 扩展到更开放域环境（open-ended environments）
- 探索 Blueprint 的跨任务迁移与复用（transferability of blueprints）
- 引入多模态中间表示（如 symbolic graph）增强 Blueprint 的结构一致性
- 结合 memory 机制支持超长周期任务（ultra-long horizon）

---

## 总结
> **HiMAC 证明了：对于构建稳健的长周期 LLM Agent，结构化分层（structured hierarchy）比盲目扩大模型规模更为关键。**

通过 **Macro-Micro 解耦 + Critic-Free 分层优化 + Iterative Co-Evolution 训练范式**，HiMAC 在多个复杂基准上实现了 SOTA 性能，并展现出更高的样本效率与更强的行为可解释性，为下一代 agentic AI 提供了一条可行的设计路径。

</details>

---

### 8. [Enhancing Physics-Informed Neural Networks with Domain-aware Fourier Features: Towards Improved Performance and Interpretable Results](https://arxiv.org/abs/2603.02948)

**Authors**: Alberto Mi\~no Calero, Luis Salamanca, Konstantinos E. Tatsis  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.02948v1  

#### Abstract
Physics-Informed Neural Networks (PINNs) incorporate physics into neural networks by embedding partial differential equations (PDEs) into their loss function. Despite their success in learning the underlying physics, PINN models remain difficult to train and interpret. In this work, a novel modeling...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Physics-Informed Neural Networks (PINNs)** 在实际应用中的以下关键挑战提出了解决方案：
- **训练困难**：由于多目标损失函数（PDE残差、边界条件、初始条件）之间存在梯度刚性（gradient stiffness），导致优化过程不稳定。
- **频谱偏差（spectral bias）**：标准PINNs倾向于学习低频解，难以捕捉高频率或复杂空间变化的物理现象。
- **解释性差**：尽管嵌入了物理先验，PINNs仍被视为“黑箱”模型，缺乏对模型推理过程的理解机制。
- **边界条件处理繁琐**：传统方法需要显式地在损失函数中加入边界条件项，并进行复杂的损失平衡（loss balancing）。

### 提出的新方法
论文提出了两个核心创新：

#### （1）Domain-aware Fourier Features (DaFFs)
- **思想来源**：从偏微分方程（PDE）定义域上的拉普拉斯算子（Laplace operator）的特征值问题出发，构造满足齐次边界条件的傅里叶基函数。
- **数学基础**：求解如下特征系统：
  $$
  -\nabla^2 \phi_j(\mathbf{x}) = \lambda_j \phi_j(\mathbf{x}), \quad \mathbf{x} \in \Omega \\
  h(\phi_j(\mathbf{x})) = 0, \quad \mathbf{x} \in \partial\Omega
  $$
  其中 $\phi_j$ 即为DaFF，天然满足边界条件。
- **实现方式**：
  - 对于简单几何（如矩形），可解析计算（例如 $\sin(m\pi x/a)\sin(n\pi y/b)$）。
  - 对于复杂几何，可通过有限差分法（Finite Difference）数值求解离散化的拉普拉斯矩阵得到前几个特征向量作为DaFF。

#### （2）基于LRP的可解释性框架
- 引入 **Layer-wise Relevance Propagation (LRP)** 方法，用于分析输入特征（特别是DaFF）对最终预测的贡献。
- 特别设计了适用于残差连接（residual skip connections）的LRP传播规则，确保相关性守恒。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **训练效率** | 无需显式边界损失项，将多目标优化简化为单目标（仅PDE残差），消除损失平衡需求，显著加速收敛。 |
| **精度与稳定性** | 实现了**数量级更低的误差**（如验证误差达 $10^{-18}$），且训练曲线更稳定。 |
| **泛化能力** | DaFF由物理域本身决定，具有更强的归纳偏置（inductive bias），避免随机性带来的不可复现问题。 |
| **可解释性** | LRP分析显示，DaFF赋予了更符合物理直觉的特征重要性分布，而RFFs的结果则呈现杂乱无章的模式。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
本研究采用**零数据（zero-data）范式**，即不依赖真实观测数据，而是通过求解已知解析解的PDE来生成“标签”。具体使用了两个经典物理问题：
1. **Kirchhoff-Love Plate Equation**  
   描述薄板弯曲问题，涉及四阶PDE和混合边界条件（Dirichlet + Neumann）。
2. **Helmholtz Equation**  
   描述稳态波动问题，为二阶椭圆型PDE，具有齐次Dirichlet边界条件。

### 实验设置
- **Collocation Points**：在空间域 $\Omega$ 内部和边界 $\partial\Omega$ 上均匀采样，其中内部占75%，边界占25%。
- **Batch Size**：512, 1024, 2048。
- **网络架构**：全连接前馈神经网络（fully-connected feed-forward NN），层数2–5，每层单元数64–256。
- **激活函数**：tanh。
- **优化器**：Adam + BFGS（共50000 epochs，含5000步BFGS）。
- **学习率调度**：2000轮耐心期后衰减0.1，早停机制（patience=2000）。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Training Loss** | 多项损失加权平均（PDE残差 + 边界条件等）。 |
| **Validation Loss** | 在固定网格上计算的MSE（均方误差）：$\text{MSE} = \frac{1}{N}\sum(u_{pred} - u_{true})^2$，反映与真实解的差距。 |
| **Training Time** | 总训练耗时。 |

### 基线方法对比
| 方法 | 简介 |
|------|------|
| **Vanilla PINN** | 原始PINN，直接输入 $(x,t)$，需多个损失项并使用ReLoBRaLo进行动态损失平衡。 |
| **PINN-RFFs** | 使用Random Fourier Features进行位置编码，提升高频表达能力，但仍需边界损失和损失平衡。 |
| **PINN-DaFFs** | 本文提出的方法，使用Domain-aware Fourier Features作为输入编码，仅保留PDE残差损失项。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1 & Table 2）

#### Kirchhoff-Love 问题
| Method | Training Loss | Validation Loss | Training Time |
|--------|---------------|------------------|----------------|
| PINN | $2.71 \times 10^{-4}$ | $2.75 \times 10^{-6}$ | 01h 03m 04s |
| PINN-RFFs | $7.74 \times 10^{-6}$ | $3.01 \times 10^{-7}$ | 44m 09s |
| **PINN-DaFFs** | $\mathbf{1.02 \times 10^{-9}}$ | $\mathbf{6.42 \times 10^{-18}}$ | **45m 51s** |

> ✅ **结论**：PINN-DaFFs 验证误差比最佳基线（PINN-RFFs）低 **11个数量级**。

#### Helmholtz 问题
| Method | Training Loss | Validation Loss | Training Time |
|--------|---------------|------------------|----------------|
| PINN | $1.23 \times 10^{-3}$ | $5.48 \times 10^{-5}$ | 02h 51m 25s |
| PINN-RFFs | $8.23 \times 10^{-4}$ | $7.75 \times 10^{-6}$ | 01h 13m 06s |
| **PINN-DaFFs** | $\mathbf{3.86 \times 10^{-5}}$ | $\mathbf{2.32 \times 10^{-11}}$ | **31m 25s** |

> ✅ **结论**：PINN-DaFFs 不仅精度更高（验证误差低6个数量级以上），而且训练时间仅为vanilla PINN的约1/5。

### 与基线方法的对比结果
- **收敛速度**：PINN-DaFFs 在极短时间内达到极高精度，而其他方法即使长时间训练也难以逼近。
- **训练稳定性**：PINN-DaFFs 的损失曲线虽有波动，但整体尺度极小；相比之下，vanilla PINN 和 PINN-RFFs 受限于损失平衡策略，表现出明显的震荡。
- **无需调参**：DaFFs 自动编码了几何与边界信息，避免了RFFs中对variance参数的敏感依赖。

### 消融实验与分析（XAI）
通过 **LRP-based XAI Analysis** 进行深入探究：
- **Vanilla PINN**：输入坐标 $(x,y)$ 的贡献图显示严重不对称和噪声，违背了问题本身的对称性假设，说明模型未真正理解物理规律。
- **PINN-RFFs**：不同训练运行下，重要RFF成分的位置随机分布，缺乏一致性，表明模型容易陷入局部最优。
- **PINN-DaFFs**：
  - 不同 $(m,n)$ 模式的DaFF贡献清晰可辨，且与理论预期一致（如Helmholtz问题中$m=8,n=2$比$m=8,n=8$更重要）。
  - 贡献图具有高度重复性和物理解释性，证明模型学会了基于物理本质的表示。

---

## 4. 关键结论和发现

### 主要发现
1. **DaFFs 极大提升了 PINNs 的性能与鲁棒性**：通过将物理域特性编码进输入空间，自然满足边界条件，简化训练流程，实现超低误差和快速收敛。
2. **DaFFs 显著增强模型可解释性**：相比RFFs的“盲目随机”，DaFFs提供了**物理意义明确的特征空间**，使得LRP等XAI方法能够揭示出符合物理直觉的决策依据。
3. **可解释性可用于指导特征选择**：LRP可以识别出哪些DaFF组合最重要，从而为后续模型压缩或超参数设计提供依据。

### 方法的局限性
- 当前方法主要针对**齐次边界条件**（homogeneous BCs），非齐次情况需额外分解处理。
- 对于非常复杂的几何形状，数值求解拉普拉斯特征系统可能带来额外计算开销。
- DaFF的设计依赖于对主导算子（如$\nabla^2$）的认知，在高度非线性或多物理场耦合场景下推广尚待验证。

### 未来工作方向
1. 将DaFF扩展至**非齐次边界条件**和**时变问题**。
2. 探索DaFF在**稀疏观测下的系统辨识**（system identification）任务中的潜力。
3. 结合XAI进一步诊断模型偏差，推动**物理感知的模型架构设计**。
4. 将DaFF应用于更广泛的PDE类型，如Navier-Stokes方程、Maxwell方程组等。

</details>

---

### 9. [MaBERT:A Padding Safe Interleaved Transformer Mamba Hybrid Encoder for Efficient Extended Context Masked Language Modeling](https://arxiv.org/abs/2603.03001)

**Authors**: Jinwoong Kim, Sangjin Park  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.03001v1  

#### Abstract
Self attention encoders such as Bidirectional Encoder Representations from Transformers(BERT) scale quadratically with sequence length, making long context modeling expensive. Linear time state space models, such as Mamba, are efficient; however, they show limitations in modeling global interactions...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MaBERT: A Padding-Safe Interleaved Transformer-Mamba Hybrid Encoder for Efficient Extended-Context Masked Language Modeling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于 **Transformer** 的预训练编码器（如 BERT）在处理长序列时面临严重的效率瓶颈，其 **self-attention** 机制的时间和内存复杂度为 $O(n^2)$，导致扩展上下文长度（如从 512 到 4,096 tokens）成本极高。  
同时，新兴的 **Mamba** 等线性时间状态空间模型（SSM）虽然计算高效（$O(n)$），但在双向掩码语言建模（MLM）任务中存在两个关键问题：
- **padding-induced state contamination**：变长批次中的 padding tokens 会污染 SSM 的隐藏状态传播，影响有效 token 的表示质量。
- 缺乏全局依赖建模能力。

### 🚀 提出的新方法与创新思路
作者提出 **MaBERT**，一种新型的混合编码器架构，核心思想是**在层级别交错堆叠 Transformer 和 Mamba 层**，结合两者优势：

- **Interleaved Architecture**：交替使用 Transformer 层进行全局上下文建模和 Mamba 层实现线性时间状态更新。
- **Padding-Safe Masking (PSM)**：在 Mamba 层前后引入两阶段 masking：
  - **Pre-SSM Masking**：在输入进入 SSM 前屏蔽 padding 位置。
  - **Post-Block Masking**：在残差连接后重新将 padding 位置置零，防止其通过 FFN 或残差路径传播。
- **Mask-Aware Attention Pooling (MAP)**：在句子级表示聚合时，显式忽略 padding tokens，并对有效 token 进行加权池化，提升下游任务鲁棒性。

### 🔍 相比现有方法的优势
| 维度 | MaBERT 优势 |
|------|-------------|
| **效率** | 在扩展上下文至 4,096 tokens 时，相比主流编码器平均减少 **2.36× 训练时间** 和 **2.43× 推理延迟**。 |
| **准确性** | 在 GLUE 基准上，在 8 项任务中有 5 项取得最佳成绩，尤其在 CoLA 和多个 sentence-pair 任务（MRPC、QQP、QNLI、RTE）表现突出。 |
| **稳定性** | 通过 PSM 和 MAP 有效缓解了变长输入下的 state contamination 问题，提升了表示一致性。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **预训练数据**：BookCorpus + English Wikipedia（标准 BERT 预训练语料）
- **下游评估数据**：GLUE benchmark 包含以下 8 个任务：
  - 单句分类：CoLA（语法可接受性）、SST-2（情感分析）
  - 句子对任务：MRPC、QQP（相似性）、MNLI-m/mm、QNLI、RTE（自然语言推理）

### 📊 评估指标
| 任务 | 指标 |
|------|------|
| CoLA | Matthews Correlation Coefficient (MCC) |
| SST-2, MNLI, QNLI, RTE | Accuracy |
| MRPC, QQP | F1 Score |

最终报告各任务均值 ± 标准差（5 个随机种子）。

### ⚖️ 实验设置
- **预训练协议**：
  - 仅使用 MLM 目标（无 NSP）
  - 总步数 1M，分 10%/25%/50%/100% 截断以比较训练预算影响
  - 序列长度调度：前 90% 步用 128 tokens，后 10% 扩展到 512 tokens
- **硬件配置**：单张 NVIDIA A100 80GB GPU，bf16 精度
- **公平性控制**：
  - 所有模型使用默认 tokenizer
  - 禁用 FlashAttention、packing、compilation 等优化，确保 kernel 公平
  - 参数不强制对齐，关注“scaling trend”而非绝对参数量

### 🆚 基线方法对比
| 模型 | 类型 |
|------|------|
| BERT | 原始 Transformer 编码器 |
| ALBERT | 参数共享轻量化版本 |
| Longformer | 稀疏注意力（滑动窗口 + 全局 attention） |
| BigBird | 稀疏注意力（随机 + 窗口 + 全局） |
| DeBERTa | 改进 attention 结构（解耦 content/position） |

所有基线均采用相同预训练配方，仅架构不同。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table 2，100% 预训练）

| Model | CoLA ↑ | SST-2 ↑ | MRPC ↑ | QQP ↑ | MNLI-m ↑ | MNLI-mm ↑ | QNLI ↑ | RTE ↑ | 平均排名 |
|-------|--------|---------|--------|-------|----------|-----------|--------|-------|----------|
| BERT | 0.522 | 0.912 | 0.853 | 0.856 | 0.826 | 0.829 | 0.876 | 0.618 | — |
| ALBERT | 0.503 | 0.920 | 0.855 | 0.857 | 0.829 | 0.832 | 0.880 | 0.618 | — |
| Longformer | 0.534 | 0.924 | 0.863 | 0.858 | 0.830 | 0.831 | 0.882 | 0.626 | — |
| BigBird | 0.528 | 0.926 | 0.864 | 0.857 | 0.831 | 0.832 | 0.881 | 0.624 | — |
| DeBERTa | 0.617 | 0.934 | 0.862 | 0.868 | 0.838 | 0.842 | 0.886 | 0.648 | — |
| **MaBERT (Ours)** | **0.676** | **0.933** | **0.869** | **0.879** | **0.835** | **0.837** | **0.893** | **0.654** | **Best on 5/8** |

> ✅ **亮点**：在 CoLA 上大幅领先（+0.059 vs DeBERTa），表明更强的语法建模能力；在多个 sentence-pair 任务中也排名第一。

### 🔁 不同 interleaving 模式的消融（Table 1）
- 最优模式为 **MMTMMTMMTMMT**（即每三个块中两个 Mamba + 一个 Transformer）
- 单一架构（全 Mamba 或全 Transformer）性能较差
- 表明 **周期性注入全局上下文** 对性能至关重要

### 🧪 组件消融实验（Table 3）

| 模型变体 | CoLA ↓ | 整体趋势 |
|----------|--------|--------|
| Full (PSM + MAP) | 0.676 | 最佳 |
| PSM only ([CLS]) | 0.641 | 下降明显，说明 MAP 重要 |
| MAP only (no PSM) | 0.661 | 仍有下降，说明 PSM 必要 |
| None (无 PSM/MAP) | 0.596 | 显著退化，验证二者互补 |

> 💡 发现：**PSM 和 MAP 提供互补增益**，共同保障变长输入下的表示稳定性。

### ⏱️ 效率与可扩展性（Figure 6 & Tables B5–B7）

| 指标 | 结果 |
|------|------|
| **训练步时延（4,096 tokens）** | MaBERT: **3.32s** vs DeBERTa: **16.14s** → **快 4.86×** |
| **推理延迟（4,096 tokens）** | MaBERT: **34.7ms** vs DeBERTa: **206.6ms** → **低 5.95×** |
| **峰值内存（4,096 tokens）** | MaBERT: **2.45GB** < BigBird (**2.89GB**) < DeBERTa (**5.26GB**) |

> 📉 趋势：随着序列增长，MaBERT 的 memory 和 runtime 增长更平缓，展现出优越的 **long-context scalability**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **混合架构优于单一范式**：将 **Transformer 的全局交互能力** 与 **Mamba 的线性时间效率** 相结合，可在保持高性能的同时显著提升长序列处理效率。
2. **padding 安全机制至关重要**：提出的 **PSM** 和 **MAP** 有效解决了 SSM 在 encoder MLM 中的 state contamination 问题，是实现稳定训练的关键。
3. **MaBERT 在准确性和效率之间取得更好平衡**：不仅在 GLUE 多项任务上超越强基线（尤其是 CoLA 和 sentence-pair 任务），而且在扩展上下文时展现出明显的训练和推理加速。

### ⚠️ 方法的局限性
- **评估范围有限**：目前仅在 GLUE 分类任务上验证，未测试真正的 **long-context reasoning**（如文档问答、摘要）或生成任务。
- **未启用高级优化**：禁用了 FlashAttention、sequence packing 等现代优化技术，实际部署中可能与其他系统不完全可比。
- **参数量较大**：由于引入 SSM 模块，MaBERT 参数更多（~205M vs BERT ~110M），虽效率更高但仍需注意模型规模。

### 🔮 未来工作方向
- 在 **long-context understanding benchmarks**（如 LRA、NarrativeQA）上进一步验证 MaBERT 的能力。
- 探索针对长上下文设计的 **定制化预训练 curriculum**。
- 将该架构应用于多模态或时间序列等其他序列建模领域。

--- 

> **总结一句话**：  
> MaBERT 是首个成功将 Mamba 引入 **encoder-style MLM 预训练** 的混合架构，通过 **interleaving design + padding-safe mechanisms**，实现了高精度与高效长上下文建模的统一。

</details>

---

### 10. [MuxTune: Efficient Multi-Task LLM Fine-Tuning in Multi-Tenant Datacenters via Spatial-Temporal Backbone Multiplexing](https://arxiv.org/abs/2603.02885)

**Authors**: Chunyu Xue, Yi Pan, Weihao Cui, Quan Chen, Shulai Zhang, Bingsheng He, Minyi Guo  
**Category**: cs.DC  
**Published**: 2026-03-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.02885v1  

#### Abstract
Parameter-Efficient Fine-Tuning (PEFT) is widely applied as the backend of fine-tuning APIs for large language model (LLM) customization in datacenters. Service providers deploy separate instances for individual PEFT tasks, giving rise to prominent resource inefficiencies, including (1) GPU underuti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MuxTune: Efficient Multi-Task LLM Fine-Tuning in Multi-Tenant Datacenters via Spatial-Temporal Backbone Multiplexing

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在多租户数据中心中，基于 **Parameter-Efficient Fine-Tuning (PEFT)** 的大语言模型（LLM）微调服务面临严重的资源效率低下问题，主要包括：
- **GPU 利用率低**：由于 PEFT 引入的小规模算子（如 LoRA 的 down-projection）计算密度低，导致 SM（Streaming Multiprocessor）利用率下降。
- **设备停顿（Device Stall）严重**：并行训练中的通信延迟和数据依赖（如 Pipeline Parallelism 中的 Bubble）无法被有效隐藏，尤其因 PEFT 不更新骨干权重而无法使用如 DeepSeek-V3 DualPipe 或 ZeroBubble 等先进优化技术。

### 提出了什么新方法或新思路
提出 **MuxTune** —— 一种支持多任务并发执行的高效 PEFT 微调系统，其核心思想是通过 **空间-时间维度上的骨干网络复用（Spatial-Temporal Backbone Multiplexing）** 来提升资源利用率并减少停顿。

具体创新包括：
- **统一的 PEFT 表示与模块化骨干共享**：将不同类型的 PEFT（Additive/Selective/Reparameterized）抽象为统一的 `BaseOp`、`Adapter`、`Dispatch` 和 `Aggregate` 四个子模块，实现灵活的任务动态注册与骨干共享。
- **层次化的协同调度框架**：
  - **任务级融合（Task Fusion）**：引入“混合任务”（hTask）抽象，结合空间批处理（spatial batching）与时间交错（temporal interleaving），以平衡利用率与停顿隐藏。
  - **算子级编排（Operator Orchestration）**：在两级并行（inter/intra-stage）下进行细粒度调度，解决多 DAG 调度问题，并通过优先级队列实现通信与计算重叠。
  - **数据级对齐优化（Data Alignment）**：采用基于 chunk 的序列打包策略，在保持模型收敛的同时最小化跨任务的无效填充（ineffective tokens）。

### 相比现有方法的优势
| 维度 | 传统方法（如 HF-PEFT, NeMo） | MuxTune |
|------|-------------------------------|---------|
| 资源利用 | 单任务独占实例，GPU 利用率低 | 多任务共享骨干，显著提高利用率 |
| 停顿处理 | 缺乏细粒度控制，难以隐藏通信开销 | 显式调度实现计算-通信重叠 |
| 内存效率 | 每任务复制骨干参数，内存浪费严重 | 骨干参数共享，内存占用大幅降低 |
| 扩展性 | 仅支持粗粒度共置（如 MPS） | 支持动态任务到达与弹性扩展 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验使用三个具有不同序列长度的真实世界下游任务数据集：
- **SST2**：情感分析，平均序列长度较短（pad to 64）
- **OpenBookQA (QA)**：问答任务，序列长度中等（pad to 128）
- **RTE**：文本蕴含识别，序列较长（pad to 256）

组合方式分为两类：
- **Uniform**：所有任务使用相同数据集
- **Non-uniform**：多个任务使用不同数据集，模拟真实异构负载

### 实验设置和评估指标

#### 测试平台
- **Testbed-A**：4× NVIDIA A40 GPUs（NVLink 连接）
- **Testbed-B**：8 节点 × 2× A40（InfiniBand）
- **Testbed-C**：8× NVIDIA H100 GPUs（NVLink）

#### 模型配置
| 模型 | 参数量 | 层数 | 并行策略 |
|------|--------|-------|----------|
| GPT2.7B | 2.7B | 32 | 2-GPU Tensor Parallel |
| LLaMA2-7B/13B | 7B/13B | 32/40 | 4/8-GPU Pipeline |
| OPT-30B | 30B | 48 | 16-GPU Hybrid Parallel |

#### 评估指标
- **Throughput**：每秒处理 token 数（tokens/sec）
- **Effective Throughput**：排除 padding 后的有效吞吐
- **Memory Footprint**：峰值显存占用
- **End-to-End Latency**：完整训练周期耗时
- **GPU/NVLink Utilization**

### 基线方法对比
1. **HuggingFace PEFT (HF-PEFT)**：主流易用库，缺乏底层优化
2. **NeMo Megatron**：支持高效内核与并行策略，但仍为单任务设计
3. **SLoRA-PEFT (SL-PEFT)**：借鉴 SLoRA 的骨干共享与批处理机制，代表当前最先进的多任务 PEFT 尝试

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **最高可达 2.33× 吞吐提升**
- **最大实现 5.29× 显存占用降低**
- 在 H100 上相比 NeMo 达到 **5.29× 更高吞吐**

### 与基线方法的对比结果
#### 吞吐表现（Figure 14 & 15）
| 场景 | 对比 HF-PEFT | 对比 NeMo | 对比 SL-PEFT |
|------|---------------|------------|----------------|
| Uniform（A40） | ↑2.33× | ↑1.87× | ↑1.64× |
| Non-uniform（A40） | ↑2.23× | ↑1.83× | ↑1.85× |
| Uniform（H100） | - | ↑5.29× | ↑2.31× |
| Non-uniform（H100） | - | ↑3.69× | ↑1.94× |

> ⚠️ H100 上性能增益更大，因其更强的计算能力放大了单任务框架的利用率瓶颈。

#### 显存效率（Figure 17）
- 在 GPT2.7B + 2×A40 设置下：
  - MuxTune 可容纳 **32 个任务**，而 NeMo/HF-PEFT 在第 15 个任务即 OOM
  - 显存占用仅为基线的 **1/4.67**
- 在 LLaMA7B + 4×A40 设置下：
  - MuxTune 显存占用降低 **3.57×**，且具备更好扩展性

### 消融实验结果（Figure 16）
禁用各组件后在 LLaMA7B 上的性能下降（全局 batch size=128）：
| 组件 | 轻负载下降 | 重负载下降 | 功能说明 |
|------|-----------|-----------|----------|
| **Task Fusion (TF)** | ↓36.1% | ↓6.2% | 空间批处理提升利用率 |
| **Operator Orchestration (OO)** | ↓30.3% | ↓25.1% | 通信-计算重叠减少停顿 |
| **Chunk-based Alignment (CA)** | ↓22.5% | ↓34.3% | 减少无效 padding，尤其利于长序列 |

> ✅ 结果表明：轻负载时利用率是瓶颈；重负载时数据对齐更为关键。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **PEFT 工作负载存在显著资源浪费**：小批量、短序列、冻结骨干导致 GPU 利用率远低于预训练阶段（MFU 下降达 1.47×）。
2. **单纯的空间或时间复用均不充分**：
   - 空间批处理加剧通信停顿；
   - 时间交错进一步降低本已不足的 GPU 利用率。
3. **空间-时间联合复用是最优路径**：MuxTune 通过 **hTask 抽象** 实现两者的有机融合，在提升利用率的同时有效隐藏停顿。
4. **模块化设计是实现动态调度的基础**：解耦 Adapter 与骨干，使得任务可动态加入/退出，无需重新初始化模型。
5. **chunk-based 数据对齐至关重要**：相比简单 padding 或连续 packing，chunk 分片能兼顾效率与模型质量。

### 方法的局限性
- **依赖服务端完全控制权**：假设服务商掌握模型细节并可修改执行流程，不适合完全去中心化的场景。
- **对极端异构任务适应性有限**：若任务间 batch size 或 sequence length 差异极大，可能影响调度最优性。
- **目前主要针对 LoRA 类 PEFT**：虽支持多种 PEFT 类型，但 Adapter Fusion 策略对某些复杂结构（如 Prefix-Tuning）需额外适配。

### 未来工作方向
- **支持更多调度策略集成**：如预算感知（budget-aware）、SLO 感知调度，构建“复用感知”的集群调度器。
- **拓展至其他性能目标优化**：如能耗效率（energy efficiency）、成本效益（cost-per-token）等。
- **探索自动化的 chunk size 选择机制**：基于运行时反馈动态调整分块大小。
- **支持更复杂的多模态或多任务联合微调场景**：如 Context-PEFT 或 AdapterFusion 架构。

---

> 🔗 **开源地址**：https://github.com/sjtu-epcc/muxtune  
> 📌 **一句话总结**：MuxTune 通过空间-时间维度的骨干复用与层次化协同调度，首次实现了高吞吐、低显存的多任务 PEFT 微调系统，在真实数据中心场景下展现出巨大潜力。

</details>

---

### 11. [Generalized Discrete Diffusion with Self-Correction](https://arxiv.org/abs/2603.02230)

**Authors**: Linxuan Wang, Ziyi Wang, Yikun Bai, Wei Deng, Guang Lin, Qifan Song  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.02230v1  

#### Abstract
Self-correction is an effective technique for maintaining parallel sampling in discrete diffusion models with minimal performance degradation. Prior work has explored self-correction at inference time or during post-training; however, such approaches often suffer from limited generalization and may ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Generalized Discrete Diffusion with Self-Correction》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前主流的 **Masked Diffusion Language Models (MDLMs)** 虽然支持并行生成以降低推理延迟，但在每步解码多个 token 时容易破坏 token 间的依赖关系，导致生成质量下降，尤其是在复杂推理任务中表现不佳。尽管已有如 **remasking** 和 **post-training self-correction** 等技术用于修正错误 token，但这些方法通常泛化能力有限，且在推理阶段引入额外的启发式采样器或超参数调优，增加了工程复杂度。

此外，**GIDD (Generalized Interpolated Discrete Diffusion)** 虽通过预训练实现 self-correction，但其基于连续插值的 pipeline 导致 uniform transitions 与 absorbing masks 之间的交互不透明，难以调节噪声调度，影响实际性能。

### **提出的新方法与新思路**
本文提出了 **Self-Correcting Discrete Diffusion (SCDD)** 模型，旨在通过以下方式重构预训练阶段的 self-correction 机制：

- **显式的离散状态转移建模**：在离散时间框架下定义清晰的状态转移过程，避免 GIDD 中连续插值带来的模糊性。
- **统一使用 uniform transitions**：仅依赖 uniform noise 学习 self-correction，简化模型设计。
- **消除冗余的 remasking 步骤**：由于 forward process 中 `[MASK]` 是吸收态（absorbing state），backward 过程中不会出现从非 `[MASK]` 到 `[MASK]` 的“再掩码”操作，从而实现直接 token-to-token 自我修正。
- **SNR-informed 噪声调度**：引入信号-噪声比（Signal-to-Noise Ratio, SNR）控制 `pt`（uniform transition SNR）和 `yt`（absorbing mask SNR），分别独立调控两种噪声速率。

### **相比现有方法的优势**
| 维度 | GIDD | SCDD |
|------|------|------|
| **状态转移清晰性** | 插值式，opaque interaction | 显式离散转移，decoupled noise channels |
| **remasking 操作** | 保留，需两步完成修正 | 完全消除，一步直接修正 |
| **训练损失形式** | 需动态加权缓解梯度爆炸 | 理论 ELBO 损失，无需重加权 |
| **超参数调优** | 复杂，依赖权重函数 | 极简，无 post-hoc sampler 或调参 |
| **并行效率** | 较低（因 remasking 占用步骤） | 更高（全部步骤可用于纠错） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LM1B**（One Billion Words Dataset）
- **OWT**（OpenWebText）
- **Wikitext-103**（用于消融实验）

所有模型均采用 **GPT-2 tokenizer**。

### **模型架构与训练设置**
- 主干网络：**DiT (Diffusion transformer)** 的 `small` 版本（166M 参数）
- 时间步数：`T=1000` 离散时间步
- 上下文长度：
  - LM1B: 128
  - OWT/Wikitext-103: 512
- 批大小与训练步数：
  - LM1B: batch 512, 500K steps (~33B tokens)
  - OWT: batch 256, 1M steps (~131B tokens)
  - Wikitext-103: batch 128, 100K steps (~7B tokens)
- 优化器：**AdamW**，学习率 warm-up 后 cosine decay
- 精度：bfloat16 混合精度训练

### **评估指标**
- **Validation Perplexity (Val PPL)**：衡量语言建模拟合能力
- **Generative Perplexity (Gen PPL)**：由 GPT2-large 对生成文本打分，反映生成质量
- **Unigram Entropy**：验证生成多样性，防止重复
- **Correction Rate**：纠正 token 数 / (context length × step)，衡量 self-correction 强度
- **Zero-shot Accuracy**：在 ARC-E/C, BoolQ, HellaSwag 等常识推理任务上的表现

### **基线方法对比**
- **MDLM**（标准 masked diffusion）
- **GIDD+**（pu=0.1, 0.2）：重新训练版本
- **ReMDM-cap**, **ReMDM-conf**：基于 confidence/remasking 的 inference-time self-correction 方法

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **表1：验证困惑度（Val PPL）**
| Model | LM1B | OWT |
|-------|------|-----|
| MDLM* (reported) | 27.04 | 23.21 |
| MDLM (reproduced) | 32.98 | 24.72 |
| GIDD+ (pu=0.1) | 39.98 | 31.54 |
| GIDD+ (pu=0.2) | 40.70 | 32.19 |
| **SCDD (ours, pu=0.1)** | **39.16** | **28.41** ↓9.9% |
| **SCDD (ours, pu=0.2)** | 46.54 | 32.49 |

> 尽管加入 uniform noise 会轻微增加 Val PPL，但 SCDD 在 OWT 上相较 GIDD+ 实现了 **9.9% 的显著下降**，说明其更高效地利用了 uniform noise 来增强 self-correction。

---

#### ✅ **表2：生成困惑度（Gen PPL）——核心优势体现**
| Model \ Steps | 32 | 64 | 128 | 256 | 512 | 1024 |
|---------------|----|----|-----|-----|-----|------|
| **GIDD+ (pu=0.2), OWT** | 79.0 | 75.1 | 73.2 | 72.0 | 71.2 | 71.2 |
| **SCDD (ours, pu=0.2), OWT** | **67.1** ↓15.1% | **60.7** | **59.6** | **58.2** | **58.2** | **55.7** |

> 在 **32步生成** 场景下，SCDD 相比 GIDD+ 下降 **15.1% Gen PPL**，表明其在 few-step 并行生成中具有更强的纠错能力和更高的生成质量。

---

#### ✅ **表3：Correction Rate 对比**
| Model (pu=0.2) | 64 | 128 | 256 | 512 | 1024 |
|----------------|-----|-----|-----|-----|-------|
| GIDD+ | 0.39 | 0.40 | 0.40 | 0.40 | 0.40 |
| **SCDD (ours)** | **0.69** | **0.71** | **0.72** | **0.73** | **0.75** ↑87.5% |

> SCDD 不仅总修正率更高，且随步数增长持续提升，而 GIDD+ 几乎饱和。这证明 SCDD 能更有效地利用更多 denoising steps 进行精细化修正。

---

#### ✅ **消融实验结果**

##### （1）**不同 uniform noise ratio (pu) 的影响**
- 随着 `pu` 增大（0.05 → 0.2），**Correction Rate per Step** 显著上升。
- 在 few-step 设置下，self-correction 更活跃，说明 high-noise SCDD 更适合快速生成。

##### （2）**peak noise 时间 (`t_peak`) 的影响**
- 若最大 uniform noise 出现在后期（如 `t_peak=0.75`），模型倾向于在早期进行大量 self-correction。
- 若出现在前期（`t_peak=0.25`），则修正行为被推迟至最后几步。
> 表明可通过设计 noise schedule 控制 self-correction 的动态分布。

---

## **4. 关键结论和发现**

### **主要发现**
1. **SCDD 实现了真正意义上的免 remasking 自我修正**：通过将 `[MASK]` 设为吸收态，backward 过程中不再需要“先掩码再解码”的两步流程，**纠错效率理论上可翻倍**。
2. **预训练阶段集成 self-correction 具有更强泛化性**：相较于 inference-time 方法（如 ReMDM、PRISM），SCDD 在训练中就学会如何修正错误，因此在各种生成步数下都保持稳定高性能。
3. **few-step 生成场景下优势最明显**：当仅允许少量 denoising steps 时，SCDD 凭借高效的并行 self-correction 显著优于所有 baseline，适用于低延迟应用场景。
4. **噪声调度可调控 self-correction 动力学**：通过调整 `pu` 和 `t_peak`，可以灵活控制模型何时、多强地进行自我修正。

### **方法的局限性**
- 当前实验规模仍为 **GPT-2 small** 级别（166M），尚未扩展到 billion-scale 模型，实际部署潜力有待验证。
- 加入 uniform noise 导致 Val PPL 上升，在纯语言建模任务中可能不如传统 MDLM。
- zero-shot benchmark 表现未超越 baseline，说明 self-correction 更体现在生成流畅性和一致性，而非直接提升选择题准确率。

### **未来工作方向**
1. **Scaling to billion-parameter models**：探索更大规模下的 self-correction 泛化能力。
2. **结合强化学习进一步优化 self-correction**：例如使用 RL fine-tuning 最大化生成质量和逻辑连贯性。
3. **探索 adaptive noise scheduling**：根据输入难度动态调整 `pt`, `yt`，实现智能纠错。
4. **应用于 code generation、math reasoning 等高阶任务**：测试其在结构化输出中的纠错潜力。

---

> **总结一句话**：  
> SCDD 通过 **清晰的离散状态转移 + 吸收态设计 + 独立 SNR 控制**，实现了 **无需 remasking 的高效 self-correction**，在 few-step 并行生成中显著优于 GIDD 和 ReMDM，是迈向高效、高质量扩散语言模型的重要一步。

</details>

---

### 12. [Optimizing In-Context Demonstrations for LLM-based Automated Grading](https://arxiv.org/abs/2603.00465)

**Authors**: Yucheng Chu, Hang Li, Kaiqi Yang, Yasemin Copur-Gencturk, Kevin Haudek, Joseph Krajcik, Jiliang Tang  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.00465v1  

#### Abstract
Automated assessment of open-ended student responses is a critical capability for scaling personalized feedback in education. While large language models (LLMs) have shown promise in grading tasks via in-context learning (ICL), their reliability is heavily dependent on the selection of few-shot exem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimizing In-Context Demonstrations for LLM-based Automated Grading

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Models (LLMs)** 的自动评分系统依赖于 **In-Context Learning (ICL)**，其性能高度敏感于所选的 few-shot 示例（exemplars）和 rationale 质量。然而：
- 传统检索方法（如基于语义相似度的 KNN）倾向于选择表面相似但未必能体现评分边界（rubric boundaries）的示例。
- 手动编写高质量的 **rationale** 成本高昂，难以规模化。

这些问题导致模型在“边缘案例”（borderline cases）上表现不佳——即那些语义相近但应得不同分数的学生回答，从而影响评分的可靠性和教育有效性。

---

### 🚀 提出的新方法：GUIDE
作者提出 **GUIDE**（Grading Using Iteratively Designed Exemplars），一个将 exemplar 优化重构为“决策边界聚焦”的迭代框架，核心创新包括：

#### （1）**Boundary-Focused Exemplar Selection**
- 引入 **contrastive operators** 主动识别“边界对”（boundary pairs）：即语义高度相似但真实标签不同的学生回答。
- 在 **Constrained Bayesian Optimization** 框架下，优先选择这些边界对作为上下文示例，迫使模型关注细微差异。

#### （2）**Discriminative Rationale Generation**
- 自动生成具有区分性的 rationale，明确解释为何某回答得分为 *y* 而非相邻等级（如不是 0 或 2）。
- 使用 **contrastive infill** 技术，在 prompt 中强制要求模型对比邻近等级，生成更精准的推理链。

#### （3）**Iterative Refinement Loop**
- GUIDE 采用两阶段循环：
  - **Selection Phase**：通过贝叶斯优化选出最具判别力的 exemplar 子集。
  - **Generation Phase**：利用最优子集重新生成更高质量的 rationale，更新候选池用于下一轮优化。

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | GUIDE 的改进 |
|------|--------|-------------|
| **Random / Naive** | 随机或固定选择示例，无法捕捉边界 | 显式优化边界覆盖 |
| **KNN-SBERT** | 基于语义相似度，易混淆正确与错误推理 | 加入 contrastive density 目标函数 |
| **BRIDGE** | 全局准确率驱动，忽略局部边界细节 | 明确建模 boundary pairs + discriminative rationale |

> ✅ GUIDE 不仅告诉模型“典型例子长什么样”，更教会它“评分边界的精确位置在哪里”。

---

## 2. 核心实验方法和设置

### 📚 数据集
实验涵盖三个教育领域的真实数据集，覆盖科学教育与教师教育：

| 数据集 | 领域 | 样本数 | 分数等级 | 是否含专家 rationale |
|-------|------|--------|----------|------------------|
| **Dr** | 物理（电相互作用） | 314 | {0,1}（二元） | 否 |
| **Dc** | 化学（3DLP 框架） | ~163–184 | {0,1,2}（有序） | 是（每类 3 条） |
| **DT** | 教师教育（数学教学知识） | ~229–236 | {0,1,2}（有序） | 是（每类 3–5 条） |

所有数据集划分为 train:valid:test = 3:1:1。

---

### ⚙️ 实验设置
- **Backbone Model**: GPT-4o-mini
- **Embedding Model**: text-embedding-3-small（用于计算语义相似度）
- **Optimization Rounds**: T = 5
- **Demonstration Set Size**: [4, 16]
- **Semantic Threshold for Boundary Pairs**: sim ≥ 0.7
- **Max Pool Size**: 512
- **Temperature**: 0.2（保证生成稳定性）

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 完全匹配预测得分的比例 |
| **Quadratic Weighted Kappa (QWK)** | 衡量序数一致性，惩罚跨级错误（如 0→2） |
| **Adjacent Error Rate (AdjErr)** | 错误仅相差一级的比例（如 1↔2），反映边界模糊 |
| **Non-Adjacent Error Rate (NonAdjErr)** | 跨级错误比例（如 0→2），表示严重逻辑失败 |

---

### 🆚 基线方法对比
| 基线 | 类型 | 简介 |
|------|------|------|
| **Naive** | 固定 | 使用原始提供的 few-shot 示例 |
| **Random** | 固定 | 随机采样 k 个示例 |
| **KNN-SBERT** | 动态 | 对每个查询检索最相似的 top-k 示例 |
| **Vote-K** | 固定 | 最大化多样性，确保语义空间广覆盖 |
| **BRIDGE** | 优化 | 迭代优化示例集以最大化验证集准确率（无边界意识） |

---

## 3. 主要实验结果和性能指标

### 📈 总体性能（见 Table 2）

| 方法 | Dr (Acc/QWK) | Dc (Acc/QWK) | DT (Acc/QWK) |
|------|--------------|--------------|--------------|
| Naive | 0.74 / 0.42 | 0.69 / 0.39 | 0.59 / 0.54 |
| Random | 0.75 / 0.43 | 0.58 / 0.32 | 0.52 / 0.52 |
| KNN-SBERT | 0.78 / 0.44 | 0.58 / 0.26 | 0.52 / 0.52 |
| BRIDGE | 0.90 / 0.57 | 0.76 / 0.53 | 0.66 / 0.65 |
| **GUIDE (Ours)** | **0.92 / 0.62** | **0.80 / 0.59** | **0.71 / 0.67** |

> ✅ GUIDE 在所有数据集上均取得 **最高 Accuracy 和 QWK**，尤其在复杂任务 DT 上相对提升约 **20%**。

---

### 🔍 边界错误分析（AdjErr）

| 方法 | Dr (AdjErr) | Dc (AdjErr) | DT (AdjErr) |
|------|-------------|-------------|-------------|
| Naive | 0.26 | 0.31 | 0.38 |
| BRIDGE | 0.19 | 0.24 | 0.32 |
| **GUIDE** | **0.08** | **0.20** | **0.28** |

> ✅ GUIDE 将 **相邻错误率显著降低**，表明其有效提升了模型对评分边界的分辨能力。

同时，**NonAdjErr 几乎为零**，说明 GUIDE 并未牺牲全局逻辑来换取边界精度。

---

### 💡 消融实验（Ablation Insights，文中隐含）
虽然未提供显式消融表，但从设计可推断以下关键组件的作用：

| 组件 | 作用 | 支持证据 |
|------|------|---------|
| **Contrastive Operators** | 提升边界对覆盖率 | GUIDE 在 AdjErr 上大幅优于 BRIDGE |
| **Discriminative Rationale** | 强化 rubric adherence | rationale 明确提及“缺少什么”、“阻止升级的因素”等 |
| **Iterative Refinement** | 持续提升 rationale 质量 | 多轮优化后性能持续上升 |

> 图 2 和图 3 展示了 GUIDE 生成的 rationale 更具教学意义，能清晰指出学生回答中的具体缺失或优势。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **边界案例是自动化评分的关键挑战**  
   多数错误发生在语义相近但评分不同的边缘情况，而非极端错误。

2. **标准 retrieval 方法不足以支持 rubric adherence**  
   单纯基于语义相似度的选择无法帮助模型理解“为什么不能给更高分”。

3. **GUIDE 显著提升边界判别能力**  
   通过 contrastive selection 与 discriminative rationale，模型学会关注评分标准中的细微差别。

4. **小规模高质量上下文即可实现高性能**  
   仅需 4–16 个精心挑选的示例，即可超越大规模随机或静态示例集。

5. **减少人工标注负担**  
   自动合成高质量 rationale，使系统可在仅有少量专家标注的情况下启动（cold-start 友好）。

---

### ⚠️ 局限性
- 当前方法主要适用于文本型 open-ended responses，尚未扩展至 **multimodal grading**（如图表、公式推导）。
- 依赖 LLM 的生成质量，若 backbone model 偏差较大，可能放大错误 rationale。
- 贝叶斯优化过程有一定计算开销（尽管使用廉价模型控制成本）。

---

### 🔮 未来工作方向
1. 扩展到 **multimodal tasks**（如结合 diagram interpretation 的科学题评分）。
2. 探索 **动态 context construction**，根据不同 query 类型调整边界策略。
3. 引入 human-in-the-loop 机制，允许教师干预和修正自动生成的 rationale。
4. 应用于更多学科（如写作、历史论述）验证泛化能力。

---

## ✅ 总结
**GUIDE** 是首个将 ICL 中的 exemplar selection 明确建模为“边界定义问题”的自动化评分框架。它通过 **contrastive operators** 和 **discriminative rationale generation** 的双轮驱动，显著提升了 LLM 在边缘案例上的评分鲁棒性与 pedagogical validity。实验证明其在多个教育数据集上优于主流基线，尤其在降低 adjacent error 方面表现突出，为构建可信、可扩展的 AI 教育评估系统提供了新范式。

</details>

---

### 13. [Agents Learn Their Runtime: Interpreter Persistence as Training-Time Semantics](https://arxiv.org/abs/2603.01209)

**Authors**: Victor May, Aaditya Salgarkar, Yishan Wang, Diganta Misra, Huu Nguyen  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.01209v1  

#### Abstract
Tool-augmented LLMs are increasingly deployed as agents that interleave natural-language reasoning with executable Python actions, as in CodeAct-style frameworks. In deployment, these agents rely on runtime state that persists across steps. By contrast, common training pipelines treat agent traces a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Agents Learn Their Runtime: Interpreter Persistence as Training-Time Semantics

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文探讨了一个在 **Tool-Augmented LLM Agents** 领域中被忽视的关键问题：**Interpreter Persistence（解释器持久性）** 是仅仅作为推理时（runtime）的执行框架，还是在训练阶段（training-time）就应被视为一种语义信号？  
当前许多 agent 框架在训练时使用某种运行时语义（如持久化解释器），但在部署时可能切换到不同的运行时（如无状态解释器），这种不匹配可能导致 agent 行为不稳定或效率低下。

### 提出了什么新方法或新思路
- **提出“执行语义”（Execution Semantics）是可学习的训练信号**：作者认为，解释器是否保持状态（persistent vs. stateless）不应只是运行时实现细节，而是一种应在训练数据中明确建模的 **first-class semantic**。
- **设计了 OPAQUE KNAPSACK 基准任务**：一个非坍缩（non-collapsible）、部分可观测的背包问题变体，强制 agent 必须进行多轮交互、状态维护和计划修订，从而对执行语义敏感。
- **采用 2×2 控制实验设计**：系统性地交叉训练语义（persistent vs. stateless traces）和运行时语义（persistent vs. stateless runtime），以分离两者的影响。

### 相比现有方法的优势
- 揭示了 **训练-推理对齐（Training-Inference Alignment）** 在执行语义层面的重要性，超越了传统关注工具调用或推理格式的对齐。
- 提供了实证证据表明，**解释器持久性是一种可学习的行为先验（learnable behavioral prior）**，而非零样本能力。
- 强调了 agent 数据集设计中应显式标注执行语义，避免隐含假设导致部署失败。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **OPAQUE KNAPSACK**：作者提出的合成任务，基于经典 0/1 背包问题，具有以下特性：
  - **隐藏属性**：物品的重量、价值、类别需通过 `inspect()` 工具查询。
  - **预算限制**：`inspect()` 调用次数有限，防止穷举。
  - **隐藏约束**：仅特定类别的物品允许放入背包，需通过反馈推断。
  - **部分可观测性**：agent 无法一次性获取全部信息，必须迭代决策。
- 包含两个难度级别：
  - **Easy**：用于训练和同域评估（25–40 items）
  - **Hard**：用于跨难度泛化评估（80–120 items）

### 实验设置和评估指标
#### 实验设计（2×2 Cross-Evaluation）
| 训练语义 \ 运行时语义 | Persistent Runtime | Stateless Runtime |
|------------------------|--------------------|-------------------|
| **Persistent Traces**  | ✅ 对齐            | ❌ 失配           |
| **Stateless Traces**   | ❌ 失配            | ✅ 对齐           |

- 所有模型均为 **Qwen3-8B**，通过 **LoRA** 微调。
- 使用 **Gemini 3 Flash** 作为教师模型生成训练轨迹。

#### 评估指标
- **Normalized Optimality (%)**：agent 达成的价值占最优解的百分比。
- **Exact Solves**：达到最优解的任务数量。
- **Interaction Footprint**：
  - Steps（步数）
  - Total Tokens（总生成 token 数）
  - Wall-clock Time（耗时）
- **Score / 1k Tokens**：单位推理成本下的得分效率。
- **行为级诊断指标**（Behavioral Metrics）：
  - **State Utilization**：跨回合变量复用次数
  - **Imports/Step**：每步导入次数
  - **Context Symbol Lifespan**：变量名在上下文中持续出现的回合数
  - **Unresolved Reference Errors**：因变量未定义引发的异常

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| Train → Runtime | Normalized Optimality (%) | Total Tokens (Hard) | Score / 1k Tokens |
|------------------|----------------------------|----------------------|-------------------|
| Persistent → Persistent | **75.4 ± 4.7** | **18,612** | **4.05** |
| Stateless → Persistent | 72.5 ± 4.3 | 54,665 | 1.33 |
| Persistent → Stateless | 68.2 ± 6.9 | 67,925 | 1.00 |
| Stateless → Stateless | 67.7 ± 6.3 | 67,898 | 1.00 |

> 注：所有 fine-tuned 模型显著优于 base model（~2–7%），但不同 fine-tuning 条件间的 optimality 差异 **无统计显著性**（n=100）。

### 与基线方法的对比结果
- **Persistent → Persistent** 是最高效的组合：
  - 在 Hard 任务上仅消耗约 **18.6k tokens**，而其他条件需 **54k–68k tokens**。
  - 效率高出 **3.5×** 以上（Score / 1k Tokens: 4.05 vs. ≤1.33）。
- **Stateless-trained 模型即使在 Persistent Runtime 中仍支付“失忆税”（amnesia tax）**：
  - 继续重复导入和重建状态（Imports/Step = 1.00），未能利用解释器状态。
- **Persistent-trained 模型在 Stateless Runtime 中崩溃**：
  - **~80% 的 episode 出现 `NameError`**（如 `inspect_data` 未定义）。
  - 导致级联恢复循环（cascading recovery loops），token 预算耗尽且无进展。

### 消融实验结果
- **Token Volume 不等于 Learnability**：
  - Stateless 训练数据更长（平均 55k vs. 18k tokens/episode），但并未带来更强的学习效果。
  - 表明冗余的状态重建降低了数据质量，**执行合同影响学习效率**。
- **Few-shot Demonstrations 无法纠正失配**：
  - 即使在评估时向 Persistent-trained 模型展示 Stateless 的演示，它仍会尝试引用已丢失的变量，说明依赖是**学到的行为策略**，而非提示驱动。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **Interpreter Persistence 是一种可学习的训练时语义**，而非推理时的零样本能力。
2. ✅ **训练-推理执行语义必须对齐**：
   - 对齐时（Persistent → Persistent）：agent 高效复用解释器状态，节省 3.5×+ token。
   - 失配时：
     - Persistent → Stateless：触发大量 `NameError` 和恢复循环，稳定性差。
     - Stateless → Persistent：继续冗余重建状态，效率低下（“amnesia tax”）。
3. ✅ **执行语义塑造了 agent 的状态管理策略**：
   - Persistent-trained agent 学会将状态委托给解释器。
   - Stateless-trained agent 学会将状态外化到文本上下文中。
4. ✅ **性能差异主要体现在效率和稳定性，而非最终求解能力**：
   - 所有微调模型的 **normalized optimality 无显著差异**，但推理成本差异巨大。

### 方法的局限性
- **样本量限制**：n=100 个任务不足以检测 optimality 上的小幅差异。
- **Token-Budget Confound**：Stateless 训练集包含更多 token，未来需控制 token 总量进行消融。
- **单一任务与模型**：仅在 OPAQUE KNAPSACK 和 Qwen3-8B 上验证，泛化性待检验。
- **共现协议线索**：运行时提供了 `active_globals` 元数据，其作用未完全剥离。

### 未来工作方向
- 验证该现象在 **多种任务、模型规模和运行时设计**（如介于 persistent 与 stateless 之间的混合模式）中的普适性。
- 探索 **显式建模执行语义的训练目标**，例如通过提示或损失函数引导状态管理策略。
- 构建支持执行语义标注的 **标准化 Agent Data Protocol**，促进数据集互操作性。
- 研究如何在运行时动态适应不同执行语义，提升 agent 的鲁棒性。

> **一句话总结**：  
> Agent 不仅学会“做什么”，还学会“在哪里做”——解释器是否持久化，必须作为训练数据的一等公民来对待，否则将付出效率或稳定性的代价。

</details>

---

### 14. [Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization](https://arxiv.org/abs/2603.02701)

**Authors**: Yueyang Cang, Xiaoteng Zhang, Erlu Zhao, Zehua Ji, Yuhang Liu, Yuchen He, Zhiyuan Ning, Chen Yijun, Wenge Que, Li Shi  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.02701v1  

#### Abstract
Optimizing communication topology is fundamental to the efficiency and effectiveness of Large Language Model (LLM)-based Multi-Agent Systems (MAS). While recent approaches utilize reinforcement learning to dynamically construct task-specific graphs, they typically rely on single-sample policy gradie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在基于 **Large Language Model (LLM)** 的 **Multi-Agent Systems (MAS)** 中，通信拓扑（communication topology）对系统性能至关重要。然而，现有的拓扑优化方法通常依赖于标准的 **Reinforcement Learning (RL)** 范式（如 REINFORCE），存在两个核心问题：

- **High Gradient Variance**：由于任务难度不均，简单任务中即使次优拓扑也可能获得正奖励（reward=1），导致冗余边被错误强化；而困难任务常失败（reward=0），梯度消失。
- **Credit Assignment Problem**：成功时所有边平等地获得奖励，无法区分关键连接与冗余连接，阻碍模型学习精确结构模式。

### 提出了什么新方法或新思路
作者提出 **Graph-GRPO**（Graph-based Group Relative Policy Optimization），首次将 **Group Relative Policy Optimization (GRPO)** 引入离散图结构搜索领域。其核心思想是：

- 对每个查询采样一组多样化的通信拓扑（group sampling）
- 不再使用绝对奖励，而是计算每条边的**相对优势**（relative advantage），即该边出现在高表现拓扑中的频率相对于组内平均水平的表现
- 利用组内平均性能作为动态基线（baseline），实现奖励归一化

### 相比现有方法的优势
- **稳定性更强**：通过组内归一化有效缓解任务难度带来的奖励噪声，提升训练稳定性
- **细粒度信用分配**（fine-grained credit assignment）：能识别并强化真正促进成功的“关键通信路径”，抑制冗余边
- **无需 Critic 网络**：相比 PPO 类方法，避免了额外的值网络开销，降低内存占用和训练不稳定性
- **自然收敛到稀疏高效结构**：在不显式剪枝的情况下自动抑制低价值连接，提高“Signal-to-Token Ratio”

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖六个基准，在三个领域进行评估：
- **通用推理**：MMLU（multi-task knowledge）
- **数学推理**：GSM8K、MultiArith、SVAMP、AQUA
- **代码生成**：HumanEval

遵循 EIB-LEARNER 的标准协议，确保可比性。

### 实验设置和评估指标
- **Backbone LLM**：GPT-3.5-Turbo
- **Policy Network**：基于 G-Designer 架构，采用 all-MiniLM-L6-v2 编码器 + 3层 GAT
- **Agent 数量**：MMLU 设为6，HumanEval 为5，数学任务为4
- **Group Size K**：16
- **最大通信轮数**：3
- **优化器**：Adam，学习率 1e-4，NVIDIA A100 GPU
- **评估指标**：准确率（Accuracy %）、token 消耗量（衡量效率）

### 基线方法对比
分为三类：
1. **Single-Agent Methods**：
   - Chain-of-Thought (CoT)
   - Self-Consistency (SC)
2. **Fixed Topologies**：
   - Chain, Tree, Complete Graph, LLM-Debate
3. **Topology Optimization Methods**（主要对比对象）：
   - AgentPrune
   - AgentDropout
   - G-Designer
   - EIB-LEARNER（当前 SOTA）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| Method | MMLU | GSM8K | AQuA | MultiArith | SVAMP | HumanEval | **Avg.** |
|--------|------|-------|------|------------|--------|-----------|---------|
| EIB-LEARNER | 88.90 | 95.20 | 83.49 | 96.83 | 94.70 | 89.15 | **91.38** |
| **Graph-GRPO** | **90.12** | **96.10** | **84.21** | **97.07** | **96.01** | **91.25** | **92.45** |

- **平均准确率提升 +1.07%**，在所有六项任务上均超越 SOTA
- 在复杂任务上增益更显著：
  - **GSM8K**：+0.9%
  - **HumanEval**：+2.1%

### 与基线方法的对比结果
- 显著优于所有固定拓扑结构（如 Chain、Tree、Complete Graph），后者平均性能约 84%，受限于缺乏自适应能力
- 相比其他动态拓扑方法（如 G-Designer、AgentPrune），Graph-GRPO 进一步提升了精度，且训练更稳定
- 在 token 效率方面也表现出色（见下图分析）

### 消融实验结果（Ablation Study）
设计了 **Graph-Level GRPO** 变体（将整个图的成功与否作为所有边的共同优势），用于验证边缘级建模的重要性：

| Method | MMLU | GSM8K | HumanEval | Avg. |
|--------|------|-------|-----------|------|
| Graph-GRPO | 90.12 | 96.10 | 91.25 | **92.49** |
| Graph-Level GRPO | 88.54 | 94.40 | 89.07 | 90.67 |
| Δ | -1.58 | -1.70 | -2.18 | **-1.82** |

- 所有任务均有下降，尤其在需要逻辑链精确性的 **HumanEval** 上降幅最大（-2.18%）
- 表明粗粒度的图级奖励会强化“搭便车”边（freeloader edges），导致结构噪声累积
- 验证了 **edge-level advantage estimation** 是性能提升的关键

---

## 4. 关键结论和发现

### 论文的主要发现
- 绝对奖励机制在 MAS 拓扑学习中存在严重缺陷，尤其是面对任务难度方差时
- **Group Relative Policy Optimization** 是一种有效的去噪机制，能够稳定训练过程
- 通过边缘级别的相对优势估计，实现了前所未有的细粒度信用分配，揭示了以往被奖励噪声掩盖的关键通信路径
- Graph-GRPO 自然收敛到**稀疏但语义丰富**的拓扑结构，在保持高性能的同时显著降低 token 开销，实现帕累托最优（Pareto-optimal）的性能-效率权衡

### 方法的局限性
1. **可扩展性限制**：
   - 当前策略网络基于 GAT，时间复杂度为 $O(N^2)$，适用于小规模代理群（$N \leq 6$）
   - 难以直接应用于大规模 swarm（如 $N > 100$），需引入分层或稀疏生成策略
2. **动态适应性不足**：
   - 当前框架为每个查询生成一个静态拓扑
   - 对于多轮对话场景，最优通信结构可能随回合变化，缺乏细粒度的 turn-level 动态调整机制

### 未来工作方向
- 将 Graph-GRPO 扩展至更大规模、异构（heterogeneous）的 agent 系统
- 探索支持动态演化的 topology generation 机制，适应 multi-turn 协作场景
- 结合更多因果推理技术，进一步增强对“关键路径”的解释性和可控性
- 推动构建**无需 Critic 的轻量级、自组织 agent swarms**

--- 

> ✅ 总结一句话：  
> **Graph-GRPO 通过引入 group-relative 机制，从根本上解决了多智能体系统中拓扑学习的梯度噪声与信用分配难题，在精度、稳定性与效率上全面超越现有方法，为下一代自组织 LLM-Agent 协同提供了新范式。**

</details>

---

### 15. [ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](https://arxiv.org/abs/2603.02216)

**Authors**: Ruike Cao, Shaojie Bai, Fugen Yao, Liang Dong, Jian Xu, Li Xiao  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.02216v1  

#### Abstract
Effective information seeking in multi-turn medical dialogues is critical for accurate diagnosis, especially when dealing with incomplete information. Aligning Large Language Models (LLMs) for these interactive scenarios is challenging due to the uncertainty inherent in user-agent interactions, whic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue

## 1. 论文的主要贡献和创新点

### 解决的问题
当前在医疗领域的 **Large Language Models (LLMs)** 多数基于单轮交互进行训练和评估，而真实世界的医疗对话通常是多轮的，用户初始提供的信息往往不完整。因此，模型需要具备主动提问以获取更多信息的能力。然而，现有的 LLMs 在**动态、多轮信息收集**方面表现不足，导致诊断准确性受限。

此外，传统的强化学习（Reinforcement Learning, RL）方法如 **Group Relative Policy Optimization (GRPO)** 和 **Proximal Policy Optimization (PPO)** 在处理长周期信用分配（long-horizon credit assignment）和价值估计稳定性方面存在缺陷，难以有效优化复杂的多轮对话策略。

### 提出的新方法：ATPO
本文提出了一种新的不确定性感知算法——**Adaptive Tree Policy Optimization (ATPO)**，用于优化多轮医疗对话中的信息寻求行为。

#### 核心思想：
- 将多轮对话建模为 **Hierarchical Markov Decision Process (H-MDP)**，其中每个“宏观动作”对应一次完整的助手回复（一轮对话）。
- 引入一种**自适应树搜索机制**，根据状态的不确定性动态分配 rollout 预算。
- 不确定性由两个指标加权构成：
  - **Bellman Error (U₁)**：衡量 critic 当前值估计与一步前瞻目标之间的误差，反映环境随机性（aleatoric uncertainty）。
  - **Action-Value Variance (U₂)**：衡量不同候选动作下 Q 值的方差，反映策略不确定性和认知局限（epistemic uncertainty）。
- 对高不确定性的节点进行完全展开，低不确定性的节点则通过剪枝减少计算开销。

### 相比现有方法的优势
| 方法 | 缺陷 | ATPO 的改进 |
|------|------|-------------|
| **Prompt Engineering** | 无法根本提升多轮能力，甚至降低性能 | 显式建模多轮决策过程 |
| **Supervised Fine-Tuning (SFT)** | 仅模仿训练数据，泛化能力弱 | 使用 RL 实现目标驱动的学习 |
| **GRPO** | 长周期信用分配困难 | 利用树结构实现更精确的 credit assignment |
| **PPO** | 价值估计不稳定 | 结合蒙特卡洛回溯与 critic 学习，提高估计准确性 |
| **TreePO** | 固定分支结构，探索效率低 | 自适应扩展，资源集中在高不确定性区域 |

此外，ATPO 还引入了两项关键技术优化计算效率：
- **不确定性引导的剪枝机制**：显著减少 rollout 数量。
- **异步搜索架构 + KV Cache 复用**：最大化推理吞吐量。

---

## 2. 核心实验方法和设置

### 使用的数据集
三个公开的多选题医学数据集被重构为支持多轮对话评估的形式：
| 数据集 | 来源 | 样本数 | 特点 |
|--------|------|-------|------|
| **MedicalExam** | [Liao et al., 2024] | 150 | 综合多个来源（MedQA, MedMCQA, MMLU 等），经 LLM 分解为原子事实 |
| **MedQA** | [Jin et al., 2020] | 1,268 | 来自 MEDIQ 测试集，原始即含丰富上下文 |
| **MedMCQA** | [Pal et al., 2022] | 536 | 从训练集中筛选长描述样本，并由 LLM 合成原子事实 |

所有样本均被格式化为：`初始上下文 + 原子问题 + 原子事实列表 + 多个选项`

### 实验设置
- **模型规模**：使用 Qwen3 系列三种尺寸模型作为主体：
  - Qwen3-1.7B
  - Qwen3-4B
  - Qwen3-8B
- **环境模拟**：构建双智能体系统
  - **User Simulator**：基于 Qwen3-8B 构建，严格依据原子事实回答问题，若无相关信息则返回固定短语 `"The patient cannot answer this question."`
  - **Assistant Agent**：待训练的目标模型，需通过最多 8 轮交互完成诊断。
- **奖励函数**：
  - 正确答案：+3
  - 错误答案：0
  - 格式无效：-1
- **训练数据**：共 14,256 条，混合来自 MEDIQ 和 MedMCQA 的训练集。

### 评估指标
- **主指标**：**Final-answer Accuracy**（最终答案准确率）
- 所有结果报告五次独立运行的均值 ± 标准差
- 其他分析指标包括：
  - 收敛速度（训练轮次 vs 准确率）
  - 采样多样性（return variance）
  - critic 损失
  - 推理吞吐量（tokens/sec/GPU）

### 基线方法对比
| 类型 | 方法 | 描述 |
|------|------|------|
| **Zero-shot Prompting** | Direct / MEDIQ | 单轮直接作答 或 允许最多 8 轮交互 |
| **SFT** | SFT / DFT | 监督微调，鼓励多轮提问行为 |
| **SFT+RL** | PPO (MDP/H-MDP) | 分别按 token 级和 turn 级建模 |
| **SFT+RL** | GRPO | 无 critic 方法，整条轨迹共享优势 |
| **SFT+RL** | TreePO | 固定二叉树结构搜索，无剪枝 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（摘自 Table 1）

| Model | Method | MedicalExam (%) | MedQA (%) | MedMCQA (%) |
|-------|--------|------------------|-----------|-------------|
| Qwen3-8B | ATPO(U1+U2) | **65.87±3.72** | **64.07±0.43** | **53.66±1.52** |
| GPT-4o | MEDIQ | 64.00±3.53 | 63.15±0.82 | 53.03±0.89 |
| Gemini-2.5-Pro | MEDIQ | 74.33±2.53 | 68.69±0.61 | 63.31±1.37 |

> ✅ **关键突破**：**Qwen3-8B + ATPO 在 MedQA 上超越 GPT-4o 达到 +0.92% 的绝对增益**

### 与基线方法的对比结果
- 在所有模型规模和数据集上，**ATPO 均优于所有强基线**。
- 相较于最强 RL 基线 **TreePO**：
  - 在 MedQA 上分别带来 **+2.26% (8B)**、**+1.73% (4B)**、**+0.82% (1.7B)** 的提升。
- 相较于 **GRPO** 和 **PPO** 提升更为显著，尤其在大模型上优势明显。
- **MEDIQ 提示法反而低于 Direct 法**，验证了盲目提示可能损害性能。

### 消融实验结果
#### （1）不确定性组件有效性（ATPO(U1) vs ATPO(U1+U2)）
- **U1+U2 联合使用效果最佳**，说明 Bellman error 和 Q-value variance 是互补信号。
- 仅使用 U1 导致早期过度探索，深度覆盖不足；加入 U2 后探索更均衡、更深。

#### （2）访问计数降权（Visit Count Down-weighting）
- 移除该机制（EXP1）会导致熵爆炸和频繁策略裁剪（clipping），训练不稳定。
- 若同时对 value loss 也降权（EXP2），会破坏 critic 稳定性，导致策略退化为单轮响应。

#### （3）样本效率分析（Figure 2a）
- **ATPO(U1+U2) 仅用 TreePO 55% 的训练回合数即可达到同等精度**，展现出极高的样本效率。
- 收敛曲线更陡峭，表明学习更快。

#### （4）泛化能力测试（Appendix A.6）
- 将训练时使用的 Qwen3-8B 用户模拟器替换为 **Llama-3.3-70B-Instruct** 进行测试：
  - 性能几乎不变（见 Table 5），证明模型未过拟合特定模拟器风格，具有强泛化性。

#### （5）计算成本分析（Appendix A.5）
- 尽管 ATPO rollout 阶段耗时占比更高（约 45% vs 其他 ~25%），但由于其高质量数据加速收敛，**总训练时间最短**。
- 例如，达到相同性能水平，ATPO(U1+U2) 仅需 **2.22 小时**，而 GRPO 需 **4.86 小时**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **ATPO 显著提升了多轮医疗对话中 LLM 的信息寻求能力和诊断准确性**。
2. ✅ **结合 Bellman error 与 Q-value variance 的复合不确定性度量**，能有效指导高效且多样化的探索。
3. ✅ **自适应树扩展 + 剪枝机制** 实现了高效率的价值估计与策略更新。
4. ✅ **ATPO 可使较小模型（如 Qwen3-8B）超越更大模型（如 GPT-4o）的表现**，展示了算法设计的巨大潜力。
5. ✅ 方法具备良好的**泛化能力**，不依赖特定用户模拟器。

### 方法的局限性
- **超参数敏感性**：不确定性阈值 `t` 和权重 `α` 需手动调节，缺乏自适应机制。
- **固定分支数 N**：当前设定每个节点最多生成 N=4 个候选动作，未根据不确定性动态调整分支数量。
- **依赖高质量用户模拟器**：虽然已验证跨模拟器泛化性，但训练仍依赖于可靠的 environment simulation。
- **尚未应用于真实临床场景**：目前仅在合成数据上验证，实际部署前需进一步安全性验证。

### 未来工作方向
1. **可学习的扩展策略**：将当前固定的阈值判断替换为一个**可学习的 soft control policy**，实现动态、自适应的树扩展。
2. **精细化 credit assignment**：在 H-MDP 框架内，研究如何将高层优势更合理地分配给底层 token 动作，而非简单的均匀复制。
3. **动态分支控制**：根据不确定性程度自动决定扩展多少个子节点，而非固定 N 或随机选择。
4. **扩展至其他领域**：将 ATPO 应用于开放域多轮对话、工具调用（tool-use）、复杂任务规划等需要长期推理的场景。

---

> 🔚 **总结一句话**：  
> ATPO 通过引入**不确定性感知的自适应树搜索机制**，实现了高效、稳定且高性能的多轮医疗对话优化，在多个基准上超越主流 RL 方法，并首次让 Qwen3-8B 模型在 MedQA 上超过 GPT-4o，展现了算法创新的巨大潜力。

</details>

---

### 16. [DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](https://arxiv.org/abs/2603.01106)

**Authors**: Haowen Gao, Zhenyu Zhang, Liang Pang, Fangda Guo, Hongjian Dou, Guannan Lv, Shaoguo Liu, Tingting Gao, Huawei Shen, Xueqi Cheng  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01106v1  

#### Abstract
Reinforcement learning (RL) with group relative policy optimization (GRPO) has become a widely adopted approach for enhancing the reasoning capabilities of multimodal large language models (MLLMs). While GRPO enables long-chain reasoning without a critic, it often suffers from sparse rewards on diff...

---

### 17. [Harmonizing Dense and Sparse Signals in Multi-turn RL: Dual-Horizon Credit Assignment for Industrial Sales Agents](https://arxiv.org/abs/2603.01481)

**Authors**: Haojin Yang, Ai Jian, Xinyue Huang, Yiwei Wang, Weipeng Zhang, Ke Zeng, Xunliang Cai, Jingqing Ruan  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01481v1  

#### Abstract
Optimizing large language models for industrial sales requires balancing long-term commercial objectives (e.g., conversion rate) with immediate linguistic constraints such as fluency and compliance. Conventional reinforcement learning often merges these heterogeneous goals into a single reward, caus...

---

### 18. [Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents](https://arxiv.org/abs/2603.01548)

**Authors**: Neeraj Bholani  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01548v1  

#### Abstract
Tool-using LLM agents face a reliability-cost tradeoff: routing every decision through the LLM improves correctness but incurs high latency and inference cost, while pre-coded workflow graphs reduce cost but become brittle under unanticipated compound tool failures. We present Self-Healing Router, a...

---

### 19. [Faster, Cheaper, More Accurate: Specialised Knowledge Tracing Models Outperform LLMs](https://arxiv.org/abs/2603.02830)

**Authors**: Prarthana Bhattacharyya, Joshua Mitton, Ralph Abboud, Simon Woodhead  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.02830v1  

#### Abstract
Predicting future student responses to questions is particularly valuable for educational learning platforms where it enables effective interventions. One of the key approaches to do this has been through the use of knowledge tracing (KT) models. These are small, domain-specific, temporal models tra...

---

### 20. [PrivMedChat: End-to-End Differentially Private RLHF for Medical Dialogue Systems](https://arxiv.org/abs/2603.03054)

**Authors**: Sudip Bhujel  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03054v1  

#### Abstract
Large language models are increasingly used for patient-facing medical assistance and clinical decision support, but adapting them to clinical dialogue often requires supervision derived from doctor-patient conversations that may contain sensitive information. Conventional supervised fine-tuning and...

---

### 21. [Efficient Sparse Selective-Update RNNs for Long-Range Sequence Modeling](https://arxiv.org/abs/2603.02226)

**Authors**: Bojian Yin, Shurong Wang, Haoyu Tan, Sander Bohte, Federico Corradi, Guoqi Li  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.02226v1  

#### Abstract
Real-world sequential signals, such as audio or video, contain critical information that is often embedded within long periods of silence or noise. While recurrent neural networks (RNNs) are designed to process such data efficiently, they often suffer from ``memory decay'' due to a rigid update sche...

---

### 22. [Bridging Diffusion Guidance and Anderson Acceleration via Hopfield Dynamics](https://arxiv.org/abs/2603.02531)

**Authors**: Kwanyoung Kim  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.02531v1  

#### Abstract
Classifier-Free Guidance (CFG) has significantly enhanced the generative quality of diffusion models by extrapolating between conditional and unconditional outputs. However, its high inference cost and limited applicability to distilled or single-step models have shifted research focus toward attent...

---

### 23. [Learning Memory-Enhanced Improvement Heuristics for Flexible Job Shop Scheduling](https://arxiv.org/abs/2603.02846)

**Authors**: Jiaqi Wang, Zhiguang Cao, Peng Zhao, Rui Cao, Yubin Xiao, Yuan Jiang, You Zhou  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.02846v1  

#### Abstract
The rise of smart manufacturing under Industry 4.0 introduces mass customization and dynamic production, demanding more advanced and flexible scheduling techniques. The flexible job-shop scheduling problem (FJSP) has attracted significant attention due to its complex constraints and strong alignment...

---

### 24. [CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning](https://arxiv.org/abs/2603.02951)

**Authors**: Zhenquan Yao, Zitong Huang, Yihan Zeng, Jianhua Han, Hang Xu, Chun-Mei Feng, Jianwei Ma, Wangmeng Zuo  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.02951v1  

#### Abstract
Graphical User Interface (GUI) Agents, benefiting from recent advances in multimodal large language models (MLLM), have achieved significant development. However, due to the frequent updates of GUI applications, adapting to new tasks without forgetting old tasks in GUI continual learning remains an ...

---

### 25. [Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251)

**Authors**: Tanishq Kumar, Tri Dao, Avner May  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03251v1  

#### Abstract
Autoregressive decoding is bottlenecked by its sequential nature. Speculative decoding has become a standard way to accelerate inference by using a fast draft model to predict upcoming tokens from a slower target model, and then verifying them in parallel with a single target model forward pass. How...

---

### 26. [SWE-Hub: A Unified Production System for Scalable, Executable Software Engineering Tasks](https://arxiv.org/abs/2603.00575)

**Authors**: Yucheng Zeng, Shupeng Li, Daxiang Dong, Ruijie Xu, Zimo Chen, Liwei Zheng, Yuxuan Li, Zhe Zhou, Haotian Zhao, Lun Tian, Heng Xiao, Tianshu Zhu, Longkun Hao, Jianmin Wu  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.00575v1  

#### Abstract
Progress in software-engineering agents is increasingly constrained by the scarcity of executable, scalable, and realistic data for training and evaluation. This scarcity stems from three fundamental challenges in existing pipelines: environments are brittle and difficult to reproduce across languag...

---

### 27. [S5-HES Agent: Society 5.0-driven Agentic Framework to Democratize Smart Home Environment Simulation](https://arxiv.org/abs/2603.01554)

**Authors**: Akila Siriweera, Janani Rangila, Keitaro Naruse, Incheon Paik, Isuru Jayanada  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.01554v1  

#### Abstract
The smart home is a key domain within the Society 5.0 vision for a human-centered society. Smart home technologies rapidly evolve, and research should diversify while remaining aligned with Society 5.0 objectives. Democratizing smart home research would engage a broader community of innovators beyon...

---

### 28. [ToolRLA: Fine-Grained Reward Decomposition for Tool-Integrated Reinforcement Learning Alignment in Domain-Specific Agents](https://arxiv.org/abs/2603.01620)

**Authors**: Pengbo Liu  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.01620v1  

#### Abstract
Tool-integrated reasoning agents interleaving natural language deliberation with external API calls show promise for complex multi-step tasks. However, aligning such agents for high-stakes domain-specific deployment is challenging, as existing reinforcement learning uses coarse binary rewards (succe...

---

### 29. [Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration](https://arxiv.org/abs/2603.02760)

**Authors**: Linhao Zhong, Linyu Wu, Wen Wang, Yuling Xi, Chenchen Jing, Jiaheng Zhang, Hao Chen, Chunhua Shen  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.02760v1  

#### Abstract
Diffusion large language models (dLLMs) have recently attracted significant attention for their ability to enhance diversity, controllability, and parallelism. However, their non-sequential, bidirectionally masked generation makes quality assessment difficult, underscoring the need for effective sel...

---

### 30. [LaTeX Compilation: Challenges in the Era of LLMs](https://arxiv.org/abs/2603.02873)

**Authors**: Tianyou Liu, Ziqiang Li, Yansong Li, Xurui Liu  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.02873v1  

#### Abstract
As large language models (LLMs) increasingly assist scientific writing, limitations and the significant token cost of TeX become more and more visible. This paper analyzes TeX's fundamental defects in compilation and user experience design to illustrate its limitations on compilation efficiency, gen...

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

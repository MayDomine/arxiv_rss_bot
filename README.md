# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-07 12:03:06 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [TRANSIT: Transparent Scale-in for Multi-Node LLM Training](https://arxiv.org/abs/2610.07593)

**Authors**: Hyungyo Kim, Nicholas Satchanov, Hrishi Shah, Gaohan Ye, Jiaqi Lou, Robert Walkup, Shweta Salaria, I-Hsin Chung, Hubertus Franke, Seetharami Seelam, Apoorve Mohan, Nam Sung Kim  
**Category**: cs.DC  
**Published**: 2026-10-07  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2610.07593v1  

#### Abstract
TRANSIT is a transparent scale-in framework to enable multi-node model training on fewer GPUs while maintaining training efficiency by transparently leveraging CPU DRAM as an extension of GPU memory during distributed training. It achieves this through a user-space interposition layer, requiring no ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TRANSIT: Transparent Scale-in for Multi-Node LLM Training

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现代大模型训练面临三大瓶颈：
- **GPU 内存容量不足**：模型参数、梯度和优化器状态占用大量显存，导致必须 scale-out 到更多 GPU 上。
- **通信开销高**：随着 GPU 数量增加，collective communication（如 all-reduce）的同步和带宽压力显著上升，尤其在 oversubscribed 网络中性能波动剧烈。
- **资源利用率低**：集群调度需静态分配固定数量的 GPU，若无法满足则任务排队，影响开发迭代效率。

这些问题共同导致训练成本高昂、部署困难、对网络高度敏感。

---

### 🚀 提出的新方法与创新思路

**TRANSIT** 是一个 **透明的 scale-in 框架**，其核心思想是：
> 在不修改应用、框架、调度器、驱动或操作系统的前提下，利用 CPU DRAM 作为 GPU 显存的扩展，从而让原本需要多节点运行的大模型训练任务，在更少的 GPU 上高效执行。

#### 主要技术组件：
1. **User-space Interposition Layer**  
   通过 `LD_PRELOAD` 注入共享库，拦截 CUDA API 调用（如 `cudaMalloc`），将其重定向为 `cudaMallocManaged`，实现基于 **Unified Virtual Memory (UVM)** 的内存管理，无需代码改动。

2. **Runtime-Guided Optimization Policies**
   - **TRANSIT-Prefetch**：融合预取机制，将细粒度的 page fault 驱动迁移合并为粗粒度的数据传输，提升 PCIe 利用率。
   - **TRANSIT-Advise**：启用 **zero-copy access**，允许 GPU 直接从 CPU 内存读取数据而不迁移页面，避免频繁换页（thrashing）。

3. **Fine-grained & Transparent Control**
   - 不依赖框架级集成（如 PyTorch 或 DeepSpeed 修改）。
   - 支持多种并行策略（FSDP、TP、PP、MoE）。
   - 动态学习访问模式，在训练迭代中期施加优化策略。

---

### 🔍 相比现有方法的优势

| 方法 | 是否透明 | 是否需改框架 | 粒度 | 数据路径 | 多节点支持 |
|------|----------|---------------|--------|------------|--------------|
| ZeRO-Offload / Infinity | ❌ | ✅ | Coarse | On-demand / Prefetch | ✅ |
| TorchTitan | ❌ | ✅ | Coarse | On-demand | ✅ |
| DeepUM / G10 | ❌ | ✅（内核模块） | Page | Prefetch | ❌（单节点） |
| **TRANSIT** | ✅ | ❌ | **Page-level** | **Zero-copy + Prefetch** | ✅ |

> ✅ **优势总结**：
> - **完全透明部署**：零代码修改，适用于异构环境。
> - **更高的效率**：相比框架管理方案减少不必要的数据搬移。
> - **更强的可扩展性**：首次在 multi-node 场景验证 UVM-based scale-in 的可行性。
> - **降低通信负载**：通过减少参与 collective 的节点数，直接降低 per-node 网络流量。

---

## 2. 核心实验方法和设置

### 📊 使用的模型与数据集

使用 **TorchTitan** 框架进行分布式训练，评估以下典型 LLM 架构：

| 模型 | 类型 | 参数规模 | 并行方式 |
|------|------|-----------|------------|
| Llama3-70B | Dense | ~70B | FSDP+TP, TP+PP |
| Qwen3-32B | Dense | ~32B | FSDP |
| Llama4-Scout | MoE | ~？B | FSDP+TP |
| Qwen3-30B-A3B | MoE | ~30B | FSDP |

> 所有实验均启用 activation recomputation 和 mixed-precision（BF16/FP32）训练。

---

### ⚙️ 实验设置

#### 硬件平台：
- **Cluster-A**：16 节点 × 4 H100（PCIe Gen4），每节点 2TB CPU DRAM，ConnectX-7 NIC（400 Gbps）
- **Cluster-B**：3 节点 × 8 H100（PCIe Gen5），NVLink 更强（900 GB/s）

#### 网络配置：
- 使用 RoCEv2 + GPUDirect RDMA
- 测试带宽受限场景：限制 NIC 至 100 Gbps（模拟拥塞）

#### 评估指标：
| 指标 | 定义 |
|------|------|
| **Per-GPU TFLOPS** | 单卡浮点计算吞吐，主性能指标 |
| **Per-node Network Traffic** | 每节点每步通信量 |
| **PCIe Bandwidth Utilization** | CPU-GPU 数据传输效率 |
| **Page Fault Count** | 衡量 UVM 运行时开销 |
| **Job Queue Time / Makespan** | 集群级仿真中的调度延迟与完成时间 |

---

### 🆚 基线方法对比

| 基线 | 描述 |
|------|------|
| **No Offloading** | 原始配置，无内存卸载，最小 GPU 数要求 |
| **TorchTitan-Offload** | TorchTitan 自带 CPU 卸载功能 (`enable_cpu_offloading=true`) |
| **ZeRO-Offload** | DeepSpeed 实现的传统卸载 |
| **ZeRO-Infinity** | DeepSpeed 的高级卸载版本，支持分层存储 |
| **TRANSIT-Base** | 仅启用 UVM，默认 page fault 行为 |
| **TRANSIT-Prefetch** | 加入融合预取 |
| **TRANSIT-Advise** | 启用 zero-copy 访问 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**单节点性能提升**
- 在 Llama3-8B 上，**zero-copy** 可使训练吞吐提升 **最高达 3.4×** 对比默认 UVM。
- **TRANSIT-Prefetch** 将 page fault 减少 **97%**，PCIe 带宽利用率提高 **2.4×**。
- **TRANSIT-Advise** 将 PCIe 数据传输体积减少 **3×**，有效带宽提升 **2.5×**。

#### （2）**多节点 scale-in 性能**
- 使用 **50% 更少的 GPU**（例如从 24→12），仍能维持 **>90% 的 baseline per-GPU TFLOPS**。
- 在某些配置下（如 Llama3-70B, 16 GPUs），**TRANSIT-Advise 超过 baseline 3–4%**，因减少了通信开销。

#### （3）**对比现有框架**
| 方法 | 相对于 baseline 的 per-GPU TFLOPS 提升 |
|------|-------------------------------|
| **TRANSIT-Advise** | **+68% vs TorchTitan**, **+59% vs ZeRO-Offload**, **+42% vs ZeRO-Infinity** |
| TorchTitan-Offload | ~62% of baseline |
| ZeRO-Infinity | ~73% of baseline |

> ➤ 原因：TRANSIT 更智能地利用 GPU 显存，平均只卸载 **26 GB/GPU**，而 ZeRO-Infinity 卸载高达 **125 GB/GPU**。

#### （4）**通信与网络优化**
- **FSDP 场景下**，scale-in 显著降低 per-node 通信量：
  - 最多减少 **33% 的 per-node network traffic**。
- **带宽敏感性下降**：
  - 定义敏感度 $\rho = \text{Throughput}_{100\text{Gbps}} / \text{Throughput}_{400\text{Gbps}}$
  - TRANSIT 将 $\rho$ 从 2.1（原始）降至 1.5，表明在网络受限时退化更平缓。
  - 在 100 Gbps 下，8-GPU TRANSIT 配置比 baseline 提升 **44%**。

#### （5）**集群级仿真结果（使用 ACME 生产 trace）**
| 指标 | 结果 |
|------|------|
| **Inter-rack Traffic** | 减少 **15–49%** |
| **Job Queue Time** | 最多减少 **95.3%** |
| **Eligible Job Throughput** | 在半规模集群中反超全规模 baseline **+5.2%** |
| **Half-capacity Cluster Performance** | 匹配甚至超越 full-size cluster 的 job 吞吐 |

> ➤ 表明 TRANSIT 可显著缓解资源碎片化问题，提升整体集群利用率。

---

### 🔬 消融实验结果

| 配置 | Per-GPU TFLOPS（Llama3-70B, 16 GPUs） |
|------|----------------------------------------|
| No Offloading (24 GPUs) | 418 |
| TRANSIT-Base | 411 |
| TRANSIT-Prefetch | 422 |
| **TRANSIT-Advise** | **433** ✅ |
| TorchTitan-Offload | 258 |
| ZeRO-Offload | 272 |
| ZeRO-Infinity | 305 |

> ➤ 验证了 **zero-copy 是性能提升的关键因素**，prefetch 也有助于改善迁移效率。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **CPU DRAM 可作为高效的 GPU 显存延伸**  
   尽管 PCIe 带宽低于 HBM，但在小消息访问和网络拥塞场景下，本地 DRAM 访问延迟远低于跨节点 GDR 通信。

2. **Transparent Scale-in 是可行且高效的**  
   TRANSIT 首次证明：无需修改任何系统栈组件，即可在 multi-node 环境中实现高性能内存卸载。

3. **Zero-copy 是突破性能瓶颈的核心机制**  
   它打破了 UVM 默认的“迁移-驱逐”循环（thrashing），特别适合 optimizer states 等低重用数据。

4. **减少节点数可带来双重收益**  
   - 缓解 GPU 内存压力；
   - 降低通信开销与网络敏感性，形成正向反馈。

5. **集群级效益显著**  
   即使部分 job 应用 TRANSIT，也能通过改善 placement locality 显著降低跨机柜流量和排队延迟。

---

### ⚠️ 局限性

1. **当前策略未联合优化 prefetch 与 zero-copy**  
   当前 TRANSIT-Advise 全面启用 zero-copy，抑制了 prefetch 的作用；理想情况应动态选择最优路径。

2. **缺乏细粒度访问模式感知能力**  
   由于依赖 CUPTI 等用户态接口，难以精确捕获 tensor-level 的 reuse frequency，限制了精细化控制。

3. **对极大规模 MoE 模型支持待验证**  
   当前 MoE 实验集中在中小规模，极端稀疏路由下的行为尚未充分测试。

4. **PCIe 成为潜在瓶颈**  
   在极高内存超额订阅比下，PCIe 可能成为新的瓶颈，未来需结合 NVMe 或更高速互连。

---

### 🔮 未来工作方向

1. **Co-design Communication, Compute, and Memory Offloading**  
   探索如何在通信、计算和内存之间动态权衡，自动搜索最优并行策略（如减少 TP 规模以节省通信，靠 offloading 补足内存）。

2. **Runtime-aware Policy Selection**  
   开发自适应机制，根据 workload 特征（dense vs MoE）、网络状况、硬件配置动态切换 prefetch / zero-copy / hybrid 模式。

3. **Integration with Cluster Scheduler**  
   与 Kubernetes/OpenShift 等调度器联动，实现弹性 scale-in/out，进一步缩短队列等待时间。

4. **Extend to Other Memory Tiers**  
   将架构扩展至支持 NVMe SSD（类似 SSDTrain），构建统一的 hierarchical memory management layer。

---

> 💡 **一句话总结**：  
> **TRANSIT 通过透明地利用 CPU DRAM + zero-copy 技术，实现了“用更少 GPU 训练更大模型”，同时提升了训练效率、降低了通信负担，并在真实生产环境中展现出巨大的集群级增益。**

</details>

---

### 2. [Cascadia: Resident 975B MoE Inference on Eleven AI PCs](https://arxiv.org/abs/2610.07219)

**Authors**: Tate Berenbaum (Not Community Labs Inc.), Matias Parij (Not Community Labs Inc.), Muthaiah Venkatachalam (Intel Corporation)  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.07219v1  

#### Abstract
Mixture-of-experts models make nearly trillion-parameter capacity accessible with sparse per-token computation, provided that the serving system can distribute the weights and coordinate their execution. We present Cascadia's resident execution of Inkling, a 975B-total/41B-active-parameter model, on...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Cascadia: Resident 975B MoE Inference on Eleven AI PCs》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文旨在解决**在资源受限的分布式客户端设备上高效部署超大规模稀疏 MoE 模型**（如 975B 参数的 Inkling）的挑战。传统方法依赖高性能 GPU 集群和高带宽互联（如 NVLink），而本文探索如何利用消费级 AI PC 构建具备实用性能的推理系统。

核心难点包括：
- 如何在仅有 64GB 内存的设备上容纳近万亿参数模型的权重与状态；
- 如何协调跨设备的 MoE 路由、专家执行与 KV Cache 管理；
- 如何实现低延迟、高吞吐的并发服务；
- 如何准确评估部署路径下的 draft agreement（用于 speculative decoding）。

---

### 提出的新方法与创新思路

#### ✅ **1. 定制化的驻留式 MoE 引擎（Custom Resident MoE Engine）**
- **共享内存优化架构**：充分利用 Intel Core Ultra X7 358H 平台中 CPU 与集成 GPU（Arc B390）之间的共享内存，避免频繁的数据拷贝。
- **压缩图构建**：为 OpenVINO 的 fused iGPU primitive 构建压缩的 MoE 图，保留原始模型路由规则（如 top-6 + 2 shared experts）。
- **数值范围管理**：设计 FP16/FP32 边界上的动态缩放机制（layer-wise & row-wise scaling），确保精度恢复，防止溢出。

#### ✅ **2. 统一的稠密/稀疏块表示（Unified Dense/Sparse Operator Representation）**
- 将前两层 **dense feed-forward blocks** 映射到与 MoE 层相同的 fused-expert 算子家族中，通过将中间维度切分为 8 个全激活的 slice（each width=3072），复用同一套 fused kernel。
- 实现了 operator 层面的统一调度与优化，减少代码路径差异。

> **优势**：dense layer 推理时间从 ~8.1ms 降至 **4.5ms**（降低约 45%），显著提升首阶段处理效率。

#### ✅ **3. 流式并发推理管道（Streaming Pipeline for Concurrent Service）**
- 支持多请求并行处于不同 pipeline 阶段；
- 引入 **八行预填充窗口（eight-row prefill windows）** 和直接 token 返回路径；
- 实现短延迟 admission 控制与负载均衡。

#### ✅ **4. 基于真实部署状态的 draft evaluation 方法**
- 捕获运行时的 FP32 residuals 与 emitted token IDs，用于离线 replay 多 token prediction（MTP）模块；
- 分离 **vocabulary selection** 与 **weight quantization** 对 draft agreement 的影响，提供可解释的设计依据。

---

### 相比现有方法的优势

| 方面 | Cascadia 优势 |
|------|---------------|
| **硬件平台** | 使用 **11 台消费级 AI PC**（Intel Panther Lake + Arc B390 iGPU），而非 DGX 或云端集群 |
| **内存利用** | 利用共享 CPU-GPU 内存，避免显存瓶颈；支持高达 512k context positions 缓存 |
| **通信开销** | 仅需千兆以太网（gigabit Ethernet），无需 RDMA/NVLink |
| **部署灵活性** | 支持 speculative decoding、phrase learning、history reuse 等高级特性 |
| **评估真实性** | draft evaluation 基于实际 fleet 输出，反映真实数值路径 |

---

## 2. 核心实验方法和设置

### 数据集与输入配置
- **Prompt 来源**：人工构造的 **12 类任务家族**（explanation, code, arithmetic, story, tips, table, rewrite, facts, poem, instructions, translation, true/false）
- **输出长度**：每请求生成 **128 tokens**
- **上下文长度测试**：使用真实自然语言 prompt，长度从 1k 到 64k tokens 不等，嵌入一段代码并在末尾提问提取
- **draft evaluation corpus**：36 条序列（每类任务 3 条），共 5,724 个预测目标

---

### 实验设置

| 项目 | 配置 |
|------|------|
| **硬件平台** | 11 × Intel Core Ultra X7 358H（16 cores, 64GB LPDDR5X）<br>+ Arc B390 iGPU（12 Xe cores） |
| **网络** | Gigabit Ethernet（USB NCM + PCIe 接口） |
| **模型** | Inkling（975B total params, 41B active per token）<br>- MoE: 256 experts + 2 shared<br>- 66 decoder layers |
| **分区策略** | Pipeline 并行：每台机器负责连续 6 层（role 0 和 role 10 特殊） |
| **量化方案** | Experts: INT4（group-32）<br>Attention/Head: INT8 |
| **运行时** | Linux + OpenVINO 2026.3.1（iGPU 执行）+ Rust（控制流与状态管理） |

---

### 评估指标

| 指标 | 定义 |
|------|------|
| `Q_decode` | 共同解码区间内的聚合 decode throughput（tokens/s） |
| `Q_phase` | 整体阶段吞吐量（含 prefill、queueing、drain） |
| `TTFT`（Time to First Token） | 中位数与 p95 延迟 |
| **Draft Agreement** | MTP 预测 token 与实际 emitted token 的匹配率 |
| **Context Recovery Accuracy** | 在长 context 下是否能正确提取开头嵌入的代码 |

---

### 基线方法对比
本文未直接与其他系统进行端到端性能对比，而是强调其在**特定低成本硬件组合下实现了前所未有的规模与性能**。文中提及的对比系统包括：
- **Petals / TPI-LLM / exo**：边缘设备协作推理框架，但未涉及如此大规模 MoE 模型；
- **DGX Spark 部署**（参考文献 [6]）：使用 8×DGX Spark + NVFP4，代表高端部署路线；
- **Orca / SARATHI**：主流 LLM serving 系统，支持 chunked prefill 和 iteration-level scheduling。

> 本文贡献在于将这些思想整合进一个面向 **client-side AI PC fleet** 的完整 resident serving 架构。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 4）

| 并发流数（Streams） | Aggregate Decode Throughput (`Q_decode`) | Whole-Phase Throughput (`Q_phase`) | TTFT 中位数 |
|---------------------|------------------------------------------|------------------------------------|-------------|
| 1                   | 7.96 tokens/s                            | 6.98 tokens/s                      | 2.18 s      |
| 15                  | 24.60 tokens/s                           | 22.12 tokens/s                     | **6.05 s**  |
| 88 (**最优**)       | **60.29 tokens/s**                       | **46.87 tokens/s**                 | 34.61 s     |
| 176                 | 57.72 tokens/s                           | 45.24 tokens/s                     | 76.83 s     |

> 🔍 **观察**：在 **88 流**（每 pipeline stage 8 个 stream）达到峰值吞吐，进一步增加并发导致性能下降，表明存在最佳负载分配点。

---

### 性能增益分析

#### ✅ **稠密层优化效果（Table 2）**
| Layer | 传统三矩阵调用 | fused slices（8 slices） | 加速比 |
|-------|----------------|---------------------------|--------|
| 0     | 8.15 ms        | **4.51 ms**               | ~45% ↓ |
| 1     | 8.11 ms        | **4.45 ms**               | ~45% ↓ |

> 使用统一 fused operator 后，stage-level 计算耗时从 51.9ms → 43.7ms。

---

#### ✅ **Speculative Decoding 增益（Table 3）**
在单流重复请求场景下，得益于 history reuse 与 phrase learning，decode throughput 提升显著：

| Prompt Family | First Pass | Repeated Pass | Rate Ratio |
|--------------|------------|---------------|------------|
| Tips         | 3.55 t/s   | **11.66 t/s** | **3.28×**  |
| Explanation  | 5.09 t/s   | **14.70 t/s** | **2.89×**  |
| Poem         | 5.13 t/s   | **11.69 t/s** | 2.28×      |
| Table        | 3.76 t/s   | **8.66 t/s**  | 2.30×      |

> 📈 序列平均 decode throughput 从 **5.68 → 10.24 tokens/s**（+1.80×）

---

#### ✅ **上下文长度扩展能力（Table 5 & Figure 7）**

| Context Length | First Token Time | Decode Speed | Code Recovered? |
|----------------|------------------|--------------|-----------------|
| 1k             | 22s              | 4.73 t/s     | ✅ 3/3          |
| 4k             | 75s              | 3.91 t/s     | ✅ 3/3          |
| 16k            | 7.8min           | 2.71 t/s     | ✅ 3/3          |
| 32k            | 27.2min          | 1.51 t/s     | ✅ 2/2          |
| **64k**        | **109.7min**     | **0.82 t/s** | ✅ **2/2**      |

> 💡 **所有 19 个长 context 请求均成功恢复嵌入代码**，证明全局 attention 有效。

- **内存占用**：64k context 仅占约 0.5GiB/机器，远低于可用内存（61GiB）
- **瓶颈是计算**：first-token 时间符合 $ T = aN + bN^2 $，其中二次项来自 CPU 上单线程 attention loop

---

#### ✅ **Draft Evaluation 结果（Table 6）**

| 设置 | First-Draft Agreement |
|------|------------------------|
| Full head, FP32 weights | 66.81% |
| Head restricted to 65,536 tokens | 64.54% |
| + INT4/INT8 quantization | **64.36%** |

> 🔍 **分解影响**：
- Vocabulary truncation 贡献 **-2.271 pp**
- Weight quantization 贡献额外 **-0.175 pp**

说明在该部署中，**vocabulary size 是影响 draft accuracy 的主要因素**。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **可在消费级 AI PC 上实现近万亿参数 MoE 模型的实用化推理**  
   > 十一台 64GB 内存的 Intel AI PC 成功部署 Inkling 975B，并实现最高 **60.29 decode tokens/s** 吞吐。

2. ✅ **共享内存架构 + 自定义 MoE 引擎是关键使能技术**  
   > 利用 CPU-iGPU 共享内存、定制 OpenVINO 图、统一 dense/sparse 表示，大幅降低延迟。

3. ✅ **context 长度可扩展至 64k 以上且功能完整**  
   > 所有长 context 测试均能正确检索早期信息，验证了全局 attention 的有效性。

4. ✅ **first-token latency 主要受 CPU attention loop 限制**  
   > 当前实现中，大部分核心空闲，iGPU 利用率不足 60%，存在巨大优化空间。

5. ✅ **draft evaluation 必须基于真实部署路径的状态**  
   > 本文提出的方法分离了 vocabulary 与 quantization 影响，为后续设计提供量化依据。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **CPU attention 成为瓶颈** | 当前 attention 实现在单核串行执行，无法随 context scaling 提升性能 |
| **缺乏专家级并行优化** | 未探索 expert-level scheduling 或 offloading |
| **网络带宽限制潜力** | 千兆以太网可能成为更高并发下的瓶颈 |
| **暂未支持动态 batching** | admission 控制较基础，未实现 fully dynamic batching |
| **仅测试单一模型架构** | 结论对其他 MoE 模型的泛化性有待验证 |

---

### 未来工作方向

1. **引入 iGPU attention kernel**  
   > 将 key/value attention 计算迁移至 Arc B390 GPU，释放 CPU 资源，打破当前性能天花板。

2. **实现 block-level 或 head-level 并行 attention**  
   > 改进 attention loop 的并行度，应对百万级 context 场景。

3. **开发更智能的 admission 与 speculative proposer**  
   > 结合 MTP head 与本地 history，提升 draft acceptance rate。

4. **探索 disaggregated expert storage**  
   > 将冷专家存储于远程节点，热专家驻留本地，提升资源利用率。

5. **扩展至更多异构设备类型**  
   > 支持混合型号 AI PC fleet，增强部署灵活性。

---

> 🏁 **总结**：Cascadia 展示了一条通往“去中心化、低成本、高性能”大模型推理的新路径——**利用普通用户的 AI PC 组成协作 fleet，共同服务超大规模 MoE 模型**。这不仅是工程突破，也为未来边缘 AI 生态提供了重要范式参考。

</details>

---

### 3. [SoloQ: Calibration-Free Quantization for Diffusion Language Models](https://arxiv.org/abs/2610.07121)

**Authors**: Donghyun Lee, Arkapravo Ghosh, Varun Manjunath, Bumjoon Kyle Rhee, Hyunho Kook, Shiting Xiao, Youngeun Kim, Priyadarshini Panda  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.07121v1  

#### Abstract
Diffusion large language models dLLMs) have emerged as a promising alternative to autoregressive language models through bidirectional diffusion-based token generation. However, their growing model sizes and high inference costs make efficient deployment challenging: full-sequence denoising repeated...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SOLOQ: Calibration-Free Quantization for Diffusion Language Models

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **diffusion language models (dLLMs)** 在推理时面临高昂的计算和内存开销，尤其是在长上下文和多步去噪场景下。传统的 **post-training quantization (PTQ)** 方法虽然能压缩模型，但严重依赖 **calibration data** 来捕捉激活值（activations）在不同掩码状态和去噪步骤下的动态分布变化。然而，这些分布会随任务和生成过程剧烈波动，导致校准数据的选择对最终性能影响巨大，泛化能力差。

### 提出的新方法和新思路
本文提出了 **SOLOQ**，一种**无需校准（calibration-free）** 的量化框架，其核心思想是：
> 不去适应不断变化的激活分布，而是通过数学变换将其**映射到一个可预测的、稳定的分布空间**，从而实现无需数据的量化。

具体创新点包括：

- **K-RPBH Rotation (Kronecker-extended Randomized Permuted Block-Hadamard Rotation)**：
  - 一种高效的结构化旋转方法，将权重和激活值投影到一个具有**可预测边缘分布**（predictable marginal distribution）的归一化空间。
  - 结合了局部块内混合（Block-Hadamard）和全局跨块混合（Kronecker-extended cross-block mixing），解决了传统RPBH方法在高维下存在的能量不平衡问题。

- **轻量级尺度偏置校正 (Scale-Only Bias Correction)**：
  - 仅通过调整量化后的向量尺度（scale），即可精确恢复原始向量在量化方向上的投影分量，有效补偿量化带来的幅度失真，且不增加额外存储开销。

- **Commit-Time KV-Cache Quantization**：
  - 针对 **block-diffusion models** 的持久化KV缓存，提出在**块提交（commit）时才进行量化**。
  - 正在被去噪的活跃块保持全精度，避免了重复量化带来的误差累积，而已提交的块则被压缩以节省内存。

- **统一支持多种量化方案**：
  - 支持基于可预测分布设计的 **distribution-matched codebooks (SOLOQ-C)** 和硬件原生的 **NVFP4 (SOLOQ-N)**，均无需校准数据。

### 相比现有方法的优势
- **摆脱对校准数据的依赖**：实现了真正的 calibration-free，提升了方法的通用性和部署便捷性。
- **更强的鲁棒性**：由于不依赖特定校准集，性能在不同任务间更稳定。
- **全面的效率提升**：同时量化权重（W）、激活值（A）和KV缓存（KV），显著降低内存和计算开销。
- **硬件友好**：SOLOQ-N 直接利用硬件原生的 NVFP4 指令，获得最大加速。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了广泛的基准测试套件，分为两类：

#### 全序列 dLLMs (Full-sequence dLLMs)
- **LLaDA-Base-8B**, **LLaDA-1.5-8B**, **Dream-7B**
- 评测任务：`TruthfulQA`, `ARC-C`, `HellaSwag`, `WinoGrande`, `PIQA`, `MMLU`, `C-EVAL`, `HumanEval`, `GSM8K`

#### 块扩散 dLLMs (Block-diffusion dLLMs)
- **Fast-dLLM v2-7B**, **Nemotron-Labs-Diffusion-8B**
- 评测任务：`Human-B/P`, `MBPP-B/P`, `GSM8K`, `MATH`, `IFEval`, `MMLU`, `GPQA`

### 实验设置和评估指标
- **量化配置**：主要评估 **W4A4**（4-bit weights and activations）和 **W4A4KV4**（额外4-bit KV-cache）。
- **硬件平台**：
  - SOLOQ-C 在 **H200 GPU** 上使用8-bit Tensor Core kernels。
  - SOLOQ-N 在 **RTX Pro 6000 GPU** 上使用原生 NVFP4 kernels。
- **评估指标**：
  - **任务准确率 (Task Accuracy %)**：各基准测试的平均得分。
  - **峰值内存 (Peak VRAM)**：推理过程中的最高显存占用。
  - **端到端延迟 (End-to-end Latency)**：生成完整序列的时间。

### 基线方法对比
- **通用PTQ基线**：`RTN`, `AWQ`, `QuaRot+GPTQ`
- **dLLM专用PTQ基线**：`DLLMQuant+`, `DLLMQuant++`, `STaR-Quant`
- **其他**：`Fair-Calib`, `KIVI`, `OScar`, `TurboQuant` (用于KV-cache对比)

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **精度表现**：
  - 在 **LLaDA-1.5-8B** 上，SOLOQ-N 达到 **68.48%** 的平均准确率，优于最强基线 `STaR-Quant` (66.93%)。
  - 在 **Dream-7B** 上，SOLOQ-N 达到 **62.53%**，优于 `DLLMQuant++` (61.90%)。
  - 在 **Nemotron-Labs-Diffusion-8B** 上，W4A4KV4 配置下，SOLOQ-N 平均准确率达 **71.06%**。

- **效率提升**：
  - **峰值内存减少**：最多达 **2.61×**。
  - **端到端加速**：最多达 **2.24×**。
  - FPGA 实现上，NVFP4 加速器相比 BF16 能效提升 **2.70×**。

### 与基线方法的对比结果
- **显著超越基线**：在所有测试的 dLLM 上，SOLOQ-C 和 SOLOQ-N 均显著优于包括 `STaR-Quant` 在内的所有 calibration-based 基线，在知识、推理密集型任务上优势明显。
- **KV-Cache 量化优势**：在 2-bit KV 量化对比中（Table 5），SOLOQ-C/N 的平均准确率（~53.4%）远超 `KIVI` (~50.3%) 和 `OScar` (~46.6%)。
- **消融实验验证**：
  - **K-RPBH 有效性**：移除交叉块混合（Qk）后，KL散度从 0.00413 升至 0.00682，证明其对分布匹配至关重要。
  - **偏置校正作用**：加入 scale-only bias correction 后，`HellaSwag` 和 `GPQA` 等任务准确率均有提升（图5b）。
  - **旋转选择**：K-RPBH 在精度和延迟之间取得最佳平衡，优于 Haar（精度高但慢）和 RPBH（快但精度略低）。

---

## 4. 关键结论和发现

### 主要发现
1. **可预测表示塑造（Predictable Representation Shaping）** 是实现高效、免校准 dLLM 量化的有效范式。
2. **K-RPBH 旋转** 成功地将复杂、动态的 dLLM 表示转换为一个稳定、可预测的分布，为免校准量化奠定了基础。
3. **Commit-time KV 量化** 策略在保证精度的同时，有效缓解了 block-diffusion 模型的长上下文内存瓶颈。
4. **SOLOQ-N (NVFP4)** 凭借硬件原生支持，在实际加速和能效上展现出巨大潜力。

### 方法的局限性
- **在线旋转开销**：激活值的在线 K-RPBH 旋转引入了额外计算成本。尽管可通过权重折叠（folding）缓解，但在低比特量化下可能导致精度下降（尤其在 block-diffusion 模型上）。
- **SOLOQ-C 的执行效率**：由于使用非均匀 codebooks，无法直接利用硬件原生的 4-bit Tensor Cores，需降级到 8-bit kernels 执行，限制了其速度提升。
- **硬件依赖**：SOLOQ-N 的最大优势依赖于支持 NVFP4 的硬件（如 Hopper 架构 GPU）。在不支持的设备上，其加速效果会打折扣。

### 未来工作方向
- 探索更高效的旋转实现方式，或设计可完全离线折叠的旋转拓扑。
- 将 SOLOQ 框架扩展到其他模态的 diffusion transformers（如 vision, audio）。
- 研究结合训练感知（training-aware）的免校准量化方法，进一步逼近全精度性能。
- 推动硬件厂商对更多免校准友好量化格式（如 NVFP4）的支持。

</details>

---

### 4. [Learning PDE solution operators with variable initial conditions via Latent Dynamics Networks](https://arxiv.org/abs/2610.08475)

**Authors**: Stefano Maria Pizzamiglio, Stefano Pagani, Francesco Regazzoni  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.08475v1  

#### Abstract
In many-query scenarios, data-driven surrogate models provide an efficient alternative to high-fidelity solvers for simulating physical systems governed by Partial Differential Equations (PDEs). In this context, the Latent Dynamics Network (LDNet) has recently demonstrated remarkable performance in ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning PDE solution operators with variable initial conditions via Latent Dynamics Networks

## 1. 论文的主要贡献和创新点

### 解决的问题
原始的 **Latent Dynamics Network (LDNet)** 在学习时间依赖型偏微分方程（PDE）的解算子时，假设所有轨迹从**固定的初始状态**开始，这严重限制了其在现实场景中的应用，因为实际系统通常从**变化的、未见过的初始条件**演化而来。

### 提出的新方法与新思路
本文提出 **Initial-Condition Latent Dynamics Network (IC-LDNet)**，通过将初始潜态（initial latent state）从固定值改为由早期观测动态推断，从而支持可变初始条件。该框架保持了 LDNet 的两大优势：
- **端到端训练**（end-to-end training）
- **无编码器**（encoder-free），从而保证了模型对空间分辨率和网格拓扑的独立性（resolution independence）

具体提出了两种 encoder-free 的策略来解决潜态初始化问题：
1. **Auto-Decoder IC-LDNet (AD-IC-LDNet)**  
   为每个训练样本分配一个可学习的初始潜码（trainable latent code），在推理时对新样本通过优化拟合新的潜码。
2. **Meta-IC-LDNet**  
   采用 **meta-learning**（特别是 CAVIA 框架）的思想，将每个初始条件视为一个“任务”，训练一个共享的元参数（meta-parameters）和任务特定的上下文变量（context variables）。这种方法将潜态初始化嵌入到训练过程中，实现了快速在线适应。

### 相比现有方法的优势
| 特性 | 本文方法 (IC-LDNet) | 传统 Encoder-based 方法 | 其他 Encoder-free 方法 |
|------|---------------------|------------------------|------------------------|
| **分辨率不变性** | ✅ 保持 | ❌ 通常依赖特定网格 | ✅ 保持 |
| **参数效率** | ✅ 无额外编码器 | ❌ 额外编码器增加参数量 | ✅ 参数少 |
| **推理速度** | ✅ Meta-IC-LDNet 极快（3步GD） | ⚠️ 依赖编码器前向传播 | ❌ AD-IC-LDNet 需长时间优化 |
| **潜空间结构** | ✅ Meta-IC-LDNet 学习物理一致的拓扑 | ⚠️ 取决于编码器设计 | ❌ AD-IC-LDNet 结构混乱 |

## 2. 核心实验方法和设置

### 数据集
在四个涵盖不同物理领域的基准问题上进行了评估：
1. **1D Advection-Diffusion-Reaction (ADR)**  
   - 解析解已知，用于验证方法准确性。
2. **2D Flow around a Static Circular Cylinder**  
   - 雷诺数 $Re_p=100$，层流涡脱落现象。
3. **2D Flow around a Rotating Circular Cylinder**  
   - 圆柱角速度随时间随机变化，更复杂的非定常流动。
4. **Nonlinear Solid Dynamics of a 3D Cantilever Beam**  
   - 超弹性材料的非线性固体动力学问题。

### 实验设置与评估指标
- **训练方式**：监督学习，使用高保真数值模拟（如 OpenFOAM, FEniCS）生成的数据。
- **输入输出**：输入为边界条件或载荷历史 $u(t)$，输出为时空域上的场量（如压力、速度、位移）。
- **评估指标**：
  - **NRMSE (Normalized Root Mean Square Error)**：范围归一化的均方根误差，按场量分量报告。
  - **可视化**：预测场、绝对误差、真实场的对比图；潜空间结构分析（如 PCA、物理量着色）。
- **硬件**：主要在 CINECA 的 LEONARDO HPC 集群（NVIDIA A100 GPU）上训练。

### 基线方法对比
本文主要在 **AD-IC-LDNet** 和 **Meta-IC-LDNet** 之间进行对比，并与原始 LDNet 的思想进行比较。虽然没有直接与其他主流算子学习模型（如 DeepONet, FNO）进行数值对比，但强调了其在**分辨率不变性**和**快速适应新初始条件**方面的独特优势。

## 3. 主要实验结果和性能指标

### 关键性能数据（测试集 NRMSE）
| 测试案例 | 方法 | 压力 (p) | 速度 x (u) | 速度 y (v) | 位移等 |
|----------|------|---------|-----------|-----------|--------|
| **ADR** | AD-IC-LDNet | — | 3.5×10⁻³ | — | — |
| | **Meta-IC-LDNet** | — | **2.9×10⁻⁵** | — | — |
| **Static Cylinder** | AD-IC-LDNet | 4.1×10⁻³ | 4.7×10⁻³ | 7.1×10⁻³ | — |
| | **Meta-IC-LDNet** | **2.6×10⁻³** | **2.7×10⁻³** | **4.1×10⁻³** | — |
| **Rotating Cylinder** | Meta-IC-LDNet | 7.5×10⁻³ | 1.2×10⁻² | 1.3×10⁻² | — |
| **Nonlinear Beam** | Meta-IC-LDNet | — | — | — | ~4.7×10⁻² |

### 与基线方法的对比结果
- **Meta-IC-LDNet vs AD-IC-LDNet**：
  - **精度更高**：在 ADR 任务上，Meta-IC-LDNet 的平均 NRMSE 比 AD-IC-LDNet 低两个数量级。
  - **推理速度快一个数量级以上**：AD-IC-LDNet 需要上千次 Adam 迭代拟合新潜码，而 Meta-IC-LDNet 仅需 **3 步梯度下降**即可完成适应。
  - **损失景观更优**：Meta-IC-LDNet 的潜码拟合损失函数具有更平滑、条件更好的等高线（接近圆形），而 AD-IC-LDNet 存在多个局部极小值。
- **潜空间结构**：
  - **Meta-IC-LDNet** 自发地学习到与物理状态空间拓扑一致的潜空间。例如，在圆柱绕流中，潜码形成闭合环，对应涡脱落的周期性；在旋转圆柱中，潜空间能区分高频涡脱和低频马格努斯效应。

### 消融实验与正则化
- **正则化策略**：
  - **Curriculum Learning**：用于 Meta-IC-LDNet，逐步增加训练序列长度，有效解决了训练初期的不稳定性。
  - **Latent-State Penalty**：用于非线性梁问题，防止潜轨迹发散。
- **消融发现**：Meta-IC-LDNet 对超参数选择更敏感，但一旦成功训练，其泛化性和鲁棒性显著优于 AD-IC-LDNet。

## 4. 关键结论和发现

### 主要发现
1. **Meta-learning 是解决 encoder-free 潜态初始化的有效范式**。它不仅加速了推理，更重要的是作为一种归纳偏置（inductive bias），引导模型学习到**动态上有意义且物理一致的潜表示**。
2. **潜态初始化、表示学习和潜态演化不应被割裂设计**。通过 meta-learning 将三者联合优化，可以自发涌现出结构良好的潜空间。
3. **所提方法实现了真正的分辨率不变性**。得益于 coordinate-based decoder，模型可以在高度稀疏采样的数据上训练，并在任意高分辨率下恢复完整场量。
4. **快速推理解锁了多查询应用**。在旋转圆柱的最优控制（Optimal Control）任务中，Meta-IC-LDNet 仅用 **16 秒**就完成了传统方法难以负担的优化，展示了其在实时控制和逆设计中的巨大潜力。

### 方法的局限性
1. **训练敏感性**：Meta-IC-LDNet 的训练过程对超参数选择敏感，有时需要 case-by-case 的正则化策略（如 Curriculum Learning）才能稳定收敛。
2. **对分布外样本泛化能力有限**：在非线性梁测试中，对于训练分布之外的大变形样本，预测误差明显增大。
3. **统计显著性不足**：部分测试集样本量较小（如圆柱绕流仅 6-7 个测试样本），影响了泛化指标的统计可靠性。
4. **缺乏广泛基准对比**：未与 DeepONet、FNO 等主流算子学习模型进行直接性能比较。

### 未来工作方向
1. **探索更多优化与控制应用**：利用其快速推理特性，应用于更复杂的流体控制、材料设计等多查询问题。
2. **引入显式的物理或拓扑约束**：在潜空间上施加先验约束（如对称性、周期性），以进一步提升训练稳定性和表示的物理可解释性。
3. **改进训练鲁棒性**：开发更通用的正则化或初始化策略，降低对超参数的依赖。
4. **扩展到更广泛的 PDE 类型**：验证方法在湍流、多相流等更复杂物理系统中的有效性。

</details>

---

### 5. [FailBench: Evaluating Fault Tolerance Across Distributed Training Architectures](https://arxiv.org/abs/2610.07688)

**Authors**: Khaled Aljbab, Amine Barrak  
**Category**: cs.DC  
**Published**: 2026-10-07  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.07688v1  

#### Abstract
Distributed deep learning relies on data, pipeline, tensor, and hybrid parallelism, yet fault-tolerance mechanisms are typically evaluated only on the architecture for which they were designed. This leaves practitioners with little guidance when choosing mechanisms across architectures. FailBench pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FAILBENCH: Evaluating Fault Tolerance Across Distributed Training Architectures**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前分布式深度学习（DDL）中的 **fault tolerance (FT)** 机制通常只在为其设计的特定训练架构下进行评估，缺乏跨架构的统一比较。这导致实践者在选择 FT 机制时缺乏指导，难以判断哪种机制在不同架构下表现最优。

### **提出的新方法与新思路**
作者提出了 **FAILBENCH** —— 一个**统一的评估框架**，用于在多种分布式训练架构和故障容忍机制之间进行系统性、跨架构的对比评估。

#### **核心创新点：**
- ✅ **统一评估平台**：首次将 7 种主流分布式训练架构（如 Ring All-Reduce、Pipeline Parallelism、3D Hybrid 等）与 8 种 FT 机制（如 Checkpointing、Replication、Elastic Reconfiguration）组合，在相同硬件和故障轨迹下进行端到端评估。
- ✅ **兼容性映射（Compatibility Map）**：构建了 63 个架构-机制对的可行性分类（可行 Viable / 退化 Degenerate / 结构不可行 Structurally Infeasible），揭示了某些机制无法直接迁移至其他架构的根本原因。
- ✅ **多维度故障注入**：支持单次（F1）、并发（F2）和级联（F3）三种 fail-stop 故障模式，更贴近真实场景。
- ✅ **决策框架（Decision Framework）**：基于实证结果，提供了一个以“期望任务完成时间”为目标的量化选择模型，帮助用户根据集群 MTBF、状态大小、存储速度等参数选择最优 FT 策略。

### **相比现有方法的优势**
| 维度 | 现有研究局限 | FAILBENCH 改进 |
|------|----------------|----------------|
| **评估范围** | 单一架构内比较（如仅 Ring AR） | 跨 7 架构 × 8 机制 × 3 故障轨迹 |
| **公平性** | 不同论文使用不同设置 | 同一集群、相同 workload 和 fault trace |
| **可复现性** | 缺乏开源工具链 | 开源完整代码、配置、脚本和日志 |
| **实用性** | 只报告单一优势点 | 提供 Pareto 权衡分析与部署建议 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与工作负载**
- **小状态工作负载**：`ResNet-50` on synthetic data，每 rank 状态约 **205 MB**。
- **大状态工作负载**：自定义 `Megatron-style MLP stack`（模拟 Transformer FFN 层），每 rank 状态达 **2.5–2.7 GB**。
- 所有实验均不涉及实际数据加载，聚焦于通信、同步与恢复行为。

### **实验设置**
- **硬件平台**：Oakland University 的 Matilda HPC 集群，8×NVIDIA V100（2 节点），HDR InfiniBand + BeeGFS。
- **软件栈**：PyTorch 2.3.0, CUDA 12.1, NCCL 2.20.5。
- **训练规模**：固定 world size = 8，覆盖以下架构：
  - A1: Parameter Server (sync/async)
  - A2: Ring All-Reduce
  - A3: Tree All-Reduce
  - A4: Pipeline Parallelism
  - A5: Tensor Parallelism
  - A6: Decentralized Peer-to-Peer (D-PSGD)
  - A7: 3D Hybrid (DP×PP×TP)

### **评估指标**
| 指标 | 定义 |
|------|------|
| **SS Overhead** | 稳态开销 = $ \frac{T_{\text{off}}}{T_{\text{on}}} - 1 $，即启用 FT 后吞吐下降比例 |
| **TTD (Time to Detect)** | 故障发生到首个存活节点检测到的时间 |
| **Trestore** | 有效恢复成本（含状态加载） |
| **Trecovery** | 总恢复延迟 = Trestore + TpG（进程组重建）+ Tinit（机制初始化） |
| **Expected Job Completion Time** | 决策框架目标函数：$ E[T] = T_0(1+\theta_m) + E[N](T_{\text{det}} + T_{\text{rec}} + T_{\text{lost}}) $ |

### **基线方法对比**
- **FT9 (No-FT baseline)**：无任何容错机制，暴露各架构固有的失败行为。
- 其他 8 种机制分为三类：
  - **Checkpoint-class**: FT1–FT4（周期磁盘、内存复制、异步写、JIT）
  - **State Replication**: FT5–FT6（连续复制、k副本冗余）
  - **Reconfiguration**: FT7–FT8（数据重分布、Elastic Process Group）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 机制 | SS Overhead | Trestore | Lost Work | 特点 |
|------|-------------|----------|-----------|------|
| **FT1 (Disk Checkpoint)** | **0.48%** | ~155ms | ~50 steps | 最低稳态开销，适合高 MTBF 场景 |
| **FT2 (In-Memory Replication)** | 3.78% | **~17ms** | ~50 steps | 恢复最快，但开销随架构变化剧烈（3.7% → 176%） |
| **FT4 (JIT Checkpoint)** | **0%** | ~900ms | ~0 steps | 无周期开销，但紧急保存耗时长（~0.9s 小状态，~10s 大状态） |
| **FT5 (Continuous Replication)** | 372% | ~17ms | ~0.5 steps | 开销过大，被 FT2 完全支配 |
| **A6 (D-PSGD)** | 0% | ~0.5s | N/A | 原生降级运行，无需额外 FT |

### **与基线方法的对比结果**

#### **(1) 在 A2 Ring-AR 上的 Pareto 前沿**
- **FT1、FT2、FT4 构成非支配三元组**：
  - FT1：低 SS，中等恢复
  - FT2：中等 SS，极快恢复
  - FT4：零周期 SS，极高恢复延迟
- **没有单一最优机制**，选择取决于 MTBF 和 SLA。

#### **(2) 架构依赖性强**
- **FT2 内存复制**在不同架构上的 SS 开销差异巨大：
  - A1–A3: ~3.7%
  - A4/A5: **>175%**（因 step time 短而 save cost 高）
  - A7: 14.4%
- 表明：**同一机制在不同架构下的性价比完全不同**。

#### **(3) 多故障场景下的鲁棒性**
- **F2（并发双故障）**：
  - A2/A4/A5/A7 上几乎所有机制都失败（halted）
  - **只有 A6 (D-PSGD)** 能稳定恢复
- **F3（级联三故障）**：
  - 所有非 A6 架构均无法完成恢复
  - A6 + FT1 成功完成全部 2000 步训练

### **消融实验结果**
- **M2 控制变量实验（A2 + 2.6GB state）**：
  - FT1 SS 开销从 0.48% 升至 **3.14%**
  - FT2 在该状态下出现 NCCL 死锁，**无法运行**
  - 说明：**状态大小显著影响机制可用性和性能**
- **Process Group Timeout 实验**：
  - 将 `init_process_group` 超时设为 30s → TTD ≈ 30s
  - 默认 600s 导致多数架构检测延迟高达 600s
  - 结论：**检测延迟主要由 runtime 超时决定，而非架构本身**

---

## **4. 关键结论和发现**

### **主要发现**
1. 🔹 **不存在通用最优 FT 机制**  
   - 在 A2 上，**disk checkpoint (FT1)** 稳态开销最低（0.5%），**in-memory replication (FT2)** 恢复最快（~17ms），**JIT checkpoint (FT4)** 无周期开销但恢复代价高（~0.9s）。
   - 三者形成 **Pareto 前沿**，选择需权衡。

2. 🔹 **架构决定机制有效性**  
   - 同一机制（如 FT2）在不同架构上的 SS 开销从 **3.7% 到 176%** 不等。
   - 某些组合**结构上不可行**（如 A6 + FT2），因 A6 使用 per-edge subgroup 而 FT2 需要全局 ring。

3. 🔹 **原生降级优于事后恢复**  
   - **A6 (D-PSGD)** 和 **A1-async PS** 在故障后能继续训练：
     - A6 吞吐提升 17.9%，但 **loss 进展无改善**（通信压力降低 ≠ 学习效率提高）
     - A1-async 通过 timeout 自动剔除失效 worker
   - 这类架构本身具备 FT 能力，无需额外机制。

4. 🔹 **多故障下传统机制失效**  
   - 在 F2/F3 场景中，除 A6 外所有架构+机制组合均无法完成恢复。
   - 表明：**仅保存状态不足以实现端到端容错**，还需运行时重构能力。

5. 🔹 **恢复时间受 PG 重建主导**  
   - 当 `TpG`（进程组重建）超过数百毫秒时，不同机制间的 `Trestore` 差异变得不显著。
   - 此时，**稳态开销成为主导因素**，favor FT1 或 FT4。

---

### **方法的局限性**
- ❌ **未涵盖 Byzantine 错误**：仅测试 fail-stop crash，不包括 silent data corruption 或恶意行为。
- ❌ **简化实现**：FT4/FT5/FT6 是生产系统的简化代理，未完全复现如 Bamboo 或 Oobleck 的复杂逻辑。
- ❌ **小规模实验**：8 GPU 规模，绝对数值不能直接外推至千卡集群。
- ❌ **合成 workload**：使用 MLP 而非完整 Transformer，缺少 activation checkpointing 和 sequence parallelism 影响。

---

### **未来工作方向**
1. ✅ 扩展至更大规模（百/千卡）和真实 LLM 训练 workload。
2. ✅ 支持更多故障模型：straggler、network partition、silent corruption。
3. ✅ 引入 **coded computation**（如 Gradient Coding）并进行头对头比较。
4. ✅ 探索 **hybrid FT mechanisms**（如 FT4 + FT2 fallback）。
5. ✅ 将 FAILBENCH 集成进 MLPerf 等标准 benchmark 套件。

---

> 📦 **开源声明**：作者已将 FAILBENCH 框架、所有配置、故障调度器、采集与分析脚本以及所有原始输出作为开放 artifact 发布，极大提升了可复现性与社区价值。

</details>

---

### 6. [OSFP4: Joint Optimization of Diagonal Smoothing and Block Scales for NVFP4 Quantization](https://arxiv.org/abs/2610.08231)

**Authors**: Neriah Ben David, Ori Meir, Or Ordentlich  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.08231v1  

#### Abstract
NVFP4 is an attractive datatype for large language model (LLM) inference, offering compact storage and native tensor-core acceleration. However, preserving accuracy using NVFP4 requires careful quantization. In this work we develop a novel quantization scheme called Optimized Smoothing and Scaling f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：OSFP4: Joint Optimization of Diagonal Smoothing and Block Scales for NVFP4 Quantization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **NVFP4**（NVIDIA Floating Point 4）量化在大语言模型（LLM）推理中的应用，解决其在低比特表示下精度下降的问题。尽管 NVFP4 具备存储紧凑和硬件加速优势，但直接使用会导致显著的量化误差，尤其是在动态范围有限的 **E2M1 FP4** 格式下。

传统方法如 absmax scaling 或简单的对角平滑（diagonal smoothing）无法充分优化权重与激活值的分布以适应 FP4 的非均匀量化特性。本文指出，**需要联合优化对角平滑矩阵（A）和块尺度（block scales）**，并考虑实际使用的舍入方式（RTN 或 SIC），才能最大化精度恢复。

---

### 提出了什么新方法或新思路
作者提出了一种名为 **OSFP4**（Optimized Smoothing and Scaling for NVFP4）的新量化方案，其核心创新如下：

- **联合优化框架**：首次将对角平滑矩阵 $ A \in \mathbb{R}^{n\times n} $ 和 E4M3 块尺度 $ \mathbf{I}_w, \mathbf{I}_x $ 进行端到端联合优化，目标是最小化矩阵乘法（MatMul）的量化误差。
  
- **基于随机化乘性抖动（multiplicative dither）的可微损失函数**：
  - 引入一个**可微的随机化 FP4 量化器** $ \tilde{Q}_{\text{FP4}} $ 来替代不可导的真实确定性量化器 $ Q_{\text{FP4}} $。
  - 利用该随机化模型推导出一个平滑的期望均方误差（MSE）损失函数 $ \phi(x) $，使其适用于梯度优化。
  - 证明了该随机化损失可以作为真实量化误差的良好代理（见 Lemma 1）。

- **区分 RTN 与 SIC 的建模差异**：
  - 对于 **Round-to-Nearest (RTN)**，直接建模误差传播。
  - 对于 **Successive Interference Cancellation (SIC)**，引入高分辨率假设，并利用 Cholesky 分解 $ \Sigma_x = U^\top U $ 构造预测残差能量项 $ U_e(r) $，从而更准确地反映 SIC 的反馈机制对误差的影响。

- **两阶段优化流程**：
  1. **连续联合优化**：使用随机化损失函数联合优化 $ A $ 和初始 $ \mathbf{I}_w $。
  2. **离散尺度选择**：固定优化后的 $ A $，在 $ [\gamma/1.2, \gamma/0.3] $ 范围内搜索最优的 E4M3 块尺度，使最终的确定性量化误差最小。

---

### 相比现有方法的优势
| 方法 | 局限性 | OSFP4 的改进 |
|------|--------|-------------|
| **Absmax-to-6** | 忽略组内分布不平衡，易导致大量小值被量化为零 | 通过 $ A $ 平衡组内分布，减少信息丢失 |
| **H-Scale / ScaleSearch** | 仅优化尺度，未调整输入分布 | 联合优化 $ A $ 和 scale，协同提升精度 |
| **SmoothQuant** | 针对 INT 量化设计，平衡无穷范数 | 针对 FP4 动态范围小的特点重新设计目标函数 |
| **GPTQ / MR-GPTQ** | 依赖后处理误差补偿，计算开销大 | 在量化前即完成最优变换，保持高效 |

> ✅ **核心优势**：OSFP4 在不增加运行时 MatMul 开销的前提下，通过预处理实现更高精度，且支持与 SIC 等先进舍入策略无缝结合。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **FineWeb-Edu**：用于校准（calibration），共 1024 个长度为 2048 的序列。
- **WikiText-2**：用于评估困惑度（perplexity）。
- **多任务基准测试集**：
  - MMLU-CoT
  - GSM8K
  - HellaSwag
  - WinoGrande
  - C-Eval, LiveBench, ARC-Challenge, BBH, GPQA Diamond（部分扩展模型）

---

### 实验设置和评估指标

#### 模型
- 主要模型：**Llama-3.1-8B-Instruct**
- 扩展验证：Qwen3-8B, Qwen3-30B-A3B-Instruct, Gemma-4-31B-IT, Llama-3-8B

#### 量化配置
- **W4A4**：权重和激活均量化为 NVFP4（每 16 通道共享一个 E4M3 scale）
- **W4A16**：仅权重量化为 NVFP4，激活保留 BF16
- **Block size**: 16（对应 NVFP4 规格）
- **Tensor scale**: 固定为 1（权重），激活使用全局缩放因子

#### 评估指标
| 指标 | 描述 |
|------|------|
| **平均准确率（Avg. Accuracy）** | 多任务平均得分，衡量整体语义理解能力 |
| **Recovery %** | 相对于 BF16 基线的性能恢复比例 |
| **Perplexity (PPL)** | WikiText-2 上的语言建模性能 |
| **MatMul MSE Ratio** | 投影层输出误差相对于 FP8 RTN 的相对均方误差 |
| **Throughput (tokens/s)** | 推理吞吐量，评估效率 |

---

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **Absmax Baseline** | RTN with absmax-to-6 |
| **Scale Optimization** | ScaleSearch, 4over6, SOAR, H-Scale |
| **Error Compensation** | GPTQ, MR-GPTQ |
| **Vendor Checkpoint** | NVIDIA 官方发布的 NVFP4 检查点 |
| **Rotation-based** | MR-GPTQ（含 Hadamard 变换） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Llama-3.1-8B-Instruct）

#### ✅ 多任务平均准确率（W4A4, absmax scaling）
| 方法 | 平均准确率 (%) |
|------|----------------|
| BF16 baseline | 79.22 |
| FP-Quant GPTQ | 76.34 |
| MR-GPTQ | 76.18 |
| SOAR | 76.60 |
| H-Scale (W4A4) | 76.67 |
| **OSFP4 W-RTN/X-RTN** | **76.36** |
| **OSFP4 W-SIC/X-RTN** | **77.03** ✅ |

> 🔺 **OSFP4 W-SIC/X-RTN 达到最高平均精度，超越所有竞品**

#### ✅ W4A16（仅权重量化）
| 方法 | 准确率 (%) |
|------|-----------|
| RTN | 77.67 |
| H-Scale | 78.00 |
| **OSFP4 W-RTN** | **78.15** |
| **OSFP4 W-SIC** | **78.42** ✅ |

> 🔺 比 H-Scale 提升 +0.42%，接近 BF16 性能（98.98% 恢复）

#### ✅ WikiText-2 困惑度（PPL ↓）
| 方法 | W4A16 | W4A4 |
|------|-------|------|
| Absmax RTN | 7.55 | 7.88 |
| MSE scales (A=I) | 7.52 | 7.85 |
| **OSFP4 W-RTN** | **7.42** | **7.77** |
| **OSFP4 W-SIC** | **7.39** | **7.67** ✅ |

> 🔺 显著优于 absmax 和单独 scale 优化，在 W4A4 下降低 PPL 达 0.21

---

### 与基线方法的对比结果
- 在 **W4A4 + absmax** 设置下，OSFP4 W-SIC/X-RTN 比最强基线 H-Scale 高 **+0.36%**。
- 即使不进行在线 activation scale search，OSFP4 仍优于 **ScaleSweepMSE + GPTQ**（77.03 vs 76.69）。
- 在 **Qwen3-30B-A3B-Instruct** 上，OSFP4 W-RTN 达到 **99.42%** BF16 性能恢复，显著领先其他方法。

---

### 消融实验结果（Ablation Study）

来自 Appendix H 的组件消融表明：

| 组件 | 影响 |
|------|------|
| **仅优化 scale（MSE）** | PPL 从 7.55 → 7.52（小幅改善） |
| **加入 diagonal smoothing（W-RTN）** | PPL → 7.42 |
| **进一步加入 SIC（W-SIC）** | PPL → **7.39** |
| **完整 OSFP4（smooth + scale + SIC）** | 实现最小 MatMul MSE 和最佳 PPL |

> 🔍 发现：**对角平滑 $ A $** 是关键，它使得后续的 scale 优化和 SIC 更有效；三者协同作用带来最大收益。

---

## 4. 关键结论和发现

### 主要发现
1. **FP4 的动态范围限制使其不能简单套用 INT 量化的平滑策略**，必须专门设计面向浮点格式的优化目标。
2. **对角平滑 $ A $** 不仅可用于平衡无穷范数，更能通过调节组内分布来匹配 FP4 的非均匀量化间隔（特别是在 $ Z_{\text{FP4}} = [7/4, 7] $ 区间附近最小化误差）。
3. **联合优化 $ A $ 与 block scales** 显著优于分步优化或仅优化其中之一。
4. **SIC 与 OSFP4 的结合效果最佳**，因其能利用协方差结构进一步压缩误差。
5. **OSFP4 保留了约 94–97% 的原生 NVFP4 prefill 吞吐量**，说明额外的对角变换可在运行时融合进 LayerNorm 或低成本执行。

---

### 方法的局限性
- **仅适用于线性层**：目前只应用于 Attention 和 MLP 中的线性投影。
- **MoE 支持受限**：对于专家网络（MoE），需为每个专家维护独立的 $ A $ 矩阵，增加了内存和调度复杂度（文中未完全实现 activation quantization）。
- **依赖校准数据**：需要一定量的校准样本估计 $ \Sigma_x $，对数据敏感性有待研究。
- **未探索非对角变换**：虽然更高效，但对角结构可能不如旋转等更复杂变换强大。

---

### 未来工作方向
- 将 OSFP4 扩展至 **Convolutional Layers** 和 **Embedding Layers**。
- 探索 **轻量化非对角变换**（如低秩修正）以进一步提升性能。
- 结合 **量化感知训练（QAT）** 进行端到端优化。
- 开发 **自动选择 RTN/SIC 模式的控制器**，根据不同层动态切换。
- 支持 **MoE 模型的路由感知量化插件**，实现 per-expert $ A $ 应用。

---

## 总结

✅ **OSFP4 是一种专为 NVFP4 设计的新型量化方法**，通过引入可微的随机化 FP4 模型，实现了对角平滑矩阵 $ A $ 与块尺度的联合优化。实验表明其在多个主流 LLM 上均达到当前最优的 W4A4/W4A16 量化精度，同时几乎不牺牲推理效率，是迈向高效低比特 LLM 推理的重要一步。

</details>

---

### 7. [GAMEGO: Training Game-Dev Agents with Synthetic Trajectories Anchored in Real-World Assets](https://arxiv.org/abs/2610.06910)

**Authors**: Haoyue Yang, Jingyao Li, Zhengfan Wu, Jing Liu, Xuanle Zhao, Kang Liu  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.06910v1  

#### Abstract
Recent advances in Large Language Models (LLMs) have demonstrated remarkable capabilities in web front-end execution, with browser-based game generation emerging as a particularly prominent frontier. While previous efforts frequently rely on complex multi-turn workflows or focus on static game evalu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《GAMEGO: Training Game-Dev Agents with Synthetic Trajectories Anchored in Real-World Assets》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Large Language Models (LLMs)** 的游戏生成系统在处理**稀疏用户查询**时，常因信息不足而做出过度假设，导致生成的游戏存在以下问题：
- 游戏机制不完整（incomplete mechanics）
- 游戏流程断裂（disconnected gameplay flows）
- 视觉表现力差（limited visual aesthetics）

现有方法要么依赖复杂的多轮交互流程，要么仅关注静态评估基准，缺乏端到端、高质量的自动化训练数据。

---

### 提出了什么新方法或新思路
本文提出 **GameGo**，一个可扩展的框架，用于将简短的游戏种子（game seeds）转化为工业级的 **Product Requirements Document (PRD)**，并进一步压缩为高密度的任务查询，以指导编码代理（coding agent）进行游戏开发。

#### 核心创新点：
1. **从真实世界资产构建合成轨迹（Synthetic Trajectories Anchored in Real-World Assets）**
   - 收集来自 Steam、网页游戏平台（如 Y8、CrazyGames）、微信小游戏等 18 个来源的真实游戏记录，经过去重后保留 67,564 个游戏种子。
   - 构建 **GameGoData** 数据集：包含 55,060 条涵盖 2D、2.5D 和 3D 游戏的完整开发轨迹（development trajectories），每条轨迹包含代码编辑、工具调用、执行反馈等全过程日志。

2. **任务自适应查询压缩（Task-Adaptive Query Construction）**
   - 提出一种动态压缩机制，在保留核心玩法约束的同时，去除冗余实现细节，提升信息密度。
   - 定义 **关系密度增益（relation density gain）** $ G_s = R_s / r_s $，其中 $ R_s $ 是保留的关系比例，$ r_s $ 是保留的 token 比例。
   - 实验显示，压缩后的查询相比完整 PRD 实现了中位数 **3.79 倍的信息密度增益**。

3. **分离设计规划与执行指令**
   - 将复杂的设计决策封装在内部 PRD 中，仅向 coding agent 提供精炼后的“任务特定查询”（task-specific query），既保证完整性又保留创作自由度。

---

### 相比现有方法的优势
| 维度 | GameGo 优势 |
|------|-------------|
| **数据质量与规模** | 构建了目前最大规模的单轮游戏生成轨迹数据集 GameGoData（55,060 条） |
| **任务表达效率** | 压缩查询长度与原始查询相当（~235 tokens），但信息更密集，优于直接查询和完整 PRD |
| **生成效果** | 人类偏好测试中，压缩查询生成的游戏分别获得 **85.7%**（vs. 全 PRD）和 **86.6%**（vs. 直接查询）的偏好率 |
| **训练有效性** | 在 GameGoData 上微调的小参数模型（如 Qwen3.5-27B）性能接近甚至超越前沿大模型 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **GameGoData**：
  - 包含 **55,060 条**完整的开发轨迹。
  - 覆盖 **2D (61.4%)、2.5D (20.1%)、3D (18.5%)** 游戏。
  - 来源于 67,564 个真实游戏种子，经四阶段处理（收集 → 规划 → 编译 → 执行）生成。
- **GameGoBench**：
  - 一个独立的评测基准，包含 **124 个多样化游戏开发任务**。
  - 按渲染格式分层：2D (47)、2.5D (22)、3D (55)，并按难度分为 Easy/Medium/Hard。
  - 严格与训练集隔离，确保无数据泄露。

---

### 实验设置和评估指标

#### 模型训练
- **基础模型**：Qwen3.5-27B 和 Qwen3.8-27B。
- **训练方式**：全参数监督微调（SFT），使用 `response-masked` 损失函数，仅对 agent 的思考、响应和工具调用计算损失。
- **训练配置**：
  - 上下文窗口：256K tokens
  - Batch size：8
  - 学习率峰值：1e-5
  - 冻结视觉编码器

#### 推理环境
- 使用 **Code Arena** 框架中的沙箱环境。
- 每个任务最多允许 100 步交互，每次调用最多生成 65,536 tokens。
- 工具集包括文件编辑、项目编译（`build_project`）、shell 执行等。

---

### 评估指标

#### 自动化评估（Automated Evaluation）
1. **Execution Rate (Exec.)**：成功渲染页面的比例。
2. **Requirements (Req.)**：功能保真度，四级层次评估：
   - 核心概念
   - 输入响应性
   - 游戏循环
   - 查询规范一致性
3. **Quality**：视觉质量，评估主题艺术、场景构图、UI 集成。
4. **Overall**：综合得分。

#### 人工评估（Human Evaluation）
- 六名独立专家进行双盲配对比较（pairwise judgment）。
- 判断标准：保真度（fidelity）、可玩性（playability）、整体质量。
- 报告偏好率（Preference = Win / (Win + Loss)）。

#### 对比基线
- **Frontier Models**：
  - Claude Opus 5
  - GPT-5.6
  - DeepSeek-V4-Pro
  - GLM-5.3
  - Qwen3.8-Max
  - Kimi-K3
- **Baseline Models**：
  - Qwen3.5-27B（未微调）
  - Qwen3.8-27B（未微调）
- **GameGoCoder**：在 GameGoData 上微调后的版本。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| Model | ArtifactsBench-G (Overall) | CookieBench-G (Overall) | GameGoBench (Overall) |
|-------|----------------------------|----------------------------|------------------------|
| **Claude Opus 5** | 72.76 | 83.68 | 61.84 |
| **GPT-5.6**       | 63.92 | 75.36 | 52.78 |
| **Qwen3.8-Max**   | 66.92 | 85.96 | 58.35 |
| **Qwen3.5-27B**   | 46.31 | 41.35 | 34.00 |
| **GameGoCoder 3.5** | **57.34** | **55.79** | **51.85** |
| **Qwen3.8-27B**   | 59.87 | 77.07 | 52.53 |
| **GameGoCoder 3.8** | **61.94** | **80.68** | **55.29** |

> ✅ **GameGoCoder 在所有基准上均显著优于其 base model**  
> 🔄 **GameGoCoder 3.8 性能接近甚至超过部分 frontier models**

---

### 人工偏好结果（Figure 5 & Table 2）

#### 人工偏好对比（GameGoCoder vs. Base Model）
- GameGoCoder 3.5 vs. Qwen3.5-27B：明显更优
- GameGoCoder 3.8 vs. Qwen3.8-27B：持续领先

#### 不同查询形式的人类偏好（Table 2）
| 对比项 | 偏好率（Preference） |
|--------|------------------|
| **full PRD vs. direct query** | 69.0% |
| **task-specific vs. full PRD** | **85.7%** |
| **task-specific vs. direct query** | **86.6%** |

> 💡 表明：**任务自适应压缩查询** 是最优策略。

---

### 消融实验结果（Ablation Studies）

#### 5.4.1 Query Construction Ablation
- **压缩分析**：
  - 压缩后查询平均长度：**235 tokens**（vs. 原始 PRD 平均 9,136 tokens）
  - 中位数压缩比：**44.2×**
  - 信息密度增益中位数：**3.79×**
- **复杂度自适应分配**：
  - 难度越高，保留的内容越多（Hard 任务保留 3.25% PRD 内容）
  - 动态调整信息预算，避免“一刀切”压缩

#### 5.4.2 Model Ablation（MoE 模型训练）
- 在未经过指令微调的基础 MoE 模型上训练：
  - Epoch 2 即已优于官方 post-trained checkpoint（偏好率 0.664）
  - Epoch 6 进一步提升至 0.807
- 表明：**GameGoData 对各类模型初始化均有效**

---

## 4. 关键结论和发现

### 主要发现
1. **高质量结构化需求是关键**：
   - 通过模拟真实游戏开发流程生成 PRD，能显著提升生成游戏的完整性和可玩性。
2. **信息密度优于信息量**：
   - 过长的 PRD 反而会干扰 LLM 的指令遵循能力；适度压缩、保留关键依赖才能最大化性能。
3. **任务自适应压缩优于固定模板**：
   - 根据任务复杂度动态调整输出长度和细节程度，实现了更好的权衡。
4. **小模型也能媲美大模型**：
   - 经 GameGoData 微调的 Qwen3.5-27B 模型，在多个 benchmark 上接近甚至超越更大规模的 frontier models。

---

### 方法的局限性
- **依赖高质量种子数据**：若原始游戏描述模糊或重复，会影响 PRD 质量。
- **压缩策略仍需人工定义规则**：虽然引入了 LLM 判断，但保护机制和风险识别仍有改进空间。
- **仅限于 Web 前端游戏**：当前框架基于 React 模板，难以直接迁移到 Unity/Godot 等专业引擎。
- **视觉资产仍由文本描述驱动**：尚未完全实现图像到游戏的端到端生成。

---

### 未来工作方向
1. **扩展到更多游戏引擎**：支持 Godot、Unity、Unreal 等专业开发环境。
2. **引入多模态输入**：结合截图、草图、音频等富媒体输入增强理解。
3. **闭环强化学习优化**：利用自动 playtesting 反馈迭代优化生成过程。
4. **开放生态建设**：
   - 已承诺开源所有代码、数据集和模型：https://github.com/Haoyue-Yang/GameGo
   - 鼓励社区基于 GameGoData 训练更多轻量化 agent。

---

> 🔚 **总结一句话**：  
> **GameGo 证明了“结构化需求 + 高密度提示 + 合成轨迹训练”是一条通往高效、高质量游戏生成 agent 的可行路径，为 LLM 驱动的创意软件工程提供了新范式。**

</details>

---

### 8. [NCCL M2N: A Layout- and Topology-Aware Collective for Distributed Tensor Resharding](https://arxiv.org/abs/2610.07516)

**Authors**: Kaushik Kandadi, Youngeun Kwon, Sreeram Potluri, Ching-Hsiang Chu, Ke Wen, Pouya Kousha, Sangkug Lym, Nitin Nitin, Manjunath Gorentla Venkata  
**Category**: cs.DC  
**Published**: 2026-10-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.07516v1  

#### Abstract
Distributed training and rollout generation often use different tensor layouts, requiring model weights to be resharded across distinct process groups. This M-to-N redistribution is not directly expressed by standard collectives. Flat direct sends duplicate traffic across destination replicas, while...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**NCCL M2N: A Layout- and Topology-Aware Collective for Distributed Tensor Resharding**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
在分布式深度学习训练中，尤其是强化学习（Reinforcement Learning with Verifiable Rewards, RLVR）等场景，模型在不同阶段（如策略训练和 rollout 生成）使用不同的并行布局（parallelism configuration），导致需要频繁进行 **M-to-N layout redistribution**（即张量从一种分片布局重新分布到另一种）。  
这一过程称为 **tensor resharding** 或 **weight refit**，是跨进程组、跨布局的数据重分布操作。

然而，当前主流通信库（如 NCCL）缺乏对这种 **跨布局、跨组 resharding** 的原生支持，导致实践中采用低效的“拼凑”方案：
- **flat direct sends**：每个源 rank 向所有目标副本发送数据，造成网络流量成倍复制；
- **gather-then-broadcast**：先在一个根节点聚集全部数据，再广播，造成单点拥塞和全量传输浪费。

这些方法在网络带宽消耗和延迟上代价高昂。例如，在 256 GPU 上运行 DeepSeek-V3 时，传统 all-gather + broadcast 的权重同步占到了整个 RL 步骤时间的 **29.4%**。

---

### **提出了什么新方法或新思路**
本文提出 **NCCL M2N** —— 一个集成于 NCCL Extensions 中的新型集体通信原语（collective primitive），专为 **跨布局的分布式张量重分布** 设计。

其核心思想是：
> **从源端和目的端的 layout 描述符自动推导出最优传输计划，并结合硬件拓扑实现高效路由。**

#### 主要创新点包括：

1. ✅ **Layout-Derived Transfer Schedule**
   - 输入两个 layout 描述符（`(S, M_src, p_src)` 和 `(S, M_dst, p_dst)`）
   - 自动计算每个源 rank 到目标 rank 所需传输的精确字节范围（overlap regions）
   - 支持任意维度分片、复制因子变化、process mesh 不相交等情况

2. ✅ **Topology-Aware Hierarchical Routing**
   - **去重传输**：每份唯一数据只在网络上传输一次（per NVLink domain 仅传一份）
   - **负载均衡**：将接收任务分散到多个“leader” rank，避免单一 root 拥塞
   - **本地复制优化**：利用高速 NVLink 在目标域内完成副本扩散（fan-out），不占用外部网络

3. ✅ **统一抽象接口**
   - 提供 C 和 Python 接口（`nccl.m2n.reshard`），可直接嵌入现有框架（如 NeMo-RL）
   - 是首个将 layout 语义与通信拓扑联合优化的 NCCL 原生 collective

---

### **相比现有方法的优势**

| 方面 | Flat Direct Sends | Gather-Broadcast | **NCCL M2N（本文）** |
|------|-------------------|------------------|------------------------|
| 网络流量 | 高（重复发送至各副本） | 极高（整张量聚集+广播） | 最小（仅必要部分，无冗余） |
| 负载均衡 | 差（源端并发高） | 极差（root 单点瓶颈） | 好（多 leader 分担） |
| 拓扑感知 | 无 | 无 | 有（NVLink 内部复制，Ring 跨域转发） |
| 易用性 | 手动编码复杂 | 框架常用但低效 | 声明式 API，自动调度 |

---

## 2. 核心实验方法和设置

### **实验平台**
- 硬件：**NVIDIA GB200 NVL72 集群**
- 网络：**NDR InfiniBand（50 GB/s 单向带宽）**
- GPU 数量：最多 **256 GPUs**
- 软件栈：NCCL Extensions (`nccl_m2n`)，CUDA，PyTorch 生态兼容接口

---

### **实验设置与评估指标**

#### **微基准测试（Microbenchmark）**
- **任务**：单个 FFN-MoE 层的专家权重从 **Expert Parallelism (EP=16)** 转移到 **Tensor Parallelism (TP=8)** 布局
- **张量大小**：`[256, 2048, 7168]` BF16 → 总计 **7168 MB**
- **变量控制**：改变目标侧的 **DP 复制数**（replication count `r_d ∈ {2,4,8}`，对应 16–64 GPUs）
- **评估指标**：
  - 单次 reshard 操作延迟（latency）
  - 实测吞吐 vs. 理论 Speed-of-Light (SOL) 下界对比

#### **端到端实验（End-to-End）**
- **模型**：**DeepSeek-V3**（BF16 精度）
- **训练配置**：
  - Trainer：TP=1, PP=8, EP=16（共 128 GPUs）
  - Generator：TP=16, DP=8（共 128 GPUs）
  - 非共址部署（non-colocated），分布在 4 个 NVL72 域中
- **任务**：RL 训练中的 **weight refit** 阶段（策略更新后同步权重）
- **评估指标**：
  - 报告的 **weight-sync time**
  - 整体 **step time**
  - KL 散度（验证数值稳定性）

---

### **基线方法对比**
1. **Flat Direct Sends**
   - 每个源 rank 独立向每个目标副本发送所需片段
2. **Legacy All-Gather + Broadcast**
   - 先在 trainer 组内 all-gather 得到完整张量
   - 由某个 root 广播给所有 generator ranks
3. **NCCL M2N（本文方法）**
   - 使用 hierarchical route 实现 layout-aware 传输

---

## 3. 主要实验结果和性能指标

### **微基准测试结果**

| 目标复制数 `r_d` | Flat Direct Sends 延迟 | NCCL M2N 延迟 | 加速比 |
|------------------|-------------------------|---------------|--------|
| 2                | 20.86 ms                | 10.25 ms      | **2.0×** |
| 4                | 35.84 ms                | 13.83 ms      | **2.6×** |
| 8                | 77.26 ms                | 9.78 ms       | **7.9×** |

> 🔥 **最高达 7.9× 加速！**

- **理论分析匹配良好**：实测延迟接近聚合数据移动下界（Aggregate Data-Movement Bound）
- **flat direct sends 延迟随复制数线性增长**，而 **M2N 几乎恒定**（因仅传输一次核心数据）
- **NVLink 成为本地复制瓶颈前已被隐藏**，网络成为主导因素

---

### **端到端实验结果（DeepSeek-V3）**

| 指标 | Legacy (All-Gather + Bcast) | NCCL M2N | 提升 |
|------|------------------------------|----------|------|
| Weight-sync time | **5.78 s** | **2.77 s** | ↓ **52.1%**, **2.09× speedup** |
| Step time | **19.68 s** | **17.18 s** | ↓ **12.7%** |
| Weight sync 占比 | 29.4% of step | ~16.1% of step | 显著降低关键路径开销 |

- KL error 对比稳定（0.001869 vs 0.001912），表明未引入数值误差
- 性能提升直接转化为整体训练效率提升

---

### **消融实验与理论建模验证**
- 提出 **Aggregate Data-Movement Model** 来预测理论最优延迟：
  $$
  T_{\text{SOL}} = \max(T_{\text{network}}, T_{\text{NVLink}})
  $$
- 实验结果显示：
  - M2N 接近该下界（仅高出 9%-54%，主要来自 staging 开销）
  - flat direct sends 和 gather-bcast 远离下界
- 验证了 **topology-aware routing 的有效性**

---

## 4. 关键结论和发现

### **主要发现**
1. ✅ **M-to-N resharding 是现代 RL 训练的关键瓶颈**，传统通信模式无法有效应对。
2. ✅ **NCCL M2N 通过 layout-aware + topology-aware 联合设计，显著减少冗余通信**。
3. ✅ **层级路由机制实现了“一次传输、多地复制”**，充分利用 NVLink 和 RDMA 拓扑优势。
4. ✅ 在真实大规模场景下（256 GPUs），**M2N 将 weight-sync 时间缩短超过一半，整体 step time 下降 12.7%**。
5. ✅ 方法通用性强，适用于多种并行策略组合（TP/EP/DP/PP）之间的转换。

---

### **局限性**
1. ❗ 当前版本仅支持 **最多一个分片维度**（single sharded axis per layout）
2. ❗ 要求 source 和 destination process mesh **互不重叠**（disjoint）
3. ❗ 不支持运行时动态调整 communicator size（如弹性扩展）
4. ❗ staging buffer 引入额外内存开销（PACK 模式）

---

### **未来工作方向**
1. ✅ **支持多维分片（multi-dimensional sharding）**
   - 如同时支持 TP×EP×DP 的 native reshard
2. ✅ **弹性 communicator API 集成**
   - 支持 shrink/grow，适应动态 rollout cluster 规模变化
3. ✅ **自适应传输选择（adaptive transport selection）**
   - 根据 tensor size、contiguity、peer count 动态选择 PACK / PIPE / host-RMA
4. ✅ **支持 overlapping meshes**
   - 实现 inplace layout transition（无需重建 communicator）
5. ✅ **融合 on-the-fly quantization**
   - 在 staging kernel 中直接量化，节省带宽并省去中间缓冲区

---

## ✅ 总结

**NCCL M2N 是首个将 layout 语义与硬件拓扑深度融合的 reshard collective，填补了现代分布式训练中跨布局通信的空白。它不仅大幅提升了 weight refit 效率，也为未来高性能、声明式的分布式张量编程提供了基础设施支撑。**

> 🚀 “Not just another collective — it’s a paradigm shift in how we think about cross-layout data movement.”

</details>

---

### 9. [Activation Denoising: A Robustness View on Parallel vs Sequential LLM Quantization](https://arxiv.org/abs/2610.07522)

**Authors**: Yan Scholten, Rachel Lawrence, James Hensman, Stephan G\"unnemann, Alicia Curth, Riccardo Grazzi  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.07522v1  

#### Abstract
Post-training quantization is a powerful tool for compressing large language models. The most scalable methods quantize every layer in parallel, but quantization errors then compound through the residual stream, as no layer corrects for the errors of the layers before it. Sequential quantization acc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Activation Denoising: A Robustness View on Parallel vs Sequential LLM Quantization

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的**Post-training Quantization (PTQ)** 方法在效率和精度之间存在显著权衡：
- **Parallel PTQ**：所有层并行量化，效率高，但忽略了前序层的量化误差，导致误差在残差流中累积，影响模型性能。
- **Sequential PTQ**：逐层量化，每一层都基于已量化前层的输出进行校准，能有效补偿误差，但必须串行执行，成为大规模模型部署的瓶颈。

### 提出的新方法和新思路
本文提出 **Activation Denoising Quantization (ADQ)**，从**鲁棒性 (robustness)** 的视角重新审视并行与串行量化的差距。

- **核心思想**：不通过串行方式逐层测量和纠正上游误差，而是将上游的量化误差视为一种**噪声 (noise)**，并通过正则化使每一层对这种噪声具有鲁棒性。
- **具体实现**：
  1. **噪声注入 (Noise Injection)**：在完整的全精度模型上进行一次额外的前向传播，在每一层的输入激活上注入随机高斯噪声，并让噪声随模型传播。
  2. **去噪滤波器 (Denoising Filter)**：利用注入噪声后的统计信息（协方差 `H`、信号-噪声交叉协方差 `C`、噪声协方差 `N`），为每个权重矩阵构建一个线性的去噪滤波器 `F`。
  3. **预变换与量化 (Pretransform and Round)**：将原始权重 `W` 预变换为 `W*F`，然后在新的度量 `G` 下进行标准的 `metric-weighted rounding`。这个过程是完全并行的。

### 相比现有方法的优势
- **高效且准确**：在保持**完全并行**量化流程的同时，恢复了大部分串行量化带来的精度增益。
- **无推理开销**：去噪滤波器被吸收进权重变换中，推理时无需任何额外计算。
- **理论统一**：该框架将并行和串行量化目标统一起来，当使用真实的串行误差作为噪声时，其特例可退化为 GPTAQ 和 QEP 等先进方法。
- **与旋转互补**：与常见的正交旋转（如 QuaRot）不同，ADQ 会降低权重的 Frobenius 范数，两者效果可以叠加。

## 2. 核心实验方法和设置

### 数据集
- **校准集 (Calibration Set)**：用于收集激活统计信息，采用 **WikiText-2** 的训练集部分，共 128 条序列，每条长度为 2048。
- **测试集 (Test Sets)**：用于评估模型性能，包括 **WikiText-2**, **C4**, 和 **FineWeb** 的测试集。

### 实验设置和评估指标
- **模型**：在 **Llama-3.2-1B, Llama-3.2-3B, Llama-3-8B, Llama-2-13B** 四个不同规模的模型上进行实验。
- **量化方案**：
  - **Scalar Quantization**: 使用 **GPTQ** 在 3-bit (`W3`) 和 4-bit (`W4`) 上进行。
  - **Vector Quantization**: 使用 **QuIP#** 在 2-bit (`W2`) 上进行。
- **评估指标**：
  - **平均困惑度 (Average Perplexity)**：在三个测试集上的平均值。
  - **零样本准确率 (Zero-shot Accuracy)**：在 8 个常识推理任务上的平均准确率。
  - **性能提升比例 (Gap Closure)**：衡量 ADQ 方法相对于“纯并行”基线和“最强串行”基线之间性能差距的缩小比例。

### 基线方法对比
- **Parallel Baseline**：标准的并行量化，使用干净的激活进行校准。
- **Sequential Baseline**：
  - **Symmetric Correction**：对称校正（如 GPTQ 的串行版本）。
  - **Asymmetric Correction**：非对称校正（如 GPTAQ），是当前最强的串行基线。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
根据 **Table 1** 的核心结果：

#### 3-bit Scalar Quantization (GPTQ)
- ADQ 在四个模型上平均关闭了 **18%-31%** 的并行到串行的困惑度差距。
- 例如，在 Llama-3.2-1B 上，ADQ 将困惑度从并行的 17.07 降低到 16.51，而最强串行基线为 15.29。

#### 2-bit Vector Quantization (QuIP#)
- ADQ 的表现更为出色，甚至超越了串行基线：
  - 在 Llama-3.2-1B 和 3B 上，分别关闭了 **38%** 和 **86%** 的差距。
  - 在更大的 Llama-3-8B 和 Llama-2-13B 上，ADQ 的性能**超过了**串行非对称校正基线（Gap > 100%）。

### 消融实验结果
- **噪声机制分析 (A.1)**：实验证明，同时包含**局部注入**和**深度传播**的噪声模型效果最好，其中深度传播的噪声贡献了大部分收益。
- **噪声尺度分析 (A.2)**：最佳配置通常位于 `p < λ` 的区域，即“部分去噪”，这表明完全去除噪声（`p=0`）并非最优。
- **与旋转的组合 (6.2)**：ADQ 与随机或学习得到的旋转（如 SpinQuant）结合使用时，性能进一步提升，证明了两种技术的互补性。
- **效率分析 (6.1, A.8)**：ADQ 只需一次额外的前向传播，其时间开销远小于串行量化。在 Llama-2-13B 上，使用 8 个 GPU 进行投影，ADQ 的量化时间预计比串行基线快 **7.1-7.5倍**。

## 4. 关键结论和发现

### 主要发现
1. **量化误差可视为鲁棒性问题**：将串行量化带来的精度提升归因于对上游误差的鲁棒性，而非精确的误差反馈，这一视角转换是本文的核心洞见。
2. **高效的并行替代方案**：**Activation Denoising** 成功地在单次并行量化过程中，恢复了大部分串行量化的效果，极大地提升了低比特量化 LLM 的实用性和效率。
3. **性能与效率的帕累托前沿**：通过简单的策略（如仅对前几个块进行串行量化，或将少数关键模块提升到 4-bit），可以在极小的代价下几乎完全消除剩余的性能差距。

### 方法的局限性
- **仍存在性能差距**：尽管 ADQ 表现优异，但在某些设置下（尤其是 3-bit）仍未完全达到最强串行基线的水平。
- **超参数调优**：方法引入了新的超参数（如噪声尺度 `γ`, `λ`, `p`），需要进行调优，尽管文中指出这些参数在不同模型间相对稳定。

### 未来工作方向
- **改进噪声模型**：探索更复杂的、带有信号相关结构的噪声模型，以进一步逼近真实误差。
- **联合优化**：尝试联合学习噪声注入模式和权重旋转。
- **扩展应用范围**：将 ADQ 框架应用于激活 (activation) 和 KV Cache 的量化。
- **结合其他技术**：与更强的舍入算法（如 KronQ）或恢复微调（recovery fine-tuning）等正交技术结合。

</details>

---

### 10. [Enhancing Diffusion Language Models with Autoregressive Post-Training Weights](https://arxiv.org/abs/2610.08108)

**Authors**: Yiming Qin, Ke Wang, Amel Abdelraheem, Adam Hazimeh, Pascal Frossard  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.08108v1  

#### Abstract
Diffusion language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) language models, offering flexible token-update orders and parallel decoding. Recent dLLMs are often initialized from pretrained AR models before diffusion conversion in order to inherit their learned re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Enhancing Diffusion Language Models with Autoregressive Post-Training Weights

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Diffusion Language Models (dLLMs)** 虽然在生成效率和并行解码方面具有优势，但通常从预训练的 **Autoregressive (AR)** 模型初始化而来，却忽略了其丰富的 **AR post-training ecosystem**（如指令微调、强化学习对齐等）。这导致 dLLMs 需要从头进行昂贵的特定扩散范式的后训练（diffusion-specific post-training），而无法直接复用已有的 AR 后训练成果。

### 提出的新方法：A2D
作者提出了 **A2D (AR-to-Diffusion Transfer and Composition)**，一个无需训练的框架，用于将 AR 模型的后训练权重更新迁移到扩散模型中。该方法基于 **task arithmetic** 思想，通过在权重空间中操作来实现能力迁移。

A2D 包含两种模式：
- **A2D-Transfer**：当没有对应的扩散后训练模型时，直接将 AR 的任务向量 $T_A = A_p - A_b$ 加到扩散基础模型 $D_b$ 上，得到增强后的模型：  
  $D_{\text{Transfer}}(\alpha) = D_b + \alpha T_A$
- **A2D-Merge**：当已有扩散后训练模型 $D_p$ 时，将 AR 和扩散的任务向量进行插值融合，以结合两者的优势：  
  $D_{\text{Merge}}(\lambda) = D_p + \lambda (T_A - T_D)$

### 相比现有方法的优势
- **无需额外训练**：整个过程是 **training-free** 的，不增加任何训练成本或推理开销。
- **高效利用资源**：能够复用现有的、通常更强的 AR 后训练检查点（checkpoints）来提升扩散模型。
- **性能提升显著**：在多个任务上均能稳定提升性能，甚至优于单独的扩散后训练。
- **兼容性强**：适用于多种 dLLM 架构（如 Dream, DiffuCoder, Nemotron-Labs-Diffusion 等）和多种后训练类型（SFT, RL, 推理专项训练等）。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了多个下游任务，使用的数据集包括：
- **数学推理**：`GSM8K`, `MATH-500`, `AIME24/25`
- **代码生成**：`HumanEval+`, `MBPP+`, `BigCodeBench (BCB)`
- **指令遵循**：`IFEval`, `IFBench`
- **医学视觉问答**：`VQA-RAD`, `SLAKE`, `VQA-Med`
- **通用知识与推理**：`MMLU`, `GPQA`

### 实验设置和评估指标
- **模型家族**：实验涵盖了多个主流模型系列，包括基于 Qwen2.5/Qwen3 的 Dream/DreamReasoner，基于 Gemma 的 DiffusionGemma，以及双模态模型 Nemotron-Labs-Diffusion。
- **评估方式**：
  - 数学任务使用 **answer accuracy**。
  - 编程任务使用 **pass@1**（通过 EvalPlus 测试）。
  - 指令遵循任务使用 IFEval 的综合得分。
- **系数选择**：每个模型组合的缩放系数 $\alpha$ 或 $\lambda$ 在一个锚定任务（anchor benchmark）上搜索确定，并在其他任务上固定使用，以验证泛化性。

### 基线方法对比
- **AR donor**：提供任务向量的 AR 后训练模型。
- **Base ($D_b$)**：未经过后训练的扩散基础模型。
- **Instruct / Post-trained ($D_p$)**：经过标准扩散后训练的模型。
- **A2D-Transfer**：仅应用 AR 任务向量的扩散模型。
- **A2D-Merge**：融合 AR 与扩散任务向量的模型。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
| 模型 | 任务 | Base | Post-trained | A2D-Transfer | A2D-Merge |
|------|------|------|--------------|---------------|-----------|
| Dream-7B | IFEval | 49.7 | 63.0 | 59.2 | **72.8** |
| Dream-7B | MATH-500 | 20.8 | 38.4 | 40.2 | **45.0** |
| Dream-Coder | HumanEval+ | 57.9 | 77.4 | 67.1 | **79.3** |
| DiffuCoder | BCB-Full | 33.9 | 34.8 | 32.0 | **38.9** |
| DreamReasoner | AIME25 | 12.1 | 56.7 | 16.2 | **67.1** |

> ✅ **观察**：A2D-Merge 在所有 9/9 的基准测试中均优于对应的 post-trained 基线。

### 与基线方法的对比结果
- **A2D-Transfer** 显著提升了基础扩散模型的能力，在缺乏扩散后训练的情况下尤其有效（如 Dream-7B 在 MATH-500 上提升 +19.4）。
- **A2D-Merge** 进一步提升了已有的扩散后训练模型，在多个任务上取得 **SOTA-like** 表现，且增益跨任务可泛化。
- 即使 AR donor 本身弱于扩散后训练模型（如 Gemma → DiffusionGemma on SLAKE），A2D 仍能带来正向增益，说明 AR 更新提供了**互补性能力**。

### 消融实验结果
- **参数空间 vs 表征空间**：
  - AR 与扩散任务向量在参数空间中几乎正交（cosine similarity ≈ 0.01–0.07）。
  - 但在表征空间中引起的改变高度对齐，说明二者虽走不同路径，但达成相似功能。
- **模块贡献分析**（Table 14）：
  - A2D-Transfer 主要依赖 MLP 层的更新。
  - A2D-Merge 则需要 MLP 和 Attention 模块共同作用才能达到最优。
- **反向迁移失败**（Table 3）：
  - 将扩散任务向量加回 AR 基础模型会**降低性能**，表明迁移具有方向性（AR → Diffusion 成功，反之不行）。
- **持续预训练不可迁移**（Table 15）：
  - 大规模的 continued pretraining 向量（norm 大 70–82×）无法有效迁移，说明 A2D 更适合**局部的后训练更新**。

---

## 4. 关键结论和发现

### 主要发现
1. **跨范式可迁移性**：尽管 AR 与 diffusion 模型在训练目标和生成机制上存在根本差异，但 AR 的 post-training 权重更新在转换为 diffusion 模型后依然**保持有效性**。
2. **参数更新的“正交但对齐”现象**：AR 与 diffusion 的任务向量在参数空间中方向几乎正交，但在模型内部表征空间中引发的变化却高度一致，说明它们通过不同路径实现了相似的功能改进。
3. **互补性增益**：AR 与 diffusion 的后训练带来了**部分互补的行为提升**，通过线性插值可以同时保留两者的优点。
4. **A2D 的普适性**：该方法在多种 dLLM 架构、任务类型（SFT, RL）、领域（数学、编程、医疗）上均表现出色，且支持双模态模型（AR & diffusion decoding）。

### 方法的局限性
- **依赖共享训练谱系**：A2D 要求 AR donor 与 diffusion recipient 来自同一 AR 家族（如 Qwen → Dream），跨家族迁移尚未验证。
- **仅适用于局部更新**：大规模的 continued pretraining 不可迁移，可能因改变了底层表示结构。
- **参数映射挑战**：对于架构差异较大的模型（如 Ministral → Nemotron），需手动处理 token embedding 映射问题。

### 未来工作方向
- 探索更远距离或独立训练模型间的跨范式迁移。
- 设计更自适应的 composition 策略（如 layer-wise scaling, conflict resolution）。
- 将 A2D 作为后续 SFT 或 RL 的初始化方案，研究其对微调动态的影响。
- 扩展至多模态 diffusion 模型或其他生成范式。

> 🔚 **总结**：A2D 揭示了 AR 与 diffusion 范式之间深层次的权重空间联系，提出了一种简单、高效、无训练成本的方法来增强 dLLMs，为统一利用两大生态系统的后训练资源开辟了新路径。

</details>

---

### 11. [A Systematic Investigation of Bias in Large Language Models for Advertising Relevance](https://arxiv.org/abs/2610.07544)

**Authors**: Weiwei Wang, Yinchuan Xu, Jialu Gao, Youkow Homma, Jian Jiao  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.07544v1  

#### Abstract
Large language models (LLMs) are increasingly used to judge how well an advertisement matches a query, but the fairness of these judgments has received limited attention. We conduct a systematic study of fairness in relevance judgments made by LLMs for queries and advertisements. Our counterfactual ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《A Systematic Investigation of Bias in Large Language Models for Advertising Relevance》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文系统地研究了**Large Language Models (LLMs)** 在广告相关性判断（advertising relevance）任务中的**公平性偏差问题**。具体关注以下三类潜在偏见：
- **Advertiser identity and popularity bias**：不同品牌或广告主的身份是否影响模型对广告相关性的判断。
- **Input language bias**：相同语义内容在不同语言输入下是否导致不同的相关性评分。
- **Demographic bias**：查询中涉及性别、种族、年龄等人口统计学信息时，是否存在刻板印象相关的偏见。

这些问题直接影响广告系统的公平性，可能造成小品牌被低估、少数群体获得不平等服务等问题。

### 🚀 提出的新方法与新思路
作者提出了一种**统一的反事实框架（unified counterfactual framework）** 来系统评估上述三类偏见：
- 对于每个原始 query-ad pair，构造仅改变目标属性（如 advertiser 名称、语言、demographic 词）而保持其余内容不变的“反事实”样本。
- 通过比较模型在原始与反事实输入上的输出差异，量化模型对无关因素的敏感度。

该框架适用于两种主流 LLM 应用形式：
- **General-purpose LLMs**（如 GPT-4o）作为分类器输出 `good/fair/bad`；
- **Fine-tuned models**（如 Qwen-7B）输出 relevance probability (`pRel`)。

### 🔍 相比现有方法的优势
| 方面 | 本文优势 |
|------|---------|
| **综合性** | 首次将 advertiser bias、language bias 和 demographic bias 统一在一个框架下进行系统评估，填补了广告相关性场景中公平性研究的空白。 |
| **现实基础 + 控制变量结合** | 广告主和语言实验基于真实广告日志数据；demographic 实验采用受控合成查询，兼顾真实性与可解释性。 |
| **实用导向的缓解策略分析** | 不仅发现问题，还评估了 inference-time masking 和 training-data rebalancing 等实际可行的 mitigation 方法。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
| 类型 | 描述 |
|------|------|
| **Real-world query-ad pairs** | 从真实广告日志中采样：<br>- **零售类广告**：2,000 对 query-ad<br>- **招聘类广告**：878 对 query-ad<br>均排除明确提及品牌的 query，以保证替换 advertiser 不改变语义匹配性。 |
| **Multilingual translations** | 将英文 query-ad 对翻译为 **中文** 和 **芬兰文**，构建多语言反事实组。 |
| **Synthetic demographic queries** | 构造控制变量的合成 query，仅修改 demographic 属性（gender/race/age），广告内容固定，覆盖 employment、housing、credit 三大高风险领域。 |

### ⚙️ 实验设置
#### 模型配置
| 模型 | 类型 | 输出形式 | 是否确定性 |
|------|------|----------|------------|
| **GPT-4o** | General-purpose LLM | Categorical label: `{good, fair, bad}` | 否（多次运行取平均） |
| **Qwen-7B** | Fine-tuned model | Predicted relevance probability (`pRel ∈ [0,1]`) | 是（deterministic） |

#### 反事实变换设计
| 偏差类型 | 变换方式 |
|--------|--------|
| **Advertiser bias** | 替换广告标题、描述、URL 中的 advertiser 名称为预设集合中的其他品牌：<br>`{Retailer A~F}`, `{Job Platform A/B}` |
| **Language bias** | 英文 → 中文 / 芬兰文 翻译，保留语义一致性 |
| **Demographic bias** | 修改 query 中的人口统计词，如 `"Jobs for female"` vs `"Jobs for male"` |

### 📈 评估指标
| 模型类型 | 主要指标 |
|--------|--------|
| **GPT-4o** | - Bad Decision Rate (Bad DR): 被判为 "bad" 的比例<br>- Fair DR: 被判为 "bad" 或 "fair" 的比例（值越低表示判断更积极）<br>- Consistency: 多轮重复运行的一致性百分比 |
| **Qwen-7B** | - Mean predicted relevance probability (`pRel`)：越高表示相关性判断更强 |
| **Mitigation 效果** | - AUC（用于 company masking）<br>- 比较不同训练数据处理后的 pRel 分布变化 |

### 🔁 基线方法对比
本文并非提出新的建模方法，而是以现有典型模型为对象进行公平性分析：
- **GPT-4o** 代表通用 LLM 判断范式；
- **Fine-tuned Qwen-7B** 代表工业级专用 relevance model。

因此，“基线”体现在：
- 原始模型表现 vs. 反事实输入下的表现（衡量 bias）
- 原始训练数据 vs. 经过 downsampling/rebalancing 的训练数据（衡量 mitigation 效果）

---

## 3. 主要实验结果和性能指标

### 📌 关键性能数据汇总

#### ✅ Advertiser Identity and Popularity Bias
| 模型 | 观察结果 |
|------|--------|
| **GPT-4o** | - Retailer A（知名品牌）获得最低 Bad DR (14.6%)，显著优于其他零售商 (15.1–16.6%)<br>- Job Platform A 比 Job Platform B 更有利（Bad DR: 14.4% vs 15.3%） |
| **Qwen-7B** | - Retailer A 得到较高 pRel (59.6%)，仅次于 Retailer E (60.7%)<br>- Job Platform A 的 pRel (65.8%) > Job Platform B (63.1%) |

> ✅ 结论：**知名广告主普遍获得更多有利的相关性判断**，存在 advertiser popularity bias。

---

#### ✅ Input Language Bias
| 模型 | 英文 | 中文 | 芬兰文 | 差异显著性 |
|------|-----|-----|-------|-----------|
| **GPT-4o** | Bad DR: 8.0%<br>Fair DR: 44.2% | Bad DR: 7.1%<br>Fair DR: 40.5% | Bad DR: 11.2%<br>Fair DR: 45.6% | Chinese vs Finnish: `p < 0.001` |
| **Qwen-7B** | pRel: 66.7% | pRel: 54.4% | pRel: 71.3% | — |

> ✅ 结论：**语言变化显著影响相关性判断**。GPT-4o 对中文最友好，Qwen-7B 却对芬兰文最有利——说明 bias 模式因模型而异。

---

#### ✅ Demographic Bias（刻板印象检测）
##### 性别 × 职业（GPT-4o 示例）
| 查询 | 工程岗位 (Engineering) | 护士岗位 (Nurse) |
|------|------------------------|------------------|
| "Jobs for females" | `0:3:7`（多数为 bad） | `0:10:0`（全为 fair） |
| "Jobs for males"   | `1:9:0`（无 bad）     | `0:0:10`（全为 bad） |

> 明显体现“工程师=男性”、“护士=女性”的社会刻板印象。

##### 种族 × 房产（Qwen-7B 示例）
| 查询 | 豪宅 (Luxury houses) | 低价房 (Low-cost houses) |
|------|--------------------|------------------------|
| White people | pRel: 0.625 | pRel: 0.611 |
| Black people | pRel: 0.538 | pRel: 0.819 |

> 黑人群体被认为更适合低价住房，反映潜在歧视性关联。

---

#### 🛠️ Bias Mitigation 实验结果

##### （1）Inference-time: Company Name Masking
| 查询类型 | 指标 | 原始 | 隐藏公司名 | 变化趋势 |
|--------|------|------|------------|----------|
| **Non-company-targeting** | Qwen-7B AUC | 0.814 | 0.814 | 几乎无损 |
| **Company-targeting**      | Qwen-7B AUC | 0.804 | 0.766 | 显著下降 |
| **GPT-4o（两类）** | Accuracy | ~60% | ↑ 至 ~64% | 未观察到准确率损失 |

> ✅ 发现：**当 query 不针对特定公司时，隐藏 advertiser 名字不会损害性能，且有助于缓解 bias**；但若用户明确搜索某品牌，则应保留名称。

##### （2）Training-time: 数据重平衡
| 训练策略 | Retailer A pRel | 排名变化 |
|--------|------------------|----------|
| 原始训练数据 | 59.6% | 第二（低于 Retailer E） |
| 随机删除一半 Retailer A 样本 | 61.7% | 仍高于大多数 |
| 删除一半 Retailer A 的 good/fair 样本（label-aware rebalancing） | 59.7% | 下降至与 Retailer C/F 相当 |

> ✅ 发现：**简单减少样本数量无效，必须按标签分布调整才能有效降低品牌偏好**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLMs 在广告相关性判断中普遍存在多种偏见**：
   - 广告主身份（尤其是知名度）会影响判断结果；
   - 输入语言不同会导致相同内容获得不同评分；
   - 查询中的人口统计词汇会触发刻板印象（如男=工程师，女=护士）。

2. **偏见存在于通用 LLM 和专用 fine-tuned 模型中**：
   - GPT-4o 和 Qwen-7B 均表现出类似趋势，表明这不是个别现象，而是系统性风险。

3. **有效的缓解策略需考虑上下文**：
   - **Company name masking** 在非品牌查询中安全有效，但在品牌查询中会损害性能。
   - **Training data rebalancing** 必须基于标签分布（label-aware），而非随机抽样，才可削弱品牌优势。

4. **语言 bias 模式复杂**：
   - GPT-4o 和 Qwen-7B 对不同语言的偏好相反（如中文 vs 芬兰文），提示不能假设某种语言天然更“公平”。

---

### ⚠️ 方法的局限性
- **反事实构造依赖人工设计**：demographic query 为合成数据，可能无法完全反映真实用户表达方式。
- **品牌匿名化限制分析深度**：出于保密原因，无法公开具体 advertiser 名称及其市场地位细节。
- **仅评估两种模型**：结论是否泛化至其他 LLM（如 Claude、Llama）尚待验证。
- **未引入用户行为反馈**：实验基于静态 judgment，未模拟最终广告展示带来的长期不公平效应。

---

### 🔮 未来工作方向（原文建议）
1. **系统研究训练数据构成的影响**：
   - 探索不同 advertiser 示例比例和 label 分布如何权衡 **fairness** 与 **predictive performance**。
2. **开发针对性的去偏算法**：
   - 如 fairness-aware fine-tuning、adversarial debiasing 等在 relevance modeling 中的应用。
3. **扩展多语言与跨文化公平性研究**：
   - 检查更多语言对之间的 bias 模式，探索文化背景对 judgment 的影响。
4. **建立标准化的 LLM-based 广告公平性 benchmark**：
   - 提供公开测试集和评估协议，推动行业共同应对 bias 挑战。

--- 

> 💡 **总体评价**：  
> 本文是首个系统性评估 LLM 在广告相关性任务中多重偏见的工作，兼具理论严谨性和工程实用性。其提出的反事实框架和 mitigation 分析为构建更公平的 AI-driven 广告系统提供了重要参考。

</details>

---

### 12. [Algorithmically Aligned Neural Agglomerative Tree Construction](https://arxiv.org/abs/2610.07271)

**Authors**: Robert R Nerem, Pranav Singh, Cheyenne Ward, Yusu Wang  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.07271v1  

#### Abstract
Linkage algorithms for hierarchical clustering (HC) are a powerful and efficient framework for constructing clustering trees, yet it is often unclear which merge rule best suits a given dataset or task. In contrast, neural approaches can learn from data, but often fail to retain the efficiency and s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Algorithmically Aligned Neural Agglomerative Tree Construction**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统的层次聚类（Hierarchical Clustering, HC）依赖于手工设计的**linkage算法**（如 Single Linkage、Complete Linkage），这些方法虽然高效且可解释，但存在以下局限：
- **固定规则**：无法根据数据或任务自适应调整合并策略。
- **局部最优**：难以捕捉复杂几何结构（如环形与圆形混合）所需的局部依赖规则。
- **目标不匹配**：许多工程和科学任务的目标函数（如时钟树布线中的零偏斜）无法被标准 linkage 函数有效优化。

神经网络方法虽能从数据中学习灵活的表示和规则，但通常缺乏传统算法的**递归结构**和**规模泛化能力**（size generalization），导致在更大输入上的推理效率低或性能下降。

---

### **提出的新方法：NN-LINKAGE**
本文提出 **NN-LINKAGE**，一种将神经网络与经典 Lance-Williams (LW) linkage 框架进行**算法对齐**（algorithmic alignment）的新型模型。

#### **核心思想**
- 将传统 LW recurrence 中的**标量相似度分数**替换为**可学习的 merge embeddings**。
- 使用共享的 MLP（`MLPinit`, `MLPupdate`）来初始化和更新这些嵌入，替代 LW 的线性组合公式。
- 保留优先队列（priority queue）结构以维持 $O(n^2 \log n)$ 时间复杂度。

#### **关键创新**
1. **算法对齐设计**：
   - 显式模仿 LW linkage 的递归合并过程，使模型具备良好的归纳偏置（inductive bias）。
   - 支持高效的 inference 和跨规模泛化。

2. **表达能力强**：
   - 可近似任意连续的 linkage 函数（定理4）。
   - 若使用 Transformer 编码器，可建模全局依赖（如 robust single linkage）。
   - 可精确实现所有对称常系数 LW recurrence（定理5）。

3. **监督学习机制**：
   - 利用目标树（target tree）作为监督信号，通过 pairwise hinge loss 学习合并顺序。
   - 不强制特定得分值，仅关注相对排序。

---

### **相比现有方法的优势**
| 方面 | 传统 linkage | 现有神经方法（如 NeuralNJ） | **NN-LINKAGE** |
|------|--------------|-------------------------------|----------------|
| **灵活性** | 固定规则 | 可学习全局表示 | 可学习局部+全局依赖的动态规则 |
| **效率** | 高效（$O(n^2\log n)$） | 通常较慢（需全图注意力） | 维持高效优先队列结构 |
| **规模泛化** | 天然支持 | 有限（依赖训练分布） | 强（理论保证 + 实验验证） |
| **可解释性** | 高 | 低 | 中等（仍基于可追踪的 merge 步骤） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **合成数据集**
   - **Rings-and-Disks**：包含两个同心环和两个实心圆，测试模型能否根据不同几何结构自动切换 SL/CL 策略。
   - **Uniform Point Clouds**：用于 clock-tree routing 实验，在 $[0,1]^2$ 上随机采样点，训练集大小为 20，测试至 1000。

2. **真实世界基准**
   - **VLSI Benchmarks**：BST-DME 和 ISPD 数据集，共 20 个芯片设计实例，用于 clock-tree routing 评估。
   - **Phylogenetic Data**
     - 合成 MSA（Multiple Sequence Alignment）：基于模拟进化生成，涵盖不同 taxa 数（20–100）和序列长度（256–1024）。
     - 真实生物数据集：JarvD5a, SongD1, TarvD7, WickD3b（来自 Zhou et al., 2018）。

---

### **实验设置与评估指标**

| 任务 | 输入 | 输出 | 主要指标 | 训练方式 |
|------|------|------|----------|-----------|
| **Clock Tree Routing** | 二维坐标点集 | HC 树 → 最小化 L1 diameter sum | 相对于 CL 的目标值改进百分比 | 监督训练，目标为最优 DimSum 动态规划解（n=20） |
| **Phylogenetic Reconstruction** | MSA 序列矩阵 | 进化树拓扑 | Normalized Robinson-Foulds (RF) Distance | 监督训练，目标为模拟生成的真实树 |
| **Rings-and-Disks** | 120维点云 | 四子树划分 | 点分类错误率 | 监督训练，目标为理想合并路径 |

#### **基线方法对比**
- **Classical Linkage**: Single Linkage (SL), Complete Linkage (CL)
- **Learned Rule Baseline**: $\alpha$-linkage（Balcan et al., 2019）——SL 与 CL 的凸组合
- **Neural Methods**:
  - **NeuralNJ**（Zhang et al., 2025）：当前 SOTA 神经系统发育推断模型
  - **BIONJ**, **RAxML-NG**：经典距离法与最大似然法

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) Rings-and-Disks 实验**
| 方法 | 分类错误率（n=120） |
|------|--------------------|
| **NN-LINKAGE** | **0.40%** |
| $\alpha$-linkage (best) | 14.82% |
| SL | 14.82% |
| CL | 24.58% |

> ✅ **结论**：NN-LINKAGE 几乎完美恢复四簇结构，并被发现**在环区域采用类似 SL 的近距离优先合并，在圆盘区域采用类似 CL 的紧凑性优先合并**，实现了真正的“局部自适应”。

---

#### **(2) Clock Tree Routing**

##### **规模泛化（Synthetic Points）**
- 在训练于 **n=20** 的情况下，NN-LINKAGE 在 **n ≤ 500** 上均优于 CL。
- 平均目标值降低达 **0.6%**，最高达 **1.1%**（见 Table 4）。
- 在一个四点反例中，CL 成本为最优的 **1.25倍**，而 NN-LINKAGE 完美恢复最优解（Fig. 9）。

##### **VLSI 基准测试**
- 作为 CL 的后处理优化器，平均提升 **0.829%**，运行时间约 3.2 秒。
- 与局部 refine 结合时（CL+k），NN-LINKAGE 扩展了 Pareto frontier：
  - 在更高 refine level 下，以更少时间获得更大收益（如 k=14 模型 vs k=17 CL）。

---

#### **(3) Phylogenetic Reconstruction**

| 设置 | 方法 | 平均 RF | 运行时间 (s) |
|------|------|--------|-------------|
| **Synthetic** (1,152 instances) | **NN-LINKAGE** | 0.1950 | **0.156** |
| | NeuralNJ | 0.1716 | 0.595 |
| | BIONJ | 0.3552 | 0.158 |
| | RAxML-NG | 0.2540 | 10.967 |
| **Real Data** (26,136 instances) | **NN-LINKAGE** | **0.6075** | **0.339** |
| | NeuralNJ | 0.6221 | 1.701 |

> ✅ **结论**：
> - 在合成数据上略逊于 NeuralNJ，但速度快 **~4×**。
> - 在真实数据上**准确率更高**（RF 更低），速度 **~5× 快**。
> - 表现出更强的 **out-of-distribution generalization** 能力。

---

### **消融实验（Ablation Studies）**
- **Transformer vs MLP Leaf Encoder**：使用 Transformer 显著提升对 robust SL 等全局依赖规则的学习能力。
- **Margin Loss 设计**：利用 priority queue 实现高效负样本筛选，避免枚举所有候选对。
- **Merge Embedding Dimension**：适当维度即可取得良好效果，参数量可控。

---

## **4. 关键结论和发现**

### **主要发现**
1. **NN-LINKAGE 成功实现了算法对齐下的神经化升级**：
   - 既保留了传统 linkage 算法的高效性和递归结构，
   - 又获得了神经网络的数据驱动、任务定制化优势。

2. **能够学习复杂的局部依赖合并策略**：
   - 如在 Rings-and-Disks 中自动区分环状与团状结构并应用不同规则。

3. **具备强大的规模泛化能力**：
   - 在 clock-tree routing 中，从 n=20 泛化到 n=1000 仍保持竞争力。

4. **在多个领域达到 SOTA 或 Pareto 最优**：
   - 在 phylogenetics 上兼顾精度与速度；
   - 在 VLSI routing 上超越经典 CL 并扩展优化边界。

---

### **方法的局限性**
- **依赖高质量监督信号**：需要已知目标树（如通过 DP 求解的小规模最优解或模拟演化树）。
- **对极端分布偏移敏感**：在某些 VLSI 实例上表现不稳定，需结合 refine 策略。
- **目前仅适用于 agglomerative HC**：未覆盖 divisive 或其他非自底向上范式。

---

### **未来工作方向**
1. **拓展至 NP-hard 目标函数**：
   - 如 Dasgupta’s HC cost，探索是否能超越 average linkage 的 1/3-approximation。

2. **研究训练数据与归纳偏置的关系**：
   - 是否可通过精心构造少量小规模实例，确保学到特定 merge rule 并泛化？

3. **推广至其他递归算法框架**：
   - 如将类似思想应用于 MST、Steiner Tree（已有 NN-Steiner 工作）、动态规划等。

4. **引入不确定性建模**：
   - 当监督树噪声较大时，如何鲁棒学习？可考虑概率输出或 ensemble。

---

> 📌 **总结一句话**：  
> **NN-LINKAGE 是一次成功的“神经算法协同设计”实践——它不是简单地用神经网络替代算法模块，而是深入理解经典算法的计算结构（LW recurrence + priority queue），并通过神经组件对其进行可学习的增强，在效率、泛化与性能之间取得了优异平衡。**

</details>

---

### 13. [Neuromotor Hierarchy Network: Physiological Inductive Biases for Robust Generalization in sEMG Decoding](https://arxiv.org/abs/2610.07713)

**Authors**: He Wang, Hongyuan Qi, Zhaoxian Zhang, Jinbin Luo, Linyi He, Mehul Motani, Changsheng Wu  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.07713v1  

#### Abstract
Surface electromyography (sEMG) provides a wearable, noninvasive interface to neuromuscular activity for movement decoding and human-computer interaction. Population-scale decoding remains difficult because the relationship between sEMG and neuromuscular activity varies across users and sessions, wh...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Neuromotor Hierarchy Network: Physiological Inductive Biases for Robust Generalization in sEMG Decoding

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **跨用户与跨会话泛化难题**：surface electromyography (sEMG) 信号在不同用户和不同记录会话之间存在显著变异性（如电极位置、皮肤接触、增益等），导致模型难以实现鲁棒的跨个体解码。
- **任务相关神经肌肉协调建模不足**：现有方法通常直接学习从 sEMG 波形到输出标签的映射，忽略了对生理上一致的**潜在线索（latent neuromotor state）** 的显式建模，限制了泛化能力。

### 提出的新方法与思路
作者提出 **Neuromotor Hierarchy Network (NHN)**，一种受神经运动系统组织启发的新型 sEMG 解码架构，其核心思想是：
- 显式地从 sEMG 中推断一个紧凑的、任务对齐的 **latent neuromotor state**，该状态代表了控制手部动作的底层神经肌肉协同模式。
- 构造过程遵循生理层级结构：  
  `sEMG → 测量适配 → 时空编码 → 候选驱动 → 时间积分 → 分级分配 → latent neuromotor state`

#### 关键创新组件：
| 组件 | 功能与生理依据 |
|------|----------------|
| **Measurement Adaptation** | 自适应调整波形统计特性（均值、协方差），同时保留相对强度信息（intensity preservation），模拟生物信号处理中的归一化机制。 |
| **Spatiotemporal Encoder** | 使用 TDS blocks 进行时间卷积，并引入参数高效的局部-全局通道交互；通过 multi-timescale adaptive gain 调制特征，反映不同时间尺度的共同输入（common drive）。 |
| **Non-negative Candidate Drives** | 投影为非负驱动，防止积分时相互抵消，符合 motor unit 驱动的生物学约束。 |
| **Temporal Integration** | 使用因果 FIR 滤波器对候选驱动进行异质性时间整合，模拟突触输入的时间累积效应。 |
| **Henneman-inspired Graded Allocation** | 受 Henneman 大小原则启发，基于当前驱动强度动态分配每个 motor primitive 的权重，实现行为依赖的资源调度。 |

### 相比现有方法的优势
- **更强的泛化能力**：通过分离“测量变异”与“神经协调”，提升跨用户、跨阶段的鲁棒性。
- **更高的参数效率**：相比主流方法减少近 50%–66% 参数量。
- **统一架构支持多任务**：同一骨干网络可用于连续姿态估计（pose）和离散打字识别（typing）。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 任务 | 内容 |
|-------|------|------|
| **emg2pose** | 连续 hand-pose 估计 | 包含 16 通道腕部 sEMG 和同步的手指关节角度，用于回归预测 3D 手势。 |
| **emg2qwerty** | 触摸打字识别 | 包含自然打字场景下的 sEMG 与对应按键序列，用于字符序列识别。 |

### 实验设置与评估指标
| 任务 | 输入 | 输出 | 评估协议 | 主要指标 |
|------|------|--------|----------|---------|
| **Pose Estimation** | 16-ch @ 2kHz, 5s 窗口 | 关节角序列 | 用户划分（User）、阶段划分（Stage）、User×Stage 划分 | Angular Error (AE, °), Landmark Distance (LD, mm) |
| **Typing Recognition** | 同上，双侧手腕分别处理 | 字符概率序列 | Zero-shot + Fine-tuning on held-out users | Character Error Rate (CER%) under Greedy / Beam Search + LM |

- **训练细节**：
  - 使用 AdamW 优化器，余弦退火学习率。
  - 所有结果报告 5 次随机种子平均后的均值 ± 标准差。
  - Backbone 输出维度 $D=64$，primitive 数 $N=32$。
- **效率评估**：报告总参数量（Params.）和每段推理所需的浮点运算数（FLOPs）。

### 基线方法对比
| 类型 | 对比模型 |
|------|--------|
| **Pose Baselines** | emg2pose, vemg2pose, NeuroPose, Sensing Dynamics, Hadidi et al. (Pos/Vel, ST/MT) |
| **Typing Baselines** | TDS ConvNet, SplashNet (Split/Mini/Upscale), Distilled Transformers |
| **特别说明**：Transformer 类虽性能强，但计算开销极大（>9000 GFLOPs），不具备可比性，仅作参考。 |

---

## 3. 主要实验结果和性能指标

### 在 emg2pose 上的表现（Regression & Tracking）
| 模型 | User AE↓ | Stage AE↓ | User×Stage AE↓ | Params. (B) | GFLOPs |
|------|--------|----------|---------------|------------|--------|
| **vemg2pose** | 12.24° | 15.22° | 15.63° | 5.98 | 3.72 |
| **Pos-MT (Hadidi)** | 11.54° | 14.02° | 14.58° | 5.97 | 3.72 |
| **NHN (Ours)** | **11.38°** | **13.78°** | **14.22°** | **3.08** | **2.40** |

- **优势总结**：
  - 在所有三个泛化划分中，AE 平均降低 **0.52% ~ 2.84%**。
  - 相比 Hadidi 最佳变体（Vel-MT for Reg, Pos-ST for Track），**参数减少 48.42%~48.51%**，FLOPs 减少约 35.6%。
  - 泛化差距越大（如 Stage/ User×Stage），NHN 提升越明显。

### 在 emg2qwerty 上的表现
| 设置 | 模型 | TDT Greedy CER↓ | TDT Beam CER↓ | Params. (M) | GFLOPs |
|------|------|------------------|----------------|-------------|--------|
| **Zero-shot** | SplashNet-Upscale | 44.78% | 35.67% | 2.58 | 71.38 |
| | **NHN (Ours)** | **38.62%** | **28.75%** | **0.88** | **17.85** |
| **Fine-tuned** | SplashNet-Upscale (Shared) | 12.39% | 5.51% | 2.58 | 71.38 |
| | **NHN (Ours)** | **9.41%** | **3.83%** | **0.88** | **17.85** |

- **关键提升**：
  - Zero-shot Beam CER 下降 **19.40%**（相对）。
  - Fine-tuned Beam CER 下降 **30.42%**（相对）。
  - 参数仅为 SplashNet-Upscale 的 **34.14%**（减少 65.86%），计算量减少 **74.99%**。
  - 在所有设置下均优于所有 SplashNet 变体及 TDS ConvNet。

### 消融实验与分析（Learned Organization）
- **Allocation 重要性验证**：
  - 若采用 uniform allocation（各 primitive 权重相等），则：
    - Pose AE ↑ 3.94°
    - Typing Greedy CER ↑ 42.12 pp
  - 表明预测头依赖于 latent state 在 primitives 上的分布模式，而非仅总激活强度。

- **Temporal Behavior 学习差异**：
  - **Pose**：训练后 adaptive gain 的时间尺度变长，强调更长上下文；integration lag 缩短，偏好近期驱动。
  - **Typing**：中间时间尺度（M）增强，短/长时间尺度减弱，体现对瞬态事件的敏感性。
  - 说明 NHN 能在同一架构内自适应学习任务特定的时间偏好。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **生理先验可有效指导表示学习**：将神经运动通路的层级结构（测量→编码→整合→分配）转化为网络设计，显著提升了 sEMG 解码的泛化性和效率。
2. ✅ **compact latent neuromotor state 是有效的抽象**：该状态捕捉了跨用户的共享运动协调模式，在 pose 中表现为分布式关节关联，在 typing 中表现为重复按键的稳定 primitive 组合。
3. ✅ **动态分配机制至关重要**：Henneman-inspired graded allocation 明确建模了 motor primitive 的优先级切换，实验证明其对最终性能有决定性影响。
4. ✅ **高泛化 + 高效率兼得**：NHN 不仅性能领先，且参数和计算成本大幅低于当前最优方法，适合部署于资源受限的可穿戴设备。

### 局限性
- ❗ **learned components ≠ 生理真实源**：primitives 是任务驱动的学习表征，不能直接解释为真实的 motor units 或 synergies。
- ❗ **仅限腕部 sEMG 验证**：未测试其他传感器布局或肢体部位的迁移能力。
- ❗ **长期跟踪误差累积**：在固定 horizon 的 tracking 任务中，误差随时间增长，尽管仍优于 baseline。

### 未来工作方向
- 探索 NHN 在其他生理信号（如 EEG, ECoG）中的应用。
- 结合解剖学 prior（如肌肉拓扑）进一步约束 spatial mixing 模块。
- 开发面向实际设备部署的轻量化版本与实时推理方案。
- 将 latent neuromotor state 用于个性化适配或零样本迁移的新范式。

--- 

> **总结一句话**：  
> NHN 成功将 **neuromotor physiology** 转化为深度网络的 inductive bias，实现了 **parameter-efficient、robust-generalizing** 的 sEMG 解码新范式，为下一代可穿戴人机接口提供了理论与技术基础。

</details>

---

### 14. [RA-MoWE: Workflow-Affinity Embeddings for Query Clustering and Agentic Workflow Generation](https://arxiv.org/abs/2610.07851)

**Authors**: Qi Cheng, Shengyu Chen, Wei Cheng, Yiqun Xie, Haoyu Wang, Haifeng Chen, Xiaowei Jia  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.07851v1  

#### Abstract
Agentic workflows enable large language models (LLMs) to solve complex tasks by coordinating reasoning, tool use, and verification. However, a workflow optimized for an entire task collection can overlook differences in the reasoning strategies that individual queries need, while searching for a new...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RA-MoWE: Workflow-Affinity Embeddings for Query Clustering and Agentic Workflow Generation  
**核心结论与实验总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Agentic workflows** 在处理复杂任务时面临两个矛盾：
- **全局优化**（如对整个任务集合搜索最优 workflow）会忽略不同 query 对推理策略的差异化需求；
- **逐 query 生成 workflow**（如 FlowReasoner）虽能定制化，但每次都需要昂贵的在线搜索，计算成本高。

因此，如何在 **个性化** 与 **可复用性** 之间取得平衡，是关键挑战。

---

### 🚀 提出的新方法：RA-MoWE 框架

RA-MoWE（**R**eusable **A**gent workflows via **Mo**dular **W**orkflow **E**xperts）提出一种基于 **workflow-affinity embeddings** 的新范式，实现“**一次构建、多次复用**”的专家 workflow 体系。

#### 核心思想
- 不再依赖语义相似性（semantic similarity）来聚类 query，而是根据 query 对一组固定参考 workflow（称为 **probes**）的实际响应效果进行聚类。
- 这种响应效果被编码为 **workflow-affinity embedding**，反映的是“哪些推理策略有效”，而非“query 内容是否相似”。

---

### 🔧 创新点

| 贡献 | 说明 |
|------|------|
| **1. Workflow-Affinity Embeddings** | 将每个 query 表示为其在多个 reference workflows 下的任务得分向量。该向量揭示了 query 所需的计算特性（如是否需要采样、自修正等），用于后续聚类和专家选择。 |
| **2. Embedding-Guided Expert Generation** | 每个 cluster 基于其平均 affinity profile 初始化一个初始 workflow，并通过 execution feedback 迭代优化，生成可复用的“专家 workflow”。 |
| **3. Embedding Encoder for Zero-Shot Selection** | 训练一个轻量级 encoder（如 MLP），直接从 query 文本预测其 affinity embedding，从而避免在线执行所有 probes 来判断归属，实现高效部署。 |

---

### ⚖️ 相比现有方法的优势

| 方法类型 | 缺陷 | RA-MoWE 如何改进 |
|--------|------|----------------|
| 全局 workflow 搜索（如 AFlow） | 忽略 query 差异，泛化能力弱 | 通过聚类识别不同计算需求群体，分别优化 |
| 逐 query 生成（如 FlowReasoner） | 推理开销大，无法复用 | 生成可复用专家，大幅降低 inference 成本 |
| 语义聚类（如 BGE/Qwen3） | 语义相似 ≠ 推理策略相同 | 使用行为响应（affinity）聚类，更贴近实际计算需求 |

---

## 2. 核心实验方法和设置

### 📚 数据集

- **MixBench-H**: 包含 600 训练 + 300 测试 query，来自四个领域：
  - **AIME**（数学）
  - **GPQA**（科学问答）
  - **CodeContests**（编程）
  - **LiveCodeBench**（代码生成）
- **SWE-bench Verified**: 软件工程修复任务，300/200 划分训练/测试集。

模型使用：`GPT-4o-mini` 和 `Claude Haiku 4.5`。

---

### 🎯 实验设置与评估指标

| 组件 | 设置 |
|------|------|
| **Reference Workflows (Probes)** | 共 12 种，包括：<br>- Direct<br>- CoT<br>- CoT-SC (self-consistency)<br>- Self-refine<br>- ReAct<br>- Debate 等 |
| **Affinity Embedding 构造** | 每个 probe 执行 3 次，取平均得分作为 embedding 一维；共 12 维；标准化后用于聚类 |
| **聚类方法** | k-means（k=6）在 affinity embedding 空间中聚类 |
| **Expert 生成机制** | 基于 cluster 的平均 embedding 初始化 workflow，使用 LLM optimizer 提出修改建议，保留提升验证集分数的变更 |
| **Embedding Encoder** | 使用冻结的 BGE 文本特征 + MLP 预测 affinity embedding，监督信号为实测 embedding |
| **部署流程** | 新 query → Encoder 预测 embedding → 最近 cluster → 执行对应 stored expert |

---

### 📊 评估指标

| 指标 | 定义 |
|------|------|
| **Task Score** | 正确率（math/science）、代码通过率（coding）、harness resolution（SWE） |
| **LLM Calls per Query** | 推理阶段调用次数，衡量成本 |
| **Utility vs Cost Trade-off** | 综合考虑性能与资源消耗 |
| **Ablation Studies** | 分析 panel 大小、聚类方式、encoder 效果等影响 |

---

### 🆚 基线方法对比

| 类别 | 基线方法 |
|------|--------|
| **Workflow Search** | AFlow, ADAS, ScoreFlow |
| **Per-Query Generation** | FlowReasoner |
| **Routing Methods** | MasRouter, FrugalGPT |
| **Clustering Baselines** | Semantic BGE/Qwen3 clusters, Random clusters |
| **Oracle & Controls** | Train-selected probe, Direct execution, Global comparator (D) |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（MixBench-H, GPT-4o-mini）

| 方法 | 平均任务得分 | LLM Calls / Query | 提升 vs Reference |
|------|-------------|--------------------|------------------|
| **Direct** | 0.2872 | 1.00 | — |
| **Train-selected probe** | 0.3481 | ~3 | — |
| **Global comparator (D)** | 0.2722 | — | — |
| **RA-MoWE (measured affinity, A)** | **0.3358** | 2.403 | +6.36 pts vs D |
| **RA-MoWE (predicted affinity, B)** | **0.3358** | **2.113** | **+4.04 pts**, **-27.7% calls** |

> ✅ **核心结果**：RA-MoWE 在 **仅使用 2.113 次 LLM 调用/查询** 的情况下，相比最佳静态 probe 提升 **4.04 个百分点**，同时比 baseline 减少 **27.7% 的推理成本**。

---

### 🔁 与基线方法对比（Table 1）

| 方法 | GPT-4o-mini 得分 | 是否显著优于 D |
|------|------------------|----------------|
| FlowReasoner (search-only) | 0.3565 | 是（但需 outcome feedback） |
| Semantic BGE clusters | 0.2939 | 否 |
| Random clusters | 0.2767 | 否 |
| **RA-MoWE (B)** | **0.2922** | **否（未达显著）** |
| **RA-MoWE (A)** | **0.2742** | **是（Haiku 上显著）** |

> ⚠️ 注意：RA-MoWE(B)（纯文本预测）在 GPT-4o-mini 上未显著胜出，但在 Haiku 上表现优异（+0.0717），说明效果受 backbone 影响。

---

### 🔍 消融实验结果

#### （1）Affinity 聚类 vs 语义聚类
- **Affinity-defined specialists** 显著优于语义聚类（BGE/Qwen3）和随机划分。
- 在 SWE 上，affinity 专家解决 **49/200** 实例，远超 global D 的 **35/200**。

#### （2）Panel Size 影响（8 vs 12 probes）
- 增加 probe 数量从 8 到 12：
  - distinct fingerprints 从 281 → 329
  - effective rank 从 2.67 → 3.08
  - 提供更丰富的行为描述，有助于聚类

#### （3）Operator Ablation（添加 `Programmer` 操作符）
- 添加新 operator 后，十二探针版本得分从 0.2922 → 0.3153（+0.0231）
- 证明扩展操作空间可进一步释放潜力

#### （4）Encoder 预测质量
- Encoder 的预测 embedding 与真实 embedding 的 **cosine similarity ≈ 0.398**
- 但 cluster assignment 准确率仅为 **37.7% (GPT-4o-mini)** 和 **12.0% (Haiku)**
- 表明当前 encoder 仍有提升空间，预测误差尚未完全跨越决策边界

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Workflow Affinity 是有效的 query 表示方式**
   - query 对不同 workflows 的响应模式具有重复性和区分性，可用于指导专家 workflow 构建。
   - “行为相似性”比“语义相似性”更能反映计算需求。

2. **可复用专家 workflow 可实现高性能与低成本平衡**
   - RA-MoWE 生成的专家 workflow 在多个 benchmark 上优于全局 workflow 和简单路由策略。
   - 通过离线构建 + 在线快速匹配，实现了 **zero-shot workflow selection without probe execution**。

3. **Embedding-Guided Search 有效**
   - 初始 workflow 由 cluster affinity 初始化，能捕获大部分最终性能。
   - 局部 refinement 可进一步提升表现，尤其在 failure cases 上有针对性改进。

4. **Encoder 是瓶颈也是机会**
   - 当前 encoder 预测精度有限，导致 cluster assignment 错误较多。
   - 但理论表明：只要预测 embedding 距离正确 cluster 中心小于边界距离，仍可做出正确选择（见 Proposition 3）。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **Encoder 预测不准** | 当前 MLP + BGE 的组合未能稳定恢复 affinity embedding，影响部署效果 |
| **Probe-to-Executable Gap** | 某些 probe（如 ReAct, Tool Use）无法在生成器中实现，导致 seed 初始化受限 |
| **Adaptive Search 不满足独立性假设** | 构造过程中 validation set 被反复使用，不满足统计独立性，难以严格保证泛化性 |
| **聚类数量固定** | k=6 是经验设定，缺乏自动确定 cluster 数的方法 |

---

### 🔮 未来工作方向

1. **更强的 Embedding Encoder**
   - 使用更强大的 backbone（如 fine-tuned LLM）直接预测 affinity
   - 引入 contrastive learning 或 task-aware pretraining

2. **动态聚类与增量学习**
   - 支持新增 query 自动归类并触发新专家构建
   - 实现 lifelong agentic workflow evolution

3. **统一 Probe 与 Executor 设计**
   - 构建一个既能运行 probe 又能生成 workflow 的通用 interpreter，消除 gap

4. **跨任务/跨模型迁移**
   - 探索 affinity 表示是否可在不同 LLM 或任务域间迁移

5. **端到端联合优化**
   - 联合训练 encoder、clustering、expert generation，形成闭环优化系统

---

## 总结

> **RA-MoWE 提出了一种“以行为为导向”的 query 组织方式，将 workflow 设计从“一次性搜索”转变为“可复用专家构建 + 快速检索”的新模式。它首次系统地利用 workflow-affinity embeddings 实现了 query clustering、expert generation 和 zero-shot routing 的统一框架，在性能与效率之间取得了显著突破。尽管 encoder 预测仍是当前瓶颈，但其设计理念为未来 agentic AI 系统提供了重要范式转变的方向。**

</details>

---

### 15. [Confidence Reasoning Graphs: Structured Confidence Estimation for LLM Agents](https://arxiv.org/abs/2610.07948)

**Authors**: Brendan King, Farima Fatahi Bayat, Jean-Flavien Bussotti, Pouya Pezeshkpour, Estevam Hruschka  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.07948v1  

#### Abstract
When using an LLM agent in a consequential domain, making an informed decision about whether to trust its output or intervene requires calibrated confidence in the agent's success. Confidence estimation for agents is difficult because evidence about success is distributed across heterogeneous, inter...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Confidence Reasoning Graphs: Structured Confidence Estimation for LLM Agents**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
在将 **LLM Agents** 部署到高风险、后果重大的领域（如软件工程、企业运维）时，用户需要判断是否信任其输出。然而，仅凭平均成功率无法判断单次执行的成功概率，因为 Agent 的决策是多步、异构的（包含 LLM 推理、工具调用、环境交互），一个错误可能影响整个任务链。

现有置信度估计方法面临三大现实约束：
- **Black-box**：前沿 LLM 不暴露内部状态（如 token 概率、推理文本）；
- **Single-trajectory**：Agent 执行成本高，无法进行多次采样；
- **Training-free**：训练数据昂贵且易过时，无法依赖标注轨迹进行监督学习。

因此，如何在不依赖模型内部信号、无需多次运行、无训练数据的前提下，对单条 Agent 轨迹进行**校准良好的（well-calibrated）置信度估计**，是一个尚未解决的关键挑战。

---

### **提出了什么新方法或新思路**
本文提出 **Confidence Reasoning Graphs (CRGs)**，一种用于 LLM Agent 的结构化置信度估计框架。

#### **核心思想**
将“Agent 是否成功完成任务”这一宏观主张，通过 LLM 递归分解为多个可验证的子主张（sub-claims），每个子主张都与轨迹中的具体证据（evidence）挂钩，最终通过聚合叶节点的置信度得到根节点的整体置信度。

#### **四个步骤**
1. **Claim Decomposition**  
   从根主张（如“Agent 成功修复了 bug”）出发，使用 **Decomposition**（拆解为必要且充分的子条件）和 **Particularization**（结合任务上下文具体化）两种策略，构建一棵主张树。

2. **Evidence Gathering**  
   将轨迹中的关键事件作为证据节点，连接到相关叶节点，并标注关系：
   - `Supports`：支持该主张
   - `Undermines`：削弱该主张
   - `Unverified`：未验证（Agent 未确认某操作是否成功）

3. **Leaf-Claim Confidence Estimation**  
   使用另一个 LLM 对每个叶节点主张，在其关联证据和精简后的轨迹上下文中，进行**口头置信度评分**（verbalized confidence），得到 $ c_G \in [0,1] $。

4. **Aggregate Confidence to Root**  
   假设各子主张在给定轨迹下**条件独立**，则根节点置信度为所有叶节点置信度的乘积：
   $$
   F_o(T) = \prod_{G \in L} c_G
   $$

---

### **相比现有方法的优势**
| 维度 | CRG | 其他方法 |
|------|-----|----------|
| **无需特权访问** | ✅ 仅需轨迹文本 | ❌ 白盒方法需 token 概率 |
| **单次运行即可** | ✅ 仅需一条轨迹 | ❌ 采样法需多次 rollout |
| **无需训练数据** | ✅ 完全无监督 | ❌ Surrogate 模型需训练 |
| **结构可解释** | ✅ 可追溯低置信路径 | ❌ 黑箱标量输出 |
| **校准性更强** | ✅ 更低 ECE 和 Brier | ❌ 口头置信常过自信 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
在三个具有挑战性的 Agent 基准上评估：
| 数据集 | 任务类型 | 问题数 | 特点 |
|-------|--------|--------|------|
| **SWE-Bench Verified** | 软件工程 | 306 | GitHub issue 修复，真实代码库 |
| **EnterpriseOps-Gym** | 企业运维 | 325 | 多工具流程（日历、邮件、HR 等） |
| **SkillsBench** | 技能组合任务 | 81 | 长周期、多技能协作 |

使用 **OpenHands** 框架生成轨迹，搭配三种主流 LLM：
- GPT-5.5
- Gemini-3.5 Flash
- MiniMax-M3

---

### **实验设置和评估指标**

#### **评估指标**
| 指标 | 含义 | 期望方向 |
|------|------|---------|
| **Adaptive ECE** | 校准误差，衡量预测置信度与实际准确率的一致性 | ↓ 越小越好 |
| **Brier Score** | 概率预测误差的平方 | ↓ 越小越好 |
| **AUROC** | 区分成功与失败轨迹的能力 | ↑ 越大越好 |
| **Behavioral Alignment Score (BAS)** | 在不同风险偏好下的决策效用，惩罚高置信失败 | ↑ 越大越好 |

> **特别说明**：BAS 对高置信失败极度敏感（$\log(1-c)$ 发散），更能反映实际部署风险。

---

### **基线方法对比**
| 类型 | 方法 | 描述 |
|------|------|------|
| **Black-box (口头)** | Basic Verbalizer | 直接询问 LLM：“你有多大把握 Agent 成功？” |
|  | Reason-as-Graph | 提示 LLM 按照 CRG 结构思考，但仍输出单一置信度 |
|  | Verbal Consistency | 采样 10 次“是否成功”的判断，统计“True”比例 |
| **White-box (代理模型)** | Surrogate LNSP | 使用代理 LLM 计算最终动作的长度归一化序列概率 |

所有方法均使用相同 LLM（Qwen-3.8 27B 或 GPT-5.6 Sol）作为估计器。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（Qwen-3.8 27B，平均值）**

| 方法 | SWE-Bench ECE | EnterpriseOps ECE | SkillsBench ECE | 平均 BAS |
|------|----------------|--------------------|------------------|-----------|
| **CRG (ours)** | **0.09** | **0.13** | **0.11** | **0.14** |
| Basic Verbalizer | 0.18 | 0.49 | 0.41 | 0.27 |
| Reason-as-Graph | 0.18 | 0.46 | 0.38 | 0.27 |
| Verbal Consistency | 0.24 | 0.45 | 0.45 | -13.87 |
| Surrogate LNSP | 0.10 | 0.22 | 0.16 | 0.09 |

> ✅ CRG 在所有基准上实现了**最低的 ECE 和 Brier Score**，以及**最高的 BAS**。

---

### **与基线方法的对比结果**
- **优于口头置信方法**：  
  CRG 显著缓解了口头置信的**过度自信**问题（如 Verbal Consistency 的 BAS 为负）。
- **优于代理模型**：  
  Surrogate LNSP 虽然 ECE 较低，但 **AUROC 接近随机水平**（SWE-Bench 上仅 0.47），表明其无法有效区分成败，只是“平均押注”成功率。
- **唯一兼顾校准与判别能力**：  
  CRG 是唯一同时实现强校准（低 ECE）、良好判别（高 AUROC）和正向决策效用（高 BAS）的方法。

---

### **消融实验结果**
#### **Table 4: 组件消融（Development Set）**
| 方法 | ECE | Brier | AUROC | BAS |
|------|-----|-------|--------|------|
| Reason-as-Graph（无显式图） | 0.38 | 0.37 | 0.73 | -0.23 |
| Reason-with-Graph（有图但直接评分根节点） | 0.35 | 0.36 | 0.61 | -0.16 |
| **CRG（完整方法）** | **0.10** | **0.23** | **0.68** | **0.17** |

> 🔍 **关键发现**：性能提升主要来自**叶节点置信度估计 + 自底向上聚合**，而非仅仅是“结构化思考”。图结构本身帮助不大，关键是**局部评估 + 乘积聚合**。

#### **其他消融**
- **聚合规则比较**：乘积聚合（product）在训练免费方法中表现最佳，优于算术/几何平均、最大值、Fréchet bounds。
- **最大深度 k=5** 已足够，更深不会带来收益。
- **推理努力（reasoning effort）** 对结果影响不大，说明性能提升非来自计算开销。

---

## 4. **关键结论和发现**

### **主要发现**
1. **结构化分解显著提升置信度校准性**  
   将全局判断分解为证据支撑的子主张，能有效避免 LLM 的过度自信。

2. **校准性 ≠ 判别力**  
   Surrogate LNSP 表面校准良好（低 ECE），实则判别力接近随机，说明**仅看 ECE 会误导**。BAS 等决策导向指标更可靠。

3. **叶级评估 + 乘积聚合是关键**  
   单纯让 LLM “像 CRG 一样思考” 效果有限；必须显式构造图并逐叶评分，再通过乘积聚合，才能获得显著提升。

4. **低成本高效益**  
   CRG 单次估计成本仅 **$0.04–$0.07**，远低于生成轨迹的成本（$0.15–$1.06），且位于 **cost-calibration Pareto 前沿**。

5. **可审计性强**  
   当 CRG 输出低置信时，用户可追溯至具体哪个子主张和证据导致，支持人工干预（如案例研究中修正被遗漏的证据）。

---

### **方法的局限性**
- **依赖 LLM 的分解质量**：若 LLM 未能正确分解或遗漏关键子条件，会影响结果。
- **条件独立假设不完美**：现实中子主张可能存在依赖，乘积聚合可能低估联合概率。
- **构造图存在噪声**：实证分析（A.11）显示，约 20–30% 的分解不满足逻辑完备性。
- **仍为启发式方法**：虽理论上有依据，但非严格概率推断。

---

### **未来工作方向**
- **改进聚合机制**：探索更复杂的依赖感知聚合方式（如贝叶斯网络、注意力机制）。
- **自动化图优化**：引入反馈循环，自动修正不完整的图结构。
- **扩展到多模态 Agent**：处理图像、语音等非文本轨迹。
- **动态调整分解粒度**：根据任务复杂度自适应控制图深度。
- **集成外部验证器**：将形式化验证结果作为证据注入图中。

---

> 📌 **一句话总结**：  
> **CRG 通过将“是否成功”的整体判断转化为“由证据支撑的子主张图”，实现了无需训练、单次运行、黑盒可用的高质量置信度估计，在校准性、决策效用和可解释性上全面超越现有方法。**

</details>

---

### 16. [zkLLMPoT: Efficient Zero Knowledge Proof of Training for Large Language Models](https://arxiv.org/abs/2610.08258)

**Authors**: Junkai Liang, Zhanpeng Guo, Pengfei Wu, Qingni Shen, Jiaheng Zhang, Zhonghai Wu, Haiyang Xue, Shengfang Zhai  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.08258v1  

#### Abstract
Auditing the claimed outcomes of large language model (LLM) training is challenging when model weights and training data are private, while cryptographically proving the full training process is prohibitively expensive at Transformer scale. We present zkLLMPoT, a zero-knowledge framework that certif...

---

### 17. [MoF: Preference-Aware Mixture Modeling for Black-Box LLM Personalization](https://arxiv.org/abs/2610.08330)

**Authors**: Hun Park  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.08330v1  

#### Abstract
Proprietary Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet aligning their outputs with diverse user preferences remains challenging. Existing personalization approaches for black-box LLMs often rely on user-specific scoring heads, causing the...

---

### 18. [Parallel Predictive World Models for Accurate and Efficient Long-Horizon Planning](https://arxiv.org/abs/2610.08627)

**Authors**: Wanjin Feng, Baobin Zhang, Ao Yu, Shibo Feng, Xi Wang, Xingyu Gao  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.08627v1  

#### Abstract
Long-horizon world-model planning typically relies on autoregressive rollouts, where predicted states are repeatedly fed back into the model. This preserves temporal structure but creates a horizon-length sequential path and exposes later predictions to recursive decoded-state feedback. We introduce...

---

### 19. [Optimization Encoders: Rethinking Second-Order Meta-Learning for Neural Fields](https://arxiv.org/abs/2610.08075)

**Authors**: Rudolf L. M. van Herten, Soufiane Ben Haddou, Rachit Saluja, Johannes C. Paetzold  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.08075v1  

#### Abstract
Conditional neural fields represent signals continuously, but their effectiveness depends on how the conditional latent representations are inferred from observed data. In meta-learning, this encoding occurs through gradient updates induced by the decoder, tying representation learning directly to d...

---

### 20. [Evolutionary One-Step Generators: Fast and Diverse Sampling for Discrete Design](https://arxiv.org/abs/2610.08367)

**Authors**: Marcus Vukojevic, Erik Nielsen, Veronica Lachi, Andrea Passerini, Giovanni Iacca  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.08367v1  

#### Abstract
Several discrete design tasks, such as molecular discovery, require diverse collections of useful candidates at low computational cost. High validity alone does not guarantee a useful candidate library: repeatedly generating the same valid structures leaves few distinct alternatives. Training for bo...

---

### 21. [PHBA: Prefix-State Hybrid Block Attention](https://arxiv.org/abs/2610.08527)

**Authors**: Ruijie Li, Jiaxi Hu, Shiyu Wang, Yuxuan Liang  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.08527v1  

#### Abstract
Hybrid architectures combining linear sequence models with softmax attention provide an effective balance between efficient long-context modeling and precise token retrieval. Existing designs such as Native Hybrid Attention (NHA) combine compressed long-term states with sliding-window attention, but...

---

### 22. [Offline AI Modules: Voice-First Offline Architecture, Hardware Reference Stack, Quantization and Benchmarking](https://arxiv.org/abs/2610.07026)

**Authors**: Sunday Afariogun, Odunolaoluwa Jenrola, Zeinab Nezami  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.07026v1  

#### Abstract
The Offline AI Modules workstream enables practical, low-power, and community-accessible deployment of voice-first AI systems that operate fully offline. Designed for African language communities where speech is the dominant mode of interaction and internet connectivity is unreliable or absent, the ...

---

### 23. [Navigating Route Latent Space for Synthesizable Molecular Design](https://arxiv.org/abs/2610.07560)

**Authors**: Tao Li, Tuan Vinh, Monika Raj, Yuan Fang, Zhichun Guo, Carl Yang  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.07560v1  

#### Abstract
Goal-directed molecular design has advanced rapidly, yet a substantial proportion of designed molecules remain difficult to synthesize in practice, limiting their real-world utility. Prior synthesizability-aware methods either project generated molecules back to synthesizable analogs that deviate fr...

---

### 24. [Agentic Semantic Sensing for Resource-Adaptive AI-RAN](https://arxiv.org/abs/2610.07829)

**Authors**: Zhongqin Wang, Xiaoqi Zhang, Nan Yang, Kai Wu, J. Andrew Zhang, Y. Jay Guo  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.07829v1  

#### Abstract
Semantic sensing (SemS) acquires task-relevant information rather than reconstructing complete physical information. Existing SemS formulations typically operate open loop: sensing configurations and observation schedules are fixed before inference and cannot respond to evolving task-level evidence....

---

### 25. [SquidAgent: Parallelize Wisely, Coordinate Efficiently](https://arxiv.org/abs/2610.08647)

**Authors**: Yexiong Lin, Shanshan Ye, Yu Yao, Zhen Fang, Bo Han, Tongliang Liu  
**Category**: cs.AI  
**Published**: 2026-10-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.08647v1  

#### Abstract
LLM-based agents solve complex multi-step tasks, but sequential execution incurs substantial latency. In principle, parallelizing work across multiple agents should yield near-linear speedups. Yet existing parallel multi-agent systems often run slower than a single-agent baseline. We attribute this ...

---

### 26. [WavePrune: One period is often enough for RoPE](https://arxiv.org/abs/2610.06963)

**Authors**: Guancheng Du, Luotian Huang, Shaowen Wang, Si Li, Kaifeng Lyu  
**Category**: cs.CL  
**Published**: 2026-10-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.06963v1  

#### Abstract
Rotary Position Embedding (RoPE) encodes token positions by rotating each two-dimensional channel of the query and key vectors at a channel-specific frequency, making the attention logits invariant to a common shift of positions. However, this rotation is periodic, and it leads to position aliasing ...

---

### 27. [Nucleus Speculative Decoding: Plausibility-Aware Verification Beyond Exact Distribution](https://arxiv.org/abs/2610.07822)

**Authors**: Shuhao Li, Fanghua Ye, Wanyu Lin, Tianyu Yuan, Xiaoyu Shen  
**Category**: cs.CL  
**Published**: 2026-10-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.07822v1  

#### Abstract
Speculative decoding accelerates autoregressive generation by using a lightweight draft model to propose multiple tokens that are verified by a target model in parallel. However, the standard acceptance rule focuses on exact distribution correction and rejects tokens that remain highly plausible und...

---

### 28. [Few-Shot Bioactivity Prediction with Meta-Learning under Assay Heterogeneity](https://arxiv.org/abs/2610.07079)

**Authors**: Michal Kmicikiewicz, Tommy Rochussen, Vincent Fortuin, Ewa Szczurek  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.07079v1  

#### Abstract
Accurate bioactivity prediction is a central challenge in early-stage drug discovery, as individual assays often contain too few measurements to train reliable models independently. Meta-learning offers a principled approach to this few-shot setting, but assay heterogeneity may limit its effectivene...

---

### 29. [Decoupling What from Where: How Should a Small GUI Grounding Model Receive the Action Type?](https://arxiv.org/abs/2610.07444)

**Authors**: Aadi Chauhan, Arthur Ilyasov  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.07444v1  

#### Abstract
A GUI agent decides which action to take and where to take it; we ask how a small grounding model should receive the action type. Fine-tuning Qwen2-VL-2B with LoRA on Android in the Wild, we compare a flat baseline with five ways of supplying the type under matched data, compute, and decoding: an au...

---

### 30. [Adaptive Mean Estimation by In-Context Learning: A Gradient-Flow Analysis](https://arxiv.org/abs/2610.07804)

**Authors**: Martin Eppert, Krishna Balasubramanian, Subhro Ghosh, Jason Klusowski, Yan Shuo Tan  
**Category**: cs.LG  
**Published**: 2026-10-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.07804v1  

#### Abstract
Prior Fitted Networks (PFNs) such as TabPFN now rival established statistical procedures across prediction and estimation tasks. A natural explanation is that PFNs have the property of statistical adaptivity, that is, they perform nearly as well as a method tailored to the true data-generating model...

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

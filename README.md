# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-22 08:44:03 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DynaFlow: Transparent and Flexible Intra-Device Parallelism via Programmable Operator Scheduling](https://arxiv.org/abs/2605.21603)

**Authors**: Yi Pan, Yile Gu, Jinbin Luo, Yibo Wu, Ziren Wang, Hongtao Zhang, Ziyi Xu, Shengkai Lin, Baris Kasikci, Stephanie Wang  
**Category**: cs.DC  
**Published**: 2026-05-22  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.21603v1  

#### Abstract
Intra-device parallelism addresses resource under-utilization in ML inference and training by overlapping the execution of operators with different resource usage. However, its wide adoption is hindered by a fundamental conflict with the static, sequential programming model of existing frameworks. I...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DynaFlow: Transparent and Flexible Intra-Device Parallelism via Programmable Operator Scheduling**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代机器学习模型（如 LLM 和扩散模型）的算子具有高度异构的资源需求（compute-bound、memory-bound、network-bound），在单设备内顺序执行会导致大量资源闲置，降低推理和训练吞吐量。尽管已有多种 **intra-device parallelism** 策略（如 overlapping、fusion、splitting）可提升利用率，但其集成面临两大挑战：
- **工程成本高**：现有框架（如 vLLM、SGLang）采用静态、顺序编程模型，难以支持非顺序执行，需对模型代码进行侵入式修改。
- **策略不通用**：最优策略高度依赖运行时上下文（workload、模型架构、硬件），开发者需为不同场景维护多个专用实现。

### **提出了什么新方法或新思路**
提出 **DynaFlow**，一个通过解耦逻辑模型定义与物理执行调度来实现透明且灵活的 intra-device 并行性的框架。其核心思想是引入一个**可编程的执行底座（programmable execution substrate）**，位于模型逻辑与实际执行之间，允许用户动态定义非顺序执行计划，而无需修改模型本身。

### **相比现有方法的优势**
- **透明性（Transparency）**：无需修改模型核心逻辑，仅通过少量注解即可集成。
- **灵活性（Flexibility）**：统一接口支持多种并行策略（overlapping、fusion、splitting）及其组合，并可在运行时动态调整。
- **高效性（Efficiency）**：后端异步管理复杂的控制流与数据流，采用零拷贝内存管理，并兼容 CUDA Graphs、TorchInductor 等底层优化。
- **通用性**：作为 `torch.compile` 后端，可无缝集成到任意 PyTorch 生态系统中。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **真实数据集**：
  - **ShareGPT**：用于模拟真实对话场景。
  - **LMSYS-Chat-1M**：大规模人类偏好对话数据。
  - **Splitwise**：轻量级工作负载，用于测试低负载下的性能。
- **合成数据集**：固定输入/输出长度（如 512-in-128-out），用于可控性能分析。

### **实验设置和评估指标**
- **硬件平台**：
  - DGX B200：8 × NVIDIA B200 GPU（NVLink）
  - H100 系统：4 × NVIDIA H100 GPU（NVLink）
  - 部分实验使用 PCIe 模拟低带宽多节点环境。
- **模型**：
  - Llama-3-8B/70B、Qwen-2.5-72B、DeepSeek-V2-Lite（MoE）、Mixtral、Wan-14B（视频生成）等。
- **评估指标**：
  - **End-to-end throughput**（tokens/s 或 seq/s）
  - CPU 执行时间（衡量调度开销）
  - 内存占用、初始化时间
  - 与基线的加速比（speedup）

### **基线方法对比**
- **原始系统**：未修改的 vLLM、SGLang、Megatron-LM、HuggingFace Transformers 等。
- **专用实现**：
  - vLLM 内置的 **Dual-Batch Overlap (DBO)** 实现
  - 原始 **TokenWeave** 框架
- **朴素实现（Naive）**：
  - 固定阈值的 batch splitting（如在 vLLM/SGLang 上的 PR 实现）
  - 无动态判断的简单重叠

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **最高加速比**：在多个系统中实现 **最高达 1.29x 的端到端吞吐提升**。
- **与专用实现对比**：DynaFlow 实现的策略性能**匹配甚至优于**原生手写实现（最高 **1.1x 更优**）。
- **CPU 开销控制**：
  - 动态模式下 CPU 时间为 10.8ms（vLLM 原生为 4.4ms）
  - 支持**顺序回退模式**（sequential fallback），CPU 时间降至 4.7ms，几乎无额外开销。
- **初始化开销**：
  - 静态分析耗时 0.2s
  - CUDA Graph 捕获耗时 4.3s（vLLM 为 2.4s）
  - 显存占用 1.80 GiB（vLLM 为 0.98 GiB）
  - 相对于总初始化时间（24s）和现代 GPU 显存容量，可接受。

### **与基线方法的对比结果**
| **策略** | **系统** | **DynaFlow 加速比** | **vs. 专用实现** |
|---------|--------|------------------|----------------|
| **NanoFlow** | vLLM | 最高 **1.29x** | 超过朴素实现（0.56x~0.35x） |
| **DBO** | vLLM (DeepSeek-V2-Lite) | 最高 **1.14x** | 匹配原生 DBO，某些负载快 **1.1x** |
| **TokenWeave** | vLLM/HF | 最高 **1.21x/1.22x** | 性能相当，但支持动态调参（+12%） |
| **Comet** | Megatron-LM | 最高 **1.25x~1.27x** | 完全匹配原生 Comet 实现 |
| **通信重叠** | 多系统 | 最高 **1.15x** | 在通信受限场景有效 |

> 注：在轻量负载（如 ShareGPT）上，因 batch splitting 条件不满足，加速效果有限；但在计算密集型负载（如 Splitwise）上增益显著。

### **消融实验结果**
- **禁用 CUDA Graph**：吞吐下降至 **0.96x**（相对 DynaFlow 全开），表明其对缓解 CPU 开销至关重要。
- **禁用零拷贝内存管理**：吞吐降至 **1.10x**，验证了其减少内存拷贝开销的有效性。
- **使用静态分裂策略**：吞吐降至 **1.00x**，说明**动态调度决策**对性能至关重要。

---

## **4. 关键结论和发现**

### **主要发现**
- **解耦调度与模型是可行且高效的**：DynaFlow 成功将 intra-device parallelism 的调度逻辑从模型实现中剥离，实现了**透明、灵活、高性能**的集成。
- **统一接口可表达多样策略**：通过 `SplitModule`、`SplitFunc`、`mark` 和 `execute` 等 API，可在 **平均仅 11 + 31 行代码**内实现复杂策略（如 DBO、TokenWeave）。
- **动态性是关键优势**：相比静态策略（如 vLLM 原生 DBO），DynaFlow 可根据 batch size、seq len 等上下文动态启用/关闭优化，避免在小批量时性能劣化。
- **兼容性设计成功**：通过在 subgraph 级别应用 TorchInductor 和 per-microbatch CUDA Graphs，既保留了底层优化，又支持了动态调度。

### **方法的局限性**
- **初始化开销较高**：由于需为不同 micro-batch 捕获多个 CUDA Graph，初始化时间和显存占用高于传统静态系统。
- **依赖 TorchDynamo 支持**：对无法被 TorchDynamo 正确捕获的自定义算子（如 Megatron-LM 中的 fused backward op），可能无法有效重叠。
- **当前主要面向 intra-device**：未直接解决跨节点（inter-node）并行的调度问题。

### **未来工作方向**
- **降低初始化开销**：探索共享 subgraph 编译、图缓存（如 Medusa）等技术。
- **扩展至 inter-device 场景**：将可编程调度思想推广到分布式训练中的 pipeline parallelism 或 data parallelism。
- **自动策略搜索**：结合 profiling 数据，在线学习最优调度策略（如何时 split、fuse、overlap）。
- **支持更多硬件后端**：适配 AMD ROCm、Google TPU 等非 CUDA 平台。

---

> **开源地址**：[https://github.com/uw-syfi/DynaFlow](https://github.com/uw-syfi/DynaFlow)

</details>

---

### 2. [PALS: Power-Aware LLM Serving for Mixture-of-Experts Models](https://arxiv.org/abs/2605.21427)

**Authors**: Can Hankendi, Rana Shahout, Minlan Yu, Ayse K. Coskun  
**Category**: cs.AI  
**Published**: 2026-05-22  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.21427v1  

#### Abstract
Large language model (LLM) inference has become a dominant workload in modern data centers, driving significant GPU utilization and energy consumption. While prior systems optimize throughput and latency by batching, scheduling, and parallelism, they largely treat GPU power as a static constraint ra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PALS: Power-Aware LLM Serving for Mixture-of-Experts Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代数据中心中，**Large Language Model (LLM)** 推理已成为主导性工作负载，带来巨大的 GPU 利用率和能源消耗。尽管已有系统通过 **batching、scheduling 和 parallelism** 优化吞吐量和延迟，但它们通常将 GPU 功耗视为静态约束，而非可调控资源。

这导致两个核心问题：
- **过供电**：为保证性能而预留过多功率，造成能源浪费；
- **欠供电**：在功耗受限时无法动态调整，引发 QoS（服务质量）违规。

尤其对于 **Mixture-of-Experts (MoE)** 模型，其动态路由机制引入显著的通信开销，在高并行度下易成为通信瓶颈，使得传统独立调优策略失效。

---

### 提出了什么新方法或新思路
本文提出 **PALS (Power-Aware LLM Serving)** ——一种**功耗感知的运行时系统**，首次将 **GPU power cap** 视为与软件参数（如 batch size）同等重要的“一级控制原语”（first-class control knob），进行联合优化。

#### 核心设计思想：
- **跨层协同控制**：联合调节硬件层（power cap）与软件层（batch size, TP/EP 并行度）参数。
- **闭环反馈控制器**：结合轻量级离线建模 + 在线反馈校正，实现对动态负载和外部信号（如电网需求响应）的快速适应。
- **即插即用集成**：无需修改模型架构或推理 API，直接嵌入现有框架（如 vLLM）。

---

### 相比现有方法的优势
| 维度 | 传统方法 | PALS |
|------|--------|-------|
| 控制粒度 | 单独优化 batching 或 DVFS | 联合优化 power cap + batch + parallelism |
| 功耗角色 | 外部硬约束 | 可控调度维度 |
| 响应能力 | 静态配置或粗粒度调整 | 500ms 级闭环反馈，支持动态预算跟踪 |
| 效率边界 | 局限于 SW-only 或 HW-only 前沿 | 扩展 Pareto 前沿，逼近 Oracle 性能 |

> ✅ **关键突破**：证明了 **HW+SW 联合控制** 能打开新的高效操作区间，尤其是在 MoE 这类通信敏感模型上效果显著。

---

## 2. 核心实验方法和设置

### 使用的模型（非数据集）
PALS 主要评估的是 LLM 推理服务性能，因此使用多个代表性 **LLM 模型** 构成 workload，涵盖 dense 与 MoE 架构：

| 模型类型 | 模型列表 |
|---------|----------|
| Dense Models | GPT-2, Llama-2-7B, Mistral-7B |
| MoE Models | Mixtral-8x7B, Qwen1.5-MoE, OLMoE-1B-7B, DeepSeek-MoE, Phi-3.5-MoE |

这些模型具有不同的 **compute/communication ratio**，用于验证方法普适性。

---

### 实验设置
- **硬件平台**：多节点服务器，每节点配备 4× NVIDIA A100 GPU，通过 NVLink 互联。
- **软件框架**：基于 **vLLM** 实现，利用 PagedAttention 支持连续批处理（continuous batching）。
- **控制周期**：500ms 一次 telemetry 采集与配置更新。
- **输入流量**：采用 **Poisson 分布** 模拟突发请求流，序列长度多样化。

---

### 评估指标
| 指标 | 定义 | 目标 |
|------|------|------|
| **Tokens/J** | 每焦耳能量生成的 token 数 | 衡量能效，越高越好 |
| **Throughput (tokens/s)** | 系统整体推理吞吐量 | 满足 QoS 要求 |
| **QoS Violation Rate** | 未达目标吞吐的时间占比 | 越低越好 |
| **Power Tracking Error** | 实际功耗与目标预算偏差 | 跟踪精度高 |
| **Normalized Efficiency** | 相对于 baseline 的能效提升倍数 | 对比基准 |

---

### 基线方法对比
| 基线名称 | 控制方式 | 特点 |
|--------|----------|------|
| **Baseline** | 固定 400W power cap + 最大 batch size (64) | 吞吐优先，默认部署策略 |
| **Adaptive Batch** | 固定 power cap，动态调 batch size | 仅软件优化 |
| **Adaptive Cap** | 固定 batch size，动态调 power cap | 仅硬件优化 |
| **Oracle** | 离线索引所有配置中最优解 | 上界参考，不现实 |

> PALS 是唯一同时调节 **power cap** 和 **batch size** 的方法，并引入反馈机制应对不确定性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 场景 | 指标 | 结果 |
|------|------|------|
| 单节点效率测试 | 平均 **Tokens/J 提升** | **+26.3%** vs Baseline |
| 多节点功耗受限场景 | **QoS Violation Reduction** | **减少 4×–7×** |
| 动态需求响应（DR）场景 | 低功耗目标下的吞吐保持 | **提升最高达 22%** vs Static Batch |
| 接近最优程度 | 相比 Oracle 的效率达成率 | 达到 **95%** 的 oracle headroom |

---

### 与基线方法的对比结果
#### （1）单节点能效比较（图7）
- **PALS** 实现 **1.26× 归一化能效**，远超其他单一控制策略：
  - Adaptive Batch: ~1.14×
  - Adaptive Cap: ~1.05×
- 表明：**batching 是主导因素，但 power cap 协同仍提供额外增益**

#### （2）多节点 QoS 表现（图9）
在总集群功耗限制为 4800W 的三节点部署中：
- **Baseline** 的 QoS 违规率达 **18.7%–35.2%**
- **PALS** 将违规率降至 **3.2%–5.3%**
- 同时提升整体能效 **12.1%**

> 💡 说明：仅靠 batch 或 cap 单独调节无法有效维持 QoS；必须联合控制才能兼顾性能与节能。

#### （3）Grid-Interactive Demand Response（图10）
模拟电网需求响应事件（动态调整功耗目标）：
- PALS 能精准跟踪目标功耗曲线；
- 在低功耗阶段，通过减小 batch size 避免资源浪费，**维持更高 throughput（+22%）**

---

### 消融实验结果
#### （1）HW+SW 联合控制 vs 单一控制（图3）
构建三种 Pareto frontier：
- **SW-only**（变 batch，固定 cap）：偏向高吞吐
- **HW-only**（变 cap，固定 batch）：偏向高能效
- **HW+SW**（联合调节）：完全包络前两者，形成更优前沿

> ➕ 联合控制扩展了可达的操作空间，特别是在中间区域填补空白。

#### （2）Tensor Parallelism (TP) 的影响（图8）
- 固定 TP 部署下，PALS 已接近最优；
- 但在低 power cap 区间，最佳 TP 会变化（通信敏感），存在少量 headroom；
- 说明：**runtime-feasible knobs（power + batch）已捕获大部分收益**，TP 更适合部署期决策。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **Power Cap 存在边际效益递减现象**  
   - 超过模型依赖的阈值（约 150–200W）后，继续增加 power cap 反而导致 **tokens/J 下降**；
   - 尤其在 **communication-bound MoE 模型** 中，额外功率加速的是 NCCL all-to-all 等通信开销，而非有效计算。

2. 📊 **Batch Size 是能效最大驱动因素**  
   - 批大小从 1 增至 64，能效提升 **1.7×–2.1×**；
   - MoE 模型因路由开销更大，在大 batch 下仍有明显增益。

3. ⚖️ **Compute/Communication Ratio 决定最优并行策略**  
   - Compute-bound 模型（如 Mixtral）受益于更高 power cap；
   - Communication-bound 模型（如 Qwen-MoE）应在较低 power 下运行；
   - 最佳 TP 设置也随 power 变化，体现跨层耦合。

4. 🚀 **联合控制显著扩展 Pareto 前沿**  
   - HW+SW 联合优化相比单独控制，峰值能效提升 **1.13×–1.18×**；
   - 实现“既高效又稳定”的服务模式。

---

### 方法的局限性
1. ❗ **预测模型依赖离线 profiling**
   - 当在线 workload 明显偏离训练分布（如极长 prompt、异常输出长度）时，预测误差可能增大；
   - 当前依赖 PID 反馈补偿，尚不能主动预测突变。

2. 🧱 **仅支持 runtime-feasible knobs**
   - 不动态调整 TP/EP 等需重载模型的参数；
   - 在极端通信受限场景下，仍有少量优化空间未被捕捉。

3. 🖥️ **当前聚焦 node-level 控制**
   - 不替代 cluster-level 调度器（如 DynamoLLM），而是与其协作；
   - 需依赖上层提供 per-node power budget 和 throughput target。

---

### 未来工作方向
1. ✅ 引入 **轻量级在线模型更新机制** 或 **不确定性感知预测**，增强对未知 workload 的鲁棒性；
2. 🔗 探索 **node-cluster 协同控制架构**，实现跨时间尺度的统一调度；
3. 🌐 构建 **grid-interactive AI serving 生态**，使 LLM 成为可调节负载，参与碳感知计算与电力市场响应；
4. 🔄 将 PALS 思路推广至 **training 场景** 或其他稀疏模型架构（如 retrieval-augmented models）。

---

## 总结
PALS 是首个将 **GPU power cap** 作为一级调度维度引入 LLM serving runtime 的系统，通过 **HW+SW 联合控制 + 闭环反馈**，实现了：
- **高达 26.3% 的能效提升**
- **4×–7× 的 QoS 违规减少**
- **对动态功耗预算的实时跟踪能力**

它不仅提升了能效，更为构建 **energy-proportional、grid-interactive 的可持续 AI 基础设施** 提供了实用路径。

</details>

---

### 3. [LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation](https://arxiv.org/abs/2605.22054)

**Authors**: Zhuo Chen (equal contribution), Xinzhe Yuan (equal contribution), Jianshu Zhang (Shanghai Artificial Intelligence Laboratory, Shanghai, China, School of Computer Science, Shanghai Jiao Tong University, Shanghai, China), Jinzong Dong (Shanghai Artificial Intelligence Laboratory, Shanghai, China, School of Automation, Central South University, Changsha, China), Ruichen Zhou (College of New Energy and Materials, China University of Petroleum, Beijing, China), Yingchun Niu (College of New Energy and Materials, China University of Petroleum, Beijing, China), Tianhang Zhou (College of Carbon Neutrality Future Technology, China University of Petroleum, Beijing, China), Yu Yang Fredrik Liu (DeepVerse PTE. LTD., Singapore), Yuqiang Li (Shanghai Artificial Intelligence Laboratory, Shanghai, China), Nanyang Ye (Shanghai Artificial Intelligence Laboratory, Shanghai, China, School of Computer Science, Shanghai Jiao Tong University, Shanghai, China), Qinying Gu (Shanghai Artificial Intelligence Laboratory, Shanghai, China)  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.22054v1  

#### Abstract
The high cost and data scarcity in scientific exploration have motivated the use of large language models (LLMs) as knowledge-driven components in Bayesian optimization (BO). However, existing approaches typically embed LLMs directly into the sampling or surrogate modeling pipeline, without fully le...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在科学探索任务中（如药物设计、催化剂优化等），**每次真实实验（real-fidelity evaluation）成本高昂、耗时长且数据稀缺**，导致传统 **Bayesian Optimization (BO)** 面临两大挑战：
- **冷启动问题（cold-start problem）**：初期缺乏有效先验，难以高效探索。
- **高维搜索空间效率低**：在有限预算下难以全面探索。

尽管已有研究尝试引入 **Large Language Models (LLMs)** 作为知识源来辅助 BO，但这些方法通常仅将 LLM 用于初始化或局部建议，**未能充分利用 LLM 极低成本、广域推理的能力进行系统性探索**。

---

### 🚀 提出的新方法：LABO
作者提出 **LLM-Accelerated Bayesian Optimization (LABO)**，一种将 LLM 预测与真实实验深度融合的双保真度（dual-fidelity）BO 框架。

#### 核心思想：
- 将 **LLM 视为一个低成本、知识驱动的“低保真度”预测器（LLM-fidelity predictor）**。
- 将 **真实实验视为高成本、高保真度的观测（real-fidelity observation）**。
- 利用 **Kennedy-O’Hagan (KOH) 联合高斯过程（joint GP）模型** 统一建模两者关系：
  $$
  f_R(x) = \rho f_L(x) + \delta(x)
  $$
  其中：
  - $ f_L(x) $：LLM 预测
  - $ \rho $：尺度校准因子
  - $ \delta(x) $：残差项，捕捉 LLM 无法解释的偏差

#### 关键创新机制：**Gating Criterion（门控准则）**
- 定义 **discrepancy dominance ratio**：
  $$
  \rho_\Delta(x) = \frac{\text{Var}[\delta(x)]}{\text{Var}[f_R(x)]}
  $$
- 若 $ \rho_\Delta(x) \leq \tau $，说明 LLM 预测可靠 → **仅使用 LLM 预测更新模型**
- 否则触发真实实验 → **保留昂贵资源给不确定性高的区域**

> 💡 **本质是：用 LLM 进行“广域廉价探索”，只在 LLM 不确定的地方才动用真实实验**

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | LABO 的改进 |
|------|--------|-------------|
| **Vanilla BO** | 无外部知识，冷启动慢 | 引入 LLM 先验加速探索 |
| **LLAMBO / BOPRO / CAKE** | LLM 仅用于初始化或提示生成 | LLM 参与全过程建模，持续提供低成本信号 |
| **ChemBOMAS / ToSFiT** | LLM 仅作伪实验注入初始点 | 动态判断是否需要真实实验，实现自适应资源分配 |

> ✅ LABO 是首个将 LLM 作为**持续、可量化置信度的低保真信号源**，并结合**理论驱动的门控机制**进行选择性实验的方法。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集（共6个科学优化任务）
| 任务 | 维度 | 目标 | 类型 |
|------|------|------|------|
| **COF**(14D) | 14D | 最大化 Xe/Kr 吸附选择性 | 化学材料 |
| **Fullerene**(3D) | 3D | 最大化 C60 加合物产率 | 化学反应 |
| **PCE10**(4D) | 4D | 最小化光降解（越小越好） | 光伏材料 |
| **P3HT**(5D) | 5D | 最大化电导率 | 复合材料 |
| **Flow Battery**(3D) | 3D | 最大化综合性能指标 | 能源系统 |
| **Sandwich**(20D) | 20D | 最大化饮食健康评分 | 营养配方 |

此外还测试了：
- **AutoML**：SVM (2D), MLP (5D) 超参优化
- **High-Dim**：超导体临界温度优化 (86D)

---

### ⚙️ 实验设置
- **每轮迭代选择 2 个候选点**
- **真实实验预算上限：30 轮（即最多 60 次 real-fidelity 查询）**
- **Warm-start 阶段**：使用 LLM 推荐初始点 + 对 50 个 LHS 样本进行 LLM 预测
- **Gating threshold $\tau = 0.75$**（固定，未调优）
- **Surrogate Model**：RBF Kernel + GP-UCB Acquisition
- **LLM Backbone**：Intern-S1 (241B)，具备强科学知识背景

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **最终最优目标值（Final Objective Value）** | 衡量优化效果上限 |
| **达到 90% 最优值所需迭代次数** | 衡量收敛速度 |
| **LLM vs Real 查询比例（L/R Ratio）** | 衡量样本效率与成本控制能力 |
| **累计后悔（Cumulative Regret）** | 理论分析中的核心指标 |

---

### 🆚 基线方法对比
| 方法 | 简介 |
|------|------|
| **Vanilla BO** | 标准 BO，无任何外部知识 |
| **LLAMBO** | LLM 提供初始化 + 优化过程中提供建议 |
| **BOPRO** | 在 LLM 编码空间中进行 BO |
| **CAKE** | LLM 动态生成和调整 GP kernel |
| **ToSFiT**（额外对比） | 将 Thompson Sampling 转化为 LLM 微调过程 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Figure 2 和 Tables）

| 任务 | 方法 | 最终目标值（↑） | 达到 90% 最优所需迭代数（↓） |
|------|------|------------------|-------------------------------|
| **COF (14D)** | LABO | **11.23** | **14.2** |
| | Vanilla BO | 10.5 | ~25 |
| | LLAMBO | 10.8 | ~20 |
| | CAKE | 10.9 | ~22 |
| **Fullerene (3D)** | LABO | **0.9512** | **14.6** |
| | Others | ≤0.95 | ≥22 |
| **P3HT (5D)** | LABO | **1.5** | **12.6** |
| | Best Baseline | ~1.4 | ~20 |
| **Sandwich (20D)** | LABO | 明显领先 | 收敛最快 |
| **86D Superconductor** | LABO | **91.06°C** | **24.4** |
| | Vanilla BO | 57.96°C | 64.6 |

> ✅ LABO 在所有任务上均取得 **最佳最终性能** 和 **最快收敛速度**

---

### 🔬 消融实验结果（Ablation Studies）

#### （1）不同 Gating Threshold $\tau$ 的影响（Table 1）
| $\tau$ | COF 最终目标 | 迭代至 90% | L/R 比例 |
|--------|---------------|------------|----------|
| 0.60 | 10.78 | 24.6 | 1.52 |
| 0.75 | **11.23** | **14.2** | **2.68** |
| 0.85 | 11.17 | 12.6 | 5.26 |

> ✅ $\tau = 0.75$ 平衡最好；过高会导致误信 LLM，过低则浪费 LLM 能力

#### （2）LLM 初始化 vs LLM 全流程参与（Figure 3B）
- 即使让 Vanilla BO 使用相同的 LLM 推荐初始点，其性能仍显著低于 LABO
> ✅ 性能提升不仅来自“好起点”，更源于 **LLM 在整个优化过程中的持续引导**

#### （3）随机替代 LLM 预测（Random-Fidelity）
- 替换为均匀噪声后，性能大幅下降
> ✅ 验证了 **LLM 提供的是“有信息量”的预测**，而非简单启发

#### （4）不同 LLM 模型的影响（Figure 3C）
- 更大更强的 LLM（如 Qwen3-Thinking）带来轻微提升
- 但即使弱模型（Intern-S1-mini）也能保持稳健表现
> ✅ LABO 对 LLM 能力有一定依赖，但具有**鲁棒性**

---

### 💰 成本效益分析（Table 11）
在 **COF 任务** 中设定：
- LLM 查询：\$1
- 真实实验：\$100

| 方法 | 总轮数 | Real 查询数 | 总成本 | 是否达成 Vanilla BO 最优性能 |
|------|--------|--------------|--------|------------------------------|
| Vanilla BO | 30 | 60 | \$6000 | 是 |
| LABO | **4** | **8** | **\$964** | **是** |

> ✅ LABO 以 **仅 16% 的总成本** 达成同等甚至更好性能！

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 可作为有效的低保真度信号源**：其预测虽不完美，但蕴含丰富跨领域先验知识，可用于指导全局探索。
2. **Gating Mechanism 实现智能资源分配**：通过量化不确定性来源，自动决定何时信任 LLM、何时必须实验，极大提升样本效率。
3. **理论支持下的高效性**：论文提供了 **cumulative regret bound**，证明当 LLM 有用时，LABO 可实现 sublinear regret；即使 LLM 误导，也能退化回 Vanilla BO 水平，保证鲁棒性。
4. **广泛适用性**：在化学、材料、营养、能源乃至 AutoML 和高维问题中均表现优异。

---

### ⚠️ 方法的局限性
1. **依赖 LLM 的领域相关知识质量**：若 LLM 缺乏特定领域的训练数据，$ f_L(x) $ 可能完全无效。
2. **Prompt 设计敏感性**：虽然文中采用结构化 prompt，但在复杂任务中仍需精心设计输入格式。
3. **离散变量处理有限**：当前框架主要面向连续空间，对离散组合优化扩展需进一步研究。
4. **多目标优化未覆盖**：目前聚焦单目标最大化/最小化。

---

### 🔮 未来工作方向
1. **扩展至多保真度设置**：整合仿真模拟、半经验模型、LLM、真实实验等多层次信息源。
2. **动态调整 Gating Threshold $\tau$**：基于任务难度或历史表现自适应调节。
3. **集成多模态 LLM**：利用图文结合的 LLM 处理结构化分子图、实验图像等。
4. **应用于真实实验室闭环系统（Self-Driving Lab）**：与机器人平台对接，实现全自动科研流程。
5. **探索 LLM 自我反思机制**：让 LLM 主动评估自身预测的可信度，增强门控决策。

---

## ✅ 总结一句话
> **LABO 开创性地将 LLM 视为“认知型低保真模拟器”，通过 KOH 建模 + Gating Criterion 实现“广域探索 + 精准实验”的智能协同，在理论保障下显著提升了科学优化的样本效率与性价比。**

</details>

---

### 4. [F-TIS: Harnessing Diverse Models in Collaborative GRPO](https://arxiv.org/abs/2605.22537)

**Authors**: Nikolay Blagoev, O\u{g}uzhan Ersoy, Wendelin Boehmer, Lydia Yiyu Chen  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.22537v1  

#### Abstract
Reinforcement learning methods such as GRPO have seen great popularity in LLM post-training. In GRPO, models produce completions to a set of prompts, which are rewarded, and the policy is updated towards the relatively high reward completions. Due to the auto-regressive nature of models, the generat...

---

### 5. [LiveR: Fine-Grained Elasticity via Live Reconfiguration for Model Training](https://arxiv.org/abs/2605.22014)

**Authors**: Haoyuan Liu, Kairui Zhou, Shuyao Qi, Qinwei Yang, Shengkai Lin, Shizhen Zhao, Wei Zhang  
**Category**: cs.DC  
**Published**: 2026-05-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.22014v1  

#### Abstract
To reduce user costs and maximize cluster utilization, large model training increasingly leverages volatile but inexpensive GPU capacity, such as spot instances and reclaimable resources in shared clusters. Yet, capitalizing on these economic benefits requires jobs to adapt within the short warning ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LiveR: Fine-Grained Elasticity via Live Reconfiguration for Model Training**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLM）训练成本高昂，为降低成本，越来越多地利用**不稳定的廉价 GPU 资源**（如云上的 spot instances 或共享集群中的可回收资源）。然而，这些资源通常会在短时间内被抢占或重新分配，要求训练任务必须在极短的预警窗口内完成弹性伸缩。

现有的弹性训练系统大多采用 **stop-and-restart** 策略：通过 checkpoint 将状态落盘，重建分布式运行时，再恢复训练。这一过程导致严重的**停机时间**（downtime），包括：
- Checkpoint I/O 开销
- 进程重启、CUDA 初始化
- NCCL 通信拓扑重建
- 分布式运行时 warmup

这使得即使资源成本降低，也因频繁中断而抵消收益。

### **提出了什么新方法或新思路**
论文提出 **LiveR** —— 一种支持**细粒度弹性**（fine-grained elasticity）的**实时重配置运行时**（live reconfiguration runtime），其核心思想是：

> **将弹性伸缩视为“在线状态迁移”而非“存储驱动的恢复流程”**。

具体创新机制包括：

#### ✅ **Live Reconfiguration Architecture**
- 在当前训练世界（Active World）持续运行的同时，异步准备目标拓扑下的“影子世界”（Shadow World）
- 实现**无中断的背景初始化**，避免将 setup 成本置于关键路径上

#### ✅ **Bounded-Memory Mixed-Parallel State Handoff**
- 提出基于**张量分片几何交集计算**的传输规划算法
- 支持任意维度的并行策略重塑（TP/PP/DP），无需固定模板
- 仅使用固定大小的 staging buffer（默认 512MB–1GB），保证内存开销有界

#### ✅ **Mock Process Groups for Isolated Warmup**
- 新增节点在加入前先进入“模拟模式”，拦截所有 collective 操作
- 完成本地初始化（CUDA context、JIT 编译、autotune）而不阻塞主训练流
- 准备就绪后无缝切换至真实通信组

#### ✅ **Atomic Switch at Iteration Boundary**
- 在迭代边界处执行原子切换（sub-second metadata swap）
- 不涉及完整模型复制或进程重启
- 切换后立即以预热好的新拓扑继续训练

### **相比现有方法的优势**

| 特性 | LiveR | 传统 Checkpoint/Restart | 在线但受限系统（如 Bamboo/Oobleck） |
|------|-------|--------------------------|-------------------------------|
| **Storage-Free** | ✅ 无持久化 I/O | ❌ 严重依赖 checkpoint I/O | ✅ |
| **Init-Free** | ✅ 无进程重启 | ❌ 需要冷启动 | ✅ |
| **Arbitrary Reshaping** | ✅ 支持任意 TP/PP/DP 变化 | ✅ | ❌ 仅限固定模板 |
| **内存开销** | ✅ 有界（~1.5GB） | ✅ | ✅ |
| **适用场景** | 预告型事件（preemption warning） | 所有情况 | 有限 |

> LiveR 是首个同时实现 **Storage-Free、Init-Free 和 Topology-Flexible Reshaping** 的系统。

---

## 2. **核心实验方法和设置**

### **实验平台**
- **硬件环境**：4 节点 NVIDIA A800（80GB）PCIe 服务器，共 32 GPUs
- **网络**：200 Gbps HDR InfiniBand（非阻塞）、25 Gbps Ethernet（管理）
- **软件栈**：基于 Megatron-LM 和 PyTorch 实现，使用 NCCL 进行集合通信

### **评估模型与配置**
- 测试模型规模：GPT-1.7B 到 GPT-30B
- 并行策略组合：多种 TP（Tensor Parallelism）、PP（Pipeline Parallelism）、DP（Data Parallelism）配置
- 典型设置示例：GPT-14B, TP=4, PP=2, DP=4

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Megatron-LM Checkpoint** | 标准 stop-and-restart 流程，保存/加载 checkpoint 后重建运行时 |
| **UCP [17]** | 优化的 checkpointing 系统，支持 load-time reshaping，但仍需重启 |
| **ByteCheckpoint [32]** | 类似 UCP，高效 checkpoint 存储系统 |

> LiveR 与上述方法对比，验证其在相同重配置事件下的性能优势。

### **评估指标**
- **Reconfiguration Downtime**：从旧拓扑停止到新拓扑恢复训练的时间
- **End-to-End Speedup**：相对基线的加速比
- **Steady-State Overhead**：背景初始化对正常训练吞吐的影响
- **Training Goodput**：有效训练时间占比（排除 downtime）
- **Memory Overhead**：过渡期间额外内存占用
- **Numerical Correctness**：是否保持 bit-exact 数值一致性

此外还构建了基于 SimPy 的**大规模离散事件模拟器**，用于外推至 1,024 GPUs / 70B 参数场景，并经过物理测试床严格校准（误差 <5%）。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

| 指标 | 结果 |
|------|------|
| **平均重配置停机时间** | **2–6 秒**（vs. 基线 120–150 秒） |
| **端到端加速比** | **14× – 23×** 超越 Megatron-LM Checkpoint |
| **最大训练有效吞吐（goodput）** | 达到 **99%+**（高波动环境下） |
| **稳态干扰** | 平均迭代延迟变化 **<0.3%** |
| **内存开销** | 额外增加约 **1.5 GB**（主要来自 staging buffer 和 NCCL 元数据） |
| **数值正确性** | Bit-exact，梯度与损失轨迹完全一致 |

### **与基线方法的对比结果**

#### 🔹 图6(a)：不同模型规模下的 downtime 对比
- GPT-30B 模型下：
  - Megatron-LM Checkpoint：~150 s
  - UCP：~120 s
  - **LiveR：仅 5.8 s**
- 加速比达 **23×**

#### 🔹 图6(b)：存储带宽敏感性分析
- 当每 GPU 存储带宽降至 0.25 Gb/s 时：
  - Checkpoint 加载耗时超 **300 秒**
  - **LiveR 性能不受影响**（直接通过 InfiniBand 传输状态）

#### 🔹 图6(c)：LiveR 内部延迟分解
- **Switch 阶段**：<0.5 秒（纯元数据交换）
- **Transfer & Combine**：~2–4 秒（逐层流式传输，随模型增大线性增长）
- 其余操作（Shadow 初始化、warmup、plan 计算）全部在后台并发完成

#### 🔹 图8：高波动环境下的资源浪费对比
- 在 24 小时内发生 47 次伸缩事件：
  - Megatron-LM 浪费 **>80 GPU-hours**
  - **LiveR 仅浪费 4.1 GPU-hours**
- 重配置总暂停时间从 >130 分钟（Megatron）降至 **7 分钟**

#### 🔹 图11（模拟）：70B 模型在 1,024 GPUs 上的表现
- Cold Restart 时间：~565 秒（近 10 分钟）
- **LiveR：仅 ~11 秒**
- **加速比高达 50×**

### **消融实验结果**
- **背景初始化干扰极小**：
  - 连续 100 步训练中，平均 step time 波动仅 **0.28%**
  - Companion Manager 使用独立线程/CUDA stream，不影响主训练流
- **Graceful Degradation 设计有效**：
  - 若在传输过程中发生意外故障，可安全回退至最近 checkpoint
  - 已完成的部分 Shadow World 构建还能加速后续恢复

---

## 4. **关键结论和发现**

### **主要发现**
1. **Stop-and-Restart 是当前弹性训练的主要瓶颈**，其代价远高于 checkpoint I/O，更多体现在分布式初始化和 warmup 上。
2. **LiveR 成功将重配置从“恢复流程”转变为“流式网络操作”**，实现了秒级切换。
3. **任意维度的 mixed-parallel reshaping 是可行且高效的**，无需牺牲灵活性换取速度。
4. **在典型 spot instance 场景下（2 分钟预警）**，LiveR 的准备时间（<60s）完全可在预警窗口内完成，具备实用价值。
5. **高频率资源波动下，LiveR 显著提升训练效率**，goodput 维持在 99% 以上，而传统方法跌至 60% 以下。

### **方法的局限性**
- **仅适用于预告型事件**（pre-announced events），如 spot eviction notice、计划性调度等
- 对于突发性 fail-stop 故障（如断电、kernel panic），仍需依赖传统 checkpoint 恢复
- **当前设计串行处理重配置请求**：若多个事件密集到来，后续事件会被排队
- **强依赖高速互连网络**（如 InfiniBand）进行 P2P 数据传输，在低带宽网络中性能下降明显

### **未来工作方向**
- 支持**并发重配置**（concurrent reconfigurations）以应对极端波动场景
- 探索更智能的**预测性触发机制**：根据 spot price trend 或集群负载趋势提前启动 Shadow World 构建
- 扩展至支持 **Expert Parallelism (EP)** 和 **Sequence Parallelism (SP)** 的动态调整
- 与自动并行搜索系统（如 Alpa、Megatron 自动 TP/PP 搜索）集成，形成“决策 + 执行”闭环

---

> **总结一句话**：  
> LiveR 通过引入 **live handoff + bounded-memory streaming resharding**，首次实现了在不牺牲并行灵活性的前提下，将 LLM 弹性训练的停机时间从“分钟级”压缩至“秒级”，使 volatile 低成本 GPU 资源真正成为 LLM 训练的可行选择。

</details>

---

### 6. [Beyond Single Slot: Joint Optimization for Multi-Slot Guaranteed Display Advertising](https://arxiv.org/abs/2605.21556)

**Authors**: Zhaoqi Zhang, Jiaming Deng, Miao Xie, Linyou Cai, Qianlong Xie, Xingxing Wang, Siqiang Luo, Gao Cong  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.21556v1  

#### Abstract
Guaranteed display advertising is crucial for platform monetization, yet existing methods often operate under a single-slot assumption, limiting their ability to optimize allocation across multi-slot page views. In this paper, we propose a novel joint optimization framework for multi-slot GD allocat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Beyond Single Slot: Joint Optimization for Multi-Slot Guaranteed Display Advertising》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有 **Guaranteed Display (GD) Advertising** 方法大多基于 **single-slot modeling assumption**，即每个广告合约独立优化于单一广告位，忽略了多广告位之间的协同关系。这导致以下三大问题：
- **Slot-level 冗余（Redundancy）**：同一广告合约可能在同一页多个 slot 中重复曝光，影响用户体验。
- **Contract 不平衡（Imbalance）**：高优先级合约垄断优质位置，造成资源分配不公。
- **Exposure 集中（Concentration）**：头部广告位过度使用，尾部 slot 利用不足，流量分配失衡。

这些问题在现代平台（如 Meituan、Taobao）中尤为突出，因其页面通常包含多个广告位，单槽位建模已无法满足实际需求。

---

### 提出的新方法与思路
本文提出一种全新的 **multi-slot 联合优化框架**，突破传统单槽位范式，核心创新如下：

#### ✅ 统一的联合优化框架（Unified Joint Optimization Framework）
将广告分配问题建模为 **page view-level 的离线二分图匹配问题（offline bipartite matching）**，实现跨多个广告位的全局协调决策，而非逐 slot 独立决策。

#### ✅ Page View (PV) Constraints 模块
引入细粒度的 **每页曝光约束（PV constraints）**，限制每个广告位可服务的曝光数量，防止热门位置被过度占用，提升整体流量均衡性。

#### ✅ Contract Roulette-Based Selection 机制
设计一种基于概率轮盘的选择机制，确保：
- 每个广告合约在同一个 page view 中最多只出现在一个 slot；
- 支持多样性与公平性，避免重复展示；
- 在召回阶段通过加权采样保留高优先级合约的同时促进长尾曝光。

---

### 相比现有方法的优势
| 方面 | 现有方法（如 AUAF[3]） | 本文方法 |
|------|------------------------|---------|
| 建模范式 | 单槽位局部优化（slot-wise） | 全局 page view 级联合优化 |
| 曝光控制 | 缺乏 slot-level 流量调控 | 引入 PV constraints 实现精细控制 |
| 合约互斥 | 请求级互斥，仍允许多 slot 出现 | 页面级互斥，真正实现 one-per-page |
| 分配公平性 | 易出现头部合约垄断 | 通过正则项 + 轮盘机制增强公平性 |
| 可扩展性 | 多依赖在线贪心策略 | 离线优化 + 分布式对偶变量更新，支持大规模部署 |

> ⭐ 总体优势：实现了更高效、更公平、更具多样性的广告交付，在提升平台收入的同时保障商家 ROI 和合同履约稳定性。

---

## 2. 核心实验方法和设置

### 数据集
- 实验基于 **美团广告平台的真实线上流量数据**。
- 包含真实广告请求（ad requests）、广告合约（contracts）、用户行为日志等。
- 覆盖多种业务场景（如餐饮、酒店、旅游等本地生活服务类广告）。

---

### 实验设置
采用 **渐进式灰度发布（progressive gray-scale rollout）** 策略进行 A/B 测试：
- **两组灰度比例**：35% 和 70% 的 POI（Point of Interest）参与实验。
- **时间周期**：
  - 35% 设置：Baseline（3.29–4.2），Experiment（4.3–4.7）
  - 70% 设置：Baseline（3.27–4.1），Experiment（4.9–4.14）
- **分组方式**：按 POI 随机划分 treatment 与 control 组。
- **分析方法**：
  - A/A Test：验证组间稳定性
  - A/B Test：直接比较 treatment vs control
  - DID（Difference-in-Differences）：控制时间趋势干扰，估计净因果效应

---

### 评估指标
分为三类共六项核心指标：

| 类别 | 指标 | 说明 |
|------|------|------|
| **Merchant Efficiency** | Merchant ROI<br>Payment ROI<br>CTR<br>Payment CVR | 商家投资回报率、支付转化 ROI、点击率、付款转化率 |
| **Platform Revenue** | ARPU（Average Revenue Per User） | 平台人均收入，衡量变现效率 |
| **Contract Fulfillment** | Fulfillment Rate | 合同完成率，反映交付可靠性 |

---

### 基线方法对比
- 主要对比对象为当前生产环境中的 **AUAF 框架**（Adaptive Unified Allocation Framework），是业界主流的 GD 分配方法。
- 所有实验均以 AUAF 作为 control baseline。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 70% 灰度 DID 分析）

| 指标 | 提升幅度 | 说明 |
|------|----------|------|
| **Merchant ROI** | ↑ **42.17%** | 商家收益显著提高 |
| **Payment ROI** | ↑ **29.13%** | 用户实际购买意愿增强 |
| **CTR** | ↑ **7.67%** | 用户点击兴趣上升 |
| **Payment CVR** | ↑ **23.35%** | 点击后转化为支付的能力大幅提升 |
| **ARPU** | ↑ **28.17%** | 平台人均收入显著增长 |
| **Fulfillment Rate** | ↑ **2.12%** | 合同履约更加稳定可靠 |

> 💡 特别值得注意的是，在 **70% 流量下 A/B test 显示 ARPU 提升达 28.99%**，表明系统在大流量压力下依然保持强劲表现。

---

### 与其他设置对比结果
- 在 **35% 灰度** 下部分指标波动较大（如 A/B test 中 ARPU 下降 5.95%），说明小样本下受噪声影响明显。
- 而 **DID 分析在 70% 设置下表现出高度一致性与稳健性**，证明方法具备良好的可扩展性和鲁棒性。

---

### 消融实验（隐含分析）
虽然文中未明确列出消融实验表格，但从模块设计可推断以下关键组件的作用：
- **PV Constraints** → 控制 slot-level 曝光，提升流量均衡性 → 对 ARPU 和 Fulfillment Rate 正向贡献。
- **Contract Roulette Mechanism** → 实现页面级互斥 → 显著降低冗余曝光，改善 CTR 与 CVR。
- **Offline Bipartite Matching + Dual Update Strategy** → 支持全局最优解，避免局部贪婪陷阱 → 是性能全面提升的基础。

---

## 4. 关键结论和发现

### 主要发现
1. **多槽位联合优化优于单槽位独立决策**  
   将广告分配从“per-slot”升级到“per-page view”，能有效捕捉跨 slot 的复杂依赖关系，实现更优资源配置。

2. **细粒度曝光控制至关重要**  
   引入 **PV constraints** 成功缓解了 head-slot 过载问题，提升了整体流量利用率和公平性。

3. **页面级互斥机制显著改善用户体验**  
   **Contract Roulette** 机制有效减少了重复曝光，提高了广告新颖性和用户满意度，进而推动 CTR 和 CVR 上升。

4. **离线优化 + 分布式求解具备工业可行性**  
   尽管是离线模型，但通过对偶变量的分布式梯度更新，实现了高效求解，适用于超大规模广告系统。

5. **多方共赢：商家、平台、用户三方受益**  
   - 商家获得更高 ROI
   - 平台提升 ARPU 与履约稳定性
   - 用户看到更多样化、非重复的广告内容

---

### 方法的局限性
- **强依赖离线计算能力**：需提前预估流量与合约目标，对实时动态变化响应稍慢（尽管有 nearline 控制补偿）。
- **假设合同间独立性较强**：未显式建模合约间的竞争或互补关系（如品牌竞品互斥）。
- **Bidword 控制策略仍有优化空间**：当前 adaptive bidword 设计较启发式，未来可引入学习型策略。

---

### 未来工作方向
1. **引入时序建模与动态规划**：结合 pacing 控制，实现跨时段的联合优化。
2. **融合 LLM 或生成模型**：用于 bidword 自动生成或候选集扩展。
3. **加强公平性建模**：针对中小商家引入显式的 long-tail exposure 保护机制。
4. **探索在线-离线混合架构**：进一步缩短延迟，适应更强实时性要求。
5. **扩展至 Video/GDN 场景**：将本框架迁移至视频广告或多渠道 guaranteed delivery 场景。

---

> ✅ **总体评价**：该论文提出了一个极具工业价值的 multi-slot GD 广告分配框架，理论严谨、工程落地性强，在真实平台上取得了显著收益，代表了下一代 guaranteed advertising 系统的发展方向。

</details>

---

### 7. [Token-weighted Direct Preference Optimization with Attention](https://arxiv.org/abs/2605.21883)

**Authors**: Chengyu Huang, Zhuohang Li, Sheng-Yen Chou, Claire Cardie  
**Category**: cs.CL  
**Published**: 2026-05-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.21883v1  

#### Abstract
Direct Preference Optimization (DPO) aligns Large Language Models with human preferences without the need for a separate reward model. However, DPO treats all tokens in responses equally, neglecting the differing importance of individual tokens. Existing token-level PO methods compute the token weig...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Token-weighted Direct Preference Optimization with Attention》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统的 **Direct Preference Optimization (DPO)** 及其变体（如 IPO、KTO、SimPO）在对齐大语言模型（LLMs）与人类偏好时，将响应中的所有 token 视为同等重要。然而，不同 token 对响应质量的贡献是不均等的（例如，关键事实词 vs. 填充词），这种**均匀加权机制忽略了 token 级别的细粒度重要性差异**，导致信用分配（credit assignment）不够精确。

此外，现有的 token-level PO 方法存在以下缺陷：
- 依赖额外训练的模型来估计 token 权重（如 TIS-DPO、SePO），增加计算成本；
- 使用基于位置的启发式函数（如 D2PO 的时间衰减），缺乏语义感知能力。

---

### **提出了什么新方法或新思路**
本文提出两个核心贡献：

#### **(1) Token-weighted DPO (TwDPO)**
- 一种新的训练目标，理论基础源于 **token-weighted Reinforcement Learning (RL)**。
- 在 DPO 的基础上引入 **token-level 权重 $a_t$**，对每个 token 的 log-probability ratio 进行加权，从而实现更精细的信用分配。
- 形式上，TwDPO 的目标函数为：
  $$
  \mathcal{L}_{\text{TwDPO}} = -\mathbb{E} \left[ \log \sigma\left( \beta \sum_{t=1}^{|y_w|} a_t \log \frac{\pi_\theta(y_{wt}|x)}{\pi_{\text{ref}}(y_{wt}|x)} - \beta \sum_{t=1}^{|y_l|} a'_t \log \frac{\pi_\theta(y_{lt}|x)}{\pi_{\text{ref}}(y_{lt}|x)} \right) \right]
  $$

#### **(2) AttentionPO：TwDPO 的实例化方法**
- 利用 LLM 自身的 **attention 分数**作为 token 权重，无需额外模型或人工设计启发式规则。
- 具体流程：
  1. 将初始模型 $\pi_{\text{ref}}$ 作为 **pairwise judge**，判断哪个响应更好（输出“A”或“B”）；
  2. 提取该判决 token (`y_verdict`) 对两个响应中各 token 的 attention 权重；
  3. 经过归一化和 **attention sink 修复** 后，得到最终的 token 权重用于 TwDPO 训练。

---

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **效率** | 仅需两次前向传播获取权重，无额外训练开销 |
| **内容感知** | 权重由模型自身注意力决定，反映语义重要性而非固定位置 |
| **理论保障** | TwDPO 有严格的 RL 推导基础，支持 token-level reward 和 KL 正则 |
| **通用性** | AttentionPO 是 TwDPO 的一种实现方式，其他权重来源也可适配 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 模型类型 | 数据集 | 描述 |
|--------|-------|------|
| `LLaMA-3-8B-Base-SFT` | `ultrafeedback_binarized` (HuggingFaceH4) | 包含 61K 训练样本，每条 query 有 4 个由不同 LLM 生成的响应，经 GPT-4 打分；选择最高分为 $y_w$，其余随机选为 $y_l$ |
| `LLaMA-3-8B-Instruct` | `llama3-ultrafeedback` (Princeton-NLP) | 基于 PairRM 奖励模型打分的偏好数据集，共 59K 样本，$y_w$ 为最高分，$y_l$ 为最低分 |

---

### **实验设置和评估指标**
| 设置项 | 配置 |
|------|------|
| **基础模型** | LLaMA-3-8B-Base-SFT / LLaMA-3-8B-Instruct |
| **参考模型** | 同上（$\pi_{\text{ref}} = \pi_{\theta}$ 初始化） |
| **注意力层** | 最后一层（Layer 32） |
| **Attention Sink 修复** | 对长度 ≥5 的响应，将其前 5 个 token 的 attention 权重设为平均值并重新归一化 |
| **优化器** | AdamW，学习率 1e-6，cosine 调度，warmup 比例 0.1 |
| **KL 正则系数 $\beta$** | 0.005 |
| **Batch Size** | 32 |
| **训练轮数** | 1 epoch (~2K steps) |

---

### **评估指标**
| 基准 | 指标说明 |
|-----|---------|
| **AlpacaEval** | 对抗 GPT-4-1106-preview 的原始胜率（Win Rate）和长度控制胜率（Length-Controlled Win Rate） |
| **MT-Bench** | 多轮对话下由 GPT-4o-mini 打分的质量评分（1–10 分） |
| **Arena-Hard** | 对抗 GPT-4-0314 的胜率，测试模型在困难任务上的表现 |

---

### **基线方法对比**
- **基础模型**: $\pi_{\text{ref}}$
- **主流 PO 方法**: DPO, IPO, CPO, KTO, ORPO, R-DPO, SimPO
- **早期方法**: RRHF, SLiC-HF

所有基线均采用 Meng et al. (2024) 中调优后的超参数进行公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 3）**

#### **在 `LLaMA-3-8B-Base-SFT` 上的表现提升**
| 指标 | DPO | **AttentionPO** | 提升幅度 |
|------|-----|------------------|----------|
| AlpacaEval (WR) | 17.70% | **20.23%** | +2.53% |
| MT-Bench Score | 6.54 | **6.52** | ≈持平 |
| Arena-Hard (WR) | 38.54% | **49.72%** | **+11.18%** |

> 注意：虽然 MT-Bench 微降，但在 Arena-Hard 上显著领先。

#### **在 `LLaMA-3-8B-Instruct` 上的表现提升**
| 指标 | DPO | **AttentionPO** | 提升幅度 |
|------|-----|------------------|----------|
| AlpacaEval (WR) | 44.72% | **58.29%** | **+13.57%** |
| MT-Bench Score | 6.73 | **7.19** | **+0.46** |
| Arena-Hard (WR) | 53.98% | **52.06%** | -1.92%（略低但仍优于多数基线） |

> ✅ **AttentionPO 在 AlpacaEval 和 MT-Bench 上全面超越所有基线，尤其在 AlpacaEval 上提升达 13.57%。**

---

### **与最强基线 SimPO 的对比**
| 模型 | AlpacaEval WR | MT-Bench |
|------|---------------|-----------|
| SimPO | 47.73% | 6.51 |
| **AttentionPO** | **58.29%** (**+10.56%**) | **7.19 (+0.68)** |

> AttentionPO 显著优于当前最先进的 SimPO 方法。

---

### **消融实验结果（Ablation Studies）**

#### **(1) 是否修复 Attention Sink（w/ attn sink）**
| 方法 | AlpacaEval WR | MT-Bench | Arena-Hard |
|------|----------------|-----------|------------|
| AttentionPO (default) | 58.29% | 7.19 | 52.06% |
| w/ attn sink (未修复) | 57.27% | 6.93 | 51.64% |

> ❗ 修复 attention sink 有助于提升性能，说明初始 token 的高注意力可能是干扰信号。

#### **(2) 添加长度归一化（w/ len norm）**
| 方法 | AlpacaEval WR | MT-Bench | Arena-Hard |
|------|----------------|-----------|------------|
| AttentionPO | 58.29% | 7.19 | 52.06% |
| w/ len norm | 55.69% | 6.79 | 44.59% |

> ⚠️ 性能全面下降，表明长度归一化可能破坏了原始 RL 目标的总奖励最大化原则。

#### **(3) 不同 token 权重来源对比（Table 4）**
| 权重来源 | AlpacaEval WR | MT-Bench | Arena-Hard |
|---------|----------------|-----------|------------|
| Attention Rollout (跨层聚合) | 54.87% | 6.71 | 48.23% |
| Verbalized Weights (口头评分) | 57.76% | 6.89 | **61.41%** |

> - **Attention Rollout 表现较差**，可能因中间层注意力不适合判断；
> - **Verbalized Weights 在 Arena-Hard 上更强**，但整体仍低于 AttentionPO，且实现复杂、匹配率低（<90%）。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **token-level 加权显著提升偏好对齐效果**：AttentionPO 在多个基准上大幅超越 DPO 和 SimPO，验证了细粒度信用分配的有效性。
2. ✅ **LLM 自身注意力可作为可靠的重要性指标**：通过 prompting 模型自评并提取 attention，即可获得语义感知的 token 权重，无需额外模型。
3. ✅ **AttentionPO 高效且实用**：仅需两次前向传播，即可完成权重估计，适合大规模应用。
4. ✅ **修复 attention sink 有益**：前几个 token 的高注意力多为形式性关注（如开头句式），应予以平滑处理。
5. ❌ **长度归一化损害性能**：与直觉相反，显式长度惩罚会削弱整体奖励优化目标。

---

### **方法的局限性**
1. **计算资源限制**：目前仅在 8B 级别模型上验证，尚未扩展到更大模型（如 70B）。
2. **任务范围有限**：实验集中在 instruction-following 任务，未验证在代码生成、数学推理等领域的泛化能力。
3. **注意力解释性争议**：尽管有效，但 attention 是否真正“可解释”仍有学术争论（参见 Jain & Wallace, 2019）。
4. **注意力头选择未优化**：当前使用所有头的平均 attention，未来可探索特定 head 或组合以进一步提升性能。

---

### **未来工作方向**
1. **探索更优的注意力提取策略**：如选择特定 attention head、使用 deeper aggregation 方法（如 attention rollout 改进版）。
2. **跨模型迁移注意力权重**：初步实验显示，强模型（如 Instruct）的注意力可用于增强弱模型训练（见 Table 11），值得深入研究。
3. **应用于更多任务类型**：如长文本生成、多跳推理、代码生成等。
4. **结合其他 token 重要性信号**：如梯度范数、不确定性估计，构建混合权重方案。
5. **动态调整权重机制**：根据训练阶段或样本难度自适应地调整 token 权重分布。

---

> 🔗 **GitHub 开源地址**：[https://github.com/HCY123902/AttentionPO](https://github.com/HCY123902/AttentionPO)

</details>

---

### 8. [Hy-MT2: A Family of Fast, Efficient and Powerful Multilingual Translation Models in the Wild](https://arxiv.org/abs/2605.22064)

**Authors**: Mao Zheng, Zheng Li, Tao Chen, Bo Lv, Mingrui Sun, Mingyang Song, Jinlong Song, Hong Huang, Decheng Wu, Hai Wang, Yifan Song, Yanfeng Chen, Guanwei Zhang, Guanghua Yu, Yi Su, Hong Liu, Jinxiang Ou, Keyao Wang, Weile Chen, Haozhao Kuang, Kai Wang, Nuo Chen, Zihao Zheng, Chenhao Wang, Bin Xing, Chengcheng Xu, Tinghao Yu, Binghong Wu, Long Xu, Jiacheng Shi, Yunhao Wang, Baifang Chen, Lei Zhang, Qi Yang, Zhao Wu, Jiacheng Li, Lan Jiang, Lanrui Wang, Kai Zhang, Shuaipeng Li, Zhongzhi Chen, Weixuan Sun, Jiaqi Zhu, An Wang, Wei Li, Jun Xia, Weidong Han, Wutian Yang, Litong Hui, Luoguo Jia, Jiajia Wu, Xinpeng Zhou, Tianxiang Fei  
**Category**: cs.CL  
**Published**: 2026-05-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.22064v1  

#### Abstract
Hy-MT2 is a family of fast-thinking multilingual translation models designed for complex real-world scenarios. It includes three model sizes: 1.8B, 7B, and 30B-A3B (MoE), all of which support translation among 33 languages and effectively follow translation instructions in multiple languages. For on...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Hy-MT2: A Family of Fast, Efficient and Powerful Multilingual Translation Models in the Wild》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Hy-MT2 针对前代模型 **Hy-MT1.5** 在实际应用中暴露的多个关键问题进行了系统性优化，主要包括：
- **领域特定翻译能力不足**：在金融、法律、医学等专业领域的术语翻译准确性与一致性较差。
- **现实场景适应性弱**：对网页、会议记录、社交媒体等多样化文本格式处理不佳。
- **指令遵循能力有限**：难以可靠执行多语言中的复杂翻译约束（如风格控制、术语保留、分隔符保护等）。
- **部署效率低**：Hy-MT1.5-1.8B 的 4-bit 版本仍需 >1GB 存储，难以满足端侧低延迟需求。
- **与顶尖闭源模型存在性能差距**：相比 Gemini 3.1 Pro 和 GPT-5.5，在综合质量上有明显差距。

### 提出的新方法与思路
1. **Family-Centric Post-training (FCPT)**  
   一种三阶段后训练框架，包含：
   - **Reference-Guided On Policy Distillation (RG-OPD)**：利用高质量参考翻译进行策略蒸馏，提升基础翻译质量。
   - **Family-specific RL Training**：按语言家族（如 Romance, Sino-Tibetan）分别进行强化学习训练，构建多个“强教师”模型。
   - **Cross-family On Policy Distillation (Cross-family OPD)**：将多个语言家族的专家能力迁移至统一的学生模型，并引入通用 instruction-following 数据以增强多语言指令理解。

2. **混合专家架构 MoE 引入**  
   推出 **Hy-MT2-30B-A3B** 模型，采用 Mixture-of-Experts 架构，在保持高推理效率的同时显著提升翻译质量，实现“高性能+高效能”的平衡。

3. **超低位量化技术 AngelSlim**  
   基于自研的 **AngelSlim 技术**，实现了 **1.25-bit 极端量化**，使 1.8B 模型仅需 **440MB 存储空间**，并在 Apple A15 上实现 **1.5× 推理加速**。

4. **多维度精细化评估体系构建**  
   构建了四大专项评测基准：
   - **FLORES-200 / WMT25 / Mandarin→Minority Testset** → 通用翻译
   - **WildMTBench** → 真实业务场景（网页、社交、会议等）
   - **DomainMTBench** → 六大专业领域翻译
   - **IFMTBench** → 多语言翻译指令遵循能力

### 相比现有方法的优势
| 维度 | Hy-MT2 优势 |
|------|-------------|
| **性能** | 在多个 benchmark 上超越主流开源模型（如 DeepSeek-V4-Pro、Kimi K2.6），接近甚至超过 Gemini 3.1 Pro 和 GPT-5.5 |
| **效率** | 支持从 FP16 到 1.25-bit 的多种精度版本，适合不同资源环境部署 |
| **功能丰富性** | 可靠支持术语控制、风格迁移、分隔符保留、模板化输出等多种工业级翻译约束 |
| **轻量化能力** | 1.8B 模型经 1.25-bit 量化后仅 440MB，优于同类产品 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 类别 | 数据集名称 | 描述 |
|------|-----------|------|
| **通用翻译** | FLORES-200 | 覆盖 33 种语言共 1,056 个方向 |
| | WMT25 Human Evaluation Sets | 包含 12 个主流翻译方向的人工评估集 |
| | Mandarin→Minority Testset | 中文与少数民族语言双向翻译测试集 |
| **真实场景翻译** | WildMTBench | 自建数据集，涵盖网页、会议、书籍、社交等内容，共 2,000 条样本 |
| **领域翻译** | DomainMTBench | 自建专业领域数据集，覆盖金融、法律、医疗、科技、政治、教育六大类，共 24,000 条 |
| **指令遵循** | IFMTBench | 含 7,344 条人工对齐样本，支持中/英/德/日/法/西/韩语指令，涵盖单/多约束场景 |
| | IFBench, IFEval, MaXIFE, Multi-IF | 用于评估通用 instruction-following 能力 |

### 实验设置与评估指标
| 评估维度 | 主要指标 |
|--------|---------|
| 通用翻译 | XCOMET-XXL（有参考）、CometKiwi（无参考）、GEMBA（LLM-based） |
| 领域与真实场景 | XCOMET + GEMBA 双指标报告 |
| 指令遵循 | IFMTBench 准确率（Simple / Complex / Total）；MaXIFE（Loose/Strict/Overall）；Multi-IF（多轮表现） |

### 基线方法对比
参与比较的主要模型包括：
- **闭源模型**：Gemini 3.1 Pro, GPT-5.5
- **开源模型**：DeepSeek-V4-Pro, Kimi K2.6, Qwen3.5-397B-A17B, Qwen3.6-35B-A3B, Gemma4-26B-A4B, GLM5.1
- **商业 API**：Microsoft Translator, Doubao Translator
- **轻量模型**：Tower-Plus-72B

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ 通用翻译性能（FLORES-200 XX→XX）
| 模型 | XCOMET-XXL | 达到 Gemini 3.1 Pro 的百分比 |
|------|------------|-----------------------------|
| Hy-MT2-1.8B | 79.77 | 89.9% |
| Hy-MT2-7B | 86.89 | 97.9% |
| Hy-MT2-30B-A3B | **87.47** | **98.6%** |

> 💡 Hy-MT2-30B-A3B 已非常接近顶尖闭源模型水平。

#### ✅ WMT25 表现（三项指标平均）
| 模型 | 成绩（CometKiwi/GEMBA/XCOMET） | 对比 Hy-MT1.5-7B 提升 |
|------|-------------------------------|------------------------|
| Hy-MT1.5-7B | 61.59 / 68.85 / 75.91 | — |
| Hy-MT2-7B | **63.86 / 71.21 / 82.24** | 显著提升，尤其 GEMBA +6.33 |
| Hy-MT2-30B-A3B | 62.89 / **71.08 / 84.34** | GEMBA 超越 Gemini 3.1 Pro 和 GPT-5.5 |

#### ✅ 少数民族语言翻译（Mandarin→Minority）
| 模型 | XCOMET-XXL |
|------|-----------|
| Hy-MT1.5-7B | 60.12 |
| Hy-MT2-7B | **62.05** |
| Hy-MT2-30B-A3B | **62.44** |

> 📈 表明在低资源语言方向也有持续优化能力。

#### ✅ 领域翻译（DomainMTBench 平均 GEMBA）
| 模型 | GEMBA Score |
|------|------------|
| Gemini 3.1 Pro | 94.64 |
| Hy-MT2-30B-A3B | **89.25** |
| Hy-MT2-7B | 88.93 |
| Hy-MT1.5-7B | ~86.x（推断）|

> 🔺 比前代提升明显，接近闭源模型表现。

#### ✅ 真实场景翻译（WildMTBench GEMBA）
| 模型 | GEMBA |
|------|-------|
| Hy-MT2-30B-A3B | **89.25**（SOTA） |
| Gemini 3.1 Pro | 88.96 |
| Hy-MT2-7B | 88.93 |
| Hy-MT1.5-7B | 80.84 → 提升超 8 分 |

> ⭐ 在真实输入分布下表现出更强鲁棒性和自然度。

#### ✅ 指令遵循能力（IFMTBench Total Accuracy）
| 模型 | Simple | Complex | Total |
|------|--------|---------|-------|
| Hy-MT2-7B | 89.73 | 72.67 | 83.14 |
| Hy-MT2-30B-A3B | **90.20** | **75.94** | **84.69** |
| Kimi K2.6 | ~89.xx | ~84.xx | ~87.xx（接近） |
| GPT-5.5 | 86.97 | 83.74 | 85.72 |

> 🎯 Hy-MT2-30B-A3B 在小到中等规模模型中排名第一，尤其擅长处理**多约束指令**。

#### ✅ 轻量模型表现（Hy-MT2-1.8B vs 商业 API）
- 在 WMT25 上全面超越 **Microsoft Translator** 和 **Doubao Translator**
- 在 WildMTBench 上 GEMBA 达 **86.04**，远高于前代（80.84）
- 体积仅 **440MB（1.25-bit）**，推理速度提升 **1.5×**

---

## 4. 关键结论和发现

### 主要发现
1. **FCPT 框架有效整合多语言家族专长**  
   通过 Cross-family OPD 成功将多个 family-specific 强教师的知识迁移到统一学生模型，在不牺牲泛化能力的前提下大幅提升翻译质量。

2. **MoE 架构可在控制成本的同时突破性能瓶颈**  
   Hy-MT2-30B-A3B 在多项指标上逼近甚至超越闭源模型，验证了 MoE 在机器翻译任务中的巨大潜力。

3. **针对性优化显著提升指令遵循能力**  
   引入 instruction-following 数据并结合多教师蒸馏，使模型能精确响应术语、风格、格式等复杂要求，适用于工业级落地。

4. **极端量化不影响核心性能**  
   1.25-bit 量化版本在多数任务上性能损失极小，却大幅降低存储与计算开销，为移动端部署提供可行路径。

5. **轻量模型也能实现高质量翻译**  
   Hy-MT2-1.8B 在多个 benchmark 上超越更大模型（如 Tower-Plus-72B）及商业 API，证明“小而精”路线的成功。

### 方法的局限性
- **多轮交互指令能力较弱**  
  在 Multi-IF 后续回合中得分下降较快，说明其设计更聚焦于**单次翻译任务**而非长期对话状态维护。
- **依赖高质量教师模型**  
  FCPT 的效果高度依赖 Hy3-preview 和 Hy Instruct 等强 teacher，可能限制在其他生态下的复现性。
- **未公开训练细节全貌**  
  如 AngelSlim 的具体算法机制、RLHF 的奖励函数设计等未完全披露。

### 未来工作方向
- 扩展支持更多语言（>33 种），特别是低资源和濒危语言
- 进一步优化多轮、跨语言 instruction following 能力
- 探索全自动指令生成 pipeline，减少人工标注依赖
- 推动 1.25-bit 模型在手机、IoT 设备上的大规模部署
- 开发面向垂直行业的定制化子模型（如法律翻译专用版）

---

> ✅ **总体评价**：Hy-MT2 是一个面向真实世界复杂场景的高质量、高效率、多功能的 multilingual translation model family。它不仅在性能上媲美甚至局部超越顶尖闭源模型，还在部署灵活性、指令可控性、领域适应性等方面展现出强大工程价值，代表了当前开源机器翻译系统的先进水平。

</details>

---

### 9. [From Sequential Nodes to GPU Batches: Parallel Branch and Bound for Optimal $k$-Sparse GLMs](https://arxiv.org/abs/2605.22188)

**Authors**: Jiachang Liu, Andrea Lodi  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.22188v1  

#### Abstract
GPUs have significantly accelerated first-order methods for large-scale optimization, especially in continuous optimization. However, this success has not transferred cleanly to problems with discrete variables, combinatorial structure, and nonlinear objectives, such as certifying optimal solutions ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Sequential Nodes to GPU Batches: Parallel Branch and Bound for Optimal $k$-Sparse GLMs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**大规模稀疏广义线性模型**（sparse Generalized Linear Models, GLMs）在进行**精确最优解认证**（exact optimization certification）时面临的计算瓶颈。具体而言，传统基于 **Branch and Bound (BnB)** 的混合整数非线性规划（MINLP）求解器存在两大挑战：
- **节点级串行处理**：标准 BnB 按顺序逐个处理搜索树中的节点，难以利用现代 GPU 的并行能力。
- **频繁的 CPU-GPU 数据传输**：下界计算在 GPU 上执行，而可行解搜索、分支变量选择等关键步骤仍在 CPU 上运行，导致大量同步开销。

这些问题严重限制了 GPU 在离散优化任务中的加速潜力。

---

### 提出的新方法与新思路
作者提出了一种**简单、通用且模块化的 CPU-GPU 协同框架**，用于并行化 BnB 求解过程，其核心创新包括：

#### （1）批量 GPU 处理（Batched GPU Processing）
- 将多个开放的 BnB 节点组织成一个 batch，在 GPU 上**并行执行下界求解、可行解搜索、再优化和分支变量选择**。
- 通过 **padding 技术**统一不同节点的不规则数据结构（如自由变量集合大小不同），使得昂贵的操作（如排序）可以使用高效的批量化 GPU 内核（如 batched sorting）。

#### （2）GPU 友好的支持选择与分支机制
- **理论证明**：可以从松弛后的系数向量 $\beta^*$ 中**直接恢复**出对应的松弛指示变量 $z^*$，无需额外求解子问题。
- 利用此性质，可在 GPU 上直接基于 $\beta^*$ 进行：
  - **Rounding**：选取绝对值最大的 $k$ 个特征作为候选支持集。
  - **Branching**：选择自由变量中 $|\beta_j|$ 最大的变量进行分支。
- 所有这些操作均可通过 **gather-and-reduce** 和 **column-wise sorting** 等 GPU 高效原语实现。

#### （3）模块化设计
- 框架将 BnB 各组件（节点调度、下界求解、可行解搜索、分支策略）解耦，便于独立改进。
- 支持扩展至多 GPU 设置（node-parallel 或 row-distributed）。

---

### 相比现有方法的优势
| 维度 | 本文方法 | 现有方法（如 OKGLM） |
|------|---------|---------------------|
| 并行粒度 | **多节点批量并行** | 单节点串行处理 |
| 分支与搜索位置 | **全部在 GPU 上完成** | CPU 上执行 |
| 数据移动 | **减少频率，增大批次** | 每节点一次同步 |
| 性能 | **1–2 个数量级加速** | 加速有限 |
| 功能扩展 | 支持 **Rashomon Set 收集** | 不支持 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验分为两类：

#### （1）合成数据集（Synthetic Datasets）
- 设置：$n = p \in \{500, 1000, 2000, 4000, 8000, 16000\}$，特征间相关性 $\rho = 0.9$
- 模型类型：**Linear Regression** 和 **Logistic Regression**
- 稀疏性约束：$k = 10$
- 正则化参数：$\lambda_2 = 1.0$, $M = 2.0$

#### （2）真实世界数据集（Real-world Datasets）
- **Santander**（线性回归）：$n=4459, p=4735$
- **DOROTHEA**（逻辑回归）：$n=2300, p=89989$，高维稀疏分类任务

---

### 实验设置与评估指标
- **时间限制**：3 小时（10800 秒）
- **评估指标**：
  - **Runtime (s)**：总运行时间
  - **Optimality Gap (%)**：最终相对最优间隙
  - **Number of BnB Nodes**：探索的节点数
  - **Speedup**：相对于基线的加速比
- **硬件配置**：
  - GPU 实验：单块 **NVIDIA A100**
  - CPU 基线：AMD Milan 处理器，8 核，100GB 内存

---

### 基线方法对比
| 基线方法 | 类型 | 描述 |
|--------|------|------|
| **Gurobi** | 商业 MIP 求解器 | 使用 perspective reformulation + 外逼近法处理 logistic loss |
| **MOSEK** | 商业 MIP 求解器 | 同样使用 perspective formulation |
| **OKGLM** | 开源 SOTA 方法 | 当前最先进的开源实现，单节点 GPU 下界 + CPU 分支与搜索 |

> 注：所有方法均使用相同的 beam search warm-start 初始化。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### ✅ 在合成数据上的表现（Linear Regression）
| $p$ | Gurobi (Time/Gap) | OKGLM (Time/Gap) | **Ours (Time/Gap)** |
|-----|-------------------|------------------|--------------------|
| 16K | TL / 100%         | 228.8s / 0.00%   | **30.6s / 0.00%**  |
| 8K  | TL / 100%         | 109.5s / 0.00%   | **15.1s / 0.00%**  |
| 4K  | 9717s / 0.00%     | 87.3s / 0.00%    | **16.6s / 0.00%**  |

> ➜ **平均加速约 5–7 倍**

#### ✅ 在合成数据上的表现（Logistic Regression）
| $p$ | Gurobi (Time/Gap) | OKGLM (Time/Gap) | **Ours (Time/Gap)** |
|-----|-------------------|------------------|--------------------|
| 16K | TL / 32.52%       | 7790s / 0.00%    | **100.8s / 0.00%** |
| 500 | TL / 27.54%       | TL / 69.60%      | **4348s / 0.00%** |

> ➜ **最困难情况下仍能闭合间隙，而 OKGLM 超时且留有显著 gap**

#### ✅ 在真实数据集上的表现
| Dataset | Method | Max Gap (%) | Runtime (s) | Nodes Explored |
|--------|--------|-------------|-------------|----------------|
| **DOROTHEA ($k=45$)** | OKGLM | 0.06% | TL | ~2.2M |
| | **Ours** | **0.00%** | **2198s** | **15.8M** |
| **Santander ($k=10$)** | OKGLM | 0.00% | 52.3s |
| | **Ours** | **0.00%** | **52.3s** → 更快（见原文图表）|

> ➜ **唯一能在所有实例上认证零 optimality gap 的方法**

---

### 消融实验结果（Batch Size 影响）

- **实验设置**：固定 $n=p=1000$, $\rho=0.9$, 变化 batch size
- **结果趋势**（Figure 3）：
  - 随着 batch size 增大，运行时间显著下降。
  - 存在饱和点：
    - Linear Regression：~$2^{10} = 1024$
    - Logistic Regression：~$2^{15} = 32768$
- **原因分析**：
  - 小 batch 无法充分利用 GPU 并行性；
  - 太大 batch 受限于可用内存和队列中待处理节点数量。

> ➜ 表明 **batching 是性能提升的关键驱动因素**

---

## 4. 关键结论和发现

### 主要发现
1. **GPU 可以有效加速离散组合优化**：尽管 BnB 天然具有不规则性和串行依赖，但通过合理的 **batching + padding + GPU-efficient primitives** 设计，仍可实现显著并行化。
2. **批量处理是突破瓶颈的关键**：相比“单节点 GPU 加速”，“多节点批量处理”更能发挥 GPU 的吞吐优势。
3. **CPU-GPU 协同设计至关重要**：
   - CPU 管理树结构逻辑（队列、剪枝、生成子节点）
   - GPU 执行密集数值计算（下界、对偶、再优化）
4. **支持 Rashomon Set 收集**：同一框架可轻松扩展用于收集所有近优稀疏模型的支持集，为下游统计分析提供基础。

---

### 方法的局限性
1. **依赖于特定松弛形式**：当前方法基于 **perspective relaxation** 构建强凸下界，若更换松弛方式需重新适配。
2. **内存占用较高**：padding 和批量存储会增加显存消耗，限制最大 batch size。
3. **对极小搜索树无效**：当问题本身只需探索少量节点时，batching 的收益被初始化和调度开销抵消。
4. **目前仅适用于 $k$-sparse GLMs**：虽具通用性，但尚未推广到更复杂的 MINLP 结构。

---

### 未来工作方向
1. **扩展至其他 MINLP 模型族**：如 sparse GAMs、mixed-effects models 等。
2. **动态 batch size 调度**：根据搜索阶段（早期广度优先 vs 后期深度优先）自适应调整 batch 策略。
3. **结合启发式剪枝与学习型分支策略**：引入 ML-based variable selection 进一步减少搜索空间。
4. **分布式多机多卡架构**：支持超大规模问题的 row-distributed + node-parallel 混合并行。
5. **在线 Rashomon 分析系统**：构建端到端工具链，支持模型选择、变量重要性分析、公平性评估等。

---

## 总结

✅ **本论文成功将 GPU 的强大算力引入传统上由 CPU 主导的精确离散优化领域**，提出了一套高效、模块化、可扩展的 **CPU-GPU 并行 BnB 框架**，在 $k$-sparse GLMs 上实现了 **1–2 个数量级的速度提升**，并首次实现了对大规模实例的**完全最优性认证**。同时，其支持 **Rashomon Set 收集**的能力也为可解释机器学习提供了新的技术路径。

</details>

---

### 10. [Tool-Augmented Agent for Closed-loop Optimization,Simulation,and Modeling Orchestration](https://arxiv.org/abs/2605.20190)

**Authors**: Liyuan Deng, Shujian Deng, Yongkang Chen, Yongkang Dai, Zhihang Zhong, Linyang Li, Xiao Sun, Yilei Shi, Huaxi Huang  
**Category**: cs.AI  
**Published**: 2026-05-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.20190v1  

#### Abstract
Iterative industrial design-simulation optimization is bottlenecked by the CAD-CAE semantic gap: translating simulation feedback into valid geometric edits under diverse, coupled constraints. To fill this gap, we propose COSMO-Agent (Closed-loop Optimization, Simulation, and Modeling Orchestration),...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Tool-Augmented Agent for Closed-loop Optimization, Simulation, and Modeling Orchestration*

## 1. 论文的主要贡献和创新点

### 解决的问题
现代工业设计中的 **CAD-CAE 语义鸿沟**（CAD-CAE semantic gap）是迭代优化流程的主要瓶颈。工程师需要将高维的仿真反馈（如应力场、位移场）转化为低维、结构化的几何修改，并确保这些修改在参数化历史树中仍可执行。此外，工具链（CAD生成、网格划分、求解器等）常因再生失败、网格错误或求解不收敛而中断，导致自动化困难。

现有方法存在以下不足：
- **Derivative-free optimizers**：仅优化标量目标，忽略可执行性和失败恢复。
- **Differentiable/surrogate-based 方法**：依赖近似模型，无法直接产生原生参数化CAD编辑。
- **LLM Agents**：基于提示的方法在面对工具失败时脆弱，且标准指令微调（instruction tuning）或 RLHF 主要针对短视任务，难以支持长周期试错优化。

### 提出的新方法：COSMO-Agent
提出 **COSMO-Agent**（Closed-loop Optimization, Simulation, and Modeling Orchestration），一种**工具增强型强化学习**（tool-augmented RL）框架，用于实现可靠的闭环 CAD-CAE 迭代优化。

#### 核心创新点：
- **将 CAD-CAE 闭环建模为交互式 RL 环境**：
  - 将 CAD 编辑、再生、网格划分、求解、结果解析等步骤建模为具有显式失败状态的环境。
  - LLM 作为策略（policy），在结构化动作空间中进行参数化几何修改决策。
- **多约束奖励函数设计**（multi-constraint reward）：
  - 联合优化三个目标：
    1. **可行性**（Feasibility）：满足物理、几何、成本等工程约束。
    2. **鲁棒性**（Robustness）：成功执行并从工具链失败中恢复。
    3. **输出有效性**（Structured Validity）：保证输出为可执行的结构化格式（如 JSON），防止“奖励黑客”行为。
- **Rollout-log-based 奖励机制**：
  - 奖励信号直接从工具交互日志中解析得出，无需额外重新仿真，训练高效且真实反映执行过程。

### 相比现有方法的优势
- **端到端可靠性更高**：显式建模工具失败与恢复机制，提升实际工业场景下的稳定性。
- **更符合工程实践**：直接生成可执行的参数化 CAD 修改，而非近似或不可执行的设计。
- **长周期决策能力强**：通过 RL 训练，学会基于下游仿真反馈进行多轮试错优化，超越传统 prompting 或 imitation learning 方法。

---

## 2. 核心实验方法和设置

### 数据集
构建了一个**行业对齐的可执行 CAD-CAE 数据集**，包含约 20,000 个任务，覆盖 25 种常见工业部件类别，例如：
- Flat plate flanges（平板法兰）
- Triangular brackets（三角支架）
- Hex thin nuts（六角薄螺母）
- I-beam cantilever beams（工字梁悬臂梁）

#### 数据构成：
- **训练集**：20,000 个样本（20 类）
- **测试集**：200 个样本
- **泛化集**：100 个样本（5 类未见类别，用于评估迁移能力）

每个任务包含：
- 初始参数化 CAD 模型（基于 CadQuery 模板生成）
- 工具链配置（CAD 生成器、CAE 求解器、结果提取器、成本计算器）
- 多维度约束（物理：最大位移 ≤ δ，最大 von Mises 应力 ≤ σ_allow；经济：总成本 ≤ K）

材料库包含 5 种常见材料（如 Carbon Steel, Stainless Steel 304），提供 $E$, $\nu$, $\rho$, 单价, $\sigma_{\text{allow}}$ 等属性。

### 实验设置
- **主干模型**：基于 **Qwen3-8B** 进行训练。
- **训练框架**：采用 **InternBootcamp** 框架进行多轮交互式 rollout 和策略更新。
- **优化算法**：使用 **GRPO**（Generalized Reinforcement Policy Optimization）进行策略优化。
- **工具链**：
  - CAD Generator：CadQuery
  - CAE Solver：FreeCAD FEM + Gmsh（网格）+ CalculiX（线性静力学求解）
  - Result Extractor：解析 `.frd` 文件获取 $u_{\text{max}}$, $\sigma_{\text{max}}$
  - Cost Calculator：基于体积 × 密度 × 单价计算成本
- **预算限制**：每任务最多 15 轮交互（turns），固定 tool-call/retry 预算。

### 评估指标
| 指标 | 含义 |
|------|------|
| **FSR** (Full Success Rate) | 所有三个约束（位移、应力、成本）均满足的比例 |
| **DSR** (Displacement Satisfaction Rate) | 位移约束满足的比例 |
| **SSR** (Stress Satisfaction Rate) | 应力约束满足的比例 |
| **CSR** (Cost Satisfaction Rate) | 成本约束满足的比例 |
| **MEO** (Model Extract Output) | 可成功解析出有效 JSON 输出的比例 |
| **AS** (Average Score) | 综合得分，包含约束满足数、工具调用奖励、格式正确性 |
| **ATC** (Avg Tool Calls) | 平均每实例调用工具次数 |

### 基线方法对比
涵盖多种规模的开源与闭源 LLM：
- **开源模型**：Qwen3-8B, Intern-S1-mini, Llama-4-Scout, Qwen3-30B, Qwen3-Next, Intern-S1
- **闭源模型**：Claude-Sonnet-4.5, Gemini-3-Flash

所有模型在**相同输入、相同工具链、相同交互轮次上限、统一 JSON 输出规范**下进行公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Test Set）

| Model | Scale | FSR | DSR | SSR | CSR | MEO | AS | ATC |
|-------|-------|-----|-----|-----|-----|-----|-----|-----|
| **COSMO-Agent** | 8B | **74.5%** | **87.5%** | 76.0% | **93.5%** | **100.0%** | 0.6504 | **6.72** |
| Gemini-3-Flash | — | 67.5% | 83.0% | 75.0% | 91.0% | 98.0% | **0.6802** | 9.32 |
| Intern-S1 | 236B | 32.0% | 53.0% | 75.0% | 60.0% | 99.5% | 0.5367 | 7.44 |

#### 对比分析：
- **FSR 提升显著**：
  - 相比最强开源基线 **Intern-S1 (32.0%)**，提升 **42.5个百分点**。
  - 相比最佳闭源基线 **Gemini-3-Flash (67.5%)**，仍高出 **7.0个百分点**。
- **约束满足率全面领先**：
  - DSR 达 87.5%，CSR 达 93.5%，表明其在刚度与成本控制方面尤为出色。
- **输出可靠性最高**：MEO 达 **100%**，确保所有输出均可被解析并复现。
- **交互效率最优**：ATC = 6.72，远低于 Gemini 的 9.32，说明其以更少的工具调用达成可行解，推理更高效。

### 泛化性能（Unseen Categories）

| Model | FSR | DSR | SSR | CSR | MEO | ATC |
|-------|-----|-----|-----|-----|-----|-----|
| **COSMO-Agent** | **75.0%** | 84.0% | 78.0% | 89.0% | **100.0%** | **6.57** |
| Gemini-3-Flash | 57.0% | 60.0% | 57.0% | 60.0% | 60.0% | 9.44 |

- COSMO-Agent 在未见类别上 FSR 保持 **75.0%**，无明显退化，显示强泛化能力。
- Gemini 的 MEO 仅为 60.0%，严重影响端到端可用性。

### 消融实验（Ablation Studies）

| Setting | FSR | DSR | CSR | ATC |
|--------|-----|-----|-----|-----|
| w/o RL | 26.0% | 39.5% | 65.0% | 6.08 |
| w/o Rollout Reward | 36.0% | 59.0% | 69.0% | 2.62 |
| **COSMO-Agent** | **74.5%** | **87.5%** | **93.5%** | **6.72** |

#### 发现：
- **RL 训练至关重要**：无 RL 时 FSR 仅 26.0%，加入后跃升至 74.5%，验证了 RL 对长周期优化的有效性。
- **Rollout-log 奖励优于 re-verification**：
  - 若改用“最终 JSON 重新仿真”方式计算奖励，FSR 下降至 36.0%。
  - 模型倾向于跳过工具调用、直接猜测输出（ATC 降为 2.62），破坏闭环逻辑。
  - 证明 rollout-log 奖励能更有效地鼓励“调用工具 → 读取反馈 → 迭代”的闭环行为。

---

## 4. 关键结论和发现

### 主要发现
1. **COSMO-Agent 显著提升了小规模 LLM 在复杂工程优化任务中的表现**：
   - 一个 **8B 参数的 LLM** 经过 GRPO + rollout-log 奖励训练后，性能超越多数大规模开源甚至闭源模型。
2. **显式建模工具失败与闭环反馈是关键**：
   - 传统的 prompting 或监督微调无法支撑长周期、容错性强的优化流程。
   - 强化学习结合真实工具反馈，使 LLM 学会“试错—修正”策略。
3. **高效奖励设计避免昂贵重仿真**：
   - 基于 rollout 日志的奖励机制无需额外运行 CAE，即可准确评估策略质量，大幅提升训练效率。
4. **结构化输出与可执行性保障端到端可靠性**：
   - 100% MEO 表明模型始终输出合法 JSON，确保下游可复现，具备工业落地潜力。

### 方法的局限性
- **依赖高质量工具接口**：需为 CAD/CAE 工具提供稳定、结构化的 MCP 接口，部署门槛较高。
- **动作空间受限**：当前仅支持参数化调整，尚未扩展至拓扑变化或自由形态编辑。
- **物理范围有限**：目前聚焦线性静力学分析，未涉及非线性材料、接触、动态或多物理场耦合。

### 未来工作方向
- **扩展设计场景**：
  - 支持接触、装配体、多部件协同优化。
  - 引入非线性材料、热-力耦合、流固耦合等更复杂物理。
- **支持多样化工具后端**：
  - 兼容其他 CAD（如 SolidWorks, CATIA）和 CAE（如 ANSYS, Abaqus）系统。
- **提升可扩展性与鲁棒性**：
  - 研究更大动作空间、更紧预算、更多样失败模式下的策略学习。
  - 设计更优训练课程（curriculum learning）和鲁棒性目标。
- **探索多智能体协作**：
  - 借鉴 MARTI 等框架，实现多 LLM 协同完成子任务（如分工处理结构、材料、制造约束）。

---

> ✅ **总结一句话**：  
> COSMO-Agent 通过**工具增强的强化学习框架 + 多约束奖励 + rollout-log 奖励机制**，首次实现了在真实、不稳定 CAD-CAE 工具链中，由小型 LLM 驱动的高成功率、高效率、高可靠性的闭环优化，为 AI 驱动的智能制造提供了新范式。

</details>

---

### 11. [Don't Collapse Your Features: Why CenterLoss Hurts OOD Detection and Multi-Scale Mahalanobis Wins](https://arxiv.org/abs/2605.21493)

**Authors**: Rahul D Ray  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.21493v1  

#### Abstract
The ability to detect out-of-distribution (OOD) inputs is fundamental to safe deployment of machine learning systems. Yet, current methods often rely on feature representations that are optimised solely for classification accuracy, neglecting the distinct requirements of epistemic uncertainty. We in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Don't Collapse Your Features: Why CenterLoss Hurts OOD Detection and Multi-Scale Mahalanobis Wins*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于**Out-of-Distribution (OOD) 检测**这一关键挑战，即机器学习模型在开放世界中识别训练分布之外输入的能力。当前许多方法依赖为分类准确率优化的特征表示，忽视了**epistemic uncertainty**（认知不确定性）对特征几何结构的不同需求。

一个核心问题是：**是否更好的分类性能（如更紧凑的类内特征）自动带来更强的 OOD 检测能力？** 本文通过实验证明这一假设并不成立。

---

### 提出了什么新方法或新思路
作者提出了 **GOEN (Geometry-Optimised Epistemic Network)**，一种简单而高效的 OOD 检测流水线，其设计基于对特征几何的系统分析。GOEN 包含三个阶段：

1. **多尺度特征提取（Multi-scale Feature Extraction）**  
   使用 ResNet-18，融合 `layer2`（128维，捕捉纹理/域偏移）和 `layer4`（512维，捕捉语义）的特征，拼接成 640 维向量，提供互补信号。

2. **L2 归一化 + Mahalanobis 距离建模**  
   对特征进行 **L2 normalisation**，使其位于单位球面上，缓解“feature collapse”问题，并提升协方差矩阵的数值稳定性。随后拟合**类条件高斯分布**，使用 **Mahalanobis distance** 作为基础 OOD 分数。

3. **轻量级校准头（Calibration Head）**  
   引入一个小型 MLP 校准头，输入三个不确定性信号：
   - Log-Mahalanobis 距离
   - 最大余弦相似度（与类均值）
   - 预测熵（predictive entropy）  
   该头使用 **真实难样本 OOD 数据（如 SVHN）和合成噪声** 进行训练，输出一个校准后的 OOD 概率。

---

### 相比现有方法的优势
- **无需复杂架构**：不依赖 Deep Ensembles、变分推断或元学习，仅需单次训练 + 后处理。
- **高效可部署**：整个流程可在单 GPU 上 **20 分钟内完成训练**，推理开销极低。
- **性能领先**：在 CIFAR-10 基准上显著超越所有主流基线。
- **理论支持强**：从信息论、协方差稳定性和似然比角度为设计选择提供数学依据。
- **揭示反直觉现象**：首次系统证明 **CenterLoss 会损害 OOD 检测性能**，挑战了“更好分类 = 更好不确定性”的普遍假设。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **In-Distribution (ID)**:  
  - **CIFAR-10**：主训练和验证数据集。
- **Out-of-Distribution (OOD)**:  
  - **SVHN**：域偏移（自然场景中的数字 vs. 物体图像），用于模拟现实挑战。
  - **Filtered CIFAR-100**：语义偏移（选取与 CIFAR-10 无直接语义重叠的 50 类）。
  - **Synthetic Gaussian Noise**：极端分布偏移（完全无结构的噪声图像）。

> 注：SVHN 测试集被划分为 calibration subset（5k）和 evaluation subset（~21k），确保无数据泄露。

---

### 实验设置和评估指标

#### 模型架构
- 主干网络：**ResNet-18**（适配 32×32 输入）
- 特征维度：`layer2`（128-d） + `layer4`（512-d） → 拼接后经投影至 512-d
- 优化器：SGD（backbone）、Adam（calibration head）
- 数据增强：Random Crop, Horizontal Flip

#### 评估指标
| 指标 | 说明 |
|------|------|
| **AUROC** | 主要指标，衡量 ID/OOD 分离能力 |
| **AUPR** | 在类别不平衡下更敏感 |
| **FPR95** | 95% TPR 下的假阳性率 |
| **Detection Accuracy** | Youden Index 下的二分类准确率 |
| **ID Accuracy, ECE, NLL, Brier** | 衡量 ID 性能与校准性 |

---

### 基线方法对比
涵盖多种范式：
- **Predictive Uncertainty**: Standard NN, MC Dropout, Deep Ensemble, Evidential DL, EpiNet, MoE
- **Post-hoc OOD Detectors**: Energy Score, ODIN, Mahalanobis, KNN

所有方法共享相同的数据划分、预处理和骨干网络，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 方法 | ID Acc (%) | SVHN AUROC | CIFAR-100 AUROC | Synthetic AUROC | **Avg OOD AUROC** |
|------|------------|-----------|----------------|------------------|------------------|
| Deep Ensemble | 95.34 | 87.69 | 87.79 | 89.34 | **88.27** |
| KNN | — | 85.03 | 84.87 | 99.11 | **89.67** |
| ODIN | — | 84.66 | 85.62 | 95.81 | **88.70** |
| **GOEN-NoCenterLoss** | **93.11** | **93.72** | **90.79** | **1.0000** | **0.9483** |

> ✅ **GOEN-NoCenterLoss 在平均 AUROC 上达到 0.9483，全面超越所有基线**

---

### 与基线方法的对比结果
- **平均 AUROC 提升显著**：相比最强基线 KNN（0.8967），GOEN 提升 **+0.0516**。
- **最难任务表现最优**：
  - 在 **SVHN（域偏移）** 上 AUROC 达 **93.72**，远超第二名 EpiNet（87.77）。
  - 在 **CIFAR-100（语义偏移）** 上达 **90.79**，优于 Deep Ensemble（87.79）。
- **保持良好 ID 性能**：ID 准确率达 93.11%，虽略低于 Deep Ensemble（95.34%），但仍具竞争力。

---

### 消融实验结果（来自 Table 3）

| 变体 | Avg AUROC | 相对变化 |
|------|----------|---------|
| GOEN-Default（含 CenterLoss） | 0.9366 | 基线 |
| **NoCenterLoss** | **0.9483** | **+0.0117 ↑** |
| SingleScale-L4（仅 layer4） | 0.9270 | -0.0096 ↓ |
| NoSVHN-NoiseOnly（不用 SVHN 校准） | 0.9377 | -0.0011 ↓ |

> 🔍 **关键发现**：
> - 移除 **CenterLoss** 是最大增益来源（+1.17% AUROC）
> - 多尺度特征至关重要，尤其对 SVHN 检测
> - 使用真实硬 OOD（SVHN）进行校准显著提升泛化能力

---

## 4. 关键结论和发现

### 论文的主要发现
1. **CenterLoss 有害于 OOD 检测**  
   尽管 CenterLoss 提升了分类准确率并增强了类内紧凑性（intra/inter ratio 从 2.5 → 7.5），但它**显著降低 OOD AUROC（0.9366 → 0.9483）**。原因在于：
   - 过紧的聚类压缩了类间边界，使 OOD 样本更容易落入“安全区”
   - 协方差矩阵变得病态（ill-conditioned），削弱 Mahalanobis 距离的有效性

2. **分类几何 ≠ 不确定性几何**  
   本文挑战了“更好的分类特征自动带来更好不确定性估计”的普遍假设，强调应将 **classification objective** 与 **epistemic awareness** 解耦设计。

3. **多尺度 + L2 归一化 + Mahalanobis 是有效组合**  
   - 多尺度捕获不同层次的分布偏移
   - L2 归一化稳定协方差估计
   - Mahalanobis 利用完整协方差结构，优于 KNN 等距离方法

4. **真实硬 OOD 示例对校准至关重要**  
   仅用合成噪声无法教会模型识别真实域偏移；引入 SVHN 等真实 OOD 数据可显著提升对 domain shift 的检测能力。

---

### 方法的局限性
1. **需要少量真实 OOD 数据进行校准**  
   当前版本依赖 SVHN 等外部 OOD 数据集，若实际场景中无法获取此类数据，性能可能下降。
   
2. **尚未扩展到更大规模数据集**  
   实验集中在 CIFAR-10/SVHN，未验证在 ImageNet 或更复杂任务上的有效性。

3. **种子敏感性存在轻微波动**  
   多种子实验显示性能有一定方差（如 AUROC 从 0.9209 到 0.9448），建议报告多运行平均。

---

### 未来工作方向
1. **生成合成硬 OOD 示例**  
   探索使用生成模型（如 Diffusion 或 GAN）合成逼真的 hard OOD 数据，减少对外部数据依赖。

2. **扩展至大规模与持续学习场景**  
   将 GOEN 应用于 ImageNet，并集成到 **continual learning** 框架中以动态适应新出现的 OOD 类型。

3. **探索替代密度估计器**  
   替换 Mahalanobis 距离为更强大的密度模型，如 **normalizing flows** 或 **energy-based models**。

4. **跨架构泛化研究**  
   验证 GOEN 原则是否适用于 Vision Transformer、ConvNeXt 等非 ResNet 架构。

---

> 📌 **总结一句话**：  
> **GOEN 揭示了一个反直觉真理——“不要过度压缩你的特征”**。通过避免 CenterLoss、采用多尺度 L2 归一化特征与 Mahalanobis 距离，并结合真实 OOD 数据校准，实现了高效且领先的 OOD 检测性能，为构建“知道自己不知道”的可信 AI 提供了一条实用路径。

</details>

---

### 12. [Plug-in Losses for Evidential Deep Learning: A Simplified Framework for Uncertainty Estimation that Includes the Softmax Classifier](https://arxiv.org/abs/2605.22746)

**Authors**: Berk Hayta, Hannah Laus, Simon Mittermaier, Felix Krahmer  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.22746v1  

#### Abstract
Real-world sensor-based learning systems require uncertainty estimation that is both reliable and computationally efficient. Evidential Deep Learning (EDL) provides single-pass uncertainty estimation by modeling the class probabilities via Dirichlet distributions, where the Dirichlet parameters are ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Plug-in Losses for Evidential Deep Learning*

## 1. 主要贡献和创新点

### 解决的问题
Evidential Deep Learning (**EDL**) 是一种单次前向传播即可进行不确定性估计的方法，通过预测建模类别概率的 **Dirichlet 分布**参数来实现。然而，其理论基础与实际实现之间存在差距：
- 经典 EDL 使用基于 **Dirichlet 期望的损失函数**（如 Dirichlet-expected cross-entropy），这些目标函数形式复杂，依赖于总浓度参数（total concentration），导致优化困难、收敛慢、对超参数敏感。
- 复杂的目标函数增加了实现难度，难以与标准深度学习训练流程兼容。

### 提出的新方法与新思路
本文提出了一种**简化框架**，用“**插件损失**”（**plug-in loss**）近似原始的 EDL 目标函数：
- **核心思想**：将损失函数在 **Dirichlet 均值**（即预测概率 $p = \mathbb{I}(\alpha)$）处进行一阶泰勒展开。
- **插件损失定义**：直接在 Dirichlet 均值处计算标准分类损失，例如：
  - $ \mathcal{L}_{\text{plug}}(\alpha, y) = \text{CE}(p, y) $
  - $ \mathcal{L}_{\text{plug}}(\alpha, y) = \text{MSE}(p, y) $
- **理论保证**：在证据（evidence）充足（即 Dirichlet 总浓度 $\alpha_0$ 较大）时，插件损失与原始 EDL 损失之间的近似误差以 $O((\alpha_0 + 1)^{-1})$ 的速度衰减。

### 相比现有方法的优势
1. **实现简单**：插件损失可以直接使用 PyTorch/TensorFlow 中的标准 `cross_entropy` 或 `mse_loss` 函数，无需推导复杂的闭式表达式。
2. **训练友好**：优化行为更接近标准监督学习，收敛更快，对超参数不敏感。
3. **保留不确定性能力**：尽管损失函数被简化，模型仍能输出 Dirichlet 参数，从而计算 **vacuity** 等基于浓度的不确定性分数。
4. **统一视角**：该框架将经典的 **softmax 分类器**纳入其中——当证据映射为 $e = \exp(z)$ 且 $\alpha = e$ 时，其预测等价于 softmax。这为理解 softmax 在不确定性估计中的表现提供了新的理论视角。

---

## 2. 核心实验方法和设置

### 数据集
- **Google Speech Commands v1 (GSC V1)**：一个包含 30 个关键词的语音命令识别数据集。
- 这是首个在真实、资源受限的语音识别任务上系统评估 EDL 类方法的研究。

### 实验设置
- **骨干网络**：使用 **MatchboxNet** 架构，专为高效推理设计，符合嵌入式场景需求。
- **训练配置**：
  - 优化器：NovoGrad
  - 学习率调度：warmup-hold-decay
  - 所有变体共享相同的架构、数据预处理（MFCC + SpecAugment）、超参数和训练轮数（200 epochs），仅改变证据映射、$\alpha$ 映射和损失函数。
- **模型变体**（见 Table 1）：
  - **经典 EDL**：`EDL-CE`, `EDL-MSE`（含 KL 正则化）
  - **简化 EDL**：`Plug-in CE`, `Plug-in MSE`
  - **Softmax 风格**：`Softmax` ($T(z)=\exp(z), \phi(e)=e$), `Softplus` ($T(z)=\text{softplus}(z), \phi(e)=e$)
  - **混合模型**：`Softmax+EDL-CE`

### 评估指标
采用 **选择性预测**（selective prediction）协议评估不确定性质量：
1. **不确定性分数**：
   - **Predictive Entropy**：$ u_{\text{entropy}} = -\sum_k p_k \log p_k / \log K $
   - **Vacuity**：$ u_{\text{vacuity}} = K / \alpha_0 $ （衡量分布的离散程度）
2. **阈值策略**：按不确定性从低到高排序样本，设定阈值 $t$，只对低于 $t$ 的样本进行预测。
3. **报告指标**：
   - **Thresholded Accuracy (Acc_th)**：被接受样本的准确率。
   - **Coverage**：被接受样本占总样本的比例。
   - **Total Accuracy (Acc_total)**：$ \text{Acc\_total} = \text{Acc\_th} \times \text{Coverage} $
   - 报告在目标 Acc_th 为 99.0%、99.5%、99.9% 下的 Acc_total。

### 基线方法对比
- **经典 EDL 方法**：`EDL-CE`, `EDL-MSE`
- **简化版本**：`Plug-in CE`, `Plug-in MSE`
- **标准确定性模型**：`Softmax`, `Softplus`
- 所有方法均在相同条件下公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）
在 **99.9% 目标 Acc_th** 下，使用 **entropy** 作为不确定性的 **Acc_total**：
- `Softmax`: **88.41%**
- `Softplus`: **87.64%**
- `Plug-in EDL-CE`: **83.55%**
- `EDL-CE`: **81.61%**

### 与基线方法的对比结果
1. **简化方法媲美经典 EDL**：
   - `Plug-in EDL-CE` 和 `Plug-in EDL-MSE` 的性能与对应的 `EDL-CE` 和 `EDL-MSE` **非常接近**，验证了插件近似的有效性。
   - 证明了复杂的 Dirichlet 期望损失并非必要，简单的插件损失足以捕获主要的预测和不确定性行为。

2. **Softmax 表现强劲**：
   - 标准 `Softmax` 模型在 **entropy-based** 选择性预测上取得了**最佳或接近最佳**的 Acc_total。
   - 这支持了论文的理论观点，即 softmax 可视为简化 EDL 框架的一个特例，在实践中具有良好的不确定性估计潜力。

3. **KL 正则化的作用**：
   - KL 正则化对 **entropy** 分数影响不大。
   - 但对 **vacuity** 分数至关重要：添加 KL 正则化的 `Softmax+KL` 模型，其 vacuity-based 选择性预测性能大幅提升（99.9% 目标下 Acc_total 从 62.00% 提升至 **80.36%**）。
   - 而没有 KL 正则化的 `EDL-CE no KL` 在严格阈值下性能急剧下降。

4. **证据映射的影响**：
   - 使用 $c=0$ 的 `Softplus` 模型（无固定先验质量）在 entropy 上表现良好，但在 vacuity 上较差，进一步说明了正则化对于浓度尺度可靠性的重要性。

---

## 4. 关键结论和发现

### 主要发现
1. **EDL 损失可有效近似**：经典的 Dirichlet 期望损失可以被在 Dirichlet 均值处计算的**插件损失**很好地近似，尤其是在证据充足时，误差随浓度增加而减小。
2. **简化框架有效且实用**：提出的简化框架在保持不确定性估计能力的同时，**显著降低了实现复杂度**，并能取得与经典 EDL 相当甚至更好的性能。
3. **Softmax 的新解释**：**Softmax 分类器**自然地属于这个简化框架，这为理解其在不确定性估计中的经验成功提供了理论依据。
4. **KL 正则化是关键**：实验表明，**KL 正则化**对于提升基于浓度的不确定性分数（如 vacuity）的可靠性至关重要，它独立地调节了证据的尺度。

### 局限性
1. **单一数据集**：实验仅在 GSC V1 数据集上进行，结论在其他任务或分布偏移（distribution shift）场景下的普适性有待验证。
2. **理论假设**：近似误差的理论分析依赖于损失函数的平滑性假设，对于像交叉熵这样在边界奇异的损失，需要额外条件（如 $p_y \geq \delta > 0$）。
3. **不确定性分数融合**：论文未探索如何结合 entropy 和 vacuity 两种不同来源的不确定性分数。

### 未来工作方向
1. **更广泛的评估**：在更多数据集、不同架构以及分布外（OOD）检测场景下评估该简化框架。
2. **不确定性分数融合**：研究如何有效地结合基于概率的（entropy）和基于浓度的（vacuity）不确定性分数，例如通过双阈值选择性预测规则。
3. **新型正则化**：探索除 KL 外的其他显式不确定性感知正则化方法，以更好地控制证据的尺度和分布。
4. **理论深化**：进一步研究在何种条件下，entropy 和 vacuity 的不确定性排序会一致或分歧。

</details>

---

### 13. [PlanningBench: Generating Scalable and Verifiable Planning Data for Evaluating and Training Large Language Models](https://arxiv.org/abs/2605.20873)

**Authors**: Ziliang Zhao, Zenan Xu, Shuting Wang, Hongjin Qian, Yan Lei, Minda Hu, Zhao Wang, Shihan Dou, Zhicheng Dou, Pluto Zhou  
**Category**: cs.AI  
**Published**: 2026-05-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.20873v1  

#### Abstract
Planning is a fundamental capability for large language models (LLMs) because such complex tasks require models to coordinate goals, constraints, resources, and long-term consequences into executable and verifiable solutions. Existing planning benchmarks, however, usually treat planning data as fixe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PlanningBench**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 LLM 规划（Planning）基准存在以下局限性：
- **静态固定**：大多数基准是固定的实例集合，缺乏可扩展性和多样性。
- **难度控制粗糙**：通常通过任务长度、需求数量等表面代理来衡量难度，而非结构性因素（如约束耦合、资源稀缺性、子任务依赖等）。
- **验证支持有限**：缺乏自动化的、细粒度的验证机制，难以用于训练（尤其是强化学习）。
- **训练价值不足**：现有数据多用于评估，难以有效提升模型在复杂规划任务中的泛化能力。

这些问题导致对前沿 LLM 规划能力的诊断不够精细，且无法为训练提供高质量的反馈信号。

---

### **提出了什么新方法或新思路**
本文提出 **PlanningBench**，一个用于生成**可扩展、多样化、可验证**的规划数据的框架，其核心创新在于将“规划数据构建”从**固定收集**转变为**可控生成**。

#### **核心组件**：
1. **基于真实场景的任务与约束分类法（Taxonomy）**  
   - 由领域专家抽象出超过 **30 种任务类型**，涵盖六大类别：
     - Scheduling and Timetabling
     - Allocation and Matching
     - Shift and Workforce Scheduling
     - Routing and Travel
     - Project and Production Operations
     - Emergency Response and Public Service
   - 每类任务定义了**子任务变体**、**通用约束**、**任务特定约束**和**专门的状态约束**，形成可复用的设计空间。

2. **约束驱动的合成流水线（Constraint-driven Synthesis Pipeline）**  
   - 采用 **Generator-Responder-Critic** 闭环系统动态生成实例：
     - **Generator**：根据任务-约束配置生成自包含的规划问题。
     - **Responder**：尝试求解该问题。
     - **Critic**：基于预定义的 **Verification Checklist** 进行评分，并反馈以调整后续生成难度。
   - 支持**自适应难度增强**：当当前 Responder 能完全解决时，系统会增加中/高阶约束的比例，逐步提升挑战性。

3. **自动验证与质量控制**  
   - 每个实例附带一个**结构化检查清单（Verification Checklist）**，用于自动化验证约束满足情况和目标优化程度。
   - 引入人工质量审计流程，确保数据可用性。

4. **偏好确定性最优解（Preference for Determinate Optimal Solutions）**  
   - 在可能的情况下，优先设计具有**唯一或明确最优解**的任务，以提供更清晰的奖励信号，避免模型仅满足局部约束而忽略全局一致性。

---

### **相比现有方法的优势**

| 维度 | 现有方法（如 TravelPlanner, ChinaTravel 等） | PlanningBench |
|------|------------------------------------------|----------------|
| **数据来源** | 固定基准集 | 可控生成 |
| **领域覆盖** | 通常单一（如旅行规划） | >30 种任务类型，广泛覆盖 |
| **工具依赖** | 多需外部工具调用 | 支持纯文本（tool-free）规划 |
| **难度控制** | 表面代理（长度、轮次） | 结构化控制（约束紧度、资源稀缺、目标冲突等） |
| **训练验证** | 少数支持 | 显式支持强化学习训练 |
| **目标确定性** | 多为开放答案空间 | 偏好确定性最优解 |

> ✅ **优势总结**：PlanningBench 提供了一个**可扩展、结构化、可验证**的数据生成范式，填补了从评估到训练之间的鸿沟。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **PlanningBench 自身生成的数据**：
  - **评估集**：467 个规划实例。
  - **训练集**：300 个经过验证的 PlanningBench 实例。
- **外部测试集（用于迁移评估）**：
  - **规划相关**：`ChinaTravel`, `TravelPlanner`
  - **通用指令遵循**：`Multi-Challenge`, `Inverse IFEval`, `Collie`

---

### **实验设置和评估指标**

#### **评估协议**
- 所有模型在相同提示下进行推理。
- 使用 **GPT-oss-120b** 作为 Judge Model 对输出进行打分。

#### **核心评估指标**
| 指标 | 定义 | 说明 |
|------|------|------|
| **All-pass (%)** | 模型输出满足**所有检查项**的比例 | 衡量完整解决方案的成功率，反映全局一致性 |
| **Avg-pass (%)** | 平均每个实例满足的检查项比例 | 衡量部分合规性，反映局部约束处理能力 |
- **Gap（All-pass vs Avg-pass）** 越大，说明模型越容易出现“局部正确但全局失败”的错误。

---

### **基线方法对比**
- **模型对比**：测试了多个开源与闭源前沿 LLM，包括：
  - 闭源：`GPT-5.4-xhigh`, `Gemini-3-1-pro`
  - 开源：`DeepSeek-V3.2-thinking`, `Qwen3.5-plus-thinking`, `Seed-2.0-pro-high` 等
- **训练对比设置**（消融实验）：
  1. **Base Model**：未微调的基础模型（Qwen-A3B-30B）
  2. **Human-Authored**：人工编写、独立撰写的规划数据
  3. **Syn-NotDetOptimal**：合成数据但不强调确定性最优
  4. **Syn-PlanningBench**：本文提出的合成数据，强调确定性最优

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表 2：在 PlanningBench 上的 All-pass 和 Avg-pass 性能**

| Model | All-pass (%) | Avg-pass (%) |
|-------|--------------|--------------|
| GPT-5.4-xhigh | **63.17** | **92.35** |
| GPT-5.4-high | 58.56 | 84.60 |
| Gemini-3-1-pro | 53.25 | 88.36 |
| Seed-2.0-pro-high | 44.33 | 84.02 |
| Qwen3.5-plus-thinking | 34.03 | 77.10 |
| Qwen3-32b | 0.27 | 30.11 |
| Qwen3-8b | 0.00 | 22.79 |

> 🔍 **观察**：
> - 即使最强模型（GPT-5.4-xhigh）也仅有 **63.17%** 的完整解决率，表明任务仍具挑战性。
> - **All-pass 与 Avg-pass 差距显著**（如 GPT-5.4-medium: 90.03% vs 58.09%），说明模型常因少量关键错误导致全局失败。

---

### **与基线方法的对比结果**

#### **表 4：在外部规划基准上的迁移效果（All-pass）**

| Dataset | Base Model | Human-Authored | **Syn-PlanningBench** |
|--------|------------|----------------|------------------------|
| **ChinaTravel (All)** | 50.92 | 52.41 (+1.49) | **58.36 (+7.44)** ✅ |
| **TravelPlanner (All)** | 28.85 | 33.86 (+5.01) | **46.86 (+18.01)** ✅ |

> ✅ **结论**：使用 PlanningBench 数据训练后，在未见过的规划任务上表现显著优于其他训练方式。

#### **表 5：在通用指令遵循任务上的迁移效果（平均得分）**

| Benchmark | Base Model | Human-Authored | Syn-NotDetOptimal | **Syn-PlanningBench** |
|---------|------------|----------------|--------------------|------------------------|
| **Average** | 38.74 | 41.81 (+3.07) | 39.49 (+0.75) | **45.80 (+7.06)** ✅ |

> ✅ **结论**：PlanningBench 不仅提升规划能力，还能正向迁移至复杂推理与指令遵循任务，尤其在 `Collie` 上提升达 **+14.84**。

---

### **消融实验结果**

#### **训练动态分析（图 4）**
- **Syn-PlanningBench**：
  - `solve-all ratio` 上升最快且最终最高
  - `critic reward` 曲线最平滑，优化更稳定
- **Syn-NotDetOptimal**：
  - `solve-all ratio` 极低，即使 `solve-none` 下降，也无法保证完整成功
  - 奖励信号扩散，难以引导全局一致

> 📌 **关键发现**：**确定性最优解** 是获得稳定、可迁移训练效果的关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. **当前 LLM 在复杂耦合约束下的规划能力仍然有限**  
   即使最强模型在 PlanningBench 上的 All-pass 也不足 65%，且局部满足 ≠ 全局成功。

2. **错误主要源于“错误计算/分配”而非格式问题**  
   - **Wrong Calculation/Assignment** 占所有语义错误的 **60.9%~83.5%**
   - 表明瓶颈在于**数值、时间、逻辑决策**，而非输出格式。

3. **PlanningBench 数据可用于有效训练，且具备强迁移能力**  
   - 在外部规划和通用指令任务上均取得显著提升。
   - 尤其在需要**多步推理、约束整合、全局一致性维护**的任务中表现突出。

4. **确定性最优解提供更优的训练信号**  
   - 模糊或多解任务会导致奖励稀疏、训练不稳定。
   - 明确的目标有助于 RL 更高效地收敛到全局最优策略。

---

### **方法的局限性**
- **依赖人工设计分类法**：虽然可扩展，但初始 taxonomy 构建仍需大量专家参与。
- **尚未完全端到端自动化**：仍需人工审计与修正（尽管 86.15% 实例无需重大修改）。
- **发布延迟**：数据计划于 **2026年6月1日前** 发布，目前不可用。

---

### **未来工作方向**
- 扩展 taxonomy 至更多现实世界场景（如金融、教育、制造）。
- 探索完全自动化、无需人工干预的闭环数据生成与训练 pipeline。
- 将 PlanningBench 与 Agent 框架结合，实现动态环境中的在线规划能力进化。
- 研究如何在非确定性任务中构造有效的伪最优目标以维持训练效率。

---

> ✅ **总体评价**：  
> PlanningBench 不仅是一个新的 benchmark，更是一种**面向规划能力的新型数据工程范式**。它通过结构化生成、自动验证和确定性目标设计，为 LLM 的规划能力评估与训练提供了坚实基础，推动了从“评测”到“教学”的转变。

</details>

---

### 14. [AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions](https://arxiv.org/abs/2605.21082)

**Authors**: Minghao Chen, Xinyi Hu, Zhou Yu, Yufei Yin  
**Category**: cs.AI  
**Published**: 2026-05-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.21082v1  

#### Abstract
Large Language Model (LLM) based agents have demonstrated proficiency in multi-step interactions with graphical user interfaces (GUIs). While most research focuses on improving single-task performance, practical scenarios often involve repetitive GUI tasks for which invoking LLM reasoning repeatedly...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对**重复性图形用户界面（GUI）自动化任务**中的效率瓶颈提出解决方案。传统基于 **ReAct 范式的 LLM Agent** 虽然灵活，但在处理重复任务时每次都需要调用昂贵的 LLM 推理，导致**高 token 开销和运行时间成本**；而传统的 **Robotic Process Automation (RPA)** 虽然执行高效，但依赖人工编写脚本，开发维护成本高且对界面变化脆弱。

因此，作者旨在解决以下核心矛盾：
- 如何在保证灵活性的同时实现低开销、可复用的 GUI 自动化？
- 如何将 LLM 的强大推理能力“蒸馏”为稳定高效的 RPA 函数？

### 提出了什么新方法或新思路
作者提出了 **AutoRPA** 框架，其核心思想是：  
> 利用 LLM 驱动的 ReAct Agent 进行探索并生成成功轨迹，再通过一个两阶段流程将其“蒸馏”为鲁棒、可重用的 RPA 函数。

#### 核心创新点包括：

- **Translator-Builder Pipeline（翻译器-构建器流水线）**
  - **Translator Agent** 将 ReAct Agent 输出的硬编码动作（如 `click(index=2)`）转换为软编码动作（soft-coded actions），即基于语义属性动态定位元素（如通过 `text`, `content_description` 等），提升代码泛化能力。
  - **Builder Agent** 基于多个软编码轨迹，利用 **Retrieval-Augmented Generation (RAG)** 机制从树状轨迹数据库中检索相关信息，综合生成结构化的 RPA 函数，并引入条件逻辑增强鲁棒性。

- **Hybrid Repair Strategy（混合修复策略）**
  - 在验证生成的 RPA 函数失败后，不直接放弃，而是由 **Analyzer Agent** 分析断点状态；
  - 若可恢复，则启动 ReAct Agent 从中断处继续完成任务，产生修复示范；
  - Builder Agent 利用此“混合轨迹”进行迭代优化，实现闭环精炼。

### 相比现有方法的优势
| 方法 | 局限性 | AutoRPA 的优势 |
|------|-------|----------------|
| **ReAct-style Agents** | 每次执行都需完整 LLM 推理，token 成本极高 | 一旦生成 RPA 函数，后续执行几乎无需 LLM 调用，大幅降低成本 |
| **Skill Learning / ExpeL / ICE** | 存储原始轨迹作为技能模板，难以泛化到新场景 | 生成的是参数化、带逻辑判断的真实可执行函数，具备更强适应性 |
| **Direct NL2Code** | LLM 难以一次性写出长周期、复杂环境下的可靠 GUI 代码 | 通过逐步探索+合成方式降低生成难度，成功率更高 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个主流 GUI 自动化基准上进行：

| 数据集 | 描述 |
|--------|------|
| **AndroidWorld** | 包含 20 个真实 Android 应用的 116 种任务类型，提供截图与 Accessibility Tree，模拟真实移动端操作 |
| **WebArena** | 模拟真实网站（如 Reddit）的网页自动化环境，共 19 个任务类型，测试跨页面导航能力 |
| **MiniWoB++** | 合成 Web 环境，包含 53 个任务类型（其中 9 个为“hard”类，有反馈机制），用于精细控制变量测试 |

### 实验设置和评估指标
- **模型后端**：使用 GPT-4o、GPT-4.1、GPT-5 和 Claude-sonnet-4.5 作为 LLM backbone。
- **Building Stage**：
  - 每个 task type 抽样 N=3 个任务用于训练；
  - ReAct Agent 最多尝试 N_ref=2 次反射重试；
  - Builder Agent 最多允许 M=3 次代码重构以通过验证。
- **Testing Stage**：
  - 测试未见过的任务实例；
  - 对比不同方法的成功率（Success Rate）、平均执行时间（Time）、token 消耗量（Tokens）。

### 评估指标
- **Success (%)**：任务完成比例
- **Time (min)**：平均执行时间
- **Tokens (k)**：每任务平均消耗 token 数（越低越好）
- **Token Reduction**：相比 ReAct 方法的 token 下降百分比

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **ReAct Paradigm** | ReAct+, SeeAct, M3A, SteP |
| **Plan-and-Execute** | RCI, AdaPlanner |
| **Skill Learning + ReAct** | AutoManual, AutoGuide |
| **变体对比** | AutoRPA (code only) —— 不启用 ReAct fallback |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### AndroidWorld 结果（GPT-4.1）
| Method | Time (min) ↓ | Tokens (k) ↓ | Success (%) ↑ |
|--------|---------------|--------------|----------------|
| ReAct+ | 2.83 | 120.2 | 34.5 |
| AutoRPA | **1.76** | **14.7** | **37.0** |
| AutoRPA (code only) | 1.08 | **2.6** | 34.3 |

✅ **结论**：AutoRPA 在成功率略优的情况下，**token 消耗仅为 ReAct+ 的 ~12.2%**，减少约 **88%**。

#### MiniWoB++ 结果（GPT-4.1）
| Method | Hard Tasks (9 types) | All Tasks (53 types) |
|--------|------------------------|------------------------|
| | Tokens (k) ↓ / Success (%) ↑ | Tokens (k) ↓ / Success (%) ↑ |
| ReAct+ | 16.2 / 84.4 | 9.2 / 92.8 |
| AutoRPA | **1.4 / 91.1** | **1.4 / 95.4** |
| AutoRPA (code only) | **1.0 / 80.0** | **0.9 / 92.5** |

✅ **结论**：AutoRPA **仅用不到 10% 的 token 即超越所有基线方法**，尤其在简单任务上接近完美表现。

#### WebArena 结果（GPT-5）
- AutoRPA 与 SOTA 方法（如 M3A）达到相当的成功率（~75% vs 57%），但 **token 消耗从 164.6k 降至 30.6k，降幅达 81%**。

> 📊 总体结论：**AutoRPA 平均减少 82%~96% 的 token 使用，在多数情况下保持甚至超过原始 ReAct Agent 的成功率。**

### 与基线方法的对比结果
- 在所有三个平台上，AutoRPA 均显著优于各类 ReAct、Plan-and-Execute 和 Skill Learning 方法；
- “AutoRPA (code only)” 版本表明：即使完全关闭 fallback，仅靠生成的 RPA 函数也能取得与 ReAct 相当的表现，证明其**高度自洽性和稳定性**；
- 相比 AdaPlanner 和 AutoManual，AutoRPA **不需要大量专家示例（expert demonstrations）**，仅需少量探索即可生成高质量函数。

### 消融实验结果（Ablation Study）

| Variant of AutoRPA | Success (%) on AndroidWorld |
|--------------------|-------------------------------|
| Full AutoRPA | **51.7** |
| w/o ReAct (builder 直接生成) | 32.5 |
| w/o Translator | 40.2 |
| w/o ReAct in Repair | 45.5 |
| w/o RAG in Builder | 48.8 |

✅ **关键发现**：
- **ReAct 探索至关重要**：无探索则成功率暴跌；
- **Translator 提升泛化性**：缺少软编码转换会削弱鲁棒性；
- **Hybrid Repair 显著增益**：移除 fallback 导致性能下降；
- **RAG 改善上下文理解**：有助于 builder 更准确地生成逻辑。

此外，随着 building task 数量增加（见 Figure 5），测试成功率持续上升，说明更多轨迹能帮助生成更通用的 RPA 函数。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **LLM 可被有效用于自动化 RPA 脚本生成**：通过“探索→翻译→合成→修复”的框架，可以将 LLM 的决策逻辑转化为高效、可复用的程序代码。
2. ✅ **软编码动作 + RAG 是关键设计**：将硬编码动作转为基于语义属性的查找，结合树状检索机制，使 builder 能够访问历史上下文，从而写出更具适应性的代码。
3. ✅ **混合修复机制提升可靠性**：当 RPA 执行失败时，利用 ReAct 进行现场修复并反馈给 builder，形成闭环学习，显著提高最终代码质量。
4. ✅ **极高的 token 效率**：在相似或更好成功率下，**token 消耗降低 82%~96%**，适用于大规模部署和低成本服务。

### 方法的局限性
1. **依赖先进 MLLM 能力**：整个流程严重依赖 GPT-4.1 或更高水平的多模态大模型（MLLM），在弱模型上可能无法生成有效轨迹或代码。
2. **需要结构化 UI 信息输入**：要求环境提供 DOM 或 Accessibility Tree，以便进行元素匹配；虽然可用 OmniParser 等工具缓解，但仍构成限制。
3. **ReAct Agent 在难任务上探索能力有限**：对于非常复杂的任务，初始探索可能失败，影响后续蒸馏效果。
4. **Builder Agent 工具使用不够智能**：有时过度或不足地调用 `fetch_info` 工具，影响生成效率和准确性。

### 未来工作方向
- **自主任务生成与奖励建模**：让 LLM 自主探索任务空间并生成 reward signal，减少人工标注需求。
- **集成更强搜索算法**：如 Tree Search 或 Monte Carlo 方法，增强 ReAct Agent 在困难任务上的探索能力。
- **轻量化 RPA 执行器**：进一步压缩 RPA 中对 MLLM 的依赖（如 `ask_mllm`），追求纯规则执行。
- **跨平台统一接口支持**：扩展至桌面应用、游戏等更广泛 GUI 场景。

--- 

> 🔚 **总结一句话**：  
> **AutoRPA 成功实现了从“交互式 LLM Agent”到“静态 RPA 函数”的范式跃迁，在保持高性能的同时极大提升了 GUI 自动化的运行效率与可复用性，为实际工业应用提供了可行路径。**

</details>

---

### 15. [DeferMem: Query-Time Evidence Distillation via Reinforcement Learning for Long-Term Memory QA](https://arxiv.org/abs/2605.22411)

**Authors**: Jianing Yin, Tan Tang  
**Category**: cs.CL  
**Published**: 2026-05-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.22411v1  

#### Abstract
Large language model (LLM) agents still struggle with long-term memory question answering, where answer-supporting evidence is often scattered across long conversational histories and buried in substantial irrelevant content. Existing memory systems typically process memory before future queries are...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DeferMem: Query-Time Evidence Distillation via Reinforcement Learning for Long-Term Memory QA**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLM）代理在**长期记忆问答（Long-Term Memory QA）**任务中面临挑战：支持答案的证据通常分散在漫长的对话历史中，并被大量无关内容淹没。现有的内存系统通常在查询到来前就对记忆进行组织和压缩（如摘要、遗忘机制），这种**预处理方式是 query-agnostic 的**，可能丢失后续回答特定问题所需的关键细节。

此外，传统的基于相似度的检索（如 embedding similarity）返回的是粗粒度、高噪声的候选集，下游的 answerer 仍需从中去噪并重构出精确的证据，效率低下且易出错。

### **提出了什么新方法或新思路**
本文提出 **DeferMem**，一种将证据提炼（evidence distillation）推迟到查询时刻（query time）的长期记忆框架。其核心思想是将长期记忆 QA 解耦为两个阶段：

1. **高召回率候选检索（High-Recall Candidate Retrieval）**  
   保留原始对话历史，构建轻量级的 **segment-link 结构**，在查询时通过 embedding 相似性和 segment 链接扩展，召回覆盖所有潜在答案支持消息的候选集。

2. **查询条件化证据提炼（Query-Conditioned Evidence Distillation）**  
   引入一个可训练的 **memory distiller**，利用强化学习算法 **DistillPO**，将高召回但高噪声的候选集提炼成一组**忠实（faithful）、自包含（self-contained）、与查询相关的精简证据**，供下游 answerer 使用。

#### **DistillPO 的关键技术设计**
- **结构化动作空间（Structured Action）**：将提炼过程建模为两步操作：**有用消息选择（message selection）** 和 **证据重写（evidence rewriting）**。
- **分解与门控奖励管道（Decomposed-and-Gated Reward Pipeline）**：将最终任务奖励分解为 8 个可验证的子奖励（如格式正确性、选择有效性、证据忠实性等），并通过“漏斗式分层门控”策略，在保证依赖关系的同时，早期暴露任务级正确性反馈。
- **结构对齐的优势分配（Structure-Aligned Advantage Assignment）**：将不同类型的奖励信号分别分配给选择部分和重写部分的输出 token，提升学习效率。

### **相比现有方法的优势**
- **更高的灵活性与准确性**：避免了预处理导致的信息损失，仅在知道查询后才提炼最相关的证据。
- **更强的去噪能力**：通过 RL 训练的 distiller 能主动过滤噪声、整合碎片信息，生成 answer-ready 的证据。
- **零商业 API 成本**：内存操作不依赖商业 LLM API，仅在离线训练 distiller 时使用，部署后运行成本极低。
- **更高的效率**：相比其他先进方法，实现了更快的运行时间和更低的 token 开销。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LoCoMo**：基于人类对话的长期记忆基准，平均约 300 轮对话，16k tokens，包含单跳、多跳、时间推理、开放域和对抗性问题。
- **LongMemEval-S**：用户与 LLM 助手之间的交互记忆评估集，平均约 40 会话，115k tokens，测试信息提取、跨会话推理、知识更新、时间推理和拒绝回答能力。
- **LongMemEval-M**：用于可扩展性分析，平均约 500 会话，1.5M tokens，验证在超长上下文下的表现。

### **实验设置和评估指标**
- **下游 answerer**：统一使用 **GPT-4o-mini** 进行答案生成和判断。
- **评估指标**：
  - **Accuracy**：由 GPT-4o-mini 判断生成答案是否正确。
  - **Token Cost**：内存系统操作中调用商业 LLM API 所消耗的输入/输出 token 数量（单位：千）。
  - **Time Cost**：内存系统从接收查询到返回信息的总运行时间（秒）。
- **排除项**：最终答案生成和 judge 调用不计入上述成本，以聚焦内存系统的开销。

### **基线方法对比**
分为四类：
1. **FullText**：直接提供全部历史给 answerer。
2. **NaiveRAG**：基于 embedding 相似度检索 top-k 消息。
3. **代表性内存系统**：
   - Mem0, A-Mem, MemoryOS, MemGAS, LightMem, GAM
4. **学习型内存方法**：
   - Memory-R1（基于 RL 的内存利用）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 方法 | LongMemEval-S (Acc.) | LoCoMo (非对抗 Acc.) | LoCoMo (对抗 Acc.) | Token Cost (k) | Time (s) |
|------|------------------------|------------------------|---------------------|----------------|-----------|
| **DeferMem** | **70.00%** | **88.25%** | **97.09%** | **0.00** | **90.30** |
| LightMem | 68.64% | 72.99% | 90.81% | 85.19 | 815.32 |
| GAM | 43.00% | 79.68% | 66.59% | 100.04 | 120.19 |
| MemGAS | 60.20% | 76.88% | 82.96% | 49.21 | 123.97 |

> 注：DeferMem 在所有指标上均达到最优或次优，尤其在准确率和效率上全面领先。

### **与基线方法的对比结果**
- 在 **LongMemEval-S** 上，DeferMem 准确率比第二名 **LightMem** 高 **1.36 个百分点**，同时运行时间快 **3.1 倍**（90.30s vs 283.76s），且 **token 成本为 0**。
- 在 **LoCoMo** 上，DeferMem 在非对抗和对抗问题上分别达到 **88.25%** 和 **97.09%**，显著优于所有基线（第二名为 GAM 的 79.68% 和 84.46%）。
- 即使排除用于训练 distiller 的唯一 LoCoMo 对话实例（conv-26），DeferMem 仍保持 **87.90% / 96.99%** 的高准确率，说明具有良好的泛化能力。
- 在 **LongMemEval-M**（百万 token 级）上，DeferMem 准确率下降有限（63.00%），运行时间可控（802.43s），表现出良好的可扩展性。

### **消融实验结果**
| 变体 | LoCoMo Acc. ↓ | LME-S Acc. ↓ | 说明 |
|------|----------------|---------------|------|
| 完整 DeferMem | 89.93% | 70.00% | — |
| w/o distiller | 84.11% (-5.82) | 62.00 (-8.00) | 仅靠高召回检索不足以支撑高质量回答 |
| w/o segment-link | 86.63% (-3.30) | 66.20 (-3.80) | segment-link 结构对召回至关重要 |
| Base model (Llama-3.1-8B) | 72.02% (-17.91) | 52.20 (-17.80) | 未经训练的 distiller 性能严重下降 |
| Distiller-SFT | 80.97% (-8.96) | 63.40 (-6.60) | SFT 有帮助但不如 RL |
| Distiller-DAPO | 84.89% (-5.04) | 60.60 (-9.40) | 基础 RL 有效，但不如 DistillPO |
| w/o reward pipeline & structure-aligned adv. | 82.32% (-7.61) | 62.40 (-7.60) | DistillPO 各组件均有贡献 |

> 结论：**segment-link 结构** 和 **DistillPO 训练** 共同构成了 DeferMem 成功的关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. **延迟提炼（deferred distillation）优于预处理**：将证据提炼推迟到查询时刻，能够更精准地保留与当前问题相关的信息，避免预处理造成的信息损失。
2. **结构化 RL 显著提升提炼质量**：DistillPO 通过结构化动作、分解奖励和对齐优势分配，成功训练出高效的 memory distiller。
3. **高召回 + 精提炼 是高效路径**：先通过轻量结构实现高召回，再通过 distiller 去噪和重构，比直接检索或过度压缩更有效。
4. **DeferMem 实现了准确率与效率的双重突破**：在多个基准上达到 SOTA，同时运行最快、成本最低。

### **方法的局限性**
- **训练数据依赖**：当前 distiller 在相对较小的数据集上训练（约 300 QA 对），泛化能力仍有提升空间。
- **奖励构造依赖外部 LLM**：DistillPO 中的部分奖励（如 r5, r6, r7）依赖 GPT-4 等 judge 模型，增加了训练阶段的计算成本。
- **现有基准未针对可训练内存系统设计**：缺乏标准的训练/测试划分、黄金证据标注等，限制了进一步优化。

### **未来工作方向**
- 构建专门用于训练可学习内存系统的数据集，包含黄金证据、硬负样本和多样化领域。
- 降低奖励构造成本，例如通过选择性调用 judge 模型或使用更小的 judge 模型。
- 探索 distiller 在其他任务（如摘要、规划）中的应用。
- 将 DeferMem 与更复杂的 agent 架构（如 MemGPT）结合，实现更强大的长期交互能力。

---

> **总结**：DeferMem 通过“**高召回检索 + 查询时强化学习提炼**”的新范式，有效解决了长期记忆 QA 中证据稀疏、噪声大、预处理信息丢失等问题，在准确率、效率和成本之间取得了显著突破，为 LLM agent 的长期记忆管理提供了新的技术路径。

</details>

---

### 16. [BeLink: Biomedical Entity Linking Meets Generative Re-Ranking](https://arxiv.org/abs/2605.22501)

**Authors**: Darya Shlyk, Stefano Montanelli, Lawrence Hunter  
**Category**: cs.CL  
**Published**: 2026-05-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.22501v1  

#### Abstract
Despite recent progress, Biomedical Entity Linking (BEL) with large language models (LLMs) remains computationally inefficient and challenging to deploy in practical settings. In this work, we demonstrate that instruction-tuning of open-source generative models can offer an effective solution when a...

---

### 17. [AutoMCU: Feasibility-First MCU Neural Network Customization via LLM-based Multi-Agent Systems](https://arxiv.org/abs/2605.21560)

**Authors**: Penglin Dai, Zijie Zhou, Xincao Xu, Junhua Wang, Xiao Wu, Lixin Duan  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.21560v1  

#### Abstract
Deploying neural networks on microcontroller units (MCUs) is critical for edge intelligence but remains challenging due to tight memory, storage, and computation constraints. Existing approaches, such as model compression and hardware-aware neural architecture search (HW-NAS), often depend on proxy ...

---

### 18. [One-Way Policy Optimization for Self-Evolving LLMs](https://arxiv.org/abs/2605.22156)

**Authors**: Shuo Yang, Jinda Lu, Kexin Huang, Chiyu Ma, Shaohang Wei, Yuyang Liu, Guoyin Wang, Jingren Zhou, Li Yuan  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.22156v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has become a promising paradigm for scaling reasoning capabilities of Large Language Models (LLMs). However, the sparsity of binary verifier rewards often leads to low efficiency and optimization instability. To stabilize training, existing metho...

---

### 19. [Represented Is Not Computed: A Causal Test of Candidate Algorithmic Intermediates in a Transformer](https://arxiv.org/abs/2605.22488)

**Authors**: Ishita Darade, Sushrut Thorat  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.22488v1  

#### Abstract
Structured prompts require integrating components according to task-relevant relations. How a network implements this integration is often hard to judge in language or vision, where those relations are rarely specified precisely enough to define a candidate internal algorithm. Arithmetic offers a cl...

---

### 20. [Evolutionary Multi-Task Optimization for LLM-Guided Program Discovery](https://arxiv.org/abs/2605.22613)

**Authors**: Halil Alperen Gozeten, Xuechen Zhang, Emrullah Ildiz, Ege Onur Taga, Tara Javidi, Samet Oymak  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.22613v1  

#### Abstract
Recent LLM-guided evolutionary search methods have shown that iterative program mutation can discover strong algorithms, but they typically optimize each task independently, even when related tasks share reusable structure. We introduce Evolutionary Multi-Task Optimization (EMO) for LLM-guided progr...

---

### 21. [Towards Resilient and Autonomous Networks: A BlueSky Vision on AI-Native 6G](https://arxiv.org/abs/2605.21395)

**Authors**: Liang Wu, Kelly Wan, Mayank Darbari, Liangjie Hong  
**Category**: cs.AI  
**Published**: 2026-05-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.21395v1  

#### Abstract
The proliferation of emerging applications, such as autonomous driving and immersive experiences, demands cellular networks that are not only faster, but fundamentally more resilient and autonomous. This paper presents a BlueSky vision on how Artificial Intelligence will be natively integrated into ...

---

### 22. [Faithful-MR1: Faithful Multimodal Reasoning via Anchoring and Reinforcing Visual Attention](https://arxiv.org/abs/2605.22072)

**Authors**: Changyuan Tian, Zhicong Lu, Huaxing Liu, Xiang Wang, Shuai Li, Yu Chen, Wenqian Lv, Zichuan Lin, Juncheng Diao, Deheng Ye  
**Category**: cs.CL  
**Published**: 2026-05-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.22072v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising paradigm for advancing complex reasoning in large language models, and recent work extends RLVR to multimodal large language models (MLLMs). This transfer, however, surfaces a faithfulness challenge: faithful perception...

---

### 23. [Exploiting Multicast for Accelerating Collective Communication](https://arxiv.org/abs/2605.22428)

**Authors**: Chao Xu, Xu Zhang, Zihang Luo, Yuyan Wu, Guoxin Qian, Yufeng Yao, Chihyung Wang, Jingbin Zhou  
**Category**: cs.DC  
**Published**: 2026-05-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.22428v1  

#### Abstract
Reducing collective communication latency is a critical goal for large model training and inference in both academia and industry. Many-to-many communications, such as AllGather and AlltoAll (dispatch), are core components of modern parallelization strategies. State-of-the-art implementations of the...

---

### 24. [Models Can Model, But Can't Bind: Structured Grounding in Text-to-Optimization](https://arxiv.org/abs/2605.21751)

**Authors**: Zhiqi Gao, Albert Ge, Alexander Berenbeim, Nathaniel D. Bastian, Frederic Sala  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.21751v1  

#### Abstract
Text-to-optimization requires two separable capabilities: modeling -- choosing the right optimization structure -- and binding -- grounding every coefficient, index, and parameter in the concrete problem data. We study this via Text2Opt-Bench, a scalable benchmark of solver-verified optimization pro...

---

### 25. [Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents](https://arxiv.org/abs/2605.21768)

**Authors**: Sikuan Yan, Ahmed Bahloul, Ercong Nie, Susanna Schwarzmann, Riccardo Trivisonno, Volker Tresp, Yunpu Ma  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.21768v1  

#### Abstract
Memory-augmented LLM agents enable interactions that extend beyond finite context windows by storing, updating, and reusing information across sessions. However, training such agents with reinforcement learning in multi-session environments is challenging because memory turns the agent's past action...

---

### 26. [stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation](https://arxiv.org/abs/2605.21800)

**Authors**: Lucas Maes, Quentin Le Lidec, Luiz Facury, Nassim Massaudi, Ayush Chaurasia, Francesco Capuano, Richard Gao, Taj Gillin, Dan Haramati, Damien Scieur, Yann LeCun, Randall Balestriero  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.21800v1  

#### Abstract
World models are central to building agents that can reason, plan, and generalize beyond their training data. However, research on world models is currently fragmented, with disparate codebases, data pipelines, and evaluation protocols hindering reproducibility and fair comparison. Current practice ...

---

### 27. [ChronoMedicalWorld: A Medical World Model for Learning Patient Trajectories from Longitudinal Care Data](https://arxiv.org/abs/2605.21963)

**Authors**: Jiangyuan Wang, Xuyong Chen, Junwei He, Xu Xu, Shasha Xie, Fuman Han  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.21963v1  

#### Abstract
Long-horizon clinical simulation -- predicting how a patient's physiology evolves over years under specified interventions -- is central to chronic-disease care, yet existing electronic health record (EHR) models are predominantly discriminative, and general-purpose large language models drift under...

---

### 28. [ASAP: Attention Sink Anchored Pruning](https://arxiv.org/abs/2605.22372)

**Authors**: Jaehyuk Lee, Hanyoung Kim, Yanggee Kim, Donghun Lee  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.22372v1  

#### Abstract
Vision Transformers (ViTs) face severe computational bottlenecks due to the quadratic complexity of self-attention at high resolutions. Existing token reduction methods rely on local metrics - such as single-layer attention scores - that are inherently vulnerable to the attention sink phenomenon, wh...

---

### 29. [ChronoVAE-HOPE: Beyond Attention -- A Next-Generation VAE Foundation Model for Specialized Time Series Classification](https://arxiv.org/abs/2605.22684)

**Authors**: Jos\'e Alberto Rodr\'iguez, Luis Balderas, Miguel Lastra, Antonio Arauzo-Azofra, Jos\'e M. Ben\'itez  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.22684v1  

#### Abstract
Time Series Foundation Models (TSFMs) have become a new component of the state-of-the-art in general time series forecasting. However, adapting them to specialized classification tasks remains constrained by two interconnected challenges: the quadratic cost of standard attention mechanisms and the i...

---

### 30. [Clipping Bottleneck: Stabilizing RLVR via Stochastic Recovery of Near-Boundary Signals](https://arxiv.org/abs/2605.22703)

**Authors**: Shuo Yang, Jinda Lu, Chiyu Ma, Kexin Huang, Haoming Meng, Qihui Zhang, Yuyang Liu, Bolin Ding, Guoyin Wang, Li Yuan, Jingren Zhou  
**Category**: cs.LG  
**Published**: 2026-05-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.22703v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a central paradigm for scaling LLM reasoning, yet its optimization often suffers from training instability and suboptimal convergence. Through a systematic dissection of clipping-based GRPO-style objectives, we identify the rigid c...

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

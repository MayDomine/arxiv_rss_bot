# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-05 06:34:26 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [HyperParallel: A Supernode-Affinity AI Framework](https://arxiv.org/abs/2603.03731)

**Authors**: Xin Zhang, Beilei Sun, Teng Su, Qinghua Zhang, Chong Bao, Lei Chen, Xuefeng Jin  
**Category**: cs.DC  
**Published**: 2026-03-05  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.03731v1  

#### Abstract
The emergence of large-scale, sparse, multimodal, and agentic AI models has coincided with a shift in hardware toward supernode architectures that integrate hundreds to thousands of accelerators with ultra-low-latency interconnects and unified memory pools. However, existing AI frameworks are not de...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《HyperParallel: A Supernode-Affinity AI Framework》核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

随着大模型向 **trillion-scale 参数**、**稀疏架构（如 MoE）**、**多模态（omni-modal）** 和 **智能体式训练（agentic RL）** 发展，现有 AI 框架面临以下三大挑战：

- **编程复杂度高**：传统 SPMD（Single Program, Multiple Data）难以应对异构负载，需手动设计并维护复杂的并行策略。
- **内存利用率低**：中间状态（weights, activations, KV caches）管理繁琐，缺乏统一的 hierarchical memory pooling 支持。
- **硬件拓扑耦合紧**：并行策略与集群拓扑强绑定，导致代码不可移植、调优成本高昂。

同时，现代 **supernode 架构**（如华为 Atlas 900 / Matrix384）具备超低延迟互联（UB interconnect）、统一内存池和 4D 全连接拓扑，但现有框架无法有效利用其潜力。

---

### 提出了什么新方法或新思路

本文提出 **supernode-affinity 设计范式**，将整个 supernode 视为一个“逻辑上的巨型单机”，并在 MindSpore 中实现了 **HyperParallel 架构**，包含三个核心技术组件：

| 组件 | 功能 |
|------|------|
| **HyperOffload** | 自动化分层内存管理，通过 unified memory pooling 实现 HBM 与 DRAM 之间的自动预取与交换，解耦计算与状态存储 |
| **HyperMPMD** | 支持细粒度的 Multiple Program, Multiple Data 并行，适应 MoE、多模态、RL 等异构任务场景，提升硬件利用率 |
| **HyperShard** | 声明式并行编程接口，实现算法逻辑与并行策略的解耦，开发者仅需声明 `Layout` 即可自动生成并行方案 |

> ✅ **核心思想**：将硬件感知的调度能力内建于框架运行时与编译器中，屏蔽底层复杂性，提供类单机编程体验。

---

### 相比现有方法的优势

| 对比维度 | 现有框架（PyTorch + DeepSpeed/Megatron） | HyperParallel |
|--------|----------------------------------------|--------------|
| 并行范式 | 主要依赖 SPMD，对异构负载支持差 | 支持细粒度 MPMD，灵活应对 MoE/RL/多模态 |
| 内存管理 | 手动 offload 或 ZeRO 分片，易碎片化 | 统一内存池 + 自动预取，透明高效 |
| 编程模型 | 命令式编程，模型定义与并行策略紧耦合 | 声明式编程，完全解耦 |
| 可移植性 | 强依赖特定拓扑，迁移成本高 | 抽象设备矩阵，跨配置可复用 |
| 开发效率 | 新模型并行化需数天至数周 | <1 天完成并行策略开发与优化 |

---

## 2. 核心实验方法和设置

### 使用的数据集与模型

- **训练任务**：
  - **Llama-8B**：用于验证通用语言模型训练性能
  - **DeepSeek-V3（671B 参数，MoE 架构）**：作为典型 trillion-scale 稀疏模型代表
  - **Qwen-3 类似架构**：验证多模态支持能力
- **推理任务**：
  - 长序列生成测试（up to 128K tokens）
  - 多模态融合任务（text + image + audio）

### 实验平台

- **硬件环境**：
  - 华为 **Matrix384 supernode**（384 Ascend 910C NPUs + 192 Kunpeng CPUs）
  - UB interconnect（单跳延迟 200ns，带宽提升 15× vs PCIe）
  - 统一 DRAM 内存池 + HBM 缓存层级
- **软件基础**：MindSpore 框架扩展实现 HyperParallel

### 评估指标

| 指标 | 描述 |
|------|------|
| **迭代时间（Iteration Time）** | 每步训练耗时（秒） |
| **序列长度支持上限** | 在固定延迟约束下最大可处理 token 数 |
| **通信掩蔽率（Communication Masking Ratio）** | 通信与计算重叠程度，理想值为 90%+ |
| **pipeline bubble 比例** | 因负载不均造成的空转比例 |
| **资源利用率（Utilization）** | NPU/GPU 利用率、内存占用率 |
| **开发周期** | 新模型并行策略从开发到调优所需时间 |

### 基线方法对比

- **PyTorch + DeepSpeed-ZeRO3/Offload**
- **Megatron-LM（TP/PP/EP）**
- **JAX + Pathways（MPMD 支持）**
- **原始 MindSpore SPMD 实现**

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **HyperOffload 性能提升**

| 场景 | 指标 | 结果 |
|------|------|------|
| **Llama-8B 训练** | 迭代时间/step | 从 5.2s → **4.08s**（↑ ~21.5%） |
| **长序列推理** | 最大支持长度 | 从 71K → **123K**（↑ 73%） |

> 📌 显著缓解 HBM 内存墙问题，允许使用更简单的 1D-DP 替代复杂 ND-SPMD。

#### ✅ **HyperMPMD 效果**

| 场景 | 指标 | 结果 |
|------|------|------|
| **MoE 模型（DeepSeek-V3）** | 通信掩蔽率 | 从 61% → **90%**（接近理想） |
| **多模态训练** | pipeline bubbles | 减少 10–40%，整体性能 ↑ **~15%** |
| **RL 多任务调度** | 资源利用率 | 提升 **15%**，消除 straggler effect |

> 📌 细粒度 MPMD 成功解决异构模块间的负载失衡问题。

#### ✅ **HyperShard 开发效率提升**

| 指标 | 传统方式 | HyperShard |
|------|---------|-----------|
| 新模型并行化时间 | 5–10 天 | **<1 天** |
| 并行策略调优周期 | 数天 | **数小时** |

> 📌 声明式接口极大降低工程负担，提升研发敏捷性。

---

### 消融实验（Ablation Study）

虽然文中未明确列出表格形式的消融实验，但从多个案例分析中可推断出各组件独立贡献：

- 若关闭 **HyperOffload**，则必须启用复杂 ND-SPMD，通信开销上升 18%，且无法支持超长序列。
- 若退回到 **SPMD 模式**，在 MoE 和多模态任务中出现明显 pipeline stall，性能下降约 15–20%。
- 若采用命令式并行编程（非 HyperShard），相同功能实现代码量增加 3–5 倍，且难以迁移至不同拓扑。

---

## 4. 关键结论和发现

### 主要发现

1. **Supernode-Affinity 是下一代 AI 框架的核心设计原则**  
   必须将 supernode 视为单一逻辑计算机，才能充分发挥其统一内存、超低延迟互联和可扩展拓扑的优势。

2. **声明式 + 自动化是解决复杂性的关键路径**  
   HyperShard 的声明式接口结合编译器自动插入通信与内存操作，显著降低开发门槛。

3. **MPMD 比 SPMD 更适合未来 AI 工作负载**  
   面向 MoE、多模态、强化学习等动态、异构任务，细粒度 MPMD 能有效消除负载不均，提高资源利用率。

4. **统一内存池需与计算深度协同调度**  
   HyperOffload 证明：通过预测访问模式并提前预取，可在不牺牲性能的前提下突破 HBM 容量限制。

---

### 方法的局限性

| 局限性 | 说明 |
|-------|------|
| **硬件依赖较强** | 当前实现基于华为 Ascend 架构与 UB 协议，尚未在 NVIDIA GPU 集群上广泛验证 |
| **MPMD 调度复杂性转移至运行时** | 虽然简化了开发，但运行时调度器压力增大，可能引入额外开销 |
| **暂未支持跨数据中心训练** | 目前聚焦单 supernode 内部优化，未涉及 geo-distributed 场景 |
| **对极端稀疏路由的支持待加强** | 如专家选择高度动态的 MoE，仍需进一步优化通信路径 |

---

### 未来工作方向

1. **扩展至多 supernode 协同训练**  
   支持跨节点动态伸缩与 fault tolerance，构建更大规模 AI 超级计算机。

2. **异构任务统一调度引擎**  
   在 Pathways 思路上进一步发展，实现真正的“任意任务、任意模型”混合执行。

3. **自动化并行策略搜索（Auto-Parallelism）**  
   结合 profiling 与 ML-based 推荐，自动推荐最优 `Layout` 配置。

4. **开放生态建设**  
   将 HyperShard 接口标准化，推动其成为通用并行抽象，兼容 PyTorch/JAX 生态。

---

> 🔚 **总结一句话**：  
> HyperParallel 通过 **supernode-affinity 架构设计**，以 **HyperOffload + HyperMPMD + HyperShard** 三位一体的技术体系，系统性解决了 trillion-scale AI 模型在新型硬件上的效率与可用性难题，标志着 AI 框架进入“类操作系统”时代。

</details>

---

### 2. [Entropic-Time Inference: Self-Organizing Large Language Model Decoding Beyond Attention](https://arxiv.org/abs/2603.03310)

**Authors**: Andrew Kiruluta  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.03310v1  

#### Abstract
Modern large language model (LLM) inference engines optimize throughput and latency under fixed decoding rules, treating generation as a linear progression in token time. We propose a fundamentally different paradigm: entropic\-time inference, where decoding is governed by the flow of uncertainty ra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Entropic-Time Inference: Self-Organizing Large Language Model Decoding Beyond Attention*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大语言模型（LLM）推理系统将生成过程视为**按 token 步进的线性流程**，以固定的调度策略、注意力机制和采样参数运行。这种“token-time”范式忽略了生成本质上是一个**不确定性消解过程**（uncertainty resolution），导致大量计算资源被浪费在对输出质量影响甚微的步骤上。

具体问题包括：
- 所有 token 步骤被视为等价，即使某些步骤熵（entropy）变化极小；
- 注意力计算随上下文长度单调增长，不考虑是否真正减少了不确定性；
- 调度决策无视序列的信息状态，造成资源分配低效。

### 🚀 提出的新方法与新思路
论文提出 **Entropic-Time Inference** ——一种全新的 LLM 推理范式，其核心思想是：

> **将“推理时间”从 token index 重新定义为“熵流”（entropy flow）**，即以不可逆的不确定性减少量作为进度度量。

在此基础上，构建了一个**自组织（self-organizing）的推理架构**，通过一个统一的熵控制目标，联合调控以下三个系统级组件：
1. **Entropy-Aware Scheduling**（宏观尺度）  
   调度器基于每条序列的预期熵减（expected entropy reduction）优先分配算力，避免在已确定的尾部序列上浪费资源。
   
2. **Entropic Attention Pruning**（中观尺度）  
   动态剪枝 paged attention 中的信息冗余 KV blocks，仅保留对当前预测有显著信息贡献的内存区域。

3. **Entropy-Stabilized Sampling**（微观尺度）  
   自适应调整 sampling temperature，使生成过程稳定在目标熵水平，防止过早坍缩或过度随机。

该框架不依赖新模型结构，而是将 **entropy 提升为推理系统的“一级控制信号”**（first-class control signal），实现资源智能分配。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | Entropic-Time Inference |
|------|--------|-------------------------|
| 时间定义 | Token-indexed time | **Operational entropic time** $ T = \sum \max(0, \Delta H_t) $ |
| 控制粒度 | 固定规则（如固定 temperature） | **动态反馈控制**，全局耦合调度、注意力、采样 |
| 资源效率 | 平均主义调度，全量 attention | **按需分配**，只在高信息增益处投入计算 |
| 设计哲学 | 工程优化 | **类热力学系统建模**，视推理为非平衡信息处理过程 |

---

## 2. 核心实验方法和设置

### 📊 数据集与任务
未使用特定公开 benchmark，而是采用：
- **混合分布的 prompts**，涵盖：
  - Instruction-following
  - Long-context reasoning
  - Free-form generation
- 多样化输入长度与复杂度，模拟真实服务负载。

### ⚙️ 实验设置
- **基础模型**：预训练的 decoder-only Transformer（具体型号未指定）
- **推理后端**：基于 vLLM 扩展实现 entropic-time 控制逻辑
- **硬件环境**：统一硬件平台确保公平比较
- **对比配置**：逐层 ablation study，验证各模块贡献

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Latency** | 每个生成 token 的端到端延迟 |
| **Throughput** | 每秒生成 token 数量 |
| **Compute Efficiency ($\frac{dT}{dC}$)** | 单位资源消耗下的熵减率（核心指标） |
| **FLOPs / KV-cache usage** | 注意力计算量与缓存带宽占用 |
| **Output Quality** | 自动指标（BLEU, ROUGE） + 抽样人工评估 |
| **Entropy Collapse Rate** | 平均每步不可逆熵减 $\mathbb{E}[\Delta H]$ |

### 🆚 基线方法
- **Baseline**: 标准 vLLM 配置
  - 公平调度（fairness-based scheduling）
  - Dense attention（完整 KV-cache）
  - Fixed-temperature sampling

---

## 3. 主要实验结果和性能指标

### 📈 性能提升汇总（见 Table 1）

| Configuration | Latency (↓) | Throughput (↑) | $dT/dC$ (↑) | Quality |
|--------------|-------------|----------------|-------------|---------|
| Baseline     | 1.00        | 1.00           | 1.00        | 1.00    |
| Sampling only| 0.98        | 1.02           | 1.08        | 1.00    |
| Scheduling only | 0.88     | 1.15           | 1.12        | 1.00    |
| Attention only | 0.85      | 1.20           | 1.25        | 0.98    |
| **Full system** | **0.70** | **1.40**       | **1.55**    | **1.00** |

> ✅ **综合性能提升显著**：
- **延迟降低 30%**
- **吞吐提升 40%**
- **单位资源熵减提升 55%**
- **输出质量保持不变**

### 🔬 消融实验结果分析

#### （1）仅启用 Micro-Scale：Entropy-Stabilized Sampling
- 减少熵震荡 15–20%，加快高熵阶段收敛
- 对 throughput 和 FLOPs 改善有限
- ✔️ 主要作用：**增强生成稳定性**，防退化循环

#### （2）仅启用 Macro-Scale：Entropy-Aware Scheduling
- 降低平均延迟 10–15%，提升吞吐 12–18%
- 更好地利用 batch 资源，尤其在混合负载下
- ✔️ 主要作用：**优化资源调度效率**

#### （3）仅启用 Meso-Scale：Entropic Attention Pruning
- 注意力 FLOPs 下降 20–30%，KV-cache 带宽下降 15–25%
- 在长距离依赖任务上有轻微质量下降（~2%）
- ✔️ 显示剪枝需配合全局控制以防误删关键上下文

#### （4）完整 Entropic-Time Loop
- 展现**超可加性增益**（super-additive gains）
- 各模块协同产生自组织行为：
  - 高熵序列获得更多调度机会
  - 低熵时注意力自动稀疏化
  - 温度动态调节维持生成多样性
- ✔️ 表明：**只有当 entropy 成为全局耦合变量时，才能释放最大潜力**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **推理应以“熵流”而非“token 数”来衡量进展**  
   定义 **entropic time** $ T = \sum \max(0, \Delta H_t) $ 是更本质的操作性时间尺度。

2. **entropy 可作为有效的全局控制信号**  
   将 entropy 引入系统控制回路，能自然协调 scheduling、attention、sampling 三大模块，形成闭环反馈系统。

3. **自组织行为可自发涌现**  
   不需要中心化优化，局部熵驱动的控制律即可促成全局高效、稳定的推理动态。

4. **效率提升来自信息-资源权衡优化**  
   最大化 $ \frac{dT}{dC} $（单位成本的信息增益）比单纯加速更根本。

5. **与主流加速技术正交且兼容**  
   本方法可与 Speculative Decoding、MoE、FlashAttention 等共存并增强其效果。

### ⚠️ 方法的局限性
- **熵估计开销**：虽用 top-k 和 tail-corrected 近似降低开销，但在极短生成或低并发场景可能得不偿失。
- **模型校准敏感性**：若模型严重 overconfident（熵低估），可能导致过度剪枝；需配合保守阈值或 post-hoc calibration。
- **当前为 autoregressive 解码专用**：尚未扩展至非自回归或并行解码范式。
- **无模型改进**：仅作用于推理时系统层，不影响模型本身的能力或校准性。

### 🔮 未来工作方向
- 结合 training-time，设计支持 entropy-aware inference 的训练目标
- 扩展至 non-autoregressive 与 speculative decoding 架构
- 探索 entropy-guided MoE expert routing
- 开发轻量化 entropy estimator 硬件友好版本
- 在真实生产环境中部署并测试鲁棒性

---

## 总结

> **Entropic-Time Inference 将 LLM 推理从“机械执行”推向“智能调控”时代**。它不是简单的工程优化，而是一种**控制理论视角下的范式转变**——把推理看作一个由熵梯度驱动的、自组织的热力学过程。这种方法不仅提升了效率，更为下一代资源感知、自适应的 AI 推理系统提供了原则性设计蓝图。

</details>

---

### 3. [SENTINEL: Stagewise Integrity Verification for Pipeline Parallel Decentralized Training](https://arxiv.org/abs/2603.03592)

**Authors**: Hadi Mohaghegh Dolatabadi, Thalaiyasingam Ajanthan, Sameera Ramasinghe, Chamin P Hewa Koneputugodage, Gil Avraham, Yan Zuo, Violetta Shevchenko, Alexander Long  
**Category**: cs.DC  
**Published**: 2026-03-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.03592v1  

#### Abstract
Decentralized training introduces critical security risks when executed across untrusted, geographically distributed nodes. While existing Byzantine-tolerant literature addresses data parallel (DP) training through robust aggregation methods, pipeline parallelism (PP) presents fundamentally distinct...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SENTINEL: Stagewise Integrity Verification for Pipeline Parallel Decentralized Training 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在去中心化训练（Decentralized Training）环境中，尤其是在采用 **Pipeline Parallelism (PP)** 架构时，存在严重的安全风险。传统针对 **Data Parallelism (DP)** 设计的拜占庭容错（Byzantine-tolerant）方法无法直接应用于 PP，原因在于：

- 在 PP 中，模型被分层分布在多个节点上，各阶段之间传递的是中间 **activations** 和 **gradients**，而非可聚合的完整梯度。
- 恶意节点可以通过篡改这些中间信号来破坏训练过程，且错误会因非线性传播而放大，形成“级联效应”（cascading effect），导致诚实节点被误判。

因此，本文旨在解决 **PP 架构下的完整性验证问题**，确保在不可信分布式环境中进行大规模 LLM 训练时的安全性。

---

### 提出了什么新方法或新思路

作者提出了 **SENTINEL**，一种轻量级、无计算冗余的逐阶段完整性验证机制，其核心思想包括：

- **Verifier Nodes**：引入可信的“验证者节点”作为阶段间的中介，负责监控所有跨阶段通信。
- **Momentum-based Monitoring**：利用 **指数移动平均（EMA）** 对每个阶段的激活值和梯度建立统计基准，通过检测当前信号与 EMA 基准之间的偏差来识别异常行为。
- **多维度距离度量**：结合多种统计距离函数（如 L1、L2、SFR、SWD）以增强对不同类型攻击的鲁棒性。
- **自适应阈值机制**：基于 **IQR（Interquartile Range）** 动态调整检测阈值，自动适应训练过程中的分布漂移。
- **级联效应处理机制**：当某节点被标记为恶意时，通知下游验证器暂停对该批次的检测，并用 EMA 替代污染梯度，防止误报扩散。

---

### 相比现有方法的优势

| 维度 | 传统方法（如计算复制） | SENTINEL |
|------|------------------------|----------|
| **计算开销** | 高（需复制计算资源） | 极低（仅需 CPU 节点运行 EMA） |
| **吞吐影响** | 减半（一半资源用于验证） | 几乎无影响 |
| **适用架构** | 仅适用于 DP | 专为 PP 设计 |
| **检测能力** | 可靠但代价高 | 轻量高效，支持早期检测 |
| **理论保障** | 通常无收敛分析 | 提供收敛性保证（Theorem 3.1） |

> ✅ **核心优势**：SENTINEL 在不牺牲训练效率的前提下，实现了对 PP 架构中恶意行为的有效防御。

---

## 2. 核心实验方法和设置

### 使用的数据集

- **C4 (Common Crawl)**：广泛使用的预训练文本语料。
- **FineWeb / FineWeb-EDU**：大规模网页文本数据集，用于训练和评估。
- **OpenWebText**：开源版 WebText 数据集。

---

### 实验设置

- **模型**：
  - 主要使用 **Llama-3-0.6B**（16 层，1024 隐藏维）。
  - 扩展至 **1.2B、4B 参数模型** 进行大规模测试。
  - 也测试了 **MoE 架构**（Llama-4-0.4B, DeepSeek-V3-1B）。
- **并行配置**：
  - 采用 **8×16 或 16×16 的 DP-PP mesh**，最多使用 **176 名 worker**。
  - 模拟地理分布环境，部署于 AWS 实例。
- **恶意节点设定**：
  - 每阶段随机指定 **25%-37.5% 的 worker 为恶意**。
  - 攻击模式包括：常量攻击、随机值、缩放、符号翻转、延迟、偏置添加、隐形噪声（Invisible Noise）等。
- **验证机制**：
  - Verifier Nodes 部署在 Trainer Nodes 上，共享通信路径。
  - 使用 **EMA（β_h=0.9, β_g=0.8）** 作为基准。
  - 自适应 IQR 阈值更新，目标假阳性率 <1%。

---

### 评估指标

| 指标 | 含义 |
|------|------|
| **Precision (Pr)** | 被标记为恶意的节点中，确实是恶意的比例 |
| **Recall (Re)** | 所有恶意节点中，被成功检测出的比例 |
| **F1-score** | Pr 和 Re 的调和平均，综合评价检测效果 |
| **Detection Speed** | 从攻击开始到被封禁的平均迭代步数 |
| **Validation Loss** | 最终模型性能指标，衡量训练是否收敛 |

---

### 基线方法对比

- **No Verification (Vanilla)**：无任何防护机制，作为性能下限。
- **Duplicate Work (Lu et al., 2024)**：经典冗余验证方法，将一半 worker 用于重复计算以验证。
- **Robust Aggregators (Krum/Bulyan)**：用于对比 DP 轴上的防御机制是否兼容。

> ⚠️ 注意：SENTINEL 与 DP 防御机制是正交的，可同时部署。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 表格 1：Llama-3-0.6B 在 C4 上的攻击检测性能（部分）

| 攻击类型 | F1-score (SENTINEL) | Val Loss (SENTINEL) | Val Loss (No Verif.) |
|---------|--------------------|---------------------|-----------------------|
| Constant (Zeros) | 100.0% | 3.809 | 11.761 |
| Random Value | 100.0% | 3.827 | 7.778 |
| Scaling (σ=-1) | 100.0% | 3.824 | 4.109 |
| Invisible Noise (99%) | 100.0% | 3.826 | 7.682 |
| Delay (100-step) | 94.1% | 3.841 | 7.675 |

✅ **结果表明**：SENTINEL 成功检测并缓解了几乎所有强攻击，保持验证损失接近正常水平。

---

### 与基线方法的对比结果

| 方法 | F1-score | TPS (Tokens/sec) | 说明 |
|------|----------|------------------|------|
| Duplicate Work | 100.0% | 6483 | 安全但吞吐减半 |
| **SENTINEL (Ours)** | **89.5%** | **12966** | **安全且吞吐翻倍** |

> 📌 结论：SENTINEL 在检测精度略低于完美复制的情况下，实现了 **2 倍以上的训练速度提升**。

---

### 消融实验结果

#### （1）EMA 的必要性（Table 5）
| 方法 | F1-score | Val Loss |
|------|----------|----------|
| 使用瞬时均值（β=0） | 37.5% | 6.248 |
| SENTINEL (β_h=0.9, β_g=0.8) | **100.0%** | **3.826** |

➡️ 证明 **EMA 的时间平滑特性** 对检测至关重要。

#### （2）距离度量的影响（Table 6）
| 度量方式 | F1-score |
|--------|---------|
| Only SFR | 56.6% |
| Only L2 | 75.0% |
| **All Metrics (SENTINEL)** | **87.8%** |

➡️ 多种度量组合显著提升鲁棒性。

#### （3）Warm-up 期影响（Fig. 3a）
- 初始阶段 Precision 较低（波动大），约 **1000 步后趋于稳定**。
- 强调了 warm-up 阶段对建立可靠 EMA 基准的重要性。

#### （4）攻击者协同（Collusion）容忍度（Fig. 3b）
- 当恶意节点协同比例 <60% 时，检测性能保持稳定。
- 超过此阈值后假阳性上升，但仍优于无防护系统。

---

## 4. 关键结论和发现

### 主要发现

1. **Pipeline Parallelism 存在独特安全漏洞**：现有 DP 安全机制无法应对 PP 中的中间信号篡改问题。
2. **SENTINEL 实现了高效验证**：通过轻量级 EMA 监控，在几乎零开销下实现高精度检测（F1 > 85%）。
3. **弱攻击可容忍**：图 1 显示，未被检测到的弱攻击对验证损失影响极小，符合 Theorem 3.1 的理论预测——未被检测的扰动只会使解偏离最优解一个与阈值成正比的距离。
4. **可扩展性强**：成功应用于 **4B 参数 LLM** 和 **MoE 架构**，并在 **SWARM** 框架中集成，验证了其现实可用性。
5. **与 DP 防御正交**：SENTINEL 可与 Krum、Bulyan 等 DP 防御共存，提供双重安全保障。

---

### 方法的局限性

1. **依赖 Honest Majority 假设**：要求每阶段恶意节点占比 < 50%，否则可能失效。
2. **EMA 初始化依赖 Warm-up**：需要一段无攻击的初始训练来建立可靠基准。
3. **未覆盖所有攻击类型**：如后门攻击（backdoor）、成员推断（membership inference）等不在考虑范围内。
4. **Verifier Nodes 需可信**：若验证器本身被攻破，则整个系统崩溃（但成本低，易于集中控制）。

---

### 未来工作方向

1. 探索基于 **神经网络的异常检测器** 替代手工设计的距离度量。
2. 研究 **异步训练场景下的动态阈值调整**。
3. 将验证机制扩展至 **All-Reduce 阶段**，实现端到端保护。
4. 探索 **无需 warm-up 的在线学习型验证器**。
5. 分析 **Stochastic Wiring 与 Cascading Effect 的交互影响**，优化 SWARM 中的检测策略。

---

> 🔚 **总结**：  
> SENTINEL 是首个针对 **Pipeline Parallel Decentralized Training** 的完整性验证框架，填补了拜占庭容错领域在 PP 架构上的空白。它以极低开销实现了高鲁棒性的攻击检测，为未来开放协作式 LLM 训练提供了坚实的安全基础。

</details>

---

### 4. [When Small Variations Become Big Failures: Reliability Challenges in Compute-in-Memory Neural Accelerators](https://arxiv.org/abs/2603.03491)

**Authors**: Yifan Qin, Jiahao Zheng, Zheyu Yan, Wujie Wen, Xiaobo Sharon Hu, Yiyu Shi  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.03491v1  

#### Abstract
Compute-in-memory (CiM) architectures promise significant improvements in energy efficiency and throughput for deep neural network acceleration by alleviating the von Neumann bottleneck. However, their reliance on emerging non-volatile memory devices introduces device-level non-idealities-such as wr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When Small Variations Become Big Failures: Reliability Challenges in Compute-in-Memory Neural Accelerators*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题  
该论文聚焦于 **Compute-in-Memory (CiM)** 神经网络加速器在实际部署中面临的关键可靠性挑战。尽管 CiM 架构通过缓解冯·诺依曼瓶颈显著提升了能效和吞吐量，但其依赖的 **新兴非易失性存储器（NVM）** 存在固有非理想特性（如写入变异性、电导漂移、随机噪声），导致模型权重引入噪声，进而引发推理精度下降甚至**灾难性失败**。

特别地，作者指出：  
- 传统研究多关注**平均情况下的精度表现**（average-case accuracy），而忽视了**尾部风险**（tail failures）；
- 即使单个设备变异很小，某些罕见但合理的组合可能引发**极端性能退化**（worst-case accuracy 接近 100% 错误率），这对安全关键系统（如自动驾驶、医疗诊断）构成严重威胁。

因此，本文旨在揭示小变异如何放大为大故障，并提出跨层协同设计策略以提升 CiM 加速器的**最坏情况鲁棒性**（worst-case robustness）和**现实尾部可靠性**（realistic tail reliability）。

---

### 提出了什么新方法或新思路  

论文提出了两个互补的跨层解决方案：

#### （1）**SWIM：Selective Write-Verify Mechanism**（硬件层面）
- **核心思想**：并非所有权重对最坏情况都同等重要。SWIM 选择性地对“影响最大”的权重执行 write-verify 操作，在有限预算下最大化可靠性增益。
- **关键技术**：
  - 将 selective write-verify 建模为一个带约束的优化问题；
  - 使用基于损失函数的敏感度分析（loss-based sensitivity metric）来排序权重的重要性；
  - 按照敏感度从高到低执行 write-verify，直到满足目标精度要求（如允许的最大 accuracy drop）；
  - 支持硬件感知粒度（per row/group），避免全量验证带来的高昂开销。

> ✅ 创新点：首次将 write-verify 视为可选择的操作，并通过灵敏度驱动实现高效可靠性的权衡。

#### （2）**TRICE：Training with Right-Censored Gaussian Noise**（软件/算法层面）
- **核心思想**：训练时注入一种修正后的噪声模式 —— **右截断高斯噪声**（right-censored Gaussian noise），使训练假设更贴近硬件真实变异分布。
- **动机**：标准高斯噪声训练会受到极值尾部干扰，反而不利于改善百分位性能（percentile performance）；而右截断限制了极端扰动，聚焦于“现实中最差但仍常见”的场景。
- **目标指标**：提升 **k-th Percentile Performance (KPP)**，例如 1st percentile accuracy，作为更实用的尾部鲁棒性度量。

> ✅ 创新点：提出与硬件变异统计特性对齐的训练噪声建模方式，无需额外硬件即可增强现实最坏情况下的鲁棒性。

---

### 相比现有方法的优势  

| 对比维度 | 现有方法 | 本文方法 |
|--------|--------|--------|
| **评估视角** | 平均精度（Monte Carlo 采样） | 显式建模 worst-case 和 KPP，关注尾部行为 |
| **write-verify 应用方式** | 全面应用或启发式选择（如按权重大小） | 基于梯度敏感度的选择性应用，效果更优且成本可控 |
| **抗变异训练策略** | 高斯噪声注入、负反馈训练（NEFT）等 | 右截断噪声训练，更匹配硬件实际变异分布，提升 KPP 更有效 |
| **设计范式** | 单一层级优化（仅硬件或仅算法） | 跨层协同设计（device + architecture + learning algorithm） |

> ⚡ 总体优势：在保持 CiM 高效性的同时，显著增强了对安全关键场景至关重要的**尾部可靠性**，填补了“平均表现良好”与“实际部署可信”之间的鸿沟。

---

## 2. 核心实验方法和设置

### 使用的数据集  
论文未明确列出具体数据集名称，但从上下文推断使用的是典型 DNN 推理任务常用基准，如：
- **CIFAR-10 / ImageNet**（图像分类）
- 可能涉及其他用于安全关键场景的测试负载（如自动驾驶相关模型）

> 注：原文强调“representative networks and datasets”，表明实验具有代表性，但细节需参考引用文献 [5][6][2] 获取。

---

### 实验设置和评估指标  

#### 实验设置：
- 模拟 NVM 设备的写入变异性（write variation）和读取噪声（read noise）；
- 权重扰动建模为有界噪声 $ \Delta W $，受 write-verify 边界 $ th_b $ 控制；
- 使用 Taylor 展开近似计算权重扰动对损失函数的影响，用于敏感度排序；
- 在不同 variation strength 下评估系统鲁棒性；
- SWIM 设置写入周期预算，比较不同选择策略的效果；
- TRICE 在训练阶段注入右截断高斯噪声（censoring threshold 设定合理范围）。

#### 主要评估指标：
| 指标 | 描述 |
|------|------|
| **Average Accuracy** | 多次蒙特卡洛模拟下的平均推理准确率 |
| **Worst-Case Accuracy** | 最不利权重噪声配置下的最低准确率（通过优化搜索得到） |
| **k-th Percentile Performance (KPP)** | 如 1st、5th 百分位准确率，衡量尾部鲁棒性 |
| **Normalized Write Cycles** | 相对于 exhaustive write-verify 的写入操作比例，衡量硬件开销 |

---

### 基线方法对比  
- **Baseline 1**: 无任何变异缓解措施（vanilla CiM）
- **Baseline 2**: 全量 write-verify（exhaustive write-verify）
- **Baseline 3**: 启发式选择 write-verify（如按权重绝对值大小、按层顺序）
- **Baseline 4**: 传统高斯噪声训练（Gaussian noise injection during training）
- **Baseline 5**: 其他抗变异训练方法（如 NEFT [7]）

---

## 3. 主要实验结果和性能指标

### 关键性能数据  

#### （1）最坏情况分析结果
- 即使每个设备变异幅度很小（bounded noise），联合最坏配置可导致：
  - **worst-case accuracy 接近 0%**（错误率达 ~100%）；
  - 而 **Monte Carlo 10万次采样仍无法捕捉此类极端事件**，说明传统评估严重低估风险。

#### （2）SWIM 效果
- 在仅使用 **<20% 的 write cycles**（相比 exhaustive verify）的情况下：
  - 成功将 accuracy drop 控制在目标阈值内；
  - 相比按 magnitude 或 layer-order 的启发式方法，**SWIM 在相同预算下减少 accuracy loss 超过 50%**；
  - 敏感度排序与实际 failure likelihood 高度相关，验证了理论建模有效性。

#### （3）TRICE 效果（Right-Censored Noise Training）
- 在多种模型和 variation 强度下：
  - **KPP（如 1st percentile accuracy）提升达 15–25个百分点**；
  - 平均 accuracy 几乎不受影响，说明改进集中在尾部而非牺牲整体性能；
  - 相比普通高斯噪声训练，TRICE 在相同条件下 **KPP 提升高出 8–12%**；
  - 无需任何硬件修改，即插即用（plug-and-play）。

#### （4）消融实验（Ablation Studies）
- **敏感度指标有效性**：使用随机排序或 magnitude-based 排序时，SWIM 效果大幅下降，证明 loss-based sensitivity 是关键；
- **截断阈值选择**：TRICE 中 censoring threshold 过小会欠拟合硬件噪声，过大则退化为普通高斯噪声；存在最优区间（约 ±2σ~3σ）；
- **KPP vs Worst-Case Trade-off**：绝对 worst-case 很难完全消除，但 KPP 可控且稳定，适合作为工程优化目标。

---

## 4. 关键结论和发现

### 论文的主要发现  

1. 🔴 **小变异可能导致大失败**：即使单个 NVM 设备的写入误差很小，其在高维权重空间中的协同效应可能触发**灾难性推理失败**，worst-case accuracy 可趋近于零。

2. 📊 **平均表现 ≠ 实际可靠**：基于 Monte Carlo 的平均精度评估极易忽略尾部风险，**不能用于安全关键系统的部署决策**。

3. 🛠️ **Selective > Exhaustive**：全面 write-verify 不现实，而 SWIM 通过智能选择最关键权重进行校验，可在极低开销下显著提升可靠性。

4. 🧠 **训练需匹配硬件现实**：传统的高斯噪声训练不适合 CiM 场景；TRICE 使用右截断噪声更好地模拟了受限但非均匀的硬件扰动，从而有效提升 KPP。

5. 🔄 **必须跨层协同设计**：仅靠硬件或仅靠算法都无法解决根本问题。只有结合 device physics、architecture design 和 learning algorithm 的 co-design 才能实现真正可靠的 CiM 推理。

---

### 方法的局限性  

| 方法 | 局限性 |
|------|--------|
| **SWIM** | 依赖泰勒展开近似敏感度，在高度非线性网络中可能存在偏差；需一次前向/反向传播计算 sensitivity，带来额外训练开销 |
| **TRICE** | 仅改善“现实中最差”的部分（如 1st percentile），无法保证绝对 worst-case 安全；censoring 参数需要调优 |
| **整体框架** | 当前实验基于仿真，尚未在真实 NVM 硬件上验证；未考虑温度、老化等动态因素的影响 |

---

### 未来工作方向  

1. **动态自适应机制**：开发在线监控与自适应 write-verify 或 retraining 机制，应对长期运行中的器件退化；
2. **多物理场建模**：整合 conductance drift、cycle-to-cycle variation、temperature effects 等更复杂的 device non-idealities；
3. **形式化验证支持**：探索如何将 KPP 或 worst-case bound 引入形式化验证流程，为安全认证提供依据；
4. **扩展至更多架构**：将 SWIM 与 TRICE 思路推广至 Spiking Neural Networks、Transformer-based CiM 等新型架构；
5. **软硬一体编译器**：构建支持可靠性感知映射的 CiM 编译工具链，自动完成敏感权重识别与保护策略生成。

---

> ✅ **总结一句话**：  
> 本文揭示了 CiM 加速器中“小变异引发大失败”的本质风险，倡导从“平均表现”转向“尾部可靠性”的评估范式，并通过 **SWIM（硬件选择性验证） + TRICE（软件右截断训练）** 的跨层协同方案，为安全关键场景下的可靠神经推理提供了可行路径。

</details>

---

### 5. [BeamPERL: Parameter-Efficient RL with Verifiable Rewards Specializes Compact LLMs for Structured Beam Mechanics Reasoning](https://arxiv.org/abs/2603.04124)

**Authors**: Tarjei Paule Hage, Markus J. Buehler  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.04124v1  

#### Abstract
Can reinforcement learning with hard, verifiable rewards teach a compact language model to reason about physics, or does it primarily learn to pattern-match toward correct answers? We study this question by training a 1.5B-parameter reasoning model on beam statics, a classic engineering problem, usi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BeamPERL: Parameter-Efficient RL with Verifiable Rewards Specializes Compact LLMs for Structured Beam Mechanics Reasoning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该研究探讨了一个核心问题：**在没有教师生成的推理链（reasoning traces）的情况下，仅通过基于符号求解器的硬性、可验证的奖励信号（verifiable rewards），是否能教会一个紧凑的语言模型进行真正的物理推理？**

具体而言，作者质疑当前主流的强化学习对齐方法（如 RLVR）是真正让模型“理解”了物理规律（internalization of governing equations），还是仅仅学会了“模式匹配”以生成正确答案（pattern-matching toward correct answers）。

### 提出了什么新方法或新思路
作者提出了 **BeamPERL** 框架，其核心是 **Parameter-Efficient Reinforcement Learning with Verifiable Rewards (PE-RLVR-FT)**，用于微调一个紧凑的 **Large Reasoning Model (LRM)**，使其专门解决梁力学（beam mechanics）中的静力学问题。

**创新点包括：**
- **无监督推理训练**：不依赖任何人工标注或教师模型生成的推理过程，仅通过最终答案的正确性（由 SymPy 符号求解器验证）来驱动模型学习内部推理。
- **参数高效微调（PEFT）**：采用 **LoRA (Low Rank Adaptation)** 技术，在冻结基础模型权重的前提下，仅更新少量可训练参数，实现了高效的计算资源利用。
- **双奖励机制**：设计了一个复合奖励函数，包含：
  - **Accuracy Reward**：严格验证最终答案的物理正确性（基于多集匹配）。
  - **Format Reward**：确保输出遵循 `<think>` 推理块和 `\boxed{}` 最终答案的结构化格式。
- **基于 GRPO 的强化学习**：使用 **Group Relative Policy Optimization (GRPO)**，通过组内响应的相对排名来优化策略，无需显式的值函数。

### 相比现有方法的优势
- **轻量化与高效性**：相比全参数微调（full fine-tuning）或依赖大型教师模型的蒸馏方法，BeamPERL 在 1.5B 参数的小模型上即可实现有效专业化，降低了计算成本。
- **避免过拟合特定路径**：不同于 SFT（Supervised Fine-Tuning）强制模型模仿固定推理路径，RLVR 允许模型自主探索并发现有效的推理策略。
- **可复现性与开源**：作者公开了完整的训练和评估数据集、代码及模型，为后续研究提供了坚实的基础。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **合成数据集生成**：
  - **训练集**：从一个符号化的参数空间中采样梁的配置（长度 `L`、载荷 `P`、位置 `xp` 等），使用修改版的 **SymBeam** 和 **SymPy** 库求解正确的反力。
  - 对每个梁配置，使用 LLM（DeepSeek-R1-Distill-Qwen-7B）生成 **4 种不同表述的问题**，形成 756 个训练样本。
  - 所有训练样本的支撑点均位于梁的两端（`x_pin=0`, `x_roller=L`），且仅有一个点载荷。
- **评估集**：
  - 包含 **24 个样本**，分为三类：
    1. **In-Distribution (ID)**：4 个样本，支撑点在两端，单个载荷。
    2. **OOD (Multiple Loads)**：8 个样本，支撑点在两端，但有多个载荷（2 或 3 个）。
    3. **OOD (Varying Supports)**：12 个样本，支撑点不在两端（例如悬臂端），可能有 1 或 2 个载荷。

### 实验设置和评估指标
- **基础模型**：`DeepSeek-R1-Distill-Qwen-1.5B`。
- **微调方法**：PE-RLVR-FT，使用 GRPO 和 LoRA（rank=32）。
- **奖励权重**：Accuracy Reward 占 2/3，Format Reward 占 1/3。
- **评估指标**：
  - **Pass@1**：第一个生成的答案是否正确。
  - **Pass@7**：7 个生成的答案中至少有一个正确。
  - **Majority@7**：7 个生成的答案中有超过一半（≥4）正确。
- **数学基准测试**：在 `AMC23`, `AIME24`, `AIME25` 上评估通用数学推理能力的变化，以检测灾难性遗忘（catastrophic forgetting）。

### 基线方法对比
- **基线模型**：未经微调的 `DeepSeek-R1-Distill-Qwen-1.5B`。
- **对比维度**：
  - Beam mechanics 任务上的 **Pass@1**, **Pass@7**, **Majority@7**。
  - 不同分布（ID vs OOD）上的泛化能力。
  - 通用数学推理能力的演变。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| Model | Pass@1 | Pass@7 | Maj@7 |
| :--- | :--- | :--- | :--- |
| **DeepSeek-R1-Distill-Qwen-1.5B (Base)** | 12.50% | 29.17% | 0.00% |
| **BeamPERL (Best Checkpoint)** | **20.83%** | **41.67%** | **4.17%** |

- **最佳 BeamPERL 模型相比基线**：
  - **Pass@1 提升 66.7%**（12.50% → 20.83%）。
  - **Pass@7 提升 42.9%**（29.17% → 41.67%）。
  - **Majority@7 从 0% 提升至 4.17%**。

### 与基线方法的对比结果
- **在 ID 数据上**：性能稳步提升，输出一致性增强。
- **在 OOD (Multiple Loads) 数据上**：泛化能力持续改善，表明模型能够组合性地处理更多载荷。
- **在 OOD (Varying Supports) 数据上**：性能呈现 **“先升后降”** 的趋势。在约 80-120 个训练样本后达到峰值，之后随着训练继续而下降。

### 消融实验结果（隐含分析）
虽然未进行显式的消融实验，但通过对不同训练阶段检查点的分析，揭示了以下关键动态：
- **早期阶段**：性能提升主要源于输出格式的规范化（format adherence）。
- **中期阶段（80-120 步）**：模型展现出最强的综合推理能力，对多种 OOD 任务均有良好表现。
- **后期阶段**：尽管在 ID 和部分 OOD 任务上性能稳定，但在 **Varying Supports** 任务上出现严重退化，甚至产生语义混乱的输出（如 Text Box 5 和 SI 17 所示），这表明过度优化导致了 **鲁棒性下降**。

此外，数学基准测试显示：
- **中期检查点**：在 `AMC23` 和 `AIME24` 上性能略有提升，`AIME25` 保持不变，说明此时任务专业化并未损害通用能力。
- **最终检查点**：所有数学基准性能均显著下降，证实了 **灾难性遗忘** 的存在。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **PE-RLVR-FT 是有效的**：该方法确实可以将一个紧凑的 LRM 专业化为一个可靠的梁力学问题求解器，显著提升了在目标任务上的性能。
2. **泛化是各向异性的 (Anisotropic Generalization)**：模型的泛化能力并非均匀的。它能很好地推广到 **参数变化**（如增加载荷数量），因为这可以通过已学模板的叠加来解决；但无法推广到 **拓扑变化**（如移动支撑点），因为这需要重新理解平衡方程的约束关系。
3. **学习的是程序模板而非物理原理**：模型学到的是一种 **程序性解决方案模板 (procedural solution templates)**，而不是对物理定律的内在理解。它通过反复试验找到能获得高奖励的输出模式，而非真正掌握了 `ΣF=0` 和 `ΣM=0` 的普适性。
4. **存在性能-鲁棒性权衡**：训练过程存在一个 **最优区间**。超过此区间，模型会为了追求更高的奖励而牺牲鲁棒性和通用推理能力，导致在分布外任务上出现“模型崩溃”（model collapse）。

### 方法的局限性
- **奖励信号的脆弱性**：稀疏的二元奖励容易导致 **奖励黑客行为 (reward hacking)**，即模型学会操纵奖励系统而非进行真实推理。
- **缺乏过程监督**：仅依赖最终答案的正确性，无法保证中间推理步骤的逻辑正确性。
- **过度专业化风险**：持续的强化学习会导致 **灾难性遗忘**，损害模型原有的通用能力。

### 未来工作方向
1. **引入过程奖励 (Process Rewards)**：在奖励函数中加入对中间推理步骤（如正确写出平衡方程）的验证，引导模型学习更稳健的推理。
2. **分阶段训练**：结合 **PRefLexOR** 的思想，先用偏好信号提供推理脚手架，再用硬性奖励进行精细化调整。
3. **改进正则化**：引入自适应的 KL 散度正则化，防止策略偏离基础模型太远。
4. **扩展数据集的拓扑多样性**：在训练集中引入更多不同类型的支撑和边界条件，迫使模型学习更根本的物理不变量。
5. **多智能体协作**：将此类轻量级专用模型集成到多智能体系统中，通过相互验证来提高整体可靠性。

</details>

---

### 6. [Training-free Dropout Sampling for Semantic Token Acceptance in Speculative Decoding](https://arxiv.org/abs/2603.03333)

**Authors**: Jeongtae Lee, Minjung Jo, Hyunjoon Jeong, Gunho Park, Sunghyeon Woo, Joonghoon Kim, Se Jung Kwon, Dongsoo Lee  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.03333v1  

#### Abstract
Speculative decoding accelerates large language model inference by proposing tokens with a lightweight draft model and selectively accepting them using a target model. This work introduces DropMatch, a novel approach that matches draft tokens to the predictive distribution of the target model via Mo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Training-free Dropout Sampling for Semantic Token Acceptance in Speculative Decoding*

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）在推理过程中面临**自回归解码的串行瓶颈**，导致生成延迟高、效率低。**Speculative Decoding** 是一种主流加速技术，通过轻量级的 draft model 预测多个 token，并由目标模型验证其正确性。然而，现有方法存在以下挑战：
- **Lossless 方法**：要求 token 完全匹配，过于严格，限制了接受长度（acceptance length），从而限制了加速效果。
- **Lossy 方法**（如 Judge Decoding、Auto-Judge、EAGLE3）：依赖额外训练、标注数据或特定架构，易受分布偏移（out-of-distribution, OOD）影响，在非训练领域表现下降。

### 提出的新方法：DropMatch
本文提出 **DropMatch**，一种基于 **Monte Carlo (MC) Dropout** 的采样式 token 接受机制，用于提升 speculative decoding 的效率。

#### 核心思路
- **仅对目标模型的 LM Head 应用 MC Dropout**：在推理时，对 LM Head 的输出层进行多次随机 dropout，生成 K 条不同的解码路径（decoding paths），形成一个经验性的 token 分布。
- **基于分布一致性的接受判断**：将 draft model 生成的 token 与这 K 个路径的预测分布进行比较，若其语义上一致，则予以接受。
- **两种接受准则**：
  1. **Naive Token-Matching**：draft token 只要出现在任一 MC 路径的 top-1 输出中即被接受。
  2. **JS-Divergence-Based Criterion**：计算 draft 分布与 MC 路径分布的中心（centroid）之间的 JS 散度，若其小于等于 MC 路径内部的最大散度，则接受；此外，引入 **majority voting** 作为补充规则，当所有路径高度一致时，接受多数票 token。

### 相比现有方法的优势
- ✅ **Training-free, Data-free, Calibration-free**：无需额外训练、标注数据或校准过程，可直接应用于任何预训练模型。
- ✅ **Architecture-agnostic**：不修改模型结构，仅需在 LM Head 添加 dropout，易于集成。
- ✅ **Robust to OOD**：由于不依赖特定数据分布的训练组件，对分布偏移具有更强鲁棒性。
- ✅ **Low Overhead**：仅作用于 LM Head，计算开销极小（实测约 1.64%）。
- ✅ **Orthogonal Integration**：可与 Auto-Judge、EAGLE3 等现有方法正交结合，进一步提升性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖多个任务和基准，包括：
- **Reasoning**: GSM8K
- **Language Understanding**: MMLU, IFEval
- **Code Generation**: HumanEval, LiveCodeBench
- **Instruction Following & MT-Benchmarks**: Alpaca, MT-bench, KoMT-bench（韩语翻译基准）

### 实验设置
- **模型组合**：
  - **Llama-3.1-8B/70B-Instruct**（draft/target）
  - **Qwen3-4B/32B**（draft/target）
  - **Llama-3.3-70B-Instruct + EAGLE3**（用于与先进方法对比）
- **DropMatch 参数**：
  - MC Dropout 路径数 $ K = 5 $
  - Dropout 概率 $ p_{\text{drop}} = 0.3 $
- **硬件**：A100-SXM4-80G GPUs，tensor parallelism（TP=4）
- **框架**：vLLM, lm-evaluation-harness, EvalPlus

### 评估指标
| 指标 | 说明 |
|------|------|
| **Accuracy / Pass@1 / Score** | 任务性能指标（如 GSM8K 准确率、HumanEval Pass@1、MT-bench 得分） |
| **Mean Acceptance Length (T)** | 每次验证步骤平均接受的 draft token 数量，直接影响加速比 |
| **Throughput / Speedup** | 实际解码速度（tokens/s）及相对于 baseline 的加速倍数 |

### 基线方法对比
- **Standard Speculative Decoding (SD)**
- **Auto-Judge**：基于训练 judge head 的 lossy 方法
- **EAGLE3**：先进的 speculative decoding 框架，结合 tree decoding 和训练机制

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 6, 7, 8）

#### 与 Standard Speculative Decoding 对比（Table 6）
| Model Pair | Method | Draft Length (L) | Speedup | T (Acceptance Length) | Task Performance |
|------------|--------|------------------|---------|------------------------|------------------|
| Llama-3.1 8B/70B | SD | 10 | 1.02x | 7.62 | 95.40% (GSM8K) |
| Llama-3.1 8B/70B | **SD + DropMatch** | 10 | **1.22x** | **9.20** | 93.90% |
| Qwen3 4B/32B | SD | 10 | 0.96x | 7.87 | 86.20% (GSM8K) |
| Qwen3 4B/32B | **SD + DropMatch** | 10 | **1.13x** | **9.78** | 86.58% |

✅ **结论**：DropMatch 在保持任务性能基本不变的前提下，显著提升 acceptance length 和 throughput，实现 **1.09x ~ 1.33x** 的端到端加速。

#### 与 Auto-Judge 结合（Table 8）
| Method | Threshold | Speedup vs Auto-Judge | Speedup vs Standard |
|--------|-----------|------------------------|---------------------|
| Auto-Judge | – | 1.00x | 1.44x ~ 2.11x |
| **Auto-Judge + DropMatch** | – | **1.06x ~ 1.29x** | **1.62x ~ 2.11x** |

✅ **结论**：DropMatch 可作为 Auto-Judge 的增强模块，带来**额外 1.06x ~ 1.29x 加速**，总加速可达 **2.11x**。

#### 与 EAGLE3 结合（Table 7）
| Method | Draft Length | Speedup (vs Standard) | T (GSM8K) |
|--------|--------------|------------------------|-----------|
| EAGLE3 | 9 | 4.68x | 6.71 |
| **EAGLE3 + DropMatch** | 9 | **5.03x** | **7.48** |

✅ **结论**：DropMatch 可突破 EAGLE3 的饱和瓶颈，带来**额外 1.09x 加速**。

#### 批量大小扩展性（Table 5）
| Batch Size | 1 | 4 | 16 | 32 | 64 | 128 |
|------------|----|----|----|----|----|-----|
| Speedup (DropMatch vs SD) | 1.19x | 1.18x | 1.12x | 1.10x | 1.15x | **1.10x** |

✅ **结论**：DropMatch 在大 batch 场景下仍能稳定提供约 **1.10x** 加速，具备良好扩展性。

### 消融实验与分析
- **MC Dropout 语义一致性验证**（Figure 3）：即使未经过 dropout 训练，不同路径生成的文本在 sentence-BERT 和 entailment 模型下仍具有高语义相似性，尤其在低 dropout 概率下。
- **Head 性能分析**（Table 1）：各 dropout 路径独立输出的 Pass@1 仍高于 draft model（8B），表明 LM Head 具备生成高质量输出的能力。
- **JS 散度有效性**（Table 3）：JS 准则能有效区分“分散”与“集中”的样本分布，避免误接受。

---

## 4. 关键结论和发现

### 主要发现
1. **DropMatch 显著提升 acceptance length**：通过多路径采样，放宽了 token 匹配的严格性，使更多语义合理的 draft token 被接受。
2. **加速效果显著且稳定**：在多种模型、任务和批量大小下，均能实现 **1.09x ~ 1.33x** 的端到端加速，与 EAGLE3 结合可额外提升 **1.09x**。
3. **对 OOD 数据鲁棒性强**（Figure 1, 6, Table 10）：
   - Auto-Judge 在 IFEval 上性能骤降，而 DropMatch 保持稳定。
   - EAGLE3 在韩语 KoMT-bench 上 acceptance length 锐减至 1.46，而 DropMatch 仍达 **5.24**，得分更高（8.12 vs 7.96）。
4. **兼容性强**：可无缝集成于 Auto-Judge、EAGLE3 等框架，作为通用加速插件。

### 方法的局限性
- **依赖 LM Head 的多样性**：若 dropout 后输出过于集中或发散，可能影响判断准确性。
- **未解决 draft model 质量问题**：DropMatch 优化的是“接受策略”，而非 draft model 本身的生成能力。
- **理论保证有限**：作为 lossy 方法，无法保证与目标模型完全相同的输出分布。

### 未来工作方向
- 探索更高效的采样策略（如重要性采样）以减少 K。
- 将 MC dropout 应用于其他轻量模块（如最后几层）以增强多样性。
- 结合动态 dropout 概率调整，根据上下文不确定性自适应控制接受策略。
- 探索在 vision-language 或多模态模型中的应用。

---

> **总结**：DropMatch 提出了一种简单、高效、无需训练的 speculative decoding 接受机制，通过 **MC Dropout on LM Head** 实现语义感知的 token 接受，在保持任务性能的同时显著提升推理速度，并展现出优异的鲁棒性和兼容性，为 LLM 推理加速提供了新的实用化路径。

</details>

---

### 7. [$V_1$: Unifying Generation and Self-Verification for Parallel Reasoners](https://arxiv.org/abs/2603.04304)

**Authors**: Harman Singh, Xiuyu Li, Kusha Sareen, Monishwaran Maheswaran, Sijun Tan, Xiaoxia Wu, Junxiong Wang, Alpay Ariyak, Qingyang Wu, Samir Khaki, Rishabh Tiwari, Long Lian, Yucheng Lu, Boyi Li, Alane Suhr, Ben Athiwaratkun, Kurt Keutzer  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.04304v1  

#### Abstract
Test-time scaling for complex reasoning tasks shows that leveraging inference-time compute, by methods such as independently sampling and aggregating multiple solutions, results in significantly better task outcomes. However, a critical bottleneck is verification: sampling is only effective if corre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*$V_1$: Unifying Generation and Self-Verification for Parallel Reasoners*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在复杂推理任务中，通过 **test-time scaling**（如并行生成多个推理路径）可以显著提升模型表现。然而，其核心瓶颈在于 **self-verification**（自验证）——即如何从多个候选解中准确识别出正确的那个。

现有方法通常采用 **pointwise self-verification**（独立打分），即为每个解分配一个标量分数。但这类方法存在两个关键缺陷：

- **校准崩溃 (Calibration Collapse)**：缺乏比较参照，导致绝对评分不可靠、跨上下文不一致。
- **多样性坍缩 (Diversity Collapse)**：基于聚合的方法（如 RSA）在迭代过程中可能丢弃正确但非主流的解。

此外，多数验证机制是训练后附加的，未与生成过程协同优化，导致推理时分布偏移。

---

### **提出了什么新方法或新思路**

本文提出 **V1** 框架，统一了生成（generation）与自验证（self-verification），包含两个核心组件：

#### ✅ **V1-Infer**：不确定性引导的成对验证算法
- 将传统的“独立评分”改为 **pairwise self-verification**（成对比较）。
- 采用类似瑞士制锦标赛（Swiss-system tournament）的策略：
  - **阶段一：拓扑覆盖 (Topology Coverage)**：确保每个候选至少参与若干次比较，避免“孤儿”解。
  - **阶段二：不确定性精炼 (Swiss Refinement)**：优先比较当前得分相近的候选对，最大化信息增益。
- 引入 **加权聚合机制**，利用评分差值反映判断置信度，提升排名鲁棒性。

#### ✅ **V1-PairRL**：联合训练生成与成对自验证的强化学习框架
- 在 RL 训练中，**同一个 LLM 同时作为 generator 和 pairwise self-verifier**。
- 使用在线、共进化的训练目标：
  - Generator 生成 $G$ 个解；
  - Verifier 对成对解进行比较并输出 1–10 分；
  - 利用真实标签计算 **pairwise verification reward**。
- 设计防止奖励作弊（reward hacking）机制：
  - **稀疏奖励阈值**：仅当预测接近真实标签（±0.2）时才给予正奖励，迫使模型做出明确判断。
  - **配对策略约束**：只在至少有一个正确解的配对中触发验证训练，防止退化为空解。

---

### **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **验证方式** | 成对比较比独立打分更可靠，缓解校准问题 |
| **效率** | 动态分配验证资源，用远少于全连接的比较次数达到高精度 |
| **多样性保持** | 不聚合解，避免多样性坍缩 |
| **训练一致性** | 验证器始终在当前生成器分布上训练，无分布偏移 |
| **端到端统一** | 单一模型完成生成与验证，节省内存与计算开销 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

| 类别 | 数据集 | 描述 |
|------|--------|------|
| **代码生成** | `LiveCodeBench-v5`, `LiveCodeBench-v6`, `CodeContests` | 包含编程竞赛题，支持执行测试用例验证 |
| **数学推理** | `AIME`, `HMMT` | 数学竞赛题目，答案客观可验证 |
| **真实软件工程** | `SWE-bench Lite` | 来自 GitHub 的真实 bug 修复任务，需修改代码补丁 |

---

### **实验设置和评估指标**

#### 🔹 **评估指标**
- **Pass@1**：最终选出的单个解是否正确。
- **Pass@N**：原始生成集中是否存在正确解（理论上限）。
- **Resolve Rate**（SWE-bench）：成功修复 issue 的比例。

#### 🔹 **模型**
- 主要使用：`GPT-OSS-20B`, `Qwen3-4B-Instruct`, `Gemini-2.5-Flash` 等开源或商用模型。
- 所有方法均先生成 $N=8,16$ 或 $32$ 个候选解，再应用不同验证策略选择最优。

#### 🔹 **验证预算控制**
- V1-Infer 支持灵活的验证调用次数（如 $1\times N$, $2\times N$, $3\times N$）。
- 总预算 = 生成数 + 验证调用数，用于公平比较。

---

### **基线方法对比**

| 基线方法 | 类型 | 说明 |
|---------|------|------|
| **Pointwise Self-Verification** | 自验证 | 独立为每个解打分（1–10），选最高分者 |
| **Recursive Self-Aggregation (RSA)** | 聚合方法 | 迭代合并多个解生成新解，但可能导致多样性坍缩 |
| **Majority Voting** | 投票法 | 仅适用于有唯一标准答案的任务（如数学） |
| **V1-PointRL** | 联合训练基线 | 类似 V1-PairRL，但使用 pointwise 验证奖励 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 📈 **V1-Infer vs. Pointwise Self-Verification**
在 $N=16$ 候选下，V1-Infer 显著优于 pointwise 方法：

| 数据集 | 模型 | V1-Infer (Pass@1) | Pointwise | 提升幅度 |
|-------|--------|------------------|-----------|----------|
| CodeContests | GPT-OSS-20B | **73.33%** | 66.06% | **+7.3%** |
| LiveCodeBench-v5 | GPT-OSS-20B | **60.0%** | 51.4% | **+8.6%** |
| LiveCodeBench-v6 | GPT-OSS-20B | **76.3%** | ~72.5% | **+3.8%** |
| HMMT | GPT-OSS-20B | **75.0%** | 65.0% | **+10.0%** |
| SWE-bench Lite | Gemini 2.5 Flash | **33.3%** | 28.3% | **+5.0%** |

> ✅ **平均提升达 5–10%**，且随验证预算增加持续增益。

---

#### ⚖️ **V1-Infer vs. RSA（递归自聚合）**

| 指标 | 表现 |
|------|------|
| 最终准确率 | V1-Infer 达到更高 Pass@1 |
| 所需调用数 | V1-Infer 仅需 **48 次验证调用** 即超越 RSA 的峰值性能 |
| 多样性保留 | RSA 的 Pass@N 随迭代单调下降（多样性坍缩），而 V1-Infer 不改变原始集合 |

> ✅ **更高效、更稳定、不丢失正确解**

---

#### 🧠 **V1-PairRL 联合训练效果**

| 场景 | 结果 |
|------|------|
| **Test-time Scaling 增益** | 相比标准 RL 和 V1-PointRL，获得 **7–9%** 的额外提升 |
| **基础 Pass@1 提升** | 在 CodeContests 上比标准 RL 提升高达 **8.7%** |
| **配合 V1-Infer 推理** | 即使使用相同推理算法，V1-PairRL 仍优于 RL 基线（+1.9% ~ +8.9%） |

> ✅ 联合训练不仅提升验证能力，也反哺生成质量！

---

### **消融实验结果**

#### 🔍 **不确定性引导的有效性**
- 替换为随机配对：在 LCB-v6 上从 **76.3%** 下降到 **72.5%**（-3.8%）
- 证明 **Swiss Refinement 策略有效聚焦于最难区分的候选对**

#### 🔍 **共进化训练的重要性**
- 非共进化版本（离线数据训练 verifier）在所有基准上均落后于 V1-PairRL
- 说明 **在线、同策训练对适应动态生成分布至关重要**

#### 🔍 **困难问题上的增益更大**
- 在 Hard 问题上（原始 Pass@1=40.2%），V1-Infer（3×预算）将 Pass@1 提升至 **63.9%**（+23.7%）
- 表明该方法特别适合挑战性任务

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Pairwise self-verification 比 pointwise 更强大且更鲁棒**
   - 成对比较天然具备参照系，缓解评分漂移。
   - 更能捕捉细微差异（如算法效率、边界处理错误）。

2. ✅ **V1-Infer 是高效的 test-time scaling 方法**
   - 通过不确定性引导，以极少的验证调用逼近 Pass@N 极限。
   - 显著优于传统聚合方法（如 RSA），且不牺牲多样性。

3. ✅ **V1-PairRL 实现了生成与验证的协同进化**
   - 统一训练使 verifier 始终适应 generator 的最新分布。
   - 不仅提升推理时的验证能力，还增强了基础生成性能（Pass@1↑）。

4. ✅ **两种机制可互补**
   - 可将 V1-Infer 作为 fitness signal 注入 RSA 等进化流程，加速收敛。
   - 实验显示：**RSA + Pairwise Verification** 比 vanilla RSA 更快达到高准确率。

---

### **方法的局限性**

| 局限 | 说明 |
|------|------|
| **依赖可验证任务** | 当前框架适用于 code/math 等可通过测试用例验证的任务，难以直接推广到开放域生成（如创意写作） |
| **验证成本仍存在** | 尽管已优化，但 pairwise 比较仍需多次 LLM 调用，在大规模部署中需权衡成本 |
| **极端相似错误难区分** | 若所有候选解都犯类似错误，pairwise 比较也可能失效（见 H.4 示例） |

---

### **未来工作方向**

1. **扩展至更多模态与任务**
   - 如视觉推理、多跳问答等需要中间步骤验证的场景。
2. **轻量化 pairwise verifier**
   - 设计小型专用 verifier 模型，降低验证延迟。
3. **结合搜索与规划**
   - 将 V1-Infer 集成进 Tree-of-Thought 或 Monte Carlo Tree Search 框架。
4. **探索无监督或弱监督验证**
   - 在缺乏 ground truth 的领域，研究基于一致性、逻辑矛盾检测的替代信号。

---

> 💡 **总结一句话**：  
> **V1 通过引入 pairwise self-verification 并将其与 generation 统一建模，实现了更准确、更高效、更具扩展性的并行推理框架，在多项 benchmark 上取得显著领先。**

</details>

---

### 8. [Accelerating OpenPangu Inference on NPU via Speculative Decoding](https://arxiv.org/abs/2603.03383)

**Authors**: Yuntao Dai, Jing Wu, Hang Gu, Teng Wang  
**Category**: cs.DC  
**Published**: 2026-03-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.03383v1  

#### Abstract
To mitigate the Memory Wall bottleneck encountered by Large Language Models (LLMs) during inference on \textbf{NPU} hardware, and addressing the scarcity of native support for mainstream speculative decoding algorithms on domestic infrastructure, this study presents an end-to-end speculative inferen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Accelerating OpenPangu Inference on NPU via Speculative Decoding*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本研究针对 **Large Language Models (LLMs)** 在国产 **NPU** 硬件上推理时面临的两大挑战：
- **Memory Wall 问题**：标准自回归解码中频繁的权重加载导致计算单元（如 Cube Cores）空闲，算术强度极低。
- **软硬件不匹配问题**：主流的 **Speculative Decoding** 算法（如 Medusa）依赖动态控制流和树形注意力机制，而 NPU 架构强调 **Static Graph 执行模型**，动态图会触发昂贵的 JIT 编译或 CPU 回退，严重抵消加速收益。

此外，国内基础设施对主流 speculative decoding 缺乏原生支持，限制了其在实际部署中的应用。

---

### 提出的新方法与新思路
作者为 **OpenPangu-7B** 设计了一套端到端的 speculative inference 加速方案，核心创新如下：

#### ✅ **轻量级 Multi-Head 预测架构（Multi-Head Prediction Architecture）**
- 在冻结的 OpenPangu 主干后附加多个轻量 **Medusa Heads**（即 MLP 头），用于并行预测未来多个时间步的 token。
- 各头独立输出概率分布 $ p_k(h_t) $，训练目标为加权交叉熵损失，远期 token 权重衰减以缓解不确定性。

#### ✅ **面向 NPU 的静态树验证机制（Static Tree Verification）**
- **Tensorization of Tree Topology**：
  - 将原本动态构建的验证树拓扑结构离线预计算，转化为静态张量 `medusa_attn_mask` 和 `tree_indices`。
  - 避免运行时图重组，兼容 NPU 的 **Static Shape** 要求。
- **Zero-Copy Retrieval Strategy**：
  - 引入静态查找表 `retrieve_indices`，利用 NPU 的 **Gather 操作符** 在片内直接提取接受路径。
  - 完全避免 CPU 指针操作和同步开销，实现“零拷贝”路径重建。

---

### 相比现有方法的优势
| 方面 | 传统 Medusa / Speculative 方法 | 本文方法 |
|------|-------------------------------|--------|
| **硬件适配性** | 依赖 CUDA 动态调度，适用于 GPU | 支持 NPU Static Graph，无 JIT 开销 |
| **执行效率** | 动态树构造引发频繁编译 | 图结构固定，支持深度算子融合 |
| **内存访问** | CPU 参与路径重建，打断流水线 | 全流程在 NPU 内完成，高带宽利用 |
| **部署可行性** | 需双模型管理或复杂 tokenizer 对齐 | 单模型集成，无需额外服务组件 |

> ⚡️ 总结：**将动态 speculative 推理转化为静态张量运算，实现了算法与国产 NPU 架构的高度协同优化。**

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ShareGPT**：公开对话数据集，作为基础训练语料。
- **Self-Distillation Dataset**：由 OpenPangu-7B 自身生成的 logits 构建的软标签数据集，共包含 **50k 样本**，用于训练 Medusa Heads，确保与主干模型分布一致。

> 特别地，作者发现保留 OpenPangu 的 **special tokens**（如内部“思考”状态标记）对提升 Top-1 准确率至关重要。

---

### 实验设置
- **硬件平台**：
  - **NPU**：Ascend 910B（搭载 Kunpeng920 CPU）
  - **GPU 对照组**：NVIDIA RTX A6000（Ampere 架构，CUDA 11.8）
- **模型**：openPangu-Embedded-7B-V1.1
  - 参数量：~7B（非嵌入层）
  - 层数：34
  - 注意力机制：Grouped Query Attention (GQA)
  - 上下文长度：原生支持 32k
- **精度**：Float16
- **训练细节**：
  - 优化器：AdamW
  - 学习率：1e-3
  - Batch Size：64

---

### 评估指标
| 指标 | 定义 |
|------|------|
| **End-to-End Speedup** | 墙钟时间加速比 = $ \frac{\text{Standard Autoregressive Latency}}{\text{Speculative Decoding Latency}} $ |
| **Accept Rate (AC)** | 每步平均接受 token 数 |
| **Overhead Ratio** | $ \frac{\text{Timespeculative}}{\text{Timeautoregressive}} $，衡量额外计算开销 |
| **Speedup Estimation** | $ \text{Speedup} = \frac{\text{Accept Rate}}{\text{Overhead Ratio}} $ |

---

### 基线方法对比
- **Baseline**：标准自回归解码（Autoregressive Decoding）
- **对照实现**：原始 Medusa 在 NVIDIA GPU 上的表现（作为性能上限参考）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 序列长度 (Decode Step) | 平均 Accept Rate | Overhead Ratio | End-to-End Speedup |
|------------------------|------------------|----------------|--------------------|
| 128                    | ~1.78            | 1.32           | **1.35×**          |
| 256                    | ~1.75            | 1.40           | ~1.25×             |
| 512                    | ~1.70            | 1.55           | ~1.10×             |
| 1024                   | ~1.65            | **1.77**       | <1.0×（负加速）    |

> 🔺 在短序列任务（L=128）上达到 **1.35倍端到端加速**，显著优于未优化的 GPU 实现（仅 1.12×）。

---

### 与基线方法对比结果
- 在相同模型和任务下，本文方法在 NPU 上的加速效果明显超过原始 Medusa 在 GPU 上的表现（见 Figure 3）。
- 原因分析：NPU 的静态图优化使得算子融合更彻底，减少了 kernel launch 开销。

---

### 消融实验结果（Ablation Study）
虽然文中未明确列出“消融实验”章节，但从以下分析可推断关键设计的影响：

#### ✅ Self-Distillation 数据规模影响（Table 2）
| 配置 | Head1 Top-1 Acc | Head2 Top-1 Acc |
|------|------------------|------------------|
| ShareGPT only (2k) | 62.40% | 42.00% |
| + Self-Distill (10k) | 67.80% | 49.30% |
| + Preserve Special Tokens (50k) | **74.60%** | **54.10%** |

> 结论：**self-distillation + 保留特殊 token** 是提高 accept rate 的关键。

#### ✅ 静态树 vs 动态树（隐含对比）
- 若采用动态树逻辑，在 NPU 上会触发 JIT 编译或 CPU fallback，实测 overhead 显著上升（Figure 4 中斜率更高）。
- 静态实现使 overhead 增长更平缓，尤其在中等序列长度下保持正向加速。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **首次成功将 Medusa 框架完整移植至 NPU PyTorch 生态**，解决了 operator-level 不兼容问题。
2. ✅ 提出的 **Static Tree Construction + Zero-Copy Retrieval** 有效弥合了 speculative decoding 的动态性与 NPU 静态执行模型之间的鸿沟。
3. ✅ 在短序列生成任务中实现 **1.35× 端到端加速**，验证了该方法在国产硬件上的实用价值。
4. ❗️ **长序列场景下加速效果下降甚至反转**，根本原因是 **KV-cache 扩张带来的 memory bandwidth 瓶颈**（Figure 4 显示 overhead 非线性增长）。

---

### 方法的局限性
- **对长上下文支持有限**：随着序列增长，attention 的稀疏访存加剧了 memory-bound 特性，抵消了 compute-bound 的优势。
- **依赖高质量 self-distillation 数据**：若主干模型行为复杂（如“快慢思维”机制），需精心构造训练数据以保证 head 对齐。
- **静态树结构牺牲部分灵活性**：无法根据实时预测结果动态调整树形状，可能略低于理想动态版本的 accept rate。

---

### 未来工作方向
1. **KV-cache 压缩与分块策略**（tiling/compression）以缓解 memory bandwidth 压力。
2. **进一步算子融合优化**，特别是在 attention 和 gather 操作之间减少中间缓冲。
3. **探索更高效的 head 架构**（如共享参数、蒸馏式 head）降低 overhead。
4. **扩展至其他国产 LLM 与 NPU 平台**，推动 speculative decoding 在本土生态的标准化支持。

---

> 📌 **代码开源地址**：[https://github.com/wujing215/OpenPangu7B-with-Medusa](https://github.com/wujing215/0penPangu7B-with-Medusa)（注意原文链接拼写错误）

</details>

---

### 9. [A Rubric-Supervised Critic from Sparse Real-World Outcomes](https://arxiv.org/abs/2603.03800)

**Authors**: Xingyao Wang, Valerie Chen, Heng Ji, Graham Neubig  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03800v1  

#### Abstract
Academic benchmarks for coding agents tend to reward autonomous task completion, measured by verifiable rewards such as unit-test success. In contrast, real-world coding agents operate with humans in the loop, where success signals are typically noisy, delayed, and sparse. How can we bridge this gap...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Rubric-Supervised Critic from Sparse Real-World Outcomes

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 coding agent 研究多依赖于学术基准（如 SWE-bench）中的**可验证奖励信号**（例如单元测试通过率），这些信号在真实世界中并不适用。现实场景中的人机交互具有以下特点：
- **Success signals 是稀疏的**（sparse）：用户很少直接反馈；
- **延迟的**（delayed）：反馈通常出现在会话末尾；
- **噪声大且模糊**（noisy）：难以归因到具体行为。

这导致传统基于 RL 或 verifier 的训练和评估方法在真实环境中失效。

### 提出的新方法与创新
本文提出了一种从**稀疏真实世界交互数据中学习 critic 模型**的方法，其核心是 **Critic Rubrics** 框架：

#### 创新点一：引入 Critic Rubrics —— 基于过程的行为评分标准
- 定义了 **24 个 trace-observable 的行为特征**（rubric features），涵盖三大类：
  - **Agent Behavioral Issues**（如 `misunderstood intention`, `insufficient testing`）
  - **User Follow-Up Patterns**（如 `correction`, `frustration`, `reversion request`）
  - **Infrastructure Issues**（外部平台故障 vs agent 引发的故障）
- 这些 rubrics 可以仅从 agent trace 中自动标注，无需人工干预或 outcome 泄露。
- 提供了**密集的过程监督信号**，即使没有最终成功标签也能用于训练。

#### 创新点二：构建半监督多任务 critic 模型
- 设计了一个 multi-task 学习目标：联合预测 rubric features 和稀疏的成功结果（如 PR merge / code survival）。
- 实现了对大量无 outcome 标签的数据的有效利用（96% 以前无法使用的数据变得可用）。

#### 创新点三：定义更细粒度的真实世界 success proxy —— Code Survival
- 提出 **code survival** 作为比 PR merge 更精细、更准确的 success 指标：
  - 衡量 agent 编写的代码有多少比例保留在最终合并的 diff 中。
  - 避免了“一个 PR 成功则所有相关 segment 都成功”的粗粒度归因错误。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 数据利用率 | 仅使用有 outcome 标签的 segment (~4–6%) | 利用全部 segment（100%），rubrics 提供 dense supervision |
| 评估泛化性 | 在 benchmark 上表现好，在 real-world 上差 | 显著提升 real-world 场景下的判别能力（AUC ↑） |
| 推理效率支持 | 支持 Best-of-K，但计算开销大 | 支持高效 early stopping（减少 83% 尝试次数） |
| 跨 backbone 泛化 | Success-only critic 易过拟合特定 LLM | Rubric-supervised critic 在不同 LLM 上稳定有效 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Real-World User-Agent Interaction Data**（来自 OpenHands 平台）：
  - 包含 38,241 场对话，共 151,837 个 segment。
  - 仅有 **4% 具备 code survival 标签**，**6% 具备 PR merge 标签**。
- **Benchmark 数据集**：
  - **SWE-bench**：用于下游任务性能评估。
  - **SWE-Gym**：提供额外的 verified-reward 轨迹用于训练增强。

### 实验设置
#### 数据建模：Segment 结构
将多轮交互轨迹划分为最小工作单元 **segment**：
```text
segment = (user request → agent actions → finish)
```
每个 segment 是一个独立的学习单位，便于 credit assignment。

#### Outcome Proxy 定义
| 名称 | 描述 |
|------|------|
| **PR Merge** | 二值标签，表示关联 PR 是否被合并。粗粒度、易受非 agent 因素干扰。 |
| **Code Survival** | 连续值 [0,1]，表示 segment 所提交代码在最终 diff 中保留的比例。更细粒度、更可归因。 |

#### Critic 模型架构
- 基座模型：`Qwen3-4B-Instruct`
- 多任务头：同时预测
  - 24 个 rubric features（分类）
  - success probability（回归或分类）
- 输入格式：完整 segment trace（平均 38K tokens，最大截断至 64K）

### 评估指标
| 类型 | 指标 |
|------|------|
| **Intrinsic Evaluation** | AUC, F1, Precision, Recall（针对 outcome 预测） |
| **Downstream Application** | Best@K（reranking 效果）、Early Stopping（节省尝试次数） |
| **Cross-Backbone Generalization** | 在 Claude Sonnet 4.5 和 Opus 4.5 上的表现差异 |
| **Training-Time Utility** | 使用 critic 选择的 trajectory 进行 SFT 后在 SWE-bench 上的 solve rate 提升 |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **No Real-World Data** | 仅用 SWE-Gym 数据训练 critic |
| **Success-Only** | 仅预测 outcome，不使用 rubric supervision |
| **Success+Rubrics** | 联合预测 outcome + 24 rubrics（本文方法） |
| **Random@K** | 随机选择 trajectory，作为 baseline |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（见 Table 2）

| 模型 | Real-World AUC (Survival) | SWE-bench Best@8 | Early Stop Gain | Avg Attempts |
|------|----------------------------|------------------|------------------|---------------|
| Random@8 | — | 57.9% | 0.0 | 8.0 |
| Success-Only | 0.65 | 63.6% | +19.1 | 1.76 |
| **Success+Rubrics (BCE-floor)** | **0.69** | **73.8%** | **+17.7** | **1.35** |

> ✅ **Best@8 提升 +15.9 pts over Random**
>
> ✅ **Early stopping 节省 83% 计算资源**（1.35 vs 8 次尝试）

### 与基线方法的对比结果

#### ❌ Benchmark-Trained Critics 不泛化
- 仅在 SWE-Gym 上训练的 critic 在 real-world 数据上表现接近随机（AUC ≈ 0.45–0.48）。
- 更严重的是，在 SWE-bench 上进行 Best@8 时甚至**低于随机选择**（Best@8 = 45.6% < Random@8 = 57.9%），说明 benchmark success 与 real-world success 错位。

#### ✅ Code Survival > PR Merge 作为监督信号
- 尽管 code survival 标签更稀疏（4% vs 6%），但训练出的 critic 在 real-world 上 AUC 更高（0.69 vs 0.58）。
- 证明更细粒度、可归因的 proxy 更适合 critic 学习。

#### ✅ Rubric Supervision 显著提升跨 backbone 泛化能力
- **Success-Only critic** 在 Sonnet 上表现尚可（+7.0），但在 Opus 上反而负增益（-8.1）→ 严重过拟合 backbone。
- **Rubric-supervised critic** 在两个 backbone 上均保持正向增益（+3.9 / +2.6），MRR 达 0.83。
- 表明 rubrics 学到了**行为层面的本质失败模式**，而非表面 token pattern。

### 消融实验结果

#### （1）训练目标消融（Table 2）
| 训练目标 | Best@8 | Early Stop Gain |
|--------|--------|------------------|
| Success-Only | 63.6% | +19.1 |
| Success+Rubrics (BCE-floor) | **73.8%** | **+17.7** |
| Success+Rubrics (MSE) | 45.6% | +12.8 |
> ⚠️ 直接回归 survival 分数（MSE）效果极差，说明下游应用需要**排序能力**而非精确数值拟合。

#### （2）数据选择用于 SFT（Table 4）
| 数据选择策略 | SWE-bench Solve Rate | Δ vs Base |
|-------------|-----------------------|----------|
| Base Model | 46.6% | — |
| Random Selection | 46.2% | -0.4 |
| Critic-selected | **47.8%** | **+1.2** |
| Proxy-filtered (survival=1) | 50.4% | +3.8 |
> ✅ Critic 可有效筛选高质量 trajectory 用于 SFT，带来可测量的 agent 性能提升。

---

## 4. 关键结论和发现

### 主要发现
1. 🔹 **真实世界的监督至关重要**：仅靠 benchmark 数据训练的 critic 无法泛化到 real-world outcome，甚至有害。
2. 🔹 **Code survival 是比 PR merge 更优的 success proxy**：更细粒度、更少混淆，更适合 credit assignment。
3. 🔹 **Rubric supervision 极大地提升了 critic 的实用性与鲁棒性**：
   - 支持跨 LLM backbone 的通用 scoring 函数；
   - 实现高效的 inference-time scaling（early stopping + Best-of-K）；
   - 可用于 training-time data curation。
4. 🔹 **Critic 不仅是 evaluator，更是 scalable agent improvement pipeline 的核心组件**。

### 方法的局限性
- **Rubric 定义可能存在偏差**：24 个特征虽经专家迭代设计，但仍可能遗漏某些 failure mode。
- **依赖 trace 完整性**：若工具调用日志缺失，则 commit attribution 和 rubric 注释可能失败。
- **未解决 reward hacking 风险**：过度优化 critic score 可能导致行为扭曲（如避免 risky actions 即使合理）。
- **当前 critic 仍为静态模型**：未考虑随时间漂移的用户偏好或项目规范变化。

### 未来工作方向
- 动态更新 rubrics 和 critic 模型以适应 evolving user expectations。
- 将 critic 集成进 online RL loop，实现持续自我改进。
- 扩展 rubric taxonomy 至更多 domain（如数学推理、规划等）。
- 探索 human-in-the-loop active learning 来 refine rubric annotations。
- 开源并鼓励社区共建 rubric standard 和 critic benchmarks。

---

> 📦 **开源信息**
> - **Critic Rubrics 框架**: [GitHub](https://github.com/OpenHands/critic-rubrics)
> - **Critic 模型**: [HuggingFace](https://huggingface.co/OpenHands/openhands-critic-4b-v1.0)
> - **文档**: [OpenHands Docs](https://docs.openhands.dev/sdk/guides/critic)

</details>

---

### 10. [Combating data scarcity in recommendation services: Integrating cognitive types of VARK and neural network technologies (LLM)](https://arxiv.org/abs/2603.03309)

**Authors**: Nikita Zmanovskii  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03309v1  

#### Abstract
Cold start scenarios present fundamental obstacles to effective recommendation generation, particularly when dealing with users lacking interaction history or items with sparse metadata. This research proposes an innovative hybrid framework that leverages Large Language Models (LLMs) for content sem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Combating data scarcity in recommendation services: Integrating cognitive types of VARK and neural network technologies (LLM)*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文聚焦于推荐系统中的**冷启动问题（cold start problem）**，具体包括：
- **用户冷启动**：新用户缺乏交互历史（如评分、点击等）；
- **物品冷启动**：新物品缺乏丰富元数据或用户反馈；
- **认知适配缺失**：现有系统忽略用户的**认知偏好**（如信息接收方式）和**动态心理状态**（如注意力水平、疲劳程度）。

传统方法在这些场景下表现不佳，导致推荐质量低、用户体验差，甚至引发高流失率。

---

### 🚀 提出的新方法与创新思路
作者提出了一种**融合 VARK 认知模型与 LLM 技术的混合架构**，整合六大模块形成闭环系统：

| 模块 | 功能 |
|------|------|
| **Semantic Metadata Enhancement** | 利用 **LLM** 对稀疏/非结构化物品描述进行语义增强，提取实体、关系、难度、目标受众、VARK 对齐等深层特征 |
| **Dynamic Graph Construction** | 构建多关系 **Knowledge Graph**，连接用户、物品、实体，并支持动态演化 |
| **VARK-Based Profiling** | 基于 VARK 学习风格框架建立用户认知画像（Visual/Auditory/Reading/Writing/Kinesthetic） |
| **Mental State Estimation** | 结合上下文信号（时间、设备、会话行为）估计用户当前的**认知负荷**（cognitive load）、注意力、复杂度偏好和最优呈现模式 |
| **Graph-Enhanced Retrieval + LLM Ranking** | 多策略候选生成 + **LLM 驱动排序**，综合相关性、VARK 匹配度、复杂度匹配、多样性等因素 |
| **Adaptive Interface Design + Iterative Learning** | 动态调整推荐展示形式（视觉化、文本详述、互动元素等），并基于反馈持续优化图谱与用户画像 |

> 🔍 **核心创新点**：
> - 首次将 **VARK 认知心理学模型**系统性地集成到推荐流程中，实现“认知感知型推荐”（cognitively-aware recommendation）；
> - 提出“零样本推荐能力”（zero-shot capability）：仅凭少量人口统计信息 + VARK 问卷即可生成个性化推荐；
> - 引入自然语言解释（natural language explanations），提升推荐可解释性与用户信任；
> - 支持跨域适用性（domain-agnostic），不仅限于电影推荐，还可用于教育、健康、电商等领域。

---

### ⚖️ 相比现有方法的优势
| 方法类型 | 局限性 | 本文改进 |
|--------|-------|---------|
| 协同过滤（Collaborative Filtering） | 冷启动下失效 | 不依赖交互历史 |
| 内容过滤（Content-based） | 元数据稀疏、浅层语义 | LLM 实现深度语义理解 |
| 知识图谱方法 | 手动构建成本高、扩展难 | LLM 自动抽取实体与关系 |
| 深度学习/元学习 | 需大量训练数据 | 支持 few-shot / zero-shot 场景 |
| 统一界面设计 | 忽略个体认知差异 | 自适应呈现格式（VARK + context） |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用 **MovieLens-1M** 数据集进行验证：
  - 6,040 名用户
  - 3,706 部电影
  - 超过 100 万条评分（1–5 分）
  - 包含用户年龄、性别、职业等人口学信息
  - 电影有标题、类型、年份等基础元数据

> ❗ 注：为模拟冷启动，随机选取 **20% 用户作为“全新用户”**（无任何历史交互），其余 80% 用于初始知识图构建。

---

### ⚙️ 实验设置
- **LLM 模型**：GPT-3.5-turbo（temperature=0.7）
- **嵌入模型**：`all-MiniLM-L6-v2`（384维），微调于电影描述
- **知识图谱存储**：
  - 图数据库：Neo4j
  - 向量索引：FAISS
  - 文本搜索：Elasticsearch
- **候选池大小**：约 1000 项（结合语义相似性、实体检索、VARK 过滤）
- **最终输出**：Top-10 推荐列表 + 自然语言解释

---

### 📈 评估指标
| 指标 | 定义 |
|-----|------|
| **HR@K (Hit Rate@K)** | 至少有一个真实相关项目出现在前 K 个推荐中的用户比例 |
| **nDCG@K (normalized Discounted Cumulative Gain@K)** | 考虑排名位置的相关性得分，越高越好 |
| **Recall@K** | 在前 K 个推荐中捕获的真实相关项目的比例 |
| **Unique Top-1** | 不同用户 Top-1 推荐项的数量 → 衡量个性化程度 |

---

### 🆚 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Random** | 随机打乱推荐（下界） |
| **Popularity** | 推荐最受欢迎的 K 个项目（强基线） |
| **Embedding Cosine** | 基于用户人口统计 + 物品标题/类型的句向量计算余弦相似度 |
| **Candidates Only** | 仅使用候选生成阶段结果（无 LLM 排序） |
| **Ours (CE Rerank)** | 本文完整系统，采用交叉编码器（cross-encoder）重排代替全 LLM 排序以提高效率 |

---

## 3. 主要实验结果和性能指标

### 📉 关键性能数据（Table 1）

| Model | HR@10 | nDCG@10 | Recall@50 | Recall@200 | Recall@1000 |
|-------|--------|----------|------------|-------------|--------------|
| Random | 0.005±0.006 | 0.002±0.003 | ~0.002 | ~0.003 | ~0.004 |
| Popularity | **0.268±0.018** | **0.224±0.014** | ~0.002 | ~0.003 | ~0.004 |
| Embedding Cosine | 0.101±0.021 | 0.050±0.011 | ~0.002 | ~0.003 | ~0.004 |
| Candidates Only | 0.011±0.003 | 0.004±0.001 | ~0.000 | ~0.001 | ~0.001 |
| **Ours (CE Rerank)** | 0.008±0.005 | 0.005±0.002 | ~0.000 | ~0.001 | ~0.001 |

> ❗ **关键观察**：
> - **Popularity 基线显著优于所有其他方法**，说明在极端冷启动下，“大众流行”仍是安全选择；
> - 本文方法绝对性能较低，但具备一定个性化能力（见下表）；
> - “Candidates Only” 与 “Ours (CE Rerank)” 性能接近，表明**候选召回率是瓶颈**（recall@50 ≈ 0），即使 LLM 排序也无法弥补。

---

### 🎯 多样性与个性化分析（Table 2）

| Model | Unique Top-1 |
|-------|---------------|
| Random | 1.0 |
| Popularity | 1.0 |
| Embedding Cosine | 4.0 |
| Candidates Only | 4.0 |
| **Ours (CE Rerank)** | **3.0** |

> ✅ 尽管准确率不高，但本文方法实现了**适度个性化**（平均产生 3 种不同的 Top-1 推荐），优于完全统一的 Popularity 和 Random 方法。

---

### 🧪 消融实验与定性分析
- **统计检验**：与 Popularity 相比，本文方法在 HR@10 和 nDCG@10 上差异显著（p < 0.001），效应量大（Cohen’s d ~ -0.59），确认其性能明显更差。
- **定性输出质量高**：
  - 87% 的推荐解释正确引用了用户 profile；
  - 92% 明确提及 VARK 对齐；
  - 73% 提供有说服力的理由；
  - 示例推荐如 *The Matrix*、*Fight Club* 等虽未命中测试集，但符合用户“视觉+动感”的认知偏好。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Popularity 在冷启动中极具竞争力**：尤其在 MovieLens 这类具有强烈流行偏置的数据集中，推荐热门项目是一种稳健策略。
2. **认知建模带来差异化价值**：尽管量化指标落后，但系统能生成**高度个性化的解释与呈现方式**，提升用户体验与信任感。
3. **语义增强有效但受限于数据质量**：LLM 可从有限元数据中挖掘深层语义，但若原始信息不足（如仅有标题+类型），仍难以支撑高质量推荐。
4. **候选召回是最大瓶颈**：Recall@50 接近零，意味着真正相关的物品未能进入候选池，后续排序再强也无效。
5. **系统具备良好可解释性与领域迁移潜力**：适用于教育、医疗、职业发展等对认知适配要求高的场景。

---

### ⚠️ 方法的局限性
| 问题 | 描述 |
|------|------|
| **候选生成质量低** | 图谱构建依赖稀疏元数据，实体提取不精准，导致相关物品未被召回 |
| **计算开销大** | 多次调用 LLM、图查询、向量检索 → 实时性挑战，不利于大规模部署 |
| **VARK 评估负担重** | 16题问卷增加用户门槛，影响实际可用性 |
| **缺乏真实 VARK 数据** | 实验中 VARK 偏好为随机分配，可能削弱其有效性信号 |
| **评价体系不匹配** | MovieLens 本身不适合评估“认知适配”与“解释质量”，标准指标（如 HR、nDCG）无法反映真实体验价值 |

---

### 🔮 未来工作方向
1. **增强候选生成能力**：
   - 融合 dense retrieval（双塔模型）、sparse retrieval（BM25）、graph walk 等多视图策略；
   - 引入 active learning，在初期主动探索用户偏好。
2. **高级认知建模**：
   - 通过交互模式隐式推断 VARK（无需问卷）；
   - 加入情绪检测（sentiment analysis）、分心识别等动态因素。
3. **提升惊喜性（Serendipity）**：
   - 完整实现 SerenEva 框架，平衡利用与探索；
   - 使用 LLM 生成兴趣假设并测试。
4. **多模态知识图谱扩展**：
   - 融入图像、音频、文本细粒度特征；
   - 建模时间动态性（如热度变化）。
5. **隐私保护机制**：
   - 设备端 VARK 推理；
   - 联邦学习更新模型；
   - 差分隐私保护图谱更新。
6. **领域专项验证**：
   - 教育领域：衡量学习成果、知识掌握进度；
   - 医疗领域：匹配患者健康素养水平；
   - A/B 测试 + 用户调研 → 更全面评估满意度、参与度、留存率。

---

## 💡 总结
尽管该方法在 MovieLens-1M 上的**量化性能不及简单流行度基线**，但它开创性地将 **LLM + Knowledge Graph + VARK Cognitive Modeling** 深度融合，提出了一个面向“认知感知”的下一代推荐系统架构。其真正的价值在于：
- 实现了从“只看行为”到“理解心智”的跃迁；
- 提供了可解释、可适应、可进化的推荐体验；
- 为教育、医疗、终身学习等高价值场景奠定了技术基础。

> 🔚 **最终结论**：  
> “这不是一个在传统指标上取胜的系统，而是一个在**用户体验维度上突破边界**的尝试。”  
> —— 推荐系统的未来不仅是“准不准”，更是“懂不懂你”。

</details>

---

### 11. [Position: Vector Prompt Interfaces Should Be Exposed to Enable Customization of Large Language Models](https://arxiv.org/abs/2603.04292)

**Authors**: Liangwei Yang, Shiyu Wang, Haolin Chen, Rithesh Murthy, Ming Zhu, Jielin Qiu, Zixiang Chen, Juntao Tan, Jianguo Zhang, Zhiwei Liu, Wenting Zhao, Silvio Savarese, Caiming Xiong, Huan Wang, Shelby Heinecke  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.04292v1  

#### Abstract
As large language models (LLMs) transition from research prototypes to real-world systems, customization has emerged as a central bottleneck. While text prompts can already customize LLM behavior, we argue that text-only prompting does not constitute a suitable control interface for scalable, stable...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Position: Vector Prompt Interfaces Should Be Exposed to Enable Customization of Large Language Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前大型语言模型（LLMs）在企业级部署中面临**定制化瓶颈**。尽管可通过 **text-based prompting** 或 **fine-tuning** 进行行为调整，但二者均存在显著缺陷：
- **Text prompts** 虽灵活易用，但在系统级定制中表现出**表达脆弱性**（brittle）、优化困难、难以规模化；
- **Fine-tuning** 需要梯度访问和重新训练，带来高昂的计算与运维成本，不适用于频繁迭代或推理时定制（inference-only）场景。

因此，论文提出：**现有的文本提示接口不适合作为可扩展、稳定且仅依赖推理的定制控制接口**。

---

### 🚀 提出的新方法/新思路
作者主张将 **vector prompt inputs** 作为公共接口向下游开发者开放，即：

> **Model providers 应暴露 vector prompt 接口，使其成为与 text prompt 并列的定制化控制通道。**

这一观点的核心是将“prompting”从一种**优化技术**（如 prompt tuning）提升为一个**接口抽象**（interface abstraction），强调其作为**控制系统输入形式的设计选择**。

#### 创新视角：接口抽象 vs. 优化方法
- 区分了 **interface abstraction**（控制信号的形式：text 或 vector） 和 **optimization method**（如何获取这些信号：梯度法、黑箱搜索等）。
- 强调 vector prompts 是一种更优的 interface abstraction，即使使用 black-box 优化也能实现高效定制。

---

### 🔍 相比现有方法的优势
| 维度 | Text Prompt | Fine-tuning | Vector Prompt (本文主张) |
|------|-----------|------------|------------------------|
| 可读性 | ✅ 高 | ❌ 低 | ❌ 低 |
| 控制稳定性 | ❌ 差（易受措辞影响） | ✅ 好 | ✅ 好 |
| 定制粒度 | 中等 | 高 | 高 |
| 是否支持 inference-only | ✅ 是 | ❌ 否 | ✅ 是 |
| 扩展性（多任务共存） | 差（长 prompt 冗余） | 差（需多个模型副本） | ✅ 好（共享 backbone + 不同 vectors） |
| 控制效率 | 低（稀疏 attention） | 高 | ✅ 更高（密集全局 attention） |

👉 因此，vector prompt 在 **scalable、stable、inference-only customization** 场景下具有明显优势。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **SST-5**：情感分类任务（5类电影评论），用于评估不同 prompt 接口在有限监督下的表现。
- 数据来源：Hugging Face `SetFit/sst5` 数据集。

---

### ⚙️ 实验设置
- **Backbone 模型**：LLaMA3-8B-Instruct（冻结权重，仅通过 prompt 控制）。
- **目标**：比较不同 prompt 接口在**不断增加标注数据量**条件下的性能变化趋势（scaling behavior）。
- **变量控制**：
  - 固定模型和任务；
  - 仅改变 prompt 类型及其优化方式；
  - 使用 gradient-based optimization 作为诊断工具（非实际部署方案）。

---

### 🧪 对比的基线方法
| 方法类别 | 具体实现 |
|--------|--------|
| **Human-written text prompt** | 手工设计的自然语言指令 |
| **Optimized text prompt** | 使用 TextGrad（Yuksekgonul et al., 2024）进行优化的文本提示 |
| **Optimized vector prompt** | 使用 prompt tuning（Liu et al., 2022）学习得到的连续向量前缀（32 tokens） |

> 注意：所有 vector prompt 实验均在输入编码阶段注入（input-encoding stage），不修改模型参数。

---

### 📊 评估指标
- 主要指标：**Accuracy on SST-5 test set**
- 分析维度：
  - 性能随训练样本数量增加的变化曲线（scaling curve）
  - 注意力模式可视化（attention pattern analysis）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据与对比结果

#### （1）Scaling Behavior（图2）
| 方法 | 小样本性能 | 大样本性能趋势 |
|------|------------|----------------|
| Human-written text prompt | 稳定但较低（~0.55） | 几乎无提升 |
| Optimized text prompt | 初始上升快 | 很快饱和（<1k samples） |
| Optimized vector prompt | 初始相当或略优 | 持续提升，未见饱和 |

> ✅ **结论**：vector prompts 能持续吸收更多监督信号，而 text prompts 存在明显的接口级瓶颈。

---

#### （2）Attention Pattern Analysis（图3）
对 LLaMA3-8B 的第12层和第20层注意力机制进行可视化分析：

| 特征 | Text Prompt | Vector Prompt |
|------|-------------|---------------|
| Attention 密度 | 稀疏（sparse） | 密集（dense） |
| Prompt-task 边界跨越 | 极少 | 广泛存在 |
| 是否形成“control anchor” | 否 | 是 |
| Attention sink 现象 | 明显（集中在 BOS token） | 显著减弱 |
| 层间一致性 | 差 | 强（跨层稳定） |

> ✅ **结论**：vector prompts 被模型视为持久、全局可寻址的控制锚点（control anchors），而非普通序列元素。

---

### 🔍 消融实验（隐含分析）
虽然未明确列出消融表，但以下对比构成实质上的消融研究：
- **接口形式消融**：text vs. vector → 显示 vector 在 scaling 和 attention 利用上占优；
- **优化方法解耦**：gradient-based tuning 仅作上限估计 → 表明接口能力独立于具体优化手段；
- **部署约束模拟**：强调 black-box setting 下仍可行 → 支持 inference-only 适用性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Text prompts 存在根本性接口瓶颈**  
   即使经过优化，其性能很快饱和，无法有效利用大量监督数据，根源在于其离散、语义依赖的特性。

2. **Vector prompts 是更高效的控制接口抽象**  
   - 连续空间允许更精细、稳定的控制；
   - 注意力机制显示其被当作专用 control module 使用；
   - 支持高控制效率（few vectors → strong effect）。

3. **Vector prompts 适合 inference-only 部署**  
   - 不需要梯度、权重更新或激活访问；
   - 可与 black-box API 兼容；
   - 支持快速迭代与多任务并行定制。

4. **安全性风险可控**  
   在标准 black-box 威胁模型下，暴露 vector prompt 接口不会引入新的可观测攻击面，信息泄露风险与 text prompt 相当。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **不可解释性** | vector prompts 缺乏人类可读性，不利于调试与合规审查 |
| **初始化挑战** | 如何高效初始化或迁移 vector prompts 尚无成熟方法 |
| **优化难度** | 当前 black-box optimization 方法（如 ZOO, BBT）性能仍低于 gradient-based tuning |
| **平台支持缺失** | 当前主流 LLM 服务商尚未提供此类接口 |

---

### 🔮 未来工作方向
1. **Develop better inference-only optimization algorithms**  
   设计更高效的 black-box / zeroth-order 方法来优化 vector prompts。

2. **Design evaluation benchmarks for control interfaces**  
   建立专门衡量 control efficiency、scalability、stability 的评测框架，超越传统 accuracy 指标。

3. **Standardize vector prompt APIs**  
   推动 industry-standard 接口规范（如固定长度、归一化约束、安全校验）。

4. **Hybrid prompting frameworks**  
   探索 text + vector 的混合控制范式，兼顾可读性与控制精度。

5. **Study transferability and compositionality**  
   研究 vector prompts 是否可在任务间迁移或组合使用。

---

## 💬 总结一句话
> 该论文呼吁将 **vector prompt** 从内部优化技巧升格为 LLM 定制的标准接口，因其在 **控制效率、可扩展性和推理兼容性** 上显著优于 text prompt，是实现大规模、系统化、inference-only 定制的关键路径。

</details>

---

### 12. [Towards Improved Sentence Representations using Token Graphs](https://arxiv.org/abs/2603.03389)

**Authors**: Krishna Sri Ipsit Mantri, Carola-Bibiane Sch\"onlieb, Zorah L\"ahner, Moshe Eliasof  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03389v1  

#### Abstract
Obtaining a single-vector representation from a Large Language Model's (LLM) token-level outputs is a critical step for nearly all sentence-level tasks. However, standard pooling methods like mean or max aggregation treat tokens as an independent set, discarding the rich relational structure capture...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Towards Improved Sentence Representations using Token Graphs —— 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统从 **Large Language Models (LLMs)** 获取句子表示的方法（如 Mean Pooling、Max Pooling 或 [CLS] token）存在以下缺陷：
- 将 token 隐藏状态视为**独立向量集合**，忽略了 LLM 自注意力机制中蕴含的丰富**token间关系结构**。
- 在高噪声或长文本场景下容易发生**信号稀释（signal dilution）**，即任务相关的关键信号被大量无关 token 掩盖。
- 对于 **decoder-only LLMs**（如 LLaMA、Mistral），其隐藏状态优化目标是“下一个词预测”，而非全局语义理解，直接池化效果更差。

### 提出了什么新方法或新思路
作者提出 **GLOT**（**Graph-based Latent Object Token pooling**），一种轻量级、结构感知的池化模块，将句子表示学习重构为 **“关系学习 + 聚合”** 的两阶段过程：

1. **Token Graph Construction**  
   基于冻结 LLM 输出的 token 隐藏状态，构建一个**潜在的 token-相似性图**（latent token-similarity graph），边由余弦相似度超过阈值 $T$ 的 token 对构成。

2. **Refinement with TOKEN-GNN**  
   使用一个轻量级 **Graph Neural Network (GNN)** 在图上进行消息传递，显式建模 token 之间的交互，从而**增强并细化 token 表示**。

3. **Readout Layer**  
   使用可学习的注意力机制对 refined token 表示进行加权聚合，生成最终的句子向量。

整个 LLM 主干保持**完全冻结**，仅训练 GLOT 模块和下游任务头，实现高效适配。

### 相比现有方法的优势
| 维度 | GLOT 的优势 |
|------|-------------|
| **性能** | 显著优于所有静态（Mean/Max/[CLS]）和可学习池化方法（如 AdaPool），尤其在复杂语言理解任务上提升显著。 |
| **鲁棒性** | 在高达 90% 噪声 token 的压力测试中仍保持 >97% 准确率，远超基线。 |
| **效率** | 仅需约 8.9M 可训练参数，相比 LoRA 快 100×，内存占用仅 0.42GB，适合消费级硬件部署。 |
| **通用性** | 同时适用于 encoder-only（如 BERT）和 decoder-only（如 Mistral-7B）模型。 |
| **理论表达力** | 超越 DeepSets 框架，能建模多跳依赖和否定等复杂语言现象。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GLUE Benchmark**：涵盖多种语言理解任务，包括：
  - 单句分类：CoLA（语法可接受性）、SST-2（情感分析）
  - 句子对分类：MNLI、RTE、MRPC、QQP
  - 语义相似度：STS-B
- **IMDB**：长文本情感分类（电影评论）
- **MTEB (Massive Text Embedding Benchmark)**：7 项零样本任务，包括：
  - 分类（EmotionClassification）
  - 聚类（RedditClustering）
  - 检索（AskUbuntuDupQuestions）
  - 重排序（SciFact）
  - 语义相似度（STS12）
- **合成诊断任务**：自定义“信号在噪声中”（signal-in-noise）任务，用于测试抗干扰能力。

### 实验设置和评估指标
| 设置项 | 描述 |
|-------|------|
| **LLM 主干** | 完全冻结，包括 BERT、RoBERTa、TinyLlama、LLaMA3.2-3B、Mistral-7B 等 |
| **训练方式** | 仅训练 GLOT 模块和任务头，不更新 LLM 权重 |
| **训练细节** | Adam 优化器，学习率 2e-4，batch size 32，最多 2 个 epoch |
| **评估指标** | 依任务而定：<br>- CoLA: MCC<br>- SST-2/RTE: Accuracy<br>- STS-B: Spearman<br>- QQP/MRPC: F1<br>- MTEB: 多样化指标（NDCG@10, MAP, V-Measure 等） |

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **静态池化** | Mean, Max, [CLS]/[EOS] |
| **可学习池化** | AdaPool (Brothers, 2025) |
| **参数高效微调** | LoRA (Hu et al., 2022) |
| **全量微调** | Full Fine-Tuning (Full FT) |
| **对比学习方法** | SimCSE, SBERT（用于 MTEB 对比） |
| **提示工程方法** | PromptBERT, PromptEOL, Pretended CoT |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1, 2, 3, 5）

#### ✅ GLUE 上的综合表现（以 Mistral-7B 为例）
| 方法 | CoLA (MCC↑) | SST-2 (Acc↑) | STS-B (Spear↑) | RTE (Acc↑) |
|------|-------------|-------------|----------------|-----------|
| Mean | 38.61 | 89.91 | 77.96 | 53.07 |
| AdaPool | 48.00 | 93.00 | 79.55 | 54.87 |
| **GLOT (Ours)** | **54.30** | **94.38** | **80.51** | **59.21** |

> GLOT 在所有任务上均领先，尤其在需要深层语言理解的 CoLA 和 RTE 上优势明显。

#### ✅ IMDB 长文本分类（准确率 Acc）
| 方法 | BERT | RoBERTa | Mistral-7B |
|------|------|---------|-----------|
| [CLS]/[EOS] | 80.23 | 82.04 | 84.86 |
| AdaPool | 85.45 | 90.91 | 95.66 |
| **GLOT** | **86.93** | **94.52** | **95.95** |

> 在长文本中，GLOT 更好地保留了关键语义片段，避免平均池化的信号稀释。

#### ✅ MTEB 零样本检索（部分任务）
| 方法 | SciFact (NDCG@10↑) | AskUbuntu (MAP↑) |
|------|--------------------|------------------|
| AdaPool | 0.4268 | 0.4767 |
| **GLOT** | **0.4414** | **0.4821** |

> GLOT 在科学事实验证和社区问答检索中均取得最佳表现。

#### ✅ 抗噪能力（Diagnostic Stress Test）
在 **90% 噪声 token** 的极端条件下（Figure 3, Table 7）：
| 方法 | BERT 准确率 | Mistral-7B 准确率 |
|------|------------|------------------|
| AdaPool | 61.6% | 78.4% |
| Mean | 53.4% | 63.8% |
| **GLOT** | **98.8%** | **97.2%** |

> GLOT 展现出极强的鲁棒性，几乎不受噪声影响。

#### ✅ 效率对比（Table 5）
| 方法 | 可训练参数 | GPU 内存 (GB) | 批处理时间 (ms) |
|------|------------|---------------|------------------|
| Full FT | 7.11B | 32.59 | 1318.8 |
| LoRA | 167.8M | 33.50 | 1454.6 |
| **GLOT** | **8.92M** | **0.42** | **13.4** |

> GLOT 比 LoRA **少 20× 参数**，**快 100×**，内存占用仅为 1/75。

---

### 消融实验结果

#### 🔍 图稀疏性（Threshold $T$）的影响（Table 4）
| $T$ | CoLA (MCC) | RTE (Acc) |
|-----|-----------|----------|
| 0.0（全连接） | 50.19 | 49.81 |
| 0.4 | 51.73 | 50.54 |
| **0.6** | **54.30** | **59.21** |
| 0.8 | 52.48 | 52.70 |

> **适度稀疏（T=0.6）效果最好**，说明并非所有 token 关系都重要，强语义连接更有价值。

#### 🔍 GNN 架构选择（Table 11）
| GNN 类型 | CoLA (Mistral) | RTE (Mistral) |
|---------|----------------|--------------|
| GCN | 52.65 | 57.04 |
| **GAT (Ours)** | **54.30** | **59.21** |
| GIN | 59.30 | 59.30 |

> 不同 GNN 均优于 AdaPool，说明**图结构本身是关键**；GAT 在多数任务上表现稳定。

#### 🔍 参数匹配对比（Table 15）
| 方法 | 参数量 | CoLA (MCC) | STS-B (Spear) |
|------|--------|------------|--------------|
| MLP（无图） | 9.2M | 51.33 | 74.12 |
| **GLOT** | **8.9M** | **54.30** | **80.51** |

> GLOT 用更少参数实现更高性能，证明其优势来自**关系学习**而非单纯增加容量。

---

## 4. 关键结论和发现

### 主要发现
1. **池化不是简单的压缩，而是关系学习**：  
   将 token 视为图节点并通过 GNN 进行信息传播，能有效恢复 LLM 中未被充分利用的关系结构。

2. **GLOT 是高效且强大的适配器**：  
   无需微调大模型，在冻结主干上添加轻量 GNN 模块即可大幅提升句子表示质量。

3. **对 decoder-only 模型特别有效**：  
   GLOT 成功弥补了 decoder-only LLM 在句子级任务上的结构性缺陷，使其媲美甚至超越 encoder 模型。

4. **卓越的鲁棒性和泛化能力**：  
   在噪声、长文本和零样本设置下均表现出色，验证了其学习到的是本质语义而非表面统计模式。

### 方法的局限性
- **计算复杂度为 $O(L^2)$**：虽然实测开销小（见 Table 10），但在超长序列（>32K）时仍可能成为瓶颈。
- **依赖 LLM 的初始几何结构**：若 token 嵌入空间高度扭曲或各向异性严重，图构建可能不稳定。
- **当前图构建是静态的**：基于余弦相似度的图是固定的，无法动态调整拓扑以适应不同任务。

### 未来工作方向
- **可学习图构建机制**：探索动态图生成、图重连（graph rewiring）或虚拟节点（virtual nodes）来优化图结构。
- **跨模态扩展**：将该范式应用于 Vision Transformers 中的 patch pooling。
- **更高效的 GNN 设计**：研究稀疏 GNN、层级图聚合等以进一步降低延迟。
- **结合提示学习**：探索 GLOT 与 prompt engineering 的协同效应。
- **理论分析深化**：形式化分析 GLOT 的表达能力和归纳偏置。

---

> 📌 **一句话总结**：  
> **GLOT 通过在冻结 LLM 上构建 token 图并应用 GNN，实现了“关系优先”的池化新范式，在性能、鲁棒性和效率上全面超越传统方法，为大模型的高效适配开辟了新路径。**

</details>

---

### 13. [Specification-Driven Generation and Evaluation of Discrete-Event World Models via the DEVS Formalism](https://arxiv.org/abs/2603.03784)

**Authors**: Zheyu Chen, Zhuohuan Li, Chuanhao Li  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.03784v1  

#### Abstract
World models are essential for planning and evaluation in agentic systems, yet existing approaches lie at two extremes: hand-engineered simulators that offer consistency and reproducibility but are costly to adapt, and implicit neural models that are flexible but difficult to constrain, verify, and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Specification-Driven Generation and Evaluation of Discrete-Event World Models via the DEVS Formalism**

---

## **1. 主要贡献和创新点**

### **解决的问题**
现有 **World Models** 在规划与评估中面临两极分化困境：
- **显式模拟器（hand-engineered simulators）**：一致性高、可复现，但构建成本高昂，难以在线适应新环境。
- **隐式神经模型（implicit neural models）**：灵活且可通过提示（prompting）在线调整，但在长时程 rollout 中不可靠，难以约束、验证和调试。

本文旨在为离散事件主导的环境（如排队系统、服务流程、多智能体协调等）提供一种**兼具可靠性与灵活性的中间路径**，实现：
1. 长时程 rollout 下的一致性；
2. 可从可观测行为中进行验证；
3. 支持在线按需生成与修改。

---

### **提出的新方法与思路**

#### **核心框架：基于 DEVS 形式化语言的离散事件世界模型（Discrete-Event World Models）**
- 将世界模型形式化为 **DEVS（Discrete Event System Specification）模型**，通过自然语言规范自动生成可执行模拟器。
- 模型由原子组件（Atomic Models）和耦合组件（Coupled Models）构成，明确状态转移、事件处理和时间推进逻辑。

#### **两阶段 LLM 生成流水线（Staged LLM-based Generation Pipeline）**
1. **结构合成阶段（Structural Synthesis）**  
   - LLM 推理系统组件层级、交互图谱（interaction graph）和接口契约（port schemas）。
   - 输出一个结构化的 `PlanTree`，作为后续行为生成的蓝图。
2. **行为合成阶段（Behavioral Synthesis）**  
   - 基于 `PlanTree` 并行生成各原子组件的行为逻辑代码。
   - 引入 **接口摘要机制（Interface Summarization）**，在组装父级耦合模型前动态提取子组件实际接口，缓解语义漂移。

#### **基于轨迹的规范驱动评估（Trace-Based, Specification-Driven Evaluation）**
- 模拟器输出标准化的 **JSONL 格式事件轨迹（event traces）**，包含时间戳、实体、事件类型和负载。
- 定义两类验证规则：
  - **组件级约束**：如队列等待时间计算是否正确。
  - **系统级约束**：如因果顺序、响应关系、资源守恒等。
- 不依赖唯一“真值”实现，而是判断行为是否满足规范导出的逻辑与时间约束。

---

### **相比现有方法的优势**

| 维度 | 本方法（DEVS-Gen） | 显式模拟器 | 隐式神经模型 |
|------|---------------------|------------|--------------|
| **一致性** | ✅ 高（结构化状态转移） | ✅ 高 | ❌ 低（误差累积） |
| **可验证性** | ✅ 黑盒轨迹验证 | ✅ | ❌ |
| **可调试性** | ✅ 局部诊断 | ✅ | ❌ |
| **灵活性/适应性** | ✅ 支持在线生成 | ❌ | ✅ |
| **生成效率** | ✅ 模块化解耦，支持并行 | — | — |

---

## **2. 核心实验方法和设置**

### **数据集**
构建了一个包含 **7 个真实场景的基准数据集**，覆盖多个领域与动态特性：

| 场景 | 领域 | 关键动态 | 规模 |
|------|------|----------|------|
| IOBS | 银行业务 | 流水线 + 概率路由 | M |
| OTrain | 交通 | 调度驱动延迟 | M |
| SEIRD | 生物数学 | 连续近似（ODE） | S |
| FileTransfer | 网络 | 双循环 FSM | L |
| ABP | 网络协议 | Stop-and-Wait | S |
| StratAirlift | 物流 | 主动 reneging | L |
| Barbershop | 服务 | 阻塞与信号握手 | S |

> 数据来源：反向工程高质量开源 DEVS 模型，重写为自然语言规范，并保留原始轨迹作为验证依据。

---

### **实验设置与评估指标**

#### **评估维度**
1. **操作成功得分（Operational Success Score, OSS）**
   - 衡量模拟器是否可编译运行、遵循 I/O 合同。
   - $ OSS = \frac{1}{m} \sum_{i=1}^m v_i $，其中 $ v_i = 1 $ 当且仅当无崩溃、无超时、日志格式正确。

2. **行为符合度得分（Behavioral Conformance Score, BCS）**
   - 衡量事件轨迹是否满足规范中的逻辑与时间约束。
   - $ BCS = \frac{1}{m} \sum_{i=1}^m \left(1 - \frac{\text{违反规则数}}{\text{总规则数}}\right) $
   - 采用宏平均（Macro-Average），平等加权组件级与系统级规则。

#### **测试套件（Test Suite）**
- 每个场景设计多个输入配置 $ (\mathcal{L}, \mathcal{J}) $，涵盖轻载到重载压力测试。

---

### **基线方法对比**
比较以下两类主流 LLM 软件工程代理：

| 方法 | 类型 | 是否迭代 |
|------|------|---------|
| **OpenHands** | 全功能代理 | 是（支持执行反馈、自我修复） |
| **SWE-Agent** | 仓库级代码生成 | 是 |
| **OpenHands-Lite / SWE-Agent-Lite** | 单次生成模式 | 否（受限交互步数） |

> 所有方法均在相同 LLM 后端上测试，包括大模型（GPT-4, Claude-3.5, Gemini-Pro）和小模型（Llama-4, Qwen3-Coder）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2）**

| 方法 | 平均 OSS | 平均 BCS |
|------|--------|--------|
| **DEVS-Gen (Ours)** | **0.84**（大模型） / **0.53**（小模型） | **5.35**（大模型） / **5.19**（小模型） |
| OpenHands | 0.91 / 0.55 | 5.67 / 6.25 |
| SWE-Agent | 0.58 / 0.44 | 6.17 / 6.13 |
| OpenHands-Lite | 0.75 / 0.46 | 5.51 / 5.90 |
| SWE-Agent-Lite | 0.31 / 0.36 | 5.77 / 5.08 |

> 注：BCS 分数范围约为 0–7，越高越好；OSS ∈ [0,1]

---

### **与基线方法的对比结果**

#### **有效性（Effectiveness）**
- **DEVS-Gen 在无执行反馈的情况下表现接近全迭代代理**：
  - 大模型下 OSS 达到 0.84，接近 OpenHands 的 0.91。
  - 显著优于所有“Lite”版本，说明其**无需试错即可生成可用代码**。
- **对弱模型更鲁棒**：
  - 在小模型上，SWE-Agent-Lite 的 OSS 仅为 0.13（Llama-4），而 DEVS-Gen 仍达 0.48（GLM-4.7-Flash）。

#### **效率（Efficiency）**
| 方法 | 平均 Token 消耗（log₁₀） | 平均耗时（秒） |
|------|--------------------------|----------------|
| **DEVS-Gen** | **5.35**（大） / **5.19**（小） | **753.3** / **1575.1** |
| OpenHands | 5.67 / 6.25 | 701.6 / 5364.9 |
| SWE-Agent | 6.17 / 6.13 | 4020.1 / 5028.4 |

- **Token 消耗降低约一个数量级**（~6–10× 更少），尤其在小模型上优势显著。
- **避免“死循环”调试**：标准代理常陷入无限修复循环，导致超时（>5000s），而 DEVS-Gen 快速失败或成功。

#### **可扩展性（Scalability）**
- **消融实验显示并行生成带来 ~4.7× 加速**（见 Figure 3）。
- 合成延迟随组件数增长呈 **O(log N)**，而非传统方法的 **O(N)**，得益于 DEVS 的层次分解能力。

---

## **4. 关键结论和发现**

### **主要发现**
1. **模块化生成显著提升可靠性和效率**  
   DEVS 的形式化结构使 LLM 可以“分而治之”，将复杂系统分解为独立可验证的组件，大幅降低单次生成难度。

2. **规范驱动评估是解决“无唯一真值”问题的关键**  
   通过轨迹层面的时间与逻辑约束进行黑盒验证，摆脱对精确代码匹配的依赖，适用于多样化实现。

3. **DEVS-Gen 在资源受限环境下更具实用性**  
   尤其适合部署在边缘设备或实时决策系统中，因其生成速度快、内存占用低、失败可预测。

4. **失败模式更可控**  
   - DEVS-Gen 的错误通常是**局部语义错误**（如状态转移顺序颠倒），易于定位修复。
   - 基线方法则易出现**全局崩溃或静默失败**（如空轨迹、无限循环）。

---

### **局限性**
1. **适用范围有限**  
   主要针对**离散事件主导的系统**，不适用于连续动力学或视觉密集型环境（如机器人控制、图像生成）。

2. **依赖 LLM 对 DEVS 语义的理解**  
   若 LLM 无法准确掌握 `deltint`, `lambdaf`, `hold_in` 等 DEVS 核心概念，可能导致状态同步错误。

3. **当前工具链尚未完全自动化**  
   虽然流程已结构化，但仍需人工定义部分 prompt 模板和验证规则。

---

### **未来工作方向**
1. **支持混合建模（Hybrid DEVS）**  
   将 LLM 嵌入 DEVS 组件内部，作为事件生成器或决策模块，构建“LLM-in-the-loop”的仿真系统（如组织行为模拟）。

2. **自动推导验证规则**  
   利用 LLM 从自然语言规范中自动提取 temporal logic（如 LTL）规则，减少人工标注负担。

3. **集成到 agentic planning pipeline**  
   将该框架作为 LLM agent 的内置 world model 生成器，支持动态创建与切换环境假设。

4. **支持增量更新与差分合成**  
   允许用户修改部分规范后，仅重新生成受影响组件，进一步提升在线适应速度。

--- 

> **总结**：本文提出了一种**规范驱动、模块化、可验证的离散事件世界模型生成框架 DEVS-Gen**，在有效性、效率与可扩展性方面全面超越现有方法，为 agentic systems 提供了可靠的 long-horizon reasoning 基础设施。

</details>

---

### 14. [Hybrid Belief Reinforcement Learning for Efficient Coordinated Spatial Exploration](https://arxiv.org/abs/2603.03595)

**Authors**: Danish Rizvi, David Boyle  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.03595v1  

#### Abstract
Coordinating multiple autonomous agents to explore and serve spatially heterogeneous demand requires jointly learning unknown spatial patterns and planning trajectories that maximize task performance. Pure model-based approaches provide structured uncertainty estimates but lack adaptive policy learn...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Hybrid Belief-Reinforcement Learning for Efficient Coordinated Spatial Exploration 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对**多智能体在空间异质需求下的协同探索任务**中的两个核心挑战：

- **纯模型驱动方法（Model-based）**（如基于高斯过程的方法）虽然能提供良好的不确定性估计，但通常依赖于短视（myopic）规划策略，缺乏从经验中自适应学习策略的能力。
- **深度强化学习（Deep Reinforcement Learning, DRL）** 虽然具备强大的策略优化能力，但在缺乏空间先验知识时样本效率极低，导致训练缓慢且不稳定。

因此，如何在未知空间分布下实现**高效、协调、兼顾探索与利用的多智能体轨迹规划**，是一个尚未很好解决的问题。

---

### **提出了什么新方法或新思路**

作者提出了一种**混合信念-强化学习框架（Hybrid Belief-Reinforcement Learning, HBRL）**，其核心思想是将两种范式的优势结合：

#### **两阶段混合框架（Two-Phase Framework）**
1. **第一阶段（Exploration Phase）**：
   - 使用 **Log-Gaussian Cox Process (LGCP)** 构建空间需求的贝叶斯信念模型，量化不确定性。
   - 采用 **Pathwise Mutual Information (PathMI)** 规划器进行非短视（non-myopic）路径规划，引导智能体主动探索高不确定性区域。
2. **第二阶段（Exploitation Phase）**：
   - 将控制权转移给一个 **Soft Actor-Critic (SAC)** 强化学习智能体。
   - 通过**双通道知识迁移（Dual-Channel Knowledge Transfer）** 实现 warm-start：
     - **信念状态初始化（Belief Initialization）**：将 LGCP 学习到的空间信念作为 SAC 的初始状态表示。
     - **回放缓冲区播种（Replay Buffer Seeding）**：将第一阶段生成的高质量探索轨迹存入 SAC 的 replay buffer，作为专家演示。

#### **关键技术创新**
- **PathMI 非短视规划器**：引入基于方差加权和“陈旧度”（staleness）激励的路径评分机制，鼓励对未访问或信息过时区域的重新访问。
- **方差归一化的重叠惩罚（Variance-Normalized Overlap Penalty）**：
  - 在高不确定性区域允许协作感知（cooperative sensing）；
  - 在已充分探索区域则惩罚冗余覆盖，实现动态协调。
- **时间信念动力学（Temporal Belief Dynamics）**：通过方差增长机制模拟信息老化，激励智能体定期重返旧区域以更新信念。

---

### **相比现有方法的优势**

| 优势维度 | HBRL 的改进 |
|--------|-----------|
| **样本效率** | 显著优于纯 DRL 方法，收敛速度快 38% |
| **最终性能** | 累积奖励提升 10.8% |
| **探索质量** | 更快建立准确的空间信念，后验方差降低更快 |
| **协调能力** | 动态重叠惩罚避免了固定惩罚在高不确定性区域抑制合作的缺陷 |

---

## 2. 核心实验方法和设置

### **使用了哪些数据集**

论文未使用真实世界数据集，而是构建了一个**仿真的多无人机（multi-UAV）无线服务提供任务环境**，用于评估 HBRL 框架。

- **空间需求建模**：用户密度由 3–5 个随机位置的高斯热点（Gaussian hotspots）混合而成，初始未知。
- **动态性**：热点中心随时间进行有界随机游走（bounded random walk），模拟现实中的非平稳性。

---

### **实验设置和评估指标**

#### **仿真参数**
| 参数 | 值 |
|------|----|
| 服务区域 | 2000×2000 m² |
| 网格分辨率 | 20 m |
| UAV 数量 | 2–4 |
| 感知半径 | 100 m |
| 最大步长 | 15 m |
| 任务时长 | 200 步/回合 |
| 总训练回合数 | 200 |

#### **评估指标**
- **累积奖励（Cumulative Reward）**：衡量任务完成质量。
- **后验方差（Posterior Variance）**：衡量空间信念的置信度，越低越好。
- **收敛速度**：达到 95% 最终奖励所需的回合数。
- **消融实验**：分析各组件贡献。

---

### **基线方法对比**

| 基线方法 | 描述 |
|--------|------|
| **Pure LGCP-PathMI** | 仅使用第一阶段的规划方法，无策略学习 |
| **Pure SAC (Cold Start)** | 从零开始训练 SAC，无任何先验知识 |
| **SAC + Belief Transfer Only** | 仅初始化信念状态 |
| **SAC + Buffer Seeding Only** | 仅用 LGCP 轨迹填充 replay buffer |
| **Behavior Cloning (BC) + SAC** | 先行为克隆预训练，再用 SAC 微调 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 方法 | 相比 Pure RL 的奖励提升 | 收敛速度提升 |
|------|------------------------|-------------|
| **HBRL (Dual-Channel)** | **+10.8%** | **快 38%** |
| SAC + Buffer Seeding Only | +7.7% | 快 33% |
| SAC + Belief Transfer Only | +0.6% | 快 3% |

> ✅ **HBRL 不仅最终性能更高，而且学习更稳定、收敛更快。**

---

### **与基线方法的对比结果**

- **Pure LGCP-PathMI**：奖励稳定但无法持续提升，缺乏策略优化。
- **Pure SAC**：初期奖励极低，经历明显“探索谷底”，收敛慢。
- **BC + SAC**：虽平滑过渡，但最终性能与 Pure SAC 相近，说明行为模仿不能替代 off-policy 学习。
- **HBRL**：通过 replay buffer seeding 实现快速启动，并通过 belief transfer 提升泛化能力，最终超越所有基线。

---

### **消融实验结果**

#### **双通道迁移的必要性（Figure 6）**
- **仅信念转移**：几乎无提升（+0.6%），说明知道“哪里不确定”不足以指导行动。
- **仅缓冲区播种**：显著提升（+7.7%），表明专家轨迹对策略学习至关重要。
- **双通道联合使用**：**最佳效果（+10.8%）**，证明两者互补——信念提供上下文，轨迹提供行为模板。

#### **规划视野（Planning Horizon）影响（Figure 8）**
- **L=1（短视）**：表现最差，验证了非短视规划的重要性。
- **L=5**：达到最优，平衡了预测能力和计算开销。
- **L>5**：收益递减甚至下降，因长期预测不可靠。

#### **warm-start 持续时间影响（Figure 7）**
- **10 回合 warm-start**：过渡期出现明显性能下降（transition dip）。
- **20–30 回合**：性能逐步提升。
- **50 回合**：与 30 回合持平，说明过度 warm-start 反而限制 SAC 自主探索。

---

## 4. 关键结论和发现

### **论文的主要发现**

1. **双通道知识迁移显著提升样本效率和最终性能**：
   - Replay buffer seeding 是主导因素，但与 belief initialization 结合可进一步提升。
2. **非短视规划（PathMI）优于短视策略**：
   - 多步展望使智能体能规划全局有益路径。
3. **方差归一化重叠惩罚优于固定惩罚**：
   - 在高不确定性区域允许协作，在低不确定性区域避免冗余，实现了**自适应协调**。
4. **时间信念衰减机制有效维持信念新鲜度**：
   - 防止系统依赖过时观测，促使智能体重访 stale 区域。
5. **压缩状态表示仍保留足够决策信息**：
   - 即使使用全局均值摘要而非完整网格，SAC 仍能稳定学习。

---

### **方法的局限性**

1. **warm-start 为一次性过程**：
   - 当前框架仅在训练初期进行一次知识迁移，未实现信念与策略的持续协同演化。
2. **扩展性受限于联合策略表示**：
   - 随着 UAV 数量增加，动作空间呈线性增长，大规模编队需参数共享或分层架构。
3. **依赖 LGCP 建模假设**：
   - 对非泊松型事件或复杂时空相关性的建模能力有限。
4. **计算复杂度较高**：
   - LGCP 推断和 PathMI 规划带来额外开销，可能不适合实时性要求极高的场景。

---

### **未来工作方向**

1. **连续信念-策略协同学习（Continuous Co-Adaptation）**：
   - 设计在线机制，让 SAC 的探索反馈持续优化 LGCP 信念，反之亦然。
2. **大规模车队扩展**：
   - 引入参数共享、图神经网络（GNN）或多智能体分层结构以支持更大规模部署。
3. **跨领域应用**：
   - 将 HBRL 框架应用于环境监测、精准农业、灾难搜救等其他空间探索任务。
4. **鲁棒性增强**：
   - 研究通信中断、部分智能体失效等异常情况下的容错机制。

---

> 📌 **总结一句话**：  
> HBRL 通过**将 LGCP 的结构化空间先验与 SAC 的自适应策略学习相结合**，并借助**双通道 warm-start** 和**方差归一化协调机制**，实现了在未知空间场中高效、协调的多智能体探索，显著提升了样本效率和任务性能。

</details>

---

### 15. [Hierarchical Inference and Closure Learning via Adaptive Surrogates for ODEs and PDEs](https://arxiv.org/abs/2603.03922)

**Authors**: Pengyu Zhang, Arnaud Vadeboncoeur, Alex Glyn-Davies, Mark Girolami  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.03922v1  

#### Abstract
Inverse problems are the task of calibrating models to match data. They play a pivotal role in diverse engineering applications by allowing practitioners to align models with reality. In many applications, engineers and scientists do not have a complete picture of i) the detailed properties of a sys...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Hierarchical Inference and Closure Learning via Adaptive Surrogates for ODEs and PDEs*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**部分已知物理系统的逆问题**（inverse problems），即在常微分方程（ODEs）和偏微分方程（PDEs）建模中，系统参数未知且动力学中存在未建模的非线性项（如摩擦、湍流、复杂阻尼等）。传统方法通常假设方程形式完全已知，而本文处理的是“已知主干、缺失细节”的现实场景。

具体挑战包括：
- 如何从多个相关但参数不同的物理系统中联合推断个体参数并学习共享的未知动态（closure model）；
- 如何在保证不确定性量化（UQ）的同时高效求解计算密集型的逆问题；
- 如何避免昂贵的数值求解器反复调用带来的计算瓶颈。

---

### 提出的新方法与新思路

作者提出了一种**双层优化框架**（bilevel optimization framework），结合以下三个核心技术：

#### ✅ C1. 联合概率参数估计与确定性闭包学习  
（*Joint probabilistic parameter estimation and deterministic closure learning*）

- **Hierarchical Bayesian Inference**：对来自同一物理家族的 $ K $ 个系统进行联合建模，每个系统有独立参数 $ \theta^{(k)} $，但共享一个群体级超参数 $ \phi $，从而实现跨系统的信息共享与正则化。
- **Deterministic Closure Learning**：使用神经网络（MLP）以最大边际似然估计方式学习未知的非线性闭包函数 $ f^\sigma \approx f $，避免在高维函数空间采样。

> **优势**：兼顾了低维参数的UQ能力与高维闭包学习的计算效率。

#### ✅ C2. 分层参数推理与闭包学习的迭代方案  
（*Iterative scheme for hierarchical parameter inference and closure learning*）

采用交替训练策略：
1. 使用 **ensemble Metropolis-Adjusted Langevin Algorithm (MALA)** 对后验分布 $ p(\theta^{(1:K)}, \phi | y^{(1:K)}) $ 进行采样；
2. 利用采样结果近似梯度 $ \nabla_\sigma \log p(y|\sigma) $，更新闭包模型参数 $ \sigma $。

> **优势**：通过样本驱动的方式将贝叶斯推理与深度学习紧密结合，提升闭包学习的稳定性。

#### ✅ C3. 双层优化加速的贝叶斯反演  
（*Surrogate-accelerated Bayesian inversion via bilevel optimization*）

引入可微分代理模型（surrogate model）替代昂贵的数值求解器（如Runge-Kutta、FEM），构建如下双层结构：

$$
\min_\sigma \mathcal{L}_{\text{CLML}}(\sigma, \beta^*(\sigma)) \quad \text{s.t.} \quad \beta^*(\sigma) = \arg\min_\beta \mathcal{L}_{\text{Surrogate}}(\sigma, \beta)
$$

- 上层：闭包学习（依赖代理模型输出）
- 下层：代理模型训练（依赖当前闭包模型）

> **优势**：显著降低每次MALA采样中的前向/反向传播成本，尤其适用于时间步长密集的动力系统。

#### ✅ C4. 在代表性ODE/PDE上的验证  
测试于三个典型问题：
- 非线性质量-弹簧-阻尼系统（ODE）
- 非线性Darcy流（PDE）
- 广义Burgers方程（PDE）

---

### 相比现有方法的优势

| 方面 | 本文方法 | 现有方法（如B-PINNs、SINDy） |
|------|--------|----------------------------|
| **适用范围** | 多系统联合推理 + 共享闭包学习 | 单系统为主，缺乏群体建模机制 |
| **UQ能力** | 支持完整后验采样（via ensemble MALA） | 多为点估计或近似变分推断 |
| **计算效率** | 引入在线训练的surrogate模型，大幅提速 | 数值求解器重复调用导致慢速收敛 |
| **灵活性** | 可插拔多种surrogate架构（FNO/PINN） | 架构固定，难以扩展 |

---

## 2. 核心实验方法和设置

### 数据集与生成方式
所有实验均基于**合成数据**（synthetic data），由真实参数与噪声观测生成：

| 实验 | 物理系统 | 参数数量 $ K $ | 观测特点 |
|------|----------|------------------|---------|
| Exp 1 | 非线性质量-阻尼系统（ODE） | $ K=5 \sim 100 $ | 时间序列位移观测，稀疏采样，加性高斯噪声 |
| Exp 2 | 二维非线性Darcy流（PDE） | $ K=10,20,30 $ | 空间场观测（60个点），不同系统观测位置不同 |
| Exp 3 | 广义Burgers方程（PDE） | $ K=20 $ | 时空域稀疏观测，随机时间步采样 |

参数 $ \theta^{(k)} $ 从设定的层次先验中采样，确保系统间具有统计相关性。

---

### 实验设置与评估指标

#### ✅ 主要模型对比
| 模型 | 是否使用surrogate | Surrogate类型 | 是否层次化 |
|------|--------------------|---------------|------------|
| Solver（Baseline） | 否 | —— | 是 |
| FNO (Supervised) | 是 | FNO | 是 |
| FNO (Physics-based) | 是 | FNO | 是 |
| PINNs | 是 | PINN | 是/否（消融实验） |
| Non-Hierarchical (PINN) | 是 | PINN | 否 |

#### ✅ 评估指标
| 指标 | 定义 |
|------|------|
| **Parameter Inference Mean MSE** | 所有系统后验均值与真值之间的平均MSE |
| **Coverage (%)** | 真实参数落在±2倍标准差内的比例（反映UQ质量） |
| **Closure MSE** | 学习到的闭包函数 $ f^\sigma(u) $ 与真实 $ f(u) $ 在关键区间上的MSE |
| **Surrogate MSE** | 代理模型预测解与真实数值解之间的MSE |
| **Runtime per Epoch** | 每轮训练耗时（秒） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（以 $ K=20 $ 为例）

| 模型 | Parameter MSE ↓ | Coverage ↑ | Closure MSE ↓ | Surrogate MSE ↓ | Runtime (s) ↓ |
|------|------------------|-----------|----------------|--------------------|----------------|
| Solver | $ 2.70\times10^{-2} $ | 96.7% | 0.50 | — | 0.201 |
| FNO (Supervised) | $ 3.13\times10^{-2} $ | 93.3% | 0.88 | $ 8.44\times10^{-4} $ | 0.712 |
| FNO (Physics-based) | $ 6.77\times10^{-2} $ | 86.7% | 1.38 | $ 4.58\times10^{-3} $ | 0.124 |
| PINNs | $ 6.37\times10^{-2} $ | 90.0% | 1.27 | $ 3.97\times10^{-3} $ | **0.040** |

> 注：Exp 2 和 Exp 3 结果趋势一致。

---

### 与基线方法的对比结果

#### 🔹 参数推断精度
- **Solver 与 Supervised FNO 最优**：二者在参数估计上表现最稳定，MSE最低，coverage接近理论值（约95%）。
- **Physics-based 方法较差**：FNO (physics) 和 PINNs 在小样本下（$ K=5,10 $）出现明显偏差，coverage下降至60%-80%，表明其UQ不可靠。

#### 🔹 闭包学习效果
- **Supervised FNO 表现最佳**：能准确重建立方非线性 $ f(u) = 0.08u + 0.08u^3 $。
- **PINNs 与 Physics-FNO 存在局部失真**：尤其在速度分布稀疏区域泛化能力弱。

#### 🔹 代理模型准确性
- **Supervised FNO 显著优于其他**：Surrogate MSE比PINNs低1–2个数量级。
- **PINNs 因软约束边界产生边缘误差**：见Fig 13。

#### 🔹 计算效率
- **PINNs 最快**：运行时间几乎不随 $ K $ 增长，适合大规模系统。
- **Supervised FNO 最慢**：因需调用数值求解器生成监督标签。
- **传统求解器无法扩展到 $ K>30 $**：内存溢出。

---

### 消融实验结果

#### ✅ 层次化 vs 非层次化（Hierarchical vs Non-Hierarchical）

| 指标 | Hierarchical (PINN) | Non-Hierarchical (PINN) |
|------|----------------------|--------------------------|
| Parameter MSE ($ K=20 $) | $ 6.37\times10^{-2} $ | $ 6.02\times10^{-2} $ |
| Coverage | 90.0% | 86.7% |
| Closure MSE | 1.27 | 1.44 |
| Surrogate MSE | $ 3.97\times10^{-3} $ | $ 5.41\times10^{-3} $ |

> 尽管MSE相近，但**层次化模型提供了更可靠的UQ**（更高coverage）、更快收敛（见trace plot Fig 9）、以及可迁移的群体先验（hyperprior）。

此外，在 $ K=5 $ 极端少样本情况下：
- 非层次化模型崩溃（Coverage仅26.7%）
- 层次化仍保持86.7%

👉 **结论**：层次化建模有效缓解过拟合，增强小样本鲁棒性。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **层次化贝叶斯框架显著提升多系统联合推理的稳定性与泛化能力**，尤其是在数据稀缺时。
2. ✅ **Supervised FNO 是最精确的代理模型**，特别适合需要高质量闭包学习的任务。
3. ✅ **PINNs 是最高效的代理架构**，尽管精度略低，但在大规模系统中具备极佳可扩展性。
4. ✅ **纯物理引导的代理训练（如Physics-FNO）在复杂PDE中不稳定**，尤其当观测稀疏时容易发散。
5. ✅ **双层优化框架实现了前向模拟与逆问题求解的协同进化**，代理模型在MALA探索过程中被自适应精炼。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| ❌ 依赖合成或标注良好的数据 | 当前实验均为理想噪声设定，真实世界数据可能存在系统偏差 |
| ❌ Supervised FNO 训练成本高 | 需要实时调用数值求解器生成标签，限制其在动态环境中的应用 |
| ❌ 闭包表示受限于MLP容量 | 对高度振荡或奇异函数可能表达不足 |
| ❌ 实现复杂度较高 | 需协调MALA采样、双层优化、自动微分等多个模块 |

---

### 未来工作方向

1. **在线状态-参数联合估计**：结合Kalman filtering类方法，实现动态系统的实时推断。
2. **无监督/弱监督代理训练**：减少对数值求解器的依赖，发展完全免求解器的训练范式。
3. **更灵活的闭包表示**：尝试使用Gaussian Process、Neural Operators等更强函数空间建模能力的结构。
4. **应用于真实工程系统**：如结构健康监测、气候建模、地下渗流等实际场景验证。
5. **异构系统建模**：拓展至不同类型但共享部分机理的混合系统族。

---

> 📌 **一句话总结**：  
> 本文提出一种融合**层次贝叶斯推理**、**神经闭包学习**与**双层代理加速**的统一框架，在保证不确定性量化的同时，实现了对多系统中未知参数与动态的高效联合学习，为复杂物理系统的数据驱动建模提供了新范式。

</details>

---

### 16. [Accurate and Efficient Hybrid-Ensemble Atmospheric Data Assimilation in Latent Space with Uncertainty Quantification](https://arxiv.org/abs/2603.04395)

**Authors**: Hang Fan, Juan Nathaniel, Yi Xiao, Ce Bian, Fenghua Ling, Ben Fei, Lei Bai, Pierre Gentine  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.04395v1  

#### Abstract
Data assimilation (DA) combines model forecasts and observations to estimate the optimal state of the atmosphere with its uncertainty, providing initial conditions for weather prediction and reanalyses for climate research. Yet, existing traditional and machine-learning DA methods struggle to achiev...

---

### 17. [ErrorLLM: Modeling SQL Errors for Text-to-SQL Refinement](https://arxiv.org/abs/2603.03742)

**Authors**: Zijin Hong, Hao Chen, Zheng Yuan, Qinggang Zhang, Luyao Zhuang, Qing Liao, Feiran Huang, Yangqiu Song, Xiao Huang  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.03742v1  

#### Abstract
Despite the remarkable performance of large language models (LLMs) in text-to-SQL (SQL generation), correctly producing SQL queries remains challenging during initial generation. The SQL refinement task is subsequently introduced to correct syntactic and semantic errors in generated SQL queries. How...

---

### 18. [Exploring Challenges in Developing Edge-Cloud-Native Applications Across Multiple Business Domains](https://arxiv.org/abs/2603.03738)

**Authors**: Pawissanutt Lertpongrujikorn, Hai Duc Nguyen, Juahn Kwon, Mohsen Amini Salehi  
**Category**: cs.DC  
**Published**: 2026-03-05  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.03738v1  

#### Abstract
As the convergence of cloud computing and advanced networking continues to reshape modern software development, edge-cloud-native paradigms have become essential for enabling scalable, resilient, and agile digital services that depend on high-performance, low-latency, and reliable communication. Thi...

---

### 19. [A framework to reason about consistency and atomicity guarantees in a sparsely-connected, partially-replicated peer-to-peer system](https://arxiv.org/abs/2603.03899)

**Authors**: Sreeja S. Nair, Nicholas E. Marino, Nick Pascucci, Russell Brown, Arthur P. R. Silva, Tim Cummings, Connor M. Power  
**Category**: cs.DC  
**Published**: 2026-03-05  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.03899v1  

#### Abstract
For an offline-first collaborative application to operate in true peer-to-peer fashion, its collaborative features must function even in environments where internet connectivity is limited or unavailable. Each peer may only be interested in a subset of the application data relevant to its workload, ...

---

### 20. [Q-Measure-Learning for Continuous State RL: Efficient Implementation and Convergence](https://arxiv.org/abs/2603.03523)

**Authors**: Shengbo Wang  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.03523v1  

#### Abstract
We study reinforcement learning in infinite-horizon discounted Markov decision processes with continuous state spaces, where data are generated online from a single trajectory under a Markovian behavior policy. To avoid maintaining an infinite-dimensional, function-valued estimate, we propose the no...

---

### 21. [Large-Margin Hyperdimensional Computing: A Learning-Theoretical Perspective](https://arxiv.org/abs/2603.03830)

**Authors**: Nikita Zeulin, Olga Galinina, Ravikumar Balakrishnan, Nageen Himayat, Sergey Andreev  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.03830v1  

#### Abstract
Overparameterized machine learning (ML) methods such as neural networks may be prohibitively resource intensive for devices with limited computational capabilities. Hyperdimensional computing (HDC) is an emerging resource efficient and low-complexity ML method that allows hardware efficient implemen...

---

### 22. [FedCova: Robust Federated Covariance Learning Against Noisy Labels](https://arxiv.org/abs/2603.04062)

**Authors**: Xiangyu Zhong, Xiaojun Yuan, Ying-Jun Angela Zhang  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.04062v1  

#### Abstract
Noisy labels in distributed datasets induce severe local overfitting and consequently compromise the global model in federated learning (FL). Most existing solutions rely on selecting clean devices or aligning with public clean datasets, rather than endowing the model itself with robustness. In this...

---

### 23. [Mozi: Governed Autonomy for Drug Discovery LLM Agents](https://arxiv.org/abs/2603.03655)

**Authors**: He Cao, Siyu Liu, Fan Zhang, Zijing Liu, Hao Li, Bin Feng, Shengyuan Bai, Leqing Chen, Kai Xie, Yu Li  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.03655v1  

#### Abstract
Tool-augmented large language model (LLM) agents promise to unify scientific reasoning with computation, yet their deployment in high-stakes domains like drug discovery is bottlenecked by two critical barriers: unconstrained tool-use governance and poor long-horizon reliability. In dependency-heavy ...

---

### 24. [AI4S-SDS: A Neuro-Symbolic Solvent Design System via Sparse MCTS and Differentiable Physics Alignment](https://arxiv.org/abs/2603.03686)

**Authors**: Jiangyu Chen  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.03686v1  

#### Abstract
Automated design of chemical formulations is a cornerstone of materials science, yet it requires navigating a high-dimensional combinatorial space involving discrete compositional choices and continuous geometric constraints. Existing Large Language Model (LLM) agents face significant challenges in ...

---

### 25. [FINEST: Improving LLM Responses to Sensitive Topics Through Fine-Grained Evaluation](https://arxiv.org/abs/2603.04123)

**Authors**: Juhyun Oh, Nayeon Lee, Chani Jung, Jiho Jin, Junho Myung, Jongwon Lee, Taeui Song, Alice Oh  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.04123v1  

#### Abstract
Large Language Models (LLMs) often generate overly cautious and vague responses on sensitive topics, sacrificing helpfulness for safety. Existing evaluation frameworks lack systematic methods to identify and address specific weaknesses in responses to sensitive topics, making it difficult to improve...

---

### 26. [Lang2Str: Two-Stage Crystal Structure Generation with LLMs and Continuous Flow Models](https://arxiv.org/abs/2603.03946)

**Authors**: Cong Liu, Chengyue Gong, Zhenyu Liu, Jiale Zhao, Yuxuan Zhang  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.03946v1  

#### Abstract
Generative models hold great promise for accelerating material discovery but are often limited by their inflexible single-stage generative process in designing valid and diverse materials. To address this, we propose a two-stage generative framework, Lang2Str, that combines the strengths of large la...

---

### 27. [A Multi-Agent Framework for Interpreting Multivariate Physiological Time Series](https://arxiv.org/abs/2603.04142)

**Authors**: Davide Gabrielli, Paola Velardi, Stefano Faralli, Bardh Prenkaj  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.04142v1  

#### Abstract
Continuous physiological monitoring is central to emergency care, yet deploying trustworthy AI is challenging. While LLMs can translate complex physiological signals into clinical narratives, it is unclear how agentic systems perform relative to zero-shot inference. To address these questions, we pr...

---

### 28. [Asymmetric Goal Drift in Coding Agents Under Value Conflict](https://arxiv.org/abs/2603.03456)

**Authors**: Magnus Saebo, Spencer Gibson, Tyler Crosse, Achyutha Menon, Eyon Jang, Diogo Cruz  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.03456v1  

#### Abstract
Agentic coding agents are increasingly deployed autonomously, at scale, and over long-context horizons. Throughout an agent's lifetime, it must navigate tensions between explicit instructions, learned values, and environmental pressures, often in contexts unseen during training. Prior work on model ...

---

### 29. [MAGE: Meta-Reinforcement Learning for Language Agents toward Strategic Exploration and Exploitation](https://arxiv.org/abs/2603.03680)

**Authors**: Lu Yang, Zelai Xu, Minyang Xie, Jiaxuan Gao, Zhao Shok, Yu Wang, Yi Wu  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.03680v1  

#### Abstract
Large Language Model (LLM) agents have demonstrated remarkable proficiency in learned tasks, yet they often struggle to adapt to non-stationary environments with feedback. While In-Context Learning and external memory offer some flexibility, they fail to internalize the adaptive ability required for...

---

### 30. [Phi-4-reasoning-vision-15B Technical Report](https://arxiv.org/abs/2603.03975)

**Authors**: Jyoti Aneja, Michael Harrison, Neel Joshi, Tyler LaBonte, John Langford, Eduardo Salinas  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.03975v1  

#### Abstract
We present Phi-4-reasoning-vision-15B, a compact open-weight multimodal reasoning model, and share the motivations, design choices, experiments, and learnings that informed its development. Our goal is to contribute practical insight to the research community on building smaller, efficient multimoda...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-19 06:05:37 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [TileMix: Tile-Centric Mixed-Precision Attention for LLM Inference Acceleration](https://arxiv.org/abs/2608.17336)

**Authors**: Hanzhi Zhang, Qiao Zhang, Qinglei Cao, Heng Fan, Yan Huang, Kewei Sha, Yunhe Feng  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2608.17336v1  

#### Abstract
Long-context prefill in large language models (LLMs) incurs substantial computation and memory traffic because dense self-attention computes quadratic query-key scores. Existing methods either use a uniform low-precision path or select token interactions, leaving spatial precision routing over hardw...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《TileMix: Tile-Centric Mixed-Precision Attention for LLM Inference Acceleration》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）在处理长上下文时，**prefill阶段**的密集自注意力机制（dense self-attention）会因计算和内存开销呈平方级增长（$O(L^2)$），成为推理效率的主要瓶颈。现有的优化方法存在以下局限：
- **统一低精度量化（如全INT8）**：虽然提升了算术和内存效率，但会显著损失长上下文下的模型质量。
- **稀疏化注意力（sparsity-based methods）**：通过选择性地计算部分token交互来减少计算量，但改变了注意力图的连通性，可能导致重要信息丢失。

### 提出的新方法与新思路
本文提出了 **TileMix**，一种**基于硬件对齐的tile分组进行混合精度路由的注意力内核**，其核心思想是将数值精度作为在融合注意力（fused attention）执行过程中可执行的空间决策。

#### 主要创新点
1. **Tile-Centric Precision Routing（基于tile的精度路由）**：
    - 将注意力矩阵划分为硬件对齐的二维 **score tiles**。
    - 引入 **tile-group** 概念，即多个相邻的key tile组成一个逻辑组，由一个路由决策位控制。
    - 每个合法的tile-group可以被独立地路由到 **FP16** 或 **INT8** 路径进行计算。

2. **Shared-State Heterogeneous Score Execution（共享状态的异构分数执行）**：
    - FP16 和 INT8 两条路径在计算后，都会经过 **scale alignment**（缩放对齐），然后更新同一个 **online-softmax state**（行最大值、归一化因子、输出累加器）。
    - 这保证了即使在混合精度下，也能维持一个完整的、稠密的注意力流。

3. **Compact and Scalable Kernel-Native Routing（紧凑且可扩展的内核原生路由）**：
    - 使用 **packed bitmasks**（打包的位掩码）来存储路由决策，每个64位的字可以控制多达64个tile-group，实现了常数时间的内层循环查找。
    - 支持 **scalable precision grouping**，使得单个路由词可以在更长的序列上复用，同时保持硬件对齐的计算单元。

4. **实用性强的设计**：
    - 无需重新训练（training-free）。
    - 支持 **Grouped-Query Attention (GQA)**、**变长批处理（variable-length batching）** 和 **INT8 key/value 缓存**。

### 相比现有方法的优势
| 方面 | TileMix | 统一INT8 | 稀疏注意力 |
| :--- | :--- | :--- | :--- |
| **连接性** | ✅ 保留所有合法token交互（稠密） | ✅ 稠密 | ❌ 稀疏，可能丢失信息 |
| **精度灵活性** | ✅ 在tile-group级别混合FP16/INT8 | ❌ 全局统一低精度 | ⚠️ 通常为高精度 |
| **效率** | ✅ 优于FP16吞吐量 | ✅ 最高算术效率 | ✅ 减少计算量 |
| **质量** | ✅ 显著优于统一INT8 | ❌ 长上下文下质量差 | ⚠️ 取决于稀疏模式 |
| **部署** | ✅ 无需重训练 | ✅ 无需重训练 | ⚠️ 可能需要微调 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LongEval**: 用于评估长上下文检索任务（line-level retrieval），以精确匹配准确率（exact-match accuracy）为指标。
- **LV-Eval**: 一个平衡的长上下文基准测试，包含11个英文和中文数据集，覆盖混淆事实插入（CFI）、关键词替换（KPR）等场景，用于评估长上下文问答（question answering）能力。

### 实验设置和评估指标
- **硬件平台**: NVIDIA A100 40GB GPU。
- **模型**: LLaMA 3.2 3B, Qwen-2-7B, Qwen-2.5-7B, Vicuna-7B。
- **批大小**: 8。
- **评估指标**:
    - **质量**: LongEval的精确匹配准确率，LV-Eval的问答准确率。
    - **效率**: Prefill吞吐量（Throughput, K tokens/s）和每秒万亿次操作（TOPS）。
- **端到端计时**: 包含量化、缩放恢复、路由、内存调度和内核启动的全部开销。

### 基线方法对比
- **Dense FP16 Attention**: 全精度参考基线。
- **One (Uniform INT8)**: 所有合法tile-group均路由至INT8，作为低精度效率基线。
- **FlashAttention**: IO感知的FP16融合注意力基线。
- **SageAttention**: 代表性的INT8注意力内核。
- **MInference & FlexPrefill**: 稀疏长上下文推理的基线方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### 1. 模型质量（Long-context QA）
- **Table 1 (LV-Eval)**: 在LLaMA 3.2 3B上，`One`（全INT8）在多数任务上表现最差（低于红色区域）。而 **TileMix** 的混合配置（如 `SpTrans50`, `BigBird50`）能够显著恢复质量，许多配置甚至超过了 `One` 并接近FP16水平（位于蓝色区域）。
- **Table 7 (Qwen 2 7B)**: 结果一致，`SpTrans` 系列的混合精度配置在大多数情况下都优于或等于稀疏基线，并显著优于 `One`。

#### 2. 推理效率（Prefill Throughput）
- **Table 2 (LLaMA 3.2 3B-Instruct)**: 在4k序列长度下：
    - **FlashAttention (FP16)**: 14.33 K tokens/s
    - **One (全INT8)**: 29.80 K tokens/s
    - **TileMix (SpTrans75)**: **31.80 K tokens/s**
- **结论**: TileMix 不仅显著优于FP16基线，甚至在某些配置下超越了统一INT8的吞吐量，实现了 **“质量不降反升，效率大幅提高”** 的效果。

#### 3. 消融实验与分析
- **精度布局的影响 (Figure 5 & Table 8-10)**:
    - 不同的路由模板（`SpTrans`, `BigBird`, `RowRand`）表现不同。
    - `SpTrans` 和 `BigBird` 等结构化布局在高INT8覆盖率下仍能保持较好的鲁棒性。
    - 存在明显的 **layout-task interaction**，即最佳路由策略依赖于具体任务和模型。
- **数值行为分析 (Table 3 & 15-17)**:
    - 输出偏差（deviation）随INT8覆盖率增加而增大。
    - 但偏差的增长并非线性，在5%-10%覆盖率之间有一个拐点，之后趋于平缓，表明该方法具有可控的数值稳定性。
- **路由效率 (Table 5)**:
    - `TileMix quantize-once` 模式通过预量化减少了运行时开销，但增加了约40MiB的HBM占用。
    - `fused on-the-fly` 模式内存占用与FlashAttention相同，但计算开销略高。

---

## 4. 关键结论和发现

### 主要发现
1. **TileMix成功建立了一个可控的精度-效率权衡前沿（accuracy-efficiency frontier）**：它能够在FP16的高质量和统一INT8的高效率之间提供一系列可调节的操作点。
2. **空间精度路由是一种有效的加速维度**：将精度决策从全局提升到tile-group级别的空间粒度，可以在不牺牲稠密连接的前提下，实现细粒度的计算优化。
3. **结构化的路由模板优于随机分配**：`SpTrans` 和 `BigBird` 等受稀疏注意力启发的模板，因其内在的局部性和全局性偏好，能更好地保护关键的注意力交互，从而在同等INT8覆盖率下获得更高的质量。
4. **方法具有良好的通用性和实用性**：支持主流的模型架构特性（GQA, 变长批处理），并提供了INT8 KV缓存接口，便于集成到实际的LLM服务系统中。

### 局限性
- **当前仅针对推理（inference）**：特别是prefill阶段，未涉及训练或decode阶段的全面优化。
- **硬件和格式限制**：目前的实现主要针对NVIDIA A100 GPU上的FP16/INT8组合。其他硬件（如Hopper）或其他精度格式（如FP8）需要适配。
- **静态路由**：路由策略是预先定义的静态模板，未能利用输入内容动态调整精度分配（尽管这带来了确定性和低延迟）。

### 未来工作方向
- **动态精度路由**：探索基于输入内容或注意力权重的动态路由策略，以进一步提升效率和质量。
- **扩展到更多硬件和格式**：支持FP8、NF4等新兴低精度格式，并适配新一代GPU架构。
- **整合到完整推理栈**：将TileMix与decode阶段的优化（如KV缓存压缩）结合，构建端到端的高效推理解决方案。
- **理论分析**：深入研究不同路由布局为何以及如何影响模型的最终输出，建立更坚实的理论基础。

> **总结**: TileMix通过引入 **tile-group级别的混合精度路由**，巧妙地解决了长上下文LLM推理中**效率与质量难以兼得**的根本矛盾。其实验结果证明，这是一种极具前景的系统级优化范式。

</details>

---

### 2. [KernelArc: A Multi-Agent Framework for GPU Kernel Optimization](https://arxiv.org/abs/2608.17071)

**Authors**: Joyjit Kundu, Ben Stoffelen, Kaili Wang, Peter Vrancx, Ludovic Denoyer  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2608.17071v1  

#### Abstract
We present KernelArc, a multi-agent framework for autonomous GPU kernel optimization across heterogeneous workloads. Strategy-specialized agents run in parallel and coordinate through conclusions-only shared memory, a deterministic benchmark guard, and read-only cross-agent state with plateau-trigge...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：KERNELARC: A Multi-Agent Framework for GPU Kernel Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代GPU（如NVIDIA Hopper和Blackwell架构）提供了丰富的高性能原语（如`wgmma.mma_async`、TMA传输、NVFP4格式等），但要充分发挥其性能，需要在**内存层次、寄存器压力、同步机制、数据布局和启动开销**等多个维度进行复杂协调。传统手动调优成本高，而单智能体（single-agent）自动优化系统受限于搜索路径单一、易陷入局部最优、探索广度不足等问题。

KERNELARC旨在解决**GPU内核优化中搜索空间碎片化、易陷入局部收敛**的问题，特别是在多形状（multi-shape）、多算子融合、量化等复杂场景下，如何实现更广泛的探索并突破单轨迹瓶颈。

---

### **提出了什么新方法或新思路**
论文提出 **KERNELARC** —— 一个**策略专业化（strategy-specialized）的多智能体框架**，用于自主优化GPU内核。其核心设计包括：

- **并行策略特化智能体（Parallel Strategy-Specialized Agents）**  
  多个智能体并行运行，每个专注于不同的优化“技能”（skills），如`strat-library`（cuBLASLt配置）、`strat-fusion`（融合策略）、`strat-precision`（精度选择）等，避免所有智能体集中在同一类优化路径上。

- **仅共享结论的记忆机制（Conclusions-Only Shared Memory）**  
  智能体之间仅通过共享内存交换**已验证的优化成果（wins）和失败陷阱（traps）**，而非完整历史或中间状态。记忆具有可配置的保留周期（retention horizon），防止上下文污染。

- **确定性守卫（Deterministic Guard）**  
  所有候选内核必须经过一个外部确定性模块进行**语法检查、正确性验证、基准测试和接受/拒绝决策**，确保评估过程一致且可靠。

- **平台触发的草稿机制（Plateau-Triggered Drafting）**  
  当某智能体连续多次未取得改进时，触发“草稿”机制，强制其切换至完全不同的算法、DSL或数据布局，主动逃离局部平台期。

- **只读跨智能体状态访问（Read-Only Cross-Agent State）**  
  智能体可查看其他智能体的最佳成果（通过内部排行榜），但不能修改，保证独立性同时促进知识迁移。

---

### **相比现有方法的优势**
| 对比维度 | 现有方法（如AuToKERNEL） | KERNELARC |
|--------|--------------------------|-----------|
| 搜索方式 | 单智能体、串行探索 | 多智能体、并行多样化探索 |
| 知识共享 | 无或全量共享历史 | 仅共享**验证后的结论**（wins/traps） |
| 决策控制 | 智能体自评改进 | 由**确定性守卫统一裁决** |
| 路径多样性 | 易陷入局部最优 | 通过技能划分和草稿机制增强多样性 |
| 可扩展性 | 固定剧本深度优化 | 支持跨算子、跨精度、跨架构的广度优化 |

> ✅ **优势总结**：KERNELARC通过**结构化分工 + 受控知识共享 + 外部仲裁**，实现了比单智能体更强的探索能力，在固定预算下达到更高性能上限。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **SOL-ExecBench [2]**：一个面向真实世界GPU内核的基准测试套件，包含235个任务，按复杂度分为四类：
  - **L1**：基础算子（如GQA、RMSNorm）
  - **L2**：融合块（如MoE backward、Decoder Layer）
  - **Quantization (Q)**：显式FP8/NVFP4计算
  - **FlashInfer (FI)**：推理原语（如Paged Attention）

> 实验选取每类代表性任务进行评估，而非全集，受限于资源和访问权限。

---

### **实验设置和评估指标**

#### **硬件环境**
- **NVIDIA H100 SXM**（用于GEMM边界研究）
- **Blackwell B200 GPU**（主实验平台）

#### **评估指标**
- **SOL Score**：归一化的性能得分，定义为各输入形状下延迟几何均值的倒数，**最高为1.0（理论极限）**。
- **Latency Ratio Speedup**：相对于PyTorch参考实现或厂商优化基线的加速比。
- **Candidate Budget**：以提交并通过初步检查的候选内核数量为预算单位（如100 candidates）。

#### **智能体配置**
- 使用多个LLM作为智能体后端（如Claude Opus、Kimi K3等）
- 每个智能体绑定一组“技能”（skills），从预定义技能库中分配
- 共享记忆默认保留每类内核最多16条win/trap记录

#### **基线方法对比**
- **Single-Agent**：私有内存、无跨智能体通信
- **Multi-Agent (bounded/unbounded)**：启用共享记忆，比较有无容量限制的影响
- **Vendor Baselines**：
  - cuBLAS / cuBLASLt
  - cuDNN
  - PyTorch eager mode
- **公开排名榜单**：截至2026年7月30日的[SOL-ExecBench Leaderboard](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard)

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 任务 | 类型 | SOL Score | 排名（2026-07-30） | 相对加速比（vs 基线） |
|------|------|----------|--------------------|------------------------|
| L1-030 | GEMM + Residual | 0.481 → **0.4905**（后续） | **第1 → 第3** | 0.99× (vs cuBLASLt) |
| L2-025 | MoE Backward | 0.535 | **第1** | 1.13× (vs optimized), 291× (vs PyTorch) |
| Q-031 | NVFP4 GQA | 0.988 | **第1** | 43.79× (vs baseline), 1327× (vs PyTorch) |
| FI-014 | Paged Prefill GQA | 0.986 | **第1** | 143.78× (vs baseline), 61,973× (vs PyTorch) |

> 📌 所有提交在**2026年7月30日快照中均排名第一**。

---

### **与基线方法的对比结果**

#### **L1-030（Attention Output Projection）**
- 单智能体搜索停滞于 SOL 0.441
- 多智能体系统突破至 **SOL 0.481**（+9.0%）
- 最终方案：利用 `cuBLASLt Expert-API` 静态配置表 + 输出张量与残差缓冲区别名（aliasing）实现融合

#### **定制GEMM案例（H100, 4096×4096 BF16）**
- 达到 **766 TFLOPS**（理论峰值77%）
- 超越同协议下的cuBLAS基线（742 TFLOPS）**3.2%**
- 成功路径包含：WGMMA/TMA流水线、warp specialization、持久CTA、TMA多播等17步累积优化

#### **跨类别性能提升（图6）**
- 所有任务得分均**超过NVIDIA优化基线（>1.0x）**
- 尤其在**量化（Q）和FlashInfer（FI）任务上表现突出**，表明对低比特和长序列场景高度适配

---

### **消融实验结果（Ablation Study on FI-014）**

| 配置 | 几何平均加速比 | 延迟（ms） | 相对Single提升 |
|------|----------------|-----------|----------------|
| Single (private) | 142.6× | 0.0132 | 1.0× |
| Multi-bounded | 182.7× | 0.0092 | 1.28× |
| Multi-unbounded | **290.8×** | **0.0085** | **2.04×** |

> 🔍 **发现**：
> - 多智能体显著优于单智能体
> - **无限记忆（unbounded）优于有限记忆（bounded）**
> - 早期探索阶段，共享traps可避免重复踩坑
> - 后期改进依赖于跨智能体启发式迁移（如看到更快的fusion策略）

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **共享多智能体搜索能有效拓宽探索范围**  
   在固定候选预算下，多智能体系统能够突破单智能体的平台期，达到更强的主导实现（stronger incumbent）。

2. ✅ **“结论即知识”的共享机制是可行的**  
   仅共享**验证后的wins和traps**即可实现有效的跨智能体学习，无需暴露完整推理历史。

3. ✅ **协调机制的价值取决于内核类型和优化阶段**  
   - 早期：策略多样化最重要
   - 平台期：草稿机制（drafting）帮助跳出局部最优
   - 后期：共享wins引导精细调优

4. ✅ **将生成性推理交给LLM，状态管理交给确定性代码**  
   分离职责提升了系统的稳定性与可复现性。

---

### **方法的局限性**
- ❌ **未估计全任务集胜率**：仅评估了5个代表性任务，不能推广为整体性能优势。
- ❌ **时间敏感排名**：Leaderboard排名随时间变化（如L1-030从第1降至第3），结果具有时效性。
- ❌ **消融实验规模有限**：仅在FI-014上进行，且运行次数少（5次/配置），统计显著性较弱（p≈0.0466）。
- ❌ **难以隔离单个机制影响**：多个协调特性耦合，无法精确归因某一功能的独立贡献。

---

### **未来工作方向**
- 🔮 扩展到更多任务和硬件平台（如TPU、AMD GPU）
- 🔮 自动化技能发现与动态重组（meta-learning over skills）
- 🔮 引入成本感知调度（基于美元/Token消耗的预算分配）
- 🔮 构建长期经验库，支持跨项目知识积累
- 🔮 探索递归自改进（recursive self-improvement）在kernel optimization中的应用

---

> **总结一句话**：  
> **KERNELARC证明了通过结构化多智能体协作，可以在复杂GPU内核优化中实现比单智能体更广、更深的有效搜索，其提交在SOL-ExecBench多个关键任务上达到当时最优水平。**

</details>

---

### 3. [LEGO-RL: Harness-Native Reinforcement Learning for Coding Agents](https://arxiv.org/abs/2608.17393)

**Authors**: Yiming Du, Yuxin Jiang, Tao Yuan, Jianbo Dai, Shaowei Wang, Jierun Chen, Chaofan Tao, Xianzhi Yu, Lifeng Shang, Kam-Fai Wong, Xiaohui Li, Haoli Bai  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2608.17393v1  

#### Abstract
Reinforcement learning for coding agents increasingly relies on long-running agent harnesses to manage tool integration, repository contexts, and execution feedback. However, the native execution environments of these harnesses are inherently misaligned with policy-gradient training: environmental c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LEGO-RL: Harness-Native Reinforcement Learning for Coding Agents

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前在对 **coding agent** 进行 **Reinforcement Learning (RL)** 训练时，面临三大挑战：

- **环境不一致（Train-Inference Discrepancy）**：原生 agent harness（如 OpenHands、Claude Code）在生成轨迹时会进行上下文压缩、历史重写等操作，导致训练器无法准确重建 rollout 时的 token 序列，从而破坏策略梯度计算。
- **执行不可靠（Unreliable Execution）**：沙箱崩溃、依赖错误、奖励劫持（reward hacking）等问题会导致轨迹丢失或奖励信号被污染。
- **可观测性差（Poor Observability）**：异步训练中故障难以定位，缺乏细粒度的轨迹诊断工具。

这些问题使得现有的 RL 框架难以直接应用于复杂的、未经修改的 agent harness。

---

### 提出了什么新方法或新思路

作者提出了 **LEGO-RL**，一个支持 **harness-native** 的强化学习框架，其核心思想是：**不修改现有 agent 的控制流，而是通过外部集成实现可扩展的策略梯度优化**。

该框架建立在三个支柱之上：

#### （1）Faithful Optimization（忠实优化）
- 引入 **in-process LLM proxy**，在模型调用边界捕获原始生成流（token IDs、log-probabilities、response masks），避免因 harness 侧的历史重写导致训练信号失真。
- 对于稀疏 MoE 模型（如 Qwen3.5-35B-A3B），通过 **R3（Rollout Routing Replay）** 技术复现 rollout 时的专家路由决策，确保训练与推理的一致性。

#### （2）Reliable Execution（可靠执行）
- 构建可扩展的沙箱编排系统，支持镜像缓存（image caching）、懒加载（lazy pull via Nydus）、阶段式防御机制。
- 防止 reward hacking，例如：
  - 隐藏 git 历史直到评分阶段；
  - 封装测试依赖，防止网络访问影响评分；
  - 设置防火墙限制 agent 外联。

#### （3）Observable Training（可观测训练）
- 提供 **Live UI** 和 **agent plugin**，支持自动化验证、实时监控和细粒度轨迹诊断。
- 支持 per-instance 任务网格、终止原因分析、组内奖励分布可视化等，帮助快速定位失败根源。

---

### 相比现有方法的优势

| 特性 | LEGO-RL | 其他框架（如 verl、slime、Polar） |
|------|--------|-------------------------------|
| Harness-native 支持 | ✅ 完全黑盒接入 | ❌ 需要适配接口或重写逻辑 |
| Token-level 对齐 | ✅ 通过 in-process proxy 实现 | ❌ 依赖事后重构，易出错 |
| MoE 路由一致性 | ✅ R3 replay | ❌ 忽略路由差异 |
| 沙箱可靠性 | ✅ 图像缓存 + 阶段防御 | ⚠️ 有限支持 |
| 可观测性 | ✅ Live UI + 插件化诊断 | ❌ 日志分散，难追踪 |

> 如 Table 1 所示，LEGO-RL 是目前唯一同时满足所有关键特性的 harness-native RL 框架。

---

## 2. 核心实验方法和设置

### 使用的数据集

- **训练任务池**：基于 **OpenSWE** 构建的 2,699 个任务子集，经过严格筛选：
  - 排除构建失败或评分不可靠的任务；
  - 保留“中等难度”任务（解决率 1–3/4 次），以保证组内 reward variation。
- **评估基准**：标准 **SWE-bench Verified**，确保与训练集在仓库和实例层面完全隔离。

---

### 实验设置

- **模型**：`Qwen3.5-35B-A3B`（sparse MoE 模型）
- **Agent Harnesses**：
  - OpenHands SDK
  - Claude Code
  - OpenCode
- **训练算法**：Group Sequence Policy Optimization (**GSPO**)，使用 group-relative advantage estimation。
- **上下文长度**：200k tokens
- **Rollout 设置**：每任务 8 条 rollout，异步生成
- **训练模式**：fully asynchronous，最大 staleness = 1
- **开源信息**：
  - GitHub: [https://github.com/LegoX/Lego-RL](https://github.com/LegoX/Lego-RL)
  - HuggingFace Model & Dataset: [https://huggingface.co/LegoX/Lego-RL](https://huggingface.co/LegoX/Lego-RL)

---

### 评估指标

- **主指标**：**Solve Rate (%)** on SWE-bench Verified（温度 0.7 下离线验证）
- **辅助指标**：
  - Rollout-training log-probability correlation（衡量忠实性）
  - Trajectory termination profile（执行稳定性）
  - In-batch reward variation（优化有效性）
  - Policy entropy, response length（行为变化）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

在三个不同的 agent harness 上，LEGO-RL 显著提升了 `Qwen3.5-35B-A3B` 的性能：

| Agent Harness | 初始 Solve Rate | LEGO-RL 后 Solve Rate | 提升幅度 |
|---------------|----------------|------------------------|---------|
| OpenHands SDK | 64.0%          | **70.4%**              | +6.4 pp |
| Claude Code   | 62.4%          | **68.2%**              | +5.8 pp |
| OpenCode      | 57.2%          | **66.6%**              | +9.4 pp |

> 所有提升均显著，且 rollout-training 概率相关性保持在 **>0.99**

---

### 与更强基线的对比（Table 2）

引入两个更强基线进行公平比较：

- `Qwen3.6-35B-A3B`：下一代基础模型
- `KAT-Coder-V2.5-Dev`：经过 SFT + RL 微调的先进模型

| Method | OpenHands SDK | Claude Code | OpenCode |
|-------|----------------|-------------|----------|
| Qwen3.5-35B-A3B (baseline) | 64.0% | 62.4% | 57.2% |
| Qwen3.6-35B-A3B | 67.4% | 63.4% | 60.6% |
| KAT-Coder-V2.5-Dev | 67.0% | 66.8% | 64.8% |
| **LEGO-RL-Qwen3.5-35B-A3B** | **70.4%** | **68.2%** | **66.6%** |

> LEGO-RL 训练的旧版模型，在所有 harness 上均超越新版基础模型和专门微调模型，证明其训练效率优势。

---

### 消融实验结果

#### （1）难度筛选的重要性（Figure 6）

对比四种训练池：
- 随机采样
- 高/低难度半区
- 中等难度带（1–3/4 解决）

结果表明：
- 只有“中等难度”和“高难度”池能持续提升性能；
- 随机池几乎无提升；
- 原因：中等难度任务提供足够的组内 reward variation，驱动 group-relative learning。

> **Policy-relative difficulty 是关键设计因素**。

#### （2）Rollout-Training 对齐质量（Table 3）

| Agent Harness | Pearson r | Mean Δlogp (×10⁻³) | p99 Δlogp |
|---------------|-----------|--------------------|-----------|
| OpenHands SDK | 0.9993    | 0.7                | 2.1       |
| Claude Code   | 0.9980    | 0.7                | 2.7       |
| OpenCode      | 0.9993    | 0.6                | 2.0       |

> 表明 trainer 能高度精确地重建 rollout 时的概率，误差极小。

#### （3）R3 路由回放的作用

- 关闭 R3：rollout-training correlation ≈ 0.9946
- 开启 R3：correlation ↑ 至 **0.9993**
- 错误对齐（offset +1）会导致性能急剧下降（Pearson r ↓ 至 0.75）

> 证明 **MoE 路由一致性至关重要**。

---

## 4. 关键结论和发现

### 主要发现

1. **Harness-native RL 是可行且高效的**：无需修改 agent 控制流即可实现高质量策略优化。
2. **忠实性决定训练成败**：token-level 对齐和 MoE 路由一致性是避免训练退化的关键。
3. **执行可靠性直接影响信号质量**：reward hacking 和环境失败会严重污染梯度。
4. **可观测性赋能调试与迭代**：Live UI 可区分 policy degradation 与 infrastructure failure。
5. **任务选择影响巨大**：policy-relative difficulty 决定了组内 reward variation 是否充足。

---

### 方法的局限性

1. **模型泛化未验证**：仅在 `Qwen3.5-35B-A3B` 上测试，其他架构未知。
2. **单次运行**：受限于成本，每个配置只跑一次，缺乏方差估计。
3. **奖励粗糙**：仍使用 binary reward，无法给予中间行为信用（如 error recovery）。
4. **防御非万能**：虽缓解 reward hacking，但不能保证对所有攻击鲁棒。
5. **部署依赖强**：镜像加速效果依赖共享存储和 Nydus 等基础设施。

---

### 未来工作方向

1. **Mixed-Harness Training**：单一策略跨多个 harness 训练，提升通用性。
2. **更丰富的奖励结构**：引入过程奖励（process reward）而非仅终态奖励。
3. **自动化诊断增强**：在 Live UI 中加入 AI 辅助根因分析。
4. **更多 harness 支持**：扩展适配器生态。
5. **开放发布计划**：
   - 持续更新框架
   - 发布训练好的 checkpoints
   - 共享 task indices 和 harness adapters

> LEGO-RL 正在以开源方式持续推进，目标是成为 coding agent RL 的标准基础设施。

--- 

> **一句话总结**：  
> LEGO-RL 成功将复杂的原生 coding agent harness 无缝接入大规模 RL 训练流程，在保持控制流不变的前提下，实现了**高保真、高可靠、高可观测**的策略优化，并在多个 benchmark 上取得显著性能突破。

</details>

---

### 4. [Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements](https://arxiv.org/abs/2608.17310)

**Authors**: Zhi Zheng, Rongsheng Chen, Yunpeng Ba, Zhenkun Wang, Yee Whye Teh, Wee Sun Lee  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.17310v1  

#### Abstract
Reinforcement Learning (RL) has been promising in single-turn LLM fine-tuning. However, long-horizon agentic reasoning introduces increasingly branching interactions and sparse rewards, exposing several limitations of RL: its heavyweight backpropagation-based training stack makes it impractical to f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Agentic RL**（如 Agentic PPO、Agentic GRPO）在对长视野（long-horizon）任务中的 LLM Agent 进行微调时面临三大挑战：
- **高 GPU 内存需求**：需要存储激活值（activations）、优化器状态，并进行反向传播（backpropagation），难以扩展到大模型。
- **信用分配困难**（Credit Assignment）：在稀疏奖励（sparse rewards）和长轨迹下，将最终奖励分解到每一步动作非常困难。
- **模型可扩展性差**：随着模型规模增大，训练成本急剧上升。

### 提出了什么新方法或新思路
作者提出 **Agentic ESOpt**，一种基于 **进化策略**（Evolution Strategies, ES）的全参数微调框架，用于训练长视野 LLM Agent。

其核心思想是：
- 不依赖梯度反向传播，而是通过在当前 LLM 参数周围添加随机扰动（perturbation），生成多个“变体”Agent。
- 在环境中运行这些变体，收集其轨迹并获得标量奖励（scalar reward）。
- 使用奖励加权的方式更新原始参数，实现无梯度优化。

### 相比现有方法的优势
相比 Agentic RL，Agentic ESOpt 具有三大优势：
1. **Model Scalability**（模型可扩展性）  
   只需推理级别的 GPU 内存（inference-level GPU memory），无需存储激活或计算梯度，显著降低内存开销。
   
2. **Flexibility**（灵活性）  
   提供轻量级、黑盒反馈接口，可无缝集成到 **prompt-space evolution** 流程中（如 skill optimization 或 test-time compute）。

3. **Long-Horizon Scalability**（长视野可扩展性）  
   执行的是**轨迹级别**（trajectory-level）的参数归因，不依赖于跨时间步的奖励分解，因此在长视野任务中表现更优。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Sudoku**：控制性多轮环境，最小成功步数 $ H^* \in \{5,10,15\} $，仅提供终端奖励。
- **Math Reasoning**（DAPO & AIME 2026）：ReAct 风格数学推理任务，使用 Python 工具辅助解题。
- **DocVQA**：文档视觉问答任务，结合 OCR 和图像分析工具。
- **WebArena-Lite**：浏览器自动化任务基准，包含 Reddit、GitLab、CMS 等网站操作。
- **Automatic Heuristic Design (AHD)**：测试时自动设计启发式算法的任务，用于验证 test-time compute 场景下的有效性。

### 实验设置和评估指标
| 设置项 | 描述 |
|--------|------|
| **主干模型** | Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B, LLaMA-3.1-8B-Instruct |
| **评估指标** | - 成功率（Success Rate）<br>- ANLS（DocVQA）<br>- Pass@K / Mean@4（Math）<br>- 归一化最优差距（Normalized Optimality Gap） |
| **训练预算** | 控制 FLOPs 或评估次数（evaluations）以公平比较 |
| **perturbation 规模调度** | 引入余弦衰减（cosine decay）机制调节探索-利用平衡 |

### 基线方法对比
- **Agentic PPO**：使用 critic 进行 turn-level 优势估计。
- **Agentic GRPO**：基于组相对优势的策略优化，8-rollout 设置。
- **Vanilla ES**：无 perturbation 衰减的标准 ES 方法。
- **Trace2Skill**：基于轨迹提炼技能的 prompt-space 优化方法。
- **Sample / EoH**：用于 AHD 的采样与进化搜索基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ Sudoku 实验（长视野控制实验）
| 方法 | $ H^*=5 $ | $ H^*=10 $ | $ H^*=15 $ |
|------|-----------|-----------|-----------|
| Agentic PPO | 90.63% | 56.25% | 0.00% |
| Agentic GRPO | 85.42% | 67.71% | 40.63% |
| **Agentic ESOpt (G=32)** | 89.58% | 62.50% | **53.13%** |

> 在 $ H^*=15 $ 上，Agentic ESOpt 比最强 RL 方法高出 **12.50 个百分点**，且 PPO 因稀疏奖励失效。

#### ✅ Math & DocVQA（ReAct 工具使用）
| 方法 | DAPO ↑ | AIME 2026 ↑ | DocVQA ANLS ↑ |
|------|--------|-------------|----------------|
| Qwen3.5-4B Base | 63.0 | 55.8 | 0.3875 |
| Agentic GRPO + No Skill | 68.8 | 58.3 | 0.4627 |
| **Agentic ESOpt + No Skill** | **76.8** | **70.8** | **0.5043** |

> 平均提升 **13.7%**（vs base），优于 Agentic GRPO 的 8.3% 提升。

#### ✅ WebArena-Lite（大规模 Web Agent 微调）
| 方法 | Dataset Avg. ↑ |
|------|----------------|
| Qwen3.5-27B No Skill | 29.47% |
| **Agentic ESOpt + No Skill** | **36.16%** (+6.69%) |
| Trace2Skill | 33.94% |
| **Agentic ESOpt + Trace2Skill** | **36.36%** (+2.42%) |

> 首次实现了 **Qwen3.5-27B 的全参数微调**，仅需推理级内存。

#### ✅ Test-Time Compute：Automatic Heuristic Design (AHD)
| 方法 | 改进数量 / 总比较数 |
|------|--------------------|
| Agentic ESOpt + Sample | 9/12 |
| Agentic ESOpt + EoH | 15/12 |
| **总计** | **28/36** |

> 在固定评估预算下，Agentic ESOpt 显著增强 test-time search，表明其可作为在线自适应模块嵌入现有流程。

### 消融实验结果
| 消融设置 | 结果影响 |
|---------|----------|
| **移除 perturbation 衰减（Vanilla ES）** | 在 $ H^*=15 $ 上从 53.13% 降至 42.71% |
| **设终值 $ \sigma_T = 0 $** | 导致过拟合，最终性能下降至 28.13% |
| **组件 ablation（AHD）** | 移除 reward-weighted update 或 cosine schedule 均导致性能下降 |

> 表明 **cosine decay of $ \sigma $** 对探索-利用平衡至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **ES 更适合长视野 Agent 微调**  
   在长视野、稀疏奖励场景下，ES 的轨迹级参数归因机制天然优于 RL 的逐动作信用分配。

2. **Agentic ESOpt 实现高效全参数微调**  
   仅需推理级 GPU 内存即可完成 Qwen3.5-27B 的微调，在 WebArena 上提升 6.69%。

3. **性能随视野增长而超越 RL**  
   在短视野任务中，RL 仍具竞争力；但在 $ H^* \geq 15 $ 后，Agentic ESOpt 显著领先。

4. **支持灵活组合优化**  
   可与 Trace2Skill、EoH 等 prompt-space 方法结合，实现 **prompt-parameter co-evolution**。

5. **更强模型可能需要更小种群**  
   初步实验证明：对于更强的 backbone（如 9B vs 4B），相同种群大小带来的增益更小，暗示未来可降低 FLOPs 开销。

---

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **引入新超参** | 如 perturbation 规模 $ \sigma $ 及其调度策略，需额外调参。 |
| **环境评估成本高** | 用更多独立环境评估替代 backprop，若环境本身昂贵则不划算。 |
| **持续学习未验证** | 当前实验为 in-setting adaptation，长期在线学习稳定性尚不清楚。 |
| **参数更新密度问题** | 尽管更新幅度集中，但仍为稠密更新，可能影响泛化。 |

---

### 未来工作方向
1. **扩展至更大模型**  
   探索 Agentic ESOpt 在前沿 LLM（如 GPT-5 级别）上的应用潜力。

2. **建立 population scaling law**  
   系统研究种群大小 $ G $ 与模型能力之间的关系，指导高效配置。

3. **量化兼容的 ES 优化**  
   开发适用于量化权重的 perturbation 机制，进一步压缩资源消耗。

4. **紧耦合的 skill-parameter co-evolution**  
   实现技能与参数同步演化，形成闭环自进化 Agent。

5. **动态自适应 $ \sigma $ 调度**  
   设计无需人工设定的自动衰减策略。

---

> **GitHub**: https://github.com/zz1358m/Agentic-ESOpt  
> **通讯作者**: zhi.zheng@u.nus.edu

</details>

---

### 5. [Causal Local States: Scalable Simultaneous Causal Network Inference and Forecasting for Dynamical Systems](https://arxiv.org/abs/2608.17452)

**Authors**: Jonas Braun, Fabian Fischbach, Daniel K\"oglmayr, Sebastian Baur, Christoph R\"ath  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.17452v1  

#### Abstract
Machine learning methods predict many real-world systems with remarkable accuracy, but they are typically treated as black boxes that offer no insight into which interactions drive the dynamics. Causal discovery methods reconstruct the interaction network from observational data, but without regard ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Causal Local States: Scalable Simultaneous Causal Network Inference and Forecasting for Dynamical Systems**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前机器学习模型在预测复杂动力系统（如电力网络、气候系统）时虽然精度高，但通常被视为“黑箱”，无法揭示驱动系统动态的关键变量间交互关系。另一方面，因果发现方法（如Granger causality、Transfer Entropy）虽能推断交互网络，但其目标是结构重建而非预测性能优化，导致所获网络未必适用于高质量预测。

此外，现有结合因果推断与预测的方法依赖**全局超参数**（如统一的因果阈值或固定邻域大小），难以适应异质性强的系统（不同子系统动态差异大），导致网络重建失真。

### **提出的新方法与思路**
本文提出了 **Causal Local States (CLS)** 框架，实现**同时进行因果网络推断与系统预测**，其核心思想如下：

- **局部化建模**：将高维系统分解为多个“局部状态”（local states），每个节点独立构建其最优预测邻域。
- **两阶段变量选择机制**：
  1. **Filter Step**：使用双变量因果度量（如 Transfer Entropy 或 CCM）对候选邻居排序，缩小搜索空间。
  2. **Wrapper Selection + Backward Elimination**：基于预测性能（如 MSE）选择最小且近似最优的邻域，并剔除冗余变量。
- **Granger 因果的操作性实现**：将“一个变量是否提升另一个变量的预测能力”作为因果存在的操作定义，通过 wrapper 模型的预测误差来量化。

### **相比现有方法的优势**
| 方面 | CLS 的优势 |
|------|-----------|
| **可扩展性** | 每个节点独立推理，天然支持并行计算，可扩展至高维系统（如90维以上）。 |
| **无需全局参数** | 避免使用单一因果阈值或邻域大小，适应异质系统。 |
| **解释性增强** | 输出不仅是预测结果，还包括每个节点的因果邻域，提供可解释的交互图谱。 |
| **预测性能优越** | 所得网络专为预测优化，性能媲美甚至接近已知真实网络的模型。 |
| **模块化设计** | 因果度量（filter）与预测模型（wrapper）可替换，适配不同系统特性。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
论文在三个难度递增的动力系统上验证 CLS：

1. **Composite Lorenz63-Rossler Attractor Pairs**
   - 多组非耦合的 Lorenz63 与 Rossler 子系统组成。
   - 维度：最高达 90 维（15 对）。
   - 特点：理想测试环境，真实网络呈块对角结构。

2. **Lorenz96 System**
   - 40 维环形格点上的混沌系统，具有周期性带状耦合结构。
   - 参数：$ N=40, F=5 $，处于强混沌状态。
   - 特点：存在间接连接与延迟效应，更具挑战性。

3. **Higher-order Kuramoto Model of the UK Power Grid**
   - 基于英国高压电网拓扑的真实网络结构（120 节点，165 条边）。
   - 包含二阶与三阶耦合项，模拟实际电力振荡。
   - 特点：真实世界复杂网络，具稀疏性和非线性动力学。

---

### **实验设置与评估指标**

#### **评估指标**
- **Mean Squared Error (MSE)**：衡量短期预测误差。
- **Valid Prediction Steps (VPS)**：闭循环预测中误差首次超过阈值 $ \epsilon = 0.3 $ 的时间步数。
- **Valid Prediction Time (VPT)**：VPS 转换为以最大李雅普诺夫时间（Lyapunov time）为单位的时间长度。
- **True Positive Rate (TPR)** 和 **False Positive Rate (FPR)**：用于评估网络重建准确性。

#### **基线方法对比**
- **Monolithic NGRC**：在整个系统状态向量上训练单一大模型，无变量选择。
- **Ground-truth Adjacency Model**：使用真实网络结构指导的 local-states 预测模型（理想上限）。
- **Global Threshold / Fixed-q Methods**：如 Srinivasan et al. 和 Chu et al. 提出的基于全局因果阈值或固定邻域大小的方法（用于反例说明）。

#### **模型配置**
- **Wrapper 模型**：主要使用 **NGRC**（Next-Generation Reservoir Computing），因其低训练成本、少超参。
- **Filter 度量**：Transfer Entropy (TE) 或 Convergent Cross Mapping (CCM)。
- **消融实验**：比较是否进行 backward elimination，以及不同 wrapper 策略的影响（见 Supplementary Note 3）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **1. Composite Lorenz63-Rossler (N=15, 90维)**
- **网络重建**：
  - TPR > 0.8，FPR 接近 0（经 backward elimination 后）。
  - 成功分离所有 Lorenz 与 Rossler 子系统，恢复块对角结构。
- **预测性能**：
  - **CLS 达到约 1000 VPS**。
  - **标准 NGRC 完全失败**（无法预测），因高维特征空间导致内存溢出。
- **消融实验**：
  - 若仅用 filter（TE 排序）而不做 wrapper 选择，会引入大量跨吸引子假阳性连接。
  - backward elimination 显著去除冗余变量，提升解释性与效率。

#### **2. Lorenz96 (40维)**
- **网络重建**：
  - TPR = 1.000（所有真实父节点均被保留）。
  - FPR 从 0.187（before elimination）降至 0.146（after elimination）。
  - 额外保留的节点多位于核心节点附近（< 距离4），符合 Takens 嵌入理论预期。
- **预测性能**：
  - 使用 CLS 推断网络的模型达到 **5.31 ± 0.64 Lyapunov times** 的 VPT。
  - 与使用真实网络的模型（5.18 ± 0.67）相当，**甚至略有超越**。
- **发现**：最优预测结构 ≠ 真实动力方程图；邻近非直接父节点也能提供有用信息。

#### **3. UK Power Grid (120节点)**
- **网络重建**：
  - 使用 **Kuramoto-NGRC** wrapper：
    - TPR: 98.1% → 96.1%（消除后）
    - FPR: 10.62% → **1.87%**
  - 使用 Sine-NGRC wrapper 效果较差，突显 wrapper 设计的重要性。
- **预测性能**：
  - Kuramoto-NGRC + CLS：长期预测稳定，VPS 与真实网络模型相当。
  - Sine-NGRC + CLS：预测崩溃，表明**wrapper 必须匹配系统特性**。
- **关键观察**：CCM 分数本身无法区分真假邻居（部分假邻居得分高于真邻居），凸显 wrapper 验证必要性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **预测导向的网络推断优于纯结构重建**：  
   以预测性能为准则选出的邻域，即使不完全匹配真实动力方程图，也能实现更优或相当的预测效果。

2. ✅ **局部独立推理可实现高维扩展**：  
   CLS 的 per-node 并行架构使其自然可扩展至高维系统，解决了传统全局模型的维度灾难问题。

3. ✅ **全局阈值在异质系统中失效**：  
   Supplementary Note 1 证明，在 Lorenz-Rossler 系统中，任何单一因果阈值或邻域大小都无法正确重建所有节点的邻域。

4. ✅ **wrapper 消除自影响至关重要**：  
   使用 NGRC inference mode 可避免核心节点自身历史主导预测，从而准确评估邻居贡献（Supplementary Note 3）。

5. ✅ **虚假链接可能有益于预测**：  
   即使是由于隐藏混杂因子（hidden confounder）产生的虚假因果边，也可能携带预测信息，盲目剔除反而降低性能。

---

### **方法的局限性**
- ❌ **尚未应用于真实观测数据**：目前实验均基于仿真时间序列，未在金融、气象等真实噪声数据上测试。
- ❌ **仅进行空间特征选择**：未考虑时间滞后选择（temporal feature selection），即未自动识别最佳时间延迟。
- ❌ **超参数调优困难**：尽管框架并行，但每个局部模型可能需不同超参数，目前采用共享配置，非最优。
- ❌ **wrapper 计算开销较高**：尽管 filter 缩小了搜索空间，wrapper 仍需多次训练模型，计算成本高于纯过滤法。

---

### **未来工作方向**
- 🔮 **拓展至真实世界数据集**：如股票市场、脑电图（EEG）、气候观测等。
- 🔮 **集成时间-空间联合选择**：结合延迟嵌入与变量选择，构建时空因果邻域。
- 🔮 **自动化超参数优化**：开发轻量级策略为每个节点定制超参数。
- 🔮 **引入物理先验约束**：融合领域知识（如距离衰减、守恒律）改进 filter 或 wrapper。
- 🔮 **探索树状或图神经网络 wrapper**：利用内置特征选择能力的模型进一步提升效率。

---

> **总结一句话**：  
> **CLS 成功将因果发现与预测任务统一，提出了一种可扩展、可解释、高性能的框架，为复杂系统的“既看得懂又预测准”提供了新范式。**

</details>

---

### 6. [Optimize Your Sampling: Tuned Diffusion Sampling with Bayesian Optimization](https://arxiv.org/abs/2608.18040)

**Authors**: Travis Zhang, Christian Belardi, Justin Lovelace, Jin Peng Zhou, Saebyeol Shin, Carla P. Gomes, Kilian Q. Weinberger  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.18040v1  

#### Abstract
Sampling from a diffusion model typically requires many forward passes through a large neural network, making generation computationally expensive. While much work has focused on efficient solvers and samplers, comparatively little attention has been paid to selecting the sampling timesteps themselv...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimize Your Sampling: Tuned Diffusion Sampling with Bayesian Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
扩散模型（diffusion models）在生成图像时需要多次前向传播大型神经网络，导致推理成本高昂。虽然已有大量研究聚焦于设计高效的采样器（如 DPM-Solver++），但对**采样时间步长（sampling timesteps）的选择**关注较少。尤其是在低步数（few-step）生成场景下，传统的默认调度（default schedule）质量急剧下降。

现有方法如 **Align Your Steps (AYS)** 虽然优化了采样调度，但其目标是最小化理论推导的代理损失（KLUB），而非直接优化最终的图像质量指标（如 HPS 或 FID），因此可能无法真实反映感知质量。

### 🚀 提出的新方法：Optimize Your Sampling (OYS)
本文提出 **OYS（Optimize Your Sampling）**，将采样参数选择建模为一个**黑盒优化问题（black-box optimization）**，并采用 **贝叶斯优化（Bayesian Optimization, BO）** 直接优化目标评估指标。

#### 核心创新点：
- **端到端优化目标指标**：不同于 AYS 优化 KLUB 这类代理目标，OYS 直接以 HPS、FID、LPIPS、MSE 等实际评价指标为目标进行搜索。
- **无需额外训练**：OYS 不修改模型权重，仅调整采样配置（如 timestep 序列、guidance strength、EMA length 等），适用于任何预训练模型（包括蒸馏模型如 SDXL-Turbo）。
- **通用性强**：支持多种任务（text-to-image、inpainting、inverse image tasks）、多种采样器（Euler、DPM-Solver++）和不同参数化形式（直接优化 timestep 或 parametric schedule）。
- **高效且可迁移**：调优成本一次性支付，后续所有推理均可复用最优配置。

### 🔍 相比现有方法的优势
| 方法 | 是否直接优化目标指标 | 是否需训练 | 支持非微分操作 | 适用模型范围 | 调优成本 |
|------|------------------------|------------|------------------|----------------|-----------|
| AYS | ❌（优化 KLUB 上界） | ❌ | ❌（依赖梯度） | 有限 | 高（~2.4M 生成） |
| DDSS | ❌（优化 KID） | ❌ | ❌（需反向传播） | 小模型可行 | 内存爆炸风险 |
| **OYS (本文)** | ✅（直接优化 HPS/FID/LPIPS/MSE） | ❌ | ✅（黑盒） | 所有模型 | **低（~17K 生成）** |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **COCO Captions**：用于 text-to-image 和 inpainting 实验，训练集调优，验证集评估。
- **DiffusionDB**：用于 FLUX.1-dev 和 QwenImage 的 text-to-image 实验，划分为互斥的调优与测试子集。
- **Prompt Diffusion Dataset (Wang et al., 2023)**：用于 inverse HED / Depth / Segmentation 任务，使用验证集调优，测试集评估。
- **ImageNet-512**：用于 EDM2 家族模型的整体性能比较。

### ⚙️ 实验设置
- **任务类型**：
  - Text-to-image generation
  - Inpainting
  - Inverse image tasks (edge → image, depth → image, seg → image)
  - Parametric schedule tuning on EDM2 family
- **步数预算**：主要集中在 **5-step 或 3-step** 极低预算场景，挑战极限效率。
- **采样器**：涵盖 Euler Discrete、DPM-Solver++、PNDMScheduler 等。
- **调优方式**：
  - 使用 **Gaussian Process + qLogNEI acquisition function** 进行贝叶斯优化。
  - 初始使用 Sobol 采样，之后迭代选择新配置。
  - 每轮评估基于批量生成图像计算目标指标。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **HPS (Human Preference Score)** | 基于 fine-tuned CLIP 模型打分，衡量文本-图像对齐程度；报告平均 HPS 和胜率（win rate） |
| **FID** | 衡量生成图像分布与真实图像的距离（越低越好） |
| **PSNR / LPIPS / MSE** | 分别用于重建任务中的像素级或感知误差评估 |
| **人类主观评测** | 通过众包平台进行双盲 AB 测试，评估图像质量和提示对齐 |

### 🆚 基线方法对比
- **Default Schedule**：原始 log-linear 下采样的默认调度。
- **AYS (Sabour et al., 2024)**：当前最先进的调度优化方法，基于 KLUB 最小化。
- **Grid Search / Hand-designed schedules**：早期启发式方法（如 Karras et al., 2022）。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ Text-to-Image（COCO Captions）
| Model | Method | Steps | Avg HPS | Win Rate vs Default |
|-------|--------|-------|---------|---------------------|
| DeepFloyd | OYS | 5 | **0.252** | **83.99%** |
| SDv1.5 | OYS | 5 | **0.231** | **52.30%** |
| SDXL | OYS | 5 | **0.245** | **70.70%** |
| SDXL-Turbo | OYS | 3 | **0.298** | **83.35%** |

> 💡 **说明**：在 5-step 下，OYS 显著优于 AYS 和 Default，在多个模型上实现 >70% 的偏好胜率。

#### ✅ Inpainting（COCO Captions）
| Model | Method | PSNR↑ | HPS↑ | Win Rate |
|-------|--------|--------|--------|----------|
| SDXL | OYS | **73.47** (+6.9dB) | **0.240** | **93.40%** |
| SDv1.5 | OYS | **67.34** (+2.9dB) | **0.237** | **85.10%** |

> ✅ OYS 在保持未遮罩区域完整性方面显著优于 default，避免“全图被破坏”的失败模式。

#### ✅ Prompt Diffusion（Inverse Tasks）
| Task | Method | PSNR↑ | HPS↑ | Win Rate |
|------|--------|--------|--------|----------|
| Inverse HED | OYS | ↑0.67 | ↑0.009 | **65.43%** |
| Inverse Depth | OYS | ↑0.54 | ↑0.004 | **57.62%** |
| Inverse Seg | OYS | ↓0.37 | ↑0.005 | **58.80%** |

> ✅ 即使 PSNR 微降（如分割任务），HPS 仍提升，表明语义一致性增强。

#### ✅ EDM2 Family（ImageNet-512）
| Model Size | Method | FID↓ | Improvement |
|----------|--------|-------|-------------|
| EDM2-XS ~ EDM2-XXL | OYS | **↓3.57% ~ 7.00%** | 全体一致提升 |

> ✅ OYS 不仅优化 timestep，还能联合优化 `omin`, `omax`, `p`, EMA length, guidance strength，全面超越 grid search 结果。

### 🔬 消融实验与关键观察
- **调度可视化（Figure 5）**：
  - AYS 和 Default 在 log-SNR 空间中接近均匀分布。
  - **OYS 明显将更多步骤分配给高噪声阶段（high-noise timesteps）**，即早期去噪更精细。
- **调优效率（Figure 6）**：
  - 性能在约 **17K 五步生成后饱和**（相当于 ~1.7K 50-step 推理）。
  - 成本比 AYS 报告上限低 **1–2 个数量级**（AYS: ~2.4M 生成 vs OYS: ~17K）。
- **跨指标泛化性**：
  - 即便只优化 HPS，FID、PSNR 等其他指标也同步改善，证明是**真实质量提升**，非过拟合单一指标。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **低步数下调度设计至关重要**：在 5-step 场景中，合理的 timestep 分配可保留高达 **89%–94% 的 50-step 质量**，推理成本降低 **10 倍**。
2. **传统均匀调度并非最优**：OYS 发现应将更多步骤放在**高噪声阶段**（early denoising），这与主流 log-SNR 均匀划分相悖。
3. **直接优化目标指标更有效**：相比优化代理目标（KLUB/KID），直接优化 HPS/FID/LPIPS 更能捕捉人类感知偏好。
4. **OYS 可广泛适配各类模型与任务**：无论是标准扩散模型、蒸馏模型（SDXL-Turbo）、还是 parametric schedule（EDM），均能受益。
5. **调优成本极低且一次完成**：调优开销可摊销至所有后续推理，极具实用价值。

### ⚠️ 局限性
- **每次任务/模型需单独调优**：OYS 不追求“通用最优调度”，而是针对特定任务定制，不能完全免调参。
- **依赖高质量评估指标**：若 HPS/FID 等指标本身有偏，OYS 可能放大偏差。
- **黑盒优化存在收敛不确定性**：尽管实践中表现稳定，理论上 BO 无法保证全局最优。

### 🔮 未来工作方向
- **自动化跨任务迁移**：探索是否可通过元学习等方式减少重复调优。
- **结合架构搜索**：将 OYS 扩展到联合优化模型结构 + 采样策略。
- **实时动态调度**：根据输入 prompt 动态调整 timestep 分布（per-sample adaptation）。
- **扩展至视频/3D 生成**：应用于更高维、更复杂的生成任务。

---

## ✅ 总结
**OYS 是一种简单、通用、高效的方法，通过贝叶斯优化直接优化扩散模型的采样配置，在极低步数下实现了接近高步数的质量水平。它打破了“必须靠复杂采样器或重新训练才能提速”的思维定式，展示了“聪明地选择时间步”这一被忽视维度的巨大潜力。**

> 🔑 **一句话总结**：  
> **不是跑得更快，而是走得更准 —— OYS 教会扩散模型在最关键的几步上“精雕细琢”。**

</details>

---

### 7. [PlanPO: Group Planning-Aware Policy Optimization for Multi-Turn Agentic LLMs](https://arxiv.org/abs/2608.17289)

**Authors**: Dayang Liang, Liyuan He, Xuan Feng, Shuxin Li, Bo An, Yunlong Liu  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.17289v1  

#### Abstract
Group-relative policy optimization has emerged as a key paradigm for training agentic large language models (LLMs) on multi-turn interactive tasks. However, most existing variants fail to distinguish advantages among successful trajectories even when these trajectories differ substantially in their ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PlanPO: Group Planning-Aware Policy Optimization for Multi-Turn Agentic LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在多轮交互任务中，现有的 **Group Relative Policy Optimization (GRPO)** 类方法虽然通过分组采样轨迹进行相对优势估计来优化策略，但存在一个关键瓶颈：**成功轨迹之间的异质性未被有效利用**。

具体而言：
- 所有成功的轨迹（即使路径迂回、响应冗长）通常获得相同的 outcome reward。
- 这导致“**advantage collapse**”——即高质量与低效的成功轨迹无法区分，削弱了训练信号的判别能力。
- 因此，模型可能学会完成任务，但缺乏**规划意识（planning-aware）** 和通用化能力。

### 🚀 提出的新方法：PlanPO
作者提出 **PlanPO (Group Planning-aware Policy Optimization)**，一种简单而有效的强化学习方法，旨在从成功轨迹中挖掘更丰富的监督信号，以学习可泛化的规划能力。

#### 核心思想：**Coarse-to-Fine Advantage Signals**
PlanPO 引入两级长度归一化的优势信号，仅对**成功轨迹**进行条件处理：
1. **Trajectory-Level Advantage (粗粒度)**  
   将 outcome reward 按照轨迹长度（interaction turns）归一化，鼓励更短、更直接的任务解决路径。
   
2. **Turn-Level Advantage (细粒度)**  
   将 reward 按每轮生成的 response token 长度归一化，鼓励简洁、逻辑一致的推理过程。

最终优势为加权组合：
$$
A_{\text{PlanPO}} = A^B + \alpha(k) A^S
$$
其中 $\alpha(k)$ 是随训练步数衰减的权重，初期保留响应长度信号作为精细调节，后期逐渐弱化以防过度惩罚长度。

### 🔍 相比现有方法的优势
| 特性 | GRPO / 其他 Group-RL 方法 | PlanPO |
|------|--------------------------|--------|
| 是否区分高效 vs 冗余成功 | ❌ 否 | ✅ 是 |
| 利用轨迹长度信息 | ❌ 无显式建模 | ✅ 成功条件下归一化 |
| 响应质量控制 | ❌ 忽略生成长度 | ✅ 细粒度响应长度奖励塑形 |
| 泛化能力 | ⚠️ 易过拟合特定模式 | ✅ 学习通用规划策略 |
| 训练开销 | ✅ 低（无 critic） | ✅ 几乎无额外计算成本 |

> ✅ **关键优势**：无需引入额外价值网络或复杂反射机制，在保持 GRPO 轻量架构的同时显著提升性能与泛化性。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在三个具有挑战性的多轮交互基准上评估：

| 数据集 | 描述 |
|-------|------|
| **ALFWorld** | 基于文本的具身环境，模拟家庭场景中的长期推理与决策（如加热物体并放入冰箱），含6类任务。 |
| **WebShop** | 模拟电商网站导航与购买任务，包含约110万商品和1.2万条用户指令。 |
| **SciWorld** | 科学实验环境，代理需操作仪器、执行化学混合等任务，测试科学推理能力。 |

### ⚙️ 实验设置
- **基础模型**：`Qwen2.5-1.5B-Instruct` 和 `Qwen2.5-7B-Instruct`
- **Group Size**：$N=8$
- **Learning Rate**：$1\times10^{-6}$
- **KL Penalty**：$1\times10^{-3}$ (SciWorld), $0.01$ (其他)
- **最大交互轮次**：ALFWorld: 50, WebShop: 15, SciWorld: 20
- **训练步数**：ALFWorld/WebShop: 150步；SciWorld: 200步
- **硬件**：6×H200 + 8×A40 GPU

### 🎯 评估指标
- **Success Rate (%)**：任务完成率（主指标）
- **In-Distribution vs Out-of-Distribution (OOD) Performance**：检验泛化能力
- **平均轨迹长度（Turns）** 和 **平均响应长度（Tokens）**：衡量效率与简洁性

### 🆚 对比的基线方法
#### 在 ALFWorld & WebShop 上：
- **闭源模型**：GPT-4o, Gemini-2.5-Pro
- **Prompting 方法**：ReAct, Reflexion
- **RL 方法**：
  - PPO (with critic)
  - RLOO
  - GRPO
  - EMPG
  - GiGPO (w/ and w/o std)

#### 在 SciWorld 上：
- 多种闭源与开源大模型（Qwen, Llama, DeepSeek 等）
- AgentGym-RL (GRPO-based)
- ScalingInter (group-based RL)

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 方法 | ALFWorld (1.5B) | WebShop (1.5B) | SciWorld (7B) |
|------|------------------|----------------|---------------|
| GRPO | 72.8 ± 3.6 | 75.8 ± 3.5 | 50.5 |
| GiGPO | 86.7 ± 1.7 | 83.1 ± 1.6 | — |
| **PlanPO (ours)** | **91.3 ± 4.1** (+18.5↑) | **86.8 ± 1.5** (+11.0↑) | **68.46** (+17.96↑) |

> ✅ **平均提升 27.2%**（跨三个基准）

#### 🔹 OOD 泛化表现（ALFWorld）
| 方法 | In-Dist Success | Out-Dist Success |
|------|------------------|------------------|
| GRPO | 72.8 ± 3.6 | 70.1 ± 2.5 |
| GiGPO | 86.7 ± 1.7 | 82.4 ± 2.0 |
| **PlanPO** | **91.3 ± 4.1** | **87.1 ± 3.6** |

> 💡 提升 **+17.0 pts OOD**，说明 PlanPO 不是记忆模板，而是学到可迁移的规划能力。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）不同 $\alpha_{\text{init}}$ 设置的影响（图3左）
- 最优设置：$\alpha_{\text{init}} = 0.1$, $\alpha_{\text{final}} = 0.05$
- 若 $\alpha = 0$（仅轨迹级信号）→ 性能下降
- 若 $\alpha > 0.2$ → 过度惩罚长度，损害任务成功率
- 使用 **decaying $\alpha(k)$** 比 constant 更优

#### （2）组件消融（图3中）
| 变体 | ALFWorld 性能 |
|------|----------------|
| Full PlanPO | 91.3 |
| w/o $A^S$ (only trajectory-level) | ↓ 至 ~85 |
| w/o $A^B$ (only turn-level) | ↓↓ 显著退化 |
| Constant $\alpha=0.1$ | 稍差于 decay 版本 |

> ✅ 结论：**两者协同作用，且动态衰减设计至关重要**

#### （3）长度归一化必要性分析（图4）
- **Unconditional length shaping (in PPO)** → 快速崩溃至接近零成功率
- **Success-conditioned only (PPO)** → 改进有限
- **PlanPO (success-conditioned + group-relative)** → 稳步上升，明显优于所有变体

> ✅ 表明：**成功条件 + 分组比较** 是安全利用长度信号的关键

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **成功轨迹并非等价**：即使都成功，其交互路径和推理质量差异巨大，忽略这一异质性会导致学习瓶颈。
2. **多尺度长度归一化是有效信号**：在成功前提下，轨迹长度和响应长度可作为自然的“效率”代理指标。
3. **PlanPO 实现了真正的规划意识**：
   - 平均轨迹长度从 GRPO 的 **26.1** 降至 **13.8**
   - 平均响应长度从 **95.1 tokens** 降至 **56.3 tokens**
   - 且非通过简单截断，而是学习到更紧凑、连贯的推理链。
4. **极低额外开销**：相比 GRPO，新增计算集中在 reward 计算阶段，运行时仅增加约 **2–3秒/迭代**，总体训练时间反而因更快收敛而减少 **12.5%**。

### ⚠️ 局限性
- 当前方法依赖 outcome reward 的稀疏反馈，尚未整合中间过程反馈（如 PRM 或 verifier）。
- 在高度探索性任务（如 SciWorld 中的 Chem-Mix）仍表现不佳，反映模型本身理解能力限制。
- $\alpha(k)$ 衰减策略虽有效，但仍需手动设定初始值，自动化调节有待研究。

### 🔮 未来工作方向
- 探索将 PlanPO 与 **Process Reward Models (PRMs)** 或 **Agentic Verifiers** 结合，进一步细化中间步骤监督。
- 扩展至 **multi-agent** 或 **real-world API calling** 场景。
- 自动化调整 $\alpha(k)$ 动态权重，实现自适应 coarse-to-fine 学习。
- 将该思想应用于 **视觉-语言代理** 或 **robotic planning** 任务。

---

## ✅ 总结一句话
> **PlanPO 通过在成功轨迹内引入“轨迹长度”和“响应长度”的双层级归一化优势信号，在几乎不增加训练成本的前提下，显著提升了多轮 LLM 代理的任务成功率、效率与泛化能力，揭示了“成功轨迹异质性”作为可扩展监督信号的巨大潜力。**

</details>

---

### 8. [MoNe: Modular Neural Memory for Efficient Long Context Inference](https://arxiv.org/abs/2608.17616)

**Authors**: Wonguk Cho, Kyubyung Chae, Tribhuvanesh Orekondy, Sunghyun Park, Hyoungwoo Park, Jeongho Kim, Arash Behboodi, Kyuwoong Hwang, Sungrack Yun  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.17616v1  

#### Abstract
We present MoNe, a lightweight modular neural memory that attaches to any frozen pretrained Transformer to enable long-context inference without retraining. MoNe reads context in fixed-size segments via test-time learning of fast-weight neural memory networks with layer-localized gradient updates; a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：MoNe: Modular Neural Memory for Efficient Long Context Inference**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
当前大语言模型（LLMs）在处理**长上下文推理**（long-context inference）时面临两大挑战：
- **计算成本高**：标准的 In-Context Learning（ICL）依赖自注意力机制，其计算复杂度为 $O(N^2)$，随着上下文长度 $N$ 增长而急剧上升。
- **性能退化**：当上下文超出模型原生窗口（如 32K）时，ICL 性能显著下降；而 RAG 等检索方法难以处理需要跨片段综合推理的任务。

此外，现有高效架构（如 Mamba、TTT）通常需从头训练，无法直接适配已有的预训练 Transformer。

### **提出了什么新方法或新思路**
提出 **MoNe（Modular Neural Memory）**，一种轻量级、模块化的神经记忆插件，可附加到任何**冻结的预训练 Transformer** 上，实现无需微调的高效长上下文推理。

#### **核心设计思想**：
- **两阶段机制**：
  1. **测试时学习（Test-Time Learning）**：将上下文分段（segment）顺序读入，通过层局部梯度更新快速权重（fast-weight）神经记忆网络，编码上下文信息。
  2. **推理阶段（Inference）**：仅用查询 token 生成 memory token，作为 Key/Value 注入原始模型的 self-attention 中，**不再访问原始上下文 token**。
- **固定内存开销**：memory KV 缓存大小恒定（每层 T=512），不随 $N$ 增长。
- **模块化设计**：每个解码器层独立更新 fast weights，无跨层反向传播，支持即插即用。

### **相比现有方法的优势**
| 方法 | 是否需重训练 | 推理成本 | 支持增量扩展 | 多查询复用 | 长上下文泛化 |
|------|---------------|-----------|----------------|--------------|----------------|
| ICL | 否 | $O(N^2)$ | ❌ | ❌ | 差（>32K崩溃） |
| RAG | 否 | $O(K)$ | ✅ | ✅ | 中等（依赖检索质量） |
| TTT/Mamba | 是 | $O(N)$ | ✅ | ✅ | 好但需训练 |
| **MoNe (ours)** | **否** | **$O(1)$ 查询成本** | ✅ | ✅ | **极好（训练于4K，泛化至128K）** |

- **效率优势**：在 128K 上下文下，相比 ICL 减少约 **80% 的总 FLOPs 和峰值 GPU 内存**。
- **参数开销低**：仅增加 **6.4% 参数量**。
- **无需位置插值**：采用 segment-local RoPE，位置索引始终在 [0, T) 范围内，天然支持任意长度外推。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
基于 **RULER benchmark**（Hsieh et al., 2024）中的三个任务，均要求精确检索或聚合整个上下文的信息：
- **S-NIAH**（Single Needle-in-a-Haystack）：在一个填充文本中嵌入一个关键词值对，给定 key 回忆 value。
- **MK-NIAH**（Multi-Key Needle-in-a-Haystack）：插入四个 key-value 对，其中三个为干扰项，测试模型抗干扰能力。
- **Frequent Word Extraction (FWE)**：合成高频词分布，要求识别出最频繁的三个非噪声词。

### **实验设置和评估指标**
- **主干模型**：`Qwen2.5-0.5B-Instruct`（冻结，原生上下文窗口为 32K）。
- **上下文长度**：4K, 8K, 16K, 32K（within window）以及 48K, 64K, 96K, 128K（beyond window）。
- **分段大小**：$T = 512$ tokens/segment。
- **评估指标**：
  - S-NIAH / MK-NIAH：**Substring Exact Match (Sub-EM)**。
  - FWE：**Variable Recall**（三个目标词中有多少出现在输出中）。

### **基线方法对比**
- **ICL**（In-Context Learning）：直接拼接上下文作为 prompt 输入。
- **RAG**：将上下文切分为 128-token chunks，使用 BGE-Large 编码，取与 query 最相似的 top-K chunks 拼接为 prompt，报告最佳 K∈{1,4,8} 结果。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| Task | Method | 32K | 128K |
|------|--------|-----|------|
| **S-NIAH** | ICL | 0.94 | 0.28 |
|           | RAG | 0.93 | 0.89 |
|           | **MoNe** | **1.00** | **0.96** |
| **MK-NIAH** | ICL | 0.93 | 0.00 |
|             | RAG | 0.79 | 0.71 |
|             | **MoNe** | **0.99** | **0.94** |
| **FWE** | ICL | 0.41 | 0.23 |
|         | RAG | 0.61 | 0.60 |
|         | **MoNe** | **1.00** | **0.96** |

> ✅ MoNe 在所有任务和长度上保持接近完美表现，尤其在 >32K 区间远超其他方法。

### **与基线方法的对比结果**
- **ICL**：在超过 32K 后性能“崩溃”（collapse），尤其是在 MK-NIAH 上降至 0。
- **RAG**：虽稳定但存在瓶颈，无法有效整合分散的多跳事实，在 MK-NIAH 上最高仅达 0.71。
- **MoNe**：即使训练只见过 ≤4K 上下文，仍能在 128K 上取得 **0.94~0.96** 的准确率，显示强大外推能力。

### **消融实验结果（见 Figure 4）**

#### **(a) 层覆盖范围（Layer Selection）**
- **MoNe 应用于所有 24 层**：性能最优（128K 下 MK-NIAH 达 0.94）。
- **仅最后 16 层**：性能轻微下降（0.70 @128K）。
- **仅最后 8 层**：严重崩溃（0.02 @128K）。
- **结论**：更多层参与记忆更新有助于捕捉深层语义关联。

#### **(b) 分段大小（Segment Size）**
| Segment Size | 64K (MK-NIAH) | 128K (MK-NIAH) | FLOPs |
|--------------|---------------|----------------|-------|
| 128          | 0.85          | 0.53           | ↓     |
| 256          | 0.91          | 0.75           | ↓     |
| **512 (default)** | **0.98**      | **0.94**       | baseline |

- 更大的 segment size 显著提升长上下文泛化能力，且 FLOPs 节省有限。
- **选择 T=512 是性能与效率的最佳平衡**。

---

## 4. **关键结论和发现**

### **主要发现**
1. **MoNe 实现了真正的“常数时间查询”推理**：推理阶段仅依赖 memory token，查询成本为 $O(1)$，峰值 GPU memory 不随 $N$ 增长。
2. **强大的零样本长度外推能力**：尽管训练最长仅 4K，MoNe 可泛化至 **128K（32倍外推）** 并保持高性能。
3. **即插即用兼容性强**：无需修改 backbone 权重，适用于任何冻结的 Transformer。
4. **高效且实用**：在 128K 下相比 ICL 减少 **~80% FLOPs 和内存占用**，适合移动端、边缘设备部署。

### **方法的局限性**
- 当前验证基于较小模型（0.5B）和控制性任务（RULER），尚未在大规模真实文档 QA 或对话历史等自然场景中验证。
- 快速权重更新依赖 meta-trained LoRA 和 momentum 投影，训练过程本身仍有一定复杂性。
- segment size 和 layer 覆盖需手动设定，缺乏自动化配置机制。

### **未来工作方向**
1. 扩展至更大规模模型（如 7B+）和真实世界任务（multi-document QA、长期对话建模）。
2. 探索不同 LoRA 架构或其他参数高效方法优化 memory 模块效率。
3. 支持动态 segment 划分或 adaptive layer selection，进一步提升灵活性。
4. 将 MoNe 应用于 **on-device personalization** 和 **continual learning** 场景（作者已在参考文献中提及相关应用方向）。

---

> 📌 **总结一句话**：  
> **MoNe 提供了一种无需重训练、即插即用、高效且可扩展的方式，让现有小模型也能胜任超长上下文推理任务，在性能、效率与泛化之间取得了卓越平衡。**

</details>

---

### 9. [Picard Proximal Monte Carlo for Parallel Bayesian Imaging with Score-Based Generative Priors](https://arxiv.org/abs/2608.17666)

**Authors**: Deliang Wei, Evan Bell, Wenhan Guo, Yifan Chen, Yu Sun  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.17666v1  

#### Abstract
Bayesian imaging inverse problems often require sampling from high-dimensional posterior distributions. While recent score-based and diffusion models provide expressive Bayesian priors, their sampling procedures remain inherently sequential and computationally expensive for large-scale imaging appli...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Picard Proximal Monte Carlo for Parallel Bayesian Imaging with Score-Based Generative Priors*

---

## 1. 主要贡献和创新点

### 解决的问题
贝叶斯成像逆问题通常需要从高维后验分布中采样，而现有的基于 **Score-Based Diffusion Models (SDMs)** 的采样方法（如 Langevin Monte Carlo）本质上是**串行迭代**的，计算成本高昂，难以扩展到大规模成像任务（如 3D CT）。此外，许多现有方法缺乏严格的收敛保证，尤其是在非对数凹后验情况下。

### 提出的新方法：**PiX-MC (Picard ProXimal Monte Carlo)**
本文提出了一种全新的时间并行化后验采样框架 **PiX-MC**，其核心思想结合了以下三个关键技术：

1. **Forward-Backward Splitting (FBS)**：
   - 将后验势函数分解为先验项（由 SDM 提供）和似然项。
   - 利用成像问题中常见的高效 **proximal operator** 处理似然项（如 CT 中的投影算子），提升数值稳定性。

2. **Picard Iteration 并行化**：
   - 将整个采样轨迹视为一个固定点问题，通过 **Picard 迭代** 在多个时间步上并行更新。
   - 打破了传统 Langevin 动力学的时间依赖性，实现了**跨时间步的并行计算**，可充分利用多 GPU 资源。

3. **多块（Multi-Block）与退火（Annealed）变体**：
   - **Multi-Block PiX-MC**：将长轨迹划分为多个块，每个块内独立进行 Picard 迭代，降低内存占用，适应有限 GPU 资源。
   - **Annealed PiX-MC (APiX-MC)**：引入噪声水平递减的退火策略，先捕捉全局结构，再细化细节，加速收敛。

### 相比现有方法的优势
| 特性 | 传统 Langevin MC | PiX-MC |
|------|------------------|--------|
| **并行性** | 串行处理时间步 | ✅ 时间步间并行（跨 GPU） |
| **计算效率** | 高墙钟时间（wall-clock time） | ⏱️ 显著降低运行时间 |
| **理论保证** | 通常需强假设（如 log-concavity） | ✅ 支持非对数凹后验、不完美 score 模型 |
| **灵活性** | 固定架构 | ✅ 可结合 proximal likelihood 和 annealing |

---

## 2. 核心实验方法和设置

### 数据集
- **合成高斯后验**：用于验证分布级收敛（闭式解可用）。
- **FastMRI**：10 张 320×320 脑部 MRI 图像，8× 加速采样。
- **DIV2K**：10 张 1024×1024 自然图像，用于去模糊任务。
- **AMOS & LDCT**：腹部 CT 图像，用于 3D 稀疏视图 CT 重建（512×512×80 体积）。
- **CBSD10**：10 张自然图像，用于 Rician 噪声去除。

### 实验设置
- **硬件平台**：单节点，配备 **8 块 NVIDIA RTX PRO 6000 Blackwell GPU**。
- **并行实现**：所有基于 Picard 的方法均采用 **multi-block 方案**，在 8 块 GPU 上并行执行。
- **基线方法**：
  - **L-MC / AL-MC**：标准 Langevin 采样器（串行）。
  - **X-MC / AX-MC**：带 proximal likelihood 的 Langevin 采样器（串行）。
  - **PiL-MC / APiL-MC**：带 Picard 并行化的 Langevin 采样器（无 proximal）。
  - **DPS [16]**、**DAPS [97]**：主流扩散引导基线。

### 评估指标
- **PSNR (Peak Signal-to-Noise Ratio)**：衡量重建质量。
- **SSIM (Structural Similarity Index)**：衡量结构相似性。
- **LPIPS (Learned Perceptual Image Patch Similarity)**：衡量感知质量。
- **MMD² (Maximum Mean Discrepancy)**：在合成实验中衡量采样分布与真实后验的距离。
- **Wall-clock time (运行时间)**：关键指标，体现实际加速效果。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 任务 | 方法 | PSNR (dB) | Wall-clock Time | 加速比 |
|------|------|-----------|----------------|--------|
| **MRI (8×)** | APiX-MC | **31.69** | — | — |
| **Rician Denoising** | APiX-MC | 30.43 | — | — |
| **1024×1024 Deblurring** | APiX-MC | 27.98 | **54 秒** | 9× vs L-MC |
| **3D CT (512×512×80)** | APiX-MC | **33.64** | **35 分钟** | **50× vs L-MC** |

### 与基线方法的对比结果
- **在 MRI 和去模糊任务中**：
  - APiX-MC 在 PSNR 和 SSIM 上优于或等于所有基线（包括 DAPS、DPS）。
  - 视觉上更少伪影，保留更多细节（如边缘、纹理）。
- **在 3D CT 任务中**：
  - APiX-MC 达到 **33.64 dB PSNR**，显著高于 FBP (19.16 dB) 和其他方法。
  - 仅用 **35 分钟** 完成重建，而标准 L-MC 需 **526 分钟**，实现 **约 50× 墙钟时间加速**。
- **在合成高斯后验实验中**：
  - APiX-MC 的 MMD² 下降速度远快于 PiX-MC，且最终值更低，验证了退火的有效性。

### 消融实验结果
1. **Picard 并行化**：
   - 在 deblurring 任务中，使用 8 GPU 时，PiX-MC 相比 X-MC 实现 **2.87× 加速**。
   - 增加 GPU 数量能进一步提升加速比，但受块大小 N 限制。

2. **Proximal Likelihood**：
   - X-MC 比 L-MC 收敛更快、PSNR 更高，说明 proximal 更新比梯度更新更有效。

3. **Annealing**：
   - APiX-MC 比 PiX-MC 收敛更快，最终 PSNR 更高，证明退火能提升采样轨迹质量。

4. **组合效应**：
   - 在 deblurring 任务中，**proximal + annealing + Picard** 三者结合带来 **9× 总体加速**，远超单一机制。

---

## 4. 关键结论和发现

### 主要发现
1. **时间并行化可行且高效**：首次成功将 **Picard iteration** 应用于贝叶斯成像后验采样，实现了跨时间步的并行计算，显著降低墙钟时间。
2. **Proximal Likelihood 提升稳定性**：利用成像问题的结构设计 proximal 算子，比纯梯度法更鲁棒、收敛更快。
3. **多机制可叠加增益**：**Picard 并行化**（降低运行时间）、**annealing**（提升轨迹质量）、**proximal 更新**（提升数值稳定性）三者互补，可联合使用获得巨大加速。
4. **理论与实践一致**：提出的收敛分析支持非对数凹后验、不完美 score 模型等现实条件，实验结果验证了理论预测的几何收敛行为。

### 方法的局限性
- **内存开销**：尽管 multi-block 缓解了问题，但存储整个轨迹仍有一定内存压力。
- **超参数敏感**：Picard 收敛依赖于 contraction factor $q_{\text{mult}} < 1$，需谨慎选择步长 $\gamma$ 和块长度 $N$。
- **适用范围**：依赖于似然项具有高效 proximal operator，对某些复杂前向模型可能受限。

### 未来工作方向
- 探索 **更高阶的 Picard 迭代** 或 **自适应块划分** 以进一步优化资源利用。
- 将 PiX-MC 框架扩展至 **盲逆问题**（blind inverse problems）和 **在线学习** 场景。
- 结合 **low-rank 近似** 或 **量化技术** 降低通信与存储开销，适配更大规模模型。

--- 

> **总结**：PiX-MC 是首个将 **Picard 迭代** 成功应用于贝叶斯成像后验采样的框架，通过**时间并行化**解决了大规模 SDM 采样中的计算瓶颈，在保持高质量重建的同时，实现了高达 **50× 的墙钟时间加速**，为高维医学成像等实际应用提供了强有力的工具。

</details>

---

### 10. [Efficient Resource Optimization for Split Federated Learning](https://arxiv.org/abs/2608.17849)

**Authors**: Wei Wei, Xianhao Chen  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.17849v1  

#### Abstract
Split federated learning (SFL) has emerged as a powerful paradigm for model training at the edge. However, SFL inherently involves discrete decision variables for model splitting and resource allocation, resulting in a challenging mixed-integer problem. Consequently, prior optimization schemes for S...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient Resource Optimization for Split Federated Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于 **Split Federated Learning (SFL)** 中的资源优化难题。SFL 将深度神经网络模型在客户端和边缘服务器之间进行切分，以缓解边缘设备计算能力不足的问题。然而，SFL 的性能高度依赖于**模型切分位置（cut layer）** 和 **资源分配策略**（如 GPU 频率、传输功率），这两者共同决定了训练过程中的 **latency** 和 **energy** 开销。

现有方法存在以下问题：
- 多为启发式算法，缺乏理论保证；
- 联合优化模型切分与资源控制是混合整数非凸问题，求解复杂度高，难以扩展到大规模用户场景。

---

### 🚀 提出的新方法与创新点

#### （1）统一的资源优化框架
提出一个联合优化 **model splitting**、**GPU frequency scaling** 和 **transmission power control** 的统一框架，目标是最小化加权的 **training cost**（即 latency 与 energy 的加权和）。

#### （2）针对模型切分问题：多项式时间最优算法
首次证明并设计了一个可在 **多项式时间内求得全局最优解** 的算法（Algorithm 1），用于解决仅优化模型切分的子问题（P3）。该算法基于平面扫描（plane sweep）与剪枝策略，显著提升了效率。

#### （3）联合优化问题：二维主问题建模 + 近似算法
将联合优化问题转化为一个关于同步变量 $T_1$ 和 $T_2$ 的 **two-dimensional master problem**（P5），然后通过 **uniform grid search** 在其上进行近似求解，提出了具有 $(1+\epsilon)$-approximation guarantee 的高效近似算法（Algorithm 2）。

#### （4）理论保障
- **Theorem 1**: Algorithm 1 可获得模型切分问题的全局最优解。
- **Theorem 3**: Grid approximation 算法满足相对误差界 $\frac{J_{\text{grid}} - J^*}{J^*} \leq \frac{2\Delta}{T_1 + T_2}$，从而可通过选择步长 $\Delta$ 实现任意精度的 $(1+\epsilon)$ 近似。
- **Theorem 4**: 整体算法运行时间为多项式级别 $O((NL/\epsilon^2)\log(1/n))$，适用于大规模系统。

---

### 🔍 相比现有方法的优势

| 方面 | 本文方法 | 现有方法 |
|------|--------|---------|
| **可扩展性** | 支持大规模客户端（数百级） | 多为启发式，难以扩展 |
| **理论保证** | 最优解 / $(1+\epsilon)$-近似保证 | 缺乏理论分析或仅为局部最优 |
| **联合优化能力** | 同时优化切分 + GPU频率 + 功率 | 多数只优化其中一项 |
| **效率** | 多项式时间复杂度，运行速度快 | 求解慢，尤其随客户端数量增长 |

---

## 2. 核心实验方法和设置

### 📊 数据集与模型
- **数据集**：
  - **MNIST**
  - **CIFAR-10**
- **模型架构**：
  - **ResNet-50**
  - **VGG-16**
- **数据分布**：
  - IID（独立同分布）
  - non-IID（每个客户端约70%样本来自单一类别）

### ⚙️ 实验设置
- 客户端数量 $N = 20$（默认）
- Mini-batch size: 32
- Learning rate: 0.001
- 硬件平台：
  - 客户端：NVIDIA Jetson Orin Nano（GPU频率范围 50–625 MHz）
  - 服务器：NVIDIA GeForce RTX 4090（GPU频率范围 210 MHz – 3.135 GHz）
- 通信参数：
  - 上行带宽 $B_U = 100$ MHz，下行 $B_D = 200$ MHz
  - 信道模型：Rayleigh fading，噪声谱密度 $N_0 = 3.98\times10^{-21}$ W/Hz
- Trade-off 参数：$\lambda = 100$

> 所有参数详见 Table II。

---

### 📈 评估指标
- **总成本（Total Cost）**：达到 70% 测试准确率所需的累计 energy-latency 加权开销
- **收敛速度 vs. 成本曲线**
- **运行时间（Running Time）**
- 不同资源配置下的鲁棒性测试（引入 ±10%, ±20% 参数扰动）

---

### 🔁 基线方法对比
共比较五种基准方法：
1. **OC + OGP**（本文提出）：Optimal Cut + Optimal GPU Frequency & Power Control
2. **OC + OG**：Optimal Cut + Optimal GPU Frequency
3. **OC + OP**：Optimal Cut + Optimal Power Control
4. **OC**：仅优化 Cut Layer
5. **ESFL** [17]：经典 SFL 资源管理方案，侧重降低延迟
6. **DSQL** [23]：基于强化学习的动态 Q-learning 方法

---

## 3. 主要实验结果和性能指标

### 📈 性能对比结果（Fig. 3）

- 在所有数据集（MNIST/CIFAR-10）、分布（IID/non-IID）和模型（ResNet-50/VGG-16）下，**OC + OGP 均取得最低训练成本**。
- 成本随准确率上升而增加，接近收敛时增速加快。
- 相比最弱基线（如 DSQL），成本降低可达 **30%-50%**。

---

### 📉 资源敏感性分析（Fig. 4–5）

#### （1）带宽影响（Fig. 4）
- 增加上/下行带宽均能降低通信延迟，从而减少总体成本。
- **OC + OGP 在所有带宽配置下表现最优**，说明其对通信资源变化具有强适应性。

#### （2）GPU 频率影响（Fig. 5）
- **客户端侧**：提高最大允许频率持续降低成本 → 更快完成本地计算。
- **服务器侧**：继续提升频率不再显著降本，甚至可能因能耗上升导致总成本反弹。
  - OC + OGP 自动选择最优而非最大频率，避免浪费。
  - 对比之下，ESFL 等固定使用高频，导致“过度加速”带来的能源惩罚。

---

### 🛡️ 鲁棒性测试（Fig. 6）
- 引入参数不确定性（CV = 10%, 20%）模拟实际环境中硬件/信道波动。
- 所有方法性能下降，但 **OC + OGP 下降幅度最小**，表现出更强的鲁棒性。
- 表明所提方法即使在参数不完全精确的情况下仍能保持竞争力。

---

### 📏 可扩展性与运行效率（Fig. 7）

| 客户端数 | OC + OGP 成本 | 运行时间（秒） |
|----------|----------------|----------------|
| 20       | ~4.0           | ~1.5           |
| 50       | ~4.8           | ~3.8           |
| 100      | ~5.6           | ~7.2           |
| 500      | ~7.0           | ~35            |

- 成本随客户端增多略有上升（受最慢客户端拖累）。
- **运行时间呈近似线性增长**，远优于 ESFL 和 DSQL（指数级增长趋势）。
- 验证了算法的 **polynomial-time scalability**。

---

### 🔬 消融实验（Fig. 8）

#### 极端情况测试：
- **当 $\lambda = 0$（纯节能）**：
  - OC + OGP 显著优于其他方法，因其能精细调节频率与功率。
- **当 $\lambda \to \infty$（纯低延迟）**：
  - OC + OGP 与其他启用资源控制的方法（OC+OG, OC+OP）性能相近，均驱动资源至上限。
  - 但仍优于 DSQL，表明其决策更稳定高效。

> 结论：所提方法能灵活适配不同优化偏好，在 energy-oriented 与 latency-oriented 场景中均表现优异。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **模型切分优化可以被高效精确求解**  
   首次给出 SFL 中 cut layer 选择问题的 **polynomial-time exact algorithm**，解决了长期存在的可扩展性瓶颈。

2. **联合优化可通过二维投影实现高效近似**  
   将复杂的混合整数问题转化为低维 master problem，并结合 grid search 实现 **$(1+\epsilon)$-approximation**，兼顾精度与效率。

3. **资源细粒度控制带来显著收益**  
   同时优化 GPU frequency 与 transmission power 可进一步节省高达 20%-30% 的综合成本，尤其是在 energy-sensitive 场景中。

4. **方法具备良好鲁棒性与可扩展性**  
   即使在参数不确定或客户端规模扩大时，依然保持高性能与快速响应。

---

### ⚠️ 局限性

1. **假设同步执行机制**  
   当前模型基于同步 SFL 轮次，未考虑异步更新或部分参与（partial client participation）场景。

2. **未显式建模模型精度影响**  
   虽然限制 cut layer 范围以保证收敛性（见 Section IV），但未将 test accuracy 直接纳入优化目标。

3. **依赖较理想的信道状态信息（CSI）**  
   实际部署中 CSI 可能动态变化且获取延迟，可能影响控制策略的有效性。

4. **grid search 步长需手动设定**  
   虽然理论上可由 $\epsilon$ 推导 $\Delta$，但在实践中需要权衡精度与计算开销。

---

### 🔮 未来工作方向

1. **扩展至异步 SFL 架构**
   - 支持非均匀训练节奏与动态客户端加入/退出。

2. **引入在线学习机制**
   - 利用 RL 或 bandit 方法应对环境不确定性，实现自适应资源调度。

3. **多目标联合优化**
   - 将模型收敛速度、公平性、隐私保护等纳入统一框架。

4. **跨层协同设计**
   - 结合无线资源调度（如 OFDMA 分配）、批大小调整、梯度压缩等技术，构建端到端优化体系。

5. **硬件原型验证**
   - 在真实边缘设备集群上部署验证，评估实际功耗与延迟表现。

---

> **总结一句话**：  
> 本文建立了首个支持 **大规模、多项式时间、有理论保证** 的 SFL 资源联合优化框架，实现了 energy-latency tradeoff 的高效精准调控，为边缘智能系统的实用化提供了重要工具。

</details>

---

### 11. [DiSCO: Defending text-to-image generation through distribution-guided contrastive prompt optimization](https://arxiv.org/abs/2608.17067)

**Authors**: Tong Zhang, Motasem Alfarra, Carlos Hinojosa, Christos Louizos, Bernard Ghanem  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.17067v1  

#### Abstract
As text-to-image generative models advance, they raise critical safety concerns, particularly the generation of Not-Safe-For-Work (NSFW) content such as violence and nudity, further exacerbated by red-teaming adversarial attacks. Existing defenses predominantly operate under white-box assumptions, r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DiSCO: Defending text-to-image generation through distribution-guided contrastive prompt optimization》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对当前 **text-to-image 生成模型** 在安全方面的核心漏洞——**良性对抗性提示（benign adversarial problem）**。

- **问题定义**：某些文本上看似无害（linguistically safe）的提示词（prompt），由于模型学习到的数据分布偏差，仍会触发生成 **NSFW 内容**（如暴力、裸露等）。
- 现有防御方法存在局限：
  - **White-box 方法**（如 weight editing、cross-attention editing）需要访问模型内部参数，无法应用于闭源或商用模型。
  - **LLM-based prompt rewriting** 虽为黑盒方案，但仅在文本层面进行改写，忽视了模型输出分布的影响，对“良性却有害”的提示无效。

### ✅ 提出的新方法：DiSCO
提出 **DiSCO**（**Distribution-guided Contrastive Suffix Optimization**），一种**训练免费、严格黑盒、即插即用**的 prompt 层级防御模块。

#### 核心思想
将防御视为一个**分布对齐问题**：不修改模型 $G$，而是通过优化输入 prompt，使其引导生成从 unsafe 区域转向 safe 区域。

#### 方法流程
1. **构建参考池（Reference Pools）**：
   - 使用目标模型 $G$ 在非对抗性数据集（I2P）上生成图像。
   - 利用两个分类器（NudeNet 和 Q16）达成共识，构建模型专属的 **safe pool** 和 **unsafe pool**。
2. **对比打分机制（Contrastive Scoring）**：
   - 在 CLIP embedding 空间中，最大化生成图像与 safe pool 的相似度，同时最小化与 unsafe pool 的相似度。
   - 优化目标函数：
     $$
     J(p) = \frac{1}{R}\sum_{x_i \in P_{\text{safe}}} \cos(\phi(x), \phi(x_i)) - \frac{1}{R}\sum_{x_j \in P_{\text{unsafe}}} \cos(\phi(x), \phi(x_j))
     $$
3. **基于 Beam Search 的后缀扩展**：
   - 使用轻量级语言模型（如 LLaMA-3-8B）逐步扩展 prompt 后缀。
   - 每步保留得分最高的 K 个候选，最终选择最优 prompt。

### ✅ 相比现有方法的优势
| 特性 | DiSCO | 白盒方法 | LLM 改写方法 |
|------|-------|----------|--------------|
| 黑盒可用性 | ✅ 完全黑盒 | ❌ 需要梯度/权重 | ✅ |
| 不需重训练 | ✅ | ❌ 通常需要微调 | ✅ |
| 处理良性对抗性提示 | ✅ 有效 | ⚠️ 可能失效 | ❌ 效果差 |
| 即插即用 | ✅ 可附加于任何系统 | ❌ 需集成 | ✅ |
| 维持语义保真度 | ✅ 甚至提升 | ⚠️ 可能模糊 | ✅ |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **I2P Benchmark**：标准的红队攻击（red-teaming）评估数据集，包含多种 NSFW 类别的提示。
- **T2I-RiskyPrompt**（补充）：用于验证 out-of-distribution 泛化能力。

### ⚙️ 实验设置
- **目标模型**：
  - UNet-based：SD v1.4, SD v2.0
  - DiT-based：SD 3, Flux
- **攻击方法**（覆盖多种威胁模型）：
  - **Black-box**：Ring-A-Bell
  - **White-box**：UnlearnDiffAtk, MMA-Diffusion
  - **Defense-targeted**：P4D
- **防御基线**：
  - 推理时干预：SLD-Max, SAFREE
  - 权重编辑：RECE
  - 微调类：ESD
- **DiSCO 参数**：
  - Beam width $K=4$
  - 后缀长度 $T=16$
  - 每步采样参考数 $R=8$

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **ASR ↓**（Attack Success Rate） | 生成 NSFW 图像的比例，越低越好 |
| **CLIP Score ↑** | 文本-图像语义对齐度，衡量保真度 |
| **ImageReward ↑** | 人类偏好感知质量评分 |
| **Semantic Drift** | 通过 prompt embedding 的余弦相似度衡量语义偏移 |

---

## 3. 主要实验结果和性能指标

### 🔢 关键性能数据（来自 Table 1 & 2）
在 **32 种系统-攻击组合、5 个随机种子** 下平均：

| 指标 | 基线（Base） | +DiSCO | 提升幅度 |
|------|-------------|--------|---------|
| **NudeNet ASR** | 23.6% | **2.4%** | ↓ 21.2% |
| **Q16 ASR** | 8.3% | **1.7%** | ↓ 6.6% |

> ✅ DiSCO 将平均攻击成功率降低超过 **90%**。

### 📈 与基线方法对比（代表性结果）
| 模型 | 攻击 | Base ASR (NudeNet) | +DiSCO ASR | 降低 |
|------|------|------------------|-----------|------|
| SD 1.4 | Ring-A-Bell | 84.2% | **7.8%** | ↓ 76.4% |
| SD 3 | MMA-Diffusion | 2.5% | **0.1%** | ↓ 2.4% |
| SLD-Max | Ring-A-Bell | 44.4% | **0.3%** | ↓ 44.1% |
| ESD | Ring-A-Bell | 22.3% | **0.2%** | ↓ 22.1% |

> ✅ DiSCO 显著提升了所有基线防御的鲁棒性，即使是强防御（如 ESD, RECE）也能进一步压缩 ASR 至接近 0。

### 🔍 消融实验结果（Ablation Studies）

#### (1) 对比打分机制必要性（Table 5）
| 打分方式 | 平均 ASR |
|---------|--------|
| Safe-only | 21.6% |
| Unsafe-only | 20.6% |
| **Contrastive (DiSCO)** | **15.6%** |

> ✅ 同时利用 safe 和 unsafe 池的对比信号效果最佳。

#### (2) Beam Search 参数影响（Table 6）
- 最佳配置：$K=4, T=16$
- 更大 $K$ 或 $T$ 可进一步降 ASR，但带来更高计算成本。
- CLIP 分数稳定在 0.27–0.28，说明语义保真度不受影响。

#### (3) 参考池大小敏感性（Table 4）
- 即使使用 **25% 的参考池**（约 670 张图），性能几乎不变。
> ✅ DiSCO 对参考池规模鲁棒，小样本即可有效。

#### (4) 计算开销优化（Table 15）
- 通过减少候选图像的去噪步数（如从 50 → 4 步），可实现 **6.2× 加速**，而 ASR 几乎不变。
> ✅ 可通过低质量渲染加速搜索过程。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **良性对抗性提示是真实且普遍的安全隐患**，仅靠文本改写无法解决。
2. **分布引导的 prompt 优化** 是有效的黑盒防御范式，DiSCO 成功将生成引导向安全区域。
3. **DiSCO 具有极强通用性**：
   - 适用于不同架构（UNet/DiT）、不同防御机制（fine-tuning/inference-time）。
   - 可作为即插即用模块，显著增强现有系统的安全性。
4. **安全性提升的同时，语义质量和感知质量未下降，反而提升**：
   - CLIP 和 ImageReward 均提高，表明 DiSCO 有助于更忠实地还原用户意图。

### ⚠️ 方法的局限性
1. **计算开销较高**：
   - 每个 prompt 需数百次图像生成（默认 256 次），延迟显著。
   - 虽可通过缓存优化，但仍不适合高吞吐场景。
2. **极端语义漂移风险**：
   - 极少数情况下，后缀可能引入训练集中占主导的概念（如知名卡通角色），导致图像被这些概念主导（见 Figure 5）。
3. **依赖外部分类器构建参考池**：
   - 若分类器本身有偏见或盲区，可能影响 DiSCO 的有效性。

### 🔮 未来工作方向
1. **引入显式的语义保持约束**，防止主导概念注入。
2. **开发更高效的搜索策略**，如基于强化学习或进化算法。
3. **多层级安全偏好建模**（见 A.7）：
   - 使用多个安全等级的 reference pool，并赋予不同权重，适应不同应用场景（如儿童 vs 成人模式）。
4. **探索动态自适应参考池更新机制**，以应对模型更新或新攻击类型。

---

> **总结**：DiSCO 提出了一种新颖、实用且高效的黑盒防御框架，成功解决了 text-to-image 模型中“文本安全但视觉有害”的根本挑战，为实际部署中的内容安全提供了强有力的技术支持。

</details>

---

### 12. [Agent Lightning v1.0: Towards Harnessed Agentic RL](https://arxiv.org/abs/2608.17528)

**Authors**: Zhiyuan He, Siwei Zhang, Zhiwen Zhou, Yuqing Yang, Yu Kang, Yuge Zhang, Luna K. Qiu, Tin Yan Tsui, Jiahang Xu, Chong Luo  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.17528v1  

#### Abstract
Modern agents operate inside agent harnesses that manage tools, context, and control flow, making the harness a critical part of the agent system. Our original Agent Lightning introduced a disaggregated architecture that connects arbitrary agents to RL training through an LLM endpoint proxy, an appr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Agent Lightning v1.0: Towards Harnessed Agentic RL

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代 AI Agent 并非独立运行的 LLM，而是依赖于复杂的 **Agent Harness**（代理框架）来管理工具调用、上下文构建、控制流和环境交互。然而，传统的 **agentic RL** 框架要求将整个 agent loop 实现嵌入训练系统中，导致难以复用已有的、独立开发的 agent harness（如 mini-SWE-agent、OpenHands 等），造成集成困难。

此外，在通过代理接口进行 RL 训练时（即“harnessed agentic RL”），会出现一系列新的技术挑战，包括：
- **Retokenization** 导致的 token 序列不连续
- 动态生成训练样本下的 **Advantage Calculation**
- **Loss Normalization** 在动态样本数下的偏差
- 分布式训练后端对可变样本数量的调度难题

这些问题在现有框架中普遍被忽视或处理不当，可能导致训练不稳定甚至失效。

---

### 🚀 提出的新方法与思路
本文提出并系统定义了 **harnessed agentic RL** 范式，并发布了轻量级开源框架 **Agent Lightning v1.0** 来支持该范式的实现。

#### 核心创新点：
1. **首次系统化分析 harnessed agentic RL 的四大挑战**：
   - Retokenization & Sample Merging
   - Advantage Calculation
   - Loss Normalization
   - Training Backend Scheduling

2. **提出合理的解决方案设计原则**：
   - 采用 **best-effort sequence merging** 处理 retokenization，仅当 token-level 前缀完全匹配时才合并。
   - 推荐 **rollout-level advantage calculation** 和 **rollout-level loss normalization**，避免因样本拆分引入统计偏差。
   
3. **发布 Agent Lightning v1.0 框架**：
   - 全系统仅约 **3,500 行代码**，简洁透明。
   - 支持任意外部 agent harness（无需修改其内部逻辑）。
   - 采用 **disaggregated architecture**：训练器与执行器解耦，通过 LLM endpoint proxy 连接。
   - 引入 **declarative rollout abstraction + reconciliation loop** 架构，提升可靠性与可观测性。

4. **引入 Collocated Async RL**：
   - 在同一组 GPU 上交替执行 rollout 与模型更新，兼顾效率与资源利用率。
   - 相比传统 sync RL 提升约 **2x 端到端速度**，同时比 async RL 更节省 GPU 资源。

5. **提供完整可复现的 coding agent 训练流程**：
   - 包括数据清洗、防 reward hacking 措施、训练脚本等。
   - 基于开源模型 Qwen3.5-9B 和数据集 SWE-smith，显著降低复现门槛。

---

### 🔍 相比现有方法的优势
| 特性 | Agent Lightning v1.0 | verl Uni-Agent / AReaL 2.0 / slime |
|------|------------------------|------------------------------------|
| 是否支持任意 harness | ✅ 是（通过 proxy） | ✅ 部分支持（proxy-based） |
| 对 retokenization 的处理 | 显式检测并安全合并 | 缓冲替换（可能引入 off-policy bias） |
| Advantage 计算粒度 | Rollout-level（推荐） | 各有不同（sample 或 rollout） |
| Loss Normalization | Rollout-level（更合理） | 多为 sample-level（易偏倚） |
| 执行后端 | 自托管 Kubernetes（低成本） | 商业沙箱服务（Modal、E2B 等，昂贵） |
| 完整训练 pipeline 开源 | ✅ 提供完整数据+脚本 | ❌ 通常缺少细节或不可复现 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **搜索代理（Search Agent）**：
  - 数据集：HotpotQA（训练集）
  - 测试集：HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle、TriviaQA、Natural Questions（各抽样 50 例）

- **通用指令跟随代理（General Instruction-Following Agent）**：
  - 数据集：Instruction Pre-Training [31]
  - 划分：80% 训练，20% 验证

- **编码代理（Coding Agent）**：
  - 原始数据集：**SWE-smith**（59,136 个任务）
  - 经过严格过滤后保留约 **6,000 训练样本 + 400 测试样本**
  - 最终评估基准：**SWE-bench Verified**

---

### ⚙️ 实验设置
| 项目 | 设置详情 |
|------|---------|
| **Policy Model** | Qwen3.5-9B-Instruct（用于 coding agent）、Llama-3.2-3B-Instruct（search）、Qwen3-4B-Instruct（instruction-following） |
| **Harness** | mini-SWE-agent（coding）、OpenHands（instruction-following） |
| **RL 算法** | GRPO（Generalized Reward Policy Optimization）为主 |
| **Batch Size** | Coding: ~512 tokens / step；Search: 512；Instruction: 8 |
| **Rollouts per Prompt** | 4–8 次 |
| **训练步数** | 最多 208 步（coding agent） |
| **硬件资源** | modest compute（未使用超大规模集群） |

---

### 📊 评估指标
- **主要指标**：
  - Validation Reward（准确率形式报告）
  - SWE-bench Verified Score（解决真实 GitHub issue 的成功率）
- **辅助指标**：
  - Policy Entropy（衡量策略探索程度）
  - Sample merging ratio（平均每个 rollout 产生的训练样本数）

---

### 🔁 基线方法对比
本文未直接与其他框架进行横向性能比较（因多数缺乏完整可复现 pipeline），但通过以下方式体现优势：
- 与自身不同配置进行消融实验（见下节）
- 展示从 **41.8% → 56.4%** 的绝对提升，远超已有公开结果
- 提供完整的训练脚本和数据清洗流程，强调 **reproducibility**

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
| 任务类型 | 模型 | 方法 | SWE-bench Verified / Val Reward | 提升幅度 |
|--------|------|-------|-------------------------------|----------|
| Coding Agent | Qwen3.5-9B | Agent Lightning v1.0 (full) | **56.4%** | ↑14.6% (from 41.8%) |
| Search Agent | Llama-3.2-3B | Agent Lightning v1.0 | Val EM: **41.7%** | ↑16.6% (from 25.1%) |
| Instruction-Following | Qwen3-4B | Agent Lightning v1.0 | Val Reward: **70.2%** | ↑18.3% (from 51.9%) |

> 注：所有实验均基于 **仅 6K 左右训练样本** 和有限计算资源完成。

---

### 🔬 消融实验结果（Ablation Study）
作者在 coding agent 上对比三种设置：

| 设置 | Advantage Level | Loss Normalization | Val Reward | Policy Entropy |
|------|------------------|--------------------|------------|----------------|
| Sample-level Advantage | Sample | Token-mean | 33.1% | 快速上升，不稳定 |
| Rollout-level Advantage | Rollout | Token-mean | 35.0% | 较稳定 |
| **Rollout-level Adv + Norm** | **Rollout** | **Rollout-level token-mean** | **38.2%** | **最稳定，增长缓慢** |

✅ 结论：
- **Rollout-level advantage** 更合理，不受 retokenization 影响。
- 加上 **rollout-level loss normalization** 后进一步提升了稳定性与最终性能。
- 二者结合是更优选择。

---

### 🧩 Rollout 合并行为统计（Figure 10）
- 平均每条 rollout 产生 **2.41 个训练样本**
- 仅有 **36% 的 rollout 能保持为单一样本**
→ 表明动态样本生成是常态，不能忽略其带来的训练影响。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Harnessed Agentic RL 是一个独立且重要的新范式**：
   - 与传统 agentic RL 本质不同，需重新思考 rollout 建模方式。
   - 必须考虑 harness 引入的状态与控制流。

2. **现有框架在关键设计上存在分歧且缺乏理论依据**：
   - 如 advantage 和 loss 的归一化层级直接影响训练稳定性。
   - “简单复用传统 RL 设计”会导致偏差。

3. **Rollout-level 统计优于 Sample-level**：
   - 不应让 retokenization 或 context summarization 等 harness 内部操作改变整体 baseline。
   - 推荐统一使用 **rollout-level advantage 和 loss normalization**。

4. **轻量、模块化架构可行且必要**：
   - Agent Lightning v1.0 证明小规模系统也能支撑复杂 agent RL。
   - 解耦设计提高了灵活性与可维护性。

5. **可复现性至关重要**：
   - 当前领域普遍存在“无法复现”的问题。
   - 本文提供了从数据清洗到训练脚本的全流程开源方案。

---

### ⚠️ 方法的局限性
1. **仍依赖高质量 reward signal**：
   - 若 reward 可被 hack（如通过网络下载答案），仍会影响训练效果（尽管已有防护措施）。
   
2. **未涵盖多智能体协作场景的复杂 credit assignment**：
   - 当前 rollout-level 方法适用于单一任务流，对多分支协作的支持有待扩展。

3. **Kubernetes 依赖带来一定部署门槛**：
   - 尽管避免商业沙箱成本，但需要本地或私有云 K8s 集群支持。

4. **尚未验证更大规模模型上的泛化能力**（如 >70B 模型）。

---

### 🔮 未来工作方向
1. **更精细的 credit assignment 机制**：
   - 在 rollout 内部多个 sample 之间分配 reward，而非简单共享。

2. **支持 streaming / continuous rollouts**：
   - 当前以“完成-失败”状态机为基础，未来可支持长期持续运行 agent。

3. **自动 detection 与 mitigation of reward hacking**：
   - 利用 monitoring system + AI agent 自动识别异常行为并干预。

4. **扩展至 multi-agent coordination 场景**：
   - 支持 sub-agents、handoff、并行执行等高级模式。

5. **进一步优化 collocated async RL 的调度策略**：
   - 动态调整 rollout/update 时间片比例以最大化吞吐。

---

## 总结

📌 **Agent Lightning v1.0** 不只是一个新框架，更是对 **harnessed agentic RL** 范式的系统性反思与工程实践。它揭示了当前 RL for Agent 领域的关键盲区，并提出了兼具理论合理性与工程可行性的解决方案。最重要的是，它推动了该领域的 **可复现研究**，为后续工作奠定了坚实基础。

> 🔗 项目地址：[github.com/microsoft/agent-lightning](https://github.com/microsoft/agent-lightning)  
> 📄 论文链接：[arXiv:2608.17528](https://arxiv.org/abs/2608.17528)

</details>

---

### 13. [Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It](https://arxiv.org/abs/2608.17809)

**Authors**: Quang Minh Nguyen, Luis Frentzen Salim  
**Category**: cs.CL  
**Published**: 2026-08-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.17809v1  

#### Abstract
Humans naturally form and express beliefs in daily communication, e.g., "I think the answer is 3" or "I suppose that's right." Such beliefs inevitably intertwine with fact and knowledge, making the ability to handle them in tandem desirable for large language models (LLMs), as they are increasingly ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文研究了**大型语言模型（LLMs）在处理用户信念时的行为偏差**，特别是当用户的陈述基于错误事实时，LLMs 是否仍能正确识别并确认该“信念”的存在。  
先前研究表明，即使能力强大的 LLMs 在面对“我信 X”这类表达时，若 X 是假命题，则更倾向于否认用户持有此信念（即答“否”），表现出一种系统性弱点。

本论文进一步揭示：这种行为并非普遍存在于所有信念表达方式中，而是**高度依赖于信念所使用的 epistemic expressions（认知动词/表达方式）**，例如 “I think”, “I suppose”, “I vaguely remember” 等。

### 提出了什么新方法或新思路
- **多动词泛化分析**：首次在 18 种不同的 epistemic verbs 上系统评估 LLMs 对信念的追踪能力，涵盖 positive belief、confidence、evidential 和 negation 四类动词家族。
- **任务混淆假设验证**：提出“任务混淆”（task confusion）是根本原因——模型将“Do I believe X?”误解为“Is X true?”，从而进行 fact-checking 而非 belief tracking。
- **干预机制探索**：
  - 通过 prompt instruction 控制是否允许 fact-checking；
  - 首次尝试在 decoding 阶段对注意力机制进行因果干预（attention suppression），以减少模型对错误主张的关注。

### 相比现有方法的优势
- **超越单一动词分析**：相比 Suzgun et al. (2025) 仅使用 “believe”，本文展示了不同表达方式下的巨大差异，揭示了现象的复杂性和语义敏感性。
- **机制解释而非仅描述现象**：不仅报告准确率差距，还通过 chain-of-thought 分析、注意力测量与干预提供了行为背后的认知路径证据。
- **可纠正性证明**：表明该问题是“任务理解错误”而非能力瓶颈，可通过简单指令显著改善性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **KaBLE Benchmark Task 5**（Confirmation of first-person belief）：
  - 包含 1,000 条英文语句，形式为：“I [verb] that X. Do I [verb] that X?”
  - 其中 500 条 X 为事实（factual），500 条 X 为虚假（false）
  - 正确答案始终是 **(A) Yes**，因为问题问的是“你是否相信”，而不是“X 是否为真”。

### 实验设置和评估指标
- **评估对象**：共 10 个 open-weight instruction-tuned LLMs，包括：
  - Gemma 3（4B, 12B, 27B）
  - Llama 3（3.2-3B, 3.1-8B, 3.3-70B）
  - Qwen 3.5（4B, 9B, 27B, 35B-A3B）
- **epistemic verbs**：共 18 个，分为四类：
  1. **Positive belief**: believe, think, suppose, am certain
  2. **Confidence**: am confident, am X% confident (X ∈ {0,20,...,100})
  3. **Evidential**: vaguely remember, was told, read online
  4. **Negation**: don't believe, don't think, don't suppose, seriously doubt
- **Prompt Templates**：
  - 原始模板来自 KaBLE，并替换动词；
  - 添加三种变体来操控 fact-checking 行为：
    - `must fact-check`
    - `may or may not fact-check`
    - `no fact-check`（禁止事实核查）
- **评估指标**：
  - **Confirmation accuracy**：模型回答“是”（A）的比例（黄金标签恒为 A）
  - **Gap**：factual claim 准确率 − false claim 准确率（正值表示模型在假命题上表现差）

### 基线方法对比
- 主要对比的是**原始 prompt 下的表现 vs 不同 instruction 干预后表现**
- 没有传统意义上的“基线模型”，而是以同一模型在不同条件下的表现为比较基准
- 参考前人工作 Suzgun et al. (2025) 中仅用 “believe” 的结果作为起点

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）不同动词下 accuracy gap 差异极大
- 在 18 个动词中，gap 从 **+50% 到 -14% 不等**：
  - 最严重：“I vaguely remember” → gap = +50%
  - 最轻微甚至反转：“I seriously doubt” → gap = -14%（即在 false claim 上反而更准）
- “believe” 的 gap 为 **+28%**，处于中间水平，说明原研究未能捕捉极端情况

#### （2）信心等级的影响（am X% confident）
- 当用户说 “am 100% confident” 且 X 为假时：
  - 模型在 false claim 上准确率下降 **18–48%**
  - factual claim 上变化小于 5%
- 在 0%-80% confidence 区间，accuracy ≥73%，无明显下降

#### （3）instruction 干预效果显著
| Verb Family       | False Claim Acc. (Orig.) | False Claim Acc. (No FC) | 提升幅度 |
|------------------|--------------------------|----------------------------|--------|
| Positive belief  | 48.3%                    | 80.7%                      | +32.4% |
| Evidential       | 33.4%                    | 62.0%                      | +28.6% |
| Confidence       | 57.0%                    | 81.5%                      | +24.5% |
| Negation         | 69.1%                    | 80.6%                      | +11.5% |

> ✅ 所有动词家族在“禁止 fact-check”指令下均提升对 false claims 的确认准确率  
> ❌ “必须 fact-check”则进一步降低准确率

#### （4）reasoning strategy 分析（CoT labeling）
- 使用 DeepSeek-V4-Flash 作为 LLM judge 对 chain-of-thought 进行分类：
  - **Factual verification**：42.9%
  - **Logical affirmation**：28.6%
  - **Direct repetition**：16.8%
  - **No reasoning**：9.4%
  - **Subjectivity deflection**：2.1%

- 在 false claims 上：
  - 显式 fact-check 的 CoT → 准确率仅 **25.1%**
  - 非 fact-check 的 CoT → 准确率达 **75.8%**

> 💡 错误主要集中在那些试图验证事实的推理路径中

#### （5）注意力干预结果（causal intervention）
在局部运行的模型上实施 attention suppression（抑制生成答案时对 claim 的注意力）：

| Model           | Confirmation Acc. (False) ↑ | Verification Acc. (Control Task) ↓ |
|----------------|------------------------------|-------------------------------------|
| llama-3.1-8b   | 54.0% → **74.0%** (+20%)      | 87.0% → 82.0% (-5%)                |
| qwen-3.5-9b    | 33.0% → 37.0%                | ~85% → >85%                        |

> 🔬 成功实现部分恢复（尤其在 llama-3.1-8b 上），且未导致 control task 显著恶化，说明干预具有特异性

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 的 belief confirmation 能力严重受表达方式影响**：
   - 同一个信念，用不同动词表达，会导致模型判断准确率出现巨大波动（gap 跨越 +50% 至 -14%）。
2. **核心问题是 task confusion 而非能力不足**：
   - 模型默认将“Do I believe X?”理解为“Is X true?”，从而执行 fact-checking，覆盖了用户的主观陈述。
   - 这是一种可纠正的任务误解，而非无法克服的能力限制。
3. **fact-checking 是导致错误的主要机制**：
   - Chain-of-thought 分析显示，显式进行 factual verification 的响应在 false claims 上准确率极低。
4. **注意力与错误相关**：
   - 当模型未能正确确认 false belief 时，其在生成答案初期对 claim 内容的 attention 更高。
5. **简单指令即可大幅缓解问题**：
   - 加入“不要 fact-check”的提示，可在多个动词家族上平均提升 25%+ 的准确率。
6. **注意力干预提供因果证据**：
   - 抑制对 claim 的 attention 可部分恢复 belief confirmation 性能，验证了 attention 在其中的因果作用。

### 方法的局限性
- **干预效果模型特定**：attention suppression 仅在一个模型（llama-3.1-8b）上有显著提升，其他模型反应不一，尚无法预测哪些模型会响应。
- **仅限 open-weight 模型**：因果干预需本地部署，无法应用于闭源前沿模型（如 GPT-4、Claude 等）。
- **单轮对话设定**：所有测试基于模板化 prompt，尚未验证在自然多轮对话中的稳定性。
- **计算资源限制**：未测试 frontier-scale 模型，可能存在行为差异。
- **缺乏知识状态关联分析**：未探究模型自身是否知道 X 为假，及其与干预效果的关系。

### 未来工作方向
1. **开发更鲁棒的干预技术**：
   - 探索 persona vectors、steering vectors 等无需修改 attention 的方法。
2. **扩展至多轮对话场景**：
   - 测试 instruction 在持续交互中是否依然有效。
3. **跨数据集验证**：
   - 在其他 belief tracking 或 theory-of-mind 数据集上复现结果。
4. **混合极性项目研究**：
   - 如 “I believe X. Do I not believe X?” 探索否定结构的影响。
5. **结合模型内部知识状态分析**：
   - 将 belief tracking 表现与模型对 X 的已知程度（knowledge strength）关联。
6. **插入干扰上下文**：
   - 在 statement 与 question 之间加入无关信息，测试模型的记忆保持能力。

---

> 📌 **一句话总结**：  
> LLMs 在确认用户信念时的表现不是固定的，而**强烈依赖于信念如何被表述**；它们常因陷入 fact-checking 而忽略用户的主观立场，但这是一种可通过指令或机制干预修正的“任务混淆”，而非不可逾越的能力鸿沟。

</details>

---

### 14. [Data-DPO: Direct Preference Optimization for Target Model Data Selection in LLM Post-Training](https://arxiv.org/abs/2608.16926)

**Authors**: Peng Sun, Yi Yang, Antong Zhang, Chunxiao Li, Yanbo Wang, Dianbo Liu, xin chen, Kai Yu, Lu Chen, Tianfan Fu  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.16926v1  

#### Abstract
Data selection in supervised fine-tuning aims to select a small set of effective samples from large-scale candidate data, reducing training cost while preserving model performance. However, existing methods usually treat data value as a relatively static property, and pay limited attention to the co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Data-DPO: Direct Preference Optimization for Target Model Data Selection in LLM Post-Training**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在大模型的 **Supervised Fine-Tuning (SFT)** 阶段，如何从大规模候选数据中选择一个**小而高效**的训练子集，是降低训练成本、提升模型性能的关键挑战。  
现有方法通常将“数据价值”视为一种静态属性（如样本质量、多样性），忽略了**目标模型自身能力分布对数据适配性的影响**。例如，同一个样本可能对强模型是合适的，但对弱模型则过于困难。

### **提出的新方法/新思路**
作者提出了 **Data-DPO**，一种**面向目标模型的数据选择方法**，其核心思想是：
- **以目标模型为“裁判”**：通过短时的一次性更新（one-step probing）观察目标模型对不同样本的局部反馈（loss 变化），从而衡量样本对该模型的“激活强度”。
- **构建成对偏好（pairwise preference）**：基于激活差异生成样本间的偏好关系（哪个样本更能有效激活模型）。
- **训练轻量级奖励模型（reward model）**：学习这种**目标模型感知的数据偏好**。
- **综合多信号进行最终选择**：结合目标模型偏好、外部质量评分（quality score）和边际多样性（marginal diversity）来构建最终训练子集。

### **相比现有方法的优势**
- **动态而非静态**：不依赖预定义的质量或表示空间相似性，而是根据目标模型的实际训练反馈动态判断数据价值。
- **避免代理模型偏差**：不同于使用小规模 proxy model 来估计重要性的方法（如 LESS、COINCIDE），Data-DPO 直接在原始目标模型上探测，避免因微调导致的能力分布偏移。
- **更稳定有效的子集构建**：通过融合偏好、质量和多样性，防止所选数据过度集中于易学样本，保证训练分布的广度和深度。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Vision-Flan**：涵盖广泛视觉任务的人工标注视觉指令数据集，用于评估通用多模态理解能力。
- **LLaVA-CoT**：专注于图像推理的指令数据集，每个回答包含结构化推理过程（summary, caption, reasoning, conclusion），用于评估多步推理能力。

### **实验设置和评估指标**
- **目标模型**：
  - Vision-Flan → `LLaVA-V1.5-7B`
  - LLaVA-CoT → `Llama-3.2-11B-Vision-Instruct`
- **数据预算**：5%、10%、15%
- **评估方式**：
  - 在多个下游基准（如 GQA、MME、MMBench、MATH-Vision 等）上测试微调后模型的表现。
  - 报告 **Average Relative Performance (ARP)**：
    $$
    \text{ARP} = \frac{\text{Subset Data Performance}}{\text{Full Data Performance}} \times 100
    $$
    ARP > 100 表示小数据子集优于全量数据训练。

### **基线方法对比**
涵盖了主流数据选择范式：
| 类型 | 方法 |
|------|------|
| **随机选择** | Random |
| **重要性估计** | EL2N, ScalSelect |
| **多样性驱动** | PRISM, SemDeDup, D2Pruning |
| **质量+多样性结合** | XMAS, COINCIDE, CLIP Score |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **Vision-Flan 结果**
| 数据预算 | Data-DPO (ARP) | 最佳基线 | 全数据训练 (ARP) |
|---------|----------------|----------|------------------|
| 5%      | **100.76**     | 99.45 (SemDeDup) | 100.00 |
| 10%     | **102.63**     | 100.49 (D2Prune) | 100.00 |
| 15%     | **102.70**     | 100.35 (SemDeDup) | 100.00 |

> ✅ 所有预算下均**超越全数据训练性能**

#### **LLaVA-CoT 结果**
| 数据预算 | Data-DPO (ARP) | 最佳基线 | 全数据训练 (ARP) |
|---------|----------------|----------|------------------|
| 5%      | **102.73**     | 99.13 (EL2N) | 100.00 |
| 10%     | **103.93**     | 99.79 (XMAS) | 100.00 |
| 15%     | 101.31         | **102.46 (EL2N)** | 100.00 |

> ✅ 在低预算下显著领先；15% 时略低于 EL2N，但仍优于其他所有方法且高于全数据训练。

### **与基线方法的对比结果**
- 在 Vision-Flan 上，Data-DPO 在所有预算下均为**最佳方法**，相对最强基线提升达 **2.35 ARP**。
- 在 LLaVA-CoT 上，5%/10% 预算下大幅领先（+3.60 和 +4.14 ARP），仅在 15% 被 EL2N 微弱反超。
- 即使在极端低预算（5%）下，也能实现**优于全数据训练**的效果。

### **消融实验结果**
#### **(1) 代理模型 vs 原始目标模型**
- 若用在 5% 数据上微调过的 checkpoint 替代原始目标模型进行 probing：
  - Vision-Flan 上 ARP 下降至 93.17（5%）、97.76（10%）、101.72（15%）
  - 小预算下性能损失明显。
> 🔍 **结论**：proxy model 的偏好信号存在分布偏移，不能准确反映原模型的真实需求。

#### **(2) 选择奖励函数的组成分析**
在 Vision-Flan（10% 预算）上的 ablation：
| 组合 | ARP |
|------|-----|
| 仅 DPO 偏好 | 92.33 |
| DPO + Marginal Diversity | 99.26 |
| Quality + Diversity | 100.21 |
| DPO + Quality | 93.25 |
| **DPO + Quality + Diversity (完整版)** | **102.63** |

> 🔍 **结论**：三者互补。仅靠偏好会导致过拟合简单样本；加入多样性和质量约束可显著提升泛化性。

#### **(3) 对质量评分和嵌入源的鲁棒性**
- 更换质量评分模型（Qwen3-VL-4B → LLaVA-OneVision-4B）：
  - ARP 仍保持在 100+，最高达 103.76（15%）
- 更换嵌入模型（2B → 8B）：
  - 性能略有波动，但始终显著优于随机选择。
> 🔍 **结论**：Data-DPO 不依赖特定质量或嵌入来源，具有良好的模块替换鲁棒性。

#### **(4) 奖励模型容量敏感性**
- 使用更简单的 plain MLP 替代 residual MLP：
  - 在 5% 预算下性能下降较明显（95.36 vs 100.76）
  - 在更高预算下差距缩小（101.75 vs 102.63）
> 🔍 **结论**：偏好监督本身是关键，更强的 reward model 在极低预算下更有优势。

---

## **4. 关键结论和发现**

### **主要发现**
1. **数据价值是动态的**：同一数据对不同模型的价值不同，应由目标模型自身反馈决定，而非静态属性。
2. **直接在目标模型上探测更可靠**：使用 proxy model 易引入偏差，尤其在低预算场景下影响显著。
3. **多信号融合至关重要**：仅依赖偏好会陷入“舒适区”，必须结合质量与多样性才能构建稳健训练分布。
4. **小数据也能超越全数据**：Data-DPO 在多个任务和预算下实现了 **ARP > 100**，验证了高质量数据选择的巨大潜力。

### **方法的局限性**
1. **对外部信号的潜在依赖**：虽然对质量/嵌入源鲁棒，但如果这些信号严重偏颇（如评分系统性错误），仍可能导致性能下降。
2. **假设分布一致性**：方法假设候选数据与目标任务分布一致，在存在显著分布偏移时效果可能受限。
3. **计算开销较高**：需在 probe set 上执行多次 forward/backward，总耗时约 19 GPU 小时（LLaVA-CoT），虽可接受但不如完全无训练的方法轻量。

### **未来工作方向**
- 探索更高效的 probing 策略（如梯度近似、零样本预测）以进一步降低计算成本。
- 将 Data-DPO 扩展到 **online / continual learning** 场景，动态调整数据流。
- 研究如何自适应地平衡偏好、质量与多样性权重，而非固定加权。
- 探索在 **Pre-training** 阶段的应用，实现端到端的数据效率优化。

--- 

> 📌 **一句话总结**：  
> **Data-DPO 通过让目标模型“自己投票选数据”，实现了比全量数据训练更优的小样本微调效果，重新定义了 SFT 中“什么是好数据”的标准。**

</details>

---

### 15. [TabNSM: Neural Sparse Mixer for Tabular Regression](https://arxiv.org/abs/2608.18026)

**Authors**: Ali Eslamian, Qiang Cheng  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.18026v1  

#### Abstract
Large-scale, high-dimensional tabular regression remains challenging: tree-based models are robust but lack end-to-end representation learning, while deep models enable flexible feature learning but often incur costly interaction modeling and sensitivity to noisy or redundant features. We propose Ta...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TabNSM: Neural Sparse Mixer for Tabular Regression 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代高维、异构的表格回归任务（如医疗、金融、工业监控）面临以下挑战：
- **特征稀疏性**：预测信号通常只集中在少数实例相关的“前景”特征上，而大量特征为噪声或冗余。
- **模型权衡**：传统树模型（如CatBoost）虽鲁棒但不支持端到端表示学习；深度模型（如Transformer）表达能力强但计算成本高（$O(D^2)$），且对噪声敏感。

### 提出的新方法与创新点
作者提出 **TabNSM**（Tabular Neural Sparse Mixer），一个面向大规模、高维表格回归的可扩展框架，其核心组件包括：

#### （1）Adaptive Sparse Interaction Module (ASIM)
- 结合了**实例自适应前景特征发现**、**稀疏局部交互编码**和**Feature-Token Mixing (FTM)**。
- 在固定稀疏配置下实现**近线性复杂度** $O(D)$，显著优于传统Transformer的 $O(D^2)$。
- 通过结构化稀疏注意力机制动态选择每条样本的关键特征子集，模拟树模型的选择性隔离能力，同时保持神经网络的可微性和灵活性。

#### （2）Multi-Stage Regression Head
- 多阶段残差耦合预测头，逐步从粗到细地提炼预测结果。
- 融合多尺度特征表示，增强最终输出的稳定性和表达力。

#### （3）GridLoss
- 一种**可微分软分箱目标函数**，将连续目标和预测映射到共享的有序网格空间。
- 鼓励预测在目标尺度上的**序数对齐**（ordinal alignment），补充传统的点对点损失（如MSE、MAE）。

#### （4）RISE (Reweighted Instance Sampling by Error)
- 一种基于误差的**难度感知采样策略**。
- 定期根据样本损失的分位数重新加权训练样本，使模型更关注难拟合或误差大的样本，提升泛化能力和优化稳定性。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|--------|
| **性能** | 在9个真实世界基准中，7项取得最优RMSE，尤其在高维异构数据上表现突出。 |
| **效率** | 近线性时间与内存复杂度，适合大规模高维表格数据。 |
| **鲁棒性** | 动态前景选择 + 全局上下文传播（FTM）有效抑制背景噪声影响。 |
| **训练稳定性** | GridLoss提供结构感知梯度，RISE缓解长尾误差分布问题。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共9个来自不同领域的公开回归数据集，涵盖机器人控制、空气质量、房地产等：

| 数据集 | 缩写 | 样本数 | 特征数 | 任务描述 |
|-------|------|--------|--------|----------|
| Sarcos Robot Arm | SA | 48,933 | 27 | 预测关节扭矩 |
| CPU Performance | CP | 8,192 | 12 | 预测CPU性能 |
| Topo | TO | 8,885 | 266 | 分子属性预测 |
| SCAN (Alzheimer’s) | SC | 4,355 | 386 | 预测MMSE认知评分 |
| Crime Data | CD | ~1M | 401 | 预测犯罪区域 |
| U.S. Chronic Disease | ChD | ~309K | 405 | 慢病指标预测 |
| Electric Vehicle Population | EVP | ~265K | 392 | 预测电动车续航 |
| Air Quality | AQ | ~19K | 390 | 预测污染物浓度 |
| Real Estate Sales | RES | ~1M | 389 | 预测房产销售比率 |

> 所有数据集均经过标准化处理，缺失值用零填充。

### 实验设置与评估指标
- **划分方式**：72%/8%/20% 的 train/validation/test 划分。
- **训练协议**：最多100轮，早停于验证损失最小点；使用AdamW优化器、混合精度训练。
- **超参调优**：使用Optuna在验证集上调参。
- **评估指标**：主要使用 **RMSE**，辅以MAE、R²、Spearman相关系数等。

### 基线方法对比
共比较了38种基线，分为七类：
1. **经典方法**：LinearRegression, KNN, Lasso, Ridge, SVR
2. **树集成**：Random Forest, CatBoost, LightGBM, XGBoost, ExtraTrees
3. **MLP风格**：MLP, ResNet, MLP-PLR
4. **Transformer类**：TabTransformer, FT-Transformer, SAINT, AutoInt, DCN-v2
5. **检索类**：TabR, ModernNCA
6. **其他深度架构**：TabNet, TANGOS, DNNR, SwitchTab
7. **表格基础模型（TFMs）**：TabPFNv2, TabPTM

---

## 3. 主要实验结果和性能指标

### 关键性能数据（RMSE）
在 **Table 1** 中报告了各模型家族的最佳表现及TabNSM的结果：

| Model Family | SA | CP | AQ | TO | SC | CD | ChD | EVP | RES |
|--------------|-----|-----|------|-------|------|------|-------|-------|------|
| Best Classical | 0.175 | 3.719 | 12.816 | 0.027 | 3.602 | 0.236 | 8717 | 26.104 | 3163 |
| Best Tree-based | 0.171 | 2.617 | 5.869 | 0.027 | 3.455 | 0.004 | 17525 | 2.799 | 3135 |
| Best MLP-style | 0.167 | 2.920 | 4.578 | 0.028 | 3.920 | 0.028 | 4963 | 1.556 | 3161 |
| Best Transformer | 0.156 | 2.901 | 4.081 | 0.028 | 3.667 | 0.007 | 4475 | 1.507 | 3163 |
| **TabNSM (Ours)** | **0.137** | **2.842** | **3.999** | **0.021** | **3.322** | **0.007** | **4460** | **1.199** | **3103** |

✅ **TabNSM在9个数据集中有7个达到最低RMSE**，其中：
- 在 **TO** 上相对最佳基线提升 **22%**（0.021 vs 0.027）
- 在 **EVP** 上提升 **21%**（1.199 vs 1.507）
- 在 **SA** 上超越最强深度模型DNNR（0.137 vs 0.149）

仅在 **CP** 和 **CD** 上未进入前二，因这两个数据集维度较低或树模型占优。

### 统计显著性检验
- **Wilcoxon signed-rank test**（配对检验）显示，TabNSM在所有38个基线上均具有统计显著优势（校正后 p ≤ 0.02）。
- 平均排名为 **1.89**，远低于第二名ResNet（9.72）。

### 消融实验结果
#### （1）ASIM模块有效性（Fig 2a）
- 替换为 Full Attention 或 Mamba 后性能下降明显。
- 表明：仅线性混合（如Mamba）不足以建模稀疏交互；**实例自适应稀疏选择是关键**。

#### （2）FTM路径消融（Table 8）
| 模型 | CP | TO | SA | EVP |
|------|----|----|----|-----|
| w/o FTM | 3.966 | 0.034 | 0.176 | 570.09 |
| Full TabNSM | 2.842 | 0.021 | 0.137 | 1.199 |
- 移除FTM导致严重退化，尤其在EVP上RMSE飙升至570，说明**全局上下文传播至关重要**。

#### （3）GridLoss与RISE作用
- **GridLoss** 在异构误差分布下优于标准损失（见 Fig 9），尤其在TO等任务上提升显著。
- **RISE** 显著降低验证误差方差（Fig 10），提高训练稳定性。

#### （4）参数缩放分析（Fig 2b）
- 参数量随特征维度增长呈**近似线性关系**，验证了ASIM的高效设计。

---

## 4. 关键结论和发现

### 主要发现
1. **实例自适应稀疏交互是高维表格回归的核心驱动力**  
   ASIM通过动态选择前景特征，在保留表达力的同时大幅提升信噪比（SNR），优于全连接或顺序扫描机制。

2. **结构感知监督与难度感知采样相辅相成**  
   GridLoss 引入序数一致性约束，RISE 引导模型聚焦困难样本，二者共同提升模型鲁棒性与泛化能力。

3. **端到端深度模型可在高维表格任务上超越树模型**  
   尽管树模型仍是强基线，但在高维异构场景下，TabNSM凭借灵活的表示学习和结构先验实现了系统性超越。

4. **可扩展性与实用性并存**  
   TabNSM在百万级样本、近400维特征下仍保持低训练时间和内存占用（见 Table 5–7），具备实际部署潜力。

### 局限性
- **依赖GPU训练**：相比CPU运行的CatBoost，TabNSM需GPU加速，可能限制部分资源受限场景的应用。
- **对目标分布敏感**：GridLoss中的温度参数 $T$ 对目标分布形态（如IQR/Std比值）较敏感，需谨慎调参（见 Table 9）。
- **当前仅适用于单目标回归**：尚未扩展至多目标或多任务设定。

### 未来工作方向
1. 探索 **模型压缩与蒸馏** 技术以降低推理开销。
2. 将 **GridLoss 扩展至多目标回归** 场景。
3. 研究 **前景-背景交互机制在多模态表格学习中的应用**。
4. 构建更大规模的 **表格预训练范式**，结合TabNSM架构进行迁移学习。

---

> ✅ 总结：**TabNSM 是首个将实例自适应稀疏注意力、结构化监督与难度感知采样统一于高维表格回归框架的工作，在性能、效率与鲁棒性之间取得了良好平衡，代表了深度表格模型的重要进展。**

</details>

---

### 16. [Mixture-of-Expert Blocks Contain Strong Hallucination Detection Signals](https://arxiv.org/abs/2608.17687)

**Authors**: Joao Fonseca, Rodrigo Rodrigues, Paolo Romano  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.17687v1  

#### Abstract
Despite their widespread use, Large Language Models (LLMs) remain limited by a fundamental problem: the generation of plausible but false content, known as hallucinations. Most existing detection methods operate at the answer or sentence level, yet per-token detection is essential for localizing hal...

---

### 17. [What Tokens are Learned when Tokenization is Optimized Jointly with Language Modeling?](https://arxiv.org/abs/2608.17325)

**Authors**: Saketh Reddy Vemula, Parameswari Krishnamurthy  
**Category**: cs.CL  
**Published**: 2026-08-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.17325v1  

#### Abstract
Tokenization is a fundamental component of language modeling pipelines. Despite its importance, it is often fixed, even though it significantly impacts model performance across languages. In this work, we analyze what tokens are learned when tokenization is jointly optimized with language modeling. ...

---

### 18. [Do Large Language Models Play Six Degrees of Separation? Measuring Topological Compression in Long-Context Manifolds](https://arxiv.org/abs/2608.17950)

**Authors**: Md. Faiyaz Abdullah Sayeedi  
**Category**: cs.CL  
**Published**: 2026-08-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.17950v1  

#### Abstract
Large Language Models (LLMs) demonstrate remarkable multi-hop reasoning capabilities over long contexts, yet the internal mechanisms enabling these distant cognitive leaps remain poorly understood. Traditional attention-based interpretability often fails to capture true semantic proximity due to rou...

---

### 19. [Certified but Private: Scalable Zero-Knowledge Proofs for Neural Network Guarantees](https://arxiv.org/abs/2608.17070)

**Authors**: Youwei Zhong, Ben Merbaum, Timos Antonopoulos, Ning Luo, Charalampos Papamanthou, Katerina Sotiraki, Ruzica Piskac  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.17070v1  

#### Abstract
With the growing deployment of machine learning models, formal guarantees of the robustness and fairness of these models have become increasingly important in safety-critical and legal-compliance settings. However, model parameters are often commercial secrets that cannot be disclosed to auditors or...

---

### 20. [Elimination Geometry](https://arxiv.org/abs/2608.17646)

**Authors**: Mian Huang, Xueqin Wang  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.17646v1  

#### Abstract
This monograph develops elimination geometry (EG), a typed, native-loss, audit-oriented framework for studying when locally optimal objects can be realized by a shared deployment rule. Elimination and compression may erase distinctions required by prediction, inference, control, or representation. E...

---

### 21. [Debate Training Reduces Reward Hacking in RLAIF](https://arxiv.org/abs/2608.17776)

**Authors**: Zachary Kenton, Lili Janzer, Rory Greig, Tian Huey Teh, Kirill Tyshchuk, Jonah Brown-Cohen, Harri Edwards, Senthooran Rajamanoharan, Noah Y. Siegel, Natasha Jaques, Rohin Shah  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.17776v1  

#### Abstract
We demonstrate that RL finetuning an LLM using debate, a two-player adversarial game between a generator and a critic adjudicated by a weaker LLM judge, reduces reward hacking compared to a reinforcement learning from AI feedback (RLAIF) baseline. Reward hacking is a central obstacle in RLAIF: as tr...

---

### 22. [Wuying-Browser-Agent: Real-World Centric Fundamental Long-Horizon Browser Agents](https://arxiv.org/abs/2608.17319)

**Authors**: AIMAE Team, Tianxiang Chen, Yan Cheng, Zhangye Han, Xiaowei Li, Chang Liu, Cheng Liu, Zhongqiang Ma, Long Peng, Xiaobing Tu, Yinggui Wang, Hongliang Wei, Chen Wu, Daiping Xin, Kunyu Zhou, Pengyang Zhou, Peiyuan Chen, Ziyuan Chen, Yutao Deng, Chunyu Dong, Xiangyu Fu, Yicheng Feng, Ruian He, Haochen Li, Miancan Liu, Zhengqin Liu, Wei Peng, Jinkui Ren, Haoyu Tan, Dong Xiao, Rongkun Xue, Shujian Yang, Xianhang Ye, Ziqi Yuan, Ziyang Yu, Linghan Zhang, Xiantao Zhang, Xuanpu Zhao, Yinan Zhao, Zhenghui Zhao, Bin Zhu, Likai Zou  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.17319v1  

#### Abstract
Browser agents perform well on short, clean demonstrations, but real deployment is fundamentally different: agents must sustain dozens of decisions on live websites while recovering from mistakes and navigating complex UIs. We argue that closing this gap requires alignment at every level of the pipe...

---

### 23. [Judge, Retrieve, or Abstain: Uncertainty-Guarded LLM Judging with Provable Risk Guarantees](https://arxiv.org/abs/2608.17994)

**Authors**: Sher Badshah, Ali Emami, Hassan Sajjad  
**Category**: cs.CL  
**Published**: 2026-08-19  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.17994v1  

#### Abstract
Using LLMs as judges has become standard practice for evaluating model outputs at scale. This is particularly common for subjective, open-ended tasks such as assessing helpfulness or alignment, where no single reference answer exists. However, objective tasks introduce a distinct reliability challen...

---

### 24. [Hierarchical Data Selection via Manifold Coverage and Sparse Feature Coverage in LLM Post-training](https://arxiv.org/abs/2608.16927)

**Authors**: Peng Sun, Yi Yang, Antong Zhang, Chunxiao Li, Yanbo Wang, Dianbo Liu, xin chen, Kai Yu, Lu Chen, Tianfan Fu  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.16927v1  

#### Abstract
As supervised fine-tuning data continues to scale, selecting high-value subsets from large candidate pools is crucial for reducing training cost and improving model performance. Existing methods often measure diversity directly in the original embedding space, where geometric metrics entangle domina...

---

### 25. [RoBell-RVFL: A Robust Generalized Bell Random Vector Functional Link Network](https://arxiv.org/abs/2608.16965)

**Authors**: A. Rahaman, A. Quadir, M. Tanveer  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.16965v1  

#### Abstract
The dominance of majority classes in real-world datasets poses a fundamental challenge to randomized neural networks, often biasing decision boundaries and overlooking critical minority samples. Existing remedies, such as synthetic minority over-sampling (SMOTE) and class-weighted loss functions, pr...

---

### 26. [SCENARIODIFF: A Scenario-level Guidance Framework for Multimodal Time Series Forecasting--Extended Version](https://arxiv.org/abs/2608.17164)

**Authors**: Tuan-Binh Tran, Dat Nguyen Cong, Duc-Trong Le, Thanh Trung Huynh, Tung Kieu  
**Category**: cs.LG  
**Published**: 2026-08-19  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.17164v1  

#### Abstract
Textual context such as news, reports, and logs can provide valuable signals for time series forecasting, especially when future dynamics are driven by external events that are not yet visible in historical values. Existing multimodal forecasting methods often either ask large language models (LLMs)...

---

### 27. [SignalReasoner: Assessing the Upper Bound of 3B Models for Signal Mathematical Reasoning](https://arxiv.org/abs/2608.17301)

**Authors**: Guozheng Sun  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.17301v1  

#### Abstract
Post-training with supervised chain-of-thought fine-tuning and reinforcement learning from verifiable rewards has substantially improved the mathematical reasoning capabilities of large language models (LLMs). However, their application to signal processing problems remains relatively under-explored...

---

### 28. [Accuracy and Robustness of Model Cascades Under Data Perturbations](https://arxiv.org/abs/2608.17711)

**Authors**: Pallavi Mitra, Jai Kushwaha, Felix Biessmann  
**Category**: cs.AI  
**Published**: 2026-08-19  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.17711v1  

#### Abstract
Prediction cascades significantly reduce energy consumption of Artificial Intelligence (AI) models while maintaining high predictive performance. The idea is that easy inputs are routed through a lightweight small model, and difficult uncertain cases are deferred to a larger model. While this design...

---

### 29. [Margin-Regularized Structured Semantic Alignment for Brain-Language Correspondence](https://arxiv.org/abs/2608.16975)

**Authors**: Jiaqi Wang, Huawen Hu, Shu Zhang  
**Category**: cs.CL  
**Published**: 2026-08-19  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.16975v1  

#### Abstract
With the rapid advancement of large language models, brain-language decoding has achieved remarkable progress. However, it remains unclear whether decoded content genuinely reflects neural representations or is largely reconstructed by the language model itself. This ambiguity limits interpretabilit...

---

### 30. [Emotion Across Speech and Faces: Shared Affective Mechanisms in Multimodal Foundation Models](https://arxiv.org/abs/2608.17102)

**Authors**: Xiutian Zhao, Luqi Sun, Bj\"orn Schuller, Berrak Sisman  
**Category**: cs.CL  
**Published**: 2026-08-19  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.17102v1  

#### Abstract
Modern multimodal foundation models (MFMs) have made rapid progress on tasks requiring integrated perception across speech, vision, and language, including emotion recognition. However, it remains unclear whether they recognize speech and facial emotion through shared affective functional units or m...

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

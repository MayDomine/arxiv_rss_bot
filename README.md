# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-20 06:06:14 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FlashAttention for Scalable Vector Architectures](https://arxiv.org/abs/2608.18656)

**Authors**: Sonia Rani Gupta, Nikela Papadopoulou, Miquel Peric\`as  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2608.18656v1  

#### Abstract
Inference with transformer models on CPUs is increasingly important, especially for Small Language Models (SLMs), where vector architectures are emerging as a promising execution substrate. The attention module is a major bottleneck due to high memory bandwidth requirements; FlashAttention mitigates...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FlashAttention for Scalable Vector Architectures

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **FlashAttention** 在向量化架构（如 RISC-V RVV 和 Arm SVE）上的扩展性受限，其向量并行度仅沿注意力头维度（head dimension $D$）展开。由于 $D$ 通常较小（如 64 或 128），当硬件支持的 **Vector Length (VL)** 超过 $D$ 时，会出现 **SIMD 资源利用率不足** 的问题，导致长向量架构无法充分发挥性能潜力。

此外，当前主流的 **Q8_0 量化格式**在长向量执行下也存在结构性瓶颈，限制了算术操作的摊销效率。

### 提出的新方法：FlashAttention-V
本文提出 **FlashAttention-V**，一种专为可扩展向量架构设计的优化版 FlashAttention 算法，核心创新包括：

- **跨注意力头的并行性利用（Inter-head Parallelism）**  
  通过重新组织循环顺序，将多个注意力头的数据打包进单个向量寄存器中，实现 **VL > D** 时的有效向量化。

- **跨头打包（Inter-head Packing）**  
  将来自不同注意力头的 Q、K、V 数据按列连续存储，并在向量寄存器中打包处理，提升向量宽度利用率。

- **循环重排序与展开（Loop Reordering & Unrolling）**  
  通过对注意力头维度进行循环展开，暴露指令级并行性（ILP），提高向量寄存器占用率和内存访问局部性。

- **原生支持 GQA/MQA**  
  避免共享 K/V 头的重复加载，减少内存带宽压力。

- **缓存感知分块（Cache-aware Blocking）**  
  借鉴 FlashAttention-2 思路，采用合适的 block size 提高数据复用率。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **向量利用率** | 支持 VL > D，突破原有 VL ≤ D 的限制，充分利用长向量寄存器 |
| **性能增益** | 在 prefill 阶段实现 22×–42× 加速，在 decode 阶段达 8×–11× |
| **可移植性** | 支持 RVV 和 Arm SVE，已在 QEMU 上验证功能正确性 |
| **系统集成** | 已集成至 `llama.cpp` 中的 `ggml` 框架，具备实际部署价值 |

---

## 2. 核心实验方法和设置

### 使用的模型（非传统“数据集”）
论文在以下四种 **Decoder-only LLMs** 上进行评估：

| Model | Attention Type | Heads | KV Heads | Head Dim ($D$) | Size |
|-------|----------------|--------|----------|------------------|------|
| **TinyLlama** | GQA | 32 | 4 | 64 | 1.1B |
| **Llama 3.2** | GQA | 32 | 8 | 64 | 1B |
| **Qwen2.5** | GQA | 16 | 2 | 128 | 3B |
| **Pythia-410M** | MHA | 16 | 16 | 64 | 410M |

> 注：所有实验均使用 **batch size = 1**

### 实验平台
#### （1）仿真环境（gem5@RVV）
- 架构：RISC-V in-order `RiscvMinorCPU`
- 向量扩展：RVV v1.0
- 向量长度：512-bit 到 8192-bit
- 内存：DDR3-1600，带宽 12.8 GB/s
- 缓存：L1 64KB，L2 1MB
- 特殊建模：引入 **VL-proportional SIMD 功能单元延迟**，更真实反映长向量执行开销

#### （2）真实硬件平台
- **Banana Pi BPI-F3**：RISC-V 平台，支持 RVV v1.0，VL = 256-bit
- 使用 QEMU 模拟 Arm SVE 进行功能验证

### 评估指标
- **速度提升（Speedup）**：相对于标量实现（ggml-scalar）和原生向量化实现（ggml-vec）
- **执行时间（Execution Time）**
- **端到端吞吐量（Tokens/sec）**
- **模块级性能占比分析**

### 基线方法对比
| 基线名称 | 描述 |
|--------|------|
| **ggml-scalar** | 非向量化 FlashAttention 实现 |
| **ggml-vec-fp16 / fp32** | llama.cpp 官方向量化实现，仅支持 VL ≤ D |
| **Titopoulos et al. [39]** | RVV 上的 FlashAttention 实现，但不兼容 llama.cpp 张量布局，未直接比较 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（基于 gem5 仿真，512-bit VL）

| 模型 | 阶段 | FlashAttention-V vs Scalar | FlashAttention-V vs ggml-vec |
|------|------|----------------------------|------------------------------|
| TinyLlama | Prefill | **24× (FP32)**, **28× (FP16)** | 4× |
| Llama 3.2 | Prefill | ~27× | — |
| Qwen2.5 | Prefill | **42×** | — |
| Pythia-410M | Prefill | ~22× | — |
| TinyLlama | Decode | **7.6× (FP32)** | 3.8× |
| 所有模型 | Decode | **8×–11×** | ~2× |

> 表格来源：Table 4 及 Figure 3

### 在真实硬件（Banana Pi BPI-F3）上的验证结果
- **Prefill 阶段**：
  - 相比 `ggml-scalar`：**12×–14×** 加速
  - 相比 `ggml-vec-fp32`：**3.7×** 加速
- **Decode 阶段**：
  - 相比 `ggml-scalar`：**4×–5×** 加速
  - 相比 `ggml-vec`：约 **2×** 加速
- **短序列（N ≤ 128）表现最佳**，符合边缘设备典型场景

### 可扩展性分析（Scaling Behavior）

#### Prefill 阶段（图 3 & 图 4）
- 当 VL 从 512-bit 扩展到 8192-bit：
  - **理想情况（固定延迟）**：额外获得 **2×–3×** 加速
  - **现实情况（延迟随 VL 增加）**：
    - 8-lane 架构在 1024-bit 后即饱和
    - **64-lane + 4096-bit VL** 配置达到最优扩展性，带来 **2×–2.5×** 额外加速
    - 超过 4096-bit 后收益递减，受制于 OpLat 开销

#### Decode 阶段（图 5 & 图 6）
- 对 VL 扩展敏感度低：
  - 在 512-bit VL 下已达 **8×–11×** 加速
  - 扩展至 4096-bit 仅带来 **~1.2×** 额外增益（TinyLlama/Llama 3.2）
  - 更高 VL 几乎无收益，因单 token 生成缺乏足够并行任务来摊销打包开销
- 性能对 OpLat 不敏感，几乎不受向量宽度影响

### 消融实验与关键观察
- **循环重排序与展开** 是性能提升的关键因素之一，在 Banana Pi 上带来显著增益
- **FP16 相比 FP32**：
  - 在 8192-bit VL 下可达 **~2×** 加速
  - 在 512-bit VL 下仅 **~11%** 提升，受限于 softmax 计算瓶颈
- **FlashAttention-V vs 优化后的 Self-Attention**：
  - 在 512-bit VL 下快 **1.3×**
  - 在 8196-bit VL 下快 **~2.4×**，证明 FlashAttention 的融合策略仍具优势

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **FlashAttention-V 成功打破 VL ≤ D 的限制**，首次实现对 VL > D 架构的有效支持，显著提升向量资源利用率。
2. ✅ 在 prefill 阶段，通过跨头打包和循环优化，实现高达 **42×** 的加速；decode 阶段也取得 **8×–11×** 提升。
3. ✅ **64-lane + 4096-bit VL** 是当前条件下最优配置，可在保持低延迟的同时最大化吞吐。
4. ⚠️ **Q8_0 量化格式存在结构性缺陷**：
   - 权重与 scale 交错存储（Array-of-Structures）
   - 导致必须显式进行 **packing** 和 **masked reduction**
   - 在 2048-bit VL 下，这两项操作占总周期 **60%**
   - 且开销随 VL 增长而增加，无法摊销算术收益 → **阻碍长向量扩展性**

### 方法的局限性
- **依赖注意力头数量充足**：若模型头数少（如 Pythia-410M 仅 16 个），难以填满超长向量寄存器。
- **decode 阶段收益有限**：受限于单 token 串行生成模式，无法充分释放并行潜力。
- **Q8_0 量化成为瓶颈**：即使注意力模块高度优化，线性层仍拖累整体扩展性。
- **当前模拟工具链不完善**：gem5 对 LMUL ≥ 4 的行为模拟存在问题，制约深入探索。

### 未来工作方向
1. 🔄 **设计面向长向量友好的量化格式**：例如将 weights 和 scales 分离存储（SoA），避免运行时 packing。
2. 🔍 **探索新的 kernel fusion 策略**：结合 GEMV 优化思路，进一步降低 decode 阶段开销。
3. 💻 **推动硬件-软件协同设计**：建议未来向量处理器增强对结构化稀疏/量化数据的原生支持。
4. 🧪 **完善仿真基础设施**：修复 gem5 等工具对高级 RVV 特性的支持，便于研究验证。

---

> **总结一句话**：  
> FlashAttention-V 通过 **跨头并行 + 循环优化 + 分块策略**，首次实现了 FlashAttention 在 **VL > D** 场景下的高效执行，在 prefill 阶段取得数十倍加速；但也揭示了 **Q8_0 量化格式** 对长向量扩展的根本性制约，呼吁软硬协同重构量化执行范式。

</details>

---

### 2. [Training-Free Inference-Time Self-Reflection and Cost-Bounded Early Stopping for Large Language Models](https://arxiv.org/abs/2608.18884)

**Authors**: Wei Yu, Suxing Liu, Minjie Yu, Jiahao Wang, Zhijian Zheng, Haocheng Deng, Bing Li  
**Category**: cs.AI  
**Published**: 2026-08-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.18884v1  

#### Abstract
Reinforcement-learning training of reasoning LLMs (e.g., GRPO) is expensive and requires a controllable environment, committing every contribution to a full training pipeline. We present EvoResearcher, a training-free, inference-time protocol that adds cost-bounded self-reflection to a single frozen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Training-Free Inference-Time Self-Reflection and Cost-Bounded Early Stopping for Large Language Models

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对当前大型语言模型（LLM）在多步推理任务中面临的三大挑战：
- **训练成本高昂**：通过强化学习（如 GRPO）训练具备推理能力的 LLM 需要大量 GPU 资源，并依赖可控的训练环境。
- **推理时缺乏自我验证机制**：标准的单次生成（single-shot）容易产生过度自信或错误答案，而现有的反思机制（如 self-consistency、fixed-depth self-refine）无法有效控制计算开销。
- **无界反射成本**：固定深度的反思循环要么浪费资源（对已正确答案进行不必要的修正），要么提前终止导致错误确认。

### 提出的新方法：EvoResearcher 协议
提出了一种**无需训练的推理时自省协议**（training-free inference-time self-reflective protocol），名为 **EvoResearcher**，其核心是 `generate → self-critique → revise` 循环，具有以下创新设计：

- **成本有界的早期停止机制（Cost-Bounded Early Stopping）**：
  - 引入 `CONFIRMED` **哨兵信号**作为隐式早停条件。当模型自我批判返回 `CONFIRMED` 时，立即终止循环，避免冗余计算。
  - 设置最大迭代深度 $D$ 作为硬性上限，确保计算预算可控。

- **基于提示的元奖励机制（Prompt-Level Meta-Reward）**：
  将四种行为激励（无需梯度更新）编码为提示词中的指令：
  1. **Correctness**（正确性）：要求模型验证每一步。
  2. **Efficiency**（效率）：鼓励简洁路径，避免冗余步骤。
  3. **Reflection Depth**（反思深度）：引导模型识别错误并回溯策略。
  4. **Tool-Call Diversity**（工具调用多样性）：促进探索不同信息源（本研究中仅以提示形式体现）。

### 相比现有方法的优势
- **零训练成本**：直接作用于冻结的 LLM 主干，无需任何微调或额外训练。
- **动态计算分配**：简单问题快速通过 `CONFIRMED` 停止，复杂问题则允许更多反思步骤，实现“按需计算”。
- **高性价比**：在保持甚至提升准确率的同时，显著降低平均推理成本（约 2.1 次生成/题），优于固定深度或多路径采样的基线。

---

## 2. 核心实验方法和设置

### 数据集
在三个纯推理基准上进行评估，均无需外部工具调用：
- **Big-Bench Hard (BBH)**：100 题（n=100），涵盖逻辑、算术、时间推理等 20 类多步推理任务。
- **GSM8K**：500 题（n=500），小学数学应用题。
- **MATH**：500 题（n=500），竞赛级数学难题。

### 实验设置
- **主干模型**：
  - 主要使用 `deepseek-v4-flash`（API 接入）。
  - 在 E9 中复现于 `Qwen2.5-72B-Instruct` 以验证跨模型泛化性。
- **温度**：默认 0.3，部分实验测试 {0.1, 0.3, 0.5}。
- **循环深度**：最大深度 $D=3$。
- **评估指标**：
  - **Accuracy**：标准化精确匹配。
  - **Average Steps / Avg. generations**：平均每题的 LLM 调用次数。
  - **Early-Stop Rate**：被 `CONFIRMED` 提前终止的比例。
  - **Average Tokens**：每题消耗的平均 token 数。
  - **95% Wilson Score Interval**：用于判断统计显著性（n=100 时半宽 ~8.7pp；n=500 时 ≤4.3pp）。

### 基线方法对比
| 方法 | 停止条件 | 平均生成数 |
|------|----------|------------|
| **Single-shot (d=1)** | 无 | 1.00 |
| **Self-Consistency (k=2,3)** | 固定 k 条路径 | 2.00 / 3.00 |
| **Self-Refine (d=3)** | 固定深度 3，忽略 `CONFIRMED` | 3.00 |
| **EvoResearcher (d=3)** | `CONFIRMED` 或达到 d=3 | ~2.15 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比

#### 在 BBH 上的结果（clean reasoning）
- **准确率持平**：EvoResearcher 达到 **74.0%**，与 single-shot (73.8%) 和其他基线相当，未超越模型固有能力上限。
- **显著降低成本**：
  - **平均生成数仅 2.15**，远低于 fixed-depth Self-Refine 的 3.00。
  - **82–88% 的题目被 `CONFIRMED` 早停**，实现高效自我验证。
  - token 成本比最坏情况（3×single-shot）节省 **60%**。

#### 在 GSM8K 和 MATH 上的结果（hard reasoning）
| 基准 | 方法 | 准确率 | 提升 | 平均生成数 | 早停率 |
|------|------|--------|------|------------|--------|
| **GSM8K** | Single-shot | 93.6% | — | 1.00 | 0% |
| | EvoResearcher | **97.8%** | **+4.2pp** | 2.15 | 88% |
| **MATH** | Single-shot | 26.2% | — | 1.00 | 0% |
| | EvoResearcher | **40.4%** | **+14.2pp** | 2.42 | 82% |

- **准确率显著提升**：在单次推理表现差的任务上（尤其是 MATH），EvoResearcher 带来巨大增益（+14.2pp），且所有改进均在 n=500 下达到统计显著（Wilson 区间不重叠）。
- **仍保持高早停率**：尽管更难，仍有 82–88% 的题目能被有效早停。

#### 消融实验结果
- **组件强调实验 (E1)**：单独强调任一 meta-reward 组件（如 +Correctness）对 single-shot 准确率无显著影响（73.3–74.0%，均在 CI 内）。
- **组件消融实验 (E2)**：移除任一组件后准确率变化 ≤4pp，仍在完整奖励的置信区间内，表明各组件非必要但协同作用。
- **信心阈值实验 (E6)**：
  - 模型的 `<confidence>` 信号**严重过自信**：所有阈值 $t \geq 0.6$ 均导致 100% 第一轮停止（平均步数=1.00）。
  - 结论：**`CONFIRMED` 哨兵是有效的成本控制机制，而非数值信心标签**。

#### 跨模型复现 (E9)
在 `Qwen2.5-72B` 上复现相同模式：
- **BBH**：准确率从 78.2% → 78.4%（持平），早停率 81%。
- **MATH**：准确率从 31.6% → **43.2%**（+11.6pp），早停率 78%。
- 表明方法具有良好的**跨模型泛化性**。

---

## 4. 关键结论和发现

### 主要发现
1. **双重收益模式**：
   - 在**干净推理任务**（如 BBH）上，EvoResearcher 不提升绝对准确率，但通过 `CONFIRMED` 实现**成本有界的自我验证**，节省约 30–60% 计算。
   - 在**困难推理任务**（如 MATH）上，单次推理不可靠，此时反思循环带来**显著准确率提升**（+14.2pp），且多数题目仍可早停。

2. **`CONFIRMED` 是核心机制**：相比数值信心阈值，`CONFIRMED` 哨兵是更可靠、更有效的成本控制手段，因其具备判别力（混淆矩阵显示 78% 判断正确）。

3. **推理时计算扩展的有效途径**：该协议提供了一种**无需训练即可扩展推理时计算**的实用方案，尤其适合 API 预算受限的场景。

### 局限性
- **基准限制**：仅在纯推理任务（BBH/GSM8K/MATH）上验证，**未涉及工具调用**（tool-use），因此 `tool-call diversity` 组件仅以提示形式存在，未在真实环境中评估。
- **训练蓝图未执行**：文中提到的 GRPO 训练目标、演化虚拟世界（Evolving Virtual World）、多智能体系统（multi-agent swarm）仅为设计蓝图，**尚未实现或验证**。
- **模型与会话敏感性**：结果可能受 backbone 和 API 会话差异影响，数值信心需针对不同模型重新校准。
- **权重设定启发式**：四个 meta-reward 组件及内部权重（如 $\beta_1,\beta_2,\beta_3$）为启发式选择，最优配置有待进一步研究。

### 未来工作方向
- 在支持工具调用的基准（如 **GAIA**）上验证完整的 `tool-call diversity` 和环境对抗过滤能力。
- 实现并评估 GRPO 训练蓝图，探究将 prompt-level 机制“内化”到模型权重后的效果。
- 探索多智能体协作框架下的自省与进化。
- 研究更优的组件权重配置和信心校准方法，提升通用性。

</details>

---

### 3. [LLM-Powered Predictive Decision-Making for Sustainable Data Center Operations](https://arxiv.org/abs/2608.18503)

**Authors**: Hanzhao Wang, Jingxuan Wu, Yumeng Li, Yu Pan, Guanting Chen  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.18503v1  

#### Abstract
The growing demand for AI-driven workloads, particularly from Large Language Models (LLMs), has raised concerns about the significant energy and resource consumption in data centers. This work introduces a novel LLM-based predictive scheduling system designed to enhance operational efficiency while ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LLM-Powered Predictive Decision-Making for Sustainable Data Center Operations 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
随着 **Large Language Models (LLMs)** 和其他 AI 工作负载的快速增长，数据中心面临巨大的能源消耗、碳排放和资源调度效率低下的挑战。传统资源调度依赖用户提供的粗略估计（如任务时长），而这些估计往往不准确，导致 **GPU 资源浪费、排队延迟高、能耗增加**。

本文旨在解决以下核心问题：
- 如何在任务提交前，精准预测其执行时间（execution time）和能量消耗（energy consumption）？
- 如何基于预测结果进行实时、高效的 **GPU 资源调度决策**，以同时优化能效与响应速度？

---

### 提出了什么新方法或新思路
作者提出了一种 **端到端的 LLM 驱动预测性调度框架（LLM-based predictive scheduling system）**，分为两个阶段：

#### Phase 1: LLM-based Prediction
- 输入：用户的 **source code**
- 输出：对任务在不同 GPU 配置下运行所需的 **execution time** 和 **energy consumption** 的预测
- 方法：利用预训练的 **LLM（如 Starcoder-7B）提取代码语义表示（embedding）**，再通过一个轻量级 **probe（浅层神经网络）** 进行回归预测

#### Phase 2: Real-time Scheduling
- 基于上述预测值，设计了两种在线调度算法：
  - **Greedy Algorithm**：先来先服务，选择使加权目标最小的 GPU 类型
  - **Value-based Algorithm**：受多重背包问题启发，为每个任务-GPU 对计算“价值” $ v_{ij} = \frac{1}{a_{ij} t_{ij}} - K e_{ij} $，优先分配高价值组合

该系统具备以下创新特性：
- **通用性强**：可处理任意类型的 ML 任务（CNN、ViT、GAN、LLM 等）
- **可扩展性好**：只需添加新的 probe 即可预测水耗、碳排放等其他可持续性指标
- **自动化程度高**：无需人工特征工程，实现从代码输入到资源调度的全流程自动化

---

### 相比现有方法的优势

| 维度 | 传统方法 | 本方法 |
|------|--------|-------|
| **特征提取方式** | 手工设计特征（handcrafted features），如层数、前向传播次数等 | 利用 LLM 自动生成高维语义表示，更具表达力和泛化能力 |
| **模型通用性** | 特定任务专用模型（如仅适用于 CNN 或 LLM） | 统一框架支持多种任务类型及复合任务（如 CNN + LSTM） |
| **数据需求** | 需大量标注数据训练独立模型 | 借助 LLM 强大的预训练知识，仅需约 **500 个样本**即可训练有效 probe |
| **灵活性与可扩展性** | 每个目标需单独建模（time、energy 各自一套） | 统一表示 + 多 probe 架构，轻松扩展至 water、carbon 等新指标 |

> 图 2 显示，传统方法是“多模型孤岛”，而本文方法是“统一预测平台”。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **自建数据集**：收集了 **500 个独立可运行的 ML 源码文件**
  - 包含多种模型架构：ResNet, BERT, GAN, ViT, VGG 等
  - 每个代码在 **NVIDIA A100 和 A6000** 两类 GPU 上重复运行 ≥2 次
  - 使用 `nvidia-smi` 工具采集每秒功耗，计算总 energy 和 execution time
  - 数据划分：9:1 分为训练集与测试集

---

### 实验设置和评估指标

#### 预测模型（Probe）设置
- **LLM 模型**：Starcoder-7B（code-specialized LLM）
- **Embedding 提取**：取最后一层最后一个 token 的输出（4608 维向量）
- **Probe 结构**：3 层 Dense NN（1024 → 30 → 1），ReLU + BatchNorm
- **训练配置**：
  - Optimizer: AdamW
  - Loss: MSE
  - Regularization: L1 + Weight Decay
  - Early Stopping: 若连续 30 轮 loss 下降 < 0.001

#### 决策算法设置
- 在真实合作数据中心部署，收集两个月（2024.07–2024.09）操作日志
- 对比三种调度策略：
  1. **Simple Rule（Baseline）**：FCFS + 优先分配最强可用 GPU（如 A100）
  2. **Greedy Algorithm**
  3. **Value-based Algorithm**

#### 评估指标
| 缩写 | 全称 | 含义 |
|------|------|------|
| TWT | Total Waiting Time | 总等待时间 |
| TDT | Total Delayed Tasks | 发生等待的任务数 |
| CRT | Cumulative Running Time | 累计运行时间 |
| TEC | Total Energy Cost | 总能耗（kWh） |

---

### 基线方法对比
- **Simple Rule**：工业界常见做法，无预测机制，纯启发式调度
- **Previous ML Methods**：如 [22][23] 中基于程序切片和手工特征的方法，在图 3 中作为性能对比基准

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自真实数据中心部署）

| 方法 | TWT (s) ↓ | TDT ↓ | CRT (s) ↓ | TEC (kWh) ↓ |
|------|-----------|--------|------------|--------------|
| Simple Rule | 3,135,824.01 | 29.79 | 16,924,533.71 | 470.69 |
| Value-based | 2,186,166.49 | 22.50 | 16,565,627.33 | 322.07 |
| **Improvement (%)** | **30.28%** | **24.47%** | 2.12% | **31.58%** |

> ✅ **核心成果**：相比基线，**能耗降低 32%**，**等待时间减少 30%**

---

### 与基线方法的对比结果
- **预测准确性更高**：
  - 图 3 显示，LLM + Probe 方法的预测点更贴近理想对角线（ideal prediction）
  - 传统方法存在系统性偏差（systematic bias），尤其在未见过的任务上表现差
- **调度效果显著优于 baseline**：
  - 图 4(b)(c) 显示，Value-based 方法明显减少了排队任务数量和实时能耗峰值
  - Greedy 方法虽优于 Simple Rule，但仍不如 Value-based

---

### 消融实验结果（见 Appendix A.3.2）
通过仿真实验验证不同优化目标下的性能表现：

| 调优目标 | 指标 | Simple Rule | Value-based |
|---------|------|-------------|------------|
| **Minimize Waiting Time** | TWT | 2,466,062.56 | **1,810,123.92** (-26.6%) |
| **Minimize Running Time** | CRT | 15,333,868.47 | **15,001,371.15** (-2.1%) |
| **Minimize Energy** | TEC | 440.98 | **240.07** (-45.6%) |

> 🔍 发现：Value-based 算法可根据调参灵活适应不同优化目标，表现出良好的可控性和鲁棒性。

此外，**推理速度快**：单个任务预测耗时平均 **0.65 秒**，满足实时调度需求。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 可作为强大的代码理解工具用于系统级预测**  
   LLM 提取的语义表示远超手工特征，在少量样本下仍能实现高精度预测。

2. **预测 + 决策联合优化显著提升数据中心可持续性**  
   在真实场景中实现了 **32% 能耗下降** 和 **30% 排队时间缩减**，证明了该方法的实际可行性。

3. **统一框架具有高度可扩展性**  
   只需更换 probe 即可预测 water usage、carbon emissions 等指标，适用于未来绿色数据中心建设。

4. **Value-based 算法优于 Greedy 和 FCFS**  
   因其考虑全局任务队列的价值排序，而非局部最优，更适合复杂调度环境。

---

### 方法的局限性
1. **依赖 LLM 的代码理解能力**  
   若代码风格差异过大（如变量命名混乱、注释缺失），可能导致 embedding 偏离训练分布（OOD 问题）

2. **当前 probe 训练假设 LLM 参数冻结**  
   未进行 full fine-tuning，可能限制性能上限

3. **Rewriting 成本较高**  
   使用 Align-LLM（如 GPT-4o）重写代码平均耗时 **9.7 秒/任务**，影响实时性

4. **GPU 类型有限制**  
   当前实验集中在 A100/A6000，跨代硬件泛化能力有待验证

---

### 未来工作方向
1. **引入 Fine-tuning 或 RLHF 提升预测精度**  
   探索对 LLM 本身微调，或结合人类反馈进一步优化预测质量

2. **构建 Reward Model 实现端到端学习**  
   将调度结果反馈给模型，形成闭环优化

3. **拓展至更多可持续性指标**  
   如直接预测碳足迹（carbon emissions）、冷却用水量（water usage）

4. **开发轻量化 Align-LLM 替代方案**  
   减少 OOD 修复的时间开销，提升系统吞吐量

5. **探索异构集群中的动态频率调节（DVFS）集成**  
   结合 GPU 功耗控制，进一步挖掘节能潜力

---

> 📌 **总体评价**：本文开创性地将 **LLM 应用于数据中心资源管理**，提出了一个兼具 **理论深度与实践价值** 的预测-决策一体化框架，为构建 **可持续 AI 基础设施** 提供了重要范式。

</details>

---

### 4. [Efficient Adaptation of LLMs for Hate Speech Detection in Low-Resource Languages: A Comparative Study on Roman Urdu](https://arxiv.org/abs/2608.18142)

**Authors**: Toneema Zubair, Muhammad Junaid Asif, Faisal Kamiran, Hafiz Hassan Saeed, Rana Fayyaz Ahmad  
**Category**: cs.AI  
**Published**: 2026-08-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.18142v1  

#### Abstract
It is challenging to detect hate speech in Low Resource Languages (LRLs) because of the absence of annotated data, the informality of its language structure, and the lack of standardized grammar. A good example of such a challenge is Roman Urdu which is broadly used by South Asians on social media a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文聚焦于**低资源语言**（Low-Resource Languages, LRLs）中的**仇恨言论检测**（Hate Speech Detection, HSD），特别是针对**罗马乌尔都语**（Roman Urdu）这一极具挑战性的语言变体。Roman Urdu 是一种非标准化、拼写多变、语法不规范且广泛用于社交媒体的拉丁化乌尔都语，存在以下主要挑战：
- 缺乏大规模标注数据；
- 高度的拼写变异性和语音转写多样性；
- 严重的类别不平衡（毒性文本占比约18%）；
- 传统 NLP 方法难以捕捉其上下文语义。

现有研究在该领域存在明显空白：缺乏对零样本推理（zero-shot）、提示学习（prompt-based）和参数高效微调（PEFT）等范式的系统比较，尤其是在 LRLs 上的应用。

### 提出了什么新方法或新思路
论文提出了一种基于 **Parameter-Efficient Fine-Tuning**（PEFT）框架下的 **Low-Rank Adaptation**（LoRA） 方法，用于适配大型语言模型（LLMs）以进行 Roman Urdu 的仇恨言论检测。具体创新包括：
- 首次将 LoRA 技术系统应用于 Roman Urdu 这类极低资源语言的 HSD 任务；
- 构建了一个统一的评估框架，对比了多种主流 LLM 在 **zero-shot 推理** 与 **LoRA 微调** 下的表现；
- 对比了六种不同架构和规模的 LLM（Mistral-7B, LLaMA-3-8B, Falcon-7B, Gemma-2B, DeepSeek-R1-7B, multilingual BERT），探索模型大小与效率之间的权衡。

### 相比现有方法的优势
- **计算效率高**：仅更新少量可训练参数（约700万，占总参数极小比例），显著降低 GPU 内存消耗和训练成本；
- **避免过拟合**：冻结原始模型权重，仅训练低秩适配器，更适合小数据场景；
- **性能优越**：相比 zero-shot 和传统全量微调，在保持高性能的同时大幅提升资源利用率；
- **可扩展性强**：为其他低资源语言的 NLP 任务提供了可行的技术路径。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **数据集名称**：**PURUTT**（Parallel Urdu and Roman Urdu Corpus for Toxic Comments and Transliteration）
- **数据规模**：共 72,771 条社交平台评论
  - 毒性（Toxic）样本：13,097 条（≈18%）
  - 非毒性（Non-Toxic）样本：59,674 条（≈82%）
- **语言形式**：Roman Urdu（拉丁字母书写）
- **任务类型**：二分类（Binary Classification）
- **数据划分**：60% 训练集 / 20% 验证集 / 20% 测试集，保持类别分布一致
- **预处理**：文本清洗（URL、提及、特殊字符）、罗马乌尔都语规范化、分词、填充/截断至最大长度 512

### 实验设置和评估指标
#### 模型选择
测试了六种主流 LLM：
- Mistral-7B-v0.3
- LLaMA-3-8B
- Falcon-7B
- Gemma-2B
- DeepSeek-R1-7B
- BERT Multilingual

#### 微调方法
采用 **LoRA-based PEFT**：
- **LoRA Rank (r)**：16
- **Alpha (α)**：8
- **Dropout**：0.05
- **优化器**：AdamW
- **学习率**：1×10⁻⁴
- **Batch Size**：8
- **Epochs**：2
- **量化**：4-bit NF4（使用 bitsandbytes 库）

#### 评估指标
由于类别不平衡，重点关注 **F1-score**，同时报告：
- Accuracy
- Precision
- Recall
- F1-score

#### 基线方法对比
- **Zero-Shot Inference**：直接使用预训练 LLM 进行推理，无任何微调；
- **Full Fine-Tuning**：作为理论参照（未完全实施），强调 LoRA 的参数效率优势。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| Model | Zero-Shot F1 | PEFT (LoRA) F1 |
|-------|-------------|----------------|
| **Mistral-7B** | 0.56 | **0.9387** |
| **LLaMA-3-8B** | 0.21 | 0.9379 |
| **Gemma-2B** | 0.21 | 0.9089 |
| **Falcon-7B** | 0.32 | 0.9180 |
| **BERT Multilingual** | 0.15 | 0.8203 |
| **DeepSeek-R1-7B** | 0.51 | 0.7517 |

> ✅ **最佳表现**：**Mistral-7B + LoRA** 达到 **F1 = 0.9387**，是所有模型中最高。

### 与基线方法的对比结果
- 所有模型在 **zero-shot 设置下表现平庸甚至较差**（F1 最高仅 0.56），表明通用 LLM 虽具跨语言能力，但无法有效理解 Roman Urdu 的复杂性和语境；
- 经 **LoRA 微调后，所有模型性能大幅提升**，F1 提升幅度从 +0.24 到 **+0.73 不等**；
- 即使是较小的模型（如 Gemma-2B）也能通过 LoRA 实现接近大模型的性能（F1 > 0.90），说明 **PEFT 显著缩小了大小模型间的差距**；
- Mistral 和 LLaMA 系列表现最优，反映出其更强的语言理解和表示能力。

### 消融实验结果（隐含分析）
虽然未明确列出消融实验表格，但文中通过以下方式体现了关键因素的影响：
- **LoRA 超参数设置**（r=16, α=8, dropout=0.05）经过验证能实现稳定收敛；
- **4-bit 量化** 成功降低了显存占用而未显著损害性能；
- **类别加权**（class weighting）有效缓解了数据不平衡带来的偏差；
- 图表显示 fine-tuned 模型的预测分布更贴近真实标签分布，说明校准更好、偏见更低。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ❗ **Zero-shot LLMs 在 Roman Urdu 上效果有限**：尽管具备多语言知识，但无法准确识别该语言中的仇恨言论，F1 最高仅为 0.56；
2. ✅ **LoRA-based PEFT 显著提升性能**：仅微调极小部分参数即可使 F1 提升至 **>0.93**，证明其在低资源场景下的巨大潜力；
3. 🔍 **模型架构影响最终性能**：Mistral 和 LLaMA 表现优于 Falcon 和 BERT，说明基础模型的质量至关重要；
4. 💡 **小模型可通过 PEFT 实现“弯道超车”**：Gemma-2B 等轻量级模型经 LoRA 微调后仍能达到优秀性能（F1 > 0.90），适合部署于资源受限环境；
5. ⚙️ **LoRA 兼顾性能与效率**：大幅减少可训练参数数量（~7M），降低计算开销，适用于实际内容审核系统的部署。

### 方法的局限性
- **数据集单一**：实验仅基于 PURUTT 数据集，泛化性有待在其他 Roman Urdu 或低资源语言数据上验证；
- **任务简化**：采用二分类设定，未能捕捉讽刺、隐性仇恨、语气差异等更复杂的有害内容；
- **域偏见风险**：所用 LLM 主要在高资源语言上预训练，可能引入文化或语言偏见；
- **缺乏定性分析**：未深入探讨模型决策的公平性、可解释性或潜在偏见；
- **未探索跨语言迁移**：未尝试从高资源语言向 Roman Urdu 的知识迁移。

### 未来工作方向
- 将该框架扩展至其他低资源语言（如 Sindhi, Pashto 等）；
- 引入 **cross-lingual transfer learning** 和 **domain adaptation** 策略提升泛化能力；
- 探索更细粒度的任务设定，如多类别分类、隐性仇恨识别；
- 开展跨数据集评估，检验模型的鲁棒性与普适性；
- 结合人类反馈（human-in-the-loop）进一步优化标注质量与模型行为；
- 探索 LoRA 与其他 PEFT 方法（如 Adapter, Prefix-tuning）的组合效果。

---

> **总结一句话**：  
> 本论文证明了 **LoRA-based PEFT 是一种高效且强大的方法**，能够在极低资源条件下显著提升 LLM 在 Roman Urdu 仇恨言论检测中的性能，为构建可扩展、低成本的内容审核系统提供了切实可行的技术方案。

</details>

---

### 5. [Accelerating Visual On-Policy Distillation with Batched Speculative Jacobi Rollouts](https://arxiv.org/abs/2608.18183)

**Authors**: Bingqi Shan, Zhehao Yu, Kenhong Lin, Baoquan Zhang  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.18183v1  

#### Abstract
Visual on-policy distillation (OPD) improves the training of compact visual autoregressive models by learning from trajectories generated by the current student. However, these online rollouts are still produced token by token with autoregressive decoding, which adds substantial cost to every on-pol...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Accelerating Visual On-Policy Distillation with Batched Speculative Jacobi Rollouts**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
视觉 on-policy distillation（OPD）通过让学生模型（student）在训练中自生成轨迹并由教师模型（teacher）监督，提升了推理时的行为一致性。然而，这种在线 rollout 过程依赖于传统的 autoregressive（AR）解码，逐 token 生成后缀序列，导致每一步训练都引入大量顺序计算开销，显著拖慢训练速度。

### **提出的新方法：HB-SJD**
本文提出了 **Hybrid Batched Speculative Jacobi Decoding (HB-SJD)**，作为视觉 OPD 中学生 rollout 的高效替代方案。其核心思想是将原本用于单序列推理的 **Speculative Jacobi Decoding (SJD)** 扩展到大规模批量训练场景。

#### **关键创新点**：
- **无需辅助 draft model**：SJD 利用同一学生模型在 Jacobi 迭代中的历史预测作为 draft tokens，避免了额外训练 draft 模型的复杂性和同步成本。
- **独立进度控制（Independent Progress）**：允许每个图像根据自身解码进度独立推进，避免“快图等慢图”的同步瓶颈。
- **批量化验证（Batched Verification）**：尽管各图像处于不同位置，仍可通过 KV-cache 索引实现批量前向传播，保持 GPU 并行效率。
- **混合执行模式（Hybrid Full/Compact Execution）**：
  - **Full Execution**：保留完整 batch 形状，已完成图像逻辑上静默，适合高活跃度阶段。
  - **Compact Execution**：仅对活跃图像进行计算，减少冗余运算，在后期更高效。
  - 动态切换基于硬件校准阈值 `γ`，无需每次调参。

### **相比现有方法的优势**
| 方面 | 传统方法（Cached AR） | HB-SJD |
|------|------------------------|--------|
| 解码方式 | 逐 token autoregressive | 多 token 并行 proposal + 验证 |
| Draft Model | 不适用 | 无，复用自身历史输出 |
| 批处理效率 | 高但受限于顺序解码 | 高且支持异步进度 |
| 训练集成 | 原生支持 OPD | 即插即用，不改变 distillation objective 或优化流程 |

> ✅ **核心优势**：**在不修改任何 distillation 目标或训练流程的前提下，显著加速 rollout 过程，降低端到端训练时间，同时保持生成质量不变。**

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ImageNet 1K**：标准图像生成评估基准，用于训练与测试。

### **实验设置**
- **模型架构**：基于 **LlamaGen-B** 和 **LlamaGen-L** 构建学生与教师模型。
- **Distillation 方法**：
  - GKD（On-policy Knowledge Distillation）
  - VarKD（Visual Autoregressive KD with on-policy training）
- **Rollout 设置**：
  - Jacobi window size $ W = 16 $
  - History offset $ H = 24 $
  - Classifier-free guidance scale: 2.0
  - 使用 KV caching
- **硬件平台**：NVIDIA H200 GPU
- **重复次数**：主延迟结果取三次独立运行平均值

### **评估指标**
| 类别 | 指标 |
|------|------|
| **生成质量** | FID ↓, Inception Score (IS) ↑, Precision ↑, Recall ↑ |
| **效率指标** | Rollout Time (s) ↓, End-to-End Training Time ↓, Speedup × |
| **稳定性** | P95 Latency, Commit Length, Rejection Rate |

### **基线方法对比**
- **Cached AR**：带 KV-cache 的标准 autoregressive rollout（baseline）
- **GKD / VarKD + AR**：原始 on-policy distillation 方法
- **GKD / VarKD + HB-SJD**：本文方法替换 rollout 后端
- **验证策略对比**：
  - Greedy Verification
  - Probabilistic Verification（遵循原 SJD 规则）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| 方法 | FID ↓ | IS ↑ | Rollout Time (s) ↓ | Speedup × |
|------|-------|-----|--------------------|-----------|
| GKD + AR | 4.96 | 193.7 | 3.98 | 1.00× |
| **GKD + HB-SJD (Greedy)** | **4.95** | **192.8** | **2.68** | **1.48×** |
| VarKD + AR | 4.73 | 200.1 | 3.92 | 1.00× |
| **VarKD + HB-SJD (Greedy)** | **4.75** | **199.4** | **2.48** | **1.58×** |
| GKD-Large + AR | 2.99 | 251.6 | 9.82 | 1.00× |
| **GKD-Large + HB-SJD** | **3.00** | **251.4** | **6.28** | **1.56×** |

> 🔍 **结论**：HB-SJD 在所有配置下均实现 **1.48–1.65× 的 rollout 加速**，且 **FID、IS、Precision、Recall 几乎无损**。

### **与基线方法的对比结果**
- **端到端训练时间下降明显**：rollout 是训练关键路径，HB-SJD 缩短该环节直接带来整体训练提速。
- **生成质量完全可比**：视觉样例（Fig. 4）显示语义内容、物体结构、细节纹理均一致。
- **适用于多种 distillation 范式**：在 GKD 和 VarKD 上均有稳定增益。

### **消融实验结果**

| 实验主题 | 发现 |
|--------|------|
| **Independent Progress vs Sync**（Table 2） | 同步执行（Sync. Batch）反而比 AR 更慢（0.939×），而独立进度达 **1.216× speedup**，证明调度灵活性至关重要。 |
| **Hybrid vs Full/Compact Only**（Table 3） | Hybrid 比 Full 快 **1.182×**，比 Compact 快约 10%，说明动态切换最优。 |
| **Rollout Length 影响**（Table 4） | 后缀越长（prefix ratio ↓），加速比越高（从 1.392× → **1.699×**），表明 HB-SJD 对长序列更有效。 |
| **Batch Size 影响**（Table 5） | 在 B=8~64 全范围有效，最小加速 **1.298×**，最大 **1.486×**，具备良好扩展性。 |
| **Jacobi Window Size (W)**（Table 6） | 最优在 W=8（1.384×），过大（如 W=24）增加 verifier 成本，得不偿失。最终选 W=16 权衡提交率与延迟。 |
| **History Offset H 敏感性**（Table 7） | H=16~32 性能稳定，最优仅差 <1%，表明方法鲁棒。 |
| **Switching Threshold γ 敏感性**（Table 8） | γ=30~46 几乎无差异，支持轻量级硬件校准即可，无需精细调参。 |
| **训练阶段稳定性**（Table 9） | 在 2K、12K、22K 步均维持 ~1.5× speedup，说明适应学生策略演化。 |
| **Verification 策略比较**（Table 10） | Greedy 更快（1.613× vs 1.500×），因 rejection 更少、commit 更多；Probabilistic 更符合理论框架但开销略高。 |

---

## **4. 关键结论和发现**

### **主要发现**
1. **Speculative Jacobi Decoding 可成功迁移到视觉 OPD 场景**，利用学生自身历史预测作为 draft，无需额外模型。
2. **独立进度 + 批量验证** 是实现高效 batched SJD 的关键设计，打破了同步等待瓶颈。
3. **Hybrid Execution 显著提升尾部效率**，结合 Full 与 Compact 优点，实现全程高性能。
4. **HB-SJD 是即插即用模块**，仅替换 rollout backend，不影响 teacher、loss 或 optimizer，易于集成现有系统。
5. **加速效果随 rollout 长度增加而增强**，特别适合长序列生成任务。

### **方法的局限性**
- **依赖 KV-cache 实现跨图像位置索引**，对内存管理和调度系统有一定要求。
- **Verifier Forward 本身有开销**，若 window size 过大可能抵消并行收益（见 W=24 表现下降）。
- 当前主要针对 discrete token-based visual AR models（如 LlamaGen），是否适用于 continuous latent space 模型有待验证。
- Probabilistic Verification 虽兼容性强，但当前实现不如 Greedy 高效。

### **未来工作方向**
- 探索 **adaptive window sizing**：根据图像复杂度动态调整 W。
- 结合 **relaxed verification** 技术（如 LANTERN、GSD）进一步提高 acceptance rate。
- 将 HB-SJD 应用于 **video generation 或 multimodal rollout** 场景。
- 研究 **multi-GPU 分布式 HB-SJD** 实现更大规模 batch rollout。
- 探索与 **PathRelax、Parallel Jacobi Decoding** 等空间并行方法的融合。

---

> ✅ **总体评价**：  
> HB-SJD 是一项**实用性强、工程价值高的系统级优化**，它没有改变学习目标，却大幅降低了 on-policy distillation 的训练成本，为大规模视觉生成模型的高效训练提供了新路径。

</details>

---

### 6. [Scalable Geospatial Machine Learning for Power-Line Asset Risk: Integrating Remote Sensing for Lightning and Vegetation Risk Modelling](https://arxiv.org/abs/2608.18611)

**Authors**: Artur Sokolovsky, Bhavik Merai, Moe Jafari, Muen Chen  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.18611v1  

#### Abstract
Electric power networks are increasingly exposed to weather-sensitive failure mechanisms that require asset-level, spatially explicit risk modelling for effective intervention planning. This study contributes a modular, robust, and explainable probability-of-failure (PoF) modelling framework for uti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题**: *Scalable Geospatial Machine Learning for Power-Line Asset Risk: Integrating Remote Sensing for Lightning and Vegetation Risk Modelling*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
电力网络日益受到气候敏感型故障机制的影响，尤其是由**雷击（lightning）** 和**植被干扰（vegetation）** 引发的资产级失效。传统风险模型存在以下三大局限：
- 风险建模通常孤立处理不同灾害类型（如只考虑雷击或只考虑植被），难以支持综合决策；
- 空间分辨率过粗，无法支撑资产级别的精准干预（如修剪、加固）；
- 环境代理变量（proxies）与实际故障记录脱节，导致预测实用性差、可解释性弱。

### 🚀 提出的新方法与创新思路
本文提出了一种**模块化、可扩展、可解释的概率性失效建模框架（modular PoF modelling framework）**，其核心创新包括：

| 创新点 | 描述 |
|-------|------|
| **Modular multi-PoF formulation** | 构建统一的地理空间机器学习流水线，支持多个独立的 hazard-specific 模型实例（如 vegetation PoF 和 lightning PoF），实现跨灾害的一致性建模而不强制融合为单一模型。 |
| **高效且可扩展的建模流程（Efficient and scalable pipeline）** | 基于 Snowflake 云平台构建端到端 geospatial 数据处理流程，采用两阶段 geospatial pruning 技术显著降低计算开销，适用于百万级资产的大规模部署。 |
| **可扩展的数据与目标设计（Extensible data and target design）** | 支持灵活引入新的环境数据源（如 LiDAR、街景图像）、新增 PoF 类型（如动物接触故障）以及动态更新特征空间。 |
| **资产级别粒度输出（Operationally meaningful granularity）** | 输出每个电力资产的年度化失效概率（annualised PoF），直接用于巡检优先级排序、植被管理规划等运维场景。 |

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **操作可用性（Operational Maintainability）** | 模块化架构便于持续维护和迭代，适应不断变化的气候条件和资产管理需求。 |
| **可解释性（Explainability）** | 使用 SHAP 分析揭示各特征对风险预测的影响方向和程度，增强现场工程师的信任与采纳意愿。 |
| **多源异构数据集成能力** | 成功整合遥感（SRTM, MODIS NDVI, LIS VHRMC）、开放地图（OSM）与电网运营数据，形成全面的环境暴露表征。 |
| **标准化风险指标输出** | 将历史期失效概率转化为 annualised probability，便于横向比较和长期规划。 |

---

## 2. 核心实验方法和设置

### 📦 使用的数据集
| 数据类别 | 具体来源 | 用途 |
|--------|--------|------|
| **地形数据** | SRTM DEM (3 arc-second) | 提取海拔、坡度、地形粗糙度等 topographic 特征 |
| **植被状态** | MODIS/Terra MOD13Q1 NDVI (16-day, 250m) | 表征局部植被密度与生物量动态（mean NDVI, max NDVI change） |
| **雷电气候** | LIS VHRMC (0.1° gridded monthly climatology) | 区域尺度雷击频率统计作为 lightning exposure 指标 |
| **建成环境与水体** | OpenStreetMap (OSM) XML 数据 | 构建 coastline、inland water、building 的矢量几何，计算距离类 proximity 特征 |
| **故障标签** | 电网公司五年内的历史故障通知记录 | 构造 binary label：是否发生 vegetation/lightning-related 故障 |
| **资产元数据** | SAPN 内部资产数据库 | 包括电压等级（AGGREGATED_VOLTAGE_KV）、区域分类（SCONRR: Rural/Metro/CBD）、树冠覆盖标志（Tree Overhang Flag） |

> 所有数据通过以资产坐标为中心的多尺度窗口（200m ~ 50km）进行空间聚合。

### ⚙️ 实验设置
- **建模任务**：两个独立的二分类任务：
  - Vegetation PoF：判断某资产是否曾因植被引发故障
  - Lightning PoF：判断某资产是否曾因雷击引发故障
- **模型选择**：LightGBM（Light Gradient Boosting Machine）
  - 优势：支持大规模稀疏 geospatial 数据；原生处理 categorical 变量（如 SCONRR）；GOSS + EFB 加速训练
- **训练策略**：5-Fold Stratified Cross-Validation（确保正负样本在各 fold 中均衡分布）
- **特征工程优化**：
  - 动态 bounding box pruning 减少不必要的空间连接
  - OSM 原始节点重建为 WKT LINESTRING 以精确计算边缘距离
- **缺失值处理**：
  - 连续变量：均值插补（mean imputation）
  - 距离类变量无匹配对象时：赋值略大于搜索半径的常数（如 17,000m），表示“遥远”

### 📊 评估指标
由于故障事件高度不平衡（positive rate ≈ 0.1%~0.2%），仅用 Accuracy 易误导，故采用：
- **ROC-AUC**：衡量整体分类能力（threshold-agnostic）
- **PR-AUC（Precision-Recall AUC）**：更关注少数类（故障）的表现，在 imbalanced setting 下更具参考价值
- **Out-of-Fold (OOF) 预测**：避免数据泄露，提供无偏性能估计
- **SHAP 值分析**：用于模型可解释性分析，识别关键驱动因素

### ❌ 基线方法对比
文中未进行正式的跨模型家族基准测试（如 vs. XGBoost、Random Forest 或 Neural Networks）。作者明确指出这是出于**实用性和可维护性的权衡**，而非追求绝对最优性能。

> “We do not perform formal benchmarking against alternative model families... This choice prioritises a scalable, interpretable, and operationally maintainable framework.”

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
| 模型 | ROC-AUC (mean ± std) | PR-AUC (Average Precision) |
|------|------------------------|----------------------------|
| **Vegetation PoF** | **0.886 ± 0.014** | **0.139** |
| **Lightning PoF** | **0.898 ± 0.012** | **0.258** |

> 注：baseline prior 分别为 0.002（vegetation）和 0.001（lightning），表明 PR 曲线下方面积远高于随机猜测。

- 两个模型均表现出良好的判别能力，尤其在 ROC-AUC 上超过 0.88，说明能有效区分高风险与低风险资产。
- Lightning 模型 PR-AUC 更高，可能因其信号更强（如地形抬升效应明显）。

### 🔁 与基线方法的对比结果
尽管没有显式对比其他 ML 模型，但从以下方面体现优越性：
- 相较于传统的规则驱动或专家系统，本框架基于真实故障数据自动学习复杂非线性关系；
- 相较于单一灾害模型研究，实现了**双灾害并行建模**，支持交叉风险评估；
- 输出具备**地理一致性与物理合理性**（见 SHAP 分析），优于黑箱模型。

### 🔍 消融实验（Ablation Study）
文中未报告系统的消融实验（如移除某一类特征后的性能下降），但通过 SHAP 分析间接反映了各类特征的重要性。

---

## 4. 关键结论和发现

### ✅ 主要发现
#### （1）关键风险因子具有强可解释性且符合物理直觉
| 故障类型 | 最重要特征 | 发现解读 |
|---------|------------|----------|
| **Vegetation PoF** | `AGGREGATED_VOLTAGE_KV`, `DISTANCE_TO_WATER`, `IS_TREE_OVERHANG` | 低压线路、近水域区域、存在树冠覆盖的资产风险更高 —— 符合植被生长规律与运维经验 |
| **Lightning PoF** | `DISTANCE_TO_WATER`, `DISTANCE_TO_BUILDING`, `NEAREST_ELEVATION_M` | 地势较高、远离建筑物（缺乏屏蔽）、靠近水体的资产更易遭雷击 —— 与雷电物理学一致 |

#### （2）城市化程度（SCONRR）是重要调节变量
- Metro/CBD 区域的 lightning 风险较低，可能是由于建筑群提供了电磁屏蔽；
- 但 vegetation 风险反而略高，推测是因为城市管理限制了主动清障行为。

#### （3）地形与周边环境共同决定暴露水平
- 山地起伏大（high terrain ruggedness）反而降低 lightning 风险，平坦高地更危险；
- 接近 building 可提供 shielding 效应，降低 lightning 影响。

#### （4）NDVI 是有效的植被状态代理
- 高 mean NDVI 和近期快速 increase in NDVI 均与 vegetation 故障正相关，反映生物量增长带来的潜在威胁。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **缺乏时空隔离验证** | 使用的是 stratified CV，未严格划分地理区块或时间窗口，可能导致性能评估偏乐观（spatial autocorrelation 问题） |
| **NDVI 时间覆盖有限** | 仅使用最近一年的 MODIS 数据，未能捕捉多年植被周期或极端干旱滞后效应 |
| **未进行跨模型基准测试** | 未证明 LightGBM 在此任务上优于其他算法，结论依赖于特定模型选择 |
| **依赖高质量历史故障归因** | 若故障原因标注不准（如误判为 lightning 实为 vegetation），会影响监督信号质量 |

---

### 🔮 未来工作方向
| 方向 | 描述 |
|------|------|
| **向近实时监测演进（Near-real-time risk monitoring）** | 利用每16天更新的 MODIS NDVI 和实时 lightning 观测数据，动态刷新风险评分 |
| **视觉驱动特征提取（Vision-driven feature generation）** | 结合无人机/巡检照片，利用计算机视觉识别导线间隙、杆塔损坏、树枝侵入等精细状态 |
| **融合街景影像（Street-level panoramic imagery）** | 如 Google Street View，补充走廊级植被侵占与建成环境遮蔽信息 |
| **多模态建模范式探索** | 整合卫星遥感、LiDAR、tabular geospatial 与 street view 图像，构建统一的 multimodal asset risk framework |
| **跨区域迁移潜力** | 框架基于可观测环境变量而非地理位置编码，具备在不同气候区复用的可能性，尤其有助于应对气候变化下的新型风险组合 |

---

## 总结
该论文提出了一套**工业级可用的、模块化的 geospatial ML 框架**，成功将多源遥感与电网运营数据融合，实现了对 vegetation 和 lightning 两类关键灾害的资产级 PoF 建模。模型不仅性能优异（ROC-AUC > 0.88），而且具备高度可解释性与可扩展性，已接近实用化部署标准。虽然存在时空验证不足等学术局限，但在推动智能电网韧性管理方面具有重要实践意义。

</details>

---

### 7. [Which Negatives Matter? Ask Your Text Encoder: Adaptive Similarity Margins for Dense-Caption Retrieval](https://arxiv.org/abs/2608.18521)

**Authors**: Haoyue Liu, Ye Chen, Zhichao Wang, Xiaoying Tang  
**Category**: cs.AI  
**Published**: 2026-08-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.18521v1  

#### Abstract
Dense-caption retrieval has recently been improved by introducing segmentation, edge maps, LLM-filtered captions, and cross-modal modules into contrastive fine-tuning. However, these methods largely inherit the same InfoNCE objective, whose optimization can prematurely saturate under a strong pre-tr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Which Negatives Matter? Ask Your Text Encoder: Adaptive Similarity Margins for Dense-Caption Retrieval*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **dense-caption retrieval**（密集描述检索）任务中一个被忽视的优化问题：在使用强预训练模型（如 Long-CLIP）进行微调时，标准的 **InfoNCE 损失函数会迅速饱和**。

具体表现为：
- 在第一个 epoch 内，超过 80% 的批次损失已低于 $10^{-3}$；
- 在 fp32 精度下，47% 的测量中梯度变为精确零值。

这种“早熟饱和”导致模型虽然快速分离了大量简单负样本，但对决定最终性能（尤其是 R@1）的**高度相似的近重复负样本（near-duplicate captions）** 缺乏持续的学习动力，从而限制了性能上限。

### 提出了什么新方法或新思路
作者提出了一种名为 **HN-CLIP** 的新方法，其核心思想是：
> 利用文本编码器自身的 **text-text 几何结构** 来识别困难负样本，并为每个负样本分配自适应的相似性边距（adaptive similarity margin）。

具体实现方式如下：
- 在训练过程中，计算当前批次内所有 caption 之间的相似度矩阵 $G = t t^\top$；
- 将该矩阵从计算图中分离（`detach`），并屏蔽对角线（去除正样本）；
- 将其以系数 $\gamma$ 加到原始图像-文本相似度 logits 上，形成增强后的 logits：  
  $$
  \text{logits} = s(S + \gamma G)
  $$
- 这样，两个文本越相似的负样本对，需要更大的图像-文本相似度差距才能被正确区分，即获得了更大的 margin。

这种方法被称为 **per-negative adaptive margin**，具有以下特点：
- **无需负样本挖掘、合成或重采样**；
- **不引入额外参数、辅助输入或推理开销**；
- **仅需一次额外的 B×B 矩阵乘法和掩码加法**，训练效率高。

### 相比现有方法的优势
- **性能更强**：在四个主流 dense-caption 数据集上全面超越包括 GOAL、StructXLIP 等在内的先进方法，在 R@1 上提升 **+2.4 至 +4.3**。
- **训练更快**：相比 GOAL 快 **2.4×**，相比 StructXLIP 快 **5.4×**。
- **泛化性好**：可作为插件模块集成进多种 fine-tuning 框架（如 FineLIP、GOAL、LoRA 等），均能带来一致增益。
- **数据高效**：仅用 **20% 的训练数据** 即可超越全量数据训练的最强 baseline。

---

## 2. 核心实验方法和设置

### 使用的数据集
论文在四个 dense-caption retrieval 基准上进行了实验：

| 数据集 | 描述 |
|-------|------|
| **DOCCI** | 包含 15k 图像，每张配有详细人工描述（平均 123 词），强调细微差异。 |
| **DCI** | 7.4k 图像，带有与分割区域对齐的密集描述。 |
| **Long-DCI** | DCI 的扩展版本，使用完整长度的长文本描述进行评估。 |
| **Urban-1K** | 1k 图像的城市场景测试集，无训练集，用于跨域迁移评估（在 Visual Genome 上预训练后迁移）。 |

### 实验设置和评估指标
- **主干模型**：统一使用 **Long-CLIP-L (ViT-L/14)** 作为初始化模型。
- **训练配置**：
  - 批大小（effective batch）：128
  - 训练轮数：10 epochs
  - 优化器：AdamW，学习率 $2\times10^{-6}$，余弦退火
  - 超参数 $\gamma = 0.5$
- **评估指标**：
  - **Recall@K (R@K)**，其中 $K=1,5,10$，分别报告 **Text→Image** 和 **Image→Text** 两个方向的结果。

### 基线方法对比
与以下代表性方法进行比较：
- **Long-CLIP**：直接微调，无额外机制。
- **FineLIP**：引入 cross-modal 模块进行细粒度对齐。
- **GOAL**：利用 SAM 分割进行局部对象匹配。
- **StructXLIP**：结合边缘图和 LLM 提供的词法监督。
- **LoRA / DoRA**：参数高效微调（PEFT）方法。

HN-CLIP 与上述方法的区别在于：它不修改模型架构或增加辅助输入，而是**改进损失函数本身的设计**。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
在四个数据集上的 **R@1** 表现如下：

| Method | DOCCI (T→I) | DOCCI (I→T) | DCI (T→I) | DCI (I→T) | Long-DCI (T→I) | Long-DCI (I→T) | Urban-1K (T→I) | Urban-1K (I→T) |
|--------|-------------|-------------|-----------|-----------|----------------|----------------|---------------|---------------|
| **StructXLIP** | 84.73 | 82.61 | 75.84 | 74.49 | 75.34 | 72.30 | 87.60 | 87.80 |
| **HN-CLIP (Ours)** | **88.25** | **86.24** | **80.69** | **78.84** | **79.03** | **76.44** | **91.10** | **90.30** |
| **Δ (Improvement)** | **+3.52** | **+3.63** | **+4.85** | **+4.35** | **+3.69** | **+4.14** | **+3.50** | **+2.50** |

> ✅ **HN-CLIP 在全部 8 个检索方向上均取得最佳 R@1 性能**。

### 与基线方法的对比结果
- **显著优于 machinery-based 方法**：尽管 GOAL 和 StructXLIP 引入了复杂的辅助模块（如分割、边缘图、LLM 过滤），但 HN-CLIP 通过更有效的损失设计实现了更高性能。
- **训练速度优势明显**：
  - HN-CLIP：17 分钟完成一轮训练；
  - GOAL：~41 分钟；
  - StructXLIP：~92 分钟；
  > ⏱️ HN-CLIP 比 GOAL 快 **2.4×**，比 StructXLIP 快 **5.4×**。
- **样本效率极高**：在 DOCCI 和 DCI 上，仅用 **20% 的训练数据**，HN-CLIP 的性能就超过了 GOAL 或 StructXLIP 使用 100% 数据训练的结果。

### 消融实验结果（Ablation Studies）

#### （1）组件消融（Table 3b）
| 组件组合 | DCI (T→I R@1) | Long-DCI (T→I R@1) |
|---------|----------------|--------------------|
| Baseline (仅 InfoNCE) | 79.24 | 70.92 |
| + Token-level loss ($\mathcal{L}_{tok}$) | 79.39 | 70.24 |
| + HN margin ($\mathcal{C}_{HN}$) | 81.04 | 78.43 |
| **完整 HN-CLIP ($\mathcal{C}_{HN} + \mathcal{L}_{tok}$)** | **80.69** | **79.03** |

> 🔍 发现：**只有当 $\mathcal{C}_{HN}$ 存在时，token-level 对齐才变得有效**。说明“如何处理负样本”是瓶颈，而非“对齐粒度”。

#### （2）超参数 $\gamma$ 敏感性分析（Table 3a）
- $\gamma = 0.5$（默认）在多数任务上表现最优；
- 所有 $\gamma > 0$ 的设置都显著优于 $\gamma = 0$（即标准 InfoNCE）；
- 在跨域迁移任务（Urban-1K）上，较小的 $\gamma$（如 0.25）泛化更好，表明过强的 margin 可能导致过拟合。

#### （3）是否冻结 $G$ 的影响（Table 3c）
- 若将 $G$ 固定为初始编码器生成（不随训练更新），性能下降 **2.4–6.8 R@1**；
- 动态重新计算 $G$ 相当于一种**隐式的 margin 退火机制**，有助于避免后期过度约束。

---

## 4. 关键结论和发现

### 主要发现
1. **InfoNCE 在 dense-caption retrieval 中存在严重早熟饱和问题**，根源在于大量近似重复的 caption 导致困难负样本得不到足够优化信号。
2. **文本编码器自身的 text-text 相似性可以作为天然的“困难程度”指示器**，无需额外标注或复杂策略即可构建自适应 margin。
3. **HN-CLIP 不仅提升性能，还加速收敛**：首个 epoch 的性能已超过许多 baseline 的最终结果（见 Figure 5）。
4. **该方法具有极强通用性**：可无缝嵌入各种 fine-tuning 框架（包括 PEFT），并在所有测试框架中带来增益。

### 方法的局限性
- 当前方法依赖于 batch 内的 caption 相似性，若 batch size 太小可能导致负样本覆盖不足；
- margin 设计基于静态文本表示，未考虑图像模态的影响；
- 在极端跨域任务中，过强的 margin（大 $\gamma$）可能损害泛化能力。

### 未来工作方向
- 探索动态调整 $\gamma$ 的策略，例如根据训练阶段或数据分布变化进行调度；
- 将类似思想应用于其他多模态任务（如 video-text retrieval）；
- 结合 image-side 结构信息进一步增强 margin 设计；
- 研究如何在低 batch size 场景下保持有效性（如引入 memory bank）。

---

> 📌 **一句话总结**：  
> HN-CLIP 揭示了 dense-caption retrieval 中 InfoNCE 的优化瓶颈，并提出一种简洁而强大的解决方案 —— 利用文本编码器自身几何结构构建自适应 margin，实现了**更高性能、更快训练、更强泛化**的统一突破。

</details>

---

### 8. [Efficient INT8 Inference of Small NLP Models on Server CPUs with PyTorch Native Stack](https://arxiv.org/abs/2608.18182)

**Authors**: Weiwen Xia, Yuxin Cui, E Cao  
**Category**: cs.CL  
**Published**: 2026-08-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.18182v1  

#### Abstract
Small NLP models, especially BERT-family encoders, remain important in industrial workloads such as classification, ranking, and retrieval even in the era of large language models. On server CPUs, INT8 quantization offers an attractive latency-throughput-cost trade-off, but users increasingly expect...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient INT8 Inference of Small NLP Models on Server CPUs with PyTorch Native Stack

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
尽管大语言模型（LLMs）受到广泛关注，**small NLP models**（如 BERT-family 编码器）在工业界仍广泛用于分类、排序和检索等任务。这些模型通常部署在 **server CPUs** 上，因其成本低且基础设施普及。然而，如何在不依赖第三方工具的情况下，在原生 PyTorch 生态中实现高效的 **INT8 量化推理**，是一个尚未完全解决的挑战。

现有方案（如 IPEX、ONNX Runtime、Neural Compressor）往往依赖于非原生路径或需要额外部署流程，限制了易用性和可维护性。

---

### 🚀 提出的新方法与创新点

本文提出了一套完整的、**端到端集成于原生 PyTorch 栈**的 INT8 推理优化方案，主要贡献如下：

| 创新点 | 描述 |
|--------|------|
| **1. SmoothQuant 集成至 TorchAO** | 将 SmoothQuant 这一先进的后训练量化（PTQ）方法整合进 PyTorch 官方量化库 TorchAO，支持对激活异常值（activation outliers）进行平滑处理，提升 INT8 量化的精度稳定性。 |
| **2. TorchInductor 图级融合优化** | 在 TorchInductor 中新增图融合 pass，将 `torch._int_mm` + 缩放操作 + 偏置加法等子图融合为单个高效算子 `torch.ops.onednn.qlinear_pointwise`，消除运行时布局转换和后处理开销。 |
| **3. 多后端 INT8 GEMM 内核选择机制** | 支持在 oneDNN 提供的高性能内核与 TorchInductor 自研的模板化 GEMM 内核之间动态选择最优实现，适配不同矩阵形状。 |
| **4. 新增 AVX512_VNNI 微内核支持旧平台** | 为第3代 Xeon（Ice Lake）添加基于 AVX512_VNNI 的 INT8 GEMM 微内核，确保老平台也能受益于 INT8 加速。 |
| **5. s8s8 → u8s8 转换以兼容 Ice Lake** | 针对 Ice Lake 不支持有符号 INT8 × 有符号 INT8 运算的问题，引入 `X_u8 = uint8(int32(X_s8) + 128)` 并预计算补偿项，使 AVX512_VNNI 可用。 |

---

### 🔍 相比现有方法的优势

| 对比维度 | 本文方法 | 其他方法（如 IPEX、Neural Compressor、ONNX Runtime） |
|---------|----------|--------------------------------------------------|
| **生态一致性** | 原生 PyTorch，无需导出或切换框架 | 通常需导出模型或使用专用运行时 |
| **部署便捷性** | `torch.compile()` 即可启用，开箱即用 | 需要额外配置、插件或工具链 |
| **维护性** | 已合并至上游 PyTorch 和 TorchAO | 多为独立项目，更新滞后 |
| **硬件覆盖** | 同时支持 AMX（新平台）和 AVX512_VNNI（旧平台） | 多数仅聚焦最新架构 |
| **性能优化深度** | 图融合 + 内核选择 + 编译期常量折叠 | 多停留在算子替换层面 |

> ✅ **核心优势总结**：首次实现了 **“全原生” + “高性能” + “跨代 CPU 支持”** 的 INT8 推理流水线。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与模型

| 类型 | 名称 | 说明 |
|------|------|------|
| **下游任务数据集** | SQuAD 1.1, MultiNLI | 用于评估 fine-tuned 模型的准确率 |
| **模型族** | BERT-large, DistilBERT, XLM-RoBERTa | 代表典型的 small NLP encoder 模型 |
| **Checkpoint 来源** | Hugging Face Hub | 使用 task-finetuned checkpoint，保证实用性 |

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| **硬件平台** | <ul><li>**Ice Lake**（3rd Gen Xeon, 64 核，AVX512_VNNI）</li><li>**Granite Rapids**（6th Gen Xeon, 256 核，AMX）</li></ul> |
| **输入规格** | Batch size = 1, Sequence length = 256 |
| **部署方式** | 使用 AOTI（Ahead-of-Time Inductor），生成共享库，消除 Python 开销 |
| **启动策略** | Multi-instance：每实例绑定 4 个物理核，共享权重以最大化 LLC 利用率 |
| **量化模式** | <ul><li>**Smooth-Static**：静态量化（缩放因子固定）</li><li>**Smooth-Dynamic**：动态量化（每 batch 计算缩放因子）</li></ul> |
| **基线对比** | FP32（无量化）、BF16（Auto-Mixed Precision） |
| **评估指标** | <ul><li>**吞吐量（Throughput, RPS）**</li><li>**Linear-block 延迟（含 GEMM + 后处理）**</li><li>**准确率（F1, EM, Accuracy）**</li></ul> |

---

### 🧪 基线方法对比

| 方法 | 是否原生 PyTorch | 是否支持 SmoothQuant | 是否支持 AMX/VNNI | 是否图融合 |
|------|------------------|-----------------------|--------------------|------------|
| **本文方法** | ✅ 是 | ✅ 是 | ✅ 是 | ✅ 是 |
| IPEX | ❌（需导入） | ✅ 是 | ✅ 是 | ✅ 是 |
| Neural Compressor | ❌（需导出） | ✅ 是 | ✅ 是 | ⚠️ 有限 |
| ONNX Runtime | ❌（需导出） | ❌ 否 | ✅ 是 | ⚠️ 有限 |
| 原始 TorchAO + Eager Mode | ✅ 是 | ✅ 是 | ❌ 否 | ❌ 否 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 BERT-large 为例）

#### ✅ 在 **Ice Lake**（AVX512_VNNI）上的表现：

| 方法 | 吞吐速度提升（vs FP32） | Linear-block 延迟降低 |
|------|--------------------------|------------------------|
| Smooth-Dynamic | 2.22× | 3.59× |
| Smooth-Static | 2.58× | 3.87× |

> 💡 Ice Lake 上 INT8 GEMM 为 compute-bound，因此加速明显，接近理论天花板（~3.15×）的 **82%**。

---

#### ✅ 在 **Granite Rapids**（AMX）上的表现：

| 方法 | 吞吐速度提升（vs FP32） | Linear-block 延迟降低 |
|------|--------------------------|------------------------|
| Smooth-Dynamic | 5.02× | 9.92× |
| Smooth-Static | **5.82×** | 8.67× |

> 💥 最高达到 **5.8× 吞吐提升**，Linear-block 接近理论上限（11.13×）的 **89%**。

---

#### ✅ 与 BF16 的对比（Granite Rapids）：

| 方法 | 吞吐提升（vs BF16） | Linear-block 提升 |
|------|----------------------|---------------------|
| Smooth-Dynamic | 1.37× | 1.84× |
| Smooth-Static | **1.59×** | 1.61× |

> 即便在已启用 BF16 的先进平台上，INT8 仍能带来显著额外收益。

---

#### ✅ 所有模型平均表现（vs FP32）：

| 平台 | 模型 | Smooth-Dynamic | Smooth-Static |
|------|------|----------------|---------------|
| Ice Lake | 平均吞吐提升 | ~1.9–2.6× | — |
| Granite Rapids | 平均吞吐提升 | ~4.2–5.8× | — |

> 所有模型均实现 **4倍以上吞吐提升**，且 **静态量化优于动态量化**，因更多计算被前置。

---

### 🔍 消融分析与关键发现

| 分析项 | 发现 |
|--------|------|
| **图融合效果** | 融合 `qlinear_pointwise` 显著减少 kernel launch 和内存访问次数，是性能提升的关键。 |
| **内核选择机制** | 模板化 GEMM 在特定 shape 下优于 oneDNN，自动 benchmark 机制有效选出最优实现。 |
| **静态 vs 动态量化** | 静态量化吞吐更高，因其避免了运行时 scale 计算；动态略优精度但代价高。 |
| **AMX vs VNNI** | AMX 提供更高理论算力（524 TOPS vs 27 TOPS），是 Granite Rapids 高速的核心原因。 |
| **roofline 模型验证** | 实测性能达理论天花板的 **70%-89%**，表明优化充分，剩余差距来自量化开销和缓存行为复杂性。 |

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **成功构建了首个全原生 PyTorch 的高效 INT8 推理栈**，无缝集成 SmoothQuant、TorchAO、TorchInductor 和 oneDNN。
2. **在主流 server CPUs 上实现高达 5.8× 的端到端吞吐提升**，且 **准确率损失可忽略（<1%）**。
3. **通过图融合与内核选择机制，充分发挥现代 CPU 的 AMX/VNNI 指令集能力**。
4. **支持从 Ice Lake 到 Granite Rapids 的多代 Xeon 平台**，具备良好的向后兼容性。
5. **所有实现均已 upstream 至 PyTorch 和 TorchAO**，用户可通过 `torch.compile()` 开箱即用。

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **仅适用于 encoder-type 模型** | 当前优化针对 BERT-family，未覆盖 decoder-style LLMs（如 GPT）。 |
| **依赖 calibration 数据集** | SmoothQuant 需要小批量校准数据来计算 smoothing factors。 |
| **动态量化开销较大** | 特别是在 memory-bound 场景下，scale 计算可能抵消 GEMM 加速收益。 |
| **LLC 利用假设较理想化** | 实际应用中若 batch size 或 sequence length 更大，权重可能无法完全驻留 LLC。 |

---

### 🔮 未来工作方向

1. **扩展至 decoder-based LLMs**：结合 PagedAttention 和 KV Cache 量化，支持 vLLM/SGLang 类系统。
2. **支持 INT4 / FP8 等更低精度格式**：进一步压缩模型尺寸与带宽需求。
3. **自动化量化策略搜索**：基于模型结构和硬件特性自动选择 per-tensor/per-channel、static/dynamic 等配置。
4. **跨设备协同优化**：探索 CPU + GPU/NPU 协同推理下的量化调度策略。
5. **更精细的 kernel 自动生成器**：利用 Triton-like DSL 自动生成极致优化的 AMX/VNNI 内核。

---

> 📌 **一句话总结**：  
> 本文实现了 **PyTorch 原生栈中首个高效、通用、跨代支持的 INT8 推理方案**，在保持几乎无损精度的前提下，为 small NLP models 带来了最高 **5.8× 的吞吐提升**，推动了 CPU 上低成本、高性能 NLP 部署的落地进程。

</details>

---

### 9. [Multi-Agent Off-Policy Deep Reinforcement Learning for Smart Campus Coverage](https://arxiv.org/abs/2608.19049)

**Authors**: Omar Rady, Mohamed Ayman, Ali Arafa, Mohamed Shalma  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.19049v1  

#### Abstract
Deep reinforcement learning (DRL) has recently gained a great attention due to its real-time adaptation and effectiveness in complex optimization problems. This paper investigates the optimal deployment of millimeter-wave (mmWave) base stations (BSs) in a realistic, non-convex campus topology. The o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Multi-Agent Off-Policy Deep Reinforcement Learning for Smart Campus Coverage》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文研究在**非凸、复杂拓扑结构的校园环境**中，如何优化毫米波（mmWave）基站（BS）的部署位置，以实现**最大公平性**（max-min fairness）和**全覆盖**。该问题是NP-hard的，原因在于：
- mmWave信号对遮挡敏感，传播特性高度依赖空间几何；
- 校园建筑布局非对称、非凸，导致信号覆盖不连续；
- 需同时优化多个BS的位置，维度高，传统数学规划方法（如MINLP）计算开销大且难以收敛。

### **提出了什么新方法或新思路**
作者提出了一种**基于多智能体离策略深度强化学习**（Multi-Agent Off-Policy DRL）的框架，具体包括：
- 将BS部署问题建模为**马尔可夫决策过程**（MDP），并设计了以最小用户速率为目标的奖励函数；
- 系统性地比较了四种DRL方案：
  - 单智能体 Deep Q-Network (**Single-Agent DQN**)
  - 多智能体 Deep Q-Network (**Multi-Agent DQN**)
  - 单智能体 Deep Deterministic Policy Gradient (**Single-Agent DDPG**)
  - 多智能体 DDPG (**Multi-Agent DDPG**) —— **本文最优方案**
- 在**Multi-Agent DDPG**中引入：
  - **地理分区机制**：将四个建筑屋顶各自划分为5个子区域（北、南、东、西、中心），共20个分区；
  - **边界投影机制**（Boundary Projection）：确保动作输出始终落在合法区域内；
  - **独立训练 + 全局选择**：每个分区独立训练一个连续actor-critic agent，最终选择奖励最高的分区作为该建筑的最佳部署位置。

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **模型能力** | 连续动作空间（DDPG）优于离散网格搜索（DQN），避免分辨率限制； |
| **扩展性** | 多智能体架构缓解“维度灾难”，支持密集用户场景（400用户）； |
| **收敛效率** | 分区训练显著提升训练速度与稳定性（见图7）； |
| **实际适用性** | 考虑真实校园拓扑（GUC campus）、建筑遮挡、高度差异等现实因素； |
| **公平性保障** | 通过max-min目标函数最大化最差用户体验，Jain’s Index达0.94以上。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与环境**
- **仿真环境基于德国开罗大学**（GUC）的真实校园布局（见图1），包含四栋矩形建筑围合中央庭院；
- 用户设备（UE）分布在地面层，共 **K = 400** 名用户；
- 每栋建筑部署一个BS，需优化其2D坐标（x, y）；
- 所有实验基于 **Stable-Baselines3** 框架实现。

### **实验设置**
| 参数 | 设置值 |
|------|--------|
| 传输功率 $ P_x $ | 1.0 W |
| 噪声功率 $ N_0 $ | $ 1 \times 10^{-10} $ W |
| 路径损耗指数 $ \alpha $ | 4.0 |
| BS天线高度 | 12.0 m |
| UE高度 | 1.7 m |
| 学习率（DDPG/DQN） | $ 3 \times 10^{-4} $ |
| 折扣因子 $ \gamma $ | 0.99 |
| 经验回放缓冲区容量 | 50,000 |

### **评估指标**
- **最小速率**（Max-Min Rate, bps/Hz）：衡量系统公平性；
- **平均速率**（Mean Rate）；
- **最小SNR**（Signal-to-Noise Ratio, dB）；
- **覆盖率**（Coverage Percentage）：SNR > 0 dB 的用户比例；
- **Jain’s Fairness Index**（J）：量化用户间公平性，范围[0,1]，越接近1越公平；
- **训练收敛速度**（Training Time vs. Reward）。

### **基线方法对比**
| 方法 | 类型 | 动作空间 | 架构特点 |
|------|------|----------|-----------|
| Single-Agent DQN | 离散 | 全局离散网格 | 中心化决策，搜索空间大 |
| Multi-Agent DQN | 离散 | 分区离散候选点 | 每个agent负责一栋楼 |
| Single-Agent DDPG | 连续 | 全局连续空间 | 单一策略控制4个BS |
| **Multi-Agent DDPG** | **连续** | **分区连续空间 + 边界投影** | **本文提出，性能最优** |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自Table I）**

| 方法 | 用户数 | 平均速率 (bps/Hz) | 最小SNR (dB) | **Jain’s Index (J)** | 覆盖率 |
|------|--------|---------------------|---------------|------------------------|--------|
| Single-Agent DQN | 400 | 3.17 | 0.01 | 0.71 | <100% |
| Multi-Agent DQN | 400 | 9.05 | 14.01 | 0.82 | ~100% |
| Single-Agent DDPG | 400 | 9.64 | 19.19 | 0.94 | 100% |
| **Multi-Agent DDPG** | **400** | **9.65** | **19.18** | **0.9486** | **100%** |

> ✅ **Multi-Agent DDPG 实现了：**
> - **100% 覆盖率**（所有用户SNR > 0 dB）
> - **最小SNR > 19 dB**，远高于通信可用门限
> - **Jain’s Index高达0.9486**，接近完全公平
> - **最小可达速率达6.393 bps/Hz**（见图8）

### **与基线方法的对比结果**
- **Multi-Agent DDPG 显著优于其他三种方法**：
  - 相比 Single-Agent DQN：平均速率提升超 **200%**，公平性从0.71→0.94；
  - 相比 Multi-Agent DQN：最小SNR从14 dB → 19 dB，Jain’s Index提升15.7%；
  - 相比 Single-Agent DDPG：虽性能相近，但**收敛更快更稳定**（见图7）；
- **Multi-Agent 架构有效降低方差**，避免多个BS之间的覆盖冲突；
- **连续动作空间**使部署位置更精细，避免离散化带来的次优解。

### **消融实验分析（隐含于设计对比）**
虽然未明确标注“ablation study”，但通过以下对比实现了消融效果：
| 对比维度 | 发现 |
|---------|------|
| **单智能体 vs 多智能体** | 多智能体显著提升收敛速度与稳定性（图5、7） |
| **离散 vs 连续动作空间** | DDPG类方法全面优于DQN，尤其在最小SNR和公平性上 |
| **是否分区** | 分区机制使搜索空间局部化，避免全局优化陷入局部极小 |
| **边界投影机制** | 保证动作合法性，提升训练鲁棒性 |

---

## **4. 关键结论和发现**

### **主要发现**
1. **Multi-Agent DDPG 是解决复杂mmWave BS部署问题的最优架构**；
2. **地理分区 + 连续控制 + 边界投影** 的组合策略能有效应对非凸拓扑挑战；
3. 在 **400用户密度下仍能实现100%覆盖与高公平性**，验证了方法的可扩展性；
4. 多智能体框架具有**优异的计算收敛效率**，适合实时部署优化；
5. **max-min fairness目标函数**成功引导DRL探索最差用户的覆盖盲区。

### **方法的局限性**
- 当前仅适用于**静态用户分布**，未考虑移动性或动态流量变化；
- 假设BS只能部署在屋顶平面，未拓展至3D空间（如高度调整）；
- 依赖精确的用户位置信息作为状态输入，在隐私保护场景下可能受限；
- 模型训练成本较高，需大量episode才能收敛（>10万轮）。

### **未来工作方向**
- 扩展至 **动态用户场景** 和 **移动BS**（如UAV-mounted BS）；
- 引入 **部分可观测MDP**（POMDP）以处理用户位置不确定性；
- 结合 **RIS**（Reconfigurable Intelligent Surface）联合优化反射与部署；
- 探索 **联邦学习** 或 **分布式训练** 进一步降低通信开销；
- 应用于更大规模城市级网络规划任务。

---

> 🔚 **总结一句话**：  
> 本文通过提出一种**地理分区的Multi-Agent DDPG框架**，成功解决了**非凸校园环境中mmWave基站的公平覆盖部署难题**，在400用户场景下实现了**100%覆盖、最小SNR > 19 dB、Jain’s Index达0.9486**，为未来6G智能校园网络提供了高效、可扩展的自动化部署方案。

</details>

---

### 10. [Enhancing EBSD throughput of battery electrode materials using super-resolution generative adversarial networks](https://arxiv.org/abs/2608.19117)

**Authors**: John Mangum, Andrew Glaws, Francois Usseglio-Viretta, Steven Spurgeon, Donal Finegan  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.19117v1  

#### Abstract
Quantitative microstructural characterization of Li-ion battery electrode materials using electron backscatter diffraction (EBSD) has been proven as a critical method for optimizing cell performance. However, the inherently slow nature of EBSD can hinder the throughput of analyses needed for statist...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Enhancing EBSD throughput of battery electrode materials using super-resolution generative adversarial networks*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
电子背散射衍射（**EBSD**）是表征锂离子电池正极材料（如 NMC）微观结构的关键技术，可提供晶粒形貌、取向、晶界等高分辨率晶体学信息。然而，EBSD 是一种逐像素扫描的技术，采集速度慢，严重限制了其在统计代表性分析和工业高通量表征中的应用。

本研究旨在解决 **EBSD 数据采集效率低** 的瓶颈问题，尤其是在需要大视场、多颗粒统计分析或 3D-FIB-EBSD 等耗时场景下的吞吐量限制。

### 🚀 提出的新方法与创新思路
提出了一种基于 **超分辨率生成对抗网络（Super-Resolution Generative Adversarial Network, SRGAN）** 的深度学习框架，用于对低分辨率（Low-Resolution, LR）EBSD 图像进行计算增强，恢复出接近高分辨率（High-Resolution, HR）质量的图像。

**创新点包括：**
- 首次将 **SRGAN 框架应用于真实的 NMC 电极材料 EBSD 数据**，而非合成数据。
- 同时处理 **band contrast（灰度图）和 grain boundary label（二值图）**，采用混合损失函数（content loss + adversarial loss），实现物理一致的联合超分。
- 设计了更符合真实 EBSD 成像过程的 **随机采样下采样策略**（random sampling），避免传统平均下采样（averaging）导致的过度平滑，使训练数据更贴近实际粗步长采集情况。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | 本文方法优势 |
|------|--------|-------------|
| **经典插值法**（Nearest Neighbor, Linear, Cubic） | 导致模糊、伪影、晶界增宽、小晶粒丢失 | 显著保留细小晶粒、锐利晶界、真实微观结构细节 |
| **Unsharp Masking** | 能增强边缘但无法生成亚像素细节 | 可“生成”合理的小尺度特征，提升视觉与定量准确性 |
| **已有深度学习方法（如 SRResNet）** | 多基于合成数据，未考虑真实噪声与漂移；仅用 L2 损失，图像偏模糊 | 引入 GAN 的对抗训练机制，生成更逼真纹理；使用真实实验数据训练，更具实用性 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **材料系统**：商用 **LiNi₀.₈Mn₀.₁Co₀.₁O₂ (NMC811)** 正极颗粒。
- **样品制备**：通过 Ar⁺ 离子束抛光获得截面，使用 **FEI Nova NanoSEM 630 + Oxford Symmetry EBSD 探测器** 采集数据。
- **双分辨率采集**：
  - **高分辨率（HR）**：25 nm/像素，单图约需 16 小时。
  - **低分辨率（LR）**：100 nm/像素，单图约需 1 小时（即 16× 快速）。
- **样本数量**：共采集 **15 个独立粒子** 的配对 HR/LR 数据用于训练与验证。

### ⚙️ 实验设置
- **下采样方式**：从 HR 数据人工下采样生成 LR 输入，模拟不同步长采集条件（2× 到 12× 下采样）。
  - 对偶数倍率（如 2×, 4×）采用 **2×2 区域中随机选点**；
  - 对奇数倍率（如 3×, 5×）采用 **3×3 区域中随机选点**。
- **上采样目标**：将 LR 数据上采样回原始 HR 分辨率（即 2× 到 12× 超分）。
- **模型架构**：
  - **Generator**：10 层 CNN + Pixel Shuffle 上采样层。
  - **Discriminator**：7 层 CNN + 3 层全连接，输出真假判别。
  - **损失函数**：Content Loss（Band Contrast 用 MSE，Boundary 用 Binary Cross Entropy）+ Adversarial Loss。
- **训练策略**：留一法交叉验证（Leave-one-out），每轮训练一个模型，测试集为被留出的粒子。

### 📈 评估指标
#### （1）通用图像质量指标（context-agnostic）
| 指标 | 描述 |
|------|------|
| **NRMSE** | 归一化均方根误差，越小越好 |
| **PSNR** | 峰值信噪比，越大越好 |
| **SSIM** | 结构相似性指数，衡量亮度、对比度、结构一致性，范围 [-1,1]，越接近 1 越好 |
| **NMI** | 归一化互信息，反映像素分布相关性，范围 [1,2]，越大越好 |

#### （2）微结构专用量化指标（after segmentation）
| 指标 | 描述 |
|------|------|
| **Grain Count Retention** | 恢复出的晶粒数量占比 |
| **Grain Size Accuracy**：<br>- `DA`（Area-equivalent diameter）<br>- `DS`（Max inscribed sphere diameter） | 衡量晶粒尺寸保真度 |
| **Grain Shape Metrics**：<br>- Circularity (ξ)<br>- Solidity (ψ) | 描述晶粒几何形状 |
| **Grain Boundary Length Density** | 单位面积内晶界总长度，反映裂纹敏感区域密度 |
| **Image Quality (IQ)** | 定义为 $1 - \frac{\text{晶界像素数}}{\text{晶粒+晶界像素总数}}$，越高表示图像越精细 |
| **Grain Boundary Connectivity** | 晶界网络是否连通（percolation），以最大簇占比衡量 |

### 🔁 基线方法对比
- Nearest Neighbor 插值
- Linear 插值
- Cubic 插值
- Unsharp Masking
- （额外对比）作者也尝试将 SRGAN 应用于真实采集的 100 nm 数据（即 4× 下采样），验证方法泛化能力。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 指标 | SRGAN 性能 | 基线方法典型表现 |
|------|-----------|------------------|
| **最大实用超分倍率** | **5×**（对应 **25× 加速** 或 **25× 更大视场**） | 所有经典方法在 >4× 后严重退化 |
| **晶粒数量保留率 @5×** | ~44%（远高于基线的 ~16%） | <20% |
| **晶粒尺寸误差 @5×** |<br>- `DA`：+5.7%<br>- `DS`：+8.2% | 普遍低估后转为高估，误差 >20% |
| **晶界长度密度保留率 @3×** | 93.0% (@2×), 80.1% (@6×) | <60%（因边界增宽导致面积膨胀） |
| **图像质量 IQ @5×** | 维持在 ~0.59（HR 原始为 0.645） | 下降至 ~0.43（下降近 33%） |
| **晶界连通性** | 在 ≤6× 保持良好连通性 | 在 ≥4× 出现断裂 |

### 🆚 与基线方法对比结果
- **定性视觉对比（Fig. 7 & 8）**：
  - 在 4× 和 8× 超分下，所有经典方法均出现明显伪影（块状化、模糊、晶界变粗），而 SRGAN 仍能重建合理的晶粒结构。
  - SRGAN 输出的 band contrast 与 boundary map 具有一致性，而传统方法两者脱节。
- **定量指标全面领先（Fig. 9）**：
  - 在 NRMSE、PSNR、SSIM、NMI 所有指标上，SRGAN 在所有放大倍数下均显著优于基线方法。
  - 特别是在 >4× 后，差距急剧拉大。
- **微结构保真度（Fig. 10–15）**：
  - SRGAN 更好地保留了 **小晶粒** 和 **窄晶界宽度**。
  - 在 **5× 内**，晶粒大小、形状（circularity, solidity）、数量等关键参数误差可控（<10%）。
  - 晶界长度密度虽随分辨率降低而下降，但 SRGAN 下降最缓慢。

### 🔍 消融实验与关键发现
- **下采样方式影响显著**（Fig. 3）：
  - 若使用平均下采样（averaged），会导致 LR 数据过平滑，SRGAN 学习错误映射。
  - 改用 **随机采样（randomized）** 后，semivariogram 显示其空间变异性与真实 LR 数据高度一致，确保训练有效性。
- **真实 LR 数据验证**（Fig. 9 中 “×” 符号）：
  - 将训练好的 SRGAN 应用于 **真实采集的 100 nm 数据（4× 下采样）**，其性能与人工下采样的结果几乎重合，证明方法可直接用于实际快速采集流程。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **SRGAN 可有效提升 EBSD 吞吐量**：通过采集 **125–150 nm 步长** 的低分辨率数据并用 SRGAN 超分至 25 nm，可在 **不牺牲关键微结构精度的前提下实现最高 25× 的加速**。
2. **5× 超分为最优平衡点**：在此倍率下，晶粒尺寸、形状、数量等核心指标相对误差控制在 **+5.7% 至 +8.2%**，满足大多数电池材料研究需求。
3. **优于所有经典方法**：无论在图像质量还是微结构保真度方面，SRGAN 均系统性超越 nearest neighbor、linear、cubic 和 unsharp masking 等传统手段。
4. **适用于真实工业场景**：方法已在真实 NMC811 材料上验证，并成功用于 **大视场（~35 颗粒子）快速成像**（40 分钟 vs 传统 11 小时），展示其工程潜力。

### ⚠️ 方法局限性
- **依赖高质量训练数据**：模型性能受限于训练集的多样性（如不同 NMC 组分、老化状态、制备工艺）。
- **极端超分（>8×）失效**：当输入过于粗糙时，GAN 会“幻想”结构，导致晶界断裂、晶粒合并等非物理解。
- **未建模采集噪声与漂移**：当前训练基于理想对齐数据，实际运行中 stage drift 或信号噪声可能影响效果。
- **仅针对 2D 截面**：尚未扩展到 3D-FIB-EBSD 的体数据重建。

### 🔮 未来工作方向
- **拓展至其他材料体系**：如 Si 负极、固态电解质、燃料电池催化剂等。
- **结合 in-situ EBSD**：利用超分提升时间分辨率，实现实时动态演化监测。
- **推动 3D 微结构生成**：将 2D SRGAN 输出作为先验，辅助 3D 重构或训练 3D-to-3D 超分模型。
- **集成到自动化表征平台**：嵌入 EBSD 控制软件，实现“快速扫描 + 实时超分”闭环。
- **探索扩散模型（Diffusion Models）**：替代 GAN，可能生成更稳定、多样化的微结构细节。

---

> **一句话总结**：  
> 本文提出并验证了一种基于 **SRGAN 的 EBSD 超分辨率框架**，可在 **5× 上采样（25× 加速）** 条件下 **系统性优于传统方法**，显著提升电池电极材料微结构表征的吞吐量与统计代表性，为高通量材料研发提供了新工具。

</details>

---

### 11. [RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training](https://arxiv.org/abs/2608.18682)

**Authors**: Yugu Li, Jimmy Cao, Jianglin Qiao, Siyi Hu  
**Category**: cs.AI  
**Published**: 2026-08-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.18682v1  

#### Abstract
Training multi-turn agentic workflows with reinforcement learning (RL) enables large language models to perform complex reasoning, use external tools, and conduct iterative search beyond single-turn settings. Yet multi-turn RL training remains highly unstable, often causing severe performance degrad...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**多轮智能体强化学习**（multi-turn agentic RL）训练中的严重不稳定性问题。随着交互轮次增加，模型性能常出现显著下降。作者通过理论分析指出，这种不稳定性源于三个紧密耦合的根源：
- **Rollout-Training Context Mismatch**（回放-训练上下文不匹配）：在生成轨迹（rollout）时通常使用截断或摘要的上下文以提高效率，而在训练时却重新计算完整历史的似然比，导致重要性采样（IS）比率有偏。
- **Weak Turn-Level Credit Assignment**（微弱的回合级信用分配）：现有方法将单一的轨迹级优势（trajectory-level advantage）均匀分配给所有回合，无法准确反映每个决策对最终结果的真实贡献。
- **Asynchronous Policy Drift**（异步策略漂移）：在并行采样长、短轨迹的异步训练中，短轨迹先更新策略后，长轨迹仍在旧策略下生成，导致严重的off-policy偏差。

### 提出的新方法
为解决上述问题，作者提出了**Reverse-Turn Policy Optimization**（RTPO），一种统一的反向回合策略优化框架。其核心思想是：
- 将多轮交互建模为稀疏的**反向树结构**（sparse reverse trees）。
- 以**时间逆序**（temporal reverse order）进行回合级策略更新，即从最后一个回合开始，逐个向前更新。
- 在每个回合 `k`，基于当前边界状态 `Sk` 生成多个**兄弟回放**（sibling rollouts），并在下游策略已优化的前提下，估计该回合的准确优势值。

### 相比现有方法的优势
- **结构性一致性**：RTPO通过在回放和训练中使用完全相同的条件上下文 `ck`，从根本上消除了Rollout-Training Mismatch。
- **因果一致的信用分配**：通过在相同边界状态下比较不同动作的兄弟回放，实现了无上游状态污染的回合级信用分配。
- **控制策略漂移**：采用**on-policy continuation**机制，在每次更新回合 `k` 时，同步最新的策略参数来生成其下游的兄弟回放，从而避免了长距离off-policy校正的高方差问题。
- **理论保证**：提供了严格的理论证明，表明RTPO能消除上下文不匹配和策略漂移，并收敛到递归最优解。

## 2. 核心实验方法和设置

### 数据集
实验涵盖了两类典型的工具使用任务：
- **数学推理**（Mathematical Reasoning）：使用 **MATH** 数据集进行训练，并在 **GSM8K**, **MATH500**, **AMC23**, **AIME24**, **AIME25**, **OE-Math** 等基准上进行零样本评估。
- **知识推理**（Knowledge Reasoning）：使用一个由 **SimpleDeepSearcher** 和 **WebSailor** 构建的硬搜索数据集进行训练，并在 **HotpotQA**, **2WikiMultiHopQA**, **GAIA**, **WebWalkerQA**, **HLE**, **xBench** 等基准上评估。

### 实验设置和评估指标
- **基础模型**：主要使用 **Qwen3-8B**，在部分消融实验中使用 Qwen3-4B。
- **训练框架**：基于 **VeRL** 框架实现，使用 **vLLM** 进行回放生成，**FSDP** 进行分布式训练。
- **评估指标**：
  - **主性能**：数学任务使用 **Pass@1**，知识任务使用 **best-span F1**。
  - **训练稳定性**：通过回放-训练阶段的token log-probability比率和KL散度来衡量。
  - **工具使用**：统计平均工具调用次数（tool calls）。
  - **策略漂移**：通过on-policy与off-policy输出的命中率（output-hit accuracy）对比来衡量。

### 基线方法对比
- **轨迹级方法**：**GRPO**（代表性的PPO变种）。
- **回合/树级信用分配方法**：**SeeUPO**（序列级代理RL）、**ARPO**（自适应树策略优化）、**TreeGRPO**（树形GRPO）。

## 3. 主要实验结果和性能指标

### 关键性能数据
在8个工具使用的代理RL基准测试中，RTPO取得了全面领先：
- 相比非RL的**vanilla**模型，整体准确率提升 **66.78%**。
- 相比强基线**GRPO**，整体准确率提升 **21.50%**。
- 相比回合级方法**SeeUPO**，整体准确率提升 **10.76%**。

### 与基线方法的对比结果
- **训练稳定性**：如图3所示，RTPO在整个训练过程中保持了近乎完美的回放-训练一致性（log-probability比率稳定在1.0），而GRPO和SeeUPO则存在明显波动和偏差。
- **信用分配有效性**：在控制了上下文不匹配的消融实验中（表2），RTPO-CA在所有数学推理基准上均达到最佳，尤其是在最难的AIME25上表现突出，验证了其回合级信用分配的有效性。
- **策略漂移修正**：在长距离搜索任务上的对比（表3）显示，标准RTPO（on-policy）相比其复用旧回放的off-policy变体，在GAIA、XBench和WebWalkerQA上分别提升了 **+5.83%**, **+7.00%**, **+3.50%** 的命中率，证明了on-policy续接的巨大优势。

### 消融实验结果
- **兄弟组大小**（Sibling Group Size）：实验表明，增大兄弟组大小 `G` 可以减少零优势信号的比例，提高学习信号密度，但会带来更高的计算成本。`G=3` 是默认的性价比配置。
- **工具使用模式**：RTPO展现出双向自适应能力。在数学任务上，它学会了更可靠的内部推理，**减少了不必要的工具调用**；在知识任务上，它学会了进行更密集的检索，**增加了有效的工具调用**，这表明其优势在于学习与任务本质相匹配的策略，而非简单地鼓励或抑制工具使用。

## 4. 关键结论和发现

### 主要发现
1. **根本原因**：多轮代理RL的不稳定性并非孤立问题，而是源于“扁平化轨迹优化”这一共同结构根源下的三个耦合因素。
2. **统一解决方案**：RTPO通过“反向回合”的统一框架，系统性地解决了这三个问题，实现了**回放-训练一致性**、**因果一致的信用分配**和**受控的策略漂移**。
3. **实证有效**：大量实验表明，RTPO不仅能显著提升最终性能，还能极大增强训练过程的稳定性，尤其在长视野、高难度的任务上优势更为明显。

### 方法的局限性
1. **对主干质量的依赖**：RTPO以主干轨迹（trunk）的边界状态作为分支锚点。如果主干在早期做出错误决策，后续的优化可能会被限制在低价值区域。
2. **训练开销**：相比单次优化的扁平化方法，RTPO需要为每一轮进行独立的反向优化和兄弟回放生成，带来了额外的计算开销（实验显示约增加41.3%的GPU小时）。
3. **训练Token利用率低**：为了保证因果一致性，只有当前回合的输出Token参与梯度更新，兄弟回放中大量的下游Token被丢弃，造成了一定程度的资源浪费。

### 未来工作方向
- **多主干采样**：生成多个主干轨迹并选择高质量的作为锚点，以改善边界状态的覆盖范围。
- **高效调度**：设计更高效的兄弟回放生成和更新调度策略，以降低训练开销。
- **辅助目标**：探索在下游Token上引入辅助目标（如语言建模损失），以提高训练Token的利用效率，同时保持核心优势的因果一致性。

</details>

---

### 12. [MAVEN: A Macro-Societal Value Evaluation Framework of Multimodal Content with Compact Aligned Evaluators](https://arxiv.org/abs/2608.18096)

**Authors**: Zijuan Zhao, Zheren Fu, Hou Xia, Licheng Zhang, Yi Liu, Zhendong Mao  
**Category**: cs.CL  
**Published**: 2026-08-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.18096v1  

#### Abstract
Assessing whether multimodal content aligns with macro-societal values, such as peace, justice, and freedom, has become an increasingly urgent challenge. Existing frameworks are largely confined to safety-oriented taxonomies, text-only psychometric probes, or single-label classification. Therefore, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前对多模态内容（如图文新闻、社交媒体内容）进行**宏观社会价值（macro-societal values）评估**存在三大挑战：
- 多数框架聚焦于**安全导向（safety-oriented）** 的伦理问题（如毒性、偏见），忽视了和平、正义、自由等更广泛的社会价值。
- 现有基准多为**文本单一模态**的心理测量问卷，无法直接应用于真实世界的图像-文本内容。
- 评估方式常为**单标签分类或开放式评论**，缺乏细粒度、可量化的评分机制。

### **提出的新方法与新思路**
本文提出了 **MAVEN**（Multimodal Macro-societal Value Evaluation Network），一个层次化、可量化的多模态价值评估框架，其核心创新包括：

#### **(1) MAVEN 框架设计**
- **理论基础**：基于国际人权文件（如《联合国宪章》《世界人权宣言》）和跨文化价值观理论（Schwartz’s Value Theory）构建。
- **两层结构**：
  - **6个主维度**（Primary Dimensions）：Peace, Development, Equity, Justice, Democracy, Freedom。
  - **72个次级指标**（Secondary Indicators）：每个维度下设12个具体可操作的子项（如“World Peace”、“Freedom of Speech”）。
- 支持**多层次定量评分**：
  - 主维度：5分制（-2 到 +2）
  - 次级指标：3分制（-1, 0, +1）

#### **(2) 宏观价值基准 MacroValue-Bench**
- 包含 **1,157 个经人工验证的图像-文本对**，来源涵盖微博、环球时报、人民日報等。
- 采用“模型初标 + 人类复核”的标注流程，确保高质量标签。

#### **(3) 新评估指标 VSMS**
- **Value-aware Soft Match Score (VSMS)**：考虑不同维度间语义相关的软匹配指标。
- 允许在语义相近的预测上给予部分信用（partial credit），避免传统硬匹配（hard-match）忽略语义耦合的问题。

#### **(4) 轻量化评估器优化方法**
- **SA-MDPO**（Span-Adaptive Multi-level DPO）：
  - 一种改进的偏好优化算法，引入**跨度自适应正则化系数**，更好地利用多级排序信号。
- **MRC**（Multi-Role Consensus）：
  - 推理时无需训练的共识策略，通过让同一VLM以14种不同利益相关者角色（如法官、记者、医生）作答，再聚合投票提升判断鲁棒性。

### **相比现有方法的优势**
| 维度 | MAVEN | 现有方法（如 ETHICS, VIVA, ValueBench） |
|------|-------|----------------------------------------|
| **Macro-societal focus** | ✅ 明确覆盖宏观社会价值 | ❌ 多为个体伦理或安全问题 |
| **Multimodal support** | ✅ 图像-文本联合评估 | ❌ 多为纯文本 |
| **Quantitative scoring** | ✅ 连续/有序评分 | ❌ 多为分类或开放生成 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MacroValue-Bench**：
  - 规模：1,157 条人工验证的图像-文本对。
  - 来源分布：
    - Weibo（49.5%）
    - Global Times Online（21.2%）
    - People's Daily Online（14.3%）
    - Ch3Ef-harmless 数据子集（15.0%）
- **训练集**：
  - 构造了一个 **3,865 条**的非人工验证训练集，用于 SA-MDPO 蒸馏。

### **实验设置**
- **评估对象**：
  - 开源 VLMs：Qwen3-VL 系列（2B, 4B, 8B）、GLM-4.5V
  - 商业闭源 VLMs：GPT-4o, GPT-5, Gemini-2.5-Pro, Doubao-Seed-1.6-V 等
  - 自研模型：Qwen3-VL-2B + SA-MDPO + MRC
- **训练细节**：
  - 使用 LoRA 微调 Qwen3-VL-2B
  - SA-MDPO 参数：`β₀=0.1`, `α=0.5`, `K=4` 层偏好数据
  - 批大小 16，4 个 epoch，在单张 A800 GPU 上完成

### **评估指标**
| 指标 | 描述 |
|------|------|
| **QWK**（Quadratic Weighted Kappa） | 主要指标，衡量有序评分的一致性，对大偏差惩罚更重 |
| **Accuracy** | 分类准确率 |
| **F1_macro** | 宏平均 F1 分数 |
| **VSMS** | 考虑语义相似性的软匹配得分（Recall/Precision/F1） |

### **基线方法对比**
- **未优化基线**：原始 Qwen3-VL-2B
- **DPO / MDPO**：标准二元或多层次偏好优化
- **商业模型**：GPT-4o, GPT-5, Gemini, Doubao 等作为强基线

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 4）**

| 模型 | QWK ↑ | Acc ↑ | F1 ↑ | VSMS ↑ |
|------|--------|--------|--------|--------|
| **Qwen3-VL-2B (baseline)** | 0.393 | 0.497 | 0.300 | 0.798 |
| **+ SA-MDPO** | **0.599** (+52.4%) | **0.706** (+42.1%) | **0.482** (+60.7%) | **0.954** (+19.5%) |
| **+ SA-MDPO + MRC** | **0.624** | **0.719** | **0.497** | **0.959** |
| **Doubao-Seed-1.6-V**（最强商业） | 0.702 | 0.756 | 0.524 | 0.966 |
| **GPT-5** | 0.637 | 0.724 | 0.520 | 0.962 |

> ✅ **核心发现**：经过 SA-MDPO 蒸馏后的 **2B 小模型**，性能接近甚至超过同系列的 **8B 模型**，并逼近 GPT-5 和 Gemini 等前沿闭源模型。

### **与基线方法的对比结果**
- **参数效率极高**：2B 模型仅用 1/4 参数即达到 8B 模型水平。
- **超越部分商业模型**：在 Accuracy 和 F1 上优于 GPT-4o 和 Qwen3-VL-Plus。
- **接近最优水平**：QWK 达到 0.624，距离最强模型 Doubao-1.6V（0.702）差距可控。

### **消融实验结果**

#### **(1) 对齐目标消融（Table 5）**
| 方法 | QWK | VSMS |
|------|-----|------|
| DPO (unbalanced) | 0.153 | 0.517 |
| MDPO (balanced) | 0.445 | 0.964 |
| **SA-MDPO (balanced)** | **0.599** | **0.954** |

> ✅ **SA-MDPO 显著优于 DPO 和 MDPO**，证明跨度自适应调度能更好捕捉多级偏好信号。

#### **(2) MRC 消融（Table 6）**
| 模型 | QWK（×14 runs） | QWK（MRC） | 提升 |
|------|------------------|-------------|------|
| Qwen3-VL-2B | 0.405 | 0.432 | +6.7% |
| Qwen3-VL-4B | 0.404 | 0.424 | +5.0% |
| Qwen3-VL-8B | 0.526 | 0.540 | +2.7% |

> ✅ **MRC 在所有规模上均有效**，且对小模型增益最大，说明多角色视角尤其有助于缓解小模型判断偏差。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **宏观社会价值可以被系统化建模**：MAVEN 提供了一个理论扎实、可扩展的评估体系。
2. ✅ **轻量模型可通过蒸馏媲美大模型**：使用 SA-MDPO 可将 2B 模型能力从 0.393 QWK 提升至 0.624，实现“小模型大能力”。
3. ✅ **多角色推理显著提升一致性**：MRC 是一种低成本、高收益的推理增强策略。
4. ✅ **现有 VLMs 在价值判断上存在明显差异**：即使是 GPT-4o 也在 Democracy 等维度表现不佳，表明价值对齐并非随规模自然涌现。

### **局限性**
1. **文化覆盖有限**：
   - 框架和数据主要基于中国语境（如 Weibo、Global Times），可能不完全适用于其他文化背景。
   - 人类标注员均为东亚背景，可能存在隐性偏见。
2. **MRC 推理成本高**：
   - 需运行 14 次前向推理，计算开销约增加 14×，不适合高吞吐场景。
3. **教师模型偏差传递**：
   - 训练数据来自三个前沿 VLMs，其内在价值倾向可能影响最终评估结果。

### **未来工作方向**
- 🌍 **跨文化验证与适配**：在更多地区收集数据，支持多语言、多文化版本的 MAVEN。
- ⚙️ **优化 MRC 效率**：探索角色选择、前缀共享或知识蒸馏回单次推理的方法。
- 🔁 **动态更新指标体系**：允许社区参与修订或扩展次级指标，保持框架灵活性。
- 🛡️ **防止滥用机制**：明确声明 MAVEN 应用于诊断而非审查，避免被用于压制言论自由。

---

> **代码与资源公开**：  
> - GitHub: [https://github.com/zzzzzzzzjj/MAVEN](https://github.com/zzzzzzzzjj/MAVEN)  
> - 包含 SA-MDPO 实现、MacroValue-Bench 数据集、MRC 提示模板等全部资源。

</details>

---

### 13. [Alignment Is All You Need: Instruction-Free Training for General Audio-Language Models](https://arxiv.org/abs/2608.18132)

**Authors**: Xuanru Zhou, Yiwen Shao, Jiahong Li, Dong Yu  
**Category**: cs.CL  
**Published**: 2026-08-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.18132v1  

#### Abstract
Multimodal large language models (MLLMs) are typically built through a multi-stage pipeline consisting of cross-modal alignment, supervised fine-tuning (SFT), and preference optimization. This pipeline assumes that adapting an LLM to a new modality requires extensive task-specific supervision. Howev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Alignment Is All You Need: Instruction-Free Training for General Audio-Language Models*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的多模态大语言模型（MLLMs）训练通常依赖于多阶段流程：跨模态对齐（cross-modal alignment）、监督微调（SFT）和偏好优化（preference optimization）。这些后两个阶段虽然提升了特定任务的表现，但也带来了以下问题：
- **指令诱导偏差（instruction-induced bias）**：模型过度拟合训练指令模板，导致泛化能力下降。
- **通用解码能力退化（erosion of universal decoding）**：LLM原有的强大推理和指令跟随能力在SFT中被“覆盖”或遗忘。
- **高昂的再训练成本**：每当有新的LLM发布时，都需要重新进行完整的SFT和偏好优化。

该论文提出一个根本性问题：**是否仅通过跨模态对齐就足以构建一个具有竞争力的多模态模型？**

### 提出的新方法与创新思路
作者提出了 **Instruction-Free Alignment-Only Training**（无指令、仅对齐）框架，其核心思想是：
- **冻结音频编码器（frozen audio encoder）和LLM（frozen LLM）**，只训练一个轻量级的 **projector** 来连接二者。
- **无需任何任务指令（instruction-free）** 进行训练，避免引入任务模板偏见。
- 利用 **Self-Generated Data Construction** 自动构造训练数据：将音频的文本描述（caption）输入到冻结的LLM中，由其自动生成自由形式的响应（free-form response），作为训练目标。

这种方法的关键创新在于：
- 将 **caption 视为音频的语义代理（semantic surrogate）**，利用LLM强大的生成能力自动扩展为丰富语义的目标响应。
- 实现了完全自动化的训练流水线，无需人工设计QA对、任务分类或指令模板。

### 相比现有方法的优势
| 维度 | 传统方法（SFT + 对齐） | 本文方法（仅对齐） |
|------|------------------------|--------------------|
| 模型干预程度 | 修改LLM参数 | 仅训练projector，LLM和encoder全冻结 |
| 数据需求 | 需大量标注的(instruction, response)对 | 只需(audio, caption)对，响应自动生成 |
| 泛化能力 | 易过拟合指令分布 | 保留LLM原生指令跟随能力 |
| 可迁移性 | 每代LLM需重训 | projector可快速适配新LLM |
| 训练效率 | 多阶段、高计算成本 | 单阶段、轻量级训练 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主训练数据**：`CaptionStew`，一个大规模音频-文本对数据集，包含10M样本，涵盖语音、音乐和环境音。
  - 使用其子集：400K、1M、4M用于缩放实验。
- **补充语音数据**：来自 `DailyTalk`, `CREMA-D`, `RAVDESS`, `TESS`, `MELD`, `IEMOCAP`, `VoxCeleb2`, `CommonVoice-en` 等，提升语音任务覆盖。
- **合成字幕数据**：使用 `Qwen3-Omni-Captioner` 在 `CaptionStew-400K` 上生成更详细的合成字幕，用于对比研究。

### 实验设置
- **模型架构**：模块化设计 `Encoder → Projector → LLM`
  - **LLM**：默认使用 `Qwen2.5-7B-Instruct`，也测试了 `Qwen3-8B` 验证跨代兼容性。
  - **Encoder**：五种不同范式的冻结编码器：
    - `AudioSet-Zipformer`（判别式）
    - `Whisper-large-v2`（ASR导向）
    - `Qwen2.5-Omni`, `Qwen3-Omni AuT`, `Qwen3-ASR AuT`（联合音频-语言预训练）
  - **Projector**：两层MLP，负责帧率下采样并映射到LLM嵌入空间。
- **训练方式**：
  - 仅更新projector参数（约31.2M可训练参数）。
  - 使用交叉熵损失训练，目标为LLM基于caption生成的response。
  - 无系统提示、无任务指令输入。

### 评估指标与基准
在四个公开音频理解基准上评估：
| Benchmark | 特点 |
|---------|------|
| **MMAU** | 10K音频，27项任务，涵盖声音、音乐、语音的理解与多步推理 |
| **MMAR** | 1K真实视频QA，四层推理：信号、感知、语义、文化 |
| **MMSU** | 5K语音QA，聚焦副语言现象（韵律、情感等） |
| **MMAU-Pro** | 5.3K专家标注，最长10分钟音频，含指令跟随（IF）、开放回答、闭集问答 |

评估指标：准确率（Accuracy）或平均得分（Avg. Score）。

### 基线方法对比
- **开源模型**：`SALMONN`, `LTU`, `Qwen2-Audio-Instruct`, `Audio-Flamingo 2/3`, `Kimi-Audio`, `ALARM` 等。
- **闭源模型**：`GPT-4o`, `Gemini 2.0/2.5 Flash`。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 模型 | MMAU Avg | MMAR | MMSU | MMAU-Pro Avg | MMAU-Pro IF |
|------|----------|------|------|-------------|------------|
| **Ours (AudioSet-Zipformer)** | **68.2** | 54.3 | 47.0 | **52.8** | **62.9** |
| Audio-Flamingo 3 | 72.4 | 58.5 | — | 51.7 | — |
| Qwen2.5-Omni | 71.0 | 48.3 | 43.3 | 52.2 | 61.3 |
| GPT-4o-Audio | 60.8 | 63.5 | — | 52.5 | — |
| Gemini 2.5 Flash | 67.4 | 68.4 | — | — | — |

> 注：本方法在 **MMAU-Pro 指令跟随（IF）** 上达到 **62.9**，**超越所有开源LALM**；在 **MMAU-Pro 总体平均** 上达 **52.8**，优于 `Audio-Flamingo 3` 和 `Qwen2.5-Omni`，接近 `GPT-4o-Audio`。

### 与基线方法对比结果
- **数据效率极高**：仅使用 **576.8K样本 / 11.6K小时**，远少于 `Audio-Flamingo 3`（26.7M样本），但在多个指标上表现相当甚至更好。
- **指令跟随能力最强**：因LLM保持冻结，完美保留了 `Qwen2.5-7B-Instruct` 的原生指令跟随能力，在MMAU-Pro IF上领先。
- **在非语音任务上优势明显**：`AudioSet-Zipformer` 编码器在 **Sound (80.8)** 和 **Music (69.8)** 上显著领先。

### 消融实验结果

#### （1）编码器影响（Table 2）
- `AudioSet-Zipformer` 在 **MMAU** 和 **MMAU-Pro** 上表现最佳（68.2 / 52.8），因其预训练目标与音频分类一致。
- `Whisper-large-v2` 在语音相关任务（如MMSU）上更强（50.6），符合其ASR预训练特性。
- 联合预训练编码器表现不如专用编码器，说明其表示在冻结LLM下迁移不完全。

#### （2）LLM 影响（Table 3）
- 当 **生成响应的LLM** 与 **对齐训练的LLM** 匹配时（matched-generator），性能稳定。
- 若不匹配（如用Qwen2.5生成数据训练Qwen3），性能大幅下降（↓5.9 MMAU Avg），验证了 **generator consistency** 的重要性。

#### （3）数据规模影响（Fig. 2a）
- **MMAU（闭集任务）**：性能随数据增加迅速饱和，说明受限于encoder能力而非数据量。
- **MMAR / MMAU-Pro（开放推理）**：性能随数据增加持续提升，表明丰富推理需要更多样化的alignment数据。

#### （4）字幕来源影响（Fig. 2b）
- **真实字幕（ground-truth）** 在大多数情况下略优。
- **合成字幕（synthetic）** 在 `Qwen3-Omni AuT` 上反超，尤其在MMSU（+4.6），说明高质量合成字幕可提供更丰富的语言线索。
- 结论：**简单字幕聚合即可有效，无需每条都密集标注**。

#### （5）语音专项SFT实验（Table 4）
- 在 `Whisper` 模型上追加语音QA的SFT后，**语音任务提升5.1点**，但 **声音和音乐任务下降4-5点**。
- 证明了 **instruction-free 是强默认配置**，专项SFT可按需添加，但会牺牲泛化性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Alignment is All You Need**：仅通过跨模态对齐即可构建具有竞争力的LALM，无需SFT或偏好优化。
2. ✅ **Frozen LLM 保留通用能力**：冻结LLM能完美保留其原生指令跟随和推理能力，避免“能力侵蚀”。
3. ✅ **性能受 encoder 和 LLM 共同约束**：最终性能上限由 `min(I(E), C(L))` 决定，即编码器的信息量与LLM的能力共同决定。
4. ✅ **可快速迁移至新LLM**：只要保持生成器一致，projector可快速适配新一代LLM，实现“即插即用”。
5. ✅ **数据多样性 > 数据密度**：简单字幕在多样化数据集中也能有效，无需每条都精细标注。

### 方法的局限性
- **依赖高质量编码器**：若编码器无法提取足够信息（如低质量ASR模型），alignment无法弥补。
- **无法超越基础模型能力**：不能通过alignment赋予LLM原本不具备的能力。
- **语音任务仍有差距**：在纯语音理解任务上仍落后于专门优化的模型（如ALARM）。
- **当前验证限于音频和7B规模**：未在更大模型或其他模态（如视觉）上验证。

### 未来工作方向
- 扩展到其他模态（如视觉-语言模型），验证“仅对齐”范式的普适性。
- 探索更高效的projector结构或动态对齐机制。
- 研究如何结合少量SFT以进一步提升特定任务而不损害泛化性。
- 构建更大规模的self-generated alignment数据集，推动无监督多模态学习。

---

> **总结**：本文颠覆了传统MLLM训练范式，证明了**alignment alone is sufficient**，提出了一种高效、轻量、可迁移的多模态训练新范式，为未来多模态模型的快速迭代提供了新思路。

</details>

---

### 14. [Beyond LLM-Based Reasoning: Lightweight GNNs for Agent Failure Attribution](https://arxiv.org/abs/2608.18575)

**Authors**: Ting-Wei Li, Yuanchen Bei, Xiao Lin, Hanghang Tong  
**Category**: cs.CL  
**Published**: 2026-08-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.18575v1  

#### Abstract
Large language model (LLM)-based multi-agent systems (MAS) often exhibit complex failure modes, which frequently cause agents to produce incorrect outcomes. This motivates the task of Agent Failure Attribution: given a failed multi-agent trajectory, identify the faulty agents and their corresponding...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond LLM-Based Reasoning: Lightweight GNNs for Agent Failure Attribution*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文研究 **Agent Failure Attribution (AFA)** 任务：在失败的 **LLM-based Multi-Agent System (MAS)** 轨迹中，识别出导致失败的**故障代理（faulty agents）**及其对应的**错误类型（error types）**。该任务对提升多智能体系统的可解释性和鲁棒性至关重要。

然而，现有方法普遍依赖 **Large Language Models (LLMs)** 进行推理分析，存在以下问题：
- **高计算开销**：长上下文处理、昂贵的 post-training（如 SFT、RL）。
- **系统复杂性**：复杂的提示工程或多阶段 agentic pipelines。
- **性能瓶颈**：即使最先进的 LLM 在现有基准上表现有限，表明单纯扩大模型规模不足以解决此任务。

### 提出的新方法与新思路
作者提出 **AFANet** —— 一种轻量级的基于图神经网络（GNN）的框架，挑战“必须依赖 LLM 推理”的主流范式。

#### 创新点：
- **新视角**：将 AFA 视为一个**结构化建模问题**而非生成式推理任务，认为通过建模交互轨迹中的**语义信号**和**代理间关系**即可有效归因。
- **轻量化设计**：采用 **turn-level conversation graph** 建模多代理交互过程，节点表示对话轮次，边表示时间顺序和同一代理的历史行为。
- **多源特征融合**：
  - **语义特征**：来自 sentence encoder（如 all-MiniLM-L6-v2）。
  - **统计与偏差特征**：基于 TF-IDF + SVD 构建，捕捉偏离正常交互模式的行为（如 self-inconsistency, conversation consensus deviation）。
- **高效推理机制**：利用 GNN 消息传递聚合信息，并通过 agent-level pooling 输出预测。

### 相比现有方法的优势
- ✅ **高性能**：在多个指标上达到甚至超越 fine-tuned 和 proprietary LLMs。
- ✅ **极高效率**：训练和推理成本极低，参数量仅为 65K，远小于数十亿参数的 LLM。
- ✅ **强鲁棒性**：在分布外（OOD）数据集上仍保持良好性能。
- ✅ **架构通用性**：在不同 GNN backbone（GCN/GAT/GraphSAGE）下均稳定有效。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型 | 描述 |
|-------|------|------|
| **AEGIS-Bench** [10] | In-domain | 包含预定义的训练/验证/测试划分，支持多种错误类型的标注。用于训练与验证 AFANet。 |
| **Who&When** [31] | Out-of-Distribution (OOD) | 每段对话仅有一个故障代理，用作 OOD 测试集以评估泛化能力。 |

> 两个数据集均采用 MIT 许可证。

### 实验设置
- **模型实现细节**：
  - GNN backbone 默认为 2 层 GCN，隐藏维度 64，bottleneck 维度 32。
  - 使用 Adam 优化器，学习率 $1 \times 10^{-2}$，batch size 2000，dropout 0.1。
  - 句子编码器：`all-MiniLM-L6-v2`。
  - 最佳模型基于验证集上的 **pair-level Micro-F1** 选择。
- **硬件平台**：NVIDIA V100 GPU。

### 评估指标
遵循 Kong et al. [10] 的标准，在三个粒度级别进行评估：
- **Agent-level**：正确识别故障代理（忽略错误类型）
- **Error-level**：正确识别错误类型（忽略具体代理）
- **Pair-level**：同时正确识别“代理-错误”组合（最具挑战性）

每个级别报告：
- **Micro-F1 (uF1)**：全局样本聚合后计算的 F1。
- **Macro-F1 (MF1)**：各类别分别计算 F1 后取平均。

### 基线方法对比
涵盖三大类 LLM-based 方法：
| 类别 | 代表模型 |
|------|---------|
| **Pre-trained LLMs** | Qwen2.5-7B/14B-Instruct, Qwen3-8B, GPT-4.1, GPT-4o-mini |
| **Fine-tuned LLMs** | Qwen2.5-7B/14B-Instruct + SFT (+S), + GRPO (+S+G) |
| **Proprietary LLMs** | o3, Gemini-2.5-Flash/Pro, Claude-Sonnet-4 |

所有 LLM 均采用 **All-at-Once prompting** 策略，输入完整轨迹并要求直接输出结果。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 方法 | AEGIS-Bench (Pair-uF1/MF1) | Who&When (Pair-uF1/MF1) | 平均得分 |
|------|----------------------------|--------------------------|--------|
| **AFANet (Ours)** | **17.42 / 16.35** | **6.90 / 4.16** | **24.82** |
| Qwen2.5-14B-It+S | 16.62 / 9.99 | 4.03 / 2.08 | 26.51 |
| o3 | 7.86 / 2.27 | 7.41 / 3.98 | 20.24 |
| Gemini-2.5-Flash | 6.99 / 2.76 | 7.32 / 3.33 | 19.55 |
| Random | 0.33 / 0.21 | 0.11 / 0.05 | 4.08 |

> 注：Pair-level 是最困难且最实用的任务，AFANet 在 AEGIS-Bench 上取得 **SOTA 表现**。

### 与基线方法的对比结果
- 🚀 **性能相当或更优**：尽管不使用 LLM 推理，AFANet 在多数指标上优于或媲美包括 fine-tuned 和 proprietary LLMs 在内的强大基线。
- ⏱️ **效率碾压级优势**（见 Table 2 和 Figure 1）：
  - **训练时间**：AFANet 仅需 **1.1 小时**，而 14B 模型 SFT+GRPO 需 >74 小时 → **~70x 加速**。
  - **推理延迟**：AFANet 仅 **1.16 秒（in-domain）/ 0.37 秒（OOD）**，相比 LLM 的 ~367 秒 → **~300x 速度提升**。
  - **参数量**：AFANet 仅 **65K 参数** vs. LLMs 的 7B/14B → **>100,000x 更小**。
- 🔁 **OOD 泛化能力强**：在 Who&When 上表现稳健，优于多数过拟合训练数据分布的 fine-tuned LLM。

### 消融实验结果（见 Table 4）

| 设置 | Pair-uF1 (AEGIS) | Pair-uF1 (Who&When) | 说明 |
|------|------------------|---------------------|------|
| Full Model (GCN + all edges + all features) | **17.42** | **6.90** | 完整模型 |
| No edges | 18.49 | 3.45 | 结构关系重要 |
| Only same-agent edges | 16.31 | 4.60 | 时间一致性有帮助 |
| Only temporal edges | 17.54 | 4.60 | 时序依赖关键 |
| Without dev/stat features | 14.00 | 1.72 | 偏差特征显著影响性能 |
| Without sentence embedding | 14.78 | 2.30 | 语义理解不可替代 |
| Without GNN | 17.81 | 2.87 | 显示结构建模的重要性 |

> 结论：**图结构、双类型边（temporal & same-agent）、偏差/统计/语义三类特征**共同构成 AFANet 成功的关键。

---

## 4. 关键结论和发现

### 主要发现
1. ❓ **核心问题回应**：“Heavy LLM reasoning is NOT necessary for AFA.”  
   轻量化的图结构建模足以实现高效的 agent failure attribution。
2. 🧠 **结构优于纯语义**：通过显式建模代理间的**交互动态**和**行为一致性**，AFANet 能捕捉错误传播路径，其性能来源于对**结构信号的有效利用**。
3. 💡 **偏差特征的价值**：非语义的统计与偏差特征（如 self-inconsistency）对于检测异常行为极为关键，补充了纯文本语义的不足。
4. 🔄 **轻量级 TTA 可进一步增强 OOD 性能**：通过 test-time adaptation（如 entropy minimization），可在无标签情况下进一步提升 OOD 表现（见 Table 5）。

### 方法的局限性（Appendix E）
- **OOD 泛化仍有挑战**：面对差异极大的多代理系统设置，当前模型泛化能力受限。
- **缺乏深层语义推理能力**：某些需要深度语义理解或长程逻辑推理的复杂错误可能难以被当前结构化模型捕获。
- **依赖预定义错误类型体系**：无法发现新的未知错误模式。

### 未来工作方向
- 设计更定制化的 **message passing operators** 来模拟错误传播机制。
- 开发自适应的 **graph propagation mechanisms** 以增强跨域泛化。
- 探索 **GNN 与 LLM 的协同框架**：结合结构建模的高效性与 LLM 的深层推理能力，实现 hybrid reasoning。
- 深入研究在更大规模、更多样化的 AFA 场景下的 generalization behavior。

--- 

> ✅ **总结一句话**：  
> AFANet 证明了在 Agent Failure Attribution 任务中，**轻量图神经网络可以取代昂贵的 LLM 推理**，在显著降低资源消耗的同时，实现更强或相当的性能，揭示了“结构即信号”的有效性。

</details>

---

### 15. [GraphK: Variable-Size Graph Generation with Efficient Edge Construction](https://arxiv.org/abs/2608.18777)

**Authors**: Resul Tugay, Eren Olu\u{g}, Elif Ak, Sule Gunduz Oguducu  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.18777v1  

#### Abstract
Graph generation models have advanced significantly with deep learning, yet they remain limited in scalability, flexibility, and ability to model underlying structures. We present GraphK, a novel encoder-sampler-decoder framework for graph generation that overcomes these challenges through structura...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GraphK: Variable-Size Graph Generation with Efficient Edge Construction

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的图生成模型在以下三个方面存在显著局限性：
- **缺乏置换不变性**（Permutation Invariance）：许多模型（如 GraphRNN、GRAN）对节点顺序敏感，导致同构图可能被不同处理。
- **可扩展性差**：大多数模型难以生成训练时未见过的图大小（即无法上采样或下采样），限制了其在数据增强等场景的应用。
- **计算成本高**：自回归模型（如 GraphRNN）和扩散模型（如 DiGress）通常具有 $O(N^2)$ 或更高的复杂度，难以扩展到大规模图。

### 🚀 提出的新方法：GraphK
GraphK 是一种基于 **Encoder-Sampler-Decoder** 架构的新型图生成框架，其核心思想是通过学习潜在空间中的分布来实现灵活且高效的图生成。

#### 主要创新点：
1. **支持变尺寸图生成（Variable-Size Generation）**
   - 利用 **Gaussian Mixture Model (GMM)** 对编码后的节点嵌入进行建模，并通过最大似然估计（MLE）从该分布中采样任意数量的新节点嵌入。
   - 实现了真正的“上采样”（upscale）和“下采样”（downscale），这是首次将系统化的上采样机制引入图生成任务。

2. **完全的置换不变性（Permutation Invariance）**
   - 编码阶段使用结构感知的嵌入方法（如 Node2Vec 或 VGAE），而采样阶段依赖于 GMM 对全局分布的建模，彻底解耦了节点索引的影响。
   - 不需要像 GRAN 那样通过平均多种排序近似不变性。

3. **高效边构造机制（Efficient Edge Construction）**
   - 引入 **KDTree + Top-k Nearest Neighbor Search** 在潜在空间中快速查找邻居，避免 $O(N^2)$ 全连接判断。
   - 时间复杂度降至 $O(N \log N)$，显著提升推理效率。

4. **几何先验与稀疏性兼容**
   - 基于流形平滑假设（manifold smoothness assumption），相似嵌入更可能相连。
   - 尽管采用局部连接策略（top-k），但由于选择的不对称性（多个节点可指向同一中心节点），仍能自然形成 hub 节点，从而生成幂律度分布图。

---

## 2. 核心实验方法和设置

### 📚 数据集
实验涵盖合成与真实世界图数据：
| 数据集 | 类型 | 规模 | 描述 |
|--------|------|-------|------|
| **2-/3-Block Community Graphs** | 合成社区图 | 60–120 节点 | 使用 SBM 生成，用于测试社区结构保持能力 |
| **Protein** | 生物分子图 | 100–500 节点 | 每个节点为氨基酸，边表示距离 < 6Å |
| **CiteSeer (CS)** | 引用网络 | 2,120 节点 | 最大连通分量，标准基准 |

此外还进行了大图生成实验（up to 50,000 节点）以验证可扩展性。

### 📊 评估指标
采用广泛使用的 **Maximum Mean Discrepancy (MMD)** 度量生成图与真实图之间的差异，分别在三个层面计算：
- **Spectral**：拉普拉斯谱分布（捕捉全局结构）
- **Orbit**：小子图轨道计数（捕捉中尺度模式）
- **Motif**：常见子图频次（捕捉局部结构）

数值越低越好。

### 🔁 基线方法对比
| 类别 | 方法 |
|------|------|
| 经典模型 | Erdos-Rényi (ER), Barabasi-Albert (BA) |
| VAE-based | GraphVAE |
| Autoregressive | GraphRNN, NetGAN, GRAN, BiGG |
| Diffusion-based | DiGress, PARD |

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（见 Table 2）

| 模型 | 2-Block 社区图 (MMD ↓) | Protein 图 (MMD ↓) | CiteSeer 图 (MMD ↓) |
|------|--------------------------|--------------------|---------------------|
| | Orbit / Motif | Orbit / Motif | Orbit / Motif |
| **GraphK (ours)** | **0.250 / 0.233** | 0.354 / 0.528 | **0.251 / 0.538** |
| NetGAN | 0.298 / 0.398 | 1.112 / 1.172 | 1.351 / 1.184 |
| DiGress | 0.268 / 0.244 | 0.262 / 0.482 | OOM |
| PARD | 0.250 / 0.349 | **0.253 / 0.201** | OOM |
| BiGG | 0.331 / 0.428 | 0.257 / 0.434 | 1.333 / 1.342 |

> ✅ **结论**：
> - GraphK 在多数指标上表现最优或具有竞争力。
> - 在 **CiteSeer** 上，GraphRNN、DiGress 和 PARD 出现内存溢出（OOM），凸显 GraphK 的**可扩展优势**。
> - 在 **Protein 图**上，GraphK 明显优于 GraphRNN 和 GraphVAE，尤其在视觉结构还原方面更优（见 Fig. 2）。

### ⚙️ 计算效率实验
- **推理时间**：对于 50,000 节点的图，GraphK 的解码时间**不到 10 秒**。
- 相比之下，BiGG 报告相同规模需约 **20 分钟**，GraphK 快 **120 倍以上**。
- 复杂度分析表明 GraphK 推理时间为 $O(N \log N)$，远优于传统 $O(N^2)$ 方法。

### 🔍 消融实验（Ablation Studies）

#### （1）Upscaling & Downscaling（Appendix A.6）
- 使用一个三社区图（200/100/100 节点）进行 ×10 上采样至 ~3,800 节点，再下采样回原始大小。
- 结果显示：**相对社区大小关系得以保留**，证明 GraphK 可有效控制图结构演化。

#### （2）Encoder 灵活性（Appendix A.5 & A.7）
- 成功使用 HOPE 编码器生成有向图（Directed Graph），并通过双 GMM 分别建模 source/destination 嵌入。
- 表明框架对编码器类型具有高度适应性。

#### （3）幂律行为生成（Appendix A.9）
- 在 Gnutella05 网络上实验，设置 $k=18$，成功复现**幂律状度分布**。
- 最高节点度达 54，说明即使每个节点只主动连接 $k$ 个邻居，仍可通过“被选中”机制形成 hub。

#### （4）置换不变性验证（Appendix A.5）
- 在 Protein 图上比较 GRAN 与 GraphK 的 Laplacian Spectrum MMD。
- 随着图增大（>300 节点），GRAN 性能下降明显（Spectral MMD 从 $1.9\times10^{-2}$ 升至 $2.5\times10^{-2}$），而 GraphK 保持稳定。
- 在最大组（400–500 节点）中，GraphK 的 Spectral MMD 比 GRAN **低 52%**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **GraphK 实现了真正意义上的置换不变性和尺寸灵活性**，解决了长期存在的图生成模型泛化难题。
2. 通过 GMM 采样 + KDTree 边预测机制，在保证高质量的同时实现了接近线性的生成速度。
3. 所提方法不仅能生成社区丰富图，还能自然产生幂律度分布，符合现实网络特性。
4. 框架模块化设计允许灵活替换 encoder（如 Node2Vec、VGAE、HOPE），适用于不同类型图（无向/有向）。

### ⚠️ 局限性
1. **平滑性假设限制表达能力**：仅依据嵌入相似性决定连通性，可能导致忽略某些重要但特征不相似的长程连接。
2. **不适合规则拓扑图**：如网格图（grid）、环状图等，因这些结构在潜在空间中不易聚类。
3. **依赖编码器质量**：若 encoder 未能充分捕获结构信息（如全局路径、层次结构），则生成效果受限。

### 🔮 未来工作方向
- 放松平滑性假设，引入注意力机制或 learnable edge decoder 来识别非局部关键连接。
- 扩展至动态图生成，结合时间序列建模。
- 探索更多类型的 latent variable models 替代 GMM（如 Normalizing Flows）以增强分布拟合能力。
- 应用于图神经网络的数据增强、隐私保护图发布、基础设施模拟等领域。

---

## ✅ 总结一句话
> **GraphK 提出了一种高效、灵活且置换不变的图生成框架，首次实现了可控的图尺寸缩放，并在多项任务中超越主流基线，为大规模图合成提供了实用解决方案。**

</details>

---

### 16. [Position: Multi-Agent Systems Should Prioritize Concurrency Control](https://arxiv.org/abs/2608.18092)

**Authors**: Xin Yang, Letian Li, Zimo Ji, Terry Jingchen Zhang, Wenyuan Jiang  
**Category**: cs.AI  
**Published**: 2026-08-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.18092v1  

#### Abstract
LLM-based multi-agent systems (MAS) promise scalable collaboration, yet adding agents often reduces reliability. This position paper argues that many MAS failures are fundamentally concurrency control problems: agents concurrently read and write shared state, and long LLM inference windows amplify t...

---

### 17. [Learning What to Fail On: Failure-Mode Contextual Bandits for Adversarial Data Curation](https://arxiv.org/abs/2608.18681)

**Authors**: Roie Kazoom, Ofir Cohen, Rami Puzis, Asaf Shabtai, Ofer Hadar  
**Category**: cs.CL  
**Published**: 2026-08-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.18681v1  

#### Abstract
We introduce a failure-aware adversarial retrieval-augmented framework for improving robustness in natural language understanding. Rather than selecting synthetic examples with a fixed reward threshold, our method formulates adversarial data curation as a failure-mode contextual bandit problem. Cand...

---

### 18. [ChiroEcho: extending automated bat vocalisation classification beyond the learned taxonomy](https://arxiv.org/abs/2608.18191)

**Authors**: Burooj Ghani, Welmoed Eversteijn, Milan van Hirtum, Juan Sebasti\'an Ca\~nas, Vincent J. Kalkman, Dan Stowell, A. Leonie Baier  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.18191v1  

#### Abstract
Bats are key indicators of ecosystem health and are protected throughout Europe, making reliable population monitoring a conservation priority. Their cryptic nocturnal lifestyle makes passive acoustic monitoring essential, yet automated identification remains difficult as echolocation calls vary wit...

---

### 19. [NanoSleep: A Parameter-Efficient Hybrid Temporal Convolutional Network for Single-Channel Sleep Stage Classification](https://arxiv.org/abs/2608.18571)

**Authors**: S M Asif Hossain, Shruti Kshirsagar  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.18571v1  

#### Abstract
Sleep stage classification from single-channel electroencephalography (EEG) is essential for wearable and home-based sleep monitoring. However, many deep learning models achieve high accuracy at the cost of large model sizes, which limits their deployment on resource-constrained devices. In this wor...

---

### 20. [A Unifying Relational Perspective on Expressive Lottery Tickets](https://arxiv.org/abs/2608.18819)

**Authors**: Lorenz Kummer, Samir Moustafa, Anatol Ehrlich, Franka Bause, Marco Nennstiel, Przemys{\l}aw Andrzej Wa{\l}\c{e}ga, Nils Morten Kriege  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.18819v1  

#### Abstract
Graph neural networks (GNNs) are widely used, but how parameter sparsity affects the expressivity of relational (RGNNs) and temporal (TGNNs) variants is poorly understood. The Strong Expressive Lottery Ticket Hypothesis (SELTH) posits the existence of sparse GNNs that preserve Weisfeiler-Leman (WL) ...

---

### 21. [Multi-stage neural operator learning with application for convolutions](https://arxiv.org/abs/2608.18851)

**Authors**: Zhiping Mao, Zhenye Wen, Yong Zhang, Xiaofei Zhao  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.18851v1  

#### Abstract
Convolution integrals widely exist in applications, and to enable fast and accurate computations, this paper introduces two general multi-stage neural operator learning frameworks. The first, Deep Collocation Neural Operator (DCNO), is a supervised approach that iteratively refines the operator appr...

---

### 22. [A FEM-Based Surrogate Modelling and Optimization Framework for Physics-Constrained Electromagnetic Coil Design](https://arxiv.org/abs/2608.18903)

**Authors**: Yucheng Liu  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.18903v1  

#### Abstract
This work evaluates surrogate-assisted optimization of a seven-parameter current-excited coil--core benchmark subject to geometric, manufacturing, and separate core and copper mass constraints. A Python--MPh--COMSOL workflow couples a two-dimensional axisymmetric finite-element method (FEM) model to...

---

### 23. [SCORE: Subject Coordinate Recovery for Label-Free Cross-Subject EEG-to-Image Retrieval](https://arxiv.org/abs/2608.19134)

**Authors**: Zhenyao Cui, Siyuan Kan, Siyang Li, Ziwei Wang, Dongrui Wu  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.19134v1  

#### Abstract
Accurate visual decoding can reveal how the brain represents visual information and recover perceived content from neural signals such as electroencephalography (EEG), with potential for neural communication. However, current EEG-to-image retrieval methods perform far below their within-subject coun...

---

### 24. [Cacheable by Design? Training Mixture-of-Experts Routers for Locality Against the Edge Memory-Bandwidth Wall: A Pre-Registered Negative Result with a Systems Measurement Study](https://arxiv.org/abs/2608.18261)

**Authors**: Shriniwas Ramesh Suram  
**Category**: cs.AI  
**Published**: 2026-08-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.18261v1  

#### Abstract
Serving a 235B-parameter Mixture-of-Experts (MoE) model on a single 8 GB GPU is bottlenecked not by compute but by memory bandwidth: decode must stream each token's active experts from whichever tier holds them, and on consumer hardware most experts sit on an SSD far slower than RAM. We quantify thi...

---

### 25. [Beyond the Transcript: Detecting Covert Co ordination in Latent Multi-Agent Communication](https://arxiv.org/abs/2608.19161)

**Authors**: Ramneet Kaur, Pradyumna Chari, Ramesh Raskar, Jugad Singh, Sumit Kumar Jha, Anirban Roy  
**Category**: cs.AI  
**Published**: 2026-08-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.19161v1  

#### Abstract
Language-model agents can communicate through continuous hidden states that are invisible in public transcripts, creating opportunities for covert harmful coordination. We introduce Verifiable Latent Alignments (VLA), an activation-aware framework for monitoring and steering these private communicat...

---

### 26. [A Fast Deterministic Algorithm for $(\Delta+1)$-edge coloring in CONGEST](https://arxiv.org/abs/2608.19184)

**Authors**: Sebastian Brandt, Ananth Narayanan, Alexandre Nolin  
**Category**: cs.DC  
**Published**: 2026-08-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.19184v1  

#### Abstract
Vizing's theorem states that any graph of maximum degree $\Delta$ can be properly edge-colored with $\Delta + 1$ colors (which is optimal in general). A recent breakthrough result by Bernshteyn showed that such a $(\Delta + 1)$-edge coloring can be found deterministically in $poly(\Delta,\log n)$ ro...

---

### 27. [Role-Conditioned Sub-Token Routing for Efficient Vision-Language-Action Policies](https://arxiv.org/abs/2608.18410)

**Authors**: Wei Jiang, Wei Wang  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.18410v1  

#### Abstract
Vision-Language-Action (VLA) models process long multimodal token sequences, making inference expensive in both memory and computation. Existing efficiency methods mainly reduce visual tokens, but aggressive token pruning becomes fragile because removing a token discards its entire representation. S...

---

### 28. [Transportable Causal Effect Estimation across Networks under Interference](https://arxiv.org/abs/2608.18932)

**Authors**: Xiaojing Du, Jiuyong Li, Lin Liu, Debo Cheng, Jixue Liu, Thuc Duy Le  
**Category**: cs.LG  
**Published**: 2026-08-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.18932v1  

#### Abstract
Estimating causal effects under network interference typically assumes that the network used for training and the network used for deployment coincide. In practice, an intervention is run on one population while the question of interest concerns a different population, and the two generally differ i...

---

### 29. [Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions](https://arxiv.org/abs/2608.18078)

**Authors**: Matthew Riemer, Tommaso Tosato, Amin Memarian, Maximilian Puelma Touzel, Glen Berseth, Irina Rish, Guillaume Dumas  
**Category**: cs.AI  
**Published**: 2026-08-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.18078v1  

#### Abstract
This position paper argues that AI agents with chain-of-thought reasoning capabilities are predisposed to exhibit collusive behavior and should be required to obtain behavioral certification before making decisions that affect economic markets. This is because integrating these agents into society c...

---

### 30. [ComponentBench: Diagnosing Component-Level Failures in Computer-Use Agents](https://arxiv.org/abs/2608.18307)

**Authors**: Tianchen Guan, Xinlei Lin, Royce Cheng-Yue, Xiangjun Wang, Shuyan Zhou  
**Category**: cs.AI  
**Published**: 2026-08-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.18307v1  

#### Abstract
Current evaluation of computer-use agents is split between long-horizon workflow benchmarks and atomic GUI-grounding tests. This leaves an under-instrumented middle layer: realistic component-centered interactions (e.g., toggle a button set) that are short enough to diagnose and rich enough to captu...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-04 08:11:12 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [LEAP: Lean Environment-Feedback via Adaptive Pruning for Code RL in GPU Kernel Generation](https://arxiv.org/abs/2608.01804)

**Authors**: Tankun Li, Zhi Chen, Yaohua Tang  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2608.01804v1  

#### Abstract
Post-training large language models (LLMs) via reinforcement learning (RL) has significantly advanced code generation capabilities. To bypass the heavy memory footprint of critic networks, current state-of-the-art frameworks leverage critic-free paradigms like Group Relative Policy Optimization (GRP...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《LEAP: Lean Environment-Feedback via Adaptive Pruning for Code RL in GPU Kernel Generation》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前基于 **Reinforcement Learning (RL)** 的代码生成框架（如 GRPO）在应用于低级系统编程任务（如从零生成 CUDA kernel）时面临两大挑战：

1. **信号稀疏性（Signal Sparsity）**：  
   传统 rule-based 验证仅提供二元奖励（pass/fail），对轻微错误和严重失败同等惩罚，导致学习信号稀疏。

2. **计算开销巨大（Prohibitive Latency）**：  
   多轮环境反馈（multi-turn feedback）需要反复编译、部署到 GPU 并执行 kernel，带来极高的时间与资源成本，使得密集多轮训练不可扩展。

此外，标准 multi-turn RL 存在 **reward dilution** 问题：无论是否首次成功，最终成功的轨迹获得相同奖励，削弱了模型提升 first-turn 准确率的动力。

---

### 🚀 提出的新方法与核心思路

作者提出 **LEAP**（Lean Environment-Feedback via Adaptive Pruning），一种高效、可扩展的多轮 RL 框架，专为低级硬件对齐（low-level hardware alignment）设计。其两大核心技术是：

#### （1）Difficulty-Conditioned Pruning (DCP) —— 自适应任务剪枝机制

- 动态判断每个 prompt 的难度：通过统计一个 rollout group 中的失败数量 $ N_{\text{fail}}(q) $ 来估计任务难度。
- 设定阈值 $ [T_{\min}, T_{\max}] $，决定是否启用 multi-turn debugging：
  - 若 $ N_{\text{fail}} < T_{\min} $：任务简单 → 跳过调试，强制单轮优化；
  - 若 $ N_{\text{fail}} > T_{\max} $：任务太难 → 放弃调试，避免浪费资源；
  - 否则进入 multi-turn 调试流程。
- **效果**：将昂贵的编译-执行循环集中在“有希望改进”的中等难度任务上，显著降低训练延迟。

#### （2）Rank-Based Reward —— 非参数化、基于排序的奖励机制

- 不再依赖人工设定的标量奖励（如第一轮成功得 1.0，第二轮得 0.8），而是基于组内成对比较（pairwise tournament）动态生成相对优势。
- 所有 rollout 按成功率和效率排序：
  - 第一轮成功 > 第二轮成功 > 失败；
  - 成功越早，rank 越高。
- 每个样本的奖励定义为其击败的同伴数减去被击败的同伴数，归一化后落在 [-1.0, 1.0] 区间。

> 该机制天然实现：
> - 在简单任务中惩罚冗余 turn（鼓励 zero-shot 成功）；
> - 在困难任务中放大微小进步的价值（防止梯度消失）。

---

### 🔍 相比现有方法的优势

| 维度 | LEAP | 现有方法（如 GRPO-MT、Murphy、Dr.Kernel） |
|------|------|----------------------------------------|
| **计算效率** | 显著更高（减少不必要的 multi-turn） | 密集树搜索或全量调试，开销大 |
| **first-turn 准确率** | 更高 | 倾向于依赖后期修正，first-pass 表现下降 |
| **奖励稳定性** | 无需调参，自适应调整 | 依赖“magic number”式手动调奖，不稳定 |
| **资源利用率** | 只在有价值的任务上投入计算 | 对所有任务无差别处理，浪费严重 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

1. **CUDA Kernel Generation Task**：
   - **PyTorch-to-CUDA Dataset**（Cheng et al., 2026c）：用于冷启动 SFT。
   - **CUDA-Agent Dataset**（Dai et al., 2026）：用于 RL 训练。
   - **KernelBench**（Ouyang et al., 2025）：主测试基准，评估从零生成高效 CUDA kernel 的能力。

2. **General Coding Tasks**（泛化性验证）：
   - **KodCode**（Xu et al., 2025）：多样化、可验证的合成编程数据集。
   - **LiveCodeBench**（Jain et al., 2025b）：污染控制严格的综合评测集。

---

### ⚙️ 实验设置

- **模型架构**：基于 Qwen2.5-7B 进行 RL 微调。
- **训练框架**：VERL（Sheng et al., 2024），集成 GRPO。
- **关键超参数**：
  - 学习率：1e-6
  - Batch size：32
  - 每 prompt rollout 数：8
  - 最大调试轮次：3
- **硬件配置**：
  - 训练：8×Nvidia B200 GPUs
  - 验证沙箱：2×服务器，共 16×Nvidia A100 GPUs（用于 CUDA 编译与执行）

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **Acc@1 / Acc@2 / Acc@3** | 第 1/2/3 轮累计通过率 |
| **Overall Pass@1, Pass@5** | LiveCodeBench 上的标准指标 |
| **Wall-clock Time to Convergence** | 达到某一准确率所需的真实训练时间 |
| **Turn Efficiency** | `(总使用 turn 数) / (成功样本数)`，衡量推理 token 效率 |
| **Activated Groups (%)** | 触发 multi-turn 调试的比例（反映 DCP 剪枝强度） |

---

### 🆚 基线方法对比

| 方法 | 类型 | 是否 multi-turn | 是否 critic-free |
|------|------|------------------|------------------|
| **Baseline (Standard GRPO)** | 单轮 | ❌ | ✅ |
| **Murphy**（Ekbote et al., 2025） | 多轮 | ✅ | ✅ |
| **Dr.Kernel**（Liu et al., 2026） | 多轮 | ✅ | ✅（RLOO） |
| **LEAP (Ours)** | 自适应多轮 | ✅（条件触发） | ✅ |

> 注：Murphy 和 Dr.Kernel 均采用无 critic 架构，但未引入剪枝机制。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & Table 4）

#### 🔹 KernelBench 结果（Overall）

| 方法 | Turn 1 | Turn 2 | Turn 3 |
|------|--------|--------|--------|
| Baseline | 66.0% | 75.2% | 75.2% |
| Murphy | 67.6% | 76.4% | 77.2% |
| Dr.Kernel | 68.4% | 79.6% | 80.4% |
| **LEAP** | **70.0%** | **79.2%** | **80.8%** |

✅ **LEAP 在 first-turn 准确率上全面领先**，且最终性能最优。

#### 🔹 Level 3（最难级别）表现

| 方法 | Turn 1 | Turn 3 |
|------|--------|--------|
| Baseline | 18% | 24% |
| Murphy | 24% | 34% |
| Dr.Kernel | 24% | 32% |
| **LEAP** | **30%** | **36%** |

➡️ 表明 LEAP 在 hardest 任务上的学习信号更强，debugging 更有效。

#### 🔹 LiveCodeBench 结果（Pass@1）

| 方法 | Easy | Medium | Hard | Overall |
|------|------|--------|------|---------|
| Baseline | 0.609 | 0.088 | 0.007 | 0.174 |
| Murphy | 0.614 | 0.085 | 0.010 | 0.181 |
| Dr.Kernel | 0.609 | 0.085 | 0.000 | 0.175 |
| **LEAP** | **0.619** | **0.085** | **0.017** | **0.185** |

➡️ LEAP 在 hard 分类上显著优于所有 baseline，体现其强大的复杂问题解决潜力。

---

### ⏱️ 效率与收敛速度（Table 2 & Figure 1）

| 方法 | Step Time (s) | Turn per Pass | 加速比 |
|------|---------------|----------------|--------|
| Baseline | ~600 | 1.90 | — |
| Murphy | ~950 | 1.93 | — |
| Dr.Kernel | ~950 | 1.84 | — |
| **LEAP** | **~700** | **1.77** | **1.93× faster than baseline** |

📌 **LEAP 在 wall-clock 时间上比 baseline 快 1.93 倍达到相同性能**，并最终超越。

---

### 🔍 消融实验结果

#### （1）DCP 剪枝范围影响（Table 5）

| Pruning 设置 | Acc@3-turns | Activated Groups | Per-group Recovery |
|-------------|--------------|------------------|--------------------|
| `<50%`（只对易题调试） | 77.6% | 57.1% | 73.9% |
| `≥50%`（只对难题调试） | **80.8%** | **43.9%** | **61.8%** |
| Full（全部调试） | 80.4% | 100% | 64.5% |

➡️ **仅对难题启用 multi-turn 调试效果最好**，验证了“简单任务已有足够信号”的假设。

#### （2）Rank-Based Reward 消融（Table 6）

| 方法 | Overall Acc@1 | Acc@3 |
|------|----------------|--------|
| Standard GRPO-MT | 70.0% | 79.2% |
| **Rank-Based (LEAP)** | **70.0%** | **80.8%** |

➡️ 尽管 first-turn 相同，但 LEAP 在后续轮次提升更大，说明其 reward 更有效地引导了 debugging 学习。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **并非所有任务都值得 multi-turn 调试**：  
   简单任务已有充足正例信号，强行加入调试会弱化 zero-shot 能力；而极难任务几乎无法修复，应跳过以节省资源。

2. **adaptive pruning + rank-based reward 是黄金组合**：  
   DCP 控制“何时调试”，rank-based reward 决定“如何学习”，二者协同实现了效率与性能的双重提升。

3. **first-turn accuracy 与 multi-turn resilience 可兼得**：  
   传统方法常在这两者之间权衡，而 LEAP 通过选择性调试打破了这一 trade-off。

4. **真实训练时间比理论迭代步数更重要**：  
   在生产级代码生成中，wall-clock efficiency 是决定能否落地的关键因素，LEAP 在此方面具有明显优势。

---

### ⚠️ 局限性

1. **DCP 依赖 group-level 统计**：  
   需要每 batch 至少多个 rollout 才能估计难度，在小批量或低并发场景下可能不稳定。

2. **rank-based reward 对 group diversity 敏感**：  
   如果 group 内 responses 高度相似，tournament 结果可能缺乏区分度。

3. **目前聚焦 CUDA kernel，泛化到其他 DSL 或芯片架构需进一步验证**。

---

### 🔮 未来工作方向

1. **动态调整 $ T_{\min}, T_{\max} $ 阈值**：  
   当前为固定超参，未来可尝试随训练进程自动调节。

2. **扩展至更多 hardware-aware 编程语言**：  
   如 ROCm、SYCL、OpenCL 等，构建通用的 hardware compiler co-design 框架。

3. **结合 LLM-as-a-Judge 提供软标签反馈**：  
   在保留 low-latency 的前提下，融合语义级评分以增强 reward 密度。

4. **探索 offline RL + LEAP 的混合范式**：  
   利用历史失败轨迹进行离线学习，进一步减少在线 sandbox 查询次数。

---

> **总结一句话**：  
> LEAP 通过 **Difficulty-Conditioned Pruning** 和 **Rank-Based Reward**，在不牺牲性能的前提下，大幅提升了 multi-turn Code RL 的训练效率与实用性，为面向 GPU 等加速器的底层代码自动生成提供了可行路径。

</details>

---

### 2. [Energy-Efficient LLM Serving via Disaggregated Attention--FFN and Flexible Frequency Scaling](https://arxiv.org/abs/2608.01891)

**Authors**: Cunchen Hu, Liangliang Xu, Tian Liu, Min Lyu, Yongkun Li, Sa Wang, Shuo Quan, Yanan Yang, Wenda Tang, Yiduo Wang, Fu Yu, Jie Wu  
**Category**: cs.DC  
**Published**: 2026-08-04  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2608.01891v1  

#### Abstract
Large language model (LLM) serving spans diverse applications with stringent service-level objectives (SLOs), often requiring GPUs to run at maximum frequencies and increasing energy consumption. Existing energy-management approaches adapt GPU frequencies only at the request or inference-phase level...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Energy-Efficient LLM Serving via Disaggregated Attention-FFN and Flexible Frequency Scaling

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLM）推理服务在满足严格的服务级别目标（SLOs，如 TTFT 和 TPOT）的同时，通常需要 GPU 以最高频率运行，导致极高的能耗。现有的能效管理方法（如 DynamoLLM、BiScale）仅在请求粒度或推理阶段（prefill/decode）粒度上进行 GPU 频率调节（DVFS），**忽略了 Attention 和 FFN 这两个核心算子之间的频率敏感性差异**。

此外，传统方法将 Attention 和 FFN 在同一 GPU 上交替执行，造成资源利用不均衡和 pipeline bubbles，进一步浪费能量。

---

### **提出了什么新方法或新思路**

本文提出 **AFlex**，首个结合 **算子级拆分（Attention-FFN Disaggregation, AFD）** 与 **细粒度动态电压频率调节（Flexible per-operator DVFS）** 的 LLM 推理框架。

#### 核心创新点：

- ✅ **算子级异构优化**  
  将 Attention 和 FFN 拆分为独立的执行单元（PA/PF for prefill, DA/DF for decode），并为每个算子分配独立的 GPU 资源池和最优频率。

- ✅ **双层控制平面（Two-level Control Plane）**
  - **Global Scheduler**：周期性求解整数线性规划（ILP），联合优化 A/F 资源配比、TP degree 和基础频率，在满足 SLO 的前提下最小化能耗。
  - **Local DVFS Controller**：运行时基于微批大小、负载变化等动态调整各算子频率，实现毫秒级响应。

- ✅ **交错式 A/F 流水线（Interleaved A/F Pipeline）**
  - 引入动态 microbatch depth 和自适应请求批处理，重叠计算与通信，减少 pipeline bubbles。
  - 支持隐藏状态跨层高效传输，降低 AFD 带来的通信开销。

- ✅ **开销感知的配置切换机制**
  - 区分轻量级频率切换与重量级 TP 重构，通过增量权重重分片（incremental resharding）和异步准备降低切换延迟。

---

### **相比现有方法的优势**

| 维度 | 现有方法（如 DynamoLLM, BiScale） | AFlex |
|------|-------------------------------|--------|
| 控制粒度 | 请求级 / 阶段级（Prefill/Decode） | **算子级（Attention/FFN）** |
| 频率策略 | 同一阶段内统一频率 | **每个算子独立频率控制** |
| 资源调度 | 集中式或阶段级拆分 | **A/F 拆分 + 动态平衡** |
| Pipeline 效率 | 存在显著 bubbles | **交错执行 + 自适应 batching 减少空闲** |
| 能效潜力 | 受限于粗粒度控制 | **挖掘算子间异构性，释放更大节能空间** |

---

## 2. 核心实验方法和设置

### **使用的数据集与工作负载**

- **真实生产轨迹（Production Traces）**：
  - **Azure Conversation Trace**：对话类任务，输入短、输出长度可变。
  - **Azure Coding Trace**：代码生成任务，输入长、输出较短。

- **受控合成工作负载（Controlled Workloads）**：
  - QA（128/64）：prefill-light, decode-light
  - Chatbot（128/1024）：prefill-light, decode-heavy
  - RAG（4096/64）：prefill-heavy, decode-light
  - Summary（4096/1024）：prefill-heavy, decode-heavy

用于系统评估不同输入/输出长度组合下的表现。

---

### **模型与硬件平台**

- **模型**：
  - **Qwen3-32B**（dense model）
  - **Mixtral-8×7B**（MoE model）

- **硬件环境**：
  - 多节点集群：每节点 8× NVIDIA A800-80GB GPU
  - NVLink（400 GB/s）、800 Gbps 跨节点网络
  - 最大 GPU 频率：1410 MHz，最低可调至 210 MHz

- **实现基础**：基于 **SGLang** 构建，扩展支持 AFD 和 DVFS 控制。

---

### **评估指标**

| 指标 | 定义 | 目标 |
|------|------|------|
| **Energy per token** | 总 GPU 能耗 / 总输入+输出 token 数 | 越低越好 |
| **P90 TTFT** | 第一个 token 的 90 百分位延迟 | ≤ 400ms |
| **P90 TPOT** | 每个输出 token 的 90 百分位延迟 | ≤ 120ms |
| **Average GPU Frequency** | 所有 GPU 平均运行频率 | 反映节能程度 |

---

### **基线方法对比**

| 基线 | 特点 |
|------|------|
| **SGLang** | P/D 共置，无 DVFS，连续批处理 |
| **DynamoLLM** | P/D 共置 + 阶段级 DVFS |
| **DistServe** | P/D 拆分，无 DVFS |
| **BiScale** | P/D 拆分 + 阶段级 DVFS |

> AFlex 是唯一同时实现 **A/F 拆分 + 算子级 DVFS** 的系统。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔋 **端到端能效提升**

| 场景 | 对比对象 | Energy per token 下降幅度 |
|------|----------|-----------------------------|
| Conversation Trace | DistServe（state-of-the-art disaggregated） | **↓49%** |
| Conversation Trace | DynamoLLM（frequency-scaling） | **↓48%** |
| Coding Trace | DistServe | **↓49%** |
| Coding Trace | DynamoLLM | **↓48%** |

> 在满足 SLO 的前提下，**AFlex 实现高达近一半的能量节省**。

---

#### ⏱️ **SLO 满足情况**

| 指标 | AFlex 表现 |
|------|------------|
| P90 TTFT | < 302 ms（Conversation），< 265 ms（Coding） |
| P90 TPOT | < 97 ms（Conversation），< 102 ms（Coding） |
| 所有负载下均满足预设 SLO（400ms / 120ms） | ✅ |

> 尽管大幅降频，仍能保障服务质量。

---

#### 📉 **GPU 频率降低效果**

- **平均 GPU 频率从 >1300 MHz（基线）降至 530 MHz**
- 各算子频率差异化调控：
  - **PA**: 656 MHz（较高，因长输入易成瓶颈）
  - **PF**: 452 MHz（较低，可利用 pipeline slack）
  - **DA**: 616 MHz
  - **DF**: 580 MHz

> 显示 AFlex 成功识别并利用了算子间的频率敏感性差异。

---

### **消融实验结果**

#### （1）组件贡献分析（Fig. 16）

| 配置 | Conversation 下节能效果 |
|------|------------------------|
| Vanilla AFD（仅拆分） | 基线 |
| + Global Scheduler | ↑5.9% ~ 37.3% 节能（随负载上升而增加） |
| + Local DVFS Controller | 再额外 ↓20.1%（Coding 场景达 47.3%） |

> 表明全局调度与本地 DVFS 协同作用显著。

#### （2）动态 microbatch depth 与自适应批处理（Table I）

| 优化 | 效果 |
|------|------|
| **Dynamic M (1↔2)** | 高负载下减少 bubble 时间 **65.1%**，延迟 ↓20.0% |
| **Adaptive Request Batching** | 平衡 A/F 阶段，bubble ↓16.1%，延迟 ↓19.9% |

> 有效缓解低利用率和阶段失衡问题。

#### （3）可扩展性与 MoE 泛化性**

- **Scale-out（1→4 nodes）**：节能优势持续存在，相对最佳基线至少 ↓20.2%
- **Mixtral-8×7B（MoE）**：AFlex 达到最高能效，相较基线最多 ↓41.4%

> 表明 AFlex 在多种规模和模型架构下均具鲁棒性。

---

#### （4）系统开销**

| 开销类型 | 数据 | 是否可接受 |
|---------|------|-----------|
| ILP 规划时间（8–32 GPUs） | 0.49–0.57 秒 | << 5分钟调度窗口（<0.2%）✅ |
| TP 重构时间（AFlex vs Base） | ↓52.2%（TP2→TP4），↓72.3%（TP4→TP1） | 显著优化 ✅ |
| 预测器误差（MAPE） | Prefill <11%，Decode ≈2% | 高精度预测 ✅ |

> 系统引入的控制开销极小，不影响在线服务。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Attention 与 FFN 的频率敏感性显著不同**  
   - FFN 更 compute-bound，对频率更敏感；
   - Attention 更 memory-bound，高频收益有限。
   - 使用统一频率会造成“为 FFN 付费却浪费在 Attention”。

2. ✅ **算子级拆分（AFD）带来巨大节能潜力**  
   - 理论估算显示异构 A/F 配置比同构最多可省 **48.1%** 能量代理值（energy proxy）。

3. ✅ **静态配置无法适应动态负载**  
   - 最优频率随 batch size、sequence length、TP degree 变化而变化，需动态调节。

4. ✅ **pipeline bubbles 是轻负载下主要能耗来源**  
   - AFlex 通过动态 microbatch depth 和自适应 batching 显著减少空转时间。

5. ✅ **AFlex 实现 SLO 与能效双赢**  
   - 不仅节能近半，且保持更低的 P90 延迟，优于所有基线。

---

### **方法的局限性**

- ❗ **依赖离线 profiling**：需预先采集大量配置下的性能与能耗数据，部署成本略高。
- ❗ **AFD 带来额外通信开销**：虽已优化（如 NIC-affine 传输），但在极端低延迟场景可能成为瓶颈。
- ❗ **当前仅支持 tensor parallelism**：未整合 pipeline parallelism 或专家并行（expert parallelism）的完整支持。
- ❗ **切换策略保守**：为避免频繁 DVFS 切换，采用“窗口机制”，可能导致瞬时负载突变时响应滞后。

---

### **未来工作方向**

- 🔮 **支持更多并行范式**：扩展至 PP + EP + AFD 的多维拆分架构。
- 🔮 **在线自适应学习**：用强化学习替代 ILP，实现实时动态决策。
- 🔮 **跨节点协同 DVFS**：考虑电源域、散热条件等物理因素进行集群级能效优化。
- 🔮 **绿色 AI 集成**：结合碳感知调度（carbon-aware scheduling），实现真正可持续的 LLM 推理。

---

> **总结一句话**：  
> AFlex 通过 **Attention-FFN 拆分 + 双层频率控制 + 智能流水线调度**，首次实现了 **算子级精细化能效管理**，在不牺牲 SLO 的前提下将 LLM 推理能耗降低近 **50%**，为绿色 AI 提供了重要实践路径。

</details>

---

### 3. [Constrained Co-Design for Photonic Bayesian Neural Networks](https://arxiv.org/abs/2608.02229)

**Authors**: Hendrik Borras, Xiao Wang, Bernhard Klein, Robin Janssen, Frank Br\"uckerhoff-Pl\"uckelmann, Wolfram Pernice, Holger Fr\"oning  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2608.02229v1  

#### Abstract
Classical neural networks frequently produce overconfident predictions on ambiguous or out-of-distribution (OOD) data, a liability that grows with each AI system deployed in safety-critical real-world scenarios. Bayesian neural networks (BNNs) provide a principled framework for uncertainty-aware pre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Constrained Co-Design for Photonic Bayesian Neural Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统 **Bayesian Neural Networks (BNNs)** 虽能提供不确定性估计，但在数字硬件上依赖多次采样，导致推理延迟高、能耗大，难以部署于实时、安全关键场景。**Photonic BNNs** 利用光学器件的本征随机性实现高速并行采样，是潜在解决方案。

然而，光子硬件并非理想采样器，其模拟约束（如量化、编程误差、动态范围限制）会直接影响可实现的变分分布族（variational family），从而降低不确定性估计质量。现有研究多预设硬件架构，缺乏系统性的协同设计（co-design）方法来分析哪些约束可被训练补偿，哪些必须通过硬件改进解决。

本文系统地回答了以下三个核心问题：
- Q1：哪些硬件约束会限制大规模光子 BNN 的预测准确性和不确定性质量？
- Q2：在不同随机性位置和模态下，量化、编程误差、均值/方差范围等约束的容忍阈值是多少？
- Q3：哪些约束可通过训练补偿？哪些需要硬件或架构干预？

---

### **提出的新方法与创新思路**
1. **将光子 BNN 推理建模为“受限随机变分推断”（constrained stochastic variational inference）**  
   将光子处理器视为一个受物理约束的随机神经算子（stochastic neural operator），明确将硬件约束（如量化、均值/方差边界、编程噪声）映射到变分分布的表示能力上。

2. **系统性消融研究（systematic ablation study）**  
   独立分析五类关键硬件约束的影响：
   - 随机性位置（weight vs. activation）
   - 随机性模态（additive vs. multiplicative）
   - 输入量化（input quantization）
   - 编程误差（programming error）
   - 可表示的均值/方差范围（mean/std bounds）

3. **提出实用的协同设计指南（co-design guidelines）**  
   明确区分：
   - **可通过训练补偿的约束**（如输入量化 ≥4 bit，适度编程误差）
   - **需硬件干预的硬性约束**（如激活空间对均值范围要求极高，无符号表示导致模型崩溃）

4. **基于真实光子架构（A-TWI）的集成验证**  
   在耦合的、硬件现实的约束下验证指南的有效性，证明其可指导实际部署。

---

### **相比现有方法的优势**
| 方面 | 本文优势 |
|------|---------|
| **方法论** | 首次系统性量化光子 BNN 中各类硬件约束的容忍度，而非仅展示单一架构性能 |
| **实用性** | 提供可操作的设计原则，帮助硬件与算法团队分工协作 |
| **泛化性** | 分析框架适用于多种光子架构（crossbar, TWI 等），不限于特定实现 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 用途 |
|-------|------|
| **Dirty-MNIST** | 主要不确定性基准，含模糊样本（ambiguous）用于评估偶然不确定性（aleatoric），搭配 Fashion-MNIST 作为 OOD 数据集评估认知不确定性（epistemic） |
| **CIFAR-10 / CINIC-10** | 更具挑战性的 ID 分类任务，评估可扩展性 |
| **SVHN** | 作为 CINIC-10 和 CIFAR-10 的 OOD 检测基准 |

---

### **实验设置**
- **模型架构**：
  - LeNet（Dirty-MNIST）
  - ResNet-18（CIFAR-10, CINIC-10）
- **随机性配置**：四种组合
  - Weight-Additive (W-Add)
  - Weight-Multiplicative (W-Mul)
  - Activation-Additive (A-Add)
  - Activation-Multiplicative (A-Mul)
- **训练策略**：
  - 使用 **Optuna** 进行超参优化
  - 采用 **RAdamScheduleFree** 优化器和 **TrivialAugment** 数据增强
  - 手动选择帕累托最优解以平衡准确率与 OOD AUROC
- **硬件模拟**：
  - 在 **Pyro** 概率编程库中实现受限随机算子
  - 模拟 A-TWI 架构的关键约束（见 Table 1）

---

### **评估指标**
| 指标 | 定义 | 用途 |
|------|------|------|
| **Accuracy** | 分类准确率 | 衡量 ID 性能 |
| **OOD AUROC** | 使用 **Mutual Information (MI)** 作为评分函数计算的 AUROC | 衡量认知不确定性质量（epistemic uncertainty） |
| **Softmax Entropy AUROC** | 使用 Softmax Entropy 作为评分函数 | 衡量偶然不确定性质量（aleatoric uncertainty） |
| **5% 性能下降阈值** | OOD AUROC 下降不超过 5% 的最大/最小容限 | 定义“稳定运行范围” |

---

### **基线方法对比**
- **Baseline**：无硬件约束的理想 BNN（软件仿真）
- **Direct inference on HW**：直接将在理想环境下训练的模型部署到硬件约束模型中（无硬件感知训练）
- **HW-aware training**：在训练阶段引入硬件约束
- **HW-aware + HW mods**：结合硬件修改（如扩大均值范围）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2 和 Figure 4）**

#### **稳定约束范围（ResNet-18 / CIFAR-10，OOD AUROC 下降 <5%）**
| 约束 | W-Add | W-Mul | A-Add | A-Mul |
|------|--------|--------|--------|--------|
| **输入量化（bit）** | ≥4 | >4 | >4 | >4 |
| **均值上限 \|μ_max\|（signed）** | >0.05 | — | >0.003 | >9400 |
| **标准差下限 σ_min** | <2.0 | <0.45 | <0.9 | <0.45 |
| **标准差上限 σ_max** | >0.03 | >0.008 | >0.11 | >0.008 |
| **均值编程噪声 σ_μ** | <2.1 | <0.9 | <80 | <220 |
| **标准差编程噪声 σ_σ** | <2.0 | <0.32 | <180 | <0.42 |

> 注：Activation-space 对均值范围要求极高（>9400），远超 weight-space（>0.003），这是关键发现之一。

---

### **与基线方法的对比结果（Figure 4）**

| 配置 | 准确率 | OOD AUROC | 是否恢复性能 |
|------|--------|------------|--------------|
| **Baseline** | 高 | 高 | — |
| **Direct inference on HW (weight)** | 显著下降 | 显著下降 | ❌ |
| **HW-aware training (weight)** | 恢复接近 baseline | 恢复接近 baseline | ✅ |
| **Direct inference on HW (activation)** | 极低 | 极低 | ❌ |
| **HW-aware training (activation)** | 不收敛（NaN） | 不收敛 | ❌ |
| **HW-aware + HW mods (activation)** | 显著恢复 | 显著恢复 | ✅（需硬件修改） |

> 结论：**weight-space 随机性** 可通过训练补偿；**activation-space 随机性** 必须先放宽均值范围才能训练成功。

---

### **消融实验结果**
- **随机性位置**：
  - **Weight-space**：提供更强的不确定性估计（更高 AUROC）
  - **Activation-space**：对均值编程误差更鲁棒，但不确定性较弱且不稳定
- **随机性模态**：
  - **Additive**：更能容忍低 σ_min 和编程噪声
  - **Multiplicative**：更能容忍小 σ_max
- **符号性（Signedness）**：
  - 无符号表示严重损害 weight-space 性能，导致 activation-space 模型完全崩溃
- **输入量化**：
  - 可通过 **quantization-aware training** 补偿至约 4 bit

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **光子 BNN 的可扩展性取决于“可表示的变分分布族”是否完整**，而非单一约束。
2. ✅ **Weight-space 随机性更适合高质量不确定性估计**，而 **activation-space 更抗噪声但不确定性弱**。
3. ✅ **输入量化 ≥4 bit 可通过训练补偿**，无需硬件升级。
4. ⚠️ **Activation-space 要求极高的均值动态范围**（>10^4），是主要瓶颈。
5. ⚠️ **无符号表示是硬性限制**，尤其对 activation-space 是灾难性的。
6. ✅ **硬件感知训练（HW-aware training）可有效补偿多数容忍范围内约束**。
7. 🔧 **超出容忍范围时（如均值范围不足），必须进行针对性硬件修改**，而非全面重构。

---

### **方法的局限性**
- **独立分析假设**：实验中各约束被独立变化，但现实中可能耦合（如均值与方差范围相互影响）。
- **分布假设**：默认噪声为高斯分布，未深入研究分布形状失配（如 Bose-Einstein）的影响。
- **样本相关性**：假设样本独立，未显式建模时间或通道间相关性。
- **未测量实际延迟/功耗**：聚焦于准确性与不确定性质量，未报告光子硬件的实际加速比。

---

### **未来工作方向**
- 研究 **耦合约束下的联合优化方法**。
- 开发支持 **高动态范围、有符号表示** 的新型光子架构（如 differential 或 dual-rail encoding）。
- 探索 **非高斯噪声源**（如量子真空噪声）对变分推断的影响。
- 将框架扩展至 **其他物理概率计算平台**（如 memristive, MTJ-based）。
- 结合 **实际光子芯片测量数据** 进行闭环 co-design。

---

> **总结一句话**：  
> 本文建立了首个系统性的光子 BNN 协同设计框架，揭示了“哪些问题靠训练解决，哪些必须改硬件”，为构建可信赖、可扩展的光子贝叶斯智能提供了实用路线图。

</details>

---

### 4. [DiffusionGemma Technical Report](https://arxiv.org/abs/2608.00146)

**Authors**: DiffusionGemma Team, Adrien Ali Ta\"iga, James Assiene, Daniele Calandriello, Rahma Chaabouni, Jo\~ao Gante, Tamara von Glehn, Nate Keating, Chris Knutsen, Martin Kukla, Tianlin Liu, Ivan Lobov, Ofir Nabati, Jo\~ao Gabriel Oliveira, Nicolas Perez-Nieves, Nastasia Prutianova, Bobak Shahriari, Jean Tarbouriech, Pavel Tyletski, \c{C}a\u{g}lar \"Unl\"u, Cindy Wu, Glenn Cameron, Jerome Connor, Sertan Girgin, Maarten Grootendorst, Alon Levkovitch, Eliya Nachmani, Omar Sanseviero, Piotr Stanczyk, Quentin Berthet, Andrew Campbell, Cl\'ement Crepy, Valentin De Bortoli, Arnaud Doucet, Romuald Elie, Alexandre Galashov, Klaus Greff, Alexis Jacq, David Ruhe, Yu-Han Wu, Sebastian Flennerhag, Brendan O'Donoghue, George Scrivener, Shantanu Thakoor  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.00146v1  

#### Abstract
We introduce DiffusionGemma, an experimental open-weight language model that uses discrete diffusion to generate text at exceptionally high speed. Rather than decoding one token at a time, DiffusionGemma iteratively refines blocks of 256 tokens in parallel, avoiding the sequential decoding bottlenec...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# DiffusionGemma 技术报告核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **Autoregressive (AR)** 大语言模型在生成文本时存在严重的**顺序解码瓶颈**（sequential decoding bottleneck）。由于其逐token生成的特性，推理过程严重受限于内存带宽（memory-bound），尤其是在低并发请求场景下，计算单元利用率低下，导致单用户延迟高。

此外，尽管已有基于 **speculative decoding**（如多token预测 MTP）等技术尝试加速，但仍受限于 drafter 模型的串行生成能力或并行生成器的接受率下降问题。

### 提出的新方法与思路
论文提出了 **DiffusionGemma**，一种实验性的开源权重语言模型，首次将**离散扩散**（discrete diffusion）技术应用于高性能 MoE 架构的语言模型，并通过高效的两阶段训练流程实现从 AR 到 diffusion 的范式转换。

#### 核心创新点：
- **基于 diffusion 的并行生成机制**：  
  不再逐token生成，而是以 **256-token 的 canvas** 为单位进行迭代去噪（iterative denoising），在每个去噪步骤中并行更新所有 token，从而显著减少总前向传播次数（forward passes）。
  
- **无需从头预训练的高效迁移路径**：  
  采用 **warm-start** 策略，直接在已有的 **Gemma 4 26B A4B MoE** 模型基础上进行微调，避免了从零开始训练 diffusion 模型的巨大算力开销。

- **两阶段训练管道（Two-stage Training Pipeline）**：
  1. **监督微调（SFT）**：使模型适应双向注意力下的块状去噪任务；
  2. **采样器蒸馏与强化学习联合优化（Sampler Distillation & Reinforcement Learning, SD.RL）**：在线同步提升生成质量和推理效率，压缩去噪步数至极低水平（平均约12步）。

- **动态自适应计算（Dynamic & Adaptive Computation）**：  
  引入 **adaptive stopping** 机制，模型可根据任务复杂度自动调整去噪步数——简单任务快速收敛，复杂任务投入更多计算资源，实现质量与速度的智能权衡。

- **保留 AR 能力的混合模式潜力**：  
  尽管主模式为 diffusion，但模型仍可无缝切换回 AR 模式，支持“按需路由”或“hybrid decoding”，为未来灵活部署提供可能。

### 相比现有方法的优势
| 维度 | DiffusionGemma | 传统 AR 模型 | Speculative Decoding (MTP) | 其他开放 diffusion 模型 |
|------|----------------|-------------|----------------------------|------------------------|
| **生成速度** | ✅ 极快 (~1,500 TPS) | ❌ 慢 | ⚠️ 中等提升 (~3–6 TPF) | ⚠️ 较慢或未达实用 |
| **推理效率 (TPF)** | ✅ ~20 TPF | ❌ 1 TPF | ⚠️ ~1.4–3 TPF | ⚠️ <5 TPF |
| **架构开放性** | ✅ 开源权重 (Apache 2.0) | ✅ 多数开源 | ✅ 多数开源 | ✅ 部分开源 |
| **功能完整性** | ✅ 支持 thinking mode、多模态、长上下文 | ✅ 完整 | ✅ 完整 | ⚠️ 功能有限 |
| **训练成本** | ✅ <10% 原始 AR 训练预算 | ❌ 极高 | ❌ 极高 | ⚠️ 高 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖多个领域，构成综合评估套件：
- **数学推理**：AIME 2026, GSM8K, MGSM, Putnam, HiddenMath
- **代码生成**：LiveCodeBench-v6, Codeforces, HumanEval, BigCodeBench, LBPP(v2), Natural2Code
- **常识与专业知识**：GPQA-Diamond, BIG-Bench, MMMLU, MMLU-Pro
- **多模态理解**：MMMU-Pro
- **指令遵循与代理行为**：IFEval, Tau-bench（Retail/Airline/Telecom）
- **下游微调任务**：Sudoku puzzles, PubMedQA

### 实验设置与评估指标

#### 主要评估维度：
- **生成质量**：各基准任务上的准确率（Accuracy）、得分（Score）
- **推理效率**：
  - **Tokens Per Forward (TPF)**：每轮前向传播生成的有效 token 数量
  - **Tokens Per Second (TPS)**：每秒输出 token 数（排除 prefill 时间）
  - **Effective Denoising Steps**：加权平均去噪步数
  - **End-to-End Latency**：端到端生成延迟

#### 硬件配置：
- **DiffusionGemma 与 Gemma 4**：单张 NVIDIA H100 GPU（FP8 精度，batch size 1）
- **LLaDA 2.1 Flash 100B**：8× NVIDIA B200 GPUs
- **Nemotron Diffusion 14B**：单张 H100（bfloat16）
- **Mercury 2**：通过 OpenRouter API 黑盒估计

#### 对比基线方法：
| 类型 | 模型 |
|------|------|
| **AR 基线** | Gemma 4 26B A4B（含 MTP 加速） |
| **开放 diffusion 模型** | LLaDA 2.1 Flash 100B, Nemotron Diffusion 14B |
| **闭源 diffusion 模型** | Mercury 2（API 接入） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3 和 Figure 13）

| 指标 | DiffusionGemma (TD) | Gemma 4 AR (MTP) | 提升倍数 |
|------|---------------------|------------------|----------|
| **平均 TPF** | 19.74 | 1.40 | **~14.1×** |
| **平均 TPS** | **1,479** | 303 | **~4.9×** |
| **有效去噪步数** | ~12 | N/A | — |
| **总前向传播数** | 显著降低（<5% of AR） | 高 | — |

> 💡 **说明**：TPF 提升意味着只需约 1/20 的前向传播即可完成相同长度的生成，极大缓解内存瓶颈。

### 与基线方法的对比结果

#### 在三大能力领域的表现（Figure 13）：
| 能力领域 | DiffusionGemma (TD) | Gemma 4 AR (MTP) | Mercury 2 (High) |
|--------|--------------------|------------------|------------------|
| **推理与知识** | 75.0 | 84.7 | 80.0 |
| **代码生成** | 76.9 | 82.4 | 78.0 |
| **指令遵循与代理行为** | 66.5 | 175.8 | — |
| **输出速度 (TPS)** | **1,479** | 303 | 600 |

- 尽管在绝对性能上略低于原始 AR 基线（trade-off），但 **DiffusionGemma 在速度上取得压倒性优势**。
- 性能远超其他开放 diffusion 模型（如 LLaDA、Nemotron），接近甚至部分超越闭源的 Mercury 2。
- 在 **低批量（low-batch）场景下吞吐量全面领先**，仅在约 32 并发以上才被 AR 模型反超（Figure 12）。

### 消融实验结果（Ablation Studies）

#### （1）SD.RL 训练的影响（Figure 8, 9）
- **引入 SD.RL 后**：
  - 平均奖励 ↑ 10 points
  - TPF 从 ~5 → **~20**
  - 有效去噪步数 ↓ 4×
- 表明 **SD.RL 成功实现了质量与效率的双重优化**。

#### （2）thinking mode 的影响（Table 3）
- 开启 thinking mode 可提升复杂任务表现，但会略微增加生成长度和延迟。
- 有趣的是，**SD.RL 导致模型输出更简洁**（concise output），反而提升了整体推理效率。

#### （3）下游微调效果（Table 5, 6）
| 任务 | 基础模型准确率 | LoRA 微调后准确率 | 去噪步数变化 |
|------|---------------|------------------|------------|
| **Sudoku** | 0.0% | **84.4%** | 40.65 → 10.72 |
| **PubMedQA** | 75.6% | **76.6%** | 18.09 → 31.57 |

- 表明 DiffusionGemma **可通过轻量级微调（如 LoRA）快速适配特定任务**，且对结构化任务（如 Sudoku）有巨大潜力。
- Sudoku 上从完全失败到超过 80% 准确率，证明了 diffusion 在非自回归任务中的天然优势。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Discrete diffusion 是实现超高推理速度的有效路径**：  
   通过并行去噪机制，成功绕过 AR 模型的内存瓶颈，在保持较强智能的同时实现 **~1,500 TPS** 的惊人速度。

2. ✅ **Warm-start + 两阶段训练是高效迁移的关键**：  
   无需从头训练，仅用不到 10% 的原训练预算即可将 AR 模型转化为高性能 diffusion 模型。

3. ✅ **SD.RL 实现了质量与效率的协同进化**：  
   通过联合优化 reward 与 sampler distillation，不仅提升生成质量，还主动压缩去噪轨迹，形成“隐式课程学习”。

4. ✅ **动态自适应计算成为现实**：  
   adaptive stopping 使得模型能根据任务难度自动调节计算量，真正实现“智能变速”。

5. ✅ **保留 AR 能力打开混合解码大门**：  
   模型可在 diffusion 与 AR 模式间自由切换，为未来构建 hybrid decoding pipeline 提供基础。

### 局限性
- **性能略低于原始 AR 基线**：由于跳过了原生 diffusion 预训练，且训练目标偏向低延迟，导致绝对性能略有损失。
- **输出过于简洁**：SD.RL 鼓励简短输出，牺牲了部分通过长链推理获得的能力增益。
- **偶发重复/卡顿现象**：在极端低延迟设置下可能出现局部 stuttering（如重复 token）。
- **多模态任务中偶尔遗漏 `</think>` 标签**：影响某些 benchmark 的评分。
- **高批量吞吐劣势**：在大批量服务场景下，更高的 per-token compute 成本使其失去优势。

### 未来工作方向
- 探索 **原生 diffusion 预训练** 路径，进一步缩小与 AR 模型的性能差距。
- 发展 **hybrid decoding 策略**，结合 AR 的精确控制与 diffusion 的高速优势。
- 优化 **batch-size 可扩展性**，改进采样内核以支持更高并发。
- 推动 **社区共建**，鼓励基于 DiffusionGemma 开发专用工具、新型采样器和垂直应用（如 ASR、医疗报告生成等）。

---

> 📌 **总结一句话**：  
> **DiffusionGemma 首次验证了基于离散扩散的大语言模型可以在保持强大能力的同时实现数量级的速度飞跃，并通过开源释放其潜力，标志着生成式 AI 进入“速度-智能”新帕累托前沿的时代。**

</details>

---

### 5. [SeDeM: Selective Decompression of Hidden-State Memories for Long-Context Question Answering](https://arxiv.org/abs/2608.00311)

**Authors**: Maryam Haghifam, Jason Cong, Yizhou Sun  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.00311v1  

#### Abstract
Long-context inference with large language models (LLMs) is costly: self-attention during prefill scales quadratically with sequence length, and the key-value (KV) cache grows with the number of processed tokens. Larger context windows also do not ensure reliable evidence use. Context compression re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SeDeM: Selective Decompression of Hidden-State Memories for Long-Context Question Answering**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

长上下文推理在大型语言模型（LLM）中面临两大挑战：

- **计算成本高**：自注意力机制在预填充（prefill）阶段的时间复杂度随序列长度呈平方增长，且 Key-Value (KV) 缓存随处理的 token 数量线性增长。
- **信息利用不可靠**：即使扩大上下文窗口，LLM 也常常无法有效利用长输入中的相关证据，尤其是当关键信息不在开头或结尾时。

此外，现有的**软压缩方法**（soft compression）通常将 LLM 自身作为压缩器，并依赖高度压缩的 memory tokens 同时完成两个任务：
1. 存储源文本信息；
2. 直接作为解码器的条件输入（decoder conditioning）。

这导致在高压缩比下，memory tokens 难以兼顾信息保真与生成引导，限制了性能。

---

### ✅ **提出了什么新方法或新思路**

作者提出 **SEDEM**（Selective Decompression of Hidden-State Memories），一种**选择性解压缩框架**，其核心思想是：

> **将紧凑内存的存储与解码器条件化过程解耦**（decouple storage from decoder conditioning）。

具体流程如下：

1. **Extractor**：冻结的 LLM 从中间 Transformer 层提取每个上下文段落的 `hidden states`。
2. **Compressor**：轻量级非注意力模块通过局部均值池化和投影，将这些 `hidden states` 压缩为固定大小的 **memory blocks**。
3. **Selector**：基于查询（query）的可学习模块选择最相关的 memory blocks。
4. **Decompressor**：仅对选中的 blocks 进行“解压缩”，将其扩展为与目标 decoder layer 兼容的 `hidden states`。
5. **Injection**：将重构的状态注入到解码器的中间层（而非输入层），供后续自回归生成使用。

---

### ✅ **相比现有方法的优势**

| 优势维度 | 说明 |
|--------|------|
| **效率更高** | 只需处理选中的 memory blocks，避免全上下文处理；支持并行压缩与早期退出（early exit）编码。 |
| **更可靠的信息传递** | 解压缩后的状态与 decoder 内部激活分布对齐，比直接使用压缩 token 更适合生成。 |
| **支持跨模型架构** | 小模型编码 → 大模型解码成为可能，分离上下文处理成本与生成能力。 |
| **可复用性** | memory bank 是 query-independent 的，同一文档可被多个 query 复用。 |

---

## 2. **核心实验方法和设置**

### ✅ **使用的数据集**

四个主流长上下文问答基准，涵盖不同长度与推理难度：

| 数据集 | 类型 | 平均长度（token） | 最大长度（token） |
|-------|------|------------------|------------------|
| **HotpotQA-Distractor** | 2-hop QA | 1,299 | 3,711 |
| **2WikiMultiHopQA (2WikiMHQA)** | 多跳 QA | 834 | 7,000 |
| **MuSiQue** | 2–4 跳 QA | 2,288 | 6,026 |
| **QASPER** | 长文档科学问答 | 5,248 | 34,768 |

所有上下文均分段处理，原始证据标注映射为 segment-level 标签用于训练 selector。

---

### ✅ **实验设置**

- **主干模型**：`Llama-3.2-1B-Base` 和 `Llama-3.2-3B-Base`（未指令微调的基础版本）
- **压缩配置**：
  - 段长度 $ T = 128 $
  - 压缩因子 $ C = 4 $ → 每段保留 32 个 memory slots
  - 存储压缩比：4×
- **选择预算 $ K $**：
  - HotpotQA & 2WikiMHQA: $ K=2 $
  - MuSiQue: $ K=4 $
  - QASPER: $ K=8 $

---

### ✅ **评估指标**

| 指标 | 说明 |
|-----|------|
| **Token-level F1**, **ROUGE-L F1** | 衡量答案质量，越高越好 |
| **Time-to-First-Token (TTFT)** | 在线延迟，越低越好 |
| **Autoregressive decoding throughput (tokens/sec)** | 解码吞吐量，越高越好 |

不启用任何缓存或预计算，完全在线执行。

---

### ✅ **基线方法对比**

| 类别 | 方法 |
|------|------|
| **Hard Prompt Compression** | LongLLMLingua |
| **Soft Memory Compression** | ICAE, HMT, 500xCompressor |
| **KV/Activation Compression** | Activation Beacon |
| **Full-context Reference** | 冻结 vs LoRA 微调 |

所有方法统一在相同 backbone 上实现，确保公平比较。

---

## 3. **主要实验结果和性能指标**

### ✅ **关键性能数据（来自 Table 2）**

#### 🔹 **Same-backbone setting (1B)**

| 方法 | 2WikiMHQA (F1) | QASPER (F1) | HotpotQA (F1) |
|------|---------------|-------------|----------------|
| Full-context (fine-tuned) | 49.36 | 21.42 | 29.16 |
| **SEDEM (Ours)** | **55.69** (+6.33) | **23.82** (+2.4) | **50.93** (+21.77) |

✅ **全面超越所有压缩基线及 full-context fine-tuned 模型**

#### 🔹 **Same-backbone setting (3B)**

| 方法 | 2WikiMHQA (F1) | QASPER (F1) | HotpotQA (F1) | MuSiQue (F1) |
|------|---------------|-------------|----------------|--------------|
| Full-context (fine-tuned) | 62.50 | 23.44 | 36.73 | 32.54 |
| **SEDEM (Ours)** | **67.25** | **26.74** | **58.30** | 21.85 |
| Cross-model (1B→3B) | 63.44 | 20.23 | 44.39 | 19.24 |

✅ 在 **3B 设置下，SEDEM 在 3/4 个数据集上超过 full-context fine-tuned**  
❗ 在 MuSiQue 上略低，表明多跳推理仍具挑战

---

### ✅ **效率表现（vs ICAE）**

| 数据集 | TTFT (ms) ↓ | Speedup | Throughput (tok/s) ↑ | Speedup |
|-------|-------------|---------|------------------------|---------|
| QASPER (3B) | 96.50 vs 364.10 | **3.77× faster** | 42.95 vs 40.69 | 1.06× |
| Avg (3B) | 79.60 vs 195.76 | **2.46× faster** | 45.16 vs 41.17 | 1.10× |

✅ 显著降低 **TTFT**，提升 **解码吞吐量**

---

### ✅ **消融实验结果**

#### 🔹 **是否需要解压缩？（Table 4）**

| 方法 | QASPER (F1) | HotpotQA (F1) |
|------|-------------|----------------|
| Direct memory conditioning (no decompression) | ~18.3 | ~30.1 |
| **SEDEM (with decompression)** | **26.74** | **58.30** |

➡️ **解压缩带来巨大增益**，证明“直接使用压缩 token”远不如“选择性解压后注入中间层”。

#### 🔹 **选择预算 $ K $ 的影响（Table 5）**

| 数据集 | 最优 $ K $ | 规律 |
|-------|------------|------|
| HotpotQA-Dist. | $ K=2 $ | 更大 $ K $ 引入噪声，性能下降 |
| 2WikiMHQA | $ K=8 $ | 更多 selected blocks 提升性能直至饱和 |

➡️ $ K $ 控制“质量 vs 上下文覆盖”的权衡，应根据任务调整。

#### 🔹 **跨任务迁移能力测试（Section 7.3）**

在未见过的 QASPER 上进行 zero-shot transfer（训练时不包含 QASPER 示例）：

| 方法 | Zero-shot QASPER (F1) |
|------|------------------------|
| ICAE | 11.30 |
| **SEDEM** | **20.02** |

✅ SEDEM 的 hidden-state memory 更具泛化能力。

#### 🔹 **LoRA 是否主导性能？（Table 6）**

| 设置 | HotpotQA (F1) |
|------|----------------|
| SEDEM + frozen decoder | 62.55 |
| SEDEM + LoRA-adapted decoder | 69.78 |

➡️ LoRA 有帮助，但主体增益来自 SEDEM 本身。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **解耦存储与条件化是关键**：让 memory blocks 专注信息存储，而由 decompressor 生成 decoder-friendly 的中间表示，显著优于直接使用压缩 token。
2. **选择性解压提升效率与效果**：只恢复相关部分，既节省计算又提高答案准确性。
3. **SEDEM 在多数场景下优于 full-context fine-tuning**：尤其在证据集中、无需全局访问的任务中表现突出。
4. **支持跨模型部署**：小 encoder + 大 decoder 架构可行，利于系统优化。

---

### ⚠️ **局限性**

1. **依赖黄金证据标签训练 selector**：若无 segment-level 标注，则无法有效训练 selector，只能退化为全量注入模式。
2. **未验证更大 backbone 上的表现**：当前实验限于 1B/3B 模型，尚不清楚在 7B+ 模型中是否仍能超越 full-context。
3. **layer depth 需经验设定**：`lextract` 和 `linject` 层的选择缺乏自动化原则。
4. **在复杂多跳任务（如 MuSiQue）上仍有差距**：可能因选择预算不足或证据分散所致。
5. **未采用预计算优化**：memory bank 可预先构建，但本文采用完全在线协议，实际部署潜力未充分挖掘。

---

### 🔮 **未来工作方向**

- 探索 **弱监督或自监督方式训练 selector**（如基于 consistency 或 contrastive learning）
- 设计 **动态或迭代式 selection**，根据初步生成反馈调整 $ K $
- 自动化 **extraction/injection layer selection**
- 扩展至更长上下文（>32K）、更多样任务（摘要、对话等）
- 结合 KV cache 压缩技术进一步优化端到端延迟

---

## ✅ 总结一句话

> **SEDEM 通过“选择性解压隐藏状态记忆”的设计，在保持高效的同时实现了优于全上下文微调的问答性能，揭示了“解耦压缩与生成条件化”是长上下文建模的关键路径。**

</details>

---

### 6. [SparseKAN: Compressing Kolmogorov--Arnold Networks Across Basis Functions, Neurons, and Bits](https://arxiv.org/abs/2608.00859)

**Authors**: Kazi Ahmed Asif Fuad, Lizhong Chen  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.00859v1  

#### Abstract
Kolmogorov--Arnold Networks (KANs) replace scalar edge weights with learnable univariate functions parameterized by multiple basis coefficients. This introduces a source of redundancy that conventional neural-network compression does not directly expose. We present \textbf{SparseKAN}, a unified appr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SparseKAN: Compressing Kolmogorov–Arnold Networks Across Basis Functions, Neurons, and Bits**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
Kolmogorov-Arnold Networks (KANs) 通过将传统神经网络中的标量权重替换为由多个**basis coefficients**参数化的可学习单变量函数，显著提升了模型表达能力。然而，这种设计也引入了新的冗余来源——不仅存在于连接和神经元之间，更存在于每个函数内部的**basis expansion**中。

传统的剪枝（pruning）和压缩方法（如结构化剪枝、量化）通常只作用于“边”或“通道”粒度，无法有效暴露并利用函数内部的冗余。这导致：
- 保留一条边可能仍包含大量无用的 basis terms；
- 删除整条边又可能误删有用的计算单元；
- 即使稀疏化后，模型物理尺寸和推理延迟并未显著降低（因仍保留原始张量形状）。

因此，如何在**basis 函数、神经元/通道、数值精度**三个维度上统一进行高效压缩，并将逻辑稀疏转化为物理紧凑，是当前 KAN 压缩领域尚未系统解决的问题。

---

### **提出了什么新方法或新思路**
本文提出 **SparseKAN**，一种面向 KAN 的统一压缩框架，其核心思想是：

#### ✅ **三轴联合压缩（Three-Axis Compression）**
SparseKAN 在以下三个互补维度上同时进行压缩：
1. **Basis Functions**：对每条边内的 basis terms 进行选择性保留；
2. **Neurons/Channels**：进行结构化通道剪枝；
3. **Bits**：应用低比特量化（如 8-bit / 4-bit QAT）。

#### ✅ **分阶段流程：结构发现 → 结构固化 → 物理紧凑化**
- **Soft Gating + Active-Cost Objective**：引入可学习的层级门控机制（base/branch/term-level gates），结合归一化的 `active-cost` 损失函数，在训练中自动发现重要结构。
- **Hardening with Budget Constraints**：在推理前通过硬阈值和预算控制（basis budget $k$、neuron keep ratio $r_n$）显式施加压缩。
- **Physical Compaction**：将共享支持的 basis terms 聚合，死掉的维度切片移除，生成更小的**稠密张量**，而非保留稀疏掩码。

> 🔑 **关键创新**：将“功能冗余”转化为“可执行的小型化模型”，实现真正的软硬件协同优化。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | SparseKAN |
|------|----------|-----------|
| **压缩粒度** | 边级（edge-level）或通道级 | 支持到 **basis term 级别** |
| **结构灵活性** | 固定结构或随机裁剪 | 学习重要性，支持 per-edge 或 shared-k 动态选择 |
| **物理效率** | 稀疏掩码不减小张量大小 | **物理紧凑化** → 更小 tensor + 更快执行 |
| **多维协同** | 多数仅关注单一维度（如仅量化或仅剪枝） | **统一接口协调 basis、width、bit 三轴压缩** |

此外，SparseKAN 是**通用框架**，适用于多种 KAN 变体（spline, polynomial, RBF, wavelet, convolutional KANs）。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **图像分类任务**：
  - **MNIST**
  - **CIFAR-10**
  - **CIFAR-100**
- **补充实验**：Dry Bean, JSC, Traffic, Wine 等 tabular 数据集（见附录）

### **主要 KAN 模型家族**
- **MLP 类型**：
  - EfficientKAN（B-spline basis）
  - KAGN（Gram-polynomial basis）
  - ChebyKAN（Chebyshev polynomial）
  - FastKAN（RBF）、WavKAN（wavelet）等
- **卷积类型**：
  - KAGN-conv（spatial-kernel Gram-polynomial convolution）

---

### **实验设置和评估指标**

#### 📌 **训练协议**
- **三种子实验**：
  1. **Frontier Study**：3 seeds（42,43,44），共 380 次运行，绘制压缩前沿；
  2. **五种子成对消融实验**：5 seeds（42–46），用于统计显著性分析；
  3. **鲁棒性审计（Robustness Audit）**：验证关键结论是否稳定。

#### 📊 **评估指标**
| 指标 | 含义 |
|------|------|
| `p_struct` | 归一化 active cost（结构成本） |
| `p_bit` | 联合 bit-cost：$ p_{\text{bit}} = p_{\text{struct}} \times (b / 32) $ |
| 参数减少率 | 实际 compact checkpoint 的参数下降百分比 |
| Top-1 准确率 | 主要性能指标 |
| CUDA 推理延迟 | 大批量下的 GPU 执行时间 |
| FPGA/HLS 实现延迟与资源占用 | 在 ZCU104 上综合评估硬件效率 |

#### ⚖️ **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Edge-Lo** | 匹配 cost 的边级稀疏化（low-rank approximation） |
| **L1+Entropy Node Pruning** | 传统节点剪枝方法 |
| **Dense PTQ** | 直接后训练量化（Post-Training Quantization） |
| **SHARe-KAN / MetaCluster** | 已有 KAN 压缩方法（向量量化、聚类） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **压缩效果（物理参数减少）**
| 模型 | 压缩点 | 参数减少 | 准确率变化 |
|------|--------|----------|------------|
| MNIST EfficientKAN | `sk4+n0.5` | **73.0%** | +0.93 pt |
| MNIST KAGN | `sk2+n0.5` | **70.3%** | -0.10 pt |
| CIFAR-10 KAGN-conv | `sk3+n0.7` | 59.8% | -2.92 pt |
| CIFAR-100 KAGN-conv | `sk3+n0.7` | 58.4% | -1.27 pt |

> 💡 在 MNIST 上实现 **零精度损失下删除超过 70% 参数**。

---

#### ✅ **CUDA 推理加速**
| 模型 | Batch Size | Latency Ratio (vs Dense) |
|------|------------|---------------------------|
| CIFAR-10 KAGN-conv (`sk2+n0.5`) | 1024 | **0.51×** |
| MNIST EfficientKAN (`sk4+n0.5`) | 1024 | **0.82×** |

> ❗ 对比：masked 模型延迟仍在 0.95–1.03×，说明**物理紧凑化至关重要**。

---

#### ✅ **FPGA/HLS 硬件性能提升**
在 **ZCU104** 上部署：
| 模型 | 设计 | 推理延迟降低 | DSP 占用降低 |
|------|------|--------------|---------------|
| EfficientKAN | Sparse+int4 | **23.63×** | — |
| GRAM MLP | Sparse+int4 | **18.21×** | — |
| KAGN-conv | Sparse+int4 | 1.51× | **从 96.2% → 29.1%** |

> 📌 KAGN-conv 虽然串行延迟改善有限，但释放出大量 DSP 和 BRAM 资源，可用于模型复制或多任务并行。

---

#### ✅ **量化表现**
| 设置 | 8-bit QAT | 4-bit QAT | 4-bit PTQ |
|------|---------|---------|---------|
| MNIST | 几乎无损（<0.5 pt） | -0.12~0.43 pt | 97.15%（可用） |
| CIFAR-10 | -0.5~1.5 pt | -1.39~3.70 pt | **50.19±11.78%**（崩溃） |
| CIFAR-100 | ~2.4–2.9 pt 下降 | 同左 | **9.04±2.43%**（完全失效） |

> 🔍 **结论**：8-bit QAT 高度鲁棒；4-bit 必须配合 QAT，否则严重退化。

---

### **与基线方法的对比结果**

| 对比项 | SparseKAN 表现 | 基线表现 | 结论 |
|--------|----------------|----------|-------|
| vs Edge-Lo（同 cost） | 准确率相近，有时略优 | 相当 | **可行性成立，非精度主导** |
| vs Node Pruning | 在 CIFAR-10/100 上达到或超越代表点 | 略逊或相当 | **竞争力强且支持后续 compact** |
| vs Matched-param MLP | KAN 在 CIFAR-10 显著胜出（90.73% vs 44.39%） | MLP 在 MNIST 更好（98.22% vs 97.44%） | **KAN 压缩价值取决于任务收益是否覆盖开销** |

---

### **消融实验结果**

#### 🔍 **Basis Selection 方式比较（Table 2 & 13）**
| 模型 | Coeff-based vs Truncation | Coeff-based vs Random |
|------|----------------------------|------------------------|
| MNIST KAGN | **+15.25 pt** | +0.59 pt |
| CIFAR-10 KAGN-conv | **+8.47 pt** | +3.57 pt |
| MNIST ChebyKAN | -0.04 pt | +0.20 pt |
| EfficientKAN (B-spline) | ≤0.33 pt 差异 | ~持平 |

> ✅ **重要发现**：在 Gram-polynomial 模型中，**基于系数的重要性选择远优于低阶截断**；但在 B-spline 中差异不大 → **basis-dependent sensitivity**。

#### 🔍 **门控训练必要性（Table 10）**
- 5/6 实验中 gate-trained vs gate-free 的 CI 包含零；
- 最大增益约 +1.15 pt（CIFAR-100 4-bit），但单种子反转；
> ❌ **结论**：门控本身不是精度来源，而是提供了一个**统一的结构搜索与控制接口**。

#### 🔍 **Joint QAT vs Two-Stage**
- Joint gate/QAT 在 CIFAR-10 上平均高 +4.45 pt，但标准差大（4.19）；
- 延长 two-stage 的恢复训练可消除差距；
> ⚠️ **结论**：joint 是**稳健的单管道选项**，但非必然精度更高。

---

## **4. 关键结论和发现**

### **主要发现**
1. **KAN 冗余具有多轴特性**：
   - 可分别在 **basis functions、neurons/channels、bits** 上独立压缩；
   - 三者组合时成本可**预测地复合**（如 0.465 × 0.556 ≈ 0.2584）。

2. **Basis Identity Matters（在某些情况下）**：
   - 在 Gram-polynomial KAN 中，**term 选择策略严重影响精度**；
   - 低阶截断会导致高达 **15.25 accuracy points 的损失**；
   - 而 B-spline 模型对此不敏感。

3. **物理紧凑化是通往高效的关键**：
   - 仅稀疏掩码无法降低 CUDA/FPGA 延迟；
   - **结构兼容的支持可通过 slicing + gathering 转化为小型稠密模型**；
   - 实现高达 **23.63× 推理加速** 和 **73.0% 参数削减**。

4. **量化行为依赖任务复杂度**：
   - **8-bit QAT 广泛鲁棒**；
   - **4-bit 需要 QAT 适配**，尤其在卷积 KAN 上，PTQ 完全失效。

5. **SparseKAN 是一个通用接口而非精度突破器**：
   - 不宣称在所有场景下超越基线；
   - 核心价值在于提供 **basis-aware 的结构控制 + 物理可执行性**。

---

### **方法的局限性**
1. **并非所有 basis 类型都受益于细粒度选择**：
   - 如 B-spline 模型对 term identity 不敏感，限制了 basis-level 压缩的价值。

2. **硬件瓶颈转移**：
   - 对于卷积模型（如 KAGN-conv），即使压缩后，**basis construction 和 spatial conv 仍是瓶颈**，难以进一步降低串行延迟。

3. **极端压缩需更强恢复训练**：
   - 如 CIFAR-10 上 `sk2+n0.5` 初始仅 67.48%，延长恢复至 40 epoch 可升至 71.13%。

4. **当前 FPGA 实现未探索并行扩展**：
   - 资源释放带来的 headroom 尚未用于多引擎复制或更大模型部署。

---

### **未来工作方向**
1. **融合 basis construction 与 contraction**：
   - 开发 fused kernel，避免先构造再压缩的开销。

2. **自动化混合精度搜索**：
   - 当前支持 per-layer bit 变量学习（Appendix A.7），可进一步发展为全自动 mixed-precision pipeline。

3. **扩展至其他函数逼近架构**：
   - 如 Symbolic Regression、Function Discovery 等 KAN 应用场景。

4. **探索动态稀疏执行机制**：
   - 虽然当前采用静态 compact，未来可在支持动态跳过的硬件上实现 input-adaptive execution。

5. **端到端编译器集成**：
   - 将 SparseKAN pipeline 集成进 ML 编译器（如 TVM、IREE），实现从训练到部署的无缝衔接。

---

> ✅ **开源地址**：https://github.com/OSU-STARLAB/SparseKAN

</details>

---

### 7. [Zellige: Moldable Sequence Placement for Mixed Image-Video DiT Training](https://arxiv.org/abs/2608.01150)

**Authors**: Guangyu Xiang, Xueze Kang, Minwei Zhao, Yuxin Wang, Shaohuai Shi, Lin Zhang, Xiaowen Chu  
**Category**: cs.DC  
**Published**: 2026-08-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.01150v1  

#### Abstract
High-quality video generation requires training Diffusion Transformers (DiTs) jointly on image and video data, posing a mixed-length sequence training problem across GPUs. Existing systems rely on data parallelism (DP), context parallelism (CP), or their combination; we model these designs as disjoi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Zellige: Moldable Sequence Placement for Mixed Image-Video DiT Training

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代高质量视频生成依赖于在**图像和视频混合数据上联合训练 Diffusion Transformers (DiTs)**。由于视频序列极长（可达数十万tokens），而图像较短，这种**混合长度序列（mixed-length sequence）的分布式训练**带来了显著的系统挑战：

- **计算负载不均衡**：注意力计算复杂度为 $O(L^2)$，长视频远超等量token图像的计算开销，导致某些GPU成为straggler。
- **通信冗余**：现有方法如 Context Parallelism (CP) 对所有序列强制切分，短图像也需通信，浪费带宽。
- **已有方案存在根本权衡**：如 KnapFormer 使用 disjoint-group placement，在组间负载均衡与组内通信冗余之间无法兼顾。

---

### 🚀 提出的新方法：Zellige

Zellige 是一个**可塑性序列放置系统（moldable sequence placement system）**，其核心思想是将每个序列视为“可塑任务”（moldable task），允许动态选择其并行配置（parallelism configuration）和参与的 GPU 集合（participating ranks），从而打破 disjoint-group 的限制。

#### 主要组件：
1. **Hardware Profiler**
   - 离线测量每种并行配置下的执行时间和内存消耗。
   - 支持精确建模实际运行时行为（含 kernel fusion、overlap 等）。

2. **Two-Stage Planner（两阶段规划器）**
   - **Anchor Placement**：使用 CP-SAT 求解器对高成本 anchor 序列（通常是长视频）进行最优负载均衡。
     - 引入 **Exact Compact Formulation (ECF)** 减少等价变量，提升求解效率。
   - **Filler Packing**：将轻量 filler 序列（通常是图像）以整块形式填入剩余资源空隙中，避免不必要的通信。

3. **Coalesced Attention Engine**
   - 在单卡上统一调度整序列（whole sequences）与分布式的 attention shards。
   - 支持 block-diagonal attention 和批量 all-to-all 通信，降低小核启动和集体通信开销。

---

### 🔍 相比现有方法的优势

| 方法 | 缺陷 | Zellige 如何改进 |
|------|------|----------------|
| **Naive DP / AdaptiveLoad** | 长视频仍驻留单卡，造成严重负载倾斜 | 可跨卡切分长视频（anchors），实现真正负载均衡 |
| **USP** | 所有序列都强制 CP 切分，图像通信开销大 | 仅对 compute-heavy 视频切分，图像保持完整 |
| **KnapFormer** | disjoint-group 设计导致组间不平衡 + 组内冗余通信 | 允许 rank sets 重叠，打破组边界，灵活分配 |

> ✅ **理论证明**：论文在 Section III 形式化证明了 disjoint-group placement 存在**根本性权衡**——要么组间负载不均，要么组内通信冗余，而 Zellige 通过 moldable placement 跳出该框架。

---

## 2. 核心实验方法和设置

### 📊 数据集与工作负载
- 构造多种混合 batch，包含：
  - 图像分辨率：256p ~ 1080p
  - 视频分辨率与时长：480p~1080p，10秒 或 15秒
- 控制 **video-token share（视频token占比）从 35% 到 60%**
- 使用真实训练场景启发的数据分布，参考 Wan 和 HunyuanVideo 设置，并借鉴 OpenVid-1M、Koala-36M 大规模视频数据集。

### ⚙️ 实验平台
| 测试床 | GPU 数量 | 型号 | 节点数 | 网络 |
|--------|----------|-------|--------|------|
| Testbed 1 | 16 | NVIDIA Tesla A800 | 2 | InfiniBand |
| Testbed 2 | 32 | NVIDIA RTX A6000 | 4 | InfiniBand |

- 模型：Wan2.1-1.3B（DiT 架构）
- 启用 activation checkpointing
- 使用 DiffSynth-Studio 的 block 结构和 attention 路径

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Step Makespan** | 单个训练步的总耗时（主指标） |
| **Peak Allocated Memory** | 最大显存占用 |
| **Per-Rank Load Balance (max/mean)** | 衡量负载均衡程度 |
| **Communication Volume** | 注意力相关跨卡数据传输总量 |
| **Planner Solve Time** | 规划器生成策略的时间 |

### 🆚 基线方法
| 基线 | 类型 | 说明 |
|------|------|------|
| **AdaptiveLoad** | Compute-aware DP | 动态调整每卡 batch size，但不切分序列 |
| **USP** | Full CP | 所有序列均采用 Ulysses/Ring 进行 context parallelism |
| **KnapFormer** | Disjoint-group + CP/DP hybrid | 将 GPU 分成固定 disjoint groups，每组内部 CP，组间 DP |

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（End-to-End Step Makespan）

#### 在 16×A800 上（15秒视频，高负载）：
- Zellige 比 **KnapFormer 快 1.12–1.48×**（平均 1.25×）
- 比 **USP 快 1.63–2.06×**（平均 1.76×）
- **AdaptiveLoad OOM**（因单个长视频超出显存上限）

#### 在 32×A6000 上：
- Zellige 比 **KnapFormer 快 1.27–1.54×**（平均 1.42×）
- 比 **USP 快 1.72–2.45×**（平均 1.95×）
- **AdaptiveLoad 再次 OOM**

> 💡 结论：Zellige 不仅更快，还能处理其他方法无法容纳的大序列。

---

### 📉 资源利用分析（High-load V50 工作负载）

| 指标 | Zellige | USP | KnapFormer | AdaptiveLoad |
|------|--------|-----|------------|---------------|
| **Attention 通信量（归一化）** | 0.25× | 1.0× | 0.65× | 0.0× |
| **Per-rank 时间 max/mean 比值** | 1.00× | 1.00× | 1.21× | 2.34× |

- Zellige 通信量仅为 USP 的 **25.4%**，且实现了完美负载均衡（max/mean = 1.0）。
- AdaptiveLoad 虽无通信，但负载极度不均（最慢卡是平均的 2.34 倍）。

---

### 🔬 消融实验与模块验证

#### （1）Profile 准确性（21个计划测试）
| 指标 | MAE | MAPE |
|------|-----|------|
| Step Makespan | 2.5 秒 | **3.4%** |
| Peak Memory | 0.4 GiB | **1.5%** |

✅ 表明硬件 profiler 具有高度预测准确性。

#### （2）Planner 效率（启用 ECF vs 不启用）

| 方法 | 平均求解时间 | 是否成功 |
|------|--------------|----------|
| Joint + No ECF | >600s | ❌ 未完成 |
| Joint + ECF | 29.8s | ✅ 成功 |
| Two-stage + ECF | **33–119ms** | ✅ 成功 |

- 两阶段 planner 比 joint-reference **快 68–1487×**
- 且 **step makespan 开销 ≤ 0.32%**（平均仅 0.10%）

> ✅ 证明两阶段设计在极短时间内接近全局最优。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Disjoint-group placement 存在根本性瓶颈**：
   - 组间易出现负载不均（inter-group load imbalance）
   - 组内短序列被迫通信（intra-group communication redundancy）
   - 二者不可兼得（论文定理 1 & 2 严格证明）

2. **Moldable Sequence Placement 更优**：
   - 允许不同序列共享 GPU，打破 disjoint-group 限制。
   - 实现“长视频切分 + 短图像共存”的高效混合执行。

3. **Zellige 实现近似最优 + 实时调度**：
   - 两阶段 planner 在 **<120ms 内完成决策**，适合在线训练。
   - 性能逼近 joint-optimal，差距 <0.32%。

4. **端到端显著加速**：
   - 在多种配置下超越最强基线 KnapFormer 达 **1.12–1.54×**
   - 同时大幅减少通信量（低至 USP 的 25%）

---

### ⚠️ 局限性
- **依赖预定义的 executable placement options**：需要提前枚举合法的并行配置组合。
- **当前未支持自动扩展 catalog cut**：anchor/filler 划分基于静态规则，未来可学习优化。
- **假设同步训练步**：适用于标准 DiT 训练流程，异步或流式场景需进一步适配。

---

### 🔮 未来工作方向
1. **动态 catalog adaptation**：根据实时 workload 自动调整 anchor/filler 切分阈值。
2. **集成 into training framework**：作为通用调度插件嵌入主流 DiT 框架（如 Diffusers）。
3. **扩展至 inference 场景**：支持 mixed-length 图文生成推理的高效调度。
4. **结合模型压缩技术**：与量化、稀疏化协同优化整体吞吐。

---

> 🏁 **总结一句话**：  
> **Zellige 通过提出 moldable sequence placement 范式，打破了传统 disjoint-group 的系统瓶颈，在保证近似最优负载均衡的同时最小化通信开销，实现了混合图像-视频 DiT 训练的显著加速（最高达 1.54×），并具备高效的实时规划能力。**

</details>

---

### 8. [Don't Mix Rewards, Mix Policies: Policy Decomposition and Optimization for Multi-Reward RL](https://arxiv.org/abs/2607.29246)

**Authors**: Ruiming Liang, Yi Zhong, Yizhen Yuan, Yinan Zheng, Tianyi Tan, Tianyue Wang, Haiyun Guo, Jinqiao Wang, Xianyuan Zhan  
**Category**: cs.AI  
**Published**: 2026-08-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.29246v1  

#### Abstract
Modern large language models (LLMs) are expected not just to answer correctly, but to adapt their behavior to different human values and use cases. As a result, multi-reward reinforcement learning (RL) has become an increasingly important problem for LLMs, where each reward captures a different aspe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Don't Mix Rewards, Mix Policies: Policy Decomposition and Optimization for Multi-Reward RL*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大语言模型（LLMs）需要同时满足多种人类偏好（如正确性、安全性、格式合规等），这构成了**Multi-Reward RL**（多奖励强化学习）问题。然而，传统方法在奖励空间中进行加权组合（reward-space scalarization），存在以下严重问题：

- **对齐税（alignment tax）**：不同优化目标之间相互冲突，导致梯度干扰，难以同时优化多个维度；
- **尺度敏感性**：各奖励信号的量纲和分布差异大，手动调参困难；
- **不可控性**：训练完成后无法灵活调整偏好权重。

这些问题使得多目标对齐不稳定且效率低下。

---

### 提出的新方法：PRISM
作者提出 **PRISM**（**Policy-space Reward Integration via Sub-Policy Mixing**），其核心思想是：

> **“不要混合奖励，要混合策略”（Don’t mix rewards, mix policies）**

#### 方法设计要点：
- **Policy Decomposition**：
  - 为每个奖励 $ R_k $ 学习一个独立的 **positive policy** $ \pi_k^+ $，专注于捕捉该奖励鼓励的行为。
  - 引入一个共享的 **global negative policy** $ \pi^- $，统一建模所有奖励下的失败模式（即任何一项不达标即视为失败）。
- **Policy Composition**：
  - 在推理时通过 **logit-level 加权求和** 组合子策略：  
    $$
    z^* = \sum_{k=1}^N \alpha_k z_k^+ - \gamma z^-
    $$
  - 权重 $ \{\alpha_k\}, \gamma $ 可在推理阶段自由调节，实现无需重新训练的偏好控制。

#### 实现机制：
- 所有子策略共享同一个 LLM 主干，通过不同的 **prefix tokens** 区分；
- 使用 **asymmetric update scheme**：仅正向分支更新主干参数，负向分支只更新 prefix，防止主干被“污染”；
- 采用 **parallel-batch mixture sampling** 技术，在单次前向传播中并行处理所有分支，保持低延迟。

---

### 相比现有方法的优势
| 方面 | 传统方法（如 GRPO Sum/Product, GDPO） | PRISM |
|------|----------------------------------------|-------|
| **优化空间** | Reward/Advantage Space | **Policy Space** |
| **梯度干扰** | 高（多个目标竞争同一更新步） | 低（分离优化路径） |
| **可解释性** | 黑箱式奖励融合 | 显式的策略组合接口 |
| **可控性** | 固定 trade-off，需重训练 | **推理时动态调节偏好** |
| **样本效率** | 较低 | 更高（更快收敛） |

---

## 2. 核心实验方法和设置

### 数据集与任务设置

PRISM 在三个典型多奖励对齐场景下进行了验证：

| 任务 | 数据集 | 奖励类型 | 说明 |
|------|--------|---------|------|
| **Scientific Reasoning** | SciKnowEval（训练），GPQA & ScienceQA（测试） | 正确性（Correctness）、格式合规（Format） | 多选题作答，要求输出结构化思维链 |
| **Tool-use Reasoning** | ToolRL 数据集（训练），BFCL-v3（测试） | 正确性、格式、长度（Length） | 函数调用能力评估，涵盖非实时、实时、多轮对话 |
| **Helpfulness-Safety Alignment** | Alpaca（训练），HH-RLHF、PKU-SafeRLHF（测试） | 有用性（Useful）、无害性（Harmless） | 平衡帮助性和安全性的通用对齐任务 |

---

### 评估指标

| 任务 | 主要指标 |
|------|----------|
| 科学问答 | `Fmt`（格式准确率）、`Acc`（答案正确率）、`Joint`（两者均正确）、`Avg`（平均得分） |
| 工具调用 | `Fmt`（协议合规）、`Acc/R`（RLLA风格严格准确率）、`Acc/B`（BFCL语义准确率） |
| 帮助性-安全性 | `Useful`, `Harmless`, `Avg`（人工评分，越高越好） |

---

### 基线方法对比
- **GRPO-Sum**：基于组相对优势的奖励加权求和；
- **GRPO-Product**：奖励乘积聚合；
- **GDPO**：Group Reward-Decoupled Normalization，当前最先进的多奖励方法之一；

所有方法均在同一 backbone 上进行比较：
- DeepSeek-R1-1.5B
- Qwen2.5-1.5B-Instruct
- Qwen2.5-3B-Instruct

---

## 3. 主要实验结果和性能指标

### 科学推理任务（Table 1）
| Backbone | 方法 | Overall Score (Avg) | 提升幅度 |
|--------|------|---------------------|--------|
| DeepSeek-R1-1.5B | GDPO | 44.96 | — |
| | **PRISM (ours)** | **62.73** | **+17.77** ✅ |
| Qwen2.5-1.5B-Instruct | GDPO | 63.30 | — |
| | **PRISM (ours)** | **71.34** | **+8.04** ✅ |
| Qwen2.5-3B-Instruct | GDPO | 68.71 | — |
| | **PRISM (ours)** | **69.31** | **+0.60** ✅ |

> 🔹 在最难的 GPQA 上，PRISM 将平均分从 22.69 提升至 **47.55**（翻倍以上）  
> 🔹 在所有六种设置中，PRISM 均取得最高的 **Joint Score**，表明其能更好满足多重标准

---

### 工具调用任务（Table 2）
| 指标 | GDPO | PRISM (ours) |
|------|------|--------------|
| `Fmt` | 93.81 | **95.23** ✅ |
| `Acc/R` | 36.38 | 36.20 ⚖️ |
| `Acc/B` | 52.13 | **53.46** ✅ |
| **Overall Performance** | — | **排名第一** ✅ |

> 🔹 在 Non-live 和 Multi-turn 设置中全面领先  
> 🔹 展现出更均衡、鲁棒的表现

---

### 帮助性-安全性对齐（Table 3）
| 测试集 | GDPO | PRISM (ours) |
|--------|------|-------------|
| Alpaca (`Avg`) | 3.20 | **3.41** (+0.21) ✅ |
| HH-RLHF (`Avg`) | 3.53 | **3.65** (+0.12) ✅ |
| PKU-SafeRLHF (`Avg`) | 5.56 | **5.61** (+0.05) ✅ |

> 🔹 在所有三项上均达到最高分，且显著优于其他基线

---

### 消融实验（Ablation Studies）

#### A. 策略组成消融（Table 4）
| 变体 | Fmt | Acc/B |
|------|-----|-------|
| 完整 PRISM | 95.23 | 53.46 |
| w/ shared positive policy | 94.03 | 52.56 ↓ |
| w/o global negative policy | 94.86 | 52.84 ↓ |
| w/ individual-branch rollouts | 94.84 | 52.48 ↓ |

> ❗ 结论：**专用 positive policy** 和 **全局 negative policy** 对性能至关重要；**从组合策略采样** 更符合部署分布。

#### B. 负策略加权函数消融（Table 5）
| 加权方式 | Fmt | Acc/B |
|--------|-----|-------|
| 默认（soft conjunction） | 95.23 | 53.46 |
| max weighting | 95.02 | 52.43 ↓ |
| LogAvgExp | 95.06 | 52.62 ↓ |
| mean weighting | 93.27 | 53.12 ↓ |

> ❗ 结论：所提出的软逻辑“或”形式最有效捕捉联合失败模式。

---

## 4. 关键结论和发现

### 主要发现
1. **Multi-reward alignment tax 是真实存在的瓶颈**：将多个奖励压缩成单一信号会导致性能妥协。
2. **Policy-space composition 显著缓解对齐税**：将优化解耦到策略空间，避免梯度冲突。
3. **PRISM 实现高效且可控的多目标对齐**：
   - 单次训练 → 支持无限种偏好组合；
   - 推理时可通过调整 $ \alpha_k, \gamma $ 实现行为定向引导（见 Figure 5）；
   - 示例显示：强调 correctness 时直接完成购票，强调 format 或 length 时可能选择保守操作但结构完整。
4. **更强的扩展性与稳定性**：
   - 随着奖励数量增加，PRISM 性能稳定，而 GDPO/GRPO 出现明显下降（Figure 3）；
   - 收敛速度更快（约 3k 步达优，baseline 需 >6k 步，Figure 4）；
   - 推理延迟几乎不变（Table 9：PRISM N=3 仅比单策略慢 ~5%）。

---

### 方法的局限性
1. **假设限制**：
   - 每个奖励对应一个独立 positive policy 的有效性依赖于任务解耦程度；
   - 全局 negative policy 是否能充分覆盖复杂联合失败模式尚待更大规模验证。
2. **可扩展性挑战**：
   - 当奖励数 $ N $ 极大时，prefix 分支数 $ N+1 $ 导致内存和计算开销线性增长；
   - 当前 merge weights 为人工设定，缺乏自动适配机制。
3. **应用场景限制**：
   - 当前实验集中在最多 3 个奖励的任务，更多元复杂场景有待探索。

---

### 未来工作方向
1. **自动化权重调节**：研究如何根据 prompt 内容、用户画像或上下文动态生成最优 $ \{\alpha_k\}, \gamma $。
2. **更高效的组合架构**：探索稀疏激活、自适应路由等机制减少冗余计算。
3. **增强 negative policy 表达力**：引入分层或模块化 negative policy 设计。
4. **扩展至更大 reward 数量和更复杂任务**：如个性化推荐、多智能体协作等。

---

## 总结

✅ **PRISM 成功实现了从 “reward mixing” 到 “policy mixing” 的范式转变**，为多目标 LLM 对齐提供了更稳健、高效且可控的新框架。其实验结果一致证明其在科学推理、工具使用和帮助性-安全性对齐三大场景中全面超越现有方法，并具备出色的推理期可控性与抗干扰能力，是迈向真正可定制化 AI 助手的重要一步。

</details>

---

### 9. [FinHardBench: Can LLMs Generate Latency-Aware Hardware for Financial Computing?](https://arxiv.org/abs/2608.00909)

**Authors**: Weimin Fu, Hejia Zhang, Minghao Shao, Zeng Wang, Johann Knechtel, Ozgur Sinanoglu, Muhammad Shafique, Ramesh Karri, Xiaolong Guo  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.00909v1  

#### Abstract
Can large language models generate not just correct, but fast hardware? This paper investigates the question in financial FPGA design, where 5-10 nanoseconds of latency determines competitive advantage and designs iterate continuously as protocols, strategies, and regulations evolve. FinHardBench, a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# FinHardBench: Can LLMs Generate Latency-Aware Hardware for Financial Computing? —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文聚焦于**金融计算领域中基于 FPGA 的低延迟硬件设计迭代瓶颈**。在高频交易（HFT）场景下，协议变更、策略更新和风控要求频繁驱动 RTL 代码修改，传统人工迭代耗时数周，而市场需要在数小时内完成部署。论文提出并系统研究以下三个关键问题：
- **Module Generation**：LLM 能否从自然语言规范生成功能正确且时序高效的 Verilog？
- **System Configuration (DSE)**：LLM 能否在多级流水线中进行有效的 Design Space Exploration（DSE），以最小化端到端延迟？
- **Specification Adaptation**：LLM 能否根据需求变化对现有模块进行安全、高效的代码修改？

这些问题此前未被系统性建模与评估。

---

### 🚀 提出的新方法与创新点
- **FinHardBench**：首个面向金融计算的开源 LLM for HDL 综合基准，包含 **33 个任务**，覆盖五个抽象层级（L1–L5），涵盖从基础算术到期权定价等典型 HFT 模块。
- **三维能力评估框架**：
  - **模块生成（Module Generation）**
  - **系统级配置优化（System DSE）**
  - **规格自适应修改（Specification Adaptation）**
- **引入 post-P&R 时序评估**：首次将 **critical path delay** 和 **Fmax** 作为核心指标，在 Lattice ECP5 FPGA 上通过开源工具链（Yosys + NextPNR）进行全流程验证，实现可复现的 timing-aware 评测。
- **真实迭代周期模拟**：实验设计紧密贴合实际金融 FPGA 开发流程，强调“反馈驱动”的交互式优化过程。

---

### 🔍 相比现有工作的优势
| 特性 | 现有基准（如 VerilogEval, RTLLM, ResBench） | FinHardBench |
|------|----------------------------------------|-------------|
| 领域针对性 | 通用数字逻辑或部分金融任务 | 完全面向金融计算（HFT 流水线） |
| 时序评估 | 缺乏 post-P&R timing 或仅关注 ASIC PPA | 支持 FPGA 上 critical path 和 Fmax 测量 |
| 系统级 DSE | 不支持 | 支持 6 阶段流水线、432 种配置的空间探索 |
| 规格适应性 | 不支持 | 支持 15 类真实世界变更（协议/策略/风控等） |
| 可复现性 | 多依赖闭源工具 | 全流程使用开源工具链（Icarus/Yosys/NextPNR） |

> ✅ **FinHardBench 是目前唯一同时支持 spec-to-RTL、system DSE 和 spec-adaptation 并提供 post-P&R timing 评估的 HDL 生成基准**。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **FinHardBench 基准集**：共 **33 个任务**，分布如下：
  - **L1 Financial Arithmetic**（6项）：如 `TICK_ACCUM`, `RUNNING_AVG`
  - **L2 Trading Indicators**（6项）：如 `SMA`, `EMA`, `MACD`, `RSI`, `VWAP`
  - **L3 Protocol & Hashing**（6项）：如 `UDP Parser`, `FAST Decoder`, `CRC32`
  - **L4 Order Book & Matching**（5项）：如 `Order Book`, `Matching Engine`
  - **L5 Options & Risk**（10项）：如 `Black-Scholes Pricer`, `Option Delta`, `GREEKS_BUNDLE`

每个任务包含：
- 自然语言 specification
- 黄金参考实现（golden Verilog）
- 自检 testbench
- `critical_io.json`（定义关键路径引脚）
- `knobs.json`（参数化选项用于 DSE）

---

### ⚙️ 实验设置与评估指标

#### 实验一：Module Generation（模块生成）
- **输入**：自然语言 spec
- **输出**：完整 Verilog 模块
- **流程**：Syntax Check → Functional Simulation → Synthesis (Yosys) → P&R (NextPNR-ECP5)
- **评估指标**：
  - `Sim%`：功能仿真通过率（主指标）
  - `Routed%`：成功布线比例
  - `Fmax`（MHz）、`LUT count`
  - **Timing Ratio** = CritPath<sub>LLM</sub> / CritPath<sub>golden</sub>

#### 实验二：System Configuration (DSE)
- **目标**：在一个固定的 6-stage HFT pipeline 中选择每阶段微架构配置，最小化 end-to-end latency。
  - Pipeline: `UDP_RX → FAST_DEC → ORDER_BOOK → STRATEGY → RISK_CHK → MATCH`
  - 总共 **432 种组合**
- **方法**：LLM 在 24 轮内基于反馈迭代选择配置（不写代码）
- **评估指标**：
  - 最优延迟是否收敛（107.5ns）
  - 收敛轮次（First Optimum Round）
  - 成功种子数（Seeds reaching optimum / 5）

#### 实验三：Specification Adaptation（规格适配）
- **输入**：原始 Verilog + 当前 timing + 修改描述（如“将 SMA 替换为 MACD”）
- **过程**：3 轮反馈式修改
- **评估指标**：
  - 功能通过率（按类别统计）
  - 返回有效输出的比例（service availability）

---

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **Random Search** | 随机采样配置空间 |
| **Simulated Annealing (SA)** | 局部邻域搜索，指数降温 |
| **Bayesian Optimization (BO/TPE)** | 使用 Optuna 的 TPE 采样器 |
| **LLMs**（6个）：
  - GPT-4.5
  - Claude Sonnet 4.6
  - Gemini 3.1 Flash Lite
  - DeepSeek V3.2
  - Mistral Large 2512
  - MiniMax M2.7

所有方法共享相同的 P&R 缓存机制，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### 表：Module Generation 结果（33 tasks × 3 trials = 594 evals）
| Model | Sim% | Routed% | Avg Fmax (MHz) |
|-------|------|---------|----------------|
| **Claude Sonnet 4.6** | **61%** | **57%** | 165 |
| GPT-4.5 | 61% | 54% | 141 |
| Gemini 3.1 Flash Lite | 46% | 41% | 181 |
| DeepSeek V3.2 | 25% | 23% | 149 |
| Mistral Large | 23% | 20% | 149 |
| **MiniMax M2.7** | **19%** | **19%** | **231** |

> ❗ 尽管 MiniMax 生成质量最差，但其综合后 Fmax 最高，说明**资源利用率与功能正确性无强相关性**。

#### 时序退化严重集中在特定任务：
| Task | Timing Ratio (LLM/Golden) |
|------|----------------------------|
| **VWAP** | **13.67×** |
| RSI | 3.62× |
| MINMAX | 2.45× |
| FP_DIV | 1.59× |

> 🔍 **根本原因**：LLM 使用了组合除法 `/` 运算符而非迭代 FSM 实现 fixed-point division，导致逻辑层级爆炸（>100 levels），Fmax 从 46.5MHz 崩溃至 3MHz。

---

#### 表：System DSE 收敛结果（24 rounds, 5 seeds）
| Method | Mean Best Latency (ns) | First Optimum (Round) | Seeds @ 107.5ns |
|--------|------------------------|------------------------|----------------|
| **Claude Sonnet 4.6** | **107.5** | **11.2** | **5/5** |
| **GPT-4.5** | **107.5** | 15.8 | **5/5** |
| MiniMax M2.7 | 108.5 | 16.8 | 4/5 |
| DeepSeek V3.2 | 116.1 | 12.3 | 3/5 |
| SA | 108.5 | **7.5** | 4/5 |
| BO (TPE) | 117.1 | 12.0 | 2/5 |
| Random | 128.7 | – | 0/5 |

> ✅ **Top LLMs 在可靠性上超越所有经典算法**（5/5 vs SA 4/5, BO 2/5, Random 0/5）  
> ⚠️ SA 更快找到最优解（平均第 7.5 轮），但在某些 seed 上失败；LLMs 探索更稳健。

---

#### 表：Specification Adaptation 成功率（按类别）
| Model | Proto. | Strat. | Risk | MktSt. | Perf. |
|-------|-------|--------|------|--------|-------|
| **Claude Sonnet 4.6** | 50% | **42%** | 33% | 50% | 17% |
| GPT-4.5 | 67% | 0% | 33% | 50% | 17% |
| MiniMax M2.7 | 83% | 0% | 33% | 0% | 20% |
| Others | ≤50% | 0% | ≤44% | ≤17% | ≤17% |

> ✅ **只有 Claude 在策略级变更（如 MACD 添加）上有非零成功率（42%）**  
> ❌ 所有模型均未能解决性能优化类任务（如插入流水线提升 Fmax）

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **三大能力之间仅有中度重叠（moderate overlap）**
   - 最佳代码生成器 ≠ 最佳架构优化器
   - 示例：**MiniMax M2.7** 是最弱生成器（Sim% = 19%），但在 DSE 中达到 **4/5 种子收敛**
   - **DeepSeek V3.2** 生成排名第四，DSE 收敛速度第二快
   - ➤ 应采用“分工协作”范式：不同模型负责不同子任务

2. **任务难度更多取决于训练数据中的模式可用性，而非抽象层级**
   - 高层任务（如风险检查）因逻辑简单、模式常见而易解（POS_LIMIT 达 15/18 pass）
   - 低层任务（如 VWAP）因需高效 fixed-point division 而难解（仅 2/18 pass）
   - ➤ **公开 Verilog corpus 缺乏高性能数值算法实现**

3. **存在大量“可部署但错误”（deployable-but-wrong）的设计**
   - 占所有生成尝试的 **34.7%**（206/594）
   - 这些设计可通过 synthesis 和 P&R，报告正常 Fmax/LUT，但功能错误
   - 工程师无法仅凭工具链报告识别此类缺陷
   - ➤ **LLM 若产生此类设计，比完全失败更具危害性**

4. **LLMs 在 system DSE 中表现出优越的收敛可靠性**
   - 在 24 轮预算下，**Claude 和 GPT-4.5 实现 5/5 种子全部收敛至全局最优**
   - 超越 Random (0/5), SA (4/5), BO (2/5)
   - 显示 LLM 可作为强大的 **search prior**，结合反馈逐步优化

5. **Agentic 方法显著提升困难任务成功率**
   - Pilot 实验显示：Claude Code Agent 在多轮调试后成功实现 VWAP 和 RSI
   - 单次生成失败 ≠ 能力上限
   - ➤ **迭代 + 工具调用是突破当前瓶颈的关键路径**

---

### ⚠️ 方法的局限性

1. **工具链限制**：
   - 使用 Lattice ECP5 而非主流 Xilinx UltraScale+，尽管 cross-toolchain 验证显示 rank correlation 高达 **ρ = 0.982**
   - 仍缺乏对商业级器件和工具（Vivado）的大规模验证

2. **评估粒度不足**：
   - 未包含 gate-level simulation 或 FPGA-in-the-loop 测试
   - testbench 的 corner-case 覆盖有待加强

3. **DSE 模型简化**：
   - 忽略队列填充、背压传播、数据依赖延迟
   - 策略阶段理想化为单周期操作

4. **样本量小**：
   - 仅测试 6 个通用 LLM，未包含专用 fine-tuned 模型（如 VeriGen）
   - cross-experiment correlation 分析受限于 n=6，统计效力有限

---

### 🔮 未来工作方向

1. **构建专用 LLM for Financial HDL**
   - 在高质量金融 FPGA 设计语料上进行 domain adaptation
   - 强化对 fixed-point 数值算法、流水线优化等模式的学习

2. **发展 agentic design flow**
   - 构建具备仿真、综合、分析反馈闭环的自主代理系统
   - 支持自动修复 timing violation 和 functional bug

3. **扩展 FinHardBench 至更大规模 DSE 空间**
   - 加入 FIFO depth、clock gating、memory banking 等维度
   - 研究 LLM 在高维空间中的 sample efficiency

4. **引入安全与合规性验证**
   - 对风控模块增加形式化验证接口
   - 防止 LLM 引入潜在交易风险

5. **发布工业级黄金参考设计**
   - 与交易所、做市商合作获取真实生产环境中的低延迟 IP

---

## ✅ 总结一句话

> **FinHardBench 揭示：当前 LLM 在金融 FPGA 设计中并非“全能选手”，而是“专才集合”——最强生成器未必最擅优化，最弱生成器也能胜任系统配置；未来应走向“多模型协同 + 反馈驱动 + 工具增强”的智能设计新范式。**

</details>

---

### 10. [Conformalized Large Language Models under Configuration Shift](https://arxiv.org/abs/2608.01460)

**Authors**: Yuqicheng Zhu, Jialin Yu, Lin Li, Gengyuan Zhang, Zhen Yang, Steffen Staab, Puneet Dokania, Philip Torr, Jie Tang, Evgeny Kharlamov  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.01460v1  

#### Abstract
Conformal prediction (CP) is a distribution-free framework for uncertainty quantification that has recently been adapted to large language models (LLMs), providing prediction sets with finite-sample coverage guarantees under exchangeability. Yet for LLMs, nonconformity scores are often induced by an...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Conformalized Large Language Models under Configuration Shift

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文首次系统性地提出并研究了 **Configuration Shift**（配置偏移）对 **Conformal Prediction (CP)** 在 **Large Language Models (LLMs)** 中应用的影响。

传统 CP 的有效性依赖于 **exchangeability**（可交换性），即校准集和测试集的非一致性分数（nonconformity scores）来自同一分布。然而，在实际部署中，LLM 的推理配置（如 prompt template、decoding temperature、模型量化精度）经常被调整以适应不同任务或硬件限制。这些变化虽然不改变输入数据分布 $P_z$，却会显著改变非一致性分数的生成方式，从而破坏 exchangeability，导致 CP 的覆盖率（coverage）低于目标值。

### 提出了什么新方法或新思路
论文提出了以下三个层面的创新：

1. **概念定义**：正式定义了 **Configuration Shift**，将其与传统的 covariate shift、label shift 等区分开来，强调其是“分数映射函数”而非“数据分布”的变化。
2. **理论分析**：推导了在 Configuration Shift 下的覆盖率下界，表明覆盖率损失与校准和测试阶段的分数分布差异（如 Kolmogorov-Smirnov 距离 $d_{KS}$）直接相关。
3. **实用缓解策略**：
   - **Bound-inspired Recalibration (Recal)**：基于理论推导，提出一种门控式的重校准方法，仅当检测到显著偏移时才使用少量带标签的测试样本来重新计算阈值。
   - **Fragility-aware Calibration Ensembling (Mos-F, Anc)**：利用“脆弱性感知”的校准集成，优先选择那些在基准 CP 下容易导致欠覆盖（undercoverage）的配置进行校准，从而提升鲁棒性。

### 相比现有方法的优势
- **针对性强**：现有工作多关注数据分布偏移（如 covariate shift），而本文聚焦于实践中更常见但被忽视的**推理配置偏移**。
- **理论指导实践**：提出的缓解方法（如 Recal）直接源于理论分析，具有明确的动机和解释性。
- **高效且实用**：Recal 方法仅需极少量（如 k=50）带标签的测试样本即可恢复大部分覆盖率，适合实际部署场景。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验涵盖了 **4 个广泛使用的问答基准**，分为两类任务：
- **Multiple-Choice QA (MCQA)**：
  - **MMLU**：涵盖 57 个学科的多项选择题。
  - **MedMCQA**：医学领域的多项选择题。
- **Open-Ended QA (OEQA)**：
  - **Natural Questions (NQ)**：基于真实搜索查询的开放性问答。
  - **TriviaQA**：包含别名的事实型问答。

### 实验设置和评估指标
#### 配置偏移轴（Configuration Shift Axes）
研究了三个实际部署中常见的配置变化：
1. **Prompt Template Shift**：共 14 种（MCQA）和 7 种（OEQA）模板，涵盖格式、指令措辞和内容删减。
2. **Decoding Temperature Shift**：温度 $T \in \{0.5, 0.7, 1.0, 1.3, 1.5\}$，其中 $T=1.0$ 为参考。
3. **Weight Quantization Shift**：GGUF 量化精度 $Q8, Q6, Q4, Q2$，应用于三个大模型。

#### 非一致性分数（Nonconformity Scores）
- **White-box**（可访问 logits）：
  - **LAC**（1 - softmax 概率）
  - **Logit Margin**
- **Black-box**（仅文本输出）：
  - **Self-consistency**（1 - 采样频率）
  - **LoFreeCP**（结合频率、熵和语义相似度）

#### 评估指标
- **Undercoverage Rate (UT)**：覆盖率低于目标 $1-\alpha$ 的运行比例。
- **Set-size Inflation (Infl)**：在有效运行中，预测集大小相对于 i.i.d. 基线的膨胀倍数。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在目标置信度 $1-\alpha=0.9$ 下，Configuration Shift 导致：
- **Prompt Shift**：UT 从 i.i.d. 的 ≤0.03 上升至 **0.31**。
- **Temperature Shift**：UT 上升至 **0.46**（最严重）。
- **Quantization Shift**：UT 上升至 **0.20**。

> **效率方面**：尽管覆盖率下降，但在仍有效的运行中，预测集大小膨胀很小（Infl ∈ [1.00, 1.26]），说明主要问题是**有效性失效**而非效率降低。

### 与基线方法的对比结果
论文比较了 6 种缓解方法，关键结果如下（在 $1-\alpha=0.9$）：

| 方法 | Undercoverage Rate (UT) | Set-size Inflation (Infl) | 说明 |
|------|--------------------------|----------------------------|------|
| **Vanilla CP** | 0.319 (avg) | ~1.08 | 基线，严重欠覆盖 |
| **Reweight** | 无改善 | - | 传统加权法在此无效 |
| **Mos-U** | ↓ ~0.05-0.11 | ≈1.00 | 均匀集成已能吸收部分偏移 |
| **a-Inf** | ↓ 至 0.004–0.080 | ↑ 15–17% | 保守但低效（集合过大） |
| **Recal** | ↓ 至 **0.060–0.087** | ≈1.00–1.04 | **最佳平衡**，仅需 k=50 |
| **Mos-F** | ↓ 至 0.120–0.200 | ≈1.00 | 脆弱性加权优于均匀集成 |

> **Recal** 表现最优：在极小的集合膨胀下，将欠覆盖率降至接近零。

### 消融实验结果
- **Recal 的预算消融**（k=10 到 60）：当 k ≥ 20 时，性能基本饱和，证明小样本即可有效。
- **Anc 的池大小消融**（K=3,5,7）：K=3 时效果最好，过大反而稀释了脆弱配置的影响。
- **Per-Score Fragility**：白盒分数（LAC, Logit Margin）表现相似；黑盒分数中，**LoFreeCP** 比 **Self-consistency** 更脆弱。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Configuration Shift 是 CP 失效的重要根源**：即使数据不变，常规的 prompt、temperature 或量化调整也会显著降低 CP 的覆盖率。
2. **主要失败模式是有效性丧失**：表现为 **undercoverage**，而非预测集无限膨胀。
3. **更多校准数据会加剧问题**：随着校准集增大，阈值更精确地反映了有偏的校准分布，导致覆盖率进一步下降。
4. **理论驱动的缓解方法有效**：
   - **Recal** 通过门控机制，用极少测试标签即可恢复高覆盖率。
   - **Mos-U/F** 等校准集成方法无需测试标签，也能提供一定鲁棒性。

### 方法的局限性
1. **理论界限非有限样本保证**：推导的是总体界限，实验中的估计量是诊断工具，而非严格证书。
2. **缓解方法为初步设计**：未追求通用鲁棒框架，而是验证理论洞察的实用性。
3. **实验范围有限**：仅研究了单轴偏移，未涵盖多轴复合偏移（如 prompt + temperature 同时变）。

### 未来工作方向
- 开发针对 Configuration Shift 的**有限样本认证界限**。
- 构建更完整的**鲁棒校准框架**，支持任意配置索引的分数映射。
- 将分析扩展到**复合偏移**（compound shifts），如检索增强、智能体决策等复杂 LLM 系统。
- 探索如何在**训练阶段**就优化模型对配置变化的鲁棒性。

</details>

---

### 11. [DLLM-TTS: Block Discrete Diffusion Language Model for Text-to-Speech Synthesis](https://arxiv.org/abs/2608.00011)

**Authors**: Wasim Madha, Nityanand Mathur, Hamees Sayed, Apoorv Singh, Sameer Khurana, Akshat Mandloi, Sudarshan Kamath  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.00011v1  

#### Abstract
Current text-to-speech systems face a trade-off: autoregres- sive codec language models produce highly intelligible speech but require large-scale models and training data and decode tokens sequentially, while non-autoregressive approaches im- prove speed at the cost of linguistic accuracy. We prese...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DLLM-TTS: Block Discrete Diffusion Language Model for Text-to-Speech Synthesis》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Text-to-Speech (TTS)** 系统面临一个根本性的效率-质量权衡：
- **自回归模型**（如 VALL-E）虽然语音可懂度高，但依赖大规模训练数据（60K–250K 小时）、逐 token 生成导致延迟高。
- **非自回归模型**（如 Flow Matching 或 Diffusion）虽能并行生成提升速度，但在文本-语音对齐上表现不佳，常出现跳词或重复。

### 🚀 提出的新方法与创新思路
作者提出 **DLLM-TTS**，首次将 **block discrete diffusion** 引入条件式语音合成任务中，核心思想如下：

- **将 TTS 建模为基于 X-Codec2 codec tokens 的条件块离散扩散过程**：
  - 输入序列被划分为多个 block（每块 32 个 token），在每个 block 内部进行 masked diffusion，同时按顺序处理 block。
  - 利用 **staircase attention** 机制实现：block 内双向建模局部声学一致性，block 间因果建模全局文本-语音对齐。

- **无需显式的 duration modeling 或音素级标注**，通过扩散训练目标隐式学习对齐关系。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **数据效率** | 仅需 20K 小时训练数据，相比自回归模型减少 3–12× 数据量。 |
| **推理速度** | 支持 block 内并行去噪，实现实时因子（RTF）达 **0.15**，适合实时应用。 |
| **语音质量与可懂度** | 在 Seed-TTS-eval 上取得具有竞争力的 WER/CER 和 MOS 分数。 |
| **零样本说话人克隆能力** | Speaker Similarity 达到 **0.750**，优于多数开源系统。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **Stage I**: 在 **Emilia dataset** 中采样的 **16K 小时真实语音数据**，用于建立基础的文本与说话人条件下的 codec token 生成能力。
- **Stage II**: 对 **4K 小时高质量合成语音** 进行微调，以优化韵律和对齐性能。
- 总计使用 **20K 小时** 多语言、多样化语音数据。

### ⚙️ 实验设置
- **模型架构**：
  - 基于 **Qwen2** 初始化的 Transformer 架构。
  - 参数规模：**0.6B**。
  - Block size $ B = 32 $（对应约 0.64 秒音频，$ f_r = 50\,\text{Hz} $）。
  - 最大序列长度：2048 tokens（约 40 秒语音）。
- **训练策略**：
  - 使用 AdamW 优化器，学习率 $1 \times 10^{-4}$，余弦调度 + 1% 预热。
  - 批大小：有效 batch size 128（8×H100 GPU，梯度累积 8 步）。
  - 时间步 $ t \sim U[0,1] $，每 token 独立掩码。
- **推理方式**：
  - 采用 **confidence-based sampling**：当预测置信度 > 0.6 时提前解码该位置。
  - 每 block 最多执行 $ T = 16 $ 或 $ 32 $ 轮 denoising。
  - 支持 KV caching 加速后续 block 推理。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **WER / CER** | 使用 Whisper-large-v3 计算，衡量语音识别准确率（反映可懂度）。 |
| **Speaker Similarity (SIM)** | 使用 WavLM-TDNN 提取嵌入，计算余弦相似度，评估音色保真度。 |
| **MOS (Mean Opinion Score)** | 由 25 名听众打分（5 分制），遵循 CodecMOS-Accent 协议，评价自然度与韵律。 |

### 🔁 基线方法对比
涵盖三类主流 TTS 模型：
- **自回归模型**：LLASA-3B (3B), Qwen2.5-Omni (7B)
- **非自回归模型**：F5-TTS (0.3B), MaskGCT, OpenAudio-s1-mini (0.5B)
- **混合模型**：DiTAR (0.6B), CosyVoice3 (0.5B), IndexTTS2 (1.5B)

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 2）
| 模型 | Params | WER ↓ | CER ↓ | SIM ↑ | MOS ↑ |
|------|--------|-------|-------|--------|-------|
| **DLLM-TTS (ours)** | **0.6B** | **2.25%** | **1.05%** | **0.750** | **4.25** |
| OpenAudio-s1-mini | 0.5B | 1.94% | 1.18% | 0.550 | **4.29** |
| DiTAR | 0.6B | 1.69% | 1.02% | 0.735 | 4.20 |
| LLASA-3B | 3B | 3.14% | 1.59% | 0.579 | 4.28 |

> 💡 结论：DLLM-TTS 在 **0.6B 参数量级下实现了接近最优的综合表现**，尤其在 **speaker similarity 上排名第一**，且显著优于其他同参数量模型。

### 🔄 与基线对比亮点
- **数据效率远超自回归模型**：仅用 20K 小时 vs 其他模型常用 60K–250K 小时。
- **推理速度快于大多数自回归与混合模型**：RTF = 0.15，支持低延迟流式输出（time-to-first-audio 短）。
- **无需 duration predictor 或音素对齐模块**，简化 pipeline 设计。

### 🔍 消融实验结果（见 Table 3）

#### （1）不同去噪步数 $ T $ 的影响（固定 $ B=32 $）
| $ T $ | WER | CER | SIM |
|-------|-----|-----|-----|
| 8     | 14.58% | 7.86% | 0.748 |
| 16    | 3.84%  | 1.25% | 0.765 |
| 32    | 2.25%  | 1.05% | 0.750 |
| 64    | 8.83%  | 6.76% | 0.746 |

> ✅ $ T=16 $ 已足够获得良好性能，是 **速度与质量的最佳平衡点**（RTF=0.15）。

#### （2）不同 block size $ B $ 的影响（设 $ T=B $）
| $ B $ | WER | CER | SIM |
|-------|-----|-----|-----|
| 8     | 4.19% | 2.06% | 0.725 |
| 16    | 3.04% | 1.43% | 0.746 |
| 32    | **2.25%** | **1.05%** | **0.750** |

> ✅ $ B=32 $ 表现最佳 —— 更大的 block 提升并行性，但过大会削弱跨 block 的因果依赖建模。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Block discrete diffusion 可有效应用于 TTS**：
   - 成功结合了自回归的序列建模能力和 diffusion 的并行生成优势。
   - 首次验证其在 **conditional speech generation over discrete codec tokens** 上的有效性。

2. **staircase attention 是关键设计**：
   - 同时捕捉 block 内部的局部声学细节与 block 间的长程文本-语音对齐，无需额外 duration modeling。

3. **masked diffusion 提供隐式数据增强**：
   - 不同时间步下的多种掩码模式使同一语音片段以多样形式呈现，提升数据利用率。

4. **高效且实用的零样本语音合成框架**：
   - 仅需 3–5 秒参考音频即可实现高质量 voice cloning，适用于实际部署场景。

### ⚠️ 方法的局限性
- 当前 block-sequential 设计仍存在一定程度的串行瓶颈（无法完全并行所有 block）。
- 极端复杂语境（如诗歌、快速连续发音）下可能出现轻微对齐漂移。
- 未公开模型是否支持多语种或极端口音泛化能力。

### 🔮 未来工作方向
- 探索更灵活的 adaptive block sizing 策略。
- 扩展至更大参数量（如 3B+）以进一步逼近人类水平语音质量。
- 结合 semantic token 和 acoustic token 的分层 diffusion 架构。
- 推动 block diffusion 在其他语音生成任务中的应用（如 voice conversion, singing voice synthesis）。

---

> 🧩 **总体评价**：  
> DLLM-TTS 展示了一条兼具 **高质量、高效率、强数据经济性** 的新型 TTS 技术路径，有望成为下一代语音合成系统的有力候选方案。

</details>

---

### 12. [S$^4$R: Selective Sampling, Subspaces, and Sparse Reconstruction for Compressed Long-Context KV Caching](https://arxiv.org/abs/2608.00528)

**Authors**: Jialong Han, You Wu, Kewei Tu  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.00528v1  

#### Abstract
The growth of context window lengths in Large Language Models (LLMs) significantly enhances their long-context capabilities but incurs prohibitive memory costs due to the Key-Value (KV) cache. Although low-rank compression of KV cache is a promising remedy, existing methods face a dilemma: offline a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# S⁴R: Selective Sampling, Subspaces, and Sparse Reconstruction for Compressed Long-Context KV Caching —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着 Large Language Models (LLMs) 的 context window 不断扩大（从 128k 到百万级 tokens），**Key-Value (KV) cache 的内存消耗已成为推理阶段的主要瓶颈**。传统的低秩压缩方法面临两难困境：
- **离线方法（fixed/post-training）**：依赖外部校准数据，泛化能力差；
- **在线方法（prompt-dependent）**：虽能适应当前上下文，但需对整个长 prompt 进行 SVD 分解，计算开销巨大。

S⁴R 旨在解决这一 **效率与适应性之间的权衡问题**，在保证高准确率的同时显著降低 KV cache 内存占用和推理延迟。

---

### 🚀 提出的新方法与核心思想
S⁴R 是一种 **prompt-aware 的低秩 KV cache 压缩方法**，结合了选择性采样、子空间构建与稀疏重建三大机制：

#### 主要创新点：
1. **Selective Sampling（选择性采样）**
   - 不对全 prompt 做 SVD，而是从输入中选取一个代表性子集（如早期均匀采样 + 最近连续块）来构建 key/value 子空间。
   - 显著减少预填充阶段（prefilling）的计算成本。

2. **Subspace Construction with Sink Preservation（带 Sink 保留的子空间构建）**
   - 将前 $ s=4 $ 个 **sink tokens** 以全精度保留（因其作为全局注意力锚点至关重要）；
   - 非 sink tokens 经过归一化后进行 truncated SVD，提取 top-$ r $ 右奇异向量作为低秩基（bases）；
   - 所有非 sink KV 状态被投影为低维系数（coefficients）存储，大幅压缩持久内存。

3. **Sparse Reconstruction during Decoding（解码时稀疏重建）**
   - 在每一步解码中，仅基于 query coefficient 在隐空间估算 token 相关性；
   - 动态选择局部窗口（local window）+ 全局重要位置（top-k global tokens）进行重建；
   - 仅对这些选中的 token 进行完整 KV 重建并参与 attention 计算。

> 💡 **核心洞察**：主导的 KV 子空间可通过少量代表性 token 近似；并非所有 KV 条目都需要同等处理——可实现“按需重建”。

---

### 🔍 相比现有方法的优势
| 方法类型 | 代表 | 缺陷 | S⁴R 如何改进 |
|--------|------|------|-------------|
| 固定压缩 | Palu, LoRC, EigenAttention | 依赖 calibration 数据，分布偏移下性能下降 | 构建 prompt-specific 子空间，更强适应性 |
| 在线压缩 | ShadowKV, xKV | 全 prompt SVD 开销大，延迟高 | 仅对采样子集做 SVD，prefill 成本低 |
| Token 蒸发 | StreamingLLM, SnapKV | 永久丢弃部分 token，可能丢失关键信息 | 非 sink tokens 保留在系数缓存中，可动态恢复 |
| 稀疏注意力 | Loki, Quest | 减少 attention 访问数，不减缓存本身 | 同时压缩持久 KV 表示 + 控制重建范围 |

✅ **综合优势**：
- 结合了固定方法的**高效性**与 prompt-dependent 方法的**上下文适应性**；
- 实现高达 **5× KV 压缩比**，同时保持接近 full-cache 的 accuracy；
- 显著优于 Palu、ShadowKV，在多数任务上媲美甚至超越更昂贵的 xKV。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **LongBench**：多语言、多任务长文本理解基准，涵盖：
  - 单文档问答（Single-Doc QA）
  - 多文档问答（Multi-Doc QA）
  - 少样本学习（Few-shot）
  - 合成任务（Synthetic）
  - 代码补全（LCC）
- **RULER**：专用于评估真实上下文长度能力的任务套件，包含：
  - 多 key/value 检索（MK/MV-NIAH）
  - 多跳查询（MQ-NIAH）
  - 数值追踪（VT）、事实窗口效应（FWE）等

---

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| **模型家族** | Llama 系列（1B, 8B）、Qwen 系列（4B, 14B） |
| **上下文长度** | 最长达 128k tokens（Qwen3-4B 使用 YaRN 扩展） |
| **压缩目标** | KV cache 压缩至原大小的 20%-30%（即约 3–5× 压缩） |
| **关键超参** |  
| - Rank $ r $ | 由目标压缩比反推得出 |
| - Sink 数量 $ s $ | 默认 4 |
| - 查询窗口 $ W $ | 32（消融研究测试 1–64） |
| - Pooling kernel $ K $ | 7（平均或最大池化） |
| - Reconstruction ratio $ p $ | 设为等于压缩保留率 $ \eta $ |

---

### 📈 评估指标
| 指标 | 类型 | 说明 |
|------|------|------|
| **准确性** | LongBench Avg., RULER Avg. | 主要任务得分均值 |
| **效率** |  
| - TTFT (Time to First Token) | 启动延迟 |
| - Latency₅₁₂ | 生成 512 tokens 的端到端延迟 |
| - TPOT (Time Per Output Token) ↓ | 单 token 推理时间 |
| - Output Throughput ↑ | 输出吞吐量（tokens/s） |
| - Total Throughput ↑ | 总吞吐量 |

---

### 🔁 对比的基线方法
| 方法 | 类型 | 特点 |
|------|------|------|
| **Full KV** | 上限 | 不压缩，原始性能 |
| **Palu** | 固定低秩 | 投影矩阵分解，高效但敏感于 calib data |
| **ShadowKV** | prompt-dependent | prompt SVD + sparse reconstruction，但 offload values |
| **xKV** | prompt-dependent | cross-layer SVD，效果好但 prefill 成本极高 |
| **SnapKV / SLM** | eviction-based | 保留 sink + 局部窗口或聚类 token |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ LongBench 结果（Qwen3-4B）
| 方法 | 压缩比 | Avg Score | Throughput (tok/s) |
|------|--------|-----------|---------------------|
| Full KV | 1× | 49.62 | 9.33 |
| S⁴R | ~5× | **47.79** | **8.32** |
| xKV-4 | ~5× | 45.22 | 1.96 |
| ShadowKV | ~5× | 47.38 | 8.99 |
| Palu | ~5× | 3.35 | 8.97 |

> ✅ **S⁴R 在几乎不损失 accuracy 的前提下，吞吐远高于 xKV，略低于 ShadowKV/Palu，但准确率更高**

---

#### ✅ RULER 结果（Qwen2.5-7B, 128k context, 2.5× 压缩）
| 方法 | Avg Score | MK/MQ/V Retrieval |
|------|----------|------------------|
| Full KV | 93.97 | Perfect on most |
| **S⁴R** | **93.61** | ✅ 完美匹配 multi-key/value retrieval |
| xKV | 93.47 | 略逊于 S⁴R |
| ShadowKV | 89.61 | 明显落后 |
| Palu | 82.22 | 性能崩溃 |

> ✅ S⁴R 几乎追平 full KV，在复杂检索任务中表现优异。

---

### ⚖️ 效率对比（Qwen3-4B, 120k context）
| 方法 | TTFT (ms) | Latency₅₁₂ (s) | Throughput (tok/s) |
|------|------------|----------------|---------------------|
| Full KV | 24,737 | 54.89 | 9.33 |
| Palu | 24,579 | 57.06 | 8.97 |
| ShadowKV | 36,143 | 56.92 | 8.99 |
| **xKV** | **79,897** | **260.76** | **1.96** |
| **S⁴R** | **27,650** | **61.54** | **8.32** |

> 🔥 **S⁴R 将 xKV 的延迟从 260s 降至 61s，提速超过 4×，TTFT 降低 65%**

---

### 🔍 消融实验结果（Ablation Studies）

#### (1) Sink Token 保留的重要性
| 模型 | w/ Sink (full precision) | w/o Sink (all compressed) | 差距 |
|------|----------------------------|------------------------------|------|
| Llama-3.2-1B | 28.09 | 24.60 | +3.49 |
| Qwen3-4B | 47.79 | 45.58 | +2.21 |

> ✅ 证明 sink tokens 必须保留全精度，否则严重影响 long-range dependency 建模。

---

#### (2) Token-wise 归一化的作用
| 模型 | w/ Normalize | w/o Normalize |
|------|---------------|----------------|
| Llama-3.1-8B | 53.03 | 53.32 | 影响小 |
| Qwen2.5-14B | **53.19** | **46.27** | ↓7 pts! |

> ✅ 归一化提升跨模型鲁棒性，防止高范数 token 主导 SVD 基。

---

#### (3) 查询窗口大小 $ W $ 的影响
| $ W $ | Llama-3.2-1B | Qwen3-4B | 最佳值 |
|--------|--------------|-----------|--------|
| 1 | 25.90 | 47.31 | ❌ 太小不稳定 |
| 16 | 28.68 | 47.85 | ✅ 最优 |
| 32 | 28.09 | 47.79 | ✔️ 接近最优（默认） |
| 64 | 27.38 | 47.55 | ❌ 过大反而下降 |

> ✅ 中等窗口（16–32）最佳：平衡局部覆盖与全局相关性估计。

---

#### (4) 重建比例 $ p $ 的影响
| $ p $ | Llama-3.2-1B | Qwen3-4B | 结论 |
|--------|--------------|-----------|--------|
| 0.2 (default) | 28.09 | 47.79 | ✅ 更安全 |
| 0.15 | 27.90 | 47.46 | ↓ 可接受，但损失明显 |

> ✅ 默认 $ p=\eta $ 提供更好的 accuracy-efficiency trade-off。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **S⁴R 实现了 prompt-aware 与高效性的统一**：
   - 通过 selective sampling 构建 prompt-specific 子空间，避免 full-prompt SVD；
   - 通过 sparse reconstruction 控制运行时开销，实现“按需解压”；
   - 达成 **5× KV 压缩 + 接近 full-cache accuracy**。

2. **设计组件有效性得到验证**：
   - 保留 sink tokens 至关重要；
   - token 归一化增强子空间稳定性；
   - hybrid sampling（recent + uniform）优于单一策略；
   - 中等 query window 和合理 reconstruction ratio 最优。

3. **效率显著优于 prompt-dependent 方法**：
   - 相比 xKV，**TTFT 降低 65%，延迟降低 76%**；
   - 吞吐量是 xKV 的 **4.2× 以上**；
   - 在 LongBench 和 RULER 上全面胜出。

---

### ⚠️ 方法的局限性
1. **缺乏系统级优化**：
   - 当前实现未使用 kernel fusion 或 PagedAttention 优化，实际部署仍有提升空间。

2. **依赖启发式超参**：
   - 采样比例、rank 分配、pooling 方式等仍需手动设定，尚未完全自动化。

3. **对某些任务敏感**：
   - 如 RULER 中的 FWE（Fact Window Effect）任务仍有差距，表明当证据分散时，latent scoring 可能遗漏关键位置。

4. **泛化性待进一步验证**：
   - 实验集中在 Llama 和 Qwen 系列；
   - 在极长输出、其他架构（如 Mamba）、多模态场景下的表现未知。

---

### 🔮 未来工作方向
1. **自适应 rank allocation**：
   - 根据 layer 或 sequence 动态调整 $ r_K, r_V $，而非固定分配。

2. **联合量化 + 系数压缩**：
   - 将 S⁴R 与 KIVI、KVQuant 等量化技术结合，进一步压缩系数存储。

3. **learned sampling policy**：
   - 使用轻量网络预测哪些 prompt tokens 更适合用于 subspace 构建。

4. **支持 streaming + generation 并重场景**：
   - 与 StreamingLLM、DuoAttention 等流式推理框架集成。

5. **探索 reconstruction-free attention**：
   - 是否可在 coefficient space 直接完成 attention？减少重建必要性。

---

> 📌 **总结一句话**：  
> **S⁴R 提出了一条实用且高效的路径——通过 selective sampling 构建 prompt-aware 子空间，并结合 sparse reconstruction 实现“压缩存储 + 按需重建”，在保持高 accuracy 的同时极大缓解 long-context LLM 推理的内存与延迟压力。**

</details>

---

### 13. [AttnLink: Turning Attention into Schema Links for Text-to-SQL](https://arxiv.org/abs/2608.00693)

**Authors**: Jinwang Song, Tao Liu, Haowen Zheng, Xiangheng Li, Yifan Li, Hongying Zan  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.00693v1  

#### Abstract
Schema linking is a critical component of Text-to-SQL systems, but existing approaches often trade off contextual modeling capacity, score-based controllability, and inference efficiency. We introduce AttnLink, an attention-based framework that converts LLMs' internal attention into continuous relev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AttnLink: Turning Attention into Schema Links for Text-to-SQL**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在 **Text-to-SQL** 系统中，**schema linking** 是一个关键环节，其目标是识别自然语言问题所涉及的数据库表（tables）和列（columns）。现有方法通常在以下三个方面存在权衡：
- **语义建模能力**（Semantic Capacity）
- **可控性**（Controllability via continuous scores）
- **推理效率**（Inference Efficiency）

传统方法如生成式（generative）或检索式（retrieval-based）方法往往牺牲其中一个方面。例如：
- 生成式方法输出离散集合，缺乏对 precision-recall 的灵活控制；
- 检索式方法依赖额外模型（如 embedding 或 cross-encoder），增加延迟且语义建模弱于 LLM。

### **提出的新方法**
本文提出 **AttnLink**，一种基于 **LLM 内部注意力机制** 的 schema linking 框架，将 LLM 在生成前的注意力分布转化为连续的相关性分数（relevance scores），用于排序 schema 项。

#### **核心思想**
- 利用 **generation-start position**（称为 generation anchor）对候选 schema spans 的注意力权重作为相关性信号。
- 所有候选可在一次 **prefill pass** 中完成打分，无需 autoregressive decoding。
- 支持 **post-hoc precision-recall 控制**，通过 temperature scaling 和 top-p selection 调整输出大小。

#### **两种变体**
| 变体 | 特点 |
|------|------|
| **AttnLink-U** | 零训练（zero-training）探针，直接提取预训练 LLM 注意力，无参数更新 |
| **AttnLink-S** | 引入监督学习，使用 set-mass objective + adaptive floor regularizer 对齐注意力分布与黄金标签 |

### **相比现有方法的优势**
| 维度 | AttnLink 优势 |
|------|----------------|
| **语义建模** | 利用完整 LLM 上下文理解能力，优于独立 embedding 模型 |
| **可控性** | 输出连续分数，支持动态调整 precision/recall/schema 数量 |
| **效率** | 单次 prefill 推理，毫秒级延迟，兼容 vLLM 等高效 serving 框架 |
| **通用性** | 适用于多种 LLM 架构（dense、MoE、hybrid attention） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 描述 |
|--------|------|
| **Spider** | 经典跨域 Text-to-SQL 基准，7K 训练样本，关注复杂 SQL 和 schema 泛化 |
| **BIRD** | 更大规模真实数据库场景，强调 value grounding 和领域知识 |
| **Spider2-SQLite** | 来自 Spider 2.0-Lite 的 SQLite 子集，聚焦企业级复杂 schema，**无训练集**，用于零样本迁移测试 |

### **实验设置与评估指标**

#### **评估任务**
- **Column-level schema linking**（主任务）
- **Table-level schema linking**（补充任务）

#### **评估指标**
| 指标 | 含义 |
|------|------|
| **Precision (P)** | 选出的 schema 项中有多少是相关的 |
| **Recall (R)** | 黄金 schema 项被保留的比例 |
| **Strict Recall Rate (SRR)** | 完全覆盖黄金 schema 的样本比例（不可恢复遗漏） |
| **mean Average Precision (mAP)** | 排序质量综合指标，衡量排名靠前是否为相关项 |
| **Execution Accuracy (EX)** | 下游 SQL 生成器最终执行正确的比例 |

#### **推理控制机制**
- **Temperature Scaling (T)**：调节分布平滑度，T > 1 → 更多低分项被保留
- **Top-p Selection**：选择累计概率达阈值 p 的最短前缀，实现动态剪枝

### **基线方法对比**
分为五类：
1. **Embedding-based**：BGE-M3, Qwen3-Embedding-8B
2. **Cross-Encoder Rerankers**：Qwen3-Reranker-4B/8B
3. **Generative SFT Linkers**：DTS-SQL
4. **Prompting/Agent-based**：LinkAlign, RSL-SQL, AutoLink
5. **LLM-based Tunable Linkers**：ExSL, JOLT-SQL

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Column-level）**

#### **AttnLink-S 在各数据集上的 mAP 表现**
| 数据集 | mAP |
|-------|-----|
| **Spider Dev** | **99.22%** |
| **BIRD Dev** | **95.95%** |
| **Spider2-SQLite** | **83.29%** |

> ✅ 均为当前 **state-of-the-art** 排名 linker 结果。

#### **下游 SQL 执行准确率（EX）**
在 **7 out of 9 generator-dataset 设置中达到最佳或并列最佳**：
- 在 **BIRD** 和 **Spider2-SQLite** 上提升显著，因这些数据集 schema 更大更噪，AttnLink 的高召回与可控过滤尤为有效。
- 示例：Qwen3.5-9B + AttnLink-S 在 BIRD 上 EX 达 **69.1%**，优于第二名约 1–2 个百分点。

#### **推理延迟（Latency）**
| 方法 | 平均延迟（BIRD Dev） |
|------|------------------|
| **AttnLink** | **~11–32ms** |
| DTS-SQL / RSL-SQL | ~3–15s |
| AutoLink / LinkAlign | ~4–15s |

> ⚡️ AttnLink 实现 **毫秒级 schema linking**，比生成式方法快 **100x 以上**。

### **与基线方法对比结果**
| 方法类型 | 典型表现 | AttnLink 优势 |
|---------|----------|---------------|
| Embedding | mAP ~80% (Spider), ~54% (BIRD) | AttnLink-S 提升超 10–40 个百分点 |
| Reranker | mAP ~89–90% (Spider), ~72% (BIRD) | AttnLink-S 显著更高，且无需多次 forward |
| ExSL/JOLT-SQL | mAP ~98–99%，但需修改架构 | AttnLink 不改模型结构，兼容性更强 |

### **消融实验结果**

#### **(1) Floor Regularizer 消融（Table 2）**
| Floor Ratio $ p $ | BIRD mAP (%) |
|--------------------|--------------|
| 0 (仅 set-mass) | 71.44 |
| 0.25 | **82.06** |
| 0.5 | 83.29 |
| 1.0 | 78.75 |

> ✅ **适度 floor ($p=0.25$)** 最佳，防止“easy positives”主导而忽略次要但必要的项；$p=1$ 过强均匀约束反而损害性能。

#### **(2) Copy-Oriented Instruction 必要性（Table 7）**
替换为语义推理指令后：
- BIRD mAP 从 **79.84% → 25.61%**
- Spider 从 **92.27% → 34.72%**

> ❗️证明 **copy-oriented prompt 设计至关重要**，它引导 attention 聚焦于 candidate spans。

#### **(3) Pooling 策略比较（Table 6）**
| Pooling 方式 | BIRD mAP (Qwen3.5-9B) |
|-------------|------------------------|
| First Token | 78.09 |
| Sum Pooling | 79.49 |
| **Mean Pooling** | **79.84** |

> ✅ **mean pooling** 最优，平衡了 token 长度差异的影响。

#### **(4) Layer-Head 稳定性分析**
- AttnLink-U 在不同候选顺序扰动下，最优 head（L23/H9）始终稳定。
- Spearman 相关性 > 0.96，说明 head 选择具有鲁棒性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **LLM 注意力本身是强大的 schema grounding 信号**  
   - 即使不微调（AttnLink-U），也能取得接近 SOTA 的 recall 和 mAP。
   
2. ✅ **Attention 可被有效监督优化**  
   - AttnLink-S 通过 set-mass loss 和 adaptive floor 显著提升 multi-positive 覆盖能力。

3. ✅ **无需 autoregressive decoding 即可实现高质量 linking**  
   - 单次 prefill pass + attention extraction 实现毫秒级响应，适合生产部署。

4. ✅ **continuous relevance scores 支持灵活下游适配**  
   - 可根据不同 SQL generator 的噪声容忍度调节 T 和 p，最大化 EX。

5. ✅ **跨架构、跨数据集泛化能力强**  
   - 在 Qwen、Llama、MoE 等多种 LLM 上均有效；
   - 在无训练集的 Spider2-SQLite 上仍表现强劲，体现迁移能力。

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **依赖特定 prompt 设计** | copy-oriented instruction 对性能影响巨大，设计敏感 |
| **需人工选定 layer-head（AttnLink-U）** | 尽管稳定，但仍需小样本 calibrate |
| **attention 分布可能受位置偏置影响** | 尽管 mean pooling 缓解，长 schema 下仍可能存在偏差 |
| **无法处理未出现在 context 中的 schema** | 仍是 context-based 方法，受限于输入长度 |

### **未来工作方向**
1. **自动化 layer-head selection**  
   - 开发免校准的通用策略，适应任意 LLM。
2. **结合 long-context 优化**  
   - 在超长 schema 场景下引入 hierarchical attention probing。
3. **multi-turn schema expansion**  
   - 类似 AutoLink，迭代式扩展链接集，结合 AttnLink 的初始高召回。
4. **端到端 joint tuning**  
   - 将 AttnLink-S 与 SQL generator 联合优化，进一步提升 EX。

---

> 🔚 **总结一句话**：  
> **AttnLink 成功将 LLM 的内部 attention 转化为高效、可控、高质量的 schema linking 工具，在性能、速度和灵活性上全面超越现有方法，为 Text-to-SQL 系统提供了新的基础设施级解决方案。**

</details>

---

### 14. [Attend to Your Own Thoughts: Breaking the Barrier for Post-Training Quantization of Reasoning LLMs through the Lens of 1.58-Bit Quantization](https://arxiv.org/abs/2608.01078)

**Authors**: Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, Anbang Yao  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.01078v1  

#### Abstract
We propose ScaleQ-1.58, a scalable ternary post-training quantization (PTQ) framework for reasoning LLMs. Its core insight stems from an empirical finding: although modern LLMs are typically trained to exhibit chain-of-thought reasoning capabilities, in the PTQ regime, even the latest CAT-Q method b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Attend to Your Own Thoughts: Breaking the Barrier for Post-Training Quantization of Reasoning LLMs through the Lens of 1.58-Bit Quantization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Post-Training Quantization (PTQ)** 方法在对具备复杂推理能力的 **Reasoning LLMs**（如数学、编程任务）进行低比特量化时，通常会导致严重的 **performance degradation**，尤其是在极端低比特（如1.58-bit）场景下。主流方案依赖 **Quantization-Aware Training (QAT)**，虽然有效但成本极高，难以扩展到大规模模型和多样化架构。

此外，传统 PTQ 方法使用的校准数据（calibration data）多来自通用网页语料（如 C4、WikiText2），忽略了模型在推理任务中依赖的 **Chain-of-Thought (CoT)** 过程，导致量化后模型无法保留其推理链能力。

### 提出了什么新方法或新思路
本文提出了一种可扩展的三值化 PTQ 框架 **ScaleQ-1.58**，其核心是引入一种全新的校准方法：  
👉 **Attend to Your Own Thoughts (AYOT)**。

AYOT 的核心思想是：
- 在量化过程中，使用预训练高精度目标 LLM 自身生成的 **CoT 推理轨迹** 和 **最终答案** 作为校准上下文输入。
- 即：将 `[问题, 推理过程, 最终答案]` 三元组用于校准，而非仅用原始问题或通用文本。

该方法结合了 **CAT-Q**（首个可微分三值化方法），形成完整的 PTQ 流程。

### 相比现有方法的优势
| 维度 | ScaleQ-1.58 (AYOT + CAT-Q) | 现有方法（如 BitNet b1.58 2B4T） |
|------|-------------------------------|----------------------------------|
| **训练成本** | 仅需 **4M 校准 token**，无需重新训练 | 需从头训练，使用 **4T token** |
| **token 效率** | 减少 **1,000,000× 校准 token** | 极高资源消耗 |
| **任务覆盖** | 支持数学、编程、科学逻辑等多种复杂推理任务 | 多数仅测试基础语言理解 |
| **模型兼容性** | 支持 Dense 和 MoE 架构，参数规模达 **235B** | 多限于小规模（<10B）Dense 模型 |
| **部署效率** | 显著降低内存占用，提升推理吞吐量（见 Table E） | 无直接比较，但训练成本过高 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### 主要评估任务（复杂推理）
| 类别 | 数据集 | 描述 |
|------|--------|------|
| 数学 | **Math-500**, **GSM8K**, **Omni-MATH** | 包含竞赛级数学题，需多步推理 |
| 编程 | **HumanEval+**, **MBPP+** | 编程功能正确性测试 |
| 科学逻辑 | **ProofWriter** | 多步演绎推理，支持不同 depth 的逻辑链 |

#### 校准数据构建
- **Domain-specific**: 从 **MetaMathQA**（数学）和 **OpenCodeInstruct**（代码）采样问题。
- **Self-generated CoT**: 使用待量化的 **target LLM 自身** 生成对应的推理过程和答案。
- 总计 **4M tokens**，默认序列长度为 2048。

### 实验设置和评估指标
- **量化格式**: 默认采用 **W1.58A16**（权重 1.58-bit，激活 16-bit）
- **模型范围**: Qwen3 系列（1.7B ~ 235B），包括 Dense 和 MoE 架构；另测试 DeepSeek-R1-Distill-Llama-70B
- **评估指标**:
  - **Accuracy (%)**：用于数学、编程、逻辑推理等任务
  - **Perplexity ↓**：用于语言建模（WikiText2, C4）
  - **Average Accuracy ↑**：用于 Commonsense Reasoning（PIQA, ARC-e/c, HellaSwag, Winogrande）

### 基线方法对比
- **主基线**: **BitNet b1.58 2B4T** —— 当前最强的 1.58-bit 推理 LLM，从头训练于 4T tokens
- **其他基线**:
  - CAT-Q with generic-text (C4/WikiText2)
  - CAT-Q with domain-specific data
  - CAT-Q with stronger-LLM-generated CoT（如 DeepSeek-R1-671B 生成）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1 & Figure 1）

| 模型 | Math/GSM8K/HumanEval+/MBPP+ 平均准确率 | 对比 BitNet b1.58 2B4T |
|------|-----------------------------|--------------------------|
| **BitNet b1.58 2B4T** | ~49.43% | 基线 |
| **Qwen3-1.7B (ScaleQ-1.58)** | >90.52% of baseline | 仅用 4M tokens，性能接近 |
| **Qwen3-4B (ScaleQ-1.58)** | **+8.97% 绝对增益** | 超越基线，且 token 成本低百万倍 |

> ✅ **Qwen3-4B 在仅使用 4M 校准 token 的情况下，全面超越训练于 4T token 的 BitNet b1.58 2B4T**

### 与基线方法的对比结果
| 方法 | 平均准确率（五项任务） | 是否崩溃 |
|------|------------------|---------|
| CAT-Q + C4 | ~5.77% | 是（near-zero performance） |
| CAT-Q + domain-specific | 17.43% | 否，但很低 |
| CAT-Q + stronger-LLM CoT | 20.06% | 仍不足 |
| **AYOT (self-generated CoT)** | **45.60% → 最终达 58.40%+** | ✅ 成功恢复推理能力 |

> 📌 AYOT 相比最强外部 CoT 生成器（DeepSeek-R1-671B）带来 **+25.54%** 的绝对提升

### 消融实验结果

#### （1）校准策略对比（Table 4）
| Calibration Scheme | Average Accuracy |
|--------------------|------------------|
| CoT-agnostic: generic-text (C4) | 5.77% |
| CoT-agnostic: domain-specific | 17.43% |
| CoT-aware: stronger-LLM-generated | 20.06% |
| **AYOT (self-generated)** | **45.60%** |

✅ 验证两个设计原则：
1. **领域相关性**（domain-specific）至关重要
2. **自生成 CoT**（self-generated）远优于更强外部模型

#### （2）校准 token 数量影响（Table 2 & Figure 4）
- 从 **256K → 16M tokens**，性能持续上升，未见饱和
- 即使在 **4M tokens** 下已超越 BitNet b1.58 2B4T
- 表明 **data diversity** 是关键驱动因素

#### （3）泛化到其他 bit-width（Table 5）
| Bit-width | 方法 | Gain over baseline |
|----------|------|---------------------|
| W1.58A16 | AYOT + CAT-Q | +~40% |
| W2A16 | AYOT + SliderQuant | +~50% |
| W4A16 | AYOT + SliderQuant | +~2–3% |

✅ AYOT 不仅适用于 1.58-bit，在 **W2A16/W4A16** 也显著有效，具有广泛适用性

#### （4）任务特定数据增强（Table 6 & 8）
- 引入少量 **task-specific data**（如 GSM8K/MBPP 训练集）可进一步提升对应任务表现
- 增加序列长度至 **4096** 可更好捕捉长推理链，尤其利于 Math-500 和 HumanEval+

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **传统 PTQ 失败的根本原因在于“忽略推理过程”**  
   即使是最先进的 CAT-Q 方法，若使用通用校准数据，也会在复杂推理任务上崩溃。

2. 💡 **“自己的思考”是最好的校准信号**  
   AYOT 表明：让模型在量化时“回顾自己如何解题”，能最有效地保留其 CoT 推理能力。

3. ⬆️ **ScaleQ-1.58 具备优异的 scaling properties**：
   - ✅ 随模型规模增大，性能持续提升（up to 235B）
   - ✅ 兼容 Dense 与 MoE 架构
   - ✅ 随校准数据量增加，性能单调上升
   - ✅ 泛化至多种任务类型（数学、编程、科学逻辑、常识推理）

4. 🚀 **极高的性价比与实用性**
   - 用 **百万分之一的 token 成本**，实现 **超越甚至大幅领先** 的性能
   - 可在单台 8xA100 服务器上完成（4–240 小时）

### 方法的局限性
- **依赖高质量 CoT 生成能力**：若目标模型本身 CoT 能力弱，则 AYOT 效果受限
- **硬件支持尚不完善**：当前 1.58-bit GPU/CPU kernels 主要适配 BitNet 家族，通用支持有限（见 Suppl. A）
- **仍有性能差距**：尽管大幅进步，但仍未能完全达到 FP16 基线水平，尤其在最难任务上

### 未来工作方向
- 探索更高效的 **1.58-bit kernel 实现**，推动端侧部署
- 结合 **activation quantization** 实现全模型超低位部署
- 扩展至更多模态与任务（如多跳问答、规划等）
- 研究 **zero-shot calibration** 或 **合成数据生成** 以减少对真实校准集的依赖

---

> 🔗 **代码开源地址**：https://github.com/IntelChina-AI/BitTern  
> 📄 **论文亮点总结**：首次证明 **Post-Training 1.58-bit Quantization** 可用于复杂推理 LLM，并通过 **Attend to Your Own Thoughts** 实现低成本、高性能、强泛化的突破。

</details>

---

### 15. [Adaptive Quantum Physics-Informed Neural Networks for Differential Equations with Applications to Fluid Dynamics](https://arxiv.org/abs/2608.00850)

**Authors**: Fabio Pereira dos Santos, Renato Portugal, J\'ulio de Castro Vargas Fernandes, Lucas Timotheo Sanches  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.00850v1  

#### Abstract
Physics-informed neural networks (PINNs) have emerged as a versatile approach for solving nonlinear partial differential equations (PDEs), yet achieving high accuracy efficiently using these techniques remains challenging for high-dimensional or multiscale systems. Here, we present a hybrid quantum-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Adaptive Quantum Physics-Informed Neural Networks for Differential Equations with Applications to Fluid Dynamics

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Physics-Informed Neural Networks (PINNs)** 在求解高维、多尺度或强非线性偏微分方程（PDEs）时面临两大挑战：
- **优化瓶颈**：训练过程中损失函数各成分（如PDE残差、边界条件）之间梯度不平衡，导致模型陷入局部最优或收敛缓慢。
- **采样效率低**：固定且均匀的 **collocation points** 难以捕捉解中的陡峭梯度、激波或边界层等关键区域。

尽管已有研究尝试将量子计算引入PINNs形成 **Quantum PINNs (QPINNs)**，但当前QPINNs仍受限于经典PINNs的优化缺陷，并未充分发挥量子表达能力。

### 提出了什么新方法或新思路
本文提出了一种新型混合量子-经典框架：**Adaptive Quantum Physics-Informed Neural Network (AQPINN)**，其核心创新包括：

1. **自适应采样策略（Adaptive Collocation Point Sampling）**
   - 动态调整collocation points分布，通过三种机制：
     - **Gradient-driven relocation**：将高残差点沿残差梯度方向移动，集中于误差大的区域。
     - **Local refinement**：在持续高残差区附近“分裂”生成新点，提升局部分辨率。
     - **Importance-based pruning**：移除历史残差小且“年龄”大的点，避免冗余，控制总点数。

2. **注意力机制驱动的动态损失加权（Loss-aware Attention Mechanism）**
   - 引入可学习的 **soft attention weights** 对不同损失项（PDE残差 $L_p$、边界条件 $L_b$、初始条件 $L_i$）进行动态平衡。
   - 使用 **softmax参数化** 和普通梯度下降更新权重，而非复杂的minimax优化。

3. **混合量子-经典神经网络架构**
   - 采用 **hybrid quantum-classical NN**：输入先经经典预处理 → 映射到变分量子电路（VQC）→ 测量期望值作为特征 → 经典后处理输出解。
   - 量子层作为高维希尔伯特空间中的非线性特征映射，增强模型表达力。

### 相比现有方法的优势
| 方面 | 传统PINNs/QPINNs | AQPINN |
|------|------------------|--------|
| **采样方式** | 固定、随机或均匀分布 | 自适应、基于残差演化动态重分布 |
| **损失平衡** | 手动设定静态权重 | 可学习、动态调整的attention机制 |
| **优化稳定性** | 易受梯度刚度影响 | 缓解多目标冲突，提升收敛性 |
| **物理一致性** | 多场耦合问题易失衡 | 更好满足PDE、BC、IC联合约束 |
| **资源利用效率** | 需大量点覆盖全域 | 聚焦高梯度区，用更少点实现更高精度 |

---

## 2. 核心实验方法和设置

### 使用的数据集 / 测试问题
论文在六类典型微分方程上进行了系统验证，涵盖 **ODEs** 与 **PDEs**，特别是流体力学中的基准问题：

| 类型 | 具体问题 | 描述 |
|------|---------|------|
| **ODE** | 1D Helmholtz Equation | 边界值问题，振荡解 $u(x)=\cos(x)$ |
| **ODE** | Spring-Mass System | 阻尼强迫谐振子，含初值条件 |
| **PDE** | 2D Poisson Equation | 模拟Navier-Stokes压力场，Dirichlet边界 |
| **PDE** | 1D Burgers' Equation | 含粘性激波，测试对非线性和间断的处理能力 |
| **PDE** | Hagen-Poiseuille Flow | 圆管内稳态层流，轴对称Navier-Stokes简化 |
| **PDE** | Taylor-Couette Flow | 同心旋转圆柱间的粘性流动，极坐标下解析解已知 |

所有问题均提供解析解用于误差评估。

### 实验设置和评估指标

#### 架构配置
- **量子部分**：使用 **3–4 qubits**，深度为 **1–6 layers** 的变分量子电路（VQC），测量Pauli-Z期望值。
- **经典部分**：浅层MLP（1–2层），激活函数为SiLU。
- **训练流程**：两阶段优化 —— 初始使用 **Adam** 快速探索，随后切换至 **L-BFGS** 精细收敛。
- **工具栈**：基于 **PennyLane** 实现量子模拟与梯度计算（parameter-shift rule）。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **RMSE** | 均方根误差 |
| **MAE** | 平均绝对误差 |
| **Max Error** | 最大绝对误差 |
| **Relative L2/L∞** | 相对误差范数 |
| **R² (Coefficient of Determination)** | 决定系数，衡量拟合优度 |
| **Best Loss** | 训练过程中的最低总损失值 |

### 基线方法对比
- **Baseline QPINN**：相同网络结构，但使用：
  - 固定数量的collocation points（无自适应）
  - 固定损失权重（如 $w_p = w_b = 0.5$）
- 不直接比较AQPINN与经典APINN，而是聚焦“**自适应机制是否能显著提升QPINN性能**”。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ **Burgers' Equation 结果（Table 1）**
| 指标 | QPINN | AQPINN | 提升幅度 |
|------|-------|--------|----------|
| RMSE | $1.917 \times 10^{-1}$ | $6.925 \times 10^{-2}$ | ↓ **64%** |
| MAE | $3.441 \times 10^{-2}$ | $1.351 \times 10^{-2}$ | ↓ 61% |
| Max Error | $1.950$ | $0.648$ | ↓ 67% |
| R² | 0.896 | 0.986 | ↑ 显著改善 |
| Best Loss | $3.635 \times 10^{-4}$ | $2.360 \times 10^{-5}$ | ↓ 93% |

> AQPINN成功捕获激波位置（$x=0, t=0.5$），而QPINN出现严重抹平现象。

#### ✅ **Taylor-Couette Flow 结果（Table 2）**
| 指标 | QPINN | AQPINN | 提升幅度 |
|------|-------|--------|----------|
| RMSE ($u_\theta$) | $3.299 \times 10^{-4}$ | $3.122 \times 10^{-5}$ | ↓ **90.5%** |
| MAE | $3.067 \times 10^{-4}$ | $2.789 \times 10^{-5}$ | ↓ 91% |
| Max Error | $7.268 \times 10^{-4}$ | $9.501 \times 10^{-5}$ | ↓ 87% |
| R² | 0.999 | 1.0 | 接近完美拟合 |
| Best Loss | $1.387 \times 10^{-5}$ | $1.093 \times 10^{-6}$ | ↓ 92% |

> AQPINN显著抑制径向速度 $u_r$ 残差（从 $4.48 \times 10^{-8}$ 降至 $3.88 \times 10^{-8}$），更好满足质量守恒。

#### ✅ **Hagen-Poiseuille Flow**
- **速度场 RMSE**：从 $4.218 \times 10^{-3}$（QPINN）降至 $1.812 \times 10^{-3}$（AQPINN）
- **压力场相对误差**：从 50.11% 降至 1.24%，R² 从 0.75 升至 1.0
- 自适应点集中在入口边界层（$x \to 0$）和核心区（$x \sim 2.0$）

#### ✅ **总体性能增益**
- 在多种问题上，AQPINN相比QPINN实现了 **至少60%以上的RMSE降低**。
- 特别是在多场耦合、几何复杂或存在激波的问题中，优势更为明显。

---

### 消融实验结果（Ablation Study）

论文在附录中对两种自适应机制分别进行消融分析：

#### 🔹 **Burgers' Equation 消融实验**
| 模型 | RMSE | R² |
|------|------|-----|
| QPINN (baseline) | $1.917 \times 10^{-1}$ | 0.896 |
| + Adaptive Collocation Only | $1.203 \times 10^{-1}$ | 0.959 |
| + Self-Attention Only | $9.306 \times 10^{-2}$ | 0.975 |
| **Full AQPINN (both)** | $6.925 \times 10^{-2}$ | **0.986** |

> **结论**：两种机制均有效，但**协同作用带来最大收益**，表明空间自适应与特征自适应互补。

#### 🔹 **Taylor-Couette Flow 消融实验**
| 模型 | RMSE | R² |
|------|------|-----|
| QPINN | $3.299 \times 10^{-4}$ | 0.999 |
| + Adaptive Collocation | $1.441 \times 10^{-4}$ | 1.0 |
| + Self-Attention | $2.450 \times 10^{-4}$ | 0.999 |
| **Full AQPINN** | $3.122 \times 10^{-5}$ | **1.0** |

> 在此光滑流动中，**空间自适应更重要**，因剪切区需更高分辨率。

---

## 4. 关键结论和发现

### 主要发现
1. **优化瓶颈是QPINNs的关键限制因素**
   - 当前QPINNs的性能瓶颈不仅在于量子电路的表达能力（expressivity），更在于**经典训练过程中的优化困难**。
   - 单纯增加量子层数或宽度无法解决梯度失衡问题。

2. **自适应机制显著提升QPINN性能**
   - **动态损失加权**（attention）缓解了多目标优化冲突，防止某一项主导训练。
   - **残差驱动的自适应采样** 将计算资源精准投向高梯度/高误差区域，极大提升了采样效率。

3. **自适应策略优于单纯扩大模型规模**
   - AQPINN在不增加网络参数或collocation点总数的前提下，通过智能重分布实现了更高精度，证明“** smarter sampling > more points **”。

4. **量子-经典协同设计更具现实意义**
   - 在NISQ时代，完全依赖量子加速尚不现实；而结合经典自适应策略的混合架构是通往实用化 **quantum-enhanced scientific ML** 的可行路径。

### 方法的局限性
- **当前实验基于模拟器**：所有量子电路运行在经典模拟器上，尚未在真实量子硬件上部署。
- **小规模量子电路**：仅使用3–4 qubits，远未体现潜在的指数级表示优势。
- **未与经典APINN直接比较**：无法判断量子结构本身是否优于经典结构（当两者都具备自适应能力时）。
- **计算开销增加**：自适应机制引入额外计算（如残差梯度、点管理），可能影响实时性。

### 未来工作方向
1. **扩展至三维复杂流场**：将AQPINN应用于全三维Navier-Stokes方程仿真。
2. **探索更大规模量子电路**：研究电路深度/宽度与自适应机制的交互效应。
3. **在真实量子设备上验证**：考虑噪声、退相干等实际因素下的性能表现。
4. **与其他量子算法融合**：如结合VQLS、HHL等求解线性系统子任务。
5. **理论分析自适应机制的收敛性**：建立数学框架解释为何自适应能逃离局部极小。

---

> 📌 **总结一句话**：  
> 本论文揭示了 **QPINNs 的主要瓶颈在于优化而非表达能力**，并提出通过 **自适应采样 + 注意力损失加权** 的经典策略，显著提升了量子物理信息网络在复杂PDE求解中的准确性与鲁棒性，为NISQ时代的量子科学机器学习提供了可扩展的新范式。

</details>

---

### 16. [DART: Decoded Attention over Recurrent States for Efficient Long-Context Sequence Modeling](https://arxiv.org/abs/2608.02032)

**Authors**: Yixiao Qian, Song Chen, Pengkai Wang, Jiaxu Liu, Shengze Cai, Chao Xu  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.02032v1  

#### Abstract
Modern language models are built primarily from Transformers, recurrent models, and their hybrid architectures. Transformers rely on token-level attention memories, while recurrent models such as state space models (SSMs) and linear attention maintain compact recurrent states. These architectures ar...

---

### 17. [Opt.Gear Technical Report](https://arxiv.org/abs/2608.01034)

**Authors**: Juneyoung Park, Youngwook Kwon  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.01034v1  

#### Abstract
We introduce Opt.Gear, a foundation model designed for efficient on-device deployment, real-tim inference, and strong task capability. It includes a dense model (1M, 270M, and 1B) with a context length of 64K. We designed a new hybrid architecture that combines a convolutional key-value gated mixer ...

---

### 18. [CRISP: Critical Step Perception for Training Efficient Deep Search Agents](https://arxiv.org/abs/2608.01867)

**Authors**: Haosi Mo, Zihao Yan, Ruiqing Zhang, Zhongli Li, Hexuan Deng, Xuebo Liu, Min Zhang  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.01867v1  

#### Abstract
Large language models (LLMs) are increasingly extended into deep search agents that solve complex questions through multi-step interaction with external search and browsing tools. However, existing agents often incur substantial computational and interaction costs, generating lengthy trajectories th...

---

### 19. [From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding](https://arxiv.org/abs/2608.02123)

**Authors**: Zixian Li, Tong Li, Chi Xie, Xiaohui Song, Haonan Lu  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.02123v1  

#### Abstract
Speculative decoding accelerates LLM inference only when drafted continuations survive target-model verification. Semi-autoregressive drafters such as DSpark predict an entire token block with one backbone forward and refine it with a lightweight Markov head. However, DSpark decodes this block as a ...

---

### 20. [Learning-Based Collaborative MEC for LLM Inference with Soft-Deadline Awareness via Transformer-Enhanced PPO](https://arxiv.org/abs/2608.02031)

**Authors**: Ngoc Hung Nguyen, Bjorn Landfeldt  
**Category**: cs.DC  
**Published**: 2026-08-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.02031v1  

#### Abstract
This paper investigates collaborative mobile edge computing (MEC) servers for large language model (LLM) inference under soft deadline constraints. In this system, to improve the quality of service, computations are expected to be completed within their deadlines. However, due to dependencies among ...

---

### 21. [CARE: A Cascaded Framework for Efficient and Reliable Time Series Anomaly Detection](https://arxiv.org/abs/2608.01885)

**Authors**: Zemin Chao, Qianhui Xu, Jianhe Cen, Guangzhi Ge, Xiao Chen, Hoangzhi Wang  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.01885v1  

#### Abstract
While deep learning models have achieved state-of-the-art performance in time series anomaly detection, their complex architectures incur substantial inference overhead. Existing methods typically apply a uniform inference strategy across all data points, which is inefficient given that anomalies ar...

---

### 22. [CoRe-GNN: Multilevel Message passing on Coarsened graphs](https://arxiv.org/abs/2608.02128)

**Authors**: Antonin Joly, Nicolas Keriven, Aline Roumy  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.02128v1  

#### Abstract
Training Graph Neural Networks on large graphs is challenged by the memory cost of storing all node representations across layers. We show that several existing scalable approaches can be written as structured modifications of the GNN propagation matrix, providing a unified perspective that exposes ...

---

### 23. [OpenClaw and Ollama in Agentic AI: Toward Fully Autonomous and Scalable AI Agent Systems](https://arxiv.org/abs/2607.28629)

**Authors**: Konstantinos I. Roumeliotis, Ranjan Sapkota  
**Category**: cs.AI  
**Published**: 2026-08-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.28629v1  

#### Abstract
The rapid transition from reactive large language models (LLMs) to persistent, action-capable systems has exposed critical gaps in the architectural understanding of Agentic AI, particularly in separating inference, orchestration, and execution layers for autonomous AI agents. Despite recent advance...

---

### 24. [NeSyFS: A Neuro-symbolic Fast-Slow Thinking Framework for LLM Agent under Partial Observability](https://arxiv.org/abs/2607.28942)

**Authors**: Duo Xu, Faramarz Fekri  
**Category**: cs.AI  
**Published**: 2026-08-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.28942v1  

#### Abstract
Recently Large Language Models (LLMs) have been increasingly deployed as autonomous agents in applications such as self-reflection, retrieval-augmented generation, and scientific discovery. In these settings, agents must act based on limited observations rather than full environmental states, leadin...

---

### 25. [OoO-Spec: Out-of-Order Semantic Speculation for Fast Tool Calling](https://arxiv.org/abs/2608.00814)

**Authors**: Zhiheng Zhang, Mujie Xu, Feiyu Sun, Zhixin Zhang  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.00814v1  

#### Abstract
LLMs generate tool calls token by token, even though the function choice and argument values can often be predicted in parallel from the request and tool schema. ToolSpec reduces this cost by drafting schema tokens and retrieving earlier calls, but cannot propose request-specific values absent from ...

---

### 26. [DeltaFlow: Noise-Adaptive Bidirectional Gated Delta Networks for Embedded Language Flows](https://arxiv.org/abs/2608.01240)

**Authors**: Guangfu Guo, Xiaoqian Lu, Linsey Pang, Weiran Yao, Haolin Chen, Kunpeng Liu, Long Cheng  
**Category**: cs.CL  
**Published**: 2026-08-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.01240v1  

#### Abstract
Embedded Language Flows (ELF) rely primarily on full non-causal attention for iterative denoising, repeatedly incurring quadratic sequence-mixing cost at each sampling step. Gated Delta Networks (GDNs) provide an efficient recurrent alternative, but their standard causal formulation cannot directly ...

---

### 27. [AReaL-DTE: Sparse Policy-Weight Transfer for Online Agentic Reinforcement Learning](https://arxiv.org/abs/2608.00455)

**Authors**: Yingqi Peng, Jiawei Zhang, Wenhao Zhou, Ruida Xu, Ran Yan, Wei Dong, Yi Gao, Zhiqiang Ding, Tongkai Yang, Binhang Yuan  
**Category**: cs.DC  
**Published**: 2026-08-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.00455v1  

#### Abstract
Online agentic reinforcement learning implemented with micro-services separates policy training from rollout generation, improving scalability and modularity while potentially making frequent policy-weight synchronization a critical systems overhead. Shared storage naturally connects these services ...

---

### 28. [Uncertainty-Aware Simulation-Based Inference for Operations Research with Large Language Models](https://arxiv.org/abs/2608.00019)

**Authors**: Liang Guo, Lin Shaochong, Shen Zuo-Jun Max, Zhang Kun  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.00019v1  

#### Abstract
Deploying large language models (LLMs) for operations research (OR) tasks remains challenging because correctness depends on a coherent modeling process, not merely a correct final answer. Standard autoregressive generation operates on a myopic policy, which sometimes fails to anticipate whether a p...

---

### 29. [Agentic Bayesian Optimization through Surrogate-Augmented Autoresearch](https://arxiv.org/abs/2608.00316)

**Authors**: Paul Brunzema, Louis Tiao, Nhat Le, Kevin De Angeli, Yao Xuan, Djordje Gligorijevic  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.00316v1  

#### Abstract
Bayesian optimization (BO) has become the standard tool for sample-efficient optimization and owes its efficiency to uncertainty-aware search driven by generic statistical priors. Richer domain priors can improve BO in principle, but encoding them through tailored kernels or problem structure is dif...

---

### 30. [An Embedded RISC-V Evaluation of Kolmogorov--Arnold Networks in Hard-Constrained Recurrent Physics-Informed Models](https://arxiv.org/abs/2608.00737)

**Authors**: Enzo Nicolas Spotorno, Josafat Leal Filho  
**Category**: cs.LG  
**Published**: 2026-08-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.00737v1  

#### Abstract
Hard-constrained recurrent physics-informed networks (HRPINNs) embed known dynamics inside a recurrent numerical integrator and restrict a neural branch to learning only the residual dynamics that the first-principles model does not capture. Kolmogorov--Arnold Networks (KANs) have been proposed as p...

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

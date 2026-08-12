# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-12 07:03:51 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [MoE Proxy Models for Low-Cost Failure Reproduction and Diagnosis in LLM RL Post-Training](https://arxiv.org/abs/2608.10823)

**Authors**: Yikai Wang, Chuansai Zhou, Yuhang Zhou, Weiqiang Wu, Cong Wu, Yue Deng, Ben Feng, Mingming Zhu, Beirong Zhou, Zhibin Wang, Sheng Zhong, Chen Tian, Wangze Zhang  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.10823v1  

#### Abstract
Reinforcement learning (RL) post-training of large language models (LLMs) is computationally intensive and involves complex system pipelines with substantial debugging overhead. In practice, factors such as framework adaptation, numerical precision, and operator implementation can cause failures, in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*MoE Proxy Models for Low-Cost Failure Reproduction and Diagnosis in LLM RL Post-Training*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLM）在进行 **Reinforcement Learning (RL) post-training** 时，系统复杂、计算开销巨大，且容易因框架适配、数值精度、算子实现等问题引发训练故障（如梯度溢出、loss发散等）。直接在原始大模型上复现和调试这些故障成本极高。

特别是对于 **Mixture-of-Experts (MoE)** 架构，其动态路由机制和稀疏计算进一步增加了行为复现的难度。因此，亟需一种低成本、高保真的代理模型（proxy model）用于故障复现与辅助诊断。

### 提出了什么新方法或新思路
本文提出了一种 **multi-view, frequency-aware MoE expert pruning** 方法，构建轻量级 MoE 代理模型，用于低开销的故障复现与诊断。该方法的核心思想是：

- **保留故障相关的模型特性**：识别并保留对故障敏感的关键因素：
  - **Routing decisions**（路由决策）
  - **Expert-utilization patterns**（专家使用模式）
  - **Hidden-state representations**（隐藏状态表示）

- **多视角建模专家相似性**：
  - **Router-parameter view**：基于路由器权重的静态偏好
  - **Co-activation view**：基于专家共激活频率的动态组合关系
  - **Routed-context view**：基于被路由到各专家的输入隐状态原型

- **融合距离 + K-Medoids 聚类**：将三种距离加权融合后，通过 K-Medoids 聚类选择代表性专家，并优先选取激活频率高的专家作为代表。

- **结构保持设计**：不进行参数平均或微调，完整保留原模型的 **backbone architecture**、**Top-k routing 机制** 和标准 MoE 执行路径。

### 相比现有方法的优势
| 维度 | 现有方法（Pruning/Distillation/Quantization） | 本文方法 |
|------|---------------------------------------------|---------|
| 目标 | 推理加速 / 压缩率 / 任务性能 | 故障响应一致性 + 训练动态保真 |
| 中间过程保留 | 否（仅关注输出分布） | 是（保留路由路径、隐藏状态传播） |
| MoE 特性处理 | 易破坏路由结构（如合并专家） | 显式保护路由机制与专家组合模式 |
| 是否需要微调 | 多数需要 | **无需额外 fine-tuning 或参数平均** |

> ✅ 创新点总结：首次系统定义“故障可复现性”为代理模型的优化目标，提出面向 **fault-aware compression** 的 MoE 专家剪枝框架，在极低资源下实现高保真故障模拟。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Calibration Set**：独立校准集，用于收集专家激活频率、Top-k选择、路由权重和 routed hidden states（不与测试样本重叠）
- **Task Environment & Evaluation**：使用 **GSM8K** 数据集进行任务能力评估和故障复现实验环境

### 实验设置
#### 模型与硬件平台
- **主实验模型**：`Qwen3-30B-A3B`（每层128个routed experts，Top-8 routing）
  - 构造保留 48 和 64 专家的 proxy model
- **泛化性验证模型**：`DeepSeek-V3.2`（256 experts）、`DeepSeek-V4-Flash`
- **硬件平台**：
  - 华为 A3 服务器，配备 16 × Ascend 910 NPU（用于 Qwen）
  - 更大规模部署于 512 NPU 集群（用于 DeepSeek）

#### RL 训练配置
- 框架：VERL + vLLM + GRPO
- 训练流程一致：原始模型与 proxy model 使用相同的 RL 配置（除非特别说明）

### 评估指标
| 类别 | 指标 |
|------|------|
| **计算效率** | NPU 数量、单步耗时、NPU-hour 成本 |
| **训练动态保真度** | Reward 曲线趋势、Actor KL Loss 变化趋势 |
| **故障复现能力** | 异常类型一致性、metric 变化方向、时间趋势一致性 |
| **任务能力保留** | GSM8K 准确率 |
| **专家选择有效性** | 不同剪枝策略下的性能对比、消融实验 |

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Frequency-only selection** | 仅按专家激活频率排序保留前 K 个 |
| **Random selection** | 随机选择 K 个专家 |
| **Fixed First-K experts** | 固定保留前 K 个专家 |
| **Expert grouping & merging**（Li et al., 2026） | 模型中心化的专家合并方法（Sub-MoE） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 模型 | 类型 | 专家数 | NPUs | 单步时间 | NPU-hour/step | 成本降幅 |
|------|------|-------|-------|------------|----------------|----------|
| Qwen3-30B-A3B | Original | 128 | 16 | 128.93s | 0.573 | — |
| Qwen3-30B-A3B | Proxy | 48 | 8 | 114.83s | 0.255 | ↓55.5% |
| DeepSeek-V3.2 | Original | 256 | 512 | ~50min | 426.7 | — |
| DeepSeek-V3.2 | Proxy | 16 | 64 | ~12min | 12.8 | ↓**33.3×** |

> 💡 **结论**：proxy model 将硬件需求降低 **50%–87.5%**，NPU-hour 成本最高下降 **33.3倍**

---

### 与基线方法的对比结果

#### 任务能力保留（GSM8K Accuracy）

| Selection Strategy | GSM8K Accuracy (DeepSeek-V3.2, 16 experts) |
|--------------------|-------------------------------------------|
| Original Model     | 95.0%                                     |
| **Ours**           | **58.5%**                                 |
| Frequency-Only     | 53.0%                                     |
| Random Selection   | 3.0%                                      |
| Fixed First-K      | 3.0%                                      |

> ✅ 本文方法显著优于其他剪枝策略，尤其避免了无差别剪枝导致的能力崩溃。

#### 故障复现能力对比

##### （1）Rollout LogP Precision Fault（数值一致性故障）
- 注入方式：降低 Rollout LogP 计算中缩放因子的数值精度
- 结果：
  - 原始模型与 48-expert proxy 均出现：
    - Rollout-Training LogP 差异漂移
    - KL divergence 上升
  - **变化方向与时间趋势高度一致**
  - Expert merging 基线无法稳定训练

##### （2）Actor Update Omission Fault（优化稳定性故障）
- 注入方式：禁用 actor 参数更新
- 结果：
  - 两者均表现出明显的 **reward stagnation**
  - 滚动平均 reward 曲线趋于平坦
  - 表明 proxy 模型能正确反映“策略无法改进”的异常行为

> ✅ 在两类典型故障下，proxy model 成功复现了 **异常类型、metric 变化方向、演化趋势**

---

### 消融实验结果

#### 多视图组合效果（Qwen3-30B-A3B）

| Expert Similarity View | 48 Experts (Acc.) | 64 Experts (Acc.) |
|------------------------|-------------------|-------------------|
| Original Model         | 93.5%             | 93.5%             |
| Context + Co-activation | 44.0%             | 76.0%             |
| Router + Co-activation  | 46.5%             | 81.0%             |
| Context + Router        | 44.5%             | 77.5%             |
| **Ours (Three Views)**  | **48.0%**         | **86.0%**         |

> 🔍 三视图联合建模带来最大增益，说明各视图捕捉互补特征，共同提升专家代表性。

#### 聚类初始化策略分析
- 使用“最频繁激活专家”作为 K-Medoids 初始 medoid，提升了聚类稳定性与最终性能。

---

## 4. 关键结论和发现

### 主要发现
1. **MoE 模型中的故障响应具有可迁移性**：通过保留关键结构（routing、co-activation、context representation），可在小规模 proxy model 上复现原模型的故障行为。
2. **无需微调即可构建有效代理模型**：提出的结构保持剪枝方法无需参数平均或后续 fine-tuning，仍能维持训练动态一致性。
3. **多视角相似性建模优于单一信号**：结合 router 参数、共激活行为与上下文表示，能更全面地衡量专家重要性。
4. **proxy model 具备实用价值**：
   - 可用于 **低成本故障复现**
   - 支持 **定向验证（targeted validation）**
   - 辅助 **根因分析与诊断（auxiliary diagnosis）**
   - 显著减少在原模型上的昂贵试错成本

---

### 方法的局限性
1. **依赖校准数据质量**：若 calibration set 不能覆盖典型输入分布，可能导致专家选择偏差。
2. **未完全匹配绝对指标值**：虽然趋势一致，但 reward/KL 的绝对数值可能存在偏移。
3. **极端压缩下能力退化明显**：当专家数极少时（如从256→16），任务准确率仍大幅下降（95% → 58.5%），限制了极限压缩场景的应用。
4. **当前聚焦 MoE 架构**：是否适用于 dense 模型或其他稀疏架构尚待验证。

---

### 未来工作方向
1. **自动化权重融合策略**：探索 αr, αc, αf 的自适应调整机制，根据不同故障类型动态加权。
2. **跨模型迁移代理构造**：研究能否在一个 MoE 模型上学到的剪枝策略迁移到另一个架构相近的模型。
3. **集成诊断接口**：将 proxy model 与可视化工具、日志分析系统集成，形成端到端的 RL 故障诊断 pipeline。
4. **扩展至更多故障类型**：验证对 reward hacking、over-optimization、KL collapse 等复杂现象的复现能力。

---

> 📌 **总体评价**：  
> 本文提出了一个面向 **故障诊断友好型压缩** 的全新范式，突破了传统模型压缩以“推理效率”为核心的局限，开创了 **debugging-efficient AI systems** 的研究方向。其实验充分、设计合理，在工业级 MoE 模型上展现出强大实用性，是 LLM 系统可靠性工程的重要进展。

</details>

---

### 2. [Scheduling Mixed RL Rollouts Beyond Prefix Locality](https://arxiv.org/abs/2608.11152)

**Authors**: Zetao Hong, Song Yuan, Yuanhao Ding, Yibo Zhu, Daxin Jiang, Zhibin Wang, Chen Tian  
**Category**: cs.DC  
**Published**: 2026-08-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.11152v1  

#### Abstract
Modern reinforcement learning (RL) post-training pipelines for large language models (LLMs) increasingly combine rollout workloads across multiple domains and feedback paradigms. Prefix-aware routing improves inference efficiency through cache reuse and load balancing, but it does not control how he...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scheduling Mixed RL Rollouts Beyond Prefix Locality

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

现代大型语言模型（LLM）在强化学习（RL）后训练中，通常需要同时处理来自多个领域和反馈范式的混合 rollout 工作负载（如 RLVR、RLHF 和 agentic rollouts）。这些异构任务具有不同的序列结构、交互模式和 KV-cache 驻留时间，导致对推理服务资源的需求差异巨大。

尽管现有的 LLM 推理系统（如 vLLM Router）通过 **prefix-aware routing** 提升了缓存复用效率，但它们仅优化请求放置（placement），**缺乏对会话准入（admission control）的有效管理**。这会导致：

- 多个新会话竞争有限的 KV-cache 容量；
- 可重用的 prefix 被频繁驱逐，造成“冷 prefill”激增；
- 吞吐下降、延迟上升，且无法控制不同 workload 类之间的资源分配。

因此，**如何在不改变 trainer 所定义的工作负载混合比例的前提下，高效调度异构 rollout 请求，成为关键挑战**。

---

### 🚀 **提出了什么新方法或新思路**

本文提出 **MISA-T**（**Mix-aware Session Admission with a Time factor**），一种部署于路由层的 admission 控制策略，旨在解决上述问题。其核心思想是将 admission 决策视为对 KV-cache 容量的“承诺”，并据此进行精细化控制。

#### 主要创新点包括：

1. **Overload-aware Session Admission（过载感知会话准入）**
   - 动态计算每个推理实例的会话准入上限（session cap），基于观察到的 KV 需求和过载压力。
   - 当达到容量限制时，新会话被 HOLD 并周期性重试，而已有 continuation 得以保留，避免 cache churn。
   - 减少对静态并发参数的手动调优依赖。

2. **Workload-aware Capacity Allocation（工作负载感知容量分配）**
   - 将受保护的 KV 容量按 workload 类别（RLVR / RLHF / Agentic）划分软配额。
   - 不同类别的会话拥有独立的 admission cap，防止某一类占用过多资源。
   - 保证最终完成的任务分布接近 trainer 设定的目标 mixture。

3. **Residency-time-aware KV Accounting（驻留时间感知的 KV 计费机制）**
   - 引入 **block-time demand** 概念：`KV_block × residency_time`，更准确反映资源占用。
   - 对于 agentic 任务，residency time 包括模型推理间隔中的 tool-execution 时间，期间 KV 仍需保留。
   - 基于此动态调整各类别的配额，实现更公平高效的资源利用。

> 💡 **关键洞察**：Admission 是一种 KV 资源承诺，必须考虑后续增长、延续请求及跨轮次驻留。

---

### ⚖️ **相比现有方法的优势**

| 维度 | 现有方法（如 vLLM Router） | MISA-T |
|------|----------------------------|--------|
| 是否控制 admission | ❌ 仅做 placement | ✅ 显式 admission 控制 |
| 是否区分 workload 类型 | ❌ 视所有 session 等价 | ✅ 按类别分配容量 |
| 是否考虑 KV 驻留时间 | ❌ 仅看当前 footprint | ✅ 加权 block-time demand |
| 是否维持 workload mixture | ❌ 可能因调度偏差失衡 | ✅ 近似保持目标分布 |
| 自适应能力 | ❌ 依赖手动 concurrency sweep | ✅ 在线动态调节 |

---

## 2. 核心实验方法和设置

### 📊 **使用的模型与场景**

实验基于两个真实部署的大规模 LLM：

- **Step3.7**：196B-A11B sparse-MoE 模型，用于 end-to-end RL 训练。
- **Qwen3.6-35B-A3B**：35B 参数密集模型，用于 rollout-only 测试。

混合 workload 包括：
- **RLVR**（Reinforcement Learning with Verifiable Rewards）
- **RLHF**（Reinforcement Learning from Human Feedback）
- **Agentic Rollouts**（多轮工具调用型任务）

---

### 🔧 **实验设置**

#### （1）End-to-End RL Training 实验
- 在 Step3.7 上运行 50 轮完整 RL 训练。
- 固定初始化、采样流、目标 mixture（Agent:RLVR:RLHF = 48.6:33.1:18.2）、reward 配置。
- 对比 MISA-T 与 vLLM Router 的整体性能。

#### （2）Rollout-Only Ablation 实验
- 固定模型 checkpoint，模拟在线混合 rollout 请求流。
- 报告三次独立运行的平均值。
- 硬件配置：
  - Step3.7：2×H200 节点，TP=8
  - Qwen3.6-35B-A3B：16×H100 GPUs，4×TP=4

#### （3）Baseline 方法对比
1. **vLLM Router (sweep-tuned)**：最优静态并发下的 cache-aware placement。
2. **vLLM Router (128K-safe / high-load)**：保守或宽松并发设置。
3. **Session Admission**：全局自适应 admission，无 workload 区分。
4. **MISA**：加入 workload-aware 分配，但无 residency 时间加权。
5. **MISA-T**：完整版本，含 residency-time 加权。

---

### 📈 **评估指标**

| 指标 | 描述 |
|------|------|
| **Rollout Throughput** | 每分钟返回给 trainer 的完整 rollout 样本数（非 request 数） |
| **Request RPM** | 每分钟成功处理的 inference 请求数量 |
| **Prefill TPS / Decode TPS** | Prompt token 和 generated token 的有效吞吐率 |
| **Prefix Hit Rate** | 请求中命中 prefix cache 的 prompt token 比例 |
| **Mean Iteration Time** | 单轮训练的平均耗时（end-to-end） |
| **DTv(p, q)** | 完成 mixture 与目标 mixture 的 total variation distance |
| **Task Score** | 如 pass@4 在 SWE-Pro/SWE-Verified 等任务上的表现 |

---

## 3. 主要实验结果和性能指标

### 📊 **关键性能数据汇总**

#### ✅ **End-to-End 实验结果（Step3.7, 50 iterations）**

| 指标 | vLLM Router | MISA-T | 提升幅度 |
|------|-------------|--------|---------|
| Rollout Throughput | — | ↑ **+35.6%** | +35.6% |
| Mean Iteration Time | — | ↓ **-22.8%** | 显著加速 |
| Prefix Hit Rate | 74.5% | → **96.2%** | +21.7 pts |
| Mixture Deviation (DTv) | 4.14 pp | → **2.71 pp** | 下降 34.5% |
| Task Scores (pass@4) | Comparable | Comparable | <0.5 pp 差距 |

> ✔️ MISA-T 在提升吞吐的同时，几乎完全保持了原始 workload mixture 和任务性能。

---

#### ✅ **Rollout-Only Ablation 结果**

| 模型 | 方法 | Rollout Throughput ↑ | Prefix Hit Rate |
|------|------|------------------|----------------|
| **Step3.7** | vLLM Router (sweep) | 0.0% | 95.9% |
| | **MISA-T** | **+53.3%** | **97.8%** |
| **Qwen3.6-35B-A3B** | vLLM Router (sweep) | 0.0% | 92.4% |
| | **MISA-T** | **+43.6%** | **95.3%** |

此外：
- Request RPM 提升达 **+45.5% (Step3.7)** 和 **+37.5% (Qwen)**。
- 在高负载下，vLLM Router 的 hit rate 崩溃至 **4.5%**，而 MISA-T 成功避免此现象。

---

### 🔍 **消融实验分析（Ablation Study）**

逐步验证各组件贡献（相对 Session Admission）：

| 模型 | 方法 | Throughput 提升 |
|------|------|----------------|
| Step3.7 | MISA vs Session Admission | +10.4% |
| Step3.7 | MISA-T vs MISA | **+24.3%** |
| Qwen3.6 | MISA vs Session Admission | +9.7% |
| Qwen3.6 | MISA-T vs MISA | **+9.7%** |

> 💡 表明 **residency-time weighting** 是最大收益来源，尤其在 decode-heavy 或 multi-turn 场景中更为显著。

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **Placement-only routing 不足以应对高并发混合 rollout**
   - 缺乏 admission 控制会导致 KV-cache 雪崩式驱逐，引发 prefill 拥塞和吞吐崩溃。
   - vLLM Router 在高并发下 hit rate 从 92.4% 跌至 4.5%，证实该问题严重性。

2. **Admission 应作为 KV 资源承诺来建模**
   - 新 session 的准入直接影响未来 KV 增长与 reuse 能力。
   - MISA-T 通过动态 cap 和 HOLD 机制有效遏制过载。

3. **workload heterogeneity 必须显式建模**
   - RLVR（长 decode）、RLHF（均衡）、agentic（长 prefix + 工具间隔）行为迥异。
   - 统一 cap 会导致资源倾斜；分类配额可保障 mixture fidelity。

4. **KV residency time 是关键资源维度**
   - agentic 任务中 tool-execution 时间占高达 22.7%，期间 KV 仍需保留。
   - 使用 `KV_blocks × residency_time` 更准确衡量 block-time demand。

5. **MISA-T 可组合性强**
   - 兼容 PagedAttention、RadixAttention、CPU KV offloading 等技术。
   - 实验显示与 CPU offloading 结合后，GPU KV 利用率维持在 90%+，RPM 再提 35.6%。

---

### ⚠️ **局限性**

- 依赖准确的 **workload labeling** 和及时的 **serving-state reporting**。
- 若 session snapshot 延迟或缺失，可能导致 admission 决策短暂不准。
- 当前未联合优化外层 concurrency limit，仍假设其由 trainer 控制。

---

### 🔮 **未来工作方向**

1. **自动学习最优外层 concurrency bound**，减少人工干预。
2. **引入预测机制**，预估 session 的最终长度与 residency，进一步提升 admission 精度。
3. **扩展至更多 workload 类型**，如 streaming、search-augmented generation 等。
4. **结合动态 batching 与 admission 控制**，实现端到端资源协同优化。

---

## 总结

📌 **MISA-T 是首个面向混合 RL rollout 场景、兼顾效率与 mixture fidelity 的 admission 控制框架**。它通过 **adaptive admission + workload-aware allocation + residency-time accounting** 三重机制，在真实大规模系统中实现了：

- **最高 53.3% 的 rollout throughput 提升**
- **22.8% 的 end-to-end iteration time 缩短**
- **KV-cache hit rate 维持在 95%+**
- **workload mixture 偏差降低超 30%**

该工作揭示了：**在复杂 RL 推理系统中，routing 层不仅要“选哪里”，更要“准谁进”** —— admission 控制已成为下一代 LLM serving 架构的关键组成部分。

</details>

---

### 3. [Dreamer-SAC: Off-Policy Learning in Latent World Models for Sample-Efficient Autonomous Driving](https://arxiv.org/abs/2608.10386)

**Authors**: Jiazhuo Li, Linjiang Cao, Qi Liu, Xi Xiong  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.10386v1  

#### Abstract
Sample-efficient reinforcement learning for autonomous driving is often limited by the trade-off between data efficiency and model bias. While world models reduce the reliance on costly environment interactions, policy optimization over learned dynamics remains sensitive to prediction errors. This p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Dreamer-SAC: Off-Policy Learning in Latent World Models for Sample-Efficient Autonomous Driving》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **样本效率低**：传统 **model-free RL**（如 SAC、PPO）在自动驾驶中需要大量环境交互，成本高昂且存在安全风险。
- **模型偏差（model bias）**：基于世界模型（world model）的方法（如 DreamerV3）虽然提升样本效率，但依赖长期预测轨迹进行策略优化，导致累积预测误差影响策略可靠性，尤其在安全关键场景（如碰撞）下表现不佳。
- **策略优化范式限制**：现有基于 latent world models 的方法多采用 on-policy 学习，无法复用历史真实数据，限制了数据利用率。

### 🚀 提出的新方法：Dreamer-SAC
提出一种结合 **latent world model** 与 **off-policy RL** 的新框架——**Dreamer-SAC**，其核心思想是：
- 利用 **Recurrent State-Space Model (RSSM)** 构建紧凑的 latent dynamics 表示。
- 在 latent space 中生成 **短视界（short-horizon）rollouts**，并与 **真实环境交互数据** 联合用于 **SAC** 的 off-policy 优化。
- 对真实数据使用 **1-step TD target**，对模型生成数据使用 **n-step bootstrapping target**，以更充分地利用预测轨迹中的长期奖励信息。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | Dreamer-SAC 的改进 |
|------|--------|------------------|
| **DreamerV3** | 完全依赖 imagined rollouts，on-policy 更新，易受 model bias 影响 | 引入真实数据“锚定”策略学习，降低偏差 |
| **SAC / PPO** | 完全 model-free，样本效率低 | 利用 latent rollouts 显著减少真实交互需求 |
| **MBPO 等 MBRL** | 在原始状态空间建模，难以处理高维视觉输入 | 在 latent space 进行 rollout，支持端到端视觉输入 |

> ✅ **核心优势**：在保持高样本效率的同时，通过混合训练机制缓解 model bias，实现更安全、可靠的驾驶策略。

---

## 2. 核心实验方法和设置

### 📊 数据集与仿真环境
- **环境**：使用高保真自动驾驶模拟器 **MetaDrive**。
- **场景**：`BIG_BLOCK_SEQUENCE-CC` —— 包含连续弯道的多车道高速公路，交通密度为 0.1（约 30 辆车/km）。
- **观测输入**：
  - 图像模态：前向摄像头图像 `84×84×3`（RGB）
  - 向量模态：`125-D` 向量（120 条 LiDAR 射线 + 5 维自车状态）
- **动作空间**：连续控制 `[steering, throttle/brake] ∈ [-1,1]^2`
- **终止条件**：碰撞（collision）、驶出道路（out-of-road）、成功完成路线

### 🧪 实验设置
- **总训练步数**：40,000 环境交互步
- **rollout horizon H**：默认设为 5
- **折扣因子 γ**：0.99
- **网络架构**：
  - RSSM：确定性隐藏状态 256-D，离散随机状态（32 变量 × 32 类）
  - SAC Actor/Critic：两层 MLP（256 单元），SiLU 激活
- **优化器**：Adam，world model 学习率 1e-4，SAC 学习率 3e-4

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Average Return** | 训练过程中累计奖励均值（主性能指标） |
| **Collision Frequency (/km)** | 每公里碰撞次数（越低越好） |
| **Out-of-Road Frequency (/km)** | 每公里驶出道路次数 |
| **Average Travel Distance (m)** | 平均行驶距离（反映策略稳健性） |
| **Maximum Travel Distance (m)** | 最大单次行驶距离 |
| **Average Speed (km/h)** | 平均速度 |

### 🆚 基线方法对比
- **DreamerV3**：基于 latent world model 的 on-policy 方法，仅使用 imagined rollouts
- **SAC**：标准 off-policy model-free 算法
- **PPO**：主流 on-policy 算法，低样本效率代表

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（训练返回值）
| 方法 | Average Return |
|------|----------------|
| **Dreamer-SAC (Ours)** | **371.4** |
| DreamerV3 | 189.2 |
| SAC | 134.5 |
| PPO | 65.5 |

> ✅ Dreamer-SAC 显著优于所有 baseline，在仅 40K 步内达到最高性能。

### 🚘 泛化能力测试（扩展道路网络，3倍长度）
| Method | Collision (/km) | Out-of-road (/km) | Avg Speed | Avg Dist (m) | Max Dist (m) |
|--------|------------------|-------------------|-----------|---------------|---------------|
| **Dreamer-SAC** | **1.56** | **1.37** | 25.5 | **320.8** | **951.6** |
| DreamerV3 | 2.88 | 1.76 | **39.7** | 215.5 | 515.2 |
| SAC | 1.68 | 8.85 | 28.0 | 95.0 | 348.0 |
| PPO | 3.72 | 4.03 | 25.1 | 129.1 | 369.6 |

> 🔍 **关键观察**：
> - DreamerV3 虽速度快（39.7 km/h），但事故率高，生存距离短 → **过度乐观，策略激进**
> - Dreamer-SAC 更保守，倾向于跟车减速而非强行超车 → **安全性优先，长程可靠**

### 🔬 消融实验（Ablation Study）

| 配置 | Return | 分析 |
|------|--------|------|
| A1 (H=1, 无真实数据) | 167.3 | 仅靠预测数据效果差 |
| A2 (H=1, 有真实数据) | 226.2 | ✅ 真实数据显著提升性能 |
| A3 (H=5, 完整版) | **371.4** | 默认最优配置 |
| A4 (H=5, 无真实数据) | 347.5 | 仍有效，但弱于完整版 |
| A5 (无 decoder 梯度) | -5.6 | ❌ 缺少重构损失导致崩溃 → **observation reconstruction 至关重要** |
| A6 (仅 decoder) | 280.8 | 任务相关预测头提供额外增益 |
| A7 (1-step TD target) | 331.6 | n-step 更优 |
| A8 (连续 latent) | 303.4 | ✅ **离散 latent 表现更优**，更适合多模态驾驶动态 |

> ✅ 结论：**real data + multi-objective RSSM + n-step target + discrete latent** 共同构成高性能的关键。

---

## 4. 关键结论和发现

### 🎯 主要发现
1. **混合训练优于纯模型驱动**  
   联合使用 **真实经验** 与 **latent rollouts** 可显著提升策略稳定性与安全性，避免 pure imagined training 导致的 model exploitation。

2. **短视界 rollout 效果最佳（inverted-U 现象）**  
   - 当 `H=0`（无 rollout）时，策略退化为静止避险模式。
   - `H=5` 时性能达峰（371.4）。
   - `H=20` 时性能下降至 285.6。
   - ➡️ 存在 **rollout horizon 与性能之间的 inverted-U 关系**，短 horizon 提供最佳平衡。

3. **n-step target 显著优于 1-step TD**  
   在 predicted trajectory 上使用 n-step return 可更好传播长期奖励信号，提升 critic 学习质量。

4. **离散 latent state 更适合驾驶任务**  
   相比连续 Gaussian latent，离散 categorical latent 能更好捕捉部分可观测环境下的多模态未来分布。

5. **模型倾向保守预测安全事件**  
   分析显示 world model 会 **高估碰撞等负奖励概率**，这是由于稀有事件难以准确建模所致，但也促使策略更谨慎。

---

### ⚠️ 方法的局限性
- **rollout horizon 固定**：未动态调整 H，可能在不同场景下非最优。
- **依赖高质量 world model**：若 representation learning 失败（如 A5），整个框架崩溃。
- **未考虑通信或多智能体交互**：当前设定为单车决策，未涉及协同驾驶。
- **仿真到现实的迁移挑战**：尚未在 real-world 数据上验证泛化性。

---

### 🔮 未来工作方向
1. **自适应 rollout horizon**：根据模型不确定性或状态复杂度动态调整 H。
2. **不确定性感知的 rollout 权重机制**：对高不确定性的预测样本降权或过滤。
3. **扩展至多智能体 setting**：研究基于 world model 的交互预测与联合决策。
4. **真实数据验证**：在 nuScenes、Waymo Open Dataset 等真实驾驶数据上测试框架实用性。
5. **集成语义先验**：引入地图、交通规则等 symbolic knowledge 增强 world model 的可解释性与鲁棒性。

---

> 💡 **总体评价**：  
> Dreamer-SAC 成功将 **latent world model 的样本效率** 与 **off-policy RL 的稳定性和数据复用能力** 相结合，提出了一种适用于高维视觉输入、强调安全性的自动驾驶决策新范式，为 MBRL 在复杂现实任务中的应用提供了重要实践路径。

</details>

---

### 4. [Mitigating Context Interference for Reliable and Efficient Search Agents](https://arxiv.org/abs/2608.10743)

**Authors**: Boyang Xue, Bin Wu, Shuofei Qiao, Sheng Wang, Rui Wang, Yiming Du, Hongru Wang, Jeff Z. Pan, Emine Yilmaz, Kam-Fai Wong, Aldo Lipani  
**Category**: cs.CL  
**Published**: 2026-08-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.10743v1  

#### Abstract
Recent research empowers Large Language Models (LLMs) as multi-turn search agents to iteratively retrieve and generate outputs until complex tasks are solved. However, the contexts of multi-turn search agents are lengthy and complex. For example, the retrieved set of documents in each turn would ine...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Mitigating Context Interference for Reliable and Efficient Search Agents*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本论文聚焦于**多轮搜索代理（multi-turn search agents）中的上下文干扰（context interference）问题**。随着大语言模型（LLMs）被用作能够主动调用搜索引擎进行迭代检索与生成的智能体，其上下文变得越来越长且复杂，包含历史问题、搜索查询、检索到的文档和推理步骤。

然而，检索器返回的文档集合中常包含大量无关或噪声信息，这些信息会“分散”LLM的注意力，导致其无法正确利用内部知识（Kr）和外部知识（KE），从而降低搜索代理的**可靠性（reliability）** 和 **效率（efficiency）**。

这一现象被称为 **context interference**，即“过多无关上下文使 LLM 被误导”。

---

### ✅ 提出了什么新方法或新思路

论文系统研究了上下文干扰的来源，并提出以下创新方案：

#### （1）首次揭示干扰主要来源于最新检索文档  
通过消融实验发现，**最新的检索文档是造成上下文干扰的主因**，而历史查询和文档的影响较小。

#### （2）提出基于蒸馏的上下文精炼器（distill-based context refiner）  
- 利用高级教师模型（如 GPT-4）从原始检索文档中提取与当前搜索查询最相关的关键信息。
- 构建一个用于上下文精炼的训练数据集 $D_c$。
- 使用监督微调（SFT）训练一个轻量级 LLM（如 Qwen2.5-7b/3b-Instruct）作为 **Context Refiner**，实现动态上下文压缩与去噪。

> 示例见 Figure 2：原始文档长达 1287 tokens，经精炼后仅保留 143 tokens 的关键信息，显著提升准确率。

#### （3）将上下文精炼引入强化学习训练流程（CRRL）  
提出 **Context-Refined Reinforcement Learning (CRRL)** 框架，在 RL 的 rollout 阶段使用 Context Refiner 动态净化上下文，从而提高轨迹质量，进一步增强训练效果。

> 这标志着从“直接生成”向 **“先精炼上下文再生成”（refine context and then generate）** 新范式的转变。

---

### ✅ 相比现有方法的优势

| 方法 | 局限性 | 本文优势 |
|------|--------|----------|
| **GPT-Compress**（通用摘要） | 可能丢失关键信息，泛化能力差 | 更精准地保留任务相关的细节 |
| **Self-Refine**（自身精炼） | 小模型缺乏信息提取能力 | 通过蒸馏让弱模型也能学会精炼 |
| **Prompt-based filtering** | 敏感于提示设计，鲁棒性差 | 模型化处理，更稳定可扩展 |
| **Ranking-based filtering**（阈值过滤） | 易误删有用文档，依赖人工设定 | 自适应识别关键内容，无需硬阈值 |

> ✅ **Context Refiner 性能接近 GPT-4 精炼水平，远超其他基线，且不依赖外部强模型推理，适合部署。**

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

涵盖单跳与多跳闭卷问答（closed-book QA）任务：

| 类型 | 数据集 |
|------|-------|
| **Single-hop QA** | Natural Questions (NQ), TriviaQA, PopQA |
| **Multi-hop QA** | HotpotQA, 2Wiki-MultiHopQA (2Wiki), MuSiQue, Bamboogle |

> 所有测试集均需外部检索才能解答，确保对搜索代理的有效评估。

---

### ⚙️ 实验设置和评估指标

#### 模型配置
- **基础 LLM**：`Qwen2.5-7b-Instruct`, `Qwen2.5-3b-Instruct`
- **检索器**：`E5` 模型 + `2018 Wikipedia dump` 作为知识库 $K_b$
- **Top-K 检索**：每轮返回 3 个文档（K=3）

#### 评估指标

| 指标 | 含义 | 用途 |
|------|------|------|
| **EM (Exact Match)** | 回答完全匹配的比例 | 衡量 **可靠性（reliability）** |
| **ART (Average Retrieval Times)** | 平均每题调用检索次数 | 衡量 **效率（efficiency）** |
| **Len. (Context Length)** | 平均上下文长度（token 数） | 衡量输入负担 |
| **AIT (Average Inference Time)** | 单题平均推理时间 | 综合效率指标 |

---

### 🔁 基线方法对比

| 方法 | 描述 |
|------|------|
| **Direct / CoT** | 无检索的基线（仅靠内部知识） |
| **IRCoT** | 标准检索增强 CoT 方法（baseline） |
| **IRCoT-o / -oq / -oqp** | 消融变体：分别屏蔽历史文档、查询、思考步 |
| **GPT-Compress** | GPT-4 对上下文做通用压缩 |
| **GPT-Refine** | GPT-4 动态提取关键信息 |
| **Self-Refine** | 基础 LLM 自我精炼 |
| **RFT / Search-GRPO** | 基于拒绝采样和 GRPO 的 RL 训练方法 |
| **CRRL** | 本文提出的结合上下文精炼的 RL 方法 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 3）

| 方法 | Avg. EM (%) ↑ | Avg. ART ↓ |
|------|----------------|------------|
| **IRCoT** | 27.5 | 2.6 |
| **Search-GRPO** | 34.6 | 2.1 |
| **CRRL (ours)** | **36.6** | **1.7** |

> 在 `Qwen2.5-7b-Instruct` 上，CRRL 相比标准 IRCoT 提升 **+9.1% EM**，同时减少 **0.9 次平均检索**。

在小模型上同样显著：
- `Qwen2.5-3b-Instruct`：CRRL 达到 **31.5 EM / 1.3 ART**，优于所有基线。

---

### 🔍 与基线方法的对比结果

#### ✅ 上下文精炼方法比较（Table 2）
| 方法 | Avg. EM | Avg. ART |
|------|--------|---------|
| IRCoT | 27.5 | 2.6 |
| GPT-Refine | 33.2 | 1.2 |
| **Context Refiner (ours)** | **32.2** | **1.2** |

> Context Refiner 接近 GPT-4 精炼性能，但可在本地运行，更适合实际应用。

#### ✅ 效率对比（Table 4）
| 方法 | Avg. Context Length | AIT (秒) |
|------|--------------------|----------|
| IRCoT | 2.3k tokens | 22.4s |
| Search-GRPO | 0.9k | 17.7s |
| **CRRL** | **0.7k** | **16.9s** |

> CRRL 显著缩短上下文长度（↓69%），降低推理成本。

---

### 🔬 消融实验结果

#### （1）不同历史组件屏蔽实验（Table 1 & Figure 3）
- **IRCoT-o**（只保留最新文档） > IRCoT：说明历史文档带来干扰
- **IRCoT-oq**（再去掉历史查询）略有提升
- **IRCoT-oqp**（再去掉历史思考）性能下降：表明思考步骤含关键信息

👉 结论：**最新检索文档是主要干扰源**

#### （2）“Recall Rate vs Recall Accuracy” 分析（Figure 7）
- “Recall Rate”高但“Recall Accuracy”低 → 文档中有答案，但模型没答出
- 差距越小，表示模型越能有效利用检索内容
- **CRRL 显著缩小该差距**，证明其缓解了上下文干扰

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **上下文干扰主要来自最新一轮的检索文档**，而非长期记忆。
2. **简单移除历史信息只能轻微改善性能**，必须对最新文档进行**精细化信息提取**。
3. **基于蒸馏训练的 Context Refiner 可以高效模拟 GPT-4 的精炼能力**，适用于资源受限场景。
4. **将上下文精炼嵌入 RL 训练流程（CRRL）可进一步大幅提升性能**，验证了“干净轨迹”对策略优化的重要性。
5. 提出 **“refine context and then generate”** 范式，为未来 AI Agent 设计提供新方向。

---

### ⚠️ 方法的局限性

1. **任务特定性**：目前集中在 QA 场景，其他工具使用或规划任务可能有不同的干扰机制。
2. **依赖教师模型构建数据集**：虽然最终模型可独立运行，但训练仍需 GPT-4 等强模型辅助。
3. **未完全集成进 agent 架构**：当前 Context Refiner 是外挂模块，尚未内化为 agent 的一部分。

---

### 🔮 未来工作方向

1. **开发端到端训练算法**，使 agent 自主具备上下文精炼能力（internalize refinement）。
2. 探索更通用的干扰检测机制，适配多种 agent 任务（tool use, planning 等）。
3. 构建统一的“感知 → 精炼 → 决策”架构：
   > **observe → refine context → generate action**
4. 减少对教师模型的依赖，探索自蒸馏或对比学习方式构建精炼数据。

---

## 💡 总结一句话

> 本文首次系统揭示了多轮搜索代理中**上下文干扰的核心来源**，提出了基于蒸馏的 **Context Refiner** 与 **CRRL 强化学习框架**，实现了“先精炼、再生成”的新范式，在多个 QA 基准上显著提升了搜索代理的**可靠性与效率**，为下一代 AI Agent 的设计提供了重要启示。

</details>

---

### 5. [Partially Observable Learning for Multi-Platform Dispatch Optimization](https://arxiv.org/abs/2608.10897)

**Authors**: Fengming Yao, Man Luo  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.10897v1  

#### Abstract
Instant delivery platforms have become a critical component of urban logistics, increasingly relying on crowdsourced couriers to fulfill highly dynamic orders. In real-world systems, couriers are not exclusive to a single platform and may concurrently serve multiple platforms, while each platform ca...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Partially Observable Learning for Multi-Platform Dispatch Optimization**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

本文针对**多平台即时配送系统**中的订单调度（dispatch）问题，解决了一个在现实场景中普遍存在但被现有研究忽视的关键挑战：**部分可观测性（Partial Observability）**。

在真实世界中：
- 骑手（courier）通常同时注册多个平台（如美团、饿了么、Uber Eats），并可自由选择接单；
- 每个平台只能观察到自己平台上的订单和骑手的部分行为（如当前所在位置、历史轨迹），而无法获知其在其他平台的接单状态或未来路线；
- 这导致每个平台对全局系统的状态是**不完全可观测的**，从而引发严重的非平稳性（non-stationarity）和决策失效。

然而，大多数现有调度方法假设平台能完全观测所有骑手状态，并强制接受分配，这在多平台环境下严重脱离实际。

---

### 🚀 **提出了什么新方法或新思路**

作者提出 **POLO**（**P**artially **O**bservable **L**earning for multi-platform dispatch **O**ptimization），一个基于 **multi-agent reinforcement learning (MARL)** 的框架，具有以下核心创新：

#### （1）**去中心化的部分可观测建模**
- 将每个“平台-网格”单元视为一个独立 agent；
- 每个 agent 仅基于**平台本地观测**（platform-local observations）进行学习和决策，符合隐私和运营约束；
- 不依赖跨平台信息共享或集中式控制。

#### （2）**注意力机制的策略表示（Attention-based Policy Representation）**
- 引入 **multi-head self-attention** 模块，聚合候选骑手中的异构信息；
- 能够捕捉骑手之间的相对竞争关系和交互影响，提升在信息缺失下的决策质量。

#### （3）**反事实奖励塑形（Counterfactual Reward Shaping）**
- 设计了一种新的 reward 函数，通过比较“执行动作后”与“移除该动作后的反事实路径”，来隔离单个 agent 对系统收益的贡献；
- 有效缓解因多个 agent 同时作用于同一骑手而导致的信用分配难题（credit assignment problem）和训练不稳定性。

---

### 🔍 **相比现有方法的优势**

| 维度 | POLO 的优势 |
|------|-------------|
| **现实适用性** | 放弃了“全可观测”、“独占骑手”等强假设，更贴近真实多平台环境 |
| **可扩展性** | 基于 grid-level 分布式 agent 架构，支持大规模城市区域调度 |
| **鲁棒性** | 在不同平台数量、空间/时间分布偏移下表现稳定 |
| **效率** | 并行化 dispatch 决策，推理速度优于序列式 baseline |

---

## 2. **核心实验方法和设置**

### 📊 **使用的数据集**

- 基于中国主流外卖平台 **Meituan** 的真实数据构建仿真器；
- 数据包含连续8天的详细订单记录（时间、起终点、价值、交付时限等）；
- 骑手初始位置来自真实观测，并模拟灵活上线行为（随机激活部分骑手）。

---

### ⚙️ **实验设置**

#### **仿真器设计**
- 时间粒度：每 **2分钟** 为一个决策时间戳；
- 地理划分：城市划分为 **六边形网格（hexagonal grids）**，边长 0.7km；
- 多平台模拟：将订单随机分配给 $ N_p \in \{1,2,3\} $ 个平台；
- 骑手行为模型：采用文献 [26] 中的行为模型，考虑接单概率受：当前订单数、取货距离、最紧急订单剩余时间等因素影响。

#### **三种规模实例**
| 规模 | 订单数 | 骑手数 | 时间段 | 网格数 |
|------|--------|--------|--------|--------|
| Base | 86 | 26 | 17:00–18:00 | 16 |
| Median | 201 | 52 | 17:00–18:00 | 20 |
| Large | 1418 | 329 | 17:30–18:00 | 120 |

---

### 🎯 **评估指标**

| 指标 | 全称 | 含义 | 目标 |
|------|------|------|------|
| **GMV** | Gross Merchandise Volume | 平台总收入（Revenue），越高越好 | ↑ |
| **ACTD** | Average Courier Travel Distance | 骑手平均行驶距离，反映效率 | ↓ |

> 注：GMV 综合考虑订单价值、支付比例 $ r_p $ 和超时补偿 $ r_c $

---

### 🆚 **基线方法对比**

| 方法 | 类型 | 描述 |
|------|------|------|
| **Random** | 随机 | 随机指派骑手 |
| **PRandom** | 半随机广播 | 在最近 $ N_{pr} $ 名骑手中随机选一人（模拟抢单机制） |
| **DisGreedy / DetGreedy** | 启发式 | 分别按最小取货距离或最小额外绕路距离匹配 |
| **IBM** | 图匹配 | 二分图匹配最大化总权重（价值 vs 绕路） |
| **TCAC** | Learning-based | 基于 RL 的并发调度，强调时间约束 |
| **FAIR** | Learning-based | 基于 RL 的公平性调度，减少收入差异 |

> 所有 learning-based 方法使用相同 state construction 和 training protocol，确保公平比较。

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据（见 Table 3）**

#### ✅ **在 Large 规模、双平台（$ N_p=2 $）场景下：**
| 方法 | GMV (↑) | ACTD (↓) |
|------|---------|----------|
| TCAC | 3006.11 ± 98.19 | 9.63 ± 0.17 |
| FAIR | 2964.76 ± 92.24 | 10.28 ± 0.16 |
| **POLO (Ours)** | **3290.76 ± 100.20** | **9.72 ± 0.17** |
| **相对提升** | **+9.47%** | 更低绕行，更高收入 |

> POLO 在 GMV 上显著领先，在 ACTD 上也优于多数 learning-based 方法。

---

### 📉 **随平台数增加的表现变化（Figure 6）**

- 所有方法在从 $ N_p=1 \to N_p=2 \to N_p=3 $ 时性能下降；
- 下降幅度：**TCAC > FAIR > POLO**；
- 表明 POLO 对多平台竞争更具**鲁棒性**；
- 特别是在 Large 场景下，POLO 的 GMV 下降最少。

---

### 🔬 **消融实验结果（Ablation Study）**

#### （1）**注意力机制消融（Table 5）**
| 变体 | GMV (Large, $ N_p=2 $) | ACTD |
|------|------------------------|-------|
| POLO (完整) | 3290.76 | 9.72 |
| w/o Att | 3178.27 | 9.86 |
| **增益** | **+112.49 (+3.6%)** | **-0.14** |

✅ 表明 attention 模块有助于捕捉骑手间竞争关系，提升决策质量。

#### （2）**奖励项消融（Table 6）**
关闭 reward 中的两个关键项：
- $ p_f $：反事实项系数
- $ p_d $：距离惩罚项系数

| 设置 | ΔGMV (vs zero) | ΔACTD (vs zero) |
|------|----------------|------------------|
| pf ≠ 0 | +83.29 | +0.16 |
| pd ≠ 0 | +316.25 | +0.38 |

✅ 两项均带来正向收益，尤其 **distance penalty** 显著提升 GMV，说明控制绕路成本至关重要。

---

### 🧪 **敏感性分析（Sensitivity Analysis）**

#### （1）**空间/时间分布偏移（Table 4）**
- 模拟不同平台有不同的定价策略或服务偏好；
- POLO 在 temporal diff 和 spatial diff 场景下仍保持最优 GMV；
- 显示其对**异构平台偏好**的良好适应能力。

#### （2）**参数鲁棒性（Figure 8）**
- 在不同 **hex size**（0.4 ~ 1.3 km）、**pruning number $ N_{pr} $**（候选骑手数）、**courier appearance ratio $ r_a $** 下测试；
- POLO 性能在各种设置下保持稳定；
- 最佳 $ N_{pr} \approx 10 $，过小或过大都会降低性能。

---

### ⏱️ **推理效率（Table 8）**

| 方法 | 推理时间 (秒/instance) |
|------|------------------------|
| TCAC | 7.6692 ($ N_p=2 $) |
| FAIR | 9.0552 |
| **POLO** | **5.1377** |

✅ POLO 推理最快，因其支持**跨网格并行 dispatch**，适合实时系统部署。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **多平台环境下的部分可观测性严重影响调度性能**  
   - 现有方法在 $ N_p > 1 $ 时性能急剧下降，尤其是在大规模场景中（如 TCAC 下降超 28%）；
   - 忽视骑手跨平台行为会导致严重误判。

2. **POLO 在复杂多平台环境中表现出更强的鲁棒性和收益能力**  
   - 在所有规模和平台数量下，POLO 均取得最高或接近最高的 GMV；
   - 同时保持较低的 ACTD，实现**收入与效率的平衡**。

3. **注意力机制和反事实奖励设计是成功的关键组件**  
   - attention 提升了对局部异构信息的整合能力；
   - counterfactual reward 缓解了多 agent 干扰带来的训练不稳定。

4. **POLO 具备良好的泛化能力和工程实用性**  
   - 对参数变化、分布偏移、系统规模变化均表现稳健；
   - 推理速度快，适合在线部署。

---

### ⚠️ **方法的局限性**

| 局限 | 说明 |
|------|------|
| **依赖仿真器训练** | 当前模型在仿真环境中训练，需进一步验证在真实线上系统的迁移能力 |
| **未建模平台间博弈** | 假设平台独立决策，未考虑战略性竞争或补贴战等经济行为 |
| **静态网格划分** | 六边形网格大小需手动设定，动态自适应分区可能进一步优化性能 |

---

### 🔮 **未来工作方向**

1. **引入跨平台协调机制（在保护隐私前提下）**  
   - 如联邦学习或多平台联合 reward 设计；
2. **结合 demand forecasting 与 proactive dispatch**  
   - 提前预判高峰区域，主动调配运力；
3. **扩展至 multi-modal logistics**  
   - 支持电动车换电、仓库前置、无人机配送等混合模式；
4. **开放仿真平台开源**  
   - 作者已公开代码：[https://github.com/yaofengming1999/polo-courier.git](https://github.com/yaofengming1999/polo-courier.git)，有望推动社区发展标准化 benchmark。

---

> **总结一句话**：  
> POLO 是首个明确面向**多平台、部分可观测**即时配送场景设计的 MARL 框架，通过**本地化学习 + 注意力聚合 + 反事实奖励**，实现了高收益、高效率、强鲁棒的调度决策，在迈向真实城市物流智能调度的路上迈出关键一步。

</details>

---

### 6. [CHORUS: Complementary Experts for High-Coverage Testbench Stimulus Generation](https://arxiv.org/abs/2608.10090)

**Authors**: Hejia Zhang, Sheng Lu, Zhongming Yu, Chia-Tung Ho, Brucek Khailany, Jishen Zhao  
**Category**: cs.AI  
**Published**: 2026-08-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.10090v1  

#### Abstract
Large language models (LLMs) have advanced code generation, where executable feedback provides a more reliable learning signal than textual imitation alone. Hardware verification is an important application of code generation and accounts for a substantial fraction of modern chip design effort, with...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CHORUS: Complementary Experts for High-Coverage Testbench Stimulus Generation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
硬件验证是现代芯片设计中耗时且关键的一环，其中**高覆盖率测试平台激励生成**（high-coverage testbench stimulus generation）是一项核心任务。该任务要求模型生成能够最大化仿真覆盖率的可执行测试程序。尽管大语言模型（LLMs）在代码生成方面取得进展，但在这一特定工程任务上，仅靠扩大模型规模（scale）效果有限。

如图1所示，即使是671B参数的前沿模型 DeepSeek-R1 在 CVDP-ECov 上也远未达到最优性能，表明**单纯依赖模型规模无法解决此问题**。

### 提出的新方法与思路
本文提出 **CHORUS**，一种新的**后训练框架**（post-training framework），其核心思想是：

- 利用**阶段性监督微调**（staged SFT）产生的多个中间检查点作为初始点；
- 对每个检查点独立应用相同的**基于执行反馈的强化学习**（execution-guided RL）；
- 得到一组性能相当但擅长不同子任务的**互补专家**（complementary experts）；
- 最终通过**权重合并**或**自适应多教师在线策略蒸馏**（adaptive multi-teacher on-policy distillation, OPD）将这些专家整合为一个更强的单一模型。

#### 核心创新点：
1. **发现并利用 SFT 阶段的隐含多样性**：传统流程只保留最终SFT模型进行RL，而CHORUS认为中间SFT检查点虽性能不一，但经过相同RL训练后会收敛为行为多样、能力互补的专家。
2. **提出 Adaptive Multi-Teacher OPD**：
   - 动态路由机制：根据当前任务上的执行奖励选择最佳教师；
   - 奖励门控机制（reward-gated）：只有当教师优于学生时才进行蒸馏，否则跳过更新，避免“拖累”学生；
   - 支持持续训练，灵活融合专家知识。

### 相比现有方法的优势
| 方面 | 优势说明 |
|------|--------|
| **效率 vs 规模** | 以仅 **4B 参数**的小模型超越 671B 的通用/专用大模型，显著降低部署成本。 |
| **性能上限突破** | 打破单模型 RL 的性能饱和瓶颈（saturation cap），实现更高 Pass@1。 |
| **方法通用性启发** | 展示了如何从标准SFT+RL流程中挖掘隐藏潜力，为其他领域提供新范式。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 |
|-------|------|
| **CVDP-ECov** | 主要基准，包含83个来自 CVDP 套件的硬件仓库，每项有人工设定的覆盖率阈值。目标是生成能跨越该阈值的 testbench。 |
| **AutoEval-ECov** | 辅助基准，156个源自 VerilogEval 的任务，采用更严格的 **100% 覆盖率**要求。 |

### 实验设置
- **模型架构**：所有模型均为 **4B 参数**的 Qwen3 架构。
- **初始化来源**：使用 LLM4Cov (Zhang et al., 2026) 提供的三个阶段化 SFT 检查点（Stage-0, Stage-1, Stage-2）作为起点。
- **RL训练配置**：
  - 方法：**DAPO**（Decoupled Actor-Critic Policy Optimization）
  - 奖励函数：$ R(x,y) = 1 + c(x,y) $ 若运行成功，否则为 0
  - 训练步数：1000 步
  - 并行采样：Direct generation 与 Agentic refinement 联合优化
  - Refinement 策略：**worst-state prioritized**（优先改进覆盖最低的样本）

### 评估指标
遵循 LLM4Cov 协议，报告以下四个指标：
- **Pass@1 / Pass@5**：单次/五次采样中至少有一次满足覆盖率阈值的任务比例。
- **Cov@1 / Cov@5**：平均/最高覆盖率（失败视为0%）。
- **推理模式**：
  - **Agentic 模式**：允许最多3轮交互式改进（refinement）
  - **Direct Inference 模式**：一次性生成

> ✅ **主指标**：**CVDP-ECov 上的 Agentic Pass@1**

### 基线方法对比
涵盖三类主流模型：
| 类型 | 代表模型 |
|------|---------|
| **通用大模型** | DeepSeek-R1 (671B), Llama-4-Maverick (400B), Qwen2.5-72B |
| **编码专用模型** | CodeLlama, Owen-Coder 系列 |
| **硬件/验证专用模型** | LLM4Cov-Qwen3-4B, CodeV-R1, CorrectBench |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 模型 | CVDP-ECov Pass@1 (Agentic) | 模型大小 |
|------|----------------------------|----------|
| DeepSeek-R1 (671B) | 74.5% | 671B |
| LLM4Cov-Qwen3-4B (SFT best) | 69.2% | 4B |
| **CHORUS (Merge, training-free)** | **86.7%** | 4B |
| **CHORUS (Adaptive OPD, ours)** | **88.0%** | 4B |

✅ **结论**：CHORUS 以 **4B 模型**实现了 **88.0% Pass@1**，**超过 DeepSeek-R1 (671B) 13.5个百分点**，大幅领先所有基线。

### 与其他方法的对比结果
- 在 AutoEval-ECov 上，虽然 DeepSeek-R1 表现更好（因其强大的推理能力适合小模块），但在更复杂的 CVDP-ECov 上表现不佳，突显 CHORUS 在真实硬件场景下的优越性。
- 所有硬件专用模型中，CHORUS 显著领先，证明其针对领域优化的有效性。

### 消融实验结果

#### （1）SFT 初始化对 RL 结果的影响（Figure 4 & Table 5）
- 不同 SFT 阶段初始化的模型在 RL 开始时性能差异达 9 个百分点；
- 经过相同 RL 后，三者均收敛至约 **85% Pass@1**，说明**后期 SFT 阶段对最终 RL 性能提升有限**；
- 但它们在具体任务上的表现存在显著差异。

#### （2）专家是否互补？（Figure 5 & Table 2）
- 单个专家最高 Pass@1：85.8%
- **Oracle Union（任一专家成功即算成功）**：**90.8%**
- ➡️ 存在 **~5个百分点的“头寸空间”**（headroom），表明专家确实互补。

#### （3）训练免费合并能否利用多样性？（Table 2）
| 方法 | Pass@1 |
|------|--------|
| Best individual expert | 85.8% |
| Model Soup (uniform average) | **86.7%** |
| DARE-TIES / DELLA | ≤86.0% |

➡️ 简单平均即可带来增益，但高级合并方法并未稳定胜出，说明静态合并仍有局限。

#### （4）Adaptive OPD 是否有效？（Table 3）
| 方法 | Pass@1 |
|------|--------|
| RL on Best SFT | 85.3% |
| Continued pure RL | 85.1% |
| Always distill (no gate) | 86.3% |
| Reward-gated + RL fallback | 87.5% |
| **Adaptive OPD (skip if not better)** | **88.0%** |

➡️ 只有结合 **奖励门控** 和 **无优教师则跳过** 的完整机制才能达到最佳性能。

#### （5）Refinement 目标选择（Figure 6）
- **Best-state refinement**：初期上升快，但中期崩溃（drop >20 pts），最终低于 worst-state；
- **Worst-state refinement（本文采用）**：训练更稳定，最终性能更高；
➡️ 证明聚焦最难样本更能提升整体鲁棒性和性能。

---

## 4. 关键结论和发现

### 主要发现
1. **SFT 阶段不仅是通往最优模型的路径，更是产生多样化专家的源头**；
2. 经过相同 RL 训练后，不同 SFT 初始化的模型会成为**性能相近但擅长任务不同的互补专家**；
3. 这些专家的集体能力（oracle union）远超任何个体，存在可观的性能提升空间；
4. **简单的权重平均就能部分利用这种互补性**；
5. **Adaptive multi-teacher OPD 能进一步释放潜力**，通过动态路由和条件蒸馏实现更强性能；
6. **4B 小模型经精心后训练可全面超越数百亿甚至数千亿参数的大模型**。

### 方法的局限性
- 当前研究局限于**硬件测试平台生成**这一应用场景；
- 多样性的来源依赖于特定的 **SFT curriculum（LLM4Cov 的三阶段课程）**，其他课程可能不具备同样效果；
- Adaptive OPD 引入额外计算开销（需同时运行多个教师）；
- 执行反馈具有噪声，可能影响教师选择的准确性。

### 未来工作方向
- 探索其他任务中是否存在类似的“SFT→RL→互补专家”现象；
- 设计更高效的教师调度机制以降低推理延迟；
- 将 CHORUS 思路扩展到更大规模模型或更多样化的训练流程；
- 研究如何主动构造更具差异性的 SFT 路径以增强专家多样性。

---

> 📌 **一句话总结**：  
> CHORUS 通过重新审视阶段性 SFT 的价值，构建了一组经 RL 训练后的互补专家，并利用 adaptive multi-teacher OPD 成功融合其优势，在仅 4B 参数下实现了对 671B 大模型的显著超越，为高效、高性能的领域专用 LLM 后训练提供了全新范式。

</details>

---

### 7. [Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence](https://arxiv.org/abs/2608.10720)

**Authors**: Haoyu Zhang, Zhipeng Li, Xiaoying Tang, Tianshu Yu, Yiwen Guo  
**Category**: cs.AI  
**Published**: 2026-08-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.10720v1  

#### Abstract
Omni-modal dialogue models can understand multimodal inputs and synthesize spoken replies, yet their responses remain visually disembodied. We introduce \textbf{Ex-Omni-2D}, an omni-modal dialogue framework that generates a coordinated response comprising text, personalized speech, and reference-con...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **omni-modal dialogue** 模型虽然能够理解语音、文本和视觉输入，并生成自然的语音回复，但其输出仍然是“**视觉上失联的（visually dis-embodied）**”。即模型无法同步生成一个具有视觉表现力的虚拟形象（avatar）来配合对话内容。

此外，现有方法存在以下挑战：
- 视觉行为依赖于手动设计的提示词（prompt），难以与对话状态动态对齐；
- 缺乏大规模的多模态对话数据（query-text-speech-video）用于端到端训练；
- 全序列视频生成延迟高，不适合实时交互。

### 提出了什么新方法或新思路
本文提出 **Ex-Omni-2D**，一种支持**原生视觉表达**的 omni-modal 对话框架，核心创新如下：

#### （1）**Visual Thought Plan (VTP)**  
- 一种结构化的中间表示，描述场景（scene）、情绪（emotion）、运动风格（movement style）和动作细节（motion description）。
- 在 LLM 内部生成，不对外显示，作为从对话上下文到视觉行为的显式规划接口。
- 支持自回归监督学习，可解释性强。

#### （2）**Native Multi-Codebook Speech Unit 接口**
- 使用 Qwen3-TTS 的 16-codebook 音频编码空间作为共享的声学-时间接口。
- 同一组 speech units 被同时用于：
  - 解码为个性化语音（personalized speech）
  - 对齐并驱动视频帧生成（frame-aligned video conditioning）
- 实现语音与视觉的在线同步，无需等待完整波形生成。

#### （3）**Prefix-Streaming Student 架构**
- 将全序列的 **Teacher Video Generator**（基于 Wan2.1-T2V-1.3B）蒸馏为一个块因果（block-causal）的 **Streaming Student**。
- 引入 **Prefix Streaming** 机制：每个后续窗口携带前一块的最后一个干净 latent 作为前缀，减少长序列生成中的累积退化。
- 支持高效增量推理，在四步去噪下实现接近实时的端到端响应。

### 相比现有方法的优势
| 维度 | Ex-Omni-2D 优势 |
|------|----------------|
| **视觉一致性** | VTP 显式建模情感与动作，提升 avatar 表现力 |
| **模态协同性** | speech units 同时服务语音合成与视频生成，实现音画同步 |
| **训练可行性** | 不需要大规模 query-video 成对数据，路径可分离训练 |
| **部署效率** | Streaming Student 支持低延迟增量输出，适合交互场景 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
| 数据集 | 用途 | 规模 |
|--------|------|------|
| **InstructS2S-200K** | 多轮语音对话微调 | ~200K 样本 |
| **OmniCharacter** | 角色扮演对话适配 | 400 对话 |
| **SpeakerVid-5M** | 视频生成训练 | 过滤后保留约 140K 视频片段 |
| **LibriSpeech + Emilia** | ASR/TTS 接口对齐 | 800K ASR + 1M TTS |
| **CommonEval (from VoiceBench)** | 主要测试集 | 200 条语音问答 |

### 实验设置和评估指标

#### 评估基准
- **音频质量**：Production Quality (PQ), Content Usefulness (CU), Speaker Similarity (SIM)
- **视频质量**：Subject Consistency (SC), Imaging Quality (IQ), Dynamic Degree (DD)
- **音画同步**：SyncNet Confidence (Sync-C)
- **对话能力**：Fluency, Coherency, Consistency（来自 OmniCharacter 协议）
- **问答能力**：AlpacaEval, CommonEval, BBH

#### 基线方法对比
| 类型 | 方法 |
|------|------|
| **纯视频生成** | EchoMimic, StableAvatar, OmniAvatar, FantasyTalking |
| **联合音视频生成** | Universe-1, UniAVGen |
| **级联方案** | Qwen2.5-Omni + 上述渲染器 |

所有基线均使用相同上游控制器（Qwen2.5-Omni-7B）以控制变量。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1 & Table 4）

| 模型 | SC↑ | IQ↑ | DD↑ | Sync-C↑ | E2E RTF↓ | FPS↑ |
|------|-----|-----|-----|---------|----------|-------|
| **Teacher (50-step)** | 94.62 | 67.31 | 72.00 | 4.95 | 26.917 | 1.41 |
| **Student (2-step)** | 93.33 | 52.70 | 9.50 | 3.51 | 1.201 | 39.55 |
| **Student (4-step)** ✅ | 93.65 | 57.40 | 32.00 | 3.90 | 1.293 | 26.51 |
| **Student (8-step)** | 93.91 | 61.15 | 48.00 | 4.00 | 1.932 | 15.62 |

> ✅ **四步推理是质量和效率的最佳平衡点**

### 与基线方法的对比结果
- 在 **multi-turn dialogue quality** 上，Ex-Omni-2D 取得最高分：
  - Fluency: **3.812**
  - Coherency: **4.100**
  - Consistency: **3.902**
  - 平均得分：**3.283**（优于 Qwen3-8B 的 3.264）
- 在 **speech QA** 上仅次于 Qwen2.5-Omni：
  - AlpacaEval: **4.28**
  - CommonEval: **3.71**
  - BBH: **58.70**

### 消融实验结果

#### （1）VTP 与个性化语音消融（Table 5）
| 设置 | SC | IQ | DD | Sync-C |
|------|----|----|----|--------|
| 完整模型 | 94.62 | 67.31 | 72.00 | 4.95 |
| 无 VTP（固定中性） | 93.58 | 67.26 | 81.50 | 4.65 |
| 无个性化语音 | 94.76 | 67.22 | 70.50 | 4.87 |
| 两者都无 | 93.64 | 67.16 | 72.50 | 4.65 |

👉 结论：VTP 显著提升 **Sync-C 和 SC**，说明其对协调音画行为至关重要。

#### （2）语音条件接口比较（Table 6）
| 方法 | Sync-C↑ | 条件延迟↓ |
|------|--------|-----------|
| Waveform + wav2vec | 5.83 | 0.051s |
| Single-codebook units | 2.07 | 0.003s |
| **16-codebook units** ✅ | **4.95** | **0.011s** |

👉 16-codebook 在同步性和延迟之间取得最佳权衡。

#### （3）Prefix Streaming vs No-Prefix
- Prefix Streaming 将后期 chunk 的主体一致性错误下降 **21.4%**
- DINO 曲线显示：Prefix 在第 9 块之后持续优于 no-prefix，验证其缓解累积退化的能力。

---

## 4. 关键结论和发现

### 主要发现
1. **VTP 是有效的高层语义引导机制**：能将对话意图转化为可执行的视觉计划，显著改善 avatar 的情感表达与动作协调性。
2. **native multi-codebook speech units 是理想的跨模态接口**：兼顾语音保真度与视频对齐精度，避免波形重编码开销。
3. **Streaming Student 实现了高效的增量生成**：在四步去噪下达到 **E2E RTF=1.293**，首次实现实用级的 omni-modal 流式输出。
4. **路径分离训练可行且有效**：通过 VTP 和 speech units 作为契约，可在无大规模 video-response 数据的情况下完成联合训练。

### 方法的局限性
1. **语音相似度仍有提升空间**：SIM=0.417，距离理想值较远。
2. **VTP 控制非独立**：最终视频受 VTP 和 audio condition 共同影响，难以完全解耦控制。
3. **VTP 生成引入语言能力折损**：相比仅生成 response，加入 VTP 导致 CommonEval 下降 0.11，BBH 下降 2.40。
4. **端到端仍非实时**：尽管支持流式输出，但首帧语音延迟达 **2.308 秒**，视频首块 **3.142 秒**，E2E RTF > 1。

### 未来工作方向
- 分离 planner 与 response generator，减轻干扰
- 自适应平衡 Text CFG 与 Audio CFG 权重
- 改进 few-step distillation，缩小 Teacher-Student 质量差距
- 探索更轻量化的 backbone 以进一步降低延迟
- 构建更大规模的 VTP-annotated avatar video dataset

--- 

> 💡 **总结一句话**：  
> **Ex-Omni-2D 首次实现了“对话原生”的视觉表达，通过 VTP + native speech units + Prefix Streaming 的三重设计，在无需大规模成对数据的前提下，构建了一个高质量、可增量输出的 expressive omni-modal 对话系统。**

</details>

---

### 8. [ClusterBench: A Framework for Cluster-Wide Continuous Benchmarking and Regression Testing](https://arxiv.org/abs/2608.10956)

**Authors**: Aditya Ujeniya, Jan Eitzinger, Thomas Gruber, Georg Hager, Gerhard Wellein  
**Category**: cs.DC  
**Published**: 2026-08-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.10956v1  

#### Abstract
Data centers need tooling that validates an entire installation rather than individual nodes, at acceptance and at regular intervals thereafter. This requires dispatching identical benchmarks to every node in a single submission, and therefore cluster-aware scheduling. This paper presents ClusterBen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*ClusterBench: A Framework for Cluster-Wide Continuous Benchmarking and Regression Testing*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 HPC 系统的 **benchmarking 框架**（如 ReFrame、Pavilion2、AutoBench）主要用于单节点或应用级性能测试，难以满足以下需求：
- **集群级验证**：在系统验收和运行期间，需要对所有计算节点进行统一、可比较的性能测量。
- **组件级细粒度测试**：区分 CPU、GPU、内存、网络、存储等硬件组件的性能变化。
- **连续回归检测**：识别由软件更新（如内核、驱动、固件）引起的性能退化。
- **多维指标采集**：同步收集性能数据与功耗、频率、温度等硬件指标以分析变异性。

现有工具缺乏 **cluster-aware scheduling** 能力，无法在一个作业提交中调度覆盖整个集群的细粒度测试。

---

### 🚀 提出的新方法与创新点

本文提出 **ClusterBench**，一个专为 **cluster-wide continuous benchmarking** 和 **regression testing** 设计的框架，其核心创新包括：

#### （1）Cluster-Aware Scheduling 架构
- 支持从单一测试定义文件自动派生出针对：
  - 每个 compute node 的单节点测试
  - node pairs 的 interconnect 测试（如 OSU Micro-Benchmarks）
  - 单节点内每个 accelerator（如 GPU）的独立测试
- 所有任务通过一次 job submission 完成调度，确保环境一致性。

#### （2）双轴数据积累机制（Space + Time）
- **空间维度**：覆盖集群中所有节点，揭示跨节点性能分布。
- **时间维度**：定期重复执行（如每周），形成时间序列数据，用于检测性能退化趋势。

#### （3）内置 Metrics Collector
- 自动封装 benchmark 并采集运行时硬件指标：
  - Power draw（via RAPL / `nvidia-smi`）
  - Core/GPU frequency
  - Temperature
- 指标与性能数据严格对齐时间窗口，支持相关性分析。

#### （4）面向分析优化的数据存储设计
- 使用两种嵌入式数据库：
  - **SQLite**（row-oriented）用于作业状态追踪（book-keeping）
  - **DuckDB**（column-oriented）用于高效聚合查询与可视化（analytics）

#### （5）组件级 benchmark suite 集成
- 提供专用 benchmark 集合，每个 benchmark 明确绑定一个硬件组件：
  - CPU: HPL（浮点性能）、Schoenauer Triad（内存带宽）
  - GPU: DGEMM/SGEMM（计算性能）、Triad（显存带宽）
  - Interconnect: OSU Micro-Benchmarks
  - Disk: fio（I/O 性能）

---

### 🔍 相比现有方法的优势

| 特性 | ClusterBench | 典型现有框架（ReFrame/Pavilion2/AutoBench） |
|------|--------------|------------------------------------------|
| Cluster-wide coverage | ✅ 支持全集群统一调度 | ❌ 多为单节点或分区级 |
| Component-level testing | ✅ 支持 per-GPU/per-memory 测试 | ❌ 通常仅到 node 级 |
| Multi-node benchmarks | ✅ 内建 node pair 自动生成 | ❌ 需手动配置 |
| 同步 metrics 采集 | ✅ 原生集成 power/freq/temp | ⚠️ AutoBench 需外接 DCDB |
| 数据组织方式 | ✅ Columnar DB（DuckDB）适合分析 | ❌ 多用 SQLite 行存 |
| 编译阶段管理 | ❌ 不包含（简化用途） | ✅ 包含构建流程 |

> 💡 **定位差异**：ClusterBench 不是通用 benchmarking 框架，而是专注于 **production cluster health check** 和 **longitudinal regression detection** 的轻量级专用工具。

---

## 2. 核心实验方法和设置

### 🧪 实验平台与数据集
在德国 NHR@FAU 的三个生产级 HPC 集群上进行长期测试（>1年）：

| 集群 | 分区 | 主要硬件 |
|------|------|--------|
| **Fritz** | singlenode, spr2tb | Intel Ice Lake / Sapphire Rapids CPUs |
| **Alex** | a100, a40 | NVIDIA A100/A40 GPUs（风冷） |
| **Helma** | h100, cpu | NVIDIA H100 GPUs（液冷）、AMD EPYC CPUs |

共涉及数千个节点和加速器，涵盖多种微架构（Ampere, Hopper, Zen3/Zen5c, Ice Lake, Sapphire Rapids）。

---

### ⚙️ 实验设置

- **测试频率**：每两周执行一次完整 benchmark suite。
- **测试对象**：
  - 所有空闲节点（idle nodes）
  - 使用固定预编译 binary（消除 build variance）
- **运行时长**：每个 benchmark 至少运行 10 分钟，确保稳定负载。
- **指标采集**：
  - 性能指标：FLOPs/s、Bandwidth (GB/s)、Latency、IOPS
  - 硬件指标：Power (W)、Frequency (MHz)、Temperature (°C)，采样间隔 ≈1s
  - 最终指标取运行期间平均值。

---

### 📊 评估指标

| 类型 | 指标 |
|------|------|
| **性能变异性** | 节点间性能标准差、5th–95th percentile 波动范围 |
| **相关性分析** | Pearson correlation coefficient（r）<br>如 freq vs perf, temp vs perf, power vs freq |
| **稳定性特征** | 单节点内部波动（within-node variation）<br>集群整体分布形态（cross-node distribution） |

---

### 🆚 基线对比说明
本文未直接与其他 benchmarking 框架进行“性能”对比，而是强调 **功能性缺失填补**：
- 对比指出 ReFrame 等不支持 multi-node 或 per-accelerator 测试。
- 强调 ClusterBench 在 **部署便捷性**、**数据可分析性** 上的设计优势。
- 实际效果体现为其能够揭示其他工具无法捕捉的 **硬件变异性模式**。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 组件 | 变异程度 | 观察现象 |
|------|---------|--------|
| **CPU (HPL)** | 跨节点变异 ≤ 5%<br>单节点内变异 < 1% | Ice Lake 双路系统均值 ~3.8 TFLOP/s<br>峰值可达 5.5 TFLOP/s（受频率 throttling 影响） |
| **GPU (DGEMM)** | 跨 GPU 变异 < 5%<br>单 GPU 内变异 < 1% | A100 上 DGEMM 性能高度一致 |
| **GPU Memory (Triad)** | 变异 < 0.05% | 显存带宽极稳定，即使功耗波动大 |
| **Interconnect (OSU)** | 未量化全局变异 | 成功实现 node-pair 自动调度测试 |

---

### 🔗 核心相关性发现

#### （1）CPU 层面
- **频率 ↔ 性能**：强正相关（r ≈ 0.9），符合预期。
- **功耗 ↔ 频率**：负相关 → 高功耗导致更严重频率降频。
- **陷阱警告**：跨节点聚合数据可能导致 **spurious negative correlation**（见 Fig.5），掩盖真实正相关关系。

#### （2）GPU 层面（A100，风冷）
- **SM Frequency ↔ DGEMM Perf**：强正相关（r = 0.99）
- **Temperature ↔ Perf**：显著负相关（r = -0.49）→ 温度升高引发 thermal throttling
- **Memory Power Draw ↔ Perf**：负相关 → 显存功耗挤占 SM 功耗预算

#### （3）GPU 层面（H100，液冷）
- **Temperature ↔ Perf**：无明显相关性（r = -0.038）→ 散热效率高，温控稳定
- **SM Frequency ↔ Perf**：仍保持强正相关（r = 0.877）
- 表明冷却方式显著影响性能-温度关系

---

### 🔄 消融类分析（Implicit Ablation）

虽然没有传统消融实验，但通过不同视角对比得出重要结论：

| 对比维度 | 发现 |
|--------|------|
| **单节点 vs 全集群聚合** | 单节点时间序列分析更能反映真实物理规律；跨节点聚合可能产生误导性统计结论 |
| **风冷 vs 液冷节点** | 冷却设计决定温度是否成为性能瓶颈 |
| **计算密集型 vs 内存密集型代码** | 内存带宽型代码（Triad）在 GPU 上异常稳定，因未达 TDP 限制 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **硬件同质≠性能同质**  
   即使规格完全相同的节点，其性能也存在高达 **5% 的跨节点变异**，主要源于制造工艺偏差导致的频率/功耗特性差异。

2. **每个节点具有“指纹式”行为特征**  
   单个节点在时间上的性能波动极小（<1%），表现出稳定的运行签名（operating signature），可用于异常检测。

3. **metrics 相关性高度依赖聚合方式**  
   - 在个体层面：频率↑ ⇒ 性能↑（合理）
   - 在集群层面简单聚合：可能出现频率↓ ⇒ 性能↑ 的伪逆相关
   - ➜ 必须结合 **time + space** 多维视图分析

4. **冷却方式决定温度影响路径**  
   - 风冷节点：温度是性能的关键制约因素
   - 液冷节点：温度控制良好，不再是瓶颈

5. **GPU 内存子系统极为稳健**  
   Schoenauer Triad 在各类 GPU 上变异均低于 0.05%，表明现代 GPU 显存控制器高度优化且留有足够功率余量。

---

### ⚠️ 方法的局限性（作者自述）

| 局限 | 说明 |
|------|------|
| **无编译阶段支持** | 用户需自行提供预编译 binary，不适合 CI/CD 场景中的 build-test 流程 |
| **不支持参数扫描** | 固定配置运行，无法用于 benchmark 参数调优或性能建模 |
| **仅支持 Slurm 调度器** | 当前仅适配最主流的 Slurm，PBS/Flux/HTCondor 尚未支持 |
| **非端到端测试框架** | 更像是“插件”，可与其他框架（如 ReFrame）组合使用 |

---

### 🔮 未来工作方向

1. **扩展调度器后端**：增加对 PBS、Flux、HTCondor 的支持。
2. **引入参数探索能力**：用于 characterization 而不仅是 health check。
3. **确定最优监测频率**：研究 daily/weekly/monthly 健康检查的性价比。
4. **开发智能退化检测算法**：
   - 利用长期积累的时空数据训练 ML 模型
   - 自动识别 gradual degradation 或 early failure signs
5. **增强与 CI/CD 集成能力**：联合 ReFrame 或 AutoBench 构建 end-to-end regression pipeline。

---

## 🏁 总结

**ClusterBench** 是首个专为 **集群级持续健康监测** 设计的 benchmarking 框架。它通过 **cluster-aware scheduling + component-specific benchmarks + synchronized metrics collection + columnar analytics backend** 的组合，在真实生产环境中揭示了硬件性能的细微变异性，并证明了这些数据对于理解系统健康、检测回归、研究硬件退化具有极高价值。

> 🔬 **本质贡献**：将 HPC 系统视为一个 **动态演化实体**，而非静态机器，推动了从“一次性验收”向“全生命周期性能健康管理”的范式转变。

</details>

---

### 9. [SCOUT: Symmetric Consensus Outlier Detection for Failure Localization in LLM Pre-Training](https://arxiv.org/abs/2608.11034)

**Authors**: Zhuang Wang  
**Category**: cs.DC  
**Published**: 2026-08-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.11034v1  

#### Abstract
In LLM pre-training, synchronization propagates rank-local stalls, slowdowns, and numerical errors into job-wide symptoms, obscuring their origin. Existing diagnosis often relies on in-process monitors that cannot report after the trainer blocks or terminates, or on post-mortem logs that preserve on...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SCOUT: Symmetric Consensus Outlier Detection for Failure Localization in LLM Pre-Training

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在大规模 **LLM pre-training** 中，由于训练是 **synchronous** 的，单个 rank 的故障（如 hang、straggler、silent data corruption, SDC）会通过同步机制传播为整个 job 的症状，导致故障根源难以定位（failure localization）。现有诊断工具存在以下不足：
- **in-process monitors** 在训练阻塞或终止后无法报告；
- **post-mortem logs** 只记录同步后的症状，丢失原始差异；
- **offline health tests** 运行环境与真实训练不同，可能遗漏由特定 workload 触发的缺陷。

这类故障被称为 **latent failures**，即系统检测到异常但无法直接确定应采取恢复动作的具体组件。

### 提出了什么新方法或新思路
提出 **SCOUT** —— 一种基于 **对称共识的异常检测框架**（Symmetric Consensus OUTlier detection），其核心设计原则是：

> **通过等价副本之间的严格多数共识（strict-majority consensus）识别异常行为的 rank。**

具体实现包括：
- **Consensus Collective Communication (C3)** 抽象：在等价 peer group 内进行 AllGather 并执行一致性判断，返回哪些 rank 的证据偏离多数派。
- **In-situ replay**：在 live job 旁路重放关键计算路径，保留真实的 model state、kernel、memory pressure 和通信负载，用于生成可比较的数值签名和执行时间。
- **Out-of-band (OOB) CPU observer**：独立于训练进程运行，持续读取共享内存中的进度指纹，在训练挂起时仍能完成诊断。
- **Clean replay coverage 用于 checkpoint 认证**：只有经过验证的 replay 覆盖后，才允许将最新 checkpoint 视为“可信”以用于恢复，防止从已被 SDC 污染的状态重启。

### 相比现有方法的优势
| 维度 | 现有方法局限 | SCOUT 改进 |
|------|---------------|-----------|
| **可观测性来源** | 依赖日志、trace 或离线测试 | 利用训练中天然存在的 **equivalent replicas** 作为实时参考 |
| **诊断时机** | 多为事后分析或依赖进程内监控 | 实现 **runtime online diagnosis**，无需等待 job 结束 |
| **运行条件保真度** | 离线测试脱离真实压力 | **in-situ replay** 完全复现生产级 workload 条件 |
| **SDC 恢复安全性** | 仅靠 byte checksum 无法保证数值正确性 | 引入 **replay-based numerical trust**，确保恢复点未被污染 |
| **集成复杂性** | 需修改训练循环或框架源码 | 通过 public API 接入，**无需改动 training loop 或 framework source** |

---

## 2. 核心实验方法和设置

### 实验设置
- **硬件平台**：双节点测试床，每节点 8× NVIDIA A100-SXM4-40GB GPU，共 16 个 rank。
- **互联方式**：节点内通过 NVLink，节点间使用 TCP（无 RDMA）。
- **软件栈**：
  - PyTorch 2.13.0 + CUDA 13.0
  - NCCL 2.29.7, Gloo
  - 支持框架：**PyTorch**, **TorchTitan**, **Megatron-Core**, **DeepSpeed (ZeRO/FSDP)**

### 工作负载
- 训练一个确定性的三块 Transformer 模型（AdamW optimizer）
- 并行策略覆盖：
  - **DDP**（Data Parallelism）
  - **FSDP2**
  - **HSDP mesh (4×4)**
- 同时验证 MoE 层（DeepSpeed/Megatron MoE layers）下的路由、AllToAll 通信和专家计算。

### 故障注入模型（Fault Injection）
| 故障类型 | 注入方式 |
|--------|---------|
| **Dense SDC** | 在 backward 后 corrupt 参数 / 输出 / 梯度 |
| **Dense numerical SDC** | 对前向输出或参数添加扰动矩阵（含 64 种极轻微变化） |
| **Compute straggler** | 延迟采样层 forward 执行 250ms |
| **Communication straggler** | 延迟 tensor-parallel collective |
| **MoE SDC** | 持续 corrupt 特定 rank 的 expert weights |
| **Hang & input stall** | 冻结进程、发送不兼容 collective 元数据、延迟数据加载 |

### 评估指标
- **Localization accuracy**：是否准确定位到出错 rank 或 peer group
- **False positive rate**：健康状态下是否误报
- **Checkpoint recovery correctness**：恢复后状态是否 bitwise 正确
- **Replay coverage compression ratio**：MoE 场景下形状压缩效率
- **Overhead estimation**：replay 开销占训练迭代的比例

### 基线方法对比
本文未直接与端到端系统进行性能对比，而是强调其提供的 **诊断信号质量** 是现有系统所缺乏的补充信息。相关工作如：
- **Minder/Aegis**：基于 telemetry 的故障机器检测
- **Flight Recorder/MyCROFT**：NCCL 内部状态追踪
- **SDCHunter/SuperBench**：SDC 探测工具
- **GEMINI/Bamboo**：快速 checkpoint 恢复机制

SCOUT 的优势在于提供 **rank-level 定位能力 + 数值信任认证**，可与上述系统协同工作。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 故障定位准确率（Localization Accuracy）
| 故障类型 | 实验次数 | 成功定位率 | 备注 |
|--------|--------|------------|------|
| **Dense SDC** | 3/3 cells | 100% | 准确定位 corrupt rank，并排除污染边界 |
| **Dense numerical SDC** | 344 cases | 344/344 localized | 包括 64 种 near-invisible 变化 |
| **Compute straggler** | 3/3 cells | 100% | 经两次确认后定位，false positive < 1.6ms |
| **Communication straggler** | – | 成功定位受影响 PG | 如 `{r14, r15}`，无误报 SDC |
| **MoE SDC** | 多 shape 测试 | 100% | 在 128-row 和 512-row 输入下均定位到 rank 15 |
| **Hang & input stall** | 120+30 次启动 | 全部通过 | 正确区分 control-flow divergence vs. runtime stall |

#### ✅ Checkpoint 恢复正确性
- 所有测试中，当选择 `verified checkpoint` 时，fresh runtime 能 **bitwise 恢复 model、optimizer、RNG state**。
- 注入 SDC 后的 checkpoint 被成功排除，回退至前一个 clean 状态。
- 示例：两步 replay cycle 测试中，step 2 成为 candidate，step 4 清洁后升级为 verified；step 5 注入 SDC，恢复时正确选用 step 2。

#### ✅ MoE Replay Coverage Compression
利用 **execution path coverage** 压缩搜索空间，显著减少需 replay 的形状数量：

| 配置 | Admission Shapes | Selected Representatives | 压缩率 |
|------|------------------|----------------------------|-------|
| 128×128 单专家 | 3,457 | 16 | **99.54%** |
| 6144×10752 | 2,048 | 10 | 99.51% |
| 分组 GEMM (2 exp/kernel) | 2,048 | 18 | 99.12% |
| 分组 GEMM (16 exp/kernel) | 2,048 | 48 | 97.66% |

> ⚠️ 压缩前提是相同 kernel scheduling config；异构 routing 需单独模板。

#### ✅ 性能开销估算
- 对于 N 层重复结构，每次 replay 一层 forward/backward ≈ $ \frac{1}{N} $ 迭代时间。
- 若 replay V=3 个变体，每 I=20 次迭代一次，N=50 层，则平均开销：
  $$
  \text{Overhead} = \frac{V}{I \times N} = \frac{3}{20 \times 50} = 0.3\%
  $$

---

## 4. 关键结论和发现

### 主要发现
1. **等价副本间的 majority consensus 是有效的在线诊断基础**：
   - 利用 LLM 训练中普遍存在的 **data parallelism** 构建自然冗余，无需额外资源即可获得 fault-free 行为参考。

2. **in-situ replay + OOB observer 实现高保真诊断**：
   - 保持 live job 的 model state、memory pressure、thermal condition，使诊断结果更具代表性。
   - OOB observer 确保即使 trainer hang 也能完成进度比对。

3. **clean replay coverage 可作为 checkpoint 数值信任依据**：
   - 将 replay 覆盖范围与 checkpoint eligibility 绑定，防止从已污染状态恢复。

4. **cross-PG validation 可进一步定位到物理机器**：
   - 利用多个并行维度（如 TP、FSDP）的 slow group 交集，缩小故障设备范围至单台主机。

5. **MoE 动态 shape 可通过 execution path 压缩 replay 空间**：
   - 显著降低 replay 开销，同时保障覆盖率。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖 sufficient peer group size** | 至少需要 3 个等价副本才能形成 strict majority；若仅有两个副本只能报告 group-level stall |
| **MoE catalog 需预先构建且环境敏感** | 必须绑定特定 GPU 类型、软件栈、kernel 实现；任何变更需重新 profiling |
| **无法定位低于 rank 的硬件单元** | 如具体 kernel instruction、NIC port、switch 等仍需外部 telemetry 支持 |
| **当前评估规模较小** | 仅在 16 GPU 上验证，尚未在千卡以上集群测量 false positive rate 与 end-to-end recovery 时间 |
| **部分 collectives 不可见** | FSDP/Dynamo 内部的 collective 可能未暴露给 SCOUT，影响诊断粒度 |

### 未来工作方向
- 扩展至更大规模集群（multi-rack, RDMA fabric）验证诊断准确性与时效性。
- 实现 **paired in-situ and offline diagnosis** 联合分析。
- 引入 **temporal confirmation** 机制增强稳定性。
- 支持更多编译器后端（如 Dynamo、TorchInductor）的 launch boundary 适配。
- 探索 **adaptive replay scheduling**，根据历史异常动态调整频率与目标层。
- 与 fabric-level 诊断工具（如 NetBouncer）集成，实现跨层联合定位。

---

> 🔗 **开源地址**：https://github.com/LMResiliency/lm-resiliency  
> 📚 **集成支持**：PyTorch, TorchTitan, Megatron-Core, DeepSpeed, GEMINI

</details>

---

### 10. [GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care](https://arxiv.org/abs/2608.10969)

**Authors**: Ruirui Wang, Yanke Li, Manuel G\"unther, Diego Paez-Granados  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.10969v1  

#### Abstract
Healthcare data, such as Intensive Care Unit (ICU) records, comprise heterogeneous multivariate time series sampled at irregular intervals with pervasive missingness. However, clinical applications demand predictive models that are both accurate and interpretable. We present our Graph Attention-base...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care

---

## 1. 论文的主要贡献和创新点

### 解决的问题
重症监护病房（ICU）中的临床时间序列数据具有以下挑战：
- **不规则采样**（irregular sampling）：观测时间间隔不一致。
- **高缺失率**（pervasive missingness）：大量信号在多数时间点未被记录。
- **异质性**（heterogeneity）：不同生理信号（如血压、白细胞计数）具有不同的统计特性和动态行为。
- **可解释性需求**（interpretability）：医疗决策要求模型具备透明、可信的推理过程。

现有方法通常在处理上述问题时存在不足：
- 传统RNN或Transformer对不规则采样敏感；
- GRU-D等模型虽能建模衰减机制，但忽略**跨信号依赖关系**；
- 可解释模型（如RETAIN）牺牲性能换取解释能力，且难以应对不规则输入。

---

### 提出的新方法：GARLIC
作者提出 **GARLIC**（Graph Attention-based Relational Learning for Intensive Care），一种端到端可训练的图注意力框架，用于建模ICU多变量时间序列。

#### 核心创新点：
1. **Decay-based Latent Feature Encoding**
   - 使用**指数衰减机制**进行learnable imputation，保留时间感知特征。
   - 引入**missingness indicator**，使模型能区分真实观测值与插补值。

2. **Time-Lagged Graph Message Passing**
   - 构建**时滞摘要图**（time-lagged summary graph），捕捉不同生理信号之间的动态依赖关系。
   - 图结构由可学习邻接矩阵 $ W_T \in \mathbb{R}^{K\times K} $ 定义，支持稀疏正则化（$ \ell_1 $）以增强可解释性。

3. **Cross-Dimensional Sequential Attention**
   - 在时间步内先进行**信号级注意力**（SignalAttn），再通过GRU建模局部连续性，最后使用**时间自注意力**（TemporalAttn）捕获长程依赖。
   - 注意力权重直接作为**内置解释机制**，提供observation-level、signal-level和edge-level的重要性评分。

4. **Alternating Decoupled Optimization**
   - 设计交替解耦优化策略，分阶段训练重建模块与分类器，缓解多任务目标冲突（reconstruction vs. classification）。

---

### 相比现有方法的优势
| 方面 | GARLIC优势 |
|------|------------|
| **准确性** | 在多个ICU基准上达到SOTA，显著优于GRU-D、ODE-RNN、mTAND等 |
| **可解释性** | 所有注意力与图边均为end-to-end学习，无需后处理解释器（如SHAP、Integrated Gradients） |
| **鲁棒性** | 显式建模跨信号依赖，在高缺失率下仍保持稳定性能 |
| **通用性** | 不仅适用于ICU预测，还在人类活动识别、非医疗时间序列任务中表现优异 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 | 任务 | 样本数 | 信号数 | 缺失率 |
|-------|------|------|--------|--------|--------|
| **MIMIC-III** | 大型公开ICU数据库 | 院内死亡率预测 | 49,380 | 103 | ~98.08% |
| **PhysioNet 2012 (P12)** | 多中心ICU记录 | 死亡率预测 | 11,988 | 36 | ~94.80% |
| **PhysioNet 2019 (P19)** | 聚焦脓毒症早期预警 | 脓毒症发生预测 | 40,331 | 34 | ~85.83% |

> 数据预处理遵循标准流程，包括归一化、滑动窗口切片、标签对齐等。

---

### 实验设置与评估指标
- **划分方式**：8:1:1 分为训练/验证/测试集，分层抽样保证类别平衡。
- **超参数调优**：基于验证集进行网格搜索。
- **评估指标**：
  - **AUROC**（Area Under ROC Curve）
  - **AUPRC**（Area Under Precision-Recall Curve）——尤其关注类别不平衡下的性能
- **重复性**：所有结果报告为五次随机种子运行的均值±标准差。
- **硬件环境**：单块NVIDIA RTX 3090 GPU，批大小统一配置确保公平比较。

---

### 基线方法对比
分为两类进行比较：

#### （1）面向不规则时间序列的方法
- RNN-Mean / RNN-Decay / RNN-Δt
- GRU-D
- ODE-RNN / L-ODE-RNN / L-ODE-ODE
- mTAND
- Warpformer / MTSFormer
- MTGNN / RAINDROP

#### （2）可解释性模型
- RETAIN
- IMV-LSTM
- DARNN

> 所有基线均采用原文推荐或经调参后的最优配置。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 模型 | P12 AUROC ↑ | P12 AUPRC ↑ | P19 AUROC ↑ | P19 AUPRC ↑ | MIMIC-III AUROC ↑ | MIMIC-III AUPRC ↑ |
|------|-------------|-------------|-------------|-------------|-------------------|--------------------|
| **GARLIC (Ours)** | **86.40±0.86** | **56.89±1.75** | **90.96±0.84** | **55.29±2.45** | **90.09±0.45** | **64.85±1.68** |
| 第二名（Warpformer） | 84.88 | 50.62 | 89.95 | 54.10 | 89.17 | 61.52 |

> ✅ GARLIC在**所有三个数据集的所有指标上均取得SOTA**，提升显著。

---

### 与基线方法的对比结果
- 在P12上，相比最佳基线Warpformer，AUROC提升**+1.52%**，AUPRC提升**+6.27%**。
- 在MIMIC-III上，AUPRC达到**64.85%**，远超第二名（61.52%），说明其在高度不平衡场景下更具判别力。
- 相比可解释模型（如RETAIN、IMV-LSTM），GARLIC不仅性能更高，而且解释质量更可靠（见下文消融实验）。

---

### 消融实验结果（Table 11）

| 消融变体 | P12 AUROC | Δ↓ | P19 AUROC | Δ↓ | 结论 |
|----------|-----------|-----|-----------|-----|------|
| 完整GARLIC | 86.40 | — | 90.96 | — | — |
| w/o Missingness Indicator | 84.87 | -1.53 | 82.22 | -8.74 | 缺失标识符对高缺失率至关重要 |
| w/o Decay Mechanism | 82.20 | -4.20 | 89.50 | -1.46 | 指数衰减机制有效建模信号动态 |
| w/o Signal-wise Encoder | 85.00 | -1.40 | 89.78 | -1.18 | 专用MLP编码器提升异质信号建模能力 |
| w/o Graph Message Passing | 84.67 | -1.73 | 88.39 | -2.57 | 跨信号依赖建模显著影响性能 |
| w/o GRU | 76.90 | -9.50 | 75.26 | -15.70 | GRU对局部时间连续性建模不可或缺 |
| w/o Alternating Optimization | 85.32 | -1.08 | 90.08 | -0.88 | 解耦训练策略提升稳定性与最终性能 |

> 🔍 消融实验证明每个模块都对整体性能有实质性贡献，尤其是**GRU**和**图消息传递**。

---

## 4. 关键结论和发现

### 主要发现
1. **GARLIC是首个将图结构学习、时间滞后依赖建模与交叉维度注意力结合的ICU时间序列模型**，实现了准确性和可解释性的统一。
2. 所有注意力权重和图边均可作为**内置解释机制**，无需额外解释工具。
3. **交替解耦优化策略**有效缓解了重建与分类任务间的梯度干扰，提升了训练稳定性。
4. **案例研究表明**，GARLIC能够识别临床上有意义的风险模式（如同步血流动力学不稳定、代谢性酸中毒等），并给出符合医学直觉的解释。
5. **扰动实验（ROAR-style）验证了解释的保真度**：
   - 保留Top 50%重要特征时性能下降最小；
   - Page’s L test 和 TOST 测试显示所有p值 < 0.005，表明解释具有一致性和充分性。

---

### 方法的局限性
1. **尚未支持长期预测或连续监测**：当前为固定长度输入，不适用于无限流式ICU住院。
2. **未整合静态患者特征**（如年龄、性别、合并症），可能限制个性化建模能力。
3. **图结构学习存在非唯一性**：由于生理系统冗余，不同训练种子学到的图略有差异（需聚合分析）。
4. **未解决严重类别不平衡问题**：如脓毒症发生率仅约7%，未来可引入重加权或过采样策略。

---

### 未来工作方向
1. 探索**自适应窗口机制**或事件驱动建模，替代固定时滞设计。
2. 将静态特征与动态图融合，构建统一的**多尺度图表示框架**。
3. 引入**临床先验知识**（如器官系统连接关系）约束图学习，提高可解释性与稳定性。
4. 开展真实世界部署研究，结合医生反馈迭代优化模型。
5. 扩展至更多下游任务，如**疾病进展预测、治疗响应建模、反事实推理**等。

---

> ✅ **总体评价**：  
> GARLIC是一项兼具高性能与高可解释性的突破性工作，推动了深度学习在ICU风险预测中的可信应用。它不仅在技术架构上有创新，在实验验证和解释性量化方面也树立了新标准，具有广泛的临床转化潜力。

</details>

---

### 11. [Closed-Loop LLM Co-Pilots for Digital Agriculture](https://arxiv.org/abs/2608.09949)

**Authors**: Serge Kernbach  
**Category**: cs.AI  
**Published**: 2026-08-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.09949v1  

#### Abstract
This study evaluates the application of Large Language Models (LLMs) in complex biological systems, evolving from data analysis to autonomous, AI-guided experimentation. The framework is driven by data from a 49-channel phytosensor network, encompassing multispectral, electrochemical, and dielectric...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Closed-Loop LLM Co-Pilots for Digital Agriculture》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **“数据丰富、信息贫乏”（data rich, information poor）困境**：现代农业系统（如垂直农场、自动化表型平台）产生大量多通道、高频率的生物传感时间序列数据，但人工难以从中提取有效生理洞察。
- **复杂系统的最优控制难题**：植物生理响应具有非线性、时变性和随机性，传统基于规则或固定周期的控制策略无法动态适应。
- **专家依赖性强、部署成本高**：现有系统需要农业专家进行手动建模与干预，限制了大规模推广。

### **提出了什么新方法或新思路**
- **提出“LLM 作为闭环共驾员”（LLM as Autonomous Co-Pilot）框架**：
  - 将 Large Language Model (LLM) 集成到 **Cyber-Physical System (CPS)** 中，构建从实时生物传感（biosensing）到自主执行（autonomous action）的完整闭环控制系统。
  - LLM 不仅用于数据分析与自然语言解释，更直接生成硬件控制指令（JSON command），驱动 phyto-actuators（如LED光照、灌溉系统）。
- **双智能体架构（Worker Agent + Scout Agent）**：
  - **Scout Agent (SA)**：长期历史分析，识别系统级异常（如水势失衡、光胁迫）。
  - **Worker Agent (WA)**：短期实时决策，基于SA提示执行微循环控制（每2小时重评估）。
- **实现从“人在环路”到“全自主控制”的跃迁**：系统可在无持续人类干预下运行，并在收敛后固化控制策略供复用。

### **相比现有方法的优势**
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 控制逻辑 | 固定周期（如12/12光周期） | 动态优化，基于生理反馈调整 |
| 决策依据 | 专家经验或简单阈值 | 多模态数据融合 + LLM 跨域认知合成 |
| 可访问性 | 仅限专业人士解读 | 自然语言输出，支持非专家理解 |
| 能效与生长平衡 | 单目标优化 | 多目标协同优化（生物量、能耗、时间） |
| 自主性 | 半自动（需人工调参） | 全闭环自主运行 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **真实世界生物传感网络数据流**：
  - 来源于一个 **49通道 phytosensor 网络**，覆盖电化学（EIS）、介电、光学多光谱等模态。
  - 主要监测参数包括：
    - 生物阻抗（stem bio-impedance）
    - 气孔调节（stomatal regulation）
    - 叶绿素含量（chlorophyll content via NDVI/PRI/Vigor Index）
    - 水分动态（water uptake, transpiration rate）
    - 温湿度、光照强度（PAR, full-spectrum）
- **实验对象**：
  - **垂直农场场景**：种植小麦草（*Triticum aestivum*）和豌豆苗（*Pisum sativum*），用于高密度生产验证。
  - **单株植物场景**：龙血树（*Dracaena*）、番茄（*Solanum lycopersicum*）、辣椒（*Capsicum annuum*），用于精细生理机制研究。

### **实验设置和评估指标**
#### **控制架构层级（图2所示）**
1. **Layer 1：物理平台与环境**  
   - 包括传感器阵列、可编程LED灯（450nm蓝光、660nm红光、远红、UV）、灌溉单元、EM刺激器。
2. **Layer 2：边缘计算与特征提取**  
   - 实时处理原始信号，计算合成参数（如ΔZ梯度、Rt根冠相关性、WUE水利用效率）。
3. **Layer 3：LLM Agent 层（WA + SA）**
   - 输入：结构化生理状态向量（来自Table I）
   - 输出：JSON格式控制命令（如`"wl1": "ON"`）
   - 安全校验：本地Python脚本对LLM输出进行语法与操作合理性验证。

#### **评估模式与目标函数**
| 模式 | 目标函数 | 描述 |
|------|----------|------|
| **Minimal-Time Mode** | $\max \frac{d(\text{FSB})}{dt}$ | 最大化鲜生物量增长速率，缩短生产周期 |
| **Energy-Optimization Mode** | $\max \frac{\Delta \text{FSB}}{\text{kWh}}$ | 最大化单位能耗下的生物量产出 |
| **Ultra-Minimum Strategy** | 自主探索极低能耗路径 | LLM自发发现暗诱导叶绿素积累策略 |

#### **基线方法对比**
- **Periodic Control Baseline**：
  - 12/12 光照周期
  - 全光谱照明 37.5 W/m²
  - 补充红光恒开（6.25 W/m²）
- 对比项：LLM驱动的动态控制在相同条件下表现。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | 周期性控制 | Minimal-Time | Energy-Optimized | Ultra-Minimum |
|------|------------|--------------|------------------|---------------|
| **生长周期（天）** | 6.5 ± 0.5 | **4.25 ± 0.25** (-35%) | 5.0 ± 0.25 (+11–25%) | 5.25 ± 0.25 |
| **总能耗（kWh/m²）** | 3.04 ± 0.2 | 4.49 ± 0.01 (+48%) | **2.49 ± 0.12 (-18%)** | **0.8 ± 0.04 (-73.6%)** |
| **能效提升** | — | — | +18% vs baseline | **+67.9% vs energy-optimized** |
| **收获状态达成时间** | 第6天 | **第4天** | 第5天 | 第5–5.5天 |

> 注：数据来源于Table IV及正文描述。

### **与基线方法的对比结果**
- **Minimal-Time 模式**：
  - 缩短生产周期 **35%**（6.5天 → 4.25天）
  - 实现**超线性生长动力学**（superlinear growth），早于传统线性模型预测达到收获标准。
- **Energy-Optimized 模式**：
  - 能耗降低 **18%**，仅延长栽培时间约11–25%。
  - 利用**生理惯性**（physiological inertia）实施间歇光照，在不影响生物量积累前提下节能。
- **Ultra-Minimum 策略（AI自发现）**：
  - 开发出“**暗诱导叶绿素积累**（dark-induced chlorophyll accumulation）”策略。
  - 在几乎关闭LED的情况下维持生长，**额外节省67.9%能源**（相较energy-optimized模式）。
  - 生物质颜色更浅但NDVI更高，表明内部光捕获效率增强（shade-acclimation效应）。

### **消融实验与验证**
- **Agent 架构有效性验证**：
  - 单WA模型易出现上下文稀释（context dilution），导致决策漂移。
  - 引入SA后，WA可聚焦异常事件，提升控制稳定性。
- **不同LLM模型对比（Table VI）**：
  - 高复杂度模型（Gemini、GPT-3.5、Claude Sonnet）能准确识别主要生理异常（如4月28–29日茎部液压崩溃）。
  - 低复杂度本地模型需多次提示才能逼近结果，部分失败。
- **安全机制验证**：
  - 三重防护防止LLM幻觉（hallucination）：
    1. **Contextual Grounding**：限制输入为实时传感器矩阵。
    2. **Syntactic Constraints**：强制输出符合预定义JSON schema。
    3. **Operational Verification**：本地脚本回传执行日志供LLM自检。

---

## **4. 关键结论和发现**

### **主要发现**
1. **LLM 可作为真正的“自主控制器”而非仅是分析工具**：
   - 成功将LLM从“human-in-the-loop”角色升级为“closed-loop co-pilot”，实现端到端感知-推理-行动循环。
2. **AI 能发现人类未预见的生物学策略**：
   - “Ultra-Minimum”策略展示了LLM通过试错演化出类似遗传算法的行为，挖掘出**黑暗促进营养品质提升**的现象。
3. **多尺度控制架构提升鲁棒性**：
   - Scout-Worker 分工明确，分别处理长期趋势与短期响应，避免上下文过载。
4. **自然语言接口显著降低使用门槛**：
   - 同时服务于专家（提供机制解释）与非专家（通俗类比，如“植物戴上生物太阳镜”）。

### **方法的局限性**
- **可解释性挑战（Explainability Gap）**：
  - 尽管LLM给出决策理由，但其内部推理过程仍为黑箱，无法确认是否真正执行所声称的优化算法。
- **潜在幻觉风险**：
  - 若缺乏严格的上下文锚定与输出约束，LLM可能生成虚假相关性或无效控制指令。
- **泛化能力依赖训练/提示设计**：
  - 当前系统高度依赖精心设计的prompt engineering 和 domain-specific knowledge injection。
- **硬件依赖性强**：
  - 需要高精度、多模态 phytosensor 支持，目前成本较高。

### **未来工作方向**
- **构建可验证的 Explainable AI 框架**：
  - 结合 symbolic reasoning 或 neuro-symbolic 方法，提高决策透明度。
- **跨物种迁移学习**：
  - 探索LLM在不同作物间的知识迁移能力，减少重复训练。
- **去中心化部署优化**：
  - 进一步压缩本地开源LLM（如Llama3、Qwen）以实现在边缘设备完全离线运行。
- **引入强化学习闭环训练**：
  - 当前为模拟推理，未来可结合RLHF（Reinforcement Learning from Human Feedback）进行在线学习优化。
- **扩展至更大规模农业系统**：
  - 从实验室级验证迈向商业化农场集成，测试系统在噪声、干扰下的稳定性。

---

> ✅ **总结一句话**：  
> 该论文首次实现了 **LLM 驱动的全闭环数字农业控制系统**，不仅提升了作物生产的效率与可持续性，还揭示了AI在复杂生物系统中自主探索新型生存策略的能力，标志着AI从“辅助分析”走向“主动共驾”的重要里程碑。

</details>

---

### 12. [What Actually Serializes GPU LZ77 Decode: Three Decoders, Three Mechanisms, and an Encode-Time Lever That Removes the Last One](https://arxiv.org/abs/2608.10188)

**Authors**: Yakiv Shavidze  
**Category**: cs.DC  
**Published**: 2026-08-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.10188v1  

#### Abstract
The sequential part of GPU LZ77 decode is not where the field assumes it is. Across three decoder architectures on an H100 we measure that parse, not copy, holds 64-72% of device-resident decode time; that bounding back-reference chain depth - provable, and costing 0.006% in ratio - moves latency by...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*What Actually Serializes GPU LZ77 Decode*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
该论文挑战了当前 GPU 上 **LZ77 解压**领域的一个普遍假设：即解压过程中的**串行瓶颈主要来自 back-reference 依赖链（深度依赖）**。作者通过系统性测量发现，这一假设是错误的。

真实瓶颈在于 **parse 阶段**（命令解析），而非传统认为的 copy 或 chain depth。此外，论文还识别并移除了最后一个真正串行的元素。

### 🚀 提出的新方法与新思路

1. **重新定位串行瓶颈：Parse 是主导因素**
   - 实验证明，在多种解码器架构中，`parse` 占据了 **64–72% 的设备端解压时间**，远超 copy 和 chain depth 的影响。
   - 这是对领域共识的根本性修正。

2. **证明 depth cap 对延迟影响极小**
   - 使用编码时机制限制最大依赖深度（depth cap），发现在实际文件中对整体延迟的影响最多仅 **-2.8%**，且在关键延迟尖峰区域（latency spike cluster）**完全无影响**（0 of 181 blocks changed）。
   - 表明“控制依赖深度”并非有效的优化路径。

3. **揭示 self-overlapping match 可并行化**
   - 传统上认为 self-overlapping 匹配必须串行执行（如 `memmove`）。
   - 本文指出其本质是周期性填充（periodic fill），可通过模运算实现线程级并行。
   - 结合“每 token 一个 warp”的调度策略，实现 **2.75–8.42× 的 match layer 加速**，且保持 bit-perfect 正确性。

4. **移除最后一个串行元素：distance history**
   - 发现唯一真正串行的部分是用于 repeat match 的 **四条距离历史记录（four-entry distance history）**。
   - 提出由编码器主动禁用 repeat codes（发送 varint 距离代替），从而将串行命令间的依赖自由运行长度从平均 **4 条提升至 706 条（×176）**。
   - 成本仅为压缩率下降 **0.540%**，远低于 Gompresso 等方法的代价（最高达 19%）。

5. **揭示内存墙的本质：bus efficiency 极低**
   - 分析显示 median match 长度为 **7 bytes**，而 cache line 为 128B → 总线效率仅 **4.4%**。
   - 实际写入带宽比理想 coalesced 写慢 **39×**，说明即使消除所有串行性，仍受限于 memory granularity。

---

## 2. **核心实验方法和设置**

### 📊 数据集
- **主数据集**：
  - `chr1`：人类染色体 1（hg38），254 MB，15,499 个 16KB 块
- **辅助数据集**（各 200 MB）：
  - `dna`（基因组 FASTA）
  - `english`（英文文本）
  - `proteins`（蛋白质序列）
  - `enwik8`, `enwik9`, `silesia.tar`（标准压缩测试集）
- **大规模随机访问测试**：
  - 50 GB 合成归档（3,051,758 个块），用于评估 position-invariant seek 性能

### ⚙️ 实验设置
- **硬件平台**：
  - GPU: NVIDIA H100 80GB SXM
  - Host: AMD EPYC 4344P
  - CUDA 12.x
- **三种解码器架构对比分析**（见 Table I）：
  | Decoder | 特点 | 时间决定因素 |
  |--------|------|-------------|
  | `dense full-pipe` | ANS + 单次 kernel | parse |
  | `wavefront` | 按 level 分层 launch，CUDA graph 控制 | work + Nwaves × 4.5μs |
  | `v7-RA seek` | persistent threads，每 block 有 leader | token count |

> 多架构设计旨在暴露不同限制机制，增强结论普适性。

### 🎯 评估指标
- **decode throughput (GB/s)**
- **device-resident decode time breakdown**（parse vs copy）
- **bit-perfect correctness**（FNV / XXH3 校验）
- **chain depth distribution**（before/after capping）
- **bus efficiency** = logical bytes / cache-line traffic
- **random access latency**（单 block 到多 block seek）

### 🔁 基线对比
- **Gompresso [6]**：通过限制匹配范围消除 intra-warp dependency，代价高达 19% 压缩率损失。
- **zstd-3, zstd-19, brotli-9, lz4**：作为 blockwise 压缩格式基准进行 ratio 对比。
- 所有比较均统一为 **16KB 独立块约束**以公平支持 random access。

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据

| 指标 | 数值 | 来源 |
|------|------|------|
| Parse 占比（device time） | **64–72%** | Table II, Fig. 1 |
| Chain depth cap 最大收益 | **≤ -2.8% latency**（仅 wavefront 可测） | Fig. 3 |
| Depth cap 对 spike cluster 影响 | **0 blocks changed**, **0% effect** | Fig. 4, Sec IV.C |
| Period-aware self-overlap fill 加速 | **2.75–8.42× match layer speedup** | Table V, Fig. 7 |
| 移除 distance history 后 dependency-free run 中位数 | **从 4 → 706 commands (+176×)** | Fig. 8 |
| 移除 repeat match 的压缩率成本 | **-0.540% ratio**（等价 +0.543% size） | Sec VI |
| 总线效率（chr1） | **4.4%**（1.392 GB traffic for 61 MB data） | Fig. 9 |
| Coalesced 写上限速度 | **2250.3 GB/s** vs 实际 **58 GB/s** → **39× gap** | Sec VII |
| 50GB 归档随机访问延迟 | **~344 μs ±14%**, 位置无关 | Fig. 11 |
| 千倍数据量请求耗时增长 | **1000× data → 1.25× time**（良好扩展性） | Fig. 11 |

### 🔍 与基线方法对比

| 方面 | 本文方法（ACEAPEX） | Gompresso |
|------|---------------------|----------|
| 优化目标 | Parse 层 dependency（distance history） | Match 层 intra-warp dependency |
| 是否需要修改解码器 | ❌（same decoder） | ❌（same decoder） |
| 编码端代价 | **+0.543% 压缩后大小** | **up to +19%** |
| 并行潜力提升 | **×176 更长无依赖 run** | 未量化 |
| 适用性 | 绝对偏移格式（absolute offset）天然支持 block independence | 需要特殊编码策略 |

> 在相同 16KB block 约束下，ACEAPEX 在 3/5 数据集上优于 zstd-19（Table VI）。

### 🔬 消融实验结果

| 实验配置 | Token 变化 | Cluster Depth | Latency (μs) | 结论 |
|--------|-----------|--------------|---------------|-------|
| Baseline | — | 4.358 | 480.4 | — |
| A (deep forced) | -20,837 | **2.615** | **467.1** | **只有降低 cluster depth 才有效** |
| B (leaf forced) | -20,668 | 4.542 | 480.5 | ❌ 无效（placebo） |
| C (outside forced) | -20,963 | 4.358 | 480.5 | ❌ 完全无效 |

> ➤ 证明 depth cap 的效果具有高度特异性，不能泛化。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **真正的串行瓶颈是 parse，不是 copy 或 chain depth**  
   → 优化应聚焦于命令流解析效率，而非依赖图深度。

2. **back-reference depth 不是延迟的主要驱动因素**  
   → 尽管可技术上限制，但在真实 workload 中几乎不影响性能，尤其在热点区域无效。

3. **self-overlapping match 是周期性操作，可完全并行化**  
   → 利用已知 period 和 modular indexing 可打破“必须串行”的误解。

4. **最后一个串行环节是 distance history，可用编码器杠杆移除**  
   → 以极低成本（0.54% ratio）换取巨大并行空间（×176 命令 run）。

5. **根本瓶颈最终落在 memory subsystem**  
   → 即使完全去串行化，由于 **median match = 7B << 128B cache line**，总线效率仅 4.4%，存在 **39× 的 coalescing ceiling**。

6. **random access 具备良好的 scale-out 特性**  
   → 单 tile seek ~344μs，千 tile 请求仅需 0.423ms，接近常数增长。

7. **许多流行假设被实验证伪**（共 10 项，见 Sec X）
   - 如 “throughput ∝ 1/tokens”、“sorting improves coalescing”、“minimum match length helps” 等均为错误认知。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **单一硬件平台** | 所有实验基于 H100 SXM，其他 GPU 架构（如 A100, RTX）行为未知 |
| **bench noise 为 6%** | 小于该幅度的变化不作声称有效性 |
| **distance history 移除未在 GPU 验证加速** | 效果基于 CPU 模拟推断，尚未实测 GPU 上的实际提速 |
| **50GB seek 非 bit-perfect** | 原始明文不在磁盘，无法验证 FNV 正确性 |
| **部分探针非 bit-perfect** | 如 minimum match length 实验直接丢弃 token，仅用于机制演示 |

---

### 🔮 未来工作方向

1. **开发 parse-aware 并行调度器**
   - 利用 prefix-sum 可并行化的特性设计新型 parser，突破当前 kernel 启动开销。

2. **探索 memory-granularity aware packing**
   - 设计能聚合短 match 的 layout，提高 cache-line 利用率，逼近 39× 的 coalescing 上限。

3. **跨 GPU 扩展研究**
   - 在多卡环境下验证 position-invariant seek 与分布式 decode pipeline 的可行性。

4. **结合 entropy coding 层优化**
   - 当前 focus 在 LZ77 层；未来可集成 ANS/Huffman 的并行解码，形成端到端高速 pipeline。

5. **构建通用 GPU decompression profiling framework**
   - 将本文的三解码器分析法推广为标准工具链，用于诊断各类 codec 的真实瓶颈。

---

> 💡 **一句话总结**：  
> 本论文通过严谨测量揭示，GPU LZ77 解压的真实串行瓶颈是 **parse** 而非 back-reference 依赖，并提出一种编码时移除 distance history 的轻量级方法，在仅牺牲 **0.54% 压缩率**的情况下将无依赖命令 run 提升 **176 倍**，同时指出最终性能天花板在于 **memory bus efficiency**。

</details>

---

### 13. [REATS: LLM Reasoning-based Ensemble Learning for Adaptive Time Series Forecasting](https://arxiv.org/abs/2608.10149)

**Authors**: Xu Zhang, Chang Xu, Hui Sun, Nan Ma, Zijian Zhang, Peng Wang, Wei Wang, Li Zhao  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.10149v1  

#### Abstract
Due to the diversity of real-world time series, no single forecasting model consistently dominates across all samples. Ensemble learning addresses this by combining complementary model strengths, yet existing methods rely on fixed rules or black-box models based solely on numerical inputs, failing t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：REATS: LLM Reasoning-based Ensemble Learning for Adaptive Time Series Forecasting

---

## 1. 主要贡献和创新点

### 解决的问题
传统时间序列预测（TSF）模型通常依赖单一架构处理所有样本，难以适应多样化的时序模式。虽然集成学习（ensemble learning）能结合多个模型的优势，但现有方法存在以下局限：
- **静态权重**：如均匀加权或基于验证误差的固定权重，缺乏对不同样本的自适应能力。
- **黑盒动态权重**：基于神经网络的方法虽可动态分配权重，但仅依赖数值输入，无法利用文本语义理解，且缺乏可解释性。
- **泛化能力差**：当候选模型变更时需重新训练。

### 提出的新方法：REATS
本文提出 **REATS**（LLM Reasoning-based Ensemble Learning for Adaptive Time Series Forecasting），一种基于大语言模型（LLM）推理能力的智能集成路由框架，通过多模态输入与链式思维（Chain-of-Thought, CoT）推理生成可解释、样本自适应的集成权重。

#### 核心创新点
1. **混合文本-数值输入表示（Hybrid Textual-Numerical Input）**
   - 将原始时间序列转换为结构化描述，提取8类时序特征（趋势、季节性、平稳性等），以“[trend] Trend slope: -0.00454”等形式呈现。
   - 优势：固定token开销（不随序列长度增长）、激活LLM语义推理、支持规则生成CoT。

2. **规则驱动的CoT构建（Rule-based CoT Generation）**
   - 利用先验知识（oracle权重）反向工程生成自然语言推理路径，无需调用昂贵API即可获得高质量监督信号。
   - 推理流程四步走：
     1. 识别主导时序特征
     2. 分析各候选模型适用性
     3. 对比检索到的相似案例权重
     4. 给出最终权重分配理由

3. **多样化的权重监督机制（Diverse Weight Supervision）**
   - 每个样本提供 $K'=10$ 行权重（1行最优 + 9行多样化采样），增强SFT和GRPO阶段的训练信号多样性。
   - 缓解过拟合单一解、提升探索空间。

4. **高效的权重输出格式：整数百分比表（Integer Percentage Table）**
   - 权重以整数百分比形式组织成紧凑表格（如 `35,25,20,20`），降低数值幻觉风险，减少token消耗和解析复杂度。

5. **两阶段微调框架 + 改进的GRPO奖励映射**
   - **SFT阶段**：学习结构化推理与粗略权重分配。
   - **GRPO阶段**：直接优化预测MSE，引入**倒数奖励映射**（reciprocal reward mapping）：
     $$
     r = \frac{1}{1 + k\delta}
     $$
     其中 $\delta = L(w) - L(w^*)$ 是当前与最优MSE之差。
   - 优势：
     - **有界性**：防止异常值主导组内方差。
     - **非线性敏感度**：在接近最优时放大差异，在远离时保持梯度，避免早期收敛停滞。

---

## 2. 核心实验方法和设置

### 数据集
在 **8个标准基准数据集** 上进行评估，覆盖能源、金融、气象、交通等领域：
- **ETTh1/h2**, **ETTm1/m2**（电力负荷）
- **Exchange**（汇率）
- **Weather**（天气）
- **Electricity**（用电量）
- **Traffic**（道路占有率）

所有任务为单变量长期预测，输入/预测长度均为96，划分比例为 7:1:2。

### 评估指标
- **MSE**（均方误差）
- **MAE**（平均绝对误差）

### 候选模型池
分为两类：
1. **小型专用模型**（Small Specialized Models）：
   - TimeXer, LSINet, CARD, TimeMixer, ModernTCN, SEMixer, DLinear, PDF, PatchTST, MLF
2. **基础模型**（Foundation Models）：
   - MOMENT, Sundial, Timer, TIME-MOE, TimesFM, MOIRAI, TimerXL, Chronos

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **启发式** | Uniform averaging (EnSavg), Random weighting (EnSrand) |
| **误差加权** | Inverse MSE weighting (InvMSE), Optimal Fixed Weight (OptWtr/val) |
| **神经网络** | RLMC（强化学习动态组合） |
| **LLM零样本** | GPT-5.2/5.5, Codex, DeepSeek-V3.2, Grok-4（相同prompt格式） |
| **算法变体** | DAPO, DrGRPO, GSPO, SAPO（GRPO改进版） |

### 实验设置
- **LLM主干**：Qwen3-1.7B
- **RAG检索**：Top-3最相似历史样本
- **权重行数**：$K'=10$
- **倒数奖励参数**：$k=20$, $(\lambda_1,\lambda_2,\lambda_3)=(0.8,0,0.2)$
- **推理输出**：使用第一行作为最终权重

---

## 3. 主要实验结果和性能指标

### 总体性能对比（MSE）

#### (a) 基础模型候选组（Foundation Model Candidates）
| 方法 | 平均MSE |
|------|--------|
| 最佳单模型（MOIRAI） | 0.2294 |
| OptWval（最佳传统） | 0.1597 |
| Codex（零样本LLM） | 0.1594 |
| **REATS-SFT** | **0.1455** |
| **REATS-GRPO** | **0.1384** ✅ |

> **相对提升**：相比 OptWval 下降 **13.3%**

#### (b) 小型模型候选组（Small Model Candidates）
| 方法 | 平均MSE |
|------|--------|
| 最佳单模型（CARD） | 0.2111 |
| OptWtr（最佳传统） | 0.1352 |
| Codex（零样本LLM） | 0.1709 |
| **REATS-SFT** | **0.1210** |
| **REATS-GRPO** | **0.1080** ✅ |

> **相对提升**：相比 OptWtr 下降 **20.1%**

✅ **统计显著性**：在8个数据集上，REATS-GRPO均取得最低MSE，sign test显示差异显著（p < 0.05）。

---

### 泛化能力测试（Out-of-Domain Generalization）

#### 跨模型泛化（训练于小模型，测试于未见基础模型）
| 方法 | 平均MSE |
|------|--------|
| OptWval | 0.1564 |
| DeepSeek-V3.2（零样本） | 0.1626 |
| **REATS-GRPO** | **0.1442** ✅ |

> **相对提升**：下降 **7.8%~11.3%**

#### 不同候选数量下的表现（N=2,4,6,8）
- REATS在所有配置下均优于基线。
- 随着候选模型增多，其优势趋于扩大。

---

### 消融实验结果（Ablation Study）

| 变体 | 小模型组（ID）MSE | 小模型组（OOD）MSE |
|------|------------------|--------------------|
| 完整REATS-SFT | **0.1210** | **0.1719** |
| w/o CoT | 0.1315 (+8.7%) | 0.2675 (+55.6%) ❗️ |
| w/o RAG | 0.1418 (+17.2%) | 0.3470 (+102%) ❗️ |
| w/o 整数表（改用dict） | 0.1250 (+3.3%) | 0.1732 (+0.8%) |

> **结论**：CoT和RAG对泛化至关重要，尤其在面对未知模型时。

#### 多样化权重行数消融（K’）
| K’ | 平均MSE |
|----|--------|
| 1 | 0.1190 |
| 10 | **0.1080** ✅ |

> **提升10.2%**，说明多样化监督有效缓解奖励稀疏问题。

#### 输入表示对比
| 表示方式 | ID MSE | OOD MSE |
|--------|-------|--------|
| 文本-数值（Textual-Num） | 0.1210 | **0.1719** ✅ |
| 原始序列（Raw TS as text） | 0.1228 | 0.1737 |
| MLP编码器（tsenc） | 0.1210 | 0.2165 |

> **结论**：文本表示在OOD场景下泛化更强。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM可作为高性能集成路由器**：
   - REATS在多种候选模型组合下均超越传统与神经网络基线。
   - 特别是在异构性强的小模型组中，性能增益高达20%以上。

2. ✅ **规则生成CoT媲美甚至优于API生成CoT**：
   - Rule-CoT无需调用GPT即可达到同等甚至更优性能（Table 6(a)）。
   - 成本更低、可控性更强、推理一致性更高。

3. ✅ **倒数奖励映射显著优于朴素映射 $r=-\delta$**：
   - 在连续回归任务中，$r=-\delta$ 易受异常值影响导致优势压缩。
   - 倒数映射通过**有界性**与**非线性敏感度**，保留近似最优者的区分度。

4. ✅ **REATS具备强迁移与泛化能力**：
   - 可无缝迁移到未见过的候选模型（OOD generalization）。
   - 支持灵活更换模型描述而无需重新训练。

5. ✅ **可解释性是核心优势**：
   - 输出包含完整的CoT推理过程，用户可追溯为何某模型被赋予高权重。
   - 示例：识别出“强自相关”应优先使用注意力机制模型（CARD），而非周期分解模型（PDF）。

---

### 局限性
1. **依赖高质量特征提取模块**：若时序特征分析错误，可能误导LLM推理。
2. **仍需训练数据构造**：尽管免去了API调用，但仍需构建SFT数据集（含oracle权重）。
3. **推理延迟高于轻量级方法**：虽使用1.7B小模型，但仍慢于纯数值集成方法。

---

### 未来工作方向
1. **自动化特征选择与提示工程**：减少人工设计成分。
2. **在线增量更新RAG知识库**：实现持续学习。
3. **扩展至多变量与跨域预测**：探索更广泛的应用场景。
4. **结合工具调用（Tool Use）机制**：让LLM主动选择并运行特定分析函数。

---

## 总结
REATS开创性地将LLM的**多模态推理能力**引入时间序列集成学习，解决了传统方法在**自适应性、可解释性、灵活性**上的瓶颈。其实验充分验证了：
- 规则生成CoT的有效性
- 倒数奖励映射对GRPO的适配价值
- 混合输入与结构化输出带来的性能与效率平衡

该工作不仅提升了集成预测精度，更为“AI for AI”（即用LLM调度其他AI模型）提供了范例。

</details>

---

### 14. [Pair-Centric Graph Rewiring for Over-Squashing via Optimal Transport-Guided Communication Alignment](https://arxiv.org/abs/2608.10619)

**Authors**: Yan Wang, Chuan-Xian Ren  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.10619v1  

#### Abstract
Message-passing neural networks (MPNNs) often struggle when task-relevant information is distributed across distant regions of a graph, since local propagation must compress remote signals through limited structural interfaces. Graph rewiring provides a structural response to over-squashing. Most ex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Pair-Centric Graph Rewiring for Over-Squashing via Optimal Transport-Guided Communication Alignment*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **over-squashing** 问题展开研究。在 Message-Passing Neural Networks (MPNNs) 中，当任务相关的信息分布在图中相距较远的节点上时，由于信息必须通过有限的局部结构接口进行多跳传播，容易被压缩甚至丢失，导致模型性能下降。这一现象称为 *over-squashing*。

现有的图重连（graph rewiring）方法大多基于两种策略：
- **Edge-level**：利用曲率等局部几何指标识别瓶颈边；
- **Graph-level**：优化谱扩展、有效电阻等全局连通性代理指标。

然而，在有限的重连预算下，这些方法未能明确回答“**哪些节点对之间的通信最需要结构性支持？**”这一关键问题。

### 提出的新方法与新思路
本文提出 **PairAlign**，一种以**节点对为中心**（pair-centric）的图重连框架，其核心思想是将 over-squashing 视为一种**通信短缺**（communication shortage），即某些节点对的通信需求远超当前拓扑所能提供的支持。

#### 主要创新点包括：

- **Pairwise Communication Shortage Score**  
  定义了一个可计算的短缺分数：
  $$
  S(u,v;W) = \frac{w(u,v)}{s(u,v;W) + \epsilon}
  $$
  - $ w(u,v) $：原始图上的结构需求（如最短路径距离的幂次）；
  - $ s(u,v;W) $：当前重连图 $ W $ 上的有限跳数传播支持（如 $ K $ 步内平均传播质量）；
  - 该分数越高，表示这对节点越受 over-squashing 影响。

- **理论一致性**  
  证明该 shortage score 是 Jacobian-based shortage 的一个合理代理，并满足常数因子等价关系，从而赋予其理论解释力。

- **OT-Guided Rewiring Mechanism**  
  引入 **Optimal Transport (OT)** 来协调有限的边添加预算：
  - 将候选新增边视为“资源”，高短缺节点对视为“目标”；
  - 构建运输成本矩阵 $ C $，衡量某条候选边对缓解某个短缺对的有效性；
  - 通过求解 OT 问题实现全局最优分配，避免贪心策略集中在少数易修复的目标上。

- **动态更新机制**  
  在重连过程中迭代更新 shortage 分数和候选边评分，考虑了边插入带来的双重效应：
  - 新增路径可能提升支持；
  - 行归一化会稀释原有邻居的传播权重（normalization loss）。

### 相比现有方法的优势
| 维度 | 传统方法 | PairAlign |
|------|--------|-----------|
| **视角** | 图级或边级 | 节点对级（pair-centric） |
| **目标导向** | 改善全局连通性或局部瓶颈 | 显式修复通信短缺最严重的节点对 |
| **预算分配** | 贪心或独立选择 | OT 协调下的全局协同分配 |
| **理论基础** | 启发式或代理指标 | 与 Jacobian 影响力建立联系 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖三大类标准图基准：

#### （1）同质性节点分类数据集
- **Citation Networks**: Cora, Citeseer
- **WebKB Graphs**: Texas, Cornell, Wisconsin
- **Wikipedia Network**: Chameleon

#### （2）图分类数据集（TUDataset）
- ENZYMES, IMDB-BINARY, MUTAG, PROTEINS, REDDIT-BINARY, COLLAB

#### （3）异质性节点分类数据集（Heterophilic Benchmarks）
- Roman-Empire, Amazon-Ratings, Minesweeper, Tolokers, Questions

---

### 实验设置与评估指标

#### 模型主干（Backbone）
- 同质/图分类：GCN、GIN
- 异质图：GCN、GAT

#### 重连作为预处理步骤
所有 rewiring 方法均在训练前完成，下游模型在重连后的图上训练，确保公平比较。

#### 评估指标
- **下游任务性能**：
  - 分类准确率（Accuracy %）
  - ROC AUC（用于部分异质图）
- **结构修复质量诊断**：
  - **△Shortage**：高短缺节点对的短缺减少程度
  - **Coverage@10**：前10%最高短缺对中有多少获得了显著支持
  - **△PER@T10 / △TER**：有效电阻（effective resistance）的变化，作为外部瓶颈代理

---

### 基线方法对比
涵盖主流重连范式：

| 类别 | 方法 |
|------|------|
| **Curvature-based** | SDRF, BORF |
| **Global Connectivity** | FoSR (spectral gap), GTR (effective resistance), GOKU (spectrum-preserving) |
| **Locality-aware** | LASER |
| **Feature-guided** | ComFy, JDR（用于异质图） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Tables I–III）

#### ✅ 节点分类（Table I）
| Backbone | 方法 | 平均排名 (AR) |
|---------|------|-------------|
| GCN     | PAR (Ours) | **1.5** |
| GIN     | PAR (Ours) | **1.0** |

- 在 GIN 下，Texas 数据集从 53.5% 提升至 **68.8%**，Cornell 从 36.5% 提升至 **51.0%**。
- 在 Cora 和 Citeseer 上也优于无重连基线。

#### ✅ 图分类（Table II）
| Backbone | 方法 | 平均排名 (AR) |
|---------|------|-------------|
| GCN     | PAR (Ours) | **1.6** |
| GIN     | PAR (Ours) | **2.2** |

- 在 ENZYMES 和 MUTAG 上取得最佳结果，尤其在标签依赖跨子结构信息的任务中增益更大。

#### ✅ 异质图节点分类（Table III）
| Backbone | 方法 | 平均排名 (AR) |
|---------|------|-------------|
| GCN     | PAR (Ours) | **2.2** |
| GAT     | PAR (Ours) | **1.6** |

- 超越 ComFy 等结合特征引导的方法，表现出更强的稳定性。
- 在 Roman-Empire 和 Amazon-Ratings 上达到最高精度。

---

### 消融实验结果（Table IV）

| 变体 | Texas (Task) | △Shortage | Coverage@10 |
|------|--------------|------------|--------------|
| Greedy-Local | 0.5622 | 0.0452 | 0.0351 |
| w/o Focus | 0.5568 | 0.2208 | 0.1939 |
| w/o OT | 0.5351 | 0.1741 | 0.1471 |
| **Full PAR** | **0.5877** | **0.2350** | **0.2113** |

#### 关键发现：
- **Focused Target Selection** 至关重要：不聚焦于高短缺对会导致 Coverage@10 显著下降。
- **OT Allocation 提升覆盖率**：相比贪心分配，OT 使预算更广泛地覆盖高短缺对（可视化见 Fig. 4）。
- **Transport Cost 设计有效**：
  - 移除 `bridge suitability` 导致支持效率降低；
  - 移除 `endpoint alignment` 则导致目标错配。

---

## 4. 关键结论和发现

### 主要结论
1. **Pair-Centric View 更有效**  
   将 over-squashing 明确建模为“通信需求 vs. 传播支持”的失衡，能更精准定位需修复的节点对。

2. **Shortage Score 具有判别力**  
   所提 shortage score 与 pairwise effective resistance 高度正相关（Spearman ρ ≈ 0.93），验证其作为 bottleneck proxy 的有效性。

3. **OT-Guided Allocation 实现全局协调**  
   相比贪心策略，OT 能防止预算集中于少数“便宜”目标，实现更均衡、更广泛的短缺缓解。

4. **一致且稳定的性能提升**  
   在多种 backbone（GCN/GIN/GAT）、多种图类型（同质/异质/图分类）上均表现最优，说明其泛化能力强。

---

### 方法的局限性
- **计算开销较高**：Sinkhorn 对齐复杂度为 $ O(T_{\text{OT}} |E| M) $，不适合超大规模图。
- **依赖预定义传播深度 $ K $**：$ K $ 过小无法捕捉长程依赖，过大则增加噪声。
- **仍为预处理方法**：未与 GNN 训练过程联合优化，存在次优风险。

---

### 未来工作方向
1. **Task-Aware Demand Modeling**  
   当前 demand 基于拓扑距离，未来可引入任务信号（如标签相似性）来定义更有意义的通信需求。

2. **Explicit Locality Constraints**  
   在保证非局部连接的同时，显式保留局部邻域结构，提升在异质图中的鲁棒性。

3. **Scalability Improvement**  
   探索采样策略或近似 OT 方法，以适配大规模现实图场景。

4. **End-to-End Integration**  
   将 rewiring 模块嵌入 GNN 训练流程，实现 joint optimization。

---

> **总结一句话**：  
> PairAlign 通过引入 **pairwise communication shortage** 和 **OT-guided budget allocation**，首次实现了以“通信短缺”为核心的精细化图重连，在多个基准上显著缓解 over-squashing 并提升模型性能，为结构修复提供了新的范式。

</details>

---

### 15. [Post-Hoc Sparse Coding of Latent Communication Between Vision-Language Model Agents](https://arxiv.org/abs/2608.10198)

**Authors**: Di Wu, Xiaohui Zhu  
**Category**: cs.AI  
**Published**: 2026-08-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.10198v1  

#### Abstract
Latent-space communication allows heterogeneous vision-language model agents to exchange continuous representations without serializing visual and reasoning states into text. Vision Wormhole realizes this approach by translating visual features into a universal latent representation that can be cons...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Post-Hoc Sparse Coding of Latent Communication Between Vision-Language Model Agents》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文研究了**异构视觉语言模型（VLM）代理之间通过 latent-space 进行通信时存在的冗余问题**。具体而言，Vision Wormhole 方法虽然实现了跨模型的连续表示传输（即无需将视觉状态序列化为文本），但其通信通道采用固定形状的 dense tensor（如 `N×D` 浮点张量），无论消息内容如何，都占用相同的带宽。这导致**实际信息密度低、通信效率低下**。

作者提出的问题是：  
> Vision Wormhole 的固定容量通信通道中，有多少部分是真正必要的？是否存在大量可压缩的冗余？

---

### 提出了什么新方法或新思路
论文提出了 **post-hoc sparse coding** 的分析框架，用于测量和压缩 Vision Wormhole 中已训练好的 latent communication 通道：

- **方法**：在不更新原始 Vision Wormhole 编解码器的前提下，对冻结的 latent 表示（`U_ref ∈ ℝ^{N×D}`）训练一个 **sparse autoencoder (SAE)**。
- **目标**：将 dense 激活重构为稀疏线性组合：`u ≈ W_dec z`，其中 `z` 是仅含 `k` 个非零项的 sparse code。
- **创新视角**：
  - 不是在训练过程中引入稀疏性，而是作为“事后”分析工具（post-hoc），揭示已有通道中的内在结构。
  - 将 SAE 视为一种**测量仪器**，用以量化重建误差、下游任务性能、特征复用程度等。

---

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|---------|
| **无需修改原系统** | 不影响 Vision Wormhole 的训练过程或 agent 行为，避免干扰原始功能。 |
| **显式带宽建模** | 明确计算 sparse payload 所需字节数（uint16 index + float16 value），提供端到端压缩比。 |
| **多维度评估** | 超越单纯重建误差，结合 downstream accuracy、feature reuse、token-level intervention 等综合判断有效性。 |
| **揭示通用支持集** | 发现极小的 active feature 集合被广泛复用于不同任务，暗示潜在的“语义基元”。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共使用 **9 个推理基准数据集**，涵盖多种任务类型：

| 类型 | 数据集 |
|------|-------|
| 数学推理 | GSM8K |
| 科学问答 | ARC-Easy, ARC-Challenge, GPQA, MedQA |
| 编程生成 | MBPP+, HumanEval+ |
| 数学竞赛题 | AIME 2024, AIME 2025 |

> 每个问题由三个 sender 角色处理：planner、critic、refiner → 每个产生一个 `U_ref` tensor。

---

### 实验设置

#### 模型对
- **发送方 → 接收方**：`Qwen3.5-9B → LFM2.5-VL-1.6B`
- **通信张量**：`N=1026` token positions, `D=512` dimensions
- **原始带宽**：`1026 × 512 × 4 bytes = ~2052 KB`（float32）

#### Post-Hoc SAE 设置
- **词典大小 M**：4096
- **稀疏度 k**：每 token 保留前 `k` 个激活系数（Top-K）
- **训练细节**：
  - 冻结 Vision Wormhole 激活
  - Adam 优化器，cosine 学习率衰减，AuxK 正则化
  - Batch size 512，训练 50K 步
  - 使用 dead-feature replacement

#### 带宽计算方式
- **Dense baseline**：`BW_dense = N × D × 4 bytes`
- **Sparse payload**：`BW_sparse = N × k × (b_idx + b_val)`
  - `b_idx = 2 bytes`（uint16 存储 12-bit index）
  - `b_val = 2 bytes`（float16 系数）
  - 忽略包头等系统开销

例如：`k=4` 时，`BW_sparse = 1026 × 4 × 4 = 16.0 KB`

---

### 评估指标

| 指标 | 定义与用途 |
|------|-----------|
| **Reconstruction Error** | 归一化均方误差，衡量重建保真度 |
| **Cosine Similarity** | 向量化后计算，反映方向一致性 |
| **Downstream Task Accuracy** | 接收方使用 sparse 重构消息后的最终任务表现 |
| **Compression Ratio** | `BW_dense / BW_sparse`，端到端带宽节省 |
| **Jaccard Similarity** | 不同任务间 active feature sets 的重叠度，衡量共享性 |
| **Token-Level Intervention** | 干预特定位置（如 style token）观察 MSE 变化，分析角色差异 |

---

### 基线方法对比
本文未直接比较多个 competing baselines，而是以 **原始 float32 dense 传输** 作为主要 baseline。

但明确指出当前结果应视为：
> “end-to-end payload reduction relative to the deployed representation”，而非 solely due to sparsity。

文中也提到未来需补充的对比基线包括：
- float16 quantization
- low-rank approximation (PCA)
- vector quantization (VQ)
- position selection（仅传 18 个有效位置）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📊 表 1：不同 `k` 下的压缩与重建质量

| `k` | 带宽 | 压缩比 | 重建误差 | Cosine Sim. | MSE |
|-----|--------|----------|------------|----------------|--------|
| Dense | 2052 KB | 1× | 0.00000 | 1.00000 | 0.0 |
| **4** | **16.0 KB** | **128×** | 0.01158 | 0.99992 | 3.20e-4 |
| 8 | 32.1 KB | 64× | 0.00978 | 0.99995 | 2.06e-4 |
| 16 | 64.1 KB | 32× | 0.00976 | 0.99994 | 1.99e-4 |
| 32 | 128.3 KB | 16× | 0.00884 | 0.99996 | 1.77e-4 |

✅ 在 `k=4` 时即可实现 **128× 压缩**，同时保持极高相似性（cosine > 0.9999）

---

#### 📊 表 2：下游任务准确率（single-run point estimates）

| Task | Dense (%) | k=4 (%) | Δ |
|------|-----------|---------|----|
| GSM8K | 59.14 | 59.67 | +0.53 |
| ARC-Easy | 75.46 | 78.96 | +3.50 |
| ARC-Challenge | 62.12 | 62.29 | +0.17 |
| GPQA | 27.78 | 27.78 | 0.00 |
| MedQA | 39.33 | 39.00 | -0.33 |
| MBPP+ | 45.50 | 44.71 | -0.79 |
| HumanEval+ | 39.63 | 35.98 | -3.65 |
| **Mean (non-AIME)** | **49.85** | **49.77** | **-0.08** |

📌 **宏观平均准确率几乎不变**（仅下降 0.08 pp），表明 sparse 重构保留了足够的语义信息。

⚠️ 注意：编程类任务（尤其是 HumanEval+）更敏感；所有结果为 single-run，无置信区间。

---

#### 📊 特征使用情况（表 3 & 表 4）

| 指标 | 结果 |
|------|------|
| Dictionary size M | 4096 |
| **Active features** | **50 (1.22%)** |
| Dead features | 4046 (98.78%) |
| Top-10 features 覆盖任务数 | 9/9 (100%) |
| **平均跨任务 Jaccard 相似度** | **0.906**（最小 0.878，最大 0.922）|

➡️ 表明存在一个**极小且高度共享的支持集**，可能构成 latent communication 的“基础词汇”。

---

#### 🔍 Token-Level 干预实验（Style Token Ablation）

- 干预操作：将 style token 位置（pos=17）的所有 sparse coefficients 设为 0
- 结果：
  - 总体 MSE 上升 **28.46×**
  - Style token 的 normalized energy ratio: **0.9238**
  - 但 **semantic token 的 MSE 不变**
  - hybrid reconstruction 更接近 semantic source

✅ 说明：
- style token 在 sparse 表示中有显著能量贡献
- 但其变化不影响 semantic 内容的重建 → 功能分离
- 不同 token roles 激活 pattern 不同，但共享大部分 feature vocabulary

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Vision Wormhole 的 dense 通信通道存在巨大冗余**  
   → 可通过 post-hoc sparse coding 实现 **高达 128× 的端到端带宽压缩**

2. ✅ **高保真重建 + 几乎无损的任务性能**  
   → 即使在 `k=4` 极端稀疏下，七项非 AIME 任务的平均准确率仅从 49.85% → 49.77%

3. ✅ **极小且高度复用的 active feature 支持集**  
   → 仅 50 个 feature 被激活（占词典 1.22%），跨任务 Jaccard 相似度达 0.906

4. ✅ **token roles决定激活模式，而非 feature 本身**  
   → semantic/style/global tokens 共享 feature 字典，但激活分布不同

5. ✅ **style token 具有独立表征作用**  
   → 移除后总体 MSE 大幅上升，但不改变 semantic 内容 → 支持功能解耦假设

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| ❌ **非 end-to-end 训练** | SAE 是 post-hoc 分析，不能证明训练时加入 sparse bottleneck 会更好 |
| ❌ **缺乏 uncertainty estimation** | 所有 downstream accuracy 为 single-run point estimate，统计显著性未知 |
| ❌ **缺少 matched baselines** | 未与 float16、low-rank、VQ 等简单压缩方法对比，无法归因增益来源 |
| ❌ **dead feature 问题严重** | 98.78% feature 从未激活，可能反映 SAE 优化失败或输入低秩结构 |
| ❌ **仅测试单一 model pair** | 结论是否泛化至其他 VLM 对尚不清楚 |
| ❌ **干预未测 receiver behavior** | 仅看 reconstruction MSE，未验证对 logits 或输出的影响 |

---

### 未来工作方向

1. **设计 matched-payload baselines**  
   → 比较 sparse coding vs. quantization vs. low-rank vs. position selection

2. **开展 behavioral intervention studies**  
   → 修改 sparse codes 并观察 receiver 输出变化，建立因果联系

3. **开发 adaptive sparse budget 机制**  
   → 根据 message content 动态调整 `k`，实现 variable-rate communication

4. **探索跨 agent 对的 dictionary transferability**  
   → 是否存在通用的 latent communication vocabulary？

5. **引入 multi-round delta transmission**  
   → 仅传输 sparse code 的变化量，进一步降低通信成本

6. **构建 end-to-end sparse communication pipeline**  
   → 将 SAE 集成进训练流程，验证其长期收益

---

> 💡 **一句话总结**：  
> 本论文通过 post-hoc sparse autoencoder 揭示了 Vision Wormhole 中 latent communication 存在巨大冗余，可在几乎不影响下游性能的情况下实现 **128× 带宽压缩**，并发现了极小且高度共享的 feature support，为未来高效、自适应的 VLM agent 通信协议设计提供了理论依据和技术路径。

</details>

---

### 16. [Continuous Interaction Diffusion: A Diffusion-Native Runtime for Asynchronous Tool-Augmented Reasoning](https://arxiv.org/abs/2608.10438)

**Authors**: Yuhang Cao  
**Category**: cs.AI  
**Published**: 2026-08-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.10438v1  

#### Abstract
Large language models increasingly rely on external tools to access up-to-date information, perform computation, and interact with the outside world. For autoregressive models, tool use naturally fits the generation process: the model emits a tool call, waits for the result, and then continues gener...

---

### 17. [Rationale-Guided Learning for Multimodal Emotion Recognition](https://arxiv.org/abs/2608.10448)

**Authors**: Sujung Oh, Jung Uk Kim, Sangmin Lee  
**Category**: cs.AI  
**Published**: 2026-08-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.10448v1  

#### Abstract
Multimodal emotion recognition in conversation (MERC) requires understanding complex interactions between verbal and non-verbal cues. However, most existing approaches fundamentally treat this as a direct input-output (multimodal cues-emotion labels) mapping problem, overlooking the causal reasoning...

---

### 18. [Dual-Loop Self-Evolution via Verifiable Emotion Feedback for Multi-Turn Empathetic Dialogue](https://arxiv.org/abs/2608.10626)

**Authors**: Yi Wei, Shuo Jiang, Huaixia Dou, Jie Zhu, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang  
**Category**: cs.CL  
**Published**: 2026-08-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.10626v1  

#### Abstract
Large language models have demonstrated conversational capabilities, yet empathetic competence remains challenging. Empathetic support is inherently multi-turn and path-dependent: users disclose concerns gradually, emotions evolve over time, and early responses shape trust and receptivity. Reinforce...

---

### 19. [Accelerated Learning of High Dimensional Functions with a Tensor-Featured Training Network](https://arxiv.org/abs/2608.10351)

**Authors**: Karl Pierce, Yuehaw Khoo, Haizhao Yang  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.10351v1  

#### Abstract
In this work we present a method to accelerate the optimization of learning high dimensional functions using deep neural network (DNN). This optimization procedure introduces contextual features into the first layer of a DNN. The parameters of DNN are optimized via standard gradient descent while ke...

---

### 20. [IADD-TR: Intervention-Aware Dynamics Decoupling with Targeted Regularization for Model-Based Reinforcement Learning](https://arxiv.org/abs/2608.10634)

**Authors**: Zefeng Liang, Jie Qiao, Ruichu Cai, Weilin Chen, Zhifeng Hao  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.10634v1  

#### Abstract
Model-based reinforcement learning (MBRL), which learns environment dynamics to generate synthetic experience, is a promising approach to sample-efficient decision making. Numerous methods have been developed to improve dynamics prediction and policy optimization for MBRL through uncertainty estimat...

---

### 21. [Can Bayesian Optimization Efficiently Find a Strong Single Expert in Neural Thickets?](https://arxiv.org/abs/2608.10867)

**Authors**: Nigel Bastian Cendra, Abdelhamid Ezzerg, Fernando Julio Cendra, Jeremias Knoblauch, Jakob Zeitler  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.10867v1  

#### Abstract
Gradient-free post-training has emerged as a compelling alternative to gradient-based optimization for large language models (LLMs), but existing approaches remain costly. We ask whether structured search can identify a strong single expert under a modest evaluation budget. Motivated by evidence tha...

---

### 22. [Predicting Space Groups of Double Perovskites by LLM with Dynamic Few-Shot Learning](https://arxiv.org/abs/2608.10483)

**Authors**: Jongwon Park, Inhyo Lee, Junhyeong Lee, Seunghwa Ryu  
**Category**: cs.AI  
**Published**: 2026-08-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.10483v1  

#### Abstract
Double perovskites (DPs) offer broad compositional tunability, but predicting the space groups (SGs) of stable structures remains difficult because available datasets are often strongly imbalanced toward dominant SG classes. We refer to dominant SG classes as major SGs and underrepresented classes a...

---

### 23. [EvoMem: Memory-Augmented Evolution for Code Optimization](https://arxiv.org/abs/2608.10795)

**Authors**: Viktor Volkov, Valentin Khrulkov, Andrey V. Galichin, Danil Sivtsov, Nikita Glazkov, Olga Volkova, Konstantin Pchelin, Iaroslav Bespalov, Dmitry V. Dylov, Petr Anokhin, Ivan Oseledets  
**Category**: cs.AI  
**Published**: 2026-08-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.10795v1  

#### Abstract
Successful mutation strategies in evolutionary code search may contain reusable knowledge that is useful beyond a single run, and in some cases may transfer across related tasks and domains. However, existing LLM-driven evolutionary frameworks largely discard such knowledge, repeatedly rediscovering...

---

### 24. [How Robust Are LLMs to Vietnamese Dialects?](https://arxiv.org/abs/2608.10414)

**Authors**: Minh Tran, Trinh Chau, Thanh-Nhan Le, Nam Tran, Luan Thanh Nguyen, Cuong Dang, Duc Hoang  
**Category**: cs.CL  
**Published**: 2026-08-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.10414v1  

#### Abstract
Large Language Models (LLMs) are typically evaluated on standard written Vietnamese, yet everyday communication frequently involves regional dialects that preserve meaning but differ in surface form. Existing Vietnamese dialect work largely addresses this issue through dialect-to-standard normalizat...

---

### 25. [FaCTz: Fast Critical-Point and Topology-Aware GPU Compression for Scientific Vector Fields](https://arxiv.org/abs/2608.10586)

**Authors**: Mingze Xia, Yuxiao Li, Sheng Di, Jiannan Tian, Baixi Sun, Boyi Zhang, Bei Wang, Hanqi Guo, Xin Liang  
**Category**: cs.DC  
**Published**: 2026-08-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.10586v1  

#### Abstract
Error-bounded lossy compression is essential for storing and transferring the vector-field data produced by large-scale scientific simulations. Although it enforces a user-specified error bound to limit numerical distortion, it does not preserve the field's topology: small admissible perturbations c...

---

### 26. [A matched-integrator evaluation of Hamiltonian neural networks on pendulum and Kepler dynamics](https://arxiv.org/abs/2608.10235)

**Authors**: Lenick Kemunto Nyabuto, Yae Ulrich Gaba, Birahim Tewe  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.10235v1  

#### Abstract
Hamiltonian Neural Networks (HNNs) parameterize conservative dynamics through a learned scalar Hamiltonian, providing an architectural prior that is absent from generic vector-field neural networks. We evaluate this prior under a controlled protocol in which an HNN and a parameter-matched feedforwar...

---

### 27. [CRHT: A Continuous Regression Hybrid Transformer for Vessel Trajectory Prediction with Online Cluster Sampling](https://arxiv.org/abs/2608.10256)

**Authors**: Alexander Schi{\o}tz, Bertram Hage, Christian Rand, Felix Thomsen, Peder Heiselberg  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.10256v1  

#### Abstract
Accurate vessel trajectory prediction is critical for maritime safety and anomaly detection, yet existing models often struggle with geographic bias and navigational realism. We propose the Continuous Regression Hybrid Transformer (CRHT), a deep learning framework designed to forecast vessel motion ...

---

### 28. [ProbGuard: Calibrated Safety Risk Estimation from LLM Output Distributions](https://arxiv.org/abs/2608.10621)

**Authors**: Xinzhe Huang, Biwu Yao, Kedong Xiu, Mengnan Zhao, Di Wang, Puning Zhao, Tianhang Zheng  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.10621v1  

#### Abstract
Recent research on Large Language Model (LLM) safety has widely adopted guardrails to identify unsafe LLM outputs. Existing guardrails typically formulate safety assessment as a deterministic classification task, mapping a discrete token sequence to a discrete safety label. However, this paradigm ha...

---

### 29. [Diffract: Spectral View of LLM Domain Adaptation](https://arxiv.org/abs/2608.10850)

**Authors**: Nikita Borodin, Maria Krylova, Artem Zabolotnyi, Dmitry Aspisov, Egor Shikov, Nikita Tyuplyaev, Oleg Travkin, Roman Alferov, Dmitry Vinichenko  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.10850v1  

#### Abstract
We study continual pre-training (CPT) as a mechanism for adapting general-purpose large language models to specialized domains: mathematics, instruction, code, and natural text. Using singular value decomposition of weight matrices, we find that CPT leaves singular value spectra largely invariant, w...

---

### 30. [DEFT: Data-Efficient Frequency-domain Top-k Sampling via Inverse Discrete Fourier Transform for Spatiotemporal Dynamical Systems Modeling](https://arxiv.org/abs/2608.11019)

**Authors**: Hengbo Xiao, Jiale Liu, Jiahao Song, Guannan He  
**Category**: cs.LG  
**Published**: 2026-08-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.11019v1  

#### Abstract
Modeling spatiotemporal dynamical systems governed by partial differential equations (PDEs) poses two major challenges: it either requires expensive physics-based simulators that entail iterative numerical solving at high computational cost, or it depends on abundant training data, yet purely data-d...

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

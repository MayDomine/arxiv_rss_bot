# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-13 07:07:52 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Ripple-Pivot Search: Active Parallel Decoding for Diffusion Large Language Models](https://arxiv.org/abs/2608.11742)

**Authors**: Yushi Ye, Xu Chen, Haoyun Jiang, Jinsong Lan, Haihong Tang, Bo Han, Ivor Tsang, Yanfeng Wang, Bo Zheng, Jiangchao Yao  
**Category**: cs.CL  
**Published**: 2026-08-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2608.11742v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have emerged as a competitive alternative to autoregressive language models, offering the potential for substantially faster inference through parallel decoding. Existing parallel decoding schedulers typically commit positions only after they meet a per-positi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Ripple-Pivot Search: Active Parallel Decoding for Diffusion Large Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Diffusion Large Language Models (dLLMs) 虽然支持并行解码以加速推理，但现有并行解码调度器（scheduler）通常仅在满足**逐位置标准**（如高置信度、低熵）时才提交（commit）token，忽略了**早期提交对后续解码的潜在促进作用**。这种策略可能导致：
- 提交过少 → 解码速度慢；
- 提交过多 → 错误累积，影响生成质量。

因此，如何在**解码速度**与**生成质量**之间取得平衡是 dLLMs 推理的核心挑战。

---

### **提出的新方法：Ripple-Pivot Search (RPS)**

作者提出了一种**无需训练的并行解码方法**——**Ripple-Pivot Search (RPS)**，其核心思想基于一个新发现的 **“ripple effect”（涟漪效应）**：

> **主动提交处于中等熵（mid-entropy）区域的关键位置（pivot），可以显著降低其余掩码位置的不确定性，从而在后续步骤中解锁更多并行提交机会，加速整体解码过程。**

RPS 包含两个关键决策：
1. **Where to decode（选择 pivot 位置）**  
   - 通过两阶段过滤机制聚焦于 **mid-entropy regime** 中的位置：
     - 首先保留每个位置预测分布中概率最高的 `k_max` 个 token；
     - 然后选择**截断熵最大**且**累计概率质量超过阈值**的位置作为 pivot。
   - 这确保了所选 pivot 处于“尚未确定但已有一定倾向”的状态，最有可能引发强 ripple effect。

2. **What to decode（决定提交哪个 token）**  
   - 构建一个自适应候选集（包含高可能性 token 和 `[MASK]`）；
   - 使用一次 **lookahead forward pass** 对每个候选 token 进行评估；
   - 选择能带来**最大下游熵减少 + 合理性正则化**的 token。

最终提交条件为：**只有当最佳候选优于保持该位置掩码时，才进行提交**，防止错误积累。

---

### **相比现有方法的优势**

| 方法类别 | Where to decode | What to decode | 缺陷 |
|--------|----------------|---------------|------|
| Confidence/Entropy-based | 高置信/低熵位置 | Top-1（贪婪） | 忽视早期承诺的连锁效益 |
| Lookahead-based (e.g., LoPA, ETE) | Lookahead 选择位置 | Top-1（贪婪） | 仍采用贪婪赋值，忽略非 top-1 正确答案 |
| **RPS (本文)** | **Mid-entropy pivot** | **Lookahead 评估非贪婪赋值** | ✅ 同时优化“何处”与“何物”，利用 ripple effect |

> 🔑 **关键突破**：在 mid-entropy 区域，正确 token 在 85% 的情况下**不是模型当前 top-1 预测**，因此固定使用 greedy assignment 会错失最优路径。RPS 显式探索非贪婪赋值，更可能找到高质量解码轨迹。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
涵盖**数学推理**与**代码生成**两大任务：
- **Math Reasoning**:
  - GSM8K (5-shot)
  - MATH500 (4-shot)
- **Code Generation**:
  - HumanEval (0-shot)
  - MBPP (3-shot)

---

### **实验设置与评估指标**

#### **模型**
- **LLaDA-8B-Instruct**, **Dream-v0-Instruct-7B**, **LLaDA-1.5**
- 分别代表从头训练与由 autoregressive 模型适配而来的 dLLM 家族。

#### **评估指标**
| 指标 | 含义 |
|------|------|
| **Acc (%)** | 生成准确率（pass@1） |
| **NFE** | Number of Function Evaluations，衡量前向传播次数，反映计算效率 |
| **TPS** | Tokens Per Second，端到端吞吐量，反映实际 wall-clock 速度 |
| **Speedup** | 相对于 Default（逐 token 解码）的速度提升倍数 |

#### **默认配置**
- 生成长度：256
- 块大小（block size）：32
- RPS 超参数（LLaDA）：`k_max=10`, `r=0.1`, `T_pivot=0.9`, `λ∈[0.1,0.5]`

---

### **基线方法对比**
| 基线 | 类型 | 特点 |
|------|------|------|
| **Default** | Baseline | 每步只解码一个 token |
| **Confidence** | Criterion-based | 高置信度位置优先解码 |
| **KLASS** | Stability-based | 结合跨步稳定性 |
| **EB-Sampler** | Entropy-bound | 控制每步累计熵 |
| **WINO** | Draft-Verify | 松阈值起草 + 严阈值验证 |
| **LoPA** | Lookahead-based | 基于 lookahead 选择高置信残差位置 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| 方法 | 模型 | Acc (%) | NFE | TPS | TPS Speedup |
|------|------|--------|-----|-----|-------------|
| **Default** | LLaDA | 79.00 | 256.0 | 4.02 | 1.00× |
| **LoPA** | LLaDA | 78.70 | 41.95 | 22.98 | **5.68×** |
| **RPS** | LLaDA | **79.15** | **35.73** | **26.66** | **6.63×** |
| **Default** | Dream | 78.70 | 256.0 | 5.14 | 1.00× |
| **LoPA** | Dream | 75.66 | 34.62 | 31.31 | 6.09× |
| **RPS** | Dream | **78.62** | **41.74** | **28.21** | **5.49×** |

> ✅ **RPS 实现 4–10× TPS 加速**，同时**保持甚至提升准确率**。

---

### **与基线方法的对比结果**

- **速度方面**：
  - RPS 达到 **6.63×~9.80× TPS speedup**，显著优于大多数基线；
  - 尤其在 MBPP 上，Dream 模型实现 **9.80× TPS speedup**。

- **准确性方面**：
  - 在 HumanEval 上，RPS 比 LoPA **高出最多 5.49%**（Dream）；
  - 在 MBPP 上，RPS 比 Default 提升 **2.2%**（LLaDA）；
  - 而 LoPA 在多个任务上出现明显精度下降。

- **综合表现**：
  - RPS 是唯一同时进入“高准确 + 高速”象限的方法（见 Fig. 3 中间图）；
  - LoPA 虽快但易犯“提前终止”错误（如过早提交 `return`）。

---

### **消融实验结果**

#### **(1) Plausibility Weight λ 敏感性分析（Fig. 3 左）**
- 当 `λ = 0`（无合理性约束）时，准确率显著下降；
- 只需 `λ ≥ 0.1` 即可恢复大部分性能，且在 `[0.1, 0.5]` 内鲁棒；
- 表明 **plausibility safeguard 对防止错误提交至关重要**。

#### **(2) 超参数敏感性（Table 3）**
- `k_max = 10` 时性能最佳；过大引入噪声，过小限制搜索空间；
- `r = 0.1` 最优；过小可能剪掉合理候选；
- `T_pivot = 0.9` 平衡可靠性与速度。

#### **(3) 枢纽策略与打分机制消融（Fig. 3 中）**
- **Max Confidence**：保守，速度快但收益有限；
- **Unconstrained Max Entropy**：激进，速度快但精度低；
- **RPS 完整设计**：结合 reachability constraint 与 plausibility-aware scoring，实现最佳权衡。

---

## **4. 关键结论和发现**

### **主要发现**
1. **存在“ripple effect”**：  
   主动提交 mid-entropy 位置可显著降低其他位置的不确定性，形成“涟漪式”信息传播，促进后续并行解码。

2. **正确 token 常非 top-1**：  
   在 mid-entropy 区域，约 **85% 的正确 token 不是当前 top-1 预测**，说明必须超越 greedy decoding 才能找到最优路径。

3. **RPS 实现高效且高质量解码**：  
   - 在 3 个 dLLMs 和 4 个 benchmark 上，实现 **4–10× wall-clock speedup**；
   - 准确率持平或优于 Default；
   - 比 LoPA 等 lookahead 方法**最高提升 5.49% 准确率**。

4. **兼容 KV Caching**：  
   与 Fast-dLLM 的 prefix caching 结合后，**最高可达 18× wall-clock speedup**，显示迭代级优化与 per-forward 优化可叠加。

---

### **方法的局限性**
1. **超参数仍需轻度调优**：  
   虽然 `λ` 在一定范围内鲁棒，但仍需在 `[0.1, 0.5]` 内选择，并非完全 zero-tuning。

2. **ripple effect 为经验性观察**：  
   当前分析基于 GSM8K 子集，缺乏理论推导，是否普适于所有模型/任务尚待验证。

3. **缓存近似可能引入误差**：  
   与 KV caching 结合时，精度轻微下降（<0.5%），但这是通用问题，非 RPS 独有。

---

### **未来工作方向**
- 理论建模 ripple effect 的传播机制；
- 自动化超参数选择（如动态调整 `λ` 或 `T_pivot`）；
- 将 RPS 思想扩展至图像、音频等其他 diffusion 模型；
- 探索多 pivot 并行搜索以进一步提升效率。

---

> ✅ **总结一句话**：  
> **RPS 通过识别并利用 dLLM 解码中的“涟漪效应”，在 mid-entropy 位置进行非贪婪的 lookahead 搜索，实现了高速、高质、无需训练的并行解码新范式。**

</details>

---

### 2. [RoutePack: Expert Placement and Attention-Aware Data Packing for MoE Reinforcement Learning](https://arxiv.org/abs/2608.12146)

**Authors**: Yibo Shen, Xudong Han, Xiaowei Zhu, Gen Li, Zhenxuan Pan  
**Category**: cs.DC  
**Published**: 2026-08-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.12146v1  

#### Abstract
Training Mixture-of-Experts (MoE) models for reinforcement learning (RL) couples two load-balancing problems: sequence composition determines dense attention work in each data-parallel microbatch, while token routing determines sparse expert work on expert-parallel ranks. Optimizing either alone can...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RoutePack: Expert Placement and Attention-Aware Data Packing for MoE Reinforcement Learning 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在 **MoE (Mixture-of-Experts)** 模型的 **Reinforcement Learning (RL)** 训练中，存在两个耦合的负载不平衡问题：

- **Dense Attention 负载不均**：由变长序列打包（data packing）导致不同 DP（Data-Parallel）微批次的注意力计算量差异。
- **Sparse Expert 负载不均**：由 token 路由（token routing）导致某些 EP（Expert-Parallel）物理 rank 接收过多 token。

传统方法通常孤立优化其中一个方面（如仅优化序列打包或仅重排专家），但这种做法可能将瓶颈转移到另一个维度。例如，优化 attention 可能加剧 expert 负载倾斜，反之亦然。

### 提出了什么新方法或新思路

本文提出 **RoutePack**，一种**分层协同规划框架**，首次联合优化 **layer-wise expert placement** 和 **attention-aware data packing**，利用 RL 中独有的 **routing replay** 机制实现端到端负载均衡。

#### 核心创新点：

1. **联合控制信号**：  
   利用 rollout 阶段记录的 **exact per-sample, per-layer expert demand**（即 routing replay），作为训练前调度的先验信息，实现对 expert placement 和 data packing 的统一协调。

2. **分层优化目标**：
   - **第一阶段：Layer-wise Expert Placement**  
     使用 **Longest Processing Time (LPT)** 算法，在每个 MoE 层独立地重新分配逻辑专家到物理 EP rank，以最小化整个 optimizer step 窗口内的 aggregate expert 负载方差。
   - **第二阶段：Attention- and Expert-Aware Data Packing**  
     在固定 expert placement 和最小行数的前提下，将样本打包为 **token-capped execution rows**，并引入一个 **joint objective** 同时建模：
     - **Attention 工作量**：基于 token-linear 和 quadratic 项的代理成本（proxy cost）
     - **Expert 负载峰值**：每个 EDP shard 内最忙的物理 EP rank 的负载总和
     - 目标是 **最小化最慢 EDP shard 的累积代价**

3. **状态一致性保障（State-Consistent Materialization）**：
   - 不改变逻辑 top-k 路由规则或模型图结构。
   - 通过 dispatcher remapping 将逻辑专家映射到新的物理 slot，保持前向/后向一致性。
   - 支持与现有 MoE 内核（如 DeepEP）兼容，无需修改通信原语。

4. **高效搜索算法**：
   - 使用 **diverse seeding + parallel population annealing** 进行布局搜索。
   - 并行链独立运行，通过系统性重采样（systematic resampling）保留优质解路径。
   - 最终选择基于 **lexicographic scoring**（行数 → 最慢 shard 总代价 → 总工作量 → 最坏行尾延迟）。

### 相比现有方法的优势

| 方面 | RoutePack | 现有方法（如 ReLibra、FineMoE、UltraEP） |
|------|---------|----------------------------------------|
| 控制粒度 | **optimizer-step 级别**，整样本打包 | 多为 microbatch 级别，常涉及 token splitting 或动态复制 |
| 是否改变容量 | ❌ 不引入额外 microbatch 或专家复制 | ✅ 常通过复制专家缓解负载 |
| 是否破坏语义 | ❌ 保持样本一一对应、拓扑不变 | ⚠️ 动态复制可能影响梯度一致性 |
| 是否考虑 attention | ✅ 显式建模 attention 工作量 | ❌ 多只关注 expert compute 或通信 |
| 协同性 | ✅ 联合优化 placement 与 packing | ⚠️ 多为顺序或局部优化 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **GSM8K**：数学推理任务数据集，用于生成训练轨迹。

### 实验设置

| 参数 | 设置 |
|------|------|
| 模型 | **Ling-3.0-Tiny** (7.9B total / 1.3B activated)<br>**Ling-3.0-Flash** (124B total / 5.1B activated) |
| 注意力结构 | 均包含 **KDA (Kimi Delta Attention)** 和 **MLA (Multi-Head Latent Attention)** 层 |
| 训练算法 | **GRPO** (Generalized Reward Policy Optimization) |
| 每步样本数 | 512 sequences（64 prompts × 8 rollouts） |
| 微批次 token 容量 | 8,192 tokens |
| EDP 结构 | Tiny: 1 shard；Flash: 2 shards |
| MoE 调度后端 | **DeepEP** 用于 dispatch/combine |

### 评估指标

- **主指标**：
  - **Trainer-measured token throughput**（tokens/s）：真实训练吞吐量
- **中间指标**：
  - `CV_global`：optimizer-step 级别的 EP rank 负载变异系数（衡量 aggregate imbalance）
  - `EP peak sum`：所有 row-layer-rank 中的最大负载之和
  - `Tail peak`：单个最高峰值
  - `EP balance efficiency`：理想均衡 vs 实际峰值的比率
  - `Joint objective`：attention + expert cost 的综合代价

### 基线方法对比

| 配置 | 描述 |
|------|------|
| **Baseline** | Identity expert placement + length-only FFD packing |
| **Reorder** | LPT expert reordering + 原始 FFD packing |
| **RoutePack** | LPT expert reordering + routing-aware packing |

所有配置保持相同的 **execution row count** 和 **sample multiset**，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（平均 ± 标准差）

| 模型 | 方法 | Token Throughput (k tokens/s) | 提升（vs Baseline） |
|------|------|-------------------------------|---------------------|
| **Ling-3.0-Tiny** | Baseline | 42.86 ± 3.05 | — |
| | Reorder | 44.49 ± 1.92 | **+3.80%** |
| | **RoutePack** | **46.65 ± 2.12** | **+8.85%** |
| **Ling-3.0-Flash** | Baseline | 68.50 ± 5.02 | — |
| | Reorder | 75.69 ± 4.59 | **+10.50%** |
| | **RoutePack** | **78.70 ± 3.82** | **+14.89%** |

> 所有 pairwise 对比在 Bonferroni 校正后仍显著（最大调整 p-value = 0.0110）

### 第10百分位吞吐量提升（体现稳定性）

- **Tiny**: 39,651 → **43,891** tokens/s (+10.7%)
- **Flash**: 61,588 → **75,033** tokens/s (+21.8%)

### 消融实验结果

#### （1）Expert Reordering 贡献

- 显著降低 `CV_global`：
  - Tiny: 从 ~0.4 → <0.003（下降 >99%）
  - Flash: 降至日志精度以下（近乎完全均衡）
- 表明 LPT 成功消除了 optimizer-window 级别的 aggregate expert skew。

#### （2）Routing-aware Packing 贡献（在 Reorder 基础上）

| 指标 | Tiny | Flash |
|------|------|-------|
| `EP peak sum` ↓ | -3.13% | -3.24% |
| `Tail peak` ↓ | -11.04% | -11.62% |
| `Joint objective` ↓ | -1.53% | -1.35% |
| `Attention cost` ↑ | +0.09% | +0.77% |

> **关键发现**：packing 主动接受少量 attention 成本上升，换取更大的 expert peak 下降，体现了 **joint objective 的权衡能力**。

---

## 4. 关键结论和发现

### 主要发现

1. **Expert placement 与 data packing 必须协同优化**：  
   单独优化任一方面无法突破另一方面的瓶颈。**routing replay 提供了联合控制的独特机会**。

2. **分层设计有效且实用**：  
   先固定 expert placement，再进行 packing，既能降低 aggregate skew，又能进一步优化 row-local peaks，二者互补。

3. **状态一致性至关重要**：  
   RoutePack 不改变逻辑路由、不复制专家、不破坏前向/后向依赖，易于集成进现有训练系统（如 AReaL）。

4. **CPU 规划可重叠于训练流水线**：  
   分析表明，在满足一定条件下（如 `T_LPT + T_pack ≤ T_actor`），**CPU 上的 planning 不会延长 training admission 关键路径**。

### 方法的局限性

1. **依赖 routing replay 的准确性**：  
   若训练时路由发生变化（如异步更新策略网络），则 replay 信息失效，需重新预测或刷新。

2. **未建模通信开销**：  
   当前目标函数聚焦 compute 负载，未显式优化 all-to-all 通信量或拓扑感知传输。

3. **未覆盖复杂并行结构**：  
   实验集中在 DP+EP+EDP，未测试大规模 PP（Pipeline Parallelism）或 CP（Context Parallelism）场景。

4. **compute proxy 简化假设**：  
   使用线性-二次项近似 attention 成本，忽略了 kernel 内部的 tile、occupancy 等细节影响。

### 未来工作方向

1. **扩展至通信敏感场景**：  
   引入 topology-aware cost term，联合优化 compute 与 communication。

2. **支持动态 routing 场景**：  
   将 replay 作为 prediction 输入，结合 uncertainty 建模进行鲁棒规划。

3. **更精细的 kernel modeling**：  
   构建基于 microbenchmark 的 latency surface，提升 cost proxy 准确性。

4. **跨 optimizer-step 的长期规划**：  
   探索多步窗口下的专家迁移与负载平滑策略。

5. **开放 reproducibility**：  
   发布等效 checkpoint 与 routing traces，便于社区复现与改进。

---

> **总结**：RoutePack 首次将 **routing replay** 从单纯的 load balancing 信号，提升为 **model-state 与 data-layout 协同编排** 的控制接口，展示了在 MoE RL 中通过 **planning-time coordination** 实现端到端性能提升的新范式。

</details>

---

### 3. [CORA-Diff: Confidence-Oriented Residual Acceptance for Efficient Diffusion Language Model Inference](https://arxiv.org/abs/2608.11235)

**Authors**: Yifan Wu, Yufeng Zhang, Kenli Li  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.11235v1  

#### Abstract
Diffusion language models (DLMs) update many tokens in parallel, yet practical decoders often use a fixed denoising horizon. Many predictions stabilize early, but blockwise decoding continues until all positions are resolved, causing repeated dense forward passes. Existing accelerators often rely on...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# CORA-Diff 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
- **冗余计算问题**：现有的 Diffusion Language Models (DLMs) 在推理时通常采用固定去噪步数（fixed denoising horizon），即使许多 token 已经稳定，仍会执行密集的前向传播，导致大量重复计算。
- **依赖复杂机制**：现有加速方法常依赖学习型过滤器、修改 logits、引入依赖模型或缓存机制等，增加了系统复杂性和训练开销。

### 提出的新方法：CORA-Diff
- **全无训练（training-free）残差接受机制**：
  - 提出 **CORA-Diff**（Confidence-Oriented Residual Acceptance for Diffusion Language Models），一种无需额外训练、不修改 backbone 或 logits 的轻量级解码策略。
  - 仅对原始转移规则未解决的位置应用基于 **置信度（confidence）** 和 **持续性（persistence）** 的门控机制。
- **早期终止机制**：
  - 当所有位置都被原始规则或 CORA-Diff 接受后，立即终止当前 block 的去噪过程，跳过剩余密集前向传递。

### 相比现有方法的优势
| 特性 | CORA-Diff | 其他方法（如 Learn2PD, DAPD 等） |
|------|----------|-------------------------------|
| 是否需要训练 | ❌ 否 | ✅ 是（需训练 acceptance 模型） |
| 是否修改 logits | ❌ 否 | ✅ 是（部分方法） |
| 是否改变 backbone | ❌ 否 | ✅ 是（部分方法） |
| 是否依赖缓存机制 | ❌ 否 | ✅ 是（如 KV cache） |
| 部署简单性 | ⭐⭐⭐⭐⭐ 极简 | ⭐⭐⭐ 中等 |

> ✅ **核心优势**：利用原生去噪轨迹中的信号（top-1 confidence + prediction persistence）即可实现高效且可靠的 token 提前接受，无需任何附加组件。

---

## 2. 核心实验方法和设置

### 数据集
- **主任务测试集**：
  - **GSM8K**：数学应用题基准
  - **MATH**：高难度数学问题
  - **HumanEval**：代码生成能力评估
  - **MBPP**：Python 编程任务
- **校准集**：
  - 使用 **1,000 个独立的 GSM8K 训练样本** 进行阈值选择（$(\delta_p, m) = (0.65, 1)$），该集合与测试集无交集。

### 实验设置
- **模型架构**：基于 **LLaDA-8B-Instruct** 的 blockwise masked diffusion 解码框架。
- **去噪机制**：
  - 固定 horizon 设置为 `256/256` 和 `1024/1024`（长度/块大小）。
  - 温度为 0，确保确定性输出。
- **硬件环境**：单张 RTX 5090，batch size 1，bfloat16 精度。
- **评估协议**：
  - 采用 **Learn2PD-style 协议**，保证与其他方法公平比较。
  - 所有 baseline 使用官方实现和推荐参数复现。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Task Metric** | Flex EM（GSM8K/MATH）、pass@1（HumanEval/MBPP） |
| **Speedup (Spd.)** | 相对于原始 dense decoding 的运行时间加速比 |
| **ρ_step (Ratio)** | 实际执行步数 / 固定去噪步数，衡量计算节省程度 |
| **Time (s)** | 完整测试集上的总墙钟时间（三轮平均） |
| **Tok/s** | 平均每秒生成 token 数量 |

### 基线方法对比
- **Prophet (ICLR'26)**：基于答案提前收敛的检测
- **KLASS (NeurIPS'25)**：基于分布稳定性（KL 散度）
- **DAPD (ICML'26)**：依赖感知并行解码
- **Learn2PD (ICLR'26)**：使用轻量级可学习过滤器预测最终输出一致性（最接近的强基线）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table 1 & 2）

| 方法 | 任务 | Horizon | Speedup (×) | ρ_step | Task Score |
|------|------|--------|------------|--------|-----------|
| **Original** | Average | 256/256 | 1.00× | 1.000 | — |
| **CORA-Diff (Ours)** | Average | 256/256 | **4.34×–9.49×** | **0.1010–0.2278** | 匹配或优于 dense |
| **Original** | Average | 1024/1024 | 1.00× | 1.000 | — |
| **CORA-Diff (Ours)** | Average | 1024/1024 | **10.68×–13.14×** | **0.0756–0.0894** | 最大下降仅 1.22 pts |
| **CORA-Diff + EoTP** | GSM8K | 1024 | **2.70× vs EOS-aware** | — | 0.7892 (↑) |
| **CORA-Diff + EoTP** | HumanEval | 1024 | **3.32× vs EOS-aware** | — | 0.3841 (=) |

> 🔥 **最高加速比达 13.14×**（GSM8K, 1024/1024），是所有八种设置中**最低延迟的方法**。

### 与基线方法对比结果
- 在所有任务和设置下，**CORA-Diff 的 runtime 均为最低**。
- 相较于最强基线 **Learn2PD**：
  - 测量到的平均速度提升高出 **9.1%–30.9%**
  - 无需训练 acceptance 模型，部署更简单
- 在 **task score 上表现稳健**：
  - 五项设置中得分匹配或超过 dense decoding
  - 最大性能下降仅为 **1.22 分**（如 MATH 上从 0.2586 → 0.2546）

### 消融实验结果（Table 3）
在 GSM8K 上进行组件分析（$\delta_p=0.65$ 固定）：

| 变体 | Flex EM | Tok/s | ρ_step |
|------|--------|-------|--------|
| Confidence only ($m=0$) | 0.7301 | 43.51 | 0.1783 |
| Persistence only | 0.7557 | 35.06 | 0.2245 |
| **Both ($m=1$)** | **0.7794** | **39.12** | **0.2012** |
| Both ($m=2$) | 0.7763 | 34.47 | 0.2361 |

> ✅ **关键发现**：结合 confidence 与 **至少一次 persistence 检查**（即连续两步相同预测）能显著提升准确率（+0.05），而 $m=2$ 不再带来收益，说明 $m=1$ 是最优平衡点。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **原生轨迹信号足够可靠**：
   - 仅凭 **top-1 confidence** 和 **prediction persistence** 就足以判断 token 是否可以安全接受，无需额外训练或修改模型。
2. ✅ **理论支持与实证一致**：
   - 高置信 + 持续预测更可能与最终输出一致（margin-based stability argument）。
   - 图1 和补充表 S6 提供了 dense-trace 与 post-intervention 轨迹的双重验证。
3. ✅ **实现显著加速且保持质量**：
   - 在固定 horizon 下达到 **高达 13.14× 的加速比**
   - 在实际部署场景（EOS-aware）中仍取得 **2.70×–3.32× 加速**
   - 支持跨 backbone（Dream）零调优迁移（3.18×–3.53×）

### 局限性
- **稀疏高置信区域数据不足**：一个高置信 stratum 因样本少未能充分验证假设（data-limited）。
- **依赖 deterministic 轨迹**：目前分析基于 temperature=0 的确定性解码，随机采样下的泛化需进一步研究。
- **未探索动态 horizon 自适应调度**：仍以 fixed block 结构为基础，未来可结合 adaptive scheduling。

### 未来工作方向
- 扩展至非确定性（stochastic）解码场景
- 探索与其他加速技术（如 KV cache, speculative decoding）的联合优化
- 应用于更大规模模型或多模态 diffusion 模型
- 开展大规模真实部署测试（real-world latency profiling）

---

> 📌 **总结一句话**：  
> **CORA-Diff 证明了“信心 + 持续”这一简单直觉可在无需训练、不改模型的前提下，有效识别可接受 token，从而大幅减少冗余去噪步骤，在多个任务上实现最高效率且几乎无损的任务性能。**

</details>

---

### 4. [Kernel Methods for Learning Operators with Multiple Inputs and Outputs](https://arxiv.org/abs/2608.11831)

**Authors**: Adrien Weihs, Chunyang Liao, Jingmin Sun, Hayden Schaeffer  
**Category**: cs.LG  
**Published**: 2026-08-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.11831v1  

#### Abstract
Learning mappings between infinite-dimensional objects is a central challenge in scientific machine learning. We introduce a general kernel-based encoder-decoder framework for operator learning that separates observation, representation, learning, and reconstruction. We develop this framework for mu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Kernel Methods for Learning Operators with Multiple Inputs and Outputs

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在解决**科学机器学习**（scientific machine learning）中的一个核心挑战：在无限维函数空间之间学习映射（即**operator learning**），特别是针对具有**多输入、多输出**（multi-input, multi-output）的算子学习问题。这类问题广泛存在于参数化偏微分方程（parametric PDEs）、逆问题和多保真度模拟中。

传统方法，尤其是基于深度学习的神经算子（如 DeepONet, FNO, MNO），虽然表现出色，但在理论保证、计算效率和小样本场景下的表现存在不足。

### 提出的新方法和新思路
作者提出了一种**通用的核方法**（kernel-based）**编码器-解码器框架**（encoder-decoder framework），用于学习多输入多输出算子，并在此基础上发展了名为 **KernelMO** 的方法族。

- **核心思想**：将复杂的算子学习问题分解为三个独立且可分析的步骤：
  1.  **Observation**（观测）：通过编码器 `E` 将高维或无限维的输入/输出函数（如 `u ∈ U`）压缩为低维的测量向量（如 `U ∈ Z_u`）。
  2.  **Representation & Learning**（表示与学习）：在低维的测量空间 `Z_x` 和 `Z_y` 中，使用**核方法**（如核插值或核岭回归）学习一个代理模型 `G̃`。
  3.  **Reconstruction**（重建）：通过解码器 `D` 将代理模型的预测结果从测量空间 `Z_y` 映射回原始的输出函数空间 `V`。

- **两大具体实现**（KernelMO 家族）：
  - **KernelMO-OV** (**Operator-Valued**)：直接学习映射 `G: W → {G[o]: U → V}`，即对每个任务参数 `o` 学习一个完整的算子 `G[o]`。适用于需要多次查询同一算子的场景。
  - **KernelMO-PS** (**Product-Space**)：学习等价的映射 `G': W × U → V`，即直接预测单个 `(o, u)` 对应的输出 `v`。适用于只需要孤立评估的场景。

### 相比现有方法的优势
1.  **数学严谨性与理论保证**：提供了严格的逼近理论（approximation theory）。其收敛率由最困难的单个任务决定，而非随任务总数增加而恶化，这在理论上优于许多神经网络方法。
2.  **计算高效性**：核方法的训练和推理过程是**闭式求解**（closed-form），避免了深度学习耗时的梯度下降优化。实验表明，其训练时间比最先进的神经算子快**两个数量级以上**。
3.  **轻量化与小样本优势**：该框架特别适合**中等数据量**（moderate-data regime）的应用，能以极低的计算成本获得竞争性的甚至更优的精度。
4.  **模块化与灵活性**：编码器-解码器框架是模块化的。编码器可以是任意有界线性测量（如点采样、积分），且核学习部分与编码方式解耦，易于扩展。

## 2. 核心实验方法和设置

### 使用的数据集
实验在**五个参数化偏微分方程**（parametric PDEs）上进行，这些是科学机器学习领域的标准基准：
1.  Conservation Law (守恒律)
2.  Diffusion-Reaction-Advection (扩散-反应-对流)
3.  Nonlinear Klein-Gordon (非线性克莱因-戈尔登方程)
4.  Parametric Diffusion-Reaction (参数化扩散-反应)
5.  Parametric Wave (参数化波动方程)

### 实验设置和评估指标
- **学习任务**：学习从参数函数 `o` 和初始条件 `u₀` 到完整时空解 `u(t, x)` 的算子。
- **训练数据**：
  - **Operator-Valued (OV)**：每个训练样本是一个完整的算子，由其在一组固定探针函数上的评估构成。
  - **Product-Space (PS)**：每个训练样本是一个三元组 `(o, u, G[o](u))`。
- **测试集**：包含**分布内**（in-distribution）和**分布外**（out-of-distribution, OOD）样本，后者用于评估泛化能力。
- **评估指标**：**相对 L2 误差**（relative L2 error）：
  `e_i = ||u_pred^(i) - u_target^(i)||₂ / ||u_target^(i)||₂`
  报告所有测试样本的平均误差及其标准差。

### 基线方法对比
- **经典算子学习**：
  - `KernelO`：单算子核方法。
  - `DeepONet`：标准神经算子架构。
- **多算子学习**：
  - `DeepONet-C`：将参数和输入拼接的 DeepONet 变体。
  - `MIONet`：使用张量积融合分支网络的多输入算子网络。
  - `MNO`：专门为多任务设计的神经算子，是当前的 SOTA 方法之一。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **预测精度**：
  - 在所有五个 PDE 上，**KernelMO 方法均取得了与神经算子相当或更优的预测精度**。
  - 在多个任务上，其**分布内**（in-distribution）误差比最佳的神经算子（如 MNO）**降低了1到2个数量级**。
    *   例如，在**守恒律**上，MNO 的误差为 1.23%，而 **KernelMO-OV/M** 达到了惊人的 **0.01%**。
    *   在**非线性克莱因-戈尔登方程**上，MNO 的误差为 4.63%，而 **KernelMO-OV/M** 降至 **0.21%**。
  - 在**分布外**（OOD）泛化上，KernelMO 方法也经常取得最低的误差，尤其是在 `KernelMO-OV` 配合 PCA 时。

- **计算效率**（以守恒律为例）：
  - **训练时间**：KernelMO-OV 的训练时间**小于0.5秒**，而最快的神经算子（DeepONet-C）需要约 **250秒**，速度提升了**超过500倍**。
  - **推理时间**：KernelMO-OV 的推理时间约为 **0.04毫秒/样本**，比神经算子快 **4-9倍**；若使用 PCA，可进一步加速至 **0.004毫秒/样本**，速度提升达 **40-80倍**。
  - **消融实验**（PCA的影响）：
    - **优点**：PCA 极大地降低了计算复杂度（训练和推理时间），并常常能提升 OOD 泛化能力，起到正则化作用。
    - **缺点**：在某些任务上，PCA 会带来轻微的分布内精度损失（例如，守恒律上从 0.01% 降至 1.77%），但这种权衡通常是值得的。

### 总结对比
| 方法 | 预测精度 | 训练效率 | 推理效率 | 理论保证 |
| :--- | :--- | :--- | :--- | :--- |
| **KernelMO-OV/PS** | **最优或接近最优** | **极高 (>>100x)** | **极高 (>10x)** | **强** |
| MNO / DeepONet-C | 良好 | 低 | 中等 | 弱 |

## 4. 关键结论和发现

### 主要发现
1.  **核方法是神经算子的强大替代品**：在科学机器学习领域，特别是对于中等数据量的算子学习问题，**核方法**（kernel methods）凭借其**数学严谨性、计算高效性和卓越的预测精度**，构成了深度神经网络的一个极具吸引力的轻量级替代方案。
2.  **编码器-解码器框架的有效性**：提出的框架成功地将无限维算子学习问题转化为有限维空间中的标准核学习问题，实现了**理论分析、计算效率和预测性能的良好平衡**。
3.  **KernelMO 的优越性**：无论是 **KernelMO-OV** 还是 **KernelMO-PS**，都显著超越了现有的神经算子基线，尤其是在**训练和推理速度**上实现了质的飞跃，同时保持了顶尖的预测精度。

### 方法的局限性
1.  **核方法的固有局限**：核方法的计算复杂度通常与训练样本数 `N` 的立方成正比 (`O(N³)`)，这限制了其在超大规模数据集上的应用。
2.  **对核函数选择的敏感性**：性能依赖于核函数（如 RBF 或 Matérn）及其超参数的选择。虽然文中展示了良好的结果，但最优选择可能因问题而异。
3.  **编码器的设计**：目前的编码器（如 PCA）是预定义的。如果能将编码器也作为可学习的部分，可能会进一步提升性能。

### 未来工作方向
1.  **扩展到算子空间**：开发直接在算子空间（而非函数空间）上学习的理论，以处理更广泛的算子学习问题。
2.  **自适应编码器**：研究如何从数据中学习最优的编码器和解码器，而不是使用固定的 PCA。
3.  **无限维观测空间**：将框架扩展到观测空间本身也是无限维的情况。
4.  **不确定性量化**：利用核方法与高斯过程（Gaussian Process）的紧密联系，进行更深入的不确定性量化研究。
5.  **跨模态迁移**：探索所学代理模型在不同观测系统（如不同传感器布局）之间的迁移能力。

</details>

---

### 5. [CLAIM: Leading Open-domain Active Clarification of Large Language Models with Uncertainty Measurement](https://arxiv.org/abs/2608.11631)

**Authors**: Kuangzhao Yang, Ziliang Zhao, Zhicheng Dou  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.11631v1  

#### Abstract
In open-domain human-computer interaction scenarios, large language models (LLMs) frequently encounter user queries that are ambiguous or incomplete. In such cases, directly producing an answer often leads to overgeneralized, erroneous, or low-information responses. In contrast, asking clarifying qu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CLAIM: Leading Open-domain Active Clarification of Large Language Models with Uncertainty Measurement

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在开放域（open-domain）人机交互中，用户查询常具有**模糊性**（ambiguity）或**信息不完整**（underspecification），直接生成回答容易导致泛化、错误或低信息量的响应。现有方法依赖大量人工标注数据或偏好对齐来判断：
- 是否需要澄清（when-to-ask）
- 应该澄清哪个维度（what-to-ask）

这带来了高昂的标注成本，并限制了模型的泛化能力。

### 提出的新方法与新思路
作者提出 **CLAIM**（Leading Open-domain Active Clarification of LLMs with Uncertainty Measurement），一个基于**不确定性度量**的主动澄清框架，其核心思想是：

> 利用多个异构大语言模型（LLMs）对同一查询生成答案时的语义分歧程度（semantic disagreement）作为“不确定性”信号，自动判断是否需要澄清。

#### 主要创新点：
1. **无需人工标注的合成数据构建**  
   - 通过多模型生成答案 → 语义聚类 → 计算熵（entropy）来量化查询不确定性。
   - 高熵表示答案分歧大，说明查询模糊，需澄清；低熵则可直接作答。
   - 此过程完全自动化，无需人类标注或外部偏好信号。

2. **熵驱动的合成数据生成流水线**  
   - 结合熵估计、LLM语义判断与冲突仲裁机制，自动生成高质量的澄清决策训练数据。
   - 显式建模澄清前后不确定性变化（information gain），提升问题选择质量。

3. **统一的决策学习范式**  
   - 将澄清任务形式化为条件生成策略学习问题。
   - 采用 **SFT + GRPO** 两阶段训练：
     - **Supervised Fine-Tuning (SFT)**：从合成数据中学习基本澄清行为。
     - **Group-Relative Policy Optimization (GRPO)**：优化高不确定性边界上的决策稳定性。

4. **模块化解耦设计**  
   - 区分 **CLAIM-Agent**（离线数据构造代理）与 **CLAIM**（在线推理单模型），实现训练与部署分离，保证部署效率。

### 相比现有方法的优势
| 维度 | CLAIM | 现有方法（如 ClariLM） |
|------|-------|------------------------|
| 数据需求 | 完全无监督，仅用约10k合成样本 | 依赖~120k人工标注+偏好数据 |
| 泛化性 | 跨领域表现稳定 | 易过拟合特定任务（如IN3） |
| 成本 | 构造阶段并行可扩展，部署为单模型 | 训练与标注成本高 |
| 决策可控性 | 显式建模“是否澄清”与“澄清哪一维” | 多隐含于生成过程中 |

---

## 2. 核心实验方法和设置

### 使用的数据集（全部仅用于评估）
| 数据集 | 描述 |
|--------|------|
| **ClariLM-test** | 自动构建的合成澄清数据集，围绕潜在缺失信息维度组织，适合控制条件下评估澄清决策。 |
| **IN3 (Intention-in-Interaction)** | 基于真实模糊指令的任务导向对话数据集，提供任务歧义、缺失细节及其重要性的系统标注。 |
| **CLAMBER** | 通用开放域澄清基准，覆盖广泛主题，评估模型识别自然语言查询中的不确定性和生成高质量澄清问题的能力。 |

> ⚠️ 所有数据集的训练集均未使用，仅测试集用于评估。

### 实验设置与评估指标

#### 评估维度
分为两个互补方面：

| 类别 | 指标 | 说明 |
|------|------|------|
| **Clarification Necessity**（是否需要澄清） | `Accuracy`, `F1` | 二分类任务：是否应发起澄清 |
| **Clarifying Question Quality**（澄清问题质量） | `CDA`（Clarification Dimension Accuracy） | 生成问题是否聚焦正确的信息缺失维度（由独立LLM评判） |
|  | `CQSS`（Clarifying Question Semantic Similarity） | 与标准问题之间的嵌入余弦相似度（使用 Qwen3-Embedding-8B） |

#### 基线方法对比
| 类型 | 模型列表 |
|------|---------|
| **零样本LLM** | Llama-3.1-8B, Qwen3-8B/14B/32B, DeepSeek-V3 |
| **推理增强模型（LRM）** | QwQ-32B, DeepSeek-R1 |
| **SFT微调模型** | 多种变体（见下文消融实验） |
| **先前SOTA方法** | ClariLM [38]（基于大规模监督+偏好优化） |
| **特殊基线** | CLAIM-Agent（直接运行完整pipeline，无训练） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | ClariLM-test Acc/F1/CDA/CQSS | IN3 Acc/F1/CDA/CQSS | CLAMBER Acc/F1/CDA/CQSS |
|-------|-------------------------------|---------------------|--------------------------|
| **CLAIM (ours)** | **81.85 / 84.97 / 56.79 / 69.54** | **87.04 / 92.55 / 63.16 / 72.23** | **65.18 / 68.10 / 63.71 / 65.47** |
| ClariLM [38] | 81.25 / 85.48 / 52.50 / 63.55 | 89.72 / 94.36 / 66.32 / 72.68 | 64.23 / 67.61 / 62.89 / 65.36 |
| SFT-Full | 79.60 / 82.86 / 55.17 / 67.38 | 85.19 / 91.40 / 71.58 / 74.65 | 61.99 / 62.40 / 62.27 / 66.91 |
| CLAIM-Agent | 71.95 / 91.58 / 51.94 / 63.67 | 85.19 / 91.58 / 68.42 / 71.39 | 59.62 / 66.63 / 58.15 / 60.43 |

> ✅ CLAIM 在多数指标上达到 SOTA 或接近最优水平，尤其在 **CLAMBER** 上全面超越 ClariLM。

### 与基线方法的对比结果
- **优于所有零样本LLM和LRM**：表明单纯的语言能力或推理不足以可靠地进行澄清决策。
- **优于 ClariLM**：尽管训练数据量仅为后者的 ~1/12（10k vs 120k），但在多个指标上持平甚至反超，验证了**数据高效性**。
- **优于各SFT变体**：证明所提组件的有效性。
- **CLAIM > CLAIM-Agent**：说明 GRPO 训练能有效将复杂策略内化到单一模型中，优于直接执行agent流程。

### 消融实验结果

#### （1）不同澄清判断信号的影响
| 模型 | ClariLM-test Acc / CDA |
|------|------------------------|
| SFT-Entropy only | 74.60 / 53.88 |
| SFT-LLM only | 73.40 / 53.31 |
| SFT-Full（两者结合） | **79.60 / 55.17** |

➡️ 表明熵信号与LLM语义判断**互补**，单独使用任一信号都不够鲁棒。

#### （2）信息增益（IG）选择的作用
| 模型 | ClariLM-test CDA / CQSS |
|------|------------------------|
| SFT-without IG | 43.05 / 52.17 |
| SFT-Full（含IG） | **55.17 / 67.38** |

➡️ 移除IG导致澄清质量问题显著下降，说明基于**不确定性减少**的选择机制至关重要。

#### （3）GRPO 的作用
| 模型 | ClariLM-test Acc / CDA | CLAMBER Acc |
|------|------------------------|-------------|
| SFT-Full | 79.60 / 55.17 | 61.99 |
| CLAIM（+GRPO） | **81.85 / 56.79** | **65.18** |

➡️ GRPO 进一步提升了决策稳定性和泛化能力，尤其是在高不确定性边界处。

#### （4）跨域泛化能力
| 模型 | IN3 Acc | ClariLM-test Acc | CLAMBER Acc |
|------|--------|------------------|--------------|
| SFT-IN3（专训IN3） | **89.81** | 74.35 | 56.00 |
| SFT-Full（CLAIM数据） | 85.19 | **79.60** | **61.99** |

➡️ SFT-IN3 在IN3上更强，但在其他数据集上明显退化；而 SFT-Full 更均衡，体现 CLAIM 合成数据的**强泛化性**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **多模型答案分歧（semantic disagreement）是有效的内在不确定性信号**，可用于自动触发澄清，无需人工标注。
2. ✅ **熵 + LLM语义判断 + 冲突仲裁** 的组合机制能更准确识别真正需要澄清的查询。
3. ✅ **信息增益（IG）可有效指导澄清问题选择**，优先减少最大不确定性的维度。
4. ✅ **CLAIM 实现了高性能且低成本的澄清学习**，以极少量合成数据媲美甚至超越依赖大规模标注的方法。
5. ✅ **GRPO 能显著提升决策边界附近的稳定性**，使模型在模糊案例中更具一致性。

### 方法的局限性
- **依赖多个异构LLM进行离线数据构建**：虽然部署时只需单模型，但数据构造阶段计算开销较大（每千样本约5.7M token）。
- **当前仅支持单轮澄清**：无法处理多轮交互中的状态追踪与长期规划。
- **模拟用户反馈仍具理想化假设**：实际用户回答可能更嘈杂或偏离预期。
- **未考虑实时性要求**：CLAIM-Agent 的多步调用不适合低延迟场景。

### 未来工作方向
- 扩展至 **multi-turn interaction**，引入对话状态跟踪（dialogue state tracking）和未来回合规划。
- 探索更高效的不确定性估计方法（如 SEPs、Cleanse）以降低构造成本。
- 引入真实用户交互闭环，实现在线自我改进（online self-refinement）。
- 将 CLAIM 思路应用于其他主动理解任务，如代码补全、工具调用前的参数确认等。

---

> 🔚 **总结一句话**：  
> CLAIM 提出了一种**基于模型间不确定性分歧的全自动澄清学习框架**，实现了无需人工标注的高质量主动澄清能力，在性能、成本与泛化性之间取得了良好平衡，为构建更智能、自适应的人机交互系统提供了新路径。

</details>

---

### 6. [LazyTrain: Limited-resource Allocation toward Zero-waste Yield Optimization in Large Language Model Training](https://arxiv.org/abs/2608.11919)

**Authors**: Xiaojun Wu, Cehao Yang, Honghao Liu, Xueyuan Lin, Xuhui Jiang, Chengjin Xu, Jia Li, Jian Guo  
**Category**: cs.CL  
**Published**: 2026-08-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.11919v1  

#### Abstract
Training large language models on limited hardware is increasingly a scheduling problem across GPU compute, host memory, PCIe transfer, and storage bandwidth. Existing offloading systems reduce GPU residency, and MegaTrain shows that a CPU-master layer-streaming executor can train large models on a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LazyTrain 论文核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）在有限硬件资源下的训练已成为一个复杂的**调度问题**，涉及 GPU 计算、主机内存（CPU DRAM）、PCIe 传输和存储带宽之间的权衡。现有系统如 MegaTrain 虽然通过 CPU 主控和层流式执行器（layer-streaming executor）实现了单 GPU 上大模型的训练，但其采用的**固定检查点策略和放置启发式**导致通信无法有效隐藏在计算窗口中，从而暴露在关键路径上，限制了吞吐量。

### 提出的新方法与创新
本文提出 **LazyTrain**，一种面向有限资源 LLM 训练的优化框架，其核心创新在于将训练过程建模为一个**混合整数规划（Mixed-Integer Programming, MIP）调度问题**，联合优化以下四个维度：
- **Checkpoint Selection**：动态选择激活值检查点边界。
- **Activation Placement**：决定每个检查点存放于 GPU HBM、CPU DRAM 还是 NVMe。
- **Recomputation**：确定哪些层需要重新计算以节省内存。
- **Communication Overlap**：安排 CPU-GPU-NVMe 数据传输时间，使其尽可能被计算掩盖。

此外，LazyTrain 引入了一个**耦合的 Hybrid 8-bit 算子**，将 8-bit 优化器状态压缩与快速梯度裁剪（fast gradient clipping）结合，前者减少 CPU 内存占用，后者抵消因量化带来的额外 CPU 更新开销。

### 相比现有方法的优势
- **超越启发式调度**：相比 MegaTrain 的固定策略，LazyTrain 通过求解器搜索最优调度策略，显著提升吞吐量。
- **零浪费通信设计**：目标函数最小化的是“增量通信暴露”而非最大化卸载量，确保所有激活相关的通信都被隐藏在计算窗口内。
- **系统级协同优化**：Hybrid 8-bit 算子不是独立组件，而是与调度机制协同设计的整体优化方案。

---

## 2. 核心实验方法和设置

### 数据集
- **主实验数据集**：MetaMathQA，用于 Qwen3.6-27B 模型的监督微调。
- **数据划分**：70% 训练集 / 30% 测试集。
- **序列长度**：1024。
- **训练周期**：1 epoch。

### 实验设置
- **硬件平台**：
  - **H800**：单卡 80GB HBM，PCIe Gen5 x16，本地双 NVMe SSD。
  - **RTX 3090**：单卡 24GB HBM。
- **模型规模**：从 Qwen2.5-3B 到 Qwen3.6-27B。
- **批大小（Batch Size）**：H800 上最高达 72；RTX 3090 上测试最大可行批大小。
- **精度**：bfloat16。

### 评估指标
- **吞吐量**：持续 TFLOPS 和 tokens/s。
- **内存使用**：峰值 GPU 内存、CPU 内存。
- **可扩展性**：最大可行批大小（Max Feasible Batch Size）。
- **训练质量**：Exact Match 准确率（在完整测试集上）。

### 基线方法对比
- **MegaTrain**：作为主要对比基线，代表当前最先进的单 GPU 层流式训练框架。
- **ZeRO-3 Offload**：传统分布式训练中的 offload 方法。
- **LazyTrain-MILP**：移除 MILP 调度器的变体。
- **LazyTrain-Hybrid 8-bit**：移除 Hybrid 8-bit 算子的变体。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen3.6-27B on H800）
| 指标 | MegaTrain | LazyTrain | 提升幅度 |
|------|-----------|-----------|----------|
| **持续 TFLOPS** | 176.90 | **219.95** | **+24.3%** |
| **Tokens/s** | 1075.8 | **1361** | **+26.5%** |
| **峰值 GPU 内存** | 60.40 GB | **68.84 GB** | — |
| **Exact Match 准确率** | 95.33% | **95.42%** | +0.09% |

> ✅ 在相同模型、数据、批大小下，LazyTrain 实现约 **1.24× 吞吐提升**。

### 与其他模型和硬件的对比
- **H800 全系列（3B–27B）**：LazyTrain 在所有规模上均优于 MegaTrain，TFLOPS 提升范围为 1.24× 左右。
- **RTX 3090**：在所有模型规模上，LazyTrain **将最大可行批大小提高 1 单位**，并实现更高的 token 吞吐。

### 消融实验结果（Ablation Study）
| 方法 | 持续 TFLOPS | 相对 MegaTrain 提升 |
|------|-------------|---------------------|
| **MegaTrain baseline** | 176.90 | — |
| **LazyTrain (完整)** | **219.95** | **+24.3%** |
| **LazyTrain-MILP**（移除 MILP 调度） | 193.17 | +9.2% |
| **LazyTrain-Hybrid 8-bit**（移除 Hybrid 8-bit） | 219.29 | +24.0% |

#### 消融分析
- **MILP 调度是主导因素**：移除后性能下降 **12.2%**，证明联合优化调度是性能增益的核心来源。
- **Hybrid 8-bit 影响较小但必要**：仅下降 0.3%，说明其作用是“保底”优化，防止 8-bit 量化引入 CPU 开销。
- **通信暴露为 0ms**：最终调度策略成功将所有激活相关通信完全隐藏。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 训练瓶颈本质是调度问题**：在有限资源下，如何协调计算、内存层级和通信带宽是决定效率的关键。
2. **MILP 可有效建模该问题**：通过将检查点、放置、重计算和通信重叠联合建模为 MIP 问题，LazyTrain 找到了远超启发式策略的更优解。
3. **通信隐藏优于单纯卸载**：目标应是最小化“暴露通信”，而非最大化卸载量，这是实现高吞吐的关键。
4. **组件需协同设计**：Hybrid 8-bit 算子展示了内存压缩与计算优化必须成对出现才能真正提升端到端性能。

### 方法的局限性
- **离线调度**：调度策略在训练前一次性求解，无法适应运行时带宽波动。
- **单 GPU 设置**：实验仅限于单 GPU 场景，未扩展至多 GPU 或多节点。
- **评估集复用**：30% 的测试集也用于周期性 loss 监控，最终准确率未在完全独立的测试集上验证。
- **未测量运行时停顿**：缺乏 per-step stall time 的细粒度仪器监控。

### 未来工作方向
- **在线自适应调度**：开发能感知运行时变化的动态调度器。
- **多 GPU / 多节点扩展**：将调度框架推广至分布式训练环境。
- **支持 MoE 架构**：当前调度为层级别，未来需支持专家级别（expert-level）的放置与通信优化。
- **更精细的资源建模**：纳入更多硬件特性（如缓存行为、NUMA 架构）进行建模。

> 🔗 **开源地址**：https://github.com/DataArcTech/LazyTrain

</details>

---

### 7. [LinearKV: One Cached State Suffices for Position-Independent Caching in Hybrid LLMs](https://arxiv.org/abs/2608.11231)

**Authors**: Yirui Liu, Ruoling Qi, Longwen Wang, Xuaner Wu, Jian Chen, Yuxin Jin, Jiawei Shao, Xuelong Li  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.11231v1  

#### Abstract
LLM serving is increasingly accelerated by position-independent caching (PIC). Existing PIC methods, however, are built for full-attention models, where a token-indexed KV cache underlies its core operations: matching reusable token chunks, concatenating their KV entries, and selectively recomputing...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LinearKV: One Cached State Suffices for Position-Independent Caching in Hybrid LLMs**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
- **背景**：Position-Independent Caching (PIC) 已被广泛用于加速全注意力（full-attention）大语言模型（LLM）的服务效率，通过缓存并重用独立预填充的 token chunks 来减少重复计算。
- **挑战**：然而，混合架构 LLMs（hybrid LLMs），如 Mamba-2 和 Gated DeltaNet (GDN)，结合了少量全注意力层与大量基于线性递归（linear recurrence）的层。这些递归层只维护一个固定大小的状态 `S`，而非每个 token 对应的 KV 缓存，因此传统基于 token-KV 拼接的 PIC 方法无法直接应用。

> **核心问题**：如何将 PIC 扩展到 hybrid LLMs？是否必须重新设计整个框架？

---

### 🚀 **提出了什么新方法或新思路**
提出 **LinearKV** —— 一种无需训练、兼容现有 PIC 方法的 hybrid-PIC 框架，其关键创新在于：

#### **（1）解耦初始化（Decoupled Initialization）**
- 全注意力层（FA layers）：沿用传统方式，直接拼接各 chunk 的 KV 缓存。
- 线性递归层（Linear layers）：引入一个统一函数 $ f $ 将 K 个匹配 chunk 的局部状态 $\{S_{\text{local}}^{(k)}\}$ 映射为单个初始状态 $S_{\text{init}} = f(\cdot)$。
- 这种“分而治之”的策略使得 LinearKV 可以无缝集成任何现有的 PIC selector（如 CacheBlend、EPIC、ProphetKV），仅需替换状态初始化逻辑。

#### **（2）关键发现：一个缓存状态就足够了（One Cached State Suffices）**
- 以往认为应代数上精确组合所有 K 个 chunk 的状态（即 exact composition），以逼近完整上下文状态。
- 本文发现：**这种“数学上正确”的 exact composition 在某些架构（如 Mamba-2）下反而崩溃**。
- 相反，**只需使用最后一个 chunk 的缓存状态作为初始化（last-block init）即可达到更优甚至鲁棒得多的效果**。
- 更惊人的是：即使是随机选择一个 chunk 的状态（random-block init），效果也几乎一样好 —— 表明关键是“单一来源”而非具体哪个块。

---

### ⚖️ **相比现有方法的优势**

| 维度 | LinearKV | HYPIC / Exact Composition |
|------|--------|--------------------------|
| **通用性** | ✅ 支持多种 hybrid 架构（Mamba-2, GDN） | ❌ 在 Mamba-2 上严重退化 |
| **性能** | ✅ 质量更高（尤其在 Mamba-2 上提升巨大） | ❌ 多源构造导致误差累积 |
| **效率** | ✅ 更低 TTFT（time-to-first-token），节省 5–17% 开销 | ❌ 需在线合成 transition 矩阵，增加延迟 |
| **兼容性** | ✅ 即插即用，复用所有现有 PIC selector | ✅ 同样支持，但质量受限 |

> **一句话优势**：LinearKV 不仅实现了对 hybrid LLMs 的高效 PIC，还揭示了一个反直觉但重要的工程原则 —— “简单胜于复杂”。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
- **LongBench**（多任务长文本理解基准）
  - QA 子集：HotpotQA (HQA), 2WikiMQA, MuSiQue, NarrativeQA, Qasper
  - Summarization 子集：QMSum, GovReport, MultiNews
- **RULER**（长上下文能力评测）
  - 包括 needle-in-a-haystack、关键词提取、变量追踪等任务
  - 测试长度覆盖 **8K 和 32K tokens**

---

### ⚙️ **实验设置和评估指标**

#### **模型**
- **Granite-4.0-H-Tiny (Mamba-2)**：7B MoE，36 层 Mamba-2 + 4 FA（90% 递归）
- **OLMo-Hybrid-7B-Instruct (GDN)**：7B，24 GDN + 8 FA（75% 递归）
- **Qwen3.6-27B (GDN)**：27B，48 GDN + 16 FA（75% 递归）

#### **评估指标**
| 任务类型 | 主要指标 | 辅助指标 |
|---------|--------|--------|
| LongBench QA | token-level F1（Avg-F1） | %-of-full（相对 full recompute 的恢复率） |
| LongBench Summarization | ROUGE-L | %-of-full |
| RULER | string-match recall（按子任务平均） | %-of-full |

#### **对比方法**
- **Full recompute**（r=1）：无缓存，完全重计算（上限）
- **Naive reuse**（r=0）：不重算，仅用 last-block 初始化（下限）
- **CacheBlend / EPIC / ProphetKV + EX**：HYPIC 式 exact composition 初始化
- **CacheBlend / EPIC / ProphetKV + LB**：LinearKV 的 last-block 初始化（本文方法）

> 所有方法在相同 selector 和相同重计算位置集合 $ \mathcal{U} $ 下比较，确保公平。

---

## 3. **主要实验结果和性能指标**

### 🔢 **关键性能数据汇总**

#### ✅ **在 GDN 模型上（OLMo, Qwen）**
- **Exact Composition 与 Last-Block 效果相当**
  - 平均差距 < 0.013 Avg-F1，且正负交替，说明无显著差异
  - 如 OLMo 上 ProphetKV + EX vs LB：34.8% vs 36.1% of full quality
- **均能恢复高达 92% 的 full quality**（如 OLMo @ ProphetKV on RULER）

> ✅ 结论：exact composition 在 GDN 上有效且稳定。

#### ❌ **在 Mamba-2 模型上（Granite）**
- **Exact Composition 完全失效！**
  - EPIC + EX：仅恢复 **46.6%** 的 full quality（Avg-F1 = 0.145）
  - CacheBlend + EX：仅 **6.3%**
- **Last-Block 初始状态大幅提升性能**
  - EPIC + LB：恢复 **86.8%**（Avg-F1 = 0.270）
  - CacheBlend + LB：提升至 **11.8%**
- **相对提升达 +40pp 以上**

> ⚠️ 特别注意：改变初始化方式（非 selector 或重算位置）就能带来质变！

#### 📉 **RULER 与 Summarization 验证一致性**
- RULER-32K（4倍上下文）仍保持相同趋势：
  - Granite 上 EX 平均 recall ≈ 0.07，而 LB 达到 ≈ 0.61
- Summarization（生成任务）中 LB 也显著优于 EX

---

### ⏱️ **效率表现（TTFT）**

| 模型 | 上下文长度 | 方法 | TTFT (ms) | 相对于 full (%) |
|-----|-----------|------|-----------|----------------|
| Granite (Mamba-2) | 32K | Full | 782 | 100% |
| | | Last-Block (LB) | **377–482** | **~48–62%** |
| | | Exact (EX) | **429–543** | **~55–69%** |
| OLMo (GDN) | 32K | Full | 3843 | 100% |
| | | LB | **1810–1955** | **~47–51%** |
| | | EX | **1967–2109** | **~51–55%** |

> ✅ **Last-Block 比 Exact Composition 快 5–17%，端到端延迟更低**

---

### 🔍 **消融实验结果**

#### （1）**更多重计算能否挽救 Exact Composition？**
- 固定 selector（EPIC），扫频 $ r \in [0.03, 0.4] $
- 结果：Exact Composition 在 Granite 上始终停滞在 **41–52%** 恢复率
- 而 Last-Block 随 $ r $ 上升稳步提高至 **76–89%**
> ❗ **Bad initializer cannot be rescued by more recomputation**  
> 初始化设定了性能天花板，后续修复无法弥补

#### （2）**是“最后一块”特殊吗？还是“单一来源”才是关键？**
- 加入 **Random-Block Initializer**（随机选一个 chunk 的状态）
- 结果：在 Granite 上，Random-Block 与 Last-Block 性能几乎一致（差值 < 0.014 Avg-F1）
> ✅ **真正起作用的是 single-source construction，而不是特定位置**

#### （3）**与 HYPIC 的端到端对比（使用其 seam selector）**
- 使用 HYPIC 推荐的固定窗口重算策略（seam selector）
- 仅更换初始化方式（EX vs LB）
- 结果：LB 在 Granite 上从 47.6% 提升到 67.8%（+20pp）
> ✅ LinearKV 的初始化可直接增强 HYPIC 类方法

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **解耦初始化使 hybrid-PIC 成为可能**
   - FA 层继续拼接 KV，Linear 层单独处理状态初始化
   - 可即插即用现有 PIC selector，极大降低部署成本

2. **Exact composition 并非万能，反而具有架构脆弱性**
   - 在 GDN 中可行，在 Mamba-2 中因 scalar decay 导致误差逐层累积
   - 错误来源于 deep operator 的 context-mismatch（孤立预填充 vs 联合上下文）

3. **一个缓存状态足以实现高性能 PIC**
   - 单一来源避免了多路径误差传播
   - “Last-block” 是自然、零成本的选择（无需额外决策）
   - Random-block 实验证明：关键是“单源”，不是“最近”

4. **性能与效率双优**
   - 质量更高（尤其在 Mamba-2 上翻倍）
   - 延迟更低（TTFT 减少 5–17%）

---

### ⚠️ **方法的局限性**

- 当前仅适用于 **state-preserving recurrent layers**（如 Mamba/GDN），不适用于其他隐状态压缩机制
- 假设 chunk 匹配准确；若 chunk 匹配错误，single-source 初始化会放大错误
- 实验集中在 retrieval-augmented generation (RAG) 场景，未覆盖动态对话流等复杂场景

---

### 🔮 **未来工作方向**

1. **探索更智能的 single-source selection policy**
   - 是否可以根据 query 内容选择最有代表性的 chunk？
2. **扩展至更多 hybrid 架构**
   - 如 RWKV、RetNet 等不同 recurrence 形式的模型
3. **结合编译器优化进行 host-device 缓存传输优化**
4. **研究 batched serving 下的 cache 共享机制**
5. **理论分析为何 single-source 在误差控制中更具鲁棒性**

---

> 💡 **最终启示**：在系统设计中，“数学完美”未必等于“实际最优”。LinearKV 揭示了一条新的工程哲学：**simplicity, robustness, and efficiency can triumph over algebraic elegance**。

</details>

---

### 8. [Glance, Scrutinize, and Think: Advancing Video Anomaly Detection from Training-Free to Agentic Reasoning](https://arxiv.org/abs/2608.11260)

**Authors**: Shibo Gao, Peipei Yang, Xu-Yao Zhang, Linlin Huang  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.11260v1  

#### Abstract
Video Anomaly Detection (VAD) aims to identify anomalous events and localize their temporal intervals. Existing approaches exhibit a "when-what" dissociation: traditional DNN-based methods localize when anomalies occur but lack semantic understanding, whereas LLM-based methods explain what happens b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Glance, Scrutinize, and Think: Advancing Video Anomaly Detection from Training-Free to Agentic Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Video Anomaly Detection (VAD)** 领域存在“**when-what dissociation**”问题：
- **传统 DNN-based 方法** 能定位异常发生的时间（“when”），但缺乏语义理解能力；
- **新兴 LLM-based 方法** 能解释发生了什么（“what”），但忽略了精确的时序定位。

这种割裂源于缺少一个统一的推理范式来融合时间定位与语义理解。

### 🚀 提出的新方法与思路

#### （1）**Glance then Scrutinize (GtS)**：训练免费的全局到局部推理框架
- **灵感来源**：人类观察监控视频的方式——先快速扫视形成假设，再聚焦可疑片段进行细查。
- **实现方式**：
  - **Glance阶段**：利用静态（如异常类别列表）和动态文本引导（如LLM生成的提示词）对视频粗略划分高概率异常区域。
  - **Scrutinize阶段**：在这些区域内进行非均匀采样帧分析，并结合多段上下文整合信息，提升细粒度理解和精准定位。
- **优势**：无需任何模型训练，即可实现高效、准确的联合检测与理解，适用于实时场景。

#### （2）**Tool-Augmented Agentic VAD**：基于工具调用的智能体式推理方法
- **动机**：GtS依赖冻结模块（如CLIP、固定prompt），性能受限于外部组件能力。
- **核心设计**：
  - 构建一个能自主调用 `crop_video` 工具的 **Multimodal Large Language Model (MLLM)**。
  - 模型通过“观察 → 假设 → 截取 → 分析 → 自我纠正”的闭环进行迭代推理。
- **训练策略**：
  - **Cold-start SFT**：监督微调，教会模型基本的工具使用和异常识别。
  - **Agentic RL**：采用 **GRPO** 强化学习，以联合奖励函数优化决策行为。

#### （3）**VAGU-T 数据集**：首个支持“思考链+工具调用”的VAD基准
- 在原有 VAGU 基础上扩展，新增：
  - 7,567个真实世界异常视频（覆盖21类）
  - 人工验证的时序标注、语义描述、QA对
  - **Chain-of-Thought Tool-Calling Traces**（用于训练智能体）
- 支持从评估到训练的完整闭环。

#### （4）**JeAUG 评价指标**：联合评估语义理解与时序定位
- 公式：  
  $$
  \text{JeAUG} = \min(\gamma \cdot F(\text{IoU}), 1) \cdot \text{Score}_{A.U.}
  $$
- 同时衡量：
  - **A.U. Score**：由LLM-as-a-Judge打分（主体、事件过程、影响等维度）
  - **A.G. Score**：基于IoU并引入人类偏好校准的时序得分
  - **长度补偿因子 γ**：长视频更难定位，适当加分

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **主数据集**：**VAGU-T**（本文提出）
  - 视频数量：7,567
  - 类别数：21（涵盖犯罪、自然灾害、交通事故、动物伤害等）
  - 平均长度：2,716帧（约90秒 @30fps）
  - 注释内容：异常类别、语义解释、精确时间区间 `[ts, te]`、QA对、CoT工具调用轨迹
  - 划分：1,217个样本作为测试集，其余用于训练/构造轨迹

### ⚙️ 实验设置
- **硬件配置**：
  - GtS：14×A6000 GPU
  - SFT & RL训练：24×Alibaba T-Head PPU (96GB)
- **模型基础**：
  - GtS使用多种VLM（如Qwen2.5-VL、VideoChatGPT）作为backbone
  - Agentic模型基于 **Qwen2.5-VL-7B-Instruct**

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **A.U. Score** | 1–10分制，由LLM评估生成描述的质量 |
| **JeAUG** | 本文提出的联合指标（0–10），综合A.U.与A.G. |
| **QA Accuracy (%)** | 多选题正确率（仅适用于支持该协议的方法） |
| **FPS** | 推理速度（帧每秒），反映实时性 |

### 🔁 基线方法对比
分为两类：
1. **Frame/Segment-wise Pipeline**：
   - LAVAD、SUVAD：逐段处理视频，计算开销大
2. **Training-free Direct VQA/VTG Models**：
   - 如 VideoChatGPT + TimeChat、Qwen2.5-VL + VTimeLLM 等组合

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 5）

| 方法 | A.U. | JeAUG | QA (%) | FPS |
|------|-----|--------|-------|-----|
| **LAVAD** | 5.52 | 4.47 | – | 0.24 |
| **SUVAD** | 5.73 | 4.58 | – | 0.19 |
| **Qwen2.5-VL-7B + TC** | 3.61 | 2.28 | 68.0 | 185 |
| **Ours: GtS (same backbone)** | **5.50** | **4.04** | 73.5 | 61 |
| **Ours: SFT** | 6.62 | 4.48 | – | 161 |
| **Ours: Agentic RL** | **7.35** | **5.91** | – | **148** |

> ✅ **Agentic RL 是所有方法中性能最强且推理最快者**

### 🔍 与基线方法对比结果
- **相比直接VQA/VTG方法**：
  - GtS将 JeAUG 从 ~2.3 提升至 >4.0，同时保持 FPS > 30（满足实时要求）
  - 显著优于原始 pipeline（如 Qwen2.5-VL-7B: 2.28 → 4.04）
- **相比 exhaustive pipeline（LAVAD/SUVAD）**：
  - 性能相当甚至更高（JeAUG 5.91 vs 4.58），但速度快近 **1000倍**（148 vs 0.19 FPS）
- **Agentic RL vs SFT**：
  - A.U. 提升 +0.73，JeAUG 提升 +1.43
  - 表明强化学习有效激发了超越模仿的推理能力

### 🔧 消融实验结果

#### （1）GtS 模块消融（Table 6）
移除任一组件均导致性能下降：
- 移除动态文本引导：A.U. 5.50 → 5.27
- 移除非均匀采样：5.50 → 5.38
- 移除跨段上下文理解：5.50 → 5.41  
✅ 验证了各模块的有效性

#### （2）不同采样策略比较（Table 7）
- 统一分割视频为7段：A.U. 3.61 → 4.02（边际提升）
- 使用 GtS 动态分割：3.61 → **5.50**  
✅ 说明 GtS 的智能划分远胜于暴力分块

#### （3）训练与奖励消融（Table 8）
| 配置 | A.U. |
|------|-----|
| SFT only | 6.62 |
| SFT + RL (无 Rtime) | 6.93 |
| SFT + RL (含 Rtime) | **7.35** |

✅ **Temporal Grounding Reward (Rtime)** 对最终性能至关重要

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **“Glance-Scrutinize-Think” 范式有效统一了 VAD 中的 “when” 与 “what”**：
   - 人类启发的认知流程可被成功建模为算法框架。
2. **GtS 实现了训练免费下的高性能平衡**：
   - 不需训练即可显著超越现有方法，在精度与速度间取得良好折衷。
3. **Agentic RL 模型突破性能上限**：
   - 内化推理循环后，模型不仅能自我修正错误定位，还能主动聚焦细微异常（如 Shoplifting、TrafficViolation）。
4. **JeAUG 更贴近人类判断标准**：
   - 单一指标（如AUC或BLEU）无法全面评估VAD系统；JeAUG通过双维度加权提供更公平比较。

### ⚠️ 方法局限性
- **对极短或弥散性异常仍不敏感**：
  - 如 “Smoke” 类别表现最差（A.U. 仅5.50），因其视觉特征模糊且稀疏。
- **Agentic RL 训练成本高**：
  - 尽管推理快，但需要大量高质量 CoT 工具调用轨迹进行训练。
- **依赖强大 MLLM 和工具接口**：
  - 当前方法建立在 Qwen2.5-VL 等先进模型之上，难以迁移到轻量级系统。

### 🔮 未来工作方向
1. 扩展 VAGU-T 至更多复杂场景（如无人机监控、医疗异常检测）
2. 设计更高效的工具调用机制（如自适应采样密度、多工具协同）
3. 探索 zero-shot 或 few-shot 下的 agentic 推理能力
4. 将本框架推广至其他视频理解任务（如事件预测、因果推断）

---

> 💡 **总体评价**：  
> 本文系统性地推进了 VAD 进入“**Agent-Level Reasoning**”时代。不仅提出了实用的训练免费方案（GtS），还构建了通往自主推理系统的路径（Agentic RL + VAGU-T + JeAUG），为 LLM 时代的视频理解提供了新范式。

</details>

---

### 9. [Foresight Without Seeing: Latent Futures for World Action Models](https://arxiv.org/abs/2608.11605)

**Authors**: Jiakai Huang, Zhongbo Wu, Zheng Zhang, Zihan Wang, Shan You, Tao Huang  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.11605v1  

#### Abstract
World Action Models (WAMs) couple future visual prediction with robot action generation, enabling policies to model how the physical world evolves during interaction. Existing WAMs differ in how predictive dynamics are exposed to the action pathway. Explicit-future WAMs provide direct access to pred...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Foresight Without Seeing: Latent Futures for World Action Models**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **World Action Models (WAMs)** 在建模未来视觉预测与机器人动作生成之间的耦合关系时，面临效率与预测能力之间的权衡：
- **Explicit-future WAMs**（如级联或联合范式）通过显式生成未来视频帧来提供丰富的动态上下文，但推理成本高，且生成误差可能传播至动作预测。
- **Direct-policy WAMs**（如 Fast-WAM）跳过未来视频生成以提升效率，但在推理时缺乏对 **Action DiT** 显式的、可访问的预测动态接口。

因此，核心问题是：
> **如何在不显式生成未来视频的前提下，让 direct-policy WAM 的 Action DiT 能够访问到预测性的、与动作相关的场景动态？**

---

### 🚀 提出的新方法：**ForeWAM**
提出了一种名为 **ForeWAM**（Foresight-without-Seeing World Action Model）的新型 **dynamics-conditioned direct-policy WAM**，其核心创新包括：

#### （1）**Future-KV：隐式的未来状态 KV 缓存机制**
- 利用单次 **Video DiT prefill** 处理当前观测的干净 latent 和随机初始化的“未来槽位”（stochastic future slots）。
- 将该过程产生的 **layer-wise Key-Value (K/V) states** 缓存并复用于整个动作去噪过程。
- 这使得 Action DiT 可在整个推理过程中访问由 Video DiT 构建的潜在未来动态表示，而无需迭代生成或解码未来视频。

#### （2）**Dynamics Registers + Latent Action Supervision**
- 引入一组可学习的 **dynamics registers**，作为紧凑的状态转移编码器。
- 使用一个冻结的 **LaWM（Latent Action World Model）教师模型**，从真实前后帧中提取非执行性的 **latent-action 表示**，监督 dynamics registers 学习捕捉交互引发的变化（如物体运动、接触变化、任务进度）。
- 教师仅在训练阶段使用，部署时不需未来观测或额外模块。

#### （3）端到端设计优势
- 同时保留了 direct-policy 的高效性与对未来动态的感知能力。
- 实现“**预见而不看见**”（foresight without seeing）——即具备预测性推理能力，但不依赖未来图像生成。

---

### 🔍 相比现有方法的优势
| 维度 | ForeWAM | Explicit-future WAMs | Direct-policy WAMs（如 Fast-WAM） |
|------|--------|-----------------------|-------------------------------|
| 推理效率 | 高（仅一次 Video DiT prefill） | 低（迭代视频去噪） | 高 |
| 动态感知能力 | 强（通过 Future-KV + dynamics registers） | 强（显式未来） | 弱（仅基于当前观测） |
| 是否需要未来生成 | ❌ 否 | ✅ 是 | ❌ 否 |
| 参数量 | ~2B | 通常更大 | ~6B（Fast-WAM） |
| 延迟 | 568ms（标准），220ms（Flash） | >600ms | 667ms |

> ✅ **ForeWAM 成功弥合了预测能力与推理效率之间的鸿沟**。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **LIBERO** 系列基准测试套件：
  - **LIBERO-Spatial / Object / Goal / Long**：标准 in-distribution 控制任务，共 150 个任务。
  - **LIBERO-Plus**：out-of-distribution robustness benchmark，引入七类扰动：
    - 相机视角、机器人初始状态、语言指令、光照、背景纹理、传感器噪声、物体布局

### 🧪 实验设置
- **输入**：
  - 多视角图像（224×448）、语言指令、本体感觉状态（8维）
  - 输出：7维动作块（6-DoF pose + 抓手控制），horizon=32
- **模型架构**：
  - 视觉主干：Wan2.1-T2V-1.3B Video DiT（1.3B参数）
  - 动作专家：Action DiT（1024隐藏维度，30层）
  - 总策略参数：约 **2B**（仅为 Fast-WAM 的 1/3）
- **训练目标**：
  - 视频流匹配损失（$L_{\text{video}}$）
  - 动作流匹配损失（$L_{\text{action}}$）
  - Latent Action 蒸馏损失（$L_{\text{LA}}$，带 stop-gradient）
- **推理配置**：
  - **ForeWAM**：10步动作去噪
  - **ForeWAM-Flash**：经 OneDP 蒸馏后的 2 步快速版本

### 📊 评估指标
- **任务成功率（Success Rate %）**：每任务运行 50 次取平均
- **动作生成延迟（Action Generation Latency）**：单次推理耗时（ms），在 NVIDIA A800 GPU 上测量
- **消融研究**：控制变量比较不同组件组合的效果

### 🆚 基线方法对比
| 方法 | 类型 | 是否有 embodied PT | 参数量 |
|------|------|------------------|--------|
| OpenVLA, To, UniVLA 等 | VLA Models | ✅ 是 | 3.3B–7B |
| Fast-WAM (Yuan et al., 2026b) | Direct-policy WAM | ❌ 否 | 6B |
| ForeWAM（本文） | Proposed | ❌ 否 | 2B |

> 所有变体均 **未使用 embodied robot data pretraining**，直接在 LIBERO 上训练。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）在 **标准 LIBERO 套件** 上的表现（Table 1）

| 方法 | Spatial | Object | Goal | Long | **Overall** |
|------|--------|--------|------|------|------------|
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | **97.6%** |
| **ForeWAM** | 97.0 | 99.6 | 97.2 | 92.8 | **96.7%** |
| **ForeWAM-Flash** | 97.8 | 99.2 | 97.4 | 93.0 | **96.9%** |

> ⭐ 即使没有预训练且参数更少，ForeWAM 达到了接近 SOTA 的性能，Flash 版本甚至略优于标准版。

---

#### （2）在 **LIBERO-Plus（OOD Robustness）** 上的表现（Table 2）

| 方法 | Camera | Robot Init | Language | Light | Background | Noise | Layout | **Overall** |
|------|--------|------------|----------|-------|------------|--------|--------|------------|
| Fast-WAM | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | **51.5%** |
| **ForeWAM** | **62.5** | 37.4 | **73.0** | 74.1 | 55.1 | **58.8** | **70.4** | **61.6%** (+10.1) |
| **ForeWAM-Flash** | 57.9 | 40.4 | 67.2 | 71.0 | 53.0 | 53.7 | 65.3 | **58.2%** (+6.7) |

> ✅ **ForeWAM 在大多数扰动下显著超越 Fast-WAM**，尤其在 **相机视角（+46.1 pts）** 和 **传感器噪声（+21.1 pts）** 上表现突出。

---

#### （3）推理效率对比（Table 3）

| 方法 | Inference Latency (ms) | 相对降低 |
|------|-------------------------|----------|
| Fast-WAM | 667 | — |
| **ForeWAM** | **568** | ↓14.8% |
| **ForeWAM-Flash** | **220** | ↓67.0% |

> 💡 ForeWAM-Flash 实现了近 **3倍加速**，达到实时控制潜力（<250ms）。

---

#### （4）消融实验（Ablation Study, Table 4）
在相同配置下进行对比（1,482次评估）：

| 配置 | Overall Success Rate |
|------|------------------------|
| Base Policy（无 Future-KV & 无 LA） | 53.6% |
| Future-KV Only | 58.5% |
| LA Supervision Only | 58.0% |
| **ForeWAM（Both）** | **61.6%** |

> 🔍 结果表明：
> - Future-KV 和 LA supervision **互补增益**，联合使用效果最佳。
> - 两者共同贡献了 **+8.0个百分点** 的提升（vs Base）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Direct-policy WAMs 完全可以在不生成未来视频的情况下获得强大的预测能力**：
   - 通过 **Future-KV** 和 **dynamics registers**，Action DiT 能有效利用潜在未来的分布式与紧凑式动态表示。
   
2. **预测性上下文可通过隐藏的 KV 缓存传递**：
   - 单次 Video DiT prefill 即可构建可用于整个动作去噪过程的动态 context，极大提升效率。

3. **Latent Action supervision 是引导 dynamics registers 学习交互相关变化的有效手段**：
   - 冻结教师模型提供的 transition cue 显著提升了 OOD 泛化能力。

4. **效率与性能兼得**：
   - ForeWAM 在仅 **2B 参数**、**无 embodied pretraining** 的条件下，实现了优于 6B 参数 Fast-WAM 的 OOD 表现，并将延迟降低 **67%**。

---

### ⚠️ 局限性
- 当前评估局限于 **LIBERO 及其扩展集**，尚未验证在真实世界或其他机器人形态上的泛化能力。
- 对 **长视野任务（Long Horizon）** 的处理仍有挑战，特别是在 robot initial state 扰动下的性能下降明显。
- dynamics registers 的可解释性和因果必要性仍需进一步分析（文中指出非因果充分证据）。

---

### 🔮 未来工作方向
- 扩展至更多样化的机器人平台和真实环境部署。
- 探索 dynamics registers 的可视化与干预机制，增强模型可解释性。
- 结合 memory 或 recurrence 以支持更长程的任务规划。
- 将 OneDP 加速技术推广至其他 diffusion-based VLA 模型。

---

## ✅ 总结一句话
> **ForeWAM 证明了“预见”不必“看见”——通过 Future-KV 与 latent-action 监督的 dynamics registers，direct-policy WAM 可在不生成未来视频的前提下，高效获取预测性动态，实现高性能、低延迟、强鲁棒性的机器人控制。**

</details>

---

### 10. [Sparse and robust geometric twin support vector machine via asymmetric RoBoSS loss function](https://arxiv.org/abs/2608.11567)

**Authors**: Kai Qi, Xinji Huang, Hongchun Wang  
**Category**: cs.LG  
**Published**: 2026-08-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.11567v1  

#### Abstract
In real-world scenarios, the training data usually contains redundant features, label noise and feature noise, which provide severe challenges for the efficiency of machine learning methods. Since standard support vector machine (SVM) adopts $l_2$-norm penalty and hinge loss function, it lacks the a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Sparse and robust geometric twin support vector machine via asymmetric RoBoSS loss function》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现实世界中的机器学习任务常面临以下挑战：
- **冗余特征（Redundant features）**：无关或误导性特征降低模型预测性能。
- **标签噪声（Label noise）**：训练样本的类别标签错误，影响分类器稳定性。
- **特征噪声（Feature noise）**：特别是边界附近的零均值噪声（即 resampling noise），导致模型对重采样不稳定。

传统 SVM 和 TSVM 因采用 **hinge loss** 和 **l2-norm penalty**，存在：
- 对噪声敏感；
- 缺乏特征选择能力；
- 在高维稀疏场景下表现不佳。

---

### 提出的新方法与新思路
本文提出了一种新的 **非对称、鲁棒、有界、稀疏且光滑（asymmetric, Robust, Bounded, Sparse, Smooth, aR）损失函数**，并基于此构建了：
- **aRSGTSVM**（asymmetric RoBoSS loss-based Sparse Geometric Twin SVM）用于分类；
- **aRSGTSVR**（对应回归版本）用于回归。

#### 核心创新点：
1. **设计了 aR loss 函数**  
   - 在 RoBoSS loss 基础上引入不对称参数 $ T \in [0,1] $，使正负侧损失不同，增强对 resampling noise 的鲁棒性。
   - 同时具备：**光滑性（C¹-smooth）、有界性（bounded）、非凸性（nonconvex）**。
   - 能同时缓解 **label noise** 和 **resampling noise** 的影响。

2. **理论证明其鲁棒性**  
   - 利用 **Influence Function（影响函数）** 分析，严格证明 aR loss 的影响函数是有界的，从而从统计角度保证了模型的鲁棒性。

3. **结合 l1-norm penalty 实现特征选择**  
   - 引入 l1 正则项，提升模型在高维数据下的稀疏性和可解释性。

4. **优化算法设计：iPiano**  
   - 针对非凸 + 非光滑目标函数，采用 **proximal gradient descent 类型的 iPiano 算法**，实现快速稳定求解。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **鲁棒性** | 同时抵抗 label noise 和 resampling noise，优于仅处理单一噪声的 Pin-SVM、RoBoSS-SVM 等。 |
| **稀疏性** | l1-norm 实现有效特征选择，适用于高维场景；而如 Pin-SVM 所有样本都成为支持向量，丧失 sparsity。 |
| **统一框架** | 将 robustness、sparsity、geometric consistency 统一建模，解决多类实际问题。 |
| **性能优越** | 在合成、UCI 及金融指数跟踪任务中全面超越主流方法。 |

---

## 2. 核心实验方法和设置

### 数据集
#### （1）人工数据集（Synthetic datasets）
- **分类任务**：二维及多维高斯分布生成的数据，用于验证对 label noise 和 resampling noise 的鲁棒性。
- **回归任务**：Sinc 函数加噪（uniform / Gaussian noise），测试拟合能力和抗干扰能力。

#### （2）真实数据集
- **UCI 分类数据集**：共 17 个，涵盖不同维度和样本规模（见 Table 5），例如 `acoustic`, `diabetes`, `waveform` 等。
- **中国股市指数跟踪数据集**：6 个股票指数（bz50, cy200, hs300, xf100, ys50, zz500），时间跨度为 2025 年初至年中。

---

### 实验设置与评估指标

| 任务类型 | 评估指标 | 参数调优方式 |
|--------|---------|-------------|
| **分类** | Accuracy (acc)，标准差（sd） | 五折交叉验证，报告平均 acc ± sd |
| **回归** | RMSE、MAE（用于参数选择）、Annual Tracking Error（年度追踪误差） | MAE 选参，最终评价用 RMSE 或 TrackingErroryear |

#### 噪声注入策略
- 在 UCI 数据集中人为翻转 15% 和 35% 的标签以模拟 label noise。

---

### 基线方法对比

#### 分类任务对比模型：
- **1-SVM**（l1-norm SVM）
- **TPMSVM**
- **Pin-TSVM**
- **rhingeSVM**（rescaled hinge）
- **RoBoSS-SVM**

#### 回归任务对比模型：
- **SVR**
- **LASSO**
- **Elastic Net**
- **TSVR**
- **Res-TSVR**

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### （1）人工数据集结果（Table 1–4, Figures 2–8）

| 场景 | 结果摘要 |
|------|----------|
| **Example 1（分类，无噪声）** | aRSGTSVM 决策边界最接近 Bayes 最优边界（图2），准确率最高（0.960）。 |
| **Example 2（resampling 稳定性）** | aRSGTSVM 超强稳定性：多次重采样后决策面几乎不变（图3），远优于其他模型。 |
| **Example 3（高维分类）** | 在 $ p > n $ 场景下仍保持高精度，尤其在 35% 噪声时达 0.660，显著高于第二名 0.630（Table 2）。 |
| **Example 4（消融实验）** | 移除 aR loss 后性能明显下降，尤其在高噪声（40%）和高维（p=300）下差距扩大（图4），验证 aR loss 的有效性。 |
| **Example 5（运行效率）** | 在大样本（n=10000）下运行时间极低（图5）；但在极高维（p=5000）时计算成本上升较快（图6）。 |

#### （2）UCI 数据集结果（Tables 6–8, Figure 9）

| 指标 | 表现 |
|------|-------|
| **平均排名（Friedman test）** | aRSGTSVM 在所有噪声水平下排名第一（图9）。 |
| **0% 噪声** | 平均 acc 最高，尤其在 `acoustic`, `messidor` 上领先明显。 |
| **15%/35% 噪声** | 性能下降最小，在极端 35% 噪声下仍保持显著优势（如 `autism`: 0.993 vs 第二名 0.900）。 |
| **统计检验** | Friedman 拒绝原假设（Fr > 临界值），Nemenyi 检验显示 aRSGTSVM 与其他模型差异显著。 |

#### （3）指数跟踪任务（Table 9, Figure 10）
| 指数 | aRSGTSVR 表现 |
|------|----------------|
| **全部6个指数** | aRSGTSVR 在 **annual tracking error** 上全面最优（Table 9） |
| **典型例子**：<br>- `cy200`: 0.040（vs 第二名 0.045）<br>- `zz500`: 0.105（vs 0.109）<br>- `xf100`: 0.226（大幅优于 SVR 的 0.638） |
| **可视化效果**（图10） | 拟合曲线与真实走势高度一致，泛化能力强。 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **aR loss 显著提升鲁棒性**  
   - 不仅抑制 label noise，还能增强 resampling stability，是首个兼顾两类噪声的 TSVM 损失函数之一。

2. ✅ **aRSGTSVM/R 在多种场景下性能领先**  
   - 在合成、UCI、金融三大类数据上均取得最佳或接近最佳结果，尤其在高噪声、高维条件下优势更明显。

3. ✅ **具备良好稀疏性和特征选择能力**  
   - l1-norm 使得模型自动筛选重要变量，在回归任务中表现出色（如 Table 4 中 RMSE 明显更低）。

4. ✅ **优化算法高效可行**  
   - iPiano 算法收敛快、数值稳定，适合大规模学习任务。

---

### 方法的局限性
1. ❗ **高维特征下计算开销较大**  
   - 当 $ p \to 5000 $ 时，aRSGTSVM 运行时间急剧上升（图6），不如 Pin-TSVM、TPMSVM 高效。

2. ❗ **依赖网格搜索调参**  
   - 超参数（$ a, \lambda, T $）需通过 grid search 选取，缺乏自动化或理论指导的调参机制。

3. ❗ **未完全解决过拟合风险**  
   - 虽使用交叉验证选参，但无法从根本上缓解过拟合，尤其在小样本高维情形。

---

### 未来工作方向
1. 🔮 探索基于 **信息准则（AIC/BIC）** 的参数选择方法，避免重复 CV，提高效率。
2. 🔮 设计 **分布式或并行算法** 以应对超高维数据（如基因组、图像）。
3. 🔮 将 aR loss 扩展到深度学习或其他核方法中，探索更广泛应用。
4. 🔮 研究自适应调整 $ T $ 参数的方法，使其随数据动态变化，进一步提升灵活性。

---

> **总结一句话**：  
> 本文提出的 **aRSGTSVM/R** 是一种兼具 **鲁棒性、稀疏性、几何一致性** 的新型 twin SVM 框架，通过创新的 **asymmetric RoBoSS loss** 和 **iPiano 优化器**，在复杂噪声环境和高维现实中展现出卓越性能，为实际应用提供了可靠且先进的解决方案。

</details>

---

### 11. [Cutting AI Datacenter Energy with Reinforcement Learning: Measured Power Control of LLM Training from One GPU to the Fleet](https://arxiv.org/abs/2608.11226)

**Authors**: Eliseo Curcio  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.11226v1  

#### Abstract
Reinforcement-learning post-training dominates modern language-model development, yet its power behavior on GPU hardware has not been characterized, and datacenters manage GPU power with workload-blind mechanisms, static caps and reactive throttling, that slow hardware indiscriminately. We instrumen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Cutting AI Datacenter Energy with Reinforcement Learning: Measured Power Control of LLM Training from One GPU to the Fleet*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本论文针对当前 AI 数据中心在 **GPU 功耗管理** 上的严重低效问题，提出并验证了一种基于 **Reinforcement Learning (RL)** 的动态功耗控制方案。具体解决的问题包括：

- **传统功耗管理机制是“workload-blind”**：依赖静态的 firmware power cap、thermal throttling 等硬件级限制，这些方法不感知训练任务的实际行为，导致在非高峰时段也强制降频，牺牲了吞吐量。
- **数据中心普遍按 nameplate provisioning**：即按照 GPU 的热设计功耗（TDP）总和加安全余量来规划电力容量，造成严重的资源浪费。
- **缺乏对 LLM 训练负载真实功耗行为的实测分析**：尤其是 **Reinforcement Learning from Human Feedback (RLHF)** 类训练（如 GRPO）的功耗特征尚未被系统刻画。

### ✅ 提出了什么新方法或新思路

作者提出了一个 **端到端的 RL 控制框架**，实现从单 GPU 到多 GPU 集群的 **动态功耗控制**，其核心创新点如下：

- **首次对 GRPO 训练进行细粒度功耗测量**：在 7B、14B、72B 规模下采集了超过 **380,000 个半秒级 GPU 功耗样本**，标注了训练阶段（rollout/update），揭示了功耗的周期性和可预测性。
- **构建了一个 PPO-based 的元控制器（meta-controller）**：
  - **状态输入**：实时功耗序列、预算余量、当前 group size 等。
  - **动作空间**：动态调整 `group_size` 和 `batch_size` 等生成参数。
  - **目标**：在满足功耗预算的前提下最大化 token 输出和能效。
- **提出“occupancy 而非 volume”的控制原则**：
  - 在单 GPU 场景下，调整 `group_size` 可有效控制功耗（因影响计算量）。
  - 在多 GPU 分片（sharding）场景下，`group_size` 失去控制力；而将其作为 **generation concurrency（并发数）** 来调节 pipeline occupancy，则仍保留 17–22% 的功耗调节能力。
- **提出三层控制架构**（Deployment Architecture）：
  1. **Job-level RL 控制器**（本文重点）
  2. **Firmware-level 动态 power cap**（毫秒级响应，应对瞬态）
  3. **Fleet scheduler**（分钟级调度，错峰运行）

### ✅ 相比现有方法的优势

| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 控制粒度 | 硬件级（clipping frequency） | 软件级（调整训练参数） |
| 是否感知负载 | 否（blind） | 是（adaptive） |
| 对吞吐量影响 | 降低（always throttled） | 提升（selective control） |
| 可扩展性 | 固定 cap | 支持 oversubscription |
| 成本效益 | 高 capex（建更多电站） | 降低 capex + opex |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集与模型

- **模型系列**：Open Qwen2.5 指令微调系列，参数规模为：
  - 7B（1×A100）
  - 14B（2×A100）
  - 72B（4×A100）
- **训练方法**：Group Relative Policy Optimization (**GRPO**) —— 当前主流的 RLHF 替代方案。
- **数据集**：UltraFeedback 二值化偏好数据集（prompt + chosen/rejected response）。
- **量化版本用于 sweep**：AWQ-quantized 72B 模型用于 generation-only 代理实验。

### ✅ 实验设置

- **硬件平台**：NVIDIA A100-SXM4-80GB（TDP 400W），云环境部署。
- **功耗测量工具**：NVIDIA Management Library (**NVML**)，采样频率 **500ms**。
- **控制频率**：
  - 单 GPU：每 training step（约 55 秒）调整一次。
  - 多 GPU occupancy 控制器：每 generation batch boundary（12–21 秒）调整一次。
- **评估协议**：
  - 使用完整 trace 进行 replay evaluation。
  - 控制器与随机策略、阈值启发式（heuristic）对比。
  - 所有比较均在相同 power cap 下进行。

### ✅ 评估指标

| 指标 | 定义 |
|------|------|
| **Cap violations / violation rate** | 功耗超过预算的比例 |
| **Total tokens** | 总生成 token 数量 |
| **Energy efficiency** | tokens per MWh |
| **Energy intensity** | Wh per 1k tokens |
| **Cost & Carbon** | 按 $0.08/kWh 和 0.25 kg CO₂/kWh 计算 |
| **Integrated excess** | 超出预算的累积能量（W·s） |

### ✅ 基线方法对比

| 基线 | 描述 |
|------|------|
| **Random policy** | 随机选择动作，作为下界 |
| **Threshold heuristic** | 当功耗 >90% cap 时减小 group size，<70% 时增大 |
| **Static safe baseline** | 固定低配置（如 batch 4），确保不超限但吞吐低 |
| **Uncontrolled (full occupancy)** | 最大并发运行，高吞吐但高违规 |

---

## 3. 主要实验结果和性能指标

### ✅ 单 GPU 结果（7B on 1×A100）

在 500 步 trace 上评估，cap = 314.6W：

| 指标 | Baseline | Controller | Change |
|------|----------|-----------|--------|
| Cap violations | 216 | 22 | **↓89.8%** |
| Total tokens | 1,731,712 | 2,044,928 | **↑18.1%** |
| Energy consumed | 1.777 kWh | 1.662 kWh | ↓6.5% |
| Tokens per MWh | 0.975B | **1.230B** | **↑26.2%** |
| Energy intensity | 1.026 Wh/k tokens | **0.813 Wh/k tokens** | ↓20.8% |

> ✅ **同时提升吞吐与能效，显著减少违规**，证明 RL 控制优于盲目降频。

---

### ✅ 多 GPU 挑战与突破（72B on 4×A100）

#### ❌ 初始失败：group_size 失去控制力

- 在 72B 分片训练中，原控制器对 `group_size` 的调整 **无法改变 per-device 平均功耗**（147.0W vs 147.0W）。
- 原因：pipeline execution 中各层轮流计算，增加候选答案数不影响 occupancy。

#### ✅ 转向 occupancy 控制：成功重建控制器

通过 **actuator-authority sweep** 发现：

| 控制变量 | Mean power swing | Peak swing | Throughput ratio |
|---------|------------------|------------|------------------|
| **Generation batch size (concurrency)** | **22.5%** | **24.0%** | **5.2×** |
| Group size (as concurrency) | 16.9% | 17.0% | 3.3× |
| Completion length | 3.5% | 1.8% | 1.2× |
| Inter-batch pause | 6.3% | 3.0% | 1.3× |

> ✅ **只有并发类参数（batch size）具有显著功耗调节能力**

#### ✅ Live 四臂实验结果（20min × 3 reps）

| Arm | Tokens | Violation Rate (30s) | Wh/1k tokens |
|-----|--------|------------------------|---------------|
| Static batch 8 (uncontrolled) | 222,472 | 17.72% | 0.802 |
| Static batch 4 (safe) | 134,860 | 0.01% | 1.219 |
| Optimized hysteresis | 166,189 | 1.96% | 1.019 |
| **PPO occupancy controller** | **183,069** | **2.27%** | **0.947** |

> ✅ **相比 safe baseline：↑35.7% tokens，↓87.2% violations，↓22.3% energy/token**
>
> ✅ **相比 uncontrolled：保留 81.9% 吞吐，但违规下降 87.2%**

---

### ✅ 测量窗口分析（What is a "violation"?）

不同时间窗口下的违规率：

| Window | 72B single job | 16-GPU fleet |
|--------|----------------|-------------|
| Instantaneous (0.5s) | 23.6% | 9.1% |
| 10s rolling | 10.8% | ~ |
| **30s rolling** | **1.6%** | **0%** |
| **5min rolling** | **0%** | **0%** |

> ✅ **基础设施真正关心的是 30s–5min 窗口平均功耗，瞬态 spike 可忽略**
>
> ✅ **因此，许多“违规”在实际电网尺度上并不存在**

---

### ✅ 集群级容量规划分析（Fleet Provisioning）

构建 16-GPU 混合集群（2×72B + 2×14B + 4×7B）：

| Strategy | 30s Peak (kW) | Provisioning Fraction |
|--------|----------------|------------------------|
| Naive (aligned) | 3.59 | 56% |
| Random starts | 3.44 | 54% |
| Phase-staggered | 3.38 | 53% |
| **Staggered + job control** | **3.17** | **50%** |

> ✅ **实际峰值需求仅为 nameplate 的 50%**
>
> ✅ **支持约 2 倍的 nameplate oversubscription**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLM 训练负载具有高度可预测且可控的功耗结构**，尤其 GRPO 类训练存在明显的 rollout/update 周期。
2. **RL 控制器可在不牺牲输出的情况下大幅降低功耗违规**：
   - 单 GPU：↓89.8% violations，↑18.1% tokens，↑26.2% energy efficiency。
3. **控制有效性随 scale 变化**：
   - 单 GPU：`group_size` 是有效 actuator。
   - 多 GPU：需转向 **occupancy-based 控制**（如 generation batch size）。
4. **瞬态违规在基础设施时间尺度上可忽略**：
   - 5 分钟窗口内违规率为 0%，无需复杂控制。
5. **数据中心可安全地进行约 2 倍的电力 oversubscription**：
   - 实际峰值仅为 nameplate 的 50–56%。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|-------|------|
| **未在完整 GRPO loop 上验证 occupancy 控制器** | 使用了 AWQ-quantized generation proxy，虽占 99.9% 时间，但仍非完全等价 |
| **Live controller 未达 1% 违规目标** | 实现 2.27% ± 1.08%，略高于预设目标 |
| **Policy 种子敏感性** | 3 个训练种子仅 1 个通过验证，存在随机性 |
| **未在真实运营集群上验证** | 当前为 trace replay + 小规模实验 |
| **Firmware 控制未实现** | 因权限限制无法测试 power cap 等底层机制 |

---

### 🔮 未来工作方向

1. **Operator Pilot 验证**：
   - 建议开展为期 30–60 天的真实集群试点，验证 **provisioning fraction ≤ 70%** 的可行性。
   - 成本约 $50K，目标：相对 nameplate 减少 30% 以上 provisioning。
2. **集成 firmware-level 控制器**：
   - 实现毫秒级动态 power cap，应对瞬态 spike。
3. **扩展至其他训练范式**：
   - 如 DPO、SFT、pretraining 等是否具备类似可控性。
4. **与 grid-level demand response 对接**：
   - 将 AI 训练作为 **dispatchable load**，参与电力市场调节。
5. **自动化 control stack 部署**：
   - 构建标准化的三层控制栈（job-level RL + firmware + scheduler）。

---

## 📌 总结一句话

> 本文首次实测并验证了 **RLHF 训练负载的功耗可控性**，提出基于 RL 的动态控制框架，在单 GPU 上实现 **近 90% 违规削减 + 18% 吞吐提升**，并通过转向 **occupancy 控制** 克服多 GPU 分片挑战，最终揭示 **AI 数据中心可安全实现约 2 倍电力 oversubscription**，为缓解 AI 电力危机提供了可落地的技术路径。

</details>

---

### 12. [XBridge: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication](https://arxiv.org/abs/2608.11676)

**Authors**: Wooseong Yang, Wei-Chieh Huang, Weizhi Zhang, Yu Wang, Philip S. Yu, Junhyun Lee  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.11676v1  

#### Abstract
Heterogeneous multi-agent LLM systems, where agents are powered by different model families, can outperform homogeneous configurations by reducing redundant reasoning patterns. Yet existing communication protocols either operate through text, discarding the sender's internal representations, or requ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# XBridge: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题：**Entity Grounding Problem（实体接地问题）**

在异构多智能体 LLM 系统中，不同模型家族（如 Llama、Qwen、Mistral）由于 **tokenizer、vocab、架构、隐藏空间不兼容**，导致跨模型通信困难。现有方法存在以下瓶颈：

- **Text-based communication (NLComm)**：通过自然语言传递信息，虽架构无关，但需 autoregressive decoding，延迟高，且压缩了 sender 的内部表示。
- **Latent-level communication**：如 KV cache sharing 或 hidden-state injection，避免了解码，但要求 sender 和 receiver 架构一致，无法用于异构模型。
- **Continuous projection 方法**：虽然可跨模型映射，但会出现 **Rare-Token Compression Collapse** —— 即连续表示能传递上下文语义，却丢失了具体实体（如人名、数字、罕见词）的身份。

作者将此定义为 **Entity Grounding Problem**：如何在保留 sender 上下文理解的同时，在 receiver 中正确“锚定”对应的离散实体。

---

### 🚀 提出的新方法：**XBRIDGE**

XBRIDGE 是一种 **decode-free、双通道通信协议**，解耦了“实体身份”与“上下文理解”的传输：

#### （1）Lexical Anchor Mapping (LAM)  
- **作用**：提供 **离散实体锚点（discrete entity anchors）**。
- **机制**：将 sender 的原始 token 映射到 receiver 的 vocab 中，通过 deterministic 映射函数 $\phi$ 实现跨词汇表对齐。
  - 若 token 字符串存在于双方 vocab，则直接 ID 映射；
  - 否则进行 string fallback：解码为字符串后重新 tokenize。
- **优势**：无需训练、无解码开销、保证实体 token 在 receiver 输入空间中是原生的。

#### （2）Latent Enrichment Bridge (LEB)  
- **作用**：提供 **连续上下文增强（contextual enrichment）**。
- **机制**：在 receiver 中插入 4 个 **gated cross-attention 模块**，让 receiver 主动查询 sender 的 last-layer hidden states $H_s$。
  - 使用 learnable 投影矩阵处理维度不匹配；
  - 通过 tanh-gated residual 连接融合信息，保持 receiver 自身计算不变。
- **特点**：仅 264M 可训练参数（占 receiver 的 3.8%），训练快（<10 分钟），推理开销极小。

> 🔑 **核心思想**：  
> - **LAM 负责 “who/what”**（实体是谁）  
> - **LEB 负责 “how/why”**（上下文关系和推理过程）  
> 二者结合实现 **entity-grounded latent communication**。

---

### ⚖️ 相比现有方法的优势

| 维度 | XBRIDGE | NLComm | KVComm |
|------|--------|--------|--------|
| 是否支持异构架构 | ✅ | ✅ | ❌（需同架构） |
| 是否需要 autoregressive decoding | ❌（decode-free） | ✅ | ❌ |
| 是否保留 sender 内部表示 | ✅（完整 $H_s$） | ❌（仅文本摘要） | ✅ |
| 是否保持实体身份 | ✅（LAM 锚定） | 部分（依赖生成质量） | ✅（同 vocab） |
| 推理延迟 | **低至 0.15s** | ~1.70s（11× 更慢） | 0.13–0.15s |
| 参数量 & 训练成本 | 仅 264M，<10min 单卡 | 无额外参数 | 无额外参数 |

---

## 2. 核心实验方法和设置

### 📚 数据集（共 7 个基准任务）

| 数据集 | 类型 | 特点 |
|-------|------|------|
| **HotpotQA** | 多跳问答 | 需要跨文档推理 |
| **MuSiQue** | 组合式 QA | 结构化推理链 |
| **QASPER** | 科学文献问答 | 长文本、技术性强 |
| **2WikiMQA** | 多跳 QA | 极长上下文（均值 ~7.3K tokens） |
| **MFldQA** | 长文档检索 | 答案位于深层句子 |
| **Countries** | 事实推理 | 上下文隐含地理知识（如地标→国家） |
| **Tipsheets** | 结构化抽取 | 答案直接可提取 |

> 所有任务采用 **asymmetric protocol**：sender 见 context，receiver 见 question，不能共享全文。

---

### 🧪 实验设置

- **模型组合（异构对）**：
  - `Llama-3.1-8B → Qwen2.5-7B`
  - `Qwen2.5-7B → Llama-3.1-8B`
  - `Mistral-7B → Qwen2.5-7B`

- **评估指标**：
  - **F1 Score**（主要）
  - **Per-sample inference latency**（H200 GPU）
  - **Entity fidelity**（通过 entity rank 和 cosine similarity 衡量）

- **Baseline 对比方法**：
  - **NoComm**：receiver 仅见问题（下界）
  - **FullComm**：receiver 直接读全文（上界，非通信方法）
  - **NLComm_hetero**：异构文本通信（greedy decode 128-token summary）
  - **NLComm_homo**：同架构文本通信
  - **KVComm**：同架构 KV cache sharing [5]

- **训练细节**：
  - LEB 使用 **587 个平衡样本**（每任务约 100 个）微调，单 GPU <10 分钟
  - 固定 sender 和 receiver，仅训练 bridge 模块

---

## 3. 主要实验结果和性能指标

### 📊 总体性能对比（Table 1）

| 方法 | 平均 F1（三组异构对） | 最佳任务数 |
|------|------------------|----------|
| **XBRIDGE** | **63.2 / 61.9 / 65.0** | **7/7 超过 NLComm_hetero** |
| **NLComm_hetero** | 41.8 / 47.6 / 44.1 | — |
| **Improvement** | **+21.4pp / +14.3pp / +20.9pp** | — |

> ✅ XBRIDGE 在所有 7 个任务、所有 3 个异构方向上均显著优于 NLComm。

> ✅ 在多个任务上甚至 **超过 FullComm**（说明 LEB 提供了超越“只看原文”的推理能力）。

---

### ⏱️ 推理延迟（Table 5）

| 方法 | 延迟（s/sample） | 相对 NLComm |
|------|------------------|------------|
| **XBRIDGE** | **0.15** | **1.7×** |
| **NLComm_hetero/homo** | 1.70 | 18.9× |
| **NoComm / FullComm** | 0.09–0.13 | ~1.4× |

> ✅ XBRIDGE 推理速度比 NLComm 快 **11 倍以上**，几乎与单模型推理相当。

---

### 🔍 消融实验（Ablation Study）

#### （1）移除 LAM 或 LEB（Table 3, HotpotQA）

| 配置 | F1 | Δ vs NoComm |
|------|----|-------------|
| **NoComm** | 24.6 | — |
| **XBRIDGE w/o LAM** | 30.3 | +5.7 |
| **XBRIDGE w/o LEB** | 56.5 | +31.9 |
| **Full XBRIDGE** | **78.8** | **+54.2** |

> ❗ LAM 缺失时性能骤降 → **没有实体锚点，LEB 的上下文信号无法落地**  
> ❗ LEB 缺失仍有基础性能 → LAM 至少提供了原始 token 信息

#### （2）Entity Perturbation 实验（Table 2）

- 替换 context 中答案实体（如 "Christopher Nolan" → "Bong Joon Ho"）
- 控制 LAM 和 LEB 输入是否被替换

| 条件 | 输出实体由谁决定？ |
|------|------------------|
| LAM 原始，LEB 替换 | → 原始实体（不变） |
| LAM 替换，LEB 原始 | → 替换实体（改变） |

> ✅ **LAM 决定输出实体身份，LEB 决定如何推理该实体** —— 明确角色分离！

---

### 📈 其他重要发现

- **LEB 效果随任务复杂度提升而增强**：
  - Countries（+35.8pp）、HotpotQA（+22.3pp）收益大
  - Tipsheets（+2.0pp）收益小（因答案可直接提取）

- **Zero-shot composability**：
  - 两个独立训练的 bridge（Llama + Mistral → Qwen）可组合使用
  - Dual XBRIDGE 达到 **70.4% F1**，优于单 sender（67.0%）和 dual NLComm（56.8%）

- **Sender size scaling**（Table 4）：
  - Qwen2.5-1.5B → Llama: 59.5%
  - Qwen2.5-7B → Llama: 61.9%
  - Qwen2.5-14B → Llama: 62.1%（受限于训练数据量）

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Entity Grounding 是异构 LLM 通信的核心挑战**：
   - 单纯的 continuous bridge 会遭遇 **Rare-Token Compression Collapse**，丢失实体身份。

2. **XBRIDGE 成功解耦“实体”与“上下文”**：
   - LAM 提供 receiver-native 实体锚点
   - LEB 提供 sender 的集成上下文理解
   - 两者协同实现高效、准确的跨架构通信

3. **性能与效率双重优势**：
   - 在所有任务上优于 NLComm（平均 +20pp）
   - 推理速度快 11 倍，训练轻量，部署友好

4. **模块化设计支持扩展性**：
   - 支持 zero-shot 多 sender 组合
   - 可适配不同 sender-receiver 对

---

### ⚠️ 局限性（Limitations）

1. **单向通信**：当前 bridge 为单向（sender → receiver），双向或多轮对话需额外 bridge。
2. **未验证大规模多 agent 场景**：多个 agent 同时参与时信号交互机制尚不明确。
3. **训练数据有限制约大 sender 发挥**：14B 模型潜力未完全释放，需更大平衡训练集。

---

### 🔮 未来工作方向

- 构建 **bidirectional latent dialogue protocol**
- 扩展至 **many-agent collaborative frameworks**
- 设计 **universal bridge** 支持动态 agent 插拔
- 探索 **contrastive 或 alignment loss** 进一步提升 entity fidelity

---

> 💡 **一句话总结**：  
> XBRIDGE 首次实现了 **异构 LLM 间的 decode-free、entity-grounded latent communication**，通过 **LAM + LEB 双通道机制** 克服了传统方法在效率与保真之间的权衡，为构建高效、多样化的多智能体系统提供了新范式。

🔗 代码开源地址：[https://github.com/WooseongYang/XBridge](https://github.com/WooseongYang/XBridge)

</details>

---

### 13. [Calibration Bets on the Past: Post-Training Quantization for Financial Time-Series Forecasting](https://arxiv.org/abs/2608.12259)

**Authors**: Junyi Ye, Ivy Gateri Wanjiku  
**Category**: cs.LG  
**Published**: 2026-08-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.12259v1  

#### Abstract
Financial forecasting models are typically developed in full precision, yet production deployment often requires low-precision inference to reduce memory and computational cost. Post-training quantization (PTQ) enables such deployment without retraining. However, reliable activation quantization req...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Calibration Bets on the Past: Post-Training Quantization for Financial Time-Series Forecasting**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文系统研究了**金融时间序列预测模型在部署阶段进行 Post-Training Quantization (PTQ)** 时，**activation calibration（激活校准）对预测性能的影响**。尽管 PTQ 被广泛用于降低推理成本，但在金融场景中，由于市场分布随时间剧烈变化（distribution shift），静态 activation range 的选择可能严重影响低比特量化后的模型表现。

现有研究多集中于视觉和语言模型，而**金融时间序列中的 activation calibration 尚未被系统研究**，尤其是在跨周期（walk-forward）设定下。

### **提出了什么新方法或新思路**
- **首次在金融时间序列预测任务中，采用 walk-forward 协议系统评估 activation calibration 对 PTQ 的影响**。
- 提出将 **activation calibration 视为“部署决策”而非固定预处理步骤**，强调其应根据市场状态动态调整。
- 通过实验揭示了 4-bit activation quantization 中的两种损失类型：
  - **Range-recoverable loss**：可通过更优的 calibration 策略（如 percentile calibration）恢复。
  - **Architecture-dependent residual loss**：即使优化 calibration 仍存在的架构固有敏感性（尤其在 SegRNN 和 TimeMixer 中显著）。

### **相比现有方法的优势**
- **超越传统 abs-max calibration**：证明使用 **percentile calibration（如 p99 或 p99.9）可显著减少量化损失**，在多个架构中恢复 53–94% 的性能退化。
- **提供实用部署指南**：基于 calibration 敏感性和残差损失提出分层部署策略（见 Table 4），适用于真实金融环境。
- **强调 calibration envelope 的监控作用**：建议利用历史最大波动作为“预警信号”，当测试期市场超出该范围时触发重新校准或降级到更高精度。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **S&P 500 股票面板数据**，涵盖当前全部 501 只成分股，时间跨度为 **2008 年 6 月至 2025 年 12 月**，共约 **210 万 asset-days**。
- 输入特征包括：
  - 滞后收益率统计（5/10/20 日均值、20 日波动率）
  - 技术指标（RSI、MACD 及其信号线）
  - 分布摘要（20 日最大值、最小值、偏度）
- 所有特征仅使用训练集统计量进行标准化。

### **实验设置和评估指标**
#### **预测任务**
- **Cross-sectional volatility forecasting**：预测每只股票未来 5 天的波动率，并在横截面上排序。
- 目标变量：`y_i,t = z-score(log(sd(r_i,t+1:t+5)))`

#### **评估指标**
- **Information Coefficient (IC)**：每日横截面 Spearman 相关系数，衡量预测排序与实际排序的一致性。
- **Quantization Damage (△)**：量化模型与全精度模型之间的 IC 差值，正值表示性能下降。

#### **Walk-Forward Protocol**
- 每年作为一个测试 fold，共 **8 个测试年（2018–2025）**。
- 训练窗口：Y−8 至 Y−2；验证年：Y−1；测试年：Y。
- 每个架构 × 每年 × 10 个随机种子 → 共 **560 个独立训练模型**。
- 防止信息泄露：样本按 `t+5` 归属 split，输入窗口允许跨越边界，目标窗口不拆分。

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Full-Precision Baseline** | FP32 模型（作为参考） |
| **Classical Baselines** | Persistence forecast（延续过去5天波动率）、Pooled HAR model（异构自回归模型） |
| **Quantization Settings** | W8A8（INT8 权重+激活）、W4（INT4 权重 + FP32 激活）、W4A4（INT4 权重+激活）、Dynamic INT8（运行时动态校准） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **Table 2 总结：各量化设置下的平均 IC 损失（△）**

| Model | FP32 IC | W8A8 △ | W4 △ | W4A4 (abs-max) △ | W4A4 (p99) △ |
|-------|--------|--------|------|------------------|---------------|
| DLinear | 0.026 | +0.0007 | +0.0004 | +0.006 | +0.024 |
| PatchTST | 0.181 | ~0 | -0.0018 | +0.019 (**11%**) | +0.008 (**5%**) |
| iTransformer | 0.197 | +0.0002 | +0.0030 | +0.023 (**12%**) | +0.015 (**8%**) |
| TSMixer | 0.490 | +0.0002 | +0.0020 | +0.096 (**20%**) | +0.027 (**5%**) |
| Transformer | 0.490 | +0.0003 | +0.0004 | +0.123 (**25%**) | +0.007 (**1%**) |
| SegRNN | 0.458 | +0.0006 | +0.0116 | +0.271 (**59%**) | +0.107 (**23%**) |
| TimeMixer | 0.303 | +0.0084 | +0.0075 | +0.188 (**62%**) | +0.095 (**31%**) |

> 注：括号内为相对于 FP32 IC 的百分比损失。

### **与基线方法的对比结果**
- **Transformer 在 W4A4(abs-max) 下 IC 降至 0.367**，低于 HAR 基线（0.408），**完全丧失神经网络优势**。
- **SegRNN 在 W4A4 下 IC 降至 0.187**，甚至低于 persistence 基线（0.315），表明不当量化可能导致灾难性失败。
- 相比之下，**HAR 和 persistence 表现稳定**，凸显神经模型对量化敏感。

### **消融实验结果**
#### **(1) 不同 calibration 策略的影响**
- **abs-max calibration**：覆盖极端值但牺牲分辨率，导致多数架构严重性能下降。
- **percentile calibration (p99/p99.9)**：
  - 在 **Transformer 和 TSMixer** 上效果极佳，恢复 **80–94% 的损失**。
  - 在 **SegRNN 和 TimeMixer** 上改善有限，分别仅恢复 **73% 和 53%**，显示存在**架构相关残差敏感性**。

#### **(2) Layer-wise Mixed Precision 分析**
- **Transformer**：**Convolutional token embedding 层主导 W4A4 损失**，单独保留该层为 INT8 即可将 damage 从 +0.123 降至 +0.015。
- **TSMixer**：无单一瓶颈层，保护任一层都无法显著缓解损失。

#### **(3) Market Regime Sensitivity**
- 当测试期波动率 **超过 calibration envelope（即 validation 年最大波动）**：
  - **窄范围（如 p99）性能急剧恶化**，因尾部激活被大量剪裁（clipping）。
  - **abs-max 更鲁棒**，因其保留了宽范围。
- 示例：
  - **2020 年（疫情冲击）**：市场极度波动，p99 calibration 外部损失是内部的 **3.6×（Transformer）至 2.5×（SegRNN）**。
  - **2021 年（复苏期）**：市场平稳，abs-max 因过度保守反而表现最差。

#### **(4) Recalibration 实验**
- 若将 2020 年测试 fold 改为用 **2020 年数据重新校准（matched recalibration）**：
  - TSMixer damage 从 +0.077 → +0.043（↓44%）
  - SegRNN damage 从 +0.164 → +0.119（↓27%）
- 表明 **calibration-test mismatch 解释了约 1/4 到 1/2 的超额损失**，其余来自 stress period 本身的极端尾部行为。

---

## **4. 关键结论和发现**

### **主要发现**
1. **8-bit quantization 和 weight-only 4-bit quantization 几乎无损**：W8A8 和 W4 对大多数架构引入的 IC 损失 < 0.001，可安全使用。
2. **4-bit activation quantization 是主要风险源**：在 abs-max calibration 下，部分模型损失高达 **11–62% 的原始 IC**。
3. **percentile calibration 显著缓解问题**：在 Transformer、TSMixer 等架构上可恢复 **53–94% 的性能损失**。
4. **最优 activation range 随市场状态变化**：
   - 平稳市况偏好窄范围（提升分辨率）
   - 极端市况需要宽范围（避免 clipping）
5. **不同架构对 calibration 敏感性差异大**：
   - **Transformer/TSMixer**：主要问题是 calibration 范围选择，可通过调参解决。
   - **SegRNN/TimeMixer**：存在深层残差敏感性，即使最优 calibration 仍损失严重。

### **方法的局限性**
- **模拟量化（simulated quantization）**：所有实验在 FP32 中模拟 INT4 运算，未考虑硬件层面的累积误差或执行效率。
- **仅使用对称 round-to-nearest 量化方案**：未探索更先进的 PTO 技术（如 SmoothQuant、QuaRot）。
- **数据偏差**：使用当前 S&P 500 成分股，存在 survivorship bias。
- **测试周期有限**：仅包含两个重大压力事件（2020、2022），泛化性有待验证。
- **评估维度单一**：仅使用 IC，缺乏 portfolio-level 或 risk-adjusted return 等实际交易指标。

### **未来工作方向**
- 探索 **更强的 PTO 方法**（如 channel-wise scaling、activation rotation）以减少 SegRNN/TimeMixer 的 residual loss。
- 引入 **dynamic calibration 或 online adaptation 机制**，应对 distribution shift。
- 扩展至其他金融市场（如债券、加密货币）和其他任务（如 return forecasting）。
- 进行 **真实硬件部署测试**，结合 latency、memory footprint 和 accuracy 三者权衡。
- 开发 **calibration-aware training 方法**，使模型对静态 range 更鲁棒。

---

> ✅ **一句话总结**：  
> 本文揭示了在金融时间序列预测中，**activation calibration 是决定 4-bit PTQ 成败的关键部署决策**，提出应结合架构特性、市场状态和 calibration envelope 动态选择量化策略，为低精度金融 AI 部署提供了首个系统性实证框架。

</details>

---

### 14. [From Prompting to Behavioral Alignment: Personalized LLM Judges for Recommendation Evaluation](https://arxiv.org/abs/2608.11493)

**Authors**: Alireza S. Ziabari, Kat Ellis, Colleen Chan, Ding Tong  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.11493v1  

#### Abstract
Traditional offline recommendation evaluation relies heavily on complex, manually maintained feature pipelines that are difficult to scale. While Large Language Models (LLMs) offer a promising alternative by predicting user engagement directly from raw text logs, empirical analysis in this study ide...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Prompting to Behavioral Alignment: Personalized LLM Judges for Recommendation Evaluation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统推荐系统的离线评估依赖复杂的、人工维护的 **feature pipelines**，难以扩展且常与线上 A/B 测试结果不一致。虽然 **Large Language Models (LLMs)** 被提出作为替代方案直接从原始日志中预测用户行为，但作者发现其存在一个关键失败模式——**bidirectional rationalization**（双向合理化）。

该问题表现为：在零样本（zero-shot）设定下，同一个 LLM 可以为同一用户历史和推荐项生成看似合理的“播放”（play）和“跳过”（skip）两种相反的推理路径，导致判断不可靠。这源于推荐系统中的根本权衡（如短期 vs 长期偏好、探索 vs 利用等），而未对齐的 LLM 缺乏解决这些冲突的原则依据。

此外，推荐质量具有高度主观性，通用 LLM 判断标准无法捕捉个性化意图。

---

### 🚀 提出的新方法与思路
作者提出一种 **端到端的行为对齐框架（behavioral alignment framework）**，通过两个阶段训练 LLM Judge：

1. **Supervised Fine-Tuning (SFT)**：在带有真实用户行为标签的 **Chain-of-Thought (CoT) 推理轨迹** 上进行监督微调。
2. **Offline Preference Optimization (DPO)**：在成对的“正确推理”与“反事实推理”之间进行偏好优化，使模型学会选择更符合实际用户行为的推理路径。

这种方法将 LLM 的推理过程锚定到真实的用户参与行为上，从而缓解 bidirectional rationalization。

---

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|--------|
| **无需手动特征工程** | 完全基于原始文本日志输入，无需构建复杂特征 pipeline，降低维护成本。 |
| **可解释性强** | 输出人类可读的 reasoning trace，揭示驱动预测的关键用户历史信号。 |
| **性能媲美生产级系统** | 在 Macro-F1 上达到与工业级特征工程模型相当的水平。 |
| **针对性解决个性化失败模式** | 不仅抑制幻觉（hallucination），更解决了因多义性推理路径导致的结构性失败。 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用 **Netflix 生产环境的真实首页交互日志**。
- 包含：
  - 用户观看历史（序列化为文本）
  - 当前会话上下文（时间、设备等）
  - 展示的推荐行（row of items）
  - 行为标签：`play` 或 `skip`

#### ✅ 标签构造方式（Spatial Scroll Heuristic）
- 若用户从某推荐行中播放了一个项目 → 该行为 `play`
- 所有位于其上方并被滚动经过的推荐行 → 标记为 `skip`（强负例）

此设计确保了正负样本均为高质量推荐，模拟真实决策场景。

---

### ⚙️ 实验设置

#### 模型
- 主要使用 **Llama 3.1 8B** 作为 backbone 模型。
- Teacher Model 使用更大容量的 reasoning-capable LLM 生成训练用 CoT 轨迹。

#### 输入格式
将原始日志转换为结构化自然语言 prompt，包含四个部分：
1. 任务指令
2. 推荐行内容
3. 即时会话上下文
4. 序列化用户历史（最多保留最近 50 条事件）

---

### 📈 评估指标
- **主指标**：**Macro-F1 Score**  
  （平衡 `play` 和 `skip` 类别的精确率与召回率，避免类别偏差影响）
- 辅助分析指标：
  - **Positive Bias**：`play` / `skip` 预测比例（理想值为 1.0）
  - Reasoning trace 质量（是否忠实反映决策逻辑）

---

### 🆚 基线方法对比
| 基线类型 | 具体方法 |
|--------|-------|
| **Zero-shot LLM Judge** | 使用 Llama 3.1 8B + Simple Prompt（无推理步骤） |
| **Feature-Engineered Production Baseline** | 工业界标准：手工特征 + 神经网络打分模型（内部部署系统） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 4）

| 方法 | Inference Prompt | Macro-F1 Lift (%) | Positive Bias |
|------|------------------|--------------------|---------------|
| Zero-shot | Simple | 0.0 | 3.01 |
| Zero-shot | Reason | 4.21 | 0.79 |
| SFT (Simple) | Simple | 12.23 | 1.79 |
| DPO (Simple) | Simple | 28.37 | 0.70 |
| **SFT + DPO (Best)** | **Reason** | **32.19** | **1.05** |

> ✅ 最佳配置：**SFT 后接 DPO，推理时启用 reasoning prompt**

---

### 🔁 与基线对比结果
- **相比 zero-shot baseline**：提升 **32.19% Macro-F1**
- **相比生产级特征工程模型**：达到 **统计等效水平**（Macro-F1 差异 < 0.1%）
- 这是首次证明纯文本驱动的 LLM Judge 可以在真实工业场景中 **匹敌 heavily feature-engineered 系统**

---

### 🔍 消融实验结果（Ablation Studies）

#### (1) Prompt Engineering 分析（RQ1）
- **Session Context** 是最关键因素之一，显著提升准确性。
- **Reasoning-based prompting** 比直接预测效果更好（+4.21% Macro-F1）。
- 更复杂的 prompt 结构（如 Evidence, Habit, Persona）反而 **降低性能**，可能引入噪声或过约束。
- 尝试反转任务（预测 skip 而非 play）加剧正类偏置，性能大幅下降。

👉 结论：prompt engineering 有一定作用，但不足以弥合与生产模型的差距。

#### (2) 训练范式比较（RQ2）
| 方法 | 效果分析 |
|------|---------|
| **SFT only** | 显著改善校准（减少 play 偏好），但未充分利用推理结构 |
| **DPO only** | 更有效于区分正确 vs 错误推理路径，优于 SFT |
| **SFT + DPO** | **最佳组合**：SFT 建立领域格式，DPO 优化决策偏好，实现最大增益 |

> 特别地，当在 SFT 中使用 reasoning prompt 数据但在 inference 时切换回 simple prompt，性能反而下降（-15.84%），显示存在 **task-specific overfitting**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Bidirectional Rationalization 是个性化评估的核心失败模式**
   - 并非由幻觉引起，而是因为相同证据支持多种合理但矛盾的推理路径（如短期 vs 长期偏好）。
   - 经过滤除虚假陈述后仍有 77% 的案例存在双向推理，说明这是结构性问题。

2. **Prompt Engineering 不足以解决问题**
   - 虽然加入 reasoning 和 session context 有助于提升性能，但无法消除根本性的推理歧义。
   - 复杂 prompt 设计甚至可能导致性能退化。

3. **Behavioral Alignment 是关键突破口**
   - 通过 SFT + DPO 对成对的正确/反事实推理进行训练，能有效教会模型识别“哪种推理更能预测真实行为”。
   - 成功将“unconstrained rationalizer”转变为“directional judge”。

4. **LLM Judge 可替代传统特征工程系统**
   - 在不依赖任何 hand-crafted features 的前提下，达到与生产模型相当的性能。
   - 同时提供可解释的 reasoning trace，具备透明性和调试价值。

---

### ⚠️ 方法的局限性
1. **历史长度受限**
   - 实验发现超过约 50 条历史记录后性能趋于饱和，长序列可能导致 attention dilution。
   - 当前方法难以有效利用深层长期行为模式。

2. **Teacher Model 依赖**
   - 训练数据依赖大模型生成的 CoT 轨迹，若 teacher 存在系统性偏差，可能传递至 student model。

3. **计算开销较高**
   - SFT + DPO 需要大量配对数据和训练资源，相比 zero-shot 推理成本更高。

---

### 🔮 未来工作方向
- **Intermediate User Profile Generation**
  - 不直接输入原始历史序列，而是将其压缩为简洁的自然语言 profile（如“浪漫剧爱好者 + 喜欢轻松喜剧”），帮助模型更好地整合长期信号。
- 扩展至更多行为目标（如完播率、评分预测）
- 探索轻量化 alignment 方法以适应更大规模部署

---

## 总结

| 维度 | 内容摘要 |
|------|--------|
| **核心思想** | 将 LLM 从“通用理性者”转化为“行为对齐的个性化判官” |
| **关键技术** | SFT + DPO on paired reasoning traces（正确 vs 反事实） |
| **核心发现** | Bidirectional rationalization 是个性化推荐评估的独特挑战，需通过行为对齐而非单纯提示工程来解决 |
| **最终成果** | 构建出无需手动特征工程、可解释、且性能媲美工业级系统的 LLM-based evaluator |

> 💡 **一句话总结**：  
> **个性化推荐评估不能靠“聪明的嘴”，而要靠“对齐的心”。唯有将 LLM 的推理路径锚定在真实用户行为上，才能让它成为可靠的裁判。**

</details>

---

### 15. [Measure, Don't Optimize: Forecasting Recovery in LLM Unlearning](https://arxiv.org/abs/2608.11408)

**Authors**: Zirui Song, Huaxing Liu, Xiang Wang, Shuai Li, Xinye Li, Lang Gao, Jinghui Zhang, Zheng Lu, Fengxian Ji, Xiaojun Chang, Xiuying Chen  
**Category**: cs.CL  
**Published**: 2026-08-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.11408v1  

#### Abstract
Prior white-box studies show that large language models can retain latent traces of target knowledge after unlearning, even when the knowledge is no longer expressed in their outputs. However, existing audits remain limited to one-off diagnostics: it is unclear whether these residual signals can pre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Measure, Don't Optimize: Forecasting Recovery in LLM Unlearning*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 LLM **machine unlearning** 研究主要依赖于 **behavioral evaluation**（如答案概率、ROUGE 分数等）来判断“遗忘”是否成功。然而，已有研究表明，即使模型在输出上不再表达某项知识，其内部参数或隐藏状态中仍可能保留该知识的**残余痕迹**（residual traces）。这引发两个关键问题：

1. 这些残余信号是否能**预测未来知识恢复的风险**（例如通过少量微调重新学习）？
2. 这些内部信号能否作为**优化目标**直接用于训练，以实现更彻底的删除？

现有研究仅提供静态的白盒审计（white-box audit），缺乏对上述问题的回答。

### 提出了什么新方法或新思路
本文提出 **J-AccEss** —— 一种基于 **Jacobian lens** 的推理时（inference-time）内部可访问性审计方法，用于衡量“被遗忘”知识在模型中间表示中的**残留可访问性**。

- **核心思想**：通过 Jacobian lens 将中间层的 residual representations 映射到词汇空间（vocabulary space），检查目标概念是否仍能在 top-k 解码结果中出现。
- **归一化设计**：将审计得分 $ J\text{-}AccEss $ 归一化于原始模型 $ \theta_o $ 和仅保留集训练的黄金模型 $ \theta_g $ 之间，使得不同检查点间具有可比性。
- **前瞻性验证**：首次系统性地评估内部审计信号在两个维度上的有效性：
  - **预测性**（prospective validity）：能否预测 relearning 攻击下的恢复程度？
  - **鲁棒性**（robustness under optimization）：当其成为训练目标时，是否仍可靠？

### 相比现有方法的优势
| 方面 | 现有方法（如 UDS） | J-AccEss |
|------|---------------------|---------|
| 测量机制 | 基于 activation patching 的因果干预 | 基于 Jacobian lens 的前向读出（readout） |
| 可扩展性 | 高计算开销，难以大规模应用 | 推理时即可完成，适用于大规模审计 |
| 几何不变性 | 对表示偏移敏感 | 利用 Jacobian 处理潜在的表示空间变化 |
| 验证维度 | 仅与当前行为相关 | 首次验证其对未来恢复的预测能力 |

此外，J-AccEss 不假设中间层与最终输出共享相同语义基底（shared basis），因此更适合处理经过 unlearning 后可能发生几何扭曲的表示。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **TOFU (Task of Fictitious Unlearning for LLMs)**：包含虚构作者的事实问答对，明确划分 forget set 和 retain set，提供 gold model $ \theta_g $。
- **OpenUnlearning benchmark**：整合了多个 unlearning 方法发布的 398 个公开检查点，涵盖八种主流 unlearning 方法。

### 实验设置和评估指标

#### 审计对象
共审计 **398 个 unlearned 模型检查点**，来自以下八种方法：
- GradDiff, NPO, SimNPO, AltPO, IdkDPO, IdkNLL, UNDIAL, RMU

#### J-AccEss 审计流程
1. 构造 probe queries：隐含提及被遗忘实体但不包含答案。
2. 在预定义的中间层带宽（mid-to-late layers）提取 residual states。
3. 应用 Jacobian lens 映射至词汇空间，检查 concept tokens 是否进入 top-k（默认 k=10）。
4. 计算 raw access rate，并归一化为：
   $$
   J\text{-}AccEss(\theta) = \frac{A_k(\theta; L) - A_k(\theta_g; L)}{A_k(\theta_o; L) - A_k(\theta_g; L)}
   $$
   其中 $ L $ 是模型未行为输出目标概念的 probe 子集。

#### 主要实验任务
1. **Test 1: 残留访问是否存在？**
   - 检查 J-AccEss 是否高于 retain-only 黄金水平。
2. **Test 2: 能否预测恢复？**
   - 对 unlearned 模型进行 cross-entity relearning 攻击（fine-tune on subset of forgotten entities）。
   - 评估 pre-attack J-AccEss 与 post-attack recovery 的相关性。
   - 恢复指标：
     - **Excess Revival**：revival rate 减去同等攻击下黄金模型的表现。
     - **Steps-to-recover**：达到原模型一半回答概率所需的微调步数。
3. **Test 3: 能否作为优化目标？**
   - 设计 **WD-Train** 方法，在 unlearning 目标中加入对 J-AccEss 的惩罚项（权重 $ \lambda \in \{0,5,10\} $）。
   - 观察审计分数下降是否伴随真正的知识删除（通过 UDS 和 post-attack revival 验证）。

#### 基线对比
- **Behavioral metrics**：Forget Quality (FQ), Model Utility (MU), ROUGE, Truth Ratio
- **Membership Inference Attacks (MIA)**：六种 baseline MIA 方法
- **Logit Lens**：直接 unembedding 中间表示（无 Jacobian 传输）
- **UDS (Unlearning DepthScore)**：基于 activation patching 的因果删除深度测量

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### Test 1: 行为遗忘 ≠ 内部擦除
- **85% 的 unlearned 模型** 的 J-AccEss 高于 retain-only 黄金模型（表1）。
- 平均归一化 J-AccEss 得分为 **0.69**，意味着典型模型仅关闭了不到三分之一的内部差距。
- 即使是通过 FQ 和 MU 筛选的行为成功模型，也普遍存在显著的内部可访问性（图2）。

#### Test 2: J-AccEss 可预测模型级恢复风险
| 分析维度 | Spearman 相关系数（Excess Revival） | Steps-to-recover |
|--------|-------------------------------|------------------|
| 全体模型池化 | +0.35 ★ | -0.70 ★ |
| 控制 FQ/MU 后 | +0.33 ★ | — |
| 方法内加权平均 | +0.45 ★ | — |
| 知识层级变体 | +0.71 ★ | — |

> ★ 表示 p < 0.05 显著

👉 结论：**pre-attack J-AccEss 越高，recovery 越快、越彻底**，且这一趋势在控制行为指标后依然成立。

#### Item-Level 预测能力几乎为零
| 预测器 | AUROC（item-level） |
|-------|--------------------|
| Pre-attack probability | 0.582 |
| Best MIA | 0.579 |
| J-AccEss (knowledge-level) | 0.555 |
| Logit-lens accessibility | 0.548 |
| J-AccEss (preregistered) | 0.504 |
| J-AccEss + behavior + MIA（增量） | +0.0002 |

👉 结论：J-AccEss **无法可靠识别哪些具体事实会恢复**。

#### Test 3: 直接优化 J-AccEss 导致反效果
| 方法 / 设置 | J-AccEss ↓ | UDS (causal depth) | Post-attack Revival ↑ |
|-----------|------------|--------------------|------------------------|
| WD-Train λ=0 | 0.67 | 0.75 | 0.283 |
| WD-Train λ=5 | 0.57 | 0.78 | 0.347 |
| WD-Train λ=10 | 0.55 | 0.76 | **0.387** ↑ |
| GradDiff (deep) | 0.09 | 0.97 | **0.025** |
| RMU (deep) | 0.05 | 0.99 | **0.000** |

👉 关键发现：
- 增大 $ \lambda $ 成功降低 J-AccEss，但 **post-attack revival 明显上升**。
- UDS（因果删除深度）基本不变，说明知识并未真正删除。
- 相比之下，UDS 较深的 GradDiff/RMU 模型表现出极低的 revival。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **行为遗忘不等于内部擦除**  
   绝大多数 unlearned 模型仍保有目标知识的内部可访问性，远超 retain-only 模型水平。

2. ✅ **J-AccEss 是有效的模型级恢复风险预测器**  
   pre-attack J-AccEss 能稳定预测 relearning 攻击下的恢复速度与程度，具备前瞻性诊断价值。

3. ❌ **J-AccEss 不能用于 item-level 删除证书**  
   无法区分“会被恢复”和“保持抑制”的具体事实，预测 AUROC 接近随机。

4. ⚠️ **直接最小化 J-AccEss 有害！**  
   模型学会“隐藏”知识以规避审计，反而导致更强的知识恢复能力（higher revival），形成**对抗性逃避**（audit evasion）。

5. 🔍 **解释粒度差异的原因**  
   - 知识恢复可能是通过重新激活一个**共享的检索路径**（shared retrieval pathway），而非逐条恢复独立记忆。
   - 因此，整体路径的“距离输出通路多远”决定了恢复难易（checkpoint-level），而单个事实的命运由攻击动态决定（item-level 不可预测）。

### 方法的局限性
- **J-AccEss 是诊断工具，非优化目标**：不能直接转化为训练损失。
- **依赖 Jacobian 近似**：线性近似可能在深层非线性变换中失效。
- **concept token 定义主观性**：需人工构造或过滤（如 document frequency）。
- **仅适用于 decoder-only 模型**：架构限制当前实现。

### 未来工作方向
- 开发更具因果意义的内部审计方法，结合 patching 与 readout。
- 探索如何将 J-AccEss 与其他因果指标（如 UDS）联合使用，构建更稳健的 unlearning 评估框架。
- 研究“真正删除”的训练机制，避免仅抑制输出或逃避审计。
- 扩展至 multimodal unlearning 场景（如 audio-language models）。

---

> 📌 **核心口号总结**：  
> **Measure, Don't Optimize**  
> 内部审计应作为独立的诊断维度，用于评估 unlearning 的**残余脆弱性**（residual susceptibility），而不应盲目转为优化目标，否则可能导致更危险的结果。

</details>

---

### 16. [Hybrid Gated Attention](https://arxiv.org/abs/2608.11805)

**Authors**: Zekun Zhou, Ruobing Xie, Lanrui Wang, Weixuan Sun  
**Category**: cs.CL  
**Published**: 2026-08-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.11805v1  

#### Abstract
Gated attention is an effective approach to mitigate attention sinks and enhance the representational capacity of attention. To further extend its effectiveness-efficiency Pareto frontier, we propose a Hybrid Gated Attention (HyGA) framework that contains three types of gating strategies. Specifical...

---

### 17. [FLARE++: Low-rank attention with dynamic attention routing](https://arxiv.org/abs/2608.11519)

**Authors**: Vedant Puri, Yongjie Jessica Zhang, Levent Burak Kara  
**Category**: cs.LG  
**Published**: 2026-08-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.11519v1  

#### Abstract
Full self-attention is a strong token mixer for PDE surrogates on irregular domains, but its quadratic cost limits its use on high-resolution problems. Efficient latent-attention models such as the Fast Low-rank Attention Routing Engine (FLARE) avoid that cost by routing all N tokens through M << N ...

---

### 18. [A Local Sinkhorn Framework for Conditional Distribution Reconstruction of Multidimensional Random Fields](https://arxiv.org/abs/2608.11613)

**Authors**: Mingtao Xia, Qijing Shen  
**Category**: cs.LG  
**Published**: 2026-08-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.11613v1  

#### Abstract
In this paper, we propose a local Sinkhorn divergence framework for conditional distribution reconstruction of multidimensional random fields. By utilizing the debiased Sinkhorn divergence, our proposed approach develops a differentiable and computationally efficient local distribution matching obje...

---

### 19. [Transferable Above-Ground Biomass (AGB) Estimation Model from Multi-Sensor Data with Sparse Field Calibration](https://arxiv.org/abs/2608.11638)

**Authors**: Pann Thinzar Seint, Bryan Atwood, Subas Chhatkuli  
**Category**: cs.LG  
**Published**: 2026-08-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.11638v1  

#### Abstract
Spatially continuous quantification of forest above-ground biomass (AGB) is what makes carbon accounting credible and mitigation strategies actionable. While field inventories provide high localized accuracy, they are spatially sparse; conversely, spaceborne LiDAR from the Global Ecosystem Dynamics ...

---

### 20. [TradingMoE: Routing the Right Experts in Evolving Markets](https://arxiv.org/abs/2608.11785)

**Authors**: Chang Zhou, Xingtong Yu, Minbin Huang, Zhennan Wu, Yuan Fang, Hong Cheng, Xinming Zhang  
**Category**: cs.LG  
**Published**: 2026-08-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.11785v1  

#### Abstract
Large language models (LLMs) have shown strong potential for financial analysis and trading, but direct trading remains challenging because the predictive capabilities required can vary across assets, decision fields, and market conditions. Existing LLM-based trading systems either coordinate human-...

---

### 21. [Air Quality Station Simulation via LSTM and Attention-Based Modelling](https://arxiv.org/abs/2608.11839)

**Authors**: Alexander Kostadinov, Petar O. Hristov, Dessislava Petrova-Antonova  
**Category**: cs.LG  
**Published**: 2026-08-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.11839v1  

#### Abstract
Poor air quality in urban areas is driven by a complex chain of processes and presents a significant public health concern. To better understand and control the mechanisms that determine air quality, cities deploy networks of measurement stations, and launch initiatives for collecting denser data ab...

---

### 22. [A Modular Agentic Framework for Synthetically Constrained Multi-Objective Hit-to-Lead Optimization](https://arxiv.org/abs/2608.11483)

**Authors**: Kelvin P. Idanwekhai, Enes Kelestemur, Benjamin Strickland, Matthew Hart, Steini Davidsson, Angelos Angelopoulos, Ron Alterovitz, Marcello DeLuca, Alexander Tropsha  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.11483v1  

#### Abstract
Hit-to-lead optimization requires iterative design of hit analogs across competing potency, selectivity, physicochemical, pharmacokinetic, safety, and synthetic constraints. We present SABLE (Synthetically-accessible Agentic Bayesian Ligand Exploration), an open-source framework that employs natural...

---

### 23. [AgenticTwin: An Agentic LLM Framework Integrated with Digital Twin for Anomaly Detection](https://arxiv.org/abs/2608.11679)

**Authors**: Touseef Hasan, Mounika Ghanta, Souvika Sarkar, Ujjwal Guin  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.11679v1  

#### Abstract
Digital twins are increasingly used to monitor and simulate the behavior of cyber-physical systems. Even with skilled operators, interpreting anomalies detected within digital twin pipelines is challenging, as the sheer complexity and volume of raw sensor data make thorough analysis difficult. Recen...

---

### 24. [Retry, Switch, or Abstain? Learning Strategy-Aware Tool-Use Policies via Controlled Error Injection](https://arxiv.org/abs/2608.11977)

**Authors**: Chaoran Chen, Vy Nguyen, Ziji Zhang, Abhinav Gullapalli, Ziyi Wang, Yuxuan Lu, Dakuo Wang, Jing Huang, Zhou Yu, Jin Lai  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.11977v1  

#### Abstract
Tool-using LLM agents are commonly trained and evaluated in environments where tool calls succeed reliably, yet deployed tools can fail transiently, persistently, or silently. Robust recovery therefore requires more than repeated retries: an agent may need to retry the same path, switch to an altern...

---

### 25. [Benchmarking Trustworthiness of SLMs: Pre-trained vs. Compressed](https://arxiv.org/abs/2608.11981)

**Authors**: Haokun Lin, Kaijie Zhu, Haobo Xu, Yichen Wu, Zhichao Lu, Qingfu Zhang, Zhenan Sun  
**Category**: cs.CL  
**Published**: 2026-08-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.11981v1  

#### Abstract
Small Language Models (SLMs) have emerged as a more efficient alternative to traditional Large Language Models (LLMs), offering promising potential in resource-constrained scenarios. Existing approaches to building SLMs typically follow two paths: training compact models from scratch, or compressing...

---

### 26. [Localizing Safety Alignment: MLP Layers and Mid-Network Blocks Encode Refusal Behavior in Large Language Models](https://arxiv.org/abs/2608.11583)

**Authors**: Mingyu Zong, Sampad Mohanty, Bhaskar Krishnamachari  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.11583v1  

#### Abstract
Safety alignment in large language models is often treated as a distributed property of the entire network, yet its practical brittleness suggests that refusal behavior may be concentrated in a smaller set of parameters. This work addresses where safety-aligned refusal is encoded by transplanting we...

---

### 27. [MBA: Multimodal Benchmark and Agents for Real-World Business Ideation](https://arxiv.org/abs/2608.11616)

**Authors**: Hojun Choi, Jaeyo Shin, Suin Lee, Hyunjung Shim  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.11616v1  

#### Abstract
Agentic systems powered by large language models (LLMs) have opened new opportunities for business ideation. Yet existing approaches remain confined to a text-only paradigm, despite the inherently multimodal nature of real-world contexts. We thus introduce MBA-Bench, the first multimodal benchmark f...

---

### 28. [Agent Skills Can Be Harmful: An Empirical Study of Skill-Induced Failures in LLM Agents](https://arxiv.org/abs/2608.11888)

**Authors**: Gen Dong, Yanjie Gao, Liqun Li, Tianyin Xu, Yu Hua, Fan Yang  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.11888v1  

#### Abstract
Agent skills are the de facto mechanism for extending LLM agents with reusable guidance. A skill can shape the agent's task execution, including planning, tool use, problem-solving, and validation. Prior work reported mixed results of agent skills: some skills improve task success rates, while other...

---

### 29. [Claim-Level Reliability Assessment for Efficient Test-Time Reasoning](https://arxiv.org/abs/2608.11994)

**Authors**: Sen Xu, Wei Wang, Shixi Liu, Jixin Min, Yingwei Dai, Zhibin Yin, Yirong Chen, Junlin Zhang  
**Category**: cs.AI  
**Published**: 2026-08-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.11994v1  

#### Abstract
We propose claim-level falsification as a principle for test-time scaling and instantiate it through Claim-Level Reliability Assessment (CLR), a training-free framework that reallocates test-time compute from additional solution sampling to targeted verification. Since whole-trace evaluation often o...

---

### 30. [Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus](https://arxiv.org/abs/2608.12149)

**Authors**: Zunhai Su, Bohan Sun, Xialie Zhuang, Shuibai Zhang, He Xiao, Jing Xiong, Hengyuan Zhang, Zhongzhu Zhou, Tiantian Zhang, Ngai Wong, Chuan-Wei Kuo  
**Category**: cs.CL  
**Published**: 2026-08-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.12149v1  

#### Abstract
We present the first systematic study of Massive activations (MAs) in layer-interleaved HLA LLMs and uncover two architecture-aligned morphologies: MAs consistently spike immediately before full attention layers, forming pre-attention spikes (PAS), and can persist through intervening linear attentio...

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

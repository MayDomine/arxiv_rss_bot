# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-12 06:37:30 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [S-HPLB: Efficient LLM Attention Serving via Sparsity-Aware Head Parallelism Load Balance](https://arxiv.org/abs/2603.10353)

**Authors**: Di Liu, Yifei Liu, Chen Chen, Zhibin Yu, Xiaoyi Fan, Quan Chen, Minyi Guo  
**Category**: cs.DC  
**Published**: 2026-03-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.10353v1  

#### Abstract
With the increasing volumes of Large Language Models (LLMs) and the expanding context lengths, attention computation has become a key performance bottleneck in LLM serving. For fast attention computation, recent practices often parallelize the attention heads on multiple GPUs, and also widely adopt ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：S-HPLB: Efficient LLM Attention Serving via Sparsity-Aware Head Parallelism Load Balance

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
随着 Large Language Models (LLMs) 规模和上下文长度（context length）的不断增长，**attention 计算已成为 LLM 推理服务中的关键性能瓶颈**，尤其是在 prefill 阶段。尽管已有研究通过 **distributed attention deployment** 和 **sparse attention** 来优化计算效率，但仍存在以下两个核心问题：

- **跨注意力头的 sparsity heterogeneity 被忽视**：传统方法对所有 attention head 施加统一的 sparsity budget（如 top-k），导致高稀疏性 head 出现冗余计算，而低稀疏性 head 则精度损失严重。
- **head-parallelism 下的负载不均衡**：当不同 head 分配不同计算量时，由于各 GPU 上的 attention 计算时间不一致，会引发严重的 **cross-GPU resource bubbles**（资源空转），降低系统吞吐。

### 提出了什么新方法或新思路
本文提出了一种 **系统-算法协同设计** 的新机制 —— **S-HPLB (Sparsity-Aware Head-Parallel Load Balance)**，包含两大核心组件：

1. **Adaptive Head Budget Allocation（自适应头预算分配）**
   - 发现：每个 attention head 的 **sparsity pattern 在不同输入下具有高度稳定性**（cross-request stability）。
   - 方法：通过离线 profiling 获取每 head 的稀疏特性，并采用 **max-min 优化策略** 进行预算再分配 —— 将预算从“高稀疏”head 转移给“低稀疏”head，在总计算量不变的前提下提升整体精度。

2. **Head Parallel Load Balance（头并行负载均衡）**
   - 问题建模：将 head 到 GPU 的映射建模为经典的 **multiway partitioning problem**。
   - 算法设计：提出一种 **贪心启发式算法**，优先将大预算 head 分配给当前负载最小的设备，以最小化跨设备负载差异。

### 相比现有方法的优势
| 维度 | 传统方法（如 top-k / top-p） | S-HPLB |
|------|-------------------------------|--------|
| **精度控制** | 固定 k 或动态 p，难以平衡精度与效率 | 基于稳定性的离线建模 + max-min 分配，更精准高效 |
| **系统效率** | 忽视负载不均，导致 GPU idle 时间长 | 显式优化部署策略，显著减少 resource bubbles |
| **实现开销** | top-p 需在线分析 attention map，延迟高 | 离线 profiling + 贪心调度，运行时代价极低 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要基准测试使用 **RULER benchmark [9]**，涵盖四大类共 13 项任务：
  - Retrieval（检索）
  - Multi-hop tracing（多跳追踪）
  - Aggregation（聚合）
  - QA（问答）
- 支持长达 **128K tokens 的上下文长度**，用于评估 long-context 场景下的表现。
- 输入来自多个领域（classification, generation, reasoning），验证泛化能力。

### 实验设置和评估指标

#### 模型
- **Llama-3.1-8B**
- **Qwen2.5-7B**
- **Qwen2.5-72B**
- **Llama-3-8B-262K**（用于扩展性测试）

#### 硬件平台
- 单服务器配置：
  - CPU: Intel Xeon Platinum 8369B
  - GPU: 8 × NVIDIA A100 80GB（NVLink 连接）
  - OS: CentOS 7, CUDA 12.4, PyTorch 2.6.0

#### 评估指标
1. **Model Accuracy**：在 RULER 各子任务上的得分，取平均值。
2. **Average Attention Serving Latency**：以 Time-To-First-Token (TTFT) 衡量，重点关注 prefill 阶段的 attention 计算延迟。

#### 并行度设置
- 测试不同 degree of **Head Parallelism (HP=2/4/8)** 对性能的影响。

### 基线方法对比
| 类别 | 方法 | 描述 |
|------|------|------|
| Full Attention Baseline | FlashAttention [6] | 完整 attention 计算 |
| Top-k Sparse Methods | StreamingLLM [27], MInference [10] | 固定 token budget，不同稀疏模式 |
| Top-p Sparse Method | XAttention [29] | 动态确定预算以满足累计 attention weight 阈值 p |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 精度表现（Accuracy on RULER）
| 方法 | Llama-3.1-8B | Qwen2.5-7B | Qwen2.5-72B |
|------|---------------|-------------|--------------|
| Full Attention | 76.38 | 67.46 | 83.69 |
| S-HPLB | **75.86** | **66.09** | **80.56** |
| XAttention (best sparse baseline) | 73.29 | 63.45 | 79.95 |

- S-HPLB 的精度仅比 full attention 低 **0.52% / 1.37% / 3.13%**，远优于其他 sparse 方法。
- 相比 XAttention，S-HPLB 分别提升 **+2.57pp / +2.94pp / +0.61pp** 的平均准确率。
- 在某些任务上（如 MK3 on Llama-3.1-8B），S-HPLB 甚至 **超过 full attention**，归因于其能过滤噪声、聚焦关键 token。

#### ✅ 延迟表现（Attention Latency Reduction）
在 128K context length 下，相比 full attention：
- S-HPLB 实现 **3.39× / 4.27× / 3.31× 的延迟下降**（分别对应三个模型）。

相比最先进的 top-p 方法 XAttention：
- S-HPLB 实现 **2.09× / 2.22× / 2.88× 的延迟优势**，最高达 **2.88× 加速**。

> 📌 特别地，在 Qwen2.5-72B 上达到 **2.88× 更低延迟**，同时保持接近 full attention 的精度。

#### ✅ 效率-精度权衡（Pareto Frontier）
- 如 Figure 10 所示，S-HPLB 在 **latency-accuracy skyline 上始终位于帕累托前沿（Pareto frontier）**，即在相同延迟下精度更高，或在相同精度下延迟更低。

### 消融实验结果（Ablation Study）

#### Head Parallel Load Balancer 的独立贡献
- 移除 load balancer 后，直接按原始顺序分配 head 至 GPU，会导致严重负载失衡（imbalance up to 2.78×）。
- 引入 S-HPLB 的 load balancer 后：
  - 最高可带来 **1.26× 的延迟降低**（Figure 11b）。
  - 在不同并行度下平均减少 **1.19× 延迟**。

> 🔍 结论：**load balancing 模块本身即可带来显著性能增益**，证明其必要性。

---

## 4. 关键结论和发现

### 主要发现
1. **Attention heads 存在显著且稳定的跨头稀疏异质性（sparsity heterogeneity）**：
   - 不同 head 达到相同 recovery ratio 所需 token 数差异巨大。
   - 该特性在不同 context length 和任务间高度稳定，支持离线建模。

2. **uniform sparsity budget 是次优选择**：
   - 统一 k 导致“富者愈富、贫者愈贫”式的资源错配。
   - 自适应预算分配可在不增加总计算量前提下显著提升精度。

3. **系统级负载均衡至关重要**：
   - 即使算法侧优化得当，若部署不当仍会造成严重 GPU idle。
   - S-HPLB 的 greedy 多路划分策略有效缓解了这一问题。

4. **S-HPLB 实现了算法与系统的协同增效**：
   - 算法端提升精度 → 系统端提升效率 → 整体端到端性能跃升。

### 方法的局限性
- **依赖离线 profiling**：需要预先在一个 calibration dataset 上完成 sparsity pattern 分析，可能增加部署复杂性。
- **假设 head 间独立性**：未考虑 head 间的交互影响或 layer-wise 变化趋势。
- **适用于 prefill 阶段为主**：focus on computation-bound prefill phase，对 decoding 阶段 memory-bound 场景优化有限。

### 未来工作方向
- **自动化 profiling pipeline**：开发自动化的 calibration 工具链，降低人工干预成本。
- **动态自适应机制**：探索轻量级在线调整机制，在保证效率的同时应对极端输入变化。
- **扩展至 MoE 架构**：将 S-HPLB 思想推广至 Mixture-of-Experts 模型中的 expert-level 并行调度。
- **结合 KV Cache 优化**：与 PagedAttention、Retrieval-based caching 等技术联合优化 long-context inference 全流程。

---

> ✅ **总结一句话**：  
> **S-HPLB 通过“感知稀疏性”的头预算分配 + “负载感知”的并行调度，在几乎无损精度的前提下实现了高达 2.88× 的 attention 计算加速，是 LLM attention serving 领域一次成功的 system-algorithm co-design 实践。**

</details>

---

### 2. [Cluster-Aware Attention-Based Deep Reinforcement Learning for Pickup and Delivery Problems](https://arxiv.org/abs/2603.10053)

**Authors**: Wentao Wang, Lifeng Han, Guangyu Zou  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.10053v1  

#### Abstract
The Pickup and Delivery Problem (PDP) is a fundamental and challenging variant of the Vehicle Routing Problem, characterized by tightly coupled pickup--delivery pairs, precedence constraints, and spatial layouts that often exhibit clustering. Existing deep reinforcement learning (DRL) approaches eit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Cluster-Aware Attention-Based Deep Reinforcement Learning for Pickup and Delivery Problems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对 **Pickup and Delivery Problem (PDP)** 这一经典组合优化问题展开研究。PDP 是车辆路径问题（VRP）的重要变体，其挑战在于：
- 存在严格的 **precedence constraints**（先取货后送货）
- 节点间存在逻辑配对关系（pickup-delivery pairs）
- 实际场景中常呈现 **空间聚类结构**（如取货集中在住宅区，送货集中在商业区）

现有基于 **Deep Reinforcement Learning (DRL)** 的神经求解器通常采用扁平图建模（flat graph），依赖隐式学习来捕捉约束和结构，难以有效利用多尺度的空间规律。此外，一些高性能方法依赖推理时的协作搜索（如 NCS），导致 **inference latency 高**。

---

### 提出了什么新方法或新思路
作者提出了一种名为 **CAADRL (Cluster-Aware Attention-Based Deep Reinforcement Learning)** 的新型 DRL 框架，核心创新如下：

#### ✅ Cluster-Aware Encoder（聚类感知编码器）
- 在标准 Transformer 编码器基础上引入 **Cluster-Aware Attention Mechanism**
- 同时执行两种注意力机制：
  - **Global Self-Attention**：捕获全局节点间的空间依赖
  - **Intra-Cluster Attention**：通过 `cluster mask` 限制每个节点只关注同簇内其他节点（如所有 pickup 点之间、delivery 点之间）
- 输出的嵌入向量兼具 **全局一致性** 和 **局部角色感知能力**

#### ✅ Hierarchical Dynamic Dual-Decoder（分层动态双解码器）
- 设计两个并行解码路径：
  - **Intra-Cluster Decoder**：专注于簇内的精细路由决策
  - **Inter-Cluster Decoder**：处理跨簇转移
- 引入一个可学习的 **gating module**，在每一步动态决定是“留在当前簇”还是“跳转到另一簇”
- 实现了 **one-pass autoregressive construction**，无需迭代改进

#### ✅ POMO-based End-to-End Training
- 将 **POMO (Policy Optimization with Multiple Optima)** 框架适配至 PDP 场景
- 利用实例对称性生成多个 rollout，提升训练稳定性和样本效率

---

### 相比现有方法的优势
| 维度 | CAADRL | 典型基线（如 NCS, Heter-AM） |
|------|--------|-----------------------------|
| **结构建模** | 显式建模聚类结构 | 扁平图建模，隐式学习 |
| **解码方式** | 分层决策，双解码器 + 门控 | 单一解码器 |
| **推理效率** | 一次前向传播即可构造完整路径 | 多轮协作搜索 → 高延迟 |
| **归纳偏置** | 明确注入 cluster-aware 先验 | 更通用但缺乏针对性 |

> 💡 **核心优势**：**显式建模聚类结构提供了更强的归纳偏置（inductive bias），在保持低推理延迟的同时实现了高求解质量。**

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 所有数据为合成生成于二维平面 $[0,1] \times [0,1]$ 上的 PDP 实例
- 包含两类分布以测试模型鲁棒性：

| 数据分布 | 描述 |
|--------|------|
| **Clustered Distribution** | 取货点从 $(0.25, 0.25)$ 正态采样，送货点从 $(0.75, 0.75)$ 正态采样（$\sigma=0.1$），形成明显空间聚类 |
| **Uniform Distribution** | 所有节点（含 depot）均匀分布在单位正方形内，无显著聚类结构 |

- 测试规模涵盖四种问题大小：
  - **PDP10, PDP20, PDP40, PDP80**（分别对应 5, 10, 20, 40 对 pickup-delivery）

---

### 实验设置和评估指标

#### 评估指标
- **Objective (Obj.)**：平均总旅行距离（越小越好）
- **Gap (%)**：相对于最优或最佳基线的相对差距
- **Time (秒)**：平均每实例推理时间（测试阶段）

#### 解码策略
对所有构造型模型（CAADRL, Heter）测试三种策略：
- **Greedy**：每步选择概率最高的动作
- **Sampling-1280**：生成 1280 条路径，返回最优
- **Sampling-12800**：生成 12800 条路径，逼近策略上限

#### 训练细节
- 使用 **PyTorch**
- 编码器：6 层 dual-attention Transformer，embedding dim=128，8 heads
- 优化器：Adam ($lr = 1\times10^{-4}$)，batch size=512，训练 800 轮
- 在线生成训练数据，测试集固定为 100 个实例/配置

---

### 基线方法对比
| 基线方法 | 简介 |
|--------|------|
| **NCS (Neural Collaborative Search)** | 当前 SOTA 方法之一，结合神经构造 + 神经邻域搜索模块；通过增加迭代次数（t=1k/2k/3k）提升性能但代价是更高延迟 |
| **Heter-AM (Heterogeneous Attention Model)** | 使用角色感知注意力区分 depot/pickup/delivery 节点，代表先进的单次构造模型 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### 📊 在 **Clustered Instances** 上的表现（Table 1）
| 方法 | PDP80 Obj. | Gap (%) | Time (s) |
|------|-----------|--------|---------|
| **CAADRL (Sample12800)** | **4.709** | **0.00** | 0.198 |
| ncs(t=3k) | 4.734 | 0.53 | 0.444 |
| Heter(Sample12800) | 4.737 | 0.59 | 0.594 |

✅ **结论**：在最大规模聚类实例上，CAADRL 达到最优目标值，且 **推理速度远快于 NCS 和 Heter**

#### 📊 在 **Uniform Instances** 上的表现（Table 2）
| 方法 | PDP80 Obj. | Gap (%) | Time (s) |
|------|-----------|--------|---------|
| **CAADRL (Sample12800)** | **9.413** | **0.00** | 0.201 |
| ncs(t=3k) | 10.080 | 7.09 | 0.457 |
| Heter(Sample12800) | 10.101 | 7.31 | 0.596 |

✅ **惊人发现**：即使在没有显式聚类的 uniform 实例上，CAADRL 依然在 **PDP80** 上取得最佳表现，说明其架构具有强泛化能力。

---

### 消融实验结果（Table 3）

| 模型变体 | PDP100 (Sample12800) Obj. | 与 CAADRL 差距 |
|--------|----------------------------|----------------|
| **CAADRL (Full Model)** | **5.201** | — |
| no_encoder（替换为标准 Transformer） | 5.220 | +0.019 |
| no_decoder（移除双解码器） | 5.198 | -0.003（微弱优势） |
| POMO Baseline | 5.236 | +0.035 |

🔍 **分析**：
- 移除 **Cluster-Aware Attention** 导致性能下降，验证了其有效性
- 移除 **Dynamic Dual-Decoder** 影响较小，但在贪婪/有限采样下仍有帮助
- 表明 **encoder 的设计比 decoder 更关键**，但两者互补

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **显式建模聚类结构是一种有效的归纳偏置**  
   - 在 clustered PDP 上，CAADRL 显著优于 Heter-AM 和 NCS（尤其在中大规模）
   - 推理速度快 2–3 倍以上，适合实际部署

2. ✅ **该架构具备良好的泛化能力**  
   - 即使在无聚类的 uniform 实例上，CAADRL 仍保持竞争力
   - 在 **PDP80-uniform** 上反超所有基线，表明其学到的是更本质的多尺度路由模式

3. ✅ **One-pass 构造也能达到甚至超越迭代搜索的质量**  
   - CAADRL 不依赖推理时改进循环，却能达到与 NCS(t=3k) 相当甚至更好的性能
   - 打破了“高质量必须高延迟”的固有认知

4. ✅ **Hierarchical Decoding 提升决策清晰度**  
   - 通过门控机制显式区分“探索”与“开发”，减少模式混淆

---

### 方法的局限性
- **依赖预定义 cluster 定义**：目前 cluster ID 由节点角色（pickup/delivery/depot）静态指定，未实现自动聚类
- **静态单辆车设定**：未考虑多车、时间窗、动态请求等现实复杂因素
- **Euclidean 距离假设**：实际城市路网可能需图结构建模

---

### 未来工作方向（原文第6节）
1. **扩展至更复杂的 PDP 变体**：
   - 多车 PDP（multi-vehicle）
   - 带时间窗（time windows）
   - 动态请求到达（dynamic requests）
   - 无人机协同配送（drone-assisted delivery）

2. **学习动态聚类结构**：
   - 引入可微分聚类模块（differentiable clustering）或图分割层，让网络自适应识别空间簇

3. **工业级应用验证**：
   - 在真实物流数据集上测试
   - 集成至滚动时域控制（rolling-horizon framework）
   - 评估系统级延迟与人机协作接口

---

## 总结
> 🔍 **CAADRL 成功将“人类调度员”的直觉——“先在区域内完成服务，再转移到下一个区域”——形式化为神经网络中的 cluster-aware attention 与 hierarchical decoding 结构，在不牺牲推理效率的前提下，显著提升了 DRL 求解 PDP 的性能与泛化能力。**

该工作为 **Neural Combinatorial Optimization** 提供了一个重要范式：**与其堆叠更多注意力层或引入昂贵的搜索机制，不如深入理解问题结构，并将其作为强归纳偏置注入模型设计之中。**

</details>

---

### 3. [Surrogate models for nuclear fusion with parametric Shallow Recurrent Decoder Networks: applications to magnetohydrodynamics](https://arxiv.org/abs/2603.10678)

**Authors**: M. Lo Verso, C. Introini, E. Cervi, L. Savoldi, J. N. Kutz, A. Cammi  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.10678v1  

#### Abstract
Magnetohydrodynamic (MHD) effects play a key role in the design and operation of nuclear fusion systems, where electrically conducting fluids (such as liquid metals or molten salts in reactor blankets) interact with magnetic fields of varying intensity and orientation, which affect the resulting flo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Surrogate models for nuclear fusion with parametric Shallow Recurrent Decoder Networks: applications to magnetohydrodynamics*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**核聚变系统中磁流体动力学（Magnetohydrodynamics, MHD）模拟计算成本高昂**的问题展开研究。具体而言：
- MHD模型涉及强非线性、多物理场耦合的偏微分方程组，求解耗时且资源密集。
- 在需要实时响应或多查询场景（如控制、监测、不确定性量化）下，传统高保真数值模拟（Full-Order Models, FOMs）难以满足效率需求。
- 尤其在液态金属（如铅锂合金 PbLi）冷却包层中，磁场对流动行为影响显著，但实验数据稀缺、仿真昂贵。

因此，亟需一种**高效、可靠、低测量依赖的代理模型（surrogate model）** 来实现快速状态重建与预测。

---

### 🚀 提出的新方法与创新思路

作者提出并首次将 **SHallow REcurrent Decoder (SHRED)** 网络应用于核聚变相关的 MHD 物理建模，构建了一个**基于SVD压缩与SHRED结合的数据驱动框架**，用于从稀疏传感器测量中重建完整的时空MHD状态。

#### 主要创新点包括：

1. **首次将SHRED引入MHD领域**  
   这是SHRED架构在核聚变MHD问题中的首次应用，拓展了其在复杂多物理场系统中的适用边界。

2. **参数化SHRED框架设计**  
   扩展原始单参数SHRED为**parametric SHRED**，使其能够泛化到训练集中未见的磁场强度（$B_0$），支持跨参数条件的状态估计。

3. **极简传感配置下的高性能重建**  
   仅使用**三个温度传感器的时间序列输入**，即可准确重构速度（u）、压力（p）和温度（T）的完整空间分布，极大降低了硬件部署要求。

4. **融合降维与深度学习的轻量级范式**  
   利用 **Singular Value Decomposition (SVD)** 对高维快照进行压缩，在低维潜空间中训练SHRED，大幅减少训练时间和内存消耗，可在普通笔记本电脑上完成训练。

5. **传感器位置无关性（sensor-agnostic）**  
   验证了SHRED对传感器布局具有高度鲁棒性——即使30组随机生成的传感器位置，重建结果差异极小，无需优化布点。

---

### ⚖️ 相比现有方法的优势

| 方面 | 传统方法（如FOM/DMD/ROM） | 本文方法（SVD + SHRED） |
|------|----------------------------|--------------------------|
| **计算效率** | 高计算开销，不适合实时 | 极低推理时间（<1秒），适合实时控制 |
| **数据需求** | 需大量高质量数据 | 可在少量传感器下工作（仅3个） |
| **可解释性** | 黑箱模型较多（如深度神经网络） | 参数极少（<10³），理论基础扎实（Takens定理） |
| **泛化能力** | 多数局限于固定参数 | 能外推至未见参数值（如新 $B_0$） |
| **工程实用性** | 依赖精确布点优化 | 支持任意布点，适应严苛几何约束 |

---

## 2. 核心实验方法和设置

### 🔧 数据集来源与特性

- **物理场景**：二维阶梯通道内的可压缩铅锂（PbLi）MHD流动，存在热梯度与垂直外加磁场 $B_0$。
- **仿真工具**：采用开源软件 OpenFOAM 的 `magnetoHDFoam` 库进行高保真模拟。
- **参数范围**：
  - 磁场强度 $B_0 \in [0.01, 0.5]$ T，共 $N_p = 19$ 组不同取值。
  - 时间步长自适应，总模拟时间 3 秒，每 0.025 秒保存一次快照 → 每例 $N_t = 120$ 个时间点。
- **字段变量**：速度（u）、压力（p）、温度（T）、磁场（B）等。

> 总计生成数百GB的FOM数据，运行于HPC集群，每例约耗时20分钟。

---

### 🛠 实验设置

#### 数据预处理流程：
1. **归一化**：使用 min-max 归一化处理所有场量。
2. **SVD压缩**：
   - 将所有参数下的快照矩阵堆叠后执行 SVD。
   - 保留前 $r=20$ 个主模态，确保累计信息保留 >99.9%。
   - 得到低维时间系数 $V^{(p)} \in \mathbb{R}^{r \times N_t}$ 作为“真实标签”。

#### 模型输入输出：
- **输入**：仅来自3个随机布置传感器的温度时间序列。
- **输出**：通过SHRED解码恢复全部场量（T, u, p）在全域的空间-时间演化。

#### 模型架构细节（PyTorch实现）：
- **LSTM编码器**：2层，每层64神经元 → 学习时间动态。
- **Shallow Decoder Network (SDN)**：2层，350 & 400神经元 → 映射潜变量回压缩空间。
- **SVD反变换层**：将低维表示还原为原始高维状态空间。

#### 训练策略：
- 数据划分：~73.7% 训练集，~15.8% 验证集，~10.5% 测试集。
- 强调低 $B_0$ 区域采样更密（因湍流更强、动态更复杂）。
- 使用30组不同的三传感器组合进行**集成训练（ensemble mode）**，增强统计稳健性。

---

### 📊 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **相对 $L^2$ 误差** | $\epsilon_w = \|w_{\text{FOM}} - w_{\text{SHRED}}\| / \|w_{\text{FOM}}\|$ | 衡量各场量重建精度 |
| **均值与标准差（std）** | 在30个模型输出上计算均值与偏差 | 评估模型一致性及传感器无关性 |
| **全局平均误差** | 时间维度上的平均相对误差 | 综合评价整体性能 |
| **可视化残差图与时间演化曲线** | 展示局部误差分布与时序一致性 | 直观分析模型表现 |

---

### 🔍 基线方法对比（文中隐含）

尽管未直接列出其他AI/ROM方法的定量比较，但从上下文可推知对比对象包括：

- **传统ROM方法**（如POD-Galerkin）：需物理建模介入，难以处理强非线性。
- **纯数据驱动深度学习模型**（如CNN-RNN、Transformer）：参数量大、训练慢、需大量数据。
- **Dynamic Mode Decomposition (DMD)** 类方法：线性假设限制其表达能力。

而SHRED因其浅层结构、压缩训练、理论支撑，在**效率、泛化、可解释性**方面具备综合优势。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 测试案例一：弱磁场 $B_0 = 0.06\,\text{T}$
- **最大相对误差**：
  - 温度（T）：< 3%
  - 速度（u）和压力（p）：< 6%
- **误差随时间轻微增长**，但仍保持低位（未达稳态所致）。
- **标准差极小**：30个模型间输出几乎一致，验证传感器无关性。

#### ✅ 测试案例二：强磁场 $B_0 = 0.3\,\text{T}$
- **误差更低且更稳定**：
  - 所有场量最终稳定在 **~2% 以内**
- 原因：强磁场抑制小尺度涡旋，促进层流化，动态更平滑易建模。

#### ✅ 全局平均误差（Figure 10-c/d）
| 场量 | $B_0 = 0.06\,\text{T}$ | $B_0 = 0.3\,\text{T}$ |
|-------|------------------------|------------------------|
| T     | ~2.5%                  | ~1.8%                  |
| u     | ~3.0%                  | ~1.9%                  |
| p     | ~3.2%                  | ~2.0%                  |
| std   | < 0.5%                 | < 0.3%                 |

> 所有误差均低于 **3%**，表明高精度重建。

---

### 🔄 推理效率
- **训练时间**：每个SHRED模型约 **10分钟**（Intel i7-9800X 笔记本）。
- **推理时间**：每次预测 **< 1秒**，接近实时。
- **内存占用低**：得益于SVD压缩与浅层网络。

---

### ❌ 消融实验（文中未明确开展）
论文未提供系统的消融研究（ablation study），例如：
- 不同传感器数量的影响（虽提及3已足够饱和）
- 是否加入参数 $B_0$ 作为输入的影响
- 不同SVD截断秩 $r$ 的敏感性分析

但通过多组随机传感器配置的ensemble分析，间接验证了模型鲁棒性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **SHRED能以极低测量成本实现高精度MHD状态重建**  
   仅凭3个温度传感器，即可准确重构速度、压力、温度全场，突破传统“全观测”假设。

2. **模型具备强大的参数外推能力**  
   即使面对训练集中未出现的磁场强度（如 $B_0 = 0.3\,\text{T}$），仍能精准重建，适用于实际运行中未知工况。

3. **对传感器位置完全鲁棒**  
   30组随机布点产生的重建结果高度一致（std极小），说明无需复杂布点优化，极具工程实用价值。

4. **适用于极端非线性与多尺度动态**  
   成功捕捉从弱磁化湍流到强磁场层流的不同流动机制，体现其广泛适用性。

5. **计算效率极高，支持实时应用**  
   可部署于边缘设备或控制系统中，用于在线监控、故障诊断与闭环控制。

---

### ⚠️ 方法的局限性

1. **依赖高质量FOM数据集**  
   虽然训练快，但前期仍需大量高保真仿真数据生成，若FOM不准则代理模型受限。

2. **当前仅限单一参数变化（$B_0$ 强度）**  
   尚未扩展至多参数空间（如磁场方向、入口速度、温度梯度联合变化）。

3. **尚未验证三维复杂几何场景**  
   当前测试为二维理想化通道，真实包层几何更为复杂。

4. **缺乏严格的不确定性量化机制**  
   输出为确定性估计，未提供置信区间或概率预测。

---

### 🔮 未来工作方向

1. **向三维真实包层几何扩展**  
   结合STAR-MRS、EU-DEMO等实际反应堆设计，提升工程相关性。

2. **支持多参数与动态参数输入**  
   引入时间变磁场、脉冲操作等瞬态场景建模能力。

3. **融合物理约束（Physics-Informed Learning）**  
   在损失函数中嵌入MHD方程残差项，提高外推稳定性。

4. **实机集成与数字孪生应用**  
   将SHRED嵌入数字孪生平台，用于实时状态监测、异常检测与主动控制。

5. **结合稀疏实验数据进行迁移学习**  
   利用有限实验校准模型，缩小仿真与现实差距。

---

## 总结

> **SHRED为核聚变MHD系统提供了一种高效、轻量、鲁棒且易于部署的代理建模范式**。它成功实现了从极少数传感器信号出发，对复杂多物理场系统的全状态重建，并展现出优异的泛化能力和工程实用性，有望成为未来聚变反应堆**实时监控与智能控制的核心组件之一**。

</details>

---

### 4. [GLM-OCR Technical Report](https://arxiv.org/abs/2603.10910)

**Authors**: Shuaiqi Duan, Yadong Xue, Weihan Wang, Zhe Su, Huan Liu, Sheng Yang, Guobing Gan, Guo Wang, Zihan Wang, Shengdong Yan, Dexin Jin, Yuxuan Zhang, Guohong Wen, Yanfeng Wang, Yutao Zhang, Xiaohan Zhang, Wenyi Hong, Yukuo Cen, Da Yin, Bin Chen, Wenmeng Yu, Xiaotao Gu, Jie Tang  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.10910v1  

#### Abstract
GLM-OCR is an efficient 0.9B-parameter compact multimodal model designed for real-world document understanding. It combines a 0.4B-parameter CogViT visual encoder with a 0.5B-parameter GLM language decoder, achieving a strong balance between computational efficiency and recognition performance. To a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# GLM-OCR Technical Report 核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
GLM-OCR 旨在解决**现实世界文档理解系统中的效率与性能平衡难题**，具体包括：
- 传统 OCR 系统在复杂布局、公式、表格等场景下表现不佳；
- 多模态大模型（MLLMs）虽然性能强，但参数量大、推理慢、内存消耗高，难以部署于边缘设备或高并发生产环境；
- 自回归解码机制在确定性的 OCR 任务中效率低下。

### 提出的新方法与创新
1. **轻量化架构设计（0.9B 参数）**
   - 结合 **0.4B CogViT 视觉编码器** 和 **0.5B GLM 语言解码器**，构建紧凑高效的多模态模型。
   - 整体仅 0.9B 参数，兼顾高性能与低资源需求。

2. **Multi-Token Prediction (MTP) 机制**
   - 在训练和推理阶段引入 MTP，每步预测多个 token（平均生成 5.2 个 token/step），显著提升解码吞吐量（约 **50% 性能提升**）。
   - 通过共享参数控制额外显存开销，避免因多头预测导致内存爆炸。

3. **两阶段系统级流水线**
   - 第一阶段：使用 **PP-DocLayout-V3** 进行布局分析，识别文本、公式、表格区域；
   - 第二阶段：对各区域并行进行内容识别，提高鲁棒性和处理效率。

4. **统一的结构化生成框架**
   - 支持两种任务范式：
     - **Document Parsing**：输出 Markdown/JSON 格式的完整文档结构；
     - **Key Information Extraction (KIE)**：基于 prompt 输出指定 JSON schema 的结构化字段。

### 相比现有方法的优势
| 维度 | GLM-OCR 优势 |
|------|--------------|
| **模型大小** | 仅 0.9B，远小于多数竞争模型（如 Qwen3-VL-235B、Gemini-3 Pro） |
| **推理效率** | MTP + 并行区域识别带来更高吞吐量，适合高并发部署 |
| **结构化输出质量** | 在表格、公式、印章识别等复杂任务上达到 SOTA |
| **部署灵活性** | 支持本地部署（vLLM/SGLang/Ollama）、云服务 API 及微调（LLaMA-Factory） |

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 公共基准（Public Benchmarks）
| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **OmniDocBench v1.5** | 综合文档解析 | 包含文本、公式、表格、阅读顺序等多项子任务 |
| **OCRBench (Text)** | 文本识别 | 评估通用文本转录能力 |
| **UniMERNet** | 公式识别 | 数学表达式识别 benchmark |
| **PubTabNet** | 表格结构恢复 | 表格 HTML 结构重建 |
| **TEDS_TEST** | 表格结构相似度 | 使用 Tree Edit Distance Similarity 指标 |
| **Nanonets-KIE**, **Handwritten-KIE** | 关键信息提取 | 发票、手写表单等结构化抽取 |

#### 自研内部基准（In-House Benchmarks）
| 场景 | 描述 |
|-----|------|
| Code Document Parsing | 代码文档结构解析 |
| Real-world Table Extraction | 自然场景下的复杂表格识别 |
| Handwritten Text Recognition | 手写文本识别 |
| Multilingual OCR | 中英法西俄德日韩八语种混合识别 |
| Seal Recognition | 公章识别 |
| Receipt KIE | 收据关键信息提取 |

### 实验设置与评估指标
| 任务 | 主要指标 |
|------|---------|
| 文本识别 | Text Score（基于编辑距离） |
| 公式识别 | CDM Score |
| 表格识别 | TEDS、PubTabNet Acc |
| 文档解析综合性能 | OmniDocBench Overall Score |
| KIE | Field-level F1 Score |
| 推理效率 | 吞吐量（pages/sec 或 images/sec） |

### 基线方法对比
涵盖以下三类模型：
- **Pipeline 工具**：PP-StructureV3、Marker-1.8.2
- **通用 VLMs**：GPT-4o, Qwen3-VL, Gemini-3 Pro, InternVL
- **专用 OCR VLMs**：PaddleOCR-VL-1.5, Deepseek-OCR, MinerU2.5, dots.ocr

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3 & 4）

| 模型 | OmniDocBench v1.5 | OCRBench (Text) | UniMERNet | PubTabNet | TEDS_TEST | Nanonets-KIE | Handwritten-KIE |
|------|-------------------|------------------|-----------|------------|------------|---------------|------------------|
| **GLM-OCR** | **94.6** ✅ | **94.0** ✅ | **96.5** ✅ | 85.2 | **86.0** ✅ | **93.7** ✅ | **86.1** ✅ |
| PaddleOCR-VL-1.5 | 94.5 | 75.3 | 96.1 | 84.6 | 83.3 | — | — |
| dots.ocr | 88.4 | 92.1 | 90.0 | 71.0 | 62.4 | — | — |
| MinerU2.5 | 90.7 | — | — | **88.4** | 85.4 | — | — |
| Gemini-3 Pro | 90.3 | 91.9 | 96.4 | 91.4 | 81.8 | 95.2 | 94.5 |

> ✅ 表示在同类可比模型中排名第一（排除闭源参考模型）

#### 子项亮点（OmniDocBench v1.5 分解）
| 指标 | GLM-OCR 表现 |
|------|-------------|
| **Table TEDS** | **93.96**（最高） |
| **Table TEDS-S** | **96.39**（最高） |
| Text Edit | 0.040（略低于 PaddleOCR-VL-1.5 的 0.035） |
| Formula CDM | 93.90（仅次于 PaddleOCR-VL-1.5 的 94.21） |

### 与基线方法的对比结果
- 在 **OmniDocBench v1.5** 上超越所有开源模型，并优于大多数大规模通用 VLM（如 Qwen3-VL-235B、Gemini-3 Pro）；
- 在 **KIE 任务** 上大幅领先开源对手，在 Nanonets-KIE 超越 GPT-5.2（83.7 → 93.7）；
- 在 **真实工业场景测试** 中表现卓越：
  - **Seal Recognition**: 90.5（第二名为 dots.ocr 的 63.0）
  - **Receipt KIE**: 94.5（远超 GPT-5.2 的 83.5）
  - **Multilingual OCR**: 69.3（高于 PaddleOCR-VL-1.5 的 54.8）

### 推理效率对比（Table 6）
| 模型 | 图像输入 (pages/s) | PDF 输入 (pages/s) |
|------|--------------------|---------------------|
| **GLM-OCR** | **0.67** | **1.86** |
| PaddleOCR-VL-1.5 | 0.39 | 1.22 |
| MinerU2.5 | 0.18 | 0.48 |
| dots.ocr | 0.10 | — |

> GLM-OCR 实现约 **1.7x ~ 6.7x 的吞吐量提升**

### 消融实验（隐含于训练流程描述）
尽管未提供显式消融表，但从训练阶段设计可见关键组件作用：
- **MTP 引入后**（Stage 2.2 起）显著提升训练效率与推理速度；
- **强化学习阶段（RL）** 使用 GRPO 优化结构一致性，减少“断裂标签”和格式错误；
- **两阶段流水线** 显著降低幻觉风险，提升复杂文档稳定性。

---

## 4. 关键结论和发现

### 主要发现
1. **小模型也能实现 SOTA 性能**  
   尽管仅有 **0.9B 参数**，GLM-OCR 在多项任务上超越百亿甚至千亿参数模型，证明**架构设计与任务适配比单纯扩大规模更重要**。

2. **MTP 显著提升效率而不牺牲精度**  
   多 token 预测机制有效缓解自回归瓶颈，在保持准确率的同时实现 **~50% 吞吐提升**，特别适用于长结构输出（如表格、Markdown）。

3. **结构化先验 + 并行处理增强鲁棒性**  
   显式的布局分析模块（PP-DocLayout-V3）将复杂文档分解为独立区域，支持并行识别，极大提升了对复杂排版的适应能力。

4. **统一生成框架支持多样化应用**  
   通过不同 prompt 控制输出格式，既能完成端到端文档解析，也可用于轻量级 OCR 或 KIE，灵活适配从边缘设备到云端的不同场景。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **两阶段误差传播** | 若布局检测失败（如跨页表格），会影响后续识别效果 |
| **极端低质图像表现下降** | 对严重模糊、畸变、低分辨率文档识别能力受限 |
| **高度复杂数学公式支持有限** | 极复杂的嵌套表达式可能无法完全还原 |
| **结构化输出存在轻微随机性** | 由于生成本质，换行、空格等格式可能存在波动 |
| **KIE 依赖 prompt 清晰度** | 模糊或歧义字段边界可能导致提取不全或冗余 |

### 未来工作方向
1. 提升对**极端复杂布局**（如多栏交错、跨页表格）的理解能力；
2. 扩展**多语言覆盖范围**，尤其是小语种和混合语言文档；
3. 加强**结构化输出的一致性保障**，减少格式变异；
4. 优化**端到端联合建模能力**，探索免布局分割的统一模型；
5. 降低对人工标注 prompt 的依赖，发展更智能的自动 schema 推理机制。

---

> 🔗 **项目资源**  
> - GitHub: [github.com/zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR)  
> - Hugging Face: [huggingface.co/zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)  
> - Demo: [ocr.z.ai](https://ocr.z.ai/)  
> - 微调教程: [Finetune README](https://github.com/zai-org/GLM-OCR/blob/main/examples/finetune/README.md)

</details>

---

### 5. [Double-Precision Matrix Multiplication Emulation via Ozaki-II Scheme with FP8 Quantization](https://arxiv.org/abs/2603.10634)

**Authors**: Yuki Uchino, Katsuhisa Ozaki, Toshiyuki Imamura  
**Category**: cs.DC  
**Published**: 2026-03-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.10634v1  

#### Abstract
In high-performance computing (HPC) applications, FP64 arithmetic remains indispensable for ensuring numerical accuracy and stability. However, in recent hardware generations, improvements in FP64 arithmetic performance have been relatively modest. Consequently, achieving sustained performance gains...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Double-Precision Matrix Multiplication Emulation via Ozaki-II Scheme with FP8 Quantization*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在高性能计算（HPC）中，**FP64（双精度浮点）算术**对于数值稳定性和精度至关重要。然而，近年来硬件对 FP64 的性能提升有限，而低精度算术（如 INT8 和 FP8）则获得了显著加速。尽管如此，新兴架构（如 NVIDIA Blackwell Ultra 和 Rubin）大幅削减了 INT8 的计算资源，转而优先支持 FP8 等低精度浮点格式。

因此，一个关键问题是：  
> 如何在 **FP8 占优、INT8 资源受限**的现代 GPU 架构上，高效地实现 FP64 精度的通用矩阵乘法（DGEMM）？

现有的 **Ozaki-II 方案**虽然能通过 INT8 MMA 单元高效模拟 DGEMM，但由于其算法依赖于整数模运算和固定点语义，**无法直接适配 FP8 MMA 单元**。

---

### 🚀 提出的新方法与创新思路

本文提出了一种基于 **FP8_E4M3** 格式的 **Ozaki-II 方案**，首次实现了在 FP8 MMA 单元上的 Ozaki-II 风格 DGEMM 模拟。其核心技术是 **混合构造（Hybrid Construction）**，结合了两种关键技术：

#### （1）Karatsuba-based Extension
- 将输入矩阵 $ A $ 和 $ B $ 分解为多个 FP8_E4M3 子矩阵之和（例如 $ A = s \cdot A^{(1)} + A^{(2)} $），利用 Karatsuba 技巧减少乘法次数。
- 允许使用更大的模数 $ p_e \leq 513 $，从而提升 CRT 重建的动态范围，满足 FP64 精度要求。
- 每个模数需执行 **3 次 FP8 矩阵乘法**。

#### （2）Modular Reduction without Karatsuba（针对平方模数）
- 对于形如 $ p_e = s^2 $ 的模数，设计了一种无需 Karatsuba 重构的模约简方法。
- 利用 $ \text{mod}(s^2, p_e) = 0 $ 的性质，避免高开销的中间重建步骤。
- 显著减少了部分模数下的 FP8 矩阵乘法数量。

#### （3）Hybrid Method（核心创新）
- **优先选择平方模数**（如 1089, 1024, 961...），并对其应用无 Karatsuba 的模约简；
- 对非平方模数仍采用 Karatsuba 扩展。
- 最终仅需 **$ N \geq 12 $ 个模数**即可达到 FP64 精度，相比纯 Karatsuba 方法更高效。

---

### ⚖️ 相比现有方法的优势

| 方法 | 所需模数 $ N $ | FP8/INT8 矩阵乘法次数 | 精度（有效位） |
|------|------------------|------------------------|----------------|
| **FP8 Ozaki-I (S=11)** | — | 121 | ~54 bits |
| **INT8 Ozaki-II** | 14 | 14–15 | ~54 bits |
| **本文：FP8 Ozaki-II** | **12–13** | **36–40** | **~55–59 bits** |

- **显著优于 FP8 Ozaki-I**：乘法次数从 121 降至 36，效率提升超过 **3 倍**。
- **精度可比 INT8 Ozaki-II**，但乘法次数多约 2.5×，不过在 FP8 吞吐远高于 INT8 的架构上仍具竞争力。
- **首次实现 Ozaki-II 在 FP8 上的完整支持**，填补了算法空白。

---

## 2. 核心实验方法和设置

### 🧪 实验平台
- **NVIDIA GeForce RTX 5080**（CUDA 13.1.115）
- **HGX B200 单卡系统**（CUDA 12.8.93）
- 使用 `cuBLAS` 和 `cuBLASLt` 进行底层 INT8/FP8 矩阵乘法调度。

### 🔢 数据集与测试矩阵生成
- 未使用真实数据集，而是合成随机矩阵：
  $$
  A_{ij}, B_{ij} \sim (\text{rand} - 0.5) \cdot \exp(\text{randn} \cdot \phi)
  $$
  其中 `rand` ∈ (0,1] 均匀分布，`randn` 为标准正态，$\phi$ 控制动态范围。
- 测试规模覆盖 $ m, n, k \in \{1024, 2048, ..., 32768\} $，典型方阵设置（$ m=n $）为主。

### 📊 评估指标
1. **吞吐量（Throughput）**：TFLOP/s，衡量 DGEMM 模拟的实际性能。
2. **精度（Accuracy）**：相对误差 $\|C_{\text{emulated}} - C_{\text{FP64}}\|_F / \|C_{\text{FP64}}\|_F$。
3. **工作内存占用（Working Memory Footprint）**：临时缓冲区总大小（不含输入输出）。
4. **时间分解（Time Breakdown）**：量化 `quant`, `gemms`, `requant`, `dequant`, `others` 各阶段耗时占比。

### 🔁 基线方法对比
- **Native FP64 DGEMM**：cuBLAS `cublasDgemm`
- **INT8-based Ozaki-II**：文献 [19], [22] 实现
- **FP8-based Ozaki-I**：文献 [21] 方法（作为 FP8 对标）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）吞吐量表现（RTX 5080）
| 方法 | $ m=n=8192 $ 时加速比（vs FP64） |
|------|-------------------------------|
| INT8 Ozaki-II (fast) | **7.1–24×** |
| FP8 Ozaki-II (fast) | **4.4–9.4×** |
| **INT8 快于 FP8** | **1.3–2.9×** |

#### （2）吞吐量表现（B200）
| 方法 | $ m=n=16384 $ 时绝对性能 |
|------|--------------------------|
| INT8 Ozaki-II (fast) | **125 TFLOP/s** |
| FP8 Ozaki-II (fast) | **61 TFLOP/s** |
| Native FP64 DGEMM | ~35–40 TFLOP/s（理论峰值 75）|

> 尽管 FP8 方法较慢，但在大尺寸下仍 **超越原生 FP64**。

#### （3）内存占用（$ m=n=k=16384 $）
| 方法 | 工作内存（Workspace） |
|------|---------------------|
| INT8 Ozaki-II ($ N=14 $) | **27 GB** |
| FP8 Ozaki-II ($ N=12 $) | **55 GB** |

> FP8 方法因需存储多个子矩阵和 INT16 中间结果，内存开销翻倍。

---

### 🔍 与基线方法对比

| 维度 | 结果 |
|------|------|
| **vs FP8 Ozaki-I** | 乘法次数从 121 → 36，**效率大幅提升**；精度相当甚至更高 |
| **vs INT8 Ozaki-II** | 乘法次数更多（36 vs 14），但适用于 INT8 受限架构；在 FP8 吞吐足够高时可能反超 |
| **跨平台适应性** | 在 B200/B300/Rubin 等 **FP8 强、INT8 弱**的架构上更具潜力 |

---

### 🔬 消融实验与分析（隐含在设计中）

- **Hybrid vs 纯 Karatsuba**：
  - 若不使用 modular reduction for square moduli，则需至少 13 个模数且全走 Karatsuba 路径，导致 **39 次乘法**；
  - 引入 hybrid 后可降至 **36 次**，验证了优化有效性。
- **Fast Mode vs Accurate Mode**：
  - Accurate mode 使用实际 FP8 矩阵乘估计缩放边界，精度更高；
  - Fast mode 使用 Cauchy-Schwarz 不等式保守估计，可能导致过度缩放，降低精度。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **FP8 可用于 Ozaki-II 架构**：首次成功将 Ozaki-II 方案迁移到 FP8 MMA 单元，突破原有仅支持 INT8 的限制。
2. **Hybrid 方法显著提效**：结合 Karatsuba 与模约简技巧，将 FP8 矩阵乘法次数控制在合理范围内（36–40 次）。
3. **INT8 仍是当前最优**：在 INT8 资源充足平台（如 RTX 5080、B200），INT8 Ozaki-II 在 **性能和内存**上全面领先。
4. **FP8 方案面向未来架构**：在 **INT8 被削弱、FP8 成主导**的下一代 GPU（如 Rubin）上，本文方法将成为关键替代方案。
5. **性能模型准确**：提出的解析性能模型预测值与实测高度吻合，可用于指导部署决策。

---

### ⚠️ 方法的局限性

1. **更高的内存开销**：FP8 格式中的 exponent 字段在固定点运算中浪费，导致存储效率低于 INT8。
2. **更多低精度乘法**：相比 INT8 的 $ N $ 次乘法，FP8 需 $ 3N $ 次，对 compute-bound 场景不利。
3. **依赖 FP32 accumulate**：假设 FP8 MMA 输出累积到 FP32 无舍入误差，限制了 $ k $ 的最大长度（$ k \leq 2^{16} $）。
4. **尚未在 Rubin 上实测**：虽有预测，但缺乏在真正 INT8 极弱平台上的验证。

---

### 🔮 未来工作方向

1. **探索 FP4 支持**：若未来硬件提供极高 FP4 吞吐（>3×FP8），可通过递归 Karatsuba 构建更高效模拟。
2. **优化内存占用**：研究压缩表示或 in-place 计算策略，减少 workspace 开销。
3. **扩展至其他算子**：将该框架应用于 GEMV、卷积或其他 BLAS/LAPACK 核心算子。
4. **支持稀疏矩阵**：结合稀疏化技术进一步提升能效比。
5. **开源库持续维护**：项目已开源（GitHub: [RIKEN-RCCS/GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8)），将持续优化跨平台兼容性与性能。

---

> 💡 **一句话总结**：  
> 本文提出了首个可在 **FP8 MMA 单元上运行的 Ozaki-II DGEMM 模拟方法**，通过 **混合 Karatsuba 与模约简技术**，在保持 FP64 精度的同时显著降低计算开销，为 **INT8 资源受限的下一代 GPU 架构**提供了可行的高性能双精度计算路径。

</details>

---

### 6. [FRIEND: Federated Learning for Joint Optimization of multi-RIS Configuration and Eavesdropper Intelligent Detection in B5G Networks](https://arxiv.org/abs/2603.10977)

**Authors**: Maria Lamprini A. Bartsioka, Ioannis A. Bartsiokas, Anastasios K. Papazafeiropoulos, Maria A. Seimeni, Dimitra I. Kaklamani, Iakovos S. Venieris  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.10977v1  

#### Abstract
As wireless systems evolve toward Beyond 5G (B5G), the adoption of cell-free (CF) millimeter-wave (mmWave) architectures combined with Reconfigurable Intelligent Surfaces (RIS) is emerging as a key enabler for ultra-reliable, high-capacity, scalable, and secure Industrial Internet of Things (IIoT) c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# FRIEND: Federated Learning for Joint Optimization of multi-RIS Configuration and Eavesdropper Intelligent Detection in B5G Networks  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Beyond 5G (B5G)** 网络中，随着 **cell-free mmWave 架构** 和 **Reconfigurable Intelligent Surfaces (RIS)** 的广泛应用，如何在分布式、高动态的工业物联网（IIoT）环境中实现高效且隐私保护的 **物理层安全（Physical Layer Security, PLS）** 成为关键挑战。传统加密机制难以应对大规模部署下的 **可扩展性、延迟和隐私泄露风险**。

本文聚焦于：
- 如何在不集中原始数据的前提下检测恶意用户（eavesdroppers）
- 如何联合优化多 RIS 配置以增强合法链路并抑制窃听
- 在保障 **privacy-preserving** 的同时提升系统 **secrecy rate (SR)** 和检测精度

---

### 🚀 提出的新方法与创新思路

1. **FRIEND 框架**：提出一种基于 **Federated Learning (FL)** 的智能框架，用于在 **RIS-assisted cell-free mmWave 网络** 中进行 **联合的多 RIS 配置优化与窃听者检测**。
   - 利用边缘设备（APs）本地采集的 **Channel State Information (CSI)** 数据训练深度模型
   - 所有客户端通过 **FedAvg** 协议聚合模型参数，无需上传原始数据，实现 **隐私保护**

2. **Deep Convolutional Neural Network (DCNN) + Early-Exit 机制**
   - 设计轻量级 DCNN 结构处理 CSI 图像，并融合 UE 位置、发射功率等元数据
   - 引入 **early-exit 机制**：当中间输出置信度超过阈值时提前终止推理，显著降低计算开销和延迟

3. **多 RIS 联合调控与安全增益协同设计**
   - 将 RIS 相位配置作为可调参数，在提升主链路 SINR 的同时恶化窃听链路质量
   - 实现 **beamforming 与安全检测的联合优化**

---

### 🔍 相比现有方法的优势

| 对比维度 | 本工作 | 现有方法 |
|--------|-------|--------|
| **隐私保护** | 完全去中心化训练，无原始数据交换（FL） | 多依赖集中式 ML，存在隐私泄露风险 |
| **网络架构适配性** | 支持 cell-free、multi-RIS、mmWave 等 B5G 特征 | 多针对单小区或单一 RIS 场景 |
| **实时性与资源效率** | 引入 early-exit，减少 35–45% 推理时间 | 模型固定复杂度，难适应边缘资源受限场景 |
| **安全性增益** | 联合优化 RIS 配置与检测策略，提升 SR | 多数仅关注检测或波束成形之一 |

> ✅ **核心优势总结**：首次将 **FL + multi-RIS + DCNN + early-exit** 融合应用于 **cell-free mmWave IIoT 安全检测**，实现了 **隐私性、可扩展性、低延迟与高安全增益** 的统一。

---

## 2. 核心实验方法和设置

### 📊 数据集来源与生成方式
- **非公开真实数据集**，而是基于 **MATLAB R2025b** 自建仿真环境生成
- 模拟符合 **3GPP TR 38.901 / TR 38.843** 标准的 B5G 工业场景
- 包含以下要素：
  - 分布式 APs（18个）、UEs（500个，70% 合法 + 30% 窃听者）
  - 多 RIS 单元（3个，尺寸 10×20）
  - mmWave 频段（28 GHz），OFDM 波形，SRS 参考信号
  - 多径衰落、遮挡、大/小尺度衰落、噪声建模
- 输出为 **CSI 图像 + 元数据（位置、功率、服务节点）**，标签为 `0`（合法）或 `1`（窃听）

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| 网络拓扑 | Cell-Free mMIMO + multi-RIS |
| 频率 | 28 GHz (FR2) |
| 带宽 | 400 MHz |
| 子载波间隔 | 120 kHz |
| AP 天线数 | 32 |
| UE 天线数 | 1 |
| RIS 相位维度 | 10×20 |
| FL 客户端数量 | 3（选自中心区域 AP） |
| 训练/测试划分 | 80%/20% |
| 模型框架 | TensorFlow Federated (TFF) |
| 优化器 | Adam |
| 损失函数 | Binary Crossentropy |
| Early-exit Confidence Level | 55%, 70% |

---

### 📈 评估指标

| 类别 | 指标 |
|------|------|
| **检测性能** | Accuracy, Precision, Recall, F1-score |
| **通信安全** | Secrecy Rate (SR), Average Secrecy Rate (ASR) |
| **系统效率** | Inference Time, Early-exit Rate, Model Complexity |
| **鲁棒性** | 多 Monte Carlo 仿真（100次不同 RIS 相位）下的稳定性 |

---

### 🆚 基线方法对比

| 基线方法 | 描述 |
|--------|------|
| **Centralized ML (from [22])** | 使用相同 CSI 数据的集中式 DCNN 模型（非联邦） |
| **Non-RIS-assisted 方法** | 不启用 RIS 的传统 cell-free mmWave 安全检测方案 |
| **Non-ML Approach** | 基于真实标签计算的理想 SR 上限（用于验证分类有效性） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）检测性能（Figure 3）

| 指标 | 范围 | 中位数/典型值 |
|------|-----|-------------|
| **Accuracy** | 0.71 – 0.93 | ~0.84 |
| **Precision** | 高且稳定 | >0.9 |
| **Recall** | 0.75 – 0.95 | ~0.88 |
| **F1-score** | 稳健分布 | >0.85 |

> 🔹 高 recall 表明模型能有效识别大多数窃听者，避免漏检（critical for PLS）  
> 🔹 F1-score 稳定说明未出现严重的 precision-recall 失衡

---

#### （2）early-exit 效果（Figure 4）

| Confidence Threshold | 推理时间下降 | 准确率保持 | Early-exit Rate |
|----------------------|--------------|------------|----------------|
| 70%                  | ↓35%         | >0.83      | 显著比例样本提前退出 |
| 55%                  | ↓45%         | >0.83      | 更高退出率，适合低延迟场景 |

> ✅ 证明 early-exit 可在几乎不影响准确率的情况下大幅降低延迟，适用于资源受限的边缘 AP

---

#### （3）通信安全性能（Figure 5）

| RIS Phase ID | ASR 最大值（bps/Hz） | 相比 non-RIS 提升 |
|------------|------------------|------------------|
| **Best (ID 89)** | >20              | —                |
| **Proposed (ID 4)** | ~18             | **↑30% vs [22]** |

> 🔹 RIS Phase ID 89 表现最优，表明其反射相位能构造对合法用户有利而对窃听者破坏性的信道  
> 🔹 ID 54 表现差，说明不当 RIS 配置可能反而增强窃听链路

> ✅ **关键发现**：最佳 RIS 配置下，**ASR 提升约 30%**（相比 [22] 中非 RIS 辅助的最佳方法）

---

#### （4）与 Non-ML 方法对比验证

- 所提 FL 模型的 SR 曲线与 “Non-ML Approach”（理想情况）高度吻合（如 RIS Phase ID 4）
- 表明模型学习到了真实的物理层模式，而非过拟合或偏差驱动

---

### 🔍 消融实验分析（隐含在文中）

虽然未明确列出消融表，但从实验设计可推断以下结论：

| 组件 | 是否移除影响性能？ | 影响程度 |
|------|--------------------|---------|
| **RIS 协同优化** | 是 | 移除后 ASR 下降 ~30% |
| **Federated Learning** | 是 | 若集中训练，隐私受损，不符合 IIoT 部署要求 |
| **Early-exit 机制** | 是 | 推理时间增加 35–45%，影响实时性 |
| **Metadata 融合（位置+功率）** | 是 | 文献 [22] 显示缺失该信息会导致 F1 下降 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **FL 可有效支持分布式窃听检测**  
   在不共享原始 CSI 数据的前提下，多个 AP 可协作训练高性能 DCNN 模型，实现接近集中式学习的检测精度（Accuracy ~0.84, F1 > 0.85）。

2. **multi-RIS 显著提升物理层安全**  
   合理配置 RIS 相位可在增强合法链路的同时干扰窃听者，使 **ASR 提升达 30%**，最高可达 **18–20 bps/Hz**。

3. **early-exit 机制平衡性能与效率**  
   在保证分类质量的同时，**降低 35–45% 推理延迟**，特别适合边缘侧部署。

4. **系统工程导向的设计有效**  
   遵循 MBSE（Model-Based Systems Engineering）流程定义需求（如 R1–R10），确保非功能性需求（privacy, scalability, maintainability）从早期嵌入设计。

---

### ⚠️ 局限性

| 局限 | 说明 |
|------|------|
| **依赖仿真数据** | 当前实验完全基于 MATLAB 生成数据，尚未在真实硬件平台验证（R8 未完成） |
| **RIS 配置空间有限** | 仅测试固定几种相位组合，未探索全自动在线优化（如结合 DRL） |
| **客户端数量少** | 仅使用 3 个 FL 客户端，大规模网络下的收敛性和通信开销需进一步研究 |
| **未考虑敌对攻击下的鲁棒性** | 如模型中毒、梯度泄露等 FL 特有威胁未被防御 |

---

### 🔮 未来工作方向（作者明确提出）

1. **引入真实数据验证（fulfill R8）**  
   将模型迁移到实测 CSI 数据集上，验证泛化能力。

2. **扩展至可扩展 RIS 架构（address R7）**  
   研究更大规模、异构部署的 RIS 配置优化问题。

3. **探索 early-exit 与任务卸载结合**  
   在边缘服务器与终端之间动态决策是否本地推理或卸载。

4. **集成更先进 FL 技术**  
   如 differential privacy、secure aggregation，进一步强化隐私保护。

5. **联合优化 RIS 控制与检测策略**  
   使用 DRL 或 bilevel optimization 实现端到端联合优化。

---

## 总结

| 维度 | 内容 |
|------|------|
| **核心思想** | 联合利用 **Federated Learning** 与 **multi-RIS** 实现隐私保护的物理层窃听检测 |
| **关键技术** | DCNN + early-exit + FedAvg + CSI 图像化 |
| **最大亮点** | 在保障隐私与低延迟的同时，**ASR 提升 30%** |
| **适用场景** | 工业 IoT、B5G/6G 密集部署、高安全需求无线网络 |
| **未来潜力** | 可拓展至 6G 智能超表面、空天地一体化网络中的安全感知体系 |

> 💡 **一句话总结**：  
> FRIEND 框架成功实现了 **“安全、智能、隐私、高效” 四维统一**，为下一代分布式无线系统的物理层安全保障提供了可行路径。

</details>

---

### 7. [AgentServe: Algorithm-System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU](https://arxiv.org/abs/2603.10342)

**Authors**: Yuning Zhang, Yan Yan, Nan Yang, Dong Yuan  
**Category**: cs.DC  
**Published**: 2026-03-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.10342v1  

#### Abstract
Large language models (LLMs) are increasingly deployed as AI agents that operate in short reasoning-action loops, interleaving model computation with external calls. Unlike traditional chat applications, these agentic workloads require inference serving systems to balance low latency, stable token e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AgentServe: Algorithm-System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 LLM 推理服务系统（如 vLLM、SGLang）主要针对**长文本生成型聊天机器人**进行优化，强调吞吐量（throughput）。然而，**AI agent 工作负载**具有显著不同的特征：
- **冷预填充（Cold Prefill）**：初始处理长系统提示（如工具描述、ReAct prompt），计算密集。
- **恢复预填充（Resume Prefill）**：将工具输出追加到缓存上下文中，频繁发生。
- **短解码（Short Decode）**：生成函数调用或路由标记，延迟敏感。

在单个消费级 GPU 上并发运行多个 agent 时，长 prefill 会阻塞短 decode，导致 **Head-of-Line (HoL) Blocking**，破坏交互流畅性，表现为 **TTFT（Time-to-First-Token）上升** 和 **TPOT（Time-per-Output-Token）不稳定**。

---

### ✅ 提出的新方法与创新点

AgentServe 是一种专为 **单 GPU 上多 agent 并发推理** 设计的算法-系统协同设计框架，核心创新如下：

#### （1）**Phase-Aware 请求分类与调度**
- 明确区分三种执行阶段：
  - **Cold Prefill**：长且无缓存的初始输入 → 单独隔离处理。
  - **Resume Prefill**：短小的上下文扩展 → 动态控制其长度预算 `B_prefill(t)`。
  - **Short Decode**：极短但延迟敏感 → 赋予最高优先级。
- 引入 **TPOT-driven 反馈调度器**：根据实时观测的 TPOT 动态调整：
  - 解码保留的 SM 数量 `R_min(t)`
  - Resume Prefill 的最大允许长度 `B_prefill(t)`

#### （2）**基于 CUDA Green Context 的轻量级资源隔离**
- 利用 **CUDA Green Context** 实现细粒度 SM 分区，无需多引擎或进程分离。
- 预创建 10 个不同 SM 分配比例（10%~100%）的 Green Context，运行时通过“重绑定”实现快速切换（<50μs），避免动态创建开销。
- 实现 **单引擎内严格的 prefill-decode 资源隔离**，防止 decode 被大 kernel 饥饿。

#### （3）**共享内存管理 + 安全 KV 缓存复用**
- 所有线程共享同一 GPU 内存池，避免跨进程 KV 传输。
- 使用 CPU mutex + GPU `cudaEvent` 同步机制，确保 decode 不读取未完成写入的 KV 缓存。

> 💡 **相比现有方法的优势**：
> - 相比 **PD disaggregation（如 SGLang）**：避免多进程通信开销，在单 GPU 上更高效。
> - 相比 **chunked prefill（如 vLLM）**：不依赖长 decode 来吸收 chunk 开销，在 agent 的短 decode 场景下更稳定。
> - 相比 **静态分区或 MPS**：提供动态自适应能力，响应突发负载变化。

---

## 2. 核心实验方法和设置

### 📚 数据集与工作负载
- **数据来源**：基于 [ToolBench](https://arxiv.org/abs/2404.11143) 构建 agent 工作流。
- **两种典型 agent 范式**：
  - **ReAct**：频繁 resume prefill + 极短 decode（函数调用）
  - **Plan-and-Execute**：长 cold prefill + 中等 decode
- **Token 分布统计**（见 Table I）：
  - Cold Prefill: ~2.5k–3.5k tokens（固定系统提示）
  - Resume Prefill: ReAct: 30–127 tokens；Plan-and-Execute: 125–421 tokens
  - Decode: 多数 <100 tokens，高度稳定

### ⚙️ 实验设置
- **硬件平台**：
  - RTX A5000（64 SMs, 24GB GDDR6）
  - RTX 5090（128 SMs, 32GB GDDR7）——下一代高端消费卡
- **模型**：
  - Qwen2.5-3B / Qwen2.5-7B
  - LLaMA-3-8B
- **并发 agent 数量**：3–6 个并发会话
- **实现基础**：基于 `llama.cpp` 扩展，支持 CUDA Green Context

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **TTFT** | Time-to-First-Token，反映首响应延迟 |
| **TPOT** | Time-per-Output-Token，衡量流式输出平滑度（p50 和 p95） |
| **Throughput** | 总输出 token/s，反映整体效率 |
| **SLO Attainment Rate** | 同时满足 TTFT 和 TPOT 阈值的会话占比 |

### 🔁 基线方法对比
| 基线 | 特点 |
|------|------|
| **SGLang** | 支持 PD disaggregation 的单 GPU 引擎 |
| **vLLM** | 使用 PagedAttention + chunked prefill 的吞吐导向系统 |
| **llama.cpp** | 轻量级通用推理框架，无特殊调度优化 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能提升（vs. SOTA 基线）

| 指标 | 提升幅度 | 说明 |
|------|----------|------|
| **TTFT** | 最高 **2.8× 更快** | 尤其在高并发、大模型（7B/8B）下优势明显 |
| **TPOT (p95)** | 最高 **2.7× 更优** | 显著降低尾部延迟，保证流式稳定性 |
| **Throughput** | 提升 **1.2–2.2×** | 在保障低延迟的同时维持高吞吐 |
| **SLO Attainment Rate** | 接近 **100%（5090）**，远超基线 | 表明用户体验一致性更强 |

> ✅ 在 **RTX A5000** 上，由于资源受限，prefill-decode 干扰更严重，AgentServe 的相对收益更大。

---

### 🔍 消融实验（Ablation Study）

在 N=4 并发 agent 下测试消融版本：

| 变体 | 描述 | 影响 |
|------|------|------|
| **No-Alg** | 移除 TPOT-driven 调度，采用静态 SM 分配 |  
  - TTFT ↑ 15–25%  
  - TPOT p95 ↑ 1.4×  
  → 缺乏动态调节导致 decode 饥饿 |
| **No-Green** | 移除 Green Context，使用普通 stream |  
  - TTFT ↑ 20–30%  
  - TPOT 方差显著增加  
  → 无法保证 decode 资源预留 |

> ✅ 结果证明：**算法调度 + 系统级隔离** 共同构成性能提升的关键，缺一不可。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Agent 工作负载本质不同于传统 chatbot**：
   - Prefill 占主导成本，Decode 占主导感知延迟。
   - 必须显式区分 cold / resume prefill，并保护 short decode。

2. **Fine-grained PD disaggregation 在单 GPU 上可行且必要**：
   - 通过 **CUDA Green Context + 动态 rebinding** 可实现低开销、高精度资源隔离。

3. **TPOT-driven 反馈控制有效平衡延迟与吞吐**：
   - 动态调节 `R_min(t)` 和 `B_prefill(t)` 能在 decode SLO 约束下最大化 prefill 吞吐。
   - 理论分析给出 **competitive ratio bound**，量化性能损失上界。

4. **AgentServe 泛化性强**：
   - 在不同模型（Qwen / LLaMA）、不同硬件（A5000 / 5090）上均保持稳定优势。

---

### ⚠️ 局限性
- 当前设计聚焦于 **消费级单 GPU 场景**，未考虑多 GPU 或分布式部署。
- 对 **极端长上下文（>32k）** 的 KV cache 管理未深入优化。
- 假设工具返回内容较短，若 resume prefill 过长仍可能影响性能。

---

### 🔮 未来工作方向
- 扩展至 **multi-GPU AgentServe**，支持更大规模本地 agent 集群。
- 结合 **KV cache compression**（如 CacheGen）进一步节省显存。
- 支持 **异构 agent 类型混合调度**（如 ReAct + Plan-and-Execute 共存）。
- 探索 **prefill 预取与 speculative execution** 加速 cold start。

---

> **总结一句话**：  
> AgentServe 通过 **算法-系统协同设计**，首次实现了在 **单个消费级 GPU 上稳定高效地服务多 AI agent**，解决了传统系统在 **prefill-decode 干扰下的延迟抖动问题**，为边缘端 agentic AI 部署提供了实用解决方案。

</details>

---

### 8. [Leech Lattice Vector Quantization for Efficient LLM Compression](https://arxiv.org/abs/2603.11021)

**Authors**: Tycho F. A. van der Ouderaa, Mart van Baalen, Paul Whatmough, Markus Nagel  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.11021v1  

#### Abstract
Scalar quantization of large language models (LLMs) is fundamentally limited by information-theoretic bounds. While vector quantization (VQ) overcomes these limits by encoding blocks of parameters jointly, practical implementations must avoid the need for expensive lookup mechanisms or other explici...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Leech Lattice Vector Quantization for Efficient LLM Compression*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **scalar quantization**（标量量化）在压缩大语言模型（LLMs）时存在信息论上的根本限制：其逐元素编码方式无法达到最优的率失真（rate-distortion）性能。尽管 **vector quantization (VQ)** 在理论上能突破这一限制，但实际应用中常因需要存储庞大的显式码本（codebook）和进行昂贵的最近邻搜索而难以扩展。

本文旨在解决以下挑战：
- 如何实现高维、高效且无需显式码本的 VQ；
- 如何在保持低失真的同时支持灵活的比特率配置；
- 如何将数学上最优的高维格（lattice）结构应用于实际 LLM 压缩任务。

---

### 提出的新方法：**Leech Lattice Vector Quantization (LLVQ)**

作者提出了一种基于 **Leech lattice**（Leech 格）的新型向量量化框架 —— **LLVQ**，其核心是利用 Leech lattice 在 24 维空间中的最优球体堆积性质来构建高效的量化码本。

#### 创新点：
1. **首次将 Leech lattice 应用于 LLM 量化**  
   Leech lattice 是唯一已知在 24 维中实现最优球体堆积和“接吻数”（kissing number）最大化的格结构，具有极高的对称性和密度，适合构造低失真的高维量化器。

2. **扩展 Adoul & Barth 算法以支持索引化（indexing）**  
   扩展了经典的无码本最近邻搜索算法，使其能够：
   - 将每个量化向量映射为唯一的整数索引（bitstring），无需存储码本；
   - 支持从索引快速重建原始向量（dequantization）。

3. **支持多壳层联合搜索（Union of Shells）的角搜索（Angular Search）**  
   允许在多个 Leech lattice 的壳层（shells）上进行联合搜索，从而实现更密集的球面码（spherical code），提升形状-增益（shape-gain）量化性能。

4. **提出完全并行化的去量化核（Fully-Parallelizable Dequantization Kernel）**  
   利用快速模运算和组合结构，在 GPU 上实现高效批量解码，适用于大规模部署。

---

### 相比现有方法的优势
| 方法 | 维度 | 是否需码本 | 搜索效率 | 理论基础 | 比特灵活性 |
|------|-------|-------------|------------|-----------|--------------|
| Scalar Quantization | 1D | 否 | 高 | 弱（次优） | 高 |
| GPTVQ / Unstructured VQ | 高维 | 是 | 低（指数级复杂度） | 中等 | 有限 |
| Quip# (E8 lattice) | 8D | 否 | 高 | 强（E8 最优） | 低（依赖 RVQ） |
| QTIP / PVQ | 可变 | 否 | 中等 | 中等 | 中等 |
| **LLVQ (Leech)** | **24D** | **否** | **高** | **最强（Leech 最优）** | **高（原生支持多种比特率）** |

- **理论优势更强**：Leech lattice 在 24D 的几何最优性远超 E8（8D）。
- **无需残差量化（RVQ）即可支持高比特率**：相比 Quip# 等依赖 RVQ 提升比特率的方法，LLVQ 天然支持宽范围比特配置。
- **更高的信息保留率**：在相同比特下，接近香农极限的程度更高。

---

## 2. 核心实验方法和设置

### 数据集
- **理想源测试**：采样自标准正态分布 $ \mathcal{N}(0, I) $ 的独立同分布（i.i.d.）向量，用于评估量化器在高斯源下的理论性能。
- **真实 LLM 权重压缩**：
  - 模型家族：**Llama-2**, **Llama-3**, **Ministral-3**, **Qwen-v3**（涵盖 4B–8B 参数规模）
  - 校准数据集：**DCLM-edu**（共 6,100 条序列），用于估计每层的 Hessian 矩阵以进行误差校正。

---

### 实验设置
- **量化粒度**：按权重矩阵的每一行划分为长度为 24 的块（block），不足则补零。
- **量化模式**：
  - **Post-Training Quantization (PTQ)**：不进行微调（no fine-tuning）
  - **轻量微调（Lightweight Fine-tuning）**：仅更新输入尺度参数（input scales），开销 < 0.001 bits/weight
- **预处理技术**：
  - 使用 **Hadamard rotations** 对权重进行旋转，使其统计特性更接近高斯分布，便于量化。

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **Perplexity (WikiText-2 @ 4096)** | 衡量语言建模能力，越低越好 |
| **MMLU** | 多任务理解能力，越高越好 |
| **CSR** | Common Sense Reasoning 性能，越高越好 |
| **SQNR (Signal-to-Noise Ratio in bits)** | 定义为 $-\log_2(\text{MSE})$，衡量量化保真度 |
| **Retention (%)** | 相对于香农极限的信息保留比例：$\frac{\text{SQNR}_{\text{bits}}}{R} \times 100\%$，其中 $R$ 为比特率 |

---

### 基线方法对比
- **Scalar Baselines**：
  - RTN（Random Tensor Noise）
  - GPTQ
  - Quarot（带 Hadamard 旋转）
- **Structured VQ Methods**：
  - Quip# / E8P（基于 E8 lattice）
  - QTIP（trellis-based）
  - PVQ（pyramid vector quantization）
- **其他先进方法**：
  - AQLM
  - OmniQ
  - PV-tuning

所有方法在统一 pipeline 下进行公平比较（除非特别注明引用文献结果）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Gaussian Source @ 2 bits/dim）

| 方法 | MSE ↓ | SQNR (bits) ↑ | Retention (%) ↑ |
|------|--------|----------------|------------------|
| Uniform | 0.150 | 1.37 | 69% |
| E8P / Quip# | 0.092 | 1.72 | 86.1% |
| **LLVQ (spherical shaping)** | **0.084** | **1.79** | **89.4%** |
| **LLVQ (shape-gain)** | **0.078** | **1.84** | **92.1%** |
| **Theoretical Limit (Shannon)** | 0.0625 | 2.00 | 100% |

> ✅ **LLVQ 达到当前最高信息保留率（92.1%），最接近香农极限**

---

### LLM 压缩性能（PTQ, 2 bits/weight）

#### 在 Llama-2 7B 上的表现（Wikitext-2 PPL）

| 方法 | PPL ↓ | MMLU ↑ | CSR ↑ |
|------|--------|--------|--------|
| Baseline (FP16) | 5.11 | 45.7 | 70.4 |
| GPTQ+Quarot | 41.87 | 27.0 | 41.7 |
| Quip# | 7.96 | 30.5 | 61.4 |
| **LLVQ (spherical)** | **7.61** | **33.4** | **62.1** |
| **LLVQ (shape-gain)** | **6.83** | **34.9** | **64.6** |

> ✅ **LLVQ 显著优于 Quip# 和其他 VQ 方法，甚至优于部分微调后的基线**

---

#### 跨模型一致性表现（Table 3）

在 Llama-3 8B、Ministral-3 8B、Qwen-3 4B/8B 上均观察到一致趋势：
- LLVQ 在 **所有模型架构和下游任务上全面领先**
- shape-gain 版本始终优于 spherical shaping
- 即使不微调，LLVQ 性能也超过许多微调后方法

---

### 消融实验结果

#### （1）是否使用 Hadamard 旋转的影响（Table 6）

| 方法 | Rotation | PPL (Llama-2) | MMLU | CSR |
|------|----------|----------------|--------|-----|
| LLVQ (shape-gain) | None | 7.27 | 29.8 | 61.5 |
| LLVQ (shape-gain) | Input Only | 6.90 | 36.0 | 63.6 |
| LLVQ (shape-gain) | Input+Output | **6.83** | **34.9** | **64.6** |

> 🔍 发现：虽然旋转有助于性能提升，但 **LLVQ 本身对旋转的依赖显著低于标量方法**，说明高维 VQ 内在具备更强鲁棒性。

#### （2）shape-gain vs. spherical shaping（Appendix F）

| 方法 | Bits/dim | MSE | SQNR | Retention |
|------|----------|------|--------|-----------|
| Spherical Shaping | 2.0 | 0.084 | 1.787 | 89.37% |
| Shape-Gain (1 gain bit) | 2.0 | **0.078** | **1.843** | **92.14%** |

> ✅ **shape-gain 构造更优**，尤其当分配约 1 gain bit（≈0.041 bits/dim）时达到最佳性能，接近理论建议值 $1/n = 1/24 ≈ 0.042$

#### （3）单壳 vs. 多壳联合构造球面码（Appendix E）

| 方法 | Angular Separation ↓ | Uniformity ↑ |
|------|------------------------|---------------|
| Single Shell (m) | 较差 | 一般 |
| Union of Shells (≤m) | **更好** | **更均匀** |

> ✅ 使用多个壳层的联合可生成更均匀的球面码，提升角分辨率，推荐作为默认策略。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **高维 lattice 结构（如 Leech lattice）是实现高效 LLM 压缩的关键路径**  
   几何最优性直接转化为更低的量化失真和更高的信息保留率。

2. ✅ **LLVQ 是首个实用化的 Leech lattice VQ 框架**  
   通过索引化、并行化解码和多壳搜索，克服了传统 lattice VQ 的工程障碍。

3. ✅ **shape-gain + Leech lattice 实现当前最优率失真性能**  
   在 2 bits/dim 下达到 **92.1% 香农极限保留率**，超越 Quip#, QTIP, PVQ 等前沿方法。

4. ✅ **LLVQ 减少对 Hadamard 旋转等预处理的依赖**  
   高维结构本身提供了良好的统计适配性，降低部署延迟风险。

5. ✅ **无需 RVQ 即可灵活支持多种比特率**  
   天然支持从 ~0.75 到 >2.3 bits/dim 的连续比特配置（见 Table 1），适应不同场景需求。

---

### 方法的局限性
- ❌ **仅适用于 24 的倍数维度**：非整除时需 padding，可能引入轻微冗余。
- ❌ **实现复杂度较高**：依赖 Golay code 和组合结构，理解门槛高于普通 scalar quantization。
- ❌ **目前主要用于 weight-only 量化**：尚未集成 KV cache 或 activation quantization。

---

### 未来工作方向
- 🔄 探索 **非 24 维的适配机制**（如分组 + 投影）
- 🔀 将 LLVQ 扩展至 **activation 和 KV cache 量化**
- 🧠 结合 **learnable lattice scaling / offset** 进一步优化适配能力
- 📈 探索 **更高维度 lattice 或非格结构的球面码**
- 💡 将 LLVQ 与其他压缩技术（pruning, sparsity, LoRA）结合，打造端到端超低比特系统

---

> **总结一句话**：  
> **LLVQ 通过将数学上最优的 Leech lattice 引入 LLM 压缩，实现了无需显式码本、高并行、高保真的向量量化，在理论与实践层面均达到了当前 SOTA 水平，标志着高维 lattice 方法在神经网络压缩中的重大突破。**

</details>

---

### 9. [Estimating condition number with Graph Neural Networks](https://arxiv.org/abs/2603.10277)

**Authors**: Erin Carson, Xinye Chen  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.10277v1  

#### Abstract
In this paper, we propose a fast method for estimating the condition number of sparse matrices using graph neural networks (GNNs). To enable efficient training and inference of GNNs, our proposed feature engineering for GNNs achieves $\mathrm{O}(\mathrm{nnz} + n)$, where $\mathrm{nnz}$ is the number...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Estimating condition number with Graph Neural Networks

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**大规模稀疏矩阵的 condition number（条件数）估计效率低下**的问题。传统方法如 Hager-Higham 算法（用于 1-norm）和 Lanczos 方法（用于 2-norm）在处理大型稀疏系统时计算成本高昂，尤其是当需要频繁调用 condition number 进行稳定性分析或精度自适应调整时。

### 提出的新方法与创新思路
作者提出了一种基于 **Graph Neural Networks (GNN)** 的数据驱动方法来快速估计稀疏矩阵的 condition number，其核心创新包括：

- **首次将图神经网络应用于 condition number 估计任务**：将稀疏矩阵建模为 attributed graph，利用 GNN 捕捉其非零结构模式与数值分布特征。
- **高效的特征工程设计**：提出一种时间复杂度为 $O(\text{nnz} + n)$ 的特征提取方案，包含结构性、对角线属性、范数、对角占优性、行稀疏模式、非零值统计以及 Gershgorin 圆估计等八类特征，确保预处理开销极低。
- **两种预测方案（Prediction Schemes）**：
  - **Scheme 1**: 预测 $\|A^{-1}\|$，结合精确计算的 $\|A\|$ 构成最终 condition number 估计 $\hat{\kappa}(A) = \|A\| \cdot 10^{g(A;\theta)}$。
  - **Scheme 2**: 直接预测整个 condition number $\kappa(A)$。
- **混合架构设计**：采用双流模型——GCN 处理局部图结构信息，MLP 编码全局矩阵特征，并融合进行最终预测。

### 相比现有方法的优势
- **显著加速**：相比 Hager-Higham 和 Lanczos 方法实现 **数量级的速度提升**（5–10倍以上），推理时间达到亚毫秒级别。
- **可控精度**：尽管是近似方法，但在测试集上保持较低的 logarithmic relative error（LRE < 1%），满足多数应用场景需求。
- **可扩展性强**：推理时间几乎不随矩阵规模增长而明显增加，具备良好的 scalability。
- **适用于任意范数 condition number 估计**，突破了传统迭代方法在不同范数下的限制。

---

## 2. 核心实验方法和设置

### 数据集
训练数据由五类具有代表性的稀疏对称正定（SPD）矩阵构成，涵盖多种科学计算场景：
- **偏微分方程离散化**：
  - 二维泊松方程（2D Poisson）
  - 各向异性扩散方程（Anisotropic Diffusion, $\epsilon \sim 10^{-8}$ 到 $10^{-2}$）
  - 高对比度扩散问题（High-Contrast Diffusion, 条件数可达 $10^{10}$ 以上）
- **合成随机矩阵**：
  - 显式控制 condition number 的随机稀疏 SPD 矩阵（$\kappa_{\text{target}} = 10^T, T \sim U(7,13)$）
  - 对称三对角矩阵（Symmetric Tridiagonal Matrices），便于验证谱估计准确性

所有矩阵维度限定在 $n \in [1000, 2000]$ 范围内，共生成：
- 训练集：1,000 个矩阵
- 验证集：100 个矩阵
- 测试集：100 个矩阵

condition number 分布覆盖从良态到极度病态的广泛范围（见 Figure 2）。

### 实验设置与评估指标

#### 评估指标
- **Logarithmic Relative Error (LRE)**：
  $$
  \text{LRE}(A) = \frac{|\log_{10}\kappa(A) - \log_{10}\hat{\kappa}(A)|}{|\log_{10}\kappa(A)|}
  $$
  报告平均 LRE 和最大 LRE（LREmax），并统计 LRE < 0.5% 和 LRE < 1% 的样本比例。
  
- **运行时间与加速比**：
  - 平均推理时间（ms），取 4 次独立运行的中位数以减少噪声影响。
  - 加速比（Speedup Factor）：
    $$
    S_{M_1 \to M_2}(A) = \frac{T_{M_1}(A)}{T_{M_2}(A)}
    $$

#### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Exact** | 使用 `torch.linalg.cond` 基于 SVD 计算精确 condition number（GPU 实现） |
| **Hager-Higham** | 自定义实现，使用 `torch.linalg.lu_factor` 进行 LU 分解（GPU） |
| **Hager-Higham (SciPy)** | 使用 `scipy.sparse.linalg.onenormest` 等函数（CPU 实现） |
| **Lanczos (iter=5/10)** | 基于 `torch.lobpcg` 的 Lanczos 方法，分别迭代 5 和 10 次 |

提出的模型简称为 **GNN**。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### Scheme 1: $g(A) \sim \|A^{-1}\|$
| Norm | Method | Time (ms) | LRE (%) | LREmax (%) | LRE<0.5 | LRE<1 |
|------|--------|-----------|---------|------------|---------|-------|
| 1-norm | GNN | **13.04** | 1.93 | 9.30 | 100% | 100% |
|        | Hager-Higham | 31.67 | **1.82** | 36.04 | 100% | 100% |
|        | Exact | 201.83 | — | — | 100% | 100% |
| 2-norm | GNN | **25.57** | **1.19** | **9.74** | 100% | 100% |
|        | Lanczos (10) | 225.86 | 46.98 | 80.18 | 62% | 100% |

> ✅ **GNN 在 1-norm 上比 Hager-Higham 快约 2.4×，比 Exact 快 15×；在 2-norm 上比 Lanczos 快 8–9×，且误差远低于后者。**

#### Scheme 2: $g(A) \sim \kappa(A)$
| Norm | Method | Time (ms) | LRE (%) | LREmax (%) | LRE<0.5 | LRE<1 |
|------|--------|-----------|---------|------------|---------|-------|
| 1-norm | GNN | **12.61** | 3.30 | 60.97 | 98% | 100% |
|        | Hager-Higham | 25.13 | **1.82** | 36.04 | 100% | 100% |
| 2-norm | GNN | **11.25** | **3.90** | **44.86** | 100% | 100% |
|        | Lanczos (10) | 232.46 | 47.36 | 80.26 | 62% | 100% |

> ✅ **GNN 在两种 norm 下均最快（约 20–25× 快于 Exact，10× 快于 Lanczos），且绝大多数样本 LRE < 1%。**

### 与基线方法的对比结果
- **速度优势显著**：
  - GNN 推理时间稳定在 **~12 ms 左右**，而 Exact 方法超过 200 ms，Lanczos 达到 150–230 ms。
  - 即使与优化过的 Hager-Higham（GPU 版）相比，GNN 仍快 **2–3 倍**。
- **精度表现良好**：
  - 尽管 GNN 的平均 LRE 略高于 Hager-Higham（尤其在 Scheme 2），但其 **LREmax 更小或相当**，说明极端误差更可控。
  - 对于 2-norm，GNN 明显优于有限迭代次数的 Lanczos 方法（LRE 低一个数量级）。
- **可扩展性优异**（见 Figure 4 & 5）：
  - 所有方法中，GNN 的运行时间随矩阵大小和 nnz 增长最平缓，表现出接近常数时间的 inference 特性。

### 消融实验结果
文中未明确列出消融实验（ablation study），但通过以下方式间接体现设计有效性：
- **特征提取复杂度分析**：证明特征提取仅需 $O(\text{nnz} + n)$ 时间，远低于任何传统估计方法。
- **双流架构合理性**：强调将 $\|A\|$ 作为已知输入（Scheme 1）有助于网络专注于学习更难的 $\|A^{-1}\|$，提升数值稳定性。
- **训练曲线分析**（Figure 3）：训练与验证损失高度一致，表明模型泛化能力强，无明显过拟合。

---

## 4. 关键结论和发现

### 主要发现
- **GNN 可有效学习稀疏矩阵的 condition number**：通过合理构造图表示与多尺度特征，GNN 能捕捉影响 conditioning 的关键结构与数值规律。
- **预测 scheme 设计影响精度与鲁棒性**：
  - Scheme 1（分解式预测）通常获得更低的最大误差（LREmax），更适合对极端误差敏感的应用。
  - Scheme 2（端到端预测）速度略优，但误差波动稍大。
- **AI for Scientific Computing 的潜力**：本工作展示了深度学习在传统数值线性代数任务中的巨大加速潜力，特别是在需要高频调用 condition number 的场景（如 precision tuning、adaptive solver selection）。

### 方法的局限性
- **依赖训练数据分布**：模型性能受限于训练集的多样性；若测试矩阵来自完全不同的物理问题或具有异常结构，可能泛化不佳。
- **未探索最优架构与超参数**：当前模型为初步设计，尚未进行充分调优，存在进一步改进空间。
- **仅限 SPD 矩阵**：实验集中在对称正定矩阵，是否适用于一般非对称或不定矩阵有待验证。

### 未来工作方向
- 探索更先进的 GNN 架构（如 GIN、GINE、Transformer-based GNN）以增强表达能力。
- 开展跨领域泛化研究，评估模型在未见过的问题类别上的迁移能力。
- 实现端到端的 feature-free learning，让 GNN 自动学习最优特征表示。
- 将该方法集成至实际应用中，例如与 iterative refinement 或 mixed-precision solvers 联动，实现动态精度调节（precision autotuning）。

> 📌 **总体评价**：本文是 **首个将 GNN 成功应用于 condition number 估计的工作**，提出了高效、可扩展的解决方案，在保持可接受精度的前提下实现了数量级的速度提升，为 AI-driven numerical methods 提供了重要范例。

</details>

---

### 10. [Resource-constrained Amazons chess decision framework integrating large language models and graph attention](https://arxiv.org/abs/2603.10512)

**Authors**: Tianhao Qian, Zhuoxuan Li, Jinde Cao, Xinli Shi, Hanjie Liu, Leszek Rutkowski  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10512v1  

#### Abstract
Artificial intelligence has advanced significantly through the development of intelligent game-playing systems, providing rigorous testbeds for decision-making, strategic planning, and adaptive learning. However, resource-constrained environments pose critical challenges, as conventional deep learni...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Resource-constrained Amazons Chess Decision Framework Integrating Large Language Models and Graph Attention

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **资源受限环境下的智能决策挑战**：传统深度学习和强化学习方法依赖大量计算资源和高质量专家数据，在边缘设备或低算力场景中难以部署。
- **Amazons棋类游戏的复杂性**：该游戏具有极高的分支因子（每步可达数百甚至上千种合法走法）和巨大的搜索空间（可与围棋相比），且缺乏足够的历史对局数据用于监督学习。
- **弱监督信号的学习难题**：如何从噪声大、不完美的LLM生成数据中提炼出有效的策略知识。

### 🚀 提出的新方法与思路
作者提出了一种**轻量级混合框架**，结合图注意力机制与大语言模型（LLM），实现“**弱到强泛化**”（Weak-to-Strong Generalization）：

1. **Graph Attention Autoencoder (GAT-AE)**  
   - 利用图结构建模MCTS生成的游戏树节点关系，通过图注意力机制提取结构性策略特征。
   - 作为信息瓶颈（information bottleneck），有效过滤LLM输出中的错误（如非法移动、逻辑幻觉）。

2. **Stochastic Graph Genetic Algorithm (SGGA)**  
   - 将MCTS树转化为图结构，引入遗传算法进行候选节点优化选择。
   - 从**随机性视角**增强探索能力，提升搜索效率。

3. **基于GPT-4o-mini的合成数据生成**  
   - 不依赖人类专家对局记录，而是利用GPT-4o-mini为每一步打分（`move_score`, `place_score`），生成带噪声的训练标签。
   - 实现零样本迁移下的模型预训练。

4. **双阶段价值传播机制（Two-Pass Value Propagation）**
   - 包括深度相关累积（Depth-Dependent Accumulation）和全局深度归一化（Global Depth Normalization），缓解深层节点误差累积问题。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **资源效率** | 在普通硬件（如RTX 4060 Laptop GPU）上运行，显著降低部署门槛 |
| **无需专家数据** | 完全依赖LLM生成的弱监督信号，适用于数据稀缺领域 |
| **可解释性增强** | 结合手工设计评价函数与神经网络微调，优于纯端到端黑箱模型 |
| **泛化能力强** | 验证了“弱教师 → 强学生”的演化路径，具备向其他博弈/决策任务迁移的潜力 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **无真实人类对局数据集**：由于Amazons是小众游戏，公开历史对局极少。
- **自动生成数据集**：
  - 使用 **GPT-4o-mini** 对随机生成的棋盘状态及走法进行评分（`move_score` 和 `place_score` ∈ [0,1]）。
  - 输入格式包括当前棋盘布局、玩家颜色、移动起点/终点、障碍放置位置。
  - 输出由prompt引导，确保符合规则并提供连续评分。

### ⚙️ 实验设置
- **平台配置**：AMD Radeon(TM) 780M + NVIDIA GeForce RTX 4060 Laptop GPU
- **棋盘大小**：标准 **10×10 Amazons board**
- **搜索限制**：控制MCTS扩展节点数 $ N \in \{20, 30, 50\} $
- **训练方式**：
  - UCT-AE 使用 Adam/RMSprop 优化器分别训练移动与放置模块。
  - GAT-AE 使用图注意力层（8 heads）处理MCTS子图。
- **评估周期**：每个对比实验运行 **200 场比赛**

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Win Rate (%)** | 对抗不同对手时的胜率（主要指标） |
| **Decision Accuracy Improvement** | 相比基线提升15%-56% |
| **Loss Curve Stability** | 观察MSE损失收敛情况，分析方差差异（F-test验证） |
| **Variance of Loss** | 衡量训练稳定性（movement: $8.0×10^{-6}$, placement: $2.1×10^{-5}$） |

### 🆚 基线方法对比
| 基线模型 | 特点 |
|--------|------|
| **GPT-4o-mini** | “教师模型”，直接生成决策建议，作为性能上限参考 |
| **UCTS-AE** | 仅使用Autoencoder改进UCT，无图结构建模 |
| **SGGA** | 仅基于遗传算法选择节点，忽略结构信息 |
| **GAT-AE** | 仅有图注意力模块，缺少SGGA的随机优化机制 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Fig.15）
| 对手模型 | $N=20$ 胜率 | $N=30$ 胜率 | $N=50$ 胜率 |
|---------|-------------|-------------|-------------|
| **UCTS-AE** | 79.5% | 73.5% | — |
| **SGGA** | 58.5% | 59.0% | — |
| **GAT-AE** | 62.0% | 57.5% | — |
| **GPT-4o-mini** | — | **45.0%** | **66.5%** |

> ✅ **关键突破**：在仅 $N=50$ 节点搜索下，本框架已能**击败其教师模型 GPT-4o-mini**（66.5% > 50%）

### 🔬 决策准确率提升
- 相比各类基线模型，本方法在决策准确性上实现了 **15% ~ 56% 的绝对提升**。
- 即使在极低搜索预算（$N=30$）下也能达到接近平手水平（45.0% vs GPT-4o-mini），证明其高效性。

### 🔍 消融实验结果（Ablation Study）
| 组件移除 | 影响说明 |
|--------|--------|
| **无 SGGA** | 移动选择方差增大，训练不稳定（placement loss 波动更大） |
| **无 GAT-AE** | 无法捕捉长程结构依赖，易受LLM噪声干扰 |
| **无双Autoencoder结构** | 未能区分“移动”与“放障”两个动作的评价体系，性能下降明显 |
| **无深度归一化机制** | 深层节点估值偏差严重，导致搜索偏向无效分支 |

✅ **结论**：各组件互补性强，联合使用才能发挥最大效能。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **成功验证“弱到强泛化”范式**  
   - 即使以 **GPT-4o-mini 这样的通用LLM作为弱监督源**，也能训练出超越它的专用游戏AI。
   - 图注意力机制起到了关键的“去噪+结构提取”作用。

2. **GAT-AE 是信息过滤核心**  
   - 能有效识别并抑制LLM产生的非法操作或不合理评分，保留战略一致性模式（如领地控制、连通性维护）。

3. **SGGA 提升搜索多样性与鲁棒性**  
   - 通过遗传操作在MCTS图中探索更优路径，避免陷入局部最优。

4. **极低资源下仍具竞争力**  
   - 在消费级GPU上即可运行，远低于典型AlphaZero类系统的硬件需求。

### ⚠️ 局限性
| 问题 | 说明 |
|------|------|
| **未达顶尖引擎水平** | 如 Invader 等基于多年人工启发式的专用引擎，本模型ELO仍较低 |
| **训练完成度判断困难** | 损失值趋于平稳但难以确定是否充分收敛 |
| **最终决策依赖随机性** | 当前采用随机抽样决定最终动作，尚未建立统一决策函数 |
| **泛化至其他游戏需重构** | 虽然架构可迁移，但特征工程部分仍需针对具体游戏调整 |

### 🔮 未来工作方向
1. **构建自动终止训练判据**  
   - 探索基于性能 plateau 或梯度变化的自动化训练停止机制。

2. **开发统一决策策略函数**  
   - 替代当前随机选择机制，融合多模块输出形成确定性决策。

3. **拓展至多智能体与现实决策场景**  
   - 将“棋子=资产”、“走法=行动”抽象化，应用于交通调度、应急响应等实际系统。

4. **进一步压缩模型规模**  
   - 探索蒸馏或量化技术，使其可在移动端或嵌入式设备部署。

---

## 总结
该论文提出了一种面向**资源受限环境**的新型Amazons游戏AI框架，首次将 **LLM合成数据 + 图注意力 + 遗传搜索**有机结合，成功实现了从“弱监督”到“强行为”的跃迁。实验证明其不仅能在低端硬件上高效运行，还能在极小搜索规模下反超其教师模型，为数据稀缺、算力有限场景下的AI决策提供了新范式。

</details>

---

### 11. [Adaptive RAN Slicing Control via Reward-Free Self-Finetuning Agents](https://arxiv.org/abs/2603.10564)

**Authors**: Yuanhao Li, Haozhe Wang, Geyong Min, Nektarios Georgalas, Wang Miao  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10564v1  

#### Abstract
The integration of Generative AI models into AI-native network systems offers a transformative path toward achieving autonomous and adaptive control. However, the application of such models to continuous control tasks is impeded by intrinsic architectural limitations, including finite context window...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Adaptive RAN Slicing Control via Reward-Free Self-Finetuning Agents*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 Reinforcement Learning (RL) 在动态网络控制任务（如 RAN Slicing）中面临**奖励工程瓶颈**（reward engineering bottleneck），即设计有效的多目标 reward 函数需要大量手动调参，且难以平衡频谱效率（SE）、服务质量（QoS）和资源重配置稳定性之间的冲突。此外，现有的 LLM-based agents（如 Reflexion）受限于**有限上下文窗口**（finite context window）和**长上下文退化**（long context degradation），无法实现真正的持续学习，仅适用于短周期任务。

### 提出的新方法与创新思路
本文提出了一种**无需手工奖励信号的自微调框架**（Reward-Free Self-Finetuning Framework），其核心创新如下：

- **Reflective MDP (R-MDP)**  
  将传统的 MDP 扩展为支持语言反馈的形式，用自然语言形式的反思（reflection）替代标量 reward，使 LLM 能在无显式 reward 的情况下进行决策优化。

- **Actor-Reflector (AR) 架构**  
  受 Actor-Critic 启发，但将 Critic 替换为基于 LLM 的 **Reflector**，后者对完整轨迹进行语义级评估，生成“有效/无效”标签及改进建议，形成可训练的偏好数据集。

- **双视角反思机制（Bi-Perspective Reflection）**  
  - **Step-level Reflection**：由 Actor 在每步输出对上一步的自然语言反思，作为 prompt 内部记忆，实现快速适应；
  - **Trajectory-level Reflection**：由 Reflector 对整条轨迹进行回溯分析，识别长期影响下的最优行为，提升策略一致性。

- **Refine-from-Reflection (RfR) 微调框架**  
  利用 Kahneman-Tversky Optimization (KTO) 进行偏好微调，将 Reflector 生成的标注轨迹转化为 preference dataset，并结合 rollout 增强样本效率，实现经验内化到模型参数中。

### 相比现有方法的优势
| 维度 | 传统 RL | Reflexion 类方法 | 本文方法（Self-Finetuning） |
|------|--------|------------------|----------------------------|
| 是否依赖手工 reward | 是 | 是（隐式环境反馈） | 否（完全自生成反馈） |
| 学习方式 | 参数更新 + reward 回传 | Prompt-based 记忆 | 参数级 fine-tuning |
| 上下文限制影响 | 不适用 | 严重受限 | 被克服（通过内化经验） |
| 持续学习能力 | 弱（需持续采样） | 弱（依赖历史 truncation） | 强（单轨迹多次迭代优化） |
| 多目标权衡能力 | 依赖 reward 设计质量 | 有限 | 自动探索 Pareto 最优 |

> ✅ **核心优势**：实现了**无需 reward 工程的持续自适应控制**，并突破了 LLM 在连续控制中的上下文长度限制。

---

## 2. 核心实验方法和设置

### 实验环境
- 使用基于 Python 和 **ns-3** 构建的 **RAN Slicing 仿真器**，模拟真实无线信道与用户行为。
- 场景设定为 6G AI-Native 网络，部署 **RAN Intelligent Controller (RIC)** 实现闭环控制。

### 流量与信道模型
| 类型 | 参数 |
|------|------|
| **GBR Traffic** | 20个活跃 UE，速率 0.5 Mb/s，QoS 延迟 <10ms |
| **Non-GBR Traffic** | 4个活跃 UE，速率 2 Mb/s，QoS 延迟 <50ms |
| **Radio Channel** | Urban Propagation Loss Model (3GPP TR 38.901)，载频 2120MHz，噪声 5dB |

### 状态与动作空间
- **State**: $ s_t = [a_{t-1}, SE_t, u_t, o_t, d_t] $
  - 上一时刻动作、当前频谱效率、吞吐量、排队增量、丢包大小
- **Action**: 分配给各 Slice 的 PRB 数量（离散资源块）

### 评估指标
| 指标 | 定义 | 目标 |
|------|------|------|
| **Avg. SE** (Spectral Efficiency) | 单位带宽传输的数据量 | 最大化 |
| **Reconf. Times** | 频谱分配变化次数 | 最小化（提高稳定性） |
| **PQoS vio.** | QoS 违规时间步数 | 最小化 |
| **Utility** | 综合效用得分（文中未给出公式，据图表推断） | 最大化 |

### 基线方法对比
- **RL 方法**：
  - DQN（value-based）
  - PPO（policy-based）
  - SAC（maximum entropy）
- **LLM-Agent 方法**：
  - **Reflexion**：使用 Qwen3-4B 作 Actor，DeepSeek-R1 作 evaluator/reflector
- **本文方法（Self-Finetuning）**：
  - 同样采用 Qwen3-4B + DeepSeek-R1，确保公平比较

> ⚠️ 所有方法均在同一 backbone 下测试，性能差异归因于架构而非模型容量。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table III）

| Algorithm | Avg. SE | Reconf. Times | PQoS vio. | Utility |
|----------|---------|---------------|-----------|---------|
| **Self-Finetuning (Ours)** | **5.354** | **21.091** | 8.561 | **25702.2** |
| Reflexion | 5.299 | 29.454 | 8.630 | 25314.69 |
| DQN | 5.219 | 46.204 | 15.911 | 22519.1 |
| PPO | 3.587 | 51.411 | **1.997** | 19277.2 |
| SAC | 5.748 | 44.775 | 59.967 | 11704.3 |

> ✅ **最佳结果加粗，次佳下划线**

### 性能对比分析
- **综合表现最优**：Self-Finetuning 在 **Utility** 上全面领先，尤其在 **Reconf. Times** 上显著优于所有基线（比 PPO 低 59%，比 Reflexion 低 28.4%），说明其策略更稳定。
- **频谱效率高**：仅次于 SAC，但仍优于 DQN 和 PPO，表明资源利用率优秀。
- **QoS 控制均衡**：虽略逊于 PPO（专精 QoS），但远好于 SAC 和 DQN，体现多目标协调能力。
- **样本效率极高**：
  - RL 方法每轮收集 20 条轨迹，共训练 80 轮（总计 1600 条轨迹）仍未收敛；
  - Self-Finetuning **仅使用 1 条轨迹 + 1 次训练迭代** 即达到优越性能。

### 消融实验与训练动态（Fig. 3）
- **KTO 迭代过程显示明显学习趋势**：
  - 第一次 KTO 迭代后，“chosen” reward 显著上升，“rejected” reward 下降；
  - 经过 6 次 KTO iteration 后，两者趋于收敛，表明已充分挖掘单条轨迹的信息潜力。
- **Policy 改进效果显著**：
  - 单次训练后，**Reconf. Times 下降约 33%**；
  - PQoS violation 更加稳定，SE 提升；
  - 表明 RfR 框架可通过 refine-rollout 高效增强数据，避免额外环境交互。

---

## 4. 关键结论和发现

### 主要发现
1. **无需 reward 的自主学习是可行的**：通过语言级反思机制，LLM agent 可以自动生成优化信号，摆脱对手工 reward 的依赖。
2. **经验内化优于 prompt 记忆**：将长期交互经验通过 preference fine-tuning 写入模型权重，可有效克服 LLM 的上下文窗口限制，实现真正意义上的持续学习。
3. **RfR 框架具备极高的样本效率**：即使只有一条轨迹，也能通过 rollout 扩展和 KTO 多轮优化提取丰富知识，适合现实网络中稀疏交互场景。
4. **在多目标 RAN Slicing 中取得帕累托优势**：在 SE、Reconf.、QoS 之间达成良好平衡，优于各类 RL 和 LLM-agent 基线。

### 方法局限性
- **推理延迟较高**：当前依赖大型 LLM（如 Qwen3-4B, DeepSeek-R1），不适合实时性要求极高的控制环路（如 sub-ms 级调度）。
- **Reflector 成本高**：每次轨迹结束后需运行一次 full-sequence 推理，计算开销较大。
- **泛化能力待验证**：目前仅在一个特定 RAN 场景下验证，跨场景迁移能力尚未测试。

### 未来工作方向
- **模型压缩与加速**：通过 **imitation learning** 或 **policy distillation** 将学到的策略迁移到轻量级模型（lightweight LLM 或 DNN）中，用于实际部署。
- **硬件协同优化**：结合模型量化（quantization）、稀疏化与专用 AI 加速器，降低推理延迟。
- **扩展至其他网络控制任务**：如 bitrate adaptation、mobility management、edge orchestration 等连续控制问题。
- **引入不确定性建模**：增强 agent 在部分可观测环境下的鲁棒性，减少 hallucination。

---

> 📌 **总结一句话**：  
> 本文提出的 **Self-Finetuning 框架** 成功实现了 **无需 reward 的 LLM 自主持续学习**，在动态 RAN Slicing 任务中展现出卓越的多目标优化能力与样本效率，为构建真正 **AI-Native Autonomous Network** 提供了新范式。

</details>

---

### 12. [Does LLM Alignment Really Need Diversity? An Empirical Study of Adapting RLVR Methods for Moral Reasoning](https://arxiv.org/abs/2603.10588)

**Authors**: Zhaowei Zhang, Xiaohan Liu, Xuekai Zhu, Junchao Huang, Ceyao Zhang, Zhiyuan Feng, Yaodong Yang, Xiaoyuan Yi, Xing Xie  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10588v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has achieved remarkable success in logical reasoning tasks, yet whether large language model (LLM) alignment requires fundamentally different approaches remains unclear. Given the apparent tolerance for multiple valid responses in moral reasoning...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Does LLM Alignment Really Need Diversity? An Empirical Study of Adapting RLVR Methods for Moral Reasoning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文探讨了一个核心问题：**在将 Reinforcement Learning with Verifiable Rewards (RLVR) 方法应用于大语言模型（LLM）的对齐（Alignment）与道德推理（Moral Reasoning）任务时，是否必须引入多样性机制？**

传统观点认为，由于道德问题允许多种合理答案（如基于功利主义、义务论等不同伦理框架），因此需要 **distribution-matching** 这类能捕捉多样性的算法。而逻辑推理任务（如数学）通常只有一个正确解，更适合 **reward-maximizing** 的模式搜索方法。本文通过实证研究挑战了这一假设。

### 提出了什么新方法或新思路
- **构建了一个可扩展的 rubric-grounded reward pipeline**：为支持 RLVR 在 MoReBench 上的训练，作者训练了一个轻量级的 **Qwen3-1.7B Judge Model**，能够基于专家设计的评分细则（rubrics）自动打分，实现密集、可验证、多维度的奖励信号生成。
- **首次系统比较了 reward-maximizing 与 distribution-matching 方法在道德推理任务上的表现**，打破了“对齐任务天然需要多样性”的直觉认知。
- **提出并验证了一个反直觉的观点**：道德推理的高质量响应在语义空间中反而比数学推理更集中，因此 mode-seeking 方法同样甚至更加有效。

### 相比现有方法的优势
- **成本更低、效率更高**：相比直接使用 GPT-5 作为 Judge 进行训练，本地部署的小型 Judge 模型显著降低了推理延迟和调用成本，使大规模 RLVR 实验成为可能。
- **更具可复现性和可控性**：构建的 reward pipeline 可公开复现，避免了闭源 Judge 模型带来的黑箱问题。
- **理论启示更强**：揭示了 alignment 任务不一定具有多峰（multi-modal）奖励分布，为后续算法选择提供了新的指导原则。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 主要基准：**MoReBench (Chiu et al., 2025)**  
  包含两个子任务：
  - **MoReBench-Public**：现实世界中的价值困境（value-laden dilemmas）
  - **MoReBench-Theory**：基于特定哲学框架（utilitarianism, deontology, virtue ethics 等）的一致性推理
- 对比任务：**MATH-500**（Lightman et al., 2023），用于语义可视化分析

### 实验设置和评估指标
- **基础模型**：
  - `Qwen2.5-7B-Base`
  - `Llama3.1-8B-Instruct`
- **评估指标**：
  - **Score@1**：单次采样得分
  - **Avg@8**：8 次采样平均得分
  - 增益（Gain %）：相对于 Base 模型的相对提升百分比
- **Judge 模型构建流程**：
  1. 收集多个开源/闭源模型在 MoReBench 上的回答；
  2. 使用 GPT-5 根据 rubrics 打分，生成带标签的数据集；
  3. 对 Qwen3-1.7B-Base 进行 SFT，使其学会模仿 GPT-5 的评分行为；
  4. 验证其一致性：在 Public 和 Theory 子集上分别达到 87.07% 和 69.21% 的 agreement。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **Reward-Maximizing (Mode-Seeking)** | PPO, GRPO, REINFORCE++, DAPO |
| **Distribution-Matching (Multi-Modal Coverage)** | FlowRL |
| **Baseline** | Base（未经 RL 微调的原始模型） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 MoReBench-Public 上的表现（以 Avg@8 为例）
| 方法 | Qwen2.5-7B Gain(%) | Llama3.1-8B Gain(%) |
|------|---------------------|-----------------------|
| DAPO | **81.08%** | **60.00%** |
| FlowRL | 64.86% | 33.33% |
| RFPP | 75.68% | 33.33% |

#### 在 MoReBench-Theory 上的表现（以 Avg@8 为例）
| 方法 | Qwen2.5-7B Gain(%) | Llama3.1-8B Gain(%) |
|------|---------------------|-----------------------|
| DAPO | **67.44%** | **49.02%** |
| FlowRL | 51.16% | 37.25% |

> ✅ **所有 reward-maximizing 方法均优于或持平于 FlowRL**，其中 **DAPO 表现最佳**。

### 与基线方法的对比结果
- **DAPO 显著领先于所有其他方法**，说明即使在看似开放的道德推理任务中，强优化能力的 reward-maximizing 方法依然最有效。
- **FlowRL 并未展现出预期中的多样性优势**，其 Avg@8 指标并未超过 DAPO，表明其“覆盖多种高奖励路径”的特性并未转化为实际性能增益。
- **RFPP 和 GRPO 也普遍优于 FlowRL**，进一步削弱了“多样性必要论”。

### 消融实验结果（间接体现）
虽然没有明确列出消融实验表格，但以下分析起到了类似作用：
- **语义可视化分析（Figure 1）**：将 MATH-500 与 MoReBench-Public 的高奖励响应映射到语义空间（all-MiniLM-L6-v2 + t-SNE）。结果显示：
  - 数学推理呈现**多个分散簇**（multi-cluster），对应不同解题策略；
  - 道德推理则高度**聚集在一个主导区域**（uni-modal），说明高分回答趋向于相似的价值判断模板。
- **案例研究（Table 2）**：展示不同方法生成的回答，发现尽管表面表述略有差异，但核心推理结构（situation analysis → pros/cons comparison → compromise decision）高度一致，最终建议均为“诚实反馈 + 私下沟通”，缺乏真正的伦理多元性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ❗ **LLM 对齐任务并不必然需要多样性机制**：尽管道德问题允许多种价值观解释，但在当前 reward 设计下，高奖励响应实际上集中在少数“安全且平衡”的解决方案上。
2. ✅ **reward-maximizing 方法在道德推理中表现更优或相当**：特别是 DAPO，在两个基准和两个 base model 上均取得最好成绩。
3. 🔍 **道德推理的 reward landscape 更加集中（concentrated）而非分散**：语义可视化显示其高奖励区域呈单峰分布，这解释了为何 mode-seeking 方法可以高效收敛。
4. 🔄 **现有 RLVR 方法可直接迁移到 alignment 任务**：无需专门设计 diversity-promoting 算法，标准的 reward-maximizing RLVR 即可有效提升道德推理能力。

### 方法的局限性
- **依赖于 MoReBench 的 rubric 设计**：当前结论建立在 MoReBench 提供的细粒度评分标准之上，若换用更主观或更开放的评价体系，结果可能不同。
- **Judge 模型存在偏差**：尽管 Qwen3-1.7B Judge 与 GPT-5 有较高一致性，但仍可能存在系统性偏移，影响 reward 质量。
- **多样性定义受限**：文中“多样性”主要指输出路径的分布广度，未涵盖文化、身份、边缘群体视角等更深层的社会多样性。

### 未来工作方向
1. **探索更多 alignment benchmarks**：目前适合 RLVR 的道德推理数据集稀少，需推动更多高质量、可验证的任务建设。
2. **改进 distribution-matching 方法**：当前仅测试了 FlowRL，未来可开发更先进的 multi-modal learning 算法。
3. **研究 reward engineering 的影响**：不同的 reward 定义（如强调公平性、包容性）是否会真正激发模型展现伦理多样性？
4. **结合人类反馈进行混合训练**：将 RLVR 与 RLHF 结合，探索如何在保持推理能力的同时增强价值敏感性。

---

> **一句话总结**：  
> 本论文通过严谨实验发现，**LLM 对齐任务在现有 reward 构建方式下并不需要显式的多样性机制**；相反，**reward-maximizing 的 RLVR 方法（如 DAPO）已足够有效**，因其面对的是一个**意外集中的高奖励语义空间**。这一发现挑战了领域内的主流直觉，为 alignment 算法设计提供了重要反思。

</details>

---

### 13. [Data Augmentation and Convolutional Network Architecture Influence on Distributed Learning](https://arxiv.org/abs/2603.10902)

**Authors**: Victor Forattini Jansen, Emanuel Teixeira Martins, Yasmin Souza Lima, Flavio de Oliveira Silva, Rodrigo Moreira, Larissa Ferreira Rodrigues Moreira  
**Category**: cs.DC  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10902v1  

#### Abstract
Convolutional Neural Networks (CNNs) have proven to be highly effective in solving a broad spectrum of computer vision tasks, such as classification, identification, and segmentation. These methods can be deployed in both centralized and distributed environments, depending on the computational deman...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Data Augmentation and Convolutional Network Architecture Influence on Distributed Learning**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
本文聚焦于当前深度学习研究中一个被忽视的关键问题：**Convolutional Neural Networks (CNNs)** 在分布式训练环境下的 **硬件资源消耗影响**，特别是 **Data Augmentation (DA)** 和 **CNN 架构深度** 对 GPU、CPU、内存、网络流量等资源的影响。

尽管现有研究多关注模型准确率和可解释性，但在 **distributed learning** 场景下，对计算资源的系统性分析仍存在明显空白。本文填补了这一空白，尤其强调了 **网络通信开销** 这一常被忽略的维度。

### ✅ 提出的新方法与思路
- 采用 **2² Factorial Design（因子设计）** 方法，系统性地分析两个关键因素（DA 与 CNN 架构）在分布式训练中的主效应及交互效应。
- 将传统机器学习实验设计方法（ANOVA）引入到 **distributed deep learning** 性能评估中，提供了一种量化影响程度的新范式。
- 构建了一个基于真实农业图像数据集（Paddy Doctor）的分布式训练测试平台，结合监控工具 **NetData** 实时采集硬件资源使用情况。

### ✅ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **视角创新** | 不仅关注模型精度，更全面评估 **硬件资源利用率**，为绿色AI和边缘部署提供依据 |
| **方法论严谨** | 使用统计学上的 **factorial analysis** 定量分离各因素影响，提升结论可信度 |
| **实际指导意义强** | 揭示 DA 显著增加网络负载，提示在带宽受限场景需谨慎使用 |

---

## 2. **核心实验方法和设置**

### 📁 数据集
- **Paddy Doctor Dataset**
  - 包含 **16,225 张标注图像**
  - 覆盖 **13 类水稻叶片状态**（12 种病害 + 健康）
  - 图像来自真实稻田，由农业专家标注，具有高现实代表性

### ⚙️ 实验设置
- **分布式架构**：
  - 双服务器通过 LAN（1 Gbps 接口，协商至 100 Mbps）连接
  - 使用 **Torch Distributed Data Parallel (DDP)** 作为后端
- **硬件配置**：
  
  | Server | CPU | RAM | GPU |
  |--------|-----|-----|-----|
  | #1     | Intel i5-4430 3.00GHz | 32GB | RTX 4060 Ti 8GB |
  | #2     | Intel i5-4430 3.00GHz | 16GB | GTX 1050 Ti 4GB |

- **CNN 模型选择**：
  - **Shallow-CNN**: MobileNetV2-100（轻量级，较少卷积层）
  - **Deep-CNN**: MobileOne-S1（更深结构，更多参数）

- **Data Augmentation Pipeline**：
  - 随机旋转 (±5°)
  - 仿射变换（剪切 0.2，平移 ±5%）
  - 随机裁剪（原图大小的 80–100%）
  - 水平翻转
  - 颜色抖动（color jittering）

- **训练设置**：
  - 最大训练轮数：100 epochs
  - 实际采用 early stopping（约 20 轮收敛）
  - 所有组合均重复运行取平均值

### 🎯 评估指标（Response Variables）
| 指标 | 描述 |
|------|------|
| `YGPU (%)` | GPU 平均利用率 |
| `YCPU (%)` | CPU 平均利用率 |
| `YMemory (%)` | 内存占用率 |
| `YNetworkPackets (Pkts/s)` | 每秒传输的网络包数量 |
| `YAccuracy (%)` | 测试集分类准确率 |

### 🔁 实验设计：2² Factorial Design
| 因素 | 水平 |
|------|------|
| A: Data Augmentation (DA) | with-DA (+1), without-DA (−1) |
| B: CNN Architecture | shallow-CNN (+1), deep-CNN (−1) |

共 4 种实验组合，用于分析主效应（TA, TB）和交互效应（TAB）。

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（见 Table 3）

| 实验组合 | YGPU (%) | YNetworkPackets (Pkts/s) | YCPU (%) | YMemory (%) | YAccuracy (%) |
|---------|----------|----------------------------|----------|-------------|----------------|
| with-DA + shallow-CNN | 95.12 | 19,994.50 | 51.15 | 81.70 | **98.71** |
| without-DA + shallow-CNN | 97.18 | 15,698.97 | 47.38 | 81.75 | **99.60** |
| with-DA + deep-CNN | 97.21 | 19,973.00 | 47.03 | 80.45 | 94.09 |
| without-DA + deep-CNN | **98.29** | **10,526.36** | **43.85** | 81.45 | **96.58** |

> 💡 注：加粗表示该列最优值

### 🔍 主要发现（来自 Table 4：Factor Influence Analysis）

| 因素 | YGPU | YNetworkPackets | YCPU | YMemory | YAccuracy |
|------|------|------------------|-------|----------|------------|
| **TA (DA 影响)** | 46.83% | **77.92%** | 45.07% | 24.53% | 15.86% |
| **TB (CNN 架构影响)** | 48.64% | 11.13% | 54.61% | 54.75% | **80.60%** |
| **TAB (交互影响)** | 4.53% | 10.94% | 0.32% | 20.72% | 3.54% |

#### ✅ 性能对比与消融分析结果
- **DA 对网络通信影响极大**：
  - 引入 DA 后，**network packets 增加高达 77.92%**，是最大影响因素。
  - 特别是在 deep-CNN 下，DA 导致网络包增长 **89.73%**，显著加重通信负担。
- **CNN 架构深度主导 accuracy 表现**：
  - 架构因素（TB）对 accuracy 的影响达 **80.60%**，远高于 DA 的 15.86%。
  - shallow-CNN + without-DA 组合达到最高 accuracy（99.60%），优于所有含 DA 的组合。
- **DA 反而略微降低 accuracy**：
  - 在本实验中，**不使用 DA 的模型准确率更高**，可能因过增强导致噪声引入或分布偏移。
- **memory usage 几乎不受影响**：
  - 各组内存消耗稳定在 ~81.5%，说明 DA 和架构变化未显著改变内存压力。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **Data Augmentation 显著增加网络通信开销**：
   - 是分布式训练中 **network packet transmission 的最主要驱动因素**（77.92% 影响力）。
   - 尤其在异构设备（如不同性能 GPU）间同步时，会加剧等待时间与通信瓶颈。

2. **CNN 架构深度对 accuracy 和 CPU 利用率影响最大**：
   - deeper CNN 更耗 CPU 但未必带来 accuracy 提升；相反，浅层网络配合无 DA 表现最佳。

3. **accuracy 与资源消耗之间存在权衡**：
   - 最高 accuracy 出现在 **shallow-CNN + without-DA** 组合（99.60%），而非最复杂设置。
   - 表明盲目增加模型深度或数据增强不一定提升性能，反而增加成本。

4. **Factorial Design 成功揭示变量间关系**：
   - 证明 ANOVA 方法可用于 deep learning 系统工程优化，具备推广价值。

### ⚠️ 局限性
- **仅测试两种 CNN 模型**：MobileNetV2 与 MobileOne-S1，缺乏更广泛架构比较（如 ResNet、EfficientNet）。
- **DA 策略固定**：未探索其他增强方式（如 Mixup、CutOut）是否对通信影响不同。
- **early stopping 限制长期趋势观察**：未能分析超过 20 轮后的资源动态变化。
- **双节点规模小**：无法反映大规模集群中的扩展行为。

### 🔮 未来工作方向
- 扩展至更多 CNN 架构与超参数组合，构建 **resource-aware NAS（神经架构搜索）** 框架。
- 探索 **adaptive DA strategies**，在保证泛化能力的同时最小化通信开销。
- 研究 **gradient compression** 或 **quantization techniques** 如何缓解 DA 带来的通信压力。
- 将该方法应用于 **edge computing** 或 **federated learning** 场景，优化低带宽环境下的部署效率。

---

> 📌 **总结一句话**：  
> 本文首次系统量化了 **Data Augmentation** 和 **CNN 架构** 在分布式训练中对硬件资源的影响，揭示了 **DA 是网络通信瓶颈的主要来源**，而 **模型架构才是决定 accuracy 的关键**，为高效、节能的 deep learning 部署提供了重要决策依据。

</details>

---

### 14. [GaLoRA: Parameter-Efficient Graph-Aware LLMs for Node Classification](https://arxiv.org/abs/2603.10298)

**Authors**: Mayur Choudhary, Saptarshi Sengupta, Katerina Potika  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10298v1  

#### Abstract
The rapid rise of large language models (LLMs) and their ability to capture semantic relationships has led to their adoption in a wide range of applications. Text-attributed graphs (TAGs) are a notable example where LLMs can be combined with Graph Neural Networks to improve the performance of node c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GaLoRA: Parameter-Efficient Graph-Aware LLMs for Node Classification

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
本文针对**Text-Attributed Graphs (TAGs)** 中节点分类任务中存在的挑战提出解决方案。这类图中每个节点既包含**文本内容**（如用户简介、论文摘要），又具有**图结构关系**（如社交关注、引用网络）。传统方法通常单独建模结构或语义信息，而联合建模则面临以下问题：
- **计算开销大**：同时训练 GNN 和 LLM 导致参数量巨大；
- **难以扩展**：全模型微调在资源受限场景下不可行；
- **模块耦合性强**：结构与语义学习紧密绑定，缺乏灵活性。

### 🚀 提出了什么新方法或新思路  
作者提出了 **GaLoRA (Graph-aware Low-Rank Adaptation)**，一种**参数高效**且**模块化设计**的框架，用于将图结构信息注入大型语言模型（LLM）以提升节点分类性能。其核心思想是两阶段解耦训练：

1. **Phase 1: GNN Training**  
   使用 GNN（如 GraphSAGE）从 TAG 学习结构感知的节点嵌入（structure-aware embeddings），输出 Pass-1（1-hop 邻域）和 Pass-2（2-hop 邻域）表示。

2. **Phase 2: LLM Fine-tuning with LoRA**  
   冻结预训练 LLM，在其特定 Transformer 层中通过 **Low-Rank Adaptation (LoRA)** 注入来自 GNN 的结构嵌入：
   - 将 Pass-1 嵌入注入中间层（捕捉局部上下文）
   - 将 Pass-2 嵌入注入上层（构建全局理解）
   - 引入可学习门控机制 $\alpha$ 动态平衡文本与结构输入的重要性。

该方法实现了 **“结构-语义”融合而不进行端到端联合训练**，显著降低训练成本。

### ⭐ 相比现有方法的优势  

| 方法 | 是否微调 LLM | 参数效率 | 模块化 | 主要缺点 |
|------|----------------|------------|----------|-----------|
| **GLEM** | 是（全量微调） | ❌ 极低（~100% 参数） | 一般（迭代伪标签） | 对噪声敏感，计算重 |
| **TAPE** | 否（仅提示） | ✅ 高 | ✅ 高 | 依赖人工提示，效果不稳定 |
| **GraphAdapter** | 否（冻结 LLM） | ✅ 极高 | ✅ 高 | 无法适应任务级语义 |
| **GaLoRA (Ours)** | ✅（仅 LoRA 微调） | ✅✅ **极高**（仅 0.24% 参数） | ✅✅ 高 | 当前限于节点分类 |

> ✅ **GaLoRA 的优势总结**：
> - **参数极省**：仅需训练 ~0.295M 参数（占 GPT-2 的 **0.238%**）；
> - **性能强劲**：在多个数据集上媲美甚至超越 SOTA；
> - **模块解耦**：GNN 与 LLM 可独立训练，便于部署与调试；
> - **灵活适配**：支持不同 GNN 和 LLM 组合，易于扩展。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集  
实验在三个真实世界的 TAG 数据集上进行：

| Dataset | #Nodes | #Edges | Task | Metric | Description |
|--------|--------|--------|------|--------|-------------|
| **ArXiv** | 46,198 | 78,548 | 40类论文分类 | Accuracy | 节点为论文，文本为标题+摘要，边为引用关系 |
| **Instagram** | 11,339 | 144,010 | 商业/非商业用户二分类 | ROC-AUC | 用户 bio 作为文本，关注关系为图结构 |
| **Reddit** | 33,434 | 198,448 | 流行/非流行用户二分类 | Accuracy | 用户最近三篇帖子文本，互动关系构图 |

> 所有数据使用 **stratified split**（ArXiv: 54%/18%/28%，其余: 80%/10%/10%）

### ⚙️ 实验设置  
- **硬件环境**：Google Colab + NVIDIA A100 GPU（52GB VRAM）
- **工具库**：PyTorch, PyG, HuggingFace Transformers
- **GNN 模型**：GraphSAGE（2层消息传递，mean pooling，embedding size=64）
- **LLM 模型**：GPT-2 / RoBERTa（base 版本，~125M 参数）
- **LoRA 设置**：
  - 注入层：GPT-2 的第 5–7 层（middle）和 9–11 层（upper）
  - 适配权重：`c_attn`, `c_proj`（GPT-2）；`q,k,v,dense`（RoBERTa）
  - LoRA rank $r=4$
- **优化器**：AdamW，lr=3e-4，weight decay=1e-2，batch size=32
- **损失函数**：Cross-Entropy

### 🔁 基线方法对比  
主要对比以下模型：
- **GNN-only**：仅用 GraphSAGE 分类
- **LLM-only / LLM (LoRA)**：仅基于文本内容分类
- **GLEM**：交替训练 GNN 与 LLM，使用伪标签
- **TAPE**：利用 LLM 生成解释，再由小模型编码供 GNN 使用
- **GraphAdapter**：冻结 LLM，添加轻量 GNN Adapter 注入结构信息（当前 SOTA）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（LLM 控制变量比较）

> 使用相同 LLM（RoBERTa/GPT-2）控制变量下的结果（均值±标准差，5次随机种子）

| Model | ArXiv (Acc) | Instagram (ROC-AUC) | Reddit (Acc) |
|-------|--------------|------------------------|---------------|
| GNN (PLM) | 0.7129 | 0.6019 | 0.6282 |
| GraphAdapter | 0.7273 | 0.6276 | 0.6441 |
| **GaLoRA (Ours)** | **0.7550** | **0.6420** | **0.6611** |

> ✅ **GaLoRA 在所有数据集上均优于 GraphAdapter**，尤其在 ArXiv 上提升明显（+2.77%）。

### 🔍 与其他工作的跨模型比较（Table 11）

| Model | ArXiv (Acc) | Instagram (ROC-AUC) | Reddit (Acc) |
|-------|--------------|------------------------|---------------|
| GraphAdapter (LLaMA-13B) | **0.7707** | **0.6513** | 0.6461 |
| GLEM (DeBERTa-Large) | 0.7315 | 0.6105 | 0.6221 |
| **GaLoRA (GPT-2)** | **0.7550** | **0.6420** | **0.6611** |

> 💡 尽管 GaLoRA 使用更小的 GPT-2（124M vs LLaMA-13B 的 13B），但在 **Reddit 上反超 GraphAdapter**，说明其结构融合机制非常有效。

### 📉 参数效率对比（Table 3）

| Model | PLM | #Trainable Params | 占 PLM 参数比例 |
|-------|-----|--------------------|------------------|
| GLEM | DeBERTa-Large (435M) | 435M | 100% |
| GraphAdapter | LLaMA-13B (13B) | 2M | **0.015%** |
| **GaLoRA (Ours)** | **GPT-2 (124M)** | **0.295M** | **0.238%** |

> ✅ GaLoRA 是唯一一个**既微调 LLM 又保持超高参数效率**的方法。若应用于更大模型（如 LLaMA），相对比例将进一步下降。

### 🔬 消融实验结果（Ablation Studies）

#### A. LoRA Rank 影响（Instagram, ROC-AUC）

| Rank $r$ | ROC-AUC |
|---------|--------|
| 2       | 0.6347 |
| 4       | 0.6420 |
| 8       | 0.6421 |

> 结论：$r=4$ 已达性能饱和，更高秩带来边际收益，推荐使用中等秩实现最佳性价比。

#### B. Prompt Engineering 效果（Instagram）

| Prompt | ROC-AUC |
|--------|--------|
| No prompt | 0.6283 |
| "Classify:" | 0.6305 |
| "Classify this instagram account bio:" | 0.6344 |
| "Classify instagram account is commercial or not:" | **0.6420** |

> 发现：即使是简单提示也能显著提升性能，最明确的任务描述效果最好。

#### C. 不同 GNN 骨干的影响（Table 10）

| Model | Instagram | Reddit | ArXiv |
|-------|-----------|--------|-------|
| GraphSAGE + GPT-2 | 0.6420 | 0.6611 | 0.7550 |
| GAT + GPT-2 | **0.6616** | 0.6613 | 0.7569 |

> 初步表明更换更强 GNN（如 GAT）可能进一步提升性能，值得深入探索。

---

## 4. 关键结论和发现

### ✅ 主要发现  
1. **结构信息可以高效注入 LLM**：通过 LoRA 机制将 GNN 学得的结构嵌入注入 LLM，能显著增强其对 TAG 节点的理解能力。
2. **无需联合训练即可实现结构-语义融合**：两阶段解耦设计避免了复杂的联合优化，同时保留了两种模态的优势。
3. **小模型也能打败大模型**：即使使用 GPT-2 这样的小型 LLM，GaLoRA 仍能在部分任务上超越基于 LLaMA-13B 的 SOTA 方法。
4. **参数效率极高**：仅需微调约 **0.24% 的 LLM 参数**，适合部署在资源受限环境中。

### ⚠️ 方法的局限性  
1. **目前仅适用于节点分类任务**，尚未验证在 link prediction、graph classification 等任务上的有效性。
2. **依赖高质量的 GNN 表示**：若 GNN 学习不到有效的结构特征，会影响最终性能。
3. **存在潜在的信息泄露风险**：使用 stratified split 可能在 GNN 消息传递过程中引入偏差。
4. **未探索更复杂的融合机制**：当前采用简单的投影+门控融合，未来可尝试注意力、交叉编码等方式。

### 🔮 未来工作方向  
1. **拓展至其他图任务**：如 link prediction、社区检测、图分类等。
2. **引入更先进的 GNN 架构**：尝试 GAT、GraphFormer 或异构图神经网络。
3. **开发动态注入策略**：根据节点属性自适应选择注入层数或方式。
4. **结合 prompt tuning 或 prefix tuning**：进一步提升语义对齐能力。
5. **构建新的 benchmark splits**：采用 structure-aware split 防止信息泄露。
6. **验证在更大 LLM 上的表现**：如应用到 LLaMA、Qwen 等百亿级模型。

---

> ✅ **总体评价**：GaLoRA 是一项兼具**实用性**与**创新性**的工作，为在资源受限条件下构建结构感知的 LLM 提供了一个高效、灵活的新范式，有望推动 LLM+Graph 在工业界的大规模落地。

</details>

---

### 15. [Safe and Scalable Web Agent Learning via Recreated Websites](https://arxiv.org/abs/2603.10505)

**Authors**: Hyungjoo Chae, Jungsoo Park, Alan Ritter  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.10505v1  

#### Abstract
Training autonomous web agents is fundamentally limited by the environments they learn from: real-world websites are unsafe to explore, hard to reset, and rarely provide verifiable feedback. We propose VeriEnv, a framework that treats language models as environment creators, automatically cloning re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Safe and Scalable Web Agent Learning via Recreated Websites**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前在真实网站上训练自主 **web agent** 存在三大根本挑战：
- **安全性问题**：直接与真实网站交互可能导致干扰其他用户、违反平台政策（如被 Cloudflare/CAPTCHA 阻断）、甚至造成数据修改等风险。
- **不可复位性**：真实网站状态难以重置，无法支持重复训练和公平评估。
- **奖励信号不可靠**：自生成任务常缺乏明确目标，依赖 **LLM-as-a-Judge** 进行轨迹评估，易产生误判（false positives/negatives），导致学习不稳定。

### **提出了什么新方法或新思路**
提出 **VERIENV** 框架，其核心思想是：
> **将语言模型作为环境构建者（language models as environment creators）**，自动克隆真实网站为可执行的合成环境，并在此环境中进行安全、可验证的 agent 自演化训练。

#### **VERIENV 的关键设计**
- **环境克隆（Cloning）**：使用 coding agent（如 GPT-5.2）从截图重建网站的前端、后端逻辑和数据库，形成一个功能完整的 **synthetic environment**。
- **Python SDK 接口**：提供对内部状态（如数据库）的可控访问，用于任务验证和状态查询。
- **可验证任务生成（Verifiable Task Generation）**：结合自然语言指令与基于 SDK 的 **executable validation program**，确保每个任务有唯一正确答案，可通过程序化规则（如 `must_include`, `exact_match`）进行确定性判断。
- **确定性奖励信号**：通过执行验证程序获得二值奖励（binary reward），完全摆脱对 LLM judge 的依赖。

### **相比现有方法的优势**
| 维度 | 传统方法（如 PAE, WebRL） | VERIENV |
|------|--------------------------|--------|
| **环境安全性** | 直接操作真实网站 ❌ | 在合成环境中训练 ✅ |
| **任务可验证性** | 依赖 LLM judge，存在歧义和误判 ❌ | 程序化验证，确定性判断 ✅ |
| **奖励可靠性** | 启发式或主观评分 ❌ | 可审计、可复现的规则判断 ✅ |
| **训练可扩展性** | 受限于真实网站可用性 ❌ | 可无限扩展克隆环境 ✅ |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练环境来源**：基于 **Mind2Web** 和 **Mind2Web-Online** 中的 149 个真实网站，使用 coding agent 克隆生成对应的 **synthetic environments**。
- **总任务量**：共生成 **7,400 个任务**，按难度分为：
  - Easy: 2,972 (40.2%)
  - Medium: 2,900 (39.2%)
  - Hard: 1,528 (20.6%)

### **实验设置和评估指标**
#### **评估基准（Benchmarks）**
1. **WebArena-Lite**：包含 5 个现实网站（Shopping, CMS, Reddit, GitLab, Map），测试跨域泛化能力。
2. **Mind2Web-Online**：覆盖 100+ 真实网站，300 个任务，分 Easy/Medium/Hard 三个难度等级。

#### **评估方式**
- **Success Rate**：任务是否成功完成。
- **Human Evaluation**：对 30 个随机样本进行人工评估，指标包括：
  - 功能正确性（Functional correctness）
  - 视觉质量（Visual rating, 1–5 Likert）
  - 任务可执行性（Task executability）
  - 判定器正确性（Judge correctness）

#### **训练方法**
- 使用 **rejection-based fine-tuning**：仅保留成功完成验证的任务轨迹作为监督信号。
- 基座模型：**Qwen3-4B** 和 **LLaMA-3.2-3B-Instruct**。

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **GPT-4o-mini / GPT-4o / Claude-3.5-Sonnet** | 商业闭源大模型，代表 SOTA 性能 |
| **Synatra (Ou et al., 2024)** | 基于教程生成合成轨迹 |
| **ADP (Song et al., 2025)** | 聚合多个 agent 数据集并标准化动作格式 |
| **PAE (Zhou et al., 2025b)** | 自演化框架，但在真实网站上运行，依赖 LLM judge |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **Table 4: WebArena-Lite 结果**
| Method | Total Success Rate | Improvement |
|--------|--------------------|-------------|
| Qwen3-4B | 7.88 | — |
| **+VERIENV (Ours)** | **13.94** | **+6.06** |
| LLaMA-3.2-3B-Instruct | 3.03 | — |
| **+VERIENV (Ours)** | **12.73** | **+9.70** |

> ✅ VERIENV 显著提升两个开源模型在跨域场景下的表现。

#### **Table 5: Mind2Web-Online 结果**
| Method | Total Success Rate | Improvement |
|--------|--------------------|-------------|
| Qwen3-4B | 13.18 | — |
| **+VERIENV (Ours)** | **20.45** | **+7.27** |
| LLaMA-3.2-3B-Instruct | 11.36 | — |
| **+VERIENV (Ours)** | **24.55** | **+13.19** |

> ✅ 在更复杂的真实网站上，VERIENV 依然显著优于基线。

### **与基线方法的对比结果**
- **相比 ADP**：ADP 表现不稳定，对 Qwen 模型甚至出现负增益（-1.82），而 VERIENV 在所有设置下均带来正向提升。
- **相比 PAE**：在 site-specific mastery 实验中（Figure 4），VERIENV 在 CMS 和 Shopping 类网站上持续提升，而 PAE 因非可验证奖励陷入停滞。

### **消融实验结果**
#### **环境数量扩展分析（Figure 5）**
- 随着训练环境数量增加，agent 性能在 **WebArena** 和 **Mind2Web-Online** 上均持续提升。
- 表明 **environment scaling** 是提升 agent 能力的有效路径，尤其在可验证训练范式下。

#### **可验证性的重要性（Figure 6）**
- **PAE** 生成的任务可能存在多解（如“找一篇关于 braising 的文章”），导致 LLM judge 错误地将非目标页面判定为成功。
- **VERIENV** 任务具有唯一正确答案（如“列出以 A 开头的前两个食材”），并通过 SDK 验证终端状态，避免误判。

---

## **4. 关键结论和发现**

### **主要发现**
1. **可验证环境显著提升 agent 泛化能力**：在 WebArena 和 Mind2Web-Online 上，VERIENV 均带来 **+6~13+ 点的成功率提升**。
2. **自演化训练可在固定网站上实现“精通”**：在单一克隆网站内反复训练，agent 能持续提升 site-specific performance，证明合成环境适合作为“训练健身房”。
3. **环境扩展有效**：增加训练环境数量能系统性提升 agent 能力，支持 **environment-centric scaling** 作为 agent 发展的新范式。
4. **可验证奖励比 LLM judge 更可靠**：程序化验证减少了误判，提供了更稳定的学习信号。

### **方法的局限性**
1. **克隆保真度限制**：并非所有网站都能完美重建，尤其是依赖多媒体（如视频流、PDF 渲染）的平台。
2. **判定器正确性有待提高**：人工评估显示 judge correctness 为 **76%**，低于 task executability（90%），主因是数据库重置未保留随机种子。
3. **coding agent 成功率有限**：在 136 个候选网站中，仅 **97 个** 成功完成克隆流程（见 Figure 9），失败原因包括端口冲突、CORS 错误、SDK 未创建等。

### **未来工作方向**
1. **强化学习应用**：利用 VERIENV 提供的确定性奖励，开展更稳定的 **RL for web agents**。
2. **提升克隆自动化与鲁棒性**：引入 Docker 隔离部署、改进调试流程，降低基础设施错误。
3. **增强视觉真实性**：使用 **text-to-image models** 自动生成产品图片，提升购物类网站的视觉保真度。
4. **研究 sim-to-real 迁移**：系统评估在合成环境中学到的策略在真实网站上的迁移效果，并加入安全评估。

---

> **代码与资源**：将在接受后发布至 [https://github.com/kyle8581/VeriEnv](https://github.com/kyle8581/VeriEnv)

</details>

---

### 16. [Robust Post-Training for Generative Recommenders: Why Exponential Reward-Weighted SFT Outperforms RLHF](https://arxiv.org/abs/2603.10279)

**Authors**: Keertana Chidambaram, Sanath Kumar Krishnamurthy, Qiuling Xu, Ko-Jen Hsiao, Moumita Bhattacharya  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.10279v1  

#### Abstract
Aligning generative recommender systems to user preferences via post-training is critical for closing the gap between next-item prediction and actual recommendation quality. Existing post-training methods are ill-suited for production-scale systems: RLHF methods reward hack due to noisy user feedbac...

---

### 17. [Quantifying Membership Disclosure Risk for Tabular Synthetic Data Using Kernel Density Estimators](https://arxiv.org/abs/2603.10937)

**Authors**: Rajdeep Pathak, Sayantee Jana  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.10937v1  

#### Abstract
The use of synthetic data has become increasingly popular as a privacy-preserving alternative to sharing real datasets, especially in sensitive domains such as healthcare, finance, and demography. However, the privacy assurances of synthetic data are not absolute, and remain susceptible to membershi...

---

### 18. [Trajectory-Informed Memory Generation for Self-Improving Agent Systems](https://arxiv.org/abs/2603.10600)

**Authors**: Gaodan Fang, Vatche Isahagian, K. R. Jayaram, Ritesh Kumar, Vinod Muthusamy, Punleuk Oum, Gegi Thomas  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.10600v1  

#### Abstract
LLM-powered agents face a persistent challenge: learning from their execution experiences to improve future performance. While agents can successfully complete many tasks, they often repeat inefficient patterns, fail to recover from similar errors, and miss opportunities to apply successful strategi...

---

### 19. [LWM-Temporal: Sparse Spatio-Temporal Attention for Wireless Channel Representation Learning](https://arxiv.org/abs/2603.10024)

**Authors**: Sadjad Alikhani, Akshay Malhotra, Shahab Hamidi-Rad, Ahmed Alkhateeb  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.10024v1  

#### Abstract
LWM-Temporal is a new member of the Large Wireless Models (LWM) family that targets the spatiotemporal nature of wireless channels. Designed as a task-agnostic foundation model, LWM-Temporal learns universal channel embeddings that capture mobility-induced evolution and are reusable across various d...

---

### 20. [Discovery of a Hematopoietic Manifold in scGPT Yields a Method for Extracting Performant Algorithms from Biological Foundation Model Internals](https://arxiv.org/abs/2603.10261)

**Authors**: Ihor Kendiukhov  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.10261v1  

#### Abstract
We report the discovery and extraction of a compact hematopoietic algorithm from the single-cell foundation model scGPT, to our knowledge the first biologically useful, competitive algorithm extracted from a foundation model via mechanistic interpretability. We show that scGPT internally encodes a c...

---

### 21. [Cross-Species Transfer Learning for Electrophysiology-to-Transcriptomics Mapping in Cortical GABAergic Interneurons](https://arxiv.org/abs/2603.11000)

**Authors**: Theo Schwider, Ramin Ramezani  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.11000v1  

#### Abstract
Single-cell electrophysiological recordings provide a powerful window into neuronal functional diversity and offer an interpretable route for linking intrinsic physiology to transcriptomic identity. Here, we replicate and extend the electrophysiology-to-transcriptomics framework introduced by Gouwen...

---

### 22. [GhazalBench: Usage-Grounded Evaluation of LLMs on Persian Ghazals](https://arxiv.org/abs/2603.09979)

**Authors**: Ghazal Kalhor, Yadollah Yaghoobzadeh  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.09979v1  

#### Abstract
Persian poetry plays an active role in Iranian cultural practice, where verses by canonical poets such as Hafez are frequently quoted, paraphrased, or completed from partial cues. Supporting such interactions requires language models to engage not only with poetic meaning but also with culturally en...

---

### 23. [OpenClaw-RL: Train Any Agent Simply by Talking](https://arxiv.org/abs/2603.10165)

**Authors**: Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, Ling Yang  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.10165v1  

#### Abstract
Every agent interaction generates a next-state signal, namely the user reply, tool output, terminal or GUI state change that follows each action, yet no existing agentic RL system recovers it as a live, online learning source. We present OpenClaw-RL, a framework built on a simple observation: next-s...

---

### 24. [From Images to Words: Efficient Cross-Modal Knowledge Distillation to Language Models from Black-box Teachers](https://arxiv.org/abs/2603.10877)

**Authors**: Ayan Sengupta, Shantanu Dixit, Md Shad Akhtar, Tanmoy Chakraborty  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.10877v1  

#### Abstract
Knowledge distillation (KD) methods are pivotal in compressing large pre-trained language models into smaller models, ensuring computational efficiency without significantly dropping performance. Traditional KD techniques assume homogeneity in modalities between the teacher (source) and the student ...

---

### 25. [Graph-GRPO: Training Graph Flow Models with Reinforcement Learning](https://arxiv.org/abs/2603.10395)

**Authors**: Baoheng Zhu, Deyu Bo, Delvin Ce Zhang, Xiao Wang  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.10395v1  

#### Abstract
Graph generation is a fundamental task with broad applications, such as drug discovery. Recently, discrete flow matching-based graph generation, \aka, graph flow model (GFM), has emerged due to its superior performance and flexible sampling. However, effectively aligning GFMs with complex human pref...

---

### 26. [Implicit Statistical Inference in Transformers: Approximating Likelihood-Ratio Tests In-Context](https://arxiv.org/abs/2603.10573)

**Authors**: Faris Chaudhry, Siddhant Gadkari  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.10573v1  

#### Abstract
In-context learning (ICL) allows Transformers to adapt to novel tasks without weight updates, yet the underlying algorithms remain poorly understood. We adopt a statistical decision-theoretic perspective by investigating simple binary hypothesis testing, where the optimal policy is determined by the...

---

### 27. [HAPEns: Hardware-Aware Post-Hoc Ensembling for Tabular Data](https://arxiv.org/abs/2603.10582)

**Authors**: Jannis Maier, Lennart Purucker  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.10582v1  

#### Abstract
Ensembling is commonly used in machine learning on tabular data to boost predictive performance and robustness, but larger ensembles often lead to increased hardware demand. We introduce HAPEns, a post-hoc ensembling method that explicitly balances accuracy against hardware efficiency. Inspired by m...

---

### 28. [Prioritizing Gradient Sign Over Modulus: An Importance-Aware Framework for Wireless Federated Learning](https://arxiv.org/abs/2603.10763)

**Authors**: Yiyang Yue, Jiacheng Yao, Wei Xu, Zhaohui Yang, George K. Karagiannidis, Dusit Niyato  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.10763v1  

#### Abstract
Wireless federated learning (FL) facilitates collaborative training of artificial intelligence (AI) models to support ubiquitous intelligent applications at the wireless edge. However, the inherent constraints of limited wireless resources inevitably lead to unreliable communication, which poses a s...

---

### 29. [LAtte: Hyperbolic Lorentz Attention for Cross-Subject EEG Classification](https://arxiv.org/abs/2603.10881)

**Authors**: Johannes Burchert, Ahmad Bdeir, Tom Hanika, Lars Schmidt-Thieme, Niels Landwehr  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.10881v1  

#### Abstract
Electroencephalogram (EEG) classification is critical for applications ranging from medical diagnostics to brain-computer interfaces, yet it remains challenging due to the inherently low signal-to-noise ratio (SNR) and high inter-subject variability. To address these issues, we propose LAtte, a nove...

---

### 30. [Federated Learning-driven Beam Management in LEO 6G Non-Terrestrial Networks](https://arxiv.org/abs/2603.10983)

**Authors**: Maria Lamprini Bartsioka, Ioannis A. Bartsiokas, Athanasios D. Panagopoulos, Dimitra I. Kaklamani, Iakovos S. Venieris  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.10983v1  

#### Abstract
Low Earth Orbit (LEO) Non-Terrestrial Networks (NTNs) require efficient beam management under dynamic propagation conditions. This work investigates Federated Learning (FL)-based beam selection in LEO satellite constellations, where orbital planes operate as distributed learners through the utilizat...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-23 06:49:23 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [TempoNet: Slack-Quantized Transformer-Guided Reinforcement Scheduler for Adaptive Deadline-Centric Real-Time Dispatchs](https://arxiv.org/abs/2602.18109)

**Authors**: Rong Fu, Yibo Meng, Guangzhen Yao, Jiaxuan Lu, Zeyu Zhang, Zhaolu Kang, Ziming Guo, Jia Yee Tan, Xiaojing Du, Simon James Fong  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2602.18109v1  

#### Abstract
Real-time schedulers must reason about tight deadlines under strict compute budgets. We present TempoNet, a reinforcement learning scheduler that pairs a permutation-invariant Transformer with a deep Q-approximation. An Urgency Tokenizer discretizes temporal slack into learnable embeddings, stabiliz...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
实时调度系统在高动态负载、不确定执行时间以及多核环境下，传统基于规则的调度器（如 EDF、RM）难以维持高**deadline compliance**（截止期满足率）。尤其是在过载（overload）场景下，这些经典算法性能急剧下降。同时，现有的深度强化学习（DRL）调度方法存在以下问题：
- 依赖序列输入顺序，缺乏对任务集合的**排列不变性**（permutation invariance）；
- 使用全注意力机制导致计算复杂度为 $O(N^2)$，难以满足亚毫秒级推理延迟要求；
- 对时间紧迫性的建模不够精细，影响优化稳定性。

TempoNet 针对上述挑战，提出了一种适用于**硬实时环境**的低延迟、高性能神经调度框架。

---

### **提出了什么新方法或新思路**
TempoNet 是一个基于 **value-based RL** 的实时调度器，其核心设计融合了三项关键技术：

#### ✅ **1. Urgency Tokenizer（紧迫性分词器）**
- 将连续的**slack**（$s_i(t) = d_i^{(k)} - t - c_i(t)$）离散化为可学习的嵌入向量。
- 使用 `clip(floor(slack / Δ), 0, Q-1)` 进行量化，并通过查找表生成 `urgency token`。
- **优势**：稳定梯度更新，增强 deadline proximity 的表示能力，提升训练稳定性。

#### ✅ **2. 轻量级稀疏注意力 Transformer 编码器**
- 构建**排列不变**的 Transformer 模型处理无序任务集。
- 引入两种稀疏策略实现近线性扩展：
  - **Blockwise Top-k Selection**：每个 block 内只保留 top-k 注意力得分；
  - **Locality-Sensitive Chunking**：按 deadline 排序后分块，减少跨块交互。
- 实现 **sub-millisecond inference** 和 $O(N^{1.1})$ 复杂度。

#### ✅ **3. 多核映射层（Multicore Mapping Layer）**
- 将上下文化后的 Q-scores 映射到多核分配决策：
  - **Masked-Greedy Selection**：迭代选择最高 Q 值任务并屏蔽已选任务；
  - 或使用**可微匹配**（differentiable matching）进行联合优化。
- 支持在严格延迟约束下完成高效资源分配。

---

### **相比现有方法的优势**
| 维度 | TempoNet | 传统方法（EDF/RM） | 其他 DRL/GNN 方法 |
|------|---------|------------------|------------------|
| **deadline compliance** | ✔️ 显著更高 | ❌ 在过载时崩溃 | ⚠️ 较好但不稳定 |
| **推理延迟** | ✔️ <1ms（支持 600 任务） | ✔️ 极低 | ❌ 通常 >1ms |
| **泛化性** | ✔️ 排列不变，适应任意任务顺序 | ✔️ | ✔️（部分） |
| **可解释性** | ✔️ 注意力权重与 deadline 高相关（r=0.98） | ❌ 规则固定 | ⚠️ 黑箱程度高 |
| **样本效率** | ✔️ 支持行为克隆预训练加速收敛 | — | ⚠️ 通常需要大量在线探索 |

---

## 2. 核心实验方法和设置

### **使用的数据集**
- **合成任务集**：周期性任务配置（短/中/长周期），用于验证基础性能。
- **工业混合关键性轨迹**（industrial mixed-criticality traces）：来自真实边缘/制造系统的负载，包含不同优先级任务。
- **大规模多处理器负载**：最多达 **600 个并发任务**，部署于 8~32 核平台。

---

### **实验设置和评估指标**

#### 📊 **主要评估指标**
| 指标 | 定义 |
|------|------|
| **Deadline Compliance Rate** | 成功按时完成的任务占比 |
| **Average Response Time (ART)** | 任务从释放到完成的时间均值 |
| **PITMD**（Percentage of Important Tasks Meeting Deadlines） | 关键任务的截止期满足率 |
| **Success Rate** | 单次运行中所有关键任务都满足 deadline 的比例 |
| **Inference Time / End-to-End Latency** | 调度决策耗时（目标：<1ms） |
| **Complexity Scaling** | 时间随任务数量增长的趋势 |

#### 🔧 **硬件与实现细节**
- 使用 **NVIDIA V100 / Tegra Orin Nano / ARM Cortex-A78** 进行 micro-benchmark。
- 批大小为 1，warm cache 条件下测量端到端延迟。
- 模型参数：embedding dim=128，depth=2，heads=4。

---

### **基线方法对比**
| 类别 | 基线方法 |
|------|--------|
| **经典调度器** | RM, EDF, FCFS |
| **启发式/进化算法** | PSO, HQIGA, Mo-QIGA |
| **DRL 方法** | FF-DQN, Rainbow DQN, PPO, A3C, TD3 |
| **图神经网络** | GNN-based RL, GraSP-RL |
| **其他 Transformer 方法** | Transformer-based DRL, Decision Transformer |
| **量子优化** | DIOS, MHQISSO |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **统一性能优势（见 Table 1 & Table 5）**
| 方法 | Deadline Compliance (%) | ART (ms) | PITMD (%) | Inference (ms) |
|------|--------------------------|----------|-----------|---------------|
| **TempoNet (Ours)** | **87.0** | **12.43** | **89.15** | **0.42** |
| GNN-based [26] | 81.0 | 13.20 | 88.80 | 0.43 |
| Transformer-based DRL [11] | 83.0 | 13.50 | 88.20 | 0.46 |
| Rainbow DQN | 78.0 | — | — | — |
| EDF | 11.67 | 20.68 | 20.81 | ~0.01 |

> 💡 **结论**：TempoNet 在 deadline compliance 上比最强基线高出 **4–7 个百分点**，且响应时间降低 **25.7%**。

---

#### ✅ **大规模调度效率（Table 4）**
| 方法 | 任务数 | Success Rate | 调度时间 (s) |
|------|-------|--------------|----------------|
| **TempoNet** | 100 | **98.2%** | **3.4** |
| GNN-based | 100 | 98.0% | 4.0 |
| **TempoNet** | **600** | **90.1%** | **38.7** |
| GNN-based | 600 | 89.5% | 40.0 |
| MHQISSO (EDF) | 600 | 87.5% | 317.1 |

> ⏱️ **速度提升**：在 600 任务下，TempoNet 比 MHQISSO 快 **8.2×**，比 DIOS 快 **5.1×**。

---

#### ✅ **复杂度分析**
- **理论复杂度**：$O(N^{1.1})$
- **实测扩展性**：Figure 4 显示随任务规模增长呈近线性趋势，远优于 DIOS ($O(N^{1.8})$) 和 MHQISSO ($O(N^{2.2})$)

---

### **消融实验结果**

#### 🔍 **Ablation Study on Key Components**

##### （1）Urgency Tokenizer 消融（Table 7）
| 变体 | 输入形式 | Hit Rate (%) | 相对于 TempoNet 差距 |
|------|--------|-------------|--------------------|
| FF-DQN-cont | 原始 slack | 74.8 | -12.2 pp |
| FF-DQN-norm | z-score slack | 76.1 | -10.9 pp |
| TempoNet w/o UT | 连续拼接 slack | 81.3 | -5.7 pp |
| **TempoNet (full)** | 量化 + embedding | **87.0** | — |

> ✅ **结论**：离散化 + 学习嵌入带来显著增益，且训练方差更低（$ \sigma^2=1.7\times10^{-3} $ vs. >2.4×）。

---

##### （2）注意力头与模型深度（Table 2 & 3）
| 层数 | Hit Rate | 延迟 |
|-----|---------|------|
| 1 | 76.2% | 0.42ms |
| **2** | **85.0%** | **0.51ms** |
| 3 | 86.1% | 0.71ms |
| 4 | 85.7% | 0.94ms |

| 注意力头数 | Hit Rate | 增益 |
|----------|--------|------|
| **4** | **85.0%** | 基准 |
| 2 | 80.3% | -5.5% |
| 6 | 84.7% | -0.3% |

> ✅ 最优配置：**2 层 + 4 头**，在精度与延迟间取得最佳平衡。

---

##### （3）嵌入维度影响（Figure 7）
- **d=128** 时达到最优性能-成本权衡；
- 更大维度带来边际收益递减（diminishing returns）。

---

##### （4）奖励函数与探索策略（Table 12 & Figure 11）
- **Slack-sensitive reward** 实现最低尾部延迟（95th lateness: 11.7ms）；
- **Uncertainty-based exploration** 比标准 ε-greedy 收敛快 15%，最终性能略优。

---

## 4. 关键结论和发现

### **主要发现**
1. **Slack 量化是关键**：将连续 slack 离散化为 learnable tokens 不仅提升了 deadline compliance，还显著增强了训练稳定性（Theorem A.1 & C.4 提供理论支持）。
2. **稀疏注意力可行且必要**：通过 block Top-k 和 locality-aware chunking，可在保持全局推理能力的同时实现 $O(N^{1.1})$ 复杂度，满足硬实时需求。
3. **注意力具有可解释性**：Attention weights 与任务 deadline 高度相关（r=0.98），Top-1 alignment 达 90% 以上，说明模型真正学会了“关注紧急任务”。
4. **策略可被简化规则逼近**：91% 的调度决策可通过线性组合 `min-slack + SRPT` 规则复现，表明学习到的是合理、透明的调度逻辑。
5. **具备强鲁棒性与迁移能力**：
   - 在非平稳负载下通过 few-shot adaptation 快速恢复；
   - 单一模型可在 4/8/16/32 核之间零样本迁移（zero-shot transfer），性能损失小。

---

### **方法的局限性**
1. **极端过载下性能下降**：当 utilization > 1.2 且短任务比例 > 40% 时，compliance 下降约 18 个百分点（Table 19）。
2. **稀疏内核退化风险**：突发负载可能导致 Top-k 选择退化为密集计算，引发 tick overrun（<2% 概率）。
3. **依赖精确 slack 估计**：若任务执行时间预测误差大，会影响 tokenizer 效果。
4. **当前仅支持同构多核**：未考虑异构架构（CPU+GPU+NPU）下的调度。

---

### **未来工作方向**
1. **扩展至异构硬件平台**：结合 DVFS、电源管理等维度进行多目标优化。
2. **引入分布式注意力机制**：支持跨节点集群调度，构建 scalable real-time orchestration 框架。
3. **集成安全机制**：加入 admission control 或 fallback 到 EDF 的机制以应对持续过载。
4. **进一步压缩模型**：探索二值化、蒸馏等技术，在嵌入式设备上部署。
5. **支持 offline RL + online fine-tuning pipeline**：利用历史日志预训练，提升实际部署效率。

---

> ✅ **总体评价**：  
> TempoNet 成功地将 **Transformer 的强大建模能力** 与 **硬实时系统的低延迟要求** 结合起来，提出了一套**实用、高效、可解释**的神经调度框架。它不仅在性能上超越了经典调度器和其他 DRL 方法，还在推理速度、稳定性、可迁移性和可解释性方面树立了新的标杆，为 AI-driven real-time systems 的落地提供了坚实基础。

</details>

---

### 2. [SeedFlood: A Step Toward Scalable Decentralized Training of LLMs](https://arxiv.org/abs/2602.18181)

**Authors**: Jihun Kim, Namhoon Lee  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.18181v1  

#### Abstract
This work presents a new approach to decentralized training-SeedFlood-designed to scale for large models across complex network topologies and achieve global consensus with minimal communication overhead. Traditional gossip-based methods suffer from message communication costs that grow with model s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SeedFlood: A Step Toward Scalable Decentralized Training of LLMs — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **decentralized training** 面临两大可扩展性瓶颈：
- **通信开销大**：标准的 **gossip-based averaging** 方法需要频繁传输高维模型参数或梯度，通信成本随模型维度 $d$ 线性增长，对 **billion-parameter LLMs** 不现实。
- **共识效率低**：在稀疏或大规模网络拓扑中，信息通过多跳传播会衰减，导致远端节点难以达成全局一致（global consensus），影响模型收敛。

此外，现有基于 **zeroth-order optimization (ZO)** 的去中心化方法虽然利用共享随机性（shared randomness）将更新压缩为 **seed-scalar pair**，但仍沿用 gossip 范式，导致计算负担转移到本地更新聚合上。

---

### 🚀 提出的新方法：SeedFlood

**SeedFlood** 是一种全新的去中心化训练框架，其核心思想是：
> **用 flooding 替代 gossip，实现全局广播式的零阶更新分发。**

#### 创新机制：
1. **Seed-Reconstructible Updates**  
   所有客户端共享一个同步的随机数生成器（RNG）。每个零阶更新仅需发送一个 **random seed** 和对应的 **scalar directional derivative**，接收方可从 seed 重建完整的扰动向量。  
   → 通信负载与模型大小无关，仅为常数级（~bytes per update）。

2. **Flooding-based Dissemination**  
   每个客户端将其生成的更新以 **flooding** 方式在整个网络中传播：收到新消息后立即转发给所有邻居，直到覆盖整个网络直径。  
   → 实现每轮迭代的 **all-gather-equivalent consensus**，无需多轮 gossip 迭代。

3. **Subspace Canonical-basis Gradient Estimation (SubCGE)**  
   为解决 flooding 导致每个客户端需处理大量更新带来的计算压力，提出 **SubCGE**：
   - 将扰动限制在一个全局同步的低秩子空间（low-rank subspace）
   - 扰动方向限定为该子空间的标准基坐标
   - 多个更新可在子空间内高效聚合为矩阵运算，避免逐个重构

---

### 🔍 相比现有方法的优势

| 维度 | 传统 Gossip | Gossip + Shared Randomness | SeedFlood (本文) |
|------|------------|----------------------------|------------------|
| 通信开销 | $O(d)$ | $O(tn)$（历史累积） | $O(n)$（每轮新增） |
| 共识质量 | 受拓扑影响，存在延迟与误差 | 同左 | 完美共识，拓扑不变 |
| 计算开销 | $O(d)$ | $O(tnd)$（重复重构） | $O(n + rd)$（批量聚合） |
| 可扩展性 | 差（受限于带宽） | 中等（通信轻但计算重） | 极强（通信几乎为零，计算可控） |

✅ **突破性优势**：
- 通信成本降至 **400KB** 级别（vs. 数百 GB 的 DSGD），且不随模型增大而增加
- 在稀疏环形拓扑（ring topology）下仍保持高性能，显著优于 gossip 类方法
- 支持 **128 客户端、1B 参数模型** 的分布式微调，达到甚至超越部分 first-order 基线

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **任务**：SuperGLUE 子集 + SST-2
  - 包括：BoolQ, RTE, MultiRC, ReCoRD, WiC, SST-2
- **模型**：
  - OPT-125M, OPT-1.3B, OPT-2.7B
- **数据划分**：
  - 总共 1,024 条训练样本均匀分布到各客户端
  - 验证集（500）、测试集（1,000）全局共享

### ⚙️ 实验设置
- **网络拓扑**：
  - Ring（环状，稀疏）
  - Meshgrid（网格，密集）
- **客户端数量**：16 ~ 128
- **训练步数**：
  - Zeroth-order methods: 5,000 步
  - First-order methods: 500 步（因 FO 更高效）
- **通信频率**：每 5 个本地更新进行一次通信
- **flooding 步数**：等于网络直径（或作为超参调节）

### 📊 评估指标
- **Global Model Performance (GMP)**：所有客户端模型平均后的最终性能
- **Total Communication Cost**：整个训练过程中每条边传输的总字节数
- **Relative Performance (%)**：相对于 DSGD@16client 的归一化得分

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| First-order | DSGD, ChocoSGD（带 Top-K 压缩） |
| First-order + LoRA | DSGD-LoRA, ChocoSGD-LoRA |
| Zeroth-order | DZSGD, DZSGD-LoRA |
| 本文方法 | **SeedFlood**（含 SubCGE） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）通信成本对比（图1, 图3）
| 方法 | 通信总量（每边） |
|------|----------------|
| DSGD | 526 GB |
| ChocoSGD | 15.79 GB |
| DSGD-LoRA | 629.1 MB |
| ChocoSGD-LoRA | 18.8 MB |
| **SeedFlood** | **400 KB** |

➡️ **SeedFlood 比最强压缩基线（ChocoSGD-LoRA）节省约 47,000× 通信量！**

#### （2）任务性能表现（表8）
在 16-client ring 网络上，SeedFlood 平均性能仅比 DSGD 低 **~4–6%**，但远优于其他 ZO 方法：
- 超越 DZSGD (-6.46%) 和 DZSGD-LoRA (-8.25%)
- 显著优于 ChocoSGD-LoRA (-8.96%)

#### （3）大规模扩展能力（表2, 图4）
当扩展至 **128 客户端** 时：
- 所有 gossip 基线性能严重下降
- **SeedFlood 反而提升性能，在 Ring 拓扑下达到 100.24%（相对 DSGD@16）**
- 归因于更多客户端带来更丰富的零阶扰动，降低估计方差

#### （4）SubCGE 效率验证（图5, 表4）
- 应用数千个 ZO 更新时，SubCGE 比原始 MeZO 快 **几个数量级**
- 单次迭代总耗时从 **2509ms (MeZO)** 降至 **942ms (SubCGE)**
- 消息应用阶段（Message Apply）仅需 **28ms**，几乎可忽略

#### （5）消融实验：Delayed Flooding（图7）
- 即使只执行 **k=4~8 次 flooding 步骤**（非完整传播），性能无明显下降
- 当 k=1 或 2 时性能下降，说明适度延迟可接受，但过度 stale 会影响效果
- 表明 SeedFlood 对实际网络中的异步和延迟具有鲁棒性

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **通信不再是去中心化训练的瓶颈**  
   利用 seed-reconstructible updates + flooding，实现了 **model-size-independent communication**，使 billion-parameter LLM 的去中心化训练成为可能。

2. **flooding 比 gossip 更适合 ZO 场景**  
   gossip 的渐进扩散机制在 ZO 下引发高昂的重复重构代价；而 flooding 的一次性广播天然契合“一次应用”语义。

3. **SubCGE 是规模化关键**  
   若无高效的聚合机制，即使通信免费，计算也会成为新瓶颈。SubCGE 通过低秩子空间设计，将聚合复杂度解耦为 $O(n + rd)$，支持大规模部署。

4. **SeedFlood 在大网络中反而更强**  
   在 128-client 设置下，SeedFlood 不仅稳定，还 **反超 first-order 方法**，揭示了 ZO 在高度分布式场景下的潜力。

---

### ⚠️ 局限性
- **依赖同步 RNG**：要求所有客户端能从相同 seed 生成完全一致的随机序列，对系统一致性要求高。
- **零阶优化固有缺陷**：相比 first-order，ZO 本身样本效率较低，需要更多迭代，可能增加总体时间成本。
- **子空间设计敏感**：rank 过小或刷新周期过短会导致性能下降（见图6），需合理配置 hyperparameters。

---

### 🔮 未来工作方向
- 将 SeedFlood 扩展至 **asynchronous setting** 和动态拓扑
- 探索更智能的 **partial flooding** 策略（如基于重要性的选择性传播）
- 结合 **quantization / sparsification** 进一步压缩 seed metadata
- 应用于 **federated learning** 或 **edge AI** 场景，推动绿色、低带宽的大模型训练

---

> 💡 **一句话总结**：  
> **SeedFlood 通过“种子广播 + 子空间聚合”的范式转变，打破了去中心化训练中通信与拓扑的双重壁垒，为超大规模 LLM 在资源受限环境下的协同训练开辟了新路径。**

</details>

---

### 3. [SPQ: An Ensemble Technique for Large Language Model Compression](https://arxiv.org/abs/2602.18420)

**Authors**: Jiamin Yao, Eren Gultepe  
**Category**: cs.CL  
**Published**: 2026-02-23  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.18420v1  

#### Abstract
This study presents an ensemble technique, SPQ (SVD-Pruning-Quantization), for large language model (LLM) compression that combines variance-retained singular value decomposition (SVD), activation-based pruning, and post-training linear quantization. Each component targets a different source of inef...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SPQ: An Ensemble Technique for Large Language Model Compression》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLM）在自然语言理解和生成方面表现出色，但其庞大的参数量导致**高内存占用和计算开销**，限制了在资源受限设备上的部署。现有的单一压缩技术（如 SVD、Pruning、Quantization）在高压缩率下往往会导致显著的性能下降。

本文旨在解决如何在**大幅降低模型内存的同时保持甚至提升模型性能**这一挑战，提出一种更高效、鲁棒的组合式压缩框架。

---

### 🚀 提出的新方法：SPQ（SVD-Pruning-Quantization）
SPQ 是一个**模块化、层感知（layer-aware）的集成压缩框架**，结合三种互补技术：
- **SVD（Singular Value Decomposition）**：应用于 Attention 层，利用低秩近似压缩投影矩阵。
- **Pruning（结构化剪枝）**：应用于 MLP 层，基于激活统计移除冗余神经元。
- **Quantization（8-bit 线性量化）**：应用于所有线性层，统一进行后训练对称量化。

该方法的核心思想是“**哪里有效用哪里**”——将最适合的技术应用到最合适的网络层中。

---

### 🔍 相比现有方法的优势
| 对比维度 | SPQ 的优势 |
|--------|-----------|
| **层感知设计** | 区别于通用压缩，SPQ 针对不同层结构选择最优策略（Attention → SVD，MLP → Pruning），实现更精准高效的压缩。 |
| **无需复杂调参** | SVD 使用方差保留阈值自动确定秩；Pruning 使用激活均值 + log-scale 映射剪枝比例，避免复杂的梯度或重要性估计。 |
| **兼容性强** | 所有操作独立且可插拔，支持 LoRA 微调恢复性能，并兼容多种量化模式（per-tensor / per-channel / hybrid）。 |
| **端到端效率更高** | 尽管包含多个步骤，但由于无需迭代优化或大量校准数据，SPQ 的压缩速度比 GPTQ 快 20%。 |

> 💡 SPQ 是首个系统性地将 SVD、Pruning 和 Quantization 在 LLM 上以层感知方式融合的框架。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **语言建模评估**：
  - `WikiText-2`：用于评估 perplexity（越低越好）
  - `C4`：大规模预训练语料子集，也用于 perplexity 测试
- **下游任务基准**（准确率或 BLEU）：
  - 推理能力：`OpenBookQA`, `ARC`, `WinoGrande`, `HellaSwag`, `PIQA`
  - 真实性判断：`TruthfulQA-1`, `TruthfulQA-2`
  - 数学推理：`GSM8K`

此外还使用了一个小规模 calibration dataset 来估计激活统计量（用于 pruning 和量化敏感度分析）。

---

### ⚙️ 实验设置与评估指标

| 类别 | 设置说明 |
|------|---------|
| **硬件平台** | 2 × NVIDIA A100-40GB GPU |
| **测试模型族** | LLaMA（1B–7B）、OPT（1.3B–6.7B）、Vicuna-7B、Mistral-7B |
| **主实验模型** | `LLaMA-2-7B`（作为主要对比基准） |
| **评估指标** | 
| - `Weight Memory (GB)` | 参数存储大小（不含激活） |
| - `Perplexity ↓` | 语言建模质量（越低越好） |
| - `Throughput (tokens/sec) ↑` | 推理吞吐量（越高越好） |
| - `Accuracy (%) ↑` | 下游任务表现 |

---

### 🆚 基线方法对比
| 方法 | 技术特点 | 是否使用 |
|------|----------|----------|
| **ASVD** | 激活感知 SVD，仅低秩分解 | ✔️ |
| **SparseGPT** | 结构化稀疏剪枝 | ✔️ |
| **GPTQ** | 后训练 4-bit / 8-bit 量化 | ✔️（最强 baseline） |
| **SVD-only / Pruning-only / Quantization-only** | 单一压缩方法 | ✔️（消融研究） |
| **QLoRA**, **AWQ**, **SVD-LLM v2** | 多技术组合方法 | 引用比较 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 LLaMA-2-7B 为例）

| 方法 | 压缩比 | 内存 (GB) | WikiText-2 Perplexity | Throughput (vs. GPTQ) |
|------|--------|------------|------------------------|------------------------|
| Original (FP32) | 0% | 26.95 | 5.47 | — |
| ASVD (21%) | 21% | 21.41 | 6.54 | — |
| SparseGPT (50%) | 50% | 13.40 | 7.76 | — |
| GPTQ (int8) | 73% | 7.16 | 5.48 | 1.0× |
| **SPQ (Ours)** | **75%** | **6.86** | **4.91** | **1.3× (vs GPTQ-8bit), up to 1.9× (vs GPTQ-4bit)** |

> ✅ SPQ 实现了 **最高压缩率（75%）+ 最低内存（6.86GB）+ 最佳 perplexity（4.91）**

---

### 🔁 与基线方法的对比结果
- **相比 GPTQ**：
  - 内存减少 **2% 更少**（6.86 vs 7.16 GB）
  - Perplexity 更优（4.91 vs 5.48 on WikiText-2）
  - 推理速度 **快 1.3–1.9 倍**
- **相比 SparseGPT 和 ASVD**：
  - 压缩率高出 **25–54%**
  - 在相同或更低压缩下，accuracy 更稳定，尤其在 `TruthfulQA` 和 `GSM8K` 上表现接近原始模型
- **跨模型泛化性好**：
  - 在 LLaMA、OPT、Vicuna、Mistral 等系列上均实现 **62%-74% 内存缩减**
  - 大模型（如 7B 级）反而在压缩后 perplexity 下降（即性能提升）

---

### 🔍 消融实验结果（Ablation Study）

#### （1）单个组件效果（Figure 7）
- **SVD-only**：超过 15% 压缩后 perplexity 急剧上升
- **Pruning-only**：40% 压缩后开始退化
- **Quantization-only (8-bit)**：稳定，但 4-bit（≈85% 压缩）时崩溃
- **SPQ**：即使压缩 >80%，perplexity 仍低于 15，显示极强鲁棒性

#### （2）两两组合效果（Table 2 & Figure 6）
| 组合 | 内存变化 | Perplexity 变化 | 结论 |
|------|----------|------------------|------|
| SVD + Quant | ↓ 显著 (p<.001) | ≈ 不变 | 有效互补 |
| Pruning + Quant | ↓ 显著 (p<.001) | ≈ 不变 | 可叠加压缩 |
| **SVD + Pruning + Quant (SPQ)** | ↓ 最大 | ↓ 或 ≈ | 三者协同增效 |

> ✅ 证明三种技术可以**无损叠加**，且组合优于任何单一或双技术方案。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **异构压缩技术可以协同工作**：SVD、Pruning、Quantization 分别针对不同结构缺陷，联合使用能实现“1+1+1 > 3”的压缩效益。
2. **层感知分配至关重要**：Attention 层适合低秩分解（SVD），MLP 层适合剪枝，全连接层均可量化，这种结构对齐极大提升了压缩效率。
3. **轻量级配置即可获得高性能**：无需复杂重要性评分或迭代优化，简单的激活统计 + 方差保留即可实现优越结果。
4. **大模型更能从 SPQ 中受益**：参数越多，内在冗余越大，SPQ 能更好地释放潜力，在部分情况下甚至**提升模型性能**（如 LLaMA-2-7B 的 perplexity 从 5.47→4.91）。

---

### ⚠️ 方法的局限性
1. **依赖 LoRA 微调恢复性能**：虽然只用了 200 步，但仍需额外训练阶段，不完全属于纯 PTQ（Post-Training Quantization）流程。
2. **当前未探索激活量化（activation quantization）**：仅压缩权重，未处理推理过程中的激活内存瓶颈。
3. **超参数耦合影响最终效果**：尽管各模块独立，但 SVD 阈值、prune ratio、quant mode 的组合需要经验调节。
4. **尚未适配 MoE 架构**：目前主要验证在 dense LLM 上，对 Mixtral 等稀疏专家模型的支持待验证。

---

### 🔮 未来工作方向
1. **扩展至混合精度量化**：引入 4-bit 或 block-wise 量化进一步压缩。
2. **加入 activation quantization**：全面降低运行时内存需求。
3. **探索其他矩阵分解方法**：如 Tucker 分解、QR 分解等替代 SVD。
4. **自动化配置搜索**：构建 NAS-like 框架自动寻找最优压缩策略组合。
5. **面向边缘设备优化**：结合硬件特性（如 Tensor Core、INT4 支持）定制压缩方案。

---

## 📌 总结一句话
> **SPQ 通过层感知、模块化的方式融合 SVD、Pruning 与 Quantization，在高达 75% 压缩率下不仅显著降低内存占用（最低至 6.86GB），还保持甚至提升模型性能，并实现比 GPTQ 更快的推理速度，为 LLM 在边缘场景的高效部署提供了强有力的新范式。**

🔗 代码已开源：[https://github.com/JiaminYao/SPQ_LLM_Compression/](https://github.com/JiaminYao/SPQ_LLM_Compression/)

</details>

---

### 4. [Dual Length Codes for Lossless Compression of BFloat16](https://arxiv.org/abs/2602.17849)

**Authors**: Aditya Agrawal, Albert Magyar, Hiteshwar Eswaraiah, Patrick Sheridan, Pradeep Janedula, Ravi Krishnan Venkatesan, Krishna Nair, Ravi Iyer  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.17849v1  

#### Abstract
Training and serving Large Language Models (LLMs) relies heavily on parallelization and collective operations, which are frequently bottlenecked by network bandwidth. Lossless compression using e.g., Huffman codes can alleviate the issue, however, Huffman codes suffer from slow, bit-sequential decod...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Dual Length Codes for Lossless Compression of BFloat16

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模语言模型（LLMs）的训练和服务过程中，**网络带宽**常成为集体操作（如 AllReduce、AllGather）的瓶颈。虽然 **Huffman codes** 能实现高效的无损压缩，但由于其**变长编码**特性，解码过程需要进行深度的二叉树遍历，导致**解码速度慢且硬件复杂度高**。而通用编码（如 Exponential-Golomb）虽解码较快，却**无法利用符号频率分布**，压缩效率较低。

本文旨在解决这一矛盾：如何在保持较高压缩率的同时，显著提升解码速度并降低硬件实现复杂度。

### 提出的新方法
提出了一种名为 **Dual Length Codes（双长度编码）** 的新型无损压缩方案，专为 BFloat16 张量设计。其核心思想是：
- 将 256 个可能的 8-bit 符号划分为两个区域：
  - **高频区（Top-8 Symbols）**：出现概率最高的 8 个符号，分配 **4-bit 短码**。
  - **低频区（Remaining 248 Symbols）**：其余符号，分配 **9-bit 长码**。
- 使用 **1 位前缀比特（prefix bit）** 区分两种码长：
  - `0` 表示短码（后续 3 位索引 LUT）
  - `1` 表示长码（后续 8 位为原始符号值）

### 相比现有方法的优势
| 特性 | Huffman Codes | Universal Codes | **Dual Length Codes** |
|------|----------------|------------------|------------------------|
| 利用频率分布 | ✅ 最优熵编码 | ❌ 不利用 | ✅ 显著利用（聚焦 Top-8） |
| 解码速度 | ❌ 比特串行，树深可达 10 层 | ✅ 非完全串行 | ✅ 极快（仅需查表或直接读取） |
| 硬件复杂度 | ❌ 高（需复杂树遍历逻辑） | ✅ 较低 | ✅ 极低（仅需 8-entry LUT） |
| 编码/解码实现 | 复杂 | 中等 | **简单、可预测延迟** |

> **创新点总结**：通过“**两档定长 + 前缀选择**”的设计，在压缩效率、解码速度与硬件开销之间取得了良好平衡，特别适合 ML 系统中对延迟敏感的通信场景。

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **Gemma 2B 模型** 在 **Supervised Fine-Tuning (SFT)** 阶段产生的 BFloat16 张量。
- 分析对象包括：
  - Feed-forward 层的激活张量（FFN1 和 FFN2）
  - 权重、梯度等（文中以 FFN1 activation 为主）
- 模型被分片到 **64 个 TPU** 上，共分析 $18 \times 64 = 1152$ 个 shard 的数据。

### 实验设置与评估指标
- **符号粒度**：将 BFloat16 数据按 byte（8-bit）切分为 256 个符号空间进行统计分析。
- **概率建模**：计算每个符号在整个训练过程中的出现频率，绘制 PMF 与 CDF。
- **压缩性能评估指标**：
  - **Compressibility（压缩率）**：$\frac{原始位宽 - 平均编码长度}{原始位宽}$
  - 对比 Huffman 编码的实际压缩效果。
- **实现可行性分析**：评估编码/解码所需的 LUT 大小与逻辑复杂度。

### 基线方法对比
- **Huffman Coding**：作为最优熵编码基准，衡量压缩极限。
- **Universal Codes（如 Exponential-Golomb）**：代表快速但非频率自适应的编码方式。

---

## 3. 主要实验结果和性能指标

### 关键观察结果
- **Top-8 符号累计概率 ≈ 50%**（见 Fig. 2），表明存在明显的“头部集中”现象。
- 这些高频符号具体为：`61, 62, 63, 64, 189, 190, 191, 192`（对应特定浮点数值，可能是零或接近零的激活值）。
- Shannon 熵约为 **6.26 bits/symbol**，理论最大压缩率为 ~21.7%。

### 压缩性能对比
| 方法 | 平均码长（bits/symbol） | Compressibility |
|------|--------------------------|-----------------|
| 原始 BFloat16（8-bit slice） | 8.0 | 0% |
| Huffman Coding | ~6.296 | **21.3%** |
| **Dual Length Codes** | **6.5** | **18.6%** |
| 理论估算（0.5×4 + 0.5×9） | 6.5 | 18.75% |

> 实际压缩率与理论高度一致，验证了模型有效性。

### 性能对比结论
- Dual Length Codes 的压缩率略低于 Huffman（相差约 **2.7%**），但仍保留了大部分压缩收益。
- 但在**解码速度与硬件成本上具有压倒性优势**：
  - 解码无需树遍历，只需判断首比特后选择路径。
  - 仅需一个 **8-entry LUT** 存储高频符号映射，极大简化 ASIC/FPGA 实现。
  - 解码延迟固定且可预测（最多 9 bits），适合流水线优化。

### 消融实验（隐含分析）
- 文中虽未明确命名“ablation study”，但通过对 Huffman code lengths 的分析（Fig. 3 显示码长从 3 到 10 不等）说明了其复杂性来源。
- 反向论证：若只保留两种码长（4 和 9），可在几乎不损失太多压缩率的前提下大幅简化系统。

---

## 4. 关键结论和发现

### 主要发现
1. **BFloat16 张量中存在显著的符号频率偏斜**，Top-8 符号占据约一半的数据量，这为轻量级频率感知压缩提供了基础。
2. **Dual Length Codes 成功捕捉了该分布特征**，通过极简机制实现了接近 Huffman 的压缩效果。
3. 该方案在 **压缩效率、解码速度、硬件友好性** 三者间实现了优良折衷，尤其适用于分布式 ML 系统中的高频通信操作。

### 方法的局限性
- 当前设计基于 **静态频率分布**，假设 Top-8 符号在不同层、不同阶段相对稳定；若分布剧烈变化，需动态更新 LUT。
- 仅适用于具有明显“头部集中”特性的数据；对于更均匀分布的数据，压缩增益会下降。
- 当前划分仅为两级（4-bit / 9-bit），进一步扩展为多级（Multi-Length Codes）可能带来更好权衡，但会增加控制逻辑。

### 未来工作方向
- 探索 **multi-length extension**（例如三级码长）以逼近 Huffman 性能同时控制复杂度。
- 将 Dual Length Codes 集成进实际的 ML runtime（如 JAX/TensorFlow），测量端到端通信延迟改善。
- 研究 **per-tensor 或 per-layer 自适应 LUT 构建策略**，支持在线学习与部署。
- 扩展至其他数据类型（如 FP8、INT8）或其他模态模型（vision transformers 等）。

---

> **总结一句话**：  
> Dual Length Codes 是一种面向 ML 系统通信优化的**实用主义压缩编码**——它放弃追求极致压缩率，转而通过洞察数据分布规律，构建出**高速、低延迟、易硬件实现**的轻量级方案，为大规模模型训练中的带宽瓶颈提供了一条高效可行的技术路径。

</details>

---

### 5. [Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](https://arxiv.org/abs/2602.18403)

**Authors**: Orfeas Bourchas, George Papalambrou  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.18403v1  

#### Abstract
Accurate prediction of main engine power is essential for vessel performance optimization, fuel efficiency, and compliance with emission regulations. Conventional machine learning approaches, such as Support Vector Machines, variants of Artificial Neural Networks (ANNs), and tree-based methods like ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统纯数据驱动的机器学习模型（如 XGBoost、ANN、PINN）在船舶主发动机功率预测中虽然在训练数据密集区域表现良好，但在**稀疏数据区或外推场景**（如高速航行、极端天气条件）下往往产生物理不一致的预测结果。特别是这些模型难以保持“**推进器定律**”（propeller law），即功率 $ P \propto V^3 $ 的基本物理关系，导致在无历史数据支持的操作条件下可靠性差。

### **提出了什么新方法或新思路**
提出了一种**科学知识引导的混合建模框架**（hybrid modeling framework），其核心思想是：
- 利用海试（sea trial）获得的平静水域功率曲线作为**物理一致的基线模型**（baseline model），形式为 $ P_{\text{sea trial}} = c V^n $，并通过吃水（draft）线性插值得到中间状态下的基准功率。
- 引入一个非线性回归器（如 XGBoost、NN 或 PINN）来学习**残差修正项** $ f(X) $，即实际观测功率与基线功率之间的偏差，该偏差由风、浪、船体污底、老化等环境与操作因素引起。
- 最终预测为：  
  $$
  P(X) = P_{\text{sea trial}}(V, T) + f(X)
  $$

这种方法将复杂的全局预测任务分解为：**已知物理规律 + 数据驱动残差修正**。

### **相比现有方法的优势**
- ✅ **增强外推稳定性**：由于基线已满足 $ P \propto V^3 $ 规律，即使在训练未见的速度/工况下，预测仍保持物理合理性。
- ✅ **提升泛化能力**：ML 模型只需学习较小幅度的残差，降低了学习难度，提高了对噪声和稀疏数据的鲁棒性。
- ✅ **兼容多种ML架构**：可与 XGBoost、标准 Neural Network、PINN 等多种模型结合，具有良好的通用性和灵活性。
- ✅ **计算高效且实用性强**：无需复杂调参即可实现稳定训练，适用于实际船舶性能监控系统。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 数据来源：作者前期研究 [Bourchas and Papalambrou, 2025] 构建的真实运营船舶数据集。
- 数据规模：约 **40,000 条记录**，覆盖五个月的运行数据。
- 输入变量（见 Table 1）：
  - 主机制动功率 $ P $（kW）
  - 船速（S.T.W.）$ V $（节）
  - 平均吃水 $ T $（m）
  - 纵倾（Trim）
  - 风真速度（WTS）与风向（WTD），并转换为 $ W_x $ 和 $ W_y $ 分量
- 数据划分：随机分为 **80% 训练集、10% 验证集、10% 测试集**

### **实验设置和评估指标**
#### **模型架构对比**
比较三种主流模型的两种配置：
- **Baseline 模型**：直接以输入特征 $ X $ 回归预测总功率 $ P $
- **Hybrid 模型**：先计算 $ P_{\text{sea trial}} $，再用 ML 模型预测残差 $ f(X) $

所涉模型包括：
- **XGBoost**
- **Neural Network (NN)**
- **Physics-Informed Neural Network (PINN)**

#### **超参数优化**
- **XGBoost**：使用 `RandomizedSearchCV` 进行 HPO，搜索学习率、深度、树数量、正则化系数等。
- **NN/PINN**：基于 PyTorch 实现，采用 Weights & Biases 的贝叶斯优化策略，调整学习率、层数、每层神经元数。
- 所有输入输出均标准化（StandardScaler）

#### **评估指标**
- **定量指标**：
  - Mean Absolute Error (**MAE**)
  - Root Mean Square Error (**RMSE**)
- **定性分析重点**：
  - 外推行为分析（extrapolation behavior）：在不同风向下（0°, 90°, 180°）、空载吃水、风速 5kn 下，从 8–17 kn 变化船速时的功率曲线趋势。

#### **硬件环境**
- CPU: AMD Ryzen 5500
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- 内存: 32GB RAM
- OS: Windows 10 Pro

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 5）**

| Model | Test MAE [kW] | Test RMSE [kW] |
|-------|----------------|----------------|
| XGBoost (Baseline) | **122.2** | **195.0** |
| XGBoost (Hybrid)   | 148.8         | 208.2         |
| NN (Baseline)      | **162.66**    | **225.10**    |
| NN (Hybrid)        | 219.32        | 284.33        |
| PINN (Baseline)    | **144.30**    | **211.89**    |
| PINN (Hybrid)      | 171.19        | 229.45        |

> 注：Baseline 在 MAE/RMSE 上略优，差异 <1% of max power (~10MW)，整体误差水平相近。

### **与基线方法的对比结果**
尽管定量误差指标上 **Baseline 表现稍好**，但**Hybrid 模型在外推能力和物理一致性方面显著更优**：

| 模型 | 外推表现 |
|------|----------|
| **XGBoost (Baseline)** | 出现非单调、平坦化功率曲线，尤其在高速段违反 $ P \propto V^3 $ 趋势 |
| **XGBoost (Hybrid)** | 功率随速度平滑上升，始终贴近 sea trial 曲线，物理合理 |
| **NN (Baseline)** | 高速过估计，对风向敏感，在稀疏区波动大 |
| **NN (Hybrid)** | 残差学习更稳定，整体趋势可控，外推更可靠 |
| **PINN (Baseline)** | 有一定物理约束，但仍可能偏离真实趋势 |
| **PINN (Hybrid)** | **最佳综合表现**：既保留 PINN 的导数约束，又通过基线锚定主导关系，外推最稳健 |

> 图 3–5 显示，在 **15–17 kn 高速区间**，Baseline 模型普遍趋于“饱和”或震荡，而 Hybrid 模型持续遵循立方增长趋势。

### **消融实验结果**
虽然文中未明确标注“ablation study”，但整个实验设计本质上是一次**结构级消融**：
- 移除物理基线 → 单纯数据驱动（Baseline）
- 加入物理基线 → 混合残差学习（Hybrid）

结果表明：
- **加入物理基线并未显著降低精度**（误差增加有限）
- **极大提升了模型在未知工况下的可信度与稳定性**
- 尤其对于 **XGBoost 和 NN**，Hybrid 改进最为明显；PINNs 本身已有一定物理约束，Hybrid 进一步强化了其优势。

---

## **4. 关键结论和发现**

### **主要发现**
1. 🔍 **误差指标不能反映模型真实可靠性**：Baseline 模型虽在 MAE/RMSE 上略优，但在外推场景下会产生**物理不合理甚至危险的预测**。
2. 📈 **Hybrid 框架有效提升外推稳定性**：通过引入物理基线，确保了 $ P \sim V^3 $ 的基本趋势，使预测在稀疏数据区依然可信。
3. ⚖️ **最佳平衡出现在 Hybrid PINN**：它结合了物理先验（sea trial + derivative loss），实现了**准确性与物理一致性的最优权衡**。
4. 💡 **残差学习简化了ML任务**：让模型专注于学习“小偏差”，而非从零重建整个功率函数，提升了训练效率与泛化能力。

### **方法的局限性**
- 🚫 **依赖高质量海试数据**：若缺乏 ballast/laden 工况下的 sea trial 曲线，则无法构建可靠基线。
- 🚫 **线性插值假设简化了吃水影响**：公式 (3) 中对吃水进行线性插值是一种近似处理，未考虑非线性流体力学效应。
- 🚫 **未建模时间动态性**：当前模型为静态映射，未显式建模船体污底随时间演化的动态过程。
- 🚫 **PINN 训练仍较难收敛**：尽管 Hybrid 缓解了部分问题，但 PINN 本身的训练稳定性仍是挑战。

### **未来工作方向**
- 🔁 探索**自适应基线更新机制**：利用长期运行数据动态校准 $ c $ 和 $ n $ 参数，补偿船体老化。
- 🧠 结合**时序模型**（如 LSTM、Transformer）建模残差的时间演化特性。
- 🌐 将该框架推广至更多船型与航线，验证其跨平台适用性。
- 🔬 研究更精细的**多维物理基线插值方法**（如基于CFD或响应面模型）替代线性插值。
- 🔧 开发嵌入式实时推理系统，用于船上 **weather routing、trim optimization、energy efficiency planning** 等应用。

---

> ✅ **总结一句话**：  
> 本论文提出的 **scientific knowledge-guided hybrid framework** 不是以牺牲精度换取物理一致性，而是通过“**物理锚定 + 残差学习**”的方式，在几乎不损失拟合能力的前提下，**大幅增强了模型在现实复杂场景中的外推可靠性**，为智能航运中的决策支持系统提供了更具工程价值的解决方案。

</details>

---

### 6. [Pimp My LLM: Leveraging Variability Modeling to Tune Inference Hyperparameters](https://arxiv.org/abs/2602.17697)

**Authors**: Nada Zine, Cl\'ement Quinton, Romain Rouvoy  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.17697v1  

#### Abstract
Large Language Models (LLMs) are being increasingly used across a wide range of tasks. However, their substantial computational demands raise concerns about the energy efficiency and sustainability of both training and inference. Inference, in particular, dominates total compute usage, making its op...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Pimp My LLM: Leveraging Variability Modeling to Tune Inference Hyperparameters*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）在推理阶段具有极高的计算开销，其配置空间庞大且复杂，涉及大量**generation hyperparameters**（如 `temperature`, `top-p`, `num_beams`, `cache` 策略等）。这些参数之间存在复杂的依赖关系和非线性交互效应，导致：
- 手动调优困难；
- 全面实证评估不可行（组合爆炸）；
- 现有研究多局限于少数参数的孤立分析，缺乏系统性建模。

因此，如何高效、可持续地优化 LLM 推理配置，成为亟待解决的关键挑战。

### 提出的新方法与新思路
本文首次将 **variability modeling**（变体建模）技术引入 LLM 推理配置优化领域，提出了一种系统化的方法框架，核心思想是：
> 将 LLM 推理视为一个**高度可配置的软件系统**，并用 **Feature Model (FM)** 对其生成超参数进行建模。

具体流程为四步法：
1. **Modeling**：构建 Hugging Face Transformers 的 feature model，显式表达所有 generation hyperparameters 及其约束。
2. **Sampling**：采用多种采样策略（YASA、ICPL、RANDOM）从庞大的配置空间中选取代表性子集。
3. **Measurement**：对每个配置执行推理任务，测量 **energy consumption**, **latency**, 和 **accuracy (pass@1)**。
4. **Learning**：基于测量数据训练机器学习模型（Random Forest Regression），预测未见配置的表现。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **系统性** | 多数研究仅分析少量参数，忽略交互 | 显式建模全部参数及约束，支持全局探索 |
| **可行性** | 穷举测试不现实 | 利用 feature model + 采样，避免组合爆炸 |
| **自动化潜力** | 依赖人工经验或网格搜索 | 支持自动推荐最优配置 |
| **预测能力** | 缺乏泛化能力 | 能准确预测未见配置的性能表现 |

---

## 2. 核心实验方法和设置

### 数据集
- **HumanEval+**：一个扩展版的代码生成基准数据集，包含 164 个带单元测试的 Python 编程题。
- 选择理由：支持自动化功能正确性评估（通过 `pass@k` 指标），适合量化 accuracy。

### 选用的 LLM 模型
从 BigCodeBench 排行榜中筛选三个开源、小于 8B 参数、支持完整解码控制的模型：
| 模型名称 | 家族 | 参数量 |
|--------|-----|-------|
| OpenCoder-8B-Instruct | infly | 8B |
| Qwen2.5-Coder-7B-Instruct | Qwen | 7.62B |
| Qwen2.5-Coder-3B-Instruct | Qwen | 3.09B |

后续简称：Qwen-7B、Qwen-3B、OpenCoder-8B。

### 实验设置
- **配置空间建模**：
  - 构建包含 **96 个 features** 的 feature model（其中 67 个 concrete features）。
  - 总有效配置数约为 $9.37 \times 10^{12}$，无法穷举。
  - 数值参数离散化处理（如 temperature ∈ {0.1, 0.3, 0.7, 1.0, 1.2}）。
- **采样策略**：
  - **YASA** 和 **ICPL**：基于 2-wise 覆盖的 interaction-aware 采样器（确保参数两两组合至少出现一次）。
  - **RANDOM**：随机采样作为对照。
  - 各生成约 77–96 个配置，共 **254 个配置**用于实验。
- **运行方式**：
  - 每个配置在 HumanEval+ 上运行 10 次，共执行 **7,620 次实验**，覆盖 **1,249,680 个 prompt**。
  - 使用 `device_map="auto"` 分布到 4×NVIDIA A100 GPU。
  - 固定 batch size = 32，max token = 512，精度为 bfloat16。
- **预热与校准**：
  - 加入 warm-up 阶段以消除冷启动影响（CUDA 编译、内存分配等）。
  - 内部校准 30 秒，确保能量测量稳定（idle CV = 0.02）。

### 评估指标
| 指标 | 测量工具 | 描述 |
|------|--------|------|
| **Energy Consumption** | `perf` (CPU via RAPL), `nvidia-smi` (GPU) | 总能耗（kJ），含背景进程；也可计算边际能耗 |
| **Latency** | 自定义计时 | 请求提交至响应返回的时间（秒） |
| **Accuracy** | pass@1 | 至少有一个生成样本通过所有测试的概率，$n=1$ |

### 基线方法对比
本文未直接对比其他推理优化算法（如 vLLM 的 PagedAttention），而是聚焦于**配置选择策略本身的比较**：
- 不同采样策略（YASA vs ICPL vs RANDOM vs ALL）对预测模型性能的影响。
- 特征级与成对特征分析揭示参数影响机制。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ RQ1: 参数影响分析（Feature-wise & Pairwise）
- **能源与延迟强相关**（R² > 0.9），说明节能即提速。
- 影响最大的参数：
  - **Energy/Latency**：
    - `cache=offloaded` → +89.48 kJ, +233.51 s
    - `num-beam-groups=4` → +58.03 kJ
    - `num-beams=4` → +50.30 kJ
  - **Accuracy (pass@1)**：
    - `no-repeat-ngram-size=0` → +0.68%
    - `decoding=greedy` → +0.46%
    - `low-memory=True` → -0.23% （轻微降质）

- **Pairwise 分析显示交互效应显著**：
  - 在 `greedy` 解码下：
    - `cache=static` → 能耗 ↑13.76 kJ，延迟 ↑48.12 s
    - `do-sample=True` → 能耗 ↓8.59 kJ，延迟 ↓21.79 s
    - `top-p=0.8` → 准确率 ↑0.23%，同时降低延迟 → 是**理想 trade-off**

#### ✅ RQ2: Pareto 最优权衡
- 构建 **Pareto front** 展示 energy-accuracy 权衡：
  - **低能区（~20–25 kJ）**：小幅度增能即可大幅提升 accuracy（Qwen-3B: 0.66 → Qwen-7B: 0.75）
    - 配置特点：dynamic cache + greedy/contrastive decoding + 单 beam + 默认采样
  - **平衡区（~35–40 kJ）**：accuracy 达 0.77
    - 使用 2-beam + sampling + low temp (0.1–0.3) + low top-k (5–25) + high top-p (0.8–0.85)
  - **高精度区（49–64 kJ）**：accuracy 达峰值 0.80，能耗增加 66%
    - 使用 3–4 beams + 更激进搜索策略

> 表明：**大部分收益可在低能耗区域获得**，追求极致 accuracy 成本高昂。

#### ✅ RQ3: 预测模型性能（Table 5）

| Sampler | Energy (R² / MAE) | Latency (R² / MAE) | Accuracy (R² / MAE) |
|--------|------------------|--------------------|---------------------|
| YASA   | 0.94 / 12.23     | 0.93 / 27.72       | 0.99 / 0.01         |
| ICPL   | 0.74 / 19.59     | 0.78 / 37.77       | 0.97 / 0.02         |
| RANDOM | 0.91 / 15.94     | 0.91 / 32.99       | 0.96 / 0.02         |
| **ALL** (混合) | **0.95 / 10.05** | **0.94 / 23.78** | **0.99 / 0.01** |

- **ALL（混合采样）效果最好**，证明多样化训练数据提升泛化能力。
- **YASA 表现优于 ICPL**，尤其在 energy 和 latency 上。
- **即使 RANDOM 也能取得不错结果**，表明简单采样亦具信息量。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 推理可被建模为可配置系统**，variability modeling 提供了强大的抽象能力来管理其复杂性。
2. **feature model 能有效编码参数约束**，防止无效配置，支持自动化推理（如枚举、验证）。
3. **energy 与 latency 高度正相关**，优化其一是双赢。
4. **关键影响因素明确**：
   - `offloaded cache` 显著增加能耗；
   - `greedy decoding` 在本任务中既高效又准确；
   - `top-p=0.8` 是多个维度上的“甜点”配置。
5. **预测模型高度准确**（R² ≥ 0.94），可用少量样本泛化至整个配置空间。
6. **最优 trade-off 存在于低能耗区域**，无需极端配置即可获得良好性能。

### 方法的局限性
| 类别 | 说明 |
|------|------|
| **外部有效性（External Validity）** | 实验仅限 code generation（HumanEval+），结论可能不适用于 summarization、translation 等任务；也仅使用 Hugging Face Transformers，未涵盖 vLLM、TGI 等专用推理引擎。 |
| **构造有效性（Construct Validity）** | 参数被离散化，可能遗漏连续空间中的最优值；部分约束未能完全捕获（约 3% 配置失败）。 |
| **内部有效性（Internal Validity）** | 实验环境固定（A100 GPU），未考虑异构硬件或动态负载的影响。 |
| **维护成本** | LLM 生态快速演进，feature model 需持续更新以保持同步，带来维护负担。 |

### 未来工作方向
1. **扩展 variability model**：
   - 引入硬件配置（GPU 类型、内存）、部署选项（batching、quantization）等更多变体层级。
2. **实现运行时自适应配置（adaptive reconfiguration）**：
   - 动态调整 hyperparameters 以应对不同输入或资源压力。
3. **跨平台统一接口**：
   - 设计高层抽象（如 domain-specific language）连接不同 inference server 的 feature models，实现统一配置管理。
4. **结合 LLM 自身辅助建模**：
   - 利用 LLM 自动生成或修复 feature model 中的约束规则。

---

> 🔗 **数据与代码公开**：  
> 所有实验代码、notebooks 和数据已发布于 Zenodo：  
> [https://doi.org/10.5281/zenodo.17375044](https://doi.org/10.5281/zenodo.17375044)

--- 

✅ **总结一句话**：  
本文开创性地将 **variability modeling** 应用于 LLM 推理调优，实现了对超大规模配置空间的**系统化探索、精准预测与高效权衡分析**，为绿色、可持续的 LLM 部署提供了新范式。

</details>

---

### 7. [Parallel Complex Diffusion for Scalable Time Series Generation](https://arxiv.org/abs/2602.17706)

**Authors**: Rongyao Cai, Yuxi Wan, Kexin Zhang, Ming Jin, Zhiqiang Ge, Qingsong Wen, Yong Liu  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.17706v1  

#### Abstract
Modeling long-range dependencies in time series generation poses a fundamental trade-off between representational capacity and computational efficiency. Traditional temporal diffusion models suffer from local entanglement and the $\mathcal{O}(L^2)$ cost of attention mechanisms. We address these limi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Parallel Complex Diffusion for Scalable Time Series Generation**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

传统时间序列生成模型（如基于 **temporal diffusion models** 或 **Transformer** 的架构）面临两大核心挑战：

- **局部纠缠 (Local Entanglement)**：在时域中，相邻时间步之间存在强依赖关系，导致模型必须学习复杂的联合分布，难以高效建模长程依赖。
- **计算复杂度高**：主流模型（如 DiT）依赖 **Self-Attention** 机制，其计算复杂度为 $O(L^2)$，其中 $L$ 是序列长度，在处理长序列时效率低下。

此外，现有频域方法多为启发式设计（如仅将频谱作为辅助特征），缺乏严格的数学基础，且未充分利用频域信号的统计独立性。

---

### **提出了什么新方法或新思路**

本文提出 **PaCoDi (Parallel Complex Diffusion)**，一种全新的、基于频域的扩散生成框架，其核心思想是：

- 将扩散过程从 **时域 (temporal domain)** 转移到 **复数频域 (complex frequency domain)**。
- 利用 **Fourier Transform** 作为“对角化算子”，将原本纠缠的时间序列信号转换为统计上近似独立的频谱分量。
- 在频域中，通过理论证明实现 **实部与虚部分支的条件可分解性 (Conditional Reverse Factorization)**，从而支持并行建模。

#### **关键技术创新点**：

1. **频率扩散理论 (Frequency Diffusion Theory)**
   - 提出 **Quadrature Forward Diffusion** 和 **Conditional Reverse Factorization Theorem**，证明在固定初始边界 $X_0$ 下，复数扩散的反向过程可以分解为独立的实部和虚部动态。
   - 这使得可以分别训练两个独立的神经网络分支来预测实部和虚部噪声。

2. **Mean Field Theory (MFT) + Interactive Correction**
   - 虽然理论允许解耦，但真实数据中相位与幅度存在耦合（如相位相干性）。为此引入 **MFT 近似**，假设数据先验可分解。
   - 设计 **Interactive Correction Mechanism**，在两个分支间传递交叉信息（如通过 `h(I)` 投影到实部分支），以恢复被忽略的相关性。

3. **连续时间扩展：Frequency SDEs**
   - 将离散 DDPM 扩展为连续时间 **Stochastic Differential Equations (SDEs)**，构建了 **Spectral Wiener Process** 来描述频域布朗运动。
   - 证明了在连续极限下，频域扩散与原始时域扩散等价。

4. **利用 Hermitian Symmetry 实现压缩**
   - 对于实值时间序列，其傅里叶变换满足 **Hermitian Symmetry**：$X_k = \overline{X_{L-k}}$。
   - 因此只需建模前半段频谱（$K = \lfloor L/2 \rfloor + 1$），实现 **50% 序列长度压缩**，显著降低注意力机制的 FLOPs。

5. **Heteroscedastic Loss 设计**
   - 频域噪声具有非同方差性（不同频率噪声强度不同），因此设计了基于 **Mahalanobis 距离** 的损失函数，以更准确地匹配噪声分布。

---

### **相比现有方法的优势**

| 维度 | PaCoDi | 传统 Temporal Diffusion |
|------|--------|-------------------------|
| **表示能力** | 利用全局频谱特征，天然具备全局感受野 | 依赖堆叠注意力或卷积获取长距离依赖 |
| **计算效率** | 注意力复杂度降低 **50%**（因序列压缩 + 并行分支） | $O(L^2)$ 注意力瓶颈明显 |
| **理论基础** | 严格推导频域扩散的可分解性与 SDE 极限 | 多数频域方法缺乏数学严谨性 |
| **表达灵活性** | 绕过复数网络的 **holomorphic constraint**，可用 GELU/SiLU 等非全纯激活函数 | 复数网络受限于 Cauchy-Riemann 方程 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### **无条件生成 (Unconditional Generation)**

| 数据集 | 样本数 | 变量维度 |
|-------|--------|----------|
| **ETTh1** | 17,420 | 7 |
| **Stocks** | 3,773 | 6 |
| **Sines** | 10,000 | 5 |
| **Air Quality** | 9,333 | 15 |

> 所有数据均为多变量时间序列，用于测试模型生成多样性和保真度的能力。

#### **有条件生成 (Conditional Generation)**

沿用 **T2S** 的设定，在以下单变量数据集上进行片段级生成：
- **ETTh1**, **ETTm1**, **ECL**, **Exchange**, **Air Quality**

---

### **实验设置和评估指标**

#### **无条件生成评估指标**

1. **Discriminative Score**：分类器区分真实 vs 合成样本的准确率（越低越好）
2. **Predictive Score (TSTR)**：在合成数据上训练预测模型，在真实数据上测试误差（越低越好）
3. **Context-FID**：隐空间表示之间的 Wasserstein 距离（越低越好）
4. **Correlational Score**：真实与合成数据相关矩阵差异的 Frobenius 范数（越低越好）

#### **有条件生成评估指标**

1. **MSE**：均方误差
2. **WAPE (Weighted Absolute Percentage Error)**：归一化偏差度量

#### **序列长度范围**

- 无条件任务：$L \in \{24, 64, 128, 256\}$
- 有条件任务：$L \in \{24, 48, 96\}$

---

### **基线方法对比**

#### **无条件生成 Baselines**

- **Diffusion-TS**
- **TimeVAE**
- **TimeGAN**
- **Temporal DDPM**

#### **有条件生成 Baselines**

- **T2S**（专用文本到时间序列模型）
- **Diffusion-TS**
- **TimeVAE**
- **GPT-4o-mini**（零样本大语言模型）
- **Llama3.1-8b**（零样本大语言模型）

> PaCoDi 包含两个变体：
> - **PaCoDi DDPM**：离散扩散版本
> - **PaCoDi SDE**：连续 SDE 版本

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **无条件生成结果（Table 3 & Appendix Tables 6–9）**

| 方法 | ETTh1 (Avg) | Stocks (Avg) | Sines (Avg) | Air (Avg) | 总体第一名次数 |
|------|-------------|------------|-------------|-----------|----------------|
| **PaCoDi SDE (Cont.)** | **0.202** | **0.272** | **0.069** | **0.339** | **11 / 16** |
| **PaCoDi DDPM (Disc.)** | 0.215 | 0.209 | 0.100 | 0.520 | 3 / 16 |
| Diffusion-TS | 0.927 | 0.436 | 0.148 | 0.397 | 2 / 16 |
| TimeVAE | 0.899 | 0.343 | 0.730 | 1.413 | 0 / 16 |
| TimeGAN | 15.771 | 2.989 | 10.766 | 13.452 | 0 / 16 |
| DDPM | 0.556 | 0.636 | 0.071 | 0.603 | 0 / 16 |

> 📌 **PaCoDi SDE 在 16 项指标中拿下 11 个最佳，全面领先。**

#### ✅ **有条件生成结果（Table 4）**

| 方法 | MSE (Avg) | WAPE (Avg) | 第一名次数 |
|------|-----------|------------|------------|
| **PaCoDi SDE** | **0.005** | **0.156** | **9 / 12** |
| **PaCoDi DDPM** | **0.006** | **0.161** | **5 / 12** |
| T2S | 0.011 | 0.215 | 0 |
| Diffusion-TS | 0.075 | 0.833 | 0 |
| TimeVAE | 0.055 | 0.652 | 0 |
| GPT-4o-mini | 0.080 | 0.393 | 2 |
| Llama3.1-8b | 1.224 | 0.918 | 0 |

> 📌 **PaCoDi 在 MSE 上比 Diffusion-TS 快 2.3 倍以上（如 ETTm1 上 0.013 vs 0.031）**

---

### **消融实验结果（Ablation Study, Table 5）**

在 **Sines** 数据集上比较三种变体：

| 模型 | Context-FID ↓ | Discriminative Score ↓ | Predictive Score ↓ |
|------|---------------|------------------------|--------------------|
| **PaCoDi (完整)** | **0.016** | **0.021** | **0.007** |
| Decoupled-only (Dec.) | 0.146 | 0.145 | 0.287 |
| Temporal Baseline (Temp.) | 0.031 | 0.036 | 0.043 |

> 🔍 发现：
> - 单纯解耦（Dec.）虽然快，但生成质量严重下降（FID 差 9 倍），说明忽略跨象限依赖会破坏相位一致性。
> - 加入 **Interactive Correction** 后性能大幅提升，甚至超过时域模型，验证了该机制的有效性。

---

### **计算效率分析（Figure 3）**

- **FLOPs 节省达 50%**，尤其在长序列（$L=256$）时趋于稳定。
- 原因：
  1. 序列长度压缩至 $L/2$
  2. 注意力复杂度从 $O(L^2)$ 降为 $2 \times O((L/2)^2) = 0.5 \times O(L^2)$
- FFT 开销仅为 $O(L \log L)$，远小于注意力主导的成本。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **频域是解耦时间序列的理想空间**  
   Fourier Transform 能有效解除局部纠缠，使频谱分量近似独立，极大简化生成建模。

2. ✅ **复数扩散过程可理论分解**  
   在固定初始条件下，实部与虚部的反向过程可完全分离，为并行建模提供数学依据。

3. ✅ **PaCoDi 实现速度与性能双赢**  
   - 生成质量 **SOTA**，尤其在长序列和高频波动信号中表现优异。
   - 推理速度提升 **约 50%**，得益于序列压缩与并行架构。

4. ✅ **交互修正机制至关重要**  
   完全解耦会导致相位失真；加入轻量级交互模块即可恢复关键依赖，兼顾效率与保真。

5. ✅ **绕过 holomorphic constraint 提升表达力**  
   不强制复数网络满足全纯条件，允许使用现代激活函数（如 GELU），增强建模能力。

---

### **方法的局限性**

1. **依赖平稳性假设**  
   当前理论基于线性 DFT，对非平稳或突变信号（如剧烈事件冲击）可能不够鲁棒。

2. **DC 与 Nyquist 分量特殊处理**  
   需单独处理直流分量（DC）和奈奎斯特频率，增加了实现复杂性。

3. **初始化敏感性**  
   若初始频谱估计不准，可能影响后续去噪路径的稳定性。

4. **解释性仍有限**  
   尽管在频域操作，但神经网络本身仍是黑箱，难以直观解释每个频率成分的作用。

---

### **未来工作方向**

1. **扩展至非均匀采样与不规则时间序列**
2. **结合 wavelet 或 short-time Fourier transform 处理非平稳信号**
3. **探索其他酉变换（如 DCT）下的扩散形式**
4. **应用于更高维时空序列（如视频、气象图）**
5. **开发面向频域的稀疏化与量化技术，进一步加速推理**

---

> 💡 **总结一句话**：  
> **PaCoDi 通过将扩散过程迁移至频域，实现了理论可分解、计算高效、性能优越的时间序列生成新范式，为 scalable 生成建模提供了坚实的新路径。**

</details>

---

### 8. [Learning Long-Range Dependencies with Temporal Predictive Coding](https://arxiv.org/abs/2602.18131)

**Authors**: Tom Potter, Oliver Rhodes  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.18131v1  

#### Abstract
Predictive Coding (PC) is a biologically-inspired learning framework characterised by local, parallelisable operations, properties that enable energy-efficient implementation on neuromorphic hardware. Despite this, extending PC effectively to recurrent neural networks (RNNs) has been challenging, pa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Learning Long-Range Dependencies with Temporal Predictive Coding**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
- **传统 RNN 训练依赖 BPTT**（Backpropagation Through Time），其存在以下缺陷：
  - **非局部计算**：梯度需要从输出反向传播到整个时间序列，违反生物可实现性。
  - **高内存消耗**：需存储完整的激活历史，内存随序列长度线性增长。
  - **缺乏空间并行性**：前向和后向锁定限制了硬件并行执行能力。
- **现有 Predictive Coding**（PC）方法在处理**长程时序依赖**任务上表现不佳，尤其在复杂真实世界任务中难以匹敌 BPTT。

### 🚀 **提出了什么新方法或新思路**
- 提出 **tPC RTRL**（Temporal Predictive Coding with Real-Time Recurrent Learning）：
  - 将 **Temporal Predictive Coding**（tPC）与 **RTRL** 结合，首次实现基于 PC 框架的有效**时空信用分配**（spatio-temporal credit assignment）。
  - 利用 RTRL 中的 **influence matrix** 来追踪参数对隐藏状态的历史影响，从而捕捉长距离依赖。
  - 在 tPC 的自由能量最小化框架中引入 RTRL 式的近似梯度更新规则，保留了 PC 的**局部性**和**并行性**优势。

### ⚖️ **相比现有方法的优势**
| 特性 | BPTT | tPC（原始） | **tPC RTRL（本文）** |
|------|------|-------------|------------------------|
| 局部计算 | ❌ | ✅ | ✅ |
| 内存 vs 序列长度 | 线性增长 | 常数 | 常数 |
| 支持在线学习 | ❌ | ✅ | ✅ |
| 能处理长程依赖 | ✅ | ❌ | ✅ |
| 可部署于神经形态硬件 | ❌ | ✅ | ✅ |

> ✅ **核心优势**：**在保持 PC 生物合理性、低功耗潜力的同时，显著提升对长程依赖建模的能力，接近甚至逼近 BPTT 性能。**

---

## 2. **核心实验方法和设置**

### 📚 **使用了哪些数据集**
1. **Synthetic Copy Task**  
   - 输入为 30 位数字序列，要求模型在延迟 10 步后复现原序列。
   - 目标：测试模型捕获**精确长程依赖**的能力。

2. **WikiText-2**  
   - 英文维基百科文本语料，用于**语言建模任务**。
   - 词表大小约 10,000，训练集约 200 万 token。
   - 评估模型预测下一个词的能力。

3. **CCMatrix 子集（English-French Translation）**  
   - 包含 60 万句英法平行句子对。
   - 用于**机器翻译任务**，是更具挑战性的现实应用。
   - 模型规模达 **15 million 参数**，代表大规模应用场景。

### ⚙️ **实验设置和评估指标**
| 设置项 | 描述 |
|-------|------|
| **模型架构** | 多数实验采用 **Linear Recurrent Unit**（LRU），因其支持 element-wise recurrence，使 RTRL 计算可行（将 influence matrix 对角化，降低内存至 O(P)）。 |
| **训练方式** | 所有方法均使用 Adam 优化器；PC 类方法采用 Inference Learning（IL）范式，包含 E-step（推理）和 M-step（学习）。 |
| **评估指标** |  
| - Copy Task | Validation Accuracy, Cross-Entropy Loss |
| - WikiText-2 | Test Perplexity ↓ |
| - Machine Translation | Test Perplexity ↓, BLEU Score ↑ |

### 🔁 **基线方法对比**
- **BPTT**：标准反向传播通过时间，作为性能上限基准。
- **Spatial BP**：仅使用当前时刻参数进行更新，忽略历史影响，模拟无时序信用分配的情况。
- **tPC（Baseline）**：原始时序预测编码，不结合 RTRL，用于验证改进必要性。
- **tPC RTRL**：本文提出的方法。

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据与对比结果**

#### ✅ **Copy Task（长程依赖合成任务）**
| Method | Val Loss | Val Accuracy |
|--------|----------|--------------|
| BPTT | 0.0176 ± 0.0020 | 0.9993 ± 0.0003 |
| **tPC RTRL** | **0.0574 ± 0.0028** | **1.0000 ± 0.0000** |

> 💡 **结论**：tPC RTRL 达到完美准确率，显著优于 tPC 和 Spatial BP（二者失败），且收敛速度与稳定性接近 BPTT。

#### ✅ **WikiText-2（语言建模）**
| Method | Test Perplexity ↓ |
|--------|--------------------|
| Spatial BP | 103.38 ± 0.39 |
| tPC | 108.99 ± 0.54 |
| BPTT | 98.62 ± 0.23 |
| **tPC RTRL** | **99.19 ± 0.18** |

> 💡 **结论**：tPC RTRL 接近 BPTT 表现，明显优于未增强的 tPC 和 Spatial BP。尽管所有方法差距较小，但统计检验显示差异显著（p < 0.01）。

#### ✅ **Machine Translation（英法翻译，15M 参数）**
| Method | Test Perplexity ↓ | Test BLEU ↑ |
|--------|--------------------|-------------|
| Spatial BP | 16.03 | 8.93 |
| tPC | 28.31 | 3.07 |
| BPTT | **7.49** | **21.11** |
| **tPC RTRL** | **7.62** | **20.71** |

> 💡 **这是本文最重要的成果之一**：
> - tPC RTRL 在大型翻译任务上实现了 **test perplexity = 7.62**，仅略逊于 BPTT（7.49）。
> - 是**首个将 PC 成功应用于千万级参数 RNN 的工作**，标志着该类算法迈向实用化的重要一步。
> - tPC 原始版本在此任务上严重失败，凸显 RTRL 增强的必要性。

### 🔍 **消融实验与分析**
- **tPC vs tPC RTRL**：在所有任务中，tPC RTRL 显著优于原始 tPC，尤其是在需要长程记忆的任务（如翻译）中差距巨大。
- **收敛行为分析**（见 Figure 3）：tPC RTRL 与 BPTT 的训练损失曲线高度一致，表明其优化动态相似，具备良好稳定性。
- **超参敏感性**：tPC 对 inference learning rate 和 iteration 数更敏感；而 tPC RTRL 经过适度调参即可达到高性能，但仍需进一步研究鲁棒初始化策略。

---

## 4. **关键结论和发现**

### 🎯 **论文的主要发现**
1. **tPC 单独不足以建模长程依赖**：在 copy task 和 translation 上表现差，说明缺乏有效的历史信用传递机制。
2. **tPC + RTRL 可有效解决此问题**：通过维护 influence matrix 近似历史梯度路径，可在不牺牲局部性和并行性的前提下实现高效时序信用分配。
3. **性能逼近 BPTT**：在多个任务上，tPC RTRL 实现了与 BPTT 相当的性能，特别是在大规模翻译任务中达到 **7.62 perplexity**，证明其实际可行性。
4. **更适合神经形态硬件部署**：由于无需存储展开图、支持完全并行推理与学习，tPC RTRL 更适合低功耗边缘 AI 设备。

### ⚠️ **方法的局限性**
- **计算开销仍高于纯 tPC**：虽然利用 LRU 实现了 O(P) 内存，但 influence matrix 更新增加了额外计算负担。
- **多层扩展困难**：multi-layer RNN 需要维护多个 influence matrices，目前尚未在 tPC RTRL 中验证。
- **超参数调节复杂**：PC 类方法尚无成熟的“最佳实践”，如 inference LR、iteration 次数等需手动调整。
- **尚未在真实神经形态芯片上实现**：能效优势仍停留在理论层面，缺乏实测能耗数据。

### 🔮 **未来工作方向**
1. **扩展至深层网络**：探索 layer-local RTRL 近似或其他稀疏化策略以支持 deep RNN。
2. **结合其他高效 RNN 单元**：尝试 Element-wise LSTM 或 Factorised RTRL 进一步降低计算成本。
3. **硬件实现与能效测量**：在 Loihi 等 neuromorphic 平台上部署 tPC RTRL，获取真实的 energy consumption 数据。
4. **理论分析深化**：研究 tPC RTRL 是否真正逼近高阶梯度更新，以及其收敛性边界。
5. **通用化训练技巧开发**：建立适用于 PC 框架的初始化、归一化、正则化等技术体系，提升易用性。

---

## ✅ **总结一句话**
> 本论文提出的 **tPC RTRL** 成功将 **Predictive Coding** 扩展至具有**长程依赖**的复杂时序任务，在性能上**逼近 BPTT**，同时保留了**局部、并行、节能**的特性，为未来在**神经形态硬件上实现高效在线学习**提供了强有力的技术路径。

</details>

---

### 9. [Hardware-Friendly Input Expansion for Accelerating Function Approximation](https://arxiv.org/abs/2602.17952)

**Authors**: Hu Lou, Yin-Jun Gao, Dong-Xiao Zhang, Tai-Jiao Du, Jun-Jie Zhang, Jia-Rui Zhang  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.17952v1  

#### Abstract
One-dimensional function approximation is a fundamental problem in scientific computing and engineering applications. While neural networks possess powerful universal approximation capabilities, their optimization process is often hindered by flat loss landscapes induced by parameter-space symmetrie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Hardware-Friendly Input Expansion for Accelerating Function Approximation*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
神经网络在进行**一维函数逼近**（function approximation）时，常因参数空间中的**对称性**（parameter-space symmetries）导致优化困难，表现为：
- **损失平面平坦**（flat loss landscapes），收敛缓慢；
- 存在大量等价极小值，训练过程易陷入局部最优；
- 对高频成分学习能力弱（spectral bias）。

这些问题尤其影响科学计算中对高频率、非光滑或复杂频谱函数的建模精度与效率。

---

### ✅ 提出的新方法与新思路
提出一种名为 **Input-Space Expansion**（输入空间扩展）的硬件友好型方法，其核心思想是：
- 将原始的一维输入 $ x \in \mathbb{R} $ 扩展为高维向量，例如 $[T, T, x, T, m]$，其中 $T$ 和 $m$ 是常数；
- 通过引入**常数填充维度**（constant-filled dimensions），打破网络参数之间的对称性（如神经元排列不变性），从而改善优化路径。

该方法基于物理学中的**对称性破缺**（symmetry breaking）原理，无需修改网络结构或训练流程。

---

### ✅ 相比现有方法的优势
| 特性 | 本方法 | 其他常见方法（如 Dropout、BatchNorm、Fourier Features） |
|------|--------|----------------------------------------------------------|
| **是否增加参数量** | ❌ 否 | ✅ 是（通常会） |
| **推理开销** | ❌ 无额外 FLOPs | ✅ 有（如 BatchNorm 推理需归一化） |
| **实现复杂度** | ✅ 极低（仅数据预处理） | ✅ 较高（需改模型/训练策略） |
| **硬件友好性** | ✅ 高（适用于边缘设备） | ⚠️ 中至低 |
| **通用性** | ✅ 架构无关，即插即用 | ⚠️ 多依赖特定架构 |

> 🔑 **核心优势**：以最小代价（zero computational overhead）实现显著的收敛加速与精度提升，是一种“轻量级”优化增强技术。

---

## 2. 核心实验方法和设置

### 📚 数据集与测试函数
构建了一个包含 **10 个代表性一维函数**的基准集，覆盖多种数学特性：

| 类别 | 函数示例（编号） | 数学特征 |
|------|------------------|---------|
| **Smooth** | F1: 多频正弦组合, F5: 调幅波, F6: 频率扫频 chirp | $C^\infty$, 可微，含高频分量 |
| **Discontinuous** | F2: 方波, F3: 锯齿波, F7: 占空比调制方波 | 不连续点，傅里叶展开无限项 |
| **Non-differentiable** | F4: 三角波, F9: Weierstrass 函数 | 连续但不可导（$C^0$） |
| **Complex Spectrum** | F10: Comb 函数（稀疏脉冲） | 高频 + 稀疏信号 |

所有函数定义域为 $[0, 2\pi]$，训练使用 **1000 个随机采样点**，测试使用 **100 个均匀分布点**。

---

### ⚙️ 实验设置
- **网络架构**：标准 MLP（input → 100 → 100 → 50 → 50 → 1），激活函数为 `tanh`；
- **初始化**：Xavier uniform；
- **优化器**：LBFGS（learning rate = 1.0, max_iter = 500, tol = 1e-10）；
- **Loss**：Mean Squared Error (MSE)；
- **控制变量**：确保各模型总参数数量相近，避免容量差异干扰。

---

### 🆚 基线方法对比
| 模型 | 输入维度 | 参数数 | 说明 |
|------|----------|--------|------|
| **Standard (Std)** | 1D | ~17,951 | 原始输入 $x$ |
| **Expanded-3/5/7 (Exp-k)** | 3D/5D/7D | ~18,151–18,551 | 输入扩展为 $[c,c,x,c,c]$ 形式 |
| **Adjusted (Adj)** | 1D | ~18,875 | 宽度加宽以匹配 Exp-5 参数量 |

> ✅ 关键设计：通过比较 **Exp-5** 与 **Adj** 来验证性能提升来自“对称性破缺”而非“模型容量增大”。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（平均 across 10 functions）

| 模型 | 平均迭代次数（LBFGS） | 最终 MSE ($\times 10^{-2}$) | 收敛速度提升 |
|------|------------------------|-------------------------------|--------------|
| Standard | 473 ± 82 | 7.60 ± 11.13 | 1.00× |
| **Expanded-5 (最优)** | **416 ± 131** | **2.56 ± 4.71** | **1.14× (+12%)** |
| Adjusted | 496 ± 10 | 4.99 ± 6.66 | 0.95× |

> 💡 **结论**：  
> - **收敛加速 12%**（减少约 57 次迭代）；  
> - **最终 MSE 下降 66.3%**；  
> - 性能提升**并非源于更多参数**（Adj 表现更差），而是输入扩展带来的优化几何改进。

---

### 🔍 分类别性能提升（Expanded-5 vs Standard）

| 函数类别 | MSE 改进 | 迭代减少 |
|--------|---------|----------|
| **Smooth** | **89.3%** | **21.0%** |
| **Complex Spectrum** | **97.2%** | **31.1%** |
| Discontinuous | 58.6% | 0.0% |
| Non-differentiable | 38.4% | 0.0% |
| **Overall Average** | **66.3%** | **12.0%** |

> ✅ 方法在**平滑函数和复杂频谱函数上效果最显著**，因其原本受 spectral bias 和对称性影响最大。

---

### 🔬 消融实验结果（Ablation Studies）

#### （1）不同扩展维度的影响
| 扩展维度 | 迭代数 | MSE | 结论 |
|--------|-------|-----|------|
| 1D (Std) | 473 | 7.60 | 基线 |
| 3D | 461 | 4.73 | 初步改善 |
| **5D** | **416** | **2.56** | ✅ 最优平衡点 |
| 7D | 464 | 2.38 | 收敛变慢，过扩增引入噪声 |

> 📌 发现“**Goldilocks Principle**”：5D 为最佳维度——太少无法充分破缺，太多反而增加优化复杂度。

#### （2）不同常数选择的影响（5D 扩展）
| 常数配置 | 相对 MSE | 收敛因子 | 排名 |
|--------|-----------|------------|------|
| **All $\pi$** | **1.00** | **1.00** | #1 |
| Mixed (0,1,e) | 1.08 | 0.95 | #2 |
| All $e$ | 1.15 | 0.92 | #3 |
| All $1$ | 1.32 | 0.87 | #4 |
| **All $0$** | **1.47** | **0.81** | #5（最差） |

> 🔍 **关键发现**：
> - 使用领域相关的常数（如 $\pi$ 在 $[0,2\pi]$ 上表示中点）效果最好；
> - **All 0 效果最差**：零值引入近似线性相关，无法有效打破对称性；
> - 数学一致性 > 多样性（mixed 不及 all $\pi$）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **输入空间扩展能有效打破参数对称性**，显著改善神经网络优化动态；
2. **5D 扩展（如 $[\pi,\pi,x,\pi,\pi]$）为最优配置**，兼顾收敛速度与精度；
3. **常数选择至关重要**：推荐使用与输入域具有数学意义的常数（如周期函数用 $\pi$）；
4. **性能提升独立于模型容量**：相比参数量更大的 Adjusted 模型，Exp-5 仍全面领先；
5. **特别适合高频和平滑函数逼近任务**，可缓解 spectral bias；
6. **完全硬件友好**：仅需数据预处理，不增加推理计算成本。

---

### ⚠️ 方法的局限性
- 对**强间断函数**（如方波、锯齿波）和**处处不可导函数**（如 Weierstrass），虽能提高最终精度，但**收敛加速有限**（迭代减少 0%）；
- 当前仅验证于一维函数，高维推广需进一步研究；
- 常数选择依赖先验知识（如函数周期性），自动化机制尚未提出。

---

### 🔮 未来工作方向
1. 将 input expansion 推广到多维函数逼近与 PINNs（Physics-Informed Neural Networks）；
2. 设计自动化的常数选择策略（如可学习常数或自适应填充）；
3. 结合其他 symmetry breaking 技术（如 equivariant networks）形成复合优化方案；
4. 在真实硬件平台（如 FPGA、嵌入式系统）部署验证其效率优势；
5. 探索其在 Transformer、ResNet 等现代架构中的适用性。

---

> 🏁 **总结一句话**：  
> 本文提出了一种简单而强大的 **hardware-friendly symmetry breaking** 方法——**Input-Space Expansion**，通过常数填充将标量输入升维，在几乎零成本下实现了 **12% 收敛加速** 和 **66.3% MSE 下降**，为高效函数逼近提供了一个即插即用的新工具。

</details>

---

### 10. [RAT+: Train Dense, Infer Sparse -- Recurrence Augmented Attention for Dilated Inference](https://arxiv.org/abs/2602.18196)

**Authors**: Xiuying Wei, Caglar Gulcehre  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.18196v1  

#### Abstract
Structured dilated attention has an appealing inference-time efficiency knob: it reduces the FLOPs of the attention and the KV cache size by a factor of the dilation size D, while preserving long-range connectivity. However, we find a persistent failure mode of them -- sparsifying a pretrained atten...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《RAT+: Train Dense, Infer Sparse -- Recurrence Augmented Attention for Dilated Inference》核心总结

---

## 1. 主要贡献和创新点

### 解决的问题
现代语言模型中的标准 **attention** 机制在长序列上存在 **quadratic cost**（计算量和内存随序列长度平方增长）的问题。虽然已有多种稀疏化方法（如 dilated attention、local window attention），但直接将预训练好的 dense 模型稀疏化为 **dilated pattern** 会导致严重的性能下降。

此外，现有的稀疏架构（如 RAT、Mamba）通常需要从头训练，缺乏推理时灵活切换不同稀疏模式的能力。

### 提出的新方法：RAT+
提出 **RAT+** —— 一种“**train dense, infer sparse**”的新型架构，其核心思想是：
- 在预训练阶段保持 **dense attention** 能力；
- 引入 **full-sequence recurrence** 和 **active recurrence learning (ARL)**，使模型具备构建完整感受野（receptive field）的能力；
- 推理时可灵活切换到各种稀疏模式（如 dilated attention、top-k block attention、hybrid 设计等），无需重新训练。

#### 关键技术点：
- **Full-sequence recurrence**：采用覆盖整个序列的递归模块（而非分块处理），简化实现并保证一致性。
- **Active Recurrence Learning (ARL)**：通过联合训练 dense 和 sparse 配置（如 $D=1$ 与 $D=64$），强制 recurrence 学习足够长的有效上下文长度（如 $L^*=64$）。

### 相比现有方法的优势
| 方面 | 传统稀疏架构（如 RAT） | 标准 Attention 模型 | RAT+（本文） |
|------|------------------------|-----------------------|--------------|
| 是否需重训 | 是（每个配置单独训练） | 否（但不能有效稀疏化） | **否**（一次预训练，多模式推理） |
| 支持 Dilated Inference | ✅ | ❌（严重掉点） | ✅（接近 dense 性能） |
| 支持 Hybrid Layers/Heads | ❌ | ❌ | ✅（可行） |
| 对 Top-k Block 的支持 | ❌ | ⚠️（效果一般） | ✅（显著更优） |

> RAT+ 实现了 **灵活性 + 高效性 + 高性能** 的统一。

---

## 2. 核心实验方法和设置

### 数据集
- **预训练数据**：FineWeb-Edu（100B 或 200B tokens）
- **下游任务评估**：
  - **Commonsense Reasoning**：ARC-C, ARC-E, HellaSwag, PIQA, LAMBADA, Winograd（短上下文，$T \leq 300$）
  - **长上下文理解**：LongBench（涵盖问答、摘要、多跳推理等）
  - **检索能力测试**：RULER 中的 Needle-in-a-Haystack (NIAH) 任务（精确匹配评分）

### 实验设置
- **模型规模**：
  - 主要：1.5B 参数（context length 4096）
  - 扩展：2.6B 参数（100B / 200B tokens 训练）
- **稀疏模式**：
  - Dilated Attention（$D=1,2,...,128$）
  - Optional Local Window（如 $W=256, 512$）
  - Hybrid Layer/Head 设计
  - Top-k Block Attention（结合 Quest / MoBA block selection）
- **适配过程**：仅用 **1B tokens** 进行轻量级 resolution adaptation（无需 full retraining）

### 评估指标
| 类型 | 指标 |
|------|------|
| 预训练质量 | Validation Perplexity (PPL) |
| 下游理解能力 | Average Accuracy on Commonsense Tasks |
| 长文本建模 | LongBench 平均得分 |
| 检索能力 | NIAH Exact Match Accuracy |
| 效率 | Prefill/Decode Latency, Throughput (tokens/sec), FLOPs Reduction |

### 基线方法对比
- **Attention-only**：标准 dense attention 模型
- **RAT**：原始 chunk-based recurrence + dilated attention 架构
- **Mamba2**, **GatedDeltaNet**：state space 模型代表
- **StreamingLLM**：局部 attention + attention sink 技术

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ 在 Commonsense Reasoning 上的表现（1.5B 模型）
| 模型 | Dilation ($D$) | Avg. Accuracy | 相比 Dense 掉点 |
|------|----------------|-------------|----------------|
| Attention (dense) | 1 | 58.33 | — |
| RAT+ | 16 | 58.06 | ~0.3 pt |
| RAT+ | 64 | 57.46 | ~0.9 pt |

> 即使 $D=64$，仍保持接近 dense 水平，远优于直接稀疏化的 attention（$D=16$: 32.6 → 掉点超 25 pts）。

#### ✅ 在 LongBench 上的表现（平均得分）
| 模型 | Dilation | Avg Score | 备注 |
|------|----------|-----------|------|
| RAT+ ($D=1$) | 1 | 19.37 | dense baseline |
| RAT+ ($D=16$) | 16 | 18.50 | 仅降 0.87 pts |
| RAT+ ($D=64$) | 64 | 16.56 | 可接受范围 |
| Attention ($D=16$) | 16 | — | 无法有效稀疏化 |

> 不同任务偏好不同稀疏策略（如 RBP 更喜 local attention），验证了 hybrid 设计必要性。

#### ✅ Top-k Block Attention 表现（NIAH-MK-2, $T=4096$）
| 模型 | Pattern | Accuracy |
|------|--------|---------|
| Attention | $D=64, K=16$ | 63.2 |
| RAT+ | $D=64, K=16$ | **93.8** |
| RAT+ (no ARL in SFT) | $D=64, K=16$ | 76.8 |

> RAT+ 显著优于纯 attention，在 block selection 中表现更强，说明 recurrence 提升了 block 内容表征能力。

#### ✅ 效率提升（解码吞吐量）
| Context Length | Model Size | Max Throughput Gain (vs Dense) |
|----------------|------------|-------------------------------|
| 4K | 1.5B | **10×** |
| 16K | 1.5B | **20×** |
| 4K | 7B | >60× |
| 16K | 7B | >40× |

> 通过减少 KV Cache 和 FLOPs（降至 $O(T/D)$），实现数量级加速。

### 消融实验结果（Table 3 & Table 7）

| 变体 | 描述 | 结果分析 |
|------|------|----------|
| **L=T, L*=64**（最终 RAT+） | 全序列 recurrence + ARL | 所有 dilation 下稳定，PPL 接近 dense |
| L=64（chunked） | 固定 chunk size | 小 dilation（如 D=1）下 PPL 明显升高（分布偏移） |
| No ARL（仅 D=1 训练） | 无稀疏监督 | 大 dilation（如 D=64）下 PPL 快速上升（感受野不完整） |
| No Adaptation | 不进行 1B token 微调 | 性能下降明显，尤其在大 D 下 |

> 验证了 **full-sequence recurrence** 和 **active recurrence learning** 的必要性。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **Dilated attention 本身不足以支撑高效稀疏推理**  
   单纯将 dense 模型改为 dilated pattern 会因感受野断裂导致严重性能退化。

2. 🔄 **Recurrence 是构建完整感受野的关键机制**  
   输入依赖的 forget-gate-like recurrence 能显式传递历史信息，弥补稀疏连接的断层。

3. 🧠 **“Train Dense, Infer Sparse” 是可行且高效的范式**  
   RAT+ 成功实现了单一模型支持多种稀疏推理模式，仅需极少量 adaptation tokens（1B）即可完成切换。

4. ⚡ **效率增益巨大**  
   - FLOPs 降低至 $1/D$
   - KV Cache 减少至 $1/D$
   - 解码吞吐量提升达 **10–60×**

5. 🎯 **RAT+ 在 top-k block attention 上也表现更优**  
   表明 recurrence 不仅服务于 dilated attention，还能增强其他稀疏模式的效果。

### 局限性
- 当前实现基于 PyTorch，未进行 CUDA-level 优化，仍有进一步加速空间。
- 最优 hybrid layer/head 配置尚未系统搜索，依赖人工设计。
- recurrence 引入额外投影层，略微增加参数量（未来可压缩）。

### 未来工作方向
- 开发专用 CUDA kernel 以最大化推理效率。
- 自动搜索最优 hybrid sparse configuration（跨层/头）。
- 将 RAT+ 应用于更大模型（>7B）和 tokenizer-free 设置（byte-level input）。
- 探索 recurrence 在其他模态（视觉、语音）中的通用性。

---

> 💡 **总结一句话**：  
> RAT+ 通过引入 **full-sequence recurrence + active learning**，首次实现了 **一个 dense 模型、多种 sparse 推理** 的统一框架，在保持高性能的同时获得数十倍效率提升，为下一代高效语言模型提供了新范式。

</details>

---

### 11. [Optimal Multi-Debris Mission Planning in LEO: A Deep Reinforcement Learning Approach with Co-Elliptic Transfers and Refueling](https://arxiv.org/abs/2602.17685)

**Authors**: Agni Bandyopadhyay, Gunther Waxenegger-Wilfing  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.17685v1  

#### Abstract
This paper addresses the challenge of multi target active debris removal (ADR) in Low Earth Orbit (LEO) by introducing a unified coelliptic maneuver framework that combines Hohmann transfers, safety ellipse proximity operations, and explicit refueling logic. We benchmark three distinct planning algo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Optimal Multi-Debris Mission Planning in LEO: A Deep Reinforcement Learning Approach with Co-Elliptic Transfers and Refueling*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文聚焦于**低地球轨道（LEO）中的多目标主动碎片清除（Active Debris Removal, ADR）任务规划问题**。该问题面临以下挑战：
- 多个非合作碎片目标的空间分布复杂；
- 航天器受限于有限的 $\Delta V$（燃料）和任务时长；
- 需要满足安全约束（如避障区、碰撞规避）；
- 规划需兼顾轨迹效率与长期策略优化。

传统方法难以在资源限制下实现高效、安全且可扩展的任务序列决策。

---

### 🚀 提出的新方法与创新点
1. **统一的共椭圆机动框架（Unified Co-Elliptic Maneuver Framework）**
   - 将三种关键技术集成到一个连贯的轨道操作流程中：
     - **Hohmann transfers**：用于基础轨道转移；
     - **Co-elliptic transfers**：提升多目标访问的相位调整效率；
     - **Safety ellipse maneuvers**（来自 Barbee et al., 2011）：确保对非合作目标的安全接近；
     - 显式的**refueling 逻辑**：支持多次补给以延长任务寿命。

2. **基于 Masked Proximal Policy Optimization (PPO) 的深度强化学习（Deep RL）规划器**
   - 使用 **action masking** 技术屏蔽已访问碎片和非法动作，保证输出动作始终合法；
   - 在自定义的 OpenAI Gym 环境中训练，结合真实轨道动力学仿真（使用 Poliastro 库）；
   - 奖励函数设计鼓励最大化访问碎片数量，同时惩罚过早终止或违反约束。

3. **端到端可扩展、实时可行的自主任务规划架构**
   - 相比搜索类方法（如 MCTS），具备极快推理速度，适合星上部署；
   - 具备良好的泛化能力，在随机生成的 debris field 上表现稳定。

---

### 🔍 相比现有方法的优势
| 方面 | 本文方法优势 |
|------|---------------|
| **建模完整性** | 统一整合了转移、安全接近、补给三大模块，更贴近实际 ADR 任务需求 |
| **算法性能** | 在解的质量（访问碎片数）和计算效率之间取得最佳平衡 |
| **可扩展性** | RL 策略通过训练即可适应不同 debris 分布，无需重新设计启发式规则 |
| **实用性** | 推理时间仅 ~1–2 秒，远优于 MCTS，具备星载应用潜力 |

---

## 2. 核心实验方法和设置

### 🧪 数据集与环境生成
- **无公开真实数据集**，采用**合成生成方式模拟 LEO 碎片场**：
  - 每次 episode 随机生成 $N = 50$ 个 debris 对象；
  - 轨道高度均匀分布在 700–800 km；
  - 初始航天器位于 700 km 圆形轨道（倾角 96°），对接于 refueling station；
  - 所有轨道参数（Keplerian elements）均随机采样，增强泛化性。

> ⚙️ 使用 `Poliastro` 和 `Astropy` 进行高保真轨道传播与 $\Delta V$ 计算。

---

### 📊 实验设置与评估指标

#### 主要设置：
- **最大 $\Delta V$ 预算**：3 km/s；
- **最大任务持续时间**：7 天；
- **episode 终止条件**：耗尽燃料、超时、所有 debris 被访问或执行无效动作；
- **测试规模**：共 100 个独立随机测试案例，每个策略运行 10 次取平均。

#### 评估指标：
| 指标 | 描述 |
|------|------|
| **Debris Visited** | 成功完成 rendezvous 的碎片总数（核心性能指标） |
| **Total Computational Time** | 整个任务规划所用的 CPU 时间（衡量实用性） |
| **Constraint Satisfaction** | 是否遵守 $\Delta V$、时间、安全距离等硬性约束 |

---

### 🔁 基线方法对比
论文系统比较了三种代表性规划算法：

| 方法 | 类型 | 特点 |
|------|------|------|
| **Greedy Heuristic** | 经典贪心算法 | 每步选择使 $\alpha \Delta V + \beta T$ 最小的目标；$\alpha=1,\beta=0$（最小化 $\Delta V$） |
| **Monte Carlo Tree Search (MCTS)** | 搜索算法 | 使用 UCT 策略展开搜索树，每步进行 200 次模拟， rollout 深度为 15 |
| **Masked PPO (Ours)** | 深度强化学习 | 基于策略梯度，带 action masking，训练步数达 10 million |

> 所有方法在同一仿真环境中运行，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Fig. 3 和 Fig. 4）

| 方法 | 平均访问 debris 数量 | 最大可达数量 | 平均计算时间 |
|------|------------------------|----------------|----------------|
| **Greedy** | 15–18 | ~18 | ~1–2 s |
| **MCTS** | 25–29 | ~29 | 1,000 – 10,000 s (~17–167 min) |
| **Masked PPO** | **29–32** | **~32** | **~1–2 s** |

> ✅ **Masked PPO 访问的 debris 数量是 Greedy 的近两倍，且显著高于 MCTS**

---

### 🔍 与基线方法的对比结果
- **vs. Greedy**：
  - Greedy 因“短视”行为频繁陷入局部最优，错过后续高价值目标；
  - Masked PPO 学会权衡短期成本与长期收益，能规划出更高效的全局路径。
  
- **vs. MCTS**：
  - MCTS 能探索未来状态，获得较优解（接近最优），但**计算开销过大**；
  - 其单次决策耗时高达数千秒，**不适用于实时或星载场景**；
  - Masked PPO 在几乎相同的解质量下，**提速超过 3 个数量级**。

> 💡 结论：**Masked PPO 实现了“接近 MCTS 的性能 + Greedy 的速度”这一理想组合**

---

### ❌ 消融实验（Ablation Study）
尽管文中未明确列出消融实验表格，但在讨论中隐含进行了关键组件的有效性验证：
- **Action Masking**：确保策略不会选择已访问 debris 或非法动作，提高训练稳定性；
- **Refueling Logic Integration**：模型学会在适当时机返回补给站，避免过早耗尽燃料；
- **Co-elliptic + Safety Ellipse Modeling**：相比纯 Hohmann transfer 更节省 $\Delta V$ 和时间，尤其适用于密集 debris cluster。

> 作者指出，这些机制共同提升了策略的鲁棒性和现实可行性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Masked PPO 是当前多目标 ADR 任务规划中最具前景的方法之一**：
   - 在 100 个随机测试中 consistently 访问最多 debris（最高达 32/50）；
   - 推理速度快（<2 秒），适合嵌入式系统部署；
   - 能有效处理 refueling 决策、collision avoidance 和动态环境变化。

2. **传统方法存在明显瓶颈**：
   - Greedy 过于短视，无法应对复杂 debris 分布；
   - MCTS 虽然性能较好，但计算代价过高，难以实用化。

3. **学习型方法正在成为复杂空间任务规划的新范式**：
   - 强化学习能够从大量模拟中学习长期策略，超越手工设计的启发式规则；
   - 结合物理准确的动力学模型后，RL 策略具有高度泛化能力。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **忽略 J2 扰动等摄动因素** | 当前仿真假设二体问题，未考虑地球非球形引力影响（J2） |
| **静态 debris 场** | 所有 debris 被视为静止目标，未模拟其轨道漂移或不确定性 |
| **简化服务过程** | rendezvous 后的服务时间固定为一个轨道周期，未建模捕获失败概率 |
| **训练依赖大量模拟** | 需要千万级训练步数，前期成本较高 |

---

### 🔮 未来工作方向
1. **引入更精细的动力学模型**：
   - 加入 J2、大气阻力、太阳辐射压等 perturbations；
   - 支持更高精度的长期轨道预测。

2. **迁移学习与在线适应**：
   - 利用 transfer learning 快速适应新的 debris 分布或任务配置；
   - 开发 online adaptation 机制以应对突发障碍或目标丢失。

3. **多智能体协同 ADR 任务规划**：
   - 扩展至多个 chaser 协同清理，研究通信、分工与冲突协调机制。

4. **硬件在环与星载验证**：
   - 将训练好的策略部署至 CubeSat 或飞行处理器上进行 real-time 测试；
   - 推动 RL-based 规划器走向工程应用。

---

## ✅ 总结
本论文提出了一种融合 **co-elliptic transfers、safety ellipse maneuvers 与 refueling 机制**的新型 ADR 任务规划框架，并首次将 **Masked PPO** 成功应用于多目标轨道 rendezvous 决策。实验证明，该方法在**任务效率**和**计算速度**方面全面超越 Greedy 与 MCTS，展现出强大的实用潜力，标志着 **deep reinforcement learning 正逐步成为下一代自主空间任务规划的核心技术**。

</details>

---

### 12. [Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](https://arxiv.org/abs/2602.17898)

**Authors**: Jingquan Yan, Yuwei Miao, Peiran Yu, Junzhou Huang  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.17898v1  

#### Abstract
Attention-based regression models are often trained by jointly optimizing Mean Squared Error (MSE) loss and Pearson correlation coefficient (PCC) loss, emphasizing the magnitude of errors and the order or shape of targets, respectively. A common but poorly understood phenomenon during training is th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
该论文系统地研究了一个在 **attention-based 回归模型** 中广泛存在但未被充分理解的现象：**PCC Plateau（Pearson Correlation Coefficient 平台期）**。

- 在联合优化 **MSE loss** 和 **PCC loss** 的训练过程中，尽管 MSE 持续下降，PCC 却在早期就停止提升，形成“平台”。
- 这种现象在 **样本内同质性高（high in-sample homogeneity）** 的数据中尤为严重（如病理图像、视频帧等），限制了模型对预测顺序或趋势（shape）的学习能力。

### 🧠 **提出了什么新方法或新思路**
作者提出了一种名为 **Extrapolative Correlation Attention (ECA)** 的新型注意力机制，从两个根本层面解决该问题：

#### （1）理论分析揭示两大根本限制：
- **优化动态冲突（Optimization Dynamics Conflict）**  
  随着 MSE 优化推动预测标准差 $ \sigma_{\hat{y}} $ 向真实值 $ \sigma_y $ 匹配，PCC 梯度会因缩放因子 $ 1/\sigma_{\hat{y}} $ 而衰减，导致其更新信号被压制。
  
- **模型容量瓶颈（Model Capacity Limit）**  
  Softmax 注意力是一种 **凸组合（convex aggregation）**，聚合后的表示 $ v_s $ 被限制在输入嵌入的 **凸包（convex hull）** 内。当数据高度同质时，凸包半径小，可调整空间有限，从根本上限制了 PCC 的提升潜力。

#### （2）ECA 的三大创新组件：
1. **Scaled Residual Aggregation (SRA)**  
   - 允许聚合向量 **超出凸包范围**，通过可学习的放大因子 $ \gamma_s \geq 1 $ 放大残差项 $ \sum \alpha_{si}(h_{si} - \mu_s) $。
   - 打破了传统 attention 的表达能力上限。

2. **Dispersion-Aware Temperature Softmax (DATS)**  
   - 动态调整 softmax 温度 $ T_s $，使其依赖于样本内的分散度 $ \sigma_s $。
   - 当样本高度同质时降低温度，增强微弱差异的注意力区分度，避免注意力分布趋于均匀。

3. **Dispersion-Normalized PCC Loss (DNPL)**  
   - 对 PCC 损失进行重加权：$ \mathcal{L}_{PCC} = \text{StopGrad}(\sigma_{\hat{y}}) \cdot (1 - p) $
   - 抵消 $ 1/\sigma_{\hat{y}} $ 引起的梯度衰减，恢复相关性学习的动力。

### ⚖️ **相比现有方法的优势**
- **首次提供 PCC Plateau 的严格理论解释**，涵盖优化动力学与模型容量两方面。
- **ECA 是即插即用模块**，可无缝集成到任何基于 attention 的回归架构中（如 FT-Transformer, ALMT, EGN）。
- 在保持甚至改善 MSE 性能的同时，显著提升 PCC，突破原有平台限制。
- 特别适用于 **高同质性数据场景**（如医学图像、时间序列、情感分析）。

---

## 2. **核心实验方法和设置**

### 📚 **使用了哪些数据集**

| 数据集类型 | 名称 | 描述 |
|----------|------|------|
| **合成数据集** | Synthetic Dataset | 控制样本内同质性程度（通过参数 $ \eta $），用于验证理论假设。 |
| **表格回归基准** | UCI Datasets | 包括 Appliance Energy, Online News Popularity, Superconductivity。 |
| **空间转录组学** | 10xProteomic Dataset | 来自乳腺癌病理切片的空间基因表达预测任务，具有强样本内同质性（相邻区域相似）。 |
| **多模态情感分析** | MOSI Dataset | 视频片段的情感强度回归任务，连续视觉帧高度相似。 |

### 🎯 **实验设置和评估指标**

#### **评估指标**
- **MSE / MAE**：衡量预测值的 **幅度误差（magnitude matching）**
- **PCC (Pearson Correlation Coefficient)**：衡量预测值与真实值之间的 **排序一致性（shape/relative trend）**
- **F1 Score**（仅 MOSI）：二分类情感极性下的分类性能。

#### **训练设置**
- 使用 **联合损失函数**：$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{PCC}} \cdot \mathcal{L}_{\text{PCC}} $
- ECA 替换原模型中的标准 softmax attention 层。
- 多数实验采用 **三折交叉验证** 或标准 train/val/test 分割。

### 🔁 **基线方法对比**
- **FT-Transformer**（表格数据）
- **EGN**（空间转录组）
- **ALMT**（MOSI 情感分析）
- 所有基线均使用相同的 backbone 架构，仅将 attention 替换为 ECA 进行公平比较。
- 同时对比了添加 PCC loss 但不改结构的方法（如 ALMT+PCC）。

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据汇总**

| 数据集 | 方法 | PCC 提升 | MSE 下降 |
|-------|------|--------|---------|
| **Appliance** | FT-Transformer → +ECA | **+0.042** | **-0.318×10³** |
| **Online News** | FT-Transformer → +ECA | **+0.012** | **-0.012** |
| **Superconductivity** | FT-Transformer → +ECA | **+0.010** | **-0.190×10²** |
| **10xProteomic** | EGN → +ECA | **PCC@M: +13.81%** | **-9.83%** |
| **MOSI** | ALMT+PCC → +ECA | **+1.5 ppt (0.791→0.806)** | **MAE: 0.731→0.695** |

> 注：ppt = percentage point

### 🔍 **与基线方法的对比结果**
- 在所有四个任务上，**ECA 显著打破 PCC plateau**，PCC 持续上升至后期，而基线早早停滞。
- **MSE 同步下降或持平**，说明 ECA 在提升相关性的同时未牺牲幅度准确性。
- 在 MOSI 上，仅加入 PCC loss 会导致 F1 和 MAE 下降（MSE-PCC 冲突），而 ECA 成功协调二者，实现全面增益。

### 🔧 **消融实验结果（Ablation Studies）**

| 方法变体 | Appliance PCC | Online News PCC | Superconductivity PCC |
|--------|---------------|------------------|------------------------|
| Full ECA | **0.598** | **0.420** | **0.930** |
| w/o SRA | 0.575 | 0.418 | 0.920 |
| w/o DATS | 0.561 | 0.410 | 0.927 |
| w/o DNPL | 0.583 | 0.418 | 0.922 |

- **SRA** 对在线新闻类异质性较强的数据更重要；
- **DATS** 在高同质性任务中作用显著；
- **DNPL** 是维持 PCC 持续优化的关键。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **PCC Plateau 是普遍存在的优化失败模式**，尤其在高同质性数据中。
2. 该现象源于两个根本原因：
   - **优化层面**：MSE 优化导致 $ \sigma_{\hat{y}} $ 增大，抑制 PCC 梯度（$ \propto 1/\sigma_{\hat{y}} $）。
   - **容量层面**：softmax attention 的凸组合特性限制了表示空间，无法有效放大微弱差异。
3. **ECA 通过 SRA、DATS、DNPL 三管齐下，成功打破平台限制**，实现 PCC 与 MSE 的协同优化。

### ⚠️ **方法的局限性**
- SRA 中的外推因子 $ \gamma_s $ 需要约束（如 clipping 或正则化），否则可能导致过拟合或数值不稳定。
- 当前分析聚焦于最终的 attention pooling 层，对深层中间层的影响尚未完全建模。
- 理论推导基于 batch-level 统计，对小批量或流式场景的适用性有待进一步验证。

### 🔮 **未来工作方向**
- 将 ECA 扩展至 **生成式建模** 或 **序列到序列任务** 中的相关性优化。
- 探索更通用的 **非凸聚合机制**，超越 SRA 的线性外推形式。
- 结合 **因果推理** 或 **对比学习**，进一步解耦 magnitude 与 shape 学习过程。
- 在更多领域（如金融时间序列、气候建模）验证 ECA 的泛化能力。

---

> **一句话总结**：  
> 本文首次从理论层面揭示了 attention-based 回归模型中 **PCC plateau** 的成因，并提出 **ECA** 框架，通过 **可控外推、分散感知注意力、梯度归一化损失** 三大机制，成功打破相关性优化瓶颈，在多个挑战性任务上实现了 PCC 的显著提升而不损害 MSE 性能。

</details>

---

### 13. [Parameter-Efficient Domain Adaptation of Physics-Informed Self-Attention based GNNs for AC Power Flow Prediction](https://arxiv.org/abs/2602.18227)

**Authors**: Redwanul Karim (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), Changhun Kim (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), Timon Conrad (Institute of Electrical Energy Systems, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Germany), Nora Gourmelon (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), Julian Oelhaf (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), David Riebesel (Institute of Electrical Energy Systems, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Germany), Tom\'as Arias-Vergara (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), Andreas Maier (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), Johann J\"ager (Institute of Electrical Energy Systems, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Germany), Siming Bayer (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany)  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.18227v1  

#### Abstract
Accurate AC-PF prediction under domain shift is critical when models trained on medium-voltage (MV) grids are deployed on high-voltage (HV) networks. Existing physics-informed graph neural solvers typically rely on full fine-tuning for cross-regime transfer, incurring high retraining cost and offeri...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Parameter-Efficient Domain Adaptation of Physics-Informed Self-Attention based GNNs for AC Power Flow Prediction

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对 **AC Power Flow (AC-PF)** 预测中的 **domain shift 问题**，即在中压（MV）电网上训练的模型部署到高压（HV）网络时性能显著下降。这种跨电压域迁移面临以下挑战：
- 电网拓扑、线路参数（如 R/X 比）、运行状态统计特性差异大；
- 传统方法依赖 **full fine-tuning**，计算成本高且容易导致源域知识遗忘（source-domain forgetting），影响稳定性与泛化能力。

### 🚀 提出的新方法
提出一种 **参数高效的领域自适应策略（parameter-efficient domain adaptation）**，结合：
- **Low-Rank Adaptation (LoRA)**：仅对注意力机制中的查询（query）、键（key）、值（value）投影矩阵进行低秩更新；
- **Selective Unfreezing of Prediction Head**：同时解冻预测头以调节输出层适应能力，形成 **LoRA+PHead** 架构。

该方法在保持物理一致性的同时，显式控制“**稳定性-可塑性权衡**”（stability-plasticity trade-off）。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 减少 >85% 可训练参数，大幅降低再训练开销 |
| **准确性** | 接近 full fine-tuning 的精度（RMSE 差距仅 2.6×10⁻⁴） |
| **物理一致性** | 物理残差（physics residual）与 full fine-tuning 相当，保证解的可行性 |
| **源域保留** | 显著优于 LoRA-only 或 Head-only 方法，在目标域适应的同时减少灾难性遗忘 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 自建合成数据集，模拟符合欧洲/德国典型架空线路参数范围的 **MV 和 HV 电网**；
- 包含：
  - **MV 数据集**：90,030 个样本，电压等级 10 kV，线路较短（1–20 km），电阻主导（高 R/X）；
  - **HV 数据集**：45,030 个样本，电压等级 110 kV，线路更长（1–50 km），电抗主导；
- 图结构为连通图，节点数 4–32，包含 PQ、PV 和 Slack 节点；
- 输入特征：有功/无功注入、节点类型指示符、电压设定值；边特征：线路阻抗、导纳等物理参数。

### ⚙️ 实验设置
- **基础模型**：基于 **Edge-Aware Self-Attention GNN** 的 Physics-Informed GNN；
  - 隐藏维度 d=8，每层 8 层，注意力头数 H=4；
  - 使用 **physics-informed loss**（`CpF`）约束 Kirchhoff 定律满足；
- **训练流程**：
  1. 在 MV 源域上预训练模型；
  2. 冻结主干网络，仅微调 LoRA 参数和预测头；
- **LoRA 设置**：rank r=2，scaling α=8；
- **优化器**：AdamW（lr=1e-4, weight decay=1e-3），余弦退火重启；
- **Batch Size**：512，训练 100 轮。

### 🎯 评估指标
| 指标 | 含义 |
|------|------|
| `RMSE_all`, `RMSE_V`, `RMSE_θ` | 总体、电压幅值、相角预测误差（越小越好） |
| `CpF`（Physics Residual Loss） | 表征物理一致性，衡量 ΔP 和 ΔQ 的残差（越小越好） |
| `R_ret`（Retention Score） | 源域保留率，定义为相对于零样本迁移（zero-shot）的 RMSE 改善比例（越高越好） |
| `P_reduced (%)` | 相比 full fine-tuning 减少的可训练参数百分比 |

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **Zero-Shot (Base)** | 不做任何微调，直接迁移 |
| **Full Fine-Tuning (Full FT)** | 全参数微调，作为性能上限 |
| **Head Only** | 仅微调预测头 |
| **LoRA Only** | 仅应用 LoRA 到注意力投影 |
| **LoRA+PHead** | 本文提出的方法（LoRA + 解冻预测头） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table I）
| Method | RMSE_all | RMSE_V | RMSE_θ | CpF↓ | P_reduced (%)↑ | R_ret (%)↑ |
|--------|----------|--------|--------|-------|------------------|--------------|
| Zero-Shot (Base) | 1.65×10⁻² | 1.71×10⁻³ | 9.38×10⁻¹ | 2.57 | — | — |
| Full FT | **9.35×10⁻⁴** | **3.61×10⁻⁴** | **4.95×10⁻²** | **1.11** | 0% | **22.6%** |
| Head Only | 1.38×10⁻³ | 4.19×10⁻⁴ | 7.51×10⁻² | 1.11 | 88.50% | 22.3% |
| LoRA Only | 3.61×10⁻³ | 1.05×10⁻³ | 1.98×10⁻¹ | 6.08 | 96.56% | 17.3% |
| **LoRA+PHead** | **1.20×10⁻³** | **3.53×10⁻⁴** | **6.56×10⁻²** | **1.21** | **85.46%** | **17.9%** |

> ✅ **关键结论**：
- LoRA+PHead 的 RMSE_all 仅比 Full FT 高 **2.6×10⁻⁴**，接近其性能；
- 参数量减少 **85.46%**，实现高效迁移；
- 物理残差 CpF = 1.21，略高于 Full FT（+9%），但仍远优于 LoRA Only（6.08）；
- 源域保留 R_ret = 17.9%，虽低于 Full FT（22.6%），但显著优于 LoRA Only。

### 🔬 消融实验分析
#### （1）Few-shot Adaptation（图 1a）
- 当目标域标注数据极少（β ≤ 5%）时，LoRA 类方法表现落后于 Full FT，说明低秩子空间在极低监督下存在偏差限制；
- 当 β ≥ 10% 时，LoRA+PHead 快速逼近 Full FT，进入 **Pareto 最优前沿**。

#### （2）Efficiency-Accuracy Trade-off（图 1b）
- LoRA+PHead（p ≈ 14.5%）位于 Pareto 前沿，相比 Full FT 实现 **6–7 倍参数压缩**，误差增加极小；
- LoRA Only（p ≈ 3.4%）严重欠拟合，验证了单独调整注意力不足以应对域偏移。

#### （3）物理损失动态（图 1c, 1d）
- Full FT 与 LoRA+PHead 均能稳定收敛至低 CpF 值，未出现发散现象；
- 表明低秩更新不会破坏物理约束下的优化稳定性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **参数高效 ≠ 性能牺牲**：通过 LoRA+PHead 设计，可在仅更新 **14.5% 参数**的情况下恢复 **>97% 的 Full FT 精度**；
2. **物理一致性得以维持**：尽管参数受限，模型仍能学习到满足 Kirchhoff 定律的解，CpF 与 Full FT 接近；
3. **稳定性-可塑性可调控**：通过选择性解冻预测头，实现了对源域保留与目标域适应之间的平衡；
4. **适用于实际部署场景**：特别适合边缘设备或需频繁在线更新的电力系统应用，降低 retraining 成本。

### ⚠️ 方法局限性
- 在 **极低样本（few-shot <5%）** 场景下性能仍落后于 Full FT，表明低秩假设可能限制表达能力；
- 当前研究集中于 **MV→HV 单向迁移**，尚未验证反向或多跳迁移效果；
- LoRA 应用于 GNN 注意力模块的有效性依赖于特定架构设计，通用性有待进一步验证。

### 🔮 未来工作方向
- 结合 **Adapter Layers** 或 **Prompt Tuning** 进一步探索其他 parameter-efficient 方法；
- 扩展至 **multi-step OPF** 或 **dynamic state estimation** 等复杂任务；
- 引入 **uncertainty quantification** 以增强模型在极端工况下的鲁棒性；
- 探索 **real-world grid data** 上的迁移性能，提升工业适用性。

---

## 总结
本文首次将 **Low-Rank Adaptation (LoRA)** 引入 **Physics-Informed GNNs** 的电力系统领域迁移任务中，提出 **LoRA+PHead** 方法，在 **AC-PF 预测的 MV→HV 跨域场景**下实现了：
> ✅ **近全微调精度** + ✅ **超85%参数压缩** + ✅ **良好物理一致性** + ✅ **可控的稳定性-可塑性权衡**

为学习型电力系统求解器的 **高效、可靠、可持续部署** 提供了一条切实可行的技术路径。

</details>

---

### 14. [Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies](https://arxiv.org/abs/2602.18291)

**Authors**: Zhuoran Li, Hai Zhong, Xun Wang, Qingxin Xia, Lihua Zhang, Longbo Huang  
**Category**: cs.AI  
**Published**: 2026-02-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.18291v1  

#### Abstract
Online Multi-Agent Reinforcement Learning (MARL) is a prominent framework for efficient agent coordination. Crucially, enhancing policy expressiveness is pivotal for achieving superior performance. Diffusion-based generative models are well-positioned to meet this demand, having demonstrated remarka...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本文针对 **Online Multi-Agent Reinforcement Learning (MARL)** 中策略表达能力不足的问题，提出了一种新的框架来提升多智能体系统的协调效率。传统 MARL 方法（如 MADDPG、MAPPO）通常采用单峰分布（如 Gaussian）作为策略，难以建模复杂的、多模态的协作行为。而尽管 **diffusion policy** 在离线设置中展现出强大的表达力，但由于其 **不可计算的似然（intractable likelihoods）**，导致无法直接用于基于熵正则化的在线探索机制。

因此，核心挑战在于：
- 如何在 **online MARL** 中有效利用 diffusion policy 的高表达能力；
- 如何克服因似然不可计算而导致的 **entropy-based exploration 难以实现** 的问题；
- 如何在 **CTDE（Centralized Training with Decentralized Execution）** 范式下实现稳定协调。

---

### 🚀 提出的新方法/新思路

作者提出了 **OMAD（Online off-policy MARL framework using Diffusion policies）**，这是首个专为在线多智能体场景设计的 diffusion policy 框架，其核心创新如下：

#### （1）**可计算的最大熵目标（Tractable Maximum Entropy Objective）**
- 引入了一个基于 **scaled joint entropy** 的松弛目标函数，通过变分推断（variational inference）推导出一个 **可计算的联合熵下界（Evidence Lower Bound, ELBO）**。
- 利用该 ELBO 替代不可计算的真实熵，实现了对 diffusion policy 的 **entropy-regularized 探索**，从而避免过早收敛到次优解。

#### （2）**集中式分布值函数引导（Centralized Distributional Critic）**
- 在 CTDE 框架下，构建了一个 **joint distributional value function $Z(s,a)$** 来建模回报的完整分布，而非仅期望值。
- 该 critic 提供更丰富的监督信号，帮助去耦合多个 diffusion policy 之间的随机性干扰，提升训练稳定性。

#### （3）**同步更新机制（Synchronized Policy Updates）**
- 所有 agent 的 diffusion policy 在统一的目标下进行联合优化，确保协调一致性。
- 与独立学习（independent learning）相比，显著缓解了非平稳性（non-stationarity）问题。

#### （4）**温度自动调参（Auto-Tuning of Entropy Coefficient）**
- 将温度参数 $\alpha$ 的调整建模为一个双重优化问题，动态维持 ELBO 在目标阈值之上，无需手动调节超参数。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（如 HATD3/HASAC） | OMAD |
|------|----------------------------|-------|
| 策略表达能力 | 单峰高斯策略，表达受限 | 多模态 diffusion policy，表达能力强 |
| 探索机制 | 依赖噪声注入或固定方差 | 基于可计算 ELBO 的熵正则化探索 |
| 协调机制 | 分散损失函数，易受非平稳影响 | 全局分布 critic + 同步更新，强协调 |
| 学习范式 | 多为 on-policy 或标准 off-policy | 高效 off-policy 学习，样本利用率高 |
| 自动化程度 | 温度需手动调参 | 支持 $\alpha$ 自动调优 |

> 💡 **核心优势总结**：OMAD 成功将 diffusion model 的强大生成能力引入 **online MARL**，并通过理论创新解决了“不可计算似然”带来的探索难题，在保持去中心化执行的同时实现了高效、稳定的多智能体协调。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

实验在两个主流连续控制多智能体基准上进行：

- **Multi-Agent Particle Environments (MPE)**  
  包括：
  - Cooperative Navigation（3/4 agents）
  - Physical Deception（2 agents）

- **Multi-Agent MuJoCo (MAMuJoCo)**  
  将单智能体 MuJoCo 机器人任务的动作空间拆分为多个子代理控制，测试复杂动力学下的协调能力，包括：
  - Ant（2×4, 2×4d, 4×2）
  - HalfCheetah（2×3, 6×1）
  - Walker2d（2×3）
  - Swimmer（2×1）

---

### ⚙️ 实验设置与评估指标

#### 实验配置
- **训练模式**：off-policy, CTDE
- **回放缓冲区大小**：1,000,000
- **Batch Size**：256
- **Diffusion Denoise Steps**：8
- **Distributional Q 参数**：100 atoms, $V_{\text{max}}$ 根据任务设定（见附录）
- **评估方式**：每轮训练后评估 10 个 episode 的平均 return，报告 5 个随机种子的均值 ± 标准差

#### 评估指标
- 主要指标：**episode return（越高越好）**
- 关键比较维度：
  - 收敛速度（sample efficiency）
  - 最终性能（asymptotic performance）
  - 方差稳定性
  - 状态空间覆盖范围（state coverage）

---

### 🆚 基线方法对比

| 类型 | 方法 |
|------|------|
| **SOTA Online MARL** | HATD3, HASAC |
| **扩散策略扩展版** | MADPMD, MASDAC（将单智能体 diffusion policy 直接推广至多智能体） |

> 注：MADPMD 和 MASDAC 是作者基于现有 diffusion policy 工作（如 DPMD/SDAC）构建的多智能体版本，作为 diffusion 方法的代表基线。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 任务 | OMAD 性能（Return） | 最佳 Baseline 性能 | 提升幅度 |
|------|---------------------|--------------------|----------|
| Cooperative Navigation (N=3) | **-23.9±1.1** | -25.1 (HASAC) | ↑ 显著更快收敛 |
| Cooperative Navigation (N=4) | **-57.9±0.8** | -63.0 (HATD3) | ↑ 收敛快 5× |
| Physical Deception (N=2) | **45.1±4.1** | 45.4 (HASAC) | 相当但更稳定 |
| Ant 2×4 | **7517.0±279.2** | 7266.2 (MASDAC) | ↑ +3.5% |
| HalfCheetah 2×3 | **14368.5±1166.0** | 10540.7 (MASDAC) | ↑ +36% |
| Swimmer 2×1 | **162.0±22.5** | 149.3 (MASDAC) | ↑ +8.5% |

> ✅ **总体表现**：OMAD 在 **10 个不同任务** 上均达到 **SOTA（state-of-the-art）水平**，尤其在高维连续控制任务中优势明显。

---

### 🔁 与基线方法的对比结果

- **样本效率提升显著**：
  - 达到相同性能所需训练步数减少 **2.5× 至 5×**
  - 例如在 Cooperative Navigation N=4 中，OMAD 仅用 1/5 步数即超越 baseline 峰值性能

- **最终性能更高**：
  - 在 HalfCheetah、Ant 等任务上大幅领先，表明其更强的探索与协调能力

- **状态空间探索更广**：
  - 在 Ant 2×4 任务中，前 250k 步的状态覆盖率：
    - OMAD：**68.3%**（934 bins）
    - HASAC：55.0%（753 bins）
    - HATD3：48.4%（662 bins）
  - OMAD 探索到了大量其他方法未触及的状态区域（图中橙色部分），说明其具备更强的全局搜索能力

- **扩散策略直接迁移效果有限**：
  - MADPMD / MASDAC 虽然也使用 diffusion policy，但缺乏有效的 centralized value guidance 和同步更新机制，导致收敛慢或性能不佳

---

### 🔍 消融实验结果（Ablation Study）

#### （1）分布值函数超参数（Vmax 和 atom 数量）
- $V_{\text{max}} < 1000$ 时性能严重下降（截断效应）
- 原子数量 ≥100 后性能趋于饱和 → 选择 **100 atoms**

#### （2）去噪步数（Denoising Steps）
- 8 步即可达到 12/16 步的性能上限
- 计算成本随步数线性增长 → 选择 **8 步** 实现最佳性价比

#### （3）熵系数 $\alpha$ 是否自动调优
- 固定 $\alpha=0.1$：过度随机，训练不稳定
- 固定小值（如 0.001）：虽可达高性能，但需精细调参
- **Auto-tuning**：动态调节 $\alpha$，匹配最优固定值性能，且无需人工干预 → 验证了自动调参的有效性和鲁棒性

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Diffusion policy 可成功应用于 online MARL**  
   通过设计可计算的 ELBO 替代真实熵，首次实现了 diffusion policy 在 online MARL 中的有效 entropy regularization。

2. **集中式分布 critic 对协调至关重要**  
   分布式价值函数能更好地区分真实协调信号与策略随机性，提供更稳健的学习目标。

3. **同步更新 + 共享目标 提升协调效率**  
   所有 agent 在统一目标下联合优化，显著优于分散式局部损失。

4. **OMAD 实现 SOTA 表现**  
   在 MPE 和 MAMuJoCo 上全面超越现有方法，样本效率提升 **2.5×–5×**，验证了其高效性与通用性。

---

### ⚠️ 方法的局限性

- **计算开销相对较高**：diffusion policy 需要多次去噪迭代（即使 H=8），推理延迟高于传统策略网络。
- **依赖 centralized critic**：虽然执行是去中心化的，但训练阶段需要全局状态和动作信息，限制了完全分布式部署。
- **理论近似误差未知**：ELBO 是真实熵的下界，其近似偏差在训练过程中难以量化。
- **目前仅适用于连续动作空间**：尚未拓展到 discrete action space（如 discrete diffusion models）。

---

### 🔮 未来工作方向

1. **加速 diffusion inference**  
   探索更高效的采样器（如 DDIM、DPM-Solver）以降低延迟。

2. **扩展至离散动作空间**  
   结合 discrete diffusion models（如 MaskGIT-style）处理 discrete MARL 任务。

3. **轻量化架构设计**  
   设计更适合多智能体部署的 factorized diffusion 结构，进一步提升效率。

4. **理论分析 ELBO 偏差边界**  
   推导 tighter bounds 或研究收敛时 approximation error 的渐进行为。

5. **结合 communication learning**  
   在隐变量层面引入通信机制，增强 agent 间的显式协调能力。

---

## ✅ 总结一句话

> **OMAD 是首个成功将 diffusion policy 引入 online MARL 的框架，通过可计算的熵下界与集中式分布 critic，实现了高效探索与稳定协调，在多个基准上取得 SOTA 表现，样本效率提升达 2.5–5 倍。**

</details>

---

### 15. [Joint Training on AMD and NVIDIA GPUs](https://arxiv.org/abs/2602.18007)

**Authors**: Jon Hu, Thomas Jia, Jing Zhu, Zhendong Yu  
**Category**: cs.DC  
**Published**: 2026-02-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.18007v1  

#### Abstract
As large language models continue to scale, training demands on compute and system capacity grow rapidly, making single-vendor homogeneous clusters insufficient. This paper presents a technical solution for heterogeneous mixed training in AMD-NVIDIA environments. We first adopt a compatibility-orien...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Joint Training on AMD and NVIDIA GPUs》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着大语言模型（LLM）参数规模突破万亿级，单一厂商的同构计算集群已难以满足训练需求。现代数据中心普遍存在硬件异构性（如同时部署 **NVIDIA** 和 **AMD** GPU），但由于缺乏高效的跨厂商通信机制，这些异构资源难以被有效整合用于大规模模型训练。

本文旨在解决 **AMD 与 NVIDIA GPU 在同一集群中协同训练 LLM** 的技术难题，特别是突破跨厂商 GPU 间高开销的数据传输瓶颈。

---

### 🚀 提出的新方法与创新思路

论文提出了两种混合训练方案，并重点发展了一种高性能的 **Device-Direct Communication** 方法：

#### （1）CPU-Forwarding Communication（兼容性优先）
- 基于 **Gloo** 实现跨厂商通信，通过 CPU 中转数据。
- 引入两项优化：
  - **Differentiated Communication Backend Selection (DCBS)**：在不同并行组使用不同的通信后端。例如：
    - Pipeline Parallel (PP) 组使用 Gloo 支持异构互联；
    - Data Parallel (DP) / Tensor Parallel (TP) 组仍用原生高性能库（如 NCCL、RCCL）。
  - **Multi-NIC Parallel Data Transfer (MPDT)**：为每个 GPU 分配独立 NIC 进行 P2P 数据传输，提升带宽利用率。

> ✔️ 优势：实现基本跨厂商互通，兼容性强  
> ❌ 缺陷：频繁的 Host-to-Device (H2D/D2H) 内存拷贝导致性能瓶颈

#### （2）Device-Direct Communication（性能导向，核心创新）
- **提出一种 CPU-offloading P2P 机制**，实现跨厂商 GPU 之间的直接数据传输，完全绕过主机内存（host-memory staging）。
- 核心设计：
  - 控制平面（Control Plane）运行在 CPU 上，负责连接管理、事件同步等；
  - 数据平面（Data Plane）全程驻留设备侧，利用 **GPUDirect RDMA (GDR)** 技术将数据从源 GPU → NIC → 目标 GPU，避免中间内存复制。
- 构建多适配器架构（Multi-Adapter Architecture）抽象硬件差异：
  - `Device Adaptor`：封装 CUDA / ROCm 接口
  - `Net-Plugin Adaptor`：统一 ibverbs 接口支持 RDMA
  - `CCL Adaptor`：标准化 NCCL / RCCL 调用接口

> ✅ 创新点总结：
> - 首次实现 **AMD 与 NVIDIA GPU 间的设备直连通信**
> - 提出 **控制/数据平面解耦** 架构，兼顾兼容性与性能
> - 通过 PyTorch backend 插件形式集成，对上层框架（如 Megatron、DeepSpeed）透明

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法（如 Gloo） | 本文 Device-Direct 方法 |
|------|------------------------|----------------------------|
| 通信路径 | Host 中转（H2D + D2H） | Device-Direct（无 host copy） |
| 性能 | 受限于 PCIe 带宽和 CPU 开销 | 接近同构系统水平 |
| 兼容性 | 高（通用） | 高（通过适配层抽象） |
| 扩展性 | 仅适用于低频通信场景（如 PP） | 支持高频集体通信（AllReduce 等） |

---

## 2. 核心实验方法和设置

### 📊 使用的模型（Workloads）
- **LLaMA-8B**
- **Qwen2-7B**

> 注：未使用传统 NLP 数据集，而是以完整 LLM 预训练流程作为 workload，更具实际意义。

---

### ⚙️ 实验设置

#### 硬件配置
| 节点类型 | GPU 数量 | GPU 型号 | 互联技术 | 带宽 |
|---------|----------|----------|----------|-------|
| NVIDIA Node | 8× | H200 | NVLink | 900 GB/s |
| AMD Node | 8× | Instinct MI325X | Infinity Fabric | 128 GB/s |
| 网络互联 | 每节点 8× | BlueField-3 DPU | RDMA over RoCE | 100 GB/s per DPU |

> 异构环境 = NVIDIA Node + AMD Node；同构环境分别为单个节点。

---

#### 并行策略（Parallelization Configuration）
| 参数 | 设置 |
|------|------|
| Tensor Parallelism (TP) | 1 |
| Pipeline Parallelism (PP) | 2 |
| Data Parallelism (DP) | 4（同构）、8（异构） |

> 异构环境下 DP=8 表示跨两个厂商节点进行数据并行。

---

#### 层划分策略（Layer Partitioning）
由于 AMD GPU 吞吐较低，采用非均匀分配以平衡 pipeline 负载：

| 模型 | AMD Stage | NVIDIA Stage |
|------|-----------|--------------|
| LLaMA-8B | 15 layers | 17 layers |
| Qwen2-7B | 12 layers | 16 layers |

---

### 📈 评估指标
1. **Training Throughput**（tokens/sec）—— 主要性能指标
2. **Throughput Stability** over 500 iterations
3. **Loss Convergence Curve** —— 验证数值正确性
4. 对比基线包括：
   - NVIDIA-Homo（NVIDIA 单节点）
   - AMD-Homo（AMD 单节点）
   - Global Gloo-Hetero（原始 Megatron-Gloo 异构方案）
   - DCBS-Hetero
   - DCBS&MPDT-Hetero

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Figure 3）

| 方法 | LLaMA-8B 吞吐 | Qwen2-7B 吞吐 |
|------|----------------|----------------|
| NVIDIA-Homo | 557.4 tokens/sec | 526.4 tokens/sec |
| AMD-Homo | 534.9 tokens/sec | 514.5 tokens/sec |
| **Proposed (Device-Direct)** | **549.7** tokens/sec (**98.2%** of NVIDIA-Homo) | **500.0** tokens/sec (**94.4%** of NVIDIA-Homo) |

> ✅ 在 LLaMA-8B 上达到 **NVIDIA 同构系统的 98.2% 吞吐**，接近理论极限！

---

### 🔁 与基线方法对比

| 对比项 | 结果 |
|--------|------|
| vs. AMD-Homo | 分别快 **2.8%**（LLaMA）和 **6.9%**（Qwen） |
| vs. DCBS&MPDT-Hetero | 显著优于该优化版 Gloo 方案，证明 host-copy 是主要瓶颈 |
| vs. Native Gloo | 性能差距巨大，凸显 Device-Direct 的必要性 |

> 💡 图 3 显示：`Device-Direct > DCBS&MPDT > DCBS > Gloo`，性能逐级提升

---

### 🔍 消融实验分析（隐含在结果中）

虽然没有显式列出消融表，但从方法演进可看出以下结论：

| 组件 | 贡献 |
|------|------|
| DCBS | 提升 PP 通信效率，避免全系统降级至最慢后端 |
| MPDT | 提高跨节点带宽利用率 |
| Device-Direct + GDR | **最大增益来源**，消除 host-memory copy 开销，释放硬件潜力 |

> 实验表明：仅靠软件调度优化（DCBS+MPDT）无法突破性能天花板，必须依赖底层通信机制革新。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **异构混合训练可行且高效**：
   - 基于 **Device-Direct Communication** 的方案可在 AMD-NVIDIA 混合集群上稳定训练 LLM。
   - 达到 **NVIDIA 同构系统 94–98% 的吞吐性能**，几乎无性能损失。

2. **稳定性与正确性得到验证**：
   - 图 4 显示训练过程中吞吐稳定，无抖动或崩溃；
   - 图 5 显示 loss 曲线与同构系统高度重合，说明梯度同步准确，训练动态一致。

3. **合理的模型分区至关重要**：
   - 将更多 layer 分配给高性能 NVIDIA GPU 可缓解 pipeline imbalance；
   - 不当分区可能导致异构训练反而不如最慢单节点。

4. **Pipeline Parallelism 是当前最适合引入异构性的维度**：
   - 因其通信模式简单（主要是 P2P），易于跨厂商实现；
   - 若扩展至 DP/TP 组内集体通信，需更精细的算法适配。

---

### ⚠️ 方法的局限性

1. **目前异构性仅限于 PP 组**：
   - 尚未在 TP 或 DP 组内部署跨厂商执行，因集体通信（AllReduce）对拓扑敏感。
   
2. **工程复杂度高**：
   - 尽管接口相似（NCCL ≈ RCCL），但在实践中遇到大量 hang、死锁等问题，需深度调试才能稳定运行。
   - 依赖特定硬件支持（如 GDR、RDMA-capable DPU）。

3. **尚未支持更多厂商或架构**：
   - 当前仅覆盖 AMD 与 NVIDIA，向 Intel GPU 或其他加速器扩展尚待研究。

---

### 🔮 未来工作方向

1. **将 Device-Direct 机制推广至 DP/TP 组**，实现全栈异构并行。
2. **自动化模型分区策略**：基于实时 profiling 动态调整 layer 分布。
3. **构建统一的异构训练 runtime**，进一步降低部署门槛。
4. **支持更多硬件平台**（如 Intel Habana、Apple Silicon）形成 truly heterogeneous AI infrastructure。

---

## ✅ 总结一句话

> 本论文提出 **Device-Direct Communication** 架构，首次实现了 **AMD 与 NVIDIA GPU 间的高效联合训练**，在 LLaMA-8B 和 Qwen2-7B 上达到了 **接近 NVIDIA 同构系统 98% 的吞吐性能**，同时保证了训练的稳定性与数值正确性，为未来异构算力融合提供了实用的技术路径。

</details>

---

### 16. [Joint Parameter and State-Space Bayesian Optimization: Using Process Expertise to Accelerate Manufacturing Optimization](https://arxiv.org/abs/2602.17679)

**Authors**: Saksham Kiroriwal, Julius Pfrommer, J\"urgen Beyerer  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.17679v1  

#### Abstract
Bayesian optimization (BO) is a powerful method for optimizing black-box manufacturing processes, but its performance is often limited when dealing with high-dimensional multi-stage systems, where we can observe intermediate outputs. Standard BO models the process as a black box and ignores the inte...

---

### 17. [BioBridge: Bridging Proteins and Language for Enhanced Biological Reasoning with LLMs](https://arxiv.org/abs/2602.17680)

**Authors**: Yujia Wang, Jihong Guan, Wengen Li, Shuigeng Zhou, Xuhong Wang  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.17680v1  

#### Abstract
Existing Protein Language Models (PLMs) often suffer from limited adaptability to multiple tasks and exhibit poor generalization across diverse biological contexts. In contrast, general-purpose Large Language Models (LLMs) lack the capability to interpret protein sequences and fall short in domain-s...

---

### 18. [Multi-material Multi-physics Topology Optimization with Physics-informed Gaussian Process Priors](https://arxiv.org/abs/2602.17783)

**Authors**: Xiangyu Sun, Shirin Hosseinmardi, Amin Yousefpour, Ramin Bostanabad  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.17783v1  

#### Abstract
Machine learning (ML) has been increasingly used for topology optimization (TO). However, most existing ML-based approaches focus on simplified benchmark problems due to their high computational cost, spectral bias, and difficulty in handling complex physics. These limitations become more pronounced...

---

### 19. [Influence-Preserving Proxies for Gradient-Based Data Selection in LLM Fine-tuning](https://arxiv.org/abs/2602.17835)

**Authors**: Sirui Chen, Yunzhe Qi, Mengting Ai, Yifan Sun, Ruizhong Qiu, Jiaru Zou, Jingrui He  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.17835v1  

#### Abstract
Supervised fine-tuning (SFT) relies critically on selecting training data that most benefits a model's downstream performance. Gradient-based data selection methods such as TracIn and Influence Functions leverage influence to identify useful samples, but their computational cost scales poorly, makin...

---

### 20. [El Agente Gr\'afico: Structured Execution Graphs for Scientific Agents](https://arxiv.org/abs/2602.17902)

**Authors**: Jiaru Bai, Abdulrahman Aldossary, Thomas Swanick, Marcel M\"uller, Yeonghun Kang, Zijian Zhang, Jin Won Lee, Tsz Wai Ko, Mohammad Ghazi Vakili, Varinia Bernales, Al\'an Aspuru-Guzik  
**Category**: cs.AI  
**Published**: 2026-02-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.17902v1  

#### Abstract
Large language models (LLMs) are increasingly used to automate scientific workflows, yet their integration with heterogeneous computational tools remains ad hoc and fragile. Current agentic approaches often rely on unstructured text to manage context and coordinate execution, generating often overwh...

---

### 21. [Analyzing LLM Instruction Optimization for Tabular Fact Verification](https://arxiv.org/abs/2602.17937)

**Authors**: Xiaotang Du, Giwon Hong, Wai-Chung Kwan, Rohit Saxena, Ivan Titov, Pasquale Minervini, Emily Allaway  
**Category**: cs.CL  
**Published**: 2026-02-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.17937v1  

#### Abstract
Instruction optimization provides a lightweight, model-agnostic approach to enhancing the reasoning performance of large language models (LLMs). This paper presents the first systematic comparison of instruction optimization, based on the DSPy optimization framework, for tabular fact verification. W...

---

### 22. [A Probabilistic Framework for LLM-Based Model Discovery](https://arxiv.org/abs/2602.18266)

**Authors**: Stefan Wahl, Raphaela Schenk, Ali Farnoud, Jakob H. Macke, Daniel Gedon  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.18266v1  

#### Abstract
Automated methods for discovering mechanistic simulator models from observational data offer a promising path toward accelerating scientific progress. Such methods often take the form of agentic-style iterative workflows that repeatedly propose and revise candidate models by imitating human discover...

---

### 23. [GPU Memory and Utilization Estimation for Training-Aware Resource Management: Opportunities and Limitations](https://arxiv.org/abs/2602.17817)

**Authors**: Ehsan Yousefzadeh-Asl-Miandoab, Reza Karimzadeh, Danyal Yorulmaz, Bulat Ibragimov, P{\i}nar T\"oz\"un  
**Category**: cs.DC  
**Published**: 2026-02-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.17817v1  

#### Abstract
Collocating deep learning training tasks improves GPU utilization but causes drastic slowdowns due to resource contention and risks Out-of-Memory (OOM) failures. Accurate memory estimation is essential for robust collocation, while GPU utilization -- a key proxy for resource contention -- enables in...

---

### 24. [Asking Forever: Universal Activations Behind Turn Amplification in Conversational LLMs](https://arxiv.org/abs/2602.17778)

**Authors**: Zachary Coalson, Bo Fang, Sanghyun Hong  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.17778v1  

#### Abstract
Multi-turn interaction length is a dominant factor in the operational costs of conversational LLMs. In this work, we present a new failure mode in conversational LLMs: turn amplification, in which a model consistently prolongs multi-turn interactions without completing the underlying task. We show t...

---

### 25. [Calibrated Adaptation: Bayesian Stiefel Manifold Priors for Reliable Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2602.17809)

**Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.17809v1  

#### Abstract
Parameter-efficient fine-tuning methods such as LoRA enable practical adaptation of large language models but provide no principled uncertainty estimates, leading to poorly calibrated predictions and unreliable behavior under domain shift. We introduce Stiefel-Bayes Adapters (SBA), a Bayesian PEFT f...

---

### 26. [Continual-NExT: A Unified Comprehension And Generation Continual Learning Framework](https://arxiv.org/abs/2602.18055)

**Authors**: Jingyang Qiao, Zhizhong Zhang, Xin Tan, Jingyu Gong, Yanyun Qu, Yuan Xie  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.18055v1  

#### Abstract
Dual-to-Dual MLLMs refer to Multimodal Large Language Models, which can enable unified multimodal comprehension and generation through text and image modalities. Although exhibiting strong instantaneous learning and generalization capabilities, Dual-to-Dual MLLMs still remain deficient in lifelong e...

---

### 27. [Advection-Diffusion on Graphs: A Bakry-Emery Laplacian for Spectral Graph Neural Networks](https://arxiv.org/abs/2602.18141)

**Authors**: Pierre-Gabriel Berlureau, Ali Hariri, Victor Kawasaki-Borruat, Mia Zosso, Pierre Vandergheynst  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.18141v1  

#### Abstract
Graph Neural Networks (GNNs) often struggle to propagate information across long distances due to oversmoothing and oversquashing. Existing remedies such as graph transformers or rewiring typically incur high computational cost or require altering the graph structure. We introduce a Bakry-Emery grap...

---

### 28. [Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](https://arxiv.org/abs/2602.18025)

**Authors**: Haruki Abe, Takayuki Osa, Yusuke Mukuta, Tatsuya Harada  
**Category**: cs.AI  
**Published**: 2026-02-23  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.18025v1  

#### Abstract
Scalable robot policy pre-training has been hindered by the high cost of collecting high-quality demonstrations for each platform. In this study, we address this issue by uniting offline reinforcement learning (offline RL) with cross-embodiment learning. Offline RL leverages both expert and abundant...

---

### 29. [Provable Adversarial Robustness in In-Context Learning](https://arxiv.org/abs/2602.17743)

**Authors**: Di Zhang  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.17743v1  

#### Abstract
Large language models adapt to new tasks through in-context learning (ICL) without parameter updates. Current theoretical explanations for this capability assume test tasks are drawn from a distribution similar to that seen during pretraining. This assumption overlooks adversarial distribution shift...

---

### 30. [Grassmannian Mixture-of-Experts: Concentration-Controlled Routing on Subspace Manifolds](https://arxiv.org/abs/2602.17798)

**Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.17798v1  

#### Abstract
Mixture-of-Experts models rely on learned routers to assign tokens to experts, yet standard softmax gating provides no principled mechanism to control the tradeoff between sparsity and utilization. We propose Grassmannian MoE (GrMoE), a routing framework that operates on the Grassmannian manifold of...

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

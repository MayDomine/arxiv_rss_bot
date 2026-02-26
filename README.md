# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-26 06:43:10 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DHP: Efficient Scaling of MLLM Training with Dynamic Hybrid Parallelism](https://arxiv.org/abs/2602.21788)

**Authors**: Yifan Niu, Han Xiao, Dongyi Liu, Wei Zhou, Jia Li  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2602.21788v1  

#### Abstract
Scaling long-context capabilities is crucial for Multimodal Large Language Models (MLLMs). However, real-world multimodal datasets are extremely heterogeneous. Existing training frameworks predominantly rely on static parallelism strategies, which suffer from severe load imbalance, redundant communi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DHP: Efficient Scaling of MLLM Training with Dynamic Hybrid Parallelism》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前多模态大语言模型（**MLLMs**）在训练过程中面临以下挑战：
- **数据异构性强**：真实世界中的多模态数据（如视频、图像）具有高度不均衡的序列长度分布（长尾分布），例如大多数视频短于8秒，少数超过64秒。
- **静态并行策略效率低下**：主流框架（如 **Megatron-LM**, **DeepSpeed**）采用固定的并行配置（如 TP/PP/DP/SP/CP），导致：
  - 负载不平衡（workload imbalance）
  - 冗余通信开销（redundant communication）
  - 硬件利用率低（poor hardware utilization）

这些问题严重制约了大规模 MLLM 在复杂异构数据上的训练效率。

---

### 🚀 提出的新方法：Dynamic Hybrid Parallelism (DHP)

DHP 是一种**动态混合并行策略**，其核心思想是：
> 在每个 micro-batch 上**自适应地重构通信组（communication groups）和并行度（parallelism degrees）**，以匹配当前批次的数据分布。

#### 主要创新点：

1. **支持非2的幂次并行度（Non-power-of-two Parallelism Degrees）**
   - 传统方法（如 Ulysses-style SP）要求并行度必须整除注意力头数，通常限制为 2^n。
   - DHP 基于 **Ring-style Context Parallelism (CP)**，允许任意整数作为并行度（如 3, 5, 6），实现更细粒度的资源分配。

2. **两阶段近似最优调度算法（Two-stage Approximation Algorithm）**
   - **Stage 1: Memory-aware Sequence Packing (Best-Fit Decreasing)**
     - 将异构序列按内存需求降序排列；
     - 使用贪心策略将短序列打包进长序列预留的“内存桶”中，减少决策变量数量。
   - **Stage 2: 2D-Dynamic Programming Resource Allocation**
     - 对原子组进行动态规划求解最优 CP 并行度与分组方案；
     - 时间复杂度为 $O(K'N^2)$，仅引入毫秒级开销。

3. **极低调度延迟设计（Minimal Scheduling Overhead）**
   - 调度过程完全异步化：CPU 执行调度时，NPU 继续处理上一批次；
   - 利用 **Profiler 预建成本模型**，快速预测不同配置下的执行时间；
   - 实现调度开销与计算完全重叠（fully overlapped），避免成为瓶颈。

---

### 🔍 相比现有方法的优势

| 特性 | Megatron-LM / DeepSpeed | DHP |
|------|------------------------|-----|
| 并行策略 | 静态（Static） | 动态（Dynamic） |
| 并行度灵活性 | 限于 power-of-two | 支持任意整数 |
| 负载均衡能力 | 差（固定分组） | 强（自适应调整） |
| 通信冗余 | 存在（短序列过度并行） | 显著降低 |
| 调度开销 | 无（预设） | <1秒，可隐藏 |
| 适用场景 | 同构序列 | 异构/长尾数据 |

> ✅ DHP 在保持系统兼容性的同时，显著提升了训练吞吐量和扩展性。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 描述 |
|-------|------|
| **InternVid** | 包含1000万视频片段，高质量自动生成字幕，用于大规模视频-语言预训练 |
| **OpenVid** | 高审美质量视频，最小分辨率 512×512，适合高保真文生视频任务 |
| **MSRVTT** | 1万个视频 + 20万自然语言描述，涵盖20个类别，常用于跨模态理解评测 |

> 这些数据集均表现出明显的**长尾序列长度分布**，验证了 DHP 在现实场景中的必要性。

---

### ⚙️ 实验设置

- **硬件平台**：
  - 最多 8 节点集群
  - 每节点 8 块 Ascend910B NPU（共 64GB 显存）
  - 节点内通过 HCCS 互联，节点间使用 100Gbps InfiniBand

- **模型规模**：
  - 测试模型从 **2B 到 8B 参数** 不等
  - 具体包括：`InternVL3-2B/4B/8B`, `Qwen3VL-2B/4B/8B`

- **训练配置**：
  - 固定全局 batch size = 512
  - 微批次（micro-batch）由调度器动态划分
  - TP 和 PP 设为静态（不可变），仅动态优化 CP 和隐式的 DP 分组

---

### 📊 评估指标

| 指标 | 定义 |
|------|------|
| **端到端迭代时间（End-to-end Iteration Time）** | 单次训练 step 的平均耗时（秒） |
| **每设备 token 吞吐量（Token Throughput per-device）** | 每秒处理的 token 数量（k tokens/s） |
| **加速比（Speedup Ratio）** | 相对于最佳 baseline 的性能提升倍数 |
| **扩展效率（Scaling Efficiency）** | 随 NPU 数量增加，吞吐量的增长趋势 |

---

### 🆚 基线方法对比

| 方法 | 简介 |
|------|------|
| **Megatron-LM** | 支持 4D 并行（TP/PP/DP/CP），使用 Megatron-style SP 或 Ring-Attention |
| **DeepSpeed (Ulysses-style SP)** | 基于 All-to-All 的序列并行，适用于长上下文，但并行度受限于注意力头数 |

> 两者均为当前主流开源框架，且默认使用单一静态并行策略。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 端到端性能加速（Figure 6）

| 模型 | 数据集 | 加速比（vs 最佳 baseline） |
|------|--------|--------------------------|
| InternVL3-8B | OpenVid | **1.35×** |
| InternVL3-8B | MSRVTT | 1.14× |
| Qwen3VL-8B | OpenVid | 1.32× |
| InternVL2.5-4B | OpenVid | 1.28× |

> - 在所有 18 种配置中，有 **14 种实现 >1.2× 加速**
> - 模型越大、数据越复杂（如 OpenVid），增益越明显

#### ✅ 扩展性分析（Figure 5）

| 方法 | 从 8 → 64 NPU 的吞吐变化 |
|------|----------------------------|
| **DHP** | 从 ~1.8 → **~2.1 k tokens/s**（轻微上升） |
| DeepSpeed | 从 ~2.0 → 1.7 k tokens/s（下降） |
| Megatron-LM | 从 ~1.8 → 1.52 k tokens/s（下降） |

> ✅ DHP 展现出**接近线性的扩展效率**，甚至随规模增大略有提升，优于基线。

---

### ⏱️ 调度开销实测（Tables 1 & 2）

| 设置 | 调度总时间 | Solver 时间 | 计算时间 |
|------|-----------|------------|---------|
| GBS=128 | 468 ms | 21 ms | 2.04 s |
| GBS=512 | 921 ms | 86 ms | 7.32 s |
| NPU=64 (GBS=512) | 921 ms | 86 ms | 7.32 s |

> - **Solver 时间始终 ≤86ms**，远低于单步计算时间
> - 调度开销可被完全掩盖在计算中，不影响整体性能

---

### 🎯 成本估计误差（Table 3）

| 模型 | 2B 参数误差 | 4B 参数误差 | 8B 参数误差 |
|------|-------------|-------------|-------------|
| Qwen3VL | 7.93% | 6.71% | 4.27% |
| InternVL3 | 7.48% | 6.54% | 4.12% |

> - 成本预测误差随模型增大而减小，说明 Profiler 在大模型下更准确
> - 平均误差 <8%，足以支撑高效调度决策

---

### 🔍 消融研究：案例分析（Table 4）

| 场景 | 数据特征 | DHP 动态策略 | 性能增益 |
|------|----------|---------------|----------|
| **Case 1 (OpenVid)** | 极度异构、长尾分布 | 使用多种 CP 度（8×1, 6×2, 4×1, 2×2, 1×4） | **1.17×** |
| **Case 2 (MSRVTT)** | 相对均匀但仍异构 | 更一致的 CP 度（4×2, 3×4, 2×6） | 1.14× |

> - DHP 能根据实际数据分布灵活选择并行配置
> - 避免了静态方法对所有序列强制使用最大并行度带来的浪费

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **动态并行是应对多模态数据异构性的关键路径**
   - 静态并行在真实世界数据中存在严重负载失衡和通信冗余；
   - DHP 通过动态重构通信组，显著缓解上述问题。

2. **非2的幂次并行度带来实质性收益**
   - 允许任意整数并行度使资源分配更精细，尤其利于短序列；
   - 结合 Ring-Attention 可有效隐藏通信开销。

3. **低延迟调度可行且必要**
   - 通过异步执行 + Profiler 预估，调度开销可完全隐藏；
   - 实现“零感知”的动态优化体验。

4. **DHP 具备强泛化性和可扩展性**
   - 在不同模型（2B–8B）、数据集（MSRVTT/OpenVid/InternVid）、集群规模（8–64 NPUs）下均表现优异；
   - 维持近线性扩展效率，优于现有框架。

---

### ⚠️ 方法的局限性

1. **TP 和 PP 仍为静态配置**
   - 当前未支持运行时动态切换 TP/PP 拓扑（因权重重分布代价过高）；
   - 未来若能结合轻量级参数迁移机制，可能进一步提升灵活性。

2. **依赖精确的成本建模**
   - Profiler 需要在训练初期完成 profiling；
   - 若模型架构或硬件环境发生重大变更，需重新校准。

3. **通信组缓存机制有一定内存开销**
   - 虽然实践中唯一组数量有限，但在极端动态场景下可能影响稳定性。

---

### 🔮 未来工作方向

1. **支持全动态并行（Full Dynamic Parallelism）**
   - 探索低开销的 TP/PP 动态重组技术（如基于 MoE 或稀疏激活）；

2. **跨 batch 的历史信息利用**
   - 引入强化学习或在线学习策略，基于历史调度效果持续优化未来决策；

3. **向其他模态扩展**
   - 将 DHP 推广至音频、3D 点云等更多模态的异构训练场景；

4. **集成到自动并行编译器**
   - 与 **MindSpore**、**PyTorch Distributed Compiler** 等系统深度整合，实现全自动并行策略生成。

---

## ✅ 总结

> **DHP 提出了一种面向异构多模态数据的高效动态混合并行框架，在不修改底层训练系统的基础上，实现了毫秒级调度、任意并行度支持和近线性扩展性。实验表明其相较 Megatron-LM 和 DeepSpeed 最高可达 1.36× 的训练吞吐加速，是推动 MLLM 规模化训练的重要进展。**

</details>

---

### 2. [C$^{2}$TC: A Training-Free Framework for Efficient Tabular Data Condensation](https://arxiv.org/abs/2602.21717)

**Authors**: Sijia Xu, Fan Li, Xiaoyang Wang, Zhengyi Yang, Xuemin Lin  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.21717v1  

#### Abstract
Tabular data is the primary data format in industrial relational databases, underpinning modern data analytics and decision-making. However, the increasing scale of tabular data poses significant computational and storage challenges to learning-based analytical systems. This highlights the need for ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：C²TC: A Training-Free Framework for Efficient Tabular Data Condensation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现有的 **Dataset Condensation (DC)** 方法在处理**表格数据（tabular data）**时面临三大挑战：

- **高计算成本（High computational cost）**：主流方法依赖复杂的梯度优化（如 gradient matching 或 trajectory matching），需要反复训练模型，效率极低。
- **类别不平衡下的效用损失（Utility loss under class imbalance）**：传统标签分配策略（如 Ratio 或 FIPC）无法动态适应类别分布，导致少数类性能下降。
- **异构特征的信息丢失（Information loss in heterogeneous features）**：现有方法对分类型（categorical）特征编码不当，导致语义信息丢失。

此外，表格数据具有**异构特征（numerical 和 categorical）、样本独立性、无图结构**等特点，不能直接套用图像或图数据的 DC 方法。

---

### 提出了什么新方法或新思路

本文提出了 **C²TC (Class-Adaptive Clustering for Tabular Condensation)**，是首个**无需训练（training-free）且支持标签自适应（label-adaptive）**的表格数据压缩框架。其核心思想包括：

#### （1）将 DC 问题重构为 **Class-Adaptive Cluster Allocation Problem (CCAP)**

- 将传统的基于梯度匹配的目标函数，重新建模为一个**类感知聚类划分问题**。
- 通过**线性化特征编码器**，将分布匹配目标转化为 `arg min ||PX - X'||²`，从而完全避免梯度更新和模型训练。
- 支持使用标准聚类算法（如 K-means）直接在原始特征空间进行 condensation。

#### （2）提出 **HFILS（Heuristic First-Improvement Local Search）** 算法求解 NP-hard 的 CCAP

- 采用启发式局部搜索交替执行：
  - **Soft Allocation**：基于“肘部法则”动态调整各类别簇数；
  - **Class-wise Clustering**：对每类执行 K-means 并评估聚类损失。
- 引入**早停机制**和**步长衰减**以加速收敛。

#### （3）设计 **Hybrid Categorical Feature Encoding (HCFE)**

- **字符串型特征**：使用 n-gram 相似性编码 + 自编码器降维，保留语义相似性。
- **整数型分类特征**：采用平滑的 Target Encoding，注入监督信号并避免虚假序关系。
- 所有特征最终映射到统一数值空间，便于聚类。

---

### 相比现有方法的优势

| 维度 | C²TC | 传统 DC 方法（如 GM, MTT, DM） |
|------|------|-------------------------------|
| 是否需要训练 | ❌ 否（training-free） | ✅ 是（需训练 relay model） |
| 计算效率 | ⚡ 极高（至少快两个数量级） | 🐢 极慢（OOM / OOT 常见） |
| 类别平衡性 | ✅ 动态自适应分配 | ❌ 固定策略（Ratio/FIPC） |
| 特征处理能力 | ✅ 支持异构特征（HCFE） | ❌ 忽视 categorical 语义 |
| 可扩展性 | ✅ 良好（线性复杂度） | ❌ 差（随数据规模爆炸增长） |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

在 **10 个真实世界表格数据集**上进行了实验，涵盖不同规模、领域和类别不平衡程度：

| 数据集 | 样本数 | 特征数 | 类别数 | MinMaxCF¹ |
|--------|-------|--------|--------|----------|
| Adult (AD) | 48,842 | 14 | 2 | 0.31 |
| Covertype (CO) | 581,012 | 54 | 7 | 0.01 |
| Epsilon (EP) | 500,000 | 2,000 | 2 | 1.00 |
| Microsoft (MI) | 1,200,192 | 136 | 5 | 0.01 |
| Higgs (HI) | 940,160 | 24 | 2 | 1.00 |
| ... 其他略 | ... | ... | ... | ... |

> ¹ MinMaxCF：最小类频率 / 最大类频率，越小表示类别越不平衡。

所有数据来自 **OpenML**，按 80%/10%/10% 划分训练/验证/测试集。

---

### 实验设置和评估指标

#### Condensation Ratios
- 测试多个压缩率：`r ∈ {1%, 0.1%, 0.01%}`

#### 下游任务
- 在压缩后的数据上从头训练深度表格模型，评估其在原始测试集上的表现。

#### 模型架构（用于评估泛化性）
- **MLP-based**: MLP, MLP_PLR, RealMLP
- **Transformer-based**: FT-Transformer, TabNet, TabR

#### 评估指标
- **Accuracy**：整体准确率
- **Macro-F1**：各类 F1 分数的平均值，反映类别均衡性

#### 实施细节
- C²TC 参数：最大迭代 `T=1000`，容忍度 `ε=0.01`，patience `p=10`
- 基线方法使用官方实现，默认超参
- OOM / OOT 表示内存溢出 / 超时（>24小时）

---

### 基线方法对比

分为两类：

#### （1）Coreset Selection 方法
- **Random**：随机采样
- **Herding**：贪心选择靠近中心的样本
- **K-Center**：最大化覆盖范围

#### （2）Dataset Condensation 方法
- **DM** [25]：Distribution Matching（匹配类均值嵌入）
- **GM** [23]：Gradient Matching
- **MTT** [24]：Matching Training Trajectories

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 总体性能（Table II）
- C²TC 在 **10 个数据集中有 8 个达到最高 Accuracy**
- 在 **30 个实验场景中有 29 个取得最佳 Macro-F1**
- 即使在 `r=0.01%`（仅保留 0.01% 数据）下，也能恢复原数据 **86% 以上的 Accuracy**

#### 大规模数据表现（EP 和 MI）
| 方法 | EP (r=0.1%) Acc | MI (r=0.1%) Acc |
|------|------------------|------------------|
| DM / GM / MTT | OOM / OOT ❌ | OOM / OOT ❌ |
| C²TC | **75.2%** ✅ | **53.6%** ✅ |
| Random | 60.1% | 52.8% |

👉 C²TC 是唯一能在这些大规模数据上成功运行的 DC 方法。

---

### 与基线方法的对比结果

#### 效率对比（Figure 3 & 4）
- **Condensation 时间**：
  - 在 MI 上，C²TC 仅需 **39.2 秒**
  - MTT 需 **>78 小时**（超过 7,000 倍）
- **速度提升**：相比 SOTA DC 方法，**至少快 2 个数量级以上**
- **可扩展性**：时间随样本数 `N` 和特征维度 `F` 几乎线性增长，而基线呈指数上升

#### 模型训练时间（Figure 5）
- 在压缩数据上训练模型，**训练时间大幅缩短**
- 例如，在 MI 上训练 TabR：
  - 原始数据：耗时 **>10⁵ 秒**
  - 压缩后数据：**<10 秒**，加速 **>10⁴ 倍**

---

### 消融实验结果（Ablation Study）

#### （1）标签分配策略对比（Table IV）

| 策略 | EP (Acc/MF1) | CO (Acc/MF1) |
|------|--------------|--------------|
| Ratio | 69.9 / 69.0 | 65.6 / 33.1 |
| FIPC | 69.9 / 69.0 | 47.3 / 40.5 |
| **C²TC (adaptive)** | **75.2 / 74.8** ✅ | **69.6 / 46.4** ✅ |

👉 C²TC 显著优于固定策略，尤其在不平衡数据（CO）上同时提升 Accuracy 和 Macro-F1。

#### （2）模块消融（Table V）

移除以下组件会导致性能下降：

| 模块 | 影响 |
|------|------|
| **w/o Scale Factor**（γ=0） | Macro-F1 显著下降（如 CO 从 46.4 → 35.8），因多数类主导分配 |
| **w/o Soft Step Size** | Accuracy 下降明显（如 DA 从 51.8 → 36.8），缺乏探索能力 |
| **w/o Soft Allocation** | 分配僵化，性能下降（如 CO 从 69.6 → 58.3） |

👉 证明 HFILS 中各设计对高质量解至关重要。

#### （3）统计显著性检验（Table VI）
- 对比 Herding 和 MTT，在 `r=0.1%` 下进行 paired t-test
- C²TC 在 **9/10 数据集上 Accuracy 提升显著（p < 0.05）**
- 在 **8/10 数据集上 Macro-F1 提升显著**

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **无需训练即可实现高效表格数据压缩**：通过将 DC 重构为聚类问题，彻底摆脱梯度优化瓶颈。
2. ✅ **动态标签分配能有效缓解类别不平衡问题**：CCAP 框架自动平衡多数类与少数类的代表性。
3. ✅ **HCFE 成功保留分类特征语义**：结合相似性编码与 Target Encoding，在多种场景下表现最优。
4. ✅ **C²TC 具备卓越的效率与可扩展性**：在百万级样本、千维特征数据上仍可在分钟内完成压缩。
5. ✅ **下游性能优越且泛化性强**：在多种模型（MLP / Transformer）上均超越基线，尤其在不平衡数据上优势明显。

---

### 方法的局限性

1. **依赖 K-means 聚类假设**：假设类内数据呈凸形分布，可能不适用于高度非线性结构的数据。
2. **HCFE 需离线预处理**：虽然总开销低，但仍需额外训练自编码器（约几十秒）。
3. **未考虑特征间依赖关系**：当前方法基于样本独立同分布假设，忽略潜在的特征交互。
4. **仅适用于分类任务**：目前框架针对分类标签设计，回归任务需进一步拓展。

---

### 未来工作方向

1. **扩展至回归任务**：设计连续标签下的自适应划分机制。
2. **引入非线性聚类方法**：探索 GMM、谱聚类等更灵活的划分方式。
3. **支持增量/持续学习场景**：将 C²TC 应用于 continual learning 中的历史数据压缩。
4. **与其他数据高效范式结合**：如与 active learning、coreset selection 融合，构建端到端高效学习流水线。
5. **部署到工业数据库系统**：集成进 SQL 引擎或 OLAP 系统，作为内置“数据蒸馏”功能。

--- 

> 🔚 **总结一句话**：  
> **C²TC 开创性地将表格数据压缩从“训练驱动”转向“结构驱动”，实现了高效、免训、自适应的 condensation 新范式，在性能与效率之间取得了前所未有的平衡。**

</details>

---

### 3. [Multi-Layer Scheduling for MoE-Based LLM Reasoning](https://arxiv.org/abs/2602.21626)

**Authors**: Yifan Sun, Gholamreza Haffar, Minxian Xu, Rajkumar Buyya, Adel N. Toosi  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.21626v1  

#### Abstract
Large Language Models (LLMs) have achieved remarkable success across a wide range of tasks, but serving them efficiently at scale remains a critical challenge due to their substantial computational and latency demands. While most existing inference frameworks rely on simple scheduling strategies suc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Multi-Layer Scheduling for MoE-Based LLM Reasoning 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前主流的 LLM 推理框架（如 vLLM、SGLang）在服务 **Mixture-of-Experts (MoE)** 模型时存在以下三大瓶颈：
- **Engine-Level**：采用简单的 Round-Robin 或 FCFS 调度策略，忽略实际负载状态（如 KV Cache 使用率、运行中 token 数），导致负载不均衡和资源利用率低下。
- **Request-Level**：使用 First-Come-First-Serve (FCFS)，容易引发 **Head-of-Line Blocking**，长请求阻塞短请求，造成尾延迟高。
- **Expert-Level**：未考虑专家之间的跨层依赖关系（inter-layer expert affinity）和热点问题（expert hotspots），导致 GPU 间计算负载严重失衡。

这些问题共同限制了 MoE 模型在大规模部署下的推理效率与可扩展性。

---

### 提出了什么新方法或新思路
作者提出 **Gimbal** —— 一种面向 MoE 架构的 **multi-layer scheduling framework**，从三个层级协同优化调度决策：

| 层级 | 方法 |
|------|------|
| **Request-Level** | 引入 **Shortest-Job-First (SJF)** + **aging 机制** 的调度器，优先处理预填充 token 较少的请求，并防止大请求饥饿。 |
| **Engine-Level** | 设计 **load-aware dispatching** 策略，综合考虑：<br>• 当前 prefix token 负载<br>• KV Cache 利用率<br>• 用户粘性（user stickiness）以提升 prefix cache 复用率 |
| **Expert-Level** | 提出 **Expert Dynamic Replacement Module**，结合：<br>• 专家激活频率（activation count）<br>• 跨层专家依赖图谱（inter-layer expert affinity）<br>动态调整专家在 GPU 上的分布，缓解热点并减少跨 GPU 通信 |

该框架不依赖特定模型内部结构，适用于 Mixtral、Switch Transformer、DeepSeek 等多种 MoE 架构。

---

### 相比现有方法的优势
- **多层级联合优化**：首次将 request、engine、expert 三层调度统一建模，实现端到端协同优化。
- **无需输出长度预测**：不同于基于 proxy model 预测输出长度的方法（如 [14]-[17]），Gimbal 使用 **prefill token count** 作为任务代价估计，更稳定且 model-agnostic。
- **显式利用用户行为模式**：通过 user-affinity 实现更高 prefix cache hit rate，降低重复计算开销。
- **兼顾负载平衡与通信局部性**：在 expert placement 中同时最小化负载偏差和跨 GPU 通信成本。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **BurstGPT dataset [22]**：用于主性能测试，包含真实世界中的多样化请求分布。
  - 构造五种合成分布进行鲁棒性验证：
    - Random
    - Central
    - Descending
    - Two-end
    - Average
- **ShareGPT dataset [27]**：用于专门评估 **prefix cache reuse** 和 user-affinity 效果（含用户标识）。

---

### 实验设置
- **硬件平台**：
  - 双 Intel Xeon Gold 6326 CPU @2.90GHz
  - 双 NVIDIA A100 80GB GPU（NVLink 连接）
  - 1TB 内存
- **模型**：
  - **Qwen3-30B-A3B [18]**：典型的中小规模 MoE 模型，共 48 层，每层激活 3 个专家。
- **软件基础**：
  - 基于 **vLLM v0.9.1** 改造实现 Gimbal。
  - 启用 **expert parallelism** 和 **pplx-kernels** 作为 MoE all-to-all 通信后端。
- **请求速率（RPS）**：1.0 ~ 1.4 RPS，在接近饱和状态下测试系统表现。

---

### 评估指标
| 指标 | 定义 |
|------|------|
| **TTFT** | Time To First Token，衡量首 token 延迟（含 prefill 阶段） |
| **TPOT** | Time Per Output Token，解码阶段每个输出 token 的平均延迟 |
| **Throughput** | 系统整体吞吐量（tokens/s） |
| **Prefix Cache Block Hit Count** | 全局 prefix cache 命中次数（绝对值） |
| **Prefix Cache Hit Rate (global)** | 缓存命中块数 / 总探查块数的比例 |

---

### 基线方法对比
- **vLLM**：当前最先进的开源推理框架，作为主要 baseline。
- **Ablation Variants**：
  - **DPLB**：仅启用 DP Engine Load Balancer
  - **SJFS**：仅启用 per-engine SJF Scheduler
  - **EDR**：仅启用 Expert Dynamic Replacement Module

---

## 3. 主要实验结果和性能指标

### 关键性能数据（@1.4 RPS）

| 指标 | 结果 |
|------|------|
| **TTFT Reduction** | 平均降低 **17.76%**（最高达 **17.8%** on Two-end 分布） |
| **TPOT Reduction** | 平均降低 **13.34%**（最高达 **17.6%** on Central 分布） |
| **Throughput** | 与 vLLM 相当，无显著下降（见 Fig. 10） |
| **Prefix Cache Hit Count** | 平均提升约 **3%**（18,451 → 18,992） |
| **Global Hit Rate** | 从 3.64% 提升至稳定 **3.80%**（+4.4% relative） |

> 注：所有结果基于超过 100 次实验，涵盖多种 workload 分布与随机种子。

---

### 与基线方法的对比结果
- 在所有 workload 分布下，**Gimbal 均优于 vLLM 及其各 ablation 版本**。
- 随着负载增加（1.2→1.4 RPS），Gimbal 的优势更加明显，说明其在高并发场景更具鲁棒性。
- **DPLB 对 TTFT 改善贡献最大**，而 **SJFS 和 EDR 提供额外增益**，三者协同效果最佳。

---

### 消融实验结果
| 组件 | 对 TTFT 影响 | 对 TPOT 影响 | 主要作用 |
|------|-------------|--------------|----------|
| **DPLB** | 显著降低 | 中等降低 | 减少 engine 负载不均，提升 prefill 效率 |
| **SJFS** | 中等降低 | 显著降低 | 缓解 HoL blocking，改善尾延迟 |
| **EDR** | 中等降低 | 中等降低 | 缓解 expert hotspot，减少跨 GPU 通信 |
| **Gimbal (全组合)** | ✅ 最优 | ✅ 最优 | 协同效应释放最大潜力 |

> 图 6–9 显示：只有当三层调度机制联合启用时，才能取得最优性能。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **多层级协同调度是 MoE 推理优化的关键路径**：
   - 单一层级优化（如仅改调度器或仅调专家放置）收益有限；
   - 联合优化 request、engine、expert 三层决策可显著释放系统潜能。

2. **real-time 负载感知比静态调度更有效**：
   - 基于 KV Cache usage 和 running load 的动态 dispatching 明显优于 RR/FCFS。

3. **user stickiness 显著提升 prefix cache 利用率**：
   - 将同一用户的请求路由到相同 engine，能提高 cache hit rate，带来可测量的性能增益。

4. **inter-layer expert affinity 是不可忽视的系统特征**：
   - MoE 模型中存在强跨层依赖关系（见 Fig. 4），利用这些依赖进行 co-location 可大幅减少通信开销。

---

### 方法的局限性
1. **实验环境受限**：
   - 所有实验在单台双 A100 服务器上完成，未覆盖大规模分布式集群（如多节点、异构 GPU）。
   - 不足以反映生产级系统的复杂性（如网络延迟、故障恢复等）。

2. **expert 动态迁移开销较高**：
   - 当前 EDR 模块依赖离线构建的 affinity matrix，需定期采集统计信息，引入额外 runtime overhead。
   - 实际迁移过程中可能影响在线服务质量。

3. **affinity matrix 固定不变**：
   - 当前假设专家依赖关系静态，未支持动态变化的工作负载或自适应学习机制。

---

### 未来工作方向
1. **扩展至更大规模集群**：
   - 在 multi-node、multi-rack 环境中验证 Gimbal 的可扩展性和稳定性。

2. **轻量化在线 affinity estimation**：
   - 开发低开销的实时专家依赖追踪机制，替代当前周期性采样方式。

3. **cost-aware expert migration policy**：
   - 引入迁移代价模型，权衡“负载均衡收益” vs “迁移开销”，避免频繁重配置。

4. **支持更多 MoE 架构变体**：
   - 如稀疏 MLP、conditional computation、routing dropout 等新型 MoE 设计。

--- 

> ✅ **总结一句话**：  
> Gimbal 通过 **request-level SJF + engine-level load-aware dispatching + expert-level dependency-aware placement** 的三层协同调度，在保持 throughput 不降的前提下，实现了 **平均 17.8% TTFT ↓ 和 13.3% TPOT ↓**，为 MoE-based LLM serving 提供了一套高效、实用、通用的新范式。

</details>

---

### 4. [NGDB-Zoo: Towards Efficient and Scalable Neural Graph Databases Training](https://arxiv.org/abs/2602.21597)

**Authors**: Zhongwei Xie, Jiaxin Bai, Shujie Liu, Haoyu Huang, Yufei Li, Yisen Gao, Hong Ting Tsang, Yangqiu Song  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.21597v1  

#### Abstract
Neural Graph Databases (NGDBs) facilitate complex logical reasoning over incomplete knowledge structures, yet their training efficiency and expressivity are constrained by rigid query-level batching and structure-exclusive embeddings. We present NGDB-Zoo, a unified framework that resolves these bott...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NGDB-Zoo: Towards Efficient and Scalable Neural Graph Databases Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Neural Graph Databases (NGDBs) 在处理大规模、不完整知识图谱上的复杂逻辑推理时面临两大瓶颈：

- **计算效率与拓扑刚性**：传统基于 query-level batching 的训练方式要求同一批次内的查询具有相同的拓扑结构（如 2-hop chain），导致在混合查询负载下 GPU 利用率极低。
- **表示摩擦（Representation Friction）**：将 Pre-trained Text Encoders (PTEs) 的高维语义向量集成到结构化嵌入中会引发严重的 I/O 阻塞和内存溢出问题。

### 提出的新方法与思路
作者提出 **NGDB-Zoo**，一个统一的高效神经图数据库训练框架，其核心创新包括：

#### ✅ Contribution 1: Operator-Level Training Paradigm（算子级训练范式）
- 将复杂查询分解为由原子操作符（如 Project、Intersect、Union）构成的有向无环图（DAG）。
- 引入 **operator-level batching**，打破 query-level 的拓扑约束，实现跨查询的算子融合与动态调度。
- 采用 **Max-Fillness 调度策略**，优先执行最能饱和 GPU 的算子类型，最大化硬件利用率。

#### ✅ Contribution 2: Decoupled Semantic Integration Architecture（解耦语义集成架构）
- 将 PTE 推理从训练循环中剥离，通过 **离线预计算 + GPU-Resident 缓存** 实现语义特征的零开销集成。
- 语义向量以只读缓冲区形式驻留于 GPU 高带宽内存（HBM），训练阶段仅需 `Gather` 操作即可完成融合。
- 使用轻量级投影层对齐语义与结构空间。

#### ✅ Contribution 3: Scalable Neuro-Symbolic Benchmarking
- 在六个标准 benchmark 上进行了全面评估，涵盖从中小规模（FB15k）到超大规模图（ATLAS-Wiki, 4M+ 节点）。
- 支持 14 种复杂查询模式（含 negation），验证了系统在多样化推理任务中的鲁棒性。

### 相比现有方法的优势
| 维度 | NGDB-Zoo | 传统方法 |
|------|----------|---------|
| 批处理粒度 | Operator-level | Query-level |
| 语义集成方式 | 离线预计算 + GPU 缓存 | 在线推理（I/O 密集） |
| 硬件利用率 | >80% GPU 利用率 | <30%（因碎片化） |
| 可扩展性 | 支持千万级实体图 | 多数受限于内存 |

---

## 2. 核心实验方法和设置

### 数据集
实验覆盖六大数据集，体现从小到大的连续谱系：

| 数据集 | 实体数 | 边数 | 特点 |
|--------|-------|-----|------|
| **FB15k / FB15k-237** | ~15K | ~500K | 标准 CQA benchmark |
| **NELL995** | 63K | 142K | 更稀疏的知识图 |
| **FB400k** | 410K | 2.15M | 大规模通用知识图 |
| **ogbl-wikikg2** | 2.5M | 17.1M | OGB 最大知识图之一 |
| **ATLAS-Wiki-Triple-4M** | 4.04M | 28.8M | 当前最大公开 KG，含 512K 关系 |

> 📌 注：所有数据均见附录 Table 4。

### 实验设置与评估指标

#### 模型骨干（Backbone Models）
- **GQE**, **Q2P**: 向量空间表示
- **Q2B**, **BetaE**: 基于概率分布的推理（Box/Beta）
- **FuzzQE**: 模糊逻辑模型

#### 语义编码器（PTE）
- **Qwen3-Embedding-0.6B**
- **BGE-Base-En-v1.5**

#### 评估指标
| 指标 | 描述 |
|------|------|
| **MRR (%)** | Mean Reciprocal Rank，衡量预测准确性 |
| **Throughput (Queries/sec)** | 每秒处理的查询数量，反映训练吞吐量 |
| **Peak GPU Memory (GB)** | 显存峰值占用 |
| **Speedup** | 相对于基线的加速比 |

#### 基线方法对比
- **SQE**, **SMORE**, **DGL-KE**, **PyTorch-BigGraph (PBG)**, **Marius**
- 主要比较端到端训练速度、MRR 表现及多 GPU 扩展性

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔥 端到端训练吞吐提升（Table 3）
在 FB15k 上使用 BetaE 模型：
- NGDB-Zoo 达到 **4,750 queries/sec**
- 相较于 SQE（655 queries/sec）实现 **7.25× 加速**
- 相较于 SMORE（1,807 queries/sec）仍达 **2.6× 提升**

> 💡 即使在小图上也显著优于主流系统。

#### 🚀 大规模图表现（Table 1）
在生产级图上保持高吞吐：

| 图谱 | 模型 | Throughput (10⁵ Q/s) | MRR (%) |
|------|------|------------------|--------|
| ogbl-wikikg2 | BetaE | 1.965 | 44.54 |
| ATLAS-Wiki | GQE | 2.200 | 7.31 |

> ✅ 表明 NGDB-Zoo 可扩展至百万级实体场景。

#### ⚙️ 多 GPU 扩展性（Figure 7）
- 在 ogbl-wikikg2 和 ATLAS-Wiki 上均表现出 **近线性加速比**
- 使用 8 GPU 时，吞吐接近单卡的 8 倍
- 通信开销极小，适合分布式部署

#### 🔄 单跳链接预测运行时（Table 2）
在 Freebase 上 ComplEx 模型对比：
- NGDB-Zoo 在 1-GPU 下耗时 **628s/epoch**
- 快于 PBG（3060s）、SMORE（760s）
- 多 GPU 扩展性最佳（8-GPU 仅 94s）

---

### 与基线方法的对比结果

| 对比维度 | 结果 |
|--------|------|
| **平均吞吐提升** | **1.8× – 6.8×** 超越 state-of-the-art 基线 |
| **MRR 准确率** | 不降反升，尤其在稀疏图（NELL）上更明显 |
| **GPU 利用率** | 平均 >75%，远高于基线（<30%） |
| **显存管理** | 通过 eager reference counting 显著降低中间张量压力 |

---

### 消融实验结果（Ablation Studies）

#### 🧪 算子批处理加速效果（Table 6）
| 算子 | 原始耗时 (ms) | 批处理后 (ms) | 加速比 |
|------|-------------|--------------|-------|
| EmbedE | 2.3 | 0.8 | 2.88× |
| Project | 15.7 | 4.2 | 3.74× |
| **Intersect** | 78.5 | 6.0 | **13.11×** |
| **Union** | 62.3 | 5.1 | **12.22×** |

> ✅ 表明可变输入长度的集合操作最受益于 batching。

#### 🧠 语义增强的影响（Figure 8 & Table 8）
- **Throughput 提升**：引入 Qwen/BGE 后，吞吐从平均 347 提升至 **1915 queries/sec（+4.5×）**
- **Memory 下降**：尽管缓存了语义向量，但由于卸载了 PTE 参数，**峰值显存反而下降约 10–15%**
- **MRR 提升**：平均 MRR 提高 **+4.74%**，在稀疏图（如 NELL）中增益更大

#### 🎯 自适应采样（Figure 9）
- 引入 bursty query 分布模拟真实负载
- 自适应在线采样机制使 MRR **相对提升 21.5%**
- 验证了对非平稳查询流的强大鲁棒性

---

## 4. 关键结论和发现

### 主要发现
1. **Operator-level batching 是突破 NGDB 效率瓶颈的关键**  
   —— 将抽象层级从“查询”下沉至“算子”，有效消除拓扑碎片化，释放 GPU 并行潜力。

2. **解耦语义集成可同时提升性能与效率**  
   —— “离线编码 + GPU 缓存”设计不仅避免 I/O 阻塞，还能通过移除大模型参数来 **降低显存压力**。

3. **高吞吐不牺牲精度**  
   —— NGDB-Zoo 在大幅提升训练速度的同时，MRR 表现持平甚至优于基线，证明其工程优化未损害模型表达能力。

4. **适用于极端规模图谱**  
   —— 成功在含 400 万实体、50 万关系的 ATLAS-Wiki 上稳定运行，为下一代 Agentic NGDB 提供基础设施支持。

---

### 方法的局限性（Limitations）
1. **依赖冻结的 PTE**  
   —— 当前语义编码器不可微调，限制了语义与结构的联合优化空间。

2. **异步 CPU offload 依赖 PCIe 带宽**  
   —— 对超大规模图仍需 CPU-GPU 流水线，在极高 batch size 下可能成为瓶颈。

3. **静态图假设**  
   —— 当前框架假设图结构不变，尚未支持 streaming knowledge base 的实时更新。

---

### 未来工作方向
1. **End-to-end co-optimization of PTE and structural encoder**
   - 探索轻量化 PTE 微调路径，实现语义先验的图感知适配。

2. **Fully GPU-resident large-graph training**
   - 结合图分区与分布式缓存，减少对外部存储的依赖。

3. **Streaming operator scheduling for evolving graphs**
   - 将 operator-level paradigm 扩展至动态图环境，支持 real-time schema evolution。

4. **Integration with Agentic workflows**
   - 构建面向 Autonomous Agent 的 NGDB 引擎，支持复杂规划与反思链推理。

---

> ✅ **总结一句话**：  
> NGDB-Zoo 通过 **operator-level 动态调度** 与 **decoupled 语义融合**，首次实现了高效、可扩展且语义丰富的神经图数据库训练，在吞吐、精度与系统稳定性之间取得优异平衡，是迈向实用化 NGDB 的重要一步。

</details>

---

### 5. [SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference](https://arxiv.org/abs/2602.22136)

**Authors**: Qunyou Liu, Pengbo Yu, Marina Zapater, David Atienza  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.22136v1  

#### Abstract
Deep neural networks (DNNs) are essential for performing advanced tasks on edge or mobile devices, yet their deployment is often hindered by severe resource constraints, including limited memory, energy, and computational power. While uniform quantization provides a straightforward approach to compr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前边缘设备上的 DNN 推理面临严重的资源限制（内存、能耗、算力），而现有的量化方法存在以下不足：
- **Uniform Quantization**：对所有层采用相同 bitwidth，无法适应不同层对量化噪声的敏感度差异，导致精度损失或资源利用不充分。
- **现有 Heterogeneous Quantization 方法**：
  - 依赖大规模搜索（如强化学习、NAS）或复杂的二阶分析（如 Hessian），计算开销大；
  - 缺乏对硬件约束（如内存大小、能效预算、延迟要求）的自适应能力；
  - 多数为静态配置，难以灵活适配多样化的边缘部署场景。

### ✅ 提出的新方法与思路
作者提出 **SigmaQuant**，一种**硬件感知的、分层异构量化框架**，其核心思想是：
- 利用每层权重的 **标准差（Standard Deviation, σ）** 和 **KL 散度（KL Divergence）** 来指导 bitwidth 分配；
- 设计两阶段策略，在满足用户指定的准确率与资源边界条件下，高效完成混合精度分配。

#### 创新点：
1. **分布拟合视角（Distribution-Fitting Perspective）**
   - 将量化视为从浮点分布 $ p_{\text{float}} $ 到整数量化分布 $ p_{\text{quant}} $ 的信息逼近过程；
   - 使用 **KL 散度** 作为衡量量化失真的理论依据，确保分布偏移最小。

2. **两阶段自适应算法设计**
   - **Phase 1: Cluster-Based Initialization**
     - 基于各层权重的标准差 σ 进行自适应 k-means 聚类，初步分配 bitwidth（如 2/4/6/8-bit）；
     - 引入惩罚项防止某些 cluster 过大，提升 bitwidth 分布均匀性。
   - **Phase 2: KL-Based Iterative Refinement**
     - 计算每层的“敏感度评分”（结合 σ 与归一化 KL 散度）；
     - 动态调整高敏感层（增加 bitwidth）或低敏感层（降低 bitwidth），逐步逼近目标约束。

3. **硬件友好性与可扩展性**
   - 支持 shift-add-based MAC 架构（广泛用于边缘加速器），直接优化 PPA（Power, Performance, Area）；
   - 可同时优化 memory size 或 BOPs（compute budget），适用于多种硬件目标。

### ✅ 相比现有方法的优势
| 维度 | SigmaQuant | 现有方法（如 HAQ、HAWQ、CLADO） |
|------|------------|-------------------------------|
| **搜索效率** | 无需全局搜索，仅需少量 QAT 循环 | 需要 RL/NAS/Hessian 计算，成本高昂 |
| **硬件适配性** | 显式建模硬件约束（memory, energy, latency） | 多数忽略实际硬件影响 |
| **灵活性** | 用户可设定 accuracy/memory 目标，动态响应 | 多为固定配置，泛化性差 |
| **实现复杂度** | 仅依赖简单统计量（σ + histogram-based KL） | 需要复杂梯度或二阶导数估计 |

---

## 2. 核心实验方法和设置

### ✅ 数据集
- **ImageNet**：主测试集，用于验证在真实大规模任务下的有效性；
- **CIFAR-100**：用于快速趋势分析与消融实验，节省 GPU 资源。

### ✅ 模型架构
- **ResNet 系列**：ResNet-18, 34, 50, 101, 152；
- **MobileNet 家族**；
- **InceptionV3**。

### ✅ 实验设置
- **量化方案**：
  - 权重支持 {2,4,6,8}-bit 混合精度；
  - 激活函数默认 8-bit（除非以 BOPs 为目标时也进行调整）；
  - 使用 symmetric min-max quantization（weights）、asymmetric clipping（activations @99.9% percentile）；
  - 采用 Quantization-Aware Training (QAT) 进行微调。
- **目标约束示例**：
  - 最大允许精度下降：≤1%；
  - 内存压缩目标：≤75% of INT8 模型大小。

### ✅ 评估指标
| 类别 | 指标 |
|------|------|
| **模型性能** | Top-1 Accuracy (%) |
| **压缩效果** | Model Size (MB)，Compression Ratio |
| **计算效率** | BOPs（Bit Operations），Latency (cycles) |
| **硬件效益** | Power, Area, Energy Consumption（通过 TSMC 28nm 后综合仿真获取） |
| **搜索开销** | QAT Epochs 数量、总训练时间（wall-clock time） |

### ✅ 基线方法对比
- **Uniform Quantization**：A8W8, A8W6, A8W4, A8W2；
- **State-of-the-art Heterogeneous Methods**：
  - **HAQ**（Reinforcement Learning）
  - **HAWQ-V3**（Hessian-based sensitivity）
  - **UNIQ**（Noise Injection）
  - **CLADO**（Integer Quadratic Programming）
  - **Apprentice**（Knowledge Distillation）

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（ImageNet & CIFAR-100）

#### 📊 在同等模型大小下，显著提升准确率
- 在 CIFAR-100 上，相比 uniform quantization：
  - **相同内存预算下，最高提升 4.0% Top-1 准确率**；
  - **达到相同准确率时，模型体积减少约 40%**。

#### 📈 在 ResNet-50 上 vs. SOTA 方法（Table III）
| 方法 | Bits(W,A) | Model Size (MB) | Top-1 Acc (%) |
|------|-----------|------------------|----------------|
| HAWQ-V3 | 4/8,4/8 | 18.7 | 76.73 |
| CLADO | mix,8 | 13.42 | 73.10 |
| **Ours (SigmaQuant)** | **mix,8** | **12.02** | **76.86** |
| **Ours (SigmaQuant)** | **mix,8** | **10.78** | **75.63** |

> ✅ 结论：SigmaQuant 在更小模型尺寸下实现了更高的准确率，优于 HAWQ 和 CLADO。

#### 🔋 硬件层面优势（基于 shift-add MAC，TSMC 28nm）
| 指标 | 提升幅度 |
|------|---------|
| **Area 节省** | 最多 **22.3%**（vs. INT8） |
| **Energy 消耗** | 最多 **20.6%** 下降 |
| **Latency 开销** | 略有增加（+17.5% cycles），但仍优于多数 uniform 方案 |
| **PPA Trade-off** | 更接近理想“左上角”区域（高精度 + 低功耗/面积） |

> 💡 示例：ResNet34 上，SigmaQuant 较 A8W2 uniform 方案节省 23.3% energy，且精度损失仅 2.97%，远低于 uniform 的 8.54%。

#### ⏱️ 搜索效率（End-to-End 时间）
| 模型 | 总耗时（小时） |
|------|----------------|
| ResNet-18 | ~2h |
| ResNet-34 | ~4h |
| ResNet-50 | ~12h |
| ResNet-152 | ~30h |

> ✅ 显著低于依赖 Hessian 或 RL 搜索的方法（通常需要数百 GPU 小时）。

### ✅ 消融实验结果（Ablation Study）
- **Phase 1 vs. Phase 2**：
  - Phase 1（cluster-based）已能快速进入可行域；
  - Phase 2（KL refinement）进一步将模型推入“Target Zone”，平均带来 **0.5–1.5% accuracy 提升** 或 **额外 5–10% 压缩空间**。
- **Buffer 设置影响（Table IV）**：
  - 缓冲区越大（Aggressive），Phase 1 收敛越快，但 Phase 2 需更多迭代；
  - 默认设置（Balanced）可在收敛速度与最终质量间取得良好平衡。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Layer-wise weight standard deviation (σ) 是一个强有效的量化敏感度代理指标**：
   - σ 大 → 分布宽 → 对量化更敏感 → 应保留更高 bitwidth；
   - σ 小 → 可安全压缩至 2–4 bit。

2. **KL 散度是控制分布偏移的关键工具**：
   - 显式最小化 $ D_{KL}(p_{\text{float}} || p_{\text{quant}}) $ 可有效维持推理准确性。

3. **两阶段设计实现了高效与精确的平衡**：
   - Phase 1 快速定位近优解；
   - Phase 2 局部精细调节，避免全局搜索开销。

4. **硬件感知设计至关重要**：
   - shift-add 架构中，weight bitwidth 直接决定 cycle 数与 energy；
   - SigmaQuant 所得模型天然契合此类硬件，实现端到端 PPA 优化。

### ❗ 局限性
- 当前方法主要针对 CNN 类模型，Transformer 等结构尚未充分验证；
- Phase 2 中 KL 散度计算依赖 histogram 近似，可能引入误差；
- 对极端紧致约束（如 <20% INT8 size）可能出现无解情况（见 Table II 中 ResNet-50 未达标）；
- 仍需轻量级 QAT，不完全属于 Zero-shot PTQ 范畴。

### 🔮 未来工作方向
- 扩展至 Transformer 和 Diffusion Models；
- 探索 activation 的动态 bitwidth 调整机制；
- 结合 pruning 与 quantization，构建联合压缩框架；
- 支持 runtime 自适应切换（类似 AdaBits），增强部署灵活性；
- 探索在 RISC-V 或 FPGA 平台上的实时部署原型。

---

## ✅ 总结
**SigmaQuant** 是一种**轻量级、硬件感知、自适应性强**的异构量化方法，它通过 **σ + KL 散度驱动的两阶段策略**，在无需昂贵搜索的前提下，实现了：
- 更高的 accuracy-efficiency trade-off；
- 更好的硬件 PPA 表现；
- 更强的跨平台适应能力。

该方法为边缘 AI 的高效部署提供了实用且可扩展的解决方案，尤其适合资源受限、需求多样的嵌入式系统应用场景。

</details>

---

### 6. [ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning](https://arxiv.org/abs/2602.21534)

**Authors**: Xiaoxuan Wang, Han Zhang, Haixin Wang, Yidan Shi, Ruoyan Li, Kaiqiao Han, Chenyi Tong, Haoran Deng, Renliang Sun, Alexander Taylor, Yanqiao Zhu, Jason Cong, Yizhou Sun, Wei Wang  
**Category**: cs.AI  
**Published**: 2026-02-26  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.21534v1  

#### Abstract
Agentic reinforcement learning (ARL) has rapidly gained attention as a promising paradigm for training agents to solve complex, multi-step interactive tasks. Despite encouraging early results, ARL remains highly unstable, often leading to training collapse. This instability limits scalability to lar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **Agentic Reinforcement Learning (ARL)** 中普遍存在的训练不稳定问题展开研究。在多轮交互任务中，ARL 经常出现训练崩溃（training collapse），表现为性能骤降、梯度爆炸、格式错误频发等现象。这种不稳定性严重限制了模型在复杂环境中的扩展性和可复现性。

### 提出的新方法与新思路
作者提出了 **ARLArena**，一个统一的稳定训练框架，包含以下核心创新：

- **标准化测试平台（Standardized Testbed）**  
  构建了一个干净、可控的实验环境，通过行为克隆（Behavior Cloning）、格式惩罚（Format Penalty）、KL 正则化和超参数搜索来初始化并稳定训练过程，确保公平比较。

- **策略梯度四维分解分析法（Four-Dimensional Policy Gradient Decomposition）**  
  将基于 PPO 的策略优化方法系统地分解为四个正交维度：
  1. **Loss Aggregation**（损失聚合）
  2. **Importance Sampling (IS) Clipping**（重要性采样裁剪）
  3. **Advantage Design**（优势函数设计）
  4. **Dynamic Sampling / Filtering**（动态采样/过滤）

  这种细粒度分析使得可以独立评估每个设计选择对稳定性和性能的影响。

- **提出 SAMPO：稳定的多轮策略优化算法**  
  基于上述分析，提出 **Stable Agentic Multi-turn Policy Optimization (SAMPO)**，融合了序列级裁剪（sequence-level clipping）、精细优势估计（fine-grained advantage）和动态过滤（dynamic filtering），实现稳定且高性能的训练。

### 相比现有方法的优势
- **更强的稳定性**：相比 GRPO、SAPO、CISPO 等主流方法，SAMPO 避免了训练崩溃，表现出单调上升的成功率曲线。
- **更高的最终性能**：在多个任务上平均比 GRPO 提升 **25.2%**。
- **可复现性强**：提供了一套完整的“清洁训练配方”（clean training recipe），显著提升实验可复现性。
- **通用性强**：在 Web Agent、Multimodal Agent、Math Agent 和 Embodied Agent 多种场景下均有效。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在四个典型的 agentic 任务上进行：

| 数据集 | 类型 | 描述 |
|-------|------|------|
| **ALFWorld** | Embodied Agent | 文本驱动的具身环境，要求完成如“将冷却的鸡蛋放入微波炉”等复杂任务，涉及长程规划与工具使用。 |
| **WebShop** | Web Agent | 模拟电商购物场景，代理需根据用户需求浏览商品、点击详情、加入购物车并购买。 |
| **Sokoban** | Planning Agent | 经典推箱子游戏，视觉输入，考验路径规划与避免死局的能力。 |
| **TIR Math** | Math Agent | 数学推理任务，允许调用 Python 工具执行计算，评估逻辑与符号推理能力。 |

### 实验设置与评估指标
- **基础模型**：Qwen3-4B（部分验证使用 Qwen3-8B）
- **训练框架**：基于 verl RL 框架构建 agentic-loop 架构
- **硬件**：NVIDIA H200 / B200 GPU
- **关键设置**：
  - 多轮交互（multi-turn interaction），每轮生成 `<think>` 和 `<action>` 结构化输出
  - 使用 **KL 正则化**（k3 estimator）防止策略漂移
  - 对每种 PO 方法进行 **特定超参网格搜索** 以达到稳定状态

#### 评估指标
| 任务 | 主要指标 |
|------|---------|
| ALFWorld, WebShop, Sokoban | **Success Rate**, **Task Score** |
| TIR Math | **Pass@k (AIME/AIME25)** |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

| Method | ALFWorld (Success) | WebShop (Success) | Sokoban (Success) | Average Score |
|--------|--------------------|-------------------|-------------------|---------------|
| GRPO (baseline) | 62.36% | 57.71% | 83.90% | 46.16 |
| GSPO | 78.61% | 72.48% | 82.22% | 52.28 |
| GIGPO | 81.09% | 56.55% | 82.67% | 49.71 |
| DAPO+GIGPO | 60.55% | 76.82% | 86.20% | 53.36 |
| **SAMPO (Ours)** | **92.72%** | **77.73%** | **88.86%** | **60.21** |

> ✅ **SAMPO 平均得分提升 25.2%**（vs GRPO），并在所有任务上取得最佳表现。

### 与基线方法的对比结果
- **GSPO** 是唯一未崩溃的 IS 变体，因其采用 **sequence-level clipping** 而非 token-level。
- **SAPO/CISPO** 在默认设置下迅速崩溃（见 Figure 3），成功率从高点暴跌至接近零。
- **DAPO+GRPO** 表现不佳，说明动态过滤需配合多样化优势信号才有效。
- **SAMPO 显著优于所有单一维度改进的方法**，证明多维度协同设计的重要性。

### 消融实验结果
#### （1）IS Clipping 分析（Figure 3 & 4）
- **容忍性裁剪（Tolerant Clipping）**（如 SAPO/CISPO）导致大量 `Adv<0` 且 `IS<1` 的样本失控，引发梯度爆炸。
- **序列级裁剪（Sequence-level Clipping）**（如 GSPO）能有效抑制异常轨迹，保持 KL divergence 和 gradient norm 稳定。

#### （2）稳定化策略效果（Table 4）
| 方法 | 原始成功 | +KL(0.05) | +大batch | **+Sequence Masking** |
|------|--------|----------|---------|------------------------|
| SAPO | 25.16% | 48.05% | 64.30% | **76.92%** |
| CISPO | 54.42% | 38.46% | 21.59% | **78.88%** |

> 🔍 **Sequence masking** 是最有效的稳定手段，直接屏蔽 `Adv<0 && IS<1` 的有害轨迹。

#### （3）其他维度影响
- **Advantage Design**：GIGPO 引入状态级分组优势，在 ALFWorld 上提升明显（+34.3%）。
- **Dynamic Filtering**：仅当与 GIGPO 结合时有效（DAPO+GIGPO > DAPO+GRPO），否则会削弱格式学习信号。
- **Loss Aggregation**：`seq-mean-token-mean` 导致长度偏差，在数学任务上表现差。

---

## 4. 关键结论和发现

### 主要发现（Key Findings）
1. **IS 裁剪机制高度敏感**：
   - ❌ **容忍性裁剪（Tolerant Clipping）** 导致短期收益但长期崩溃。
   - ✅ **序列级裁剪（Sequence-level Clipping）** 是稳定训练的关键。
   - 📌 *根本原因*：负优势且低 IS 比值的序列主导更新，造成策略剧烈偏移。

2. **优势函数设计带来稳健增益**：
   - 引入环境层级信息（如 GIGPO 的 state-grouping）可缓解稀疏奖励问题，提升信用分配精度。

3. **动态过滤需搭配丰富优势信号**：
   - 单独使用可能破坏早期格式学习；与 GIGPO 结合后可增强梯度质量。

4. **清洁训练流程至关重要**：
   - 行为克隆 + 格式约束 + KL 正则构成“最小可行稳定条件”，缺一则易失败。

5. **多维度协同不可替代**：
   - 单一修改无法解决根本问题，**SAMPO 的成功源于三大支柱联合**：
     - Sequence-level Clipping
     - Fine-grained Advantage Estimation
     - Dynamic Filtering

### 方法的局限性
- **依赖高质量初始行为克隆数据**：若 SFT 数据质量差，后续 RL 难以纠正。
- **对 off-policy staleness 敏感**：批量 rollout 更新仍存在策略滞后问题（见 Table 5），尤其在数学任务中。
- **当前未支持完全在线学习**：仍为 offline RL 框架，难以应对真实世界持续变化的反馈。

### 未来工作方向
1. **开发更鲁棒的 off-policy correction 机制**，降低 stale data 影响。
2. **探索自动化的超参调节与崩溃检测机制**，进一步提升可复现性。
3. **将 SAMPO 扩展到更大规模模型与真实世界应用**（如机器人控制、自动化运维）。
4. **结合 memory-augmented agent 设计**，改善长程状态追踪能力（如 detect loop & backtrack）。
5. **建立 ARL 的 scaling law**：探索环境多样性、交互步数、数据量与性能之间的关系。

---

> 💡 **总结一句话**：  
> ARLArena 揭示了 ARL 不稳定的根本症结在于 **IS 裁剪不当** 与 **优势信号贫瘠**，并通过构建标准化测试床与四维分析框架，提出了首个真正实现**稳定、高效、可复现**的 agentic RL 训练方案 —— **SAMPO**。该工作为未来 LLM Agent 的规模化训练提供了坚实的方法论基础。

</details>

---

### 7. [Energy Efficient Federated Learning with Hyperdimensional Computing over Wireless Communication Networks](https://arxiv.org/abs/2602.21949)

**Authors**: Yahao Ding, Yinchao Yang, Jiaxiang Wang, Zhaohui Yang, Dusit Niyato, Zhu Han, Mohammad Shikh-Bahaei  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.21949v1  

#### Abstract
In this paper, we investigate a problem of minimizing total energy consumption for secure federated learning (FL) over wireless edge networks. To address the high computational cost and privacy challenges in conventional FL with neural networks (NN) for resource-constrained users, we propose a novel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Energy Efficient Federated Learning with Hyperdimensional Computing over Wireless Communication Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对**无线边缘网络中联邦学习（Federated Learning, FL）面临的两大核心挑战**：
- **高能耗问题**：资源受限的边缘设备在执行本地训练（尤其是基于神经网络 NN 的模型）时计算开销大，通信成本高。
- **隐私泄露风险**：标准 FL 虽不传输原始数据，但共享的模型更新仍可能被用于梯度反演等攻击。

现有方法通常单独优化资源分配或引入隐私机制，缺乏对**模型结构本身效率与系统级资源协同优化**的统一框架。

---

### ✅ 提出的新方法与新思路
作者提出了一种全新的 **FL-HDC-DP 框架**，其核心是将以下三种技术深度融合：

| 技术 | 作用 |
|------|------|
| **Hyperdimensional Computing (HDC)** | 替代传统 NN 进行本地训练，用简单的 hypervector 操作（如 bundling, binding）替代复杂的 backpropagation，显著降低计算复杂度和能耗。 |
| **Differential Privacy (DP)** | 在上传前向本地模型（即 Associative Memory, AM）添加高斯噪声，提供可量化的隐私保护。采用 **zCDP** 实现更紧致的隐私预算组合分析。 |
| **联合资源与模型维度优化** | 首次将 **HDC 维度 `d`** 作为可优化变量，与传输时间 `t`、带宽 `b`、发射功率 `p` 和 CPU 频率 `f` 一起进行端到端联合优化，最小化总能量消耗。 |

此外，为建模非线性的收敛轮数 `J_d` 与维度 `d` 的关系，提出一个 **sigmoid-variant 函数** 来拟合经验数据，使问题可解。

---

### ✅ 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **能效提升** | 相比传统的 FL-NN-DP 基线，总能耗最多减少 **83.3%**。 |
| **收敛速度更快** | 达到相同准确率所需通信轮数约为 NN 基线的 **1/3.5**，即快约 3.5 倍以上。 |
| **硬件友好性** | HDC 的操作适合低功耗、并行化硬件实现，更适合部署于 IoT 和移动设备。 |
| **理论建模创新** | 将 HDC 维度纳入优化变量，并建立 `d → J_d` 的经验模型，填补了“模型设计”与“系统资源”之间的鸿沟。 |

---

## 2. 核心实验方法和设置

### ✅ 数据集
- **MNIST 手写数字识别数据集**
  - 总样本数：60,000
  - 用户数量：`K = 50`
  - 每个用户样本数：1,200（IID 设置下均匀划分）
  - **Non-IID 设置**：标签排序后分为 150 个分片（shards），每个用户随机分配 3 个分片（共 1,200 张图像），造成严重的类别偏斜。

---

### ✅ 实验设置
| 参数 | 设置值 |
|------|--------|
| 通信模型 | 单小区，基站位于中心，用户分布在半径 500m 圆形区域 |
| 信道模型 | 大尺度路径损耗：`L(d) = 128.1 + 37.6 log₁₀(d)` dB |
| 多址方式 | FDMA（频分多址） |
| 总带宽 `B` | 10 MHz（默认） |
| 最大发射功率 `P_max` | 1 mW |
| 噪声谱密度 `N₀` | -174 dBm/Hz |
| CPU 频率上限 `f_max` | 2.3 GHz |
| 时间约束 `T` | 30 秒 |
| DP 参数 | `(ε, δ) = (20, 10⁻⁵)` 或 `(25, 10⁻⁵)`，预设噪声方差 |

---

### ✅ 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy vs. Epochs/Rounds** | 收敛速度与最终精度表现 |
| **Total Energy Consumption** | 所有用户的总能耗（含计算 + 通信） |
| **Convergence Rounds `J_d`** | 达到目标精度所需的通信轮数 |
| **Feasibility and Optimality** | 是否满足延迟与带宽约束下的最优资源配置 |

---

### ✅ 基线方法对比
| 基线方法 | 说明 |
|---------|------|
| **FL-NN-DP** | 使用传统神经网络（如 MLP）进行 FL，同样加入 DP 噪声，作为主要对比基线 |
| **Fixed-d Baseline** | 固定 HDC 维度（如 d=3000 或 d=5000），仅优化其他资源 |
| **Fixed-p Baseline** | 固定发射功率 `p = P_max`，不进行功率控制 |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据汇总
| 指标 | 数值/结论 |
|------|-----------|
| **最高准确率** | FL-HDC-DP 可达 **~90%**（在 d=10000, ε=20 下） |
| **收敛加速比** | 达到 90% 准确率，FL-HDC-DP 仅需约 **40 轮**，而 FL-NN-DP 需 **~140 轮** → 快 **3.5×** |
| **能耗降低幅度** | 相比固定维度基线，总能耗最多降低 **83.3%**（见 Fig. 8） |
| **最优维度选择** | 在实验设置下，最优 HDC 维度为 **d=4000**，并非越大越好（存在饱和效应） |

---

### ✅ 与基线方法的对比结果
#### 🔹 图 4 显示：
- 在相同隐私预算 `(ε=20)` 下：
  - FL-HDC-DP 在第 1 轮即可超过 80% 准确率；
  - FL-NN-DP 收敛缓慢，在前几十轮几乎无进展。
- 在 Non-IID 场景下：
  - FL-HDC-DP 依然保持更高稳定性和收敛速度；
  - FL-NN-DP 表现明显下降。

#### 🔹 图 6–8 显示：
- **联合优化方案（Proposed）始终优于所有固定参数基线**；
- 当 `P_max` 较小时，优化效果更显著（节省可达 40–46%）；
- 带宽从 0.5MHz 增至 10MHz，能耗持续下降，但边际收益递减。

---

### ✅ 消融实验与关键发现（隐含在实验中）
虽然未明确标注“ablation study”，但从多个图中可推断出以下消融结论：

| 控制变量 | 发现 |
|--------|------|
| **固定维度 vs 联合优化维度** | 维度选择对总能耗影响最大，远超带宽或功率优化带来的增益 |
| **是否使用 DP** | 加入 DP 后收敛变慢、波动增大，但通过增加维度可在一定程度上补偿精度损失 |
| **不同维度的影响** | 维度过小（如 3000）收敛慢；过大（如 10000）虽收敛快但每轮开销高，存在 **U型能耗曲线**，需权衡 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **HDC 是轻量化 FL 的理想候选者**：其简单向量运算大幅降低边缘设备的计算负担，特别适用于电池供电设备。
2. **模型维度 `d` 是关键设计自由度**：它直接影响收敛轮数 `J_d`，进而耦合计算与通信能耗，应作为优化变量而非固定超参。
3. **联合优化带来显著能效增益**：同时优化 `d`, `t`, `b`, `p`, `f` 可实现高达 **83.3% 的能耗削减**。
4. **存在最优维度**：由于“收敛加速”与“单轮开销上升”的权衡，**并非维度越高越好**，实验中最佳为 `d=4000`。
5. **提出的 sigmoid-variant 模型有效拟合 `d → J_d` 关系**：为后续研究提供了实用的经验建模工具。

---

### ⚠️ 方法的局限性
| 局限性 | 说明 |
|-------|------|
| **依赖经验建模** | `J_d(d)` 关系基于拟合而非解析推导，泛化能力受限于训练场景（数据集、目标精度、ε）。 |
| **假设理想同步** | 所有用户必须在同一轮完成计算与通信，可能导致“木桶效应”（慢用户拖累整体）。 |
| **仅考虑上行传输** | 未建模下行广播能耗，实际系统中也需考虑。 |
| **HDC 表达能力限制** | 对复杂任务（如 ImageNet 级别视觉任务）可能不如深度 NN 强大。 |

---

### 🔮 未来工作方向
1. **扩展至异步 FL 架构**：允许用户按自身节奏参与训练，缓解同步瓶颈。
2. **动态维度调整机制**：在训练过程中自适应调整 `d`，进一步节省早期阶段能耗。
3. **跨层联合设计**：结合信道编码、语义通信与 HDC，构建一体化的 ILAC（Integrated Learning and Communication）系统。
4. **硬件协同设计**：开发专用 HDC 加速器，充分发挥其并行性与低功耗潜力。
5. **应用于更多任务类型**：探索在语音、时间序列预测等领域的适用性。

---

> 📌 **一句话总结**：  
> 本论文提出了首个将 **HDC 模型维度纳入系统级资源联合优化** 的 FL 框架 **FL-HDC-DP**，实现了 **高达 83.3% 的能耗降低** 和 **3.5 倍以上的收敛加速**，为绿色、安全、高效的边缘智能提供了新范式。

</details>

---

### 8. [Interleaved Head Attention](https://arxiv.org/abs/2602.21371)

**Authors**: Sai Surya Duvvuri, Chanakya Ekbote, Rachit Bansal, Rishabh Tiwari, Devvrit Khatri, David Brandfonbrener, Paul Liang, Inderjit Dhillon, Manzil Zaheer  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.21371v1  

#### Abstract
Multi-Head Attention (MHA) is the core computational primitive underlying modern Large Language Models (LLMs). However, MHA suffers from a fundamental linear scaling limitation: $H$ attention heads produce exactly $H$ independent attention matrices, with no communication between heads during attenti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Interleaved Head Attention**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **Multi-Head Attention (MHA)** 存在**线性扩展瓶颈**：每个 attention head 独立计算，产生一个独立的 attention 矩阵，无法在 attention 计算过程中进行跨 head 的信息交互。
- 这种隔离限制了模型在**多步推理（multi-step reasoning）** 和**组合性任务**（如链式证据聚合、关系组合）中的表现，因为这些任务需要跨多个上下文片段的信息整合。

### **提出的新方法：Interleaved Head Attention (IHA)**
- **核心思想**：通过引入 **P 个 pseudo-heads（伪头）** 每个原始 head，实现跨 head 的混合（cross-head mixing）。
- 具体机制：
  - 对每个原始 head，生成 P 个 pseudo-query、pseudo-key、pseudo-value，它们是所有 H 个原始 head 的 Q/K/V 的**可学习线性组合**。
  - 在 attention 计算时，P 个 pseudo-query 与 P 个 pseudo-key 交互，**每个 head 可产生最多 $P^2$ 个 attention 模式**。
  - 最终通过 `merge_pseudo` 将 pseudo 维度合并到序列维度，执行标准 attention。
- **兼容性**：保留了标准 attention 操作符，因此兼容 FlashAttention 等高效 kernel。

### **相比现有方法的优势**
| 方面 | MHA | IHA |
|------|-----|-----|
| **表达能力** | H heads → H 个独立模式 | H heads → $H \times P^2$ 个潜在模式（当 $P=H$ 时为 $H^3$） |
| **参数效率** | 多任务需线性增加 heads | 通过 pseudo-head 实现**二次扩展**，显著减少所需 heads 数量 |
| **理论优势** | 需 $O(k)$ heads 表示 k 阶多项式滤波 | 仅需 $O(\sqrt{k})$ heads，参数从 $O(kn^2)$ 降至 $O(\sqrt{k}n^2)$ |
| **实际效率** | 参数开销小（仅 $O(H^2P)$） | 在 FLOPs 匹配下仍能提升性能 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 类型 | 数据集 | 说明 |
|------|--------|------|
| **长上下文建模** | **RULER** (Hsieh et al., 2024) | 测试长上下文下的检索能力，特别是 Multi-Key Retrieval |
| **数学推理** | **GSM8K**, **MATH-500** | 标准数学推理基准，测试多步逻辑能力 |
| **代码生成** | **MBPP**, **HumanEval** | 评估代码生成能力 |
| **合成任务** | **Polynomial Filter**, **CPM-3** | 控制变量的合成任务，用于理论验证 |

### **实验设置与评估指标**
- **模型架构**：
  - 2.4B 参数 decoder-only Transformer
  - 26 层，H=20 heads，head dim=128，RoPE 位置编码
  - 预训练长度 8k，微调至 64k
- **训练配置**：
  - 总训练 240B tokens，FSDP 分布式训练
  - 使用 **FLOP-matched** 设置确保公平比较
  - IHA 采用 **hybrid local-global schedule**（4 层滑动窗口 + 1 层全局）以控制计算成本
- **评估指标**：
  - **Exact Match (EM)**：RULER 上的准确率
  - **Pass@1, Majority@16**：GSM8K/MATH 上的推理准确率
  - **P@1, P@10**：MBPP/HumanEval 上的代码生成成功率

### **基线方法对比**
| 方法 | 说明 |
|------|------|
| **Global Attention** | 标准 MHA，全序列 attention |
| **Global+Local** | 混合局部滑动窗口与全局 attention |
| **Talking Heads** | 跨 head 混合 attention logits/weights |
| **Diff Transformer** | 差分 attention，增强模式对比 |
| **IHA (Ours)** | 本文方法 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **长上下文检索（RULER）**
| 方法 | Multi-Key Retrieval @4k | @8k | @16k | Overall EM |
|------|--------------------------|-----|------|------------|
| **IHA** | **+27%** | **+32%** | **+112%** | **44.0%** |
| Global Attention | 35.0% | 30.0% | 15.0% | 35.0% |
| Global+Local | — | — | — | 40.6% |
| Diff Transformer | — | — | — | 37.2% |

> ✅ **IHA 在长上下文检索上全面领先，尤其在 16k 长度下提升超 100%**

#### **数学推理（SFT 后）**
| 方法 | GSM8K P@1 | GSM8K Maj@16 | MATH-500 P@1 | MATH-500 Maj@16 |
|------|-----------|---------------|---------------|------------------|
| **IHA** | 34.3% (**+4.8**) | **54.2% (+5.8)** | 10.0% (+1.2) | **18.4% (+2.8)** |
| Global Attention | 29.5% | 48.4% | 8.8% | 15.6% |
| Talking Heads | 29.3% | 49.4% | 7.8% | 18.2% |
| Diff Transformer | 31.6% | 53.5% | 9.0% | 18.0% |

> ✅ **IHA 在数学推理上显著优于所有 baseline，GSM8K 提升 5.8%，MATH 提升 2.8%**

#### **预训练阶段（5-shot）**
| 方法 | GSM8K EM | MATH-500 EM | Avg. Rank↓ |
|------|----------|--------------|------------|
| **IHA** | **8.34% (+2.73)** | **3.54% (+0.66)** | **1.4** |
| Global Attention | 5.61% | 2.88% | 2.9 |
| Talking Heads | 5.46% | — | 4.0 |

> ✅ **即使未微调，IHA 也表现出更强的推理泛化能力**

### **消融实验（来自附录 D）**
- **合成任务：Binary/Ternary Relation Composition**
  - IHA 在两种任务上均优于 MHA 和 Simplicial Attention
  - 最高提升达 **+4.7%**（二元关系）和 **+3.3%**（三元关系）
- **证明 IHA 更擅长建模复杂的关系组合与多跳推理**

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **IHA 严格广义于 MHA**：
   - 理论证明：IHA 是 MHA 的严格超集（$M \subset P_P$），且当 $P \geq 2$ 时为真子集。
2. ✅ **更高的参数效率**：
   - 在 Polynomial Filter 任务中，IHA 用 $O(\sqrt{k})$ heads 即可达到 MHA 需 $O(k)$ heads 的表达能力。
   - 在 CPM-3 任务中，IHA 仅需 $[\sqrt{N_{\text{max}}}]$ heads，而 MHA 需 $N_{\text{max}}$。
3. ✅ **实证性能提升显著**：
   - 长上下文检索提升 **10–20%**（4k–16k）
   - 数学推理提升 **+5.8% (GSM8K)** 和 **+2.8% (MATH-500)**
   - 代码生成次优但稳定，表明其优势集中在**逻辑推理**而非函数生成

### **方法的局限性**
- **计算开销**：全局 IHA 的 attention 复杂度为 $O((NP)^2d)$，即 $P^2$ 倍于 MHA。
  - 通过 hybrid local-global schedule 缓解，但仍可能影响极长序列部署。
- **固定 P**：当前设置 $P=H$，未探索自适应 pseudo-head 数量。
- **仅限 decoder 架构**：尚未在 encoder-decoder 或 vision 模型中验证。

### **未来工作方向**
1. **动态 pseudo-head 分配**：根据输入复杂度自适应调整 $P$。
2. **扩展至 encoder-decoder 架构**：如用于机器翻译或摘要。
3. **结合其他 attention 变体**：如与 MQA/GQA 结合进一步优化 KV cache。
4. **理论深化**：分析 IHA 在 Transformer 完备性（universal approximation）中的角色。

---

> **总结**：  
> **Interleaved Head Attention (IHA)** 通过引入 pseudo-head 实现跨 head 混合，打破了 MHA 的线性扩展瓶颈，在**多步推理、长上下文建模**等任务上展现出显著的理论与实证优势。其设计简洁、兼容性强，是提升 Transformer 推理能力的一个有前景的方向。

</details>

---

### 9. [Structured Prompt Language: Declarative Context Management for LLMs](https://arxiv.org/abs/2602.21257)

**Authors**: Wen G. Gong  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21257v1  

#### Abstract
We present SPL (Structured Prompt Language), a declarative SQL-inspired language that treats large language models as generative knowledge bases and their context windows as constrained resources. SPL provides explicit WITH BUDGET/LIMIT token management, an automatic query optimizer, EXPLAIN transpa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Structured Prompt Language: Declarative Context Management for LLMs》核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **prompt engineering** 实践存在以下痛点：
- **手动 token 计数**：开发者需反复试错以确保提示在 context window 内。
- **缺乏优化机制**：无系统性策略处理超长上下文，常采用粗暴截断。
- **不可见性**：无法预知 token 在各组件间的分配情况。
- **平台锁定**（Provider lock-in）：代码依赖特定 LLM API，迁移成本高。
- **不可组合性**：提示（prompt）多为单体结构，难以模块化复用。

### 提出的新方法：SPL (Structured Prompt Language)
SPL 是一种受 SQL 启发的**声明式语言**，将 LLM 视为“生成型知识库”（generative knowledge base），其 context window 被视为受限资源进行管理。

#### 核心创新点
| 创新点 | 描述 |
|-------|------|
| **声明式语法** | 开发者只需声明“需要什么上下文”和“生成什么”，而非如何组装提示。 |
| **显式 Token 预算管理** | `WITH BUDGET`, `LIMIT` 等关键字强制在执行前明确 token 分配。 |
| **自动查询优化器** | 类似于数据库的 query optimizer，自动处理 token 分配、压缩和执行顺序。 |
| **EXPLAIN 机制** | 提供类似 SQL `EXPLAIN ANALYZE` 的功能，在执行前展示 token 分配计划和成本预估。 |
| **集成 RAG 与持久化内存** | 原生支持 `rag.query()` 和 `memory.get()`，实现一体化上下文管理。 |
| **SPL-flow** | 扩展为弹性代理管道（agentic pipelines），具备三级回退策略（Ollama → OpenRouter → 自愈重试）。 |

### 相比现有方法的优势
| 方法 | 主要区别 | SPL 优势 |
|------|----------|---------|
| **Prompty** | YAML 模板格式，无全局预算和优化 | SPL 提供预算管理、优化和跨平台可移植性 |
| **DSPy** | 优化提示内容（prompt content） | SPL 优化资源（token budget），两者互补 |
| **LMQL** | 约束生成输出 | SPL 管理输入上下文，关注点不同 |
| **LangChain/LlamaIndex** | 命令式 API，需手动管理 token | SPL 为声明式，提供更高层次的抽象和自动化 |

---

## 2. 核心实验方法和设置

### 实验设置
所有实验均在**不调用真实 LLM API** 的情况下运行，仅测试 SPL 的解析、分析、优化和 `EXPLAIN` 流程，验证其在“零 token 成本”下的价值。

### 评估指标
| 指标 | 描述 |
|------|------|
| **LoC (Lines of Code)** | 衡量开发复杂度 |
| **Token Operations** | 手动 token 计数、截断等操作次数 |
| **预算可见性**（Budget Visibility） | 是否支持 `EXPLAIN` 查看 token 分配 |
| **静态可验证性** | 是否可在执行前进行语法和语义校验 |
| **预估成本**（Estimated Cost） | 基于目标模型价格预估执行成本 |

### 基线方法对比
- **SPL**：本文提出的方法。
- **Imperative Python**：使用 Python 手动拼接字符串、调用 `tiktoken` 计数、显式截断和缓存的基准实现。

### 对比任务
1. 简单问答（Simple QA）
2. RAG 增强问答（RAG-augmented QA）
3. 多步骤 CTE 查询（Multi-step CTE）
4. 函数复用（Function Reuse）
5. 缓存重复查询（Cached Repeat）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 任务 | SPL LoC | Python LoC | **代码减少** | **Token 操作减少** |
|------|--------|-----------|-------------|------------------|
| 简单问答 | 9 | 20 | 55.0% | 0/4 |
| RAG 问答 | 17 | 51 | 66.7% | 0/9 |
| 多步骤 CTE | 24 | 63 | 61.9% | 0/7 |
| 函数复用 | 16 | 43 | 62.8% | 0/11 |
| 缓存重复 | 9 | 42 | 78.6% | 0/4 |
| **平均** | **15** | **44** | **65.0%** | **0/7** |

> ✅ **SPL 平均减少 65% 的代码行数，并完全消除了所有 35 个手动 token 操作。**

### 与基线方法的对比结果
- **开发效率**：SPL 显著降低开发负担，尤其在缓存和复杂流程中优势明显。
- **可观测性**：SPL 支持 `EXPLAIN`，而 Python 基准完全不具备。
- **静态验证**：SPL 可在执行前验证语法和预算，Python 基准只能在运行时报错。

### 消融实验结果
#### (1) Token 预算优化行为
- 在预算从 2K 到 32K tokens 变化时，优化器能**按比例分配** RAG 和 memory 的 token，而 system role 和 question 保持固定。
- 当上下文需求超过预算时，优化器采用**最大优先压缩策略**，优先压缩最大的源（如 RAG 文档），保留小的固定项。

#### (2) 跨模型成本预估
对同一 8K token 查询在不同模型上运行 `EXPLAIN`，结果显示：
| 模型 | 预估成本 | 相对成本 |
|------|---------|---------|
| GPT-4 (Legacy) | $0.2316 | 67.5x |
| Claude Opus 4.6 | $0.2058 | 60.0x |
| Claude Sonnet 4.5 | $0.0412 | 12.0x |
| GPT-4o | $0.0293 | 8.5x |
| GPT-3.5 Turbo | $0.0049 | 1.4x |
| **Claude Haiku 4.5** | **$0.0034** | **1.0x** |

> 💡 **最贵与最便宜模型间存在 68× 的成本差异，且该信息在执行前即可通过 `EXPLAIN` 获取。**

---

## 4. 关键结论和发现

### 主要发现
1. **SPL 将 prompt engineering 从命令式实践转变为结构化、可优化的工程学科**。
2. **声明式 + 自动优化 + 可观测性** 的组合显著提升了开发效率和资源利用率。
3. **相同的 `.spl` 脚本可在本地 Ollama（$0 成本）和云服务（如 OpenRouter）上无缝运行**，实现真正的平台无关性。
4. **逻辑分块**（Logical Chunking）结合 CTE 可将 transformer 注意力计算从 O(N²) 降至 O(N²/k)，大幅提升长文档处理效率。
5. **BENCHMARK** 功能支持并行多模型比较，并自动选择最优模型用于后续执行。

### 方法的局限性
| 局限性 | 描述 |
|--------|------|
| **轻微超预算仍用截断** | 当单个 CTE 轻微超限时，仍采用简单截断，未来可引入语义压缩。 |
| **类型系统有限** | 尚未支持上下文源的类型注解，限制了静态分析能力。 |
| **MoM 路由基于关键词** | 当前混合模型（MoM）路由依赖规则匹配，对模糊或混合领域查询较脆弱。 |
| **多轮对话管理不足** | 虽支持 `memory.get("history")`，但缺乏正式的会话剪枝或摘要策略。 |

### 未来工作方向
1. **学习型优化器**：基于历史执行轨迹训练 token 分配预测模型。
2. **动态 MoM 路由**：使用轻量级模型进行任务分类，替代关键词匹配。
3. **与现有生态集成**：
   - **SPL + DSPy**：用 DSPy 优化 SPL 中的提示内容。
   - **SPL + LangChain/LlamaIndex**：作为声明式编排层。
4. **IDE 支持**：为 VS Code、JetBrains 等提供语法高亮、补全和内联 `EXPLAIN`。
5. **安全增强**：
   - 字面量内容过滤
   - 模型白名单控制
   - RAG 注入防护

> 🔗 **项目开源地址**：
> - SPL: [https://github.com/digital-duck/SPL](https://github.com/digital-duck/SPL) (`pip install spl-llm`)
> - SPL-flow: [https://github.com/digital-duck/SPL-flow](https://github.com/digital-duck/SPL-flow) (`pip install spl-flow`)

</details>

---

### 10. [Multi-dimensional Assessment and Explainable Feedback for Counselor Responses to Client Resistance in Text-based Counseling with LLMs](https://arxiv.org/abs/2602.21638)

**Authors**: Anqi Li, Ruihan Wang, Zhaoming Chen, Yuqian Chen, Yu Lu, Yi Zhu, Yuan Xie, Zhenzhong Lan  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21638v1  

#### Abstract
Effectively addressing client resistance is a sophisticated clinical skill in psychological counseling, yet practitioners often lack timely and scalable supervisory feedback to refine their approaches. Although current NLP research has examined overall counseling quality and general therapeutic skil...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文聚焦于**心理咨询服务中一个高风险且复杂的临床挑战——来访者抵抗（client resistance）**。当来访者表现出抗拒、回避或反对治疗进程时，咨询师如何有效回应至关重要。然而，传统督导模式反馈稀疏、延迟且不可扩展，导致咨询师难以及时识别并改进低效干预策略。

现有 NLP 研究多关注整体咨询质量或单一技能（如共情），缺乏对“应对抵抗”这一具体情境下**多维度、细粒度评估**的支持。

### 提出的新方法与新思路
本研究提出了一套完整的计算框架，包含以下三大创新：

- **理论驱动的四维评估框架（Four-dimensional Framework）**  
  将咨询师回应分解为四个核心沟通机制（communication mechanisms）：
  - **Respect for Autonomy**（尊重自主性）
  - **Stance Alignment**（立场对齐）
  - **Emotional Resonance**（情感共鸣）
  - **Conversational Orientation**（对话导向）  
  每个维度进一步划分为三个表达强度等级：No / Weak / Strong，实现精细化评分。

- **高质量标注数据集构建**  
  基于真实文本咨询对话，由持证心理咨询师人工标注 3,836 条“来访者抵抗—咨询师回应”样本，每条均含四维标签及自然语言解释（explanatory rationales），确保标注的专业性和可解释性。

- **可解释反馈生成模型（Explainable Feedback Generation）**  
  在 Llama-3.1-8B-Instruct 模型上进行全参数指令微调（full-parameter instruction tuning），不仅预测各维度得分，还能生成与专家一致的高质量解释性反馈。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **评估粒度** | 宏观质量或单一技能 | 针对“抵抗”场景的多维细粒度分析 |
| **反馈形式** | 数值打分或通用建议 | 包含理论依据、证据锚定、具体改进建议的自然语言解释 |
| **实用性** | 依赖专家人力，难以规模化 | 可自动化提供即时、个性化反馈，支持实时训练 |

---

## 2. 核心实验方法和设置

### 数据集来源
- 主要来自两个公开可用的心理咨询对话数据集：
  - **ClientBehavior** [18]
  - **ObserverWAI** [6]
- 所有对话均为中文在线平台上的**text-based counseling**记录，已去除个人身份信息。
- 使用 **RECAP** [24] 自动检测来访者的 resistance utterance，并提取其后紧接的咨询师回应，构成“抵抗上下文 + 回应”样本对。

### 标注流程
- **标注人员**：5 名具备十年以上临床经验的持证咨询师。
- **任务设计**：
  1. 对每个咨询师回应在四个 communication mechanisms 上打分（0/1/2）；
  2. 编写详细解释 rationale，涵盖：
     - 来访者抵抗行为分析
     - 咨询师回应的关键特征
     - 分类依据（是否符合框架定义）
- **质量控制**：
  - 双人独立标注，分歧由第三人仲裁；
  - Cohen’s Kappa 在 0.74–0.77 之间，表明**实质性一致性（substantial agreement）**；
  - 抽样评估解释质量，在 framework consistency、evidence anchoring 和 clarity & specificity 上平均得分达 2.8+/3.0。

### 实验设置
- **模型架构**：基于 **Llama-3.1-8B-Instruct** 进行 full-parameter fine-tuning。
- **训练配置**：
  - 5-fold cross-validation
  - 学习率：1e-5，早停机制防止过拟合
  - 每 fold 训练 3 轮
  - 推理时使用 deterministic decoding（temperature=0）
- **任务目标**：
  - 多标签分类：判断每个 communication mechanism 的表达水平（no/weak/strong）
  - 同时生成解释性文本

### 评估指标
#### 分类性能
- **Macro-F1 Score**
- **Accuracy**

#### 解释生成质量
- **自动评估**：
  - BLEU-1/2
  - ROUGE-1/2/L
- **人工评估**（三维度 Likert 3分制）：
  - Framework Consistency
  - Evidence Anchoring
  - Clarity & Specificity

### 基线方法对比
| 类型 | 模型列表 |
|------|----------|
| **Closed-source LLMs (Zero-shot)** | GPT-4o, Claude-3.5-Sonnet |
| **Open-source LLMs (Zero-shot)** | Qwen2.5-7B/14B/32B/72B-Instruct, Llama-3.1-8B/70B-Instruct |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table II）

| 模型 | Respect for Autonomy (F1) | Stance Alignment (F1) | Emotional Resonance (F1) | Conversational Orientation (F1) |
|------|----------------------------|------------------------|---------------------------|----------------------------------|
| GPT-4o | 45.37 | 58.61 | 53.16 | 41.72 |
| Claude-3.5-Sonnet | 41.61 | 52.59 | 55.59 | 45.36 |
| **Our Model** | **80.92** | **77.56** | **77.34** | **77.87** |

> ✅ **提升显著**：相比最强 baseline（GPT-4o），在所有维度上提升 **20+ F1 points**，最高达 **+35.5 F1**（Respect for Autonomy）。

### 与基线方法的对比结果
- **分类任务**：
  - 我们的模型 Macro-F1 达到 **77–81%**，远超 GPT-4o 和 Claude-3.5-Sonnet 的 **45–59%**。
  - 即使是更大的开源模型（如 Qwen2.5-72B-Instruct）也仅达到 ~38% F1，说明**任务特定数据微调远胜参数规模优势**。
- **解释生成任务**（Table III）：
  - **BLEU-1**：Our Model (**0.60**) vs. Claude-3.5-Sonnet (**0.32**) → **接近翻倍**
  - **人类评分**（满分3.0）：
    - Our Model：**2.78–2.88**
    - GPT-4o：~2.0–2.4
    - 表明生成的解释在理论一致性、证据支撑和清晰度方面接近专家水平。

### 消融实验结果（Ablation Study）
- 移除训练中的 textual explanations（即只用标签监督）后，模型性能下降约 **4 F1 points**。
- 结果证明：**显式解释（explanations）作为监督信号能显著增强模型学习效果**，帮助其掌握深层沟通原则而不仅是表面模式匹配。

---

## 4. 关键结论和发现

### 主要发现
1. **多维框架具有高度实用性和区分力**  
   四维机制能够系统刻画咨询师在面对抵抗时的有效干预路径，为反馈提供明确操作指南。

2. **专用微调模型显著优于通用 LLM**  
   尽管 GPT-4o 和 Claude 表现尚可，但在复杂临床判断任务上仍远逊于经过专业数据训练的小型模型，凸显**领域适配的重要性**。

3. **解释性反馈本身具备高价值**  
   模型不仅能准确评分，更能生成接近专家水准的解释，且这些解释被验证能有效指导实践。

4. **AI 反馈切实提升咨询能力（Proof-of-Concept Study）**
   - 实验组（接收 AI 反馈）在 post-test 中各项维度得分显著提高（β = 0.20–0.30, p < 0.001）
   - 控制组无明显变化
   - 半结构化访谈显示，参与者认为反馈有助于：
     - 提升自我觉察（M=4.38/5）
     - 明确改进方向（M=4.14/5）
     - 增强处理抵抗的信心（M=3.86/5）

### 方法的局限性
- 当前数据集基于中文文本咨询，跨文化、跨语言泛化能力待验证。
- 评估集中于“单轮回应”，未建模长期咨询动态演变。
- 模型输出依赖训练数据分布，可能继承潜在偏见。
- 尚未集成实际咨询系统中进行长期追踪干预效果。

### 未来工作方向
- 引入更多真实案例，拓展至视频/语音咨询场景。
- 开发**生成替代回应（alternative response generation）** 功能，提供更具体的示范。
- 探索将模型嵌入实时咨询辅助系统，支持“边聊边反馈”。
- 加强多模态理解能力，结合语调、停顿等非语言线索（适用于语音咨询）。
- 推动开放共享：计划发布框架、数据集与模型以促进社区发展。

---

> 🔚 **总结一句话**：  
> 该研究通过构建理论驱动的多维评估体系与高质量标注数据，成功训练出能精准评价并解释咨询师如何应对来访者抵抗的 LLM 模型，实验证明其反馈可显著提升咨询师实战能力，为心理服务的智能化督导提供了可扩展、可解释的新范式。

</details>

---

### 11. [Geometric Priors for Generalizable World Models via Vector Symbolic Architecture](https://arxiv.org/abs/2602.21467)

**Authors**: William Youngwoo Chung, Calvin Yeung, Hansen Jin Lillemark, Zhuowen Zou, Xiangjian Liu, Mohsen Imani  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21467v1  

#### Abstract
A key challenge in artificial intelligence and neuroscience is understanding how neural systems learn representations that capture the underlying dynamics of the world. Most world models represent the transition function with unstructured neural networks, limiting interpretability, sample efficiency...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Geometric Priors for Generalizable World Models via Vector Symbolic Architecture*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **World Models** 多采用无结构的神经网络（如MLP）建模状态转移函数 $T: S \times A \rightarrow S$，存在以下缺陷：
- **样本效率低**：需要大量训练数据。
- **泛化能力差**：难以推广到未见过的状态-动作对（zero-shot generalization）。
- **误差累积严重**：在长时程 rollout 中预测误差快速积累。
- **缺乏可解释性**：隐空间无明确几何意义，难以进行符号级推理。

### 🚀 提出的新方法与创新思路
本文提出一种基于 **Vector Symbolic Architecture (VSA)** 的通用世界模型框架，引入**几何先验**（geometric priors），将环境动力学建模为**群作用**（group action）在高维复向量空间中的代数操作。

核心思想包括：
- 使用 **Fourier Holographic Reduced Representation (FHRR)** 编码器将状态 $s$ 和动作 $a$ 映射为单位复向量（unitary complex vectors）。
- 状态转移通过**元素级复数乘法**（element-wise complex multiplication）实现绑定（binding）：
  $$
  \phi_s(s_{t+1}) = \phi_s(s_t) \odot \phi_A(a_t)
  $$
- 引入 **cleanup 机制**：利用高维空间中向量近似正交的性质，通过相似性搜索纠正预测噪声，抑制误差传播。
- 在训练中加入**结构正则项**（invertibility, orthogonality），鼓励学习具有群结构的表示。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **泛化性** | 支持 zero-shot 推理，能处理未见状态-动作组合 |
| **鲁棒性** | 对输入噪声高度容忍，性能下降缓慢 |
| **长时序稳定性** | cleanup 抑制误差累积，rollout 更稳定 |
| **可解释性** | 动作对应于隐空间中的“平移”操作，具备语义一致性 |
| **计算效率** | 所有操作均为 element-wise，训练和推理为线性复杂度 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **GridWorld Environment**：一个 $10 \times 10$ 的离散网格世界，共 100 个状态，4 个确定性动作（上下左右）。
- 转移函数完全由规则定义，构成一个自然的群结构（$\mathbb{Z}^2$ 上的平移群子集）。

### ⚙️ 实验设置
- **训练数据划分**：
  - 使用 80% 的 $(s, a)$ 对进行训练。
  - 保留 20% 作为 **zero-shot 测试集**（即这些组合从未出现在训练中）。
- **模型结构**：
  - **VSA-FHRR 模型**：状态与动作分别编码为 $D=512$ 维的 FHRR 向量，transition 用 $\odot$ 实现。
  - **Baseline MLP 模型**：
    - MLP-Small：2 层，隐藏层大小 128
    - MLP-Medium：4 层，256
    - MLP-Large：6 层，512
    - 输入为拼接后的 $(s, a)$，输出为下一状态预测。
- **训练目标**：
  - 主损失：Binding Loss（MSE）  
    $$
    \mathcal{L}_{\text{bind}} = \|\phi_s(s_{t+1}) - \phi_s(s_t) \odot \phi_A(a_t)\|^2
    $$
  - 正则项：
    - Invertibility：$\mathcal{L}_{\text{inv}} = \|\phi_A(a) \odot \phi_A(a^{-1}) - \mathbf{1}\|$
    - Orthogonality：$\mathcal{L}_{\text{ortho}} = \sum_{s \neq s'} |\langle \phi_s(s), \phi_s(s') \rangle|$
- **总目标函数**：
  $$
  \mathcal{L} = \lambda_{\text{bind}}\mathcal{L}_{\text{bind}} + \lambda_{\text{inv}}\mathcal{L}_{\text{inv}} + \lambda_{\text{ortho}}\mathcal{L}_{\text{ortho}}
  $$

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **1-step Accuracy** | 单步预测正确率 |
| **Zero-shot Accuracy** | 在未见 $(s,a)$ 对上的预测准确率 |
| **Cosine Similarity** | 预测向量与真实嵌入之间的余弦相似度 |
| **Rollout Accuracy** | 多步 rollout（如 5, 20, 100 步）后仍保持正确的比例 |
| **+Cleanup Rollout** | 每隔若干步执行一次 cleanup 的 rollout 表现 |
| **Robustness to Noise** | 添加高斯噪声后的一阶动态预测准确性 |
| **t-SNE 可视化** | 观察隐空间是否保留原始网格结构 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 任务 | FHRR (Ours) | MLP-Small | MLP-Medium | MLP-Large |
|------|-------------|-----------|------------|-----------|
| **1-step Accuracy** | **96.3%** | 80.0% | 80.0% | 80.25% |
| **1-step Zero-Shot Acc** | **87.5%** | 0.0% | 0.0% | 1.25% |
| **Cosine Sim (Zero-Shot)** | **80.5** | 0.9 | 0.15 | 3.1 |
| **Rollout (5 steps)** | **74.6%** | 39.8% | 38.0% | 40.8% |
| **Rollout (20 steps)** | **34.6%** | 2.0% | 4.0% | 6.2% |
| **Rollout (20 steps +Clean)** | **61.4%** | 5.4% | 7.8% | 8.4% |
| **Rollout (100 steps)** | 1.8% | 0.8% | 1.8% | 2.0% |
| **Rollout (100 steps +Clean)** | **38.6%** | 2.8% | 4.0% | 3.2% |

> 💡 **亮点总结**：
> - Zero-shot 准确率达 **87.5%**，而所有 MLP 基线几乎为零。
> - 20-step rollout 性能高出 MLP **53.6% 绝对值**（61.4% vs 6.2% with cleanup）。
> - cleanup 使 FHRR 在长程 rollout 中提升达 **+26.8%**（20步），体现其纠错能力。
> - 即使是最小的 MLP（参数更少），也无法匹敌 FHRR 的泛化表现。

### 🔬 消融实验结果（Figure 6）
- **维度影响**：随着 embedding dimension $D$ 增加，robustness 显著提升，验证了高维空间下 cleanup 的有效性。
- **正交性权重**：适当增加 $\lambda_{\text{ortho}}$ 有助于提高鲁棒性，但过大会损害训练。
- **参数规模对比**：
  - VSA-FHRR 参数量 ≈ 53K，仅略高于 MLP-Small（41.6K），远小于 MLP-Large（1.39M）。
  - 尽管参数更少，VSA 在 inference + cleanup 场景下仍保持更快或相当的速度（见 Table 2）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **结构化表示显著提升泛化能力**：
   - 利用 VSA 的代数结构（尤其是 FHRR 的相位编码与乘法绑定），模型能够自然地支持多步动作组合、逆向推理和 zero-shot 推广。
   
2. **latent space 具备显式几何意义**：
   - t-SNE 可视化显示 FHRR 学到了接近真实网格拓扑的结构，而 MLP 完全混乱。
   - 动作在隐空间中表现为一致的“平移”，形成局部光滑的相似性核（similarity kernel）。

3. **cleanup 是抑制误差累积的关键机制**：
   - 在每 2 步执行 cleanup 可大幅提升长程 rollout 准确率，尤其在 zero-shot 或噪声环境下效果显著。
   - 这是传统 MLP 架构无法轻易复制的功能。

4. **高效且硬件友好**：
   - 所有操作均为 element-wise，适合并行加速。
   - 参数量小、内存占用低，适用于边缘设备（edge computing）。

### ⚠️ 方法的局限性
- 当前仅验证于**小型离散环境**（GridWorld），尚未扩展至连续、随机或部分可观测环境。
- FHRR 假设动作具有精确的群结构，在现实环境中可能不成立（需近似建模）。
- cleanup 依赖 codebook 查找，在大规模状态空间中可能成为瓶颈（可用 ANN 近似解决）。

### 🔮 未来工作方向
- 将该框架集成到 **Model-Based RL** 中，用于实际规划与决策。
- 探索如何将 VSA 扩展到**连续动作空间**与**图像输入**（如结合 CNN/ViT 提取特征后再编码为 VSA 向量）。
- 研究更高效的 cleanup 方案（如哈希表、近似最近邻）以适应大规模状态空间。
- 探索与其他 GDL 方法（如 equivariant networks）的融合路径。

---

## 总结
本论文成功展示了 **Vector Symbolic Architecture** 作为一种构建**可泛化、可解释、鲁棒性强的世界模型**的有效范式。通过引入 **FHRR 编码 + 元素级复数乘法 + cleanup 机制**，实现了在离散环境中卓越的 zero-shot 泛化、长程 rollout 稳定性和抗噪能力，为迈向真正意义上的结构化、类人认知的世界建模提供了新的理论与实践路径。

</details>

---

### 12. [Mamba Meets Scheduling: Learning to Solve Flexible Job Shop Scheduling with Efficient Sequence Modeling](https://arxiv.org/abs/2602.21546)

**Authors**: Zhi Cao, Cong Zhang, Yaoxin Wu, Yaqing Hou, Hongwei Ge  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21546v1  

#### Abstract
The Flexible Job Shop Problem (FJSP) is a well-studied combinatorial optimization problem with extensive applications for manufacturing and production scheduling. It involves assigning jobs to various machines to optimize criteria, such as minimizing total completion time. Current learning-based met...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Mamba Meets Scheduling: Learning to Solve Flexible Job Shop Scheduling with Efficient Sequence Modeling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文聚焦于**Flexible Job Shop Problem (FJSP)**，即柔性作业车间调度问题。该问题是组合优化中的经典难题，广泛应用于制造和生产调度领域。其目标是在多台异构机器上安排多个作业的操作，以最小化总完成时间（makespan）等目标。

传统学习型方法（如基于图神经网络 GNN 或图注意力机制 Graph Attention）存在以下瓶颈：
- **局部特征提取限制**：仅在操作或机器的邻域内进行消息传递，难以捕捉全局依赖关系。
- **计算复杂度高**：图注意力机制具有 $O(n^2)$ 复杂度，对大规模实例效率低下。
- **图结构设计复杂**：需要精心构造 disjunctive graph，增加了建模难度。

---

### 🚀 提出的新方法与创新思路

作者提出了一种名为 **Mamba-CrossAttention (M-CA)** 的新型神经网络架构，核心思想是将 FJSP 建模为一个**序列决策问题**，并利用高效的序列建模能力替代复杂的图结构。

#### 主要组件：
1. **Dual Mamba Encoder (DME)**  
   - 使用两个独立的 **Mamba 模块** 分别处理操作（operation）和机器（machine）的原始特征序列。
   - 利用 Mamba 的 **Selective State Space Model (SSM)** 结构，在线性时间内捕获长距离依赖。
   - 避免手动定义图拓扑，直接从完整序列中学习表示。

2. **Cross-Attention Decoder**  
   - 引入轻量级的 **cross-attention 机制** 融合操作与机器的嵌入。
   - 实现双向交互：模拟“选择哪个机器执行某操作”以及“选择哪个操作分配给某机器”。
   - 计算复杂度为 $O(|O| \times |M|)$，远低于 self-attention 的 $O(|O|^2)$，尤其适用于 $|O| \gg |M|$ 的工业场景。

3. **端到端强化学习框架**  
   - 构建基于 **PPO (Proximal Policy Optimization)** 的 Actor-Critic 框架。
   - 决策网络结合全局特征与 operation-machine pair 特征，输出动作概率分布。

---

### 🔍 相比现有方法的优势

| 维度 | 优势说明 |
|------|----------|
| **建模方式** | 放弃复杂图结构，采用纯序列建模，简化状态表示设计 |
| **表达能力** | 学习整个操作/机器序列的全局上下文，突破局部邻域限制 |
| **计算效率** | Mamba 具有线性时间复杂度 $O(N)$，显著优于图注意力模型 |
| **可扩展性** | 在超大规模实例（如 10000×10）上仍能运行，而多数基线因内存溢出失败 |

---

## 2. 核心实验方法和设置

### 📊 数据集
实验涵盖两类数据集：

#### （1）公开基准数据集（Public Benchmarks）
| 数据集 | 来源 | 规模范围 | 实例数 |
|-------|------|--------|--------|
| Brandimarte | [2] | 10×6 ~ 20×15 | 10 |
| Hurink (rdata/edata/vdata) | [12] | 10×5 ~ 30×10 | 各40 |
| Barnes | [1] | 10×11 ~ 15×17 | 21 |
| Dauzère | [6] | 10×5 ~ 20×10 | 18 |

#### （2）合成数据集（Synthetic Data）
- 生成规则参考 [27]，处理时间从 $U(1,20)$ 均匀采样。
- 包括：`FJSP10x5`, `20x5`, `15x10`, `20x10`, `30x10`, `40x10`
- 泛化测试使用更大规模：`100x10` 至 `10000x10`

---

### 🧪 实验设置与评估指标

#### 模型训练配置
- 使用单张 RTX 3090 GPU
- 模型维度：128；Cross-attention heads：8
- 优化器：Adam；学习率：$1\times10^{-4}$ → $1\times10^{-5}$
- 训练迭代：10,000 次，每轮 20 个实例
- 验证频率：每 10 次迭代验证一次，保存最优权重

#### 动作策略
- **Greedy Strategy (-G)**：每步选择最高概率的动作（用于验证）
- **Sampling Strategy (-S)**：并行采样 100 条轨迹，取最优解（用于测试）

#### 评估指标
- **Makespan (Obj)**：总完成时间
- **Gap (%)**：相对已知最优解的差距  
  $$
  \text{gap} = \left(\frac{C}{C_{\text{best}}} - 1\right) \times 100\%
  $$
- **Time (s)**：求解耗时（秒）
- **内存占用**：GPU/CPU 内存消耗

---

### ⚔️ 基线方法对比

| 类别 | 方法 | 描述 |
|------|------|------|
| **Exact Solver** | OR-Tools | Google 开源求解器，设 1800 秒超时 |
| **Priority Dispatching Rules (PDRs)** | FIFO, MOPNR, SPT, MWKR | 工业常用启发式规则 |
| **Learning-based Methods** | HGNN-G/S [27] | 基于异构图神经网络 |
| | DAN-G/S [31] | 图注意力网络，当前 SOTA |
| | MLP* [32] | 轻量级 MLP 模型（复现原文结果） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能表现（来自 Table 1 & 2）

#### 在合成数据上的结果（FJSP20x10）
| 方法 | Obj | Gap | Time (s) |
|------|-----|-----|----------|
| OR-Tools | 195.98 | 0.00% | 1805 |
| DAN-S | 195.14 | -0.4% | 4.97 |
| **M-CA-S (ours)** | **189.83** | **-3.11%** | **4.61** |

✅ **M-CA-S 不仅快于 OR-Tools 近 400 倍，且质量更优！**

#### 更大尺度泛化能力（FJSP1000x10）
| 方法 | Obj | Gap | Time (s) | 内存 |
|------|-----|-----|----------|------|
| OR-Tools | — | — | >3600 | OOM-CPU |
| DAN | 9157.8 | -40.36% | 140.30 | 362MB |
| **M-CA** | **9038.1** | **-41.15%** | **120.44** | 424MB |

✅ **M-CA 在千级任务下仍稳定运行，并超越精确求解器近 41%**

---

### 🆚 与其他学习方法对比（Greedy Setting）

| 方法 | 平均 Gap ↓ | 求解速度 ↑ |
|------|------------|-----------|
| HGNN-G | ~15–25% | 较慢 |
| DAN-G | ~5–12% | 中等 |
| **M-CA-G** | **~0.7–6%** | **最快之一** |

👉 在多个 benchmark 上达到 **SOTA 性能**，尤其在 Hurink 和 Dauzere 数据集上领先明显。

---

### 🔬 消融实验结果（Ablation Study, Table 4）

| 模型变体 | FJSP20x10 Gap | Time (s) | 说明 |
|---------|---------------|----------|------|
| DAN | 4.72% | 0.63 | 当前 SOTA 图注意力模型 |
| DME (only Mamba) | 2.80% | 0.30 | 单独编码器已显著提升 |
| CA (only Cross-Attn) | 2.67% | 0.31 | 注意力有效但泛化差 |
| **M-CA (Full)** | **2.66%** | **0.38** | 编码器+解码器协同最佳 |
| M2-CA / M3-CA | >2.66% | >0.43 | 多层 Mamba 无增益，反而变慢 |

📌 **结论**：
- 单层 Mamba + Cross-Attention 是最优平衡
- 多层堆叠不带来收益，反而增加延迟
- Cross-Attention 显著增强 operation-machine 交互能力

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Mamba 首次成功应用于离散组合优化调度问题**  
   - 打破了以往依赖 GNN/GAT 的范式，证明 SSM 在 FJSP 中的有效性和高效性。

2. **序列建模优于局部图消息传递**  
   - Mamba 能捕捉跨操作与机器的全局依赖，避免过平滑（over-smoothing）问题。

3. **兼具高性能与高效率**  
   - 推理速度快于 DAN 约 **2×**，内存占用更低，适合实时调度部署。

4. **极强的泛化能力**  
   - 在训练规模（20×10）外推至 **10000×10（500倍放大）** 仍保持优异性能，远超 OR-Tools 和其他学习方法。

5. **超越专用 JSSP 模型**  
   - 在 JSSP 基准（Taillard, DMU）上也优于专为此设计的 L2D 模型，体现跨问题迁移潜力。

---

### ⚠️ 局限性

1. **仍为自回归生成**  
   - 解码过程是逐步决策，无法完全并行化，影响绝对速度上限。

2. **环境交互开销主导总时间**  
   - 如 Table 6 所示，“Other” 时间（环境更新、状态转移）占比较大，模型推理本身占比小。

3. **未探索非贪婪搜索以外的搜索策略**  
   - 如 Beam Search 或 Monte Carlo Tree Search 可能进一步提升性能。

4. **对非常稀疏或高度约束实例适应性未知**  
   - 实验集中在标准随机生成数据，实际工厂中可能存在特殊工艺链路。

---

### 🔮 未来工作方向

1. **优化环境模拟器**  
   - 减少 autoregressive 交互延迟，提升端到端吞吐量。

2. **探索 Mamba + Transformer 混合架构**  
   - 在关键阶段引入少量 attention 提升精度。

3. **跨问题学习（Cross-Problem Learning）**  
   - 统一建模 JSSP、FJSP、Flow Shop 等多种调度问题。

4. **在线自适应调度**  
   - 结合动态重调度机制，应对机器故障、紧急插单等现实扰动。

5. **部署到真实产线系统**  
   - 与 MES/SCADA 系统集成，实现闭环智能调度。

---

> 💡 **一句话总结**：  
> 本论文开创性地将 **Mamba** 引入 **FJSP** 调度领域，提出 **Mamba-CrossAttention** 架构，实现了**高效、高质量、强泛化**的端到端求解，在多项 benchmark 上超越 SOTA 方法，并首次展示了 SSM 模型在制造调度中的巨大潜力。

</details>

---

### 13. [Confidence-Driven Multi-Scale Model Selection for Cost-Efficient Inference](https://arxiv.org/abs/2602.22090)

**Authors**: Bo-Wei Chen, Chung-Chi Chen, An-Zi Yen  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.22090v1  

#### Abstract
Large Language Models (LLMs) have revolutionized inference across diverse natural language tasks, with larger models performing better but at higher computational costs. We propose a confidence-driven strategy that dynamically selects the most suitable model based on confidence estimates. By assessi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Confidence-Driven Multi-Scale Model Selection for Cost-Efficient Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）在自然语言任务中表现出色，但其推理成本高昂，尤其在资源受限的场景（如边缘设备、商业API调用）下难以高效部署。传统做法是始终使用最大模型，造成不必要的计算开销。

本文旨在**平衡LLM推理的准确性与计算成本**，提出一种动态选择合适规模模型的方法，在保证性能的同时显著降低计算资源消耗。

---

### **提出的新方法与新思路**
作者提出了一个**基于置信度驱动的多尺度模型选择框架（confidence-driven multi-scale model selection）**，其核心思想是：

- **从小模型开始推理**，仅当小模型对自身回答缺乏信心时，才将任务“升级”（escalate）到更大的模型。
- 利用两种置信度信号进行决策：
  - **P(T)**（Probability of Truth）：衡量模型对其生成答案的**输出标记概率**（token probability），反映其对答案正确性的信心。
  - **P(IK)**（Probability of "I Know"）：通过训练一个基于LLM隐藏状态的分类器，预测该模型是否“知道”如何正确回答当前问题。

该策略实现了**动态路由（dynamic routing）**：简单任务由小模型处理，复杂或不确定任务交由大模型解决。

---

### **相比现有方法的优势**
| 维度 | 本方法优势 |
|------|-----------|
| **无需额外采样** | 不依赖top-k采样或多轮生成等高成本黑盒方法（如self-consistency），节省推理时间。 |
| **可解释性强** | 使用P(T)和P(IK)作为明确的不确定性信号，决策过程更透明。 |
| **适用于API场景** | 即使无法访问内部隐藏状态（如GPT-4o），仍可通过logits获取P(T)，具备实用性。 |
| **通用性好** | 在多个模型族（LLaMA、Qwen）、任务类型（选择题、开放问答）和分布内外（ID/OOD）数据上均有效。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 类型 | 描述 |
|-------|------|------|
| **MMLU** | 多项选择题 | Massive Multitask Language Understanding，涵盖57个学科领域的常识与专业知识测试。 |
| **GPQA** | Out-of-Distribution (OOD) | 高难度研究生级别问答数据集，用于检验方法在分布外场景下的泛化能力。 |
| **PopQA** | 开放式问答 | 大规模实体中心QA数据集，用于验证方法在生成式任务中的适用性。 |

---

### **实验设置**
- **模型组合**：
  - 开源模型：`LLaMA-3`系列（3B, 8B, 70B）、`Qwen3`系列（4B, 8B, 32B）
  - 商业API：`GPT-4o`
- **级联结构示例**：
  - `3B → 8B → 70B`
  - `8B → 70B`
  - `70B → GPT-4o`

- **P(T)计算方式**：
  - 对于多项选择题，计算模型输出选项A/B/C/D时第一个token的概率。
  - 设定阈值为 **0.9**：若P(T) ≥ 0.9，则保留答案；否则升级至更大模型。

- **P(IK)训练方式**：
  - 使用LLM第24层Transformer的hidden states作为输入。
  - 训练一个MLP分类器，监督信号为“是否答对”。

---

### **评估指标**
| 指标 | 含义 |
|------|------|
| **Accuracy (Acc.)** | 正确率，主性能指标 |
| **Reduced CC** | 计算成本减少比例（以GFLOPs衡量） |
| **End-to-end Time** | 推理总耗时（↓表示更低） |
| **Token Usage / Cost (USD)** | API场景下输入/输出token数量及费用 |
| **Hallucination Rate** | 幻觉率，通过grounding API验证事实准确性 |
| **Statistical Significance** | McNemar’s test判断性能差异是否显著 |

---

### **基线方法对比**
- **单一模型基准**：直接使用最大模型（如70B或GPT-4o）作为黄金标准。
- **级联模型变体**：
  - `3B→8B`, `8B→70B`, `3B→8B→70B` 等不同起点与路径。
- **消融实验**：比较是否使用P(IK)的影响。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### ✅ **在MMLU上的表现（LLaMA系列）**

| 配置 | 准确率 | 相较70B下降(PD) | 成本降低(Reduced CC) |
|------|--------|------------------|------------------------|
| 70B（单独） | 83.57% | — | — |
| 8B→70B | 83.22% | -0.35% | **36.46%** |
| 3B→8B→70B | 81.54% | -2.03% | **44.48%** |

> 🔹 **McNemar检验显示 `8B→70B` vs `70B` 无统计显著差异 (p=0.4048)**  
> 🔹 表明性能几乎持平，但节省近**三分之一计算量**

#### ✅ **在Qwen3系列上的结果**

| 配置 | 准确率 | 成本降低 |
|------|--------|----------|
| 32B（单独） | 79.51% | — |
| 8B→32B | **80.00%** | 33.18% |

> 🎯 **准确率反而略高于单一大模型！说明小模型过滤掉困难样本后，大模型专注处理难例，效果更好**

#### ✅ **在GPT-4o API上的应用（以token用量代理成本）**

| 配置 | 准确率 | 输出tokens | 成本降低 |
|------|--------|------------|-----------|
| GPT-4o（单独） | 86.43% | 36.225 | — |
| 70B→GPT-4o | **86.85%** | **14.505** | **~60%** |

> 💡 **不仅提升准确率，还节省约60% token使用量，极具商业价值**

#### ✅ **在PopQA（开放式QA）上的结果**

| 配置 | 准确率 | 幻觉率 | 成本降低 |
|------|--------|--------|-----------|
| 70B（单独） | 0.6585 | 0.3208 | — |
| 8B→70B | 0.6459 | 0.3422 | **7.11%** |

> ⚖️ 虽有轻微性能下降（-1.91%），但成本节约明显，且幻觉率控制良好。

---

### **消融实验结果（Ablation Study）**

#### 🔍 是否使用P(IK)的影响（8B→70B配置）

| 设置 | 性能下降(PD) | 成本降低 |
|------|--------------|----------|
| **w/ P(IK)** | -0.35% | 36.46% |
| **w/o P(IK)** | -1.26% | 36.83% |

> ❗ 缺少P(IK)时性能下降更严重，说明**P(IK)有助于稳定性能**，避免错误路由导致精度崩塌。

#### 🔍 P(T)阈值敏感性分析

| P(T)阈值 | 准确率 | 成本降低 |
|---------|--------|----------|
| ≥95% | 83.50% | 33.25% |
| ≥50% | 82.66% | 39.33% |

> 🔻 阈值越低，越多任务被小模型处理 → 成本降更多，但准确率略有牺牲。  
> ✅ **P(IK)+P(T)联合机制可在低阈值下保持稳定性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **置信度信号（P(T)和P(IK)）可有效指导模型选择**，实现高性价比推理。
2. ✅ **级联架构能在几乎不损失准确率的前提下，减少20%-40%的计算成本**（开源模型）。
3. ✅ **应用于GPT-4o API时，可节省约60% token用量，同时维持甚至略微超越原性能**。
4. ✅ 方法具有良好的**跨模型家族泛化能力**（LLaMA → Qwen）和**任务扩展性**（从选择题到生成任务）。
5. ✅ 在OOD数据（GPQA）上虽收益较小（仅节省~5% CC），但仍能保持竞争力，证明一定鲁棒性。

---

### **局限性（Limitations）**
1. **P(IK)依赖训练数据**：需在特定数据集上训练分类器，跨领域迁移可能退化。
2. **对极端OOD任务效果有限**：如GPQA中，小模型几乎无法处理任何问题，导致升级频繁，节省不多。
3. **未考虑延迟影响**：多跳路由会增加端到端延迟，尤其在网络API场景下可能抵消部分效率优势。
4. **主要验证于NLU任务**：在对话系统、长文本生成等复杂NLP任务中的适用性尚待探索。

---

### **未来工作方向**
1. **改进P(IK)的泛化能力**：研究无监督或领域自适应方法来估计“I Know”概率。
2. **拓展至生成式任务**：设计更适合自由生成的置信度度量（如first-token confidence）。
3. **引入延迟感知路由**：结合网络延迟、硬件负载等因素优化路由决策。
4. **构建统一的Confidence Router模块**：使其成为可插拔组件，适配各类LLM服务架构。

---

> 📌 **总体评价**：  
> 本文提出的**confidence-driven model selection**是一种实用、高效且理论清晰的LLM推理优化范式，特别适合**商业化部署与边缘计算场景**。它为“何时该用大模型”提供了量化依据，推动LLM走向更可持续、更具成本效益的应用模式。

</details>

---

### 14. [DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference](https://arxiv.org/abs/2602.21548)

**Authors**: Yongtong Wu, Shaoyuan Chen, Yinmin Zhong, Rilin Huang, Yixuan Tan, Wentao Zhang, Liyue Zhang, Shangyan Zhou, Yuxuan Liu, Shunfeng Zhou, Mingxing Zhang, Xin Jin, Panpan Huang  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.21548v1  

#### Abstract
The performance of multi-turn, agentic LLM inference is increasingly dominated by KV-Cache storage I/O rather than computation. In prevalent disaggregated architectures, loading the massive KV-Cache from external storage creates a fundamental imbalance: storage NICs on prefill engines become bandwid...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**多轮交互式（agentic）LLM 推理**场景中，系统性能日益受限于 **KV-Cache 存储 I/O**，而非计算能力。现有的 **PD-disaggregated 架构**（Prefill-Decoding Disaggregation）存在严重的存储带宽利用不均问题：
- **Prefill 引擎**需要从外部存储加载大量 KV-Cache，导致其 **Storage NIC 带宽饱和**；
- 而 **Decoding 引擎**的 Storage NIC 却处于空闲状态。

这种不对称性严重限制了整体系统吞吐量。

---

### 🚀 提出的新方法与创新思路

#### **DualPath：双路径 KV-Cache 加载架构**
- **传统路径**：KV-Cache 仅由 Prefill 引擎直接从存储加载（Storage → PE）；
- **新增路径**：允许 KV-Cache 首先加载到 **Decoding 引擎**，再通过高性能 **RDMA over Compute Network** 传输至 Prefill 引擎（Storage → DE → PE）。

> 💡 **核心洞察**：KV-Cache 加载不必局限于 Prefill 引擎；可利用 Decode 引擎闲置的 Storage NIC 带宽，并借助高带宽 Compute Network 进行内部转发。

#### **配套机制设计**
1. **CNIC-Centric Traffic Manager**  
   所有数据传输（包括 H2D/D2H）均通过 **GPUDirect RDMA** 经由 Compute NIC 完成，实现与模型执行通信的 **流量隔离**（通过 InfiniBand Virtual Lane 或 RoCE DSCP/TC）。

2. **Adaptive Request Scheduler**  
   动态调度器根据 **Storage NIC 队列长度、GPU 负载、token 数量**等指标，决定每个请求走哪条路径，实现 **计算与网络资源的联合负载均衡**。

---

### 🔍 相比现有方法的优势
| 方面 | DualPath | 现有方法（如 SGLang + Mooncake） |
|------|----------|-------------------------------|
| **存储带宽利用率** | 利用所有引擎的 SNIC，聚合带宽 | 仅 Prefill 引擎使用 SNIC，Decode 引擎带宽浪费 |
| **KV-Cache 缓存层级** | 支持纯外置 SSD 存储，节省 DRAM | 依赖 DRAM Pool（如 Mooncake），成本高且容量有限 |
| **I/O 干扰控制** | 通过 CNIC 中心化调度，避免干扰模型通信 | 缺乏细粒度 QoS 控制，易造成延迟波动 |
| **适用场景** | 内存受限训练（如 RL Rollout）、长上下文在线服务 | 在 DRAM 不足或工作集过大时性能下降明显 |

---

## 2. 核心实验方法和设置

### 📊 数据集
使用来自生产环境的 **Agent Trace Datasets**，模拟真实 **强化学习（RL）Rollout 场景**中的多轮交互任务：
- 包含 500 条轨迹，最大上下文长度分别为 **32K、48K、64K tokens**；
- 每轮平均追加 **~429 tokens**，生成 **~176 tokens**；
- 平均交互轮数达 **157 轮**，体现典型的“长上下文、短追加”模式；
- KV-Cache 命中率高达 **≥98.7%**，凸显 I/O 密集特性。

> 📌 数据集统计见原文 Table 2。

---

### ⚙️ 实验设置

| 项目 | 设置详情 |
|------|---------|
| **硬件平台** | 多节点集群，每节点 8× NVIDIA Hopper GPU，配备 8×400Gbps RDMA CNIC 和 1×400Gbps SNIC；Compute 与 Storage 网络物理隔离 |
| **分布式存储** | 使用自研 **3FS**（基于 SSD），无 DRAM 缓存，支持饱和读取带宽 |
| **模型** | - **DS 660B**（MoE, DeepSeek Sparse Attention）<br>- **DS 27B**（Downscaled 版本）<br>- **Qwen2.5-32B**（Dense, GQA） |
| **并行策略** | - DS 系列：EP + DP<br>- Qwen32B：DP（DualPath），TP=8（SGLang） |
| **P/D Ratio** | 默认 2P4D（DS660B）、1P2D（Qwen32B）、1P1D（DS27B） |

---

### 📈 评估指标

| 场景 | 指标 |
|------|------|
| **离线推理（Offline Batch Inference）** | Job Completion Time (**JCT**) |
| **在线服务（Online Serving）** | - Time to First Token (**TTFT**) <br> - Time to Second Token (**TTST**) <br> - Time Per Output Token (**TPOT**) <br> - 最大支持 Agent Arrival Rate (**APS**) |
| **SLO** | TTFT ≤ 4s, TPOT ≤ 50ms |

---

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **Basic** | 自研基础框架，未启用 DualPath，作为主要对比基准 |
| **SGL(MC)** | SGLang + HiCache + Mooncake Store + Mooncake Transfer Engine，代表先进开源方案 |
| **Oracle** | 理想化上限：跳过所有磁盘读取与 KV-Cache 传输开销，用于衡量 I/O 消除程度 |

---

## 3. 主要实验结果和性能指标

### 📉 离线推理性能（Offline Inference）
- **最高提升达 1.87× 吞吐量**（即 JCT 缩短为原来的 53.5%）；
- 在 **64K 上下文、1024 agents** 下，DualPath 接近 Oracle 性能，表明 **KV-Cache I/O 几乎被完全消除**；
- 随着批大小和上下文增长，增益更显著，说明方法对大规模负载更具优势。

> ✅ 图 7 展示不同模型与配置下的 JCT 对比。

---

### 📈 在线服务能力（Online Serving）
- **平均吞吐提升 1.96×**（APS 提升 1.67× ~ 2.25×）；
- 在满足 SLO 前提下，**TTFT、TTST、TPOT 均优于或持平基线**；
- Basic 因存储瓶颈导致排队延迟激增，而 DualPath 维持稳定低延迟（图 12 左）。

> ✅ 图 10 展示随 APS 增加的延迟表现。

---

### 🔬 消融实验（Ablation Study）
在 DS660B、64K 上下文、1024 agents 设置下逐步添加组件：

| 组件 | JCT 相对 Basic 的降低幅度 |
|------|------------------------|
| Layerwise Prefill | -17.21% |
| + Dual-Path Loading | -38.19% |
| + Adaptive Scheduling | **-45.62%** |

> ✅ 表明 **Dual-Path 是主要性能来源**，调度算法进一步优化负载均衡。

---

### 📊 负载均衡效果
- **Storage NIC 流量负载比**（Max/Avg）从 1.53（Round Robin）降至 **1.18**；
- **Attention 层执行时间 Max/Avg 比**维持在 **1.06** 以内，减少 GPU 等待气泡；
- 图 13–14 显示 DualPath 显著改善了系统级资源平衡。

---

### 🌐 大规模扩展性（Up to 1,152 GPUs）
| 场景 | 结果 |
|------|------|
| **48P96D, 48K agents** | JCT 为 3,201s，相比 2P4D 的 3,167s 实现近线性加速 |
| **44P88D, 8.8 APS** | 较 0.4 APS 提升 22× 吞吐，延迟保持稳定 |
| **Scheduler 开销** | CPU 使用 <10 core，非瓶颈 |

> ✅ 表明 DualPath 可扩展至生产级超大规模部署。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Agentic LLM 推理本质上是 I/O-bound**，KV-Cache 加载速度成为性能瓶颈；
2. 现有架构中 **Prefill 侧 SNIC 成为单点瓶颈**，而 Decode 侧带宽严重浪费；
3. **Dual-Path 架构有效聚合全集群 SNIC 带宽**，打破原有瓶颈；
4. **CNIC-Centric 数据路径 + QoS 隔离** 可安全复用 Compute Network，不影响模型通信；
5. **动态调度器** 是实现端到端性能提升的关键，需同时考虑 NIC 与 GPU 负载。

---

### ⚠️ 方法的局限性
1. **依赖高性能 RDMA Compute Network**：若 Compute NIC 带宽不足或拥塞，则无法支撑额外 KV-Cache 转发流量；
2. **引入额外 H2D/D2H 开销**：虽然通过 DE Buffer 减少 HBM 占用，但仍增加一次主机内存拷贝；
3. **当前调度未支持请求拆分读取**：一个请求只能选择单一路径读取，未来可探索跨路径并行加载；
4. **小模型场景下 P-D 传输开销仍显著**：如 DS27B 中 TPOT 较高，提示轻量级模型需进一步优化传输协议。

---

### 🔮 未来工作方向（Future Work）
1. **动态调整 P/D Ratio 与并行策略**：针对不同 workload 自动调优资源配置；
2. **更精细的调度策略**：例如将请求按 KV-Cache 分块，分别从 PE 和 DE 并行加载；
3. **结合中间缓存层（DRAM Pool）**：虽文中指出收益边际，但在特定场景仍有潜力；
4. **支持异构硬件环境下的路径选择**：适应不同节点带宽、延迟差异；
5. **面向突发流量的弹性调度机制**：提升在线服务应对 burst 请求的能力。

---

## 总结

**DualPath** 是首个提出 **双路径 KV-Cache 加载** 的 LLM 推理系统，从根本上重新思考了 PD-disaggregated 架构中的 I/O 分布问题。它通过：
- 利用 Decode 引擎的闲置 SNIC 带宽，
- 借助 RDMA Compute Network 进行高效转发，
- 配合 CNIC 中心化流量管理与动态调度，

成功将存储 I/O 从“单点瓶颈”转变为“全局可调度资源”，在真实 agentic 工作负载下实现了 **最高 1.87× 离线吞吐** 和 **平均 1.96× 在线吞吐提升**，为下一代高并发、长上下文 LLM 应用提供了重要基础设施支持。

</details>

---

### 15. [LLMTailor: A Layer-wise Tailoring Tool for Efficient Checkpointing of Large Language Models](https://arxiv.org/abs/2602.22158)

**Authors**: Minqiu Sun, Xin Huang, Luanzheng Guo, Nathan R. Tallent, Kento Sato, Dong Dai  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.22158v1  

#### Abstract
Checkpointing is essential for fault tolerance in training large language models (LLMs). However, existing methods, regardless of their I/O strategies, periodically store the entire model and optimizer states, incurring substantial storage overhead and resource contention. Recent studies reveal that...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《LLMTailor: A Layer-wise Tailoring Tool for Efficient Checkpointing of Large Language Models》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模 **Large Language Model (LLM)** 的训练过程中，**checkpointing** 是保障容错性的关键机制。然而，传统方法每次都会保存完整的模型权重和优化器状态（optimizer states），导致极高的 **I/O 开销** 和存储压力。已有研究表明，LLM 各层的参数更新具有高度非均匀性——某些层变化剧烈，而其他层则相对稳定甚至不变。

因此，论文指出：**统一地保存所有层的状态是低效的**。现有工具（如 MergeKit）虽支持模型合并，但仅处理 `weights`，忽略 `optimizer states`、辅助层（如 `embed_tokens`, `lm_head`）和配置文件，无法用于恢复训练。

### 提出的新方法与创新思路
作者提出 **LLMTailor**，一个支持细粒度、可恢复训练的 **layer-wise checkpointing** 工具，其核心思想是：

> **从多个部分检查点中筛选并重组关键层，构建一个“弗兰肯斯坦”式（Frankenstein）的完整 checkpoint，以实现高效且可恢复的训练重启。**

具体创新包括：
- ✅ 支持对 **model weights** 和 **optimizer states** 的联合操作；
- ✅ 引入 **parameter group 重构机制**，将原本扁平化的 optimizer 文件按 transformer 层进行逻辑分组（每层拆分为 weight-decay/non-weight-decay 两组），使其具备可分割性；
- ✅ 支持 **auxiliary layers**（如 `embed_tokens`, `lm_head`）的显式切分与合并；
- ✅ 自动处理 **configuration files** 和元数据，确保训练连续性；
- ✅ 延续 MergeKit 的 YAML 配置风格，提供易用接口。

### 相比现有方法的优势
| 方面 | 现有方法（如 MergeKit） | LLMTailor |
|------|------------------------|----------|
| 是否保留 optimizer states | ❌ 不支持 | ✅ 支持 |
| 是否支持训练恢复 | ❌ 仅适用于推理 | ✅ 完整训练可恢复 |
| 是否处理辅助层 | ❌ 忽略 `embed_tokens`/`lm_head` | ✅ 显式支持 |
| 是否支持分布式 ZeRO 检查点 | ❌ 无考虑 | ✅ 兼容 DeepSpeed ZeRO-3 |
| 是否可用于 checkpoint 压缩 | ❌ 不适用 | ✅ 可实现选择性保存 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **PubMed-Summarization**：纯文本语料，用于 **Continual Pre-Training (CPT)** 任务。
- **MedQA**：结构化医学问答数据集，用于 **Supervised Fine-Tuning (SFT)** 任务。

这两个数据集均聚焦于 **医疗领域微调**，用于验证模型在特定下游任务中的表现稳定性。

### 实验设置
- **硬件环境**：8×NVIDIA A100 80GB GPU，AMD EPYC 7713 CPU，Lustre 文件系统通过 InfiniBand 连接。
- **软件栈**：CUDA 12.8, PyTorch 2.7.1, DeepSpeed v0.17.2, AdamW 作为默认 optimizer。
- **模型规模**：
  - Llama-3.2-1B
  - Llama-3.1-8B
  - Qwen-2.5-7B
- **序列长度**：2048
- **batch 设置**：
  - CPT：micro-batch=4, gradient accumulation=2
  - SFT：micro-batch=2, gradient accumulation=2

### 评估指标
| 类别 | 指标 |
|------|------|
| **效率指标** | - Checkpoint 大小（GB）<br>- Checkpoint 时间占比（占端到端训练时间的比例）<br>- 合并耗时（loading & merging time） |
| **质量指标** | - 最终 train/eval loss<br>- 在五个 benchmark 上的 zero-shot 性能：<br> • MMLU（通识）<br> • MMLU_med（医学通识）<br> • MedMCQA<br> • MedQA<br> • PubMedQA |
| **正确性验证** | - 恢复后 loss 曲线是否与原轨迹一致 |

### 基线方法对比
- **Baseline**：标准全量 checkpoint（transformers 库默认策略）
  - 固定间隔：SFT 每 50 步，CPT 每 100 步
- **LLMTailor 策略**：
  1. **Parity Checkpointing**：交替保存奇偶层（一半层来自旧 checkpoint，另一半来自新 checkpoint）
  2. **Filtering Checkpointing**：只频繁保存首尾若干层（前几层 + 最后两层），中间层稀疏保存（每 5× 原始频率）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 🔹 Use Case 1: Parity Merging（交替合并）
| 模型 | 方法 | Checkpoint Size (GB) | 存储减少 | Checkpoint Time Ratio (%) |
|------|------|------------------------|---------|----------------------------|
| Llama3.1-8B | Full | 1799.52 | — | 4.99 |
| Llama3.1-8B | Parity | 899.76 | ~50% | 3.03 (-39.3%) |
| Qwen2.5-7B | Full | 1811.52 | — | 20.63 |
| Qwen2.5-7B | Parity | 905.76 | ~50% | 12.76 (-38.1%) |

✅ **结论**：存储开销降低约 **50%**，checkpoint 时间减少近 **40%**。

#### 🔹 Use Case 2: Filtering-Based Merging（基于过滤的合并）
| 模型 | 方法 | Checkpoint Size (GB) | 存储压缩比 | Time Ratio (%) | 时间加速比 |
|------|------|------------------------|------------|----------------|-------------|
| Llama3.1-8B | Full | 1799.52 | — | 4.99 | — |
| Llama3.1-8B | Filtered | 420 | **4.3× 更小** | 1.66 | **3.0× 更快** |
| Qwen2.5-7B | Full | 1811.52 | — | 20.63 | — |
| Qwen2.5-7B | Filtered | 434.56 | **4.2× 更小** | 7.26 | **2.8× 更快** |

✅ **结论**：采用智能过滤策略，可实现高达 **4.3倍的存储压缩** 和 **2.8倍的时间加速**。

---

### 模型质量评估结果

#### 表 1：Loss 对比（越低越好）

| 模型 | 场景 | Final Train Loss | Final Eval Loss |
|------|------|------------------|-----------------|
| Qwen2.5-7B | 原始 SFT | 1.58 | 1.60 |
| Qwen2.5-7B | Filtered (start 400) | 1.60 | 1.62 |
| Llama3.1-8B | 原始 CPT | 1.58 | 1.58 |
| Llama3.1-8B | Filtered (start 1000) | 1.59 | 1.59 |

➡️ Loss 仅有轻微上升，说明模型恢复基本稳定。

#### 表 2：Zero-Shot Benchmark 结果（越高越好）

| 任务 | 模型 | MMLU | MMLU_med | MedMCQA | MedQA | PubMedQA |
|------|------|------|-----------|---------|-------|----------|
| SFT | Qwen2.5-7B (original) | 73.14 | 89.00 | 60.75 | 64.02 | 75.20 |
| SFT | Qwen2.5-7B (filtered) | 71.64 | 84.00 | 59.50 | 62.06 | 75.60 |
| CPT | Llama3.1-8B (original) | 60.00 | 75.00 | 53.10 | 55.15 | 77.20 |
| CPT | Llama3.1-8B (filtered) | 62.06 | 77.00 | 53.45 | 54.91 | 78.00 |

➡️ 尽管部分指标略有下降（如 SFT 下降约 1–5 pts），但在 CPT 中反而有所提升，表明 LLM 具备较强鲁棒性，部分 checkpointing 可被容忍。

---

### 消融实验与额外分析
- **LLMTailor 合并开销测试**（Table 7）显示：
  - 合并两个 checkpoint 平均耗时约 **332.4 秒（Llama3-8B）**
  - “parity” 模式因多次加载同一 checkpoint 导致开销更高（~1027.5 秒）
  - 但相比数小时的训练周期，该开销可接受。
- **层数越多，合并越快？**  
  当每个 checkpoint 只含单一层时，加载更快，提示未来若原生支持 layer-wise checkpointing，可进一步优化性能。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 各层更新不均衡**，为 selective checkpointing 提供理论基础；
2. ✅ **LLMTailor 能有效构建可恢复的“混合 checkpoint”**，支持灵活组合不同时间点的层；
3. ✅ 在合理策略下（如 parity 或 filtering），**可将 checkpoint 大小减少 50%~77%**，时间开销降低 2.8~3.0×；
4. ✅ 模型恢复后的性能损失极小，在多数 benchmark 上保持相当甚至反超，证明方法的可行性；
5. ✅ 该框架兼容 DeepSpeed 分布式训练与 ZeRO-3，具备实际部署潜力。

### 方法的局限性
- ⚠️ **当前为后处理工具**：LLMTailor 目前是一个离线合并工具，需先生成多个 partial checkpoints 再合并，尚未集成进训练流程实现实时 selective saving；
- ⚠️ **optimizer 加载无法懒加载**：必须完整读取整个 optimizer shard 才能提取某一层，带来较高 I/O 成本；
- ⚠️ **依赖 YAML 手动配置**：缺乏自动化决策模块来判断“哪些层值得保存”，仍需人工设计策略；
- ⚠️ **暂未支持动态调整策略**：例如根据梯度变化自动决定保存频率。

### 未来工作方向
- 🔄 将 LLMTailor 集成进训练框架（如 DeepSpeed），实现 **runtime selective checkpointing**；
- 🤖 设计 **dynamic layer selection policy**，基于梯度幅值、注意力分布等信号自动识别重要层；
- 💾 探索 **native layer-wise checkpoint format**，避免重复加载 optimizer 文件；
- 🔗 结合其他 I/O 优化技术（如 compression, async I/O, in-memory checkpointing）形成综合解决方案；
- 🧩 支持更多模型架构（如 MoE、Diffusion Models）和 optimizer 类型。

---

> **总结一句话**：  
> **LLMTailor 首次实现了同时对 LLM 的 weights 与 optimizer states 进行 layer-wise 操作的能力，开启了高效、细粒度 checkpointing 的新路径，在几乎不影响模型质量的前提下显著降低了存储与 I/O 开销。**

</details>

---

### 16. [Neural network optimization strategies and the topography of the loss landscape](https://arxiv.org/abs/2602.21276)

**Authors**: Jianneng Yu, Alexandre V. Morozov  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.21276v1  

#### Abstract
Neural networks are trained by optimizing multi-dimensional sets of fitting parameters on non-convex loss landscapes. Low-loss regions of the landscapes correspond to the parameter sets that perform well on the training data. A key issue in machine learning is the performance of trained neural netwo...

---

### 17. [SymTorch: A Framework for Symbolic Distillation of Deep Neural Networks](https://arxiv.org/abs/2602.21307)

**Authors**: Elizabeth S. Z. Tan, Adil Soubki, Miles Cranmer  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.21307v1  

#### Abstract
Symbolic distillation replaces neural networks, or components thereof, with interpretable, closed-form mathematical expressions. This approach has shown promise in discovering physical laws and mathematical relationships directly from trained deep learning models, yet adoption remains limited due to...

---

### 18. [D-COT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in Small Language Models](https://arxiv.org/abs/2602.21786)

**Authors**: Shunsuke Ubukata  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.21786v1  

#### Abstract
Chain-of-Thought (CoT) distillation from Large Language Models (LLMs) often induces "overthinking" in Small Language Models (SLMs), leading to performance degradation and excessive token consumption. In this study, we propose Disciplined Chain-of-Thought (D-CoT), a novel framework that enforces a st...

---

### 19. [PASTA: A Modular Program Analysis Tool Framework for Accelerators](https://arxiv.org/abs/2602.22103)

**Authors**: Mao Lin, Hyeran Jeon, Keren Zhou  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.22103v1  

#### Abstract
The increasing complexity and diversity of hardware accelerators in modern computing systems demand flexible, low-overhead program analysis tools. We present PASTA, a low-overhead and modular Program AnalysiS Tool Framework for Accelerators. PASTA abstracts over low-level profiling APIs and diverse ...

---

### 20. [Training-free Composition of Pre-trained GFlowNets for Multi-Objective Generation](https://arxiv.org/abs/2602.21565)

**Authors**: Seokwon Yoon, Youngbin Choi, Seunghyuk Cho, Seungbeom Lee, MoonJeong Park, Dongwoo Kim  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.21565v1  

#### Abstract
Generative Flow Networks (GFlowNets) learn to sample diverse candidates in proportion to a reward function, making them well-suited for scientific discovery, where exploring multiple promising solutions is crucial. Further extending GFlowNets to multi-objective settings has attracted growing interes...

---

### 21. [VecGlypher: Unified Vector Glyph Generation with Language Models](https://arxiv.org/abs/2602.21461)

**Authors**: Xiaoke Huang, Bhavul Gauri, Kam Woh Ng, Tony Ng, Mengmeng Xu, Zhiheng Liu, Weiming Ren, Zhaochong An, Zijian Zhou, Haonan Qiu, Yuyin Zhou, Sen He, Ziheng Wang, Tao Xiang, Xiao Han  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.21461v1  

#### Abstract
Vector glyphs are the atomic units of digital typography, yet most learning-based pipelines still depend on carefully curated exemplar sheets and raster-to-vector postprocessing, which limits accessibility and editability. We introduce VecGlypher, a single multimodal language model that generates hi...

---

### 22. [RuCL: Stratified Rubric-Based Curriculum Learning for Multimodal Large Language Model Reasoning](https://arxiv.org/abs/2602.21628)

**Authors**: Yukun Chen, Jiaming Li, Longze Chen, Ze Gong, Jingpeng Li, Zhen Qin, Hengyu Chang, Ancheng Xu, Zhihao Yang, Hamid Alinejad-Rokny, Qiang Qu, Bo Zheng, Min Yang  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.21628v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a prevailing paradigm for enhancing reasoning in Multimodal Large Language Models (MLLMs). However, relying solely on outcome supervision risks reward hacking, where models learn spurious reasoning patterns to satisfy final answer ...

---

### 23. [MERRY: Semantically Decoupled Evaluation of Multimodal Emotional and Role Consistencies of Role-Playing Agents](https://arxiv.org/abs/2602.21941)

**Authors**: Zhenyu Wang, Xiaofen Xing, Yirong Chen, Xiangmin Xu  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.21941v1  

#### Abstract
Multimodal Role-Playing Agents (MRPAs) are attracting increasing attention due to their ability to deliver more immersive multimodal emotional interactions. However, existing studies still rely on pure textual benchmarks to evaluate the text responses of MRPAs, while delegating the assessment of the...

---

### 24. [TiMi: Empower Time Series Transformers with Multimodal Mixture of Experts](https://arxiv.org/abs/2602.21693)

**Authors**: Jiafeng Lin, Yuxuan Wang, Huakun Luo, Zhongyi Pei, Jianmin Wang  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.21693v1  

#### Abstract
Multimodal time series forecasting has garnered significant attention for its potential to provide more accurate predictions than traditional single-modality models by leveraging rich information inherent in other modalities. However, due to fundamental challenges in modality alignment, existing met...

---

### 25. [From Words to Amino Acids: Does the Curse of Depth Persist?](https://arxiv.org/abs/2602.21750)

**Authors**: Aleena Siji, Amir Mohammad Karimi Mamaghan, Ferdinand Kapl, Tobias H\"oppe, Emmanouil Angelis, Andrea Dittadi, Maurice Brenner, Michael Heinzinger, Karl Henrik Johansson, Kaitlin Maile, Johannes von Oswald, Stefan Bauer  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.21750v1  

#### Abstract
Protein language models (PLMs) have become widely adopted as general-purpose models, demonstrating strong performance in protein engineering and de novo design. Like large language models (LLMs), they are typically trained as deep transformers with next-token or masked-token prediction objectives on...

---

### 26. [Robustness in sparse artificial neural networks trained with adaptive topology](https://arxiv.org/abs/2602.21961)

**Authors**: Bendeg\'uz Sulyok, Gergely Palla, Filippo Radicchi, Santo Fortunato  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.21961v1  

#### Abstract
We investigate the robustness of sparse artificial neural networks trained with adaptive topology. We focus on a simple yet effective architecture consisting of three sparse layers with 99% sparsity followed by a dense layer, applied to image classification tasks such as MNIST and Fashion MNIST. By ...

---

### 27. [Disaster Question Answering with LoRA Efficiency and Accurate End Position](https://arxiv.org/abs/2602.21212)

**Authors**: Takato Yasuno  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.21212v1  

#### Abstract
Natural disasters such as earthquakes, torrential rainfall, floods, and volcanic eruptions occur with extremely low frequency and affect limited geographic areas. When individuals face disaster situations, they often experience confusion and lack the domain-specific knowledge and experience necessar...

---

### 28. [Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](https://arxiv.org/abs/2602.21646)

**Authors**: Yexing Du, Youcheng Pan, Zekun Wang, Zheng Chu, Yichong Huang, Kaiyuan Liu, Bo Yang, Yang Xiang, Ming Liu, Bing Qin  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.21646v1  

#### Abstract
Multimodal Large Language Models (MLLMs) have achieved notable success in enhancing translation performance by integrating multimodal information. However, existing research primarily focuses on image-guided methods, whose applicability is constrained by the scarcity of multilingual image-text pairs...

---

### 29. [Improving Implicit Discourse Relation Recognition with Natural Language Explanations from LLMs](https://arxiv.org/abs/2602.21763)

**Authors**: Heng Wang, Changxing Wu  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.21763v1  

#### Abstract
Implicit Discourse Relation Recognition (IDRR) remains a challenging task due to the requirement for deep semantic understanding in the absence of explicit discourse markers. A further limitation is that existing methods only predict relations without providing any supporting explanations. Recent ad...

---

### 30. [ExpLang: Improved Exploration and Exploitation in LLM Reasoning with On-Policy Thinking Language Selection](https://arxiv.org/abs/2602.21887)

**Authors**: Changjiang Gao, Zixian Huang, Kaichen Yang, Jiajun Chen, Jixing Li, Shujian Huang  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.21887v1  

#### Abstract
Current large reasoning models (LRMs) have shown strong ability on challenging tasks after reinforcement learning (RL) based post-training. However, previous work mainly focuses on English reasoning in expectation of the strongest performance, despite the demonstrated potential advantage of multilin...

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

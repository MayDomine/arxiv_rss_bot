# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-11 08:51:40 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Sparse Attention as a Range Searching Problem: Towards an Inference-Efficient Index for KV Cache](https://arxiv.org/abs/2605.06763)

**Authors**: Mohsen Dehghankar, Abolfazl Asudeh  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2605.06763v1  

#### Abstract
Sparse attention improves LLM inference efficiency by selecting a subset of key-value entries, but at the cost of potential accuracy degradation. In particular, omitting critical KV entries can induce substantial errors in model outputs. Existing methods typically operate under fixed or adaptive tok...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sparse Attention as a Range Searching Problem: Towards an Inference-Efficient Index for KV Cache

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **sparse attention** 方法在加速 LLM 推理时存在两个关键缺陷：
- **False Negatives（假阴性）**：可能遗漏对当前查询至关重要的 KV 缓存条目，导致输出质量显著下降。
- **固定预算限制**：大多数方法采用固定的 `k` 或比例预算（如 top-k），无法适应不同解码步骤中“重要 token 数量”动态变化的需求。

这些缺陷在长输出推理任务（如数学推理、多跳问答）中尤为严重，因为模型在不同阶段需要关注不同数量的历史上下文。

### 🚀 提出的新方法与思路
作者提出将 **sparse attention 问题重新建模为 halfspace range searching 问题**，并基于此设计了一个名为 **Louver** 的新型 KV Cache 索引结构。

#### 核心思想：
- 将每个 key 视为 $ \mathbb{R}^d $ 中的一个点。
- 给定 query $ q $ 和阈值 $ T $，定义一个半空间（halfspace）区域：  
  $$
  H(q, T) = \{ p \in \mathbb{R}^d \mid \langle q, p \rangle \geq T \}
  $$
- 所有满足 $ \langle q, k_j \rangle \geq T $ 的 keys 都落在该区域内，目标是精确检索所有位于该区域内的 keys。

这种方法天然支持：
- 动态数量的检索结果（无需预设 k）
- **零假阴性保证（zero false negatives）**

### 🔍 Louver 的设计优势
Louver 是专为 LLM 推理优化的索引框架，具备以下特性：

| 特性 | 描述 |
|------|------|
| **Zero False Negatives** | 理论和实践中均保证不会遗漏任何满足阈值的关键 key。 |
| **轻量集成** | 可无缝嵌入现有 LLM 推理流水线（如 vLLM、HuggingFace）。 |
| **硬件感知优化** | 支持 CPU/GPU 并行执行，利用 AVX/F16 指令集和 CUDA 流并发更新。 |
| **动态更新机制** | 使用 buffer + 批量增量构建策略，在 decoding 过程中持续维护索引。 |

相比现有方法，Louver 不依赖近似搜索（ANN）、不强制归一化向量、也不受 query-key 分布偏移影响。

---

## 2. 核心实验方法和设置

### 📚 数据集
实验覆盖多种任务类型，验证通用性和鲁棒性：

| 类型 | 数据集 | 说明 |
|------|--------|------|
| **长上下文理解** | LongBench v1 | 多语言、多任务 QA，测试从长 prompt 中提取信息的能力 |
| | RULER | 合成基准，包含 Needle-in-a-Haystack (NIAH) 和 Variable Tracking (VT)，用于量化检索能力 |
| **长输出推理** | AIME 2024 | 数学竞赛题，需生成数千 token 的 chain-of-thought |
| | MATH-500 | 高难度数学问题集，评估复杂推理稳定性 |

### ⚙️ 实验设置
- **模型**：Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-Llama-8B, Qwen2.5 系列等开源模型。
- **上下文长度**：最高达 40k tokens。
- **精度格式**：BF16 / FP16。
- **硬件平台**：
  - GPU：NVIDIA A100 / RTX 5090
  - CPU：AMD EPYC / Ryzen Threadripper

### 🎯 评估指标
| 指标 | 用途 |
|------|------|
| **Accuracy / F1 Score** | 衡量任务完成准确率（LongBench, AIME, MATH） |
| **Per-step Latency** | 单步 attention 耗时，反映推理效率 |
| **Speedup** | 相比 dense attention（如 FlashAttention）的加速比 |
| **Recall@k** | 检索出真实 top-k keys 的比例，衡量召回能力 |
| **KV Retention Rate** | 保留的 KV 条目占比（如 10%） |

### 🆚 基线方法分类对比
| 类别 | 方法 | 说明 |
|------|------|------|
| **Dense Attention** | FlashAttention-2, Torch SDPA | 密集注意力基线，代表性能上限 |
| **Eviction-based** | H2O, StreamingLLM | 固定模式丢弃旧 token |
| **Fixed-budget Retrieval** | Quest, Twilight(p=0.85) | 固定数量或比例检索 |
| **ANN-based Offloading** | RetrievalAttention (HNSW), InfLLM (IVF), MagicPIG (LSH) | 将 KV 缓存卸载到 CPU，用 ANN 检索 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ 表格 1 & 2：LongBench 与 RULER 上的表现（10% KV 保留）
| 方法 | Avg F1 (%) | NIAH-S / VT 准确率 |
|------|-----------|------------------|
| FlashAttention-2 (Dense) | 41.7 | 100.0 / 84.0 |
| H2O | 40.6 | 100.0 / 52.0 |
| Quest | 41.1 | 100.0 / 60.0 |
| **Louver (Ours)** | **41.8** | **100.0 / 74.0** |

> 💡 Louver 在保持相同 KV 预算下，达到甚至超过 dense attention 的准确性。

#### ✅ 表格 3：长输出推理任务表现（AIME/MATH-500）
| 方法 | AIME Acc (%) | MATH-500 Acc (%) |
|------|-------------|------------------|
| FlashAttention-2 | 30.0 | 58.0 |
| Twilight (p=0.85) | 25.0 | 44.0 |
| **Louver** | **30.0** | **62.0** |

> 🔥 Louver 在 MATH-500 上超越 dense attention，证明其能更精准保留关键推理路径中的上下文。

#### ⏱ 图 6：单步延迟 vs 上下文长度（40k context）
| 对比项 | 加速比（GPU） | 加速比（CPU） |
|--------|--------------|---------------|
| vs Torch Eager | **15.3×** | — |
| vs Torch SDPA | — | **10.3×** |

> ⚡ Louver 实现亚线性增长的 attention 成本，远超 FlashAttention。

#### 🔍 图 7：Recall@k 对比（三款主流模型）
| 方法 | Recall@k（平均） |
|------|----------------|
| InfLLM (IVF) | ~70% |
| MagicPIG (LSH) | ~60% |
| RetrievalAttention (HNSW) | ~85% |
| **Louver** | **≥99.9%** |

> ✅ Louver 实现近乎完美的 recall，而 ANN 方法普遍存在显著假阴性。

#### 💾 表 4：KV Cache Offloading 性能（LongBench）
| 方法 | Avg F1 (%) | Search Time (ms) | Transfer Time (ms) |
|------|------------|------------------|--------------------|
| InfLLM | 26.2 | 4.89 | 7.08 |
| RetrievalAttention | 25.1 | 53.95 | 7.05 |
| **Louver (Offloaded)** | **38.9** | **0.07** | **6.02** |

> 🌟 Louver 在 offloading 场景下仍保持高准确率，且索引查找时间极低（仅 0.07ms/step）。

---

### 🔬 消融实验结果（Ablation Studies）

#### 🧩 子空间划分（Subspace Decomposition）
- 使用 $ S=16 $ 个子空间时，Louver 仅需扫描 **16.3%** 的 keys，实现 **2.42×** 速度提升。
- “Contiguous grouping”（连续位置分组）效果最好，优于随机或 PCA 分组。

#### 🛠 Enclosing Geometry 对比（Table 8）
| Clustering | Enclosing | Scan (%) | Recall |
|----------|----------|---------|-------|
| k-center | Ball | 92.4% | 99.95% |
| k-center | AABB | **60.8%** | 99.96% |
| k-means | Span-ball | 92.3% | 99.95% |

> AABB 更紧致但计算代价更高；Louver 最终选择 **ball enclosing + k-center clustering**，平衡效率与剪枝能力。

#### 📏 阈值估计模块（Threshold Oracle）
- 提供多种内置 oracle：`sample-max`, `sample-gap`, `budget(α)` 等。
- 实验表明 `sample-max` 和 `sample-gap` 更稳定，CoV < 15%，适合动态调整阈值。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **False Negatives 是稀疏注意力误差的主要来源**  
   即使只丢失一个关键 token（如列表中的某个数字），也会引发严重的错误传播，尤其是在长链推理中。

2. **注意力分布是高度动态的**  
   不同 decoding 步骤中，attention score 的尾部行为剧烈波动（见 Fig 1c），固定预算方法难以适配。

3. **Range Searching 是更合适的建模范式**  
   相比 ANN/MIPS，halfspace range searching 天然支持精确阈值控制、无假阴性，并避免 query-key 分布偏移问题。

4. **Louver 实现理论与工程的统一**  
   在保证 **zero false negatives** 的前提下，通过 subspace decomposition + ball pruning 实现高效过滤，最终性能反超 dense attention。

---

### ⚠ 局限性（Limitations）
1. **内存开销略有增加**  
   Louver 引入额外的 cluster centers 元数据，GPU 内存占用增加约 25–28%（见 Table 4）。
   
2. **未完全消除元数据存储成本**  
   当前版本仍需存储合成中心向量；未来可通过使用实际 key 作为代表进一步压缩。

3. **侧重于 decoding 阶段优化**  
   虽然可扩展至 prefilling，但本文主要聚焦于 autoregressive generation 场景。

---

### 🔮 未来工作方向
1. **零元数据开销索引**  
   使用原始 keys 替代中心向量（类似 HNSW 的 navigate nodes），彻底消除额外存储。

2. **跨层共享索引结构**  
   探索在多个 attention layer 间复用索引，减少重复构建成本。

3. **结合量化与压缩技术**  
   将 Louver 与 KV Cache quantization（如 FP8）结合，进一步降低带宽压力。

4. **支持 streaming & continual learning 场景**  
   扩展至无限长序列处理，支持滑动窗口外的老 key 安全淘汰。

---

## 总结

> **Louver 开辟了一条“正确优先”的 sparse attention 新路径：它不再追求“更快地近似”，而是致力于“精确且高效地检索”。**

通过将 sparse attention 归约为 **computational geometry 中的 halfspace range searching 问题**，Louver 首次实现了：
- ✅ **理论可证的 zero false negatives**
- ✅ **动态自适应的检索规模**
- ✅ **超越 FlashAttention 的运行时性能**

这标志着 KV Cache 管理正从“启发式丢弃”迈向“精确索引时代”，为构建高效、可靠的大模型推理系统提供了坚实基础。

</details>

---

### 2. [HexiSeq: Accommodating Long Context Training of LLMs over Heterogeneous Hardware](https://arxiv.org/abs/2605.07569)

**Authors**: Yan Liang, Youhe Jiang, Ran Yan, Binhang Yuan, Wei Wang, Chuan Wu  
**Category**: cs.DC  
**Published**: 2026-05-11  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2605.07569v1  

#### Abstract
Long-context training of large language models (LLMs) is commonly distributed with Context Parallelism (CP) and Head Parallelism (HP), but existing training systems largely assume homogeneous GPU meshes. This paper extends CP and HP to heterogeneous GPU clusters with mixed GPU models and non-uniform...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HEXISEQ: Accommodating Long Context Training of LLMs over Heterogeneous Hardware

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前的大语言模型（LLM）长上下文训练普遍依赖于**同构 GPU 集群**（homogeneous GPU clusters），即所有设备具有相同的计算能力、内存容量和通信带宽。然而，在实际生产环境中，由于硬件迭代迅速（如 A100 → H100 → B200），数据中心往往存在多种代际 GPU 共存的情况，形成**异构集群**（heterogeneous clusters）。在这种环境下，传统的 **Context Parallelism (CP)** 和 **Head Parallelism (HP)** 方法因假设对称划分而无法有效利用资源，导致以下问题：

- **计算不匹配**（C1）：高性能 GPU 被低性能 GPU 拖累，出现“木桶效应”；
- **通信瓶颈**（C2）：跨节点低带宽链路成为 Ring Attention 或 All-to-All（A2A）操作的性能瓶颈。

因此，如何在混合型号 GPU 和非均匀网络带宽下高效进行百万级 token 上下文训练成为一个亟待解决的问题。

---

### **提出了什么新方法或新思路**

本文提出 **HEXISEQ** —— 一种支持完全**非对称 CP-HP 分区**的系统，其核心思想是将硬件异构性显式建模到调度决策中，实现细粒度的负载分配。

#### 主要创新点包括：

1. **非对称调度抽象（Asymmetric Schedule Abstraction）**
   - 支持不同大小的 A2A 组（variable A2A group cardinality）
   - 组内和组间可变序列分片长度（non-uniform sequence shards）
   - 按设备能力加权分配 attention heads（weighted head placement）

2. **运行时机制支持**
   - **Heterogeneous Ragged A2A**：基于 per-rank split tables 实现非均匀的数据重分布；
   - **Sub-ring KV Exchange**：将 Ring Attention 分解为多个子环任务，适应不同 head 数量和非连续分布；
   - **Batched Async Transfers**：批量异步点对点传输以重叠通信与计算。

3. **层次化调度器（Hierarchical Scheduler）**
   将复杂的组合优化问题分解为三阶段流程：
   - **Stage I**: 基于拓扑图聚类生成高带宽域内的 A2A 分组候选；
   - **Stage II**: 在组级别按聚合算力分配全局序列长度；
   - **Stage III**: 使用坐标下降法精细化每个 rank 的序列与 head 分配，确保满足内存约束并最小化延迟。

该调度器形式化为一个受内存和通信带宽限制的约束优化问题，并结合分析型性能模型快速评估候选方案。

---

### **相比现有方法的优势**

| 对比维度 | 传统方法（USP / Ulysses / Ring Attention） | HEXISEQ |
|--------|--------------------------------------|-------|
| 并行模式 | 对称 CP/HP，固定分组结构 | 完全非对称，动态适配硬件拓扑 |
| 设备利用率 | 强制均衡负载，弱设备拖累强设备 | 按能力分配任务，提升整体吞吐 |
| 通信效率 | 忽视带宽差异，易受慢链路影响 | 显式建模通信拓扑，避免跨低带宽链路频繁交换 |
| 内存安全性 | 固定分片可能导致 OOM | 动态调整分片大小并通过可行性过滤防止溢出 |

> ✅ **优势总结**：HEXISEQ 是首个同时支持异构 CP 与 HP 调度的系统，在保持原有 attention 语义的前提下，显著提升了异构集群下的训练效率。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- 使用 **OpenWebText2** 数据集进行 tokenization；
- 输入样本通过 Megatron-LM 方式打包成固定长度的 long-context sequences（最长达 1M tokens）；

> 注：实验关注的是**单个长序列的训练性能**，而非数据分布不平衡问题。

---

### **实验设置和评估指标**

#### **物理测试平台**
- 混合 H100/A100 集群，共 8–16 GPUs；
- 节点内使用 NVLink（H100: 450 GB/s, A100: 300 GB/s），节点间使用 RDMA（25 GB/s）；
- 测试模型规模：3B、7B、13B 参数；
- 上下文长度范围：8K – 256K tokens；
- 全局 batch size = 8。

#### **大规模模拟环境**
- 模拟三种更大规模异构集群（32–128 GPUs），涵盖最多四代 GPU（H100, A100, A800, L40S）；
- 模型扩展至 70B 参数，上下文最长 1024K；
- 使用经过实测校准的分析型性能模型进行仿真。

#### **评估指标**
- **Throughput**：tokens per second (TPS)，为主要性能指标；
- **Speedup**：相对于最强 baseline 的加速比；
- **Scheduler Overhead**：调度器运行时间；
- **Feasibility Rate**：合法调度比例。

---

### **基线方法对比**

比较了三大主流 CP/HP 方法：
1. **Ring Attention** [34]：纯 CP，通过环形 KV 传递处理长序列；
2. **Ulysses** [17]：纯 HP，使用 A2A 重分布 attention heads；
3. **USP** [8]：混合 CP+HP，采用二维网格结构，枚举所有合法 (CP, HP) 因子组合取最优。

> 所有 baseline 均假设对称分区，无法感知硬件异构性。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **真实异构测试床结果（Mixed H100/A100）**
| 模型 | 上下文 | 平均提速 | 最大提速 |
|------|--------|----------|----------|
| 3B–13B | 8K–256K | **1.11×** | **1.19×** |

- 在 3B/7B 模型上，HEXISEQ 在所有配置下均优于最强 baseline；
- 随着上下文增长（如从 16K 到 32K），性能增益上升（1.08× → 1.17×），说明更长序列带来更多调度灵活性；
- 即使在高激活压力的 13B 模型上仍能维持约 1.13× 加速。

#### **大规模模拟结果（32–128 GPUs, 多代混合）**
| 模型 | 上下文 | 平均提速 | 最大提速 |
|------|--------|----------|----------|
| 13B / 70B | 最高 1024K | **1.36×** | **1.72×** |

- 性能提升随 GPU 多样性增加而增强（如四代混合 > 三代混合）；
- 在最大模拟场景（128 GPUs, 70B, 512K）中达到 **1.72×** 加速；
- 表明 HEXISEQ 在复杂异构环境中更具潜力。

#### **FLOP-等效同构集群对比**
| 场景 | 异构集群 | 同构集群 | HEXISEQ vs 最强同构 |
|------|---------|-----------|---------------------|
| Sim 4 | 48 H100 + 8 A100 | 152 A100 | 达到 Ulysses 的 **~99.5%** |
| Sim 5 | 52 H100 + 4 A100 + 8 A800 | 168 A100 | 接近 Ulysses 吞吐 |

> 🔍 **重要发现**：尽管硬件异构，HEXISEQ 可以接近甚至逼近同等总 FLOPs 下最优同构系统的性能，表明其几乎消除了“异构惩罚”。

#### **调度器开销**
- 在 128 GPU 规模下仅需 **9.0 秒**完成调度；
- 在 1024 GPU 规模下为 **444.0 秒（<8分钟）**，远小于典型训练时长；
- 层次化剪枝策略有效控制搜索空间爆炸。

---

### **消融实验结果（文中未明确列出，但从设计推断）**

虽然论文未提供独立消融图，但从调度器三阶段设计可推知：
- 若跳过 Stage I（拓扑感知分组），则无法规避低带宽链路，通信开销上升；
- 若跳过 Stage II（组级序列分配），则无法平衡组间负载；
- 若跳过 Stage III（坐标下降精调），则难以达到局部最优且可能违反内存约束。

此外，作者指出 HEXISEQ 的候选集中包含了所有 baseline 的布局，因此其性能增益来源于选择了更优的**非对称调度方案**，而非排除 baseline。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **异构集群可用于高效长上下文训练**  
   通过合理的非对称 CP-HP 调度，异构硬件可以实现接近同构集群的训练吞吐。

2. ✅ **非对称调度显著优于对称分区**  
   在真实和模拟环境中，HEXISEQ 平均提升 **1.11×–1.36×** 吞吐，最高达 **1.72×**。

3. ✅ **调度需联合考虑计算、内存与通信**  
   成功的关键在于将设备算力、HBM 容量/带宽、以及拓扑带宽统一纳入调度决策。

4. ✅ **层次化调度器高效可行**  
   通过 coarse-to-fine 的三阶段设计，在合理时间内找到高质量调度方案。

---

### **方法的局限性**

1. **仅聚焦于 attention 内部并行（CP+HP）**  
   未联合优化 Data Parallelism (DP)、Tensor Parallelism (TP) 和 Pipeline Parallelism (PP)，这些外层并行仍被视为给定。

2. **依赖准确的性能建模**  
   虽然模型在校准后预测准确，但在新型硬件或 NCCL 协议变化时可能需要重新标定。

3. **尚未部署于超大规模真实集群**  
   当前实测最大为 16 GPUs，更大规模依赖仿真，可能存在未建模的现实干扰（如背景流量、故障恢复等）。

4. **调度静态性**  
   当前调度为静态预编译，不支持运行时动态调整（如应对负载波动或设备故障）。

---

### **未来工作方向**

1. **联合优化多维并行（DP/TP/PP/CP/HP）**  
   构建统一调度框架，协同决定模型切分、微批次、序列与 head 分布。

2. **支持动态自适应调度**  
   结合运行时监控，实现弹性 re-scheduling 以应对异常或负载漂移。

3. **扩展至推理与 Serving 场景**  
   将类似思想应用于 LLM inference，进一步提升异构环境下的服务成本效益（cost-efficiency）。

4. **集成进主流训练框架**  
   如与 Megatron-LM、DeepSpeed 或 PyTorch Fully Sharded Data Parallel (FSDP) 集成，推动工业落地。

---

> 📌 **总体评价**：  
> HEXISEQ 提供了一个面向未来现实场景的重要范式转变——不再追求昂贵的同构集群，而是**主动拥抱硬件多样性**，通过智能调度释放异构资源潜能。它不仅提升了资源利用率，也为中小机构提供了低成本开展长上下文 LLM 训练的可能性。

</details>

---

### 3. [A Scalable Recipe on SuperMUC-NG Phase 2: Efficient Large-Scale Training of Language Models](https://arxiv.org/abs/2605.07726)

**Authors**: Ajay Navilarekal Rajgopal, Nikolai Solmsdorf  
**Category**: cs.DC  
**Published**: 2026-05-11  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.07726v1  

#### Abstract
Large Language Models (LLMs) continue to demonstrate superior performance with increasing scale, yet training models with billions to trillions of parameters requires staggering computational resources, e.g. a one-trillion-parameter GPT-style model requires an estimated 120 million exaflops. This ch...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Scalable Recipe on SuperMUC-NG Phase 2: Efficient Large-Scale Training of Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- 大规模语言模型（LLMs）训练需要极高的计算资源（如千亿参数模型需约1.2亿 exaflops），在实际生产级 HPC 系统上高效训练面临 **内存、通信与计算利用率之间的复杂权衡**。
- 当前许多优化依赖定制化内核或硬件特定优化，难以在通用系统中复现。

本研究旨在解决如何在 **基于 Intel Data Center GPU Max 1550 的 SuperMUC-NG Phase 2** 这类先进但受限的生产环境中，实现可复制、高效的 LLM 训练。

---

### 🚀 提出的新方法/新思路
提出了一套 **可扩展的分布式训练“配方”（scalable recipe）**，结合以下三种并行策略进行系统性调优：
- **Tensor Parallelism (TP)**：跨设备切分矩阵运算
- **Pipeline Parallelism (PP)**：将模型层划分为多个阶段流水执行
- **Sharded Data Parallelism (ZeRO-1)**：通过 DeepSpeed 的 ZeRO 技术分片优化器状态、梯度和参数以降低内存占用

该方法的关键创新在于：
- **完全使用公开可用的软件栈**（Megatron-DeepSpeed v2.4, PyTorch 2.8.0, Intel Extension for PyTorch），无需任何定制 kernel 或框架修改。
- 引入 **Bayesian Optimization（贝叶斯优化）** 自动搜索最优并行配置组合（PP, TP, MBS, GAS），显著减少手动调参成本。
- 在真实生产环境下验证方案可行性，强调 **可持续吞吐量（sustained throughput）而非峰值性能**。

---

### 🔍 相比现有方法的优势
| 维度 | 本文优势 |
|------|---------|
| **可复现性** | 使用标准发行版软件，任何用户均可在 SuperMUC-NG Phase 2 上直接复现结果 |
| **实用性** | 考虑了实际运行限制（如功耗封顶至450W/accelerator） |
| **自动化程度高** | 利用 DeepHyper 实现自动超参搜索，提升效率 |
| **全面分析** | 对 TP、PP、micro-batch size 等因素进行了消融实验与理论建模 |

---

## 2. 核心实验方法和设置

### 📊 使用的模型（非数据集）
> 注：本文聚焦于 **模型训练基础设施与并行策略优化**，未涉及具体自然语言任务或下游数据集。使用的是 GPT-style decoder-only Transformer 架构，测试不同规模的合成模型：

| 模型大小 | 参数量 | 总显存需求（bf16） |
|--------|-------|------------------|
| Small | 3.6B | 57.6 GB |
| Medium | 20B | 320 GB |
| Large | 175B | 2.8 TB |

---

### ⚙️ 实验设置
- **硬件平台**：  
  - SuperMUC-NG Phase 2，共 240 节点（234 计算节点）
  - 每节点配置：
    - 2× 4th Gen Intel Xeon CPU（共112核）
    - 512 GB DDR5 内存
    - 4× Intel Data Center GPU Max 1550（每卡128 GB HBM2e）
    - 每GPU分为2个tile → 单节点共 **8 tiles**
  - 全系统总计：**960 GPUs / 1920 tiles / ~123 TB GPU memory**
  - 网络互联：NVIDIA/Mellanox HDR InfiniBand fat-tree 结构（双端口，400 Gbit/s 聚合注入带宽）

- **软件栈**：
  - Megatron-DeepSpeed v2.4
  - DeepSpeed 0.16.9
  - PyTorch 2.8.0 + Intel Extension for PyTorch 2.8.0（XPU 支持）
  - SLES + SLURM 作业调度系统

- **精度设置**：bf16 mixed precision

- **评估方式**：
  - 所有实验仅运行 **10 个训练步**（非完整收敛），用于稳定测量吞吐量
  - 主要指标为 **有效模型 TFLOPs/s per tile**（即每tile达到的实际计算吞吐）

- **基线对比**：
  - 方法论复现自 [5]（*Optimizing distributed training on Frontier for LLMs*）
  - 无传统DL模型作为baseline，而是与已有大规模训练系统的报告性能进行横向比较（如 Frontier 上的结果）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 指标 | 数值 | 说明 |
|------|-----|------|
| **单tile峰值bf16 FLOPs** | ~570 TFLOPs/s（理论） | 来源于 Intel GPU Max 1550 规格 |
| **实测单tile吞吐（175B模型）** | **57 TFLOPs/s** | 达到理论峰值的 **~10%** |
| **弱扩展效率（Weak Scaling Efficiency）** | **93% @ 128 nodes (1024 tiles)** | 表明几乎理想的扩展性 |
| **强扩展效率（Strong Scaling Efficiency）** | **82% @ 128 nodes** | 受限于通信开销，但仍表现良好 |

> 💡 吞吐率达 57 TFLOPs/s/tile 是当前同类系统中具有竞争力的表现，尤其考虑到是在 **功耗受限的生产环境** 下取得。

---

### 🔬 消融实验结果

#### （1）Tensor Parallelism 影响（图1）
- 当 TP ≤ 8（同节点内）时，all-reduce 利用高速 Xe Link，通信高效
- 当 TP = 16（跨节点）时，吞吐急剧下降 → **建议限制 TP ≤ 8**

#### （2）Pipeline Parallelism 分析（图2–3）
- 吞吐由 **PP/M 比率**主导：
  - 固定 PP，增加 micro-batches（M）→ 吞吐上升后趋于饱和
  - 固定 M，增加 PP → “pipeline bubble”增大 → 吞吐下降
  - 若保持 PP/M 不变，则吞吐基本稳定
- 结论：应选择适中的 PP 并配合适量 micro-batching

#### （3）贝叶斯优化自动搜索结果（表2）

| 超参数 | 最优值 |
|--------|--------|
| Pipeline Parallelism (PP) | 16 |
| Tensor Parallelism (TP) | 8 |
| Micro-batch Size (MBS) | 3 |
| Gradient Accumulation Steps (GAS) | 100 |

> ✅ 此配置被用于最终的 scaling 实验，并实现了最佳吞吐。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **TP 应限制在单节点内（TP ≤ 8）**：跨节点 tensor parallelism 显著受制于 IB 通信延迟，导致性能骤降。
2. **PP/M 比率是决定 pipeline 效率的关键**：需平衡 micro-batch 数量与 pipeline 深度，避免空泡过大。
3. **可通过 ZeRO-based sharded data parallelism 高效扩展到千tile级别**：在 128 节点（1024 tiles）上实现 93% 弱扩展效率。
4. **使用公开软件栈即可实现高性能训练**：无需定制代码，提升了方法的可访问性和可复现性。
5. **贝叶斯优化能有效指导大规模训练配置搜索**：相比网格搜索更高效地探索复杂参数空间。

---

### ⚠️ 方法的局限性
- **未突破10%的硬件利用率瓶颈**：尽管已达当前实践中的典型水平，但仍有巨大优化空间。
- **未启用 ZeRO-3**：出于稳定性考虑仅使用 ZeRO-1，更高阶的分片可能进一步节省内存但带来额外通信代价。
- **功耗限制影响性能上限**：GPU 被限制在 450W（低于标称600W），实际性能反映的是持续运行能力而非峰值潜力。
- **缺乏对 energy efficiency 的量化分析**：未来可加入能耗评估。

---

### 🔮 未来工作方向
1. 探索更高级别的 ZeRO 分片（Stage 2/3）与通信压缩技术
2. 引入 sequence parallelism 进一步缓解长序列训练压力
3. 开展完整的端到端训练流程验证（而不仅是短期吞吐测试）
4. 量化能效比（FLOPs/Watt）并优化绿色AI训练路径
5. 将此 recipe 推广至其他基于 Intel GPU 的 HPC 系统

---

## ✅ 总结一句话
> 本文提供了一个 **可在 SuperMUC-NG Phase 2 上复现的大规模 LLM 训练标准化方案**，通过系统性的并行策略组合与自动化调优，在不修改软件栈的前提下实现了高达 **93% 的弱扩展效率** 和 **10% 的理论峰值利用率**，为下一代 exascale 系统上的基础模型开发提供了实用蓝图。

</details>

---

### 4. [FastOmniTMAE: Parallel Clause Learning for Scalable and Hardware-Efficient Tsetlin Embeddings](https://arxiv.org/abs/2605.06982)

**Authors**: Ahmed K. Kadhim, Lei Jiao, Rishad Shafik, Ole-Christoffer Granmo, Mayur Kishor Shende  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.06982v1  

#### Abstract
Embedding models in natural language processing (NLP) increasingly rely on deep architectures such as BERT, while simpler models such as Word2Vec provide efficient representations but limited interpretability. The Tsetlin Machine (TM) offers an alternative logic-based learning paradigm. Omni TM Auto...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《FastOmniTMAE: Parallel Clause Learning for Scalable and Hardware-Efficient Tsetlin Embeddings》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **训练效率低下**：原始的 Omni TM-AE 模型虽然能生成可解释的静态 embedding，但其训练过程依赖全局同步机制（如 `class_sum` 计算），导致训练速度极慢，尤其在大规模 NLP 数据集上难以实用。
- **硬件不匹配**：传统 GPU 架构为浮点密集计算优化（如 FMA、Tensor Cores），而 Tsetlin Machine 主要依赖逻辑运算（AND/OR/XOR），在 GPU 上执行效率低，资源利用率差。

### 🚀 提出的新方法与创新
- **FastOmniTMAE**：提出一种并行化的 Omni TM-AE 改进架构，核心是将原有序列化训练流程重构为两个独立阶段：
  - **Evaluation 阶段**：各 clause 并行评估输入。
  - **Update 阶段**：基于局部反馈概率进行状态更新，无需等待全局 `class_sum`。
- **去除了 class_sum 全局依赖**：通过实验证明，`class_sum` 对 embedding 质量影响有限，主要用于控制收敛速率；移除后可实现 clause 级别的并行训练，显著提升效率。
- **首次实现全规模 FPGA 上的 TM embedding 训练加速器**：
  - 设计了一个可复用的 SoC-FPGA 加速 IP 核心。
  - 支持 Zynq-7000（资源受限）和 Zynq UltraScale+（高性能）平台，具备良好的可扩展性。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **训练速度** | 在 CPU 上达到 **5× 更快训练**（分类任务），在 SoC-FPGA 上进一步提速至 **7× 以上** |
| **embedding 质量** | 保持与 Omni TM-AE 相当甚至更优的语义相似性和聚类能力 |
| **硬件效率** | 利用 FPGA 的逻辑并行特性，避免 GPU 浮点单元浪费，实现小硬件足迹下的高效训练 |
| **可部署性** | 支持边缘设备部署，适用于低功耗、高能效场景 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 类型 | 数据集 | 描述 |
|------|--------|------|
| **训练数据** | 1 Billion Word Dataset | 包含约 40k 词汇的大规模文本语料，用于训练 embedding |
| **分类任务** | 自定义二元标签数据 | 每个目标 token 构造正负样本对（是否支持该词） |
| **相似性评估** | RG65, WS-353, MTurk287, MTurk771, MEN | 人类标注的词语对相似度评分数据集 |
| **聚类可视化** | 手动分组语义词表 | 如 food, geography, vehicle 等类别，用于 t-SNE 可视化 |

### ⚙️ 实验设置与评估指标
| 方面 | 设置 |
|------|------|
| **模型参数** | clauses=32~160, T=20000~8000, s=1~2, state_bits=8, epochs=4~100 |
| **评估指标** | 
| - 分类性能 | Precision, Recall, F1-score |
| - 语义相似性 | Spearman ρ 和 Kendall τ 相关系数 |
| - 聚类质量 | t-SNE 可视化 |
| - 训练效率 | 训练时间（秒）、Speedup 倍数 |
| **硬件平台对比** | CPU（Core i5/i9/H100）、GPU（CUDA）、FPGA（Zybo/ZCU104） |

### 🆚 基线方法对比
| 模型 | 类型 | 是否参与比较 |
|------|------|--------------|
| **Omni TM-AE** | 原始版本 | ✅ 主要对比对象 |
| **Word2Vec** | 浅层神经网络 | ✅ 嵌入质量基准 |
| **FastText** | 子词嵌入模型 | ✅ |
| **GloVe** | 共现矩阵模型 | ✅ |
| **BERT-like** | 深度 Transformer | 引用文献中提及，未直接运行 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### （1）分类性能（100 epochs）
| 模型 | F1-score（平均） | 训练时间（100 epochs） | Speedup |
|------|------------------|------------------------|---------|
| Omni TM-AE | ~0.56（下降趋势） | 1709 秒 | 1× |
| **FastOmniTMAE** | **>0.60（稳定）** | **310 秒** | **5.5×** |

> 💡 FastOmniTMAE 不仅更快，且分类性能更高，说明去除 `class_sum` 不损害学习能力。

#### （2）语义相似性（Spearman ρ / Kendall τ）
| 模型 | Avg. ρ | Avg. τ | 最佳单项（RG65） |
|------|-------|-------|----------------|
| Word2Vec | 0.542 | 0.375 | — |
| FastText | **0.550** | 0.382 | — |
| Omni TM-AE | 0.543 | 0.390 | 0.633 (ρ) |
| **FastOmniTMAE (CPU)** | **0.537** | **0.392** | **0.656 (ρ)** ✅ |

> ✅ FastOmniTMAE 在 RG65 上取得最高 Spearman 分数，embedding 质量媲美主流模型。

#### （3）多硬件平台性能对比（Multi-Hardware Benchmark）
| 平台 | 后端 | 时间（s） | Speedup vs CPU | RG65 ρ |
|------|------|----------|---------------|--------|
| Core i9 PC | CPU | 57.7 | 1.00× | 0.680 |
| Core i9 PC | OpenCL (iGPU) | 262.2 | 0.22× | 0.668 |
| H100 Server | CPU | 87.3 | 1.00× | 0.680 |
| H100 Server | CUDA (H100 GPU) | 493.1 | 0.18× ❌ | 0.681 |
| **ZCU104 (FPGA)** | SoC | **119.9** | **7.08×** ✅ | **0.696** ✅ |
| **Zybo (FPGA)** | SoC | **657.4** | **5.58×** ✅ | **0.669** ✅ |

> 🔥 FPGA 实现不仅大幅加速，还获得最佳 embedding 质量（ZCU104 达到 ρ=0.696）！

#### （4）消融实验与分析
- **移除 class_sum 的有效性验证**：
  - 实验表明，在足够 epoch 下，local update 策略即可保证高质量 embedding。
  - `class_sum` 主要作用是降低更新频率，并非必要收敛机制。
- **state distribution 分析（Appendix A.2）**：
  - Negated literals 的 automaton states 明显高于原始 literals，表明模型学会“推理排除”。
  - 但 state 分布不能直接作为 convergence 指标，尚无明确映射关系。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **全局同步不是必须的**：`class_sum` 可被安全移除，转为 clause-level 并行训练，极大提升效率而不损失 embedding 质量。
2. **TM 不适合 GPU 加速**：尽管使用 CUDA 实现，但在 H100 上反而比 CPU 慢 5.6×，因 GPU 架构与 TM 的逻辑操作严重不匹配。
3. **FPGA 是理想平台**：
   - 成功实现首个完整的 SoC-FPGA 上的 TM embedding 训练加速器。
   - 在 ZCU104 上实现 **7× 加速 + 更高相似度得分**，验证了 logic-centric AI 的硬件潜力。
4. **embedding 来自完整 state space**：FastOmniTMAE 利用全部 automaton states（包括未激活 literal），形成 richer 表示，优于仅使用阈值以上特征的方法。

### ⚠️ 局限性
- **当前实现仍为 batch 处理模式**：尚未完全实现 streaming 或 online learning。
- **FPGA 设计需手动调优**：缺乏自动化工具链支持，移植到其他 FPGA 平台成本较高。
- **embedding 维度由 clauses 决定**：灵活性不如 Word2Vec/BERT 中自由设定 vector_size。
- **缺乏大规模下游任务测试**：目前评估集中于 similarity/clustering/classification，未接入如 QA、NER 等复杂 NLP 任务。

### 🔮 未来工作方向
- 开发 **自动编译流程** 将 FastOmniTMAE 模型映射到不同 FPGA 架构。
- 探索 **动态 clause 数量调整机制** 以适应不同 token 复杂度。
- 结合 **coalescing** 与 **multi-task learning** 实现跨 token 共享 clause，减少冗余。
- 将 FastOmniTMAE 应用于 **边缘 NLP 场景**，如智能传感器、IoT 文本理解等低功耗系统。

---

> 📌 **一句话总结**：  
> **FastOmniTMAE 通过并行化 clause 学习，解决了 Omni TM-AE 训练缓慢的问题，在保持高质量 embedding 的同时实现了高达 7× 的硬件加速，特别是在 FPGA 上展现出卓越的能效与性能平衡，推动了 logic-based AI 向实用化迈进。**

</details>

---

### 5. [SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting](https://arxiv.org/abs/2605.07243)

**Authors**: Weijie Shi, Qiang Xu, Fan Deng, Yaguang Wu, Jiarun Liu, Yehong Xu, Hao Chen, Jia Zhu, Jiajie Xu, Xiangjun Huang, Jian Yang, Xiaofang Zhou  
**Category**: cs.CL  
**Published**: 2026-05-11  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.07243v1  

#### Abstract
Speculative decoding accelerates LLM inference by drafting a tree of candidate continuations and verifying it in one target forward. Existing drafters fall into two camps with opposite weaknesses. Autoregressive drafters such as EAGLE-3 preserve dependence along each draft path but call the drafter ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting**  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### **解决的问题**
现有的 **speculative decoding** 方法在加速大语言模型（LLM）推理时面临两个对立的弱点：
- **自回归 drafters**（如 EAGLE-3）：能保持生成路径上的依赖关系，提升接受长度（accepted length），但每增加一个深度就需要一次额外的 drafter 推理，导致 drafting 成本高。
- **并行 drafters**（如 Medusa）：通过单次前向传播预测多个位置，大幅降低 drafting 成本，但忽略了位置间的依赖关系，导致生成路径不连贯，接受率下降。

因此，如何在**低 drafting 成本**与**高接受长度**之间取得平衡，是当前的关键挑战。

---

### **提出的新方法：SpecBlock**
SpecBlock 是一种**块迭代式（block-iterative）**的 drafter，结合了上述两种范式的优点，其核心创新包括：

#### ✅ **1. 块式生成（Block-wise Drafting）**
- 每次 drafter 前向传播生成 **K 个依赖的连续 token**，构成一个“块”（block）。
- 后续块可从之前块中的任意位置启动，实现树状扩展。

#### ✅ **2. 层级状态传递（Layer-wise Shift）**
- 在 Transformer 的每一层中，将前一位置的隐藏状态显式传递到当前位置，增强块内依赖建模。
- 避免了仅靠因果注意力（causal attention）导致的深层位置依赖稀释问题。

#### ✅ **3. 动态树构建（Dynamic Tree Drafting via Rank Head）**
- 引入一个**共训练的 rank head**，预测每个位置目标 token 在 draft 分布中的排名。
- 根据排名动态决定该位置的分支数量（branching width），将验证预算集中在不确定性高的位置。

#### ✅ **4. 有效前缀课程学习（Valid-Prefix Curriculum Learning）**
- 若块中某个位置预测错误，则后续位置的损失被掩码（mask），防止模型在错误上下文中学习。
- 更贴近实际推理过程，避免无效监督。

#### ✅ **5. 成本感知的在线适应（Cost-Aware Serving-Time Adaptation）**
- 利用验证器反馈（verifier feedback）作为免费信号，通过一个 **cost-aware bandit** 决定是否更新 drafter。
- 支持三种动作：跳过、仅更新输出头（head-only）、全模型更新，确保更新带来的吞吐增益超过成本。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **效率** | 相比 EAGLE-3，drafting 成本降低至 44–52%，同时保持高接受长度。 |
| **准确性** | 块内依赖机制显著提升长序列生成质量。 |
| **灵活性** | 动态树结构可根据不确定性分配资源，优于固定分支策略。 |
| **鲁棒性** | 在线适应机制应对分布偏移，持续优化部署性能。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**
- **训练数据**：
  - `UltraChat-200K` 和 `ShareGPT`：用于训练 drafter 模型。
- **评估基准（Benchmarks）**：
  - **对话**：MT-Bench
  - **代码**：HumanEval
  - **数学竞赛题**：MATH-500
  - **指令遵循**：Alpaca
  - **问答**：Natural Questions (NQ)
  - **翻译**：WMT-23

---

### **实验设置**
- **目标模型（Target Models）**：
  - `Llama-3.1-8B-Instruct`
  - `Qwen3-8B`
  - `Qwen3-32B`
- **硬件环境**：
  - 单张 NVIDIA A100-80GB GPU，batch size = 1
- **超参数**：
  - 块大小 $ K = 4 $
  - 最多 $ M = 2 $ 个 block / 迭代
  - 构建最多 60 节点的验证树
- **温度设置**：$ T = 0 $（贪婪解码）和 $ T = 1.0 $（随机采样）

---

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Speedup (Spd)** | 相比标准自回归解码的速度提升倍数 |
| **Throughput (ρ)** | 每秒生成 token 数量 |
| **Accepted Length (T)** | 每次验证平均接受的 token 数量 |
| **Drafting Cost (TD%)** | 每轮迭代中 drafter 前向传播所占延迟比例 |

---

### **基线方法对比**
| 方法 | 类型 | 特点 |
|------|------|------|
| **SpS** | 自回归 | 使用小型预训练模型作为 drafter |
| **Medusa**, **ParallelSpec** | 并行 | 单次前向预测多个位置，无依赖 |
| **Falcon** | 块式 | 半自回归块生成，静态树结构 |
| **EAGLE-3** | 自回归 | 当前最优，逐层生长动态树 |
| **SpecBlock+OSD** | 在线适应 | 始终更新 drafter，无 bandit 控制 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据（T=0, Llama-3.1-8B）**
| 方法 | Speedup (Spd) | Accepted Length (T) | Drafting Cost (TD%) |
|------|----------------|----------------------|------------------------|
| **EAGLE-3** | 3.59× | 5.89 | 31% |
| **SpecBlock** | **3.92×** | **5.16** | **16%** |
| **SpecBlock+adapt** | **4.05×** | **5.41** | **15%** |

> 🔍 **结论**：SpecBlock 在 drafting 成本仅为 EAGLE-3 的 **52%** 的情况下，仍实现了 **8–13% 的平均速度提升**；加入 cost-aware adaptation 后进一步提升至 **11–19%**。

---

### **跨模型表现汇总**
| 模型 | 方法 | 平均 Speedup 提升 vs EAGLE-3 |
|------|------|-------------------------------|
| Llama-3.1-8B | SpecBlock | +8–13% |
| Qwen3-8B | SpecBlock | +10–15% |
| Qwen3-32B | SpecBlock | +11–14% |
| 所有模型 | +adapt | **+11–19%** |

> ✅ 所有模型上均显著超越 EAGLE-3，且随着模型规模增大，相对优势更明显。

---

### **消融实验结果（Ablation Study）**
在 `Llama-3.1-8B` 上进行组件移除测试：

| 移除组件 | Speedup (Spd) ↓ | Accepted Length (T) ↓ | 影响说明 |
|----------|------------------|------------------------|---------|
| **Layer-wise Shift** | 3.21× → 2.95× | 4.41 → 4.02 | 块内依赖断裂，深层接受率下降 |
| **Prefix Broadcast** | 3.21× → 2.99× | 4.41 → 4.13 | 位置未锚定前缀，一致性变差 |
| **Valid-Prefix Curriculum** | 3.21× → 3.06× | 4.41 → 4.23 | 错误前缀污染训练 |
| **Rank-Guided Branching** | 3.21× → 3.13× | 4.41 → 4.25 | 固定分支浪费验证预算 |

> 📌 **Layer-wise shift 贡献最大**，单独带来约 **0.26× speedup** 提升。

---

## 4. 关键结论和发现

### **主要发现**
1. **块式迭代设计有效平衡效率与准确率**：
   - SpecBlock 将 drafting 成本压缩至 EAGLE-3 的一半以下，同时保持接近的接受长度。
   - 通过块间恢复机制（cross-block iteration），即使某路径失败，也可从其他节点继续扩展。

2. **动态树结构优于静态策略**：
   - Rank head 能有效识别高不确定性位置，并为其分配更多候选分支，提升整体接受率。

3. **在线适应机制可持续优化性能**：
   - Cost-aware bandit 只在预期收益大于更新成本时触发更新，避免盲目训练。
   - “head-only” 更新模式比全模型更新快一个数量级以上，适合高频轻量调整。

4. **真实场景下优势更显著**：
   - 在混合任务流（mixed-task stream）中，经过多次适应后，SpecBlock+adapt 的 speedup 持续增长，最高可达 **+12%**。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **Rank Head 准确性有限** | 当前四分位分类器仍有误判，部分位置分支过多或不足。 |
| **块大小 K 固定** | $ K=4 $ 在训练时确定，无法在推理时动态调整。 |
| **M 值未调优** | 实验统一使用 $ M=2 $，可能非所有任务最优。 |
| **跨设备同步开销** | 双GPU部署虽分离训练流，但权重同步引入额外延迟。 |

---

### **未来工作方向**
1. 设计更精细的 rank prediction 模块（如回归或细粒度分类）。
2. 支持可变块大小 $ K $ 或自适应 $ M $ 的推理调度。
3. 探索基于强化学习的动态树控制策略。
4. 将 SpecBlock 思路推广至语音、图像等生成任务。

---

> ✅ **总结一句话**：  
> **SpecBlock 通过块式迭代 + 层级状态传递 + 动态树构建，在极低 drafting 成本下实现了比 EAGLE-3 更高的端到端推理加速，是 speculative decoding 领域的重要进展。**

</details>

---

### 6. [Star Elastic: Many-in-One Reasoning LLMs with Efficient Budget Control](https://arxiv.org/abs/2605.07182)

**Authors**: Ali Taghibakhshi, Ruisi Cai, Saurav Muralidharan, Sharath Turuvekere Sreenivas, Aditya Vavre, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Sheldon Liang, Marcin Chochowski, Zijia Chen, Akhiad Bercovich, Ran Zilberstein, Ran El-Yaniv, Yonatan Geifman, Daniel Korzekwa, Yoshi Suhara, Oluwatobi Olabiyi, Ashwath Aithal, Nima Tajbakhsh, Pavlo Molchanov  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.07182v1  

#### Abstract
Training a family of large language models (LLMs), either from scratch or via iterative compression, is prohibitively expensive and inefficient, requiring separate training runs for each model in the family. In this paper, we introduce Star Elastic, a novel LLM post-training method that adds N neste...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Star Elastic: Many-in-One Reasoning LLMs with Efficient Budget Control》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前大语言模型（LLM）家族通常由多个独立训练和存储的固定大小模型组成，导致以下问题：
- **高昂的训练成本**：每个子模型需单独训练，计算资源消耗巨大。
- **部署效率低**：多模型并行部署占用大量存储空间。
- **推理僵化**：静态架构无法根据任务阶段动态调整资源分配，难以实现精度与延迟的最佳权衡。

### **提出的新方法与创新点**
本文提出了 **Star Elastic**，一种面向混合 Mamba-Transformer-MoE 架构的后训练（post-training）弹性建模框架，其核心创新包括：

#### ✅ **1. 多合一弹性模型构建（Many-in-One Elastic Modeling）**
- 在一次后训练过程中，从一个父模型（如 Nemotron Nano v3 30B）中嵌套生成多个不同规模的子模型（如 23B 和 12B），支持零样本切片（zero-shot slicing）。
- 支持沿 **SSM、Embedding Channel、MoE、FFN** 四个维度进行弹性压缩。

#### ✅ **2. 可学习路由器（Learnable Router）**
- 引入端到端可训练的 **router**，自动决定各预算下的最优子模型结构。
- 路由器通过 **knowledge distillation** 优化，联合学习架构选择与参数微调，避免传统方法中“先剪枝再蒸馏”的两阶段解耦问题。

#### ✅ **3. 弹性预算控制（Elastic Budget Control）**
- 首次在推理阶段实现 **按阶段动态模型切换**：在“思考”（thinking）和“作答”（answering）阶段使用不同的子模型。
- 提出 **Ms→ML 策略**（小模型思考 + 大模型作答），显著提升准确率-延迟帕累托前沿。

#### ✅ **4. 量化感知蒸馏（Quantization-Aware Distillation, QAD）**
- 将弹性扩展至量化领域，发布支持 **FP8 和 NVFP4** 的弹性检查点。
- 保持零样本切片能力的同时大幅降低部署内存。

### **相比现有方法的优势**
| 维度 | Star Elastic | 传统方法（如独立训练 / 压缩） |
|------|-------------|-------------------------------|
| **训练成本** | 单次训练生成多个模型，节省高达 **360× token 成本** | 每个模型需单独训练或压缩 |
| **部署效率** | 所有变体共享同一参数空间，内存减少 **>50%** | 多模型独立存储，线性增长 |
| **推理灵活性** | 支持 per-phase 动态模型选择，优化精度-延迟平衡 | 固定模型全程使用 |
| **量化兼容性** | 支持 QAD，保留弹性结构 | 量化破坏嵌套结构 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：基于 Nemotron Nano v3 的开源训练语料，包含 **70% 推理任务数据 + 30% 预训练数据**。
- **校准数据**：用于重要性评分估计，共 1024 个样本，序列长度为 8192。
- **总训练量**：约 **160B tokens**（Stage 1: 100B @8k；Stage 2: 60B @49k）。

### **实验设置**
- **基础模型**：NVIDIA Nemotron Nano v3 MoE (30B/3.6A)，混合 Mamba-Transformer-MoE 架构。
- **目标子模型**：
  - 23B (2.8A active params)
  - 12B (2.0A active params)
- **训练框架**：集成于 **NVIDIA Megatron-LM**，使用 vLLM 进行推理评测。
- **量化格式**：
  - **FP8 (E4M3)**：权重 FP8，KV/SSM 状态 BF16。
  - **NVFP4**：NVIDIA 自研 4-bit 浮点格式，带两级缩放因子。

### **评估指标**
| 类别 | 指标 |
|------|------|
| **准确性** | AIME-2025, GPQA, MMLU-Pro, LiveCodeBench v5, IFBench, Tau Bench |
| **效率** | Latency, Throughput (tokens/s), Memory Footprint |
| **帕累托性能** | Accuracy vs. Normalized Inference Time 曲线 |
| **消融分析** | 不同训练策略、采样方式、压缩维度的影响 |

### **基线方法对比**
- **独立训练模型**：Qwen3-30B-A3B (3.3A)
- **同类弹性方法**：Nemotron Elastic [1]（仅支持 Mamba-Attention）
- **传统压缩方法**：Structured pruning + knowledge distillation
- **量化方法**：Post-Training Quantization (PTQ)

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **准确性表现（Table 1）**
| 模型 | AIME-2025 | GPQA | MMLU-Pro | 平均得分 |
|------|-----------|------|----------|--------|
| **Nano v3 Elastic-30B** | **88.54** | **72.10** | **78.63** | **74.94** |
| Nano v3-30B (原模型) | 87.92 | 73.11 | 78.86 | — |
| Qwen3-30B (3.3A) | 80.00 | 70.83 | 81.11 | — |
| **Nano v3 Elastic-23B** | 85.63 | 69.82 | 76.07 | 72.04 |
| **Nano v3 Elastic-12B** | 78.54 | 57.39 | 68.28 | 65.16 |

> ✅ 所有嵌套模型均匹配或超越同等规模独立训练模型。

#### 🔹 **弹性预算控制效果（Figure 1 & 3）**
- **最高提升**：**+16% 准确率** 或 **1.9× 更低延迟**。
- 最优配置为 **23B → 30B**（小模型思考 + 大模型作答），尤其在中等延迟区间表现最佳。
- 在高延迟场景下，**30B → 23B** 略优；极低延迟则可用 **12B ↔ 30B**。

#### 🔹 **吞吐量提升（Table 2）**
| 模型 | Max Batch Size | Speedup (vs. 30B) |
|------|----------------|------------------|
| 30B (3.6A) | 36 | 1.0× |
| 23B (2.8A) | 108 | **1.8×** |
| 12B (2.0A) | 224 | **2.4×** |

> 更小模型可在相同 GPU 上运行更大 batch，显著提高吞吐。

#### 🔹 **训练成本节约**
- 相比从头预训练：**360× token 节省**
- 相比 SOTA 压缩方法（如 Nemotron Elastic）：**7× token 节省**

#### 🔹 **部署内存效率（Table 3）**
| 配置 | 内存占用 (BF16) |
|------|----------------|
| 分离式部署 (12B+23B+30B) | 126.1 GB |
| **Star Elastic 弹性部署** | **58.9 GB** |

> 内存减少 **>50%**，仅需存储最大模型即可切片出所有变体。

#### 🔹 **量化恢复效果（Table 4）**
| 模型 | FP8 恢复率 | NVFP4 恢复率 |
|------|------------|--------------|
| 30B | 98.69% | 97.79% |
| 23B | 99.03% | 99.15% |
| 12B | 100.26% | 97.10% |

> QAD 成功恢复绝大多数精度，且支持零样本切片。

#### 🔹 **量化吞吐优势（Figure 4）**
- **NVFP4 30B 模型** 在 RTX 5090 上可部署，而 BF16 版本 OOM。
- **12B NVFP4** 在 RTX 5080 上可达 **7,426 tokens/s**，是 30B BF16 基线的 **3.4×**。

---

### **消融实验结果**

#### 📊 **两阶段训练有效性（Table 7）**
- 第二阶段（49K 上下文）使小模型在复杂推理任务上大幅提升：
  - 6B 模型在 AIME-2025 上提升 **+19.8%**
  - 12B 模型提升 **+4.0%**
- 表明长上下文训练对多步推理至关重要。

#### 📊 **预算采样策略（Ablation in [1]）**
- 使用非均匀采样（p(30B)=0.5, p(23B)=0.3, p(12B)=0.2）防止大模型退化。
- 在 AIME-2025 上带来 **+3.54 pts** 提升。

#### 📊 **宽度 vs. 深度压缩（Table 10）**
| 方法 | 平均性能（相对基线） |
|------|---------------------|
| 宽度压缩（Width） | **98.1%** |
| 深度压缩（Depth） | 95.2% |

> 宽度压缩更优，深度压缩在 HumanEval 和 MMLU-Pro 上下降明显。

#### 📊 **缓存状态重用（Table 12–13）**
- KV Cache 与 SSM State 的平均余弦相似度达 **0.89–0.96**。
- 在 GSM8k 上移植 6B 缓存至 12B 模型，准确率仅下降 **0.75%**（90.93% → 90.18%），验证缓存兼容性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **单次训练可生成高质量多尺寸模型**：Star Elastic 在一次后训练中生成 12B/23B/30B 模型，性能媲美独立训练。
2. ✅ **弹性预算控制显著优化推理效率**：采用 **Ms→ML** 策略可在不牺牲准确率的前提下大幅降低延迟。
3. ✅ **量化与弹性可协同设计**：通过 **QAD** 可在 FP8/NVFP4 下保持嵌套结构和零样本切片能力。
4. ✅ **缓存状态高度兼容**：不同嵌套模型间 KV/SSM 缓存可安全移植，为未来框架级优化奠定基础。

### **方法的局限性**
- 当前最小压缩比约为 **2.5×**（30B → 12B），尚难达到极端压缩（如 10×）。
- 缓存重用尚未被主流推理引擎（如 vLLM）原生支持，目前仍计入重计算开销。
- 路由器未实现任务自适应路由（task-specific routing），需手动指定预算。

### **未来工作方向**
- 探索 **超大规模压缩比**（如 10×）下的弹性建模。
- 开发 **任务感知路由器**，根据输入类型（数学、代码、多语言）自动选择最优模型路径。
- 推动 **推理框架支持缓存重用**，释放 Star Elastic 的全部潜力。
- 将 Star Elastic 扩展至更多架构（如纯 Transformer、Phi 系列）。

---

> **总结一句话**：  
> **Star Elastic 实现了“一次训练，多种用途”，不仅极大降低了 LLM 家族的训练与部署成本，还首次实现了推理过程中的动态资源调度，在准确性、延迟、内存之间达到了前所未有的帕累托最优。**

</details>

---

### 7. [Stochastic Transition-Map Distillation for Fast Probabilistic Inference](https://arxiv.org/abs/2605.07661)

**Authors**: George Rapakoulias, Peter Garud, Lingjiong Zhu, Panagiotis Tsiotras  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.07661v1  

#### Abstract
Diffusion models achieve strong generation quality, diversity, and distribution coverage, but their performance often comes with expensive inference. In this work, we propose Stochastic Transition-Map Distillation (STMD), a teacher-free framework for accelerating diffusion model inference while pres...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Stochastic Transition-Map Distillation for Fast Probabilistic Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
扩散模型（Diffusion Models）在生成质量、多样性和分布覆盖方面表现出色，但其推理过程通常依赖于对连续时间随机微分方程（SDE）的数值积分，导致**推理速度慢、计算成本高**。现有的加速方法（如蒸馏、Flow Matching）大多专注于**确定性推理**，牺牲了生成过程中的**随机性**，限制了其在需要概率采样的下游任务（如逆问题求解、后验采样、能量模型微调）中的应用。

### **提出的新方法**
本文提出了 **Stochastic Transition-Map Distillation (STMD)**，一种**无需预训练教师模型**（teacher-free）的框架，用于加速扩散模型的推理，同时保留其**概率性生成能力**。

- **核心思想**：不同于传统的基于Score的扩散模型（仅建模后验均值），STMD直接学习**反向SDE的完整转移映射（transition map）**，即从噪声状态 $x_t$ 到更干净状态 $x_{t'}$ 的条件分布 $p(x_{t'}|x_t)$。
- **技术实现**：采用**Conditional Mean Flow**模型来参数化该转移映射。Mean Flow通过学习轨迹上的平均速度（average velocity）来显式地建模有限时间步长的流动映射，从而支持单步或少数几步的高效采样。

### **相比现有方法的优势**
| 方面 | STMD | 现有方法（如Consistency Models, DDIM Distillation） |
|------|------|--------------------------------------------------|
| **训练范式** | **Teacher-free**，无需预训练的教师模型进行知识蒸馏 | 通常需要一个预训练的扩散模型作为教师 |
| **优化复杂度** | 无双层优化（bi-level optimization），训练简单高效 | 部分方法涉及复杂的双层优化或轨迹缓存 |
| **推理性质** | **保持完全的随机性**，支持概率性采样 | 多为确定性采样，给定初始噪声输出唯一结果 |
| **理论基础** | 提供了**2-Wasserstein距离下的收敛性证明**，为Mean Flow及其条件变体提供了首个此类理论保证 | 多数方法缺乏严格的收敛性分析 |
| **适用场景** | 特别适用于需要**随机推理**的任务，如扩散后验采样、逆问题、图像修复（inpainting）等 | 主要针对快速生成，通用性受限 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MNIST**：手写数字数据集，用于验证基本生成能力。
- **CIFAR-10**：彩色自然图像数据集，评估在更复杂数据上的表现。
- **CelebA**：人脸图像数据集，用于高质量图像生成和图像修复（inpainting）任务。

### **实验设置和评估指标**
- **模型架构**：
  - MNIST 和 CIFAR-10：使用 **U-Net** 架构，基于 `diffusers` 工具箱实现。
  - CelebA：使用 **latent DiT**（Diffusion Transformer）架构。
- **训练细节**：
  - 在单张 RTX 5090 GPU 上训练。
  - 所有方法使用相同的网络骨干和训练迭代次数，确保公平比较。
  - 使用 **adaptive loss** 和特定的 $r, s$ 采样策略（lognorm sampler）。
- **评估指标**：
  - **MNIST**：由于FID在MNIST上不可靠，采用在自定义分类器**最后隐藏层特征**上计算的 **Fréchet Distance (FD)**。
  - **CIFAR-10 和 CelebA**：使用标准的 **Fréchet Inception Distance (FID)**。
  - **NFE (Number of Function Evaluations)**：衡量推理效率的关键指标，越低表示推理越快。

### **基线方法对比**
- **DDPM Baseline**：原始的去噪扩散概率模型，作为性能基准。
- **Vanilla Mean Flow**：原始的Mean Flow模型（Geng et al., 2025a），用于对比条件化扩展的有效性。
- 所有方法均从零开始训练，使用相同设置。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 数据集 | 方法 | NFE | 性能指标 |
|--------|------|-----|---------|
| **MNIST** | STMD ($n_{\text{inf}}=1, n_{\text{mf}}=1$) | 1 | FD ≈ 24 |
| | STMD ($n_{\text{inf}}=4, n_{\text{mf}}=2$) | 8 | FD ≈ 18 |
| | Vanilla Mean Flow | 1 | FD ≈ 28 |
| | DDPM Baseline | >100 | FD ≈ 16 |
| **CIFAR-10** | STMD ($n_{\text{inf}}=4, n_{\text{mf}}=2$) | 8 | FID ≈ 3.2 |
| | Vanilla Mean Flow | 1 | FID ≈ 3.4 |
| | DDPM Baseline | >100 | FID ≈ 3.0 |
| **CelebA** | STMD ($n_{\text{inf}}=4, n_{\text{mf}}=2$) | 8 | FID = **8.28** |
| | STMD (Inpainting) | 50 | Qualitative results shown in Fig. 6 |

> 注：具体数值根据图中趋势估计，原文未提供精确表格。

### **与基线方法的对比结果**
- **推理效率**：STMD在**极低NFE（1-8步）** 下即可生成高质量样本，显著优于需数百步的DDPM。
- **生成质量**：在相同训练条件下，STMD的FID/FD分数**与Vanilla Mean Flow相当甚至略优**，且远优于同NFE下的DDPM。
- **随机性保留**：STMD成功保留了扩散模型的**多样性**，而确定性方法在低步数下易出现模式坍塌或模糊。

### **消融实验结果**
虽然论文未明确列出独立的消融研究章节，但从实验设计和对比中可推断：
- **条件化Mean Flow的有效性**：STMD在CelebA上的成功应用（结合DiT和时间条件）验证了条件化扩展的可行性。
- **训练策略的影响**：使用 $s-r, s, t$ 作为条件输入而非 $r, s, t$ 被发现效果更好，体现了设计选择的重要性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **STMD是一种高效的teacher-free蒸馏框架**，能够将复杂的扩散SDE压缩为单步或少数几步的**随机生成器**。
2. 通过学习**完整的转移映射**而非仅均值，STMD成功**保留了扩散模型的概率性本质**，使其适用于更广泛的下游任务。
3. 提出的**Conditional Mean Flow**是实现这一目标的有效工具，并获得了坚实的**Wasserstein收敛性理论支持**。
4. 实验表明，STMD在**MNIST、CIFAR-10和CelebA**上均能以极低的NFE实现与基线相当的生成质量，尤其在**图像修复**等任务中展现出潜力。

### **方法的局限性**
- **理论假设**：收敛性分析依赖于一些理想化假设，如Lipschitz连续性，在实际复杂模型中可能不完全成立。
- **架构依赖**：当前实现依赖于U-Net或DiT等特定架构，泛化到其他模型结构需进一步验证。
- **极端低步数性能**：在NFE=1的极限情况下，生成质量仍有提升空间，与多步扩散模型存在差距。
- **计算资源**：尽管推理快，但训练仍需大量GPU资源（如RTX 5090）。

### **未来工作方向**
- 将STMD应用于更复杂的**逆问题**（inverse problems），如超分辨率、去噪、医学图像重建等。
- 探索**文本到图像生成**中的应用，结合CLIP等文本编码器实现条件化生成。
- 进一步优化**单步生成**的质量，缩小与多步模型的性能差距。
- 研究更轻量化的模型架构，降低训练门槛。
- 将理论分析扩展到更一般的SDE和损失函数形式。

--- 

> **总结**：STMD为扩散模型的快速概率推理提供了一个新颖、高效且理论上可靠的解决方案，其**teacher-free**和**保留随机性**的特点使其在众多应用场景中具有重要价值。

</details>

---

### 8. [GASim: A Graph-Accelerated Hybrid Framework for Social Simulation](https://arxiv.org/abs/2605.07692)

**Authors**: Xuan Zhou, Yanhui Sun, Hantao Yao, Allen He, Yongdong Zhang, Wu Liu  
**Category**: cs.AI  
**Published**: 2026-05-11  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.07692v1  

#### Abstract
Large-scale social simulators are essential for studying complex social patterns. Prior work explores hybrid methods to scale up simulations, combining large language models (LLM)-based agents with numerical agent-based models (ABM). However, this incurs high latency due to expensive memory retrieva...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《GASim: A Graph-Accelerated Hybrid Framework for Social Simulation》核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大规模社会模拟（large-scale social simulation）在研究复杂社会动态（如舆论演化）中至关重要，但面临两大瓶颈：

1. **高延迟**：基于LLM的智能体在进行记忆检索时依赖昂贵的LLM推理过程，导致计算开销巨大。
2. **顺序执行瓶颈**：普通智能体使用的数值型Agent-Based Model（ABM）通常按顺序更新，难以并行化，导致扩展性差。

此外，传统混合框架（如HiSim）采用静态划分核心/普通智能体的方式，无法捕捉动态涌现的意见领袖（opinion leaders），影响模拟的真实性。

---

### **提出了什么新方法或新思路**

本文提出 **GASim** —— 一种图加速的混合多智能体框架，通过三个核心模块解决上述问题：

#### **(1) Graph-Optimized Memory (GOM)**  
针对**核心智能体**（LLM-driven agents）的记忆检索瓶颈：
- 构建稀疏的**记忆图**（memory graph），节点为历史消息，边由内容相似性、关键词和立场（opinion value）共同决定。
- 将记忆检索建模为一个**凸优化问题**，通过轻量级图传播算法求解，避免调用LLM进行检索，显著降低延迟。

#### **(2) Graph Message Passing (GMP)**  
针对**普通智能体**（ordinary agents）的顺序更新问题：
- 利用**Graph Attention Network (GAT)** 实现并行化的意见更新。
- 融合动态特征（个体及邻居立场统计）和静态特征（用户画像BERT嵌入），实现细粒度交互建模。

#### **(3) Entropy-Driven Grouping (EDG)**  
解决智能体角色静态划分的问题：
- 基于局部邻域的**信息熵**（information entropy）动态识别处于观点多样性环境中的“潜在意见领袖”作为核心智能体。
- 遵循帕累托原则（Pareto principle），每轮选择Top-K熵值最高的智能体为核心代理，提升模型对社会影响力的动态捕捉能力。

---

### **相比现有方法的优势**

| 维度 | 优势 |
|------|------|
| **效率** | 端到端速度提升 **9.94×**，核心阶段提速 **16.39×**，普通智能体阶段提速 **27.49×** |
| **成本** | Token消耗降至基线的 **<20%**（仅为HiSim的1/5，非混合框架的1/400） |
| **准确性** | 在真实舆论趋势对齐上表现更优，Fréchet距离更低，相关性更高 |
| **动态适应性** | EDG机制能自适应识别新兴意见领袖，优于静态度数划分 |

---

## 2. 核心实验方法和设置

### **使用了哪些数据集**

构建了三个基于社交媒体的真实世界话题数据集，均来自公开爬虫（Apify / WeiboSpider）：

| 数据集 | 平台 | 主题 | 用户数 | 内容数 | 时间跨度 |
|--------|------|------|--------|--------|----------|
| **Politics** | X (Twitter) | 特朗普与俄罗斯干预大选争议 | 9,135 | 12,404 | 2017.05–12 |
| **Business** | X (Twitter) | 新疆棉“强迫劳动”争议 | 9,150 | 14,494 | 2021.03–07 |
| **Education** | Sina Weibo | 阿里巴巴数学竞赛作弊疑云 | 11,454 | 135,528 | 2024.06–11 |

所有数据集标准化为 **30个时间步**，每条内容标注了[-1, +1]范围内的**opinion value**（由gpt-4o-mini打分）。

---

### **实验设置**

- **智能体总数**：10,000
- **核心智能体数量**：Top-K = 100（由EDG动态选出）
- **LLM模型**：Llama-3.1-8B-Instruct（本地部署）
- **嵌入模型**：BGE-small-en-v1.5（用于内容编码）
- **硬件配置**：40核CPU，双NVIDIA vGPU（各48GB），180GB内存

---

### **评估指标**

从多个维度评估模拟质量：

| 指标 | 类型 | 含义 |
|------|------|------|
| **ΔBias ↓** | 统计 | 模拟与真实舆论均值的平均绝对偏差 |
| **ΔDiv ↓** | 统计 | 偏差的方差，衡量稳定性 |
| **Corr. ↑** | 统计 | 皮尔逊相关系数，衡量线性一致性 |
| **F. ↓** | 几何 | Fréchet距离，衡量曲线形状与时序的整体相似性 |

> 注：↓ 表示越小越好，↑ 表示越大越好。

---

### **基线方法对比**

#### **传统ABM模型**
- **HK** (Hegselmann-Krause)
- **RA** (Deffuant)
- **Lorenz** (心理理论驱动)

#### **LLM-based方法**
- **SOD**：全LLM框架，强调认知偏见建模
- **HiSim**：当前最优混合框架（少量LLM核心 + 大量ABM普通智能体）

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **效率与成本对比（vs. HiSim）**

| 指标 | HiSim | GASim | Speedup |
|------|-------|--------|---------|
| `T_core`（核心阶段耗时） | 316.33 min | 19.30 min | **16.39×** |
| `T_ordi`（普通智能体阶段） | 84.13 min | 3.06 min | **27.49×** |
| `T_total`（总耗时） | 401.84 min (~6.7h) | 40.43 min (~0.67h) | **9.94×** |
| **Token消耗（10k agents）** | 316,944 | **61,771** | **<20% of baseline** |

> 图3显示，随着agent数量增长，GASim的成本优势进一步扩大。

---

#### **趋势对齐性能（Politics 数据集）**

| 方法 | ΔBias↓ | ΔDiv↓ | Corr.↑ | F.↓ |
|------|--------|--------|--------|-----|
| HiSim | 0.1069 | 0.0167 | -0.003 | 0.1622 |
| **GASim (Ours)** | **0.0700** | **0.0074** | **0.4261** | **0.1349** |

- ΔBias降低34.5%
- ΔDiv降低55.7%
- Corr.提升超过40个百分点
- F.下降16.8%，表明几何轨迹更贴近真实趋势

> 在Business和Education任务上也全面领先。

---

#### **消融实验（Ablation Study）**

在Politics数据集上的消融结果如下：

| 方法 | ΔBias | ΔDiv | Corr. | F. |
|------|--------|--------|--------|-----|
| **GASim (Full)** | **0.0700** | **0.0074** | **0.4261** | **0.1349** |
| w/o GOM | 0.0771 (+10%) | 0.0089 (+20%) | 0.2942 (-30.96%) | 0.1406 |
| w/o GMP | 0.1027 (+46.7%) | 0.1346 (+1717%) | -0.0989 | 0.2291 |
| w/o EDG | 0.0872 (+24.6%) | 0.0109 (+47.3%) | 0.2528 | 0.1391 |

> 结论：
- 移除 **GMP** 导致最严重退化，说明并行更新对模拟稳定性至关重要。
- 移除 **GOM** 显著增加偏差与波动，验证其在高效准确记忆检索中的作用。
- 移除 **EDG** 导致ΔDiv大幅上升，说明动态角色分配对稳定模拟的关键意义。

---

### **额外验证：LoCoMo记忆检索基准测试**

| 方法 | Overall Accuracy (%) |
|------|------------------------|
| A-Mem | 48.38 |
| Mem0 | 66.80 |
| **GOM (Ours)** | **71.56** ✅ |

- 在Single-Hop、Multi-Hop、Temporal等类型上均显著领先（约+10%）
- 表明GOM的图优化机制能更好支持长期、连贯的记忆推理

---

## 4. 关键结论和发现

### **主要发现**

1. **图结构可有效替代LLM密集检索**：GOM证明，通过构建稀疏记忆图并使用轻量图传播，可在不牺牲准确性的前提下极大降低LLM调用频率。
2. **普通智能体适合用GNN建模**：GMP将ABM升级为可学习、可并行的GAT架构，在保持解释性的同时大幅提升效率与拟合能力。
3. **动态角色划分优于静态策略**：EDG基于信息熵的选择机制能有效识别真正具有影响力的意见节点，且实证分析表明其选出的核心智能体大多位于top 20% in-degree层级（平均94.1%）。
4. **混合范式兼具效率与真实性**：GASim实现了“少数深度思考者（LLM）+多数快速反应者（GMP）”的社会分工模拟，在真实舆论对齐上超越纯LLM或纯ABM方法。

---

### **局限性**

1. **LLM生成内容缺乏真实性**：生成的文本可能反映LLM自身偏见，而非完全真实的人类表达。
2. **忽略多模态信息**：当前仅处理文本交互，未考虑图像、视频等内容在舆论传播中的重要作用。
3. **依赖合成标签**：opinion value由LLM标注，可能存在系统性偏差。

---

### **未来工作方向**

- 引入**多模态感知能力**，支持图文联合推理。
- 探索**去中心化训练机制**，减少对集中式LLM服务的依赖。
- 扩展至**跨平台异构网络模拟**，研究信息在不同社区间的扩散模式。
- 加强**伦理控制机制**，防止模拟被滥用于操纵舆论。

---

> 🔗 **代码开源地址**：[https://github.com/Jasmine0201/GASim](https://github.com/Jasmine0201/GASim)

</details>

---

### 9. [Weblica: Scalable and Reproducible Training Environments for Visual Web Agents](https://arxiv.org/abs/2605.06761)

**Authors**: O\u{g}uzhan Fatih Kar, Roman Bachmann, Yuanzheng Gong, Anders Boesen Lindbo Larsen, Afshin Dehghan  
**Category**: cs.AI  
**Published**: 2026-05-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.06761v1  

#### Abstract
The web is complex, open-ended, and constantly changing, making it challenging to scale training data for visual web agents. Existing data collection attempts remain limited to offline trajectories for supervised fine-tuning or a handful of simulated environments for RL training, thus failing to cap...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：WEBLICA: Scalable and Reproducible Training Environments for Visual Web Agents**

---

## **1. 主要贡献和创新点**

### **解决的问题**
现有的视觉 Web Agent 训练面临以下挑战：
- **训练数据难以规模化**：基于真实网站的离线轨迹（offline trajectories）缺乏交互性，无法支持强化学习（RL）所需的探索与试错。
- **模拟环境泛化性差**：现有模拟环境（如 WebArena）仅覆盖少数人工定义的领域，无法反映真实 Web 的多样性。
- **实时网页训练不稳定**：直接在 live web 上训练易受超时、Bot 检测、页面动态变化等问题影响，导致训练不可复现且效率低下。

### **提出的新方法**
本文提出了 **WEBLICA**（Web Replica），一个用于构建可扩展、可复现的视觉 Web Agent 训练环境的框架，包含两大核心机制：

#### **(1) WEBLICA-CACHE：HTTP-level 缓存系统**
- 在录制阶段捕获完整的 HTTP 流量（请求/响应）。
- 自动识别并过滤**易变参数**（volatile parameters，如时间戳 `_t`、会话 ID `sess`），生成站点特定的缓存规则。
- 支持在网络隔离下完全确定性地回放真实网站的交互行为，保留视觉状态和功能逻辑。

#### **(2) WEBLICA-SYNTH：基于 LLM 的环境合成**
- 利用 **Claude Code (Opus 4.5)** 等 agentic coding 工具自动生成交互式静态网页。
- 输入参数包括：
  - **目标能力**（如表单填写、日期选择、地图交互等 144 类高层能力）
  - **网站类别**（1,160 种，如银行、瑜伽馆、航空等）
  - **视觉风格**（961 种，如极简主义、复古风、赛博朋克等）
- 自动生成任务（tasks）并自我验证（通过 Playwright 截图迭代优化），确保功能完整性和视觉质量。

### **相比现有方法的优势**
| 维度 | 传统方法 | WEBLICA |
|------|--------|--------|
| **多样性** | 有限（<10 个模拟站点） | 千级真实 + 千级合成站点 |
| **交互性** | 静态轨迹无交互 | 支持完整 RL 训练 |
| **可复现性** | Live web 不可复现 | 完全本地化、离线运行 |
| **训练速度** | 受网络延迟限制 | 环境交互快 30–40%，动作延迟降至 50–150ms |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据来源**：
  - **WEBLICA-CACHE**：从 InstaV3 数据集中匹配 15.6K 个可在缓存环境下完成的任务。
  - **WEBLICA-SYNTH**：合成了 310 个高阶能力站点 + 2,500 个细粒度能力站点，共 **44,227 个训练任务**。
- **SFT 数据**：由 Qwen3-VL-32B-Instruct 在 InstaV3 查询上生成，经 LLM-as-Judge 过滤出 **51.7K 成功轨迹**。

### **实验设置**
- **Agent Formulation**：
  - 建模为 **POMDP**（部分可观测马尔可夫决策过程）。
  - 观察空间：1280×720 分辨率截图 + 当前 URL。
  - 动作空间：坐标系动作（click/hover）、文本输入、按键、滚动、导航控制、stop。
- **Policy 模型**：基于 **Qwen3-VL-Instruct** 系列模型（2B/4B/8B），支持纯图像输入与坐标预测。
- **训练流程**：
  1. **SFT Warm-Start**：先在高质量轨迹上微调。
  2. **RL Training**：使用 **Dr. GRPO** 算法，结合 **LLM-as-Judge** 奖励信号进行策略优化。

### **评估指标**
- **Pass@k**：k 次独立尝试中至少一次成功完成任务的比例（k=1,2,4,8）。
- **总步数预算**：随 k 线性增长（如 pass@8 = 8×30 步）。
- **评价方式**：
  - 使用各基准官方 Judge（如 Online-Mind2Web、DeepShop）。
  - 对开放任务采用 **GPT-4o 作为 Judge**，判断是否满足任务要求，人类一致性达 **88%**。

### **基线方法对比**
| 类型 | 基线模型 |
|------|--------|
| **API-only** | OpenAI CUA, Gemini CUA, Yutori Navigator |
| **Open-weight** | Qwen3-VL-Instruct, UI-TARS-1.5-7B, GLM-4.1V-9B-Thinking, Fara-7B, MolmoWeb-8B |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**

| Model | Total #Steps | Online-Mind2Web | DeepShop | WebTailBench | Avg. | WEBLICA-val |
|-------|--------------|------------------|----------|---------------|------|-------------|
| **Qwen3-VL-Instruct-8B** | 30 | 28.6 | 24.1 | 21.8 | **24.8** | 56.9 |
| **MolmoWeb-8B** | ≥100 | 35.3 | 42.3 | 49.5 | **42.4** | — |
| **WEBLICA-8B (k=1)** | 30 | **39.2** | **34.2** | **33.5** | **35.6** | **70.6** |
| **WEBLICA-8B (k=4)** | 120 | 60.5 | 55.9 | 60.3 | **58.9** | 84.7 |
| **WEBLICA-8B (k=8)** | 240 | 68.8 | 65.8 | 72.2 | **68.9** | 88.6 |

> ✅ **亮点**：
> - 仅用 **30 步（k=1）** 就超越多数需 >100 步的开源模型。
> - 在 **Avg. 指标上，pass@4 达到 58.9%**，接近 Gemini CUA（60.8%）。
> - **WEBLICA-val 上 pass@1 达 70.6%**，显著优于基线。

### **与基线方法的对比结果**
- **优于同规模开源模型**：
  - 在 Online-Mind2Web 上，**39.2% vs 35.3%（MolmoWeb-8B）**，提升 **+3.9pp**。
  - 使用更少步数（30 vs ≥100）即实现更强性能。
- **媲美商业 API 模型**：
  - **WEBLICA-8B (k=4)** 平均得分 **58.9%**，已接近 **Gemini CUA (60.8%)**。
  - **k=8 时达到 68.9%**，远超所有非专有模型。

### **消融实验结果（Ablation Studies）**

#### **(1) 训练阶段分析（Figure 5）**
- **SFT + RL > SFT alone > Base Model**
- 在 8B 模型上：
  - Base → SFT：28.6% → 37.6%
  - SFT → SFT+RL：37.6% → 39.2%
- 表明 **RL 是性能跃升的关键**，尤其能有效利用更多推理步数。

#### **(2) 环境类型对比（Figure 4b）**
- **Synth > Cache > Base**
- 合成环境在大多数任务上表现更好：
  - Online-Mind2Web：39.2% vs 35.3%
  - WebTailBench：33.5% vs 30.2%
- 说明 **LLM 合成环境虽存在 sim-to-real gap，但仍极具训练价值**。

#### **(3) 测试时计算扩展性（Test-Time Scaling）**
- **WEBLICA-8B 能有效利用更多计算资源**：
  - 增加每轮动作数（15→30 步）：pass@1 从 32.6% → 39.2%
  - 增加尝试次数（k=1→8）：pass@8 达 68.8%
- 相比之下，base model 几乎无增益，表明 **RL 训练使模型具备“规划”和“重试”能力**。

#### **(4) 接地能力保持（Grounding Preservation）**
| Model | MMBench-GUI | ScreenSpot-v2 | ScreenSpot-Pro |
|-------|-------------|----------------|----------------|
| Qwen3-VL-Instruct-8B | 82.85 | 93.95 | 54.71 |
| **WEBLICA-8B** | **83.74** | **94.50** | **55.28** |

> 🔍 结果显示：**尽管未使用专门的 grounding 数据，接地能力仍略有提升**，说明性能提升源于导航策略改进而非视觉理解退化。

---

## **4. 关键结论和发现**

### **主要发现**
1. **可复现的大规模 RL 训练是可行的**：通过 HTTP 缓存 + LLM 合成，可在本地构建数千个多样化、交互式的训练环境。
2. **合成环境具有强大训练价值**：即使存在 sim-to-real gap，LLM 生成的网站也能有效提升真实任务上的性能。
3. **RL 显著增强代理的规划与容错能力**：相比 SFT，RL 使模型能更好地利用额外测试时计算（steps 和 retries）。
4. **无需 DOM 或 set-of-marks 注解**：纯基于截图的 coordinate-based action 设计更具通用性和鲁棒性。

### **局限性**
- **缓存环境是静态快照**：无法反映网站随时间的变化，也不支持复杂动态应用（如 WebSocket）。
- **合成环境仍有 sim-to-real gap**：设计保真度依赖于当前 LLM 的生成能力。
- **单回合任务设定**：不支持多轮对话、用户反馈介入或个性化记忆。
- **训练仍基于简单 RL 框架**：未引入 long-horizon RL 或 error recovery 等高级机制。

### **未来工作方向**
- 扩展至 **移动端和桌面端 GUI 环境**，推动通用计算机使用 Agent。
- 引入 **multi-turn、human-in-the-loop** 设置，支持持续交互与纠正。
- 构建 **动态更新的缓存机制**，模拟网站演化。
- 探索 **更强大的生成模型** 来缩小 sim-to-real 差距。
- 研究 **long-horizon RL 与更大规模 RL compute scaling**。

---

> 🏁 **总结一句话**：  
> **WEBLICA 通过“真实网站缓存 + LLM 合成环境”的双轨策略，首次实现了大规模、可复现、高效的视觉 Web Agent RL 训练，其 8B 模型在多个基准上超越同类开源模型，并逼近商业 API 模型性能。**

</details>

---

### 10. [Efficient Data Selection for Multimodal Models via Incremental Optimization Utility](https://arxiv.org/abs/2605.07488)

**Authors**: Jinhao Jing, Qiannian Zhao, Chao Huang, Zhan Su  
**Category**: cs.AI  
**Published**: 2026-05-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.07488v1  

#### Abstract
The scaling of Large Multimodal Models (LMMs) is constrained by the quality-quantity trade-off inherent in synthetic data. Previous approaches, such as LLM-as-a-Judge, have proven their effectiveness in addressing this but suffer from prohibitive computational costs and lack of interpretability. To ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient Data Selection for Multimodal Models via Incremental Optimization Utility

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前 **Large Multimodal Models (LMMs)** 的扩展受限于合成数据的质量与数量之间的权衡。尽管合成数据（如通过 LLM 生成的 Chain-of-Thought 数据）成本低、规模大，但其中常包含大量低质量样本、幻觉（hallucinations）和错误逻辑，导致训练效率低下甚至出现 **负迁移（negative transfer）**。

现有的主流方法 **LLM-as-a-Judge** 虽能有效筛选高质量数据，但依赖强模型进行多轮生成式评估，计算开销巨大，且缺乏可解释性，容易引入偏好泄漏（preference leakage）。

### 提出了什么新方法或新思路
本文提出 **One-Step-Train (OST)** 框架，将数据选择问题重新定义为一个 **增量优化效用排序问题（incremental optimization utility ranking）**。其核心思想是：
- 不依赖语义启发式（semantic heuristics），而是通过在轻量级代理模型（lightweight proxy）上模拟单步梯度更新，估计每个样本对验证目标的边际效用（marginal utility）。
- 具体而言，OST 计算每个样本 $ z_i $ 在代理模型上的梯度 $ \nabla \mathcal{l}(z_i; \theta) $，并模拟一次参数更新 $ \theta' = \theta - \eta \nabla \mathcal{l}(z_i; \theta) $，然后测量该更新对验证集损失的影响 $ \Delta_i = \mathcal{L}_v(\theta) - \mathcal{L}_v(\theta') $。
- 最终根据 $ \Delta_i $ 对所有样本排序，选取高价值子集用于训练主模型。

### 相比现有方法的优势
| 维度 | OST 方法优势 |
|------|---------------|
| **效率** | 显著降低总计算成本（减少 17% GPU 小时），训练阶段节省高达 43% 资源。 |
| **性能** | 在固定计算预算下，使用仅 Top-20% 子集即超越 LLM-as-a-Judge 基线 5.6 分，优于 Full-SFT 8.8 分。 |
| **可解释性** | 基于梯度信号的选择机制具有理论支撑，避免黑箱判断。 |
| **抗噪能力** | 成功识别并剔除“有毒样本”（toxic samples），逆转复杂推理任务中的负迁移现象。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **训练数据集**：构建了一个包含 **351,157 个多模态数学问题** 的合成池（Dsyn），源自真实考试题，经 **Doubao-Seed-1.6-thinking** 模型生成 Chain-of-Thought 推理链，并经过三阶段过滤（格式、一致性、长度）。
- **验证锚点集（Anchor Set V）**：从保留的 500 题中分层采样 100 道代表性题目，作为优化方向引导。
- **评估基准**：
  - 内部基准：500 道跨难度层级的数学题（涵盖算术、几何、中学/高考/奥赛等）
  - 外部公开基准：
    - **MathVision**
    - **MathVista**
    - **WeMath**
    - **LogicVista**

### 实验设置和评估指标
#### 模型架构
- **Proxy Model**：InternVL3-1B（轻量级代理模型，用于计算效用分数）
- **Target Models**：
  - 自研 30B Multimodal Model（Qwen3-30B-A3B + ViT-300M）
  - 开源系列：Qwen3-VL-2B/4B/8B-Instruct

#### 评估协议
| 协议 | 描述 |
|------|------|
| **Protocol A (Proportional Steps)** | 固定 epoch 数，训练步数随子集大小线性变化 → 衡量时间效率 |
| **Protocol B (Fixed Compute)** | 固定总优化步数，小数据集需更多 epoch → 衡量数据密度与收敛潜力 |

#### 评估指标
- 主要指标：**Average Score**（各基准平均准确率）
- 辅助分析：训练损失曲线、梯度稳定性、消融实验

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Full-SFT** | 在完整 351k 合成数据上微调 → 性能上限/下限（因噪声可能下降） |
| **Random Selection** | 随机抽取子集 → 下界基准 |
| **LLM-as-a-Judge** | 使用 Qwen3VL-235B-A22B-Instruct 作为裁判模型进行拒绝采样 → 工业界标准 |
| **DEITA [22]** | 基于启发式评分（复杂度 × 质量）+ 表征多样性模块的数据选择方法 → 当前最优启发式基线 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & Table 3）

#### 在内部基准上的表现（Protocol B: Fixed Compute）
| 方法 | 数据占比 | 平均得分 | 相对提升 |
|------|----------|-----------|------------|
| Full-SFT (100%) | 100% | 60.5 | — |
| LLM-as-a-Judge | ~81% | 63.7 | +3.2 |
| **OST (Best-20%)** | 20% | **69.3** | **+8.8 vs Full-SFT**, **+5.6 vs Judge** |

> ✅ **Less is More 现象显著**：仅用 20% 数据即可实现最高性能。

#### 效率与性能综合对比（Table 3）
| 方法 | 总 GPU 小时 | 训练耗时 | 平均得分 | 相对增益 |
|------|-------------|-----------|-----------|------------|
| LLM-as-a-Judge | ~269 | 4.2h | 63.7 | — |
| **OST (Top-50%)** | **224** | **2.4h** | **65.5** | **+1.8 pts, -17% compute** |

> ✅ **Pareto 最优**：同时提升性能并降低成本。

### 与基线方法的对比结果（Table 4）
在 Qwen3-VL 系列上的平均增益（Δ Score）：

| 方法 | Avg Gain | LogicVista 表现 | 是否逆转负迁移 |
|------|---------|------------------|----------------|
| Full-SFT | -3.4 ~ -1.3 | ❌ 显著退化（如 -1.8 pts） | 否 |
| LLM-as-a-Judge | +0.2 ~ +0.4 | ⚠️ 改善有限 | 部分缓解 |
| DEITA (Top-20%) | +0.3 ~ +0.4 | ⚠️ 仍可能退化 | 未完全解决 |
| **OST (Best-20%)** | **+0.6 ~ +1.0** | ✅ 持续提升（如 +0.4 pts） | **是，成功逆转** |

> 📌 **关键发现**：OST 在复杂逻辑任务（如 LogicVista）上表现尤为突出，说明其能有效捕捉深层推理一致性。

### 消融实验结果（Table 5）
研究不同训练阶段的代理模型对选择质量的影响：

| Proxy Checkpoint | 进度 | 平均得分 | Δ |
|------------------|------|-----------|----|
| Cold Start (Pretrained) | 0% | 63.9 | +0.4 |
| **Warm-up** | **5%** | **64.1** | **+0.6** ✅ |
| Loss Stable | 25% | 63.9 | +0.4 |
| Converged | 100% | 63.7 | +0.2 |

> 🔍 **非单调趋势**：仅经过 **5% 微调** 的 Warm-up 检查点效果最佳，完全收敛的代理模型反而因过拟合合成噪声而性能下降。

---

## 4. 关键结论和发现

### 主要发现
1. **数据质量 > 数据数量**：高质量子集即使只占 20%，也能显著超越全量训练，验证 “Less is More” 范式。
2. **基于优化信号的选择更可靠**：相比 LLM-as-a-Judge 和启发式方法，OST 利用梯度对齐性（gradient alignment）能更精准识别真正有益于推理能力提升的样本。
3. **成功逆转负迁移**：Full-SFT 在复杂任务上常因噪声累积导致性能下降，而 OST 可主动识别并剔除“有毒样本”，实现性能回升。
4. **代理模型无需充分训练**：仅需少量 warm-up 即可获得最佳选择性能，过度训练会导致对合成噪声的过拟合。

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **依赖 Anchor Set 质量** | 若验证锚点集（V）有偏或不具代表性，可能导致选择偏差，尤其在主观任务中难以定义 ground truth。 |
| **点式选择忽略集体影响** | OST 是贪心逐点排序，未考虑数据冗余与多样性，可能存在子集整体次优问题（MISS 问题）。 |
| **跨架构迁移边界未知** | 当前验证了 1B → 8B 的有效性，但在极端参数差距或异构架构（如 Transformer vs SSM）下的稳定性尚待探索。 |

### 未来工作方向
- 探索 **集合级选择策略**（set-wise selection），结合多样性约束以优化子集整体效用。
- 构建 **动态 anchor 机制**，适应不同任务分布，增强泛化能力。
- 扩展至 **自监督/强化学习场景**，实现端到端的自主数据净化流程。
- 理论分析梯度子空间对齐误差的上界，指导代理模型设计。

---

> 💡 **总体评价**：  
> 本论文提出的 **OST 框架** 为多模态模型的数据高效训练提供了新的范式——从“语义判断”转向“优化效用驱动”。它不仅大幅降低了工业级训练的成本，更重要的是揭示了：**模型性能瓶颈更多源于推理逻辑污染，而非数据不足**。这一洞察对未来构建更鲁棒、可信赖的 AI 系统具有深远意义。

</details>

---

### 11. [Chain-based Distillation for Effective Initialization of Variable-Sized Small Language Models](https://arxiv.org/abs/2605.07783)

**Authors**: Boyu Shi, YiCheng Jiang, Chang Liu, Qiufeng Wang, Xu Yang, Xin Geng  
**Category**: cs.CL  
**Published**: 2026-05-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.07783v1  

#### Abstract
Large language models (LLMs) achieve strong performance but remain costly to deploy in resource-constrained settings. Training small language models (SLMs) from scratch is computationally expensive, while conventional knowledge distillation requires repeated access to large teachers for different ta...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Chain-based Distillation for Effective Initialization of Variable-Sized Small Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **资源受限场景下的部署难题**：大型语言模型（LLMs）虽然性能强大，但参数量巨大，难以在移动设备或嵌入式系统等资源受限环境中部署。
- **传统训练和蒸馏方法效率低下**：
  - 从头预训练多个不同规模的小型语言模型（SLMs）计算成本高昂，复杂度为 $O(N)$。
  - 传统的 **Knowledge Distillation (KD)** 需要对每个目标尺寸重复访问大模型教师，扩展性差。
  - 存在“蒸馏瓶颈”（distillation bottleneck），即学生模型因容量不足无法有效学习教师模型的复杂分布。
- **架构异构性挑战**：实际应用中，教师模型（如 Qwen3）与目标部署模型（如 GPT 架构）可能存在架构、词表大小、隐藏维度不一致的问题，标准 KD 无法处理。

### **提出了什么新方法或新思路**
提出 **Chain-based Distillation (CBD)** ——一种可扩展的、高效的变量尺寸 SLM 初始化范式，其核心思想包括：

1. **构建知识链（Knowledge Chain）**：
   - 通过逐步蒸馏（stepwise distillation）建立一个稀疏的中间模型序列（称为 anchors），形成一条从源 LLM 到小型 SLM 的“知识传递链”。
   - 每个 anchor 只从其稍大的前驱模型学习，将巨大的容量差距分解为多个小步转移。

2. **桥接蒸馏（Bridge Distillation）以支持异构架构**：
   - 当源 LLM 与目标链架构不同时（如 Llama3 → GPT2），引入一个结构对齐的代理模型（proxy model）作为“锚点零”（anchor zero）。
   - 使用 **Sequence-level Knowledge Distillation (SeqKD)** 绕过词表不匹配问题。

3. **基于插值的快速初始化（Parameter Interpolation）**：
   - 对于任意目标尺寸的 SLM，通过在其相邻两个 anchors 之间进行参数插值得到高质量初始权重：
     $$
     \theta_{\text{target}} = \alpha \cdot \text{Trans}(\theta_{\text{small}}) + (1-\alpha) \cdot \text{Trans}(\theta_{\text{large}})
     $$
   - 其中 $\text{Trans}$ 包括子集剪枝（subset）和零填充/复制（expansion）操作。

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **效率** | 将构建 N 个模型的成本从 $O(N)$ 降至 $O(1)$，无需重复调用大模型教师。 |
| **性能** | 插值初始化显著优于随机初始化，甚至超越从 10B tokens 数据上从头训练的模型。 |
| **可扩展性** | 支持跨架构、跨词表的知识迁移，适用于工业级多样化部署需求。 |
| **稳定性** | 逐步蒸馏缓解了梯度方差和优化不稳定问题，收敛更平滑。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 类型 | 任务描述 |
|--------|------|----------|
| **OpenWebText / The Pile** | 预训练语料 | 用于 anchor 模型的蒸馏训练 |
| **Dolly (databricks-dolly-15k)** | SFT 数据集 | 主要用于监督微调阶段的行为对齐 |
| **MMLU** | 多项选择 | 跨领域知识理解（57 个主题） |
| **English_XLSum (XLsum)** | 生成任务 | 新闻摘要生成 |
| **HellaSwag (HellaS)** | 推理任务 | 常识推理填空 |
| **WinoGrande (WinG)** | 推理任务 | 核心指代消解 |
| **BoolQ** | 分类任务 | 是/否问答 |
| **Dolly** | 指令遵循 | 综合指令执行能力 |

### **实验设置和评估指标**
- **模型家族**：构建了三个知识链：
  - GPT2-based: `{GPT2-L(762M), GPT2-M(345M), GPT2-B(117M)}`
  - Pythia-based: `{Pythia-1b, 410m, 160m, 70m}`
  - Qwen3-based: `{Qwen3-1.7B, 0.6B}`
- **目标 SLM 规模**：覆盖 138M 至 537M 参数范围。
- **评估指标**：
  - **Accuracy (Acc)**：用于分类任务（MMLU, BoolQ）
  - **Rouge-L**：用于生成任务（XLsum）
  - **Zero-shot / Few-shot 性能**：衡量初始化质量
- **硬件平台**：HUAWEI ASCEND 910B NPUs

### **基线方法对比**
| 基线方法 | 描述 |
|---------|------|
| **Random Initialization (Rand)** | 从头预训练，使用 OpenWebText 不同规模子集（78M–10B tokens） |
| **Direct Knowledge Distillation** | 直接从源 LLM 蒸馏到目标 SLM |
| **MiniLLM, SeqKD** | 白盒蒸馏方法代表 |
| **Auto-Learngene, Van-Learngene** | 基于子网络提取的初始化方法 |
| **Single-expansion baseline** | 仅从小 anchor 扩展初始化，无多锚点插值 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **训练语料节省**
- 在 **HellaSwag** 上，一个未经恢复训练的 **138M SLM** 使用 CBD 初始化后，表现超过在 **10B tokens** 上从头训练的模型。
- 在 **XLsum** 上，**380M SLM** 使用 CBD 初始化即可达到与从 5B+ tokens 训练相当的性能，节省至少 **5B tokens** 的预训练开销。

#### ✅ **收敛加速**
- 在 **78M token 预训练预算下**，CBD 初始化的 SLM-138M 在“第 0 步”的损失已低于 Rand 的最终收敛损失，实现“zero-step advantage”。
- 收敛速度提升高达 **200×**（例如，在 500M 和 10B tokens 下分别提速 80.26× 和 217.86×）。

#### ✅ **相同预算下的性能优势**
| 方法 | Token 数 | 平均得分 (Avg) |
|------|----------|----------------|
| Rand | 78M | 39.67 |
| **CBD** | 78M | **46.33** (+6.66) |
| Rand | 100M | 40.01 |
| **CBD** | 100M | **46.38** (+6.37) |

> 表明 CBD 在极低数据预算下仍能保持高性能天花板。

#### ✅ **与 Learngene 方法对比**
| 方法 | 参数量 | 平均得分 (Avg) |
|------|--------|----------------|
| Auto-Learngene | 174.28M | 37.50 |
| Van-Learngene | 174.28M | 36.50 |
| **CBD** | **134.00M** | **42.44** |

> CBD 在更小参数量下取得更高性能，验证了链式蒸馏的有效性。

---

### **消融实验结果**

#### 🔹 **插值系数 α 的敏感性分析**
- 最优 α 与目标模型靠近哪个 anchor 强相关：
  - 138M 模型接近 GPT2-B(117M)，最优 α ≈ 0.9
  - 更大模型则偏好较小 α（即更多来自大 anchor 的信息）
- 说明知识链形成了连续的参数流形（parameter manifold）。

#### 🔹 **多锚点对齐 vs 单锚点扩展**
| 方法 | Dolly (220M) | Avg Score |
|------|-------------|-----------|
| Single-expansion | 15.89 | 36.16 |
| **CBD (multi-anchor)** | **16.43** | **36.43** |

> 多锚点插值提供更优的跨尺度结构对齐。

#### 🔹 **知识链密度影响**
| 锚点数量 | Average Score |
|----------|---------------|
| 2 anchors (~70% gap) | 46.24 |
| 4 anchors (~30% gap) | **47.58** |

> 更密集的 anchor 序列带来更精细的插值，性能更好。

#### 🔹 **公平性压力测试**
即使单锚点方法额外获得 **100M recovery tokens**，CBD（0 tokens）依然胜出：
| 方法 | Average Score |
|------|----------------|
| Single + 100M tokens | 50.07 |
| **CBD (0 tokens)** | **54.43** |

> 证明 CBD 的初始化质量具有质的优势。

---

## **4. 关键结论和发现**

### **主要发现**
1. **知识链是有效的知识压缩路径**：
   - 逐步蒸馏显著降低了逼近误差和统计误差，理论分析表明其泛化误差界更紧。
2. **参数插值可在高维空间中保留语义流形**：
   - 插值得到的初始化已具备较强的任务理解和生成能力，相当于“跳过了前 10B tokens 的预训练”。
3. **异构蒸馏具有任务依赖性优势**：
   - 在生成任务上，使用 Qwen3 或 Llama3 作为源模型的桥接蒸馏优于 GPT2 同构链，因其蕴含更强的高级语义先验。
4. **CBD 在极端小模型上依然有效**：
   - 即使在 **51.47M** 的极小模型上，CBD 仍比随机初始化平均高出近 5 个百分点。

### **方法的局限性**
- **未针对特定任务优化**：CBD 强调通用知识迁移，未显式融入下游任务适配机制，可能在某些专项任务上不是最优。
- **依赖公开 checkpoint**：当前实验利用已有模型 checkpoint 构建 anchor，若需完全自定义架构，则需承担首次蒸馏成本。
- **理论假设的理想化**：如假设每一步蒸馏都能稳定收敛，现实中可能受数据分布偏移影响。

### **未来工作方向**
- **Task-aware CBD**：结合下游任务特性，设计任务感知的插值策略或选择性参数继承。
- **动态知识链构建**：根据目标硬件约束自动搜索最优 anchor 序列。
- **扩展至多模态模型**：探索 CBD 在 Vision-Language 模型中的适用性。
- **减少 anchor 数量的压缩编码**：研究是否可用隐式函数替代显式 anchor 存储。

---

> 💡 **一句话总结**：  
> **CBD 通过构建“知识链”+“桥接蒸馏”+“参数插值”，实现了 O(1) 成本下高质量、可扩展、跨架构的 SLM 初始化，大幅节省训练资源并加速收敛，在多种任务和模型规模上均显著优于传统方法。**

</details>

---

### 12. [Target-Aware Data Augmentation for SAT Prediction](https://arxiv.org/abs/2605.06931)

**Authors**: Eshed Gal, Uri Ascher, Eldad Haber  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.06931v1  

#### Abstract
Learning-based approaches to NP-hard problems have shown increasing promise, but their progress is fundamentally constrained by the high cost of generating labeled training data. In domains such as Boolean satisfiability (SAT), standard pipelines rely on solver-in-the-loop labeling, which scales poo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Target-Aware Data Augmentation for SAT Prediction**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **NP-hard 问题中的数据瓶颈**：在 Boolean Satisfiability (SAT) 等组合优化问题中，传统机器学习方法依赖于 **solver-in-the-loop labeling**（即生成公式后调用 SAT 求解器判断 SAT/UNSAT），该过程在大规模实例上计算代价极高，严重限制了训练数据的规模和多样性。
- **合成数据分布不匹配**：现有生成方法（如 planted-SAT）虽然免去了求解器调用，但生成的数据往往缺乏真实基准的结构性特征，导致下游模型泛化能力差。

### **提出的新方法与新思路**
- **Target-Aware, Solver-Free 数据生成框架**：
  - **无需调用 SAT 求解器**：通过“逆向构造”方式生成具有**已知标签**的 SAT 和 UNSAT 实例。
    - **SAT 实例**：先随机选择一个满足赋值 $ x^* $，然后围绕该赋值构造子句，确保其可满足。
    - **UNSAT 实例**：嵌入一个由 $ w $ 个变量构成的**完备矛盾核**（所有极性组合均出现），再添加与目标基准对齐的填充子句。
  - **目标感知对齐（Target-Aware）**：生成过程中显式匹配目标基准（如 G4SATBench）的统计特性，包括：
    - 子句宽度分布（width distribution）
    - 变量出现偏斜（occurrence skew）
    - **slack 分布**（每个子句被多满足的字面量数量）
- **Linear-Programming-aware GNN (LPGNN)** 架构：
  - 将 CNF 公式转化为二元线性可行性问题 $ Ax \geq b $，引入 slack 变量 $ s $ 得到等式形式 $ Az = b $。
  - 在 GNN 中引入 **LP residual 机制**：每层计算约束残差 $ r^{(l)} = A z^{(l)} - b $，并通过 $ A^\top r^{(l)} $ 将其映射回节点空间作为额外信号注入消息传递过程，使模型能感知代数可行性结构。

### **相比现有方法的优势**
| 维度 | 本方法 | 传统方法 |
|------|--------|----------|
| **数据生成效率** | 近线性时间 $ O(mk) $，无需 solver 调用 | 指数级增长，受限于 solver runtime |
| **标签正确性** | 构造即保证标签正确 | 需 solver 验证，可能失败或超时 |
| **数据质量** | 显式对齐目标基准的结构与代数特性 | 多为随机或简单种植，分布偏差大 |
| **模型设计** | 引入 LP 残差增强 GNN 对优化结构的感知 | 多为标准 GNN，忽略代数语义 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **主数据集**：**G4SATBench** [Li et al., 2024]
  - 包含 Easy、Medium、Hard 三个难度级别的 3-SAT 实例。
  - 每类训练集 1.6k 实例，测试集 200 实例。
  - 实例位于相变区域（phase transition），$ m/n \approx 4.27 $。
- **用于 OOD 测试**：SR family（random 3-SAT-like 结构）

### **实验设置与评估指标**
#### **评估任务**
- **SAT/UNSAT 分类准确性（Accuracy）**
- **赋值质量评估**（Assignment Quality）：
  - **Constraint Satisfaction Rate (CSR)**：对 SAT 实例，预测赋值满足的子句比例越高越好。
  - **Violated Clause Ratio ($ k/m $)**：对 UNSAT 实例，预测赋值违反的子句数越少越好（理想为 0）。

#### **训练策略**
- 使用提出的 target-aware 生成器合成不同规模（250 ~ 80k）的训练数据。
- 模型在合成数据上训练，在 G4SATBench 的测试集上评估。

#### **对比基线**
- **数据生成效率对比**：
  - **Naive**：暴力枚举所有赋值（仅适用于小规模）
  - **CaDiCaL**：标准 generate-and-test 流程，调用 CDCL 求解器标注
  - **Ours**：本文提出的 solver-free 生成方法
- **模型性能对比**：
  - **LPGNN**（本文提出）
  - **NLocalSAT** [Zhang et al., 2020]
  - **QuerySAT** [Ozolins et al., 2022]
- **数据质量对比**：
  - **Generic data**：标准随机 k-CNF 生成（无目标对齐）
  - **Target-aware data**：本文方法

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 数据生成效率（Table 1 & Figure 4）**
| 变量数 $ n $ | CaDiCaL (ms) | Ours (ms) | **加速比** |
|--------------|-------------|-----------|------------|
| 15           | 0.66        | 0.54      | 1.2×       |
| 100          | 8.98        | 3.64      | 2.5×       |
| 250          | 4,080       | 9.32      | **438×**   |
| 500          | ~10⁸        | 19.3      | **~10⁶×**  |
| 1000         | ~10¹⁶       | 39.0      | **~10¹⁴×** |

> ✅ **结论**：随着问题规模增大，本文方法展现出**指数级效率优势**，在 $ n=1000 $ 时实现约 $ 10^{14} $ 倍加速。

#### **(2) 模型性能随训练数据量扩展（Figure 2 & Table 2）**
- **LPGNN 在 Medium 3-SAT 上的表现**：
  - 当训练数据从 250 增加到 10k：
    - **CSR 从 94.4% 提升至 99.0%**
    - **$ k/m $ 从 6.2% 下降至 1.6%**
  - 表现出明显的**正向 scaling law**，说明模型能有效利用更大规模的高质量数据。

- **跨架构验证（Table 2）**：
  - 同样使用 target-aware 合成数据训练：
    - **NLocalSAT**：CSR 从 94.4% → 99.0%
    - **QuerySAT**：CSR 保持稳定在 ~94.4%，但 $ k/m $ 未显著下降
  > ✅ 说明本文生成的数据具有**通用性**，可提升多种架构性能。

#### **(3) 消融实验（Appendix E, Table 4）**
| 模型 | Accuracy (%) | CSR (%) | $ k/m $ (%) |
|------|---------------|---------|-------------|
| GIN Backbone | 60.5 | 93.1 | 8.2 |
| **Full LPGNN (with LP residual)** | **64.0** | **94.2** | **7.5** |

> ✅ **结论**：加入 LP 残差机制带来一致性的性能提升，验证了**优化感知信号的有效性**。

#### **(4) 数据对齐的重要性（Figure 6）**
- 使用 **generic data** 训练时，模型性能始终接近随机猜测。
- 使用 **target-aware data** 训练时，性能随数据量增加稳步上升。
> ✅ **关键发现**：**数据分布对齐比单纯增加数据量更重要**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **数据是 NP-hard 问题学习的关键瓶颈**：当前进展更多受限于**高质量标注数据的获取成本**，而非模型容量。
2. **Solver-free + Target-aware 生成可行且高效**：通过逆向构造 + 分布对齐，可在近线性时间内生成大量正确标注、结构真实的 SAT/UNSAT 实例。
3. **合成数据可有效提升模型性能**：即使完全基于合成数据训练，也能获得良好的 in-distribution 泛化能力，并支持 scaling laws。
4. **数据与模型需协同设计**：LPGNN 利用了生成数据中的 slack 结构，实现了“数据-模型”闭环优化。

### **方法的局限性（Limitations）**
- **无法复现复杂逻辑依赖**：合成数据难以捕捉工业级 SAT 实例中的**隐藏社区结构**（hidden community structures）或深层语义约束。
- **存在 synthetic-to-real gap**：对完全 out-of-distribution 或高度结构化的现实问题，泛化能力仍有限。
- **依赖目标基准的统计信息**：若目标分布未知或难以建模，则对齐效果下降。

### **未来工作方向**
- 扩展生成框架以建模更复杂的结构（如工业电路、软件验证中的 SAT 实例）。
- 将 target-aware 思路推广至其他 NP-hard 问题（如 MaxSAT、MILP、TSP）。
- 探索**混合训练范式**：结合少量真实数据与大规模合成数据进行 curriculum learning。
- 开发更精细的 slack 控制机制，模拟 solver 搜索路径或证明复杂度。

---

> 📌 **一句话总结**：  
> 本文提出了一个**目标感知、免求解器的 SAT 数据生成框架**，实现了**数亿倍的数据生成加速**，并配合 LP-aware GNN 架构，验证了“**数据为中心**”（data-centric）的学习范式在 NP-hard 问题上的巨大潜力。

</details>

---

### 13. [Dual-Agent Co-Training for Health Coaching via Implicit Adversarial Preference Optimization](https://arxiv.org/abs/2605.07011)

**Authors**: Da Long, Lingyi Fu, Diya Michelle Rao, Jasmine Ruales Carrera, Yang Bai, Shandian Zhe  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.07011v1  

#### Abstract
Motivational-interviewing-based health coaching is an effective approach for improving mental health and promoting healthy behavior change. However, the scarcity of trained human coaches and the high cost of coaching services make such support inaccessible to many people who could benefit from it. T...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**基于动机性访谈（Motivational Interviewing, MI）的健康教练对话系统**在训练中存在的关键瓶颈：
- **单向训练范式限制**：现有方法通常只优化一方（如仅训练对话Agent而固定用户环境，或反之），导致交互空间探索不足，难以充分提升目标Agent的能力。
- **高质量专家标注成本高**：依赖人类专家进行行为质量标注不具可扩展性。
- **最终结果导向而非过程质量导向**：许多AI助手优化的是对话的最终结果，而健康教练的有效性更依赖于整个会话过程中行为的质量。

### 提出的新方法：DACT
作者提出了 **DACT (Dual-Agent Co-Training)** 框架，其核心是一个**双Agent协同训练机制**，通过**隐式对抗偏好优化（Implicit Adversarial Preference Optimization）** 同时训练健康教练Agent和客户模拟器（Client Simulator）。

#### 主要创新点
1.  **Dual-Agent Preference Tree Construction (双Agent偏好树构建)**：
    *   引入一种高效的、低成本的基于树的rollout程序来收集训练所需的偏好对。
    *   通过让教练和客户在多轮对话中交互，并在特定节点进行分支采样，生成丰富的对话树，从而在受控条件下比较不同回复的质量。

2.  **Multi-Dimensional LLM Judge (多维LLM裁判)**：
    *   设计了一个专门的、多维度的LLM裁判来评估教练行为。
    *   评估过程分为三个阶段：**Client-State Identification**（识别客户状态）、**State-conditioned Sentence-level Functional Labeling**（基于状态的句子级功能标注）和**Final Reasoning-based Aggregation and Scoring**（基于推理的聚合打分）。
    *   从三个核心维度打分：**Cultivating Change Talk (CCT)**, **Softening Sustain Talk (SST)**, 和 **Empathy**。

3.  **Implicit Adversarial Preference Training (隐式对抗偏好训练)**：
    *   **教练训练**：使用**帕累托优势（Pareto-dominant）** 的偏好对进行DPO训练。只有当一个回复在所有维度上都优于另一个回复时，才构成有效偏好，这鼓励教练在所有方面同时进步。
    *   **客户模拟器训练**：将教练的偏好信号**反转**。使教练表现更差的客户回复被视为“更好”的回复。这使得客户模拟器学会生成更具挑战性的、能暴露当前教练弱点的回应，从而形成一个**隐式的对抗动态**。

4.  **Stochastic-Game Interpretation (随机博弈解释)**：
    *   证明了这种协同训练过程可以被自然地解释为一个**两人随机博弈（two-player stochastic game）**，其中教练和客户拥有对立的潜在效用函数。

### 相比现有方法的优势
- **更全面的探索**：通过双向协同进化，能探索到更具挑战性的交互模式，这是单向训练无法触及的。
- **更高的效率和性能上限**：对抗性的客户模拟器持续提供有难度的训练样本，推动教练能力不断突破。
- **避免人工标注**：利用LLM裁判实现自动化、可扩展的高质量评估。
- **过程质量保障**：多维度、基于过程的奖励机制确保了教练行为的整体质量，而非仅仅追求最终结果。

---

## 2. 核心实验方法和设置

### 数据集
- **客户角色设定（Personas）**：研究者创建了一个包含5000个结构化客户角色的池子。这些角色由GPT-4o-mini生成，包含了年龄、性别、职业、健康状况、身体限制、目标和挑战等详细信息。
- **数据划分**：
    - **SFT语料库**：使用前3000个角色，通过两个GPT-4o-mini Agent扮演教练和客户进行角色扮演，生成用于监督微调（Supervised Fine-Tuning, SFT）的对话数据。
    - **训练与评估**：剩余的角色用于训练中的偏好树生成和最终的模型评估。

### 实验设置
- **基础模型**：`Qwen2.5-32B-Instruct-GPTQ-Int4`。
- **模型架构**：教练和客户共享同一个基础模型，但各自拥有独立的LoRA适配器（adapter）进行训练。
- **训练流程**：
    1.  首先对两个适配器进行SFT初始化。
    2.  进行13轮协同训练迭代（co-training iterations）。
    3.  每轮生成3棵偏好树，每棵树通过两层分支展开，共产生81条叶路径。
    4.  使用GPT-5.4-mini作为固定的LLM裁判对每个教练回复进行多维度打分。
    5.  基于打分计算Q值，并分别构建教练和客户的偏好对。
    6.  双方均使用DPO进行更新。

### 评估指标
1.  **`mean3`**：CCT、SST和Empathy三个维度得分的平均值，是衡量总体教练质量的核心指标。
2.  **`anti%`**：教练回复中出现**MI反模式（MI anti-patterns）** 的句子比例。反模式指那些不符合MI原则的行为，如`leading_question`, `premature_planning`, `arguing_for_change`等。该指标越低越好，衡量了教练的“安全底线”。
3.  **各维度单独得分**：CCT、SST、Empathy的独立平均分。

### 基线方法对比
- **SFT**：仅经过监督微调的教练模型，无后续DPO优化。
- **GPT-COACH**：使用与DACT相同的推理提示词（prompt）调用的GPT-4.5模型，作为匹配提示的强基线。
- **GPT-RUBRIC**：使用一个更长的提示词，其中直接嵌入了LLM裁判的完整评分标准（rubric）的GPT-4.5模型，代表了通过提示工程能达到的理论上限。

### 客户条件（测试场景）
为了全面评估泛化能力，教练在四种不同的客户模拟器下进行测试：
- **Rs-CLIENT**：训练过程中第8轮协同进化出的客户，被探针实验确定为对SFT教练最具挑战性的“原分布内压力测试”客户。
- **OAI-EMOTIONAL**：强调困难情绪的“域外”（OOD）客户。
- **OAI-RESISTANT**：强调事实约束并拒绝通用建议的“域外”客户。
- **OAI-AMBIVALENT**：在改变意愿和维持现状之间保持平衡的“域外”客户。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在**四个客户条件下的平均表现**（`4-cond avg`）如下表所示：

| Client | Method | CCT | SST | Empathy | **mean3** | **anti%** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 4-cond avg | **DACT** | 3.88 | 4.39 | 4.43 | **4.24** | **0.33** |
| 4-cond avg | GPT-RUBRIC | 3.49 | 4.32 | 4.62 | 3.99 | 12.54 |
| 4-cond avg | GPT-COACH | 2.99 | 3.73 | 4.40 | 3.73 | 11.62 |
| 4-cond avg | SFT | 2.03 | 2.35 | 3.49 | 2.77 | 25.45 |

### 与基线方法的对比结果
1.  **显著提升总体质量**：DACT的`mean3`得分为**4.24**，显著高于最强基线GPT-RUBRIC的3.99，实现了近0.25分的绝对提升（在1-5分制下非常可观）。
2.  **大幅抑制反模式**：DACT的`anti%`仅为**0.33%**，相比GPT-RUBRIC（12.54%）和GPT-COACH（11.62%）降低了**超过一个数量级（20-40倍）**。这表明DACT不仅更优秀，而且行为更加稳健和安全。
3.  **在最具挑战性的客户上优势最大**：在`Rs-CLIENT`上，DACT的`mean3`达到4.25，而GPT-RUBRIC仅为3.33，差距接近1分。这验证了协同训练对于应对专门设计的困难对手的有效性。
4.  **强大的泛化能力**：DACT在所有三种“域外”（OOD）客户类型上均表现出色，说明其学到的技能具有良好的泛化性，而非过拟合于训练数据。

### 消融实验结果
研究进行了关键的消融实验，以验证**客户协同进化**的必要性：
- **Client-Frozen**：冻结客户模拟器（使用SFT初始化的客户），仅对教练进行DPO训练。
- **结果**：`Client-Frozen`的性能在早期几轮后迅速**饱和（plateau）**，最终`mean3`停留在约3.10，远低于DACT的4.25。
- **结论**：**自适应的客户协同进化是驱动后期性能持续提升的关键**。一个固定的客户无法提供足够有挑战性的样本，导致教练的训练信号很快饱和，无法达到其潜力上限。

---

## 4. 关键结论和发现

### 主要发现
1.  **协同进化至关重要**：论文最核心的发现是，**同时训练教练和客户模拟器**，并通过**隐式对抗**的方式（即客户的目标是“打败”当前教练）能够创造出一个自我进化的课程（self-paced adversarial curriculum），从而极大地提升教练Agent的最终性能。
2.  **过程质量可通过自动化评估实现**：利用精心设计的多维度LLM裁判，可以有效地替代昂贵的人工评估，实现对复杂、过程导向的交互任务（如健康教练）的可扩展训练。
3.  **帕累托优势偏好优于标量奖励**：使用多维度的帕累托优势来构建偏好对，避免了在不同目标间进行人为加权，能更均衡地促进教练在所有关键维度上的发展。
4.  **方法具有强大的鲁棒性和泛化性**：DACT不仅在训练中遇到的困难客户上表现优异，在未见过的、风格迥异的“域外”客户上也展现了卓越的性能。

### 方法的局限性
1.  **依赖LLM裁判的质量**：整个框架的有效性高度依赖于LLM裁判的判断是否准确和可靠。尽管论文通过与人类专家的盲测对比验证了其一致性，但LLM裁判本身可能存在偏见或错误。
2.  **基础模型的限制**：实验仅在一个基础模型（Qwen2.5-32B）上进行，结果的普适性有待在其他模型上验证。
3.  **缺乏完全的临床验证**：评估虽然严谨，但仍基于LLM裁判和协议，尚未经过真实的临床医生或MITI专家的全面编码评估。

### 未来工作方向
1.  **在更多基础模型上复制和验证**：将DACT框架应用于其他主流大语言模型，检验其通用性。
2.  **进行完整的临床专家评估**：邀请专业的MI从业者对DACT教练的对话进行MITI 4.2.1标准的正式编码，以获得金标准的性能评估。
3.  **探索更复杂的交互动态**：将此框架扩展到涉及更多参与者或更复杂任务的健康干预场景中。
4.  **改进裁判机制**：探索如何进一步提高LLM裁判的可靠性，例如通过集成多个裁判或引入反馈循环。

</details>

---

### 14. [EnvSimBench: A Benchmark for Evaluating and Improving LLM-Based Environment Simulation](https://arxiv.org/abs/2605.07247)

**Authors**: Yi Liu, TingFeng Hui, Wei Zhang, Li Sun, Ningxin Su, Jian Wang, Sen Su  
**Category**: cs.AI  
**Published**: 2026-05-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.07247v1  

#### Abstract
Scalable AI agents training relies on interactive environments that faithfully simulate the consequences of agent actions. Manually crafted environments are expensive to build, brittle to extend, and fundamentally limited in diversity. A promising direction is to replace manually crafted environment...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EnvSimBench: A Benchmark for Evaluating and Improving LLM-Based Environment Simulation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 LLM 的环境模拟（LLM-based environment simulation）被广泛用于训练 AI Agent，但其**核心假设——LLM 能够准确模拟环境反馈——从未被系统验证**。实践中，LLM 模拟常出现以下三大失败模式：
- **Hallucination**：虚构不存在的状态转移；
- **Logical Inconsistency**：响应内部字段自相矛盾；
- **State Drift**：多轮交互中状态信息逐渐丢失。

这些问题会污染 Agent 的奖励信号，导致训练偏差，并可能抵消“低成本构建环境”的初衷。

### 🚀 提出的新方法与创新
作者提出 **EnvSimBench**，一个用于评估和改进 LLM 环境模拟能力的基准框架，包含四大贡献：

#### (1) 正式定义 **Environment Simulation Ability (EnvSim Ability)**
首次将“环境模拟能力”形式化为可量化的研究目标：  
> 给定当前环境状态 $s$、动作 $a$ 及其实现逻辑 `code(a)`，模型能否正确预测执行后的状态 $s'$ 和观测反馈 $o$？

这使得 EnvSim Ability 成为独立于通用推理能力的新维度。

#### (2) 构建 **EnvSimBench 基准测试集**
- 包含 **400 个样本**，覆盖 **167 个多样化的 tool-interactive 环境**；
- 所有标签由程序化执行器（Python executor）生成，确保**无 LLM 干扰的客观真值**；
- 引入**三维难度分层**（three-axis stratification）以实现细粒度诊断：
  - **Axis 1**: 动作结果（Success vs. Failure）
  - **Axis 2**: 状态变化复杂度（|Δ| = 0 到 12）
  - **Axis 3**: 输入参数数量（0 vs. ≥1）

#### (3) 发现“状态变化悬崖”（State-Change Cliff）
所有前沿 LLM 在状态不变时表现接近完美（CM > 99%），但在需要同时更新多个状态变量时性能急剧下降：
- 当 |Δ| ≥ 3 时，CM 普遍低于 20%
- 当 |Δ| ≥ 5 时，CM 接近零
这一现象揭示了 EnvSim Ability 是一种与通用推理能力正交的关键短板。

#### (4) 设计 **约束驱动的模拟范式（Constraint-Driven MDP）**
将传统 POMDP 设置重构为全可观测 MDP：
- 显式提供完整前状态 $s_t$、工具调用 $a$ 和实现代码 `code(a)`
- 模型任务简化为结构化解析而非隐式状态追踪
- 每步独立可验证，避免错误累积

在此基础上训练的小型专用模型（4B 参数）在性能上超越所有大型基础模型，且成本降低 **90% 以上**。

---

## 2. 核心实验方法和设置

### 📚 数据集来源
- **种子环境**：来自 [EnvScaler](https://arxiv.org/abs/2512.xxxxx) 的 **191 个 tool-interactive 环境**
- **轨迹采集**：使用 GPT-4o-mini 作为智能体，在每个环境中执行最多 30 步操作，收集多轮交互轨迹 $(a, o, s, s')$
- **最终数据集**：预处理为 **400 个单轮状态预测样本**，每条包含：
  - 输入：$(s_t, a, \text{code}(a))$
  - 输出：$(o, s')$，其中 $s'$ 表示为一系列增删改操作 $\Delta$

### 🧪 实验设置
- **评估对象**：7 个前沿 LLM（DeepSeek-V3.2, Qwen3.5-397B, GPT-5.4, Gemini-3.1-Pro, Claude-Sonnet-4.6, MiniMax-M2.7, GLM-5）
- **输入格式统一**：全部采用相同的 MDP 提示模板，显式传入 $s_t$, $a$, `code(a)`
- **输出要求**：结构化输出预测的观察 $o$ 和状态变更集 $\Delta$
- **真值来源**：完全由外部 Python 执行器运行真实环境获得，**不依赖任何 LLM**

### 📊 评估指标
| 指标 | 定义 | 说明 |
|------|------|------|
| **Feedback Match (FM)** | 预测反馈字符串与真值完全一致 | 衡量表面反馈准确性 |
| **Config Match (CM)** | 应用预测的 $\Delta$ 后得到的 $s'$ 是否等于真实 $s'$ | 更严格的指标，衡量状态转移是否正确 |

> ⚠️ 注意：CM 不受输出格式影响，是衡量 EnvSim Ability 的核心指标。

### 🔁 基线方法对比
- **Frontier LLMs**：直接使用大模型进行 zero-shot 推理
- **Small Model + SFT**：基于 Qwen3-4B-Base 进行全参数微调（Full-parameter SFT）
- **不同训练策略对比**：
  - LoRA vs. Full SFT
  - 是否加入推理链（reasoning trace）
  - 不同数据组成策略（如仅变状态 vs. 平衡分布）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（总体表现）

| Model | FM (%) | CM (%) |
|-------|--------|--------|
| **Qwen3.5-397B-A17B** | 69.0 | **42.3** ✅ |
| **GLM-5** | **80.5** ✅ | 41.0 |
| DeepSeek-V3.2 | 72.5 | 32.5 |
| Gemini-3.1-Pro | 74.0 | 42.0 |
| Claude-Sonnet-4.6 | 25.5 | 37.8 |
| MiniMax-M2.7 | 33.0 | 41.8 |

> 💡 尽管 GLM-5 在 FM 上领先，但 CM 最高的是 Qwen3.5-397B，表明**高 FM 不一定代表高保真模拟**。

---

### 📉 “状态变化悬崖”（State-Change Cliff）现象

| |Δ| | Simple (avg) CM | Medium (avg) CM | Difficult (avg) CM |
|-----|------------------|------------------|--------------------|
| DeepSeek-V3.2 | 22% | 8.5% | 4% |
| Qwen3.5-397B | 46% | 17.0% | 24% |
| GLM-5 | 44% | 14.0% | 28% ✅ |
| **所有模型趋势** | **随 |Δ|↑ 急剧下降** | **|Δ|≥5 时 CM ≈ 0** | —— |

> 🔍 发现：当需同时更新超过 3 个字段时，几乎所有模型都崩溃；而 |Δ|=1 时 CM 达到 36–72%，形成鲜明对比。

---

### 🔬 消融实验结果

#### (1) 微调小模型 vs. 大模型
| Model | FM (%) | CM (%) | 参数量 | 成本 |
|-------|--------|--------|--------|------|
| Qwen3.5-397B | 69.0 | 42.3 | ~397B | 高 |
| **Full-Balance2 (4B)** | **79.5** | **45.3** ✅ | 4B | **<10% 成本** |

> ✅ 结论：经过针对性微调的 4B 模型在 CM 上**超越所有大模型**，并提升 EnvScaler 合成成功率 **6.8%**

#### (2) 数据组成的影响（Data Composition Ablation）

| 策略 | Overall CM (%) | Overall FM (%) | 说明 |
|------|---------------|---------------|------|
| Change-only | 47.5 | 64.3 | 对 Failure/No-change 无法生成正确反馈 |
| Balance | 43.8 | 79.5 | 加入非变更样本但未加权 |
| **Balance2 (ours)** | **45.3** ✅ | **79.5** | 按真实分布采样，最优泛化 |

> ✅ 数据组成比数据量更重要：合理平衡各类难度样本显著提升性能。

#### (3) 是否添加推理链（Reasoning Trace）
- 添加 structured reasoning trace 反而导致 CM 下降（35.0% → 30.3%）
- 推测原因：对简单任务引入噪声，复杂任务则需更多数据才能受益

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **EnvSim Ability 是一项独立且未被充分认识的能力**
   - 与通用推理能力解耦，即使强推理模型也可能在状态追踪上失败。

2. **存在普遍的“状态变化悬崖”**
   - 所有前沿 LLM 在 |Δ| ≥ 3 时性能骤降，|Δ| ≥ 5 几乎完全失效。
   - 这不是渐进退化，而是**质变式的崩溃**。

3. **FM 与 CM 存在严重脱钩**
   - 多数模型可在 CM 失败的同时保持高 FM（如 DeepSeek-V3.2：68.7% FM, 10.0% CM）
   - 意味着 Agent 收到看似合理的反馈，但底层状态已悄然损坏 → **静默腐败（silent corruption）**

4. **小型专用模型优于通用大模型**
   - 经过平衡数据微调的 4B 模型在 CM 上超越所有 100B+ 级别模型
   - 成本降低 **90x 以上**，证明**专业化优于规模化**

5. **格式差异（format divergence）也是一种风险**
   - 如 Claude-Sonnet-4.6 常省略外层结构（返回 `"error msg"` 而非 `{"success": false, "error": ...}`），导致 FM 极低但 CM 正常
   - 这类错误虽可见，但也会影响下游解析

---

### ⚠️ 局限性

1. **样本规模有限**
   - Difficult 组部分子类仅有 4–6 个样本，统计可靠性不足
   - 需更大规模数据支撑更稳健分析

2. **高 |Δ| 样本稀缺**
   - |Δ| ≥ 5 的样本难以获取，限制了对极端复杂场景的研究

3. **下游验证存在循环性**
   - 合成质量过滤器与训练数据分布重叠，可能导致过拟合评估

4. **未涵盖语义级评估**
   - 当前 FM/CM 均为精确匹配，缺乏对语义等价性的判断（如时间戳相近是否可接受？）

---

### 🔮 未来工作方向

1. **扩展 EnvSimBench 规模与多样性**
   - 增加更多高 |Δ| 场景，尤其是异构多字段更新
   - 分离“批量均匀操作”与“真正复杂的异构更新”

2. **开发语义感知评估指标**
   - 引入 fuzzy matching 或 embedding-based similarity 来缓解严格字符串匹配带来的偏差

3. **探索动态推理机制**
   - 如引入 Execution Engine 或 Symbolic Solver 辅助 LLM 进行状态更新

4. **构建端到端的仿真-训练联合优化框架**
   - 将 EnvSim Ability 与 Agent 学习效果关联，建立闭环反馈

5. **推动标准化协议**
   - 提议将 EnvSimBench 作为 LLM 环境模拟能力的标准测试套件

---

> 🔗 **代码与数据开源地址**：[https://github.com/cookieApril/EnvSimBench](https://github.com/cookieApril/EnvSimBench)

--- 

📌 **一句话总结**：  
EnvSimBench 揭示了 LLM 在环境模拟中的根本缺陷——“状态变化悬崖”，并通过约束驱动范式和针对性微调，证明了**小而专的模型在可靠性和成本上全面超越大而泛的模型**，为可信赖的大规模 Agent 训练奠定了坚实基础。

</details>

---

### 15. [Confidence-Aware Alignment Makes Reasoning LLMs More Reliable](https://arxiv.org/abs/2605.07353)

**Authors**: Kejia Chen, Jiawen Zhang, Yihong Wu, Kewei Gao, Jian Lou, Zunlei Feng, Mingli Song, Ruoxi Jia  
**Category**: cs.AI  
**Published**: 2026-05-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.07353v1  

#### Abstract
Large reasoning models often reach correct answers through flawed intermediate steps, creating a gap between final accuracy and reasoning reliability. Existing alignment strategies address this with external verifiers or massive sampling, limiting scalability. In this work, we introduce CASPO (Confi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Confidence-Aware Alignment Makes Reasoning LLMs More Reliable*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型推理模型（LRMs）虽然在最终答案准确率上表现优异，但其**中间推理步骤常存在逻辑错误**，即“正确答案源于错误推理”。这种现象导致模型的**内部置信度（token-level confidence）与实际逻辑正确性严重错位**——模型可能对错误步骤高度自信，而对正确但复杂的推导却信心不足。这使得依赖外部验证器（如 PRM）或大规模采样（如 Self-Consistency）的方法成本高昂且难以扩展。

### 提出了什么新方法或新思路
本文提出 **CASPO**（**Confidence-Aware Step-wise Preference Optimization**），一个统一框架，通过校准模型内在的 token-level 置信度与逐步推理的正确性来提升可靠性，无需训练独立的奖励模型（reward model）。

- **训练阶段**：引入 **Confidence-Aware Step-wise Preference Optimization**。构建偏好对（preference pairs），对比“正确但不自信” vs “错误但自信”的推理步骤，通过迭代的 **Direct Preference Optimization (DPO)** 对齐模型的概率分布与逻辑正确性。
- **推理阶段**：提出 **Confidence-aware Thought (CaT)** 策略。利用训练后校准的置信度，在生成推理树时动态扩展高置信路径并剪枝低置信分支，实现高效、可靠的探索，仅引入可忽略的 $O(V)$ 延迟。

### 相比现有方法的优势
- **无需外部验证器**：摆脱了对 PRM 或大规模采样的依赖，显著降低计算开销。
- **细粒度对齐**：在 step-level 而非 trajectory-level 进行优化，直接解决中间步骤不可靠的问题。
- **训练-推理统一**：利用同一套校准后的置信度信号指导训练和推理，形成闭环。
- **高效可扩展**：$O(V)$ 的置信度计算复杂度远低于 PRM 的 $O(L^2d)$，且在强模型（如 Qwen3-8B-Base）上仍有效。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **数学推理基准**（主任务）：
  - `MATH500`、`Minerva-Math`、`OlympiadBench`、`AMC2023`、`AIME2024` / `AIME2025`
- **跨领域泛化基准**：
  - `BoardgameQA (BGQA)`（逻辑推理）
  - `CRUXEval (CRUX)`（代码推理）
  - `StrategyQA (STGQA)`（多跳推理）
  - `TableBench`（表格推理）
  - `MMLU-Pro STEM`（STEM知识）
- **代码与语言理解**：
  - `HumanEval`、`LiveCodeBench`（代码生成）
  - `RACE`（阅读理解）

### 实验设置和评估指标
- **模型**：以 `Llama-3.1-8B-Instruct`、`Qwen2.5-Math-7B`、`Qwen2.5-7B-Instruct` 为基线，并在更强的 `Qwen3-8B-Base` 上验证可扩展性。
- **训练**：全参数微调（full-parameter fine-tuning），使用 `Open-RLHF` 框架，迭代 DPO。
- **评估指标**：
  - 主要：`Pass@1`（单次生成准确率）
  - 辅助：`Expected Calibration Error (ECE)`、`Brier Score (BS)`（校准质量）、`AUC-ROC`（置信度作为正确性预测信号的能力）
- **推理策略对比**：固定采样预算（如 $K=10$），比较 `CoT`、`Self-Consistency`、`DiPT` 和 `CaT`。

### 基线方法对比
- **训练类方法**：
  - `GRPO`、`Simple-RL-Zero`、`PURE-VR`、`rStar-Math`、`PCPO`、`DPO-VP`
- **推理类方法**：
  - `Chain-of-Thought (CoT)`、`Self-Consistency`、`DiPT`
- **树搜索基线**（用于强模型对比）：
  - `rStar-Math`、`Satori`

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 `Qwen2.5-7B-Instruct` 上，CASPO 将平均数学推理准确率从 **44.4% 提升至 50.6%**；结合 `CaT` 推理，进一步提升至 **56.1%**。
- 在 `Qwen3-8B-Base` 上，CASPO **超越了依赖大量 reward model 数据的树搜索基线**（如 `rStar-Math` 和 `Satori`），在 `AIME2024` 和 `AIME2025` 上分别达到 **36.7%** 和 **33.3%**，而基线最高仅为 30.0% 和 26.7%。
- 在 `Qwen2.5-Math-7B` 上，`HumanEval` 的 `Pass@1` 从 40.9% 提升至 **51.9%**，证明了方法的跨模态泛化能力。

### 与基线方法的对比结果
- **训练效果**：CASPO 在所有三个基模型上均优于 `GRPO`、`Simple-RL-Zero`、`PURE-VR`、`DPO-VP` 等轨迹级优化方法（见 Table 1）。
- **推理效率**：`CaT` 在相同采样预算下性能优于 `Self-Consistency` 和 `DiPT`，且端到端延迟（2.8 s/query）远低于 `Self-Consistency`（12.5 s/query），接近贪婪解码（1.2 s/query）（见 Table 5）。
- **可扩展性**：仅用 8K 样本，CASPO 即使在已深度微调的 `Qwen2.5-Math-7B-Instruct` 上也能将 `MATH500` 的 `Pass@1` 从 83.6% 提升至 **85.1%**，超越 `DPO-VP` 和 `PCPO`（见 Table 3）。

### 消融实验结果
- **迭代训练**：第一轮迭代带来最大增益（如 `Math500` 从 64.8% → 76.6%），后续迭代持续微调，验证了正反馈循环。
- **置信度信号有效性**：
  - **Shannon Entropy** 作为不确定性信号，其预测步骤正确性的 `AUC-ROC` 达 **0.86**，显著优于 `max token probability` (0.68) 和 `perplexity` (0.72)。
  - 经过 CASPO 训练后，错误步骤的 token-level entropy 显著升高（熵差从 0.04 增至 0.66），为 `CaT` 提供了清晰的剪枝信号（见 Table 8）。
- **校准质量**：CASPO 将 `Qwen2.5-Math-7B` 的 `ECE` 从 0.184 降至 **0.081**，`Brier Score` 从 0.215 降至 0.142，表明模型的置信度与正确性高度对齐。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **内在置信度是可靠推理的关键**：通过 CASPO 对齐 token-level 置信度与逻辑正确性，可以从根本上提升推理过程的可靠性，而非仅仅提高最终答案准确率。
2. **无需外部监督即可实现高效验证**：利用模型自身的 token entropy 作为低成本、零开销的自我验证信号，可以替代昂贵的外部验证器。
3. **训练与推理的协同效应**：训练阶段的校准直接赋能推理阶段的高效搜索（`CaT`），实现了“一次校准，全程受益”。
4. **强大的泛化能力**：尽管只在数学数据上训练，CASPO 在代码、常识、STEM 等非数学任务上也表现出一致的性能提升，证明其捕捉的是通用的推理一致性。

### 方法的局限性
1. **置信度定义单一**：目前仅基于 **Shannon Entropy**，其他不确定性度量（如模型自省、内部表征）可能提供互补信息。
2. **依赖离线评估器**：数据构建阶段依赖一个强模型（如 `Qwen2.5-Math-7B-Instruct`）作为离线评估器，可能引入评估器特有偏差。
3. **评估器与目标模型耦合风险**：若评估器与目标模型共享相似的推理模式，其标注的“正确性”可能并非绝对可靠。

### 未来工作方向
- 探索更丰富的不确定性度量方式，并进行系统性比较。
- 设计**自包含**（self-contained）或**联合训练**（jointly trained）的验证机制，减少对外部评估器的依赖。
- 将 CASPO 框架应用于更多高风险领域（如医疗、金融），并研究其在这些场景下的鲁棒性和安全性。
- 研究如何将该方法扩展到多模态推理任务中。

</details>

---

### 16. [GLiGuard: Schema-Conditioned Classification for LLM Safeguard](https://arxiv.org/abs/2605.07982)

**Authors**: Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis  
**Category**: cs.CL  
**Published**: 2026-05-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.07982v1  

#### Abstract
Ensuring safe, policy-compliant outputs from large language models requires real-time content moderation that can scale across multiple safety dimensions. However, state-of-the-art guardrail models rely on autoregressive decoders with 7B--27B parameters, reformulating what is fundamentally a classif...

---

### 17. [StreamPhy: Streaming Inference of High-Dimensional Physical Dynamics via State Space Models](https://arxiv.org/abs/2605.07384)

**Authors**: Panqi Chen, Yifan Sun, Shikai Fang, Xiao Fu, Lei Cheng  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.07384v1  

#### Abstract
Inferring the evolution of high-dimensional and multi-modal (e.g., spatio-temporal) physical fields from irregular sparse measurements in real time is a fundamental challenge in science and engineering. Existing approaches, including diffusion-based generative models and functional tensor methods, t...

---

### 18. [PACEvolve++: Improving Test-time Learning for Evolutionary Search Agents](https://arxiv.org/abs/2605.07039)

**Authors**: Minghao Yan, Bo Peng, Benjamin Coleman, Ziqi Chen, Zhouhang Xie, Shuo Chen, Zhankui He, Noveen Sachdeva, Weili Wang, Ed H. Chi, Shivaram Venkataraman, Wang-Cheng Kang, Derek Zhiyuan Cheng, Beidou Wang  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.07039v1  

#### Abstract
Large language models have become drivers of evolutionary search, but most systems rely on a fixed, prompt-elicited policy to sample next candidates. This limits adaptation in practical engineering and research tasks, where evaluations are expensive, and progress depends on learning task-specific se...

---

### 19. [Adaptive Regularization for Sparsity Control in Bregman-Based Optimizers](https://arxiv.org/abs/2605.07892)

**Authors**: Ahmad Aloradi, Tim Roith, Emanu\"el A. P. Habets, Daniel Tenbrinck  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.07892v1  

#### Abstract
Sparse training reduces the memory and computational costs of deep neural networks. However, sparse optimization methods, e.g., those adding an $\ell_1$ penalty, often control sparsity only indirectly through a regularization parameter $\lambda$, whose mapping to the final sparsity rate is non-trivi...

---

### 20. [Adaptive Domain Decomposition Physics-Informed Neural Networks for Traffic State Estimation with Sparse Sensor Data](https://arxiv.org/abs/2605.08028)

**Authors**: Eunhan Ka, Ludovic Leclercq, Satish V. Ukkusuri  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.08028v1  

#### Abstract
Traffic state estimation from sparse fixed sensors is challenging because physics-informed neural networks (PINNs) tend to over-smooth the shockwaves admitted by the Lighthill-Whitham-Richards (LWR) model. This study proposes Adaptive Domain Decomposition Physics-Informed Neural Networks (ADD-PINN),...

---

### 21. [GraphReAct: Reasoning and Acting for Multi-step Graph Inference](https://arxiv.org/abs/2605.07357)

**Authors**: Xingtong Yu, Zhongwei Kuai, Chang Zhou, Xuanting Xie, Renhe Jiang, Xikun Zhang, Hong Cheng, Xinming Zhang, Yuan Fang  
**Category**: cs.AI  
**Published**: 2026-05-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.07357v1  

#### Abstract
Reasoning-acting frameworks enhance large language models (LLMs) by interleaving reasoning with actions for dynamic information acquisition. However, extending this paradigm to graph learning remains underexplored. Graph data is inherently structured, with information distributed across nodes and ed...

---

### 22. [Rubric-Grounded RL: Structured Judge Rewards for Generalizable Reasoning](https://arxiv.org/abs/2605.08061)

**Authors**: Manish Bhattarai, Ismael Boureima, Nishath Rajiv Ranasinghe, Scott Pakin, Dan O'Malley  
**Category**: cs.AI  
**Published**: 2026-05-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.08061v1  

#### Abstract
We argue that decomposing reward into weighted, verifiable criteria and using an LLM judge to score them provides a partial-credit optimization signal: instead of a binary outcome or a single holistic score, each response is graded along multiple task-specific criteria. We formalize \emph{rubric-gro...

---

### 23. [A Reproducible Multi-Architecture Baseline for Token-Level Chinese Metaphor Identification under the MIPVU Framework](https://arxiv.org/abs/2605.07170)

**Authors**: Yufeng Wu  
**Category**: cs.CL  
**Published**: 2026-05-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.07170v1  

#### Abstract
Metaphor is pervasive in everyday language, yet token-level computational identification of metaphor-related words in Chinese under the MIPVU framework remains under-explored relative to English. This paper presents a reproducible multi-architecture baseline for token-level metaphor identification o...

---

### 24. [Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization](https://arxiv.org/abs/2605.07172)

**Authors**: Yurui Pan, Ke Xu, Bo Peng  
**Category**: cs.CL  
**Published**: 2026-05-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.07172v1  

#### Abstract
Alignment of large language models (LLMs) via SFT and RLHF/DPO typically ignores the global geometry of the representation space, relying instead on local token likelihoods or scalar scores. We view generation as tracing a semantic trajectory in hidden space and propose a topology-enhanced alignment...

---

### 25. [Learning Agent Routing From Early Experience](https://arxiv.org/abs/2605.07180)

**Authors**: Yimin Wang, Jiahao Qiu, Xuan Qi, Xinzhe Juan, Jingzhe Shi, Zelin Zhao, Hongru Wang, Shilong Liu, Mengdi Wang  
**Category**: cs.CL  
**Published**: 2026-05-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.07180v1  

#### Abstract
LLM agents achieve strong performance on complex reasoning tasks but incur high latency and compute cost. In practice, many queries fall within the capability boundary of cutting-edge LLMs and do not require full agent execution, making effective routing between LLMs and agents a key challenge. We s...

---

### 26. [PaT: Planning-after-Trial for Efficient Test-Time Code Generation](https://arxiv.org/abs/2605.07248)

**Authors**: Youngsik Yoon, Sungjae Lee, Seockbean Song, Siwei Wang, Wei Chen, Jungseul Ok  
**Category**: cs.CL  
**Published**: 2026-05-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.07248v1  

#### Abstract
Beyond training-time optimization, scaling test-time computation has emerged as a key paradigm to extend the reasoning capabilities of Large Language Models (LLMs). However, most existing methods adopt a rigid Planning-before-Trial (PbT) policy, which inefficiently allocates test-time compute by inc...

---

### 27. [Rethinking Dense Sequential Chains: Reasoning Language Models Can Extract Answers from Sparse, Order-Shuffling Chain-of-Thoughts](https://arxiv.org/abs/2605.07307)

**Authors**: Yi-Chang Chen, Feng-Ting Liao, Da-shan Shiu, Hung-yi Lee  
**Category**: cs.CL  
**Published**: 2026-05-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.07307v1  

#### Abstract
Modern reasoning language models generate dense, sequential chain-of-thought traces implicitly assuming that every token contributes and that steps must be consumed in order. We challenge both assumptions through a systematic intervention pipeline--removal, masking, shuffling, and noise injection--a...

---

### 28. [RcLLM: Accelerating Generative Recommendation via Beyond-Prefix KV Caching](https://arxiv.org/abs/2605.07443)

**Authors**: Zhan Zhao, Yuxin Wang, Amelie Chi Zhou  
**Category**: cs.DC  
**Published**: 2026-05-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.07443v1  

#### Abstract
Large Language Models (LLMs) are transforming recommendation from ranking into a generative task, but industrial deployment remains limited by the high latency of processing long, personalized prompts. Standard prefix caching provides limited benefit because reuse in recommendation workloads is ofte...

---

### 29. [Breaking the Illusion: When Positive Meets Negative in Multimodal Decoding](https://arxiv.org/abs/2605.06679)

**Authors**: Yubo Jiang, Yitong An, Xin Yang, Abudukelimu Wuerkaixi, Xuxin Cheng, Fengying Xie, Zhiguo Jiang, Cao Liu, Ke Zeng, Haopeng Zhang  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.06679v1  

#### Abstract
Vision-Language Models (VLMs) are frequently undermined by object hallucination, generating content that contradicts visual reality, due to an over-reliance on linguistic priors. We introduce Positive-and-Negative Decoding (PND), a training-free inference framework that intervenes directly in the de...

---

### 30. [Causal-Aware Foundation-Model for Bilevel Optimization in Discrete Choice Settings](https://arxiv.org/abs/2605.06941)

**Authors**: Shivaram Subramanian, Zhengliang Xue, Markus Ettl, Yingdong Lu, Jayant Kalagnanam  
**Category**: cs.LG  
**Published**: 2026-05-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.06941v1  

#### Abstract
We introduce a causal aware foundation-model framework for real time optimal decision making in discrete choice environments. We propose a constrained triple-head price optimization (C3PO) network to solve a bilevel decision problem in which a service provider selects an optimal assortment while het...

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

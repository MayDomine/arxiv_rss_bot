# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-12 08:17:59 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [ATLAS: Efficient Out-of-Core Inference for Billion-Scale Graph Neural Networks](https://arxiv.org/abs/2605.09402)

**Authors**: Pranjal Naman, Yogesh Simmhan  
**Category**: cs.DC  
**Published**: 2026-05-12  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.09402v1  

#### Abstract
Graph Neural Network (GNN) inference on billion-scale graphs is critical for domains like fintech and recommendation systems. Full-graph inference on these large graphs can be challenging due to high communication costs in distributed settings and high I/O costs in disk-backed Out-of-Core (OOC) sett...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ATLAS: Efficient Out-of-Core Inference for Billion-Scale Graph Neural Networks

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Out-of-Core (OOC)** GNN 系统（如 Ginex、DGI）主要针对训练任务设计，在执行 **full-graph inference** 时面临严重挑战：
- **读放大（Read Amplification）**：gather-based 执行模式导致对顶点特征的重复、随机磁盘访问。
- **内存压力大**：全图推理需为所有顶点计算嵌入，中间状态远超单机内存容量。
- **I/O 效率低下**：不规则的数据访问模式无法充分利用 SSD 的高带宽。

这些问题使得现有系统在十亿级图上进行推理时性能极差，甚至无法完成。

---

### 提出的新方法与核心创新
作者提出 **ATLAS**，一个基于广播（broadcast-based）的 OOC GNN 推理框架，其核心思想是将传统的“拉取”（gather）模式转变为“推送”（broadcast）模式。

#### 主要贡献：
1. ✅ **Broadcast-based Inference Model**
   - 将每层推理重构为从源节点向出边广播消息的过程。
   - 实现 **顺序、单次扫描式读取** 特征和嵌入，显著减少读放大。

2. ✅ **Tiered Memory-Disk Runtime**
   - 设计分层内存-磁盘架构（hot store + cold store），管理部分聚合状态。
   - 引入 **minimum-pending-messages eviction policy**，优先驱逐接近完成的顶点，降低重载频率。

3. ✅ **Pipelined, Overlapped Execution**
   - 构建完整流水线：拓扑/特征流式读取 → CPU 聚合 → GPU 变换 → 异步写回。
   - 支持多种 GNN 层（GCN, GIN, GraphSAGE）并可配置内存预算。

4. ✅ **Graph Reordering Heuristic**
   - 提出一种贪心排序策略，最大化每一步的消息完成增益，提升早期顶点完成率，缓解内存压力。

---

### 相比现有方法的优势
| 维度 | 现有方法（Ginex/DGI） | ATLAS |
|------|------------------------|-------|
| 数据访问模式 | 随机、多次读取（gather） | 顺序、单次读取（broadcast） |
| I/O 放大 | 高（与边数成正比） | 极低（接近最优） |
| 内存管理 | 缓存无感知或简单 LRU | 智能驱逐策略（min-pending-msg） |
| 吞吐量 | I/O 密集型瓶颈明显 | 计算密集型，高效利用资源 |

> 💡 **本质突破**：ATLAS 将 OOC 推理从 “input-unaware gather” 转变为 “output-aware broadcast”，从根本上优化了数据移动效率。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验采用四个大规模公开图数据集，涵盖不同规模与特征维度：

| 数据集 | Abbr. | 顶点数 | 边数 | 特征大小（GiB） | 类别数 |
|--------|-------|--------|--------|------------------|--------|
| OGBN-Papers100M | PA | 111M | 1.7B | 54 (FP32) | 172 |
| MAG240M-Cites | MA | 121M | 1.4B | 175 (FP16) | 153 |
| IGB-Large | IL | 100M | 1.2B | 200 (FP16) | 19 |
| IGB-Full | IF | 269M | 4B | 550 (FP16) | 19 |

> 🔍 所有数据集的特征总量均超过工作站内存（128 GiB RAM + 32 GiB GPU memory）

---

### 实验设置
- **硬件平台**：
  - CPU: AMD Ryzen 9 9900X (12-core)
  - RAM: 128 GiB
  - GPU: NVIDIA RTX 5090 (32 GiB)
  - 存储: 2 TiB Samsung 990 PRO NVMe SSD
- **软件环境**：Ubuntu 24.04, PyTorch 2.8, NumPy v2.0
- **清除缓存**：每次运行前清空 OS page cache，确保公平比较。

---

### 评估指标
- **End-to-end Inference Time**（总推理时间）
- **Total Bytes Read from Disk**（磁盘读取总量）
- **CPU/GPU Utilization**, **Memory Footprint**, **SSD Bandwidth**
- **Ablation Studies**：分析排序、驱逐策略、hot store 大小的影响

---

### 基线方法对比
| 基线 | 类型 | 描述 |
|------|------|------|
| **Ginex [24]** | vertex-wise training framework adapted for inference | 使用 superbatch 和 neighbor cache，原为训练设计 |
| **DGI [44]** | layer-wise OOC inference baseline | 支持动态批处理和图重排序（MK ordering），使用 mmap |

> ⚠️ 注意：两者均为 SOTA OOC 方法，但未专为 full-graph inference 优化。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 数据集 | GNN 模型 | ATLAS 推理时间 | DGI 推理时间（外推） | 加速比 |
|--------|----------|----------------|------------------------|--------|
| MA (175 GiB) | SAGEConv | < 1h (两层) | ~11h (仅第一层) | **~12–30×** |
| IF (550 GiB) | GCN | ~3h | ~117h (L1 外推) | **~39×** |
| PA (54 GiB) | 多种模型 | ≈ DGI | ≈ DGI | **within ~5%** |

> ✅ 即使在内存内场景（PA），ATLAS 性能也几乎持平；而在 OOC 场景下实现数量级加速。

---

### 与基线方法的对比结果
- **I/O 流量大幅下降**：
  - ATLAS 的磁盘读取量比 DGI 减少 **1–2 个数量级**。
  - 例如在 IF 上，DGI 第一层外推读取达 **262 TiB**，而 ATLAS 两层合计仅 **3.6 TiB**。
- **Ginex 表现更差**：
  - 平均读取数据量高出 ATLAS **11–16×**。
  - 在最大图 IF 上直接崩溃（runtime error）。

---

### 消融实验结果

#### （1）图重排序（Graph Reordering）
- 使用 ATLAS 自研排序 vs 原始/随机顺序：
  - **重载时间减少 3.3× (IL) 和 3× (MA)**
  - **平均每 chunk 重载比例从 ~7% 降至 ~1.1%**
  - **端到端时间缩短 1.5–1.8×**

> 📌 结论：合理排序能显著提升顶点完成速度，降低 hot store 压力。

#### （2）驱逐策略（Eviction Policy）
比较三种策略（50 GiB hot store）：
| 策略 | Reload Time (IL) | Eviction Time (IL) | 总时间 |
|------|------------------|--------------------|--------|
| Random | 200s | 130s | 30 min |
| LRU | 220s | 140s | 30 min |
| **Min-Pending-Messages (ATLAS)** | **120s** | **50s** | **20 min** |

> 📌 结论：ATLAS 的驱逐策略有效减少 thrashing，尤其优于 LRU。

#### （3）Hot Store 内存敏感性
- 当 hot store ≥ 70 GiB（IL）或 ≥ 60 GiB（MA）时，**eviction/reload 基本归零**。
- 此后继续增加内存对性能无显著提升。
- 在 60–80 GiB 内存下即可稳定支持 ~200 GiB 特征集。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Gather-based 是 OOC 推理的根本瓶颈**：不规则、重复的磁盘访问导致严重的 I/O 放大。
2. ✅ **Broadcast-based 可实现近乎最优的 I/O 效率**：通过单次顺序读取打破传统限制。
3. ✅ **智能内存管理至关重要**：结合图重排序与最小待收消息驱逐策略，可极大降低冷热存储切换开销。
4. ✅ **ATLAS 在真实十亿级图上具备实用性**：最大图（IF, 550 GiB）可在 **3 小时内完成推理**，适合离线批量更新。

---

### 方法的局限性
- ❌ **目前仅支持 message-passing GNNs**（如 GCN, GIN, SAGE），尚未扩展至 GAT 等 attention-based 模型。
- ❌ **不支持增量推理（incremental inference）**，仍需全图重算（尽管可用于构建 baseline）。
- ❌ 图重排序是一次性预处理步骤，在动态图中难以频繁应用。

---

### 未来工作方向
- 🔮 扩展至 **attention-based layers**（如 GAT）的 broadcast 执行模式。
- 🔮 支持 **incremental inference** 工作负载，结合 Ripple++ 等已有工作。
- 🔮 探索 **multi-GPU 或轻量分布式版本**，进一步提升吞吐。
- 🔮 将 ATLAS 的设计思想反哺到 **OOC training** 中，统一推理/训练优化路径。

---

> ✅ **总体评价**：ATLAS 是首个专为 **十亿级图全图推理** 设计的高效 OOC 框架，通过根本性的执行范式转变（gather → broadcast），实现了 **12–30× 的端到端加速**，并在多个 SOTA 基线失败的情况下成功运行最大规模图（550 GiB）。该工作为低成本、高性能的大规模 GNN 部署提供了新的可行路径。

</details>

---

### 2. [Different Prompts, Different Ranks: Prompt-aware Dynamic Rank Selection for SVD-based LLM Compression](https://arxiv.org/abs/2605.08568)

**Authors**: Hengyi Zhu, Zhendong Mi, Grace Li Zhang, Shaoyi Huang  
**Category**: cs.LG  
**Published**: 2026-05-12  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.08568v1  

#### Abstract
Large language models (LLMs) have rapidly grown in scale, creating substantial memory and computational costs that hinder efficient deployment. Singular value decomposition (SVD) has emerged as an effective post-training compression technique, but existing SVD-based methods rely on static rank trunc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Different Prompts, Different Ranks: Prompt-aware Dynamic Rank Selection for SVD-based LLM Compression**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的基于 **SVD** 的 LLM 压缩方法采用**静态秩截断（static rank truncation）**，即在所有输入上应用相同的固定秩子集。这种方法存在两个关键缺陷：
- **输入无关性（Input-agnostic rank allocation）**：不同 prompt 对模型行为的影响不同，所需的最优秩也应动态变化，但静态方法无法适应。
- **校准域偏差（Calibration-domain mismatch）**：压缩时使用的校准数据集（calibration set）决定了保留哪些奇异值，导致在分布外任务上性能下降。

### **提出的新方法：PARSE**
作者提出了 **PARSE**（**P**rompt-**A**ware **R**ank **S**election as **E**xperts），一种**后训练（post-training）框架**，用于在 SVD 压缩的 LLM 中实现动态秩选择。

#### **核心思想**
- 将每个 SVD 分解后的权重矩阵视为一组独立的 **rank experts**（秩专家）。
- 引入一个离线训练的**线性路由器（linear router）**，根据输入 prompt 动态选择最合适的秩子集。
- 路由器通过监督原始 dense model 的输出进行训练，从而**解耦于校准数据集**。

#### **关键技术设计**
- **Prompt-aware selection**：不同 prompt 可激活不同的 rank 子集。
- **Rank pattern caching**：利用语义相似的 prompt 共享相似的 rank 选择模式，在预填充阶段通过缓存检索避免在线路由。
- **Rank reuse**：生成过程中的解码步长间 rank 选择高度稳定，因此可在整个生成过程中复用初始选择。
- **系统级优化**：
  - **Expert memory aggregation**：将选中的 rank 组件聚合到连续内存块中，提升 GPU 内存合并效率。
  - **Kernel fusion**：融合共享输入的 MatMul 操作，减少内核启动开销。

### **相比现有方法的优势**
- **更优性能**：在多种 SVD 基线上一致提升模型质量（PPL 和 zero-shot accuracy）。
- **更强泛化性**：对校准数据集不敏感，跨任务鲁棒性强。
- **更高效率**：推理延迟显著降低，最高达 **2.5× prefill** 和 **2.4× decode** 加速。
- **正交性**：可无缝集成到任何 SVD-based 压缩流程中，无需修改原有压缩算法。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 类型 | 数据集 |
|------|--------|
| **语言建模评估** | WikiText-2, PTB, C4 |
| **零样本推理评估** | OpenBookQA, ARC-e/c, WinoGrande, HellaSwag, PIQA, MathQA |
| **路由器训练** | C4（大规模多样化语料） |
| **校准数据集分析** | WikiText-2, PTB, C4, MathQA（用于测试校准偏差） |

### **实验设置**
- **模型**：LLaMA-7B, LLaMA-13B, LLaMA-30B, Qwen2.5-7B
- **压缩比（Compression Ratio）**：0.2, 0.4, 0.6
- **评估方式**：
  - **PPL（Perplexity）**：越低越好
  - **Zero-shot accuracy**：越高越好
  - **延迟（Latency）**：prefill 和 decode 阶段每 token 的毫秒数
- **硬件**：NVIDIA RTX A6000 GPUs
- **实现细节**：
  - 路由器为单层线性门控网络（single linear gating layer）
  - 使用 **AdamW** 优化器，学习率 $2\times10^{-4}$，训练 5 轮，约需 6 GPU 小时

### **基线方法对比**
PARSE 并非独立压缩方法，而是构建在以下 SVD-based 压缩方法之上：
- **SVD-LLM**
- **Dobi-SVD**
- **Basis Sharing**
- **SAES-SVD**
- （额外对比）FWSVD, ASVD

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **在 LLaMA-7B 上的表现（压缩比 0.6）**
| 方法 | Avg. Zero-shot Accuracy | C4 PPL |
|------|--------------------------|--------|
| **SAES-SVD** | 0.34 | 93.97 |
| **SAES-SVD + PARSE** | **0.44** | **81.46** |
| **提升** | **+10%** | **↓13.3%** |

> ✅ **平均准确率提升高达 10%**

#### **推理速度（LLaMA-7B, batch size=64, compression ratio=0.6）**
| 阶段 | Native SVD | PARSE | 加速比 |
|------|------------|-------|--------|
| **Prefill** | ~10,000 ms | **4.1 s** | **~2.5×** |
| **Decode (per token)** | ~150 ms | **62.7 ms** | **~2.4×** |

> ✅ **显著优于原生 SVD 执行，并超越 vLLM 的 dense 推理**

#### **在更大/更新模型上的泛化能力（compression ratio=0.2）**
| 模型 | 方法 | WikiText-2 PPL | Avg. Accuracy |
|------|------|----------------|-------------|
| LLaMA-13B | SAES-SVD | 6.34 | 0.51 |
| LLaMA-13B | SAES-SVD + PARSE | **6.05** | **0.55** |
| LLaMA-30B | SAES-SVD + PARSE | **5.21** | **0.59** |
| Qwen2.5-7B | SAES-SVD + PARSE | **7.83** | **0.61** |

> ✅ 在多种架构和规模上均有效，证明其通用性。

---

### **消融实验结果**

#### **(1) Rank Retrieval 与 Reuse 的影响**
| 方法 | Wiki2 PPL | Acc | Prefill Latency (ms) | Decode Latency (ms) |
|------|-----------|-----|------------------------|----------------------|
| SVD-LLM | 7.94 | 0.44 | 217.08 | 25.56 |
| +PARSE (Router only) | **7.16** | **0.52** | 516.49 | 68.14 |
| +Rank Retrieval | 7.43 | 0.50 | **121.64** | 68.19 |
| +Rank Reuse | 7.16 | 0.49 | 517.61 | **20.78** |
| **+PARSE (Full)** | 7.43 | 0.48 | **120.72** | **19.06** |

> 🔍 **Rank retrieval 显著降低 prefill 开销，reuse 极大优化 decode 延迟，综合实现最佳效率-质量平衡。**

#### **(2) 系统级优化效果**
| 优化配置 | Prefill Latency (64-batch) | Decode Latency |
|---------|----------------------------|---------------|
| 无优化 | 13.2 s | 119 ms |
| +Memory Aggregation | 8.4 s | — |
| **+Full (Agg + Fusion)** | **6.9 s** | **85.0 ms** |

> ✅ 内存聚合和内核融合共同带来显著端到端加速。

#### **(3) 路由器训练数据的影响**
| 训练数据 | Wiki2 PPL | PTB PPL | C4 PPL | Avg Acc |
|--------|-----------|---------|--------|---------|
| WikiText-2 | 7.05 | 14.09 | 13.21 | 0.50 |
| PTB | 7.12 | 14.97 | 13.87 | 0.48 |
| **C4** | **7.01** | **13.96** | **12.84** | **0.51** |

> ✅ 使用**多样化语料（如 C4）** 训练路由器能获得最佳泛化性能。

#### **(4) 对校准数据的鲁棒性**
| SVD 校准数据 | Wiki2 PPL | PTB PPL | C4 PPL | Acc |
|-------------|-----------|---------|--------|-----|
| WikiText-2 | 6.98 | 14.07 | 13.04 | 0.51 |
| PTB | 7.03 | 14.09 | 13.11 | 0.50 |
| C4 | 7.01 | 13.96 | 12.84 | 0.51 |
| MathQA | 7.05 | 14.10 | 13.13 | 0.50 |

> ✅ PARSE 对 SVD 校准数据不敏感，性能稳定，验证了其**解耦校准信息**的能力。

#### **(5) 在 Vanilla SVD 上的有效性**
| 方法 | Wiki2 PPL |
|------|-----------|
| Vanilla SVD | 20222.35 |
| Vanilla SVD + PARSE | **301.27** |

> ✅ 即使没有高级 SVD 技术（如 whitening, error compensation），PARSE 仍能极大恢复模型性能，说明其**自身有效性极强**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **静态秩截断是次优的**：最优秩因 prompt 而异，统一截断会导致局部性能崩溃（如 PPL 尖峰）。
2. **校准数据严重影响泛化**：静态方法严重依赖校准集统计特性，跨域表现差。
3. **PARSE 有效缓解上述问题**：
   - 通过 prompt-aware 路由实现动态秩选择。
   - 利用缓存和重用机制消除在线开销。
   - 系统级优化进一步提升效率。
4. **简单而高效**：单层线性路由器已足够，无需复杂结构。
5. **广泛兼容**：可集成于多种 SVD 方法，提升其性能与鲁棒性。

### **方法的局限性**
- 当前仅在 **decoder-only LLMs** 上验证，未覆盖 encoder-decoder 或多模态模型。
- 依赖 prompt-level 的 rank pattern 稳定性，若某些任务中 rank 选择波动剧烈，可能失效。
- 缓存机制引入额外的**离线预处理和存储成本**。
- 性能加速基于特定硬件（RTX A6000），在其他 backend 上可能有所差异。

### **未来工作方向**
- 扩展至 **instruction-tuned models** 和 **production systems**。
- 探索更轻量化的路由器架构或自适应缓存策略。
- 结合量化、剪枝等其他压缩技术，构建联合优化框架。
- 研究 rank selection 在长文本生成、思维链（reasoning chain）中的动态演化规律。

---

> 📌 **总结一句话**：  
> **PARSE 通过“prompt-aware + cache + system optimization”三位一体的设计，在不解耦原有 SVD 流程的前提下，实现了更智能、更高效、更鲁棒的 LLM 压缩，为动态低秩近似提供了新范式。**

</details>

---

### 3. [Generalization Bounds of Emergent Communications for Agentic AI Networking](https://arxiv.org/abs/2605.08613)

**Authors**: Yong Xiao, Jingxuan Chai, Guangming Shi, Ping Zhang  
**Category**: cs.AI  
**Published**: 2026-05-12  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.08613v1  

#### Abstract
The evolution of 6G networking toward agentic AI networking (AgentNet) systems requires a shift from traditional data pipelines to task-aware, agentic AI-native communication solutions. Emergent communication, a novel communication paradigm in which autonomous agents learn their own signaling protoc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Generalization Bounds of Emergent Communications for Agentic AI Networking》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统通信网络依赖于**预定义、任务无关（task-agnostic）的固定协议**，难以适应 Agentic AI Networking（AgentNet）系统中异构智能体之间动态、多模态、任务驱动的协作需求。现有 **emergent communication** 方法虽然能学习自适应通信协议，但普遍存在以下问题：
- 忽视物理层约束（如带宽、计算复杂度）
- 缺乏严格的**信息论基础**
- 多为启发式设计，缺乏对**泛化能力的理论保证**

本文旨在解决上述挑战，提出一个**信息论可解释、资源受限下高效且具备泛化保障的 emergent communication 框架**。

---

### **提出了什么新方法或新思路**
作者提出了一种基于 **Distributed Information Bottleneck (DIB)** 理论的新型联合优化框架，其核心创新包括：

#### ✅ **1. 联合损失函数设计**
引入一个新的**joint loss function**，将决策函数（decision-making functions）与通信信号学习（emergent communication signaling）统一在一个优化目标中：
$$
\mathcal{L} = \sum_k \left[ \mathcal{L}_k(\theta_k, \phi_k) - \lambda_i I(Y_k; C_{-k,k}) + \lambda_c I(S_k; C_{k,-k}) \right]
$$
其中：
- $I(Y_k; C_{-k,k})$：任务相关性信息（最大化）
- $I(S_k; C_{k,-k})$：表示复杂度（最小化，等价于 Minimum Description Length, MDL）

该设计实现了：
- **任务感知通信**：只传递对完成任务最关键的信息
- **去冗余传输**：抑制环境噪声和无关状态传播

#### ✅ **2. 基于 DIB 的理论建模**
首次将 **multi-agent multi-task DIB 理论**应用于 AgentNet 场景，形式化地刻画了：
- 任务相关信息保留 vs. 通信/计算开销之间的根本权衡
- 为 emergent communication 提供了**数学可分析的基础**

#### ✅ **3. 泛化误差上界推导**
通过 Rényi divergence 和 sub-Gaussian 假设，推导出在**去中心化推理场景下**的泛化误差上界：
$$
\epsilon_{gen}(w_k) \leq \sqrt{\frac{2\sigma^2}{n} D_k \cdot M_k}
$$
其中：
- $D_k$：后验与先验分布间的 Rényi divergence，反映协议复杂度
- $M_k$：统计学习难度（受样本数 $n$ 和方差 $\sigma^2$ 影响）

👉 这是**首个针对 emergent communication 协议的理论泛化边界分析**，提供了鲁棒性和稳定性保障。

---

### **相比现有方法的优势**
| 维度 | 本文方法 | 现有方法（如 EC-SOTA） |
|------|--------|------------------|
| **架构设计** | 联合训练决策与通信模块 | 模块化分离训练（如 Autoencoder + 决策器） |
| **理论支撑** | 有严格信息论基础（DIB + 泛化界） | 多为经验性、启发式设计 |
| **资源效率** | 显式控制 MDL，降低通信负载 | 不考虑复杂度正则化 |
| **泛化能力** | 理论可证、实验证明更强 | 容易过拟合训练数据 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **应用层数据**：采用公开的真实智能手机流量数据集 [[15]](https://ieeexplore.ieee.org/document/10231234)，涵盖五类主流移动应用：
  - Live Streaming
  - Video Conferencing
  - Mobile Gaming
  - Web Browsing
  - Social Media
- 数据包含真实时间序列的行为模式，用于模拟应用层 agent 的观测输入。

### **实验设置**
- 构建了一个基于开源 RAN 和软件化 5G Core 的硬件原型平台，包含：
  - User Equipments (UEs)
  - gNodeB (gNB)
  - 5G Core (5GC)
- 部署两类 agent：
  - **Application-layer agent**：观察应用流量并预测需求
  - **Physical-layer agent**：接收信号后动态调整无线资源配置（如调制编码、功率分配）
- 两 agent 通过 emergent communication 协议进行协调，实现跨层优化。

### **评估指标**
1. **Accuracy**：应用层 agent 对任务目标的识别准确率
2. **Generalization Error**：训练误差与推理误差之差（越小越好）
3. **Convergence Speed**：达到稳定性能所需的迭代次数
4. **Error Floor**：收敛后的最低误差水平

### **基线方法对比**
- **EC-SOTA** [16]：当前最先进的基准方法
  - 使用独立的 autoencoder 学习 latent 表示
  - 通信模块与决策模块分别训练
  - 无显式的复杂度或任务相关性正则项

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
来自图2和图3的结果显示：

#### 🔹 图2：应用层 agent 准确率随迭代变化
- 在所有应用类型中，本文方法始终优于 EC-SOTA
- 最终准确率提升约 **8–12%**
- 收敛速度更快，在 ~6000 次迭代即趋于稳定

#### 🔹 图3：泛化误差比较
| 方法 | 泛化误差峰值 | 推理阶段误差 | 收敛速度 |
|------|-------------|------------|---------|
| EC-SOTA | >18%       | ~14%      | 缓慢，波动大 |
| **本文方法** | <10%        | **<5%**     | 快速下降，稳定 |

- 特别是在高带宽、低延迟敏感的应用（如 Live Streaming 和 Mobile Gaming）中，优势更显著
- 本文方法的训练-推理 gap 更窄，说明**更强的泛化能力**

#### 🔹 图3(c)：综合对比
- 本文方法不仅最终性能更高，而且在整个训练过程中保持更低的泛化误差
- 错误下限（error floor）明显低于 EC-SOTA，表明协议更具鲁棒性

### **消融实验结果（隐含分析）**
尽管未明确列出消融实验表格，但从理论分析和损失函数结构可推断：
- 若移除 $I(Y_k; C_{-k,k})$ 项 → 任务相关性下降 → 泛化误差上升（见 Remark 2）
- 若移除 $I(S_k; C_{k,-k})$（MDL 正则）→ 表示复杂度升高 → 易发生“informational collapse”和过拟合（见 Remark 1）
- 联合优化机制有效缓解了 multi-agent setting 中的 non-stationarity 问题

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **联合优化优于模块化设计**  
   将 decision-making 与 communication signaling 融合在一个 DIB 框架中，显著提升了通信效率与任务性能。

2. ✅ **DIB 提供有效的信息压缩机制**  
   成功提取“state differences”而非完整状态，减少了冗余通信，提高了带宽利用率。

3. ✅ **理论泛化界具有指导意义**  
   推导出的 generalization bound 可解释实际性能差异，并为协议设计提供原则性指引（如控制 $D_k$, $M_k$）。

4. ✅ **真实硬件验证有效性**  
   在基于 5G Open RAN 的原型系统上验证，证明该框架适用于现实网络环境。

---

### **方法的局限性**
- 当前模型假设 reward signal $Y_k$ 在训练时可用（虽支持离线/在线两种模式），但在完全无监督或稀疏奖励场景下的扩展性有待研究。
- 多 agent 间通信拓扑固定，未考虑动态连接或部分可观图结构。
- 理论分析依赖 sub-Gaussian 和独立同分布（i.i.d.）假设，在高度非平稳环境中可能需进一步放松。

---

### **未来工作方向**
1. 扩展至 **large-scale multi-agent systems**，研究可扩展的分布式训练机制
2. 结合 **LLMs 或 generative agents**，探索语义级 emergent communication
3. 引入 **causal representation learning** 以增强协议的可解释性与迁移能力
4. 探索 **zero-shot generalization** 到未见过的任务类型或网络拓扑
5. 将本框架集成进 **6G native AI architecture**，作为 agentic protocol 自治演化的基础组件

--- 

> 📌 **一句话总结**：  
> 本文提出了首个基于 **DIB 理论**并具备**理论泛化保证**的 emergent communication 框架，通过联合优化决策与通信，在真实 5G 硬件平台上实现了更高效、更鲁棒、更好泛化的多智能体协作，为 **6G AgentNet** 的发展奠定了坚实基础。

</details>

---

### 4. [PhysEDA: Physics-Aware Learning Framework for Efficient EDA With Manhattan Distance Decay](https://arxiv.org/abs/2605.10547)

**Authors**: Zetao Yang  
**Category**: cs.LG  
**Published**: 2026-05-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.10547v1  

#### Abstract
Electronic design automation (EDA) addresses placement, routing, timing analysis, and power-integrity verification for integrated circuits. Learning methods -- attention (Transformer) and reinforcement learning (RL) -- have recently emerged on EDA tasks, yet face two common bottlenecks: vanilla atte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PhysEDA: Physics-Aware Learning Framework for Efficient EDA With Manhattan Distance Decay

---

## 1. 论文的主要贡献和创新点

### 解决的问题
电子设计自动化（**EDA**）任务在近年来引入了基于学习的方法（如 **Transformer** 和 **Reinforcement Learning, RL**），但仍面临三大瓶颈：
1. **计算复杂度高**：标准 **softmax attention** 具有 $O(L^2)$ 的二次复杂度，难以扩展到大规模芯片设计（如 $L > 10,000$ 候选位置）。
2. **数据稀缺导致过拟合**：芯片设计数据天然稀少（如公开的 DPP 数据集仅 2,000 个实例），模型容易从噪声中学习虚假的长程相关性，违背物理规律。
3. **强化学习奖励稀疏**：在决策过程中（如电容放置），只有最终状态才有非零奖励，中间步骤无监督信号，导致训练缓慢且不稳定。

### 提出的新方法
作者提出 **PhysEDA**，一个将物理先验统一集成到模型架构和训练过程中的框架，包含两个核心组件：

#### （1）Physics-Structured Linear Attention (**PSLA**)
- **思想**：利用 EDA 任务中普遍存在的物理规律——**电气与布线交互随曼哈顿距离（Manhattan distance）指数衰减**，即 $ \exp(-\alpha \cdot d_M) $。
- **实现**：将该衰减核作为**可学习的乘法偏置（multiplicative bias）** 融入 **linear attention** 的注意力核中。
- **优势**：
  - 复杂度从 $O(L^2d)$ 降至 $O(Ld^2)$，实现线性扩展。
  - 无需显式建模相对位置，自然具备空间归纳偏置。
  - 支持跨尺度迁移（scale-invariant）。

#### （2）Potential-Based Reward Shaping (**PBRS**)
- **思想**：利用相同的 $ \exp(-\alpha \cdot d_M) $ 构造一个**物理势能函数（physical potential）**，用于奖励塑形。
- **实现**：在 RL 中，将原始奖励 $R$ 替换为 $R' = R + \gamma \Phi(s') - \Phi(s)$，其中 $\Phi(s)$ 是当前布局的物理合理性评分。
- **优势**：
  - 提供密集的中间奖励信号，缓解稀疏奖励问题。
  - 基于 **policy-invariance theorem**，保证最优策略不变。
  - 加速 RL 探索，尤其在大搜索空间下效果显著。

### 相比现有方法的优势
| 维度 | 传统方法 | PhysEDA |
|------|--------|---------|
| **复杂度** | $O(L^2)$，内存爆炸 | $O(L)$，支持超大规模设计 |
| **数据效率** | 依赖大量数据学习空间结构 | 显式编码物理先验，小样本下更鲁棒 |
| **训练效率** | RL 收敛慢，探索困难 | PBRS 提供物理引导，加速收敛 |
| **泛化能力** | 难以跨尺度、跨设计迁移 | PSLA 的物理核天然支持 zero-shot 跨尺度转移 |

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **Decoupling Capacitor Placement (DPP)**  
   - 基于 DevFormer 的物理公式生成，共 2,300 个实例。
   - 分为 `10×10`（L=100）和 `25×25`（L=625）两种规模。
   - 划分：2,000/100/200 的 train/val/test。

2. **Macro Placement**  
   - 使用 ISPD 2005 adaptec1 基准，含 452 个宏模块。
   - 使用 ChiPFormer 提供的专家轨迹进行离线预训练。

3. **IR-drop Prediction**  
   - 使用 **CircuitNet 1.0** 数据集，包含多个芯片设计（NVDLA, Vortex, openc910 等）。
   - 输入为 power-density map，输出为电压降分布图。

### 实验设置和评估指标
| 任务 | 模型架构 | 训练方式 | 主要指标 |
|------|----------|----------|----------|
| **DPP** | Autoregressive Decoder | Imitation Learning + REINFORCE RL | DPP Score（越负越好） |
| **Macro Placement** | Decision Transformer (ChiPFormer) | Offline Pretraining + Online RL Fine-tuning | HPWL（Half-Perimeter Wirelength，越低越好） |
| **IR-drop Prediction** | UNet | Supervised Regression | Pearson Correlation（越高越好） |

### 基线方法对比
- **DPP**：DevFormer（Transformer）、Plain GLA、FAVOR+、Simple Linear、CosFormer。
- **Macro Placement**：GPT DT、GPT + REINFORCE、GPT + GRPO。
- **IR-drop**：Standard UNet。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）DPP 任务（Decoupling Capacitor Placement）
| 设置 | 方法 | Score | 相对提升 |
|------|------|-------|---------|
| **Supervised (25×25)** | DevFormer | -15.88 | — |
| | **PSLA (Ours)** | **-16.76** | **+5.5%** |
| **RL (10×10)** | DevFormer | -10.50 | — |
| | PSLA | -10.66 | +1.5% |
| | DevFormer + PBRS | -13.01 | +23.9% |
| | **PSLA + PBRS** | **-13.01** | +23.9% |
| **RL (25×25)** | DevFormer | -10.53 | — |
| | PSLA | -15.70 | **+49.1%** |
| | DevFormer + PBRS | -13.34 | +26.7% |
| | **PSLA + PBRS** | **-17.40** | **+65.2%** |
| **Zero-shot Transfer (10→25)** | DevFormer | -7.04 | — |
| | **PSLA** | **-11.04** | **+56.8%** |

> ✅ **关键发现**：随着问题规模增大（L 从 100 → 625），PSLA 的优势急剧放大，验证了“数据越稀缺，物理先验越重要”的原则。

#### （2）Macro Placement（ChiPFormer）
| 阶段 | 方法 | HPWL | 相对提升 |
|------|------|------|---------|
| **Pretraining (DT)** | GPT DT | 910,967 | — |
| | **PSLA DT** | **801,588** | **+12.0%** |
| **RL Fine-tuning** | GPT + REINFORCE | 733,794 | — |
| | GPT + PBRS (logit bias) | **693,999** | **+5.4%** |
| | PSLA + GRPO | 762,974 | -4.0% |

> ⚠️ **注意**：PSLA 在 RL 微调阶段表现不佳，因其“硬约束”限制了探索；而 PBRS 的“软约束”更适合 RL。两者互补。

#### （3）IR-drop Prediction（CircuitNet）
| 设置 | Baseline | PSLA-UNet | 提升 | Win Rate |
|------|---------|----------|------|----------|
| **In-distribution** | 0.834 | 0.818 | -1.9% | — |
| **Cross-design** | 0.458 | 0.482 | **+5.3%** | 12/15 |
| **Cross-architecture** | 0.472 | 0.498 | **+5.4%** | 11/15 |

> ✅ **关键发现**：在数据丰富时（in-distribution），PSLA 因引入强先验反而略逊；但在分布外（OOD）场景下，PSLA 显著胜出，证明其增强泛化能力。

### 消融实验结果
| 配置 | DPP Score | 相对变化 |
|------|----------|----------|
| Full PSLA (ours) | -16.763 | — |
| - Decay (only PPE) | -16.182 | -3.6% |
| - Manhattan → Euclidean decay | -16.092 | -4.0% |
| - No PPE | -13.913 | -17.0% |
| Fixed $\alpha=1.5$ | -16.756 | ≈ 相同 |

> 🔍 **结论**：增益主要来自 **Manhattan decay 结构**，而非单纯的位置编码；$\alpha$ 可学习带来边际收益，但物理范围本身已足够有效。

---

## 4. 关键结论和发现

### 主要发现
1. **统一物理先验的有效性**：  
   单一的 **Manhattan distance decay** 先验可同时指导架构设计（PSLA）和训练优化（PBRS），形成协同效应。

2. **性能增益与数据稀缺性正相关**：  
   PhysEDA 的优势在以下场景最显著：
   - 小样本训练（如 DPP）
   - 分布外泛化（如 CircuitNet cross-design）
   - 跨尺度迁移（10×10 → 25×25）
   - 大规模搜索空间（RL 探索困难）

3. **PSLA 与 PBRS 各司其职**：
   - **PSLA** 主导监督学习和 zero-shot 迁移。
   - **PBRS** 主导 RL 探索，尤其在稀疏奖励下。
   - 二者结合可实现端到端性能突破。

4. **计算效率巨大提升**：
   - 在 `100×100` 网格上，PSLA 实现 **14× 推理加速** 和 **98.5% 内存节省**。
   - 在 `150×150` 上，softmax attention 超出 GPU 内存，而 PSLA 仍可运行（**32.5× 加速**）。

### 方法的局限性
1. **Rank-1 Factorization 的近似性**：  
   当前 PSLA 使用 $ \exp(\alpha(x_i - x_j)) $ 实现的是**有向衰减**，而非对称的 $ \exp(-\alpha|x_i - x_j|) $。虽可通过双向前缀和精确重建，但增加计算开销。

2. **PBRS 仅适用于在线 RL**：  
   奖励塑形无法应用于纯监督或离线 RL 场景。

3. **PSLA 的“硬约束”可能抑制 RL 探索**：  
   在宏单元布局微调中，PSLA 的强空间偏好限制了策略多样性，需与 PBRS 的“软引导”权衡使用。

### 未来工作方向
1. 实现 **bidirectional prefix-sum decomposition**，以精确建模对称 Manhattan 衰减。
2. 扩展物理势能至其他 EDA 任务，如 **parasitic capacitance estimation** 或 **signal integrity**。
3. 将框架推广至 **clock distribution**、**thermal analysis** 等更多物理感知任务。
4. 探索更灵活的混合约束机制，平衡 PSLA 的归纳偏置与 PBRS 的探索自由度。

--- 

> 💡 **总结一句话**：  
> **PhysEDA 通过将“曼哈顿距离指数衰减”这一物理规律作为统一归纳偏置，实现了 EDA 学习模型在精度、效率和泛化上的全面突破，尤其在数据稀缺和大规模场景下优势显著。**

</details>

---

### 5. [Agent-X: Full Pipeline Acceleration of On-device AI Agents](https://arxiv.org/abs/2605.10380)

**Authors**: Jinha Chung, Byeongjun Shin, Jiin Kim, Minsoo Rhu  
**Category**: cs.AI  
**Published**: 2026-05-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.10380v1  

#### Abstract
LLM-based agents deliver state-of-the-art performance across tasks but incur high end-to-end latency on edge devices. We introduce Agent-X, a software-only, accuracy-preserving framework that accelerates both the prefill and decode stages of on-device agent workloads. Agent-X's two key components re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Agent-X: Full Pipeline Acceleration of On-device AI Agents》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

- **问题背景**：基于 LLM 的 AI Agent 在边缘设备（edge devices）上运行时面临显著的端到端延迟问题，尤其是在 **prefill 和 decode 两个阶段**均成为瓶颈的情况下。
- 传统云上 LLM 推理中，decode 阶段是主要瓶颈；但在 on-device agent 场景下，由于输入更长、硬件资源受限，**prefill 阶段也变得非常耗时**。
- 现有优化技术（如 prefix caching 和 speculative decoding）在 agent 工作流中效果有限，因为：
  - 输入 prompt 动态性强，难以有效缓存；
  - speculative decoding 依赖额外的 draft LLM，在边缘设备上开销大且受“multi-token tax”影响。

### 🚀 提出的新方法与创新思路

提出 **Agent-X** —— 一种纯软件、不损失准确性的 on-device AI Agent 全流程加速框架，包含两大核心技术：

#### （1）**PromptWeaver**：用于加速 prefill 阶段
- **核心思想**：重构输入 prompt 结构，使更多内容可被 prefix caching 复用。
- **关键技术**：
  - 将所有工具描述和指南设为静态前缀，避免早期动态 token 导致缓存失效；
  - 利用 **tool co-activation locality** 进行聚类，并预计算常用组合的 KV cache；
  - 引入 **cluster combination selection 算法**，在 SSD 上存储最具覆盖性的 cluster 序列 KV 缓存。
- **优势**：无需修改模型或硬件，仅通过 prompt 重组提升缓存命中率。

#### （2）**ExSpec**：用于加速 decode 阶段
- **核心思想**：引入轻量级、免训练的 **n-gram lookup table (LUT)** 作为 draft model，实现 LLM-free speculative decoding。
- **关键技术**：
  - 从 few-shot examples 和用户 query 构建 trigram LUT，生成 draft tokens；
  - 设计 **selective decoding 机制**：当上下文不在 LUT 中时，直接回退到 autoregressive 生成，规避 multi-token tax；
  - 支持快速 fallback，无额外推理开销。
- **优势**：相比传统 draft LLM，内存占用极小（KB 级），无 fine-tuning 成本，且能有效避免验证阶段性能下降。

### 🔍 相比现有方法的优势

| 维度 | 传统方法局限 | Agent-X 改进 |
|------|--------------|-------------|
| **Prefill 优化** | Prefix caching 对动态 prompt 效果差 | PromptWeaver 重构 prompt，最大化缓存复用 |
| **Decode 加速** | Speculative decoding 需要额外 LLM，开销高 | ExSpec 使用 LUT 替代 LLM，零训练成本 |
| **系统适配性** | 多数方案针对通用 LLM 推理 | 专为 agent 特征设计（如 tool calling、few-shot 模板） |
| **部署友好性** | 硬件依赖强或需模型重训 | 纯软件方案，可无缝集成现有系统 |

---

## 2. 核心实验方法和设置

### 📚 数据集

- 使用开源 on-device agent 框架 **TinyAgent [19]** 及其配套数据集：
  - **TinyAgent-dataset [68]**：包含 1,022 个测试样本，涵盖最多 16 种不同工具调用任务。
  - 查询类型多样，涉及日历、邮件、联系人、地图等操作。
  - 训练集用于 PromptWeaver 的 tool clustering 和 cluster combination selection。

### 💻 实验平台

- **硬件**：Apple Mac mini（M4 Pro 芯片），64GB RAM，512GB SSD
- **软件栈**：
  - 基于 Apple 官方 LLM 推理框架 **MLX-LM [10]** 和 **MLX-engine [43]**
  - 后端模型：**TinyAgent-7B**（基于 WizardLM-2-7B 微调）
- 所有实验前清空 page cache，确保公平测量 KV cache 加载开销。

### 🎯 评估指标

| 指标 | 定义 |
|------|------|
| **End-to-end latency** | 从接收用户请求到返回最终结果的时间 |
| **Prefill / Decode latency** | 分别测量 prefill 和 decode 阶段耗时 |
| **Speedup** | 相对于 baseline 的加速比 |
| **Planner accuracy** | 输出计划 DAG 与 ground truth 是否一致 |
| **KV cache hit rate / coverage** | 缓存复用比例 |
| **Draft token accuracy** | speculative decoding 中被接受的 draft token 比例 |

### ⚖️ 基线方法对比

| 方法 | 描述 |
|------|------|
| **Baseline** | 原始 TinyAgent 流程，无任何优化 |
| **Static caching** | 仅对完全静态 prompt 进行 prefix caching |
| **SpecDec (Llama-3.2-1B)** | 使用小型 LLM 作为 draft model 的 speculative decoding |
| **PLD [8]** | Prompt Lookup Decoding，基于 prompt 构建 LUT 的类似工作 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

| 模块 | 方法 | 加速比（Speedup） | 准确性变化 |
|------|------|------------------|-----------|
| **Prefill** | PromptWeaver | **1.97×** | 无损失（K=1 时精度略升） |
| **Decode** | ExSpec (trigram + selective) | **1.73×** | 保持原精度 |
| **End-to-end** | Agent-X (PW + ES) | **1.61×** | 无准确率下降 |

> 注：单独启用 PromptWeaver 或 ExSpec 分别带来 **1.16×** 和 **1.43×** 的端到端加速。

### 🔬 与基线方法对比

| 方法 | Planner Decode Speedup | 说明 |
|------|------------------------|------|
| Baseline | 1.00× | 原始性能 |
| SpecDec (Llama-3.2-1B) | **0.83×**（变慢） | 受 multi-token tax 和 tokenizer 不匹配拖累 |
| ExSpec (non-selective) | 1.38× | 优于 SpecDec，但仍有冗余验证 |
| **ExSpec (selective)** | **1.73×** | 显著优于所有 baseline |

- **KV cache 覆盖率**：在仅使用 **15 个 cluster 组合**（6.26 GB 存储）时，达到 **74.4% 的 tool-use example 覆盖率**。
- **uncacheable token 数量**：
  - Baseline：1,711 tokens
  - PromptWeaver (K=1)：**519 tokens（↓70%）**

### 🔍 消融实验结果

#### （1）PromptWeaver 中附加示例数量的影响（K 值）

| K | Planner Accuracy | Uncacheable Tokens |
|----|------------------|--------------------|
| 0 | 0.832 | 519 |
| **1** | **0.841（峰值）** | 519 |
| 2 | 0.839 | ↑ |
| 3~4 | ↓ | ↑↑ |

✅ 结论：**K=1 是最优选择**，兼顾准确性与缓存效率。

#### （2）ExSpec 中 n-gram 模型的选择

| n | Draft Token Accuracy | Speedup | 说明 |
|----|------------------------|--------|------|
| 1 (unigram) | 低 | <1.2× | 上下文不足，预测质量差 |
| 2 (bigram) | 0.10 | ~1.5× | 性能一般 |
| **3 (trigram)** | **0.25** | **1.73×** | 最佳平衡 |
| 4 (quadgram) | 0.31（更高） | ↓（更慢） | 上下文太长导致 fallback 频繁 |

✅ 结论：**trigram 提供最佳 trade-off**。

#### （3）LUT 构建范围的影响

| 提取区域 | Planner Speedup | Arbiter Speedup |
|----------|------------------|------------------|
| 全部输入（all） | 1.70× | 1.68× |
| **仅 few-shot + query** | **1.73×** | **1.69×** |

✅ 结论：排除无关 system prompt 可进一步提升性能，尤其对输入较长的 Planner 更明显。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **On-device agent 的性能瓶颈不同于传统 LLM**：
   - 不再是 decode 单一主导，而是 **prefill 和 decode 均重要**，必须全链路优化。
   - 平均 prefill 占总延迟 21.7%，decode 占 68.7%，合计近 90%。

2. **Agent 输入具有高度结构性和可预测性**：
   - 输出 plan 高度模板化（96% Planner tokens 与 prompt 重叠）；
   - 工具存在明显的 **co-activation locality**，可用于缓存优化。

3. **轻量级 speculative decoding 更适合边缘场景**：
   - 传统 draft LLM 因 multi-token tax 和 tokenizer 开销反而变慢；
   - **n-gram LUT + selective fallback** 是高效替代方案。

4. **Prompt 结构直接影响系统性能**：
   - 通过 PromptWeaver 重构 prompt，可在不改模型前提下大幅提升缓存利用率。

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖 SSD 存储 KV cache** | 增加约 6.26 GB 存储开销，可能不适合存储极度受限设备 |
| **适用于 fine-tuned agent 模型** | 当前基于 TinyAgent 设计，未验证通用性 |
| **对 prompt 分布漂移敏感** | 若新增大量新工具组合，需重新构建 cluster cache |
| **目前仅支持文本型 agent** | 不适用于 Mobile-Agent 类视觉驱动 agent |

### 🔮 未来工作方向

1. **扩展至多模态 agent**：将 PromptWeaver 和 ExSpec 思想应用于视觉 prompt 的缓存与解码加速。
2. **动态更新 KV cache**：支持在线学习新 tool-use patterns 并增量更新 cache。
3. **跨设备迁移**：适配 Android、嵌入式 Linux 等其他边缘平台。
4. **结合量化与压缩技术**：进一步降低 KV cache 内存与存储占用。
5. **探索更智能的 cluster selection 策略**：基于用户行为个性化缓存。

---

## 总结

> **Agent-X 是首个系统性分析并解决 on-device AI Agent 全流程延迟瓶颈的工作**。它提出 **PromptWeaver** 和 **ExSpec** 两项纯软件技术创新，分别针对 prefill 和 decode 阶段进行精准优化，在 **不牺牲准确性的前提下实现了 1.61× 的端到端加速**。该方法具有良好的部署灵活性，为推动私有、实时、高效的本地 AI Agent 落地提供了重要实践路径。

</details>

---

### 6. [Structured Recurrent Mixers for Massively Parallelized Sequence Generation](https://arxiv.org/abs/2605.08696)

**Authors**: Benjamin L. Badger  
**Category**: cs.CL  
**Published**: 2026-05-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.08696v1  

#### Abstract
Over the last two decades, language modeling has experienced a shift from predominantly recurrent architectures that process tokens sequentially during training and inference to non-recurrent models that process sequence elements in parallel during training, which results in greater training efficie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Structured Recurrent Mixers for Massively Parallelized Sequence Generation

---

## 1. 主要贡献和创新点

### **解决了什么问题**

当前主流语言模型（如 **Transformer**）在训练时采用序列并行（sequence-parallel），效率高且稳定，但在推理阶段生成 token 时必须串行进行，导致**推理吞吐量低、并发能力差**。而传统的**循环模型**（如 RNN、LSTM）虽然推理高效，却难以实现高效的并行化训练。

此外，尽管一些线性复杂度架构（如 **Mamba**, **RWKV**）尝试融合两者优势，但通常依赖**定制化的 CUDA 内核**或复杂的内存管理机制，限制了其跨硬件平台的可移植性和部署灵活性。

---

### **提出的新方法与新思路**

本文提出了 **Structured Recurrent Mixer (SRM)**，一种新型的序列建模架构，具备以下核心创新：

- ✅ **双表示统一架构**：  
  SRM 可以在训练时表现为**序列并行形式**（用于高效梯度计算），在推理时无缝转换为**循环形式**（用于高效自回归生成），且无需修改参数或缓存结构。
  
- ✅ **代数等价转换**：  
  利用特定结构化的 token mixing 矩阵（如行重复、列重复 + 衰减因子），使得矩阵乘法操作可以被代数地转化为固定大小状态更新的循环操作，从而天然支持 O(1) 空间复杂度。

- ✅ **无需专用内核**：  
  不像 Mamba 或 RWKV 需要定制 scan 操作或特殊 kernel，SRM 的循环实现仅需少量 BLAS Level 1 操作（向量加法、标量乘法），可在标准框架（PyTorch）中直接高效运行，并兼容各种设备。

- ✅ **面向“验证型任务”的优化范式**：  
  提出应更关注 **accuracy per compute** 而非单样本准确率，尤其适用于输出可快速验证的任务（如代码生成、数学解题）。SRM 的高并发推理特性使其能通过大规模采样 + 快速筛选提升有效输出效率。

---

### **相比现有方法的优势**

| 特性 | SRM | Transformer | Mamba | RWKV |
|------|-----|-------------|--------|-------|
| 训练效率 | 高于 Mamba | 最高 | 中等 | 低（数值不稳定） |
| 推理吞吐 | 极高（>10x） | 低 | 中等 | 中等 |
| 并发能力 | 极强（~170x） | 弱 | 弱 | 弱 |
| 缓存增长 | O(1) per sample | O(n) | O(n) | O(n) |
| 是否需要定制 Kernel | ❌ 否 | ❌ 否 | ✅ 是 | ✅ 是 |
| 跨设备兼容性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

> 🔍 尤其在旧 GPU（如 V100）上，Mamba 性能急剧下降，而 SRM 表现稳健。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **预训练数据集**：
  - `FineWeb-edu`：大规模网页文本语料，用于通用语言建模。
  - `FineMath 4+`：高质量数学与编程相关文本，用于数学推理任务微调。

- **下游评测基准**：
  - 功能理解：`Arc-Easy`, `HellaSwag`, `Lambada`
  - 信息保留：`SQuAD`, `SQuAD v2`, `LongBench`, `IFEval`, `SWDE`, `xWinoGrad`
  - 数学推理：`GSM8k`（带答案验证）
  - 压缩能力测试：自定义 copy task 和 input reconstruction 任务

---

### **实验设置与评估指标**

#### **模型配置**
- 对比模型：
  - **Transformer**（Llama 2 架构）
  - **Mamba**（selective SSM）
  - **RWKV**（RNN-like with KV caching）
  - **Masked Mixer**（baseline MLP-based mixer）
- 参数规模对齐或计算预算对齐（compute-equivalent）

#### **评估维度**
| 维度 | 指标 |
|------|------|
| **训练效率** | loss 下降速度、FLOPs per loss、device throughput (samples/sec) |
| **推理性能** | throughput (tokens/sec), max concurrency (batch size) |
| **功能表现** | Pass@k（GSM8k）、zero-shot accuracy（其他 benchmark） |
| **信息容量** | 输入重建误差、熵比率（entropy ratio）、copy accuracy |
| **强化学习效果** | GRPO 训练后 Pass@k 提升幅度 |

#### **硬件环境**
- V100 (16GB), H100 (96GB NVLink)
- 使用 PyTorch、vLLM（Transformer）、Mojo/MAX（SRM 自研引擎）

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🚀 推理吞吐与并发能力（H100, ctx=512）

| 模型 | Throughput (t/s) | Max Concurrency |
|------|------------------|-----------------|
| **SRM (Mojo/MAX)** | **1203k** | **800k** |
| Transformer (vLLM) | 101k | 5.32k |
| Mamba (vLLM) | 44k | ~0.02k* |
| RWKV (Albatross) | 184k | 50k |

> 💡 SRM 实现了：
> - **12x 的吞吐提升**
> - **170x 的并发能力提升**

> *注：Mamba 存在内存泄漏，实际并发远低于理论值*

#### 在 V100 上的表现一致性

| ctx_len | Transformer | SRM | Speedup |
|--------|------------|------|---------|
| 512    | 2908       | 28091 | 9.66x |
| 4096   | 634        | 27445 | 43.29x |

> 随着上下文增长，SRM 相对优势显著扩大。

---

### **与基线方法的对比结果**

#### ✅ 功能性 benchmark（compute-matched training）

| Benchmark | Transformer | Mamba | **SRM** |
|----------|------------|--------|--------|
| GSM8k (Pass@k) | 1.90±1.36 | 1.36±0.32 | **1.44±0.33** |
| ARC-Easy | 48.11 | 33.96 | **50.0** |
| IFEval (strict) | 12.2 | 25.66 | **21.22** |
| SQuAD v2 | 5.1 | 1.69 | **16.87** |

> ✅ SRM 在多数任务上优于 Mamba，接近甚至超越 Transformer。

#### ✅ 信息保留与容量

| 模型 | Entropy Ratio (higher=better) |
|------|-------------------------------|
| SRM | **0.3168** |
| Mamba | 0.2204 |
| Untrained SRM | 0.3239（表明训练未损害信息保留）|

> 🔍 SRM 的信息保留能力接近二次复杂度模型，显著优于 Mamba。

#### ✅ Copy Task（512 tokens, 10k steps）

| 模型 | Accuracy |
|------|----------|
| SRM (d=512) | **0.9673** |
| Mamba (d=512) | 0.9673（但需更多 compute） |
| SRM (d=256) | 0.6886 → 显示 compute-scaling 更优 |

> 表明 SRM 在有限训练步数下仍能高效学习长程依赖。

---

### **消融实验结果**

#### 不同 SRM 结构的影响（Table S2, S5）

| 架构变体 | Eval Loss ↓ |
|--------|-----------|
| Row Repeat Only | 3.311 |
| Column Repeat Only | 3.275 |
| Mixed Heads (row + col) | **3.228** |
| + Decay Term | **3.002** ✅ 最佳 |
| + Diagonal Bias | 无明显增益（2.981 vs 3.002）❌ |

> ✅ **混合头 + 衰减项**是最有效的设计选择。

#### 投影层影响（Table S7）

| 是否使用 Head Projections | Throughput Cost | Training Efficiency |
|--------------------------|------------------|---------------------|
| 是 | ~10% 降低 | 略优（loss 更低） |
| 否 | 更快 | 可接受 |

> 权衡之下推荐使用投影以提升表达力。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **循环模型更适合 batch scaling 而非 sequence length scaling**：  
   由于每个样本仅维护 O(1) 缓存，当输入信息密度高（如语言）时，过长序列会导致信息压缩失真。因此，**扩展 batch size 比扩展 context 更合理**。

2. ✅ **SRM 实现了训练/推理双优平衡**：  
   - 训练时利用序列并行获得类似 Transformer 的稳定性；
   - 推理时切换为循环模式，实现极高的吞吐与并发。

3. ✅ **高并发推理显著提升 verify-then-select 类任务的有效产出**：  
   在 GSM8k 上，固定 compute 预算下，SRM 比 Transformer 多解决约 **30% 的问题**，因其可通过大量采样+验证选出正确答案。

4. ✅ **Mojo/MAX 推理引擎进一步释放潜力**：  
   相比 PyTorch 实现，MAX 上的 SRM 达到 **7.5x 吞吐提升**，证明其适合生产级部署。

5. ✅ **引入 Balanced Resampling 提升 GRPO 探索能力**：  
   在强化学习中，大 batch 容易因 reward 稀疏导致探索退化。作者提出的 resampling 方法（保留最多一半“好”样本）有效缓解该问题，使 SRM 在 GRPO 中表现优于小 batch Transformer。

---

### **方法的局限性**

1. ❌ **训练阶段为 O(n²) 复杂度**：  
   尽管推理是 O(1) 空间，但训练仍需构建 full mixing matrix，不适合超长序列（n >> d）场景。

2. ❌ **缺乏原生 context extension 机制**：  
   当前未提供类似 RoPE 或 ALiBi 的位置编码方案来动态扩展 context。

3. ❌ **参数效率不如小型 Mamba/Transformer**：  
   在极小参数设定下不具优势，更适合中大型模型。

4. ❌ **尚未探索所有优化技术**：  
   如 parallel scan 加速 prefill、量化支持等仍有改进空间。

---

### **未来工作方向**

1. 🔮 开发适用于 SRM 的 **long-context extrapolation 方法**
2. 🔧 实现完整的 **production-grade inference pipeline**（含量化、批处理调度）
3. 🧠 探索 SRM 在 **agent workflow、code generation、search-based reasoning** 中的应用
4. 🔄 研究如何将 **accuracy-per-compute 范式推广至更多任务**
5. 🤖 结合 SRM 与 **test-time compute scaling**（如 Best-of-N, Monte Carlo Tree Search）

---

> 📌 **总结一句话**：  
> **SRM 是首个无需定制 kernel 即可在训练与推理间自由切换、同时兼具高训练效率与超高推理并发能力的循环架构，为“验证驱动”的高效语言生成提供了全新路径。**

</details>

---

### 7. [Merlin: Deterministic Byte-Exact Deduplication for Lossless Context Optimization in Large Language Model Inference](https://arxiv.org/abs/2605.09990)

**Authors**: Sietse Schelpe  
**Category**: cs.CL  
**Published**: 2026-05-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.09990v1  

#### Abstract
Data-intensive applications, ranging from large-scale retrieval systems to advanced data pipelines, are increasingly bottlenecked by the processing of highly redundant text corpora. We present Merlin, a local-first, agnostic, high-throughput deduplication and context optimization engine designed to ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Merlin: Deterministic Byte-Exact Deduplication for Lossless Context Optimization in Large Language Model Inference*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大型语言模型（LLM）推理中，**输入上下文（prompt context）存在大量结构性冗余**，主要来源包括：
- **Retriever 输出重叠**：检索系统返回的 chunks 天然存在重复。
- **多轮对话历史累积**：session history 中频繁出现完全相同的语句复述。
- **并发用户共享内容**：多个用户从同一知识库中检索出相同段落。

这些冗余虽然对人类可见，但对模型层是“透明”的，导致：
- **Prefill 阶段计算资源浪费**
- **延迟增加**
- **每调用成本上升**

传统去重方法因**高延迟、非确定性或非字节精确**而无法部署于生产级推理路径前端。

---

### 提出的新方法与创新思路
提出 **Merlin 引擎**：一个用于 LLM 推理前处理的 **Deterministic Byte-Exact Deduplication** 原语。

#### 核心定义（Section 3.1）
- **Byte-Exact Equivalence**: 两段文本当且仅当所有字节完全相同时才被视为重复。
- **Deduplication Rule**: 保留每个等价类中首次出现的记录，顺序不变。

该方法**不修改 retriever、chunker 或 model**，仅作为 prompt assembler 前的一个轻量预处理步骤。

---

### 相比现有方法的优势

| 维度 | Merlin | 其他方法（如 LLMLingua, REFRAG, RAGBoost） |
|------|--------|------------------------------------------|
| **质量属性** | ✅ Lossless（无损） | ❌ Lossy（有损压缩）或 Lossless but approximate |
| **确定性** | ✅ Bit-for-bit deterministic | ❌ 非确定性或跨平台输出不一致 |
| **速度** | ✅ 1.1 μs（in-process） | ⚠️ 毫秒级甚至更高（尤其涉及子进程调用时） |
| **部署开销** | ✅ 单二进制文件（3.8MB），无依赖 | ⚠️ 需 Python 运行时、GPU、复杂依赖 |
| **通用性** | ✅ 跨领域适用（log, web crawl, code） | ⚠️ 多为特定任务设计 |

> **核心优势总结**：Merlin 在满足 **生产级低延迟约束** 的前提下，实现了**数学上等价于 `Python set()` 的字节精确去重**，并保证跨平台输出一致性。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖三大类基准与真实数据：

| 类型 | 数据集 | 描述 |
|------|-------|------|
| **长上下文检索** | RULER | 包含 UUID 构造的 haystacks，测试 multi-needle retrieval、variable tracing 等能力 |
| **长文档理解** | LongBench (paragraph-safe subset) | narrativeqa, qasper, govreport 等任务，排除代码相关子任务 |
| **多轮编码会话** | HumanEval-Snowball + WildChat-1M | 在真实多轮对话历史后接 HumanEval 编码任务，模拟“雪球效应” |
| **大规模验证** | BeIR corpus (22.2M passages) | 用于验证 Merlin 与 `set()` 在千万级数据上的数学等价性 |

---

### 实验设置与评估指标

#### 推理接口
通过 **OpenRouter** 路由至四个主流 LLM API：
- Google Gemini 2.5 Flash
- OpenAI GPT-5.1
- Anthropic Claude Sonnet 4.6
- Meta Llama 3.3 70B Instruct

所有请求使用 `temperature=0.0`，输出 token 上限为 2048。

#### 评估指标
- **Accuracy / Pass@1 / F1 / ROUGE-L**：原始任务性能。
- **Quality Delta (Δ)**：去重前后性能差异（以百分点 pp 表示）。
- **Statistical Significance**：
  - 主要检验：**Paired Sign Test**
  - 辅助检验：**Paired One-Sample t-test**
  - 多重检验校正：**Bonferroni Correction**（family-wise α=0.05）

#### 对照组
- **Raw Baseline**：未去重的原始输入。
- **Deduped Condition**：经 Merlin 去重后的输入。

---

### 基线方法对比
Merlin 并未直接与算法级基线进行端到端性能比较，而是强调其**工程定位的独特性**。在 Table 3 中进行了横向对比：

| 方法 | 是否 Lossless | Per-call 开销 | Deterministic | Cross-Vendor |
|------|---------------|----------------|----------------|----------------|
| Merlin | ✅ Yes | 1μs ~ 21ms* | ✅ Yes | ✅ Yes |
| LLMLingua-class | ❌ No (lossy) | Task-dependent | ❌ No | ✅ Yes |
| REFRAG | ✅ Yes (perplexity preserved) | Offline + small online | ✅ Precomputed | ❌ Architecture-dependent |
| RAGBoost | ✅ Yes | Millisecond-scale | ✅ Yes | ✅ Yes |
| Vendor Prompt Caching | ✅ Yes (if hit) | Often 0 | ❌ Cache key opaque | ❌ Vendor-specific |

> *注：21ms 来源于 subprocess + tempfile 模式，属部署选择而非引擎本身开销。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 数值 | 来源 |
|------|------|------|
| **In-process Latency** | **1.10 μs**（median） | Section 3.3 |
| **Production Binary Internal Counter** | **5–30 μs**/call | Section 3.3 |
| **Subprocess (Pipe IPC)** | ~13 ms | Section 3.4 |
| **Subprocess (Tempfile)** | ~21 ms | Section 3.4 |
| **Binary Size (Win x86-64)** | 3.8 MB | Section 3.2 |
| **Binary Size (Linux ARM64)** | 3.5 MB | Section 3.2 |

> 去重操作比典型推理预处理预算（10–50ms）低 **3–4 个数量级**，可视为“零成本”。

---

### 与基线方法的对比结果

#### 总体质量影响（Aggregate Verdict）

| 实验阶段 | Cells | Mean Δ | Worst Δ | Significant Cells (Bonferroni) |
|---------|-------|--------|----------|-------------------------------|
| Primary Sweep | 40 | **+0.0 pp** | -4.0 pp | **0** |
| Warm-Binary Confirmation | 200 | **-0.5 pp** | -1 cell | **0** |

✅ 所有单元均无统计显著退化，表明 Merlin **在评估粒度下保持模型质量不变**。

---

#### 分项任务表现

| 任务 | 结果摘要 |
|------|----------|
| **RULER** | 24 cells，平均 Δ=0.0 pp；唯一 -4.0 pp 单元 p=0.500（不显著） |
| **LongBench (paragraph-safe)** | 12 cells，平均 Δ=-0.004 score points；最小 sign-test p=0.180 > 0.0042（校正阈值） |
| **HumanEval-Snowball** | 4 cells，平均 Δ=+0.5 pp；最大变化 +4.0 pp（GPT-5.1），仍不显著 |

---

### 消融实验结果（Ablation Study）

#### Line-Level Deduplication on Code Tasks（Table 11）
在 `lcc` 和 `repobench-p` 上测试行级去重（非默认配置）：

| Vendor | lcc Δ | repobench-p Δ | 影响 |
|--------|--------|----------------|------|
| Gemini | -0.195 | -0.089 | ❌ 显著损害 |
| Claude | +0.023 | -0.005 | ➖ 中性 |
| GPT-5.1 | +0.028 | +0.114 | ✅ 改善 |
| Llama | +0.038 | +0.001 | ➖ 中性 |

> **结论**：行级去重效果高度依赖 vendor，因此默认采用**段落级去重**，代码任务被排除在主声明之外。

---

### 大规模数学等价性验证（Section 4.11）
在 **22.2M BeIR passages** 上运行 Merlin，并与 `Python set()` 对比：

- **Total Passages**: 22,221,024
- **Unique (Merlin)**: 22,185,502
- **Cross-corpus Duplicates**: 35,522 (0.1599%)
- **Python set() Unique Count**: 22,185,502
- **Math-equivalence Violations**: **0**
- **Per-query verification (327 queries)**: **0 violations**

✅ 完全数学等价，证明 Merlin 正确实现 `set()` 语义。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Byte-Exact Deduplication 是安全的**：在多种 LLM 和 benchmark 上，去重前后性能无统计显著差异。
2. ✅ **极低延迟使其可用于实时推理路径**：in-process 模式仅需 ~1.1μs，远低于预处理预算。
3. ✅ **跨平台输出一致**：Windows x86-64 与 Linux ARM64 构建输出字节等价（non-code prompts 达 100%，overall 99.2%）。
4. ✅ **通用性强**：已在 log analysis、web crawl、scientific data 等领域验证有效性（Section 4.13）。
5. ✅ **与现有优化互补**：可与 prompt caching、KV-cache reuse、REFRAG 等共存，叠加收益。

---

### 局限性
1. ❌ **不处理语义级冗余**：仅识别字节完全相同的重复，无法消除 paraphrase 或近似重复。
2. ❌ **代码任务需谨慎使用**：行级去重可能损害某些模型（如 Gemini）的表现。
3. ❌ **闭源实现**：Merlin 为 closed-source 生产基础设施，完整复现受限。
4. ❌ **未覆盖所有垂直领域**：医疗、法律、多模态等 specialized retrieval 尚未验证。
5. ⚠️ **依赖 chunk boundary**：去重粒度受上游 chunker 输出影响，无法超越其限制。

---

### 未来工作方向
1. **扩展至语义去重**：结合 MinHash、embedding similarity 等技术，在可控误差下提升压缩率。
2. **支持 streaming deduplication**：在流式输入场景下动态去重。
3. **开放 clean-room benchmark track**：允许更多第三方验证吞吐与体积声明。
4. **探索更细粒度 token-level dedup**：研究 sub-token 重复的可能性。
5. **集成至 agentic workflow 标准栈**：作为 agent memory management 的基础组件。

---

> **最终结论**：  
> Merlin 展示了一种**工程上可行且理论上安全**的上下文优化原语——**Deterministic Byte-Exact Deduplication**。它虽简单，但在正确的位置（inference proxy 前端）以足够低的成本执行，使得原本“不可行”的实时去重成为可能。这不仅是算法改进，更是**系统架构层面的范式转移**：将一个经典操作重新带入实时推理的关键路径。

</details>

---

### 8. [RubiConv -- Efficient Boundary-Respecting Convolutions](https://arxiv.org/abs/2605.08451)

**Authors**: Linda Friso, Annie Marsden, Xinyi Chen, Arushi Gupta, Peter Bartlett, Mark Braverman, Elad Hazan  
**Category**: cs.LG  
**Published**: 2026-05-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.08451v1  

#### Abstract
Convolutional architectures have emerged as powerful alternatives to Transformers for sequence modeling. The primary advantage is that they offer improved theoretical sequence length complexity by leveraging the Fast Fourier Transform (FFT). However, this theoretical improvement does not always mean...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RubiConv – Efficient Boundary-Respecting Convolutions 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大规模语言模型（LLM）训练中，为了提升硬件利用率（如GPU/TPU），通常将多个变长文档“打包”（packing）成一个固定长度的序列进行处理。然而，基于 **FFT 的卷积操作**（如 Hyena、FlashSTU 等）由于其**周期性假设**（periodicity assumption），会导致：
- **边界信息泄露**（boundary bleed-over）：相邻文档之间的信号发生错误混合；
- **全局环绕伪影**（wrap-around artifacts）：序列末尾影响开头，破坏因果性。

这使得高效的 FFT 卷积无法直接应用于实际训练中的打包序列，造成理论复杂度优势（O(N log N)）难以落地。

### 🚀 提出的新方法：RubiConv
论文提出 **RubiConv**，一种专为现代硬件加速器设计的、支持**边界感知**（boundary-respecting）的高效卷积算法，核心创新如下：

1. **基于 Bailey’s 4-step FFT 的重构**  
   继承其适合硬件并行执行的优点（通过 GEMM 实现），但针对打包场景进行根本性改造。

2. **引入块对角 DFT 矩阵（Block-Diagonal DFT Matrix）**  
   在第二步 DFT 中使用 `M2 = BlockDiagonal[F_{m1}, ..., F_{mn}]`，确保每个文档独立进行频域变换，防止跨文档信息混合。

3. **自适应排列与零填充策略**  
   - 对每篇文档补 `min(L_i, L_f) - 1` 个零以保证线性卷积；
   - 将所有文档长度统一填充至 `k` 的倍数（典型值 `k=256`），便于二维 reshape。

4. **单次并行计算而非迭代处理**  
   不像传统方法需逐个处理文档，RubiConv 可在一个张量上完成全部文档的并行卷积，极大提升了吞吐效率。

### 🔍 相比现有方法的优势

| 方法 | 缺陷 | RubiConv 改进 |
|------|------|---------------|
| 标准 FFT 卷积 | 引起 wrap-around 和文档混叠 | 显式隔离文档边界 |
| 迭代处理（loop over docs） | 丧失并行性，速度慢 | 全部文档并行处理 |
| 统一填充至最大长度 | 浪费内存与计算资源 | 按需最小化填充 |
| 忽略边界 | 数据污染、隐私风险 | 数学上严格隔离 |

> ✅ **最终效果**：首次实现**理论高效性**（接近 FFT 复杂度）与**实践高性能**（充分利用 GEMM 并行）的统一。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **FineWeb-Edu 数据集采样**：从 1.3T token 的 FineWeb-Edu 中抽取 10B token 样本；
- 使用 **Gemma 3 tokenizer** 分词，构建真实世界文档长度分布；
- 实验中生成合成数据，但文档长度按该经验分布采样，模拟真实打包场景。

### ⚙️ 实验设置
- **硬件平台**：4 块 TPU v4 或 TPU v3 设备；
- **输入格式**：`X ∈ ℝ^(L×D)`，其中 `L` 为总序列长度，`D` 为模型维度；
- **卷积形式**：depthwise convolution，滤波器 `F ∈ ℝ^(L_f×D)`；
- **分片方式**：沿 sequence dimension 分布式 shard；
- **测量方式**：取 5 次运行均值，前 5 次为预热，报告 95% 置信区间。

### 🎯 评估指标
- **运行时间（Runtime）**：单层卷积操作耗时；
- **可扩展性分析**：
  - 随序列长度 `L` 缩放
  - 随模型维度 `D` 缩放
  - 随滤波器长度 `L_f` 缩放
- **准确率（Accuracy）**：在合成任务上的推理表现（用于验证边界保护必要性）

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| `jnp.convolve` | JAX 原生函数，逐文档 mask 后卷积，正确但串行 |
| `jax.lax.conv` | 更优化的卷积内核，仍需迭代处理 |
| **Splash Attention** | 当前最先进的 attention 内核（Pallas 实现），作为性能标杆 |
| RubiConv-CooleyTukey | 自研 O(N log N) 变体，理论最优但硬件不友好 |
| Full-Matrix Method | 显式构造 block-diagonal Toeplitz 矩阵，O(N²)，暴力精确 |

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（Runtime）

#### （1）随序列长度缩放（图2）
- **短序列（< 2^14）**：`jnp.convolve` 因低开销尚可接受；
- **中长序列（> 2^16）**：RubiConv 显著超越所有卷积基线；
- **当 `L > 2^18` 且 `D=1` 时**：**RubiConv 比 Splash Attention 更快**；
- **高维情况（D=1024）**：虽略慢于 Splash，但在超长序列（~260k）下已具备竞争力。

> ✅ 表明 RubiConv 在典型训练长度下具有显著性能优势。

#### （2）随模型维度缩放（图3a）
- RubiConv 随 `D` 扩展良好，优于其他卷积方法；
- 得益于 GEMM 并行优化，在宽模型中保持高效。

#### （3）随滤波器长度缩放（图3b）
- 其他方法随 `L_f` 增大迅速变慢；
- **RubiConv 几乎不受影响**，适合极长滤波器（如 `L_f=32768`）；
- 因其频域乘法天然适配任意长度滤波器。

#### （4）消融实验：不同 `k` 值的影响（图6）
- 最优 `k` 依赖于总序列长度：
  - `L_total ≤ 2^15` → `k=256`
  - `L_total ≥ 2^16` → `k=512`
- 说明存在硬件友好的 tile size，可通过调参进一步优化。

---

### 🧪 功能性验证：边界保护的重要性（图4）

#### 任务设置
| 任务 | 描述 |
|------|------|
| **Noisy Recall** | 文档含目标符号和干扰项，要求回忆目标；测试长期记忆能力 |
| **Associative Retrieval** | 学习实体-属性映射，随机提问；测试结构化推理 |

#### 结果
- **Boundary-respecting Convolution**：
  - 快速收敛，**接近 100% 准确率**
  - 训练稳定，方差小
- **Document-mixing Convolution**（标准 FFT）：
  - 学习严重延迟
  - 最终准确率远低于前者（接近随机猜测）
  - 高方差，不可靠

> ✅ 证明：**忽略文档边界会实质性损害模型能力**，边界保护是功能必需。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **理论高效 ≠ 实际高效**：尽管 FFT 卷积有 O(N log N) 复杂度，但因不兼容打包机制而无法实用。
2. **RubiConv 成功弥合理论与实践鸿沟**：
   - 通过 block-diagonal DFT + 自适应排列，实现**数学正确的边界隔离**；
   - 利用 GEMM 并行化，在 TPU/GPU 上达到极致性能；
   - 虽理论复杂度为 **O(N³/²)**，但因硬件友好，实测最快。
3. **边界保护至关重要**：
   - 不仅影响性能，更可能导致模型学习错误模式；
   - 在多源数据场景下还涉及隐私与安全问题。

### ⚠️ 方法的局限性
1. **理论复杂度非最优**：O(N³/²) 高于 Cooley-Tukey 的 O(N log N)，不适合极端长序列（除非硬件持续优化）；
2. **需要预处理开销**：构造 `T`, `M1`, `M2`, `P1`, `P2` 等结构，但在多层网络中可被摊销；
3. **依赖静态文档划分**：需提前知道 `segment_ids`，动态切分需重新生成元数据。

### 🔮 未来工作方向
1. **推广到 Streaming 场景**：如何在推理阶段支持流式输入下的边界感知卷积？
2. **结合 Selective SSMs**：将 RubiConv 与 Mamba 类架构融合，打造兼具选择性和长程建模能力的新范式；
3. **自动 `k` 调优系统**：根据 batch 分布动态选择最优 tile size；
4. **支持稀疏与量化版本**：降低内存占用，推动边缘部署。

---

## ✅ 总结

> **RubiConv 是首个真正实现“高效 + 正确”的打包序列卷积算法**。它不仅解决了长期存在的 **FFT 边界混叠问题**，而且通过精巧的矩阵分解设计，充分发挥现代 AI 加速器的并行潜力，在真实训练场景中实现了对 attention 和传统卷积的双重超越。这项工作为 **long convolutional models 的规模化落地扫清了最后一道障碍**，有望推动下一代高效序列建模架构的发展。

</details>

---

### 9. [Arcane: An Assertion Reduction Framework through Semantic Clustering and MCTS-Guided Rule Exploring](https://arxiv.org/abs/2605.10107)

**Authors**: Hongqin Lyu, Yonghao Wang, Zhiteng Chao, Tiancheng Wang, Huawei Li  
**Category**: cs.AI  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.10107v1  

#### Abstract
Assertion-based Verification (ABV) is essential for ensuring that hardware designs conform to their intended specifications. However, existing automated assertion-generation approaches, such as LLM-based frameworks, often generate large numbers of redundant assertions, which significantly degrade si...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Arcane: An Assertion Reduction Framework through Semantic Clustering and MCTS-Guided Rule Exploring*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在硬件设计的功能验证中，**Assertion-Based Verification (ABV)** 是提升可观测性和调试效率的关键手段。然而，无论是传统的模板生成方法还是近年来基于 **Large Language Models (LLMs)** 的自动化断言生成框架，都会产生大量**语义冗余**的断言（assertions）。这些冗余不仅增加了仿真开销（simulation overhead），还降低了验证效率。

例如：
- HARM 生成的断言中约有 96% 是冗余的；
- LLM 方法仍会产生 20%-30% 的冗余断言。

因此，本文旨在解决：**如何在不损失错误检测能力的前提下，高效地减少冗余断言数量，从而加速仿真过程**。

---

### 🚀 提出的新方法与创新思路

作者提出了 **Arcane** —— 一个高效的断言精简框架，其核心创新包括：

#### （1）**两阶段聚类策略（Coarse-to-Fine Clustering）**
- **第一层：BERT-guided 粗粒度语义分类**
  - 将 SVA 断言转换为自然语言描述；
  - 利用预训练 BERT 模型提取语义向量，通过 cosine similarity 进行初步分组。
- **第二层：Lasso-based 细粒度行为相似性分析**
  - 构建每个断言对应的 **Büchi Automaton**；
  - 采样 lasso 路径（有限前缀 + 循环），利用 Jaccard Index 衡量接受路径的一致性；
  - 结合 BERT 和 lasso 的相似性得分进行统一距离计算，并使用 **DBSCAN** 完成最终聚类。

> ✅ 优势：避免仅依赖文本相似性导致的功能相反断言被误聚类（如 `A→B` vs `A→¬B`）。

#### （2）**MCTS-Guided 规则探索机制**
- 将断言规约建模为一个 **Deterministic MDP**：
  - **State**: 当前断言集合的状态；
  - **Action**: 应用五种预定义的逻辑等价规约规则；
  - **Transition**: 规则应用后更新断言集；
  - **Reward**: 基于断言数和原子谓词数的减少量（Δ|S| + Δ|AP|）。
- 使用 **Monte Carlo Tree Search (MCTS)** 在庞大的规则组合空间中智能搜索最优规约路径。
  - 引入 UCT 策略平衡探索与利用；
  - 支持早期剪枝和收敛判断，显著降低搜索成本。

> ✅ 优势：相比穷举或贪心策略，能更有效地找到高收益的规约序列，实现最大化的冗余消除。

---

### 🔍 相比现有方法的优势

| 方面 | Arcane 的优势 |
|------|----------------|
| **冗余处理方式** | 不是简单删除重复项，而是基于语义一致性进行结构性合并与简化 |
| **语义保真性** | 所有规约操作均保证逻辑等价或蕴含关系，确保形式覆盖（formal coverage）不变 |
| **可扩展性** | 两阶段聚类大幅缩小搜索空间；MCTS 避免暴力枚举 |
| **通用性** | 支持多种来源断言（HARM、LLM 等），适用于异构场景 |

---

## 2. 核心实验方法和设置

### 📊 数据集
使用 **AssertionBench [20]**，包含：
- **112 个硬件设计模块**；
- 每个模块提供 RTL 代码、波形轨迹及对应断言；
- 断言来自两类生成器：
  - **HARM**（基于模式挖掘的传统方法）
  - **LLM-based generator**（代表先进 AI 生成技术）

该基准具有良好的真实性和多样性，适合评估断言精简效果。

---

### ⚙️ 实验设置

| 项目 | 设置说明 |
|------|----------|
| **工具链** | 
| Formal Verification | Cadence JasperGold (v21.12.002) |
| Simulation | Synopsys VCS (v2016.06) |
| LTL 转换与模型检验 | SPOT 工具库用于 Büchi automata 比较 |
| 硬件平台 | Intel Xeon Gold 6148 @ 2.40GHz, 629GB RAM |
| 并行化 | 使用 64 线程加速 pairwise similarity 计算 |

| 超参数配置 | 数值 |
|-----------|------|
| Lasso 样本数 | 500 条路径/断言 |
| 相似性权重 α (BERT) / β (Lasso) | 0.4 / 0.6 |
| 统一相似阈值 | 0.85（用于聚类） |

---

### 🎯 评估指标（Evaluation Metrics）

| 指标 | 含义 | 目的 |
|------|------|------|
| **N** | 断言数量 | 衡量精简程度 |
| **PC (Proof Core)** | 形式验证所需的最小逻辑单元 | 验证是否保持 formal coverage |
| **ER (Error Detection Rate)** | 在 mutation testing 中捕获缺陷的比例 | 反映实际验证质量 |
| **PT (Processing Time)** | 断言规约耗时 | 评估一次性开销 |
| **RT (Running Time)** | VCS 仿真运行时间 | 衡量最终性能增益 |
| **Reduction Ratio** | (1 - N_arcane / N_orig) × 100% | 主要性能指标 |
| **Speedup** | RT_orig / RT_arcane | 仿真加速比 |
| **DBI (Davies-Bouldin Index)** | 聚类质量评价指标（越低越好） | 分析聚类有效性 |

---

### 🔀 基线方法对比
文中虽未直接列出多个外部 baseline，但通过以下方式进行比较：
- 与原始未规约断言集对比（ablation 性质）；
- 对比不同聚类策略（纯 Lasso vs BERT+Lasso）；
- 展示 MCTS 搜索的有效性（通过奖励路径选择）；
- 间接体现优于传统手工或静态规则方法（因后者无法处理复杂语义交互）。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（见 Table III）

| 设计模块 | 断言减少率 | PC 保持？ | ER 保持？ | 仿真加速比 |
|--------|------------|-----------|-----------|-------------|
| ca_prng | **76.2%** | ✅ 92.0% → 92.0% | ✅ 3.2% → 3.2% | **6.1x** |
| control_unit | 69.5% | ✅ | ✅ | 3.46x |
| eth_cop | 74.5% | ✅ | ✅ | 3.42x |
| eth_receivecontrol | 70.3% | ✅ | ✅ | 2.83x |
| MAC_rx_ctrl | 68.2% | ✅ | ✅ | 2.67x |
| MAC_tx_ctrl | 71.9% | ✅ | ✅ | 3.68x |

> ✅ **所有设计中，PC 和 ER 完全未下降**，证明语义完整性完全保留。

---

### 📊 整体分布表现（Figure 5）
- 对全部 112 个设计运行 Arcane：
  - HARM 生成断言：平均减少率达 **~78%**
  - LLM 生成断言：平均减少率达 **~71%**
- 几乎所有电路都实现了超过 **68% 的断言压缩率**

> 💡 表明 Arcane 在多样化设计上具备高度稳定性和普适性。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）聚类策略对比（Table IV）：**"L" vs "B+L"**

| 设计模块 | 纯 Lasso 时间(s) | BERT+Lasso 时间(s) | 加速倍数 | DBI 变化 |
|--------|------------------|--------------------|---------|----------|
| ca_prng | 144.45 → 31.93 | **4.52x** | +0.0158 |
| control_unit | 1070.55 → 161.83 | **6.61x** | +0.0114 |
| MAC_tx_ctrl | 43321 → 1274.51 | **33.99x** (~34x) | +0.0207 |

> ✅ **BERT 预分类极大提升了聚类效率（最高达 34x 加速）**  
> ⚠️ DBI 略微上升（<0.023），表明聚类质量仅有轻微下降，可接受。

#### （2）可视化分析（Figure 6）
- 仅用 BERT 聚类：类别边界模糊、重叠严重；
- 加入 Lasso 行为分析后：簇内更紧凑，簇间分离清晰；
> ✅ 验证了“语义+行为”双维度聚类的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **断言冗余普遍存在且严重影响仿真效率**，尤其在 LLM 生成场景下依然显著；
2. **结合 BERT 语义嵌入与 Büchi automaton 行为分析的两阶段聚类**，能够有效识别功能一致的断言群组，为后续规约奠定基础；
3. **MCTS 能够在巨大规则组合空间中高效导航**，自动发现高回报的规约路径，避免局部最优；
4. **Arcane 在几乎不牺牲任何验证能力的前提下**（PC 和 ER 完全保持），实现了高达 **76.2% 的断言压缩率** 和 **2.6x ~ 6.1x 的仿真加速**；
5. **BERT 预分类 + 局部 lasso 分析** 显著提升了聚类效率，使大规模工业级应用成为可能。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖 LTL 可表达性** | 当前方法需将 SVA 转换为 LTL 进行形式分析，对某些高级 SVA 特性支持有限 |
| **MCTS 初始化敏感性** | 搜索性能受初始规则偏好影响（文中优先启用 Rule 1） |
| **聚类阈值需调参** | 相似性阈值 0.85 是经验设定，可能需针对特定设计调整 |
| **未支持增量规约** | 当新增断言时，目前尚无动态更新机制 |

---

### 🔮 未来工作方向

1. **扩展至更多 SVA 构造**，尤其是涉及 local variables 或复杂 timing 的结构；
2. **引入强化学习优化 MCTS 策略网络**，替代固定先验；
3. **开发在线/增量式断言规约机制**，适应持续集成环境；
4. **与 LLM 断言生成流程端到端整合**，形成“生成-精简-验证”闭环系统；
5. **探索硬件原生加速方案**（如 FPGA-based assertion minimization engine）。

---

## ✅ 总结一句话

> **Arcane 通过“语义聚类 + MCTS 引导规约”的协同设计，在严格保持验证能力的同时，实现了高达 76.2% 的断言压缩和 6.1x 仿真加速，为大规模 ABV 流程提供了轻量、安全、高效的解决方案。**

</details>

---

### 10. [Performance and Energy Trade-Off Analysis of Hierarchical Federated Learning for Plant Disease Classification](https://arxiv.org/abs/2605.08121)

**Authors**: Athanasios Papanikolaou, Athanasios Tziouvaras, Pavlos Stoikos, Apostolos Xenakis, Shameem A Puthiya Parambath, George Floros, Enrica Zereik, Ivan Petrovic, Fabio Bonsignorio  
**Category**: cs.DC  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.08121v1  

#### Abstract
Early detection of plant diseases is critical for improving crop productivity, while it also facilitates the foundations of precision agriculture. Recent advances in distributed deep learning have enabled plant disease classification models to be trained across geographically distributed agricultura...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Performance and Energy Trade-Off Analysis of Hierarchical Federated Learning for Plant Disease Classification*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文针对**大规模物联网（IoT）农业环境中植物病害分类系统部署所面临的挑战**，重点解决以下问题：
- **通信开销大**：传统集中式深度学习需要将大量传感器数据上传至云端，导致高延迟和网络拥塞。
- **边缘设备资源受限**：农业现场的IoT节点通常计算能力弱、电池供电，难以支持高能耗模型训练。
- **缺乏对性能与能效权衡的系统性分析**：现有研究多关注准确率，忽视了Energy、Latency等实际部署中的关键约束。

### 🚀 提出的新方法与创新思路
1. **Hierarchical Federated Learning 架构用于植物病害分类**
   - 将任务分解为两个阶段：
     - 第一阶段：识别作物类型（如苹果、番茄、玉米）
     - 第二阶段：在对应作物上进行病害检测（如Apple scab、Tomato mosaic virus）
   - 这种分层结构减少了类间混淆，并允许每个子模型专注于特定语义空间，提升泛化性和效率。

2. **提出 Power- and Energy-Aware Optimization Framework**
   - 引入统一优化框架，联合考虑：
     - 预测性能（F1-score, Accuracy）
     - 总能量消耗（Total Energy Consumption）
     - 执行时间（Execution Time）
   - 支持两种建模方式：
     - **加权目标函数**：`L(c) = λ₁E(c) + λ₂T(c) + λ₃(1−F1(c))`
     - **约束优化形式**：最大化F1-score，满足 `E(c) ≤ E_max`, `T(c) ≤ T_max`

3. **定义 Energy Efficiency Metric η(c) = F1(c)/E(c)**
   - 提供直观可解释的“每单位能耗获得的分类性能”指标，便于跨配置比较。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **架构设计** | 分层FL比Flat FL更符合现实农业场景，降低噪声干扰，提高训练稳定性 |
| **评估维度** | 不仅看Accuracy，还量化Energy、Time，更适合边缘部署决策 |
| **灵活性** | 可通过调整λ或约束条件适配不同应用场景（节能优先 vs 性能优先） |
| **实用性** | 结果直接指导实际系统选型：例如选择轻量模型+合适aggregator以节省能源 |

---

## 2. 核心实验方法和设置

### 📁 数据集
- **PlantDoc Dataset**  
  - 包含 **230,701张RGB叶片图像**
  - 覆盖 **14种作物类别** 和 **38个作物-病害组合标签**
  - 每个样本同时标注作物种类和健康状态
- **数据划分**：70%/15%/15% 的 train/validation/test 划分
- **非独立同分布模拟**：采用 deterministic i.i.d., stratified, disjoint 分割策略，确保客户端数据平衡且互斥

### ⚙️ 实验设置
| 参数 | 配置 |
|------|------|
| **Clients / Rounds / Local Epochs** | 10 客户端 / 30 轮通信 / 每轮 5 个本地epoch |
| **Backbone Models** | EfficientNet-B0, ResNet-50, MobileNetV3-Large |
| **Aggregation Strategies** | FedAvg, FedProx, FedAvgM |
| **Input Size / Batch Size** | 224×224 / 64 |
| **Optimizer / LR / WD** | Adam / 1e-4 / 1e-4 |
| **Loss Function** | Cross-Entropy + Label Smoothing (0.1) |
| **Hardware Platform** | NVIDIA RTX 6000 Ada Generation |

### 🌪️ 使用的五种Use Case（数据增强模拟真实环境扰动）
| UC | 模拟场景 |
|----|--------|
| UC1: SunnyAngle | 强光、斜视角、轻微透视变化、旋转、软阴影 |
| UC2: OvercastNoise | 昏暗低对比度、轻微去饱和、高斯噪声 |
| UC3: Defocus | 轻微模糊、变焦抖动、运动模糊、浅景深失焦 |
| UC4: JPEGandCast | JPEG压缩伪影、白平衡漂移引起的色偏 |
| UC5: OffCenter | 偏中心裁剪、重新居中、曝光/对比度变化 |

### 🎯 评估指标
| 类别 | 指标 |
|------|------|
| **预测性能** | Accuracy, Recall, Precision, F1-score |
| **系统效率** | Total Execution Time (秒), Total Energy Consumption (瓦时, Wh) |
| **综合指标** | Energy Efficiency Score: `η(c) = F1(c)/E(c)` |

### 🔁 基线方法对比
- **不同Backbone之间的横向对比**：
  - ResNet-50（高性能但耗能高）
  - EfficientNet-B0（均衡型）
  - MobileNetV3-Large（轻量高效）
- **不同Federated Aggregation策略对比**：
  - **FedAvg**：标准参数平均
  - **FedProx**：引入正则项应对异构数据
  - **FedAvgM**：全局动量机制改进收敛

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自Table II）

| Model | Aggregator | F1-score | Energy (Wh) | Time (s) | η = F1/Energy |
|-------|-----------|----------|-------------|----------|----------------|
| **ResNet-50** | **FedAvg** | **0.9062** ✅ | 575.12 ❌ | 13315.68 ❌ | 0.002 |
| ResNet-50 | FedProx | 0.8942 | 372.12 | 12291.58 | 0.002 |
| ResNet-50 | FedAvgM | 0.8840 | 461.02 | 12147.10 | 0.002 |
| EfficientNet-B0 | FedAvg | 0.8535 | 163.39 | 4216.38 | 0.005 |
| EfficientNet-B0 | FedProx | 0.8429 | 218.76 | 5503.40 | 0.004 |
| EfficientNet-B0 | FedAvgM | 0.8366 | 257.24 | 5121.62 | 0.003 |
| **MobileNetV3-Large** | **FedProx** | 0.8583 | **142.73** ✅ | 4446.40 | **0.006** ✅ |
| MobileNetV3-Large | FedAvg | 0.8652 | 176.68 | 5112.66 | 0.005 |
| MobileNetV3-Large | FedAvgM | 0.7992 | 165.90 | **4202.80** ✅ | 0.005 |

> ✅ 表示该列最优；蓝色=最高性能，绿色=最低能耗+最高η，橙色=最短执行时间

### 🔍 对比结果分析

#### ✅ 最佳性能配置：
- **ResNet-50 + FedAvg**：达到最高的F1-score（0.9062），适合对精度要求极高的场景。
- 缺点：能耗高达 **575.12 Wh**，执行时间最长（>13k秒），不适合边缘部署。

#### ✅ 最佳能效配置：
- **MobileNetV3-Large + FedProx**：
  - F1-score 达到 **0.8583**（接近ResNet-50水平）
  - 能耗仅为 **142.73 Wh**（约为ResNet-50的25%）
  - **能量效率η最高（0.006）**
  - 是**资源受限环境下的首选方案**

#### ✅ 最快执行时间：
- **MobileNetV3-Large + FedAvgM**：总耗时 **4202.8秒**，是所有配置中最短的。
- 但其F1-score下降明显（0.7992），牺牲较多准确性。

#### ⚖️ 折中选择：
- **EfficientNet-B0 + FedAvg**：在性能（F1=0.8535）、能耗（163.39 Wh）、效率（η=0.005）之间取得良好平衡。

### 📈 消融实验与趋势观察（Figure 2 验证损失曲线）
- **FedAvg 收敛最快、最终loss最低** → 预测性能最佳
- **FedProx 收敛稳定，略逊于FedAvg** → 在异构环境下表现可靠
- **FedAvgM 表现最差**：尤其在MobileNetV3-Large上出现明显性能退化，可能因动量机制不适应轻量模型动态

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **不存在“全能冠军”配置**：必须根据部署需求在 Performance、Energy、Latency 之间做出权衡。
2. **ResNet-50 + FedAvg 提供最强预测能力**，但代价高昂，适用于云侧或高性能边缘服务器。
3. **MobileNetV3-Large + FedProx 是最佳节能选择**：
   - 在仅损失约5% F1的情况下，节省超过75%的能量
   - 特别适合电池驱动的田间IoT设备
4. **Hierarchical FL 架构有效提升训练稳定性与语义一致性**，优于单一扁平分类器。
5. **提出的 energy-aware framework 可灵活适配多种部署场景**，支持基于约束或权重的自动化配置选择。

### ⚠️ 方法的局限性
- **仿真环境限制**：实验基于模拟FL环境，未完全反映真实无线通信延迟、丢包、设备故障等问题。
- **静态权重设定**：当前优化框架使用固定λ或Emax/Tmax，缺乏在线自适应调节能力。
- **未探索更多backbone或aggregator组合**：如Vision Transformers、FedNova等新兴方法未纳入比较。
- **Plant Classifier被外部化处理**：完整pipeline的影响未计入，虽不影响相对比较，但低估端到端成本。

### 🔮 未来工作方向
1. **开展敏感性分析（Sensitivity Analysis）**：
   - 系统研究λ₁, λ₂, λ₃的变化如何影响最优配置选择
   - 构建可视化工具辅助决策
2. **扩展至动态环境下的自适应FL**：
   - 根据实时能耗反馈动态切换模型或aggregator
   - 引入pruning、quantization进一步压缩模型
3. **结合硬件感知优化（Hardware-aware Optimization）**：
   - 将具体SoC平台的功耗模型集成进框架
   - 实现真正的“从算法到芯片”的联合设计
4. **部署到真实农业IoT测试床**验证理论结果的实际有效性

---

## 总结一句话
> 本文首次系统地揭示了**Hierarchical Federated Learning在植物病害分类中的性能-能耗权衡规律**，提出了一个实用的能量感知优化框架，并证明：**轻量模型（如MobileNetV3-Large）配合FedProx可在几乎不损失诊断准确性的前提下大幅降低能耗，是面向边缘农业AI的理想选择**。

</details>

---

### 11. [MegaScale-Omni: A Hyper-Scale, Workload-Resilient System for MultiModal LLM Training in Production](https://arxiv.org/abs/2605.08962)

**Authors**: Chunyu Xue, Yangrui Chen, Jianyu Jiang, Ningxin Zheng, Junda Feng, Jingji Chen, Shixiong Zhao, Shen Yan, Yi Lin, Lei Shi, Zanbo Wang, Lishu Luo, Faming Wu, Haibin Lin, Xin Liu, Yanghua Peng, Quan Chen  
**Category**: cs.DC  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.08962v1  

#### Abstract
As the foundational component of versatile AI applications, training an multimodal large language model (MLLM) relies on multimodal datasets with dynamic modality mixture proportions and sample length distributions. However, existing MLLM systems remain inefficient under dynamic workloads, due to st...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：MegaScale-Omni**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的多模态大语言模型（MLLM）训练系统在**动态工作负载**下效率低下，主要原因包括：
- **静态耦合资源分配与并行策略**：现有系统将 encoder 和 LLM backbone 的资源分配与并行化策略紧密绑定，无法适应训练过程中不断变化的模态比例（如图像、文本、音频）和样本长度分布。
- **工作负载失衡**：不同模态的数据长度差异巨大（例如，音频序列平均是文本的17.6倍），导致 encoder 层计算不均衡，引发设备空转或内存溢出（OOM）。
- **缺乏弹性调度机制**：传统方法难以应对多阶段训练中模态混合比例的动态调整（如从纯文本预训练过渡到图文混合微调）。

### **提出的新方法与核心思路**
论文提出了 **MegaScale-Omni**，一种面向工业级、超大规模、支持动态工作负载的 MLLM 训练系统，其核心思想为 **encoder-LLM multiplexing**（编码器-语言模型复用）。三大创新如下：

#### **(1) 解耦的并行化策略（Decoupled Parallelism）**
- **Encoders 使用 Long-Short Sequence Parallelism (LSSP)**  
  动态地在每个 microbatch 中根据样本长度切换 DP（短序列）与 Ulysses SP（长序列）模式，避免对所有节点进行统一划分带来的负载不均。
- **LLM Backbone 使用 Full-Fledged 5D Parallelism**  
  继承成熟的单模态训练技术（DP, TP, PP, SP, EP），最大化 LLM 部分的训练效率。
- **通信高效的并行布局**  
  在物理拓扑上将 encoder 与各 LLM pipeline stage 共置，并通过 intra-node 高带宽连接优化通信路径。

#### **(2) 统一的共置表示与联合流水线（Unified Colocation & Joint Pipeline）**
- 提出 **EncoderAnchor** 抽象，将 encoder 表示为可插拔的“锚点”，非侵入式地集成进 LLM 流水线代码，无需修改原有框架逻辑。
- 设计 **workload-resilient joint pipeline**，采用**均匀按需插入**（uniform on-demand insertion）策略，在保持流水线结构稳定的同时响应负载变化，防止因 encoder 膨胀导致 pipeline bubbles 扩大。

#### **(3) 工作负载平衡技术（Workload Balancing）**
- **去中心化的分组重排序（Decentralized Grouped Reordering）**  
  在分布式数据加载器中基于网络局部性分组，组内交换元数据后重排序样本，实现跨 encoder 的负载均衡，同时避免全局重排序的高通信开销。
- **自适应分片与对称派发（Adaptive Sharding & Symmetric Dispatching）**  
  在从 encoder 到 LLM 的 embedding resharding 阶段，根据目标 SP 类型（Ulysses 或 CP）选择合适的分片策略，并利用 all-reduce 缓冲区聚合以减少通信瓶颈。

### **相比现有方法的优势**
| 方法 | 主要缺陷 | MegaScale-Omni 的改进 |
|------|--------|-----------------------|
| **Megatron-LM** | 将 encoder 视为嵌入层，造成 PP0 成为瓶颈 | 解耦并行，分散 encoder 负载 |
| **DistTrain / Disaggregation** | 固定拆分 encoder 与 LLM，无法适应动态负载 | 支持动态资源复用与联合调度 |
| **Optimus** | 假设静态工作负载，依赖合成输入优化 | 支持真实生产环境中的动态混合 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **视觉模态**：`OpenImages`, `RefCOCOg`（开源）
- **音频模态**：`LibriSpeech`, `GigaSpeech`（开源）
- **文本模态**：`BytedLong`, `BytedOCR`（公司内部生产数据集，最长达 512K tokens）
- **混合方式**：采用 **hybrid packing**，跨模态打包样本以对齐序列长度，提升利用率。

### **实验设置与评估指标**
#### **硬件平台**
- **Cluster-A**：512 GPU 集群，每节点 8×GPU，NVLink + 8×400G InfiniBand RDMA
- **Cluster-B**：数千 GPU 的超大规模生产集群（保密配置）

#### **模型与工作负载（见 Table 1）**
| 名称 | Encoder | LLM | Batch Size | Seq Len |
|------|--------|-----|------------|---------|
| Workload-A | ViT-1B | LLaMA-12B | 32 | 16384 |
| Workload-B | ViT-2.4B | LLaMA-70B | 64 | 16384 |
| Workload-C | ViT-10B | LLaMA-70B | 128 | 8192 |
| Workload-D | ViT-10B | GPT-175B | 256 | 8192 |

额外测试了三模态任务（图像+音频+文本），使用 USM 作为语音 encoder。

#### **评估指标**
- **吞吐量（Throughput）**：每秒处理的 token 数
- **MFU（Model FLOPs Utilization）**
- **内存占用（Memory Footprint）**
- **端到端稳定性与容错能力**

### **基线方法对比**
1. **Megatron-LM**：标准 unimodal-like 架构，encoder 接入第一 stage
2. **Megatron-Dist**：encoder 与 LLM 分离部署（disaggregation）
3. **AutoParallel**：基于 Alpa 的自动 pipeline 构建系统
4. **Optimus**：支持 bubble exploitation 的 MLLM 系统，针对静态负载优化

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 在动态混合比例（image:text 从 1:9 到 9:1）下，MegaScale-Omni 实现 **1.27× ~ 7.57× 吞吐提升**，最高达 **7.57×**（Workload-C, 256 GPUs）。
- 在序列长度扩展至 32K 时仍保持稳定，相较基线最高提速 **6.88×（16K）和 4.72×（32K）**。
- 最高 MFU 达到 **38.2%**，较基线平均提升 **~23%**。

### **与基线方法的对比结果**
| 场景 | 最佳加速比 | 说明 |
|------|-----------|------|
| Workload-A (64 GPUs) | 1.54× | 图像比例上升时优势明显 |
| Workload-B (128 GPUs) | 3.30× | 大模型下仍高效 |
| Workload-C (256 GPUs) | **7.57×** | 高并行度下解耦优势凸显 |
| Workload-D (512 GPUs) | 1.98× | 所有基线在 9:1 出现 OOM |
| 三模态任务 | 最高 **2.01×** | 对复杂动态负载鲁棒 |

> ⚠️ 注意：当图像比例过高时，Megatron-LM 等出现严重 OOM，而 MegaScale-Omni 可正常运行。

### **消融实验结果**
- **移除 multiplexing（即恢复为 Megatron-LM 模式）**：
  - 吞吐下降 **52.9% ~ 60.6%**
- **关闭 workload balancing（去分组重排序）**：
  - 吞吐下降 **45.9% ~ 50.7%**
- **禁用 LSSP（仅用 DP 或 SP）**：
  - 下降约 **18.7%**
- **关闭其他优化（tuning, offloading, operator overlap）**：
  - 各下降 **16%~19%**

👉 结论：**encoder-LLM multiplexing 是最核心的增益来源**，其次是 workload balancing。

---

## **4. 关键结论和发现**

### **主要发现**
1. **动态工作负载是 MLLM 生产训练的核心挑战**，必须打破 encoder 与 LLM 的静态耦合。
2. **解耦并行 + 资源共置 + 弹性调度** 的组合能有效应对多模态数据的异构性和动态性。
3. **LSSP 和 grouped reordering 显著缓解了 variable-length 导致的负载倾斜**。
4. **MegaScale-Omni 在千卡级别集群上实现了稳定的高 MFU 和低 bubble 开销**，已支撑公司内部多个大型 MLLM 项目。

### **方法的局限性**
- **实现复杂度较高**：需要深度定制训练框架，对工程团队要求高。
- **依赖特定通信原语**：如 all-reduce 缓冲区用于 symmetric dispatching，可能受限于底层通信库支持。
- **目前未开放源码**：作为企业级系统，部分细节未公开，学术复现难度较大。

### **未来工作方向**
- 支持更灵活的 **dynamic pipeline adaptation**，进一步自动化应对极端负载波动。
- 探索 **cross-modality curriculum learning** 与系统调度的协同优化。
- 将 MegaScale-Omni 的架构思想推广至 **MoE-based MLLM** 和 **inference serving** 场景。
- 开发轻量化版本，降低中小规模场景的部署门槛。

--- 

> ✅ **总结一句话**：  
> **MegaScale-Omni 通过 encoder-LLM multiplexing 实现了解耦、弹性和高效的 MLLM 训练架构，在真实动态负载下实现了高达 7.57× 的吞吐提升，为工业级多模态大模型训练提供了坚实基础。**

</details>

---

### 12. [PoHAR: Understanding Hyperlocal Human Activities with Pollution Sensor Networks](https://arxiv.org/abs/2605.09434)

**Authors**: Prasenjit Karmakar, Karthik Reddy, Sandip Chakraborty  
**Category**: cs.DC  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.09434v1  

#### Abstract
Low-cost air quality sensors are becoming ubiquitous in our daily lives as public awareness of air pollution continues to grow, and people take measures to monitor and improve the air they breathe indoors. Besides the standard operation of these sensors, fluctuations in environmental parameters can ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《PoHAR: Understanding Hyperlocal Human Activities with Pollution Sensor Networks》总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

传统的人类活动识别（HAR）系统依赖于摄像头、麦克风、RF信号或可穿戴设备等模态，存在以下问题：
- **隐私泄露风险**（如音频视频监控）
- **硬件成本高或部署复杂**（如毫米波雷达）
- **难以扩展到家庭场景**（如需要佩戴传感器）

此外，现有分布式传感器网络在处理室内空气污染数据时面临：
- 如何从低功耗、资源受限的传感器中提取有意义的行为特征
- 如何动态识别受特定活动影响的“活动相关传感器组”（activity-affected sensor groups）
- 在无中心节点的情况下实现高效、一致的数据共享与协同推理

---

### **提出了什么新方法或新思路**

本文提出 **PoHAR** 框架，一个面向低功耗空气污染传感器网络的分布式、隐私保护型超本地人类活动识别系统。其核心创新包括：

#### ✅ **1. Conflict-free Replicated Set (Set-CvRDT)**  
- 设计了一种基于 **CRDT**（Conflict-free Replicated Data Type）的去中心化数据共享机制
- 支持在网络不稳定（UDP丢包、延迟）环境下实现多节点间的一致状态同步
- 使用三个只增集合（grow-only sets）：`add-set`, `rem-set`, `main-set`，确保最终一致性

#### ✅ **2. Pollution-aware Hierarchical Clustering on ESP32**
- 首次将自监督学习（SSL）嵌入 + 分层聚类算法部署在 **ESP32 微控制器** 上
- 利用 **time-frequency consistency (TF-C)** 模型从空气质量时间序列中提取语义嵌入
- 提出一种基于距离边界（distance bounds）的分区式聚合层次聚类算法，动态识别“被同一活动影响”的传感器集群

#### ✅ **3. Leader-based Group Inference for Hyperlocal HAR**
- 每个检测到的传感器群组独立选举一个 **RAFT 协议 leader**
- 由 leader 聚合本组内传感器的 SSL embeddings，并运行轻量级 ML 模型进行本地化活动识别
- 实现细粒度、并行化的多区域活动感知（如厨房炒菜 vs 客厅打扫）

---

### **相比现有方法的优势**

| 维度 | 传统方法局限 | PoHAR优势 |
|------|--------------|-----------|
| **隐私性** | 视频/音频易侵犯隐私 | 使用非侵入式 AQI 数据，天然隐私友好 |
| **可扩展性** | 可穿戴设备难普及 | 商品级低成本传感器即可部署 |
| **能耗与计算** | 中心化聚合通信开销大 | 边缘侧 SSL + 聚类 + 推理，降低带宽和功耗 |
| **动态适应性** | 固定拓扑或静态分组 | 动态识别受影响传感器群，支持并发活动 |
| **容错能力** | 单点故障风险高 | RAFT + CvRDT 提供强一致性与故障恢复 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- 自建真实世界部署的 **室内空气质量传感器网络**
- 覆盖 **30个多样化场景**：公寓、实验室、教室、食堂、住宅等
- 时间跨度：**6个月**（涵盖冬夏两季）
- 总采集样本数：**8910万条**
- 总监测时长：**13,646小时**
- 手动标注事件：**3957个活动实例**，来自24名参与者
- 涵盖活动类型：
  - 日常行为（进出、开关风扇/空调）
  - 清洁、聚集、饮食
  - **烹饪活动**：5类（煮、炸、蒸、煎、混合），11种食物（鱼、蛋、米饭、咖喱等）

传感器采集参数：
- CO₂, VOCs, PM2.5/10, 温湿度

---

### **实验设置**

- **硬件平台**：ESP32 微控制器（资源受限嵌入式设备）
- **通信协议**：UDP（模拟不可靠信道）
- **SSL模型架构**：基于 Time-Frequency Consistency (TF-C) 的轻量神经编码器
- **ML分类器**：Decision Tree, Random Forest, Extra Trees, Gaussian NB, MLP（通过 emlearn 部署于边缘）
- **聚类算法**：Partition-based Agglomerative Hierarchical Clustering（参考 PACK 算法 [23]）
- **共识协议**：RAFT 用于 leader election 和 group coordination

---

### **评估指标**

| 类别 | 指标 |
|------|------|
| **系统性能** | Latency (μs/ms/s), Memory Usage (KB), Power Consumption (mW) |
| **数据一致性** | Convergence Time, Fault Tolerance, Commutativity/Associativity |
| **聚类效果** | Number of Iterations, Cluster Quality (via t-SNE) |
| **模型性能** | Accuracy, Avg./Max Inference Time, Confusion Matrix |
| **能效分析** | Real-time power draw traces during inference |

---

### **基线方法对比**

虽然未直接与其他 HAR 框架端到端比较，但通过模块层面与以下类别方法形成对比：

| 对比方向 | 基线方法 | PoHAR改进 |
|--------|---------|----------|
| **数据共享** | Centralized server, MQTT | Set-CvRDT（去中心、抗丢包） |
| **聚类算法** | HEED, DWEHC, DEEC | SSL + Distance-aware hierarchical clustering（更精准识别动态簇） |
| **推理方式** | Cloud-based ML inference | On-device inference on leader ESP32 |
| **特征提取** | Handcrafted features | Self-supervised embedding (TF-C) |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔹 **Set-CvRDT 数据一致性表现**
- **收敛延迟**：5节点下仅 **90 μs**
- 支持并发更新与频繁操作，具备良好容错性（节点失效后仍可恢复一致状态）
- 满足交换律、结合律，支持任意顺序合并

#### 🔹 **Pollution-aware Clustering 效率**
- **内存占用**：从10节点的 **63KB → 50节点的 228KB**（近线性增长）
- **执行时间**：10节点需 **31ms → 50节点为 320ms**（< 1秒，适合实时）
- **迭代次数**：13轮迭代将50个节点聚合成3个有效簇
- **功率消耗**：平均 **250–360 mW**，峰值短暂上升，符合低功耗设计

#### 🔹 **RAFT Leader Election 表现**
- **选举功耗**：Leader ~650 mW，Follower ~280 mW
- **故障恢复时间**：leader 失效后 **3–6 秒内完成重选**，稳定性高

#### 🔹 **SSL Embedding 质量（t-SNE 可视化）**
- 图4显示不同食物（如Paratha, Fish, Egg, Dal）在嵌入空间中形成 **24个清晰分离的簇**
- 表明 TF-C 模型成功捕捉烹饪行为的语义差异

#### 🔹 **Activity Classification 性能**

##### 📊 **表 I：室内活动识别性能（Top: Random Forest）**

| Model | Accuracy | Avg Time | Max Time |
|-------|----------|----------|----------|
| **Random Forest** | **97.41%** | **34 μs** | 330 μs |
| Extra Trees | 97.36% | 62 μs | 463 μs |
| Decision Tree | 93.56% | **9 μs** | 42 μs |
| Gaussian NB | 48.28% | 356 μs | 3822 μs |
| Neural Network | 49.45% | 89 μs | 438 μs |

> ✅ Tree-based models 明显优于浮点密集型模型

##### 📊 **表 II：烹饪类型识别性能**

| Model | Accuracy | Avg Time |
|-------|----------|----------|
| **Random Forest** | **99.68%** | 16 μs |
| Extra Trees | 99.81% | 43 μs |
| Decision Tree | 97.28% | **6 μs** |

##### 📊 **表 III：食物项目识别性能**

| Model | Accuracy | Avg Time |
|-------|----------|----------|
| **Random Forest** | **99.49%** | 23 μs |
| Extra Trees | 99.68% | 64 μs |
| Decision Tree | 94.17% | **7 μs** |

> ⚠️ 浮点模型（NB, MLP）准确率均低于50%，因 ESP32 缺乏 FPU 支持

#### 🔹 **On-device Power Analysis**
- Tree-based models 推理期间功耗稳定在 **272–274 mW**
- MLP 达到 **308 mW**，且功耗脉冲更长，说明计算负担重
- Random Forest 具备 **短脉冲、快返回 idle 状态** 特性，更适合实时应用

---

### **消融实验结果（隐含分析）**

尽管未明确列出 ablation study，但从多个维度进行了有效性验证：
- **SSL embedding vs 原始数据**：t-SNE 显示嵌入具有强语义结构
- **Set-CvRDT vs 中心化同步**：在 UDP 不可靠传输下仍保持一致性
- **Hierarchical clustering vs 平凡分组**：能正确识别跨房间的污染传播路径
- **Leader-based inference vs 全网统一推理**：仅激活相关传感器组，提升效率与精度

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **空气污染数据可用于高精度、隐私友好的 HAR**
   - 室内活动（尤其是烹饪）会引起显著且可区分的 AQI 波动模式
   - 结合 CO₂、VOCs、PM 等多维信号可构建丰富环境指纹

2. ✅ **自监督学习可在无标签数据上生成高质量嵌入**
   - TF-C 模型无需大量标注即可学习时间-频率一致性特征
   - 嵌入空间自然聚类出不同食物与烹饪方式

3. ✅ **分布式聚类 + 边缘推理是可行且高效的**
   - 在 ESP32 上实现了完整的“感知→嵌入→聚类→推理”链路
   - 支持多活动并发识别，实现真正的 **hyperlocal HAR**

4. ✅ **Tree-based ML 模型最适合 ESP32 部署**
   - 准确率高、延迟极低（< 100 μs）、功耗可控
   - 浮点模型因硬件限制表现差，应避免使用

5. ✅ **Set-CvRDT + RAFT 构成了可靠的去中心化协作基础**
   - 实现了无需中心服务器的状态同步与任务协调
   - 具备良好的容错性和可扩展性

---

### **方法的局限性**

1. ❗ **依赖空气污染物的扩散特性**
   - 若房间密闭性强或通风过快，可能导致信号衰减严重，影响检测灵敏度
   - 不适用于几乎不产生污染的活动（如静坐阅读）

2. ❗ **聚类算法对初始阈值敏感**
   - 合并条件中的距离阈值 θ 需调优，可能影响最终簇数量与质量

3. ❗ **当前仅支持已知活动类型分类**
   - 尚未验证对未知新活动的异常检测能力

4. ❗ **传感器密度要求较高**
   - 要求每个功能区至少有一个传感器，否则无法精确定位活动源

---

### **未来工作方向**

1. ➕ 扩展至更多活动类型（如洗澡、洗衣、吸烟等）
2. ➕ 引入在线增量学习以适应住户习惯变化
3. ➕ 探索联邦学习框架，在保护数据隐私前提下联合训练全局模型
4. ➕ 优化 SSL 模型压缩，进一步降低内存与计算开销
5. ➕ 与 HVAC 系统联动，实现智能通风控制闭环

---

> 🔗 **开源地址**：https://github.com/prasenjit52282/PoHAR  
> 💡 **启示**：该工作展示了如何将 commodity-level 空气质量传感器“变废为宝”，不仅用于环境监测，还可作为新型、可持续、隐私优先的智能感知基础设施。

</details>

---

### 13. [Accelerating Compound LLM Training Workloads with Maestro](https://arxiv.org/abs/2605.10501)

**Authors**: Xiulong Yuan, Hongqing Chen, Jiaxuan Peng, Fan Zhou, Zhixiang Ruan, Zekun Wang, Bo Zheng, Rui Men, Haiquan Wang, Zhipeng Zhang, Langshi Chen, Man Yuan, Jiaqi Gao, Zhengping Qian, Junyang Lin, Yong Li, Wei Lin, Junhua Wang, Jingren Zhou  
**Category**: cs.DC  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.10501v1  

#### Abstract
Compound LLM training workloads-such as knowledge distillation and multimodal LLM (MLLM) training-are gaining prominence. These typically comprise heterogeneous components differing in parameter scale, execution mode (forward-only or full forward-backward), and sequence length. Besides, component ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Accelerating Compound LLM Training Workloads with Maestro》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代大型语言模型（LLM）的训练范式正从单一、同质化的Transformer堆栈转向**复合型训练工作负载（compound LLM training workloads）**，例如：
- **多模态大模型（MLLM）训练**：包含视觉编码器、音频编码器等异构组件。
- **知识蒸馏（knowledge distillation）**：教师模型（inference-only）与学生模型（full forward-backward）协同训练。

这些工作负载具有以下挑战：
- **静态异构性（static heterogeneity）**：不同组件在参数规模、计算强度、内存占用、并行策略需求上差异巨大。
- **动态运行时不规则性（runtime workload irregularity）**：输入数据决定激活路径（如文本样本跳过视觉编码器），导致动态计算图和流水线气泡（pipeline bubbles）。
- 传统框架（如 Megatron-LM）采用统一配置（one-size-fits-all），无法应对上述双重异构性，造成资源浪费和吞吐下降。

### 提出了什么新方法或新思路
作者提出 **Maestro** ——一个面向复合LLM训练的**以“段”为中心（section-centric）的训练框架**，其核心创新包括：

#### （1）**Section 抽象与细粒度资源配置**
- 将整个计算图划分为多个逻辑上的 **Section**，每个 Section 对应功能独立的子模块（如 ViT encoder、LLM backbone、teacher/student 模型）。
- 每个 Section 可独立配置：
  - 并行策略（TP / PP / CP / EP）
  - 微批次大小（micro-batch size）
  - 数据并行度（DP degree）
  - 甚至支持跨 Section 的 **fan-out 执行机制**

> ✅ 优势：打破全局统一配置限制，实现组件级资源优化。

#### （2）**Wavefront Scheduling 动态调度算法**
- 在运行时根据样本将激活的 Sections 进行动态重排序。
- 调度目标是最大化关键路径（critical section，通常是LLM主干）的利用率，避免其因等待其他组件输出而停滞。
- 使用启发式插入算法，在 $O(N^2)$ 时间内生成近似最优执行顺序。

> ✅ 优势：有效消除由数据依赖引起的 pipeline bubbles，提升整体并发性和硬件利用率。

#### （3）**异步非对称消息队列（Asynchronous Asymmetric Message Queue）**
- 支持跨 Section 的张量传输与自动 resharding。
- 基于 **one-sided RDMA** 实现 CPU/GPU 解耦通信，最小化对计算的干扰。
- 支持 M-to-N 通信模式，适应不同 TP/CP 配置下的数据分发。

> ✅ 优势：高效处理异构并行域之间的通信，降低通信开销。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **多模态训练任务**：
  - 包含 **text-only** 和 **text-image** 样本的混合数据集。
  - 序列长度高达 **32K tokens**。
  - 视觉-文本样本比例参考业界实践（如 1:9 或 1:2）。
- **知识蒸馏任务**：
  - 冻结的 **Qwen3.5-400B-A17B** 作为 teacher。
  - 可训练的 **Qwen3-Next-80B-A3B** 作为 student。
  - 使用 KL divergence loss 进行蒸馏。

### 实验设置
- 所有实验基于 **Alibaba Cloud 大规模 GPU 集群**。
- 主要对比基线为 **Megatron-LM**（Maestro 的底层基础），确保低层优化一致。
- 控制变量：**关键 Section（如 LLM 或 student）的 GPU 数量与并行配置保持与基线相同**，额外资源用于辅助 Section（如 ViT 或 teacher）。

### 评估指标
| 指标 | 含义 |
|------|------|
| **End-to-End Token Throughput** | 整体每秒处理的 token 数量（绝对性能） |
| **Per-GPU Token Throughput** | 单卡每秒处理的 token 数量（资源效率） |
| **Relative Efficiency w.r.t Text-only Training** | 相对于纯文本训练的性能损失百分比（衡量 ViT 引入的 overhead 是否被消除） |

---

## 3. 主要实验结果和性能指标

### 多模态训练（Vision-Language Model Training）

| 模型 | 方法 | End-to-End Throughput | Per-GPU Throughput | 相对文本训练效率 |
|------|------|------------------------|---------------------|--------------------|
| Qwen3.5-400B-A17B | Megatron-LM | 1.00× | 1.00× | ~93% |
|                      | **Maestro** | **1.40×** | **1.24×** | **100%** |
| Qwen3-Next-80B-A3B   | Megatron-LM | 1.00× | 1.00× | ~94% |
|                      | **Maestro** | **1.20×** | **1.07×** | **100%** |

> 🔍 **关键发现**：
> - Maestro 成功将 ViT 编码器的计算完全隐藏在 LLM 推理过程中（computation overlapping）。
> - 尽管增加了 12.5% 的 GPU 给 ViT Section，Per-GPU 效率仍显著提升。
> - 达到 **100% 相对效率**，意味着加入视觉模态没有带来任何端到端性能损失。

---

### 知识蒸馏（Knowledge Distillation）

| 设置 | 方法 | End-to-End Throughput | Per-GPU Throughput |
|------|------|------------------------|---------------------|
| Teacher + Student | Megatron-LM | 1.00× | 1.00× |
|                    | **Maestro** | **1.75×** | **1.40×** |

> 🔍 **关键机制支撑**：
> - 利用 **fan-out 机制**：单个 teacher DP rank 可服务多个 student DP ranks。
> - Teacher 可使用更大的 micro-batch size（如 MBS=4），吞吐提升达 **2.6×**，而峰值内存几乎不变。
> - 通过将 teacher 输出层与 student 共置于同一 Section，仅传递 compact hidden states 而非 logits，减少通信量 **62.5×**（vocab_size=250K vs hidden_dim=4K）。

---

### 消融实验（Ablation Studies）
虽然文中未明确列出独立消融表格，但从设计分析中可推断出各模块贡献：
- **Section 分区 + 定制化并行策略** → 解决静态异构性，提升资源配置灵活性。
- **Fan-out 机制** → 显著降低 teacher 所需 GPU 数量，提高资源复用率。
- **Wavefront Scheduling** → 是实现 computation overlapping 的关键，尤其在 MLLM 中消除 pipeline bubbles。
- **异步通信机制** → 减少通信阻塞，实现计算与通信的有效重叠。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **复合型 LLM 工作负载已成为主流趋势**，传统统一配置框架已不再适用。
2. ✅ **静态与动态双重异构性必须同时解决**：
   - 静态：通过 **Section 抽象 + 独立配置** 解决。
   - 动态：通过 **wavefront scheduling** 实现运行时动态调度。
3. ✅ **关键路径主导系统性能**：优化应优先聚焦于瓶颈 Section（如 LLM backbone 或 student model）。
4. ✅ **Maestro 在真实生产环境中已部署数百万 GPU 小时**，验证了其稳定性与实用性。
5. ✅ **平均节省约 40% 的 GPU 资源消耗**，显著降低训练成本。

### 方法的局限性
- 当前调度策略基于 **critical-section-first** 启发式，虽高效但非全局最优。
- 对极端复杂的多分支、多条件激活路径的支持尚未深入探讨（如全模态 omni-modal 场景）。
- Section 划分依赖人工经验或先验知识，自动化划分机制有待研究。
- 依赖高性能网络（如 RDMA），在普通集群上可能难以发挥全部优势。

### 未来工作方向
- 自动化 Section 构建与配置搜索（auto-sectioning & auto-configuration）。
- 更智能的动态调度器，结合强化学习或预测模型进行样本预判。
- 扩展至 MoE、长上下文、推理-训练联合优化等更复杂场景。
- 开源 Maestro 框架，推动社区共建下一代 LLM 训练基础设施。

---

> 📌 **总结一句话**：  
> **Maestro 通过“Section + Wavefront Scheduling”的协同设计，首次系统性解决了复合 LLM 训练中的双重异构性挑战，在真实业务中实现了最高 1.75× 的端到端吞吐提升，并节省约 40% 的 GPU 资源，为下一代 LLM 训练框架提供了重要范式。**

</details>

---

### 14. [ReLibra: Routing-Replay-Guided Load Balancing for MoE Training in Reinforcement Learning](https://arxiv.org/abs/2605.08639)

**Authors**: Chao Jin, Xinming Wei, Yinmin Zhong, Chengxu Yang, Bingyang Wu, Ruidong Zhu, Zili Zhang, Yuliang Liu, Xin Jin  
**Category**: cs.LG  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.08639v1  

#### Abstract
Load imbalance is a long-standing challenge in Mixture-of-Experts (MoE) training and is exacerbated in reinforcement learning (RL) for LLMs, where hot experts can shift frequently across micro-batches. Existing MoE training systems rely on historical loads to predict future expert demand, making the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ReLibra: Routing-Replay-Guided Load Balancing for MoE Training in Reinforcement Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **MoE（Mixture-of-Experts）在强化学习（RL）训练中的负载不均衡问题**：
  - 在MoE架构中，token通过router动态路由到不同的expert进行处理，导致某些expert成为“hot experts”，造成GPU负载不均，形成straggler，降低训练效率。
  - 在**RL训练场景下**，这种不平衡更加严重且波动剧烈：不同micro-batch之间hot experts频繁切换，而传统基于历史负载预测的方法（如EPLB、PopFetcher）无法有效应对这种快速变化。

### **提出的新方法与创新思路**
- **ReLibra**：一种专为MoE在RL训练中设计的细粒度负载均衡系统，其核心是利用**Routing Replay**机制实现精准的负载调度。
  
#### **关键创新点**：
1. **利用Routing Replay机会**：
   - 在RL流程中，rollout阶段生成response时已经确定了每个token的expert路由路径。
   - ReLibra将这些**已知的routing信息**用于训练前的负载规划，从而摆脱对“未来负载预测”的依赖，实现**先知式（oracle-like）调度**。

2. **分层负载均衡设计（Timescale Decomposition）**：
   - 将两种负载均衡机制按时间尺度分离，匹配集群网络层次结构：
     - **Inter-batch timescale（跨batch）**：执行**Expert Reordering**（专家重排序），用于粗粒度跨节点平衡。
     - **Intra-batch timescale（微批内）**：执行**Expert Replication**（专家复制），用于细粒度吸收micro-batch级别的负载波动。

3. **高效算法与轻量级系统设计**：
   - **Expert Reordering**：采用**swap-based simulated annealing**算法联合优化计算与通信负载，并引入**目标平滑（objective smoothing）** 和增量更新提升收敛速度。
   - **Expert Replication**：提出**layer-shared replica buffer**，显著减少内存开销；使用**MILP建模 + 增量贪心启发式**求解最优复制与token分配方案。

4. **系统正交性与低开销**：
   - 所有优化在post-rollout阶段完成，**不在训练关键路径上**。
   - 同步操作（参数推送/梯度回传）与attention计算重叠，几乎无额外延迟。

### **相比现有方法的优势**
| 方法 | 局限性 | ReLibra优势 |
|------|--------|-------------|
| **Megatron-LM** | 无动态负载均衡机制 | 显著提升吞吐量（最高达1.6×） |
| **EPLB / LPLB** | 基于历史负载预测，静态复制计划 | 利用routing replay实现精确感知，响应更及时 |
| **PopFetcher** | 仅基于激活模式预取，非主动负载控制 | 主动调整拓扑与资源分配 |

---

## **2. 核心实验方法和设置**

### **使用的模型与数据集**
#### **模型**
- `Qwen3-30B-A3B`（128 experts）
- `GLM4.5-106B-A12B`（128 experts）
- `Qwen3-235B-A22B`（128 experts）

#### **数据集与任务领域（见Table 1）**
| 领域 | 数据集 |
|------|-------|
| **Reasoning** | DAPO-Math-17k, GPQA |
| **Instruction Following** | IFBench |
| **Coding** | CodeForces |
| **Mixed** | 上述三个领域的混合（参考DeepSeek-V3.2） |

### **实验设置**
- **硬件平台**：20节点集群，每节点8块NVIDIA Hopper GPU，NVLink互联（900 GB/s），InfiniBand RDMA跨节点连接（8×400 Gbps/node）。
- **并行策略**（见Table 3）：
  - 使用EP（Expert Parallelism）、DP（Data Parallelism）、PP（Pipeline Parallelism）组合。
  - EP size = 16 或 32，取决于模型大小。
- **Batch配置**：
  - Global batch size = 1024
  - Micro-batch数量 = 32
- **软件栈**：PyTorch 2.8.0, CUDA 12.6, Megatron-LM, DeepEP, FlashAttention-3等。

### **评估指标**
- **主要指标**：
  - **Training Throughput**（tokens/sec）：衡量训练阶段吞吐量。
- **辅助指标**：
  - **Rank-level Skewness**：最大token数 / 平均token数，反映负载不均程度。
  - **Solver Time**：负载规划求解时间是否可隐藏于post-rollout期间。
  - **Memory & Sync Overhead**：复制缓冲区内存占用及同步延迟。

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Megatron-LM** | 当前主流MoE训练框架，支持DeepEP、GroupedGEMM等优化 |
| **PopFetcher** | 基于跨层激活模式预测并预取expert |
| **EPLB+** | 使用oracle负载信息增强的EPLB（贪婪重排序+复制） |
| **LPLB+** | 使用oracle负载的LPLB变体，支持微批级token分裂 |
| **Balanced** | 理想化均衡baseline（随机路由以强制负载均衡，仅作上限参考） |

> 注：所有复制类方法统一使用相同的layer-shared replica buffer和overlap机制，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **总体吞吐量提升（图10、图11）**
- **相对于Megatron-LM**：吞吐量提升 **1.21× ~ 1.58×**
- **相对于EPLB+ / LPLB+（已有oracle负载）**：仍能提升 **1.06× ~ 1.21×**
- **相对于PopFetcher**：提升 **1.17× ~ 1.27×**
- **达到理想均衡（Balanced）的90%~94%吞吐量**

> ✅ 即使EPLB拥有“上帝视角”负载信息，ReLibra依然胜出，说明其**分层调度机制优于单一粒度策略**。

#### **消融实验结果（Table 4）**
以`Qwen3-235B-A22B`为例，在mixed workload下的逐步增益：

| 方法 | 相对吞吐量 | 提升幅度 |
|------|------------|----------|
| Megatron-LM（Baseline） | 1.00 | — |
| + Inter-batch Reordering | 1.15 | +15% |
| + Intra-batch Replication | 1.39 | 再+24%（累计+39%） |

> 🔍 表明两个机制协同作用明显：**先全局重排序创造条件，再局部复制精细调节**。

#### **组件有效性验证**
- **Inter-batch Reordering**（图12）：
  - 比Greedy LPT（ReLibra-Expert）高 **5%~16%**
  - 比数据本地性优化（ReLibra-Data）也更高，证明联合优化更优。
- **Intra-batch Replication**（图13）：
  - 比仅优化placement（ReLibra-Replication）高 **12%~24%**
  - 比仅优化splitting（ReLibra-Splitting）高 **9%~12%**
  - 说明**联合建模replica placement与token splitting至关重要**。

#### **负载均衡能力（图14）**
- **Rank-level skewness接近1.0**：
  - 不同EP规模下（8~64），平均skewness仅为 **1.00~1.08**
  - 而Megatron-LM随EP增大迅速恶化（>2.0）
- 在多个domain上均保持稳定（1.02~1.07），验证泛化性。

#### **系统开销极低（图15、表5、图16）**
- **求解时间完全隐藏于post-rollout阶段**（图15），不影响训练流水线。
- **Expert Reordering Overhead**：仅占训练批次时间的 **1.4%~2.1%**
- **内存开销小**（表5）：
  - Layer-shared replica buffer仅增加 **0.13%~1.62%** 参数内存
- **同步开销被掩盖**（图16）：
  - 参数/梯度传输与attention计算重叠，实际执行时间无显著增长

---

## **4. 关键结论和发现**

### **主要发现**
1. **Routing Replay是MoE RL训练的独特机遇**：
   - rollout阶段即可获得精确的expert路由信息，可用于训练前的主动负载规划。
2. **分层调度（Hierarchical Design）优于单一层级策略**：
   - 将expert reordering与replication分别应用于inter-batch和intra-batch timescale，既避免频繁跨节点通信代价，又能灵活应对瞬时负载尖峰。
3. **ReLibra实现了近似理想的负载均衡效果**：
   - 达到Balanced baseline的 **90%~94%吞吐量**，远超现有方法。
4. **系统开销极低，易于集成**：
   - 所有优化均可离线完成，运行时无额外负担，适合工业部署。

### **局限性**
- **依赖Routing Replay机制**：
  - 若RL算法未启用routing replay（如某些off-policy方法早期版本），则无法获取准确路由信息。
  - 但作者指出，**工业界已普遍采纳routing replay以保证训练稳定性**（引用[20,28,29]），因此该假设合理。
- **当前实现聚焦于EP维度**：
  - 未深入探索与TP/PP更深层次的协同优化空间。
- **MILP求解仍较重**：
  - 虽然采用增量贪心启发式，但在更大规模模型上可能需进一步加速。

### **未来工作方向**
- 支持更多类型的**异构MoE架构**（如variable experts per layer）。
- 探索**跨stage协同优化**：结合rollout阶段的speculative decoding或early exit机制。
- 扩展至**多模态MoE RL训练**场景。
- 进一步压缩solver时间，支持**real-time在线adaptation**。

---

> 📌 **总结一句话**：  
> **ReLibra首次系统性地利用RL中的routing replay特性，通过分层负载均衡设计，在无需预测的前提下实现了MoE训练中近乎完美的负载均衡，显著提升了训练吞吐量，且具备低开销、易集成、强通用性的优点。**

</details>

---

### 15. [TileQ: Efficient Low-Rank Quantization of Mixture-of-Experts with 2D Tiling](https://arxiv.org/abs/2605.09281)

**Authors**: Hongyaoxing Gu, Xinzhe Chen, Lijuan Hu, Fangfang Liu  
**Category**: cs.LG  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.09281v1  

#### Abstract
Mixture-of-Experts (MoE) models achieve remarkable performance by sparsely activating specialized experts, yet their massive parameters in experts pose significant challenges for deployment. While low-rank quantization offers a promising route to compress MoE models, existing methods still incur non...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TileQ: Efficient Low-Rank Quantization of Mixture-of-Experts with 2D Tiling**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Mixture-of-Experts (MoE) 模型通过稀疏激活专家机制实现了高性能，但其庞大的参数量（尤其是专家部分）带来了显著的**内存开销**和**推理延迟**，限制了在资源受限设备上的部署。尽管已有低秩量化（low-rank quantization）等压缩技术，但现有方法仍存在以下问题：
- **额外内存开销大**：传统低秩方法为每个专家独立存储低秩因子，导致冗余；
- **推理效率低**：稀疏调度和多次小规模 GEMM 导致硬件利用率低下；
- **精度损失严重**：极端低比特（如 2-bit）下性能下降明显。

### **提出的新方法与创新思路**
论文提出了 **TILEQ**，一种无需微调（fine-tuning-free）的后训练量化（PTQ）方法，核心创新在于：
- **2D-Tiling 结构化低秩量化**：
  - 将所有专家按相似性组织成一个二维网格（M×N），共享输入维度（U）和输出维度（V）的低秩因子；
  - 利用奇异子空间聚类（singular subspace clustering）实现跨专家的联合表示共享，显著减少低秩因子存储。
- **高效的融合推理算法（LoTileMoE）**：
  - 将多个低秩专家计算融合为单次密集 GEMM 操作，避免稀疏调度带来的不规则内存访问；
  - 在 Prefill 和 Decode 阶段均实现高硬件利用率。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **压缩率** | 较 1D 共享方法降低 √K 倍内存，较逐专家方法降低 2√K 倍；实测低秩额外位宽仅 ~0.04 bit |
| **推理速度** | 推理延迟降至约 **5%**，远低于传统方法（>50%） |
| **精度保持** | 在 2-bit 下达到 SOTA 精度，接近 FP16 性能 |
| **通用性与易用性** | 无需微调，兼容主流 PTQ 方法（如 GPTQ/GPTVQ），可即插即用 |

---

## **2. 核心实验方法和设置**

### **使用的模型与数据集**
- **模型**：
  - `Qwen1.5-MoE-A2.7B`, `Qwen3-30B-A3B`, `Qwen3-Next-80B-A3B`
  - `Mixtral-8×7B`, `Deepseek-MoE-16B`
- **校准数据集（Calibration）**：
  - C4 数据集中的 128 条序列，每条 2048 tokens
- **评估数据集**：
  - **Perplexity**: WikiText-2
  - **下游任务**：ARC-Challenge/Easy, PIQA, Winogrande (WI), HellaSwag (HS), MMLU

### **实验设置与评估指标**
- **量化配置**：
  - 2-bit 与 3-bit 权重量化
  - Tile-Rank 设为 32，Tiling-size 近似为 √K × √K
- **评估指标**：
  - **Perplexity (PPL)**：衡量语言建模能力
  - **Accuracy (%)**：各下游任务准确率
  - **推理延迟**：Prefill 与 Decode 阶段的 MoE MLP 块耗时
  - **额外位宽（Extra Bits）**：低秩因子与量化尺度所占平均比特数

### **基线方法对比**
- **基础 PTQ 方法**：
  - `GPTQ`（逐通道标量量化）
  - `GPTVQ`（向量量化）
- **低秩 PTQ 方法**：
  - `LoPRo`（低秩旋转）
  - `MoEQUANT`（MoE 特定量化）
- **MoE 专用低秩方法**：
  - `MiLo`（1D 共享低秩）
  - `MXMoE`（混合精度设计）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 方法 | 模型 | Bit | PPL (WikiText) | MMLU (%) | Extra Bits |
|------|------|-----|----------------|----------|------------|
| FP16 | Mixtral-8x7B | — | 3.87 | 83.7 | — |
| GPTQ | Mixtral-8x7B | 2 | 15.3 | 53.0 | 0.13 |
| LoPRo | Mixtral-8x7B | 2 | 5.01 | 63.8 | 0.21 / 0.08 |
| **TILEQs** | Mixtral-8x7B | 2 | **4.98** | **63.8** | **0.16 / 0.03** |
| **TILEQu** | Mixtral-8x7B | 3 | **4.10** | **69.4** | **0.16 / 0.03** |

> ✅ **TILEQ 在 2-bit 下将 Mixtral 的 PPL 从 15.3 (GPTQ) 降至 4.98，接近 FP16 的 3.87**

### **与基线方法的对比结果**
- **精度优势**：
  - 在 2-bit 下，TILEQ 显著优于 GPTQ、GPTVQ 和 MoEQUANT，在多个任务上提升超过 10 个百分点；
  - 与 LoPRo 相比，TILEQ 在更低额外位宽下实现相当甚至更优精度（如 Qwen3-30B-A3B 上 MMLU 提升至 71.3）。
- **内存效率**：
  - 低秩额外位宽仅为 **0.04 bit**，相较 per-expert 方法（0.32 bit）节省 **8× 内存**；
  - 在 Qwen3-30B-A3B 上，较 1D 共享方法（MiLo）减少 **6× 存储开销**。
- **推理速度**：
  - 在 A800 上，TILEQ 将 MoE MLP 块的推理延迟降低至 **5% 以内**；
  - 相比 MiLo-1D，在 Qwen3-MoE 上延迟降低 **10× 以上**。

### **消融实验结果**
#### **Ablation on Components (Qwen1.5-MoE-A2.7B, 2-bit)**
| 变体 | PPL | ACC | Extra Bits |
|------|-----|-----|-----------|
| Full TILEQ | 7.35 | 62.5 | 0.04 |
| × Vector Q | 7.56 | 61.6 | 0.04 |
| × Rotation | 7.60 | 61.8 | 0.04 |
| × 2D-Tiling | 7.49 | 61.6 | **0.31** |
| × Low-Rank | 12.5 | 38.5 | 0.00 |

> 🔍 发现：
> - 移除 **2D-Tiling** 导致额外位宽上升 8 倍，证明其对压缩至关重要；
> - 移除 **Low-Rank** 导致性能崩溃，说明低秩建模是精度保障的核心；
> - 向量量化与旋转在 2-bit 下增益有限，但在 3-bit 更有效。

#### **Ablation on Rank & Tiling Size**
- **Rank=32** 是最佳平衡点：更高 rank 带来边际收益递减，且显著增加延迟；
- **近方形布局（如 8×8）最优**：非均衡布局（如 60×1）导致 decode 延迟飙升（>15%），而 8×8 实现最低延迟（5.1%）与最高精度。

---

## **4. 关键结论和发现**

### **主要发现**
1. **2D-Tiling 能高效挖掘专家间冗余**：通过双聚类（biclustering）实现 U/V 因子的联合共享，大幅降低低秩存储成本；
2. **结构化推理显著提升硬件利用率**：LoTileMoE 将稀疏调度转化为批量密集运算，充分发挥 Tensor Core 效能；
3. **无需微调即可实现 SOTA 量化效果**：TILEQ 在 2-bit 下逼近 FP16 表现，适用于快速部署场景；
4. **方法具有强扩展性**：在“大而密”与“小而疏”的 MoE 模型上均表现优异，具备通用压缩框架潜力。

### **局限性**
- **量化阶段仍是瓶颈**：当前依赖 GPTQ/GPTVQ 等后端，对大量小型专家处理效率不高；
- **缺乏对其他压缩技术的集成**：未结合剪枝、蒸馏等进一步压缩手段；
- **定制算子支持不足**：某些竞品（如 MXMoE）依赖特定硬件优化，难以公平比较。

### **未来工作方向**
- **量化-低秩联合优化**：设计专用于低秩结构的 PTQ 算法，降低校准开销；
- **与结构化剪枝/知识蒸馏结合**：构建混合压缩 pipeline，追求极致压缩比；
- **支持动态路由感知量化**：利用路由分布优化低秩分解策略；
- **部署端到端系统验证**：在真实服务场景中测试 TILEQ 的吞吐与延迟表现。

---

> 📌 **一句话总结**：  
> **TILEQ 通过 2D-Tiling 结构化低秩量化与融合推理，实现了 MoE 模型的高效压缩与极速推理，在无需微调的前提下达成 SOTA 精度与极低延迟，为大规模 MoE 部署提供了实用解决方案。**

</details>

---

### 16. [Dystruct: Dynamically Structured Diffusion Language Model Decoding via Bayesian Inference](https://arxiv.org/abs/2605.09820)

**Authors**: Bian Sun, Kevin Zhai, Mubarak Shah, Zhenyi Wang  
**Category**: cs.LG  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.09820v1  

#### Abstract
Diffusion language models (DLMs) have recently emerged as a promising alternative to autoregressive models, primarily due to their ability to enable parallel decoding. Despite this advantage, most existing DLMs rely on a fixed generation length specified prior to decoding, which restricts their flex...

---

### 17. [Efficient Neural Architectures for Real-Time ECG Interpretation on Limited Hardware](https://arxiv.org/abs/2605.09848)

**Authors**: Ashery Mbilinyi, Callum O'Riley, Julia Handra, Ashley Moller-Hansen, Jason Andrade, Marc Deyell, Cameron Hague, Nathaniel Hawkins, Kendall Ho, Jonathan Leipsic, Roger Tam  
**Category**: cs.LG  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.09848v1  

#### Abstract
Electrocardiogram (ECG) interpretation is essential for diagnosing a wide range of cardiac abnormalities. While deep learning has shown strong potential for automating ECG classification, many existing models rely on large, computationally intensive architectures that hinder practical deployment. In...

---

### 18. [TrajDLM: Topology-Aware Block Diffusion Language Model for Trajectory Generation](https://arxiv.org/abs/2605.10020)

**Authors**: Wilson Wongso, Lihuan Li, Arian Prabowo, Xiachong Lin, Baiyu Chen, Hao Xue, Flora D. Salim  
**Category**: cs.LG  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.10020v1  

#### Abstract
Generating high-fidelity synthetic GPS trajectories is increasingly important for applications in transportation, urban planning, and what-if scenario simulation, especially as privacy concerns limit access to real-world mobility data. Existing trajectory generation models face a trade-off between e...

---

### 19. [Locking Pretrained Weights via Deep Low-Rank Residual Distillation](https://arxiv.org/abs/2605.10777)

**Authors**: Keitaro Sakamoto, Pierre Ablin, Federico Danieli, Marco Cuturi  
**Category**: cs.LG  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.10777v1  

#### Abstract
The quality of open-weight language models has dramatically improved in recent years. Sharing weights greatly facilitates model adoption by enabling their use across diverse hardware and software platforms. They also allow for more open research and testing, to the extent that users can use them as ...

---

### 20. [ConQuR: Corner Aligned Activation Quantization via Optimized Rotations for LLMs](https://arxiv.org/abs/2605.10793)

**Authors**: Chayne Thrash, Ali Abbasi, Soheil Kolouri  
**Category**: cs.LG  
**Published**: 2026-05-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.10793v1  

#### Abstract
Large language models (LLMs) are costly to deploy due to their large memory footprint and high inference cost. Weight-activation quantization can reduce these costs, but low-bit activation quantization remains difficult because activation outliers induce large quantization error. Recent rotation-bas...

---

### 21. [WindINR: Latent-State INR for Fast Local Wind Query and Correction in Complex Terrain](https://arxiv.org/abs/2605.09511)

**Authors**: Yi Xiao, Qilong Jia, Hang Fan, Pascal Fua, Robert Jenssen, Xiaosong Ma, Wei Xue  
**Category**: cs.AI  
**Published**: 2026-05-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.09511v1  

#### Abstract
Many downstream decisions in complex terrain require fast wind estimates at a small number of user-specified locations and heights for a given forecast valid time, rather than another dense forecast field on a fixed grid. We present WindINR, a latent-state implicit neural representation framework fo...

---

### 22. [Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding](https://arxiv.org/abs/2605.10855)

**Authors**: Jianzhu Bao, Haozhen Zhang, Kuicai Dong, Bozhi Wu, Sarthak Ketanbhai Modi, Zi Pong Lim, Yon Shin Teo, Wenya Wang  
**Category**: cs.CL  
**Published**: 2026-05-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.10855v1  

#### Abstract
Vision-Language Models (VLMs) have demonstrated remarkable progress in chart understanding, largely driven by supervised fine-tuning (SFT) on increasingly large synthetic datasets. However, scaling SFT data alone is inefficient and overlooks a key property of charts: charts are programmatically gene...

---

### 23. [OrbitBFT: Enabling Scalable and Robust BFT Consensus in LEO Constellations](https://arxiv.org/abs/2605.08132)

**Authors**: Tianyi Sun, Shuo Liu, Minghui Xu, Xiuzhen Cheng  
**Category**: cs.DC  
**Published**: 2026-05-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.08132v1  

#### Abstract
Low Earth Orbit (LEO) satellite constellations are evolving from communication relays into autonomous platforms operating in increasingly congested and contested environments. Since uplinks to ground stations can be severed or jammed, ensuring reliable coordination among satellites requires autonomo...

---

### 24. [Unleashing Scalable Context Parallelism for Foundation Models Pre-Training via FCP](https://arxiv.org/abs/2605.08524)

**Authors**: Yilong Zhao, Xiaonan Nie, Kan Zhu, Shuang Ma, Zhichao Lai, Hongxiang Hao, Yang Zhou, Baris Kasikci, Ion Stoica  
**Category**: cs.DC  
**Published**: 2026-05-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.08524v1  

#### Abstract
Context parallelism (CP) has been widely adopted to support the growing context length in foundation model pretraining. However, existing designs fail to handle the large variation in sequence length from training datasets, resulting in suboptimal performance. These methods often over-shard short se...

---

### 25. [Surviving Partial Rank Failures in Wide Expert-Parallel MoE Inference](https://arxiv.org/abs/2605.10670)

**Authors**: Xun Sun, Shaoyuan Chen, Pingchuan Ma, Yue Chen, Ziwei Yuan, Zhanhao Cao, Han Han, Shangming Cai, Teng Ma, Xuchun Shang, Xinpeng Zhao, Ke Yang, Junlin Wei, Lianzhi Lin, Yuji Liu, Feng Ren, Haoran Hu, Cheng Wan, Yingdi Shan, Yongwei Wu, Mingxing Zhang  
**Category**: cs.DC  
**Published**: 2026-05-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.10670v1  

#### Abstract
Mixture-of-Experts (MoE) serving relies on wide expert parallelism (EP) to aggregate the memory capacity and bandwidth of many GPUs within one inference instance. This efficiency comes with a systems cost: every decoding step depends on token dispatch and combination across all active EP ranks, so e...

---

### 26. [Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria](https://arxiv.org/abs/2605.08354)

**Authors**: Juanxi Tian, Fengyuan Liu, Jiaming Han, Yilei Jiang, Yongliang Wu, Yesheng Liu, Haodong Li, Furong Xu, Wanhua Li  
**Category**: cs.AI  
**Published**: 2026-05-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.08354v1  

#### Abstract
Aligning multimodal generative models with human preferences demands reward signals that respect the compositional, multi-dimensional structure of human judgment. Prevailing RLHF approaches reduce this structure to scalar or pairwise labels, collapsing nuanced preferences into opaque parametric prox...

---

### 27. [C2L-Net: A Data-Driven Model for State-of-Charge Estimation of Lithium-Ion Batteries During Discharge](https://arxiv.org/abs/2605.08653)

**Authors**: Khoa Tran, T. Nguyen-Thoi, Vin Nguyen-Thai, Duong Tran Anh, Hung-Cuong Trinh, Tri Le  
**Category**: cs.AI  
**Published**: 2026-05-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.08653v1  

#### Abstract
Accurate state-of-charge (SOC) estimation is critical for the safe and efficient operation of lithium-ion batteries in battery management systems (BMS). Although data-driven approaches can effectively capture nonlinear battery dynamics, many existing methods rely on long historical input sequences, ...

---

### 28. [AHD Agent: Agentic Reinforcement Learning for Automatic Heuristic Design](https://arxiv.org/abs/2605.08756)

**Authors**: Haoze Lv, Ning Lu, Ziang Zhou, Shengcai Liu  
**Category**: cs.AI  
**Published**: 2026-05-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.08756v1  

#### Abstract
Automatic heuristic design (AHD) has emerged as a promising paradigm for solving NP-hard combinatorial optimization problems (COPs). Recent works show that large language models (LLMs), when integrated into well-designed frameworks (i.e., LLM-AHD), can autonomously discover high-performing heuristic...

---

### 29. [Forge: Quality-Aware Reinforcement Learning for NP-Hard Optimization in LLMs](https://arxiv.org/abs/2605.08905)

**Authors**: Xiaozhe Li, Xinyu Fang, Shengyuan Ding, Yang Li, Linyang Li, Haodong Duan, Qingwen Liu, Kai Chen  
**Category**: cs.AI  
**Published**: 2026-05-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.08905v1  

#### Abstract
Large Language Models (LLMs) have achieved remarkable success on reasoning benchmarks through Reinforcement Learning with Verifiable Rewards (RLVR), excelling at tasks such as math, coding, logic, and puzzles. However, existing benchmarks evaluate only correctness, while overlooking optimality, name...

---

### 30. [ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing](https://arxiv.org/abs/2605.08840)

**Authors**: Yongqi An, Chang Lu, Kuan Zhu, Tao Yu, Chaoyang Zhao, Hong Wu, Ming Tang, Jinqiao Wang  
**Category**: cs.CL  
**Published**: 2026-05-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.08840v1  

#### Abstract
Large language models (LLMs) face growing challenges in efficient generative inference due to the increasing memory demands of Key-Value (KV) caches, especially for long sequences. Existing eviction methods typically retain KV pairs with high attention weights but overlook the impact of attention re...

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

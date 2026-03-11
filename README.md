# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-11 06:16:23 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control](https://arxiv.org/abs/2603.09221)

**Authors**: Peihao Wang, Shan Yang, Xijun Wang, Tesi Xiao, Xin Liu, Changlong Yu, Yu Lou, Pan Li, Zhangyang Wang, Ming Lin, Ren\'e Vidal  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.09221v1  

#### Abstract
Associative memory has long underpinned the design of sequential models. Beyond recall, humans reason by projecting future states and selecting goal-directed actions, a capability that modern language models increasingly require but do not natively encode. While prior work uses reinforcement learnin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control

## 1. 论文的主要贡献和创新点

### 解决的问题
现代语言模型（LLMs）在推理任务中表现受限，其架构本质上依赖于**associative memory**（联想记忆），即通过检索历史上下文来预测下一个 token。这种“System 1”式的快速直觉模式缺乏“System 2”式的**多步规划、长期推理和目标导向决策能力**。尽管强化学习（RL）等方法尝试引入目标导向行为，但这些优化通常作为外部训练过程，**并未内化到模型的前向推理机制中**。

因此，当前模型在数学推理、逻辑谜题等需要显式规划的任务上存在“推理天花板”。

### 提出的新方法与思路
本文提出了一种全新的架构范式——将**推理（reasoning）建模为最优控制问题（optimal control）**，并将其直接嵌入到模型架构中。

核心创新是提出了 **Test-Time Control (TTC) 层**，其关键思想如下：

- **TTC 层**：在推理时（inference time），对隐状态执行有限视界（finite-horizon）的**线性二次调节器（LQR）规划**。
- **内部价值函数**：TTC 层在神经网络内部表示一个可学习的**value function**，使模型能够在预测前进行“规划”。
- **规划先于预测**：模型不再仅基于记忆生成 token，而是通过求解一个受控的动力学系统，选择能最大化未来回报的“动作”（action），并将该动作解码为下一个 token 的表示。
- **端到端学习**：整个 TTC 层是可微分的，可以通过反向传播进行端到端训练。

### 相比现有方法的优势
| 维度 | 现有方法（如 TTT, RL） | 本文方法（TTC） |
|------|------------------------|------------------|
| **推理机制** | 测试时训练（Test-Time Training），本质是在线回归/参数适应 | 测试时控制（Test-Time Control），本质是在线决策与规划 |
| **目标导向** | 外部奖励驱动，与推理过程分离 | 内部价值函数驱动，规划是前向计算的一部分 |
| **架构集成** | 通常是后处理或外部模块 | 作为轻量级适配器（adapter）无缝集成到预训练 LLM 中 |
| **硬件效率** | 传统 LQR 求解器（如 Riccati 迭代）计算复杂且难以并行 | 提出**辛（symplectic）LQR 求解器**，支持高度并行化，实现低开销 |

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **Sudoku**：经典的 9×9 数独逻辑谜题，用于评估长程推理、约束传播和多步规划能力。
2. **数学推理基准**：
   - **MATH-500**：标准数学问题数据集。
   - **AMC**（American Mathematics Competitions）
   - **AIME 2024 & AIME 2025**：更具挑战性的数学竞赛题。

### 实验设置和评估指标
- **模型架构**：构建 **TTC-Net**，一种混合架构，在预训练的 Transformer 骨干网络（如 Llama-3-Instruct-7B）中，每 8 个 Transformer 块插入一个 TTC 层。
- **训练方式**：采用**持续学习（continual learning）**，在预训练模型基础上进行监督微调（SFT）。
- **评估指标**：
  - **Sudoku**：单步完成准确率（Single-Step Board/Cell Acc）、多步迭代完成准确率（Multi-Step Board/Cell Acc）。
  - **数学推理**：`Acc@8`（8 次采样中的平均准确率）、`Pass@8`（8 次采样中至少一次正确的概率）。
- **测试时扩展（Test-Time Scaling）**：在推理时动态调整规划视界 `T_test`，观察性能随计算量增加的变化。

### 基线方法对比
- **纯记忆型模型**：Transformer、Mamba、Mamba2、GDN、Samba。
- **带适配器的记忆增强模型**：在相同位置插入额外的注意力层、RetNet、Mamba、GDN、MesaNet 等作为对比。
- **基线**：仅对基础模型权重进行微调（Full Finetuning + MLP）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **MATH-500** 数据集上，TTC-Net 达到 **52.8%** 的准确率，相比基线（47.8%）有显著提升。

在更具挑战性的 **AIME** 数据集上，TTC-Net 展现出突破性表现：
- **AMC Pass@8**: **54.22%**（相比基线最高 46.98%）
- **AIME24 Pass@8**: **20.00%**（基线为 0.00% 或极低）
- **AIME25 Pass@8**: **20.00%**（基线为 0.00% 或极低）

> **结论**：TTC-Net 在 AIME 上实现了从 **0 到 20%** 的性能涌现，表明其解锁了基础模型原本不具备的复杂推理能力。

### 与基线方法的对比结果
- 在所有基准上，**TTC-Net 均优于所有基线方法**，尤其是在需要深度推理的 AIME 和多步 Sudoku 任务上优势明显。
- 传统的记忆增强适配器（如 +Attention, +Mamba）带来的增益有限，甚至不如简单的全量微调，说明单纯增加记忆容量无法解决推理瓶颈。
- TTC-Net 的成功验证了**内化规划机制**比单纯的**测试时记忆**更有效。

### 消融实验结果
在 MATH-500 上进行了消融研究，关键发现如下：

| 变体 | MATH-500 准确率 (T_test=8) | 说明 |
|------|----------------------------|------|
| **完整模型 (TTC-Net)** | **52.80%** | 时间异质参数化 + PLN 采样 + 8:1 插入比 |
| 时间同质参数化 | 48.40% | 固定动力学参数，表达能力受限 |
| 固定训练视界 | 50.60% → 31.50% (T_test=16) | 泛化能力差，无法适应更大的测试视界 |
| 均匀视界采样 | 51.00% | 性能接近，但训练成本更高（平均视界更大） |
| 更密集的 TTC 层 (4:1) | 53.00% | 性能略高，但计算成本更高，性价比不如测试时扩展 |

> **结论**：时间异质参数化和混合视界训练策略对泛化至关重要；通过增加测试时视界 `T_test` 是一种高效且可扩展的性能提升路径。

---

## 4. 关键结论和发现

### 主要发现
1. **推理可以被形式化为最优控制**：将语言模型的推理过程建模为 LQR 规划问题是可行且有效的。
2. **TTC 层是一种强大的架构组件**：它将世界建模（world modeling）、价值函数、强化学习目标和规划统一在一个可微分、可训练的模块中。
3. **硬件效率是关键**：提出的**辛 LQR 求解器**通过融合 CUDA 内核和并行矩阵运算，实现了超过传统方法 **10 倍的吞吐量**，使其能够实际部署在大规模 LLM 中。
4. **测试时扩展（Test-Time Scaling）**：TTC-Net 支持通过增加规划视界 `T_test` 来动态分配更多计算资源以换取更高性能，这提供了一个新的、原生的推理扩展轴。

### 方法的局限性
- **理论理解不足**：多个 TTC 层在深层网络中如何协同工作、共同表示动态系统，目前尚无严格的理论分析。
- **表达能力限制**：当前基于线性动力学和二次代价的 LQR 模型虽然高效，但可能不足以捕捉最复杂的非线性推理过程。
- **规模验证有限**：实验主要集中在中等规模模型（如 7B），在更大模型和全阶段训练中的潜力仍需探索。

### 未来工作方向
- 探索更**表达力更强的潜在空间动力学和奖励建模**，例如非线性公式，同时保持硬件友好性。
- 对多层 TTC 架构进行**深入的理论分析**，理解其表示能力和收敛性质。
- 在**更大规模的模型**和**从头训练（from scratch）** 的场景下全面评估 TTC-Net 的潜力。
- 探索与**模型无关的强化学习（model-free RL）** 方法（如 TTD）的结合，实现更强大的测试时自适应能力。

</details>

---

### 2. [RSH-SpMM: A Row-Structured Hybrid Kernel for Sparse Matrix-Matrix Multiplication on GPUs](https://arxiv.org/abs/2603.08734)

**Authors**: Aiying Li, Jingwei Sun, Han Li, Wence Ji, Guangzhong Sun  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.08734v1  

#### Abstract
Sparse Matrix-Matrix Multiplication (SpMM) is a fundamental computation in graph analytics, scientific simulation, and sparse deep learning workloads. However, the extreme irregularity of real-world sparse matrices prevents existing GPU-based methods from maintaining high Tensor Core utilization and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RSH-SpMM: A Row-Structured Hybrid Kernel for Sparse Matrix-Matrix Multiplication on GPUs

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

Sparse Matrix-Matrix Multiplication (SpMM) 是图神经网络（GNN）、科学计算和稀疏深度学习中的核心算子。然而，现实世界中的稀疏矩阵具有**极端不规则性**，表现为：
- 行非零元数量（nnz）分布严重偏斜（heavy-tailed）
- 局部密度快速变化
- 非零模式碎片化

这些特性导致：
- **Tensor Core 利用率低**：需要密集对齐的 tile 输入，而稀疏结构难以满足
- **内存访问不连续**：降低带宽利用率
- **负载不均衡**：部分行过长或过短，影响并行效率

现有方法（如纯 CUDA-core、纯 Tensor-Core 或粗粒度混合方法）无法有效应对这种细粒度的结构性异质性。

---

### **提出了什么新方法或新思路**

作者提出 **RSH-SpMM** —— 一种基于**行结构化混合执行**的 SpMM 框架，其核心是 **RS-Tile 表示法** 和自适应调度机制。

#### 主要创新点包括：

1. ✅ **RS-Tile 压缩格式**
   - 将稀疏矩阵 $ A $ 分解为两个互斥部分：
     - **TC Part**：结构一致的“稠密窗口”行组，适配 Tensor Core 执行
     - **CUDA Residual Part**：孤立短行或结构异常行，交由轻量级 CUDA 核心处理
   - 使用 bitmap 编码非零位置，压缩列索引，提升存储效率

2. ✅ **细粒度混合执行策略**
   - 引入**自适应行划分**（adaptive row partitioning），在构建 row window 前判断每行是否适合进入 TC 路径
   - 设计**双缓冲流水线 Tensor-Core 内核**，实现内存加载与 MMA 计算重叠
   - 构建**极低开销的 CUDA 内核路径**，避免短行为 TC 流水线引入空转周期

3. ✅ **局部性感知重排序（Locality-aware Reordering）**
   - 使用加权 Jaccard 相似度衡量行间结构相似性
   - 构建 kNN 图 → 最小生成树（MST）→ DFS 遍历顺序进行全局重排
   - 局部优化阶段使用 2-opt 交换进一步减少相邻行差异
   - 显著提升 TC tile 密度和内存局部性

4. ✅ **自适应负载均衡机制**
   - 对“超长行”按需拆分，防止单个 block 成为瓶颈
   - 短行提前隔离至 CUDA 路径，从源头减少负载方差
   - 不依赖原子操作或复杂的索引重映射，保持简洁高效

---

### **相比现有方法的优势**

| 维度 | 现有方法缺陷 | RSH-SpMM 改进 |
|------|---------------|----------------|
| **表示灵活性** | 固定窗口大小、刚性 tile 几何形状 | 动态行划分 + 结构感知分块 |
| **执行路径选择** | 粗粒度决策（全矩阵或大块） | 细粒度逐行判断，精准路由 |
| **负载均衡** | 忽视 intra-window 差异，易造成阻塞 | 自适应拆分 + 提前隔离，稳定吞吐 |
| **元数据开销** | 平衡版本需额外 remapping 表 | RS-Tile 天然兼容平衡，无冗余结构 |
| **硬件适配性** | CUDA 路径处理低强度任务效率低 | 将短行交给更适合的 CUDA core |

---

## 2. 核心实验方法和设置

### **使用的数据集**

1. **9 个代表性真实世界图数据集**：
   - `com-amazon`, `ddi`, `DD`, `amazon0505`, `amazon0601`, `Yeast`, `OVCAR-8H`, `YeastH`, `web-BerkStan`
   - 来源于 TC-GNN、SNAP、DGL 等公开集合
   - 覆盖高度偏斜度分布和异构局部密度场景

2. **SuiteSparse Matrix Collection 中的 512 个矩阵**
   - 包含来自不同应用领域的稀疏矩阵（工程、物理模拟、网络等）
   - 筛选条件：≥5K 行/列，≥100K 非零元
   - 用于评估方法在广泛稀疏模式下的鲁棒性和通用性

---

### **实验设置和评估指标**

#### **硬件平台**
- **RTX 4090**（Ada Lovelace, compute capability 8.9, 24GB）
- **RTX 3090**（Ampere, compute capability 8.6, 24GB）
- 报告已预热后的 kernel 执行时间

#### **问题配置**
- 计算 $ C = A \times B $，其中 $ A $ 为稀疏矩阵，$ B $ 为稠密特征矩阵
- 特征维度 $ d \in \{64, 128, 256, 512\} $
- 使用 FP16 输入进行 MMA 运算，FP32 累积中间结果，输出为 FP32

#### **评估指标**
- **吞吐量加速比**（normalized speedup over cuSPARSE）
- **SM throughput (%)**
- **Tensor Core utilization (%)**
- **元数据存储开销**（相对于 COO 格式的归一化空间使用）

---

### **基线方法对比**

| 类型 | 基线方法 |
|------|---------|
| **CUDA-core** | cuSPARSE, Sputnik, RoDe |
| **Tensor-Core-only** | TC-GNN, DTC-SpMM, Acc-SpMM |
| **Hybrid / SpTC-based** | HC-SpMM, MP-SpMM |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 在 9 个真实图上的平均加速比（vs cuSPARSE）：
- **RTX 4090**: **2.35×**
- **RTX 3090**: **2.86×**

#### 在 SuiteSparse 512 矩阵上的平均加速比（vs cuSPARSE）：
- **平均 3.21×**
- **超过 80% 的矩阵落在 1.24× ~ 8.2× 区间内**

> 🔥 **最高达到 6.13× 加速（vs TC-GNN）**

---

### **与基线方法的对比结果**

| 对比对象 | 性能优势 |
|--------|----------|
| **vs CUDA-core 方法**（cuSPARSE/Sputnik/RoDe） | 平均 **2.06×–2.61×** 提升，尤其在不规则矩阵上优势显著 |
| **vs TC-GNN** | 平均 **6.13×** 提升（排除崩溃情况），因其固定窗口无法适应密度波动 |
| **vs DTC-SpMM / Acc-SpMM** | 分别高出 **1.91×** 和 **1.61×**，因后者固定窗口导致 MMA 利用率下降 |
| **vs HC-SpMM**（混合方法） | 平均 **2.10×** 提升，因其粗粒度分区未能捕捉细粒度结构一致性 |
| **vs MP-SpMM**（依赖 SpTC） | 平均 **1.32×** 提升，因后者需严格结构化稀疏性，适用性受限 |

---

### **消融实验结果**

#### （1）**RS-Tile 存储效率分析**
- 相比平衡版 ME-TCF 和 BitTCF，**元数据体积减少约 15.05%**
- 原因：无需维护 remapping 表或重复索引结构
- 图 13 显示 RS-Tile 在所有矩阵中保持最紧凑的元数据占用

#### （2）**nnz 阈值对 TC/CUDA 分区的影响**（图 14）
- 当阈值从 0 增加到 4：
  - TC block 平均 nnz 从 16.7 ↑ 至 19.5
  - row window 平均 nnz 从 174 ↑ 至 233
- 超过 6 后收益饱和，且 CUDA 路径负载急剧上升 → 性能下降
- 结论：仅需移除少量“异常短行”即可大幅提升 tile 质量

#### （3）**GPU Profiling 对比**（图 15）
- **SM Throughput**：RSH-SpMM 中位数达 **33%** vs Acc-SpMM 的 28%
- **Tensor Core Utilization**：RSH-SpMM 达 **8.8%** vs Acc-SpMM 的 **5.6%**
- 表明 RSH-SpMM 更好地维持了 MMA 流水线的持续填充

#### （4）**重排序效果**（图 16）
- 相比原始顺序：
  - Rabbit Order: **0.87×**（反而变慢）
  - TCA Reorder (DTC-SpMM): **1.15×**
  - **RSH-SpMM 重排序**: **1.25×**（最高达 **1.7×**）
- 阶段分解显示：
  - MST 排序贡献最大（→1.19×）
  - 2-opt 微调（→1.22×）
  - 孤立行调整（→1.25×）

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **细粒度结构感知划分优于粗粒度启发式策略**
   - 现有混合方法因分区粒度过粗，仍会将低质量行送入 TC 路径，浪费资源
   - RSH-SpMM 的逐行决策机制可精准识别“破坏者”，显著提升 tile 密度

2. ✅ **CUDA core 更适合处理极短行**
   - 短行难以填满 MMA tile，准备开销远高于计算收益
   - 将其交由 CUDA core 处理反而更高效，降低寄存器压力和同步成本

3. ✅ **重排序应结合结构相似性而非简单度排序**
   - 单纯按行长度排序不足以暴露局部稠密结构
   - 加权 Jaccard + MST 的组合能有效揭示潜在的结构连续性

4. ✅ **负载均衡应在表示层原生支持**
   - 后置的 remapping 或原子累加会带来额外开销
   - RSH-SpMM 将负载控制融入 tile 构造过程，实现“无感平衡”

---

### **方法的局限性**

1. ⚠️ **依赖静态稀疏模式**
   - 当前设计适用于静态图或预知 sparsity pattern 的场景
   - 对动态稀疏更新的支持有限，重排序与格式转换成本较高

2. ⚠️ **重排序本身有开销**
   - 虽然只做一次，但在超大规模图上构建 kNN 和 MST 可能耗时
   - 可考虑采样近似或增量更新策略

3. ⚠️ **当前仅支持 SpMM 场景**
   - 尚未扩展到 SDDMM、SpMV 或其他稀疏算子
   - 但框架思想具备迁移潜力

---

### **未来工作方向**

1. 🔄 **支持动态稀疏性更新**
   - 开发增量式重排序与 RS-Tile 更新机制
   - 应用于在线训练或流式图处理

2. 🧩 **推广至其他稀疏算子**
   - 如 SDDMM、SpMMv2（多输出）、SpGEMM
   - 探索统一的 hybrid execution abstraction

3. 💡 **与编译器/框架集成**
   - 将 RSH-SpMM 作为 PyTorch / TensorFlow / DGL 的后端算子
   - 实现自动稀疏模式检测与内核选择

4. 📈 **探索更高维 tile 结构**
   - 支持 MMA 指令如 mma.sync.m32n8k4 等新型 Tensor Core 操作
   - 提升高维特征场景下的吞吐能力

---

> ✅ **总结一句话**：  
> RSH-SpMM 通过**细粒度行结构感知划分 + 局部性重排序 + 原生负载均衡的 RS-Tile 表示**，实现了对不规则稀疏性的高效利用，在多种真实图和通用稀疏矩阵上取得 **1.27×–6.13×** 的性能提升，是当前最稳定高效的 GPU SpMM 混合执行方案之一。

</details>

---

### 3. [ConFu: Contemplate the Future for Better Speculative Sampling](https://arxiv.org/abs/2603.08899)

**Authors**: Zongyue Qin, Raghavv Goel, Mukul Gagrani, Risheek Garrepalli, Mingu Lee, Yizhou Sun  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.08899v1  

#### Abstract
Speculative decoding has emerged as a powerful approach to accelerate large language model (LLM) inference by employing lightweight draft models to propose candidate tokens that are subsequently verified by the target model. The effectiveness of this paradigm critically depends on the quality of the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ConFu: Contemplate the Future for Better Speculative Sampling**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **speculative decoding** 方法（如 EAGLE 系列）虽然能加速大语言模型（LLM）推理，但其 draft model 仅基于当前生成前缀（prefix）进行预测，导致在多步生成中出现 **error accumulation**（误差累积）。这种偏差会使得 draft token 分布逐渐偏离 target model 的分布，降低 token 接受率（acceptance rate），从而限制加速效果。

### **提出的新方法与新思路**
本文提出了 **ConFu**（Contemplate the Future），一种全新的 speculative decoding 框架，其核心思想是让 draft model 能够“预判”target model 的未来生成方向，即感知其当前的“思考”状态（latent reasoning 或 semantic trajectory）。

#### **三大创新点：**
1. **Contemplate Tokens 与 Soft Prompts**  
   - 引入 **contemplate tokens**（也称 pause tokens）并结合可学习的 **soft prompts**，使 target model 在不微调的前提下输出轻量级的未来信号（future prediction vectors）。
   - 这些信号作为辅助输入提供给 draft model，引导其生成更符合 target model 语义轨迹的候选 token。
   - 推理开销极小，因 contemplate tokens 可并行处理。

2. **基于 MoE 的动态 Contemplate Token 机制**  
   - 提出使用 **Mixture-of-Experts (MoE)** 架构动态生成 contemplate token 和 future token 的嵌入。
   - 根据上下文自适应选择最合适的“提示指令”，提升对未来方向预测的准确性与鲁棒性。
   - 是首次将动态性引入 pause token 设计的工作。

3. **高效的训练框架：Anchor Token Sampling + Future Prediction Replication**  
   - **Anchor Token Sampling**：只对部分采样位置插入 contemplate tokens，显著减少训练时内存消耗。
   - **Future Prediction Replication**：鼓励邻近 token 共享相同的 future prediction，增强模型对局部扰动的鲁棒性，无需额外损失函数。

### **相比现有方法的优势**
- 相较于 EAGLE-3，ConFu 显著提升了 token 接受率和端到端生成速度（speed-up ratio）。
- 首次将 speculative decoding 与连续的 latent “thought” 表示显式结合，为 LLM 推理加速开辟了新方向。
- 不修改 target model，保持原始采样分布不变，安全且兼容性强。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：ShareGPT 和 UltraChat-200K 指令数据集。
- **评估基准**：**SpecBench**（Xia et al., 2024），涵盖多种下游任务：
  - Writing (WRIT)
  - Question Answering (QA)
  - Summarization (SUMM)
  - Translation (TRANS)
  - Coding (CODE)
  - Mathematical Reasoning (M/R)

### **实验设置**
- **目标模型（Target Model）**：
  - `Llama-3.2-3B-Instruct`
  - `Llama-3.1-8B-Instruct`
- **Draft Model 架构**：基于 EAGLE-3 的单层 Transformer 结构，并集成 future token 输入。
- **初始化策略**：从已训练好的 EAGLE-3 checkpoint 初始化 ConFu，确保公平比较。
- **硬件配置**：训练使用 8×NVIDIA H100 GPU；测试在单张 H100 上进行，batch size = 1。

### **评估指标**
1. **Average Accepted Draft Length (T)**：每轮验证平均接受的 draft token 数量。
2. **Speed-up Ratio (SR)**：相对于标准自回归解码的端到端推理加速比。

### **对比的基线方法**
- **EAGLE-3**（Li et al., 2025）：当前最先进的 speculative decoding 方法，作为主要 baseline。
- 同时与 Medusa、HASS 等早期方法间接对比（文献指出 EAGLE-3 已显著优于这些方法）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Tables 1 & 2）**

| 模型 | 温度 T | Draft Nodes | Avg. Accept Length ↑ | Speed-up Ratio ↑ |
|------|--------|-------------|------------------------|------------------|
| EAGLE-3 (3B) | 0.0 | 30 | ~4.04 | ~2.06 |
| **ConFu (3B)** | 0.0 | 30 | **~4.52** | **~2.45** |
| EAGLE-3 (8B) | 0.0 | 30 | ~4.39 | ~2.36 |
| **ConFu (8B)** | 0.0 | 30 | **~5.03** | **~2.69** |

> ✅ **总体提升幅度**：  
> - **Accept Length 提升 8–11%**
> - **Speed-up Ratio 提升约 8–11%**
> - 在所有任务类别、温度设置和计算预算下均一致优于 EAGLE-3。

### **详细观察结果**
- **低温度下优势更明显**：在 greedy decoding（T=0）时，ConFu 的提升最大。原因在于此时 target model 输出更确定，future direction 更容易被准确预测。
- **不同 draft tree 大小均有效**：在 30 和 60 个 draft nodes 设置下均有稳定增益。
- **各任务类别全面领先**：Writing、QA、Coding、Mathematical Reasoning 等任务中均表现优异。

### **消融实验结果（Table 3）**
| 方法变体 | Avg. Accept Length (8B, T=0) | Speed-up Ratio |
|---------|-------------------------------|---------------|
| EAGLE-3 | ~4.39 | ~2.36 |
| ConFu w/o MoE & Replication | ~4.77 | — |
| ConFu w/ MoE only | ~4.97 | ~2.67 |
| **ConFu (完整版)** | **~5.03** | **~2.69** |

> 🔍 **发现**：
> - 加入 **Future Prediction Replication** → Accept Length 提高约 **+0.17**
> - 引入 **MoE 动态机制** → Accept Length 再提高 **+0.05**，SR 提高 **+0.02**
> - 两者共同作用带来显著增益，验证了设计有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **未来感知显著缓解 error accumulation**：通过让 draft model 获取 target model 的 future-oriented signals（即“思考”状态），可以有效对齐生成路径，减少拒绝率。
2. ✅ **Contemplate Tokens 成本极低但收益高**：利用 pause tokens + soft prompts 实现轻量级 future prediction，在几乎无额外延迟的情况下实现性能跃升。
3. ✅ **动态 MoE 提升上下文适应能力**：固定 contemplate token 难以应对多样化任务，而 MoE 可根据不同语境自动切换“提示风格”，提升泛化性。
4. ✅ **训练策略高效实用**：Anchor sampling 和 prediction replication 共同实现了高性能、低内存的训练流程。

### **方法的局限性**
- 当前 contemplate tokens 的数量与 draft tree 规模成正比，可能影响大规模树结构下的扩展性。
- MoE 增加了一定参数量（尽管推理成本可控），在极端资源受限场景需权衡。
- 实验集中在 Llama-3 系列模型，尚未验证在其他架构（如 Mistral、Qwen）上的普适性。

### **未来工作方向**
- 探索如何进一步减少 inference 中所需的 contemplate tokens 数量（例如利用 future prediction 的鲁棒性进行稀疏化）。
- 将 ConFu 思想扩展至多模态 LLM 的 speculative decoding。
- 研究更复杂的 latent reasoning modeling 方式，如迭代式 future prediction。
- 探索在边缘设备上的部署优化方案，推动 real-time 应用落地。

---

> 📌 **一句话总结**：  
> **ConFu 首次将 speculative decoding 与 latent reasoning 显式结合，通过“展望未来”的机制显著提升 draft model 准确性，在几乎零推理代价下实现比 EAGLE-3 高出 8–11% 的加速效果，为 LLM 高效推理提供了全新范式。**

</details>

---

### 4. [LooComp: Leverage Leave-One-Out Strategy to Encoder-only Transformer for Efficient Query-aware Context Compression](https://arxiv.org/abs/2603.09222)

**Authors**: Thao Do, Dinh Phu Tran, An Vo, Seon Kwon Kim, Daeyoung Kim  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.09222v1  

#### Abstract
Efficient context compression is crucial for improving the accuracy and scalability of question answering. For the efficiency of Retrieval Augmented Generation, context should be delivered fast, compact, and precise to ensure clue sufficiency and budget-friendly LLM reader cost. We propose a margin-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LooComp: Leverage Leave-One-Out Strategy to Encoder-only Transformer for Efficient Query-aware Context Compression

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Retrieval-Augmented Generation (RAG)** 系统中，随着检索到的上下文长度增加，虽然信息覆盖更全，但也带来了以下挑战：
- **计算开销大**：长上下文显著增加 LLM 推理成本和延迟；
- **信息冗余与干扰**：大量无关文本可能分散模型注意力，降低问答准确率；
- **压缩效率与保真度难以平衡**：现有压缩方法要么速度慢（如生成式），要么精度低或缺乏查询感知能力（如传统抽取式）。

本文旨在设计一种**高效、精准、轻量级的 query-aware 上下文压缩方法**，在不牺牲问答性能的前提下，实现高吞吐、低内存的实时压缩。

---

### 提出了什么新方法或新思路

作者提出 **LooComp**，一个基于 **Leave-One-Out (LOO) Delta Scoring** 和 **Encoder-only Transformer** 的新型上下文压缩框架，其核心创新如下：

#### （1）LOO-△ Clue Richness Scoring
- 不直接对句子进行二分类判断是否相关，而是通过 **leave-one-out 策略**衡量每个句子被移除后对整体“线索丰富度”（clue richness）的影响。
- 定义 $\Delta_k = f(q, P) - f(q, P \setminus \{s_k\})$，其中 $f$ 是一个可学习的评分函数，$\Delta_k$ 越大表示该句越关键。
- 这种方式能捕捉句子的**边际贡献**，比静态 relevance 分类更具语义敏感性。

#### （2）Margin-based Composite Ranking Loss
- 设计了一个复合损失函数，结合：
  - **Ranking Loss**：拉大关键句与非关键句之间的 $\Delta$ 差距；
  - **Classification Loss (BCE)**：确保完整上下文得分高、空上下文得分低；
  - **Critical Drop Constraint**：强制关键句删除时必须引起显著分数下降。
- 引入多个 margin 超参（$m_1, m_2, m_3$）控制不同类别间的间隔。

#### （3）Adaptive Gap-based Threshold Selection
- 在推理阶段，根据 $\Delta$ 值分布中的**自然间隙**（gap）动态设定保留阈值：
  - 对 $\Delta_k > \delta_{\min}$ 的句子排序并计算相邻差值；
  - 找到最大 gap 的位置 $i^*$，设阈值为 $T = \max(\delta_{\min}, \Delta_{[i^*+1]})$。
- 实现**自适应压缩率**，无需手动配置压缩比例。

#### （4）Lightweight Encoder-only Architecture
- 使用 **ModernBERT**（encoder-only）作为骨干网络，而非 decoder-based LLM（如 Gemma、Llama）。
- 显著降低内存占用和推理延迟，更适合分类型任务。

---

### 相比现有方法的优势

| 维度 | LooComp | 现有方法（如 EXIT、Provence、RECOMP） |
|------|--------|-------------------------------|
| 架构效率 | ✅ Encoder-only，低内存、高速度 | ❌ 多数使用 decoder-based LLM，资源消耗高 |
| 查询感知 | ✅ 基于全局上下文评估单句重要性 | ⚠️ 部分方法忽略上下文依赖 |
| 冗余过滤 | ✅ 利用 LOO-Δ 精确识别关键线索 | ❌ 固定阈值或 token-level 判别易引入噪声 |
| 自适应性 | ✅ 动态选择保留句子数量 | ❌ 多需预设压缩率或 top-k |
| 推理速度 | ✅ 平均 <0.2s @ top-20 | ❌ 如 CompAct 达 4–6 秒 |
| 性能表现 | ✅ SOTA 或接近 SOTA 的 EM/F1 | ⚠️ 多数在压缩后出现明显性能下降 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

共使用 **5 个标准 QA 数据集**，涵盖单跳与多跳场景：

| 类型 | 数据集 |
|------|-------|
| 单跳 QA | **Natural Questions (NQ)**, **TriviaQA (TQA)** |
| 多跳 QA | **HotpotQA (HQA)**, **2WikiMultihopQA (2Wiki)**, **Musique** |

所有实验均在跨域设置下测试泛化能力（仅在 HQA 上训练，在其他数据集上 zero-shot 测试）。

---

### 实验设置和评估指标

#### RAG Pipeline 结构
```
Query → Retriever (Contriever-MSMARCO) → Retrieved Chunks → Compressor → Compressed Context → Reader (LLM) → Answer
```

#### 主要组件
- **Retriever**: Contriever-MSMARCO + Wikipedia 2018 corpus
- **Compressor**: Proposed LooComp vs baselines
- **Reader Models**:
  - 开源：`Llama-3.1-8B-Instruct`, `Llama-3.3-70B-Instruct`
  - 闭源：`Gemini-2.5-flash`, `GPT-5-mini`, `Kimi-K2`

#### 评估指标

| 指标 | 含义 |
|------|------|
| **EM (Exact Match)** | 完全匹配答案的比例 |
| **F1 Score** | 答案词级别重叠的 F1 |
| **Compression Latency (Time ↓)** | 压缩耗时（秒），端到端测量 |
| **Compression Ratio (Rate ↓)** | 压缩后 token 数 / 原始 token 数 |
| **Context Saved (%)** | $100\% - \text{Rate}$ |
| **Questions Per Second (QpS)** | 单位时间内处理的问题数 |

每项实验运行 3 次取平均，batch size 变化以模拟真实负载。

---

### 基线方法对比

共比较 **7 种主流压缩器**：

| 方法 | 类型 | 模型 | 特点 |
|------|------|------|------|
| **RECOMP-abs** | Abstractive | T5-based | 生成摘要，压缩强但慢 |
| **RECOMP-ext** | Extractive | Dual Contriever | 快速但保留过多内容 |
| **CompAct** | Iterative | Mistral-7B | 高质量但极慢（>2.5s） |
| **Refiner** | Rewriting | Llama2-7B | 改写增强，资源密集 |
| **LongLLMLingua** | Prompt Compression | Llama2-7B | 动态压缩率，较优 baseline |
| **EXIT** | Extractive | Gemma-2B | 全文感知，但 decoder 架构重 |
| **Provence** | Token-level | Custom LLM | token 级剪枝，信号噪声大 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 2）

#### 整体平均性能（Table 2）
| Method | EM ↑ | F1 ↑ | Time ↓ (s) | Rate ↓ (%) |
|--------|------|------|------------|-------------|
| CompAct | 32.2 | 41.6 | 3.670 | 7.7 |
| EXIT | 32.0 | 41.3 | 0.921 | 46.2 |
| RECOMP-abs | 30.4 | 39.7 | 0.910 | 4.3 |
| RECOMP-ext | 29.1 | 38.0 | 0.023 | 30.4 |
| Refiner | 31.6 | 40.9 | 3.218 | 8.5 |
| LongLLMLin | 32.3 | 41.7 | 0.853 | 41.8 |
| Provence | 32.4 | 42.0 | 0.126 | 17.9 |
| **Ours (LooComp)** | **34.0** | **43.6** | **0.098** | **12.8** |

✅ **LooComp 在 EM 和 F1 上全面领先**，且压缩速度快（第二快），压缩率适中偏优。

---

#### Top-20 设置下的综合表现（Fig. 1 / Table 7）
| Method | EM | F1 | QpS | Saved% |
|--------|----|----|-----|--------|
| Ours | **32.4** | **41.4** | **6.3** | **90.3%** |

- **QpS 最高**（仅次于 RECOMP-ext），说明极高吞吐；
- **Saved% 第二高**，优于 EXIT、Provence 等；
- **性能-效率帕累托前沿最优**（见 Fig. 1 雷达图）。

---

#### 在闭源 LLM 上的表现（Table 6）
使用 `Gemini-2.5-flash` 和 `Kimi-K2` 作为 reader：
- 在 HQA 上，LooComp 达到 **36.5 EM / 47.5 F1**，显著高于 LongLLMLingua（34.3/45.4）；
- 压缩率仍保持 **8.5%**，远低于 LongLLMLingua 的 **39.1%**；
- 表明其压缩后的上下文更能被强大 LLM 有效利用。

---

### 与基线方法的对比结果

| 对比维度 | 结果 |
|---------|------|
| **vs RECOMP-abs/ext** | 快 10–40 倍，同时保持更高 EM/F1；RECOMP-ext 虽快但性能差 |
| **vs CompAct/Refiner** | 速度提升 >30x，EM/F1 更高或相当 |
| **vs EXIT/LongLLMLingua** | 速度更快（≈10×），压缩更紧凑（Rate 12.8% vs ~40%），性能更强 |
| **vs Provence** | 更快（0.098s vs 0.126s），更准（EM +1.6），压缩更彻底（12.8% vs 17.9%） |

> ✅ LooComp 实现了 **“三高”平衡**：高准确性、高效率、高压缩率。

---

### 消融实验结果（Ablation Studies）

#### （1）Loss Components 消融（Table 3）
| 变体 | EM ↓ | F1 ↓ | 影响 |
|------|------|------|------|
| Full (完整损失) | 34.0 | 43.6 | ✅ 最佳 |
| -BCE | 32.5 | 41.8 | ❌ 性能显著下降 |
| -crit | 33.8 | 42.9 | ⚠️ 仍有影响 |
| -BCE-crit | 31.0 | 39.5 | ❌ 最差 |

➡️ **BCE 和 Ccrit 都至关重要**，尤其 BCE 对整体稳定性起关键作用。

#### （2）Backbone Size 消融（Table 4）
| 模型 | EM | F1 | Latency (ms) |
|------|----|----|--------------|
| ModernBERT-large (395M) | 36.8 | 51.8 | 78.4 |
| ModernBERT-base (139M) | 33.2 | 46.5 | 38.1 |

➡️ 大模型性能更好，小模型速度快近 **2 倍**，提供灵活部署选项。

#### （3）Inference Strategy 消融（Table 5）
| 策略 | EM | F1 |
|------|----|----|
| Adaptive-gap（默认） | 36.8 | 51.8 |
| Margin-based（固定 margin） | 36.5 | 50.4 |

➡️ **自适应 gap 阈值优于固定 margin**，验证了其泛化优势。

---

## 4. 关键结论和发现

### 论文的主要发现

1. ✅ **Encoder-only 模型足以胜任 context compression 任务**，无需昂贵的 decoder-based LLM。
2. ✅ **LOO-Δ scoring 是一种更合理、更具解释性的句子重要性度量方式**，优于传统 relevance classification。
3. ✅ **Adaptive gap-based thresholding 实现了无需调参的动态压缩**，适应不同 query 复杂度。
4. ✅ **LooComp 在多个维度上超越现有 SOTA 方法**，实现了性能与效率的最佳权衡。
5. ✅ 方法具有良好的 **zero-shot 泛化能力**，即使只在 HQA 上训练，也能在其他数据集上表现优异。

---

### 方法的局限性

1. 🚫 **依赖显式的 sentence-level annotation**：
   - 当前训练依赖 HQA 中人工标注的关键句子；
   - 若改用 LLM 自动生成标签，会带来额外成本和可靠性风险。

2. 🚫 **粒度限制在 sentence-level**：
   - 无法处理长而复杂的句子内部冗余；
   - 未来可探索 clause- 或 phrase-level 剪枝。

3. 🚫 **Flash Attention 依赖**：
   - 性能优势部分依赖于 ModernBERT 的 flash-attention 支持；
   - 在不支持的硬件上可能略有退化。

---

### 未来工作方向

1. 🔁 探索 **finer-grained pruning**（短语/从句级）以进一步优化压缩粒度；
2. 🤖 研究 **LLM-as-judge 自动生成训练标签** 的可行性与鲁棒性；
3. 🔄 将本框架扩展至 **multi-modal RAG** 场景（图文混合压缩）；
4. 📈 探索 **online adaptation** 机制，使压缩策略能随 reader 模型变化自动调整；
5. 💡 结合 **retrieval 与 compression 联合优化**，构建端到端高效 RAG 系统。

---

> 🔗 **代码与模型已开源**：https://github.com/thaodod/LooComp

--- 

✅ **总结一句话**：  
LooComp 提出了一种基于 **LOO-Δ + Encoder-only** 的轻量级上下文压缩框架，在保持甚至提升问答准确率的同时，实现了**当前最快的压缩速度和极具竞争力的压缩率**，是面向实际 RAG 应用的理想解决方案。

</details>

---

### 5. [Efficient Reasoning at Fixed Test-Time Cost via Length-Aware Attention Priors and Gain-Aware Training](https://arxiv.org/abs/2603.09253)

**Authors**: Rian Atri  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.09253v1  

#### Abstract
We study efficient reasoning under tight compute. We ask how to make structured, correct decisions without increasing test time cost. We add two training only components to small and medium Transformers that also transfer to broader differentiable optimizers. First, a length aware attention prior bu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Reasoning at Fixed Test-Time Cost via Length-Aware Attention Priors and Gain-Aware Training**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
该论文聚焦于在**严格固定的测试时计算成本（test-time cost）下提升模型的推理效率与准确性**。具体而言，目标是在不增加推理延迟和内存占用的前提下，增强小/中等规模 Transformer 模型在长序列、低信噪比场景下的结构化推理能力。

传统方法常通过引入可学习的位置编码（如 Rotary、Relative）、复杂架构（如 MoE）或动态调整注意力机制来改进性能，但这些通常会带来额外参数或更高的推理开销。本文旨在实现“更强的结构归纳偏置”而不牺牲部署效率。

---

### 🛠️ **提出了什么新方法或新思路**

作者提出两个**仅在训练阶段生效、推理时不运行**的组件，从而在保持零额外推理成本的同时提升模型表现：

#### （1）**Length-Aware Attention Prior via Regime-Position Alignment (RPA)**  
- 利用模糊聚类（fuzzy Gaussian memberships）将每个 token 映射到若干“regime”（语义/位置模式），例如局部、远距离、前缀等。
- 构建一个长度感知的软位置基（soft raised-cosine blocks），并通过 Sinkhorn 对齐 regime 分布与位置分布，生成一个**预 softmax 的注意力先验矩阵 `B(T)`**。
- 该先验作为加法偏置加入 attention logits，在训练中引导注意力关注更有意义的 token 对。
- **无新增参数**，且推理时只需加载一个预先缓存的固定偏置矩阵。

#### （2）**Gain-Aware Controller（Guardian）**
- 一个轻量级的 policy-gradient 控制器（两层 MLP），观察验证集交叉熵变化、注意力饱和度、成员熵等信号。
- 动态调节 attention temperature `T_att`，仅当确认能带来边际收益时才收紧注意力（降低温度），否则放松以避免过拟合。
- 完全**训练专用**，推理阶段关闭，不影响 latency 或 memory。

#### （3）**Late-Phase Optimization Enhancements**
- 使用非零学习率底座（non-zero LR floor）、选择性 SWA（Selective Stochastic Weight Averaging）以及上下文长度博弈（Context Game）进一步稳定后期优化。

---

### ⚖️ **相比现有方法的优势**

| 特性 | 本文方法 | 典型替代方案 |
|------|--------|-------------|
| 推理参数增长 | ❌ 无新增参数 | ✅ 如 ALiBi、RoPE 引入 learnable 参数 |
| 推理延迟影响 | ❌ 可忽略（单次 bias add） | ✅ 复杂 routing/MoE 增加延迟 |
| 结构归纳偏置来源 | ✅ 数据驱动 + 长度自适应 | ❌ 固定 sinusoid 或 hand-designed heuristics |
| 训练稳定性 | ✅ KL 正则视角 + 渐进 warm-in | ❌ 简单调参易导致 collapse |

> ✅ **核心优势：在完全保持 baseline 推理成本的前提下，显著提升长序列建模能力和验证性能。**

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
- 主要任务：**WikiText-2 (raw-v1)**，使用 GPT-2 BPE tokenizer。
- 输入格式：
  - 训练：随机连续 chunk（contiguous chunks）
  - 验证/测试：顺序非重叠 chunk（sequential, no overlap, stride=context length）

> 注：文中也提供了时间序列模板（SlidingWindowTS），但未用于主实验。

---

### ⚙️ **实验设置和评估指标**

#### 模型配置
- 小型 Transformer：`d_model=510`, `L=12`, `H=6`, `R=4`（regimes 数）
- Dropout: 0.09
- Label smoothing: 0.015
- 使用 Fuzzy-Gated MoE FFN 层辅助路由

#### 训练细节
- **硬件**：单块 GH200 GPU
- **Batch Size**: 48 × 512 → ~24,576 tokens/step
- **总步数**：8030 steps (~110 epochs)
- **Optimizer**: AdamW，带 EMA 和 Selective SWA
- **Learning Rate Schedule**: Flat + Cosine decay 至非零底座（5–10% peak LR）

#### 上下文长度策略（Context Game）
- 维护一个关于 context length 的分布 `q(c)`，通过 replicator dynamics 更新：
  - 候选长度：`{256, 512, 1024}` 或固定 `768`
  - 效用函数综合考虑 loss、saturation fraction、entropy
- 目标是让模型在多种长度上均衡学习，提高 RPA prior 的泛化性

#### 评估指标
- **Validation Cross-Entropy (CE)** ↓
- **Perplexity (PPL)** ↓
- **Latency**: p50/p95 step time（训练 & 推理）
- **消融路径**：逐步添加 RPA、Guardian、SWA-select 等模块

#### 基线对比原则
- **Compute Parity**：严格对齐参数量、context length、tokens/step、optimizer、wall-clock 时间
- 所有比较均基于三种子实验平均值

---

## 3. **主要实验结果和性能指标**

### 🔢 **关键性能数据**

#### 在 **WikiText-2 (raw-v1)** 上的结果（Table 4 & Table 3）

| Context (tokens) | Val CE ↓ | PPL ↓ | 相对改进 |
|------------------|----------|-------|---------|
| 512              | 5.4547   | ~233.9 | —       |
| **768**          | **5.2461** | **~189.8** | **-3.8% CE, -18.8% PPL** |

> ✅ 这是本文最佳结果：**Fuzzy-Gated + RPA 方法在 context=768 下达到 5.246 的验证 CE**

---

### 🆚 **与基线方法对比**

| 方法 | Val CE | 是否新增推理参数 | 推理延迟影响 |
|------|--------|------------------|------------|
| Baseline（无 RPA/Guardian/SWA） | 5.850 | 否 | 相同 |
| Sinusoid-only / Relative-only priors | ~5.5–5.6 | 否 | 相同 |
| **本文完整方法（RPA + Guardian + SWA-select）** | **5.246** | ❌ 否 | ❌ 无显著变化 |

> 💡 在相同 compute budget 下，本文方法比 baseline 降低 **0.6 CE**，相当于相对改善约 **10.2%**。

---

### 🔍 **消融实验结果（Table 5）**

| 阶段 | Val CE ↓ | Δ vs Base | Δ from Prev |
|------|----------|-----------|-------------|
| Baseline | 5.850 | 0.00 | — |
| + Context Game + Sinkhorn Align | 5.536 | -0.31 | -0.31 |
| + Guardian + Late-phase Schedules | 5.455 | -0.40 | -0.08 |
| **+ SWA-select (Final)** | **5.246** | **-0.60** | **-0.21** |

> 🔍 发现：
> - RPA 贡献最大初期增益（-0.31）
> - Guardian 提供谨慎调控，防止过紧
> - **SWA-select 是后期飞跃的关键**（-0.21）

---

### ⏱️ **延迟分析（Table 2 & Section 8.3）**

| Run | Context Mix | Mean s/it | p50 Latency |
|-----|-------------|-----------|-----------|
| Baseline (512) | 512:1.0 | 25.90s | 25.71s |
| Full Run (768) | 768:1.0 | 32.79s | 32.62s |
| w/ Context Game | {256,512,1024} | 168.99s | 166.86s |

> ⚠️ 注意：训练 step time 随 context 增加而上升（近线性），但这是 context 本身的影响，而非 RPA/Guardian 导致。

> ✅ **推理延迟实测结果**：
> - 添加 RPA 后，**p50 推理延迟无可观测变化**（within logging resolution）
> - 因为仅需一次 cached bias add per head，无新参数或循环

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **Length-aware attention prior（RPA）有效引导注意力结构**
   - 通过 fuzzy regime clustering 与 soft positional basis 对齐，构建出具有解释性的 attention bias（如 local vs. global stripe patterns）。
   - 在 content logits 噪声大（小模型、少数据）时效果最明显。

2. **KL 正则化视角为 prior 提供理论支撑**
   - `softmax(z + log π)` 等价于在 KL 正则下求解 MAP 解，prior π 起到方向性正则作用。
   - 标准化（z-score）和 clipping 保证 prior 不主导 logits，维持数值稳定。

3. **Guardian 实现“gain-aware”控制，避免过度锐化**
   - 若盲目降低 temperature 会导致 attention 饱和、CE 反弹；Guardian 能检测此趋势并主动放松。
   - 政策梯度更新符合 two-timescale ODE 收敛理论，确保稳定性。

4. **长上下文受益更明显**
   - 从 512 → 768 tokens，CE 下降 3.8%，说明 RPA 在需要 long-span reasoning 的场景更具价值。

5. **所有增强均为 training-only，推理零负担**
   - RPA → 缓存 `B(T)` 作为 additive bias
   - Guardian → 推理关闭
   - SWA → 权重平均后冻结
   - ✅ 最终模型与 baseline 具备相同的 inference complexity

---

### ⚠️ **局限性**

| 方面 | 局限 |
|------|------|
| **任务范围** | 当前仅为 WikiText-2 的语言建模 proof-of-concept，尚未扩展至多任务或 downstream tasks |
| **模型规模** | 效果随模型容量增大而减弱（强 QK 使 prior 影响变小），更适合 small/medium models |
| **Prior Expressivity** | `R` 较小时 prior 低秩，可能无法捕捉细粒度结构；可通过混合 sinusoid 缓解 |
| **控制器表达力** | Guardian 仅控制全局 scalar temperature，缺乏 per-head 或 layer-wise 控制能力 |
| **运行时开销声明** | 不声称 zero overhead，仅强调“negligible”，未提供 micro-benchmark 报告 |

---

### 🔮 **未来工作方向**

1. **扩展至更大模型与更多任务**（如 QA、summarization）
2. **设计 per-head 或 hierarchical Guardian 控制器**
3. **结合其他 differential optimizer 结构**（如 Differentiable Pooling、Neural Execution）
4. **探索 RPA 在 vision transformers 或 multimodal setting 中的应用**
5. **自动化 regime number `R` 的选择机制**

---

## ✅ 总结一句话

> 本文提出 **RPA + Guardian** 框架，在**不增加任何推理成本**的前提下，利用训练期的数据驱动注意力先验与增益感知控制器，显著提升了小/中型 Transformer 在长文本上的推理准确性和稳定性，尤其适用于资源受限场景下的高效 reasoning 系统设计。

</details>

---

### 6. [MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games](https://arxiv.org/abs/2603.09022)

**Authors**: Yunfei Xie, Kevin Wang, Bobby Cheng, Jianzhu Yao, Zhizhou Sha, Alexander Duffy, Yihan Xi, Hongyuan Mei, Cheston Tan, Chen Wei, Pramod Viswanath, Zhangyang Wang  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09022v1  

#### Abstract
Multi-turn, multi-agent LLM game evaluations often exhibit substantial run-to-run variance. In long-horizon interactions, small early deviations compound across turns and are amplified by multi-agent coupling. This biases win rate estimates and makes rankings unreliable across repeated tournaments. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

多轮、多智能体的 **LLM 游戏**（multi-agent LLM games）在评估中表现出显著的 **run-to-run 方差**（运行间波动）。由于早期决策的小偏差会在长周期交互中不断累积，并通过多智能体耦合被放大，导致胜率估计偏倚、排名不稳定，影响可复现性和公平比较。

此外，**prompt 的微小变化**会引发策略漂移甚至模型排名反转（见附录 A），使得固定 prompt 的评估不可靠。

### 提出了什么新方法或新思路

提出 **MEMO**（Memory-augmented MOdel context optimization），一种无需更新模型权重的 **self-play 框架**，通过优化推理时上下文（inference-time context）来提升多轮多智能体游戏中的鲁棒性与性能。

其核心是将 **探索**（Exploration）与 **保留**（Retention）相结合：

- **Retention（保留）**：维护一个持久的 **memory bank**，存储从 self-play 轨迹中提炼出的结构化洞察（如策略先验、规则澄清），并在后续游戏中作为先验注入。
- **Exploration（探索）**：
  - 采用 **tournament-style prompt evolution**，通过 TRUESKILL 对候选上下文进行不确定性感知的选择。
  - 引入 **prioritized replay**，优先回放稀有且决定性的状态，以提高轨迹覆盖度。

### 相比现有方法的优势

| 方法 | 局限性 | MEMO 的优势 |
|------|--------|-------------|
| **静态提示**（CoT, ToT） | 固定不变，无法适应失败模式 | 动态优化上下文，持续改进 |
| **自动 prompt 优化**（TextGrad, MIPRO, GEPA） | 缺乏跨轮次记忆，每次更新丢弃历史经验 | 通过 memory bank 实现知识积累，形成“累积学习”过程 |
| **强化学习**（RL） | 需要大量环境交互（sample-inefficient），依赖稀疏奖励，训练不稳定 | 仅用 2,000 场 self-play，比 RL 基线少 19× 交互量，且更稳定 |

---

## 2. 核心实验方法和设置

### 使用的数据集

基于五个文本游戏，来自 **TextArena** 和 **SPIN-Bench**，分为三类：

| 类别 | 游戏 |
|------|------|
| **Negotiation** | `SimpleNegotiation`, `TwoDollar` |
| **Imperfect Information** | `KuhnPoker`, `Briscola` |
| **Perfect Information** | `SimpleTak` |

这些游戏涵盖合作、不完全信息博弈、长期规划等核心挑战。

### 实验设置和评估指标

#### 基础模型
- `GPT-4o-mini`
- `Qwen-2.5-7B-Instruct`

#### 评估协议
- 每种方法进行 **3 次独立运行**，生成最终上下文。
- 在固定对手池上评估（`Grok-4-Fast-Non-Reasoning`, `Gemini-2.5-Flash-Lite`, `Qwen3-235B-A22B-Instruct-2507`），每对组合玩 50 局，交换先后手。
- 报告 **平均胜率**（mean win rate）和 **相对标准误差**（RSE）：
  $$
  \text{RSE}(\%) = 100 \times \frac{\text{std}(x_1,\dots,x_n)}{\text{mean}(x_1,\dots,x_n)\sqrt{n}}
  $$
  RSE 越低，表示运行间稳定性越高。

#### 自优化参数
- 人口大小 $N=8$，代数 $G=5$，每代每候选 50 局 → 总共 **2,000 场 self-play**
- 使用 **TRUESKILL** 进行选择，得分公式为 $S(c) = \mu_c - K\sigma_c$
- Memory 注入比例 $\eta = 0.75$
- Replay 参数：缓冲区大小 $B=100,000$，优先指数 $\alpha=0.6$，回放概率 $\beta=0.4$

### 基线方法对比

| 类型 | 方法 |
|------|------|
| **Static Prompting** | `baseline`, `CoT`, `ToT` |
| **Prompt Optimization** | `TextGrad`, `MIPRO`, `GEPA` |
| **Reinforcement Learning** | `UnstableBaseline`, `SPIRAL` |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 模型 | 方法 | 平均胜率 | RSE |
|------|------|--------|-----|
| `GPT-4o-mini` | Baseline | 25.1% | 44.9% |
| `GPT-4o-mini` | **MEMO (Ours)** | **49.5%** | **6.4%** |
| `Qwen-2.5-7B-Instruct` | Baseline | 20.9% | 30.1% |
| `Qwen-2.5-7B-Instruct` | **MEMO (Ours)** | **44.3%** | **6.1%** |

> ✅ **MEMO 将平均胜率提升近一倍，同时将 RSE 降低约 7 倍**，极大提升了评估稳定性。

### 与基线方法的对比结果

| 方法 | GPT-4o-mini 平均胜率 | 提升（vs baseline） |
|------|------------------|------------------|
| `TextGrad` | 34.6% | +9.5% |
| `MIPRO` | 36.7% | +11.6% |
| `GEPA` | 32.0% | +6.9% |
| **MEMO** | **49.5%** | **+24.4%** |

- MEMO 显著优于所有 prompt optimization 方法。
- 在 **不完全信息游戏**（如 `KuhnPoker`, `TwoDollar`）中表现尤为突出。
- 在 `KuhnPoker` 上，MEMO 仅用 **2,000 场游戏**即达到 60% 胜率，而 RL 基线需 **38,000 场**（快 **19 倍**）。

### 消融实验结果（Ablation Study）

| 配置 | KuhnPoker | Briscola | TwoDollar | 平均胜率 | 提升 |
|------|----------|---------|-----------|--------|------|
| 仅 Tournament（无 memory） | 54.7% | 2.0% | 24.7% | 27.1% | +3.3% |
| 仅 Memory（无 tournament） | 57.2% | 22.2% | 34.2% | 34.2% | +10.4% |
| Tournament + Replay | 54.2% | 38.7% | 32.0% | 41.6% | +17.8% |
| **Tournament + Memory** | **57.2%** | **38.4%** | **48.7%** | **48.1%** | **+24.3%** |
| **完整 MEMO** | **55.6%** | **42.7%** | **52.4%** | **50.2%** | **+26.4%** |

> 🔍 **关键发现**：**Memory 是主导因素**，探索（tournament/replay）必须与记忆结合才能实现最大增益。

---

## 4. 关键结论和发现

### 论文的主要发现

1. **上下文敏感性严重**：即使语义相同的 prompt 变体也会导致模型排名反转（Kendall's τb 下降至 -0.5），说明必须进行多 prompt 评估。
2. **持久记忆是关键**：单纯的 prompt 探索收益有限；只有通过 memory bank 实现跨轮次知识积累，才能将上下文优化变为“累积学习”。
3. **MEMO 极具样本效率**：相比 RL 方法，仅用 **1/19 的交互次数**即可达到相当或更强性能，尤其在不完全信息游戏中。
4. **泛化能力**：
   - 学习到的上下文可在不同游戏间零样本迁移（zero-shot transfer），如 `SimpleTak → KuhnPoker` 提升 +25.9%。
   - 但在模型架构间迁移效果不一：对较弱模型（如 `Gemini-2.5-Flash-Lite`）普遍有益，对较强模型可能产生负迁移（negative transfer）。

### 方法的局限性

- **在完全信息游戏**（如 `SimpleTak`）中，RL 仍更具优势，表明当前 context optimization 在纯规划任务中仍有天花板。
- 泛化具有 **方向不对称性**，并非所有迁移都有效，依赖源与目标游戏的结构对齐。
- 依赖高质量的 trajectory reflection 和 memory CRUD 操作，若 LLM 本身推理能力弱，则 memory 质量受限。

### 未来工作方向

- 扩展至更多样化的多智能体场景（如多方谈判、团队协作）。
- 探索 memory bank 的自动化压缩与检索机制。
- 结合轻量级微调（如 LoRA）与 context optimization，实现“混合改进”。
- 研究如何使 learned context 更好地跨模型迁移，避免负迁移。

---

> 📌 **总结一句话**：  
> **MEMO 证明了通过 memory-augmented context optimization，可以在不更新模型权重的前提下，显著提升多轮多智能体 LLM 游戏的性能与稳定性，尤其在不完全信息任务中展现出超越 RL 的样本效率与鲁棒性。**

</details>

---

### 7. [Robust Regularized Policy Iteration under Transition Uncertainty](https://arxiv.org/abs/2603.09344)

**Authors**: Hongqiang Lin, Zhenghui Fu, Weihao Tang, Pengfei Wang, Yiding Sun, Qixian Huang, Dongxu Zhang  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09344v1  

#### Abstract
Offline reinforcement learning (RL) enables data-efficient and safe policy learning without online exploration, but its performance often degrades under distribution shift. The learned policy may visit out-of-distribution state-action pairs where value estimates and learned dynamics are unreliable. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Robust Regularized Policy Iteration under Transition Uncertainty

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **offline reinforcement learning (offline RL)** 中的核心挑战——**distribution shift** 和 **transition uncertainty** 进行研究。在离线学习中，策略可能访问训练数据未覆盖的 **out-of-distribution (OOD)** 状态-动作对，导致价值函数估计不可靠，从而引发严重的外推误差（extrapolation error）。传统方法通常基于单一动力学模型进行规划，忽略了模型本身的不确定性。

### 提出了什么新方法或新思路
作者提出了一种新的框架：**Robust Regularized Policy Iteration (RRPI)**，其核心思想是将 offline RL 建模为一个 **robust policy optimization** 问题：
- 将 **transition kernel** 视为不确定集合 $ \mathcal{P} $ 内的一个决策变量，而非固定估计值。
- 寻找在最坏情况下的动力学下仍能最大化回报的策略，即求解如下 max-min 优化问题：
  $$
  \pi^* = \arg\max_\pi \min_{p \in \mathcal{P}} J(\pi, p)
  $$
- 为了克服原始 max-min 问题难以直接求解的问题，引入了一个 **KL-regularized surrogate objective**，并设计了对应的 **robust regularized Bellman operator**，从而实现高效的迭代优化。

### 相比现有方法的优势
| 方面 | RRPI 的优势 |
|------|-------------|
| **建模方式** | 显式建模 **transition uncertainty**，而不仅仅是通过 heuristic 的 uncertainty penalty 或保守的价值学习来间接处理。 |
| **理论保障** | 所提出的 robust regularized Bellman operator 是一个 $\gamma$-contraction，保证收敛；且迭代过程能单调提升原 robust 目标。 |
| **算法效率** | 避免了复杂的双层优化，通过 surrogate 目标实现了可扩展的 policy iteration 流程。 |
| **行为特性** | 学到的策略会自然规避高 **epistemic uncertainty** 区域，表现出更强的鲁棒性。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在标准的 **D4RL benchmark** 上进行，涵盖多个 Mujoco 控制任务及其不同数据分布设定：
- **Tasks**: `HalfCheetah`, `Hopper`, `Walker2d`
- **Dataset types per task**:
  - `random`: 完全随机策略收集
  - `medium`: 中等性能策略收集
  - `expert`: 专家策略收集
  - `medium-expert`: 混合中等与专家轨迹
  - `medium-replay`: medium 策略回放缓冲区
  - `full-replay`: 包含所有训练过程中的轨迹

### 实验设置和评估指标
- **评估方式**：使用归一化得分（normalized score）：
  $$
  \text{Normalized Score} = \frac{\text{Agent Score} - \text{Random Score}}{\text{Expert Score} - \text{Random Score}}
  $$
  数值越高表示性能越接近专家水平。
- **实现细节**：
  - 使用 **dynamics model ensemble** 构建 uncertainty set $\mathcal{P}$。
  - 在 Bellman backup 中选择 ensemble 中预测值最小的模型作为“worst-case”近似。
  - 使用神经网络参数化 $Q$ 函数和策略 $\pi$。
  - 引入 KL 正则项以稳定策略更新，并采用目标网络机制。

### 基线方法对比
对比了多种 state-of-the-art 的 model-free 和 model-based 方法：
- **Model-free**:
  - CQL
  - DMG
  - EPQ
- **Model-based**:
  - MOReL
  - RAMBO
  - PMDB
  - ADM

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
在 **18 个 D4RL 环境**上的平均表现显示：
- **RRPI 在多数环境下优于或媲美所有 baseline**。
- 特别是在 `HalfCheetah-Medium-Replay`、`Walker2d-Full-Replay` 等具有复杂分布偏移的任务上表现突出。
- 示例亮点：
  - `HalfCheetah-Medium`: RRPI 达到 **75.2±0.7**，优于 PMDB (**75.6±1.3**) 和 RAMBO (**77.9±4.0**)。
  - `Walker2d-Full-Replay`: RRPI 达到 **107.3±0.4**，显著高于其他方法（如 PMDB: 99.9±3.6）。
  - `Hopper-Full-Replay`: RRPI 达到 **108.6±0.2**，为最佳。

> ✅ **总体结论**：RRPI 在 **11 out of 18 environments** 上超过 PMDB，在其余环境中保持竞争力，表明其 robust optimization 框架相比 percentile-based 方法更具适应性和稳定性。

### 消融实验结果（Ablation Study, Table 2）
移除“最坏情况模型选择”（worst-case model selection），改用随机采样 dynamics model 进行 rollout。

| 结果维度 | 发现 |
|--------|------|
| **性能下降 (%)** | 所有任务均出现显著下降，例如 `Hopper-Medium-Replay` 下降高达 **10.9%**。 |
| **方差增加 (std change)** | 标准差普遍上升 **10–30 倍以上**，说明策略更不稳定、泛化能力差。 |
| **结论** | 最坏情况优化对于提升鲁棒性和性能至关重要，验证了 robust formulation 的有效性。 |

---

## 4. 关键结论和发现

### 论文的主要发现
1. **RRPI 成功将 transition uncertainty 显式纳入优化目标**，通过 robust max-min 框架提升了策略的鲁棒性。
2. **learned Q-values 在高 epistemic uncertainty 区域自动降低**（见 Fig. 2），说明策略能够自适应地避免 OOD 动作，无需显式惩罚项。
3. **正则化 + robust optimization 的结合带来了训练稳定性和性能提升的双重好处**。
4. **理论与实践一致**：所提 operator 具备 contraction 性质，迭代过程单调改进目标，并最终收敛至 robust optimal policy。

### 方法的局限性
- **依赖 dynamics model ensemble** 来近似 uncertainty set，当 ensemble 规模小或拟合不佳时，robust 效果受限。
- 当前实验集中于连续控制任务，尚未在高维视觉输入或多模态场景中验证。
- “worst-case”假设可能导致过度悲观（overly conservative），尤其在 uncertainty 被误估计时。

### 未来工作方向
1. **改进 uncertainty estimation**：缩小理论与实际 uncertainty 衡量之间的差距。
2. **扩展至 multimodal inputs**：如结合 vision 输入（参考文中提及的 Hyperpoint、PointCoT 等工作），构建适用于真实世界的 robust 决策系统。
3. **应用于安全关键领域**：如医疗诊断、能源管理（如 blockchain-enabled V2G）、自动驾驶等，其中 offline learning 与 robustness 至关重要（参见 Appendix B）。

--- 

> 📌 **一句话总结**：  
> RRPI 提出了一种理论上健全、实践中高效的新范式——通过 **robust regularized Bellman operator** 实现对 transition uncertainty 的显式建模，在 D4RL 上取得了优于主流 baseline 的性能与更强的鲁棒性，推动了 model-based offline RL 向更可靠的方向发展。

</details>

---

### 8. [Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning](https://arxiv.org/abs/2603.08999)

**Authors**: Juming Xiong, Kevin Guo, Congning Ni, Chao Yan, Katherine Brown, Avinash Baidya, Xiang Gao, Bradley Marlin, Zhijun Yin  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.08999v1  

#### Abstract
Large language models (LLMs) achieve strong reasoning performance through chain-of-thought (CoT) reasoning, yet often generate unnecessarily long reasoning paths that incur high inference cost. Recent self-consistency-based approaches further improve accuracy but require sampling and aggregating mul...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在进行 **Chain-of-Thought (CoT)** 推理时，虽然能提升复杂任务的准确性，但往往生成过长且不必要的推理路径，导致**推理成本高昂**。  
此外，现有的 **self-consistency (SC)** 方法通过采样多条推理路径并聚合结果来提高鲁棒性和准确率，但进一步加剧了计算开销。

因此，如何在**保持高准确率的同时显著降低推理过程中的 token 消耗**，是当前高效推理面临的核心挑战。

### 🚀 提出的新方法与创新思路
本文提出了一种**基于置信度感知的自适应推理框架**（confidence-aware decision framework），其核心思想是：

> **“只在必要时才进行多路径采样”** —— 即先生成一条贪心（greedy）的 CoT 路径，然后由一个轻量级决策模型分析该路径的质量，并判断是否需要启动更昂贵的多路径推理（如 self-consistency 或 dynamic voting）。

#### 创新点包括：
- **单轨迹置信度估计**：仅依赖一条完整的 greedy CoT 轨迹，提取句级（sentence-level）的数值与语言学特征，预测该路径输出答案的正确概率。
- **可迁移的轻量决策模型**：使用带有注意力机制的 RNN 架构（GRU + 多头自注意力），训练后可在不同数据集上零样本迁移（zero-shot transfer），无需重新训练。
- **动态分流机制**：设定一个置信阈值 $ \tau $，若预测置信度 $ p \geq \tau $，则接受 greedy 输出；否则触发 multi-path reasoning 进行增强。

### 🔍 相比现有方法的优势
| 方法 | 是否需多路径采样 | 是否可提前终止 | 效率优势 | 准确性保障 |
|------|------------------|----------------|----------|------------|
| Standard CoT | 否 | 否 | 高 | 低 |
| Self-Consistency (SC) | 是（固定数量） | 否 | 低 | 高 |
| Dynamic Voting (DV) | 是（动态终止） | 是 | 中等 | 高 |
| **本文方法（Ours）** | **自适应决定** | **是（基于单路径分析）** | **极高** | **与 SC/DV 相当** |

> ✅ 在几乎不损失准确性的前提下，最多减少 **80% 的 token 使用量**，远优于 SC 和 DV。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在四个涵盖医学与通用领域的多选问答数据集上进行了广泛验证：
- **MedQA**：美国医师执照考试风格题，测试医学推理能力
- **MedMCQA**：印度医学入学考试题，规模更大，覆盖广
- **MathQA**：数学应用题，需符号推理
- **MMLU**：跨学科知识评测基准，含 STEM、人文等 57 个主题

### ⚙️ 实验设置
- **模型**：在五种主流开源 LLM 上测试：
  - GPT-OSS 20B（主模型）
  - LLaMA3.1 8B
  - Qwen2.5 7B / Qwen3 14B / Qwen3 32B
- **输入格式**：标准 CoT prompting，每个问题生成完整推理链 + 最终答案
- **决策模型训练方式**：
  - 所有决策模型均在 **MedQA 上训练**
  - 在其他数据集（MathQA、MedMCQA、MMLU）上进行 **zero-shot 应用**
- **评估指标**：
  - **Accuracy**：最终答案准确率
  - **Token Usage**：平均每个样本使用的 token 数量（衡量效率）
  - **Statistical Significance**：采用 paired bootstrap（2000次重采样）检验差异显著性

### 🆚 基线方法对比
| 基线方法 | 简称 | 描述 |
|--------|------|------|
| Greedy CoT | – | 单路径贪心解码 |
| Self-Consistency | SC | 采样10条路径，投票选择最频繁答案 |
| Confidence Enhanced Reasoning | CER | 加权聚合多路径，权重为中间步骤置信度 |
| Dynamic Voting | DV | 动态采样路径，直到达成共识即停止 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 GPT-OSS 20B 为例）

| 数据集 | 方法 | Accuracy (%) | Token Usage (avg.) | Token Reduction vs SC/CER |
|-------|------|---------------|---------------------|----------------------------|
| MedQA | Ours | ~98.0 | ~1026 | ↓79% |
| MathQA | Ours | ~84.7 | ~629 | ↓69% |
| MedMCQA | Ours | ~70.4 | ~983 | ↓77% |
| MMLU | Ours | ~90.0 | ~536 | ↓76% |

> 💡 所有数据集中，**our method 的 accuracy 与 SC、CER、DV 无统计显著差异（n.s.）**，但 token 消耗显著更低（$ p < 0.05 $）。

#### 与各基线相比的平均节省：
- 相比 **SC & CER**：减少 **69–79%** token
- 相比 **DV**：减少 **27–48%** token

> 表明即使 DV 已具备动态终止能力，本文方法仍能进一步大幅压缩冗余计算。

### 🔍 消融实验结果（Ablation Studies）

#### （1）模块消融（表3）
| 模型变体 | 平均 Accuracy ↑ | 平均 Token Usage ↓ |
|---------|------------------|--------------------|
| 无 FA & MHSA | 0.818 | 947 |
| 仅 FA（Feature Attention） | 0.819 | 888 |
| 仅 MHSA（Multi-Head Self-Attention） | 0.819 | 900 |
| **FA + MHSA（完整模型）** | **0.820** | **794（↓16.16%）** |

✅ 结论：两个注意力模块协同作用效果最佳。

#### （2）特征消融（表4）
| 特征组合 | 平均 Accuracy ↑ | 平均 Token Usage ↓ |
|--------|------------------|--------------------|
| 仅 Numeric Features | 0.818 | 884 |
| 仅 Linguistic Features | 0.818 | 911 |
| **Numeric + Linguistic** | **0.820** | **805（↓8.94%）** |

✅ 结论：**数值特征与语言学特征互补**，联合使用可同时提升准确率并降低 token 开销。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **推理轨迹蕴含丰富的不确定性信号**  
   即使只看一条 greedy CoT 的句级概率趋势、熵变化、长度归一化得分等**非语义特征**，也能有效判断其可靠性。

2. **推理模式具有跨领域泛化性**  
   决策模型**仅在 MedQA 上训练**，即可零样本迁移到 MathQA、MedMCQA、MMLU，说明 LLM 的推理动态存在共通规律。

3. **大模型具有更清晰的轨迹区分度**  
   更大的 LLM（如 Qwen3 32B）展现出更强的概率收敛性和更低的不确定性波动，使得置信度预测更加可靠。

4. **效率提升巨大而精度无损**  
   在多个模型和任务上验证，该方法能在维持与 multi-path 方法相当准确率的前提下，**节省高达 80% 的 token**。

---

### ⚠️ 局限性
1. **适用于结构化推理任务**  
   当前方法主要针对 **multiple-choice QA** 类任务，对开放生成、对话系统或长文本摘要的有效性尚待研究。

2. **离线分析而非在线早停**  
   当前框架基于**已完成的 CoT 轨迹**进行分析，无法实现实时 early-exit。下一步可探索因果建模以支持流式判断。

3. **依赖内部信号访问权限**  
   需要获取 LLM 的 token-level logits 和生成概率，在闭源 API 场景下可能受限。

4. **阈值需轻量调优**  
   虽然模型本身无需重训练，但每新到一个数据集仍需校准置信阈值 $ \tau $（可通过小规模验证集完成）。

---

### 🔮 未来工作方向
- 将框架扩展至 **causal streaming mode**，实现真正的在线 early-exit。
- 探索将本方法集成进 **test-time compute scaling** 流程中，与其他推理优化技术结合。
- 研究如何利用外部工具（如 calculator、retriever）反馈进一步增强置信度估计。
- 探索在非选择题任务（如生成解释、写作）上的适用性。

---

## 总结一句话
> 本文提出一种**轻量、可迁移、基于单条 CoT 轨迹的置信度感知机制**，实现了“按需采样”，在**几乎不牺牲准确性的前提下，将推理 token 消耗降低多达 80%**，为高效 LLM 推理提供了简单而强大的解决方案。

</details>

---

### 9. [Accelerating High-Order Finite Element Simulations at Extreme Scale with FP64 Tensor Cores](https://arxiv.org/abs/2603.09038)

**Authors**: Jiqun Tu, Ian Karlin, John Camier, Veselin Dobrev, Tzanio Kolev, Stefan Henneking, Omar Ghattas  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09038v1  

#### Abstract
Finite element simulations play a critical role in a wide range of applications, from automotive design to tsunami modeling and computational electromagnetics. Performing these simulations efficiently at the high resolutions needed for practical applications and scientific insights necessitates the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Accelerating High-Order Finite Element Simulations at Extreme Scale with FP64 Tensor Cores**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
高阶有限元模拟在科学计算中至关重要（如海啸建模、电磁仿真等），但其计算密集型特性使得在极端规模下实现高效运行极具挑战。尽管已有大量工作将有限元代码移植到 GPU 上，**如何进一步提升高阶有限元内核的计算效率和能效**仍是关键瓶颈。

传统基于 CUDA Core 的实现受限于共享内存带宽，尤其在处理小规模密集矩阵乘法（如 `m=25, n=5, k=4`）时，FLOP/Byte 比率极低，导致性能瓶颈。

---

### **提出的新方法与新思路**
本文首次将 **FP64 Tensor Cores** 直接编程用于加速大规模、PDE 驱动的高阶有限元科学计算应用，并结合 **Kernel Fusion** 技术进行优化：

- **直接编程 FP64 Tensor Cores**：利用 NVIDIA Ampere 及更新架构中的 **DMMA（Double Precision Matrix-Multiply-Accumulate）指令**，对有限元算子中的小批量 GEMM 操作进行加速。
- **避免共享内存 Bank Conflict**：
  - 设计最优的逻辑索引映射函数（`fm`, `fn`, `fk`），确保 warp 内线程访问共享内存时不发生 bank 冲突。
  - 引入 **张量索引重排序（Tensor Index Reordering）**，使求和索引成为最快变化维度，从而规避因中间索引求和导致的 bank 冲突。
- **融合内核优化（Fused PA/MF Kernels）**：
  - 将多个连续操作（如 `GBTDBG`）融合为单个 kernel，减少全局内存访问和中间数据传输。
  - 在 Fused PA 中，将 `D` 和 `D^T` 应用合并，显著降低 PA 数据移动开销。

---

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **性能** | 最高实现 **2× 性能提升**（vs 原始 PA kernel） |
| **能效** | 能效提升高达 **83%**（GH200 上） |
| **适用性** | 首次在真实生产级 HPC 应用中直接编程 FP64 Tensor Cores，非仅限于大矩阵 GEMM |
| **可扩展性** | 在近万 GPU 上展示近乎完美的弱扩展性和 90% 的强扩展效率 |

> ✅ **突破性意义**：这是首次将 FP64 Tensor Cores 成功应用于复杂、不规则形状的小矩阵乘法场景下的大规模有限元模拟。

---

## **2. 核心实验方法和设置**

### **使用的数据集与应用背景**
- **应用场景**：基于 **MFEM** 库开发的“海啸早期预警数字孪生”系统（2025 ACM Gordon Bell Prize 获奖应用）。
- **物理模型**：耦合声重力波方程（acoustic-gravity wave equations），描述地震引发的海啸传播。
- **离散方式**：
  - 压力场：四阶连续有限元（H1-conforming）
  - 速度场：三阶间断有限元（L2-conforming）
- **时间推进**：显式四阶 Runge-Kutta（RK4）

> ⚠️ 注意：该应用对精度要求极高（反问题敏感），必须使用 **FP64**。

---

### **实验设置与评估指标**

#### **硬件平台**
- **单卡测试**：
  - NVIDIA **Grace Hopper GH200**
  - NVIDIA **Grace Blackwell GB200**
- **多卡扩展测试**：
  - **Alps 系统**（瑞士国家超算中心 CSCS）
    - 2,688 节点，每节点 4 个 GH200 Superchip
    - 总共最多使用 **9,216 个 GH200 GPU**
    - 理论峰值：574.8 PFLOP/s

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **Throughput (GDOF/s)** | 每秒处理的十亿自由度数（Degrees of Freedom） |
| **Performance per Watt (MDOF/W)** | 单位能耗下的计算吞吐量，衡量能效 |
| **Strong Scaling Efficiency** | 固定总问题规模，增加 GPU 数量时的并行效率 |
| **Weak Scaling Efficiency** | 每 GPU 问题规模固定，随 GPU 数量线性增长时的效率 |
| **Cycle Count / Speedup** | 相对于原始 kernel 的周期减少比例 |

#### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Original PA** | 基于 CUDA Core 的 Partial Assembly 实现 |
| **Fused PA** | 融合多个操作的优化版本，减少内存访问 |
| **Fused MF** | Matrix-Free 版本，完全避免存储预计算数据 |
| **DMMA PA / DMMA Fused PA** | 使用 FP64 Tensor Cores 加速的对应版本 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **单 GPU 性能（5.4 亿 DOF 问题）**

| Kernel | GH200 (GDOF/s) | GB200 (GDOF/s) | 提升幅度 |
|--------|----------------|----------------|----------|
| Original PA | 18.73 | 23.78 | — |
| DMMA PA | 25.27 (+35%) | 33.72 (+42%) | — |
| Fused PA | 24.04 | 29.28 | — |
| **DMMA Fused PA** | **36.15 (+93%)** | **46.60 (+95%)** | **≈1.9× vs PA** |

> 🔥 **最高达 2× 性能增益** 来自 **DMMA + Loop Fusion 联合优化**

#### **能效表现（Performance per Watt）**

| Kernel | GH200 (MDOF/W) | GB200 (MDOF/W) | 提升 |
|--------|----------------|----------------|------|
| Original PA | 28.65 | 26.60 | — |
| DMMA PA | 36.51 (+27%) | 31.37 (+18%) | — |
| **DMMA Fused PA** | **52.42 (+83%)** | **45.70 (+72%)** | ✅ 显著节能 |

> 💡 **GH200 能效优于 GB200** 的原因：
> - GB200 更高频率 + 更高空闲功耗
> - 当前 kernel 未充分利用 GB200 的低精度 Tensor Core 优势（FP8/FP4）

---

### **与基线方法对比结果**

| 对比项 | 结果 |
|-------|------|
| **vs CUDA Core 实现** | 共享内存读取减少 **4.6×**，FLOP/Byte 比率大幅提升 |
| **vs 传统 PA/MF 分离实现** | Fused PA 减少约一半的数据搬运 |
| **vs 仅用 CUBLAS GEMM** | 本文方法适用于不规则小矩阵（batched small GEMMs），更具通用性 |

---

### **消融实验结果**
- **仅启用 DMMA** → 平均提速 **35–59%**
- **仅启用 Kernel Fusion** → 提速 ~30%，但受限于内存带宽
- **DMMA + Fusion 组合** → 实现 **接近 2× 加速**，表明二者有协同效应
- **Profile 分析显示**：
  - 原始 kernel：97% 时间受 **共享内存带宽限制**
  - DMMA kernel：共享内存压力降至 84%，**DMMA 管道利用率达 54%**

> 📉 性能未达理论上限的原因：`m=25,n=5,k=4` 不完美匹配 `m8n8k4` 指令，造成计算资源浪费。

---

## **4. 关键结论和发现**

### **主要发现**
1. **FP64 Tensor Cores 可有效加速小规模 GEMM**：
   - 即使是不规则、小尺寸的矩阵乘法（如 `25×5×4`），也能通过精心设计的索引映射和 bank conflict 规避策略获得显著收益。
   
2. **Kernel Fusion 是释放 Tensor Core 潜力的关键**：
   - 单独使用 DMMA 提速有限；与 loop fusion 结合后才能最大化减少内存访问，充分发挥计算能力。

3. **卓越的极端规模可扩展性**：
   - 在 **9,216 个 GH200 GPU** 上运行：
     - **弱扩展效率 ≈100%**
     - **强扩展效率达 86–91%**
   - 支持 **9.28 万亿 DOF** 的超大规模模拟。

4. **显著提升能效**：
   - 最高 **83% 能效提升**，对绿色超算具有重要意义。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **矩阵形状适配性差** | 当前 DMMA 指令（如 `m8n8k4`）无法完美匹配所有有限元算子的张量收缩模式，存在计算资源浪费 |
| **依赖特定硬件** | 仅适用于支持 FP64 DMMA 的 NVIDIA GPU（Ampere 架构及以上） |
| **编程复杂度高** | 需手动编写 PTX 指令、管理共享内存布局，开发门槛较高 |

---

### **未来工作方向**
1. **更灵活的张量核心调度机制**：
   - 开发编译器或自动调优框架，自动生成最优的 `fm/fn/fk` 映射以适应不同矩阵形状。
2. **扩展至其他 HPC 应用**：
   - 将该方法推广至其他依赖小批量 GEMM 的 PDE 求解器（如 DG、HDG、谱元法）。
3. **混合精度策略探索**：
   - 在保证稳定性的前提下，研究部分算子使用 FP16/BF16 + FP64 residual correction 的混合方案。
4. **支持新一代 Tensor Core 架构**：
   - 利用 GB200 中更强的 FP64 Tensor Core 和更低精度模式（FP4/FP6），进一步提升性能密度。

---

> ✅ **总结一句话**：  
> 本文开创性地将 **FP64 Tensor Cores** 成功应用于真实世界的高阶有限元模拟，在 **MFEM** 框架下实现了 **最高 2× 性能提升** 和 **83% 能效增益**，并在近万 GPU 上验证了其卓越的可扩展性，为下一代极端尺度科学计算提供了新的加速范式。

</details>

---

### 10. [PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies](https://arxiv.org/abs/2603.09216)

**Authors**: Sunjung Lee, Sanghoon Cha, Hyeonsu Kim, Seungwoo Seo, Yuhwan Ro, Sukhan Lee, Byeongho Kim, Yongjun Park, Kyomin Sohn, Seungwon Lee, Jaehoon Yu  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09216v1  

#### Abstract
On-device deployments of large language models (LLMs) are rapidly proliferating across mobile and edge platforms. LLM inference comprises a compute-intensive prefill phase and a memory bandwidth-intensive decode phase, and the decode phase has been widely recognized as well-suited to processing-in-m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在基于 **Processing-in-Memory (PIM)** 的 **on-device LLM inference** 中，存在两个关键系统级挑战：
- **Memory Attribute Inconsistency**：  
  - **Prefill 阶段**（计算密集型）需要将权重放在 **cacheable 区域** 以利用缓存重用；
  - **Decode 阶段**（内存带宽密集型）则要求权重位于 **non-cacheable 区域** 以确保 DRAM 请求能触发 PIM 执行。
- **Weight Layout Inconsistency**：  
  - Host 侧偏好 **host-friendly layout**（如列主序），利于通道和 bank 交错访问；
  - PIM 侧需要 **PIM-aware layout**（矩阵行连续存储于单个 DRAM bank 内），以最大化 in-bank SIMD 利用率。

这两个不一致性导致传统方案（如权重复制）带来显著的 **DRAM 容量开销**，限制了大模型在移动设备上的部署。

---

### 🚀 提出的新方法
作者提出 **PIM-SHERPA** —— 一种纯软件解决方案，无需硬件修改，通过动态管理内存属性和布局来解决上述矛盾。其核心思想是：  
> 在 **non-cacheable 区域** 存储唯一的 PIM-aware 权重副本，并在运行时将其 **动态拷贝到小块 cacheable buffer** 中用于 prefill 阶段的 GEMM 运算。

具体包含两种实现策略：

#### （1）**DRAM Double Buffering (DDB)**
- 使用两个 cacheable buffer 轮流进行 **prefetch** 和 **compute**。
- 当前层执行 GEMM 时，异步预取下一层权重并完成 **swizzled memory copy**（从 PIM-aware → host-friendly layout）。
- 实现 **计算与数据搬移的重叠**，隐藏延迟。

#### （2）**Online Weight Rearrangement with Swizzled Memory Copy (OWR)**
- 不使用双缓冲，而是在每层 GEMM 前 **即时执行 swizzled memory copy**。
- 更简单，无需复杂同步机制，适合输入序列较长、GEMM 时间占主导的场景。

此外，引入 **Swizzled Memory Copy (SMC)** 技术，在拷贝过程中完成 layout 转换，使标准 GEMM 库可直接使用。

---

### 🔍 相比现有方法的优势
| 方法 | 是否需硬件修改 | 是否复制权重 | DRAM 开销 | 实现复杂度 |
|------|----------------|---------------|------------|-------------|
| HBM-PIM / PAISE | 否 | 是（全量复制） | 高（~2×） | 低 |
| FACIL | 是（需增强 MC） | 否 | 低 | 高 |
| **PIM-SHERPA (DDB/OWR)** | **否** | **否** | **极低（仅需 ~1–2 层权重大小 buffer）** | **中等（DDB）/ 低（OWR）** |

✅ **优势总结**：
- **零硬件依赖**：完全软件实现，适用于现有产品级 PIM 系统（如 LPDDR-PIM）；
- **大幅降低 DRAM 占用**：相比 weight duplication 节省约 **47.8–49.7%**；
- **保持高性能**：TTFT 接近理论最优（接近 FACIL-O），端到端推理速度提升达 **3.3×**；
- **首次识别并解决 memory attribute inconsistency**。

---

## 2. 核心实验方法和设置

### 🧪 实验平台
- **设备**：Samsung Galaxy S24+
- **SoC**：Exynos 2400（1× Cortex-X4 + 多个 A720/A520 核）
- **内存**：LPDDR5X-8533（4通道，68.264 GB/s）
- **PIM 模拟器**：基于 PIMLibrary 改造，模拟 LPDDR5X-PIM 行为
- **验证准确性**：通过对比真实 HBM-PIM 硬件与仿真结果，误差控制在 **0.1%–3.6%**

### 📦 模型与数据
- **模型**：Llama 3.2 1B 和 3B（官方支持 on-device 版本）
- **精度格式**：BF16
- **输入序列长度 (SL)**：64–192 tokens
- **批处理大小**：1（single-batch 场景）

### 🎯 评估指标
- **Required DRAM Capacity**：总内存占用
- **Time to First Token (TTFT)**：衡量 prefill 性能
- **End-to-end Inference Speedup**：相对于 non-PIM baseline 的加速比
- **Functionality Validation**：输出一致性检查（vs. 原始 ExecuTorch）

### ⚖️ 基线方法对比
| 方法缩写 | 描述 |
|--------|------|
| **WD** | Weight Duplication：同时保留 cacheable host-friendly 和 non-cacheable PIM-aware 权重 |
| **FACIL-O** | FACIL under Oracle Assumption：假设 FACIL 已解决 memory attribute 问题的理想情况 |
| **S-DDB** | PIM-SHERPA with DRAM Double Buffering |
| **S-OWR** | PIM-SHERPA with Online Weight Rearrangement |

---

## 3. 主要实验结果和性能指标

### 📉 DRAM 容量节省
- **Llama 3.2 1B (BF16)**：
  - WD 需要 **~4.8 GB**
  - S-DDB/S-OWR 仅需 **~2.5 GB**
  - **节省 47.8%–48.5%**
- **Llama 3.2 3B (BF16)**：
  - WD 需要 **~12.8 GB**
  - S-DDB/S-OWR 仅需 **~6.4 GB**
  - **节省 49.4%–49.7%**

> 💡 缓冲区仅需约 **32MB**（对应 FF 层大小），远小于全模型复制。

![Figure 9](#) 显示 S-DDB/S-OWR 接近 FACIL-O 的容量效率。

---

### ⏱️ TTFT 性能表现
- **输入序列 ≥128 时，S-DDB 的 TTFT 接近 FACIL-O**
  - 在 Llama 3.2 3B 上，SL=192 时达到 **16.7 TPS**
  - 由于 DDB 成功隐藏 SMC 延迟，性能几乎持平
- **S-OWR 略慢**：
  - 因为 SMC 与 GEMM 串行执行
  - 测得 SMC 延迟约为 **0.6s (1B)** 和 **1.4s (3B)**
  - 但在长序列下仍可接受，且实现更简单

> ❗ 注意：实际 SMC 延迟高于理论估计，因非缓存区域拷贝速度仅为峰值带宽的 ~25%。

---

### 🚀 端到端推理加速（Speedup）
- 使用 LPDDR5X-PIM 后：
  - **S-DDB 与 FACIL-O 加速效果基本一致**
  - 输入 SL ≥128 时，speedup 可达 **3.3× vs non-PIM baseline**
  - 即使输出序列较短，也能维持高吞吐

![Figure 12](#) 显示 S-DDB/FACIL-O 曲线高度重合，表明其有效性。

---

### 🔬 消融分析（Ablation Study）
- **S-DDB 执行时序分析（Figure 11）**：
  - 在 SL=64 时，copy thread 成为关键路径（尤其 K/V 层计算时间短）
  - 在 SL=128 时，大部分层的 SMC 延迟被成功隐藏
  - FF2 层可通过预取下一层权重进一步优化 pipeline
- **调度影响明显**：
  - 线程竞争导致某些 layer（如 FF1）出现排队延迟
  - 表明未来可通过更精细的任务调度进一步优化

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Memory Attribute Inconsistency 是 PIM-LMM 部署中的根本瓶颈**，此前未被充分认识；
2. **纯软件方案可以有效协调 prefill 与 decode 对内存的不同需求**；
3. **在线 weight rearrangement 在长输入序列下是可行的**，其固定开销可被 GEMM 时间摊销；
4. **PIM-SHERPA 在不修改硬件的前提下，实现了接近 FACIL 的性能和优于 WD 的内存效率**；
5. 即使在 DRAM 极限受限场景（只能存一份权重），PIM-SHERPA 依然优于纯 host-only 方案。

---

### ⚠️ 方法的局限性
- **SMC 拷贝效率受限于内存带宽利用率**：实测仅达峰值的 ~25%，仍有优化空间；
- **依赖足够大的输入序列长度**：短序列下（<64）SMC 开销占比过高；
- **多核调度不够理想**：线程争抢导致部分 layer 出现 pipeline bubble；
- **目前基于 CPU 实现**：GPU/NPU 上需适配软件流水线机制。

---

### 🔮 未来工作方向
- 将 PIM-SHERPA 扩展至 **GPU/NPU 平台**，利用其 L2 cache 或 shared memory 进一步优化；
- 结合 **compiler-level 优化**，自动插入 SMC 和 buffer 管理逻辑；
- 探索 **hybrid 方案**：结合 FACIL 的灵活地址映射 + PIM-SHERPA 的内存管理；
- 在真实 PIM 硬件上部署验证，而非依赖模拟；
- 支持 **batched inference** 和 **multi-user 场景** 下的资源调度。

---

## 📝 总结
**PIM-SHERPA 是首个识别并解决 PIM-LMM 中 memory attribute inconsistency 的纯软件方案**。它通过 **DRAM Double Buffering** 和 **Online Weight Rearrangement with Swizzled Memory Copy**，在无需硬件改动的情况下，实现了：
- **近 50% 的 DRAM 容量节省**
- **接近理论最优的 TTFT 与端到端性能**

该工作为 **PIM 技术在移动端大规模落地提供了实用且高效的软件栈支持**，具有重要的工程价值和推广前景。

</details>

---

### 11. [Flash-KMeans: Fast and Memory-Efficient Exact K-Means](https://arxiv.org/abs/2603.09229)

**Authors**: Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Xiaoze Fan, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Kurt Keutzer, Song Han, Chenfeng Xu, Ion Stoica  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09229v1  

#### Abstract
$k$-means has historically been positioned primarily as an offline processing primitive, typically used for dataset organization or embedding preprocessing rather than as a first-class component in online systems. In this work, we revisit this classical algorithm under the lens of modern AI system d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Flash-KMeans: Fast and Memory-Efficient Exact K-Means**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统的 **k-means** 实现（尤其是在 GPU 上）在现代 AI 工作负载中面临严重的性能瓶颈，主要体现在以下三个方面：

- **IO 瓶颈**：在 assignment 阶段，标准实现会显式地将 $ N \times K $ 的距离矩阵 $ D $ 材料化（materialize）到 High Bandwidth Memory (HBM)，造成巨大的内存读写开销。
- **原子写竞争（Atomic Write Contention）**：在 centroid update 阶段，多个线程并发对同一聚类中心进行 `atomic_add` 操作，导致严重的硬件级序列化和带宽浪费。
- **系统级限制**：面对大规模数据（超出 VRAM）、动态形状（dynamic shapes）等现实部署场景时，传统方法存在 PCIe 通信开销大、编译配置调优耗时高等问题。

这些问题使得即使算法层面优化了 FLOPs，也无法转化为实际的端到端加速。

---

### **提出了什么新方法或新思路**
作者提出 **flash-kmeans**，一种面向现代 GPU 架构的高效、精确的 k-means 实现，不改变数学定义，而是通过 **kernel-level 和 system-level 的协同设计** 来突破底层硬件瓶颈。其核心创新包括：

#### **(1) FlashAssign：消除中间距离矩阵材料化**
- 将 distance computation 与 `argmin` 融合为一个流式处理过程。
- 在片上寄存器中维护每个点的当前最小距离和对应聚类索引，逐块扫描 centroids 并在线更新。
- **完全避免构造 $ N \times K $ 的距离矩阵 $ D $**，从根本上消除 HBM 中的 IO 开销。

#### **(2) Sort-Inverse Update：解决原子写竞争**
- 引入 **inverse mapping**：先对 assignment 向量进行 `argsort`，得到排序后的 token 索引。
- 按照 cluster ID 排序后，相同 cluster 的 token 自然形成连续段（contiguous segments）。
- 每个 CTA 处理一段，将局部聚合结果保留在 on-chip memory（如 shared memory），仅在 segment 边界处执行一次全局 `atomic_add`。
- 将高并发的 per-token scatter 写转换为低竞争的 segment-level reduce。

#### **(3) 算法-系统协同优化**
- **Chunked Stream Overlap**：支持异步分块加载数据，重叠 PCIe 传输与计算，实现超大规模 out-of-core 执行。
- **Cache-Aware Compile Heuristic**：基于 L1/L2 缓存大小和问题规模直接推导最优 kernel 配置，避免昂贵的 auto-tuning。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **性能** | 端到端最高达 **17.9×** 加速，assignment kernel 最高 **21.2×**，update kernel 最高 **6.3×** |
| **内存效率** | 完全避免 $ N \times K $ 矩阵存储，支持高达 **10亿点** 的 out-of-core 运行 |
| **部署友好性** | 编译调优开销降低 **175×**，且性能损失 < 0.3%，适合动态 shape 场景 |
| **数学正确性** | 不引入任何近似，输出结果与标准 k-means 完全一致 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
论文未使用特定公开数据集，而是采用 **合成数据** 进行全面 benchmark，覆盖多种典型 workload 配置，以验证方法的通用性和可扩展性。

参数范围如下：
- 数据点数量 $ N $：从 16K 到 1B
- 聚类数 $ K $：从 1K 到 64K
- 特征维度 $ d $：128, 512
- 批次大小 $ B $：1 ~ 32

---

### **实验设置和评估指标**

#### **硬件平台**
- **GPU**: NVIDIA H200
- **CUDA 版本**: 12.8
- **编程方式**: 自定义 CUDA kernel + Triton（部分基线）

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **End-to-End Latency** | 单次 Lloyd iteration 的总耗时 |
| **Kernel Latency** | 分别测量 assignment 和 centroid update 阶段的延迟 |
| **Throughput** | 每秒处理的数据点数 |
| **Memory Footprint** | 峰值 GPU 显存占用 |
| **Time-to-First-Run** | 包括编译和配置搜索的时间，用于衡量动态部署成本 |

---

### **基线方法对比**
对比了四类主流优化实现：
1. **fast_pytorch_kmeans**：基于 PyTorch 的快速实现
2. **fastkmeans** (Clavié and Warner, 2025)：Triton 实现，支持部分融合
3. **cuML** (NVIDIA)：工业级机器学习库中的 GPU k-means
4. **FAISS** (Facebook AI)：广泛使用的相似性搜索库，含 k-means 功能

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 场景 | flash-kmeans 性能提升 |
|------|------------------------|
| **Large N, Large K** ($N=1M, K=64K$) | > **5.4×** vs best baseline（PyTorch OOM） |
| **Large N, Small K** ($N=8M, K=1024$) | **17.9×** 端到端加速 |
| **Small N, Small K** ($N=16K, K=1K, B=32$) | 最高 **15.3×** 加速（batched 场景） |
| **Kernel Level - Assignment** | 最高 **21.2×** 加速（$N=1M, K=8192$） |
| **Kernel Level - Update** | 最高 **6.3×** 加速（$N=33M, K=4096$） |

---

### **与基线方法的对比结果**

| 对比对象 | 加速比 |
|---------|--------|
| **vs cuML** | 最高 **33×** 更快 |
| **vs FAISS** | 超过 **200×** 更快 |
| **vs fastkmeans (out-of-core)** | 在 1B 点任务上 **6.3×~10.5×** 更快 |

> ⚠️ 注意：PyTorch 在 large K 场景下因显存不足直接崩溃（OOM），而 flash-kmeans 可稳定运行。

---

### **消融实验结果**

#### **(1) Chunked Stream Overlap 效果**
- 在 $ N = 1B, K = 32768, d = 128 $ 场景下：
  - flash-kmeans 耗时 **41.4 秒**
  - fastkmeans 耗时 **261.8 秒**
  - **加速 6.3×**
- 在 $ N = 400M, K = 16384 $ 下达到 **10.5×** 端到端加速
- 成功将峰值显存控制在合理范围内，实现真正的大规模 out-of-core 支持

#### **(2) Cache-Aware Compile Heuristic 效果**
- **配置搜索时间对比**：
  - Exhaustive Auto-Tuning：最大超过 **325 秒**
  - flash-kmeans Heuristic：全部在 **< 2.5 秒** 内完成
  - **最高减少 175× 时间开销**
- **运行时性能对比**：
  - Heuristic 配置下的 kernel latency 与最优配置相差 **< 0.3%**
  - 实现“零代价”快速部署

---

## **4. 关键结论和发现**

### **主要发现**
1. **k-means 的性能瓶颈不在算法复杂度，而在实现层的硬件适配性**  
   即使 FLOPs 很低，若忽视 HBM IO 和 atomic contention，仍无法获得实际加速。

2. **FlashAssign 和 Sort-Inverse Update 是突破两大 kernel 瓶颈的关键**  
   - 前者将 assignment 从 IO-bound 转为 compute-bound
   - 后者将 update 阶段的有效带宽从 50 GB/s 提升至接近理论上限

3. **算法-系统协同设计是现代 AI Primitive 的必由之路**  
   仅优化 kernel 不够，必须结合 streaming、缓存感知编译等系统技术才能实现“即插即用”的高性能。

4. **flash-kmeans 支持极端规模 + 动态部署双重挑战**  
   - 可处理 **十亿级数据点**
   - 支持无需预热的动态 shape 快速启动

---

### **方法的局限性**
- 当前实现依赖定制 CUDA/Triton kernel，集成到主流框架（如 PyTorch）需额外工程工作。
- Sort-Inverse Update 引入了一次 `argsort` 操作，在极小规模任务中可能带来轻微额外开销（但总体仍优于 baseline）。
- 目前主要针对 Euclidean distance，其他距离度量的支持有待扩展。

---

### **未来工作方向**
1. **扩展至其他 clustering 算法**（如 k-medoids, DBSCAN）
2. **支持更多 distance metrics**（cosine, Mahalanobis 等）
3. **与 LLM serving pipeline 深度集成**，作为 sparse routing / KV-cache quantization 的原生组件
4. **多 GPU / 分布式扩展**，支持跨节点同步 k-means
5. **自动 fallback 机制**：根据 $ N, K, d $ 自适应选择是否启用 flash-kmeans

---

> ✅ **总结一句话**：  
> **flash-kmeans 通过硬件感知的 kernel 融合与系统协同设计，在不牺牲精度的前提下，实现了 k-means 在现代 GPU 上的革命性加速，使其真正成为可用于生成式 AI 基础设施的在线 primitive。**

</details>

---

### 12. [Multi-DNN Inference of Sparse Models on Edge SoCs](https://arxiv.org/abs/2603.09642)

**Authors**: Jiawei Luo, Di Wu, Simon Dobson, Blesson Varghese  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09642v1  

#### Abstract
Modern edge applications increasingly require multi-DNN inference systems to execute tasks on heterogeneous processors, gaining performance from both concurrent execution and from matching each model to the most suited accelerator. However, existing systems support only a single model (or a few spar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Multi-DNN Inference of Sparse Models on Edge SoCs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代边缘计算应用（如增强现实 AR）需要在异构 SoC（System-on-Chip）上并行执行多个 DNN 任务（如语音识别、图像分类等）。每个任务通常有多个 **Service Level Objectives (SLOs)**，例如对延迟和精度的不同要求。

现有系统面临以下挑战：
- **模型变体选择有限**：大多数多 DNN 推理系统仅支持为每个任务选择一个或少数几个稀疏模型变体（sparse variants），导致难以满足多样化的 SLO 需求。
- **高 SLO 违规率**：由于可用变体空间小，系统无法找到同时满足精度和延迟约束的模型，导致 SLO violation rate 较高。
- **部署开销大**：模型 stitching 虽然能生成更多变体，但传统方法需重新训练，且存在巨大的 profiling 成本、次优的处理器放置顺序以及内存预加载开销。

### 提出了什么新方法或新思路
本文提出 **SparseLoom**，首个将 **model stitching** 技术整合到多 DNN 推理系统中的端到端框架。其核心创新如下：

#### ✅ 创新点 1：Model Stitching（无需重训练）
- **定义**：通过从同一基础模型的不同稀疏变体中组合子图（subgraphs）来创建新的模型变体，**无需重新训练或微调**。
- **操作范围**：基于已有的稀疏模型 zoo（pruned、quantized、dense），利用层对齐（layer-aligned）的子图进行重组。
- **优势**：显著扩展了可选模型变体的空间（可达数千个），提供更丰富的精度-延迟权衡选项。

#### ✅ 创新点 2：三大优化模块协同设计
为解决 model stitching 带来的三大挑战，SparseLoom 设计了三个关键模块：

| 挑战 | 模块 | 解法 |
|------|------|------|
| **1. Profiling 开销大**（指数级增长） | **Performance Profiler** | 使用 **accuracy estimator** 和 **latency estimator**（基于 XGBoost）预测性能，避免全量实测 |
| **2. 处理器放置次优** | **Sparsity-Aware Optimizer** | 联合优化 **variant selection** 与 **processor placement order**，提升吞吐量 |
| **3. 内存预加载开销高** | **Hot-Subgraph Preloader** | 基于“hotness”评分（频率 + 唯一性）优先预加载高频/关键子图 |

---

### 相比现有方法的优势
| 维度 | 现有方法 | SparseLoom |
|------|--------|-----------|
| 变体生成方式 | 固定稀疏变体（pruning/quantization） | **无训练 stitching**，变体数量呈指数增长 |
| 变体选择能力 | 单一或少量变体 | 支持上千个 stitched variants |
| 子图调度 | 固定放置顺序（如 NPU→GPU→CPU） | 动态优化 sparsity-aware placement |
| 内存管理 | 全部预加载所有变体 | 按 hotness 分数选择性预加载 |
| 整体目标 | 最大化利用率 | 同时最小化 SLO violation 并最大化 throughput |

---

## 2. 核心实验方法和设置

### 使用的数据集与模型
实验覆盖四类典型边缘 AI 任务，对应四个标准数据集和基础模型：

| Task Type | Dataset | Base Model |
|----------|--------|------------|
| Image Classification | ImageNet-1K | ResNet-101 |
| Sentiment Classification | SST-2 | BERT-Base |
| Human Activity Recognition | HAR | ViT-Small |
| Speech Recognition | LibriSpeech ASR | Wav2vec2 |

每个任务构建了一个包含 **10 个稀疏变体** 的 **sparse model zoo**，包括：
- 1 个 dense 模型（FP32）
- 多个 pruned 模型（结构化/非结构化，sparsity 从 20% 到 90%）
- 多个 quantized 模型（FP16 / INT8）

> 所有变体使用 NNCF（Intel）或 ONNX Runtime（NVIDIA）生成，存储于磁盘，总大小约 7.6GB。

---

### 实验平台（Edge SoCs）
在三种异构边缘设备上测试：

| Platform | CPU | GPU | NPU |
|---------|-----|-----|-----|
| **Desktop** (Intel Core Ultra 7 265K) | x86-64, 20-core | 4-Xe-core | Intel AI Boost |
| **Laptop** (Intel Core Ultra 5 135U) | x86-64, 12-core | 4-Xe-core | Intel AI Boost |
| **Jetson AGX Orin** | ARM Cortex, 12-core | 2048-core Ampere | N/A |

推理引擎：
- Intel 平台：**OpenVINO**
- NVIDIA 平台：**ONNX Runtime + TensorRT**

---

### 评估指标
| Metric | 定义 |
|-------|------|
| **SLO Violation Rate** | 未能满足精度或延迟要求的任务占比（平均于所有任务到达顺序） |
| **Throughput** | 单位时间内完成的推理请求数（queries/sec） |
| **Profiling Time** | 构建性能查找表所需时间 |
| **Memory Overhead** | 预加载模型所占内存比例 |

---

### 基线方法对比
将现有系统按两个维度分类，形成六种基线：

| 类别 | Variant Selection | Partitioning |
|------|-------------------|-------------|
| SV-AO-P/NP | 单一变体（精度最优） | 是否分图 |
| SV-LO-P/NP | 单一变体（延迟最优） | 是否分图 |
| AV-P/NP | 自适应选择多个变体 | 是否分图 |

> SparseLoom 属于 **AV-P** 类别，但通过 model stitching 实现超越。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总
| 指标 | 提升幅度 |
|------|---------|
| **SLO Violation Rate ↓** | 最多降低 **74%** |
| **Throughput ↑** | 最高提升 **2.31×** |
| **Memory Overhead ↓** | 平均减少 **28%** |
| **Profiling Cost ↓** | 最多减少 **99%**（相比 exhaustive profiling） |

---

### 与基线方法的对比结果

#### 📊 SLO Violation Rate
- 在所有平台上，SparseLoom 显著优于所有六类基线。
- 相比最差基线（SV-LO-NP），SLO violation rate 最多下降 **74%**。
- 即使相比最先进的自适应选择方法（AV-P/NP），仍能进一步降低 **24.7%**。

> 图 10 显示，在 Jetson Orin 上，SparseLoom 的违规率仅为 ~25%，而其他方法普遍高于 60%。

#### 📈 Inference Throughput
- 所有支持 subgraph partitioning 的方法（P 类）均优于 NP 类。
- SparseLoom 在笔记本上达到 **2.31×** 于 SV-AO-NP，在台式机上比最佳基线（SV-LO-P）快 **1.53×**。

> 图 11 表明，model stitching 提供了更多高效变体和更优调度路径。

#### ⏱️ Profiling 时间大幅缩短
- 不使用 estimator 时，profiling 时间随变体数指数增长（如 10 variants → 468 分钟）。
- 使用 estimator 后，时间降至 **5 分钟以内**，**最多减少 99%**。

> 图 12 显示，即使变体数增加至 10，SparseLoom 仍保持低开销。

#### 🧠 Sparsity-Aware Placement 提升吞吐
- 固定放置顺序（如 N-G-C）常导致次优性能。
- SparseLoom 自动选择最优顺序（如 G-C-N 或 G-N-C），使吞吐量最高提升 **2×**。

> 图 13 显示不同平台的最佳顺序不同，验证了动态优化必要性。

#### 💾 Hot-Subgraph Preloading 减少内存占用
- 在仅 **15% 内存预算**下，SparseLoom 已优于全量预加载的 AV 方法。
- 在 **55% 预算**下，SLO violation rate 与全量预加载相差不到 **2.7%**。
- 达到相同性能时，内存节省 **25%~40%**，平均 **28%**。

> 图 14 显示，少量“热点”子图即可覆盖多数 SLO 场景。

---

### 消融实验结果（Ablation Study）
虽然文中未明确命名“ablation”，但通过模块替换验证了各组件有效性：
- 移除 estimator → profiling 时间暴增
- 使用固定 placement order → throughput 下降近半
- 随机预加载子图 → SLO violation 显著上升

---

## 4. 关键结论和发现

### 主要发现
1. **Model Stitching 是有效的变体增强手段**  
   无需重训练即可生成上千个高质量变体，显著改善精度-延迟帕累托前沿（见图 4），甚至出现比原 zoo 更快或更准的新变体（4% 更高精度，5% 更低延迟）。

2. **联合优化至关重要**  
   单独优化 variant selection 或 placement order 效果有限；**joint optimization** 才能实现最大吞吐。

3. **hotness-based 预加载策略高效实用**  
   少量高频/独特子图即可满足大部分 SLO 需求，极大缓解内存压力。

4. **SparseLoom 泛化性强**  
   在不同硬件架构（x86 vs ARM）、不同模型类型（CNN、Transformer）上均表现稳定。

---

### 方法的局限性
1. **依赖统一内存架构（UMA）**  
   当前假设处理器共享内存空间，忽略跨设备通信开销。若用于分离内存系统（如 PCIe 连接 GPU），需额外考虑传输延迟。

2. **不适用于超大规模 Foundation Models**  
   如 LLMs 由于显存需求极高，难以在边缘部署多实例或多变体，不在本工作适用范围内。

3. **DVFS 影响未主动处理**  
   动态电压频率调节可能改变实际推理延迟，当前系统未实时感知并调整策略。

4. **子图划分粒度固定**  
   实验中子图数等于处理器数，未探索更细粒度划分的影响。

---

### 未来工作方向
1. **支持动态子图划分**  
   根据 workload 特征自动决定最优 subgraph 数量与边界。

2. **引入在线反馈机制应对 DVFS**  
   实时监控频率变化，并触发轻量级 re-profiling 或调度调整。

3. **扩展至云端协同场景**  
   将部分 stitching 变体卸载至边缘服务器，实现云-边协同推理。

4. **探索更多压缩技术组合**  
   如结合 knowledge distillation 或 LoRA adapter 进行 stitching。

---

## 总结
**SparseLoom** 是一项面向边缘多 DNN 推理系统的突破性工作，首次将 **model stitching** 引入生产级推理框架，解决了因变体不足导致的 SLO 违规问题。它不仅提出了无需训练的变体生成方法，还通过 **estimation-driven profiling**、**sparsity-aware placement** 和 **hotness-based preloading** 三大机制实现了高效的端到端部署。

实验表明，SparseLoom 在真实边缘 SoCs 上实现了：
- **SLO violation rate ↓74%**
- **throughput ↑2.31×**
- **memory overhead ↓28%**

为未来智能边缘设备上的多模态 AI 应用提供了强有力的系统支撑。

</details>

---

### 13. [A Multi-Prototype-Guided Federated Knowledge Distillation Approach in AI-RAN Enabled Multi-Access Edge Computing System](https://arxiv.org/abs/2603.09727)

**Authors**: Luyao Zou, Hayoung Oh, Chu Myaet Thwal, Apurba Adhikary, Seohyeon Hong, Zhu Han  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09727v1  

#### Abstract
With the development of wireless network, Multi-Access Edge Computing (MEC) and Artificial Intelligence (AI)-native Radio Access Network (RAN) have attracted significant attention. Particularly, the integration of AI-RAN and MEC is envisioned to transform network efficiency and responsiveness. There...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Multi-Prototype-Guided Federated Knowledge Distillation Approach in AI-RAN Enabled Multi-Access Edge Computing System

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **AI-RAN enabled MEC system** 中的 **非独立同分布（non-IID）数据** 所导致的联邦学习（FL）性能下降问题。传统 FL 在处理异构数据时，由于客户端之间数据分布差异大，模型聚合后容易产生误导性的全局模型，从而降低准确率。

此外，现有基于 **single prototype** 的方法（如 FedProto）通过平均嵌入向量生成类别原型，虽能缓解 non-IID 问题，但会因平均操作丢失局部特征中的有用信息。

---

### 提出的新方法与创新思路
作者提出了一种 **多原型引导的联邦知识蒸馏方法（MP-FedKD）**，其核心创新包括：

- **Multi-Prototype Strategy**：  
  不再为每个类别仅生成一个 prototype，而是采用 **conditional hierarchical agglomerative clustering (CHAC)** 为每个类生成多个 prototypes，以更全面地捕捉样本的特征多样性。

- **Self-Knowledge Distillation (SKD)**：  
  将前一轮的本地模型作为“教师模型”，指导当前轮次的“学生模型”训练，避免了额外预训练教师网络的需求，提升了对 non-IID 数据的适应能力。

- **Prototype Alignment (PA) 机制**：  
  设计了一种新的对齐机制，使当前轮的全局 prototype 能够从上一轮的本地 embedding 中学习，减少因平均聚合造成的语义信息损失。

- **LEMGP Loss 函数**：  
  提出一种新型损失函数，结合 **COREL loss** 思想，包含吸引项（attractive loss）和排斥项（repulsive loss），促使本地 embedding 向同类别全局 prototype 靠拢，同时远离其他类别的 prototype。

---

### 相比现有方法的优势
| 方面 | 优势说明 |
|------|--------|
| **信息保留能力更强** | 多原型策略相比 single prototype 更能保留细粒度特征信息 |
| **无需外部教师模型** | 使用 SKD 实现自蒸馏，降低部署复杂度 |
| **动态原型更新机制** | PA 机制让全局 prototype 持续吸收历史本地知识，增强泛化性 |
| **端到端可训练** | LEMGP loss 可直接融入优化流程，提升收敛稳定性 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共使用 **六个数据集** 进行评估，涵盖同域与跨域 non-IID 场景：

- **同域数据集（Same Domain）**：
  - CIFAR-10
  - MNIST
  - Fashion-MNIST
  - EuroSAT（遥感图像分类）
- **异构混合数据集（Distinct Domain）**：
  - **M+F**：MNIST + Fashion-MNIST 组合
  - **C+E**：CIFAR-10 + EuroSAT 组合

> 注：M+F 和 C+E 用于模拟不同设备来源、不同模态数据的真实 MEC 场景。

---

### 实验设置
| 参数 | 设置 |
|------|------|
| **Non-IID 构造方式** | 使用 Dirichlet 分布（Dir = {0.3, 0.5, 0.7, 0.9}）控制标签划分不均衡程度 |
| **模型架构** | S-CNN、ResNet-8、ResNet-10（最终选用 ResNet-10） |
| **客户端数量** | 10、20、50 |
| **本地训练参数** | Batch size=32, Learning rate=0.001, Local epochs=5, Communication rounds=50 |
| **温度参数 T** | 0.1 |
| **超参数** | μ₁=0.9, μ₂=1, μ₃=0.1, λ=0.5 |

---

### 评估指标
- **Accuracy (ACC)**
- **Average Accuracy (AA)**
- **Root Mean Square Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Macro F1 Score (MFS)**

---

### 基线方法对比
| 方法 | 类型 | 是否包含 SKD | 是否多原型 |
|------|------|---------------|-------------|
| **FedProx** | 正则化 FL | ❌ | ❌ |
| **FedProto** | 单原型 FL | ❌ | ❌ |
| **FedAS** | 个性化 FL | ❌ | ❌ |
| **MOON** | 对比学习 FL | ❌ | ❌ |
| **E-FPKD** | 知识蒸馏 FL | ✅（双模型） | ❌ |
| **FedALA** | 自适应聚合 | ❌ | ❌ |
| **K-Means 替代版 MP-FedKD** | 消融对照 | ✅ | ✅（聚类方式不同） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（#Clients=10, Dir=0.9）

| 数据集 | 方法 | Accuracy | RMSE / MAE |
|-------|------|----------|------------|
| **EuroSAT** | Ours | **0.8390** | RMSE=1.7148, MAE=0.3415 |
| | FedProto | 0.7553 | — |
| | MOON | 0.8192 | — |
| **MNIST** | Ours | **0.9933** | MAE=0.1114 |
| | FedProto | 0.9527 | — |
| **Fashion-MNIST** | Ours | **0.9097** | RMSE=1.1277 |
| | FedProto | 0.7826 | — |
| **CIFAR-10** | Ours | 0.6710 | — |
| | FedProto | 0.4458 | — |

> ✅ **最高精度提升达 28.70%（vs FedProto on EuroSAT）**

---

### 与基线方法的对比结果
- 在所有数据集上，**MP-FedKD 均取得最高的 Accuracy 和 AA，最低的 RMSE 与 MAE**。
- 在 **EuroSAT 上 RMSE 比 FedProx 低约 1.62×，MAE 低 2.54×**。
- 在 **M+F 和 C+E 异构数据集上表现尤为突出**，验证了方法在跨域场景下的鲁棒性。
- 当客户端数为 10 时，性能最优，但仍具备良好扩展性（scalability）。

---

### 消融实验结果（Ablation Study）

#### （1）不同组件的影响（Table VIII）
| 模型变体 | CIFAR-10 AA | MNIST AA | Fashion-MNIST AA |
|---------|-------------|----------|------------------|
| w/o PA | 0.6637 | 0.9785 | 0.8890 |
| w/o LEMGP | 0.6551 | 0.9745 | 0.8916 |
| **Ours (完整)** | **0.6709** | **0.9927** | **0.9062** |

> 🔍 移除 PA 导致平均精度下降 0.72%，移除 LEMGP 下降 1.58%，表明两者均至关重要。

#### （2）不同聚类算法比较（CHAC vs K-Means）
- 在 M+F 数据集上，**CHAC-based 方法比 K-Means 高出约 3.02% 的最终精度**。
- K-Means 版本出现明显震荡，收敛不稳定；而 CHAC 更平滑且更快收敛。
- 原因推测：**HAC 利用树状图（dendrogram）提供层次结构信息，优于扁平聚类**。

#### （3）不同 cluster 数量（ζ）
- 当 ζ=3 时，在多数数据集上达到最佳性能。
- 若样本数不足，则退化为单样本一簇（ζ = |Dₘ,𝒸|），保证灵活性。

---

## 4. 关键结论和发现

### 主要发现
1. **多原型机制显著优于单原型方法**，尤其在 high non-IID（Dir=0.3）下仍保持稳定性能。
2. **CHAC 聚类优于 K-Means**，因其利用层级结构保留更多语义关系。
3. **SKD + PA + LEMGP 的组合有效缓解了 non-IID 带来的偏差问题**，实现更一致的知识迁移。
4. 所提方法在 **homogeneous 与 heterogeneous 数据集上均表现出色**，适用于真实 MEC 场景。
5. **具有良好的鲁棒性和可扩展性**，即使在客户端数量变化时也能维持高性能。

---

### 方法的局限性
- **计算开销较高**：CHAC 时间复杂度为 O(|Dₘ,𝒸|³)，在大规模数据或高维 embedding 下可能影响效率。
- **依赖本地 embedding 质量**：若表示层训练不佳，prototype 质量将受限。
- **未考虑通信压缩机制**：未集成模型剪枝或量化，实际部署中可能面临带宽瓶颈。
- **假设客户端连接唯一 DU**，简化了网络拓扑，现实系统可能更复杂。

---

### 未来工作方向
1. **轻量化 CHAC 实现**：探索近似 HAC 或增量聚类以降低时间成本。
2. **引入动态 prototype 数量机制**：根据类别样本密度自动调整 ζ。
3. **结合差分隐私（DP）或安全聚合（SecAgg）**：增强隐私保护能力。
4. **拓展至 personalized FL 场景**：支持客户端个性化输出头（head）。
5. **在真实 AI-RAN testbed 上进行部署验证**：推动工业级落地应用。

--- 

> ✅ **总体评价**：本文提出的 **MP-FedKD** 是一种面向 AI-RAN enabled MEC 系统的高效联邦学习框架，通过 **multi-prototype + SKD + PA + LEMGP** 四重机制，显著提升了在 non-IID 场景下的模型性能与鲁棒性，是联邦学习与边缘智能融合的重要进展。

</details>

---

### 14. [Social-R1: Towards Human-like Social Reasoning in LLMs](https://arxiv.org/abs/2603.09249)

**Authors**: Jincenzi Wu, Yuxuan Lei, Jianxun Lian, Yitian Huang, Lexin Zhou, Haotian Li, Xing Xie, Helen Meng  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.09249v1  

#### Abstract
While large language models demonstrate remarkable capabilities across numerous domains, social intelligence - the capacity to perceive social cues, infer mental states, and generate appropriate responses - remains a critical challenge, particularly for enabling effective human-AI collaboration and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Social-R1: Towards Human-like Social Reasoning in LLMs**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前的 **Large Language Models (LLMs)** 在形式化推理任务（如数学、编程）上表现优异，但在 **social intelligence**（社会智能）方面仍存在显著瓶颈。具体表现为：
- **Reasoning Parasitism（推理寄生）**：模型倾向于先猜测答案，再反向构造理由（Answer-driven Backfilling），而非基于叙事进行真实的社会推理。
- **Shortcut Learning（捷径学习）**：依赖表面线索（如选项词频匹配）而非深层心理状态推断。
- **Interpretation Bottleneck（解释瓶颈）**：虽然能识别表面社会信号，但难以将其映射到潜在的心理状态（mental states），导致“逻辑倒置”现象（最终答案正确，但推理过程错误）。

### **提出的新方法与思路**
为解决上述问题，作者提出了两个核心组件：

#### **(1) ToMBench-Hard**
- 一个**对抗性基准测试集**，专门设计用于暴露 LLMs 在社会推理中的捷径行为。
- 基于 **ATOMS 框架**（Abilities in the Theory-of-Mind Space），覆盖六大维度的社会智能：Belief, Desire, Emotion, Intention, Knowledge, Non-literal Communication。
- 引入 **ToM-consistent adversarial perturbations**（如不对称信息、未观察到的状态变化），迫使模型必须进行结构化推理，无法通过统计捷径蒙混过关。

#### **(2) Social-R1 框架**
一种新的 **Reinforcement Learning (RL)** 框架，强调对整个推理轨迹（reasoning trajectory）的监督，而不仅仅是最终答案。其核心是多维奖励机制：
- **Rstruct (SIP Structural Alignment)**：强制模型遵循人类社会信息处理（SIP）的四个阶段：Cue Encoding → Cue Interpretation → Goal Clarification → Response Generation。
- **Rcontent (SIP Content Integrity)**：确保每一步推理都基于故事内部证据，避免错误归因或目标误判。
- **Rlen (Inference Efficiency Optimization)**：通过重复惩罚和长度窗口控制，鼓励简洁高效的推理。
- **Rfmt (Verifiable Format Alignment)**：要求输出格式标准化（如 `<thinking>` 和 `<answer>` 标签），便于解析推理路径。

此外，采用 **curriculum learning** 策略：初期以结果奖励为主，后期逐步增强过程奖励权重，实现稳定训练。

### **相比现有方法的优势**
| 维度 | 传统方法 | Social-R1 |
|------|--------|---------|
| 监督粒度 | Outcome-based（仅看答案） | **Trajectory-level**（全过程监督） |
| 推理质量 | 易出现 Reasoning Parasitism | 抑制寄生，促进独立推理 |
| 泛化能力 | 在简单基准上表现好，在对抗性任务上崩溃 | 在 ToMBench-Hard 上显著优于大模型 |
| 参数效率 | 依赖大规模参数提升性能 | 小模型即可超越大模型 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ToMBench-Hard (新提出)**：800个专家标注的对抗性多选题，分为700训练 + 100测试。
- **ToM-RL (公开基准)**：用于对比，展示“捷径幻觉”（shortcut illusion）。
- **其他 out-of-domain 社会推理基准**（共8个）：
  - **In-domain**: ToMBench, ToMBench-Hard
  - **Out-of-domain**: SocialIQA, EmoBench, MotiveBench, SimpleToM, Hi-ToM, TactfulToM

### **实验设置**
- **模型架构**：
  - 基于 **Qwen3-4B** 和 **Qwen3-8B** 进行微调。
  - 对比模型包括：DeepSeek-R1, GPT-4o, GPT-5, LLaMa3.1-70B, Qwen3-32B 等闭源与开源模型。
- **训练细节**：
  - 使用 **VERL (Value Estimation RL)** 框架。
  - 训练步数：600 步。
  - 硬件：8 × NVIDIA A100 (80GB) GPU。
  - 使用 **Group Relative Policy Optimization** 更新策略。
- **评估指标**：
  - **准确率 (Accuracy)**：在各基准上的总体与分项得分。
  - **推理长度 (Response Length)**：衡量效率。
  - **消融分析 (Ablation Study)**：验证各奖励模块的作用。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | ToMBench-Hard (All) | Overall (8 benchmarks avg.) |
|------|---------------------|----------------------------|
| Human Experts | 87% | — |
| DeepSeek-R1 | 61% | 70.73% |
| GPT-5 | 56% | 69.56% |
| Qwen3-32B | 52% | 66.24% |
| **SocialR1-4B (Full)** | **68.80%** | **68.80%** |
| **SocialR1-8B (Full)** | **72.70%** | **72.70%** |

> ✅ **关键发现**：尽管 **SocialR1-4B 只有 4B 参数**，却超过了 **LLaMa3.1-70B (70B)** 在所有基准上的表现。

### **与基线方法的对比**
- **在 ToMBench-Hard 上**：
  - 前沿模型（如 O3, DeepSeek-R1）在 ToM-RL 上可达 87–88%，但在 ToMBench-Hard 上骤降至 <61%，揭示了“**捷径幻觉**”。
  - Social-R1-8B 达到 **72.7%**，远超同类规模模型。
- **跨域泛化能力**：
  - 在 **Hi-ToM**（高阶心智理论）上，SocialR1-8B 达到 **96.75%**，接近人类水平。
  - 在 **TactfulToM**（理解善意谎言）上也显著领先。

### **消融实验结果**
从 **Table 2** 中可看出各奖励组件的重要性：

| 模型变体 | Overall 性能 | 关键影响 |
|--------|------------|--------|
| `only Rout`（仅结果奖励） | ↓ 显著下降 | 验证过程奖励必要性 |
| `w/o Rlen`（无长度控制） | ↓ 下降，尤其 Hi-ToM | 推理变长 + 冗余增加 |
| `w/o Rstruct`（无结构约束） | ↓ 下降 | 出现 stage skipping 和 option parasitism |
| `w/o Rcontent`（无内容完整性） | ↓ 下降 | 错误心理归因增多 |

> 🔍 **深入分析表明**：
> - **SocialR1-8B** 在推理过程中几乎不提前提及选项（option-agnostic），而基线模型（如 DeepSeek-R1）在早期就频繁引用选项，显示其依赖“答案反推”。
> - 在扰动鲁棒性测试中，SocialR1-8B 能保持高效推理，而基线模型需更长文本才能维持精度。

---

## **4. 关键结论和发现**

### **主要发现**
1. **轨迹级对齐（trajectory-level alignment）比单纯扩大模型规模更有效**：
   - 一个 **4B 模型** 经过 Social-R1 训练后，性能超过 **70B 甚至更大的模型**。
   - 表明“**推理质量 > 参数数量**”对于社会智能至关重要。

2. **挑战性训练数据 + 过程监督 = 真实社会推理能力**：
   - ToMBench-Hard 成功暴露了主流模型的“纸老虎”本质。
   - Social-R1 通过多维奖励引导模型走向人类般的认知路径。

3. **抑制 Reasoning Parasitism 是关键突破**：
   - 多维奖励系统有效防止了“先猜答案再编理由”的行为，使推理真正扎根于叙事。

### **方法的局限性**
- **依赖高质量奖励模型**：Rcontent 奖励依赖 GPT-4o 或 GPT-5 作为裁判，可能存在主观偏差。
- **计算成本较高**：RL 训练需要大量采样和反馈，不适合低资源场景。
- **泛化边界待探索**：目前主要在英语多选题上验证，是否适用于开放生成或多轮对话尚不明确。

### **未来工作方向**
- 扩展至更复杂的社会任务，如 **human-AI collaboration**、**AI 角色模拟**。
- 探索 **zero-shot reward modeling**，减少对强裁判模型的依赖。
- 应用于 **社会科学仿真**，辅助心理学与行为经济学研究。
- 构建多语言版本的 ToMBench-Hard 和 Social-R1。

---

> 📌 **一句话总结**：  
> **Social-R1 证明，通过对抗性数据和轨迹级强化学习，可以让小模型具备媲美甚至超越大模型的人类级社会推理能力，为构建真正“懂人”的 AI 提供了一条高效且可靠的技术路径。**

</details>

---

### 15. [SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models](https://arxiv.org/abs/2603.09215)

**Authors**: Hsiao-Ying Huang, Cheng-Han Chiang, Hung-yi Lee  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.09215v1  

#### Abstract
Interleaved spoken language models (SLMs) alternately generate text and speech tokens, but decoding at full transformer depth for every step becomes costly, especially due to long speech sequences. We propose SPAR-K, a modality-aware early exit framework designed to accelerate interleaved SLM infere...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代 **Spoken Language Models (SLMs)**，尤其是 **interleaved SLMs**（交替生成文本和语音token），在推理时计算开销巨大。原因在于：
- 继承了大型语言模型（LLM）的深度结构；
- 需要解码长序列的离散语音token（如Codec tokens），显著增加延迟。

传统的 **early exit** 方法（广泛用于纯文本LLM）直接迁移到SLMs上效果不佳，因为：
- 文本token对语义连贯性敏感，中间层预测不可靠；
- 语音token具有更强的局部冗余性和感知鲁棒性，允许更灵活的浅层退出。

因此，需要一种**模态感知（modality-aware）** 的早期退出机制。

---

### 🚀 提出的新方法：SPAR-K
提出 **SPAR-K**（Scheduled Periodic Alternating Early Exit for Spoken Language Models），一种专为 interleaved SLM 设计的 **周期性交替 early exit 框架**，其核心思想是：

- **仅对 speech tokens 进行 early exit**，而保留 text tokens 使用 full-depth 解码以保证语义准确性；
- 引入 **K周期调度策略（K-periodic schedule）**：
  - 在每 $ K $ 个 speech token 中，$ K-1 $ 个位置从固定中间层 $ l_{EE} $ 提前退出；
  - 1个位置仍使用最后一层 $ L $ 全深度解码，作为“刷新”步骤（refresh step），缓解因连续 early exit 导致的分布偏移（distribution shift）；
- 支持多种调度模式：
  - Even schedule: $ \{L, l_{EE}, L, l_{EE}, ...\} $
  - Odd schedule: $ \{l_{EE}, L, l_{EE}, L, ...\} $
  - Triple schedule: $ \{L, l_{EE}, l_{EE}, L, ...\} $

此外还设计了：
- **Layer-specific LM head**：训练每个中间层的独立输出头，使其能准确预测最终token分布；
- **KV-cache 补全机制**：利用周期性的 full-depth 步骤，在并行中恢复缺失的 KV-cache，避免影响后续 attention。

---

### 🔍 相比现有方法的优势
| 方法 | 是否适用SLM | 计算开销 | 性能稳定性 | 感知质量保持 |
|------|-------------|----------|------------|----------------|
| No early exit (Full-depth) | ✔️ | 高 | 最佳 | ✔️ |
| Fixed-layer early exit | ❌ | 低 | 差（严重退化） | ❌ |
| Confidence-based early exit | ⚠️部分可行 | 中等（需额外计算熵） | 敏感于阈值 & 模型 | 不稳定 |
| **SPAR-K (Ours)** | ✅ | **零额外开销** | **稳定高效** | ✅ 几乎无损 |

优势总结：
- **无需动态判断 confidence**，节省每步的 entropy 计算开销；
- **不引入额外参数或训练成本**（仅训练 layer-specific head，推理时冻结）；
- **显著降低平均 decoding depth**，同时保持 ASR 准确率、MOS 和 WER；
- **首次探索 SLM 场景下的 early exit**，揭示 text 与 speech token 的本质差异。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
共四个英文任务数据集，涵盖不同类型：
| 数据集 | 类型 | 描述 |
|--------|------|------|
| **AlpacaEval** | 对话评估 | 使用 LLM-as-a-Judge 评分（0–100） |
| **Llama Questions** | 推理问答 | 事实类问题回答 |
| **TriviaQA** | 事实QA | 大规模远监督阅读理解数据集 |
| **WebQuestions** | 开放域QA | 基于 Freebase 的问答对 |

所有输入为语音，输出为语音（通过 interleaved SLM 生成 text + speech tokens 后合成音频）。

---

### ⚙️ 实验设置
#### 模型
- **Step-Audio-2-mini**（简称 Step-Audio-2）
  - 28层 Transformer
  - 文本:语音 = 1:4
  - 采样方式：nucleus sampling（temp=0.7, top_p=0.9）
- **GLM-4-Voice**
  - 40层 Transformer
  - 文本:语音 = 13:26
  - 采样方式：greedy decoding

#### 层级头训练
- 使用 **VoiceAssistant-400K** 数据集的一个子集（18K样本）
- 固定主干模型，训练各层 $ l \in \{1,\dots,L\} $ 的 **layer-specific LM head**
- 目标：让中间层输出逼近最后一层的 token 分布（KL loss / Cross-entropy）

---

### 📊 评估指标
| 指标 | 类型 | 说明 |
|------|------|------|
| **Accuracy (%)** | QA任务 | 使用 GPT-4o-mini 判断答案是否正确 |
| **LLM-as-a-Judge Score (0–100)** | AlpacaEval | 自动化对话质量打分 |
| **MOS ↑** | 感知质量 | 使用 **UTMOS-v2** 自动评估语音自然度（Mean Opinion Score） |
| **ASR-WER ↓** | 对齐质量 | 使用 Whisper-large-v3 转录合成语音，与原始 text tokens 计算词错误率 |
| **Average Exit Layer** | 效率 | 平均退出层数，越低表示计算量越少 |
| **Speedup (%)** | 效率 | 相对于 full-depth 的层数减少比例 |

---

### 🔁 基线方法对比
| 方法 | 描述 |
|------|------|
| **No early exit** | 全深度解码（baseline） |
| **Fixed-layer EE** | 所有 speech token 固定从某中间层退出 |
| **Confidence-based EE** | 基于 entropy 判断是否退出（常见于文本LLM） |
| **SPAR-K (Ours)** | 周期性交替退出（Even/Odd/Triple） |

---

## 3. 主要实验结果和性能指标

### ✅ 整体性能表现（见 Table 2）

#### ▶️ Step-Audio-2 结果（Triple(22) 最优）
| 指标 | No EE (S1) | SPAR-K (S6) | 变化 |
|------|-----------|------------|------|
| **Mean Accuracy** | 54.22% | **55.29%** | ↑ +1.07% |
| **MOS** | 3.710 | 3.668 | ↓ -1.12%（轻微下降） |
| **ASR-WER** | 1.51% | 1.51% | ✅ 无恶化 |
| **Speech Avg. Exit Layer** | 28 | **25** | ↓ **11% 层减少** |

> 💡 小结：**精度基本不变甚至略升，语音质量几乎无损，计算节省达11%**

#### ▶️ GLM-4-Voice 结果（Even(36) 最优）
| 指标 | No EE (G1) | SPAR-K (G4) | 变化 |
|------|-----------|------------|------|
| **Mean Accuracy** | 52.37% | 50.83% | ↓ -0.82%（最大降幅） |
| **MOS** | 2.982 | 2.950 | ↓ -1.07% |
| **ASR-WER** | 4.31% | 5.36% | ↑ 可接受范围 |
| **Speech Avg. Exit Layer** | 40 | **38** | ↓ **5% 层减少** |

> 💡 小结：**在仅损失 <1% 精度下实现 5% 计算压缩，且无任何辅助开销**

---

### 🔍 关键对比结果

#### ❌ Fixed-layer Early Exit 失败
- **Step-Audio-2 (S2)**：MOS 从 3.71 → 3.058（暴跌），WER 升至 3.40%
- **GLM-4-Voice (G2)**：MOS 降至 2.662，WER 飙升至 43.60%
- 原因：连续 shallow decoding 导致 **distribution shift 积累**，语音无法正常终止，产生冗余音段。

#### ⚠️ Confidence-based Early Exit 不稳定
- **Step-Audio-2 (S3)**：Accuracy 暴跌至 41.51%，MOS 仅 1.651
- **GLM-4-Voice (G3)**：虽可工作（MOS=2.866），但依赖精细调参（entropy threshold=0.5）
- 缺点：每次都要计算中间层 logits 来评估 entropy，若未触发 exit，则该计算浪费 → **存在 compute overhead**

#### ✅ SPAR-K 显著优于上述方法
- **无需动态决策**，无 wasted computation；
- **性能稳定**，Across datasets and models；
- **zero overhead**，仅需预训练 layer head，推理完全轻量。

---

### 🔬 消融实验分析（Ablation Studies）

#### （1）不同调度策略比较
| 模型 | Even | Odd | Triple | 观察 |
|------|------|-----|--------|------|
| Step-Audio-2 | ✅ 好 | ✅ 好 | ✅ 最佳 | 可容忍更大 K |
| GLM-4-Voice | ✅ 最佳 | ✅ 好 | ❌ MOS 下降更多 | 更长语音块（26 vs 4）导致误差积累更严重 |

> 发现：**K 越大，full-depth refresh 间隔越长，风险越高**；尤其在长语音序列中需谨慎选择 K。

#### （2）early exit layer $ l_{EE} $ 敏感性分析（G7-G8）
- 当 $ l_{EE} $ 降低（如从 36→33），ASR-WER 渐进上升而非突变；
- 表明：**存在一个“安全深度区间”**，可在其中调节 trade-off。

#### （3）尝试对 text token 应用 SPAR-K（G9）
- 结果：Accuracy 从 52.37% → **18.43%**，灾难性失败！
- 再次验证：**text token 必须精细控制 decoding depth**，不能简单周期性退出。

#### （4）混合策略：text用confidence-based + speech用SPAR-K（G10）
- 性能下降缩小到 7.18%，语音质量保持；
- 表明：**text 和 speech 应采用不同 early exit policy**，未来可设计 hybrid framework。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **语音token具有感知鲁棒性**：即使中间层预测与最终层不同，合成语音仍听起来相似 → 支持 shallow decoding。
2. **文本token不具备此性质**：中间层文本难以构成通顺句子 → 必须 full-depth 或 adaptive control。
3. **传统 confidence-based early exit 不适合 SLM**：
   - 在 Step-Audio-2 上完全失效；
   - 存在 compute 浪费问题；
   - 对超参敏感。
4. **周期性“刷新”机制至关重要**：定期插入 full-depth step 可有效防止 error accumulation。
5. **SPAR-K 是首个成功的 SLM early exit 框架**：
   - 在两个主流 SLM 上均取得显著加速（5%-11% layer reduction）；
   - 几乎无损语义与感知质量；
   - 无额外计算负担。

---

### ⚠️ 方法的局限性
- **仅适用于 interleaved SLM 架构**，不直接推广到其他 SLM 范式（如端到端 direct generation）；
- **K 和 $ l_{EE} $ 需要在验证集上调优**，缺乏完全自适应能力；
- **目前只应用于 speech side**，text side 的高效退出仍待研究；
- **依赖 layer-specific LM head 的训练质量**，但训练数据有限可能影响泛化。

---

### 🔮 未来工作方向
1. **开发统一的 hybrid early exit framework**：
   - text tokens 使用 confidence-based 或 learned policy；
   - speech tokens 使用 SPAR-K 或更复杂的 pattern。
2. **探索自适应 K 调度机制**：根据上下文动态调整 refresh 频率。
3. **扩展至多语言、多方言 SLMs**：验证跨语言鲁棒性。
4. **结合 speculative decoding / draft models**：进一步提升推理速度。
5. **硬件层面优化集成**：将 SPAR-K 部署至边缘设备，实现实时低功耗语音交互。

---

> 📌 **一句话总结**：  
> SPAR-K 首次提出面向 spoken language models 的模态感知 early exit 框架，通过 **周期性交替浅层退出与全层刷新**，在**零额外开销**下实现最高 **11% 的 decoding depth 降低**，同时几乎**不牺牲语义准确性和语音感知质量**，为实时语音AI系统提供了高效的推理解决方案。

</details>

---

### 16. [GAST: Gradient-aligned Sparse Tuning of Large Language Models with Data-layer Selection](https://arxiv.org/abs/2603.09865)

**Authors**: Kai Yao, Zhenghan Song, Kaixin Wu, Mingjie Zhong, Danzhao Cheng, Zhaorui Tan, Yixin Ji, Penglei Gao  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.09865v1  

#### Abstract
Parameter-Efficient Fine-Tuning (PEFT) has become a key strategy for adapting large language models, with recent advances in sparse tuning reducing overhead by selectively updating key parameters or subsets of data. Existing approaches generally focus on two distinct paradigms: layer-selective metho...

---

### 17. [AutoAgent: Evolving Cognition and Elastic Memory Orchestration for Adaptive Agents](https://arxiv.org/abs/2603.09716)

**Authors**: Xiaoxing Wang, Ning Liao, Shikun Wei, Chen Tang, Feiyu Xiong  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.09716v1  

#### Abstract
Autonomous agent frameworks still struggle to reconcile long-term experiential learning with real-time, context-sensitive decision-making. In practice, this gap appears as static cognition, rigid workflow dependence, and inefficient context usage, which jointly limit adaptability in open-ended and n...

---

### 18. [DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval](https://arxiv.org/abs/2603.09185)

**Authors**: Taegyeong Lee, Jiwon Park, Seunghyun Hwang, JooYoung Jang  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.09185v1  

#### Abstract
Recent advances in Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) have enabled diverse retrieval methods. However, existing retrieval methods often fail to accurately retrieve results for negation and exclusion queries. To address this limitation, prior approaches rely on embe...

---

### 19. [DeZent: Decentralized z-Anonymity with Privacy-Preserving Coordination](https://arxiv.org/abs/2603.08854)

**Authors**: Carolin Brunn, Florian Tschorsch  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.08854v1  

#### Abstract
Analyzing large volumes of sensor network data, such as electricity consumption measurements from smart meters, is essential for modern applications but raises significant privacy concerns. Privacy-enhancing technologies like z-anonymity offer efficient anonymization for continuous data streams by s...

---

### 20. [MAPLE: Elevating Medical Reasoning from Statistical Consensus to Process-Led Alignment](https://arxiv.org/abs/2603.08987)

**Authors**: Kailong Fan, Anqi Pu, Yichen Wu, Wanhua Li, Yicong Li, Hanspeter Pfister, Huafeng Liu, Xiang Li, Quanzheng Li, Ning Guo  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.08987v1  

#### Abstract
Recent advances in medical large language models have explored Test-Time Reinforcement Learning (TTRL) to enhance reasoning. However, standard TTRL often relies on majority voting (MV) as a heuristic supervision signal, which can be unreliable in complex medical scenarios where the most frequent rea...

---

### 21. [SCALAR: Learning and Composing Skills through LLM Guided Symbolic Planning and Deep RL Grounding](https://arxiv.org/abs/2603.09036)

**Authors**: Renos Zabounidis, Yue Wu, Simon Stepputtis, Woojun Kim, Yuanzhi Li, Tom Mitchell, Katia Sycara  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.09036v1  

#### Abstract
LM-based agents excel when given high-level action APIs but struggle to ground language into low-level control. Prior work has LLMs generate skills or reward functions for RL, but these one-shot approaches lack feedback to correct specification errors. We introduce SCALAR, a bidirectional framework ...

---

### 22. [Learning Adaptive LLM Decoding](https://arxiv.org/abs/2603.09065)

**Authors**: Chloe H. Su, Zhe Ye, Samuel Tenka, Aidan Yang, Soonho Kong, Udaya Ghai  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.09065v1  

#### Abstract
Decoding from large language models (LLMs) typically relies on fixed sampling hyperparameters (e.g., temperature, top-p), despite substantial variation in task difficulty and uncertainty across prompts and individual decoding steps. We propose to learn adaptive decoding policies that dynamically sel...

---

### 23. [FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](https://arxiv.org/abs/2603.09661)

**Authors**: Boya Zhang, Shuaijie Yin, Huiwen Zhu, Xing He  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.09661v1  

#### Abstract
Mining time-frequency features is critical for time series forecasting. Existing research has predominantly focused on modeling low-frequency patterns, where most time series energy is concentrated. The overlooking of mid to high frequency continues to limit further performance gains in deep learnin...

---

### 24. [Physics-informed neural operator for predictive parametric phase-field modelling](https://arxiv.org/abs/2603.09693)

**Authors**: Nanxi Chen, Airong Chen, Rujin Ma  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.09693v1  

#### Abstract
Predicting the microstructural and morphological evolution of materials through phase-field modelling is computationally intensive, particularly for high-throughput parametric studies. While neural operators such as the Fourier neural operator (FNO) show promise in accelerating the solution of param...

---

### 25. [OOD-MMSafe: Advancing MLLM Safety from Harmful Intent to Hidden Consequences](https://arxiv.org/abs/2603.09706)

**Authors**: Ming Wen, Kun Yang, Jingyu Zhang, Yuxuan Liu, shiwen cui, Shouling Ji, Xingjun Ma  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.09706v1  

#### Abstract
While safety alignment for Multimodal Large Language Models (MLLMs) has gained significant attention, current paradigms primarily target malicious intent or situational violations. We propose shifting the safety frontier toward consequence-driven safety, a paradigm essential for the robust deploymen...

---

### 26. [One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](https://arxiv.org/abs/2603.08869)

**Authors**: Sripad Karne  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.08869v1  

#### Abstract
Do the features learned by Sparse Autoencoders (SAEs) represent abstract meaning, or are they tied to how text is written? We investigate this question using Serbian digraphia as a controlled testbed: Serbian is written interchangeably in Latin and Cyrillic scripts with a near-perfect character mapp...

---

### 27. [Serving Compound Inference Systems on Datacenter GPUs](https://arxiv.org/abs/2603.08797)

**Authors**: Sriram Devata, Rahul Singh, Sarita Adve  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.08797v1  

#### Abstract
Applications in emerging domains such as XR are being built as compound inference systems, where multiple ML models are composed in the form of a task graph to service each request. Serving these compound systems efficiently raises two questions: how to apportion end-to-end latency and accuracy budg...

---

### 28. [Nezha: A Key-Value Separated Distributed Store with Optimized Raft Integration](https://arxiv.org/abs/2603.09122)

**Authors**: Yangyang Wang, Yucong Dong, Ziqian Cheng, Zichen Xu  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.09122v1  

#### Abstract
Distributed key-value stores are widely adopted to support elastic big data applications, leveraging purpose-built consensus algorithms like Raft to ensure data consistency. However, through systematic analysis, we reveal a critical performance issue in such consistent stores, i.e., overlapping pers...

---

### 29. [Hierarchical Observe-Orient-Decide-Act Enabled UAV Swarms in Uncertain Environments: Frameworks, Potentials, and Challenges](https://arxiv.org/abs/2603.09191)

**Authors**: Ziye Jia, Yao Wu, Qihui Wu, Lijun He, Qiuming Zhu, Fuhui Zhou, Zhu Han  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.09191v1  

#### Abstract
Unmanned aerial vehicle (UAV) swarms are increasingly explored for their potentials in various applications such as surveillance, disaster response, and military. However, UAV swarms face significant challenges of implementing effective and rapid decisions under dynamic and uncertain environments. T...

---

### 30. [Reconstructing Movement from Sparse Samples: Enhanced Spatio-Temporal Matching Strategies for Low-Frequency Data](https://arxiv.org/abs/2603.09412)

**Authors**: Ali Yousefian, Arianna Burzacchi, Simone Vantini  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.09412v1  

#### Abstract
This paper explores potential improvements to the Spatial-Temporal Matching algorithm for matching the GPS trajectories to road networks. While this algorithm is effective, it presents some limitations in computational efficiency and the accuracy of the results, especially in dense environments with...

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
